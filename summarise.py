"""Summarise meeting transcripts with a local LLM (gpt-oss-20b) through an OpenAI-compatible API.

Reads llm/inbox/<recording>.md, writes llm/summaries/<recording>.md and .json (summary, decisions,
action items, open questions), then moves the transcript to llm/processed/.

Works with any server exposing /v1/chat/completions: Ollama (default, `ollama pull gpt-oss:20b`),
LM Studio, llama.cpp's llama-server, ... Settings are the "llm" section of settings.json (see settings.py;
the Settings tab edits them), e.g.
  {"llm": {"base_url": "http://127.0.0.1:1234/v1", "model": "openai/gpt-oss-20b"}}

Usage:
  uv run summarise.py --check     # is the LLM server reachable and the model available?
  uv run summarise.py             # summarise everything in llm/inbox/
  uv run summarise.py NAME ...    # summarise specific recordings
  uv run summarise.py --models    # list the chat models the server has
  uv run summarise.py --compare qwen3.5:9b gpt-oss:20b NAME ...   # side-by-side test, see compare_one()
  uv run summarise.py --label [--workspace ID] NAME ...  # suggest labels only (meetings already summarised)
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import labels
import library
from mcp_server import guardrails
import settings

ROOT = Path(__file__).parent
PROMPT_FILE = ROOT / "prompts" / "meeting_summary.md"
SUMMARY_CTX = 32768  # context a model needs for long meetings (Ollama's default of 4096 is ~15 min)


def config():
    return settings.get("llm")


def _request(cfg, path, payload=None, timeout=10):
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def save_model(model):
    """Make `model` the default for every summary (manual and auto), stored in settings.json."""
    if model not in models():
        raise library.LibraryError(f"model {model!r} is not available")
    settings.update({"llm": {"model": model}})


def _ollama_url(cfg):
    """Ollama's native API (for model details, memory use, unloading); None for other servers."""
    base = cfg["base_url"].rstrip("/")
    return base[:-3] if base.endswith("/v1") else None


def _ollama(cfg, path, payload=None, timeout=10):
    return _request({**cfg, "base_url": _ollama_url(cfg)}, path, payload, timeout)


def models():
    """Names of the chat models the server has. With Ollama, embedding-only models (which can't write a
    summary, e.g. qwen3-embedding:4b) are left out; other servers list everything they report."""
    cfg = config()
    try:
        names = [m["id"] for m in _request(cfg, "/models").get("data", [])]
    except (urllib.error.URLError, OSError, ValueError):
        return []
    if not _ollama_url(cfg):
        return sorted(names)
    chat = []
    for n in names:
        try:
            caps = _ollama(cfg, "/api/show", {"model": n}).get("capabilities")
        except (urllib.error.URLError, OSError, ValueError):
            caps = None
        if caps is None or "completion" in caps:
            chat.append(n)
    return sorted(chat)


def model_details():
    """[{name, ctx, max_ctx, copy_32k}] for the Settings tab. ctx: the context the server will give the
    model (Ollama's default 4096 unless the model sets num_ctx); max_ctx: what the model supports;
    copy_32k: name of an existing 32k copy of it, if any. Only Ollama reports these; elsewhere None."""
    cfg = config()
    names = models()
    out = []
    for n in names:
        info = {"name": n, "ctx": None, "max_ctx": None, "copy_32k": None}
        if _ollama_url(cfg):
            try:
                d = _ollama(cfg, "/api/show", {"model": n})
                m = re.search(r"^num_ctx\s+(\d+)", d.get("parameters") or "", re.M)
                info["ctx"] = int(m.group(1)) if m else 4096
                info["max_ctx"] = next((v for k, v in (d.get("model_info") or {}).items()
                                        if k.endswith(".context_length") and "original" not in k), None)
            except (urllib.error.URLError, OSError, ValueError):
                pass
            copy = copy_32k_name(n)
            info["copy_32k"] = next((c for c in names if c in (copy, copy + ":latest")), None)
        out.append(info)
    return out


def copy_32k_name(model):
    """qwen3.5:4b -> qwen3.5-4b-32k (the naming the README uses for gpt-oss-20b-32k)."""
    base = model.removesuffix(":latest").replace(":", "-").replace("/", "-")
    return base if base.endswith("-32k") else base + "-32k"


def create_32k(model):
    """Make a 32k-context copy of an Ollama model: same weights (no download), num_ctx 32768.
    Returns the new model's name."""
    cfg = config()
    if not _ollama_url(cfg):
        raise library.LibraryError("creating a 32k copy needs Ollama")
    if model not in models():
        raise library.LibraryError(f"model {model!r} is not available")
    name = copy_32k_name(model)
    _ollama(cfg, "/api/create", {"model": name, "from": model, "parameters": {"num_ctx": SUMMARY_CTX},
                                 "stream": False}, timeout=600)
    return name + ":latest"


def unload(model):
    """Free a model's memory now instead of after Ollama's 5-minute keep-alive."""
    cfg = config()
    if _ollama_url(cfg):
        try:
            _ollama(cfg, "/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
        except (urllib.error.URLError, OSError, ValueError):
            pass


def memory(cfg, model):
    """How much memory Ollama holds for a loaded model: {'total_gb', 'gpu_gb', 'context'}, or None."""
    if not _ollama_url(cfg):
        return None
    try:
        loaded = _ollama(cfg, "/api/ps").get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return None
    for m in loaded:
        if model in (m.get("name"), m.get("model")):
            return {"total_gb": round(m.get("size", 0) / 1e9, 2), "gpu_gb": round(m.get("size_vram", 0) / 1e9, 2),
                    "context": m.get("context_length")}
    return None


def unload_all():
    """Ask Ollama to free every loaded model, so a comparison measures one model at a time (two 9-20B
    models don't fit side by side in 16 GB) and each starts from a cold load, like a real run."""
    cfg = config()
    if not _ollama_url(cfg):
        return
    try:
        for m in _ollama(cfg, "/api/ps").get("models", []):
            _ollama(cfg, "/api/generate", {"model": m["name"], "keep_alive": 0}, timeout=60)
    except (urllib.error.URLError, OSError, ValueError):
        pass


def check():
    """{'ok': bool, 'message': str} for the UI and --check."""
    cfg = config()
    try:
        models = [m["id"] for m in _request(cfg, "/models").get("data", [])]
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {"ok": False, "model": cfg["model"], "base_url": cfg["base_url"],
                "message": f"LLM server not reachable at {cfg['base_url']} ({getattr(e, 'reason', e)})"}
    if cfg["model"] not in models:
        return {"ok": False, "model": cfg["model"], "base_url": cfg["base_url"],
                "message": f"server is up but model {cfg['model']!r} is not available (has: {', '.join(models) or 'none'})"}
    return {"ok": True, "model": cfg["model"], "base_url": cfg["base_url"], "message": "ready"}


def _parse_json(text):
    """The model is asked for bare JSON; tolerate code fences or text around it."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("model reply contained no JSON object")
    return json.loads(m.group(0))


UNDEFINED = "undefined"


def _known(v):
    """A people field the model filled from the transcript ("", None and "undefined" are not)."""
    v = (v or "").strip() if isinstance(v, str) else ""
    return "" if v.lower() in ("undefined", "unknown", "unclear", "n/a", "none", "null") else v


def _person(p):
    """One "people" entry as a line: "Name (role · organisation, reports to X): notes". A role the meeting
    didn't establish shows as "role undefined"; organisation and manager are only shown when stated.
    Older or odd replies may give plain strings."""
    if not isinstance(p, dict):
        return str(p)
    role, org, boss = _known(p.get("role")), _known(p.get("organisation")), _known(p.get("reports_to"))
    notes = (p.get("notes") or "").strip()
    head = " · ".join(x for x in (role, org) if x) + (", " if (role or org) and boss else "") + \
        (f"reports to {boss}" if boss else "")
    return f"**{p.get('name') or '?'}**" + (f" ({head})" if head else "") + (f": {notes}" if notes else "")


def check_hierarchy(notes):
    """Reporting lines must form a tree. Where reports_to loops (A > B > A, or longer), the model mixed
    up the direction: clear those reports_to to "undefined" and say so under open questions, rather than
    publish a hierarchy that contradicts itself. Returns the names whose line was cleared."""
    people = [p for p in notes.get("people") or [] if isinstance(p, dict)]
    key = lambda n: " ".join((n or "").lower().split())
    boss = {key(p.get("name")): key(_known(p.get("reports_to"))) for p in people if _known(p.get("reports_to"))}
    looped = set()
    for start in boss:
        seen, n = [], start
        while n in boss and n not in seen:
            seen.append(n)
            n = boss[n]
        if n in seen:  # walked back into the chain: everything from n onward is a loop
            looped |= set(seen[seen.index(n):])
    cleared = []
    for p in people:
        if key(p.get("name")) in looped:
            cleared.append(p.get("name"))
            p["reports_to"] = UNDEFINED
    if cleared:
        notes.setdefault("open_questions", []).append(
            "Reporting line unclear: " + ", ".join(cleared) + " (the notes had them reporting to each other).")
    return cleared


def to_markdown(name, notes, model):
    def section(title, items):
        return f"## {title}\n\n" + ("\n".join(f"- {i}" for i in items) if items else "_None._") + "\n"
    actions = [f"**{a.get('owner') or 'unassigned'}**: {a.get('task')}" + (f" (due {a['due']})" if a.get("due") else "")
               for a in notes.get("action_items", [])]
    return (f"# {notes.get('title') or name}\n\n_Recording: {name} · summarised by {model}_\n\n"
            + section("Summary", notes.get("summary", [])) + "\n"
            + (section("People", [_person(x) for x in notes["people"]]) + "\n" if notes.get("people") else "")
            + section("Decisions", notes.get("decisions", [])) + "\n"
            + section("Action items", actions) + "\n"
            + section("Open questions", notes.get("open_questions", [])))


def summary_prompt(ws):
    """The summary instructions plus the label instructions for workspace `ws` (one LLM call does both)."""
    return (PROMPT_FILE.read_text().rstrip() + "\n\n" + labels.prompt_block(ws).rstrip()
            + '\n\nPut "labels" in the same JSON object, as one more key next to "title", "summary" and the rest.\n')


def labels_prompt(ws):
    return ("You label meeting transcripts. They were produced automatically: Hindi/English speech translated to "
            "English, speakers labelled SPEAKER_00, SPEAKER_01, ... by voice.\n\n" + labels.prompt_block(ws).rstrip()
            + '\n\nReply with a single JSON object and nothing else: {"labels": {...}}\n')


def _summarise(cfg, model, name, transcript, prompt, timeout=None):
    """Run the LLM on one transcript with system prompt `prompt`; returns the parsed JSON with timing
    and memory in _meta."""
    if len(transcript) / 4 > cfg["max_input_tokens"]:  # ~4 characters per token
        raise library.LibraryError(f"transcript is ~{len(transcript) // 4} tokens, over max_input_tokens")
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": transcript}],
        "temperature": 0.2,
        #"response_format": {"type": "json_object"},
    }
    if _ollama_url(cfg):  # a long summary makes several calls; don't let Ollama unload in between
        payload["keep_alive"] = cfg.get("keep_alive") or "30m"
    if cfg.get("reasoning_effort"):  # how long a thinking model thinks; others are sent nothing
        payload["reasoning_effort"] = cfg["reasoning_effort"]
    try:
        reply = _request(cfg, "/chat/completions", payload, timeout=timeout or cfg["timeout_s"])
    except urllib.error.HTTPError as e:
        if e.code != 400 or "reasoning_effort" not in payload:
            raise
        payload.pop("reasoning_effort")  # a model without thinking can refuse the option: ask again without it
        reply = _request(cfg, "/chat/completions", payload, timeout=timeout or cfg["timeout_s"])
    seconds = round(time.perf_counter() - t0, 1)
    mem = memory(cfg, model)  # read while the model is still loaded
    # Servers like Ollama silently cut the input to their context window (default 4096 tokens), and the
    # reply then runs out of room. Report that clearly instead of failing on half-written JSON.
    sent = (len(prompt) + len(transcript)) / 4  # rough estimate, ~4 characters per token
    usage = reply.get("usage") or {}
    got = usage.get("prompt_tokens")
    hint = "raise the model's context length (for Ollama: a model with a larger num_ctx, see README)"
    if got and got < 0.6 * sent:
        raise library.LibraryError(f"the LLM server read only {got} of ~{int(sent)} input tokens "
                                   f"(input truncated); {hint}")
    if reply["choices"][0].get("finish_reason") == "length":
        raise library.LibraryError(f"the reply was cut off at the context/output limit; {hint}")
    notes = _parse_json(reply["choices"][0]["message"]["content"])
    out = usage.get("completion_tokens")
    notes["_meta"] = {"model": model, "seconds": seconds, "usage": reply.get("usage"), "memory": mem,
                      "tokens_per_s": round(out / seconds, 1) if out and seconds else None}
    return notes


PART_NOTE = """

This meeting is long, so its transcript comes in parts and you are writing the notes for ONE part. The
parts' notes are joined afterwards in order, so cover only what is said in this part, with the same
rules and the same JSON shape; don't summarise earlier parts again. "People established in earlier
parts", when given, is context for knowing who someone is: list a person again only if this part adds
something about them. When this part states or confirms someone's role, organisation or manager
(including someone already listed, whose field may still be "undefined"), fill that field with the time
of the line in this part, even if an earlier part already had it. The title should describe the meeting
as far as this part shows.
"""
_LINE_T = re.compile(r"^\[(\d+):(\d\d):(\d\d)\]", re.M)


def _secs(t):
    """"hh:mm:ss" or "mm:ss" -> seconds, else None."""
    m = re.fullmatch(r"\s*\[?(?:(\d+):)?(\d{1,2}):(\d\d)\]?\s*", str(t or ""))
    return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None


def split_transcript(transcript, minutes):
    """The handoff markdown (header, then "[hh:mm:ss] SPEAKER: text" lines) as parts of about `minutes`
    each: [(header, part_text, start_s, end_s)]. One part when the meeting is short enough; a last part
    of under a third of `minutes` joins the one before it."""
    head, sep, body = transcript.partition("# Transcript\n")
    if not sep or not minutes:
        return [(transcript, "", 0, 0)]
    lines = [l for l in body.strip("\n").split("\n") if l.strip()]
    times = [(_secs(m.group(0)) if (m := _LINE_T.match(l)) else None) for l in lines]
    span = minutes * 60
    groups, start = [[]], None
    for line, t in zip(lines, times):
        if t is not None:
            start = t if start is None else start
            if t - start >= span and groups[-1]:
                groups.append([])
                start = t
        groups[-1].append((line, t))
    if len(groups) > 1:
        last = [t for _, t in groups[-1] if t is not None]
        if not last or last[-1] - last[0] < span / 3:
            groups[-2] += groups.pop()
    out = []
    for g in groups:
        ts = [t for _, t in g if t is not None]
        out.append((head + sep, "\n".join(l for l, _ in g), ts[0] if ts else 0, ts[-1] if ts else 0))
    return out


def _norm(x):
    return " ".join(re.sub(r"[^\w\s]", " ", str(x).lower()).split())


def merge_parts(parts):
    """Join per-part notes in order, in code (no extra LLM call): lists concatenated without repeats,
    people merged by name (a field stated in any part wins over "undefined"; two different managers for
    one person become "undefined" plus an open question), labels combined."""
    out = {"title": next((p.get("title") for p in parts if p.get("title")), ""), "summary": [], "people": [],
           "decisions": [], "action_items": [], "open_questions": []}
    seen = {k: set() for k in out if k not in ("title", "people")}
    people, conflicts = {}, []
    for part in parts:
        for key in seen:
            for x in part.get(key) or []:
                k = _norm(json.dumps(x, sort_keys=True) if isinstance(x, dict) else x)
                if k and k not in seen[key]:
                    seen[key].add(k)
                    out[key].append(x)
        for p in part.get("people") or []:
            if not isinstance(p, dict) or not p.get("name"):
                continue
            k = _norm(p["name"])
            if k not in people:
                people[k] = dict(p)
                continue
            have = people[k]
            for f in ("role", "organisation", "reports_to"):
                new, old = _known(p.get(f)), _known(have.get(f))
                if new and not old:
                    have[f], have[f + "_at"] = new, p.get(f + "_at")
                elif new and old and _norm(new) != _norm(old):
                    if f == "reports_to":
                        conflicts.append(f"{have['name']}: reports to {old} or {new}?")
                        have[f], have[f + "_at"] = UNDEFINED, None
                    elif f == "role" and _norm(new) not in _norm(old):  # roles can change in a meeting
                        have[f] = f"{old}; {new}"
            notes = [n for n in (have.get("notes"), p.get("notes")) if n and str(n).strip()]
            have["notes"] = "; ".join(dict.fromkeys(str(n).strip() for n in notes))
    out["people"] = list(people.values())
    out["open_questions"] += [f"Reporting line unclear: {c}" for c in conflicts]
    labs = [p["labels"] for p in parts if isinstance(p.get("labels"), dict)]
    if labs:
        types = [str(l.get("type") or "") for l in labs if l.get("type")]
        merged = {"type": max(set(types), key=types.count) if types else ""}
        for f in ("clients", "projects", "people", "topics"):
            vals = [v for l in labs for v in (l.get(f) or []) if isinstance(v, str)]
            merged[f] = list(dict.fromkeys(vals))[:8 if f == "topics" else None]
        out["labels"] = merged
    return out


_GENERIC = set("the and for with team lead head owner senior member of new main key".split())


def _close(a, b):
    """Two spellings of one name as speech recognition gives them ("Tamsen"/"Tamsin", "Marlo"/"Marlow")."""
    return a == b or (len(a) > 3 and len(b) > 3 and a[0] == b[0] and SequenceMatcher(None, a, b).ratio() >= 0.8)


def check_evidence(notes, transcript, ws=None):
    """Clear a person's role, organisation or manager unless the model cited the transcript line that
    states it: the cited time must be a real line, and the person (any registry spelling) must be named
    within the 90 s before it or 30 s after (so "So he is the product manager" counts after "Tamsin...").
    Returns ["name: field", ...] for what was cleared."""
    lines = [(_secs(m.group(0)), l) for l in transcript.split("\n") if (m := _LINE_T.match(l))]
    spellings = {}
    if ws and labels.exists(ws):
        for e in labels.load(ws)["entities"]:
            spellings[_norm(e["name"])] = [e["name"], *e.get("aliases", [])]
    cleared = []
    for p in notes.get("people") or []:
        if not isinstance(p, dict):
            continue
        names = spellings.get(_norm(p.get("name")), [p.get("name") or ""])
        words = {w for n in names for w in _norm(n).split() if len(w) > 2}
        for f in ("role", "organisation", "reports_to"):
            if not _known(p.get(f)):
                continue
            t = _secs(p.get(f + "_at"))
            near = [l for s, l in lines if t is not None and t - 90 <= s <= t + 30]
            said = {w for l in near for w in _norm(l).split()}
            # the value must be said there too: a content word of it, matched on its first 5 letters
            # ("architects" backs "Architect"); for reports_to that is the manager's name
            forms = spellings.get(_norm(p[f]), [p[f]]) if f == "reports_to" else [p[f]]
            value = {w[:5] for v in forms for w in _norm(v).split() if len(w) > 2 and w not in _GENERIC}
            if f == "role" and not value:  # only generic words ("Team Member", "Lead"): says nothing
                p[f], p[f + "_at"] = UNDEFINED, None
                cleared.append(f"{p.get('name')}: {f}")
                continue
            if f == "reports_to":  # a name: close spellings count, as for the person
                backed = any(_close(w, x) for v in forms for w in _norm(v).split() if len(w) > 2 for x in said)
            else:
                backed = not value or bool(value & {w[:5] for w in said})
            # a company is usually set for the whole conversation, so for organisation the company named at
            # the cited line is enough; a role or a manager must also be said near the person's name
            ok = t is not None and any(abs(s - t) <= 15 for s, _ in lines) and backed and \
                (f == "organisation" or any(_close(w, x) for w in words for x in said))
            if not ok:
                p[f], p[f + "_at"] = UNDEFINED, None
                cleared.append(f"{p.get('name')}: {f}")
    return cleared


def _people_context(notes):
    def row(p):
        cited = ", ".join(f"{f} {_known(p.get(f))} [{p.get(f + '_at')}]" for f in ("role", "organisation", "reports_to")
                          if _known(p.get(f)))
        return f"{p.get('name')}: {cited or 'role, organisation and manager not yet stated'}" + \
            (f". {p['notes']}" if p.get("notes") else "")
    rows = [row(p) for p in notes.get("people") or [] if isinstance(p, dict)]
    return "People established in earlier parts:\n" + "\n".join(f"- {r}" for r in rows) + "\n\n" if rows else ""


TIDY_FILE = ROOT / "prompts" / "meeting_tidy.md"
def drop_sensitive(notes):
    """Remove sentences the guardrails match (deal figures and terms, remarks about race, origin or private
    life, swearing and insults) from every text in the notes, whatever the model wrote. The MCP server
    runs the same filter on everything it sends out. Returns how many sentences were removed."""
    count = [0]

    def clean(text):
        parts = re.split(r"(?<=[.;!?])\s+", text)
        keep = [x for x in parts if not guardrails.why(x)]
        count[0] += len(parts) - len(keep)
        return " ".join(keep).strip()

    def walk(x, key=""):
        if isinstance(x, str):
            return x if key.endswith("_at") or key in ("name", "_meta") else clean(x)
        if isinstance(x, list):
            return [v for v in (walk(v, key) for v in x) if v not in ("", None, {})]
        if isinstance(x, dict):
            return {k: (v if k == "_meta" else walk(v, k)) for k, v in x.items()}
        return x
    for key in ("title", "summary", "people", "decisions", "action_items", "open_questions"):
        if key in notes:
            notes[key] = walk(notes[key], key)
    notes["action_items"] = [a for a in notes.get("action_items") or [] if not isinstance(a, dict) or a.get("task")]
    return count[0]


def tidy(cfg, model, name, notes):
    """One short LLM call over the merged notes (not the transcript): merge repeats, join name variants.
    Returns (tidied notes, _meta of the call); on any failure the merged notes are kept as they are."""
    body = {k: notes[k] for k in ("title", "summary", "people", "decisions", "action_items", "open_questions")
            if k in notes}
    try:
        out = _summarise(cfg, model, name, json.dumps(body, ensure_ascii=False, indent=1), TIDY_FILE.read_text())
    except Exception as e:
        print(f"{name}: tidy step failed, keeping the merged notes: {type(e).__name__}: {e}")
        return notes, None
    meta = out.pop("_meta", None)
    if not isinstance(out.get("summary"), list) or not out["summary"]:
        return notes, meta
    tidied = {**notes, **{k: out[k] for k in body if k in out}}
    _keep_verified(tidied, notes.get("people") or [])
    return tidied, meta


def _keep_verified(notes, verified):
    """The tidy step may only reword: role, organisation and manager (with their times) come from the
    checked entries, matched by name or, for a joined entry ("Orrin" + "Orrin Bexley"), by any name word."""
    by = {}
    for v in verified:
        if isinstance(v, dict) and v.get("name"):
            by.setdefault(_norm(v["name"]), []).append(v)
    for p in notes.get("people") or []:
        if not isinstance(p, dict):
            continue
        key = _norm(p.get("name"))
        src = by.get(key) or [v for k, vs in by.items() for v in vs if set(k.split()) & set(key.split())]
        for f in ("role", "organisation", "reports_to"):
            known = next((v for v in src if _known(v.get(f))), None)
            p[f], p[f + "_at"] = (known[f], known.get(f + "_at")) if known else (UNDEFINED, None)


_FILLER = re.compile(r"(?i)^(?:(?:he|she|they|was|is|were|only|just|briefly|also)\s+)*(?:mentioned|referred to|named|"
                     r"brought up|discussed)(?:\s+(?:in|during|by)\s+(?:the\s+)?(?:transcript|meeting|passing|call|"
                     r"discussion|name))?\s*\.?$")


# a note that only records that someone came up ("Mentioned as the person X sat with", "Addressed by Y to ...")
_PASSING = re.compile(r"(?i)^(?:(?:he|she|they|was|is|were|only|just|briefly|also)\s+)*(?:mentioned|addressed|"
                      r"referred to|named|greeted|thanked|called out|asked (?:to|not to))\b")


def _similar(a, b):
    """Two points saying the same: one contains the other (4+ words), 70% of their words are shared, or
    they are 85% alike as text."""
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    short, long_ = sorted((a, b), key=len)
    if len(short.split()) >= 4 and short in long_:
        return True
    wa, wb = set(a.split()), set(b.split())
    if min(len(wa), len(wb)) >= 4 and len(wa & wb) / len(wa | wb) >= 0.7:  # same words, a few added
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.85


def _dedupe(items, text=str):
    """`items` without near-repeats; of two alike, the longer (more detailed) one is kept in place."""
    out = []
    for x in items:
        i = next((j for j, y in enumerate(out) if _similar(text(x), text(y))), None)
        if i is None:
            out.append(x)
        elif len(text(x)) > len(text(out[i])):
            out[i] = x
    return out


def dedupe_notes(notes):
    """Remove repeated points: the parts of a long meeting each restate what was said before, and a
    person's notes are joined from every part. Returns the number of points removed."""
    n = 0
    for key in ("summary", "decisions", "open_questions"):
        items = [x for x in notes.get(key) or [] if str(x).strip()]
        kept = _dedupe(items)
        n += len(items) - len(kept)
        notes[key] = kept
    acts = [a for a in notes.get("action_items") or [] if isinstance(a, dict)]
    kept = []
    for a in acts:  # the same task for the same owner (or one side unassigned)
        o = _norm(a.get("owner"))
        i = next((j for j, b in enumerate(kept) if _similar(a.get("task"), b.get("task"))
                  and (o == _norm(b.get("owner")) or "unassigned" in (o, _norm(b.get("owner"))))), None)
        if i is None:
            kept.append(a)
        elif _norm(kept[i].get("owner")) == "unassigned" or len(str(a.get("task"))) > len(str(kept[i].get("task"))):
            kept[i] = {**kept[i], **{k: v for k, v in a.items() if v and _norm(v) != "unassigned"}}
    n += len(acts) - len(kept)
    notes["action_items"] = kept
    for p in notes.get("people") or []:
        if isinstance(p, dict) and p.get("notes"):
            parts = [x.strip() for x in re.split(r";\s+|(?<=[.!?])\s+", str(p["notes"])) if x.strip(" .;")]
            kept = _dedupe(parts)
            n += len(parts) - len(kept)
            p["notes"] = " ".join(x if x.endswith((".", "!", "?")) else x + "." for x in kept)
    return n


def check_people(notes, transcript):
    """Keep only people the meeting actually talked about: their name (or a close spelling) is in the
    transcript, and something concrete is known about them (a stated role, organisation or manager, an
    action item, or notes beyond "mentioned"). Names the model took from the known-names list, or that
    only came up in passing, are dropped. Returns the dropped names."""
    words = set(_norm(transcript).split())
    owners = {_norm(a.get("owner")) for a in notes.get("action_items") or [] if isinstance(a, dict)}

    def heard(name):
        toks = [t for t in _norm(name).split() if len(t) >= 3]
        return any(t in words or any(_close(t, w) for w in words if w[:1] == t[:1]) for t in toks)

    lines = [_norm(l) for l in transcript.splitlines()]
    voices = {_norm(m.group(1)) for m in re.finditer(r"(?m)^\[[\d:]+\] ([^:\n]+):", transcript)}

    def times_said(name):
        toks = [t for t in _norm(name).split() if len(t) >= 3]
        return sum(1 for l in lines if any(t in l.split() or any(_close(t, w) for w in l.split() if w[:1] == t[:1])
                                           for t in toks))

    def concrete(p):
        if any(_known(p.get(f)) for f in ("role", "organisation", "reports_to")):
            return True
        if _norm(p.get("name")) in owners or any(o and o in _norm(p.get("name")).split() for o in owners):
            return True
        sents = [x for x in re.split(r";\s+|(?<=[.!?])\s+", str(p.get("notes") or "")) if x.strip(" .;")]
        real = [x for x in sents if not _FILLER.match(x.strip())]
        if real and _norm(p.get("name")) not in voices and times_said(p.get("name")) <= 1 \
                and all(_PASSING.match(x.strip()) for x in real):
            return False  # named once, in passing: "Mentioned as the person X sat with"
        return bool(real)

    kept, dropped = [], []
    for p in notes.get("people") or []:
        if isinstance(p, dict) and p.get("name") and heard(p["name"]) and concrete(p):
            kept.append(p)
        elif isinstance(p, dict):
            dropped.append(str(p.get("name") or ""))
    notes["people"] = kept
    return dropped


def check_owners(notes):
    """An action item whose owner is named in the task itself ("Marlow: understand the current state with
    Marlow") is about that person, not done by them: the owner becomes "unassigned". Returns the tasks changed."""
    changed = []
    for a in notes.get("action_items") or []:
        if not isinstance(a, dict):
            continue
        owner = str(a.get("owner") or "").strip()
        if not owner or owner.lower() == "unassigned" or re.fullmatch(r"(?i)speaker[ _]?\d+", owner):
            continue
        words = [w for w in _norm(owner).split() if len(w) > 2]
        task = _norm(a.get("task") or "").split()
        if any(_close(w, t) for w in words for t in task):
            a["owner"] = "unassigned"
            changed.append(a.get("task"))
    return changed


_SPEAKER = re.compile(r"(?i)\bspeaker[ _]?\d+\b('s)?")


def drop_speaker_labels(notes):
    """No SPEAKER_nn in the notes: a voice label means nothing to a reader and is easily the wrong person.
    An owner that is a label becomes "unassigned"; in text, "a participant". Name the voices (the
    meeting's Speakers box) to get names instead. Returns the number of labels replaced."""
    count = [0]

    def fix(text):
        def one(m):
            count[0] += 1
            start = m.start() == 0 or text[:m.start()].rstrip().endswith((".", "!", "?", ":"))
            return ("A" if start else "a") + " participant" + (m.group(1) or "")
        return _SPEAKER.sub(one, text)

    for a in notes.get("action_items") or []:
        if isinstance(a, dict) and _SPEAKER.fullmatch(str(a.get("owner") or "").strip()):
            a["owner"] = "unassigned"
            count[0] += 1
    for key in ("title", "summary", "decisions", "open_questions"):
        if isinstance(notes.get(key), str):
            notes[key] = fix(notes[key])
        elif isinstance(notes.get(key), list):
            notes[key] = [fix(x) if isinstance(x, str) else x for x in notes[key]]
    for item in (notes.get("people") or []) + (notes.get("action_items") or []):
        if isinstance(item, dict):
            for k in ("notes", "task"):
                if isinstance(item.get(k), str):
                    item[k] = fix(item[k])
    return count[0]


def make_notes(cfg, model, name, transcript, ws, log=print, pause=None):
    """The notes for one transcript: a single LLM call for a short meeting, or one call per part of
    llm.chunk_minutes, merged in code. Spelling, evidence and hierarchy checks run on the result.
    `pause`, if given, is called before every call after the first: it can wait (a recording started).
    Returns notes with "labels" (raw) still in them and _meta covering all calls."""
    prompt = summary_prompt(ws)
    parts = split_transcript(transcript, cfg.get("chunk_minutes") or 0)
    if len(parts) == 1:
        notes = _summarise(cfg, model, name, transcript, prompt)
    else:
        done, metas = [], []
        for i, (head, text, a, b) in enumerate(parts, 1):
            if pause and i > 1:
                pause()
            log(f"{name}: part {i}/{len(parts)} ({library._fmt(a)}-{library._fmt(b)})")
            context = _people_context(merge_parts(done)) if done else ""
            user = (f"{head}\nPART {i} OF {len(parts)} ({library._fmt(a)} to {library._fmt(b)})\n\n"
                    f"{context}{text}\n")
            n = _summarise(cfg, model, name, user, prompt + PART_NOTE)
            metas.append(n.pop("_meta"))
            n, _ = labels.respell(n, ws)
            check_evidence(n, transcript, ws)  # before merging, so a field only travels with its line
            done.append(n)
        notes = merge_parts(done)
        if pause:
            pause()
        log(f"{name}: tidying the joined notes")
        notes, meta = tidy(cfg, model, name, notes)
        if meta:
            metas.append(meta)
        usage = {k: sum((m.get("usage") or {}).get(k) or 0 for m in metas)
                 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        secs = round(sum(m["seconds"] for m in metas), 1)
        notes["_meta"] = {"model": model, "seconds": secs, "usage": usage, "memory": metas[-1].get("memory"),
                          "calls": [{"seconds": m["seconds"], "tokens_per_s": m.get("tokens_per_s")} for m in metas],
                          "tokens_per_s": round(usage["completion_tokens"] / secs, 1) if secs else None}
    raw = notes.pop("labels", None)
    notes["title"] = re.sub(r"\s*[(\[]?\bpart\s*\d+(\s*(of|/)\s*\d+)?[)\]]?\s*$", "", str(notes.get("title") or ""),
                            flags=re.I).strip(" -–:")
    notes, _ = labels.respell(notes, ws)  # names the model kept as heard, in the registry spelling
    notes["people"] = [p for p in notes.get("people") or []
                       if not (isinstance(p, dict) and re.fullmatch(r"(?i)speaker[ _]?\d+", str(p.get("name") or "").strip()))]
    cleared = check_evidence(notes, transcript, ws)
    people_dropped = check_people(notes, transcript)
    repeats = dedupe_notes(notes)
    check_hierarchy(notes)
    owners = check_owners(notes)
    unlabelled = drop_speaker_labels(notes)
    dropped = drop_sensitive(notes)
    notes["_meta"]["reasoning_effort"] = cfg.get("reasoning_effort") or None
    if cleared:
        notes["_meta"]["cleared_without_evidence"] = cleared
    if dropped:
        notes["_meta"]["sensitive_sentences_dropped"] = dropped
    if owners:
        notes["_meta"]["owner_unassigned"] = owners
    if people_dropped:
        notes["_meta"]["people_dropped"] = people_dropped
    if repeats:
        notes["_meta"]["repeats_removed"] = repeats
    if unlabelled:
        notes["_meta"]["speaker_labels_replaced"] = unlabelled
    if raw is not None:
        notes["labels"] = raw
    return notes


def summarise_one(name, model=None, pause=None, log=print):
    """Summarise one queued transcript. Returns the parsed notes; raises on failure (input stays queued)."""
    cfg = config()
    model = model or cfg["model"]
    library.audio_path(name)  # must be a real recording: a plain file name, so the paths below stay in llm/
    src = library.LLM_INBOX / f"{name}.md"
    if not src.is_file():
        raise library.LibraryError(f"{name} is not queued; move it to 'To summarise' first")
    ws = labels.of(name)  # the recording's own workspace, whichever is selected in the header now
    try:
        notes = make_notes(cfg, model, name, src.read_text(), ws, log=log, pause=pause)
    finally:
        if cfg["unload_after_summary"]:
            unload(model)
    raw = notes.pop("labels", None)
    (library.LLM_SUMMARIES / f"{name}.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1))
    (library.LLM_SUMMARIES / f"{name}.md").write_text(to_markdown(name, notes, model))
    src.rename(library.LLM_PROCESSED / src.name)
    library.mark_summarised(name)
    if isinstance(raw, dict):  # a label problem never costs the summary; labels can be re-suggested
        try:
            labels.save_suggestion(name, labels.from_llm(raw, ws, model))
        except Exception as e:
            print(f"labels for {name} not saved: {type(e).__name__}: {e}")
    return notes


def respell_one(name):
    """Re-apply the registry spellings to an existing summary (after you taught the registry new
    spellings). No LLM call. Returns the number of names changed."""
    library.audio_path(name)
    js, md = library.LLM_SUMMARIES / f"{name}.json", library.LLM_SUMMARIES / f"{name}.md"
    if not js.is_file():
        raise library.LibraryError(f"{name} has no summary")
    notes, n = labels.respell(json.loads(js.read_text()), labels.of(name))
    if n:
        m = re.search(r"summarised by (.+?)_\n", md.read_text()) if md.is_file() else None
        js.write_text(json.dumps(notes, ensure_ascii=False, indent=1))
        md.write_text(to_markdown(name, notes, m.group(1) if m else "the summariser"))
    return n


def _transcript(name):
    """The transcript handed to the summariser, queued or already summarised."""
    library.audio_path(name)
    src = next((p for p in (library.LLM_INBOX / f"{name}.md", library.LLM_PROCESSED / f"{name}.md")
                if p.is_file()), None)
    if not src:
        raise library.LibraryError(f"{name} has no transcript handed to the summariser yet; "
                                   "move it to 'To summarise' first")
    return src.read_text()


def label_one(name, ws=None, model=None):
    """Suggest labels for a meeting without re-summarising it (e.g. meetings summarised before labels
    existed). Replaces earlier labels, confirmed ones too, with a new suggestion to review."""
    cfg = config()
    model = model or cfg["model"]
    ws = labels.of(name) if ws is None else ws
    try:
        # Thinking stays on: with it off (Ollama reasoning_effort "none") the 9b was 4x faster but added
        # registry names that were never said and missed real ones. With it on, one meeting took 7 min
        # and another ran past the 15-min summary timeout, so labels get their own, longer limit.
        reply = _summarise(cfg, model, name, _transcript(name), labels_prompt(ws), timeout=cfg["labels_timeout_s"])
    finally:
        if cfg["unload_after_summary"]:
            unload(model)
    raw = reply.get("labels", reply)
    if not isinstance(raw, dict):
        raise library.LibraryError("the model's reply had no labels")
    return labels.save_suggestion(name, labels.from_llm(raw, ws, model), force=True)


# ---------- model comparison ----------
# Test summaries live apart from the real one, in llm/compare/<recording>.model=<model>.json/.md: running
# a comparison never changes the recording's summary or its folder.
def _compare_path(name, model, ext):
    slug = re.sub(r"[^\w.-]+", "_", model)
    return library.LLM_COMPARE / f"{name}.model={slug}{ext}"


def compare_one(name, model):
    """Summarise a recording's transcript (queued or already summarised) with `model`, for comparison."""
    cfg = config()
    notes = make_notes(cfg, model, name, _transcript(name), labels.of(name))
    notes.pop("labels", None)  # same prompt as a real summary; labels aren't compared
    _compare_path(name, model, ".json").write_text(json.dumps(notes, ensure_ascii=False, indent=1))
    _compare_path(name, model, ".md").write_text(to_markdown(name, notes, model))
    return notes


def comparisons(name):
    """Saved comparison runs of a recording, one per model (latest run wins)."""
    library.audio_path(name)
    runs = []
    for p in sorted(library.LLM_COMPARE.glob(f"{library.glob_escape(name)}.model=*.json")):
        try:
            runs.append(json.loads(p.read_text()))
        except ValueError:
            pass
    return runs


def queued():
    return sorted(p.name[:-3] for p in library.LLM_INBOX.glob("*.md"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="recordings to summarise (default: everything queued)")
    p.add_argument("--check", action="store_true", help="only check the LLM connection")
    p.add_argument("--models", action="store_true", help="list the chat models the server has")
    p.add_argument("--compare", nargs="+", metavar="MODEL",
                   help="summarise NAMES with each MODEL into llm/compare/ (doesn't touch the real summary)")
    p.add_argument("--label", action="store_true", help="only suggest labels for NAMES (no new summary)")
    p.add_argument("--workspace", help="with --label: workspace id to label in (default: the active one)")
    p.add_argument("--respell", action="store_true",
                   help="re-apply registry spellings to the summaries of NAMES (default: all summaries)")
    a = p.parse_args()
    if a.respell:
        for name in a.names or sorted(p.name[:-5] for p in library.LLM_SUMMARIES.glob("*.json")):
            try:
                print(f"{'✓' if (n := respell_one(name)) else '·'} {name}: {n} name(s) respelled")
            except Exception as e:
                print(f"✗ {name}: {type(e).__name__}: {e}")
        return
    if a.label:
        for name in a.names:
            try:
                lab = label_one(name, a.workspace)
                found = {f: lab[f] for f in labels.KINDS.values() if lab[f]}
                print(f"✓ {name} [{lab['workspace'] or 'no workspace'}] {lab['type']}: {found}"
                      + (f" new: {[n['name'] for n in lab['new']]}" if lab["new"] else ""))
            except Exception as e:
                print(f"✗ {name}: {type(e).__name__}: {e}")
        return
    if a.models:
        default = config()["model"]
        print("\n".join(f"{m}{'  (default)' if m == default else ''}" for m in models()) or "no models found")
        return
    if a.compare:
        if not a.names:
            p.error("--compare needs recording names after the models (use -- before them)")
        for name in a.names:
            for model in a.compare:
                unload_all()
                try:
                    n = compare_one(name, model)
                    m = n["_meta"]
                    mem = f"{m['memory']['total_gb']} GB" if m["memory"] else "memory n/a"
                    print(f"✓ {name} · {model}: {m['seconds']}s, {m['tokens_per_s'] or '?'} tok/s, {mem}, "
                          f"{len(n.get('action_items', []))} action items")
                except Exception as e:
                    print(f"✗ {name} · {model}: {type(e).__name__}: {e}")
        unload_all()
        return
    status = check()
    print(f"LLM: {status['model']} at {status['base_url']}: {status['message']}")
    if a.check or not status["ok"]:
        sys.exit(0 if status["ok"] else 1)
    for name in a.names or queued():
        try:
            notes = summarise_one(name)
            print(f"✓ {name}: {len(notes.get('action_items', []))} action items "
                  f"({notes['_meta']['seconds']}s) -> llm/summaries/{name}.md")
        except Exception as e:
            print(f"✗ {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
