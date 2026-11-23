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
from pathlib import Path

import labels
import library
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


def to_markdown(name, notes, model):
    def section(title, items):
        return f"## {title}\n\n" + ("\n".join(f"- {i}" for i in items) if items else "_None._") + "\n"
    actions = [f"**{a.get('owner') or 'unassigned'}**: {a.get('task')}" + (f" (due {a['due']})" if a.get("due") else "")
               for a in notes.get("action_items", [])]
    return (f"# {notes.get('title') or name}\n\n_Recording: {name} · summarised by {model}_\n\n"
            + section("Summary", notes.get("summary", [])) + "\n"
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
    reply = _request(cfg, "/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": transcript}],
        "temperature": 0.2,
        #"response_format": {"type": "json_object"},
    }, timeout=timeout or cfg["timeout_s"])
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


def summarise_one(name, model=None):
    """Summarise one queued transcript. Returns the parsed notes; raises on failure (input stays queued)."""
    cfg = config()
    model = model or cfg["model"]
    library.audio_path(name)  # must be a real recording: a plain file name, so the paths below stay in llm/
    src = library.LLM_INBOX / f"{name}.md"
    if not src.is_file():
        raise library.LibraryError(f"{name} is not queued; move it to 'To summarise' first")
    ws = labels.of(name)  # the recording's own workspace, whichever is selected in the header now
    try:
        notes = _summarise(cfg, model, name, src.read_text(), summary_prompt(ws))
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
    notes = _summarise(cfg, model, name, _transcript(name), summary_prompt(labels.of(name)))
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
    a = p.parse_args()
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
