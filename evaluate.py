"""How much of a meeting made it into its summary: a coverage check of the notes against the transcript.

    uv run evaluate.py NAME                      the real summary (llm/summaries/)
    uv run evaluate.py NAME --compare MODEL      a comparison run (llm/compare/)
    uv run evaluate.py NAME --model M            judge with another local model

Two checks:
1. Names (no LLM): people, clients and projects said in the transcript (registry names and their "heard
   as" spellings, plus capitalised words that recur mid-sentence) that the notes never mention.
2. Points (LLM): the transcript is cut into segments of about five minutes; for each, the model lists the
   points a reader of the notes would need (people and roles, decisions, commitments, numbers and dates,
   risks, plans) and says whether the notes cover each one. Score = covered share, high-importance
   points counting double, partial as half. Segments make the judge read every part of the meeting, which
   is exactly where a one-pass summary loses things (the middle of a long call).

The report is saved to llm/evals/<recording>.<summary|model>.json.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter

import labels
import library
import summarise
from transcribe import fmt

EVALS = library.ROOT / "llm" / "evals"
SEGMENT_S = 300
STOP = set("""I A An The And But Or So Okay OK Yes No Yeah Hmm Right Sir Madam Thank Thanks Hello Hi
Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July August
September October November December English Hindi India Indian AI ML API CEO CIO CTO VP AVP SVP HR BAU
SPEAKER""".split())

JUDGE = """You check meeting notes against the meeting transcript.

You get one part of the transcript (with [mm:ss] times) and the full notes written for the whole
meeting. List the important points made in THIS part that someone reading only the notes would need:
people and their roles or changes in responsibility, organisation and team structure, decisions,
commitments and who owns them, numbers and dates, risks and problems, plans and next steps.
Skip small talk, filler and repetition. At most 6 points; fewer if the part is light.

For each point judge the notes:
- "covered": the notes state it (any wording)
- "partial": the notes touch it but miss something that matters (e.g. a person's name or role)
- "missing": not in the notes

Reply with JSON only:
{"points": [{"point": "one sentence", "at": "mm:ss", "importance": "high" or "medium",
             "coverage": "covered" or "partial" or "missing", "why": "for partial/missing: what is absent"}]}
"""


def _notes_text(notes):
    parts = [notes.get("title") or ""]
    for key in ("summary", "people", "decisions", "open_questions"):
        for x in notes.get(key) or []:
            parts.append(json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x))
    for a in notes.get("action_items") or []:
        parts.append(f"{a.get('owner')}: {a.get('task')} {a.get('due') or ''}" if isinstance(a, dict) else str(a))
    return "\n".join(p for p in parts if p)


def load_notes(name, compare=None):
    path = summarise._compare_path(name, compare, ".json") if compare else library.LLM_SUMMARIES / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"no notes at {path}")
    return json.loads(path.read_text())


def name_gaps(name, lines, notes):
    """Names said in the meeting (at least twice) that the notes never mention: [{name, mentions, first_at}]."""
    text_notes = _notes_text(notes).lower()
    ws = labels.of(name)
    spoken = Counter()
    first = {}
    known = {}
    if ws and labels.exists(ws):
        for e in labels.load(ws)["entities"]:
            if e["kind"] in labels.NAMED:
                for n in [e["name"], *e.get("aliases", [])]:
                    known[n.lower()] = e["name"]
    pat = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(known, key=len, reverse=True)) + r")(?!\w)",
                     re.I) if known else None
    for l in lines:
        found = set()
        if pat:
            found |= {known[m.group(0).lower()] for m in pat.finditer(l["text"])}
        # capitalised words not at the start of a sentence: likely names ("The third one is Tamsin.")
        found |= {w for w in re.findall(r"(?<=[a-z,;] )([A-Z][a-z]{2,}|[A-Z]{3,})\b", l["text"]) if w not in STOP}
        for n in found:
            n = next((k for k in spoken if k.lower() == n.lower()), n)  # "Orrin" and "ORRIN" are one name
            spoken[n] += 1
            first.setdefault(n, l["start"])
    out = []
    for n, c in spoken.most_common():
        if c < 2:
            continue
        aliases = [k for k, v in known.items() if v == n] or [n.lower()]
        if not any(re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", text_notes) for a in aliases):
            out.append({"name": n, "mentions": c, "first_at": fmt(first[n])})
    return out


def segments(lines, seconds=SEGMENT_S):
    segs, cur, start = [], [], None
    for l in lines:
        if start is None:
            start = l["start"]
        if l["start"] - start >= seconds and cur:
            segs.append(cur)
            cur, start = [], l["start"]
        cur.append(l)
    if cur:
        segs.append(cur)
    return segs


def judge(cfg, model, seg, notes_text, effort="low"):
    part = "\n".join(f"[{fmt(l['start'])}] {l['speaker']}: {l['text']}" for l in seg)
    reply = summarise._request(cfg, "/chat/completions", {
        "model": model, "temperature": 0,
        # judging one part is a narrow task: a thinking model at full effort took 10+ min per part
        **({"reasoning_effort": effort} if effort else {}),
        "messages": [{"role": "system", "content": JUDGE},
                     {"role": "user", "content": f"NOTES:\n{notes_text}\n\nTRANSCRIPT PART "
                                                 f"({fmt(seg[0]['start'])}-{fmt(seg[-1]['end'])}):\n{part}"}],
    }, timeout=cfg["timeout_s"])
    try:
        return summarise._parse_json(reply["choices"][0]["message"]["content"]).get("points", [])
    except Exception as e:  # one bad reply shouldn't lose the whole evaluation
        return [{"point": f"(judge reply unreadable: {e})", "at": fmt(seg[0]["start"]), "importance": "medium",
                 "coverage": "error"}]


def evaluate(name, compare=None, model=None, log=print, effort="low"):
    cfg = summarise.config()
    model = model or cfg["model"]
    notes = load_notes(name, compare)
    t = library.best_transcript(name)
    if not t:
        raise SystemExit(f"{name} has no transcript")
    lines = [l for l in t["lines"] if l.get("text", "").strip()]
    t0 = time.perf_counter()
    gaps = name_gaps(name, lines, notes)
    notes_text = _notes_text(notes)
    points = []
    segs = segments(lines)
    for i, seg in enumerate(segs, 1):
        log(f"  judging part {i}/{len(segs)} ({fmt(seg[0]['start'])}-{fmt(seg[-1]['end'])})…")
        points += judge(cfg, model, seg, notes_text, effort)
    weight = {"high": 2, "medium": 1}
    value = {"covered": 1, "partial": 0.5, "missing": 0}
    scored = [p for p in points if p.get("coverage") in value]
    total = sum(weight.get(p.get("importance"), 1) for p in scored)
    got = sum(weight.get(p.get("importance"), 1) * value[p["coverage"]] for p in scored)
    report = {
        "recording": name, "notes": f"compare:{compare}" if compare else "summary",
        "notes_model": (notes.get("_meta") or {}).get("model"), "judge": model,
        "score": round(got / total, 2) if total else None,
        "points": len(scored), "covered": sum(p["coverage"] == "covered" for p in scored),
        "partial": sum(p["coverage"] == "partial" for p in scored),
        "missing": sum(p["coverage"] == "missing" for p in scored),
        "missing_names": gaps,
        "gaps": [p for p in scored if p["coverage"] != "covered"],
        "all_points": points, "seconds": round(time.perf_counter() - t0, 1),
    }
    EVALS.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^\w.-]+", "_", compare) if compare else "summary"
    out = EVALS / f"{name}.{label}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    report["saved_to"] = str(out.relative_to(library.ROOT))
    return report


def show(r):
    print(f"\n{r['recording']} · notes: {r['notes']} ({r['notes_model']}) · judge: {r['judge']} · {r['seconds']}s")
    print(f"coverage score {r['score']}: {r['covered']} covered, {r['partial']} partial, {r['missing']} missing "
          f"of {r['points']} points")
    if r["missing_names"]:
        print("names said in the meeting but absent from the notes: " +
              ", ".join(f"{g['name']} ({g['mentions']}x, from {g['first_at']})" for g in r["missing_names"]))
    for p in sorted(r["gaps"], key=lambda p: (p.get("importance") != "high", p.get("at", ""))):
        print(f"  [{p.get('at')}] {p['coverage'].upper():8} {p.get('importance', ''):6} {p['point']}"
              + (f"  — {p['why']}" if p.get("why") else ""))
    print(f"saved: {r['saved_to']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--compare", metavar="MODEL", help="evaluate the comparison run of MODEL instead")
    ap.add_argument("--model", help="judge model (default: the summariser's)")
    ap.add_argument("--effort", default="low", help='reasoning effort for thinking models: none, low, medium, high, '
                                                    'or "" to leave the model\'s default (default: low)')
    a = ap.parse_args()
    show(evaluate(a.name, a.compare, a.model, log=lambda m: print(m, file=sys.stderr), effort=a.effort))
