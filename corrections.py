"""Correct a misheard word in a meeting and teach it, so the transcriber and summariser get it right next time.

    correct("rec.wav", wrong="marlo", right="Marlow", kind="person", everywhere=False)

1. Learn (labels.learn_spelling): "marlo" becomes an alias of Marlow in the meeting's workspace registry, or
   of a plain word for terms that aren't meeting labels.
2. Fix this meeting, or every meeting of its workspace: the summary (JSON and Markdown), the saved
   transcripts, the copy handed to the summariser, and the labels file. Whole words, any case.
3. From then on:
   - transcription: Whisper is primed with the workspace's names (labels.vocabulary), and every saved
     transcript is respelled through the registry (finalize in server.py);
   - summaries: the prompt lists each name with its "heard as" spellings, and labels.respell runs after.
Nothing is retrained: the models stay as they are and the registry steers their spelling.
"""
import json
import re

import labels
import library
import summarise
from transcribe import fmt


def _pattern(wrong):
    words = wrong.split()
    return re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", re.IGNORECASE)


def _fix_strings(x, pat, right, count):
    if isinstance(x, str):
        new, n = pat.subn(right, x)
        count[0] += n
        return new
    if isinstance(x, list):
        return [_fix_strings(v, pat, right, count) for v in x]
    if isinstance(x, dict):
        # keys and bookkeeping stay as they are; only the meeting's own text changes
        return {k: v if k in ("_meta", "recording", "status", "workspace") else _fix_strings(v, pat, right, count)
                for k, v in x.items()}
    return x


def apply(name, wrong, right):
    """Replace `wrong` with `right` in one meeting's files. Returns {"summary": n, "transcript": n, ...}."""
    library.audio_path(name)  # validates the name before it's used in paths
    pat, out = _pattern(wrong), {}

    js, md = library.LLM_SUMMARIES / f"{name}.json", library.LLM_SUMMARIES / f"{name}.md"
    if js.is_file():
        n = [0]
        notes = _fix_strings(json.loads(js.read_text()), pat, right, n)
        if n[0]:
            m = re.search(r"summarised by (.+?)_\n", md.read_text()) if md.is_file() else None
            js.write_text(json.dumps(notes, ensure_ascii=False, indent=1))
            md.write_text(summarise.to_markdown(name, notes, m.group(1) if m else "the summariser"))
        out["summary"] = n[0]

    n = [0]
    for tj in library.TRANSCRIPTS.glob(f"{glob_escape(name)}.*.json"):
        t = json.loads(tj.read_text())
        before = n[0]
        for line in t.get("lines", []):
            line["text"] = _fix_strings(line["text"], pat, right, n)
        if n[0] > before:
            tj.write_text(json.dumps(t, ensure_ascii=False, indent=1))
            tj.with_suffix(".txt").write_text(
                "\n".join(f"[{fmt(l['start'])}] {l['speaker']}: {l['text']}" for l in t["lines"]) + "\n")
    out["transcript"] = n[0]

    n = [0]
    for p in (library.LLM_INBOX / f"{name}.md", library.LLM_PROCESSED / f"{name}.md"):
        if p.is_file():
            new = _fix_strings(p.read_text(), pat, right, n)
            p.write_text(new)
    out["summariser_copy"] = n[0]

    lab = library.LLM_LABELS / f"{name}.json"
    if lab.is_file():
        n = [0]
        fixed = _fix_strings(json.loads(lab.read_text()), pat, right, n)
        if n[0]:
            lab.write_text(json.dumps(fixed, ensure_ascii=False, indent=1))
        out["labels"] = n[0]
    return out


def glob_escape(s):
    return re.sub(r"([*?\[])", r"[\1]", s)


def correct(name, wrong, right, kind="", everywhere=False):
    """Learn the spelling in the meeting's workspace and fix this meeting (or all of the workspace's).
    Returns {"learned": sentence, "workspace": id, "meetings": {name: counts}, "changed": [names]}."""
    wrong, right = " ".join((wrong or "").split()), " ".join((right or "").split())
    ws = labels.of(name)
    if not ws:
        raise labels.LabelError(f"{name} is in no workspace; move it to one first")
    learned = labels.learn_spelling(ws, wrong, right, kind)
    names = [it["name"] for it in library.items(ws) if it["folder"] != "trash"] if everywhere else [name]
    meetings = {n: apply(n, wrong, right) for n in names}
    changed = [n for n, c in meetings.items() if any(c.values())]
    return {"learned": learned, "workspace": ws, "meetings": {n: meetings[n] for n in changed} or {name: meetings[name]},
            "changed": changed}
