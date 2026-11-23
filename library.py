"""Recording library: folders, rename, delete, and the hand-off folders an LLM reads transcripts from.

Folders are labels kept in library.json; audio always stays in recordings/ (so playback and transcription
never break). The llm/ folders are real, because they are the contract with the summariser:

  llm/inbox/      transcript + metadata as markdown, written when a recording moves to "To summarise"
  llm/summaries/  the LLM's summary and action items (<recording>.md and .json)
  llm/processed/  transcripts that have been summarised
"""
import json
import re
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
RECORDINGS = ROOT / "recordings"
TRANSCRIPTS = ROOT / "out" / "ui"
LLM_INBOX = ROOT / "llm" / "inbox"
LLM_PROCESSED = ROOT / "llm" / "processed"
LLM_SUMMARIES = ROOT / "llm" / "summaries"
LLM_COMPARE = ROOT / "llm" / "compare"  # test summaries by other models, see summarise.compare_one
INDEX = ROOT / "library.json"
for d in (RECORDINGS, TRANSCRIPTS, LLM_INBOX, LLM_PROCESSED, LLM_SUMMARIES, LLM_COMPARE):
    d.mkdir(parents=True, exist_ok=True)

FOLDERS = {"inbox": "Inbox", "to_summarise": "To summarise", "summarised": "Summarised",
           "archive": "Archive", "trash": "Trash"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".aiff", ".flac", ".mp4"}
HANDOFF_VARIANTS = ("medium.en", "medium")  # prefer the English transcript for the LLM

_lock = threading.RLock()  # index + file moves happen from FastAPI's thread pool


class LibraryError(Exception):
    pass


def _load():
    return json.loads(INDEX.read_text()) if INDEX.is_file() else {}


def _save(index):
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True))
    tmp.replace(INDEX)  # atomic: a crash never leaves a half-written index


def audio_path(name):
    p = (RECORDINGS / name).resolve()
    if p.parent != RECORDINGS.resolve() or not p.is_file():
        raise LibraryError(f"recording not found: {name}")
    return p


def transcript_files(name):
    """All saved transcripts of a recording: <name>.<variant>.json / .txt."""
    return sorted(TRANSCRIPTS.glob(f"{glob_escape(name)}.*"))


def glob_escape(s):
    return re.sub(r"([\[\]*?])", r"[\1]", s)


def llm_files(name):
    """Hand-off and summary files belonging to a recording."""
    return [p for p in (LLM_INBOX / f"{name}.md", LLM_PROCESSED / f"{name}.md",
                        LLM_SUMMARIES / f"{name}.md", LLM_SUMMARIES / f"{name}.json") if p.is_file()] + \
        sorted(LLM_COMPARE.glob(f"{glob_escape(name)}.model=*"))


def best_transcript(name):
    for v in HANDOFF_VARIANTS:
        p = TRANSCRIPTS / f"{name}.{v}.json"
        if p.is_file():
            return json.loads(p.read_text())
    return None


def _title(name):
    js = LLM_SUMMARIES / f"{name}.json"
    try:
        return json.loads(js.read_text()).get("title") if js.is_file() else None
    except ValueError:
        return None


def folder_of(name, index=None):
    return (index if index is not None else _load()).get(name, {}).get("folder", "inbox")


def items():
    """Every recording with its folder and what has been produced for it."""
    with _lock:
        index = _load()
        out = []
        for p in sorted(RECORDINGS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if p.suffix.lower() not in AUDIO_EXT:
                continue
            t = best_transcript(p.name)
            stats = (t or {}).get("stats", {})
            out.append({
                "name": p.name,
                "folder": folder_of(p.name, index),
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes"),
                "duration": stats.get("duration"),
                "speakers": stats.get("speakers"),
                "transcripts": sorted(f.name[len(p.name) + 1:-5] for f in TRANSCRIPTS.glob(f"{glob_escape(p.name)}.*.json")),
                "queued": (LLM_INBOX / f"{p.name}.md").is_file(),
                "summary": (LLM_SUMMARIES / f"{p.name}.md").is_file(),
                "title": _title(p.name),
            })
        return out


def counts():
    c = {f: 0 for f in FOLDERS}
    for it in items():
        c[it["folder"]] += 1
    return c


def _fmt(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def export_handoff(name):
    """Write the transcript as markdown for the LLM: metadata header, then one line per utterance."""
    t = best_transcript(name)
    if not t:
        raise LibraryError(f"{name} has no transcript yet; transcribe it first")
    s = t["stats"]
    meta = {
        "recording": name,
        "recorded": datetime.fromtimestamp(audio_path(name).stat().st_mtime).isoformat(timespec="minutes"),
        "duration": _fmt(s["duration"]),
        "speakers": s.get("speakers"),
        "spoken_languages": ", ".join(s.get("languages", [])),
        "transcript_language": "English (translated by Whisper)" if s.get("task") == "translate" else "as spoken",
        "transcribed_with": f"whisper-{s.get('model')} + pyannote community-1",
    }
    body = "\n".join(f"[{_fmt(l['start'])}] {l['speaker']}: {l['text']}" for l in t["lines"])
    text = "---\n" + "\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n---\n\n# Transcript\n\n" + body + "\n"
    (LLM_INBOX / f"{name}.md").write_text(text)


def move(names, folder):
    if folder not in FOLDERS:
        raise LibraryError(f"unknown folder {folder!r}")
    with _lock:
        index = _load()
        for name in names:
            audio_path(name)
            if folder == "to_summarise":
                export_handoff(name)  # raises if there is no transcript, leaving the index unchanged
            elif folder_of(name, index) == "to_summarise":
                (LLM_INBOX / f"{name}.md").unlink(missing_ok=True)  # taken out of the queue
            index.setdefault(name, {})["folder"] = folder
        _save(index)


def mark_summarised(name):
    with _lock:
        index = _load()
        index.setdefault(name, {})["folder"] = "summarised"
        _save(index)


def rename(old, new_stem):
    """Rename a recording and everything derived from it, keeping the audio's extension."""
    with _lock:
        src = audio_path(old)
        stem = re.sub(r"[^\w.-]+", "_", new_stem.strip())[:100].strip("._")
        if not stem:
            raise LibraryError("new name is empty")
        new = f"{stem}{src.suffix}"
        if new == old:
            return old
        if (RECORDINGS / new).exists():
            raise LibraryError(f"{new} already exists")
        src.rename(RECORDINGS / new)
        for f in transcript_files(old):
            dest = TRANSCRIPTS / (new + f.name[len(old):])
            if f.suffix == ".json":  # the transcript records which recording it belongs to
                data = json.loads(f.read_text())
                data["recording"] = new
                dest.write_text(json.dumps(data, ensure_ascii=False, indent=1))
                f.unlink()
            else:
                f.rename(dest)
        for f in llm_files(old):
            dest = f.with_name(new + f.name[len(old):])
            if f.suffix == ".md" and f.parent in (LLM_INBOX, LLM_PROCESSED):
                dest.write_text(f.read_text().replace(f"recording: {old}\n", f"recording: {new}\n", 1))
                f.unlink()
            else:
                f.rename(dest)
        index = _load()
        if old in index:
            index[new] = index.pop(old)
            _save(index)
        return new


def delete(names, permanent=False):
    """Move to Trash, or (permanent) remove audio, transcripts, LLM files and the index entry."""
    with _lock:
        index = _load()
        for name in names:
            path = audio_path(name)
            if not permanent:
                (LLM_INBOX / f"{name}.md").unlink(missing_ok=True)
                index.setdefault(name, {})["folder"] = "trash"
                continue
            for f in [path, *transcript_files(name), *llm_files(name)]:
                f.unlink(missing_ok=True)
            index.pop(name, None)
        _save(index)


def empty_trash():
    trash = [it["name"] for it in items() if it["folder"] == "trash"]
    delete(trash, permanent=True)
    return len(trash)


def _summary_texts(data):
    """(section, text) pairs from a summary's JSON, for search."""
    out = [("title", data.get("title") or "")]
    for key in ("summary", "decisions", "open_questions"):
        out += [(key, str(x)) for x in data.get(key) or []]
    for a in data.get("action_items") or []:
        out.append(("action item", f"{a.get('owner') or 'unassigned'}: {a.get('task', '')}" if isinstance(a, dict) else str(a)))
    return out


def search(q, limit_hits=3):
    """Recordings whose name, summary or transcript contain every word of q (case-insensitive).
    Returns {name: {"hits": [{"where", "text", "start"?}], "count": total matching lines}}."""
    words = [w for w in q.lower().split() if w]
    if not words:
        return {}
    match = lambda text: all(w in text.lower() for w in words)
    out = {}
    for p in RECORDINGS.iterdir():
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        hits = [{"where": "name", "text": p.name}] if match(p.name.replace("_", " ")) or match(p.name) else []
        js = LLM_SUMMARIES / f"{p.name}.json"
        if js.is_file():
            try:
                hits += [{"where": w, "text": t} for w, t in _summary_texts(json.loads(js.read_text())) if match(t)]
            except ValueError:
                pass
        t = best_transcript(p.name)
        hits += [{"where": "transcript", "text": ln["text"], "start": ln.get("start")}
                 for ln in (t or {}).get("lines", []) if match(ln.get("text", ""))]
        if hits:
            out[p.name] = {"hits": hits[:limit_hits], "count": len(hits)}
    return out


def summary(name):
    audio_path(name)  # rejects anything but a recording's plain file name (no ../ paths)
    md, js = LLM_SUMMARIES / f"{name}.md", LLM_SUMMARIES / f"{name}.json"
    if not md.is_file():
        raise LibraryError("no summary yet")
    return {"markdown": md.read_text(), "data": json.loads(js.read_text()) if js.is_file() else None}
