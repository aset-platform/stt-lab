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

import settings

ROOT = Path(__file__).parent
RECORDINGS = ROOT / "recordings"
TRANSCRIPTS = ROOT / "out" / "ui"
LLM_INBOX = ROOT / "llm" / "inbox"
LLM_PROCESSED = ROOT / "llm" / "processed"
LLM_SUMMARIES = ROOT / "llm" / "summaries"
LLM_COMPARE = ROOT / "llm" / "compare"  # test summaries by other models, see summarise.compare_one
LLM_LABELS = ROOT / "llm" / "labels"  # meeting labels (suggested / confirmed), see labels.py
LLM_SPEAKERS = ROOT / "llm" / "speakers"  # who each SPEAKER_nn of a meeting is, see speaker_names
INDEX = ROOT / "library.json"
for d in (RECORDINGS, TRANSCRIPTS, LLM_INBOX, LLM_PROCESSED, LLM_SUMMARIES, LLM_COMPARE, LLM_LABELS, LLM_SPEAKERS):
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
                        LLM_SUMMARIES / f"{name}.md", LLM_SUMMARIES / f"{name}.json",
                        LLM_LABELS / f"{name}.json", LLM_SPEAKERS / f"{name}.json") if p.is_file()] + \
        sorted(LLM_COMPARE.glob(f"{glob_escape(name)}.model=*"))


_WORDS = re.compile(r"\S+")


def _collapse_loops(text, times=3):
    """A phrase of 1-8 words said `times` or more times in a row is kept once: Whisper fills silence and
    noise by looping ("okay okay okay ...", "thank you. thank you. thank you."). Real speech
    repeating a phrase twice is left alone."""
    words = _WORDS.findall(text)
    key = [re.sub(r"[^\w]", "", w.lower()) for w in words]
    out, i = [], 0
    while i < len(words):
        for n in range(1, 9):
            reps = 1
            while i + (reps + 1) * n <= len(words) and key[i + reps * n:i + (reps + 1) * n] == key[i:i + n] \
                    and any(key[i:i + n]):
                reps += 1
            if reps >= times:
                out += words[i:i + n]
                i += reps * n
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def drop_repeats(lines, window=4):
    """Lines without Whisper's silence filler: loops inside a line are collapsed, and a line that repeats
    one of the `window` lines before it word for word is dropped ("Thanks for watching." over and over).
    A short reply ("Okay.", "Yes.") is only dropped when the same speaker repeats it. Returns (lines, n)."""
    out, recent, n = [], [], 0
    for l in lines:
        text = _collapse_loops(l.get("text") or "")
        key = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
        if not key:
            n += 1
            continue
        same = [r for r in recent if r[0] == key]
        if same and (len(key.split()) >= 3 or any(r[1] == l.get("speaker") for r in same)):
            n += 1
            continue
        if text != (l.get("text") or ""):
            n += 1
        out.append({**l, "text": text})
        recent = (recent + [(key, l.get("speaker"))])[-window:]
    return out, n


_LABEL = re.compile(r"SPEAKER_\d+")
UNKNOWN = "?"  # "don't know": the voice is answered for, and stays SPEAKER_nn in the transcript


def speaker_names(name):
    """{"SPEAKER_01": "Marlow", ...}: the names you gave this meeting's voices (empty if none)."""
    p = LLM_SPEAKERS / f"{name}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def set_speaker_names(name, names):
    """Save who each voice is. Blank names are dropped; one name for two labels is allowed (diarisation
    sometimes splits a voice). Returns the saved mapping."""
    audio_path(name)
    clean = {}
    for label, who in (names or {}).items():
        who = re.sub(r"\s+", " ", str(who or "")).strip()[:60]
        if not _LABEL.fullmatch(str(label)):
            raise LibraryError(f"not a speaker label: {label!r}")
        if who:
            if who != UNKNOWN and _LABEL.fullmatch(who.upper().replace(" ", "_")):
                raise LibraryError(f"give {label} a person's name, not a label")
            clean[label] = who
    p = LLM_SPEAKERS / f"{name}.json"
    if clean:
        p.write_text(json.dumps(clean, ensure_ascii=False, indent=1))
    else:
        p.unlink(missing_ok=True)
    with _lock:
        if folder_of(name) == "to_summarise":
            export_handoff(name)  # already queued: the summariser reads the names from the hand-off
    return clean


def name_speakers(lines, names):
    """The lines with each named SPEAKER_nn replaced by the person's name (new dicts; others unchanged)."""
    if not names:
        return lines
    return [{**l, "speaker": who if (who := names.get(l.get("speaker"))) and who != UNKNOWN else l.get("speaker")}
            for l in lines]


def voices(t):
    """The SPEAKER_nn labels that say something in a transcript (as saved, before names are applied)."""
    return sorted({l["speaker"] for l in (t or {}).get("lines", []) if l.get("speaker") and l.get("text", "").strip()})


def missing_speakers(name, t=None, save=False):
    """Voices with neither a name nor "don't know". A meeting with a single voice is yours: with your name
    set (Settings → Recording) it counts as named, and `save` writes that down."""
    t = t if t is not None else best_transcript(name, named=False)
    labels_ = voices(t)
    names = speaker_names(name)
    missing = [l for l in labels_ if l not in names]
    me = str(settings.get("recording").get("your_name") or "").strip()
    if len(labels_) == 1 and missing and me:
        if save:
            set_speaker_names(name, {labels_[0]: me})
        return []
    return missing


def names_required():
    return bool(settings.get("llm").get("require_speaker_names", True))


def needs_names(name, t=None):
    """True while a meeting must wait for its voices to be named before it is summarised."""
    return names_required() and bool(missing_speakers(name, t))


def best_transcript(name, named=True):
    """The transcript used for summaries, the index and evaluation, without Whisper's silence filler
    (drop_repeats; transcripts saved before the filter existed are cleaned here too). With `named`, the
    voices you named (speaker_names) carry the person's name instead of SPEAKER_nn."""
    for v in HANDOFF_VARIANTS:
        p = TRANSCRIPTS / f"{name}.{v}.json"
        if p.is_file():
            t = json.loads(p.read_text())
            t["lines"], _ = drop_repeats(t.get("lines") or [])
            if named:
                t["lines"] = name_speakers(t["lines"], speaker_names(name))
            return t
    return None


def speaker_overview(name):
    """Each voice of the meeting with its talk time and a few of its longer lines, to tell who is who."""
    t = best_transcript(name, named=False)
    if not t:
        raise LibraryError(f"{name} has no transcript yet")
    out = {}
    for l in t["lines"]:
        sp = out.setdefault(l["speaker"], {"label": l["speaker"], "seconds": 0.0, "lines": 0, "samples": []})
        sp["seconds"] += max(0.0, float(l.get("end", 0)) - float(l.get("start", 0)))
        sp["lines"] += 1
        sp["samples"].append((len(l["text"]), l["start"], l["text"]))
    for sp in out.values():
        best = sorted(sp["samples"], reverse=True)[:3]
        sp["samples"] = [{"at": _fmt(a), "text": t} for _, a, t in sorted(best, key=lambda x: x[1])]
        sp["seconds"] = round(sp["seconds"])
    return sorted(out.values(), key=lambda s: s["label"])


def _title(name):
    js = LLM_SUMMARIES / f"{name}.json"
    try:
        return json.loads(js.read_text()).get("title") if js.is_file() else None
    except ValueError:
        return None


def _label_status(name):
    """None, "suggested" (waiting for review) or "confirmed"."""
    p = LLM_LABELS / f"{name}.json"
    try:
        return json.loads(p.read_text()).get("status") if p.is_file() else None
    except ValueError:
        return None


def folder_of(name, index=None):
    return (index if index is not None else _load()).get(name, {}).get("folder", "inbox")


def workspace_of(name, index=None):
    """The workspace (firm) a recording belongs to; "" = none. Set when it is recorded or added, and
    when you confirm its labels in another workspace."""
    entry = (index if index is not None else _load()).get(name, {})
    if "workspace" in entry:
        return entry["workspace"]
    p = LLM_LABELS / f"{name}.json"
    try:
        return (json.loads(p.read_text()).get("workspace") or "") if p.is_file() else ""
    except ValueError:
        return ""


def set_workspace(names, ws):
    with _lock:
        index = _load()
        for name in names:
            audio_path(name)
            index.setdefault(name, {})["workspace"] = ws or ""
        _save(index)


def items(ws=None):
    """Every recording (of workspace `ws`, or all when None) with its folder and what has been produced for it."""
    with _lock:
        index = _load()
        out = []
        for p in sorted(RECORDINGS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if p.suffix.lower() not in AUDIO_EXT:
                continue
            if ws is not None and workspace_of(p.name, index) != ws:
                continue
            t = best_transcript(p.name, named=False)
            stats = (t or {}).get("stats", {})
            out.append({
                "name": p.name,
                "folder": folder_of(p.name, index),
                "workspace": workspace_of(p.name, index),
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="minutes"),
                "duration": stats.get("duration"),
                "speakers": stats.get("speakers"),
                "transcripts": sorted(f.name[len(p.name) + 1:-5] for f in TRANSCRIPTS.glob(f"{glob_escape(p.name)}.*.json")),
                "queued": (queued := (LLM_INBOX / f"{p.name}.md").is_file()),
                # voices still to name before the queue may summarise it (0 when not required)
                "needs_names": len(missing_speakers(p.name, t)) if queued and t and names_required() else 0,
                "summary": (LLM_SUMMARIES / f"{p.name}.md").is_file(),
                "title": _title(p.name),
                "labels": _label_status(p.name),
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
    named = speaker_names(name)
    if named:  # the lines already carry these names; said here so the model trusts them
        meta["named_voices"] = ", ".join(f"{who} (was {label})" for label, who in sorted(named.items()))
    me = str(settings.get("recording").get("your_name") or "").strip()
    if me:
        meta["recorded_by"] = f"{me} (made this recording and is one of the speakers)"
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
                missing_speakers(name, save=True)  # a single voice gets your name before the hand-off
                export_handoff(name)  # raises if there is no transcript, leaving the index unchanged
            elif folder_of(name, index) == "to_summarise":
                (LLM_INBOX / f"{name}.md").unlink(missing_ok=True)  # taken out of the queue
            index.setdefault(name, {})["folder"] = folder
        _save(index)
    return [n for n in names if folder == "to_summarise" and needs_names(n)]


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


def search(q, limit_hits=3, ws=None):
    """Recordings whose name, summary or transcript contain every word of q (case-insensitive).
    Returns {name: {"hits": [{"where", "text", "start"?}], "count": total matching lines}}."""
    words = [w for w in q.lower().split() if w]
    if not words:
        return {}
    match = lambda text: all(w in text.lower() for w in words)
    out, index = {}, _load()
    for p in RECORDINGS.iterdir():
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        if ws is not None and workspace_of(p.name, index) != ws:
            continue
        hits = [{"where": "name", "text": p.name}] if match(p.name.replace("_", " ")) or match(p.name) else []
        lj = LLM_LABELS / f"{p.name}.json"
        if lj.is_file():
            try:
                lab = json.loads(lj.read_text())
                words_ = [lab.get("type", ""), *(n for f in ("clients", "projects", "people", "topics")
                                                 for n in lab.get(f, []))]
                if match(" ".join(words_)):
                    hits.append({"where": "labels", "text": ", ".join(w for w in words_ if w)})
            except ValueError:
                pass
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
