"""Local UI for meeting transcription: upload/record audio, stream transcript, manage recordings,
hand transcripts to a local LLM for summaries and action items.

Run: uv run server.py   then open http://127.0.0.1:8001
"""
import asyncio
import gc
import os
import json
import queue
import re
import subprocess
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx_whisper
import numpy as np
from fastapi import Body, FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

import index
import labels
import library
import observe
import settings
import share_filter
import summarise
import corrections
import evaluate
import transcribe
from library import AUDIO_EXT, RECORDINGS, TRANSCRIPTS, LibraryError
from transcribe import SR, diarize, fmt, merge

ROOT = Path(__file__).parent
# Only Whisper medium: best English translation of Hindi/Hinglish that still keeps up live on the M4
# (small mistranslated more; large-v3-turbo can't translate). One model = one copy in memory.
MODELS = {"medium": "mlx-community/whisper-medium-mlx"}
MAX_CHUNK_S = 25  # under Whisper's 30s window, so each chunk is one decode pass

# MLX keeps freed GPU buffers for reuse with no size limit by default; over a 27-min meeting that cache
# grew Whisper medium from ~2 GB to an 8.6 GB peak. Capping it keeps the reuse benefit without the growth.
mx.set_cache_limit(512 * 2**20)

# MLX keeps per-thread GPU streams, so every job runs on this one thread (also serialises jobs).
worker = ThreadPoolExecutor(max_workers=1)
app = FastAPI()
busy = set()  # recordings being transcribed right now: not renamable or deletable
# LLM summaries are Ollama calls, not MLX: they run on their own thread, one at a time, so a recording
# that starts mid-summary isn't stuck behind it on the MLX thread
summariser = ThreadPoolExecutor(max_workers=1)


class Activity:
    """Counts live recordings and transcriptions: while any runs, summaries don't start and pause."""

    def __init__(self):
        self.n, self.lock = 0, threading.Lock()

    def start(self):
        """Count one in; returns its end(), safe to call more than once."""
        with self.lock:
            self.n += 1
        ended = []

        def end():
            with self.lock:
                if not ended:
                    ended.append(1)
                    self.n -= 1
        return end


activity = Activity()

# ---------- meeting index (Postgres + pgvector) ----------
# Kept in step with the library by a sync after every change (confirm, move, rename, delete, summary):
# it only re-embeds meetings whose labels or summary changed, so a sync with nothing to do is one query.
indexer = ThreadPoolExecutor(max_workers=1)  # embeddings are Ollama calls: off the MLX thread
_sync_state = {"queued": False, "last": None}
_sync_lock = threading.Lock()


def reindex_soon():
    if not index.config()["enabled"]:
        return
    with _sync_lock:
        if _sync_state["queued"]:
            return  # the queued sync will see this change too
        _sync_state["queued"] = True
    indexer.submit(_sync)


def _sync():
    with _sync_lock:
        _sync_state["queued"] = False
    try:
        done, removed, failed = index.sync(log=lambda m: print(f"[index] {m}"))
        _sync_state["last"] = {"at": datetime.now().isoformat(timespec="seconds"), "indexed": done,
                               "removed": removed, "failed": failed}
    except Exception as e:  # database or embedding model not running: the library works without them
        _sync_state["last"] = {"at": datetime.now().isoformat(timespec="seconds"), "error": str(e)}


@app.exception_handler(LibraryError)
@app.exception_handler(settings.SettingsError)
def library_error(_, e):
    return JSONResponse({"detail": str(e)}, status_code=400)


class KeepAwake:
    """Hold off idle sleep while there is work (a live session, a transcription, a summary).

    On battery macOS idle-sleeps a few minutes after the display goes off: a meeting that ended at 09:22
    froze mid speaker-identification at 09:24 and only finished when the Mac was woken at 09:34.
    `caffeinate -i` blocks idle sleep only (closing the lid still sleeps) and `-w` ends it with this
    process. Counted, so overlapping jobs share one caffeinate."""

    def __init__(self):
        self.n, self.proc, self.lock = 0, None, threading.Lock()

    def __enter__(self):
        with self.lock:
            self.n += 1
            if self.n == 1 and (self.proc is None or self.proc.poll() is not None):
                try:
                    self.proc = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except OSError:  # not macOS
                    self.proc = None
        return self

    def run(self, fn, *args):
        with self:
            return fn(*args)

    def __exit__(self, *exc):
        with self.lock:
            self.n -= 1
            if self.n == 0 and self.proc is not None:
                self.proc.terminate()
                self.proc = None


awake = KeepAwake()


def release_models():
    """Free Whisper (~1.5 GB) and pyannote (~0.3 GB + cached GPU memory) before running the LLM:
    gpt-oss-20b needs ~12-13 GB and both won't fit in 16 GB together. They reload on the next job."""
    from mlx_whisper.transcribe import ModelHolder
    ModelHolder.model = ModelHolder.model_path = None
    transcribe._pipeline.cache_clear()
    gc.collect()
    mx.clear_cache()
    import torch
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def safe_path(name):
    p = (RECORDINGS / name).resolve()
    if p.parent != RECORDINGS.resolve() or not p.is_file():
        raise HTTPException(404, "recording not found")
    return p


def speech_chunks(samples):
    """Split audio at pauses into chunks of at most MAX_CHUNK_S seconds of wall time."""
    opts = VadOptions(min_silence_duration_ms=500, max_speech_duration_s=MAX_CHUNK_S, speech_pad_ms=200)
    chunks = []
    for ts in get_speech_timestamps(samples, opts, sampling_rate=SR):
        if chunks and ts["end"] - chunks[-1][0] <= MAX_CHUNK_S * SR:
            chunks[-1][1] = ts["end"]
        else:
            chunks.append([ts["start"], ts["end"]])
    return chunks


def variant(model, task):
    """Saved-transcript key: 'small' for original language, 'small.en' for English translation."""
    return f"{model}.en" if task == "translate" else model


_vocab_cache = {}


def _vocabulary():
    """Whisper's initial prompt: the active workspace's people, clients, projects and corrected words.
    It nudges the decoder toward those spellings; respell (finalize) then fixes what it still mishears.
    Cached on the registry file's mtime, since live sessions call this for every chunk."""
    if not settings.get("transcription").get("prime_with_names", True):
        return None
    ws = labels.active()
    try:
        mtime = (labels.REGISTRY / f"{ws}.json").stat().st_mtime
    except (OSError, TypeError):
        return None
    if _vocab_cache.get("key") != (ws, mtime):
        names = labels.vocabulary(ws)
        _vocab_cache.update(key=(ws, mtime), prompt=f"Names: {names}." if names else None)
    return _vocab_cache["prompt"]


def transcribe_chunk(chunk, offset, idx, model, lang, task):
    """Transcribe one pause-delimited chunk. Returns (language, words for merging, segment events)."""
    r = mlx_whisper.transcribe(
        chunk,
        path_or_hf_repo=MODELS[model],
        language=lang,
        task=task,  # "translate" outputs English whatever the spoken language
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=_vocabulary(),  # the workspace's names, so Whisper leans toward their spelling
    )
    words, segments = [], []
    for seg in r["segments"]:
        seg_words = [{**w, "start": w["start"] + offset, "end": w["end"] + offset, "seg": (idx, seg["id"])}
                     for w in seg.get("words", [])]
        if task == "translate" and seg_words:
            # Translated English words don't line up with the Hindi audio, so give
            # the whole sentence one speaker rather than splitting it word by word.
            seg_words = [{"word": seg["text"], "start": seg_words[0]["start"],
                          "end": seg_words[-1]["end"], "seg": (idx, seg["id"])}]
        words += seg_words
        if seg_words:
            segments.append({"type": "segment", "start": seg_words[0]["start"], "end": seg_words[-1]["end"],
                             "text": seg["text"].strip(), "lang": r["language"]})
    return r["language"], words, segments


def finalize(path, samples, words, langs, model, lang, task, speakers, timings, t_start=None):
    """Identify speakers over the whole recording, attach them to the words, save the transcript.

    t_start: when processing began (file jobs); live sessions omit it and total = transcribe + diarize.
    """
    mx.clear_cache()  # hand Whisper's spare GPU buffers back before pyannote's ~3-4 GB peak
    t0 = time.perf_counter()
    turns = diarize(samples, speakers)
    duration = len(samples) / SR
    lines = [{"speaker": l["speaker"], "start": l["start"], "end": l["end"], "text": l["text"].strip()}
             for l in merge(words, turns)]
    lines, _ = library.drop_repeats(lines)  # Whisper's silence filler: loops and repeated lines
    try:  # spellings you corrected before ("marlo" -> "Marlow"), in the recording's workspace
        lines, _ = labels.respell(lines, labels.of(path.name) or labels.active())
    except Exception:
        pass
    stats = {"duration": duration, **timings, "diarize_s": time.perf_counter() - t0,
             "speakers": len({t[2] for t in turns}), "languages": sorted(langs), "model": model,
             "task": task, "lang": lang or "auto", "created": datetime.now().isoformat(timespec="seconds")}
    stats["total_s"] = time.perf_counter() - t_start if t_start else stats["transcribe_s"] + stats["diarize_s"]
    stats["rtf"] = stats["total_s"] / duration if duration else None
    result = {"recording": path.name, "stats": stats, "lines": lines}
    key = variant(model, task)
    (TRANSCRIPTS / f"{path.name}.{key}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (TRANSCRIPTS / f"{path.name}.{key}.txt").write_text(
        "\n".join(f"[{fmt(l['start'])}] {l['speaker']}: {l['text']}" for l in lines) + "\n")
    return result


def summarise_after(name, emit):
    """Workflow after a recording is transcribed and speaker-labelled: queue it in "To summarise", run the
    LLM, and (summarise_one does this on success) move it to "Summarised". Runs on the job thread.
    Always ends with exactly one summary_queued, summary_done or summary_error event. With llm.auto_pickup
    on it only queues: the picker summarises it as soon as nothing is recording."""
    try:
        library.move([name], "to_summarise")
    except Exception as e:
        return emit({"type": "summary_error", "message": f"Could not queue for summary: {e}"})
    if settings.get("llm").get("auto_pickup", True):
        return emit({"type": "summary_queued", "name": name,
                     "message": "Transcript done. Queued in 'To summarise': it is summarised automatically "
                                "as soon as nothing is recording."})
    status = summarise.check()
    if not status["ok"]:
        return emit({"type": "summary_error",
                     "message": f"Queued in 'To summarise', but the LLM isn't available ({status['message']})"})
    emit({"type": "status", "message": f"Transcript done. Summarising with {status['model']} (1–3 min)…"})
    try:
        release_models()  # make room for the LLM; speech models reload on the next transcription
        # no pausing here: this waits on the MLX thread, which a new recording would need
        notes = summariser.submit(summarise_queued, name, False, False).result()
        if notes is None:
            raise LibraryError("it was summarised by another run")
        reindex_soon()
        emit({"type": "summary_done", "name": name, "title": notes.get("title"),
              "action_items": len(notes.get("action_items") or []), "seconds": notes["_meta"]["seconds"]})
    except Exception as e:
        emit({"type": "summary_error", "message": f"Summary failed, it stays in 'To summarise': {e}"})


def run_job(path, model, lang, speakers, task, emit, extra_stats=None):
    busy.add(path.name)
    end = activity.start()
    try:
        with awake:
            return _run_job(path, model, lang, speakers, task, emit, extra_stats)
    finally:
        end()
        busy.discard(path.name)


def _run_job(path, model, lang, speakers, task, emit, extra_stats):
    t_start = time.perf_counter()
    samples = decode_audio(str(path), sampling_rate=SR)
    duration = len(samples) / SR
    chunks = speech_chunks(samples)
    emit({"type": "start", "duration": duration, "chunks": len(chunks)})

    words, langs = [], set()
    for i, (s, e) in enumerate(chunks):
        language, chunk_words, segments = transcribe_chunk(samples[s:e], s / SR, i, model, lang, task)
        langs.add(language)
        words += chunk_words
        for ev in segments:
            emit(ev)
        emit({"type": "progress", "done": i + 1, "total": len(chunks), "audio_pos": e / SR,
              "elapsed": time.perf_counter() - t_start})
    t_transcribe = time.perf_counter() - t_start

    emit({"type": "status", "message": "Identifying speakers…"})
    result = finalize(path, samples, words, langs, model, lang, task, speakers,
                      {"transcribe_s": t_transcribe, **(extra_stats or {})}, t_start)
    emit({"type": "done", **result})


# Live chunks are shorter than file chunks: text for a sentence can't appear before its chunk closes,
# and fluent speakers rarely pause, so 25 s chunks meant up to 20 s delay (median 7 s) on the debate clip.
LIVE_PAUSE_S = 0.5  # silence after speech that closes a live chunk
LIVE_MAX_CHUNK_S = 10
LIVE_VAD = VadOptions(min_silence_duration_ms=300, max_speech_duration_s=LIVE_MAX_CHUNK_S, speech_pad_ms=200)


@app.websocket("/ws/live")
async def live(ws: WebSocket, model: str = "medium", lang: str = "", task: str = "translate",
               speakers: int = 0, source: str = "mic", prefix: str = "", auto: int = 0):
    end = activity.start()  # summaries wait (and pause between parts) while this records
    try:
        await _live(ws, model, lang, task, speakers, source, prefix, auto, end)
    finally:
        end()


async def _live(ws, model, lang, task, speakers, source, prefix, auto, end):
    """Live session: browser streams 16 kHz mono int16 PCM; we transcribe each chunk once its speaker
    pauses, send the text back, and on "stop" save the audio and add speaker labels to the whole thing."""
    await ws.accept()
    if model not in MODELS or task not in {"transcribe", "translate"}:
        await ws.send_json({"type": "error", "message": "invalid model/output combination"})
        return await ws.close()
    loop = asyncio.get_running_loop()
    lang = lang or None
    pcm = bytearray()  # everything received so far, int16
    committed = 0  # samples already transcribed (or skipped as silence)
    chunk_idx, max_lag = 0, 0.0
    stopped = asyncio.Event()

    async def send(msg):
        try:
            await ws.send_json(msg)
        except Exception:
            pass  # browser went away; keep going so the recording is still saved

    async def close():
        try:
            await ws.close()
        except Exception:
            pass

    async def receive():
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect" or msg.get("text") == "stop":
                return stopped.set()
            if msg.get("bytes"):
                pcm.extend(msg["bytes"])

    def audio(a, b):
        return np.frombuffer(bytes(pcm[a * 2:b * 2]), dtype=np.int16).astype(np.float32) / 32768

    async def process(final):
        """Transcribe every chunk that is complete: speech followed by a pause, or MAX_CHUNK_S of audio."""
        nonlocal committed, chunk_idx, max_lag
        while True:
            total = len(pcm) // 2
            if total - committed < (1 if final else SR):
                return
            x = audio(committed, total)
            ts = await asyncio.to_thread(get_speech_timestamps, x, LIVE_VAD, SR)
            if not ts:  # silence only: skip it, but keep the last 0.5 s in case speech is starting
                committed = total if final else max(committed, total - SR // 2)
                return
            if final:
                cut = len(x)
            elif len(x) - ts[-1]["end"] >= LIVE_PAUSE_S * SR:
                cut = ts[-1]["end"]
            elif len(x) >= LIVE_MAX_CHUNK_S * SR:  # no pause yet: cut at the last short gap we have
                ends = [t["end"] for t in ts[:-1]]
                cut = ends[-1] if ends else LIVE_MAX_CHUNK_S * SR
            else:
                return  # still talking
            start = ts[0]["start"]
            _, _, segments = await loop.run_in_executor(
                worker, transcribe_chunk, x[start:cut], (committed + start) / SR, chunk_idx, model, lang, task)
            chunk_idx += 1
            committed += cut
            recorded = len(pcm) // 2 / SR
            for ev in segments:
                max_lag = max(max_lag, recorded - ev["end"])
                await send({**ev, "lag": recorded - ev["end"]})
            await send({"type": "progress", "recorded": recorded, "transcribed": committed / SR})

    receiver = asyncio.create_task(receive())
    try:
        # Load the model before speech arrives so the first sentence isn't delayed by it.
        await loop.run_in_executor(worker, transcribe_chunk, np.zeros(SR, np.float32), 0, -1, model, lang, task)
        await send({"type": "ready"})
        while not stopped.is_set():
            await process(final=False)
            try:
                await asyncio.wait_for(stopped.wait(), 0.3)
            except asyncio.TimeoutError:
                pass
        await receiver
        await send({"type": "status", "message": "Finishing the last sentence…"})
        await process(final=True)
    except Exception as e:  # keep whatever audio we have; report the failure
        await send({"type": "error", "message": f"{type(e).__name__}: {e}"})
        receiver.cancel()

    if not pcm:
        return await close()
    prefix = re.sub(r"[^\w-]+", "_", prefix).strip("_")[:60] or f"rec-live-{source}"
    stem = f"{prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    name, n = f"{stem}.wav", 1
    while (RECORDINGS / name).exists():
        name, n = f"{stem}-{n}.wav", n + 1
    with wave.open(str(RECORDINGS / name), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR), w.writeframes(bytes(pcm))
    claim(name)
    # The live text came from short 10 s chunks; redo the whole recording the same way as the
    # Transcribe button (25 s chunks, more context) and label speakers, streaming progress back.
    events = asyncio.Queue()

    def emit(ev):
        loop.call_soon_threadsafe(events.put_nowait, ev)

    async def pump(job):  # forward the job's events until it has finished
        while not (job.done() and events.empty()):
            try:
                await send({**await asyncio.wait_for(events.get(), 0.5), "final": True})
            except asyncio.TimeoutError:
                pass

    job = loop.run_in_executor(worker, run_job, RECORDINGS / name, model, lang, speakers or None, task,
                               emit, {"max_lag_s": max_lag})
    await send({"type": "final_start", "name": name})
    await pump(job)
    try:
        job.result()
    except Exception as e:
        await send({"type": "error", "message": f"Final transcript failed ({type(e).__name__}: {e}); "
                                                f"the audio is saved as {name}"})
        return await close()
    end()  # recording and final transcript done: a waiting summary may go on
    if auto:  # keeps going (and saves the summary) even if the browser has gone away
        await pump(loop.run_in_executor(worker, summarise_after, name, emit))
    await close()


@app.middleware("http")
async def keep_index_in_step(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if request.method == "POST" and response.status_code < 400 and (
            path.startswith("/api/library/") or path == "/api/llm/summarise"
            or path == "/api/settings" or path.startswith("/api/registry")  # active workspace, registries
            or (path.startswith("/api/labels/") and path.endswith("/confirm"))):
        reindex_soon()
    return response


@app.on_event("startup")
def index_on_start():
    reindex_soon()
    threading.Thread(target=_picker, daemon=True, name="summary-picker").start()


# ---------- summaries picked up from the queue ----------
_auto = {"running": None, "paused": False, "failed": {}}  # failed: name -> inbox mtime when it failed


def _inbox_mtime(name):
    p = library.LLM_INBOX / f"{name}.md"
    return p.stat().st_mtime if p.is_file() else None


def _wait_for_quiet():
    """Between a summary's LLM calls: while a recording or transcription runs, free the LLM's memory
    for Whisper and wait; the summary goes on from the next part when it is quiet again."""
    if not activity.n:
        return
    _auto["paused"] = True
    print("[llm] paused: a recording or transcription is running")
    try:
        summarise.unload(summarise.config()["model"])
    except Exception:
        pass
    while activity.n:
        time.sleep(5)
    _auto["paused"] = False
    print("[llm] resuming")
    worker.submit(release_models).result()


share_filter.wait = _wait_for_quiet  # index screening uses the LLM too: it waits for recordings


def _picker():
    """Summarise "To summarise" on its own (llm.auto_pickup), oldest first, one at a time, starting only
    while nothing records or transcribes. A meeting whose summary failed is not retried until its
    transcript changes or it is summarised by hand."""
    while True:
        time.sleep(15)
        try:
            if not settings.get("llm").get("auto_pickup", True) or activity.n or _auto["running"]:
                continue
            todo = sorted((n for n in summarise.queued()
                           if n not in busy and _auto["failed"].get(n) != _inbox_mtime(n)),
                          key=lambda n: _inbox_mtime(n) or 0)
            if not todo or not summarise.check()["ok"]:
                continue
            name = todo[0]
            print(f"[summary] picked up {name}")
            try:
                notes = summariser.submit(summarise_queued, name).result()
                if notes is not None:
                    reindex_soon()
                    print(f"[summary] ✓ {name}: {notes['_meta']['seconds']} s")
            except Exception as e:
                _auto["failed"][name] = _inbox_mtime(name)
                print(f"[summary] ✗ {name}: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[summary] picker: {type(e).__name__}: {e}")


MCP_AUDIT = ROOT / "logs" / "mcp" / "mcp-audit.jsonl"  # written by the MCP server's container


@app.get("/api/mcp/audit")
def mcp_audit(limit: int = 200):
    """The MCP server's latest tool calls, newest first: client, tool, arguments, meetings shared."""
    if not MCP_AUDIT.is_file():
        return {"calls": [], "total": 0}
    rows = []
    for line in MCP_AUDIT.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return {"calls": rows[::-1][:max(1, min(limit, 2000))], "total": len(rows)}


@app.get("/api/observe")
def observe_report():
    """The Observability page: what is held back, what MCP clients got, self-checks, alerts."""
    return observe.report()


@app.post("/api/observe/scan")
def observe_scan():
    """Look for restricted texts in the AI clients' session logs on this Mac (Codex, Claude Code)."""
    return observe.scan()


@app.get("/api/index/status")
def index_status():
    return {**index.status(), "last_sync": _sync_state["last"], "syncing": _sync_state["queued"]}


@app.post("/api/index/sync")
def index_sync():
    reindex_soon()
    return {"ok": True}


@app.get("/api/index/search")
def index_search(q: str, workspace: str | None = None, client: str | None = None, project: str | None = None,
                 person: str | None = None, kind: str | None = None, since: str | None = None,
                 until: str | None = None, k: int = 8):
    """Meaning search over confirmed meetings, in the selected workspace unless one is given."""
    return index.search(q, labels.active() if workspace is None else workspace, client, project, person,
                        kind, since, until, min(k, 50))


@app.get("/")
def index_page():
    return FileResponse(ROOT / "ui.html")


@app.get("/api/recordings")
def recordings():
    return [it for it in library.items(labels.active()) if it["folder"] != "trash"]


def claim(name):
    """A new recording joins the workspace selected in the header."""
    library.set_workspace([name], labels.active())


# ---------- library (Manage tab) ----------
def _not_busy(names):
    if clash := busy & set(names):
        raise HTTPException(409, f"{', '.join(sorted(clash))} is being transcribed; try again when it finishes")


@app.get("/api/library")
def library_list():
    items = library.items(labels.active())
    counts = {f: 0 for f in library.FOLDERS}
    for it in items:
        counts[it["folder"]] += 1
    return {"folders": [{"id": f, "label": label, "count": counts[f]} for f, label in library.FOLDERS.items()],
            "items": items, "busy": sorted(busy)}


@app.post("/api/library/move")
def library_move(names: list[str] = Body(), folder: str = Body()):
    _not_busy(names)
    library.move(names, folder)
    return {"ok": True}


@app.get("/api/speakers/{name}")
def speakers_get(name: str):
    """The meeting's voices (talk time, a few long lines) with the names given to them, and the workspace's
    people to pick from."""
    name = safe_path(name).name
    ws = labels.of(name)
    people = sorted(e["name"] for e in (labels.load(ws)["entities"] if ws else [])
                    if e.get("kind") == "person" and e.get("status") != "archived")
    return {"voices": library.speaker_overview(name), "names": library.speaker_names(name), "people": people,
            "your_name": settings.get("recording").get("your_name") or ""}


@app.post("/api/speakers/{name}")
def speakers_set(name: str, names: dict[str, str] = Body(), resummarise: bool = Body(False)):
    """Name the voices. With `resummarise`, the meeting goes back to "To summarise" (the queue picks it up)."""
    name = safe_path(name).name
    _not_busy([name])
    saved = library.set_speaker_names(name, names)
    if resummarise:
        library.move([name], "to_summarise")
    return {"names": saved, "queued": resummarise}


@app.post("/api/library/correct")
def library_correct(name: str = Body(), wrong: str = Body(), right: str = Body(), kind: str = Body(""),
                    everywhere: bool = Body(False)):
    """Fix a misheard word in a meeting (or all of its workspace's) and teach the registry the spelling."""
    names = [it["name"] for it in library.items(labels.of(name))] if everywhere else [name]
    _not_busy(names)
    result = corrections.correct(name, wrong, right, kind, everywhere)
    if result["changed"] and index.config()["enabled"]:
        # a changed transcript doesn't alter what the index compares (labels, summary time): re-embed explicitly
        def refresh(changed=tuple(result["changed"])):
            for n in changed:
                try:
                    index.remove(n)
                except Exception:
                    pass
            _sync()
        indexer.submit(refresh)
    return result


@app.post("/api/library/workspace")
def library_workspace(names: list[str] = Body(), workspace: str = Body()):
    """Move recordings to another workspace ("" = none). Their labels stay until you re-label them."""
    if workspace and not labels.exists(workspace):
        raise HTTPException(400, f"no workspace {workspace!r}")
    library.set_workspace(names, workspace)
    return {"ok": True}


@app.post("/api/library/rename")
def library_rename(name: str = Body(), new_name: str = Body()):
    _not_busy([name])
    return {"name": library.rename(name, new_name)}


@app.post("/api/library/delete")
def library_delete(names: list[str] = Body(), permanent: bool = Body(False)):
    _not_busy(names)
    library.delete(names, permanent)
    return {"ok": True}


@app.post("/api/library/empty-trash")
def library_empty_trash():
    return {"deleted": library.empty_trash()}


@app.get("/api/library/search")
def library_search(q: str = ""):
    return library.search(q, ws=labels.active())


@app.get("/api/library/summary/{name}")
def library_summary(name: str):
    out = library.summary(name)
    try:  # names said at least twice in the meeting that the summary never mentions (evaluate.py, no LLM)
        t = library.best_transcript(name)
        if out.get("data") and t:
            out["missing_names"] = evaluate.name_gaps(name, [l for l in t["lines"] if l.get("text", "").strip()],
                                                      out["data"])
    except Exception:
        pass
    return out


# ---------- LLM summaries ----------
@app.get("/api/llm/status")
def llm_status():
    # a recording being auto-summarised is still in the inbox; don't offer it for a second run
    return {**summarise.check(), "models": summarise.models(),
            "queued": [n for n in summarise.queued() if n not in busy and library.workspace_of(n) == labels.active()],
            "auto": {"on": bool(settings.get("llm").get("auto_pickup", True)), "running": _auto["running"],
                     "paused": _auto["paused"], "failed": sorted(_auto["failed"])}}


# ---------- Settings tab ----------
@app.get("/api/settings")
def settings_get():
    return settings.current()


@app.post("/api/settings")
def settings_set(changes: dict = Body(embed=True)):
    """Save {section: {key: value}}; the summariser model must be one the LLM server has."""
    model = changes.get("llm", {}).get("model")
    if model is not None and model not in summarise.models():
        raise HTTPException(400, f"model {model!r} is not available")
    ws = changes.get("labels", {}).get("workspace")
    if ws:
        labels.load(ws)  # must exist
    return settings.update(changes)


# ---------- labels and workspaces ----------
@app.get("/api/registry")
def registry_list():
    return {"workspaces": labels.workspaces(), "active": labels.active(), "kinds": list(labels.KINDS)}


@app.post("/api/registry")
def registry_create(name: str = Body(embed=True)):
    return labels.create_workspace(name)


@app.get("/api/registry/{ws}")
def registry_get(ws: str):
    return labels.load(ws)


@app.post("/api/registry/{ws}/entity")
def registry_entity(ws: str, entity: dict = Body(), original: str | None = Body(None)):
    return labels.upsert_entity(ws, entity, original)


@app.post("/api/registry/{ws}/entity/delete")
def registry_entity_delete(ws: str, kind: str = Body(), name: str = Body()):
    labels.delete_entity(ws, kind, name)
    return {"ok": True}


@app.post("/api/registry/{ws}/types")
def registry_types(ws: str, types: list[str] = Body(embed=True)):
    return labels.set_types(ws, types)


@app.get("/api/labels/{name}")
def labels_get(name: str):
    return {"labels": labels.get(name), "active": labels.of(name), "workspaces": labels.workspaces()}


@app.post("/api/labels/{name}/suggest")
async def labels_suggest(name: str, workspace: str | None = Body(None, embed=True)):
    """Ask the LLM for labels (no new summary). Runs on the job thread, like summaries."""
    _not_busy([name])
    status = summarise.check()
    if not status["ok"]:
        raise HTTPException(503, status["message"])
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(worker, release_models)
    busy.add(name)
    try:
        return await loop.run_in_executor(worker, awake.run, summarise.label_one, name, workspace)
    finally:
        busy.discard(name)


@app.post("/api/labels/{name}/confirm")
def labels_confirm(name: str, labels_: dict = Body(alias="labels"), learn: list[dict] = Body([])):
    return labels.confirm(name, labels_, learn)


@app.get("/api/llm/models")
def llm_models():
    """Chat models with their context sizes, for the Settings tab's model picker."""
    return {"models": summarise.model_details(), "current": summarise.config()["model"],
            "needed_ctx": summarise.SUMMARY_CTX}


@app.post("/api/llm/create-32k")
def llm_create_32k(model: str = Body(embed=True)):
    """Make a 32k-context copy of a model (same weights, nothing downloaded)."""
    return {"model": summarise.create_32k(model)}


@app.get("/api/llm/compare/{name}")
def llm_comparisons(name: str):
    return summarise.comparisons(name)


@app.post("/api/llm/compare")
async def llm_compare(name: str = Body(), models: list[str] = Body()):
    """Summarise one recording with each model in turn, into llm/compare/ (the real summary is untouched).
    Each model runs alone, after the others are unloaded, so its memory figure is its own."""
    available = summarise.models()
    if missing := [m for m in models if m not in available]:
        raise HTTPException(400, f"not available: {', '.join(missing)}")
    _not_busy([name])
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(worker, release_models)
    results = []
    for model in models:
        try:
            await loop.run_in_executor(worker, summarise.unload_all)
            notes = await loop.run_in_executor(worker, awake.run, summarise.compare_one, name, model)
            results.append({"model": model, "ok": True, "seconds": notes["_meta"]["seconds"]})
        except Exception as e:
            results.append({"model": model, "ok": False, "error": f"{type(e).__name__}: {e}"})
    await loop.run_in_executor(worker, summarise.unload_all)
    return {"results": results}


@app.post("/api/llm/summarise")
async def llm_summarise(names: list[str] = Body(embed=True)):
    """Summarise queued recordings one by one on the job thread (so it never overlaps a transcription),
    after unloading the speech models to make room for the LLM."""
    status = summarise.check()
    if not status["ok"]:
        raise HTTPException(503, status["message"])
    loop = asyncio.get_running_loop()
    results = []
    for name in names:
        try:
            notes = await loop.run_in_executor(summariser, summarise_queued, name)
            if notes is None:  # an auto-summary got to it first
                results.append({"name": name, "ok": True, "skipped": "already summarised"})
            else:
                results.append({"name": name, "ok": True, "action_items": len(notes.get("action_items", []))})
        except Exception as e:
            results.append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
    return {"results": results}


def summarise_queued(name, release=True, pause=True):
    """One summary, on the summariser thread (manual, picked up, or after a recording). Returns None when
    the recording was summarised while this waited its turn (it's no longer queued, but its summary
    exists). release: free Whisper first (via the MLX thread; False when the caller is on it).
    pause: wait while something records, before starting and between parts."""
    library.audio_path(name)  # validates the name before it's used in a path
    if name not in summarise.queued() and (library.LLM_SUMMARIES / f"{name}.json").is_file():
        return None
    busy.add(name)
    _auto["running"] = name
    try:
        if pause:
            _wait_for_quiet()
        if release and not activity.n:
            worker.submit(release_models).result()
        _auto["failed"].pop(name, None)
        return awake.run(summarise.summarise_one, name, None, _wait_for_quiet if pause else None,
                         lambda m: print(f"[summary] {m}"))
    finally:
        _auto["running"] = None
        busy.discard(name)


@app.post("/api/upload")
async def upload(file: UploadFile):
    stem, ext = Path(file.filename or "audio").stem, Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXT:
        raise HTTPException(400, f"unsupported file type {ext!r}")
    stem = re.sub(r"[^\w.-]+", "_", stem)[:80] or "audio"
    dest = RECORDINGS / f"{stem}{ext}"
    n = 1
    while dest.exists():
        dest, n = RECORDINGS / f"{stem}-{n}{ext}", n + 1
    dest.write_bytes(await file.read())
    claim(dest.name)
    return {"name": dest.name}


@app.post("/api/youtube")
def youtube(url: str = Body(embed=True)):
    """Download a video's audio track (no ffmpeg needed: native m4a/webm audio streams)."""
    from yt_dlp import YoutubeDL

    if not re.match(r"https?://(www\.|m\.)?(youtube\.com|youtu\.be)/", url):
        raise HTTPException(400, "not a YouTube link")
    opts = {"format": "bestaudio[ext=m4a]/bestaudio", "noplaylist": True, "quiet": True, "no_warnings": True,
            "outtmpl": str(RECORDINGS / "yt-%(title).60B-%(id)s.%(ext)s"), "restrictfilenames": True}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as e:
        raise HTTPException(400, f"download failed: {e}")
    claim(path.name)
    return {"name": path.name, "title": info.get("title"), "duration": info.get("duration")}


@app.get("/audio/{name}")
def audio(name: str):
    return FileResponse(safe_path(name))


@app.get("/api/transcript/{name}")
def transcript(name: str, model: str = "medium", task: str = "translate"):
    name = safe_path(name).name
    p = TRANSCRIPTS / f"{name}.{variant(model, task)}.json"
    if not p.is_file():
        raise HTTPException(404, "no transcript yet")
    t = json.loads(p.read_text())
    t["lines"] = library.name_speakers(t.get("lines") or [], library.speaker_names(name))  # names you gave the voices
    return t


@app.get("/api/transcribe/{name}")
def transcribe_stream(name: str, model: str = "medium", lang: str = "", speakers: int = 0, task: str = "translate",
                      auto: int = 0):
    path = safe_path(name)
    if model not in MODELS:
        raise HTTPException(400, "unknown model")
    if task not in {"transcribe", "translate"}:
        raise HTTPException(400, "unknown task")
    q = queue.Queue()

    def job():
        try:
            run_job(path, model, lang or None, speakers or None, task, q.put)
        except Exception as e:  # surface failures in the UI instead of a silent hang
            q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        else:
            if auto:
                summarise_after(path.name, q.put)
        finally:
            q.put(None)

    worker.submit(job)

    def events():
        yield f"data: {json.dumps({'type': 'queued'})}\n\n"
        while (ev := q.get()) is not None:
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def fetch_models():
    """Download Whisper on first start (public, no token), then use the local copy so later runs are offline."""
    from huggingface_hub import snapshot_download
    for name, repo in MODELS.items():
        try:
            MODELS[name] = snapshot_download(repo, local_files_only=True)
        except Exception:
            print(f"First run: downloading Whisper {name} (~1.5 GB, once)...", flush=True)
            MODELS[name] = snapshot_download(repo)


if __name__ == "__main__":
    import uvicorn

    fetch_models()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8001")))
