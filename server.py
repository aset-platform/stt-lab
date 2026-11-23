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

import library
import settings
import summarise
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


@app.exception_handler(LibraryError)
@app.exception_handler(settings.SettingsError)
def library_error(_, e):
    return JSONResponse({"detail": str(e)}, status_code=400)


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


def transcribe_chunk(chunk, offset, idx, model, lang, task):
    """Transcribe one pause-delimited chunk. Returns (language, words for merging, segment events)."""
    r = mlx_whisper.transcribe(
        chunk,
        path_or_hf_repo=MODELS[model],
        language=lang,
        task=task,  # "translate" outputs English whatever the spoken language
        word_timestamps=True,
        condition_on_previous_text=False,
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
    Always ends with exactly one summary_done or summary_error event."""
    try:
        library.move([name], "to_summarise")
    except Exception as e:
        return emit({"type": "summary_error", "message": f"Could not queue for summary: {e}"})
    status = summarise.check()
    if not status["ok"]:
        return emit({"type": "summary_error",
                     "message": f"Queued in 'To summarise', but the LLM isn't available ({status['message']})"})
    emit({"type": "status", "message": f"Transcript done. Summarising with {status['model']} (1–3 min)…"})
    busy.add(name)
    try:
        release_models()  # make room for the LLM; speech models reload on the next transcription
        notes = summarise.summarise_one(name)
        emit({"type": "summary_done", "name": name, "title": notes.get("title"),
              "action_items": len(notes.get("action_items") or []), "seconds": notes["_meta"]["seconds"]})
    except Exception as e:
        emit({"type": "summary_error", "message": f"Summary failed, it stays in 'To summarise': {e}"})
    finally:
        busy.discard(name)


def run_job(path, model, lang, speakers, task, emit, extra_stats=None):
    busy.add(path.name)
    try:
        return _run_job(path, model, lang, speakers, task, emit, extra_stats)
    finally:
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
    if auto:  # keeps going (and saves the summary) even if the browser has gone away
        await pump(loop.run_in_executor(worker, summarise_after, name, emit))
    await close()


@app.get("/")
def index():
    return FileResponse(ROOT / "ui.html")


@app.get("/api/recordings")
def recordings():
    return [it for it in library.items() if it["folder"] != "trash"]


# ---------- library (Manage tab) ----------
def _not_busy(names):
    if clash := busy & set(names):
        raise HTTPException(409, f"{', '.join(sorted(clash))} is being transcribed; try again when it finishes")


@app.get("/api/library")
def library_list():
    items = library.items()
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
    return library.search(q)


@app.get("/api/library/summary/{name}")
def library_summary(name: str):
    return library.summary(name)


# ---------- LLM summaries ----------
@app.get("/api/llm/status")
def llm_status():
    # a recording being auto-summarised is still in the inbox; don't offer it for a second run
    return {**summarise.check(), "models": summarise.models(),
            "queued": [n for n in summarise.queued() if n not in busy]}


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
    return settings.update(changes)


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
            notes = await loop.run_in_executor(worker, summarise.compare_one, name, model)
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
    await loop.run_in_executor(worker, release_models)
    results = []
    for name in names:
        try:
            notes = await loop.run_in_executor(worker, summarise_queued, name)
            if notes is None:  # an auto-summary got to it first
                results.append({"name": name, "ok": True, "skipped": "already summarised"})
            else:
                results.append({"name": name, "ok": True, "action_items": len(notes.get("action_items", []))})
        except Exception as e:
            results.append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
    return {"results": results}


def summarise_queued(name):
    """Job-thread body of a manual summary. Returns None when the recording was summarised while this
    request waited its turn (it's no longer queued, but its summary exists)."""
    library.audio_path(name)  # validates the name before it's used in a path
    if name not in summarise.queued() and (library.LLM_SUMMARIES / f"{name}.json").is_file():
        return None
    busy.add(name)
    try:
        return summarise.summarise_one(name)
    finally:
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
    return {"name": path.name, "title": info.get("title"), "duration": info.get("duration")}


@app.get("/audio/{name}")
def audio(name: str):
    return FileResponse(safe_path(name))


@app.get("/api/transcript/{name}")
def transcript(name: str, model: str = "medium", task: str = "translate"):
    p = TRANSCRIPTS / f"{safe_path(name).name}.{variant(model, task)}.json"
    if not p.is_file():
        raise HTTPException(404, "no transcript yet")
    return json.loads(p.read_text())


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
