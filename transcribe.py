"""Meeting transcript with speaker labels: mlx-whisper (text) + pyannote (who spoke when).

Usage: uv run transcribe.py <audio> [--lang hi] [--speakers 3]   (original-language output; the UI also translates)
The token is only needed for the first run (model download); after that it runs offline.
"""
import argparse
import functools
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from faster_whisper.audio import decode_audio  # PyAV decode, avoids needing ffmpeg

SR = 16000
# Shipped in models/ (CC-BY-4.0) so no Hugging Face account is needed; falls back to the gated hub repo.
_LOCAL_DIARIZATION = Path(__file__).parent / "models" / "pyannote-speaker-diarization-community-1"
DIARIZATION_MODEL = str(_LOCAL_DIARIZATION) if _LOCAL_DIARIZATION.is_dir() else "pyannote/speaker-diarization-community-1"


def transcribe(samples, model, lang):
    import mlx_whisper

    r = mlx_whisper.transcribe(
        samples,
        path_or_hf_repo=f"mlx-community/whisper-{model}-mlx",
        language=lang,
        word_timestamps=True,  # needed to split a segment that spans two speakers
        condition_on_previous_text=False,  # stops errors snowballing on long or repetitive audio
    )
    # Tag each word with its Whisper segment so lines also break at sentence boundaries.
    words = [{**w, "seg": i} for i, s in enumerate(r["segments"]) for w in s.get("words", [])]
    return r["language"], words


MIN_TURN_S = 2.0  # only turns this long define speakers; shorter ones ("OK", "Yeah") join the nearest speaker
SAME_SPEAKER_DIST = 0.7  # cosine distance; 0.6-0.8 gave identical results on our test clips
MIN_SPEAKER_SHARE = 0.03  # a speaker needs 3% of all speech, else folded into the nearest (27-min meeting: 10 -> 2)


@functools.lru_cache(maxsize=1)
def _pipeline():
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=os.environ.get("HF_TOKEN"))
    if torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    # Slide pyannote's 10 s analysis window by 5 s instead of 1 s. Its per-window embeddings were
    # ~95% of diarization time (we redo the grouping anyway): 27-min meeting 157 s -> 35 s, same
    # speaker counts on all test clips. Larger embedding batches (128) were 3x slower on 16 GB.
    pipeline._segmentation.step = 0.5 * pipeline._segmentation.duration
    # Batch 16 instead of 32: peak GPU memory +3.2 GB instead of +4.4 GB, ~2 s slower on a 27-min meeting.
    pipeline.embedding_batch_size = 16
    return pipeline


def diarize(samples, num_speakers):
    """Speaker turns: pyannote finds when someone is speaking, then we regroup the turns by voice.

    num_speakers is a minimum, not a cap: voices are grouped by distance as when it is 0, and only if that
    finds fewer than num_speakers are the turns split into exactly that many. A third or fourth person
    in a call with the default of 2 is still found.

    pyannote's own clustering merged distinct speakers on every test clip (e.g. a moderator into a
    participant, two men into one), while its per-turn voice embeddings told them apart clearly.
    So we keep its turns and redo only the grouping: agglomerative clustering on per-turn embeddings.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    pipeline = _pipeline()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def embed(start, end):
        # At least 1 s of audio, kept inside the recording: pyannote can report a turn that starts at
        # or past the end of the audio (its windows are padded), which gave an empty clip and
        # "AssertionError: choose a window size 400 that is [2, 0]" from the embedding's fbank.
        n = len(samples)
        s = max(0, min(int(start * SR), n - SR))
        clip = samples[s:min(max(int(end * SR), s + SR), n)]
        v = np.asarray(pipeline._embedding(torch.from_numpy(clip)[None, None].to(device))).reshape(-1)
        return v / np.linalg.norm(v)

    # Pass audio in memory so pyannote doesn't need ffmpeg/torchcodec to read the file.
    audio = {"waveform": torch.from_numpy(samples).unsqueeze(0), "sample_rate": SR}
    out = pipeline(audio)
    annotation = getattr(out, "speaker_diarization", out)  # 4.x wraps the Annotation
    turns = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]
    if torch.backends.mps.is_available():
        # pyannote's batches peak at ~3-4 GB of GPU memory; PyTorch otherwise keeps it reserved
        # after the job (measured: 5.5 GB held vs 1.1 GB after release).
        torch.mps.empty_cache()

    long_idx = [i for i, (s, e, _) in enumerate(turns) if e - s >= MIN_TURN_S]
    if len(long_idx) < 2:
        return turns  # too little speech to regroup; keep pyannote's labels
    E = np.array([embed(s, e) for s, e, _ in turns])
    Z = linkage(E[long_idx], "average", metric="cosine")
    labels = fcluster(Z, SAME_SPEAKER_DIST, "distance")
    if num_speakers and len(set(labels)) < num_speakers:
        labels = fcluster(Z, num_speakers, "maxclust")
    centroids = {c: E[long_idx][labels == c].mean(0) for c in set(labels)}
    durations = np.array([e - s for s, e, _ in turns])

    def assign():
        return np.array([max(centroids, key=lambda c: v @ centroids[c]) for v in E])

    speaker = assign()
    # Fold away speakers with too little speech: over a long recording a few odd-sounding
    # one-word turns otherwise each become a "speaker". Never below the minimum that was given.
    while len(centroids) > max(1, num_speakers or 1):
        totals = {c: durations[speaker == c].sum() for c in centroids}
        smallest = min(totals, key=totals.get)
        if totals[smallest] >= MIN_SPEAKER_SHARE * durations.sum():
            break
        del centroids[smallest]
        speaker = assign()

    # Number speakers in order of first appearance: SPEAKER_00 speaks first.
    order = {c: n for n, c in enumerate(dict.fromkeys(speaker[i] for i in range(len(turns))))}
    return [(s, e, f"SPEAKER_{order[speaker[i]]:02d}") for i, (s, e, _) in enumerate(turns)]


def speaker_for(start, end, turns):
    """Speaker whose turns overlap [start, end] the most; nearest turn if none overlap."""
    overlap = {}
    for ts, te, spk in turns:
        o = min(end, te) - max(start, ts)
        if o > 0:
            overlap[spk] = overlap.get(spk, 0) + o
    if overlap:
        return max(overlap, key=overlap.get)
    mid = (start + end) / 2
    return min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])))[2] if turns else "SPEAKER_?"


def merge(words, turns):
    """Group consecutive words by the same speaker and Whisper segment into utterances."""
    lines = []
    for w in words:
        spk = speaker_for(w["start"], w["end"], turns)
        if lines and lines[-1]["speaker"] == spk and lines[-1]["seg"] == w["seg"]:
            lines[-1]["end"] = w["end"]
            lines[-1]["text"] += w["word"]
        else:
            lines.append({"speaker": spk, "seg": w["seg"], "start": w["start"], "end": w["end"], "text": w["word"]})
    return lines


def fmt(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--model", default="medium")
    p.add_argument("--lang", default=None, help="force language, e.g. hi or en")
    p.add_argument("--speakers", type=int, default=None, help="at least this many speakers (more are still detected)")
    a = p.parse_args()

    samples = decode_audio(a.audio, sampling_rate=SR)
    print(f"audio: {len(samples) / SR:.1f}s")

    t0 = time.perf_counter()
    lang, words = transcribe(samples, a.model, a.lang)
    print(f"transcribe ({a.model}, lang={lang}): {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    turns = diarize(samples, a.speakers)
    print(f"diarize ({len({t[2] for t in turns})} speakers): {time.perf_counter() - t0:.1f}s")

    lines = merge(words, turns)
    out = Path("out") / f"{Path(a.audio).stem}.diarized.{a.model}.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(f"[{fmt(l['start'])}] {l['speaker']}: {l['text'].strip()}" for l in lines) + "\n")
    print(f"transcript -> {out}")


if __name__ == "__main__":
    main()
