# Meeting Transcribe Lab

Local speech-to-text for meetings on an Apple Silicon Mac: English or Hindi/Hinglish speech in,
**English transcript with speaker labels** out, organised in folders and handed to a local LLM
(gpt-oss-20b) for summaries and action items. Upload a file, fetch a YouTube video, or record a browser
tab / microphone — live, with text appearing while people talk. Nothing leaves the machine except the
one-time model download (and YouTube downloads, if you use that).

Built and measured on an **Apple M4, 16 GB RAM, macOS**.

## Final stack

| Job | Model / library | Runs on |
|---|---|---|
| Speech → English text | **Whisper medium** via [`mlx-whisper`](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (`mlx-community/whisper-medium-mlx`), `task="translate"` | Apple GPU (MLX) |
| When is someone speaking | **pyannote `speaker-diarization-community-1`** (segmentation + voice embeddings) | Apple GPU (PyTorch MPS) |
| Who is speaking | Our own grouping of pyannote's per-turn voice embeddings (see [Speaker labels](#speaker-labels)) | CPU, negligible |
| Cutting audio at pauses | Silero VAD, bundled with `faster-whisper` | CPU |
| Decoding mp3/m4a/webm | PyAV, bundled with `faster-whisper` — **no ffmpeg needed** | CPU |

Only these models are loaded by the app. Whisper medium is the only Whisper option (the server rejects
others); it loads on the first job and stays loaded. `faster-whisper` is used only for audio decoding and
VAD, not for transcription.

Why these choices (all measured, see [Evaluation](#evaluation-results)):
- **mlx-whisper, not faster-whisper**: faster-whisper has no Apple-GPU support, so on a Mac it runs on CPU.
  On the same audio mlx-whisper was 3.6× faster (0.16× vs 0.57× real time, `small`).
- **medium, not small**: translates Hinglish noticeably better ("but suits you very well" vs small's
  "but you are very stubborn") for little extra time, and still keeps up live.
- **Not large-v3-turbo**: it was trained without translation data, so it can't produce English from Hindi.
- **community-1, not 3.1**: identical results on every test clip; community-1 is the newest open model.
  (`speaker-diarization-precision` is newer but runs only on pyannoteAI's cloud.)

## Setup

Requirements: **Apple Silicon Mac** (M1 or newer; MLX does not run on Intel Macs), 16 GB RAM recommended
(24 GB+ is more comfortable with the LLM), [`uv`](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), Chrome or Edge (for tab recording), and
[Ollama](https://ollama.com/download) for summaries. No Hugging Face account, token or ffmpeg is needed.

### 1. Clone and install

```bash
git clone git@github.com:aset-platform/stt-lab.git    # private repo: needs access to the aset-platform org
cd stt-lab
uv sync                                  # Python 3.11 + all dependencies into .venv (~2 GB, a few minutes)
```

### 2. Enable the pre-commit checks (once per clone)

```bash
git config core.hooksPath .githooks
```

### 3. Summariser (Ollama + gpt-oss-20b)

```bash
ollama pull gpt-oss:20b                  # ~13 GB, once
uv run server.py                         # then Settings tab → Summariser: "Create 32k copy", then "Use"
uv run summarise.py --check              # should print: ... ready
```

Any chat model in Ollama works (e.g. `qwen3.5:9b`). The Settings tab lists them with their context size and
makes the 32k copy for you (same weights, no extra download). To test models against each other, open a
recording in Manage → **Compare models**: each runs alone and you see the summaries side by side with time,
speed and memory.

The 32k-context copy matters: Ollama's default 4096-token context silently cuts any meeting longer than
~10 minutes (see [Summaries](#summaries-and-action-items-gpt-oss-20b)). Skip this step if you only need
transcripts; recordings then wait in *To summarise* until an LLM is available.

### 4. Run

```bash
uv run server.py                         # then open http://127.0.0.1:8001
PORT=9000 uv run server.py               # if 8001 is taken
```

The first start downloads **Whisper medium** (~1.5 GB, public) into `~/.cache/huggingface`
(`First run: downloading Whisper medium...`); later starts need no internet. **pyannote community-1**
(speaker labels, 31 MB) ships in `models/` (CC-BY-4.0, see its `ATTRIBUTION.md`), so the gated
Hugging Face download is not needed.

Two **sample recordings** with saved transcripts appear in the Recordings list on first start, so you can try
the UI straight away: `sample-english-meeting.wav` (12 s, two speakers) and `sample-hinglish-standup.wav`
(35 s, Hindi + English, translated to English). Both are synthetic (macOS text-to-speech).

The server listens on localhost only. Stop it with Ctrl+C.

## Using the UI

**Add audio** (left panel):
- **Drop a file** (mp3, wav, m4a, webm, …): it plays and transcribes immediately.
- **Paste a YouTube link → Fetch**: downloads the audio track only, then plays and transcribes.
- **Record**: pick a **Source**, keep **Live transcript** ticked, click **● Record**, then **Stop**.
  - *Browser tab* — sound straight from a Chrome tab (YouTube, Google Meet / Teams on the web).
    Clean digital audio; best for testing.
  - *Microphone* — the room, for in-person meetings. Don't use it for a video playing on the same
    laptop: speaker → room → mic audio ruins speaker labelling (see results below).

**Tab recording in Chrome:** open the video tab first → Record → in the share dialog choose the
**"Chrome Tab"** pane (on macOS, "Window" and "Entire Screen" share no audio) → pick the tab → keep
**"Also share tab audio"** on → Share → press play. The green level bar must move; if it stays flat,
no audio is being captured.

**Settings** (above the transcript):
- **Spoken**: `auto`, English or Hindi. Auto detects per chunk; force Hindi if detection flips.
- **Output**: `English` (Whisper translates as it transcribes) or `As spoken` (Hindi comes out in
  Devanagari).
- **Speakers**: `0` = detect automatically; enter the count if you know it — it gives exactly that many.

**Transcript**: click a line to play from there; the line being played is highlighted.
Saved recordings and their transcripts reload when you select them in **Recordings**.

## Managing recordings (Manage tab)

Every recording is in one folder: **Inbox** (new recordings land here), **To summarise**, **Summarised**,
**Archive** or **Trash**. Folders are labels stored in `library.json`; the audio itself always stays in
`recordings/`, so moving never breaks playback.

- **Rename** renames the audio and everything derived from it (transcripts, LLM files); the extension is kept.
- **Move** a row with its menu, or tick several rows and use *Move selected to…*.
- **Delete** sends to Trash (hidden from the Transcribe tab). From Trash: *Restore*, *Delete permanently*,
  or *Empty trash*, which removes the audio, transcripts and summaries for good.
- **Open** jumps to the Transcribe tab with that recording loaded.
- Click a row to see its details and, once summarised, its summary, decisions, action items and open questions.
- Rename/move/delete are refused while that recording is being transcribed.

## Summaries and action items (gpt-oss-20b)

**Automatic workflow:** with **Auto-summarise** ticked (next to Live transcript; on by default), every recording
goes straight through: transcribe → label speakers → *To summarise* → LLM summary → *Summarised*. Non-live
recordings are transcribed automatically too. If the LLM isn't running or the summary fails, the recording
stays in *To summarise* and the reason is shown. **Copy transcript** (Transcribe tab) and **Copy summary**
(Manage tab, Markdown) put the text on the clipboard.

Moving a recording to **To summarise** writes its transcript (English if available) with a metadata header
to `llm/inbox/<recording>.md`. The summariser reads from there:

```
llm/inbox/       transcripts waiting for the LLM
llm/summaries/   <recording>.md (readable) and .json (title, summary, decisions, action_items, open_questions)
llm/processed/   transcripts that have been summarised
```

`summarise.py` talks to any OpenAI-compatible server. Default: Ollama on `http://127.0.0.1:11434/v1` with
model `gpt-oss:20b`. Pick the model in the **Settings** tab; other LLM settings are the `llm` section of
`settings.json`, e.g. for LM Studio: `{"llm": {"base_url": "http://127.0.0.1:1234/v1", "model": "openai/gpt-oss-20b"}}`.
By default the model is unloaded as soon as each summary is written (`unload_after_summary`), so its memory
is free again for recording and transcription.

**Context length (Ollama):** Ollama loads models with a 4096-token context by default and silently cuts longer
input, which is too small for meetings over ~10 min. Create a 32k-context copy and use it: Settings tab →
**Create 32k copy**, or by hand:

```bash
printf 'FROM gpt-oss:20b\nPARAMETER num_ctx 32768\n' > Modelfile && ollama create gpt-oss-20b-32k -f Modelfile
```

If the input or reply is still cut off, the summariser reports it as an error and the transcript stays queued.

The instructions given to the model are in `prompts/meeting_summary.md`; edit them to change the output.

Once gpt-oss is installed (e.g. `ollama pull gpt-oss:20b`), either click **Summarise queued** in the Manage
tab, or run:

```bash
uv run summarise.py --check     # is the LLM reachable and the model available?
uv run summarise.py             # summarise everything in llm/inbox/
```

Until then the Manage tab shows *LLM not connected* and queued transcripts simply wait.

**Memory:** gpt-oss-20b needs ~12–13 GB, so it can't run alongside the speech models on 16 GB. Before
summarising from the UI, the server unloads Whisper and pyannote (they reload on the next transcription),
and summarising runs on the same job queue as transcription so the two never overlap. When using
`summarise.py` directly, stop the server first.

## How it works

### File mode (upload, YouTube, Transcribe button)
1. Decode to 16 kHz mono.
2. Silero VAD finds speech; audio is cut at pauses into chunks of ≤ 25 s (one Whisper window each).
3. Each chunk is transcribed with Whisper medium; lines stream to the page as each chunk finishes.
4. Speaker labelling runs over the whole recording, each line gets a speaker, and the result is saved.

### Live mode (Record with Live transcript)
1. The page captures audio, resamples to 16 kHz in an AudioWorklet and streams 16-bit PCM to the server
   over a WebSocket every ~250 ms.
2. The server closes a chunk when the speaker pauses for 0.5 s, or after 10 s without a pause, and sends
   the text back. Lines show without speakers (speaker labelling needs the whole recording).
3. On **Stop**: the audio is saved as `recordings/rec-live-<source>-<time>.wav`, then **re-transcribed
   at full quality** exactly like file mode (25 s chunks give Whisper more context than the 10 s live
   ones), speakers are labelled, and the result replaces the live lines.

### Speaker labels
pyannote's own speaker clustering merged distinct people on every test clip (a moderator into a
participant; two men into one), even though its voice embeddings told them apart clearly (same voice
≈ 0.9 cosine similarity, different voices ≤ 0.33). So we keep pyannote's speech turns and redo only the
grouping (`diarize()` in `transcribe.py`):
1. pyannote finds speech turns.
2. Each turn gets a voice embedding; turns ≥ 2 s are grouped by average-linkage clustering
   (cosine distance 0.7), or into exactly N groups if you gave a speaker count.
3. Shorter turns ("OK", "Yeah") join the nearest speaker.
4. A speaker with < 3% of all speech is folded into the nearest one. Without this, a 27-min two-person
   meeting produced 10 "speakers" made of stray one-word turns.

When the output is English, each **sentence** gets one speaker: translated English words don't line up
in time with the Hindi audio, so word-level assignment split sentences between speakers.

## Tuned settings

| Setting | Value | Where | Why |
|---|---|---|---|
| Whisper `condition_on_previous_text` | `False` | `server.py` | Stops errors snowballing on long/repetitive audio |
| File chunk length | ≤ 25 s | `MAX_CHUNK_S` | One Whisper window per chunk |
| Live chunk: pause / max length | 0.5 s / 10 s | `LIVE_PAUSE_S`, `LIVE_MAX_CHUNK_S` | 25 s chunks gave up to 20 s live delay on fluent speakers |
| Speaker grouping distance | 0.7 | `SAME_SPEAKER_DIST` | 0.6–0.8 gave identical results on all test clips |
| Min turn length to define a speaker | 2 s | `MIN_TURN_S` | Short turns give unreliable embeddings |
| Min share of speech per speaker | 3% | `MIN_SPEAKER_SHARE` | Removes fake speakers in long meetings |
| pyannote window step | 5 s (was 1 s) | `_pipeline()` | Diarization 157 s → 35 s on a 27-min meeting, same results |
| pyannote embedding batch | 16 (was 32) | `_pipeline()` | −1.2 GB peak memory for ~2 s |
| MLX buffer cache limit | 512 MB | `server.py` | Without it Whisper grew to 8.6 GB over a 27-min meeting |
| Release PyTorch GPU cache after diarization | yes | `diarize()` | Otherwise 3–5 GB stayed reserved after every job |

## Performance (M4, 16 GB)

| Measurement | Result |
|---|---|
| 27-min meeting, full job (transcribe + speakers) | ~210 s (≈ 0.13× real time) |
| — of which speaker labelling | ~35–37 s |
| Live mode: delay of text behind speech | median 4.1 s, worst 8.7 s (1-min debate clip) |
| Live mode: after Stop (24 s session) | final transcript in 9.3 s |
| Live mode: after Stop, long meeting | ≈ the file-mode time for that length (estimate, not measured) |

## Memory (measured on the 27-min meeting)

| Component | Memory |
|---|---|
| Python + PyTorch + MLX + pyannote code | ~0.25 GB |
| Meeting audio in memory | ~0.3 GB per 27 min (~0.6 GB/hour) |
| Silero VAD | ~15 MB |
| Whisper medium | ~1.5 GB loaded, ~3 GB while transcribing |
| pyannote community-1 | ~0.3 GB loaded, +~3.2 GB while labelling speakers |
| Live audio buffer | ~0.1 GB/hour |

| Phase | Process memory |
|---|---|
| Server started, nothing loaded | 0.26 GB |
| Transcribing | 3.4 GB |
| Labelling speakers (peak) | **5.8 GB** |
| Idle after a job (models stay loaded) | 3.7 GB |

On Apple Silicon the GPU shares RAM, so these figures include GPU memory. **Plan for ~6 GB free** plus
the browser. A local LLM such as gpt-oss-20b (~12–13 GB) will not fit alongside this on 16 GB — run them
one after the other. Stop the server between meetings to free the 3.7 GB the loaded models hold.

## Evaluation results

Speaker counts with the final setup:

| Clip | Correct | Result |
|---|---|---|
| Real meeting, 27 min, 2 people (mic) | 2 | 2 ✓ (pyannote alone: 1; before the 3% rule: 10) |
| Mock group discussion (YouTube), first minute | 3 | 3 ✓ (pyannote alone: 2) |
| 3-person dialogue, first 2 min (mic) | 3 | 3 ✓, lines alternate correctly (pyannote alone merged two men) |
| Two synthetic voices | 2 | 2 ✓ |
| Mean Girls scene, 4 people, 1–2 s lines | 4 | ✗ ~51–57% of lines right (labelling every line as one person scores 54%) |
| Same 3-person dialogue, full 5:42 | 3? | 4 — unverified whether a 4th person appears later |

Transcription: Whisper medium with English output produced clean, natural English from Hindi and mixed
speech; on clean YouTube audio it was essentially word-perfect.

## Known limitations

- **Rapid short exchanges** (1–2 s lines) can't be labelled reliably — voice embeddings need a few
  seconds of speech. Normal meeting turns work.
- **Quiet participants** with < 3% of total speech are merged into another speaker (≈ 1.8 min in a
  60-min meeting). Enter the speaker count to avoid this.
- **Similar voices through a laptop speaker + mic** lose the detail speaker labelling depends on. Real
  in-person meetings (people speaking directly into the room) and tab capture don't have this problem.
- **Live text lags** 4–9 s and has no speaker labels until Stop. The final transcript replaces it.
- **Whisper can invent text in silence** (e.g. "Thanks for watching!").
- **Tab recording** works only in Chrome/Edge, and only from the "Chrome Tab" pane on macOS.

## Files

| Path | What |
|---|---|
| `server.py` | FastAPI server: UI, uploads, YouTube fetch, file transcription (SSE), live sessions (WebSocket) |
| `transcribe.py` | Speaker labelling (`diarize`), word/speaker merge, and a CLI: `uv run transcribe.py audio.mp3 [--speakers N]` (original-language output) |
| `ui.html` | The single-page UI |
| `recordings/` | Uploaded, fetched and recorded audio (git-ignored, except the `sample-*` files) |
| `out/ui/` | Saved transcripts, `<recording>.medium.en.json/.txt` (git-ignored, except `sample-*`) |
| `scripts/repo_guard.py`, `.githooks/pre-commit`, `.github/workflows/repo-guard.yml` | Pre-commit and CI checks: audio only as `recordings/sample-*`, at most 10 minutes |
| `settings.json` | All settings, one section per feature, edited in the Settings tab; defaults in `settings.py` (git-ignored; replaces the older `llm.json`, which is moved in automatically) |
| `library.py` | Folders, rename, delete, and writing transcripts to `llm/inbox/` |
| `summarise.py` | LLM summariser (OpenAI-compatible API) and its CLI |
| `prompts/meeting_summary.md` | Instructions given to the LLM |
| `library.json` | Which folder each recording is in (git-ignored) |
| `llm/` | Hand-off folders for the LLM (git-ignored) |
| `models/pyannote-speaker-diarization-community-1/` | Bundled speaker model (CC-BY-4.0) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies and the pinned Python, managed by `uv` (`playwright` is a dev-only dependency for UI tests) |

## Troubleshooting

- **`401` / `403` / `GatedRepoError` for pyannote**: the `models/` folder is missing; copy it back from
  the zip.
- **First start fails while downloading Whisper**: check the internet connection and start again; the
  download resumes.
- **Tab recording: green bar doesn't move**: you shared a Window/Screen or turned off "Also share tab
  audio"; stop and pick the tab in the "Chrome Tab" pane.
- **Mac slows down / heavy swapping**: close other heavy apps; the speaker step peaks at ~5.8 GB.
