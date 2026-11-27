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

### Prerequisites

| What | Needed for | Install / check |
|---|---|---|
| **Apple Silicon Mac** (M1 or newer), macOS | everything: MLX (Whisper) does not run on Intel Macs, Linux or Windows | Apple menu → About This Mac |
| **16 GB RAM** (24 GB+ is more comfortable with the LLM) | transcription peaks at ~5.8 GB, the summariser LLM needs ~13 GB more | |
| **~20 GB free disk** | Python deps (~2 GB), Whisper medium (~1.5 GB), gpt-oss-20b (~13 GB) | |
| **git** | cloning | `xcode-select --install` (Command Line Tools) |
| [**uv**](https://docs.astral.sh/uv/) | Python 3.11 and all dependencies (installs Python for you) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [**Ollama**](https://ollama.com/download) | summaries, action items and the search-index embeddings | `ollama --version` |
| [**Docker Desktop**](https://www.docker.com/products/docker-desktop/) | optional: the search index (Postgres + pgvector) and the MCP server | `docker info` |
| **Chrome or Edge** | optional: recording a browser tab (meetings in Meet/Teams/Zoom web) | |

Not needed: a Hugging Face account or token, ffmpeg, or any cloud API key. Everything runs on your Mac.

### 1. Clone and install

```bash
git clone https://github.com/aset-platform/stt-lab.git
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
ollama pull qwen3-embedding:0.6b         # ~0.6 GB, embeddings for the search index (skip without Docker)
uv run server.py                         # then Settings tab → Summariser: "Create 32k copy", then "Use"
uv run summarise.py --check              # should print: ... ready
```

Any chat model in Ollama works (e.g. `qwen3.5:9b`). The Settings tab lists them with their context size and
makes the 32k copy for you (same weights, no extra download). To test models against each other, open a
recording in Manage → **Compare models**: each runs alone and you see the summaries side by side with time,
speed and memory.

The 32k-context copy matters: Ollama's default 4096-token context silently cuts any meeting longer than
~10 minutes (see [Summaries](#summaries-and-action-items)). Skip this step if you only need
transcripts; recordings then wait in *To summarise* until an LLM is available.

### 4. Run

```bash
./stt-lab up                             # index database (Docker) + Ollama check + app, opens the browser
./stt-lab status | logs | down           # down stops the app and the database (data is kept)
```

Or without the index database (everything else works; the index catches up when the database is back):

```bash
uv run server.py                         # then open http://127.0.0.1:8001
PORT=9000 uv run server.py               # if 8001 is taken
```

Only the database runs in Docker. The app and Ollama run natively: Docker on macOS runs a Linux VM with no
access to the Apple GPU, so Whisper (MLX) and the summariser would fall back to the CPU and live
transcription could not keep up.

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
- **Speakers**: the *least* number of people, `2` by default. It is not a cap: voices are grouped by how
  they sound, and a third or fourth person is still found; only if fewer than this are found is the audio
  split into this many. `0` = detect only (can merge two similar voices into one speaker, which leaves
  "you" in the summary without an owner).

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

## Summaries and action items

**The queue summarises itself.** Anything in **To summarise** is picked up automatically
(`llm.auto_pickup`, Settings → Summariser → *Queue*; on by default), oldest first, one at a time:

- it only starts while nothing is recording or transcribing, so recording always comes first;
- if a recording starts mid-summary, it stops after the current part (about a minute), unloads the LLM to
  give Whisper the memory, and goes on from the next part when the recording and its transcript are done;
- a summary that fails is not retried until its transcript changes or you run it by hand (**Summarise
  queued**); the reason is in `logs/server.log` (`[summary]` lines);
- the Manage tab shows which meeting is being summarised and whether it is paused.

With **Auto-summarise** ticked (next to Live transcript) a finished recording is moved to *To summarise* by
itself; untick it to choose what goes there. With auto pickup off, only **Summarise queued** runs the queue. **Copy transcript** (Transcribe tab) and **Copy summary**
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
They are written for any model, not one in particular.

**Long meetings go in parts.** A meeting longer than `llm.chunk_minutes` (10) is summarised in parts of about
10 minutes; each part gets the people found so far, the parts are merged in code, and one last call
(`prompts/meeting_tidy.md`) joins repeats and name variants without adding anything. A 45-minute meeting
takes ~7 min with `qwen3.5-9b-32k` and thinking off (`llm.reasoning_effort: "none"`), instead of timing out
as one call. `llm.keep_alive` ("30m") keeps the model loaded between the parts.

**What the prompt and the checks after it enforce:**

| Rule | How |
|---|---|
| No guessed roles | role, organisation and reporting line only when the transcript says them, each with the time it was said (`role_at`, …); a field whose cited line doesn't mention it, or a role of only generic words, becomes `undefined`. People with nothing known show as a plain name |
| One reporting line | "A reports to B" never both ways; loops are cleared (`check_hierarchy`) |
| Misheard tech terms | "Colibra" → Collibra and similar; names, clients and projects use the registry |
| Owners | "With Tamsin, you have to…" is the listener's task; an owner named in their own task becomes `unassigned` |
| Only people who were discussed | a person must be named in the transcript and have something concrete (a stated role, an action item, notes beyond "mentioned"); names the model took from the registry list are dropped, and so is someone named once whose notes only say they came up ("Mentioned as…", "Addressed by…") (`check_people`) |
| No voice labels | `SPEAKER_01` never appears in the notes: an owner that is a label becomes `unassigned`, and in text it becomes "a participant" (`drop_speaker_labels`). Name the voices to get names instead ([below](#who-is-who-naming-the-voices)) |
| Each point once | near-repeats in summary, decisions, questions, action items and a person's notes are removed, the more detailed one kept (`dedupe_notes`). Reworded repeats too: two decisions or questions sharing 55% of their key words (at least 5, in any order) are one point; summary bullets need 75% and 8 words, as they share words by topic |
| Left out | deal and commercial details, any money amount, remarks about someone's background, swearing and insults, sarcasm, gossip. The guardrail patterns then drop any sentence that still has one ([below](#keeping-sensitive-content-away-from-ai-assistants)) |

`_meta` in each summary's JSON records what the checks cleared (`cleared_without_evidence`,
`sensitive_sentences_dropped`, `owner_unassigned`, `people_dropped`, `repeats_removed`,
`speaker_labels_replaced`).

### Who is who (naming the voices)

Diarisation only tells voices apart (`SPEAKER_00`, `SPEAKER_01`); left to guess, the model easily puts
one person's tasks on the other ("your name is Marlow" is said *to* Marlow, not by him). So the voices are
named **before** a meeting is summarised:

- **Names first.** Moving a meeting to "To summarise" (or a recording queueing itself) opens a **Who is
  speaking?** dialog: each voice with its talk time and three of its longer lines, and a name field
  (workspace people are suggested). **Save and summarise** queues it; **Later** leaves it in "To
  summarise" marked **needs speaker names**, and the summary line says how many wait. The queue, the
  "Summarise queued" button and `summarise.py` all skip it until every voice has a name.
- **Don't know.** A guest you can't place, or a stray voice diarisation made from a few lines, can be
  ticked "don't know": it counts as answered and stays `SPEAKER_nn` (the notes then say "a participant").
  Two labels may get the same name when diarisation split one person.
- **One voice is you.** A meeting with a single voice takes your name (below) without asking.
- **Fixing later.** The **Speakers** box under a summary changes names; **Save and re-summarise** queues
  it again. Names are kept in `llm/speakers/<recording>.json` and used by the summary, the transcript
  view and the index; a meeting already queued gets the names in its hand-off at once.
- **Your name** (Settings → Recording): the hand-off header says `recorded_by: <you>`. The prompt also
  says a name someone is addressed by belongs to the other speaker, and that whoever says "I will…" owns
  the task.
- **Off switch.** Settings → Summariser → *Require speaker names before summarising*
  (`llm.require_speaker_names`); off, meetings are summarised with `SPEAKER_nn` as before.

**Whisper's silence filler** is removed before anything reads a transcript (`library.drop_repeats`): a phrase
looped three or more times in a row ("okay okay okay…") is kept once, and a line that repeats one of
the four before it word for word is dropped (a short "Okay." only when the same speaker repeats it). New
transcripts are saved without it; older ones are cleaned when summarised or indexed. `evaluate.py` scores a summary against its transcript
(names missed, plus an LLM judge).

Once gpt-oss is installed (e.g. `ollama pull gpt-oss:20b`), either click **Summarise queued** in the Manage
tab, or run:

```bash
uv run summarise.py --check     # is the LLM reachable and the model available?
uv run summarise.py             # summarise everything in llm/inbox/
```

Until then the Manage tab shows *LLM not connected* and queued transcripts simply wait.

**Memory:** a 9B model needs ~8–9 GB and gpt-oss-20b ~12–13 GB, so neither fits alongside the speech models on
16 GB. Before summarising, the server unloads Whisper and pyannote (they reload on the next transcription).
Summaries run on their own thread, one at a time, and wait or pause while anything records or transcribes;
index screening (which uses the same model) waits the same way.
When using `summarise.py` directly, stop the server first.

## Meeting labels (workspaces and registry)

Each meeting gets labels for search and filtering, and later for the vector store: **workspace**, meeting
**type**, **clients**, **projects**, **people** and **topics**. The summariser suggests them in the same LLM
call as the summary; **you confirm them** in Manage (a meeting's detail → Labels) before they count.

- **Workspace = one firm.** Each has its own registry, `registry/<id>.json` (git-ignored: it holds real
  names), with the meeting types and the known clients, projects, people and topics. Working with another
  firm means a new, empty workspace (Settings → Labels); older meetings keep their own workspace.
- **Misheard names.** Speech recognition spells names differently each time. Every registry entry has
  "heard as" spellings; the model is given the registry and writes the known name, and the review maps
  anything left over. For each name not in the registry you decide: a new entry, a misheard known name
  (saved as its alias, so it is recognised next time), or noise. A similar-sounding known name is shown as
  a hint, never chosen for you.
- **Things change.** Entries can be archived (kept on old meetings, marked "past" for the model), and
  confirming a meeting records the first/last date each entry was seen.
- Meetings summarised before labels existed: "Suggest labels" in the detail, or
  `uv run summarise.py --label [--workspace ID] NAME ...` (labels only, no new summary).
- **Summaries use the registry spellings too.** The model is asked to, and afterwards every "heard as"
  spelling of a client, project or person left in the summary is replaced with its registry name
  (whole words; a match that differs only in capitals is left alone). After teaching the registry new
  spellings, `uv run summarise.py --respell [NAME ...]` re-applies them to existing summaries (no LLM call).

Labels are stored in `llm/labels/<recording>.json` and are covered by the Manage search box.

### Fixing misheard words ("Fix a word")

Under a summary, select the misheard word (or type it), give the right spelling and press **Fix**:

- **This meeting is corrected:** summary (JSON and Markdown), saved transcripts, the summariser's
  copy and the labels, whole words in any case; tick *all meetings in this workspace* to fix them all.
  The search index re-embeds what changed.
- **The workspace learns it:** the wrong spelling becomes a "heard as" alias of the person, client,
  project or topic. If the right spelling is new, pick its type, or *other word / term* for jargon and
  product names that aren't meeting labels (kept under `words` in the registry).
- **Next time:**
  - Whisper is given the workspace's names as its starting prompt, so it leans toward those spellings
    (Settings → Transcription → *Prime with known names*).
  - Every new transcript is respelled through the registry.
  - The summariser's prompt lists every name and word with its "heard as" spellings, and respell runs
    on its output.

The models themselves aren't retrained: the registry steers them. Speech recognition can still hear a
new variant ("marlowe"); fix it once and that spelling is covered too.

## Search index (Postgres + pgvector)

Confirmed meetings are indexed for search by meaning, the base for the MCP server.

- **One index per workspace.** Each firm's meetings live in their own Postgres schema (`ws_<id>`) with their
  own tables and HNSW index; nothing is shared or mixed, a search runs inside one workspace, and a firm's
  index can be dropped as a whole (`uv run index.py drop-workspace ID`). Moving a meeting to another
  workspace moves its vectors; confirmed meetings in no workspace are not indexed.
- **Database:** `docker-compose.yml` runs `pgvector/pgvector` on `127.0.0.1:5433` (this Mac only), data in
  the `stt-lab-pgdata` volume. Settings "index" section, or `STT_DB_URL`, points elsewhere.
- **Embeddings:** `qwen3-embedding:0.6b` in Ollama (0.6 GB, fits next to the summariser; `./stt-lab up`
  pulls it once). A meeting embeds in a few seconds.
- **What is stored:** a `meetings` row (workspace, date, title, type, clients, projects, people, topics) and
  `passages`: each summary point, decision, action item (with owner and due date) and open question, and
  the transcript in one-minute passages, at most 2000 characters (whole lines; an over-long line is split
  at sentence ends, each starting ~15 s before the previous one ends,
  so an exchange across a boundary stays whole) with start time and speakers. Each passage is embedded with
  its meeting's title, date, clients and projects in front, so short points keep their context.
- **Index:** HNSW (cosine) on the embeddings with iterative scan, so a filtered search still returns the
  number of results asked for; label filters are exact (lowercase keys such as `client:acme`, GIN index)
  and a misheard name given as a filter is resolved through the workspace registry first. Changing the
  chunking settings re-indexes meetings on the next sync.
- **Only confirmed meetings**, so the label filters can be trusted. The index follows the library: confirm,
  move, rename, delete and new summaries trigger a sync that re-embeds only what changed; Trash is removed.
- **Settings → Search index** shows each workspace's index and has a search box to try it. **Sync now**
  is only needed if the database or Ollama was down when something changed (or a summary was changed
  outside the app, e.g. `--respell`): it indexes new or changed confirmed meetings and removes those in
  Trash, without re-embedding anything unchanged.

```bash
uv run index.py status
uv run index.py backfill                                   # index confirmed meetings that are new or changed
uv run index.py search "what did the client ask for" --client Acme --kind action --since 2026-11-01
```

Changing the embedding model needs `uv run index.py reset` and a backfill (vectors of two models don't mix).

## MCP server (meetings for Claude and other MCP clients)

`./stt-lab up` also starts an MCP server in Docker on **http://127.0.0.1:8765/mcp** (this Mac only). It
reads the search index with a read-only database login, and works inside one workspace per call (the
active one unless `workspace` is given), so firms are never mixed in an answer.

| Tool | What it gives |
|---|---|
| `list_workspaces` | Workspaces, their meeting counts and date range, the active one, today's date |
| `list_meetings` | Meetings by date phrase (`yesterday`, `2 days ago`, `last week`, `monday`, ISO dates or ranges), client, project, person, type |
| `get_meeting` | One meeting's summary, decisions, action items, open questions and labels |
| `get_transcript` | The full transcript as `[mm:ss] SPEAKER: text`, in pages for long meetings |
| `search` | Search by meaning across the workspace's notes and transcript passages, with filters |
| `action_items` | Action items in a period, by owner, client or project |
| `registry` | Known clients, projects, people and topics with their correct spellings |

Prompts (ready-made workflows the client can offer): **meetings_digest** (insights, decisions and action
items for a day or period), **feature_requirements** (a 1 to 5 page requirements document from one or more
meetings, every requirement traced to a meeting and timestamp), **client_brief**, **action_tracker**.

Connect a client:

```bash
# Claude Code (all projects)
claude mcp add --transport http --scope user stt-lab http://127.0.0.1:8765/mcp
```

Claude Desktop: Settings → Developer → Edit Config, add a local bridge to the HTTP server:

```json
{ "mcpServers": { "stt-lab": { "command": "npx", "args": ["-y", "mcp-remote", "http://127.0.0.1:8765/mcp"] } } }
```

Then ask, for example, "what were the action items from my meetings yesterday?" or use the
`feature_requirements` prompt with the meetings' names.

- **Time zone:** `./stt-lab up` passes the Mac's time zone, so "yesterday" is your yesterday.
- **Token (optional):** set `STT_MCP_TOKEN` (e.g. in `.env`) and clients must send
  `Authorization: Bearer <token>` (`claude mcp add ... --header "Authorization: Bearer <token>"`).
- **ChatGPT:** its MCP connectors run from OpenAI's servers, so they can't reach a server on
  `127.0.0.1`. That needs a public HTTPS tunnel with authentication, which puts your meeting data on the
  internet; not set up here.
- **Read-only:** the server's database role can only `SELECT`; it can't change the index or the app.
- **Audit log:** every tool call is logged in `logs/mcp/mcp-audit.jsonl`, one JSON line each:
  - time, the calling app (its User-Agent: Codex, Claude Code, ...), tool and arguments
  - the workspace, and what kind of content went out (list, notes, transcript, registry)
  - the meetings returned (name, title, date) and the size of the answer

  The meeting text itself isn't logged. It survives container rebuilds, and Settings → **MCP access**
  shows it. Remember the calling app may send what it receives to its own model provider.

### Codex

Codex runs on the Mac, so it can use the local address. In `~/.codex/config.toml`:

```toml
[mcp_servers.stt-lab]
url = "http://127.0.0.1:8765/mcp"
# bearer_token_env_var = "STT_MCP_TOKEN"   # if you set a token
```

## Keeping sensitive content away from AI assistants

The MCP server feeds meetings to outside services (ChatGPT, Claude, Codex). Deal figures, remarks about people
and abuse must not reach them, so there are three layers:

1. **The summary prompt** asks the model to leave them out (see [Summaries](#summaries-and-action-items)).
2. **Guardrail patterns** (`mcp_server/guardrails.py`) catch amounts ("$1.3M", "1.3 million for a year",
   "2 crore"; technical counts such as "1.3 million rows" are kept), deal terms (pricing, budget, margin,
   invoice…), remarks about origin, race or religion, swearing in English and Hindi, and gossip (office politics, people
   not getting along, "incompetent"; "political risk" and other business uses are left alone). They run twice: on
   every summary after the model wrote it (the sentence is dropped) and on every MCP answer (it becomes
   `[removed]`; the audit log counts them as `removed`).
3. **Screening by meaning when indexing** (`share_filter.py`, `prompts/share_filter.md`): the local LLM reads
   every transcript line (in batches of 40, with the lines around it) and every point of the notes, and flags
   deal talk said vaguely ("around 1.3 for the year"), remarks about someone's background, insults, sarcasm,
   gossip and a person's compliance issues. When unsure about deals or background it flags.

A flagged item is **not deleted**: it stays in the index, marked `restricted` with its reason, so search on
this Mac still finds it. The MCP server's read-only login can only read two views, `shared_passages` and
`shared_lines`, which leave flagged passages out and show flagged lines as `[removed]`; it has no access to
the tables themselves. Verdicts are cached by text in `llm/guard/<recording>.json`, so re-indexing doesn't
ask the model again. If the model can't be reached, the meeting is not indexed: nothing unscreened is
shared. Expect some over-flagging (a plain "she lives in Ireland" can be held back).

| Setting | Default | What |
|---|---|---|
| `index.share_filter` | `true` | Screen by meaning before indexing (the patterns always run) |
| `index.share_model` | `""` | Model for screening (`""` = the summariser's) |
| `STT_MCP_TRANSCRIPTS` (env, `./stt-lab up`) | `on` | `off`: MCP clients get notes only; `get_transcript` refuses and search returns no transcript passages |

Patterns can't recognise everything (a rude remark in polite words); for client meetings with sensitive
talk, `STT_MCP_TRANSCRIPTS=off ./stt-lab up` keeps transcript text away from MCP clients entirely.

### Observability page

The **Observability** tab answers one question: has anything left this Mac that shouldn't have? It shows

- **a verdict and alerts.** Every MCP call logs a fingerprint (16-character sha1) of each point and transcript
  line it sent, never the text. The page matches them against what is restricted *now*, so an item restricted
  later (a new screening, a fixed rule) still shows which calls had already shared it ("shared, now
  restricted"). Calls logged before fingerprints existed are checked by meeting ("not verifiable"), and calls
  from before the guardrails as "before guardrails";
- **safety checks.** The MCP database login can read only the screened views and write nothing, and the
  views really hide every restricted line and point;
- **AI client logs.** *Scan now* looks for every restricted text (30+ characters) in the session logs Codex
  (`~/.codex/sessions`) and Claude Code (`~/.claude/projects`) keep on this Mac. A hit means the text was in
  a conversation with that app, so its model provider has it too. Hits in Claude Code sessions of this repo
  (building stt-lab) are listed apart;
- **what is held back**, per meeting and reason, with the texts (on this Mac only), and the recent MCP calls.

`uv run observe.py` prints the same summary in the terminal; the last scan is in `logs/observe-scan.json`.

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
| `labels.py`, `prompts/meeting_labels.md` | Meeting labels, workspaces and registries; `registry/` and `llm/labels/` hold the data (git-ignored) |
| `corrections.py` | "Fix a word": corrects a meeting's summary and transcripts and teaches the registry the spelling |
| `logs/mcp/mcp-audit.jsonl` | The MCP server's audit log: one line per tool call (git-ignored) |
| `index.py` | Search index: confirmed meetings as passages with embeddings in Postgres (pgvector) |
| `docker-compose.yml`, `stt-lab` | The index database and MCP server containers, and the one-command launcher (`./stt-lab up`) |
| `mcp_server/` | MCP server (tools and prompts over the index), its Dockerfile, and `shared.py` (what the indexer and server agree on) |
| `settings.json` | All settings, one section per feature, edited in the Settings tab; defaults in `settings.py` (git-ignored; replaces the older `llm.json`, which is moved in automatically) |
| `library.py` | Folders, rename, delete, and writing transcripts to `llm/inbox/` |
| `summarise.py` | LLM summariser (OpenAI-compatible API): parts, merge, evidence/hierarchy/owner checks, and its CLI |
| `prompts/meeting_summary.md`, `prompts/meeting_tidy.md` | Instructions given to the LLM: the summary, and the tidy pass after a meeting in parts |
| `mcp_server/guardrails.py` | Guardrail patterns: amounts, deal terms, sensitive remarks, abuse, gossip; used on summaries and MCP answers |
| `share_filter.py`, `prompts/share_filter.md` | Screening by meaning before indexing; restricted items never reach MCP clients |
| `evaluate.py` | Scores a summary against its transcript: names missed, LLM judge |
| `observe.py` | The Observability page: restricted items, MCP calls matched by fingerprint, self-checks, client-log scan |
| `library.json` | Which folder each recording is in (git-ignored) |
| `llm/` | Hand-off folders for the LLM (git-ignored) |
| `models/pyannote-speaker-diarization-community-1/` | Bundled speaker model (CC-BY-4.0) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies and the pinned Python, managed by `uv` (`playwright` is a dev-only dependency for UI tests) |

## Troubleshooting

- **`401` / `403` / `GatedRepoError` for pyannote**: the `models/` folder is missing or incomplete;
  restore it with `git checkout -- models/`.
- **`./stt-lab up` says Docker is not running**: start Docker Desktop, or skip the index and run
  `uv run server.py` instead.
- **Summaries never start**: `uv run summarise.py --check` tells you whether Ollama is running and the
  model is pulled.

## Contributing

`main` is protected: changes go through a pull request, and every pull request needs the owner's
approval before it can be merged (see `.github/CODEOWNERS`). Fork the repo (or create a branch if you
have access), enable the pre-commit checks (`git config core.hooksPath .githooks`), and open a pull
request against `main`. Never commit real meeting recordings, transcripts, summaries or registries:
only the synthetic `sample-*` files are tracked, and CI rejects other audio.

## License

[MIT](LICENSE). The bundled pyannote speaker model is CC-BY-4.0, see
`models/pyannote-speaker-diarization-community-1/ATTRIBUTION.md`.
- **First start fails while downloading Whisper**: check the internet connection and start again; the
  download resumes.
- **Tab recording: green bar doesn't move**: you shared a Window/Screen or turned off "Also share tab
  audio"; stop and pick the tab in the "Chrome Tab" pane.
- **Mac slows down / heavy swapping**: close other heavy apps; the speaker step peaks at ~5.8 GB.
