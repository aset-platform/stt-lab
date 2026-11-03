"""Summarise meeting transcripts with a local LLM (gpt-oss-20b) through an OpenAI-compatible API.

Reads llm/inbox/<recording>.md, writes llm/summaries/<recording>.md and .json (summary, decisions,
action items, open questions), then moves the transcript to llm/processed/.

Works with any server exposing /v1/chat/completions: Ollama (default, `ollama pull gpt-oss:20b`),
LM Studio, llama.cpp's llama-server, ... Override the defaults below in llm.json, e.g.
  {"base_url": "http://127.0.0.1:1234/v1", "model": "openai/gpt-oss-20b"}

Usage:
  uv run summarise.py --check     # is the LLM server reachable and the model available?
  uv run summarise.py             # summarise everything in llm/inbox/
  uv run summarise.py NAME ...    # summarise specific recordings
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import library

ROOT = Path(__file__).parent
CONFIG_FILE = ROOT / "llm.json"
PROMPT_FILE = ROOT / "prompts" / "meeting_summary.md"
DEFAULTS = {
    "base_url": "http://127.0.0.1:11434/v1",  # Ollama's OpenAI-compatible endpoint
    "model": "gpt-oss:20b",
    "api_key": "local",  # local servers ignore it, but the header must be present for some
    "timeout_s": 900,
    "max_input_tokens": 100_000,  # gpt-oss supports 128k context; ~2 h of meeting is ~30k tokens
}


def config():
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.is_file():
        cfg.update(json.loads(CONFIG_FILE.read_text()))
    return cfg


def _request(cfg, path, payload=None, timeout=10):
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


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


def summarise_one(name):
    """Summarise one queued transcript. Returns the parsed notes; raises on failure (input stays queued)."""
    cfg = config()
    library.audio_path(name)  # must be a real recording: a plain file name, so the paths below stay in llm/
    src = library.LLM_INBOX / f"{name}.md"
    if not src.is_file():
        raise library.LibraryError(f"{name} is not queued; move it to 'To summarise' first")
    transcript = src.read_text()
    if len(transcript) / 4 > cfg["max_input_tokens"]:  # ~4 characters per token
        raise library.LibraryError(f"transcript is ~{len(transcript) // 4} tokens, over max_input_tokens")
    t0 = time.perf_counter()
    reply = _request(cfg, "/chat/completions", {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": PROMPT_FILE.read_text()},
                     {"role": "user", "content": transcript}],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }, timeout=cfg["timeout_s"])
    # Servers like Ollama silently cut the input to their context window (default 4096 tokens), and the
    # reply then runs out of room. Report that clearly instead of failing on half-written JSON.
    sent = (len(PROMPT_FILE.read_text()) + len(transcript)) / 4  # rough estimate, ~4 characters per token
    got = (reply.get("usage") or {}).get("prompt_tokens")
    hint = "raise the model's context length (for Ollama: a model with a larger num_ctx, see README)"
    if got and got < 0.6 * sent:
        raise library.LibraryError(f"the LLM server read only {got} of ~{int(sent)} input tokens "
                                   f"(input truncated); {hint}")
    if reply["choices"][0].get("finish_reason") == "length":
        raise library.LibraryError(f"the reply was cut off at the context/output limit; {hint}")
    notes = _parse_json(reply["choices"][0]["message"]["content"])
    notes["_meta"] = {"model": cfg["model"], "seconds": round(time.perf_counter() - t0, 1),
                      "usage": reply.get("usage")}
    (library.LLM_SUMMARIES / f"{name}.json").write_text(json.dumps(notes, ensure_ascii=False, indent=1))
    (library.LLM_SUMMARIES / f"{name}.md").write_text(to_markdown(name, notes, cfg["model"]))
    src.rename(library.LLM_PROCESSED / src.name)
    library.mark_summarised(name)
    return notes


def queued():
    return sorted(p.name[:-3] for p in library.LLM_INBOX.glob("*.md"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="recordings to summarise (default: everything queued)")
    p.add_argument("--check", action="store_true", help="only check the LLM connection")
    a = p.parse_args()
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
