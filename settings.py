"""All of stt-lab's settings in one place: settings.json (git-ignored), one section per feature.

    {"recording": {...}, "transcription": {...}, "llm": {...}}

Every section has defaults here, so settings.json only holds what was changed (in the Settings tab or by
hand). A feature reads its section with get("llm") and the Settings tab writes with update(). Future
features (vector store, MCP server, email) add their own section to DEFAULTS and get the same handling.

llm.json, the older home of the LLM section, is folded in on first use and then removed.
"""
import json
import threading
from pathlib import Path

ROOT = Path(__file__).parent
FILE = ROOT / "settings.json"
LEGACY_LLM = ROOT / "llm.json"

DEFAULTS = {
    "recording": {
        "live_transcript": True,   # transcribe while recording; speakers are labelled when you stop
        "auto_summarise": True,    # after transcript + speakers: summarise and file under Summarised
    },
    "transcription": {             # starting values of the Transcribe tab's controls
        "language": "",            # "" = detect per chunk, or "en", "hi"
        "task": "translate",       # "translate" = English output, "transcribe" = as spoken
        "speakers": 0,             # 0 = detect automatically
    },
    "llm": {
        "base_url": "http://127.0.0.1:11434/v1",  # Ollama's OpenAI-compatible endpoint
        "model": "gpt-oss:20b",
        "api_key": "local",        # local servers ignore it, but the header must be present for some
        "timeout_s": 900,
        "max_input_tokens": 100_000,  # ~2 h of meeting is ~30k tokens
        "unload_after_summary": True,  # free the model's memory as soon as a summary is written
        "labels_timeout_s": 2700,  # labels-only pass (a reasoning model can think for 15+ min)
    },
    "labels": {
        "workspace": "",           # registry/<id>.json new meetings are labelled in; "" = none
    },
}

_lock = threading.Lock()


class SettingsError(ValueError):
    pass


def _read():
    saved = json.loads(FILE.read_text()) if FILE.is_file() else {}
    if LEGACY_LLM.is_file():  # one-time move of llm.json into its section
        saved["llm"] = {**json.loads(LEGACY_LLM.read_text()), **saved.get("llm", {})}
        FILE.write_text(json.dumps(saved, indent=1) + "\n")
        LEGACY_LLM.unlink()
    return saved


def current():
    """Every section, defaults filled in."""
    with _lock:
        saved = _read()
    return {s: {**d, **saved.get(s, {})} for s, d in DEFAULTS.items()}


def get(section):
    return current()[section]


def update(changes):
    """Merge {section: {key: value}} into settings.json. Only known keys, with the default's type."""
    with _lock:
        saved = _read()
        for section, values in changes.items():
            if section not in DEFAULTS:
                raise SettingsError(f"unknown settings section {section!r}")
            for key, value in values.items():
                default = DEFAULTS[section].get(key, ...)
                if default is ...:
                    raise SettingsError(f"unknown setting {section}.{key}")
                if type(value) is not type(default) and not (type(default) is float and type(value) is int):
                    raise SettingsError(f"{section}.{key} must be {type(default).__name__}")
                saved.setdefault(section, {})[key] = value
        FILE.write_text(json.dumps(saved, indent=1) + "\n")
    return current()
