"""Audit log of MCP tool calls: who asked, which tool, with what arguments, which meetings went out.

One JSON line per call in $STT_MCP_AUDIT (docker-compose.yml mounts it to logs/mcp/ on the Mac, so it
survives container rebuilds). It records what was shared, never the shared text itself: meeting names,
titles and dates, the size of the answer, and "fp": a fingerprint (sha1, 16 hex) of every point and
transcript line that went out. The Observability page (observe.py) matches the fingerprints against what
is restricted, so a later screening that restricts an item shows which calls had already shared it.
"""
import hashlib
import re
import contextvars
import functools
import json
import os
import threading
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PATH = os.environ.get("STT_MCP_AUDIT", "/audit/mcp-audit.jsonl")
client = contextvars.ContextVar("client", default="")  # set per HTTP request by track_client
_lock = threading.Lock()
try:  # log times in the Mac's zone (./stt-lab up passes it), not the container's UTC
    TZ = ZoneInfo(os.environ.get("STT_TZ") or "UTC")
except ZoneInfoNotFoundError:
    TZ = timezone.utc


def track_client(app):
    """ASGI middleware: remember the caller's User-Agent (codex, claude-code, python-httpx...) for the log."""
    async def tracked(scope, receive, send):
        if scope["type"] == "http":
            ua = dict(scope.get("headers") or []).get(b"user-agent", b"").decode("latin-1")
            token = client.set(ua[:120])
            try:
                return await app(scope, receive, send)
            finally:
                client.reset(token)
        await app(scope, receive, send)
    return tracked


def shared(result):
    """The meetings an answer contains, as [{meeting, title, date}], deduplicated, in order."""
    seen, out = set(), []

    def walk(x):
        if isinstance(x, dict):
            m = x.get("meeting")
            if isinstance(m, str) and m not in seen:
                seen.add(m)
                out.append({"meeting": m, "title": x.get("title"), "date": x.get("date")})
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(result)
    return out


_HEAD = re.compile(r"^\[[\d:]+\]\s*")  # "[12:34] " before a transcript line
_NOT_TEXT = {"meeting", "recording", "workspace", "workspace_id", "date", "start", "at", "first_seen", "last_seen",
             "status", "kind", "error", "today", "range", "title", "owner", "due", "held", "started_at", "type"}


def fp(text):
    """The fingerprint of one shared text: the same as share_filter's cache key for it."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def fingerprints(result):
    """Fingerprints of every point and every transcript line ("SPEAKER: text") in an answer."""
    keys = set()

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, str) and key not in _NOT_TEXT and x.strip():
            keys.add(fp(x.strip()))
            if "\n" in x or _HEAD.match(x):
                for line in x.split("\n"):
                    line = _HEAD.sub("", line).strip()
                    if line and line != "[removed]":
                        keys.add(fp(line))
    walk(result)
    return sorted(keys)


def _kind(tool, result):
    """What sort of meeting content went out: names only, notes (summaries), transcript text, or registry."""
    if tool == "get_transcript":
        return "transcript"
    if tool == "registry":
        return "registry"
    if tool == "search":
        kinds = {r.get("kind") for r in (result or {}).get("results", [])}
        return "transcript+notes" if "transcript" in kinds else "notes"
    if tool in ("get_meeting", "action_items"):
        return "notes"
    return "list"


def write(entry):
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _lock:
        try:
            os.makedirs(os.path.dirname(PATH), exist_ok=True)
            with open(PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:  # never fail a tool call because the log can't be written
            print(f"[audit] could not write {PATH}: {e}", flush=True)


def audited(fn):
    """Log every call of an MCP tool. functools.wraps keeps the signature the MCP schema is built from."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result, error = None, None
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, dict) and "error" in result:
                error = result["error"]
            return result
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            ok = result if isinstance(result, dict) and not error else {}
            write({
                "at": datetime.now(TZ).isoformat(timespec="seconds"),
                "client": client.get(),
                "tool": fn.__name__,
                "args": {k: v for k, v in kwargs.items() if v not in ("", None)},
                "workspace": ok.get("workspace"),
                "shared": _kind(fn.__name__, ok) if ok else None,
                "meetings": shared(ok),
                "chars": len(json.dumps(result, ensure_ascii=False, default=str)) if result is not None else 0,
                "removed": (result or {}).get("removed_by_guardrails", 0) if isinstance(result, dict) else 0,
                "fp": fingerprints(ok) if ok and fn.__name__ not in ("list_workspaces", "registry") else [],
                "ms": round((time.perf_counter() - t0) * 1000),
                **({"error": str(error)[:300]} if error else {}),
            })
    return wrapper
