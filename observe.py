"""Observability: has anything left this Mac that shouldn't have?

The Observability page (ui.html) is built from report() and scan(). Four sources:

1. What is held back: every restricted point and transcript line in the index (share_filter.py), with
   its reason. These are the things MCP clients must never get.
2. What MCP clients got: the audit log (logs/mcp/mcp-audit.jsonl). Each call records fingerprints of the
   points and lines it shared (mcp_server/audit.py), so a call that shared an item that is restricted
   now (it was screened later, or the screening changed) is an alert. Calls logged before fingerprints
   existed can only be judged by meeting: they shared notes or transcript of a meeting that now has
   restricted items. Calls from before the guardrails (no "removed" field) got unfiltered text.
3. Self-checks of the protection itself: the MCP database login can read only the screened views, and
   the views really hide every restricted item.
4. The AI clients' own session logs on this Mac (Codex, Claude Code): a restricted text found there was
   in a conversation with that client, so it has also reached its provider. Run on demand (scan()).
"""
import json
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import index
import labels
import library
import share_filter
from mcp_server import guardrails

AUDIT = library.ROOT / "logs" / "mcp" / "mcp-audit.jsonl"
SCAN_FILE = library.ROOT / "logs" / "observe-scan.json"
CLIENT_LOGS = [("Codex", Path.home() / ".codex" / "sessions"), ("Claude Code", Path.home() / ".claude" / "projects")]
DEV_PROJECT = "-" + str(library.ROOT).strip("/").replace("/", "-")  # Claude Code's folder for this repo
MCP_TABLES = {"meetings", "entities", "shared_passages", "shared_lines", "workspaces", "index_info"}
MIN_SCAN_CHARS = 30  # shorter texts ("Yes, around 1.3.") would match by chance in unrelated logs
SHARES_TEXT = {"notes", "transcript", "transcript+notes"}
_scan_lock = threading.Lock()


def _norm(text):
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


def _reason(r):
    """"pattern:money" -> "money", "deal" -> "deal"; a passage's joined reasons -> the first."""
    r = str(r or "").split(",")[0].strip()
    return r.split(":", 1)[1] if r.startswith("pattern:") else r


def _ws_names():
    try:
        return {w["id"]: w["name"] for w in labels.workspaces()}
    except Exception:
        return {}


def restricted_items():
    """Every restricted item in the index, as dicts with workspace, meeting, title, held, kind, reason,
    text, at (seconds, transcript lines) and key (the fingerprint MCP calls are matched against)."""
    names, out = _ws_names(), []
    with index.connect() as conn:
        for ws, schema in index._schemas(conn).items():
            q = index._sql("""select p.meeting, m.title, m.held, p.kind, p.restricted, p.text, p.owner
                              from {s}.passages p join {s}.meetings m on m.name = p.meeting
                              where p.restricted is not null and p.kind <> 'transcript'""", schema)
            for meeting, title, held, kind, why, text, owner in conn.execute(q):
                out.append({"workspace": names.get(ws, ws), "meeting": meeting, "title": title, "held": str(held or ""),
                            "kind": kind, "reason": _reason(why), "text": text, "owner": owner, "at": None,
                            "key": share_filter._key(text)})
            q = index._sql("""select l.meeting, m.title, m.held, l.start_s, l.speaker, l.text, l.restricted
                              from {s}.lines l join {s}.meetings m on m.name = l.meeting
                              where l.restricted is not null order by l.meeting, l.seq""", schema)
            for meeting, title, held, start, speaker, text, why in conn.execute(q):
                out.append({"workspace": names.get(ws, ws), "meeting": meeting, "title": title, "held": str(held or ""),
                            "kind": "transcript", "reason": _reason(why), "text": text, "speaker": speaker,
                            "at": start, "key": share_filter._key(f"{speaker}: {text}")})
    return out


def self_checks():
    """[{ok, check, detail}]: the MCP login's rights, and that the views hide every restricted item."""
    checks = []
    with index.connect() as conn:
        rows = conn.execute("""
            select n.nspname, c.relname,
                   has_table_privilege(%(r)s, c.oid, 'SELECT'),
                   has_table_privilege(%(r)s, c.oid, 'INSERT') or has_table_privilege(%(r)s, c.oid, 'UPDATE')
                     or has_table_privilege(%(r)s, c.oid, 'DELETE') or has_table_privilege(%(r)s, c.oid, 'TRUNCATE')
            from pg_class c join pg_namespace n on n.oid = c.relnamespace
            where c.relkind in ('r', 'v', 'm') and n.nspname not in ('pg_catalog', 'information_schema')
              and n.nspname not like 'pg\\_%%'""", {"r": index.READONLY_ROLE}).fetchall()
        readable = sorted(f"{s}.{t}" for s, t, sel, _ in rows if sel)
        extra = [x for x in readable if x.split(".", 1)[1] not in MCP_TABLES]
        writes = [f"{s}.{t}" for s, t, _, w in rows if w]
        checks.append({"ok": not extra, "check": "The MCP login reads only the screened views",
                       "detail": f"can also read {', '.join(extra)}" if extra
                       else f"{len(readable)} tables and views, none with restricted text"})
        checks.append({"ok": not writes, "check": "The MCP login cannot change the index",
                       "detail": f"can write {', '.join(writes)}" if writes else "no write rights"})
        for ws, schema in index._schemas(conn).items():
            q = index._sql("""select count(*) from {s}.shared_lines sl join {s}.lines l using (meeting, seq)
                              where l.restricted is not null and sl.text <> '[removed]'""", schema)
            lines = conn.execute(q).fetchone()[0]
            q = index._sql("""select count(*) from {s}.shared_passages sp join {s}.passages p using (id)
                              where p.restricted is not null and p.kind <> 'transcript'""", schema)
            notes = conn.execute(q).fetchone()[0]
            # transcript passages: the shared text must not contain any restricted line
            q = index._sql("""select p.meeting, p.shared_text, p.start_s, p.end_s from {s}.passages p
                              where p.kind = 'transcript' and p.restricted is not null and p.shared_text is not null""",
                           schema)
            bad = 0
            for meeting, shared, a, b in conn.execute(q).fetchall():
                rl = conn.execute(index._sql("""select speaker, text from {s}.lines where meeting = %s
                                                and restricted is not null and start_s between %s and %s""", schema),
                                  (meeting, a, b)).fetchall()
                bad += sum(1 for spk, t in rl if f"{spk}: {t}" in shared)
            name = _ws_names().get(ws, ws)
            n = lines + notes + bad
            checks.append({"ok": n == 0, "check": f"{name}: restricted items are hidden from MCP",
                           "detail": f"{n} restricted item(s) readable through the views" if n
                           else "every restricted line reads [removed]; restricted points are left out"})
    return checks


def calls():
    if not AUDIT.is_file():
        return []
    out = []
    for line in AUDIT.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _item(i):
    return {k: i.get(k) for k in ("workspace", "meeting", "title", "held", "kind", "reason", "text", "at", "speaker", "owner")}


def alerts(cs, items):
    """Alerts from the audit log, newest first. level: "leak" (a call shared an item that is restricted
    now), "unprotected" (before the guardrails: unfiltered text of a meeting with restricted items) or
    "unverified" (no fingerprints logged: the call may have shared some of them)."""
    by_key = {i["key"]: i for i in items}
    per_meeting = {}
    for i in items:
        per_meeting[i["meeting"]] = per_meeting.get(i["meeting"], 0) + 1
    out = []
    for c in cs:
        base = {"at": c.get("at"), "client": c.get("client") or "", "tool": c.get("tool"), "args": c.get("args")}
        if c.get("error") or c.get("shared") not in SHARES_TEXT:
            continue
        if "fp" in c:
            hits = [by_key[k] for k in c.get("fp") or [] if k in by_key]
            if hits:
                out.append({**base, "level": "leak", "items": [_item(i) for i in hits],
                            "why": f"shared {len(hits)} item(s) that are restricted now"})
            continue
        risky = [(m.get("title") or m.get("meeting"), per_meeting[m.get("meeting")])
                 for m in c.get("meetings") or [] if per_meeting.get(m.get("meeting"))]
        if not risky:
            continue
        level = "unverified" if "removed" in c else "unprotected"
        what = "the full transcript" if c.get("tool") == "get_transcript" else f"{c.get('shared')} text"
        out.append({**base, "level": level, "meetings": [{"title": t, "restricted": n} for t, n in risky],
                    "why": (f"sent {what} before the guardrails existed" if level == "unprotected"
                            else f"sent {what}; logged before fingerprints, so which items went out is unknown")})
    return out[::-1]


def report():
    """Everything the Observability page shows, except the client-log scan (see scan())."""
    cs = calls()
    try:
        items, idx_error = restricted_items(), None
    except Exception as e:  # database down: the audit part still works
        items, idx_error = [], str(e)
    try:
        checks = self_checks()
    except Exception as e:
        checks = [{"ok": False, "check": "Index database reachable", "detail": str(e)}]
    al = alerts(cs, items)
    day = (datetime.now().astimezone() - timedelta(days=1)).isoformat(timespec="seconds")
    reasons, meetings = {}, {}
    for i in items:
        reasons[i["reason"]] = reasons.get(i["reason"], 0) + 1
        m = meetings.setdefault(i["meeting"], {"meeting": i["meeting"], "title": i["title"], "held": i["held"],
                                               "workspace": i["workspace"], "reasons": {}, "items": []})
        m["reasons"][i["reason"]] = m["reasons"].get(i["reason"], 0) + 1
        m["items"].append(_item(i))
    clients = {}
    for c in cs:
        k = (c.get("client") or "unknown").split(" ")[0].split("/")[0]
        clients[k] = clients.get(k, 0) + 1
    flagged = {(a["at"], a["tool"]): a["level"] for a in al}
    scan_last = json.loads(SCAN_FILE.read_text()) if SCAN_FILE.is_file() else None
    return {
        "summary": {
            "calls": len(cs), "calls_24h": sum(1 for c in cs if str(c.get("at") or "") >= day),
            "clients": clients, "removed": sum(c.get("removed") or 0 for c in cs),
            "restricted": len(items), "reasons": reasons,
            "leaks": sum(1 for a in al if a["level"] == "leak"),
            "unprotected": sum(1 for a in al if a["level"] == "unprotected"),
            "unverified": sum(1 for a in al if a["level"] == "unverified"),
            "checks_failed": sum(1 for c in checks if not c["ok"]),
            "scan_hits": len([h for h in (scan_last or {}).get("hits", []) if not h.get("dev")]),
            "fingerprinted_since": next((c.get("at") for c in cs if "fp" in c), None),
            "guarded_since": next((c.get("at") for c in cs if "removed" in c), None),
        },
        "alerts": al, "checks": checks, "index_error": idx_error,
        "restricted": sorted(meetings.values(), key=lambda m: m["held"], reverse=True),
        "calls": [{**{k: c.get(k) for k in ("at", "client", "tool", "args", "shared", "meetings", "chars", "removed",
                                             "error")},
                   "items": len(c.get("fp") or []) if "fp" in c else None,
                   "flag": flagged.get((c.get("at"), c.get("tool")))} for c in cs[::-1][:200]],
        "scan": scan_last,
    }


def _strings(x, out):
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            _strings(v, out)
    elif isinstance(x, list):
        for v in x:
            _strings(v, out)


def scan():
    """Look for restricted texts in the AI clients' session logs on this Mac. A hit means the text was in a
    conversation with that client (and so with its model provider). Hits in Claude Code sessions of this
    repo are development work on stt-lab itself and are marked dev. Saved to logs/observe-scan.json."""
    with _scan_lock:
        t0 = time.time()
        items = restricted_items()
        needles = {}
        for i in items:
            n = _norm(i["text"])
            if len(n) >= MIN_SCAN_CHARS:
                needles.setdefault(n, i)
        hits, files, size = [], 0, 0
        for client, root in CLIENT_LOGS:
            if not root.is_dir():
                continue
            for f in sorted(root.rglob("*.jsonl")):
                files += 1
                size += f.stat().st_size
                parts = []
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        _strings(json.loads(line), parts)
                    except ValueError:
                        parts.append(line)
                text = _norm("\n".join(parts))
                found = [i for n, i in needles.items() if n in text]
                dev = client == "Claude Code" and DEV_PROJECT in f.parts
                for i in found:
                    hits.append({"client": client, "dev": dev, "file": "~/" + str(f.relative_to(Path.home())),
                                 "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes"),
                                 **_item(i)})
        result = {"at": datetime.now().astimezone().isoformat(timespec="seconds"), "files": files, "bytes": size,
                  "texts": len(needles), "seconds": round(time.time() - t0, 1), "hits": hits,
                  "roots": [f"~/{r.relative_to(Path.home())}" for _, r in CLIENT_LOGS if r.is_dir()]}
        SCAN_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCAN_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        return result


if __name__ == "__main__":
    r = report()
    print(json.dumps(r["summary"], indent=1))
    for c in r["checks"]:
        print("✓" if c["ok"] else "✗", c["check"], "-", c["detail"])
    for a in r["alerts"][:10]:
        print(a["level"], a["at"], a["tool"], a["why"])
