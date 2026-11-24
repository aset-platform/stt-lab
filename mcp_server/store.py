"""Read-only access to the meeting index for the MCP server: one workspace per call, never mixed.

Reads only Postgres (as the read-only role) and asks Ollama to embed questions; it needs no files from
the Mac. The indexer (index.py) keeps the tables current.
"""
import difflib
import os
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .shared import KINDS, EmbedError, embed, label_key, norm

DB_URL = os.environ.get("STT_MCP_DB_URL", "postgresql://sttlab_mcp:sttlab_mcp@127.0.0.1:5433/sttlab")
EMBED_URL = os.environ.get("STT_EMBED_URL", "http://127.0.0.1:11434")
TZ = ZoneInfo(os.environ.get("STT_TZ") or "UTC")
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class StoreError(Exception):
    """A problem the caller can fix (unknown workspace or meeting, bad date): shown to the model as is."""


def connect():
    import psycopg
    from pgvector.psycopg import register_vector

    try:
        conn = psycopg.connect(DB_URL, connect_timeout=3)
    except psycopg.OperationalError:
        raise StoreError("the meeting index database is not running (start stt-lab with ./stt-lab up)")
    register_vector(conn)
    return conn


def _q(text, schema):
    from psycopg import sql
    return sql.SQL(text).format(s=sql.Identifier(schema))


def today():
    return datetime.now(TZ).date()


# ---------- workspaces ----------
def _workspace(conn, workspace):
    """(id, name, schema) of the named workspace, or of the active one when none is given."""
    rows = conn.execute("select id, name, schema, active from public.workspaces order by id").fetchall()
    if not rows:
        raise StoreError("no workspaces in the index yet")
    if not workspace:
        hit = next((r for r in rows if r[3]), None)
        if not hit:
            raise StoreError("no active workspace: pass `workspace` (see list_workspaces)")
    else:
        hit = next((r for r in rows if workspace.strip().lower() in (r[0], r[1].lower())), None)
        if not hit:
            raise StoreError(f"no workspace {workspace!r}; known: {', '.join(r[1] for r in rows)}")
    has = conn.execute("select 1 from information_schema.tables where table_schema = %s and table_name = 'meetings'",
                       (hit[2],)).fetchone()
    if not has:
        raise StoreError(f"workspace {hit[1]} has no indexed meetings yet")
    return hit[0], hit[1], hit[2]


def list_workspaces():
    with connect() as conn:
        out = []
        for ws_id, name, schema, active in conn.execute(
                "select id, name, schema, active from public.workspaces order by id").fetchall():
            try:
                n, first, last = conn.execute(_q("select count(*), min(held), max(held) from {s}.meetings", schema)).fetchone()
            except Exception:
                conn.rollback()
                n, first, last = 0, None, None
            out.append({"workspace": ws_id, "name": name, "active": active, "meetings": n,
                        "first_meeting": first and first.isoformat(), "last_meeting": last and last.isoformat()})
    return {"today": today().isoformat(), "workspaces": out}


# ---------- dates ----------
def parse_when(when):
    """(since, until) for phrases like "today", "yesterday", "2 days ago", "last 7 days", "this week",
    "last week", "this month", "last month", "monday", "2026-11-20" or "2026-11-01..2026-11-15"."""
    if not when or not str(when).strip():
        return None, None
    w, t = " ".join(str(when).lower().split()), today()
    if m := re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*(?:\.\.|to|–|-{2})\s*(\d{4}-\d{2}-\d{2})", w):
        return date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", w):
        d = date.fromisoformat(w)
        return d, d
    if w == "today":
        return t, t
    if w == "yesterday":
        d = t - timedelta(days=1)
        return d, d
    if w in ("day before yesterday", "the day before yesterday"):
        d = t - timedelta(days=2)
        return d, d
    if m := re.fullmatch(r"(\d+) days? ago", w):
        d = t - timedelta(days=int(m.group(1)))
        return d, d
    if m := re.fullmatch(r"(?:last|past) (\d+) days?", w):
        return t - timedelta(days=int(m.group(1)) - 1), t
    if w in ("this week", "week"):
        return t - timedelta(days=t.weekday()), t
    if w == "last week":
        start = t - timedelta(days=t.weekday() + 7)
        return start, start + timedelta(days=6)
    if w in ("this month", "month"):
        return t.replace(day=1), t
    if w == "last month":
        end = t.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end
    if w.removeprefix("last ") in WEEKDAYS:  # the most recent such day, today included
        back = (t.weekday() - WEEKDAYS.index(w.removeprefix("last "))) % 7
        d = t - timedelta(days=back or (7 if w.startswith("last ") else 0))
        return d, d
    raise StoreError(f"can't read the date {when!r}: use e.g. today, yesterday, 2 days ago, last 7 days, "
                     "this week, last week, this month, monday, 2026-11-20 or 2026-11-01..2026-11-15")


def _range(when, since, until):
    s, u = parse_when(when)
    s = date.fromisoformat(since) if since else s
    u = date.fromisoformat(until) if until else u
    return s, u


# ---------- names ----------
def _resolve(conn, schema, kind, name):
    """The registry spelling of a client/project/person, also from a misheard or partial name."""
    rows = conn.execute(_q("select name, aliases from {s}.entities where kind = %s", schema), (kind,)).fetchall()
    keys = {}
    for n, aliases in rows:
        for a in [n, *aliases]:
            keys.setdefault(norm(a), n)
    k = norm(name)
    if k in keys:
        return keys[k]
    starts = sorted({v for key, v in keys.items() if key.startswith(k)}) if len(k) >= 3 else []
    if len(starts) == 1:
        return starts[0]
    close = difflib.get_close_matches(k, list(keys), n=1, cutoff=0.8)
    return keys[close[0]] if close else name


def _label_filters(conn, schema, client, project, person):
    keys, resolved = [], {}
    for kind, val in (("client", client), ("project", project), ("person", person)):
        if val:
            canonical = _resolve(conn, schema, kind, val)
            resolved[kind] = canonical
            keys.append(label_key(kind, canonical))
    return keys, resolved


def registry(workspace=None, kind=None):
    with connect() as conn:
        ws_id, ws_name, schema = _workspace(conn, workspace)
        args, where = [], ""
        if kind:
            where, args = "where kind = %s", [kind]
        rows = conn.execute(_q(f"""select kind, name, aliases, status, org, role, client, note, first_seen, last_seen
                                    from {{s}}.entities {where} order by kind, name""", schema), args).fetchall()
    cols = ["kind", "name", "heard_as", "status", "org", "role", "client", "note", "first_seen", "last_seen"]
    out = []
    for r in rows:
        e = {c: v for c, v in zip(cols, r) if v not in (None, "", [])}
        for c in ("first_seen", "last_seen"):
            if c in e:
                e[c] = e[c].isoformat()
        out.append(e)
    return {"workspace": ws_name, "entities": out}


# ---------- meetings ----------
MEETING_COLS = "name, title, held, started_at, type, clients, projects, people, topics, duration_s, summary_model"


def _meeting_row(r):
    name, title, held, started, mtype, clients, projects, people, topics, dur, model = r
    return {"meeting": name, "title": title, "date": held.isoformat() if held else None,
            "started": started.strftime("%Y-%m-%d %H:%M") if started else None,
            "weekday": held.strftime("%A") if held else None, "type": mtype,
            "duration_min": round(dur / 60) if dur else None, "clients": clients, "projects": projects,
            "people": people, "topics": topics}


def list_meetings(workspace=None, when=None, since=None, until=None, client=None, project=None, person=None,
                  meeting_type=None, limit=50):
    s, u = _range(when, since, until)
    with connect() as conn:
        ws_id, ws_name, schema = _workspace(conn, workspace)
        keys, resolved = _label_filters(conn, schema, client, project, person)
        where, args = [], []
        if keys:
            where.append("label_keys @> %s"); args.append(keys)
        if s:
            where.append("held >= %s"); args.append(s)
        if u:
            where.append("held <= %s"); args.append(u)
        if meeting_type:
            where.append("lower(type) = lower(%s)"); args.append(meeting_type)
        rows = conn.execute(_q(f"select {MEETING_COLS} from {{s}}.meetings "
                               f"{'where ' + ' and '.join(where) if where else ''} "
                               "order by started_at desc nulls last limit %s", schema), [*args, int(limit)]).fetchall()
    return {"workspace": ws_name, "today": today().isoformat(),
            "range": {"from": s and s.isoformat(), "to": u and u.isoformat()}, "filters_resolved_to": resolved,
            "meetings": [_meeting_row(r) for r in rows]}


def _find_meeting(conn, schema, meeting):
    """A meeting by its recording name, or by a unique part of its title or name."""
    if not meeting:
        raise StoreError("pass `meeting`: its name from list_meetings")
    row = conn.execute(_q(f"select {MEETING_COLS} from {{s}}.meetings where name = %s", schema), (meeting,)).fetchone()
    if row:
        return row
    rows = conn.execute(_q(f"select {MEETING_COLS} from {{s}}.meetings where title ilike %s or name ilike %s "
                           "order by started_at desc", schema), (f"%{meeting}%", f"%{meeting}%")).fetchall()
    if len(rows) == 1:
        return rows[0]
    if not rows:
        raise StoreError(f"no meeting matching {meeting!r} in this workspace (see list_meetings)")
    raise StoreError(f"{meeting!r} matches {len(rows)} meetings; use one of: "
                     + "; ".join(f"{r[0]} ({r[1]}, {r[2]})" for r in rows[:10]))


def get_meeting(meeting, workspace=None):
    with connect() as conn:
        ws_id, ws_name, schema = _workspace(conn, workspace)
        row = _find_meeting(conn, schema, meeting)
        notes = conn.execute(_q("select kind, text, owner, due from {s}.shared_passages where meeting = %s and kind <> 'transcript' "
                                "order by seq", schema), (row[0],)).fetchall()
        n_lines, speakers = conn.execute(_q("select count(*), array_agg(distinct speaker) from {s}.shared_lines where meeting = %s",
                                            schema), (row[0],)).fetchone()
    out = {"workspace": ws_name, **_meeting_row(row), "summary": [], "decisions": [], "action_items": [],
           "open_questions": [], "transcript": {"lines": n_lines, "speakers": sorted(speakers or []),
                                                "note": "speakers are SPEAKER_xx labels, not names; read it with get_transcript"}}
    for kind, text, owner, due in notes:
        if kind == "summary":
            out["summary"].append(text)
        elif kind == "decision":
            out["decisions"].append(text)
        elif kind == "action":
            out["action_items"].append({"owner": owner or "unassigned", "task": text, "due": due})
        elif kind == "question":
            out["open_questions"].append(text)
    return out


def _mmss(s):
    return f"{int(s // 60):02d}:{int(s % 60):02d}" if s is not None else "--:--"


def get_transcript(meeting, workspace=None, from_minute=None, to_minute=None, max_chars=60000):
    """The transcript as `[mm:ss] SPEAKER: text` lines, in pages of at most `max_chars`."""
    with connect() as conn:
        ws_id, ws_name, schema = _workspace(conn, workspace)
        row = _find_meeting(conn, schema, meeting)
        where, args = ["meeting = %s"], [row[0]]
        if from_minute is not None:
            where.append("start_s >= %s"); args.append(float(from_minute) * 60)
        if to_minute is not None:
            where.append("start_s < %s"); args.append(float(to_minute) * 60)
        lines = conn.execute(_q(f"select start_s, speaker, text from {{s}}.shared_lines where {' and '.join(where)} order by seq",
                                schema), args).fetchall()
    text, more = [], None
    size = 0
    for start, speaker, t in lines:
        ln = f"[{_mmss(start)}] {speaker}: {t}"
        if size + len(ln) > max(2000, int(max_chars)) and text:
            more = round(start / 60, 2)
            break
        text.append(ln)
        size += len(ln) + 1
    return {"workspace": ws_name, "meeting": row[0], "title": row[1], "date": row[2] and row[2].isoformat(),
            "people_in_meeting": row[7], "lines": len(text), "transcript": "\n".join(text),
            "continue_from_minute": more,
            "note": "Speaker labels are voices, not names; people_in_meeting lists who was there or mentioned."
                    + (" The transcript continues: call again with from_minute=continue_from_minute." if more else "")}


def search(query, workspace=None, kind=None, client=None, project=None, person=None, when=None, since=None,
           until=None, k=10):
    if not query or not query.strip():
        raise StoreError("pass a `query`")
    s, u = _range(when, since, until)
    kinds = [kind] if isinstance(kind, str) and kind else list(kind or [])
    if set(kinds) - set(KINDS):
        raise StoreError(f"kind must be among {', '.join(KINDS)}")
    with connect() as conn:
        ws_id, ws_name, schema = _workspace(conn, workspace)
        model = conn.execute("select value from public.index_info where key = 'embed_model'").fetchone()[0]
        keys, resolved = _label_filters(conn, schema, client, project, person)
    try:
        import numpy as np
        vec = np.array(embed([query], EMBED_URL, model, query=True)[0], dtype=np.float32)
    except EmbedError as e:
        raise StoreError(str(e))
    where, args = [], []
    if keys:
        where.append("m.label_keys @> %s"); args.append(keys)
    if kinds:
        where.append("p.kind = any(%s)"); args.append(kinds)
    if s:
        where.append("m.held >= %s"); args.append(s)
    if u:
        where.append("m.held <= %s"); args.append(u)
    sql = _q(f"""
        with hits as materialized (
          select m.name, m.title, m.held, p.kind, p.text, p.owner, p.due, p.start_s, p.end_s,
                 p.embedding <=> %s as distance
          from {{s}}.shared_passages p join {{s}}.meetings m on m.name = p.meeting
          {"where " + " and ".join(where) if where else ""}
          order by p.embedding <=> %s limit %s)
        select * from hits order by distance""", schema)
    with connect() as conn, conn.transaction():
        conn.execute("set local hnsw.ef_search = 100")
        conn.execute("set local hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(sql, [vec, *args, vec, min(int(k), 50)]).fetchall()
    return {"workspace": ws_name, "filters_resolved_to": resolved, "results": [
        {"meeting": name, "title": title, "date": held.isoformat() if held else None, "kind": kind_,
         "text": text, **({"owner": owner} if owner else {}), **({"due": due} if due else {}),
         **({"at": f"{_mmss(st)}–{_mmss(en)}", "from_minute": round(st / 60, 2)} if kind_ == "transcript" else {}),
         "relevance": round(1 - float(dist), 3)}
        for name, title, held, kind_, text, owner, due, st, en, dist in rows]}


def action_items(workspace=None, when=None, since=None, until=None, owner=None, client=None, project=None):
    s, u = _range(when, since, until)
    with connect() as conn:
        ws_id, ws_name, schema = _workspace(conn, workspace)
        keys, resolved = _label_filters(conn, schema, client, project, None)
        where, args = ["p.kind = 'action'"], []
        if keys:
            where.append("m.label_keys @> %s"); args.append(keys)
        if s:
            where.append("m.held >= %s"); args.append(s)
        if u:
            where.append("m.held <= %s"); args.append(u)
        if owner:
            who = _resolve(conn, schema, "person", owner)
            resolved["owner"] = who
            where.append("(p.owner ilike %s or p.owner ilike %s)"); args += [f"%{who}%", f"%{owner}%"]
        rows = conn.execute(_q(f"""select m.name, m.title, m.held, p.owner, p.text, p.due
                                    from {{s}}.shared_passages p join {{s}}.meetings m on m.name = p.meeting
                                    where {' and '.join(where)} order by m.started_at desc, p.seq""", schema),
                            args).fetchall()
    return {"workspace": ws_name, "range": {"from": s and s.isoformat(), "to": u and u.isoformat()},
            "filters_resolved_to": resolved, "count": len(rows),
            "action_items": [{"owner": o or "unassigned", "task": t, "due": d, "meeting": n, "title": ti,
                              "date": h.isoformat() if h else None} for n, ti, h, o, t, d in rows],
            "note": "Action items as recorded in each meeting's summary; the index does not track completion."}
