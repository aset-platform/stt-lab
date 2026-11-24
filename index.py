"""The meeting index: summaries and transcripts as vectors in Postgres (pgvector), with their labels.

Each workspace (firm) has its own index: a Postgres schema ws_<id> with its own tables and HNSW index.
Searches run inside one workspace; nothing is shared or mixed between firms.

Only meetings whose labels you confirmed are indexed, so the client / project / person filters can be
trusted. A meeting becomes:
  - one row in `meetings`: date, title, type, clients, projects, people, topics
  - rows in `passages`: each summary bullet, decision, action item and open question, and the
    transcript in passages of about a minute (with start/end time and speakers), each with an embedding

Embeddings come from Ollama (settings "index" section; qwen3-embedding:0.6b by default, 0.6 GB, so it fits
next to the summariser). The database runs from docker-compose.yml (./stt-lab up).

    uv run index.py status                    # database, embedding model, what is indexed / pending
    uv run index.py backfill                  # index every confirmed meeting that is new or changed
    uv run index.py search "what did the client ask for" [--workspace ID] [--client X] [--person X]
                                              [--project X] [--kind action] [--since 2026-11-01] [-k 8]
    uv run index.py meeting NAME              # index one meeting now
    uv run index.py remove NAME
    uv run index.py drop-workspace ID         # remove one firm's whole index
"""
import argparse
import json
import os
import re
import threading
import urllib.request
from datetime import date, datetime

import library
import labels
import settings
from mcp_server.shared import KINDS, READONLY_ROLE, EmbedError, label_key, schema_name
from mcp_server import shared

SCHEMA_VERSION = "4"
_init_lock = threading.Lock()
_ready = set()  # database urls whose schema is known to be in place


class IndexError_(library.LibraryError):
    pass


def config():
    cfg = dict(settings.get("index"))
    cfg["database_url"] = os.environ.get("STT_DB_URL") or cfg["database_url"]
    cfg["mcp_db_password"] = os.environ.get("STT_MCP_DB_PASSWORD") or cfg["mcp_db_password"]
    return cfg


# ---------- embeddings ----------
def embed(texts, cfg=None, query=False):
    cfg = cfg or config()
    try:
        return shared.embed(texts, cfg["embed_url"], cfg["embed_model"], query)
    except EmbedError as e:
        raise IndexError_(str(e))


# ---------- database ----------
# One Postgres schema per workspace (ws_<id>), each with its own meetings and passages tables and its own
# HNSW index: firms never share a table or a vector index, a search runs inside one workspace only, and a
# firm's data can be dropped as a whole. public.index_info holds what all of them share (embedding model).
def schema_of(ws):
    if not labels.exists(ws):
        raise IndexError_(f"no workspace {ws!r}")
    return schema_name(ws)


def _ws_of_schema(schema):
    return schema[3:].replace("_", "-")


def _sql(text, schema):
    from psycopg import sql
    return sql.SQL(text).format(s=sql.Identifier(schema))


def connect(cfg=None):
    import psycopg
    from pgvector.psycopg import register_vector

    cfg = cfg or config()
    try:
        conn = psycopg.connect(cfg["database_url"], connect_timeout=3)
    except psycopg.OperationalError as e:
        raise IndexError_(f"index database not reachable ({e.__class__.__name__}); start it with ./stt-lab up")
    with _init_lock:
        if cfg["database_url"] not in _ready:
            _init(conn, cfg)
            _ready.add(cfg["database_url"])
    register_vector(conn)
    return conn


def _schemas(conn):
    """{workspace id: schema} of the workspaces that have an index."""
    rows = conn.execute("select schema_name from information_schema.schemata "
                        "where schema_name like 'ws\\_%' order by 1").fetchall()
    return {_ws_of_schema(r[0]): r[0] for r in rows}


def _init(conn, cfg):
    """Shared set-up. The vector size comes from the embedding model; another model needs a rebuild
    (reset), because vectors of two models can't be compared."""
    with conn.transaction():
        conn.execute("create extension if not exists vector")
        conn.execute("create table if not exists index_info (key text primary key, value text not null)")
        info = dict(conn.execute("select key, value from index_info").fetchall())
        if info and info.get("embed_model") != cfg["embed_model"]:
            raise IndexError_(f"the index was built with {info.get('embed_model')}, settings say "
                              f"{cfg['embed_model']}: run `uv run index.py reset` and then backfill")
        if info and info.get("schema") == SCHEMA_VERSION:
            return
        # first run, or an older layout: the index is derived data, so rebuild; the next sync refills it
        conn.execute("drop table if exists public.passages, public.meetings")  # layout 1-2: one shared table
        for schema in _schemas(conn).values():
            conn.execute(_sql("drop schema {s} cascade", schema))
        conn.execute("delete from index_info")
        conn.execute("drop table if exists public.workspaces")
        conn.execute("""create table public.workspaces (id text primary key, name text not null,
                        schema text not null, active boolean not null default false,
                        updated_at timestamptz not null default now())""")
        # the MCP server logs in as this role: it can read the index and nothing else
        from psycopg import sql
        exists = conn.execute("select 1 from pg_roles where rolname = %s", (READONLY_ROLE,)).fetchone()
        conn.execute(sql.SQL("{} role {} login password {}").format(
            sql.SQL("alter" if exists else "create"), sql.Identifier(READONLY_ROLE), sql.Literal(cfg["mcp_db_password"])))
        conn.execute(_sql("grant select on public.workspaces, public.index_info to {s}", READONLY_ROLE))
        dim = len(embed(["dimension check"], cfg)[0])
        with conn.cursor() as cur:
            cur.executemany("insert into index_info values (%s, %s)",
                            [("schema", SCHEMA_VERSION), ("embed_model", cfg["embed_model"]), ("dim", str(dim))])


def _ensure(conn, ws):
    """The workspace's schema, created on first use."""
    schema = schema_of(ws)
    dim = int(conn.execute("select value from index_info where key = 'dim'").fetchone()[0])
    conn.execute(_sql("create schema if not exists {s}", schema))
    conn.execute(_sql("""
        create table if not exists {s}.meetings (
          name          text primary key,
          held          date,
          started_at    timestamp,                  -- local time the recording started
          title         text,
          type          text,
          clients       text[] not null default '{{}}',
          projects      text[] not null default '{{}}',
          people        text[] not null default '{{}}',
          topics        text[] not null default '{{}}',
          label_keys    text[] not null default '{{}}',  -- 'client:acme', 'person:ravi': exact filters
          duration_s    real,
          summary_model text,
          labels_confirmed_at text,
          summary_mtime double precision,
          chunking      text,
          indexed_at    timestamptz not null default now())""", schema))
    conn.execute(_sql(f"""
        create table if not exists {{s}}.passages (
          id        bigserial primary key,
          meeting   text not null references {{s}}.meetings(name) on delete cascade on update cascade,
          kind      text not null,
          seq       int not null,
          text      text not null,
          owner     text,
          due       text,
          start_s   real,
          end_s     real,
          speakers  text[],
          embedding vector({dim}) not null)""", schema))
    conn.execute(_sql("""
        create table if not exists {s}.lines (      -- the transcript as spoken, for reading in full
          meeting  text not null references {s}.meetings(name) on delete cascade on update cascade,
          seq      int not null,
          start_s  real,
          end_s    real,
          speaker  text,
          text     text not null,
          primary key (meeting, seq))""", schema))
    conn.execute(_sql("""
        create table if not exists {s}.entities (   -- the workspace registry, to resolve misheard names
          kind text not null, name text not null, aliases text[] not null default '{{}}',
          status text, note text, org text, role text, client text, first_seen date, last_seen date,
          primary key (kind, name))""", schema))
    conn.execute(_sql("create index if not exists passages_embedding on {s}.passages "
                      "using hnsw (embedding vector_cosine_ops)", schema))
    conn.execute(_sql("create index if not exists passages_meeting on {s}.passages (meeting)", schema))
    conn.execute(_sql("create index if not exists meetings_held on {s}.meetings (held)", schema))
    conn.execute(_sql("create index if not exists meetings_label_keys on {s}.meetings using gin (label_keys)", schema))
    conn.execute(_sql("create index if not exists meetings_started on {s}.meetings (started_at)", schema))
    from psycopg import sql
    ro = sql.Identifier(READONLY_ROLE)
    conn.execute(sql.SQL("grant usage on schema {s} to {r}").format(s=sql.Identifier(schema), r=ro))
    conn.execute(sql.SQL("grant select on all tables in schema {s} to {r}").format(s=sql.Identifier(schema), r=ro))
    return schema


def reset():
    """Drop the whole index (e.g. after changing the embedding model). Backfill re-adds the meetings."""
    cfg = config()
    with connect(cfg) as conn, conn.transaction():
        for schema in _schemas(conn).values():
            conn.execute(_sql("drop schema {s} cascade", schema))
        conn.execute("drop table if exists index_info, workspaces")
    _ready.discard(cfg["database_url"])


def drop_workspace(ws):
    """Remove one workspace's index (e.g. when work with that firm ends)."""
    with connect() as conn, conn.transaction():
        schema = _schemas(conn).get(ws)
        if schema:
            conn.execute(_sql("drop schema {s} cascade", schema))
        return bool(schema)


# ---------- what goes in ----------
def _started(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2})-(\d{2})-(\d{2})-(\d{2})", name)
    if m:
        return datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}")
    return datetime.fromtimestamp(library.audio_path(name).stat().st_mtime).replace(microsecond=0)


def _held(name):
    m = re.search(r"\d{4}-\d{2}-\d{2}", name)
    if m:
        return m.group(0)
    return datetime.fromtimestamp(library.audio_path(name).stat().st_mtime).date().isoformat()


def _chunking(cfg):
    """Fingerprint of the chunking settings: a meeting indexed with other settings is re-indexed."""
    return f"t{cfg['chunk_seconds']}o{cfg['chunk_overlap_seconds']}c{cfg['chunk_max_chars']}"


def _split_long(ln, max_chars):
    """A line longer than a passage may hold is split at sentence ends (then spaces), keeping its time."""
    text = ln["text"].strip()
    if len(text) <= max_chars:
        return [ln]
    parts, cur = [], ""
    for piece in re.split(r"(?<=[.!?])\s+", text):
        while len(piece) > max_chars:  # one endless "sentence": cut at the last space that fits
            cut = piece.rfind(" ", 0, max_chars) if " " in piece[:max_chars] else max_chars
            parts.append(piece[:cut]); piece = piece[cut:].strip()
        if cur and len(cur) + 1 + len(piece) > max_chars:
            parts.append(cur); cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        parts.append(cur)
    return [{**ln, "text": p} for p in parts]


def _transcript_passages(name, seconds, overlap, max_chars=2000, min_chars=200):
    """Windows of whole lines, each about `seconds` long and at most `max_chars`, the next one starting
    `overlap` seconds before this one ends (never mid-line, always moving forward). A line too long for a
    passage is split at sentence ends; a last window under `min_chars` joins the one before it."""
    lines = [part for ln in (library.best_transcript(name) or {}).get("lines", []) if ln.get("text", "").strip()
             for part in _split_long(ln, max_chars)]
    size = lambda a, b: sum(len(l["text"]) + 14 for l in lines[a:b])  # + "SPEAKER_00: " and newline
    spans, i = [], 0
    while i < len(lines):
        j = i + 1
        while (j < len(lines) and lines[j]["start"] - lines[i]["start"] < seconds
               and size(i, j + 1) <= max_chars):
            j += 1
        spans.append((i, j))
        if j >= len(lines):
            break
        cut = lines[j]["start"] - overlap  # the next window starts with the lines spoken in the last `overlap` s
        i = next((k for k in range(i + 1, j) if lines[k]["start"] >= cut), j)
    if len(spans) > 1 and size(*spans[-1]) < min_chars and size(spans[-2][0], spans[-1][1]) <= max_chars:
        spans[-2:] = [(spans[-2][0], spans[-1][1])]
    return [{"kind": "transcript", "text": "\n".join(f"{l['speaker']}: {l['text'].strip()}" for l in g),
             "start_s": g[0]["start"], "end_s": g[-1].get("end", g[-1]["start"]),
             "speakers": sorted({l["speaker"] for l in g})} for g in (lines[a:b] for a, b in spans)]


def document(name, cfg=None):
    """The workspace, meeting row and passages for `name`, or raise if it is not ready to be indexed."""
    cfg = cfg or config()
    library.audio_path(name)
    js = library.LLM_SUMMARIES / f"{name}.json"
    lab = labels.get(name)
    ws = labels.of(name)
    if not js.is_file():
        raise IndexError_(f"{name} has no summary yet")
    if not lab or lab.get("status") != "confirmed":
        raise IndexError_(f"{name}: confirm its labels first (only confirmed meetings are indexed)")
    if not ws:
        raise IndexError_(f"{name} is in no workspace; move it to one (every workspace has its own index)")
    notes = json.loads(js.read_text())
    md = library.LLM_SUMMARIES / f"{name}.md"
    model = re.search(r"summarised by (.+?)_\n", md.read_text()) if md.is_file() else None
    t = library.best_transcript(name)
    meeting = {
        "name": name, "held": _held(name), "started_at": _started(name),
        "title": notes.get("title") or name, "type": lab.get("type"),
        **{f: lab.get(f, []) for f in ("clients", "projects", "people", "topics")},
        "label_keys": [label_key(k, n) for k, f in labels.KINDS.items() for n in lab.get(f, [])],
        "duration_s": (t or {}).get("stats", {}).get("duration"),
        "summary_model": model.group(1) if model else None,
        "labels_confirmed_at": lab.get("confirmed_at"), "summary_mtime": js.stat().st_mtime,
        "chunking": _chunking(cfg),
    }
    passages = [{"kind": "summary", "text": s} for s in notes.get("summary", [])]
    passages += [{"kind": "decision", "text": s} for s in notes.get("decisions", [])]
    passages += [{"kind": "action", "text": a.get("task", ""), "owner": a.get("owner"), "due": a.get("due")}
                 for a in notes.get("action_items", []) if isinstance(a, dict)]
    passages += [{"kind": "question", "text": s} for s in notes.get("open_questions", [])]
    passages = [p for p in passages if isinstance(p["text"], str) and p["text"].strip()]
    passages += _transcript_passages(name, cfg["chunk_seconds"], cfg["chunk_overlap_seconds"], cfg["chunk_max_chars"])
    lines = [{"start_s": l.get("start"), "end_s": l.get("end"), "speaker": l.get("speaker"), "text": l["text"].strip()}
             for l in (t or {}).get("lines", []) if l.get("text", "").strip()]
    return ws, meeting, passages, lines


def _embed_text(meeting, p):
    """What is embedded: the passage with its meeting's context, so "the Acme demo" finds the right one."""
    who = f" ({p['owner']})" if p.get("owner") else ""
    ctx = ", ".join(meeting["clients"] + meeting["projects"])
    return f"{meeting['title']} · {meeting['held']}" + (f" · {ctx}" if ctx else "") + \
           f"\n{p['kind']}{who}: {p['text']}"


def index_one(name):
    """(Re)index one meeting in its workspace's index (and out of any other). Returns the passage count."""
    cfg = config()
    ws, meeting, passages, lines = document(name, cfg)
    vectors = embed([_embed_text(meeting, p) for p in passages], cfg)
    cols = list(meeting)
    with connect(cfg) as conn, conn.transaction():
        for other in _schemas(conn).values():  # it may have moved from another workspace
            conn.execute(_sql("delete from {s}.meetings where name = %s", other), (name,))
        schema = _ensure(conn, ws)
        conn.execute(_sql(f"insert into {{s}}.meetings ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))})",
                          schema), [meeting[c] for c in cols])
        with conn.cursor() as cur:
            cur.executemany(
                _sql("insert into {s}.passages (meeting, kind, seq, text, owner, due, start_s, end_s, speakers,"
                     " embedding) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", schema),
                [(name, p["kind"], i, p["text"], p.get("owner"), p.get("due"), p.get("start_s"), p.get("end_s"),
                  p.get("speakers"), _vec(v)) for i, (p, v) in enumerate(zip(passages, vectors))])
            cur.executemany(
                _sql("insert into {s}.lines (meeting, seq, start_s, end_s, speaker, text) values (%s, %s, %s, %s, %s, %s)",
                     schema),
                [(name, i, l["start_s"], l["end_s"], l["speaker"], l["text"]) for i, l in enumerate(lines)])
    return len(passages)


def _vec(v):
    import numpy as np
    return np.array(v, dtype=np.float32)


def remove(name, ws=None):
    """Take a meeting out of the index (of `ws` only, or of every workspace)."""
    with connect() as conn, conn.transaction():
        schemas = _schemas(conn)
        targets = [schemas[ws]] if ws else list(schemas.values())
        return sum(conn.execute(_sql("delete from {s}.meetings where name = %s", s), (name,)).rowcount
                   for s in targets if s)


def publish_registries():
    """Copy the workspaces (and which is active) and each registry into the database, so the MCP server
    can list workspaces and resolve misheard names without reading files on the Mac."""
    active = labels.active()
    with connect() as conn, conn.transaction():
        conn.execute("delete from public.workspaces")
        for w in labels.workspaces():
            schema = _ensure(conn, w["id"])
            conn.execute("insert into public.workspaces (id, name, schema, active) values (%s, %s, %s, %s)",
                         (w["id"], w["name"], schema, w["id"] == active))
            conn.execute(_sql("delete from {s}.entities", schema))
            with conn.cursor() as cur:
                cur.executemany(
                    _sql("insert into {s}.entities (kind, name, aliases, status, note, org, role, client, first_seen,"
                         " last_seen) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing", schema),
                    [(e["kind"], e["name"], e.get("aliases") or [], e.get("status"), e.get("note"), e.get("org"),
                      e.get("role"), e.get("client"), e.get("first_seen"), e.get("last_seen"))
                     for e in labels.load(w["id"])["entities"]])


# ---------- what needs doing ----------
def pending():
    """(to index, to remove, unplaced): confirmed meetings (not in Trash) missing from their workspace's
    index or changed since (labels, summary, chunking settings); indexed (workspace, meeting) pairs that
    should not be there (Trash, moved to another workspace, labels reopened); confirmed meetings in no
    workspace, which are not indexed."""
    have = {}
    with connect() as conn:
        for ws, schema in _schemas(conn).items():
            for r in conn.execute(_sql("select name, labels_confirmed_at, summary_mtime, chunking from {s}.meetings",
                                       schema)).fetchall():
                have[(ws, r[0])] = (r[1], r[2], r[3])
    chunking = _chunking(config())
    todo, ready, unplaced = [], set(), []
    for it in library.items():
        if it["folder"] == "trash" or not it["summary"] or it["labels"] != "confirmed":
            continue
        ws = labels.of(it["name"])
        if not ws:
            unplaced.append(it["name"])
            continue
        ready.add((ws, it["name"]))
        lab = labels.get(it["name"])
        mtime = (library.LLM_SUMMARIES / f"{it['name']}.json").stat().st_mtime
        if have.get((ws, it["name"])) != (lab.get("confirmed_at"), mtime, chunking):
            todo.append(it["name"])
    return todo, sorted(set(have) - ready), unplaced


def sync(log=print):
    """Index what is pending and drop what should not be indexed. Returns (indexed, removed, failed)."""
    publish_registries()
    todo, stale, _ = pending()
    for ws, name in stale:  # first, so a meeting moved between workspaces is never in two
        remove(name, ws)
        log(f"- {name}: removed from {ws} (Trash, another workspace, or labels no longer confirmed)")
    done, failed = 0, []
    for name in todo:
        try:
            n = index_one(name)
            done += 1
            log(f"✓ {name}: {n} passages")
        except Exception as e:
            failed.append(name)
            log(f"✗ {name}: {type(e).__name__}: {e}")
    return done, len(stale), failed


def status():
    cfg = config()
    out = {"enabled": cfg["enabled"], "embed_model": cfg["embed_model"], "ok": False}
    try:
        todo, stale, unplaced = pending()
        with connect(cfg) as conn:
            by_ws = {ws: {"meetings": conn.execute(_sql("select count(*) from {s}.meetings", s)).fetchone()[0],
                          "passages": conn.execute(_sql("select count(*) from {s}.passages", s)).fetchone()[0]}
                     for ws, s in _schemas(conn).items()}
        out.update(ok=True, by_workspace=by_ws, pending=todo, stale=stale, unplaced=unplaced,
                   meetings=sum(v["meetings"] for v in by_ws.values()),
                   passages=sum(v["passages"] for v in by_ws.values()))
    except Exception as e:
        out["message"] = str(e)
    return out


# ---------- search ----------
def search(q, workspace, client=None, project=None, person=None, kind=None, since=None, until=None, k=8):
    """Passages of one workspace closest in meaning to `q`, best first, filtered by labels and dates.
    There is no search across workspaces: each firm's meetings stay in their own index."""
    if not workspace:
        raise IndexError_("choose a workspace: each workspace has its own index")
    schema_of(workspace)  # must be a real workspace
    where, args, keys = [], [], []
    reg = labels.load(workspace)
    for ent, val in (("client", client), ("project", project), ("person", person)):
        if val:  # a misheard or differently written name finds the registry entry it belongs to
            keys.append(label_key(ent, labels.resolve(reg, ent, val)[0] or val))
    if keys:
        where.append("m.label_keys @> %s"); args.append(keys)  # uses the GIN index
    if kind:
        kinds = [kind] if isinstance(kind, str) else list(kind)
        if set(kinds) - set(KINDS):
            raise IndexError_(f"kind must be among {', '.join(KINDS)}")
        where.append("p.kind = any(%s)"); args.append(kinds)
    if since:
        where.append("m.held >= %s"); args.append(date.fromisoformat(str(since)))
    if until:
        where.append("m.held <= %s"); args.append(date.fromisoformat(str(until)))
    with connect() as conn:
        schema = _schemas(conn).get(workspace)
    if not schema:
        return []  # nothing confirmed in this workspace yet
    vec = _vec(embed([q], query=True)[0])
    # With filters, HNSW alone finds the ef_search nearest passages and filters afterwards, which can leave
    # fewer than k. Iterative scan keeps walking the graph until k rows pass the filters; its relaxed
    # order is put right by the outer sort.
    query = _sql(f"""
        with hits as materialized (
          select m.name, m.title, m.held, m.type, m.clients, m.projects, m.people,
                 p.kind, p.text, p.owner, p.due, p.start_s, p.end_s, p.embedding <=> %s as distance
          from {{s}}.passages p join {{s}}.meetings m on m.name = p.meeting
          {"where " + " and ".join(where) if where else ""}
          order by p.embedding <=> %s limit %s)
        select *, 1 - distance as score from hits order by distance""", schema)
    with connect() as conn, conn.transaction():
        conn.execute("set local hnsw.ef_search = 100")
        conn.execute("set local hnsw.iterative_scan = relaxed_order")
        cur = conn.execute(query, [vec, *args, vec, int(k)])
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["workspace"] = workspace
        r["held"] = r["held"].isoformat() if r["held"] else None
        r["score"] = round(float(r.pop("score")), 3)
        r.pop("distance", None)
    return rows


def _fmt_time(s):
    return f"{int(s // 60)}:{int(s % 60):02d}" if s is not None else ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["status", "backfill", "search", "meeting", "remove", "reset",
                                        "drop-workspace"])
    p.add_argument("args", nargs="*")
    p.add_argument("--workspace"); p.add_argument("--client"); p.add_argument("--project")
    p.add_argument("--person"); p.add_argument("--kind"); p.add_argument("--since"); p.add_argument("--until")
    p.add_argument("-k", type=int, default=8)
    a = p.parse_args()
    if a.command == "status":
        print(json.dumps(status(), indent=1, ensure_ascii=False))
    elif a.command == "backfill":
        done, removed, failed = sync()
        print(f"indexed {done}, removed {removed}, failed {len(failed)}")
    elif a.command == "search":
        for r in search(" ".join(a.args), a.workspace or labels.active(), a.client, a.project, a.person, a.kind, a.since, a.until, a.k):
            where = f" @ {_fmt_time(r['start_s'])}" if r["kind"] == "transcript" else ""
            owner = f" [{r['owner']}]" if r["owner"] else ""
            print(f"{r['score']:.3f}  {r['held']}  {r['title']}  · {r['kind']}{owner}{where}")
            print("       " + r["text"].replace("\n", "\n       ")[:400])
    elif a.command == "meeting":
        for name in a.args:
            print(f"✓ {name}: {index_one(name)} passages")
    elif a.command == "remove":
        for name in a.args:
            print(f"{name}: {'removed' if remove(name) else 'not indexed'}")
    elif a.command == "drop-workspace":
        for ws in a.args:
            print(f"{ws}: {'index dropped' if drop_workspace(ws) else 'had no index'}")
    elif a.command == "reset":
        reset()
        print("index dropped; run `uv run index.py backfill`")


if __name__ == "__main__":
    main()
