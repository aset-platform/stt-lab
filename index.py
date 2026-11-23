"""The meeting index: summaries and transcripts as vectors in Postgres (pgvector), with their labels.

Only meetings whose labels you confirmed are indexed, so the client / project / person filters can be
trusted. A meeting becomes:
  - one row in `meetings`: workspace, date, title, type, clients, projects, people, topics
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

SCHEMA_VERSION = "2"
QUERY_INSTRUCTION = ("Instruct: Given a question about past meetings, retrieve the meeting notes and "
                     "transcript passages that answer it\nQuery: ")
KINDS = ("summary", "decision", "action", "question", "transcript")
_init_lock = threading.Lock()
_ready = set()  # database urls whose schema is known to be in place


class IndexError_(library.LibraryError):
    pass


def config():
    cfg = dict(settings.get("index"))
    cfg["database_url"] = os.environ.get("STT_DB_URL") or cfg["database_url"]
    return cfg


# ---------- embeddings ----------
def embed(texts, cfg=None, query=False):
    """Vectors for `texts` from Ollama's /api/embed, in batches."""
    cfg = cfg or config()
    out = []
    for i in range(0, len(texts), 32):
        batch = [(QUERY_INSTRUCTION + t) if query else t for t in texts[i:i + 32]]
        req = urllib.request.Request(cfg["embed_url"].rstrip("/") + "/api/embed",
                                     data=json.dumps({"model": cfg["embed_model"], "input": batch,
                                                      "keep_alive": "5m"}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out += json.loads(r.read())["embeddings"]
        except OSError as e:
            raise IndexError_(f"embedding model {cfg['embed_model']} not reachable at {cfg['embed_url']}: {e}")
    return out


# ---------- database ----------
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


def _init(conn, cfg):
    """Create the schema. The vector size comes from the embedding model; a different model needs a
    rebuild (reset), because vectors of two models can't be compared."""
    with conn.transaction():
        conn.execute("create extension if not exists vector")
        conn.execute("create table if not exists index_info (key text primary key, value text not null)")
        info = dict(conn.execute("select key, value from index_info").fetchall())
        if info and info.get("embed_model") != cfg["embed_model"]:
            raise IndexError_(f"the index was built with {info.get('embed_model')}, settings say "
                              f"{cfg['embed_model']}: run `uv run index.py reset` and then backfill")
        if info and info.get("schema") != SCHEMA_VERSION:
            # the index is derived data: rebuild it in the new layout; the next sync refills it
            conn.execute("drop table if exists passages, meetings")
            conn.execute("delete from index_info")
            info = {}
        if info:
            return
        dim = len(embed(["dimension check"], cfg)[0])
        conn.execute("""
            create table if not exists meetings (
              name          text primary key,
              workspace     text not null default '',
              held          date,
              title         text,
              type          text,
              clients       text[] not null default '{}',
              projects      text[] not null default '{}',
              people        text[] not null default '{}',
              topics        text[] not null default '{}',
              duration_s    real,
              summary_model text,
              labels_confirmed_at text,
              summary_mtime double precision,
              chunking      text,
              label_keys    text[] not null default '{}',  -- 'client:acme', 'person:ravi': exact filters
              indexed_at    timestamptz not null default now())""")
        conn.execute(f"""
            create table if not exists passages (
              id        bigserial primary key,
              meeting   text not null references meetings(name) on delete cascade on update cascade,
              kind      text not null,
              seq       int not null,
              text      text not null,
              owner     text,
              due       text,
              start_s   real,
              end_s     real,
              speakers  text[],
              embedding vector({dim}) not null)""")
        conn.execute("create index if not exists passages_embedding on passages using hnsw (embedding vector_cosine_ops)")
        conn.execute("create index if not exists passages_meeting on passages (meeting)")
        conn.execute("create index if not exists meetings_ws_held on meetings (workspace, held)")
        conn.execute("create index if not exists meetings_label_keys on meetings using gin (label_keys)")
        with conn.cursor() as cur:
            cur.executemany("insert into index_info values (%s, %s)",
                            [("schema", SCHEMA_VERSION), ("embed_model", cfg["embed_model"]), ("dim", str(dim))])


def reset():
    """Drop the index (e.g. after changing the embedding model). Meetings are re-added by backfill."""
    cfg = config()
    with connect(cfg) as conn, conn.transaction():
        conn.execute("drop table if exists passages, meetings, index_info")
    _ready.discard(cfg["database_url"])


# ---------- what goes in ----------
def _held(name):
    m = re.search(r"\d{4}-\d{2}-\d{2}", name)
    if m:
        return m.group(0)
    return datetime.fromtimestamp(library.audio_path(name).stat().st_mtime).date().isoformat()


def _chunking(cfg):
    """Fingerprint of the chunking settings: a meeting indexed with other settings is re-indexed."""
    return f"t{cfg['chunk_seconds']}o{cfg['chunk_overlap_seconds']}"


def label_key(kind, name):
    return f"{kind}:{' '.join(str(name).lower().split())}"


def _transcript_passages(name, seconds, overlap):
    """Windows of whole lines, each about `seconds` long, the next one starting `overlap` seconds before
    this one ends (never mid-line, always moving forward)."""
    lines = [ln for ln in (library.best_transcript(name) or {}).get("lines", []) if ln.get("text", "").strip()]
    out, i = [], 0
    while i < len(lines):
        j = i + 1
        while j < len(lines) and lines[j]["start"] - lines[i]["start"] < seconds:
            j += 1
        out.append(lines[i:j])
        if j >= len(lines):
            break
        cut = lines[j]["start"] - overlap  # the next window starts with the lines spoken in the last `overlap` s
        i = next((k for k in range(i + 1, j) if lines[k]["start"] >= cut), j)
    return [{"kind": "transcript", "text": "\n".join(f"{l['speaker']}: {l['text'].strip()}" for l in g),
             "start_s": g[0]["start"], "end_s": g[-1].get("end", g[-1]["start"]),
             "speakers": sorted({l["speaker"] for l in g})} for g in out]


def document(name, cfg=None):
    """The meeting row and passages for `name`, or raise if it is not ready to be indexed."""
    cfg = cfg or config()
    library.audio_path(name)
    js = library.LLM_SUMMARIES / f"{name}.json"
    lab = labels.get(name)
    if not js.is_file():
        raise IndexError_(f"{name} has no summary yet")
    if not lab or lab.get("status") != "confirmed":
        raise IndexError_(f"{name}: confirm its labels first (only confirmed meetings are indexed)")
    notes = json.loads(js.read_text())
    md = library.LLM_SUMMARIES / f"{name}.md"
    model = re.search(r"summarised by (.+?)_\n", md.read_text()) if md.is_file() else None
    t = library.best_transcript(name)
    meeting = {
        "name": name, "workspace": library.workspace_of(name), "held": _held(name),
        "title": notes.get("title") or name, "type": lab.get("type"),
        **{f: lab.get(f, []) for f in ("clients", "projects", "people", "topics")},
        "duration_s": (t or {}).get("stats", {}).get("duration"),
        "summary_model": model.group(1) if model else None,
        "labels_confirmed_at": lab.get("confirmed_at"), "summary_mtime": js.stat().st_mtime,
        "chunking": _chunking(cfg),
        "label_keys": [label_key(k, n) for k, f in labels.KINDS.items() for n in lab.get(f, [])],
    }
    passages = [{"kind": "summary", "text": s} for s in notes.get("summary", [])]
    passages += [{"kind": "decision", "text": s} for s in notes.get("decisions", [])]
    passages += [{"kind": "action", "text": a.get("task", ""), "owner": a.get("owner"), "due": a.get("due")}
                 for a in notes.get("action_items", []) if isinstance(a, dict)]
    passages += [{"kind": "question", "text": s} for s in notes.get("open_questions", [])]
    passages = [p for p in passages if isinstance(p["text"], str) and p["text"].strip()]
    passages += _transcript_passages(name, cfg["chunk_seconds"], cfg["chunk_overlap_seconds"])
    return meeting, passages


def _embed_text(meeting, p):
    """What is embedded: the passage with its meeting's context, so "the Acme demo" finds the right one."""
    who = f" ({p['owner']})" if p.get("owner") else ""
    ctx = ", ".join(meeting["clients"] + meeting["projects"])
    return f"{meeting['title']} · {meeting['held']}" + (f" · {ctx}" if ctx else "") + \
           f"\n{p['kind']}{who}: {p['text']}"


def index_one(name):
    """(Re)index one meeting. Returns the number of passages."""
    cfg = config()
    meeting, passages = document(name, cfg)
    vectors = embed([_embed_text(meeting, p) for p in passages], cfg)
    cols = list(meeting)
    with connect(cfg) as conn, conn.transaction():
        conn.execute("delete from meetings where name = %s", (name,))
        conn.execute(f"insert into meetings ({', '.join(cols)}) values ({', '.join(['%s'] * len(cols))})",
                     [meeting[c] for c in cols])
        with conn.cursor() as cur:
            cur.executemany(
                "insert into passages (meeting, kind, seq, text, owner, due, start_s, end_s, speakers, embedding)"
                " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [(name, p["kind"], i, p["text"], p.get("owner"), p.get("due"), p.get("start_s"), p.get("end_s"),
                  p.get("speakers"), _vec(v)) for i, (p, v) in enumerate(zip(passages, vectors))])
    return len(passages)


def _vec(v):
    import numpy as np
    return np.array(v, dtype=np.float32)


def remove(name):
    with connect() as conn, conn.transaction():
        return conn.execute("delete from meetings where name = %s", (name,)).rowcount


def rename(old, new):
    with connect() as conn, conn.transaction():
        conn.execute("update meetings set name = %s where name = %s", (new, old))


def set_workspace(names, ws):
    with connect() as conn, conn.transaction():
        conn.execute("update meetings set workspace = %s where name = any(%s)", (ws or "", list(names)))


# ---------- what needs doing ----------
def pending():
    """Confirmed meetings (not in Trash) that are missing from the index or changed since: new labels,
    a new summary. And indexed meetings that should not be there any more."""
    with connect() as conn:
        have = {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
            "select name, labels_confirmed_at, summary_mtime, chunking from meetings").fetchall()}
    chunking = _chunking(config())
    todo, ready = [], set()
    for it in library.items():
        if it["folder"] == "trash" or not it["summary"] or it["labels"] != "confirmed":
            continue
        ready.add(it["name"])
        lab = labels.get(it["name"])
        mtime = (library.LLM_SUMMARIES / f"{it['name']}.json").stat().st_mtime
        if have.get(it["name"]) != (lab.get("confirmed_at"), mtime, chunking):
            todo.append(it["name"])
    return todo, sorted(set(have) - ready)


def sync(log=print):
    """Index what is pending and drop what should not be indexed. Returns (indexed, removed, failed)."""
    todo, stale = pending()
    done, failed = 0, []
    for name in todo:
        try:
            n = index_one(name)
            done += 1
            log(f"✓ {name}: {n} passages")
        except Exception as e:
            failed.append(name)
            log(f"✗ {name}: {type(e).__name__}: {e}")
    for name in stale:
        remove(name)
        log(f"- {name}: removed (in Trash, or its labels are no longer confirmed)")
    return done, len(stale), failed


def status():
    cfg = config()
    out = {"enabled": cfg["enabled"], "embed_model": cfg["embed_model"], "ok": False}
    try:
        todo, stale = pending()
        with connect(cfg) as conn:
            out.update(ok=True,
                       meetings=conn.execute("select count(*) from meetings").fetchone()[0],
                       passages=conn.execute("select count(*) from passages").fetchone()[0],
                       by_workspace=dict(conn.execute(
                           "select workspace, count(*) from meetings group by workspace").fetchall()),
                       pending=todo, stale=stale)
    except Exception as e:
        out["message"] = str(e)
    return out


# ---------- search ----------
def search(q, workspace=None, client=None, project=None, person=None, kind=None, since=None, until=None,
           k=8):
    """Passages closest in meaning to `q`, best first, filtered by labels and dates."""
    where, args, keys = [], [], []
    reg = labels.load(workspace) if workspace and labels.exists(workspace) else None
    for ent, val in (("client", client), ("project", project), ("person", person)):
        if val:  # a misheard or differently written name finds the registry entry it belongs to
            canonical = labels.resolve(reg, ent, val)[0] if reg else None
            keys.append(label_key(ent, canonical or val))
    if keys:
        where.append("m.label_keys @> %s"); args.append(keys)  # uses the GIN index
    if workspace is not None:
        where.append("m.workspace = %s"); args.append(workspace)
    if kind:
        kinds = [kind] if isinstance(kind, str) else list(kind)
        if set(kinds) - set(KINDS):
            raise IndexError_(f"kind must be among {', '.join(KINDS)}")
        where.append("p.kind = any(%s)"); args.append(kinds)
    if since:
        where.append("m.held >= %s"); args.append(date.fromisoformat(str(since)))
    if until:
        where.append("m.held <= %s"); args.append(date.fromisoformat(str(until)))
    vec = _vec(embed([q], query=True)[0])
    # With filters, HNSW alone finds the ef_search nearest passages and filters afterwards, which can leave
    # fewer than k. Iterative scan keeps walking the graph until k rows pass the filters; its relaxed
    # order is put right by the outer sort.
    sql = f"""
        with hits as materialized (
          select m.name, m.title, m.held, m.workspace, m.type, m.clients, m.projects, m.people,
                 p.kind, p.text, p.owner, p.due, p.start_s, p.end_s, p.embedding <=> %s as distance
          from passages p join meetings m on m.name = p.meeting
          {"where " + " and ".join(where) if where else ""}
          order by p.embedding <=> %s limit %s)
        select *, 1 - distance as score from hits order by distance"""
    with connect() as conn, conn.transaction():
        conn.execute("set local hnsw.ef_search = 100")
        conn.execute("set local hnsw.iterative_scan = relaxed_order")
        cur = conn.execute(sql, [vec, *args, vec, int(k)])
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        r["held"] = r["held"].isoformat() if r["held"] else None
        r["score"] = round(float(r.pop("score")), 3)
        r.pop("distance", None)
    return rows


def _fmt_time(s):
    return f"{int(s // 60)}:{int(s % 60):02d}" if s is not None else ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["status", "backfill", "search", "meeting", "remove", "reset"])
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
        for r in search(" ".join(a.args), a.workspace, a.client, a.project, a.person, a.kind, a.since, a.until, a.k):
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
    elif a.command == "reset":
        reset()
        print("index dropped; run `uv run index.py backfill`")


if __name__ == "__main__":
    main()
