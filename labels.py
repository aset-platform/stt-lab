"""Meeting labels: workspace, meeting type, clients, projects, people and topics.

Workspaces
  One registry per firm you work with: registry/<id>.json (git-ignored, it holds real names). A registry
  lists the meeting types and the known entities (clients, projects, people, topics), each with the
  other spellings speech recognition produces for it (aliases), a status (active / archived: people,
  clients and projects come and go) and the first/last meeting it was confirmed in. Moving to another
  firm means a new, empty workspace; old meetings keep their labels and their own workspace.
  The active workspace (settings "labels" section) is used for new meetings.

Meeting labels
  llm/labels/<recording>.json, written by the summariser as "suggested" and only "confirmed" by you in
  the Manage tab. Names the model wrote that are not in the registry are kept under "new" until you
  decide: add as a new entity, record as an alias of a known one, or ignore. Confirming teaches the
  registry, so the same misspelling is resolved automatically next time.
"""
import difflib
import json
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path

import library
import settings

ROOT = Path(__file__).parent
REGISTRY = ROOT / "registry"
LABELS = library.LLM_LABELS
PROMPT_FILE = ROOT / "prompts" / "meeting_labels.md"
KINDS = {"client": "clients", "project": "projects", "person": "people", "topic": "topics"}
NAMED = ("client", "project", "person")  # kinds whose unknown names need your decision
DEFAULT_TYPES = ["client call", "internal standup", "1:1", "demo", "HR/admin", "interview", "other"]
REGISTRY.mkdir(exist_ok=True)
_lock = threading.RLock()


class LabelError(library.LibraryError):
    pass


# ---------- workspaces ----------
def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _path(ws):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", ws or ""):
        raise LabelError(f"bad workspace id {ws!r}")
    return REGISTRY / f"{ws}.json"


def load(ws):
    p = _path(ws)
    if not p.is_file():
        raise LabelError(f"no workspace {ws!r}")
    return json.loads(p.read_text())


def _save(reg):
    p = _path(reg["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=1) + "\n")
    os.replace(tmp, p)


def workspaces():
    out = []
    for p in sorted(REGISTRY.glob("*.json")):
        try:
            reg = json.loads(p.read_text())
        except ValueError:
            continue
        out.append({"id": reg["id"], "name": reg["name"], "entities": len(reg.get("entities", []))})
    return out


def exists(ws):
    return bool(ws) and re.fullmatch(r"[a-z0-9][a-z0-9-]*", ws) is not None and _path(ws).is_file()


def active():
    """The workspace selected in the header: the pages show its meetings and new recordings join it."""
    ws = settings.get("labels")["workspace"]
    return ws if exists(ws) else ""


def of(name):
    """The workspace a recording is labelled in: its own, or none if that registry is gone."""
    ws = library.workspace_of(name)
    return ws if exists(ws) else ""


def create_workspace(name, types=None):
    name = (name or "").strip()
    ws = slug(name)
    if not ws:
        raise LabelError("give the workspace a name")
    with _lock:
        if _path(ws).is_file():
            raise LabelError(f"workspace {name!r} already exists")
        reg = {"id": ws, "name": name, "created": date.today().isoformat(),
               "types": list(types or DEFAULT_TYPES), "entities": []}
        _save(reg)
    return reg


def set_types(ws, types):
    types = list(dict.fromkeys(t.strip() for t in types if t.strip()))
    if not types:
        raise LabelError("keep at least one meeting type")
    with _lock:
        reg = load(ws)
        reg["types"] = types
        _save(reg)
    return reg


# ---------- entities ----------
def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _clean(names, exclude=""):
    seen, out = {_norm(exclude)}, []
    for n in names or []:
        n = n.strip() if isinstance(n, str) else ""
        if n and _norm(n) and _norm(n) not in seen:
            seen.add(_norm(n))
            out.append(n)
    return out


def _find(reg, kind, name):
    """The entity of `kind` whose name or alias is `name` (spelling-insensitive), or None."""
    k = _norm(name)
    for e in reg["entities"]:
        if e["kind"] == kind and k in {_norm(e["name"]), *map(_norm, e.get("aliases", []))}:
            return e
    return None


def upsert_entity(ws, ent, original=None):
    """Add an entity, or edit the one called `original`. ent: {kind, name, aliases, status, note,
    client (projects), org and role (people)}."""
    kind, name = ent.get("kind"), (ent.get("name") or "").strip()
    if kind not in KINDS:
        raise LabelError(f"kind must be one of {', '.join(KINDS)}")
    if not name:
        raise LabelError("an entity needs a name")
    with _lock:
        reg = load(ws)
        cur = _find(reg, kind, original or name)
        if original and not cur:
            raise LabelError(f"no {kind} {original!r} in {reg['name']}")
        if not original and cur:
            raise LabelError(f"{name!r} is already known as {kind} {cur['name']!r}")
        aliases = _clean(ent.get("aliases", cur.get("aliases", []) if cur else []), exclude=name)
        for n in [name, *aliases]:
            other = _find(reg, kind, n)
            if other and other is not cur:
                raise LabelError(f"{n!r} already belongs to {kind} {other['name']!r}")
        new = cur if cur is not None else {"kind": kind}
        if cur is None:
            reg["entities"].append(new)
        new.update({"name": name, "aliases": aliases,
                    "status": ent.get("status") if ent.get("status") in ("active", "archived") else new.get("status", "active")})
        for f in ("note", "client", "org", "role"):
            if f in ent:
                new[f] = (ent[f] or "").strip()
        _save(reg)
    return new


def delete_entity(ws, kind, name):
    with _lock:
        reg = load(ws)
        e = _find(reg, kind, name)
        if not e:
            raise LabelError(f"no {kind} {name!r}")
        reg["entities"].remove(e)
        _save(reg)


def resolve(reg, kind, name):
    """(canonical name, None) if known; else (None, closest known name or None) for the review."""
    e = _find(reg, kind, name) if reg else None
    if e:
        return e["name"], None
    if not reg:
        return None, None
    keys = {}
    for x in reg["entities"]:
        if x["kind"] == kind:
            for n in [x["name"], *x.get("aliases", [])]:
                keys[_norm(n)] = x["name"]
    close = difflib.get_close_matches(_norm(name), list(keys), n=1, cutoff=0.75)
    return None, keys[close[0]] if close else None


# ---------- prompt ----------
def prompt_block(ws):
    """Label instructions for the LLM, with the workspace's known names."""
    reg = load(ws) if ws else {"types": DEFAULT_TYPES, "entities": []}

    def known(kind):
        rows = []
        for e in reg["entities"]:
            if e["kind"] != kind:
                continue
            extra = [f"heard as: {', '.join(e['aliases'])}"] if e.get("aliases") else []
            extra += [x for x in (e.get("org"), e.get("client"), e.get("role")) if x]
            if e.get("status") == "archived":
                extra.append("past")
            rows.append(f"- {e['name']}" + (f" ({'; '.join(extra)})" if extra else ""))
        return "\n".join(rows) or "- (none yet)"

    text = PROMPT_FILE.read_text()
    for key, value in {"types": ", ".join(reg["types"]), "clients": known("client"),
                       "projects": known("project"), "people": known("person"), "topics": known("topic")}.items():
        text = text.replace("{{" + key + "}}", value)
    return text


# ---------- meeting labels ----------
def _file(name):
    library.audio_path(name)  # a real recording's plain file name only
    return LABELS / f"{name}.json"


def get(name):
    p = _file(name)
    return json.loads(p.read_text()) if p.is_file() else None


def status(name):
    lab = get(name)
    return lab and lab.get("status")


def meeting_date(name):
    m = re.search(r"\d{4}-\d{2}-\d{2}", name)
    return m.group(0) if m else date.today().isoformat()


def from_llm(raw, ws, model):
    """Turn the model's "labels" object into a suggestion: known names made canonical, the rest in "new"."""
    reg = load(ws) if ws else None
    types = reg["types"] if reg else DEFAULT_TYPES
    t = str(raw.get("type") or "").strip().lower()
    lab = {"workspace": ws, "status": "suggested", "model": model,
           "suggested_at": datetime.now().isoformat(timespec="seconds"),
           "type": next((x for x in types if x.lower() == t), "other" if "other" in types else types[-1]),
           **{field: [] for field in KINDS.values()}, "new": []}
    for kind, field in KINDS.items():
        for n in _clean(raw.get(field) if isinstance(raw.get(field), list) else []):
            if re.fullmatch(r"(?i)speaker[ _]?\d+|unassigned|unknown", n):
                continue
            canonical, maybe = resolve(reg, kind, n)
            if canonical:
                lab[field] = _clean([*lab[field], canonical])
            elif kind == "topic":  # topics are free text; they join the registry when you confirm
                lab[field] = _clean([*lab[field], n])
            elif _norm(n) not in {_norm(x["name"]) for x in lab["new"]}:
                lab["new"].append({"name": n, "kind": kind, "maybe": maybe})
    return lab


def save_suggestion(name, lab, force=False):
    """Store a suggestion. A confirmed label is only replaced when asked (force)."""
    with _lock:
        if not force and status(name) == "confirmed":
            return get(name)
        _file(name).write_text(json.dumps(lab, ensure_ascii=False, indent=1) + "\n")
    return lab


def confirm(name, lab, learn=()):
    """Save your reviewed labels and teach the registry.

    lab:   {workspace, type, clients, projects, people, topics}
    learn: [{name, kind, action: "add" | "alias" | "ignore", as: corrected spelling (add),
           of: canonical name (alias)}] for the names that were not in the registry.
    A meeting type that is not in the workspace yet is added to it."""
    ws = lab.get("workspace") or ""
    when = meeting_date(name)
    out = {"workspace": ws, "status": "confirmed", "type": str(lab.get("type") or "other"),
           "confirmed_at": datetime.now().isoformat(timespec="seconds"), "new": []}
    prev = get(name) or {}
    if prev.get("model"):
        out["model"] = prev["model"]
    lists = {field: _clean(lab.get(field)) for field in KINDS.values()}
    with _lock:
        reg = load(ws) if ws else None
        out["type"] = out["type"].strip() or "other"
        if reg:  # a type you typed in joins the workspace's list (same spelling as an existing one is reused)
            same = next((t for t in reg["types"] if t.lower() == out["type"].lower()), None)
            if same:
                out["type"] = same
            else:
                reg["types"].append(out["type"])
        for item in learn or []:
            kind, n, action = item.get("kind"), (item.get("name") or "").strip(), item.get("action")
            if kind not in KINDS or not n or action == "ignore":
                continue
            field = KINDS[kind]
            if action == "alias" and reg:
                target = _find(reg, kind, item.get("of") or "")
                if not target:
                    raise LabelError(f"no {kind} {item.get('of')!r} to add {n!r} to")
                if _norm(n) not in map(_norm, [target["name"], *target.get("aliases", [])]):
                    target.setdefault("aliases", []).append(n)
                lists[field] = _clean([*lists[field], target["name"]])
            elif action == "add" and reg:
                # "as" is your spelling (e.g. heard "Jon Smyth", it is "John Smith"); the heard one is kept as
                # an alias so the same mishearing resolves by itself next time.
                right = (item.get("as") or "").strip() or n
                e = _find(reg, kind, right)
                if e is None:
                    e = {"kind": kind, "name": right, "aliases": [], "status": "active"}
                    reg["entities"].append(e)
                if _norm(n) not in map(_norm, [e["name"], *e.get("aliases", [])]):
                    other = _find(reg, kind, n)
                    if other and other is not e:
                        raise LabelError(f"{n!r} already belongs to {kind} {other['name']!r}")
                    e.setdefault("aliases", []).append(n)
                lists[field] = _clean([*lists[field], e["name"]])
            elif action in ("add", "alias"):
                lists[field] = _clean([*lists[field], (item.get("as") or "").strip() or n])
        if reg:
            for kind, field in KINDS.items():
                names = []
                for n in lists[field]:
                    e = _find(reg, kind, n)
                    if e is None:  # confirmed by you, so it joins the registry
                        e = {"kind": kind, "name": n, "aliases": [], "status": "active"}
                        reg["entities"].append(e)
                    e["first_seen"] = min(filter(None, [e.get("first_seen"), when]))
                    e["last_seen"] = max(filter(None, [e.get("last_seen"), when]))
                    names.append(e["name"])
                lists[field] = _clean(names)
            _save(reg)
        out.update(lists)
        _file(name).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    library.set_workspace([name], ws)  # labelled in a workspace = belongs to it
    return out


def search_text(name):
    """Label words for the Manage search box."""
    lab = get(name)
    if not lab:
        return ""
    return " ".join([lab.get("type", ""), *(n for f in KINDS.values() for n in lab.get(f, []))])
