"""Screen a meeting by meaning before it is indexed, so MCP clients never get what shouldn't leave the Mac.

    screen(name, texts) -> [reason or None, ...]

Every transcript line and every point of the notes is checked twice:
1. the word patterns of mcp_server/guardrails.py (amounts, deal terms, remarks about origin, swearing);
2. the local LLM, by meaning (prompts/share_filter.md): deals said vaguely ("around 1.3 for the year"),
   remarks about someone's background, insults, sarcasm, gossip, compliance matters about a person.
   Lines go in batches of BATCH consecutive lines, so each is read with the lines around it.

A flagged item stays in the index for this Mac's own search, marked restricted with its reason; the
database views the MCP server reads (index.py) leave it out or show "[removed]". Verdicts are cached in
llm/guard/<recording>.json by text, so re-indexing (new labels, a corrected spelling elsewhere) doesn't
ask the model again. If the model can't be reached, screening raises and the meeting isn't indexed:
nothing unscreened is ever shared.
"""
import hashlib
import json

import library
import settings
import summarise
from mcp_server import guardrails

PROMPT_FILE = library.ROOT / "prompts" / "share_filter.md"
CACHE = library.ROOT / "llm" / "guard"
BATCH = 40
REASONS = {"deal", "personal", "abuse", "sarcasm", "gossip", "compliance"}
VERSION = "s1"  # part of the index fingerprint: a new version re-screens every meeting
wait = None  # set by the server: called before each batch, it holds screening while a recording runs


def enabled():
    return bool(settings.get("index").get("share_filter", True))


def _key(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _ask(cfg, model, name, texts):
    """{position: reason} for the flagged items of one batch."""
    numbered = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts, 1))
    reply = summarise._summarise(cfg, model, name, numbered, PROMPT_FILE.read_text())
    out = {}
    for f in reply.get("flagged") or []:
        try:
            n = int(f.get("n"))
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= n <= len(texts):
            reason = str(f.get("reason") or "").strip().lower()
            out[n - 1] = reason if reason in REASONS else "flagged"
    return out


def screen(name, texts, log=print):
    """A reason (e.g. "deal", "pattern:money") or None for each text, in order. Raises if the model fails."""
    reasons = [("pattern:" + w[0]) if (w := guardrails.why(t)) else None for t in texts]
    if not enabled() or not texts:
        return reasons
    path = CACHE / f"{name}.json"
    cache = json.loads(path.read_text()) if path.is_file() else {}
    cfg = summarise.config()
    model = settings.get("index").get("share_model") or cfg["model"]
    for a in range(0, len(texts), BATCH):
        batch = texts[a:a + BATCH]
        if all(_key(t) in cache for t in batch):
            continue
        if wait:
            wait()
        log(f"{name}: screening items {a + 1}-{a + len(batch)} of {len(texts)}")
        flagged = _ask(cfg, model, name, batch)
        for i, t in enumerate(batch):
            cache[_key(t)] = flagged.get(i, "")
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=0))  # after every batch: an interrupted run resumes
    return [r or cache.get(_key(t)) or None for r, t in zip(reasons, texts)]
