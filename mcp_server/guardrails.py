"""Guardrails: keep deal figures, sensitive remarks and abuse out of anything that leaves this Mac.

One filter, used twice:
- summarise.py runs it on every summary after the model has written it (the prompt asks the model to
  leave these things out; this catches what it didn't);
- the MCP server runs it on every answer, including transcript lines and search passages, which never
  pass through the summary prompt.

A sentence (or transcript line) that matches is replaced by "[removed]" (MCP) or dropped (summaries).
Pattern matching is reliable for amounts, deal terms, swear words and remarks about race or origin.
It cannot recognise sarcasm or a rude remark in polite words: the summary prompt asks the model to
leave those out, and transcripts can be kept away from MCP clients entirely (STT_MCP_TRANSCRIPTS=off).
Words that also have an ordinary technical meaning (race condition, CSS margin, deal with...) are only
matched in their commercial or personal sense, so technical notes survive.
"""
import re

REMOVED = "[removed]"

# "$1.3M", "1.3 million dollars", "₹2 crore", "USD 500k", "2 lakh", "1.3 million for a year": money,
# unless the amount counts something technical ("1.3 million rows")
_UNITS = r"(?:k|mn|m|bn|b|million|billion|thousand|lakhs?|lacs?|crores?|cr)"
_COUNTED = (r"(?!\s*(?:rows?|records?|events?|messages?|requests?|users?|tokens?|lines?|files?|documents?|docs?|"
            r"calls?|quer(?:y|ies)|transactions?|customers?|param(?:eter)?s?|vectors?|embeddings?|images?|pages?|"
            r"bytes?|[kmgt]b|hits?|views?|downloads?|records?|steps?|samples?|points?|rows?)\b)")
MONEY = re.compile(
    r"[$€£₹]\s?\d[\d,.]*(?:\s?" + _UNITS + r")?\b"
    r"|\b(?:usd|inr|eur|gbp|aud|cad|sgd|rs\.?|rupees?|dollars?)\s?\d[\d,.]*"
    r"|\b\d[\d,.]*\s?" + _UNITS + r"?\s?(?:usd|inr|eur|gbp|dollars?|rupees?|euros?|pounds?|bucks)\b"
    r"|\b\d[\d,.]*\s?(?:million|billion|lakhs?|lacs?|crores?|cr|mn|bn)\b" + _COUNTED,
    re.I)

DEAL = re.compile(
    r"\b(?:deal\s?(?:size|value|amount|closure|closing|terms|pipeline|stage|win|won|lost|loss)"
    r"|(?:the|this|that|a|our|their|big|new)\s+deals?\b(?!\s+with)|deals\b(?!\s+with)"
    r"|contract\s+(?:value|size|amount|renewal)|tcv|acv|arr|mrr|booking\s+value"
    r"|pric(?:e|es|ed|ing)\b|rate\s?cards?|billing\s+rates?|hourly\s+rates?|per\s+hour\s+rate|day\s+rates?"
    r"|discount(?:s|ed|ing)?|(?:gross|net|profit|our|healthy|thin|deal)\s+margins?|margins?\s+of\s+\d"
    r"|revenue|budget(?:s|ed|ing)?|invoic(?:e|es|ed|ing)|commercials?\b|payment\s+terms|purchase\s+orders?"
    r"|bid(?:s|ding)?\b|proposal\s+value|fees?\b|retainer|upsell\w*|cross-?sell\w*|sales\s+quota)\b",
    re.I)

# remarks about race, origin, religion, caste, health, private life ("Is he Indian or white?")
_ORIGIN = r"(?:indian|white|black|brown|asian|chinese|african|american|british|european|desi|gora|firang\w*|foreign)"
SENSITIVE = re.compile(
    r"\b(?:diversity|desi|caste|ethnic\w*|racial|racis\w*|religio\w*|nationality|foreigners?|pregnan\w*|"
    r"illness|divorce\w*|gay|lesbian)\b"
    r"|\b(?:is|are|was|were)\s+(?:he|she|they|him|her)\s+(?:an?\s+)?" + _ORIGIN + r"\b"
    r"|\b" + _ORIGIN + r"\s+(?:guy|guys|man|men|woman|women|girl|girls|people|person|folks|chap|fellow|lady)\b"
    r"|\b" + _ORIGIN + r"\s+or\s+" + _ORIGIN + r"\b",
    re.I)

# swearing and insults, English and Hindi as speech recognition writes it
ABUSE = re.compile(
    r"\b(?:fuck\w*|shit\w*|bullshit|bastard\w*|bitch\w*|asshole\w*|arsehole\w*|dickhead\w*|crap(?:py)?|damn\w*|"
    r"bloody|idiot\w*|stupid\w*|moron\w*|dumb\w*|retard\w*|scum\w*|jerks?|clowns?|useless\s+(?:guy|people|team|client)|"
    r"chutiy\w*|bhenchod|behenchod|madarchod|bhosd\w*|gaand\w*|saala|saali|kamin[ae]|kameen[ae]|harami\w*|"
    r"bewakoof\w*|ullu|gadh[ae]|lund|randi|kutt[ae]|kuttiya|bc|mc)\b",
    re.I)

# office politics and judgements of a person's character ("political issues between Marlow and Tamsin"):
# gossip, not a work fact. Narrow on purpose: plain "conflict between" (schedules) or "non-technical
# users" are ordinary work talk and stay.
GOSSIP = re.compile(
    r"\b(?:politic(?:s|al|king)(?!\s+(?:risks?|exposures?|part(?:y|ies)|science|climate|instability))|office\s+politics|power\s+struggles?|bad\s+blood|backstab\w*|"
    r"(?:not|don't|doesn't|do\s+not|does\s+not)\s+get\s+along|not\s+on\s+(?:good\s+)?(?:speaking\s+)?terms|"
    r"incompeten\w*|clueless|ego\s+(?:clash|issues?|problems?))\b",
    re.I)

RULES = {"money": MONEY, "deal": DEAL, "sensitive": SENSITIVE, "abuse": ABUSE, "gossip": GOSSIP}
_SENTENCES = re.compile(r"(?<=[.;!?])\s+")
_LINE_HEAD = re.compile(r"^(\[[\d:]+\]\s*[^:\n]{1,40}:\s*)(.*)$")  # "[12:34] SPEAKER_00: text"
# keys whose values are identifiers or bookkeeping, never meeting text
SKIP = {"meeting", "recording", "workspace", "workspace_id", "date", "start", "at", "first_seen", "last_seen",
        "status", "kind", "_meta", "role_at", "organisation_at", "reports_to_at", "error", "today", "range"}


def why(text):
    """The rule names a text matches (empty when it is clean)."""
    return [name for name, rx in RULES.items() if rx.search(text or "")]


def clean_text(text, mark=REMOVED):
    """`text` with every matching sentence replaced by `mark` (or dropped when mark is ""). Multi-line
    text is handled line by line; a transcript line keeps its "[time] SPEAKER:" head. Returns (text, n)."""
    n = 0
    out_lines = []
    for line in str(text).split("\n"):
        m = _LINE_HEAD.match(line)
        head, body = (m.group(1), m.group(2)) if m else ("", line)
        parts = _SENTENCES.split(body)
        kept = []
        for p in parts:
            if why(p):
                n += 1
                if mark and (not kept or kept[-1] != mark):
                    kept.append(mark)
            else:
                kept.append(p)
        body = " ".join(kept).strip()
        if m:
            out_lines.append(head + (body or mark))
        elif body or not line.strip():
            out_lines.append(body)
    return "\n".join(out_lines), n


def scrub(x, mark=REMOVED, key=""):
    """Apply clean_text to every string inside `x` (dicts, lists), except identifier keys.
    Returns (scrubbed copy, number of sentences removed)."""
    if isinstance(x, str):
        return (x, 0) if key in SKIP else clean_text(x, mark)
    if isinstance(x, list):
        total, out = 0, []
        for v in x:
            v2, n = scrub(v, mark, key)
            total += n
            if v2 not in ("", None, mark):  # an item that was nothing but removed text goes entirely
                out.append(v2)
        return out, total
    if isinstance(x, dict):
        total, out = 0, {}
        for k, v in x.items():
            v2, n = scrub(v, mark, k)
            total += n
            out[k] = v2
        return out, total
    return x, 0
