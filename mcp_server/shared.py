"""What the indexer (index.py, on the Mac) and the MCP server (in Docker) must agree on: how a question is
embedded, how label filters are keyed, which passage kinds exist. Standard library only."""
import json
import re
import urllib.request

KINDS = ("summary", "decision", "action", "question", "transcript")
QUERY_INSTRUCTION = ("Instruct: Given a question about past meetings, retrieve the meeting notes and "
                     "transcript passages that answer it\nQuery: ")
READONLY_ROLE = "sttlab_mcp"  # the MCP server's database login: SELECT only


class EmbedError(Exception):
    pass


def norm(s):
    """Spelling-insensitive form of a name, as the registry compares them."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def label_key(kind, name):
    return f"{kind}:{' '.join(str(name).lower().split())}"


def schema_name(ws):
    return "ws_" + ws.replace("-", "_")  # workspace ids are [a-z0-9-], so this can't collide


def embed(texts, url, model, query=False):
    """Vectors for `texts` from Ollama's /api/embed, in batches. Questions get the model's instruction."""
    out = []
    for i in range(0, len(texts), 32):
        batch = [(QUERY_INSTRUCTION + t) if query else t for t in texts[i:i + 32]]
        req = urllib.request.Request(url.rstrip("/") + "/api/embed",
                                     data=json.dumps({"model": model, "input": batch, "keep_alive": "5m"}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out += json.loads(r.read())["embeddings"]
        except OSError as e:
            raise EmbedError(f"embedding model {model} not reachable at {url}: {e}")
    return out
