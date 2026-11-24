"""stt-lab MCP server: your meeting notes and transcripts for Claude (or any MCP client) on this Mac.

Read-only. Every call works inside one workspace (firm); without `workspace` it uses the one selected in
stt-lab. Runs in Docker next to the index database (docker-compose.yml) on http://127.0.0.1:8765/mcp.

    STT_MCP_DB_URL   read-only database login     STT_EMBED_URL  Ollama, for embedding questions
    STT_TZ           your time zone ("yesterday")  STT_MCP_TOKEN  optional: require "Authorization: Bearer <token>"
    STT_MCP_AUDIT    audit log file (one JSON line per tool call: client, tool, arguments, meetings shared)
"""
import os
import secrets
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from pydantic import Field

from . import audit, store

INSTRUCTIONS = """\
Meeting notes and transcripts recorded with stt-lab: summaries, decisions, action items, open questions,
labels (clients, projects, people, topics) and full transcripts with timestamps.

- Each workspace is a separate firm. Never combine results from different workspaces in one answer.
  Leave `workspace` empty to use the active one; list_workspaces shows them.
- Dates: pass plain phrases such as "today", "yesterday", "2 days ago", "last 7 days", "this week",
  "last week", "monday" or ISO dates; results echo the resolved range and today's date.
- Start with list_meetings or action_items for "what happened / what's open"; get_meeting for one
  meeting's notes; get_transcript when exact wording, details or requirements matter; search to find where
  a topic was discussed across meetings.
- Transcripts were translated to English by speech recognition: names and terms can be misspelled and
  speaker labels (SPEAKER_00...) are voices, not names. Prefer the registry spellings (registry tool).
- Cite the meeting title and date (and [mm:ss] for transcript quotes) for every fact you use.
  Never invent requirements, owners or dates; mark anything inferred as an assumption.
"""

mcp = MCPServer("stt-lab", title="stt-lab meetings", instructions=INSTRUCTIONS, version="1.0")
READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

Workspace = Annotated[str, Field(description="Workspace id or name; empty = the active workspace")]
When = Annotated[str, Field(description='Date phrase: "today", "yesterday", "2 days ago", "last 7 days", '
                                        '"this week", "last week", "this month", "monday", "2026-11-20" '
                                        'or "2026-11-01..2026-11-15"; empty = any date')]
Day = Annotated[str, Field(description="ISO date YYYY-MM-DD; overrides that end of `when`")]


def _call(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except store.StoreError as e:
        return {"error": str(e)}


@mcp.tool(annotations=READ)
@audit.audited
def list_workspaces() -> dict:
    """The workspaces (one per firm) with their number of indexed meetings and date range, which one is
    active, and today's date."""
    return _call(store.list_workspaces)


@mcp.tool(annotations=READ)
@audit.audited
def list_meetings(workspace: Workspace = "", when: When = "", since: Day = "", until: Day = "",
                  client: str = "", project: str = "", person: str = "",
                  meeting_type: Annotated[str, Field(description="e.g. client call, 1:1, internal standup")] = "",
                  limit: int = 50) -> dict:
    """Meetings in a date range, newest first, with title, date, start time, type, duration, clients,
    projects, people and topics. Filter by client / project / person (misheard spellings are resolved
    through the workspace registry)."""
    return _call(store.list_meetings, workspace, when, since or None, until or None, client or None,
                 project or None, person or None, meeting_type or None, limit)


@mcp.tool(annotations=READ)
@audit.audited
def get_meeting(meeting: Annotated[str, Field(description="Recording name from list_meetings, or a unique part of the title")],
                workspace: Workspace = "") -> dict:
    """One meeting's notes: summary points, decisions, action items (owner, due), open questions and
    labels, plus the size of its transcript."""
    return _call(store.get_meeting, meeting, workspace)


@mcp.tool(annotations=READ)
@audit.audited
def get_transcript(meeting: Annotated[str, Field(description="Recording name from list_meetings, or a unique part of the title")],
                   workspace: Workspace = "",
                   from_minute: Annotated[float | None, Field(description="Start at this minute")] = None,
                   to_minute: Annotated[float | None, Field(description="Stop before this minute")] = None,
                   max_chars: Annotated[int, Field(description="Page size; long meetings come in pages")] = 60000) -> dict:
    """The full transcript as "[mm:ss] SPEAKER_xx: text" lines, for exact wording, details and
    requirements. A long meeting comes in pages: call again with from_minute = continue_from_minute."""
    return _call(store.get_transcript, meeting, workspace, from_minute, to_minute, max_chars)


@mcp.tool(annotations=READ)
@audit.audited
def search(query: Annotated[str, Field(description="What you are looking for, in plain words")],
           workspace: Workspace = "",
           kind: Annotated[Literal["", "summary", "decision", "action", "question", "transcript"],
                           Field(description="Only this kind of passage; empty = all")] = "",
           client: str = "", project: str = "", person: str = "", when: When = "", since: Day = "",
           until: Day = "", k: Annotated[int, Field(description="Number of results, at most 50")] = 10) -> dict:
    """Search by meaning across the workspace's meetings: summary points, decisions, action items, open
    questions and one-minute transcript passages, best match first, each with its meeting and date (and
    time in the recording for transcript passages)."""
    return _call(store.search, query, workspace, kind or None, client or None, project or None, person or None,
                 when, since or None, until or None, k)


@mcp.tool(annotations=READ)
@audit.audited
def action_items(workspace: Workspace = "", when: When = "", since: Day = "", until: Day = "",
                 owner: Annotated[str, Field(description="Person who took the task on")] = "",
                 client: str = "", project: str = "") -> dict:
    """Action items from the meetings in a date range (owner, task, due date, meeting), newest meeting
    first. Filter by owner, client or project."""
    return _call(store.action_items, workspace, when, since or None, until or None, owner or None,
                 client or None, project or None)


@mcp.tool(annotations=READ)
@audit.audited
def registry(workspace: Workspace = "",
             kind: Annotated[Literal["", "client", "project", "person", "topic"], Field(description="Only this kind")] = "") -> dict:
    """The workspace's known clients, projects, people and topics with their correct spelling, the
    spellings speech recognition produced for them (heard_as), role/org notes and first/last seen."""
    return _call(store.registry, workspace, kind or None)


# ---------- prompts: ready-made workflows ----------
@mcp.prompt(title="Meeting digest")
def meetings_digest(when: str = "yesterday", workspace: str = "") -> str:
    """Insights, decisions and action items from the meetings on a day or in a period."""
    ws = f' in workspace "{workspace}"' if workspace else " in the active workspace"
    return f"""Give me a digest of my meetings from "{when}"{ws}.

1. Call list_meetings with when="{when}"{f', workspace="{workspace}"' if workspace else ''}. If there are none, say so and stop.
2. Call get_meeting for each meeting. Use get_transcript only where a point is unclear or needs exact detail.
3. Write:
   - **Overview**: one line per meeting (title, date and time, type, who/which client).
   - **Key insights**: what matters across these meetings: progress, risks, client sentiment, changes in
     direction, things that connect between meetings.
   - **Decisions**.
   - **Action items**: a table of owner, task, due, meeting. Flag tasks with no owner or no date.
   - **Open questions and follow-ups** I should chase.
Cite the meeting for each point. Do not invent anything that is not in the notes or transcripts."""


@mcp.prompt(title="Feature requirements document")
def feature_requirements(meetings: str, feature: str = "", workspace: str = "", pages: str = "1 to 5") -> str:
    """Turn one or more meetings' discussion into a feature requirements document (1 to 5 pages)."""
    about = f' for "{feature}"' if feature else ""
    return f"""Write a feature requirements document{about} from these meetings: {meetings}{f' (workspace "{workspace}")' if workspace else ''}.

Gather the source material first:
1. Find each meeting (list_meetings or search if I gave a topic or date rather than names).
2. Call get_meeting for each, then get_transcript for each and read it fully (follow continue_from_minute
   until the end). Requirements live in the details of the discussion, not only in the summary.
3. Use search for the feature's key terms to find related discussion in other meetings of the workspace.
4. Use registry for the correct spelling of clients, projects and people.

Then write the document, {pages} pages, sized to what was actually discussed:
1. **Overview**: the feature in two or three sentences; source meetings (title, date).
2. **Problem and context**: who has the problem, why now, the current situation.
3. **Goals** and **Non-goals**.
4. **Users and scenarios**: who uses it, the main flows, step by step.
5. **Functional requirements**: numbered FR-1, FR-2, ... each testable, with priority (Must / Should /
   Could) and its source as (meeting title, [mm:ss]).
6. **Non-functional requirements**: performance, security, compliance, data, integrations, as discussed.
7. **Acceptance criteria**: Given / When / Then for the Must requirements.
8. **Dependencies, constraints and risks**.
9. **Decisions made** (with source) and **Open questions** (with who should answer).
10. **Assumptions**: anything you inferred rather than heard; keep this list honest.

Rules: every requirement must trace to something said in a meeting; quote short phrases where wording
matters; never invent numbers, dates, owners or scope. Speaker labels are voices, not names: name a person
only when the transcript or notes make it clear."""


@mcp.prompt(title="Client brief")
def client_brief(client: str, when: str = "last 30 days", workspace: str = "") -> str:
    """Everything discussed with or about a client in a period: status, commitments, risks."""
    return f"""Prepare a brief on client "{client}" from my meetings in "{when}"{f' (workspace "{workspace}")' if workspace else ''}.

1. list_meetings with client="{client}" and when="{when}"; get_meeting for each.
2. action_items with client="{client}" and when="{when}".
3. search for the client's open issues, commitments and concerns, and read transcripts where needed.

Write: **Where things stand**, **What we committed to** (owner, due, source), **What they asked for**,
**Risks and concerns**, **Open questions**, **Suggested next steps**. Cite meeting and date for each point."""


@mcp.prompt(title="Action tracker")
def action_tracker(owner: str = "", when: str = "last 7 days", workspace: str = "") -> str:
    """Open action items by owner for a period, with what each depends on."""
    who = f' for "{owner}"' if owner else ""
    return f"""List the action items{who} from my meetings in "{when}"{f' (workspace "{workspace}")' if workspace else ''}.

Call action_items with when="{when}"{f', owner="{owner}"' if owner else ''}. Group by owner; for each task give
the task, due date, meeting and date. Point out items without an owner or due date, and items that look
like the same task raised in several meetings. Use get_meeting or get_transcript if a task needs context."""


# ---------- serving ----------
def _guard(app, token):
    """Require "Authorization: Bearer <token>" on every request when STT_MCP_TOKEN is set."""
    expected = f"Bearer {token}".encode()

    async def guarded(scope, receive, send):
        if scope["type"] == "http":
            got = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not secrets.compare_digest(got, expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"text/plain"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b"missing or wrong bearer token"})
                return
        await app(scope, receive, send)
    return guarded


def app():
    # Published on the Mac's 127.0.0.1 only (docker-compose.yml); the Host check also blocks DNS rebinding
    # from web pages, which is how a browser tab could otherwise reach a localhost server.
    port = os.environ.get("STT_MCP_PORT", "8765")
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", f"127.0.0.1:{port}", f"localhost:{port}"],
        allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"])
    asgi = mcp.streamable_http_app(streamable_http_path="/mcp", stateless_http=True, transport_security=security,
                                   host="0.0.0.0")
    asgi = audit.track_client(asgi)  # the audit log names the client that called each tool
    token = os.environ.get("STT_MCP_TOKEN", "").strip()
    return _guard(asgi, token) if token else asgi


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app(), host=os.environ.get("STT_MCP_HOST", "0.0.0.0"), port=int(os.environ.get("STT_MCP_PORT", "8765")),
                log_level="info")
