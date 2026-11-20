"""The MCP client: all inventory access to Gmail goes through the two tools.

Also turns raw tool results into small message records, keeping only what the
inventory needs.

Reads run in parallel (READ_CONCURRENCY at a time, see gather_in_order), and
results are always used in the original order, so reports do not depend on
which read finished first. If Gmail rate-limits a read after its own retries,
the client switches to slow mode for the rest of the run: one read at a time,
with backoff, before giving up.
"""

import asyncio
import contextlib
import json
import os
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from email.utils import parseaddr
from typing import TypeVar

from . import config

GMAIL_TOOL_NAMES = {"search_emails", "read_email"}
LIST_PAGE_SIZE = 500
MCP_CALL_TIMEOUT_SECONDS = 180  # Long enough for Google's retries and backoff.
READ_CONCURRENCY = 8  # Matches the server's worker threads.
EXCERPT_CHARS = 2000  # Part of the classification cache key: changing it re-classifies everything.

BULK_HEADERS = (
    "list-id", "list-unsubscribe", "list-post", "feedback-id", "x-campaign",
    "x-campaign-id", "x-campaignid", "x-mc-user", "x-sg-eid", "x-mailgun-sid",
)
BULK_PRECEDENCE_VALUES = {"bulk", "list", "junk"}
BULK_CATEGORY_LABELS = {"CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS"}


class GmailToolError(Exception):
    """A Gmail tool failed. `code` is one of the prefixes the server raises."""

    CODES = ("not_found", "auth_expired", "rate_limited", "gmail_error")

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    @classmethod
    def from_tool_text(cls, text: str) -> "GmailToolError":
        for code in cls.CODES:
            position = text.find(f"{code}:")
            if position != -1:
                return cls(code, text[position:][:300])
        return cls("gmail_error", "The Gmail tool reported an unrecognized error.")


T = TypeVar("T")
R = TypeVar("R")


async def gather_in_order(items: Sequence[T], worker: Callable[[T], Awaitable[R]],
                          limit: int = READ_CONCURRENCY) -> list[R]:
    """Run worker(item) for every item, at most `limit` at once; results come back in input order.

    If one worker fails, the others are cancelled and that error is raised as is.
    If the caller is cancelled (Ctrl-C), every worker is cancelled and awaited first.
    """
    semaphore = asyncio.Semaphore(limit)

    async def guarded(item: T) -> R:
        async with semaphore:
            return await worker(item)

    tasks = [asyncio.ensure_future(guarded(item)) for item in items]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class GmailMcpClient:
    RATE_LIMIT_RETRIES = 3
    RATE_LIMIT_BACKOFF_SECONDS = 2.0  # Doubles on each retry: 2, 4, 8 seconds.

    def __init__(self, session):
        self.session = session
        self.slow_mode = False  # Set once Gmail rate-limits a read; then reads go one at a time.
        self._one_at_a_time = asyncio.Lock()

    async def verify_tools(self):
        tools = await self.session.list_tools()
        names = {tool.name for tool in tools.tools}
        if names != GMAIL_TOOL_NAMES:
            raise ValueError(f"Expected MCP tools {sorted(GMAIL_TOOL_NAMES)}, found {sorted(names)}.")

    async def call_tool(self, name: str, arguments: dict) -> dict:
        import anyio
        from mcp.shared.exceptions import McpError

        try:
            result = await self.session.call_tool(
                name, arguments, read_timeout_seconds=timedelta(seconds=MCP_CALL_TIMEOUT_SECONDS),
            )
        except (McpError, anyio.ClosedResourceError, anyio.BrokenResourceError) as error:
            raise GmailToolError("gmail_error", f"MCP call failed ({type(error).__name__}).") from None
        text = "\n".join(item.text for item in result.content if item.type == "text")
        if result.isError:
            raise GmailToolError.from_tool_text(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise GmailToolError("gmail_error", "The Gmail tool returned invalid JSON.") from None

    async def search(self, query: str, page_token: str | None) -> dict:
        return await self.call_tool(
            "search_emails", {"query": query, "page_token": page_token, "page_size": LIST_PAGE_SIZE},
        )

    async def read(self, arguments: dict) -> dict:
        """read_email, with slow mode and backoff when Gmail rate-limits us."""
        attempt = 0
        while True:
            try:
                if self.slow_mode:
                    async with self._one_at_a_time:
                        return await self.call_tool("read_email", arguments)
                return await self.call_tool("read_email", arguments)
            except GmailToolError as error:
                if error.code != "rate_limited" or attempt >= self.RATE_LIMIT_RETRIES:
                    raise
                self.slow_mode = True
                await asyncio.sleep(self.RATE_LIMIT_BACKOFF_SECONDS * 2 ** attempt)
                attempt += 1

    async def read_metadata(self, message_id: str) -> dict:
        return await self.read({"message_id": message_id, "mode": "metadata"})

    async def read_body(self, message_id: str, max_chars: int = EXCERPT_CHARS) -> dict:
        return await self.read({"message_id": message_id, "mode": "full", "max_chars": max_chars})


def gmail_server_parameters():
    from mcp import StdioServerParameters
    from mcp.client.stdio import get_default_environment

    # The MCP library passes only a few safe variables to the server, so pass
    # the data location (and PYTHONPATH for development checkouts) explicitly.
    environment = get_default_environment()
    environment["NEWSLETTER_AGENT_HOME"] = str(config.app_home())
    if os.environ.get("PYTHONPATH"):
        environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return StdioServerParameters(
        command=sys.executable, args=["-m", "newsletter_agent", "mcp-server"], env=environment,
    )


@contextlib.asynccontextmanager
async def open_gmail_client():
    """Start the MCP server as a subprocess and stop it cleanly afterwards."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(gmail_server_parameters()) as (reader, writer), ClientSession(reader, writer) as session:
        await session.initialize()
        client = GmailMcpClient(session)
        await client.verify_tools()
        yield client


async def check_connection(live: bool = False):
    """`newsletter-agent check`: test MCP; with live=True also search and read one message."""
    async with open_gmail_client() as gmail:
        tools = await gmail.session.list_tools()
        print("MCP tools:", ", ".join(sorted(tool.name for tool in tools.tools)))
        if not live:
            print("Local MCP check passed. Gmail access has not been tested.")
            return
        page = await gmail.search("newer_than:7d", None)
        print(f"Gmail search returned {len(page['messages'])} messages from the last 7 days (first page).")
        if page["messages"]:
            await gmail.read_metadata(page["messages"][0]["id"])
            print("Read one message's headers successfully; nothing was printed or sent to OpenAI.")


def describe_session_failure(error: BaseException) -> str:
    """MCP's task groups wrap errors in exception groups; find the original one."""
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    if isinstance(error, (ValueError, OSError, GmailToolError)):
        # This program's own readable messages. A GmailToolError carries only the server's
        # short coded reason (for example "auth_expired: ... Run: newsletter-agent login").
        return str(error)
    return f"the Gmail MCP session failed ({type(error).__name__})."


# --- Message records --------------------------------------------------------

def clean_text(value, limit: int = 300) -> str:
    """Make untrusted text safe to print: no control characters or terminal escapes."""
    text = "".join(char if char.isprintable() else " " for char in str(value or ""))
    return " ".join(text.split())[:limit]


def normalize_list_id(value: str) -> str:
    """'Weekly News <weekly.example.com>' becomes 'weekly.example.com'."""
    match = re.search(r"<([^<>]+)>", value)
    return (match.group(1) if match else value).strip().lower()


def bulk_signals_from(raw: dict) -> list[str]:
    headers = raw.get("headers", {})
    signals = [name for name in BULK_HEADERS if headers.get(name)]
    if headers.get("precedence", "").strip().lower() in BULK_PRECEDENCE_VALUES:
        signals.append("precedence")
    signals.extend(sorted(BULK_CATEGORY_LABELS.intersection(raw.get("labels", []))))
    return signals


def make_message_record(raw: dict) -> dict:
    """Keep only what the inventory needs.

    The List-Unsubscribe value may be a personalized URL, so only its presence
    is kept. Bodies are never put in a record.
    """
    headers = raw.get("headers", {})
    return {
        "id": raw["id"],
        "thread_id": raw.get("thread_id"),
        "internal_date": int(raw.get("internal_date", 0)),
        "labels": list(raw.get("labels", [])),
        "from": clean_text(headers.get("from")),
        "subject": clean_text(headers.get("subject")),
        "date": clean_text(headers.get("date"), 80),
        "list_id": clean_text(normalize_list_id(headers.get("list-id", ""))),
        "has_list_unsubscribe": bool(headers.get("list-unsubscribe")),
        "bulk_signals": bulk_signals_from(raw),
        "size_estimate": int(raw.get("size_estimate", 0) or 0),
    }


def sender_address(record: dict) -> str:
    return parseaddr(record["from"])[1].lower() or record["from"].lower() or "unknown"


def remove_links(text: str) -> str:
    """Links add cost, and many are personalized tracking URLs."""
    return re.sub(r"(https?://|www\.)\S+", "[link]", text)
