"""Shared test helpers. All email here is synthetic.

Workflow tests use the real FastMCP server and real MCP protocol in memory.
Only the outermost services are fake: Google's API (SyntheticGmail from the
demo module) and OpenAI (FakeOpenAI below). No test contacts Gmail or OpenAI.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from mcp.shared.memory import create_connected_server_and_client_session

from newsletter_agent.briefing import BriefingBatch, SummaryBullet
from newsletter_agent.classify import ClassificationBatch, MessageClassification
from newsletter_agent.demo import SyntheticGmail, make_message
from newsletter_agent.gmail_client import GmailMcpClient
from newsletter_agent.gmail_server import build_mcp_server
from newsletter_agent.inventory import run_inventory
from newsletter_agent.scan import InventoryOptions
from newsletter_agent.store import InventoryStore

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
NOW_MS = int(NOW.timestamp() * 1000)
DAY_MS = 86_400_000
WINDOW_START_MS = NOW_MS - 90 * DAY_MS


def message(message_id, when=None, **kwargs):
    return make_message(message_id, when_ms=NOW_MS - DAY_MS if when is None else when, **kwargs)


def newsletter(message_id, list_id="weekly.example.com", **kwargs):
    headers = {"List-Id": f"Weekly <{list_id}>", "List-Unsubscribe": "<https://unsub.example.com/SECRET-TOKEN>"}
    headers.update(kwargs.pop("headers", {}))
    return message(message_id, headers=headers, **kwargs)


def many_newsletters(count):
    return [newsletter(f"m{i:03}") for i in range(count)]


def fake_gmail(messages=(), pages=None):
    """Small pages, so tests exercise pagination."""
    return SyntheticGmail(messages, pages=pages, page_size=2)


def default_classifier(item):
    """Label a synthetic message from its subject."""
    subject = item["subject"].lower()
    category = "transactional" if "receipt" in subject else "editorial_newsletter"
    if "sale" in subject:
        category = "promotional_subscription"
    name = item["list_id"].split(".")[0].title() if item["list_id"] else None
    return MessageClassification(
        message_id=item["id"], classification=category, publication_name=name,
        confidence="high" if item["list_id"] else "medium", rationale="Synthetic test label.",
    )


def default_handler(items):
    return ClassificationBatch(classifications=[default_classifier(item) for item in items])


def default_summary_handler(issues):
    return BriefingBatch(bullets=[SummaryBullet(source_message_id=issue["id"], text=f"Summary of {issue['subject']}")
                                  for issue in issues])


def fake_response(parsed, status="completed", output=(), incomplete_reason=None, tokens=120):
    return SimpleNamespace(
        status=status, output=list(output), output_parsed=parsed,
        incomplete_details=SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None,
        usage=SimpleNamespace(input_tokens=tokens - 20, output_tokens=20, total_tokens=tokens),
    )


class FakeOpenAI:
    """Answers classification and briefing calls with handlers that tests can swap."""

    def __init__(self, handler=default_handler, summary_handler=default_summary_handler):
        self.handler = handler
        self.summary_handler = summary_handler
        self.calls = []
        self.responses = self

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        data = json.loads(kwargs["input"])
        if kwargs["text_format"] is ClassificationBatch:
            result = self.handler(data["messages"])
        else:
            result = self.summary_handler(data["issues"])
        if asyncio.iscoroutine(result):
            result = await result
        return fake_response(result) if isinstance(result, (ClassificationBatch, BriefingBatch)) else result


def sent_items(fake_openai):
    return [item for call in fake_openai.calls if call["text_format"] is ClassificationBatch
            for item in json.loads(call["input"])["messages"]]


def run_workflow(tmp_path, google, fake_openai=None, confirm=lambda preflight: True, now=NOW, **options):
    """Run the real inventory through a real in-memory MCP session."""
    fake_openai = fake_openai or FakeOpenAI()
    store = InventoryStore(tmp_path / "inventory")

    async def go():
        async with create_connected_server_and_client_session(build_mcp_server(lambda: google)) as session:
            gmail = GmailMcpClient(session)
            await gmail.verify_tools()
            return await run_inventory(gmail, InventoryOptions(**options), store,
                                       openai_factory=lambda: fake_openai, confirm=confirm, now=now)

    return asyncio.run(go()), fake_openai


def all_groups(report):
    return report["publications_observed"] + report["publications_provisional"]


def group_by_key(report, key):
    return next(group for group in all_groups(report) if group["group_key"] == key)
