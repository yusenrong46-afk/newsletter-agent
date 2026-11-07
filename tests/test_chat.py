"""Chat: read-only tools over the saved inventory, with confirmation before reading email."""

import json
from types import SimpleNamespace

import httpx
import openai

from newsletter_agent import chat
from newsletter_agent.briefing import SummaryResult

REPORT = {
    "run": {"window_start": "2026-08-08", "window_end": "2026-11-06", "status": "complete",
            "report_generated_at": "2026-11-06T12:00:00+00:00"},
    "counts": {"publications_observed": 2, "publications_provisional": 1, "subscription_messages": 30, "subscription_unread": 20},
    "category_breakdown": {"editorial_newsletter": 25, "promotional_subscription": 5},
    "publications_observed": [
        {"group_key": "list:a", "name": "The Quiet Ledger", "category": "editorial_newsletter", "received_count": 10, "unread_count": 2, "status": "observed"},
        {"group_key": "list:b", "name": "Byte Garden", "category": "editorial_newsletter", "received_count": 15, "unread_count": 15, "status": "observed"},
    ],
    "publications_provisional": [
        {"group_key": "sender:c", "name": "Mapleleaf", "category": "promotional_subscription", "received_count": 5, "unread_count": 3, "status": "provisional"},
    ],
    "messages": [
        {"message_id": "m1", "group_key": "list:b", "classification": "editorial_newsletter", "internal_date": 1_778_000_000_000,
         "publication_name": "Byte Garden", "from": "", "subject": "Issue 1", "received": "today"},
    ],
}


def test_overview_and_listing_come_from_the_report():
    assert chat.get_inventory_overview(REPORT)["publications_observed"] == 2
    listed = chat.list_publications(REPORT, category="editorial_newsletter", status="any", sort_by="unread", limit=5)
    assert [p["name"] for p in listed["publications"]] == ["Byte Garden", "The Quiet Ledger"]
    assert chat.get_inventory_overview(None)["error"].startswith("No inventory")


def test_summarize_asks_first_and_does_nothing_when_declined():
    def must_not_summarize(issues):
        raise AssertionError("Declined requests must not read email.")

    result = chat.summarize_newsletters(REPORT, "garden", "editorial_newsletter", 14, ask=lambda q: False,
                                        summarize=must_not_summarize, now=_now())
    assert "declined" in result["result"]


def test_summarize_returns_bullets_with_sources():
    async def fake_summarize(issues):
        return SummaryResult(bullets={"m1": ["A useful point."]})

    result = chat.summarize_newsletters(REPORT, None, "editorial_newsletter", 14, ask=lambda q: True,
                                        summarize=fake_summarize, now=_now())
    assert result["issues"][0]["bullets"] == ["A useful point."]
    assert result["issues"][0]["subject"] == "Issue 1"


def _now():
    from datetime import datetime, timezone
    return datetime.fromtimestamp(1_778_000_000, tz=timezone.utc)


class ScriptedClient:
    """A stand-in for the sync OpenAI client that replays prepared responses."""

    def __init__(self, responses):
        self.queue = list(responses)
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]


def item(**fields):
    return SimpleNamespace(**fields, model_dump=lambda exclude_none=True: dict(fields))


def test_tool_loop_runs_tools_and_returns_the_answer():
    call = item(type="function_call", call_id="call_1", name="list_publications",
                arguments=json.dumps({"category": "any", "status": "any", "sort_by": "received", "limit": 3}))
    client = ScriptedClient([SimpleNamespace(output=[call], output_text=""),
                             SimpleNamespace(output=[item(type="message")], output_text="Byte Garden sends the most.")])
    conversation = [{"role": "user", "content": "What do I get most?"}]
    assert chat.answer_question(client, conversation, REPORT) == "Byte Garden sends the most."
    outputs = [entry for entry in conversation if entry.get("type") == "function_call_output"]
    assert outputs[0]["call_id"] == "call_1" and "Byte Garden" in outputs[0]["output"]
    request = client.requests[0]
    assert request["store"] is False and all(tool["strict"] for tool in request["tools"])


def test_tool_loop_is_capped():
    call = item(type="function_call", call_id="c", name="get_inventory_overview", arguments="{}")
    client = ScriptedClient([SimpleNamespace(output=[call], output_text="")])
    answer = chat.answer_question(client, [], REPORT)
    assert "stopped after several tool calls" in answer
    assert len(client.requests) == chat.MAX_TOOL_ROUNDS


def test_real_sdk_serializes_the_tool_loop():
    """The installed SDK against a local fake server, checking what a real request would contain."""
    bodies = []

    def fake_api(request):
        bodies.append(json.loads(request.content))
        base = {"id": f"resp_{len(bodies)}", "object": "response", "created_at": 1, "model": "gpt-6-luna",
                "status": "completed", "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10,
                          "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}
        if len(bodies) == 1:
            output = [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque"},
                      {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_inventory_overview",
                       "arguments": "{}", "status": "completed"}]
        else:
            output = [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": "You have 2 observed publications.", "annotations": []}]}]
        return httpx.Response(200, json={**base, "output": output})

    client = openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(fake_api)))
    answer = chat.answer_question(client, [{"role": "user", "content": "How many?"}], REPORT)
    assert answer == "You have 2 observed publications."
    first, second = bodies
    assert first["store"] is False and first["include"] == ["reasoning.encrypted_content"]
    types = [entry.get("type") for entry in second["input"]]
    assert types[1:] == ["reasoning", "function_call", "function_call_output"]
    assert second["input"][3]["call_id"] == "call_1"
