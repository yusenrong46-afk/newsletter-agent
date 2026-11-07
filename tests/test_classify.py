"""Classification: every bad model answer becomes an explicit unresolved item."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from helpers import FakeOpenAI, default_classifier, fake_gmail, fake_response, newsletter, run_workflow
from newsletter_agent.classify import ClassificationBatch, MessageClassification


def label(message_id, category="editorial_newsletter"):
    return MessageClassification(message_id=message_id, classification=category,
                                 publication_name="Pub", confidence="high", rationale="x")


def test_bad_ids_become_unresolved_and_valid_items_survive(tmp_path):
    def bad_batch(items):
        return ClassificationBatch(classifications=[label("good"), label("dup"), label("dup"), label("invented-id")])

    report, _ = run_workflow(tmp_path, fake_gmail([newsletter("good"), newsletter("dup"), newsletter("missing")]),
                             FakeOpenAI(bad_batch))
    assert report["counts"]["classified"] == 1
    assert {item["message_id"]: item["reason"] for item in report["unresolved"]} == {
        "dup": "duplicate_id_in_response", "missing": "missing_from_response"}
    assert report["usage"]["fabricated_ids_discarded"] == 1
    assert report["run"]["status"] == "partial_failure"
    assert report["reconciliation"]["ok"]
    assert sum(report["category_breakdown"].values()) == 1  # Failures are not counted as any category.


def test_unknown_category_is_rejected_by_the_schema(tmp_path):
    def invalid(items):
        ClassificationBatch.model_validate({"classifications": [{
            "message_id": items[0]["id"], "classification": "spam", "publication_name": None,
            "confidence": "high", "rationale": "x"}]})

    report, _ = run_workflow(tmp_path, fake_gmail([newsletter("a")]), FakeOpenAI(invalid))
    assert report["unresolved"][0]["reason"] == "invalid_output"


@pytest.mark.parametrize("response, reason", [
    (fake_response(None, status="incomplete", incomplete_reason="max_output_tokens"), "incomplete_response:max_output_tokens"),
    (fake_response(None, output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])]), "refused"),
    (fake_response(None), "invalid_output"),
])
def test_refused_incomplete_and_empty_responses_are_unresolved(tmp_path, response, reason):
    report, _ = run_workflow(tmp_path, fake_gmail([newsletter("a")]), FakeOpenAI(lambda items: response))
    assert report["unresolved"][0]["reason"] == reason
    assert report["run"]["status"] == "partial_failure"
    assert report["usage"]["total_for_this_window"]["total_tokens"] == 120


def test_real_sdk_request_and_response_shape(tmp_path):
    """The installed OpenAI SDK against a local fake HTTP server; nothing leaves the machine."""
    captured = {}

    def fake_api(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        items = json.loads(captured["body"]["input"])["messages"]
        text = ClassificationBatch(classifications=[default_classifier(i) for i in items]).model_dump_json()
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 1, "model": "gpt-6-luna",
            "status": "completed", "parallel_tool_calls": False, "tool_choice": "none", "tools": [],
            "output": [{"id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        })

    client = openai.AsyncOpenAI(api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake_api)))
    report, _ = run_workflow(tmp_path, fake_gmail([newsletter("a")]), client)
    body = captured["body"]
    assert captured["path"].endswith("/responses")
    assert body["model"] == "gpt-6-luna" and body["store"] is False
    assert not body.get("tools")  # The classifier cannot run tools.
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"] is True
    assert report["counts"]["classified"] == 1
    assert report["usage"]["total_for_this_window"]["total_tokens"] == 60
