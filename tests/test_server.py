"""Local Agent Workbench server: REST API, SSE streaming, inline approvals, and security."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest
from starlette.testclient import TestClient

from newsletter_agent.server import WorkbenchState, create_app

FIXED_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    event_type = "message"
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            events.append((event_type, json.loads("\n".join(data_lines))))
    return events


def test_demo_rest_endpoints_and_security_headers():
    app = create_app(mode="demo", now=FIXED_NOW)
    with TestClient(app) as client:
        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.headers["x-content-type-options"] == "nosniff"
        assert status.headers["x-frame-options"] == "DENY"
        assert status.headers["referrer-policy"] == "no-referrer"
        assert status.headers["cache-control"] == "no-store"
        body = status.json()
        assert body["mode"] == "demo"
        assert body["has_inventory"] is True and body["has_briefing"] is True
        assert body["indexed_documents"] > 0
        assert set(body["tools"]) == {
            "get_inventory_overview", "list_publications", "search_briefings", "summarize_newsletters",
        }

        inv = client.get("/api/inventory").json()
        assert inv["available"] is True
        assert inv["report"]["counts"]["publications_observed"] == 7

        brief = client.get("/api/briefings").json()
        assert brief["available"] is True
        assert brief["latest"]["counts"]["summarized"] > 0

        # Lexical + hybrid search
        res = client.get("/api/search", params={"q": "river otters", "mode": "hybrid"})
        assert res.status_code == 200
        hits = res.json()["results"]
        assert hits and hits[0]["publication"] == "Field Notes Daily"

        # Conceptual semantic search with zero exact keyword overlap
        sem = client.get("/api/search", params={"q": "central bank monetary policy cpi", "mode": "hybrid"})
        assert sem.status_code == 200
        assert any(hit["publication"] == "The Quiet Ledger" for hit in sem.json()["results"])

        # Bad search params
        assert client.get("/api/search", params={"q": "x", "mode": "bogus"}).status_code == 400
        assert client.get("/api/search", params={"q": "x", "days": "0"}).status_code == 400


def test_csrf_origin_guard_and_input_validation():
    app = create_app(mode="demo", now=FIXED_NOW)
    with TestClient(app) as client:
        blocked = client.post(
            "/api/chat/stream",
            json={"message": "hello"},
            headers={"Origin": "https://evil.example"},
        )
        assert blocked.status_code == 403

        allowed = client.post(
            "/api/chat/stream",
            json={"message": "Give me an overview of my inventory"},
            headers={"Origin": "http://127.0.0.1:8765"},
        )
        assert allowed.status_code == 200

        assert client.post("/api/chat/stream", json={"message": "   "}).status_code == 400
        assert client.post("/api/chat/approve", json={"approval_id": "missing", "approved": True}).status_code == 404


def test_demo_chat_stream_emits_tool_trace_tokens_and_citations():
    app = create_app(mode="demo", now=FIXED_NOW)
    with TestClient(app) as client:
        resp = client.post(
            "/api/chat/stream",
            json={"message": "Which newsletter mentioned solar sails?", "session_id": "s1"},
        )
        assert resp.status_code == 200
        events = parse_sse(resp.text)
        types = [ev_type for ev_type, _ in events]
        assert types[0] == "session"
        assert "tool_call" in types and "tool_result" in types
        assert "token" in types and types[-1] == "done"

        tool_call = dict(events)["tool_call"]
        assert tool_call["name"] == "search_briefings"
        done = dict(events)["done"]
        assert "Orbit Weekly" in done["answer"]
        assert any(c["publication"] == "Orbit Weekly" for c in done["citations"])

        assert client.post("/api/chat/reset", json={"session_id": "s1"}).status_code == 200


@pytest.mark.parametrize("decision", [True, False])
def test_inline_human_in_the_loop_approval_flow(decision):
    import socket

    import uvicorn

    async def run():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        port = sock.getsockname()[1]

        state = WorkbenchState(mode="demo", now=FIXED_NOW, approval_timeout=10.0)
        app = create_app(mode="demo", state=state)
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        server_task = asyncio.create_task(server.serve(sockets=[sock]))
        try:
            while not server.started:
                await asyncio.sleep(0.02)

            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=15.0) as client:
                collected: list[tuple[str, dict]] = []
                approval_posted = False

                async with client.stream(
                    "POST",
                    "/api/chat/stream",
                    json={"message": "Summarize Byte Garden from the last 7 days", "session_id": "apr-test"},
                ) as response:
                    assert response.status_code == 200
                    current_event = "message"
                    async for line in response.aiter_lines():
                        if line.startswith("event:"):
                            current_event = line.split(":", 1)[1].strip()
                        elif line.startswith("data:"):
                            payload = json.loads(line.split(":", 1)[1].strip())
                            collected.append((current_event, payload))
                            if current_event == "approval_required" and not approval_posted:
                                approval_posted = True
                                ack = await client.post(
                                    "/api/chat/approve",
                                    json={"approval_id": payload["approval_id"], "approved": decision},
                                )
                                assert ack.status_code == 200 and ack.json()["approved"] is decision

                assert approval_posted is True
                by_type = {ev_type: payload for ev_type, payload in collected}
                assert by_type["approval_required"]["tool_name"] == "summarize_newsletters"
                assert by_type["approval_resolved"]["approved"] is decision
                assert by_type["tool_result"]["name"] == "summarize_newsletters"
                if decision:
                    assert len(by_type["tool_result"]["output"]["issues"]) >= 1
                    assert "Byte Garden" in by_type["done"]["answer"]
                    assert by_type["done"]["citations"]
                else:
                    assert "declined" in by_type["tool_result"]["output"]["result"].lower()
                    assert "declined" in by_type["done"]["answer"].lower()
                assert state.pending_approvals == {}
        finally:
            server.should_exit = True
            await server_task
            sock.close()

    asyncio.run(run())


def test_live_async_agent_streams_tokens_and_runs_tools():
    class FakeStreamingOpenAI:
        def __init__(self):
            self.turn = 0
            self.responses = self

        async def create(self, **kwargs):
            self.turn += 1
            if self.turn == 1:
                call = SimpleNamespace(
                    type="function_call",
                    call_id="call_live_1",
                    name="get_inventory_overview",
                    arguments="{}",
                    model_dump=lambda exclude_none=True: {
                        "type": "function_call",
                        "call_id": "call_live_1",
                        "name": "get_inventory_overview",
                        "arguments": "{}",
                    },
                )
                return SimpleNamespace(
                    output=[call],
                    output_text="",
                    usage=SimpleNamespace(input_tokens=20, output_tokens=10, total_tokens=30),
                )

            async def token_stream():
                for chunk in ("You have ", "7 observed ", "publications."):
                    yield SimpleNamespace(type="response.output_text.delta", delta=chunk)
                msg = SimpleNamespace(
                    type="message",
                    model_dump=lambda exclude_none=True: {"type": "message"},
                )
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        output=[msg],
                        output_text="You have 7 observed publications.",
                        usage=SimpleNamespace(input_tokens=30, output_tokens=12, total_tokens=42),
                    ),
                )

            return token_stream()

    async def run():
        demo_state = WorkbenchState(mode="demo", now=FIXED_NOW)
        await demo_state.startup()
        try:
            # Reuse the populated report/index in live mode with a mocked AsyncOpenAI client.
            demo_state.mode = "live"
            demo_state.openai_client_factory = FakeStreamingOpenAI
            app = create_app(mode="live", state=demo_state)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8765") as client:
                resp = await client.post("/api/chat/stream", json={"message": "How many publications?"})
                assert resp.status_code == 200
                events = parse_sse(resp.text)
                deltas = [payload["delta"] for ev_type, payload in events if ev_type == "token"]
                assert deltas == ["You have ", "7 observed ", "publications."]
                done = dict(events)["done"]
                assert done["answer"] == "You have 7 observed publications."
                assert done["usage"]["total_tokens"] == 72
        finally:
            await demo_state.shutdown()

    asyncio.run(run())


def test_spa_serving_and_path_traversal_guard(tmp_path):
    dist = tmp_path / "ui_dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><div id='root'>OK</div>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")

    app = create_app(mode="demo", now=FIXED_NOW, ui_dist_dir=dist)
    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200 and "id='root'" in root.text

        # Traversal attempt falls back to index.html and never serves parent files
        escaped = client.get("/../secret.txt")
        assert escaped.status_code == 200 and "TOP SECRET" not in escaped.text
