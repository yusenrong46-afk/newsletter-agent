"""Local Agent Workbench server (`newsletter-agent ui [--demo]`).

Opt-in local Starlette + Uvicorn server bound strictly to 127.0.0.1.
Static HTML reports (`newsletter-agent report` and `brief`) remain zero-port;
this server powers the interactive React Agent Workbench with:

  - REST endpoints for inventory, briefings, and hybrid semantic search
  - Server-Sent Events (SSE) streaming for the async tool-using agent
  - Inline human-in-the-loop approval for `summarize_newsletters`
  - Zero-credential `--demo` mode running the real local tools + in-memory MCP server
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
import webbrowser
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import config
from .briefing import Budget, run_briefing, summarize_issues
from .chat import (
    CHAT_SUMMARY_BUDGET,
    TOOLS,
    stream_agent_turn,
    stream_demo_agent_turn,
    summarize_with_gmail,
)
from .classify import OPENAI_SDK_RETRIES, OPENAI_TIMEOUT_SECONDS
from .demo import OfflineModel, SyntheticGmail, build_demo_mailbox, demo_gmail_client
from .inventory import run_inventory
from .scan import InventoryOptions
from .search import HybridSearchIndex, SearchMode, sync_saved_artifacts_to_index
from .store import InventoryStore, list_briefing_paths, load_latest_briefing, load_latest_report

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_SESSIONS = 50
MAX_SESSION_ITEMS = 120
MAX_MESSAGE_CHARS = 4000
DEFAULT_APPROVAL_TIMEOUT = 120.0
LOCAL_ORIGINS = frozenset({"127.0.0.1", "localhost"})

UI_DIST_DIR = Path(__file__).parent / "ui_dist"


def _is_allowed_origin(origin: str) -> bool:
    parsed = urlparse(origin)
    return parsed.scheme in ("http", "https") and (parsed.hostname or "").lower() in LOCAL_ORIGINS


class LocalSecurityMiddleware(BaseHTTPMiddleware):
    """Add strict security headers and reject cross-origin POSTs to localhost APIs."""

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method in ("POST", "PUT", "PATCH", "DELETE") and request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin and not _is_allowed_origin(origin):
                return JSONResponse({"error": "Cross-origin requests are not allowed."}, status_code=403)
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response


def _sse_frame(event_type: str, payload: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class WorkbenchState:
    """Holds server state: SQLite store, search index, reports, sessions, and approval futures."""

    def __init__(
        self,
        *,
        mode: Literal["demo", "live"] = "demo",
        approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT,
        now: datetime | None = None,
        openai_client_factory: Callable[[], Any] | None = None,
        ui_dist_dir: Path | None = None,
    ):
        self.mode: Literal["demo", "live"] = mode
        self.approval_timeout = approval_timeout
        self.now = now
        self.openai_client_factory = openai_client_factory
        self.ui_dist_dir = ui_dist_dir or UI_DIST_DIR

        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.store: InventoryStore | None = None
        self.search_index: HybridSearchIndex | None = None
        self.report: dict[str, Any] | None = None
        self.briefing: dict[str, Any] | None = None
        self.briefing_history: list[dict[str, Any]] = []
        self.demo_mailbox: SyntheticGmail | None = None
        self.sessions: dict[str, list[dict[str, Any]]] = {}
        self.pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._ready = False

    async def startup(self) -> None:
        if self._ready:
            return
        if self.mode == "demo":
            await self._startup_demo()
        else:
            self._startup_live()
        self._ready = True

    async def _startup_demo(self) -> None:
        moment = self.now or datetime.now(UTC)
        self.now = moment
        self._temp_dir = tempfile.TemporaryDirectory(prefix="newsletter-agent-ui-demo-")
        root = Path(self._temp_dir.name)
        briefings_dir = root / "briefings"
        briefings_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

        mailbox = build_demo_mailbox(moment)
        self.demo_mailbox = mailbox
        model = OfflineModel()
        settings: dict[str, Any] = dict(config.DEFAULT_CONFIG["briefing"])

        store = InventoryStore(root / "inventory")
        briefing_store = InventoryStore(root / "inventory", checkpoint_name="briefing_checkpoint.json")
        try:
            async with demo_gmail_client(mailbox) as gmail:
                self.report = await run_inventory(
                    gmail,
                    InventoryOptions(days=90, approved=True, quiet=True),
                    store,
                    openai_factory=lambda: model,
                    now=moment,
                    demo=True,
                )
                self.briefing = await run_briefing(
                    gmail,
                    briefing_store,
                    settings,
                    briefings_dir,
                    openai_factory=lambda: model,
                    approved=True,
                    now=moment,
                    demo=True,
                    remember=False,
                )
        finally:
            briefing_store.close()

        self.store = store
        self.search_index = HybridSearchIndex(store)
        if self.briefing:
            self.briefing_history = [self._briefing_summary(self.briefing)]

    def _startup_live(self) -> None:
        self.report = load_latest_report()
        self.briefing = load_latest_briefing()
        self.store = InventoryStore()
        self.search_index = HybridSearchIndex(self.store)
        if self.search_index.count_documents() == 0:
            sync_saved_artifacts_to_index(self.store)
        history: list[dict[str, Any]] = []
        for path in reversed(list_briefing_paths()[-14:]):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            history.append(self._briefing_summary(data, filename=path.name))
        self.briefing_history = history

    @staticmethod
    def _briefing_summary(briefing: dict[str, Any], filename: str = "") -> dict[str, Any]:
        counts = briefing.get("counts", {})
        return {
            "date": str(briefing.get("generated_at", ""))[:10],
            "filename": filename,
            "window": briefing.get("window", ""),
            "summarized": counts.get("summarized", 0),
            "publications": counts.get("publications", 0),
            "top_stories": counts.get("top_stories", len(briefing.get("top_stories", []))),
        }

    async def shutdown(self) -> None:
        for future in list(self.pending_approvals.values()):
            if not future.done():
                future.set_result(False)
        self.pending_approvals.clear()
        if self.store is not None:
            self.store.close()
            self.store = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None
        self._ready = False

    def get_session(self, session_id: str) -> list[dict[str, Any]]:
        if session_id not in self.sessions:
            if len(self.sessions) >= MAX_SESSIONS:
                oldest = next(iter(self.sessions))
                del self.sessions[oldest]
            self.sessions[session_id] = []
        history = self.sessions[session_id]
        if len(history) > MAX_SESSION_ITEMS:
            del history[:-MAX_SESSION_ITEMS]
        return history

    def make_openai_client(self) -> Any:
        if self.openai_client_factory is not None:
            return self.openai_client_factory()
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=config.get_api_key(),
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=OPENAI_SDK_RETRIES,
        )

    async def summarize_for_mode(self, issues: list[dict]) -> Any:
        if self.mode == "demo" and self.demo_mailbox is not None:
            model = OfflineModel()
            budget = Budget(CHAT_SUMMARY_BUDGET.max_calls, CHAT_SUMMARY_BUDGET.max_tokens)
            async with demo_gmail_client(self.demo_mailbox) as gmail:
                return await summarize_issues(gmail, model, issues, budget)
        return await summarize_with_gmail(issues)

    def create_approval(self, request_event: dict[str, Any]) -> asyncio.Task[bool]:
        """Register the approval future immediately (before SSE yield) and return a task awaiting it."""
        approval_id = str(request_event["approval_id"])
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self.pending_approvals[approval_id] = future

        async def _wait() -> bool:
            try:
                return bool(await asyncio.wait_for(future, timeout=self.approval_timeout))
            except TimeoutError:
                return False
            finally:
                self.pending_approvals.pop(approval_id, None)

        return loop.create_task(_wait())

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        future = self.pending_approvals.get(approval_id)
        if future is None or future.done():
            return False
        future.set_result(bool(approved))
        return True


def _placeholder_html(mode: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Newsletter Agent Workbench</title></head>
<body style="font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;">
<h1>Newsletter Agent Workbench API ({mode})</h1>
<p>The backend API is running on localhost. Build the React frontend in <code>web/</code> to load the full UI.</p>
<ul>
  <li><code>GET /api/status</code></li>
  <li><code>GET /api/inventory</code></li>
  <li><code>GET /api/briefings</code></li>
  <li><code>GET /api/search?q=otters&amp;mode=hybrid</code></li>
  <li><code>POST /api/chat/stream</code> (SSE)</li>
  <li><code>POST /api/chat/approve</code></li>
</ul>
</body></html>"""


def create_app(
    *,
    mode: Literal["demo", "live"] = "demo",
    state: WorkbenchState | None = None,
    approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT,
    now: datetime | None = None,
    openai_client_factory: Callable[[], Any] | None = None,
    ui_dist_dir: Path | None = None,
) -> Starlette:
    """Create the Starlette ASGI application for the local Agent Workbench."""
    wb = state or WorkbenchState(
        mode=mode,
        approval_timeout=approval_timeout,
        now=now,
        openai_client_factory=openai_client_factory,
        ui_dist_dir=ui_dist_dir,
    )

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        await wb.startup()
        try:
            yield
        finally:
            await wb.shutdown()

    async def api_status(_request: Request) -> JSONResponse:
        doc_count = wb.search_index.count_documents() if wb.search_index else 0
        models = wb.search_index.stored_embedding_models() if wb.search_index else {}
        return JSONResponse({
            "mode": wb.mode,
            "model": "offline-demo-agent" if wb.mode == "demo" else config.MODEL,
            "has_api_key": config.has_api_key(),
            "has_inventory": wb.report is not None,
            "has_briefing": wb.briefing is not None,
            "indexed_documents": doc_count,
            "embedding_models": models,
            "tools": [tool["name"] for tool in TOOLS],
        })

    async def api_inventory(_request: Request) -> JSONResponse:
        if wb.report is None:
            return JSONResponse({"available": False, "report": None})
        return JSONResponse({"available": True, "report": wb.report})

    async def api_briefings(_request: Request) -> JSONResponse:
        return JSONResponse({
            "available": wb.briefing is not None,
            "latest": wb.briefing,
            "history": wb.briefing_history,
        })

    async def api_search(request: Request) -> JSONResponse:
        query = request.query_params.get("q", "").strip()
        raw_mode = request.query_params.get("mode", "hybrid").strip().lower()
        if raw_mode not in ("hybrid", "lexical", "vector"):
            return JSONResponse({"error": "mode must be hybrid, lexical, or vector."}, status_code=400)
        search_mode: SearchMode = raw_mode  # type: ignore[assignment]
        publication = request.query_params.get("publication", "").strip() or None
        raw_days = request.query_params.get("days", "").strip()
        raw_limit = request.query_params.get("limit", "10").strip()
        try:
            days = int(raw_days) if raw_days else None
            limit = int(raw_limit)
        except ValueError:
            return JSONResponse({"error": "days and limit must be integers."}, status_code=400)
        if days is not None and days < 1:
            return JSONResponse({"error": "days must be at least 1."}, status_code=400)
        limit = min(max(limit, 1), 30)

        if not query or wb.search_index is None:
            return JSONResponse({"query": query, "mode": search_mode, "total_results": 0, "results": []})

        ref_ms = int(wb.now.timestamp() * 1000) if wb.now is not None else None
        hits = wb.search_index.search(
            query, mode=search_mode, publication=publication, days=days, limit=limit, now_ms=ref_ms,
        )
        return JSONResponse({
            "query": query,
            "mode": search_mode,
            "publication": publication,
            "days": days,
            "total_results": len(hits),
            "results": [hit.to_dict() for hit in hits],
        })

    async def api_chat_stream(request: Request) -> Response:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)
        if not isinstance(body, dict):
            return JSONResponse({"error": "Request body must be a JSON object."}, status_code=400)
        message = str(body.get("message", "")).strip()
        if not message:
            return JSONResponse({"error": "message must not be empty."}, status_code=400)
        if len(message) > MAX_MESSAGE_CHARS:
            return JSONResponse({"error": f"message exceeds {MAX_MESSAGE_CHARS} characters."}, status_code=400)

        session_id = str(body.get("session_id") or f"sess_{uuid.uuid4().hex[:10]}")
        conversation = wb.get_session(session_id)

        async def event_stream() -> AsyncIterator[str]:
            yield _sse_frame("session", {"session_id": session_id, "mode": wb.mode})
            try:
                if wb.mode == "demo":
                    stream = stream_demo_agent_turn(
                        message,
                        conversation,
                        report=wb.report,
                        search_index=wb.search_index,
                        request_approval=wb.create_approval,
                        summarize_fn=wb.summarize_for_mode,
                        now=wb.now,
                    )
                else:
                    try:
                        client = wb.make_openai_client()
                    except (ValueError, OSError) as error:
                        yield _sse_frame("error", {"message": f"OpenAI API key is not configured: {error}"})
                        return
                    stream = stream_agent_turn(
                        client,
                        message,
                        conversation,
                        report=wb.report,
                        search_index=wb.search_index,
                        request_approval=wb.create_approval,
                        summarize_fn=wb.summarize_for_mode,
                        now=wb.now,
                    )

                async for event in stream:
                    yield _sse_frame(str(event.get("type", "message")), event)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001
                yield _sse_frame("error", {"message": f"Agent stream failed: {type(error).__name__}: {error}"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def api_chat_approve(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            return JSONResponse({"error": "Request body must be valid JSON."}, status_code=400)
        if not isinstance(body, dict) or "approval_id" not in body or "approved" not in body:
            return JSONResponse({"error": "approval_id and approved are required."}, status_code=400)
        approval_id = str(body["approval_id"])
        approved = bool(body["approved"])
        resolved = wb.resolve_approval(approval_id, approved)
        if not resolved:
            return JSONResponse({"error": "Unknown or expired approval_id."}, status_code=404)
        return JSONResponse({"ok": True, "approval_id": approval_id, "approved": approved})

    async def api_chat_reset(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            body = {}
        session_id = str(body.get("session_id", "")) if isinstance(body, dict) else ""
        if session_id and session_id in wb.sessions:
            del wb.sessions[session_id]
        return JSONResponse({"ok": True})

    async def serve_spa(request: Request) -> Response:
        dist = wb.ui_dist_dir.resolve()
        index_html = dist / "index.html"
        if not index_html.is_file():
            return HTMLResponse(_placeholder_html(wb.mode))
        rel = request.path_params.get("path", "").lstrip("/")
        if rel:
            candidate = (dist / rel).resolve()
            if candidate.is_relative_to(dist) and candidate.is_file():
                return FileResponse(candidate)
        return FileResponse(index_html)

    routes = [
        Route("/api/status", api_status, methods=["GET"]),
        Route("/api/inventory", api_inventory, methods=["GET"]),
        Route("/api/briefings", api_briefings, methods=["GET"]),
        Route("/api/search", api_search, methods=["GET"]),
        Route("/api/chat/stream", api_chat_stream, methods=["POST"]),
        Route("/api/chat/approve", api_chat_approve, methods=["POST"]),
        Route("/api/chat/reset", api_chat_reset, methods=["POST"]),
        Route("/", serve_spa, methods=["GET"]),
        Route("/{path:path}", serve_spa, methods=["GET"]),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    app.add_middleware(LocalSecurityMiddleware)
    app.state.workbench = wb
    return app


def run_ui_command(demo: bool = False, port: int = DEFAULT_PORT, open_browser: bool = True) -> int:
    """CLI handler for `newsletter-agent ui [--demo] [--port 8765] [--no-open]`."""
    import uvicorn

    if not 1 <= port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    mode: Literal["demo", "live"] = "demo" if demo else "live"
    url = f"http://{DEFAULT_HOST}:{port}"
    print(f"Starting Newsletter Agent Workbench ({mode} mode) on {url}")
    if mode == "demo":
        print("Demo mode: synthetic mailbox, real local tools and in-memory MCP server. No Gmail or OpenAI key needed.")
    if open_browser:
        webbrowser.open(url)
    app = create_app(mode=mode)
    uvicorn.run(app, host=DEFAULT_HOST, port=port, log_level="warning")
    return 0
