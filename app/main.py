"""FastAPI application: chat API (Server-Sent Events)."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.llm.openrouter import OpenRouterClient
from app.rag.retriever import HybridRetriever
from app.rag.service import ChatEvent, ChatService, Turn

log = logging.getLogger("app")


# --------------------------------------------------------------------------- schemas
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)


# --------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """In-memory sliding-window limiter per client. Fine for a single instance;
    swap for Redis when scaling horizontally."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> float | None:
        """Record a hit. Returns None if allowed, else seconds until retry."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_s:
            hits.popleft()
        if len(hits) >= self.limit:
            return self.window_s - (now - hits[0])
        hits.append(now)
        return None


def client_ip(request: Request) -> str:
    # Hugging Face Spaces (and most PaaS) sit behind a proxy that sets X-Forwarded-For.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# --------------------------------------------------------------------------- dependencies
def get_service(request: Request) -> ChatService:
    return request.app.state.service


def rate_limit(request: Request) -> None:
    retry = request.app.state.limiter.check(client_ip(request))
    if retry is not None:
        raise HTTPException(
            429,
            "Too many requests — please wait a moment.",
            headers={"Retry-After": str(int(retry) + 1)},
        )


# --------------------------------------------------------------------------- wiring
def _build_service(settings: Settings) -> ChatService:
    embedder = None
    if settings.enable_embeddings:
        try:
            from app.rag.embeddings import Embedder

            embedder = Embedder(settings.embedding_model, settings.embedding_cache_dir)
        except Exception:
            log.exception("could not load embedding model — using BM25-only retrieval")
    retriever = HybridRetriever.from_index_dir(settings.index_dir, embedder)

    llm = None
    if settings.openrouter_api_key:
        llm = OpenRouterClient(
            settings.openrouter_api_key.get_secret_value(),
            settings.models,
            base_url=settings.openrouter_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
            referer=settings.public_url,
            title=settings.app_title,
        )
    else:
        log.warning("OPENROUTER_API_KEY is not set — chat will return a configuration error")
    return ChatService(
        retriever, llm, top_k=settings.top_k, max_history_turns=settings.max_history_turns
    )


def _manifest(settings: Settings) -> dict[str, Any]:
    path = settings.index_dir / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def create_app(settings: Settings | None = None, service: ChatService | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = service or _build_service(settings)
        app.state.manifest = _manifest(settings)
        app.state.limiter = RateLimiter(settings.rate_limit_per_minute)
        log.info(
            "ready: %d documents, dense=%s, llm=%s",
            len(app.state.service.retriever.documents),
            app.state.service.retriever.dense_enabled,
            app.state.service.llm is not None,
        )
        yield
        if app.state.service.llm is not None:
            await app.state.service.llm.aclose()

    app = FastAPI(
        title=settings.app_title,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.url.path.startswith("/api/"):
            log.info(
                "%s %s %s %.0fms rid=%s",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - t0) * 1000,
                rid,
            )
        return response

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        svc: ChatService = request.app.state.service
        return {
            "status": "ok",
            "documents": len(svc.retriever.documents),
            "dense_retrieval": svc.retriever.dense_enabled,
            "llm_configured": svc.llm is not None,
            "models": list(svc.llm.models) if svc.llm else [],
            "index": request.app.state.manifest,
        }

    @app.post("/api/chat", dependencies=[Depends(rate_limit)])
    async def chat(
        body: ChatRequest, svc: Annotated[ChatService, Depends(get_service)]
    ) -> StreamingResponse:
        message = body.message.strip()
        if not message:
            raise HTTPException(422, "Message is empty.")
        if len(message) > settings.max_message_chars:
            raise HTTPException(
                422, f"Message is too long (max {settings.max_message_chars} characters)."
            )
        history = [Turn(m.role, m.content) for m in body.history]

        async def events() -> AsyncIterator[bytes]:
            async for ev in svc.stream(message, history):
                yield _sse(ev)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    return app


def _sse(event: ChatEvent) -> bytes:
    return f"event: {event.type}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n".encode()


app = create_app()  # uvicorn app.main:app
