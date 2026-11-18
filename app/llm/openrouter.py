"""Async streaming client for OpenRouter's OpenAI-compatible chat completions API.

Free models are frequently rate-limited or temporarily unavailable, so the client
tries a list of models in order and falls back to the next one on retryable
errors — but only *before* the first token is sent to the user.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({400, 404, 408, 409, 425, 429, 500, 502, 503, 504})
# 400/404 included: a free model that was renamed/removed answers with these.


class LLMError(RuntimeError):
    """Raised when no model could produce an answer. Message is safe to show users."""


class _ModelFailed(Exception):
    pass


@dataclass(slots=True, frozen=True)
class Delta:
    text: str
    model: str


class LLMClient(Protocol):
    models: Sequence[str]

    def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[Delta]: ...

    async def aclose(self) -> None: ...


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        models: Sequence[str],
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.2,
        max_tokens: int = 1500,
        disable_reasoning: bool = True,
        timeout_s: float = 60.0,
        referer: str | None = None,
        title: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one model is required")
        self.models = list(models)
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Reasoning models spend hidden "thinking" tokens out of the same max_tokens
        # budget, which truncated answers. Grounded RAG answers don't need it.
        self._disable_reasoning = disable_reasoning
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[Delta]:
        for model in self.models:
            started = False
            try:
                async for text in self._stream_one(model, messages):
                    started = True
                    yield Delta(text, model)
                if started:
                    return
                log.warning("model %s returned an empty response; trying next", model)
            except _ModelFailed as exc:
                if started:
                    raise LLMError("The answer was interrupted. Please try again.") from exc
                log.warning("model %s unavailable: %s", model, exc)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if started:
                    raise LLMError("The answer was interrupted. Please try again.") from exc
                log.warning("model %s network error: %r", model, exc)
        raise LLMError("All AI models are busy right now. Please try again in a minute.")

    async def _stream_one(self, model: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._disable_reasoning:
            payload["reasoning"] = {"enabled": False}
        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code in (401, 403):
                raise LLMError("The AI service rejected the API key. Check OPENROUTER_API_KEY.")
            if resp.status_code != 200:
                body = (await resp.aread())[:300].decode(errors="replace")
                if resp.status_code in RETRYABLE_STATUS or resp.status_code == 402:
                    raise _ModelFailed(f"HTTP {resp.status_code}: {body}")
                raise LLMError(f"AI service error (HTTP {resp.status_code}).")
            async for text in iter_sse_content(resp.aiter_lines()):
                yield text


async def iter_sse_content(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield content deltas from an OpenAI-style SSE stream.

    Handles keep-alive comments (": OPENROUTER PROCESSING"), [DONE], and
    mid-stream error objects.
    """
    async for raw in lines:
        line = raw.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            log.debug("skipping malformed SSE line: %.100s", data)
            continue
        if err := chunk.get("error"):
            raise _ModelFailed(
                f"stream error: {err.get('message', err) if isinstance(err, dict) else err}"
            )
        for choice in chunk.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                yield content
            if choice.get("finish_reason") == "length":
                log.warning("answer truncated by max_tokens (model=%s)", chunk.get("model"))
