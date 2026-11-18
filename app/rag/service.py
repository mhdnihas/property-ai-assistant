"""Chat orchestration: retrieve -> build prompt -> stream LLM answer as typed events."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from app.llm.openrouter import LLMClient, LLMError
from app.rag.prompt import SYSTEM_PROMPT, build_user_prompt
from app.rag.query_parser import QueryFilters, parse_query
from app.rag.retriever import HybridRetriever, RetrievalResult

log = logging.getLogger(__name__)

FOLLOW_UP_MAX_WORDS = 8


@dataclass(slots=True, frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True)
class ChatEvent:
    type: Literal["sources", "token", "done", "error"]
    data: dict[str, Any] = field(default_factory=dict)


def _merge_filters(current: QueryFilters, previous: QueryFilters) -> QueryFilters:
    """For short follow-ups ("what about Jeddah?"), inherit unspecified constraints."""
    merged = replace(current)
    for name in (
        "city",
        "purpose",
        "property_type",
        "bedrooms",
        "min_bedrooms",
        "min_price",
        "max_price",
    ):
        if getattr(merged, name) is None:
            setattr(merged, name, getattr(previous, name))
    merged.listing_intent = current.listing_intent or previous.listing_intent
    return merged


class ChatService:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient | None,
        *,
        top_k: int = 8,
        max_history_turns: int = 6,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_history_turns = max_history_turns

    def retrieve(self, message: str, history: list[Turn]) -> RetrievalResult:
        filters = parse_query(message)
        query = message
        prev_user = next((t.content for t in reversed(history) if t.role == "user"), None)
        if prev_user and len(message.split()) <= FOLLOW_UP_MAX_WORDS:
            query = f"{prev_user} {message}"
            filters = _merge_filters(filters, parse_query(prev_user))
        return self.retriever.search(query, filters, k=self.top_k)

    def build_messages(
        self, message: str, history: list[Turn], result: RetrievalResult
    ) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history[-self.max_history_turns * 2 :]:
            msgs.append({"role": turn.role, "content": turn.content[:2000]})
        msgs.append(
            {
                "role": "user",
                "content": build_user_prompt(
                    message, result.documents, result.filters, result.matching_listings
                ),
            }
        )
        return msgs

    @staticmethod
    def sources_payload(result: RetrievalResult) -> list[dict[str, Any]]:
        out = []
        for n, sd in enumerate(result.documents, start=1):
            d = sd.document
            item: dict[str, Any] = {
                "n": n,
                "title": d.title,
                "url": d.url,
                "source": d.source,
                "kind": d.kind,
            }
            if d.kind == "listing":
                item |= {
                    k: d.meta.get(k) for k in ("city", "district", "purpose", "price", "bedrooms")
                }
            out.append(item)
        return out

    async def stream(self, message: str, history: list[Turn]) -> AsyncIterator[ChatEvent]:
        t0 = time.perf_counter()
        result = self.retrieve(message, history)
        yield ChatEvent("sources", {"sources": self.sources_payload(result)})

        if self.llm is None:
            yield ChatEvent(
                "error", {"message": "The AI model is not configured (missing OPENROUTER_API_KEY)."}
            )
            return

        model = None
        n_chars = 0
        try:
            async for delta in self.llm.stream_chat(self.build_messages(message, history, result)):
                model = delta.model
                n_chars += len(delta.text)
                yield ChatEvent("token", {"t": delta.text})
        except LLMError as exc:
            log.warning("llm failure: %s", exc)
            yield ChatEvent("error", {"message": str(exc)})
            return
        except Exception:
            log.exception("unexpected error while streaming")
            yield ChatEvent("error", {"message": "Something went wrong. Please try again."})
            return

        elapsed = time.perf_counter() - t0
        log.info(
            "answered in %.2fs model=%s chars=%d docs=%d",
            elapsed,
            model,
            n_chars,
            len(result.documents),
        )
        yield ChatEvent("done", {"model": model, "elapsed_s": round(elapsed, 2)})
