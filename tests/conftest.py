from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from app.llm.openrouter import Delta, LLMError
from app.rag.models import Document
from app.rag.retriever import HybridRetriever


def _listing(
    i: int, city: str, purpose: str, ptype: str, beds: int | None, price: float
) -> Document:
    return Document(
        id=f"l{i}",
        source="wasalt",
        kind="listing",
        title=f"{ptype} {i} in {city}",
        url=f"https://wasalt.sa/en/property/{purpose}/x-{i}",
        text=f"{ptype} for {purpose} in {city}. Price: {price:,.0f} SAR. Bedrooms: {beds}.",
        meta={
            "listing_id": i,
            "city": city,
            "purpose": purpose,
            "property_type": ptype,
            "bedrooms": beds,
            "price": price,
        },
    )


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(
            id="p1",
            source="darglobal",
            kind="project",
            title="Trump Tower Jeddah",
            url="https://darglobal.co.uk/trump-tower-jeddah",
            text="Trump Tower Jeddah is a 47-floor seafront tower on the Jeddah Corniche with Red Sea views.",
        ),
        Document(
            id="p2",
            source="darglobal",
            kind="project",
            title="Da Vinci Tower by Pagani",
            url="https://darglobal.co.uk/davinci-tower-by-pagani",
            text="Da Vinci Tower in Dubai offers interiors by Pagani, a pool, gym and concierge.",
        ),
        Document(
            id="f1",
            source="wasalt",
            kind="faq",
            title="Average villa price in Riyadh?",
            url="https://wasalt.sa/en/villas-for-sale-in-riyadh",
            text="Q: Average villa price in Riyadh? A: From around SAR 800,000 to SAR 5 million.",
        ),
        _listing(1, "Riyadh", "sale", "Villa", 4, 1_800_000),
        _listing(2, "Riyadh", "sale", "Villa", 5, 3_500_000),
        _listing(3, "Jeddah", "rent", "Apartment", 3, 45_000),
        _listing(4, "Jeddah", "sale", "Apartment", 2, 900_000),
    ]


@pytest.fixture
def retriever(documents: list[Document]) -> HybridRetriever:
    return HybridRetriever(documents)  # BM25-only: no model download in unit tests


class FakeLLM:
    """Deterministic LLM double that records the messages it receives."""

    def __init__(
        self, chunks: Sequence[str] = ("Hello ", "world [1]."), fail: str | None = None
    ) -> None:
        self.models = ["fake/model:free"]
        self.chunks = list(chunks)
        self.fail = fail
        self.calls: list[list[dict[str, str]]] = []
        self.closed = False

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[Delta]:
        self.calls.append(messages)
        if self.fail:
            raise LLMError(self.fail)
        for c in self.chunks:
            yield Delta(c, self.models[0])

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
