"""Core data types shared by the pipeline and the API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Source = Literal["darglobal", "wasalt"]
Kind = Literal["project", "page", "catalogue", "listing", "faq", "market_guide", "market_stats"]


@dataclass(slots=True)
class Document:
    """A retrievable unit of text with metadata used for filtering and citations."""

    id: str
    source: Source
    kind: Kind
    title: str
    url: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        return cls(**d)

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n{self.text}"


@dataclass(slots=True, frozen=True)
class ScoredDocument:
    document: Document
    score: float
