"""Hybrid retrieval: BM25 (exact names, IDs, numbers) + dense embeddings (meaning),
fused with Reciprocal Rank Fusion, plus hard metadata filters for listings.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.models import Document, ScoredDocument
from app.rag.query_parser import QueryFilters

log = logging.getLogger(__name__)

RRF_K = 60  # standard constant from Cormack et al. (2009)
CANDIDATES = 50

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "show",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
        "about",
        "any",
        "some",
        "please",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass(slots=True)
class RetrievalResult:
    documents: list[ScoredDocument]
    filters: QueryFilters
    matching_listings: int  # how many listings satisfy the filters (before top-k cut)


class HybridRetriever:
    def __init__(
        self,
        documents: list[Document],
        embeddings: np.ndarray | None = None,
        embedder: QueryEmbedder | None = None,
    ) -> None:
        if not documents:
            raise ValueError("retriever needs at least one document")
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError(f"{len(embeddings)} embeddings for {len(documents)} documents")
        self.documents = documents
        self._bm25 = BM25Okapi([tokenize(d.embedding_text) for d in documents])
        self._emb = embeddings if embedder is not None else None
        self._embedder = embedder if embeddings is not None else None
        self._is_listing = np.array([d.kind == "listing" for d in documents])

    @property
    def dense_enabled(self) -> bool:
        return self._emb is not None

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_index_dir(cls, index_dir: Path, embedder: QueryEmbedder | None) -> HybridRetriever:
        docs_path = index_dir / "documents.jsonl"
        if not docs_path.exists():
            raise FileNotFoundError(f"{docs_path} not found — run `make index` first")
        with docs_path.open(encoding="utf-8") as f:
            documents = [Document.from_dict(json.loads(line)) for line in f if line.strip()]
        emb_path = index_dir / "embeddings.npy"
        embeddings = np.load(emb_path) if emb_path.exists() and embedder is not None else None
        if embedder is not None and embeddings is None:
            log.warning("embeddings.npy missing — falling back to BM25-only retrieval")
        return cls(documents, embeddings, embedder)

    # ------------------------------------------------------------------ search
    def search(self, query: str, filters: QueryFilters, k: int = 8) -> RetrievalResult:
        n = len(self.documents)
        allowed = np.ones(n, dtype=bool)
        matching = 0
        if filters.has_constraints:
            listing_ok = np.array(
                [d.kind != "listing" or filters.matches(d.meta) for d in self.documents]
            )
            allowed &= listing_ok
            matching = int((self._is_listing & listing_ok).sum())

        rankings: list[np.ndarray] = []
        tokens = tokenize(query)
        if tokens:
            bm25 = np.asarray(self._bm25.get_scores(tokens))
            bm25[~allowed] = -np.inf
            rankings.append(_top(bm25, CANDIDATES, positive_only=True))
        if self._emb is not None and self._embedder is not None:
            dense = self._emb @ self._embedder.embed_query(query)
            dense[~allowed] = -np.inf
            rankings.append(_top(dense, CANDIDATES))

        fused = _rrf(rankings, n)

        # With explicit listing filters, structured matches are the answer: make sure
        # they surface even when their text is a weak lexical match for the question.
        if filters.listing_intent and filters.has_constraints and matching:
            fused[self._is_listing & allowed] += 1.0 / RRF_K

        order = [int(i) for i in np.argsort(-fused) if fused[i] > 0 and allowed[i]]
        if filters.listing_intent and filters.has_constraints:
            listings = [i for i in order if self._is_listing[i]][: max(k - 2, 1)]
            context = [i for i in order if not self._is_listing[i]][: k - len(listings)]
            picked = sorted(listings + context, key=lambda i: -fused[i])
        else:
            picked = order[:k]

        return RetrievalResult(
            documents=[ScoredDocument(self.documents[i], float(fused[i])) for i in picked],
            filters=filters,
            matching_listings=matching,
        )


def _top(scores: np.ndarray, n: int, positive_only: bool = False) -> np.ndarray:
    valid = np.isfinite(scores) & ((scores > 0) if positive_only else True)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return idx
    best = idx[np.argsort(-scores[idx], kind="stable")[:n]]
    return best


def _rrf(rankings: list[np.ndarray], n: int) -> np.ndarray:
    fused = np.zeros(n, dtype=np.float64)
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] += 1.0 / (RRF_K + rank + 1)
    return fused
