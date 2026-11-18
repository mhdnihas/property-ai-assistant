import numpy as np
import pytest

from app.rag.models import Document
from app.rag.query_parser import parse_query
from app.rag.retriever import HybridRetriever, tokenize


def test_tokenize_drops_stopwords_keeps_numbers() -> None:
    assert tokenize("What is the price of 2.5 villas?") == ["price", "2.5", "villas"]


def test_project_question_finds_project(retriever: HybridRetriever) -> None:
    q = "Tell me about Trump Tower Jeddah"
    res = retriever.search(q, parse_query(q), k=3)
    assert res.documents[0].document.id == "p1"


def test_filters_exclude_non_matching_listings(retriever: HybridRetriever) -> None:
    q = "villas for sale in Riyadh under 2 million"
    res = retriever.search(q, parse_query(q), k=5)
    listing_ids = [d.document.id for d in res.documents if d.document.kind == "listing"]
    assert listing_ids == ["l1"]
    assert res.matching_listings == 1


def test_no_match_returns_zero_matching(retriever: HybridRetriever) -> None:
    q = "villas for rent in Madinah"
    res = retriever.search(q, parse_query(q), k=5)
    assert res.matching_listings == 0
    assert all(d.document.kind != "listing" for d in res.documents)


def test_dense_scores_are_fused(documents: list[Document]) -> None:
    class StubEmbedder:
        def embed_query(self, text: str) -> np.ndarray:
            return np.eye(len(documents), dtype=np.float32)[1]  # always "closest" to doc index 1

    emb = np.eye(len(documents), dtype=np.float32)
    r = HybridRetriever(documents, emb, StubEmbedder())
    assert r.dense_enabled
    res = r.search("zzz unknown words", parse_query("zzz unknown words"), k=1)
    assert res.documents[0].document.id == "p2"


def test_rejects_mismatched_embeddings(documents: list[Document]) -> None:
    with pytest.raises(ValueError, match="embeddings"):
        HybridRetriever(documents, np.zeros((2, 4), dtype=np.float32), object())  # type: ignore[arg-type]
