import json
from pathlib import Path

import numpy as np
import pytest

from app.rag.models import Document
from app.rag.retriever import HybridRetriever
from pipeline.build_index import write_index


def _docs(n: int) -> list[Document]:
    return [
        Document(id=f"d{i}", source="darglobal", kind="page", title=f"Doc {i}", url="u", text="t")
        for i in range(n)
    ]


def test_write_index_creates_consistent_files(tmp_path: Path) -> None:
    out = tmp_path / "index"
    write_index(out, _docs(3), np.ones((3, 4), dtype=np.float32), "model-x")

    assert sorted(p.name for p in out.iterdir()) == [
        "documents.jsonl",
        "embeddings.npy",
        "manifest.json",
    ]
    assert len((out / "documents.jsonl").read_text().splitlines()) == 3
    assert np.load(out / "embeddings.npy").shape == (3, 4)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["documents"] == 3
    assert manifest["embedding_model"] == "model-x"
    assert manifest["dim"] == 4
    assert [p.name for p in tmp_path.iterdir()] == ["index"]  # no temp dirs left behind


def test_write_index_without_embeddings(tmp_path: Path) -> None:
    write_index(tmp_path / "index", _docs(2), None, "model-x")
    assert not (tmp_path / "index" / "embeddings.npy").exists()
    assert json.loads((tmp_path / "index" / "manifest.json").read_text())["embedding_model"] is None


def test_failed_write_keeps_previous_index(tmp_path: Path) -> None:
    out = tmp_path / "index"
    write_index(out, _docs(2), np.ones((2, 4), dtype=np.float32), "m")
    before = (out / "documents.jsonl").read_text()

    with pytest.raises(ValueError, match="vectors"):
        write_index(out, _docs(5), np.ones((4, 4), dtype=np.float32), "m")

    assert (out / "documents.jsonl").read_text() == before
    assert np.load(out / "embeddings.npy").shape == (2, 4)


def test_retriever_falls_back_to_bm25_on_mismatched_index(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "index"
    write_index(out, _docs(3), np.ones((3, 4), dtype=np.float32), "m")
    np.save(out / "embeddings.npy", np.ones((2, 4), dtype=np.float32))  # simulate a stale file

    class StubEmbedder:
        def embed_query(self, text: str) -> np.ndarray:
            return np.ones(4, dtype=np.float32)

    with caplog.at_level("ERROR"):
        retriever = HybridRetriever.from_index_dir(out, StubEmbedder())
    assert len(retriever.documents) == 3
    assert not retriever.dense_enabled
    assert "index mismatch" in caplog.text
