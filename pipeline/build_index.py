"""Embed processed documents and write the search index (data/index).

    python -m pipeline.build_index

Outputs
  documents.jsonl   the documents, in index order
  embeddings.npy    float32 [n_docs, dim], L2-normalised
  manifest.json     model name, counts, build time — served by /api/health

The index is built in a temporary directory and swapped into place only when
complete, so an interrupted build never leaves documents and embeddings that
disagree.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.config import ROOT_DIR, get_settings
from app.rag.models import Document

log = logging.getLogger("pipeline.index")

DOCS_PATH = ROOT_DIR / "data" / "processed" / "documents.jsonl"


def load_documents(path: Path) -> list[Document]:
    with path.open(encoding="utf-8") as f:
        return [Document.from_dict(json.loads(line)) for line in f if line.strip()]


def write_index(
    out: Path,
    docs: list[Document],
    vectors: np.ndarray | None,
    embedding_model: str | None,
) -> None:
    """Atomically replace the index directory `out` with a new one."""
    if vectors is not None and len(vectors) != len(docs):
        raise ValueError(f"{len(vectors)} vectors for {len(docs)} documents")

    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.tmp-{os.getpid()}"
    old = out.parent / f".{out.name}.old-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        with (tmp / "documents.jsonl").open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
        manifest: dict[str, object] = {
            "documents": len(docs),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "embedding_model": embedding_model if vectors is not None else None,
        }
        if vectors is not None:
            np.save(tmp / "embeddings.npy", vectors.astype(np.float32))
            manifest["dim"] = int(vectors.shape[1])
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Swap directories: each rename is atomic, so readers see either the old
        # complete index or the new complete index.
        if out.exists():
            out.rename(old)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if old.exists() and not out.exists():
            old.rename(out)  # restore the previous index
        raise
    shutil.rmtree(old, ignore_errors=True)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DOCS_PATH)
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--no-embeddings", action="store_true", help="BM25-only index")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    docs = load_documents(args.docs)
    vectors = None
    if not args.no_embeddings:
        from app.rag.embeddings import Embedder

        t0 = time.perf_counter()
        embedder = Embedder(settings.embedding_model, settings.embedding_cache_dir)
        vectors = embedder.embed_documents([d.embedding_text for d in docs])
        log.info("embedded %d docs in %.1fs", len(docs), time.perf_counter() - t0)

    write_index(args.out, docs, vectors, settings.embedding_model)
    log.info("index written -> %s (%d documents)", args.out, len(docs))


if __name__ == "__main__":
    main()
