"""Embed processed documents and write the search index (data/index).

    python -m pipeline.build_index

Outputs
  documents.jsonl   the documents, in index order
  embeddings.npy    float32 [n_docs, dim], L2-normalised
  manifest.json     model name, counts, build time — served by /api/health

The index is built in a temporary directory and swapped into place only when
complete, so an interrupted build never leaves documents and embeddings that
disagree.

Embeddings of documents whose id and text are unchanged since the previous
index (built with the same model) are reused, so adding a few documents takes
seconds instead of re-embedding everything. Use --full to force a rebuild.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from collections.abc import Callable
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


def load_previous_vectors(index_dir: Path, model: str) -> dict[tuple[str, str], np.ndarray]:
    """Map (doc id, embedding text) -> vector from an existing, consistent index."""
    try:
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != model:
            return {}
        docs = load_documents(index_dir / "documents.jsonl")
        vectors = np.load(index_dir / "embeddings.npy")
    except (OSError, ValueError):
        return {}
    if len(docs) != len(vectors):
        log.warning("previous index is inconsistent; not reusing its embeddings")
        return {}
    return {(d.id, d.embedding_text): v for d, v in zip(docs, vectors, strict=True)}


def embed_with_reuse(
    docs: list[Document],
    previous: dict[tuple[str, str], np.ndarray],
    embed: Callable[[list[str]], np.ndarray],
) -> tuple[np.ndarray, int]:
    """Return vectors for `docs`, embedding only documents not found in `previous`.

    Returns (vectors, number of newly embedded documents).
    """
    if not docs:
        raise ValueError("no documents to index")
    keys = [(d.id, d.embedding_text) for d in docs]
    missing = [i for i, k in enumerate(keys) if k not in previous]
    fresh = embed([keys[i][1] for i in missing]) if missing else None

    dim = fresh.shape[1] if fresh is not None else len(next(iter(previous.values())))
    vectors = np.empty((len(docs), dim), dtype=np.float32)
    fresh_rows = dict(zip(missing, fresh if fresh is not None else [], strict=True))
    for i, key in enumerate(keys):
        vectors[i] = fresh_rows[i] if i in fresh_rows else previous[key]
    return vectors, len(missing)


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
    parser.add_argument("--full", action="store_true", help="re-embed every document")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    docs = load_documents(args.docs)
    vectors = None
    if not args.no_embeddings:
        from app.rag.embeddings import Embedder

        previous = {} if args.full else load_previous_vectors(args.out, settings.embedding_model)
        embedder: Embedder | None = None

        def embed(texts: list[str]) -> np.ndarray:
            nonlocal embedder
            embedder = embedder or Embedder(settings.embedding_model, settings.embedding_cache_dir)
            return embedder.embed_documents(texts)

        t0 = time.perf_counter()
        vectors, n_new = embed_with_reuse(docs, previous, embed)
        log.info(
            "embedded %d new/changed docs, reused %d, in %.1fs",
            n_new,
            len(docs) - n_new,
            time.perf_counter() - t0,
        )

    write_index(args.out, docs, vectors, settings.embedding_model)
    log.info("index written -> %s (%d documents)", args.out, len(docs))


if __name__ == "__main__":
    main()
