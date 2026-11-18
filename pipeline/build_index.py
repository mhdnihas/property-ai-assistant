"""Embed processed documents and write the search index (data/index).

    python -m pipeline.build_index

Outputs
  documents.jsonl   the documents, in index order
  embeddings.npy    float32 [n_docs, dim], L2-normalised
  manifest.json     model name, counts, build time — served by /api/health
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.config import ROOT_DIR, get_settings
from app.rag.embeddings import Embedder
from app.rag.models import Document

log = logging.getLogger("pipeline.index")

DOCS_PATH = ROOT_DIR / "data" / "processed" / "documents.jsonl"


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=DOCS_PATH)
    parser.add_argument("--out", type=Path, default=settings.index_dir)
    parser.add_argument("--no-embeddings", action="store_true", help="BM25-only index")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with args.docs.open(encoding="utf-8") as f:
        docs = [Document.from_dict(json.loads(line)) for line in f if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.docs, args.out / "documents.jsonl")

    manifest: dict[str, object] = {
        "documents": len(docs),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "embedding_model": None,
    }
    if not args.no_embeddings:
        t0 = time.perf_counter()
        embedder = Embedder(settings.embedding_model, settings.embedding_cache_dir)
        vectors = embedder.embed_documents([d.embedding_text for d in docs])
        np.save(args.out / "embeddings.npy", vectors)
        manifest |= {"embedding_model": settings.embedding_model, "dim": int(vectors.shape[1])}
        log.info(
            "embedded %d docs (dim=%d) in %.1fs",
            len(docs),
            vectors.shape[1],
            time.perf_counter() - t0,
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("index written -> %s", args.out)


if __name__ == "__main__":
    main()
