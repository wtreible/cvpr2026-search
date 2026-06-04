"""Build sentence-transformer embeddings over all scraped papers.

Embeds "<title>. <abstract>" per paper with all-MiniLM-L6-v2 (~80MB model,
384-dim, ~5-10 min on CPU for 4k papers, seconds on GPU/MPS).

Usage:
    python indexer.py                       # build/update
    python indexer.py --model all-mpnet-base-v2  # use larger model
    python indexer.py --rebuild             # force re-embed everything
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "papers.db"
EMB_PATH = DATA_DIR / "embeddings.npy"
IDS_PATH = DATA_DIR / "embedding_ids.npy"
META_PATH = DATA_DIR / "embedding_meta.txt"


def build_text(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    if not abstract:
        return title
    return f"{title}. {abstract}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}. Run scraper.py first.", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title, abstract FROM papers ORDER BY id"
    ).fetchall()
    if not rows:
        print("No papers in DB.", file=sys.stderr)
        return 1

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [build_text(r[1], r[2]) for r in rows]

    existing_model = None
    if META_PATH.exists():
        existing_model = META_PATH.read_text().strip()

    if (
        not args.rebuild
        and EMB_PATH.exists()
        and IDS_PATH.exists()
        and existing_model == args.model
    ):
        old_ids = np.load(IDS_PATH)
        if np.array_equal(old_ids, ids):
            print(f"Embeddings up to date ({len(ids)} papers, model={args.model}).")
            return 0
        print("DB changed — re-embedding.")

    print(f"Loading model: {args.model}")
    from sentence_transformers import SentenceTransformer  # lazy import

    model = SentenceTransformer(args.model)
    print(f"Embedding {len(texts)} papers (batch={args.batch_size})...")
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s, shape={emb.shape}")

    np.save(EMB_PATH, emb)
    np.save(IDS_PATH, ids)
    META_PATH.write_text(args.model)
    print(f"Wrote {EMB_PATH} and {IDS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
