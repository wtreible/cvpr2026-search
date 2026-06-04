"""Semantic search over the embedded CVPR 2026 papers.

Usage:
    python search.py "3D gaussian splatting"
    python search.py -k 20 "diffusion model for video editing"
    python search.py --json "open vocabulary detection"
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "papers.db"
EMB_PATH = DATA_DIR / "embeddings.npy"
IDS_PATH = DATA_DIR / "embedding_ids.npy"
META_PATH = DATA_DIR / "embedding_meta.txt"


class Searcher:
    def __init__(self):
        if not EMB_PATH.exists():
            raise SystemExit(
                f"Embeddings not found at {EMB_PATH}. Run indexer.py first."
            )
        self.embeddings = np.load(EMB_PATH)  # (N, D), already L2-normalized
        self.paper_ids = np.load(IDS_PATH)
        self.model_name = META_PATH.read_text().strip()
        # check_same_thread=False is safe here: the Searcher only reads, and
        # SQLite handles concurrent readers fine. This lets the Flask dev
        # server (which dispatches each request in a new thread) reuse the
        # single connection opened at startup.
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def search(self, query: str, k: int = 10) -> list[dict]:
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)
        scores = self.embeddings @ q  # cosine sim (both normalized)
        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        ids = [int(self.paper_ids[i]) for i in top_idx]
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT id, title, authors, abstract, pdf_url, oa_html_url, "
            f"virtual_id, thumbnail_path FROM papers WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        by_id = {r["id"]: r for r in rows}

        out = []
        for i, pid in zip(top_idx, ids):
            r = by_id.get(pid)
            if r is None:
                continue
            out.append(
                {
                    "score": float(scores[i]),
                    "id": pid,
                    "title": r["title"],
                    "authors": r["authors"],
                    "abstract": r["abstract"],
                    "pdf_url": r["pdf_url"],
                    "oa_html_url": r["oa_html_url"],
                    "virtual_id": r["virtual_id"],
                    "thumbnail_path": r["thumbnail_path"],
                }
            )
        return out


def format_result(r: dict, snippet_chars: int = 280) -> str:
    abstract = (r["abstract"] or "").strip().replace("\n", " ")
    if len(abstract) > snippet_chars:
        abstract = abstract[:snippet_chars].rstrip() + "..."
    wrapped = textwrap.fill(abstract, width=88, initial_indent="    ", subsequent_indent="    ")
    parts = [
        f"[{r['score']:.3f}] {r['title']}",
        f"    {r['authors']}" if r["authors"] else "",
        f"    pdf: {r['pdf_url']}" if r["pdf_url"] else "",
        wrapped,
    ]
    return "\n".join(p for p in parts if p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", nargs="+", help="search query")
    ap.add_argument("-k", "--top-k", type=int, default=10)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--snippet", type=int, default=280, help="abstract chars to print")
    args = ap.parse_args()

    query = " ".join(args.query)
    s = Searcher()
    results = s.search(query, k=args.top_k)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"\nQuery: {query!r}   (top {len(results)})\n")
    for r in results:
        print(format_result(r, snippet_chars=args.snippet))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
