"""Flask web UI for browsing/searching CVPR 2026 papers.

Usage:
    python app.py                # http://127.0.0.1:5001
    python app.py --port 8080
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from search import Searcher

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
THUMB_DIR = DATA_DIR / "thumbnails"

app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
_searcher: Searcher | None = None


def get_searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return _searcher


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 20))
    if not q:
        return jsonify({"results": [], "query": ""})
    results = get_searcher().search(q, k=k)
    return jsonify({"results": results, "query": q})


@app.route("/thumbnails/<path:name>")
def thumbnail(name: str):
    return send_from_directory(THUMB_DIR, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # warm the searcher so first request isn't slow
    print("Loading embeddings & model...")
    get_searcher()
    print(f"Ready. Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    main()
