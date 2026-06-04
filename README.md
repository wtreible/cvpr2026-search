# CVPR 2026 Paper Browser

Scrape, embed, and semantically search the CVPR 2026 open-access proceedings.
A small Flask UI on top renders the results with thumbnails.

```
scraper.py     →  data/papers.db + data/thumbnails/
fetch_model.py →  ~/.cache/huggingface/...        (one-time, optional)
indexer.py     →  data/embeddings.npy + ids + meta
search.py      →  CLI search
app.py         →  Web UI at http://127.0.0.1:5001
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Activate the venv (`source .venv/bin/activate`) or prefix every command with
`.venv/bin/python` — examples below assume the latter.

## 1. Scrape the papers

Pulls metadata from `openaccess.thecvf.com`, matches against the virtual-site
poster pages, and downloads thumbnails. Run once; safe to re-run incrementally.

```bash
.venv/bin/python scraper.py                # metadata + thumbnails (default)
.venv/bin/python scraper.py --no-thumbnails
.venv/bin/python scraper.py --pdfs         # also pull all PDFs (tens of GB)
.venv/bin/python scraper.py --skip-scrape  # only run the download steps
.venv/bin/python scraper.py --workers 20   # tune concurrency
```

Output: `data/papers.db` (SQLite) and `data/thumbnails/<virtual_id>.png`.

## 2. (Optional) Pre-fetch the embedding model

The indexer will download `sentence-transformers/all-MiniLM-L6-v2` from
HuggingFace on first run. On a flaky connection that download often stalls
silently inside `sentence-transformers`. `fetch_model.py` wraps
`snapshot_download` with a visible progress bar, framework filtering
(~90 MB instead of ~930 MB), and an outer retry loop that resumes from the
HF cache after every failure.

```bash
.venv/bin/python fetch_model.py            # default model, hf-xet enabled
.venv/bin/python fetch_model.py --no-xet   # plain HTTP — more resilient
.venv/bin/python fetch_model.py --model sentence-transformers/all-mpnet-base-v2
```

Skip this if your connection is solid — `indexer.py` will download the model
itself.

## 3. Build the embeddings

Embeds `"<title>. <abstract>"` for every paper with the model and writes a
single (N, D) float32 array. Re-running is a no-op when the model and paper
set haven't changed.

```bash
.venv/bin/python indexer.py                              # default: all-MiniLM-L6-v2
.venv/bin/python indexer.py --model sentence-transformers/all-mpnet-base-v2
.venv/bin/python indexer.py --batch-size 128
.venv/bin/python indexer.py --rebuild                    # force re-embed everything
```

Output: `data/embeddings.npy`, `data/embedding_ids.npy`, `data/embedding_meta.txt`.

## 4. Search

### CLI

```bash
.venv/bin/python search.py "3D gaussian splatting for dynamic scenes"
.venv/bin/python search.py -k 20 "diffusion model for video editing"
.venv/bin/python search.py --json "open vocabulary detection"
.venv/bin/python search.py --snippet 500 "neural radiance fields"
```

### Web UI

Default port is 5001 (avoiding macOS's AirPlay Receiver on 5000).

```bash
.venv/bin/python app.py                    # http://127.0.0.1:5001
.venv/bin/python app.py --port 8080
.venv/bin/python app.py --host 0.0.0.0     # expose on LAN
.venv/bin/python app.py --debug
```

The server warms the embeddings and model at startup, so the first request
is fast.

## Layout

```
app.py            Flask UI + /api/search + /thumbnails/<id>.png
search.py         Searcher class (cosine over normalized embeddings) + CLI
indexer.py        Build embeddings.npy
fetch_model.py    Resilient HuggingFace model pre-fetcher
scraper.py        CVPR open-access + virtual-site scraper
templates/        index.html
data/             papers.db, thumbnails/, embeddings.npy, ...
requirements.txt
```
