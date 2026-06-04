"""Scrape CVPR 2026 papers from openaccess.thecvf.com and cvpr.thecvf.com/virtual.

Stores metadata (title, authors, abstract, pdf url, virtual poster id, thumbnail
path) in data/papers.db. Optionally downloads thumbnails and PDFs.

Usage:
    python scraper.py                  # scrape metadata + thumbnails
    python scraper.py --pdfs           # also download PDFs (~10s of GB)
    python scraper.py --no-thumbnails  # skip thumbnails
    python scraper.py --workers 20     # tune concurrency
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

OA_BASE = "https://openaccess.thecvf.com"
OA_ALL = f"{OA_BASE}/CVPR2026?day=all"
VIRTUAL_PAPERS = "https://cvpr.thecvf.com/virtual/2026/papers.html"
VIRTUAL_BASE = "https://cvpr.thecvf.com"

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "papers.db"
THUMB_DIR = DATA_DIR / "thumbnails"
PDF_DIR = DATA_DIR / "pdfs"

USER_AGENT = "CVPR2026-Browser/1.0 (personal research tool)"


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def init_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            oa_slug       TEXT UNIQUE,        -- filename stem from openaccess
            title         TEXT NOT NULL,
            title_norm    TEXT NOT NULL,
            authors       TEXT,               -- comma-separated
            abstract      TEXT,
            pdf_url       TEXT,
            supp_url      TEXT,
            oa_html_url   TEXT,
            virtual_id    TEXT,               -- /virtual/2026/poster/<id>
            thumbnail_url TEXT,
            thumbnail_path TEXT,
            pdf_path      TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_title_norm ON papers(title_norm)")
    conn.commit()
    return conn


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation/whitespace for matching across sources."""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


# ---------- openaccess ----------


def parse_oa_listing(html: str) -> list[dict]:
    """Extract (oa_html_url, title) for each paper on the all-papers page."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for dt in soup.select("dt.ptitle"):
        a = dt.find("a")
        if not a or not a.get("href"):
            continue
        out.append(
            {
                "oa_html_url": urljoin(OA_BASE, a["href"]),
                "title": a.get_text(strip=True),
            }
        )
    return out


def parse_oa_paper(html: str) -> dict:
    """Pull title, authors, abstract, pdf_url, supp_url from a paper detail page."""
    soup = BeautifulSoup(html, "lxml")

    title_el = soup.find(id="papertitle")
    title = title_el.get_text(strip=True) if title_el else ""

    authors = ""
    authors_el = soup.find(id="authors")
    if authors_el:
        b = authors_el.find("b")
        if b:
            i = b.find("i")
            if i:
                authors = i.get_text(strip=True)

    abstract_el = soup.find(id="abstract")
    abstract = abstract_el.get_text(strip=True) if abstract_el else ""

    pdf_url = supp_url = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if text == "pdf" and href.endswith(".pdf"):
            pdf_url = urljoin(OA_BASE, href)
        elif text == "supp" and href.endswith(".pdf"):
            supp_url = urljoin(OA_BASE, href)

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "pdf_url": pdf_url,
        "supp_url": supp_url,
    }


def fetch_oa_paper(sess: requests.Session, listing_entry: dict) -> dict | None:
    url = listing_entry["oa_html_url"]
    slug = url.rsplit("/", 1)[-1].removesuffix(".html")
    try:
        r = sess.get(url, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"  ! fetch {url}: {e}", file=sys.stderr)
        return None
    data = parse_oa_paper(r.text)
    data["oa_slug"] = slug
    data["oa_html_url"] = url
    if not data["title"]:
        data["title"] = listing_entry["title"]
    data["title_norm"] = normalize_title(data["title"])
    return data


# ---------- virtual (thumbnails) ----------


def parse_virtual_listing(html: str) -> list[dict]:
    """Extract (virtual_id, title) per poster link. Deduplicates by virtual_id."""
    soup = BeautifulSoup(html, "lxml")
    seen = {}
    for a in soup.find_all("a", href=True):
        m = re.match(r"^/virtual/2026/poster/(\d+)$", a["href"])
        if not m:
            continue
        vid = m.group(1)
        title = a.get_text(strip=True)
        if not title or vid in seen:
            continue
        seen[vid] = title
    return [{"virtual_id": vid, "title": t} for vid, t in seen.items()]


def match_virtual_to_oa(conn: sqlite3.Connection, virtual_entries: list[dict]) -> int:
    """Update papers table with virtual_id + thumbnail_url where title matches."""
    cur = conn.cursor()
    matched = 0
    for v in virtual_entries:
        tn = normalize_title(v["title"])
        thumb_url = (
            f"{VIRTUAL_BASE}/media/PosterPDFs/CVPR%202026/{v['virtual_id']}-thumb.png"
        )
        cur.execute(
            "UPDATE papers SET virtual_id = ?, thumbnail_url = ? "
            "WHERE title_norm = ? AND virtual_id IS NULL",
            (v["virtual_id"], thumb_url, tn),
        )
        if cur.rowcount > 0:
            matched += 1
    conn.commit()
    return matched


# ---------- downloads ----------


def download_file(sess: requests.Session, url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sess.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 15):
                    if chunk:
                        f.write(chunk)
            tmp.rename(dest)
        return True
    except Exception as e:
        print(f"  ! download {url}: {e}", file=sys.stderr)
        return False


def download_thumbnails(conn: sqlite3.Connection, workers: int) -> None:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, virtual_id, thumbnail_url FROM papers "
        "WHERE thumbnail_url IS NOT NULL AND thumbnail_path IS NULL"
    ).fetchall()
    if not rows:
        print("No new thumbnails to fetch.")
        return

    sess = session()

    def job(row):
        pid, vid, url = row
        dest = THUMB_DIR / f"{vid}.png"
        ok = download_file(sess, url, dest)
        return pid, dest if ok else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, r) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs), desc="thumbs"):
            pid, dest = f.result()
            if dest is not None:
                conn.execute(
                    "UPDATE papers SET thumbnail_path = ? WHERE id = ?",
                    (str(dest.relative_to(ROOT)), pid),
                )
    conn.commit()


def download_pdfs(conn: sqlite3.Connection, workers: int) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, oa_slug, pdf_url FROM papers "
        "WHERE pdf_url IS NOT NULL AND pdf_path IS NULL"
    ).fetchall()
    if not rows:
        print("No new PDFs to fetch.")
        return

    sess = session()

    def job(row):
        pid, slug, url = row
        dest = PDF_DIR / f"{slug}.pdf"
        ok = download_file(sess, url, dest)
        return pid, dest if ok else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, r) for r in rows]
        for f in tqdm(as_completed(futs), total=len(futs), desc="pdfs"):
            pid, dest = f.result()
            if dest is not None:
                conn.execute(
                    "UPDATE papers SET pdf_path = ? WHERE id = ?",
                    (str(dest.relative_to(ROOT)), pid),
                )
    conn.commit()


# ---------- main scrape ----------


def scrape_metadata(conn: sqlite3.Connection, workers: int) -> None:
    sess = session()

    print(f"Fetching openaccess listing: {OA_ALL}")
    r = sess.get(OA_ALL, timeout=60)
    r.raise_for_status()
    listing = parse_oa_listing(r.text)
    print(f"  found {len(listing)} paper entries")

    cur = conn.cursor()
    existing = {row[0] for row in cur.execute("SELECT oa_slug FROM papers").fetchall()}
    todo = [
        e
        for e in listing
        if e["oa_html_url"].rsplit("/", 1)[-1].removesuffix(".html") not in existing
    ]
    print(f"  {len(todo)} new (rest already in DB)")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_oa_paper, sess, e) for e in todo]
        for f in tqdm(as_completed(futs), total=len(futs), desc="papers"):
            data = f.result()
            if data is None:
                continue
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO papers
                    (oa_slug, title, title_norm, authors, abstract, pdf_url, supp_url, oa_html_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data["oa_slug"],
                        data["title"],
                        data["title_norm"],
                        data["authors"],
                        data["abstract"],
                        data["pdf_url"],
                        data["supp_url"],
                        data["oa_html_url"],
                    ),
                )
            except Exception as e:
                print(f"  ! insert {data.get('oa_slug')}: {e}", file=sys.stderr)
    conn.commit()


def scrape_virtual(conn: sqlite3.Connection) -> None:
    sess = session()
    print(f"Fetching virtual listing: {VIRTUAL_PAPERS}")
    r = sess.get(VIRTUAL_PAPERS, timeout=60)
    r.raise_for_status()
    entries = parse_virtual_listing(r.text)
    print(f"  found {len(entries)} virtual poster entries")
    matched = match_virtual_to_oa(conn, entries)
    print(f"  matched {matched} to openaccess records")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=10, help="concurrent HTTP workers")
    ap.add_argument("--no-thumbnails", action="store_true", help="skip thumbnail download")
    ap.add_argument("--pdfs", action="store_true", help="download paper PDFs (large)")
    ap.add_argument("--skip-scrape", action="store_true", help="only run download steps")
    args = ap.parse_args()

    conn = init_db()

    if not args.skip_scrape:
        scrape_metadata(conn, args.workers)
        scrape_virtual(conn)

    if not args.no_thumbnails:
        download_thumbnails(conn, args.workers)

    if args.pdfs:
        download_pdfs(conn, args.workers)

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    with_thumb = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE thumbnail_path IS NOT NULL"
    ).fetchone()[0]
    with_pdf = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE pdf_path IS NOT NULL"
    ).fetchone()[0]
    print(f"\nDB: {total} papers — {with_thumb} with thumbnail, {with_pdf} with pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
