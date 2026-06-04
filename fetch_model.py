"""Fetch the embedding model with line-buffered progress, so flaky-wifi
downloads are visible and resumable. Run before indexer.py when network
is shaky.

Usage:
    python fetch_model.py
    python fetch_model.py --model sentence-transformers/all-mpnet-base-v2
    python fetch_model.py --no-xet         # disable hf-xet, plain HTTP
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# sentence-transformers only needs PyTorch weights + tokenizer/config. The HF
# repo also ships TF, Flax, ONNX, OpenVINO, and Rust variants — skipping them
# cuts the download from ~930 MB to ~90 MB.
IGNORE_PATTERNS = [
    "*.h5",
    "*.msgpack",
    "*.ot",
    "rust_model*",
    "onnx/*",
    "openvino/*",
    "*.onnx",
]


def progress_loop(cache_dir: Path, total_bytes: int, stop: threading.Event) -> None:
    """Emit a progress line every ~2s while the snapshot_download runs."""
    bar_width = 30
    last_reported = -1
    while not stop.is_set():
        bytes_now = 0
        try:
            for f in cache_dir.rglob("*"):
                if f.is_file():
                    try:
                        bytes_now += f.stat().st_size
                    except OSError:
                        pass
        except FileNotFoundError:
            pass

        if total_bytes > 0 and bytes_now != last_reported:
            mb_now = bytes_now / (1024 * 1024)
            mb_tot = total_bytes / (1024 * 1024)
            pct = min(100.0, bytes_now / total_bytes * 100)
            filled = int(bar_width * min(bytes_now, total_bytes) / total_bytes)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"[{bar}] {mb_now:6.1f} / {mb_tot:6.1f} MB ({pct:5.1f}%)",
                flush=True,
            )
            last_reported = bytes_now

        stop.wait(2.0)


def _matches_ignore(path: str, patterns: list[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def resolve_total_bytes(model: str, ignore_patterns: list[str]) -> int:
    """Sum file sizes for the repo's main revision, skipping ignored files."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        tree = api.list_repo_tree(repo_id=model, recursive=True)
        total = 0
        for item in tree:
            size = getattr(item, "size", None)
            path = getattr(item, "path", "")
            if size and not _matches_ignore(path, ignore_patterns):
                total += size
        return total
    except Exception as e:
        print(f"  (couldn't fetch total size: {e})", flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-attempts", type=int, default=10)
    ap.add_argument(
        "--no-xet",
        action="store_true",
        help="disable hf-xet and use the plain HTTP downloader",
    )
    args = ap.parse_args()

    if args.no_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

    print(f"fetching model: {args.model}", flush=True)
    print(f"  hf-xet: {'disabled' if args.no_xet else 'enabled'}", flush=True)

    total_bytes = resolve_total_bytes(args.model, IGNORE_PATTERNS)
    if total_bytes:
        print(f"  total size: {total_bytes / (1024*1024):.1f} MB", flush=True)
    else:
        print("  total size: unknown (progress %% will be unavailable)", flush=True)

    cache_subdir = Path(HF_HUB_CACHE) / f"models--{args.model.replace('/', '--')}"

    for attempt in range(1, args.max_attempts + 1):
        stop = threading.Event()
        thread = threading.Thread(
            target=progress_loop, args=(cache_subdir, total_bytes, stop), daemon=True
        )
        thread.start()

        try:
            t0 = time.time()
            path = snapshot_download(
                repo_id=args.model, ignore_patterns=IGNORE_PATTERNS
            )
            stop.set()
            thread.join(timeout=3)
            # snapshot_download can return the cache dir even when it couldn't
            # reach HF — verify a weight file actually landed.
            weights_present = any(
                (Path(path) / name).exists()
                for name in ("model.safetensors", "pytorch_model.bin")
            )
            if not weights_present:
                raise OSError(
                    "snapshot_download returned but no weight file "
                    "(model.safetensors / pytorch_model.bin) is present in "
                    f"{path}"
                )
            dt = time.time() - t0
            print(f"DONE in {dt:.1f}s -> {path}", flush=True)
            return 0
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError, ConnectionError) as e:
            stop.set()
            thread.join(timeout=3)
            print(
                f"attempt {attempt}/{args.max_attempts} failed: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            if attempt == args.max_attempts:
                print("giving up.", file=sys.stderr, flush=True)
                return 1
            backoff = min(2 ** attempt, 60)
            print(f"  sleeping {backoff}s then retrying (resumes from cache)...", flush=True)
            time.sleep(backoff)

    return 1


if __name__ == "__main__":
    sys.exit(main())
