"""
Parallel, resumable downloader for the two datasets.

Many servers limit the speed of each single connection. This script splits the file into
16 MB chunks and downloads 8 of them at once, so it is usually several times faster than
wget. If it is interrupted (Ctrl-C, SSH drop), run the same command again: finished
chunks are kept and only the missing ones are downloaded.

Usage
  python fast_download.py cub              # CUB-200-2011 only (enough for quick_evidence.py)
  python fast_download.py aircraft
  python fast_download.py cub aircraft --connections 12
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASETS = {
    "cub": ("CUB_200_2011.tgz",
            "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"),
    "aircraft": ("fgvc-aircraft-2013b.tar.gz",
                 "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"),
}
HEADERS = {"User-Agent": "Mozilla/5.0"}
CHUNK = 16 * 2**20


def probe(url):
    """Follow redirects, return (final_url, size, supports_ranges)."""
    req = urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        final = r.geturl()
        if r.status == 206:
            size = int(r.headers["Content-Range"].split("/")[-1])
            return final, size, True
        return final, int(r.headers.get("Content-Length", 0)), False


def fetch_range(url, path, start, end, progress, lock, retries=5):
    for attempt in range(retries):
        got = 0
        try:
            req = urllib.request.Request(url, headers={**HEADERS, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "r+b") as f:
                f.seek(start)
                while buf := r.read(1 << 18):
                    f.write(buf)
                    got += len(buf)
                    with lock:
                        progress[0] += len(buf)
            if got != end - start + 1:
                raise IOError(f"short read {got} / {end - start + 1}")
            return start
        except Exception as e:
            with lock:
                progress[0] -= got          # undo partial progress; chunk is retried
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def download(name, url, out_dir, connections):
    path = os.path.join(out_dir, name)
    state_path = path + ".chunks.json"
    if os.path.exists(path) and not os.path.exists(state_path):
        print(f"{name}: already complete ({os.path.getsize(path) / 2**20:,.0f} MB)")
        return
    final_url, size, ranges = probe(url)
    print(f"{name}: {size / 2**20:,.0f} MB, parallel={'yes' if ranges else 'NO (single stream)'}")

    if not ranges or size == 0:                       # fall back to one stream
        req = urllib.request.Request(final_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
            done = 0
            while buf := r.read(1 << 20):
                f.write(buf)
                done += len(buf)
                sys.stdout.write(f"\r  {done / 2**20:,.0f} MB")
        print("\n  done")
        return

    chunks = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    done = set()
    if os.path.exists(state_path) and os.path.exists(path) and os.path.getsize(path) == size:
        done = set(json.load(open(state_path))["done"])
        print(f"  resuming: {len(done)}/{len(chunks)} chunks already downloaded")
    else:
        with open(path, "wb") as f:
            f.truncate(size)
    todo = [c for c in chunks if c[0] not in done]

    progress, lock = [sum(e - s + 1 for s, e in chunks if s in done)], threading.Lock()
    t0, start_bytes = time.time(), progress[0]
    stop = threading.Event()

    def reporter():
        while not stop.wait(1.0):
            with lock:
                p = progress[0]
            rate = (p - start_bytes) / max(1e-6, time.time() - t0)
            eta = (size - p) / rate if rate > 0 else 0
            sys.stdout.write(f"\r  {p / 2**20:,.0f} / {size / 2**20:,.0f} MB  "
                             f"{rate / 2**20:5.1f} MB/s  ETA {eta / 60:4.1f} min   ")
            sys.stdout.flush()

    threading.Thread(target=reporter, daemon=True).start()
    try:
        with ThreadPoolExecutor(connections) as pool:
            futs = [pool.submit(fetch_range, final_url, path, s, e, progress, lock) for s, e in todo]
            for fut in as_completed(futs):
                done.add(fut.result())
                with open(state_path, "w") as f:
                    json.dump({"done": sorted(done)}, f)
    finally:
        stop.set()
    os.remove(state_path)
    print(f"\n  done in {(time.time() - t0) / 60:.1f} min -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+", choices=list(DATASETS))
    ap.add_argument("--out", default=".")
    ap.add_argument("--connections", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for d in args.datasets:
        name, url = DATASETS[d]
        download(name, url, args.out, args.connections)


if __name__ == "__main__":
    main()
