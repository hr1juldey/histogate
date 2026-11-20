#!/usr/bin/env python3
"""Parallel-range downloader for Zenodo files (aria2c unavailable).

Splits the file into N ranges fetched concurrently into part files, then
concatenates. Resumable: existing complete parts are kept.
Usage: parallel_fetch.py URL OUT [N_PARTS]
"""
import concurrent.futures as cf
import os
import sys
import urllib.request

URL, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 16


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(i: int, start: int, end: int) -> str:
    part = f"{OUT}.part{i:03d}"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    want = end - start + 1
    if have == want:
        print(f"part {i} complete ({have}B)", flush=True)
        return part
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start + have}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    got = os.path.getsize(part)
    assert got == want, f"part {i}: {got} != {want}"
    print(f"part {i} done ({want}B)", flush=True)
    return part


size = head_size(URL)
step = (size + N - 1) // N
ranges = [(i, i * step, min((i + 1) * step - 1, size - 1)) for i in range(N)]
ranges = [(i, s, e) for i, s, e in ranges if s <= e]
print(f"{size} bytes in {len(ranges)} parts", flush=True)

with cf.ThreadPoolExecutor(max_workers=len(ranges)) as ex:
    parts = list(ex.map(lambda a: fetch(*a), ranges))

with open(OUT, "wb") as out:
    for p in parts:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 22)
                if not chunk:
                    break
                out.write(chunk)
for p in parts:
    os.remove(p)
print("ASSEMBLED", OUT, os.path.getsize(OUT), flush=True)
