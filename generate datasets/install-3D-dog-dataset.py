#!/usr/bin/env python3
"""
download_mvsydog_3d.py

Downloads and extracts MV-SyDog (multi-view synthetic dog dataset with 2D/3D pose GT).

Why this is useful for DeepLabCut 3D:
  - DLC 3D workflows typically rely on multi-view data + camera calibration/geometry.
  - MV-SyDog provides multi-view frames and per-frame metadata (see frame_data.json),
    along with 2D/3D pose ground truth files.
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path


MVSYDOG_TAR_URL = "https://cvssp.org/data/MVSyDog/MVSyDog_full.tar.gz"


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[OK] Already exists: {out_path}")
        return

    print(f"[DL] {url}")
    print(f" -> {out_path}")
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        total = r.length or 0
        downloaded = 0
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100.0 / total
                print(f"\r    {downloaded/1e6:.1f} MB / {total/1e6:.1f} MB  ({pct:.1f}%)", end="")
        print()


def extract_tar(tar_path: Path, out_dir: Path) -> None:
    print(f"[EXTRACT] {tar_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(out_dir)
    print("[OK] Extract done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/mvsydog_3d", help="Output folder")
    ap.add_argument("--keep-tar", action="store_true", help="Keep the downloaded tar.gz")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tar_path = out_dir / "MVSyDog_full.tar.gz"
    download_file(MVSYDOG_TAR_URL, tar_path)
    extract_tar(tar_path, out_dir)

    if not args.keep_tar:
        try:
            tar_path.unlink()
            print(f"[CLEAN] Removed {tar_path}")
        except Exception as e:
            print(f"[WARN] Could not remove tarball: {e}")

    print("\nDone.")
    print(f"Dataset root: {out_dir}")
    print("Tip: look for per-frame metadata like '*.frame_data.json' and annotations folder after extraction.")


if __name__ == "__main__":
    main()