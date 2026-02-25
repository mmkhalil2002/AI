#!/usr/bin/env python3
"""
download_stanfordextra_2d.py

Downloads:
  1) Stanford Dogs images (images.tar)  [required]
  2) StanfordExtra annotations (optional; requires the emailed link after filling the form)

Why this is useful for DeepLabCut:
  - You can create a DLC 2D project using these images/videos and label the keypoints you care about.
  - StanfordExtra provides 2D keypoints + segmentations for many of these dogs (if you download annotations).
"""

import argparse
import hashlib
import os
import tarfile
import urllib.request
from pathlib import Path


STANFORD_DOGS_IMAGES_TAR_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"


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
    ap.add_argument("--out", default="datasets/stanfordextra_2d", help="Output folder")
    ap.add_argument(
        "--ann-url",
        default="",
        help="(Optional) Direct StanfordExtra annotations URL you received by email after filling the Google form.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Stanford Dogs images
    images_tar = out_dir / "images.tar"
    download_file(STANFORD_DOGS_IMAGES_TAR_URL, images_tar)
    extract_tar(images_tar, out_dir / "StanfordDogs")

    # 2) Optional StanfordExtra annotations (user provides the emailed URL)
    if args.ann_url.strip():
        ann_path = out_dir / "StanfordExtra_annotations.json"
        download_file(args.ann_url.strip(), ann_path)
        print(f"[OK] StanfordExtra annotations saved to: {ann_path}")
    else:
        print("\n[NOTE] StanfordExtra annotations are not downloaded automatically.")
        print("       Fill the StanfordExtra Google form, then re-run with --ann-url <emailed_link>.")
        print("       (The StanfordExtra repo explains this flow.)")

    print("\nDone.")
    print(f"Images folder: {out_dir / 'StanfordDogs'}")


if __name__ == "__main__":
    main()