"""Download the UCI Online Shoppers Purchasing Intention Dataset (ID 468).

Source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The script:
  1. downloads the official zip into backend/data/raw/
  2. extracts ONLY the expected CSV member (never arbitrary paths from the archive)
  3. writes backend/data/online_shoppers_intention.csv (never overwrites unless --force)
  4. records its SHA-256 in backend/data/online_shoppers_intention.sha256 so later
     steps can verify the original file was never modified.

Run from the backend directory:
    python -I ml/download_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
CSV_MEMBER = "online_shoppers_intention.csv"

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = RAW_DIR / "online_shoppers_purchasing_intention_dataset.zip"
CSV_PATH = DATA_DIR / CSV_MEMBER
CHECKSUM_PATH = DATA_DIR / f"{Path(CSV_MEMBER).stem}.sha256"

MAX_CSV_BYTES = 50 * 1024 * 1024  # the real file is ~1 MB; refuse anything absurd


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def tls_context() -> ssl.SSLContext:
    """Verified TLS context using certifi's current Mozilla CA bundle when available.

    Some Windows certificate stores lack newer Let's Encrypt roots, so OpenSSL builds
    the chain through an expired cross-signed intermediate and fails. certifi ships
    the current roots. Certificate and hostname verification stay fully enabled.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "shopsense-ai-data-fetch"})
    with (
        urllib.request.urlopen(req, timeout=60, context=tls_context()) as resp,
        dest.open("wb") as out,
    ):
        while chunk := resp.read(1 << 16):
            out.write(chunk)


def extract_csv(zip_path: Path, member: str, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"Archive members: {names}")
        matches = [n for n in names if Path(n).name == member and not n.endswith("/")]
        if len(matches) != 1:
            raise SystemExit(f"Expected exactly one '{member}' in archive, found {matches}")
        info = zf.getinfo(matches[0])
        if info.file_size > MAX_CSV_BYTES:
            raise SystemExit(f"'{member}' is unexpectedly large ({info.file_size} bytes)")
        # Read bytes and write to a fixed path: no archive-supplied paths are used.
        dest.write_bytes(zf.read(info))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download and overwrite")
    args = parser.parse_args()

    if CSV_PATH.exists() and not args.force:
        print(f"Dataset already present: {CSV_PATH} (use --force to re-download)")
        return 0

    print(f"Downloading {DATASET_URL}")
    download(DATASET_URL, ZIP_PATH)
    print(f"Saved archive -> {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")

    extract_csv(ZIP_PATH, CSV_MEMBER, CSV_PATH)
    digest = sha256_of(CSV_PATH)
    CHECKSUM_PATH.write_text(f"{digest}  {CSV_MEMBER}\n", encoding="utf-8")

    print(f"Extracted CSV -> {CSV_PATH} ({CSV_PATH.stat().st_size:,} bytes)")
    print(f"SHA-256: {digest} (recorded in {CHECKSUM_PATH.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
