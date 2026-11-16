"""Download the model artifacts named in models/model_manifest.json and verify them.

The trained .joblib files are not in git. At deploy time (e.g. a Render build) this script downloads
them from a base URL, such as a GitHub Release, and saves each file ONLY if its SHA-256 matches the
checksum committed in model_manifest.json. A tampered or corrupted download is never written, so it
can never be unpickled.

    python -m ml.fetch_artifacts https://github.com/<user>/shopsense-ai/releases/download/models-v1
    # or: set MODEL_ARTIFACTS_URL and run without an argument

Files already present with the right checksum are skipped. Exit code 1 on any failure.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlsplit

from ml.inference import MODELS_DIR, load_manifest, safe_artifact_path, sha256_file
from ml.integrity import _entries

MAX_BYTES = 100 * 1024 * 1024        # largest artifact is ~8.5 MB
TIMEOUT_S = 60
USER_AGENT = "ShopSenseAI-artifact-fetch/1.0"


def _check_base(base: str) -> str:
    parts = urlsplit(base)
    loopback = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
        raise ValueError("MODEL_ARTIFACTS_URL must be an https:// URL (http:// only for localhost testing).")
    return base.rstrip("/")


def _download(url: str, dest: Path, expected_sha: str, opener=urllib.request.urlopen) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest, size = hashlib.sha256(), 0
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=".download-", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, opener(req, timeout=TIMEOUT_S) as resp:
            while chunk := resp.read(1 << 20):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError(f"{dest.name}: larger than {MAX_BYTES} bytes, refusing")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != expected_sha:
            raise ValueError(f"{dest.name}: checksum MISMATCH (got {digest.hexdigest()[:16]}…, "
                             f"manifest {expected_sha[:16]}…); file discarded")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def fetch(base: str, models_dir: Path = MODELS_DIR, opener=urllib.request.urlopen) -> list[str]:
    base = _check_base(base)
    problems: list[str] = []
    for variant, node, file_key, sha_key in _entries(load_manifest(models_dir)):
        name, expected = node[file_key], node.get(sha_key)
        if not expected:
            problems.append(f"{variant}: {name} has no recorded checksum")
            continue
        try:
            dest = safe_artifact_path(models_dir, name)      # plain file name inside models/ only
            if dest.exists() and sha256_file(dest) == expected:
                print(f"present  {name}")
                continue
            _download(f"{base}/{quote(name)}", dest, expected, opener)
            print(f"fetched  {name}  sha256 {expected[:16]}… ok")
        except Exception as exc:   # report every file, then fail
            problems.append(f"{name}: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    base = argv[0] if argv else os.environ.get("MODEL_ARTIFACTS_URL", "")
    if not base:
        print("Usage: python -m ml.fetch_artifacts <base-url>   (or set MODEL_ARTIFACTS_URL)")
        return 2
    try:
        problems = fetch(base)
    except ValueError as exc:
        print("FAIL    ", exc)
        return 1
    for msg in problems:
        print("FAIL    ", msg)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
