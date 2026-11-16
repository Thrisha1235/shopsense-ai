"""Record or verify SHA-256 checksums of model artifacts in models/model_manifest.json.

joblib artifacts are pickles. The API verifies each file against these checksums BEFORE
unpickling it, so a swapped or corrupted file is refused instead of executed.

    python -m ml.integrity --record   # write checksums for the current artifacts (no model changes)
    python -m ml.integrity            # verify (exit code 1 on any mismatch), e.g. after a download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml.inference import MANIFEST_NAME, MODELS_DIR, load_manifest, safe_artifact_path, sha256_file

# (manifest location, file-name key, checksum key)
ARTIFACTS = [
    ((), "model_file", "model_sha256"),
    ((), "preprocessor_file", "preprocessor_sha256"),
    (("calibration",), "calibrator_file", "calibrator_sha256"),
]


def _entries(manifest: dict):
    for variant, meta in manifest["models"].items():
        for path_keys, file_key, sha_key in ARTIFACTS:
            node = meta
            for k in path_keys:
                node = node.get(k) or {}
            if node.get(file_key):
                yield variant, node, file_key, sha_key


def record(models_dir: Path = MODELS_DIR) -> dict:
    manifest = load_manifest(models_dir)
    for variant, node, file_key, sha_key in _entries(manifest):
        node[sha_key] = sha256_file(safe_artifact_path(models_dir, node[file_key]))
        print(f"recorded {variant:22s} {node[file_key]:45s} {node[sha_key][:16]}…")
    (Path(models_dir) / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify(models_dir: Path = MODELS_DIR) -> list[str]:
    problems = []
    manifest = load_manifest(models_dir)
    for variant, node, file_key, sha_key in _entries(manifest):
        path = safe_artifact_path(models_dir, node[file_key])
        if not path.exists():
            problems.append(f"{variant}: {node[file_key]} missing")
        elif sha_key not in node:
            problems.append(f"{variant}: {node[file_key]} has no recorded checksum")
        elif sha256_file(path) != node[sha_key]:
            problems.append(f"{variant}: {node[file_key]} checksum MISMATCH")
        else:
            print(f"ok       {variant:22s} {node[file_key]}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--record", action="store_true", help="write checksums into the manifest")
    args = p.parse_args(argv)
    if args.record:
        record()
        return 0
    problems = verify()
    for msg in problems:
        print("FAIL    ", msg)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
