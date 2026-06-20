#!/usr/bin/env python3
"""
Download the trained classifier bundle from the GCS model registry into the
local path the API loads from (``models/classifier-v1`` by default).

This is run at container startup (see docker/cloudrun-entrypoint.sh) so the
deployment always pulls the latest published model from the registry instead of
baking weights into the image. Authentication uses Application Default
Credentials — on Cloud Run that is the runtime service account automatically.

Env vars (all optional, sensible defaults):
  MODEL_GCS_URI    gs://bucket/prefix of the model bundle
                   (default: gs://prompt-firewall-mlops-dvc/models/v2)
  MODEL_LOCAL_DIR  local destination (default: models/classifier-v1)
  MODEL_FETCH_FORCE  "1" to re-download even if the dir already has config.json
  MODEL_FETCH_OPTIONAL "1" to exit 0 (warn only) if download fails — lets the
                   API boot in rules-only fallback mode instead of crash-looping.

Usage:
  python scripts/fetch_model.py
  python scripts/fetch_model.py --gcs-uri gs://my-bucket/models/v3 --dest models/classifier-v1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DEFAULT_GCS_URI = "gs://prompt-firewall-mlops-dvc/models/v2"
DEFAULT_LOCAL_DIR = "models/classifier-v1"


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"MODEL_GCS_URI must start with gs:// (got: {uri})")
    without = uri[len("gs://"):]
    bucket, _, prefix = without.partition("/")
    return bucket, prefix.strip("/")


def download_prefix(gcs_uri: str, dest: Path) -> int:
    """Download every object under the GCS prefix into ``dest`` (flattened)."""
    from google.cloud import storage  # imported lazily so --help works without the dep

    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket, prefix=f"{prefix}/" if prefix else None))
    # only keep files (skip "directory placeholder" blobs ending in /)
    blobs = [b for b in blobs if not b.name.endswith("/")]
    if not blobs:
        raise FileNotFoundError(f"No objects found under {gcs_uri}")

    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in blobs:
        # flatten: strip the prefix so files land directly in dest/
        rel = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        print(f"  ↓ {blob.name} -> {target}")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch classifier model from GCS registry.")
    parser.add_argument("--gcs-uri", default=os.getenv("MODEL_GCS_URI", DEFAULT_GCS_URI))
    parser.add_argument("--dest", default=os.getenv("MODEL_LOCAL_DIR", DEFAULT_LOCAL_DIR))
    parser.add_argument("--force", action="store_true",
                        default=os.getenv("MODEL_FETCH_FORCE") == "1")
    args = parser.parse_args()

    dest = Path(args.dest)
    optional = os.getenv("MODEL_FETCH_OPTIONAL") == "1"

    if (dest / "config.json").exists() and not args.force:
        print(f"Model already present at {dest} (config.json found) — skipping download.")
        return 0

    print(f"Fetching model: {args.gcs_uri} -> {dest}")
    try:
        n = download_prefix(args.gcs_uri, dest)
    except Exception as exc:  # noqa: BLE001 — startup robustness
        msg = f"Model fetch failed: {exc}"
        if optional:
            print(f"WARNING: {msg}\nContinuing without a model (rules-only fallback).",
                  file=sys.stderr)
            return 0
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    if not (dest / "config.json").exists():
        print("ERROR: download completed but config.json is missing — bundle looks incomplete.",
              file=sys.stderr)
        return 0 if optional else 1

    print(f"Downloaded {n} files to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
