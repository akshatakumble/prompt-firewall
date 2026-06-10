"""Manifest generation with counts and checksums."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "label" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}


def build_manifest(
    *,
    version: str,
    artifacts: dict[str, Path],
    splits: dict[str, pd.DataFrame] | None = None,
    extra: dict[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Build a registry manifest with row counts and SHA-256 checksums."""

    manifest: dict[str, Any] = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {},
        "splits": {},
    }

    for name, path in artifacts.items():
        rel_path = path
        if project_root is not None:
            try:
                rel_path = path.relative_to(project_root)
            except ValueError:
                rel_path = path
        rel_path_str = rel_path.as_posix()
        entry: dict[str, Any] = {"path": rel_path_str}
        if path.exists() and path.is_file():
            entry["checksum_sha256"] = file_sha256(path)
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
                entry["rows"] = len(df)
                entry["label_counts"] = label_counts(df)
        manifest["artifacts"][name] = entry

    if splits:
        for split_name, split_df in splits.items():
            manifest["splits"][split_name] = {
                "rows": len(split_df),
                "label_counts": label_counts(split_df),
            }

    if extra:
        manifest.update(extra)

    return manifest


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
