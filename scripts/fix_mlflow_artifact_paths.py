#!/usr/bin/env python3
"""Rewrite Colab artifact_uri paths so MLflow UI can serve local artifacts."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "mlruns" / "mlflow.db"


def fix_artifact_uris() -> int:
    if not DB_PATH.exists():
        print(f"Missing {DB_PATH}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT run_uuid, artifact_uri FROM runs").fetchall()
    updated = 0

    for run_uuid, uri in rows:
        if not uri:
            continue
        local_artifacts = PROJECT_ROOT / "mlruns" / "1" / run_uuid / "artifacts"
        if not local_artifacts.exists():
            continue
        # MLflow on Windows expects a file:// URI to the artifacts directory.
        new_uri = local_artifacts.resolve().as_uri()
        if uri != new_uri:
            conn.execute(
                "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                (new_uri, run_uuid),
            )
            print(f"fixed {run_uuid[:8]}...")
            print(f"  old: {uri}")
            print(f"  new: {new_uri}")
            updated += 1

    conn.commit()
    conn.close()
    print(f"\nUpdated {updated} run(s). Restart `mlflow ui` and refresh the champion run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(fix_artifact_uris())
