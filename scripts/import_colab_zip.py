#!/usr/bin/env python3
"""Extract Colab artifact folders from a project zip into the local repo."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

PREFIXES = (
    "models/classifier-v1/",
    "reports/",
    "mlruns/",
    "data/bias/",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", help="Path to prompt-firewall.zip from Colab")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    root = Path(args.project_root)
    if not zip_path.exists():
        print(f"Missing zip: {zip_path}", file=sys.stderr)
        return 1

    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not any(name.startswith(prefix) for prefix in PREFIXES):
                continue
            if name.endswith("/"):
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(name))
            extracted += 1

    print(f"Extracted {extracted} files into {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
