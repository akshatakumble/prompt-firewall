#!/usr/bin/env python3
"""Run regression attack suite and emit CI-friendly metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_regression_suite.py", "-q"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    report = {
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    out_path = PROJECT_ROOT / "reports" / "regression_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
