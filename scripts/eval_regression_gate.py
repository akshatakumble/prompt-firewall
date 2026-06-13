#!/usr/bin/env python3
"""Security SLA regression gate — fails if Salad-Data eval misses recall/FPR targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EVAL_PATH = PROJECT_ROOT / "reports" / "eval_salad-data.json"
MIN_RECALL = 0.82
MAX_FPR = 0.08


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--min-recall", type=float, default=MIN_RECALL)
    parser.add_argument("--max-fpr", type=float, default=MAX_FPR)
    args = parser.parse_args()

    eval_path = Path(args.eval_json)
    if not eval_path.exists():
        print(f"Missing eval report: {eval_path}", file=sys.stderr)
        sys.exit(1)

    with open(eval_path, encoding="utf-8") as handle:
        metrics = json.load(handle)

    recall = float(metrics.get("recall", 0.0))
    fpr = float(metrics.get("fpr", 1.0))
    passed = recall >= args.min_recall and fpr <= args.max_fpr

    gate = {
        "passed": passed,
        "recall": recall,
        "fpr": fpr,
        "min_recall": args.min_recall,
        "max_fpr": args.max_fpr,
        "eval_path": str(eval_path),
    }

    out_path = PROJECT_ROOT / "reports" / "regression_gate.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2)

    print(json.dumps(gate, indent=2))
    if not passed:
        print(
            f"REGRESSION GATE FAILED: recall={recall:.3f} (min {args.min_recall}), "
            f"FPR={fpr:.3f} (max {args.max_fpr})",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
