"""Hugging Face dataset normalizers for the unified schema."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from firewall.data.schema import BENIGN_DATA_TYPES, INJECTION_DATA_TYPES

logger = logging.getLogger(__name__)

WILDJAILBREAK_REPO = "allenai/wildjailbreak"
WILDJAILBREAK_FALLBACK_REPO = "allenai/wildjailbreak-r1-v2-format-filtered"


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def make_id(text: str, source: str) -> str:
    return hashlib.md5(f"{source}:{text}".encode("utf-8")).hexdigest()


def _row(
    *,
    text: str,
    label: str,
    source: str,
    attack_type: str,
    metadata: dict[str, Any],
) -> dict[str, str]:
    return {
        "id": make_id(text, source),
        "text": text,
        "label": label,
        "source": source,
        "attack_type": attack_type,
        "metadata": json.dumps(metadata, sort_keys=True),
    }


def _format_tactics(tactics: Any) -> str:
    if tactics is None:
        return "unknown"
    if isinstance(tactics, list):
        return ",".join(str(item) for item in tactics) if tactics else "unknown"
    return str(tactics)


def label_from_data_type(data_type: str) -> str:
    if data_type in INJECTION_DATA_TYPES:
        return "INJECTION"
    if data_type in BENIGN_DATA_TYPES:
        return "BENIGN"
    return "INJECTION" if "harmful" in data_type else "BENIGN"


def _extract_user_prompt(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "").strip()
    return ""


def normalize_wildjailbreak_row(row: dict[str, Any], index: int) -> dict[str, str] | None:
    adversarial = str(row.get("adversarial") or "").strip()
    vanilla = str(row.get("vanilla") or "").strip()
    text = adversarial or vanilla
    if not text:
        return None

    data_type = str(row.get("data_type") or "").strip()
    label = label_from_data_type(data_type)

    return _row(
        text=text,
        label=label,
        source="train_wildjailbreak",
        attack_type=_format_tactics(row.get("tactics")) if label == "INJECTION" else "benign",
        metadata={
            "index": index,
            "data_type": data_type,
            "has_adversarial": bool(adversarial),
            "hf_repo": WILDJAILBREAK_REPO,
        },
    )


def normalize_wildjailbreak_filtered_row(row: dict[str, Any], index: int) -> dict[str, str] | None:
    text = _extract_user_prompt(row.get("messages"))
    if not text:
        return None

    data_type = str(row.get("prompt_harm_label") or "").strip()
    label = label_from_data_type(data_type)

    return _row(
        text=text,
        label=label,
        source="train_wildjailbreak",
        attack_type=data_type if label == "INJECTION" else "benign",
        metadata={
            "index": index,
            "data_type": data_type,
            "model_used": row.get("model_used"),
            "regenerated_model_type": row.get("regenerated_model_type"),
            "hf_repo": WILDJAILBREAK_FALLBACK_REPO,
        },
    )


def _save_jsonl_snapshot(dataset: Any, snapshot_path: Path) -> None:
    if not snapshot_path.exists():
        dataset.to_json(str(snapshot_path), orient="records", lines=True)
        logger.info("Saved raw snapshot to %s", snapshot_path)


def _load_allenai_wildjailbreak_train(raw_dir: Path | None) -> tuple[pd.DataFrame, str]:
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    token = hf_token()
    if not token:
        logger.warning("HF_TOKEN not set; gated datasets may fail to download.")

    try:
        logger.info("Loading %s train split...", WILDJAILBREAK_REPO)
        ds = load_dataset(
            WILDJAILBREAK_REPO,
            "train",
            delimiter="\t",
            keep_default_na=False,
            token=token,
        )["train"]
        normalizer = normalize_wildjailbreak_row
        source_repo = WILDJAILBREAK_REPO
    except DatasetNotFoundError as exc:
        logger.warning(
            "Could not access %s (%s). Falling back to %s.",
            WILDJAILBREAK_REPO,
            exc,
            WILDJAILBREAK_FALLBACK_REPO,
        )
        logger.warning(
            "For the full 262K dataset, accept access at "
            "https://huggingface.co/datasets/allenai/wildjailbreak while logged in."
        )
        ds = load_dataset(WILDJAILBREAK_FALLBACK_REPO, split="train", token=token)
        normalizer = normalize_wildjailbreak_filtered_row
        source_repo = WILDJAILBREAK_FALLBACK_REPO

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        snapshot_name = source_repo.split("/")[-1] + "_train.jsonl"
        _save_jsonl_snapshot(ds, raw_dir / snapshot_name)

    rows: list[dict[str, str]] = []
    for index, row in enumerate(ds):
        normalized = normalizer(row, index)
        if normalized:
            rows.append(normalized)

    logger.info("Normalized %s rows from %s", len(rows), source_repo)
    return pd.DataFrame(rows), source_repo


def load_wildjailbreak(raw_dir: Path | None = None) -> pd.DataFrame:
    local_snapshot = None
    if raw_dir is not None:
        for candidate in (
            raw_dir / "wildjailbreak_train.jsonl",
            raw_dir / "wildjailbreak-r1-v2-format-filtered_train.jsonl",
        ):
            if candidate.exists():
                local_snapshot = candidate
                break

    if local_snapshot is not None:
        logger.info("Loading WildJailbreak from local snapshot %s", local_snapshot)
        df = pd.read_json(local_snapshot, lines=True)
        rows: list[dict[str, str]] = []
        normalizer = (
            normalize_wildjailbreak_row
            if "data_type" in df.columns
            else normalize_wildjailbreak_filtered_row
        )
        for index, row in df.iterrows():
            normalized = normalizer(row.to_dict(), int(index))
            if normalized:
                rows.append(normalized)
        return pd.DataFrame(rows)

    df, _ = _load_allenai_wildjailbreak_train(raw_dir)
    return df


def load_salad_data(raw_dir: Path | None = None) -> pd.DataFrame:
    from datasets import load_dataset

    logger.info("Loading OpenSafetyLab/Salad-Data subsets...")
    token = hf_token()
    base_ds = load_dataset("OpenSafetyLab/Salad-Data", "base_set", split="train", token=token)
    attack_ds = load_dataset(
        "OpenSafetyLab/Salad-Data",
        "attack_enhanced_set",
        split="train",
        token=token,
    )

    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for name, dataset in (("base_set", base_ds), ("attack_enhanced_set", attack_ds)):
            snapshot_path = raw_dir / f"salad_{name}.jsonl"
            _save_jsonl_snapshot(dataset, snapshot_path)

    rows: list[dict[str, str]] = []

    for index, row in enumerate(base_ds):
        text = str(row.get("question") or "").strip()
        if not text:
            continue
        category = str(row.get("2-category") or row.get("1-category") or "adversarial")
        rows.append(
            _row(
                text=text,
                label="INJECTION",
                source="eval_salad-data",
                attack_type=category,
                metadata={
                    "index": index,
                    "subset": "base_set",
                    "qid": row.get("qid"),
                    "source_origin": row.get("source"),
                },
            )
        )

    for index, row in enumerate(attack_ds):
        text = str(row.get("augq") or "").strip()
        if not text:
            continue
        category = str(row.get("method") or row.get("2-category") or "jailbreak")
        rows.append(
            _row(
                text=text,
                label="INJECTION",
                source="eval_salad-data",
                attack_type=str(category),
                metadata={
                    "index": index,
                    "subset": "attack_enhanced_set",
                    "aid": row.get("aid"),
                    "method": row.get("method"),
                },
            )
        )

    logger.info("Normalized %s Salad-Data rows", len(rows))
    return pd.DataFrame(rows)


TRAIN_LOADERS = {
    "wildjailbreak": load_wildjailbreak,
}

BENCHMARK_LOADERS = {
    "salad-data": load_salad_data,
}
