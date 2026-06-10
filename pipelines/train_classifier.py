#!/usr/bin/env python3
"""Fine-tune DistilBERT binary injection classifier with MLflow tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PromptDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def label_to_int(label: str) -> int:
    return 1 if label == "INJECTION" else 0


def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        pred = torch.argmax(outputs.logits, dim=-1)
        preds.extend(pred.cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return preds, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    splits_dir = PROJECT_ROOT / config["paths"]["curated"] / "splits"
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "validation.parquet")

    model_name = config["classifier"]["model_name"]
    model_path = PROJECT_ROOT / config["classifier"]["model_path"]
    model_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    max_length = config["classifier"]["max_length"]
    train_ds = PromptDataset(
        train_df["text"].tolist(),
        [label_to_int(x) for x in train_df["label"]],
        tokenizer,
        max_length,
    )
    val_ds = PromptDataset(
        val_df["text"].tolist(),
        [label_to_int(x) for x in val_df["label"]],
        tokenizer,
        max_length,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    mlflow.set_tracking_uri(config.get("mlflow_tracking_uri", "sqlite:///./mlruns/mlflow.db"))
    with mlflow.start_run(run_name="distilbert-injection-classifier"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "dataset_version": config["dataset_version"],
            }
        )

        for epoch in range(args.epochs):
            loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            preds, labels = evaluate(model, val_loader, device)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="binary", pos_label=1
            )
            print(f"Epoch {epoch + 1}: loss={loss:.4f}, recall={recall:.3f}, f1={f1:.3f}")
            mlflow.log_metrics({"train_loss": loss, "val_recall": recall, "val_f1": f1}, step=epoch)

        preds, labels = evaluate(model, val_loader, device)
        report = classification_report(labels, preds, target_names=["BENIGN", "INJECTION"], output_dict=True)
        mlflow.log_metrics(
            {
                "val_precision_injection": report["INJECTION"]["precision"],
                "val_recall_injection": report["INJECTION"]["recall"],
                "val_f1_injection": report["INJECTION"]["f1-score"],
            }
        )

        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        mlflow.log_artifact(str(model_path))

        metrics_path = PROJECT_ROOT / "reports" / "train_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
