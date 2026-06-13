"""DistilBERT fine-tuning for binary injection classification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from firewall.training.metrics import metrics_from_arrays


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


@dataclass
class DistilBertClassifier:
    model_name: str
    max_length: int = 128
    model: Any = None
    tokenizer: Any = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    class_weights: torch.Tensor | None = None

    def build(self, train_labels: list[int]) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=np.array(train_labels))
        self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)
        self.model.to(self.device)

    def _loader(self, texts: list[str], labels: list[int], batch_size: int, shuffle: bool) -> DataLoader:
        ds = PromptDataset(texts, labels, self.tokenizer, self.max_length)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def train(
        self,
        train_texts: list[str],
        train_labels: list[int],
        val_texts: list[str],
        val_labels: list[int],
        *,
        epochs: int = 2,
        batch_size: int = 32,
        lr: float = 2e-5,
        warmup_ratio: float = 0.1,
    ) -> list[dict[str, Any]]:
        assert self.model is not None and self.tokenizer is not None
        train_loader = self._loader(train_texts, train_labels, batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        total_steps = max(len(train_loader) * epochs, 1)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_f1 = -1.0
        best_state: dict[str, Any] | None = None
        history: list[dict[str, Any]] = []

        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            val_metrics = self.evaluate(val_texts, val_labels, batch_size=batch_size)
            val_metrics["epoch"] = epoch + 1
            val_metrics["train_loss"] = round(train_loss, 4)
            history.append(val_metrics)

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return history

    def _train_epoch(self, loader: DataLoader, optimizer, scheduler) -> float:
        assert self.model is not None and self.class_weights is not None
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = F.cross_entropy(outputs.logits, labels, weight=self.class_weights)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def predict_proba(self, texts: list[str], *, batch_size: int = 32) -> list[float]:
        assert self.model is not None and self.tokenizer is not None
        loader = self._loader(texts, [0] * len(texts), batch_size, shuffle=False)
        self.model.eval()
        probs: list[float] = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
            outputs = self.model(**batch)
            batch_probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().tolist()
            probs.extend(batch_probs)
        return probs

    def evaluate(
        self,
        texts: list[str],
        labels: list[int],
        *,
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        prob = self.predict_proba(texts, batch_size=batch_size)
        pred = [1 if p >= threshold else 0 for p in prob]
        return metrics_from_arrays(labels, pred, prob)

    def save(self, path: str) -> None:
        assert self.model is not None and self.tokenizer is not None
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.model.eval()
