from __future__ import annotations

import logging
from pathlib import Path

from firewall.config import ClassifierConfig, PROJECT_ROOT

logger = logging.getLogger(__name__)


class ClassifierScorer:
    """Binary injection classifier with graceful fallback to rules-only mode."""

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        model_path = PROJECT_ROOT / self.config.model_path
        try:
            if model_path.exists():
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self._model.eval()
                self._loaded = True
                logger.info("Loaded fine-tuned classifier from %s", model_path)
            else:
                logger.warning(
                    "No fine-tuned model at %s; using heuristic scorer fallback",
                    model_path,
                )
        except Exception as exc:
            logger.warning("Classifier load failed (%s); using heuristic fallback", exc)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def score(self, text: str) -> float:
        if self._loaded and self._model is not None and self._tokenizer is not None:
            return self._score_with_model(text)
        return self._heuristic_score(text)

    def _score_with_model(self, text: str) -> float:
        import torch

        inputs = self._tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # label 1 = INJECTION
            return float(probs[0][1].item())

    def _heuristic_score(self, text: str) -> float:
        """Lightweight fallback when no trained model is available."""
        lowered = text.lower()
        signals = [
            ("ignore previous", 0.7),
            ("system prompt", 0.65),
            ("jailbreak", 0.8),
            ("do anything now", 0.75),
            ("developer mode", 0.7),
            ("pretend you are", 0.5),
            ("bypass", 0.55),
            ("override", 0.5),
        ]
        score = 0.0
        for phrase, weight in signals:
            if phrase in lowered:
                score = max(score, weight)
        if len(text) > 2000:
            score = max(score, 0.35)
        return min(score, 1.0)
