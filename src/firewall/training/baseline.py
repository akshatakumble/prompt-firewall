"""TF-IDF + Logistic Regression baseline classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from firewall.training.metrics import metrics_from_arrays


@dataclass
class BaselineModel:
    pipeline: Pipeline

    def fit(self, texts: list[str], labels: list[int]) -> None:
        self.pipeline.fit(texts, labels)

    def predict(self, texts: list[str], *, threshold: float = 0.5) -> tuple[list[int], list[float]]:
        prob = self.pipeline.predict_proba(texts)[:, 1]
        pred = [1 if p >= threshold else 0 for p in prob]
        return pred, prob.tolist()

    def evaluate(self, texts: list[str], labels: list[int], *, threshold: float = 0.5) -> dict[str, Any]:
        pred, prob = self.predict(texts, threshold=threshold)
        return metrics_from_arrays(labels, pred, prob)

    def save(self, path: str) -> None:
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> BaselineModel:
        return cls(pipeline=joblib.load(path))


def build_baseline() -> BaselineModel:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    return BaselineModel(pipeline=pipeline)
