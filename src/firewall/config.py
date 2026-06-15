from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Thresholds(BaseModel):
    allow_max: float = 0.50
    sanitize_min: float = 0.50
    sanitize_max: float = 0.80
    block_min: float = 0.80
    near_miss_min: float = 0.45


class ClassifierConfig(BaseModel):
    model_name: str = "distilbert-base-uncased"
    model_path: str = "models/classifier-v1"
    fallback_mode: str = "rules_only"
    max_length: int = 256


class AppConfig(BaseModel):
    policy_version: str = "v1.0"
    model_version: str = "v1.0"
    dataset_version: str = "v1.0"
    thresholds: Thresholds = Field(default_factory=Thresholds)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    llm: dict[str, Any] = Field(default_factory=dict)
    datasets: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, str] = Field(default_factory=dict)
    telemetry: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseSettings):
    groq_api_key: str = ""
    hf_token: str = ""
    victim_llm_provider: str = "groq"
    victim_llm_model: str = "llama-3.1-8b-instant"
    database_url: str = "sqlite:///./data/firewall.db"
    mlflow_tracking_uri: str = "sqlite:///./mlruns/mlflow.db"
    policy_version: str = "v1.0"
    model_version: str = "v1.0"
    dataset_version: str = "v1.0"

    model_config = SettingsConfigDict(
        env_file=(
            str(PROJECT_ROOT / ".env"),
            str(PROJECT_ROOT / ".secrets" / "groq.env"),
        ),
        extra="ignore",
    )


@lru_cache
def load_app_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or CONFIG_DIR / "app.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)


@lru_cache
def load_rules_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or CONFIG_DIR / "rules.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_settings() -> Settings:
    return Settings()
