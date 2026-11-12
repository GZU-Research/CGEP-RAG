"""CGEP-RAG configuration and defaults."""

import json
from dataclasses import dataclass, fields
from pathlib import Path


PAPER_MODELS = ("gpt-3.5-turbo-0125", "gpt-4o-mini-2024-07-18")


@dataclass
class CoreConfig:
    model: str = "gpt-3.5-turbo-0125"
    temperature: float = 0.0
    top_p: float = 1.0
    extraction_max_tokens: int = 512
    answer_max_tokens: int = 64
    k_seed: int = 15
    k_exp: int = 5
    cluster_size: int = 10
    min_cluster_size: int = 5
    evidence_budget: int = 20
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_batch_size: int = 32
    extraction_workers: int = 15

    def validate(self) -> None:
        integer_fields = (
            "extraction_max_tokens", "answer_max_tokens", "k_seed", "k_exp",
            "cluster_size", "min_cluster_size", "evidence_budget",
            "embedding_batch_size", "extraction_workers",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.min_cluster_size > self.cluster_size:
            raise ValueError("min_cluster_size must not exceed cluster_size.")
        if self.model not in PAPER_MODELS:
            raise ValueError(f"model must be one of {PAPER_MODELS}.")
        if self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("temperature must be 0 and top_p must be 1.")
        if not isinstance(self.embedding_model, str) or not self.embedding_model:
            raise ValueError("embedding_model must identify a BGE checkpoint or local path.")

    @classmethod
    def from_json(cls, path: Path) -> "CoreConfig":
        with path.open(encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("The configuration must be a JSON object.")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config
