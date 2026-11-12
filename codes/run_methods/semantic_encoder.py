"""Lazy BGE encoding for the evidence-purification stage."""

from typing import List, Optional

import numpy as np


class SemanticEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = 32,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return unit embeddings without adding query instructions or prefixes."""
        if not texts:
            return np.empty((0, 0), dtype=float)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("All texts passed to the encoder must be nonempty strings.")
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=self.device, cache_folder=self.cache_dir
            )
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=float)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(embeddings).all() or (norms == 0).any():
            raise ValueError("The encoder returned a nonfinite or zero embedding.")
        return embeddings / norms
