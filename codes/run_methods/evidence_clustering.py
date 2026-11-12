"""Query-anchored clustering and three-signal evidence purification."""

import math
from typing import Dict, Iterable, List

import numpy as np


class DocumentClusterer:
    """Cluster and rank passages using query, seed, and clue signals."""

    def __init__(self, encoder, initial_size: int = 10, min_size: int = 5):
        if initial_size < 1 or min_size < 1 or min_size > initial_size:
            raise ValueError("Cluster sizes must satisfy 1 <= min_size <= initial_size.")
        self.encoder = encoder
        self.initial_size = initial_size
        self.min_size = min_size

    @staticmethod
    def _clue_text(clue: Dict) -> str:
        # The source sentence is not part of the triple embedding.
        fields = [clue.get(name) for name in ("head", "relation", "tail")]
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("Clues must contain nonempty head, relation, and tail strings.")
        return " ".join(value.strip() for value in fields)

    @staticmethod
    def _normalize_signals(clusters: List[Dict]) -> None:
        for name in ("S_q", "S_w", "S_c"):
            values = [cluster[name] for cluster in clusters]
            minimum, maximum = min(values), max(values)
            span = maximum - minimum
            for cluster in clusters:
                cluster[name + "_normalized"] = (
                    (cluster[name] - minimum) / span if span > 0.0 else 0.0
                )
        for cluster in clusters:
            cluster["score"] = sum(
                cluster[name + "_normalized"] for name in ("S_q", "S_w", "S_c")
            ) / 3.0

    def purify(
        self,
        question: str,
        candidate_documents: List[Dict],
        seed_document_ids: Iterable[str],
        clues: List[Dict],
        budget: int = 20,
    ) -> Dict:
        """Cluster the full pool, rank clusters, and take up to ``budget`` passages.

        Candidate order breaks equal query/root similarities; cluster creation
        order breaks equal fused scores. Every candidate belongs to one cluster.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string.")
        if budget < 1:
            raise ValueError("The source-passage budget must be positive.")
        if not candidate_documents:
            return {"selected_documents": [], "selected_document_ids": [], "clusters": []}

        document_ids = []
        document_texts = []
        for document in candidate_documents:
            doc_id = document.get("id")
            text = document.get("text")
            if doc_id is None or not str(doc_id).strip():
                raise ValueError("Every candidate must have a stable, nonempty id.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Candidate {doc_id!r} has no passage text.")
            document_ids.append(str(doc_id))
            document_texts.append(text)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Candidate passages must be deduplicated by id before purification.")

        seed_ids = {str(doc_id) for doc_id in seed_document_ids}
        clue_texts = [self._clue_text(clue) for clue in clues]
        embeddings = self.encoder.encode([question] + document_texts + clue_texts)
        document_count = len(candidate_documents)
        query_embedding = embeddings[0]
        document_embeddings = embeddings[1 : document_count + 1]
        clue_embeddings = embeddings[document_count + 1 :]

        # Unit-vector dot products give cosine similarity.
        query_similarities = document_embeddings @ query_embedding
        document_similarities = document_embeddings @ document_embeddings.T
        if clue_texts:
            document_clue_scores = (document_embeddings @ clue_embeddings.T).mean(axis=1)
        else:
            document_clue_scores = np.zeros(document_count, dtype=float)

        remaining = list(range(document_count))
        requested_size = self.initial_size
        clusters = []
        cluster_indices = {}
        while remaining:
            root = min(remaining, key=lambda index: (-query_similarities[index], index))
            neighbors = sorted(
                (index for index in remaining if index != root),
                key=lambda index: (-document_similarities[root, index], index),
            )
            member_indices = [root] + neighbors[: requested_size - 1]
            # Query ranks define both S_w positions and passage selection order.
            member_indices.sort(key=lambda index: (-query_similarities[index], index))
            cluster_id = len(clusters)
            cluster_indices[cluster_id] = member_indices
            seed_alignment = sum(
                1.0 / math.log2(position + 1)
                for position, index in enumerate(member_indices, start=1)
                if document_ids[index] in seed_ids
            )
            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "root_doc_id": document_ids[root],
                    "document_ids": [document_ids[index] for index in member_indices],
                    "query_similarities": [float(query_similarities[index]) for index in member_indices],
                    "requested_size": requested_size,
                    "size": len(member_indices),
                    "S_q": float(query_similarities[member_indices].mean()),
                    "S_w": float(seed_alignment),
                    "S_c": float(document_clue_scores[member_indices].mean()),
                }
            )
            assigned = set(member_indices)
            remaining = [index for index in remaining if index not in assigned]
            requested_size = max(requested_size // 2, self.min_size)

        self._normalize_signals(clusters)
        clusters.sort(key=lambda cluster: (-cluster["score"], cluster["cluster_id"]))
        selected_documents = []
        for rank, cluster in enumerate(clusters, start=1):
            cluster["rank"] = rank
            remaining_budget = budget - len(selected_documents)
            if remaining_budget > 0:
                indices = cluster_indices[cluster["cluster_id"]][:remaining_budget]
                selected_documents.extend(candidate_documents[index] for index in indices)

        return {
            "selected_documents": selected_documents,
            "selected_document_ids": [str(document["id"]) for document in selected_documents],
            "clusters": clusters,
        }
