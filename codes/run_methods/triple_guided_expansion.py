"""One-pass BM25 expansion, with one query per retained relational clue."""

from typing import Dict, List


def unique_documents(documents: List[Dict]) -> List[Dict]:
    """Deduplicate by corpus document ID while preserving retrieval order."""
    result = []
    seen = set()
    for document in documents:
        doc_id = document.get("id", document.get("doc_id"))
        if doc_id is None or not str(doc_id).strip():
            raise ValueError("Retrieved passages must have stable document IDs.")
        doc_id = str(doc_id)
        if doc_id not in seen:
            seen.add(doc_id)
            result.append({**document, "id": doc_id})
    return result


class EntityExpansionRetriever:
    def __init__(self, retriever, kg_extractor, k_seed: int = 15, k_exp: int = 5):
        if k_seed < 1 or k_exp < 1:
            raise ValueError("k_seed and k_exp must be positive.")
        self.retriever = retriever
        self.kg_extractor = kg_extractor
        self.k_seed = k_seed
        self.k_exp = k_exp

    def retrieve(self, question: str) -> Dict:
        seeds = unique_documents(self.retriever.retrieve(question, top_k=self.k_seed))
        extraction = self.kg_extractor.extract_from_documents(question, seeds)
        clues = extraction["clues"]
        expansion_results = []
        for clue in clues:
            clue_text = f"{clue['head']} {clue['relation']} {clue['tail']}"
            expanded_query = f"{question}\n{clue_text}"
            expansion_results.extend(self.retriever.retrieve(expanded_query, top_k=self.k_exp))
        candidates = unique_documents(seeds + expansion_results)
        seed_ids = {document["id"] for document in seeds}
        expanded = [document for document in candidates if document["id"] not in seed_ids]
        return {
            "seed_documents": seeds,
            "expanded_documents": expanded,
            "candidate_documents": candidates,
            "clues": clues,
            "search_requests": 1 + len(clues),
            "llm_calls": extraction["llm_calls"],
            "token_usage": extraction["usage"],
        }
