"""Lexical retrieval used in both stages of clue-guided expansion."""

from typing import Dict, List

if __package__:
    from .bm25_search import BM25Retriever
else:
    from bm25_search import BM25Retriever


class PassageRetriever:
    def __init__(self, bm25_retriever: BM25Retriever):
        self.bm25_retriever = bm25_retriever

    def retrieve(self, query: str, top_k: int = 15) -> List[Dict]:
        return self.bm25_retriever.search_direct(query, top_k)
