"""BM25 passage retrieval from an existing Elasticsearch corpus index."""

from typing import Dict, List, Optional

from elasticsearch import Elasticsearch


class BM25Retriever:
    """Use the same passage index for seed and per-clue retrieval."""

    def __init__(
        self,
        index_name: str,
        url: str = "http://localhost:9200",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ca_certs: Optional[str] = None,
    ):
        if not index_name:
            raise ValueError("A corpus-specific BM25 index name is required.")
        if bool(username) != bool(password):
            raise ValueError("Set both BM25_USERNAME and BM25_PASSWORD.")
        if api_key and username:
            raise ValueError("Use either BM25_API_KEY or username/password authentication.")
        self.index_name = index_name
        options = {"request_timeout": timeout}
        if api_key:
            options["api_key"] = api_key
        elif username:
            options["basic_auth"] = (username, password)
        if ca_certs:
            options["ca_certs"] = ca_certs
        self.es = Elasticsearch(url, **options)

    def check_index(self) -> None:
        if not self.es.indices.exists(index=self.index_name):
            raise ValueError(f"BM25 index does not exist: {self.index_name}")

    def search_direct(self, query: str, top_k: int) -> List[Dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The retrieval query must be a nonempty string.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        response = self.es.search(
            index=self.index_name,
            body={
                "query": {"match": {"text": {"query": query, "operator": "or"}}},
                "size": top_k,
                "_source": ["doc_id", "title", "text"],
            },
        )
        documents = []
        seen_ids = set()
        for hit in response["hits"]["hits"]:
            source = hit.get("_source", {})
            raw_id = source.get("doc_id")
            if raw_id is None:
                raw_id = hit.get("_id")
            text = source.get("text")
            if raw_id is None or not str(raw_id).strip():
                raise ValueError("Every indexed passage must have a stable document ID.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Indexed passage {raw_id!r} has no text.")
            doc_id = str(raw_id)
            if doc_id in seen_ids:
                raise ValueError(f"Duplicate doc_id in the BM25 index: {doc_id}")
            seen_ids.add(doc_id)
            documents.append(
                {
                    "id": doc_id,
                    "title": str(source.get("title") or ""),
                    "text": text,
                    "bm25_score": float(hit["_score"] or 0.0),
                }
            )
        return documents

    def close(self) -> None:
        self.es.close()
