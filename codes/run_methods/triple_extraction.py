"""Query-conditioned relational clue extraction from the fixed seed set."""

import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Tuple

from openai import OpenAI

try:
    from .prompts import build_triple_prompt
except ImportError:
    from prompts import build_triple_prompt


PLACEHOLDERS = {"none", "n/a", "unknown", "null"}
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def normalize_clue_field(value: str) -> str:
    """Normalize Unicode, case, and whitespace for clue comparisons."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def iter_json_objects(text: str) -> Iterator[Dict]:
    """Read consecutive JSON objects, including multiline and fenced outputs."""
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(text):
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = end
        if isinstance(value, dict):
            yield value


def valid_clue(record: Dict) -> Dict | None:
    """Reject incomplete clues, placeholder triples, and self-loops."""
    fields = ("head", "relation", "tail", "evidence")
    if any(not isinstance(record.get(field), str) for field in fields):
        return None
    clue = {field: record[field].strip() for field in fields}
    if not all(clue.values()):
        return None
    normalized = tuple(normalize_clue_field(clue[field]) for field in fields[:3])
    if any(value in PLACEHOLDERS for value in normalized):
        return None
    if normalized[0] == normalized[2]:
        return None
    return clue


def response_usage(response) -> Dict[str, int]:
    """Read token counts from the API response."""
    usage = response.usage
    if usage is None or any(getattr(usage, field, None) is None for field in TOKEN_FIELDS):
        raise RuntimeError("The model response did not include complete token usage.")
    return {field: int(getattr(usage, field)) for field in TOKEN_FIELDS}


class KGExtractor:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0,
        api_base: str | None = None,
        max_tokens: int = 512,
        top_p: float = 1.0,
        max_workers: int = 15,
    ):
        if max_tokens < 1 or max_workers < 1:
            raise ValueError("max_tokens and max_workers must be positive.")
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.max_workers = max_workers

    def _extract_one(self, question: str, document: Dict) -> Tuple[List[Dict], Dict[str, int]]:
        doc_id = document.get("id", document.get("doc_id"))
        if doc_id is None or not str(doc_id).strip():
            raise ValueError("Each seed passage must have a stable document ID.")
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Seed passage {doc_id!r} has no text.")
        title = str(document.get("title", "")).strip()
        reference = f"{title}\n{text}" if title else text
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": build_triple_prompt(question, reference)}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"The extraction response for passage {doc_id!r} was empty.")
        clues = []
        for record in iter_json_objects(content):
            clue = valid_clue(record)
            if clue is not None:
                clue["source_doc_id"] = str(doc_id)
                clues.append(clue)
        return clues, response_usage(response)

    def extract_from_documents(self, question: str, documents: List[Dict]) -> Dict:
        usage = {field: 0 for field in TOKEN_FIELDS}
        if not documents:
            return {"clues": [], "usage": usage, "llm_calls": 0}
        # Preserve seed order so duplicate triples keep the first source.
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(documents))) as executor:
            results = list(executor.map(lambda doc: self._extract_one(question, doc), documents))
        clues = []
        seen = set()
        for document_clues, document_usage in results:
            for field in TOKEN_FIELDS:
                usage[field] += document_usage[field]
            for clue in document_clues:
                key = tuple(normalize_clue_field(clue[field]) for field in ("head", "relation", "tail"))
                if key not in seen:
                    seen.add(key)
                    clues.append(clue)
        return {"clues": clues, "usage": usage, "llm_calls": len(documents)}
