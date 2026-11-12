"""Run CGEP-RAG on a question or a JSON/JSONL question file."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

if __package__:
    from .answer_synthesis import AnswerGenerator
    from .bm25_search import BM25Retriever
    from .configuration import CoreConfig, PAPER_MODELS
    from .evidence_clustering import DocumentClusterer
    from .initial_retrieval import PassageRetriever
    from .semantic_encoder import SemanticEncoder
    from .triple_extraction import KGExtractor
    from .triple_guided_expansion import EntityExpansionRetriever
else:
    from answer_synthesis import AnswerGenerator
    from bm25_search import BM25Retriever
    from configuration import CoreConfig, PAPER_MODELS
    from evidence_clustering import DocumentClusterer
    from initial_retrieval import PassageRetriever
    from semantic_encoder import SemanticEncoder
    from triple_extraction import KGExtractor
    from triple_guided_expansion import EntityExpansionRetriever


class CGEPRAG:
    def __init__(
        self,
        expansion: EntityExpansionRetriever,
        clusterer: DocumentClusterer,
        generator: AnswerGenerator,
        config: CoreConfig,
    ):
        self.expansion = expansion
        self.clusterer = clusterer
        self.generator = generator
        self.config = config

    def answer(self, question: str, question_id: str) -> Dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string.")
        expanded = self.expansion.retrieve(question)
        seed_ids = [str(doc["id"]) for doc in expanded["seed_documents"]]
        purified = self.clusterer.purify(
            question=question,
            candidate_documents=expanded["candidate_documents"],
            seed_document_ids=set(seed_ids),
            clues=expanded["clues"],
            budget=self.config.evidence_budget,
        )
        selected = purified["selected_documents"]
        evidence_text = "\n\n".join(
            f"Document {i}: {doc.get('title', '')}\n{doc['text']}"
            for i, doc in enumerate(selected, start=1)
        )
        answer = self.generator.generate_answer(evidence_text, question)
        extraction_usage = expanded["token_usage"]
        answer_usage = {
            key: answer[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        online_usage = {
            key: extraction_usage[key] + answer_usage[key]
            for key in answer_usage
        }
        return {
            "question_id": str(question_id),
            "question": question,
            "answer": answer["answer"],
            "model": answer["model"],
            "configuration": asdict(self.config),
            "seed_document_ids": seed_ids,
            "clues": expanded["clues"],
            "candidate_document_ids": [
                str(doc["id"]) for doc in expanded["candidate_documents"]
            ],
            "clusters": purified["clusters"],
            "selected_document_ids": [str(doc["id"]) for doc in selected],
            "selected_documents": selected,
            "search_requests": expanded["search_requests"],
            "llm_calls": expanded["llm_calls"] + 1,
            "token_usage": {
                "extraction": extraction_usage,
                "answer": answer_usage,
                "online": online_usage,
            },
            "full_answer_response": answer["full_response"],
        }


def load_questions(path: Path, limit: int = None) -> List[Dict]:
    """Read question fields only; retrieve all evidence from the corpus index."""
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            rows = json.load(handle)
    if isinstance(rows, dict):
        rows = rows.get("questions")
    if not isinstance(rows, list):
        raise ValueError("Question input must be a list or an object containing questions.")
    questions = []
    seen_ids = set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Question record {position} must be an object.")
        question = row.get("question")
        raw_id = row.get("id")
        if raw_id is None:
            raw_id = row.get("_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Question record {position} has no question text.")
        if raw_id is None or not str(raw_id).strip():
            raise ValueError(f"Question record {position} requires id or _id.")
        question_id = str(raw_id)
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question ID: {question_id}")
        seen_ids.add(question_id)
        questions.append({"id": question_id, "question": question})
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive.")
        questions = questions[:limit]
    if not questions:
        raise ValueError("No questions were supplied.")
    return questions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[2] / "configs" / "paper_core.json",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="One question for inference.")
    source.add_argument("--questions", type=Path, help="A JSON or JSONL question file.")
    parser.add_argument("--limit", type=int, help="Read the first N records; no random sampling.")
    parser.add_argument("--model", choices=PAPER_MODELS)
    parser.add_argument("--budget", type=int, help="Final source-passage budget.")
    parser.add_argument("--device", default=os.getenv("CGEPRAG_DEVICE"))
    parser.add_argument("--model-cache", default=os.getenv("CGEPRAG_MODEL_CACHE"))
    parser.add_argument("--index", default=os.getenv("BM25_ELASTICSEARCH_INDEX"))
    parser.add_argument(
        "--bm25-url", default=os.getenv("BM25_ELASTICSEARCH_URL", "http://localhost:9200"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/cgep_rag_results.jsonl"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = CoreConfig.from_json(args.config)
    if args.model:
        config.model = args.model
    if args.budget is not None:
        config.evidence_budget = args.budget
    config.validate()
    if not args.index:
        parser.error("Set --index or BM25_ELASTICSEARCH_INDEX to the benchmark corpus index.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        parser.error("Set OPENAI_API_KEY before running inference.")
    if args.question is not None:
        if not args.question.strip():
            parser.error("--question must not be empty.")
        if args.limit is not None:
            parser.error("--limit applies only to --questions.")
        question_id = hashlib.sha256(args.question.encode("utf-8")).hexdigest()
        questions = [{"id": question_id, "question": args.question}]
    else:
        questions = load_questions(args.questions, args.limit)
    if args.output.exists():
        parser.error(f"Output already exists; choose a new path: {args.output}")
    api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None
    bm25 = BM25Retriever(
        index_name=args.index,
        url=args.bm25_url,
        api_key=os.getenv("BM25_API_KEY"),
        username=os.getenv("BM25_USERNAME"),
        password=os.getenv("BM25_PASSWORD"),
        ca_certs=os.getenv("BM25_CA_CERTS"),
    )
    try:
        bm25.check_index()
        retriever = PassageRetriever(bm25)
        extractor = KGExtractor(
            api_key=api_key, model=config.model, temperature=config.temperature,
            api_base=api_base, max_tokens=config.extraction_max_tokens,
            top_p=config.top_p, max_workers=config.extraction_workers,
        )
        expansion = EntityExpansionRetriever(
            retriever, extractor, k_seed=config.k_seed, k_exp=config.k_exp,
        )
        encoder = SemanticEncoder(
            model_name=config.embedding_model, device=args.device,
            cache_dir=args.model_cache, batch_size=config.embedding_batch_size,
        )
        clusterer = DocumentClusterer(
            encoder, initial_size=config.cluster_size, min_size=config.min_cluster_size,
        )
        generator = AnswerGenerator(
            api_key=api_key, model=config.model, temperature=config.temperature,
            max_tokens=config.answer_max_tokens, api_base=api_base, top_p=config.top_p,
        )
        pipeline = CGEPRAG(expansion, clusterer, generator, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            for item in questions:
                result = pipeline.answer(item["question"], item["id"])
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                print(f"{item['id']}: {result['answer']}")
    finally:
        bm25.close()


if __name__ == "__main__":
    main()
