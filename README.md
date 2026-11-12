# CGEP-RAG

Implementation of *CGEP-RAG: A Dual-Process Framework for Multi-Hop Retrieval-Augmented Generation*. The pipeline extracts relational clues from seed passages, expands BM25 retrieval, and selects evidence by ranking passage clusters.

## Setup

Use Python 3.10+ and an Elasticsearch index with BM25 on the `text` field. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

Each indexed passage needs `text`. Use a stable `doc_id` or Elasticsearch `_id`; `title` is optional. Question files can be JSON arrays, JSON objects with a `questions` array, or JSONL. Each question needs an `id` (or `_id`) and `question`.

## Run inference

Set `OPENAI_API_KEY` in your environment, then run:

```bash
bash run.sh \
  --questions /path/to/questions.json \
  --bm25-url http://localhost:9200 \
  --index your_corpus_index \
  --output outputs/predictions.jsonl
```

Use `--question "Your question"` for a single input and `bash run.sh --help` for the full list of options. Choose a new output path for each run. The JSONL output includes answers, selected passages, and request statistics.

The default configuration is in [configs/paper_core.json](configs/paper_core.json). It uses 15 seed passages, 5 passages per clue, cluster sizes of 10 and 5, and a final budget of 20 passages.

## SMHQA

The SMHQA annotations, source metadata, and corpus reconstruction scripts are planned for the next repository update.
