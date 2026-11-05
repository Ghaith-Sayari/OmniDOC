# OmniDoc

Enterprise-ready, multi-format Retrieval-Augmented Generation (RAG) for
turning document collections into grounded, citation-aware answers.

OmniDoc combines dense vector retrieval, BM25 keyword retrieval, reciprocal
rank fusion, cross-encoder re-ranking, and LLM generation with citation
verification. It can be used through a command-line interface or a FastAPI
service.

## Features

- Parse PDF, DOCX, XLSX, CSV, Markdown, HTML, and PPTX documents.
- Create hierarchical chunks for retrieval.
- Combine Qdrant dense retrieval with BM25 sparse retrieval.
- Fuse and re-rank results before generation.
- Generate answers with Claude or GPT-4o backends.
- Verify answer citations against retrieved context.
- Evaluate question-and-answer samples with Ragas metrics.
- Run locally or with Docker Compose.

## Architecture

```text
Documents
   |
   v
Parser -> Hierarchical Chunker -> Embeddings -> Qdrant + BM25
                                                |
Question -> Dense + Sparse Retrieval -> RRF Fusion -> Re-ranker
                                                |
                                LLM Answer + Citation Verification
```

## Requirements

- Python 3.11+
- Qdrant running locally or remotely
- An embedding backend:
  - `bge` for local Hugging Face embeddings, or
  - `openai` for OpenAI embeddings
- An LLM API key for `claude` or `gpt4o`
- A Cohere API key when using the default `cohere` re-ranker

## Installation

```bash
git clone https://github.com/Ghaith-Sayari/OmniDOC.git
cd OmniDOC

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For scanned-PDF OCR support, install the optional dependencies:

```bash
pip install -e ".[ocr]"
```

OCR also requires the Tesseract system package to be installed and available
on `PATH`.

## Configuration

Copy the example environment file and set the providers you want to use:

```bash
cp .env.example .env
```

Important settings include:

| Variable | Default | Description |
| --- | --- | --- |
| `OMNIDOC_QDRANT_HOST` | `localhost` | Qdrant hostname |
| `OMNIDOC_QDRANT_PORT` | `6333` | Qdrant REST port |
| `OMNIDOC_EMBEDDING_BACKEND` | `bge` | `bge` or `openai` |
| `OMNIDOC_RERANKER_BACKEND` | `cohere` | `cohere` or `bge` |
| `OMNIDOC_LLM_BACKEND` | `claude` | `claude` or `gpt4o` |
| `OMNIDOC_RETRIEVAL_TOP_K` | `100` | Candidates retrieved per retriever |
| `OMNIDOC_RERANKER_TOP_N` | `5` | Context chunks sent to the LLM |

Set the provider keys used by your configuration:

```dotenv
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
COHERE_API_KEY=...
```

## Start Qdrant

To run only Qdrant locally:

```bash
docker compose -f docker/docker-compose.yml up -d qdrant
```

To run both Qdrant and the OmniDoc API:

```bash
docker compose -f docker/docker-compose.yml up --build
```

The API is then available at `http://localhost:8000`, with interactive
documentation at `http://localhost:8000/docs`.

## CLI usage

### Start the API

```bash
omnidoc serve --host 0.0.0.0 --port 8000
```

Use `--reload` during development:

```bash
omnidoc serve --reload
```

### Ingest documents

Ingest one file:

```bash
omnidoc ingest data/raw/policy.pdf
```

Ingest a directory recursively:

```bash
omnidoc ingest data/raw/
```

### Query the indexed corpus

```bash
omnidoc query "What is the document retention policy?"
```

Adjust retrieval sizes when needed:

```bash
omnidoc query "What is the refund policy?" --top-k 50 --top-n 5
```

### Evaluate answers

The input file should be a JSON array containing records with `question`,
`answer`, `contexts`, and optionally `ground_truth`.

```bash
omnidoc evaluate data/evaluation/samples.json
```

Reports are written to `data/evaluation/` by default.

Equivalent helper scripts are available under `scripts/`:

```bash
python scripts/ingest_sample.py --path data/raw/
python scripts/query_corpus.py "What is the refund policy?"
python scripts/run_evaluation.py data/evaluation/samples.json
```

## API

### Health check

```bash
curl http://localhost:8000/health
```

### Upload a document

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@data/raw/policy.pdf"
```

### Ingest a directory

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{"directory":"data/raw","recursive":true}'
```

### Check ingestion status

```bash
curl http://localhost:8000/ingest/status
```

### Query the corpus

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the document retention policy?",
    "top_k": 100,
    "top_n": 5
  }'
```

### Run an evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"samples":[
    {
      "question":"What is RAG?",
      "answer":"RAG is Retrieval-Augmented Generation.",
      "contexts":["RAG combines document retrieval with language model generation."],
      "ground_truth":"Retrieval-Augmented Generation"
    }
  ]}'
```

## Development

Run the test suite:

```bash
pytest
```

Run linting and type checks:

```bash
ruff check .
mypy omnidoc
```

Runtime indexes, metadata, evaluation reports, and uploaded files are stored
under `data/` and are intentionally excluded from version control.

## License

OmniDoc is configured as an MIT-licensed project in `pyproject.toml`.
