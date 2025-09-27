# Retrieval-Augmented Generation (RAG)

RAG combines sparse/semantic retrieval with LLM generation: for each query, the system retrieves top-k document chunks from a vector index and asks the model to answer using that context. ALUE provides a complete pipeline: ingestion → retrieval → prompting → inference → evaluation.

---

## Prerequisites

* **Python**: 3.10 (recommended; repo includes `.python-version`).
* **Environment**: install dependencies via `uv sync` (preferred) or `pip -r requirements.txt`.
* **Settings**: configure `ALUE_*` variables in `.env` (see *Models & Backends* and *Configuration* pages).

  * Inference backend: `openai` | `vllm` | `tgi` | `ollama` (OpenAI-compatible API) | `transformers` (offline).
  * **LLM Judge** (for evaluation): same choices; recommended to use a *different* model than the generator to reduce bias.
* **Vector DB**: ChromaDB (persistent). Either:

  * Point to an existing DB/collection, or
  * Build it from PDFs via `alue.rag_utils` (see below).

---

## Dataset format

RAG expects a JSON file with **few-shot examples** and **SQuAD-style questions**. If you have ground-truth chunk IDs, ALUE can compute Recall@k; otherwise it will evaluate with an LLM judge.

```json
{
  "examples": [
    {"query": "...", "context": "...", "answer": "..."}
  ],
  "data": [
    {
      "title": "...",
      "paragraphs": [
        {
          "qas": [
            {
              "id": "0",
              "question": "...",
              "answers": [
                {"text": "ground truth answer", "document_id": ["chunk_id_optional"]}
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

* `document_id` should match the **chunk IDs** stored in ChromaDB. If omitted, Recall@k is skipped.

---

## Build / use the vector database

### Option A — Use an existing ChromaDB

Provide:

* `--database-path` (folder path)
* `--collection-name` (string)

### Option B — Create ChromaDB from PDFs

```bash
python -m alue.rag_utils \
  --document-directory ./docs_pdfs \
  --database-path ./chroma_db \
  --collection-name documents \
  --output-path ./artifacts \
  --partition-strategy hi_res \
  --chunk-hard-max 1200 \
  --chunk-soft-max 700 \
  --overlap-size 50
```

**Chunking strategy notes**

* `hi_res`: higher-fidelity partitioning via a CV model (downloads a detector on first use).
* `fast`: no CV model; quicker and lighter.
* Resulting chunks are upserted with stable `chunk_id`s suitable for Recall@k.

Embedding provider is selected via `EMBEDDING_ENDPOINT_TYPE` (`openai`, `ollama`, `hf`, `local`, `openai-compatible`). If you omit `--embedding-model`, ALUE chooses a sensible default per provider.

---

## Templates and schemas

Templates live under `templates/rag/`:

* `system.jinja2` – receives `examples` (few-shot).
* `user.jinja2` – receives `query` and aggregated `context`.

**Expected template variables**

```python
message = build_messages(
    task_type=args.task_type,                  # "rag" by default
    system_kwargs={"examples": examples},      # list of {input/context/output} triples
    user_kwargs={"query": question, "context": context}
)
```

> RAG typically does **not** require structured generation; leave `--schema_class` unset unless you define your own Pydantic schema.

---

## Running RAG

### Inference only

```bash
python scripts/rag.py inference \
  -i data/ASRS_rag/rag_qa.json \
  -o runs/rag \
  -m gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents \
  --top-k 5 \
  --num-examples 3 \
  --task_type rag \
  --temperature 0.1 \
  --max_tokens 150
```

Outputs in a timestamped folder, e.g. `runs/rag_YYYYMMDD_HHMMSS/`:

* `predictions.json` (includes `answer`, `ground_truth_answer`, `predicted_doc_ids`, `question`)
* `results.json` (summary + params)

### Evaluation only

```bash
python scripts/rag.py evaluation \
  -i data/ASRS_rag/rag_qa.json \
  -o runs/rag_eval \
  --predictions_file runs/rag_YYYYMMDD_HHMMSS/predictions.json \
  --llm_judge_model_name gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents \
  --top-k 5 \
  --evaluate_retrieval \
  --evaluate_generation \
  --use_recall_k    # include only if ground-truth chunk IDs exist
```

Artifacts:

* `rag_evaluation_summary.json`
* `context_relevancy.json` (per-chunk relevancy) — if DB/collection provided
* `doc_retrieval.json` (Recall@k) — if `--use_recall_k`
* `composite_correctness.json` (claim-level scoring)

### Inference + Evaluation

```bash
python -m scripts.rag both \
  -i data/ASRS_rag/rag_qa.json \
  -o runs/rag \
  -m gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents \
  --top-k 5 \
  --llm_judge_model_name gpt-4o-mini \
  --evaluate_retrieval \
  --evaluate_generation
```

> **LLM Judge note**: You may point judge settings to the same backend as inference, but a different model is recommended to mitigate bias.

---

## Metrics (RAG)

**Recall@k**
Fraction of ground-truth chunk IDs retrieved among top-k. Requires annotated `document_id` labels matching your ChromaDB chunk IDs. Reported as overall average across questions.

**Context Relevancy (LLM-as-Judge)**
For each retrieved chunk, an LLM judge assigns {0,1} indicating whether the chunk is relevant to answering the question. The per-question score is the average across retrieved chunks; the final score is the average across questions. Does **not** require ground-truth chunk IDs (uses DB lookups from `predicted_doc_ids`).

**Composite Correctness (LLM-as-Judge)**
The generated answer is decomposed into atomic **claims** by an LLM. Each claim is checked for:

1. containment in the reference answer;
2. contradiction with the reference;
3. support from any **relevant** retrieved context.
   If at least one **main** claim correctly answers the question, the correctness scores of all claims are averaged; otherwise the response scores 0. Reports an overall average.

---

## Troubleshooting

* **“No module/model downloaded” on `hi_res` partitioning**: switch to `--partition-strategy fast` to avoid CV model downloads, or allow the first-run download.
* **Recall@k missing**: ensure your dataset includes `document_id` arrays and they match **exact** chunk IDs in ChromaDB.
* **Empty or low Context Relevancy**: verify `predicted_doc_ids` exist in `predictions.json` and point to the correct `--database-path`/`--collection-name`.
* **LLM Judge latency/cost**: reduce `--num_questions`, or run `evaluation` with only `--evaluate_retrieval` first.

---

## Minimal working example

1. Build a small ChromaDB from PDFs:

```bash
python -m alue.rag_utils \
  --document-directory ./tests/resources \
  --database-path ./chroma_db \
  --collection-name documents \
  --partition-strategy fast
```

2. Run inference:

```bash
python -m scripts.rag inference \
  -i data/dummy_rag/rag_qa.json \
  -o runs/rag \
  -m gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents
```

3. Evaluate (context relevancy + composite correctness):

```bash
python -m scripts.rag evaluation \
  -i data/dummy_rag/rag_qa.json \
  -o runs/rag_eval \
  --predictions_file runs/rag_*/predictions.json \
  --llm_judge_model_name gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents
```


