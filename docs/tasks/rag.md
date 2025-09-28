# Retrieval-Augmented Generation (RAG)

RAG combines document retrieval with LLM generation to answer questions based on retrieved context. For each query, the system retrieves the top-k most relevant document chunks from a vector database and prompts the model to generate an answer using only that context.

ALUE provides a complete RAG pipeline: dataset preparation → vector database setup → retrieval → inference → evaluation.

---

## Dataset Format

RAG expects a JSON file with **few-shot examples** and **SQuAD-style questions**:

```json
{
  "examples": [
    {
      "query": "What is the purpose of FAA Order 8040.1C?",
      "context": "This order describes the Federal Aviation Administration's (FAA) authority and assigns responsibility for the development and issuance of Airworthiness Directives (AD) in accordance with applicable statutes and regulations.",
      "answer": "To describe the FAA's authority and assign responsibility for developing and issuing Airworthiness Directives in accordance with applicable statutes and regulations."
    }
  ],
  "data": [
    {
      "title": "FAA Order 8040.1C - Airworthiness Directives",
      "paragraphs": [
        {
          "qas": [
            {
              "id": "0",
              "question": "What is the effective date of FAA Order 8040.1C on Airworthiness Directives?",
              "answers": [
                {
                  "text": "10/03/07",
                  "document_id": ["faa_order_8040_1c"]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

**Key fields:**
- `examples` — Few-shot examples for prompt construction (optional but recommended)
- `data[].paragraphs[].qas[]` — Evaluation questions
- `answers[].text` — Ground-truth answer text
- `answers[].document_id` — **Optional but required for Recall@k**: Chunk IDs in your vector database that contain the answer

> **Important:** If you want to compute **Recall@k**, the `document_id` values must exactly match the chunk IDs stored in your ChromaDB collection.

---

## Quick Start

### Prerequisites

Ensure you have configured your environment (see [Getting Started](../getting-started.md)):
- **Inference backend** (e.g., OpenAI, vLLM, Ollama)
- **LLM Judge** for evaluation (recommended: different from inference model)
- **Embedding provider** for vector database

### Minimal Working Example

Assuming you have an existing ChromaDB database:

```bash
# Run inference + evaluation
python -m scripts.rag both \
  -i data/ASRS_rag/rag_qa.json \
  -o runs/rag \
  -m gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents \
  --top-k 5 \
  --num-examples 3 \
  --llm_judge_model_name gpt-4o-mini \
  --evaluate_retrieval \
  --evaluate_generation
```

This will:
1. Retrieve top-5 chunks for each question
2. Generate answers using the inference model
3. Evaluate retrieval quality (Context Relevancy)
4. Evaluate generation quality (Composite Correctness)

Results are saved to `runs/rag_<timestamp>/`.

---

## Running RAG

The RAG script supports three modes: **inference**, **evaluation**, or **both**.

### Inference Only

Generate answers without evaluation:

```bash
python -m scripts.rag inference \
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

**Output:**
- `predictions.json` — Contains `answer`, `ground_truth_answer`, `predicted_doc_ids`, `question` for each item
- `results.json` — Run parameters and summary

### Evaluation Only

Evaluate existing predictions:

```bash
python -m scripts.rag evaluation \
  -i data/ASRS_rag/rag_qa.json \
  -o runs/rag_eval \
  --predictions_file runs/rag_<timestamp>/predictions.json \
  --llm_judge_model_name gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents \
  --top-k 5 \
  --evaluate_retrieval \
  --evaluate_generation \
  --use_recall_k
```

**Outputs:**
- `rag_evaluation_summary.json` — Overall metrics
- `context_relevancy.json` — Per-chunk relevancy scores (if `--evaluate_retrieval`)
- `doc_retrieval.json` — Recall@k scores (if `--use_recall_k` and ground-truth chunk IDs exist)
- `composite_correctness.json` — Claim-level scoring (if `--evaluate_generation`)

> **Note:** `--use_recall_k` requires `document_id` annotations in your dataset.

### Inference + Evaluation

Run both steps in sequence:

```bash
python -m scripts.rag both \
  -i data/ASRS_rag/rag_qa.json \
  -o runs/rag \
  -m gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents \
  --top-k 5 \
  --num-examples 3 \
  --llm_judge_model_name gpt-4o-mini \
  --evaluate_retrieval \
  --evaluate_generation
```

---

## Templates and Variables

RAG templates are located in `templates/rag/`:

**system.jinja2**
```jinja2
You are an aviation safety analyst. Answer questions using only the provided context.

{% if examples %}
Here are some examples:
{% for example in examples %}
    Question: {{ example.query }}
    Context: {{ example.context }}
    Answer: {{ example.answer }}
{% endfor %}
{% endif %}
```

**user.jinja2**
```jinja2
Question: {{ query }}
Context: {{ context }}
```

### Expected Template Variables

Templates must reference:

| Variable | Description |
|----------|-------------|
| `examples` | Few-shot examples (list of `{query, context, answer}`) |
| `query` | The user's question |
| `context` | Retrieved document chunks (concatenated) |

> **Note:** RAG typically does not require structured generation schemas. Leave `--schema_class` unset unless you define a custom Pydantic schema.

---

## Building a Vector Database

If you don't have an existing ChromaDB, you can build one from PDFs using `alue.rag_utils`.

### From PDF Documents

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

**Parameters:**
- `--document-directory` — Folder containing PDF files
- `--database-path` — Output path for ChromaDB
- `--collection-name` — Name for the collection
- `--partition-strategy` — `hi_res` (higher quality, uses CV model) or `fast` (faster, no CV)
- `--chunk-hard-max` — Maximum chunk size in characters
- `--chunk-soft-max` — Target chunk size (will try to split at sentence boundaries)
- `--overlap-size` — Number of characters to overlap between chunks

### Chunking Strategies

**`hi_res` (recommended for quality):**
- Uses a computer vision model for better document structure detection
- Downloads a detector model on first use (~300MB)
- Better handling of tables, figures, and complex layouts

**`fast` (recommended for speed):**
- No CV model required
- Faster processing
- Good for simple text documents

### Embedding Configuration

The embedding provider is controlled by your `.env` file:

```env
EMBEDDING_ENDPOINT_TYPE=local  # or openai, ollama, hf, openai-compatible
```

See [Configuration Reference](../model-configuration.md) for embedding setup details.

### Chunk IDs for Recall@k

When building a database with `rag_utils`, chunks are assigned stable IDs based on the source document and chunk position. To use Recall@k evaluation:

1. Note the chunk IDs during database creation (they're logged)
2. Add these IDs to your dataset's `answers[].document_id` fields
3. Use `--use_recall_k` flag during evaluation

---

## Evaluation Metrics

### Recall@k (Retrieval Supervision)

**What it measures:** Fraction of ground-truth chunk IDs retrieved among top-k results.

**Requirements:**
- Dataset must include `document_id` annotations
- Chunk IDs must match exactly with ChromaDB collection
- Enable with `--use_recall_k` flag

**Formula:** `Recall@k = (# ground-truth chunks retrieved) / (# total ground-truth chunks)`

Reported as overall average across all questions.

### Context Relevancy (LLM-as-Judge)

**What it measures:** Whether each retrieved chunk is relevant to answering the question.

**How it works:**
- LLM judge assigns {0, 1} to each retrieved chunk
- Per-question score = average across retrieved chunks
- Final score = average across all questions

**Requirements:**
- Requires `--database-path` and `--collection-name` (to fetch chunk content)
- Enable with `--evaluate_retrieval` flag
- Does **not** require ground-truth chunk IDs

### Composite Correctness (LLM-as-Judge)

**What it measures:** Whether the generated answer is factually correct and grounded in retrieved context.

**How it works:**
1. Generated answer is decomposed into atomic **claims**
2. Each claim is checked for:
   - Containment in the reference answer
   - Contradiction with the reference
   - Support from relevant retrieved context
3. If at least one **main** claim correctly answers the question, all claim scores are averaged; otherwise the response scores 0

**Requirements:**
- Enable with `--evaluate_generation` flag
- Automatically uses retrieved context from `predicted_doc_ids`

**Reported metrics:**
- `average_composite_correctness` — Overall score across questions
- Per-claim breakdowns in `composite_correctness.json`

---

## Configuration Notes

### Required Settings

**For inference:**
```env
ALUE_ENDPOINT_TYPE=openai
ALUE_OPENAI_API_KEY=sk-...
EMBEDDING_ENDPOINT_TYPE=local
```

**For evaluation:**
```env
ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai
ALUE_LLM_JUDGE_OPENAI_API_KEY=sk-...
```

> **Recommendation:** Use a different model for the LLM judge than for inference to reduce evaluation bias. Mixing backends is supported.

### When to Use Each Evaluation Metric

| Metric | Use When | Requires |
|--------|----------|----------|
| **Recall@k** | You have ground-truth chunk annotations | `document_id` in dataset |
| **Context Relevancy** | You want to evaluate retrieval quality without annotations | ChromaDB access |
| **Composite Correctness** | You want to evaluate generation quality | LLM judge |

You can enable any combination with `--evaluate_retrieval`, `--evaluate_generation`, and `--use_recall_k` flags.

---

## Using an Existing ChromaDB

If you already have a ChromaDB collection from another source:

1. Ensure your embedding configuration matches the embeddings in the database
2. Provide the database path and collection name to the RAG script
3. For Recall@k, manually add chunk IDs to your dataset's `document_id` fields

Example with external database:

```bash
python -m scripts.rag inference \
  -i data/ASRS_rag/rag_qa.json \
  -o runs/rag \
  -m gpt-4o-mini \
  --database-path /path/to/external/chroma_db \
  --collection-name my_collection \
  --top-k 5
```

---

## See Also

* [Configuration Reference](../model-configuration.md) — Embedding and LLM judge setup
* [Creating Datasets](../creating-datasets.md) — RAG dataset format specification
* [Models & Backends](../running-models.md) — Supported inference engines