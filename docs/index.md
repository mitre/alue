# ALUE: Aerospace Language Understanding and Evaluation

The **Aerospace Language Understanding and Evaluation (ALUE)** framework is an open-source system for benchmarking and evaluating large language models (LLMs) on tasks relevant to aerospace, safety-critical domains, and general language understanding.  

ALUE provides:
- A consistent interface for multiple task types, including **multiple-choice question answering (MCQA)**, **summarization**, and **retrieval-augmented generation (RAG)**.  
- Evaluation methods that combine **traditional metrics** (e.g., recall@k, token-level F1) with **LLM-based evaluation metrics** (e.g., context relevancy, composite correctness, claim decomposition).  
- Extensible templates and configuration utilities to support additional domains and tasks.

---

## Key Features

- **Backend Flexibility**  
  ALUE integrates with a variety of inference and embedding providers:  
  - *Inference backends*: `openai`, `vllm`, `tgi`, `ollama`, `transformers`  
  - *Embedding providers*: `openai`, `ollama`, `hf`, `local`, `openai-compatible`

- **Evaluation Beyond Token Overlap**  
  Incorporates LLM-judge metrics that provide a more nuanced and robust assessment of correctness and factual grounding, particularly for long-form and generative responses.

- **Structured Prompting**  
  All tasks use message templates with defined variables. This enables transparent, reproducible, and customizable prompt construction.

- **Task-Specific Evaluation**  
  Each task is accompanied by its own evaluation methodology and metrics tailored to the problem type.

---

## Documentation Structure

- [Setup](getting-started.md): Installation and configuration of ALUE, including inference and embedding backends.  
- [Tasks](tasks/index.md): Task-specific documentation and examples:  
  - [MCQA](tasks/mcqa.md)  
  - [Summarization](tasks/summarization.md)  
  - [RAG](tasks/rag.md)  
- [Contributing](contributing.md): Guidelines for extensions and contributions.  
- [API Reference](api.md): Generated reference documentation for ALUE modules.  

---

Here's the complete, improved Quickstart section for your `index.md`:

```markdown
## Quickstart

### 1. Install dependencies

```bash
# Recommended: using uv
uv sync

# Alternative: using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env to add your API keys and endpoints
# At minimum, set:
# ALUE_ENDPOINT_TYPE=openai
# ALUE_OPENAI_API_KEY=sk-...
```

### 3. Run a simple example

```bash
# Example: Multiple Choice QA with OpenAI
python -m scripts.mcqa inference \
  -i data/aviation_knowledge_exam/3_1_aviation_test.json \
  -o runs/mcqa \
  -m gpt-4o-mini \
  --task_type aviation_exam \
  --num_examples 3

# Results will be saved to runs/mcqa_<timestamp>/predictions.json
```

### 4. Verify installation

```bash
# Run the test suite
pytest tests
```

### Next Steps

* [Getting Started](getting-started.md) — Detailed installation and configuration
* [Tasks](tasks/index.md) — Task-specific guides:
  * [MCQA](tasks/mcqa.md) — Multiple choice question answering
  * [RAG](tasks/rag.md) — Retrieval-augmented generation
  * [Summarization](tasks/summarization.md) — Narrative summarization
  * [Extractive QA](tasks/extractive-qa.md) — Span extraction
* [Models & Backends](running-models.md) — Supported inference engines and embedding providers
* [Configuration](model-configuration.md) — Complete environment variables reference



# Citation

If you use ALUE in academic or applied work, please cite:

@inproceedings{alue2025,
  title        = {ALUE: Aerospace Language Understanding and Evaluation},
  author       = {…},
  booktitle    = {AIAA Scitech Forum},
  year         = {2025},
  doi          = {10.2514/6.2025-3247}
}