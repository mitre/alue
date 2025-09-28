# Getting Started

This section describes how to install dependencies, configure the environment, and verify that ALUE is set up correctly.  

ALUE has been tested primarily on **Python 3.10** (with partial testing on 3.11).  
Other versions may work, but are not officially supported.

---

## 1. Installation

ALUE supports two installation methods.  
We recommend [**uv**](https://docs.astral.sh/uv/) for reproducibility and speed.

### Using `uv` (preferred)

```bash
# install dependencies into a managed virtual environment
uv sync
```

This creates a `.venv` directory automatically.
No manual `venv` creation is needed.

### Using `pip` (fallback)

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

---

## 2. Environment Configuration

ALUE uses a `.env` file for configuration. Create one from the example:

```bash
cp .env.example .env
```

Then edit `.env` to add your API keys and endpoints. At minimum, you'll need to configure:

**For OpenAI (simplest):**
```env
ALUE_ENDPOINT_TYPE=openai
ALUE_OPENAI_API_KEY=sk-...
```

**For local Ollama:**
```env
ALUE_ENDPOINT_TYPE=ollama
ALUE_ENDPOINT_URL=http://localhost:11434
```

**For vLLM:**
```env
ALUE_ENDPOINT_TYPE=vllm
ALUE_ENDPOINT_URL=http://localhost:8000/v1
HF_TOKEN=hf_...
```

> **Note:** Different tasks require different configuration. RAG and Summarization tasks require additional LLM Judge and embedding settings. See the [Configuration Reference](model-configuration.md) for complete details on all available settings and backend-specific requirements.

---

## 3. Quick Verification

To check that ALUE is installed and functional:

```bash
# run the test suite
pytest tests

# or run a sample task (MCQA)
python -m scripts.mcqa inference \
  -i data/aviation_knowledge_exam/3_1_aviation_test.json \
  -o runs/mcqa \
  -m gpt-4o-mini \
  --task_type aviation_exam \
  --num_examples 3
```

If successful, predictions will be written to `runs/mcqa_<timestamp>/predictions.json`.

---

## Next Steps

* [Configuration Reference](model-configuration.md) — Complete `.env` variables reference with troubleshooting
* [Models & Backends](running-models.md) — Overview of supported inference and embedding engines
* [Tasks](tasks/index.md) — Detailed guides per task:
  * [MCQA](tasks/mcqa.md) — Multiple choice question answering
  * [RAG](tasks/rag.md) — Retrieval-augmented generation
  * [Summarization](tasks/summarization.md) — Narrative summarization
  * [Extractive QA](tasks/extractive-qa.md) — Span extraction
* [Creating Datasets](creating-datasets.md) — Dataset format specifications for each task type
```