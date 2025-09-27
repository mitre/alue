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
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

---

## 2. Environment Configuration

ALUE uses [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) to manage configuration.
All variables can be set either via **environment variables** or a `.env` file at the project root.

Start by creating a `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` to provide API keys, model endpoints, and embedding configuration.
(See [Configuration](model-configuration.md) for the complete list of variables.)

---

## 3. Quick Verification

To check that ALUE is installed and functional:

```bash
# run the test suite
pytest tests

# or run a sample task (MCQA)
python scripts/mcqa.py inference \
  -i data/aviation_knowledge_exam/3_1_aviation_test.json \
  -o runs/mcqa \
  -m gpt-4o-mini
```

If successful, predictions will be written to `runs/mcqa_<timestamp>/predictions.json`.

---

## Next Steps

* [Models & Backends](running-models.md) — overview of supported inference and embedding engines.
* [Configuration](model-configuration.md) — complete `.env` variables reference.
* [Tasks](../tasks/rag.md) — detailed guides per task (RAG, MCQA, Summarization, etc.).
