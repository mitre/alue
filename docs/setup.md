# Setup

This document describes how to install and configure ALUE, including the supported inference engines, embedding providers, and database backends.  

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-org>/ALUE.git
cd ALUE
pip install -e .
````

Dependencies are managed via `pyproject.toml` and resolved automatically during installation.

---

## Inference Engines

ALUE supports multiple inference backends. Users should select the backend most suited to their infrastructure.

| Backend        | Description                                                          | Example Models                             |
| -------------- | -------------------------------------------------------------------- | ------------------------------------------ |
| `openai`       | OpenAI ChatCompletions API                                           | GPT-4o, GPT-4o-mini                        |
| `vllm`         | Local inference using [vLLM](https://github.com/vllm-project/vllm)   | LLaMA-3, Mistral                           |
| `tgi`          | Hugging Face Text Generation Inference (TGI) server                  | Falcon, BLOOM                              |
| `ollama`       | [Ollama](https://ollama.ai/) runtime for local lightweight inference | LLaMA-2, Mistral-7B                        |
| `transformers` | Direct inference using Hugging Face `transformers` library           | Any model available through `transformers` |

Select an engine via configuration or command-line flags (see [Tasks](tasks/index.md) for details).

---

## Embedding Providers

For retrieval-based tasks such as RAG, ALUE integrates with several embedding models:

| Provider            | Description                                   | Example Models                                     |
| ------------------- | --------------------------------------------- | -------------------------------------------------- |
| `openai`            | Embeddings via OpenAI API                     | `text-embedding-3-small`, `text-embedding-3-large` |
| `hf`                | Hugging Face embedding models                 | `sentence-transformers/all-MiniLM-L6-v2`           |
| `ollama`            | Embeddings generated through Ollama           | `nomic-embed-text`                                 |
| `local`             | Locally hosted embedding model (custom path)  | Any fine-tuned embedding model                     |
| `openai-compatible` | Any service exposing an OpenAI-compatible API | Azure OpenAI, custom deployments                   |

The embedding provider must be configured consistently with the retrieval database (ChromaDB).

---

## Database Backend

ALUE currently supports [ChromaDB](https://www.trychroma.com/) as its vector database for retrieval.

* **Database Path**: Specify the path to the persistent database directory (default: `./chroma_db`).
* **Collection Name**: Specify the collection used for storing and retrieving document embeddings (default: `documents`).

---

## Verification

After installation, verify the environment by running:

```bash
python -m alue.run_task --help
```

This should display available tasks, arguments, and configuration options.

---

## Next Steps

* Review [Tasks](tasks/index.md) for task-specific configurations.
* Configure prompt templates and schemas as described in each task section.
* Ensure consistency between the inference engine, embedding provider, and retrieval database.

```

This version sets up ALUE as a **modular, research-grade framework**, and positions the rest of the docs (task-specific pages) as natural extensions.  

Want me to draft **`tasks/index.md`** next (overview table of tasks + configs), or should we go straight into **RAG** since it’s the most complete task so far?
```
