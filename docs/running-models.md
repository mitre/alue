# Models & Backends

ALUE is designed to be **backend-agnostic**, supporting multiple inference engines and embedding providers.  
All configuration is controlled through environment variables (see [Configuration](model-configuration.md)).

---

## Inference Engines

ALUE supports the following inference engines for running LLMs:

| Engine          | Alias (`ALUE_ENDPOINT_TYPE`) | Connection Mode                             | Description |
|-----------------|-------------------------------|---------------------------------------------|-------------|
| **OpenAI**      | `openai`                     | API                                         | Direct connection to the OpenAI Chat Completions API. Requires `ALUE_OPENAI_API_KEY`. |
| **vLLM (online)** | `vllm`                     | API (OpenAI-compatible)                     | Connects to a running [vLLM](https://vllm.ai/) server via REST using the OpenAI-compatible API. |
| **TGI**         | `tgi`                        | API (OpenAI-compatible)                     | Connects to [Hugging Face Text Generation Inference (TGI)](https://huggingface.co/docs/text-generation-inference) endpoints. |
| **Ollama**      | `ollama`                     | API (OpenAI-compatible)                     | Connects to [Ollama](https://ollama.ai/) locally via its OpenAI-compatible API. |
| **vLLM (offline)** | `vllm-offline`            | Local (no API)                              | Runs vLLM entirely offline inside the Python process (no server needed). |
| **Transformers**| `transformers`               | Local (no API)                              | Runs models directly using Hugging Face `transformers`. Requires local model weights. |

**Key distinction:**  
- **API (OpenAI-compatible)**: vLLM, TGI, and Ollama integrate with ALUE by exposing an **OpenAI-style endpoint**.  
- **Offline/local**: vLLM in offline mode (`vllm-offline`) and Hugging Face `transformers` run fully within your Python process — no external endpoint required.

Example setting in `.env`:

```bash
ALUE_ENDPOINT_TYPE=openai
ALUE_OPENAI_API_KEY=sk-xxxx
```

---

## LLM Judge Engines

Some tasks (e.g., **RAG** and **Summarization**) require an additional LLM as a **judge** for evaluation.
Configuration mirrors the main inference engine, with the following variables:

* `ALUE_LLM_JUDGE_ENDPOINT_TYPE`
* `ALUE_LLM_JUDGE_ENDPOINT_URL`
* `ALUE_LLM_JUDGE_OPENAI_API_KEY`

By default, the judge can be the same engine as the inference model, but **this is not recommended** (to reduce evaluation bias).

---

## Embedding Engines

For retrieval-based tasks (e.g., RAG), ALUE requires embeddings.
Supported embedding backends:

| Engine                | Alias (`EMBEDDING_ENDPOINT_TYPE`) | Notes                                                                          |
| --------------------- | --------------------------------- | ------------------------------------------------------------------------------ |
| **OpenAI**            | `openai`                          | Embeddings via OpenAI API (requires `EMBEDDING_API_KEY`).                      |
| **Ollama**            | `ollama`                          | Uses local Ollama models for embeddings.                                       |
| **Hugging Face**      | `hf`                              | Uses Hugging Face models (e.g., `sentence-transformers`). Requires `HF_TOKEN`. |
| **Local**             | `local` (default)                 | Uses bundled local embedding models.                                           |
| **OpenAI-compatible** | `openai-compatible`               | Custom OpenAI-style embedding endpoints.                                       |

Example setting in `.env`:

```bash
EMBEDDING_ENDPOINT_TYPE=hf
HF_TOKEN=hf_xxxx
```

---