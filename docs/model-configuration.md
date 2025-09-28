# Configuration Reference

ALUE uses [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) for configuration management. All settings can be provided via:
- Environment variables
- A `.env` file at the project root (recommended)

This page provides the complete reference for all configuration variables.

---

## Configuration Method

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then edit `.env` with your settings. The application will automatically load these values on startup.

---

## Complete Variable Reference

### Inference Backend Configuration

These settings control the primary LLM used for task inference (MCQA, RAG, Summarization, etc.).

| Variable | Type | Options | Description |
|----------|------|---------|-------------|
| `ALUE_ENDPOINT_TYPE` | **Required** | `openai`, `vllm`, `tgi`, `ollama`, `transformers`, `vllm-offline` | Backend type for inference |
| `ALUE_OPENAI_API_KEY` | Conditional | API key string | Required when `ALUE_ENDPOINT_TYPE=openai` |
| `ALUE_ENDPOINT_URL` | Conditional | URL | Required for `vllm`, `tgi`, `ollama`, or any OpenAI-compatible endpoint |
| `HF_TOKEN` | Conditional | Token string | Required for `transformers` backend or when downloading HuggingFace models |

**Example configurations:**

```env
# OpenAI
ALUE_ENDPOINT_TYPE=openai
ALUE_OPENAI_API_KEY=sk-...

# vLLM server
ALUE_ENDPOINT_TYPE=vllm
ALUE_ENDPOINT_URL=http://localhost:8000/v1
HF_TOKEN=hf_...

# Local Ollama
ALUE_ENDPOINT_TYPE=ollama
ALUE_ENDPOINT_URL=http://localhost:11434

# Transformers (offline)
ALUE_ENDPOINT_TYPE=transformers
HF_TOKEN=hf_...
```

---

### LLM Judge Configuration

These settings control the LLM used for evaluation in **RAG** and **Summarization** tasks. The judge is used for metrics like Context Relevancy, Composite Correctness, and Claim Decomposition.

| Variable | Type | Options | Description |
|----------|------|---------|-------------|
| `ALUE_LLM_JUDGE_ENDPOINT_TYPE` | Optional | Same as `ALUE_ENDPOINT_TYPE` | Backend type for LLM judge |
| `ALUE_LLM_JUDGE_OPENAI_API_KEY` | Conditional | API key string | Required when judge uses OpenAI |
| `ALUE_LLM_JUDGE_ENDPOINT_URL` | Conditional | URL | Required when judge uses `vllm`, `tgi`, `ollama`, or OpenAI-compatible endpoint |

> **Note:** LLM Judge is **only required** for RAG and Summarization evaluation. MCQA and Extractive QA do not use it.

> **Important:** Using the same model for both inference and judging is not recommended, as it can introduce evaluation bias. Mixing backends is explicitly supported and encouraged.

**Example configurations:**

```env
# Same backend, different model (if running multiple vLLM servers)
ALUE_LLM_JUDGE_ENDPOINT_TYPE=vllm
ALUE_LLM_JUDGE_ENDPOINT_URL=http://localhost:8001/v1

# Different backend entirely (recommended)
ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai
ALUE_LLM_JUDGE_OPENAI_API_KEY=sk-...
```

---

### Embedding Configuration

These settings control the embedding provider used for **RAG** tasks (vector database creation and retrieval).

| Variable | Type | Options | Description |
|----------|------|---------|-------------|
| `EMBEDDING_ENDPOINT_TYPE` | Optional | `openai`, `ollama`, `hf`, `local`, `openai-compatible` | Embedding provider type. Default: `local` |
| `EMBEDDING_API_KEY` | Conditional | API key string | Required for `openai` embeddings |
| `EMBEDDING_ENDPOINT_URL` | Conditional | URL | Required for `ollama`, `hf`, or `openai-compatible` |
| `HF_TOKEN` | Conditional | Token string | Required for Hugging Face embedding models (`hf`) |

**Example configurations:**

```env
# OpenAI embeddings
EMBEDDING_ENDPOINT_TYPE=openai
EMBEDDING_API_KEY=sk-...

# Local embeddings (default, no external API needed)
EMBEDDING_ENDPOINT_TYPE=local

# Hugging Face embeddings
EMBEDDING_ENDPOINT_TYPE=hf
HF_TOKEN=hf_...
EMBEDDING_ENDPOINT_URL=http://localhost:8080

# Ollama embeddings
EMBEDDING_ENDPOINT_TYPE=ollama
EMBEDDING_ENDPOINT_URL=http://localhost:11434
```

---

## Backend-Specific Notes

### OpenAI
- Requires valid API key with appropriate model access
- Supports all OpenAI chat models (GPT-4, GPT-3.5, etc.)
- Embedding support includes `text-embedding-ada-002` and newer models

### vLLM
- **Online mode** (`vllm`): Requires a running vLLM server with OpenAI-compatible API
- **Offline mode** (`vllm-offline`): Runs vLLM directly in-process (no server needed)
- Server must be started separately: `vllm serve <model_name> --port 8000`

### TGI (Text Generation Inference)
- Requires a running TGI server
- Must expose OpenAI-compatible endpoint
- See [HuggingFace TGI documentation](https://huggingface.co/docs/text-generation-inference)

### Ollama
- Requires Ollama running locally or on accessible host
- Default URL: `http://localhost:11434`
- Models must be pulled first: `ollama pull <model_name>`
- Supports both inference and embeddings

### Transformers
- Runs models directly via Hugging Face `transformers` library
- No external server required
- Requires sufficient GPU/CPU memory for model
- Model weights downloaded automatically on first use (requires `HF_TOKEN` for gated models)

---

## Complete `.env.example`

Here's a comprehensive example showing all available settings:

```env
# === Inference Backend Configuration ===
# Options: openai | vllm | tgi | ollama | transformers | vllm-offline
ALUE_ENDPOINT_TYPE=openai

# Required for OpenAI
ALUE_OPENAI_API_KEY=sk-...

# Required for vLLM, TGI, Ollama, or openai-compatible endpoints
ALUE_ENDPOINT_URL=http://localhost:8000/v1

# Required for Hugging Face models (transformers backend)
HF_TOKEN=hf_...

# === LLM Judge Configuration (for RAG and Summarization evaluation) ===
# Options: same as ALUE_ENDPOINT_TYPE
ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai

# Required if judge uses OpenAI
ALUE_LLM_JUDGE_OPENAI_API_KEY=sk-...

# Required if judge uses vLLM, TGI, Ollama, or openai-compatible
ALUE_LLM_JUDGE_ENDPOINT_URL=http://localhost:8001/v1

# === Embedding Configuration (for RAG tasks) ===
# Options: openai | ollama | hf | local | openai-compatible
EMBEDDING_ENDPOINT_TYPE=local

# Required for OpenAI embeddings
EMBEDDING_API_KEY=sk-...

# Required for ollama, hf, or openai-compatible embeddings
EMBEDDING_ENDPOINT_URL=http://localhost:11434

# Required for Hugging Face embedding models
HF_TOKEN=hf_...
```

---

## Common Configuration Patterns

### All-OpenAI (Simplest)
```env
ALUE_ENDPOINT_TYPE=openai
ALUE_OPENAI_API_KEY=sk-...
ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai
ALUE_LLM_JUDGE_OPENAI_API_KEY=sk-...
EMBEDDING_ENDPOINT_TYPE=openai
EMBEDDING_API_KEY=sk-...
```

### All-Local (No External APIs)
```env
ALUE_ENDPOINT_TYPE=ollama
ALUE_ENDPOINT_URL=http://localhost:11434
ALUE_LLM_JUDGE_ENDPOINT_TYPE=ollama
ALUE_LLM_JUDGE_ENDPOINT_URL=http://localhost:11434
EMBEDDING_ENDPOINT_TYPE=local
```

### Mixed (Recommended for Production)
```env
# Fast inference with vLLM
ALUE_ENDPOINT_TYPE=vllm
ALUE_ENDPOINT_URL=http://localhost:8000/v1
HF_TOKEN=hf_...

# Separate judge to reduce bias
ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai
ALUE_LLM_JUDGE_OPENAI_API_KEY=sk-...

# Local embeddings (no API costs)
EMBEDDING_ENDPOINT_TYPE=local
```

---

## Troubleshooting

### "Missing API key" errors
- Verify the appropriate `*_API_KEY` variable is set for your backend
- Check for typos in the `.env` file
- Ensure `.env` is in the project root directory

### "Connection refused" errors
- Verify the server is running (for vLLM, TGI, Ollama)
- Check that `*_ENDPOINT_URL` points to the correct host and port
- Ensure no firewall is blocking the connection

### "Model not found" errors
- For Ollama: Run `ollama pull <model_name>` first
- For Transformers: Verify `HF_TOKEN` is set and has access to the model
- For vLLM/TGI: Ensure the server was started with the correct model

### "HuggingFace token invalid"
- Generate a new token at https://huggingface.co/settings/tokens
- Ensure the token has read access
- For gated models, accept the model's license agreement first

---

## See Also

* [Getting Started](getting-started.md) — Quick setup guide
* [Models & Backends](running-models.md) — Detailed backend comparison
* [Tasks](tasks/index.md) — Task-specific configuration requirements
```


