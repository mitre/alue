# Configuration

Set values via environment or `.env` (loaded by `pydantic-settings`).

```env
ALUE_ENDPOINT_TYPE=openai            # openai|vllm|tgi|ollama|transformers
ALUE_ENDPOINT_URL=                   # for vLLM/TGI/Ollama/openai-compatible
ALUE_OPENAI_API_KEY=sk-...
HF_TOKEN=...

ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai
ALUE_LLM_JUDGE_ENDPOINT_URL=
ALUE_LLM_JUDGE_OPENAI_API_KEY=

EMBEDDING_ENDPOINT_TYPE=local        # openai|ollama|hf|local|openai-compatible
EMBEDDING_ENDPOINT_URL=
EMBEDDING_API_KEY=
```

The same model can be used for LLM-judge, but we don’t recommend it due to bias.