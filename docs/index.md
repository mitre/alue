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

## Quickstart

```bash
# Install ALUE in editable mode
pip install -e .


# Example: Run MCQA
python -m alue.run_task \
    --task mcqa \
    --config configs/mcqa_example.yaml
```

# Citation

If you use ALUE in academic or applied work, please cite:

@inproceedings{alue2025,
  title        = {ALUE: Aerospace Language Understanding and Evaluation},
  author       = {…},
  booktitle    = {AIAA Scitech Forum},
  year         = {2025},
  doi          = {10.2514/6.2025-3247}
}