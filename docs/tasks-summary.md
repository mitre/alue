# Tasks Summary

ALUE supports multiple task types (RAG, MCQA, Summarization, Extractive QA, Classification, etc.).  
Each task uses a **template configuration** (`templates/<task>/`) and optionally a **schema configuration** (`schemas/<task>/`) for structured generation.

This page summarizes the expected **template variables** per task and notes where **schemas** are typically used.

---

## Template Variables

All tasks rely on Jinja2 templates for prompt construction. ALUE uses `build_messages()` to inject task-specific variables into `system.jinja2` and `user.jinja2`.

| Task | `system.jinja2` variables | `user.jinja2` variables | Notes |
|---|---|---|---|
| **RAG** | `examples`, containing `query` and `context` | `query`, `context` | `examples` are optional; `context` contains retrieved chunks. |
| **MCQA** | `examples`, containing `input` and `output` | `input` | `input` = question with options; `output` = single correct choice. |
| **Summarization** | `examples` (if `"split": "example"`) | `input` | Typically unstructured, no schema. |
| **Extractive QA** | `examples` containing `question`, `transcript` and `answer` | `query`, `context` | Aligns with SQuAD-style datasets. |

---

## Schemas (Structured Generation)

Schemas are defined under `schemas/<task>/schema.py` as Pydantic models. They enable **structured outputs** (JSON conforming to a schema) instead of free-form text.

Note that while the schema is not available for many of the tasks, the user is free to add their own if they would like.

| Task | Schema availability | Example usage |
|---|---|---|
| **RAG** | Not used (free-form) | Evaluation focuses on correctness and relevancy. |
| **MCQA** | Optional | Can enforce `{ "answer": str }`. |
| **Summarization** | Not used | Free-form generation. |
| **Extractive QA** | Often used | Exact spans extracted as list of strings. |
| **Classification** | Common | Structured label assignment. |
| **NER / Token Classification** | Common | Structured entities with tags and spans. |

Schemas are **optional for inference** but recommended where outputs must be validated or normalized for evaluation.

---

## Guidance

- **Templates**: Begin with the provided `system.jinja2` and `user.jinja2`. Modify or extend for custom strategies.  
- **Schemas**: Use for tasks with strict label spaces (classification, NER). Skip for free-form tasks (RAG, summarization).  
- **Custom tasks**: Create a new folder under both `templates/` and `schemas/` with matching names to enable integration.
