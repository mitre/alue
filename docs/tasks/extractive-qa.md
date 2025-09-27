
# Extractive Question Answering (Extractive QA)

Extractive QA evaluates a model’s ability to identify **text spans** from a transcript or document that directly answer a given question.  
This task is especially relevant for NTSB transcripts or structured regulatory text where answers must be extracted verbatim.

---

## Dataset Format

Datasets follow a **SQuAD-style format** with `examples` (few-shot for prompt construction) and `paragraphs` (test/eval instances):

```json
{
  "version": "squad_v1",
  "data": [
    {
      "title": "ntsb_extractive_qa",
      "task": "Extract the exact span ...",
      "examples": [
        {
          "question": "Is a tail number mentioned ...?",
          "transcript": "On March 18, 2008 ... a Robinson R22 beta, M3056T, ...",
          "answer": "[M3056T]"
        }
      ],
      "paragraphs": [
        {
          "context": "The pilot reported while making an approach ...",
          "qas": [
            {
              "id": "e80026dc-c640-484f-94c6-4f7597cfce1f",
              "question": "Is a tail number mentioned ...?",
              "answers": [{ "text": "['NONE']" }]
            }
          ]
        }
      ]
    }
  ]
}
````

* `examples` → Few-shot examples for prompt construction
* `paragraphs` → Contexts and QA pairs for inference/evaluation
* `answers.text` → Gold-standard text span(s)

---

## Prompt Templates

Example prompt for **tail number extraction** (`templates/extract_tail_number/`):

**system.jinja2**

```jinja2
You are a helpful AI assistant that identifies tail numbers from NTSB transcripts, if a tail number exists.

{% if examples %}
{% for example in examples %}
Question: {{ example.question }}
Context: {{ example.transcript }}
Tail Number: {{ example.answer }}

{% endfor %}
{% endif %}

Do not provide sentence-long spans or explanations in your answers. Just provide the tail numbers formatted as a list.
```

**user.jinja2**

```jinja2
Question: {{query}}
Context: {{context}}
```

---

## Template Variables

For Extractive QA, templates must reference:

| Variable   | Description                               |
| ---------- | ----------------------------------------- |
| `query`    | The input question                        |
| `context`  | The transcript or document passage        |
| `examples` | Few-shot QA pairs for in-context learning |

---

## Running Extractive QA

The entrypoint script `scripts/extractive_qa.py` supports three modes: **inference**, **evaluation**, or **both**.

### Example: Run Inference Only

```bash
python scripts/extractive_qa.py inference \
  -i data/extractive_qa/tail_number.json \
  -o runs/extractive_qa \
  -m gpt-4o-mini \
  --task_type extract_tail_number \
  --num_examples 3
```

### Example: Run Evaluation Only

```bash
python scripts/extractive_qa.py evaluation \
  -i data/extractive_qa/tail_number.json \
  -o runs/extractive_qa \
  --predictions_file runs/extractive_qa/predictions.json \
  --normalizer_func normalize_tail_extraction_predictions
```

### Example: Run Inference + Evaluation

```bash
python scripts/extractive_qa.py both \
  -i data/extractive_qa/tail_number.json \
  -o runs/extractive_qa \
  -m gpt-4o-mini \
  --task_type extract_tail_number \
  --num_examples 3 \
  --normalizer_func normalize_tail_extraction_predictions
```

---

## Structured Output with Schemas (Optional)

ALUE supports **structured generation** using Pydantic schemas.
For example, the tail number extraction task can enforce output via:

```python
from pydantic import BaseModel
from typing import List

class TailNumberResponse(BaseModel):
    tail_numbers: List[str]
```

Run with schema enforcement:

```bash
python scripts/extractive_qa.py inference \
  -i data/extractive_qa/tail_number.json \
  -o runs/extractive_qa \
  -m gpt-4o-mini \
  --task_type extract_tail_number \
  --num_examples 3 \
  --schema_class TailNumberResponse \
  --field_to_extract tail_numbers
```

**Result Example:**

```json
{
  "e80026dc-c640-484f-94c6-4f7597cfce1f": ["N404EH"],
  "7a661465-9721-4ab0-b27f-f935c4e166a0": ["NONE"]
}
```

---

## Normalization Functions

Since models may output answers in varied formats (lists, strings, or raw spans), ALUE provides normalization utilities.

Example: `normalize_tail_extraction_predictions`

```python
def normalize_tail_extraction_predictions(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize tail number predictions.

    [] -> "[NONE]"
    ["N66372"] -> "[N66372]"
    "[NONE]" -> "[NONE]"
    "[N66372]" -> "[N66372]"
    """
```

Call via CLI with:

```bash
--normalizer_func normalize_tail_extraction_predictions
```

---

## Metrics

Extractive QA uses **SQuAD-style evaluation**:

* **Exact Match (EM)** → % of answers exactly matching gold spans
* **F1 Score** → Token-level overlap between prediction and gold span(s)
* Handles **multiple spans**, **no-answer cases**, and **normalized predictions**

---

## Configuration Notes

* **LLM Judge** is **not required** for Extractive QA (unlike RAG or Summarization).
* **Embedding configuration is not required** (no retrieval step involved).
* Schema and normalization are optional.
