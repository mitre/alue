# Multiple Choice Question Answering (MCQA)

MCQA evaluates a model’s ability to answer **multiple choice questions** with a single correct option.  
This task is well-suited for aviation knowledge exams and other structured assessments.

---

## Dataset Format

MCQA datasets follow this structure:

```json
{
  "task_info": {
    "task_type": "mcqa",
    "task_name": "Aviation Knowledge Exam",
    "description": "Multiple choice questions on aviation knowledge"
  },
  "examples": [
    {
      "id": 1,
      "input": "What part of a rotary-wing aircraft makes directional control possible?\nA) the teeter hinge\nB) the swashplate\nC) the ducted fan",
      "output": "B"
    }
  ],
  "data": [
    {
      "id": 1,
      "input": "What should a pilot do if after crossing a stop bar, the taxiway centerline lead-on lights inadvertently extinguish?\nA) Proceed with caution\nB) Hold their position and contact ATC\nC) Turn back towards the stop bar",
      "output": "B"
    }
  ]
}
````

* `task_info` → metadata about the task
* `examples` → few-shot examples for prompt construction
* `data` → test/evaluation questions with ground truth answers

---

## Prompt Templates

Example templates for **aviation exam** MCQA (`templates/aviation_exam/`):

**system.jinja2**

```jinja2
Based on the given question and options, select the correct answer. Choose only one option that best answers the question.

{% if examples %}
Here are some examples:
{% for example in examples %}
    Question: {{ example.input }}
    Answer: {{ example.output }}
{% endfor %}
{% endif %}

Choose only one option that best answers the question
```

**user.jinja2**

```jinja2
Question: {{ input }}
```

---

## Template Variables

For MCQA, templates must reference:

| Variable   | Description                                |
| ---------- | ------------------------------------------ |
| `input`    | Question text with multiple choice options |
| `examples` | Few-shot QA pairs (input + output)         |

---

## Running MCQA

The entrypoint script `scripts/mcqa.py` supports three modes: **inference**, **evaluation**, or **both**.

### Example: Run Inference Only

```bash
python scripts/mcqa.py inference \
  -i data/mcqa/aviation_exam.json \
  -o runs/mcqa \
  -m gpt-4o-mini \
  --task_type aviation_exam \
  --num_examples 3
```

### Example: Run Evaluation Only

```bash
python scripts/mcqa.py evaluation \
  -i data/mcqa/aviation_exam.json \
  -o runs/mcqa \
  --predictions_file runs/mcqa/predictions.json
```

### Example: Run Inference + Evaluation

```bash
python scripts/mcqa.py both \
  -i data/mcqa/aviation_exam.json \
  -o runs/mcqa \
  -m gpt-4o-mini \
  --task_type aviation_exam \
  --num_examples 3 \
  --schema_class MCQAResponse \
  --field_to_extract answer
```

---

## Structured Output with Schemas (Recommended)

MCQA strongly benefits from schema enforcement.
Define an explicit schema for answer choice selection:

```python
from pydantic import BaseModel
from enum import Enum

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"

class MCQAResponse(BaseModel):
    answer: AnswerChoice
```

Run with schema enforcement:

```bash
python scripts/mcqa.py inference \
  -i data/mcqa/aviation_exam.json \
  -o runs/mcqa \
  -m gpt-4o-mini \
  --task_type aviation_exam \
  --num_examples 3 \
  --schema_class MCQAResponse \
  --field_to_extract answer
```

**Result Example:**

```json
{
  "1": "B",
  "2": "C"
}
```

---

## Metrics

MCQA is evaluated using **accuracy**:

* **Accuracy** = Correct predictions ÷ Total questions
* Reports: `accuracy`, `correct`, `total` in `metrics.json`

---

## Configuration Notes

* **LLM Judge** is **not required** for MCQA.
* **Embedding configuration is not required** (no retrieval step).
* Using schemas (`MCQAResponse`) is strongly recommended to avoid invalid outputs.
