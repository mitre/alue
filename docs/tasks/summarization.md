# Summarization

Summarization generates concise summaries of aviation narratives. ALUE provides a templated prompting interface, JSONL dataset format, and LLM-as-judge evaluation based on **claim decomposition** (precision/recall/F1).

---

## Dataset Format

A line-delimited JSON file where each object has `input`, `output`, and an optional `split`/`id`:

```jsonl
{"input": "Flight 247 departed from Denver International Airport at 2:15 PM, 30 minutes behind schedule due to a delayed inbound aircraft. The Boeing 737 carried 142 passengers and 6 crew members on the route to Seattle. During the flight, passengers were served complimentary snacks and beverages. The captain announced smooth weather conditions and an on-time arrival despite the initial delay. Flight attendants conducted the usual safety demonstrations and assisted passengers throughout the journey. The aircraft landed at Seattle-Tacoma International Airport at 4:45 PM local time, where passengers disembarked normally at gate B12. Ground crews efficiently unloaded baggage, and the aircraft was prepared for its next scheduled departure.", "output": "Flight 247 departed Denver 30 minutes late but arrived on time in Seattle with 142 passengers aboard. The Boeing 737 flight proceeded normally with standard service and a routine landing.", "split": "example"}
{"input": "A small Cessna 172 aircraft made an emergency landing in a farmer's field near Springfield Airport yesterday afternoon. The pilot, conducting a routine training flight with an instructor, reported engine trouble at approximately 3,000 feet altitude. Following emergency procedures, the instructor took control and successfully guided the aircraft to the open field without injuries to either occupant. Local fire department and paramedics responded to the scene as a precaution. The Federal Aviation Administration has been notified and will investigate the incident. The aircraft sustained minor damage to its landing gear, but both the student pilot and certified flight instructor walked away unharmed. The field owner was cooperative and assisted with the aircraft recovery.", "output": "Cessna 172 made emergency landing in field near Springfield Airport due to engine trouble during training flight. Both instructor and student pilot escaped injury despite minor aircraft damage.", "split": "test", "id": "dummy_test_1"}
{"input": "The regional airport authority announced plans to expand Terminal A with a new international wing scheduled to open in 2025. The $45 million construction project will add six additional gates capable of handling wide-body aircraft and a new customs facility. Airport officials expect the expansion to increase annual passenger capacity by 40% and attract new international carriers to the region. Construction will begin next month and is designed to minimize disruption to current operations. The project includes upgraded baggage handling systems, expanded parking facilities, and improved ground transportation access. Local business leaders praised the development as a catalyst for regional economic growth and increased tourism opportunities.", "output": "Regional airport plans $45 million Terminal A expansion with six new international gates opening in 2025. The project aims to increase passenger capacity by 40% and attract new international airlines.", "split": "test", "id": "dummy_test_2"}
```

**Key fields:**
- `input` — Full narrative text to be summarized
- `output` — Ground-truth reference summary
- `split` — `"example"` for few-shot examples, `"test"` for evaluation items
- `id` — Unique identifier (optional)

The data loader treats `split == "example"` as few-shot examples; all other items are used for evaluation.

---

## Quick Start

### Prerequisites

Ensure you have configured your environment (see [Getting Started](../getting-started.md)):
- **Inference backend** (e.g., OpenAI, vLLM, Ollama)
- **LLM Judge** for evaluation (recommended: different from inference model)

### Minimal Working Example

```bash
# Run inference + evaluation
python -m scripts.summarization both \
  -i data/summarization/sample.jsonl \
  -o runs/sum \
  -m gpt-4o-mini \
  --task_type summarization \
  --num_examples 3 \
  --llm_judge_model_name gpt-4o-mini \
  --verbose
```

This will:
1. Generate summaries using the inference model
2. Evaluate summaries using claim decomposition (precision/recall/F1)

Results are saved to `runs/sum_<timestamp>/`.

---

## Running Summarization

The summarization script supports three modes: **inference**, **evaluation**, or **both**.

### Inference Only

Generate summaries without evaluation:

```bash
python -m scripts.summarization inference \
  -i data/summarization/sample.jsonl \
  -o runs/sum \
  -m gpt-4o-mini \
  --task_type summarization \
  --num_examples 3 \
  --temperature 0.1 \
  --max_tokens 150
```

**Outputs:**
- `predictions.json` — Contains `narrative`, `ground_truth_summary`, `predicted_summary` for each item
- `metadata.json` — Run parameters

> **Note:** Structured generation is optional. If you use a custom schema, pass `--schema_class YourSchema --field_to_extract field_name`. Otherwise omit both for free-form text output.

### Evaluation Only

Evaluate existing predictions:

```bash
python -m scripts.summarization evaluation \
  -i data/summarization/sample.jsonl \
  -o runs/sum_eval \
  --predictions_file runs/sum_<timestamp>/predictions.json \
  --llm_judge_model_name gpt-4o-mini \
  --verbose
```

**Outputs:**
- `claim_decomposition_detailed.json` — Per-item claims, classification matrices, and scores
- `summarization_metrics.json` — `average_precision`, `average_recall`, `average_f1`, `total_samples`

### Inference + Evaluation

Run both steps in sequence:

```bash
python -m scripts.summarization both \
  -i data/summarization/sample.jsonl \
  -o runs/sum \
  -m gpt-4o-mini \
  --task_type summarization \
  --num_examples 3 \
  --temperature 0.1 \
  --max_tokens 150 \
  --llm_judge_model_name gpt-4o-mini \
  --verbose
```

---

## Templates and Variables

Summarization templates are located in `templates/summarization/`:

**system.jinja2**
```jinja2
You are an aviation safety analyst. Analyze the following narrative from a pilot that describes a potentially risky safety event, and then generate a one to two sentence summary of the event.
{% if examples %}
For example:
{% for example in examples %}
    Input Text: {{ example.input }}
    Summary: 
    {{ example.output }}
{% endfor %}
{% endif %}
```

**user.jinja2**
```jinja2
Input Text: {{ input }}
```

### Expected Template Variables

Templates must reference:

| Variable | Description |
|----------|-------------|
| `examples` | Few-shot examples (list of `{input, output}`) |
| `input` | The narrative text to summarize |

If you customize templates, ensure these variables are preserved.

---

## Evaluation Metrics

### Claim Decomposition (LLM-as-Judge)

**What it measures:** Precision, recall, and F1 of generated summaries based on atomic claim analysis.

**How it works:**

For each item:
1. **Decompose ground-truth summary** into atomic claims; keep only claims supported by the original narrative
2. **Decompose predicted summary** into atomic claims
3. **Classify each predicted claim** as:
   - **Strong** — supported by a ground-truth claim
   - **Weak** — supported by the narrative but not by a ground-truth claim
   - **Incorrect** — supported by neither
4. **Compute scores:**
   - `precision = (1.0·strong + 0.5·weak − 0.1·incorrect) / total_predicted_claims` (floored at 0)
   - `recall = (ground_truth_claims_covered) / total_ground_truth_claims`
   - `F1 = 2·precision·recall / (precision + recall)`

> **Note:** Weights (1.0/0.5/0.1) are configurable in code; defaults are shown above.

**Reported metrics:**
- `average_precision` — Overall precision across all summaries
- `average_recall` — Overall recall across all summaries
- `average_f1` — Overall F1 score across all summaries
- Per-item breakdowns in `claim_decomposition_detailed.json`

---

## Configuration Notes

### Required Settings

**For inference:**
```env
ALUE_ENDPOINT_TYPE=openai
ALUE_OPENAI_API_KEY=sk-...
```

**For evaluation:**
```env
ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai
ALUE_LLM_JUDGE_OPENAI_API_KEY=sk-...
```

> **Recommendation:** Use a different model for the LLM judge than for inference to reduce evaluation bias. Mixing backends is supported.

See [Configuration Reference](../model-configuration.md) for complete setup details.

---


## See Also

* [Configuration Reference](../model-configuration.md) — LLM judge and inference setup
* [Creating Datasets](../creating-datasets.md) — Summarization dataset format specification
* [Models & Backends](../running-models.md) — Supported inference engines
