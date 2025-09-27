# Summarization

This task generates concise summaries of aviation narratives. ALUE provides a templated prompting interface, a simple JSON dataset format, and an LLM-as-judge evaluation based on **claim decomposition** (precision/recall/F1).

---

## Prerequisites

* **Python** 3.10; environment installed via `uv sync` (preferred) or `pip -r requirements.txt`.
* **Inference backend** configured (see *Models & Backends*).
* **LLM Judge** must be configured for evaluation (see *Configuration*). Set:

  * `ALUE_LLM_JUDGE_ENDPOINT_TYPE`, `ALUE_LLM_JUDGE_ENDPOINT_URL` (if applicable),
  * `ALUE_LLM_JUDGE_OPENAI_API_KEY` (if needed).

---

## Templates

Location: `templates/summarization/`

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

### Expected template variables (required by ALUE)

```python
message = build_messages(
    task_type=args.task_type,                  # "summarization"
    system_kwargs={"examples": examples},      # list of {input, output}
    user_kwargs={"input": input_text}
)
```

If you customize templates, ensure these variables are preserved.

---

## Dataset format

A line-delimited JSON file where each object has `input`, `output`, and an optional `split`/`id`.

**Examples**

```json
{"input": "Flight 247 departed from Denver International Airport at 2:15 PM, 30 minutes behind schedule due to a delayed inbound aircraft. The Boeing 737 carried 142 passengers and 6 crew members on the route to Seattle. During the flight, passengers were served complimentary snacks and beverages. The captain announced smooth weather conditions and an on-time arrival despite the initial delay. Flight attendants conducted the usual safety demonstrations and assisted passengers throughout the journey. The aircraft landed at Seattle-Tacoma International Airport at 4:45 PM local time, where passengers disembarked normally at gate B12. Ground crews efficiently unloaded baggage, and the aircraft was prepared for its next scheduled departure.", "output": "Flight 247 departed Denver 30 minutes late but arrived on time in Seattle with 142 passengers aboard. The Boeing 737 flight proceeded normally with standard service and a routine landing.", "split": "example"}
{"input": "A small Cessna 172 aircraft made an emergency landing in a farmer's field near Springfield Airport yesterday afternoon. The pilot, conducting a routine training flight with an instructor, reported engine trouble at approximately 3,000 feet altitude. Following emergency procedures, the instructor took control and successfully guided the aircraft to the open field without injuries to either occupant. Local fire department and paramedics responded to the scene as a precaution. The Federal Aviation Administration has been notified and will investigate the incident. The aircraft sustained minor damage to its landing gear, but both the student pilot and certified flight instructor walked away unharmed. The field owner was cooperative and assisted with the aircraft recovery.", "output": "Cessna 172 made emergency landing in field near Springfield Airport due to engine trouble during training flight. Both instructor and student pilot escaped injury despite minor aircraft damage.", "split": "test", "id": "dummy_test_1"}
{"input": "The regional airport authority announced plans to expand Terminal A with a new international wing scheduled to open in 2025. The $45 million construction project will add six additional gates capable of handling wide-body aircraft and a new customs facility. Airport officials expect the expansion to increase annual passenger capacity by 40% and attract new international carriers to the region. Construction will begin next month and is designed to minimize disruption to current operations. The project includes upgraded baggage handling systems, expanded parking facilities, and improved ground transportation access. Local business leaders praised the development as a catalyst for regional economic growth and increased tourism opportunities.", "output": "Regional airport plans $45 million Terminal A expansion with six new international gates opening in 2025. The project aims to increase passenger capacity by 40% and attract new international airlines.", "split": "test", "id": "dummy_test_2"}
```

* The loader treats `split == "example"` as few-shot examples; others are evaluation/test items.

---

## Run: Inference

```bash
python scripts/summarization.py inference \
  -i data/summarization/sample.jsonl \
  -o runs/sum \
  -m gpt-4o-mini \
  --task_type summarization \
  --num_examples 3 \
  --temperature 0.1 \
  --max_tokens 150
```

Outputs in `runs/sum_YYYYMMDD_HHMMSS/`:

* `predictions.json` (keys = input `id`; values = `narrative`, `ground_truth_summary`, `predicted_summary`)
* `metadata.json` (run parameters)

> Structured generation is optional. If you use a custom schema, pass `--schema_class YourSchema --field_to_extract field_name`. Otherwise omit both (free-form text output).

---

## Run: Evaluation

Evaluation uses an LLM judge to decompose summaries into claims and compute **precision**, **recall**, and **F1**.

```bash
python scripts/summarization.py evaluation \
  -i data/summarization/sample.jsonl \
  -o runs/sum_eval \
  --predictions_file runs/sum_YYYYMMDD_HHMMSS/predictions.json \
  --llm_judge_model_name gpt-4o-mini \
  --verbose
```

Artifacts:

* `claim_decomposition_detailed.json` — per-item claims, matrices, and scores
* `summarization_metrics.json` — averages: `average_precision`, `average_recall`, `average_f1`, `total_samples`


---

Absolutely — here’s an additional section you can append to the Summarization page.

---

## Run: Both (inference + evaluation)

```bash
python scripts/summarization.py both \
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

This executes inference first, writes `predictions.json`, then immediately runs evaluation using the specified LLM judge. The timestamped output directory (e.g., `runs/sum_YYYYMMDD_HHMMSS/`) will contain:

* `predictions.json` — generated summaries.
* `metadata.json` — run parameters.
* `claim_decomposition_detailed.json` — per-item claim analysis.
* `summarization_metrics.json` — `average_precision`, `average_recall`, `average_f1`, `total_samples`.


### Metric definition (claim decomposition)

For each item:

1. Decompose **ground-truth** summary into claims; keep only claims supported by the original **narrative**.
2. Decompose **predicted** summary into claims.
3. Classify each predicted claim as:

   * **Strong** — supported by a ground-truth claim,
   * **Weak** — supported by the narrative but not by a ground-truth claim,
   * **Incorrect** — supported by neither.
4. Compute:

   * `precision = (1.0·strong + 0.5·weak − 0.1·incorrect) / total_predicted_claims` (floored at 0),
   * `recall = (ground_truth_claims_covered) / total_ground_truth_claims`,
   * `F1 = 2·precision·recall / (precision + recall)`.

Weights (1.0/0.5/0.1) are configurable in code; defaults are shown.

---

## Notes on configuration

* **LLM judge settings** (required for evaluation) are read from environment via `ALUE_LLM_JUDGE_*` in `settings.py`. Ensure the judge’s backend is reachable and authorized.
* **Inference backend** is independent of the judge backend; using a different model for judging is recommended to reduce bias.

---