# Creating Datasets

ALUE expects **structured input files** per task. This page provides the authoritative schemas and format specifications for each supported task type.

## Overview

Datasets generally include:
- **Few-shot examples** (optional but recommended) - Used for in-context learning
- **Evaluation data** - The test set to be evaluated

The data loader (`alue.data_utils.load_data`) reads these files and exposes:
- `get_examples(num_examples=...)` - Returns few-shot examples for prompting
- `get_test_data()` - Returns the evaluation set

---

## Summary Table

| Task              | File Type | Few-shot location                 | Eval location               | Required Fields (eval items)                                                            | Optional Fields                        |
| ----------------- | --------- | --------------------------------- | --------------------------- | --------------------------------------------------------------------------------------- | -------------------------------------- |
| **MCQA**          | JSON      | `examples[]`                      | `data[]`                    | `id`, `input` (stem + labeled choices), `output` (label)                                | `task_info` (metadata)                 |
| **Extractive QA** | JSON      | `data[].examples[]`               | `data[].paragraphs[].qas[]` | `qas[].id`, `qas[].question`, `qas[].answers[].text`, enclosing `context` per paragraph | `title`, `task`                        |
| **Summarization** | JSONL     | `split: "example"` lines          | `split: "test"` lines       | `input`, `output`                                                                       | `id`, `split`                          |
| **RAG**           | JSON      | `examples` (query/context/answer) | `data[].paragraphs[].qas[]` | `id`, `question`, `answers[].text`                                                      | `answers[].document_id` (for Recall@k) |

---

## Quick Reference: Required Fields

### MCQA
```json
{
  "id": "unique_id",
  "input": "Question text with options\nA) option1\nB) option2",
  "output": "A"
}
```

### Extractive QA
```json
{
  "context": "The passage containing the answer...",
  "qas": [{
    "id": "unique_id",
    "question": "What is...?",
    "answers": [{"text": "exact span from context"}]
  }]
}
```

### Summarization
```jsonl
{"input": "Long narrative text...", "output": "Brief summary.", "split": "test", "id": "unique_id"}
```

### RAG
```json
{
  "id": "unique_id",
  "question": "What is...?",
  "answers": [{
    "text": "The answer text",
    "document_id": ["chunk_id_in_chromadb"]
  }]
}
```

---

## Multiple-Choice QA (MCQA)

**Purpose:** Select one correct option from a set of choices.

**File format:** JSON with top-level `task_info`, `examples`, and `data`.

### Schema

* `task_info` (object): descriptive metadata (not used by evaluation)
* `examples` (list): few-shot, each with
  * `id` (int or str)
  * `input` (str): stem + labeled choices (e.g., "A) … B) …")
  * `output` (str): correct label (e.g., `"A"`)
* `data` (list): evaluation set, same fields as `examples`

### Example Dataset

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
      "input": "What part of a rotary-wing aircraft makes directional control possible?\r\nA) the teeter hinge\r\nB) the swashplate\r\nC) the ducted fan",
      "output": "B"
    }
  ],
  "data": [
    {
      "id": 1,
      "input": "What should a pilot do if after crossing a stop bar, the taxiway centerline lead-on lights inadvertently extinguish?\r\nA) Proceed with caution\r\nB) Hold their position and contact ATC\r\nC) Turn back towards the stop bar",
      "output": "B"
    }
  ]
}
```

---

## Extractive QA

**Purpose:** Extract **exact spans** from `context` that answer the question (SQuAD-style).

**File format:** JSON with top-level `version` and `data`.

### Schema

* `data` (list of datasets), each with:
  * `title` (str)
  * `task` (str): instruction string provided to the model
  * `examples` (list of calibration/illustration items), each with:
    * `question` (str)
    * `transcript` (str)
    * `answer` (str)
  * `paragraphs` (list for evaluation), each with:
    * `context` (str)
    * `qas` (list), each with:
      * `id` (str): unique question id
      * `question` (str)
      * `answers` (list of objects) with:
        * `text` (str): answer text

### Example Dataset

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
```

---

## Summarization

**Purpose:** Produce concise summaries of aviation narratives.

**File format:** JSONL (one JSON object per line).

### Schema per line

* `input` (str): narrative text
* `output` (str): reference summary
* `split` (str): `"example"` or `"test"` (used by loader to separate few-shot examples from evaluation set)
* `id` (str): unique identifier

### Example Dataset

```jsonl
{"input": "Flight 247 departed from Denver International Airport at 2:15 PM, 30 minutes behind schedule due to a delayed inbound aircraft. The Boeing 737 carried 142 passengers and 6 crew members on the route to Seattle. During the flight, passengers were served complimentary snacks and beverages. The captain announced smooth weather conditions and an on-time arrival despite the initial delay. Flight attendants conducted the usual safety demonstrations and assisted passengers throughout the journey. The aircraft landed at Seattle-Tacoma International Airport at 4:45 PM local time, where passengers disembarked normally at gate B12. Ground crews efficiently unloaded baggage, and the aircraft was prepared for its next scheduled departure.", "output": "Flight 247 departed Denver 30 minutes late but arrived on time in Seattle with 142 passengers aboard. The Boeing 737 flight proceeded normally with standard service and a routine landing.", "split": "example"}
{"input": "A small Cessna 172 aircraft made an emergency landing in a farmer's field near Springfield Airport yesterday afternoon. The pilot, conducting a routine training flight with an instructor, reported engine trouble at approximately 3,000 feet altitude. Following emergency procedures, the instructor took control and successfully guided the aircraft to the open field without injuries to either occupant. Local fire department and paramedics responded to the scene as a precaution. The Federal Aviation Administration has been notified and will investigate the incident. The aircraft sustained minor damage to its landing gear, but both the student pilot and certified flight instructor walked away unharmed. The field owner was cooperative and assisted with the aircraft recovery.", "output": "Cessna 172 made emergency landing in field near Springfield Airport due to engine trouble during training flight. Both instructor and student pilot escaped injury despite minor aircraft damage.", "split": "test", "id": "dummy_test_1"}
{"input": "The regional airport authority announced plans to expand Terminal A with a new international wing scheduled to open in 2025. The $45 million construction project will add six additional gates capable of handling wide-body aircraft and a new customs facility. Airport officials expect the expansion to increase annual passenger capacity by 40% and attract new international carriers to the region. Construction will begin next month and is designed to minimize disruption to current operations. The project includes upgraded baggage handling systems, expanded parking facilities, and improved ground transportation access. Local business leaders praised the development as a catalyst for regional economic growth and increased tourism opportunities.", "output": "Regional airport plans $45 million Terminal A expansion with six new international gates opening in 2025. The project aims to increase passenger capacity by 40% and attract new international airlines.", "split": "test", "id": "dummy_test_2"}
```

---

## Retrieval-Augmented Generation (RAG)

**Purpose:** Free-form answers generated *only* from retrieved context. Optionally supports retrieval supervision via ground-truth `document_id`s.

**File format:** JSON with top-level `examples` and `data`.

### Schema

- **`examples`** (few-shot)
  - `query` (str): user question
  - `context` (str): supporting context text
  - `answer` (str): reference answer

- **`data`** (evaluation set)
  - `title` (str)
  - `paragraphs` (list)
    - `qas` (list)
      - `id` (str): unique question id
      - `question` (str)
      - `answers` (list of objects)
        - `text` (str): ground-truth short answer
        - `document_id` (list[str], optional but recommended): the **chunk IDs** in your vector DB that contain the answer

> **Important:** If you plan to compute **Recall@k**, the `document_id` values here must match the **chunk IDs** you ingested into ChromaDB (see `alue/rag_utils.py`). If you do **not** provide `document_id`, you can still evaluate retrieval via **Context Relevancy** (LLM judge).

### Example Dataset

```json
{
  "examples": [
    {
      "query": "What is the purpose of FAA Order 8040.1C?",
      "context": "This order describes the Federal Aviation Administration's (FAA) authority and assigns responsibility for the development and issuance of Airworthiness Directives (AD) in accordance with applicable statutes and regulations.",
      "answer": "To describe the FAA's authority and assign responsibility for developing and issuing Airworthiness Directives in accordance with applicable statutes and regulations."
    },
    {
      "query": "Which directorates have authority to issue Airworthiness Directives?",
      "context": "The managers of the four Aircraft Certification Directorates (Engine and Propeller, Rotorcraft, Small Airplane, and Transport Airplane) have the authority and responsibility for the aircraft certification programs assigned to them and for the products under their geographical jurisdiction. Directorate managers may redelegate authority to issue ADs for their Directorate to the Assistant Directorate Manager, but no further.",
      "answer": "The Engine and Propeller, Rotorcraft, Small Airplane, and Transport Airplane Certification Directorates have authority to issue ADs, and they may delegate this authority to Assistant Directorate Managers."
    }
  ],
  "data": [
    {
      "title": "FAA Order 8040.1C - Airworthiness Directives",
      "paragraphs": [
        {
          "qas": [
            {
              "id": "0",
              "question": "What is the effective date of FAA Order 8040.1C on Airworthiness Directives?",
              "answers": [
                {
                  "text": "10/03/07",
                  "document_id": ["faa_order_8040_1c"]
                }
              ]
            },
            {
              "id": "1",
              "question": "Which branch is delegated responsibility for assigning AD numbers and final printing and distribution of all ADs?",
              "answers": [
                {
                  "text": "The Aircraft Engineering Division, Delegation and Airworthiness Programs Branch, AIR-140",
                  "document_id": ["faa_order_8040_1c"]
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Next Steps

* [RAG](tasks/rag.md) — Retrieval, chunking, and evaluation (Context Relevancy, Recall@k, Composite Correctness)
* [MCQA](tasks/mcqa.md) — Prompt templates, schema, evaluation
* [Summarization](tasks/summarization.md) — Claim-based evaluation settings
* [Extractive QA](tasks/extractive-qa.md) — Token-level scoring and normalization
```