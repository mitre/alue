
# ALUE: Aerospace Language Understanding and Evaluation

<p align="center">
<a href="https://arc.aiaa.org/doi/10.2514/6.2025-3247">
<img src="https://img.shields.io/badge/Read-Paper-green?style=flat" alt="Read Paper" />
</a>
</p>

ALUE (Aerospace Language Understanding Evaluation) is a comprehensive framework designed to facilitate the evaluation and inference of Language Learning Models (LLMs) on aerospace-specific datasets. The framework is user-friendly and versatile, supporting custom datasets, preferred models, user-defined prompts, and quantitative metrics of performance. Contact Kunal K. Sarkhel (ksarkhel@mitre.org) with inquiries.

## Key Features

- **Multiple Task Types**: MCQA, Summarization, RAG, and Extractive QA
- **Flexible Backends**: Works with OpenAI, vLLM, TGI, Ollama, Transformers, and more
- **Advanced Evaluation**: Combines traditional metrics with LLM-as-judge evaluation

## Quick Start

`alue` relies on multiple python libraries for data processing which uses several OS-level packages such as `poppler` (needed for images and pdfs) and `tesseract` (for OCR on images and pdfs).


##### Ubuntu

```bash
sudo apt update
sudo apt install poppler-utils
sudo apt install tesseract-ocr
```

##### MacOS (through Homebrew)

We will use `Homebrew` to install system packages on MacOS. If you do not already have `Homebrew` installed, you can run the following command to install it. 

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

```bash
brew install poppler
brew install tesseract
```

##### For Windows

The best way to go about using `alue` on Windows would be through Ubuntu on the Windows Subsystem for Linux (WSL).

Once you have Ubuntu running on WSL, you can use the above Ubuntu commands to install the necessary dependancies.

More info on Ubuntu setup on WSL can be found [here](https://documentation.ubuntu.com/wsl/en/latest/guides/install-ubuntu-wsl2/).


### Installation

```bash
# Recommended: using uv
uv sync

# Alternative: using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```bash
cp .env-example .env
```

Minimal configuration for OpenAI:

```env
ALUE_ENDPOINT_TYPE=openai
ALUE_OPENAI_API_KEY=sk-...
```

> **Note:** Additional configuration required for some tasks:
> - **RAG**: Requires `EMBEDDING_*` settings (will default to a local embedder if not specified)
> - **RAG & Summarization**: Require `ALUE_LLM_JUDGE_*` settings for evaluation
> 
> See [Configuration Reference](docs/model-configuration.md) for complete setup.

### Run Your First Task

```bash
# Multiple Choice QA example
python -m scripts.mcqa inference \
  -i data/aviation_knowledge_exam/3_1_aviation_test.json \
  -o runs/mcqa \
  -m gpt-4o-mini \
  --task_type aviation_exam \
  --num_examples 3 \
  --num_questions 2 \
  --schema_class schemas.aviation_exam.schema.MCQAResponse \
  --field_to_extract answer
```

Results saved to `runs/mcqa_<timestamp>/predictions.json`

## Supported Tasks

### Multiple Choice QA (MCQA)
Evaluate model's ability to answer multiple choice questions with a single correct option.

```bash
python -m scripts.mcqa both \
  -i data/mcqa/aviation_exam.json \
  -o runs/mcqa \
  -m gpt-4o-mini \
  --task_type aviation_exam \
  --num_examples 3 \
  --num_questions 2 \
  --schema_class schemas.aviation_exam.schema.MCQAResponse \
  --field_to_extract answer
```

**Metrics**: Accuracy

### Retrieval-Augmented Generation (RAG)

#### Vector Database Set-up

```bash
python -m alue.rag_utils \
  --document-directory ./tests/resources/ \
  --database-path ./chroma_db \
  --collection-name documents \
  --output-path ./artifacts \
  --partition-strategy hi_res \
  --chunk-hard-max 1200 \
  --chunk-soft-max 700 \
  --overlap-size 50
```

#### Run RAG Inference + Evaluation

```bash
python -m scripts.rag both \
  -i data/dummy_rag/rag_qa.json \
  -o runs/rag \
  -m gpt-4o-mini \
  --database-path ./chroma_db \
  --collection-name documents \
  --llm_judge_model_name gpt-4o-mini
```

**Metrics**: Recall@k, Context Relevancy, Composite Correctness

### Summarization
Generate concise summaries of aviation narratives.

```bash
python -m scripts.summarization both \
  -i data/dummy_summarization/data.jsonl \
  -o runs/sum \
  -m gpt-4o-mini \
  --llm_judge_model_name gpt-4o-mini
```

**Metrics**: Precision, Recall, F1 (via claim decomposition)

### Extractive QA
Extract exact text spans from documents that answer questions.

```bash
python -m scripts.extractive_qa both \
  -i data/ntsb_tail_extraction/ntsb_tail_extraction_dummy.json \
  -o runs/extractive_qa \
  -m gpt-4o-mini \
  --task_type ntsb_extract_tail_number
```

**Metrics**: Exact Match, F1 Score

## Supported Backends

### Inference Engines

| Backend | Type | Description |
|---------|------|-------------|
| **OpenAI** | API | Direct OpenAI API access |
| **vLLM** | API/Local | Fast inference with OpenAI-compatible API |
| **TGI** | API | HuggingFace Text Generation Inference |
| **Ollama** | API | Local model serving |
| **Transformers** | Local | Direct HuggingFace transformers |

### Embedding Providers (for RAG)

- OpenAI embeddings
- Ollama embeddings
- HuggingFace embeddings
- Local embeddings (default)
- OpenAI-compatible endpoints

## Configuration Examples

### All-OpenAI (Simplest)
```env
ALUE_ENDPOINT_TYPE=openai
ALUE_OPENAI_API_KEY=sk-...
ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai
ALUE_LLM_JUDGE_OPENAI_API_KEY=sk-...
EMBEDDING_ENDPOINT_TYPE=openai
EMBEDDING_API_KEY=sk-...
```

### Local with Ollama
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

# Separate judge to reduce bias
ALUE_LLM_JUDGE_ENDPOINT_TYPE=openai
ALUE_LLM_JUDGE_OPENAI_API_KEY=sk-...

# Local embeddings (no API costs)
EMBEDDING_ENDPOINT_TYPE=local
```

## Evaluation Modes

All tasks support three modes:

1. **Inference only** - Generate predictions
2. **Evaluation only** - Evaluate existing predictions
3. **Both** - Run inference then evaluation

Example:
```bash
# Inference only
python -m scripts.mcqa inference -i data.json -o runs -m gpt-4o-mini

# Evaluation only
python -m scripts.mcqa evaluation -i data.json -o runs --predictions_file runs/predictions.json

# Both
python -m scripts.mcqa both -i data.json -o runs -m gpt-4o-mini
```

## Additional Configurable Features

### LLM-as-Judge Evaluation

For RAG and Summarization tasks, ALUE uses LLM judges for nuanced evaluation:

- **Context Relevancy**: Assesses whether retrieved chunks are relevant
- **Composite Correctness**: Evaluates answer factuality via claim decomposition
- **Claim-based Scoring**: Precision/Recall/F1 for summaries

### Structured Generation

Use Pydantic schemas to enforce structured outputs:

```bash
python -m scripts.mcqa inference \
  -i data.json \
  -o runs \
  -m gpt-4o-mini \
  --schema_class MCQAResponse \
  --field_to_extract answer
```

### Custom Templates

Customize prompts via Jinja2 templates in `templates/<task>/`:
- `system.jinja2` - System message with few-shot examples
- `user.jinja2` - User query template

### Vector Database for RAG

Build ChromaDB from PDFs:

```bash
python -m alue.rag_utils \
  --document-directory ./docs_pdfs \
  --database-path ./chroma_db \
  --collection-name documents \
  --partition-strategy hi_res
```

## Documentation

- **[Getting Started](https://mitre.github.io/alue/getting-started)** - Installation and setup
- **[Configuration Reference](https://mitre.github.io/alue/model-configuration)** - Complete config guide
- **[Models & Backends](https://mitre.github.io/alue/running-models)** - Backend comparison
- **[Creating Datasets](https://mitre.github.io/alue/creating-datasets)** - Dataset format specs
- **Tasks**:
  - [MCQA](https://mitre.github.io/alue/tasks/mcqa)
  - [RAG](https://mitre.github.io/alue/tasks/rag)
  - [Summarization](https://mitre.github.io/alue/tasks/summarization)
  - [Extractive QA](https://mitre.github.io/alue/tasks/extractive-qa)



## Requirements

- Python 3.10+ (3.11 partially tested)
- Dependencies managed via `uv` or `pip`
- API keys for chosen backend(s)
- Optional: GPU for local model inference


## Contributing

ALUE is an open-science initiative and community contributions will help us further ALUE and LLM usage on aerospace data. We welcome:

    🔧 New Tools: Specialized analysis functions and algorithms
    📊 Datasets: Curated aerospace data and knowledge bases
    💻 Software: Integration of existing LLM evaluation software packages
    📋 Benchmarks: Evaluation datasets and performance metrics
    📚 Misc: Tutorials, examples, and use cases
    🔧 Update existing tools: many current tools, benchmarks, metrics are not optimized - fixes and replacements are welcome!

Check out the [Contributing Guide](CONTRIBUTING.md) on how to contribute to the ALUE project.

If you have particular tool/database/software in mind that you want to add, you can also submit to this form and the ALUE team may implement them depending on our resource constraints.

## Support

- **Documentation**: [Full docs](https://mitre.github.io/alue/)

---

**Note**: ALUE has been tested primarily on Python 3.10. Other versions may work but are not officially supported.



## Citation

You can cite this work using either:

Eugene Mangortey, Satyen Singh, Shuo Chen and Kunal Sarkhel. "Aviation Language Understanding Evaluation (ALUE) – Large Language Model Benchmark with Aviation Datasets," AIAA 2025-3247. _AIAA AVIATION FORUM AND ASCEND 2025._ July 2025.

It can also be cited using BibTeX.

```
@inbook{doi:10.2514/6.2025-3247,
author = {Eugene Mangortey and Satyen Singh and Shuo Chen and Kunal Sarkhel},
title = {Aviation Language Understanding Evaluation (ALUE) – Large Language Model Benchmark with Aviation Datasets},
booktitle = {AIAA AVIATION FORUM AND ASCEND 2025},
chapter = {},
pages = {},
doi = {10.2514/6.2025-3247},
URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2025-3247},
eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2025-3247},
    abstract = { Large Language Models (LLMs) present revolutionary potential for the aviation industry, enabling stakeholders to derive critical intelligence and improve operational efficiencies through automation. However, given the safety-critical nature of aviation, a rigorous domain-specific evaluation of LLMs is paramount before their integration into workflows. General-purpose LLM benchmarks often do not capture the nuanced understanding of aerospace-specific knowledge and the phraseology required for reliable application. This paper introduces the Aerospace Language Understanding Evaluation (ALUE) benchmark, an aviation-specific framework designed for scalable evaluation, assessment, and benchmarking of LLMs against specialized aviation datasets and language tasks. ALUE incorporates diverse datasets and tasks, including binary and multiclass classification for hazard identification, extractive question answering for precise information retrieval (e.g., tail numbers, runways), sentiment analysis, and multiclass token classification for fine-grained analysis of air traffic control communications. ALUE also introduces several metrics for evaluating the correctness of generated responses utilizing LLMs to identify and judge claims made in generated responses. Our findings demonstrate that structured prompts and in-context examples significantly improve model performance, highlighting that general models struggle with aviation tasks without such guidance and often produce verbose or unstructured outputs. ALUE provides a crucial tool for guiding the development and safe deployment of LLMs tailored to the unique demands of the aviation and aerospace domains. }
}
```

If you'd like to cite this work using a citation manager such as [Zotero](https://www.zotero.org/), please see the [CITATION.cff](CITATION.cff) file for more information citing this work. For more informaton on the CITATION.cff file format, please see [What is a `CITATION.cff` file?](https://citation-file-format.github.io/)

# Copyright Notices

This is the copyright work of The MITRE Corporation, and was produced for the U. S. Government under Contract Number 693KA8-22-C-00001, and is subject to Federal Aviation Administration Acquisition Management System Clause 3.5-13, Rights In Data-General (Oct. 2014), Alt. III and Alt. IV (Jan. 2009). No other use other than that granted to the U. S. Government, or to those acting on behalf of the U. S. Government, under that Clause is authorized without the express written permission of The MITRE Corporation. For further information, please contact The MITRE Corporation, Contracts Management Office, 7515 Colshire Drive, McLean, VA 22102-7539, (703) 983-6000.

&copy; 2025 The MITRE Corporation. All Rights Reserved.

Approved for Public Release, Distribution Unlimited. PRS Case 25-2078.
