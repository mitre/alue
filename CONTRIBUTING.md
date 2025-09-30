This is the copyright work of The MITRE Corporation, and was produced for the U. S. Government under Contract Number 693KA8-22-C-00001, and is subject to Federal Aviation Administration Acquisition Management System Clause 3.5-13, Rights In Data-General, Alt. III and Alt. IV (Jan 2009). No other use other than that granted to the U. S. Government, or to those acting on behalf of the U. S. Government, under that Clause is authorized without the express written permission of The MITRE Corporation. For further information, please contact The MITRE Corporation, Contracts Management Office, 7515 Colshire Drive, McLean, VA 22102-7539, (703) 983-6000. 2024 The MITRE Corporation. © All Rights Reserved.


This guide will help you understand how to extend ALUE with new tasks, backends, evaluation metrics, and more.

---

## Getting Started

1. **Fork and clone the repository**
2. **Set up your development environment:**
   ```bash
   uv sync
   # or
   pip install -r requirements.txt
   ```
3. **Run tests to ensure everything works:**
   ```bash
   pytest tests
   ```

---

## Project Structure

Understanding the project structure will help you contribute effectively:

```
alue/
├── alue/                          # Core library
│   ├── inference.py              # Inference engine interfaces
│   ├── evaluation.py             # Evaluation metrics
│   ├── llm_judge_metrics.py      # LLM-as-judge evaluation
│   ├── data_utils.py             # Dataset loading utilities
│   ├── rag_utils.py              # RAG-specific utilities (ChromaDB, embeddings)
│   ├── prompt_utils.py           # Prompt template handling
│   ├── settings.py               # Configuration management (pydantic-settings)
│   ├── squad_evaluation.py       # SQuAD-style metrics
│   ├── doc_retrieval_metrics.py  # Retrieval evaluation metrics
│   └── output_normalizations.py  # Output normalization utilities
├── scripts/                       # Task-specific entry points
│   ├── mcqa.py                   # Multiple choice QA
│   ├── rag.py                    # Retrieval-augmented generation
│   ├── summarization.py          # Narrative summarization
│   ├── extractive_qa.py          # Extractive QA
│   ├── binary_classification.py
│   ├── sequence_classification.py
│   ├── token_classification_ner.py
│   └── utils.py                  # Shared script utilities
├── templates/                     # Jinja2 prompt templates per task
│   ├── rag/
│   │   ├── system.jinja2
│   │   └── user.jinja2
│   ├── aviation_exam/
│   ├── summarization/
│   └── ntsb_extract_tail_number/
├── schemas/                       # Pydantic schemas for structured generation
│   ├── aviation_exam/
│   │   └── schema.py
│   ├── extractive_qa/
│   │   └── schema.py
│   └── ntsb_extract_tail_number/
│       └── schema.py
├── data/                          # Example datasets
│   ├── aviation_knowledge_exam/
│   ├── ASRS_rag/
│   ├── asrs_summarization/
│   └── ntsb_tail_extraction/
├── docs/                          # Documentation (MkDocs)
│   ├── index.md
│   ├── getting-started.md
│   ├── model-configuration.md
│   ├── creating-datasets.md
│   └── tasks/
│       ├── rag.md
│       ├── mcqa.md
│       ├── summarization.md
│       └── extractive-qa.md
└── tests/                         # Test suite
    ├── test_inference.py
    ├── test_rag_utils.py
    ├── test_data_utils.py
    ├── test_prompt_utils.py
    └── resources/                # Test data
```

---

## Adding a New Task

To add a new task type to ALUE:

### 1. Create the Task Script

Create a new script in `scripts/<task_name>.py` following this pattern:

```python
import argparse
from pathlib import Path
from alue.data_utils import load_data
from alue.inference import run_inference
from alue.evaluation import evaluate_<task>
from alue.prompt_utils import build_messages

def inference(args):
    """Run inference for the task."""
    # Load dataset
    dataset = load_data(args.input_data_json_path, task_type=args.task_type)
    examples = dataset.get_examples(args.num_examples)
    test_data = dataset.get_test_data()
    
    # Run inference
    predictions = {}
    for item in test_data:
        messages = build_messages(
            task_type=args.task_type,
            system_kwargs={"examples": examples},
            user_kwargs={"input": item["input"]}
        )
        prediction = run_inference(messages, args.model_name)
        predictions[item["id"]] = prediction
    
    # Save predictions
    save_predictions(predictions, args.output_dir)

def evaluation(args):
    """Run evaluation for the task."""
    # Load predictions and ground truth
    predictions = load_predictions(args.predictions_file)
    dataset = load_data(args.input_data_json_path, task_type=args.task_type)
    
    # Evaluate
    metrics = evaluate_<task>(predictions, dataset)
    
    # Save metrics
    save_metrics(metrics, args.output_dir)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    
    # Add inference, evaluation, and both subcommands
    # See existing scripts (mcqa.py, rag.py, etc.) for the full pattern
    
    args = parser.parse_args()
    
    if args.mode == 'inference':
        inference(args)
    elif args.mode == 'evaluation':
        evaluation(args)
    elif args.mode == 'both':
        inference(args)
        evaluation(args)

if __name__ == "__main__":
    main()
```

### 2. Create Prompt Templates

Create a directory `templates/<task_name>/` with two files:

**`system.jinja2`** - System prompt with few-shot examples:
```jinja2
You are an assistant that performs <task description>.

{% if examples %}
Here are some examples:
{% for example in examples %}
    Input: {{ example.input }}
    Output: {{ example.output }}
{% endfor %}
{% endif %}
```

**`user.jinja2`** - User message template:
```jinja2
Input: {{ input }}
```

### 3. Create Dataset Format Specification

Document your dataset format in `docs/creating-datasets.md` following the existing patterns. Include:
- Purpose of the task
- File format (JSON/JSONL)
- Complete schema with all required and optional fields
- Example dataset with real data
- Notes about special requirements

### 4. Add Evaluation Metrics

If your task requires custom evaluation metrics, add them to `alue/evaluation.py`:

```python
class TaskEval:
```

### 5. Add Task Documentation

Create `docs/tasks/<task_name>.md` following the structure of existing task pages:
1. Brief introduction (what the task does)
2. Dataset format with example
3. Quick start section
4. Running the task (inference/evaluation/both modes)
5. Templates and variables reference
6. Evaluation metrics explanation
7. Configuration notes
8. Troubleshooting section

### 6. Update Documentation Index

1. Add your task to `docs/tasks/index.md`
2. Update the summary table in `docs/creating-datasets.md`
3. Update `mkdocs.yml` to include the new task page in navigation

---

## Adding a New Backend

To add support for a new inference backend:

### 1. Update Settings

Add new configuration variables to `alue/settings.py`:

```python
class Settings(BaseSettings):
    # ... existing settings
    
    # New backend settings
    new_backend_endpoint: Optional[str] = None
    new_backend_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
```

### 2. Implement Backend Interface

Add backend support to `alue/model_utils.py`:

```python
class NewEngine(BaseInferenceEngine):
```

### 3. Update Documentation

- Add backend to `docs/running-models.md` with:
  - Description and use case
  - Setup/installation instructions
  - Configuration requirements
  - Any special considerations
- Add configuration example to `docs/model-configuration.md`
- Update the backend comparison table
- Add to `.env.example` with commented example values

---

## Adding Structured Generation Schemas

To add a new schema for structured output:

### 1. Create Schema Module

Create `schemas/<task_name>/schema.py`:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class AnswerChoice(str, Enum):
    """Valid answer choices for the task."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"

class TaskResponse(BaseModel):
    """Response schema for <task_name>."""
    answer: AnswerChoice = Field(description="The selected answer choice")
    confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1"
    )
    
    class Config:
        use_enum_values = True
```

> **Note:** Some legacy schemas may be in JSON format (`output_schema.json`), but new schemas should be Python Pydantic models for better type safety and validation.

### 2. Use Schema in Task Script

```python
from schemas.<task_name>.schema import TaskResponse

# In your inference function
prediction = run_inference(
    messages, 
    model_name,
    schema_class=TaskResponse,
    field_to_extract="answer"
)
```

### 3. Document Schema Usage

Add schema documentation to your task's documentation page in `docs/tasks/<task_name>.md`, including:
- When to use the schema
- How to specify it via command-line arguments
- What fields can be extracted
- Example output format


---

## Utility Modules

ALUE includes several utility modules in `alue/`:

- **`squad_evaluation.py`** - SQuAD-style F1 and Exact Match metrics for extractive QA
- **`doc_retrieval_metrics.py`** - Recall@k and retrieval evaluation for RAG tasks
- **`llm_judge_metrics.py`** - LLM-as-judge evaluation (Context Relevancy, Composite Correctness, Claim Decomposition)
- **`output_normalizations.py`** - Pattern-based output normalization for classification tasks
- **`prompt_utils.py`** - Jinja2 template rendering and message construction
- **`data_utils.py`** - Dataset loading and validation

When adding new evaluation metrics or utilities, consider whether they belong in one of these existing modules or if a new module is needed.

---

## Documentation Structure

ALUE uses [MkDocs](https://www.mkdocs.org/) for documentation. The configuration is in `mkdocs.yml`.

### Preview Documentation Locally

```bash
mkdocs serve
```

Then visit `http://127.0.0.1:8000` in your browser.

### Adding New Documentation

1. Create `.md` files in `docs/`
2. Update `mkdocs.yml` to include new pages in the navigation structure
3. Follow the existing documentation structure and style:
   - Use clear headers with proper hierarchy
   - Include code examples with syntax highlighting
   - Add tables for structured information
   - Use admonitions (notes, warnings) where appropriate
   - Link to related documentation pages

### Documentation Style Guide

- Use sentence case for headers
- Include concrete examples for all features
- Provide both minimal and complete examples
- Add troubleshooting sections for common issues
- Keep navigation shallow (prefer fewer nesting levels)

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests
```

### Test Resources

Test data should be placed in `tests/resources/`. Currently includes:
- Sample PDF documents for RAG testing
- Example datasets for various tasks

### Adding Tests

When adding new functionality, include tests in `tests/`:

```python
import pytest
from alue.<module> import <function>

def test_<functionality>():
    """Test description of what is being tested."""
    # Arrange
    input_data = ...
    expected_output = ...
    
    # Act
    result = <function>(input_data)
    
    # Assert
    assert result == expected_output

def test_<functionality>_error_case():
    """Test that appropriate errors are raised."""
    with pytest.raises(ValueError):
        <function>(invalid_input)
```

---

## Documentation

When contributing, please:

1. **Update relevant documentation** in `docs/`
2. **Add code comments** for complex logic
3. **Include docstrings** with examples for public functions
4. **Update the README** if adding major features
5. **Verify documentation builds** with `mkdocs serve`

---

## Pull Request Process

1. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits
   ```bash
   git commit -m "Add feature X: brief description"
   ```

3. **Add/update tests** for your changes
   - Ensure all tests pass: `pytest tests`
   - Add new tests for new functionality

4. **Update documentation** as needed
   - Update relevant docs in `docs/`
   - Add examples and usage instructions
   - Update `mkdocs.yml` if adding new pages

5. **Run pre-commit hooks** (if configured)
   ```bash
   pre-commit run --all-files
   ```

6. **Submit a pull request** with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable
   - Notes on any breaking changes

### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Changes Made
- List key changes
- One per line

## Testing
- [ ] Added new tests
- [ ] All tests passing
- [ ] Tested locally

## Documentation
- [ ] Updated relevant docs
- [ ] Added examples
- [ ] Updated CHANGELOG.md

## Related Issues
Fixes #<issue_number>
```

---

## Development Workflow

### Setting Up for Development

1. Clone and install in editable mode:
   ```bash
   git clone <repository>
   cd alue
   uv sync  # or pip install -e .
   ```

2. Set up pre-commit hooks (if using):
   ```bash
   pre-commit install
   ```

3. Create your `.env` file:
   ```bash
   cp .env.example .env
   # Edit with your API keys
   ```

### Making Changes

1. Create a feature branch
2. Make changes with frequent commits
3. Write/update tests as you go
4. Update documentation alongside code changes
5. Run tests frequently: `pytest tests`

### Before Submitting

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Code follows style guidelines
- [ ] No unnecessary files in commit (check `.gitignore`)
- [ ] CHANGELOG.md is updated

---