"""Simple prompt utilities using Jinja2 directly."""

from jinja2 import Template
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_template(filepath: str) -> str:
    """Load template from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def render_prompt(template_str: str, **kwargs) -> str:
    """Render prompt with Jinja2."""
    return Template(template_str).render(**kwargs)


def build_messages(task_type: str, input_data: str, examples: Optional[List[Dict[str, Any]]] = None, **kwargs) -> List[Dict[str, str]]:
    """Build messages format from clean templates.

    Args:
        task_type: Type of task (e.g., 'aviation_exam_clean', 'rag_clean', 'classification_clean')
        input_data: The actual input/question to process
        examples: Optional list of examples to include as instructions
        **kwargs: Additional template variables (e.g., context, domain, instructions, etc.)

    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    template_dir = Path("templates") / task_type
    
    system_template = load_template(template_dir / "system.jinja2")
    user_template = load_template(template_dir / "user.jinja2")

    # Render system prompt with examples as instructions and any additional kwargs
    system_content = render_prompt(system_template, examples=examples, **kwargs)

    # Render user prompt with the actual input and any additional kwargs
    user_content = render_prompt(user_template, input=input_data, **kwargs)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]