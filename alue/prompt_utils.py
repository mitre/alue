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


def build_messages(
    task_type: str,
    system_kwargs: Optional[Dict[str, Any]] = None,
    user_kwargs: Optional[Dict[str, Any]] = None,
    shared_kwargs: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """Build messages format from clean templates.

    Args:
        task_type: Type of task
        system_kwargs: Variables only for system template
        user_kwargs: Variables only for user template
        shared_kwargs: Variables for both templates

    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    template_dir = Path("templates") / task_type
    
    system_template = load_template(template_dir / "system.jinja2")
    user_template = load_template(template_dir / "user.jinja2")

    # Build kwargs for each template
    system_vars = {}
    if shared_kwargs:
        system_vars.update(shared_kwargs)
    if system_kwargs:
        system_vars.update(system_kwargs)

    user_vars = {}
    if shared_kwargs:
        user_vars.update(shared_kwargs)
    if user_kwargs:
        user_vars.update(user_kwargs)

    # Render templates
    system_content = render_prompt(system_template, **system_vars)
    user_content = render_prompt(user_template, **user_vars)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]