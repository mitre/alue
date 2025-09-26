"""
This module provides utilities for loading Jinja2 templates, rendering prompts,
and building message structures for chat-based language model APIs. It supports
task-specific template organization and flexible variable injection.
"""

from jinja2 import Template
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_template(filepath: str) -> str:
    """Load a Jinja2 template from a file.
    
    Args:
        filepath: Path to the template file to load.
        
    Returns:
        The template content as a string.
        
    Raises:
        FileNotFoundError: If the template file does not exist.
        IOError: If there are issues reading the file.
        
    Example:
        >>> template_str = load_template('templates/qa/system.jinja2')
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def render_prompt(template_str: str, **kwargs: Any) -> str:
    """Render a Jinja2 template string with provided variables.
    
    This function takes a Jinja2 template string and renders it with the
    provided keyword arguments as template variables.
    
    Args:
        template_str: A Jinja2 template string to render.
        **kwargs: Arbitrary keyword arguments to pass as template variables.
        
    Returns:
        The rendered template as a string.
        
        
    Example:
        >>> template = "Hello, {{ name }}!"
        >>> render_prompt(template, name="Alice")
        'Hello, Alice!'
    """
    return Template(template_str).render(**kwargs)


def build_messages(
    task_type: str,
    system_kwargs: Optional[Dict[str, Any]] = None,
    user_kwargs: Optional[Dict[str, Any]] = None,
    shared_kwargs: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """Build a chat message list from task-specific Jinja2 templates.
    
    This function loads system and user templates for a given task type,
    renders them with the provided variables, and returns a structured
    message list suitable for chat-based language model APIs.
    
    The function expects templates to be organized in a directory structure:
    templates/{task_type}/system.jinja2 and templates/{task_type}/user.jinja2
    
    Args:
        task_type: The type of task, used to locate template files in the
            templates directory (e.g., 'qa', 'classification', 'summarization').
        system_kwargs: Variables to use only in the system template. These
            override any shared_kwargs with the same keys. Defaults to None.
        user_kwargs: Variables to use only in the user template. These
            override any shared_kwargs with the same keys. Defaults to None.
        shared_kwargs: Variables to use in both system and user templates.
            These can be overridden by system_kwargs or user_kwargs.
            Defaults to None.
            
    Returns:
        A list of message dictionaries, each containing 'role' and 'content'
        keys. The list always contains exactly two messages: one with role
        'system' and one with role 'user'.
        
        
    Example:
        >>> messages = build_messages(
        ...     task_type='qa',
        ...     shared_kwargs={'examples': example_list},
        ...     user_kwargs={'question': 'What is AI?', 'context': 'AI is...'}
        ... )
        >>> print(messages)
        [
            {'role': 'system', 'content': 'You are a helpful assistant...'},
            {'role': 'user', 'content': 'Question: What is AI?\\nContext: AI is...'}
        ]
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