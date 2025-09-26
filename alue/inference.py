"""LLM inference utilities for batch processing with structured output support.

This module provides a high-level interface for running inference across
multiple messages with support for both structured (JSON) and unstructured
(text) outputs. It handles field extraction from structured responses and
provides progress tracking via tqdm.
"""

from typing import List, Optional, Type, Union, Dict, Any

from pydantic import BaseModel

from .model_utils import create_model_engine
from tqdm import tqdm


def run_llm_inference(
    messages: List[List[Dict[str, str]]],
    model_name: str,
    schema: Optional[Type[BaseModel]] = None,
    fields_to_extract: Optional[Union[str, List[str]]] = None,
    temperature: float = 0.1,
    judge_mode: bool = False,
    **generation_kwargs: Any
) -> List[Union[str, Dict[str, Any], Any]]:
    """Execute inference using a language model with optional structured output.
    
    This function processes a list of messages through an LLM, optionally
    enforcing structured JSON output via a Pydantic schema. It supports
    extracting specific fields from structured responses and provides progress
    tracking for large inputs.
    
    Args:
        messages: List of message lists, where each message list contains
            dictionaries with 'role' and 'content' keys in the format:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        model_name: Model identifier or path for inference (e.g., 'gpt-4',
            'meta-llama/Llama-2-7b-hf').
        schema: Optional Pydantic model class defining the expected JSON
            structure for structured generation. If None, generates unstructured
            text. Defaults to None.
        fields_to_extract: Specifies which field(s) to extract from structured
            responses. Options:
            - None: Return the full structured response dictionary
            - str: Extract a single field by name (e.g., 'answer')
            - List[str]: Extract multiple fields as a dictionary
            Ignored when schema is None. Defaults to None.
        temperature: Sampling temperature for generation. Lower values (0.0-0.3)
            produce more deterministic outputs, higher values (0.7-1.0) increase
            randomness. Defaults to 0.1.
        judge_mode: If True, use judge-specific model configuration and
            credentials. Useful for evaluation or scoring tasks. Defaults to False.
        **generation_kwargs: Additional generation parameters to pass to the
            model engine (e.g., max_tokens, top_p, frequency_penalty).
            
    Returns:
        List of predictions with length equal to input messages. Each element's
        type depends on the extraction configuration:
        - Unstructured (no schema): str
        - Structured with no extraction: Dict[str, Any]
        - Structured with single field: Any (type depends on field)
        - Structured with multiple fields: Dict[str, Any]
        - On error: str "ERROR"
        
    Raises:
        ValueError: If fields_to_extract is not None, str, or List[str].
        
    Examples:
        >>> # Unstructured generation
        >>> messages = [[
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "What is AI?"}
        ... ]]
        >>> results = run_llm_inference(messages, "gpt-4")
        >>> print(results[0])  # String response
        
        >>> # Structured generation with field extraction
        >>> from pydantic import BaseModel
        >>> class Answer(BaseModel):
        ...     answer: str
        ...     confidence: float
        >>> results = run_llm_inference(
        ...     messages,
        ...     "gpt-4",
        ...     schema=Answer,
        ...     fields_to_extract="answer"
        ... )
        >>> print(results[0])  # Just the answer string
        
        >>> # Multiple field extraction
        >>> results = run_llm_inference(
        ...     messages,
        ...     "gpt-4",
        ...     schema=Answer,
        ...     fields_to_extract=["answer", "confidence"]
        ... )
        >>> print(results[0])  # {"answer": "...", "confidence": 0.95}
        
    Note:
        - Errors during inference are caught and replaced with "ERROR" string
          to prevent inference from failing completely
        - Default max_tokens is set to 150 but can be overridden via generation_kwargs
    """

    engine = create_model_engine(model_name, judge_mode=judge_mode)

    gen_kwargs = {"temperature": temperature, "max_tokens": 150, **generation_kwargs}
    predictions = []

    for message in tqdm(messages, desc="Processing messages"):
        try:
            if schema:
                response = engine.generate_structured(message, schema=schema, **gen_kwargs)
                print(f"response: {response}")

                if fields_to_extract is None:
                    prediction = response  # Return full response
                elif isinstance(fields_to_extract, str):
                    prediction = response[fields_to_extract]  # Single field
                elif isinstance(fields_to_extract, list):
                    prediction = {field: response.get(field) for field in fields_to_extract}  # Multiple fields
                else:
                    prediction = response
            else:
                response = engine.generate_unstructured(message, **gen_kwargs)
                prediction = response

            predictions.append(prediction)
        except Exception as e:
            print(f"Error processing message: {e}")
            predictions.append("ERROR")

    return predictions