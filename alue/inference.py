import json
import os
from typing import Dict, List, Any, Optional, Type, Union
from pathlib import Path

from pydantic import BaseModel

from .data_utils import load_data
from .prompt_utils import build_messages
from .model_utils import create_model_engine
from tqdm import tqdm



def run_mcqa_inference(
    messages: List[str],
    model_name: str,
    schema: Optional[Type[BaseModel]] = None,
    field_to_extract: str = "answer",
    temperature: float = 0.1,
    **generation_kwargs
) -> List[str]:
    """
    Execute MCQA inference on a batch of messages.

    Args:
        messages: List of formatted messages ready for inference
        model_name: Model name for inference
        backend_type: Backend type (openai, tgi, etc.)
        schema: Optional Pydantic model for structured output
        field_to_extract: Field name to extract from structured response (default: "answer")
        temperature: Generation temperature
        **generation_kwargs: Additional generation parameters

    Returns:
        List of prediction strings
    """
    engine = create_model_engine(model_name)

    gen_kwargs = {"temperature": temperature, "max_tokens": 150, **generation_kwargs}
    predictions = []

    for message in tqdm(messages):
        try:
            if schema:
                response = engine.generate_structured(message, schema=schema, **gen_kwargs)
                print(f"response: {response}")

                prediction = response[field_to_extract]
            else:
                response = engine.generate_unstructured(message, **gen_kwargs)
                prediction = response

            predictions.append(prediction)
        except Exception as e:
            print(e)
            predictions.append("ERROR")

    return predictions