from typing import List, Optional, Type, Union

from pydantic import BaseModel

from .model_utils import create_model_engine
from tqdm import tqdm


def run_llm_inference(
    messages: List[str],
    model_name: str,
    schema: Optional[Type[BaseModel]] = None,
    fields_to_extract: Optional[Union[str, List[str]]] = None,
    temperature: float = 0.1,
    judge_mode: bool = False,
    **generation_kwargs
) -> List[str]:
    """
    Execute inference using LLM.

    Args:
        messages: List of formatted messages ready for inference
        model_name: Model name for inference
        schema: Optional Pydantic model for structured output
        fields_to_extract: Field name(s) to extract from structured response. 
                          Can be a string for single field, list for multiple fields, 
                          or None for full response
        temperature: Generation temperature
        judge_mode: Use llm judge
        **generation_kwargs: Additional generation parameters

    Returns:
        List of prediction strings
    """
    engine = create_model_engine(model_name, judge_mode=judge_mode)

    gen_kwargs = {"temperature": temperature, "max_tokens": 150, **generation_kwargs}
    predictions = []

    for message in tqdm(messages):
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
            print(e)
            predictions.append("ERROR")

    return predictions