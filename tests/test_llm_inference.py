import pytest
import os

from alue.inference import run_llm_inference
from alue.prompt_utils import build_messages

@pytest.fixture(scope="module")
def examples():
    return [{
        "input": "example_input",
        "output": "example_output",
    }]


@pytest.fixture(scope="module")
def prompt(examples):
    input = "test_input"
    return build_messages(
        task_type="aviation_exam",
        system_kwargs={'examples': examples},
        user_kwargs={'input': input}
    )

# def test_trivial_inference(prompt):
#     predictions = run_llm_inference(
#         messages=[prompt],
#         model_name=args.inference_model_name,
#     )

#     assert(predictions != None)