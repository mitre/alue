import pytest
import os

from alue.inference import run_llm_inference
from alue.prompt_utils import build_messages

@pytest.fixture(scope="module")
def examples():
    return [{
        "input": "What is the unit of measure for airspeed?\r\nA) knots\r\nB) AOA\r\nC) rate of climb",
        "output": "A",
    }]


@pytest.fixture(scope="module")
def prompt(examples):
    input = "What is the unit of measure for altitude?\r\nA) feet\r\nB) feet per minute\r\nC) flight level\r\nD) both A and C"
    return build_messages(
        task_type="aviation_exam",
        system_kwargs={'examples': examples},
        user_kwargs={'input': input}
    )



def test_trivial_inference(prompt):
    predictions = run_llm_inference(
        messages=[prompt],
        model_name="nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    )

    print(predictions)
    assert(predictions != "")