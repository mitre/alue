import pytest
import os

from alue.prompt_utils import build_messages


@pytest.fixture(scope="module")
def current_dir():
    return os.path.dirname(__file__)


@pytest.fixture(scope="module")
def rag_examples():
    return [{
        "input": "example_input",
        "output": "example_output",
        "context": "example_context",
    }]


@pytest.fixture(scope="module")
def examples():
    return [{
        "input": "example_input",
        "output": "example_output",
    }]


@pytest.fixture(scope="module")
def classification_examples():
    return [{
        "input": "example_input",
        "output": "example_output",
    }]


def test_rag_prompt(rag_examples):
    # mock examples and input
    question = "test_question"
    context = "test_context"
    message = build_messages(
        task_type="rag",
        system_kwargs={'examples': rag_examples},
        user_kwargs={"query": question, "context": context}
    )

    assert(message[0]['role'] == "system")
    assert(message[0]['content'] == (
        "You are an expert AI assistant that helps answer user queries using ONLY the "
        "information provided in the context.\n\nHere are a few examples:\n\n"
        "Question: example_input\nContext:\n    example_context\n\nAnswer: example_output\n\n"
    ))
    assert(message[1]['role'] == "user")
    assert(message[1]['content'] == "Query: test_question\nContext: test_context")


def test_qa_prompt(examples):
    input = "test_input"
    message = build_messages(
        task_type="aviation_exam",
        system_kwargs={'examples': examples},
        user_kwargs={'input': input}
    )

    print(message)
    assert(message[0]['role'] == "system")
    assert(message[0]['content'] == (
        "Based on the given question and options, select the correct answer. Choose only one "
        "option that best answers the question.\n\n\nHere are some examples:\n\n    "
        "Question: example_input\n    Answer: example_output\n\n\n\n"
        "Choose only one option that best answers the question"
    ))
    assert(message[1]['role'] == "user")
    assert(message[1]['content'] == "Question: test_input")