import json
import random
from collections.abc import Generator

import pandas as pd


def get_classification_input(data: dict) -> Generator:
    """
    Extracts samples from the dataset.
    ...

    Parameters
    ----------
    data : dict
        The dataset in dictionary format.

    Yields
    ------
    ...
    Generator
        A generator containing sample id, text input, task specification, task background, and examples.
    """
    for dataset in data["data"]:
        title = dataset["title"]  # noqa: F841
        classification_task = dataset["classification_task"]
        task_background = dataset["task_background"]
        examples = dataset["examples"]
        for qas in dataset["qas"]:
            yield {
                "id": qas["id"],
                "classification_input": qas["text_input"],
                "classification_task": classification_task,
                "task_background": task_background,
                "examples": examples,
            }


def get_classification_examples(
    input_json_data_path: str,
    nbr_examples: int = 10,
    randomize_selection: bool = False,
    randomize_order: bool = False,
    random_seed: int = 49,
) -> list:
    """
    Get list of all examples from dataset
    ...

    Parameters
    ----------
    input_json_data_path : str
        path to json dataset
    ...

    Returns
    -------
    list
        list of examples from dataset
    """
    with open(input_json_data_path) as file:
        data = json.load(file)
        examples = data["data"][0]["examples"]
        # randomly select nbr_examples examples if randomize is set
        if nbr_examples < len(examples):
            if randomize_selection:
                random.seed(random_seed)
                examples = random.sample(examples, nbr_examples)
            else:
                examples = examples[0:nbr_examples]

            # randomize order of examples if randomize is set
            if randomize_order:
                random.seed(random_seed)
                examples = random.sample(examples, k=len(examples))

        return examples


def get_classification_inputs_list(input_json_data_path: str) -> list:
    """
    Get list of all articles from dataset
    ...

    Parameters
    ----------
    input_json_data_path : str
        path to squad-like json dataset
    ...

    Returns
    -------
    list
        list of articles from dataset
    """
    with open(input_json_data_path) as file:
        data = json.load(file)

    sample_iterable = get_classification_input(data=data)
    prompts_list = list(sample_iterable)

    return prompts_list


def get_qa_input(data: dict) -> Generator:
    """
    Extracts articles from the dataset.

    Parameters
    ----------
    data : dict
        The dataset in dictionary format.

    Yields
    ------
    Generator
        A generator containing article id, question, context.
    """
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qas in paragraph["qas"]:
                question = qas["question"]
                q_id = qas["id"]
                yield {"id": q_id, "question": question, "context": context}


def get_qa_examples(
    input_json_data_path: str,
    nbr_examples: int = 10,
    randomize_selection: bool = False,
    randomize_order: bool = False,
    random_seed: int = 49,
) -> list:
    """
    Get list of all examples from dataset
    ...

    Parameters
    ----------
    input_json_data_path : str
        path to json dataset
    ...

    Returns
    -------
    list
        list of examples from dataset
    """
    with open(input_json_data_path) as file:
        data = json.load(file)
        for dataset in data["data"]:
            examples = dataset["examples"]

            if nbr_examples < len(examples):
                if randomize_selection:
                    # randomly select nbr_examples examples if randomize is set
                    random.seed(random_seed)
                    examples = random.sample(examples, nbr_examples)
                else:
                    # take the first nbr_examples examples if randomize not set
                    examples = examples[0:nbr_examples]

            if randomize_order:
                random.seed(random_seed)
                examples = random.sample(examples, k=len(examples))

            return examples


def get_qa_inputs_list(input_json_data_path: str) -> list:
    """
    Get list of all articles from dataset

    Parameters
    ----------
    input_json_data_path : str
        path to squad-like json dataset

    Returns
    -------
    list
        list of articles from dataset
    """
    with open(input_json_data_path) as file:
        data = json.load(file)

    squad_iterable = get_qa_input(data=data)
    prompts_list = list(squad_iterable)

    return prompts_list


def get_rag_input(data: dict) -> Generator:
    """
    Extracts questions and answers from the dataset.

    Parameters
    ----------
    data : dict
        The dataset in dictionary format.

    Yields
    ------
    Generator
        A generator containing question id, question, and answer.
    """
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qas in paragraph["qas"]:
                question = qas["question"]
                q_id = qas["id"]
                for answer in qas["answers"]:
                    yield {
                        "id": q_id,
                        "question": question,
                        "answer": answer["text"],
                        "document_id": answer["document_id"],
                    }


def get_rag_examples(
    input_json_data_path: str,
    nbr_examples: int = 10,
    randomize_selection: bool = False,
    randomize_order: bool = False,
    random_seed: int = 49,
):
    with open(input_json_data_path) as file:
        data = json.load(file)
    examples = data["examples"]
    if nbr_examples < len(examples):
        if randomize_selection:
            # randomly select nbr_examples examples if randomize is set
            random.seed(random_seed)
            examples = random.sample(examples, nbr_examples)
        else:
            # take the first nbr_examples examples if randomize not set
            examples = examples[0:nbr_examples]

    if randomize_order:
        random.seed(random_seed)
        examples = random.sample(examples, k=len(examples))

    return examples


def get_rag_inputs_list(input_json_data_path: str) -> list:
    """
    Get list of all questions and answers from dataset

    Parameters
    ----------
    input_json_data_path : str
        path to squad-like json dataset

    Returns
    -------
    list
        list of questions and answers from dataset
    """
    with open(input_json_data_path) as file:
        data = json.load(file)
    squad_iterable = get_rag_input(data=data)
    prompts_list = list(squad_iterable)
    return prompts_list


def get_definitions_terms_dataset(input_excel_sheet: str) -> list[dict[str, str]]:
    """
    Get list of all terms and definitions from dataset

    Parameters
    ----------
    input_excel_sheet : str
    """
    # Load the Excel file
    df = pd.read_excel(input_excel_sheet, index_col=None)

    glossary_terms_dataset = []

    for index, row in df.iterrows():
        term = row["question"].split(":")[1].strip(".")
        definition = row["answer"]
        glossary_terms_dataset.append(
            {"id": index, "term": term, "ground_truth_definition": definition}
        )
    return glossary_terms_dataset


def get_mcqa_dataset(input_json_data_path: str) -> list[dict[str, str]]:
    """
    Get list of all terms and definitions from dataset

    Parameters
    ----------
    input_json_data_path : str
        path to json dataset

    Returns
    -------
    list
        list of questions and answers from dataset
    """
    with open(input_json_data_path) as file:
        data = json.load(file)["data"]

    return data


def get_mcqa_examples(
    input_json_data_path: str,
    nbr_examples: int = 10,
    randomize_selection: bool = False,
    randomize_order: bool = False,
    random_seed: int = 49,
) -> list:
    """
    Get list of all examples from dataset
    ...

    Parameters
    ----------
    input_json_data_path : str
        path to json dataset
    ...

    Returns
    -------
    list
        list of examples from dataset
    """
    with open(input_json_data_path) as file:
        examples = json.load(file)["examples"]

        if nbr_examples < len(examples):
            if randomize_selection:
                # randomly select nbr_examples examples if randomize is set
                random.seed(random_seed)
                examples = random.sample(examples, nbr_examples)
            else:
                # take the first nbr_examples examples if randomize not set
                examples = examples[0:nbr_examples]

            if randomize_order:
                random.seed(random_seed)
                examples = random.sample(examples, k=len(examples))

        return examples


def get_summarization_dataset(
    input_json_data_path: str,
    load_examples=False,
    nbr_examples: int = 10,
    randomize_selection: bool = False,
    randomize_order: bool = False,
    random_seed: int = 49,
):
    with open(input_json_data_path) as f:
        data = [json.loads(line) for line in f]

    if load_examples:
        filepath_split = input_json_data_path.split("/")
        filepath_split[-1] = "examples.jsonl"
        examples_path = "/".join(filepath_split)
        with open(examples_path) as f:
            data = [json.loads(line) for line in f]

        if nbr_examples < len(data):
            if randomize_selection:
                # randomly select nbr_examples examples if randomize is set
                random.seed(random_seed)
                data = random.sample(data, nbr_examples)
            else:
                # take the first nbr_examples examples if randomize not set
                data = data[0:nbr_examples]

            if randomize_order:
                random.seed(random_seed)
                data = random.sample(data, k=len(data))

    return data


def get_binary_classification_dataset(
    input_json_data_path: str,
    load_examples=False,
    nbr_examples: int = 5,
    randomize_selection: bool = False,
    randomize_order: bool = False,
    random_seed: int = 49,
):
    """
    Load JSON data and optionally sample examples while maintaining class distribution.

    Args:
        input_json_data_path (str): Path to the input JSON file.
        load_examples (bool): Whether to load examples from a separate file.
        nbr_examples (int): Number of examples to sample.
        randomize_selection (bool): Whether to randomly select examples.
        randomize_order (bool): Whether to randomize the order of examples.
        random_seed (int): Seed for randomization.

    Returns
    -------
        List[Dict]: A list of dictionaries with fields "id", "text", and "label".
    """
    # Load the main JSON data
    with open(input_json_data_path) as f:
        data = json.load(f)

    # Load examples if specified
    if load_examples:
        filepath_split = input_json_data_path.split("/")
        filepath_split[-1] = "examples.json"
        examples_path = "/".join(filepath_split)
        with open(examples_path) as f:
            data = json.load(f)

        # Separate data into positive and negative examples
        positive_examples = {
            key: value for key, value in data.items() if value["redacted"]
        }
        negative_examples = {
            key: value for key, value in data.items() if not value["redacted"]
        }

        # Calculate proportions
        total_examples = len(data)
        positive_ratio = len(positive_examples) / total_examples
        # negative_ratio = len(negative_examples) / total_examples

        # Determine number of samples for each class
        positive_nbr = int(nbr_examples * positive_ratio)
        negative_nbr = (
            nbr_examples - positive_nbr
        )  # Remaining examples go to the negative class

        # Sample from each class
        random.seed(random_seed)
        if randomize_selection:
            positive_sampled_keys = random.sample(
                list(positive_examples.keys()),
                min(positive_nbr, len(positive_examples)),
            )
            negative_sampled_keys = random.sample(
                list(negative_examples.keys()),
                min(negative_nbr, len(negative_examples)),
            )
        else:
            positive_sampled_keys = list(positive_examples.keys())[
                : min(positive_nbr, len(positive_examples))
            ]
            negative_sampled_keys = list(negative_examples.keys())[
                : min(negative_nbr, len(negative_examples))
            ]

        # Combine sampled examples
        sampled = {key: positive_examples[key] for key in positive_sampled_keys}
        sampled.update({key: negative_examples[key] for key in negative_sampled_keys})

        # Randomize order if specified
        if randomize_order:
            random.seed(random_seed)
            randomized_keys = random.sample(list(sampled.keys()), len(sampled))
            sampled = {key: sampled[key] for key in randomized_keys}
    else:
        sampled = data  # If examples are not loaded, use the full data

    # Transform the sampled data into the desired format
    transformed_data = [
        {"id": key, "text": value["text"], "label": value["label"]}
        for key, value in sampled.items()
    ]

    return transformed_data
