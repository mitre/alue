import json

import pandas as pd


def convert_df_to_sequence_classification_qa_json(
    df: pd.DataFrame,
    title: str,
    classification_task: str,
    task_background: str = None,
    examples: str = None,
    version: str = "classification_qa_v1",
    output_file_path: str = "data.json",
) -> dict:
    qas = []
    for id, row in df.iterrows():
        qas.append(
            {
                "id": id,
                "text_input": (
                    [row.preceding_transcription, row.transcription]
                    if "preceding_transcription" in row.index
                    else [row.transcription]
                ),
                "labels": str(row.label),
            }
        )
    data = {
        "title": title,
        "classification_task": classification_task,
        "task_background": task_background,
        "examples": examples,
        "qas": qas,
    }
    json_data = {"version": version, "data": [data]}

    with open(output_file_path, "w") as f:
        json.dump(json_data, f)

    return json_data


def convert_df_to_token_classification_qa_json(
    df: pd.DataFrame,
    title: str,
    classification_task: str,
    task_background: str = None,
    examples: str = None,
    version: str = "classification_qa_v1",
    output_file_path: str = "data.json",
) -> dict:
    qas = []
    for id, row in df.iterrows():
        qas.append(
            {
                "id": id,
                "text_input": [row.transcription],
                "labels": str(row.label),
            }
        )
    data = {
        "title": title,
        "classification_task": classification_task,
        "task_background": task_background,
        "examples": examples,
        "qas": qas,
    }
    json_data = {"version": version, "data": [data]}

    with open(output_file_path, "w") as f:
        json.dump(json_data, f)

    return json_data


def convert_df_to_squad_qa_json(
    df: pd.DataFrame,
    title: str,
    task: str,
    examples: str = None,
    version: str = "squad_v1",
    output_file_path: str = "data.json",
) -> dict:
    paragraphs = []
    for id, row in df.iterrows():
        paragraph = {
            "context": row.transcription,
            "qas": [
                {
                    "id": id,
                    "question": row.question,
                    "answers": [
                        {
                            "text": row.answer,
                        }
                    ],
                }
            ],
        }
        paragraphs.append(paragraph)

    data = {
        "title": title,
        "task": task,
        "examples": examples,
        "paragraphs": paragraphs,
    }
    json_data = {"version": version, "data": [data]}

    with open(output_file_path, "w") as f:
        json.dump(json_data, f)

    return json_data
