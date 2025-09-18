import argparse
import json

import openpyxl
import pandas as pd

ID_COLUMN = "index"
TITLE_COLUMN = "title"
CONTEXT_COLUMN = "context"
QUESTION_COLUMN = "question"
ANSWERS_COLUMN = "answer"


def find_headers(path_to_file: str) -> int:
    start_row = 0
    book = openpyxl.load_workbook(path_to_file)
    sheet = book.worksheets[0]
    for i, row in enumerate(sheet.iter_rows(), start=1):
        if row[0].value == ID_COLUMN:
            start_row = i
            break
    book.close()
    return start_row


def convert_to_squad(
    path_to_file: str, version: str = "squad_v1", output_file_path: str = "data.json"
) -> dict:
    data = []
    start_row = find_headers(path_to_file=path_to_file)
    df = pd.read_excel(path_to_file, header=start_row - 1)

    for title, df_title in df.groupby(TITLE_COLUMN):
        paragraphs = []

        for context, df_context in df_title.groupby(CONTEXT_COLUMN):
            qas = []

            for _, row in df_context.iterrows():
                answers = [{"text": str(row[ANSWERS_COLUMN])}]
                qas.append(
                    {
                        "id": row[ID_COLUMN],
                        "question": row[QUESTION_COLUMN],
                        "answers": answers,
                    }
                )

            paragraphs.append({"context": context, "qas": qas})

        data.append({"title": title, "paragraphs": paragraphs})

    json_data = {"version": version, "data": data}

    with open(output_file_path, "w") as f:
        json.dump(json_data, f)

    return json_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert to squad like json format.")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="provide the path for the input data file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="provide the path for storing squad like json file",
        default="data.json",
    )

    args = parser.parse_args()

    path_to_file = args.input
    output_file = args.output
    json_data = convert_to_squad(
        path_to_file=path_to_file, output_file_path=output_file
    )
