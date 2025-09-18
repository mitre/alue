import argparse
import ast
import concurrent.futures
import json
import logging
import os
import re

import numpy as np
import output_normalizations
import pandas as pd
import squad_evaluation as squad_eval
from config import MODELS
from doc_retrieval_metrics import overall_recall_at_k, recall_at_k_per_query
from haystack.components.builders import PromptBuilder
from huggingface_hub import InferenceClient
from inference import load_prompt_from_file
from io_utilities import get_definitions_terms_dataset
from jinja2 import Template
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from schemas.extractive_qa.adjudicator_schema import Score

OPENAI_API_KEY = ""


class ExtractiveQAEval:
    """
    A class used to evaluate the performance of an extractive question answering model.
    ...

    Attributes
    ----------
    data_file : str
        the file containing the ground truth data
    pred_file : str
        the file containing the model's predictions
    out_dir : str
        the directory where the evaluation results will be saved
    na_prob_file : str, optional
        the file containing the probabilities of no answer being correct
    na_prob_thresh : int, optional
        the threshold for considering a no answer prediction correct
    out_image_dir : str, optional
        the directory where evaluation images will be saved
    verbose : bool, optional
        whether to print verbose output

    Methods
    -------
    perform_evaluation():
        Performs the evaluation and saves the results to the specified output directory.
    """

    def __init__(
        self,
        data_file: str,
        pred_file: str,
        out_dir: str,
        na_prob_file: str = None,
        na_prob_thresh: float = 1.0,
        out_image_dir: str = None,
        verbose: bool = False,
        llm_judge_model: str = "",
    ) -> None:
        """
        Parameters
        ----------
        data_file : str
            The file containing the ground truth data
        pred_file : str
            The file containing the model's predictions
        out_dir : str
            The directory where the evaluation results will be saved
        na_prob_file : str, optional
            The file containing the probabilities of no answer being correct
        na_prob_thresh : int, optional
            The threshold for considering a no answer prediction correct
        out_image_dir : str, optional
            The directory where evaluation images will be saved
        verbose : bool, optional
            Whether to print verbose output
        """
        self.data_file = data_file
        self.pred_file = pred_file
        self.out_dir = out_dir
        self.na_prob_file = na_prob_file
        self.na_prob_thresh = na_prob_thresh
        self.out_image_dir = out_image_dir
        self.verbose = verbose
        self.llm_judge_model = llm_judge_model
        squad_eval.OPTS = argparse.Namespace(
            data_file=self.data_file,
            pred_file=self.pred_file,
            out_file=self.out_dir,
            na_prob_file=self.na_prob_file,
            na_prob_thresh=self.na_prob_thresh,
            out_image_dir=self.out_image_dir,
            verbose=self.verbose,
        )

        if self.llm_judge_model:
            self.client = self.load_llm_judge()

    def perform_squad_evaluation(self) -> None:
        """
        Performs the evaluation and saves the results to the specified output directory.
        """
        squad_eval.main()

    def load_llm_judge(self):
        model_url = MODELS[self.llm_judge_model]["aip_endpoint"]

        client = OpenAI(base_url=model_url, api_key=OPENAI_API_KEY)
        return client

    def perform_llm_judge_evaluation(self, llm_judge_model: str, llm_judge_examples):
        """
        Performs the evaluation using LLM Judge and saves the results to the specified output directory.

        Parameters
        ----------
        llm_judge_schema : str
            The path to the LLM Judge schema file
        llm_judge_examples : str
            The path to the LLM Judge examples file
        """
        system_prompt = """
        You are an expert evaluation system designed to assess the accuracy of generated answer provided by a question answering chatbot. You are given the following components: a question, a reference answer, and a generated answer. Your task is to evaluate the generated answer's correctness based on the reference answer. Assign a score of 1 if the generated answer is correct, or 0 if it is incorrect. Provide a single score that reflects a comprehensive evaluation.
        A score of 1 indicates a correct answer. A score of 0 indicates a wrong answer.

        Here are a few examples:
        {% for example in examples %}
        Question: {{ example.question }}
        Transcript: {{ example.transcript }}
        Reference answer: {{ example.answer }}
        Generated answer: {{ example.predicted_answer }}
        Correctness score: {{ example.correctness_score }}
        {% endfor %}
        """

        user_prompt = """
        Question: {{ question }}
        Transcript: {{ transcript }}
        Reference answer: {{ answer }}
        Generated answer: {{ predicted_answer }}
        """

        # Load examples file
        with open(llm_judge_examples) as f:
            examples = json.load(f)

        # Render system prompt using examples
        system_template_prompt = Template(system_prompt)
        system_prompt = system_template_prompt.render(examples=examples)

        # Load dataset file
        with open(self.data_file) as f:
            dataset_json = json.load(f)

        dataset = dataset_json["data"]

        # Load predictions file
        with open(self.pred_file) as f:
            preds_json = json.load(f)

        # Loop through the dataset and predictions
        evaluation_data = []
        evaluation_results = {}
        for entry in dataset:
            paragraphs = entry.get("paragraphs", [])
            for paragraph in paragraphs:
                context = paragraph.get("context", "")
                qas = paragraph.get("qas", [])
                for qa in qas:
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    answer = answers[0].get("text", "") if answers else ""
                    qa_id = str(
                        qa.get("id", "")
                    )  # Convert ID to string to match predictions JSON

                    # Get the predicted answer from predictions file
                    predicted_answer = preds_json.get(qa_id, "")

                    # Append the evaluation example
                    evaluation_data.append(
                        {
                            "question": question,
                            "transcript": context,
                            "answer": answer,
                            "id": qa_id,
                            "predicted_answer": predicted_answer,
                        }
                    )

        user_template_prompt = Template(user_prompt)
        for data in evaluation_data:
            print(f"data: {data}")
            user_prompt = user_template_prompt.render(
                question=data["question"],
                transcript=data["transcript"],
                answer=data["answer"],
                predicted_answer=data["predicted_answer"],
            )

            # Send the prompt to the LLM Judge model
            response = self.client.beta.chat.completions.parse(
                model=llm_judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json",
                    "value": json.dumps(Score.model_json_schema()),
                },
                seed=42,
                timeout=120,
            )
            print(f"judge response: {response}")
            response = json.loads(response.choices[0].message.content)
            score = response["score"]

            evaluation_results[data["id"]] = score

        return evaluation_results

    def perform_evaluation(self, llm_judge_examples=None):
        """
        Performs the evaluation. If `self.llm_judge_model` is set, runs both
        `perform_squad_evaluation` and `perform_llm_judge_evaluation` in parallel.
        Otherwise, only runs `perform_squad_evaluation`.
        """
        if self.llm_judge_model:
            # Run both evaluations in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit tasks to the executor
                squad_future = executor.submit(self.perform_squad_evaluation)
                llm_judge_future = executor.submit(
                    self.perform_llm_judge_evaluation,
                    llm_judge_model=self.llm_judge_model,
                    llm_judge_examples=llm_judge_examples,
                )

                # Wait for both tasks to complete
                squad_result = (  # noqa: F841
                    squad_future.result()
                )  # Wait for SQuAD evaluation to finish
                llm_judge_result = (
                    llm_judge_future.result()
                )  # Wait for LLM Judge evaluation to finish

                # Optionally, process or save the results
                print("SQuAD evaluation completed.")
                print("LLM Judge evaluation completed.")
                print("LLM Judge results:", llm_judge_result)

                # Calculate the final correctness score (average of all scores)
                scores = list(llm_judge_result.values())
                final_correctness_score = sum(scores) / len(scores) if scores else 0

                # Add the final correctness score to the results
                llm_judge_result["final_correctness_score"] = final_correctness_score

                # Save LLM Judge results to a file
                llm_judge_output_file = os.path.join(
                    self.out_dir, "llm_judge_results.json"
                )
                with open(llm_judge_output_file, "w") as f:
                    json.dump(llm_judge_result, f, indent=4)

                llm_summary_file = os.path.join(
                    self.out_dir, "llm_judge_results_summary.json"
                )
                with open(llm_summary_file, "w") as f:
                    json.dump(
                        {"final_correctness_score": final_correctness_score},
                        f,
                        indent=4,
                    )
                    # Save the final correctness score to a file

                print(f"LLM Judge results saved to {llm_judge_output_file}")

        else:
            # Only run SQuAD evaluation
            self.perform_squad_evaluation()


class SequenceClassificationEval:
    """
    A class used to evaluate the performance on a sequence classification task.
    ...

    Attributes
    ----------
    data_file : str
        the file containing the ground truth data
    pred_file : str
        the file containing the model's predictions
    out_dir : str
        the directory where the evaluation results will be saved
    out_image_dir : str, optional
        the directory where evaluation images will be saved
    ...

    Methods
    -------
    perform_evaluation():
        Performs the evaluation and saves the results to the specified output directory.
    """

    def __init__(
        self,
        data_file: str,
        pred_file: str,
        out_dir: str,
        out_image_dir: str = None,
    ) -> None:
        """
        Parameters
        ----------
        data_file : str
            The file containing the ground truth data
        pred_file : str
            The file containing the model's predictions
        out_dir : str
            The directory where the evaluation results will be saved
        out_image_dir : str, optional
            The directory where evaluation images will be saved
        """
        self.data_file = data_file
        self.pred_file = pred_file
        self.out_dir = out_dir
        self.out_image_dir = out_image_dir

    def split_string_to_array(
        self, string_array: str, is_labels: bool = False
    ) -> list[str | int]:
        """
        Helper function that converts a string into an array of integers (for indices)
        or strings (for labels)
        """
        if is_labels:
            return list(
                set(
                    [
                        s.strip()
                        for s in string_array.strip("] [").replace("'", "").split(",")
                    ]
                )
            )
        else:
            return [
                int(s.strip())
                for s in string_array.strip("] [").replace("'", "").split(",")
            ]

    def parse_string_result(
        self,
        s: str,
        is_multiclass: bool,
        is_labels: bool,
    ) -> list[str] | str:
        if isinstance(s, list):
            return s
        else:
            if is_multiclass:
                return str(s)
            else:
                return self.split_string_to_array(s, is_labels)

    def perform_evaluation(
        self,
        is_labels: bool = False,
        is_multiclass: bool = False,
        normalize: bool = False,
        exact_match_only: bool = False,
        schema: bool = False,
        use_label_names: bool = False,
        task_normalization_name: str = "",
        task_specific_normalization_lookup: dict[str, str] = None,
        output_report_name: str = "classification_report.json",
    ) -> None:
        """
        Performs the evaluation and saves the results to the specified output directory.
        """
        # Get the classes from the task-specific normalization lookup
        label_names = task_specific_normalization_lookup["label_names"]
        data_dict = {}
        # ingest data indices and ground truth labels from input data file
        with open(self.data_file) as file:
            ground_truths = json.load(file)
            for dataset in ground_truths["data"]:
                for qas in dataset["qas"]:
                    if isinstance(qas["labels"], list):
                        qas["labels"] = ", ".join(qas["labels"])
                    data_dict[str(qas["id"])] = [qas["text_input"], qas["labels"]]

        if schema:
            with open(self.pred_file) as file:
                predictions = json.load(file)
            preds = output_normalizations.normalize_schema_output(
                task_normalization_name=task_normalization_name, data_dict=predictions
            )
            print(f"preds: {preds}")

            for sample in preds:
                data_dict[sample].append(preds[sample])

            df = pd.DataFrame.from_dict(
                data_dict,
                orient="index",
                columns=["text_input", "ground_truth_labels", "prediction"],
            )
        else:
            data_dict = {}

            # ingest data indices and ground truth labels from input data file
            with open(self.data_file) as file:
                ground_truths = json.load(file)
                for dataset in ground_truths["data"]:
                    for qas in dataset["qas"]:
                        data_dict[str(qas["id"])] = [qas["text_input"], qas["labels"]]

            # add predictions by index from predictions data file
            with open(self.pred_file) as file:
                predictions = json.load(file)
                for sample in predictions:
                    data_dict[sample].append(predictions[sample])

            # create pandas dataframe
            df = pd.DataFrame.from_dict(
                data_dict,
                orient="index",
                columns=["text_input", "ground_truth_labels", "prediction"],
            )

            # normalize prediction text if necessary
            if normalize:
                print(f"task specific table: {task_specific_normalization_lookup}")
                # overwrite prediction column with normalized prediction text
                # use exact_match_only flag to indicate whether normalization is strict or generous
                df["prediction"] = df["prediction"].apply(
                    lambda s: output_normalizations.normalize_sequence_classification_generation_output(
                        exact_output_pattern=re.compile(
                            task_specific_normalization_lookup["EXACT_OUTPUT_PATTERN"]
                        ),
                        search_pattern=re.compile(
                            task_specific_normalization_lookup["SEARCH_PATTERN"]
                        ),
                        text=s,
                        exact_match_only=exact_match_only,
                        no_match_return_value=task_specific_normalization_lookup[
                            "no_match_return_value"
                        ],
                    )
                )

        # parse ground truth string array into list
        df["ground_truth_labels"] = df["ground_truth_labels"].apply(
            lambda s: self.parse_string_result(s, is_multiclass, is_labels)
        )
        # parse predicted string from raw text
        df["predicted_labels"] = df["prediction"].apply(
            lambda s: self.parse_string_result(s, is_multiclass, is_labels)
        )

        # print out predictions and ground truths being analyzed
        tmp_filename = os.path.join(
            self.out_dir,
            f"{'raw' if not normalize else ('exact_match' if exact_match_only else 'generous_match')}_predictions.csv",
        )
        df[["ground_truth_labels", "predicted_labels"]].to_csv(tmp_filename)

        # return report as a dictionary
        if is_multiclass or not is_labels:
            classification_report_dict = classification_report(
                np.array(df["ground_truth_labels"].tolist()),
                np.array(df["predicted_labels"].tolist()),
                output_dict=True,
            )
            logging.info(f"Output Report: \n{classification_report_dict}")
        else:
            mlb = (
                MultiLabelBinarizer()
                if not label_names
                else MultiLabelBinarizer(classes=label_names)
            )
            classification_report_dict = classification_report(
                mlb.fit_transform(df["ground_truth_labels"].tolist()),
                mlb.fit_transform(df["predicted_labels"].tolist()),
                output_dict=True,
            )
            if label_names:
                labels = mlb.classes_

                # Replace classification report key with class name for non-excluded keys
                excluded_keys = {
                    "micro avg",
                    "macro avg",
                    "weighted avg",
                    "samples avg",
                }
                keys_to_update = [
                    k for k in classification_report_dict if k not in excluded_keys
                ]

                for k in keys_to_update:
                    # Update the key in the original dictionary
                    classification_report_dict[labels[int(k)]] = (
                        classification_report_dict.pop(k)
                    )

            logging.info(f"Output Report: \n{classification_report_dict}")

        # write out classification report
        report_filename = os.path.join(self.out_dir, output_report_name)
        with open(report_filename, "w") as file:
            json.dump(classification_report_dict, file)

        return classification_report_dict


class TokenClassificationEval:
    """
    A class used to evaluate the performance on a token classification task.
    ...

    Attributes
    ----------
    data_file : str
        the file containing the ground truth data
    pred_file : str
        the file containing the model's predictions
    out_dir : str
        the directory where the evaluation results will be saved
    out_image_dir : str, optional
        the directory where evaluation images will be saved
    ...

    Methods
    -------
    perform_evaluation():
        Performs the evaluation and saves the results to the specified output directory.
    """

    def __init__(
        self,
        data_file: str,
        pred_file: str,
        out_dir: str,
        out_image_dir: str = None,
    ) -> None:
        """
        Parameters
        ----------
        data_file : str
            The file containing the ground truth data
        pred_file : str
            The file containing the model's predictions
        out_dir : str
            The directory where the evaluation results will be saved
        out_image_dir : str, optional
            The directory where evaluation images will be saved
        """
        self.data_file = data_file
        self.pred_file = pred_file
        self.out_dir = out_dir
        self.out_image_dir = out_image_dir

    def parse_ner_labels(
        self, y_true_text: str, y_labels: list[str], y_true: list[str], pred_text: str
    ) -> list[str]:
        """
        Parses the predicted text to generate a list of predicted tags for each word/token in the true text.

        params:
        y_true_text: a string of text that corresponds to the ground truth transcript.
        y_labels: the list of all possible ground truth labels or ner tags
        y_true:  a list of the ground truth tags for each word/token in the true text
        pred_text: the predicted output text in a markup format

        Returns
        -------
        y_pred: a list of predicted tags for each word/token in the true text
        """
        # Extract markup tags with corresponding tag text
        tag_names, tag_texts = output_normalizations.extract_markup_tags_and_text(
            pred_text
        )

        # Generate a default y predicted label list
        y_pred = ["0" for _ in range(len(y_true))]

        # Find if tag belongs to the list and if tagged text belongs to the true text
        # Modify the default y_pred as applicable
        for i, name in enumerate(tag_names):
            if name in y_labels:
                indexes = output_normalizations.find_substring_indexes(
                    y_true_text, tag_texts[i]
                )
                for index in indexes:
                    if 0 <= index < len(y_pred):
                        y_pred[index] = name

        return y_pred

    def perform_evaluation(
        self,
        output_report_name: str = "classification_report.json",
        output_summary_name: str = "summary.json",
    ) -> None:
        """
        Performs the evaluation and saves the results to the specified output directory.
        """
        data_dict = {}

        # ingest data indices and ground truth labels from input data file
        with open(self.data_file) as file:
            ground_truths = json.load(file)
            for dataset in ground_truths["data"]:
                for qas in dataset["qas"]:
                    text_input = qas["text_input"][0]
                    text_ner_labels = ast.literal_eval(qas["labels"])
                    data_dict[str(qas["id"])] = [text_input, text_ner_labels]

        # add predictions by index from predictions data file
        with open(self.pred_file) as file:
            predictions = json.load(file)
            for sample in predictions:
                data_dict[sample].append(predictions[sample])

        # possible labels from ground truth input file. If empty, evaluator should fail.
        labels = ground_truths["data"][0]["gt_labels"]

        predictions_summary = {}

        ys_true = []
        ys_pred = []

        for key, value in data_dict.items():
            ground_truth_text = value[0]
            ground_truth_labels = value[1]
            prediction_text = value[2]
            y_pred = self.parse_ner_labels(
                y_true_text=ground_truth_text,
                y_labels=labels,
                y_true=ground_truth_labels,
                pred_text=prediction_text,
            )
            predictions_summary[key] = {
                "gt": ground_truth_text,
                "gt_labels": ground_truth_labels,
                "prediction": prediction_text,
                "prediction_labels": y_pred,
                "f1": f1_score(ground_truth_labels, y_pred, average="weighted"),
            }
            ys_true.append(ground_truth_labels)
            ys_pred.append(y_pred)

        ys_true_flat = [x for xs in ys_true for x in xs]
        ys_pred_flat = [x for xs in ys_pred for x in xs]

        metrics = classification_report(ys_true_flat, ys_pred_flat, output_dict=True)

        accuracy_batch = accuracy_score(ys_true_flat, ys_pred_flat)
        metrics["accuracy"] = {"accuracy": accuracy_batch}

        logging.info(f"Output Report: \n{metrics}")

        # Stringify results for json dump
        for key, value in metrics.items():
            for sub_key, sub_value in value.items():
                metrics[key][sub_key] = str(sub_value)

        # write out classification report
        report_filename = os.path.join(self.out_dir, output_report_name)
        with open(report_filename, "w") as file:
            json.dump(metrics, file)

        # write out summary
        summary_filename = os.path.join(self.out_dir, output_summary_name)
        with open(summary_filename, "w") as file:
            json.dump(predictions_summary, file)


class RAGEval(ExtractiveQAEval):
    def __init__(
        self,
        data_file: str,
        pred_file: str,
        out_dir: str,
        na_prob_file: str = None,
        na_prob_thresh: float = 1.0,
        out_image_dir: str = None,
        verbose: bool = False,
        k: int = 5,
    ) -> None:
        """
        A class used to evaluate the performance of the RAG pipeline.
        ...

        Attributes
        ----------
        data_file : str
            the file containing the ground truth data
        pred_file : str
            the file containing the model's predictions
        out_dir : str
            the directory where the evaluation results will be saved
        na_prob_file : str, optional
            the file containing the probabilities of no answer being correct
        na_prob_thresh : int, optional
            the threshold for considering a no answer prediction correct
        out_image_dir : str, optional
            the directory where evaluation images will be saved
        verbose : bool, optional
            whether to print verbose output
        k: int, optional
            top_k value for doc retrieval, default is 5


        Methods
        -------
        perform_evaluation():
            Performs the evaluation of the qa and saves the results to the specified output directory.

        evaluate_recall_at_k():
            Performs evaluation for document retrieval aspect of rag pipeline
        """
        super().__init__(
            data_file,
            pred_file,
            out_dir,
            na_prob_file,
            na_prob_thresh,
            out_image_dir,
            verbose,
        )
        self.k = k
        self.out_dir = out_dir

    def perform_evaluation(self) -> None:
        """
        Performs the evaluation of qa and saves the results to the specified output directory.
        """
        super().perform_evaluation()
        self.evaluate_recall_at_k()

    def evaluate_recall_at_k(self):
        """
        Performs the evaluation for retrieval and saves the results to the specified output directory.
        """
        with open(self.data_file) as f:
            ground_truth_data = json.load(f)
        with open(self.pred_file) as f:
            predicted_data = json.load(f)

        ground_truth_ids = []
        predicted_ids = []
        ground_truth_data = ground_truth_data["data"]

        for data in ground_truth_data:
            paragraphs = data["paragraphs"]
            for paragraph in paragraphs:
                qas = paragraph["qas"]
                for qa in qas:
                    ground_truth_ids.extend(
                        [answer["document_id"] for answer in qa["answers"]]
                    )
                    qid = qa["id"]
                    if (
                        isinstance(predicted_data[qid], dict)
                        and "predicted_doc_ids" in predicted_data[qid]
                    ):
                        predicted_ids.append(
                            list(set(predicted_data[qid]["predicted_doc_ids"]))
                        )
                    else:
                        predicted_ids.append([])

        recall_values = recall_at_k_per_query(ground_truth_ids, predicted_ids)
        overall_recall = overall_recall_at_k(recall_values)

        doc_retrieval_metrics_file = "doc_retrieval.json"

        with open(os.path.join(self.out_dir, doc_retrieval_metrics_file), "w") as f:
            recall_k = {"k": self.k, "recall@k": overall_recall}
            json.dump(recall_k, f)

        print(f"Overall recall@{self.k}: {overall_recall}")


class GlossaryTermsEval:
    def __init__(
        self,
        data_file: str,
        pred_file: str,
        out_dir: str,
        correctness_template_path: str,
        schema_path: str,
        llm_adjudicator_name: str,
    ):
        self.out_dir = out_dir
        self.dataset = get_definitions_terms_dataset(input_excel_sheet=data_file)
        self.correctness_tempalate_path = load_prompt_from_file(
            correctness_template_path
        )
        self.schema_path = schema_path
        with open(pred_file) as f:
            self.predictions = json.load(f)
        model_url = MODELS[llm_adjudicator_name]["aip_endpoint"]

        self.llm_adjudicator = InferenceClient(model=model_url, token=False)

    def perform_evaluation(self):
        prompt_builder = PromptBuilder(self.correctness_tempalate_path)

        with open(self.schema_path) as f:
            schema = json.load(f)

        schema_str = json.dumps(schema)

        eval_results = []

        overall_correctness_score = 0
        for row in self.dataset:
            term_id = row["id"]
            term = row["term"]
            ground_truth_definition = row["ground_truth_definition"]
            predicted_definition = self.predictions[str(term_id)]["answer"][
                "definition"
            ]

            prompt = prompt_builder.run(
                judge_schema_json=schema_str,
                user_query=term,
                reference_answer=ground_truth_definition,
                generated_answer=predicted_definition,
            )["prompt"]

            score_with_schema = self.llm_adjudicator.text_generation(
                prompt,
                max_new_tokens=225,  # 25
                seed=42,
                grammar={"type": "json", "value": schema},
            )
            score_with_schema = json.loads(score_with_schema)
            eval_results.append(
                {
                    "term_id": term_id,
                    "term": term,
                    "ground_truth_definition": ground_truth_definition,
                    "predicted_definition": predicted_definition,
                    "score_with_schema": score_with_schema,
                }
            )
            overall_correctness_score += score_with_schema["score"]

        overall_correctness_score = overall_correctness_score / len(self.dataset)
        metrics_dict = {"overall_correctness_score": overall_correctness_score}

        eval_results_path = os.path.join(self.out_dir, "eval_results.json")
        metrics_path = os.path.join(self.out_dir, "metrics.json")

        with open(eval_results_path, "w") as f:
            json.dump(eval_results, f)

        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f)

        return eval_results


class MCQAEval:
    def __init__(self, data_file, pred_file, out_dir):
        self.data_file = data_file
        self.pred_file = pred_file
        self.out_dir = out_dir

    def perform_evaluation(self):
        with open(self.data_file) as f:
            data = json.load(f)["data"]

        with open(self.pred_file) as f:
            pred = json.load(f)

        correct = 0

        for item in data:
            question_id = item["id"]
            gt_answer = item["answer"]
            pred_answer = pred[str(question_id)]

            if gt_answer == pred_answer:
                correct += 1

        accuracy = correct / len(data)

        metrics_path = os.path.join(self.out_dir, "metrics.json")
        eval_metrics = {"accuracy": accuracy}

        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f)

        return eval_metrics


class BinaryClassificationEval:
    """
    A class used to evaluate the performance of a binary classification model.
    ...

    Attributes
    ----------
    pred_file : str
        the file containing the model's predictions
    out_dir : str
        the directory where the evaluation results will be saved

    Methods
    -------
    perform_evaluation():
        Performs the evaluation and saves the results to the specified output directory.
    """

    def __init__(self, pred_file: str, out_dir: str) -> None:
        """
        Parameters
        ----------
        pred_file : str
            The file containing the model's predictions
        out_dir : str
            The directory where the evaluation results will be saved
        """
        self.pred_file = pred_file
        self.out_dir = out_dir

    def perform_evaluation(self) -> dict[str, float]:
        """
        Performs the evaluation and saves the results to the specified output directory.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the accuracy and other metrics.
        """
        # Load predictions file
        with open(self.pred_file) as f:
            predictions = json.load(f)

        # Initialize counters
        correct = 0
        total = 0

        # Iterate through predictions and compare ground truth with predicted labels
        for _k, v in predictions.items():
            gt_label = v["gt_label"]
            pred_label = v["pred_label"]

            if gt_label == pred_label:
                correct += 1
            total += 1

        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0.0

        # Save metrics to a file
        metrics = {"accuracy": accuracy}
        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Accuracy: {accuracy}")
        print(f"Metrics saved to {metrics_path}")

        return metrics
