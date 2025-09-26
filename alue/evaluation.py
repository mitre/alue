import argparse
import ast
import concurrent.futures
import json
import logging
import os
import re
import tempfile

import numpy as np
# import output_normalizations
import pandas as pd
from alue import squad_evaluation as squad_eval
from alue.output_normalizations import normalize_tail_extraction_predictions
from alue.llm_judge_metrics import ContextRelevancyJudge, CompositeCorrectnessJudge, ClaimDecompositionJudge
from alue.doc_retrieval_metrics import overall_recall_at_k, recall_at_k_per_query

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from typing import Optional, Callable, Any, Dict


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
        normalizer_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
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
        self.normalizer_func = normalizer_func


    def _normalize_predictions(self, pred_file: str) -> str:
        """
        Normalize predictions using the provided normalizer function.
        If no normalizer is provided, predictions are used as-is.
        
        Returns path to normalized predictions file.
        """
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
        
        # Apply normalization if function is provided
        if self.normalizer_func:
            normalized_preds = self.normalizer_func(predictions)
        else:
            normalized_preds = predictions
        
        # Create temporary file with normalized predictions
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.json', 
            delete=False, 
            dir=os.path.dirname(pred_file)
        )
        
        with temp_file as f:
            json.dump(normalized_preds, f, indent=2)
        
        return temp_file.name

    def perform_squad_evaluation(self) -> None:
        """
        Performs the evaluation and saves the results to the specified output directory.
        """
        # Normalize predictions first
        normalized_pred_file = self._normalize_predictions(self.pred_file)
        
        try:
            # Update OPTS with normalized predictions file
            squad_eval.OPTS = argparse.Namespace(
                data_file=self.data_file,
                pred_file=normalized_pred_file,
                out_file=self.out_dir,
                na_prob_file=self.na_prob_file,
                na_prob_thresh=self.na_prob_thresh,
                out_image_dir=self.out_image_dir,
                verbose=self.verbose,
            )
            
            squad_eval.main()
            
        finally:
            # Clean up temporary file
            if os.path.exists(normalized_pred_file):
                os.unlink(normalized_pred_file)

    def perform_evaluation(self):
        self.perform_squad_evaluation()


# class SequenceClassificationEval:
#     """
#     A class used to evaluate the performance on a sequence classification task.
#     ...

#     Attributes
#     ----------
#     data_file : str
#         the file containing the ground truth data
#     pred_file : str
#         the file containing the model's predictions
#     out_dir : str
#         the directory where the evaluation results will be saved
#     out_image_dir : str, optional
#         the directory where evaluation images will be saved
#     ...

#     Methods
#     -------
#     perform_evaluation():
#         Performs the evaluation and saves the results to the specified output directory.
#     """

#     def __init__(
#         self,
#         data_file: str,
#         pred_file: str,
#         out_dir: str,
#         out_image_dir: str = None,
#     ) -> None:
#         """
#         Parameters
#         ----------
#         data_file : str
#             The file containing the ground truth data
#         pred_file : str
#             The file containing the model's predictions
#         out_dir : str
#             The directory where the evaluation results will be saved
#         out_image_dir : str, optional
#             The directory where evaluation images will be saved
#         """
#         self.data_file = data_file
#         self.pred_file = pred_file
#         self.out_dir = out_dir
#         self.out_image_dir = out_image_dir

#     def split_string_to_array(
#         self, string_array: str, is_labels: bool = False
#     ) -> list[str | int]:
#         """
#         Helper function that converts a string into an array of integers (for indices)
#         or strings (for labels)
#         """
#         if is_labels:
#             return list(
#                 set(
#                     [
#                         s.strip()
#                         for s in string_array.strip("] [").replace("'", "").split(",")
#                     ]
#                 )
#             )
#         else:
#             return [
#                 int(s.strip())
#                 for s in string_array.strip("] [").replace("'", "").split(",")
#             ]

#     def parse_string_result(
#         self,
#         s: str,
#         is_multiclass: bool,
#         is_labels: bool,
#     ) -> list[str] | str:
#         if isinstance(s, list):
#             return s
#         else:
#             if is_multiclass:
#                 return str(s)
#             else:
#                 return self.split_string_to_array(s, is_labels)

#     def perform_evaluation(
#         self,
#         is_labels: bool = False,
#         is_multiclass: bool = False,
#         normalize: bool = False,
#         exact_match_only: bool = False,
#         schema: bool = False,
#         use_label_names: bool = False,
#         task_normalization_name: str = "",
#         task_specific_normalization_lookup: dict[str, str] = None,
#         output_report_name: str = "classification_report.json",
#     ) -> None:
#         """
#         Performs the evaluation and saves the results to the specified output directory.
#         """
#         # Get the classes from the task-specific normalization lookup
#         label_names = task_specific_normalization_lookup["label_names"]
#         data_dict = {}
#         # ingest data indices and ground truth labels from input data file
#         with open(self.data_file) as file:
#             ground_truths = json.load(file)
#             for dataset in ground_truths["data"]:
#                 for qas in dataset["qas"]:
#                     if isinstance(qas["labels"], list):
#                         qas["labels"] = ", ".join(qas["labels"])
#                     data_dict[str(qas["id"])] = [qas["text_input"], qas["labels"]]

#         if schema:
#             with open(self.pred_file) as file:
#                 predictions = json.load(file)
#             preds = output_normalizations.normalize_schema_output(
#                 task_normalization_name=task_normalization_name, data_dict=predictions
#             )
#             print(f"preds: {preds}")

#             for sample in preds:
#                 data_dict[sample].append(preds[sample])

#             df = pd.DataFrame.from_dict(
#                 data_dict,
#                 orient="index",
#                 columns=["text_input", "ground_truth_labels", "prediction"],
#             )
#         else:
#             data_dict = {}

#             # ingest data indices and ground truth labels from input data file
#             with open(self.data_file) as file:
#                 ground_truths = json.load(file)
#                 for dataset in ground_truths["data"]:
#                     for qas in dataset["qas"]:
#                         data_dict[str(qas["id"])] = [qas["text_input"], qas["labels"]]

#             # add predictions by index from predictions data file
#             with open(self.pred_file) as file:
#                 predictions = json.load(file)
#                 for sample in predictions:
#                     data_dict[sample].append(predictions[sample])

#             # create pandas dataframe
#             df = pd.DataFrame.from_dict(
#                 data_dict,
#                 orient="index",
#                 columns=["text_input", "ground_truth_labels", "prediction"],
#             )

#             # normalize prediction text if necessary
#             if normalize:
#                 print(f"task specific table: {task_specific_normalization_lookup}")
#                 # overwrite prediction column with normalized prediction text
#                 # use exact_match_only flag to indicate whether normalization is strict or generous
#                 df["prediction"] = df["prediction"].apply(
#                     lambda s: output_normalizations.normalize_sequence_classification_generation_output(
#                         exact_output_pattern=re.compile(
#                             task_specific_normalization_lookup["EXACT_OUTPUT_PATTERN"]
#                         ),
#                         search_pattern=re.compile(
#                             task_specific_normalization_lookup["SEARCH_PATTERN"]
#                         ),
#                         text=s,
#                         exact_match_only=exact_match_only,
#                         no_match_return_value=task_specific_normalization_lookup[
#                             "no_match_return_value"
#                         ],
#                     )
#                 )

#         # parse ground truth string array into list
#         df["ground_truth_labels"] = df["ground_truth_labels"].apply(
#             lambda s: self.parse_string_result(s, is_multiclass, is_labels)
#         )
#         # parse predicted string from raw text
#         df["predicted_labels"] = df["prediction"].apply(
#             lambda s: self.parse_string_result(s, is_multiclass, is_labels)
#         )

#         # print out predictions and ground truths being analyzed
#         tmp_filename = os.path.join(
#             self.out_dir,
#             f"{'raw' if not normalize else ('exact_match' if exact_match_only else 'generous_match')}_predictions.csv",
#         )
#         df[["ground_truth_labels", "predicted_labels"]].to_csv(tmp_filename)

#         # return report as a dictionary
#         if is_multiclass or not is_labels:
#             classification_report_dict = classification_report(
#                 np.array(df["ground_truth_labels"].tolist()),
#                 np.array(df["predicted_labels"].tolist()),
#                 output_dict=True,
#             )
#             logging.info(f"Output Report: \n{classification_report_dict}")
#         else:
#             mlb = (
#                 MultiLabelBinarizer()
#                 if not label_names
#                 else MultiLabelBinarizer(classes=label_names)
#             )
#             classification_report_dict = classification_report(
#                 mlb.fit_transform(df["ground_truth_labels"].tolist()),
#                 mlb.fit_transform(df["predicted_labels"].tolist()),
#                 output_dict=True,
#             )
#             if label_names:
#                 labels = mlb.classes_

#                 # Replace classification report key with class name for non-excluded keys
#                 excluded_keys = {
#                     "micro avg",
#                     "macro avg",
#                     "weighted avg",
#                     "samples avg",
#                 }
#                 keys_to_update = [
#                     k for k in classification_report_dict if k not in excluded_keys
#                 ]

#                 for k in keys_to_update:
#                     # Update the key in the original dictionary
#                     classification_report_dict[labels[int(k)]] = (
#                         classification_report_dict.pop(k)
#                     )

#             logging.info(f"Output Report: \n{classification_report_dict}")

#         # write out classification report
#         report_filename = os.path.join(self.out_dir, output_report_name)
#         with open(report_filename, "w") as file:
#             json.dump(classification_report_dict, file)

#         return classification_report_dict


# class TokenClassificationEval:
#     """
#     A class used to evaluate the performance on a token classification task.
#     ...

#     Attributes
#     ----------
#     data_file : str
#         the file containing the ground truth data
#     pred_file : str
#         the file containing the model's predictions
#     out_dir : str
#         the directory where the evaluation results will be saved
#     out_image_dir : str, optional
#         the directory where evaluation images will be saved
#     ...

#     Methods
#     -------
#     perform_evaluation():
#         Performs the evaluation and saves the results to the specified output directory.
#     """

#     def __init__(
#         self,
#         data_file: str,
#         pred_file: str,
#         out_dir: str,
#         out_image_dir: str = None,
#     ) -> None:
#         """
#         Parameters
#         ----------
#         data_file : str
#             The file containing the ground truth data
#         pred_file : str
#             The file containing the model's predictions
#         out_dir : str
#             The directory where the evaluation results will be saved
#         out_image_dir : str, optional
#             The directory where evaluation images will be saved
#         """
#         self.data_file = data_file
#         self.pred_file = pred_file
#         self.out_dir = out_dir
#         self.out_image_dir = out_image_dir

#     def parse_ner_labels(
#         self, y_true_text: str, y_labels: list[str], y_true: list[str], pred_text: str
#     ) -> list[str]:
#         """
#         Parses the predicted text to generate a list of predicted tags for each word/token in the true text.

#         params:
#         y_true_text: a string of text that corresponds to the ground truth transcript.
#         y_labels: the list of all possible ground truth labels or ner tags
#         y_true:  a list of the ground truth tags for each word/token in the true text
#         pred_text: the predicted output text in a markup format

#         Returns
#         -------
#         y_pred: a list of predicted tags for each word/token in the true text
#         """
#         # Extract markup tags with corresponding tag text
#         tag_names, tag_texts = output_normalizations.extract_markup_tags_and_text(
#             pred_text
#         )

#         # Generate a default y predicted label list
#         y_pred = ["0" for _ in range(len(y_true))]

#         # Find if tag belongs to the list and if tagged text belongs to the true text
#         # Modify the default y_pred as applicable
#         for i, name in enumerate(tag_names):
#             if name in y_labels:
#                 indexes = output_normalizations.find_substring_indexes(
#                     y_true_text, tag_texts[i]
#                 )
#                 for index in indexes:
#                     if 0 <= index < len(y_pred):
#                         y_pred[index] = name

#         return y_pred

#     def perform_evaluation(
#         self,
#         output_report_name: str = "classification_report.json",
#         output_summary_name: str = "summary.json",
#     ) -> None:
#         """
#         Performs the evaluation and saves the results to the specified output directory.
#         """
#         data_dict = {}

#         # ingest data indices and ground truth labels from input data file
#         with open(self.data_file) as file:
#             ground_truths = json.load(file)
#             for dataset in ground_truths["data"]:
#                 for qas in dataset["qas"]:
#                     text_input = qas["text_input"][0]
#                     text_ner_labels = ast.literal_eval(qas["labels"])
#                     data_dict[str(qas["id"])] = [text_input, text_ner_labels]

#         # add predictions by index from predictions data file
#         with open(self.pred_file) as file:
#             predictions = json.load(file)
#             for sample in predictions:
#                 data_dict[sample].append(predictions[sample])

#         # possible labels from ground truth input file. If empty, evaluator should fail.
#         labels = ground_truths["data"][0]["gt_labels"]

#         predictions_summary = {}

#         ys_true = []
#         ys_pred = []

#         for key, value in data_dict.items():
#             ground_truth_text = value[0]
#             ground_truth_labels = value[1]
#             prediction_text = value[2]
#             y_pred = self.parse_ner_labels(
#                 y_true_text=ground_truth_text,
#                 y_labels=labels,
#                 y_true=ground_truth_labels,
#                 pred_text=prediction_text,
#             )
#             predictions_summary[key] = {
#                 "gt": ground_truth_text,
#                 "gt_labels": ground_truth_labels,
#                 "prediction": prediction_text,
#                 "prediction_labels": y_pred,
#                 "f1": f1_score(ground_truth_labels, y_pred, average="weighted"),
#             }
#             ys_true.append(ground_truth_labels)
#             ys_pred.append(y_pred)

#         ys_true_flat = [x for xs in ys_true for x in xs]
#         ys_pred_flat = [x for xs in ys_pred for x in xs]

#         metrics = classification_report(ys_true_flat, ys_pred_flat, output_dict=True)

#         accuracy_batch = accuracy_score(ys_true_flat, ys_pred_flat)
#         metrics["accuracy"] = {"accuracy": accuracy_batch}

#         logging.info(f"Output Report: \n{metrics}")

#         # Stringify results for json dump
#         for key, value in metrics.items():
#             for sub_key, sub_value in value.items():
#                 metrics[key][sub_key] = str(sub_value)

#         # write out classification report
#         report_filename = os.path.join(self.out_dir, output_report_name)
#         with open(report_filename, "w") as file:
#             json.dump(metrics, file)

#         # write out summary
#         summary_filename = os.path.join(self.out_dir, output_summary_name)
#         with open(summary_filename, "w") as file:
#             json.dump(predictions_summary, file)


class RAGEval:
    def __init__(
        self,
        pred_file: str, 
        out_dir: str,
        model_name: str,
        data_file: str = None,  
        database_path: str = None,  
        collection_name: str = None,  
        evaluate_retrieval: bool = True,
        evaluate_generation: bool = True,
        use_recall_k: bool = False,  # Only if document_ids available
        k: int = 5,
        verbose: bool = False
    ):
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
        
        self.k = k
        self.out_dir = out_dir
        self.use_recall_k = use_recall_k
        self.verbose = verbose
        self.pred_file = pred_file
        self.data_file = data_file
        self.model_name = model_name
        self.database_path = database_path
        self.collection_name = collection_name
        self.evaluate_retrieval = evaluate_retrieval
        self.evaluate_generation = evaluate_generation

    def perform_evaluation(self) -> None:
        """
        Performs the evaluation of qa and saves the results to the specified output directory.
        """
        results = {}

        if self.evaluate_retrieval:
            results.update(self.evaluate_retrieval_metrics())
        
        if self.evaluate_generation:
            results.update(self.evaluate_generation_metrics())

        with open(os.path.join(self.out_dir, "rag_evaluation_summary.json"), "w") as f:
            json.dump(results, f, indent=2)


    def evaluate_retrieval_metrics(self):
        """
        Evaluate retrieval performance using Context Relevancy and optionally Recall@K.
        
        Returns:
            dict: Dictionary containing retrieval evaluation results
        """
        retrieval_results = {}
        
        # Always evaluate context relevancy using LLM judge
        if self.database_path and self.collection_name:
            print("[RAGEval] Running Context Relevancy evaluation...")
            
            context_judge = ContextRelevancyJudge(
                model_name=self.model_name,
                collection_name=self.collection_name,
                database_path=self.database_path,
                explanations=False
            )

            self.cr_output_path = os.path.join(self.out_dir, "context_relevancy.json")
            
            context_results = context_judge.evaluate(
                filename=self.pred_file,
                store_output=True,
                output_path=self.cr_output_path
            )
            
            # Calculate average context relevancy per question, then average those
            question_scores = []
            for item in context_results:
                question_relevant = sum(context["context_relevancy"] for context in item["context"])
                question_avg = question_relevant / len(item["context"])
                question_scores.append(question_avg)
            
            avg_context_relevancy = sum(question_scores) / len(question_scores)
            
            retrieval_results["context_relevancy"] = {
                "average_score": avg_context_relevancy,
                "total_questions": len(question_scores)
            }
            
            print(f"[RAGEval] Average Context Relevancy: {avg_context_relevancy:.3f}")
            
            # Store context results for generation evaluation
            self.context_evaluation_results = context_results
            
        else:
            print("[RAGEval] Skipping Context Relevancy - database_path or collection_name not provided")
            self.context_evaluation_results = None
        
        # Optionally evaluate recall@k if document IDs are available
        if self.use_recall_k:
            print(f"[RAGEval] Running Recall@{self.k} evaluation...")
            recall_results = self._evaluate_recall_at_k()
            retrieval_results["recall_at_k"] = recall_results
    
        return retrieval_results

    def _evaluate_recall_at_k(self):
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


    def evaluate_generation_metrics(self):
        """
        Evaluate generation performance using Composite Correctness.
        
        Returns:
            dict: Dictionary containing generation evaluation results
        """
        generation_results = {}
        
        # Use context evaluation results if available, otherwise load from predictions file
        if hasattr(self, 'context_evaluation_results') and self.context_evaluation_results:
            print("[RAGEval] Using context evaluation results for generation metrics...")
            dataset = self.context_evaluation_results
        else:
            print("[RAGEval] Loading predictions file for generation metrics...")
            # If no context evaluation was run, we need to load and format the data
            dataset = self._load_cr_predictions_for_generation()
        
        # Run Composite Correctness evaluation
        print("[RAGEval] Running Composite Correctness evaluation...")
        
        correctness_judge = CompositeCorrectnessJudge(
            model_name=self.model_name,
            explanations=False if not self.verbose else True  # Set to True if you want detailed explanations
        )
        
        correctness_results = correctness_judge.evaluate(
            dataset=dataset,
            store_output=True,
            output_path=os.path.join(self.out_dir, "composite_correctness.json")
        )
        
        # Extract the average composite correctness score
        avg_composite_correctness = correctness_results.get("composite_correctness_average", 0)
        
        generation_results["composite_correctness"] = {
            "average_score": avg_composite_correctness,
            "total_questions": len(dataset)
        }
        
        print(f"[RAGEval] Average Composite Correctness: {avg_composite_correctness:.3f}")
        
        return generation_results

    def _load_cr_predictions_for_generation(self, cr_output_path: str = None):
        """
        Load predictions file and format for generation evaluation when context evaluation wasn't run.
        This would need to be implemented based on your predictions file format.
        """
        
        if cr_output_path:
            self.cr_output_path = cr_output_path
        with open(self.cr_output_path, 'r') as f:
            dataset = json.load(f)
     
        return dataset


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

        # Extract ground truth and predictions
        ground_truth = []
        predictions = []

        for item in data:
            question_id = str(item["id"])

            if question_id in pred:
                gt_answer = item["output"]
                pred_answer = pred[question_id]

                ground_truth.append(gt_answer)
                predictions.append(pred_answer)

        # Calculate metrics using the same logic as mcqa_simplified.py
        correct = sum(1 for pred_ans, true_ans in zip(predictions, ground_truth) if pred_ans == true_ans)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0

        metrics_path = os.path.join(self.out_dir, "metrics.json")
        eval_metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }

        with open(metrics_path, "w") as f:
            json.dump(eval_metrics, f)

        return eval_metrics
    

class SummarizationEval:
    """
    A class used to evaluate the performance of a summarization model using claim decomposition metrics.
    """

    def __init__(
        self,
        pred_file: str,
        out_dir: str,
        model_name: str,
        verbose: bool = False
    ) -> None:
        """
        Parameters
        ----------
        pred_file : str
            The file containing the model's predictions in the expected format
        out_dir : str
            The directory where the evaluation results will be saved
        model_name : str
            The name of the model to use for LLM judge evaluation
        verbose : bool, optional
            Whether to print verbose output
        """
        self.pred_file = pred_file
        self.out_dir = out_dir
        self.model_name = model_name
        self.verbose = verbose

    def perform_evaluation(self) -> dict:
        """
        Performs the evaluation and saves the results to the specified output directory.
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.out_dir, exist_ok=True)
        
        claim_judge = ClaimDecompositionJudge(
            model_name=self.model_name,
            explanations=self.verbose
        )
        
        # Run claim decomposition evaluation
        print("[SummarizationEval] Running claim decomposition evaluation...")
        claim_results_path = os.path.join(self.out_dir, "claim_decomposition_detailed.json")
        
        claim_results = claim_judge.evaluate(
            filename=self.pred_file,
            store_output=True,
            output_path=claim_results_path
        )
        
        # Calculate summary metrics
        print("[SummarizationEval] Calculating summary metrics...")
        precisions = []
        recalls = []
        
        for item_data in claim_results.values():
            if "precision" in item_data and "recall" in item_data:
                precisions.append(item_data["precision"])
                recalls.append(item_data["recall"])
        
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        summary_metrics = {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
            "total_samples": len(precisions)
        }

        # Save summary metrics to file
        summary_path = os.path.join(self.out_dir, "summarization_metrics.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        
        print(f"[SummarizationEval] Summary metrics saved to: {summary_path}")
        
        return summary_metrics
# class BinaryClassificationEval:
#     """
#     A class used to evaluate the performance of a binary classification model.
#     ...

#     Attributes
#     ----------
#     pred_file : str
#         the file containing the model's predictions
#     out_dir : str
#         the directory where the evaluation results will be saved

#     Methods
#     -------
#     perform_evaluation():
#         Performs the evaluation and saves the results to the specified output directory.
#     """

#     def __init__(self, pred_file: str, out_dir: str) -> None:
#         """
#         Parameters
#         ----------
#         pred_file : str
#             The file containing the model's predictions
#         out_dir : str
#             The directory where the evaluation results will be saved
#         """
#         self.pred_file = pred_file
#         self.out_dir = out_dir

#     def perform_evaluation(self) -> dict[str, float]:
#         """
#         Performs the evaluation and saves the results to the specified output directory.

#         Returns
#         -------
#         Dict[str, float]
#             A dictionary containing the accuracy and other metrics.
#         """
#         # Load predictions file
#         with open(self.pred_file) as f:
#             predictions = json.load(f)

#         # Initialize counters
#         correct = 0
#         total = 0

#         # Iterate through predictions and compare ground truth with predicted labels
#         for _k, v in predictions.items():
#             gt_label = v["gt_label"]
#             pred_label = v["pred_label"]

#             if gt_label == pred_label:
#                 correct += 1
#             total += 1

#         # Calculate accuracy
#         accuracy = correct / total if total > 0 else 0.0

#         # Save metrics to a file
#         metrics = {"accuracy": accuracy}
#         metrics_path = os.path.join(self.out_dir, "metrics.json")
#         with open(metrics_path, "w") as f:
#             json.dump(metrics, f, indent=4)

#         print(f"Accuracy: {accuracy}")
#         print(f"Metrics saved to {metrics_path}")

#         return metrics
