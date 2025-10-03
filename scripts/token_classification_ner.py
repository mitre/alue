import argparse
import json
import os
from datetime import datetime
from typing import Any

from evaluation import TokenClassificationEval
from haystack.components.builders import PromptBuilder
from inference import ClassificationInference
from io_utilities import get_classification_examples, get_classification_inputs_list
from tqdm import tqdm

from alue import config


def get_inference(
    user_input: dict[str, str], inference_pipeline: ClassificationInference
) -> dict[str, str]:
    """
    Generates predictions from the selected model.

    Parameters
    ----------
    user_input : Dict[str, str]
        A dictionary containing task id and task inputs.
    inference_pipeline : ClassificationPipeline
        The classification pipeline for inference.

    Returns
    -------
    Dict[str, str]
        A dictionary containing task id, task inputs, and inference output.
    """
    output = inference_pipeline._get_inference_per_batch([user_input])
    return output


def perform_inference(
    input_data_json_path: str,
    output_eval_res_dir: str,
    inference_pipeline: ClassificationInference,
) -> str:
    """
    Collects predictions and performs evaluation.

    Parameters
    ----------
    input_data_json_path : str
        The path to the squad-like JSON dataset.
    output_eval_res_dir : str
        The path to store evaluation results.
    inference_pipeline : ClassificationPipeline
        The classification pipeline for inference.

    Returns
    -------
    Path to the JSON file with predictions.
    """
    # load input samples for inference
    inputs_list = get_classification_inputs_list(input_data_json_path)

    # loop over input samples and execute inference,
    # accumulate predictions to final_predictions dictionary
    final_predictions = {}
    for i in tqdm(range(len(inputs_list))):
        user_input = inputs_list[i]
        pred = get_inference(
            user_input=user_input, inference_pipeline=inference_pipeline
        )
        final_predictions[pred["answers"][0]["id"]] = pred["answers"][0][
            "classification_output"
        ]

    # write predictions to disk
    if not os.path.exists(output_eval_res_dir):
        os.makedirs(output_eval_res_dir)

    predictions_filename = os.path.join(output_eval_res_dir, "predictions.json")
    with open(predictions_filename, "w") as file:
        json.dump(final_predictions, file)

    return predictions_filename


def perform_evaluation(
    input_data_json_path: str,
    output_eval_res_dir: str,
    predictions_filename: str,
    output_report_name: str = "classification_report.json",
):
    """
    Collects predictions and performs evaluation.

    Parameters
    ----------
    input_data_json_path : str
        The path to the squad-like JSON dataset.
    output_eval_res_dir : str
        The path to store evaluation results.
    predictions_filename : str
        Path to the JSON file with predictions.
    """
    eval = TokenClassificationEval(
        data_file=input_data_json_path,
        pred_file=predictions_filename,
        out_dir=output_eval_res_dir,
    )
    eval.perform_evaluation(
        output_report_name=output_report_name,
    )


def run_token_classification_evaluation(args):
    perform_evaluation(
        input_data_json_path=args.input_data_json_path,
        output_eval_res_dir=args.output_eval_res_dir,
        predictions_filename=args.predictions_filename,
    )


def run_token_classification_inference(args):
    # specify generation arguments, specific to classification with a generative model
    generation_kwargs = {
        "return_full_text": False,
        "max_new_tokens": 50,
        "temperature": 0.000001,
    }
    print(f"Generation args: {generation_kwargs}")
    # specify sequence classification arguments, needed for discriminative model
    classification_kwargs = {
        "aggregation_strategy": "none",
        "ignore_labels": [],
    }
    print(f"Classification args: {classification_kwargs}")
    # retrieve examples from input dataset
    examples = get_classification_examples(
        args.input_data_json_path,
        nbr_examples=args.nbr_examples,
        randomize_selection=args.randomize_selection,
        randomize_order=args.randomize_order,
        random_seed=args.random_seed,
    )
    # instantiate inference pipeline
    classification_pipeline = ClassificationInference(
        model_type=args.model_type,
        prompt_template_path=args.prompt_template,
        use_tgi=args.use_tgi,
        tgi_endpoint=args.tgi_endpoint,
        use_aip=args.use_aip,
        quantized=args.quantized,
        classification_kwargs=classification_kwargs
        if args.model_type == "roberta"
        else generation_kwargs,
        examples=examples,
    )

    # Skip evaluation if the --inference-only flag is passed.
    perform_inference(
        input_data_json_path=args.input_data_json_path,
        output_eval_res_dir=args.output_eval_res_dir,
        inference_pipeline=classification_pipeline,
    )


def run_token_classification_both(args):
    run_token_classification_inference(args)

    predictions_filename = os.path.join(args.output_eval_res_dir, "predictions.json")
    perform_evaluation(
        input_data_json_path=args.input_data_json_path,
        output_eval_res_dir=args.output_eval_res_dir,
        predictions_filename=predictions_filename,
    )


def get_model_info(
    model_type: str,
    prompt_template: str,
    classification_kwargs: dict[str, Any] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    quantized: bool = False,
    examples: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Get the generation_kwargs and model_path from the config file.

    Get the generation_kwargs and model_path from the config file
    and the prompt from prompt_templates.py based on the model_type.

    Parameters
    ----------
    model_type : str
        The type of model.
    generation_kwargs : Optional[Dict[str, Any]]
        Additional keyword arguments for the model generation. Defaults to None.
    quantized: bool, optional
        Whether or not a quantized model is used

    Returns
    -------
    Dict[str, Any]
        A dictionary containing generation_kwargs, model_path, and prompt.
    """
    model_path = config.MODELS[model_type]
    builder = PromptBuilder(template=prompt_template)
    prompt = builder.run(examples=examples)["prompt"]

    model_info = {
        "generation_kwargs": generation_kwargs if generation_kwargs else {},
        "classification_kwargs": classification_kwargs if classification_kwargs else {},
        "model_path": model_path,
        "quantized": quantized,
        "prompt": prompt,
    }

    return model_info


if __name__ == "__main__":
    NOW = str(datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S"))

    parser = argparse.ArgumentParser(description="Run token classification evaluation")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Inference subparser
    inference_parser = subparsers.add_parser("inference", help="Perform inference")
    inference_parser.add_argument(
        "-i",
        "--input_data_json_path",
        type=str,
        help="Path to the input JSON data file.",
        required=True,
    )
    inference_parser.add_argument(
        "-o",
        "--output_eval_res_dir",
        type=str,
        help="Path to store evaluation results.",
        required=True,
    )
    inference_parser.add_argument(
        "-m", "--model_type", type=str, help="Model type", required=True
    )
    inference_parser.add_argument(
        "-t",
        "--use_tgi",
        action="store_true",
        help="whether to use TGI (default: False)",
    )
    inference_parser.add_argument(
        "--tgi_endpoint",
        type=str,
        help="a specific TGI endpoint that is different from the default http://127.0.0.1:3000/generate (default: None)",
    )
    inference_parser.add_argument(
        "-a",
        "--use_aip",
        action="store_true",
        help="whether to use AIP endpoint (default: False)",
    )
    inference_parser.add_argument(
        "-q",
        "--quantized",
        action="store_true",
        help="whether to use quantized model (bitsandbytes)",
    )
    inference_parser.add_argument(
        "--prompt-template", type=str, help="path to prompt template file"
    )
    inference_parser.add_argument(
        "--nbr_examples",
        type=int,
        help="specify number of examples to include in prompt",
    )
    inference_parser.add_argument(
        "--randomize_selection",
        action="store_true",
        help="specify whether selection of examples should be randomized (default: False)",
        default=False,
        required=False,
    )
    inference_parser.add_argument(
        "--randomize_order",
        action="store_true",
        help="specify whether ordering of examples should be randomized (default: False)",
        default=False,
        required=False,
    )
    inference_parser.add_argument(
        "--random_seed",
        type=int,
        help="specify number with which to seed random selection of examples",
        default=49,
    )

    # Evaluation subparser
    evaluation_parser = subparsers.add_parser("evaluation", help="Perform evaluation")
    evaluation_parser.add_argument(
        "-i",
        "--input_data_json_path",
        type=str,
        help="Path to the input JSON data file.",
        required=True,
    )
    evaluation_parser.add_argument(
        "-o",
        "--output_eval_res_dir",
        type=str,
        help="Path to the output evaluation results directory.",
        required=True,
    )
    evaluation_parser.add_argument(
        "--predictions_filename",
        type=str,
        help="Path to predictions JSON file",
        required=True,
    )

    # Both subparser
    both_parser = subparsers.add_parser(
        "both", help="Perform both inference and evaluation"
    )
    both_parser.add_argument(
        "-i",
        "--input_data_json_path",
        type=str,
        help="Path to the input JSON data file.",
        required=True,
    )
    both_parser.add_argument(
        "-o",
        "--output_eval_res_dir",
        type=str,
        help="Path to store evaluation results.",
        required=True,
    )
    both_parser.add_argument(
        "-m", "--model_type", type=str, help="Model type", required=True
    )
    both_parser.add_argument(
        "-t",
        "--use_tgi",
        action="store_true",
        help="whether to use TGI (default: False)",
    )
    both_parser.add_argument(
        "--tgi_endpoint",
        type=str,
        help="a specific TGI endpoint that is different from the default http://127.0.0.1:3000/generate (default: None)",
    )
    both_parser.add_argument(
        "-a",
        "--use_aip",
        action="store_true",
        help="whether to use AIP endpoint (default: False)",
    )
    both_parser.add_argument(
        "-q",
        "--quantized",
        action="store_true",
        help="whether to use quantized model (bitsandbytes)",
    )
    both_parser.add_argument(
        "--prompt-template", type=str, help="path to prompt template file"
    )
    both_parser.add_argument(
        "--nbr_examples",
        type=int,
        help="specify number of examples to include in prompt",
    )
    both_parser.add_argument(
        "--randomize_selection",
        action="store_true",
        help="specify whether selection of examples should be randomized (default: False)",
        default=False,
        required=False,
    )
    both_parser.add_argument(
        "--randomize_order",
        action="store_true",
        help="specify whether ordering of examples should be randomized (default: False)",
        default=False,
        required=False,
    )
    both_parser.add_argument(
        "--random_seed",
        type=int,
        help="specify number with which to seed random selection of examples",
        default=49,
    )
    args = parser.parse_args()

    print(f"Args: {args}")
    args.output_eval_res_dir = f"{args.output_eval_res_dir}_{NOW}"
    print(args.output_eval_res_dir)
    if not os.path.exists(args.output_eval_res_dir):
        os.makedirs(args.output_eval_res_dir)
    # Handle the different modes
    if args.mode == "inference":
        run_token_classification_inference(args)
    elif args.mode == "evaluation":
        run_token_classification_evaluation(args)
    elif args.mode == "both":
        run_token_classification_both(args)
