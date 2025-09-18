import argparse
import os
from datetime import datetime
from typing import Any

import output_normalizations
from evaluation import SequenceClassificationEval
from haystack.components.builders import PromptBuilder
from inference import ClassificationInference
from io_utilities import get_classification_examples

from alue import config


def run_sequence_classification_evaluation(args):
    eval = SequenceClassificationEval(
        data_file=args.input_data_json_path,
        pred_file=args.predictions_filename,
        out_dir=args.output_eval_res_dir,
    )
    # check if schema or schema_path is provided
    print(f"ARGUMENTS: {args}")
    if ("schema" in args and args.schema is not None) or (
        "schema_path" in args and args.schema_path is not None
    ):
        schema = True
    else:
        schema = False

    eval.perform_evaluation(
        is_labels=args.is_labels,
        is_multiclass=args.is_multiclass,
        normalize=args.normalize_output,
        exact_match_only=args.exact_match_only,
        schema=schema,
        use_label_names=args.use_label_names,
        task_normalization_name=args.task_normalization_name,
        task_specific_normalization_lookup=output_normalizations.TASK_SPECIFIC_PATTERNS[
            args.task_normalization_name
        ],
        output_report_name="classification_report.json",
    )


def run_sequence_classification_inference(args):
    # specify generation arguments, specific to classification with a generative model
    generation_kwargs = {
        "return_full_text": False,
        "max_new_tokens": 50,
    }
    # specify sequence classification arguments, needed for discriminative model
    classification_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "default",
    }

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
        vllm_offline=args.vllm_offline,
        num_gpus=args.num_gpus,
        use_aip=args.use_aip,
        quantized=args.quantized,
        classification_kwargs=classification_kwargs
        if args.model_type == "roberta"
        else generation_kwargs,
        examples=examples,
        batch_size=args.batch_size,
        schema_path=args.schema_path,
    )

    classification_pipeline.full_inference(
        input_data_json_path=args.input_data_json_path,
        output_eval_res_dir=args.output_eval_res_dir,
        batch_size=args.batch_size,
    )


def run_sequence_classification_both(args):
    run_sequence_classification_inference(args)

    predictions_filename = os.path.join(args.output_eval_res_dir, "predictions.json")
    args.predictions_filename = predictions_filename
    run_sequence_classification_evaluation(args)


def get_model_info(
    model_type: str,
    prompt_template: str,
    classification_kwargs: dict[str, Any] | None = None,
    generation_kwargs: dict[str, Any] | None = None,
    quantized: bool = False,
    examples: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Gets the generation_kwargs and model_path from the config file
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

    parser = argparse.ArgumentParser(
        description="Run sequence classification evaluation"
    )
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
        "--vllm_offline", action="store_true", help="whether to use vLLM offline"
    )
    inference_parser.add_argument(
        "--num_gpus", type=int, help="number of GPUs to use for vLLM offline", default=1
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
    inference_parser.add_argument("--schema-path", type=str, help="path to schema file")
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
    inference_parser.add_argument(
        "--batch_size", type=int, help="specify batch size", default=8
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
    evaluation_parser.add_argument(
        "--is_labels",
        action="store_true",
        help="flag indicating if prediction output consists of string labels instead of integer indices",
        default=False,
        required=False,
    )
    evaluation_parser.add_argument(
        "--is_multiclass",
        action="store_true",
        help="flag indicating if the model is multiclass (one label class per sample) instead of multilabel (multiple label classes per sample)",
        default=False,
        required=False,
    )
    evaluation_parser.add_argument(
        "--normalize_output",
        action="store_true",
        help="whether or not to perform normalization on the raw prediction output",
        default=False,
        required=False,
    )
    evaluation_parser.add_argument(
        "--exact_match_only",
        action="store_true",
        help="only accept exact pattern matches in output when normalizing raw prediction output",
        default=False,
        required=False,
    )
    evaluation_parser.add_argument(
        "--task_normalization_name",
        type=str,
        help="name of task when retrieving output normalization regular expressions from lookup",
        default=None,
        required=False,
    )
    evaluation_parser.add_argument(
        "--schema",
        action="store_true",
        help="if schema was used or not",
        default=False,
        required=False,
    )
    evaluation_parser.add_argument(
        "--use_label_names",
        action="store_true",
        help="if passing in label names or not",
        default=False,
        required=False,
    )

    # # Both subparser
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
        "--vllm_offline", action="store_true", help="whether to use vLLM offline"
    )
    both_parser.add_argument(
        "--num_gpus", type=int, help="number of GPUs to use for vLLM offline", default=1
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
    both_parser.add_argument("--schema-path", type=str, help="path to schema file")
    both_parser.add_argument(
        "--is_labels",
        action="store_true",
        help="flag indicating if prediction output consists of string labels instead of integer indices",
        default=False,
        required=False,
    )
    both_parser.add_argument(
        "--is_multiclass",
        action="store_true",
        help="flag indicating if the model is multiclass (one label class per sample) instead of multilabel (multiple label classes per sample)",
        default=False,
        required=False,
    )
    both_parser.add_argument(
        "--normalize_output",
        action="store_true",
        help="whether or not to perform normalization on the raw prediction output",
        default=False,
        required=False,
    )
    both_parser.add_argument(
        "--exact_match_only",
        action="store_true",
        help="only accept exact pattern matches in output when normalizing raw prediction output",
        default=False,
        required=False,
    )
    both_parser.add_argument(
        "--task_normalization_name",
        type=str,
        help="name of task when retrieving output normalization regular expressions from lookup",
        default=None,
        required=False,
    )
    both_parser.add_argument(
        "--batch_size", type=int, help="specify batch size", default=8
    )
    both_parser.add_argument(
        "--use_label_names",
        action="store_true",
        help="if passing in label names or not",
        default=False,
        required=False,
    )

    # # Inference arguments

    # parse commandline arguments
    args = parser.parse_args()
    # print(args)

    args.output_eval_res_dir = f"{args.output_eval_res_dir}_{NOW}"
    if not os.path.exists(args.output_eval_res_dir):
        os.makedirs(args.output_eval_res_dir)

    if args.mode == "inference":
        # run inference
        run_sequence_classification_inference(args)
    elif args.mode == "evaluation":
        # run evaluation
        run_sequence_classification_evaluation(args)
    elif args.mode == "both":
        # run inference and evaluation
        run_sequence_classification_both(args)

    # log model info
    # prompt_template = load_prompt_from_file(args.prompt_template)
    # specify generation arguments, specific to classification with a generative model
    generation_kwargs = {
        "return_full_text": False,
        "max_new_tokens": 200,
    }
    # specify sequence classification arguments, needed for discriminative model
    classification_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "default",
    }
    # examples = get_classification_examples(
    #     args.input_data_json_path,
    #     nbr_examples=args.nbr_examples,
    #     randomize_selection=args.randomize_selection,
    #     randomize_order=args.randomize_order,
    #     random_seed=args.random_seed,
    # )
    # model_info = get_model_info(
    #     model_type=args.model_type,
    #     prompt_template=args.prompt_template,
    #     classification_kwargs=classification_kwargs if args.model_type == "roberta" else generation_kwargs,
    #     quantized=args.quantized,
    #     examples=examples,
    # )
    # model_info_filename = os.path.join(args.output_eval_res_dir, "model_info.json")
    # with open(model_info_filename, "w") as file:
    #     json.dump(model_info, file)
