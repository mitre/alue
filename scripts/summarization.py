import argparse
import os
from datetime import datetime
from typing import Any

from haystack.components.builders import PromptBuilder

# from evaluation import ExtractiveQAEval
from inference import SummarizationInference
from io_utilities import get_summarization_dataset

from alue import config

# def run_extractive_qa_evaluation(args):
#     eval = ExtractiveQAEval(
#         data_file=args.input_data_json_path,
#         pred_file=args.predictions_filename,
#         out_dir=args.output_eval_res_dir,
#         llm_judge_model=args.llm_judge_model
#     )
#     eval.perform_evaluation(llm_judge_examples=args.llm_judge_examples)


def run_summarization_inference(args):
    generation_kwargs = {"return_full_text": False, "max_new_tokens": 512, "top_k": 0}
    examples = get_summarization_dataset(
        input_json_data_path=args.input_data_json_path,
        load_examples=True,
        nbr_examples=args.nbr_examples,
        randomize_selection=args.randomize_selection,
        randomize_order=args.randomize_order,
        random_seed=args.random_seed,
    )
    inference_pipeline = SummarizationInference(
        model_type=args.model_type,
        generation_kwargs=generation_kwargs,
        prompt_template_path=args.prompt_template,
        examples=examples,
        use_aip=args.use_aip,
        use_tgi=args.use_tgi,
        quantized=args.quantized,
        batch_size=args.batch_size,
    )
    inference_pipeline.full_inference(
        input_data_json_path=args.input_data_json_path,
        output_eval_res_dir=args.output_eval_res_dir,
        batch_size=args.batch_size,
    )


def run_summarization_both(args):
    run_summarization_inference(args)

    # predictions_filename = os.path.join(args.output_eval_res_dir, "predictions.json")
    # args.predictions_filename = predictions_filename

    # run_extractive_qa_evaluation(args)


def get_model_info(
    model_type: str,
    prompt_template: str,
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
        "model_path": model_path,
        "quantized": quantized,
        "prompt": prompt,
    }

    return model_info


if __name__ == "__main__":
    NOW = str(datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S"))

    parser = argparse.ArgumentParser(
        description="Run extractive question answering evaluation"
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
    inference_parser.add_argument(
        "--batch_size", type=int, help="specify batch size", default=8
    )

    # Evaluation subparser
    # evaluation_parser = subparsers.add_parser("evaluation", help="Perform evaluation")
    # evaluation_parser.add_argument("-i", "--input_data_json_path", type=str, help="Path to the input JSON data file.", required=True)
    # evaluation_parser.add_argument("-o", "--output_eval_res_dir", type=str, help="Path to the output evaluation results directory.", required=True)
    # evaluation_parser.add_argument("--predictions_filename", type=str, help="Path to predictions JSON file", required=True)
    # evaluation_parser.add_argument("--llm_judge_model", type=str, help="Name of the LLM Judge model (optional)", required=False)
    # evaluation_parser.add_argument("--llm_judge_examples", type=str, help="Path to LLM Judge examples file (optional)", required=False)

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
    both_parser.add_argument(
        "--batch_size", type=int, help="specify batch size", default=8
    )
    # both_parser.add_argument("--llm_judge_model", type=str, help="Name of the LLM Judge model (optional)", required=False)
    # both_parser.add_argument("--llm_judge_examples", type=str, help="Path to LLM Judge examples file (optional)", required=False)
    args = parser.parse_args()

    print(f"Args: {args}")
    args.output_eval_res_dir = f"{args.output_eval_res_dir}_{NOW}"
    print(args.output_eval_res_dir)
    if not os.path.exists(args.output_eval_res_dir):
        os.makedirs(args.output_eval_res_dir)
    # Handle the different modes
    if args.mode == "inference":
        run_summarization_inference(args)
    # elif args.mode == "evaluation":
    #     run_extractive_qa_evaluation(args)
    elif args.mode == "both":
        run_summarization_both(args)
