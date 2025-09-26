import argparse
import os
from datetime import datetime
from typing import Any

from haystack.components.builders import PromptBuilder

# from evaluation import ExtractiveQAEval
from alue.data_utils import load_data
from alue.prompt_utils import build_messages
from alue.inference import run_llm_inference
from alue.evaluation import SummarizationEval
from .utils import load_schema, load_normalizer
import json


def run_inference(args):
    """Run Summarization inference."""
    print(f"Summarization Inference: {args.inference_model_name}")
    print("=" * 50)

    # Load data
    print("Loading data...")
    loader = load_data(args.input_data_json_path)
    examples = loader.get_examples(num_examples=args.num_examples)
    test_data = loader.get_test_data()

    if args.num_questions:
        test_data = test_data[:args.num_questions]

    print(f"Loaded {len(test_data)} questions, using {len(examples)} examples")

    # Build messages
    print("Building messages...")
    messages = []
    ground_truth = []
    question_ids = []
    all_retrieved_doc_ids = []  
    questions = []
    ground_truth_answers = []

    for item in test_data:
        print(item)
        
        question = item["input"]
        retrieved_docs = chroma.query_collection(
            query=question,
            collection_name=args.collection_name,
            embedding_function=embedding_function,
            n_results=args.top_k
        )
        

        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"Document {i}:\n{doc['text']}")

        context = "\n\n".join(context_parts)

        message = build_messages(
            task_type=args.task_type,
            system_kwargs={"examples": examples},
            user_kwargs={"query": question, "context": context}
        )
        messages.append(message)
        ground_truth.append(item['output'])
        question_ids.append(item["metadata"]['id'])
        all_retrieved_doc_ids.append([doc["id"] for doc in retrieved_docs]) 
        questions.append(question)
        ground_truth_answers.append(item["output"])

    # Load schema and run inference
    schema = load_schema(args.schema_class)
    
    print("Running inference...")
    predictions = run_llm_inference(
        messages=messages,
        model_name=args.inference_model_name,
        schema=schema,
        fields_to_extract=args.field_to_extract,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions with document IDs and questions in the desired format
    predictions_dict = {}
    for i, (pred, doc_ids, question, gt_answer) in enumerate(zip(predictions, all_retrieved_doc_ids, questions, ground_truth_answers)):
        predictions_dict[str(i)] = {  # Use index as key like in your example
            "answer": pred,
            "ground_truth_answer": gt_answer,
            "predicted_doc_ids": doc_ids,  # Already extracted document IDs
            "question": question
        }
    
    predictions_file = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions_dict, f, indent=2)

    # Save full results (keeping the original format for other purposes)
    results = {
        "model": args.inference_model_name,
        "task_type": args.task_type,
        "num_questions": len(test_data),
        "num_examples": args.num_examples,
        "total": len(predictions),
        "predictions": predictions,
        "ground_truth": ground_truth,
        "questions": [item['input'] for item in test_data],
        "temperature": args.temperature
    }
    
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved to: {args.output_dir}")

    return predictions_file



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
