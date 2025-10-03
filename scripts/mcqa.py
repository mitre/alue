"""Entry point script for Multiple Choice Question Answering tasks.

This module provides a command-line interface for running inference and evaluation
on MCQA tasks. It supports running inference only, evaluation only, or both in
sequence.
"""

import argparse
from datetime import datetime
import json
import os

from alue.data_utils import load_data
from alue.prompt_utils import build_messages
from alue.inference import run_llm_inference
from alue.evaluation import MCQAEval

from .utils import load_schema, parse_fields_to_extract


def run_inference(args: argparse.Namespace) -> str:
    """Run multiple choice QA inference on a dataset.
    
    Loads data, builds prompts with few-shot examples, runs LLM inference,
    and saves predictions and results to the output directory.
    
    Args:
        args: Parsed command-line arguments containing:
            - input_data_json_path: Path to input data file
            - output_dir: Directory to save results
            - model_name: Model identifier for inference
            - task_type: Task type for template selection
            - num_examples: Number of few-shot examples
            - num_questions: Optional limit on questions to process
            - schema_class: Optional Pydantic schema for structured output
            - field_to_extract: Field(s) to extract from structured response
            - temperature: Sampling temperature
            - max_tokens: Maximum tokens to generate
            
    Returns:
        Path to the saved predictions JSON file.
        
    Example:
        >>> args = parser.parse_args(['inference', '-i', 'mcqa.json', ...])
        >>> predictions_file = run_inference(args)
        >>> print(f"Predictions saved to {predictions_file}")
    """
    
    print(f"MCQA Inference: {args.model_name}")
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

    for item in test_data:
        message = build_messages(
            task_type=args.task_type,
            system_kwargs={'examples': examples},
            user_kwargs={'input': item['input']}
        )
        messages.append(message)
        ground_truth.append(item['output'])
        question_ids.append(item['id'])

    # Load schema and run inference
    schema = load_schema(args.schema_class)
    
    print("Running inference...")
    predictions = run_llm_inference(
        messages=messages,
        model_name=args.model_name,
        schema=schema,
        fields_to_extract=args.field_to_extract,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )


    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save predictions for evaluation
    predictions_file = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_file, 'w') as f:
        json.dump({str(qid): pred for qid, pred in zip(question_ids, predictions)}, f, indent=2)

    # Save full results
    results = {
        "model": args.model_name,
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


def run_evaluation(args: argparse.Namespace) -> None:
    """Run evaluation on MCQA prediction results.
    
    Loads predictions and ground truth data, and computes evaluation metrics
    specific to multiple choice questions (accuracy, per-class metrics, etc.).
    
    Args:
        args: Parsed command-line arguments containing:
            - input_data_json_path: Path to input data file with ground truth
            - predictions_file: Path to predictions JSON file
            - output_dir: Directory to save evaluation results
            
    Example:
        >>> args = parser.parse_args(['evaluation', '-i', 'mcqa.json', ...])
        >>> run_evaluation(args)
        Evaluation complete. Results saved to output_dir/
    """
    print("Running evaluation...")
    eval_engine = MCQAEval(
        data_file=args.input_data_json_path,
        pred_file=args.predictions_file,
        out_dir=args.output_dir
    )
    eval_engine.perform_evaluation()


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add inference-related arguments to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "-i", "--input_data_json_path",
        required=True,
        help="Path to input JSON data file"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "-m", "--model_name",
        required=True,
        help="Model name (e.g., gpt-4o-mini)"
    )
    parser.add_argument(
        "--task_type",
        default="aviation_exam",
        help="Task type for prompt templates"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=3,
        help="Number of few-shot examples"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        help="Limit number of questions (default: all)"
    )
    parser.add_argument(
        "--schema_class",
        help="Pydantic schema class (e.g., MCQAResponse)"
    )
    parser.add_argument(
        "--field_to_extract",
        type=parse_fields_to_extract,
        default="answer",
        help="Field(s) to extract from structured response. Can be single field, "
             "comma-separated list, or 'none' for full response"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands for inference and evaluation.
    
    Creates a parser with three subcommands:
    - inference: Run inference only
    - evaluation: Run evaluation only
    - both: Run inference followed by evaluation
    
    Returns:
        Configured ArgumentParser with all subcommands.
        
    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(['both', '-i', 'mcqa.json', ...])
    """
    parser = argparse.ArgumentParser(description="MCQA script")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Inference subparser
    inf_parser = subparsers.add_parser("inference", help="Run inference only")
    add_inference_args(inf_parser)

    # Evaluation subparser  
    eval_parser = subparsers.add_parser("evaluation", help="Run evaluation only")
    eval_parser.add_argument(
        "-i", "--input_data_json_path",
        required=True,
        help="Path to input JSON data file"
    )
    eval_parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Output directory for results"
    )
    eval_parser.add_argument(
        "--predictions_file",
        required=True,
        help="Path to predictions JSON file"
    )

    # Both subparser
    both_parser = subparsers.add_parser(
        "both",
        help="Run inference + evaluation"
    )
    add_inference_args(both_parser)

    return parser


def main() -> None:
    """Main entry point for the MCQA script.
    
    Parses command-line arguments, adds timestamp to output directory,
    and executes the requested mode (inference, evaluation, or both).
    
    The script supports three modes:
    1. inference: Generate predictions from a model
    2. evaluation: Evaluate existing predictions
    3. both: Run inference then evaluation in sequence
    
    Example:
        Run inference only:
        $ python mcqa.py inference -i mcqa_data.json -o results -m gpt-4
        
        Run evaluation only:
        $ python mcqa.py evaluation -i mcqa_data.json -o results \\
            --predictions_file results_20240101_120000/predictions.json
        
        Run both:
        $ python mcqa.py both -i mcqa_data.json -o results -m gpt-4 \\
            --schema_class MCQAResponse --field_to_extract answer
    """
    parser = create_parser()
    args = parser.parse_args()

    # Add timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = f"{args.output_dir}_{timestamp}"

    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")

    # Execute based on mode
    if args.mode == "inference":
        run_inference(args)
    elif args.mode == "evaluation":
        run_evaluation(args)
    elif args.mode == "both":
        predictions_file = run_inference(args)
        args.predictions_file = predictions_file
        run_evaluation(args)


if __name__ == "__main__":
    main()