"""Entry point script for Summarization tasks.

This module provides a command-line interface for running inference and evaluation
on summarization tasks. It supports generating summaries from input texts and
comprehensive evaluation using LLM-as-judge metrics.
"""

import argparse
import os
from datetime import datetime
from typing import Any


from alue.data_utils import load_data
from alue.prompt_utils import build_messages
from alue.inference import run_llm_inference
from alue.evaluation import SummarizationEval
from .utils import load_schema, parse_fields_to_extract
import json


def run_inference(args: argparse.Namespace) -> str:
    """Run Summarization inference with text generation.
    
    For each input text:
    1. Loads input narratives and ground truth summaries
    2. Builds prompts with few-shot examples
    3. Generates summaries using LLM
    4. Saves predictions for evaluation
    
    Args:
        args: Parsed command-line arguments containing:
            - input_data_json_path: Path to input data file
            - output_dir: Directory to save results
            - inference_model_name: Model identifier for generation
            - task_type: Task type for template selection
            - num_examples: Number of few-shot examples
            - num_questions: Optional limit on texts to process
            - schema_class: Optional Pydantic schema for structured output
            - field_to_extract: Field(s) to extract from structured response
            - temperature: Sampling temperature
            - max_tokens: Maximum tokens to generate
            
    Returns:
        Path to the saved predictions JSON file.
        
    Example:
        >>> args = parser.parse_args(['inference', '-i', 'summarization_data.json', ...])
        >>> predictions_file = run_inference(args)
        >>> print(f"Predictions saved to {predictions_file}")
    """
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
    input_ids = []  
    inputs = []
    ground_truth_summaries = []

    for item in test_data:
        print(item)
        
        input_text = item["input"]
        message = build_messages(
            task_type=args.task_type,
            system_kwargs={"examples": examples},
            user_kwargs={"input": input_text}
        )
        
        messages.append(message)
        input_ids.append(item['id'])
        inputs.append(input_text)
        ground_truth_summaries.append(item["output"])

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
    
    
    predictions_dict = {}
    for pred, input_id, question, gt_summary in zip(predictions, input_ids, inputs, ground_truth_summaries):
        predictions_dict[input_id] = {  # Use the actual ID as the key
            "narrative": question,  # Original input text to be summarized
            "ground_truth_summary": gt_summary,  # Reference summary
            "predicted_summary": pred  # Model's predicted summary
        }
    
    predictions_file = os.path.join(args.output_dir, "predictions.json")
    with open(predictions_file, 'w') as f:
        json.dump(predictions_dict, f, indent=2)

    # Save metadata
    metadata = {
        "model": args.inference_model_name,
        "task_type": args.task_type,
        "num_questions": len(test_data),
        "num_examples": args.num_examples,
        "total_predictions": len(predictions),
        "temperature": args.temperature
    }
    
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved predictions to: {predictions_file}")
    print(f"Saved metadata to: {metadata_file}")

    return predictions_file


def run_evaluation(args: argparse.Namespace) -> None:
    """Run comprehensive summarization evaluation on prediction results.
    
    Evaluates summary quality using LLM-as-judge metrics including relevance,
    coherence, consistency, and fluency.
    
    Args:
        args: Parsed command-line arguments containing:
            - predictions_file: Path to predictions JSON file
            - output_dir: Directory to save evaluation results
            - llm_judge_model_name: Model for LLM-as-judge evaluation
            - verbose: Whether to output detailed explanations
            
    Example:
        >>> args = parser.parse_args(['evaluation', '--predictions_file', 'predictions.json', ...])
        >>> run_evaluation(args)
        Evaluation complete. Summary quality metrics: {...}
    """
    print("Running summarization evaluation...")
    eval_engine = SummarizationEval(
        pred_file=args.predictions_file,
        out_dir=args.output_dir,
        model_name=args.llm_judge_model_name,
        verbose=args.verbose
    )
    eval_engine.perform_evaluation()


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add summarization inference-related arguments to an argument parser.
    
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
        "-m", "--inference_model_name",
        required=True,
        help="Model name (e.g., gpt-4o-mini)"
    )
    parser.add_argument(
        "--task_type",
        default="rag",
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


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    """Add summarization evaluation-related arguments to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--llm_judge_model_name",
        required=True,
        help="Model name for LLM judges"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output with explanations"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands for summarization inference and evaluation.
    
    Creates a parser with three subcommands:
    - inference: Run summarization generation only
    - evaluation: Run evaluation only on existing predictions
    - both: Run inference followed by evaluation
    
    Returns:
        Configured ArgumentParser with all subcommands.
        
    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(['both', '-i', 'summarization_data.json', ...])
    """
    parser = argparse.ArgumentParser(description="Summarization script")
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
    add_evaluation_args(eval_parser)

    # Both subparser
    both_parser = subparsers.add_parser("both", help="Run inference + evaluation")
    add_inference_args(both_parser)
    add_evaluation_args(both_parser)

    return parser


def main() -> None:
    """Main entry point for the Summarization script.
    
    Parses command-line arguments, adds timestamp to output directory,
    and executes the requested mode (inference, evaluation, or both).
    
    The script supports three modes:
    1. inference: Generate summaries for input texts
    2. evaluation: Evaluate existing predictions using LLM-as-judge
    3. both: Run inference then evaluation in sequence
    
    Example:
        Run inference only:
        $ python summarization.py inference -i summarization_data.json -o results \\
            -m gpt-4
        
        Run evaluation only:
        $ python summarization.py evaluation -i summarization_data.json -o results \\
            --predictions_file results_20240101_120000/predictions.json \\
            --llm_judge_model_name gpt-4
        
        Run both:
        $ python summarization.py both -i summarization_data.json -o results \\
            -m gpt-4 --llm_judge_model_name gpt-4 --verbose
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