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


def run_evaluation(args):
    """Run Summarization evaluation."""
    print("Running summarization evaluation...")
    eval_engine = SummarizationEval(
        pred_file=args.predictions_file,
        out_dir=args.output_dir,
        model_name=args.llm_judge_model_name,
        verbose=args.verbose
    )
    eval_engine.perform_evaluation()


def create_parser():
    """Create argument parser with shared arguments."""
    parser = argparse.ArgumentParser(description="MCQA script")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared inference arguments
    def add_inference_args(p):
        p.add_argument("-i", "--input_data_json_path", required=True,
                      help="Path to input JSON data file")
        p.add_argument("-o", "--output_dir", required=True,
                      help="Output directory for results")
        p.add_argument("-m", "--inference_model_name", required=True,
                      help="Model name (e.g., gpt-4o-mini)")
        p.add_argument("--task_type", default="rag",
                      help="Task type for prompt templates")
        p.add_argument("--num_examples", type=int, default=3,
                      help="Number of few-shot examples")
        p.add_argument("--num_questions", type=int,
                      help="Limit number of questions (default: all)")
        p.add_argument("--schema_class",
                      help="Pydantic schema class (e.g., MCQAResponse)")
        p.add_argument("--field_to_extract", type=parse_fields_to_extract, default="answer",
                       help="Field(s) to extract from structured response. Can be single field, comma-separated list, or 'none' for full response")
        p.add_argument("--temperature", type=float, default=0.1,
                      help="Generation temperature")
        p.add_argument("--max_tokens", type=int, default=150,
                      help="Maximum tokens to generate")

    # Inference subparser
    inf_parser = subparsers.add_parser("inference", help="Run inference only")
    add_inference_args(inf_parser)

    # Evaluation subparser  
    eval_parser = subparsers.add_parser("evaluation", help="Run evaluation only")
    eval_parser.add_argument("-i", "--input_data_json_path", required=True,
                            help="Path to input JSON data file")
    eval_parser.add_argument("-o", "--output_dir", required=True,
                            help="Output directory for results")
    eval_parser.add_argument("--predictions_file", required=True,
                            help="Path to predictions JSON file")
    eval_parser.add_argument("--llm_judge_model_name", required=True,
                            help="Model name for LLM judges")
    eval_parser.add_argument("--verbose", action="store_true",
                            help="Verbose output with explanations")

    # Both subparser
    both_parser = subparsers.add_parser("both", help="Run inference + evaluation")
    add_inference_args(both_parser)
    # Add evaluation-specific arguments
    both_parser.add_argument("--llm_judge_model_name", required=True,
                            help="Model name for LLM judges")
    both_parser.add_argument("--verbose", action="store_true",
                        help="Verbose output with explanations")

    return parser


def main():
    """Main entry point."""
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





