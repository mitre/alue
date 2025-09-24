import argparse
import os
from datetime import datetime
from typing import Any

from alue.data_utils import load_data
from alue.prompt_utils import build_messages

from alue.evaluation import ExtractiveQAEval
from alue.output_normalizations import normalize_tail_extraction_predictions


import argparse
import json
import os
from datetime import datetime

from alue.data_utils import load_data
from alue.prompt_utils import build_messages
from alue.inference import run_llm_inference

import sys


def load_schema(schema_class_name: str):
    """Load Pydantic schema class dynamically."""
    if not schema_class_name:
        return None
        
    try:
        module_name, class_name = schema_class_name.rsplit('.', 1)
        module = __import__(module_name, fromlist=[class_name])
        schema = getattr(module, class_name)
        print(f"Using schema: {schema.__name__}")
        return schema
        
    except Exception as e:
        print(f"Warning: Could not import schema {schema_class_name}: {e}")
        return None
    
def load_normalizer(normalizer_name: str):
    """Load normalization function dynamically."""
    if not normalizer_name:
        return None
        
    try:
        if '.' in normalizer_name:
            # Full module path provided
            module_name, func_name = normalizer_name.rsplit('.', 1)
        else:
            # Just function name, use default path
            module_name = "alue.output_normalizations"
            func_name = normalizer_name

        module = __import__(module_name, fromlist=[func_name])
        normalizer_func = getattr(module, func_name)
        print(f"Using normalizer: {normalizer_func.__name__}")
        return normalizer_func
        
    except Exception as e:
        print(f"Warning: Could not import normalizer {normalizer_name}: {e}")
        return None
    

def run_inference(args):
    """Run Extractive QA Inference"""

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
            user_kwargs={
                'query': item['input'],
                'context': item['context']}
        )
        messages.append(message)
        ground_truth.append(item['output'])
        question_ids.append(item["metadata"]['id'])


    # Load schema and run inference
    schema = load_schema(args.schema_class)

    print("Running inference...")
    predictions = run_llm_inference(
        messages=messages,
        model_name=args.model_name,
        backend_type=args.backend_type,
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
        "backend_type": args.backend_type,
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


def run_evaluation(args):
    print("Running evaluation...")
    
    # Load normalizer function
    normalizer_func = load_normalizer(args.normalizer_func) if hasattr(args, 'normalizer_func') and args.normalizer_func else None
    
    eval = ExtractiveQAEval(
        data_file=args.input_data_json_path,
        pred_file=args.predictions_file,
        out_dir=args.output_dir,
        normalizer_func=normalizer_func
    )
    eval.perform_evaluation()



def create_parser():
    """Create argument parser with shared arguments."""
    parser = argparse.ArgumentParser(description="Extractive QA script")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared inference arguments
    def add_inference_args(p):
        p.add_argument("-i", "--input_data_json_path", required=True,
                      help="Path to input JSON data file")
        p.add_argument("-o", "--output_dir", required=True,
                      help="Output directory for results")
        p.add_argument("-m", "--model_name", required=True,
                      help="Model name (e.g., gpt-4o-mini)")
        p.add_argument("--backend_type", default="openai",
                      help="Backend type (openai, tgi, etc.)")
        p.add_argument("--task_type", default="aviation_exam",
                      help="Task type for prompt templates")
        p.add_argument("--num_examples", type=int, default=3,
                      help="Number of few-shot examples")
        p.add_argument("--num_questions", type=int,
                      help="Limit number of questions (default: all)")
        p.add_argument("--schema_class",
                      help="Pydantic schema class (e.g., MCQAResponse)")
        p.add_argument("--field_to_extract", default="answer",
                      help="Field to extract from structured response")
        p.add_argument("--temperature", type=float, default=0.1,
                      help="Generation temperature")
        p.add_argument("--max_tokens", type=int, default=150,
                      help="Maximum tokens to generate")

    # Shared evaluation arguments
    def add_evaluation_args(p):
        p.add_argument("--normalizer_func",
                      help="Normalization function (e.g., normalize_tail_extraction_predictions)")

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
    add_evaluation_args(eval_parser)

    # Both subparser
    both_parser = subparsers.add_parser("both", help="Run inference + evaluation")
    add_inference_args(both_parser)
    add_evaluation_args(both_parser)  # <- This line was missing

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