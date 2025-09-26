
"""Entry point script for Retrieval-Augmented Generation tasks.

This module provides a command-line interface for running inference and evaluation
on RAG tasks. It supports document retrieval from ChromaDB, generation with
retrieved context, and comprehensive evaluation including both retrieval and
generation quality metrics.
"""

import argparse
from datetime import datetime
import json
import os
import sys

from alue.data_utils import load_data
from alue.prompt_utils import build_messages
from alue.rag_utils import ChromaInterface, get_embedding_function
from alue.inference import run_llm_inference
from alue.evaluation import RAGEval

from .utils import load_schema, parse_fields_to_extract

    
def run_inference(args: argparse.Namespace) -> str:
    """Run RAG inference with document retrieval and generation.
    
    For each question:
    1. Retrieves top-k relevant documents from ChromaDB
    2. Builds context from retrieved documents
    3. Generates answer using LLM with retrieved context
    4. Saves predictions with document IDs for evaluation
    
    Args:
        args: Parsed command-line arguments containing:
            - input_data_json_path: Path to input data file
            - output_dir: Directory to save results
            - inference_model_name: Model identifier for generation
            - task_type: Task type for template selection
            - num_examples: Number of few-shot examples
            - num_questions: Optional limit on questions to process
            - schema_class: Optional Pydantic schema for structured output
            - field_to_extract: Field(s) to extract from structured response
            - temperature: Sampling temperature
            - max_tokens: Maximum tokens to generate
            - database_path: Path to ChromaDB database
            - collection_name: ChromaDB collection name
            - top_k: Number of documents to retrieve
            - embedding_model: Optional embedding model name
            
    Returns:
        Path to the saved predictions JSON file.
        
    Example:
        >>> args = parser.parse_args(['inference', '-i', 'rag_data.json', ...])
        >>> predictions_file = run_inference(args)
        >>> print(f"Predictions saved to {predictions_file}")
    """
    """Run RAG inference."""
    print(f"RAG Inference: {args.inference_model_name}")
    print("=" * 50)
    chroma = ChromaInterface(database_path=args.database_path)
    embedding_function = get_embedding_function(
        model=args.embedding_model
    )

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
        "ground_truth": ground_truth_answers,
        "questions": [item['input'] for item in test_data],
        "temperature": args.temperature,
        "top_k": args.top_k
    }
    
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved to: {args.output_dir}")

    return predictions_file


def run_evaluation(args: argparse.Namespace) -> None:
    """Run comprehensive RAG evaluation on prediction results.
    
    Evaluates both retrieval quality (recall@k, context relevancy) and
    generation quality (composite correctness score) using the RAGEval engine.
    
    Args:
        args: Parsed command-line arguments containing:
            - input_data_json_path: Path to input data file with ground truth
            - predictions_file: Path to predictions JSON file
            - output_dir: Directory to save evaluation results
            - llm_judge_model_name: Model for LLM-as-judge evaluation
            - database_path: Path to ChromaDB (for context relevancy)
            - collection_name: ChromaDB collection name
            - evaluate_retrieval: Whether to evaluate retrieval metrics
            - evaluate_generation: Whether to evaluate generation metrics
            - use_recall_k: Whether to calculate recall@k
            - top_k: k value for recall@k metric
            - verbose: Whether to output detailed explanations
            
    Example:
        >>> args = parser.parse_args(['evaluation', '-i', 'rag_data.json', ...])
        >>> run_evaluation(args)
        Evaluation complete. Retrieval metrics: {...}
    """
    print("Running evaluation...")
    eval_engine = RAGEval(
        data_file=args.input_data_json_path,
        pred_file=args.predictions_file,
        out_dir=args.output_dir,
        model_name=args.llm_judge_model_name,
        database_path=args.database_path,
        collection_name=args.collection_name,
        evaluate_retrieval=args.evaluate_retrieval,
        evaluate_generation=args.evaluate_generation,
        use_recall_k=args.use_recall_k,
        k=args.top_k,
        verbose=args.verbose
    )
    eval_engine.perform_evaluation()


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add RAG inference-related arguments to an argument parser.
    
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
        help="Model name for generation (e.g., gpt-4o-mini)"
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
        help="Pydantic schema class (e.g., RAGResponse)"
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
    parser.add_argument(
        "--database-path",
        type=str,
        default="./chroma_db",
        help="Path to ChromaDB database"
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="documents",
        help="Name of the ChromaDB collection"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name (optional, uses default if not specified)"
    )


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    """Add RAG evaluation-related arguments to an argument parser.
    
    Args:
        parser: ArgumentParser to add arguments to.
    """
    parser.add_argument(
        "--llm_judge_model_name",
        required=True,
        help="Model name for LLM-as-judge evaluation"
    )
    parser.add_argument(
        "--evaluate_retrieval",
        action="store_true",
        default=True,
        help="Evaluate retrieval metrics (recall@k, context relevancy)"
    )
    parser.add_argument(
        "--evaluate_generation",
        action="store_true",
        default=True,
        help="Evaluate generation metrics (composite correctness score)"
    )
    parser.add_argument(
        "--use_recall_k",
        action="store_true",
        help="Calculate recall@k if document IDs available"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed explanations"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands for RAG inference and evaluation.
    
    Creates a parser with three subcommands:
    - inference: Run retrieval and generation only
    - evaluation: Run evaluation only on existing predictions
    - both: Run inference followed by evaluation
    
    Returns:
        Configured ArgumentParser with all subcommands.
        
    Example:
        >>> parser = create_parser()
        >>> args = parser.parse_args(['both', '-i', 'rag_data.json', ...])
    """
    parser = argparse.ArgumentParser(description="RAG script")
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
    eval_parser.add_argument(
        "--database-path",
        type=str,
        help="Path to ChromaDB database (required for context relevancy evaluation)"
    )
    eval_parser.add_argument(
        "--collection-name",
        type=str,
        help="ChromaDB collection name (required for context relevancy evaluation)"
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="k value for recall@k metric"
    )
    add_evaluation_args(eval_parser)

    # Both subparser
    both_parser = subparsers.add_parser(
        "both",
        help="Run inference + evaluation"
    )
    add_inference_args(both_parser)
    add_evaluation_args(both_parser)

    return parser


def main() -> None:
    """Main entry point for the RAG script.
    
    Parses command-line arguments, adds timestamp to output directory,
    and executes the requested mode (inference, evaluation, or both).
    
    The script supports three modes:
    1. inference: Retrieve documents and generate answers
    2. evaluation: Evaluate existing predictions (retrieval + generation quality)
    3. both: Run inference then evaluation in sequence
    
    Example:
        Run inference only:
        $ python rag.py inference -i rag_data.json -o results \\
            -m gpt-4 --database-path ./chroma_db --collection-name docs
        
        Run evaluation only:
        $ python rag.py evaluation -i rag_data.json -o results \\
            --predictions_file results_20240101_120000/predictions.json \\
            --llm_judge_model_name gpt-4 --database-path ./chroma_db
        
        Run both:
        $ python rag.py both -i rag_data.json -o results \\
            -m gpt-4 --llm_judge_model_name gpt-4 \\
            --database-path ./chroma_db --collection-name docs \\
            --top-k 5 --evaluate_retrieval --evaluate_generation
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