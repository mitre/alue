
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

    
def run_inference(args):
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


def run_evaluation(args):
    """Run RAG evaluation."""
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
        p.add_argument("--database-path", type=str, default="./chroma_db",
                       help="Path to store ChromaDB database")
        p.add_argument("--collection-name", type=str, default="documents",
                       help="Name for the ChromaDB collection")
        p.add_argument("--top-k", type=int, default=5,
                       help="Number of documents to retrieve")
        p.add_argument("--embedding-model", type=str, default=None,
                        help="Specific model name (optional)")

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
    eval_parser.add_argument("--database-path", type=str,
                        help="Path to ChromaDB database (for context relevancy)")
    eval_parser.add_argument("--collection-name", type=str,
                        help="Name for the ChromaDB collection (for context relevancy)")
    eval_parser.add_argument("--evaluate_retrieval", action="store_true", default=True,
                        help="Evaluate retrieval metrics")
    eval_parser.add_argument("--evaluate_generation", action="store_true", default=True,
                        help="Evaluate generation metrics")
    eval_parser.add_argument("--use_recall_k", action="store_true",
                        help="Calculate recall@k if document IDs available")
    eval_parser.add_argument("--top-k", type=int, default=5,
                        help="Number of documents for recall@k")
    eval_parser.add_argument("--verbose", action="store_true",
                        help="Verbose output with explanations")

    # Both subparser
    both_parser = subparsers.add_parser("both", help="Run inference + evaluation")
    add_inference_args(both_parser)
    # Add evaluation-specific arguments
    both_parser.add_argument("--llm_judge_model_name", required=True,
                            help="Model name for LLM judges")
    both_parser.add_argument("--evaluate_retrieval", action="store_true", default=True,
                        help="Evaluate retrieval metrics")
    both_parser.add_argument("--evaluate_generation", action="store_true", default=True,
                        help="Evaluate generation metrics")
    both_parser.add_argument("--use_recall_k", action="store_true",
                        help="Calculate recall@k if document IDs available")
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


