import argparse
import os
from datetime import datetime

from evaluation import RAGEval
from inference import RAGInference
from io_utilities import get_rag_examples

NOW = datetime.now().strftime("%Y%m%d_%H%M%S")

# def get_inference(
#     user_input: Dict[str, str],
#     inference_pipeline: RAGInference,
#     top_k: int = 5,
#     doc_id_field: str = "ACN"
# ) -> Dict[str, str]:
#     """
#     Generates predictions from the selected model.
#     Parameters
#     ----------
#     user_input : Dict[str, str]
#         A dictionary containing question id, question, and context.
#     inference_pipeline : RAGInference
#         The QA pipeline for inference.
#     top_k : int, optional
#         Number of documents to retrieve (only for RAGInference).
#     Returns
#     -------
#     Dict[str, str]
#         A dictionary containing question id, question, context, answer_text.
#     """
#     question = user_input["question"]

#     output = {}
#     output["id"] = user_input["id"]
#     output["question"] = question

#     answer = inference_pipeline.run_inference(
#         query=question, top_k=top_k, doc_id_field=doc_id_field
#     )
#     output["answer_text"] = answer["answer"]
#     output["predicted_doc_ids"] = answer["doc_ids"]

#     print(f"Output: {output} \n")
#     print("=" * 100)
#     return output

# def perform_inference(
#     input_data_json_path: str,
#     output_eval_res_dir: str,
#     inference_pipeline: RAGInference,
#     top_k: int = 5,
#     doc_id_field: str = "ACN"
# ) -> str:
#     """
#     Collects predictions and performs evaluation.

#     Parameters
#     ----------
#     input_data_json_path : str
#         The path to the squad-like JSON dataset.
#     inference_pipeline : RAGInference
#         The QA pipeline for inference.
#     output_eval_res_dir : str
#         The path to store evaluation results.
#     top_k : int, optional
#         Number of documents to retrieve (only for RAGInference).

#     Returns
#     -------
#     Path to the JSON file with predictions.
#     """
#     inputs_list = get_rag_inputs_list(input_json_data_path=input_data_json_path)
#     final_predictions = {}

#     for i in tqdm(range(len(inputs_list))):
#         user_input = inputs_list[i]
#         pred = get_inference(
#             user_input=user_input,
#             inference_pipeline=inference_pipeline,
#             top_k=top_k,
#             doc_id_field=doc_id_field
#         )
#         print(f"pred: {pred}")

#         final_predictions[pred["id"]] = {"answer": pred["answer_text"],
#                                          "predicted_doc_ids": pred["predicted_doc_ids"],
#                                          "question": pred["question"]}

#     if not os.path.exists(output_eval_res_dir):
#         os.makedirs(output_eval_res_dir)

#     predictions_filename = os.path.join(output_eval_res_dir, "predictions.json")
#     with open(predictions_filename, "w") as file:
#         json.dump(final_predictions, file)

#     return predictions_filename


def perform_evaluation(
    input_data_json_path: str,
    output_eval_res_dir: str,
    predictions_filename: str,
    top_k: int = 5,
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
    eval = RAGEval(
        data_file=input_data_json_path,
        pred_file=predictions_filename,
        out_dir=output_eval_res_dir,
        k=top_k,
    )
    eval.perform_evaluation()


def run_rag_evaluation(args):
    perform_evaluation(
        input_data_json_path=args.input_data_json_path,
        output_eval_res_dir=args.output_eval_res_dir,
        predictions_filename=args.predictions_filename,
        top_k=args.top_k,
    )


def run_rag_inference(args):
    generation_kwargs = {
        "return_full_text": False,
        "max_new_tokens": 512,
        "temperature": None,
        # "pad_token_id": 2,
        "do_sample": True,
        # "early_stopping": True
    }
    examples = get_rag_examples(
        input_json_data_path=args.input_data_json_path,
        nbr_examples=args.nbr_examples,
        randomize_selection=args.randomize_selection,
        randomize_order=args.randomize_order,
        random_seed=args.random_seed,
    )

    inference_pipeline = RAGInference(
        model_type=args.model_type,
        generation_kwargs=generation_kwargs,
        prompt_template_path=args.prompt_template,
        collection_name=args.collection_name,
        persist_path=args.vector_db_local_path,
        examples=examples,
        use_aip=args.use_aip,
        use_tgi=args.use_tgi,
        quantized=args.quantized,
        use_local_embedding_model=args.use_local_embedding_model,
        embedding_model_name=args.embedding_model_name,
        batch_size=args.batch_size,
    )

    inference_pipeline.full_inference(
        input_data_json_path=args.input_data_json_path,
        output_eval_res_dir=args.output_eval_res_dir,
        top_k=args.top_k,
        doc_id_field=args.doc_id_field,
        batch_size=args.batch_size,
    )
    # perform_inference(
    #     input_data_json_path=args.input_data_json_path,
    #     output_eval_res_dir=args.output_eval_res_dir,
    #     inference_pipeline=inference_pipeline,
    #     top_k=args.top_k,
    #     doc_id_field=args.doc_id_field
    # )


def run_rag_both(args):
    run_rag_inference(args)

    predictions_filename = os.path.join(args.output_eval_res_dir, "predictions.json")
    perform_evaluation(
        input_data_json_path=args.input_data_json_path,
        output_eval_res_dir=args.output_eval_res_dir,
        predictions_filename=predictions_filename,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference and/or evaluation for RAG."
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
        "--top_k", type=int, default=5, help="Number of documents to retrieve"
    )
    inference_parser.add_argument(
        "--doc_id_field", type=str, default="ACN", help="Document ID field"
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
        "--collection_name",
        type=str,
        help="Name of Chroma Collection that holds all the documents for RAG",
    )
    inference_parser.add_argument(
        "--vector_db_local_path",
        type=str,
        help="Path to the persistent vector database",
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
        "--use_local_embedding_model",
        action="store_true",
        help="whether to use local embedding model",
    )
    inference_parser.add_argument(
        "--embedding_model_name",
        type=str,
        help="Name of the embedding model to use",
        default="BAAI/bge-m3",
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
        "--top_k", type=int, default=5, help="Number of documents to retrieve"
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
        "--top_k", type=int, default=5, help="Number of documents to retrieve"
    )
    both_parser.add_argument(
        "--doc_id_field", type=str, default="ACN", help="Document ID field"
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
        "--collection_name",
        type=str,
        help="Name of Chroma Collection that holds all the documents for RAG",
    )
    both_parser.add_argument(
        "--vector_db_local_path",
        type=str,
        help="Path to the persistent vector database",
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
        "--use_local_embedding_model",
        action="store_true",
        help="whether to use local embedding model",
    )
    both_parser.add_argument(
        "--embedding_model_name",
        type=str,
        help="Name of the embedding model to use",
        default="BAAI/bge-m3",
    )
    both_parser.add_argument(
        "--batch_size", type=int, help="specify batch size", default=8
    )
    args = parser.parse_args()

    print(f"Args: {args}")
    args.output_eval_res_dir = f"{args.output_eval_res_dir}_{NOW}"
    print(args.output_eval_res_dir)
    if not os.path.exists(args.output_eval_res_dir):
        os.makedirs(args.output_eval_res_dir)
    # Handle the different modes
    if args.mode == "inference":
        run_rag_inference(args)
    elif args.mode == "evaluation":
        run_rag_evaluation(args)
    elif args.mode == "both":
        run_rag_both(args)


# if __name__ == "__main__":
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Perform inference and/or evaluation for RAG.")
#     parser.add_argument("-m", "--mode", type=str, required=True, choices=["inference", "evaluation", "both"])

#     # arguments needed for both modalities
#     parser.add_argument("--input_data_json_path", type=str, help="Path to the input JSON data file.", required=True)
#     parser.add_argument("-p", "--predictions_filename", type=str, help="Path to predictions JSON file (required for evaluation)", required=True)
#     parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve", required=True)

#     # arguments needed for only inference
#     parser.add_argument("-mt", "--model_type", type=str, help="Model type")
#     parser.add_arguement("-f", "--few-shot", type=int, help="number of examples to use in prompt")
#     parser.add_argument("-t", "--use_tgi", action="store_true", help="whether to use TGI (default: False)")
#     parser.add_argument("-a", "--use_aip", action="store_true", help="whether to use AIP endpoint (default: False)")
#     parser.add_argument("-q", "--quantized", action="store_true", help="whether to use quantized model (bitsandbytes)")
#     parser.add_argument("-pt", "--prompt-template", type=str, help="path to prompt template file")
#     parser.add_argument("--nbr_examples", type=int, help="specify number of examples to include in prompt")
#     parser.add_argument("--doc_id_field", type=str, default="ACN", help="Document ID field")
#     parser.add_argument("--collection_name" , type=str, help="Name of Chroma Collection that holds all the documents for RAG")
#     parser.add_argument("--vector_db_local_path" , type=str, help="Path to the persistant vector database")


#     # arguments needed for only evaluation
#     parser.add_argument("--output_eval_res_dir", type=str, help="Path to the output evaluation results directory.")
