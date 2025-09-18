import json
import os
from pathlib import Path
from typing import Any

from config import MODELS, USE_AIP, USE_TGI
from custom_components import (
    BinaryClassificationPipeline,
    ClassificationPipeline,
    ExtractiveQAPipeline,
    GlossaryTermsPipeline,
    MCQAPipeline,
    RAGPipeline,
    SummarizationPipeline,
)
from io_utilities import (
    get_binary_classification_dataset,
    get_classification_inputs_list,
    get_mcqa_dataset,
    get_qa_inputs_list,
    get_rag_inputs_list,
    get_summarization_dataset,
)
from rag_utils import load_or_create_db
from tqdm import tqdm


def load_prompt_from_file(filepath: str) -> str:
    prompt = Path(filepath).read_text()
    return prompt


class ExtractiveQAInference:
    """
    A class used to run extractive question answering inference pipeline.

    Attributes
    ----------
    model_type : str
        The type of model to use for inference.
    use_tgi : bool
        Whether to use TGI.
    generation_kwargs : Optional[Dict[str, Any]]
        Additional keyword arguments for the generation process.
    pipeline : ExtractiveQAPipeline
        The pipeline for extractive question answering.

    Methods
    -------
    run_inference(query: str, context: str) -> Dict[str, Any]:
        Runs the inference and returns the answer to the query.
    """

    def __init__(
        self,
        model_type: str,
        prompt_template_path: str,
        use_tgi: bool = USE_TGI,
        vllm_offline: bool = False,
        num_gpus: int = 1,
        tgi_endpoint: str = None,
        use_aip: bool = USE_AIP,
        quantized: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        log_prompt: bool = True,
        model_lookup_table: dict[str, str] = MODELS,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the ExtractiveQAInference component.

        Parameters
        ----------
        model_type : str
            The type of model to use for inference.
        prompt_template_path : str
            Path to prompt template file
        use_tgi : bool, optional
            Whether to use TGI, by default USE_TGI
        tgi_endpoint : str, optional
            A specific TGI endpoint that is different from the default
        use_aip : bool, optional
            Whether to use AIP endpoints, by default USE_AIP
        quantized : bool, default False
            Whether or not a quantized model is used
        quantized: bool, optional
            Whether or not a quantized model is used
        generation_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments for the generation process, by default None
        examples : list[dict[str, str]] | None, default None
            List of question and answer pairs.
        log_prompt : bool, default False
            Whether to log the prompt after it is built
        """
        self.model_type = model_type
        self.use_tgi = use_tgi
        self.tgi_endpoint = tgi_endpoint
        self.vllm_offline = vllm_offline
        self.num_gpus = num_gpus
        self.use_aip = use_aip
        self.generation_kwargs = generation_kwargs
        self.quantized = quantized
        self.prompt_template_path = prompt_template_path
        self.examples = examples
        self.log_prompt = log_prompt
        self.batch_size = batch_size

        self.pipeline = ExtractiveQAPipeline(
            model_type=model_type,
            prompt_template_path=prompt_template_path,
            use_tgi=use_tgi,
            vllm_offline=vllm_offline,
            num_gpus=num_gpus,
            tgi_endpoint=tgi_endpoint,
            use_aip=use_aip,
            generation_kwargs=generation_kwargs,
            quantized=quantized,
            examples=examples,
            log_prompt=log_prompt,
            model_lookup_table=model_lookup_table,
            batch_size=self.batch_size,
        )

    def _get_inference_per_batch(
        self, user_inputs: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Generates predictions from the selected model for a batch of inputs.

        Parameters
        ----------
        user_inputs : List[Dict[str, str]]
            A list of dictionaries containing question id, question, and context.
        inference_pipeline : ExtractiveQAPipeline
            The QA pipeline for inference.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries containing question id, question, context, answer_text.
        """
        questions = [user_input["question"] for user_input in user_inputs]
        contexts = [user_input["context"] for user_input in user_inputs]

        answers = self.pipeline.run(
            queries=questions,
            contexts=contexts,
        ).get("answers")
        print(f"length of answers: {len(answers)}")

        outputs = []
        for user_input, answer in zip(user_inputs, answers, strict=False):
            output = {
                "id": user_input["id"],
                "question": user_input["question"],
                "context": user_input["context"],
                "answer_text": answer,
            }
            outputs.append(output)
            print(f"Answer: {answer} \n")
            print("=" * 100)

        print(f"Outputs: {outputs}")
        print(f"length of outputs: {len(outputs)}")

        return outputs

    def full_inference(
        self,
        input_data_json_path: str,
        output_eval_res_dir: str,
        batch_size: int = 1,
    ) -> str:
        """
        Collects predictions and performs evaluation.

        Parameters
        ----------
        input_data_json_path : str
            The path to the squad-like JSON dataset.
        output_eval_res_dir : str
            The path to store evaluation results.
        inference_pipeline : ExtractiveQAInference
            The QA pipeline for inference.
        batch_size : int, optional
            The number of inputs to process in each batch, by default 8.

        Returns
        -------
        Path to the JSON file with predictions.
        """
        inputs_list = get_qa_inputs_list(input_json_data_path=input_data_json_path)
        final_predictions = {}

        if self.vllm_offline:
            batch_predictions = self._get_inference_per_batch(user_inputs=inputs_list)

            for pred in batch_predictions:
                final_predictions[pred["id"]] = pred["answer_text"]

        else:
            for i in tqdm(range(0, len(inputs_list), batch_size)):
                batch_inputs = inputs_list[i : i + batch_size]
                batch_predictions = self._get_inference_per_batch(
                    user_inputs=batch_inputs,
                )
                for pred in batch_predictions:
                    final_predictions[pred["id"]] = pred["answer_text"]

        if not os.path.exists(output_eval_res_dir):
            os.makedirs(output_eval_res_dir)

        predictions_filename = os.path.join(output_eval_res_dir, "predictions.json")
        with open(predictions_filename, "w") as file:
            json.dump(final_predictions, file)

        return predictions_filename


class ClassificationInference:
    """
    A class used to run classification inference pipeline.

    Attributes
    ----------
    model_type : str
        The type of model to use for inference.
    use_tgi : bool
        Whether to use TGI.
    generation_kwargs : Optional[Dict[str, Any]]
        Additional keyword arguments for the generation process.
    pipeline : ExtractiveQAPipeline
        The pipeline for extractive question answering.

    Methods
    -------
    run_inference(query: str, context: str) -> Dict[str, Any]:
        Runs the inference and returns the answer to the query.
    """

    def __init__(
        self,
        model_type: str,
        prompt_template_path: str,
        schema_path: str | None = None,
        use_tgi: bool = USE_TGI,
        vllm_offline: bool = False,
        num_gpus: int = 1,
        tgi_endpoint: str | None = None,
        use_aip: bool = USE_AIP,
        quantized: bool = False,
        classification_kwargs: dict[str, Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        log_prompt: bool = True,
        model_lookup_table: dict[str, str] = MODELS,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the ExtractiveQAInference component.

        Parameters
        ----------
        model_type : str
            The type of model to use for inference.
        prompt_template_path : str
            Path to prompt template file
        use_tgi : bool, optional
            Whether to use TGI, by default USE_TGI
        tgi_endpoint : str, optional
            A specific TGI endpoint that is different from the default
        use_aip : bool, optional
            Whether to use AIP endpoints, by default USE_AIP
        quantized: bool, optional
            Whether or not a quantized model is used
        classification_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments for the classification process, by default None
        examples : list[dict[str, str]] | None, default None
            List of question and answer pairs.
        log_prompt : bool, default False
            Whether to log the prompt after it is built
        """
        self.model_type = model_type
        self.use_tgi = use_tgi
        self.tgi_endpoint = tgi_endpoint
        self.vllm_offline = vllm_offline
        self.num_gpus = num_gpus
        self.use_aip = use_aip
        self.classification_kwargs = classification_kwargs
        self.quantized = quantized
        self.prompt_template_path = prompt_template_path
        # self.prompt_template = load_prompt_from_file(prompt_template_path)
        self.examples = examples
        self.log_prompt = log_prompt
        self.batch_size = batch_size

        self.pipeline = ClassificationPipeline(
            model_type=model_type,
            prompt_template_path=self.prompt_template_path,
            schema_path=schema_path,
            use_tgi=use_tgi,
            vllm_offline=vllm_offline,
            num_gpus=num_gpus,
            tgi_endpoint=tgi_endpoint,
            use_aip=use_aip,
            classification_kwargs=classification_kwargs,
            quantized=quantized,
            examples=examples,
            log_prompt=log_prompt,
            model_lookup_table=model_lookup_table,
            batch_size=batch_size,
        )

    def _get_inference_per_batch(
        self, user_input_batch: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Runs the inference and returns the answer to the query.

        Parameters
        ----------
        user_input_batch : List[Dict[str,str]]
            A list of dictionaries with the following keys:
                id (str) - Unique identifier for inference sample
                classification_input (str) - The text that the model should process to infer the classification labels,
        """
        classification_outputs = self.pipeline.run(user_input_batch)
        return classification_outputs

    def full_inference(
        self, input_data_json_path: str, output_eval_res_dir: str, batch_size: int = 1
    ) -> str:
        """
        Runs the inference and returns the answer to the query.

        Parameters
        ----------
        input_data_json_path : str
            Path to the input data json file.
        output_eval_res_dir : str
            Path to the output data json file.
        batch_size : int
            Batch size for inference.
        """
        inputs_list = get_classification_inputs_list(input_data_json_path)

        final_predictions = {}
        if self.vllm_offline:
            batch_predictions = self._get_inference_per_batch(
                user_input_batch=inputs_list
            )["answers"]

            for prediction in batch_predictions:
                final_predictions[prediction["id"]] = prediction[
                    "classification_output"
                ]

        else:
            for i in tqdm(range(0, len(inputs_list), batch_size)):
                batch = inputs_list[i : i + batch_size]
                batch_predictions = self._get_inference_per_batch(
                    user_input_batch=batch
                )["answers"]
                # print(f"type: {type(batch_predictions)}")
                # print(f"Batched predictions: {batch_predictions}")

                for prediction in batch_predictions:
                    final_predictions[prediction["id"]] = prediction[
                        "classification_output"
                    ]

        if not os.path.exists(output_eval_res_dir):
            os.makedirs(output_eval_res_dir)

        predictions_filename = os.path.join(output_eval_res_dir, "predictions.json")
        with open(predictions_filename, "w") as f:
            json.dump(final_predictions, f)

        return predictions_filename


class RAGInference:
    """
    A class used to run RAG inference pipeline.

    Attributes
    ----------
    model_type : str
        The type of model to use for inference.
    prompt_template_path : str
            Path to prompt template file
    use_tgi : bool, optional
        Whether to use TGI, by default USE_TGI
    use_aip : bool, optional
        Whether to use AIP, by default USE_AIP
    quantized: bool, optional
        Whether or not a quantized model is used
    classification_kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments for the classification process, by default None
    examples : list[dict[str, str]] | None, default None
        List of question and answer pairs.
    log_prompt : bool, default False
        Whether to log the prompt after it is built

    Methods
    -------
    run_inference(query: str, context: str) -> Dict[str, Any]:
        Runs the inference and returns the answer to the query.
    """

    def __init__(
        self,
        model_type: str = "",
        prompt_template_path: str = "",
        collection_name: str = "",
        persist_path: str = "",
        use_tgi: bool = USE_TGI,
        use_aip: bool = USE_AIP,
        quantized: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        log_prompt: bool = True,
        model_lookup_table: dict[str, str] = MODELS,
        use_local_embedding_model: bool = False,
        embedding_model_name: str = "BAAI/bge-m3",
        batch_size: int = 1,
    ) -> None:
        self.model_type = model_type
        self.use_tgi = use_tgi
        self.use_aip = use_aip
        self.collection_name = collection_name
        self.persist_path = persist_path
        self.generation_kwargs = generation_kwargs
        self.quantized = quantized
        self.prompt_template_path = prompt_template_path

        self.examples = examples
        self.log_prompt = log_prompt
        self.model_lookup_table = model_lookup_table
        self.use_local_embedding_model = use_local_embedding_model
        self.embedding_model_name = embedding_model_name
        self.document_store = (
            load_or_create_db(
                collection_name=collection_name, persist_path=persist_path
            )
            if persist_path
            else None
        )

        self.batch_size = batch_size

        print(collection_name)

        if model_type:
            self.rag_pipeline = RAGPipeline(
                model_type=self.model_type,
                use_tgi=self.use_tgi,
                use_aip=self.use_aip,
                generation_kwargs=self.generation_kwargs,
                quantized=self.quantized,
                prompt_template_path=self.prompt_template_path,
                examples=self.examples,
                log_prompt=self.log_prompt,
                model_lookup_table=self.model_lookup_table,
                use_rag=True,
                document_store=self.document_store,
                embedding_model_name=self.embedding_model_name,
                use_local_embedding_model=self.use_local_embedding_model,
                batch_size=self.batch_size,
            )

    def _get_inference_per_batch(
        self,
        user_inputs: list[dict[str, str]],
        top_k: int = 5,
        doc_id_field: str = "ACN",
    ) -> list[dict[str, str]]:
        """
        Performs inference on a batch of user inputs.

        Parameters
        ----------
        user_inputs : List[Dict[str, str]]
            A list of dictionaries containing the user inputs.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries containing the inference results.
        """
        print(f"queries: {user_inputs}")

        answers = self.rag_pipeline.run(
            queries=user_inputs, top_k=top_k, doc_id_field=doc_id_field
        ).get("answers")

        outputs = []
        for user_input, answer in zip(user_inputs, answers, strict=False):
            output = {
                "id": user_input["id"],
                "question": user_input["question"],
                "answer_text": answer["answer"],
                "predicted_doc_ids": answer["doc_ids"],
                "ground_truth_answer": user_input["answer"],
            }
            outputs.append(output)

        return outputs

    def full_inference(
        self,
        input_data_json_path: str,
        output_eval_res_dir: str,
        batch_size: int = 1,
        top_k: int = 5,
        doc_id_field: str = "ACN",
    ):
        """
        Performs inference on a batch of user inputs.

        Parameters
        ----------
        input_data_json_path : str
            Path to the input data JSON file.
        output_eval_res_dir : str
            Path to the output directory for the evaluation results.
        batch_size : int, default = 1
            Batch size for inference.
        top_k : int, default = 5
            Number of top documents to retrieve.
        doc_id_field : str, default = "ACN"
            Document ID field to use for retrieval.
        """
        inputs_list = get_rag_inputs_list(input_json_data_path=input_data_json_path)
        final_predictions = {}

        for i in tqdm(range(0, len(inputs_list), batch_size)):
            batch_inputs = inputs_list[i : i + batch_size]
            batch_predictions = self._get_inference_per_batch(
                user_inputs=batch_inputs, top_k=top_k, doc_id_field=doc_id_field
            )

            for pred in batch_predictions:
                final_predictions[pred["id"]] = {
                    "answer": pred["answer_text"],
                    "predicted_doc_ids": pred["predicted_doc_ids"],
                    "question": pred["question"],
                    "question_id": pred["id"],
                    "ground_truth_answer": pred["ground_truth_answer"],
                }

        if not os.path.exists(output_eval_res_dir):
            os.makedirs(output_eval_res_dir)

        predictions_filename = os.path.join(output_eval_res_dir, "predictions.json")
        with open(predictions_filename, "w") as file:
            json.dump(final_predictions, file)

        return predictions_filename


class GlossaryTermsInference:
    """
    A class used to run glossary terms inference pipeline.

    Attributes
    ----------
    model_type : str
        The type of model to use for inference.
    use_tgi : bool
        Whether to use TGI.
    generation_kwargs : Optional[Dict[str, Any]]
        Additional keyword arguments for the generation process.
    pipeline : GlossaryTermsPipeline
        The pipeline for extractive question answering.

    Methods
    -------
    run_inference(query: str, context: str) -> Dict[str, Any]:
        Runs the inference and returns the answer to the query.
    """

    def __init__(
        self,
        model_type: str,
        prompt_template_path: str,
        schema_path: str,
        use_tgi: bool = USE_TGI,
        tgi_endpoint: str = None,
        use_aip: bool = USE_AIP,
        quantized: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        log_prompt: bool = True,
        model_lookup_table: dict[str, str] = MODELS,
    ) -> None:
        """
        Initialize the GlossaryTerms component.

        Parameters
        ----------
        model_type : str
            The type of model to use for inference.
        prompt_template_path : str
            Path to prompt template file
        schema_path : str
            Path to schema file
        use_tgi : bool, optional
            Whether to use TGI, by default USE_TGI
        tgi_endpoint : str, optional
            A specific TGI endpoint that is different from the default
        use_aip : bool, optional
            Whether to use AIP endpoints, by default USE_AIP
        quantized : bool, default False
            Whether or not a quantized model is used
        quantized: bool, optional
            Whether or not a quantized model is used
        generation_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments for the generation process, by default None
        examples : list[dict[str, str]] | None, default None
            List of question and answer pairs.
        log_prompt : bool, default False
            Whether to log the prompt after it is built
        """
        self.model_type = model_type
        self.use_tgi = use_tgi
        self.tgi_endpoint = tgi_endpoint
        self.use_aip = use_aip
        self.generation_kwargs = generation_kwargs
        self.quantized = quantized
        self.prompt_template_path = prompt_template_path
        self.prompt_template = load_prompt_from_file(prompt_template_path)
        self.schema_path = schema_path
        self.examples = examples
        self.log_prompt = log_prompt

        self.pipeline = GlossaryTermsPipeline(
            model_type=model_type,
            prompt_template=self.prompt_template,
            schema_path=self.schema_path,
            use_tgi=use_tgi,
            tgi_endpoint=tgi_endpoint,
            use_aip=use_aip,
            generation_kwargs=generation_kwargs,
            quantized=quantized,
            examples=examples,
            log_prompt=log_prompt,
            model_lookup_table=model_lookup_table,
        )

    def run_inference(
        self,
        glossary_term: str,
    ) -> dict[str, Any]:
        """
        Runs the inference and returns the answer to the query.

        Parameters
        ----------
        glossary_term : str
            The term for which to generate definition for.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the definition to the glossary_term.
        """
        answer = self.pipeline.run(glossary_term=glossary_term)

        return answer


class MCQAInference:
    """A class for running inference on a multiple choice question answering model."""

    def __init__(
        self,
        model_type: str,
        prompt_template_path: str,
        schema_path: str,
        use_tgi: bool = USE_TGI,
        tgi_endpoint: str = None,
        use_aip: bool = USE_AIP,
        quantized: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        log_prompt: bool = True,
        model_lookup_table: dict[str, str] = MODELS,
    ):
        self.model_type = model_type
        self.prompt_template_path = prompt_template_path
        self.schema_path = schema_path
        self.use_tgi = use_tgi
        self.tgi_endpoint = tgi_endpoint
        self.use_aip = use_aip
        self.quantized = quantized
        self.generation_kwargs = generation_kwargs or {}
        self.examples = examples
        self.log_prompt = log_prompt
        self.model_lookup_table = model_lookup_table

        self.pipeline = MCQAPipeline(
            model_type=model_type,
            prompt_template_path=prompt_template_path,
            schema_path=schema_path,
            use_tgi=use_tgi,
            tgi_endpoint=tgi_endpoint,
            use_aip=use_aip,
            quantized=quantized,
            generation_kwargs=generation_kwargs,
            examples=examples,
            log_prompt=log_prompt,
            model_lookup_table=model_lookup_table,
        )

    def _get_inference_per_batch(
        self, user_input_batch: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Generates predictions from the selected model for a batch of inputs.

        Parameters
        ----------
        user_input_batch : List[Dict[str, str]]
            A list of dictionaries containing question id, question, and context.
        inference_pipeline : MCQAPipeline
            The QA pipeline for inference.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries containing question id, question, context, answer_text.
        """
        questions = [user_input["question"] for user_input in user_input_batch]

        answers = self.pipeline.run(
            queries=questions,
        ).get("answers")
        print(f"length of answers: {len(answers)}")

        outputs = []
        for user_input, answer in zip(user_input_batch, answers, strict=False):
            output = {
                "id": user_input["id"],
                "question": user_input["question"],
                "answer": answer.get("selected_option"),
            }
            outputs.append(output)
            print(f"Answer: {answer} \n")
            print("=" * 100)

        print(f"Outputs: {outputs}")
        print(f"length of outputs: {len(outputs)}")

        return outputs

    def full_inference(
        self,
        input_data_json_path: str,
        output_eval_res_dir: str,
        batch_size: int = 1,
    ) -> str:
        """
        Collects predictions and performs evaluation.

        Parameters
        ----------
        input_data_json_path : str
            The path to the squad-like JSON dataset.
        output_eval_res_dir : str
            The path to store evaluation results.
        inference_pipeline : ExtractiveQAInference
            The QA pipeline for inference.
        batch_size : int, optional
            The number of inputs to process in each batch, by default 8.

        Returns
        -------
        Path to the JSON file with predictions.
        """
        inputs_list = get_mcqa_dataset(input_json_data_path=input_data_json_path)
        final_predictions = {}

        for i in tqdm(range(0, len(inputs_list), batch_size)):
            batch_inputs = inputs_list[i : i + batch_size]
            batch_predictions = self._get_inference_per_batch(
                user_input_batch=batch_inputs,
            )
            for pred in batch_predictions:
                final_predictions[pred["id"]] = pred["answer"]

        if not os.path.exists(output_eval_res_dir):
            os.makedirs(output_eval_res_dir)

        predictions_filename = os.path.join(output_eval_res_dir, "predictions.json")
        with open(predictions_filename, "w") as file:
            json.dump(final_predictions, file)

        return predictions_filename


class SummarizationInference:
    """
    A class used to run summarization inference pipeline.

    Attributes
    ----------
    model_type : str
        The type of model to use for inference.
    use_tgi : bool
        Whether to use TGI.
    generation_kwargs : Optional[Dict[str, Any]]
        Additional keyword arguments for the generation process.
    pipeline : ExtractiveQAPipeline
        The pipeline for extractive question answering.

    Methods
    -------
    run_inference(query: str, context: str) -> Dict[str, Any]:
        Runs the inference and returns the answer to the query.
    """

    def __init__(
        self,
        model_type: str,
        prompt_template_path: str,
        use_tgi: bool = USE_TGI,
        tgi_endpoint: str = None,
        use_aip: bool = USE_AIP,
        quantized: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        log_prompt: bool = True,
        model_lookup_table: dict[str, str] = MODELS,
        batch_size: int = 1,
    ) -> None:
        """
        Initialize the ExtractiveQAInference component.

        Parameters
        ----------
        model_type : str
            The type of model to use for inference.
        prompt_template_path : str
            Path to prompt template file
        use_tgi : bool, optional
            Whether to use TGI, by default USE_TGI
        tgi_endpoint : str, optional
            A specific TGI endpoint that is different from the default
        use_aip : bool, optional
            Whether to use AIP endpoints, by default USE_AIP
        quantized : bool, default False
            Whether or not a quantized model is used
        quantized: bool, optional
            Whether or not a quantized model is used
        generation_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments for the generation process, by default None
        examples : list[dict[str, str]] | None, default None
            List of question and answer pairs.
        log_prompt : bool, default False
            Whether to log the prompt after it is built
        """
        self.model_type = model_type
        self.use_tgi = use_tgi
        self.tgi_endpoint = tgi_endpoint
        self.use_aip = use_aip
        self.generation_kwargs = generation_kwargs
        self.quantized = quantized
        self.prompt_template_path = prompt_template_path
        self.examples = examples
        self.log_prompt = log_prompt
        self.batch_size = batch_size

        self.pipeline = SummarizationPipeline(
            model_type=model_type,
            prompt_template_path=prompt_template_path,
            use_tgi=use_tgi,
            tgi_endpoint=tgi_endpoint,
            use_aip=use_aip,
            generation_kwargs=generation_kwargs,
            quantized=quantized,
            examples=examples,
            log_prompt=log_prompt,
            model_lookup_table=model_lookup_table,
            batch_size=self.batch_size,
        )

    def _get_inference_per_batch(
        self, user_inputs: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Generates predictions from the selected model for a batch of inputs.

        Parameters
        ----------
        user_inputs : List[Dict[str, str]]
            A list of dictionaries containing input id and input_text.
        inference_pipeline : SummarizationPipeline
            The QA pipeline for inference.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries containing question id, question, context, answer_text.
        """
        inputs = [user_input["input"] for user_input in user_inputs]

        answers = self.pipeline.run(inputs=inputs).get("answers")

        print(f"length of answers: {len(answers)}")

        outputs = []
        for user_input, answer in zip(user_inputs, answers, strict=False):
            output = {
                "id": user_input["id"],
                "input_text": user_input["input"],
                "ground_truth_summary": user_input["output"],
                "summary": answer,
            }
            outputs.append(output)
            print(f"Answer: {answer} \n")
            print("=" * 100)

        print(f"Outputs: {outputs}")
        print(f"length of outputs: {len(outputs)}")

        return outputs

    def full_inference(
        self,
        input_data_json_path: str,
        output_eval_res_dir: str,
        batch_size: int = 1,
    ) -> str:
        """
        Collects predictions and performs evaluation.

        Parameters
        ----------
        input_data_json_path : str
            The path to the summarization dataset.
        output_eval_res_dir : str
            The path to store infernece results.
        inference_pipeline : SummarizationInference
            The QA pipeline for inference.
        batch_size : int, optional
            The number of inputs to process in each batch, by default 8.

        Returns
        -------
        Path to the JSON file with predictions.
        """
        inputs_list = get_summarization_dataset(
            input_json_data_path=input_data_json_path, load_examples=False
        )
        final_predictions = {}

        for i in tqdm(range(0, len(inputs_list), batch_size)):
            batch_inputs = inputs_list[i : i + batch_size]
            batch_predictions = self._get_inference_per_batch(
                user_inputs=batch_inputs,
            )
            for pred in batch_predictions:
                final_predictions[pred["id"]] = {
                    "predicted_summary": pred["summary"],
                    "ground_truth_summary": pred["ground_truth_summary"],
                    "narrative": pred["input_text"],
                }

        if not os.path.exists(output_eval_res_dir):
            os.makedirs(output_eval_res_dir)

        predictions_filename = os.path.join(output_eval_res_dir, "predictions.json")
        with open(predictions_filename, "w") as file:
            json.dump(final_predictions, file)

        return predictions_filename


class BinaryClassificationInference:
    """A class for running inference on a multiple choice question answering model."""

    def __init__(
        self,
        model_type: str,
        prompt_template_path: str,
        schema_path: str,
        choices_path: str,
        vllm_offline: bool = False,
        num_gpus: int = 1,
        use_tgi: bool = USE_TGI,
        tgi_endpoint: str = None,
        use_aip: bool = USE_AIP,
        quantized: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        log_prompt: bool = True,
        model_lookup_table: dict[str, str] = MODELS,
    ):
        self.model_type = model_type
        self.prompt_template_path = prompt_template_path
        self.schema_path = schema_path
        self.choices_path = choices_path
        self.use_tgi = use_tgi
        self.vllm_offline = vllm_offline
        self.num_gpus = num_gpus
        self.tgi_endpoint = tgi_endpoint
        self.use_aip = use_aip
        self.quantized = quantized
        self.generation_kwargs = generation_kwargs or {}
        self.examples = examples
        self.log_prompt = log_prompt
        self.model_lookup_table = model_lookup_table

        self.pipeline = BinaryClassificationPipeline(
            model_type=model_type,
            prompt_template_path=prompt_template_path,
            schema_path=schema_path,
            choices_path=choices_path,
            use_tgi=use_tgi,
            vllm_offline=vllm_offline,
            num_gpus=num_gpus,
            tgi_endpoint=tgi_endpoint,
            use_aip=use_aip,
            quantized=quantized,
            generation_kwargs=generation_kwargs,
            examples=examples,
            log_prompt=log_prompt,
            model_lookup_table=model_lookup_table,
        )

    def _get_inference_per_batch(
        self, user_input_batch: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        Generates predictions from the selected model for a batch of inputs.

        Parameters
        ----------
        user_input_batch : List[Dict[str, str]]
            A list of dictionaries containing question id, question, and context.
        inference_pipeline : MCQAPipeline
            The QA pipeline for inference.

        Returns
        -------
        List[Dict[str, str]]
            A list of dictionaries containing question id, question, context, answer_text.
        """
        texts = [user_input["text"] for user_input in user_input_batch]

        answers = self.pipeline.run(
            texts=texts,
        ).get("answers")
        print(f"length of answers: {len(answers)}")

        outputs = []
        for user_input, answer in zip(user_input_batch, answers, strict=False):
            answer = answer["label"] if isinstance(answer, dict) else answer
            output = {
                "id": user_input["id"],
                "text": user_input["text"],
                "gt_label": user_input["label"],
                "pred_label": answer,
            }
            outputs.append(output)
            print(f"Answer: {answer} \n")
            print("=" * 100)

        print(f"Outputs: {outputs}")
        print(f"length of outputs: {len(outputs)}")

        return outputs

    def full_inference(
        self,
        input_data_json_path: str,
        output_eval_res_dir: str,
        batch_size: int = 1,
    ) -> str:
        """
        Collects predictions and performs evaluation.

        Parameters
        ----------
        input_data_json_path : str
            The path to the squad-like JSON dataset.
        output_eval_res_dir : str
            The path to store evaluation results.
        inference_pipeline : ExtractiveQAInference
            The QA pipeline for inference.
        batch_size : int, optional
            The number of inputs to process in each batch, by default 8.

        Returns
        -------
        Path to the JSON file with predictions.
        """
        inputs_list = get_binary_classification_dataset(
            input_json_data_path=input_data_json_path
        )
        final_predictions = {}

        if self.vllm_offline:
            batch_predictions = self._get_inference_per_batch(
                user_input_batch=inputs_list
            )
            for pred in batch_predictions:
                final_predictions[pred["id"]] = {
                    "text": pred["text"],
                    "gt_label": pred["gt_label"],
                    "pred_label": pred["pred_label"],
                }
        else:
            for i in tqdm(range(0, len(inputs_list), batch_size)):
                batch_inputs = inputs_list[i : i + batch_size]
                batch_predictions = self._get_inference_per_batch(
                    user_input_batch=batch_inputs,
                )
                for pred in batch_predictions:
                    final_predictions[pred["id"]] = {
                        "text": pred["text"],
                        "gt_label": pred["gt_label"],
                        "pred_label": pred["pred_label"],
                    }

        if not os.path.exists(output_eval_res_dir):
            os.makedirs(output_eval_res_dir)

        predictions_filename = os.path.join(output_eval_res_dir, "predictions.json")
        with open(predictions_filename, "w") as file:
            json.dump(final_predictions, file)

        return predictions_filename
