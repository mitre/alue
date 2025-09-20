## TO FIX: this import does not work with the new huggingface package
# from huggingface_hub.inference._text_generation import ValidationError
import json
import os
import random
import time
from ast import literal_eval
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

import chromadb
import outlines
import requests
import torch
from config import (EMBEDDING_MODELS, 
                    MODELS, 
                    ENDPOINT_TYPE, 
                    ENDPOINT_URL)
from haystack import Document, Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import (
    OpenAIDocumentEmbedder,
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.generators import HuggingFaceLocalGenerator, OpenAIGenerator
from haystack.components.readers import ExtractiveReader
from haystack.utils import Secret
from huggingface_hub import InferenceClient
from outlines import Template
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from dotenv import load_dotenv
from settings import settings, secret_from_env

@component
class CustomPromptBuilder:
    def __init__(self, run_local: bool, prompt_template: str, use_grammar: bool):
        self.run_local = run_local
        self.prompt_template = prompt_template
        self.use_grammar = use_grammar
        print(f"prompt template: {self.prompt_template}")

        if self.run_local and self.use_grammar:
            self.prompt_builder = Template.from_string(self.prompt_template)
            # self.prompt_builder = Prompt.from_str(self.prompt_template)
        else:
            self.prompt_builder = PromptBuilder(self.prompt_template)

    @component.output_types(prompts=list[str])
    def run(
        self,
        template_variables_list: list[dict[str, str]],
        documents_list: list[list[Document]] = None,
    ) -> dict[str, list[str]]:
        """
        Generate prompts for a batch of template variables and documents.

        Parameters
        ----------
        template_variables_list : List[Dict[str, str]]
            A list of dictionaries containing template variables for each prompt.
        documents_list : List[List[Document]], optional
            A list of lists of documents for each prompt, by default an empty list.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary containing the generated prompts.
        """
        if documents_list is None:
            documents_list = []
        prompts = []

        for i, template_variables in enumerate(template_variables_list):
            # print(f"template variables here: {template_variables}")
            # sys.exit()
            documents = documents_list[i] if documents_list else []
            if "query" in template_variables and isinstance(
                template_variables["query"], Document
            ):
                question = template_variables["query"].content
                template_variables["query"] = question

                print(f"question here: {question}")

            if self.run_local and self.use_grammar:
                outlines_prompt_rendered = self.prompt_builder(**template_variables)
                prompts.append({"prompt": outlines_prompt_rendered})
            else:
                print("using haystack prompt builder")
                if documents:
                    combined_data = {**template_variables, "documents": documents}
                else:
                    combined_data = template_variables

                prompt_rendered = self.prompt_builder.run(**combined_data)
                print(f"rendered prompt: {prompt_rendered}")
                prompts.append(prompt_rendered)

        return {"prompts": prompts}


@component
class TGIGenerator:
    DEFAULT_ENDPOINT_URL = "http://127.0.0.1:3000/generate"
    DEFAULT_TIMEOUT = 120  # seconds

    def __init__(
        self, tgi_endpoint: str | None = None, timeout_sec: int | None = DEFAULT_TIMEOUT
    ):
        """
        Initialize the TGIGenerator component.

        Parameters
        ----------
        tgi_endpoint : str, optional
            The endpoint for the TGI service, by default "http://127.0.0.1:3000/generate"
        timeout: int, optional
            The timeout for the TGI service, by default 120 seconds.
        """
        self.tgi_endpoint = tgi_endpoint or os.getenv(
            "TGI_ENDPOINT_URL", self.DEFAULT_ENDPOINT_URL
        )
        self.timeout = timeout_sec

    @component.output_types(replies=list[str])
    def run(
        self,
        query: str,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """
        Run the TGI generation process.

        Parameters
        ----------
        query : str
            The query to generate from.
        generation_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments for the generation process, by default {}

        Returns
        -------
        Dict[str, List[str]]
            A dictionary containing the generated replies.
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        headers = {
            "Content-Type": "application/json",
        }

        data = {
            "inputs": query,
            "parameters": generation_kwargs if generation_kwargs is not None else {},
        }

        response = requests.post(
            self.tgi_endpoint, headers=headers, json=data, timeout=self.timeout
        )

        response = response.json()
        print(f"response: {response}")
        return {"replies": [response["generated_text"]]}


@component
class AIPGenerator:
    """
    A custom components to generate using AIP endopoints

    Attributes
    ----------
    client : InferenceClient
        An instance of the Huggingface Hub InferenceClient with a specified AIP model endpoint.

    Methods
    -------
    run(query: str, generation_kwargs: Optional[Dict[str, Any]] = {})
        Generates responses based on the provided query and optional arguments.
    """

    def __init__(self, model_type: str) -> None:
        """
        Constructs all the necessary attributes for the AIPGenerator object.

        Parameters
        ----------
            model_type : str
                model to use with the AIP Endpoints.
        """
        self.model_url = MODELS[model_type]["aip_endpoint"]
        self.client = InferenceClient(
            model=MODELS[model_type]["aip_endpoint"], token=False
        )

    @component.output_types(replies=list[str])
    def run(
        self,
        grammar: dict[str, Any] | None = None,
        query: str = "",
        generation_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, list[str]]:
        """
        Generates responses based on the provided query and optional arguments.

        Parameters
        ----------
            query : str
                The query to be processed by the model.
            grammar: Dict[str, Any], optional
                Use for constrained generation, when using schema
            generation_kwargs : [Dict[str, Any], optional
                Additional keyword arguments to be passed to the text generation method (default is {}).

        Returns
        -------
            Dict[str, List[str]]
                A dictionary containing the generated output text.
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        if grammar is None:
            grammar = {}
        if not generation_kwargs:
            generation_kwargs = {}

        # print(f"PROMPT: ", query)
        # print(f"GRAMMAR: ", grammar)
        if grammar:
            output_text = self.client.text_generation(
                prompt=query, grammar=grammar, **generation_kwargs
            )
        else:
            output_text = self.client.text_generation(prompt=query, **generation_kwargs)

        return {"replies": [output_text]}


@component
class CustomOpenAIGenerator:
    def __init__(self, model_type: str, api_key: str, api_base_url: str | None = None):
        """
        Initialize the CustomOpenAIGenerator component.

        Parameters
        ----------
        model : str
            The OpenAI model to use (e.g., "gpt-4").
        api_key : str
            The API key for accessing OpenAI services.
        api_base_url : str, optional
            The base URL for OpenAI API, by default None (uses OpenAI's default endpoint).
        generation_kwargs : Optional[Dict[str, Any]], optional
            Default generation parameters to use, by default {}.
        """
        print(f"model_type: {model_type}")
        print(f"api_base_url: {api_base_url}")
        self.generator = OpenAIGenerator(
            model=model_type,
            api_key=api_key,
            api_base_url=api_base_url,
            # generation_kwargs=generation_kwargs
        )

    @component.output_types(replies=list[str])
    def run(
        self, prompts: list[str], generation_kwargs: dict[str, Any] | None = None
    ) -> dict[str, list[str]]:
        """
        Run the OpenAI generation process for a batch of prompts.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts to generate responses for.
        generation_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments for the generation process, by default {}.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary containing the generated replies.
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        RETRY_SLEEP_SECONDS = 5  # Time to sleep between retries in seconds
        # print(f"prompt: {prompts[0]['prompt']}")
        # response = self.generator.run(prompt=prompts[0]["prompt"], generation_kwargs=generation_kwargs)

        while True:
            try:
                print(f"OPENAI prompt: {prompts[0]}")
                response = self.generator.run(
                    prompt=prompts[0]["prompt"], generation_kwargs=generation_kwargs
                )
                print(f"response: {response}")
                return response
            except Exception as e:
                if "Request timed out" in str(e) or "upstream connect error" in str(e):
                    print("Timeout error occurred. Retrying...")
                    time.sleep(RETRY_SLEEP_SECONDS)
                else:
                    # If the error is not a timeout, re-raise it
                    raise e

        print(f"OPENAI prompt: {prompts[0]}")
        response = self.generator.run(
            prompt=prompts[0]["prompt"], generation_kwargs=generation_kwargs
        )
        print(f"response: {response}")
        return response

        # for prompt in prompts:
        #     response = self.generator.run(prompt=prompt, generation_kwargs=generation_kwargs)
        #     replies.append(response["replies"][0])  # Append the generated reply
        #     meta_data.append(response["meta"][0])   # Append metadata for debugging or analysis

        # return {"replies": replies, "meta": meta_data}


@component
class LocalGenerator:
    """A component to generate text using a local model."""

    def __init__(
        self,
        model_type: str,
        quantized: bool,
        schema_path: str,
        choices_path: str,
        batch_size: int = 1,
    ):
        self.model_type = model_type
        self.quantized = quantized
        self.batch_size = batch_size

        if choices_path:
            with open(choices_path) as f:
                self.choices = json.load(f)

        else:
            self.choices = None

        print(f"choices: {self.choices}")
        if schema_path:
            with open(schema_path) as f:
                self.grammar = json.load(f)
        else:
            self.grammar = None

        if isinstance(MODELS[self.model_type], dict):
            self.model_path = MODELS[self.model_type]["local_path"]
            if "/outputs/" in self.model_path:
                adapter_config_file = os.path.join(
                    self.model_path, "adapter_config.json"
                )
                with open(adapter_config_file) as f:
                    adapter_config = json.load(f)
                self.base_model_path = adapter_config["base_model_name_or_path"]
        else:
            self.model_path = MODELS[self.model_type]

        if self.grammar:
            # structured generation using outlines wrapper
            if not self.quantized:
                model = outlines.models.transformers(
                    self.model_path, model_kwargs={"device_map": "auto"}
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = outlines.models.transformers(
                    self.model_path,
                    model_kwargs={
                        "device_map": "auto",
                        "quantization_config": bnb_config,
                    },
                )
            schema_json = json.dumps(self.grammar)
            self.generator = outlines.generate.json(model, schema_json)

        elif self.choices:
            if not self.quantized:
                model = outlines.models.transformers(
                    self.model_path, model_kwargs={"device_map": "auto"}
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                model = outlines.models.transformers(
                    self.model_path,
                    model_kwargs={
                        "device_map": "auto",
                        "quantization_config": bnb_config,
                    },
                )
            literal_type_code = (
                f"Literal[{', '.join(repr(choice) for choice in self.choices)}]"
            )
            literal_type = literal_eval(literal_type_code)
            print(literal_type)
            self.generator = outlines.Generator(model, literal_type)

        else:
            # unstructured generation
            llm, gen_tokenizer = self.load_gen_model_and_tokenizer()
            if self.batch_size > 1:
                self.generator = llm
                self.gen_tokenizer = gen_tokenizer
            else:
                self.generator = HuggingFaceLocalGenerator(
                    llm,
                    huggingface_pipeline_kwargs={
                        "device_map": "auto",
                        "task": "text-generation",
                        "model": llm,
                        "tokenizer": gen_tokenizer,
                    },
                )
                self.generator.warm_up()

    def load_gen_model_and_tokenizer(
        self,
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the base model, apply the LoRA adapter if present, and return tokenizer + model.
        """
        base_path = getattr(self, "base_model_path", self.model_path)

        if self.quantized:
            print("Using quantized model...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_path,
                device_map="auto",
                trust_remote_code=True,
            )

        if hasattr(self, "base_model_path"):
            gen_model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            gen_model = base_model

        gen_model.eval()
        gen_tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side="left")
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
        # gen_tokenizer.pad_token = "<pad>"

        return gen_model, gen_tokenizer

    @component.output_types(replies=list[str])
    def run(
        self, query: str | list[str], generation_kwargs: dict[str, Any] | None = None
    ) -> dict[str, list[str]]:
        # print(f"len of query: {len(query)}")
        # print(f"batch size: {self.batch_size}")

        if generation_kwargs is None:
            generation_kwargs = {}
        if not self.grammar and not self.choices:
            # disable inference time compilation specifically for gemma2 and gemma3 models
            generation_kwargs["disable_compile"] = True
            generation_kwargs.pop("return_full_text")

            if self.batch_size > 1:
                query = [q["prompt"] for q in query]
                # print(f"query: {query[0]}")
                # sys.exit()

                # inputs = self.gen_tokenizer(query,
                #                             return_tensors="pt",
                #                             padding=True,
                #                             ).to("cuda")
                # input_ids = inputs.input_ids
                # input_lengths = input_ids.shape[1]
                # outputs = self.generator.generate(**inputs, **generation_kwargs)
                # generated_texts = self.gen_tokenizer.batch_decode(outputs[:, input_lengths:], skip_special_tokens=True)
                # return {"replies": generated_texts}

                inputs = self.gen_tokenizer(
                    query,
                    return_tensors="pt",
                    padding=True,
                    # truncation=True,
                ).to("cuda")

                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                input_lengths = attention_mask.sum(
                    dim=1
                )  # Get per-example prompt lengths

                # Debug: Print original prompts (decoded from token IDs)
                decoded_inputs = [
                    self.gen_tokenizer.decode(input_ids[i], skip_special_tokens=False)
                    for i in range(len(input_ids))
                ]

                print("\n--- Prompt Inspection ---")
                for i, decoded in enumerate(decoded_inputs):
                    print(f"[Prompt {i}] {decoded}")

                # sys.exit()

                outputs = self.generator.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )

                generated_texts = [
                    self.gen_tokenizer.decode(
                        output[input_lengths[i] :], skip_special_tokens=True
                    )
                    for i, output in enumerate(outputs)
                ]

                return {"replies": generated_texts}

            # print(f"QUERY TO RUN: ", query)
            print(f"type of generator: {type(self.generator)}")
            response = self.generator.run(
                prompt=query[0]["prompt"], generation_kwargs=generation_kwargs
            )
            print(f"response: {response}")
            return response

        elif self.grammar or self.choices:
            print(f"QUERY TO RUN: {query}")
            result = self.generator(
                query[0]["prompt"],
                # seed=789001,
            )
            print(f"Answer: {result} \n\n")

            output_text = result
            return {"replies": [output_text]}


@component
class VLLMGenerator:
    """A component to generate text using a local model."""

    def __init__(
        self,
        model_type: str,
        quantized: bool,
        schema_path: str,
        choices_path: str,
        mode: str = "offline",
        num_gpus=1,
    ):
        self.model_type = model_type
        self.model_path = MODELS[self.model_type]
        self.quantized = quantized
        self.mode = mode
        if choices_path:
            with open(choices_path) as f:
                self.choices = json.load(f)

        else:
            self.choices = None

        print(f"choices: {self.choices}")
        # sys.exit()

        if schema_path:
            with open(schema_path) as f:
                self.grammar = json.load(f)
        else:
            self.grammar = None

        if self.mode == "offline":
            if not self.quantized:
                self.generator = LLM(
                    model=self.model_path, tensor_parallel_size=num_gpus
                )
            else:
                self.generator = LLM(
                    model=self.model_path,
                    tensor_parallel_size=num_gpus,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    quantization="bitsandbytes",
                    load_format="bitsandbytes",
                )

    @component.output_types(replies=list[str])
    def run(
        self, prompts: list[str], generation_kwargs: dict[str, Any] | None = None
    ) -> dict[str, list[str]]:
        # print(f"len of query: {len(query)}")
        # print(f"batch size: {self.batch_size}")

        if generation_kwargs is None:
            generation_kwargs = {}
        print(f"prompts: {prompts}")
        # sys.exit()
        generation_kwargs.pop("return_full_text")

        if "do_sample" in generation_kwargs:
            generation_kwargs.pop("do_sample")

        # replace max_new_tokens in generation kwargs with max_tokens
        generation_kwargs["max_tokens"] = generation_kwargs.pop("max_new_tokens")

        prompts = [prompt["prompt"] for prompt in prompts]

        if "gemma" in self.model_type:
            generation_kwargs["disable_compile"] = True

        if not self.grammar and not self.choices:
            sampling_params = SamplingParams(**generation_kwargs)
            outputs = self.generator.generate(
                prompts=prompts, sampling_params=sampling_params
            )
            responses = [output.outputs[0].text for output in outputs]

            return {"replies": responses}

        else:
            # setting max tokens to a larger amount when doing structured outputs
            # vllm default set to 16
            if "max_tokens" in generation_kwargs:
                generation_kwargs["max_tokens"] = 20

            if self.grammar:
                guided_decoding_params_json = GuidedDecodingParams(
                    json=json.dumps(self.grammar)
                )
                sampling_params = SamplingParams(
                    **generation_kwargs, guided_decoding=guided_decoding_params_json
                )

                print(prompts[0])
                time.sleep(10)
                outputs = self.generator.generate(
                    prompts=prompts, sampling_params=sampling_params
                )

                responses = []

                for output in outputs:
                    print(output.outputs[0].text)
                    # time.sleep(10)
                    try:
                        json_response = json.loads(output.outputs[0].text)
                        responses.append(json_response)

                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON: {output.outputs[0].text}")
                        responses.append(output.outputs[0].text)

                return {"replies": responses}

            elif self.choices:
                guided_decoding_params_choice = GuidedDecodingParams(
                    choice=self.choices
                )

                sampling_params = SamplingParams(
                    **generation_kwargs, guided_decoding=guided_decoding_params_choice
                )

                print(prompts[0])
                time.sleep(10)
                outputs = self.generator.generate(
                    prompts=prompts, sampling_params=sampling_params
                )

                responses = []

                for output in outputs:
                    response = output.outputs[0].text
                    if "True" in response:
                        responses.append(True)
                    elif "False" in response:
                        responses.append(False)
                    print(response)
                    # responses.append(response)
                    # time.sleep(10)

                return {"replies": responses}


@component
class BatchChromaEmbeddingRetriever:
    """A component to retrieve documents from a Chroma database using embeddings."""

    def __init__(
        self,
        document_store: chromadb.Collection,
    ):
        self.document_store = document_store

    @component.output_types(documents=list[list[Document]])
    def run(self, query_docs: list[Document], top_k: int = 3) -> list[list[Document]]:
        """Retrieve documents from a Chroma database using embeddings."""
        # print(f"query_docs: {query_docs}")
        query_embeddings = [embedding.embedding for embedding in query_docs]

        retrieved_docs = self.document_store.query(
            query_embeddings=query_embeddings, n_results=top_k
        )
        retrieved_documents = []
        doc_ids = retrieved_docs["ids"]
        doc_contents = retrieved_docs["documents"]
        doc_metas = retrieved_docs["metadatas"]

        for ids, contents, metas in zip(doc_ids, doc_contents, doc_metas, strict=False):
            per_batch = []
            for idx, content, meta in zip(ids, contents, metas, strict=False):
                haystack_doc = Document(id=idx, content=content, meta=meta)
                per_batch.append(haystack_doc)
            retrieved_documents.append(per_batch)

        return {"documents": retrieved_documents}


@component
class LogPromptPassthrough:
    """A component to log the input and pass it to the next node."""

    def __init__(self, log_fn: Callable = None):
        self.log_fn = log_fn
        if self.log_fn is None:
            # import logging
            # self._logger = logging.getLogger(__name__)
            # self.log_fn  = self._logger.debug
            self.log_fn = print

    @component.output_types(prompt=str)
    def run(self, prompt: str):
        self.log_fn(prompt)
        return {"prompt": prompt}


class BaseAbstractPipeline:
    def __init__(
        self,
        model_type: str,
        prompt_template_path: str,
        endpoint_type: str = ENDPOINT_TYPE,
        endpoint_url: str = ENDPOINT_URL,
        vllm_offline: bool = False,
        num_gpus: int = 1,
        quantized: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
        classification_kwargs: dict[str, Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        schema_path: str | None = None,
        choices_path: str | None = None,
        log_prompt: bool = True,
        model_lookup_table: dict[str, str] = MODELS,
        use_rag: bool = False,
        document_store: Any = None,
        embedding_model_name: str = "BAAI/bge-m3",
        use_local_embedding_model: bool = False,
        batch_size: int = 1,
    ):
        """
        Initialize the BaseAbstractPipeline component.

        Parameters
        ----------
        model_type : str
            The type of model to use.
        prompt_template_path : str
            Path to Jinja2 template from which to build the prompt.
        use_tgi : bool, optional
            Whether to use TGI, by default USE_TGI
        tgi_endpoint : str, optional
            A specific TGI endpoint that is different from the default
        quantized : bool, default False
            Whether or not a quantized model is used
        use_aip : bool, optional
            Whether to use AIP endpoints, by default USE_AIP
        generation_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments for the generation process, by default {}
        classification_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments for the classification process, by default {}
        examples : List[dict[str, str]], default []
            List of question and answer pairs.
        schema_path : str
            Path to the schema file
        log_prompt : bool, default False
            Whether to log the prompt after it is built
        use_rag : bool, default False
            Whether to use RAG
        document_store : Any, default None
            When using RAG, specify document_store
        embdedding_model_name : str, default "BAAI/bge-m3"
            When using RAG, specify embedding model name
        use_local_embedding_model : bool, default False
            Whether to use local embedding model
        batch_size : int, default 1
            Batch size for the model pipeline
        """
        if examples is None:
            examples = []
        if classification_kwargs is None:
            classification_kwargs = {}
        if generation_kwargs is None:
            generation_kwargs = {}
        self.model_type = model_type
        self.endpoint_type = endpoint_type
        self.endpoint_url = endpoint_url
        self.vllm_offline = vllm_offline
        self.num_gpus = num_gpus
        self.quantized = quantized
        self.generation_kwargs = generation_kwargs
        self.classification_kwargs = classification_kwargs
        self.log_prompt = log_prompt
        self.log_prompt = False

        self.examples = examples
        self.schema_path = schema_path
        self.choices_path = choices_path
        self.use_rag = use_rag
        self.document_store = document_store
        self.embedding_model_name = embedding_model_name
        self.use_local_embedding_model = use_local_embedding_model

        self.model_pipeline = Pipeline()
        self.model_lookup_table = model_lookup_table
        self.batch_size = batch_size
        self.prompt_template_path = prompt_template_path
        self.prompt_template = self.load_prompt_from_file()
        self.load_model()

    def load_prompt_from_file(self) -> str:
        prompt_template = Path(self.prompt_template_path).read_text()
        return prompt_template

    def load_model(self):
        """
        Load the model based on the model type.
        """
        if self.model_type == "roberta":
            self.discriminative_model_pipeline()
        elif self.model_type in list(MODELS.keys()):
            self.generative_model_pipeline()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def discriminative_model_pipeline(self):
        """
        Load the discriminative model and add it to the pipeline.
        """
        print(
            f"Instantiating discriminative model extractive pipeline from {self.model_lookup_table[self.model_type]}"
        )
        reader = ExtractiveReader(model=self.model_lookup_table[self.model_type])
        self.model_pipeline.add_component(name="reader", instance=reader)

    def generative_model_pipeline(self):
        """
        Load the generative model and add it to the pipeline.
        """
        if self.use_rag:
            if not self.use_local_embedding_model:
                embeddings_api_key = secret_from_env(
                    "EMBEDDINGS_API_KEY",
                )
                api_base_url = (settings.EMBEDDINGS_URL or "http://embeddings-bge.tld/v1")
                text_embedder = OpenAIDocumentEmbedder(
                    api_key=embeddings_api_key,
                    api_base_url=api_base_url,
                    model="tgi",
                )
            else:
                text_embedder = SentenceTransformersDocumentEmbedder(
                    model=EMBEDDING_MODELS[self.embedding_model_name]["local_path"]
                )

            # retriever  = ChromaEmbeddingRetriever(document_store=self.document_store)

            retriever = BatchChromaEmbeddingRetriever(
                document_store=self.document_store
            )

            self.model_pipeline.add_component("text_embedder", text_embedder)
            self.model_pipeline.add_component("retriever", retriever)
        # prompt_builder = PromptBuilder(self.prompt_template)
        prompt_builder = CustomPromptBuilder(
            prompt_template=self.prompt_template,
            use_grammar=False if not self.schema_path else True,  # noqa: SIM211
            run_local=not (self.use_aip or self.use_tgi or self.vllm_offline),
        )
        self.model_pipeline.add_component(
            name="prompt_builder", instance=prompt_builder
        )

        if self.use_aip:
            model_url = MODELS[self.model_type]["aip_endpoint"]
            if "v1" in model_url:
                self.generation_kwargs["max_tokens"] = self.generation_kwargs.pop(
                    "max_new_tokens"
                )
                self.generation_kwargs.pop("return_full_text", None)
                self.generation_kwargs.pop("do_sample", None)

                self.model_pipeline.add_component(
                    name="generator",
                    instance=CustomOpenAIGenerator(
                        model_type=self.model_type,
                        api_key=secret_from_env("OPENAI_API_KEY"),
                        api_base_url=model_url,
                    ),
                )
            else:
                self.model_pipeline.add_component(
                    name="generator", instance=AIPGenerator(model_type=self.model_type)
                )

        elif self.use_tgi:
            self.model_pipeline.add_component(
                name="generator", instance=TGIGenerator(tgi_endpoint=self.tgi_endpoint)
            )

        elif self.vllm_offline:
            self.model_pipeline.add_component(
                name="generator",
                instance=VLLMGenerator(
                    model_type=self.model_type,
                    quantized=self.quantized,
                    schema_path=self.schema_path,
                    choices_path=self.choices_path,
                    mode="offline",
                    num_gpus=self.num_gpus,
                ),
            )

        else:
            self.model_pipeline.add_component(
                name="generator",
                instance=LocalGenerator(
                    model_type=self.model_type,
                    quantized=self.quantized,
                    schema_path=self.schema_path,
                    choices_path=self.choices_path,
                    batch_size=self.batch_size,
                ),
            )
            # llm, gen_tokenizer = self.load_gen_model_and_tokenizer()
            # generator = HuggingFaceLocalGenerator(
            # llm,
            # huggingface_pipeline_kwargs={
            #     "device_map": "auto",
            #     "task": "text-generation",
            #     "model": llm,
            #     "tokenizer": gen_tokenizer,
            # },
            # )

            # self.model_pipeline.add_component(name="generator", instance=generator)

        if self.use_rag:
            self.model_pipeline.connect("text_embedder", "retriever")
            self.model_pipeline.connect("retriever", "prompt_builder.documents_list")
        # Optionally log the prompt
        if self.log_prompt:
            self.model_pipeline.add_component(
                name="log_prompt", instance=LogPromptPassthrough()
            )
            self.model_pipeline.connect("prompt_builder", "log_prompt")
            self.model_pipeline.connect("log_prompt", "generator")
        else:
            self.model_pipeline.connect("prompt_builder", "generator")

        # if self.use_rag:
        #     answer_builder = AnswerBuilder()
        #     self.model_pipeline.add_component(name="answer_builder", instance=answer_builder)
        #     self.model_pipeline.connect("generator.replies", "answer_builder.replies")
        #     self.model_pipeline.connect("retriever.documents", "answer_builder.documents")


@component
class ExtractiveQAPipeline(BaseAbstractPipeline):
    @component.output_types(answers=list[str])
    def run(
        self,
        queries: str | list[str],
        contexts: str | list[str],
    ) -> dict[str, list[str]]:
        """
        Run the pipeline with the given queries and contexts.

        Parameters
        ----------
        queries : Union[str, List[str]]
            The query or list of queries to process.
        contexts : Union[str, List[str]]
            The context or list of contexts corresponding to each query.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary containing the answers.
        """
        print(f"batch size in custom components: {self.batch_size}")
        # Ensure inputs are lists
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(contexts, str):
            contexts = [contexts]

        if len(queries) != len(contexts):
            raise ValueError("The number of queries must match the number of contexts.")

        docs = [Document(content=context) for context in contexts]

        if self.model_type == "roberta":
            outputs = []
            for query, doc in zip(queries, docs, strict=False):
                output = self.model_pipeline.run(
                    data={"reader": {"query": query, "top_k": 1, "documents": [doc]}}
                )
                print(output)
                answer = output["reader"]["answers"][0].data
                outputs.append(answer)
            return {"answers": outputs}

        else:
            try:
                output = self.model_pipeline.run(
                    data={
                        "generator": {"generation_kwargs": self.generation_kwargs},
                        "prompt_builder": {
                            "template_variables_list": [
                                {
                                    "query": query if query else "",
                                    "context": context,
                                    "examples": self.examples if self.examples else "",
                                }
                                for query, context in zip(
                                    queries, contexts, strict=False
                                )
                            ]
                        },
                    }
                )

                print(f"Raw Output: {output}")
            except KeyError as err:
                self.examples.pop()
                print(f"Reducing few-shot examples to {len(self.examples)}")
                if len(self.examples) == 0:
                    raise ValueError(
                        "Cannot reduce few_shot_examples further."
                    ) from err

            answers = output["generator"]["replies"]
            return {"answers": answers}


@component
class ClassificationPipeline(BaseAbstractPipeline):
    def load_model(self):
        """
        Load the model based on the model type.
        """
        if self.model_type == "roberta":
            self.discriminative_model_pipeline()
        elif self.model_type in MODELS:
            self.generative_model_pipeline()
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def discriminative_model_pipeline(self):
        """
        Load the discriminative model and add it to the pipeline.

        For sequence classification, classification_kwargs should specify:
        - return_all_scores=True,
        - function_to_apply : str
            Optional, sequence classification task only, function to apply to the model outputs in order to
            retrieve the scores. Accepts four different values:
                "default": if the model has a single label, will apply the sigmoid function on the output.
                           If the model has several labels, will apply the softmax function on the output.
                "sigmoid": Applies the sigmoid function on the output.
                "softmax": Applies the softmax function on the output.
                "none": Does not apply any function on the output.

        For token classification, classification_kwargs should specify:
        - aggregation_strategy="none",
        - ignore_labels : List[str]
            Optional, token classification task's list of token labels to ignore, defaults to []
        """
        # set device to utilize GPU if available
        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # the device to load the model onto

        print(
            f"Instantiating discriminative model classification pipeline from {self.model_lookup_table[self.model_type]}"
        )
        self.model_pipeline = pipeline(
            model=self.model_lookup_table[self.model_type],
            tokenizer=self.model_lookup_table[self.model_type],
            device=device,
            **self.classification_kwargs,
        )

    # FIXME: Type annotations for classification_output need to be verified correct
    @component.output_types(
        id=str,
        classification_input=str,
        classification_output=Union[str, list[tuple[str, str, str, str]]],  # noqa: UP007
        confidence=float,
    )
    def run(
        self,
        input_dict_batch: list[dict[str, str]],
        classification_type: str = "sequence",
    ) -> dict[str, str]:
        """
        Run the pipeline with the given query and context.

        Parameters
        ----------
        input_dict : Dict[str,str]
            A dictionary with the following keys:
                id (str) - Unique identifier for inference sample
                classification_input (str) - The text that the model should process to infer the classification labels,
                    e.g. "Transcript: american twelve level three five zero smooth"
        classification_type : str
            String indicating whether classification is sequence or token level, affects discriminative model
                One of {sequence (default), token}

        Returns
        -------
        Dict[str, str]
            A dictionary containing the answer.
        """
        if self.model_type == "roberta":
            if classification_type == "token":
                return self.run_discriminative_model_token_classification(
                    input_dict_batch
                )
            else:
                return self.run_discriminative_model_sequence_classification(
                    input_dict_batch
                )

        else:
            return self.run_generative_model_classification(input_dict_batch)

    def run_discriminative_model_sequence_classification(
        self, input_dict: dict[str, str]
    ) -> dict[str, Any]:
        """
        Runs the sequence classification inference pipeline.
        Ref: https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TextClassificationPipeline
        ...

        Parameters
        ----------
        input_dict : Dict[str,str]
            A dictionary with the following keys:
                id (str) - Unique identifier for inference sample
                classification_input (str) - The text that the model should process to infer the classification labels,
                    e.g. "Transcript: american twelve level three five zero smooth"
        ...

        Returns
        -------
        output_dict : Dict
            A dictionary with the following keys:
                id (str) - Unique identifier for inference sample
                classification_input (str) - The specific text that the model should process to infer the classification labels
                classification_output (str) - The classification output returned as an answer to the prompt.
                confidence (float) - The probability of the label
        """
        # Each result comes as a dictionary with the following keys:
        #      label (str) — The label predicted.
        #      score (float) — The corresponding probability.
        print(
            f"Executing sequence classification with {self.model_type} on input {input_dict['classification_input']}"
        )
        pipeline_output_dict = self.nlp_pipeline(input_dict["classification_input"])

        return {
            "id": input_dict["id"],
            "classification_input": input_dict["classification_input"],
            "classification_output": pipeline_output_dict["label"],
            "confidence": pipeline_output_dict["score"],
        }

    def run_discriminative_model_token_classification(
        self, input_dict: dict[str, str]
    ) -> dict[str, Any]:
        """
        Runs the token classification inference pipeline.
        Ref: https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.TokenClassificationPipeline
        ...

        Parameters
        ----------
        input_dict : Dict[str,str]
            A dictionary with the following keys:
                id (str) - Unique identifier for inference sample
                classification_input (str) - The text that the model should process to infer the classification labels,
                    e.g. "Transcript: american twelve level three five zero smooth"
        ...

        Returns
        -------
        output_dict : Dict
            A dictionary with the following keys:
                id (str) - Unique identifier for inference sample
                classification_input (str) - The specific text that the model should process to infer the classification labels
                classification_output (List[Tuple[int,str,str,float]]) - A list of tuples containing the token classification labels
        """
        # Each result comes as a list of dictionaries (one for each token in the corresponding input) with the following keys:
        #     word (str) — The token/word classified. This is obtained by decoding the selected tokens.
        #         If you want to have the exact string in the original sentence, use start and end.
        #     score (float) — The corresponding probability for entity.
        #     entity (str) — The entity predicted for that token/word.
        #     index (int) — The index of the corresponding token in the sentence.
        #     start (int, optional) — The index of the start of the corresponding entity in the sentence.
        #         Only exists if the offsets are available within the tokenizer
        #     end (int, optional) — The index of the end of the corresponding entity in the sentence.
        #         Only exists if the offsets are available within the tokenizer
        print(
            f"Executing token classification with {self.model_type} on input {input_dict['classification_input']}"
        )
        pipeline_output_dict = self.nlp_pipeline(input_dict["classification_input"])

        return {
            "id": input_dict["id"],
            "classification_input": input_dict["classification_input"],
            "classification_output": [
                (d["index"], d["word"], d["entity"], d["score"])
                for d in pipeline_output_dict
            ],
            "confidence": sum([float(d["score"]) for d in pipeline_output_dict])
            / len(pipeline_output_dict),
        }

    def run_generative_model_classification(
        self, input_dict_batch: list[dict[str, str]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Runs classification inference by wrapping the query in task-specific prompt,
        then returns the answer to the query.

        Parameters
        ----------
        input_dict_batch : List[Dict[str, str]]
            A batch of dictionaries containing classification tasks and inputs.

        Returns
        -------
        output_dict : Dict[str, List[Dict[str, Any]]]
            A dictionary containing classification results for each input.
        """
        # print(f"Executing classification with {self.model_type} on input {input_dict_batch}")

        # Load schema if provided
        json_schema = None
        if self.schema_path:
            with open(self.schema_path) as f:
                json_schema = json.dumps(json.load(f))

        # Prepare prompt variables for the model
        template_variables_list = [
            {
                "examples": self.examples if self.examples else "",
                "classification_input": input_dict["classification_input"],
                **({"json_schema": json_schema} if json_schema else {}),
            }
            for input_dict in input_dict_batch
        ]

        # Prepare model input data
        model_input_data = {
            "generator": {"generation_kwargs": self.classification_kwargs},
            "prompt_builder": {"template_variables_list": template_variables_list},
        }
        if json_schema and (self.use_aip or self.use_tgi):
            model_input_data["generator"]["grammar"] = {
                "type": "json",
                "value": json_schema,
            }

        # Run the model pipeline
        try:
            # print(f"data: {model_input_data}")
            output = self.model_pipeline.run(data=model_input_data)
            answers = output["generator"]["replies"]
        except KeyError as err:
            print(f"KeyError encountered: {err}")
            if self.examples:
                self.examples.pop()
                print(f"Reducing few-shot examples to {len(self.examples)}")
                if len(self.examples) == 0:
                    raise ValueError(
                        "Cannot reduce few-shot examples further."
                    ) from err
            return self.run_generative_model_classification(
                input_dict_batch
            )  # Retry with fewer examples

        # Construct output
        output_list = [
            {
                "id": input_dict["id"],
                "classification_input": input_dict["classification_input"],
                "classification_output": answer,
                "confidence": -1.0,  # Placeholder for confidence
            }
            for input_dict, answer in zip(input_dict_batch, answers, strict=False)
        ]

        return {"answers": output_list}

        # return {
        #     "id" : input_dict["id"],
        #     "classification_input" : input_dict["classification_input"],
        #     "classification_output" : answer,
        #     "confidence" : -1.0,
        # }


@component
class RAGPipeline(BaseAbstractPipeline):
    @component.output_types(answers=list[dict[str, list[str]]])
    def run(
        self, queries: list[str], top_k: int = 5, doc_id_field: str = "ACN"
    ) -> dict[str, Any]:
        """
        Ask a question to the RAG agent.
        Args:
            query (str): The question to ask.
            top_k (str, optional): Number of documents to retrieve. Default is 5
        Returns:
            Dict[str, Any]: The response from the RAG agent.
        """
        # print(f"Question Query: ", query)

        queries = [
            Document(content=query["question"], meta={"type": "question"})
            for query in queries
        ]

        while True:
            try:
                result = self.model_pipeline.run(
                    {
                        "text_embedder": {"documents": queries},
                        "retriever": {"top_k": top_k},
                        "prompt_builder": {
                            "template_variables_list": [
                                {
                                    "query": query,
                                    "examples": self.examples if self.examples else "",
                                }
                                for query in queries
                            ]
                        },
                        # "answer_builder": {"query": queries},
                        "generator": {"generation_kwargs": self.generation_kwargs},
                    },
                    include_outputs_from=["retriever"],
                )
                # print(f"Result: {result}")
                # print(f"PRINT RESULTS FROM PROMPT BUILDER: {result['prompt_builder']}")

                break
            except KeyError as err:
                self.examples.pop()
                print(f"reducing few shot examples to {len(self.examples)}")
                if len(self.examples) == 0:
                    raise ValueError(
                        "Cannot reduce few_shot_examples further."
                    ) from err

        answers = result["generator"]["replies"]
        retrieved_documents = result["retriever"]["documents"]

        retrieved_doc_ids = []
        for retrieved_documents_batch in retrieved_documents:
            if doc_id_field == "default":
                doc_ids = [doc.id for doc in retrieved_documents_batch]
            else:
                doc_ids = [doc.meta[doc_id_field] for doc in retrieved_documents_batch]

            retrieved_doc_ids.append(doc_ids)

        responses = [
            {"answer": answer, "doc_ids": doc_ids}
            for answer, doc_ids in zip(answers, retrieved_doc_ids, strict=False)
        ]
        return {"answers": responses}


@component
class GlossaryTermsPipeline(BaseAbstractPipeline):
    @component.output_types(answer=list[str])
    def run(self, glossary_term: str):
        with open(self.schema_path) as f:
            json_schema = json.load(f)

        result = self.model_pipeline.run(
            {
                "prompt_builder": {
                    "template_variables": {
                        "json_schema": json_schema,
                        "glossary_term": glossary_term,
                    }
                },
                "generator": {
                    "grammar": {"type": "json", "value": json_schema},
                    "generation_kwargs": self.generation_kwargs,
                },
            }
        )

        json_response = json.loads(result["generator"]["replies"][0])
        return {"answer": json_response}


@component
class MCQAPipeline(BaseAbstractPipeline):
    @component.output_types(answers=list[str])
    def run(self, queries: list[str]):
        with open(self.schema_path) as f:
            json_schema = json.load(f)

        json_schema = json.dumps(json_schema)

        if self.use_aip or self.use_tgi:
            result = self.model_pipeline.run(
                {
                    "prompt_builder": {
                        "template_variables": {
                            "json_schema": json_schema,
                            "query": queries,
                        }
                    },
                    "generator": {
                        "grammar": {"type": "json", "value": json_schema},
                        "generation_kwargs": self.generation_kwargs,
                    },
                }
            )
            json_response = json.loads(result["generator"]["replies"][0])
            return {"answer": json_response}
        else:
            result = self.model_pipeline.run(
                {
                    "prompt_builder": {
                        "template_variables_list": [
                            {
                                "query": query if query else "",
                                "examples": self.examples if self.examples else "",
                            }
                            for query in queries
                        ]
                    },
                }
            )

            # print("result: ", result)
            result = result["generator"]["replies"][0]

            return {"answers": result}


@component
class SummarizationPipeline(BaseAbstractPipeline):
    @component.output_types(answers=list[str])
    def run(
        self,
        inputs: str | list[str],
    ) -> dict[str, list[str]]:
        """
        Run the pipeline with the given queries and contexts.

        Parameters
        ----------
        queries : Union[str, List[str]]
            The query or list of queries to process.
        contexts : Union[str, List[str]]
            The context or list of contexts corresponding to each query.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary containing the answers.
        """
        print(f"batch size in custom components: {self.batch_size}")
        # Ensure inputs are lists
        if isinstance(inputs, str):
            inputs = [inputs]

        try:
            output = self.model_pipeline.run(
                data={
                    "generator": {"generation_kwargs": self.generation_kwargs},
                    "prompt_builder": {
                        "template_variables_list": [
                            {
                                "input": input_text if input_text else "",
                                "examples": self.examples if self.examples else "",
                            }
                            for input_text in inputs
                        ]
                    },
                }
            )

            print(f"Raw Output: {output}")
        except KeyError as err:
            self.examples.pop()
            print(f"Reducing few-shot examples to {len(self.examples)}")
            if len(self.examples) == 0:
                raise ValueError("Cannot reduce few_shot_examples further.") from err

        answers = output["generator"]["replies"]
        return {"answers": answers}


@component
class BinaryClassificationPipeline(BaseAbstractPipeline):
    @component.output_types(answers=list[str])
    def run(self, texts: list[str]):
        json_schema = {}

        if self.schema_path:
            with open(self.schema_path) as f:
                json_schema = json.load(f)

            json_schema = json.dumps(json_schema)

        template_variables_list = [
            {
                "examples": self.examples if self.examples else "",
                "classification_input": text,
                **({"json_schema": json_schema} if json_schema else {}),
            }
            for text in texts
        ]

        # Prepare model input data
        model_input_data = {
            "generator": {"generation_kwargs": self.generation_kwargs},
            "prompt_builder": {"template_variables_list": template_variables_list},
        }
        if json_schema and (self.use_aip or self.use_tgi):
            model_input_data["generator"]["grammar"] = {
                "type": "json",
                "value": json_schema,
            }

        # print(f"data: {model_input_data}")
        output = self.model_pipeline.run(data=model_input_data)
        answers = output["generator"]["replies"]

        return {"answers": answers}
