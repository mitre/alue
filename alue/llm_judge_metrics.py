# General Imports
import json
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import outlines
from dotenv import load_dotenv
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from jinja2 import Environment, FileSystemLoader

# LLM Specific Imports
from pydantic import BaseModel, conlist
from tqdm import tqdm
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

outlines.disable_cache()


class Score_NoExp(BaseModel):
    score: int


class Score_Exp(BaseModel):
    score: int
    explanation: str


class Claim(BaseModel):
    claim_list: conlist(item_type=str, min_length=1, max_length=10)


# ========================================
# BASE INFERENCE ENGINE CLASS
# ========================================


class BaseInferenceEngine(ABC):
    """Abstract base class for all inference engines"""

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.model_name_or_path = None
        self.client = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self) -> Any:
        """Load and return the model/client"""
        pass

    @abstractmethod
    def generate_structured(self, messages: list[dict], schema: dict) -> str:
        """Generate structured output given messages and schema"""
        pass

    @abstractmethod
    def generate_unstructured(self, prompt: str) -> str:
        """Generate unstructured text output"""
        pass

    def _load_tokenizer(self, model_path: str):
        """Helper method to load tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            print(f"[{self.__class__.__name__}] Tokenizer loaded for {model_path}")
        except Exception as e:
            print(f"[{self.__class__.__name__}] Warning: Could not load tokenizer: {e}")
            self.tokenizer = None


# ========================================
# CONCRETE INFERENCE ENGINE IMPLEMENTATIONS
# ========================================


class TransformersEngine(BaseInferenceEngine):
    """Inference engine using Transformers/Outlines for local inference"""

    def __init__(self, model_type: str, quantized: bool, **quantization_kwargs):
        super().__init__(model_type)
        self.quantized = quantized
        self.quantization_kwargs = quantization_kwargs

        # Import your actual TransformersEngine
        from model_engines import TransformersEngine as TE

        self.engine = TE(model_type, quantized, **quantization_kwargs)
        self.model_name_or_path = self.engine.model_name_or_path

    def load_model(self) -> Any:
        """Load the transformers model"""
        self.client = self.engine.load_model()
        self._load_tokenizer(self.model_name_or_path)
        return self.client

    def generate_structured(self, messages: list[dict], schema: dict) -> str:
        """Generate structured output using outlines"""
        try:
            import outlines

            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Create structured generator
            generator = outlines.generate.json(self.client, schema)
            result = generator(prompt)

            # Convert to JSON string
            if isinstance(result, dict):
                return json.dumps(result)
            elif hasattr(result, "model_dump"):
                return json.dumps(result.model_dump())
            else:
                return json.dumps({"error": "Invalid structured output"})

        except Exception as e:
            print(f"[TransformersEngine] Structured generation failed: {e}")
            return self.generate_unstructured(messages[0]["content"])

    def generate_unstructured(self, prompt: str) -> str:
        """Fallback generation without structured output"""
        # Implement based on your transformers setup
        return '{"claim_list": ["Fallback: structured generation not available"]}'


class vLLMOfflineEngine(BaseInferenceEngine):
    """Inference engine using vLLM for offline inference"""

    def __init__(self, model_type: str, quantized: bool, dtype=None):
        super().__init__(model_type)
        self.quantized = quantized
        self.dtype = dtype

        # Import your actual vLLMofflineEngine
        from model_engines import vLLMofflineEngine

        self.engine = vLLMofflineEngine(model_type, quantized, dtype)
        self.model_name_or_path = self.engine.model_name_or_path

    def load_model(self) -> Any:
        """Load the vLLM model"""
        self.client = self.engine.load_model()
        self._load_tokenizer(self.model_name_or_path)
        return self.client

    def generate_structured(self, messages: list[dict], schema: dict) -> str:
        """Generate structured output using vLLM guided decoding"""
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            guided_decoding_params = GuidedDecodingParams(json=schema)

            sampling_params = SamplingParams(
                temperature=0.0, max_tokens=512, guided_decoding=guided_decoding_params
            )

            outputs = self.client.generate([prompt], sampling_params)
            response_text = outputs[0].outputs[0].text.strip()

            return response_text

        except Exception as e:
            print(f"[vLLMOfflineEngine] Structured generation failed: {e}")
            return '{"error": "Generation failed"}'

    def generate_unstructured(self, prompt: str) -> str:
        """Generate unstructured output"""
        try:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
            outputs = self.client.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text.strip()
        except Exception as e:
            print(f"[vLLMOfflineEngine] Unstructured generation failed: {e}")
            return "Generation failed"


class vLLMOnlineEngine(BaseInferenceEngine):
    """Inference engine using vLLM online server (OpenAI-compatible)"""

    def __init__(
        self,
        model_type: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = None,
    ):
        super().__init__(model_type)
        self.base_url = base_url
        self.api_key = api_key

        # Import your actual vLLMonlineEngine
        from model_engines import vLLMonlineEngine

        self.engine = vLLMonlineEngine(model_type, base_url, api_key)
        self.model_name_or_path = self.engine.model_type

    def load_model(self) -> Any:
        """Initialize the OpenAI client for vLLM server"""
        self.client = self.engine.load_model()
        return self.client

    def generate_structured(self, messages: list[dict], schema: dict) -> str:
        """Generate structured output using OpenAI-compatible API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                extra_body={"guided_json": schema},
                temperature=0.0,
                seed=42,
                max_tokens=512,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[vLLMOnlineEngine] Structured generation failed: {e}")
            return '{"error": "API request failed"}'

    def generate_unstructured(self, prompt: str) -> str:
        """Generate unstructured output"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[vLLMOnlineEngine] Unstructured generation failed: {e}")
            return "Generation failed"


class OllamaEngine(BaseInferenceEngine):
    """Inference engine using Ollama server (OpenAI-compatible)"""

    def __init__(
        self,
        model_type: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        super().__init__(model_type)
        self.base_url = base_url
        self.api_key = api_key

        # Import your actual OllamaEngine
        from model_engines import OllamaEngine as OE

        self.engine = OE(model_type, base_url, api_key)
        self.model_name_or_path = self.engine.model_name

    def load_model(self) -> Any:
        """Initialize the OpenAI client for Ollama server"""
        self.client = self.engine.load_model()
        return self.client

    def generate_structured(self, messages: list[dict], schema: dict) -> str:
        """Generate structured output using OpenAI-compatible API with Ollama"""
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name_or_path,
                messages=messages,
                temperature=0.0,
                response_format=schema,
                max_tokens=200,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[OllamaEngine] Structured generation failed: {e}")
            return '{"error": "Ollama request failed"}'

    def generate_unstructured(self, prompt: str) -> str:
        """Generate unstructured output"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[OllamaEngine] Unstructured generation failed: {e}")
            return "Generation failed"


# ========================================
# BASE JUDGE CLASS
# ========================================


class BaseLLMJudge(ABC):
    """Abstract base class for all LLM-based evaluation judges"""

    def __init__(
        self,
        inference_engine: BaseInferenceEngine,
        template_env_path: str = "/home/ssingh/alue/templates/llm_judge",
        task_type: str = "RAG",
    ):
        load_dotenv()
        print(f"[BaseLLMJudge] Initializing {self.__class__.__name__}")

        self.inference_engine = inference_engine
        self.template_env_path = template_env_path
        self.task_type = task_type

        # Initialize the inference engine
        self.inference_engine.load_model()

    def _load_template(self, template_name: str) -> Any:
        """Load a Jinja2 template by its relative path within the template directory."""
        try:
            print(
                f"[BaseLLMJudge] Loading template: {template_name} from {self.template_env_path}"
            )
            full_template_env = Environment(
                loader=FileSystemLoader(self.template_env_path), autoescape=True
            )
            template = full_template_env.get_template(template_name)
            print(f"[BaseLLMJudge] Template {template_name} loaded successfully.")
            return template
        except Exception as e:
            print(f"[BaseLLMJudge] Failed to load template {template_name}: {e}")
            return None

    def decompose_claims(self, input_text: str) -> list[str]:
        """Decompose claims from input text using structured generation"""
        print(f"[BaseLLMJudge] Decomposing claims for input: {input_text[:50]}...")

        # System prompt with decomposition instructions
        system_prompt = """You are an expert at decomposing text into individual claims. Break down the given text into a list of specific, factual statements.
        Key requirements:
            - Each claim must be understandable when read independently
            - Include necessary context within each claim (subjects, timeframes, etc.)
            - Break complex sentences into simpler, atomic statements
            - Return 1-10 claims as appropriate
        Respond with a JSON object containing a 'claim_list' field with an array of claim strings."""

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input Text: {input_text}"},
        ]

        # Get schema
        claim_schema = Claim.model_json_schema()

        # Make structured request through inference engine
        response_content = self.inference_engine.generate_structured(
            messages, claim_schema
        )

        # Parse response
        try:
            json_resp = json.loads(response_content)
            claims = json_resp["claim_list"]
            print(f"[BaseLLMJudge] Decomposed claims: {claims}")
            return claims
        except json.JSONDecodeError as e:
            print(f"[BaseLLMJudge] Failed to parse JSON response: {e}")
            return []

    def client_response(
        self, input_text: str, explanations: bool, timeout: int = 5
    ) -> dict:
        """Generate score and explanation for input text"""
        print(f"[BaseLLMJudge] Getting client response. Explanations: {explanations}")

        # Determine schema based on explanations flag
        if explanations:
            score_schema = Score_Exp.model_json_schema()
            fallback_response = {"score": 0, "explanation": "JSON parsing failed"}
        else:
            score_schema = Score_NoExp.model_json_schema()
            fallback_response = {"score": 0}

        # Prepare messages
        messages = [{"role": "user", "content": input_text}]

        # Retry logic
        while True:
            try:
                print(f"[BaseLLMJudge] Sending prompt to LLM: {input_text[:60]}...")

                response_content = self.inference_engine.generate_structured(
                    messages, score_schema
                )

                # Parse response with fallback
                try:
                    if isinstance(response_content, str):
                        result = json.loads(response_content)
                    else:
                        result = response_content

                    print(
                        "[BaseLLMJudge] LLM response received and parsed successfully."
                    )
                    return result

                except json.JSONDecodeError as e:
                    print(
                        f"[BaseLLMJudge] JSON parsing failed: {e}. Using fallback score of 0."
                    )
                    return fallback_response

            except Exception as e:
                error_message = str(e)
                if "failure to get a peer from the ring-balancer" in error_message:
                    print(
                        f"Encountered error: {error_message}. Retrying in {timeout} seconds..."
                    )
                    time.sleep(timeout)
                elif (
                    "Request timed out" in error_message
                    or "upstream connect error" in error_message
                ):
                    print("Timeout error occurred. Retrying...")
                    time.sleep(timeout)
                else:
                    print(f"An unexpected error occurred: {error_message}")
                    raise

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Any:
        """Abstract method for evaluation logic - to be implemented by specific judges"""
        pass


# ========================================
# CONCRETE JUDGE IMPLEMENTATIONS
# ========================================


class ContextRelevancyJudge(BaseLLMJudge):
    """Judge for evaluating context relevancy in RAG systems"""

    def __init__(
        self,
        inference_engine: BaseInferenceEngine,
        collection_name: str,
        persist_path: str,
        explanations: bool = False,
        **kwargs,
    ):
        print("[ContextRelevancyJudge] Initializing...")
        super().__init__(inference_engine, **kwargs)

        self.explanations = explanations
        self.doc_store = ChromaDocumentStore(
            collection_name=collection_name, persist_path=persist_path
        )
        print("[ContextRelevancyJudge] ChromaDocumentStore initialized.")

    def _load_from_predictions_file(self, filename: str) -> list[dict]:
        """Load data from a predictions.json format file"""
        print(f"[ContextRelevancyJudge] Loading predictions from {filename}")

        with open(filename) as f:
            curr_predictions = json.load(f)
        print("[ContextRelevancyJudge] Predictions loaded")

        overall_dataset = []
        for i, curr_prediction in tqdm(curr_predictions.items()):
            print(f"[ContextRelevancyJudge] Processing prediction {i}")

            context_metadata = []
            for curr_context_id in curr_prediction["predicted_doc_ids"]:
                print(f"curr_context_id: {curr_context_id}")
                filter_1 = {"field": "id", "operator": "==", "value": curr_context_id}
                # print(f"filtered: {self.doc_store.filter_documents(filters=filter_1)}")

                retrieved_context = self.doc_store.filter_documents(filters=filter_1)[
                    0
                ].content
                context_metadata.append({curr_context_id: retrieved_context})
                print(
                    f"[ContextRelevancyJudge] Retrieved context for doc_id {curr_context_id}"
                )

            overall_dataset.append(
                {
                    "question": curr_prediction["question"],
                    "reference": curr_prediction["ground_truth_answer"],
                    "answer": curr_prediction["answer"],
                    "context": context_metadata,
                }
            )

        print("[ContextRelevancyJudge] Finished loading predictions file.")
        return overall_dataset

    def evaluate(
        self,
        filename: str,
        store_output: bool = True,
        output_path: str = "context_rel.json",
    ) -> list[dict]:
        """Calculate Context Relevancy Scores for the entire file"""
        print(f"[ContextRelevancyJudge] Starting evaluation for {filename}")

        dataset = self._load_from_predictions_file(filename)
        # dataset = dataset[:5]
        print("[ContextRelevancyJudge] Dataset loaded!")

        cr_template = self._load_template("context_relevancy.jinja2")
        if cr_template is None:
            raise ValueError("Could not load context_relevancy.jinja2 template")

        evaluated_output = []
        for curr_item in tqdm(dataset):
            print(
                f"[ContextRelevancyJudge] Evaluating question: {curr_item['question'][:50]}..."
            )

            context_scores = []
            for curr_context in curr_item["context"]:
                curr_context_id = next(iter(curr_context))
                retrieved_context = list(curr_context.values())[0]

                cr_template_formatted = cr_template.render(
                    user_query=curr_item["question"], retrieved_chunk=retrieved_context
                )

                resp = self.client_response(
                    input_text=cr_template_formatted, explanations=self.explanations
                )

                print(
                    f"[ContextRelevancyJudge] Context Relevancy Score generated for context {curr_context_id}"
                )

                context_scores.append(
                    {
                        curr_context_id: retrieved_context,
                        "context_relevancy": resp["score"],
                    }
                )

            evaluated_output.append(
                {
                    "question": curr_item["question"],
                    "reference": curr_item["reference"],
                    "answer": curr_item["answer"],
                    "context": context_scores,
                }
            )

        if store_output:
            print(f"[ContextRelevancyJudge] Saving evaluated output to {output_path}")
            with open(output_path, "w") as f:
                json.dump(evaluated_output, f)

        print("[ContextRelevancyJudge] Evaluation complete.")
        return evaluated_output


class CompositeCorrectnessJudge(BaseLLMJudge):
    """Judge for evaluating composite correctness in RAG Q&A systems"""

    def __init__(
        self,
        inference_engine: BaseInferenceEngine,
        explanations: bool = False,
        **kwargs,
    ):
        print("[CompositeCorrectnessJudge] Initializing...")
        super().__init__(inference_engine, **kwargs)
        self.explanations = explanations

    def _process_data(self, dataset: list[dict]) -> list[dict]:
        """Creates the dataset with the claim decompositions"""
        print("[CompositeCorrectnessJudge] Processing data for claim decomposition...")

        overall_dataset = []
        for item in tqdm(dataset):
            print(
                f"[CompositeCorrectnessJudge] Decomposing answer for question: {item['question'][:50]}..."
            )
            overall_dataset.append(
                {
                    "question": item["question"],
                    "reference": item["reference"],
                    "answer": item["answer"],
                    "context": item["context"],
                    "decomposed_response": self.decompose_claims(item["answer"]),
                }
            )
        print("[CompositeCorrectnessJudge] Data processing complete.")
        return overall_dataset

    def _main_idea_check(self, question: str, claim: str) -> dict:
        """Check if the claim is a main idea in answering the question"""
        print(f"[CompositeCorrectnessJudge] Checking if claim is main idea: {claim}")

        main_idea_template = self._load_template("main_idea.jinja2")
        if main_idea_template is None:
            raise ValueError("Could not load main_idea template")

        main_idea_template_formatted = main_idea_template.render(
            user_query=question, claim=claim
        )

        main_idea = self.client_response(
            input_text=main_idea_template_formatted, explanations=self.explanations
        )
        print(f"[CompositeCorrectnessJudge] Main idea check result: {main_idea}")
        return main_idea

    def _claim_contained_reference_check(self, claim: str, reference: str) -> dict:
        """Check if the claim is contained in the reference response"""
        print(
            "[CompositeCorrectnessJudge] Checking if claim is contained in reference."
        )

        claim_contained_reference_template = self._load_template(
            "claim_contained_reference.jinja2"
        )
        if claim_contained_reference_template is None:
            raise ValueError("Could not load claim_contained_reference template")

        claim_contained_reference_template_formatted = (
            claim_contained_reference_template.render(
                claim=claim, reference_response=reference
            )
        )

        claim_contained = self.client_response(
            input_text=claim_contained_reference_template_formatted,
            explanations=self.explanations,
        )
        print(
            f"[CompositeCorrectnessJudge] Claim contained in reference check result: {claim_contained}"
        )
        return claim_contained

    def _claim_contradicts_reference_check(self, claim: str, reference: str) -> dict:
        """Check if the claim directly contradicts any part of the reference response"""
        print("[CompositeCorrectnessJudge] Checking if claim contradicts reference.")

        claim_contradicts_reference_template = self._load_template(
            "claim_contradicts_reference.jinja2"
        )
        if claim_contradicts_reference_template is None:
            raise ValueError(
                "Could not load claim_contradicts_reference.jinja2 template"
            )

        claim_contradicts_reference_template_formatted = (
            claim_contradicts_reference_template.render(
                claim=claim, reference_response=reference
            )
        )

        claim_contradicts = self.client_response(
            input_text=claim_contradicts_reference_template_formatted,
            explanations=self.explanations,
        )
        print(
            f"[CompositeCorrectnessJudge] Claim contradicts reference check result: {claim_contradicts}"
        )
        return claim_contradicts

    def _context_support_claim_check(self, claim: str, context: str) -> dict:
        """Check if the claim is supported by the context"""
        print("[CompositeCorrectnessJudge] Checking if context supports claim.")

        context_support_claim_template = self._load_template(
            "context_supported_claim.jinja2"
        )
        if context_support_claim_template is None:
            raise ValueError("Could not load context_supported_claim.jinja2 template")

        context_support_claim_template_formatted = (
            context_support_claim_template.render(claim=claim, context_info=context)
        )

        claim_contained_in_context = self.client_response(
            input_text=context_support_claim_template_formatted,
            explanations=self.explanations,
        )
        print(
            f"[CompositeCorrectnessJudge] Context support claim check result: {claim_contained_in_context}"
        )
        return claim_contained_in_context

    def evaluate(
        self,
        dataset: list[dict],
        store_output: bool = True,
        output_path: str = "comp_correctness.json",
    ) -> dict:
        """Calculate Composite Correctness Scores for the dataset"""
        print("[CompositeCorrectnessJudge] Starting evaluation...")

        dataset = self._process_data(dataset)

        # Store all data
        all_question_data = {}
        composite_correctness_average = 0

        for idx, item in tqdm(enumerate(dataset)):
            print(
                f"[CompositeCorrectnessJudge] Evaluating Question {idx}: {item['question'][:50]}..."
            )

            question = item["question"]
            answer = item["answer"]
            reference = item["reference"]
            decomposed_response = item["decomposed_response"]

            metadata = {"question": question, "answer": answer, "reference": reference}
            main_correct = False
            composite_correctness = 0

            # Analyze each claim within the question-response pair
            for i, claim in tqdm(enumerate(decomposed_response)):
                print(f"[CompositeCorrectnessJudge] Evaluating claim {i}: {claim}")

                claim_info = {"claim": claim}

                # Identify if the claim is a main idea or not
                main_idea = self._main_idea_check(question=question, claim=claim)
                claim_info["main_idea"] = (
                    "main" if main_idea["score"] == 1 else "supporting"
                )
                if self.explanations:
                    claim_info["main_idea_explanation"] = main_idea["explanation"]

                # Check if the claim is contained in the reference
                claim_contained = self._claim_contained_reference_check(
                    claim=claim, reference=reference
                )
                claim_info["claim_contained_reference_score"] = claim_contained["score"]
                if self.explanations:
                    claim_info["claim_contained_reference_explanation"] = (
                        claim_contained["explanation"]
                    )

                # If claim is part of reference
                if claim_info["claim_contained_reference_score"] == 1:
                    print(
                        f"[CompositeCorrectnessJudge] Claim {i} is contained in reference."
                    )
                    claim_info["claim_correctness_score"] = 1
                    composite_correctness += 1
                    claim_info["main_idea"] = "main"
                    main_correct = True

                else:
                    claim_correctness_score = 0

                    # Check to see if the claim contradicts the reference response
                    claim_contradicts = self._claim_contradicts_reference_check(
                        claim=claim, reference=reference
                    )
                    claim_info["claim_contradicts_reference_score"] = claim_contradicts[
                        "score"
                    ]
                    if self.explanations:
                        claim_info["claim_contradicts_reference_explanation"] = (
                            claim_contradicts["explanation"]
                        )

                    # If the claim does contradict the reference, give it a correctness score of 0
                    if claim_contradicts["score"] == 1:
                        print(
                            f"[CompositeCorrectnessJudge] Claim {i} contradicts reference."
                        )
                        claim_correctness_score = 0

                    # If the claim does not contradict the reference, analyze each of the contexts
                    else:
                        for j, context in enumerate(item["context"]):
                            curr_context_id = list(context.keys())[0]
                            curr_context_text = list(context.values())[0]
                            curr_context_rel = list(context.values())[1]

                            context_info = {
                                "id": curr_context_id,
                                "text": curr_context_text,
                                "context_relevancy": curr_context_rel,
                            }

                            # Check if the context is relevant to the claim
                            claim_contained_in_context = (
                                self._context_support_claim_check(
                                    claim=claim, context=curr_context_text
                                )
                            )

                            # If context is relevant to claim and has a relevancy score of 1, set the claim correctness score to 1
                            if (
                                claim_contained_in_context["score"] == 1
                                and curr_context_rel == 1
                            ):
                                print(
                                    f"[CompositeCorrectnessJudge] Context {j} supports claim {i}."
                                )
                                context_info["context_claim_correctness_score"] = 1
                                if self.explanations:
                                    context_info["explanation"] = (
                                        claim_contained_in_context["explanation"]
                                    )
                                claim_correctness_score = 1  # If even 1 context supports the claim, give it a score of 1
                                if claim_info["main_idea"] == "main":
                                    main_correct = True

                            else:
                                context_info["context_claim_correctness_score"] = 0
                                if self.explanations:
                                    context_info["explanation"] = (
                                        claim_contained_in_context["explanation"]
                                    )

                            claim_info[f"context {j}"] = (
                                context_info  # Add info about current context to overall claim
                            )

                    claim_info["claim_correctness_score"] = (
                        claim_correctness_score  # Add claim correctness score to the claim
                    )
                    composite_correctness += claim_info[
                        "claim_correctness_score"
                    ]  # Add claim correctness to overall question correctness

                metadata[f"claim {i}"] = (
                    claim_info  # Add claim info to overall question info
                )

            # Only considers score if at least one main claim is true
            metadata["main_correct"] = main_correct
            metadata["composite_correctness_score"] = (
                composite_correctness / len(decomposed_response) if main_correct else 0
            )
            print(
                f"[CompositeCorrectnessJudge] Composite correctness for Question {idx}: {metadata['composite_correctness_score']}"
            )

            composite_correctness_average += metadata["composite_correctness_score"]
            all_question_data[f"Question {idx}"] = metadata

        composite_correctness_average_final = composite_correctness_average / len(
            dataset
        )
        print(
            f"[CompositeCorrectnessJudge] Final composite correctness average: {composite_correctness_average_final}"
        )

        all_question_data["composite_correctness_average"] = (
            composite_correctness_average_final
        )

        if store_output:
            print(
                f"[CompositeCorrectnessJudge] Saving evaluation results to {output_path}"
            )
            with open(output_path, "w") as f:
                json.dump(all_question_data, f)

        print("[CompositeCorrectnessJudge] Evaluation complete.")
        return all_question_data


class ClaimDecompositionJudge(BaseLLMJudge):
    """Judge for evaluating claim decomposition metrics with precision and recall"""

    def __init__(
        self,
        inference_engine: BaseInferenceEngine,
        explanations: bool = False,
        w_strong: float = 1.0,
        w_weak: float = 0.5,
        w_incorrect: float = 0.1,
        **kwargs,
    ):
        print("[ClaimDecompositionJudge] Initializing...")
        super().__init__(inference_engine, **kwargs)
        self.explanations = explanations
        self.w_strong = w_strong
        self.w_weak = w_weak
        self.w_incorrect = w_incorrect

    def _load_from_predictions_file(self, filename: str) -> list[dict]:
        """Load data from a predictions.json format file"""
        print(f"[ClaimDecompositionJudge] Loading predictions from {filename}")

        with open(filename) as f:
            curr_predictions = json.load(f)
        print("[ClaimDecompositionJudge] Predictions loaded")

        overall_dataset = []
        for i, curr_prediction in tqdm(curr_predictions.items()):
            # Store all information associated with that one question
            overall_dataset.append(
                {
                    "narrative": curr_prediction["narrative"],
                    "ground_truth_summary": curr_prediction["ground_truth_summary"],
                    "predicted_summary": curr_prediction["predicted_summary"],
                    "index_value": i,
                }
            )

        print("[ClaimDecompositionJudge] Dataset processing complete.")
        return overall_dataset

    def _context_support_claim_check(self, claim: str, context: str) -> dict:
        """Check if the claim is supported by the context"""
        print("[ClaimDecompositionJudge] Checking if context supports claim.")

        context_support_claim_template = self._load_template(
            "context_supported_claijinja2"
        )
        if context_support_claim_template is None:
            raise ValueError("Could not load context_supported_claim.jinja2 template")

        context_support_claim_template_formatted = (
            context_support_claim_template.render(claim=claim, context_info=context)
        )

        claim_contained_in_context = self.client_response(
            input_text=context_support_claim_template_formatted,
            explanations=self.explanations,
        )
        print(
            f"[ClaimDecompositionJudge] Context support claim check result: {claim_contained_in_context}"
        )
        return claim_contained_in_context

    def evaluate(
        self,
        filename: str,
        store_output: bool = True,
        output_path: str = "claim_decomposition.json",
    ) -> dict:
        """Calculate Claim Decomposition Scores for the entire file"""
        print(f"[ClaimDecompositionJudge] Starting evaluation for {filename}")

        dataset = self._load_from_predictions_file(filename)
        print("[ClaimDecompositionJudge] Dataset loaded!")

        evaluated_output = {}
        for curr_item in tqdm(dataset):
            print(
                f"[ClaimDecompositionJudge] Processing item {curr_item['index_value']}"
            )

            key = curr_item["index_value"]
            gt_summary = curr_item["ground_truth_summary"]
            narrative_input = curr_item["narrative"]

            # Compute gt_claims
            gt_claims_temp = self.decompose_claims(gt_summary)

            # Check if each gt_claim is supported by the input narrative
            gtclaim_support_vector = np.zeros(len(gt_claims_temp))
            for index_gt, cur_gt_claim in enumerate(gt_claims_temp):
                gtclaim_supported_by_input_narrative = (
                    self._context_support_claim_check(
                        claim=cur_gt_claim, context=narrative_input
                    )
                )
                gtclaim_support_vector[index_gt] = gtclaim_supported_by_input_narrative[
                    "score"
                ]

            # Include only the gt claims that support input narrative
            valid_gt_claims = np.where(gtclaim_support_vector == 1)
            gt_claims = [gt_claims_temp[i] for i in valid_gt_claims[0].tolist()]

            if len(gt_claims) > 0:
                # Compute predicted_claims
                pr_summary = curr_item["predicted_summary"]
                predicted_claims = self.decompose_claims(pr_summary)

                # Compute "strong" type scores for each predicted claim
                strong_support_matrix = np.zeros(
                    (len(predicted_claims), len(gt_claims))
                )
                strong_support_vector = np.zeros(len(predicted_claims))
                weak_support_vector = np.zeros(len(predicted_claims))
                incorrect_support_vector = np.zeros(len(predicted_claims))

                for index_pre, cur_predicted_claim in enumerate(predicted_claims):
                    # Check if the predicted claim is supported by the ground truth claim
                    for index_gt, cur_gt_claim in enumerate(gt_claims):
                        claim_supported_by_reference_claim = (
                            self._context_support_claim_check(
                                claim=cur_predicted_claim, context=cur_gt_claim
                            )
                        )
                        strong_support_matrix[index_pre, index_gt] = (
                            claim_supported_by_reference_claim["score"]
                        )
                    strong_support_vector[index_pre] = np.max(
                        strong_support_matrix[index_pre, :]
                    )

                    # Check if the predicted claim is supported by the input narrative
                    claim_supported_by_input_narrative = (
                        self._context_support_claim_check(
                            claim=cur_predicted_claim, context=narrative_input
                        )
                    )
                    weak_support_vector[index_pre] = claim_supported_by_input_narrative[
                        "score"
                    ]

                    if strong_support_vector[index_pre] == 1:
                        weak_support_vector[index_pre] = 0
                        incorrect_support_vector[index_pre] = 0

                    # Form incorrect score vector by deduction
                    if (
                        strong_support_vector[index_pre] == 0
                        and weak_support_vector[index_pre] == 0
                    ):
                        incorrect_support_vector[index_pre] = 1

                    if (
                        strong_support_vector[index_pre] == 0
                        and weak_support_vector[index_pre] == 1
                    ):
                        incorrect_support_vector[index_pre] = 0

                result_matrix = np.transpose(
                    np.array(
                        [
                            strong_support_vector,
                            weak_support_vector,
                            incorrect_support_vector,
                        ]
                    )
                )
                print("Result matrix:", result_matrix)
                print("Strong support matrix:", strong_support_matrix)

                # Compute precision and recall
                sc = np.array(result_matrix)
                str_support = np.array(strong_support_matrix)
                num_pr_claims = sc.shape[0]
                num_gt_claims = str_support.shape[1]

                # num_strong_claims = int(np.sum(sc[:, 0]))
                num_weak_claims = int(np.sum(sc[:, 1]))
                num_incorrect_claims = int(np.sum(sc[:, 2]))

                # Compute number of independent strong claims by examining strong_support_matrix
                # num_independent_strong_claims: number of predicted claims that are affiliated with one gt_claim at a time
                col_sum = np.sum(str_support, axis=1)
                temp_col_sum = np.where(col_sum == 1)
                num_independent_strong_claims = len(temp_col_sum[0])

                # Compute number of gt claims covered by predicted claims
                row_sum = np.sum(str_support, axis=0)
                temp_row_sum = np.where(row_sum == 0)
                num_gtclaims_covered_by_predictedclaims = num_gt_claims - len(
                    temp_row_sum[0]
                )

                precision = (
                    self.w_strong * num_independent_strong_claims
                    + self.w_weak * num_weak_claims
                    - self.w_incorrect * num_incorrect_claims
                ) / num_pr_claims
                # If precision becomes negative due to the penalty component, set precision value to 0.0
                if precision < 0:
                    precision = 0.0

                recall = num_gtclaims_covered_by_predictedclaims / num_gt_claims

                print("Precision:", precision)
                print("Recall:", recall)

                cur_output = {
                    "predicted_summary": pr_summary,
                    "ground_truth_summary": gt_summary,
                    "narrative": narrative_input,
                    "ground_truth_claims": gt_claims,
                    "predicted_claims": predicted_claims,
                    "strong_support_matrix": strong_support_matrix.tolist(),
                    "score_matrix": result_matrix.tolist(),
                    "precision": precision,
                    "recall": recall,
                }
                evaluated_output[key] = cur_output

        if store_output:
            print(
                f"[ClaimDecompositionJudge] Saving evaluation results to {output_path}"
            )
            with open(output_path, "w") as f:
                json.dump(evaluated_output, f)

        print("[ClaimDecompositionJudge] Evaluation complete.")
        return evaluated_output

    """Judge for evaluating composite correctness in RAG Q&A systems"""


# ========================================
# FACTORY FUNCTIONS
# ========================================


def create_inference_engine(engine_type: str, **kwargs) -> BaseInferenceEngine:
    """Factory function to create the appropriate inference engine"""
    engine_classes = {
        "transformers": TransformersEngine,
        "vllm_offline": vLLMOfflineEngine,
        "vllm_online": vLLMOnlineEngine,
        "ollama": OllamaEngine,
    }

    if engine_type not in engine_classes:
        raise ValueError(
            f"Unknown engine type: {engine_type}. Available: {list(engine_classes.keys())}"
        )

    return engine_classes[engine_type](**kwargs)


def create_judge(
    judge_type: str, inference_engine: BaseInferenceEngine, **kwargs
) -> BaseLLMJudge:
    """Factory function to create the appropriate judge type"""
    judge_classes = {
        "context_relevancy": ContextRelevancyJudge,
        "composite_correctness": CompositeCorrectnessJudge,
        "claim_decomposition": ClaimDecompositionJudge,
    }

    if judge_type not in judge_classes:
        raise ValueError(
            f"Unknown judge type: {judge_type}. Available: {list(judge_classes.keys())}"
        )

    return judge_classes[judge_type](inference_engine=inference_engine, **kwargs)


# ========================================
# USAGE EXAMPLES
# ========================================

if __name__ == "__main__":
    engine = create_inference_engine(
        engine_type="vllm_online",
        model_type="llama_instruct",
        base_url="http://localhost:8003/v1",
        api_key="",
        # quantized=True,
        # dtype=torch.bfloat16
    )

    cr_judge = create_judge(
        judge_type="context_relevancy",
        inference_engine=engine,
        collection_name="ast_ac",
        persist_path="/home/alue/data/aviation-llm/rag_eval/rag_eval",
        explanations=False,
        # template_env_path="/home/ssingh/alue/templates"
    )

    cr_outputs = cr_judge.evaluate(
        filename="/home/alue/results/2025_rag/rag_full/mixtral_22b_instruct_atc_rag_results_one_shot_20250125_213424/predictions.json"
    )

    cc_judge = create_judge(
        judge_type="composite_correctness",
        inference_engine=engine,
    )

    cc_outputs = cc_judge.evaluate(dataset=cr_outputs)
    print(cc_outputs)
