# General Imports
import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import outlines
from dotenv import load_dotenv
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# LLM Specific Imports
from pydantic import BaseModel, conlist
from tqdm import tqdm
from .inference import run_llm_inference
from .rag_utils import ChromaInterface

outlines.disable_cache()


class Score_NoExp(BaseModel):
    score: int


class Score_Exp(BaseModel):
    score: int
    explanation: str


class Claim(BaseModel):
    claim_list: conlist(item_type=str, min_length=1, max_length=10)



# ========================================
# BASE JUDGE CLASS
# ========================================


class BaseLLMJudge(ABC):
    """Abstract base class for all LLM-based evaluation judges"""

    def __init__(
        self,
        model_name: str,  
        task_type: str = "RAG",
    ):
        load_dotenv()
        print(f"[BaseLLMJudge] Initializing {self.__class__.__name__}")

        self.model_name = model_name
        self.task_type = task_type


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


        try:
            predictions = run_llm_inference(
                [messages], 
                self.model_name, 
                schema=Claim,
                fields_to_extract="claim_list",
                temperature=0.1,
                max_tokens=300
            )
            import sys
            print(f"decompose preds: {predictions}")
            # sys.exit()
            if predictions and predictions[0] != "ERROR":
                claims = predictions[0]
                print(f"[BaseLLMJudge] Decomposed claims: {claims}")
                return claims
            else:
                print(f"[BaseLLMJudge] Failed to decompose claims")
                return []
        
        except Exception as e:
            print(f"[BaseLLMJudge] Error in claim decomposition: {e}")
            return []
        
    def _context_support_claim_check(self, claim: str, context: str) -> dict:
        """Check if the claim is supported by the context"""
        print("[CompositeCorrectnessJudge] Checking if context supports claim.")

        system_prompt = """You are an expert evaluation system. Given a claim and context information, evaluate whether the claim is supported by the provided context.
        Instructions:
        - If the claim is supported by information in the context, return a score of 1
        - If the claim is not supported by the context, return a score of 0"""

        user_prompt = f"Claim: {claim}\nContext Information: {context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        schema = Score_Exp if self.explanations else Score_NoExp

        try:
            predictions = run_llm_inference(
                [messages],
                self.model_name,
                schema=schema,
                fields_to_extract=None,
                temperature=0.1,
                max_tokens=200
            )
            
            claim_contained_in_context = predictions[0] if predictions and predictions[0] != "ERROR" else {"score": 0}
            
        except Exception as e:
            print(f"[CompositeCorrectnessJudge] Context support claim check failed: {e}")
            claim_contained_in_context = {"score": 0}
        return claim_contained_in_context

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
        collection_name: str,
        database_path: str,
        explanations: bool = False,
        **kwargs,
    ):
        print("[ContextRelevancyJudge] Initializing...")
        super().__init__(**kwargs)

        self.explanations = explanations
        self.chroma_interface = ChromaInterface(database_path=database_path)
        self.collection = self.chroma_interface.get_or_create_collection(collection_name)
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
                result = self.collection.get(ids=[curr_context_id])
                retrieved_context = result["documents"][0] if result["documents"] else ""
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

        evaluated_output = []
        for curr_item in tqdm(dataset):
            print(
                f"[ContextRelevancyJudge] Evaluating question: {curr_item['question'][:50]}..."
            )

            context_scores = []
            for curr_context in curr_item["context"]:
                curr_context_id = next(iter(curr_context))
                retrieved_context = list(curr_context.values())[0]

                system_prompt = """You are an expert evaluation system. Given a user query and a retrieved context chunk, evaluate whether the context is relevant to answering the query.
                Instructions:
                - If the context contains information that is relevant to answering the user query, return a score of 1
                - If the context is not relevant to answering the user query, return a score of 0"""

                user_prompt = f"User Query: {curr_item['question']}\nRetrieved Context: {retrieved_context}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                schema = Score_Exp if self.explanations else Score_NoExp

                try:
                    predictions = run_llm_inference(
                        [messages],
                        self.model_name,
                        schema=schema,
                        fields_to_extract=None,
                        temperature=0.1,
                        max_tokens=200
                    )
                    
                    resp = predictions[0] if predictions and predictions[0] != "ERROR" else {"score": 0}
                    
                except Exception as e:
                    print(f"[ContextRelevancyJudge] Context relevancy check failed: {e}")
                    resp = {"score": 0}

                print(
                    f"[ContextRelevancyJudge] Context Relevancy Score generated for context {curr_context_id}"
                )

                

                context_scores.append(
                    {
                        curr_context_id: retrieved_context,
                        "context_relevancy": resp["score"],
                        "explanation": resp["explanation"] if "explanation" in resp else ""
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
        explanations: bool = False,
        **kwargs,
    ):
        print("[CompositeCorrectnessJudge] Initializing...")
        super().__init__(**kwargs)
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

        system_prompt = """You are an expert evaluation system. Given a question and a claim, evaluate whether the claim directly answers any part of the question.
        Instructions:
        - If the claim directly answers any part of the question, return a score of 1
        - If the claim does not directly answer any parts of the question, return a score of 0"""

        user_prompt = f"Question: {question}\nClaim: {claim}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}    
        ]

        schema = Score_Exp if self.explanations else Score_NoExp
        try:
            predictions = run_llm_inference(
                [messages],
                self.model_name,
                fields_to_extract=None,
                schema=schema,
                temperature=0.1,
                max_tokens=200
            )
            
            main_idea = predictions[0] if predictions and predictions[0] != "ERROR" else {"score": 0}
            
        except Exception as e:
            print(f"[CompositeCorrectnessJudge] Main idea check failed: {e}")
            main_idea = {"score": 0}

        return main_idea

    def _claim_contained_reference_check(self, claim: str, reference: str) -> dict:
        """Check if the claim is contained in the reference response"""
        print(
            "[CompositeCorrectnessJudge] Checking if claim is contained in reference."
        )

        system_prompt = """You are an expert evaluation system. Given a claim and a reference response, evaluate whether the claim is contained within or supported by the reference response.
        Instructions:
        - If the reference answer supports or directly mentions the claim, return a score of 1.
        - If the claim is not contained in or supported by the reference response, return a score of 0"""

        user_prompt = f"Claim: {claim}\nReference Response: {reference}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        schema = Score_Exp if self.explanations else Score_NoExp

        try:
            predictions = run_llm_inference(
                [messages],
                self.model_name,
                schema=schema,
                fields_to_extract=None,
                temperature=0.1,
                max_tokens=200
            )
            
            claim_contained = predictions[0] if predictions and predictions[0] != "ERROR" else {"score": 0}
            
        except Exception as e:
            print(f"[CompositeCorrectnessJudge] Claim contained reference check failed: {e}")
            claim_contained = {"score": 0}

        return claim_contained

    def _claim_contradicts_reference_check(self, claim: str, reference: str) -> dict:
        """Check if the claim directly contradicts any part of the reference response"""
        print("[CompositeCorrectnessJudge] Checking if claim contradicts reference.")

        system_prompt = """You are an expert evaluation system. Given a claim and a reference response, evaluate whether the claim directly contradicts any part of the reference response.
        Instructions:
        - If the claim directly contradicts, return a score of 1
        - If the claim does not contradict the reference response, return a score of 0"""

        user_prompt = f"Claim: {claim}\nReference Response: {reference}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        schema = Score_Exp if self.explanations else Score_NoExp

        try:
            predictions = run_llm_inference(
                [messages],
                self.model_name,
                schema=schema,
                fields_to_extract=None,
                temperature=0.1,
                max_tokens=200
            )
            
            claim_contradicts = predictions[0] if predictions and predictions[0] != "ERROR" else {"score": 0}
            
        except Exception as e:
            print(f"[CompositeCorrectnessJudge] Claim contradicts reference check failed: {e}")
            claim_contradicts = {"score": 0}

        return claim_contradicts

    

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
        explanations: bool = False,
        w_strong: float = 1.0,
        w_weak: float = 0.5,
        w_incorrect: float = 0.1,
        **kwargs,
    ):
        print("[ClaimDecompositionJudge] Initializing...")
        super().__init__(**kwargs)
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
# USAGE EXAMPLES
# ========================================

if __name__ == "__main__":
    judge = ContextRelevancyJudge(
        model_name="llama_instruct",
        collection_name="test-collection-fast4",  # Update with your actual collection name
        database_path="/Users/ssingh/alue/test_chroma_db",   # Update with your actual Chroma database path
        explanations=True  # Set to True if you want explanations
    )
    print(f"judge: {judge}")

    results = judge.evaluate(
        filename="/Users/ssingh/alue/results_rag_20250924_123652/predictions.json",
        store_output=True,
        output_path="test_context_relevancy_exp.json"
    )

    judge = CompositeCorrectnessJudge(
        model_name="llama_instruct",  # Update with your actual model name
        explanations=True  # Set to True to see detailed explanations
    )

    results = judge.evaluate(
        dataset=results,
        store_output=True,
        output_path="sample_composite_correctness.json"
    )

