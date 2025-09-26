"""
This module provides a unified interface for multiple LLM inference backends,
supporting both unstructured text generation and structured JSON output. It
abstracts away the differences between OpenAI API, self-hosted APIs (vLLM, TGI,
Ollama), and local transformers models.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from alue.settings import get_settings


class BaseInferenceEngine(ABC):
    """Abstract base class for all inference engines.
    
    This class defines the interface that all inference engines must implement,
    providing methods for both unstructured and structured text generation.
    
    Attributes:
        model_name: The name or identifier of the model to use.
        settings: Application settings loaded from environment or config.
    """

    def __init__(self, model_name: str):
        """Initialize the base inference engine.
        
        Args:
            model_name: Name or path of the model to use for inference.
        """
        self.model_name = model_name
        self.settings = get_settings()

    @abstractmethod
    def generate_unstructured(self, 
                              messages: List[Dict[str, str]], 
                              **kwargs: Any) -> str:
        """Generate unstructured text response from messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.).
            
        Returns:
            Generated text response as a string.
        """
        pass

    @abstractmethod
    def generate_structured(self, 
                            messages: List[Dict[str, str]], 
                            schema: Dict[str, Any], 
                            **kwargs: Any) -> str:
        """Generate structured JSON response conforming to a schema.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            schema: JSON schema or Pydantic model defining the output structure.
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.).
            
        Returns:
            Generated JSON response as a string.
        """
        pass

    def _apply_chat_template(self, 
                             messages: List[Dict[str, str]]) -> str:
        """Convert messages to a prompt string using a simple fallback template.
        
        This method provides a basic chat template when the model's native
        template is unavailable. It formats messages as:
        "System: {content}\\n\\nUser: {content}\\n\\nAssistant:"
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            
        Returns:
            Formatted prompt string ready for generation.
        """
        # Simple fallback template
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)


class OpenAIEngine(BaseInferenceEngine):
    """OpenAI API inference engine.
    
    Supports both standard OpenAI models and structured output via the beta
    completions API.
    
    Attributes:
        client: OpenAI API client instance.
    """

    def __init__(self, 
                 model_name: str, 
                 judge_mode: bool = False):
        """Initialize OpenAI inference engine.
        
        Args:
            model_name: OpenAI model identifier (e.g., 'gpt-4', 'gpt-3.5-turbo').
            judge_mode: If True, use judge-specific API credentials. Defaults to False.
        """
        super().__init__(model_name)
        from openai import OpenAI

        if judge_mode:
            api_key = self.settings.llm_judge_openai_api_key_str
        else:
            api_key = self.settings.openai_api_key_str
       
        self.client = OpenAI(api_key=api_key)

    def generate_unstructured(self, 
                              messages: List[Dict[str, str]], 
                              **kwargs: Any) -> str:
        """Generate unstructured text using OpenAI's chat completions API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Generation parameters such as temperature, max_tokens, top_p.
            
        Returns:
            Generated text response.
            
        Raises:
            openai.OpenAIError: If the API request fails.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self._convert_kwargs(kwargs)
        )
        return response.choices[0].message.content

    def generate_structured(self, 
                            messages: List[Dict[str, str]], 
                            schema: Any, 
                            **kwargs: Any) -> str:
        """Generate structured JSON using OpenAI's beta parse API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            schema: Pydantic model defining the expected JSON structure.
            **kwargs: Generation parameters such as temperature, max_tokens, top_p.
            
        Returns:
            Generated JSON response as a string.
            
        Raises:
            openai.OpenAIError: If the API request fails.
        """
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=messages,
            response_format=schema,
            **self._convert_kwargs(kwargs)
        )
        return response.choices[0].message.content

    def _convert_kwargs(self, 
                        kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generation parameters to OpenAI API format.
        
        Maps common parameter names to OpenAI-specific names and filters
        out unsupported parameters.
        
        Args:
            kwargs: Dictionary of generation parameters.
            
        Returns:
            Dictionary with OpenAI-compatible parameter names.
        """
        api_kwargs = {}
        if "max_tokens" in kwargs or "max_new_tokens" in kwargs:
            api_kwargs["max_tokens"] = kwargs.get("max_tokens", kwargs.get("max_new_tokens"))
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]
        return api_kwargs


class APIEngine(BaseInferenceEngine):
    """Generic OpenAI-compatible API inference engine.
    
    Supports multiple backend types including vLLM, TGI (Text Generation
    Inference), and Ollama. Each backend may have different capabilities
    for structured generation.
    
    Attributes:
        backend_type: The type of backend ('vllm', 'tgi', or 'ollama').
        client: OpenAI-compatible API client instance.
    """

    def __init__(self, 
                 model_name: str, 
                 backend_type: str, 
                 judge_mode: bool = False) -> None:
        """Initialize generic API inference engine.
        
        Args:
            model_name: Model identifier recognized by the backend.
            backend_type: Type of backend ('vllm', 'tgi', or 'ollama').
            judge_mode: If True, use judge-specific endpoint and credentials.
                Defaults to False.
        """
        super().__init__(model_name)
        self.backend_type = backend_type
        from openai import OpenAI

        if judge_mode:
            base_url = self.settings.llm_judge_endpoint_url
            api_key = self.settings.llm_judge_openai_api_key_str
        else:
            base_url = self.settings.endpoint_url
            api_key = self.settings.openai_api_key_str
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY"
        )

    def generate_unstructured(self, 
                              messages: List[Dict[str, str]], 
                              **kwargs: Any) -> str:
        """Generate unstructured text using OpenAI-compatible API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Generation parameters such as temperature, max_tokens, top_p.
            
        Returns:
            Generated text response.
            
        Raises:
            openai.OpenAIError: If the API request fails.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self._convert_kwargs(kwargs)
        )
        return response.choices[0].message.content

    def generate_structured(self, 
                            messages: List[Dict[str, str]], 
                            schema: Any, 
                            **kwargs) -> str:
        """Generate structured JSON with backend-specific methods.
        
        Different backends support structured generation differently:
        - vLLM: Uses guided_json in extra_body
        - Ollama: Uses response_format in beta API
        - TGI: Falls back to unstructured generation
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            schema: Pydantic model or JSON schema defining output structure.
            **kwargs: Generation parameters such as temperature, max_tokens, top_p.
            
        Returns:
            Generated JSON response as a string, or error JSON if generation fails.
        """
        try:
            if self.backend_type == "vllm":
                # vLLM supports guided_json
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    extra_body={"guided_json": schema.model_json_schema()},
                    **self._convert_kwargs(kwargs)
                )
            elif self.backend_type == "ollama":
                # Ollama supports response_format
                response = self.client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=messages,
                    response_format=schema,
                    **self._convert_kwargs(kwargs)
                )
            else:
                # TGI and others - fallback to unstructured
                return self.generate_unstructured(messages, **kwargs)

            return response.choices[0].message.content

        except Exception as e:
            print(f"Structured generation failed: {e}")
            return '{"error": "Structured generation failed"}'

    def _convert_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generation parameters to OpenAI API format.
        
        Maps common parameter names to API-compatible names and filters
        out unsupported parameters.
        
        Args:
            kwargs: Dictionary of generation parameters.
            
        Returns:
            Dictionary with API-compatible parameter names.
        """
        api_kwargs = {}
        if "max_tokens" in kwargs or "max_new_tokens" in kwargs:
            api_kwargs["max_tokens"] = kwargs.get("max_tokens", kwargs.get("max_new_tokens"))
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]
        return api_kwargs


class TransformersEngine(BaseInferenceEngine):
    """Local transformers inference engine with optional structured generation.
    
    Uses HuggingFace transformers for model loading and inference. Supports
    quantization via bitsandbytes and structured generation via outlines.
    
    Attributes:
        use_structured: Whether structured generation with outlines is enabled.
        tokenizer: HuggingFace tokenizer instance.
        model: HuggingFace model instance.
        structured_model: Outlines-wrapped model for structured generation (optional).
    """

    def __init__(self, 
                 model_name: str, 
                 **kwargs: Any):
        """Initialize local transformers inference engine.
        
        Args:
            model_name: HuggingFace model identifier or local path.
            **kwargs: Additional configuration options:
                - use_structured_generation (bool): Enable outlines for structured
                  output. Defaults to True.
                - quantized (bool): Load model in 4-bit quantization. Defaults to False.
        """
        super().__init__(model_name)
        self.use_structured = kwargs.get("use_structured_generation", True)
        self._load_model(**kwargs)

    def _load_model(self, **kwargs: Any):
        """Load the model and tokenizer with optional quantization.
        
        Args:
            **kwargs: Configuration options including:
                - quantized (bool): If True, load model with 4-bit quantization
                  using bitsandbytes.
                  
        Raises:
            ImportError: If required libraries (transformers, torch) are not installed.
            OSError: If the model cannot be loaded from HuggingFace or local path.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantized = kwargs.get("quantized", False)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if quantized:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, quantization_config=config, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, device_map="auto", torch_dtype=torch.float16
            )

        # For structured generation, wrap with outlines
        if self.use_structured:
            import outlines
            self.structured_model = outlines.models.transformers(self.model_name)

    def generate_unstructured(self, 
                              messages: List[Dict[str, str]], 
                              **kwargs: Any) -> str:
        """Generate unstructured text using local transformers model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Generation parameters such as temperature, max_tokens, top_p.
            
        Returns:
            Generated text response.
        """
        prompt = self._apply_chat_template_with_tokenizer(messages)
        return self._generate_with_transformers(prompt, **kwargs)

    def generate_structured(self, 
                            messages: List[Dict[str, str]], 
                            schema: Dict[str, Any], 
                            **kwargs: Any) -> str:
        """Generate structured JSON using outlines library.
        
        Attempts to generate JSON conforming to the provided schema using
        outlines. Falls back to unstructured generation if outlines fails.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            schema: JSON schema or Pydantic model defining output structure.
            **kwargs: Generation parameters such as temperature, max_tokens, top_p.
            
        Returns:
            Generated JSON response as a string, or fallback to unstructured
            generation if structured generation fails.
        """
        try:
            import outlines
            prompt = self._apply_chat_template_with_tokenizer(messages)

            generator = outlines.generate.json(self.structured_model, schema)
            result = generator(prompt)

            if isinstance(result, dict):
                return json.dumps(result)
            elif hasattr(result, "model_dump"):
                return json.dumps(result.model_dump())
            else:
                return json.dumps({"error": "Invalid structured output"})

        except Exception as e:
            print(f"Structured generation failed: {e}")
            return self.generate_unstructured(messages, **kwargs)

    def _apply_chat_template_with_tokenizer(self, 
                                            messages: List[Dict[str, str]]) -> str:
        """Apply chat template using the model's tokenizer if available.
        
        Tries to use the model's native chat template via the tokenizer's
        apply_chat_template method. Falls back to the base class template
        if unavailable.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            
        Returns:
            Formatted prompt string with chat template applied.
        """
        """Apply chat template using model's tokenizer."""
        try:
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            pass
        return self._apply_chat_template(messages)

    def _generate_with_transformers(self, prompt: str, **kwargs) -> str:
        """Generate text using raw transformers model.generate() method.
        
        Args:
            prompt: The prompt string to generate from.
            **kwargs: Generation parameters including:
                - max_tokens or max_new_tokens (int): Maximum tokens to generate.
                - temperature (float): Sampling temperature.
                - top_p (float): Nucleus sampling parameter.
                
        Returns:
            Generated text with special tokens removed.
        """
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", kwargs.get("max_new_tokens", 100)),
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if "temperature" in kwargs:
            gen_kwargs["temperature"] = kwargs["temperature"]
            gen_kwargs["do_sample"] = True
        if "top_p" in kwargs:
            gen_kwargs["top_p"] = kwargs["top_p"]
            gen_kwargs["do_sample"] = True

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


class ModelEngine:
    """Unified interface for all inference engines.
    
    This is the main entry point for model inference. It automatically selects
    the appropriate backend engine based on configuration settings and provides
    a consistent API regardless of the underlying engine.
    
    Attributes:
        model_name: The model identifier.
        judge_mode: Whether operating in judge/evaluation mode.
        settings: Application settings.
        engine: The underlying inference engine instance.
    """

    def __init__(self, 
                 model_name: str, 
                 judge_mode: bool = False,  
                 **kwargs: Any):
        """Initialize the unified model engine.
        
        Args:
            model_name: Model identifier or path.
            judge_mode: If True, use judge-specific configuration. Defaults to False.
            **kwargs: Additional engine-specific configuration options passed to
                the underlying engine.
        """
        self.model_name = model_name
        self.judge_mode = judge_mode
        self.settings = get_settings()
        self.engine = self._create_engine(**kwargs)

    def _create_engine(self, **kwargs: Any) -> BaseInferenceEngine:
        """Create the appropriate inference engine based on configuration.
        
        Determines which engine to use based on the endpoint_type setting:
        - 'openai': OpenAI API
        - 'vllm', 'tgi', 'ollama': OpenAI-compatible APIs
        - Default: Local transformers
        
        Args:
            **kwargs: Engine-specific configuration options.
            
        Returns:
            An instantiated inference engine.
        """
        # Check settings for backend type
        if self.judge_mode:
            backend_type = self.settings.llm_judge_endpoint_type

        else:
            backend_type = self.settings.endpoint_type

        print(f"backend type: {backend_type}")

        if backend_type == "openai":
            return OpenAIEngine(self.model_name, judge_mode=self.judge_mode)
        elif backend_type in ["vllm", "tgi", "ollama"]:
            return APIEngine(self.model_name, backend_type, judge_mode=self.judge_mode)
        else:
            # Default to transformers
            return TransformersEngine(self.model_name, **kwargs)


    def generate_unstructured(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate unstructured text response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Generation parameters (temperature, max_tokens, etc.).
            
        Returns:
            Generated text as a string.
        """
        return self.engine.generate_unstructured(messages, **kwargs)

    def generate_structured(self, messages: List[Dict[str, str]], schema: Dict[str, Any], **kwargs) -> str:
        """Generate structured JSON response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            schema: JSON schema or Pydantic model defining output structure.
            **kwargs: Generation parameters (temperature, max_tokens, etc.).
            
        Returns:
            Parsed JSON as a dictionary if valid JSON, otherwise the raw string.
        """
        response = self.engine.generate_structured(messages, schema, **kwargs)
        try:
            return json.loads(response)

        except Exception:
            return response


def create_model_engine(model_name: str, 
                        judge_mode: bool = False, 
                        **kwargs: Any) -> ModelEngine:
    """Convenience function to create a model engine.
    
    Args:
        model_name: Model identifier or path.
        judge_mode: If True, use judge-specific configuration. Defaults to False.
        **kwargs: Additional engine-specific configuration options.
        
    Returns:
        A configured ModelEngine instance.
        
    Example:
        >>> engine = create_model_engine('gpt-4')
        >>> messages = [
        ...     {'role': 'system', 'content': 'You are helpful.'},
        ...     {'role': 'user', 'content': 'Hello!'}
        ... ]
        >>> response = engine.generate_unstructured(messages)
    """
    return ModelEngine(model_name, judge_mode=judge_mode, **kwargs)