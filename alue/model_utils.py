"""Clean inference engines for ALUE framework."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from alue.config import MODELS
from alue.settings import get_settings


class BaseInferenceEngine(ABC):
    """Base class for all inference engines."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.settings = get_settings()

    @abstractmethod
    def generate_unstructured(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate unstructured text response."""
        pass

    @abstractmethod
    def generate_structured(self, messages: List[Dict[str, str]], schema: Dict[str, Any], **kwargs) -> str:
        """Generate structured JSON response."""
        pass

    def _get_model_name(self) -> str:
        """Get model name for API calls."""
        if self.model_type in MODELS:
            config = MODELS[self.model_type]
            return (
                config.get("api_model_name") or
                config.get("model_name") or
                self.model_type
            )
        return self.settings.model_name or self.model_type

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to prompt string."""
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
    """OpenAI API inference engine."""

    def __init__(self, model_type: str):
        super().__init__(model_type)
        from openai import OpenAI
       
        self.client = OpenAI(api_key=self.settings.openai_api_key_str)

    def generate_unstructured(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self._get_model_name(),
            messages=messages,
            **self._convert_kwargs(kwargs)
        )
        return response.choices[0].message.content

    def generate_structured(self, messages: List[Dict[str, str]], schema, **kwargs) -> str:
        response = self.client.beta.chat.completions.parse(
            model=self._get_model_name(),
            messages=messages,
            response_format=schema,
            **self._convert_kwargs(kwargs)
        )
        return response.choices[0].message.content

    def _convert_kwargs(self, kwargs):
        """Convert to OpenAI API format."""
        api_kwargs = {}
        if "max_tokens" in kwargs or "max_new_tokens" in kwargs:
            api_kwargs["max_tokens"] = kwargs.get("max_tokens", kwargs.get("max_new_tokens"))
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]
        return api_kwargs


class APIEngine(BaseInferenceEngine):
    """Generic OpenAI-compatible API engine (TGI, vLLM online, Ollama)."""

    def __init__(self, model_type: str, backend_type: str):
        super().__init__(model_type)
        self.backend_type = backend_type
        from openai import OpenAI
        self.client = OpenAI(
            base_url=self.settings.endpoint_url,
            api_key=self.settings.openai_api_key_str or "EMPTY"
        )

    def generate_unstructured(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self._get_model_name(),
            messages=messages,
            **self._convert_kwargs(kwargs)
        )
        return response.choices[0].message.content

    def generate_structured(self, messages: List[Dict[str, str]], schema: Dict[str, Any], **kwargs) -> str:
        try:
            if self.backend_type == "vllm":
                # vLLM supports guided_json
                response = self.client.chat.completions.create(
                    model=self._get_model_name(),
                    messages=messages,
                    extra_body={"guided_json": schema},
                    **self._convert_kwargs(kwargs)
                )
            elif self.backend_type == "ollama":
                # Ollama supports response_format
                response = self.client.beta.chat.completions.parse(
                    model=self._get_model_name(),
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

    def _convert_kwargs(self, kwargs):
        """Convert to OpenAI API format."""
        api_kwargs = {}
        if "max_tokens" in kwargs or "max_new_tokens" in kwargs:
            api_kwargs["max_tokens"] = kwargs.get("max_tokens", kwargs.get("max_new_tokens"))
        if "temperature" in kwargs:
            api_kwargs["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            api_kwargs["top_p"] = kwargs["top_p"]
        return api_kwargs


class TransformersEngine(BaseInferenceEngine):
    """Local transformers inference engine."""

    def __init__(self, model_type: str, **kwargs):
        super().__init__(model_type)
        self.use_structured = kwargs.get("use_structured_generation", True)
        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        """Load model and tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = self._get_model_path()
        quantized = kwargs.get("quantized", False)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
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
                model_path, quantization_config=config, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", torch_dtype=torch.float16
            )

        # For structured generation, wrap with outlines
        if self.use_structured:
            import outlines
            self.structured_model = outlines.models.transformers(model_path)

    def generate_unstructured(self, messages: List[Dict[str, str]], **kwargs) -> str:
        prompt = self._apply_chat_template_with_tokenizer(messages)
        return self._generate_with_transformers(prompt, **kwargs)

    def generate_structured(self, messages: List[Dict[str, str]], schema: Dict[str, Any], **kwargs) -> str:
        if not self.use_structured:
            return self.generate_unstructured(messages, **kwargs)

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

    def _apply_chat_template_with_tokenizer(self, messages: List[Dict[str, str]]) -> str:
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
        """Raw transformers generation."""
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

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _get_model_path(self) -> str:
        """Get model path."""
        if self.model_type in MODELS:
            config = MODELS[self.model_type]
            if "local_path" in config and config["local_path"]:
                return config["local_path"]
            if "model_name" in config:
                return config["model_name"]
        return self.model_type


class ModelEngine:
    """Unified interface for all inference engines."""

    def __init__(self, model_type: str, **kwargs):
        self.model_type = model_type
        self.settings = get_settings()
        self.engine = self._create_engine(**kwargs)

    def _create_engine(self, **kwargs) -> BaseInferenceEngine:
        """Create the appropriate engine based on configuration."""
        # Check settings for backend type
        backend_type = (
            kwargs.get("backend_type") or
            self.settings.endpoint_type or
            self._get_config_backend()
        )

        if backend_type == "openai":
            return OpenAIEngine(self.model_type)
        elif backend_type in ["vllm", "tgi", "ollama"]:
            return APIEngine(self.model_type, backend_type)
        else:
            # Default to transformers
            return TransformersEngine(self.model_type, **kwargs)

    def _get_config_backend(self) -> str:
        """Get backend from model config."""
        if self.model_type in MODELS:
            return MODELS[self.model_type].get("backend", "transformers")
        return "transformers"

    def generate_unstructured(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate unstructured response."""
        return self.engine.generate_unstructured(messages, **kwargs)

    def generate_structured(self, messages: List[Dict[str, str]], schema: Dict[str, Any], **kwargs) -> str:
        """Generate structured response."""
        response = self.engine.generate_structured(messages, schema, **kwargs)
        try:
            return json.loads(response)

        except Exception:
            return response

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "engine_type": type(self.engine).__name__,
            "model_name": self.engine._get_model_name(),
        }


def create_model_engine(model_type: str, **kwargs) -> ModelEngine:
    """Convenience function to create a model engine."""
    return ModelEngine(model_type, **kwargs)