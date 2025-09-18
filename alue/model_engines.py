# General Imports
import os
from typing import Any

import outlines
import requests
import torch
from config import MODELS

# LLM Specific Imports
from openai import OpenAI
from transformers import BitsAndBytesConfig

outlines.disable_cache()


# from templates.llm_judge.prompts import *


class TransformersEngine:
    """
    A class for managing and loading transformer models with optional quantization.

    This class handles model path resolution (local vs. Hugging Face), quantization
    configuration, and model loading using the outlines library.

    Args:
        model_type (str): The type/name of the model as defined in MODELS config
        quantized (bool): Whether to apply quantization to the model
        **quantization_kwargs: Additional keyword arguments for BitsAndBytesConfig

    Attributes
    ----------
        model_type (str): The model type identifier
        model_name_or_path (str): Resolved path to the model (local or HF hub)
        quantized (bool): Whether quantization is enabled
        quantization_kwargs (Dict[str, Any]): Custom quantization parameters
        bnb_config (Optional[BitsAndBytesConfig]): Quantization configuration object

    Raises
    ------
        ValueError: If model_type is not registered in MODELS config
        FileNotFoundError: If local model path is specified but doesn't exist
        RuntimeError: If model loading fails
    """

    def __init__(
        self, model_type: str, quantized: bool, **quantization_kwargs: Any
    ) -> None:
        self.model_type = model_type
        self.quantized = quantized
        self.quantization_kwargs = quantization_kwargs

        # Resolve model path
        self.model_name_or_path = self._resolve_model_path(model_type)

        # Setup quantization config
        self.bnb_config = self._get_bnb_config()

    def _resolve_model_path(self, model_type: str) -> str:
        """
        Resolve the model path from the MODELS configuration.

        Args:
            model_type (str): The model type to resolve

        Returns
        -------
            str: The resolved model path (local or Hugging Face hub)

        Raises
        ------
            ValueError: If model_type is not in MODELS config
        """
        try:
            if model_type not in MODELS:
                raise ValueError(
                    f"Model type '{model_type}' is not registered in MODELS config"
                )

            model_config = MODELS[model_type]

            # Validate model config structure
            if "local_path" not in model_config:
                raise ValueError(
                    f"Invalid model config for '{model_type}'. Must contain 'local_path'"
                )

            local_path = model_config["local_path"]

            # Check if local path is specified and exists
            if local_path and local_path.strip():
                if os.path.exists(local_path):
                    print(f"Loading {model_type} from local path: {local_path}")
                    return local_path
                else:
                    print(
                        f"Warning: Local path '{local_path}' does not exist. Falling back to Hugging Face."
                    )

        except Exception as e:
            raise ValueError(
                f"Failed to resolve model path for '{model_type}': {str(e)}"
            ) from e

    def _get_bnb_config(self) -> BitsAndBytesConfig | None:
        """
        Create BitsAndBytesConfig based on quantization settings.

        Returns
        -------
            Optional[BitsAndBytesConfig]: Quantization config if quantization is enabled,
                                        None otherwise

        Raises
        ------
            ValueError: If invalid quantization parameters are provided
        """
        if not self.quantized:
            return None

        try:
            # Default 4-bit quantization config
            default_config = {
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            }

            # Override defaults with any provided kwargs
            config_params = {**default_config, **self.quantization_kwargs}

            # Validate that we don't have conflicting quantization modes
            if config_params.get("load_in_4bit") and config_params.get("load_in_8bit"):
                raise ValueError(
                    "Cannot enable both 4-bit and 8-bit quantization simultaneously"
                )

            return BitsAndBytesConfig(**config_params)

        except Exception as e:
            raise ValueError(f"Failed to create quantization config: {str(e)}") from e

    def load_model(self) -> Any:
        """
        Load the transformer model with the configured settings.

        Returns
        -------
            Any: The loaded model object from outlines.models.transformers

        Raises
        ------
            RuntimeError: If model loading fails
        """
        try:
            model_kwargs = {"device_map": "auto"}

            if self.quantized and self.bnb_config is not None:
                model_kwargs["quantization_config"] = self.bnb_config
                print(f"Loading quantized model with config: {self.bnb_config}")
            else:
                print("Loading model without quantization")

            print(f"Loading model from: {self.model_name_or_path}")
            model = outlines.models.transformers(
                self.model_name_or_path, model_kwargs=model_kwargs
            )

            print(f"Successfully loaded model: {self.model_type}")
            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.model_type}': {str(e)}"
            ) from e

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns
        -------
            Dict[str, Any]: Dictionary containing model configuration details
        """
        return {
            "model_type": self.model_type,
            "model_path": self.model_name_or_path,
            "quantized": self.quantized,
            "quantization_config": self.bnb_config.__dict__
            if self.bnb_config
            else None,
            "quantization_kwargs": self.quantization_kwargs,
        }

    def __repr__(self) -> str:
        """String representation of the TransformersJudge instance."""
        return (
            f"TransformersJudge(model_type='{self.model_type}', "
            f"quantized={self.quantized}, "
            f"model_path='{self.model_name_or_path}')"
        )


class vLLMofflineEngine:
    def __init__(
        self,
        model_type: str,
        quantized: bool,
        dtype: torch.dtype = torch.float16,
    ):
        # Store settings
        self.quantized = quantized
        self.dtype = dtype
        self.model_type = model_type

        # Validate GPU compatibility if quantized
        if quantized:
            self._validate_gpu_compatibility(dtype=dtype)

        # Resolve model path
        self.model_name_or_path = self._resolve_model_path(model_type)

    def _validate_gpu_compatibility(self, dtype: torch.dtype | None) -> None:
        """
        Validate GPU compatibility for quantization.

        Args:
            dtype: The data type to use

        Raises
        ------
            ValueError: If GPU is incompatible with requested settings
        """
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Quantization requires a GPU.")

        # Get GPU compute capability
        device_capability = torch.cuda.get_device_capability()
        compute_capability = device_capability[0] + device_capability[1] / 10

        # Check if trying to use bfloat16 on incompatible GPU
        if dtype == torch.bfloat16 and compute_capability < 8.0:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                f"Bfloat16 is only supported on GPUs with compute capability of at least 8.0. "
                f"Your {gpu_name} GPU has compute capability {compute_capability}. "
                f"Tesla T4 GPUs are not supported for bfloat16 quantization. "
                f"Please use dtype=torch.float16 instead."
            )

        # Restrict to only float16 and bfloat16
        if dtype is not None and dtype not in [torch.float16, torch.bfloat16]:
            raise ValueError(
                f"Only torch.float16 and torch.bfloat16 are supported for quantized models. "
                f"Got: {dtype}"
            )

    def _resolve_model_path(self, model_type: str) -> str:
        """
        Resolve the model path from the MODELS configuration.

        Args:
            model_type (str): The model type to resolve

        Returns
        -------
            str: The resolved model path (local or Hugging Face hub)

        Raises
        ------
            ValueError: If model_type is not in MODELS config
        """
        try:
            if model_type not in MODELS:
                raise ValueError(
                    f"Model type '{model_type}' is not registered in MODELS config"
                )

            model_config = MODELS[model_type]

            # Validate model config structure
            if "local_path" not in model_config:
                raise ValueError(
                    f"Invalid model config for '{model_type}'. Must contain 'local_path"
                )

            local_path = model_config["local_path"]
            # model_name = model_config["model_name"]

            # Check if local path is specified and exists
            if local_path and local_path.strip():
                if os.path.exists(local_path):
                    print(f"Loading {model_type} from local path: {local_path}")
                    return local_path
                else:
                    raise ValueError(f"Local path '{local_path}' does not exist.")

        except Exception as e:
            raise ValueError(
                f"Failed to resolve model path for '{model_type}': {str(e)}"
            ) from e

    def load_model(self) -> Any:
        """
        Load the vLLM model with the configured settings.

        Returns
        -------
            Any: The loaded LLM model object from vLLM

        Raises
        ------
            RuntimeError: If model loading fails
        """
        try:
            from vllm import LLM

            model_kwargs = {
                "model": self.model_name_or_path,
                "trust_remote_code": True,
                "max_model_len": 8192,
            }

            if self.quantized:
                model_kwargs["dtype"] = self.dtype
                model_kwargs["quantization"] = "bitsandbytes"
                print(f"Loading quantized model with dtype: {self.dtype}")
            else:
                print("Loading model without quantization")

            print(f"Loading model from: {self.model_name_or_path}")
            model = LLM(**model_kwargs)

            print(f"Successfully loaded model: {self.model_type}")
            return model

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{self.model_type}': {str(e)}"
            ) from e


class vLLMonlineEngine:
    def __init__(
        self,
        model_type: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
    ):
        self.model_type = model_type
        self.base_url = base_url
        self.api_key = api_key

    def load_model(self) -> Any:
        """
        Connect to the hosted vLLM server and return an OpenAI client.

        Returns
        -------
            OpenAI: Configured OpenAI client for the hosted server

        Raises
        ------
            RuntimeError: If connection to server fails
            ImportError: If openai package is not installed
        """
        try:
            # Check if server is accessible
            if not self.is_server_running():
                raise RuntimeError(f"vLLM server is not accessible at {self.base_url}")

            client = OpenAI(base_url=self.base_url, api_key=self.api_key or "EMPTY")

            print(f"Successfully connected to vLLM server at {self.base_url}")
            print(f"Model: {self.model_type}")

            return client

        except Exception as e:
            raise RuntimeError(f"Failed to connect to vLLM server: {str(e)}") from e

    def is_server_running(self) -> bool:
        """
        Check if the vLLM server is running and accessible.

        Returns
        -------
            bool: True if server is running and responding
        """
        try:
            # Parse base_url to get health endpoint
            if self.base_url.endswith("/v1"):
                health_url = self.base_url.replace("/v1", "/health")
            else:
                health_url = f"{self.base_url.rstrip('/')}/health"

            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns
        -------
            Dict[str, Any]: Dictionary containing model configuration details
        """
        return {
            "model_type": self.model_type,
            "model_path": self.model_name_or_path,
            "quantized": self.quantized,
            "dtype": self.dtype if self.quantized else None,
            "server_url": self.base_url,
            "api_key_set": self.api_key is not None,
            "server_running": self.is_server_running(),
        }

    def __repr__(self) -> str:
        """String representation of the vLLMonline instance."""
        return (
            f"vLLMonline(model_type='{self.model_type}', "
            f"server_url='{self.base_url}', "
            f"connected={self.is_server_running()})"
        )


class OllamaEngine:
    def __init__(
        self,
        model_type: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):  # Ollama requires api_key but it's unused
        self.model_type = model_type
        self.base_url = base_url
        self.api_key = api_key  # Required by OpenAI client but unused by Ollama

        # Resolve model name
        self.model_name = self._resolve_model_name(model_type)

    def _resolve_model_name(self, model_type: str) -> str:
        """
        Resolve the model name from the MODELS configuration.

        Args:
            model_type (str): The model type to resolve

        Returns
        -------
            str: The resolved model name for Ollama

        Raises
        ------
            ValueError: If model_type is not in MODELS config
        """
        try:
            if model_type not in MODELS:
                raise ValueError(
                    f"Model type '{model_type}' is not registered in MODELS config"
                )

            model_config = MODELS[model_type]

            # For Ollama, we need the model name (e.g., 'llama3.2', 'mistral', etc.)
            if "ollama_model" in model_config:
                model_name = model_config["ollama_model"]
            elif "model_name" in model_config:
                model_name = model_config["model_name"]
            else:
                raise ValueError(
                    f"Invalid model config for '{model_type}'. Must contain 'ollama_model' or 'model_name'"
                )

            if not model_name or not model_name.strip():
                raise ValueError(f"No valid model name specified for '{model_type}'")

            print(f"Using Ollama model: {model_name}")
            return model_name

        except Exception as e:
            raise ValueError(
                f"Failed to resolve model name for '{model_type}': {str(e)}"
            ) from e

    def load_model(self) -> Any:
        """
        Connect to the hosted Ollama server and return an OpenAI client.

        Returns
        -------
            OpenAI: Configured OpenAI client for the hosted Ollama server

        Raises
        ------
            RuntimeError: If connection to server fails
            ImportError: If openai package is not installed
        """
        try:
            # Check if server is accessible
            if not self.is_server_running():
                raise RuntimeError(
                    f"Ollama server is not accessible at {self.base_url}"
                )

            # Check if model is available
            if not self.is_model_available():
                raise RuntimeError(
                    f"Model '{self.model_name}' is not available on Ollama server. "
                    f"Please pull the model first: ollama pull {self.model_name}"
                )

            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,  # Required but unused by Ollama
            )

            print(f"Successfully connected to Ollama server at {self.base_url}")
            print(f"Model: {self.model_name}")

            return client

        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama server: {str(e)}") from e

    def is_server_running(self) -> bool:
        """
        Check if the Ollama server is running and accessible.

        Returns
        -------
            bool: True if server is running and responding
        """
        try:
            # Ollama uses /api/tags endpoint to check server health
            if self.base_url.endswith("/v1"):
                tags_url = self.base_url.replace("/v1", "/api/tags")
            else:
                tags_url = f"{self.base_url.rstrip('/')}/api/tags"

            response = requests.get(tags_url, timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def is_model_available(self) -> bool:
        """
        Check if the specified model is available on the Ollama server.

        Returns
        -------
            bool: True if model is available
        """
        try:
            if self.base_url.endswith("/v1"):
                tags_url = self.base_url.replace("/v1", "/api/tags")
            else:
                tags_url = f"{self.base_url.rstrip('/')}/api/tags"

            response = requests.get(tags_url, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]
                # Check if model name matches (accounting for tags like :latest)
                return any(self.model_name in model for model in available_models)
            return False
        except requests.exceptions.RequestException:
            return False

    def list_available_models(self) -> list:
        """
        Get list of available models on the Ollama server.

        Returns
        -------
            list: List of available model names
        """
        try:
            if self.base_url.endswith("/v1"):
                tags_url = self.base_url.replace("/v1", "/api/tags")
            else:
                tags_url = f"{self.base_url.rstrip('/')}/api/tags"

            response = requests.get(tags_url, timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
            return []
        except requests.exceptions.RequestException:
            return []

    def pull_model(self) -> bool:
        """
        Pull the model to the Ollama server.
        Note: This requires direct access to Ollama API, not OpenAI compatible endpoint.

        Returns
        -------
            bool: True if model was successfully pulled
        """
        try:
            if self.base_url.endswith("/v1"):
                pull_url = self.base_url.replace("/v1", "/api/pull")
            else:
                pull_url = f"{self.base_url.rstrip('/')}/api/pull"

            payload = {"name": self.model_name}
            response = requests.post(
                pull_url, json=payload, timeout=300
            )  # 5 min timeout for model pull

            if response.status_code == 200:
                print(f"Successfully pulled model: {self.model_name}")
                return True
            else:
                print(f"Failed to pull model: {self.model_name}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error pulling model: {e}")
            return False

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model configuration.

        Returns
        -------
            Dict[str, Any]: Dictionary containing model configuration details
        """
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "server_url": self.base_url,
            "server_running": self.is_server_running(),
            "model_available": self.is_model_available(),
            "available_models": self.list_available_models(),
        }

    def __repr__(self) -> str:
        """String representation of the OllamaJudge instance."""
        return (
            f"OllamaJudge(model_type='{self.model_type}', "
            f"model_name='{self.model_name}', "
            f"server_url='{self.base_url}', "
            f"connected={self.is_server_running()})"
        )
