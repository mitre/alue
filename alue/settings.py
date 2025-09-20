from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
from haystack.utils import Secret
import os
from dotenv import load_dotenv
from typing import Literal

load_dotenv()

EndpointType = Literal["tgi", "vllm", "ollama", "openai", "azure", "none"]


class Settings(BaseSettings):
    ENDPOINT_TYPE: EndpointType = "none"
    ENDPOINT_URL: str | None = None


    OPENAI_API_KEY: str = ""
    OPENAI_COMPATIBLE_LIB: str | None = None
    MODEL_NAME: str | None = None

    AZURE_OPENAI_ENDPOINT: AnyHttpUrl | None = None
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_MODEL_NAME: str | None = None
    
    EMBEDDINGS_API_KEY: str = ""
    EMBEDDINGS_URL: AnyHttpUrl | None = None


    class Config:
        env_prefix = "ALUE_"
        case_sensitive = False

settings = Settings()


def secret_from_env(*env_keys: str):
    for key in env_keys:
        value = os.getenv(key)
        if value:
            return Secret.from_token(value)
        
    raise ValueError(
        f"Misssing required API key env variable. Tried: {', '.join(env_keys)}"
    )