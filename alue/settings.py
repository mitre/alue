from typing import Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class ALUESettings(BaseSettings):
    """ALUE settings with validation and security best practices."""

    endpoint_type: Optional[str] = Field(None, alias="ALUE_ENDPOINT_TYPE")
    endpoint_url: Optional[str] = Field(None, alias="ALUE_ENDPOINT_URL") 
    openai_api_key: Optional[SecretStr] = Field(None, alias="ALUE_OPENAI_API_KEY")
    hf_token: Optional[SecretStr] = Field(None, alias="HF_TOKEN")

    llm_judge_endpoint_type: Optional[str] = Field(None, alias="ALUE_LLM_JUDGE_ENDPOINT_TYPE")
    llm_judge_endpoint_url: Optional[str] = Field(None, alias="ALUE_LLM_JUDGE_ENDPOINT_URL") 
    llm_judge_openai_api_key: Optional[SecretStr] = Field(None, alias="ALUE_LLM_JUDGE_OPENAI_API_KEY")

    # Embedding provider to use, choices: ["openai", "ollama", "hf", "local", "openai-compatible"]
    embedding_endpoint_type: Optional[str] = Field("local", pattern=r'openai|ollama|hf|local|openai-compatible', alias="EMBEDDING_ENDPOINT_TYPE")
    # URL for Ollama or OpenAI-compatible endpoints
    embedding_endpoint_url: Optional[str] = Field(None, alias="EMBEDDING_ENDPOINT_URL")
    embedding_api_key: Optional[SecretStr] = Field(None, alias="EMBEDDING_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def openai_api_key_str(self) -> Optional[str]:
        """Get the actual API key string."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None
    
    @property  
    def llm_judge_openai_api_key_str(self) -> Optional[str]:
        return self.llm_judge_openai_api_key.get_secret_value() if self.llm_judge_openai_api_key else None
    
    @property
    def embedding_api_key_str(self) -> Optional[str]:
        return self.embedding_api_key.get_secret_value() if self.embedding_api_key else None

    @property
    def hf_token_str(self) -> Optional[str]:
        """Get the HuggingFace token string."""
        return self.hf_token.get_secret_value() if self.hf_token else None

def get_settings() -> ALUESettings:
    """Get settings."""
    return ALUESettings()