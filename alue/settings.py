"""Application settings management using Pydantic.

This module provides configuration management for the ALUE framework, including
API credentials, endpoint configurations, and embedding provider settings. All
sensitive values (API keys, tokens) are handled securely using Pydantic's SecretStr.
"""

from typing import Optional, Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class ALUESettings(BaseSettings):
    """ALUE framework settings with validation and security best practices.
    
    This class manages all configuration for the ALUE framework including:
    - Primary LLM endpoint configuration
    - LLM judge endpoint configuration (for evaluation)
    - Embedding provider configuration
    
    All settings can be configured via environment variables or a .env file.
    Sensitive values like API keys are stored as SecretStr to prevent accidental
    exposure in logs or error messages.
    
    Attributes:
        endpoint_type: Type of primary LLM endpoint ('openai', 'vllm', 'tgi', 
            'ollama', or 'transformers').
        endpoint_url: URL for API-based endpoints (vLLM, TGI, Ollama).
        openai_api_key: OpenAI API key (stored securely).
        hf_token: HuggingFace token for model downloads (stored securely).
        llm_judge_endpoint_type: Type of LLM judge endpoint for evaluation.
        llm_judge_endpoint_url: URL for LLM judge endpoint.
        llm_judge_openai_api_key: API key for LLM judge endpoint.
        embedding_endpoint_type: Embedding provider type.
        embedding_endpoint_url: URL for Ollama or OpenAI-compatible embedding endpoints.
        embedding_api_key: API key for embedding provider (stored securely).
        
    Example:
        >>> settings = ALUESettings()
        >>> settings.endpoint_type
        'openai'
        >>> settings.openai_api_key_str  # Access secret value safely
        'sk-...'
    """

    endpoint_type: Optional[Literal["openai", "vllm", "tgi", "ollama", "transformers"]] = Field(
        None, 
        alias="ALUE_ENDPOINT_TYPE",
        description="Type of LLM inference backend"
    )
    endpoint_url: Optional[str] = Field(None, alias="ALUE_ENDPOINT_URL") 
    openai_api_key: Optional[SecretStr] = Field(None, alias="ALUE_OPENAI_API_KEY")
    hf_token: Optional[SecretStr] = Field(None, alias="HF_TOKEN")

    llm_judge_endpoint_type: Optional[Literal["openai", "vllm", "tgi", "ollama", "transformers", None]] = Field(
        None, 
        alias="ALUE_LLM_JUDGE_ENDPOINT_TYPE",
        description="Type of LLM judge endpoint for evaluation"
    )
    llm_judge_endpoint_url: Optional[str] = Field(None, alias="ALUE_LLM_JUDGE_ENDPOINT_URL") 
    llm_judge_openai_api_key: Optional[SecretStr] = Field(None, alias="ALUE_LLM_JUDGE_OPENAI_API_KEY")

    # Embedding provider to use, choices: ["openai", "ollama", "hf", "local", "openai-compatible"]
    embedding_endpoint_type: Optional[Literal["openai", "ollama", "hf", "local", "openai-compatible"]] = Field(
        "local",
        alias="EMBEDDING_ENDPOINT_TYPE",
        description="Embedding provider type"
    )
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
        """Get the OpenAI API key as a plain string.
        
        Returns:
            The API key string, or None if not set.
            
        Note:
            Use this property to access the actual key value since the
            openai_api_key field stores it securely as SecretStr.
        """
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None
    
    @property  
    def llm_judge_openai_api_key_str(self) -> Optional[str]:
        """Get the LLM judge API key as a plain string.
        
        Returns:
            The API key string, or None if not set.
        """
        return self.llm_judge_openai_api_key.get_secret_value() if self.llm_judge_openai_api_key else None
    
    @property
    def embedding_api_key_str(self) -> Optional[str]:
        """Get the embedding API key as a plain string.
        
        Returns:
            The API key string, or None if not set.
        """
        return self.embedding_api_key.get_secret_value() if self.embedding_api_key else None

    @property
    def hf_token_str(self) -> Optional[str]:
        """Get the HuggingFace token as a plain string.
        
        Returns:
            The token string, or None if not set.
            
        Note:
            Required for downloading models from HuggingFace Hub when using
            the transformers backend.
        """
        return self.hf_token.get_secret_value() if self.hf_token else None

def get_settings() -> ALUESettings:
    """Get or create the ALUE settings instance.
    
    This function loads settings from environment variables or a .env file.
    Settings are validated according to the ALUESettings schema.
    
    Returns:
        Configured ALUESettings instance.
        
    Raises:
        ValidationError: If settings don't match the expected schema.
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.endpoint_type)
        'openai'
    """
    return ALUESettings()
