from typing import Optional
from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class ALUESettings(BaseSettings):
    """ALUE settings with validation and security best practices."""

    endpoint_type: Optional[str] = Field(None, alias="ALUE_ENDPOINT_TYPE")
    endpoint_url: Optional[str] = Field(None, alias="ALUE_ENDPOINT_URL") 
    openai_api_key: Optional[SecretStr] = Field(None, alias="ALUE_OPENAI_API_KEY")

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

def get_settings() -> ALUESettings:
    """Get settings."""
    return ALUESettings()