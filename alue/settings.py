
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from pydantic import Field, SecretStr

class ALUESettings(BaseSettings):
    """ALUE settings with validation and security best practices."""

    endpoint_type: Optional[str] = Field(None, env="ALUE_ENDPOINT_TYPE")
    endpoint_url: Optional[str] = Field(None, env="ALUE_ENDPOINT_URL") 
    model_name: Optional[str] = Field(None, env="ALUE_MODEL_NAME")
    openai_api_key: Optional[SecretStr] = Field(None, env=["ALUE_OPENAI_API_KEY", "OPENAI_API_KEY"])

    class Config:
        env_file = "~/.env"
        case_sensitive = False

    @property
    def openai_api_key_str(self) -> Optional[str]:
        """Get the actual API key string."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

def get_settings() -> ALUESettings:
    """Get settings."""
    return ALUESettings()