from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    LLM_BASE_URL: str = "http://localhost:8080/v1"
    LLM_MODEL: str = "phi-4-mini"
    LLM_API_KEY: str = "sk-no-key-required"

settings = Settings()