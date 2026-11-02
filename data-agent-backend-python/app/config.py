from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    CHAT_MODEL: str = "qwen-plus"
    EMBEDDING_MODEL: str = "text-embedding-v4"
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/data_agent_tutorial"
    HOST: str = "0.0.0.0"
    PORT: int = 9933

settings = Settings()
