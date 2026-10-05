from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = (
        "postgresql+psycopg://eki_user:eki_password"
        "@localhost:5432/eki_db"
    )

    redis_url: str = "redis://localhost:6379/0"

    app_name: str = "Enterprise Knowledge Intelligence Platform"
    app_version: str = "0.1.0"

    # Embedding configuration
    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dimension: int = 384

    # Reranking configuration
    reranker_model: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_threshold: float = 4.1838

    # Retrieval configuration
    retrieval_similarity_threshold: float = 0.7
    retrieval_top_k: int = 10

    # Local LLM configuration
    llm_model: str = "llama3.2:3b"
    llm_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

