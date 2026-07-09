"""Application configuration.

This module centralizes all configuration values used across the
FinTech Compliance Copilot project.

Configuration values are loaded from the `.env` file using
Pydantic Settings.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Project Paths
    # ------------------------------------------------------------------

    project_root: Path = Path(__file__).resolve().parent.parent

    raw_data_dir: Path = project_root / "data" / "raw"
    processed_data_dir: Path = project_root / "data" / "processed"
    chunks_dir: Path = project_root / "data" / "chunks"

    logs_dir: Path = project_root / "logs"

    vectorstore_dir: Path = project_root / "vectorstore"

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset_name: str = "ObliQA_MultiPassage_train.json"

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    chunk_size: int = Field(default=300, gt=0)

    chunk_overlap: int = Field(default=50, ge=0)

    # ------------------------------------------------------------------
    # Embedding Model
    # ------------------------------------------------------------------

    embedding_model_name: str = "BAAI/bge-m3"

    embedding_device: str = "cpu"

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------

    chroma_collection_name: str = "fintech_compliance"

    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------

    google_api_key: str

    gemini_model_name: str = "gemini-2.0-flash"

    temperature: float = 0.0

    max_output_tokens: int = 1024

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: str = "INFO"

    # ------------------------------------------------------------------
    # FastAPI
    # ------------------------------------------------------------------

    api_host: str = "0.0.0.0"

    api_port: int = 8000

    # ------------------------------------------------------------------
    # Helper Properties
    # ------------------------------------------------------------------

    @property
    def dataset_path(self) -> Path:
        """Return the complete path of the ObliQA dataset."""
        return self.raw_data_dir / self.dataset_name

    @property
    def chroma_db_path(self) -> Path:
        """Return the ChromaDB persistence directory."""
        return self.vectorstore_dir / "chroma_db"

    def create_directories(self) -> None:
        """Create required directories if they do not exist."""

        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.chunks_dir,
            self.logs_dir,
            self.chroma_db_path,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    settings = Settings()
    settings.create_directories()
    return settings