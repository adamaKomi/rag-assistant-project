"""
Configuration centralisée avec Pydantic pour validation et typage fort.
Cette approche garantit la robustesse et facilite la maintenance.
"""
import os
from typing import Optional, List
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import validator
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig(BaseSettings):
    """Configuration base de données - PostgreSQL pour métadonnées"""
    host: str = "localhost"
    port: int = 5432
    name: str = "rag_assistant"
    user: str = "postgres"
    password: str = "password"
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class VectorStoreConfig(BaseSettings):
    """Configuration ChromaDB - plus évolutif que FAISS"""
    persist_directory: str = "vectorstore/chroma"
    collection_name: str = "rag_documents"
    embedding_model: str = "BAAI/bge-large-en-v1.5"  # Meilleur que bge-small
    chunk_size: int = 1000
    chunk_overlap: int = 200


class LLMConfig(BaseSettings):
    """Configuration Mistral via Ollama"""
    model_name: str = "mistral:7b-instruct"
    temperature: float = 0.1  # Précision maximale
    max_tokens: int = 2048
    top_p: float = 0.9
    ollama_base_url: str = "http://localhost:11434"
    
    # Pour le fine-tuning futur
    fine_tuned_model_path: Optional[str] = None
    use_fine_tuned: bool = False


class IngestionConfig(BaseSettings):
    """Configuration ingestion multi-formats"""
    pdf_dir: str = "app/data/pdfs/"
    excel_dir: str = "app/data/excel/"
    web_urls: List[str] = []
    supported_formats: List[str] = [".pdf", ".xlsx", ".json", ".xml", ".txt"]
    max_file_size_mb: int = 100
    
    @validator('pdf_dir', 'excel_dir')
    def create_directories(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class ProductMatchingConfig(BaseSettings):
    """Configuration pour la logique de correspondance produits"""
    brand_similarity_threshold: float = 0.8
    reference_similarity_threshold: float = 0.85
    characteristic_similarity_threshold: float = 0.8  # 80% comme demandé
    max_results_per_stage: int = 10


class APIConfig(BaseSettings):
    """Configuration API FastAPI"""
    title: str = "RAG Assistant E-commerce"
    version: str = "2.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    user_agent: str = "RAGAssistant/2.0"


class CacheConfig(BaseSettings):
    """Configuration Redis pour cache"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    cache_ttl: int = 3600  # 1 heure


class Settings(BaseSettings):
    """Configuration globale centralisée"""
    database: DatabaseConfig = DatabaseConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    llm: LLMConfig = LLMConfig()
    ingestion: IngestionConfig = IngestionConfig()
    product_matching: ProductMatchingConfig = ProductMatchingConfig()
    api: APIConfig = APIConfig()
    cache: CacheConfig = CacheConfig()
    
    # Environnement
    environment: str = "development"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


# Instance globale
settings = Settings()
