from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import os
from app.core.logger import logger
from typing import List
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

def build_vectorstore(docs: List[Document], persist_dir: str) -> None:
    """
    Construit et sauvegarde un vectorstore avec cache d'embeddings
    Args:
        docs: Liste de documents LangChain
        persist_dir: Chemin de sauvegarde
    """
    try:
        # 1. Configuration des embeddings avec cache
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Cache local pour éviter de recalculer les embeddings
        store = LocalFileStore(os.getenv("EMBEDDING_CACHE_DIR", "./embedding_cache"))
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            embeddings,
            store,
            namespace=embeddings.model_name
        )

        # 2. Création du vectorstore
        os.makedirs(persist_dir, exist_ok=True)
        FAISS.from_documents(
            documents=docs,
            embedding=cached_embeddings,
        ).save_local(persist_dir)
        
        logger.info(f"Vectorstore généré avec succès dans {persist_dir}")
        
    except Exception as e:
        logger.error(f"Échec de la création du vectorstore: {e}", exc_info=True)
        raise RuntimeError("Erreur lors de la construction de l'index")