from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface  import HuggingFaceEmbeddings
from app.ingestion.splitter import split_docs
from app.core.logger import logger
from typing import Optional, Union

class HybridRetriever:
    def __init__(self, persist_dir: str, docs: list = None):
        self.persist_dir = persist_dir
        self.docs = docs
        self.vector_retriever = None
        self.bm25_retriever = None
        self._init_components()

    def _init_components(self):
        """Initialise les composants avec gestion robuste des erreurs"""
        # 1. Configuration des embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 2. Initialisation FAISS
        try:
            self.vector_store = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.vector_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": 3, "score_threshold": 0.4}
            )
        except Exception as e:
            logger.error(f"Erreur FAISS: {e}", exc_info=True)
            raise RuntimeError("Échec du chargement du vector store")

        # 3. Initialisation BM25 (si docs fournis)
        if self.docs:
            try:
                splits = split_docs(self.docs)
                self.bm25_retriever = BM25Retriever.from_documents(splits)
                self.bm25_retriever.k = 3
            except Exception as e:
                logger.warning(f"BM25 désactivé - Erreur: {e}")
                self.bm25_retriever = None

    def get_retriever(self, query: Optional[str] = None) -> Union[EnsembleRetriever, FAISS.as_retriever]:
        """Retourne le retriever adapté au type de requête"""
        try:
            # Adaptation dynamique pour les questions complexes
            if query and len(query.split()) > 6:
                self.vector_retriever.search_kwargs["k"] = 6
                if self.bm25_retriever:
                    self.bm25_retriever.k = 6

            if self.bm25_retriever:
                return EnsembleRetriever(
                    retrievers=[self.vector_retriever, self.bm25_retriever],
                    weights=[0.6, 0.4]
                )
            return self.vector_retriever
        except Exception as e:
            logger.error(f"Erreur de retrieval: {e}")
            raise

def get_retriever(persist_dir: str, docs: list = None, query: str = None):
    """Factory simplifiée avec gestion des erreurs"""
    try:
        hybrid = HybridRetriever(persist_dir, docs)
        return hybrid.get_retriever(query)
    except Exception as e:
        logger.critical(f"Échec de l'initialisation du retriever: {e}")
        raise