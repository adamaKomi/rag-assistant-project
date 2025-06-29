from app.core.config import VECTORSTORE_DIR, PDF_DIR
from app.vectorstore.retriever import get_retriever
from app.chains.rag_chain import get_rag_chain
from app.core.logger import logger
from app.ingestion.pdf_loader import load_pdfs
from typing import Dict, Any, Optional
from functools import lru_cache
import re

@lru_cache(maxsize=1)
def load_documents_cached():
    """Cache les documents pour BM25 (évite les rechargements inutiles)"""
    return load_pdfs(PDF_DIR)

def format_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Post-traitement standardisé des réponses"""
    response = {
        "result": result.get("result", "Je n'ai pas d'information pertinente sur ce sujet."),
        "sources": [],
        "status": "success"
    }

    # Extraction et formatage des sources
    for i, doc in enumerate(result.get("source_documents", [])[:3]):  # Limite à 3 sources
        source_info = {
            "id": i+1,
            "content": re.sub(r'\s+', ' ', doc.page_content)[:150] + ('...' if len(doc.page_content) > 150 else ''),
            "metadata": doc.metadata.get('source', 'Document interne')
        }
        response["sources"].append(source_info)
    
    return response

def process_query(query: str, conversation_history: Optional[list] = None) -> Dict[str, Any]:
    """
    Traite une requête RAG avec gestion de conversation
    Args:
        query: Question utilisateur
        conversation_history: Historique sous forme [("question", "réponse"), ...]
    Returns:
        {
            "result": "Réponse générée",
            "sources": [{"id": 1, "content": "...", "metadata": "source.pdf"}],
            "status": "success"|"error"
        }
    """
    try:
        pdf_docs = load_documents_cached()
        retriever = get_retriever(VECTORSTORE_DIR, docs=pdf_docs, query=query)
        
        memory = bool(conversation_history)
        qa = get_rag_chain(retriever, memory=memory)
        
        # Modification clé ici - utilisez directement le dictionnaire avec la clé 'query'
        inputs = {"query": query}  # Au lieu de {"question": query}
        
        if conversation_history:
            inputs["chat_history"] = "\n".join(
                f"Q: {q}\nR: {r}" for q, r in conversation_history[-3:]
            )
        
        result = qa.invoke(inputs)
        return format_response(result)
        
    except Exception as e:
        logger.error(f"Erreur RAG: {e}", exc_info=True)
        return {
            "result": "Désolé, une erreur est survenue. Reformulez votre question.",
            "sources": [],
            "status": "error"
        }