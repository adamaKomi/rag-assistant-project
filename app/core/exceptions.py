"""
Gestionnaire d'exceptions personnalisées pour un debugging efficace.
Chaque exception fournit un contexte détaillé pour faciliter la maintenance.
"""
from typing import Optional, Dict, Any


class RAGAssistantException(Exception):
    """Exception de base pour l'application"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        super().__init__(self.message)


class DocumentProcessingError(RAGAssistantException):
    """Erreur lors du traitement de documents"""
    pass


class ProductExtractionError(RAGAssistantException):
    """Erreur lors de l'extraction de produits"""
    pass


class VectorStoreError(RAGAssistantException):
    """Erreur liée au vectorstore"""
    pass


class LLMError(RAGAssistantException):
    """Erreur liée au modèle de langage"""
    pass


class MatchingError(RAGAssistantException):
    """Erreur lors du matching de produits"""
    pass


class ConfigurationError(RAGAssistantException):
    """Erreur de configuration"""
    pass


class ValidationError(RAGAssistantException):
    """Erreur de validation des données"""
    pass
