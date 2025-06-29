"""
Modèles de données Pydantic pour une gestion robuste et typée.
Ces modèles garantissent la cohérence des données dans tout le pipeline.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    """Types de documents supportés"""
    PDF = "pdf"
    EXCEL = "excel" 
    WEB = "web"
    JSON = "json"
    XML = "xml"
    TEXT = "text"


class MatchingStage(str, Enum):
    """Étapes de correspondance"""
    BRAND_MATCHING = "brand_matching"      # Étape 1
    REFERENCE_MATCHING = "reference_matching"  # Étape 2
    CHARACTERISTIC_MATCHING = "characteristic_matching"  # Étape 3


class ProductBrand(BaseModel):
    """Modèle pour les marques de produits"""
    name: str = Field(..., description="Nom de la marque")
    normalized_name: str = Field(..., description="Nom normalisé pour recherche")
    aliases: List[str] = Field(default_factory=list, description="Noms alternatifs")
    confidence: float = Field(ge=0.0, le=1.0, description="Confiance d'extraction")


class ProductReference(BaseModel):
    """Modèle pour les références produits"""
    reference: str = Field(..., description="Référence produit")
    normalized_reference: str = Field(..., description="Référence normalisée")
    format_pattern: Optional[str] = Field(None, description="Pattern de format détecté")
    confidence: float = Field(ge=0.0, le=1.0, description="Confiance d'extraction")


class ProductCharacteristic(BaseModel):
    """Caractéristique d'un produit"""
    name: str = Field(..., description="Nom de la caractéristique")
    value: str = Field(..., description="Valeur de la caractéristique")
    unit: Optional[str] = Field(None, description="Unité de mesure")
    category: Optional[str] = Field(None, description="Catégorie de caractéristique")
    normalized_name: str = Field(..., description="Nom normalisé")
    normalized_value: str = Field(..., description="Valeur normalisée")


class Product(BaseModel):
    """Modèle principal d'un produit"""
    id: Optional[str] = Field(None, description="Identifiant unique")
    name: str = Field(..., description="Nom du produit")
    brand: ProductBrand = Field(..., description="Marque du produit")
    reference: ProductReference = Field(..., description="Référence du produit")
    characteristics: List[ProductCharacteristic] = Field(
        default_factory=list, 
        description="Caractéristiques du produit"
    )
    description: Optional[str] = Field(None, description="Description détaillée")
    category: Optional[str] = Field(None, description="Catégorie produit")
    price: Optional[float] = Field(None, ge=0, description="Prix du produit")
    currency: Optional[str] = Field("EUR", description="Devise")
    availability: Optional[bool] = Field(None, description="Disponibilité")
    
    # Métadonnées
    source_document: Optional[str] = Field(None, description="Document source")
    extraction_confidence: float = Field(ge=0.0, le=1.0, description="Confiance globale")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class DocumentMetadata(BaseModel):
    """Métadonnées d'un document"""
    file_path: str
    file_name: str
    document_type: DocumentType
    file_size: int
    checksum: str
    processed_at: datetime = Field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    encoding: Optional[str] = None


class MatchingResult(BaseModel):
    """Résultat d'une correspondance"""
    stage: MatchingStage
    input_product: Product
    matched_products: List[Product]
    similarity_scores: List[float]
    matching_criteria: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(..., description="Explication du matching")
    processing_time: float


class RAGQuery(BaseModel):
    """Requête RAG structurée"""
    query: str = Field(..., min_length=1, description="Question de l'utilisateur")
    context_filters: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = Field(default=True)
    language: str = Field(default="fr", description="Langue de la réponse")


class RAGResponse(BaseModel):
    """Réponse RAG structurée"""
    answer: str = Field(..., description="Réponse générée")
    sources: List[str] = Field(default_factory=list, description="Sources utilisées")
    confidence: float = Field(ge=0.0, le=1.0, description="Confiance de la réponse")
    processing_time: float = Field(description="Temps de traitement")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Pour le matching de produits
    matching_results: Optional[List[MatchingResult]] = None


class InputDocument(BaseModel):
    """Document d'entrée pour analyse"""
    content: str = Field(..., description="Contenu du document")
    metadata: DocumentMetadata
    extracted_products: List[Product] = Field(default_factory=list)
    processing_status: str = Field(default="pending")
    error_message: Optional[str] = None
