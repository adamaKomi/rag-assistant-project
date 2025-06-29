"""
Gestionnaire de vectorstore avancé avec ChromaDB.
Implémente une recherche hybride pour des performances optimales.
"""
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ..models.schemas import Product, DocumentMetadata, MatchingStage
from ..core.logger import get_logger
from ..core.exceptions import VectorStoreError
from ..core.config import settings

logger = get_logger(__name__)


class HybridRetriever:
    """
    Système de recherche hybride combinant :
    1. Recherche vectorielle (sémantique)
    2. Recherche BM25 (lexicale)
    3. Recherche par métadonnées
    """
    
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedding_model = None
        self.bm25_index = None
        self.documents = []
        self.products = []
        
        self._initialize_chroma()
        self._initialize_embedding_model()
    
    def _initialize_chroma(self):
        """Initialise ChromaDB"""
        try:
            # Configuration ChromaDB
            chroma_settings = Settings(
                persist_directory=settings.vectorstore.persist_directory,
                anonymized_telemetry=False
            )
            
            self.chroma_client = chromadb.PersistentClient(
                path=settings.vectorstore.persist_directory,
                settings=chroma_settings
            )
            
            # Création ou récupération de la collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=settings.vectorstore.collection_name,
                metadata={"description": "E-commerce RAG products and documents"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {settings.vectorstore.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise VectorStoreError(
                "Failed to initialize vector store",
                error_code="CHROMA_INIT_ERROR",
                context={"error": str(e)}
            )
    
    def _initialize_embedding_model(self):
        """Initialise le modèle d'embeddings"""
        try:
            self.embedding_model = SentenceTransformer(
                settings.vectorstore.embedding_model,
                cache_folder="embedding_cache/"
            )
            logger.info(f"Embedding model loaded: {settings.vectorstore.embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise VectorStoreError(
                "Failed to load embedding model",
                error_code="EMBEDDING_MODEL_ERROR",
                context={"model": settings.vectorstore.embedding_model, "error": str(e)}
            )
    
    def add_products(self, products: List[Product]) -> None:
        """Ajoute des produits au vectorstore"""
        if not products:
            return
        
        start_time = datetime.now()
        
        try:
            # Préparation des documents pour ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for product in products:
                # Création du texte searchable
                searchable_text = self._create_searchable_text(product)
                documents.append(searchable_text)
                
                # Métadonnées pour filtrage
                metadata = {
                    "product_id": product.id,
                    "product_name": product.name,
                    "brand_name": product.brand.name,
                    "brand_normalized": product.brand.normalized_name,
                    "reference": product.reference.reference,
                    "reference_normalized": product.reference.normalized_reference,
                    "source_document": product.source_document or "",
                    "extraction_confidence": product.extraction_confidence,
                    "created_at": product.created_at.isoformat(),
                    "category": product.category or "unknown",
                    "price": product.price or 0.0,
                    "currency": product.currency or "EUR",
                    "characteristics_count": len(product.characteristics)
                }
                metadatas.append(metadata)
                
                # ID unique
                doc_id = product.id or str(uuid.uuid4())
                ids.append(doc_id)
            
            # Ajout à ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Mise à jour de l'index BM25
            self._update_bm25_index()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.log_performance(
                "add_products_to_vectorstore",
                processing_time,
                products_count=len(products)
            )
            
        except Exception as e:
            logger.error(f"Failed to add products to vectorstore: {str(e)}")
            raise VectorStoreError(
                "Failed to add products to vector store",
                error_code="ADD_PRODUCTS_ERROR",
                context={"products_count": len(products), "error": str(e)}
            )
    
    def search_products(
        self,
        query: str,
        stage: MatchingStage,
        max_results: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Product, float]]:
        """
        Recherche de produits avec approche hybride.
        Adapte la stratégie selon l'étape de matching.
        """
        start_time = datetime.now()
        
        try:
            if stage == MatchingStage.BRAND_MATCHING:
                results = self._search_by_brand(query, max_results, similarity_threshold, filters)
            elif stage == MatchingStage.REFERENCE_MATCHING:
                results = self._search_by_reference(query, max_results, similarity_threshold, filters)
            elif stage == MatchingStage.CHARACTERISTIC_MATCHING:
                results = self._search_by_characteristics(query, max_results, similarity_threshold, filters)
            else:
                # Recherche générale hybride
                results = self._hybrid_search(query, max_results, similarity_threshold, filters)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.log_product_matching(
                stage.value,
                len(results),
                np.mean([score for _, score in results]) if results else 0.0,
                query=query,
                processing_time=processing_time
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search products: {str(e)}")
            raise VectorStoreError(
                "Failed to search products",
                error_code="SEARCH_ERROR",
                context={"query": query, "stage": stage.value, "error": str(e)}
            )
    
    def _search_by_brand(
        self,
        query: str,
        max_results: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Product, float]]:
        """Recherche spécialisée pour les marques (Étape 1)"""
        # Création des filtres pour les marques
        where_clause = {"$or": [
            {"brand_name": {"$contains": query}},
            {"brand_normalized": {"$contains": query.upper()}},
        ]}
        
        if filters:
            where_clause.update(filters)
        
        # Recherche dans ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=max_results,
            where=where_clause
        )
        
        return self._process_chroma_results(results, similarity_threshold)
    
    def _search_by_reference(
        self,
        query: str,
        max_results: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Product, float]]:
        """Recherche spécialisée pour les références (Étape 2)"""
        # Normalisation de la query
        normalized_query = query.upper().replace('-', '').replace(' ', '')
        
        # Filtres pour les références
        where_clause = {"$or": [
            {"reference": {"$contains": query}},
            {"reference_normalized": {"$contains": normalized_query}},
        ]}
        
        if filters:
            where_clause.update(filters)
        
        # Recherche dans ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=max_results,
            where=where_clause
        )
        
        return self._process_chroma_results(results, similarity_threshold)
    
    def _search_by_characteristics(
        self,
        query: str,
        max_results: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Product, float]]:
        """Recherche par caractéristiques (Étape 3) - 80% de similarité"""
        # Pour cette étape, on utilise la recherche sémantique pure
        return self._semantic_search(query, max_results, 0.8, filters)  # 80% comme demandé
    
    def _hybrid_search(
        self,
        query: str,
        max_results: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Product, float]]:
        """Recherche hybride combinant vectorielle et lexicale"""
        # 1. Recherche sémantique
        semantic_results = self._semantic_search(query, max_results, similarity_threshold, filters)
        
        # 2. Recherche BM25 (si index disponible)
        bm25_results = self._bm25_search(query, max_results, filters) if self.bm25_index else []
        
        # 3. Fusion des résultats avec pondération
        combined_results = self._combine_search_results(semantic_results, bm25_results)
        
        return combined_results[:max_results]
    
    def _semantic_search(
        self,
        query: str,
        max_results: int,
        similarity_threshold: float,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Product, float]]:
        """Recherche sémantique pure avec ChromaDB"""
        results = self.collection.query(
            query_texts=[query],
            n_results=max_results,
            where=filters or {}
        )
        
        return self._process_chroma_results(results, similarity_threshold)
    
    def _bm25_search(
        self,
        query: str,
        max_results: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Tuple[Product, float]]:
        """Recherche BM25 lexicale"""
        if not self.bm25_index or not self.documents:
            return []
        
        # Tokenisation de la query
        query_tokens = query.lower().split()
        
        # Recherche BM25
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Tri et sélection des meilleurs résultats
        top_indices = np.argsort(scores)[::-1][:max_results]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Seuil minimum
                product = self.products[idx]
                # Application des filtres
                if self._apply_filters(product, filters):
                    # Normalisation du score BM25
                    normalized_score = min(scores[idx] / 10.0, 1.0)
                    results.append((product, normalized_score))
        
        return results
    
    def _combine_search_results(
        self,
        semantic_results: List[Tuple[Product, float]],
        bm25_results: List[Tuple[Product, float]]
    ) -> List[Tuple[Product, float]]:
        """Combine les résultats sémantiques et BM25"""
        # Dictionnaire pour éviter les doublons
        combined = {}
        
        # Pondération : 70% sémantique, 30% BM25
        semantic_weight = 0.7
        bm25_weight = 0.3
        
        # Ajout des résultats sémantiques
        for product, score in semantic_results:
            product_id = product.id
            combined[product_id] = (product, score * semantic_weight)
        
        # Ajout/fusion des résultats BM25
        for product, score in bm25_results:
            product_id = product.id
            if product_id in combined:
                # Fusion des scores
                existing_product, existing_score = combined[product_id]
                new_score = existing_score + (score * bm25_weight)
                combined[product_id] = (existing_product, new_score)
            else:
                combined[product_id] = (product, score * bm25_weight)
        
        # Tri par score combiné
        results = list(combined.values())
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _process_chroma_results(
        self,
        results: Dict[str, Any],
        similarity_threshold: float
    ) -> List[Tuple[Product, float]]:
        """Traite les résultats de ChromaDB"""
        processed_results = []
        
        if not results['ids'] or not results['ids'][0]:
            return processed_results
        
        ids = results['ids'][0]
        distances = results['distances'][0] if results['distances'] else [0.0] * len(ids)
        metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)
        documents = results['documents'][0] if results['documents'] else [''] * len(ids)
        
        for i, doc_id in enumerate(ids):
            # Conversion distance -> similarité
            similarity = 1.0 - distances[i] if distances else 0.5
            
            if similarity >= similarity_threshold:
                # Reconstruction du produit à partir des métadonnées
                metadata = metadatas[i]
                product = self._reconstruct_product_from_metadata(metadata, documents[i])
                processed_results.append((product, similarity))
        
        return processed_results
    
    def _reconstruct_product_from_metadata(self, metadata: Dict[str, Any], document: str) -> Product:
        """Reconstruit un objet Product à partir des métadonnées"""
        # Note: Dans une implémentation complète, on ferait un appel à la base de données
        # Ici, on reconstruit un produit minimal pour la démonstration
        from ..models.schemas import ProductBrand, ProductReference
        
        brand = ProductBrand(
            name=metadata.get('brand_name', ''),
            normalized_name=metadata.get('brand_normalized', ''),
            aliases=[],
            confidence=1.0
        )
        
        reference = ProductReference(
            reference=metadata.get('reference', ''),
            normalized_reference=metadata.get('reference_normalized', ''),
            confidence=1.0
        )
        
        return Product(
            id=metadata.get('product_id'),
            name=metadata.get('product_name', ''),
            brand=brand,
            reference=reference,
            characteristics=[],
            category=metadata.get('category'),
            price=metadata.get('price'),
            currency=metadata.get('currency', 'EUR'),
            source_document=metadata.get('source_document'),
            extraction_confidence=metadata.get('extraction_confidence', 1.0)
        )
    
    def _create_searchable_text(self, product: Product) -> str:
        """Crée un texte searchable pour un produit"""
        parts = [
            product.name,
            product.brand.name,
            product.reference.reference,
        ]
        
        if product.description:
            parts.append(product.description)
        
        if product.category:
            parts.append(product.category)
        
        # Ajout des caractéristiques
        for char in product.characteristics:
            char_text = f"{char.name}: {char.value}"
            if char.unit:
                char_text += f" {char.unit}"
            parts.append(char_text)
        
        return ' | '.join(filter(None, parts))
    
    def _update_bm25_index(self):
        """Met à jour l'index BM25"""
        try:
            # Récupération de tous les documents
            all_results = self.collection.get()
            
            if all_results['documents']:
                # Tokenisation des documents
                tokenized_docs = [doc.lower().split() for doc in all_results['documents']]
                
                # Création de l'index BM25
                self.bm25_index = BM25Okapi(tokenized_docs)
                self.documents = all_results['documents']
                
                # Reconstruction des produits (simplifié)
                self.products = [
                    self._reconstruct_product_from_metadata(meta, doc)
                    for meta, doc in zip(all_results['metadatas'], all_results['documents'])
                ]
                
                logger.info(f"BM25 index updated with {len(tokenized_docs)} documents")
            
        except Exception as e:
            logger.warning(f"Failed to update BM25 index: {str(e)}")
    
    def _apply_filters(self, product: Product, filters: Optional[Dict[str, Any]]) -> bool:
        """Applique les filtres à un produit"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key == "brand_name" and product.brand.name != value:
                return False
            elif key == "category" and product.category != value:
                return False
            elif key == "min_price" and (product.price or 0) < value:
                return False
            elif key == "max_price" and (product.price or 0) > value:
                return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du vectorstore"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": settings.vectorstore.collection_name,
                "embedding_model": settings.vectorstore.embedding_model,
                "bm25_enabled": self.bm25_index is not None
            }
        except Exception as e:
            logger.error(f"Failed to get vectorstore stats: {str(e)}")
            return {"error": str(e)}


# Instance globale
hybrid_retriever = HybridRetriever()
