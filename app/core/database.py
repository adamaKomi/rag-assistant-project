"""
Gestionnaire de base de données avec SQLAlchemy pour la persistance.
Gère les métadonnées des documents et l'historique des traitements.
"""
from sqlalchemy import create_engine, Column, String, DateTime, Float, Integer, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime

from ..core.config import settings
from ..core.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class DocumentRecord(Base):
    """Table pour stocker les métadonnées des documents"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    checksum = Column(String, nullable=False, unique=True)
    processed_at = Column(DateTime, default=datetime.now)
    processing_time = Column(Float)
    page_count = Column(Integer)
    language = Column(String)
    encoding = Column(String)
    metadata = Column(JSON)
    
    def __repr__(self):
        return f"<DocumentRecord(file_name='{self.file_name}', type='{self.document_type}')>"


class ProductRecord(Base):
    """Table pour stocker les produits extraits"""
    __tablename__ = "products"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    brand_name = Column(String, nullable=False)
    brand_normalized = Column(String, nullable=False)
    reference = Column(String, nullable=False)
    reference_normalized = Column(String, nullable=False)
    characteristics = Column(JSON)  # Liste des caractéristiques
    description = Column(Text)
    category = Column(String)
    price = Column(Float)
    currency = Column(String, default="EUR")
    availability = Column(Boolean)
    source_document_id = Column(UUID(as_uuid=True))
    extraction_confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    def __repr__(self):
        return f"<ProductRecord(name='{self.name}', brand='{self.brand_name}')>"


class QueryRecord(Base):
    """Table pour stocker l'historique des requêtes"""
    __tablename__ = "queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    confidence = Column(Float)
    processing_time = Column(Float, nullable=False)
    sources_used = Column(JSON)
    matching_results = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<QueryRecord(query='{self.query_text[:50]}...')>"


class DatabaseManager:
    """Gestionnaire de base de données"""
    
    def __init__(self):
        self.engine = create_engine(
            settings.database.url,
            echo=settings.environment == "development",
            pool_pre_ping=True,
            pool_recycle=3600
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Crée toutes les tables"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    @contextmanager
    def get_session(self) -> Session:
        """Context manager pour les sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    def save_document(self, document_data: Dict[str, Any]) -> DocumentRecord:
        """Sauvegarde un document en base"""
        with self.get_session() as session:
            document = DocumentRecord(**document_data)
            session.add(document)
            session.flush()
            logger.info(f"Document saved: {document.file_name}")
            return document
    
    def save_product(self, product_data: Dict[str, Any]) -> ProductRecord:
        """Sauvegarde un produit en base"""
        with self.get_session() as session:
            product = ProductRecord(**product_data)
            session.add(product)
            session.flush()
            logger.info(f"Product saved: {product.name}")
            return product
    
    def save_query(self, query_data: Dict[str, Any]) -> QueryRecord:
        """Sauvegarde une requête en base"""
        with self.get_session() as session:
            query = QueryRecord(**query_data)
            session.add(query)
            session.flush()
            logger.info(f"Query saved: {query.query_text[:50]}...")
            return query
    
    def get_document_by_checksum(self, checksum: str) -> Optional[DocumentRecord]:
        """Récupère un document par son checksum"""
        with self.get_session() as session:
            return session.query(DocumentRecord).filter(
                DocumentRecord.checksum == checksum
            ).first()
    
    def search_products(
        self, 
        brand: Optional[str] = None,
        reference: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[ProductRecord]:
        """Recherche de produits avec filtres"""
        with self.get_session() as session:
            query = session.query(ProductRecord)
            
            if brand:
                query = query.filter(ProductRecord.brand_normalized.ilike(f"%{brand}%"))
            if reference:
                query = query.filter(ProductRecord.reference_normalized.ilike(f"%{reference}%"))
            if category:
                query = query.filter(ProductRecord.category.ilike(f"%{category}%"))
            
            return query.limit(limit).all()
    
    def get_recent_queries(self, limit: int = 50) -> List[QueryRecord]:
        """Récupère les requêtes récentes"""
        with self.get_session() as session:
            return session.query(QueryRecord).order_by(
                QueryRecord.created_at.desc()
            ).limit(limit).all()


# Instance globale
db_manager = DatabaseManager()
