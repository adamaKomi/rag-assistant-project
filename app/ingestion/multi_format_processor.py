"""
Gestionnaire multi-format pour l'ingestion de documents.
Supporte PDF, Excel, JSON, XML et sources web.
"""
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import hashlib
from datetime import datetime
import mimetypes

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..models.schemas import DocumentMetadata, DocumentType, InputDocument, Product
from ..extraction.product_extractor import ProductExtractor
from ..core.logger import get_logger
from ..core.exceptions import DocumentProcessingError
from ..core.config import settings

logger = get_logger(__name__)


class MultiFormatProcessor:
    """
    Processeur multi-format pour l'ingestion de documents.
    Centralise le traitement de tous les types de fichiers supportés.
    """
    
    def __init__(self):
        self.product_extractor = ProductExtractor()
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.json': self._process_json,
            '.xml': self._process_xml,
            '.txt': self._process_text,
        }
    
    def process_document(self, file_path: str) -> InputDocument:
        """
        Traite un document et extrait les produits.
        Point d'entrée principal pour tout type de document.
        """
        start_time = datetime.now()
        
        try:
            # Validation du fichier
            self._validate_file(file_path)
            
            # Détection du type de document
            doc_type = self._detect_document_type(file_path)
            
            # Création des métadonnées
            metadata = self._create_metadata(file_path, doc_type)
            
            # Traitement selon le type
            processor = self.supported_formats.get(Path(file_path).suffix.lower())
            if not processor:
                raise DocumentProcessingError(
                    f"Unsupported file format: {Path(file_path).suffix}",
                    error_code="UNSUPPORTED_FORMAT",
                    context={"file_path": file_path}
                )
            
            # Extraction du contenu et des produits
            content, products = processor(file_path)
            
            # Création du document d'entrée
            input_document = InputDocument(
                content=content,
                metadata=metadata,
                extracted_products=products,
                processing_status="completed"
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            metadata.processing_time = processing_time
            
            logger.log_performance(
                "document_processing",
                processing_time,
                file_path=file_path,
                document_type=doc_type.value,
                products_extracted=len(products)
            )
            
            return input_document
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}", file_path=file_path)
            
            # Retourner un document avec erreur
            metadata = self._create_metadata(file_path, DocumentType.PDF)  # Fallback
            return InputDocument(
                content="",
                metadata=metadata,
                extracted_products=[],
                processing_status="error",
                error_message=str(e)
            )
    
    def process_web_url(self, url: str) -> InputDocument:
        """Traite une URL web"""
        start_time = datetime.now()
        
        try:
            # Téléchargement du contenu
            response = requests.get(
                url, 
                headers={'User-Agent': settings.api.user_agent},
                timeout=30
            )
            response.raise_for_status()
            
            # Parsing HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extraction du texte
            content = self._extract_text_from_html(soup)
            
            # Extraction des produits
            products = self.product_extractor.extract_from_text(content, url)
            
            # Métadonnées
            metadata = DocumentMetadata(
                file_path=url,
                file_name=url.split('/')[-1] or 'webpage',
                document_type=DocumentType.WEB,
                file_size=len(response.content),
                checksum=hashlib.md5(response.content).hexdigest(),
                processing_time=(datetime.now() - start_time).total_seconds(),
                language=self._detect_language(content),
                encoding=response.encoding
            )
            
            return InputDocument(
                content=content,
                metadata=metadata,
                extracted_products=products,
                processing_status="completed"
            )
            
        except Exception as e:
            logger.error(f"Error processing web URL: {str(e)}", url=url)
            raise DocumentProcessingError(
                f"Failed to process web URL: {url}",
                error_code="WEB_PROCESSING_ERROR",
                context={"url": url, "error": str(e)}
            )
    
    def _validate_file(self, file_path: str):
        """Valide qu'un fichier peut être traité"""
        path = Path(file_path)
        
        if not path.exists():
            raise DocumentProcessingError(
                f"File not found: {file_path}",
                error_code="FILE_NOT_FOUND"
            )
        
        if not path.is_file():
            raise DocumentProcessingError(
                f"Path is not a file: {file_path}",
                error_code="INVALID_FILE_PATH"
            )
        
        # Vérification de la taille
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.ingestion.max_file_size_mb:
            raise DocumentProcessingError(
                f"File too large: {file_size_mb:.1f}MB (max: {settings.ingestion.max_file_size_mb}MB)",
                error_code="FILE_TOO_LARGE"
            )
        
        # Vérification du format
        if path.suffix.lower() not in settings.ingestion.supported_formats:
            raise DocumentProcessingError(
                f"Unsupported format: {path.suffix}",
                error_code="UNSUPPORTED_FORMAT"
            )
    
    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Détecte le type de document"""
        suffix = Path(file_path).suffix.lower()
        
        type_mapping = {
            '.pdf': DocumentType.PDF,
            '.xlsx': DocumentType.EXCEL,
            '.xls': DocumentType.EXCEL,
            '.json': DocumentType.JSON,
            '.xml': DocumentType.XML,
            '.txt': DocumentType.TEXT
        }
        
        return type_mapping.get(suffix, DocumentType.TEXT)
    
    def _create_metadata(self, file_path: str, doc_type: DocumentType) -> DocumentMetadata:
        """Crée les métadonnées d'un document"""
        path = Path(file_path)
        
        # Calcul du checksum
        with open(file_path, 'rb') as f:
            content = f.read()
            checksum = hashlib.md5(content).hexdigest()
        
        return DocumentMetadata(
            file_path=str(path.absolute()),
            file_name=path.name,
            document_type=doc_type,
            file_size=len(content),
            checksum=checksum,
            encoding=self._detect_encoding(file_path)
        )
    
    def _process_pdf(self, file_path: str) -> tuple[str, List[Product]]:
        """Traite un fichier PDF"""
        products = self.product_extractor.extract_from_pdf(file_path)
        
        # Extraction du texte pour le contenu
        content = self.product_extractor._extract_text_from_pdf(file_path)
        
        return content, products
    
    def _process_excel(self, file_path: str) -> tuple[str, List[Product]]:
        """Traite un fichier Excel"""
        try:
            # Lecture de toutes les feuilles
            excel_file = pd.ExcelFile(file_path)
            all_content = []
            all_products = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Conversion en texte
                sheet_content = self._dataframe_to_text(df, sheet_name)
                all_content.append(sheet_content)
                
                # Extraction des produits
                products = self.product_extractor.extract_from_excel_dataframe(df, file_path, sheet_name)
                all_products.extend(products)
            
            return '\n\n'.join(all_content), all_products
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process Excel file: {file_path}",
                error_code="EXCEL_PROCESSING_ERROR",
                context={"file_path": file_path, "error": str(e)}
            )
    
    def _process_json(self, file_path: str) -> tuple[str, List[Product]]:
        """Traite un fichier JSON"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Conversion en texte lisible
            content = self._json_to_text(data)
            
            # Extraction des produits
            products = self.product_extractor.extract_from_json(data, file_path)
            
            return content, products
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process JSON file: {file_path}",
                error_code="JSON_PROCESSING_ERROR",
                context={"file_path": file_path, "error": str(e)}
            )
    
    def _process_xml(self, file_path: str) -> tuple[str, List[Product]]:
        """Traite un fichier XML"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Conversion en texte
            content = self._xml_to_text(root)
            
            # Extraction des produits
            products = self.product_extractor.extract_from_xml(root, file_path)
            
            return content, products
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process XML file: {file_path}",
                error_code="XML_PROCESSING_ERROR",
                context={"file_path": file_path, "error": str(e)}
            )
    
    def _process_text(self, file_path: str) -> tuple[str, List[Product]]:
        """Traite un fichier texte"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extraction des produits
            products = self.product_extractor.extract_from_text(content, file_path)
            
            return content, products
            
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process text file: {file_path}",
                error_code="TEXT_PROCESSING_ERROR",
                context={"file_path": file_path, "error": str(e)}
            )
    
    def _extract_text_from_html(self, soup: BeautifulSoup) -> str:
        """Extrait le texte d'un document HTML"""
        # Suppression des scripts et styles
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extraction du texte principal
        text = soup.get_text()
        
        # Nettoyage
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _dataframe_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """Convertit un DataFrame en texte structuré"""
        content = [f"=== FEUILLE: {sheet_name} ===\n"]
        
        # En-têtes
        headers = ' | '.join(str(col) for col in df.columns)
        content.append(f"Colonnes: {headers}\n")
        
        # Données
        for idx, row in df.iterrows():
            row_text = ' | '.join(str(val) for val in row.values if pd.notna(val))
            content.append(f"Ligne {idx + 1}: {row_text}")
        
        return '\n'.join(content)
    
    def _json_to_text(self, data: Any, indent: int = 0) -> str:
        """Convertit des données JSON en texte lisible"""
        lines = []
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                lines.append(f"{prefix}[{i}]:")
                lines.append(self._json_to_text(item, indent + 1))
        
        else:
            lines.append(f"{prefix}{data}")
        
        return '\n'.join(lines)
    
    def _xml_to_text(self, element: ET.Element, indent: int = 0) -> str:
        """Convertit un élément XML en texte"""
        lines = []
        prefix = "  " * indent
        
        # Tag et attributs
        tag_info = element.tag
        if element.attrib:
            attrs = ' '.join(f"{k}={v}" for k, v in element.attrib.items())
            tag_info += f" ({attrs})"
        
        lines.append(f"{prefix}{tag_info}")
        
        # Texte de l'élément
        if element.text and element.text.strip():
            lines.append(f"{prefix}  {element.text.strip()}")
        
        # Éléments enfants
        for child in element:
            lines.append(self._xml_to_text(child, indent + 1))
        
        return '\n'.join(lines)
    
    def _detect_encoding(self, file_path: str) -> Optional[str]:
        """Détecte l'encodage d'un fichier"""
        try:
            import chardet
            
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Premier chunk
                result = chardet.detect(raw_data)
                return result.get('encoding')
        except ImportError:
            return 'utf-8'  # Fallback
        except Exception:
            return None
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Détecte la langue d'un texte"""
        try:
            from langdetect import detect
            return detect(text)
        except ImportError:
            # Heuristique simple pour français/anglais
            french_words = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'est', 'avec']
            english_words = ['the', 'and', 'or', 'is', 'with', 'for', 'to', 'of', 'in', 'on']
            
            text_lower = text.lower()
            french_count = sum(1 for word in french_words if word in text_lower)
            english_count = sum(1 for word in english_words if word in text_lower)
            
            if french_count > english_count:
                return 'fr'
            elif english_count > french_count:
                return 'en'
            else:
                return 'unknown'
        except Exception:
            return None
