"""
Extracteur de produits intelligent pour analyser les PDF d'entrée.
Implémente la logique métier spécifique aux exigences du client.
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import hashlib
from datetime import datetime

import pymupdf  # PyMuPDF pour extraction PDF
import pandas as pd
from fuzzywuzzy import fuzz, process

from ..models.schemas import Product, ProductBrand, ProductReference, ProductCharacteristic
from ..core.logger import get_logger
from ..core.exceptions import ProductExtractionError
from ..core.config import settings

logger = get_logger(__name__)


@dataclass
class ExtractionPattern:
    """Patterns pour l'extraction d'informations"""
    name: str
    pattern: str
    confidence_modifier: float = 1.0


class ProductExtractor:
    """
    Extracteur de produits intelligent pour les PDF d'entrée.
    Analyse le document pour extraire marques, références et caractéristiques.
    """
    
    def __init__(self):
        self.brand_patterns = self._load_brand_patterns()
        self.reference_patterns = self._load_reference_patterns()
        self.characteristic_patterns = self._load_characteristic_patterns()
        
    def _load_brand_patterns(self) -> List[ExtractionPattern]:
        """Patterns pour détecter les marques"""
        return [
            ExtractionPattern("brand_prefix", r"(?i)(?:marque|brand|fabricant|manufacturer)[\s:]*([A-Z][A-Za-z\s&-]+)", 0.9),
            ExtractionPattern("brand_caps", r"\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b", 0.7),
            ExtractionPattern("brand_mixed", r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", 0.6),
            ExtractionPattern("brand_with_symbols", r"\b([A-Z][A-Za-z0-9&-]+(?:\s+[A-Z][A-Za-z0-9&-]+)*)\b", 0.5)
        ]
    
    def _load_reference_patterns(self) -> List[ExtractionPattern]:
        """Patterns pour détecter les références"""
        return [
            ExtractionPattern("ref_prefix", r"(?i)(?:référence|reference|ref|réf|model|modèle)[\s:]*([A-Z0-9-]+)", 0.95),
            ExtractionPattern("ref_alphanumeric", r"\b([A-Z]{1,4}[0-9]{2,8}[A-Z0-9-]*)\b", 0.8),
            ExtractionPattern("ref_numeric", r"\b([0-9]{4,12})\b", 0.6),
            ExtractionPattern("ref_mixed", r"\b([A-Z0-9]{6,15})\b", 0.7)
        ]
    
    def _load_characteristic_patterns(self) -> List[ExtractionPattern]:
        """Patterns pour détecter les caractéristiques"""
        return [
            ExtractionPattern("dimension", r"(?i)((?:longueur|largeur|hauteur|diameter|diamètre|taille|size))[\s:]*([0-9]+(?:[.,][0-9]+)?)\s*([a-z]{1,3})", 0.9),
            ExtractionPattern("weight", r"(?i)((?:poids|weight|masse))[\s:]*([0-9]+(?:[.,][0-9]+)?)\s*(kg|g|lb)", 0.9),
            ExtractionPattern("material", r"(?i)((?:matériau|material|matière))[\s:]*([A-Za-z\s]+)", 0.8),
            ExtractionPattern("color", r"(?i)((?:couleur|color|coloration))[\s:]*([A-Za-z\s]+)", 0.8),
            ExtractionPattern("voltage", r"(?i)((?:tension|voltage|volt))[\s:]*([0-9]+(?:[.,][0-9]+)?)\s*([vV])", 0.9),
            ExtractionPattern("power", r"(?i)((?:puissance|power|watt))[\s:]*([0-9]+(?:[.,][0-9]+)?)\s*([wW])", 0.9),
            ExtractionPattern("generic_numeric", r"([A-Za-z\s]+)[\s:]*([0-9]+(?:[.,][0-9]+)?)\s*([A-Za-z]+)", 0.6)
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> List[Product]:
        """
        Extrait les produits d'un PDF.
        Méthode principale appelée par le client.
        """
        start_time = datetime.now()
        
        try:
            # Lecture du PDF
            text_content = self._extract_text_from_pdf(pdf_path)
            
            # Préprocessing du texte
            cleaned_text = self._preprocess_text(text_content)
            
            # Extraction des produits
            products = self._extract_products_from_text(cleaned_text, pdf_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.log_performance(
                "pdf_product_extraction",
                processing_time,
                file_path=pdf_path,
                products_found=len(products)
            )
            
            return products
            
        except Exception as e:
            logger.error(f"Error extracting products from PDF: {str(e)}", file_path=pdf_path)
            raise ProductExtractionError(
                f"Failed to extract products from {pdf_path}",
                error_code="PDF_EXTRACTION_ERROR",
                context={"file_path": pdf_path, "error": str(e)}
            )
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrait le texte d'un PDF avec PyMuPDF"""
        try:
            doc = pymupdf.open(pdf_path)
            text_content = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()
                text_content += "\n\n"  # Séparateur de pages
            
            doc.close()
            return text_content
            
        except Exception as e:
            raise ProductExtractionError(
                f"Failed to read PDF file: {pdf_path}",
                error_code="PDF_READ_ERROR",
                context={"file_path": pdf_path, "error": str(e)}
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Nettoie et prépare le texte pour l'extraction"""
        # Suppression des caractères de contrôle
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text)
        
        # Normalisation des caractères spéciaux
        text = text.replace('–', '-').replace('—', '-')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('"', '"').replace('"', '"')
        
        return text.strip()
    
    def _extract_products_from_text(self, text: str, source_path: str) -> List[Product]:
        """Extrait les produits du texte nettoyé"""
        products = []
        
        # Diviser le texte en sections potentielles de produits
        sections = self._split_into_product_sections(text)
        
        for section_idx, section in enumerate(sections):
            try:
                product = self._extract_single_product(section, source_path, section_idx)
                if product:
                    products.append(product)
            except Exception as e:
                logger.warning(f"Failed to extract product from section {section_idx}: {str(e)}")
                continue
        
        return products
    
    def _split_into_product_sections(self, text: str) -> List[str]:
        """Divise le texte en sections de produits potentielles"""
        # Patterns de séparation (titres, numéros, etc.)
        split_patterns = [
            r'\n\s*\d+[\.\)]\s+',  # Numérotation
            r'\n\s*[A-Z][A-Z\s]{10,}\n',  # Titres en majuscules
            r'\n\s*[-=]{3,}\s*\n',  # Séparateurs
            r'\n\s*PRODUIT\s*\d*\s*[:\.]\s*',  # Mentions explicites de produit
        ]
        
        sections = [text]
        
        for pattern in split_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section, flags=re.IGNORECASE)
                new_sections.extend([part.strip() for part in parts if part.strip()])
            sections = new_sections
        
        # Filtrer les sections trop courtes ou trop longues
        filtered_sections = []
        for section in sections:
            if 50 <= len(section) <= 2000:  # Taille raisonnable pour un produit
                filtered_sections.append(section)
        
        return filtered_sections if filtered_sections else [text]
    
    def _extract_single_product(self, text: str, source_path: str, section_idx: int) -> Optional[Product]:
        """Extrait un seul produit d'une section de texte"""
        # Extraction des marques
        brands = self._extract_brands(text)
        if not brands:
            return None
        
        # Extraction des références
        references = self._extract_references(text)
        if not references:
            return None
        
        # Extraction des caractéristiques
        characteristics = self._extract_characteristics(text)
        
        # Extraction du nom du produit (première ligne significative)
        product_name = self._extract_product_name(text)
        
        # Calcul de la confiance globale
        confidence = self._calculate_extraction_confidence(brands, references, characteristics)
        
        # Création du produit
        product = Product(
            id=self._generate_product_id(source_path, section_idx),
            name=product_name,
            brand=brands[0],  # Meilleure marque
            reference=references[0],  # Meilleure référence
            characteristics=characteristics,
            source_document=source_path,
            extraction_confidence=confidence,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return product
    
    def _extract_brands(self, text: str) -> List[ProductBrand]:
        """Extrait les marques du texte"""
        brands = []
        
        for pattern in self.brand_patterns:
            matches = re.finditer(pattern.pattern, text)
            for match in matches:
                brand_name = match.group(1).strip()
                
                # Validation de la marque
                if self._is_valid_brand(brand_name):
                    confidence = pattern.confidence_modifier * self._calculate_brand_confidence(brand_name, text)
                    
                    brand = ProductBrand(
                        name=brand_name,
                        normalized_name=self._normalize_brand_name(brand_name),
                        aliases=[],
                        confidence=confidence
                    )
                    brands.append(brand)
        
        # Déduplication et tri par confiance
        unique_brands = self._deduplicate_brands(brands)
        return sorted(unique_brands, key=lambda x: x.confidence, reverse=True)[:3]
    
    def _extract_references(self, text: str) -> List[ProductReference]:
        """Extrait les références du texte"""
        references = []
        
        for pattern in self.reference_patterns:
            matches = re.finditer(pattern.pattern, text)
            for match in matches:
                ref_value = match.group(1).strip()
                
                # Validation de la référence
                if self._is_valid_reference(ref_value):
                    confidence = pattern.confidence_modifier * self._calculate_reference_confidence(ref_value, text)
                    
                    reference = ProductReference(
                        reference=ref_value,
                        normalized_reference=self._normalize_reference(ref_value),
                        format_pattern=pattern.name,
                        confidence=confidence
                    )
                    references.append(reference)
        
        # Déduplication et tri par confiance
        unique_references = self._deduplicate_references(references)
        return sorted(unique_references, key=lambda x: x.confidence, reverse=True)[:3]
    
    def _extract_characteristics(self, text: str) -> List[ProductCharacteristic]:
        """Extrait les caractéristiques du texte"""
        characteristics = []
        
        for pattern in self.characteristic_patterns:
            matches = re.finditer(pattern.pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    char_name = match.group(1).strip()
                    char_value = match.group(2).strip()
                    char_unit = match.group(3).strip() if len(match.groups()) >= 3 else None
                    
                    # Validation de la caractéristique
                    if self._is_valid_characteristic(char_name, char_value):
                        characteristic = ProductCharacteristic(
                            name=char_name,
                            value=char_value,
                            unit=char_unit,
                            category=self._categorize_characteristic(char_name),
                            normalized_name=self._normalize_characteristic_name(char_name),
                            normalized_value=self._normalize_characteristic_value(char_value, char_unit)
                        )
                        characteristics.append(characteristic)
        
        return self._deduplicate_characteristics(characteristics)
    
    def _extract_product_name(self, text: str) -> str:
        """Extrait le nom du produit"""
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 100:
                # Éviter les lignes qui ressemblent à des métadonnées
                if not re.match(r'^\d+[\.\)]', line) and not re.match(r'^[A-Z\s]+$', line):
                    return line
        
        # Fallback : première ligne non vide
        for line in lines:
            line = line.strip()
            if line:
                return line[:80]  # Limiter la longueur
        
        return "Produit extrait"
    
    # Méthodes utilitaires pour validation et normalisation
    
    def _is_valid_brand(self, brand: str) -> bool:
        """Valide si une chaîne est une marque valide"""
        return (
            len(brand) >= 2 and 
            len(brand) <= 50 and
            not brand.isdigit() and
            not re.match(r'^[^A-Za-z]*$', brand)
        )
    
    def _is_valid_reference(self, reference: str) -> bool:
        """Valide si une chaîne est une référence valide"""
        return (
            len(reference) >= 3 and 
            len(reference) <= 20 and
            bool(re.search(r'[A-Za-z0-9]', reference))
        )
    
    def _is_valid_characteristic(self, name: str, value: str) -> bool:
        """Valide si une caractéristique est valide"""
        return (
            len(name) >= 2 and len(name) <= 50 and
            len(value) >= 1 and len(value) <= 100
        )
    
    def _normalize_brand_name(self, brand: str) -> str:
        """Normalise un nom de marque"""
        return re.sub(r'\s+', ' ', brand.upper().strip())
    
    def _normalize_reference(self, reference: str) -> str:
        """Normalise une référence"""
        return re.sub(r'[^A-Z0-9]', '', reference.upper())
    
    def _normalize_characteristic_name(self, name: str) -> str:
        """Normalise un nom de caractéristique"""
        return re.sub(r'\s+', ' ', name.lower().strip())
    
    def _normalize_characteristic_value(self, value: str, unit: Optional[str] = None) -> str:
        """Normalise une valeur de caractéristique"""
        normalized = value.replace(',', '.').strip()
        if unit:
            normalized += f" {unit.lower()}"
        return normalized
    
    def _categorize_characteristic(self, name: str) -> str:
        """Catégorise une caractéristique"""
        name_lower = name.lower()
        
        if any(word in name_lower for word in ['longueur', 'largeur', 'hauteur', 'dimension', 'taille']):
            return 'dimension'
        elif any(word in name_lower for word in ['poids', 'masse', 'weight']):
            return 'poids'
        elif any(word in name_lower for word in ['matériau', 'material', 'matière']):
            return 'materiau'
        elif any(word in name_lower for word in ['couleur', 'color']):
            return 'couleur'
        elif any(word in name_lower for word in ['tension', 'voltage', 'volt']):
            return 'electrique'
        elif any(word in name_lower for word in ['puissance', 'power', 'watt']):
            return 'puissance'
        else:
            return 'autre'
    
    def _calculate_brand_confidence(self, brand: str, text: str) -> float:
        """Calcule la confiance pour une marque"""
        confidence = 0.5
        
        # Boost si la marque apparaît plusieurs fois
        occurrences = len(re.findall(re.escape(brand), text, re.IGNORECASE))
        confidence += min(occurrences * 0.1, 0.3)
        
        # Boost si la marque est en majuscules
        if brand.isupper():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_reference_confidence(self, reference: str, text: str) -> float:
        """Calcule la confiance pour une référence"""
        confidence = 0.5
        
        # Boost pour les patterns typiques de références
        if re.match(r'^[A-Z]{2,4}[0-9]{3,}', reference):
            confidence += 0.2
        
        # Boost si proche de mots-clés
        if re.search(r'(?i)(?:ref|référence|model|modèle)[\s:]*' + re.escape(reference), text):
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def _calculate_extraction_confidence(self, brands: List[ProductBrand], references: List[ProductReference], characteristics: List[ProductCharacteristic]) -> float:
        """Calcule la confiance globale d'extraction"""
        brand_conf = brands[0].confidence if brands else 0
        ref_conf = references[0].confidence if references else 0
        char_conf = min(len(characteristics) / 5.0, 1.0)  # Plus de caractéristiques = plus de confiance
        
        return (brand_conf * 0.4 + ref_conf * 0.4 + char_conf * 0.2)
    
    def _deduplicate_brands(self, brands: List[ProductBrand]) -> List[ProductBrand]:
        """Supprime les doublons de marques"""
        seen = set()
        unique_brands = []
        
        for brand in brands:
            key = brand.normalized_name
            if key not in seen:
                seen.add(key)
                unique_brands.append(brand)
        
        return unique_brands
    
    def _deduplicate_references(self, references: List[ProductReference]) -> List[ProductReference]:
        """Supprime les doublons de références"""
        seen = set()
        unique_refs = []
        
        for ref in references:
            key = ref.normalized_reference
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        
        return unique_refs
    
    def _deduplicate_characteristics(self, characteristics: List[ProductCharacteristic]) -> List[ProductCharacteristic]:
        """Supprime les doublons de caractéristiques"""
        seen = set()
        unique_chars = []
        
        for char in characteristics:
            key = (char.normalized_name, char.normalized_value)
            if key not in seen:
                seen.add(key)
                unique_chars.append(char)
        
        return unique_chars
    
    def _generate_product_id(self, source_path: str, section_idx: int) -> str:
        """Génère un ID unique pour le produit"""
        source_name = Path(source_path).stem
        content = f"{source_name}_{section_idx}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def extract_from_text(self, text: str, source_path: str) -> List[Product]:
        """Extrait les produits d'un texte simple"""
        return self._extract_products_from_text(text, source_path)
    
    def extract_from_excel_dataframe(self, df, source_path: str, sheet_name: str) -> List[Product]:
        """Extrait les produits d'un DataFrame Excel"""
        products = []
        
        # Conversion du DataFrame en texte
        text_content = ""
        for idx, row in df.iterrows():
            row_text = ' | '.join(str(val) for val in row.values if pd.notna(val))
            text_content += f"Ligne {idx + 1}: {row_text}\n"
        
        # Utilisation de l'extracteur de texte standard
        products = self._extract_products_from_text(text_content, f"{source_path}#{sheet_name}")
        
        return products
    
    def extract_from_json(self, data: Any, source_path: str) -> List[Product]:
        """Extrait les produits de données JSON"""
        # Conversion en texte pour réutiliser la logique existante
        text_content = self._json_to_searchable_text(data)
        return self._extract_products_from_text(text_content, source_path)
    
    def extract_from_xml(self, root, source_path: str) -> List[Product]:
        """Extrait les produits d'un document XML"""
        # Conversion en texte pour réutiliser la logique existante
        text_content = self._xml_to_searchable_text(root)
        return self._extract_products_from_text(text_content, source_path)
    
    def _json_to_searchable_text(self, data: Any) -> str:
        """Convertit JSON en texte recherchable"""
        text_parts = []
        
        def extract_strings(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, str):
                        text_parts.append(f"{key}: {value}")
                    else:
                        extract_strings(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_strings(item, f"{path}[{i}]")
            elif isinstance(obj, str):
                text_parts.append(obj)
        
        extract_strings(data)
        return '\n'.join(text_parts)
    
    def _xml_to_searchable_text(self, element) -> str:
        """Convertit XML en texte recherchable"""
        text_parts = []
        
        # Texte de l'élément
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())
        
        # Attributs
        for key, value in element.attrib.items():
            text_parts.append(f"{key}: {value}")
        
        # Éléments enfants
        for child in element:
            text_parts.append(self._xml_to_searchable_text(child))
        
        return '\n'.join(text_parts)
