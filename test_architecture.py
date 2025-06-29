"""
Script de test pour valider l'architecture refactorisée.
Teste l'extraction et la recherche de produits.
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent))

from app.extraction.product_extractor import ProductExtractor
from app.ingestion.multi_format_processor import MultiFormatProcessor
from app.vectorstore.hybrid_retriever import HybridRetriever
from app.models.schemas import MatchingStage
from app.core.logger import get_logger
from app.core.config import settings

logger = get_logger(__name__)


def test_architecture():
    """Test complet de l'architecture"""
    print("🚀 Test de l'architecture RAG Assistant refactorisée")
    print("=" * 60)
    
    try:
        # 1. Test de l'extracteur de produits
        print("\n1. Test ProductExtractor...")
        extractor = ProductExtractor()
        
        # Exemple de texte avec produits
        sample_text = """
        PRODUIT 1:
        Nom: Perceuse électrique BOSCH GSB120
        Marque: BOSCH
        Référence: GSB120-LI
        Puissance: 120W
        Tension: 12V
        Poids: 1.2kg
        
        PRODUIT 2:
        Nom: Scie circulaire MAKITA HS7100
        Marque: MAKITA  
        Référence: HS7100
        Puissance: 1400W
        Diamètre lame: 190mm
        """
        
        products = extractor.extract_from_text(sample_text, "test_document.txt")
        print(f"   ✅ {len(products)} produits extraits")
        
        for i, product in enumerate(products):
            print(f"   Produit {i+1}: {product.name}")
            print(f"   - Marque: {product.brand.name} (confiance: {product.brand.confidence:.2f})")
            print(f"   - Référence: {product.reference.reference} (confiance: {product.reference.confidence:.2f})")
            print(f"   - Caractéristiques: {len(product.characteristics)}")
        
        # 2. Test du processeur multi-format
        print("\n2. Test MultiFormatProcessor...")
        processor = MultiFormatProcessor()
        
        # Créer un fichier de test
        test_file = Path("test_sample.txt")
        test_file.write_text(sample_text, encoding='utf-8')
        
        document = processor.process_document(str(test_file))
        print(f"   ✅ Document traité: {document.metadata.file_name}")
        print(f"   - Type: {document.metadata.document_type}")
        print(f"   - Taille: {document.metadata.file_size} bytes")
        print(f"   - Produits extraits: {len(document.extracted_products)}")
        print(f"   - Statut: {document.processing_status}")
        
        # Nettoyage
        test_file.unlink()
        
        # 3. Test du retriever hybride
        print("\n3. Test HybridRetriever...")
        retriever = HybridRetriever()
        
        # Ajout des produits au vectorstore
        if products:
            retriever.add_products(products)
            print(f"   ✅ {len(products)} produits ajoutés au vectorstore")
            
            # Test des différentes étapes de recherche
            
            # Étape 1: Recherche par marque
            print("\n   Test Étape 1 - Recherche par marque:")
            brand_results = retriever.search_products(
                "BOSCH",
                MatchingStage.BRAND_MATCHING,
                max_results=5
            )
            print(f"   - Résultats pour 'BOSCH': {len(brand_results)}")
            for product, score in brand_results:
                print(f"     * {product.name} (score: {score:.3f})")
            
            # Étape 2: Recherche par référence
            print("\n   Test Étape 2 - Recherche par référence:")
            ref_results = retriever.search_products(
                "GSB120",
                MatchingStage.REFERENCE_MATCHING,
                max_results=5
            )
            print(f"   - Résultats pour 'GSB120': {len(ref_results)}")
            for product, score in ref_results:
                print(f"     * {product.name} (score: {score:.3f})")
            
            # Étape 3: Recherche par caractéristiques
            print("\n   Test Étape 3 - Recherche par caractéristiques:")
            char_results = retriever.search_products(
                "perceuse 12V puissante",
                MatchingStage.CHARACTERISTIC_MATCHING,
                max_results=5
            )
            print(f"   - Résultats pour 'perceuse 12V puissante': {len(char_results)}")
            for product, score in char_results:
                print(f"     * {product.name} (score: {score:.3f})")
        
        # 4. Test de la configuration
        print("\n4. Test Configuration...")
        print(f"   ✅ Environnement: {settings.environment}")
        print(f"   - Modèle embedding: {settings.vectorstore.embedding_model}")
        print(f"   - Collection ChromaDB: {settings.vectorstore.collection_name}")
        print(f"   - Seuil similarité marques: {settings.product_matching.brand_similarity_threshold}")
        print(f"   - Seuil similarité références: {settings.product_matching.reference_similarity_threshold}")
        print(f"   - Seuil similarité caractéristiques: {settings.product_matching.characteristic_similarity_threshold}")
        
        # 5. Statistiques du vectorstore
        print("\n5. Statistiques vectorstore...")
        stats = retriever.get_stats()
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        print("\n" + "=" * 60)
        print("🎉 Tous les tests sont passés avec succès !")
        print("✅ Architecture refactorisée fonctionnelle")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_architecture()
    sys.exit(0 if success else 1)
