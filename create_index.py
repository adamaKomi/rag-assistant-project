import os
import sys
from dotenv import load_dotenv
from app.ingestion.pdf_loader import load_pdfs
from app.ingestion.web_loader import load_websites
from app.ingestion.splitter import split_docs
from app.vectorstore.build_store import build_vectorstore
from app.core.config import VECTORSTORE_DIR, PDF_DIR
from app.core.logger import logger
import re

# Forcer l'encodage UTF-8
if sys.platform.startswith('win'):
    import locale
    locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
load_dotenv()

# def clean_text(text):
#     """Nettoyage des caractères spéciaux"""
#     # Supprime les caractères non-ASCII, espaces multiples, et lignes vides
#     text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # ASCII seulement
#     text = re.sub(r'\s+', ' ', text).strip()  # Espaces multiples
#     return text.encode('utf-8', 'ignore').decode('utf-8')

# def clean_text(text):
#     """Nettoyage des caractères spéciaux tout en conservant les accents"""
#     # Garde les caractères UTF-8 standards (inclut les accents)
#     text = re.sub(r'[^\w\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ.,;:!?\-]', ' ', text)
#     # Réduit les espaces multiples et supprime les espaces en début/fin
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

def clean_text(text):
    """Nettoyage conservant les accents et caractères spéciaux utiles"""
    # Garde lettres, chiffres, ponctuation et symboles courants
    text = re.sub(r'[^\w\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ.,;:!?\-()\'"@]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("’", "'")  # Normalise les apostrophes courbes
    return text


def main():
    try:
        # Charger les documents
        pdf_docs = load_pdfs(PDF_DIR)
        # web_docs = load_websites(os.getenv("WEB_SOURCES", "").split(","))
        web_sources = [url.strip() for url in os.getenv("WEB_SOURCES", "").split(",") if url.strip() and not url.startswith("#")]
        web_docs = load_websites(web_sources) if web_sources else []
        all_docs = pdf_docs + web_docs
        
        print(f"Avant nettoyage : {all_docs[0].page_content[:100]}...")
        # Nettoyer les textes
        for doc in all_docs:
            doc.page_content = clean_text(doc.page_content)
        print(f"Après nettoyage : {clean_text(all_docs[0].page_content[:100])}...")
        
        # Découper les documents
        splits = split_docs(all_docs)
        print(f"Exemple de chunk : {splits[0].page_content[:100]}...")
        logger.info(f"{len(splits)} chunks créés")
        
        # Construire et sauvegarder le vectorstore
        build_vectorstore(splits, VECTORSTORE_DIR)
        logger.info(f"Index FAISS sauvegardé dans {VECTORSTORE_DIR}")
    except Exception as e:
        logger.error(f"Erreur lors de la création de l'index: {e}")
        raise

if __name__ == "__main__":
    main()