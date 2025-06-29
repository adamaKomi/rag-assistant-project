from langchain_community.document_loaders import PyPDFLoader
import os

# def load_pdfs(pdf_dir):
#     docs = []
#     for file in os.listdir(pdf_dir):
#         if file.endswith(".pdf"):
#             # loader = PyPDFLoader(os.path.join(pdf_dir, file))
#             loader = PyPDFLoader(os.path.join(pdf_dir, file), encoding='utf-8')  # Force l'UTF-8
#             docs.extend(loader.load())
#     return docs

def load_pdfs(pdf_dir):
    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(pdf_dir, file))
                # Conversion explicite en UTF-8 après chargement
                pages = loader.load()
                for page in pages:
                    page.page_content = page.page_content.encode('utf-8', errors='replace').decode('utf-8')
                docs.extend(pages)
            except Exception as e:
                print(f"Erreur sur le fichier {file}: {str(e)}")
                continue
    return docs