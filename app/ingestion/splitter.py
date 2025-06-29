from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(docs):
    # Paramètres optimisés pour équilibrer contexte et performance
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Réduit pour des chunks plus ciblés
        chunk_overlap=50,  # Suffisant pour garder le contexte
        separators=["\n\n", "\n", ". ", " ", ""]  # Hiérarchie de séparation
    )
    # return splitter.split_documents(docs)
    splits = splitter.split_documents(docs)
    return [doc for doc in splits if len(doc.page_content) > 20]