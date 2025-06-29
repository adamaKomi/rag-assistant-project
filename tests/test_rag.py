import pytest
from app.services.rag_service import process_query
from app.core.config import VECTORSTORE_DIR

def test_rag_chain():
    result = process_query("Quelle est la capitale de la France ?")
    assert isinstance(result, dict)
    assert "result" in result
    assert "source_documents" in result
    assert isinstance(result["result"], str)
    assert isinstance(result["source_documents"], list)

def test_retriever():
    from app.vectorstore.retriever import get_retriever
    retriever = get_retriever(VECTORSTORE_DIR)
    assert retriever is not None