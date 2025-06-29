from fastapi import APIRouter, Request
from app.services.rag_service import process_query

router = APIRouter()

@router.post("/rag")
async def rag_endpoint(request: Request):
    data = await request.json()
    query = data.get("query", "")
    return process_query(query)
