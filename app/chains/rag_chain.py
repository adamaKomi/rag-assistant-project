from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from app.core.logger import logger
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any

def get_rag_chain(retriever, memory: bool = False) -> RetrievalQA:
    """
    Crée une chaîne RAG optimisée pour le français
    """
    # Template amélioré
    prompt_template = """[Instruction] Tu es un expert technique francophone. 
    Réponds en français en suivant strictement ces règles :
    1. Base-toi uniquement sur ce contexte :
    {context}

    2. Question : {question}

    3. Règles de réponse :
    - Maximum 3 phrases
    - Structure claire : [Réponse] + [Sources]
    - Précision technique
    - Si inconnu : "Je n'ai pas d'information pertinente"

    [Contexte]: {context}
    [Question]: {question}
    [Réponse]:"""

    custom_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "query"],  
        template_format="f-string"
    )

    # Configuration corrigée du LLM
    callbacks = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = OllamaLLM(
        model="mistral",
        temperature=0.2,
        callback_manager=callbacks,
        system="Tu es un assistant technique précis en français",  # 'system' au lieu de 'system_prompt'
        top_k=30,
        top_p=0.9
    )

    # Mémoire conversationnelle
    memory_obj = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="result"
    ) if memory else None

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        memory=memory_obj,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": custom_prompt,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
        },
        verbose=True,
        output_key="result"
    )