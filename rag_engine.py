"""
rag_engine.py
=============
Motor RAG (Retrieval-Augmented Generation) — Fase 3.

Responsabilidades:
  1. Cargar el índice FAISS local usando el mismo modelo de embeddings que
     usó document_processor.py (paraphrase-multilingual-MiniLM-L12-v2).
  2. Exponer un retriever que devuelva los k chunks más relevantes.
  3. Configurar el LLM Google Gemini (gemini-1.5-flash) via LangChain.
  4. Construir y ejecutar la cadena RAG completa usando LCEL.
  5. Devolver la respuesta y las fuentes al endpoint /api/chat.
"""

import logging
from pathlib import Path
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("rag_api.engine")

# ---------------------------------------------------------------------------
# Constantes (deben coincidir exactamente con document_processor.py)
# ---------------------------------------------------------------------------

HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# Dataclass de resultado
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Resultado del pipeline RAG."""
    answer: str
    source_chunks: list[dict]   # Lista de {"text": str, "source": str, "page": int | None}


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Devuelve el modelo de embeddings HuggingFace.
    DEBE ser idéntico al usado en document_processor.py.
    """
    logger.info("Cargando modelo de embeddings: %s", HF_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_faiss_index(faiss_index_path: str) -> FAISS:
    """
    Carga el índice FAISS desde disco.

    Raises
    ------
    FileNotFoundError
        Si el índice aún no existe (ningún documento ha sido subido).
    """
    index_dir = Path(faiss_index_path)
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise FileNotFoundError(
            f"El índice FAISS no existe en '{faiss_index_path}'. "
            "Sube al menos un documento usando POST /api/documents/upload antes de consultar."
        )

    logger.info("Cargando índice FAISS desde '%s'.", faiss_index_path)
    return FAISS.load_local(
        faiss_index_path,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def _build_prompt() -> ChatPromptTemplate:
    """
    Prompt estricto: el LLM debe responder SÓLO con el contexto proporcionado.
    Si no tiene información suficiente, debe indicarlo claramente.
    """
    system_message = (
        "Eres un asistente experto que responde preguntas basándose EXCLUSIVAMENTE "
        "en los fragmentos de contexto que se te proporcionan a continuación.\n\n"
        "REGLAS ESTRICTAS:\n"
        "1. Responde ÚNICAMENTE usando la información del contexto provisto.\n"
        "2. Si la información para responder la pregunta NO está en el contexto, "
        '   responde exactamente: "No tengo información suficiente en los documentos '
        '   disponibles para responder esta pregunta."\n'
        "3. No inventes datos, fechas, nombres ni cifras que no estén en el contexto.\n"
        "4. Responde siempre en el mismo idioma en que está escrita la pregunta.\n"
        "5. Sé conciso y directo.\n\n"
        "CONTEXTO:\n"
        "{context}"
    )
    human_message = "PREGUNTA: {question}"

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])


def _format_docs(docs) -> str:
    """Concatena el contenido de los documentos recuperados en un único string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def run_rag_query(
    *,
    question: str,
    google_api_key: str,
    faiss_index_path: str = "./faiss_index",
    gemini_model: str = "gemini-1.5-flash",
    top_k: int = DEFAULT_TOP_K,
) -> RAGResult:
    """
    Ejecuta el pipeline RAG completo.

    Parámetros
    ----------
    question         : Pregunta en lenguaje natural del usuario.
    google_api_key   : API Key de Google AI Studio.
    faiss_index_path : Ruta al índice FAISS en disco.
    gemini_model     : Nombre del modelo Gemini a usar.
    top_k            : Número de chunks a recuperar del vector store.

    Retorna
    -------
    RAGResult con la respuesta generada y los chunks fuente.

    Raises
    ------
    FileNotFoundError
        Si el índice FAISS no existe todavía.
    Exception
        Cualquier error de conectividad con la API de Gemini.
    """
    # 1. Cargar índice FAISS
    vector_store = _load_faiss_index(faiss_index_path)

    # 2. Crear retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    # 3. Configurar LLM (Gemini)
    logger.info("Configurando LLM: %s", gemini_model)
    llm = ChatGoogleGenerativeAI(
        model=gemini_model,
        google_api_key=google_api_key,
        temperature=0.2,
    )

    # 4. Construir cadena RAG con LCEL
    prompt = _build_prompt()

    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Recuperar chunks por separado para extraer las fuentes
    retrieved_docs = retriever.invoke(question)
    logger.info(
        "Chunks recuperados: %d para la pregunta: '%s'",
        len(retrieved_docs), question[:80],
    )

    # 6. Generar respuesta
    logger.info("Invocando LLM (%s)...", gemini_model)
    answer = rag_chain.invoke(question)
    logger.info("Respuesta generada correctamente.")

    # 7. Construir lista de fuentes
    source_chunks = []
    for doc in retrieved_docs:
        meta = doc.metadata or {}
        source_chunks.append({
            "text": doc.page_content,
            "source": meta.get("source", "desconocido"),
            "page": meta.get("page", None),
        })

    return RAGResult(answer=answer, source_chunks=source_chunks)
