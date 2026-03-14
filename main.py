"""
Sistema RAG Avanzado - Backend API
===================================
Fase 1: Esqueleto del backend con endpoints base.

Endpoints:
  POST /api/documents/upload  - Carga y procesamiento de documentos
  POST /api/chat              - Consulta RAG (pregunta → respuesta)
  POST /api/feedback          - Registro de feedback del usuario
"""

import uuid
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Carga de variables de entorno
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Configuración centralizada (pydantic-settings lee desde .env automáticamente)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Configuración de la aplicación leída desde variables de entorno."""

    # Servidor
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"

    # Proveedor LLM activo
    llm_provider: Literal["gemini", "azure"] = "gemini"

    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment_name: str = ""
    azure_openai_api_version: str = "2024-02-01"

    # CORS
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    # RAG
    retrieval_top_k: int = 4
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_store_path: str = "./vector_store"
    embeddings_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------------------------
# Fábrica del LLM (abstracción Gemini / Azure OpenAI)
# ---------------------------------------------------------------------------

def get_llm():
    """
    Devuelve el LLM configurado según la variable LLM_PROVIDER.

    En fases posteriores se inyectará en el pipeline RAG.
    """
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
        logger.info("LLM: Google Gemini (%s)", settings.gemini_model)
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )

    if settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI  # type: ignore
        logger.info("LLM: Azure OpenAI (%s)", settings.azure_openai_deployment_name)
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_deployment_name,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            temperature=0.2,
        )

    raise ValueError(f"LLM_PROVIDER no reconocido: '{settings.llm_provider}'")


# ---------------------------------------------------------------------------
# Ciclo de vida de la aplicación
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y limpieza al arrancar / apagar el servidor."""
    logger.info("🚀 Iniciando Sistema RAG (provider: %s, env: %s)",
                settings.llm_provider, settings.app_env)
    # TODO Fase 2: inicializar vector store y cargar índice persistente
    yield
    logger.info("🛑 Sistema RAG detenido.")


# ---------------------------------------------------------------------------
# Instancia principal de FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sistema RAG Avanzado",
    description=(
        "API backend para el sistema RAG con soporte de carga de documentos, "
        "consulta en lenguaje natural y retroalimentación de usuario."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# Modelos Pydantic (Schemas de request / response)
# ===========================================================================

# --- Documentos ---

class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="ID único del documento procesado")
    filename: str
    status: str
    message: str


# --- Chat ---

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Pregunta del usuario")
    session_id: str | None = Field(None, description="ID de sesión (opcional)")


class SourceDocument(BaseModel):
    filename: str
    page: int | None = None
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = []
    session_id: str
    message_id: str


# --- Feedback ---

class FeedbackRequest(BaseModel):
    message_id: str = Field(..., description="ID del mensaje que recibe feedback")
    rating: Literal["like", "dislike"] = Field(..., description="Valoración del usuario")
    comment: str | None = Field(None, max_length=1000, description="Comentario opcional")


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


# ===========================================================================
# Endpoints
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /  — Health check simple
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Verifica que la API esté en línea."""
    return {
        "status": "ok",
        "service": "Sistema RAG Avanzado",
        "version": app.version,
        "env": settings.app_env,
        "llm_provider": settings.llm_provider,
    }


# ---------------------------------------------------------------------------
# POST /api/documents/upload  — Carga de documentos
# ---------------------------------------------------------------------------

@app.post(
    "/api/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Documentos"],
    summary="Carga y procesa un documento para indexarlo en el vector store",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Recibe un archivo (PDF, DOCX o TXT), lo procesa y lo indexa.

    **Fase 1:** respuesta mockeada.
    **Fase 2:** implementar extracción de texto, chunking y embeddings.
    """
    ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    }

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo de archivo no soportado: {file.content_type}. "
                   "Use PDF, DOCX o TXT.",
        )

    document_id = str(uuid.uuid4())
    logger.info("Documento recibido: %s (id=%s)", file.filename, document_id)

    # TODO Fase 2: leer bytes, extraer texto, dividir en chunks, generar
    #             embeddings y guardar en el vector store.

    return DocumentUploadResponse(
        document_id=document_id,
        filename=file.filename or "desconocido",
        status="pending",
        message=(
            f"Documento '{file.filename}' recibido correctamente. "
            "El procesamiento e indexación se implementarán en la Fase 2."
        ),
    )


# ---------------------------------------------------------------------------
# POST /api/chat  — Consulta RAG
# ---------------------------------------------------------------------------

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Envía una pregunta y recibe la respuesta generada por el pipeline RAG",
)
async def chat(request: ChatRequest):
    """
    Recibe la pregunta del usuario, ejecuta el pipeline RAG y devuelve
    la respuesta junto con las fuentes utilizadas.

    **Fase 1:** respuesta mockeada.
    **Fase 2:** integrar retriever + LLM con LangChain.
    """
    session_id = request.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    logger.info(
        "Pregunta recibida [session=%s, msg=%s]: %s",
        session_id, message_id, request.question,
    )

    # TODO Fase 2: ejecutar pipeline RAG
    #   1. Generar embedding de la pregunta
    #   2. Recuperar chunks relevantes del vector store
    #   3. Construir prompt con contexto
    #   4. Llamar a get_llm() para obtener la respuesta
    #   5. Extraer fuentes de los documentos recuperados

    mock_answer = (
        f"[MOCK] Respuesta a: '{request.question}'. "
        "En la Fase 2 esta respuesta será generada por el pipeline RAG "
        f"usando {settings.llm_provider.upper()} como proveedor LLM."
    )

    return ChatResponse(
        answer=mock_answer,
        sources=[
            SourceDocument(
                filename="documento_ejemplo.pdf",
                page=1,
                excerpt="Fragmento relevante de ejemplo (mock).",
            )
        ],
        session_id=session_id,
        message_id=message_id,
    )


# ---------------------------------------------------------------------------
# POST /api/feedback  — Registro de feedback
# ---------------------------------------------------------------------------

@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Feedback"],
    summary="Registra el feedback del usuario (like, dislike o comentario) sobre una respuesta",
)
async def submit_feedback(request: FeedbackRequest):
    """
    Almacena la valoración del usuario sobre una respuesta del chat.

    **Fase 1:** registro en log (sin persistencia).
    **Fase 2:** guardar en base de datos para análisis de calidad.
    """
    feedback_id = str(uuid.uuid4())

    logger.info(
        "Feedback [id=%s] msg=%s | rating=%s | comment=%s",
        feedback_id, request.message_id, request.rating,
        request.comment or "—",
    )

    # TODO Fase 2: persistir en BD (SQLite / PostgreSQL)
    #             y calcular métricas de calidad (tasa likes/dislikes).

    emoji = "👍" if request.rating == "like" else "👎"
    return FeedbackResponse(
        feedback_id=feedback_id,
        message=f"Feedback {emoji} registrado correctamente. ¡Gracias!",
    )
