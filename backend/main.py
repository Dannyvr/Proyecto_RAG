"""
Sistema RAG Avanzado - Backend API
===================================
Fase 4: Interfaz Web - Integración Frontend/Backend.

Endpoints:
  GET  /                      - Sirve la interfaz HTML (index.html)
  POST /api/documents/upload  - Carga, procesa e indexa documentos en FAISS
  POST /api/chat              - Consulta RAG real (retriever FAISS + Gemini LLM)
  POST /api/feedback          - Registro de feedback del usuario
"""

import uuid
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from document_processor import process_and_index_document, SUPPORTED_EXTENSIONS
from rag_engine import run_rag_query, RAGResult

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
    gemini_model: str = "gemini-2.5-flash"

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
    vector_store_path: str = "./faiss_index"
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
    logger.info("📂 Índice FAISS se cargará bajo demanda desde: %s",
                settings.vector_store_path)
    yield
    logger.info("🛑 Sistema RAG detenido.")


# ---------------------------------------------------------------------------
# Configuración de rutas para templates y static
# ---------------------------------------------------------------------------

# Obtener la ruta absoluta de la carpeta raíz del proyecto
# (un nivel arriba de donde está main.py que ahora está en backend/)
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "frontend" / "templates"
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"

logger.info("📁 Raíz del proyecto: %s", PROJECT_ROOT)
logger.info("📁 Templates: %s", TEMPLATES_DIR)
logger.info("📁 Static: %s", STATIC_DIR)

# ---------------------------------------------------------------------------
# Instancia principal de FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sistema RAG Avanzado",
    description=(
        "API backend para el sistema RAG con soporte de carga de documentos, "
        "consulta en lenguaje natural y retroalimentación de usuario."
    ),
    version="0.4.0",
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

# ---------------------------------------------------------------------------
# Montaje de archivos estáticos
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    logger.info("✅ Carpeta static montada en /static")
else:
    logger.warning("⚠️ Carpeta static no encontrada en %s", STATIC_DIR)

# ---------------------------------------------------------------------------
# Configuración de templates
# ---------------------------------------------------------------------------

if TEMPLATES_DIR.exists():
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    logger.info("✅ Carpeta templates configurada")
else:
    logger.error("❌ Carpeta templates no encontrada en %s", TEMPLATES_DIR)
    templates = None

# ===========================================================================
# Modelos Pydantic (Schemas de request / response)
# ===========================================================================

# --- Documentos ---

class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="ID único del documento procesado")
    filename: str
    status: str
    chunks_indexed: int = Field(0, description="Número de chunks indexados en el vector store")
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
# GET /  — Página principal (index.html)
# ---------------------------------------------------------------------------

@app.get("/", tags=["Frontend"])
async def root(request: Request):
    """Sirve la interfaz HTML principal (index.html)."""
    if templates is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="La carpeta de templates no está configurada correctamente.",
        )
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request},
    )


# ---------------------------------------------------------------------------
# GET /api/health  — Health check para la API
# ---------------------------------------------------------------------------

@app.get("/api/health", tags=["Health"])
async def health_check():
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
    status_code=status.HTTP_201_CREATED,
    tags=["Documentos"],
    summary="Carga, procesa e indexa un documento en el vector store FAISS",
)
async def upload_document(file: UploadFile = File(...)):
    """
    Recibe un archivo (PDF, DOCX o TXT), extrae su texto, lo divide en chunks,
    genera embeddings con Google Gemini y los almacena en el índice FAISS local.

    - Formatos soportados: `.pdf`, `.docx`, `.txt`
    - Respuesta: `document_id`, nombre del archivo y cantidad de chunks indexados.
    """
    filename = file.filename or "desconocido"
    document_id = str(uuid.uuid4())
    logger.info("Documento recibido: %s (document_id=%s)", filename, document_id)

    # Leer bytes completos del archivo en memoria
    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error("No se pudo leer el archivo '%s': %s", filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Error al leer el archivo. Asegúrate de que no esté corrupto.",
        )

    # Procesar e indexar
    try:
        chunks_indexed = process_and_index_document(
            file_bytes=file_bytes,
            filename=filename,
            google_api_key=settings.google_api_key,
            faiss_index_path=settings.vector_store_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    except ValueError as exc:
        # Extensión no soportada o archivo sin texto extraíble
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        )

    except RuntimeError as exc:
        # Documento vacío / sin texto
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    except Exception as exc:
        logger.error("Error inesperado procesando '%s': %s", filename, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar el documento. Revisa los logs del servidor.",
        )

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        status="indexed",
        chunks_indexed=chunks_indexed,
        message=(
            f"'{filename}' procesado correctamente. "
            f"{chunks_indexed} chunks indexados en el vector store."
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
    Recibe la pregunta del usuario, ejecuta el pipeline RAG completo y devuelve
    la respuesta generada por Gemini junto con las fuentes utilizadas.

    Pipeline:
      1. Carga el índice FAISS local.
      2. Recupera los chunks más relevantes vía similitud semántica.
      3. Construye el prompt con el contexto recuperado.
      4. Llama a Gemini (gemini-2.5-flash) para generar la respuesta.
      5. Retorna la respuesta y los fragmentos fuente.
    """
    session_id = request.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())

    logger.info(
        "Pregunta recibida [session=%s, msg=%s]: %s",
        session_id, message_id, request.question,
    )

    # Ejecutar pipeline RAG
    try:
        result: RAGResult = run_rag_query(
            question=request.question,
            google_api_key=settings.google_api_key,
            faiss_index_path=settings.vector_store_path,
            gemini_model=settings.gemini_model,
            top_k=settings.retrieval_top_k,
        )

    except FileNotFoundError as exc:
        # El índice FAISS aún no existe → el usuario no ha subido documentos
        logger.warning("Índice FAISS no encontrado: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El vector store aún no contiene documentos. "
                "Sube al menos un documento usando POST /api/documents/upload."
            ),
        )

    except Exception as exc:
        logger.error(
            "Error en el pipeline RAG [session=%s]: %s", session_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar la consulta. Revisa los logs del servidor.",
        )

    # Construir respuesta con fuentes
    sources = [
        SourceDocument(
            filename=chunk["source"],
            page=chunk["page"],
            excerpt=chunk["text"][:300],   # Primeros 300 caracteres del chunk
        )
        for chunk in result.source_chunks
    ]

    logger.info(
        "Respuesta enviada [session=%s, msg=%s] — fuentes: %d",
        session_id, message_id, len(sources),
    )

    return ChatResponse(
        answer=result.answer,
        sources=sources,
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
