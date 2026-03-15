"""
Sistema RAG Avanzado - Backend API
===================================
Fase 4: Interfaz Web - Integración Frontend/Backend.

Endpoints:
  GET  /                      - Sirve la interfaz HTML (index.html)
  POST /api/documents/upload  - Carga, procesa e indexa documentos en FAISS
  DELETE /api/documents/reset - Borra el índice FAISS y reinicia la base de conocimientos
  POST /api/chat              - Consulta RAG real (retriever FAISS + Gemini LLM)
  POST /api/feedback          - Registro de feedback del usuario
"""

import uuid
import logging
import shutil
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

    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

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
# Ciclo de vida de la aplicación
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicialización y limpieza al arrancar / apagar el servidor."""
    logger.info("🚀 Iniciando Sistema RAG (LLM: Google Gemini, env: %s)",
                settings.app_env)
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


class DocumentResetResponse(BaseModel):
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
    rating: Literal["like", "dislike", "neutral"] = Field(..., description="Valoración del usuario")
    comment: str | None = Field(None, max_length=1000, description="Comentario opcional")


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


# --- Analytics ---

class CommentRecord(BaseModel):
    feedback_id: str
    rating: str
    comment: str
    timestamp: str


class AnalyticsResponse(BaseModel):
    total_interactions: int = Field(..., description="Total de interacciones con feedback")
    total_likes: int = Field(..., description="Total de likes")
    total_dislikes: int = Field(..., description="Total de dislikes")
    approval_rate: float = Field(..., description="Porcentaje de aprobación (likes / total * 100)")
    comments_with_text: list[CommentRecord] = Field(..., description="Comentarios que contienen texto")
    total_comments: int = Field(..., description="Total de comentarios con texto")


class AnalyticsClearResponse(BaseModel):
    status: str
    message: str
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
        "llm_model": settings.gemini_model,
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
# DELETE /api/documents/reset  — Reinicia la base de conocimientos
# ---------------------------------------------------------------------------

@app.delete(
    "/api/documents/reset",
    response_model=DocumentResetResponse,
    status_code=status.HTTP_200_OK,
    tags=["Documentos"],
    summary="Elimina el índice FAISS y reinicia la base de conocimientos",
)
async def reset_documents():
    """
    Elimina completamente el índice FAISS almacenado en disco.
    Esto reinicia la base de conocimientos a un estado vacío.

    **Advertencia:** Esta operación es irreversible. Se borrarán todos los documentos indexados.

    Retorna:
      - status: "success" si se eliminó correctamente
      - message: Confirmación con detalles
    """
    vector_store_path = Path(settings.vector_store_path)
    logger.info("🗑️ Reinicio de base de conocimientos solicitado (ruta: %s)", vector_store_path)

    try:
        if vector_store_path.exists() and vector_store_path.is_dir():
            shutil.rmtree(vector_store_path)
            logger.info("✅ Índice FAISS eliminado correctamente de: %s", vector_store_path)
            return DocumentResetResponse(
                status="success",
                message=f"Base de conocimientos reiniciada. Se eliminó el índice en '{vector_store_path}'.",
            )
        else:
            # La carpeta no existe, pero no es error
            logger.warning("⚠️ Intento de reset pero la carpeta FAISS no existe: %s", vector_store_path)
            return DocumentResetResponse(
                status="success",
                message="La base de conocimientos ya estaba vacía (no había índice que eliminar).",
            )

    except Exception as exc:
        logger.error("❌ Error al reiniciar la base de conocimientos: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar el índice FAISS: {str(exc)}",
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

    **Fase 1:** Persiste en un archivo JSON local (data/feedback.json).
    **Fase Future:** Migrar a base de datos SQL para análisis de calidad en escala.

    Registra:
      - feedback_id: UUID único del feedback
      - message_id: ID de la respuesta que recibe feedback
      - rating: "like" o "dislike"
      - comment: Comentario opcional del usuario
      - timestamp: Marca de tiempo de creación
    """
    import json
    from datetime import datetime

    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Crear carpeta data/ si no existe
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    feedback_file = data_dir / "feedback.json"

    # Construir registro de feedback
    feedback_record = {
        "feedback_id": feedback_id,
        "message_id": request.message_id,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": timestamp,
    }

    # Leer feedback existente o inicializar con lista vacía
    try:
        if feedback_file.exists():
            with open(feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)
        else:
            feedbacks = []
    except json.JSONDecodeError:
        feedbacks = []

    # Agregar nuevo feedback
    feedbacks.append(feedback_record)

    # Guardar en archivo
    try:
        with open(feedback_file, "w", encoding="utf-8") as f:
            json.dump(feedbacks, f, indent=2, ensure_ascii=False)
        logger.info(
            "✅ Feedback guardado [id=%s] msg=%s | rating=%s | file=%s",
            feedback_id, request.message_id, request.rating, feedback_file,
        )
    except Exception as exc:
        logger.error("❌ Error al guardar feedback en archivo: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al guardar el feedback. Revisa los logs del servidor.",
        )

    # Retornar confirmación exitosa
    emoji = "👍" if request.rating == "like" else "👎" if request.rating == "dislike" else "💬"
    return FeedbackResponse(
        feedback_id=feedback_id,
        message=f"Feedback {emoji} registrado correctamente. ¡Gracias por tu opinión!",
    )


# ---------------------------------------------------------------------------
# GET /api/analytics  — Obtener métricas de calidad
# ---------------------------------------------------------------------------

@app.get(
    "/api/analytics",
    response_model=AnalyticsResponse,
    tags=["Analytics"],
    summary="Obtiene las métricas de calidad basadas en el feedback del usuario",
)
async def get_analytics():
    """
    Lee el archivo `data/feedback.json` y retorna estadísticas agregadas:
      - Total de interacciones
      - Likes vs Dislikes
      - Tasa de aprobación
      - Lista de comentarios con texto
    """
    import json

    feedback_file = PROJECT_ROOT / "data" / "feedback.json"

    # Inicializar contadores
    total_likes = 0
    total_dislikes = 0
    comments_with_text = []

    # Leer feedback si existe
    if feedback_file.exists():
        try:
            with open(feedback_file, "r", encoding="utf-8") as f:
                feedbacks = json.load(f)

            for feedback in feedbacks:
                rating = feedback.get("rating", "").lower()
                if rating == "like":
                    total_likes += 1
                elif rating == "dislike":
                    total_dislikes += 1

                # Recopilar comentarios con texto (maneja None correctamente)
                comment = (feedback.get("comment") or "").strip()
                if comment:
                    comments_with_text.append(
                        CommentRecord(
                            feedback_id=feedback.get("feedback_id", ""),
                            rating=rating,
                            comment=comment,
                            timestamp=feedback.get("timestamp", ""),
                        )
                    )

        except json.JSONDecodeError:
            logger.warning("⚠️ Error decodificando feedback.json, retornando valores vacíos")
    else:
        logger.info("ℹ️ Archivo feedback.json no existe aún")

    # Calcular métricas
    total_interactions = total_likes + total_dislikes
    approval_rate = (
        (total_likes / total_interactions * 100) if total_interactions > 0 else 0
    )

    logger.info(
        "📊 Analytics: total=%d, likes=%d, dislikes=%d, approval=%.1f%%",
        total_interactions, total_likes, total_dislikes, approval_rate,
    )

    return AnalyticsResponse(
        total_interactions=total_interactions,
        total_likes=total_likes,
        total_dislikes=total_dislikes,
        approval_rate=approval_rate,
        comments_with_text=comments_with_text,
        total_comments=len(comments_with_text),
    )


# ---------------------------------------------------------------------------
# DELETE /api/analytics/clear  — Eliminar todo el feedback y comentarios
# ---------------------------------------------------------------------------

@app.delete(
    "/api/analytics/clear",
    response_model=AnalyticsClearResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analytics"],
    summary="Elimina todo el historial de feedback y comentarios",
)
async def clear_analytics():
    """
    Elimina el archivo `data/feedback.json` completamente.
    Esto reinicia las métricas de calidad a cero.

    **Advertencia:** Esta operación es irreversible. Se borrarán todos los feedbacks.
    """
    import json

    feedback_file = PROJECT_ROOT / "data" / "feedback.json"
    logger.info("🗑️ Limpieza de feedback solicitada (ruta: %s)", feedback_file)

    try:
        if feedback_file.exists():
            feedback_file.unlink()  # Elimina el archivo
            logger.info("✅ Archivo de feedback eliminado correctamente")
            return AnalyticsClearResponse(
                status="success",
                message="Historial de feedback y comentarios eliminado correctamente.",
            )
        else:
            logger.warning("⚠️ Intento de clear pero feedback.json no existe")
            return AnalyticsClearResponse(
                status="success",
                message="El historial de feedback ya estaba vacío.",
            )

    except Exception as exc:
        logger.error("❌ Error al eliminar feedback.json: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar el historial de feedback: {str(exc)}",
        )
