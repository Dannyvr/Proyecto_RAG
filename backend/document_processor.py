"""
document_processor.py
=====================
Módulo de ingesta de documentos para el Sistema RAG.

Responsabilidades:
  1. Guardar temporalmente el archivo recibido.
  2. Extraer texto según la extensión (.pdf, .docx, .txt).
  3. Dividir el texto en chunks con RecursiveCharacterTextSplitter.
  4. Generar embeddings con HuggingFace (local, sin API key).
  5. Crear o actualizar el índice FAISS local.

Decisión de diseño — Embeddings:
  Se usa 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
  localmente en lugar de los embeddings de Google/Gemini, porque el
  endpoint 'embedContent' de Google AI Studio no está disponible en
  todas las cuentas/regiones.
  El modelo HuggingFace es multilingüe (español incluido), gratuito,
  funciona offline y ya está declarado en requirements.txt.
"""

import logging
import os
import tempfile
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS

logger = logging.getLogger("rag_api.processor")

# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------

FAISS_INDEX_PATH = "./faiss_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Modelo multilingüe local (descargado automáticamente la primera vez, ~330 MB)
HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Extensión → loader de LangChain
_LOADER_MAP: dict[str, type] = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
}

SUPPORTED_EXTENSIONS = set(_LOADER_MAP.keys())


# ---------------------------------------------------------------------------
# Fábrica de embeddings locales
# ---------------------------------------------------------------------------

def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Devuelve el modelo de embeddings local (HuggingFace).
    El modelo se descarga automáticamente la primera vez (~330 MB).
    Las ejecuciones siguientes usan la caché local de HuggingFace.
    """
    logger.info("Cargando modelo de embeddings local: %s", HF_EMBEDDING_MODEL)
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def process_and_index_document(
    *,
    file_bytes: bytes,
    filename: str,
    google_api_key: str,       # Se mantiene en la firma para compatibilidad con main.py
    faiss_index_path: str = FAISS_INDEX_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """
    Procesa un documento y lo indexa en el vector store FAISS.

    Parámetros
    ----------
    file_bytes       : Contenido binario del archivo recibido por FastAPI.
    filename         : Nombre original del archivo.
    google_api_key   : No usado en esta implementación (compatible con main.py).
    faiss_index_path : Ruta local donde se guarda/carga el índice FAISS.
    chunk_size       : Tamaño de cada chunk de texto.
    chunk_overlap    : Solapamiento entre chunks consecutivos.

    Retorna
    -------
    int : Número de chunks indexados.
    """
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensión '{extension}' no soportada. "
            f"Use: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # 1. Guardar bytes en archivo temporal
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp_file:
        tmp_path = tmp_file.name
        tmp_file.write(file_bytes)

    try:
        # 2. Cargar y extraer texto
        documents = _load_documents(tmp_path, extension, filename)
        logger.info(
            "Documento '%s' cargado → %d páginas/secciones.", filename, len(documents)
        )

        # 3. Dividir en chunks
        chunks = _split_documents(documents, chunk_size, chunk_overlap)
        logger.info("Chunks generados: %d", len(chunks))

        if not chunks:
            raise RuntimeError(
                f"No se pudo extraer texto del documento '{filename}'. "
                "Verifica que el archivo no esté vacío o dañado."
            )

        # 4. Generar embeddings y 5. Actualizar/crear índice FAISS
        embeddings = _get_embeddings()
        _upsert_faiss_index(chunks, embeddings, faiss_index_path)
        logger.info(
            "Índice FAISS actualizado en '%s' con %d chunks.", faiss_index_path, len(chunks)
        )

        return len(chunks)

    except (ValueError, RuntimeError):
        raise

    except Exception as exc:
        logger.error("Error procesando '%s': %s", filename, exc, exc_info=True)
        raise RuntimeError(f"Error inesperado procesando '{filename}': {exc}") from exc

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_documents(tmp_path: str, extension: str, original_filename: str):
    """Carga el documento con el loader adecuado y añade metadata de origen."""
    loader_class = _LOADER_MAP[extension]

    if extension == ".txt":
        loader = loader_class(tmp_path, encoding="utf-8")
    else:
        loader = loader_class(tmp_path)

    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = original_filename

    return documents


def _split_documents(documents, chunk_size: int, chunk_overlap: int):
    """Divide los documentos en chunks más pequeños."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _upsert_faiss_index(chunks, embeddings, faiss_index_path: str):
    """
    Crea un índice FAISS si no existe, o carga el existente y agrega
    los nuevos chunks (merge). Guarda el índice en disco.
    """
    index_dir = Path(faiss_index_path)

    if index_dir.exists() and any(index_dir.iterdir()):
        logger.info("Cargando índice FAISS existente desde '%s'.", faiss_index_path)
        vector_store = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        vector_store.add_documents(chunks)
    else:
        logger.info("Creando nuevo índice FAISS en '%s'.", faiss_index_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(faiss_index_path)


# ---------------------------------------------------------------------------
# Carga del vector store para el retriever (Fase 3)
# ---------------------------------------------------------------------------

def load_vector_store(
    google_api_key: str,
    faiss_index_path: str = FAISS_INDEX_PATH,
) -> FAISS:
    """
    Carga el índice FAISS desde disco y lo devuelve listo para búsquedas.
    Usado en el pipeline RAG de la Fase 3.
    """
    index_dir = Path(faiss_index_path)
    if not index_dir.exists() or not any(index_dir.iterdir()):
        raise FileNotFoundError(
            "El índice FAISS no existe. Sube al menos un documento primero "
            f"(ruta esperada: '{faiss_index_path}')."
        )

    return FAISS.load_local(
        faiss_index_path,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )
