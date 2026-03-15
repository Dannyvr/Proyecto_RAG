# Sistema RAG Avanzado con Interfaz Web y Feedback de Usuario

## 📋 Descripción del Proyecto

Este proyecto implementa un **Sistema RAG (Retrieval-Augmented Generation)** avanzado diseñado para resolver la dificultad de encontrar información relevante en grandes volúmenes de documentos organizacionales.

### Problema que resuelve
- **Crecimiento de documentos digitales:** Las organizaciones acumulan manuales, reportes, normativas, guías e investigaciones sin una forma eficiente de consultarlos.
- **Búsquedas tradicionales limitadas:** Los motores de búsqueda tradicionales no entienden la intención del usuario.
- **Alucinaciones de LLMs:** Los modelos generativos sin contexto pueden inventar respuestas.
- **Falta de evaluación:** No existe forma de evaluar la calidad de las respuestas generadas.

### Solución propuesta
El enfoque **RAG** permite responder preguntas usando **conocimiento real almacenado en documentos**, mitigando alucinaciones y proporcionando respuestas contextualizadas con trazabilidad de fuentes.

---

## 🎯 Objetivos del Proyecto

1. ✅ Consultar documentos mediante **lenguaje natural**
2. ✅ Recuperar información relevante usando **búsqueda semántica**
3. ✅ Generar **respuestas contextualizadas** con fuentes citadas
4. ✅ Permitir **carga dinámica de documentos** (PDF, DOCX, TXT)
5. ✅ Recopilar **retroalimentación del usuario** (Like/Dislike + Comentarios)

---

## 🏗️ Arquitectura del Sistema

El proyecto está dividido en **4 capas principales** para asegurar escalabilidad y profesionalismo:

### 1. **Capa de Interfaz (Frontend)**
- **Chat Interactivo:** Interfaz para realizar preguntas, ver respuestas generadas y visualizar fuentes utilizadas.
- **Portal de Gestión de Documentos:** Sistema para cargar, procesar e indexar documentos automáticamente.

### 2. **Capa de Aplicación (Backend)**
- **API REST (FastAPI):** Orquesta las peticiones de usuarios, llama al motor RAG y registra feedback.
- **Endpoints principales:**
  - `POST /api/query` - Procesa preguntas del usuario
  - `POST /api/upload` - Ingesta de documentos
  - `POST /api/feedback` - Sistema de evaluación (Like/Dislike/Comentarios)

### 3. **Capa de Conocimiento (RAG Pipeline)**
- **Ingesta:** Extracción automática de texto desde PDF, DOCX, TXT
- **Limpieza y Chunking:** División inteligente del texto en fragmentos semánticamente coherentes
- **Embeddings:** Conversión de fragmentos a vectores semánticos usando modelos de HuggingFace
- **Almacenamiento Vectorial:** Base de datos FAISS para búsqueda rápida y eficiente

### 4. **Capa de Generación (LLM)**
- **Modelo de Lenguaje:** Google Gemini API para generar respuestas finales
- **Contexto Recuperado:** Utiliza fragmentos más relevantes como contexto
- **Estructuración:** Formatea respuestas con referencias a las fuentes

---

## ⚡ Características Principales

### 📄 Ingesta Dinámica de Documentos
- Soporte para múltiples formatos: **PDF, DOCX, TXT**
- Extracción automática de texto
- Limpieza y normalización de contenido
- Chunking inteligente con overlapping

### 🔍 Búsqueda Semántica
- Generación de embeddings de preguntas en tiempo real
- Recuperación de fragmentos más relevantes (top-k)
- Búsqueda por similitud coseno en vectores

### 💬 Sistema de Retroalimentación
- **Like/Dislike:** Valoración rápida de respuestas
- **Comentarios:** Campo abierto para feedback detallado
- **Persistencia:** Almacenamiento en `data/feedback.json` con timestamps
- **Análisis:** Datos para mejora continua del sistema

### 🎨 Interfaz Intuitiva
- Chat en tiempo real
- Visualización de fuentes consultadas
- Indicadores visuales de feedback (emojis + colores)
- Interfaz responsiva y moderna

---

## 🛠️ Tecnologías Utilizadas

### Backend
- **Framework:** FastAPI + Uvicorn
- **LLM:** Google Gemini AI (via `langchain-google-genai`)
- **RAG Framework:** LangChain (core, community, huggingface)
- **Embeddings:** HuggingFace (`sentence-transformers`)
- **Vector Store:** FAISS (CPU)
- **Procesamiento de Docs:** PyPDF, python-docx, docx2txt

### Frontend
- **HTML/CSS/JavaScript:** Interfaz interactiva
- **Template Engine:** HTML5 + Fetch API

### Dependencias Clave
```
fastapi==0.115.6
langchain==0.3.13
langchain-google-genai==4.2.1
sentence-transformers==3.3.1
faiss-cpu==1.9.0
pypdf==5.1.0
python-docx==1.1.2
```

---

## 🚀 Instalación y Configuración

### Prerequisites
- Python 3.8+
- pip (gestor de paquetes)

### Pasos de Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone <repository-url>
   cd Proyecto_RAG
   ```

2. **Crear un entorno virtual:**
   ```bash
   python -m venv venv
   ```
   - **En Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **En Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

4. **Configurar variables de entorno:**
   Crear un archivo `.env` en la raíz del proyecto:
   ```env
   # Google Gemini API
   GOOGLE_API_KEY=tu_clave_api_aqui
   
   # Configuración del servidor
   HOST=0.0.0.0
   PORT=8000
   
   # Base de datos vectorial
   FAISS_INDEX_PATH=./data/faiss_index
   ```

5. **Crear directorios necesarios:**
   ```bash
   mkdir -p data
   ```

---

## 📂 Estructura del Proyecto

```
Proyecto_RAG/
├── README.md                 # Este archivo
├── Contexto/                 # Documentación del proyecto
│   └── contexto.md          # Especificación del sistema
├── backend/                  # Backend FastAPI
│   ├── main.py              # Aplicación principal y endpoints
│   ├── rag_engine.py        # Núcleo del sistema RAG
│   ├── document_processor.py # Procesamiento de documentos
│   ├── requirements.txt      # Dependencias de Python
│   └── __pycache__/
├── frontend/                 # Frontend (HTML + JavaScript)
│   ├── static/              # Archivos CSS y JS
│   └── templates/
│       └── index.html       # Interfaz principal
├── data/                    # Directorio de datos
│   ├── feedback.json        # Retroalimentación del usuario
│   ├── faiss_index          # Índice vectorial (generado)
│   └── documents/           # Documentos cargados
└── __pycache__/
```

---

## 🔧 Uso del Sistema

### Iniciar el servidor

1. **Activar el entorno virtual** (si no está activado)

2. **Ejecutar el backend:**
   ```bash
   cd backend
   python main.py
   ```
   El servidor estará disponible en `http://localhost:8000`

3. **Acceder a la interfaz:**
   - Abrir navegador: `http://localhost:8000`
   - O navegar a: `http://localhost:8000/static/index.html`

### Flujo de usuario

#### 1. **Cargar documentos**
   - Ir a la pestaña "Gestión de Documentos"
   - Seleccionar archivos (PDF, DOCX, TXT)
   - Hacer clic en "Procesar"
   - Esperar a que se creen los embeddings

#### 2. **Realizar consultas**
   - Ir a la pestaña "Chat"
   - Escribir una pregunta en lenguaje natural
   - El sistema RAG recuperará fragmentos relevantes
   - Se mostrará la respuesta con fuentes citadas

#### 3. **Proporcionar retroalimentación**
   - Hacer clic en 👍 (Like) o 👎 (Dislike)
   - (Opcional) Hacer clic en "Comentario" para agregar feedback detallado
   - Los datos se almacenan automáticamente en `data/feedback.json`

---

## 📊 Endpoints de la API

### Consultas RAG
**POST** `/api/query`
```json
{
  "question": "¿Cuál es la política de vacaciones?",
  "top_k": 5
}
```
**Response:**
```json
{
  "answer": "La política de vacaciones permite...",
  "sources": ["Documento1.pdf (pág. 5)", "Documento2.docx (pág. 12)"],
  "message_id": "uuid-xxx"
}
```

### Carga de documentos
**POST** `/api/upload`
- Multipart form data con archivos
- Procesa automáticamente y crea embeddings

### Retroalimentación
**POST** `/api/feedback`
```json
{
  "message_id": "uuid-xxx",
  "rating": "like",
  "comment": "Respuesta muy útil"
}
```
**Response:**
```json
{
  "feedback_id": "uuid-yyy",
  "message": "✓ Gracias por tu feedback"
}
```

---

## 🧠 Sistema de Feedback (Fase 5) ✅

### Características Implementadas

#### Frontend
- **Botones Like/Dislike** con emojis 👍 👎
- **Cambio visual** al hacer click (colores verde/rojo)
- **Campo de comentarios** con validación
- **Mensajes de confirmación** con auto-cierre

#### Backend
- **Endpoint `/api/feedback`** que valida y almacena feedback
- **UUID único** para cada feedback
- **Timestamps** para análisis temporal
- **Manejo de errores** robusto

#### Almacenamiento
```json
{
  "data/feedback.json": [
    {
      "feedback_id": "550e8400-e29b-41d4-a716-446655440000",
      "message_id": "msg-123",
      "rating": "like",
      "comment": "Muy buena respuesta",
      "timestamp": "2026-03-14T15:30:45.123456"
    }
  ]
}
```

---

## 🔮 Roadmap Futuro

- [ ] Dashboard de Analytics (gráficos like/dislike rate)
- [ ] Migración a BD SQL (SQLite/PostgreSQL)
- [ ] Email notifications para feedbacks bajos
- [ ] A/B testing de prompts según feedback
- [ ] Autenticación de usuarios
- [ ] Historial de conversaciones
- [ ] Fine-tuning de embeddings
- [ ] Soporte multi-idioma

---

## 📝 Variables de Entorno

Se requiere configurar las siguientes variables en `.env`:

```env
# API de Google Gemini (requerido)
GOOGLE_API_KEY=tu_clave_secreta_aqui

# Configuración del servidor
HOST=0.0.0.0
PORT=8000

# Rutas de datos
FAISS_INDEX_PATH=./data/faiss_index
FEEDBACK_PATH=./data/feedback.json

# Modelos de embeddings
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🐛 Troubleshooting

### Error: "API key no válida"
- Verificar que `GOOGLE_API_KEY` está correctamente configurado
- Regenerar la clave en Google AI Studio

### Error: "No such file or directory: 'data/faiss_index'"
- Ejecutar: `mkdir -p data`
- Cargar al menos un documento para crear el índice

### Error de CUDA/GPU
- El proyecto usa `faiss-cpu` por defecto
- Para usar GPU: `pip install faiss-gpu` (requiere CUDA)

---

## 👤 Autor

**Nombre del estudiante:** Danny  
**Institución:** Cenfotec (Cuatrimestre 4)  
**Materia:** Aplicaciones AI  
**Fecha de entrega:** 19 de marzo de 2026

---

## 📄 Licencia

Este proyecto es de uso académico y educativo