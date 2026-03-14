# Sistema RAG Avanzado con Interfaz Web y Feedback de Usuario

## Descripción del Proyecto
[cite_start]Este proyecto implementa un sistema RAG (Retrieval-Augmented Generation) avanzado diseñado para resolver la dificultad de encontrar información relevante en grandes volúmenes de documentos organizacionales[cite: 6, 15, 17]. [cite_start]El sistema permite a los usuarios consultar documentos mediante lenguaje natural, recuperando información a través de búsqueda semántica y generando respuestas contextualizadas, mitigando así el riesgo de alucinaciones de los LLMs tradicionales[cite: 10, 11, 20].

## Arquitectura del Sistema
[cite_start]El proyecto está dividido en cuatro capas principales para asegurar escalabilidad y profesionalismo[cite: 40, 41]:

1. [cite_start]**Capa de Interfaz:** - *Frontend 1:* Chat interactivo para preguntas, respuestas y visualización de fuentes[cite: 44, 45, 25, 28].
   - [cite_start]*Frontend 2:* Portal de gestión para la carga y procesamiento dinámico de documentos[cite: 46, 47, 49, 51].
2. [cite_start]**Capa de Aplicación:** Backend API (FastAPI/Flask) que orquesta las peticiones, llama al motor RAG y registra el feedback[cite: 52, 53, 54, 61].
3. [cite_start]**Capa de Conocimiento:** Pipeline de ingesta (PDF, DOCX, TXT), generación de embeddings y almacenamiento en base de datos vectorial[cite: 62, 63, 67, 76, 77, 78, 81].
4. [cite_start]**Capa de Generación:** Integración con el Modelo de Lenguaje (LLM) para estructurar la respuesta final.

## Características Principales
- [cite_start]**Ingesta Dinámica:** Extracción de texto, limpieza y chunking automatizado[cite: 82, 83, 84].
- [cite_start]**Búsqueda Semántica:** Generación de embeddings de la pregunta y recuperación de los fragmentos más relevantes.
- [cite_start]**Sistema de Retroalimentación:** Los usuarios pueden evaluar las respuestas (Like/Dislike) y dejar comentarios para la mejora continua del sistema[cite: 35, 36, 37, 38, 139].

## Instrucciones de Instalación (Setup Local)

1. Clonar el repositorio.
2. Crear un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate