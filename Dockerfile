# ===================================================================
# Dockerfile — Agente Tributario Autónomo
# ===================================================================

FROM python:3.11-slim

# Variables de entorno
# PYTHONPATH=/app permite que `from app.core...` encuentre el paquete
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# HF Spaces requiere un usuario no-root con UID 1000
RUN useradd -m -u 1000 user

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias (capa cacheable)
COPY --chown=user requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código
COPY --chown=user app/ ./app/
COPY --chown=user data/sample/ ./data/sample/
COPY --chown=user scripts/ ./scripts/

# Directorios persistentes (HF Spaces los persiste en /data si es necesario)
RUN mkdir -p /app/chroma_db /app/data/raw /app/data/processed /app/logs && \
    chown -R user:user /app

# Cambiar a usuario no-root (requisito HF Spaces)
USER user

# Pre-descargar el modelo de embeddings durante el build
# Esto evita los ~20s de descarga en cada arranque del contenedor
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Indexar el corpus de muestra durante el build
RUN python -m app.core.ingest --source sample || echo "Ingest se ejecutara al primer arranque"

# Puerto estándar de HF Spaces
EXPOSE 7860

# Comando de arranque
CMD streamlit run app/streamlit_app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false