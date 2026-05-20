# ===================================================================
# Dockerfile — Agente Tributario Autónomo
# Imagen ligera con Python 3.11, optimizada para Render y local
# ===================================================================

FROM python:3.11-slim

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Dependencias del sistema mínimas (poppler para PDFs si fuese necesario)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero (cache de capas)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código de la aplicación
COPY app/ ./app/
COPY data/sample/ ./data/sample/
COPY scripts/ ./scripts/

# Crear directorios persistentes
RUN mkdir -p /app/chroma_db /app/data/raw /app/data/processed /app/logs

# Indexar el corpus de muestra durante el build (para que la app arranque rápido)
RUN python -m app.core.ingest --source sample || echo "Ingest se ejecutará al primer arranque"

# Puerto que expone Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de arranque
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
