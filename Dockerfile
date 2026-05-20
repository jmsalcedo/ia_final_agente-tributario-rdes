# ===================================================================
# Dockerfile — Agente Tributario Autónomo# ===================================================================
# Dockerfile — Agente Tributario Autónomo
# Optimizado para Render plan FREE (512MB RAM, 0.1 CPU)
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
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias (capa cacheable)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código
COPY app/ ./app/
COPY data/sample/ ./data/sample/
COPY scripts/ ./scripts/

# Directorios persistentes
RUN mkdir -p /app/chroma_db /app/data/raw /app/data/processed /app/logs

# Indexar el corpus de muestra durante el build
RUN python -m app.core.ingest --source sample || echo "Ingest se ejecutara al primer arranque"

# Render asigna puerto dinámicamente vía $PORT (10000 en su caso)
EXPOSE 10000


# Comando de arranque
# - $PORT lo inyecta Render automaticamente
# - --server.address=0.0.0.0 obligatorio para que Render exponga el puerto
CMD streamlit run app/streamlit_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
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
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias (capa cacheable)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código
COPY app/ ./app/
COPY data/sample/ ./data/sample/
COPY scripts/ ./scripts/

# Directorios persistentes
RUN mkdir -p /app/chroma_db /app/data/raw /app/data/processed /app/logs

# Indexar el corpus de muestra durante el build
RUN python -m app.core.ingest --source sample || echo "Ingest se ejecutara al primer arranque"

# Render asigna puerto dinámicamente vía $PORT (10000 en su caso)
EXPOSE 10000


# Comando de arranque
# - $PORT lo inyecta Render automaticamente
# - --server.address=0.0.0.0 obligatorio para que Render exponga el puerto
CMD streamlit run app/streamlit_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false