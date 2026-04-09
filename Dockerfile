FROM python:3.12-slim

# Env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /immo

# Install system deps + uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer uv (important)
RUN pip install uv

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock ./

# Installer les deps (sans installer ton projet)
RUN uv sync --no-dev

# Copier le code
COPY src/ ./src

# Créer dossiers
RUN mkdir -p /data/models && mkdir -p /agentia/bdd

# Copier le modèle au bon endroit
COPY src/app/data/models/xgb_pipeline.pkl data/models/xgb_pipeline.pkl
COPY data/clean_dvf.csv /data/clean_dvf.csv

# Copier DB
COPY src/agentia/bdd/donnees_immo.db src/agentia/bdd/donnees_immo.db

# Note: HEALTHCHECK, EXPOSE, and CMD are overridden by K3S deployments
# - MCP: port 8001, command src/mcp_server/server.py
# - FastAPI: port 8000, command src/app/main.py
# - Gradio: port 7860, command src/agentia/main.py
# Each K3S deployment has its own livenessProbe and readinessProbe

# Default entry point (can be overridden)
CMD ["uv", "run", "python", "src/app/main.py"]