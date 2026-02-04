# ==================================
# Stage 1: Builder - Installation des dépendances
# ==================================
FROM python:3.11-slim as builder

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installer Poetry
ENV POETRY_VERSION=2.3.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HTTP_TIMEOUT=600 \
    POETRY_INSTALLER_MAX_WORKERS=1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copier les fichiers de configuration Poetry pour utiliser le cache Docker
COPY pyproject.toml poetry.lock ./

# Installer d'abord torch séparément avec pip (plus fiable pour les gros packages)
RUN pip install --no-cache-dir torch==2.10.0

# Installer les autres dépendances Python avec Poetry (en excluant torch)
RUN poetry install --only main --no-root --no-directory || \
    (echo "Première tentative échouée, nouvelle tentative..." && sleep 5 && poetry install --only main --no-root --no-directory)

# ==================================
# Stage 2: Runtime - Image finale légère
# ==================================
FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer uniquement le client PostgreSQL (pas gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances Python installées depuis le builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# S'assurer que les scripts Python sont dans le PATH
ENV PATH=/usr/local/bin:$PATH

# Copier le code de l'application
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p logs data

# Exposer le port Flask
EXPOSE 5000

# Créer un utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Changer vers l'utilisateur non-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/', timeout=5)" || exit 1

# Commande pour démarrer l'application
CMD ["python", "app.py"]
