FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

# Dépendances principales
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn==1.3.2 fastapi uvicorn mlflow

# Copier requirements.txt
COPY requirements.txt .

# Installer dépendances supplémentaires
RUN pip install --no-cache-dir -r requirements.txt

# Créer répertoire résultat
RUN mkdir -p /app/resultat

# Copier tout le code
COPY . .

# Copier et rendre exécutable le script entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["/entrypoint.sh"]
