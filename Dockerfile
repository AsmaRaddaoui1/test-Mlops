# Utilise une image Python légère
FROM python:3.11-slim

# Met à jour le système et installe les dépendances nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements ou Makefile
COPY requirements.txt ./


# Installer les outils de lint globalement
RUN pip install --no-cache-dir black isort flake8 pylint mypy bandit safety radon pydocstyle

# Copier le reste du code
COPY . .

# Commande par défaut pour entrer dans le conteneur
CMD ["/bin/bash"]
