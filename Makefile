# Makefile pour d:\projet_Mlopq
# Variables
PYTHON = python3
ENV_NAME = mlops_env
REQUIREMENTS = requirements.txt
MAIN = main.py
PROJECT_DIR = /mnt/d/projet_Mlopq

# Détecte le python de l'env virtuel (Unix/WSL ou Windows)
VENV_PY := $(shell if [ -f $(ENV_NAME)/bin/python ]; then echo $(ENV_NAME)/bin/python; elif [ -f $(ENV_NAME)/Scripts/python.exe ]; then echo $(ENV_NAME)/Scripts/python.exe; else echo $(PYTHON); fi)

.PHONY: help all setup install prepare train evaluate test lint lint-all format typecheck security ci notebook clean

all: help

help:
	@echo "Cibles disponibles :"
	@echo "  setup       -> créer venv et installer dépendances"
	@echo "  install     -> réinstaller les dépendances dans l'env"
	@echo "  prepare     -> préparer les données (main.py --prepare)"
	@echo "  train       -> entraîner le modèle (main.py --train)"
	@echo "  evaluate    -> évaluer le modèle (main.py --evaluate)"
	@echo "  test        -> lancer les tests (pytest)"
	@echo "  format      -> formater le code (black)"
	@echo "  lint        -> vérification qualité (flake8)"
	@echo "  lint-all    -> vérification complète (style, types, sécurité, qualité)"
	@echo "  typecheck   -> vérification types (mypy)"
	@echo "  security    -> vérifications sécurité (bandit/safety)"
	@echo "  ci          -> exécute la pipeline CI locale (format check, lint, tests, security)"
	@echo "  notebook    -> lancer jupyter notebook"
	@echo "  clean       -> supprimer artefacts et env"

# I. Installation / environnement
setup:
	@echo "Création de l'environnement virtuel..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@echo "Mise à jour pip et installation des dépendances..."
	@$(VENV_PY) -m pip install --upgrade pip
	@$(VENV_PY) -m pip install -r $(REQUIREMENTS)
	@echo "✅ Environnement créé : $(ENV_NAME)"
	@echo "Activation (Windows cmd) : $(ENV_NAME)\\Scripts\\activate"
	@echo "Activation (PowerShell) : .\\$(ENV_NAME)\\Scripts\\Activate.ps1"
	@echo "Activation (Unix/WSL) : source $(ENV_NAME)/bin/activate"

install:
	@echo "Installation des dépendances dans $(ENV_NAME)..."
	@$(VENV_PY) -m pip install -r $(REQUIREMENTS)

# II. Exécution des étapes liées au modèle (utilise main.py)
prepare:
	@echo "Préparation des données (main.py --prepare)..."
	@$(VENV_PY) $(MAIN) --prepare

train:
	@echo "Entraînement du modèle (main.py --train)..."
	@$(VENV_PY) $(MAIN) --train

evaluate:
	@echo "Évaluation du modèle (main.py --evaluate)..."
	@$(VENV_PY) $(MAIN) --evaluate

# III. CI / Qualité / Sécurité
format:
	@echo "Formatage automatique (black)..."
	@$(VENV_PY) -m black .

lint:
	@echo "Vérification qualité (flake8)..."
	@$(VENV_PY) -m flake8 .

# Lint global (style + qualité + sécurité + types + imports)
lint-all:
	@echo "💡 Vérification imports (isort)..."
	@$(VENV_PY) -m isort --check-only . || true
	@echo "💡 Vérification qualité code (flake8)..."
	@$(VENV_PY) -m flake8 . || true
	@echo "💡 Vérification style + erreurs (pylint)..."
	@$(VENV_PY) -m pylint . || true
	@echo "💡 Vérification types (mypy)..."
	@$(VENV_PY) -m mypy . || true
	@echo "💡 Vérification sécurité (bandit)..."
	@$(VENV_PY) -m bandit -r . || true
	@echo "💡 Vérification vulnérabilités dépendances (safety)..."
	@$(VENV_PY) -m safety check || true
	@echo "💡 Vérification complexité code (radon)..."
	@$(VENV_PY) -m radon cc . -s || true
	@$(VENV_PY) -m radon mi . || true
	@echo "💡 Vérification documentation (pydocstyle)..."
	@$(VENV_PY) -m pydocstyle . || true
	@echo "✅ Lint complet terminé"

typecheck:
	@echo "Vérification des types (mypy)..."
	@$(VENV_PY) -m mypy .

security:
	@echo "Vérifications sécurité (bandit / safety si installés)..."
	@$(VENV_PY) -m bandit -r . || true
	@$(VENV_PY) -m safety check || true

# CI : exécute les checks (non-destructifs)
ci:
	@echo "Exécution pipeline CI locale..."
	@$(VENV_PY) -m isort --check-only . || true
	@$(VENV_PY) -m flake8 . || true
	@$(VENV_PY) -m mypy . || true
	@$(VENV_PY) -m bandit -r . || true
	@$(VENV_PY) -m pytest -q || true
	@echo "CI locale terminée."

# Tests et notebook
test:
	@echo "Exécution des tests (pytest)..."
	@$(VENV_PY) -m pytest -q

notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@$(VENV_PY) -m jupyter notebook --notebook-dir=$(PROJECT_DIR)

# Nettoyage
clean:
	@echo "Nettoyage des artefacts..."
	@-rm -rf __pycache__ .pytest_cache .mypy_cache build dist *.egg-info
	@-rm -rf $(ENV_NAME)
	@echo "✅ Nettoyage terminé"

.PHONY: docker-lint

docker-lint:
	docker build -t mlops-lint .
	docker run --rm -v $(PWD):/app mlops-lint make lint-all


# IV. Lancer l'API FastAP

# Variables
APP = app:app
HOST = 0.0.0.0
PORT = 8000

# Commande pour lancer l'API
api:
	@echo "🚀 Démarrage de l'API FastAPI..."
	uvicorn $(APP) --reload --host $(HOST) --port $(PORT)

# Commande pour test health
health:
	curl http://127.0.0.1:$(PORT)/health

# Lancer l'application Streamlit
# Streamlit
STREAMLIT_APP = streamlit_app.py
STREAMLIT_PORT = 8501
streamlit:
	@echo "🚀 Démarrage de Streamlit..."
	streamlit run $(STREAMLIT_APP) --server.port $(STREAMLIT_PORT)


# Nom de l'image Docker
DOCKER_IMAGE = asma_raddaoui_ds6_mlops

# Construire l'image Docker
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# Lancer FastAPI via Docker
docker-run-fastapi:
	docker run -e SERVICE=fastapi -p 8000:8000 -v /mnt/d/projet_Mlopq/resultat:/app/resultat $(DOCKER_IMAGE)

# Lancer les tests via Docker
docker-test:
	docker run -e SERVICE=test $(DOCKER_IMAGE)

# Taguer l'image pour Docker Hub
docker-tag:
	docker tag $(DOCKER_IMAGE) ton_dockerhub_utilisateur/$(DOCKER_IMAGE):latest

# Pousser l'image sur Docker Hub

docker-push:
	docker push ton_dockerhub_utilisateur/$(DOCKER_IMAGE):latest
docker-run-streamlit:
	docker run \
		-e SERVICE=streamlit \
		-p 8501:8501 \
		--add-host=host.docker.internal:host-gateway \
		-v /mnt/d/projet_Mlopq/resultat:/app/resultat \
		$(DOCKER_IMAGE)


