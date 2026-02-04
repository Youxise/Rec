.PHONY: help install update test run docker-build docker-up docker-down clean

help:  ## Afficher cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Installer les dépendances avec Poetry
	poetry install

update:  ## Mettre à jour les dépendances
	poetry update

test:  ## Exécuter les tests
	poetry run pytest tests/ -v --cov=src --cov-report=html

run:  ## Lancer l'application en local
	poetry run python app.py

docker-build:  ## Construire les images Docker
	docker-compose build

docker-up:  ## Démarrer les conteneurs Docker
	docker-compose up -d

docker-down:  ## Arrêter les conteneurs Docker
	docker-compose down

docker-logs:  ## Voir les logs des conteneurs
	docker-compose logs -f

clean:  ## Nettoyer les fichiers temporaires
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
