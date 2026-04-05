# ============================================================
# Makefile — Statistical Audit Platform
# Usage : make <target>
# ============================================================

.PHONY: help install run test test-cov lint validate clean

# Couleurs
GREEN  := \033[0;32m
YELLOW := \033[1;33m
NC     := \033[0m  # No Color

help:           ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'

install:        ## Installe les dépendances
	@echo "$(YELLOW)Installation des dépendances...$(NC)"
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

run:            ## Lance l'application Streamlit
	@echo "$(YELLOW)Démarrage de l'application...$(NC)"
	streamlit run app.py

test:           ## Lance tous les tests unitaires
	@echo "$(YELLOW)Lancement des tests...$(NC)"
	pytest tests/ -q

test-cov:       ## Tests avec rapport de couverture
	pytest tests/ --cov=core --cov=utils --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Rapport HTML disponible dans htmlcov/index.html$(NC)"

test-fast:      ## Tests rapides (sans intégration)
	pytest tests/ -q -m "not integration"

lint:           ## Vérifie la qualité du code
	@pip install flake8 --quiet
	flake8 core/ utils/ app.py --max-line-length=120 --ignore=E501,W503,E203

validate:       ## Vérifie l'installation locale
	python utils/validate_structure.py

clean:          ## Nettoie les fichiers temporaires
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ htmlcov/ .coverage 2>/dev/null || true
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

setup-secrets:  ## Crée le fichier secrets depuis le template
	@if [ ! -f .streamlit/secrets.toml ]; then \
		cp .streamlit/secrets.toml.example .streamlit/secrets.toml; \
		echo "$(YELLOW)⚠️  Éditez .streamlit/secrets.toml et ajoutez votre clé Anthropic$(NC)"; \
	else \
		echo "$(GREEN)✅ .streamlit/secrets.toml existe déjà$(NC)"; \
	fi
