# Statistical Audit Platform

Plateforme d'audit statistique automatisée — nettoyage de données, analyse multivariée (ACP, Analyse Factorielle) et interprétation par LLM via Claude (Anthropic).

Demo live : https://stat-audit-eliezer.streamlit.app
GitHub : https://github.com/eliezermoise63-glitch/stat-audit-platform

![CI](https://github.com/eliezermoise63-glitch/stat-audit-platform/actions/workflows/ci.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red.svg)
![Status](https://img.shields.io/badge/status-en%20développement-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Ce que fait ce projet

Upload d'un fichier CSV, nettoyage automatique des données, double analyse statistique et génération d'interprétations en langage naturel via un LLM.

| Etape | Module | Ce qui se passe |
|-------|--------|-----------------|
| 1. Ingestion | app.py | Upload CSV, détection auto du séparateur |
| 2. Fiabilisation | core/sanitizer.py | Suppression colonnes constantes, imputation médiane, suppression outliers z-score |
| 3. Corrélations | core/engine.py | Matrice de Pearson avec p-values, masquage non-significatif |
| 4. ACP | core/engine.py | Sélection automatique des composantes par variance cumulée, biplot, loadings |
| 5. Analyse Factorielle | core/engine.py | Validation KMO et Bartlett, critère de Kaiser, rotation Varimax, communautés |
| 6. Synthèse LLM | utils/llm.py | Prompts structurés vers Claude pour interprétation métier en langage naturel |

---

## Aperçu

![Audit et fiabilisation](assets/screenshot_audit.png)

![Analyse multivariée](assets/screenshot_analyse.png)

![Synthèse IA](assets/screenshot_ia.png)

---

## Fonctionnalités clés

- Nettoyage automatique : valeurs manquantes, outliers, variables non informatives
- ACP avec sélection automatique du nombre de composantes (seuil de variance configurable)
- Analyse Factorielle validée statistiquement (KMO, Bartlett) avec rotation Varimax
- Matrice de corrélation avec masquage automatique des corrélations non significatives
- Interprétation en langage naturel par Claude — séparation claire entre inférence statistique et LLM
- Interface interactive Streamlit avec configuration en temps réel (sidebar)
- 53 tests unitaires et d'intégration, CI/CD GitHub Actions

---

## Démarrage rapide

Prérequis : Python 3.10 ou supérieur. Clé API Anthropic optionnelle (onglet Synthèse IA uniquement).

```bash
git clone https://github.com/eliezermoise63-glitch/stat-audit-platform.git
cd stat-audit-platform

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
streamlit run app.py
```

Configuration de la clé API :

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Editez .streamlit/secrets.toml et ajoutez :
# ANTHROPIC_API_KEY = "sk-ant-votre-cle-ici"
```

---

## Architecture

```
stat-audit-platform/
├── app.py                      # Point d'entrée Streamlit (4 onglets)
├── core/
│   ├── sanitizer.py            # Pipeline de fiabilisation des données
│   └── engine.py               # Moteur ACP + Analyse Factorielle + Corrélations
├── utils/
│   ├── charts.py               # Visualisations matplotlib/seaborn
│   └── llm.py                  # Interface Claude (Anthropic)
├── tests/                      # 53 tests unitaires et d'intégration
└── .github/workflows/ci.yml    # CI GitHub Actions (Python 3.10 et 3.11)
```

---

## Détails techniques

**DataSanitizer**

```
DataFrame brut
    -> Sélection colonnes numériques
    -> Suppression colonnes constantes (nunique <= 1)
    -> Suppression colonnes quasi-constantes (ratio < 1%)
    -> Imputation médiane robuste
    -> Suppression lignes outliers (|z-score| > seuil configurable, défaut 3)
DataFrame fiabilisé + SanitizationReport traçable
```

**MultivariateEngine**

ACP : standardisation obligatoire (StandardScaler), sélection automatique des composantes par variance cumulée (seuil configurable, défaut 80%), biplot PC1 x PC2.

Analyse Factorielle : validation KMO (seuil 0.6) et test de Bartlett (p < 0.05), critère de Kaiser pour le choix automatique du nombre de facteurs, rotation Varimax pour la simplicité de structure, rapport de communautés par variable.

---

## Limites actuelles

| Limite | Impact | Priorité |
|--------|--------|----------|
| factor-analyzer incompatible avec scikit-learn >= 1.6 | Contournement en place (sklearn épinglé < 1.6) | Haute |
| Support CSV uniquement | XLSX, JSON, Parquet non supportés | Moyenne |
| Variables catégorielles ignorées | Perte d'information potentielle | Moyenne |
| Pas de cache Streamlit | Recalcul à chaque interaction | Moyenne |
| Rapport non exportable | Résultats non persistants | Basse |

---

## Roadmap

Court terme (v0.2) :
- Compatibilité scikit-learn >= 1.6
- Support XLSX et JSON
- Cache Streamlit (@st.cache_data)
- Export PDF du rapport

Moyen terme (v0.3) :
- Clustering post-ACP : K-Means sur les composantes principales, choix automatique de K par méthode du coude et silhouette score, visualisation des clusters
- ACM pour les variables catégorielles
- KNN Imputer en alternative à la médiane
- Isolation Forest en complément du z-score

Long terme (v1.0) :
- DBSCAN pour clusters de forme arbitraire
- Rapport automatique avec interprétation LLM incluse
- Comparaison de datasets avant/après traitement
- Tests de normalité intégrés (Shapiro-Wilk, Kolmogorov-Smirnov)

---

## Datasets de test

| Dataset | Lien | Usage |
|---------|------|-------|
| Iris | https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv | ACP classique |
| Boston | https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv | Analyse factorielle |
| Heart | https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv | Robustesse du sanitizer |

---

## Tests

```bash
pytest tests/ -v
pytest tests/ --cov=core --cov=utils --cov-report=term-missing
```

---

## Pitch (30 secondes)

"J'ai développé une plateforme d'audit statistique automatisée. Elle commence par un module de fiabilisation qui gère les valeurs manquantes, les outliers et les variables non informatives. Ensuite, elle applique une double analyse : ACP avec sélection automatique des composantes, et analyse factorielle validée par KMO et Bartlett avec rotation Varimax. La prochaine étape majeure est un module de clustering K-Means sur les composantes ACP. Enfin, Claude traduit ces mathématiques en recommandations métier, avec une séparation claire entre inférence statistique et interprétation sémantique."

---

## Licence

MIT - Eliezer Moise