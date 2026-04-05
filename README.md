# Statistical Audit Platform

Projet en cours de développement — Data Science & Statistiques Appliquées.
Un outil d'audit statistique automatisé, d'analyse multivariée et d'interprétation par LLM.

![CI](https://github.com/eliezermoise63-glitch/stat-audit-platform/actions/workflows/ci.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red.svg)
![Status](https://img.shields.io/badge/status-en%20développement-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Aperçu

![Audit et fiabilisation](assets/screenshot_audit.png)

![Analyse multivariée](assets/screenshot_analyse.png)

![Synthèse IA](assets/screenshot_ia.png)

---

## Statut du projet

Ce projet est en cours de développement actif. Les fonctionnalités core sont opérationnelles, mais plusieurs axes d'amélioration sont identifiés et documentés ci-dessous.

---

## Ce que fait ce projet

| Etape | Module | Ce qui se passe |
|-------|--------|-----------------|
| 1. Ingestion | app.py | Upload CSV, détection auto du séparateur |
| 2. Fiabilisation | core/sanitizer.py | Suppression colonnes constantes, imputation médiane, suppression outliers (z-score) |
| 3. Corrélations | core/engine.py | Matrice de Pearson + p-values, masquage non-significatif |
| 4. ACP | core/engine.py | Sélection auto des composantes (variance threshold), biplot, loadings |
| 5. Analyse Factorielle | core/engine.py | Kaiser + Varimax, validation KMO & Bartlett, communautés |
| 6. Synthèse LLM | utils/llm.py | Prompts structurés vers Claude (Anthropic) pour interprétation métier |

---

## Démarrage rapide

Prérequis : Python 3.10 ou supérieur. Une clé API Anthropic est optionnelle (uniquement pour l'onglet Synthèse IA).

**1. Cloner et installer**

```bash
git clone https://github.com/eliezermoise63-glitch/stat-audit-platform.git
cd stat-audit-platform

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Note de compatibilité : scikit-learn est épinglé à < 1.6 en raison d'un conflit avec factor-analyzer. Ce point est sur la roadmap de correction.

**2. Configurer la clé API (optionnel)**

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Editez `.streamlit/secrets.toml` et ajoutez votre clé :

```toml
ANTHROPIC_API_KEY = "sk-ant-votre-cle-ici"
```

Sans clé : les 3 premiers onglets fonctionnent normalement. Seul l'onglet Synthèse IA est désactivé.

**3. Lancer**

```bash
streamlit run app.py
```

---

## Architecture du projet

```
stat-audit-platform/
│
├── app.py                      # Point d'entrée Streamlit (4 onglets)
├── core/
│   ├── sanitizer.py            # Pipeline de fiabilisation des données
│   └── engine.py               # Moteur ACP + Analyse Factorielle + Corrélations
├── utils/
│   ├── charts.py               # Visualisations matplotlib/seaborn
│   └── llm.py                  # Interface Claude (Anthropic)
├── tests/                      # 53 tests unitaires et d'intégration
├── .github/workflows/ci.yml    # CI GitHub Actions (Python 3.10 et 3.11)
└── requirements.txt
```

---

## Détails techniques

**DataSanitizer**

```
DataFrame brut
    -> Sélection colonnes numériques
    -> Suppression colonnes constantes (nunique <= 1)
    -> Suppression colonnes quasi-constantes (ratio < 1%)
    -> Imputation médiane (robuste aux outliers)
    -> Suppression lignes outliers (|z-score| > seuil, défaut = 3)
DataFrame fiabilisé + SanitizationReport
```

**MultivariateEngine**

ACP : standardisation obligatoire, sélection automatique des composantes par variance cumulée, biplot PC1 x PC2.

Analyse Factorielle : validation KMO et Bartlett, critère de Kaiser pour le choix automatique du nombre de facteurs, rotation Varimax, rapport de communautés par variable.

Corrélations : Pearson avec p-values, masquage automatique des corrélations non significatives (p >= 0.05).

---

## Limites actuelles

| Limite | Impact | Priorité |
|--------|--------|----------|
| factor-analyzer incompatible avec scikit-learn >= 1.6 | Contournement en place (sklearn épinglé) | Haute |
| Support CSV uniquement | XLSX, JSON, Parquet non supportés | Moyenne |
| Variables catégorielles ignorées | Perte d'information potentielle | Moyenne |
| Pas de cache Streamlit | Recalcul à chaque interaction | Moyenne |
| Rapport non exportable | Résultats non persistants | Basse |

---

## Roadmap

Court terme (v0.2) :
- Compatibilité scikit-learn >= 1.6
- Support XLSX et JSON
- Cache Streamlit
- Export PDF du rapport

Moyen terme (v0.3) :
- Clustering post-ACP : K-Means sur les composantes principales, choix automatique de K par méthode du coude et silhouette score
- ACM pour les variables catégorielles
- KNN Imputer en alternative à la médiane
- Isolation Forest en complément du z-score

Long terme (v1.0) :
- DBSCAN pour clusters de forme arbitraire
- Rapport automatique avec interprétation LLM incluse
- Comparaison de datasets
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

## Déploiement sur Streamlit Cloud

1. Pusher ce dépôt sur GitHub
2. Aller sur share.streamlit.io
3. Create app -> sélectionner ce repo -> fichier principal : app.py
4. Advanced settings -> Secrets -> ajouter ANTHROPIC_API_KEY
5. Deploy

---

## Pitch (30 secondes)

"J'ai développé une plateforme d'audit statistique automatisée, encore en construction mais déjà fonctionnelle. Elle commence par un module de fiabilisation qui gère les valeurs manquantes, les outliers et les variables non informatives. Ensuite, elle applique une double analyse : ACP avec sélection automatique des composantes, et analyse factorielle validée par KMO et Bartlett avec rotation Varimax. La prochaine étape majeure est un module de clustering K-Means sur les composantes ACP. Enfin, Claude traduit ces mathématiques en recommandations métier, avec une séparation claire entre inférence statistique et interprétation sémantique."

---

## Licence

MIT - Eliezer Moise