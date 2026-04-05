# 📊 Statistical Audit Platform

> **Projet en cours de développement — Data Science & Statistiques Appliquées**
> Un outil d'audit statistique automatisé, d'analyse multivariée et d'interprétation par LLM.

[![CI](https://github.com/eliezermoise/stat-audit-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/eliezermoise/stat-audit-platform/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red.svg)](https://streamlit.io)
[![Status](https://img.shields.io/badge/status-en%20développement-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ⚠️ Statut du projet

Ce projet est **en cours de développement actif**. Les fonctionnalités core sont opérationnelles, mais plusieurs axes d'amélioration sont identifiés et documentés ci-dessous. Les contributions et retours sont bienvenus.

---

## 🎯 Ce que fait ce projet

| Étape | Module | Ce qui se passe |
|-------|--------|-----------------|
| **1. Ingestion** | `app.py` | Upload CSV, détection auto du séparateur |
| **2. Fiabilisation** | `core/sanitizer.py` | Suppression colonnes constantes, imputation médiane, suppression outliers (z-score) |
| **3. Corrélations** | `core/engine.py` | Matrice de Pearson + p-values, masquage non-significatif |
| **4. ACP** | `core/engine.py` | Sélection auto des composantes (variance threshold), biplot, loadings |
| **5. Analyse Factorielle** | `core/engine.py` | Kaiser + Varimax, validation KMO & Bartlett, communautés |
| **6. Synthèse LLM** | `utils/llm.py` | Prompts structurés → Claude (Anthropic) → interprétation métier |

---

## 🚀 Démarrage rapide

### Prérequis
- Python ≥ 3.10
- Une clé API Anthropic (optionnel — uniquement pour l'onglet Synthèse IA)

### 1. Cloner et installer

```bash
git clone https://github.com/eliezermoise/stat-audit-platform.git
cd stat-audit-platform

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

> **Note compatibilité :** `scikit-learn` est épinglé à `< 1.6` en raison d'un conflit
> avec `factor-analyzer` (paramètre `force_all_finite` renommé dans sklearn 1.6).
> Ce point est sur la roadmap de correction.

### 2. Configurer la clé API (optionnel)

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Éditer .streamlit/secrets.toml et y mettre votre clé
```

```toml
ANTHROPIC_API_KEY = "sk-ant-votre-cle-ici"
```

Sans clé : les 3 premiers onglets fonctionnent normalement. Seul **Synthèse IA** est désactivé.

### 3. Lancer

```bash
streamlit run app.py
```

→ [http://localhost:8501](http://localhost:8501)

---

## 🏗️ Architecture du projet

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
├── .github/workflows/ci.yml   # CI GitHub Actions (Python 3.10 & 3.11)
└── requirements.txt
```

---

## 📐 Détails techniques

### DataSanitizer

```
DataFrame brut
    ↓ Sélection colonnes numériques
    ↓ Suppression colonnes constantes (nunique ≤ 1)
    ↓ Suppression colonnes quasi-constantes (ratio < 1%)
    ↓ Imputation médiane (robuste aux outliers)
    ↓ Suppression lignes outliers (|z-score| > seuil, défaut = 3)
DataFrame fiabilisé + SanitizationReport
```

### MultivariateEngine

**ACP :** Standardisation → sélection auto des composantes (variance cumulée ≥ seuil) → biplot

**Analyse Factorielle :** Validation KMO + Bartlett → critère de Kaiser → rotation Varimax → communautés

**Corrélations :** Pearson + p-values → masquage automatique des corrélations non significatives (p ≥ 0.05)

---

## ⚠️ Limites actuelles connues

| Limite | Impact | Priorité |
|--------|--------|----------|
| `factor-analyzer` incompatible avec `scikit-learn >= 1.6` | Contournement en place (sklearn épinglé) | 🔴 Haute |
| Support CSV uniquement | XLSX, JSON, Parquet non supportés | 🟡 Moyenne |
| Variables catégorielles ignorées | Perte d'information potentielle | 🟡 Moyenne |
| Pas de cache Streamlit (`@st.cache_data`) | Recalcul à chaque interaction | 🟡 Moyenne |
| Rapport non exportable | Résultats non persistants | 🟢 Basse |

---

## 🗺️ Roadmap — Développements prévus

### Court terme (v0.2)
- [ ] Compatibilité `scikit-learn >= 1.6` — migration vers `ensure_all_finite`
- [ ] Support XLSX et JSON en plus du CSV
- [ ] Cache Streamlit (`@st.cache_data`)
- [ ] Export PDF du rapport d'audit

### Moyen terme (v0.3)
- [ ] **Clustering post-ACP** — K-Means sur les composantes principales, choix automatique de K (méthode du coude + silhouette score), visualisation des clusters
- [ ] **ACM** (Analyse des Correspondances Multiples) pour les variables catégorielles
- [ ] **KNN Imputer** en alternative à la médiane
- [ ] **Détection d'anomalies** — Isolation Forest en complément du z-score

### Long terme (v1.0)
- [ ] **DBSCAN** pour clusters de forme arbitraire
- [ ] Rapport automatique markdown/PDF avec interprétation LLM incluse
- [ ] Comparaison de datasets (avant/après traitement)
- [ ] Tests de normalité (Shapiro-Wilk, Kolmogorov-Smirnov) intégrés

---

## 💡 Datasets de test

| Dataset | Lien | Usage |
|---------|------|-------|
| Iris | [CSV](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) | ACP classique |
| Tips | [CSV](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) | Analyse factorielle |
| Titanic | [CSV](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) | Robustesse du sanitizer |

---

## 🧪 Tests

```bash
pytest tests/ -v
pytest tests/ --cov=core --cov=utils --cov-report=term-missing
```

---

## 🌐 Déploiement sur Streamlit Cloud

1. Pusher ce dépôt sur GitHub
2. [share.streamlit.io](https://share.streamlit.io) → **Create app** → `app.py`
3. **Advanced settings → Secrets** → ajouter `ANTHROPIC_API_KEY`
4. **Deploy** ✅

---

## 🎤 Pitch (30 secondes)

> *"J'ai développé une plateforme d'audit statistique automatisée, encore en construction mais déjà fonctionnelle. Elle commence par un module de fiabilisation qui gère les valeurs manquantes, les outliers et les variables non informatives. Ensuite, elle applique une double analyse : ACP avec sélection automatique des composantes, et analyse factorielle validée par KMO et Bartlett avec rotation Varimax. La prochaine étape majeure est un module de clustering K-Means sur les composantes ACP. Enfin, Claude traduit ces mathématiques en recommandations métier, avec une séparation claire entre inférence statistique et interprétation sémantique."*

---

## 📄 Licence

MIT © Eliezer Moïse
