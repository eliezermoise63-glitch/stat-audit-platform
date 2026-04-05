# Changelog

Toutes les modifications notables de ce projet sont documentées ici.
Format : [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)

---

## [0.1.0] — 2025-04-04

### Ajouté
- **DataSanitizer** : pipeline de fiabilisation en 4 étapes
  (colonnes constantes, quasi-constantes, imputation médiane, suppression outliers z-score)
- **MultivariateEngine** : ACP avec sélection automatique de composantes,
  Analyse Factorielle (Kaiser + Varimax), validation KMO + Bartlett
- **Interface Streamlit** : 4 onglets (Audit, Corrélations, Multidimensionnel, IA)
- **Prompts Claude** : interprétation métier des facteurs latents et des composantes
- **Visualisations** : heatmap corrélation, biplot ACP, scree plot, heatmap loadings
- **Tests unitaires** : 24 tests (sanitizer + engine)
- **CI GitHub Actions** : pytest + flake8 sur Python 3.10 et 3.11
- **Rapport de sanitisation** (`SanitizationReport`) avec métriques traçables

### Infrastructure
- `requirements.txt` avec versions épinglées
- `.gitignore` sécurisé (secrets.toml exclu)
- `GITHUB_SETUP.md` : guide de déploiement pas-à-pas
- `validate_structure.py` : script de vérification locale

---

## [À venir] — 0.2.0

- Export rapport PDF
- Support fichiers XLSX
- ACM pour variables catégorielles
- Clustering post-ACP (K-Means)
- Cache Streamlit (`@st.cache_data`)
- Intégration module dans la plateforme parente
