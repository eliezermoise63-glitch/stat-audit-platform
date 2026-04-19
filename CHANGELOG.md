# Changelog

Toutes les modifications notables de ce projet sont documentées ici.
Format : [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/)

---

## [0.2.0] — 2026-04-19

### Ajouté
- **VariableDetector** : détection automatique du type de chaque variable
  (continue, catégorielle, binaire, ignorée) avec seuil configurable
- **Prompts LLM pour ACM et AFDM** : `build_acm_prompt` et `build_afdm_prompt` dans `utils/llm.py`
- **ACM** (`run_acm`) : Analyse des Correspondances Multiples via `prince.MCA`
  pour les variables catégorielles
- **AFDM** (`run_afdm`) : Analyse Factorielle des Données Mixtes via `prince.FAMD`
  pour les datasets mixtes (continues + catégorielles)
- **Rotation Promax** : alternative à Varimax pour l'Analyse Factorielle
  (facteurs obliques, adaptée aux sciences humaines et sociales)
- **Corrélation automatique Pearson/Spearman** : test de normalité Shapiro-Wilk
  sur chaque variable, choix automatique de la méthode
- **Top-N variables par composante ACP** (`top_variables_per_component`)
- **Bandeau de détection** dans l'onglet Ingestion : résumé des types détectés
- **Sidebar v0.2.0** : nouveau slider "Seuil catégoriel" pour ajuster la détection

### Corrigé
- Bug silencieux `np.argmax` dans `run_pca` : retournait `n_components=1`
  quand le seuil de variance n'était pas atteignable — remplacé par `np.where`
- `applymap` déprécié (pandas 2.1+) → remplacé par `map` dans `charts.py`
- Import `anthropic` inline dans `call_claude` → remonté en tête de fichier
- Import dupliqué `MultivariateEngine` dans `tab_ingestion` de `app.py`
- Imports `tempfile` et `sqlalchemy` déplacés en tête de `app.py`
- Ratio `nunique` dans le sanitizer calculé sur valeurs non-nulles après imputation
- Imputation déplacée avant la détection des quasi-constantes dans le sanitizer
- Constante `PALETTE_SEQUENTIAL` inutilisée supprimée de `charts.py`

### Modifié
- `MultivariateEngine` initialisé uniquement sur les variables continues détectées
- `tab_multivariate` restructurée : affichage conditionnel selon le type de dataset
- Sous-titre app mis à jour : `ACP · AF · ACM · AFDM · Interprétation par LLM`
- Version bumped `v0.1.0` → `v0.2.0`

### Infrastructure
- `prince>=0.7.1,<1.0.0` ajouté dans `requirements.txt`
- `core/__init__.py` mis à jour : exports `VariableDetector`, `DetectionReport`,
  `ACMResult`, `AFDMResult`
- `utils/__init__.py` mis à jour : export `build_pca_prompt`

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

## [À venir] — 0.3.0

- Export rapport PDF
- Support fichiers XLSX
- Clustering post-ACP (K-Means, CAH)
- Cache Streamlit (`@st.cache_data`)
- Prompts LLM pour ACM et AFDM
- Intégration module dans la plateforme parente
