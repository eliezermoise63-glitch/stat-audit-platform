"""
pages/1_About.py
----------------
Page "À propos" — architecture, crédits, FAQ.
"""

import streamlit as st

st.set_page_config(page_title="À propos — Stat Audit Platform", page_icon="ℹ️", layout="wide")

st.title("ℹ️ À propos du projet")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ##  Architecture technique

    Ce projet est un **sous-module indépendant** d'une plateforme Data Science
    plus large. Il peut être utilisé de façon autonome ou intégré comme composant.

    ### Pipeline de traitement

    ```
    CSV / SQLite / URL
        ↓
    DataSanitizer          ← core/sanitizer.py
        │  • Suppression colonnes constantes
        │  • Imputation médiane (robuste)
        │  • Suppression outliers (z-score)
        ↓
    VariableDetector       ← core/detector.py
        │  • Détection : continue / catégorielle / binaire / ignorée
        │  • Seuil catégoriel configurable
        ↓
    MultivariateEngine     ← core/engine.py
        │  • ACP (variance threshold auto) — variables continues
        │  • AF (Kaiser + Varimax/Promax + KMO/Bartlett) — continues
        │  • ACM (Analyse des Correspondances Multiples) — catégorielles
        │  • AFDM (Analyse Factorielle Données Mixtes) — dataset mixte
        │  • Corrélations Pearson/Spearman (Shapiro-Wilk auto)
        ↓
    Visualisations         ← utils/charts.py
        │  • Heatmap corrélation
        │  • Biplot ACP · Scree plot · Loadings heatmap
        ↓
    Interprétation LLM     ← utils/llm.py
           • Prompts structurés → Claude (Anthropic)
           • Nommage des facteurs latents (AF)
           • Interprétation des axes (ACP, ACM, AFDM)
           • Recommandations métier
    ```

    ---

    ##  Choix techniques justifiés

    | Choix | Raison |
    |-------|--------|
    | **Imputation médiane** | Robuste aux outliers (vs moyenne) |
    | **Z-score ≥ 3** | Seuil standard ; configurable via sidebar |
    | **Détection automatique des types** | ACP/AF inadaptées aux catégorielles |
    | **Variance threshold 80%** | Compromis information/parsimonie ; ajustable |
    | **Critère de Kaiser** | Standard psychométrique (valeurs propres > 1) |
    | **Varimax / Promax** | Varimax : facteurs indépendants. Promax : oblique (SHS) |
    | **KMO + Bartlett** | Validation obligatoire avant AF |
    | **Pearson/Spearman auto** | Shapiro-Wilk → méthode adaptée à la distribution |
    | **ACM via prince** | Standard pour variables qualitatives |
    | **AFDM via prince** | Généralise ACP + ACM aux datasets mixtes |
    | **Séparation stats/LLM** | Inférence ≠ interprétation — architecture propre |
    """)

with col2:
    st.markdown("""
    ## FAQ

    **Q : Pourquoi ACP *et* AF ?**

    > L'ACP maximise la variance (réduction de dimension).
    > L'AF modélise les variables latentes (interprétation causale).
    > Les deux sont complémentaires.

    ---

    **Q : Pourquoi standardiser ?**

    > L'ACP dépend des variances. Sans standardisation,
    > les variables à grande échelle dominent artificiellement.

    ---

    **Q : Quand l'AF est-elle invalide ?**

    > KMO < 0.6 → variables trop peu corrélées.
    > Bartlett p > 0.05 → matrice de corrélation = identité.
    > Dans les deux cas, l'AF n'a pas de sens.

    ---

    **Q : Varimax ou Promax ?**

    > Varimax (orthogonale) : facteurs indépendants.
    > Promax (oblique) : facteurs corrélés — plus réaliste
    > en sciences humaines et sociales.

    ---

    **Q : Quelle différence entre ACM et AFDM ?**

    > ACM : uniquement des variables catégorielles.
    > AFDM : dataset mixte (continues + catégorielles).
    > Les deux produisent un espace factoriel commun.

    ---

    **Q : Que fait le LLM exactement ?**

    > Il reçoit les résultats statistiques (loadings, coordonnées,
    > inertie) et les *interprète* en langage métier.
    > Il ne fait **pas** d'inférence statistique.
    """)

st.markdown("---")
st.markdown("""
##  Stack technique

| Composant | Bibliothèque | Version |
|-----------|-------------|---------|
| UI | Streamlit | ≥ 1.30 |
| Data | Pandas, NumPy | ≥ 2.0, ≥ 1.24 |
| ACP / AF | scikit-learn, factor-analyzer | ≥ 1.3, ≥ 0.4.1 |
| ACM / AFDM | prince | ≥ 0.7.1 |
| Stats | SciPy, statsmodels | ≥ 1.11, ≥ 0.14 |
| Viz | Matplotlib, Seaborn | ≥ 3.7, ≥ 0.12 |
| LLM | Anthropic (Claude) | ≥ 0.25 |

---
*Statistical Audit Platform v0.2.0 — Eliezer Moïse*
""")
