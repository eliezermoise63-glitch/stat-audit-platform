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
    ## 📐 Architecture technique

    Ce projet est un **sous-module indépendant** d'une plateforme Data Science
    plus large. Il peut être utilisé de façon autonome ou intégré comme composant.

    ### Pipeline de traitement

    ```
    CSV Upload
        ↓
    DataSanitizer          ← core/sanitizer.py
        │  • Suppression colonnes constantes
        │  • Imputation médiane (robuste)
        │  • Suppression outliers (z-score)
        ↓
    MultivariateEngine     ← core/engine.py
        │  • ACP (variance threshold auto)
        │  • AF (Kaiser + Varimax + KMO/Bartlett)
        │  • Corrélations (Pearson + p-values)
        ↓
    Visualisations         ← utils/charts.py
        │  • Heatmap corrélation
        │  • Biplot ACP
        │  • Scree plot + loadings heatmap
        ↓
    Interprétation LLM     ← utils/llm.py
           • Prompts structurés → Claude (Anthropic)
           • Nommage des facteurs latents
           • Recommandations métier
    ```

    ---

    ## 🔬 Choix techniques justifiés

    | Choix | Raison |
    |-------|--------|
    | **Imputation médiane** | Robuste aux outliers (vs moyenne) |
    | **Z-score ≥ 3** | Seuil standard ; configurable via sidebar |
    | **Variance threshold 80%** | Compromis information/parsimonie ; ajustable |
    | **Critère de Kaiser** | Standard psychométrique (valeurs propres > 1) |
    | **Rotation Varimax** | Simplifie la structure → interprétabilité maximale |
    | **KMO + Bartlett** | Validation obligatoire avant AF |
    | **Séparation stats/LLM** | Inférence ≠ interprétation — architecture propre |
    """)

with col2:
    st.markdown("""
    ##  FAQ 

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

    **Q : Pourquoi Varimax ?**

    > Pour maximiser la variance des saturations.
    > Résultat : chaque variable sature fort sur *un* facteur.
    > → Interprétation plus claire.

    ---

    **Q : Que fait le LLM exactement ?**

    > Il reçoit les loadings + KMO + communautés
    > et *nomme* les facteurs latents en termes métier.
    > Il ne fait **pas** d'inférence statistique.
    """)

st.markdown("---")
st.markdown("""
## 📦 Stack technique

| Composant | Bibliothèque | Version |
|-----------|-------------|---------|
| UI | Streamlit | ≥ 1.30 |
| Data | Pandas, NumPy | ≥ 2.0, ≥ 1.24 |
| ACP | scikit-learn | ≥ 1.3 |
| AF | factor-analyzer | ≥ 0.4.1 |
| Stats | SciPy, statsmodels | ≥ 1.11, ≥ 0.14 |
| Viz | Matplotlib, Seaborn | ≥ 3.7, ≥ 0.12 |
| LLM | Anthropic (Claude) | ≥ 0.25 |

---
*Statistical Audit Platform v0.1.0 — Eliezer Moïse*
""")
