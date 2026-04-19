"""
app.py
------
Point d'entrée de la plateforme d'Audit & Analyse Statistique.
Lance avec : streamlit run app.py
"""

import logging
import tempfile
import traceback

import numpy as np
import pandas as pd
import sqlalchemy
import streamlit as st

from core.sanitizer import DataSanitizer
from core.engine import MultivariateEngine
from core.detector import VariableDetector, DetectionReport
from utils.charts import (
    plot_correlation_heatmap,
    plot_fa_loadings_heatmap,
    plot_pca_biplot,
    plot_pca_variance,
    plot_scree,
)
from utils.llm import build_fa_prompt, build_pca_prompt, build_acm_prompt, build_afdm_prompt, call_claude

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Stat Audit Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .warning-box { background: #fff3cd; border-left: 4px solid #ffc107;
                   padding: 12px; border-radius: 4px; margin: 8px 0; }
    .info-box    { background: #d1ecf1; border-left: 4px solid #17a2b8;
                   padding: 12px; border-radius: 4px; margin: 8px 0; }
    </style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """
    Affiche la sidebar de configuration et retourne les paramètres choisis.

    Retourne
    --------
    dict
        Paramètres : zscore_threshold, imputation, categorical_threshold,
        variance_threshold, n_factors, rotation, user_context.
    """
    with st.sidebar:
        st.title("Configuration")
        st.markdown("---")

        st.subheader("Sanitizer")
        zscore_threshold = st.slider(
            "Seuil z-score (outliers)", 2.0, 5.0, 3.0, 0.5,
            help="Lignes dont le z-score dépasse ce seuil seront supprimées.",
        )
        imputation = st.selectbox(
            "Stratégie d'imputation", ["median", "mean", "most_frequent"],
            help="Méthode pour remplacer les valeurs manquantes.",
        )

        st.markdown("---")
        st.subheader("Détection des variables")
        categorical_threshold = st.slider(
            "Seuil catégoriel (%)", 1, 20, 5, 1,
            help=(
                "Une variable numérique est traitée comme catégorielle si le ratio "
                "n_unique / n_lignes est inférieur à ce seuil. "
                "Augmentez si vos variables catégorielles ont beaucoup de modalités."
            ),
        ) / 100

        st.markdown("---")
        st.subheader("ACP")
        variance_threshold = st.slider(
            "Variance cumulée cible (%)", 60, 95, 80, 5,
            help="Nombre de composantes retenues pour atteindre ce % de variance.",
        ) / 100

        st.markdown("---")
        st.subheader("Analyse Factorielle")
        n_factors_mode = st.radio(
            "Choix du nombre de facteurs", ["Automatique (Kaiser)", "Manuel"]
        )
        n_factors_manual = None
        if n_factors_mode == "Manuel":
            n_factors_manual = st.number_input("Nombre de facteurs", 1, 10, 3)

        rotation = st.radio(
            "Rotation",
            ["varimax", "promax"],
            help=(
                "Varimax (orthogonale) : les facteurs sont supposés indépendants. "
                "Promax (oblique) : les facteurs peuvent être corrélés entre eux — "
                "plus adapté en sciences humaines et sociales."
            ),
        )

        st.markdown("---")
        st.subheader("LLM")
        user_context = st.text_area(
            "Contexte métier (optionnel)",
            placeholder="Ex : données RH d'une entreprise industrielle...",
            height=80,
        )

        st.markdown("---")
        st.caption("Statistical Audit Platform v0.2.0")

    return {
        "zscore_threshold": zscore_threshold,
        "imputation": imputation,
        "categorical_threshold": categorical_threshold,
        "variance_threshold": variance_threshold,
        "n_factors": n_factors_manual,
        "rotation": rotation,
        "user_context": user_context,
    }


# ── Helpers UI ────────────────────────────────────────────────────────────────

def get_api_key() -> str | None:
    """Récupère la clé API Anthropic depuis les secrets Streamlit."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


def show_error(msg: str, exception: Exception | None = None) -> None:
    """Affiche un message d'erreur avec stacktrace optionnelle."""
    st.error(f"Erreur : {msg}")
    if exception and st.checkbox("Afficher les détails techniques", key=f"err_{hash(msg)}"):
        st.code(traceback.format_exc())


def show_kmo_badge(kmo: float) -> None:
    """Affiche un badge coloré indiquant la qualité du KMO."""
    if kmo is None or (isinstance(kmo, float) and np.isnan(kmo)):
        label, color = "Non calculable", "#6c757d"
    elif kmo >= 0.8:
        label, color = "Excellent", "#28a745"
    elif kmo >= 0.7:
        label, color = "Bon", "#17a2b8"
    elif kmo >= 0.6:
        label, color = "Acceptable", "#fd7e14"
    else:
        label, color = "Insuffisant", "#dc3545"
    kmo_display = f"{kmo:.3f}" if kmo is not None and not (isinstance(kmo, float) and np.isnan(kmo)) else "n/a"
    st.markdown(
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-weight:bold;">KMO {kmo_display} — {label}</span>',
        unsafe_allow_html=True,
    )


def show_detection_summary(detection: DetectionReport) -> None:
    """
    Affiche un bandeau résumant la détection automatique des types de variables
    et les méthodes qui seront appliquées.
    """
    col_c, col_cat, col_ign = st.columns(3)
    col_c.metric("Variables continues", len(detection.continues),
                 help="→ ACP + Analyse Factorielle")
    col_cat.metric("Variables catégorielles", len(detection.categorielles),
                   help="→ ACM (Analyse des Correspondances Multiples)")
    col_ign.metric("Variables ignorées", len(detection.ignorees),
                   help="Non numériques ou quasi-constantes")

    if detection.is_mixed:
        st.markdown(
            '<div class="info-box">Dataset mixte détecté — '
            "ACP + AF (continues) · ACM (catégorielles) · AFDM (ensemble)</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Détail par variable"):
        type_labels = {
            "continue": "🔵 Continue",
            "categorielle": "🟢 Catégorielle",
            "binaire": "🟡 Binaire",
            "ignoree": "⚫ Ignorée",
        }
        rows = [
            {"Variable": col, "Type détecté": type_labels.get(t, t)}
            for col, t in detection.types.items()
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Variable"), use_container_width=True)


# ── Onglets ───────────────────────────────────────────────────────────────────

def tab_ingestion(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    report,
    detection: DetectionReport,
) -> None:
    """Affiche le rapport de fiabilisation, la détection des types et les stats descriptives."""
    st.subheader("Rapport de Fiabilité")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Lignes (entrée)", report.n_rows_input)
    c2.metric("Lignes (sortie)", report.n_rows_output,
              delta=f"-{report.outliers_removed} outliers")
    c3.metric("Colonnes conservées", report.n_cols_output,
              delta=f"-{len(report.dropped_constant_cols)+len(report.dropped_low_variance_cols)}")
    c4.metric("Valeurs imputées", report.imputed_values)
    c5.metric("Lignes conservées", f"{report.pct_rows_retained:.1f}%")

    if report.pct_rows_retained < 70:
        st.markdown(
            '<div class="warning-box">Plus de 30% des lignes ont été supprimées. '
            "Vérifiez la qualité de vos données ou augmentez le seuil z-score.</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Colonnes supprimées par le sanitizer"):
        if report.dropped_constant_cols:
            st.write(f"Constantes : {', '.join(report.dropped_constant_cols)}")
        if report.dropped_low_variance_cols:
            st.write(f"Quasi-constantes : {', '.join(report.dropped_low_variance_cols)}")
        if not report.dropped_constant_cols and not report.dropped_low_variance_cols:
            st.success("Aucune colonne supprimée.")

    st.markdown("---")
    st.subheader("Détection automatique des variables")
    show_detection_summary(detection)

    st.markdown("---")
    st.subheader("Statistiques Descriptives")

    try:
        engine_temp = MultivariateEngine(df_clean)
        desc = engine_temp.descriptive_stats()
        st.dataframe(
            desc.style.format("{:.3f}").background_gradient(cmap="Blues", subset=["mean", "std"]),
            use_container_width=True,
        )
    except Exception:
        st.dataframe(df_clean.describe().T, use_container_width=True)

    st.markdown("---")
    st.subheader("Aperçu des données brutes")
    st.dataframe(df_raw.head(50), use_container_width=True)


def tab_correlations(engine: MultivariateEngine) -> None:
    """Affiche la matrice de corrélation avec sélection automatique Pearson/Spearman."""
    st.subheader("Matrice de Corrélation")

    show_pvalues = st.checkbox(
        "Masquer les corrélations non significatives (p > 0.05)", value=True
    )

    try:
        with st.spinner("Test de normalité (Shapiro-Wilk) + calcul des corrélations..."):
            corr_result = engine.compute_correlation_matrix()

        if corr_result.all_normal:
            st.info(
                "📊 **Pearson** retenu — toutes les variables passent le test de "
                "normalité de Shapiro-Wilk (α = 0.05)."
            )
        else:
            st.warning(
                "📊 **Spearman** retenu — une ou plusieurs variables ne suivent pas "
                "une distribution normale (Shapiro-Wilk, α = 0.05). "
                "Spearman est plus robuste dans ce cas."
            )

        fig = plot_correlation_heatmap(
            corr_result.corr_matrix,
            p_matrix=corr_result.p_matrix if show_pvalues else None,
        )
        st.pyplot(fig, use_container_width=True)

        if show_pvalues:
            st.caption(
                f"Cases vides = corrélations non significatives (p ≥ 0.05). "
                f"Méthode : {corr_result.method.capitalize()}."
            )

        with st.expander("Matrice brute"):
            st.dataframe(
                corr_result.corr_matrix.style.format("{:.3f}").background_gradient(
                    cmap="RdYlGn", vmin=-1, vmax=1
                ),
                use_container_width=True,
            )

        with st.expander("Résultats Shapiro-Wilk par variable"):
            st.dataframe(corr_result.normality_results, use_container_width=True)
            st.caption(
                "H₀ : la variable suit une loi normale. "
                "p ≥ 0.05 → on ne rejette pas H₀ → distribution compatible avec la normalité."
            )

    except Exception as e:
        show_error("Erreur lors du calcul de la matrice de corrélation.", e)


def _section_pca(engine: MultivariateEngine, config: dict):
    """
    Section ACP dans l'onglet Analyse Multidimensionnelle.

    Retourne
    --------
    PCAResult ou None en cas d'erreur.
    """
    st.subheader("ACP — Réduction de Dimension")
    try:
        with st.spinner("Calcul de l'ACP..."):
            pca_result = engine.run_pca()

        if pca_result.variance_threshold_reached:
            st.success(
                f"{pca_result.n_components} composantes retenues "
                f"({pca_result.cumulative_variance[pca_result.n_components-1]*100:.1f}% de variance)"
            )
        else:
            st.warning(
                f"Seuil de variance non atteint — {pca_result.n_components} composantes retenues "
                f"(max atteignable : {pca_result.cumulative_variance[-1]*100:.1f}%). "
                f"Référence 80% : {pca_result.n_components_80pct} composante(s)."
            )

        tab_v, tab_b, tab_l = st.tabs(["Variance", "Biplot", "Loadings"])

        with tab_v:
            fig = plot_pca_variance(
                pca_result.cumulative_variance,
                pca_result.n_components,
                config["variance_threshold"],
            )
            st.pyplot(fig, use_container_width=True)

        with tab_b:
            if pca_result.n_components >= 2:
                fig = plot_pca_biplot(pca_result.features, pca_result.loadings)
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Le biplot nécessite au moins 2 composantes.")

        with tab_l:
            st.dataframe(
                pca_result.loadings.style.format("{:.3f}").background_gradient(
                    cmap="RdYlGn", vmin=-1, vmax=1
                ),
                use_container_width=True,
            )
            st.markdown("**Top 5 variables par composante**")
            st.dataframe(
                engine.top_variables_per_component(pca_result, n_top=5),
                use_container_width=True,
            )
            st.caption(
                "Classement par valeur absolue du loading. "
                "Le signe indique la direction de la contribution."
            )

        return pca_result

    except Exception as e:
        show_error("Erreur lors de l'ACP.", e)
        return None


def _section_fa(engine: MultivariateEngine, config: dict):
    """
    Section Analyse Factorielle dans l'onglet Analyse Multidimensionnelle.

    Retourne
    --------
    FAResult ou None en cas d'erreur.
    """
    st.subheader("Analyse Factorielle")
    try:
        with st.spinner("Calcul de l'Analyse Factorielle..."):
            fa_result = engine.run_factor_analysis(
                n_factors=config["n_factors"],
                rotation=config["rotation"],
            )

        show_kmo_badge(fa_result.kmo_score)
        st.caption(f"Rotation : **{fa_result.rotation.capitalize()}**")

        if fa_result.bartlett_p_value < 0.05:
            st.success(f"Test de Bartlett : p = {fa_result.bartlett_p_value:.4f}")
        else:
            st.warning(f"Bartlett : p = {fa_result.bartlett_p_value:.4f} (non significatif)")

        if not fa_result.fa_valid:
            st.markdown(
                '<div class="warning-box">Conditions statistiques partiellement non remplies. '
                "Interprétez avec prudence.</div>",
                unsafe_allow_html=True,
            )

        st.info(f"{fa_result.n_factors} facteur(s) retenus (critère de Kaiser)")

        tab_h, tab_s, tab_c = st.tabs(["Heatmap", "Scree", "Communautés"])

        with tab_h:
            fig = plot_fa_loadings_heatmap(fa_result.loadings, rotation=fa_result.rotation)
            st.pyplot(fig, use_container_width=True)
            st.caption("Bordures renforcées = saturations |loading| >= 0.4")

        with tab_s:
            fig = plot_scree(fa_result.eigenvalues)
            st.pyplot(fig, use_container_width=True)

        with tab_c:
            comm_df = fa_result.communalities.to_frame()
            comm_df["Qualité"] = comm_df["Communauté"].apply(
                lambda x: "Bonne" if x >= 0.5 else ("Moyenne" if x >= 0.3 else "Faible")
            )
            st.dataframe(
                comm_df.style.format({"Communauté": "{:.3f}"}),
                use_container_width=True,
            )

        return fa_result

    except Exception as e:
        show_error("Erreur lors de l'Analyse Factorielle.", e)
        return None


def _section_acm(engine: MultivariateEngine, detection: DetectionReport):
    """
    Section ACM (Analyse des Correspondances Multiples).

    Retourne
    --------
    ACMResult ou None si non applicable ou en cas d'erreur.
    """
    st.subheader("ACM — Analyse des Correspondances Multiples")

    if not detection.categorielles:
        st.info("Aucune variable catégorielle détectée — ACM non applicable.")
        return None

    st.caption(f"Variables utilisées : {', '.join(detection.categorielles)}")

    try:
        with st.spinner("Calcul de l'ACM..."):
            acm_result = engine.run_acm(detection.categorielles)

        st.success(
            f"{acm_result.n_components} composantes retenues "
            f"({acm_result.total_inertia_explained*100:.1f}% d'inertie expliquée)"
        )

        tab_i, tab_ind, tab_var = st.tabs(["Inertie", "Individus", "Modalités"])

        with tab_i:
            st.dataframe(
                acm_result.inertia_summary.style.format("{:.3f}"),
                use_container_width=True,
            )
            st.caption(
                "L'inertie en ACM est l'équivalent de la variance en ACP. "
                "Elle mesure la dispersion des profils de réponse."
            )

        with tab_ind:
            st.dataframe(acm_result.row_coordinates.round(3), use_container_width=True)
            st.caption("Coordonnées des individus sur les axes factoriels.")

        with tab_var:
            st.dataframe(acm_result.column_coordinates.round(3), use_container_width=True)
            st.caption(
                "Coordonnées des modalités. "
                "Les modalités proches ont des profils de réponse similaires."
            )

        return acm_result

    except Exception as e:
        show_error("Erreur lors de l'ACM.", e)
        return None


def _section_afdm(engine: MultivariateEngine, detection: DetectionReport):
    """
    Section AFDM (Analyse Factorielle des Données Mixtes).

    Retourne
    --------
    AFDMResult ou None si non applicable ou en cas d'erreur.
    """
    st.subheader("AFDM — Analyse Factorielle des Données Mixtes")

    if not detection.is_mixed:
        st.info(
            "Le dataset ne contient pas à la fois des variables continues et catégorielles "
            "— AFDM non applicable."
        )
        return None

    st.caption(
        f"Continues : {', '.join(detection.continues)} | "
        f"Catégorielles : {', '.join(detection.categorielles)}"
    )

    try:
        with st.spinner("Calcul de l'AFDM..."):
            afdm_result = engine.run_afdm(
                continuous_cols=detection.continues,
                categorical_cols=detection.categorielles,
            )

        st.success(
            f"{afdm_result.n_components} composantes retenues "
            f"({afdm_result.total_inertia_explained*100:.1f}% d'inertie expliquée)"
        )

        tab_i, tab_ind, tab_var = st.tabs(["Inertie", "Individus", "Variables"])

        with tab_i:
            st.dataframe(
                afdm_result.inertia_summary.style.format("{:.3f}"),
                use_container_width=True,
            )

        with tab_ind:
            st.dataframe(afdm_result.row_coordinates.round(3), use_container_width=True)
            st.caption("Coordonnées des individus dans l'espace mixte.")

        with tab_var:
            st.dataframe(afdm_result.column_coordinates.round(3), use_container_width=True)
            st.caption(
                "Coordonnées des variables continues et modalités catégorielles "
                "sur les axes factoriels."
            )

        return afdm_result

    except Exception as e:
        show_error("Erreur lors de l'AFDM.", e)
        return None


def tab_multivariate(
    engine: MultivariateEngine,
    config: dict,
    detection: DetectionReport,
) -> tuple:
    """
    Onglet Analyse Multidimensionnelle.

    Affiche conditionnellement ACP + AF (variables continues),
    ACM (variables catégorielles) et AFDM (dataset mixte)
    selon le DetectionReport.

    Retourne
    --------
    tuple : (pca_result, fa_result, acm_result, afdm_result)
        Chaque résultat peut être None si la méthode n'est pas applicable
        ou si une erreur s'est produite.
    """
    pca_result = None
    fa_result = None
    acm_result = None
    afdm_result = None

    # ── ACP + AF : variables continues ───────────────────────────────────────
    if len(detection.continues) >= 2:
        col_pca, col_fa = st.columns(2)
        with col_pca:
            pca_result = _section_pca(engine, config)
        with col_fa:
            fa_result = _section_fa(engine, config)
    elif len(detection.continues) == 1:
        st.warning(
            "Une seule variable continue détectée — "
            "ACP et Analyse Factorielle nécessitent au moins 2 variables continues."
        )
    else:
        st.info("Aucune variable continue détectée — ACP et AF non applicables.")

    # ── ACM : variables catégorielles ─────────────────────────────────────────
    if detection.categorielles:
        st.markdown("---")
        acm_result = _section_acm(engine, detection)

    # ── AFDM : dataset mixte ──────────────────────────────────────────────────
    if detection.is_mixed:
        st.markdown("---")
        afdm_result = _section_afdm(engine, detection)

    # ── Cas : aucune méthode applicable ──────────────────────────────────────
    if not detection.continues and not detection.categorielles:
        st.error(
            "Aucune variable utilisable détectée après la fiabilisation. "
            "Vérifiez la qualité de vos données ou ajustez le seuil catégoriel."
        )

    return pca_result, fa_result, acm_result, afdm_result


def tab_ia(pca_result, fa_result, acm_result, afdm_result, config: dict) -> None:
    """
    Onglet Synthèse IA — interprétation par Claude des résultats ACP, AF, ACM et AFDM.
    """
    st.subheader("Interprétation par Claude")

    api_key = get_api_key()

    if not api_key:
        st.warning(
            "Clé API Anthropic non configurée. "
            "Ajoutez ANTHROPIC_API_KEY dans le fichier .streamlit/secrets.toml"
        )
        st.code('ANTHROPIC_API_KEY = "sk-ant-..."', language="toml")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Analyse Factorielle")
        if fa_result is None:
            st.info("Lancez d'abord l'Analyse Factorielle (onglet précédent).")
        else:
            if st.button("Interpréter les Facteurs", use_container_width=True, type="primary"):
                prompt = build_fa_prompt(
                    fa_result.loadings,
                    fa_result.communalities,
                    fa_result.kmo_score,
                    fa_result.bartlett_p_value,
                    context=config["user_context"],
                    rotation=fa_result.rotation,
                )
                with st.spinner("Analyse en cours..."):
                    try:
                        response = call_claude(prompt, api_key)
                        st.session_state["fa_interpretation"] = response
                    except Exception as e:
                        show_error("Erreur lors de l'appel à l'API Claude.", e)

            if "fa_interpretation" in st.session_state:
                st.markdown(st.session_state["fa_interpretation"])

    with col2:
        st.markdown("#### Analyse en Composantes Principales")
        if pca_result is None:
            st.info("Lancez d'abord l'ACP (onglet précédent).")
        else:
            if st.button("Interpréter l'ACP", use_container_width=True):
                prompt = build_pca_prompt(
                    pca_result.loadings,
                    pca_result.explained_variance_ratio.tolist(),
                    pca_result.n_components,
                    context=config["user_context"],
                )
                with st.spinner("Analyse en cours..."):
                    try:
                        response = call_claude(prompt, api_key)
                        st.session_state["pca_interpretation"] = response
                    except Exception as e:
                        show_error("Erreur lors de l'appel à l'API Claude.", e)

            if "pca_interpretation" in st.session_state:
                st.markdown(st.session_state["pca_interpretation"])

    # ── ACM ───────────────────────────────────────────────────────────────────
    if acm_result is not None:
        st.markdown("---")
        st.markdown("#### ACM — Analyse des Correspondances Multiples")
        if st.button("Interpréter l'ACM", use_container_width=True):
            prompt = build_acm_prompt(
                acm_result.column_coordinates,
                acm_result.inertia_summary,
                acm_result.variables,
                context=config["user_context"],
            )
            with st.spinner("Analyse en cours..."):
                try:
                    response = call_claude(prompt, api_key)
                    st.session_state["acm_interpretation"] = response
                except Exception as e:
                    show_error("Erreur lors de l'appel à l'API Claude.", e)

        if "acm_interpretation" in st.session_state:
            st.markdown(st.session_state["acm_interpretation"])

    # ── AFDM ──────────────────────────────────────────────────────────────────
    if afdm_result is not None:
        st.markdown("---")
        st.markdown("#### AFDM — Analyse Factorielle des Données Mixtes")
        if st.button("Interpréter l'AFDM", use_container_width=True):
            prompt = build_afdm_prompt(
                afdm_result.column_coordinates,
                afdm_result.inertia_summary,
                afdm_result.continuous_cols,
                afdm_result.categorical_cols,
                context=config["user_context"],
            )
            with st.spinner("Analyse en cours..."):
                try:
                    response = call_claude(prompt, api_key)
                    st.session_state["afdm_interpretation"] = response
                except Exception as e:
                    show_error("Erreur lors de l'appel à l'API Claude.", e)

        if "afdm_interpretation" in st.session_state:
            st.markdown(st.session_state["afdm_interpretation"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Point d'entrée principal de l'application Streamlit."""
    config = render_sidebar()

    st.title("Statistical Audit Platform")
    st.markdown(
        "Nettoyage automatique · Détection des types · "
        "ACP · AF · ACM · AFDM · Interprétation par LLM"
    )
    st.markdown("---")

    # ── Sélection de la source de données ────────────────────────────────────
    st.markdown("#### Source de données")
    source_tab1, source_tab2, source_tab3 = st.tabs([
        "Fichier CSV", "Base SQLite", "URL directe"
    ])

    df_raw = None
    source_label = ""

    with source_tab1:
        uploaded_file = st.file_uploader(
            "Charger un dataset CSV",
            type=["csv"],
            help="Fichier CSV avec séparateur auto-détecté. Max 200 Mo.",
        )
        st.markdown(
            "Datasets de démonstration : "
            "[Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) · "
            "[Boston](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) · "
            "[Heart](https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv)"
        )
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")
                source_label = uploaded_file.name
            except Exception as e:
                show_error("Impossible de lire le fichier CSV.", e)
                return

    with source_tab2:
        st.markdown("Chargez une base SQLite locale (.db ou .sqlite) et entrez une requête SQL.")
        sqlite_file = st.file_uploader(
            "Charger une base SQLite",
            type=["db", "sqlite", "sqlite3"],
            help="Fichier de base de données SQLite.",
            key="sqlite_uploader",
        )
        if sqlite_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
                tmp.write(sqlite_file.read())
                tmp_path = tmp.name

            try:
                engine_sql = sqlalchemy.create_engine(f"sqlite:///{tmp_path}")
                with engine_sql.connect() as conn:
                    tables = sqlalchemy.inspect(engine_sql).get_table_names()

                st.success(f"Base chargée — {len(tables)} table(s) disponible(s) : {', '.join(tables)}")

                default_query = (
                    f"SELECT * FROM {tables[0]} LIMIT 1000"
                    if tables else "SELECT * FROM ma_table LIMIT 1000"
                )
                sql_query = st.text_area(
                    "Requête SQL",
                    value=default_query,
                    height=80,
                    help="Entrez votre requête SELECT. Résultat limité à 10 000 lignes.",
                )

                if st.button("Exécuter la requête", type="primary"):
                    try:
                        with engine_sql.connect() as conn:
                            df_raw = pd.read_sql(sql_query, conn)
                        source_label = f"{sqlite_file.name} — requête SQL"
                        st.session_state["sql_df"] = df_raw
                        st.session_state["sql_label"] = source_label
                        st.success(
                            f"Requête exécutée : {df_raw.shape[0]} lignes x {df_raw.shape[1]} colonnes."
                        )
                    except Exception as e:
                        show_error("Erreur lors de l'exécution de la requête SQL.", e)
                        return

                if df_raw is None and "sql_df" in st.session_state:
                    df_raw = st.session_state["sql_df"]
                    source_label = st.session_state.get("sql_label", "SQLite")

            except Exception as e:
                show_error("Impossible d'ouvrir la base SQLite.", e)
                return
        else:
            st.info(
                "Chargez un fichier .db ou .sqlite. "
                "Vous pouvez créer une base de test avec : "
                "`python assets/demo_data_generator.py --sqlite`"
            )

    with source_tab3:
        url_input = st.text_input(
            "URL d'un fichier CSV public",
            placeholder="https://raw.githubusercontent.com/.../dataset.csv",
        )
        if url_input:
            if st.button("Charger depuis l'URL", type="primary"):
                try:
                    df_raw = pd.read_csv(url_input, sep=None, engine="python")
                    source_label = url_input.split("/")[-1]
                    st.session_state["url_df"] = df_raw
                    st.session_state["url_label"] = source_label
                    st.success(
                        f"Chargé : {df_raw.shape[0]} lignes x {df_raw.shape[1]} colonnes."
                    )
                except Exception as e:
                    show_error("Impossible de charger l'URL.", e)
                    return

            if df_raw is None and "url_df" in st.session_state:
                df_raw = st.session_state["url_df"]
                source_label = st.session_state.get("url_label", "URL")

    # ── Vérifications ─────────────────────────────────────────────────────────
    if df_raw is None:
        st.info("Choisissez une source de données dans les onglets ci-dessus pour démarrer l'audit.")
        return

    if df_raw.empty:
        st.error("Le dataset chargé est vide.")
        return

    # Vider les interprétations si la source change
    if st.session_state.get("last_file") != source_label:
        for key in ("fa_interpretation", "pca_interpretation", "acm_interpretation", "afdm_interpretation"):
            st.session_state.pop(key, None)
        st.session_state["last_file"] = source_label

    st.success(
        f"Données chargées : **{df_raw.shape[0]}** lignes x **{df_raw.shape[1]}** colonnes."
    )

    # ── Sanitization ──────────────────────────────────────────────────────────
    sanitizer = DataSanitizer(
        zscore_threshold=config["zscore_threshold"],
        imputation_strategy=config["imputation"],
    )

    try:
        with st.spinner("Fiabilisation des données en cours..."):
            df_clean, report = sanitizer.fit_transform(df_raw)
    except Exception as e:
        show_error("Erreur lors de la sanitisation.", e)
        return

    check = DataSanitizer.check_minimum_requirements(df_clean)
    if not check["valid"]:
        for issue in check["issues"]:
            st.error(issue)
        return

    # ── Détection des types de variables ──────────────────────────────────────
    detector = VariableDetector(
        categorical_threshold=config["categorical_threshold"],
    )
    detection = detector.detect(df_clean)

    # ── Moteur (initialisé sur les variables continues uniquement) ─────────────
    # Le moteur MultivariateEngine travaille sur des données numériques standardisées.
    # ACM et AFDM reçoivent le df_clean complet via engine.run_acm / engine.run_afdm.
    engine = None
    if detection.continues:
        try:
            engine = MultivariateEngine(
                df_clean,
                variance_threshold=config["variance_threshold"],
            )
        except Exception as e:
            show_error("Impossible d'initialiser le moteur d'analyse.", e)
            return

    # ── Onglets ───────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "Ingestion & Audit",
        "Corrélations",
        "Analyse Multidimensionnelle",
        "Synthèse IA",
    ])

    with tabs[0]:
        tab_ingestion(df_raw, df_clean, report, detection)

    with tabs[1]:
        if engine is not None:
            tab_correlations(engine)
        else:
            st.info("Aucune variable continue disponible pour le calcul des corrélations.")

    with tabs[2]:
        if engine is not None:
            pca_result, fa_result, acm_result, afdm_result = tab_multivariate(
                engine, config, detection
            )
        else:
            pca_result, fa_result, acm_result, afdm_result = None, None, None, None
            if detection.categorielles:
                st.info(
                    "Pas de variables continues — seule l'ACM est disponible. "
                    "Veuillez ajuster le seuil catégoriel si nécessaire."
                )

        st.session_state["pca_result"] = pca_result
        st.session_state["fa_result"] = fa_result
        st.session_state["acm_result"] = acm_result
        st.session_state["afdm_result"] = afdm_result

    with tabs[3]:
        tab_ia(
            st.session_state.get("pca_result"),
            st.session_state.get("fa_result"),
            st.session_state.get("acm_result"),
            st.session_state.get("afdm_result"),
            config,
        )


if __name__ == "__main__":
    main()
