"""
app.py
------
Point d'entrée de la plateforme d'Audit & Analyse Statistique.
Lance avec : streamlit run app.py
"""

import logging
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from core.sanitizer import DataSanitizer
from core.engine import MultivariateEngine
from utils.charts import (
    plot_correlation_heatmap,
    plot_fa_loadings_heatmap,
    plot_pca_biplot,
    plot_pca_variance,
    plot_scree,
)
from utils.llm import build_fa_prompt, build_pca_prompt, call_claude

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
    </style>
""", unsafe_allow_html=True)


def render_sidebar() -> dict:
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

        st.markdown("---")
        st.subheader("LLM")
        user_context = st.text_area(
            "Contexte métier (optionnel)",
            placeholder="Ex : données RH d'une entreprise industrielle...",
            height=80,
        )

        st.markdown("---")
        st.caption("Statistical Audit Platform v0.1.0")

    return {
        "zscore_threshold": zscore_threshold,
        "imputation": imputation,
        "variance_threshold": variance_threshold,
        "n_factors": n_factors_manual,
        "user_context": user_context,
    }


def get_api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


def show_error(msg: str, exception: Exception | None = None) -> None:
    st.error(f"Erreur : {msg}")
    if exception and st.checkbox("Afficher les détails techniques", key=f"err_{hash(msg)}"):
        st.code(traceback.format_exc())


def show_kmo_badge(kmo: float) -> None:
    if kmo >= 0.8:
        label, color = "Excellent", "#28a745"
    elif kmo >= 0.7:
        label, color = "Bon", "#17a2b8"
    elif kmo >= 0.6:
        label, color = "Acceptable", "#ffc107"
    else:
        label, color = "Insuffisant", "#dc3545"
    st.markdown(
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-weight:bold;">KMO {kmo:.3f} — {label}</span>',
        unsafe_allow_html=True,
    )


def tab_ingestion(df_raw: pd.DataFrame, df_clean: pd.DataFrame, report) -> None:
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

    with st.expander("Colonnes supprimées"):
        if report.dropped_constant_cols:
            st.write(f"Constantes : {', '.join(report.dropped_constant_cols)}")
        if report.dropped_low_variance_cols:
            st.write(f"Quasi-constantes : {', '.join(report.dropped_low_variance_cols)}")
        if not report.dropped_constant_cols and not report.dropped_low_variance_cols:
            st.success("Aucune colonne supprimée.")

    st.markdown("---")
    st.subheader("Statistiques Descriptives")

    from core.engine import MultivariateEngine as ME
    try:
        engine_temp = ME(df_clean)
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
    st.subheader("Matrice de Corrélation")

    show_pvalues = st.checkbox(
        "Masquer les corrélations non significatives (p > 0.05)", value=True
    )

    try:
        with st.spinner("Calcul des corrélations..."):
            corr_matrix, p_matrix = engine.compute_correlation_matrix()

        fig = plot_correlation_heatmap(
            corr_matrix,
            p_matrix=p_matrix if show_pvalues else None,
        )
        st.pyplot(fig, use_container_width=True)

        if show_pvalues:
            st.caption("Cases vides = corrélations non significatives (p >= 0.05).")

        with st.expander("Matrice brute"):
            st.dataframe(
                corr_matrix.style.format("{:.3f}").background_gradient(
                    cmap="RdYlGn", vmin=-1, vmax=1
                ),
                use_container_width=True,
            )
    except Exception as e:
        show_error("Erreur lors du calcul de la matrice de corrélation.", e)


def tab_multivariate(engine: MultivariateEngine, config: dict) -> tuple:
    pca_result = None
    fa_result = None

    col_pca, col_fa = st.columns(2)

    with col_pca:
        st.subheader("ACP — Réduction de Dimension")
        try:
            with st.spinner("Calcul de l'ACP..."):
                pca_result = engine.run_pca()

            st.success(
                f"{pca_result.n_components} composantes retenues "
                f"({pca_result.cumulative_variance[pca_result.n_components-1]*100:.1f}% de variance)"
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

        except Exception as e:
            show_error("Erreur lors de l'ACP.", e)

    with col_fa:
        st.subheader("Analyse Factorielle")
        try:
            with st.spinner("Calcul de l'Analyse Factorielle..."):
                fa_result = engine.run_factor_analysis(n_factors=config["n_factors"])

            show_kmo_badge(fa_result.kmo_score)

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
                fig = plot_fa_loadings_heatmap(fa_result.loadings)
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

        except Exception as e:
            show_error("Erreur lors de l'Analyse Factorielle.", e)

    return pca_result, fa_result


def tab_ia(pca_result, fa_result, config: dict) -> None:
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
            return

        if st.button("Interpréter les Facteurs", use_container_width=True, type="primary"):
            prompt = build_fa_prompt(
                fa_result.loadings,
                fa_result.communalities,
                fa_result.kmo_score,
                fa_result.bartlett_p_value,
                context=config["user_context"],
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
            return

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


def main() -> None:
    config = render_sidebar()

    st.title("Statistical Audit Platform")
    st.markdown(
        "Nettoyage automatique · Analyse multivariée (ACP & AF) · Interprétation par LLM"
    )
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Charger un dataset CSV",
        type=["csv"],
        help="Fichier CSV avec séparateur auto-détecté. Max 200 Mo.",
    )

    if uploaded_file is None:
        st.info("Déposez un fichier CSV pour démarrer l'audit.")
        st.markdown("#### Datasets de démonstration")
        st.markdown(
            "Cliquez pour télécharger puis uploadez le fichier dans l'app : "
            "[Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) · "
            "[Boston](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) · "
            "[Heart](https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv)"
        )
        return

    # ── Ingestion ─────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")

        current_file = uploaded_file.name
        if st.session_state.get("last_file") != current_file:
            st.session_state.pop("fa_interpretation", None)
            st.session_state.pop("pca_interpretation", None)
            st.session_state["last_file"] = current_file

    except Exception as e:
        show_error("Impossible de lire le fichier CSV.", e)
        return

    if df_raw.empty:
        st.error("Le fichier CSV est vide.")
        return

    st.success(
        f"Fichier chargé : **{df_raw.shape[0]}** lignes x **{df_raw.shape[1]}** colonnes."
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

    # ── Moteur ────────────────────────────────────────────────────────────────
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
        tab_ingestion(df_raw, df_clean, report)

    with tabs[1]:
        tab_correlations(engine)

    with tabs[2]:
        pca_result, fa_result = tab_multivariate(engine, config)
        st.session_state["pca_result"] = pca_result
        st.session_state["fa_result"] = fa_result

    with tabs[3]:
        tab_ia(
            st.session_state.get("pca_result"),
            st.session_state.get("fa_result"),
            config,
        )


if __name__ == "__main__":
    main()