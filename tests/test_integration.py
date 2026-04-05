"""
tests/test_integration.py
--------------------------
Tests d'intégration du pipeline complet :
  DataSanitizer → MultivariateEngine → Charts

Ces tests valident que les modules s'enchaînent sans erreur
sur des données réalistes.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from core.engine import MultivariateEngine
from core.sanitizer import DataSanitizer
from utils.charts import (
    plot_correlation_heatmap,
    plot_fa_loadings_heatmap,
    plot_pca_biplot,
    plot_pca_variance,
    plot_scree,
)


@pytest.fixture(scope="module")
def full_pipeline_output(iris_like_df):
    """
    Exécute le pipeline complet sur le dataset Iris-like
    et retourne (df_clean, report, engine, pca, fa).
    """
    sanitizer = DataSanitizer(zscore_threshold=3.0)
    df_clean, report = sanitizer.fit_transform(iris_like_df)

    engine = MultivariateEngine(df_clean, variance_threshold=0.80)

    pca = engine.run_pca()
    fa = engine.run_factor_analysis()

    return df_clean, report, engine, pca, fa


class TestFullPipeline:

    def test_pipeline_produces_clean_data(self, full_pipeline_output):
        df_clean, report, *_ = full_pipeline_output
        assert not df_clean.empty
        assert df_clean.isnull().sum().sum() == 0

    def test_pipeline_report_coherent(self, full_pipeline_output):
        _, report, *_ = full_pipeline_output
        assert report.n_rows_output <= report.n_rows_input
        assert report.n_cols_output <= report.n_cols_input

    def test_pca_components_valid(self, full_pipeline_output):
        *_, pca, fa = full_pipeline_output
        assert pca.n_components >= 1
        assert pca.loadings.shape[0] >= 2
        assert len(pca.explained_variance_ratio) == pca.n_components

    def test_fa_factors_valid(self, full_pipeline_output):
        *_, pca, fa = full_pipeline_output
        assert fa.n_factors >= 1
        assert fa.loadings.shape[1] == fa.n_factors
        assert len(fa.communalities) > 0

    def test_fa_kmo_between_0_and_1(self, full_pipeline_output):
        *_, pca, fa = full_pipeline_output
        assert 0.0 <= fa.kmo_score <= 1.0

    def test_correlation_matrix_shape(self, full_pipeline_output):
        _, _, engine, *_ = full_pipeline_output
        corr, p = engine.compute_correlation_matrix()
        n = engine.data.shape[1]
        assert corr.shape == (n, n)
        assert p.shape == (n, n)

    def test_charts_all_render(self, full_pipeline_output):
        df_clean, _, engine, pca, fa = full_pipeline_output
        corr, p = engine.compute_correlation_matrix()

        figs = [
            plot_correlation_heatmap(corr, p),
            plot_pca_variance(pca.cumulative_variance, pca.n_components),
            plot_fa_loadings_heatmap(fa.loadings),
            plot_scree(fa.eigenvalues),
        ]
        if pca.n_components >= 2:
            figs.append(plot_pca_biplot(pca.features, pca.loadings))

        for fig in figs:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_pipeline_with_dirty_data(self, df_with_all_issues):
        """Pipeline doit fonctionner même sur données très sales."""
        sanitizer = DataSanitizer()
        df_clean, report = sanitizer.fit_transform(df_with_all_issues)

        check = DataSanitizer.check_minimum_requirements(df_clean)
        if not check["valid"]:
            pytest.skip("Dataset trop petit après nettoyage — cas limite attendu.")

        engine = MultivariateEngine(df_clean)
        pca = engine.run_pca()
        assert pca.n_components >= 1

    def test_reproducibility(self, iris_like_df):
        """Deux exécutions consécutives donnent le même résultat."""
        san = DataSanitizer(zscore_threshold=3.0)
        df1, r1 = san.fit_transform(iris_like_df)
        df2, r2 = san.fit_transform(iris_like_df)

        assert r1.n_rows_output == r2.n_rows_output
        assert r1.n_cols_output == r2.n_cols_output
        pd.testing.assert_frame_equal(df1, df2)
