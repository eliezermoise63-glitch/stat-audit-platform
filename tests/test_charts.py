"""
tests/test_charts.py
--------------------
Tests unitaires pour le module utils/charts.py.
On vérifie que chaque fonction retourne bien une Figure matplotlib,
sans lever d'exception — pas de test visuel ici.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")  # backend non-interactif pour les tests

from utils.charts import (
    plot_correlation_heatmap,
    plot_fa_loadings_heatmap,
    plot_pca_biplot,
    plot_pca_variance,
    plot_scree,
)


@pytest.fixture
def sample_corr():
    np.random.seed(0)
    data = pd.DataFrame(np.random.randn(50, 4), columns=["A", "B", "C", "D"])
    return data.corr()


@pytest.fixture
def sample_loadings_pca():
    return pd.DataFrame(
        {"PC1": [0.8, -0.3, 0.6, 0.1], "PC2": [0.2, 0.7, -0.4, 0.9]},
        index=["var1", "var2", "var3", "var4"],
    )


@pytest.fixture
def sample_loadings_fa():
    return pd.DataFrame(
        {"F1": [0.85, 0.76, 0.12, -0.05], "F2": [-0.10, 0.08, 0.82, 0.79]},
        index=["var1", "var2", "var3", "var4"],
    )


@pytest.fixture
def sample_features():
    np.random.seed(1)
    return np.random.randn(50, 2)


class TestCharts:

    def test_correlation_heatmap_returns_figure(self, sample_corr):
        fig = plot_correlation_heatmap(sample_corr)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_correlation_heatmap_with_pvalues(self, sample_corr):
        p_matrix = pd.DataFrame(
            np.random.rand(4, 4) * 0.1,
            index=sample_corr.index,
            columns=sample_corr.columns,
        )
        fig = plot_correlation_heatmap(sample_corr, p_matrix=p_matrix)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pca_variance_returns_figure(self):
        cumvar = np.cumsum([0.35, 0.25, 0.20, 0.12, 0.08])
        fig = plot_pca_variance(cumvar, n_components_selected=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pca_biplot_returns_figure(self, sample_features, sample_loadings_pca):
        fig = plot_pca_biplot(sample_features, sample_loadings_pca)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_scree_returns_figure(self):
        eigenvalues = np.array([3.2, 1.8, 0.9, 0.5, 0.3, 0.1])
        fig = plot_scree(eigenvalues)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_fa_loadings_heatmap_returns_figure(self, sample_loadings_fa):
        fig = plot_fa_loadings_heatmap(sample_loadings_fa)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_charts_no_exception_on_small_data(self):
        """Les graphiques ne doivent pas planter avec un minimum de données."""
        small_corr = pd.DataFrame(
            {"X": [1.0, 0.5], "Y": [0.5, 1.0]},
            index=["X", "Y"],
        )
        fig = plot_correlation_heatmap(small_corr)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
