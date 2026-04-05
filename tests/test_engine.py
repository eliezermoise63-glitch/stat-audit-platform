"""
tests/test_engine.py
--------------------
Tests unitaires pour le module core.engine.
"""

import numpy as np
import pandas as pd
import pytest

from core.engine import MultivariateEngine, PCAResult, FAResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame de test avec corrélations connues."""
    np.random.seed(0)
    n = 150
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    return pd.DataFrame({
        "a": f1 + 0.1 * np.random.randn(n),
        "b": f1 + 0.2 * np.random.randn(n),
        "c": f2 + 0.1 * np.random.randn(n),
        "d": f2 + 0.3 * np.random.randn(n),
        "e": np.random.randn(n),
    })


# ── Tests Engine ──────────────────────────────────────────────────────────────

class TestMultivariateEngine:

    def test_init_valid(self, sample_df):
        engine = MultivariateEngine(sample_df)
        assert engine.data.shape == sample_df.shape

    def test_init_empty_raises(self):
        with pytest.raises(ValueError, match="vide"):
            MultivariateEngine(pd.DataFrame())

    def test_init_single_column_raises(self):
        with pytest.raises(ValueError):
            MultivariateEngine(pd.DataFrame({"x": [1, 2, 3]}))

    def test_pca_returns_correct_type(self, sample_df):
        engine = MultivariateEngine(sample_df)
        result = engine.run_pca()
        assert isinstance(result, PCAResult)

    def test_pca_n_components_within_bounds(self, sample_df):
        engine = MultivariateEngine(sample_df)
        result = engine.run_pca()
        assert 2 <= result.n_components <= sample_df.shape[1]

    def test_pca_variance_threshold_respected(self, sample_df):
        threshold = 0.75
        engine = MultivariateEngine(sample_df, variance_threshold=threshold)
        result = engine.run_pca()
        actual_var = result.cumulative_variance[result.n_components - 1]
        assert actual_var >= threshold or result.n_components == sample_df.shape[1]

    def test_pca_loadings_shape(self, sample_df):
        engine = MultivariateEngine(sample_df)
        result = engine.run_pca()
        assert result.loadings.shape == (sample_df.shape[1], result.n_components)

    def test_fa_returns_correct_type(self, sample_df):
        engine = MultivariateEngine(sample_df)
        result = engine.run_factor_analysis()
        assert isinstance(result, FAResult)

    def test_fa_n_factors_positive(self, sample_df):
        engine = MultivariateEngine(sample_df)
        result = engine.run_factor_analysis()
        assert result.n_factors >= 1

    def test_fa_manual_n_factors(self, sample_df):
        engine = MultivariateEngine(sample_df)
        result = engine.run_factor_analysis(n_factors=2)
        assert result.n_factors == 2
        assert result.loadings.shape[1] == 2

    def test_fa_loadings_shape(self, sample_df):
        engine = MultivariateEngine(sample_df)
        result = engine.run_factor_analysis(n_factors=2)
        assert result.loadings.shape[0] == sample_df.shape[1]

    def test_fa_communalities_non_negative(self, sample_df):
        engine = MultivariateEngine(sample_df)
        result = engine.run_factor_analysis()
        assert (result.communalities >= 0).all()

    def test_correlation_matrix_symmetric(self, sample_df):
        engine = MultivariateEngine(sample_df)
        corr, p = engine.compute_correlation_matrix()
        pd.testing.assert_frame_equal(corr, corr.T)

    def test_correlation_diagonal_ones(self, sample_df):
        engine = MultivariateEngine(sample_df)
        corr, _ = engine.compute_correlation_matrix()
        np.testing.assert_array_almost_equal(np.diag(corr.values), np.ones(corr.shape[0]))

    def test_descriptive_stats_columns(self, sample_df):
        engine = MultivariateEngine(sample_df)
        desc = engine.descriptive_stats()
        assert "skewness" in desc.columns
        assert "kurtosis" in desc.columns
        assert "missing_pct" in desc.columns
