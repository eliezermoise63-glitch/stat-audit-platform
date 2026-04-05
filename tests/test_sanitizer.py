"""
tests/test_sanitizer.py
-----------------------
Tests unitaires pour le module core.sanitizer.
"""

import numpy as np
import pandas as pd
import pytest

from core.sanitizer import DataSanitizer, SanitizationReport


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_df():
    """DataFrame propre, sans NaN, sans outliers."""
    np.random.seed(42)
    return pd.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100) * 2,
        "x3": np.random.randn(100) + 1,
    })


@pytest.fixture
def dirty_df(clean_df):
    """DataFrame avec NaN et outliers."""
    df = clean_df.copy()
    df.loc[0:10, "x1"] = np.nan       # NaN
    df.loc[50, "x2"] = 1000.0          # outlier extrême
    df["const"] = 42.0                  # colonne constante
    return df


# ── Tests Sanitizer ───────────────────────────────────────────────────────────

class TestDataSanitizer:

    def test_returns_dataframe_and_report(self, clean_df):
        san = DataSanitizer()
        result, report = san.fit_transform(clean_df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(report, SanitizationReport)

    def test_drops_constant_columns(self, dirty_df):
        san = DataSanitizer()
        result, report = san.fit_transform(dirty_df)
        assert "const" not in result.columns
        assert "const" in report.dropped_constant_cols

    def test_imputes_missing_values(self, dirty_df):
        san = DataSanitizer()
        result, report = san.fit_transform(dirty_df)
        assert result.isnull().sum().sum() == 0
        assert report.imputed_values > 0

    def test_removes_outliers(self, dirty_df):
        san = DataSanitizer(zscore_threshold=3.0)
        result, report = san.fit_transform(dirty_df)
        assert report.outliers_removed >= 1

    def test_report_metrics(self, dirty_df):
        san = DataSanitizer()
        _, report = san.fit_transform(dirty_df)
        assert report.n_rows_input == len(dirty_df)
        assert report.n_cols_output <= report.n_cols_input
        assert 0 <= report.pct_rows_retained <= 100

    def test_no_numeric_columns_returns_empty(self):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "city": ["Paris", "Lyon"]})
        san = DataSanitizer()
        result, report = san.fit_transform(df)
        assert result.empty or result.shape[1] == 0

    def test_check_minimum_requirements_pass(self, clean_df):
        check = DataSanitizer.check_minimum_requirements(clean_df)
        assert check["valid"] is True
        assert check["issues"] == []

    def test_check_minimum_requirements_fail(self):
        tiny_df = pd.DataFrame({"x": [1, 2]})
        check = DataSanitizer.check_minimum_requirements(tiny_df, min_rows=10, min_cols=2)
        assert check["valid"] is False
        assert len(check["issues"]) > 0

    def test_zscore_threshold_effect(self, clean_df):
        # Avec un seuil très bas, on supprime plus de lignes
        san_strict = DataSanitizer(zscore_threshold=1.5)
        san_loose = DataSanitizer(zscore_threshold=5.0)
        _, report_strict = san_strict.fit_transform(clean_df)
        _, report_loose = san_loose.fit_transform(clean_df)
        assert report_strict.outliers_removed >= report_loose.outliers_removed

    def test_report_to_dict(self, clean_df):
        san = DataSanitizer()
        _, report = san.fit_transform(clean_df)
        d = report.to_dict()
        assert "Lignes (entrée)" in d
        assert "% lignes conservées" in d
