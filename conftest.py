"""
conftest.py
-----------
Fixtures partagées pour tous les tests pytest.
Ce fichier est automatiquement chargé par pytest.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def iris_like_df():
    """
    Dataset de type Iris (150 obs, 4 variables numériques corrélées).
    Reproductible via seed fixe.
    """
    np.random.seed(42)
    n = 150
    # 2 facteurs sous-jacents simulés
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)
    return pd.DataFrame({
        "sepal_length": f1 * 0.8 + np.random.randn(n) * 0.3,
        "sepal_width":  f1 * 0.4 + np.random.randn(n) * 0.5,
        "petal_length": f2 * 0.9 + np.random.randn(n) * 0.2,
        "petal_width":  f2 * 0.85 + np.random.randn(n) * 0.25,
    })


@pytest.fixture(scope="session")
def small_df():
    """
    Petit dataset (20 lignes) pour tester les cas limites.
    """
    np.random.seed(7)
    return pd.DataFrame({
        "x1": np.random.randn(20),
        "x2": np.random.randn(20),
        "x3": np.random.randn(20),
    })


@pytest.fixture(scope="session")
def df_with_all_issues():
    """
    Dataset avec tous les problèmes possibles :
    - NaN
    - Outliers extrêmes
    - Colonne constante
    - Colonne quasi-constante
    """
    np.random.seed(99)
    n = 100
    df = pd.DataFrame({
        "clean1": np.random.randn(n),
        "clean2": np.random.randn(n) * 2 + 1,
        "nan_col": np.where(np.random.rand(n) < 0.15, np.nan, np.random.randn(n)),
        "outlier_col": np.concatenate([np.random.randn(95), [100, -100, 200, -200, 50]]),
        "constant_col": np.ones(n) * 42.0,
        "low_var_col": np.concatenate([np.ones(98), [2.0, 3.0]]),
    })
    return df
