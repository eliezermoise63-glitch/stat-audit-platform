"""
core/engine.py
--------------
Moteur d'Analyse Multivariée.

Fonctionnalités :
  - ACP (PCA) avec sélection automatique du nombre de composantes
  - Analyse Factorielle (FA) avec critère de Kaiser + rotation Varimax
  - Validation statistique : KMO + Test de sphéricité de Bartlett
  - Matrice de corrélation + p-values

Design : stateless par méthode, chaque appel retourne un résultat complet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Résultats typés ───────────────────────────────────────────────────────────

@dataclass
class PCAResult:
    features: np.ndarray              # (n_samples, n_components)
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    loadings: pd.DataFrame            # (n_features, n_components)
    n_components: int
    n_components_80pct: int           # composantes pour 80% de variance


@dataclass
class FAResult:
    loadings: pd.DataFrame            # (n_features, n_factors)
    communalities: pd.Series
    n_factors: int
    eigenvalues: np.ndarray
    kmo_score: float
    bartlett_p_value: float
    fa_valid: bool                    # True si KMO > 0.6 et Bartlett p < 0.05


# ── Moteur principal ──────────────────────────────────────────────────────────

class MultivariateEngine:
    """
    Moteur d'analyse multivariée sur un DataFrame numérique propre.

    Paramètres
    ----------
    df : pd.DataFrame
        Données fiabilisées (sortie de DataSanitizer).
    variance_threshold : float
        Seuil de variance cumulée pour l'ACP automatique (défaut 0.80 = 80%).
    kmo_threshold : float
        Seuil KMO minimum pour valider l'AF (défaut 0.60).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        variance_threshold: float = 0.80,
        kmo_threshold: float = 0.60,
    ) -> None:
        if df.empty:
            raise ValueError("Le DataFrame fourni au moteur est vide.")

        self.data: pd.DataFrame = df.select_dtypes(include=[np.number]).copy()

        if self.data.shape[1] < 2:
            raise ValueError("Au moins 2 colonnes numériques sont nécessaires.")

        self.variance_threshold = variance_threshold
        self.kmo_threshold = kmo_threshold

        # Standardisation : obligatoire pour ACP et AF
        self.scaler = StandardScaler()
        self.data_scaled: np.ndarray = self.scaler.fit_transform(self.data)
        logger.info(
            f"[Engine] Données standardisées : {self.data.shape[0]} lignes × {self.data.shape[1]} colonnes."
        )

    # ── ACP ───────────────────────────────────────────────────────────────────

    def run_pca(self) -> PCAResult:
        """
        ACP avec sélection automatique du nombre de composantes (variance threshold).

        Retourne un PCAResult complet.
        """
        # 1. ACP complète pour calculer la variance cumulée
        pca_full = PCA().fit(self.data_scaled)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        # 2. Nombre de composantes pour 80% de variance
        n_components_80 = int(np.argmax(cumulative_variance >= 0.80) + 1)

        # 3. Nombre de composantes selon le seuil défini
        n_components = int(np.argmax(cumulative_variance >= self.variance_threshold) + 1)
        n_components = max(2, min(n_components, self.data.shape[1]))  # au moins 2, au plus p

        # 4. ACP finale
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(self.data_scaled)

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i + 1}" for i in range(n_components)],
            index=self.data.columns,
        )

        logger.info(
            f"[PCA] {n_components} composantes retenues "
            f"({cumulative_variance[n_components - 1] * 100:.1f}% de variance)."
        )

        return PCAResult(
            features=features,
            explained_variance_ratio=pca.explained_variance_ratio_,
            cumulative_variance=cumulative_variance,
            loadings=loadings,
            n_components=n_components,
            n_components_80pct=n_components_80,
        )

    # ── Analyse Factorielle ───────────────────────────────────────────────────

    def run_factor_analysis(self, n_factors: Optional[int] = None) -> FAResult:
        """
        Analyse Factorielle avec :
          - Validation KMO + Bartlett
          - Critère de Kaiser pour le choix automatique du nombre de facteurs
          - Rotation Varimax

        Retourne un FAResult complet.
        """
        # 1. Validation préalable
        kmo_score, bartlett_p = self._validate_fa()

        fa_valid = (kmo_score >= self.kmo_threshold) and (bartlett_p < 0.05)
        if not fa_valid:
            logger.warning(
                f"[FA] Conditions non idéales : KMO={kmo_score:.3f}, Bartlett p={bartlett_p:.4f}. "
                "Les résultats sont à interpréter avec prudence."
            )

        # 2. Choix automatique du nombre de facteurs (critère de Kaiser : valeurs propres > 1)
        # On utilise self.data (DataFrame pandas) plutôt que self.data_scaled (numpy array)
        # pour éviter le conflit check_array() entre factor_analyzer et scikit-learn récent.
        data_for_fa = pd.DataFrame(
            self.data_scaled,
            columns=self.data.columns,
        )

        if n_factors is None:
            fa_temp = FactorAnalyzer(rotation=None)
            fa_temp.fit(data_for_fa)
            eigenvalues, _ = fa_temp.get_eigenvalues()
            n_factors = max(1, int(np.sum(eigenvalues > 1)))
            logger.info(f"[FA] Critère de Kaiser : {n_factors} facteur(s) retenus.")

        # Sécurité : n_factors ne peut pas dépasser n_cols - 1
        n_factors = min(n_factors, self.data.shape[1] - 1)

        # 3. AF avec rotation Varimax
        fa = FactorAnalyzer(n_factors=n_factors, rotation="varimax")
        fa.fit(data_for_fa)

        loadings = pd.DataFrame(
            fa.loadings_,
            columns=[f"F{i + 1}" for i in range(n_factors)],
            index=self.data.columns,
        )

        communalities = pd.Series(
            fa.get_communalities(),
            index=self.data.columns,
            name="Communauté",
        )

        eigenvalues_full, _ = fa.get_eigenvalues()

        logger.info(
            f"[FA] AF complète : {n_factors} facteur(s), KMO={kmo_score:.3f}."
        )

        return FAResult(
            loadings=loadings,
            communalities=communalities,
            n_factors=n_factors,
            eigenvalues=eigenvalues_full,
            kmo_score=kmo_score,
            bartlett_p_value=float(bartlett_p),
            fa_valid=fa_valid,
        )

    # ── Corrélations ─────────────────────────────────────────────────────────

    def compute_correlation_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retourne (matrice de corrélation de Pearson, matrice de p-values).
        """
        n = self.data.shape[1]
        corr_matrix = self.data.corr(method="pearson")
        p_matrix = pd.DataFrame(np.ones((n, n)), index=self.data.columns, columns=self.data.columns)

        for i, col_i in enumerate(self.data.columns):
            for j, col_j in enumerate(self.data.columns):
                if i != j:
                    _, p = stats.pearsonr(self.data[col_i], self.data[col_j])
                    p_matrix.loc[col_i, col_j] = p

        return corr_matrix, p_matrix

    # ── Statistiques descriptives ─────────────────────────────────────────────

    def descriptive_stats(self) -> pd.DataFrame:
        """Statistiques descriptives étendues (skewness, kurtosis inclus)."""
        desc = self.data.describe().T
        desc["skewness"] = self.data.skew()
        desc["kurtosis"] = self.data.kurtosis()
        desc["missing_pct"] = (self.data.isnull().sum() / len(self.data) * 100).round(2)
        return desc

    # ── Validation AF (privé) ─────────────────────────────────────────────────

    def _validate_fa(self) -> Tuple[float, float]:
        """Calcule KMO et test de Bartlett. Utilise un DataFrame pour compatibilité."""
        try:
            # DataFrame pandas requis par calculate_kmo / calculate_bartlett_sphericity
            # pour éviter le conflit check_array() avec scikit-learn >= 1.6
            data_df = pd.DataFrame(self.data_scaled, columns=self.data.columns)
            _, kmo_score = calculate_kmo(data_df)
            chi2, p_value = calculate_bartlett_sphericity(data_df)
            return float(kmo_score), float(p_value)
        except Exception as e:
            logger.warning(f"[FA Validation] Erreur lors du calcul KMO/Bartlett : {e}")
            return 0.0, 1.0
