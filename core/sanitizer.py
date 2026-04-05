"""
core/sanitizer.py
-----------------
Module de fiabilisation des données (Data Sanitizer).

Pipeline :
  1. Sélection des colonnes numériques
  2. Suppression des colonnes constantes ou quasi-constantes
  3. Imputation des valeurs manquantes (médiane — robuste aux outliers)
  4. Détection et suppression des outliers multivariés (z-score)
  5. Rapport complet de traçabilité

Design : stateless, réutilisable, testable unitairement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


# ── Rapport de sanitisation ───────────────────────────────────────────────────

@dataclass
class SanitizationReport:
    """Résumé complet du pipeline de fiabilisation."""
    n_rows_input: int = 0
    n_cols_input: int = 0
    dropped_constant_cols: List[str] = field(default_factory=list)
    dropped_low_variance_cols: List[str] = field(default_factory=list)
    imputed_values: int = 0
    outliers_removed: int = 0
    n_rows_output: int = 0
    n_cols_output: int = 0

    # ── Propriétés calculées ──────────────────────────────────────────────────
    @property
    def pct_rows_retained(self) -> float:
        if self.n_rows_input == 0:
            return 0.0
        return self.n_rows_output / self.n_rows_input * 100

    @property
    def pct_missing(self) -> float:
        total = self.n_rows_input * self.n_cols_input
        if total == 0:
            return 0.0
        return self.imputed_values / total * 100

    def to_dict(self) -> dict:
        return {
            "Lignes (entrée)": self.n_rows_input,
            "Colonnes (entrée)": self.n_cols_input,
            "Colonnes constantes supprimées": len(self.dropped_constant_cols),
            "Colonnes quasi-constantes supprimées": len(self.dropped_low_variance_cols),
            "Valeurs imputées": self.imputed_values,
            "Outliers supprimés": self.outliers_removed,
            "Lignes (sortie)": self.n_rows_output,
            "Colonnes (sortie)": self.n_cols_output,
            "% lignes conservées": f"{self.pct_rows_retained:.1f}%",
            "% valeurs manquantes": f"{self.pct_missing:.2f}%",
        }


# ── Sanitizer principal ───────────────────────────────────────────────────────

class DataSanitizer:
    """
    Fiabilise un DataFrame brut pour l'analyse multivariée.

    Paramètres
    ----------
    zscore_threshold : float
        Seuil z-score au-delà duquel une ligne est considérée outlier (défaut 3).
    min_unique_ratio : float
        Ratio minimal d'unicité pour garder une colonne (défaut 0.01 = 1%).
    imputation_strategy : str
        Stratégie d'imputation sklearn ('median', 'mean', 'most_frequent').
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        min_unique_ratio: float = 0.01,
        imputation_strategy: str = "median",
    ) -> None:
        self.zscore_threshold = zscore_threshold
        self.min_unique_ratio = min_unique_ratio
        self.imputation_strategy = imputation_strategy

    # ── Pipeline principal ────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, SanitizationReport]:
        """
        Applique le pipeline complet de fiabilisation.

        Retourne
        --------
        df_clean : pd.DataFrame
            Données fiabilisées, prêtes pour l'analyse.
        report : SanitizationReport
            Rapport détaillé de traçabilité.
        """
        report = SanitizationReport(
            n_rows_input=len(df),
            n_cols_input=df.shape[1],
        )

        # Étape 0 — Colonnes numériques uniquement
        df_num = df.select_dtypes(include=[np.number]).copy()
        logger.info(f"[Sanitizer] {df_num.shape[1]} colonnes numériques extraites.")

        # Étape 1 — Suppression des colonnes constantes (nunique == 1)
        constant_cols = [c for c in df_num.columns if df_num[c].nunique(dropna=False) <= 1]
        df_num = df_num.drop(columns=constant_cols)
        report.dropped_constant_cols = constant_cols
        logger.info(f"[Sanitizer] {len(constant_cols)} colonne(s) constante(s) supprimée(s).")

        # Étape 2 — Suppression des colonnes quasi-constantes (faible variance)
        low_var_cols = []
        for col in df_num.columns:
            n_unique = df_num[col].nunique(dropna=True)
            ratio = n_unique / max(len(df_num), 1)
            if ratio < self.min_unique_ratio:
                low_var_cols.append(col)
        df_num = df_num.drop(columns=low_var_cols)
        report.dropped_low_variance_cols = low_var_cols
        logger.info(f"[Sanitizer] {len(low_var_cols)} colonne(s) quasi-constante(s) supprimée(s).")

        # Étape 3 — Imputation (médiane par défaut : robuste)
        n_missing = int(df_num.isnull().sum().sum())
        if n_missing > 0:
            imputer = SimpleImputer(strategy=self.imputation_strategy)
            df_imputed = pd.DataFrame(
                imputer.fit_transform(df_num),
                columns=df_num.columns,
                index=df_num.index,
            )
        else:
            df_imputed = df_num.copy()
        report.imputed_values = n_missing
        logger.info(f"[Sanitizer] {n_missing} valeur(s) imputée(s).")

        # Étape 4 — Suppression outliers (z-score absolu > seuil)
        if len(df_imputed) > 1 and df_imputed.shape[1] > 0:
            z = np.abs(zscore(df_imputed, nan_policy="omit"))
            mask = (z < self.zscore_threshold).all(axis=1)
            df_clean = df_imputed[mask].reset_index(drop=True)
            outliers_removed = int((~mask).sum())
        else:
            df_clean = df_imputed.copy()
            outliers_removed = 0
        report.outliers_removed = outliers_removed
        logger.info(f"[Sanitizer] {outliers_removed} ligne(s) outlier(s) supprimée(s).")

        # Rapport final
        report.n_rows_output = len(df_clean)
        report.n_cols_output = df_clean.shape[1]
        logger.info(
            f"[Sanitizer] Terminé : {report.n_rows_output} lignes × {report.n_cols_output} colonnes."
        )

        return df_clean, report

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def check_minimum_requirements(df: pd.DataFrame, min_rows: int = 10, min_cols: int = 2) -> dict:
        """Vérifie que le DataFrame est exploitable pour l'analyse."""
        issues = []
        if len(df) < min_rows:
            issues.append(f"Trop peu de lignes ({len(df)} < {min_rows} minimum).")
        if df.shape[1] < min_cols:
            issues.append(f"Trop peu de colonnes ({df.shape[1]} < {min_cols} minimum).")
        return {"valid": len(issues) == 0, "issues": issues}
