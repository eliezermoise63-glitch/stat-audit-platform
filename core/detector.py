"""
core/detector.py
----------------
Détection automatique du type statistique de chaque variable d'un DataFrame.

Types détectés :
  - 'continue'      : variable numérique continue (ACP, AF, AFDM)
  - 'categorielle'  : variable qualitative à faible cardinalité (ACM, AFDM)
  - 'binaire'       : variable à exactement 2 valeurs (traitée comme catégorielle)
  - 'ignoree'       : colonne non numérique ou quasi-constante

Design : stateless, testable unitairement, aucune dépendance à Streamlit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VariableType = Literal["continue", "categorielle", "binaire", "ignoree"]


@dataclass
class DetectionReport:
    """Résultat de la détection automatique des types de variables."""

    types: Dict[str, VariableType] = field(default_factory=dict)

    @property
    def continues(self) -> List[str]:
        """Liste des variables continues (ACP + AF)."""
        return [c for c, t in self.types.items() if t == "continue"]

    @property
    def categorielles(self) -> List[str]:
        """Liste des variables catégorielles et binaires (ACM)."""
        return [c for c, t in self.types.items() if t in ("categorielle", "binaire")]

    @property
    def ignorees(self) -> List[str]:
        """Liste des colonnes ignorées (non numériques ou quasi-constantes)."""
        return [c for c, t in self.types.items() if t == "ignoree"]

    @property
    def is_mixed(self) -> bool:
        """True si le dataset contient à la fois des variables continues et catégorielles (→ AFDM)."""
        return len(self.continues) > 0 and len(self.categorielles) > 0

    def summary(self) -> str:
        """Résumé lisible du rapport de détection."""
        lines = [
            f"Variables continues ({len(self.continues)}) : {', '.join(self.continues) or '—'}",
            f"Variables catégorielles ({len(self.categorielles)}) : {', '.join(self.categorielles) or '—'}",
            f"Variables ignorées ({len(self.ignorees)}) : {', '.join(self.ignorees) or '—'}",
            f"Dataset mixte : {'oui → AFDM recommandée' if self.is_mixed else 'non'}",
        ]
        return "\n".join(lines)


class VariableDetector:
    """
    Détecte automatiquement le type statistique de chaque colonne d'un DataFrame.

    Paramètres
    ----------
    categorical_threshold : float
        Ratio n_unique / n_rows en dessous duquel une variable numérique est
        considérée catégorielle (défaut 0.05 = 5%).
    min_unique : int
        Nombre minimum de valeurs uniques pour qu'une variable soit considérée
        continue. En dessous de ce seuil, la colonne est ignorée (quasi-constante).
        (défaut 10)
    """

    def __init__(
        self,
        categorical_threshold: float = 0.05,
        min_unique: int = 10,
    ) -> None:
        """
        Initialise le détecteur avec les seuils de classification.

        Paramètres
        ----------
        categorical_threshold : float
            Ratio n_unique / n_rows en dessous duquel une variable numérique
            est traitée comme catégorielle (défaut 0.05).
        min_unique : int
            Seuil minimum de valeurs uniques pour classer une variable comme
            continue. En dessous, elle est ignorée (défaut 10).
        """
        if not 0 < categorical_threshold < 1:
            raise ValueError(
                f"categorical_threshold doit être entre 0 et 1, reçu : {categorical_threshold}"
            )
        if min_unique < 2:
            raise ValueError(
                f"min_unique doit être >= 2, reçu : {min_unique}"
            )
        self.categorical_threshold = categorical_threshold
        self.min_unique = min_unique

    def detect(self, df: pd.DataFrame) -> DetectionReport:
        """
        Analyse chaque colonne du DataFrame et retourne un rapport de détection.

        Règles appliquées dans l'ordre :
          1. Colonne non numérique → ignorée
          2. n_unique == 2 → binaire (ex: 0/1, True/False)
          3. n_unique / n_rows < categorical_threshold → catégorielle
          4. n_unique < min_unique → ignorée (quasi-constante)
          5. Sinon → continue

        Paramètres
        ----------
        df : pd.DataFrame
            DataFrame brut ou fiabilisé à analyser.

        Retourne
        --------
        DetectionReport
            Rapport contenant le type de chaque colonne et les listes par catégorie.
        """
        report = DetectionReport()
        n_rows = max(len(df), 1)

        for col in df.columns:
            col_type = self._classify_column(df[col], n_rows)
            report.types[col] = col_type

        logger.info(
            f"[Detector] {len(report.continues)} continue(s), "
            f"{len(report.categorielles)} catégorielle(s), "
            f"{len(report.ignorees)} ignorée(s). "
            f"Mixte : {report.is_mixed}."
        )
        return report

    def _classify_column(self, series: pd.Series, n_rows: int) -> VariableType:
        """
        Classifie une série pandas selon son type statistique.

        Paramètres
        ----------
        series : pd.Series
            Colonne à classifier.
        n_rows : int
            Nombre total de lignes du DataFrame (pour le calcul du ratio).

        Retourne
        --------
        VariableType
            Type détecté : 'continue', 'categorielle', 'binaire' ou 'ignoree'.
        """
        # Règle 1 : colonne non numérique → ignorée
        if not pd.api.types.is_numeric_dtype(series):
            return "ignoree"

        n_unique = series.nunique(dropna=True)

        # Règle 2a : colonne constante (0 ou 1 valeur unique) → ignorée
        if n_unique <= 1:
            return "ignoree"

        # Règle 2b : exactement 2 valeurs uniques → binaire
        if n_unique == 2:
            return "binaire"

        # Règle 3 : ratio faible → catégorielle
        ratio = n_unique / n_rows
        if ratio < self.categorical_threshold:
            return "categorielle"

        # Règle 4 : trop peu de valeurs uniques → ignorée (quasi-constante)
        if n_unique < self.min_unique:
            return "ignoree"

        # Règle 5 : par défaut → continue
        return "continue"
