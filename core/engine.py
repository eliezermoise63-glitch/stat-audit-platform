"""
core/engine.py
--------------
Moteur d'Analyse Multivariée.

Fonctionnalités :
  - ACP (PCA) avec sélection automatique du nombre de composantes
  - Tableau Top-N variables par composante ACP
  - Analyse Factorielle (FA) avec critère de Kaiser + rotation Varimax ou Promax
  - Validation statistique : KMO + Test de sphéricité de Bartlett
  - Matrice de corrélation avec test de normalité (Shapiro-Wilk) →
    choix automatique Pearson (données normales) ou Spearman (données non normales)
  - ACM (Analyse des Correspondances Multiples) via prince
  - AFDM (Analyse Factorielle des Données Mixtes) via prince

Design : stateless par méthode, chaque appel retourne un résultat complet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (
    calculate_bartlett_sphericity,
    calculate_kmo,
)

try:
    import prince as _prince
except ImportError:  # pragma: no cover
    _prince = None  # type: ignore[assignment]

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
    variance_threshold_reached: bool  # False si le seuil n'a pas pu être atteint


@dataclass
class FAResult:
    loadings: pd.DataFrame            # (n_features, n_factors)
    communalities: pd.Series
    n_factors: int
    eigenvalues: np.ndarray
    kmo_score: float
    bartlett_p_value: float
    fa_valid: bool                    # True si KMO > 0.6 et Bartlett p < 0.05
    rotation: str                     # "varimax" ou "promax"


@dataclass
class CorrelationResult:
    corr_matrix: pd.DataFrame         # matrice de corrélation (Pearson ou Spearman)
    p_matrix: pd.DataFrame            # matrice de p-values
    method: str                       # "pearson" ou "spearman"
    normality_results: pd.DataFrame   # résultats Shapiro-Wilk par variable
    all_normal: bool                  # True si toutes les variables passent Shapiro (α=0.05)


@dataclass
class ACMResult:
    row_coordinates: pd.DataFrame     # coordonnées des individus (n_samples × n_components)
    column_coordinates: pd.DataFrame  # coordonnées des modalités (n_modalities × n_components)
    inertia_summary: pd.DataFrame     # inertie et % expliqué par composante
    n_components: int                 # nombre de composantes retenues
    total_inertia_explained: float    # % d'inertie cumulée expliquée
    variables: List[str]              # variables catégorielles utilisées


@dataclass
class AFDMResult:
    row_coordinates: pd.DataFrame     # coordonnées des individus (n_samples × n_components)
    column_coordinates: pd.DataFrame  # coordonnées des variables/modalités
    inertia_summary: pd.DataFrame     # inertie et % expliqué par composante
    n_components: int                 # nombre de composantes retenues
    total_inertia_explained: float    # % d'inertie cumulée expliquée
    continuous_cols: List[str]        # variables continues utilisées
    categorical_cols: List[str]       # variables catégorielles utilisées


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
        """
        Initialise le moteur et standardise les données.

        Paramètres
        ----------
        df : pd.DataFrame
            Données fiabilisées (sortie de DataSanitizer). Seules les colonnes
            numériques sont conservées.
        variance_threshold : float
            Seuil de variance cumulée cible pour l'ACP automatique (défaut 0.80 = 80%).
            Si le seuil n'est pas atteignable, toutes les composantes sont retenues.
        kmo_threshold : float
            Seuil KMO minimum pour valider l'Analyse Factorielle (défaut 0.60).

        Raises
        ------
        ValueError
            Si le DataFrame est vide ou contient moins de 2 colonnes numériques.
        """
        if df.empty:
            raise ValueError("Le DataFrame fourni au moteur est vide.")

        # Données complètes (pour ACM et AFDM qui ont besoin des catégorielles)
        self.data_full: pd.DataFrame = df.copy()

        # Données numériques uniquement (pour ACP, AF, corrélations)
        self.data: pd.DataFrame = df.select_dtypes(include=[np.number]).copy()
        # Remplacement défensif des inf et NaN résiduels
        self.data = self.data.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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

        # 2. Nombre de composantes pour 80% de variance (référence fixe)
        indices_80 = np.where(cumulative_variance >= 0.80)[0]
        n_components_80 = int(indices_80[0]) + 1 if len(indices_80) > 0 else self.data.shape[1]

        # 3. Nombre de composantes selon le seuil défini
        # np.where évite le bug silencieux de np.argmax qui retourne 0
        # quand aucune valeur ne satisfait la condition.
        indices = np.where(cumulative_variance >= self.variance_threshold)[0]
        if len(indices) == 0:
            n_components = self.data.shape[1]
            variance_threshold_reached = False
            logger.warning(
                f"[PCA] Seuil de variance {self.variance_threshold*100:.0f}% non atteint "
                f"— toutes les {n_components} composantes retenues."
            )
        else:
            n_components = int(indices[0]) + 1
            variance_threshold_reached = True

        n_components = max(2, min(n_components, self.data.shape[1]))

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
            variance_threshold_reached=variance_threshold_reached,
        )

    def top_variables_per_component(
        self, pca_result: PCAResult, n_top: int = 5
    ) -> pd.DataFrame:
        """
        Retourne un tableau des N variables les plus contributives par composante ACP.

        Pour chaque composante, les variables sont triées par valeur absolue
        de leur loading, du plus fort au plus faible.

        Paramètres
        ----------
        pca_result : PCAResult
            Résultat d'un appel à run_pca().
        n_top : int
            Nombre de variables à afficher par composante (défaut 5).
            Ajusté automatiquement si le dataset a moins de n_top variables.

        Retourne
        --------
        pd.DataFrame
            Index = rang (#1 à #n_top), colonnes = PC1, PC2, ...
            Chaque cellule : "variable (+loading)"
        """
        # Ajuster n_top si le dataset a moins de variables que demandé
        n_top = min(n_top, len(pca_result.loadings))

        rows = []
        for component in pca_result.loadings.columns:
            col = pca_result.loadings[component]
            top = col.abs().nlargest(n_top)
            rows.append([
                f"{var} ({pca_result.loadings.loc[var, component]:+.3f})"
                for var in top.index
            ])

        df_top = pd.DataFrame(rows, index=pca_result.loadings.columns).T
        df_top.index = [f"#{i+1}" for i in range(n_top)]
        return df_top

    # ── Analyse Factorielle ───────────────────────────────────────────────────

    def run_factor_analysis(
        self,
        n_factors: Optional[int] = None,
        rotation: Literal["varimax", "promax"] = "varimax",
    ) -> FAResult:
        """
        Analyse Factorielle avec validation KMO + Bartlett et rotation choisie.

        Varimax (orthogonale) suppose que les facteurs sont indépendants.
        Promax (oblique) est plus adapté quand les facteurs peuvent être corrélés,
        cas fréquent en sciences humaines et sociales.

        Paramètres
        ----------
        n_factors : int, optionnel
            Nombre de facteurs à extraire. Si None, critère de Kaiser (valeurs propres > 1).
        rotation : str
            Rotation à appliquer : 'varimax' (défaut) ou 'promax'.

        Retourne
        --------
        FAResult
        """
        # 1. Validation préalable
        kmo_score, bartlett_p = self._validate_fa()
        fa_valid = (kmo_score >= self.kmo_threshold) and (bartlett_p < 0.05)

        if not fa_valid:
            logger.warning(
                f"[FA] Conditions non idéales : KMO={kmo_score:.3f}, Bartlett p={bartlett_p:.4f}. "
                "Les résultats sont à interpréter avec prudence."
            )

        # 2. DataFrame pour factor_analyzer (évite le conflit check_array() avec sklearn >= 1.6)
        # Remplissage défensif des NaN et inf résiduels après standardisation
        data_for_fa = pd.DataFrame(self.data_scaled, columns=self.data.columns)
        data_for_fa = data_for_fa.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # Suppression des colonnes à variance nulle (provoquent des inf dans corr_mtx)
        zero_var_cols = data_for_fa.columns[data_for_fa.std() < 1e-10].tolist()
        if zero_var_cols:
            logger.warning(f"[FA] Colonnes à variance nulle supprimées : {zero_var_cols}")
            data_for_fa = data_for_fa.drop(columns=zero_var_cols)
        if data_for_fa.shape[1] < 2:
            raise ValueError(
                "Pas assez de variables exploitables pour l'Analyse Factorielle "
                "après suppression des colonnes à variance nulle."
            )

        # 3. Choix automatique du nombre de facteurs (critère de Kaiser)
        if n_factors is None:
            fa_temp = FactorAnalyzer(rotation=None)
            fa_temp.fit(data_for_fa)
            eigenvalues, _ = fa_temp.get_eigenvalues()
            n_factors = max(1, int(np.sum(eigenvalues > 1)))
            logger.info(f"[FA] Critère de Kaiser : {n_factors} facteur(s) retenus.")

        n_factors = min(n_factors, data_for_fa.shape[1] - 1)

        # 4. AF avec rotation choisie
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(data_for_fa)

        loadings = pd.DataFrame(
            fa.loadings_,
            columns=[f"F{i + 1}" for i in range(n_factors)],
            index=data_for_fa.columns,
        )
        communalities = pd.Series(
            fa.get_communalities(), index=data_for_fa.columns, name="Communauté"
        )
        eigenvalues_full, _ = fa.get_eigenvalues()

        logger.info(
            f"[FA] AF complète : {n_factors} facteur(s), rotation={rotation}, KMO={kmo_score:.3f}."
        )

        return FAResult(
            loadings=loadings,
            communalities=communalities,
            n_factors=n_factors,
            eigenvalues=eigenvalues_full,
            kmo_score=kmo_score,
            bartlett_p_value=float(bartlett_p),
            fa_valid=fa_valid,
            rotation=rotation,
        )

    # ── ACM ──────────────────────────────────────────────────────────────────

    def run_acm(
        self,
        categorical_cols: List[str],
        n_components: int = 2,
    ) -> ACMResult:
        """
        Analyse des Correspondances Multiples (ACM) sur les variables catégorielles.

        L'ACM est l'extension de l'ACP aux variables qualitatives. Elle visualise
        les ressemblances entre individus et entre modalités dans un espace factoriel.
        Les variables numériques à faible cardinalité (détectées comme catégorielles
        par VariableDetector) sont converties en chaînes de caractères avant l'analyse.

        Paramètres
        ----------
        categorical_cols : List[str]
            Noms des colonnes catégorielles à analyser (issues de DetectionReport).
        n_components : int
            Nombre de composantes factorielles à calculer (défaut 2).

        Retourne
        --------
        ACMResult

        Raises
        ------
        ImportError
            Si le package 'prince' n'est pas installé.
        ValueError
            Si aucune colonne catégorielle n'est fournie ou disponible dans les données.
        """
        if _prince is None:
            raise ImportError(
                "Le package 'prince' est requis pour l'ACM. "
                "Installez-le avec : pip install prince"
            )

        # Sélection et validation des colonnes dans le DataFrame complet
        cols_available = [c for c in categorical_cols if c in self.data_full.columns]
        if not cols_available:
            raise ValueError(
                f"Aucune des colonnes catégorielles demandées n'est disponible. "
                f"Demandées : {categorical_cols}. Disponibles : {list(self.data_full.columns)}"
            )

        if len(cols_available) < 2:
            raise ValueError(
                "L'ACM nécessite au moins 2 variables catégorielles. "
                f"Colonne disponible : {cols_available}"
            )

        # Conversion en chaînes (prince.MCA exige des colonnes de type objet/catégorie)
        df_cat = self.data_full[cols_available].copy()
        for col in df_cat.columns:
            df_cat[col] = df_cat[col].astype(str)

        n_components = min(n_components, len(cols_available))

        # Ajustement : prince MCA attend n_components <= n_cols - 1
        n_components = min(n_components, len(cols_available) - 1)
        n_components = max(n_components, 1)

        logger.info(
            f"[ACM] {len(cols_available)} variable(s), {n_components} composante(s)."
        )

        # Fit MCA
        mca = _prince.MCA(n_components=n_components, random_state=42)
        mca = mca.fit(df_cat)

        # Coordonnées des individus (API prince >= 0.7 : row_coordinates(df) ou row_coordinates_)
        if hasattr(mca, "row_coordinates_"):
            row_coords = mca.row_coordinates_
        else:
            row_coords = mca.row_coordinates(df_cat)
        row_coords = row_coords.copy()
        row_coords.columns = [f"Dim{i+1}" for i in range(n_components)]

        # Coordonnées des modalités (API prince >= 0.7 : column_coordinates_ ou column_coordinates)
        if hasattr(mca, "column_coordinates_"):
            col_coords = mca.column_coordinates_
        else:
            col_coords = mca.column_coordinates(df_cat)
        col_coords = col_coords.copy()
        col_coords.columns = [f"Dim{i+1}" for i in range(n_components)]

        # Inertie par composante — forcer numpy array pour éviter les problèmes d'index pandas
        eigenvalues = np.array(mca.eigenvalues_).flatten()
        total_inertia = float(eigenvalues.sum()) if eigenvalues.sum() > 0 else 1.0
        inertia_pct = eigenvalues / total_inertia
        inertia_cum = np.cumsum(inertia_pct)

        inertia_summary = pd.DataFrame({
            "Valeur propre": np.round(eigenvalues, 4),
            "Inertie (%)": np.round(inertia_pct * 100, 2),
            "Inertie cumulée (%)": np.round(inertia_cum * 100, 2),
        }, index=[f"Dim{i+1}" for i in range(n_components)])

        total_explained = float(inertia_cum[-1]) if len(inertia_cum) > 0 else 0.0

        logger.info(
            f"[ACM] Terminée : {total_explained*100:.1f}% d'inertie expliquée."
        )

        return ACMResult(
            row_coordinates=row_coords,
            column_coordinates=col_coords,
            inertia_summary=inertia_summary,
            n_components=n_components,
            total_inertia_explained=total_explained,
            variables=cols_available,
        )

    # ── AFDM ─────────────────────────────────────────────────────────────────

    def run_afdm(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
        n_components: int = 2,
    ) -> AFDMResult:
        """
        Analyse Factorielle des Données Mixtes (AFDM) via prince.FAMD.

        L'AFDM généralise l'ACP (variables continues) et l'ACM (variables catégorielles)
        aux datasets mixtes. Elle produit un espace factoriel commun où individus,
        variables continues et modalités catégorielles sont représentés ensemble.

        Paramètres
        ----------
        continuous_cols : List[str]
            Noms des colonnes continues à inclure (issues de DetectionReport).
        categorical_cols : List[str]
            Noms des colonnes catégorielles à inclure (issues de DetectionReport).
        n_components : int
            Nombre de composantes factorielles à calculer (défaut 2).

        Retourne
        --------
        AFDMResult

        Raises
        ------
        ImportError
            Si le package 'prince' n'est pas installé.
        ValueError
            Si les colonnes demandées sont absentes ou insuffisantes.
        """
        if _prince is None:
            raise ImportError(
                "Le package 'prince' est requis pour l'AFDM. "
                "Installez-le avec : pip install prince"
            )

        # Validation et sélection des colonnes dans le DataFrame complet
        cont_available = [c for c in continuous_cols if c in self.data_full.columns]
        cat_available = [c for c in categorical_cols if c in self.data_full.columns]

        if not cont_available:
            raise ValueError(
                f"Aucune colonne continue disponible. Demandées : {continuous_cols}"
            )
        if not cat_available:
            raise ValueError(
                f"Aucune colonne catégorielle disponible. Demandées : {categorical_cols}"
            )

        # Construction du DataFrame mixte
        # — continues : valeurs numériques telles quelles
        # — catégorielles : converties en str pour que prince.FAMD les traite correctement
        df_mixed = pd.concat([
            self.data_full[cont_available],
            self.data_full[cat_available].astype(str),
        ], axis=1)

        n_components = min(n_components, df_mixed.shape[1] - 1)
        n_components = max(n_components, 1)

        logger.info(
            f"[AFDM] {len(cont_available)} continue(s), {len(cat_available)} catégorielle(s), "
            f"{n_components} composante(s)."
        )

        # Fit FAMD (Factor Analysis of Mixed Data)
        famd = _prince.FAMD(n_components=n_components, random_state=42)
        famd = famd.fit(df_mixed)

        # Coordonnées des individus
        if hasattr(famd, "row_coordinates_"):
            row_coords = famd.row_coordinates_
        else:
            row_coords = famd.row_coordinates(df_mixed)
        row_coords = row_coords.copy()
        row_coords.columns = [f"Dim{i+1}" for i in range(n_components)]

        # Coordonnées des variables (continues) et modalités (catégorielles)
        if hasattr(famd, "column_coordinates_"):
            col_coords = famd.column_coordinates_
        else:
            col_coords = famd.column_coordinates(df_mixed)
        col_coords = col_coords.copy()
        col_coords.columns = [f"Dim{i+1}" for i in range(n_components)]

        # Inertie par composante — forcer numpy array pour éviter les problèmes d'index pandas
        eigenvalues = np.array(famd.eigenvalues_).flatten()
        total_inertia = float(eigenvalues.sum()) if eigenvalues.sum() > 0 else 1.0
        inertia_pct = eigenvalues / total_inertia
        inertia_cum = np.cumsum(inertia_pct)

        inertia_summary = pd.DataFrame({
            "Valeur propre": np.round(eigenvalues, 4),
            "Inertie (%)": np.round(inertia_pct * 100, 2),
            "Inertie cumulée (%)": np.round(inertia_cum * 100, 2),
        }, index=[f"Dim{i+1}" for i in range(n_components)])

        total_explained = float(inertia_cum[-1]) if len(inertia_cum) > 0 else 0.0

        logger.info(
            f"[AFDM] Terminée : {total_explained*100:.1f}% d'inertie expliquée."
        )

        return AFDMResult(
            row_coordinates=row_coords,
            column_coordinates=col_coords,
            inertia_summary=inertia_summary,
            n_components=n_components,
            total_inertia_explained=total_explained,
            continuous_cols=cont_available,
            categorical_cols=cat_available,
        )

    # ── Corrélations ─────────────────────────────────────────────────────────

    def compute_correlation_matrix(self) -> CorrelationResult:
        """
        Calcule la matrice de corrélation avec sélection automatique de la méthode.

        Pipeline :
          1. Test de normalité Shapiro-Wilk sur chaque variable (α = 0.05).
             Shapiro est fiable pour n < 5000 ; au-delà, Spearman est retenu par défaut.
          2. Si toutes les variables sont normales → Pearson (paramétrique).
             Sinon → Spearman (non paramétrique, robuste aux non-normalités).
          3. P-values calculées sur le triangle supérieur puis symétrisation.

        Retourne
        --------
        CorrelationResult
            Inclut la méthode retenue et les résultats Shapiro-Wilk pour justification.
        """
        normality_rows = []
        for col in self.data.columns:
            values = self.data[col].dropna().values
            if len(values) < 3:
                stat, p_val, is_normal = np.nan, np.nan, False
            elif len(values) > 5000:
                stat, p_val, is_normal = np.nan, np.nan, False
            else:
                stat, p_val = stats.shapiro(values)
                is_normal = bool(p_val >= 0.05)
            normality_rows.append({
                "Variable": col,
                "Stat Shapiro-Wilk": round(stat, 4) if not np.isnan(stat) else "N/A",
                "p-value": round(p_val, 4) if not np.isnan(p_val) else "N/A",
                "Normale (α=0.05)": "✅ Oui" if is_normal else "❌ Non",
            })

        normality_df = pd.DataFrame(normality_rows).set_index("Variable")
        all_normal = all(r["Normale (α=0.05)"] == "✅ Oui" for r in normality_rows)
        method = "pearson" if all_normal else "spearman"

        logger.info(
            f"[Corrélation] Shapiro-Wilk → toutes normales : {all_normal}. "
            f"Méthode retenue : {method}."
        )

        corr_matrix = self.data.corr(method=method)

        n = self.data.shape[1]
        cols = self.data.columns
        p_matrix = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
        corr_fn = stats.pearsonr if method == "pearson" else stats.spearmanr

        for i in range(n):
            for j in range(i + 1, n):
                _, p = corr_fn(self.data.iloc[:, i], self.data.iloc[:, j])
                p_matrix.iloc[i, j] = p
                p_matrix.iloc[j, i] = p

        return CorrelationResult(
            corr_matrix=corr_matrix,
            p_matrix=p_matrix,
            method=method,
            normality_results=normality_df,
            all_normal=all_normal,
        )

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
        """
        Calcule KMO et test de Bartlett pour valider l'Analyse Factorielle.

        Utilise un DataFrame pandas (plutôt que le numpy array data_scaled) pour
        éviter le conflit check_array() entre factor_analyzer et scikit-learn >= 1.6.

        Retourne
        --------
        Tuple[float, float]
            (kmo_score, bartlett_p_value). En cas d'erreur : (0.0, 1.0).
        """
        try:
            data_df = pd.DataFrame(self.data_scaled, columns=self.data.columns)
            data_df = data_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Supprimer colonnes à variance nulle avant KMO
            zero_var = data_df.columns[data_df.std() < 1e-10].tolist()
            if zero_var:
                data_df = data_df.drop(columns=zero_var)
            if data_df.shape[1] < 2:
                return 0.0, 1.0
            _, kmo_score = calculate_kmo(data_df)
            _, p_value = calculate_bartlett_sphericity(data_df)
            return float(kmo_score), float(p_value)
        except Exception as e:
            logger.warning(f"[FA Validation] Erreur lors du calcul KMO/Bartlett : {e}")
            return 0.0, 1.0
