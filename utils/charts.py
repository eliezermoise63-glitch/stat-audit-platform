"""
utils/charts.py
---------------
Fonctions de visualisation pour l'audit statistique.
Toutes les fonctions retournent une figure matplotlib pour l'intégration Streamlit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style global cohérent — fond sombre pour correspondre à Streamlit dark theme
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.labelcolor": "#fafafa",
    "xtick.color": "#fafafa",
    "ytick.color": "#fafafa",
    "text.color": "#fafafa",
})

PALETTE_DIVERGING = "RdBu_r"
PALETTE_FA = "PRGn"


# ── Corrélation ───────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    p_matrix: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Matrice de Corrélation",
) -> plt.Figure:
    """
    Heatmap de corrélation avec masque de significativité optionnel.

    Paramètres
    ----------
    corr_matrix : pd.DataFrame
        Matrice de corrélation (Pearson ou Spearman).
    p_matrix : pd.DataFrame, optionnel
        Matrice de p-values. Si fournie, les cases non significatives (p >= 0.05)
        sont affichées vides.
    figsize : tuple
        Dimensions de la figure matplotlib.
    title : str
        Titre du graphique.

    Retourne
    --------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True  # masque triangle supérieur

    annot_data = corr_matrix.copy()
    if p_matrix is not None:
        # CORRECTION : applymap déprécié depuis pandas 2.1, remplacé par map
        annot_data = corr_matrix.map(lambda x: f"{x:.2f}")
        for i in corr_matrix.index:
            for j in corr_matrix.columns:
                if p_matrix.loc[i, j] >= 0.05 and i != j:
                    annot_data.loc[i, j] = ""

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=annot_data if p_matrix is not None else True,
        fmt="" if p_matrix is not None else ".2f",
        cmap=PALETTE_DIVERGING,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    # Rendre le fond des cases masquées identique au fond de la figure
    ax.set_facecolor("#0e1117")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


# ── ACP ───────────────────────────────────────────────────────────────────────

def plot_pca_variance(
    cumulative_variance: np.ndarray,
    n_components_selected: int,
    variance_threshold: float = 0.80,
    figsize: Tuple[int, int] = (9, 4),
) -> plt.Figure:
    """
    Courbe de variance cumulée avec annotation du seuil et du nombre de composantes choisi.
    """
    fig, ax = plt.subplots(figsize=figsize)

    components = np.arange(1, len(cumulative_variance) + 1)

    ax.plot(components, cumulative_variance * 100, "o-", color="#2E75B6", linewidth=2, markersize=6)
    ax.fill_between(components, cumulative_variance * 100, alpha=0.15, color="#2E75B6")

    # Ligne seuil
    ax.axhline(variance_threshold * 100, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Seuil {variance_threshold*100:.0f}%")
    ax.axvline(n_components_selected, color="#27AE60", linestyle=":", linewidth=1.5,
               label=f"Composantes retenues : {n_components_selected}")

    ax.set_xlabel("Nombre de Composantes", fontsize=11)
    ax.set_ylabel("Variance Cumulée (%)", fontsize=11)
    ax.set_title("Variance Cumulée Expliquée — ACP", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xticks(components)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pca_biplot(
    features: np.ndarray,
    loadings: pd.DataFrame,
    figsize: Tuple[int, int] = (9, 7),
) -> plt.Figure:
    """
    Biplot ACP : projection des individus (PC1 × PC2) + vecteurs des variables.

    Paramètres
    ----------
    features : np.ndarray
        Coordonnées des individus dans l'espace factoriel (n_samples × n_components).
    loadings : pd.DataFrame
        Matrice des contributions des variables. Doit contenir 'PC1' et 'PC2'.
    figsize : tuple
        Dimensions de la figure matplotlib.

    Retourne
    --------
    plt.Figure

    Raises
    ------
    ValueError
        Si les colonnes 'PC1' ou 'PC2' sont absentes des loadings.
    """
    if "PC1" not in loadings.columns or "PC2" not in loadings.columns:
        raise ValueError(
            "Le biplot nécessite au moins 2 composantes (PC1 et PC2). "
            f"Colonnes disponibles : {list(loadings.columns)}"
        )
    fig, ax = plt.subplots(figsize=figsize)

    # Points individus
    ax.scatter(features[:, 0], features[:, 1], alpha=0.5, s=25, color="#5DADE2", zorder=2)

    # Vecteurs variables
    scale = np.max(np.abs(features)) * 0.9
    for var in loadings.index:
        x, y = loadings.loc[var, "PC1"] * scale, loadings.loc[var, "PC2"] * scale
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.8),
            zorder=3,
        )
        ax.text(x * 1.07, y * 1.07, var, fontsize=9, color="#2C3E50", ha="center", va="center")

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title("Biplot ACP (PC1 × PC2)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_scree(
    eigenvalues: np.ndarray,
    figsize: Tuple[int, int] = (9, 4),
) -> plt.Figure:
    """
    Diagramme de l'éboulis (Scree Plot) avec ligne de Kaiser (valeur propre = 1).
    """
    fig, ax = plt.subplots(figsize=figsize)
    idx = np.arange(1, len(eigenvalues) + 1)

    ax.bar(idx, eigenvalues, color="#8E44AD", alpha=0.75, edgecolor="white")
    ax.axhline(1, color="#E74C3C", linestyle="--", linewidth=1.5, label="Critère de Kaiser (λ = 1)")
    ax.set_xlabel("Facteur", fontsize=11)
    ax.set_ylabel("Valeur propre (Eigenvalue)", fontsize=11)
    ax.set_title("Diagramme de l'Éboulis (Scree Plot)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(idx)
    fig.tight_layout()
    return fig


# ── Analyse Factorielle ───────────────────────────────────────────────────────

def plot_fa_loadings_heatmap(
    loadings: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 7),
    title: str = "Saturations Factorielles",
    rotation: str = "varimax",
) -> plt.Figure:
    """
    Heatmap des saturations factorielles avec mise en évidence des saturations fortes.

    Les cases dont la valeur absolue dépasse 0.4 sont entourées d'un cadre noir
    pour faciliter l'identification des variables contributives par facteur.

    Paramètres
    ----------
    loadings : pd.DataFrame
        Matrice des saturations après rotation (n_features × n_factors).
    figsize : tuple
        Dimensions de la figure matplotlib.
    title : str
        Titre de base du graphique. La rotation est ajoutée automatiquement.
    rotation : str
        Rotation appliquée ('varimax' ou 'promax'). Ajoutée au titre.

    Retourne
    --------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    full_title = f"{title} ({rotation.capitalize()})"
    annot = loadings.round(2).astype(str)

    sns.heatmap(
        loadings,
        annot=annot,
        fmt="",
        cmap=PALETTE_FA,
        vmin=-1,
        vmax=1,
        center=0,
        square=False,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )

    # Bordures renforcées pour les saturations fortes (> 0.4)
    for y_idx, var in enumerate(loadings.index):
        for x_idx, fac in enumerate(loadings.columns):
            val = abs(loadings.loc[var, fac])
            if val >= 0.4:
                ax.add_patch(
                    plt.Rectangle(
                        (x_idx, y_idx), 1, 1,
                        fill=False, edgecolor="#2C3E50", lw=2, zorder=3,
                    )
                )

    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Facteurs", fontsize=11)
    ax.set_ylabel("Variables", fontsize=11)
    fig.tight_layout()
    return fig
