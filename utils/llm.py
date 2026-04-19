"""
utils/llm.py
------------
Interface avec l'API Claude (Anthropic) pour l'interprétation métier.

Séparation claire : inférence statistique (engine.py) ≠ interprétation sémantique (ici).
"""

from __future__ import annotations

import logging

import pandas as pd

try:
    import anthropic as _anthropic
except ImportError:  # pragma: no cover
    _anthropic = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4000


def build_fa_prompt(
    fa_loadings: pd.DataFrame,
    communalities: pd.Series,
    kmo_score: float,
    bartlett_p: float,
    context: str = "",
    rotation: str = "varimax",
) -> str:
    """
    Construit le prompt d'interprétation pour l'Analyse Factorielle.

    Paramètres
    ----------
    fa_loadings : pd.DataFrame
        Matrice des saturations après rotation (n_features × n_factors).
    communalities : pd.Series
        Communautés par variable (proportion de variance expliquée par les facteurs).
    kmo_score : float
        Indice Kaiser-Meyer-Olkin (entre 0 et 1).
    bartlett_p : float
        P-value du test de sphéricité de Bartlett.
    context : str, optionnel
        Contexte métier libre pour orienter l'interprétation.
    rotation : str, optionnel
        Rotation appliquée ('varimax' ou 'promax'). Mentionnée dans le prompt
        pour que Claude adapte son interprétation (corrélation entre facteurs si promax).

    Retourne
    --------
    str
        Prompt formaté prêt à être envoyé à l'API Claude.
    """
    loadings_str = fa_loadings.round(3).to_string()
    communalities_str = communalities.round(3).to_string()
    context_block = f"Contexte métier : {context}\n" if context.strip() else ""
    rotation_label = rotation.capitalize()
    rotation_note = (
        "Les facteurs sont supposés indépendants (rotation orthogonale)."
        if rotation == "varimax"
        else "Les facteurs peuvent être corrélés entre eux (rotation oblique)."
    )

    return f"""Tu es un expert en statistiques appliquées et en analyse de données métier.
Réponds en français uniquement en texte courant, comme un rapport rédigé.
Aucun titre, aucun sous-titre, aucune liste à puces, aucun emoji, aucune mise en gras.
Rédige des paragraphes fluides et concis, séparés par des sauts de ligne.
Maximum 300 mots au total.

{context_block}
Indice KMO : {kmo_score:.3f} ({'Bonne adéquation' if kmo_score >= 0.7 else 'Adéquation acceptable' if kmo_score >= 0.6 else 'Adéquation faible'})
Test de Bartlett (p-value) : {bartlett_p:.4f} ({'Corrélations significatives' if bartlett_p < 0.05 else 'Corrélations non significatives'})
Rotation : {rotation_label} — {rotation_note}

Matrice des saturations (rotation {rotation_label}) :
{loadings_str}

Communautés par variable :
{communalities_str}

Nomme chaque facteur en identifiant le concept sous-jacent à partir des variables dont le loading dépasse 0.4 en valeur absolue.
Interprète la communauté de chaque variable, en signalant celles inférieures à 0.3 comme mal représentées.
Donne 2 à 3 recommandations concrètes.
Identifie les risques principaux.

Aucun titre, aucune liste, aucun emoji. Texte courant uniquement."""


def build_pca_prompt(
    loadings: pd.DataFrame,
    explained_variance: list,
    n_components: int,
    context: str = "",
) -> str:
    """
    Construit le prompt d'interprétation pour l'ACP.

    Paramètres
    ----------
    loadings : pd.DataFrame
        Matrice des contributions des variables aux composantes (n_features × n_components).
    explained_variance : list
        Liste des ratios de variance expliquée par composante (valeurs entre 0 et 1).
    n_components : int
        Nombre de composantes retenues.
    context : str, optionnel
        Contexte métier libre pour orienter l'interprétation.

    Retourne
    --------
    str
        Prompt formaté prêt à être envoyé à l'API Claude.
    """

    loadings_str = loadings.round(3).to_string()
    variance_info = "\n".join(
        [f"PC{i+1} : {v*100:.1f}%" for i, v in enumerate(explained_variance)]
    )
    context_block = f"Contexte : {context}\n" if context.strip() else ""

    return f"""Tu es expert en analyse de données.
Réponds en français uniquement en texte courant, comme un rapport rédigé.
Aucun titre, aucun sous-titre, aucune liste, aucun emoji, aucune mise en gras.
Paragraphes fluides et concis. Maximum 200 mots au total.

{context_block}
Composantes principales retenues : {n_components}
Variance expliquée par composante :
{variance_info}

Matrice des contributions (loadings) :
{loadings_str}

Identifie ce que représente chaque composante à partir des variables qui y contribuent le plus.
Evalue si la réduction est satisfaisante.
Propose comment utiliser ces composantes pour la modélisation ou la segmentation.

Aucun titre, aucune liste, aucun emoji. Texte courant uniquement."""


def build_acm_prompt(
    column_coordinates: pd.DataFrame,
    inertia_summary: pd.DataFrame,
    variables: list,
    context: str = "",
) -> str:
    """
    Construit le prompt d'interprétation pour l'ACM.

    Paramètres
    ----------
    column_coordinates : pd.DataFrame
        Coordonnées des modalités sur les axes factoriels.
    inertia_summary : pd.DataFrame
        Tableau d'inertie par composante (valeur propre, %, cumulé).
    variables : list
        Noms des variables catégorielles analysées.
    context : str, optionnel
        Contexte métier libre pour orienter l'interprétation.

    Retourne
    --------
    str
        Prompt formaté prêt à être envoyé à l'API Claude.
    """
    coords_str = column_coordinates.round(3).to_string()
    inertia_str = inertia_summary.to_string()
    context_block = f"Contexte métier : {context}\n" if context.strip() else ""
    vars_str = ", ".join(variables)

    return f"""Tu es un expert en statistiques appliquées et en analyse des données qualitatives.
Réponds en français uniquement en texte courant, comme un rapport rédigé.
Aucun titre, aucun sous-titre, aucune liste à puces, aucun emoji, aucune mise en gras.
Rédige des paragraphes fluides et concis, séparés par des sauts de ligne.
Maximum 300 mots au total.

{context_block}
Variables analysées : {vars_str}

Tableau d'inertie par axe factoriel :
{inertia_str}

Coordonnées des modalités sur les axes factoriels :
{coords_str}

Identifie ce qui oppose les modalités sur chaque axe factoriel (quelles modalités sont aux extrêmes, que signifie cet axe).
Repère les modalités proches entre elles et explique ce que cette proximité signifie.
Interprète le premier axe comme le gradient principal de différenciation.
Donne 2 recommandations concrètes issues de cette analyse.

Aucun titre, aucune liste, aucun emoji. Texte courant uniquement."""


def build_afdm_prompt(
    column_coordinates: pd.DataFrame,
    inertia_summary: pd.DataFrame,
    continuous_cols: list,
    categorical_cols: list,
    context: str = "",
) -> str:
    """
    Construit le prompt d'interprétation pour l'AFDM.

    Paramètres
    ----------
    column_coordinates : pd.DataFrame
        Coordonnées des variables continues et modalités catégorielles sur les axes.
    inertia_summary : pd.DataFrame
        Tableau d'inertie par composante (valeur propre, %, cumulé).
    continuous_cols : list
        Noms des variables continues incluses dans l'analyse.
    categorical_cols : list
        Noms des variables catégorielles incluses dans l'analyse.
    context : str, optionnel
        Contexte métier libre pour orienter l'interprétation.

    Retourne
    --------
    str
        Prompt formaté prêt à être envoyé à l'API Claude.
    """
    coords_str = column_coordinates.round(3).to_string()
    inertia_str = inertia_summary.to_string()
    context_block = f"Contexte métier : {context}\n" if context.strip() else ""
    cont_str = ", ".join(continuous_cols)
    cat_str = ", ".join(categorical_cols)

    return f"""Tu es un expert en statistiques appliquées spécialisé dans l'analyse de données mixtes.
Réponds en français uniquement en texte courant, comme un rapport rédigé.
Aucun titre, aucun sous-titre, aucune liste à puces, aucun emoji, aucune mise en gras.
Rédige des paragraphes fluides et concis, séparés par des sauts de ligne.
Maximum 350 mots au total.

{context_block}
Variables continues : {cont_str}
Variables catégorielles : {cat_str}

Tableau d'inertie par axe factoriel :
{inertia_str}

Coordonnées des variables et modalités sur les axes factoriels :
{coords_str}

Identifie ce que représente chaque axe factoriel en tenant compte à la fois des variables continues et des modalités catégorielles qui y contribuent.
Décris les associations entre variables continues et modalités catégorielles qui se dégagent de l'espace factoriel commun.
Identifie les profils types d'individus qui se distinguent sur ces axes.
Donne 2 à 3 recommandations concrètes pour la décision ou la modélisation.

Aucun titre, aucune liste, aucun emoji. Texte courant uniquement."""


def call_claude(
    prompt: str,
    api_key: str,
    model: str = CLAUDE_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Envoie un prompt à l'API Claude et retourne le texte de la réponse.

    Paramètres
    ----------
    prompt : str
        Texte du prompt à envoyer.
    api_key : str
        Clé API Anthropic (récupérée depuis st.secrets).
    model : str
        Identifiant du modèle Claude à utiliser (défaut : CLAUDE_MODEL).
    max_tokens : int
        Nombre maximum de tokens dans la réponse (défaut : MAX_TOKENS).

    Retourne
    --------
    str
        Texte de la réponse générée par Claude.

    Raises
    ------
    ImportError
        Si le package 'anthropic' n'est pas installé.
    ValueError
        Si la clé API est absente ou vide.
    """
    if _anthropic is None:
        raise ImportError("Le package 'anthropic' n'est pas installé.")

    if not api_key or api_key.strip() == "":
        raise ValueError("Clé API Anthropic manquante.")

    client = _anthropic.Anthropic(api_key=api_key)
    logger.info(f"[LLM] Appel Claude ({model}) — {len(prompt)} caractères.")

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text: str = message.content[0].text
    logger.info(f"[LLM] Réponse reçue ({len(response_text)} caractères).")
    return response_text