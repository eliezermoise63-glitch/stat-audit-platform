"""
utils/llm.py
------------
Interface avec l'API Claude (Anthropic) pour l'interprétation métier.

Séparation claire : inférence statistique (engine.py) ≠ interprétation sémantique (ici).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4000


def build_fa_prompt(
    fa_loadings: pd.DataFrame,
    communalities: pd.Series,
    kmo_score: float,
    bartlett_p: float,
    context: str = "",
) -> str:

    loadings_str = fa_loadings.round(3).to_string()
    communalities_str = communalities.round(3).to_string()
    context_block = f"Contexte métier : {context}\n" if context.strip() else ""

    return f"""Tu es un expert en statistiques appliquées et en analyse de données métier.
Réponds en français uniquement en texte courant, comme un rapport rédigé.
Aucun titre, aucun sous-titre, aucune liste à puces, aucun emoji, aucune mise en gras.
Rédige des paragraphes fluides et concis, séparés par des sauts de ligne.
Maximum 300 mots au total.

{context_block}
Indice KMO : {kmo_score:.3f} ({'Bonne adéquation' if kmo_score >= 0.7 else 'Adéquation acceptable' if kmo_score >= 0.6 else 'Adéquation faible'})
Test de Bartlett (p-value) : {bartlett_p:.4f} ({'Corrélations significatives' if bartlett_p < 0.05 else 'Corrélations non significatives'})

Matrice des saturations (rotation Varimax) :
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


def call_claude(
    prompt: str,
    api_key: str,
    model: str = CLAUDE_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> str:

    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Le package 'anthropic' n'est pas installé.") from e

    if not api_key or api_key.strip() == "":
        raise ValueError("Clé API Anthropic manquante.")

    client = anthropic.Anthropic(api_key=api_key)
    logger.info(f"[LLM] Appel Claude ({model}) — {len(prompt)} caractères.")

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text: str = message.content[0].text
    logger.info(f"[LLM] Réponse reçue ({len(response_text)} caractères).")
    return response_text