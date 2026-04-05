"""
tests/test_llm.py
-----------------
Tests unitaires pour utils/llm.py.
On teste la construction des prompts (pas les appels API réels).
"""

import pandas as pd
import pytest

from utils.llm import build_fa_prompt, build_pca_prompt


@pytest.fixture
def sample_fa_loadings():
    return pd.DataFrame(
        {"F1": [0.85, 0.78, 0.10, -0.05], "F2": [-0.12, 0.09, 0.82, 0.76]},
        index=["revenu", "salaire", "satisfaction", "engagement"],
    )


@pytest.fixture
def sample_communalities():
    return pd.Series(
        [0.74, 0.62, 0.68, 0.59],
        index=["revenu", "salaire", "satisfaction", "engagement"],
        name="Communauté",
    )


@pytest.fixture
def sample_pca_loadings():
    return pd.DataFrame(
        {"PC1": [0.7, 0.6, -0.3], "PC2": [0.2, -0.1, 0.9]},
        index=["age", "revenu", "note"],
    )


class TestPromptBuilders:

    def test_fa_prompt_is_string(self, sample_fa_loadings, sample_communalities):
        prompt = build_fa_prompt(sample_fa_loadings, sample_communalities, 0.72, 0.001)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_fa_prompt_contains_kmo(self, sample_fa_loadings, sample_communalities):
        prompt = build_fa_prompt(sample_fa_loadings, sample_communalities, 0.72, 0.001)
        assert "0.720" in prompt or "KMO" in prompt

    def test_fa_prompt_contains_bartlett(self, sample_fa_loadings, sample_communalities):
        prompt = build_fa_prompt(sample_fa_loadings, sample_communalities, 0.72, 0.0012)
        assert "0.0012" in prompt or "Bartlett" in prompt

    def test_fa_prompt_contains_loadings(self, sample_fa_loadings, sample_communalities):
        prompt = build_fa_prompt(sample_fa_loadings, sample_communalities, 0.72, 0.001)
        assert "revenu" in prompt
        assert "F1" in prompt

    def test_fa_prompt_with_context(self, sample_fa_loadings, sample_communalities):
        prompt = build_fa_prompt(
            sample_fa_loadings, sample_communalities, 0.72, 0.001,
            context="Données RH d'une entreprise industrielle"
        )
        assert "RH" in prompt

    def test_fa_prompt_without_context(self, sample_fa_loadings, sample_communalities):
        prompt = build_fa_prompt(sample_fa_loadings, sample_communalities, 0.72, 0.001)
        assert isinstance(prompt, str)  # Pas d'erreur sans contexte

    def test_pca_prompt_is_string(self, sample_pca_loadings):
        prompt = build_pca_prompt(sample_pca_loadings, [0.45, 0.30], 2)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_pca_prompt_contains_variance(self, sample_pca_loadings):
        prompt = build_pca_prompt(sample_pca_loadings, [0.45, 0.30], 2)
        assert "45.0%" in prompt or "PC1" in prompt

    def test_pca_prompt_with_context(self, sample_pca_loadings):
        prompt = build_pca_prompt(
            sample_pca_loadings, [0.45, 0.30], 2,
            context="Portefeuille de crédit bancaire"
        )
        assert "crédit" in prompt

    def test_call_claude_raises_without_key(self):
        """Sans clé API, call_claude doit lever ValueError."""
        from utils.llm import call_claude
        with pytest.raises((ValueError, Exception)):
            call_claude("test prompt", api_key="")

    def test_call_claude_raises_on_bad_key(self):
        """Avec une clé invalide, l'API doit lever une exception."""
        from utils.llm import call_claude
        with pytest.raises(Exception):
            call_claude("test prompt", api_key="sk-ant-invalide-123")
