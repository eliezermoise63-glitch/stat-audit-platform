"""
utils/validate_structure.py
----------------------------
Script de validation rapide de l'installation.
Lance avec : python utils/validate_structure.py

Vérifie :
  - Imports des dépendances clés
  - Fonctionnement de base du Sanitizer
  - Fonctionnement de base du Moteur
"""

import sys


def check_imports():
    print("🔍 Vérification des imports...")
    required = [
        ("pandas", "pd"),
        ("numpy", "np"),
        ("sklearn.decomposition", "PCA"),
        ("factor_analyzer", "FactorAnalyzer"),
        ("scipy.stats", "zscore"),
        ("streamlit", "st"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
    ]
    all_ok = True
    for module, alias in required:
        try:
            exec(f"import {module}")
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module} — {e}")
            all_ok = False

    # Test Anthropic séparément (optionnel)
    try:
        import anthropic
        print(f"  ✅ anthropic")
    except ImportError:
        print("  ⚠️  anthropic — non installé (onglet IA désactivé)")

    return all_ok


def check_sanitizer():
    print("\n🛡️ Vérification du Sanitizer...")
    try:
        import numpy as np
        import pandas as pd
        from core.sanitizer import DataSanitizer

        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.randn(50),
            "y": np.random.randn(50),
            "z": np.random.randn(50),
            "const": 1.0,
        })
        df.loc[0:5, "x"] = np.nan  # NaN
        df.loc[25, "y"] = 999.0    # outlier

        san = DataSanitizer()
        df_clean, report = san.fit_transform(df)

        assert "const" not in df_clean.columns, "Colonne constante non supprimée"
        assert df_clean.isnull().sum().sum() == 0, "NaN non imputés"
        assert len(df_clean) < len(df), "Outlier non supprimé"

        print(f"  ✅ Sanitizer OK — {report.n_rows_output} lignes sorties")
        return True
    except Exception as e:
        print(f"  ❌ Sanitizer ÉCHEC — {e}")
        return False


def check_engine():
    print("\n🔬 Vérification du Moteur...")
    try:
        import numpy as np
        import pandas as pd
        from core.engine import MultivariateEngine

        np.random.seed(1)
        f = np.random.randn(100)
        df = pd.DataFrame({
            "a": f + 0.1 * np.random.randn(100),
            "b": f + 0.2 * np.random.randn(100),
            "c": np.random.randn(100),
            "d": np.random.randn(100),
        })

        engine = MultivariateEngine(df)

        pca = engine.run_pca()
        assert pca.n_components >= 1
        print(f"  ✅ ACP OK — {pca.n_components} composantes")

        fa = engine.run_factor_analysis()
        assert fa.n_factors >= 1
        print(f"  ✅ AF OK — {fa.n_factors} facteurs (KMO={fa.kmo_score:.3f})")

        return True
    except Exception as e:
        print(f"  ❌ Moteur ÉCHEC — {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("  Statistical Audit Platform — Validation")
    print("=" * 50)

    ok_imports = check_imports()
    ok_sanitizer = check_sanitizer()
    ok_engine = check_engine()

    print("\n" + "=" * 50)
    if ok_imports and ok_sanitizer and ok_engine:
        print("✅ TOUT EST OK — Lancez : streamlit run app.py")
        sys.exit(0)
    else:
        print("❌ Des erreurs ont été détectées. Corrigez-les avant de lancer l'app.")
        sys.exit(1)
