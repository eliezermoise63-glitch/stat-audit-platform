"""
assets/demo_data_generator.py
------------------------------
Génère un dataset de démonstration réaliste pour tester la plateforme.

Usage :
    python assets/demo_data_generator.py
    → Crée : assets/demo_hr_dataset.csv (200 lignes, 8 variables RH)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 200

# ── 2 facteurs latents sous-jacents ──────────────────────────────────────────
# F1 = "Performance individuelle"
# F2 = "Bien-être au travail"

f_perf = np.random.randn(n)
f_wellbeing = np.random.randn(n)

df = pd.DataFrame({
    # Variables liées à la performance (F1)
    "note_annuelle":        np.clip(f_perf * 1.2 + 3.0 + np.random.randn(n) * 0.3, 0, 5).round(2),
    "productivite_score":   np.clip(f_perf * 15 + 70 + np.random.randn(n) * 5, 0, 100).round(1),
    "nb_projets_completes": np.clip((f_perf * 2 + 5 + np.random.randn(n)).round(0), 0, 12).astype(int),

    # Variables liées au bien-être (F2)
    "satisfaction_job":     np.clip(f_wellbeing * 1.5 + 3.5 + np.random.randn(n) * 0.5, 1, 5).round(1),
    "nb_conges_pris":       np.clip((f_wellbeing * 3 + 20 + np.random.randn(n) * 2).round(0), 0, 30).astype(int),
    "score_engagement":     np.clip(f_wellbeing * 12 + 65 + np.random.randn(n) * 8, 0, 100).round(1),

    # Variables mixtes
    "anciennete_annees":    np.clip((np.random.exponential(5, n)).round(1), 0.5, 30),
    "salaire_k_eur":        np.clip(35 + f_perf * 8 + f_wellbeing * 2 + np.random.randn(n) * 3, 25, 80).round(1),
})

# Introduire quelques NaN (~5%)
for col in ["note_annuelle", "satisfaction_job", "nb_conges_pris"]:
    mask = np.random.rand(n) < 0.05
    df.loc[mask, col] = np.nan

# Introduire 2-3 outliers
df.loc[10, "productivite_score"] = 200.0   # outlier haut
df.loc[50, "salaire_k_eur"] = 5.0          # outlier bas
df.loc[99, "nb_projets_completes"] = 50    # outlier haut

output_path = os.path.join(os.path.dirname(__file__), "demo_hr_dataset.csv")
df.to_csv(output_path, index=False)

print(f"✅ Dataset généré : {output_path}")
print(f"   {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   NaN introduits : {df.isnull().sum().sum()}")
print(f"   Outliers introduits : 3")
print("\nAperçu :")
print(df.head())
