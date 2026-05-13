"""
Step 1 do ciclo EDA: Descriptive Statistics.
Estende media/mediana com moda, std, skewness e kurtosis (classificada),
seguindo Lecture 6 (EDA).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Bootstrap: roda a partir da raiz do projeto independente do cwd
os.chdir(Path(__file__).resolve().parents[2])

pd.options.display.float_format = "{:,.4f}".format

df = pd.read_csv("dataset_limpo/cleaned_dataset_improved.csv")

cols_sentimento = [
    "intensity_imputed",
    "sentiment_encoded",
    "emotion_encoded_v2",
]
cols_teorias = [
    "num_theories",
    "avg_plausibility_imputed",
    "max_plausibility_imputed",
]
cols_temas = [
    "num_themes",
    "num_canonical_themes",
    "theme_character_development",
    "theme_nostalgia",
    "theme_plot_elements",
    "theme_relationships",
    "theme_theories",
    "theme_cinematography",
    "theme_music",
]
cols_engagement = [
    "engagement_score_v2",
    "engagement_score_winsorized",
    "summary_length",
]

grupos = {
    "Sentimento": cols_sentimento,
    "Teorias":    cols_teorias,
    "Temas":      cols_temas,
    "Engagement": cols_engagement,
}

# =========================================================
# describe() global formatado (Lecture 6: float_format ".2f")
# =========================================================
print("=" * 70)
print("describe() - resumo estatistico global")
print("=" * 70)
all_numeric = [c for grupo in grupos.values() for c in grupo]
print(df[all_numeric].describe().T.to_string())

# =========================================================
# Tendencia central (mean, median, mode) + dispersao (std)
# + forma (skewness, kurtosis) com classificacao.
# =========================================================
def classify_skew(s):
    if s >  0.5: return f"right-skewed ({s:+.2f})"
    if s < -0.5: return f"left-skewed ({s:+.2f})"
    return f"symmetric ({s:+.2f})"

def classify_kurt(k):
    # pandas .kurt() retorna excesso de kurtosis (Fisher): k_normal = 0
    if k >  0.5: return f"leptokurtic ({k:+.2f})"
    if k < -0.5: return f"platykurtic ({k:+.2f})"
    return f"mesokurtic ({k:+.2f})"

resultados = []

for grupo, colunas in grupos.items():
    for col in colunas:
        if col not in df.columns:
            continue
        serie = df[col].dropna()
        if serie.empty:
            continue
        mode_vals = serie.mode()
        moda = float(mode_vals.iloc[0]) if not mode_vals.empty else np.nan

        resultados.append({
            "grupo":      grupo,
            "coluna":     col,
            "n":          int(serie.count()),
            "media":      round(serie.mean(),   4),
            "mediana":    round(serie.median(), 4),
            "moda":       round(moda,           4),
            "std":        round(serie.std(),    4),
            "skewness":   classify_skew(serie.skew()),
            "kurtosis":   classify_kurt(serie.kurt()),
        })

stats_df = pd.DataFrame(resultados)

for grupo in stats_df["grupo"].unique():
    print("\n" + "=" * 70)
    print(f"  {grupo.upper()}")
    print("=" * 70)
    subset = stats_df[stats_df["grupo"] == grupo].drop(columns=["grupo"])
    print(subset.to_string(index=False))

stats_df.to_csv("dataset_limpo/stats_mean_median.csv", index=False)
print(f"\nTabela completa salva em: dataset_limpo/stats_mean_median.csv")

# =========================================================
# Binning - distribuicao de engagement em intervalos
# (Lecture 6: 'binning' para variavel continua)
# =========================================================
print("\n" + "=" * 70)
print("Binning - engagement_score_v2 em intervalos")
print("=" * 70)

bins = [0, 3, 4, 5, 6, 10]
labels = ["[0-3)", "[3-4)", "[4-5)", "[5-6)", "[6-10]"]
binned = pd.cut(df["engagement_score_v2"], bins=bins, labels=labels,
                include_lowest=True, right=False)

fi  = binned.value_counts().sort_index()
fri = (fi / len(df) * 100).round(2)
Fi  = fi.cumsum()
Fri = (Fi / len(df) * 100).round(2)

freq_table = pd.DataFrame({
    "fi (absoluta)":       fi,
    "fri (%) relativa":    fri,
    "Fi acumulada":        Fi,
    "Fri (%) acum.relativa": Fri,
})
print(freq_table.to_string())
