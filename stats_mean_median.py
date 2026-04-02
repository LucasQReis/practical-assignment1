import pandas as pd

df = pd.read_csv("cleaned_dataset_final.csv")

cols_sentimento = [
    "intensity_clean",
    "sentiment_encoded",
    "emotion_encoded",
]

cols_teorias = [
    "num_theories",
    "avg_theory_plausibility",
    "max_theory_plausibility",
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

grupos = {
    "Sentimento": cols_sentimento,
    "Teorias":    cols_teorias,
    "Temas":      cols_temas,
}

resultados = []

for grupo, colunas in grupos.items():
    for col in colunas:
        serie = df[col].dropna()
        resultados.append({
            "grupo":   grupo,
            "coluna":  col,
            "n":       int(serie.count()),
            "media":   round(serie.mean(),   4),
            "mediana": round(serie.median(), 4),
        })

stats_df = pd.DataFrame(resultados)

for grupo in stats_df["grupo"].unique():
    print(f"\n{'='*50}")
    print(f"  {grupo.upper()}")
    print(f"{'='*50}")
    subset = stats_df[stats_df["grupo"] == grupo][["coluna","n","media","mediana"]]
    print(subset.to_string(index=False))

stats_df.to_csv("stats_mean_median.csv", index=False)
print("\n✅ Exportado: stats_mean_median.csv")