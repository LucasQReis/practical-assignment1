# =========================================================
# data.py - Selecao de colunas finais
# ---------------------------------------------------------
# Recebe o cleaned_dataset_enriched.csv (187 x 110) gerado por
# main.py e mantem apenas as 31 colunas relevantes para as
# analises seguintes, descartando colunas auxiliares e dados
# brutos aninhados que ja foram extraidos como features.
# Saida: dataset_limpo/cleaned_dataset_final.csv (187 x 31).
# =========================================================

import os
from pathlib import Path
import pandas as pd

# Bootstrap: cwd para a raiz de assignment2/.
os.chdir(Path(__file__).resolve().parents[2])

df = pd.read_csv("dataset_limpo/cleaned_dataset_enriched.csv")

# Lista de colunas que sao mantidas: identificadores, texto bruto
# minimo (para inspecao), features de sentimento/teorias/temas e
# atributos temporais derivados.
keep = [
    "discussion_id",
    "title",
    "url",
    "summary",
    "published",
    "_source_file",
    "sentiment_clean",
    "sentiment_encoded",
    "intensity_clean",
    "key_emotion_raw",
    "emotion_group",
    "emotion_encoded",
    "num_theories",
    "avg_theory_plausibility",
    "max_theory_plausibility",
    "num_themes",
    "num_canonical_themes",
    "theme_character_development",
    "theme_nostalgia",
    "theme_plot_elements",
    "theme_relationships",
    "theme_theories",
    "theme_cinematography",
    "theme_music",
    "engagement_score",
    "summary_length",
    "published_hour",
    "published_dayofweek",
    "published_month",
    "days_since_first_post",
    "crawl_lag_hours",
]

df_final = df[keep]
removed = len(df.columns) - len(keep)

df_final.to_csv("dataset_limpo/cleaned_dataset_final.csv", index=False)
