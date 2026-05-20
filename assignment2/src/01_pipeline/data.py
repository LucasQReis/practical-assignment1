import os
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])

df = pd.read_csv("dataset_limpo/cleaned_dataset_enriched.csv")

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
