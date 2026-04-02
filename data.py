import pandas as pd

df = pd.read_csv("cleaned_dataset_enriched.csv")

print(f"Shape original: {df.shape}")

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
]

df_final = df[keep]

removed = len(df.columns) - len(keep)

df_final.to_csv("cleaned_dataset_final.csv", index=False)