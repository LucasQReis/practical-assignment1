import pandas as pd
import numpy as np

df = pd.read_csv("dataset_limpo/cleaned_dataset_final.csv")

intensity_map = {
    # (sentiment, emotion_group) → intensidade estimada
    ("positive", "excitement"):     8,
    ("positive", "appreciation"):   7,
    ("positive", "humor"):          6,
    ("positive", "nostalgia"):      6,
    ("positive", "curiosity"):      5,
    ("positive", "other"):          6,
    ("mixed",    "nostalgia"):      5,
    ("mixed",    "curiosity"):      5,
    ("mixed",    "frustration"):    6,
    ("mixed",    "disappointment"): 6,
    ("mixed",    "other"):          5,
    ("negative", "frustration"):    7,
    ("negative", "disappointment"): 7,
    ("negative", "skepticism"):     6,
    ("negative", "other"):          6,
}

def impute_intensity(row):
    if not pd.isna(row["intensity_clean"]):
        return row["intensity_clean"]
    key = (row["sentiment_clean"], row["emotion_group"])
    if key in intensity_map:
        return intensity_map[key]
    # fallback: média por sentimento
    fallback = {"positive": 7, "mixed": 5, "negative": 6}
    return fallback.get(row["sentiment_clean"], 5)

df["intensity_imputed"] = df.apply(impute_intensity, axis=1)
df["intensity_was_imputed"] = df["intensity_clean"].isna().astype(int)

EXTRA_EMOTION_MAP = {
    "anger":        "frustration",
    "outrage":      "frustration",
    "defensiveness":"frustration",
    "disapproval":  "frustration",
    "betrayal":     "disappointment",
    "sadness":      "disappointment",
    "hesitation":   "disappointment",
    "indifference": "disappointment",
    "suspicion":    "skepticism",
    "ecstatic":     "excitement",
    "optimism":     "excitement",
    "supportive":   "appreciation",
    "pride":        "appreciation",
    "relief":       "appreciation",
    "confusion":    "curiosity",
}

EMOTION_RANK = {
    "excitement": 0, "frustration": 1, "disappointment": 2,
    "nostalgia": 3, "curiosity": 4, "appreciation": 5,
    "humor": 6, "skepticism": 7, "other": 8,
}

def remap_emotion(row):
    if row["emotion_group"] != "other":
        return row["emotion_group"]
    raw = row["key_emotion_raw"]
    if isinstance(raw, str):
        return EXTRA_EMOTION_MAP.get(raw.lower().strip(), "other")
    return "other"

df["emotion_group_v2"] = df.apply(remap_emotion, axis=1)
df["emotion_encoded_v2"] = df["emotion_group_v2"].map(EMOTION_RANK)

before = (df["emotion_group"] == "other").sum()
after  = (df["emotion_group_v2"] == "other").sum()

PLAUS_MEDIAN = 7.0

def impute_plausibility(row, col):
    if not pd.isna(row[col]):
        return row[col]
    if row["num_theories"] == 0:
        return 0.0      # ausência de teoria → plausibilidade 0
    return PLAUS_MEDIAN  # teoria existe mas sem score → mediana

df["avg_plausibility_imputed"] = df.apply(lambda r: impute_plausibility(r, "avg_theory_plausibility"), axis=1)
df["max_plausibility_imputed"] = df.apply(lambda r: impute_plausibility(r, "max_theory_plausibility"), axis=1)
df["plausibility_was_imputed"] = (df["avg_theory_plausibility"].isna() & (df["num_theories"] > 0)).astype(int)

df["engagement_score_v2"] = (
    df["intensity_imputed"]          * 0.50 +
    df["num_theories"].clip(upper=5) * 0.30 +
    df["num_themes"].clip(upper=5)   * 0.20
).round(2)

p05 = df["engagement_score_v2"].quantile(0.05)
p95 = df["engagement_score_v2"].quantile(0.95)
df["engagement_score_winsorized"] = df["engagement_score_v2"].clip(lower=p05, upper=p95).round(2)

Q1  = df["engagement_score_v2"].quantile(0.25)
Q3  = df["engagement_score_v2"].quantile(0.75)
IQR = Q3 - Q1
n_outliers = ((df["engagement_score_v2"] < Q1 - 1.5*IQR) |
              (df["engagement_score_v2"] > Q3 + 1.5*IQR)).sum()

polarity_map = {1: "positive", 0: "mixed", -1: "negative"}
df["sentiment_polarity"] = df["sentiment_encoded"].map(polarity_map)
df["is_positive"] = (df["sentiment_clean"] == "positive").astype(int)
df["is_negative"] = (df["sentiment_clean"] == "negative").astype(int)

melhorias = [
    ("intensity_imputed",            df["intensity_imputed"].isna().sum(),    "0 NaN (imputado por sentimento+emoção)"),
    ("emotion_group_v2",             (df["emotion_group_v2"]=="other").sum(), f"'other' reduzido: {before}→{after}"),
    ("avg_plausibility_imputed",     df["avg_plausibility_imputed"].isna().sum(), "0 NaN (0 se sem teoria, mediana se com)"),
    ("engagement_score_v2",          0,                                       "corrigido com intensity_imputed"),
    ("engagement_score_winsorized",  0,                                       f"outliers limitados [{p05:.2f}–{p95:.2f}]"),
    ("is_positive / is_negative",    0,                                       "binárias para análises simples"),
]

for col, nans, nota in melhorias:
    print(f"  {col:<35} NaN={nans}  |  {nota}")

df.to_csv("dataset_limpo/cleaned_dataset_improved.csv", index=False)