"""
Step 6 (imputacao baseada em dominio) + Step 7 (otimizacao de memoria).
Alinhado a Lecture 6 / notebook ANCINE.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Bootstrap: roda a partir da raiz do projeto independente do cwd
os.chdir(Path(__file__).resolve().parents[2])

df = pd.read_csv("dataset_limpo/cleaned_dataset_final.csv")

print("=" * 60)
print("Diagnostico inicial de NaN antes da imputacao")
print("=" * 60)
null_counts = df.isna().sum()
print(null_counts[null_counts > 0].to_string())

# =========================================================
# Step 6: Imputacao baseada em dominio
# - intensity_clean: mapa (sentimento, emocao) com fallback por sentimento
# - emotion_group "other": remapeada via dicionario auxiliar
# - plausibility: 0 se sem teoria, mediana (7) se com teoria sem score
# Cada decisao e justificada pela semantica da variavel.
# =========================================================
print("\n" + "=" * 60)
print("Step 6: Imputacao baseada em dominio")
print("=" * 60)

intensity_map = {
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

df["emotion_group_v2"]   = df.apply(remap_emotion, axis=1)
df["emotion_encoded_v2"] = df["emotion_group_v2"].map(EMOTION_RANK)

before_other = (df["emotion_group"]    == "other").sum()
after_other  = (df["emotion_group_v2"] == "other").sum()

PLAUS_MEDIAN = 7.0

def impute_plausibility(row, col):
    if not pd.isna(row[col]):
        return row[col]
    if row["num_theories"] == 0:
        return 0.0
    return PLAUS_MEDIAN

df["avg_plausibility_imputed"] = df.apply(lambda r: impute_plausibility(r, "avg_theory_plausibility"), axis=1)
df["max_plausibility_imputed"] = df.apply(lambda r: impute_plausibility(r, "max_theory_plausibility"), axis=1)
df["plausibility_was_imputed"] = (
    df["avg_theory_plausibility"].isna() & (df["num_theories"] > 0)
).astype(int)

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
n_outliers = int(
    ((df["engagement_score_v2"] < Q1 - 1.5 * IQR) |
     (df["engagement_score_v2"] > Q3 + 1.5 * IQR)).sum()
)

polarity_map = {1: "positive", 0: "mixed", -1: "negative"}
df["sentiment_polarity"] = df["sentiment_encoded"].map(polarity_map)
df["is_positive"] = (df["sentiment_clean"] == "positive").astype(int)
df["is_negative"] = (df["sentiment_clean"] == "negative").astype(int)

melhorias = [
    ("intensity_imputed",            int(df["intensity_imputed"].isna().sum()),    "0 NaN (imputado por sentimento+emocao)"),
    ("emotion_group_v2",             int((df["emotion_group_v2"] == "other").sum()), f"'other' reduzido: {before_other} -> {after_other}"),
    ("avg_plausibility_imputed",     int(df["avg_plausibility_imputed"].isna().sum()), "0 NaN (0 se sem teoria, mediana se com)"),
    ("engagement_score_v2",          0,                                            "corrigido com intensity_imputed"),
    ("engagement_score_winsorized",  0,                                            f"outliers limitados [{p05:.2f}-{p95:.2f}]"),
    ("is_positive / is_negative",    0,                                            "binarias para analises simples"),
]
for col, nans, nota in melhorias:
    print(f"  {col:<35} NaN={nans}  |  {nota}")

print(f"\n  Outliers (IQR 1.5x) no engagement_score_v2: {n_outliers}")

# =========================================================
# Step 7: Otimizacao de memoria
# - category: hash-map para strings de baixa cardinalidade
# - downcast: inteiros para o menor tipo possivel
# Reduz RAM em ate ~90% conforme demonstrado no notebook ANCINE.
# =========================================================
print("\n" + "=" * 60)
print("Step 7: Otimizacao de memoria (category + downcast)")
print("=" * 60)

print("Antes:")
print(f"  Memoria deep: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

categorical_cols = [
    "sentiment_clean", "sentiment_polarity",
    "emotion_group", "emotion_group_v2", "key_emotion_raw",
    "_source_file",
]
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

integer_cols = [
    "discussion_id", "num_theories", "num_themes", "num_canonical_themes",
    "intensity_was_imputed", "plausibility_was_imputed",
    "is_positive", "is_negative",
    "sentiment_encoded", "emotion_encoded", "emotion_encoded_v2",
] + [c for c in df.columns if c.startswith("theme_")]

for col in integer_cols:
    if col in df.columns and df[col].notna().all():
        df[col] = pd.to_numeric(df[col], downcast="integer")

float_cols = [
    "intensity_clean", "intensity_imputed",
    "avg_theory_plausibility", "max_theory_plausibility",
    "avg_plausibility_imputed", "max_plausibility_imputed",
    "engagement_score", "engagement_score_v2", "engagement_score_winsorized",
]
for col in float_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

print("\nDepois:")
print(f"  Memoria deep: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\nTipos apos otimizacao:")
print(df.dtypes.value_counts().to_string())

# =========================================================
# Persistencia
# =========================================================
out_csv = "dataset_limpo/cleaned_dataset_improved.csv"
df.to_csv(out_csv, index=False)
print(f"\nDataset melhorado salvo em: {out_csv}")
