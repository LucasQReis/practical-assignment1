"""
Stranger Things S5 Finale - Pipeline de Preparacao de Dados
Estruturado conforme Lecture 6 (EDA): Steps 1 a 7.
"""

import os
import json
import re
import pandas as pd
from pathlib import Path

# Bootstrap: roda a partir da raiz do projeto independente do cwd
os.chdir(Path(__file__).resolve().parents[2])

# =========================================================
# Step 1: Consolidacao dos arquivos de origem (JSON)
# Analogo a 'CSV File Consolidation' do notebook ANCINE.
# Usa generator para nao carregar todos os arquivos em RAM.
# =========================================================
print("=" * 60)
print("Step 1: Consolidacao das fontes JSON")
print("=" * 60)

dataset_dir = Path("dataset")
file_labels = {
    "finale_day1.json":  "day1",
    "finale_day3.json":  "day3",
    "finale_week1.json": "week1",
}

def load_discussions(path: Path, label: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    discussions = data.get("discussions", [])
    for d in discussions:
        d["_source_file"] = label
    return discussions

all_discussions = []
for fname, label in file_labels.items():
    all_discussions.extend(load_discussions(dataset_dir / fname, label))

df = pd.json_normalize(all_discussions)
print(f"Dimensoes consolidadas: {df.shape}")

# =========================================================
# Step 2: Export em Parquet (preservacao de schema + I/O eficiente)
# Snappy compression conforme recomendado no notebook ANCINE.
# =========================================================
print("\n" + "=" * 60)
print("Step 2: Export Parquet (schema-preserving)")
print("=" * 60)

out_dir = Path("dataset_limpo")
out_dir.mkdir(exist_ok=True)

try:
    parquet_path = out_dir / "discussions_raw.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Parquet salvo em: {parquet_path}")
except Exception as e:
    print(f"[aviso] export Parquet falhou ({e}); seguindo apenas com CSV.")

# =========================================================
# Step 3: Verificacao de duplicatas
# Reflexao: duplicatas podem vir de janelas temporais sobrepostas (day1/day3/week1).
# =========================================================
print("\n" + "=" * 60)
print("Step 3: Verificacao de duplicatas")
print("=" * 60)

duplicate_mask = df.duplicated(subset=["url"])
total_dup = int(duplicate_mask.sum())
pct_dup = (total_dup / len(df)) * 100
print(f"Duplicatas por URL: {total_dup}  ({pct_dup:.2f}% do total)")

if total_dup > 0:
    df_dups = df[df.duplicated(subset=["url"], keep=False)].sort_values("url")
    df_dups.to_csv(out_dir / "duplicates_dataset.csv", index=False)
    print(f"Amostra de duplicatas salva em: {out_dir / 'duplicates_dataset.csv'}")

df_clean = df.drop_duplicates(subset=["url"], keep="first").copy()
print(f"Apos deduplicacao: {df_clean.shape}")

# =========================================================
# Step 4: Type casting / Parsing temporal
# Datas em ISO 8601 -> datetime64[ns]; coerce para NaT em falha.
# =========================================================
print("\n" + "=" * 60)
print("Step 4: Parsing temporal (object -> datetime64[ns])")
print("=" * 60)

df_clean["published_dt"] = pd.to_datetime(
    df_clean["published"], utc=True, errors="coerce"
)
df_clean["timestamp_dt"] = pd.to_datetime(
    df_clean["timestamp"], utc=True, errors="coerce"
)

print("Tipos apos parsing:")
print(df_clean[["published_dt", "timestamp_dt"]].dtypes)

# Features temporais derivadas
df_clean["published_hour"]      = df_clean["published_dt"].dt.hour
df_clean["published_dayofweek"] = df_clean["published_dt"].dt.dayofweek
df_clean["published_month"]     = df_clean["published_dt"].dt.month

min_date = df_clean["published_dt"].min()
df_clean["days_since_first_post"] = (df_clean["published_dt"] - min_date).dt.days
df_clean["crawl_lag_hours"] = (
    (df_clean["timestamp_dt"] - df_clean["published_dt"])
    .dt.total_seconds().div(3600).round(1)
)

# =========================================================
# Feature engineering - extracao de campos aninhados
# (sentimento, intensidade, emocao, teorias, temas)
# =========================================================
print("\n" + "=" * 60)
print("Feature engineering: extracao de campos aninhados")
print("=" * 60)

def extract_sentiment(row):
    sentiment = row.get("analysis.sentiment.sentiment")
    if isinstance(sentiment, str):
        return sentiment
    raw = row.get("analysis.sentiment.raw")
    if isinstance(raw, str):
        for label in ["positive", "negative", "mixed"]:
            if label in raw:
                return label
    return "unknown"

def extract_intensity(row):
    intensity = row.get("analysis.sentiment.intensity")
    if isinstance(intensity, (int, float)):
        return intensity
    raw = row.get("analysis.sentiment.raw")
    if isinstance(raw, str):
        m = re.search(r'"intensity":\s*(\d+)', raw)
        if m:
            return int(m.group(1))
    return None

def count_theories(row):
    theories = row.get("analysis.theory")
    if isinstance(theories, list):
        return len([t for t in theories if isinstance(t, dict) and t.get("has_theory")])
    if isinstance(theories, dict):
        fan = theories.get("fan_theories", [])
        return len([t for t in fan if isinstance(t, dict) and t.get("has_theory")])
    return 0

df_clean["sentiment_clean"] = df_clean.apply(extract_sentiment, axis=1)
df_clean["intensity_clean"] = df_clean.apply(extract_intensity, axis=1)
df_clean["num_theories"]    = df_clean.apply(count_theories, axis=1)

df_clean = df_clean.reset_index(drop=True)
df_clean["discussion_id"] = df_clean.index + 1

source_map = {"day1": 1, "day3": 3, "week1": 7}
df_clean["source_file_id"] = df_clean["_source_file"].map(source_map)

sentiment_map = {"positive": 1, "mixed": 0, "negative": -1, "unknown": None}
df_clean["sentiment_encoded"] = df_clean["sentiment_clean"].map(sentiment_map)

EMOTION_GROUPS = {
    "excitement":     ["excitement", "excited", "enthusiasm", "enthusiastic", "hopeful"],
    "frustration":    ["frustration", "frustrated", "irritation", "annoyed"],
    "disappointment": ["disappointment", "disappointed", "letdown"],
    "nostalgia":      ["nostalgia", "nostalgic", "sentimental"],
    "curiosity":      ["curiosity", "curious", "interest", "intrigued"],
    "appreciation":   ["appreciation", "admiration", "awe", "impressed"],
    "humor":          ["amusement", "humor", "humorous", "amusing"],
    "skepticism":     ["skeptical", "skepticism", "doubtful"],
    "other":          [],
}
EMOTION_RANK = {k: i for i, k in enumerate(EMOTION_GROUPS.keys())}

def extract_emotion_raw(row):
    raw = row.get("analysis.sentiment.raw", "")
    if not isinstance(raw, str):
        return None
    m = re.search(r'"key_emotion":\s*"([^"]+)"', raw)
    return m.group(1).lower().strip() if m else None

def map_emotion_group(emotion_raw):
    if not emotion_raw or not isinstance(emotion_raw, str):
        return "other"
    for group, keywords in EMOTION_GROUPS.items():
        if group == "other":
            continue
        if any(kw in emotion_raw for kw in keywords):
            return group
    return "other"

df_clean["key_emotion_raw"] = df_clean.apply(extract_emotion_raw, axis=1)
df_clean["emotion_group"]   = df_clean["key_emotion_raw"].apply(map_emotion_group)
df_clean["emotion_encoded"] = df_clean["emotion_group"].map(EMOTION_RANK)

def _theory_scores(row):
    theories = row.get("analysis.theory")
    if isinstance(theories, list):
        items = theories
    elif isinstance(theories, dict):
        items = theories.get("fan_theories", [])
    else:
        return []
    return [t["plausibility"] for t in items
            if isinstance(t, dict) and t.get("has_theory") and "plausibility" in t]

def _avg_plaus(row):
    s = _theory_scores(row)
    return round(sum(s) / len(s), 2) if s else None

def _max_plaus(row):
    s = _theory_scores(row)
    return max(s) if s else None

df_clean["avg_theory_plausibility"] = df_clean.apply(_avg_plaus, axis=1)
df_clean["max_theory_plausibility"] = df_clean.apply(_max_plaus, axis=1)

df_clean["num_themes"] = df_clean["analysis.themes"].apply(
    lambda x: len(x) if isinstance(x, list) else 0
)

title_freq = df_clean["title"].value_counts()
df_clean["title_freq"] = df_clean["title"].map(title_freq)

CANONICAL_THEMES = {
    "character_development": ["character development", "character arc"],
    "nostalgia":              ["nostalgia", "nostalgic"],
    "plot_elements":          ["plot elements", "plot element", "plot twist", "plot twists"],
    "relationships":          ["relationships", "relationship dynamics"],
    "theories":               ["theories", "theory"],
    "cinematography":         ["cinematography", "atmospheric cinematography"],
    "music":                  ["music", "soundtrack"],
}

def has_theme(themes_list, keywords):
    if not isinstance(themes_list, list):
        return 0
    normalized = [t.lower().strip() for t in themes_list]
    return int(any(any(kw in t for kw in keywords) for t in normalized))

for theme_col, keywords in CANONICAL_THEMES.items():
    df_clean[f"theme_{theme_col}"] = df_clean["analysis.themes"].apply(
        lambda x, kw=keywords: has_theme(x, kw)
    )

df_clean["num_canonical_themes"] = df_clean[
    [f"theme_{k}" for k in CANONICAL_THEMES]
].sum(axis=1)

df_clean["summary_length"] = df_clean["summary"].apply(
    lambda x: len(str(x).split()) if isinstance(x, str) else 0
)
df_clean["title_length"] = df_clean["title"].apply(
    lambda x: len(str(x).split()) if isinstance(x, str) else 0
)

df_clean["engagement_score"] = (
    df_clean["intensity_clean"].fillna(0) * 0.5
    + df_clean["num_theories"].clip(upper=5) * 0.3
    + df_clean["num_themes"].clip(upper=5) * 0.2
).round(2)

# =========================================================
# Step 5: Diagnostico de esparsidade (mapeamento de NaN)
# Tabela com total e proporcao percentual, ordenada.
# =========================================================
print("\n" + "=" * 60)
print("Step 5: Diagnostico de esparsidade")
print("=" * 60)

null_counts = df_clean.isna().sum()
cols_with_nulls = null_counts[null_counts > 0]

if not cols_with_nulls.empty:
    null_diag = pd.DataFrame({
        "Total Nulls":      cols_with_nulls,
        "Proportion (%)":   (cols_with_nulls / len(df_clean) * 100).round(4),
    }).sort_values("Proportion (%)", ascending=False)
    print(null_diag.to_string())
else:
    print("Integridade maxima: sem NaN detectados.")

# =========================================================
# Step 6: Diagnostico estrutural (RAM footprint)
# info(memory_usage='deep') inspeciona o tamanho real de strings.
# =========================================================
print("\n" + "=" * 60)
print("Step 6: Diagnostico estrutural (info / memory deep)")
print("=" * 60)
df_clean.info(memory_usage="deep")

# =========================================================
# Persistencia
# Notas: a imputacao baseada em dominio (Step "domain-based imputation"
# do notebook ANCINE) ocorre em dataset_impv.py por modularidade.
# A otimizacao de memoria (Step 7) tambem fica em dataset_impv.py,
# apos as imputacoes, para downcast efetivo de inteiros que sairam de NaN.
# =========================================================
out_csv = "dataset_limpo/cleaned_dataset_enriched.csv"
df_clean.to_csv(out_csv, index=False)
print(f"\nDataset enriquecido salvo em: {out_csv}")
print(f"Dimensoes finais: {df_clean.shape}")
