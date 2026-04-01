import json
import pandas as pd

# ================================
# 1. CARREGAR ARQUIVOS
# ================================

files = ["finale_day1.json", "finale_day3.json", "finale_week1.json"]

all_discussions = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        discussions = data.get("discussions", [])
        all_discussions.extend(discussions)

df = pd.json_normalize(all_discussions)

# ================================
# 2. IDENTIFICAR DUPLICATAS
# ================================

# Marca duplicatas baseado na URL
df["is_duplicate"] = df.duplicated(subset=["url"], keep=False)

# DataFrame apenas com duplicatas
duplicates_df = df[df["is_duplicate"] == True].copy()

# Contagem de duplicatas
num_duplicates = duplicates_df.shape[0]

print(f"Total de registros duplicados: {num_duplicates}")

# ================================
# 3. REMOVER DUPLICATAS (mantendo primeira ocorrência)
# ================================

df_clean = df.drop_duplicates(subset=["url"], keep="first")

# ================================
# 4. TRATAMENTO DE SENTIMENTO
# ================================

def extract_sentiment(row):
    sentiment = row.get("analysis.sentiment.sentiment")
    
    if isinstance(sentiment, str):
        return sentiment
    
    raw = row.get("analysis.sentiment.raw")
    if isinstance(raw, str):
        if "positive" in raw:
            return "positive"
        elif "negative" in raw:
            return "negative"
        elif "mixed" in raw:
            return "mixed"
    
    return "unknown"

df_clean["sentiment_clean"] = df_clean.apply(extract_sentiment, axis=1)

# ================================
# 5. EXTRAIR INTENSIDADE
# ================================

def extract_intensity(row):
    intensity = row.get("analysis.sentiment.intensity")
    
    if isinstance(intensity, (int, float)):
        return intensity
    
    raw = row.get("analysis.sentiment.raw")
    if isinstance(raw, str):
        import re
        match = re.search(r'"intensity":\s*(\d+)', raw)
        if match:
            return int(match.group(1))
    
    return None

df_clean["intensity_clean"] = df_clean.apply(extract_intensity, axis=1)

# ================================
# 6. CONTAGEM DE TEORIAS
# ================================

def count_theories(row):
    theories = row.get("analysis.theory")
    
    if isinstance(theories, list):
        return len(theories)
    elif isinstance(theories, dict):
        return 1
    return 0

df_clean["num_theories"] = df_clean.apply(count_theories, axis=1)

# ================================
# 7. FEATURES NUMÉRICAS
# ================================

sentiment_counts = df_clean["sentiment_clean"].value_counts()

num_positive = sentiment_counts.get("positive", 0)
num_negative = sentiment_counts.get("negative", 0)
num_mixed = sentiment_counts.get("mixed", 0)
num_unknown = sentiment_counts.get("unknown", 0)

avg_intensity = df_clean["intensity_clean"].mean()

num_discussions = len(df_clean)
total_theories = df_clean["num_theories"].sum()

missing_sentiment_ratio = (df_clean["sentiment_clean"] == "unknown").mean()

# ================================
# 8. RESUMO FINAL
# ================================

summary = {
    "num_discussions": num_discussions,
    "num_duplicates": num_duplicates,
    "num_positive": num_positive,
    "num_negative": num_negative,
    "num_mixed": num_mixed,
    "num_unknown": num_unknown,
    "avg_intensity": avg_intensity,
    "total_theories": total_theories,
    "missing_sentiment_ratio": missing_sentiment_ratio
}

summary_df = pd.DataFrame([summary])

print(summary_df)

# ================================
# 9. EXPORTAR RESULTADOS
# ================================

df_clean.to_csv("cleaned_dataset.csv", index=False)
duplicates_df.to_csv("duplicates_dataset.csv", index=False)
summary_df.to_csv("summary_metrics.csv", index=False)