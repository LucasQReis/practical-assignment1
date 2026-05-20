import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

os.chdir(Path(__file__).resolve().parents[2])

pd.options.display.float_format = "{:,.3f}".format

df = pd.read_csv("dataset_limpo/cleaned_dataset_improved.csv")
out_dir = Path("dataset_limpo")
img_dir = Path("plot_image")
img_dir.mkdir(exist_ok=True)
fe_log = []

def log_action(category, action, target, motivation):
    fe_log.append({
        "category": category, "action": action,
        "target": target, "motivation": motivation,
    })

df["log_num_theories"] = np.log1p(df["num_theories"])
sk_before = df["num_theories"].skew()
sk_after  = df["log_num_theories"].skew()
log_action("transformation", "log1p", "num_theories",
           f"skew {sk_before:+.2f} (leptokurtic) -> {sk_after:+.2f}")
num_cols_scale = ["intensity_imputed", "summary_length",
                  "engagement_score_v2", "log_num_theories"]
scaler_std = StandardScaler()
scaled_std = scaler_std.fit_transform(df[num_cols_scale])
for i, col in enumerate(num_cols_scale):
    df[f"{col}_std"] = scaled_std[:, i]
log_action("transformation", "StandardScaler", ", ".join(num_cols_scale),
           "escalas diferentes (intensity 0-10, summary_length 0-101); "
           "necessario para distance-based (kNN, k-Means, SVM, PCA)")

num_cols_mm = ["intensity_imputed", "engagement_score_v2"]
scaler_mm = MinMaxScaler()
scaled_mm = scaler_mm.fit_transform(df[num_cols_mm])
for i, col in enumerate(num_cols_mm):
    df[f"{col}_mm"] = scaled_mm[:, i]
log_action("transformation", "MinMaxScaler [0,1]", ", ".join(num_cols_mm),
           "redes neurais e algoritmos que esperam input em [0,1]")

p05, p95 = df["summary_length"].quantile([0.05, 0.95])
df["summary_length_wins"] = df["summary_length"].clip(p05, p95)
log_action("transformation", "winsorize p05-p95", "summary_length",
           f"limita outliers; range [{p05:.0f}, {p95:.0f}]")

df["theory_density"] = df["num_theories"] / df["summary_length"].clip(lower=1)
log_action("creation", "ratio", "theory_density",
           "teorias por palavra do resumo - normaliza por tamanho do post")

df["theme_diversity"] = df["num_canonical_themes"] / 7
log_action("creation", "ratio", "theme_diversity",
           "proporcao dos 7 temas canonicos presentes no post")

df["is_weekend"] = (df["published_dayofweek"] >= 5).astype(int)
log_action("creation", "binary indicator", "is_weekend",
           "captura efeito de fim de semana sobre engajamento")

df["hour_sin"] = np.sin(2 * np.pi * df["published_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["published_hour"] / 24)
log_action("creation", "cyclical encoding (sin/cos)", "published_hour",
           "preserva continuidade circular - hora 23 fica adjacente a hora 0")

df["dow_sin"] = np.sin(2 * np.pi * df["published_dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["published_dayofweek"] / 7)
log_action("creation", "cyclical encoding (sin/cos)", "published_dayofweek",
           "preserva continuidade semanal")

df["intensity_x_themes"] = df["intensity_imputed"] * df["num_canonical_themes"]
log_action("creation", "interaction term", "intensity_x_themes",
           "efeito combinado de intensidade emocional e riqueza tematica")

RARE_THRESHOLD = 5
emotion_counts = df["emotion_group_v2"].value_counts()
rare_emotions = emotion_counts[emotion_counts < RARE_THRESHOLD].index.tolist()
df["emotion_grouped"] = df["emotion_group_v2"].apply(
    lambda x: "rare" if x in rare_emotions else x
)
log_action("encoding", "group rare categories", "emotion_group_v2",
           f"categorias com n<{RARE_THRESHOLD} ({rare_emotions}) "
           "agrupadas para evitar colunas esparsas no one-hot")

ohe_cols = ["sentiment_clean", "emotion_grouped", "_source_file"]
df_ohe = pd.get_dummies(df, columns=ohe_cols, prefix=ohe_cols, drop_first=False)
new_ohe = [c for c in df_ohe.columns if c not in df.columns]
log_action("encoding", "one-hot", ", ".join(ohe_cols),
           f"variaveis nominais sem ordem natural -> {len(new_ohe)} colunas binarias. "
           "Ordinal encoding criaria ordem espuria entre categorias.")
df = df_ohe

log_action("encoding", "ordinal (pre-existing)", "emotion_encoded_v2",
           "ranking 0-8 para uso opcional em modelos baseados em arvore")

num_features = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = {"discussion_id", "source_file_id"}
num_features = [c for c in num_features if c not in exclude]
corr = df[num_features].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr_pairs = [
    (col, row, upper.loc[row, col])
    for col in upper.columns
    for row in upper.index
    if upper.loc[row, col] >= 0.9
]
high_corr_pairs.sort(key=lambda x: -x[2])
to_drop = set()
keep_priority = {
    "intensity_imputed": 1, "intensity_clean": 2,
    "engagement_score_v2": 1, "engagement_score": 2, "engagement_score_winsorized": 3,
    "num_canonical_themes": 1, "num_themes": 2,
    "avg_plausibility_imputed": 1, "avg_theory_plausibility": 2,
    "max_plausibility_imputed": 1, "max_theory_plausibility": 2,
    "log_num_theories": 1, "num_theories": 2,
}

for a, b, r in high_corr_pairs[:15]:
    pa = keep_priority.get(a, 5)
    pb = keep_priority.get(b, 5)
    drop = a if pa > pb else b
    if drop in to_drop or (a in to_drop or b in to_drop):
        continue
    print(f"    {a}  <->  {b}   |r|={r:.3f}   -> drop '{drop}'")
    to_drop.add(drop)

for col in to_drop:
    log_action("selection", "drop redundant", col,
               "multicolinearidade |r| >= 0.9 com feature mais informativa")

LOW_VAR_THRESHOLD = 0.01
binary_cols = [c for c in df.columns
               if df[c].dropna().nunique() == 2 and df[c].dtype in [int, float, bool, "int64", "float64", "int8", "int16", "int32"]]
for col in binary_cols:
    p = df[col].mean()
    if p < LOW_VAR_THRESHOLD or p > 1 - LOW_VAR_THRESHOLD:
        to_drop.add(col)
        log_action("selection", "drop low variance", col,
                   f"quase-constante (proporcao={p:.3f}, abaixo de {LOW_VAR_THRESHOLD})")

text_id_cols = ["title", "url", "summary", "key_emotion_raw", "published",
                "emotion_group", "emotion_group_v2", "sentiment_polarity"]
for col in text_id_cols:
    if col in df.columns:
        to_drop.add(col)

df_ml = df.drop(columns=list(to_drop & set(df.columns)))
theme_cols = [c for c in df.columns
              if c.startswith("theme_") and df[c].dropna().nunique() == 2]

pca = PCA(n_components=min(5, len(theme_cols)))
theme_pca = pca.fit_transform(df[theme_cols])
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

for i, (ev, cv) in enumerate(zip(explained, cumulative), 1):
    print(f"    PC{i}: explica {ev*100:.1f}% (acumulado {cv*100:.1f}%)")

for i in range(3):
    df_ml[f"theme_pc{i+1}"] = theme_pca[:, i]
log_action("dimensionality", "PCA(3) em 7 temas binarios", "theme_*",
           f"3 PCs explicam {cumulative[2]*100:.1f}% da variancia "
           "- reduz 7 -> 3 dimensoes mantendo info")

out_csv = out_dir / "cleaned_dataset_ml_ready.csv"
df_ml.to_csv(out_csv, index=False)

log_csv = out_dir / "feature_engineering_log.csv"
pd.DataFrame(fe_log).to_csv(log_csv, index=False, encoding="utf-8")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

axes[0, 0].hist(df["num_theories"], bins=10, color="#E24B4A", alpha=0.75, edgecolor="white")
axes[0, 0].set_title(f"num_theories (skew={df['num_theories'].skew():+.2f})\nAntes",
                     fontsize=10, fontweight="bold")
axes[0, 0].set_xlabel("Valor"); axes[0, 0].set_ylabel("Frequencia")
axes[0, 0].grid(axis="y", color="#e8e8e8", linewidth=0.7); axes[0, 0].set_axisbelow(True)

axes[0, 1].hist(df["log_num_theories"], bins=10, color="#149945", alpha=0.75, edgecolor="white")
axes[0, 1].set_title(f"log1p(num_theories) (skew={df['log_num_theories'].skew():+.2f})\nDepois",
                     fontsize=10, fontweight="bold")
axes[0, 1].set_xlabel("Valor"); axes[0, 1].set_ylabel("Frequencia")
axes[0, 1].grid(axis="y", color="#e8e8e8", linewidth=0.7); axes[0, 1].set_axisbelow(True)

hours = np.arange(24)
axes[0, 2].plot(hours, np.sin(2*np.pi*hours/24), "o-", label="sin", color="#378ADD")
axes[0, 2].plot(hours, np.cos(2*np.pi*hours/24), "s-", label="cos", color="#BA7517")
axes[0, 2].set_title("Cyclical encoding: hora (0-23)", fontsize=10, fontweight="bold")
axes[0, 2].set_xlabel("hora"); axes[0, 2].set_ylabel("valor codificado")
axes[0, 2].legend(); axes[0, 2].grid(color="#e8e8e8", linewidth=0.7); axes[0, 2].set_axisbelow(True)

axes[1, 0].hist(df["intensity_imputed"], bins=10, color="#7F77DD", alpha=0.75, edgecolor="white")
axes[1, 0].set_title(f"intensity_imputed (raw)\nmean={df['intensity_imputed'].mean():.2f}, "
                     f"std={df['intensity_imputed'].std():.2f}",
                     fontsize=10, fontweight="bold")
axes[1, 0].grid(axis="y", color="#e8e8e8", linewidth=0.7); axes[1, 0].set_axisbelow(True)

axes[1, 1].hist(df["intensity_imputed_std"], bins=10, color="#7F77DD", alpha=0.75, edgecolor="white")
axes[1, 1].set_title(f"intensity_imputed_std\nmean={df['intensity_imputed_std'].mean():.2f}, "
                     f"std={df['intensity_imputed_std'].std():.2f}",
                     fontsize=10, fontweight="bold")
axes[1, 1].grid(axis="y", color="#e8e8e8", linewidth=0.7); axes[1, 1].set_axisbelow(True)
axes[1, 2].bar(range(1, len(explained)+1), explained*100, alpha=0.75,
               color="#378ADD", edgecolor="white", label="por componente")
axes[1, 2].plot(range(1, len(cumulative)+1), cumulative*100, "o-",
                color="#E24B4A", linewidth=2, label="acumulado")
axes[1, 2].axhline(80, color="#888", linestyle="--", linewidth=1, alpha=0.6)
axes[1, 2].set_title("PCA dos 7 temas - variancia explicada", fontsize=10, fontweight="bold")
axes[1, 2].set_xlabel("Componente Principal"); axes[1, 2].set_ylabel("% Variancia")
axes[1, 2].legend(); axes[1, 2].grid(axis="y", color="#e8e8e8", linewidth=0.7)
axes[1, 2].set_axisbelow(True)

fig.suptitle("Feature Engineering - Antes vs Depois das Transformacoes",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
img_path = img_dir / "feature_transformations.png"
plt.savefig(img_path, dpi=150, bbox_inches="tight")
plt.show()

summary = pd.DataFrame(fe_log).groupby("category").size()
for cat, n in summary.items():
    print(f"  {cat:.<25} {n} operacoes")
