"""
Tabelas de frequencia (absoluta, relativa, acumulada) para variaveis
qualitativas - Lecture 6 (EDA), secao 'Frequency'.

Tambem aplica binning a variaveis continuas (engagement_score_v2,
intensity_imputed, summary_length) conforme slide 'Binning'.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Bootstrap: roda a partir da raiz do projeto independente do cwd
os.chdir(Path(__file__).resolve().parents[2])

pd.options.display.float_format = "{:,.2f}".format

df = pd.read_csv("dataset_limpo/cleaned_dataset_improved.csv")
out_dir = Path("dataset_limpo")
img_dir = Path("plot_image")
img_dir.mkdir(exist_ok=True)

BLUE   = "#378ADD"
AMBER  = "#BA7517"


def frequency_table(series: pd.Series, name: str) -> pd.DataFrame:
    """Calcula fi, fri (%), Fi, Fri (%) - 4 medidas do slide Frequency."""
    fi = series.value_counts(dropna=False).sort_index() if pd.api.types.is_numeric_dtype(series) \
         else series.value_counts(dropna=False)
    total = fi.sum()
    fri = (fi / total * 100).round(2)
    Fi  = fi.cumsum()
    Fri = (Fi / total * 100).round(2)
    table = pd.DataFrame({
        "fi (absoluta)":           fi.values,
        "fri (%) (relativa)":      fri.values,
        "Fi (acumulada)":          Fi.values,
        "Fri (%) (acum. rel.)":    Fri.values,
    }, index=fi.index)
    table.index.name = name
    return table


# =========================================================
# Variaveis qualitativas (categoricas)
# =========================================================
categorical_targets = [
    ("sentiment_clean",   "Sentimento (positive/mixed/negative)"),
    ("emotion_group_v2",  "Emocao agrupada (apos remapeamento)"),
    ("_source_file",      "Arquivo de origem (day1/day3/week1)"),
]

print("=" * 70)
print("FREQUENCIAS - Variaveis Qualitativas")
print("=" * 70)

freq_tables = {}
for col, label in categorical_targets:
    if col not in df.columns:
        continue
    table = frequency_table(df[col], col)
    freq_tables[col] = table
    print(f"\n>>> {label}")
    print("-" * 70)
    print(table.to_string())

# =========================================================
# Variaveis qualitativas binarias (temas one-hot)
# Frequencia agregada: quantos posts contem cada tema.
# =========================================================
print("\n" + "=" * 70)
print("FREQUENCIAS - Temas canonicos (binarios)")
print("=" * 70)

theme_cols = [c for c in df.columns if c.startswith("theme_")]
theme_freq = pd.DataFrame({
    "fi (presentes)":      [int(df[c].sum())          for c in theme_cols],
    "fri (%) (presentes)": [round(df[c].mean() * 100, 2) for c in theme_cols],
}, index=[c.replace("theme_", "") for c in theme_cols]).sort_values(
    "fi (presentes)", ascending=False
)
theme_freq.index.name = "tema"
print(theme_freq.to_string())
freq_tables["themes"] = theme_freq

# =========================================================
# Binning de variaveis continuas (Lecture 6: slide 'Binning')
# =========================================================
print("\n" + "=" * 70)
print("BINNING - engagement_score_v2 em intervalos")
print("=" * 70)
eng_bins = [0, 3, 4, 5, 6, 10]
eng_labels = ["[0-3)", "[3-4)", "[4-5)", "[5-6)", "[6-10]"]
eng_binned = pd.cut(df["engagement_score_v2"], bins=eng_bins,
                    labels=eng_labels, include_lowest=True, right=False)
table_eng = frequency_table(eng_binned, "engagement_bin")
print(table_eng.to_string())
freq_tables["engagement_binned"] = table_eng

print("\n" + "=" * 70)
print("BINNING - intensity_imputed em intervalos")
print("=" * 70)
int_bins = [0, 5, 6, 7, 8, 11]
int_labels = ["baixa [0-5)", "media-baixa [5-6)", "media [6-7)",
              "alta [7-8)", "muito alta [8-10]"]
int_binned = pd.cut(df["intensity_imputed"], bins=int_bins,
                    labels=int_labels, include_lowest=True, right=False)
table_int = frequency_table(int_binned, "intensity_bin")
print(table_int.to_string())
freq_tables["intensity_binned"] = table_int

print("\n" + "=" * 70)
print("BINNING - summary_length em intervalos")
print("=" * 70)
sum_bins = [0, 20, 40, 60, 80, 500]
sum_labels = ["curtissimo [0-20)", "curto [20-40)", "medio [40-60)",
              "longo [60-80)", "muito longo [80+)"]
sum_binned = pd.cut(df["summary_length"], bins=sum_bins,
                    labels=sum_labels, include_lowest=True, right=False)
table_sum = frequency_table(sum_binned, "summary_length_bin")
print(table_sum.to_string())
freq_tables["summary_binned"] = table_sum

# =========================================================
# Persistencia das tabelas de frequencia
# =========================================================
out_csv = out_dir / "frequency_tables.csv"
with open(out_csv, "w", encoding="utf-8") as f:
    for name, table in freq_tables.items():
        f.write(f"### {name}\n")
        table.to_csv(f)
        f.write("\n")
print(f"\nTabelas consolidadas salvas em: {out_csv}")

# =========================================================
# Visualizacao - barras horizontais para variaveis categoricas
# =========================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

plots = [
    ("sentiment_clean",  "Sentimento",        BLUE),
    ("emotion_group_v2", "Emocao (v2)",       AMBER),
    ("_source_file",     "Fonte (snapshot)",  "#639922"),
]

for i, (col, label, color) in enumerate(plots):
    if col not in freq_tables:
        continue
    table = freq_tables[col].sort_values("fi (absoluta)", ascending=True)
    ax = axes[i]
    ax.barh(table.index.astype(str), table["fi (absoluta)"].values,
            color=color, alpha=0.85, edgecolor="white")
    for j, (val, pct) in enumerate(zip(table["fi (absoluta)"], table["fri (%) (relativa)"])):
        ax.text(val + 0.3, j, f"{int(val)} ({pct:.1f}%)",
                va="center", fontsize=8, color="#333")
    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Frequencia absoluta (fi)", fontsize=9)
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

# Temas - barra horizontal
ax = axes[3]
themes = freq_tables["themes"].sort_values("fi (presentes)", ascending=True)
ax.barh(themes.index.astype(str), themes["fi (presentes)"].values,
        color="#7F77DD", alpha=0.85, edgecolor="white")
for j, (val, pct) in enumerate(zip(themes["fi (presentes)"], themes["fri (%) (presentes)"])):
    ax.text(val + 0.3, j, f"{int(val)} ({pct:.1f}%)",
            va="center", fontsize=8, color="#333")
ax.set_title("Temas canonicos (presenca por post)", fontsize=11, fontweight="bold", pad=8)
ax.set_xlabel("Posts contendo o tema", fontsize=9)
ax.grid(axis="x", color="#e0e0e0", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

fig.suptitle("Frequencias categoricas - Stranger Things S5 Finale Dataset",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()

img_path = img_dir / "frequency_categorical.png"
plt.savefig(img_path, dpi=150, bbox_inches="tight")
print(f"Grafico salvo em: {img_path}")
plt.show()
