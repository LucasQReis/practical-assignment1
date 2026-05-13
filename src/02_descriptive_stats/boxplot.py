import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Bootstrap: roda a partir da raiz do projeto independente do cwd
os.chdir(Path(__file__).resolve().parents[2])

df = pd.read_csv("dataset_limpo/cleaned_dataset_improved.csv")

cols = [
    "intensity_imputed",
    "avg_plausibility_imputed",
    "num_themes",
    "engagement_score_v2",
    "summary_length",
    "num_canonical_themes",
]

labels = [
    "Intensidade emocional",
    "Plausibilidade média\ndas teorias",
    "Nº de temas\npor post",
    "Engagement\nscore",
    "Tamanho do\nresumo (palavras)",
    "Temas canônicos\npresentes",
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

BLUE   = "#378ADD"
MEDIAN = "#1a1a2e"
RED    = "#E24B4A"

for i, (col, label) in enumerate(zip(cols, labels)):
    ax = axes[i]
    data = df[col].dropna()

    bp = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor=BLUE + "33", color=BLUE, linewidth=1.5),
        medianprops=dict(color=MEDIAN, linewidth=2.5),
        whiskerprops=dict(color=BLUE, linewidth=1.5, linestyle="--"),
        capprops=dict(color=BLUE, linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor=RED, markeredgecolor=RED,
                        markersize=5, alpha=0.7),
    )

    mean   = data.mean()
    median = data.median()
    skew   = data.skew()
    Q1     = data.quantile(0.25)
    Q3     = data.quantile(0.75)
    IQR    = Q3 - Q1
    n_out  = int(((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum())

    ax.axhline(mean, color="#BA7517", linewidth=1.2, linestyle=":", alpha=0.8)

    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    stats_text = (
        f"média={mean:.2f}  mediana={median:.2f}\n"
        f"skew={skew:.2f}  outliers={n_out}"
    )
    ax.text(
        0.97, 0.03, stats_text,
        transform=ax.transAxes,
        fontsize=8, color="#555555",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

# Legenda global
median_patch = mpatches.Patch(color=MEDIAN,        label="Mediana")
mean_patch   = mpatches.Patch(color="#BA7517",      label="Média (linha pontilhada)")
iqr_patch    = mpatches.Patch(facecolor=BLUE+"33",
                               edgecolor=BLUE,       label="IQR (Q1–Q3)")
out_patch    = mpatches.Patch(color=RED,             label="Outliers")

fig.legend(
    handles=[iqr_patch, median_patch, mean_patch, out_patch],
    loc="lower center", ncol=4, fontsize=9,
    frameon=True, framealpha=0.9, edgecolor="#cccccc",
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Boxplots — Stranger Things S5 Finale Dataset",
    fontsize=14, fontweight="bold", y=1.01,
)

plt.tight_layout()
plt.savefig("plot_image/boxplots.png", dpi=150, bbox_inches="tight")
plt.show()