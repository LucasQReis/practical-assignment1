import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    "Plausibilidade média das teorias",
    "Nº de temas por post",
    "Engagement score",
    "Tamanho do resumo (palavras)",
    "Temas canônicos presentes",
]

bins_map = {
    "intensity_imputed":         range(4, 11),
    "avg_plausibility_imputed":  np.arange(0, 9.5, 0.5),
    "num_themes":                range(0, 7),
    "engagement_score_v2":       np.arange(2, 6.5, 0.25),
    "summary_length":            np.arange(0, 110, 10),
    "num_canonical_themes":      range(0, 8),
}

BLUE   = "#378ADD"
MEAN_C = "#BA7517"
MED_C  = "#1a1a2e"

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, (col, label) in enumerate(zip(cols, labels)):
    ax = axes[i]
    data = df[col].dropna()

    mean   = data.mean()
    median = data.median()
    skew   = data.skew()
    std    = data.std()

    ax.hist(
        data,
        bins=bins_map[col],
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
        zorder=2,
    )

    ax.axvline(mean,   color=MEAN_C, linewidth=1.8, linestyle="--", label=f"Média  {mean:.1f}", zorder=3)
    ax.axvline(median, color=MED_C,  linewidth=1.8, linestyle="-",  label=f"Mediana {median:.1f}", zorder=3)

    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Valor", fontsize=9)
    ax.set_ylabel("Frequência", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if skew > 0.5:
        skew_label = f"assimetria positiva ({skew:.2f})"
    elif skew < -0.5:
        skew_label = f"assimetria negativa ({skew:.2f})"
    else:
        skew_label = f"simétrico ({skew:.2f})"

    stats_text = f"std={std:.2f} · {skew_label}"
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8, color="#555555",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

    ax.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc")

fig.suptitle(
    "Histogramas — Stranger Things S5 Finale Dataset",
    fontsize=14, fontweight="bold", y=1.01,
)

plt.tight_layout()
plt.savefig("plot_image/histogramas.png", dpi=150, bbox_inches="tight")
plt.show()