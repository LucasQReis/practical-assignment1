# =========================================================
# histogram.py - Histogramas univariados
# ---------------------------------------------------------
# Gera um grid 2x3 com 6 histogramas (intensity, plausibility,
# num_themes, engagement, summary_length, canonical_themes).
# Cada subplot mostra:
#   - barras do histograma
#   - curva KDE sobreposta (gaussian_kde)
#   - linhas verticais de media e mediana
#   - card com std, moda, skew e classificacao de kurtosis
# Saida: plot_image/histogramas.png.
# =========================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scistats

# Bootstrap: cwd para a raiz de assignment2/.
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

KDE_C = "#7F77DD"

for i, (col, label) in enumerate(zip(cols, labels)):
    ax = axes[i]
    data = df[col].dropna()

    mean   = data.mean()
    median = data.median()
    mode_vals = data.mode()
    mode_v = float(mode_vals.iloc[0]) if not mode_vals.empty else np.nan
    skew   = data.skew()
    kurt   = data.kurt()
    std    = data.std()

    counts, bin_edges, _ = ax.hist(
        data,
        bins=bins_map[col],
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
        zorder=2,
    )

    # KDE sobreposto (Lecture 6: histplot(..., kde=True))
    if data.nunique() > 1 and data.std() > 0:
        try:
            kde = scistats.gaussian_kde(data)
            xs = np.linspace(bin_edges[0], bin_edges[-1], 200)
            bin_width = bin_edges[1] - bin_edges[0]
            ax.plot(xs, kde(xs) * len(data) * bin_width,
                    color=KDE_C, linewidth=1.6, alpha=0.85, zorder=4,
                    label="KDE")
        except Exception:
            pass

    ax.axvline(mean,   color=MEAN_C, linewidth=1.8, linestyle="--", label=f"Media  {mean:.1f}", zorder=3)
    ax.axvline(median, color=MED_C,  linewidth=1.8, linestyle="-",  label=f"Mediana {median:.1f}", zorder=3)

    ax.set_title(label, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Valor", fontsize=9)
    ax.set_ylabel("Frequencia", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if skew > 0.5:
        skew_label = f"skew+ ({skew:+.2f})"
    elif skew < -0.5:
        skew_label = f"skew- ({skew:+.2f})"
    else:
        skew_label = f"simetrico ({skew:+.2f})"

    if kurt > 0.5:
        kurt_label = f"leptokurtic ({kurt:+.2f})"
    elif kurt < -0.5:
        kurt_label = f"platykurtic ({kurt:+.2f})"
    else:
        kurt_label = f"mesokurtic ({kurt:+.2f})"

    stats_text = (
        f"std={std:.2f}  moda={mode_v:.1f}\n"
        f"{skew_label}\n{kurt_label}"
    )
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=7.5, color="#555555",
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#cccccc", alpha=0.85),
    )

    ax.legend(fontsize=7.5, framealpha=0.9, edgecolor="#cccccc", loc="upper left")

fig.suptitle(
    "Histogramas — Stranger Things S5 Finale Dataset",
    fontsize=14, fontweight="bold", y=1.01,
)

plt.tight_layout()
plt.savefig("plot_image/histogramas.png", dpi=150, bbox_inches="tight")
plt.show()