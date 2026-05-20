import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

os.chdir(Path(__file__).resolve().parents[2])

df = pd.read_csv("dataset_limpo/cleaned_dataset_improved.csv")

BLUE   = "#378ADD"
AMBER  = "#BA7517"
RED    = "#E24B4A"
GREEN  = "#639922"
GRAY   = "#888780"
PURPLE = "#7F77DD"

sentiment_colors = {
    "positive": GREEN,
    "negative": RED,
    "mixed":    AMBER,
}
df["sent_color"] = df["sentiment_clean"].map(sentiment_colors).fillna(GRAY)

var_types = {
    "intensity_imputed":       "C",
    "engagement_score_v2":     "C",
    "num_theories":            "C",
    "avg_plausibility_imputed":"C",
    "num_themes":              "C",
    "num_canonical_themes":    "C",
    "summary_length":          "C",
    "sentiment_encoded":       "O",
    "emotion_encoded_v2":      "O",
}

pairs = [
    ("intensity_imputed",        "engagement_score_v2",
     "Intensidade x Engagement",         "Positiva Linear Forte"),
    ("num_theories",             "avg_plausibility_imputed",
     "No teorias x Plausibilidade media","Positiva Linear Forte"),
    ("num_themes",               "num_canonical_themes",
     "Temas brutos x Temas canonicos",   "Positiva Linear Forte\n(variavel derivada — esperado)"),
    ("summary_length",           "engagement_score_v2",
     "Tamanho do resumo x Engagement",   "Nula"),
    ("intensity_imputed",        "sentiment_encoded",
     "Intensidade x Sentimento",         "Positiva Linear Fraca"),
    ("emotion_encoded_v2",       "engagement_score_v2",
     "Emocao x Engagement",              "Negativa Linear Fraca"),
]

fig1, axes = plt.subplots(2, 3, figsize=(17, 11))
axes = axes.flatten()

for i, (xcol, ycol, title, corr_type) in enumerate(pairs):
    ax = axes[i]
    x = df[xcol]
    y = df[ycol]
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values

    r_p, pval_p = stats.pearsonr(x_clean, y_clean)
    r_s, pval_s = stats.spearmanr(x_clean, y_clean)
    r_k, pval_k = stats.kendalltau(x_clean, y_clean)

    coef = np.polyfit(x_clean, y_clean, 1)
    trend = np.poly1d(coef)
    residuals = y_clean - trend(x_clean)
    res_std = np.std(residuals)
    is_outlier = np.abs(residuals) > 2 * res_std

    colors_arr = df["sent_color"][mask].values

    ax.scatter(x_clean[~is_outlier], y_clean[~is_outlier],
               c=colors_arr[~is_outlier], alpha=0.65, s=35,
               edgecolors="white", linewidths=0.4, zorder=3)

    if is_outlier.any():
        ax.scatter(x_clean[is_outlier], y_clean[is_outlier],
                   c=colors_arr[is_outlier], alpha=0.9, s=80,
                   edgecolors="black", linewidths=1.3,
                   marker="D", zorder=5)

    xs = np.linspace(x_clean.min(), x_clean.max(), 200)
    ax.plot(xs, trend(xs), color=PURPLE, linewidth=1.8,
            linestyle="--", alpha=0.85, zorder=4)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel(xcol.replace("_", " "), fontsize=9)
    ax.set_ylabel(ycol.replace("_", " "), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(color="#e8e8e8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    is_ordinal = (var_types.get(xcol) == "O" or var_types.get(ycol) == "O")
    n_small    = len(x_clean) < 100
    if n_small and is_ordinal:
        rec = "Kendall (n<100, ordinal)"
    elif is_ordinal:
        rec = "Spearman (ordinal)"
    elif n_small:
        rec = "Kendall (n<100)"
    else:
        rec = "Pearson (linear + n>=100)"

    def sig(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))

    def strength(r):
        a = abs(r)
        if a >= 0.7: return "strong"
        if a >= 0.4: return "moderate"
        if a >= 0.2: return "weak"
        return "none"

    info = (f"r Pearson  = {r_p:+.2f} {sig(pval_p)}  [{strength(r_p)}]\n"
            f"rho Spearm = {r_s:+.2f} {sig(pval_s)}  [{strength(r_s)}]\n"
            f"tau Kendall= {r_k:+.2f} {sig(pval_k)}  [{strength(r_k)}]\n"
            f"Tipo: {corr_type}\n"
            f"Recomendado: {rec}")
    ax.text(0.98, 0.04, info, transform=ax.transAxes,
            fontsize=7.5, ha="right", va="bottom", color="#333",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#bbb", alpha=0.95))

    if abs(r_p) >= 0.7 and xcol != "num_themes":
        ax.text(0.02, 0.97, "! correlacao != causalidade",
                transform=ax.transAxes, fontsize=7.5, ha="left", va="top",
                color="#7a4800",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff8e1",
                          edgecolor="#e0b020", alpha=0.95))

    n_out = is_outlier.sum()
    if n_out > 0:
        ax.text(0.02, 0.04, f"Outliers: {n_out}",
                transform=ax.transAxes, fontsize=7.5, ha="left", va="bottom",
                color="#333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fce8e8",
                          edgecolor="#e08080", alpha=0.9))

legend_handles = [
    mpatches.Patch(color=GREEN,  label="Positivo"),
    mpatches.Patch(color=RED,    label="Negativo"),
    mpatches.Patch(color=AMBER,  label="Misto"),
    plt.Line2D([0], [0], color=PURPLE, linewidth=1.8,
               linestyle="--", label="Tendencia"),
    plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=GRAY,
               markeredgecolor="black", markersize=9,
               label="Outlier (|residuo| > 2sigma)"),
]
fig1.legend(handles=legend_handles, loc="lower center", ncol=5,
            fontsize=9, frameon=True, framealpha=0.95,
            edgecolor="#ccc", bbox_to_anchor=(0.5, -0.02))

fig1.suptitle(
    "Scatter Plots — Stranger Things S5 Finale Dataset\n"
    "Pearson r  |  Spearman rho  |  Tipo de correlacao  |  Outliers (losango)",
    fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("plot_image/scatter_plots.png", dpi=150, bbox_inches="tight")
plt.show()

cols_heat = [
    "intensity_imputed",
    "avg_plausibility_imputed",
    "num_theories",
    "num_themes",
    "num_canonical_themes",
    "engagement_score_v2",
    "summary_length",
    "sentiment_encoded",
    "emotion_encoded_v2",
]

labels_heat = [
    "Intensidade (C)",
    "Plausibilidade\nmédia (C)",
    "Nº teorias (C)",
    "Nº temas (C)",
    "Temas\ncanônicos (C)",
    "Engagement (C)",
    "Tamanho\nresumo (C)",
    "Sentimento\nencoded (O)",
    "Emoção\nencoded (O)",
]

corr_pearson  = df[cols_heat].corr(method="pearson")
corr_spearman = df[cols_heat].corr(method="spearman")
corr_kendall  = df[cols_heat].corr(method="kendall")

cmap = LinearSegmentedColormap.from_list(
    "bwr_custom", ["#378ADD", "#ffffff", "#E24B4A"], N=256)

fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 9))


def draw_heatmap(ax, corr_matrix, title, labels):
    n = len(labels)
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8.5)

    for row in range(n):
        for col in range(n):
            val = corr_matrix.values[row, col]
            text_color = "white" if abs(val) > 0.65 else "#222222"
            weight = "bold" if abs(val) >= 0.7 and row != col else "normal"
            ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight=weight)

    for k in range(n):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                   fill=False, edgecolor="#999999",
                                   linewidth=0.8, zorder=5))

    for row in range(n):
        for col in range(n):
            if row != col and abs(corr_matrix.values[row, col]) >= 0.70:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                           fill=False, edgecolor="#FFB800",
                                           linewidth=2.2, zorder=6))

    ax.axhline(6.5, color="#555", linewidth=1.5, linestyle=":")
    ax.axvline(6.5, color="#555", linewidth=1.5, linestyle=":")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    return im


im1 = draw_heatmap(
    ax1, corr_pearson,
    "Pearson r\n(linear, normal, sensivel a outliers - p/ (C))",
    labels_heat)

im2 = draw_heatmap(
    ax2, corr_spearman,
    "Spearman rho\n(monotonica, sem assuncao distribucional - p/ (O))",
    labels_heat)

im3 = draw_heatmap(
    ax3, corr_kendall,
    "Kendall tau\n(monotonica, robusto a outliers, ideal n<100)",
    labels_heat)

cbar = fig2.colorbar(im3, ax=[ax1, ax2, ax3], fraction=0.014, pad=0.02)
cbar.set_label("Coeficiente de correlacao", fontsize=10)
cbar.ax.tick_params(labelsize=9)

note = (
    "(C) = variavel continua   |   (O) = variavel ordinal   |   "
    "Borda amarela = multicolinearidade (|r| >= 0.70)   |   "
    "Linha pontilhada = separacao C / O\n"
    "Forca: |r|>=0.7 strong  |  0.4-0.7 moderate  |  0.2-0.4 weak  |  <0.2 none. "
    "Correlacao != causalidade - variaveis intervenientes (sentimento, fonte) podem mediar associacoes."
)
fig2.text(0.5, -0.04, note, ha="center", fontsize=9, color="#444",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f9f9",
                    edgecolor="#cccccc", alpha=0.95))

fig2.suptitle(
    "Heatmap de Correlacao - Pearson | Spearman | Kendall\n"
    "Stranger Things S5 Finale Dataset",
    fontsize=14, fontweight="bold", y=1.03)

plt.tight_layout()
plt.savefig("plot_image/heatmap_correlacao.png", dpi=150, bbox_inches="tight")
plt.show()
