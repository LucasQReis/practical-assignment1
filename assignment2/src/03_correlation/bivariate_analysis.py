# =========================================================
# bivariate_analysis.py - Analise bivariada
# ---------------------------------------------------------
# Investiga o conceito de variavel interveniente (mediador):
#   1. Mapeamento dos papeis das variaveis no dominio
#      (Independent/Dependent/Intervening/Extraneous).
#   2. Correlacoes condicionais - mesmo par X,Y avaliado
#      globalmente e por grupo de sentimento/emocao/snapshot,
#      revelando casos de Paradoxo de Simpson.
#   3. Boxplots agrupados (engagement por sentiment/emocao/
#      source) + teste Kruskal-Wallis para validar mediadores.
#   4. Scatter condicional colorido por sentimento + linha de
#      tendencia global vs por grupo.
# Saidas: plot_image/boxplots_grouped.png +
#         plot_image/scatter_conditional.png.
# =========================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Bootstrap: cwd para a raiz de assignment2/.
os.chdir(Path(__file__).resolve().parents[2])

pd.options.display.float_format = "{:,.3f}".format

df = pd.read_csv("dataset_limpo/cleaned_dataset_improved.csv")
img_dir = Path("plot_image")
img_dir.mkdir(exist_ok=True)

BLUE   = "#378ADD"
AMBER  = "#BA7517"
RED    = "#E24B4A"
GREEN  = "#639922"
PURPLE = "#7F77DD"
GRAY   = "#888780"

VARIABLE_ROLES = {
    "Independent (drivers)": [
        ("intensity_imputed",        "Intensidade emocional do post"),
        ("num_theories",             "Numero de teorias do post"),
        ("num_themes",               "Riqueza tematica do post"),
        ("summary_length",           "Tamanho do resumo (proxy de profundidade)"),
    ],
    "Dependent (outcomes)": [
        ("engagement_score_v2",      "Score composto de engajamento"),
        ("avg_plausibility_imputed", "Plausibilidade media das teorias"),
    ],
    "Intervening (mediators)": [
        ("sentiment_clean",          "Sentimento - media entre estimulo e engajamento"),
        ("emotion_group_v2",         "Emocao agrupada - canaliza tipo de resposta"),
    ],
    "Extraneous (controls)": [
        ("_source_file",             "Snapshot temporal (day1/day3/week1)"),
        ("published_dayofweek",      "Dia da semana - sazonalidade da plataforma"),
        ("published_hour",           "Hora do dia - sazonalidade circadiana"),
    ],
}

for role, items in VARIABLE_ROLES.items():
    print(f"\n[{role}]")
    for col, desc in items:
        present = "OK " if col in df.columns else "-- "
        print(f"  {present}{col:<28} {desc}")

def triple_corr(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan, int(mask.sum())
    a, b = x[mask].values, y[mask].values
    return (stats.pearsonr(a, b)[0],
            stats.spearmanr(a, b)[0],
            stats.kendalltau(a, b)[0],
            int(mask.sum()))


def conditional_correlation_report(x_col, y_col, by_col, df_=df):
    print(f"\n>>> Par: {x_col} x {y_col}  (mediador: {by_col})")
    print("-" * 72)
    rp, rs, rk, n = triple_corr(df_[x_col], df_[y_col])
    print(f"  GLOBAL (n={n}):  Pearson={rp:+.3f}  Spearman={rs:+.3f}  Kendall={rk:+.3f}")
    rows = []
    for grp, sub in df_.groupby(by_col):
        rp, rs, rk, n = triple_corr(sub[x_col], sub[y_col])
        rows.append({"grupo": grp, "n": n,
                     "Pearson": round(rp, 3), "Spearman": round(rs, 3), "Kendall": round(rk, 3)})
    cond = pd.DataFrame(rows).sort_values("n", ascending=False)
    print(cond.to_string(index=False))
    return cond


conditional_correlation_report("intensity_imputed", "engagement_score_v2", "sentiment_clean")
conditional_correlation_report("num_theories", "engagement_score_v2", "emotion_group_v2")
conditional_correlation_report("intensity_imputed", "engagement_score_v2", "_source_file")
conditional_correlation_report("summary_length", "engagement_score_v2", "sentiment_clean")

categorical_targets = [
    ("sentiment_clean",   ["positive", "mixed", "negative"], "Sentimento (mediador)"),
    ("emotion_group_v2",  None,                              "Emocao agrupada (mediador)"),
    ("_source_file",      ["day1", "day3", "week1"],         "Snapshot (extraneous)"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (cat_col, order, title) in zip(axes, categorical_targets):
    groups = order if order is not None else (
        df[cat_col].value_counts().index.tolist()
    )
    data = [df.loc[df[cat_col] == g, "engagement_score_v2"].dropna().values
            for g in groups]
    bp = ax.boxplot(
        data, vert=True, patch_artist=True, widths=0.55, labels=groups,
        boxprops=dict(facecolor=BLUE + "33", color=BLUE, linewidth=1.5),
        medianprops=dict(color="#1a1a2e", linewidth=2.3),
        whiskerprops=dict(color=BLUE, linewidth=1.4, linestyle="--"),
        capprops=dict(color=BLUE, linewidth=1.4),
        flierprops=dict(marker="o", markerfacecolor=RED, markeredgecolor=RED,
                        markersize=5, alpha=0.7),
    )
    means = [np.mean(d) if len(d) else np.nan for d in data]
    ax.scatter(range(1, len(groups) + 1), means, marker="D",
               color=AMBER, s=45, zorder=10, label="Media")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("engagement_score_v2", fontsize=9)
    ax.tick_params(axis="x", labelsize=9, rotation=20)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    valid = [d for d in data if len(d) >= 2]
    if len(valid) >= 2:
        H, pval = stats.kruskal(*valid)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        ax.text(0.97, 0.97,
                f"Kruskal-Wallis H={H:.2f}\np={pval:.3g} {sig}",
                transform=ax.transAxes, fontsize=8, color="#333",
                ha="right", va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#bbb", alpha=0.95))

fig.suptitle("Engagement por categoria - distribuicao agrupada (categorica x continua)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
img_path = img_dir / "boxplots_grouped.png"
plt.savefig(img_path, dpi=150, bbox_inches="tight")
fig2, ax = plt.subplots(figsize=(10, 7))

sent_colors = {"positive": GREEN, "negative": RED, "mixed": AMBER}
mask_all = df["intensity_imputed"].notna() & df["engagement_score_v2"].notna()

for sent, color in sent_colors.items():
    m = mask_all & (df["sentiment_clean"] == sent)
    if m.sum() == 0:
        continue
    x = df.loc[m, "intensity_imputed"].values
    y = df.loc[m, "engagement_score_v2"].values
    ax.scatter(x, y, c=color, alpha=0.55, s=45, edgecolors="white",
               linewidths=0.5, label=f"{sent} (n={m.sum()})", zorder=3)
    if len(x) >= 3:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.poly1d(coef)(xs), color=color, linewidth=1.7,
                linestyle="--", alpha=0.8, zorder=4)

xg = df.loc[mask_all, "intensity_imputed"].values
yg = df.loc[mask_all, "engagement_score_v2"].values
coef_g = np.polyfit(xg, yg, 1)
xs_g = np.linspace(xg.min(), xg.max(), 100)
ax.plot(xs_g, np.poly1d(coef_g)(xs_g), color=PURPLE, linewidth=2.4,
        linestyle="-", alpha=0.9, label="Tendencia GLOBAL", zorder=5)

rp_g = stats.pearsonr(xg, yg)[0]
rs_g = stats.spearmanr(xg, yg)[0]
rk_g = stats.kendalltau(xg, yg)[0]

ax.set_title(
    "Scatter condicional: intensity x engagement, mediado por sentiment\n"
    f"GLOBAL: r={rp_g:+.2f} | rho={rs_g:+.2f} | tau={rk_g:+.2f}  "
    "(linhas tracejadas = tendencia por sentimento)",
    fontsize=11, fontweight="bold", pad=10)
ax.set_xlabel("intensity_imputed", fontsize=10)
ax.set_ylabel("engagement_score_v2", fontsize=10)
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(color="#e8e8e8", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

img_path2 = img_dir / "scatter_conditional.png"
plt.savefig(img_path2, dpi=150, bbox_inches="tight")
plt.show()
