import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

os.chdir(Path(__file__).resolve().parents[2])

pd.options.display.float_format = "{:,.4f}".format

df = pd.read_csv("dataset_limpo/cleaned_dataset_improved.csv")
out_dir = Path("dataset_limpo")
img_dir = Path("plot_image")
img_dir.mkdir(exist_ok=True)

ALPHA = 0.05
N = len(df)

def dispersion_summary(series: pd.Series) -> pd.Series:
    s = series.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return pd.Series({
        "count":            int(s.count()),
        "min":              s.min(),
        "max":              s.max(),
        "range":            s.max() - s.min(),
        "mean":             s.mean(),
        "median":           s.median(),
        "variance":         s.var(),
        "standard_deviation": s.std(),
        "q1":               q1,
        "q3":               q3,
        "iqr":              q3 - q1,
        "coefficient_of_variation": s.std() / s.mean() if s.mean() != 0 else np.nan,
    })

disp_compare = pd.DataFrame({
    "engagement_score_v2": dispersion_summary(df["engagement_score_v2"]),
    "num_theories":        dispersion_summary(df["num_theories"]),
}).T.round(3)

results = []

def add_result(hid, name, stat_name, stat_value, pvalue, effect_name=None, effect_value=None,
               notes=""):
    decision = "reject H0" if pvalue < ALPHA else "fail to reject H0"
    results.append({
        "id":            hid,
        "hypothesis":    name,
        "test_statistic": stat_name,
        "statistic":     round(stat_value, 4) if isinstance(stat_value, (int, float, np.floating)) else stat_value,
        "p_value":       pvalue,
        "alpha":         ALPHA,
        "decision":      decision,
        "effect_size":   effect_name,
        "effect_value":  round(effect_value, 3) if isinstance(effect_value, (int, float, np.floating)) else effect_value,
        "notes":         notes,
    })
    if effect_name:
        print(f"  Effect size: {effect_name} = {effect_value:.3f}")
    print(f"  Decision (alpha={ALPHA}): {decision}")
    if notes:
        print(f"  Note: {notes}")

for hid, col in [("H1", "engagement_score_v2"), ("H2", "num_theories")]:
    data = df[col].dropna()
    W, pval = stats.shapiro(data)
    add_result(hid, f"{col} ~ Normal", "W (Shapiro-Wilk)", W, pval,
               notes=f"skew={data.skew():.2f}, kurt={data.kurt():.2f}")

# QQ-plot lado a lado
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (hid, col, label) in zip(axes, [
    ("H1", "engagement_score_v2", "Engagement (esperamos quasi-normal)"),
    ("H2", "num_theories",        "Num theories (esperamos NAO normal)"),
]):
    stats.probplot(df[col].dropna(), dist="norm", plot=ax)
    ax.set_title(f"[{hid}] {label}", fontsize=11, fontweight="bold")
    ax.grid(color="#e8e8e8", linewidth=0.7)
    ax.set_axisbelow(True)
plt.tight_layout()
qq_path = img_dir / "qqplots_normality.png"
plt.savefig(qq_path, dpi=150, bbox_inches="tight")

obs = df["sentiment_clean"].value_counts().reindex(["positive", "negative", "mixed"]).fillna(0)
exp = np.repeat(N / 3, 3)
chi2_stat, pval = stats.chisquare(f_obs=obs.values, f_exp=exp)

k = len(obs)
cramers_v_gof = np.sqrt(chi2_stat / (N * (k - 1)))

add_result("H3", "Proporcao de sentimentos = 1/3",
           "Chi-square (goodness-of-fit)", chi2_stat, pval,
           effect_name="Cramer's V", effect_value=cramers_v_gof,
           notes=f"observados: pos={int(obs['positive'])}, neg={int(obs['negative'])}, mixed={int(obs['mixed'])}")

for hid, x_col, y_col in [
    ("H4", "intensity_imputed", "engagement_score_v2"),
    ("H5", "summary_length",     "engagement_score_v2"),
]:
    x = df[x_col].dropna()
    y = df[y_col].dropna()
    mask = df[x_col].notna() & df[y_col].notna()
    x, y = df.loc[mask, x_col], df.loc[mask, y_col]

    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    r_k, p_k = stats.kendalltau(x, y)

    add_result(f"{hid}-P", f"rho({x_col}, {y_col}) = 0  [Pearson]",
               "Pearson r", r_p, p_p, effect_name="|r|", effect_value=abs(r_p))
    add_result(f"{hid}-S", f"rho({x_col}, {y_col}) = 0  [Spearman]",
               "Spearman rho", r_s, p_s, effect_name="|rho|", effect_value=abs(r_s))
    add_result(f"{hid}-K", f"rho({x_col}, {y_col}) = 0  [Kendall]",
               "Kendall tau", r_k, p_k, effect_name="|tau|", effect_value=abs(r_k))

df_2g = df[df["sentiment_clean"].isin(["positive", "negative"])].copy()
g_pos = df_2g.loc[df_2g["sentiment_clean"] == "positive", "engagement_score_v2"].dropna()
g_neg = df_2g.loc[df_2g["sentiment_clean"] == "negative", "engagement_score_v2"].dropna()

t_stat, t_pval = stats.ttest_ind(g_pos, g_neg, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(g_pos, g_neg, alternative="two-sided")

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled

d = cohens_d(g_pos, g_neg)

add_result("T1", "engagement: mu_pos = mu_neg [Welch]",
           "Welch t-stat", t_stat, t_pval,
           effect_name="Cohen's d", effect_value=d,
           notes=f"|d|={abs(d):.2f} -> "
                 + ("small" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"))
add_result("T2", "engagement: F_pos = F_neg [Mann-Whitney]",
           "U stat", u_stat, u_pval,
           notes="non-parametric, robust to non-normality")

def eta_squared(df_g, group_col, value_col):
    grand_mean = df_g[value_col].mean()
    levels = df_g[group_col].dropna().unique()
    ss_between = sum(
        len(df_g[df_g[group_col] == lvl])
        * (df_g[df_g[group_col] == lvl][value_col].mean() - grand_mean) ** 2
        for lvl in levels
    )
    ss_total = ((df_g[value_col] - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else 0.0


for hid, group_col, label in [
    ("H6", "sentiment_clean", "engagement por sentimento (mediador)"),
    ("H7", "_source_file",    "engagement por snapshot (extraneous)"),
]:
    sub = df[[group_col, "engagement_score_v2"]].dropna()
    groups = [g["engagement_score_v2"].values for _, g in sub.groupby(group_col)]
    f_stat, p_anova = stats.f_oneway(*groups)
    h_stat, p_kw    = stats.kruskal(*groups)
    eta2 = eta_squared(sub, group_col, "engagement_score_v2")
    descr = sub.groupby(group_col)["engagement_score_v2"].agg(["count", "mean", "median", "std"]).round(3)

    add_result(hid, f"{label} [ANOVA]",
               "F", f_stat, p_anova,
               effect_name="eta_squared", effect_value=eta2,
               notes=("eta^2 " + ("small" if eta2 < 0.06 else "medium" if eta2 < 0.14 else "large")))
    add_result(f"{hid}-KW", f"{label} [Kruskal-Wallis]",
               "H", h_stat, p_kw,
               notes="non-parametric backup")

ct = pd.crosstab(df["sentiment_clean"], df["emotion_group_v2"])
ct_pct = pd.crosstab(df["sentiment_clean"], df["emotion_group_v2"], normalize="index") * 100
chi2_ind, p_ind, dof, expected = stats.chi2_contingency(ct)
n_total = ct.values.sum()
min_dim = min(ct.shape) - 1
cramers_v = np.sqrt(chi2_ind / (n_total * min_dim))

add_result("X1", "sentiment vs emotion_group: independentes",
           f"Chi-square (df={dof})", chi2_ind, p_ind,
           effect_name="Cramer's V", effect_value=cramers_v,
           notes=("V " + ("weak" if cramers_v < 0.2 else "moderate" if cramers_v < 0.4 else "strong"))
           + f", n={n_total}")

exp_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(1)
res_df = pd.DataFrame(results)
k_tests = len(res_df)
alpha_bonf = ALPHA / k_tests

res_df["decision_bonf"] = np.where(
    res_df["p_value"] < alpha_bonf, "reject H0 (Bonferroni)", "fail to reject H0 (Bonferroni)"
)

display_cols = ["id", "hypothesis", "test_statistic", "statistic",
                "p_value", "decision", "decision_bonf", "effect_size", "effect_value"]

csv_path = out_dir / "hypothesis_tests_results.csv"
res_df.to_csv(csv_path, index=False, encoding="utf-8")
fig, ax = plt.subplots(figsize=(13, 9))

res_plot = res_df.copy()
res_plot["nlog_p"] = -np.log10(res_plot["p_value"].clip(lower=1e-300))
res_plot = res_plot.sort_values("nlog_p")

colors = ["#149945" if d.startswith("reject") else "#888888" for d in res_plot["decision_bonf"]]
y_pos = np.arange(len(res_plot))

ax.barh(y_pos, res_plot["nlog_p"].clip(upper=20),
        color=colors, alpha=0.85, edgecolor="white")
ax.axvline(-np.log10(ALPHA), color="#E24B4A", linestyle="--",
           linewidth=1.6, label=f"alpha = {ALPHA}  (-log10 = {-np.log10(ALPHA):.2f})")
ax.axvline(-np.log10(alpha_bonf), color="#7a4800", linestyle="--",
           linewidth=1.6, label=f"alpha_bonf = {alpha_bonf:.5f}  (-log10 = {-np.log10(alpha_bonf):.2f})")

ax.set_yticks(y_pos)
ax.set_yticklabels([f"[{r.id}] {r.hypothesis[:60]}" for r in res_plot.itertuples()],
                   fontsize=8)
ax.set_xlabel("-log10(p-value)   (maior = mais evidencia contra H0)", fontsize=10)
ax.set_title(
    f"Sumario de testes ({k_tests} hipoteses)  -  Stranger Things S5 Finale\n"
    f"Verde = rejeita H0 com Bonferroni  |  Cinza = nao rejeita",
    fontsize=12, fontweight="bold", pad=12,
)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", color="#e8e8e8", linewidth=0.7)
ax.set_axisbelow(True)
ax.set_xlim(0, 21)

plt.tight_layout()
summary_path = img_dir / "hypothesis_tests_summary.png"
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.show()
