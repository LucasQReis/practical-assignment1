import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scistats

os.chdir(Path(__file__).resolve().parents[2])
pd.options.display.float_format = "{:,.3f}".format

df = pd.read_csv("dataset_limpo/cleaned_dataset_improved.csv")
out_dir = Path("dataset_limpo")
img_dir = Path("plot_image")
img_dir.mkdir(exist_ok=True)

n_sample = len(df)
framework = {
    "Populacao alvo":
        "Todas as discussoes/posts publicos em redes sociais (Reddit) sobre o "
        "finale da temporada 5 de Stranger Things - elemento populacional teorico.",
    "Frame amostral":
        "Posts crawled em 3 janelas (day1, day3, week1) atraves do feed Reddit RSS, "
        "analisados por LLM (campos sentiment/intensity/emotion/theory/themes).",
    "Amostra observada":
        f"n = {n_sample} discussoes apos deduplicacao por URL (3 snapshots + jsonnormalize).",
    "Limitacoes amostrais (caveat slide 25)":
        "(a) viesado por discoverability do Reddit RSS; "
        "(b) janela curta (1 semana); "
        "(c) rotulagem por LLM - introduz ruido sistematico nas variaveis derivadas; "
        "(d) auto-selecao - quem posta nao representa quem assiste.",
}
for k, v in framework.items():
    print(f"\n[{k}]\n  {v}")

param_stat = pd.DataFrame([
    {"Conceito": "Media de intensidade",
     "Parametro (pop.)": "mu_intensity",
     "Estatistica (amostra)": f"x_bar = {df['intensity_imputed'].mean():.3f}",
     "Tipo de inferencia": "Estimativa pontual + IC"},
    {"Conceito": "Proporcao de posts positivos",
     "Parametro (pop.)": "pi_positive",
     "Estatistica (amostra)": f"p = {df['is_positive'].mean():.3f}",
     "Tipo de inferencia": "Estimativa pontual + IC"},
    {"Conceito": "Desvio padrao do engagement",
     "Parametro (pop.)": "sigma_engagement",
     "Estatistica (amostra)": f"s = {df['engagement_score_v2'].std():.3f}",
     "Tipo de inferencia": "Estimativa pontual"},
    {"Conceito": "Correlacao intensidade x engagement",
     "Parametro (pop.)": "rho (Pearson) ou rho_s (Spearman)",
     "Estatistica (amostra)": f"r = {df['intensity_imputed'].corr(df['engagement_score_v2']):.3f}",
     "Tipo de inferencia": "Teste de associacao"},
    {"Conceito": "Diferenca de engagement entre sentimentos",
     "Parametro (pop.)": "mu_pos - mu_neg (e mu_pos - mu_mixed)",
     "Estatistica (amostra)":
         f"x_bar_pos = {df.loc[df['sentiment_clean']=='positive','engagement_score_v2'].mean():.3f}, "
         f"x_bar_neg = {df.loc[df['sentiment_clean']=='negative','engagement_score_v2'].mean():.3f}",
     "Tipo de inferencia": "Teste de medias / Kruskal-Wallis"},
])

hypotheses = [
    {
        "id": "H1",
        "categoria": "Univariada (distribuicao)",
        "variavel": "engagement_score_v2",
        "achado_EDA":
            "Skew=-0.09, kurt=+0.10 -> aproximadamente simetrico e mesokurtico "
            "(parecido com normal).",
        "H0":  "engagement_score_v2 segue distribuicao Normal na populacao "
               "(F = N(mu, sigma^2))",
        "H1":  "engagement_score_v2 NAO segue distribuicao Normal",
        "parametro": "forma da distribuicao F",
        "teste_sugerido": "Shapiro-Wilk ou Kolmogorov-Smirnov contra N",
        "tipo": "two-tailed (bilateral)",
        "alpha_sugerido": 0.05,
        "Type_I_no_contexto":
            "Concluir 'engagement nao e normal' quando na verdade e -> "
            "abandono desnecessario de testes parametricos.",
        "Type_II_no_contexto":
            "Concluir 'engagement e normal' quando nao e -> aplicacao "
            "indevida de Pearson, t-test, etc.",
    },
    {
        "id": "H2",
        "categoria": "Univariada (distribuicao)",
        "variavel": "num_theories",
        "achado_EDA":
            "Skew=+2.54, kurt=+6.20 -> fortemente assimetrica a direita, "
            "leptokurtica. Media=0.35; mediana=0; moda=0.",
        "H0":  "num_theories segue Normal (improvavel a priori, baseline da aula)",
        "H1":  "num_theories deve assimetria a direita - melhor modelada por "
               "Poisson/Negative-Binomial",
        "parametro": "forma da distribuicao F",
        "teste_sugerido": "Shapiro-Wilk; visual: QQ-plot",
        "tipo": "two-tailed",
        "alpha_sugerido": 0.05,
        "Type_I_no_contexto":
            "Rejeitar normalidade quando ela vale - inocua aqui (esperamos rejeitar).",
        "Type_II_no_contexto":
            "Nao rejeitar normalidade aqui significaria poder amostral baixo demais.",
    },
    {
        "id": "H3",
        "categoria": "Univariada (proporcao)",
        "variavel": "is_positive",
        "achado_EDA":
            "49.2% dos posts sao positive (vs 31% negative, 20% mixed).",
        "H0":  "pi_positive = 0.33 (proporcao aleatoria entre 3 categorias)",
        "H1":  "pi_positive != 0.33 (sentimento positivo nao e aleatorio)",
        "parametro": "pi_positive",
        "teste_sugerido": "Teste z de proporcao (one-sample) ou Binomial test",
        "tipo": "two-tailed",
        "alpha_sugerido": 0.05,
        "Type_I_no_contexto":
            "Afirmar que ha viuncia para positividade quando e aleatorio - "
            "narrativa enviesada sobre a recepcao do finale.",
        "Type_II_no_contexto":
            "Nao detectar uma tendencia real de positividade - subestimar a "
            "recepcao do finale.",
    },
    {
        "id": "H4",
        "categoria": "Bivariada (associacao)",
        "variavel": "intensity_imputed x engagement_score_v2",
        "achado_EDA":
            "Scatter mostra padrao linear forte. r_p=+0.80, rho=+0.82, tau=+0.71 "
            "(todos 'strong' na escala da Lecture 7).",
        "H0":  "rho = 0 (sem associacao entre intensidade e engajamento na populacao)",
        "H1":  "rho != 0 (existe associacao linear)",
        "parametro": "rho (coef. correlacao populacional)",
        "teste_sugerido": "Pearson r test (assume normalidade) OU Spearman/Kendall",
        "tipo": "two-tailed",
        "alpha_sugerido": 0.05,
        "Type_I_no_contexto":
            "Concluir que intensidade associa com engagement quando e ruido - "
            "feature spuria seria usada em modelos.",
        "Type_II_no_contexto":
            "Falhar em detectar uma associacao real - perderiamos a feature mais "
            "informativa do dataset.",
    },
    {
        "id": "H5",
        "categoria": "Bivariada (associacao nula)",
        "variavel": "summary_length x engagement_score_v2",
        "achado_EDA":
            "Correlacao GLOBAL praticamente nula: r=+0.04, rho=+0.04, tau=+0.03. "
            "MAS condicional varia: +0.36 em 'mixed', -0.01 em 'positive'.",
        "H0":  "rho = 0 globalmente (tamanho do resumo nao se associa com engagement)",
        "H1":  "rho != 0 (existe associacao - pode ser mediada por sentimento)",
        "parametro": "rho (global) e rho_g (condicional por grupo g)",
        "teste_sugerido": "Spearman (global) + Spearman por grupo de sentiment",
        "tipo": "two-tailed",
        "alpha_sugerido": 0.05,
        "Type_I_no_contexto":
            "Afirmar relacao espuria - paradoxo de Simpson sugere cautela aqui.",
        "Type_II_no_contexto":
            "Nao detectar a relacao real dentro dos grupos - perda de feature interativa.",
    },
    {
        "id": "H6",
        "categoria": "Comparacao de grupos",
        "variavel": "engagement_score_v2 por sentiment_clean",
        "achado_EDA":
            "Medias: pos=4.50, neg=4.30, mixed=3.70. Boxplots distintos. "
            "Kruskal-Wallis observado: H=41.83, p~8e-10 (Lecture 7).",
        "H0":  "F_pos = F_neg = F_mixed (mesma distribuicao de engagement nos "
               "3 grupos de sentimento)",
        "H1":  "Pelo menos um grupo tem distribuicao diferente",
        "parametro": "F_g (distribuicao por grupo g de sentimento)",
        "teste_sugerido": "Kruskal-Wallis (nao-parametrico; nao assume normalidade)",
        "tipo": "right-tailed (H sempre positivo)",
        "alpha_sugerido": 0.05,
        "Type_I_no_contexto":
            "Afirmar que sentimento separa engagement quando e ruido - "
            "modelaria-se um mediador falso.",
        "Type_II_no_contexto":
            "Tratar sentimento como extraneous quando e intervening - "
            "perda do principal achado da Lecture 7.",
    },
    {
        "id": "H7",
        "categoria": "Comparacao de grupos (extraneous)",
        "variavel": "engagement_score_v2 por _source_file",
        "achado_EDA":
            "Boxplots por snapshot similares. Kruskal-Wallis: H=5.06, p=0.08 (ns). "
            "Sugere que snapshot e variavel extraneous, nao intervening.",
        "H0":  "F_day1 = F_day3 = F_week1 (snapshot nao muda distribuicao)",
        "H1":  "Pelo menos uma janela difere",
        "parametro": "F_g (distribuicao por snapshot)",
        "teste_sugerido": "Kruskal-Wallis",
        "tipo": "right-tailed",
        "alpha_sugerido": 0.05,
        "Type_I_no_contexto":
            "Concluir que o tempo de crawl afeta engagement quando nao afeta - "
            "estratificacao desnecessaria.",
        "Type_II_no_contexto":
            "Ignorar viuncia temporal real - mascarar mudanca de narrativa "
            "ao longo da semana.",
    },
]

# Tabela achatada
hyp_df = pd.DataFrame(hypotheses)
out_csv = out_dir / "hypotheses_table.csv"
hyp_df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\nTabela com {len(hyp_df)} hipoteses salva em: {out_csv}\n")

for h in hypotheses:
    print("-" * 78)
    print(f"[{h['id']}] {h['categoria']}  |  variavel(eis): {h['variavel']}")
    print(f"  Achado EDA: {h['achado_EDA']}")
    print(f"  H0: {h['H0']}")
    print(f"  H1: {h['H1']}")
    print(f"  Parametro: {h['parametro']}")
    print(f"  Teste sugerido: {h['teste_sugerido']}   ({h['tipo']}, alpha={h['alpha_sugerido']})")
    print(f"  Type I (no contexto):  {h['Type_I_no_contexto']}")
    print(f"  Type II (no contexto): {h['Type_II_no_contexto']}")

caveats = [
    "Significancia estatistica != relevancia pratica. r=+0.80 com n=187 e "
    "altamente significativo, mas o efeito pratico depende do dominio.",
    "Falhar em rejeitar H0 != provar H0. Se H7 (snapshot) der p>0.05, NAO "
    "afirmamos que os snapshots sao iguais - apenas nao temos evidencia.",
    "Associacao significante != causalidade. r(intensity, engagement) pode "
    "refletir confounders (tipo de post, hora do crawl, vies da LLM rotuladora).",
    "Multiplas comparacoes inflam erro Type I. Com 7 testes a alpha=0.05, "
    "prob(>=1 falso positivo) ~30%. Considerar correcao Bonferroni (alpha/k=0.007).",
    "n=187 e amostra modesta. Para H4 (correlacao), tem poder; para H7 "
    "(igualdade de 3 grupos), pode ter poder limitado para detectar efeitos pequenos.",
]
for i, c in enumerate(caveats, 1):
    print(f"  {i}. {c}")

fig, ax = plt.subplots(figsize=(11, 6))

x = np.linspace(-4, 4, 500)
y = scistats.norm.pdf(x)
alpha = 0.05
crit_lo = scistats.norm.ppf(alpha / 2)
crit_hi = scistats.norm.ppf(1 - alpha / 2)

ax.plot(x, y, color="#378ADD", linewidth=2.2, label="Distribuicao sob H0 (Normal padrao)")
ax.fill_between(x, 0, y, where=(x <= crit_lo),
                color="#E24B4A", alpha=0.35, label=f"Regiao critica α/2 = {alpha/2:.3f}")
ax.fill_between(x, 0, y, where=(x >= crit_hi),
                color="#E24B4A", alpha=0.35)
ax.fill_between(x, 0, y, where=(x > crit_lo) & (x < crit_hi),
                color="#7F77DD", alpha=0.18, label="Regiao nao-critica (1 - α)")

ax.axvline(crit_lo, color="#a02020", linewidth=1.4, linestyle="--",
           label=f"Valores criticos +-{abs(crit_lo):.2f}")
ax.axvline(crit_hi, color="#a02020", linewidth=1.4, linestyle="--")

r_intensity_eng = df["intensity_imputed"].corr(df["engagement_score_v2"])
z_h4 = np.arctanh(r_intensity_eng) * np.sqrt(n_sample - 3)

r_summary_eng = df["summary_length"].corr(df["engagement_score_v2"])
z_h5 = np.arctanh(r_summary_eng) * np.sqrt(n_sample - 3)

for z_obs, label, color in [
    (z_h4, f"H4: z={z_h4:.1f} (rejeita H0)", "#149945"),
    (z_h5, f"H5: z={z_h5:.2f} (nao rejeita H0)", "#BA7517"),
]:
    z_clip = np.clip(z_obs, -3.95, 3.95)
    ax.axvline(z_clip, color=color, linewidth=2.0, linestyle=":",
               label=label, alpha=0.9)
    ax.annotate("", xy=(z_clip, 0.02), xytext=(z_clip, 0.06),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

ax.set_xlabel("Estatistica de teste (z-equivalente)", fontsize=10)
ax.set_ylabel("Densidade de probabilidade sob H0", fontsize=10)
ax.set_title(
    "Decision Rule: Two-tailed Test  (Lecture 8, slide 22)\n"
    f"alpha = {alpha} -> critical values +/-{abs(crit_lo):.2f}",
    fontsize=12, fontweight="bold", pad=10,
)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax.grid(color="#e8e8e8", linewidth=0.7)
ax.set_axisbelow(True)
ax.set_ylim(-0.005, 0.45)

img_path = img_dir / "two_tailed_test.png"
plt.tight_layout()
plt.savefig(img_path, dpi=150, bbox_inches="tight")
fig2, (ax_pop, ax_arrow, ax_sample) = plt.subplots(1, 3, figsize=(15, 6),
                                                    gridspec_kw={"width_ratios": [3, 1, 1.5]})

rng = np.random.default_rng(42)
n_pop = 5000
pop_x = rng.normal(0, 1, n_pop)
pop_y = rng.normal(0, 1, n_pop)
ax_pop.scatter(pop_x, pop_y, c="#378ADD", alpha=0.18, s=18, edgecolors="none")
ax_pop.set_title(f"Populacao alvo\n(todos os posts sobre o finale; ~?)", fontsize=11, fontweight="bold")
ax_pop.set_xticks([]); ax_pop.set_yticks([])
ax_pop.set_xlim(-3.5, 3.5); ax_pop.set_ylim(-3.5, 3.5)
for spine in ax_pop.spines.values():
    spine.set_edgecolor("#999")
ax_pop.text(0, -3.2, "Parametros (desconhecidos): μ, σ, ρ, π",
            ha="center", fontsize=9, color="#444", style="italic")

ax_arrow.axis("off")
ax_arrow.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                  xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", lw=3, color="#444"))
ax_arrow.text(0.5, 0.6, "Sampling", ha="center", fontsize=11,
              fontweight="bold", color="#444")
ax_arrow.text(0.5, 0.4, "(Reddit RSS,\n3 snapshots)", ha="center", fontsize=8,
              color="#666", style="italic")

sample_idx = rng.choice(n_pop, n_sample, replace=False)
ax_sample.scatter(pop_x[sample_idx], pop_y[sample_idx],
                  c="#E24B4A", alpha=0.75, s=35, edgecolors="white", linewidths=0.4)
ax_sample.set_title(f"Amostra observada\nn = {n_sample}", fontsize=11, fontweight="bold")
ax_sample.set_xticks([]); ax_sample.set_yticks([])
ax_sample.set_xlim(-3.5, 3.5); ax_sample.set_ylim(-3.5, 3.5)
for spine in ax_sample.spines.values():
    spine.set_edgecolor("#999")
ax_sample.text(0, -3.2, "Estatisticas (calculaveis): x̄, s, r, p̂",
               ha="center", fontsize=9, color="#444", style="italic")

fig2.text(0.5, 0.02, "Inferencia estatistica  <---  (de cima: amostragem;  de baixo: generalizacao via H0/H1)",
          ha="center", fontsize=10, color="#a02020", fontweight="bold")

fig2.suptitle("Da Amostra para a Populacao (Lecture 8, slides 13-15)",
              fontsize=13, fontweight="bold", y=0.98)
img_path2 = img_dir / "sample_vs_population.png"
plt.tight_layout()
plt.savefig(img_path2, dpi=150, bbox_inches="tight")
print(f"Diagrama populacao x amostra salvo em: {img_path2}")

plt.show()
