# Trabalho 1 - Data Science (Optativa I)

Pipeline completo de EDA + Inferencia Estatistica sobre discussoes do Reddit
acerca do finale da temporada 5 de **Stranger Things**.

Estruturado conforme as aulas 6, 7 e 8 do Prof. Maiko Spiess.

---

## Estrutura do repositorio

```
.
|-- dataset/                     # Dados brutos (3 snapshots Reddit RSS em JSON)
|-- dataset_limpo/               # CSVs gerados pelos scripts
|-- plot_image/                  # Visualizacoes (.png)
|-- src/
|   |-- 01_pipeline/             # Lecture 6: ingestao + limpeza + Steps 1-7
|   |   |-- main.py              #   Steps 1-6: consolidacao, parsing, sparsity, info()
|   |   |-- data.py              #   Selecao de colunas finais
|   |   `-- dataset_impv.py      #   Imputacao + otimizacao de memoria (Step 7)
|   |-- 02_descriptive_stats/    # Lecture 6: estatistica descritiva
|   |   |-- stats_mean_median.py #   describe, moda, std, skewness, kurtosis
|   |   |-- frequency_analysis.py#   fi, fri%, Fi, Fri% + binning
|   |   |-- histogram.py         #   histogramas com KDE, media, mediana
|   |   `-- boxplot.py           #   boxplots com IQR e outliers
|   |-- 03_correlation/          # Lecture 7: analise bivariada/multivariada
|   |   |-- scatter_correlation.py # Pearson + Spearman + Kendall + heatmap 3x
|   |   `-- bivariate_analysis.py  # Papeis das variaveis + correlacao condicional
|   |-- 04_hypothesis/           # Lecture 8: inferencia e hipoteses
|   |   |-- hypotheses.py        #   Formulacao H0/H1 + framework inferencial
|   |   `-- hypothesis_tests.py  #   Execucao: Shapiro, Welch, ANOVA, chi-square...
|   `-- 05_feature_engineering/  # Lecture 10: feature engineering
|       `-- feature_engineering.py  # 4 operacoes: transformacao, criacao, encoding, selecao
|-- feature_engineering_checklist.md  # Checklist preenchido (Lecture 10 exercise)
|-- relatorio_trabalho1.docx     # Relatorio final
`-- trabalho 1 - Lucas Queiroz.xlsx
```

## Como executar

Os scripts usam bootstrap interno (`os.chdir` para a raiz do projeto),
entao **funcionam a partir de qualquer diretorio**. Ordem recomendada:

```bash
# 1. Preparacao de dados
python src/01_pipeline/main.py
python src/01_pipeline/data.py
python src/01_pipeline/dataset_impv.py

# 2. Estatistica descritiva
python src/02_descriptive_stats/stats_mean_median.py
python src/02_descriptive_stats/frequency_analysis.py
python src/02_descriptive_stats/histogram.py
python src/02_descriptive_stats/boxplot.py

# 3. Analise de correlacao
python src/03_correlation/scatter_correlation.py
python src/03_correlation/bivariate_analysis.py

# 4. Inferencia
python src/04_hypothesis/hypotheses.py
python src/04_hypothesis/hypothesis_tests.py

# 5. Feature engineering (Lecture 10)
python src/05_feature_engineering/feature_engineering.py
```

## Dependencias

```
pandas numpy scipy matplotlib scikit-learn
```

## Outputs gerados

**`dataset_limpo/`**
- `cleaned_dataset_enriched.csv` - dataset apos feature engineering (187 x 110)
- `cleaned_dataset_final.csv` - subset com 31 colunas selecionadas
- `cleaned_dataset_improved.csv` - apos imputacao + otimizacao de memoria
- `duplicates_dataset.csv` - duplicatas por URL detectadas no Step 3
- `frequency_tables.csv` - tabelas de frequencia (fi, fri, Fi, Fri)
- `stats_mean_median.csv` - tabela descritiva (mean, median, mode, std, skew, kurt)
- `hypotheses_table.csv` - 7 hipoteses formuladas com H0/H1/teste sugerido
- `hypothesis_tests_results.csv` - 16 testes executados com decisao + Bonferroni
- `cleaned_dataset_ml_ready.csv` - dataset final (187 x 56) apos feature engineering
- `feature_engineering_log.csv` - log estruturado das transformacoes aplicadas

**`plot_image/`**
- `histogramas.png`, `boxplots.png` - distribuicoes univariadas
- `frequency_categorical.png` - frequencias para categoricas + temas
- `scatter_plots.png` - scatter Pearson/Spearman/Kendall + classificacao de forca
- `heatmap_correlacao.png` - 3 paineis (Pearson | Spearman | Kendall)
- `boxplots_grouped.png`, `scatter_conditional.png` - analise condicional (mediador)
- `qqplots_normality.png` - QQ-plots para testes de normalidade
- `two_tailed_test.png`, `sample_vs_population.png` - diagramas conceituais
- `hypothesis_tests_summary.png` - sumario -log10(p) das 16 hipoteses
- `feature_transformations.png` - antes/depois das transformacoes + PCA
