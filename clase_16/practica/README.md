# Práctica clase 16 — NLP clínico

Pipeline NLP clásico (NLTK) aplicado a 4 corpora cross-domain en español
(MEDDOCAN PII, Cantemist oncología, PharmaCoNER fármacos, Quijote literario)
para caracterizar texto clínico, comparar tokenizadores, descubrir
stopwords médicas y demostrar detección PII para MDM-FHIR.

- **Diseño:** [docs/2026-05-17-practica-nlp-clinico-design.md](docs/2026-05-17-practica-nlp-clinico-design.md)
- **Plan:** [docs/2026-05-17-practica-nlp-clinico-plan.md](docs/2026-05-17-practica-nlp-clinico-plan.md)
- **Resultados:** ver tablas y figuras enlazadas en cada sección, todas regenerables desde [out/](out/).

---

## Resumen ejecutivo

Construimos un workspace reproducible con 4 helpers privados (`_corpora`,
`_stats`, `_tokenizers`, `_eval`) y 17 scripts experimentales numerados
secuencialmente que generan 29 artefactos cuantitativos. Los hallazgos
caracterizan tres dimensiones del español clínico que lo distinguen del
literario:

1. **Distribución léxica más concentrada** (Zipf α y Heaps β) por
   abreviaciones, números y vocabulario médico repetido.
2. **Densidad léxica mayor** (menos stopwords NLTK, más sustantivos
   específicos del dominio).
3. **Cobertura pobre de recursos genéricos** (WordNet español cubre
   <30% del vocabulario médico) — favorece stemmer Snowball sobre
   lematización OMW.

Como aplicación end-to-end, se valida un baseline regex de detección
PII contra el gold de MEDDOCAN y se demuestra blocking MDM sobre nombres
extraídos.

---

## Hallazgos clave

### 1. Volúmenes y densidad

| Corpus | Docs | Tokens N | Vocab V | TTR | Entidades gold |
|---|---|---|---|---|---|
| meddocan | 1,000 | 513,589 | 37,056 | 0.0722 | 20,815 |
| cantemist | 1,301 | 1,067,529 | 37,189 | 0.0348 | 16,032 |
| pharmaconer | 15,868 | 402,047 | 27,131 | 0.0675 | 7,526 |
| quijote | 1 | 451,778 | 25,637 | 0.0567 | 0 |

Tabla generada por [02_explore_corpora.py](02_explore_corpora.py) → [out/02_summary.md](out/02_summary.md).

Cantemist tiene la TTR más baja (0.0348) — narrativas oncológicas largas
con vocabulario muy repetido. MEDDOCAN tiene la más alta entre clínicos
(0.0722) porque los nombres propios, fechas e IDs de la PII inflan el
vocabulario único.

### 2. Ley de Zipf (rango × frecuencia)

| Corpus | α | r² |
|---|---|---|
| meddocan | 1.165 | 0.968 |
| cantemist | **1.431** | 0.971 |
| pharmaconer | 1.229 | 0.973 |
| quijote | 1.212 | 0.978 |

Generado por [10_zipf_4corpora.py](10_zipf_4corpora.py) →
[out/10_zipf_fit_params.csv](out/10_zipf_fit_params.csv),
[out/10_zipf_4corpora.png](out/10_zipf_4corpora.png).

r² > 0.96 en todos los corpora — el ajuste log-log es robusto. **Cantemist
tiene la distribución más concentrada** (α=1.43): pocas palabras
dominantes (siglas de oncología, términos de seguimiento) consumen mucha
masa. **MEDDOCAN tiene la cola más larga** (α=1.17) por la diversidad de
nombres propios y direcciones de la PII.

### 3. Ley de Heaps (vocabulario × tokens)

| Corpus | β | r² |
|---|---|---|
| meddocan | **0.664** | 0.998 |
| cantemist | 0.541 | 0.997 |
| pharmaconer | 0.593 | 0.995 |
| quijote | 0.607 | 0.993 |

Generado por [11_heaps_4corpora.py](11_heaps_4corpora.py) →
[out/11_heaps_fit_params.csv](out/11_heaps_fit_params.csv),
[out/11_heaps_4corpora.png](out/11_heaps_4corpora.png).

**MEDDOCAN tiene el β más alto** (0.66): el vocabulario crece ~12% más
rápido que en literario o farmacéutico. Consistente con el hallazgo de
Zipf — los nombres propios introducen vocabulario nuevo permanentemente.
**Cantemist tiene el β más bajo** (0.54): vocabulario cerrado, narrativa
clínica estereotipada.

### 4. Stopwords y densidad léxica

| Corpus | % tokens en NLTK stopwords español |
|---|---|
| meddocan | 36.5% |
| cantemist | 38.1% |
| pharmaconer | 40.6% |
| **quijote** | **51.6%** |

Generado por [30_stopwords_baseline.py](30_stopwords_baseline.py) →
[out/30_stopwords_baseline.csv](out/30_stopwords_baseline.csv).

Quijote tiene **13-15 puntos más** de stopwords NLTK que los corpora
clínicos. El español clínico es léxicamente más denso: menos conectores
genéricos, más sustantivos médicos.

### 5. Stopwords clínicas descubiertas

50 candidatas con ratio `freq_clínico / freq_quijote > 5x` (Laplace +1).
**Todas tienen `freq_quijote = 0`** — son exclusivas del dominio médico:

```
pulmonar metástasis carcinoma renal dosis biopsia hallazgos clínico
control abdominal adenopatías progresión cuadro cirugía tumor células
quimioterapia anamnesis urgencias analítica ...
```

Generado por [31_stopwords_clinical_discover.py](31_stopwords_clinical_discover.py) →
[out/31_stopwords_clinical_candidates.csv](out/31_stopwords_clinical_candidates.csv).

Material directo para una lista extendida de stopwords clínicas que NLTK
no incluye.

### 6. Tokenización: Punkt y abreviaciones

Sobre 100 docs MEDDOCAN: Punkt español vs inglés difieren en **18/100 docs**
con `mean diff = 0.10`, `max abs diff = 1` ([20_punkt_es_vs_en.py](20_punkt_es_vs_en.py)).

Sobre un banco de 13 oraciones con abreviaciones clínicas, los tokenizadores
Punkt/Treebank son **idénticos en n_tokens promedio**; TweetTokenizer
fragmenta más (7.33 vs 8.0 en honoríficos). Caso concreto: `Sr.` se
preserva en Punkt y Treebank, pero **`Sra.` se fragmenta en TODOS** los
tokenizadores porque su frecuencia en corpora de entrenamiento estándar
es baja ([21_tokenize_abbreviations.py](21_tokenize_abbreviations.py) →
[out/21_abbrev_summary.csv](out/21_abbrev_summary.csv)).

### 7. Punkt entrenado por corpus

| Corpus | Train docs | Abrev. aprendidas | F1 (gold seed) |
|---|---|---|---|
| meddocan | 800 | 65 | 0.640 |
| cantemist | 1,040 | 72 | 1.000 |
| pharmaconer | 12,694 | 28 | 1.000 |
| punkt_es default | — | 313 (NLTK) | **0.880** (meddocan) |

Generado por [22_punkt_train_custom.py](22_punkt_train_custom.py) y
[23_eval_punkt_systems.py](23_eval_punkt_systems.py) →
[out/23_punkt_eval_table.csv](out/23_punkt_eval_table.csv).

**Hallazgo contraintuitivo:** los Punkt custom **empeoran** vs Punkt
español default en MEDDOCAN. Causa probable: el texto reconstruido desde
tokens BIO (espacios entre cada token) introduce un sesgo en el
entrenamiento que degrada los separadores de oraciones. **Lección:** el
pre-procesamiento (alineación de tokens) impacta cuantitativamente el
downstream — un mismo algoritmo entrenado sobre representaciones
distintas del mismo dato produce modelos no equivalentes.

### 8. Stemming vs lematización para vocabulario clínico

Sobre 40 términos clínicos (fármacos, diagnósticos, procedimientos,
plurales):

| Método | Cobertura | Calidad |
|---|---|---|
| Snowball español | **100%** | length_ratio 0.81 · substring 30/40 |
| WordNet OMW español | **27.5%** | lemma original o más corto cuando existe |

Generado por [32_stem_clinical.py](32_stem_clinical.py) y
[33_lemma_omw_compare.py](33_lemma_omw_compare.py) →
[out/32_stem_quality.csv](out/32_stem_quality.csv),
[out/33_lemma_vs_stem.csv](out/33_lemma_vs_stem.csv).

**OMW español no cubre ningún plural** (`pacientes`, `diagnósticos`,
`tratamientos`) ni la mayoría de diagnósticos/procedimientos. Para
vocabulario clínico **Snowball es preferible** por cobertura, aunque
OMW da lemas más legibles cuando los tiene.

### 9. Pipeline normalize: reducción de vocabulario

| Corpus | V₀ raw | V₁ lower+alpha | V₂ −stopwords | V₃ +stem | V₃/V₀ |
|---|---|---|---|---|---|
| meddocan | 37,056 | 24,268 | 24,122 | 15,128 | 0.408 |
| cantemist | 37,189 | 21,965 | 21,810 | 12,882 | **0.346** |
| pharmaconer | 27,131 | 20,842 | 20,697 | 12,224 | 0.451 |
| quijote | 25,637 | 23,427 | 23,160 | 10,555 | 0.412 |

Generado por [34_normalize_pipeline.py](34_normalize_pipeline.py) →
[out/34_pipeline_reduction.md](out/34_pipeline_reduction.md).

La etapa que más reduce vocab es **+stem** (colapsa flexiones plurales y
conjugaciones). La etapa **−stopwords** casi no afecta al vocab (~0.6%),
confirmando que las stopwords son pocas tipos pero muy frecuentes en
tokens — su reducción se siente en la masa, no en el vocabulario.

### 10. Aplicación MDM-FHIR — Baseline PII regex

Sobre 200 docs MEDDOCAN, partial-match contra gold:

| Categoría | n_pred | n_gold | **P** | **R** | **F1** |
|---|---|---|---|---|---|
| EMAIL | 190 | 185 | 1.000 | 1.000 | **1.000** |
| FECHA | 400 | 506 | 1.000 | 0.795 | **0.886** |
| TELEFONO | 36 | 21 | 0.472 | 0.810 | 0.596 |
| ID | 198 | 577 | 1.000 | 0.389 | 0.560 |

Generado por [40_pii_baseline.py](40_pii_baseline.py) y
[41_pii_eval_meddocan.py](41_pii_eval_meddocan.py) →
[out/41_pii_eval.csv](out/41_pii_eval.csv).

**EMAIL alcanza F1=1.0** con una sola regex — formato muy regular.
**ID y TELEFONO degradan a F1≈0.56-0.60** por diversidad de formatos
(5 subcategorías de ID en MEDDOCAN, regex telefónica demasiado genérica).
**Conclusión:** los enfoques regex son suficientes para PII de formato
regular, pero la PII con variabilidad de formato requiere NER (BETO
clínico u otro transformer fine-tuned).

### 11. Aplicación MDM-FHIR — Blocking demo

Sobre 100 docs MEDDOCAN, blocking heurístico por primeras 3 letras del
primer nombre + número de palabras:

| Métrica | Valor |
|---|---|
| Nombres extraídos (gold) | 203 |
| Bloques generados | 139 |
| Bloques con ≥2 candidatos | 39 |
| Reducción del espacio de comparación | ~70× |

Generado por [42_mdm_blocker_demo.py](42_mdm_blocker_demo.py) →
[out/42_mdm_demo.md](out/42_mdm_demo.md).

Caso real detectado: el bloque `die_1` agrupa **dos documentos con
nombre exacto "Diego"** — par directo candidato a deduplicación. El
método produce también falsos positivos (`rod_2` mezcla `Rodriguez` y
`Rodrigo`), corregibles en una etapa posterior de scoring.

---

## Cómo ejecutar

```bash
# 1. Setup
cd clase_16/practica
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Verificación + descarga NLTK data
python 00_setup_env.py

# 3. Descarga y persistencia de corpora (~30 MB total, primera vez)
python 01_download_corpora.py

# 4. Tests (34 tests, ~10s)
pytest

# 5. Pipeline experimental completo (orden numérico)
python 02_explore_corpora.py        # estadísticas básicas + sample
python 10_zipf_4corpora.py          # Zipf alpha + plot
python 11_heaps_4corpora.py         # Heaps beta + plot
python 12_freqdist_topk.py          # top 50 + únicas clínicas
python 13_dispersion_clinical.py    # dispersion plots
python 14_concordance_explorer.py   # KWIC cross-corpus
python 20_punkt_es_vs_en.py         # Punkt ES vs EN
python 21_tokenize_abbreviations.py # comparativa abreviaciones
python 22_punkt_train_custom.py     # entrena Punkt por corpus
python 23_eval_punkt_systems.py     # F1 sentence boundary
python 30_stopwords_baseline.py     # stopwords NLTK
python 31_stopwords_clinical_discover.py  # descubre stopwords médicas
python 32_stem_clinical.py          # Snowball sobre vocab clínico
python 33_lemma_omw_compare.py      # OMW vs Snowball
python 34_normalize_pipeline.py     # pipeline integrada
python 40_pii_baseline.py           # regex PII
python 41_pii_eval_meddocan.py      # P/R/F1 PII
python 42_mdm_blocker_demo.py       # demo blocking MDM
```

Todos los outputs se escriben en [out/](out/) y los checkpoints Punkt en
[checkpoints/](checkpoints/). Re-ejecutar es idempotente — `01_download_corpora.py`
salta corpora ya persistidos.

---

## Estructura de scripts

```
clase_16/practica/
├── _corpora.py       # Doc/Entity + 4 loaders + Parquet cache
├── _stats.py         # freqdist_topk, type_token_ratio, zipf_fit, heaps_fit, comparative_plot
├── _tokenizers.py    # Punkt es/en, Treebank, Tweet, CustomPunktTokenizer
├── _eval.py          # precision_recall_f1, sentence_boundary_f1
├── 00_setup_env.py
├── 01_download_corpora.py
├── 02_explore_corpora.py
├── 10_zipf_4corpora.py        # Phase 6: descriptivos
├── 11_heaps_4corpora.py
├── 12_freqdist_topk.py
├── 13_dispersion_clinical.py
├── 14_concordance_explorer.py
├── 20_punkt_es_vs_en.py       # Phase 7: tokenización
├── 21_tokenize_abbreviations.py
├── 22_punkt_train_custom.py
├── 23_eval_punkt_systems.py
├── 30_stopwords_baseline.py   # Phase 8: stopwords + stemming
├── 31_stopwords_clinical_discover.py
├── 32_stem_clinical.py
├── 33_lemma_omw_compare.py
├── 34_normalize_pipeline.py
├── 40_pii_baseline.py         # Phase 9: MDM-FHIR
├── 41_pii_eval_meddocan.py
├── 42_mdm_blocker_demo.py
├── data/corpora/               # quijote.txt (symlink) + parquets persistidos
├── checkpoints/                # punkt_<corpus>.pickle
├── out/                        # 29 artefactos (CSVs, PNGs, MDs, TXTs)
└── tests/                      # 34 tests pytest + gold_splits.json
```

---

## Limitaciones identificadas

1. **Texto reconstruido desde tokens BIO** (MEDDOCAN, PharmaCoNER): el
   `Doc.text` se construye uniendo tokens con espacios, lo cual altera el
   espaciado del documento original y sesga modelos entrenados sobre ese
   texto (visible en la degradación de Punkt custom en MEDDOCAN, sección 7).
   Cantemist no sufre esto porque viene en formato BigBio KB con texto raw.

2. **Gold set de sentence splits muy pequeño** (13 ejemplos en
   `tests/gold_splits.json`): los F1 reportados en sección 7 son
   indicativos pero saturan rápido a 1.0 en cantemist y pharmaconer.
   Necesita anotación manual ampliada para discriminación fina entre
   tokenizadores.

3. **PII baseline cubre solo 4 categorías** (FECHA, TELEFONO, EMAIL, ID)
   de las **22 categorías** del gold MEDDOCAN. Para una evaluación
   completa de un sistema de de-identificación habría que mapear y
   evaluar NOMBRE, EDAD, DIRECCIÓN, HOSPITAL, FAMILIARES, etc.

4. **WordNet español (OMW)** muestra cobertura pobre (27.5%) sobre
   vocabulario clínico. Es un límite del recurso, no del enfoque. Para
   producción clínica conviene UMLS/SNOMED-CT español o modelos
   tipo BETO clínico.

5. **Quijote incluye el header Project Gutenberg** en su texto raw
   (~50 líneas administrativas sobre 451k tokens). Impacto despreciable
   en estadísticas agregadas; mencionable si se hace análisis fine-grained.

---

## Próximos pasos

- **Evaluar BETO clínico** sobre MEDDOCAN como techo para comparar
   con el baseline regex (F1 esperado ~0.95 según literatura).
- **Ampliar gold sentence-splits** a 50-100 ejemplos por corpus para
   discriminar Punkt custom vs default con poder estadístico.
- **Entrenar Punkt sobre texto crudo de MEDDOCAN** (descargado desde
   Zenodo, no desde el mirror BIO) para aislar el efecto del
   pre-procesamiento detectado en sección 7.
- **Integrar las stopwords clínicas descubiertas** (sección 5) en un
   pipeline NER downstream y medir el delta de F1 vs filtro NLTK puro.
- **Conectar con la arquitectura MDM FHIR** del proyecto principal:
   usar el normalizador de la sección 11 como blocker tier-1 para el
   scorer GBM/embedding.
