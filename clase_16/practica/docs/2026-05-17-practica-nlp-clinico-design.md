# Práctica local clase 16 — NLP clásico sobre corpus clínico

**Fecha:** 2026-05-17
**Estado:** Diseño aprobado, pendiente plan de implementación.
**Alcance:** Bloque 1 del Práctico 16 (NLTK) aplicado a corpora clínicos en español.

---

## 1. Motivación y objetivos

Esta práctica complementa el recorrido celda a celda del Practico_16.ipynb del Diplomado IA UC. El objetivo es **llevar las técnicas del NLP clásico a un dominio aplicado** — específicamente texto clínico en español — para construir intuición sobre cómo se comportan estos métodos en corpus reales y conectarlos con el trabajo en FHIR-MDM.

### Objetivos específicos

1. **Caracterizar cuantitativamente** corpus clínicos en español usando estadísticas léxicas clásicas (Zipf, Heaps, FreqDist).
2. **Comparar tokenizadores** (Punkt español default vs entrenado custom vs inglés) en su capacidad de manejar abreviaciones clínicas.
3. **Identificar stopwords y patrones de stemming** específicos del dominio que NLTK español no captura.
4. **Demostrar un baseline NLP-clásico** para detección de PII y normalización aplicable a un pipeline MDM-FHIR.

### Lo que esta práctica NO cubre

- Modelos neurales (BERT, BETO, BERT clínico) — fuera del alcance.
- spaCy, NLLB-200, VADER, Bag-of-Words classification — son bloques posteriores del lab, se cubrirán después.
- Análisis de sentimientos médicos — no aplicable directamente al dominio.
- Anotación manual de corpus — se reutilizan annotations gold existentes.

---

## 2. Arquitectura del proyecto

### Estructura de directorios

```
clase_16/practica/
├── .venv/                          # virtualenv local (uv venv)
├── README.md
├── pyproject.toml                  # deps: nltk, datasets, pandas, matplotlib, pytest
├── _corpora.py                     # interfaz uniforme: load_meddocan(), load_cantemist(), ...
├── _stats.py                       # zipf_fit, heaps_fit, type_token_ratio, comparative_plot
├── _tokenize.py                    # wrappers Punkt/Treebank/Tweet con interfaz uniforme
├── _eval.py                        # métricas: agreement contra gold, error rates
│
├── 00_setup_env.py
├── 01_download_corpora.py
├── 02_explore_corpora.py
│
├── 10_zipf_4corpora.py
├── 11_heaps_4corpora.py
├── 12_freqdist_topk.py
├── 13_dispersion_clinical.py
├── 14_concordance_explorer.py
│
├── 20_punkt_es_vs_en.py
├── 21_tokenize_abbreviations.py
├── 22_punkt_train_custom.py
├── 23_eval_punkt_systems.py
│
├── 30_stopwords_baseline.py
├── 31_stopwords_clinical_discover.py
├── 32_stem_clinical.py
├── 33_lemma_omw_compare.py
├── 34_normalize_pipeline.py
│
├── 40_pii_baseline.py
├── 41_pii_eval_meddocan.py
├── 42_mdm_blocker_demo.py
│
├── data/                           # corpora cacheados (post 01_)
├── out/                            # figuras + CSVs generados
├── checkpoints/                    # modelos Punkt custom entrenados
└── tests/
    ├── test_corpora.py
    ├── test_stats.py
    └── test_tokenize.py
```

**Total**: ~20 scripts ejecutables + 4 helpers privados + 3 archivos de test.

### Convenciones heredadas de `clase_14/practica/`

- Scripts numerados ejecutables individualmente: `python 10_zipf_4corpora.py`.
- Helpers privados con prefijo `_`.
- pytest para tests de regresión.
- Outputs versionados en `out/` con nombre auto-generado del script.
- venv local con uv, dependencies en `pyproject.toml`.

---

## 3. Data layer

### Corpora y fuentes

| Corpus | Fuente | Tamaño | Loader |
|---|---|---|---|
| **MEDDOCAN** | HuggingFace `bigbio/meddocan` | ~1000 docs, ~250k tokens | `datasets.load_dataset()` |
| **Cantemist** | HuggingFace `bigbio/cantemist` | ~1000 docs onco, ~200k tokens | `datasets.load_dataset()` |
| **PharmaCoNER** | HuggingFace `bigbio/pharmaconer` | ~1000 docs farma, ~190k tokens | `datasets.load_dataset()` |
| **Quijote** | Symlink desde `clase_14/practica/quijote.txt` | 2.2 MB, ~400k tokens | `open()` directo |

### Flujo de descarga

`01_download_corpora.py`:
1. Para cada corpus clínico: `datasets.load_dataset("bigbio/<corpus>", trust_remote_code=True)`.
2. Cachear en `~/.cache/huggingface/datasets/`.
3. Convertir a formato uniforme `List[Doc]`.
4. Persistir como Parquet en `data/corpora/<corpus>.parquet`.
5. Para Quijote: symlink `data/corpora/quijote.txt` → `../clase_14/practica/quijote.txt`.

### Interfaz uniforme

```python
@dataclass
class Doc:
    id: str
    text: str
    source: str
    annotations: List[Entity]
    metadata: Dict[str, Any]

@dataclass
class Entity:
    start: int
    end: int
    label: str
    text: str

def load_corpus(name: str) -> List[Doc]
def list_corpora() -> List[str]
def doc_iter(name: str) -> Iterator[Doc]
```

Todos los scripts posteriores usan `load_corpus("meddocan")` sin importar cómo se almacena.

### Anotaciones gold disponibles

- **MEDDOCAN**: 21 categorías PII (`NOMBRE_SUJETO_ASISTENCIA`, `FECHA`, `HOSPITAL`, `DIREC_PACIENTE`, etc.).
- **Cantemist**: entidades `MORFOLOGIA_NEOPLASIA` con códigos CIE-O-3.
- **PharmaCoNER**: `NORMALIZABLES`, `NO_NORMALIZABLES`, `PROTEINAS`, `UNCLEAR`.

Cargadas como `List[Entity]` con `(start, end, label, text)`.

### Tamaños y licencias

- Cache HuggingFace: ~80 MB.
- `data/corpora/`: ~30 MB parquet.
- Licencia CC-BY-NC-SA-4.0 mayoritariamente. Uso académico permitido.

---

## 4. Scripts y experimentos

### Setup / Exploración (00-02)

- **00_setup_env.py** — verifica deps + `nltk.download()` mínimo. Sale 0/1.
- **01_download_corpora.py** — descarga + normaliza + persiste. Reporta totales.
- **02_explore_corpora.py** — sample docs + N/V/TTR/top entidades. Genera `out/02_summary.md`.

### Descriptivos: Zipf, Heaps, FreqDist (10-14)

- **10_zipf_4corpora.py** — `f(r) = K/r^α` por OLS log-log para los 4 corpora. Hipótesis: α universal, K difiere.
- **11_heaps_4corpora.py** — `V(N) = K·N^β`. Hipótesis: β_clínico > β_literario.
- **12_freqdist_topk.py** — top 50 lado a lado. Top 30 dominados por funcionales (Zipf); 30-50 dominio-específicos.
- **13_dispersion_clinical.py** — dispersion plot de keywords médicos.
- **14_concordance_explorer.py** — KWIC de términos de interés cross-corpus.

### Tokenización comparada (20-23)

- **20_punkt_es_vs_en.py** — Punkt español vs inglés sobre 100 docs MEDDOCAN. Cuenta divergencias.
- **21_tokenize_abbreviations.py** — 50 oraciones con `pte.`, `dx.`, `tto.`, `s/o`, `HTA`. Compara 4 tokenizadores.
- **22_punkt_train_custom.py** — `PunktTrainer` sobre 80% de cada sub-corpus. Imprime abreviaciones aprendidas.
- **23_eval_punkt_systems.py** — 5 tokenizadores × 3 corpora gold-annotated. Métrica: F1 boundary detection.

### Stop-words y stemming médico (30-34)

- **30_stopwords_baseline.py** — `stopwords.words('spanish')` aplicado a cada corpus. Reporta % filtrado.
- **31_stopwords_clinical_discover.py** — palabras de alta frecuencia clínica ausentes en NLTK español. Top 50 candidatos por TF-IDF inverso.
- **32_stem_clinical.py** — Snowball ES sobre 200 términos clínicos. Tasa de "destrucción".
- **33_lemma_omw_compare.py** — `WordNetLemmatizer` con omw-1.4 vs Snowball ES.
- **34_normalize_pipeline.py** — pipeline `load → tokenize → stopword → stem`. Reporta reducción de vocabulario por paso.

### Aplicación a MDM-FHIR (40-42)

- **40_pii_baseline.py** — heurísticas + regex + FreqDist para detectar nombres y fechas.
- **41_pii_eval_meddocan.py** — evaluación contra annotations gold. Precision/recall/F1 por categoría PII.
- **42_mdm_blocker_demo.py** — pipeline NLP → features → blocker MDM. Conecta con la arquitectura FHIR-MDM existente.

---

## 5. Helpers y testing

### Helpers privados

#### `_corpora.py`
- `load_corpus(name) -> List[Doc]`
- `list_corpora()`, `cache_dir()`, `doc_iter()`
- `Doc` y `Entity` dataclasses

#### `_stats.py`
- `zipf_fit(tokens) -> (alpha, K, r²)`
- `heaps_fit(tokens) -> (beta, K, r²)`
- `freqdist_topk(tokens, k)`, `type_token_ratio(tokens)`
- `comparative_plot()` para gráficos comparativos

#### `_tokenize.py`
Protocol `Tokenizer` con `name`, `tokenize()`, `sent_tokenize()`. Implementaciones:
- `NLTKPunktTokenizer`
- `NLTKTreebankTokenizer`
- `TweetTokenizer`
- `CustomPunktTokenizer(model_path)`
- `list_tokenizers() -> Dict[str, Tokenizer]` para iteración uniforme.

#### `_eval.py`
- `precision_recall_f1(predicted, gold, match_mode)` — exact/partial/type_only
- `sentence_boundary_f1(predicted, gold)`
- `confusion_table(predicted, gold) -> pd.DataFrame`

### Testing

Tests con pytest en `tests/`:
- **test_corpora.py** — cada corpus carga, tiene los campos esperados, conteo mínimo.
- **test_stats.py** — Zipf y Heaps sobre corpus sintético, verifica recuperar parámetros con r² ≥ 0.95.
- **test_tokenize.py** — tokenizadores no se rompen en `"U.S.A. y EE.UU."`, `"Esta es una oración. Otra."`, etc.

Smoke tests rápidos (<5s).

### Reproducibilidad

- `pyproject.toml` con versiones pinneadas (managed por uv).
- Seed fijo (42) en cualquier sampling aleatorio.
- README documenta version hash de cada corpus.
- Modelos Punkt custom versionados con timestamp.

### Tooling

- **uv** para venv + deps.
- **pytest** para tests.
- **matplotlib** para gráficos.
- **pandas** para CSVs.
- Sin Jupyter — scripts puros.

---

## 6. Outputs y reproducibilidad

### Estructura de `out/`

Cada script genera artefactos `<numero>_<nombre>.{png,csv,md,txt}` reproducibles. Permite borrar `out/` y regenerar todo con un script driver.

Lista completa (~30 artefactos generados):
- `02_summary.md`
- `10_zipf_4corpora.png`, `10_zipf_fit_params.csv`
- `11_heaps_4corpora.png`, `11_heaps_fit_params.csv`
- `12_topk_table.csv`, `12_topk_clinical_unique.md`
- `13_dispersion_<corpus>.png` × 3
- `14_concordance_<term>.txt`
- `20_punkt_es_vs_en.csv`
- `21_abbrev_accuracy.csv`
- `22_punkt_<corpus>_learned.txt` × 3
- `23_punkt_eval_table.csv`
- `30_stopwords_baseline.csv`
- `31_stopwords_clinical_candidates.csv`
- `32_stem_quality.csv`
- `33_lemma_omw_compare.csv`
- `34_pipeline_reduction.md`
- `40_pii_predictions.csv`
- `41_pii_eval.csv`
- `42_mdm_demo.md`
- `README.md` — índice maestro con hallazgos.

### README final

1. Resumen ejecutivo (3 párrafos).
2. Hallazgos clave (5-7 bullets cuantitativos).
3. Cómo ejecutar (paso a paso).
4. Estructura de scripts.
5. Limitaciones.
6. Próximos pasos.

---

## 7. Estimación y riesgos

### Tiempo

| Bloque | Sesiones |
|---|---|
| Setup + descarga datos | 1 (~2-3 h) |
| Descriptivos (10-14) | 1 (~2 h) |
| Tokenización (20-23) | 1-2 (~3-4 h) |
| Stopwords / stem (30-34) | 1 (~2 h) |
| MDM (40-42) | 1-2 (~3-4 h) |
| **Total** | **5-6 sesiones** |

### Riesgos identificados

| Riesgo | Mitigación |
|---|---|
| Datasets BigBio requieren `trust_remote_code=True` o tienen requisitos extra | Plan B: descarga manual desde fuentes originales (Plan TL del Gobierno Español) |
| WordNet español (OMW) tiene cobertura pobre para vocabulario clínico | Esperable — parte del hallazgo. Documentar cuantitativamente |
| Punkt custom mejora marginalmente sobre Punkt español default | Reportar honestamente — el método es válido aunque la mejora sea pequeña |
| MEDDOCAN annotations en BRAT son verbosos de parsear | Parser una sola vez en `_corpora.py` con tests |

---

## 8. Siguiente paso

Aprobado el diseño:
1. Commit de este documento.
2. Invocar skill `writing-plans` para generar plan de implementación detallado.
3. Ejecutar plan en sesiones siguientes.

### Memorias relacionadas

- `feedback_lab_walkthrough_strategy` — playbook del recorrido del Colab.
- `project_fhir_mdm_architecture` — arquitectura FHIR-MDM (target de aplicación).
- `user_fhir_expert` — perfil del usuario (FHIR + Go).
- `feedback_no_argentinian_spanish` — usar tuteo neutro/chileno en docs y código.
- `feedback_papers_download_analyze` — descargar y analizar papers cuando se mencionan.
- `feedback_complete_research` — investigar exhaustivamente.
