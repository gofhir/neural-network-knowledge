# IA para un FHIR Server en Go — Casos de Uso y Modelos Entrenables

> Análisis generado con 5 agentes paralelos especializados en CQL, UCUM, FHIRPath, Validators, IGs y Server Core.

---

## Modelos que Podemos Construir (Entrenar / Fine-Tunear)

Esta sección prioriza los modelos que **podemos entrenar nosotros mismos** — no llamadas a OpenAI — con datasets disponibles en el ecosistema FHIR/clínico.

---

### Modelo 1: Plausibilidad Clínica de Valores UCUM

**Qué hace:** Dado un valor numérico + unidad UCUM + código LOINC, predice si la medición es fisiológicamente plausible o es un error de transcripción/unidad.

**Arquitectura:** Isolation Forest o MLP pequeño (tabular)

**Features de entrada:**
```
[valor_numérico, loinc_code_embedding, ucum_canonical_unit, sexo, edad, setting]
```

**Dataset para entrenar:**
- MIMIC-IV (PhysioNet) — millones de observaciones clínicas reales con LOINC + UCUM
- Synthea — datos sintéticos masivos con distribuciones realistas
- Generar ejemplos negativos: permutar unidades entre observaciones del mismo tipo

**Por qué es valioso:** Errores de unidad (hemoglobina 150 g/dL vs g/L) son un riesgo de seguridad documentado. El UCUM engine valida dimensionalmente; el modelo detecta valores estadísticamente imposibles.

**Go integration:**
```
FHIR write path → UCUM engine (dimensional check) → ONNX model (plausibility) → DetectedIssue resource
```

**Export:** scikit-learn IsolationForest → ONNX via `sklearn-onnx` → `onnxruntime_go`

**Tamaño del modelo:** < 1 MB. Inference: < 2ms en proceso.

---

### Modelo 2: Normalización de Unidades No-UCUM

**Qué hace:** Dado un string de unidad no canónico (`"mcg"`, `"cc"`, `"K/uL"`, `"U"`), predice la expresión UCUM canónica más probable.

**Arquitectura:** Bi-encoder (embedding retrieval) + ranker

**Dos etapas:**
1. **Lookup determinístico** (Go puro): cubre ~75% de los casos con tabla fuzzy (edit distance + token overlap)
2. **Embedding similarity** (modelo): para el 25% largo: codificar la unidad ambigua y buscar el UCUM más cercano en un índice pre-construido

**Dataset para entrenar:**
- UCUM official synonyms y aliases (fuente pública)
- LOINC `UNITSREQUIRED` + `EXAMPLE_UCUM_UNITS` — miles de pares (unidad informal → UCUM)
- NLM Value Set Authority Center: unidades en ValueSets clínicos publicados
- OMOP CDM: concepto de unidad (domain = "Unit") → UCUM mapping

**Fine-tuning:** `nomic-embed-text` (Apache 2.0) o `BioBERT` como encoder base, fine-tuneado con triplets (unidad_ambigua, ucum_correcto, ucum_incorrecto).

**Go integration:**
```
Incoming Observation.valueQuantity.unit
  → Go lookup table (fast path)
  → Miss: hugot/ONNX encoder → pgvector ANN search → UCUM engine validates
  → Accept high confidence | Queue low confidence for review
```

---

### Modelo 3: Desambiguación de Unidades Ambiguas (clasificador contextual)

**Qué hace:** Para tokens conocidamente ambiguos (`"U"` = unidades de insulina vs. otros, `"T"` = Tesla vs. tabletas), predice la interpretación correcta usando contexto clínico.

**Arquitectura:** Gradient Boosted Decision Tree (XGBoost/LightGBM) — tabular features

**Features de entrada:**
```
[token_ambiguo, loinc_code, substance_rxnorm, valor_magnitud, care_setting, resource_type]
```

**Dataset para entrenar:**
- FDA FAERS (adverse event reports) — errores documentados de unidades
- MIMIC-IV medication orders — contexto de sustancia + unidad
- Ejemplos negativos: asignar aleatoriamente unidades erróneas y marcarlas

**Export:** LightGBM → ONNX via `onnxmltools` O compilar a código Go con `treelite`

**Go integration:** 100% in-process, sin sidecar. `onnxruntime_go`. < 1ms.

---

### Modelo 4: Pre-filtro de Conformidad de Perfil IG (clasificador binario rápido)

**Qué hace:** Antes de ejecutar validación completa (50-500ms), predice si un recurso FHIR pasará o fallará contra un perfil IG específico. Actúa como fast path.

**Arquitectura:** XGBoost/LightGBM sobre features binarias

**Features de entrada (por recurso):**
```
[presencia_de_cada_must-support_element (bool vector),
 membership en ValueSets requeridos (bool),
 cardinalidad de arrays vs. min/max del perfil,
 presencia de extensiones requeridas]
```

**Un modelo por perfil IG** (US Core Patient, US Core Observation, etc.)

**Dataset para entrenar:**
- Synthea genera millones de recursos; validarlos con HAPI FHIR validator
- Introducir mutaciones sistemáticas (quitar campos, usar códigos fuera de ValueSet) para generar negativos
- Output label: pass/fail + qué categoría de constraint falló

**Go integration:**
```
Write path / bulk ingest:
  resource → feature extractor (Go struct traversal) → ONNX classifier (2ms)
  → predicted pass: downstream pipeline
  → predicted fail / uncertain: full validator (expensive)
```

**Impacto en throughput:** 10x en pipelines de bulk ingest (millones de recursos de $export de EHR).

---

### Modelo 5: Master Patient Index — Similaridad entre Pacientes

**Qué hace:** Dado un par (Patient A, Patient B), predice probabilidad de que sean la misma persona. Base del MPI (Master Patient Index).

**Arquitectura:** Dos capas:
1. **Blocking** (determinístico en Go): Soundex/DoubleMetaphone sobre apellidos + rango de fecha de nacimiento. Reduce O(N²) a O(N×k) comparaciones candidatas.
2. **Similarity model** (neural): MLP o gradient boosted classifier sobre features de similitud del par

**Features para el par (A, B):**
```
[jaro_winkler(nombre), soundex_match(apellido), dob_delta_días,
 zip_match, gender_match, phone_match, mrn_system_overlap,
 diagnosis_code_jaccard_similarity]
```

**Dataset para entrenar:**
- Record linkage benchmarks: `FEBRL` (Freely Extensible Biomedical Record Linkage), `RLdata500`
- Generar sintéticos: tomar pacientes Synthea y crear duplicados con errores realistas (typos, abreviaciones, transposiciones)
- Etiquetas positivas: misma persona con variaciones; negativas: personas distintas con nombres similares

**Fine-tuning posible:** usar `sentence-transformers` para codificar nombre+apellido como embedding y entrenar con contrastive loss (SimCSE).

**Go integration:**
```
Patient create/update
  → matchr (Soundex blocking) → retrieve candidates from index
  → feature extraction (Go) → ONNX model → confidence score
  → high conf: Patient.link (seealso) automático
  → medium conf: Task resource para revisión humana
  → low conf: no acción
```

---

### Modelo 6: Detección de Anomalías en Recursos FHIR (al escribir)

**Qué hace:** Detecta recursos estadísticamente anómalos en el momento de escritura — errores de entrada, artefactos de migración, datos mal mapeados.

**Arquitectura:** Isolation Forest por tipo de recurso (Observation, MedicationRequest, etc.)

**Features (por tipo):**
```
Observation: [valor_numérico, loinc_category, tiempo_desde_anterior, componentes_presentes]
MedicationRequest: [dosis_log_normalizada, ruta_administración, frecuencia, días_supply]
Patient: [edad, nro_condiciones, nro_medicamentos, nro_encuentros_30d]
```

**Dataset para entrenar:**
- MIMIC-IV FHIR (PhysioNet) — datos reales "buenos" como distribución positiva
- Synthea — datos sintéticos adicionales
- **No se necesitan etiquetas negativas** — Isolation Forest es unsupervised; solo necesita la distribución normal

**Go integration:**
```
Pre-write middleware (sync, ~1-3ms):
  resource → feature extractor → onnxruntime_go (IsolationForest ONNX)
  → anomaly_score > threshold: escribir AuditEvent + extension en recurso + OperationOutcome warning
  → anomaly_score > critical: return 422 con OperationOutcome detallado
  → Prefer: handling=lenient header para override
```

---

### Modelo 7: Autocompletion / Ranker para FHIRPath y CQL

**Qué hace:** Dado el contexto de una expresión FHIRPath/CQL en edición (tipo de recurso, segmento escrito, tipo esperado), rankea los candidatos de completación por probabilidad.

**Arquitectura:** Small transformer fine-tuneado (GPT-2 small / 125M params) o CodeBERT distilado

**Dataset para entrenar:**
- Todo el ecosistema de IGs públicos en GitHub (`hl7.org`, `fhir.org`) contiene miles de expresiones FHIRPath
- IG publisher repos: StructureDefinition invariants, SearchParameter expressions, Questionnaire enableWhen
- eCQM library (CMS): cientos de medidas con CQL completo — corpus paralelo
- Augmentación: mutar expresiones válidas para generar variantes

**Go integration:** LSP (Language Server Protocol) server en Go — compatible con VS Code, Neovim, cualquier editor.
```
deterministic StructureDefinition graph walker (Go) → candidate list
  → ONNX ranker (small transformer) → ranked completions
```

---

### Modelo 8: Detección de Anomalías en Patrones de Acceso (Security)

**Qué hace:** Detecta patrones de acceso sospechosos en el stream de AuditEvents — exfiltración, credenciales comprometidas, insider threat.

**Arquitectura:** Isolation Forest (point anomalies) + opcional LSTM autoencoder (sequence anomalies)

**Features por sesión/usuario:**
```
[recursos_accedidos_por_minuto, diversidad_tipos_recurso,
 hora_del_día, delta_desde_último_acceso, geoip_nuevo (bool),
 recursos_por_paciente_distintos, proporción_reads_vs_writes]
```

**Dataset para entrenar:**
- Los propios logs del servidor (unsupervised — Isolation Forest)
- BETH dataset (AWS CloudTrail anomaly detection benchmark) como referencia metodológica
- No requiere etiquetas de "ataque" — modela la distribución normal y flagea desviaciones

**Go integration:**
```
AuditEvent write → in-process channel → rolling feature accumulator per user/app
  → ONNX Isolation Forest → anomaly score
  → high score: Flag resource en Practitioner + webhook a SIEM
  → critical: revocar token via introspection endpoint del auth server
```

---

## Resumen: Qué Modelo Usar Para Cada Dominio

| Dominio | Modelo entrenable | Arquitectura | Dataset principal |
|---|---|---|---|
| UCUM plausibilidad | Isolation Forest / MLP | Tabular | MIMIC-IV, Synthea |
| UCUM normalización | Bi-encoder (retrieval) | Sentence transformer | LOINC UCUM, OMOP |
| UCUM desambiguación | GBDT (XGBoost) | Tabular | FDA FAERS, MIMIC-IV |
| IG pre-validación | GBDT / XGBoost | Tabular binario | Synthea + HAPI validator |
| MPI pacientes | MLP + blocking | Tabular de similitud | FEBRL, Synthea + noise |
| Anomalía recursos | Isolation Forest | Tabular | MIMIC-IV FHIR, Synthea |
| FHIRPath/CQL ranking | Small transformer | Seq/token | IGs GitHub, eCQM library |
| Seguridad accesos | Isolation Forest | Tabular secuencial | Logs propios (unsupervised) |

---

## Casos que Requieren LLM (No Entrenables Fácilmente, Sí Integrables)

Estos son valiosos pero dependen de modelos pre-entrenados grandes (via API o Ollama local):

| Use Case | Técnica | Modelo recomendado |
|---|---|---|
| NL → CQL / FHIRPath / FHIR Search | Fine-tuning + constrained decoding | Llama 3.1 70B (Ollama) o GPT-4o |
| CQL Explanation (ELM → lenguaje natural) | Prompting estructurado sobre ELM AST | Claude 3.5 Sonnet / GPT-4o |
| Validation Error Remediation (agentic loop) | JSON Patch generation + re-validación | GPT-4o / Claude 3.5 |
| FSH Generation | Code LLM + SUSHI validation loop | DeepSeek Coder / GPT-4o |
| IG Q&A RAG | RAG (dense + BM25) | Llama 3.1 + nomic-embed-text |
| CDS Hooks AI Cards | Constrained structured output | Claude 3.5 Sonnet + guardrails |

---

## Arquitectura Recomendada en Go

```
┌────────────────────────────────────────────────────────────────┐
│                      Go FHIR Server                            │
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ FHIRPath │  │Validator │  │CQL Engine│  │UCUM Engine   │  │
│  │ Engine   │  │          │  │(ELM AST) │  │              │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │              │                │          │
│  ┌────▼──────────────▼──────────────▼────────────────▼──────┐  │
│  │              AI Middleware Layer (Go)                     │  │
│  │   onnxruntime_go  │  pgvector  │  hugot (HF tokenizers)  │  │
│  │   (tabular models)│  (ANN)     │  (embedding models)     │  │
│  └─────────────────────────┬─────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │            LLM Sidecar (opcional, async)                │   │
│  │   Ollama (Llama 3.1 cuantizado)  │  OpenAI-compatible   │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

**Librerías Go clave:**
- [`github.com/yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go) — inference ONNX in-process (modelos tabular, IsolationForest, GBDT)
- [`github.com/knights-analytics/hugot`](https://github.com/knights-analytics/hugot) — HuggingFace tokenizers + ONNX para modelos NLP/embedding
- [`github.com/sashabaranov/go-openai`](https://github.com/sashabaranov/go-openai) — cliente OpenAI-compatible para LLMs
- [`github.com/antzucaro/matchr`](https://github.com/antzucaro/matchr) — Soundex/Metaphone para blocking en MPI
- [`github.com/sony/gobreaker`](https://github.com/sony/gobreaker) — circuit breaker para llamadas a modelos externos

---

## Principios de Diseño

1. **El AI layer nunca bloquea** — toda feature AI tiene fallback determinístico
2. **UCUM engine como invariante** — todo output de AI que involucra unidades pasa por validación dimensional
3. **ELM/AST como interfaz para CQL/FHIRPath** — nunca operar sobre texto CQL crudo
4. **ONNX para modelos pequeños** — in-process, sin dependencia Python en runtime
5. **LLMs son async y cacheados** — nunca en el critical path síncrono
6. **AuditEvent por cada decisión AI** — versión del modelo, hash del input, output, confidence score
7. **Determinístico primero** — reglas cubren 70-80% de casos; AI cubre el long tail

---

## Secuencia de Construcción Recomendada

### Fase 1 — Mayor ROI, Menor Riesgo (modelos entrenables sin LLM)
1. Plausibilidad clínica UCUM (Isolation Forest sobre MIMIC-IV)
2. Normalización de unidades (bi-encoder + UCUM oracle)
3. Detección de anomalías en recursos (Isolation Forest)
4. Pre-filtro de validación IG (XGBoost sobre Synthea + validator labels)

### Fase 2 — Inteligencia Core
5. Master Patient Index (similarity model + blocking)
6. FHIRPath/CQL autocompletion ranker (small transformer fine-tuning)
7. NL → FHIR Search (LLM + search parser como oracle)
8. IG Compatibility Analysis (pgvector embeddings)

### Fase 3 — Ecosistema IG
9. FSH Generation (code LLM + SUSHI validation loop)
10. Validation Error Remediation (agentic JSON Patch loop)
11. IG Q&A RAG (hybrid retrieval)
12. Conformance Test Generation

### Track de investigación
- Dosage Intelligence (UCUM + pharmacovigilance + GBDT)
- CQL Semantic Equivalence (GNN + formal methods)
- Federated Learning sobre datos FHIR
- Cross-IG Mapping con graph embeddings

---

## Datasets Públicos Clave

| Dataset | Contenido | Uso |
|---|---|---|
| [MIMIC-IV FHIR](https://physionet.org/content/mimic-iv-fhir-demo/2.0/) | Observaciones clínicas reales con LOINC+UCUM | Plausibilidad, anomalías, MPI |
| [Synthea](https://github.com/synthetichealth/synthea) | Pacientes FHIR sintéticos masivos | Labels de validación, entrenamiento |
| [LOINC DB](https://loinc.org/downloads/) | Codes + EXAMPLE_UCUM_UNITS | Normalización de unidades |
| [OMOP CDM Unit concepts](https://athena.ohdsi.org) | Unit vocabulary → UCUM | Cross-terminology |
| [eCQM Library (CMS)](https://ecqi.healthit.gov/ecqms) | CQL medidas completas | Fine-tuning CQL models |
| [HL7 IGs en GitHub](https://github.com/HL7) | FSH + StructureDefinitions | FHIRPath corpus, FSH generation |
| [FDA FAERS](https://open.fda.gov/data/faers/) | Adverse events con medication+dose | Dosage intelligence, unit disambiguation |
| [FEBRL](https://sourceforge.net/projects/febrl/) | Benchmark record linkage | MPI similarity model |
