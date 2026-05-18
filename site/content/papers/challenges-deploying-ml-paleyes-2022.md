---
title: "Challenges in Deploying ML (Paleyes)"
weight: 197
math: true
---

{{< paper-card
    title="Challenges in Deploying Machine Learning: a Survey of Case Studies"
    authors="Paleyes, Urma, Lawrence"
    year="2022"
    venue="ACM Computing Surveys"
    pdf="/papers/challenges-deploying-ml-paleyes-2022.pdf"
    arxiv="2011.09926" >}}
Survey academico riguroso (159 referencias, 29 paginas) del Cambridge ML Group liderado por **Neil Lawrence**. Mapea sistematicamente los **challenges reales** que aparecen en case studies industriales publicados al **workflow de deployment ML** (Data management → Model learning → Model verification → Model deployment + cross-cutting: ethics, law, trust, security). Aporta el **mapa de los problemas** que las herramientas MLOps intentan resolver.
{{< /paper-card >}}

---

## Contexto

Para 2020-2022 ya se sabia que desplegar ML era dificil — [Sculley 2015](/papers/hidden-technical-debt-sculley-2015) lo habia gritado siete anos antes. Lo que faltaba era una **revision sistematica academica** que recolectara case studies industriales publicados, los mapeara a etapas del workflow, y sirviera de research agenda.

El paper viene del **Cambridge Machine Learning Group**. Bias academico-British-pragmatic, no Google/IBM. Cita previas que motivan:

- **McKinsey (2019)**: ML crece ~25% YoY en adopcion empresarial.
- **Algorithmia (2019)**: mayoria de empresas toma 8-90 dias para deployar un modelo; 18% mas.
- **IDC**: porcion significativa de deployments ML **fallan**.

## Ideas principales

### Workflow adoptado

Cuatro etapas + cross-cutting aspects, basado en Ashmore et al. 2019:

```
Data management → Model learning → Model verification → Model deployment
                                                                ↓
   Cross-cutting (afectan TODAS las etapas):
   Ethics + Law + End-users' trust + Security
```

### Tabla 1 — el mapa maestro

Cada sub-paso del workflow tiene challenges identificados en case studies:

| Stage | Step | Challenges |
|---|---|---|
| Data management | Collection | Data discovery |
| | Preprocessing | Data dispersion, cleaning |
| | Augmentation | Labeling volume, expert access, low-variance data |
| | Analysis | Data profiling |
| Model learning | Selection | Complexity, resource constraints, interpretability |
| | Training | Computational cost, environmental impact, privacy |
| | Hyper-param | Resource-heavy, unknown search space |
| Model verification | Requirements | Performance vs business metrics |
| | Formal | Regulatory frameworks |
| | Test-based | Simulation-based, data validation, edge cases |
| Model deployment | Integration | Operational support, reuse, SE anti-patterns |
| | Monitoring | Feedback loops, outlier detection |
| | Updating | Concept drift, continuous delivery |
| Cross-cutting | Ethics | Aggravation of biases, fairness, authorship |
| | Law | Regulations, GDPR, FDA |
| | End users' trust | UX, explainability score |
| | Security | Data poisoning, model stealing, model inversion |

### Case studies memorables

| Case | Leccion |
|---|---|
| **AirBnB search** (Haldar et al. 2019) | Complexity prematura mata. Comenzar con NN simple de 1 layer hidden |
| **Europa Clipper spacecraft** (Wagstaff et al. 2019) | Hardware constraints fuerzan PCA + DT, no DL |
| **Booking.com 150 modelos** (Bernardi et al. 2019) | **Performance metric ≠ business value**. Clicks no convierten a conversiones |
| **BERT** (Sharir et al. 2020) | **USD 50k - 1.6M** por full training cycle |
| **Strubell 2019** | NAS training = CO₂ de 4 autos en vida util |
| **ISS Cognitive Engine** (Hackett et al. 2018) | Simulacion nunca reemplaza real-world testing safety-critical |
| **Pinterest universal embeddings** (Zhai et al. 2019) | Reuse de modelos paga dividendos |
| **Microsoft Tay** (2016) | Feedback loops abiertos = data poisoning. Bajado en 16 horas |
| **Tramèr 2016** | **Model stealing** en 70-2088 s con 650-4013 queries |
| **Buolamwini-Gebru "Gender Shades"** (2018) | Bias en datasets se manifiesta como inequidad social. Origen subfield Fairness in ML |
| **Sepsis Watch** (Sendak et al. 2020) | **Trust = communication + early engagement**, no solo interpretability |
| **"Brilliant Doctor"** (Wang et al. 2021) | UX context-aware esencial para adopcion |

### Cross-cutting destacado: Security

Tres ataques canonicos sobre modelos deployed:

1. **Data poisoning** — corromper integrity del training data. Caso medical: 8% poisoning rate → dosage incorrecta para mitad de pacientes.
2. **Model stealing** — reverse engineering via queries a la API. Tramèr et al. 2016 replicaron logistic regression, decision trees, SVMs, NNs de Google/Amazon/Microsoft.
3. **Model inversion** — recuperar partes del training set explotando confidence values. Critico GDPR.

### Potenciales soluciones

**Tools and services** (incluye dependencias propias):
- AWS SageMaker, Microsoft ML, Uber Michelangelo, TensorFlow TFX, MLflow.
- Especificas: Jenga (robustness), CheckList (NLP testing), Snorkel (weak supervision), TPOT (AutoML), Alibi Detect (drift).

**Holistic approaches:**
- **MLTRL (ML Technology Readiness Levels)** — Lavin et al. 2021.
- **Datasheets for Datasets** — Gebru et al. 2018.
- **Model Cards** — Mitchell et al. 2019.
- **Data Oriented Architectures (DOA)** — Lawrence 2019.

## Resultados experimentales

El paper no presenta experimentos propios. Su rigor viene de:

- **Solo case studies publicados** (ultimos 5 anos: 2016-2021).
- **159 referencias** cuidadosamente seleccionadas y categorizadas.
- **Tres tipos de papers:** case studies, review papers, "lessons learned".
- **Mapeo formal** challenges → workflow steps.

## Limitaciones reconocibles

- **Solo case studies publicados** — sesgo de que empresas publican. Failures muchas veces no se cuentan.
- **No incluye interviews propias** (mencionan como future work).
- **Survey vs solucion** — identifica problemas, no provee recetas detalladas.
- **Pre-LLM-era** — concept drift en fine-tuning de LLMs no se aborda.
- **Geographic bias** — case studies dominados por US/UK/Europa.

## Por que importa hoy

Es la **referencia academica obligada** para entender los **problemas reales** del ML productivo. Cited by:

- **Cursos universitarios** de ML production (MIT 6.S965, Stanford CS329S, CMU 10-718).
- **Chip Huyen, "Designing ML Systems"** (O'Reilly 2022) lo cita constantemente.
- [Kreuzberger 2023](/papers/mlops-overview-kreuzberger-2023) como antecedente directo.

El estilo Cambridge ML Group (riguroso, holistic, sociotecnico) influencio el discurso academico de **data-centric AI** liderado por Andrew Ng (2021+).

## Notas y enlaces

- **Venue:** ACM Computing Surveys (CSUR), Vol. 55, No. 6, Article 114, enero 2022.
- **DOI:** 10.1145/3533378.
- **arXiv:** 2011.09926 (v3, mayo 2022).
- **Autores:**
  - **Andrei Paleyes** — Cambridge.
  - **Raoul-Gabriel Urma** — Cambridge Spark.
  - **Neil D. Lawrence** — Cambridge, ex-director ML en Amazon, figura historica del ML probabilistico.

## Conexion con el diplomado

Este paper aporta lo que el PDF del prof Javier Rojas (clase 19) **omite por enfoque tooling-centrico**: el mapa de los **problemas reales** que las herramientas MLOps intentan resolver.

| Concepto del prof | Aporta Paleyes |
|---|---|
| 36-37 — "ejecutar datos en modelo" | Que significa "ejecutar" en prod: integration, monitoring, updating |
| 38-39 — esquema cliente-modelo | Section 6.1 (integration) — anti-patterns SE en ML |
| 40 — "uso real de usuarios" | **Cross-cutting**: ethics/law/trust/security — ausente en el prof |
| 47-49 — Cloud Run escalable | Solucion a problemas Section 4.2 + 6.2 |
| 51-54 — "¿cuando un producto IA esta terminado?" | Section 6.3 (updating) — concept drift, backward compatibility |

Conceptos criticos que el PDF omite y Paleyes aporta:

- **Business-driven metrics** (caso Booking.com): performance ≠ value.
- **Ethics & bias** (Buolamwini-Gebru, O'Neil).
- **Adversarial attacks** (poisoning, stealing, inversion).
- **Backward compatibility** en model updates.
- **UX context-aware** (caso Brilliant Doctor).
- **MLTRL, Datasheets, Model Cards** como holistic governance.
- **Sepsis Watch**: trust-building via comunicacion + early engagement.
- **Costo ambiental** del training de LLMs (Strubell et al.).

Lectura asociada:

- [Clase 19 - Entrenamiento, Deployment y MLOps](/clases/clase-19) — todos los case studies de Paleyes son citables en teoria/profundizacion.
- [Fundamento: MLOps](/fundamentos/mlops) — los anti-patterns Sculley + case studies Paleyes son ejemplos canonicos.
- [Sculley et al. 2015](/papers/hidden-technical-debt-sculley-2015) — el manifesto original que Paleyes confirma empiricamente.
- [Kreuzberger et al. 2023](/papers/mlops-overview-kreuzberger-2023) — la arquitectura solucion-side del problema-side que Paleyes mapea.

### El triangulo canonico

```
            Sculley 2015
         (los problemas
       desde Google ads)
              /\
             /  \
            /    \
           /      \
   Paleyes 2022   Kreuzberger 2023
   (case studies   (definicion + arch
    industriales    + roles + tools)
    transversales)
```
