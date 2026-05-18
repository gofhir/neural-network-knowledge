---
title: "MLOps: Overview, Definition, Architecture (Kreuzberger)"
weight: 196
math: true
---

{{< paper-card
    title="Machine Learning Operations (MLOps): Overview, Definition, and Architecture"
    authors="Kreuzberger, Kühl, Hirschl"
    year="2023"
    venue="IEEE Access"
    pdf="/papers/mlops-overview-kreuzberger-2023.pdf"
    arxiv="2205.02302" >}}
La **definicion academica formal de MLOps**. Mixed-method research (literature review + tool review + 8 expert interviews) que destila desde literatura y practica industrial: **9 principios + 9 componentes tecnicos + 7 roles + arquitectura end-to-end de 4 zonas**. Es el documento que el slide 56 del prof Javier Rojas (los 3 lobulos Design/Development/Operations) parafrasea simplificadamente.
{{< /paper-card >}}

---

## Contexto

Para 2022-2023 MLOps llevaba 5 anos como termino popular pero el campo carecia de:
- Definicion consensuada.
- Arquitectura de referencia.
- Lista canonica de principios, componentes y roles.

Diferentes vendors usaban "MLOps" con significados parcialmente solapados. Los autores (Karlsruhe Institute of Technology + IBM) plantean explicitamente la **research question**:

> **RQ: What is MLOps?**

El paper es academic-grade — mixed-method con theoretical sampling y saturation. Se volvio la referencia obligada para definir MLOps formalmente.

## Ideas principales

### Definicion formal

> MLOps is a paradigm, including aspects like best practices, sets of concepts, as well as a development culture when it comes to the end-to-end conceptualization, implementation, monitoring, deployment, and scalability of machine learning products. Most of all, it is an engineering practice that leverages three contributing disciplines: machine learning, software engineering (especially DevOps), and data engineering.

Cinco ingredientes:
1. **Paradigma** (no herramienta).
2. Incluye **best practices, conceptos, cultura**.
3. Es **engineering practice** que combina ML + SE (DevOps) + Data Engineering.
4. Objetivo: **productionize** ML systems, cerrando gap Dev/Ops.
5. Medios: los 9 principios.

### Los 9 principios

| ID | Principio |
|---|---|
| P1 | CI/CD automation |
| P2 | Workflow orchestration |
| P3 | Reproducibility |
| P4 | Versioning (data + modelo + codigo) |
| P5 | Collaboration |
| P6 | Continuous ML training & evaluation |
| P7 | ML metadata tracking/logging |
| P8 | Continuous monitoring |
| P9 | Feedback loops |

Los principios **no son independientes** — cada componente tecnico implementa varios.

### Los 9 componentes tecnicos

| ID | Componente | Implementa | Ejemplos |
|---|---|---|---|
| C1 | CI/CD Component | P1, P6, P9 | Jenkins, GitHub Actions |
| C2 | Source Code Repository | P4, P5 | GitHub, GitLab |
| C3 | Workflow Orchestration | P2, P3, P6 | Airflow, Kubeflow, Flyte, Vertex Pipelines |
| C4 | **Feature Store** | P3, P4 | Feast, Tecton, Vertex Feature Store |
| C5 | Model Training Infrastructure | P6 | K8s, SageMaker, Vertex Training |
| C6 | Model Registry | P3, P4 | MLflow Registry, Vertex Registry |
| C7 | ML Metadata Stores | P4, P7 | MLflow, Kubeflow ML Metadata |
| C8 | Model Serving | P1 | Triton, TorchServe, KServe, Vertex Endpoints |
| C9 | Monitoring | P8, P9 | Prometheus, Evidently, Vertex Monitoring |

### Los 7 roles

| ID | Rol | Foco |
|---|---|---|
| R1 | Business Stakeholder | goal de negocio, ROI |
| R2 | Solution Architect | arquitectura, tecnologias |
| R3 | Data Scientist | ML problem, modeling |
| R4 | Data Engineer | pipelines de data, feature store |
| R5 | Software Engineer | design patterns, producto |
| R6 | DevOps Engineer | CI/CD, orquestacion |
| R7 | **ML/MLOps Engineer** | **cross-domain** centro del diagrama de Venn |

R7 (ML/MLOps Engineer) es el **centro** del diagrama: combina aspectos de Data Scientist + Data Engineer + Software Engineer + DevOps + Backend Engineer. Su emergencia justifica la disciplina entera.

### Arquitectura end-to-end (4 zonas)

```
A. MLOps Project Initiation       → R1+R2+R3: goal, arch, ML problem, initial data
B. Feature Engineering Pipeline   → R4: transform rules → Feature Store (C4)
C. Experimentation                → R3+R5: itera, commit → CI/CD (C1)
D. Automated ML Workflow          → R6+R7: K8s + orchestration (C3) →
                                    train → Model Registry (C6) →
                                    Serving (C8) ← Monitoring (C9) feedback loop
```

28 pasos formalmente descritos en la Figura 4 del paper. Es **technology-agnostic** y sirve de blueprint para cualquier implementacion concreta.

## Resultados experimentales

El paper no presenta experimentos de modelos. Su rigor metodologico viene de:

- **Literature review:** 1.864 articulos → screening de 194 → 27 finales peer-reviewed.
- **Tool review:** 11 herramientas open-source + cloud ML services.
- **Interview study:** 8 expertos con seniority y diversidad geografica/genero, theoretical saturation.

Cada afirmacion se referencia a fuentes especificas (letras griegas α-θ para interviewees, numeros para papers).

## Limitaciones reconocibles

- **Cutoff mayo 2021** — pre-LLM boom, pre-GPT-4. No incluye LLMOps especifico.
- **Solo 8 interviews** — saturation argumentable, muestra pequena.
- **Bias hacia tooling cloud-native** — Kubernetes, Vertex, SageMaker dominan ejemplos.
- **Poco sobre fine-tuning, RLHF, prompt engineering** — emergentes para 2024 pero no centrales en 2022.
- **Definicion abstracta** — sirve para clasificar pero menos para implementar paso a paso.
- **No cuantifica ROI** — "mas MLOps maturity = mas business improvement" se asume sin medicion.

## Por que importa hoy

Es la **referencia academica obligada para definir MLOps formalmente**. Sus contribuciones se volvieron canon:

1. **Vocabulario** P1-P9, C1-C9, R1-R7 es citable.
2. **Diagrama 4 zonas** es el blueprint que la industria sigue.
3. **El rol "MLOps Engineer"** queda formalizado como cross-disciplinary.
4. **MLOps = paradigma + cultura**, no solo tooling.

Influencio:
- **Vertex AI** alineo su tooling (Pipelines = C3, Feature Store = C4, Model Registry = C6, Endpoints = C8, Model Monitoring = C9).
- **SageMaker** y **Azure ML** ofrecen mappings equivalentes.
- **LLMOps** y **FMOps** emergen como sub-disciplinas con el mismo vocabulario.

## Notas y enlaces

- **Venue:** IEEE Access 2023 (preprint arXiv 2205.02302, mayo 2022).
- **DOI:** 10.1109/ACCESS.2023.3262138.
- **Sitio comunitario:** ml-ops.org codifica visualmente la arquitectura del paper.
- **Alineados:**
  - Google Cloud MLOps maturity model (whitepaper Google 2020) — niveles 0/1/2.
  - AWS Well-Architected ML Lens.
  - Microsoft MLOps maturity model.

## Conexion con el diplomado

Este paper es la justificacion arquitectonica del **slide 56** del prof Javier Rojas (clase 19) — los 3 lobulos Design ↔ Development ↔ Operations son la Figura 4 del paper **simplificada**.

| Slide del prof | Aporte Kreuzberger |
|---|---|
| 55 — "MLOps: paradigma repetible para mantener modelos confiables y eficientes" | **Definicion literal** del paper, seccion 6 |
| 56 — diagrama 3 lobulos | Figura 4 del paper, simplificada |
| 57 — "Pipeline = flujo de componentes con I/O" | Implementa P2 + DAGs explicitos |
| 58-59 — Load Dataset / Train / Test / Deploy | Mapping a C4/C5/C6/C8 |
| 60 — Kubeflow, Airflow, Flyte | C3 ejemplos canonicos del paper |
| 61 — Azure ML, SageMaker, Vertex AI | Plataformas que integran C1-C9 |

El PDF del prof omite tres conceptos del paper:
- **Feature Store** (C4) — componente core no mencionado.
- **Model Lineage** — trazabilidad data+codigo→modelo.
- **Los 7 roles** — especialmente R7 cross-functional MLOps Engineer.

Lectura del site:

- [Clase 19 - Entrenamiento, Deployment y MLOps](/clases/clase-19) — toda la seccion 6 referencia este paper.
- [Fundamento: MLOps](/fundamentos/mlops) — los 9 principios + 9 componentes + 7 roles estan documentados ahi.
- [Sculley et al. 2015](/papers/hidden-technical-debt-sculley-2015) — el problema que Kreuzberger resuelve.
- [Paleyes et al. 2022](/papers/challenges-deploying-ml-paleyes-2022) — survey de los problemas reales.
