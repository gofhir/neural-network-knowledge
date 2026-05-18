---
title: "MLOps"
weight: 299
math: true
---

**MLOps (Machine Learning Operations)** es un **paradigma operacional** para crear, desplegar y mantener sistemas ML en produccion de forma confiable, escalable y reproducible. No es una herramienta ni un servicio: es la combinacion de **principios de ingenieria** (CI/CD, versioning, monitoring) + **cultura de equipo** + **arquitectura tecnica** que permite que un modelo no se quede como notebook sino que viva como producto.

Este fundamento sintetiza la teoria canonica de MLOps tal como la define **[Kreuzberger et al. 2023](/papers/mlops-overview-kreuzberger-2023)**, con los antecedentes de **[Sculley et al. 2015](/papers/hidden-technical-debt-sculley-2015)** y las realidades industriales documentadas en **[Paleyes et al. 2022](/papers/challenges-deploying-ml-paleyes-2022)**.

---

## 1. Por que MLOps existe

> *"developing and deploying ML systems is relatively fast and cheap, but maintaining them over time is difficult and expensive."* — Sculley et al. 2015

[Sculley et al. 2015](/papers/hidden-technical-debt-sculley-2015) identifica que en sistemas ML reales el **codigo de modelo es solo el 5%**; el otro 95% es infraestructura: data verification, feature extraction, configuration, serving infra, monitoring tools.

```
            ╔═════════════╗
            ║   ML code   ║   ← solo 5%
            ╚═════════════╝
   ┌─────────┐ ┌─────────┐ ┌──────────────┐
   │ Config  │ │ Data    │ │ Feature      │
   │         │ │ collect │ │ extraction   │
   └─────────┘ └─────────┘ └──────────────┘
   ┌─────────┐ ┌─────────┐ ┌──────────────┐
   │ Data    │ │ Process │ │ Machine      │
   │ verif.  │ │ mgmt    │ │ resource     │
   └─────────┘ └─────────┘ │ mgmt         │
   ┌─────────┐ ┌─────────┐ └──────────────┘
   │ Analysis│ │ Serving │   ┌──────────┐
   │ tools   │ │ infra   │   │Monitoring│
   └─────────┘ └─────────┘   └──────────┘
```

MLOps es el conjunto de practicas que **opera ese 95%** sistematicamente.

{{< concept-alert type="clave" >}}
**Diferencia con DevOps:** los sistemas tradicionales cambian solo en codigo. Los sistemas ML cambian en **tres ejes simultaneos: codigo + modelo + datos**. MLOps extiende DevOps para gestionar esa triple mutabilidad.
{{< /concept-alert >}}

---

## 2. Definicion formal (Kreuzberger 2023)

> **MLOps** is a paradigm, including aspects like best practices, sets of concepts, as well as a development culture when it comes to the end-to-end conceptualization, implementation, monitoring, deployment, and scalability of machine learning products.

Ingredientes:
1. Es un **paradigma**, no una herramienta.
2. Incluye **best practices, conceptos, cultura**.
3. Es **engineering practice** que combina **ML + Software Engineering (DevOps) + Data Engineering**.
4. Objetivo: **productionizar** sistemas ML, cerrando gap Dev/Ops.
5. Medios: los 9 principios siguientes.

---

## 3. Los 9 principios (Kreuzberger 2023)

| ID | Principio | Esencia |
|---|---|---|
| **P1** | CI/CD automation | Build/test/deliver/deploy automatizados |
| **P2** | Workflow orchestration | Tareas como DAG (Directed Acyclic Graph) |
| **P3** | Reproducibility | Re-ejecutar experimento y obtener mismo resultado |
| **P4** | Versioning | Versionar **datos, modelo y codigo** |
| **P5** | Collaboration | Trabajo colaborativo + reducir silos |
| **P6** | Continuous ML training & evaluation | Retraining periodico con nuevos datos |
| **P7** | ML metadata tracking/logging | Logging por task: params, metrics, lineage |
| **P8** | Continuous monitoring | Assessment data/model/code/infra/serving |
| **P9** | Feedback loops | Loops monitoring → training y experimentacion → feature engineering |

---

## 4. Los 9 componentes tecnicos

Cada componente implementa varios principios:

| Componente | Implementa | Ejemplos |
|---|---|---|
| **C1** CI/CD Component | P1, P6, P9 | Jenkins, GitHub Actions, GitLab CI |
| **C2** Source Code Repository | P4, P5 | GitHub, GitLab, Bitbucket |
| **C3** Workflow Orchestration | P2, P3, P6 | Airflow, Kubeflow, Flyte, Vertex AI Pipelines |
| **C4** **Feature Store** | P3, P4 | Feast, Tecton, Vertex Feature Store |
| **C5** Model Training Infrastructure | P6 | K8s, GCP Vertex Training, AWS SageMaker |
| **C6** Model Registry | P3, P4 | MLflow Registry, Vertex Model Registry |
| **C7** ML Metadata Store | P4, P7 | MLflow, Kubeflow ML Metadata |
| **C8** Model Serving | P1 | Triton, TorchServe, KServe, Vertex Endpoints |
| **C9** Monitoring | P8, P9 | Prometheus, Evidently, Arize, Vertex Model Monitoring |

### 4.1 Feature Store — el componente menos conocido

Una **feature store** es un repositorio central de **features ya computados** con dos backends:

- **Offline store** (BigQuery, Snowflake) — latencia normal, para training/experimentation.
- **Online store** (Redis, Bigtable) — baja latencia, para serving real-time.

Resuelve el problema **training/serving skew**: si tu feature lo computa una pipeline batch para training y otra logica para serving online, las distribuciones divergen → modelo degrada silenciosamente.

```
                ┌──────────────────────┐
   Raw data ──→ │  Feature Pipeline    │
                │  (transform + agg)   │
                └──────────────────────┘
                          │
                          ↓
                ┌──────────────────────┐
                │   Feature Store      │
                │ ┌──────┐  ┌────────┐ │
                │ │offline│  │online │ │
                │ │ store │  │ store │ │
                │ └───┬──┘  └───┬────┘ │
                └─────│─────────│──────┘
                      ↓         ↓
                  training   serving
```

---

## 5. Los 7 roles

| Rol | Foco |
|---|---|
| **R1** Business Stakeholder | Define goal de negocio, comunica ROI |
| **R2** Solution Architect | Disena arquitectura, elige tecnologias |
| **R3** Data Scientist | Traduce problema → ML problem, modeling |
| **R4** Data Engineer | Pipelines de data, ingesta a feature store |
| **R5** Software Engineer | Aplica design patterns, convierte ML en producto |
| **R6** DevOps Engineer | Bridge Dev/Ops: CI/CD, orquestacion |
| **R7** **ML/MLOps Engineer** | **Cross-domain**: Data Scientist + Data Engineer + Software Engineer + DevOps + Backend Engineer |

R7 es el rol cuya emergencia justifica la disciplina. Es el **centro** del diagrama de Venn de roles.

---

## 6. Arquitectura end-to-end (4 zonas)

La arquitectura canonica de Kreuzberger 2023 (Figura 4 del paper):

```
┌────────────────────────────────────────────────────────────────┐
│ Zona A: MLOps Project Initiation                               │
│ R1 → goal → R2 → arquitectura → R3 → ML problem               │
│ R3+R4 → identificar data + initial checks                      │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Zona B: Feature Engineering Pipeline                           │
│ B1 (Requirements): R4 define transform rules                   │
│ B2 (Build):       Pipeline → Feature Store (C4)                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Zona C: Experimentation                                        │
│ R3 itera: analysis → preparation → training → validation       │
│ R5 ayuda con engineering. R3 commits → CI/CD (C1)              │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ Zona D: Automated ML Workflow (Production)                     │
│ R6/R7 manejan K8s + workflow orchestration (C3)                │
│ Pull versioned features → train → eval → Model Registry (C6)   │
│ CI/CD (C1) → deploy → Model Serving (C8)                       │
│ Monitoring (C9) → feedback loop → triggea retraining           │
└────────────────────────────────────────────────────────────────┘
```

Detalles paso a paso (28 pasos formales) en [Kreuzberger 2023](/papers/mlops-overview-kreuzberger-2023).

---

## 7. Google MLOps maturity levels

| Nivel | Descripcion | Estado tipico |
|---|---|---|
| **0** | Manual | Jupyter → email → ingeniero deploya. La mayoria de equipos. |
| **1** | ML pipeline automation | Pipeline de training automatizado, retraining triggers. |
| **2** | CI/CD pipeline automation | Cambios de codigo → CI testea → CD redeploya. Production seria. |

Mapping a principios:
- **Nivel 0:** ninguno implementado.
- **Nivel 1:** P2 (orquestacion) + P6 (continuous training).
- **Nivel 2:** P1 + P2 + P6 + P9 (CI/CD + orquestacion + continuous training + feedback loops).

---

## 8. Drift detection — concepto central

[Sculley 2015](/papers/hidden-technical-debt-sculley-2015) lo plantea cualitativamente. La formalizacion:

### 8.1 Tres tipos

| Tipo | Que cambia | Notacion |
|---|---|---|
| **Covariate shift (data drift)** | $P(X)$ | distribucion de features cambia |
| **Concept drift** | $P(Y\|X)$ | la relacion features→label cambia |
| **Label shift** | $P(Y)$ | distribucion de labels cambia |

### 8.2 Metricas

**Population Stability Index (PSI)** — estandar industrial:

$$
\text{PSI} = \sum_{i=1}^{B} (a_i - e_i) \cdot \ln\left(\frac{a_i}{e_i}\right)
$$

Interpretacion:
- $\text{PSI} < 0.1$: sin cambio.
- $0.1 \leq \text{PSI} < 0.25$: cambio menor.
- $\text{PSI} \geq 0.25$: cambio mayor, **retraining recomendado**.

**KL divergence:**
$$
D_{\text{KL}}(P_{\text{prod}} \| P_{\text{train}}) = \sum_x P_{\text{prod}}(x) \ln \frac{P_{\text{prod}}(x)}{P_{\text{train}}(x)}
$$

**Kolmogorov-Smirnov test** para features continuas.

Codigo Python en [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) seccion IV.2.

---

## 9. Retraining triggers — 3 patrones

| Patron | Senal | Pros | Cons |
|---|---|---|---|
| **Schedule-based** | "Cada 7 dias" | Simple, predecible | Malgasta compute |
| **Performance-based** | Metrica online < umbral | Reactivo a degradacion | Requiere ground truth, lag |
| **Drift-based** | PSI/KL > umbral | Proactivo, sin ground truth | Falsos positivos |

Production seria combina los tres.

---

## 10. Stack moderno por capa

```
┌─────────────────────────────────────────────────────────────┐
│ Experiment tracking:  MLflow, W&B, Neptune, Comet, ClearML  │
├─────────────────────────────────────────────────────────────┤
│ Data versioning:      DVC, LakeFS, Pachyderm, Delta Lake    │
├─────────────────────────────────────────────────────────────┤
│ Feature store:        Feast, Tecton, Vertex Feature Store   │
├─────────────────────────────────────────────────────────────┤
│ Model registry:       MLflow Registry, Vertex Model Reg.    │
├─────────────────────────────────────────────────────────────┤
│ Orchestration:        Kubeflow, Airflow, Flyte, Prefect,    │
│                       Dagster, Argo Workflows               │
├─────────────────────────────────────────────────────────────┤
│ Serving:              Triton, TorchServe, TF Serving, KServe│
├─────────────────────────────────────────────────────────────┤
│ Monitoring:           Evidently, WhyLabs, Arize, Prometheus │
├─────────────────────────────────────────────────────────────┤
│ CI/CD:                GitHub Actions, GitLab CI, Tekton     │
└─────────────────────────────────────────────────────────────┘
```

---

## 11. Anti-patterns clasicos (Sculley 2015)

Los siete pecados capitales del ML en produccion:

1. **Glue code** — codigo "pegamento" entre frameworks que freeza arquitectura.
2. **Pipeline jungles** — scripts ad-hoc encadenados con cron y bash.
3. **Dead experimental codepaths** — branches condicionales en production code que acumulan complejidad.
4. **CACE (Changing Anything Changes Everything)** — feature, hyperparam, sampling, todo entrelazado.
5. **Undeclared consumers** — outputs del modelo consumidos por sistemas que no sabes que existen.
6. **Unstable data dependencies** — features que vienen de otros equipos que los cambian sin avisar.
7. **Configuration debt** — lineas de config superan lineas de codigo, sin code review.

Detalles en [Sculley 2015](/papers/hidden-technical-debt-sculley-2015).

---

## 12. Conexion con el resto del site

Este fundamento se cita en:

- [Clase 19 - Entrenamiento, Deployment y MLOps](/clases/clase-19) — slides 50-62.
- [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) — Parte IV.
- [Fundamento: Cloud Computing](/fundamentos/cloud-computing) — Vertex AI Pipelines.
- [Fundamento: Docker y Containers](/fundamentos/docker-containers) — pipelines ejecutan steps como containers.
- [Fundamento: Model Serving](/fundamentos/model-serving) — serving es C8 del MLOps.

---

## 13. Lecturas

- **[Sculley et al. (2015)](/papers/hidden-technical-debt-sculley-2015)** — Hidden Technical Debt in Machine Learning Systems.
- **[Kreuzberger et al. (2023)](/papers/mlops-overview-kreuzberger-2023)** — MLOps: Overview, Definition, and Architecture.
- **[Paleyes et al. (2022)](/papers/challenges-deploying-ml-paleyes-2022)** — Challenges in Deploying Machine Learning.
- **Zinkevich (Google, 2017)** — *Rules of Machine Learning: Best Practices for ML Engineering*. 43 reglas operacionales.
- **Huyen (2022)** — *Designing Machine Learning Systems* (O'Reilly). Libro de referencia.
- **ml-ops.org** — sitio comunitario.
- **Google Cloud MLOps maturity model** — whitepaper sobre niveles 0/1/2.
