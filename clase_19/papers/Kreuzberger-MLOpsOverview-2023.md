# Kreuzberger, Kühl & Hirschl (2023) — Machine Learning Operations (MLOps): Overview, Definition, and Architecture

| Campo | Valor |
|---|---|
| Autores | Dominik Kreuzberger (KIT), Niklas Kühl (KIT), Sebastian Hirschl (IBM) |
| Filiación | Karlsruhe Institute of Technology + IBM Germany |
| Venue | IEEE Access, 2023 (preprint arXiv 2205.02302, mayo 2022) |
| DOI | 10.1109/ACCESS.2023.3262138 (también disponible en arXiv) |
| Páginas | ~16 |
| Tipo | Mixed-method research (literature review + tool review + expert interviews) |
| PDF | [Kreuzberger-MLOpsOverview-2023.pdf](Kreuzberger-MLOpsOverview-2023.pdf) |

---

## 1. Contexto histórico y problema

Para 2022-2023, MLOps llevaba ~5 años como término popular (acuñado alrededor de 2017-2018 por la comunidad DevOps que migraba a ML). Sin embargo, el campo carecía de:

- **Definición consensuada** de qué es MLOps exactamente.
- **Arquitectura de referencia** que sirva como blueprint.
- **Lista canónica de principios, componentes y roles**.

Diferentes vendors, comunidades open-source y papers usaban "MLOps" con significados parcialmente solapados. Esto creaba miscommunication entre researchers y practitioners. Los autores plantean una **research question** explícita:

> **RQ: What is MLOps?**

El paper toma el enfoque académico clásico: descubrir la definición que **ya existe implícitamente** en literatura + práctica industrial, y consolidarla. Es el paper "estado del arte" canónico de MLOps al 2023.

---

## 2. Metodología — mixed-method

Tres pilares:

### 2.1 Literature review (27 papers seleccionados)

- Bases: Google Scholar, Web of Science, Science Direct, Scopus, AIS eLibrary.
- Query: `((("DevOps" OR "CICD" OR "CI/CD" OR "Continuous Integration" OR "Continuous Delivery" OR "Continuous Deployment") AND "Machine Learning") OR "MLOps" OR "CD4ML")`
- Fecha: mayo 2021. Recolectaron 1.864 artículos → screening de 194 → 27 finales.
- Mayoría peer-reviewed; algunos workshops/preprints.

### 2.2 Tool review (11 herramientas open-source + commercial ML cloud services)

Identificaron tooling existente para entender qué componentes técnicos son canónicos en la práctica.

### 2.3 Interview study (8 expertos)

Semi-estructuradas según método Myers & Newman. Selección **teorética** vía LinkedIn — distintas industrias, países, géneros, niveles de seniority. Llegaron a *theoretical saturation* (Glaser & Strauss) en 8 interviews. Codificación abierta.

Los 8 entrevistados se referencian con letras griegas α, β, γ, δ, ε, ζ, η, θ a lo largo del paper para atribuir afirmaciones a fuentes específicas.

> **Nota metodológica:** este rigor metodológico (mixed-method, theoretical sampling, saturation) es lo que diferencia este paper de blogs y tutoriales de MLOps. Es **academic-grade**.

---

## 3. Los 9 principios de MLOps (sección 4.1)

Identifican **9 principios** transversales. Un "principio" se entiende como "best practice del sector profesional" — guía de comportamiento, no algoritmo.

| ID | Principio | Esencia |
|---|---|---|
| **P1** | CI/CD automation | Build/test/deliver/deploy automatizados con feedback rápido al desarrollador |
| **P2** | Workflow orchestration | Tareas del workflow ML como DAG (Directed Acyclic Graph) con orden por dependencias |
| **P3** | Reproducibility | Capacidad de re-ejecutar un experimento ML y obtener exactamente el mismo resultado |
| **P4** | Versioning | Versionado de **datos, modelo y código** para trazabilidad (compliance, auditing) |
| **P5** | Collaboration | Trabajo colaborativo en datos/modelo/código + cultura comunicativa que reduzca silos |
| **P6** | Continuous ML training & evaluation | Retraining periódico con nuevos feature data, soportado por monitoring + feedback loop + workflow pipeline |
| **P7** | ML metadata tracking/logging | Logging de metadata por cada task del pipeline (training date, duration, params, metrics, model lineage) |
| **P8** | Continuous monitoring | Assessment periódico de data/modelo/código/infra/serving performance para detectar errores y degradaciones |
| **P9** | Feedback loops | Múltiples loops conectando monitoring → scheduler → training, y experimentación → feature engineering |

**Insight del paper:** los principios **no son independientes**. Cada componente técnico de la sección 4.2 implementa varios principios simultáneamente. La matriz componentes×principios cierra el bucle.

---

## 4. Los 9 componentes técnicos (sección 4.2)

Cada componente lleva paréntesis con los principios que implementa.

### C1 — CI/CD Component (P1, P6, P9)

Build/test/delivery/deploy continuo. **Ejemplos:** Jenkins, GitHub Actions.

### C2 — Source Code Repository (P4, P5)

Storage + versioning de código. Multi-developer commit + merge. **Ejemplos:** Bitbucket, GitLab, GitHub, Gitea.

### C3 — Workflow Orchestration Component (P2, P3, P6)

DAGs para orquestar tareas ML. **Ejemplos:** Apache Airflow, Kubeflow Pipelines, Luigi, AWS SageMaker Pipelines, Azure Pipelines.

### C4 — Feature Store System (P3, P4)

Storage central de features comunes. **Dos bases de datos:**
- Offline store: latencia normal, para experimentación/training.
- Online store: baja latencia, para inferencia en producción.

**Ejemplos:** Google Feast, Amazon SageMaker Feature Store, Tecton.ai, Hopsworks.ai.

**Implicación:** la mayoría del data para training debería venir del feature store, no de extracciones ad-hoc. Esto resuelve el "unstable data dependencies" de Sculley.

### C5 — Model Training Infrastructure (P6)

CPU/RAM/GPU foundational. **Distribuida y escalable es lo recomendado**. Ejemplos: máquinas locales (no escalables), cloud computing, Kubernetes, Red Hat OpenShift.

### C6 — Model Registry (P3, P4)

Storage central de modelos entrenados + metadata. **Ejemplos:** MLflow, SageMaker Model Registry, Azure ML Model Registry, Neptune.ai. Storage simple: Azure Storage, Google Cloud Storage, Amazon S3.

### C7 — ML Metadata Stores (P4, P7)

Tracking de metadata por task: training date/duration, params, performance metrics, **model lineage** (data + código usados). **Ejemplos:** orquestrators con metadata built-in (Kubeflow, SageMaker, Azure, IBM Watson Studio). MLflow ofrece advanced metadata store + model registry combinado.

### C8 — Model Serving Component (P1)

Serving online (real-time) o batch. Infraestructura serving recomendada: **escalable y distribuida**. Ejemplo de configuración:
- Kubernetes + Docker para containerizar
- Python web framework como Flask/FastAPI con REST API
- **Otros frameworks Kubernetes-supported:** KServing of Kubeflow, TensorFlow Serving, Seldon.io.

**Batch:** Apache Spark.

**Cloud:** Azure ML REST API, AWS SageMaker Endpoints, IBM Watson Studio, Google Vertex AI prediction service.

### C9 — Monitoring Component (P8, P9)

Monitoring continuo de model serving performance (accuracy), infrastructure, CI/CD, orchestration. **Ejemplos:** Prometheus + Grafana, ELK stack (Elasticsearch + Logstash + Kibana), TensorBoard. Con built-in: Kubeflow, MLflow, AWS SageMaker model monitor, AWS CloudWatch.

---

## 5. Los 7 roles (sección 4.3)

Los autores identifican siete roles necesarios para realizar MLOps; algunos son intersección de varios.

| ID | Rol | Foco |
|---|---|---|
| **R1** | Business Stakeholder (similar: Product Owner, Project Manager) | Define el goal de negocio; comunica ROI |
| **R2** | Solution Architect (similar: IT Architect) | Diseña la arquitectura; elige tecnologías tras evaluación |
| **R3** | Data Scientist (similar: ML Specialist, ML Developer) | Traduce business problem en ML problem; model engineering, algoritmo, hiperparámetros |
| **R4** | Data Engineer (similar: DataOps Engineer) | Pipelines de data; ingesta a la feature store |
| **R5** | Software Engineer | Aplica design patterns + guidelines + best practices; convierte el problema raw en producto ingenieriado |
| **R6** | DevOps Engineer | Bridge dev/ops; CI/CD, orquestación, deployment, monitoring |
| **R7** | **ML Engineer / MLOps Engineer** | **Cross-domain**: Data Scientist + Data Engineer + Software Engineer + DevOps + Backend Engineer. Construye/opera el ML infra automatizada, pipelines, deployment, monitoring |

**Figura 3 del paper** muestra que R7 (ML/MLOps Engineer) es el **centro del diagrama de Venn** que intersecta Data Scientist, Backend Engineer, Software Engineer, DevOps Engineer, Data Engineer. Es el rol cuya emergencia justifica la disciplina entera.

---

## 6. Arquitectura end-to-end (sección 5)

El paper deriva una arquitectura technology-agnostic con **4 zonas funcionales** que avanzan secuencialmente con loops de feedback. Es **la Figura 4 del paper** — el plano más completo de MLOps publicado académicamente.

### Zona A — MLOps Project Initiation

5 pasos:
1. **R1** analiza el business problem → define goal.
2. **R2** diseña la arquitectura del sistema ML, decide tecnologías.
3. **R3** deriva el ML problem desde el goal (clasificación, regresión, etc.).
4. **R4 + R3** identifican data requerida.
5. **R4 + R3** se conectan a raw data, hacen distribución/quality/validation checks, aseguran labels.

### Zona B — Feature Engineering Pipeline

**B1 (Requirements):**
6. **R4** define rules de transformación (normalización, agregación, cleaning).
7. **R3 + R4** definen feature engineering rules (cálculo de features nuevos a partir de existentes).

**B2 (Pipeline construido):**
8. **R4 + R5** construyen prototipo del pipeline.
9. Pipeline conecta a raw data (streaming, batch, cloud storage).
10. Data preprocessing (transformación + cleaning) — task fundacional para que rule artifact sea utilizable.
11. Feature engineering calcula features avanzados.
12. Data ingestion job carga al **feature store system (C4)** — offline o online DB.

### Zona C — Experimentation

Liderada por R3 con apoyo de R5.
13-16. Data analysis, preparación/validación, split train/test, **estimación del best-performing algoritmo + hyperparams** (model engineering), iteración de model training/validation.
17. R3 exporta modelo y commits del código.

Aquí entra **CI/CD (C1)**: el commit triggera CI/CD automáticamente. Build → test → delivery. El artifact (containerizado, e.g., docker image) se push a image registry.

### Zona D — Automated ML Workflow Pipeline (Production)

Manejado por **R6 (DevOps) y R7 (MLOps Engineer)**. Infra Kubernetes (C5). Workflow orchestration (C3) ejecuta las tasks pull-eando artifacts del registry.

18-22. Pulling versioned features → preparación + validación → training final → evaluación → adjustment de hiperparams (si necesario) → exportar modelo → push al **Model Registry (C6)**.

ML metadata store (C7) registra params, metrics, lineage por iteración.

Cuando un modelo "well-performing" pasa de staging a production, **CI/CD (C1)** triggera el deployment pipeline. Build + test del serving code (preparado por R5). Deploy al **Model Serving Component (C8)**.

23-26. C8 hace predicciones sobre nueva data — features online (low latency) para real-time, offline (normal latency) para batch. Model-serving suele containerizarse + REST API. R7 gestiona la serving infra.

**(27) Feedback loop al Monitoring Component (C9)** que observa serving performance + infra. Si threshold se alcanza (e.g., accuracy baja), info se forwardea via feedback loop.

**(28) Continuous training:** monitoring detecta drift via comparaciones de distribución → forwardea al scheduler → triggera automated ML workflow pipeline para retraining. Retraining también puede ser scheduled o triggered por nueva data.

---

## 7. Definición formal de MLOps (sección 6)

Después de toda la conceptualización, el paper propone **la** definición:

> **MLOps (Machine Learning Operations) is a paradigm, including aspects like best practices, sets of concepts, as well as a development culture when it comes to the end-to-end conceptualization, implementation, monitoring, deployment, and scalability of machine learning products. Most of all, it is an engineering practice that leverages three contributing disciplines: machine learning, software engineering (especially DevOps), and data engineering. MLOps is aimed at productionizing machine learning systems by bridging the gap between development (Dev) and operations (Ops). Essentially, MLOps aims to facilitate the creation of machine learning products by leveraging these principles: CI/CD automation, workflow orchestration, reproducibility, versioning of data, model and code; collaboration; continuous ML training and evaluation; ML metadata tracking and logging; continuous monitoring; and feedback loops.**

Cinco ingredientes en la definición:
1. Es un **paradigma** (no una herramienta).
2. Incluye **best practices, conceptos, cultura**.
3. Es **engineering practice** que combina ML + SE (especialmente DevOps) + Data Engineering.
4. Su objetivo: **productionize** sistemas ML, cerrando gap Dev/Ops.
5. Sus medios: los 9 principios.

---

## 8. Open Challenges (sección 7)

El paper agrupa los desafíos pendientes en 3 categorías:

### Organizational

- **Mindset shift** de model-driven a **product-oriented** discipline.
- Falta de profesionales skill-fuls para roles especializados (arquitectos, data engineers, ML engineers, DevOps).
- MLOps **no es típicamente parte de data science education** — los students aprenden model building, no operations.

### ML system

- **Fluctuating demand**, especialmente en training. Difícil estimar recursos (CPU/RAM/GPU) → infra necesita alta flexibilidad de escalado.

### Operational

- Stack de software y hardware complejo → operación manual no viable.
- Stream de nueva data fuerza retraining → tarea repetitiva que necesita alta automatización.
- Estos producen **muchos artifacts** que requieren **strong governance** + versioning de data/modelo/código (asegurando robustez + reproducibilidad).
- Resolver support requests difícil porque hay muchas partes y componentes — falla puede combinar ML infra y software.

---

## 9. Aportes únicos del paper

1. **Primera definición académica consensuada** de MLOps que combina literatura + práctica industrial.
2. **Vocabulario canónico** P1-P9, C1-C9, R1-R7 que se puede citar/referenciar.
3. **Arquitectura end-to-end** technology-agnostic que sirve como blueprint para implementaciones concretas.
4. **El rol "MLOps Engineer"** queda formalizado como cross-disciplinary, no como "data scientist+".
5. Identifica que MLOps es **paradigma + cultura**, no solo tooling — alineado con la enseñanza de Sculley 2015.

---

## 10. Limitaciones del paper

- **Lit review hasta mayo 2021** — pre-LLM boom, pre-GPT-4. No incluye LLMOps específico.
- **Solo 8 interviews** — saturación teórica argumentable, pero muestra pequeña.
- **Bias hacia tooling cloud-native** — Kubernetes, vertex, sagemaker dominan ejemplos.
- **Poco sobre fine-tuning, RLHF, prompt engineering** — emergentes para 2024 pero no centrales en 2022.
- **Definición algo abstracta** — sirve para clasificar pero menos para implementar paso a paso.
- **No cuantifica** ROI o success metrics de adoptar MLOps. La afirmación "más MLOps maturity = más business improvement" se acepta sin medición.

---

## 11. Trabajos derivados (2023-2026)

- **LLMOps** emerge como sub-disciplina con vocabulario propio (vector DB, prompt registry, eval frameworks como Promptfoo, OpenAI Evals, LangSmith).
- **FMOps** (Foundation Model Ops) — operations específicas para modelos de fundación.
- **DataOps + MLOps convergence** — feature stores como Tecton/Feast se vuelven el lingua franca.
- **Cloud providers** alinean su tooling con los 9 componentes:
  - Vertex AI: Vertex Pipelines (C3), Feature Store (C4), Model Registry (C6), Endpoints (C8), Model Monitoring (C9).
  - SageMaker: Pipelines, Feature Store, Model Registry, Endpoints, Model Monitor.
  - Azure ML: Pipelines, Datasets, Models, Endpoints, Data Drift Monitors.

---

## 12. Conexión con la clase 19 — paper crítico para slide 56

Este paper es **literalmente la justificación arquitectónica del slide 56** del prof Javier Rojas (los tres lóbulos Design ↔ Development ↔ Operations).

### Mapeo directo PDF → paper

| Slide del prof | Aporte de Kreuzberger |
|---|---|
| 55 — "MLOps: paradigma repetible para implementar y mantener modelos confiables y eficientes" | Es **exactamente** la definición del paper (sección 6). El prof la parafraseó. |
| 56 — Diagrama 3 lóbulos: ML Design / Model Development / Operations | Es la Figura 4 del paper **simplificada**. Las 4 zonas A/B/C/D del paper colapsan en 3 lóbulos. |
| 57 — "Pipeline = flujo de trabajo con componentes y interacciones por entradas/salidas" | Implementa P2 (Workflow orchestration) + DAGs explícitos. |
| 58-59 — Pipeline boxes (Load Model, Load Dataset, Train, Test, Deploy) | Mapping a componentes de Kreuzberger: Load Dataset → C4 (feature store), Train → C5 (training infra), Deploy → C8 (serving), todo orquestado por C3 (workflow). |
| 60 — Frameworks: Kubeflow, Airflow, Flyte | C3 (workflow orchestration) ejemplos canónicos del paper. |
| 61 — Cloud: Azure ML Studio, AWS Sagemaker, Vertex AI | Plataformas que integran C1-C9 en una sola oferta. |
| 62 — Screenshot Vertex AI Pipeline | Implementación concreta del paradigma. |

### Conceptos del paper que el prof omite

- **Feature Store (C4)** — el prof no lo menciona en absoluto. Es un componente core de MLOps moderno.
- **Model Lineage** (parte de C7) — trazabilidad data + código → modelo. Fundamental para compliance.
- **Los 7 roles** — el prof habla del docente/proyecto pero no de qué roles humanos requiere un equipo MLOps.
- **El rol MLOps Engineer cross-functional** (R7) — el centro del diagrama de Venn.
- **Continuous training (P6) + feedback loops (P9)** — el prof menciona retraining pero no como un loop arquitectónico explícito.

### Para Fase 2 del site

Este paper debe ser:
- **Card en `clases/clase-19/_index.md`** sección "Papers de esta clase".
- **Citado en `clases/clase-19/teoria.md`** sección 6 (MLOps) — específicamente al introducir los 3 lóbulos.
- **Base para `fundamentos/mlops.md`** — los 9 principios + 9 componentes + 7 roles son el contenido natural del fundamento.

---

## 13. Quotes memorables

> "MLOps is a paradigm, including aspects like best practices, sets of concepts, as well as a development culture..."

> "MLOps Engineer combines aspects of several roles and thus has cross-domain knowledge."

> "Data scientists alone cannot achieve the goals of MLOps. A multi-disciplinary team is required, thus MLOps needs to be a group process."

> "The mindset and culture of data science practice is a typical challenge in organizational settings... a culture shift away from model-driven ML toward a product-oriented discipline."

---

## 14. Recursos asociados

- Repositorio companion: el paper menciona reproducibilidad pero no provee un repo canónico.
- **ml-ops.org** — sitio comunitario que codifica visualmente la arquitectura del paper.
- **Google Cloud MLOps maturity model (2020)** — niveles 0/1/2 que se alinean con la implementación de los 9 principios:
  - **Nivel 0:** procesos manuales (ningún P implementado).
  - **Nivel 1:** ML pipeline automation (P2 + P6 implementados).
  - **Nivel 2:** CI/CD pipeline automation (P1 + P2 + P6 + P9 implementados completamente).
- **AWS Well-Architected Machine Learning Lens** — alinea con Kreuzberger desde AWS.
- **Microsoft MLOps maturity model** — equivalente Microsoft.
