# Paleyes, Urma & Lawrence (2022) — Challenges in Deploying Machine Learning: a Survey of Case Studies

| Campo | Valor |
|---|---|
| Autores | Andrei Paleyes, Raoul-Gabriel Urma, Neil D. Lawrence |
| Filiación | University of Cambridge + Cambridge Spark |
| Venue | ACM Computing Surveys, Vol. 55, No. 6, Article 114 (January 2022) |
| DOI | 10.1145/3533378 |
| arXiv | 2011.09926 (v3 — May 2022) |
| Páginas | 29 |
| Tipo | Systematic survey de case studies industriales |
| Referencias | 159 |
| PDF | [Paleyes-ChallengesDeployingML-2022.pdf](Paleyes-ChallengesDeployingML-2022.pdf) |

---

## 1. Contexto histórico

Para 2020-2022 ya se sabía que desplegar ML en producción era difícil — Sculley 2015 lo había gritado siete años antes. Lo que faltaba era una **revisión sistemática académica** que:

1. Recolectara case studies industriales publicados (no más opiniones de Google).
2. Mapeara los problemas reales a las etapas del workflow ML.
3. Cuantificara la frecuencia con que aparecen distintos tipos de challenges.
4. Sirviera de research agenda para la comunidad académica.

El paper viene del **Cambridge Machine Learning Group** liderado por Neil Lawrence (figura histórica del ML probabilístico, ex-director de ML en Amazon). El sesgo no es Google ni IBM como en Sculley/Kreuzberger: es **académico-British-pragmatic**.

Citas previas que motivan el paper:
- **McKinsey (2019)**: ML crece ~25 % YoY en adopción empresarial.
- **Algorithmia (2019)**: mayoría de empresas toma **8-90 días** para deployar un modelo; 18 % más de 90 días.
- **IDC**: porción significativa de deployments ML **fallan** por lack of expertise + bias + costos.
- **O'Reilly survey**: la pregunta ya no es "¿dónde se usa ML?" sino "¿qué tan difícil es usarlo?".

---

## 2. Metodología

### 2.1 Tipos de papers incluidos

- **Case studies**: experiencia con un proyecto de deployment específico.
- **Review papers**: aplicaciones de ML en una industria con resumen de challenges.
- **"Lessons learned"**: reflexiones de experiencias pasadas.

**Restricción:** solo los últimos 5 años (2016-2021), con algunas excepciones.

### 2.2 Definición de "ML deployment workflow" usado

Adoptan el framework de **Ashmore et al. (2019)** que divide el proceso en 4 etapas:

1. **Data management** — preparar data necesaria.
2. **Model learning** — selección + training.
3. **Model verification** — adherencia a requirements funcionales/performance.
4. **Model deployment** — integración + mantenimiento + updates.

Plus **cross-cutting aspects** que afectan todas las etapas: ethics, law, end-users' trust, security.

Cada etapa se desglosa en sub-pasos. Total: 16 sub-pasos × N challenges = la tabla maestra del paper.

### 2.3 Tabla 1 — el mapa maestro

Esta es la tabla central del survey. Se reproduce parcialmente abajo (challenges destacados):

| Stage | Step | Challenges identificados |
|---|---|---|
| **Data management** | Data collection | Data discovery |
| | Data preprocessing | Data dispersion, Data cleaning |
| | Data augmentation | Labeling de grandes volúmenes, Access to experts, Lack of high-variance data |
| | Data analysis | Data profiling |
| **Model learning** | Model selection | Model complexity, Resource-constrained environments, Interpretability |
| | Training | Computational cost, Environmental impact, Privacy-aware training |
| | Hyper-parameter selection | Resource-heavy techniques, Unknown search space, Hardware-aware optimization |
| **Model verification** | Requirement encoding | Performance metrics, Business-driven metrics |
| | Formal verification | Regulatory frameworks |
| | Test-based verification | Simulation-based testing, Data validation routines, Edge case testing |
| **Model deployment** | Integration | Operational support, Reuse of code/models, SE anti-patterns, Mixed team dynamics |
| | Monitoring | Feedback loops, Outlier detection, Custom design tooling |
| | Updating | Concept drift, Continuous delivery |
| **Cross-cutting** | Ethics | Aggravation of biases, Fairness/accountability, Authorship, Decision making |
| | Law | Country-level regulations, Existing legislation, Tech-only focus |
| | End users' trust | User involvement, UX, Explainability score |
| | Security | Data poisoning, Model stealing, Model inversion |

---

## 3. Hallazgos por etapa — los problemas reales reportados

### 3.1 Data Management (sección 3)

#### Data collection — el problema de "no sé qué datos existen"

> "Finding data sources and understanding their structure is a major task, which may prevent data scientists from even getting started."

**Case study: Twitter** (Lin & Ryaboy 2013). Twitter internamente consiste en muchos servicios siguiendo "single responsibility principle". Cada servicio responsable de **una operación** sobre la misma entity (e.g., un user). Arquitectura flexible para scaling/modification. **Flip side:** a gran escala es muy difícil rastrear qué data sobre la entity está storage en qué servicio y en qué forma. Algunos datos solo existen como logs no fácilmente parseables. Caso peor: el data no está storage en ningún lugar — para construir un dataset hay que **generar synthetic API calls al servicio**.

#### Data preprocessing — data dispersion

**Case study: Firebird** (Atlanta Fire Department). Sistema para priorizar inspecciones contra incendios. Data collection involucró **12 datasets**: incidentes, licencias de negocio, households, etc. **Joining geospatial data** (cada dataset codificaba edificios diferente) fue el bloqueo más difícil. Diferencias menores en spelling, formatos espaciales distintos. Cleaning consumió enormous time.

#### Data augmentation — labeling at scale

Tres factores para falta de labels:

1. **Volume** — análisis de tráfico de red: 1 GB/s = 1.5M paquetes/s. Aun con downsampling masivo, hay que rastrear cada paquete. Dos approaches: uncontrolled (traffic real, ground truth difícil) o controlled (emulado, baja calidad). Estudios muestran herramientas de labeling pueden introducir errores hasta **100 %** en tráfico encriptado.

2. **Access to experts** — análisis de imagen médica requiere expertos múltiples cuyos labels se agregan. Raramente factible. Alternativas: **noisy oracles** o **weak supervision** — labels imprecisos que degradan accuracy. En healthcare las pérdidas son inaceptables ("The Final Percent challenge" — Budd et al. 2021).

3. **High-variance data** — en RL especialmente, el agente puede explorar solo dentro de un espacio restringido por seguridad → política de low-variance → no entrenada para situaciones inseguras. Caso: control de vehículos autónomos (Dulac-Arnold et al. 2021).

#### Data analysis — falta de herramientas para profiling

Microsoft survey: data scientists piensan que **data issues son la principal razón** para dudar de la calidad de su trabajo.

### 3.2 Model Learning (sección 4)

#### Model selection — complexity vs interpretability

> "the selection of a model is decided by one key characteristic: complexity."

**Case study: AirBnB search** (Haldar et al. 2019). El equipo arrancó con un complex deep model. Fue **abrumado por su complejidad** y consumieron development cycles. Tras varios fallos, simplificaron **drásticamente** a una NN simple de un layer hidden con 32 ReLU activations. Incluso ese modelo simple valió la pena porque permitió construir el pipeline de deployment completo. Eventualmente añadieron un segundo hidden layer pero **nunca alcanzaron la complejidad originalmente prevista**.

**Case study: Europa Clipper spacecraft** (Wagstaff et al. 2019). NASA tenía constraints de hardware extremos (peso, robustez, power). Deep learning **no fue considerado**. Usaron threshold simple + PCA en 3 tasks de anomaly detection.

**Case study: banking churn prediction** (Keramati et al. 2016). Decision trees usados ampliamente **por interpretabilidad** — el negocio necesita explicar a clientes/reguladores qué features influyen.

#### Training — el costo económico y ambiental

> "the overall cost of training NLP is only growing."

**BERT**: $50k - $1.6M USD por full training cycle. Inalcanzable para la mayoría de instituciones.

**Strubell et al. 2019**: NAS training emite CO₂ comparable a lo que **cuatro autos promedio emiten en toda su vida útil**.

**Privacy concerns** durante training: Shokri et al. demostraron **membership inference attacks** sobre ML-as-a-service con 70-94 % accuracy. Mitigaciones (differential privacy, homomorphic encryption, federated learning) **cuestan accuracy**.

#### Hyper-parameter selection

HPO grow exponencialmente con número de hiperparams. Incluso Hyperband, Bayesian optimization no escalan bien para deep learning. Definir **search space bounds** es otro problema — uno de los principales obstáculos para uso amplio de HPO state-of-the-art (Yang & Shami 2020).

**Hardware-aware HPO**: para deploy en mobile/embedded, optimizar conjuntamente accuracy y constraints de energy/memory.

### 3.3 Model Verification (sección 5)

#### Requirement encoding — ¡performance ≠ business value!

**Case study canónico: Booking.com** (Bernardi et al. 2019). Deployaron **150 modelos en producción**. Descubrieron que **una mejora de performance del modelo NO se traduce automáticamente en business value**. Por ejemplo: una métrica proxy como "clicks" no convierte a la business metric "conversiones". Métricas business-driven (conversion, customer service tickets, cancellations) requieren **cross-disciplinary effort** entre modeling, engineering, business.

**Implicación práctica:** debes definir métricas business antes de iterar el modelo, no después. Y monitorear ambos tipos en producción.

#### Formal verification — regulatory frameworks

Ejemplo banking (post-2008 crisis). UK Prudential Regulation Authority y European Central Bank publicaron guidelines de "model risk management". Requieren **frameworks formales para todas las decision-making solutions**. Implementar requiere test suites extensos para entender comportamiento del modelo.

Healthcare: FDA US publicó action plan para regulatory framework de medical ML.

#### Test-based verification — simulación tiene límites

**Case study: ISS Cognitive Engine** (Hackett et al. 2018). RL-based engine para software-defined radio en la International Space Station. Preparation usó emulated environment con extensive ground testing que informó hyperparameter choices. **Cuando el software se deployó en ISS, las condiciones reales fueron tan harsh que solo pudieron testear un subset de experimentos planeados**. CE no pudo manejar emergencies. Lección: **simulación nunca reemplaza completamente real-world testing**, especialmente para safety-critical systems.

### 3.4 Model Deployment (sección 6)

> "Machine learning systems running in production are complex software systems that have to be maintained over time."

#### Integration — reuse y mixed team dynamics

**Case study: Pinterest** (Zhai et al. 2019). Tres modelos internos usaban **similar embeddings** mantenidas separadamente. Cada effort se multiplicaba por tres. Decidieron aprender un **universal embedding set**. Resultó posible, **simplificó deployment pipelines** y mejoró performance individual de cada task.

**Sobre Sculley 2015**: Paleyes lo cita explícitamente como referencia maestra. Confirma que muchos problemas SE-related que aparecen en ML son **anti-patterns conocidos en SE clásica**, exacerbados por dependencias en data externa.

**Team dynamics**: línea típica research-vs-engineering produce silos. Solución: researchers en el journey de deployment, mismo codebase, code reviews, etc. Onboarding más lento pero benefits long-term en velocidad + quality.

#### Monitoring — el problema abierto

> "the ML community is in the early stages of understanding what are the key metrics of data and models to monitor."

**Outlier detection** (Klaise et al. 2020) es señal clave para flag predictions no usables. Dos razones para outliers: incapacidad del modelo de generalizar fuera del training set, overconfident predictions sobre out-of-distribution instances. Deploy del outlier detector mismo es challenge — labeled outlier data es escaso, training del detector típicamente semi-supervised o unsupervised.

**Case study: police EIS** (Ackermann et al. 2018). Early Intervention System para dos departamentos de policía US. Objetivos de monitoring "parecían standard": data integrity checks, anomaly detection, performance. Pero tuvieron que construir todo **desde cero** porque out-of-the-box tools no fitteaban las particularidades. Lección: **plataformas ML genéricas no fittean bien por la especificidad de cada problema**.

#### Updating — concept drift

**Concept drift = dataset shift** = cambios en $P(X, Y)$ donde $X$ es input y $Y$ es output. Puede ocurrir discreta (post-evento externo) o continua (gradual).

Examples:
- Finance industry durante crisis 2008.
- Predictive maintenance: drift microscópico en wear/tear ("Zenisek et al. 2020").
- Marine images: drift por slight changes (Langenkämper et al. 2020).

**Continuous delivery for ML** (Sato et al. 2019): aproach al problema de delivering el model artifact a producción. ML difiere de software regular porque cambia en **tres ejes: code, model, data**.

**Backward compatibility issue** (Bansal et al. 2019, Srivastava et al. 2020): un model update puede degradar performance del equipo AI-human aunque la accuracy individual del modelo mejore. Optimization stochasticity + noisy training datasets causan incompatibilities. Necesita de-noising + compatibility-aware training methods.

### 3.5 Cross-cutting aspects (sección 7)

#### Ethics

> "Since ML models use previously seen data to make decisions, they can rely on hidden biases that already exist in data — a behavior that is hard to foresee and detect."

Ejemplos:
- **Criminal justice** (O'Neil 2016): "risk scores" marketeados como removing human bias usan demographic info que sirve de proxy para race/income.
- **Machine translation** (Prates et al. 2020): Google Translate muestra default masculino para roles típicamente con distribución desbalanceada (STEM).
- **Disaster Risk Management** (Soden et al. 2019): biased training datasets aggravan inequalities.
- **Civil unrest forecasting EMBERS** (Muthiah et al. 2016): herramienta de forecasting puede ser misused por gobiernos (deliberate o no).
- **Facial analysis** (Buolamwini-Gebru 2018): el paper canónico "Gender Shades". Datasets imbalanced en skin colour → women dark-skinned son the most misclassified group. Motivó toda la subdiscipline **Fairness in ML**.

#### Law

- **GDPR (EU)**, ethical screening laws (Asia). Healthcare especialmente afectada.
- **Cat-and-mouse game** (Malan 2018, World Economic Forum): legislación toma años, ML avanza meses. Por el momento las regs son obsoletas.
- **GDPR "right to explanation"** (Wachter et al.): GDPR carece de lenguaje preciso → fails to guarantee el derecho.
- **DeepMind + Royal Free NHS Foundation Trust** (Streams): la collaboration inicial **no fue específica enough** sobre patient data use → investigation, compromise of data protection.

#### End users' trust

**Case study: Sepsis Watch** (Sendak et al. 2020). Médicos escépticos por intentos previos fallidos. Equipo priorizó **trust-building** via communication, early engagement, accountability mechanisms. Lección: **interpretability tiene límites como trust-building tool**.

**Case study: "Brilliant Doctor"** (Wang et al. 2021). AI-powered diagnosis tool en China rural. Mayoría de doctores **no lo usó productivamente** porque UX no consideraba el ambiente clínico real (screen sizes, interacción con otras app). Lección: **UX context-aware es esencial**.

**Case study: Wang et al. 2020** (loan decisions). XGBoost outperformed traditional scorecards pero **carece de componente de explanation**. Tuvieron que desarrollar custom **loan decision explanation** technique (similar a SHAP) para QuickBooks Capital.

#### Security — adversarial attacks

Tres ataques canónicos sobre modelos deployed:

1. **Data poisoning** — corromper integrity del training data. Caso: **medical setting con linear model** → solo **8 % de muestras malicious** introducidas resultó en dosage incorrecta para **mitad de pacientes** (Jagielski et al. 2018).

   También ocurre vía feedback loops. **Microsoft Tay** (2016): chatbot diseñado para mejorar com el tiempo, deluged por mensajes deliberadamente abusivos. **Dentro de 16 horas** una porción significativa de sus mensajes era abusiva/offensive. Tay fue **bajado**.

2. **Model stealing** (Tramèr et al. 2016) — reverse engineering via queries a la prediction API + monitoring de outputs. Replicaron modelos en producción de Google, Amazon, Microsoft (logistic regression, decision trees, SVMs, NNs) con **650-4013 queries** en **70-2088 segundos**. Pérdida de IP es la consecuencia.

3. **Model inversion** (Fredrikson et al.) — recuperar parts del private training set explotando models que reportan confidence values con sus predictions. Crítico para GDPR compliance.

---

## 4. Potenciales soluciones (sección 8)

### 4.1 Tools and services

El paper menciona como **end-to-end platforms** que solucionan partes de los challenges:
- **AWS SageMaker**
- **Microsoft ML**
- **Uber Michelangelo**
- **TensorFlow TFX**
- **MLflow**

Estas plataformas reducen burden operacional pero **introducen dependencias** que son su propio mantenance burden.

Otras tools específicas:
- **Jenga** (Schelter et al. 2021) — robustness against data errors.
- **CheckList** (Ribeiro et al. 2020) — formal approach a NLP model quality.
- **Data Linter** (Hynes et al. 2017) — inspect ML datasets para issues.
- **Snorkel, Snuba, cleanlab** — weak supervision.
- **AutoML**: Auto-keras, Auto-sklearn, TPOT (Olson & Moore 2016).
- **Drift detection**: Alibi Detect, Azure ML monitor, SageMaker Clarify.

### 4.2 Holistic approaches

- **ML Technology Readiness Levels (MLTRL)** (Lavin et al. 2021) — framework process para producir ML systems robustos, considerando key differences ML vs SE clásica.
- **DELVE** (Royal Society) — aplicó MLTRL durante COVID-19.
- **Datasheets for Datasets** (Gebru et al. 2018) — documentación motivation, composition, collection process, intended uses.
- **DVC (Data Version Control)** — Git-like para datasets.
- **Model Cards** (Mitchell et al. 2019) — documentos cortos describiendo performance, intended use, contexto.
- **Data Oriented Architectures (DOA)** (Lawrence 2019) — replace micro-service architecture con dataflow-based architectures. Hace data flow más explícito, simplifica data discovery + labeling.

### 4.3 Guidelines

- **Zinkevich's "Rules of ML"** (Google, 2017) — coleección de 43 reglas/advice.
- **VDI guidelines** (Verein Deutscher Ingenieure) para big data en manufactura.

---

## 5. Aportes únicos del paper

1. **Survey académico riguroso** — 159 referencias, methodology sistemática, mapeo formal de challenges a workflow steps.
2. **Tabla 1** — el mapa canónico de challenges en deployment, citado en literatura subsiguiente.
3. **Case studies industriales reales** — no opiniones, ejemplos concretos verificables (AirBnB, Pinterest, Booking, Twitter, Firebird, ISS, Sepsis Watch...).
4. **Identifica cross-cutting aspects** (ethics, law, trust, security) como first-class — pre-Sculley no se enfatizaban.
5. **Distingue tools vs holistic approaches** — tools resuelven problemas específicos, approaches reformulan el paradigma.
6. **Open research agenda explícita** — sección 9 (Further Work) propone qué falta investigar.

---

## 6. Limitaciones del paper

- **Solo case studies publicados** — sesgo de qué empresas/equipos publican. Muchos failures no se cuentan.
- **No incluye interviews propias** (mencionan como future work).
- **Survey vs solución** — el paper identifica problemas, no provee receta para cada uno.
- **Pre-LLM-era** — concept drift en context de fine-tuning de LLMs no se aborda.
- **Énfasis SE/data engineering** — poco en aspectos sociotécnicos (gobernanza, audit).
- **Geographic bias** — case studies dominados por US/UK/Europa.

---

## 7. Impacto y legado

Para 2026 el paper es referencia estándar en:

- **Cursos universitarios** de ML production (MIT 6.S965, Stanford CS329S, CMU 10-718).
- **Books**: Chip Huyen "Designing ML Systems" (O'Reilly 2022) lo cita constantemente.
- **Otras surveys**: Steidl et al. 2023, Mäkinen et al. 2021.
- **Cited by Kreuzberger 2023** como antecedente directo.

El estilo Cambridge ML Group (riguroso, holistic, sociotécnico) influencia el discurso académico de "data-centric AI" liderado por Andrew Ng (2021+).

---

## 8. Conexión con la clase 19

### Mapeo al PDF del prof Javier Rojas

El prof presenta MLOps de forma **tooling-céntrica** (qué frameworks, qué cloud services). Paleyes 2022 da el **mapa de los problemas reales** que esas tools intentan resolver. Mapeo:

| Slide del prof | Paleyes 2022 aporta |
|---|---|
| 36-37 — Inferencia: "ejecutar datos en modelo para calcular resultado" | Sección 6 (deployment) profundiza qué significa "ejecutar" en producción: integration, monitoring, updating |
| 38-39 — Esquema cliente-modelo (web/móvil/consola) | Sección 6.1 (integration) — reuse de código, mixed team dynamics, anti-patterns SE |
| 40 — Deployment: "disponibilizar modelos para uso real" | "Real" introduce ethics/law/trust/security (sección 7) — el prof omite estos completamente |
| 47-49 — Cloud Run, Vertex AI, multi-nodos | Soluciones a problemas de **escalabilidad** que Paleyes lista en 4.2 (training) y 6.2 (monitoring) |
| 51-54 — "¿Cuándo un producto IA está terminado?" | Paleyes sección 6.3 (updating) — nunca, por concept drift, backward compatibility, continuous delivery for ML |
| 54 — "datos cambian con el tiempo, versiones del modelo más novedosas" | Paleyes 6.3 (updating) profundiza: concept drift formal, examples industriales, backward compatibility |
| 60-61 — Frameworks MLOps + cloud services | Paleyes 8.1 (Tools and services) — lista similar pero **con caveat**: introducen sus propias dependencias |

### Conceptos críticos que el PDF omite y Paleyes aporta

- **Business-driven metrics** (Booking.com case): performance ≠ value.
- **Ethics & bias** (Buolamwini-Gebru, O'Neil).
- **Adversarial attacks** (poisoning, stealing, inversion).
- **Backward compatibility** en model updates.
- **UX context-aware** (Brilliant Doctor case).
- **MLTRL, Datasheets, Model Cards** como holistic governance.
- **Trust-building via comunicación + early engagement** (Sepsis Watch).
- **Costo ambiental** de training NLP/LLM (Strubell et al.).

### Para Fase 2 del site

Este paper debe:
- Card en `clases/clase-19/_index.md`.
- Citarse en `clases/clase-19/teoria.md` cuando se habla de deployment + MLOps.
- Base para fundamento `fundamentos/ml-production-challenges.md` o capítulo dentro de `fundamentos/mlops.md`.
- Múltiples case studies se prestan para anecdóticos pedagógicos.

---

## 9. Case studies memorables para citar en clase

| Case | Lección |
|---|---|
| **AirBnB search** | Complexity premature kills projects. Simplifica primero. |
| **Europa Clipper** | Hardware constraints fuerzan modelos clásicos (DT, PCA). |
| **Booking.com 150 modelos** | Performance metrics ≠ business metrics. |
| **BERT $50k-$1.6M** | Costo de training crece exponencial; environmental impact CO₂. |
| **ISS Cognitive Engine** | Simulación nunca reemplaza real-world testing en safety-critical. |
| **Pinterest universal embeddings** | Reuse de modelos paga dividendos. |
| **Microsoft Tay** | Feedback loops abiertos = data poisoning. |
| **Tramèr 2016** | Model stealing en horas con miles de queries → IP loss. |
| **Buolamwini-Gebru "Gender Shades"** | Bias en datasets se manifiesta como inequidad social. |
| **Sepsis Watch** | Trust = communication + early engagement, no solo interpretability. |
| **Brilliant Doctor China rural** | UX context-aware es clave para adopción. |

---

## 10. Quotes memorables

> "Just as with any other field, there are significant differences between what works in an academic setting and what is required by a real world system."

> "Many concepts that are routinely used in software engineering are now being reinvented in the ML context."

> "Performance metrics should also reflect audience priorities."

> "MLTRL describes a process of producing robust ML systems that takes into account key differences between ML and traditional software engineering with a specific focus on the quality of the intermediate outcome of each stage of the project."

---

## 11. Relación con otros papers del set

- **Sculley 2015** — Paleyes lo cita como ancestor; confirma que la mayoría de SE anti-patterns en ML eran ya conocidos pero "reinvented in ML context". [ver análisis](Sculley-HiddenTechnicalDebt-2015.md).
- **Kreuzberger 2023** — Paleyes provee los **problemas**, Kreuzberger los **componentes de solución arquitectónica**. Complementarios. [ver análisis](Kreuzberger-MLOpsOverview-2023.md).

Los tres forman el **triángulo canónico** para entender MLOps académicamente:

```
        Sculley 2015
         (los problemas
       desde Google ads)
              /\
             /  \
            /    \
           /      \
   Paleyes 2022   Kreuzberger 2023
   (case studies   (definición + arch
    industriales    + roles + tools)
    transversales)
```
