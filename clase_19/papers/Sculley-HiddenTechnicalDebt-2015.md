# Sculley et al. (2015) — Hidden Technical Debt in Machine Learning Systems

| Campo | Valor |
|---|---|
| Autores | D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-François Crespo, Dan Dennison |
| Filiación | Google, Inc. |
| Venue | NeurIPS 2015 (NIPS 2015) |
| Páginas | 9 |
| Versión corta previa | SE4ML Workshop, NIPS 2014, Montreal |
| Citaciones (~2026) | >7.000 — uno de los papers más citados sobre ML en producción |
| PDF | [Sculley-HiddenTechnicalDebt-2015.pdf](Sculley-HiddenTechnicalDebt-2015.pdf) |

---

## 1. Contexto histórico

Para 2015, machine learning ya estaba pasando del laboratorio a la producción a gran escala en empresas como Google, Facebook, Microsoft. La comunidad académica seguía midiendo éxito en accuracy/F1 sobre datasets fijos, pero los equipos de Google que operaban modelos en vivo notaban un fenómeno incómodo:

> "developing and deploying ML systems is relatively fast and cheap, but maintaining them over time is difficult and expensive."

El paper toma prestada la metáfora de **technical debt** acuñada por Ward Cunningham en 1992 para razonar sobre los costos de moverse rápido en software. La tesis central: los sistemas ML acumulan deudas técnicas adicionales **a nivel de sistema** que el debt tradicional de código no captura. Esa deuda es **silenciosa** porque vive en las interacciones entre componentes, datos y el mundo externo, no en el código.

El paper **no propone algoritmos nuevos**. Es un manifesto/taxonomía dirigido a researchers y practitioners. Su impacto histórico es enorme: catalizó el campo que años después se llamaría **MLOps**.

---

## 2. La Figura 1 — el insight central

La figura más citada del paper muestra que en sistemas ML reales el **ML code es una fracción minúscula** del sistema completo:

```
            ╔═════════════╗
            ║   ML code   ║   ← el cuadradito pequeño
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

Esto justifica empíricamente el "**5% ML code, 95% glue code**" que el propio paper cita más adelante en la sección de anti-patterns. Es la imagen que aparece en **todas** las presentaciones de MLOps desde 2016.

---

## 3. Las 7 categorías de deuda técnica ML-específica

### 3.1 Complex Models Erode Boundaries (sección 2)

**Premisa:** los sistemas software clásicos dependen de fronteras de abstracción estrictas (encapsulación, módulos). Los modelos ML las erosionan porque su comportamiento depende de **datos externos**, no de una lógica explícita.

#### 3.1.1 Entanglement — el principio CACE

> **CACE: Changing Anything Changes Everything.**

Si un modelo usa features $x_1, \ldots, x_n$ y se modifica la distribución de $x_1$, los pesos e importancia de los $n-1$ features restantes también cambian. Aplica a:
- Inputs (features)
- Hiperparámetros
- Sampling, convergence thresholds
- Selección de datos
- Esencialmente cualquier ajuste

**Mitigaciones planteadas:**

1. **Aislar modelos y servir ensembles** — útil cuando subproblemas se descomponen naturalmente (multi-clase disjunta). Pero los ensembles funcionan porque los errores son no correlacionados; mejorar un componente individual puede empeorar el sistema si los errores residuales pasan a estar más correlacionados.

2. **Detectar cambios de comportamiento en predicción** — visualizaciones de alta dimensión (referencia: McMahan et al. 2013, "Ad click prediction: a view from the trenches", el otro paper canónico de Google de la época). Métricas slice-by-slice.

#### 3.1.2 Correction Cascades

Modelo $m_a$ resuelve problema $A$. Se necesita problema ligeramente distinto $A'$ → tentación de aprender $m'_a$ que toma $m_a$ como input y corrige. Crea dependencia. Si luego $A''$ pasa sobre $m'_a$... **improvement deadlock**: mejorar un componente individual deteriora el sistema completo.

**Mitigaciones:**
- Augmentar $m_a$ con features que distingan los casos, manteniendo un solo modelo.
- Aceptar el costo de un modelo separado para $A'$.

#### 3.1.3 Undeclared Consumers

Predicciones expuestas en logs/files que otros equipos consumen sin acuerdo formal. **Visibility debt** en software clásico. Crea acoplamiento oculto.

> "any improvement to $m_a$ can break downstream consumers you didn't know existed."

**Mitigaciones:** access controls, strict SLAs, separación entre runtime y storage.

### 3.2 Data Dependencies Cost More than Code Dependencies (sección 3)

**Premisa:** las code dependencies se detectan con compilers/linkers. Las **data dependencies no tienen herramientas equivalentes** → invisibles, peligrosas.

#### 3.2.1 Unstable Data Dependencies

Inputs que cambian a lo largo del tiempo:
- Implícitamente: viene de otro modelo ML que se actualiza, o tabla TF/IDF.
- Explícitamente: otro equipo es dueño del signal y lo modifica sin avisar.

**Caso peligroso:** si un signal estaba mal calibrado y el modelo consumidor "aprendió" esa mala calibración, una "mejora" del signal **degradará** el modelo consumidor.

**Mitigación:** versioned copies de signals. Costo: staleness y mantenimiento de múltiples versiones.

#### 3.2.2 Underutilized Data Dependencies

Features que aportan poco valor incremental pero crean vulnerabilidad. Cuatro orígenes:
- **Legacy Features** — incluidos al inicio, hechos redundantes por features nuevos pero nunca removidos.
- **Bundled Features** — grupo evaluado como beneficioso, todos agregados juntos por presión de deadline, algunos sin valor.
- **ε-Features** — añadidos por ganancia marginal de accuracy.
- **Correlated Features** — dos correlacionadas pero solo una causal; el modelo distribuye crédito mal.

**Detección:** evaluaciones **leave-one-feature-out** exhaustivas, periódicas.

**Anécdota canónica del paper** (vale repetirla literal):

> "suppose that to ease the transition from an old product numbering scheme to new product numbers, both schemes are left in the system as features. New products get only a new number, but old products may have both and the model continues to rely on the old numbers for some products. A year later, the code that stops populating the database with the old numbers is deleted. **This will not be a good day for the maintainers of the ML system.**"

#### 3.2.3 Static Analysis of Data Dependencies

Propone tooling automático para anotar data sources y features, verificar dependencies, resolver árboles transitivos. Cita McMahan 2013 como ejemplo.

### 3.3 Feedback Loops (sección 4)

Sistemas live influencian su propio comportamiento → **analysis debt**.

#### 3.3.1 Direct Feedback Loops

El modelo influye qué datos verá en el futuro (e.g., un recomendador que aprende de clicks que él mismo genera). La solución teórica son contextual bandits, pero no escalan al tamaño de acciones reales. Mitigación: randomización parcial, aislar fracción del tráfico del modelo.

#### 3.3.2 Hidden Feedback Loops

Dos sistemas se influyen mutuamente **a través del mundo**, sin acoplamiento directo. Ejemplos:
- Dos componentes de una misma página web (productos + reviews) cuyas mejoras se afectan vía clicks.
- Dos modelos de bolsa de distintas firmas que se influyen vía precios.

Más difíciles de detectar y razonar.

### 3.4 ML-System Anti-Patterns (sección 5)

#### 3.4.1 Glue Code

"5% ML code, 95% glue code" — el código que envuelve paquetes genéricos (TensorFlow, scikit-learn) para meter/sacar datos. Costoso porque **fija el sistema a las peculiaridades del paquete**. Cambiar paquete se vuelve prohibitivo.

**Mitigación:** wrappear black boxes en APIs comunes, reusables.

#### 3.4.2 Pipeline Jungles

Caso especial de glue code, en data prep. Crecen orgánicamente: scrapes, joins, sampling, files intermedios. Difícil de testear (requiere integration tests caros).

**Mitigación:** pensar holísticamente sobre data collection y feature extraction; a veces **clean-slate redesign** es la única salida.

Causa raíz cultural: separación rígida research vs engineering. La solución es equipos híbridos.

#### 3.4.3 Dead Experimental Codepaths

Branches condicionales para experimentar dentro del código de producción. Acumulan complejidad ciclomática exponencial. **Ejemplo famoso:** Knight Capital perdió USD 465 millones en 45 minutos por código experimental obsoleto que volvió a activarse.

**Mitigación:** revisar y borrar branches no usadas periódicamente.

#### 3.4.4 Abstraction Debt

ML carece de abstracciones tan robustas como la base de datos relacional o el sistema de archivos. ¿Cuál es la interfaz correcta para describir un stream de datos, un modelo, una predicción? El paper menciona Map-Reduce como abstracción pobre para ML iterativo. El **parameter server** (Li et al. OSDI'14) emergía como candidato pero con especificaciones competidoras.

> **Reflexión 2026:** una década después, el campo aún no tiene consenso. Tenemos competidores parciales: ONNX para intercambio de modelos, MLflow/W&B para tracking, Kubeflow para orquestación, Feast para feature stores. Cada uno es una abstracción razonable, ninguna es **la** abstracción.

#### 3.4.5 Common Smells

- **Plain-Old-Data Type Smell:** rich info codificada como floats raw. Un parámetro debería saber si es log-odds o threshold; una predicción debería saber qué modelo la produjo.
- **Multiple-Language Smell:** mezclar lenguajes encarece testing y transferencia.
- **Prototype Smell:** depender de prototipos pequeños sin un sistema full-scale robusto es señal de fragilidad.

### 3.5 Configuration Debt (sección 6)

Área sorprendentemente densa en deuda. La cantidad de líneas de configuración en sistemas ML maduros **supera** las líneas de código. Cada línea es potencial para errores.

Ejemplos típicos de complejidad config:
- "Feature A se loggeó mal del 9/14 al 9/17"
- "Feature B no está disponible antes del 10/7"
- "Feature D no está en producción → usar substitutos D' y D''"
- "Si Z se usa, los training jobs necesitan +memoria"
- "Q no se puede usar con R por latencia"

**Principios de buenos sistemas de config:**

1. Fácil expresar config como pequeño cambio de una previa.
2. Difícil cometer errores manuales, omisiones, oversights.
3. Fácil ver diff visual entre dos configs.
4. Fácil verificar automáticamente facts básicos (features usados, dependencias transitivas).
5. Posible detectar settings no usados o redundantes.
6. Configs revisadas en code review y versionadas.

### 3.6 Dealing with Changes in the External World (sección 7)

> "the external world is rarely stable. This background rate of change creates ongoing maintenance cost."

#### 3.6.1 Fixed Thresholds in Dynamic Systems

Thresholds de decisión (predict true/false, marcar spam o no) suelen elegirse manualmente para optimizar precision/recall. Si el modelo se reentrena con datos nuevos, el threshold viejo queda inválido. Manual update across many models = quebradizo.

**Mitigación:** aprender thresholds via evaluación en heldout.

#### 3.6.2 Monitoring and Testing

Unit/integration tests no son suficientes. Comprehensive live monitoring + automated response es **crítico**.

**¿Qué monitorear? El paper propone 3 puntos de partida:**

1. **Prediction Bias** — distribución de labels predichos debería igualar distribución de labels observados. Test débil (null model lo cumple) pero **diagnóstico sorprendentemente útil**. Detecta shifts del mundo. Slice por dimensiones para localizar issues.
2. **Action Limits** — sistemas que toman acciones reales (bidding, marking spam) deben tener límites enforced. Si se alcanza el límite → alerta + intervención.
3. **Up-Stream Producers** — datos vienen de procesos upstream que deben tener SLAs que tomen en cuenta las necesidades downstream. Alertas upstream se propagan al control plane del modelo.

**Respuesta debe ser en tiempo real**, no solo human paging. Invertir en automated response vale.

### 3.7 Other Areas (sección 8)

- **Data Testing Debt** — "si data reemplaza a código, y código se testea, entonces data se testea". Sanity checks + tests más sofisticados de distribuciones de input.
- **Reproducibility Debt** — algoritmos randomizados, no-determinismo de paralelismo, dependencias en initial conditions, interacciones con mundo externo.
- **Process Management Debt** — sistemas maduros tienen docenas/cientos de modelos corriendo simultáneamente. Actualizar muchos configs de modelos similares de forma segura/automática. Asignar recursos por prioridad de negocio. Detectar bloqueos en el flujo. **Anti-patrón:** procesos con muchos pasos manuales.
- **Cultural Debt** — línea dura research/engineering es contraproducente para health a largo plazo. Equipos deben **premiar** deletion de features, reducción de complejidad, reproducibilidad, estabilidad, monitoring, no solo accuracy.

---

## 4. Cómo medir la deuda — sección 9 (Conclusiones)

> "Technical debt is a useful metaphor, but it unfortunately does not provide a strict metric."

Preguntas diagnósticas que propone el paper para evaluar el estado de un sistema:

1. ¿Qué tan fácilmente puede probarse a full scale un approach algorítmico completamente nuevo?
2. ¿Cuál es la transitive closure de todas las data dependencies?
3. ¿Qué tan precisamente puede medirse el impacto de un cambio nuevo?
4. ¿Mejorar un modelo o signal degrada otros?
5. ¿Qué tan rápido un nuevo miembro del equipo se pone al día?

Insight final: pagar deuda ML requiere **shift de cultura de equipo**, no solo más herramientas. Recompensar, priorizar y reconocer este esfuerzo es esencial para health a largo plazo.

---

## 5. Aportes históricos y conceptos que el paper introdujo

| Concepto | Status hoy (2026) |
|---|---|
| Principio CACE | Mantra estándar en MLOps |
| Figura "5% ML code" | Citada en >90 % de slides intro a MLOps |
| Glue code / Pipeline jungles | Vocabulario estándar |
| Hidden feedback loops | Tema central en RecSys e RLHF research |
| Plain-Old-Data Type Smell | Antecedente directo de tools como Pydantic, Pandera para data validation |
| Static analysis of data deps | Inspiró DVC, lineage tracking en MLflow, Pachyderm |
| Monitoring prediction bias | Práctica estándar; Evidently AI, Arize, WhyLabs lo implementan |
| Configuration debt | Motivó OmegaConf, Hydra, llmconf y los paradigmas declarativos modernos |
| Cultural debt | Argumento académico clave para el rol "MLOps Engineer" como cross-functional |

---

## 6. Limitaciones del paper

- **No cuantifica** la deuda. Es taxonomía cualitativa.
- **Sesgo Google** — todos los autores son de Google, los ejemplos vienen de ads/search a escala masiva. Equipos chicos pueden tener prioridades distintas.
- **Pre-deep-learning-LLM-era** — no aborda problemas específicos de modelos grandes pretrained (data contamination, prompt engineering como configuración).
- **Pocas recetas concretas** — el paper plantea problemas más que soluciones. Las mitigaciones son a veces genéricas ("invest in tooling", "shift culture").
- **No discute observabilidad moderna** — Prometheus, distributed tracing, log aggregation aún no eran ubicuos.

---

## 7. Impacto y legado

### Citas e influencia académica

El paper es **el ancestro común** de prácticamente toda la literatura MLOps. Trabajos que lo extienden:
- Polyzotis et al. 2018 (SIGMOD): "Data lifecycle challenges in production ML" — profundiza en data validation.
- Amershi et al. 2019 (ICSE): "Software engineering for ML: a case study at Microsoft" — versión Microsoft de Sculley.
- Paleyes et al. 2022 (CSUR): "Challenges in deploying ML" — survey academic que cita Sculley constantemente, [ver análisis](Paleyes-ChallengesDeployingML-2022.md).
- Kreuzberger et al. 2023 (IEEE Access): "MLOps: Overview, definition, architecture" — operacionaliza la solución, [ver análisis](Kreuzberger-MLOpsOverview-2023.md).

### Influencia en herramientas

- **TFX (TensorFlow Extended)** — Google's response interno, hace concreto el data validation y schema enforcement.
- **MLflow** — versioning de experimentos como respuesta al config debt.
- **Feast** — feature stores como abstracción para evitar unstable data deps.
- **DVC, Pachyderm, LakeFS** — data versioning como mitigación de unstable deps.
- **Evidently, WhyLabs, Arize, Fiddler** — monitoring de prediction bias y drift.
- **Kubeflow, Flyte, Prefect** — orquestadores que reemplazan pipeline jungles con DAGs explícitos.

### Adopción del lenguaje

Términos del paper que se volvieron jerga estándar: *CACE*, *glue code*, *pipeline jungle*, *hidden feedback loop*, *undeclared consumer*, *correction cascade*, *configuration debt*.

---

## 8. Conexión con la clase 19 (Entrenamiento, Deployment y MLOps)

El PDF del prof Javier Rojas, slide 56, muestra el diagrama de tres lóbulos **Design → Development → Operations** sin profundizar **por qué** existe MLOps. Sculley 2015 da exactamente esa respuesta:

| Slide del prof | Sculley aporta |
|---|---|
| Slide 50-54 "¿Cuándo un producto de IA se considera terminado?" | El paper completo es la respuesta: nunca, por las 7 deudas. Especialmente sección 7 (Changes in External World). |
| Slide 55 "MLOps = paradigma repetible para implementar y mantener modelos confiables" | La motivación de "confiable y eficiente" viene de **toda** la sección 8 (Other areas) — reproducibility, process management, cultural debt. |
| Slide 56 lóbulos Design/Dev/Ops | Sculley explica por qué Operations no es algo opcional o posterior, sino central: el debt vive en operations. |
| Slide 57-59 Pipelines (Load Dataset → Train → Test → Deploy) | El paper advierte: si esto es un pipeline jungle con scripts ad-hoc, acabaste creando glue code masivo. Justifica frameworks como KFP/Airflow/Flyte. |
| Slide 60-61 Frameworks (Kubeflow, Airflow, Flyte, Vertex AI...) | Sculley es la justificación filosófica de **por qué** invertir en estos frameworks, aunque parezcan overhead. |

El paper también justifica conceptos que **el PDF del prof omite completamente**:

- **Monitoring de prediction bias** — el prof solo menciona "datos cambian con el tiempo".
- **CACE principle** — clave para entender por qué retraining es difícil.
- **Hidden feedback loops** — relevante en cualquier sistema de recomendación.
- **Configuration debt** — explica por qué Hydra/OmegaConf existen.
- **Dead experimental codepaths** — explica por qué feature flags + experiment tracking importan.

---

## 9. Quotes memorables para el site

> "Hidden debt is dangerous because it compounds silently."

> "CACE: Changing Anything Changes Everything."

> "It may be surprising to the academic community to know that only a tiny fraction of the code in many ML systems is actually devoted to learning or prediction."

> "Because a mature system might end up being (at most) 5% machine learning code and (at least) 95% glue code, it may be less costly to create a clean native solution rather than re-use a generic package."

> "Paying down ML-related technical debt requires a specific commitment, which can often only be achieved by a shift in team culture."

---

## 10. Lectura asociada y siguientes pasos

**Papers que extienden o complementan a Sculley 2015:**

1. **McMahan et al. 2013** *"Ad click prediction: a view from the trenches"* (KDD) — la pre-historia, citada por Sculley en sus mitigaciones.
2. **Breck et al. 2017** *"The ML test score: a rubric for ML production readiness"* — operacionaliza Sculley en un score 0-28 puntos.
3. **Polyzotis et al. 2018** *"Data lifecycle challenges in production machine learning"* (SIGMOD) — profundiza solo en data.
4. **Amershi et al. 2019** *"Software engineering for ML: a case study at Microsoft"* (ICSE).
5. **Kreuzberger 2023** — definición arquitectónica formal de MLOps. [ver análisis](Kreuzberger-MLOpsOverview-2023.md).
6. **Paleyes 2022** — survey de challenges con case studies. [ver análisis](Paleyes-ChallengesDeployingML-2022.md).

**Recursos prácticos derivados:**

- *Rules of Machine Learning* (Martin Zinkevich, Google, 2017) — 43 reglas operacionales derivadas del espíritu Sculley.
- *Designing Machine Learning Systems* (Chip Huyen, O'Reilly, 2022) — libro entero estructurado alrededor de los problemas que Sculley identifica.

**Conexión con `clase_19/teoria.md`:** este paper debería citarse en la introducción de la sección 6 (MLOps) del recorrido slide-a-slide. Es la justificación filosófica de toda la sección.
