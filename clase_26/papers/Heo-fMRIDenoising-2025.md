# Sparsely Labeled fMRI Data Denoising with Meta-Learning-Based Semi-Supervised Domain Adaptation — Análisis interno exhaustivo

## 1. Metadata y resumen ejecutivo

**Título completo:** *Sparsely Labeled fMRI Data Denoising with Meta-Learning-Based Semi-Supervised Domain Adaptation*

**Autores:** Keun-Soo Heo, Ji-Wung Han, Soyeon Bak, Minjoo Lim, Bogyeong Kang, Sang-Jun Park (Department of Artificial Intelligence, Korea University, Republic of Korea); Weili Lin (Department of Radiology, University of North Carolina at Chapel Hill, USA); Han Zhang, Dinggang Shen (School of Biomedical Engineering, ShanghaiTech University, China); Tae-Eui Kam (autor de correspondencia, Korea University, `kamte@korea.ac.kr`).

**Venue:** MICCAI 2025 (Medical Image Computing and Computer-Assisted Intervention).

**Código:** `https://github.com/KeunsooHeo/metaclean`

**El problema en una frase.** El *denoising* de fMRI mediante clasificación de componentes ICA en señal/ruido funciona muy bien cuando se entrena un modelo por dataset con etiquetas abundantes, pero se rompe al cambiar de dataset (distinto escáner, protocolo, pipeline de preprocesamiento) y al carecer de anotaciones expertas, que son caras. Peor aún: distintos centros etiquetan con criterios distintos, así que reutilizar etiquetas de otros datasets introduce un *criteria shift* que confunde al modelo.

**La idea central.** Un framework de *semi-supervised domain adaptation* basado en *meta-learning* que separa dos preocupaciones que normalmente van mezcladas en una sola red:
1. **Un feature extractor "dataset-irrelevant"** (invariante al dataset), entrenado con meta-learning (estilo MAML) sobre múltiples datasets etiquetados (*source domains*) más el dataset escasamente etiquetado (*target domain*). Captura patrones de ruido transversales.
2. **Clasificadores "dataset-specific"** (uno por dataset), entrenados con *decoupled training* (congelando el extractor), de modo que cada clasificador absorbe el criterio de etiquetado idiosincrático de su dataset sin contaminar la representación compartida.

**Resultados clave (reales, de la Tabla 2).** En cuatro datasets de fMRI (HCP, BCP, WHII-MB6, WHII-STD), tanto con 10% de etiquetas (*sparsely labeled*) como con 100% (*fully labeled*), la configuración completa (Setting C, con los tres componentes PT+ML+DT) alcanza la mejor accuracy en la mayoría de datasets. Un resultado notable: en WHII-MB6, el modelo propuesto con **solo 10% de etiquetas** (F1 = 95.60%, GM = 98.13%) **supera al baseline entrenado con 100%** de etiquetas (F1 = 95.18%, GM = 96.77%). En el caso extremo de WHII-STD al 10%, el baseline colapsa por completo en sensibilidad (SEN = 0.00%, no detecta ninguna señal), mientras que el método propuesto recupera SEN = 91.12%.

**Por qué es relevante para Roberto.** El "criteria shift entre datasets/centros" es exactamente el problema que tú enfrentas en armonización de datos clínicos multi-institucionales: distintos hospitales codifican el mismo concepto con criterios distintos. La arquitectura "representación compartida invariante + cabezas específicas por fuente" es un patrón directamente transportable a escenarios FHIR multi-centro y a tu trabajo de MDM/matching en FALP. Lo desarrollo en la sección 11.

---

## 2. El problema clínico: denoising de fMRI resting-state

### Qué es la señal y qué es el ruido

La resonancia magnética funcional (fMRI) mide la señal **BOLD** (Blood-Oxygen-Level-Dependent), un proxy hemodinámico de la actividad neuronal. Es una de las técnicas más usadas para estudiar los *functional connectomes* del cerebro (las referencias [2,3,13,15] del paper). El problema es que la señal BOLD tiene una relación señal-ruido (SNR) baja y está contaminada por fuentes no neuronales:

- **Movimiento de cabeza** del sujeto (especialmente severo en poblaciones difíciles: bebés —ver dataset BCP— o adultos mayores).
- **Fluctuaciones fisiológicas** no neuronales: ciclo cardiaco, respiración, pulsatilidad vascular (el paper cita [2,7,20,21]).
- **Artefactos del escáner**: deriva de baja frecuencia, ruido térmico, inhomogeneidades de campo, artefactos específicos del hardware.

El paper lo enuncia directamente: *"fMRI data are easily affected by noise originating from experimental environments, scanner artifacts, and non-neuronal physiological fluctuations"*, y este ruido *"degrades the signal-to-noise ratio (SNR) and reduces the statistical power of fMRI studies in interpreting brain functions"* (referencia [17]).

### Por qué importa para conectividad

En resting-state fMRI no hay tarea: el sujeto solo descansa. El análisis de interés es la **conectividad funcional** — correlaciones temporales entre regiones cerebrales que definen redes. El ruido de movimiento es especialmente insidioso aquí porque introduce **correlaciones espurias** que se confunden con conectividad real, sesgando todo el análisis downstream (diagnóstico de MCI temprano, ASD, etc., que son las aplicaciones que citan los propios autores en [13,15,28]). De ahí que el denoising sea un paso de preprocesamiento esencial, no opcional.

### El enfoque clásico: ICA + clasificación de componentes

El método dominante descompone el fMRI mediante **ICA espacial** (Independent Component Analysis). El paper explica que la ICA espacial *"decomposes fMRI data into independent components (ICs), which consist of pairs of a 3D spatial map and its corresponding 1D time series"*. Es decir, cada componente independiente (IC) es un par:

- **Mapa espacial 3D**: dónde en el cerebro ocurre el patrón.
- **Serie temporal 1D**: cómo evoluciona ese patrón en el tiempo.

El denoising se reduce entonces a un **problema de clasificación binaria**: etiquetar cada IC como *signal* (actividad neuronal genuina) o *noise* (movimiento, fisiología, artefacto), y luego reconstruir el fMRI limpio descartando los componentes de ruido (regresión de las series temporales de ruido).

Históricamente:
- **FIX** (referencia [24], Salimi-Khorshidi et al.) clasifica ICs con clasificadores de machine learning entrenados sobre *features* espaciales y temporales hechas a mano (*hand-crafted*).
- **ICA-AROMA** (referencia [22], Pruim et al.) usa patrones de ruido predefinidos para artefactos de movimiento.
- Métodos recientes de **deep learning** ([11,14,19]) usan CNNs para extracción automática de features y clasificación. El paper [19] (Lim et al., 2024) — *"A unified multi-modality fusion framework for deep spatio-spectral-temporal feature learning"* — es el **baseline directo** de este trabajo y aporta la arquitectura del feature extractor.

### El cuello de botella: anotación cara y criterios variables

Aquí está el punto crítico que motiva todo el paper. Dos problemas encadenados:

1. **Anotar señal/ruido requiere expertos y es caro.** El paper: *"applying previous approaches to new datasets requires extensive expert annotations for signal/noise ICs, which demand significant time and cost"*. Un experto debe revisar manualmente cientos o miles de componentes por dataset (la Tabla 1 muestra entre 2,585 y 22,877 ICs por dataset).

2. **Los criterios de etiquetado VARÍAN entre datasets/centros.** Esta es la observación más fina y clínicamente realista: *"labeling criteria for signal/noise annotations vary depending on research objectives and expert judgment [7,11,27], leading to inconsistencies across datasets"*. Lo que un centro etiqueta como ruido marginal, otro lo puede etiquetar como señal débil, dependiendo del objetivo del estudio y del juicio del anotador. Esto no es ruido aleatorio en las etiquetas: es un **sesgo sistemático por dataset** (un *criteria shift*).

Estas dos dificultades, sumadas a la *inter-dataset variation* (escáner, protocolo, pipeline), son las que hacen que un modelo entrenado en un dataset no transfiera bien a otro.

---

## 3. El setup: sparsely labeled + source datasets con inter-dataset variation

### Definición del escenario

El paper plantea un problema de **semi-supervised domain adaptation** con la siguiente estructura:

- **Source domains** (D datasets): datasets *fully labeled*, con anotaciones expertas completas de señal/ruido. Funcionan como conocimiento previo a explotar.
- **Target domain** (1 dataset): el dataset nuevo que queremos limpiar, **sparsely labeled** — solo una fracción pequeña de sus ICs está etiquetada (en los experimentos, **10%**), más una gran mayoría sin etiquetar.

El término *semi-supervised* aplica al target: combina un pequeño *labeled set* con un gran *unlabeled set*. El término *domain adaptation* aplica al hecho de que source y target provienen de distribuciones distintas (distinto escáner/protocolo) **y además** con etiquetas potencialmente bajo criterios distintos.

### Por qué el transfer learning ingenuo falla

El paper distingue su escenario de los dos enfoques previos y por qué ambos fallan:

- **Modelos dataset-specific** ([19]): entrenas un modelo por dataset. Funciona si tienes muchas etiquetas, pero no aprovecha datos de otros datasets y exige anotación completa en cada nuevo dataset.
- **Predefined noise patterns** ([22]): reglas fijas que no generalizan a nuevas características de ruido.

La intuición del fracaso del transfer simple está en los datos del propio paper: en la condición fully-labeled, el Setting (A) — que es esencialmente *pretraining conjunto* sobre todos los datasets — a veces **empeora** respecto al baseline dataset-specific. Cita textual: *"jointly training with the other datasets does not always enhance performance due to variations in noise characteristics and inconsistencies in labeling criteria. For instance, in the HCP dataset, Setting (A) slightly reduces accuracy (99.12%) compared to the baseline (99.31%)"*.

Esto es la evidencia empírica de que mezclar datasets sin manejar el criteria shift es activamente dañino: el modelo recibe señales de supervisión contradictorias (mismo input, etiquetas opuestas según el dataset de origen). De ahí la necesidad de **separar** lo invariante de lo específico.

---

## 4. La idea central: feature extractor dataset-irrelevant + clasificadores dataset-specific

La hipótesis arquitectónica del paper es una factorización limpia del problema:

> **Lo que es común a todos los datasets** (cómo se ve el ruido de movimiento, la fisiología, los artefactos) **debe vivir en una representación compartida e invariante al dataset.** **Lo que difiere entre datasets** (el criterio de etiquetado, dónde se traza la frontera señal/ruido) **debe vivir en cabezas separadas, una por dataset.**

Concretamente, dos componentes:

1. **Dataset-irrelevant feature extractor** (parámetros $\theta$, basado en la CNN de [19]): mapea cada IC a una representación de features. Se entrena con **meta-learning** para que capture *"a wide range of noise patterns across multiple well-labeled datasets and a sparsely labeled dataset"*. El meta-learning es el mecanismo para forzar invarianza: en lugar de minimizar el error promedio sobre la mezcla de datasets (que produce un compromiso sesgado), aprende una inicialización que **se adapta rápido a cualquier dataset**, lo que empuja la representación hacia lo transversal.

2. **Dataset-specific classifiers** (parámetros $\phi^1, \dots, \phi^D, \psi$): $D+1$ clasificadores, uno por cada source domain ($\phi^d$) y uno para el target ($\psi$). Cada uno es una red *fully connected* de una sola capa. Se entrenan con **decoupled training**: el feature extractor se **congela** y cada clasificador se optimiza por separado con las etiquetas de su propio dataset. Así cada cabeza absorbe el criterio de etiquetado de su dataset sin alterar la representación compartida.

La elegancia del diseño está en que la representación nunca tiene que "elegir" entre los criterios contradictorios de los distintos datasets — esa decisión se delega a las cabezas. La representación solo se preocupa de separar bien las clases de manera que cualquier cabeza pueda trazar su propia frontera.

---

## 5. La arquitectura de 3 etapas (Fig. 1)

El framework opera en tres etapas, descritas en la Sección 2 del paper y resumidas en la Fig. 1.

### Etapa A — Pretraining del feature extractor y clasificador (Sección 2.1)

Se entrena un único feature extractor (parámetros $\omega$) y un único clasificador (parámetros $\mu$) usando los **source domains fully labeled** más el **labeled set del target**. La función de pérdida es **cross-entropy** ($\mathcal{L}_{cls}$). El propósito es inicializar de forma estable.

- El feature extractor está basado en la arquitectura CNN de [19].
- El clasificador $\mu$ es una red fully-connected de una sola capa que predice signal vs noise.
- Tras $M$ pasos de entrenamiento (en los experimentos $M = 10$), los parámetros del extractor $\omega_M$ se fijan como $\theta_0$ (inicialización para la etapa de meta-learning).
- Los parámetros del clasificador $\mu_M$ inicializan los $D+1$ clasificadores dataset-specific: $\phi^1_0, \dots, \phi^D_0, \psi_0$.

Cita: *"Incorporating labeled data at this stage reduces ambiguity in signal/noise annotations and offers a stable initialization for subsequent processing stages."*

### Etapa B — Meta-learning del feature extractor dataset-irrelevant (Sección 2.2)

Aquí está el corazón del método, basado en **MAML** (Model-Agnostic Meta-Learning, Finn et al. [5]). Se forma un **task** por cada emparejamiento (source domain $d$, target domain), dando $D$ pares/tasks. El proceso itera $K$ veces (en los experimentos $K = 300$) con dos bucles anidados:

- **Inner loop**: para el par $d$, se entrenan parámetros task-specific $\tilde{\theta}^d_k$ minimizando dos pérdidas: la de clasificación $\mathcal{L}_{cls}$ (cross-entropy) y la de **alineación** $\mathcal{L}_{align}$ (contrastive-based loss [16], *Supervised Contrastive Learning* de Khosla et al.). Usa el **support set** del source $d$ y el **unlabeled set** del target.

- **Outer loop**: una vez obtenidos todos los $\tilde{\theta}^d_k$ ($d = 1,\dots,D$) en el inner loop, se actualizan los parámetros del feature extractor $\theta_k$ validando todos los parámetros task-specific con el **query set** del source y el **labeled set** del target, de nuevo minimizando $\mathcal{L}_{cls} + \mathcal{L}_{align}$.

El paper justifica: *"The meta-learning approach enables robust adaptation to sparsely labeled data under limited supervision [6,9]"* (cita su propio trabajo previo Meta-EEG [9] y *Generalized inner loop meta-learning* [6]).

### Etapa C — Decoupled training de los clasificadores dataset-specific (Sección 2.3)

Tras obtener el feature extractor invariante, se optimizan los $D+1$ clasificadores ($\phi^1,\dots,\phi^D,\psi$) para manejar las inconsistencias de etiquetado:

- **El feature extractor se CONGELA** (*"we freeze the feature extractor to prevent interference with dataset-irrelevant feature extraction"*, referencia [29] — *Decoupled training* de Wang et al., AAAI 2024).
- Cada clasificador se actualiza **independientemente**, minimizando $\mathcal{L}_{cls}$ por separado para cada dataset, usando los source domains y el labeled set del target.

Contraste explícito con el baseline [11,19]: *"Unlike previous approaches that jointly update feature extraction and classification, we train the dataset-specific classifiers independently from the feature extractor to accommodate dataset-specific labeling criteria."*

### Alternancia iterativa

El framework **alterna** entre Etapa B (meta-learning) y Etapa C (decoupled training) durante $K$ iteraciones, mejorando progresivamente la clasificación mientras refuerza la adaptación de los source fully-labeled al target sparsely-labeled *"without compromising the dataset-irrelevant feature extractor"*.

> Nota sobre la Fig. 1: el texto extraído del PDF no incluye el diagrama, solo su descripción (caption). La descripción de las tres etapas anterior se basa íntegramente en el texto de las Secciones 2.1–2.3, que sí está completo.

---

## 6. La matemática del meta-learning

El paper describe el meta-learning verbalmente y con notación de parámetros, pero **no escribe las ecuaciones explícitas de actualización** (no hay ecuaciones numeradas en el texto extraído). Reconstruyo aquí la formulación estándar de MAML aplicada a esta notación, dejando claro qué es del paper y qué es la formulación canónica de MAML [5] que el paper invoca.

### Notación del paper (literal)

- $\theta$: parámetros del feature extractor dataset-irrelevant. $\theta_0 = \omega_M$ (inicializado del pretraining), $\theta_k$ tras la iteración $k$.
- $\tilde{\theta}^d_k$: parámetros task-specific para el par $d$ en el paso $k$ (resultado del inner loop).
- $\phi^d$: clasificador del source domain $d$. $\psi$: clasificador del target.
- $\mathcal{L}_{cls}$: cross-entropy. $\mathcal{L}_{align}$: contrastive-based loss [16].

### Inner loop (formulación MAML estándar, consistente con el texto)

Para cada par $d$, partiendo de $\theta_k$ y usando el support set del source $\mathcal{S}^d$ y el unlabeled set del target $\mathcal{U}^t$:

$$
\tilde{\theta}^d_k = \theta_k - \alpha \, \nabla_{\theta} \Big[ \mathcal{L}_{cls}\big(\mathcal{S}^d; \theta_k, \phi^d\big) + \lambda \, \mathcal{L}_{align}\big(\mathcal{S}^d, \mathcal{U}^t; \theta_k\big) \Big]
$$

donde $\alpha$ es el learning rate del inner loop (en los experimentos el learning rate global es 0.01) y $\lambda$ pondera la alineación. El paper confirma que el inner loop minimiza *"both the classification loss and the alignment loss using the support set of the dth source domain and the unlabeled set of the target domain"*.

### Outer loop (meta-update)

El feature extractor se actualiza evaluando los parámetros adaptados $\tilde{\theta}^d_k$ sobre el query set del source $\mathcal{Q}^d$ y el labeled set del target $\mathcal{L}^t$:

$$
\theta_{k+1} = \theta_k - \beta \, \nabla_{\theta} \sum_{d=1}^{D} \Big[ \mathcal{L}_{cls}\big(\mathcal{Q}^d, \mathcal{L}^t; \tilde{\theta}^d_k, \phi^d, \psi\big) + \lambda \, \mathcal{L}_{align}\big(\mathcal{Q}^d, \mathcal{L}^t; \tilde{\theta}^d_k\big) \Big]
$$

con $\beta$ el learning rate del outer loop. El texto confirma: *"the outer loop updates the parameters of the feature extractor $\theta_k$ by validating all task-specific parameters using the query set of the source domain and the labeled set of the target domain, while minimizing both the classification loss and the alignment loss"*.

> Importante: los símbolos $\alpha$, $\beta$, $\lambda$ y la división support/query son la formulación canónica de MAML que el paper invoca al citar [5]; el texto extraído no los define explícitamente con esas letras. Las pérdidas $\mathcal{L}_{cls}$ y $\mathcal{L}_{align}$ y la estructura inner/outer sí son del paper.

### Por qué esto separa lo invariante de lo específico

La intuición matemática del meta-update:

- **Cross-entropy ($\mathcal{L}_{cls}$)** empuja a clasificar bien señal/ruido en cada dataset.
- **Alignment loss contrastiva ($\mathcal{L}_{align}$)** — la *Supervised Contrastive Loss* [16] — acerca en el espacio de features las muestras de la misma clase y aleja las de clases distintas, **a través** del source etiquetado y el target sin etiquetar. Esto es lo que alinea las distribuciones source/target (la parte "domain adaptation") y permite usar el unlabeled set del target.
- **La estructura bilevel de MAML** hace que el outer loop no optimice $\theta$ para un dataset concreto sino para que $\theta$ **se adapte bien a cualquier dataset tras unos pocos pasos del inner loop**. Esto es lo que produce la invarianza: $\theta$ no encarna ningún criterio de etiquetado particular, sino una representación desde la cual cualquier criterio es fácilmente alcanzable.

La factorización final es entonces: $\theta$ (compartido, invariante) + $\{\phi^d, \psi\}$ (cabezas, específicas por criterio). El gradiente del decoupled training (Etapa C) nunca toca $\theta$ (congelado), por lo que los criterios contradictorios no pueden corromper la representación.

---

## 7. Experimentos

### Datasets (Tabla 1)

Cuatro datasets de fMRI de estudios distintos, elegidos precisamente para exhibir *inter-dataset variation*:

| Dataset | Escáner | TR (s) | Resolución (mm³) | # Sujetos | Edad | # Samples | # ICs | # Signal ICs | # Noise ICs |
|---|---|---|---|---|---|---|---|---|---|
| **HCP** [26] (Human Connectome Project) | 3T Siemens Connectome Skyra | 0.7 | 2×2×2 | 25 | 22–35 años | 100 | 22,877 | 2,326 | 20,551 |
| **BCP** [12] (Baby Connectome Project) | 3T Siemens Prisma | 0.8 | 2×2×2 | 32 | 379±186 días | 99 | 14,850 | 3,406 | 11,444 |
| **WHII-MB6** [4] (Whitehall II Multi-band) | 3T Siemens Magnetom Verio | 1.3 | 2×2×2 | 25 | 63–75 años | 39 | 5,143 | 795 | 4,348 |
| **WHII-STD** [4] (Whitehall II Standard) | 3T Siemens Magnetom Verio | 3.0 | 3×3×3 | 45 | 63–75 años | 40 | 2,585 | 422 | 2,783 |

Observaciones clave que justifican el problema:
- **Rango etario extremo**: bebés (~1 año en BCP) vs. adultos jóvenes (HCP) vs. adultos mayores (Whitehall, 63–75). El movimiento y la fisiología cambian radicalmente.
- **Resolución temporal (TR) muy distinta**: de 0.7 s (HCP) a 3.0 s (WHII-STD) — un factor de ~4.
- **Resolución espacial distinta**: 2×2×2 vs 3×3×3 mm³.
- **Fuerte desbalance de clases**: en todos, los noise ICs superan ampliamente a los signal ICs (en HCP, 20,551 ruido vs 2,326 señal — ratio ~9:1; en WHII-STD, 2,783 vs 422 — ratio ~6.6:1). Por eso F1 y G-measure son métricas necesarias, no solo accuracy.

### Setup experimental

- **5-fold cross-validation**, particionando uniformemente las muestras en cinco folds.
- **Métricas** (evaluadas por sample [18]): accuracy (ACC), sensitivity (SEN), specificity (SPEC), F1-score (F1), G-measure (GM).
  - **SEN** = detección de señal ICs (la clase minoritaria, lo difícil).
  - **SPEC** = detección de noise ICs (la clase mayoritaria, lo fácil).
  - **F1 y GM** dan evaluación balanceada bajo desbalance de clases. GM (G-measure) es típicamente la media geométrica de sensibilidad y especificidad, robusta al desbalance.
- **Hiperparámetros**: batch size = 12, $M = 10$ iteraciones de pretraining, dimensión de features = 96, $K = 300$ ciclos de alternancia meta-learning/decoupled, learning rate = 0.01.

### Configuraciones evaluadas

Estudio incremental que añade componentes uno a uno:
- **Baseline [19]**: el modelo dataset-specific previo (sin PT, ML ni DT).
- **Setting (A)**: + Pretraining conjunto (PT).
- **Setting (B)**: + Meta-learning del feature extractor (ML).
- **Setting (C)**: + Decoupled training de clasificadores (DT) — el framework completo.

Significancia estadística vs. Setting (C) con **test de Wilcoxon signed-rank** [30] (`*p < 0.05`, `**p < 0.001`).

### Resultados — condición sparsely labeled (10%)

Números reales de la Tabla 2 (Setting C en negrita por ser el mejor):

| Target | Método | ACC (%) | SEN (%) | SPEC (%) | F1 (%) | GM (%) |
|---|---|---|---|---|---|---|
| **HCP** | Baseline | 98.48 | 91.61 | 99.26 | 92.36 | 95.25 |
| | Setting (C) | **98.93** | **97.82** | 99.06 | **94.99** | **98.42** |
| **BCP** | Baseline | 96.13 | 86.21 | 98.33 | 88.74 | 91.63 |
| | Setting (C) | **96.60** | 90.01 | 97.92 | **90.55** | 93.61 |
| **WHII-MB6** | Baseline | 93.89 | 75.70 | 95.82 | 78.01 | 82.85 |
| | Setting (C) | **98.83** | **97.19** | 99.13 | 95.60 | **98.13** |
| **WHII-STD** | Baseline | 86.03 | **0.00** | 100.00 | **0.00** | **0.00** |
| | Setting (C) | **97.67** | **91.12** | 98.90 | **91.07** | **94.79** |

El resultado más dramático: en **WHII-STD al 10%**, el baseline obtiene **SEN = 0.00%, F1 = 0.00%, GM = 0.00%** — el modelo clasifica TODO como ruido (de ahí SPEC = 100%) porque con pocas etiquetas y desbalance brutal aprende el atajo trivial de la clase mayoritaria. El framework propuesto recupera SEN = 91.12% y F1 = 91.07%. Esto ilustra exactamente para qué sirve el método: rescatar la clase minoritaria (señal) en condiciones de escasez de etiquetas.

### Resultados — condición fully labeled (100%)

| Target | Método | ACC (%) | SEN (%) | SPEC (%) | F1 (%) | GM (%) |
|---|---|---|---|---|---|---|
| **HCP** | Baseline | 99.31 | 97.07 | 99.57 | 96.71 | 98.29 |
| | Setting (C) | **99.31** | **98.10** | 99.47 | **96.85** | **98.77** |
| **BCP** | Baseline | 96.02 | 91.67 | 96.78 | 89.48 | 93.96 |
| | Setting (C) | **96.69** | 90.18 | 98.00 | **90.56** | 93.71 |
| **WHII-MB6** | Baseline | 98.66 | 94.40 | 99.31 | 95.18 | 96.77 |
| | Setting (C) | **99.07** | **96.92** | **99.60** | **97.37** | **98.23** |
| **WHII-STD** | Baseline | 97.81 | 87.35 | 99.65 | 91.27 | 92.98 |
| | Setting (C) | **98.52** | **94.14** | 99.32 | **94.20** | **96.60** |

El framework completo gana en ACC, SEN, F1 y GM en la mayoría de datasets incluso con etiquetas completas, demostrando que el beneficio no es solo "tapar la falta de etiquetas" sino mejorar la representación per se.

### El resultado estrella: 10% supera al 100% del baseline

Cita literal: *"In the WHII-MB6 dataset, Setting (C) in the 10% labeled condition achieves 95.60% F1-score and 98.13% G-measure, outperforming 95.18% F1-score and 96.77% G-measure of the baseline in the 100% labeled condition."*

Es decir: el método con **una décima parte de las etiquetas** supera al baseline con **todas** las etiquetas en WHII-MB6 (en F1 y GM). El conocimiento transferido desde los otros datasets vía meta-learning compensa con creces la falta de etiquetas locales.

### Evaluación visual (Fig. 2)

Comparación cualitativa sobre una muestra del HCP al 10% labeled. Tres modalidades de input visualizadas: (a) mapa espacial del IC, (b) serie temporal del IC, (c) imagen wavelet-transformada (generada de la serie temporal con un kernel wavelet aprendible de [19]). Se usan mapas de explicación **Grad-CAM** [25]. Ambos modelos detectan clusters positivos en la corteza visual primaria, pero **el baseline clasifica erróneamente un signal IC como noise**, mientras el framework propuesto lo identifica correctamente como señal — evidencia cualitativa de la mejor adaptación al dataset escasamente etiquetado.

---

## 8. Ablations

El estudio incremental de la Tabla 2 ES el ablation study. Desglose del aporte de cada componente:

### Aporte del pretraining conjunto (Setting A vs Baseline)

Añadir PT generalmente mejora la sensibilidad respecto al baseline dataset-specific, **especialmente en condición sparse**. Ejemplo extremo: en WHII-STD al 10%, pasar del baseline (SEN = 0.00%) a Setting (A) recupera SEN = 93.95% — solo con incorporar datos de otros datasets al pretraining ya se evita el colapso. Pero PT por sí solo es inconsistente: en fully-labeled HCP, Setting (A) **empeora** la accuracy (99.12% vs baseline 99.31%) porque mezclar criterios contradictorios sin manejarlos daña. Este es precisamente el argumento para necesitar ML y DT.

### Aporte del meta-learning (Setting B vs A)

Añadir ML refuerza la robustez. En sparse, *"Setting (B) further enhances robustness, achieving the highest sensitivity (SEN = 91.08%) in the BCP dataset"*. En fully-labeled, *"Setting (B) further improves sensitivity and F1-score across most datasets, particularly in the BCP dataset, where sensitivity increases to 92.58% and F1-score increases to 91.04%"*. El meta-learning convierte la mezcla cruda de datasets (PT) en una representación más transversal.

### Aporte del decoupled training (Setting C vs B)

DT es el que cierra el círculo: *"Setting (C) shows the outstanding performance, achieving the highest accuracy across datasets"*. Al darle a cada dataset su propia cabeza, resuelve las inconsistencias de criterio que ML por sí solo no podía absorber. Mejoras notables en WHII-STD fully-labeled (SEN = 94.14%, GM = 96.60%). La significancia estadística se mide siempre **respecto a Setting (C)**: muchas entradas del baseline, A y B llevan `*` o `**`, indicando que C es significativamente mejor.

### Efecto de la fracción de etiquetas

El paper evalúa dos fracciones: **10%** (sparse) y **100%** (full). No reporta un barrido continuo de fracciones intermedias (no hay datos de 1%, 25%, 50%, etc. en el texto extraído). El hallazgo cualitativo es que **el beneficio relativo del framework es mucho mayor en la condición sparse** — donde el baseline colapsa (WHII-STD 0% sensibilidad) — que en la full, donde las mejoras son más modestas pero consistentes. La conclusión práctica: cuanto más escasas las etiquetas, más vale el método.

> Nota: el texto extraído no contiene un ablation separado de $\mathcal{L}_{align}$ vs $\mathcal{L}_{cls}$, ni una ablación de la dimensión de features (96) o de $K$ (300). Si esos análisis existen, no están en el texto disponible.

---

## 9. Por qué importa

Este paper es un caso de estudio limpio de **meta-learning para semi-supervised domain adaptation bajo criteria shift** — un problema endémico de los datos clínicos multi-centro, no exclusivo de fMRI:

1. **Label noise estructurado (criteria shift), no aleatorio.** La mayoría de la literatura de label noise asume ruido aleatorio o dependiente de la clase. Aquí el "ruido" es un **sesgo sistemático por origen de datos** (cada centro etiqueta con su criterio). El paper muestra que la respuesta correcta no es "limpiar las etiquetas" sino **factorizar la arquitectura**: representación compartida invariante + cabezas específicas por fuente. Esto es generalizable a cualquier dominio donde varios anotadores/instituciones etiquetan con criterios distintos.

2. **Aprovechar datos legacy abundantes para arrancar datasets nuevos con poca anotación.** El patrón "tengo datasets viejos bien etiquetados y un dataset nuevo casi sin etiquetar" es ubicuo en salud (cada vez que llega un escáner nuevo, un protocolo nuevo, una población nueva). Reducir la anotación experta de 100% a 10% manteniendo (o superando) el rendimiento tiene impacto económico directo.

3. **Separación de invarianza vs especificidad como principio de diseño.** El congelamiento del extractor durante el decoupled training (Etapa C) es una decisión arquitectónica deliberada para impedir que los criterios contradictorios corrompan la representación. Es un patrón transferible: "no dejes que las cabezas específicas reescriban la columna vertebral compartida".

---

## 10. Limitaciones

Limitaciones reales, algunas reconocidas en el paper y otras inferidas de su alcance:

1. **Específico de fMRI/ICA denoising.** El método opera sobre ICs (pares mapa espacial 3D + serie temporal 1D) producidos por ICA espacial. Hereda toda la dependencia del paso de ICA previo: si la descomposición ICA es mala, la clasificación posterior no lo arregla. No es un método de denoising end-to-end del fMRI crudo.

2. **Dependencia de los datasets source.** El framework necesita datasets source *fully labeled* de buena calidad para transferir. Si los source son escasos, mal etiquetados, o demasiado distintos del target, el meta-learning tiene poco que explotar. El método es "semi-supervisado en el target" pero "fuertemente supervisado en los sources".

3. **Complejidad del pipeline de 3 etapas.** Pretraining + meta-learning bilevel (MAML, costoso por los gradientes de segundo orden / inner loop) + decoupled training alternado por $K = 300$ ciclos. Es notablemente más complejo de implementar, depurar y ajustar que un clasificador end-to-end. MAML es conocido por su sensibilidad a hiperparámetros y su costo computacional (Hessianos o aproximaciones first-order).

4. **Solo dos fracciones de etiquetas evaluadas (10% y 100%).** No hay caracterización fina de la curva rendimiento-vs-fracción. No sabemos dónde está el punto de quiebre (¿funciona al 1%? ¿al 5%?).

5. **Escala modesta de datasets.** Entre 39 y 100 samples por dataset, 25–45 sujetos. Son tamaños típicos de neuroimagen pero pequeños para conclusiones de generalización fuertes. La validación es 5-fold CV, no un test set externo independiente totalmente held-out a nivel de centro.

6. **Cuatro datasets, todos Siemens 3T.** Aunque hay variación de escáner/protocolo/edad, los cuatro escáneres son Siemens a 3 Tesla. No hay evidencia sobre GE/Philips ni sobre 7T o 1.5T. La generalización a un universo de escáneres realmente heterogéneo queda sin demostrar.

7. **$D+1$ clasificadores escala linealmente con el número de fuentes.** Y para un dataset target completamente nuevo en inferencia, hay que entrenar su cabeza $\psi$ con el labeled set disponible — no es zero-shot puro.

---

## 11. Conexión con la Clase 26 y con el trabajo de Roberto

### Con la Clase 26 (aplicaciones de IA en medicina, MICCAI 2025)

Este es un paper MICCAI 2025 emblemático del giro de la comunidad de medical imaging hacia los problemas **operacionales** del despliegue clínico, más allá de la arquitectura del modelo:

- **El cuello de botella ya no es el modelo, es la anotación y la heterogeneidad multi-centro.** Igual que los otros papers de la clase abordan eficiencia de datos, robustez de dominio y label efficiency, este ataca de frente el costo de la anotación experta y la variación entre sitios.
- **Meta-learning + domain adaptation** como herramientas de eficiencia de datos en salud, donde las etiquetas expertas son el recurso más caro.
- **Explicabilidad integrada** (Grad-CAM en la Fig. 2) como requisito de facto en medical imaging.

### Con tu trabajo, Roberto: el criteria shift es tu problema de armonización FHIR multi-institucional

Aquí está el paralelo que más te sirve. El "criteria shift entre datasets" del paper es **estructuralmente idéntico** a la heterogeneidad de codificación/criterios entre sistemas de salud que enfrentas a diario:

- **El paper:** distintos centros etiquetan el mismo IC como signal/noise según su criterio experto. El paper menciona explícitamente el problema multi-sitio citando trabajos de *bias en datasets de neuroimagen multi-sitio* [27] y *statistical harmonization para corregir site effects* [31] — exactamente el vocabulario de "armonización de datos".
- **Tu mundo FHIR:** distintas instituciones codifican el mismo concepto clínico con criterios distintos — un mismo diagnóstico mapeado a códigos ICD/SNOMED/CIE-10 diferentes según las reglas locales del hospital; el mismo `Observation` con unidades o rangos de referencia distintos; el mismo concepto de "alergia activa" definido con criterios institucionales divergentes.

**La lección arquitectónica transferible a tu trabajo de MDM/matching en FALP:**

El patrón "**representación compartida invariante al origen + cabezas específicas por institución**" es directamente aplicable a tu pipeline de patient matching y armonización. Recuerda que en tu arquitectura MDM tienes un bi-encoder como blocker y un GBM (XGBoost ONNX) como scorer principal. La factorización del paper sugiere un diseño análogo:

- **Feature extractor dataset-irrelevant** ↔ el **embedding/representación de paciente invariante a la institución** que tu blocker debería producir: que dos registros del mismo paciente de hospitales distintos caigan cerca en el espacio latente, independientemente de las convenciones de codificación locales. El meta-learning aquí sería una receta para entrenar ese embedding a ser robusto al "site effect" institucional.
- **Clasificadores dataset-specific (cabezas $\phi^d$)** ↔ **reglas/scorers específicos por institución** que capturan las idiosincrasias de codificación de cada sistema fuente, sin contaminar la representación compartida. El decoupled training (congelar la columna, ajustar solo las cabezas) es justo lo que querrías para incorporar un hospital nuevo sin re-entrenar todo el modelo de matching.
- **El escenario sparsely labeled** ↔ tu realidad: tienes datasets de matching ya curados (pares verificados) de algunos sistemas, y un sistema nuevo con poquísimas verificaciones manuales (caras, requieren un experto en MPI/MDM). El framework muestra cómo arrancar el sistema nuevo con 10% de etiquetas explotando los antiguos.

Y conecta con tu memoria sobre el paper de **retornos decrecientes de ML en MDM LATAM**: este paper de Heo et al. refuerza el matiz de que el valor del ML está en escenarios de **escasez de etiquetas y heterogeneidad multi-sitio**, no en exprimir el último 0.1% de accuracy con etiquetas abundantes. Donde el baseline ya tenía 99.31% con etiquetas completas (HCP fully-labeled), la ganancia del framework es marginal; el valor real aparece donde hay escasez (WHII-STD 10%, donde rescata de 0% a 91% de sensibilidad). Mismo principio que tu argumento de retornos decrecientes: invertir el esfuerzo de ML donde la curva es empinada (datos escasos/heterogéneos), no donde ya es plana.

**Una advertencia honesta para el traslado:** el método de Heo es costoso (MAML bilevel, pipeline de 3 etapas, $K=300$ ciclos) y depende de sources fully-labeled de calidad. Para tu caso de FALP, el principio de diseño (factorizar invariante/específico, congelar la columna al añadir instituciones) es lo valioso; la maquinaria exacta de MAML probablemente sea sobre-ingeniería frente a tu GBM + reglas, que ya separan razonablemente la representación (embedding blocker) del scoring específico.
