---
title: "fMRI Denoising (Meta-Learning Domain Adaptation)"
weight: 270
math: true
---

{{< paper-card
    title="Sparsely Labeled fMRI Data Denoising with Meta-Learning-Based Semi-Supervised Domain Adaptation"
    authors="Keun-Soo Heo, Ji-Wung Han, Soyeon Bak, et al."
    year="2025"
    venue="MICCAI 2025"
    pdf="/papers/fmri-denoising-heo-2025.pdf" >}}
El *denoising* de fMRI por clasificación de componentes ICA (señal vs. ruido) funciona muy bien con un modelo por dataset y etiquetas abundantes, pero se rompe al cambiar de centro (otro escáner, protocolo, pipeline) y, sobre todo, al carecer de anotaciones expertas. El golpe extra: distintos centros etiquetan con **criterios distintos**, así que reutilizar etiquetas de otros datasets introduce un *criteria shift* que confunde al modelo. La propuesta es un framework de *semi-supervised domain adaptation* basado en meta-aprendizaje que **factoriza** el problema: un extractor de *features* invariante al dataset (meta-aprendido) más clasificadores específicos por dataset (entrenados con la columna congelada). Código en `github.com/KeunsooHeo/metaclean`.
{{< /paper-card >}}

---

## El problema: denoising de fMRI y el costo de anotar señal/ruido

La resonancia magnética funcional (fMRI) mide la señal **BOLD** (*Blood-Oxygen-Level-Dependent*), un proxy hemodinámico de la actividad neuronal y la base para estudiar los *functional connectomes* del cerebro. El problema es que la señal BOLD tiene una relación señal-ruido (SNR) baja y está contaminada por fuentes no neuronales: movimiento de cabeza (severo en poblaciones difíciles como bebés o adultos mayores), fluctuaciones fisiológicas (ciclo cardiaco, respiración, pulsatilidad vascular) y artefactos del escáner (deriva de baja frecuencia, ruido térmico, inhomogeneidades de campo). El paper lo enuncia directo: el ruido *"degrades the signal-to-noise ratio (SNR) and reduces the statistical power of fMRI studies"*.

En resting-state fMRI el sujeto solo descansa, y el análisis de interés es la **conectividad funcional** (correlaciones temporales entre regiones). El ruido de movimiento es especialmente insidioso porque introduce **correlaciones espurias** que se confunden con conectividad real, sesgando todo el análisis downstream (diagnóstico temprano de deterioro cognitivo, trastorno del espectro autista). Por eso el denoising no es opcional: es un paso esencial de preprocesamiento.

El enfoque dominante descompone el fMRI con **ICA espacial** (*Independent Component Analysis*) en componentes independientes (ICs). Cada IC es un par: un **mapa espacial 3D** (dónde ocurre el patrón) y una **serie temporal 1D** (cómo evoluciona en el tiempo). El denoising se reduce entonces a un problema de **clasificación binaria**: etiquetar cada IC como *signal* (actividad neuronal genuina) o *noise* (movimiento, fisiología, artefacto), y reconstruir el fMRI limpio descartando los componentes de ruido. Métodos clásicos como FIX e ICA-AROMA usan clasificadores sobre *features* hechas a mano o patrones de ruido predefinidos; los métodos recientes de *deep learning* usan CNNs para extracción automática de *features*.

El cuello de botella es doble. Primero, **anotar señal/ruido requiere expertos y es caro**: un especialista debe revisar manualmente cientos o miles de componentes por dataset (en los experimentos, entre 2.585 y 22.877 ICs por dataset). Segundo —y más sutil— los **criterios de etiquetado varían entre datasets**: lo que un centro marca como ruido marginal, otro lo puede marcar como señal débil, según el objetivo del estudio y el juicio del anotador. Esto no es ruido aleatorio en las etiquetas, sino un **sesgo sistemático por dataset**.

---

## El reto: criterios de etiquetado distintos entre datasets/centros (inter-dataset variation)

El paper plantea un escenario de **semi-supervised domain adaptation** con esta estructura:

- **Source domains** ($D$ datasets): *fully labeled*, con anotaciones expertas completas. Son el conocimiento previo a explotar.
- **Target domain** (1 dataset): el dataset nuevo a limpiar, **sparsely labeled** — solo una fracción pequeña de sus ICs está etiquetada (en los experimentos, **10%**), más una gran mayoría sin etiquetar.

El término *semi-supervised* aplica al target (combina un pequeño conjunto etiquetado con un gran conjunto sin etiquetar); *domain adaptation* aplica al hecho de que source y target vienen de distribuciones distintas (escáner/protocolo) **y además** con etiquetas bajo criterios distintos. Esto es lo que hace que el transfer learning ingenuo falle.

La evidencia empírica del fracaso del transfer simple está en los propios datos: en la condición *fully-labeled*, entrenar conjuntamente sobre todos los datasets a veces **empeora** respecto al baseline por dataset. El paper: *"jointly training with the other datasets does not always enhance performance due to variations in noise characteristics and inconsistencies in labeling criteria. For instance, in the HCP dataset, Setting (A) slightly reduces accuracy (99.12%) compared to the baseline (99.31%)"*. Mezclar datasets sin manejar el criteria shift es activamente dañino: el modelo recibe señales de supervisión contradictorias (mismo input, etiquetas opuestas según el dataset de origen). De ahí la necesidad de **separar** lo invariante de lo específico.

---

## La idea: feature extractor dataset-irrelevant (meta-aprendido) + clasificadores dataset-specific

La hipótesis arquitectónica es una factorización limpia:

> **Lo que es común a todos los datasets** (cómo se ve el ruido de movimiento, la fisiología, los artefactos) **debe vivir en una representación compartida e invariante al dataset.** **Lo que difiere entre datasets** (el criterio de etiquetado, dónde se traza la frontera señal/ruido) **debe vivir en cabezas separadas, una por dataset.**

Dos componentes:

1. **Dataset-irrelevant feature extractor** (parámetros $\theta$, una CNN): mapea cada IC a una representación de *features*. Se entrena con **meta-aprendizaje** para capturar *"a wide range of noise patterns across multiple well-labeled datasets and a sparsely labeled dataset"*. El meta-aprendizaje es el mecanismo para forzar invarianza: en vez de minimizar el error promedio sobre la mezcla de datasets (que produce un compromiso sesgado), aprende una inicialización que **se adapta rápido a cualquier dataset**, lo que empuja la representación hacia lo transversal.

2. **Dataset-specific classifiers** (parámetros $\phi^1, \dots, \phi^D, \psi$): $D+1$ clasificadores, uno por cada source ($\phi^d$) y uno para el target ($\psi$). Cada uno es una red *fully connected* de una sola capa, entrenada con **decoupled training**: el feature extractor se **congela** y cada clasificador se optimiza por separado con las etiquetas de su propio dataset. Así cada cabeza absorbe el criterio de etiquetado idiosincrático de su dataset sin alterar la representación compartida.

La elegancia está en que la representación nunca tiene que "elegir" entre los criterios contradictorios de los distintos datasets: esa decisión se delega a las cabezas. La columna solo se preocupa de separar bien las clases de modo que cualquier cabeza pueda trazar su propia frontera.

---

## La arquitectura de 3 etapas (pretraining, meta-learning, decoupled training)

### Etapa A — Pretraining

Se entrena un único feature extractor (parámetros $\omega$) y un único clasificador (parámetros $\mu$) usando los source *fully labeled* más el conjunto etiquetado del target, minimizando *cross-entropy* ($\mathcal{L}_{cls}$). Tras $M$ pasos (en los experimentos $M=10$), los parámetros del extractor $\omega_M$ se fijan como inicialización $\theta_0$ para el meta-aprendizaje, y el clasificador $\mu_M$ inicializa los $D+1$ clasificadores específicos. El paper: *"Incorporating labeled data at this stage reduces ambiguity in signal/noise annotations and offers a stable initialization."*

### Etapa B — Meta-learning del feature extractor invariante

El corazón del método, basado en **MAML** (Finn et al.). Se forma un *task* por cada par (source domain $d$, target), dando $D$ tasks. El proceso itera $K$ veces (en los experimentos $K=300$) con dos bucles anidados.

**Inner loop**: para el par $d$, se entrenan parámetros task-specific $\tilde{\theta}^d_k$ minimizando la pérdida de clasificación $\mathcal{L}_{cls}$ y una pérdida de **alineación** $\mathcal{L}_{align}$ (*Supervised Contrastive Loss*), usando el *support set* del source $d$ y el conjunto **sin etiquetar** del target. La formulación canónica de MAML que el paper invoca:

$$
\tilde{\theta}^d_k = \theta_k - \alpha \, \nabla_{\theta} \Big[ \mathcal{L}_{cls}\big(\mathcal{S}^d; \theta_k, \phi^d\big) + \lambda \, \mathcal{L}_{align}\big(\mathcal{S}^d, \mathcal{U}^t; \theta_k\big) \Big]
$$

**Outer loop** (meta-update): una vez obtenidos todos los $\tilde{\theta}^d_k$, se actualiza el feature extractor validando los parámetros adaptados sobre el *query set* del source $\mathcal{Q}^d$ y el conjunto etiquetado del target $\mathcal{L}^t$:

$$
\theta_{k+1} = \theta_k - \beta \, \nabla_{\theta} \sum_{d=1}^{D} \Big[ \mathcal{L}_{cls}\big(\mathcal{Q}^d, \mathcal{L}^t; \tilde{\theta}^d_k, \phi^d, \psi\big) + \lambda \, \mathcal{L}_{align}\big(\mathcal{Q}^d, \mathcal{L}^t; \tilde{\theta}^d_k\big) \Big]
$$

La intuición: la **cross-entropy** empuja a clasificar bien señal/ruido; la **alignment loss contrastiva** acerca en el espacio de *features* las muestras de la misma clase y aleja las de clases distintas, **a través** del source etiquetado y el target sin etiquetar (es la parte "domain adaptation" y lo que permite usar los datos sin etiquetar del target); y la **estructura bilevel de MAML** hace que el outer loop no optimice $\theta$ para un dataset concreto, sino para que $\theta$ se adapte bien a cualquier dataset tras unos pocos pasos del inner loop. Eso produce la invarianza: $\theta$ no encarna ningún criterio de etiquetado particular.

### Etapa C — Decoupled training de los clasificadores

Con el extractor invariante listo, se optimizan los $D+1$ clasificadores. El feature extractor se **congela** (*"we freeze the feature extractor to prevent interference with dataset-irrelevant feature extraction"*) y cada clasificador se actualiza **independientemente**, minimizando $\mathcal{L}_{cls}$ por separado para cada dataset. Contraste con el baseline: *"Unlike previous approaches that jointly update feature extraction and classification, we train the dataset-specific classifiers independently from the feature extractor to accommodate dataset-specific labeling criteria."* Como el gradiente del decoupled training nunca toca $\theta$, los criterios contradictorios no pueden corromper la representación.

El framework **alterna** entre la Etapa B (meta-learning) y la Etapa C (decoupled training) durante $K$ iteraciones, mejorando la clasificación mientras refuerza la adaptación de los source al target *"without compromising the dataset-irrelevant feature extractor"*.

---

## Resultados (datasets de fMRI, métricas señal/ruido — números reales)

Cuatro datasets elegidos para exhibir *inter-dataset variation*: **HCP** (Human Connectome Project, adultos jóvenes 22-35 años, TR=0.7s), **BCP** (Baby Connectome Project, ~1 año, TR=0.8s), **WHII-MB6** y **WHII-STD** (Whitehall II, adultos mayores 63-75 años, TR=1.3s y 3.0s respectivamente). El rango etario es extremo, el TR varía por un factor de ~4, y hay **fuerte desbalance de clases**: en HCP, 20.551 ICs de ruido vs. 2.326 de señal (ratio ~9:1). Por eso F1 y G-measure (GM, media geométrica de sensibilidad y especificidad) son necesarias, no solo accuracy. Se usa 5-fold cross-validation; hiperparámetros: batch size 12, $M=10$, dimensión de *features* 96, $K=300$, learning rate 0.01.

El estudio es incremental: **Baseline** (modelo por dataset) → **Setting (A)** +pretraining → **Setting (B)** +meta-learning → **Setting (C)** +decoupled training (framework completo). Significancia vs. Setting (C) con test de Wilcoxon signed-rank.

**Condición sparsely labeled (10%)** — números reales de la Tabla 2:

| Target | Método | ACC (%) | SEN (%) | SPEC (%) | F1 (%) | GM (%) |
|---|---|---|---|---|---|---|
| HCP | Baseline | 98.48 | 91.61 | 99.26 | 92.36 | 95.25 |
| HCP | Setting (C) | 98.93 | 97.82 | 99.06 | 94.99 | 98.42 |
| BCP | Baseline | 96.13 | 86.21 | 98.33 | 88.74 | 91.63 |
| BCP | Setting (C) | 96.60 | 90.01 | 97.92 | 90.55 | 93.61 |
| WHII-MB6 | Baseline | 93.89 | 75.70 | 95.82 | 78.01 | 82.85 |
| WHII-MB6 | Setting (C) | 98.83 | 97.19 | 99.13 | 95.60 | 98.13 |
| WHII-STD | Baseline | 86.03 | 0.00 | 100.00 | 0.00 | 0.00 |
| WHII-STD | Setting (C) | 97.67 | 91.12 | 98.90 | 91.07 | 94.79 |

El resultado más dramático: en **WHII-STD al 10%** el baseline obtiene **SEN = 0.00%, F1 = 0.00%, GM = 0.00%** — clasifica TODO como ruido (de ahí SPEC = 100%) porque con pocas etiquetas y desbalance brutal aprende el atajo trivial de la clase mayoritaria. El framework propuesto recupera **SEN = 91.12%** y **F1 = 91.07%**. Esto ilustra exactamente para qué sirve el método: rescatar la clase minoritaria (señal) bajo escasez de etiquetas.

**Condición fully labeled (100%)**: el framework completo también gana en ACC, SEN, F1 y GM en la mayoría de datasets (HCP Setting C: F1 96.85%, GM 98.77%; WHII-STD Setting C: SEN 94.14%, GM 96.60%), demostrando que el beneficio no es solo "tapar la falta de etiquetas", sino mejorar la representación per se.

El resultado estrella: *"In the WHII-MB6 dataset, Setting (C) in the 10% labeled condition achieves 95.60% F1-score and 98.13% G-measure, outperforming 95.18% F1-score and 96.77% G-measure of the baseline in the 100% labeled condition."* Es decir, el método con **una décima parte de las etiquetas** supera al baseline con **todas** las etiquetas en WHII-MB6: el conocimiento transferido vía meta-learning compensa con creces la falta de etiquetas locales. Una evaluación cualitativa con Grad-CAM (Fig. 2) confirma que el baseline clasifica erróneamente un *signal IC* como ruido mientras el framework lo identifica bien.

---

## Por qué importa: meta-learning para domain adaptation con criteria shift

Este es un caso de estudio limpio de **meta-aprendizaje para semi-supervised domain adaptation bajo criteria shift**, un problema endémico de los datos clínicos multi-centro y no exclusivo de fMRI. Tres lecciones:

1. **Label noise estructurado (criteria shift), no aleatorio.** La mayoría de la literatura de *label noise* asume ruido aleatorio o dependiente de la clase. Aquí el "ruido" es un sesgo sistemático por origen de datos. La respuesta correcta no es "limpiar las etiquetas", sino **factorizar la arquitectura**: representación compartida invariante + cabezas específicas por fuente. Generalizable a cualquier dominio con múltiples anotadores o instituciones.

2. **Aprovechar datos legacy abundantes para arrancar datasets nuevos con poca anotación.** "Tengo datasets viejos bien etiquetados y un dataset nuevo casi sin etiquetar" es ubicuo en salud (escáner nuevo, protocolo nuevo, población nueva). Reducir la anotación experta de 100% a 10% manteniendo o superando el rendimiento tiene impacto económico directo.

3. **Separar invarianza de especificidad como principio de diseño.** Congelar el extractor durante el decoupled training impide que los criterios contradictorios corrompan la representación. Es un patrón transferible: no dejes que las cabezas específicas reescriban la columna compartida.

Las limitaciones, en honestidad: el método es específico de denoising fMRI/ICA (hereda la calidad del paso de ICA previo), depende de sources *fully labeled* de buena calidad, su pipeline de 3 etapas con MAML bilevel es costoso y sensible a hiperparámetros, evalúa solo dos fracciones de etiquetas (10% y 100%, sin caracterizar la curva intermedia), opera a escala modesta (39-100 *samples* por dataset) y los cuatro escáneres son Siemens 3T (sin evidencia sobre GE/Philips ni 7T/1.5T).

---

## Conexión con la Clase 26 y con la armonización de datos clínicos multi-institucionales

Este paper MICCAI 2025 es emblemático del giro de la comunidad de *medical imaging* hacia los problemas **operacionales** del despliegue clínico: el cuello de botella ya no es el modelo, sino la anotación y la heterogeneidad multi-centro. Combina meta-aprendizaje y domain adaptation como herramientas de eficiencia de datos (donde las etiquetas expertas son el recurso más caro) y explicabilidad integrada (Grad-CAM) como requisito de facto.

El **criteria shift entre datasets** del paper es estructuralmente idéntico a la heterogeneidad de codificación entre sistemas de salud: distintas instituciones codifican el mismo concepto clínico con criterios distintos (un mismo diagnóstico mapeado a ICD/SNOMED/CIE-10 diferentes según reglas locales, el mismo `Observation` con unidades o rangos de referencia distintos, el mismo concepto de "alergia activa" definido con criterios divergentes). El paper menciona explícitamente el problema multi-sitio citando trabajos de *bias en datasets de neuroimagen multi-sitio* y de *statistical harmonization para corregir site effects* — el vocabulario exacto de la armonización de datos.

La lección arquitectónica es transferible a un pipeline FHIR multi-institucional y a trabajo de MDM/patient matching: el patrón **representación compartida invariante al origen + cabezas específicas por institución** sugiere que (a) el embedding de paciente del *blocker* debería ser invariante a la institución, de modo que dos registros del mismo paciente de hospitales distintos caigan cerca en el espacio latente; (b) reglas o *scorers* específicos por institución capturan las idiosincrasias de codificación de cada sistema fuente sin contaminar la representación compartida; y (c) el *decoupled training* (congelar la columna, ajustar solo las cabezas) es justo lo que se querría para incorporar un hospital nuevo sin reentrenar todo el modelo de matching. Y refuerza el matiz de los **retornos decrecientes del ML**: el valor real aparece donde hay escasez de etiquetas y heterogeneidad multi-sitio (WHII-STD 10%, de 0% a 91% de sensibilidad), no donde las etiquetas ya son abundantes y la curva es plana (HCP fully-labeled, ganancia marginal sobre 99.31%). La advertencia honesta para el traslado: la maquinaria exacta de MAML probablemente sea sobre-ingeniería frente a un GBM + reglas; lo valioso es el principio de diseño, no el pipeline completo.

---

## Notas y enlaces

Ver fundamentos: [Meta-aprendizaje](/fundamentos/meta-aprendizaje) - [Transfer learning](/fundamentos/transfer-learning) - [Few-shot learning](/fundamentos/few-shot-learning).

Papers relacionados: [MAML (Finn 2017)](/papers/maml-finn-2017) - [Meta-Disentanglement (Liu 2021)](/papers/meta-disentanglement-liu-2021) - [MetaSeg (Vyas 2025)](/papers/metaseg-vyas-2025).

Código: [github.com/KeunsooHeo/metaclean](https://github.com/KeunsooHeo/metaclean).

Clase: Ver [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
