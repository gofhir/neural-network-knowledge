---
title: "Wide & Deep Learning for Recommender Systems"
weight: 259
math: true
---

{{< paper-card
    title="Wide & Deep Learning for Recommender Systems"
    authors="Cheng, Koc, Harmsen, Shaked, Chandra, et al."
    year="2016"
    venue="DLRS@RecSys 2016"
    pdf="/papers/wide-and-deep-cheng-2016.pdf"
    arxiv="1606.07792" >}}
Paper de **Google** que resuelve un dilema clásico de los recomendadores: **memorizar** co-ocurrencias frecuentes (que hacen muy bien los modelos lineales con *cross-products*, pero **no generalizan** a combinaciones nuevas) versus **generalizar** a pares query-ítem no vistos (que hacen los *embeddings* densos, pero **sobre-generalizan** en datos dispersos de alto rango). La propuesta es **entrenar conjuntamente** un componente lineal *wide* y una red profunda *deep* bajo una misma pérdida logística, combinando ambas virtudes. Llevado a producción en **Google Play** (mil millones de usuarios, un millón de apps), logró **+3.9%** de adquisiciones de apps en un A/B test, y dio origen a toda una familia de modelos (DeepFM, DCN).
{{< /paper-card >}}

---

## Contexto

Un sistema de recomendación puede verse como un sistema de *search ranking*: dada una *query* (features del usuario y del contexto), recupera ítems candidatos y los ordena por la probabilidad de una acción objetivo —en este caso, instalar una app. A escala industrial (Google Play tiene más de **mil millones de usuarios activos** y **más de un millón de apps**) lo dominante eran los **modelos lineales generalizados** (regresión logística) sobre features dispersas con one-hot encoding, porque son simples, escalables e interpretables.

El paper articula con nitidez una tensión de dos fuerzas:

- **Memorización:** aprender la co-ocurrencia frecuente de features en los datos históricos. Se logra con **transformaciones cross-product** sobre features binarias —por ejemplo `AND(user_installed_app=netflix, impression_app=pandora)`, que vale 1 si el usuario instaló Netflix y luego vio Pandora. Es efectivo e interpretable, pero requiere *feature engineering* manual y **no generaliza a pares query-ítem que no aparecieron en el entrenamiento**.
- **Generalización:** explorar combinaciones de features nuevas o raras vía **embeddings** densos de baja dimensión (factorization machines o redes profundas), con menos ingeniería de features. El problema: cuando la matriz query-ítem es **dispersa y de alto rango** (usuarios con gustos muy específicos, apps de nicho), los embeddings densos **sobre-generalizan** y producen predicciones no nulas para *todos* los pares, recomendando ítems poco relevantes. Un modelo lineal con cross-products memoriza estas "reglas de excepción" con muchísimos menos parámetros.

La pregunta del paper: **¿cómo tener memorización y generalización en un solo modelo?**

## Ideas principales

El modelo combina dos componentes entrenados de forma **conjunta** (no como ensemble).

### Componente wide: cross-products lineales (memorización)

Es un modelo lineal generalizado $y = \mathbf{w}^T \mathbf{x} + b$, donde $\mathbf{x}$ incluye features crudas y transformadas. La transformación clave es el **cross-product**:

$$ \phi_k(\mathbf{x}) = \prod_{i=1}^{d} x_i^{\,c_{ki}}, \quad c_{ki} \in \{0,1\} $$

$c_{ki}$ vale 1 si la $i$-ésima feature participa en la $k$-ésima transformación. Para features binarias, `AND(gender=female, language=en)` vale 1 solo si todas sus features constituyentes valen 1. Esto **captura interacciones entre features y añade no-linealidad** al modelo lineal. Es el mecanismo de memorización.

### Componente deep: embeddings + MLP (generalización)

Una red feed-forward. Cada **feature categórica** dispersa (string como `language=en`) se convierte en un **vector embedding** denso de baja dimensión (orden de $O(10)$ a $O(100)$), inicializado aleatoriamente y entrenado para minimizar la pérdida. Esos embeddings alimentan capas ocultas:

$$ a^{(l+1)} = f\big(W^{(l)} a^{(l)} + b^{(l)}\big) $$

con $f$ típicamente ReLU. Es el mecanismo de generalización a pares no vistos.

### Joint training: suma de log-odds bajo una pérdida común

Wide y deep se combinan mediante una **suma ponderada de sus log-odds**, alimentada a una única pérdida logística:

$$ P(Y=1\mid\mathbf{x}) = \sigma\!\Big( \mathbf{w}_{wide}^T [\mathbf{x}, \phi(\mathbf{x})] + \mathbf{w}_{deep}^T a^{(l_f)} + b \Big) $$

donde $\sigma$ es la sigmoide, $\phi(\mathbf{x})$ los cross-products, $a^{(l_f)}$ las activaciones finales del deep, y $\mathbf{w}_{wide}$, $\mathbf{w}_{deep}$ los pesos respectivos.

La distinción **joint training vs ensemble** es central: en un ensemble los modelos se entrenan por separado y se combinan solo en inferencia, por lo que cada uno debe ser grande. En joint training todos los parámetros se optimizan *simultáneamente* por backpropagation, así la parte wide solo necesita **complementar las debilidades** de la deep con unos pocos cross-products, en vez de ser un modelo wide completo. En los experimentos usaron **FTRL con regularización $L_1$** para el wide y **AdaGrad** para el deep.

## Resultados experimentales

En producción, la estructura usada concatena un embedding de **32 dimensiones** por cada feature categórica junto con las features continuas, dando un vector denso de **~1200 dimensiones** que pasa por **3 capas ReLU** (1024 → 512 → 256) hasta la unidad logística. El wide consistió en el cross-product de *user installed apps* × *impression apps*. Los modelos se entrenaron sobre **más de 500 mil millones de ejemplos**, con **warm-starting** para reaprovechar embeddings y pesos del modelo anterior.

**A/B test online de 3 semanas** en Google Play (control = modelo wide-only altamente optimizado):

| Modelo | AUC offline | Ganancia online de adquisición |
|---|---|---|
| Wide (control) | 0.726 | 0% |
| Deep | 0.722 | +2.9% |
| Wide & Deep | 0.728 | **+3.9%** |

Wide & Deep mejoró las adquisiciones **+3.9% relativo al control** (estadísticamente significativo) y **+1% sobre el deep-only**. Un detalle aleccionador: el **AUC offline apenas se movió** (0.728 vs 0.726), pero el **impacto online fue mucho mayor**; el paper lo atribuye a que el sistema online genera recomendaciones exploratorias nuevas y aprende de las respuestas, mientras que el dataset offline está fijo —un recordatorio de la brecha entre métricas offline y de negocio.

En **serving**, a tráfico pico scorean más de 10 millones de apps por segundo; el multithreading bajó la latencia de 31 ms (single-thread) a **14 ms**.

## Limitaciones reconocibles

- La mejora **offline es marginal** (0.726 → 0.728); toda la evidencia fuerte descansa en un A/B online no reproducible fuera de Google.
- El **feature engineering del wide sigue siendo manual**: hay que elegir a mano qué pares cruzar. El modelo no aprende *cuáles* interacciones importan —exactamente el hueco que llenarían DeepFM y DCN.
- Usa **dos optimizadores distintos** (FTRL+$L_1$ y AdaGrad), añadiendo complejidad de tuning.
- Modela un **objetivo binario** (instalar sí/no); no aborda ranking multi-objetivo ni engagement posterior.
- Los requisitos de escala (500 mil millones de ejemplos, warm-starting, serving multithreaded) son específicos de Google; la transferencia a entornos pequeños no se discute.

## Por qué importa hoy

Más allá de la arquitectura puntual, Wide & Deep estableció un **principio de diseño**: combinar memorización + generalización mediante entrenamiento conjunto. Fue la base de toda una familia de modelos de *deep recommendation / CTR prediction*:

- **[DeepFM](/papers/deepfm-guo-2017)** (Guo et al., 2017) reemplaza el wide manual por una **Factorization Machine** que aprende automáticamente las interacciones de segundo orden, compartiendo embeddings con la parte deep.
- **Deep & Cross Network (DCN)** introduce una *cross network* que aprende interacciones de orden creciente de forma automática.
- La API `DNNLinearCombinedClassifier` de TensorFlow popularizó el patrón, y le siguieron xDeepFM, DIN, AutoInt, entre otros.

## Conexión con la Clase 25

La [Clase 25](/clases/clase-25) trata cómo **representar y combinar tipos de datos heterogéneos** en recomendadores multimodales. Wide & Deep es un caso canónico de varios principios de la clase:

- **Embeddings de categóricos:** cada feature categórica de alta cardinalidad (`language=en`, `user_installed_app=netflix`) se mapea a un vector denso entrenable de baja dimensión (32 dims en Google Play). Es exactamente la técnica de pasar de one-hot disperso a embeddings densos aprendidos end-to-end. Ver [recommender systems](/fundamentos/recommender-systems) y [representación de datos](/fundamentos/representacion-datos).
- **Features continuas:** se normalizan a $[0,1]$ por cuantiles de su CDF antes de entrar al modelo.
- **Concatenación de representaciones heterogéneas:** el paso central es **concatenar todos los embeddings categóricos + las features continuas** en un único vector denso (~1200 dims) que alimenta el MLP. Esta concatenación es justamente el mecanismo de "combinar tipos de datos distintos" que enseña la clase.
- **Fusión de paradigmas:** la suma ponderada de log-odds wide + deep es una forma temprana de *late fusion* a nivel de logit, antecedente conceptual de las fusiones multimodales modernas.

Es el puente entre los modelos lineales clásicos de recomendación y las arquitecturas neuronales actuales.

## Notas y enlaces

- **PDF:** [/papers/wide-and-deep-cheng-2016.pdf](/papers/wide-and-deep-cheng-2016.pdf)
- **arXiv:** [1606.07792](https://arxiv.org/abs/1606.07792)
- **Venue:** DLRS@RecSys 2016 (1st Workshop on Deep Learning for Recommender Systems)
- **Implementación open-source:** tutorial Wide & Deep en TensorFlow (`DNNLinearCombinedClassifier`).
- Relacionado: [DeepFM](/papers/deepfm-guo-2017), [fundamentos de recommender systems](/fundamentos/recommender-systems), [representación de datos](/fundamentos/representacion-datos), [Clase 25](/clases/clase-25).
