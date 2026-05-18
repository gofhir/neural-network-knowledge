---
title: "Dense Correspondence y UV Mapping"
weight: 132
math: true
---

La **correspondencia densa** consiste en mapear *cada píxel* de una imagen 2D a un punto en una superficie 3D de referencia. En el contexto de cuerpos humanos, significa que para cada píxel del foreground humano, el modelo predice **a qué punto de la malla SMPL corresponde**. Es la representación que sustenta [DensePose](/papers/densepose-guler-2018), Virtual Try-On y muchas técnicas de body recovery 3D. Este fundamento explica las dos piezas clave — **UV mapping** y **multidimensional scaling (MDS)** — y conecta con el modelo [SMPL](/papers/smpl-loper-2015) subyacente.

---

## 1. ¿Por qué correspondencia densa?

Los **keypoints** (17 puntos COCO) representan el cuerpo como un *grafo discreto*. Esto pierde tres tipos de información cruciales:

1. **Orientación de superficie**: un brazo puede estar visto desde el frente, lado o atrás — los keypoints no lo distinguen.
2. **Deformación local**: la curvatura del cuerpo (musculatura, postura) requiere modelar la superficie completa.
3. **Contacto con objetos / ropa**: predecir cómo la ropa cae sobre el cuerpo, o cómo dos cuerpos interactúan, necesita el mapeo continuo.

Dense correspondence resuelve los tres: cada píxel del cuerpo apunta a una posición específica en una superficie 3D estándar (SMPL).

{{< concept-alert type="clave" >}}
Keypoints describen **el esqueleto**. Dense correspondence describe **la superficie**. La diferencia es análoga a representar un retrato con puntos del rostro vs. con una topografía completa.
{{< /concept-alert >}}

## 2. UV Mapping — el concepto base

En gráficos por computador, **UV mapping** es la técnica clásica de envolver una textura 2D sobre una superficie 3D. Cada vértice $v_i$ de la malla 3D tiene asociado un par $(u_i, v_i) \in [0, 1]^2$ que indica **dónde en la imagen de textura** se busca su color.

Pensar en **una pelota de globo deflada**: si pinchas un globo y lo aplastas sobre la mesa, los puntos del globo 3D quedan distribuidos sobre una región 2D. Esa región 2D es el espacio UV; las coordenadas $(u, v)$ del punto aplastado son su "UV coordinate".

Formalmente:

$$
\text{uv-map} : \mathcal{S}_{3D} \to [0, 1]^2
$$

donde $\mathcal{S}_{3D}$ es la superficie del mesh. Debe ser **bijectiva localmente** (no auto-superpuesta) y minimizar **distorsión** (preservar áreas y ángulos lo más posible).

### 2.1 Problemas técnicos

- **Topología**: una esfera no puede aplanarse sin cortes (teorema egregium de Gauss). Hay que **cortar** la superficie en parches y aplanar cada uno por separado.
- **Distorsión**: aplanar una superficie curva produce stretch o squash. Existen métricas (ARAP, conformal) para minimizar el efecto.
- **Atlas**: el conjunto de parches 2D + cómo se ensamblan se llama *texture atlas*.

## 3. UV Mapping del cuerpo humano en SMPL

El modelo [SMPL](/papers/smpl-loper-2015) trae un **UV atlas pre-definido** que mapea su malla de 6890 vértices a un plano 2D. Es un atlas con cortes a lo largo de costuras anatómicas (espalda, lados de brazos y piernas) para minimizar distorsión.

El atlas oficial de SMPL es público y se usa en herramientas como `smplx`, Blender plugins, y bases de datos como SURREAL.

### 3.1 Las 24 partes de DensePose

[DensePose](/papers/densepose-guler-2018) usa un **atlas alternativo** — diseñado específicamente para que cada parte sea **isomorfa a un disco 2D** (sin cortes, sin auto-superposición). Particiona el cuerpo en **24 partes** semánticas:

| Parte # | Descripción |
|---|---|
| 1, 2 | Torso frontal, dorsal |
| 3-6 | Brazos (sup/inf, izq/der) frontal y dorsal partidos en mitades |
| 7-12 | Piernas (sup/inf, izq/der) frontal y dorsal partidas |
| 13-16 | Manos (frontal/dorsal, izq/der) |
| 17-20 | Pies (frontal/dorsal, izq/der) |
| 21-24 | Cabeza (frontal, dorsal, etc.) |

Cada parte tiene su propio sistema $(u, v) \in [0, 1]^2$. Un píxel humano queda etiquetado entonces con **tres números**:

$$
(c, u, v), \quad c \in \{1, ..., 24\}, \quad (u, v) \in [0, 1]^2
$$

— **parte + coordenadas intra-parte**. Combinado, $(c, u, v)$ identifica unívocamente un punto de la superficie SMPL.

## 4. Cómo se construye el UV map de DensePose: MDS

El paper de DensePose detalla el procedimiento (Sección 2.1):

- **Cabeza, manos, pies**: usan el UV map oficial de SMPL — son partes con topología conveniente (cuasi-discos) y SMPL ya provee sus coordenadas.
- **Resto del cuerpo**: aplican **Multidimensional Scaling (MDS)** sobre las **distancias geodésicas** del mesh.

### 4.1 ¿Qué es MDS?

Multidimensional scaling es una técnica clásica de embedding (Torgerson 1952, Shepard 1962) que **encuentra una representación de bajo dimensional** preservando las **distancias** entre puntos.

Dada una matriz de distancias $D = (d_{ij})$ entre $N$ puntos en un espacio arbitrario, MDS busca puntos $\{p_i\}_{i=1}^N \in \mathbb{R}^k$ (típicamente $k = 2$) que minimicen:

$$
\text{Stress}(\{p_i\}) = \sum_{i < j} \bigl( \|p_i - p_j\|_2 - d_{ij} \bigr)^2
$$

Para MDS **clásico** (linear), se hace double-centering del cuadrado de distancias y luego eigendecomposition — equivalente a PCA sobre el espacio implícito.

### 4.2 Aplicación a SMPL

Para cada parte del cuerpo:

1. Computar la matriz de **distancias geodésicas** entre todos los vértices de la parte (distancia mínima caminando sobre la superficie de la malla, vía Dijkstra o heat method).
2. Aplicar MDS para obtener una representación 2D de la parte.
3. Normalizar las coordenadas resultantes a $[0, 1]^2$ → estas son las $(u, v)$ de cada vértice.

El resultado es un **unwrap aproximadamente isométrico** (preserva distancias) y plano de cada parte. El profesor Vergara muestra el resultado en el slide 32: cada parte queda parametrizada como una "lámina" continua.

## 5. La métrica geodésica

La distancia entre dos vértices del mesh **no es la Euclidiana 3D** — es la **geodésica** sobre la superficie. Esto es crítico para:

- **Definir UV maps** (preserva la geometría intrínseca de la superficie).
- **Métricas de evaluación** (ver GPS más abajo).
- **Smoothness regularization** en métodos de superficie.

Algoritmos para computar geodésicas sobre meshes:
- **Dijkstra** sobre el grafo de aristas (rápido, pero solo sobre rutas en aristas — sesga).
- **Fast Marching** (Sethian 1996) — propagación de un frente sobre la superficie.
- **Heat method** (Crane et al. 2013) — resolver una ecuación de calor.

En DensePose se usan precomputadas: la tabla de distancias geodésicas entre todos los pares de vértices SMPL se computa una vez y se cachea.

## 6. La métrica de evaluación: GPS

[DensePose](/papers/densepose-guler-2018) introduce **Geodesic Point Similarity** (Ec. 2 del paper):

$$
\text{GPS}_j = \frac{1}{|P_j|} \sum_{p \in P_j} \exp\!\left( -\frac{g(i_p, \hat i_p)^2}{2 \kappa^2} \right)
$$

con:
- $P_j$ — los puntos ground-truth para la persona $j$.
- $i_p$ — el vértice predicho para el punto $p$.
- $\hat i_p$ — el vértice ground-truth.
- $g(\cdot, \cdot)$ — distancia geodésica sobre la superficie SMPL.
- $\kappa = 0.255$ — constante calibrada (un error geodésico de ~30 cm da GPS = 0.5).

A partir de GPS se computan AP y AR a thresholds 0.5 a 0.95 — análogo a OKS para keypoints.

## 7. Inpainting de supervisión densa

Un detalle importante de DensePose: solo **~150 píxeles por persona** están anotados manualmente. Si solo se entrena en esos píxeles, el modelo está sub-supervisado.

**Solución de distillation**:

1. Entrenar primero una **teacher network** (FCN simple) con la supervisión escasa.
2. Inferir las predicciones de la teacher **sobre toda la región humana**.
3. Usar esas predicciones como **supervisión densa** para entrenar el modelo final.
4. Restringir el loss al **foreground humano** (ignorar background).

Esto se llama **inpainting del signal de supervisión**. Ganancia: +5-7 AUC consistentes.

## 8. Aplicaciones más allá de DensePose

### 8.1 Virtual Try-On

La superficie del cuerpo, parametrizada con $(c, u, v)$, permite **mapear texturas de ropa** sobre el cuerpo de cualquier persona. La textura se diseña en el espacio UV de SMPL, y el modelo "viste" automáticamente el cuerpo al hacer el lookup texture-to-pixel.

Ejemplos: VITON, CP-VTON, TryOnDiffusion (Google 2023).

### 8.2 Texture transfer entre personas

Si conozco el mapeo $(c, u, v)$ de la persona A y de la persona B, puedo transferir el patrón de ropa de A a B preservando la pose: cada píxel de B busca su correspondiente $(c, u, v)$ en A.

### 8.3 Pose 3D recovery

Métodos como HMR (Kanazawa 2018) y VIBE (Kocabas 2020) usan supervisión DensePose como **señal auxiliar** para fittear SMPL: si la red predice los parámetros $(\vec\beta, \vec\theta)$, los vértices proyectados deben coincidir con la UV inferida de la imagen.

### 8.4 Continuous Surface Embeddings (CSE)

Generalización a animales por Neverova et al. (2020). En vez de un atlas humano, se aprende un **embedding continuo** de cada vértice mediante self-supervised learning sobre videos de animales. Permite DensePose para vacas, perros, monos, etc.

### 8.5 Mesh recovery con SMPL-X

[SMPL-X](https://smpl-x.is.tue.mpg.de/) (Pavlakos 2019) extiende a cara + manos. Métodos como **PIXIE**, **PyMAF-X**, **ExPose** usan dense correspondence como prior fuerte para fitear el modelo completo.

## 9. Conexiones con la Clase 17

- **Slides 26-36**: introducción a DensePose, alternativa a keypoints.
- **Slide 31**: el diagrama "Patch + U + V" — exactamente lo que esta página formaliza.
- **Slide 32**: las tres componentes (patch, U, V) — sección 3.1 acá.
- **Slide 34**: el pipeline de anotación humana — sección 4 del fundamento de pose estimation.
- **Slide 35**: la mención explícita de SMPL — su rol como sustrato 3D.

## 10. Recursos relacionados

- [DensePose (Güler 2018)](/papers/densepose-guler-2018) — el paper canónico.
- [SMPL (Loper 2015)](/papers/smpl-loper-2015) — el modelo subyacente.
- [Pose estimation](/fundamentos/pose-estimation) — el fundamento general.
- [Clase 17](/clases/clase-17) — pose recognition completo.
- Geodesic distance on meshes: Crane et al. *Geodesics in heat* (2013), https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/
- SMPL family: SMPL, SMPL-H (manos), FLAME (cara), SMPL-X (todo).
- Continuous Surface Embeddings: https://research.facebook.com/publications/continuous-surface-embeddings/
