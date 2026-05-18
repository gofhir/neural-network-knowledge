# OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

**Autores:** Zhe Cao, Tomas Simon, Shih-En Wei, Yaser Sheikh (The Robotics Institute, Carnegie Mellon University)
**Año:** 2017 (CVPR)
**arXiv:** 1611.08050 (publicado abril 2017, versión extendida en TPAMI 2019)
**Código:** https://github.com/CMU-Perceptual-Computing-Lab/openpose (license restrictiva para uso comercial)
**Video demo:** https://youtu.be/pW6nZXeWlGM

---

## 1. Contexto histórico

En 2016, la **estimación de pose multi-persona** era un problema dominado por dos enfoques contrapuestos:

- **Top-down** — un person-detector (Faster R-CNN, SSD) propone bounding boxes y un estimator single-person (Convolutional Pose Machines, Stacked Hourglass) corre dentro de cada bbox. Métodos representativos: Iqbal & Gall 2016, Papandreou et al. 2017. **Problema**: el costo crece linealmente con el número de personas, y si el detector falla, no hay recuperación posible.

- **Bottom-up** — detectar todos los keypoints primero y agruparlos en personas. Métodos previos: **DeepCut** (Pishchulin et al., 2016) y **DeeperCut** (Insafutdinov et al., 2016). Ambos formulaban el problema como **Integer Linear Programming sobre un grafo completamente conectado** — NP-hard, con tiempos de varios minutos por imagen.

OpenPose se posiciona como el **primer método bottom-up en tiempo real** para pose multi-persona, atacando las dos debilidades del bottom-up clásico:

1. La asociación de partes es **NP-hard** si se modela como grafo completo.
2. Los métodos previos usaban representaciones débiles (midpoints, association embeddings) que confundían personas adyacentes.

Sale en CVPR 2017 (con preprint en noviembre 2016) y **gana la inaugural COCO 2016 Keypoints Challenge**. El video demo en YouTube se vuelve viral — es la primera vez que se ve esqueletos multi-persona ejecutándose a video-rate en una laptop común.

## 2. Contribución central

Tres aportes esenciales:

1. **Part Affinity Fields (PAFs)** — una **representación no-paramétrica vectorial** de la asociación entre partes del cuerpo. En lugar de modelar la conexión como un escalar (un *midpoint detector* clásico), cada limb (antebrazo izquierdo, muslo derecho, etc.) tiene un **campo vectorial 2D** que codifica simultáneamente **posición** y **orientación** del miembro sobre toda su superficie de soporte.

2. **Arquitectura de dos ramas con supervisión intermedia** — una sola CNN aprende **simultáneamente** los confidence maps de los keypoints y los PAFs, con una arquitectura de varias etapas (6 stages) y pérdidas $L_2$ aplicadas al final de cada etapa para mitigar vanishing gradients.

3. **Greedy bipartite parsing** — un algoritmo de matching que, gracias a la riqueza de los PAFs, **no requiere ILP**. Reduce un problema NP-hard a una secuencia de **bipartite matchings tractables** (Hungarian algorithm) sobre los pares de partes conectadas, manteniendo cuasi-toda la precisión global.

Resultado headline: **mAP 75.6% en MPII Multi-Person dataset** (vs. ~58% de Iqbal & Gall y ~75% de DeeperCut con scale search) y **AP 61.8% en COCO test-dev**, **a 8.8 fps en video con 19 personas** (vs. minutos por imagen de los predecesores).

## 3. Arquitectura

### 3.1 Pipeline general

Para cada imagen RGB de tamaño $w \times h$:

1. **Feature extraction**: las primeras 10 capas de **VGG-19** (preentrenadas en ImageNet) producen feature maps $\mathbf{F}$.
2. **Stage 1**: dos ramas paralelas, cada una conv-net pequeña, predicen $(\mathbf{S}^1, \mathbf{L}^1) = (\rho^1(\mathbf{F}), \phi^1(\mathbf{F}))$.
3. **Stages 2…T**: cada etapa $t$ ve la concatenación $[\mathbf{F}, \mathbf{S}^{t-1}, \mathbf{L}^{t-1}]$ y refina las predicciones.
4. **Loss intermedio** al final de cada etapa para combatir vanishing gradients.
5. **Inference**: del último stage se extraen mapas de confianza $\mathbf{S}$ y PAFs $\mathbf{L}$ que entran al parsing.

Donde:
- $\mathbf{S} = (\mathbf{S}_1, \ldots, \mathbf{S}_J)$ son $J$ confidence maps, uno por tipo de keypoint ($J=18$ para COCO + neck inferido).
- $\mathbf{L} = (\mathbf{L}_1, \ldots, \mathbf{L}_C)$ son $C$ PAFs, uno por tipo de limb (19 en COCO).

### 3.2 Confidence maps de partes

Para cada persona $k$ y parte $j$, el ground-truth se construye como una Gaussiana centrada en la posición real $\mathbf{x}_{j,k}$:

$$
\mathbf{S}^*_{j,k}(\mathbf{p}) = \exp\!\left(-\frac{\|\mathbf{p} - \mathbf{x}_{j,k}\|_2^2}{\sigma^2}\right)
$$

Cuando hay varias personas, el mapa global es el **máximo punto a punto** sobre todas las personas (no la suma):

$$
\mathbf{S}^*_j(\mathbf{p}) = \max_k \mathbf{S}^*_{j,k}(\mathbf{p})
$$

Esto preserva picos distintos cuando dos personas están cerca, en vez de mezclarlos.

### 3.3 Part Affinity Fields — la idea central

Para cada limb $c$ que va del keypoint $j_1$ al $j_2$ de la persona $k$, el PAF ground-truth $\mathbf{L}^*_{c,k}$ es un **campo vectorial 2D**:

$$
\mathbf{L}^*_{c,k}(\mathbf{p}) = \begin{cases} \mathbf{v} & \text{si } \mathbf{p} \text{ está sobre el limb} \\ \mathbf{0} & \text{en otro caso} \end{cases}
$$

donde $\mathbf{v} = (\mathbf{x}_{j_2,k} - \mathbf{x}_{j_1,k}) / \|\mathbf{x}_{j_2,k} - \mathbf{x}_{j_1,k}\|_2$ es el **vector unitario en dirección del limb**.

Un punto $\mathbf{p}$ está "sobre el limb" si su proyección sobre la línea $\mathbf{x}_{j_1,k}\mathbf{x}_{j_2,k}$ cae en el segmento y la distancia perpendicular es $\leq \sigma_l$ píxeles.

Para múltiples personas, los PAFs se **promedian** sobre las personas que contribuyen al pixel:

$$
\mathbf{L}^*_c(\mathbf{p}) = \frac{1}{n_c(\mathbf{p})} \sum_k \mathbf{L}^*_{c,k}(\mathbf{p})
$$

**Por qué importa**: cada limb tiene **dos canales** (dx, dy) que codifican simultáneamente:
- *Posición*: el limb existe donde el vector es no-nulo.
- *Orientación*: la dirección del vector dice hacia dónde va el limb.

Esto es estrictamente más informativo que un *midpoint detector* (que solo dice "hay un punto medio aquí") o que los **association embeddings** de Newell et al. (que dan un escalar abstracto sin estructura espacial).

### 3.4 Multi-person parsing con PAFs

Dadas las posiciones candidatas de partes $\mathcal{D}_{\mathcal{J}}$ (extraídas por NMS sobre $\mathbf{S}$), hay que decidir qué candidatos van juntos. Para cada limb tipo $c$ entre dos candidatos $\mathbf{d}_{j_1}$ y $\mathbf{d}_{j_2}$, el score de asociación es la **integral lineal del PAF a lo largo del segmento**:

$$
E = \int_{u=0}^{u=1} \mathbf{L}_c(\mathbf{p}(u)) \cdot \frac{\mathbf{d}_{j_2} - \mathbf{d}_{j_1}}{\|\mathbf{d}_{j_2} - \mathbf{d}_{j_1}\|_2} \, du
$$

con $\mathbf{p}(u) = (1-u)\mathbf{d}_{j_1} + u\mathbf{d}_{j_2}$.

En la práctica se aproxima sampleando ~10 puntos uniformes sobre el segmento. El score combina:
1. **Magnitud** del PAF en el segmento (¿es realmente un limb?).
2. **Alineación direccional** con la dirección del segmento candidato (¿apunta hacia donde queremos?).

### 3.5 Reducción del NP-hard a bipartite matching

El problema completo es: dado el grafo $K$-partito (un partition por tipo de keypoint), encontrar la asignación que **maximiza la suma de scores E sin que dos limbs del mismo tipo compartan un endpoint**. Esto es **K-dimensional matching**, NP-hard.

Los autores aplican **dos relajaciones**:

1. **Spanning tree**: en lugar del grafo completo del cuerpo (donde podrías conectar codo-cadera, hombro-rodilla, etc.), restringen el esqueleto a un **árbol de cobertura mínima** sobre el cuerpo humano — 13 conexiones en vez de las 91 posibles entre 14 partes.
2. **Decomposición en bipartite matchings independientes**: cada *par adyacente* en el árbol (e.g., hombro-codo, codo-muñeca) es un bipartite matching independiente, que se resuelve con **Hungarian algorithm en O(n³)**.

Esta descomposición es válida porque el receptive field grande de la CNN ya codifica implícitamente la información de vecinos no-adyacentes en los PAFs locales.

**Validación experimental** (Tabla 2 del paper, sobre 343 imágenes de validación):
- Fully-connected graph (NP-hard, ILP): mAP 78.3.
- Tree structure + ILP: mAP 77.6.
- **Tree structure + greedy (paper)**: mAP 77.4 — **0.9 puntos menos por 4 órdenes de magnitud menos de tiempo**.

### 3.6 Velocidad y escalabilidad

- **Runtime CNN**: $O(1)$ en número de personas — la red corre una sola vez.
- **Runtime parsing**: $O(n^2)$ pero con constantes pequeñas (0.58 ms para 9 personas vs. 99.6 ms de la CNN).
- **Throughput total**: 8.8 fps sobre video 1080×1920 con 19 personas, en una **NVIDIA GTX 1080** sola.

Comparación con top-down: el método top-down (single-person CPM) escala linealmente — para 20 personas, OpenPose es ~**6× más rápido**.

## 4. Ejemplo de uso en el laboratorio (lab 17 IA UC)

El lab 17 usa la implementación **pytorch-openpose** (fork no oficial, Hzzone), no el código original de Caffe. La estructura de salida que importa para el clasificador MLP es:

```python
from src.body import Body

# Carga del modelo (pesos descargados manualmente desde Dropbox)
openpose_model = Body('pytorch-openpose/model/body_pose_model.pth')

# Inferencia
candidate, subset = openpose_model(opencv_image_bgr)
# candidate : (N_keypoints_detected, 4)  -> (x, y, score, id)
# subset    : (N_persons, 20)            -> indices a candidate + score total
```

El lab construye un tensor `(18, 3)` de keypoints por persona aplanando `candidate[subset[i, :18]]` y lo usa como **input al MLP**. Es decir: OpenPose actúa como *feature extractor* visual y el MLP aprende a mapear configuraciones de 54 floats (18 keypoints × 3 atributos) → categoría de acción.

**Gotcha del lab**: pytorch-openpose detecta **18 keypoints** (los 17 de COCO + el "neck" derivado). El paper original usa 19 limbs sobre estas 18 partes; el formato keypoint-índice del fork no es idéntico al de PifPaf, por eso el lab entrena MLPs separados para cada modelo en vez de compartir el clasificador.

**Por qué OpenPose es "no fácil de usar"** (citado en el lab): no hay `pip install` oficial multi-plataforma. El repo CMU original requiere compilar Caffe, descargar pesos, configurar CUDA y manejar dependencias C++. Los forks PyTorch son la única vía práctica fuera de Docker.

## 5. Experimentos clave

### 5.1 MPII Multi-Person (mAP por parte, full testing set)

| Método | Hea | Sho | Elb | Wri | Hip | Kne | Ank | mAP | s/image |
|---|---|---|---|---|---|---|---|---|---|
| DeeperCut (Insafutdinov 2016) | 78.4 | 72.5 | 60.2 | 51.0 | 57.2 | 52.0 | 45.4 | 59.5 | 485 |
| Iqbal & Gall 2016 | 58.4 | 53.9 | 44.5 | 35.0 | 42.2 | 36.7 | 31.1 | 43.1 | 10 |
| **OpenPose** | **91.2** | **87.6** | **77.7** | **66.8** | **75.4** | **68.9** | **61.7** | **75.6** | **0.005** |

**Mayor salto del paper**: 4 órdenes de magnitud de speedup sobre DeeperCut con +16 mAP. Esto es lo que permite el uso en video.

### 5.2 COCO 2016 Keypoints Challenge

| Equipo | AP | AP⁵⁰ | AP⁷⁵ | APᴹ | APᴸ |
|---|---|---|---|---|---|
| **OpenPose (winner)** | **60.5** | **83.4** | **66.4** | **55.1** | **68.1** |
| G-RMI (Papandreou et al., top-down) | 59.8 | 81.0 | 65.1 | 56.7 | 66.7 |
| DL-61 | 53.3 | 75.1 | 48.5 | 55.5 | 54.8 |
| R4D | 49.7 | 74.3 | 54.5 | 45.6 | 55.6 |

OpenPose gana el challenge en `test-challenge` set y mantiene liderazgo en `test-dev`. Pierde levemente con G-RMI en `APᴹ` (escalas medianas), donde top-down todavía es competitivo por su capacidad de rescalear cada bbox.

### 5.3 Ablation: PAFs vs midpoint representation

| Representación | mAP (PCKh 0.5, MPII val) |
|---|---|
| Midpoint, 1 punto | 78.4 |
| Midpoint, 2 puntos intermedios | 78.8 |
| **PAFs (paper)** | **81.6** |

PAFs ganan **+3.2 mAP sobre midpoint** porque codifican orientación, no solo posición.

### 5.4 Análisis de runtime

- Top-down (single-person CPM): **runtime crece linealmente** con número de personas.
- OpenPose: **runtime cuasi-constante** (la CNN es $O(1)$ y el parsing es $O(n^2)$ pero pequeño).
- A partir de **4-5 personas en escena**, OpenPose es ya estrictamente más rápido que cualquier top-down.

## 6. Limitaciones reconocidas

Fig. 9 del paper documenta los modos de falla principales:

1. **Poses raras o nuevas apariencias** (atletas en pose extrema, ropa atípica): los confidence maps son débiles.
2. **Partes faltantes o falsas**: si la persona está parcialmente fuera del frame, OpenPose puede inventar la pose completa o saltarse la persona.
3. **Partes compartidas entre personas**: cuando dos personas se abrazan o están muy cerca, un mismo keypoint detectado puede pertenecer a ambas y el matching greedy comete errores.
4. **Conexión incorrecta entre dos personas**: el PAF de un brazo de una persona puede solaparse con el brazo de otra detrás, creando un esqueleto Frankenstein.
5. **Falsos positivos en estatuas o animales**: la red dispara sobre cualquier silueta antropomorfa.

Limitaciones estructurales:
- **Sin información temporal** — pose por frame, sin tracking nativo.
- **Topología fija**: 13-19 conexiones hardcoded sobre el esqueleto humano. No generaliza a animales o objetos sin re-entrenar.
- **Sensibilidad a escala**: AP en escalas pequeñas es mucho peor que en grandes (problema heredado por PifPaf y mitigado por ViTPose).
- **Sin manejo de incertidumbre**: cada predicción es un punto duro, sin spread aprendido (esto lo introduce PifPaf con la Laplace loss).

## 7. Impacto y legado

- **Github stars**: >32k en el repo CMU original (uno de los repos de visión más populares de la historia).
- **Definición de la era bottom-up**: PifPaf, PersonLab, HigherHRNet, DEKR, CID — todos descienden directamente de la idea de PAFs.
- **Adopción industrial**: usado masivamente en fitness apps, AR, análisis deportivo, danza, animación, medicina. Apple Fitness+, Tonal, Mirror y la mayoría de apps de yoga usan derivados.
- **Versión extendida en TPAMI 2019** (Cao et al.) extiende la arquitectura con redundant PAFs y mejora la precisión en escalas pequeñas.
- **Limitación de licencia**: el código CMU tiene licencia restrictiva para uso comercial, lo que llevó a la explosión de **forks PyTorch** y eventualmente a **MediaPipe BlazePose** de Google (2020) como alternativa libre.

## 8. Conexión con la clase 17

OpenPose es **el** ejemplo canónico de bottom-up que aparece en los slides 28-36 del PDF. En el lab 17:

- Se compara directamente OpenPose vs. PifPaf como **dos generaciones del enfoque bottom-up** sobre el dataset Stanford 40 Actions.
- El MLP entrenado sobre features de OpenPose y el entrenado sobre features de PifPaf miden cuál representación captura mejor la información de acción.
- Empíricamente, PifPaf suele dar **mejores accuracy** porque sus keypoints tienen mayor precisión sub-pixel y mejor manejo de cuerpos parcialmente ocluidos — exactamente las debilidades que [[PifPaf-Kreiss-2019]] reclamó en su paper.

Cross-links:
- [[PifPaf-Kreiss-2019]] — la generación siguiente bottom-up que supera OpenPose en baja resolución.
- [[BlazePose-Bazarevsky-2020]] — la respuesta mobile-first a OpenPose desde Google.
- [[DensePose-Guler-2018]] — la alternativa top-down que va más allá de keypoints (mesh denso).
- [[ViTPose-Xu-2022]] — la nueva era top-down con backbones Transformer.
- [[fundamentos/pose-estimation.md]] — bottom-up vs. top-down formal.

## 9. Enlaces

- Paper: https://arxiv.org/abs/1611.08050
- Versión TPAMI 2019: https://arxiv.org/abs/1812.08008
- Código oficial CMU: https://github.com/CMU-Perceptual-Computing-Lab/openpose
- Fork PyTorch usado en el lab: https://github.com/Hzzone/pytorch-openpose
- Video demo: https://youtu.be/pW6nZXeWlGM
