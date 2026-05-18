---
title: "OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"
weight: 81
math: true
---

{{< paper-card
    title="Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"
    authors="Cao, Simon, Wei, Sheikh"
    year="2017"
    venue="CVPR 2017 (versión extendida TPAMI 2019)"
    pdf="/papers/openpose-cao-2017.pdf"
    arxiv="1611.08050" >}}
Primer método **bottom-up** de pose multi-persona en tiempo real. Introduce **Part Affinity Fields (PAFs)** — campos vectoriales 2D que codifican simultáneamente posición y orientación de los limbs — y reduce el matching NP-hard a una secuencia de **bipartite matchings tractables** (Hungarian algorithm). Ganó la inaugural COCO 2016 Keypoints Challenge con 8.8 fps sobre video de 19 personas, abriendo la era multi-persona consumer-grade.
{{< /paper-card >}}

---

## Contexto

En 2016 había dos enfoques antagónicos en pose multi-persona:

- **Top-down** (Iqbal & Gall, Papandreou et al.): detector de personas + single-person estimator dentro de cada bbox. Runtime crece **linealmente** con número de personas. Si el detector falla, no hay recuperación.
- **Bottom-up** (DeepCut, DeeperCut): detectar todos los keypoints y agruparlos en personas. Mejor robustez ante oclusión pero requería **Integer Linear Programming** sobre grafo completo (NP-hard, **minutos por imagen**).

OpenPose es el **primer bottom-up viable para video real-time**, atacando las dos debilidades históricas:
1. Representación débil para la asociación (midpoints, embeddings) → reemplazada por PAFs vectoriales.
2. Matching NP-hard → reducido a bipartite matching greedy con dos relajaciones.

## Ideas principales

### Part Affinity Fields (PAFs) — la representación clave

Para cada limb $c$ (antebrazo izquierdo, muslo derecho, etc.) que va del keypoint $j_1$ al $j_2$ de la persona $k$, el ground-truth $\mathbf{L}^*_{c,k}$ es un **campo vectorial 2D** sobre la imagen:

$$
\mathbf{L}^*_{c,k}(\mathbf{p}) = \begin{cases} \mathbf{v} & \text{si } \mathbf{p} \text{ está sobre el limb} \\ \mathbf{0} & \text{en otro caso} \end{cases}
$$

donde $\mathbf{v} = (\mathbf{x}_{j_2,k} - \mathbf{x}_{j_1,k}) / \|\mathbf{x}_{j_2,k} - \mathbf{x}_{j_1,k}\|_2$ es el **vector unitario** en dirección del limb.

Un punto $\mathbf{p}$ está "sobre el limb" si su proyección sobre la línea $\mathbf{x}_{j_1,k}\mathbf{x}_{j_2,k}$ cae en el segmento **y** la distancia perpendicular es $\leq \sigma_l$ píxeles.

**Por qué importa**: cada limb tiene **dos canales** (dx, dy) que codifican simultáneamente:
- *Posición* — el limb existe donde el vector es no-nulo.
- *Orientación* — el vector apunta en la dirección del limb.

Esto es estrictamente más informativo que un *midpoint detector* (escalar sin dirección) o que **association embeddings** (Newell et al., 2017) que dan escalares abstractos sin estructura espacial.

### Confidence maps con `max` punto a punto

Para cada parte $j$ y persona $k$:

$$
\mathbf{S}^*_{j,k}(\mathbf{p}) = \exp\!\left(-\frac{\|\mathbf{p} - \mathbf{x}_{j,k}\|_2^2}{\sigma^2}\right)
$$

El mapa multi-persona usa **máximo** (no suma):

$$
\mathbf{S}^*_j(\mathbf{p}) = \max_k \mathbf{S}^*_{j,k}(\mathbf{p})
$$

Esto **preserva picos distintos** cuando dos personas están cerca, en vez de mezclarlos en un blob.

### Arquitectura de dos ramas + supervisión intermedia

Una sola CNN con backbone VGG-19 (primeras 10 capas) seguida de **6 stages** iterativos. En cada stage:

- **Rama 1** predice los confidence maps $\mathbf{S}^t$ (J=18 keypoints).
- **Rama 2** predice los PAFs $\mathbf{L}^t$ (C=19 limbs).
- Stage $t+1$ recibe $[\mathbf{F}, \mathbf{S}^t, \mathbf{L}^t]$ concatenado.
- Loss $L_2$ aplicada al final de **cada stage** para mitigar vanishing gradients.

### Matching: del NP-hard al bipartite

Dadas candidatos de partes $\mathcal{D}_{\mathcal{J}}$ extraídos por NMS, el score de asociación entre $\mathbf{d}_{j_1}$ y $\mathbf{d}_{j_2}$ es la **integral lineal del PAF a lo largo del segmento**:

$$
E = \int_{u=0}^{u=1} \mathbf{L}_c(\mathbf{p}(u)) \cdot \frac{\mathbf{d}_{j_2} - \mathbf{d}_{j_1}}{\|\mathbf{d}_{j_2} - \mathbf{d}_{j_1}\|_2} \, du
$$

Aproximado samplando ~10 puntos uniformes. Combina **magnitud** del PAF + **alineación direccional**.

**Dos relajaciones** del problema $K$-dimensional matching NP-hard:

1. **Spanning tree**: solo 13 conexiones del esqueleto humano en lugar de las 91 posibles entre 14 partes.
2. **Descomposición en bipartite matchings independientes**: cada par adyacente del árbol se resuelve con Hungarian algorithm en $O(n^3)$.

Resultado (Tabla 2 del paper, 343 imágenes de validación MPII):

| Método | mAP | s/image |
|---|---|---|
| Fully-connected graph + ILP (NP-hard) | 78.3 | 362 |
| Tree structure + ILP | 77.6 | 43 |
| **Tree structure + greedy (paper)** | **77.4** | **0.005** |

**0.9 puntos menos por 4 órdenes de magnitud menos de tiempo**. Las relajaciones funcionan porque el receptive field grande de la CNN ya codifica implícitamente la información de vecinos no-adyacentes en los PAFs locales.

## Resultados clave

### MPII Multi-Person (full test set)

| Método | Hea | Sho | Elb | Wri | Hip | Kne | Ank | mAP | s/img |
|---|---|---|---|---|---|---|---|---|---|
| DeeperCut (2016) | 78.4 | 72.5 | 60.2 | 51.0 | 57.2 | 52.0 | 45.4 | 59.5 | 485 |
| Iqbal & Gall (2016) | 58.4 | 53.9 | 44.5 | 35.0 | 42.2 | 36.7 | 31.1 | 43.1 | 10 |
| **OpenPose** | **91.2** | **87.6** | **77.7** | **66.8** | **75.4** | **68.9** | **61.7** | **75.6** | **0.005** |

**+16 mAP sobre DeeperCut con 4 órdenes de magnitud de speedup**.

### COCO 2016 Keypoints Challenge

| Equipo | AP | AP⁵⁰ | AP⁷⁵ | APᴹ | APᴸ |
|---|---|---|---|---|---|
| **OpenPose (winner)** | **60.5** | **83.4** | **66.4** | 55.1 | **68.1** |
| G-RMI (Papandreou, top-down) | 59.8 | 81.0 | 65.1 | **56.7** | 66.7 |
| DL-61 | 53.3 | 75.1 | 48.5 | 55.5 | 54.8 |

OpenPose gana el challenge inaugural. Pierde levemente con G-RMI en escalas medianas (APᴹ) donde top-down todavía es competitivo por rescalear cada bbox.

### Ablation: PAFs vs midpoint

| Representación | mAP (PCKh 0.5, MPII val) |
|---|---|
| Midpoint, 1 punto | 78.4 |
| Midpoint, 2 puntos intermedios | 78.8 |
| **PAFs (paper)** | **81.6** |

**+3.2 mAP** porque PAFs codifican orientación, no solo posición.

### Runtime crece $O(1)$ con personas

- **CNN**: $O(1)$ en número de personas (corre una sola vez).
- **Parsing**: $O(n^2)$ pero con constantes pequeñas (0.58 ms para 9 personas vs. 99.6 ms de CNN).
- **Throughput**: 8.8 fps sobre video 1080×1920 con 19 personas en una GTX 1080.

A partir de **4-5 personas en escena**, OpenPose es estrictamente más rápido que cualquier top-down.

## Limitaciones reconocibles

Modos de falla documentados (Fig. 9 del paper):

1. **Poses raras** (atletas en posiciones extremas, ropa atípica) → confidence maps débiles.
2. **Partes faltantes o falsas** cuando la persona está parcialmente fuera del frame.
3. **Partes compartidas entre personas** → cuando dos personas se abrazan, un keypoint puede pertenecer a ambas y el matching greedy comete errores.
4. **Conexiones incorrectas** entre personas adyacentes → "esqueletos Frankenstein".
5. **Falsos positivos** en estatuas o animales antropomorfos.

Limitaciones estructurales:

- **Sin información temporal** — pose por frame, sin tracking.
- **Topología fija** — 13-19 conexiones hardcoded para esqueleto humano.
- **Sensibilidad a escala** — AP cae con personas pequeñas.
- **Sin manejo de incertidumbre** — cada predicción es un punto duro (esto lo introduce PifPaf con Laplace loss).
- **Licencia restrictiva** del código CMU bloqueó adopción comercial directa, motivando la migración a [BlazePose](/papers/blazepose-bazarevsky-2020) como alternativa libre mobile-first.

## Impacto y legado

- **>32k stars en GitHub** del repo CMU original.
- **Definió la era bottom-up**: PifPaf, PersonLab, HigherHRNet, DEKR, CID — todos descienden de PAFs.
- **Adopción industrial masiva**: fitness apps (Apple Fitness+, Tonal, Mirror), análisis deportivo, danza, medicina.
- **Versión extendida TPAMI 2019** (Cao et al.) extiende la arquitectura con redundant PAFs y mejora la precisión en escalas pequeñas.
- **Forks PyTorch** ([Hzzone/pytorch-openpose](https://github.com/Hzzone/pytorch-openpose)) son la única vía práctica en 2026 porque el código original está en Caffe (framework muerto desde 2018).

## Conexión con el laboratorio

El [Lab 17](/laboratorios/lab-17) usa OpenPose como **uno de dos modelos comparables** (junto con [PifPaf](/papers/pifpaf-kreiss-2019)) en un A/B test sobre Stanford 40 Actions:

- Se entrena un MLP simple sobre los **18 keypoints flatten** que OpenPose produce.
- Se compara accuracy del clasificador downstream contra el MLP entrenado sobre features de PifPaf.
- El lab demuestra empíricamente la **friction de adopción**: OpenPose requiere `git clone` + `wget` + `sys.path hack` mientras PifPaf es `pip install`. Esto es la diferencia entre research code académico y librería moderna mantenida.

Cross-links:

{{< cards >}}
  {{< card link="/laboratorios/lab-17" title="Lab 17 - Pose Recognition" subtitle="A/B test PifPaf vs. OpenPose + clasificación de acciones" icon="academic-cap" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="PifPaf (Kreiss 2019)" subtitle="La siguiente generación bottom-up que supera OpenPose en baja resolución" icon="document-text" >}}
  {{< card link="/papers/blazepose-bazarevsky-2020" title="BlazePose (Bazarevsky 2020)" subtitle="La respuesta mobile de Google a OpenPose" icon="document-text" >}}
  {{< card link="/fundamentos/pose-estimation" title="Fundamento: Pose Estimation 2D" subtitle="Bottom-up vs. top-down, heatmaps, OKS/PCK" icon="book-open" >}}
  {{< card link="/clases/clase-17" title="Clase 17 - Pose Recognition" subtitle="Recorrido teórico" icon="academic-cap" >}}
{{< /cards >}}
