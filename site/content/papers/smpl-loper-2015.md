---
title: "SMPL: A Skinned Multi-Person Linear Model"
weight: 84
math: true
---

{{< paper-card
    title="SMPL: A Skinned Multi-Person Linear Model"
    authors="Loper, Mahmood, Romero, Pons-Moll, Black"
    year="2015"
    venue="SIGGRAPH Asia 2015"
    pdf="/papers/smpl-loper-2015.pdf"
    arxiv="" >}}
Modelo paramétrico realista del cuerpo humano: $M(\vec\beta, \vec\theta)$ → malla de 6890 vértices con 10 parámetros de shape ($\vec\beta$) y 72 de pose ($\vec\theta$, axis-angle por joint). Aprendido de ~4000 scans 3D, compatible con engines de animación estándar (Maya, Blender, Unity, Unreal) vía **Linear Blend Skinning**. Es el modelo paramétrico de cuerpo humano más usado en visión y gráficos del último decenio y el sustrato 3D bajo DensePose, HMR, VIBE, AMASS y la mayoría de pipelines de body recovery actuales.
{{< /paper-card >}}

---

## Contexto

A inicios de 2015 había una brecha entre:

- **Industria de animación** — usaba LBS estándar con rigging manual, fácil de renderizar pero limitado en realismo.
- **Academia** — modelos estadísticos como SCAPE eran realistas pero **incompatibles con engines de animación** (basados en deformaciones de triángulos, no LBS vertex-based).

SMPL es la primera *fusión funcional*: realismo aprendido de scans + LBS estándar + run-time real (>50fps).

## Ideas principales

### Modelo paramétrico

$$
M(\vec\beta, \vec\theta; \Phi) : \mathbb{R}^{|\beta|+|\theta|} \to \mathbb{R}^{3 \times 6890}
$$

con:
- $\vec\beta \in \mathbb{R}^{10}$ — shape (PCA de variación CAESAR).
- $\vec\theta \in \mathbb{R}^{72}$ — pose (3 axis-angle × 23 joints + 3 root global).

### Shape blend shapes

$$
B_S(\vec\beta; \mathcal{S}) = \sum_{n=1}^{|\beta|} \beta_n S_n
$$

con $S_n$ las componentes PCA de variación de forma sobre el dataset.

### Pose blend shapes (la innovación clave, Ec. 9)

$$
B_P(\vec\theta; \mathcal{P}) = \sum_{n=1}^{9K}\bigl(R_n(\vec\theta) - R_n(\vec\theta^*)\bigr) P_n
$$

donde $R_n(\vec\theta)$ es el $n$-ésimo elemento del **stack de matrices de rotación** $R(\vec\theta) \in \mathbb{R}^{9K}$. Condicionar sobre **elementos de rotación** (acotados en $[-1, 1]$) en vez de ángulos hace que las deformaciones generalicen mejor.

### Joint regressor

$$
J(\vec\beta) = \mathcal{J}(\bar T + B_S(\vec\beta))
$$

Matriz aprendida $\mathcal{J}$ que predice las posiciones 3D de los 23 joints como combinación lineal **esparsa** de los vértices superficiales rest-pose. Adaptar a un nuevo cuerpo solo requiere multiplicar.

### Linear Blend Skinning

$$
M(\vec\beta, \vec\theta) = W\bigl(T_P(\vec\beta, \vec\theta),\ J(\vec\beta),\ \vec\theta,\ \mathcal{W}\bigr)
$$

con $T_P = \bar T + B_S + B_P$ la malla rest deformada por blend shapes y $\mathcal{W}$ los blend weights. El paper experimenta tanto LBS estándar como Dual-Quaternion Blend Skinning (DQBS) — SMPL-LBS y SMPL-DQBS dan errores similares.

### Entrenamiento

1. **Multi-pose dataset** (1786 registrations, 40 sujetos): aprender $\mathcal{J}, \mathcal{W}, \mathcal{P}$ minimizando reconstrucción + simetría + regularización.
2. **Multi-shape dataset** (~3700 sujetos CAESAR en T-pose): pose-normalizar usando el modelo learned-so-far, luego PCA → $\bar T, \mathcal{S}$.
3. Iterar hasta convergencia.

## Resultados experimentales

### Generalization error (mean abs vertex, mm)

| Modelo | Model generalization | Pose generalization |
|---|---|---|
| LBS (vanilla) | ~7 | ~7.5 |
| BlendSCAPE | ~3.5 | ~5 |
| **SMPL-LBS** | **~3.2** | **~4.5** |
| **SMPL-DQBS** | **~3.2** | **~4.0** |

SMPL supera consistentemente a BlendSCAPE *con LBS estándar*, demostrando que aprender pose blend shapes vence a usar triangle deformations sofisticadas.

### Run-time

- 1 CPU core: ~5 ms por sample.
- BlendSCAPE: ~25 ms.
- En GPU con engine integrado: tiempo real en Unity 5.

## Limitaciones reconocibles

1. **Sin cara articulada ni manos**: solo 23 joints + root. Versiones posteriores: **SMPL-H** (manos), **FLAME** (cara), **SMPL-X** (todo, Pavlakos 2019).
2. **Topología única**: 6890 vértices fijos, no captura discapacidad, amputaciones.
3. **No incluye ropa**: el cuerpo SMPL es "desnudo".
4. **PCA lineal**: efectos no lineales (e.g., obesidad extrema con folds) no se capturan bien. **STAR** (Osman 2020) corrige.
5. **Pose blend shapes lineales en R**: aproximación local, imperfecta para poses extremas (yoga, contorsionismo).
6. **Diversidad demográfica** del CAESAR dataset es limitada.

## Por qué importa hoy

**Casi cualquier paper de body recovery posterior a 2015 lo cita o usa**:

- **Sucesores directos**: SMPL-H, FLAME, SMPL-X, STAR, SUPR, GHUM/GHUML.
- **Inferencia desde imagen**: SMPLify (2016), HMR (2018), VIBE (2020), PIFu, PROX, PARE, 4DHumans.
- **Datasets que dependen de SMPL**: DensePose-COCO (UV definidos sobre SMPL), AMASS, SURREAL, AGORA.
- **Industria**: Apple, Meta, Adobe, Disney usan SMPL (o variantes propietarias) en pipelines de avatares y mocap.
- **VR/AR**: cualquier pipeline de body tracking moderno (Quest, Vision Pro) usa modelos en la familia SMPL.

## Conexión con la clase 17

SMPL aparece explícitamente en el slide 35 del PDF de Clase 17: *"With the points correspondence they can generate the U and V images using the Skinned Multi-Person Linear (SMPL) model."*

Es el **modelo paramétrico subyacente a DensePose** — sin SMPL no hay coordenadas U/V. Conecta visión por computador con gráficos por computador y establece el lenguaje $(\vec\beta, \vec\theta)$ que aparece en cualquier paper moderno de body pose estimation.

## Notas y enlaces

- Project page: https://smpl.is.tue.mpg.de/
- Código Python oficial (`smplx`): https://github.com/vchoutas/smplx (incluye SMPL, SMPL-H, SMPL-X, MANO, FLAME)
- AMASS dataset: https://amass.is.tue.mpg.de/
- Análisis interno con código PyTorch/TF/JAX en el repositorio del curso.
