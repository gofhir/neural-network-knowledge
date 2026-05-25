---
title: "Spatial Transformer Networks (STN)"
weight: 108
math: true
---

{{< paper-card
    title="Spatial Transformer Networks"
    authors="Jaderberg, Simonyan, Zisserman, Kavukcuoglu"
    year="2015"
    venue="NeurIPS 2015"
    pdf="/papers/stn-jaderberg-2015.pdf"
    arxiv="1506.02025" >}}
Jaderberg y colaboradores (Google DeepMind) introducen un modulo diferenciable que **aprende transformaciones espaciales** (afin, proyectiva, thin-plate spline) condicionadas en el input y las aplica end-to-end usando solo la perdida de la tarea final. Sin etiquetas geometricas, sin REINFORCE: backprop estandar. El STN se vuelve la primitiva canonica del **muestreo geometrico diferenciable** y es la base directa de la etapa de rectificacion en scene text recognition (RARE, ASTER, MORAN).
{{< /paper-card >}}

---

## El problema

Para 2015, las CNN ya dominaban clasificacion, deteccion y segmentacion, pero su **invariancia geometrica** seguia siendo limitada y rigida:

- **Max-pooling** $2 \times 2$ provee invariancia translacional **local**, dentro de la ventana de pooling. No maneja rotaciones de $30^\circ$ ni cambios de escala $\times 2$ sin pagar discriminabilidad.
- **Weight sharing convolucional** otorga equivariancia translacional (en teoria, exacta), pero no rotacional ni de escala.
- **Data augmentation** (rotaciones, zooms, crops) es la solucion practica, pero infla el dataset, no es aprendida y no permite a la red **decidir en inferencia** que transformacion aplicar a la muestra de turno.
- Capsules (Hinton 2011), scattering networks (Bruna-Mallat 2013) y hard attention con RL (RAM, DRAM) intentan equivariancia explicita, pero o no escalan o requieren gradientes de alta varianza (REINFORCE).

El gap: no existia un componente (a) plenamente diferenciable, (b) entrenable por backprop estandar, (c) capaz de aplicar una transformacion espacial **arbitraria** sobre el feature map y (d) que no requiriera supervision de la transformacion. STN cierra ese gap.

---

## Arquitectura del Spatial Transformer

El ST es un sub-modulo que recibe un feature map $U \in \mathbb{R}^{H \times W \times C}$ y produce $V \in \mathbb{R}^{H' \times W' \times C}$ aplicando una transformacion geometrica $T_\theta$ aprendida y condicionada en el propio $U$. Tres componentes:

### Localisation network

Una red pequena $f_{\text{loc}}: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^d$ que regresa los parametros $\theta = f_{\text{loc}}(U)$. La dimension $d$ depende de la familia de transformaciones (afin 2D: $d=6$; proyectiva: $d=8$; TPS con $K$ control points: $d=2K$). La arquitectura interna es libre: FCs para MNIST, mezclas conv + FC para SVHN y CUB. El detalle critico es la **inicializacion del ultimo layer**: pesos a cero y bias igual a la identidad. Asi el ST arranca como cable transparente y descubre transformaciones utiles gradualmente, en vez de explotar a un crop degenerado en las primeras iteraciones.

### Grid generator

Define una grilla regular en coordenadas del output $G = \{(x_i^t, y_i^t)\}$ y aplica $T_\theta$ para obtener **coordenadas fuente** en el input:

$$
(x_i^s, y_i^s) = T_\theta(G_i)
$$

Para una afin 2D:

$$
\begin{pmatrix} x_i^s \\ y_i^s \end{pmatrix} =
\begin{pmatrix} \theta_{11} & \theta_{12} & \theta_{13} \\ \theta_{21} & \theta_{22} & \theta_{23} \end{pmatrix}
\begin{pmatrix} x_i^t \\ y_i^t \\ 1 \end{pmatrix}
$$

La transformacion va **de output a input** (backward warping, como en texture mapping de graficos): para cada pixel del output se calcula desde donde muestrear, lo que evita huecos.

### Sampler

Para cada coordenada fuente $(x_i^s, y_i^s)$ se extrae un valor del input con interpolacion **bilineal sub-diferenciable**:

$$
V_i^c = \sum_{n=1}^{H} \sum_{m=1}^{W} U_{nm}^c \, \max(0, 1 - |x_i^s - m|) \, \max(0, 1 - |y_i^s - n|)
$$

Los gradientes $\partial V_i^c / \partial U_{nm}^c$ y $\partial V_i^c / \partial x_i^s$ son analiticos y fluyen hacia atras por el grid generator hasta $\theta$ y, desde ahi, hasta los pesos de la localisation network. En la practica la sumatoria $H \times W$ se evalua solo en los **4 vecinos** del pixel fuente, asi que es $O(1)$ por pixel del output. Perfecto para GPU.

---

## Tipos de transformacion parametrizada

El ST es agnostico a la familia $T_\theta$ siempre que sea diferenciable en $\theta$:

| Familia | Parametros | Cobertura | Uso tipico |
|---|---|---|---|
| **Atencion** (location + scale) | 3 | Crop con zoom isotropico | Hard attention diferenciable |
| **Afin 2D** | 6 | Traslacion, escala anisotropica, rotacion, shear | Default en SVHN, CUB |
| **Proyectiva** (homografia) | 8 | Perspectiva | Objetos vistos oblicuamente |
| **Thin-Plate Spline (TPS)** | $2K$ (tipicamente $K = 16$ o $20$) | Deformaciones no rigidas | **Rectificacion de texto curvado** (RARE, ASTER) |

El TPS (Bookstein 1989) suma un termino afin mas funciones radiales $\phi(r) = r^2 \log r^2$ centradas en los puntos de control. Es la opcion correcta para des-curvar texto o revertir distorsiones elasticas. El paper tambien describe una extension 3D con interpolacion trilineal (apendice A.3).

---

## Propiedades clave

- **Diferenciable end-to-end.** Backprop estandar a traves de sampler $\to$ grid $\to$ localisation net. Sin REINFORCE, sin etiquetas de transformacion.
- **Modular y "drop-in".** Se inserta en cualquier punto de cualquier red, no solo al input. En capas intermedias warpea **feature maps abstractos**.
- **Supervision solo desde la loss final.** La transformacion emerge porque facilita la tarea.
- **Multiples STs paralelos.** En CUB-200 cada ST se especializa en una parte distinta del ave (cabeza, cuerpo) sin etiquetas de keypoint.
- **Multiples STs en serie.** Permiten transformaciones progresivamente mas abstractas.
- **Costo computacional bajo.** ST-CNN Multi en SVHN es solo ~6% mas lento que la baseline.

---

## Experimentos del paper

Los autores validan el ST en cuatro escenarios crecientes en dificultad: MNIST con distorsiones controladas, secuencias de digitos en SVHN, clasificacion fine-grained en CUB-200 y co-localizacion semi-supervisada con triplet loss.

### MNIST distorsionado (error %)

| Modelo | R (rot) | RTS (rot+trans+scale) | P (perspectiva) | E (elastica) |
|---|---|---|---|---|
| FCN | 2.1 | 5.2 | 3.1 | 3.2 |
| CNN | 1.2 | 0.8 | 1.5 | 1.4 |
| ST-CNN Aff | 0.7 | 0.5 | 0.8 | 1.2 |
| ST-CNN TPS | **0.7** | **0.5** | **0.8** | **1.1** |

TPS gana claramente en deformaciones elasticas (E) porque puede revertir deformaciones no rigidas. En clutter pesado ($60 \times 60$ con 6 distractores), CNN logra 3.5% error y **ST-CNN baja a 1.7%**.

### SVHN multi-digit (error % secuencia)

| Modelo | 64 px | 128 px |
|---|---|---|
| Maxout CNN (Goodfellow 2013) | 4.0 | -- |
| DRAM (Ba 2015, attention + RL) | 3.9 | 4.5 |
| ST-CNN Multi (4 STs) | **3.6** | **3.9** |

ST-CNN Multi supera a DRAM (un modelo recurrente con atencion reforzada por RL y ensemble) usando **una sola pasada forward**. Las afines visualizadas muestran como cada ST recorta progresivamente la region relevante de la secuencia.

### CUB-200-2011 fine-grained birds (accuracy %)

| Modelo | Accuracy |
|---|---|
| Bilinear CNN (Lin 2015) | 80.9 |
| CNN baseline (Inception, 224 px) | 82.3 |
| 2x ST-CNN, 448 px | 83.9 |
| 4x ST-CNN, 448 px | **84.1** |

Sin ninguna anotacion de keypoint, los STs aprenden **localizacion emergente** de partes (cabeza, cuerpo) solo por gradient descent sobre cross-entropy de clase. Cuando los STs paralelos se inicializan a la misma region, colapsan a la misma parte; cuando se inicializan a posiciones distintas del plano espacial, **se reparten el ave** y mejoran la accuracy monotonicamente al aumentar el numero de STs (1 -> 2 -> 4).

### Co-localizacion semi-supervisada

En un escenario sin etiquetas de clase pero con conjuntos de imagenes que comparten un objeto comun, el ST se entrena con **triplet loss** sobre embeddings de crops producidos por el transformer:

$$
\sum_{n=1}^{N} \sum_{m \neq n}^{M} \max\!\Big(0,\; \|e(I_n^{T}) - e(I_m^{T})\|_2^2 - \|e(I_n^{T}) - e(I_n^{\text{rand}})\|_2^2 + \alpha\Big)
$$

donde $I_n^T = T_\theta(I_n)$ es el crop producido por el ST e $I_n^{\text{rand}}$ es un parche aleatorio del mismo input. En MNIST sobre canvas $84 \times 84$, el ST localiza correctamente el 100% de los digitos en el caso translated y 75-94% con clutter, **sin ningun label de bounding box**. Es uno de los primeros ejemplos claros de localizacion por gradient descent puro sobre una loss de similitud.

---

## Aplicaciones en Scene Text Recognition

Es la linea de impacto **directamente relevante** para la clase 21: el STN aparece explicitamente en la slide *"Image preprocessing stage: STN, TPS, Other networks"*. La cadena cronologica:

- **RARE** (Shi, Wang, Lyu, Yao, Bai -- CVPR 2016). "Robust scene text recognition with Automatic REctification". Primer uso explicito de STN + TPS como modulo de rectificacion. La localisation net predice las coordenadas de $K = 20$ control points; el TPS resuelve la deformacion inversa; el sampler produce una imagen rectificada que entra a un recognizer attention-based. Mejora notable sobre CRNN en irregular text.
- **ASTER** (Shi, Yang, Lyu, Bai -- TPAMI 2018). Refina RARE: STN bidireccional que predice keypoints de la linea superior e inferior del texto, aplica TPS y luego encoder-decoder con atencion. Gana ~5% en irregular text (ICDAR15, CUTE80, Total-Text) y es el baseline canonico hasta ~2020.
- **MORAN** (Luo, Jin, Sun -- Pattern Recognition 2019). Generaliza ASTER con rectificacion pixel a pixel mas recognizer attention-based. Mantiene el sampler bilineal de STN como bloque base.

La linea de scene text recognition 2016-2022 se puede ordenar en dos paradigmas:

1. **Rectify then recognize.** RARE, ASTER, MORAN -- todos usan STN+TPS o variantes.
2. **Recognize without rectification.** CRNN puro, SATRN con 2D self-attention, ABINet con language model, ABCNet con BezierAlign.

La clase 21 contrasta justamente estos paradigmas.

---

## Limitaciones

- **Riesgo de colapso a identidad** si la inicializacion es mala o la loss no penaliza claramente la transformacion. ASTER y MORAN anaden inicializacion con keypoints predichos para acelerar la convergencia del TPS.
- **Aliasing del sampler bilineal en downsampling agresivo.** El paper aplica average pool tras el ST en algunos modelos como parche.
- **Sensible a inicializacion** (pesos cero + bias identidad es estandar y hay que respetarlo). Si hay multiples STs paralelos, se inicializan a posiciones distintas para evitar que todos converjan a la misma region.
- **Una sola transformacion global por ST.** Oclusiones, multiples objetos con poses muy distintas o discontinuidades requieren STs paralelos -- el numero es hiperparametro fijo, sin seleccion dinamica.
- **No produce equivariancia formal.** El ST aplica una transformacion, pero las layers downstream siguen viendo features pasadas por un warp; no son representaciones explicitamente equivariantes (para eso, group-equivariant CNNs).
- **lr de la localisation network mucho menor.** Detalle critico no obvio: en SVHN es 1/10 de la lr base; en CUB es $10^{-4}$ veces. Sin esto, el ST oscila y rompe entrenamiento.

---

## Impacto extended en vision profunda

El STN trasciende su problema original. La tecnica de **sampling bilineal con gradiente analitico** se vuelve un building block ubicuo:

- **Deformable Convolutions** (Dai et al., ICCV 2017). Generaliza la idea STN al kernel mismo: aprende offsets por posicion de receptive field, $y(p_0) = \sum_{p_n} w(p_n) \cdot x(p_0 + p_n + \Delta p_n)$. Usa el mismo sampler bilineal sub-diferenciable. Backbone clave para detection (DCNv1, DCNv2).
- **Dynamic Filter Networks** (De Brabandere et al., NeurIPS 2016). En vez de transformar el input, aprende **el filtro convolucional** condicionado en el input.
- **RoIAlign** (Mask R-CNN, He et al., ICCV 2017). Hereda directamente el sampler bilineal STN para hacer RoI pooling diferenciable -- mejora sobre el RoI pooling no diferenciable de Fast R-CNN.
- **NeRF y differentiable rendering** (Mildenhall et al., 2020). El sampler bilineal/trilineal diferenciable con gradientes analiticos sobre coordenadas es exactamente la misma tecnica matematica.
- **Optical flow** (RAFT, FlowNet). El warping de features por flujos predichos usa el mismo sampler.
- **Vision Transformer attention** (Dosovitskiy et al., 2020). Filosoficamente cercano: atender un patch a otros patches es una forma diferenciable de "elegir donde mirar". ViT no usa STN, pero comparte el espiritu.

---

## Contraste con ABCNet

[ABCNet](/papers/abcnet-liu-2020) (Liu et al., CVPR 2020 oral) **no usa STN ni rectifica**. En su lugar:

- Modela cada texto curvado con una **curva de Bezier** (8 puntos de control).
- Muestrea features directamente a lo largo de la curva con **BezierAlign** (un sampler bilineal a lo largo de la curva, tambien diferenciable -- heredando del STN la tecnica de muestreo, no la idea de rectificar).

El contraste pedagogico es perfecto:

| Paradigma | Idea | Geometria | Recognizer |
|---|---|---|---|
| **STN + TPS** (RARE, ASTER, MORAN) | "Enderezar el input antes del recognizer" | Rectificar a rejilla rectilinea | Recognizer rectilineo (CRNN, attention) |
| **ABCNet + BezierAlign** | "Aceptar la curva y muestrear sobre ella" | Mantener la curva como primitiva | Operacion local sobre features curveadas |

Ambos comparten la herencia matematica del **muestreo geometrico diferenciable**. Difieren en si la curva se elimina (rectificacion) o se preserva (Bezier).

---

## Por que importa

El paper aporta una primitiva arquitectonica simple y profunda: **muestreo geometrico diferenciable** desde un input hacia un output, con la transformacion condicionada en el propio input via una pequena red de regresion y aplicada via interpolacion bilineal. Es un modulo "drop-in" entrenable por backprop estandar, sin etiquetas de transformacion, que recupera invariancia frente a transformaciones grandes que el max-pooling y la convolucion no manejan bien por si solos.

Sus resultados experimentales son solidos pero quiza menos influyentes que su **idea matematica**. La tecnica de "sampling bilineal con gradiente analitico sobre coordenadas fuente" se ha convertido en building block ubicuo: aparece en RoIAlign (detection), en deformable convolutions, en NeRF y differentiable rendering, en optical flow estimation, en RAFT, en VITON y try-on networks, y de forma central en **scene text recognition** (RARE, ASTER, MORAN), donde TPS sobre STN es la receta canonica para rectificar texto curvado o en perspectiva.

Para el curso, este paper es la base teorica para entender por que la clase 21 dedica una slide entera a "Image preprocessing stage: STN, TPS, Other networks" -- y por que ese paradigma de *rectify-then-recognize* compite (y a veces pierde) con paradigmas mas recientes que aceptan la geometria arbitraria del texto sin rectificar (ABCNet con BezierAlign, SATRN con 2D self-attention). Cualquier ingeniero que quiera entender la linea historica de scene text recognition tiene que pasar por aqui.

---

## Notas y enlaces

- El paper introduce el sampler bilineal diferenciable con gradiente analitico que se convertira en building block ubicuo de la decada siguiente.
- La **localisation network** suele ser sorprendentemente pequena: `fc[32]-fc[32]` basta para SVHN, justamente porque solo regresa pocos parametros geometricos.
- La **inicializacion** (identidad para 1 ST, posiciones distintas para STs paralelos) y la **lr reducida** de la localisation net son los dos detalles practicos no obvios.
- Codigo de la comunidad: implementaciones en PyTorch (`F.affine_grid`, `F.grid_sample`) y en TensorFlow (`tf.contrib.image.dense_image_warp`). El sampler bilineal de PyTorch es directamente el sampler STN.

**Fundamentos relacionados:** [Scene Text Recognition](/fundamentos/scene-text-recognition) · [Redes Convolucionales](/fundamentos/redes-convolucionales) · [Mecanismo de Atencion](/fundamentos/mecanismo-atencion).

**Papers relacionados:** [ABCNet (Liu 2020)](/papers/abcnet-liu-2020) -- paradigma contrastante (BezierAlign sin rectificacion). [CRNN (Shi 2017)](/papers/crnn-shi-2017) -- recognizer canonico que ASTER/MORAN sustituyen tras la rectificacion STN+TPS. [Text Recognition in the Wild (Chen 2020)](/papers/text-recognition-wild-chen-2020) -- survey que organiza la era de scene text recognition con y sin rectificacion.

**Clase:** [Clase 21 -- Scene Text Detection y Recognition](/clases/clase-21).
