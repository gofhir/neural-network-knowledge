---
title: "Wan-Animate por dentro"
weight: 4
math: true
---

El segundo notebook anima el personaje de una imagen con el movimiento de un video. La clase enseñó [FOMM](/papers/fomm-siarohin-2019) para esto; el notebook usa [Wan2.2-Animate-14B](/papers/wan-animate-2025), seis años posterior. Entre ambos hay una inversión que conviene leer despacio.

## La inversión

| | FOMM (2019) | Wan-Animate (2025) |
|---|---|---|
| Arquitectura | U-Net + campo de movimiento denso | **Diffusion Transformer**, 14B parámetros |
| Movimiento del cuerpo | ~10 keypoints **aprendidos sin supervisión** + jacobianos | Esqueleto **explícito** de un detector supervisado |
| Rostro | los mismos keypoints | Features **implícitas**, por cross-attention |
| Fondo | el de la imagen, deformado | generado, anclado a la imagen |
| Escala | corre en una GPU modesta | 34,5 GB en bf16, pensado para GPUs de 80 GB |

**FOMM aprendía los keypoints sin supervisión; Wan-Animate usa un detector supervisado.** Es al revés de lo que uno esperaría de la evolución del campo, y la razón es histórica: en 2019 no existía un estimador de pose lo bastante bueno y general, así que Siarohin tuvo que descubrir las "partes" del objeto desde cero — y eso es justamente lo que hacía a FOMM funcionar con caras, cuerpos y hasta objetos no humanoides. En 2025 ese detector existe.

## Cuerpo explícito, rostro implícito

La asimetría es deliberada y el paper justifica cada mitad por separado (§3.3).

**Para el cuerpo**, las alternativas eran esqueleto 2D o SMPL renderizado. SMPL representa mejor las relaciones entre extremidades, pero:

> *"las imágenes SMPL renderizadas contienen información de forma del personaje. Esto puede hacer que el modelo se apoye en las pistas de forma embebidas en la señal de movimiento, **lo que complica el aprendizaje de consistencia de identidad**."*

**Para el rostro**, el análogo del esqueleto serían los landmarks, pero:

> *"sufren pérdida de detalle fino durante la extracción (...) y, al ser señales densas, exigen alta precisión; si no, **comprometen severamente la consistencia de identidad, especialmente en escenarios cross-identity con disparidad significativa de forma facial**."*

{{< concept-alert type="clave" >}}
La simetría del argumento es lo interesante: para el **cuerpo**, una señal *demasiado informativa* (SMPL) filtra la identidad equivocada; para el **rostro**, una señal *demasiado pobre* (landmarks) destruye la identidad correcta.

El punto óptimo cae en lados opuestos del mismo eje. No hay una regla general de "más señal es mejor" ni de "menos es mejor": depende de qué información contamina y cuál falta.
{{< /concept-alert >}}

El Face Adapter reutiliza el encoder de LIA (Wang et al., 2022) con **Linear Motion Decomposition** para ortogonalizar las features, redimensiona cada rostro a **512 × 512**, lo comprime a un vector por frame, lo alinea temporalmente con convoluciones causales 1D y lo inyecta por cross-attention **cada 5 capas del DiT de 40**, o sea en 8 puntos.

Esa cifra de 512 × 512 tiene una consecuencia práctica que se explota en [el playbook de alineación](06-el-playbook-de-alineacion): cuanto más pequeño sea el rostro en el video de control, más se interpola para llegar a esa resolución, y peor queda la señal que produce la sincronía labial.

## El modelo nunca ve keypoints

Este es el punto que reordena todo lo demás. El docstring de `fetch_preprocess_models.py`:

> *"La sustitución es segura porque **Wan2.2-Animate nunca ve keypoints. Ve `src_pose.mp4`, un esqueleto renderizado que pasa por el VAE**, de modo que cualquier estimador que produzca el mismo layout COCO-WholeBody de 133 puntos renderiza la misma imagen."*

Encaja exacto con el paper (§3.3): *"estos frames de pose son comprimidos por el Wan-VAE para alinearse espacial y temporalmente con los latentes objetivo"*. **La interfaz del modelo es una imagen, no coordenadas.**

Eso permite una sustitución con ahorro notable:

| | Oficial | En el notebook |
|---|---|---|
| Estimador de pose | ViTPose-H wholebody, **2.549 MB**, exportado como un directorio de **394 archivos** | DWPose (RTMPose-l destilado), **134 MB**, un archivo |
| Detector | YOLOv10m, **AGPL-3.0** | YOLOX-l, Apache-2.0 |
| Total | 2.742 MB | **351 MB** |

El propio docstring explica el porqué: *"En Colab eso son minutos de round-trips HTTP antes de procesar un solo frame"*, y *"YOLOv10 es AGPL-3.0, una licencia incómoda de heredar en un pipeline por dibujar una caja"*.

Pero la propiedad tiene su reverso, y aparece en [el resultado](/laboratorios/lab-44): como el esqueleto **es** la señal, lo que el estimador dibuje se convierte en verdad para el difusor — incluidos los miembros extrapolados fuera de cuadro con keypoints de baja confianza.

## Los dos modos

**Animation Mode** (el que usa el notebook): el personaje de la imagen se anima con el movimiento del video, **y el fondo sale de la imagen**. Es un I2V.

**Replacement Mode**: el mismo personaje se inserta *dentro* del video de referencia, heredando su fondo e iluminación. Ahí entra la `Relighting LoRA`.

La diferencia práctica al elegir insumos: **el fondo del resultado es el de la foto, no el del video de movimiento**.

## La aritmética de latentes, verificada

El runner imprime, para 77 frames a 832×464:

```
[denoise] 832x464 x 77f -> latents [21, 58, 104] = 31,668 tokens
```

Cada número se puede reconstruir:

| Paso | Cuenta |
|---|---|
| Compresión espacial del VAE (8×) | $464/8 = 58$ · $832/8 = 104$ |
| Compresión temporal del VAE (4×) | $(77-1)/4 + 1 = 20$ |
| **+1 latente de referencia** | **21** |
| Voxels | $21 \times 58 \times 104 = 126.672$ |
| Patchify 2×2 del DiT | $126.672 / 4 = \mathbf{31.668}$ |

El `+1` es el latente de referencia del §3.2 —*"concatenado con los latentes condicionales a lo largo de la dimensión temporal"*— y §4.3 lo confirma sin ambigüedad: *"cada segmento de inferencia consiste en **78 frames**. Un frame se reserva estáticamente para la imagen del personaje"*.

También explica de dónde sale la regla $4n+1$ del conteo de frames: con $F = 4n+1$, la cuenta $(F-1)/4+1 = n+1$ es entera. Cualquier otro valor deja un latente parcial.

## Segmentos, y el mecanismo anti-drift

`--frame-num` es el tamaño del **segmento** —eso es lo que limita la VRAM—; `--max-frames` es la longitud **total**. Si excede el segmento, se generan varios encadenados:

```
[preprocess] 128 frames -> 149 after padding to whole segments (2 x 77f, 5f overlap)
[denoise] 2 segment(s) of 77f with 5f overlap (stride 72)
[mem:overlap-decode] ... peak 15.72 GB | 13.8s
[vae_decode] seg 1/2 kept 77 frames    seg 2/2 kept 72 frames (running total 149)
```

Esos 13,8 segundos entre segmentos son el mecanismo del §3.2:

> *"para permitir animación de longitud arbitraria, la generación de segmentos posteriores debe condicionarse en los **últimos frames del segmento previo**. Esto provee guía temporal y asegura continuidad."*

Los últimos 5 latentes del segmento 1 se **decodifican a píxeles** y se re-inyectan como condición del segmento 2. Por eso el segundo aporta 72 frames y no 77: los 5 del solape se descartan. $77 + 72 = 149$, luego recortado a los 128 pedidos.

El valor 5 no es arbitrario: `--refert-num` **solo acepta 1 o 5**, porque §4.3 dice *"se usan 1 o 5 frames como frames de referencia temporal"*. No hay otros porque el paper no entrenó otros.

## La resolución la decide la imagen

```
[preprocess] reference 832x468 -> 832x464
[denoise] using 832x464 from preprocessing (not 832x480): the pose video was rendered
          at that size to preserve the reference image's aspect ratio.
```

Dos cosas pasan ahí. Primero, §4.3: *"en Animation Mode, el aspect ratio de salida se ajusta al de la imagen del personaje de entrada"*. **El preset no manda; la imagen sí.** Una foto vertical produce salida vertical aunque se pida `832×480`.

Segundo, 468 no es múltiplo de 16 y debe serlo: el VAE comprime 8× y el DiT hace patchify 2×, $8 \times 2 = 16$. El pipeline baja a 464 ($29 \times 16$).

Consecuencia para elegir la imagen: si se quiere el **aspecto nativo de entrenamiento** del modelo, la referencia debe ser apaisada 16:9.

## Las cinco etapas, en procesos separados

```
wan_t4/stages/
    preprocess.py     ①  video -> esqueleto + recortes de rostro   (CPU)
    text_encode.py    ②  prompt -> embeddings umT5
    clip_encode.py    ③  imagen de referencia -> embedding CLIP
    denoise.py        ④  el DiT: 40 capas x N pasos                <- la etapa cara
    vae_decode.py     ⑤  latentes -> frames -> MP4
```

Cada una es un `subprocess` independiente, y entre una y otra la VRAM vuelve a cero. Es el mismo patrón `.cuda()`/`.cpu()` de [TorToise](01-la-arquitectura-de-tortoise) llevado un nivel más allá: cuando vaciar la caché de PyTorch ya no alcanza, se mata el proceso.

Eso habilita el mecanismo de recuperación: `--from denoise` o `--from decode` retoma desde artefactos parciales en disco, sin repetir lo hecho.

Los tiempos medidos en una L4 para la corrida de 128 frames:

| Etapa | Tiempo | Nota |
|---|---:|---|
| preprocess | 53,8 s | 128 frames a 2,6 fps de DWPose **en CPU, a propósito** |
| text | 51,6 s | carga del umT5: 39,6 s (I/O, no cómputo) |
| clip | 16,4 s | |
| **denoise** | **733,3 s** | 2 segmentos × 6 pasos × 52,0 s + carga + overlap |
| decode | 46,2 s | VAE por chunks temporales |
| **Total** | **901 s = 15,0 min** | |

Que DWPose corra en CPU es una decisión explícita del `requirements.txt`: *"el execution provider de CPU es el camino por defecto e intencionado: mantiene la GPU libre para el DiT"*.

## El prompt es secundario, y el paper lo dice

El notebook aconseja *"para desactivarlo se recomienda usar una oración genérica"*. El paper (§4.1) explica por qué no se puede desactivar del todo:

> *"Si bien Wan-Animate soporta cierto grado de control textual, **la señal de movimiento es el factor de control dominante**, lo que hace del control por texto una característica no central. En la práctica, recomendamos usar un prompt de texto por defecto."*

La etapa `text_encode` es obligatoria en el plan, así que el DiT siempre recibe embeddings; una frase genérica es lo más cercano a "sin condicionamiento".

El negative prompt por defecto está en chino y es el oficial de Wan: *"tonos chillones, sobreexpuesto, estático, detalles borrosos, subtítulos, ... dedos de más, manos mal dibujadas, caras mal dibujadas, deformado, extremidades deformadas, dedos fusionados, imagen inmóvil, fondo desordenado, tres piernas, mucha gente de fondo, caminando hacia atrás"*. El umT5 es multilingüe, así que funciona tal cual.

Vale la pena notar que **tres de sus ~25 términos son sobre manos**. Es donde el prior falla, y el laboratorio lo confirma.

## Dos mensajes que asustan y son benignos

**`Some weights ... were not used: ['motion_encoder.enc.net_app.convs.*.kernel', ...]`**

`motion_encoder` **es el Face Adapter**, y `net_app` su *appearance network* — la estructura de LIA que menciona §3.3. Los `.kernel` son buffers fijos de filtros de blur/`upfirdn`, no parámetros aprendidos: `diffusers` los regenera. Es, de paso, una confirmación de que el código implementa lo que el paper describe.

**`Quantized parameter condition_embedder.time_embedder.linear_* is required to remain in FP32`**

Los embeddings de timestep se fuerzan a FP32. Estándar en difusión: son sinusoidales de alta frecuencia y cuantizarlos rompe la noción de en qué paso del proceso se está.

---

**Siguiente:** [El presupuesto de memoria](05-el-presupuesto-de-memoria) — cuantización GGUF, el chequeo que rechaza lo que cabe, y cómo escala el tiempo.
