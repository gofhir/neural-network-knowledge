---
title: "Wan-Animate: Unified Character Animation and Replacement (2025)"
weight: 471
math: true
---

{{< paper-card
    title="Wan-Animate: Unified Character Animation and Replacement with Holistic Replication"
    authors="HumanAIGC Team, Tongyi Lab, Alibaba"
    year="2025"
    venue="arXiv:2509.14055"
    arxiv="2509.14055"
    pdf="/papers/wan-animate-2025.pdf" >}}
El modelo que anima el personaje en el [Laboratorio 44](/laboratorios/lab-44), en reemplazo del [FOMM](/papers/fomm-siarohin-2019) que enseña la clase. Un **Diffusion Transformer de 14B parámetros** construido sobre Wan-I2V que resuelve dos tareas con un mismo modelo: animar el personaje de una imagen con el movimiento de un video (*Animation Mode*), o insertarlo dentro de ese video reemplazando al original y heredando su iluminación (*Replacement Mode*). El cuerpo se controla con **esqueletos explícitos**; el rostro, con **features implícitas** del recorte facial.
{{< /paper-card >}}

---

## Qué cambió en seis años

Entre [FOMM](/papers/fomm-siarohin-2019) (2019) y este paper hay una inversión que vale la pena leer con cuidado:

| | FOMM (2019) | Wan-Animate (2025) |
|---|---|---|
| Arquitectura | U-Net + campo de movimiento denso | **Diffusion Transformer**, 14B |
| Movimiento del cuerpo | ~10 keypoints **aprendidos sin supervisión** + jacobianos | Esqueleto **explícito** de [ViTPose](/papers/vitpose-xu-2022) |
| Rostro | los mismos keypoints | Features **implícitas**, vía cross-attention |
| Base | entrenado desde cero por categoría | post-entrenamiento sobre un modelo fundacional de video |

Lo llamativo es la dirección: **FOMM aprendía los keypoints sin supervisión y Wan-Animate usa un detector supervisado**. En 2019 no existía un estimador de pose lo bastante bueno y general, así que Siarohin tuvo que descubrir las "partes" del objeto desde cero — y eso es lo que hacía a FOMM funcionar con caras, cuerpos y hasta objetos no humanoides. En 2025 ese detector existe, y reaprenderlo no tiene sentido.

## Los dos modos

**Animation Mode.** El personaje de la imagen se anima con el movimiento del video, **preservando el fondo de la imagen**. Es análogo a un Image-to-Video.

**Replacement Mode.** El mismo personaje, con el mismo movimiento, se integra **en el entorno del video**, reemplazando al sujeto original. Es Video-to-Video, y trae una `Relighting LoRA` propia para que la iluminación del personaje coincida con la escena.

La diferencia práctica es de dónde sale el fondo: de la imagen en el primero, del video en el segundo. Un único modelo entrenado conjuntamente cubre ambos; solo la Relighting LoRA es exclusiva del segundo.

## La formulación de entrada

Wan-I2V recibe tres cosas: latente de ruido, latente condicional y máscara binaria. Para I2V la imagen es el **primer frame**, así que la máscara vale 1 solo ahí. La animación de personajes impone tres requisitos distintos (§3.2):

1. La imagen es una **referencia de apariencia**, no un punto de partida: el contenido lo dictan las señales de control.
2. Para longitud arbitraria, cada segmento debe condicionarse en los **últimos frames del anterior**.
3. Los dos modos deben unificarse en una representación compatible.

La solución: el latente de referencia se concatena con los latentes condicionales **a lo largo de la dimensión temporal**, con máscara 1. Para la guía temporal se seleccionan los primeros latentes del segmento como *temporal latents*, también con máscara 1. Y hay un detalle elegante: *"el proceso de denoising genera la secuencia completa, incluidas las porciones correspondientes a las referencias. Los frames resultantes que corresponden a esas referencias se descartan después"*.

{{< concept-alert type="clave" >}}
**Esta descripción se puede verificar contando latentes.** En el [Laboratorio 44](/laboratorios/lab-44/04-wan-animate-por-dentro), el runner reporta para 77 frames a 832×464:

$$\texttt{latents [21, 58, 104]} = 31.668\ \text{tokens}$$

Y sale exacto: la compresión temporal es 4×, así que $(77-1)/4 + 1 = 20$ latentes de video, **más 1 de referencia** = 21. Espacialmente, $464/8 = 58$ y $832/8 = 104$. Y $21 \times 58 \times 104 = 126.672$, dividido por 4 del *patchify* 2×2 = **31.668**.

Ese `+1` es literalmente el latente de referencia del §3.2. El paper lo confirma en §4.3: *"cada segmento de inferencia consiste en **78 frames**. Un frame se reserva estáticamente para la imagen del personaje"*.
{{< /concept-alert >}}

## Las señales de control, y por qué son distintas

Aquí está la decisión de diseño más interesante del paper: **el cuerpo usa una señal explícita y el rostro una implícita**, y cada elección se justifica por separado (§3.3).

**Cuerpo — esqueleto 2D.** Las alternativas eran esqueleto o [SMPL](/papers/smpl-loper-2015) renderizado en 3D. SMPL representa mejor las relaciones entre extremidades en poses complejas, pero:

> *"las imágenes SMPL renderizadas contienen información de forma del personaje. Esto puede hacer que el modelo se apoye en las pistas de forma embebidas en la señal de movimiento, lo que complica el aprendizaje de consistencia de identidad."*

El esqueleto, en cambio, *"ofrece mejor generalidad, particularmente para personajes no humanoides con formas no convencionales"*. Se extrae con ViTPose, se **renderiza como imagen**, y el Body Adapter la comprime con el Wan-VAE para alinearla espacial y temporalmente con los latentes objetivo. Un detalle: *"el latente de referencia no recibe información de pose"*, lo que lo distingue temporalmente de los latentes objetivo.

**Rostro — features implícitas.** Los landmarks faciales serían el análogo del esqueleto, pero:

> *"sufren pérdida de detalle fino durante la extracción (...) y, al ser señales densas, exigen alta precisión; si no, **comprometen severamente la consistencia de identidad, especialmente en escenarios cross-identity con disparidad significativa de forma facial**."*

En su lugar se usa la **imagen cruda del rostro**, recortada usando el esqueleto. Como el entrenamiento es auto-supervisado, hay que desacoplar identidad de expresión para evitar *identity leakage*, y se usan dos estrategias: comprimir el rostro a un **latente 1D** (lo que reduce el almacenamiento de información de bajo nivel, específica de la identidad) y aplicar aumentaciones agresivas —escala, jitter de color, ruido— que introducen discrepancias deliberadas entre el rostro aumentado y el objetivo.

La arquitectura del Face Adapter reutiliza el encoder de LIA (Wang et al. 2022) con **Linear Motion Decomposition** para ortogonalizar las features. Los rostros se redimensionan a **512 × 512**, se comprimen a un vector por frame, se alinean temporalmente con convoluciones causales 1D, y se inyectan por cross-attention en "Face Blocks": **cada 5 capas del DiT de 40 capas, o sea 8 puntos de inyección**.

La simetría del argumento vale la pena: para el cuerpo, una señal *demasiado* informativa (SMPL) filtra la identidad equivocada; para el rostro, una señal *demasiado pobre* (landmarks) destruye la identidad correcta. El óptimo cae en lados opuestos.

## Detalles de inferencia que el laboratorio verifica

De §4.3, tres afirmaciones que se comprueban ejecutando:

**1. El aspecto lo manda la imagen.** *"En Animation Mode, el aspect ratio de salida se ajusta al de la imagen del personaje de entrada."* La resolución final se elige calculando un conteo de tokens objetivo a partir de una resolución estándar (1280×720) y escogiendo, para el aspecto dado, la que produzca el conteo más cercano. En el laboratorio eso aparece como el flag `--resolution-area` y como un mensaje explícito del runner cuando la resolución del preset no coincide con el aspecto de la imagen.

**2. Los frames de referencia temporal son 1 o 5.** *"Para cualquier segmento que no sea el primero, se usan 1 o 5 frames como frames de referencia temporal, tomados del final del segmento precedente."* Es exactamente el parámetro `--refert-num {1,5}` del runner: no admite otros valores porque el paper no entrenó otros.

**3. CFG está desactivada por defecto.** *"Para mantener alta eficiencia de inferencia, classifier-free guidance (CFG) está deshabilitada por defecto. Sin embargo, en escenarios donde se desea control más fino de la expresión facial, CFG puede habilitarse opcionalmente para la entrada de condicionamiento facial."*

Eso explica el `guidance_scale=1.0` que traen **todos** los presets del laboratorio. Y contrasta de forma instructiva con [TorToise](/papers/tortoise-tts-betker-2023), donde la guía sin condicionamiento sí se paga en inferencia y **cuesta 2,14× por paso** medido: mismo mecanismo, dos decisiones opuestas sobre dónde pagarlo.

**4. El prompt es secundario.** *"Si bien Wan-Animate soporta cierto grado de control textual, la señal de movimiento es el factor de control dominante, lo que hace del control por texto una característica no central. En la práctica, recomendamos usar un prompt de texto por defecto."* El notebook del laboratorio dice lo mismo con otras palabras: *"para desactivarlo se recomienda usar una oración genérica"*.

## Datos y entrenamiento

Videos centrados en personas —hablando, expresiones, movimiento corporal— filtrados con medidas de calidad estándar y verificando que **cada clip contenga un solo personaje consistente**. Los esqueletos extraídos cumplen doble función: anotación de la señal de movimiento y criterio de filtrado por comportamiento. Para Replacement Mode se usa SAM 2 (ver [SAM 3](/papers/sam3-meta-2025)) para extraer máscaras del personaje, y **QwenVL2.5-72B** para generar las descripciones textuales.

El entrenamiento carga cuatro modelos —DiT, T5, VAE y CLIP— con FSDP para los dos que dominan la memoria y Context Parallelism (RingAttention + Ulysses) para el DiT.

## Resultados

Auto-reconstrucción sobre un benchmark propio: el primer frame como referencia, el resto como señal de movimiento.

**Cuerpo completo**, contra frameworks de animación de personajes:

| Método | SSIM ↑ | LPIPS ↓ | FVD ↓ |
|---|---|---|---|
| Moore-AA | 0,761 | 0,288 | 170,07 |
| MicmicMotion | 0,742 | 0,307 | 184,71 |
| Unianimate | 0,787 | 0,271 | 155,03 |
| StableAnimator | 0,794 | 0,265 | 147,92 |
| **Wan-Animate** | **0,813** | **0,227** | **118,65** |

**Retratos**, contra métodos especializados en animación facial:

| Método | SSIM ↑ | LPIPS ↓ | FVD ↓ |
|---|---|---|---|
| LivePortrait | 0,811 | 0,231 | 118,67 |
| AniPortrait | 0,791 | 0,252 | 135,08 |
| X-portrait2 | 0,825 | 0,212 | 98,03 |
| SkyReel-A1 | 0,821 | 0,231 | 101,45 |
| **Wan-Animate** | **0,834** | **0,205** | **94,65** |

El argumento del paper no es solo el número: los métodos previos son *"limitados en comprehensividad — los modelos guiados por cuerpo carecen de reenactment de expresión efectivo, los guiados por expresión no incluyen el cuerpo"*. La evaluación humana se hace contra dos sistemas **cerrados** (Runway Act-two y ByteDance DreamActor-M1), porque son los únicos comparables.

## Limitaciones reconocibles

- **Escala.** 14B parámetros, 34,5 GB de pesos en bf16 (el repo los publica en fp32: **72,1 GB**), pensado para GPUs de 80 GB. Correrlo en Colab exige cuantización GGUF agresiva y ejecución por etapas en procesos separados.
- **La cuantización toca el face adapter.** El propio notebook del laboratorio advierte que `Q3_K_M` distorsiona rostros porque cuantiza también ese componente.
- **Un solo personaje.** El filtro de datos lo dice: clips con *"un único personaje consistente"*. Con dos personas en el video, la extracción de pose se confunde.
- **La calidad del esqueleto acota todo.** El manejo de errores del notebook es explícito: *"si el esqueleto está incorrecto o inexistente, el problema es extracción de pose, no el modelo de difusión"*.

## Una consecuencia arquitectónica útil

El modelo **nunca ve keypoints**: ve `src_pose.mp4`, un esqueleto **renderizado** que pasa por el VAE. Eso hace que cualquier estimador que produzca el mismo layout COCO-WholeBody de 133 puntos sea intercambiable, y el laboratorio lo aprovecha: sustituye ViTPose-H (2.549 MB, exportado como un directorio de 394 archivos) más YOLOv10 (licencia AGPL-3.0) por **DWPose** (134 MB, un archivo, Apache-2.0) más YOLOX-l. **351 MB en vez de 2.742 MB**, sin cambiar lo que el modelo recibe.

No es un atajo: es una propiedad de la interfaz. Y tiene su reverso — como el esqueleto es una imagen, **lo que el estimador dibuje se convierte en verdad para el difusor**, incluidos los miembros extrapolados fuera de cuadro.

---

**Ver también:** [FOMM](/papers/fomm-siarohin-2019) (el método que enseña la clase) · [TorToise-TTS](/papers/tortoise-tts-betker-2023) (la otra mitad del laboratorio) · [ViTPose](/papers/vitpose-xu-2022) · [Latent Diffusion](/papers/latent-diffusion-rombach-2022) · [Classifier-Free Guidance](/papers/classifier-free-guidance-ho-2022) · [Clase 44](/clases/clase-44) · [Lab 44](/laboratorios/lab-44).
