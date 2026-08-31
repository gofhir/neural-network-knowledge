---
title: "Lab 44 - Deep Fakes: clonación de voz con TorToise y animación con Wan2.2-Animate"
weight: 440
math: true
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga (DCC, Pontificia Universidad Católica de Chile)
**Módulo:** Audio y Video — **último laboratorio del diplomado**
**Notebooks origen:** `clase_44/material/Laboratorio/` — tres notebooks encadenados
**Notebooks ejecutados:** [1 · voz](/notebooks/lab44a.ipynb) ([HTML](/notebooks-html/lab44a.html)) · [2 · animación](/notebooks/lab44b.ipynb) ([HTML](/notebooks-html/lab44b.html)) · [3 · unión](/notebooks/lab44c.ipynb) ([HTML](/notebooks-html/lab44c.html))

## Encuadre

La contraparte práctica de la [clase 44](/clases/clase-44), y el laboratorio que **no ejecuta lo que la clase enseña**.

La clase explicó deep fakes con [SV2TTS](/papers/sv2tts-jia-2018) (Jia et al., 2018) del lado del audio y [FOMM](/papers/fomm-siarohin-2019) (Siarohin et al., 2019) del lado del video. Los notebooks usan **[TorToise-TTS](/papers/tortoise-tts-betker-2023)** (Betker, 2023) y **[Wan2.2-Animate-14B](/papers/wan-animate-2025)** (Alibaba, 2025). La huella de la migración está a la vista: el sufijo `v18` del primer archivo, y una celda de créditos que todavía cita el Colab de `RealTimeVoiceCloning` mientras el código clona `tortoise-tts`.

No es un detalle administrativo. Entre una generación y otra cambia **dónde se paga la calidad**, y ese es el hilo que recorre el laboratorio.

{{< concept-alert type="clave" >}}
**El mismo mecanismo, dos decisiones opuestas sobre dónde pagarlo.**

TorToise ofrece *classifier-free guidance* como opción de inferencia: `cond_free=True` hace dos pasadas por paso de difusión y las combina como $\text{cond}\cdot(k{+}1) - \text{uncond}\cdot k$ con $k=2$ — escala 3, el mismo mecanismo que el `guidance_scale` de Stable Diffusion. Medido: **2,14× más lento por paso** (18,05 → 8,44 it/s), contra la predicción teórica de 2×. Es lo que separa `ultra_fast` de `fast`, y la causa real del timbre plano que el enunciado llama "sonido robótico".

Wan-Animate llega tres años después al mismo problema con un modelo de 14B, donde ese factor 2 sería letal. Su paper lo resuelve al revés: *"para mantener alta eficiencia de inferencia, CFG está deshabilitada por defecto"*, y un **LoRA de destilación de 704 MB** trae la guía horneada en los pesos. El docstring del pipeline lo cuantifica: *"40 pasos con CFG se convierten en 4-6 sin ella, 13-20× menos cómputo del DiT"*.

Mismo problema. Uno lo paga en inferencia, el otro lo pagó en entrenamiento.
{{< /concept-alert >}}

![Un frame del resultado: el personaje de la imagen de referencia animado con el movimiento del video de control](/laboratorios/lab-44/deepfake-frame.jpg)

La tesis del laboratorio, en una línea: **la clase pregunta qué parte de la salida estaba en la entrada y qué parte la puso el prior; acá esa pregunta se puede responder con números** — y en dos ocasiones la respuesta sale al revés de lo que uno supondría.

## Resultados consolidados (medidos)

### Parte 1 — TorToise-TTS sobre Tesla T4

| Preset | Cómputo | Audio | RTF | Autoregresivo | CLVP+CVVP | Difusión |
|---|---:|---:|---:|---:|---:|---:|
| `ultra_fast` | 50,1 s | 3,81 s | 13,2× | 36,4 s (**73 %**) | 1,5 s (3 %) | 1,7 s (3 %) |
| `fast` | 203,2 s | 4,22 s | 48,1× | 178,2 s (**88 %**) | 9,6 s (5 %) | 9,5 s (5 %) |

**El costo lo domina el decodificador autoregresivo, no el modelo de difusión.** Es contraintuitivo en un sistema que combina ambos, y tiene explicación estructural: el difusor opera sobre espectrogramas MEL —~256× comprimidos respecto de la onda— y paraleliza sobre toda la longitud; el autoregresivo decodifica token a token.

De ahí se sigue una corrección práctica a la tabla de presets del propio repositorio: `standard` y `high_quality` comparten `num_autoregressive_samples=256` y difieren solo en pasos de difusión, o sea en la parte barata. **`high_quality` cuesta un 4,6 % más que `standard`**, no el doble. Es exactamente por eso que Betker escribe que no vale la pena.

### Los 18 segundos que el modelo realmente ve

El notebook descarga 90 s de voz de referencia y corta tres clips de 10 s. `api.py` los recorta otra vez:

$$\underbrace{132300 / 22050 = 6{,}0\ \text{s}}_{\text{rama autoregresiva, offset aleatorio}} \qquad \underbrace{102400 / 24000 = 4{,}27\ \text{s}}_{\text{rama de difusión, desde el inicio}}$$

$3 \times 6{,}0 = 18$ s y $3 \times 4{,}27 = 12{,}8$ s: **el 20 % de lo descargado**. Más audio de referencia no mejora nada. Y como el recorte usa `random.randint`, **el descriptor de voz cambia entre corridas** aunque los archivos sean idénticos — una fuente de variabilidad que el notebook no menciona.

### Parte 2 — Wan2.2-Animate sobre NVIDIA L4

| Configuración | Tokens | s/paso | Pesos | Pico VRAM | Total |
|---|---:|---:|---:|---:|---:|
| `t4-draft` · 480×480 · 33f · `q3_k_m` | 9.000 | 11,1 | 8,04 GiB | 10,87 GiB | 2,8 min |
| `t4-max` · 832×464 · 77f · `q4_k_m` · 2 segmentos | 31.668 | 52,0 | 10,71 GiB | 16,87 GiB | 15,0 min |

Con dos puntos se puede separar el escalamiento. Ajustando $T = aN + bN^2$:

$$a = 1{,}071\times10^{-3}\ \text{s/token} \qquad b = 1{,}802\times10^{-8}\ \text{s/token}^2$$

| | 9.000 tokens | 31.668 tokens |
|---|---:|---:|
| Término lineal (MLPs, proyecciones) | 9,64 s (**87 %**) | 33,92 s (**65 %**) |
| Término cuadrático (atención) | 1,46 s (13 %) | 18,07 s (35 %) |

El exponente efectivo es $N^{1{,}23}$, y el cruce donde la atención pasaría a dominar está en $N = a/b = \mathbf{59.400}$ **tokens** — fuera del rango en que se trabajó. El mensaje de error del runner advierte que *"la atención es cuadrática en el conteo de tokens"*: cierto para el **tiempo** y asintóticamente, pero la **memoria** de atención resultó lineal ($3{,}78\times10^{-5}$ contra $3{,}80\times10^{-5}$ GB/token en ambas configuraciones), porque el backend SDPA *memory-efficient* nunca materializa la matriz $N \times N$.

### La aritmética de latentes, verificada

$$832 \times 464 \times 77\ \text{frames} \;\longrightarrow\; \texttt{latents [21, 58, 104]} = 31.668\ \text{tokens}$$

| Paso | Cuenta |
|---|---|
| Compresión espacial del VAE (8×) | $464/8 = 58$, $832/8 = 104$ |
| Compresión temporal del VAE (4×) | $(77-1)/4 + 1 = 20$ |
| **+1 latente de referencia** | **21** |
| Voxels | $21 \times 58 \times 104 = 126.672$ |
| Patchify 2×2 del DiT | $126.672 / 4 = \mathbf{31.668}$ |

Ese `+1` no es un ajuste: es el latente de referencia que describe el §3.2 del paper, y §4.3 lo confirma — *"cada segmento consiste en 78 frames; un frame se reserva estáticamente para la imagen del personaje"*. También explica la restricción $4n+1$ del conteo de frames.

### Lo que puso el prior

![El esqueleto renderizado por DWPose: la señal de control que recibe el difusor](/laboratorios/lab-44/esqueleto-dwpose.jpg)

La imagen de referencia era un recorte de cabeza y hombros, **sin brazos**, y el video de control tampoco los mostraba. El resultado tiene brazos cruzados y un encuadre más abierto que la referencia.

La explicación inmediata sería "lo puso el prior". Pero el esqueleto lo desmiente: **DWPose sí dibujó brazos y manos**, extrapolados hasta el borde inferior del cuadro con keypoints de baja confianza que el renderizador trazó igual. Y como el difusor **no ve keypoints sino esta imagen**, esos miembros inventados por el estimador de pose se convirtieron en señal de control legítima.

Ahí es donde se concentran los artefactos. No es que el modelo alucinara sin evidencia: es que la evidencia venía alucinada de dos etapas antes — el mismo patrón de propagación que documenta el [Lab 42](/laboratorios/lab-42/02-anatomia-de-un-id-switch), donde un error de detección se manifiesta como error de asignación.

### El presupuesto que rechaza lo que sí cabe

| | Chequeo de `run.py` | Etapa `denoise` |
|---|---|---|
| Presupuesto | **14,0 GiB fijos** | **21,03 GiB** (la tarjeta real) |
| Veredicto para 15,26 GiB | `DOES NOT FIT` | `FITS` |

El orquestador compara contra una constante calibrada para una T4 de 16 GB; la etapa que hace el trabajo lee `torch.cuda.get_device_properties`. El `--force` no fuerza nada riesgoso: corrige un supuesto obsoleto. Pico predicho 16,40 GiB, real **16,87 GiB**.

Y un costo de la cuantización que rara vez se menciona: el LoRA **no puede fusionarse** en pesos GGUF —`merge_and_unload` requiere pesos densos—, así que queda residente (+0,75 GiB) y se aplica por hooks sobre 488 módulos en cada paso.

## Bloques del lab

{{< cards >}}
  {{< card link="01-la-arquitectura-de-tortoise" title="La arquitectura de TorToise" subtitle="Cinco redes y 4,3 GB de pesos: el GPT-2 que resuelve el alineamiento, el DDPM que resuelve la calidad, CLVP puntuando tokens en vez de audio, el TorToise Trick, y el detector de deep fakes que viene incluido con el generador" icon="cube-transparent" >}}
  {{< card link="02-donde-esta-el-costo" title="Dónde está el costo" subtitle="El 88 % que se lleva el autoregresivo, el cond_free medido en 2,14× contra la predicción de 2×, el re-ranking que cuesta 92 ms por candidato, y la tabla de presets corregida: high_quality cuesta 4,6 % más que standard, no el doble" icon="chart-bar" >}}
  {{< card link="03-los-18-segundos" title="Los 18 segundos de los 90" subtitle="cond_length=132300 y el rand_start que cambia el descriptor entre corridas, el código que contradice al paper con h[:, :, 0] en vez de un promedio, y la prosodia medida: 1,65 palabras por segundo" icon="scissors" >}}
  {{< card link="04-wan-animate-por-dentro" title="Wan-Animate por dentro" subtitle="El esqueleto explícito para el cuerpo y las features implícitas para el rostro, y por qué el óptimo cae en lados opuestos. La aritmética de latentes verificada, el +1 de referencia, y las cinco etapas en procesos separados" icon="film" >}}
  {{< card link="05-el-presupuesto-de-memoria" title="El presupuesto de memoria" subtitle="GGUF comprimiendo 8,4×, el chequeo que rechaza contra 14 GB fijos lo que la etapa acepta contra 21, el LoRA que no se puede fusionar, y el escalamiento N^1,23 con el cruce de la atención en 59.400 tokens" icon="scale" >}}
  {{< card link="06-el-playbook-de-alineacion" title="El playbook de alineación" subtitle="Grabar en playback sobre el audio ya generado, calcular los frames desde la duración medida y no estimada, y por qué alejarse para mostrar las manos degrada justo la señal que produce el lip sync" icon="clock" >}}
  {{< card link="07-los-defectos-de-los-notebooks" title="Los defectos de los notebooks" subtitle="Nueve defectos: el ffmpeg que escribe sobre su propia entrada, el archivo de entrada que ningún notebook crea, el -shortest que no corta, el mvhd que miente, y las rutas relativas que hacen que un reinicio re-descargue 4,3 GB" icon="exclamation" >}}
{{< /cards >}}

## Clase y fundamentos

{{< cards >}}
  {{< card link="/clases/clase-44" title="Clase 44 - Aplicaciones de Audio y Video (cierre)" subtitle="El marco: siete aplicaciones donde la información no está en la entrada, y el índice del diplomado completo" icon="academic-cap" >}}
  {{< card link="/clases/clase-44/profundizacion" title="Profundización de la clase" subtitle="El jacobiano de FOMM medido, el informed guess cuantificado en 3855 a 1, y la asimetría entre generar y detectar" icon="beaker" >}}
  {{< card link="/fundamentos/sintesis-de-medios" title="Síntesis de Medios" subtitle="La taxonomía de cinco técnicas, el estado de la detección y por qué generaliza mal entre generadores" icon="book-open" >}}
  {{< card link="/fundamentos/modelos-de-difusion" title="Modelos de Difusión" subtitle="DDPM, DDIM y classifier-free guidance: el mecanismo que aparece en los dos modelos del laboratorio" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-voz" title="Reconocimiento de Voz" subtitle="La factorización identidad/contenido, aquí usada para generar en vez de analizar" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de Audio" subtitle="MEL, Nyquist y las tasas de muestreo: por qué 22.050 Hz en la entrada y 24.000 en la salida" icon="book-open" >}}
{{< /cards >}}

## Papers que aparecen en el laboratorio

{{< cards >}}
  {{< card link="/papers/tortoise-tts-betker-2023" title="TorToise-TTS (2023)" subtitle="Betker — DALL·E aplicado a voz: autoregresivo para el alineamiento, DDPM para la calidad, CLVP para el re-ranking. 49.000 horas y 8 RTX-3090 durante un año" icon="document-text" >}}
  {{< card link="/papers/wan-animate-2025" title="Wan-Animate (2025)" subtitle="Tongyi Lab — DiT de 14B: esqueleto explícito para el cuerpo, features implícitas para el rostro, y CFG deshabilitada por defecto porque la trae destilada" icon="document-text" >}}
  {{< card link="/papers/sv2tts-jia-2018" title="SV2TTS (2018)" subtitle="Jia et al. — la clonación de voz que la clase enseña y el laboratorio reemplazó" icon="document-text" >}}
  {{< card link="/papers/fomm-siarohin-2019" title="First Order Motion Model (2019)" subtitle="Siarohin et al. — keypoints sin supervisión más jacobianos. El método que Wan-Animate invierte al usar un detector supervisado" icon="document-text" >}}
  {{< card link="/papers/vitpose-xu-2022" title="ViTPose (2022)" subtitle="Xu et al. — el estimador de pose que el paper especifica y que la implementación sustituye por DWPose sin cambiar lo que el modelo recibe" icon="document-text" >}}
  {{< card link="/papers/classifier-free-guidance-ho-2022" title="Classifier-Free Guidance (2022)" subtitle="Ho y Salimans — el mecanismo que TorToise paga en inferencia y Wan destila en los pesos" icon="document-text" >}}
  {{< card link="/papers/faceforensics-rossler-2019" title="FaceForensics++ (2019)" subtitle="Rössler et al. — el contrapeso: detectar generaliza mal, y TorToise publica su propio detector junto al generador" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Lab 43 - E2E-AVSR](/laboratorios/lab-43) (audio y video para *analizar*, aquí para *generar*) · [Lab 41 - Speaker Recognition](/laboratorios/lab-41) (la misma factorización identidad/contenido, y otro checkpoint que desmiente a su paper) · [Lab 29 - Stable Diffusion](/laboratorios/lab-29) (donde apareció por primera vez el `guidance_scale`) · [Lab 42 - Tracking](/laboratorios/lab-42) (la propagación de un error entre etapas) · Dominios [Audio](/dominios/audio) y [Video](/dominios/video).
