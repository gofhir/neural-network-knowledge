---
title: "Lab 38 - Action Recognition con I3D"
weight: 380
sidebar:
  open: true
---

**Profesora:** Bianca Del Solar Medrano
**Módulo:** Video — modelos pre-entrenados
**Notebook origen:** `clase_38/material/Laboratorio/Lab_38_Action_Recognition_I3D_Final.ipynb`
**Notebook ejecutado:** [lab38.ipynb](/notebooks/lab38.ipynb) · [HTML](/notebooks-html/lab38.html)

## Encuadre

La contraparte práctica de la [clase 38](/clases/clase-38). El lab es corto y no entrena nada: carga el módulo **I3D-RGB pre-entrenado en Kinetics-400** desde TensorFlow Hub y lo usa para clasificar tres videos. En su forma original son 51 celdas de inferencia y tres preguntas conceptuales.

Pero la actividad **falla**: el video de rápel del enunciado se clasifica como `rock climbing` (87,66 %) en lugar de `abseiling` (11,23 %, rank #2). Diagnosticar ese fallo con cinco experimentos controlados convirtió un tutorial de copiar-y-pegar en la parte más informativa del práctico, y descubrió un **desajuste de preproceso en el tutorial oficial de TensorFlow Hub**: el notebook normaliza los píxeles a $[0, 1]$, pero el repositorio de DeepMind especifica $[-1, 1]$. Corregir esa línea invierte la predicción.

El lab también deja algo que ningún ajuste numérico arregla: sobre `ApplyEyeMakeup`, una clase que **no existe** entre las 400 de Kinetics, el modelo responde `filling eyebrows` con 98,13 % de confianza. Es el argumento —medido— de por qué *Quo Vadis* reemplaza la capa final antes de reportar sus 98,0 % en UCF101.

![Los tres videos evaluados y su resultado](/laboratorios/lab-38/tres-regimenes.jpg)

## Resultados consolidados (medidos en el notebook)

### Los tres regímenes de un modelo pre-entrenado

| Video | ¿La clase está en Kinetics-400? | Top-1 con $[0,1]$ | Top-1 con $[-1,1]$ | Entropía |
|---|---|---|---|---|
| `archery` (HuggingFace) | **sí** (índice 5) | `archery` 99,66 % ✅ | `archery` **99,97 %** ✅ | 0,060 |
| `abseiling_k400` (Kinetics) | **sí** (índice 0) | `rock climbing` 87,66 % ❌ | `abseiling` **62–93 %** ✅ | 0,415 |
| `ApplyEyeMakeup` (UCF101) | **no** | `filling eyebrows` 98,23 % | `filling eyebrows` 79,50 % | 0,104 |

Entropía máxima posible: $\ln(400) = 5{,}991$ nats.

### El diagnóstico: cinco hipótesis, cinco respuestas

| | Experimento | Resultado | Veredicto |
|---|---|---|---|
| **H1** | Rango de entrada $[0,1]$ vs $[-1,1]$ | 10,86 % → **62,31 %**; el top-1 cambia | **Causa raíz** |
| **H2** | `crop256 + resize224` vs `crop224` directo | +2,9 pts | Contribuyente menor |
| **H3** | Crop izquierdo / centro / derecho | 9,3 % / 65,2 % / **75,3 %**; $H = 2{,}75$ en el izquierdo | El sujeto no está centrado |
| **H4** | Ventana temporal | primeros 64 frames: **92,9 %**; últimos 100: 39,4 % (falla) | Gradiente temporal fuerte |
| **H5** | Video invertido en el tiempo | 65,2 % → 67,2 % (sin cambio) | I3D **no** usa la dirección del tiempo |

### La configuración importa más que el modelo

| Configuración | p(`abseiling`) |
|---|---|
| Baseline del tutorial ($[0,1]$, crop del notebook, 250 frames) | **10,86 %** ❌ rank #2 |
| $[-1,1]$, mismo crop, mismos frames | 62,31 % ✅ |
| $[-1,1]$ + crop 224 directo | 65,19 % ✅ |
| $[-1,1]$ + crop derecho | 75,28 % ✅ |
| $[-1,1]$ + 3-crop promediado + primeros 64 | 75,46 % ✅ |
| **$[-1,1]$ + crop centro + primeros 64** | **92,85 %** ✅ |
| *Referencia externa: demo oficial de GluonCV* | *99,1 %* |

**El mismo modelo, los mismos pesos, el mismo video: de 10,86 % a 92,85 % sólo cambiando el preproceso.**

## Las lecciones del lab

1. **Un error de preproceso no rompe el modelo — lo degrada en la frontera de decisión.** El rango $[0,1]$ no afectó a `archery` (99,66 % correcto) pero invirtió la predicción en `abseiling`. Por eso este tipo de bug puede vivir años en un tutorial oficial: el 90 % de los casos sigue funcionando.
2. **El softmax obliga a responder.** Con una clase ausente del vocabulario, el modelo no puede decir "no sé": reparte toda la masa al vecino semántico más cercano y lo hace con 98 % de confianza. Un error **estructural**, no numérico, y sólo se arregla con fine-tuning.
3. **La entropía detecta ambigüedad, no ausencia de vocabulario.** Disparó en `abseiling` (0,415) y en el crop sin sujeto (2,753), pero fue ciega en `ApplyEyeMakeup` (0,104 con la clase correcta inexistente).
4. **Un ensemble sólo ayuda si sus miembros son competentes.** Promediar los 3 crops horizontales *empeoró* el resultado 17,4 puntos, porque el crop izquierdo vota activamente por la clase equivocada. El protocolo estándar de multi-crop supone un sujeto centrado.
5. **Promediar sobre el clip completo diluye la evidencia.** I3D promedia logits sobre el tiempo *dentro* de la red: la información de los primeros 2,6 s (92,9 %) se mezcla con la de los últimos 4 s (39,4 %) y no hay forma de recuperarla después. Es la limitación *trimmed* medida.
6. **I3D no usó la dirección del tiempo.** Invertir el video no cambió la predicción, pese a que un rápel invertido es visualmente una escalada. Discrimina por apariencia y movimiento de corto alcance — el mismo sesgo que el [lab 36](/laboratorios/lab-36) midió en una CNN 2D, ahora en una CNN 3D.

## Bloques del lab

{{< cards >}}
  {{< card link="01-el-pipeline-y-sus-fosiles" title="El pipeline y sus fósiles" subtitle="Los 6,93 GB que se descargan para leer 294 KB, la función identidad disfrazada de descargador, el crop que tira el 25 % del ancho y la línea que resultó ser la causa raíz" icon="code" >}}
  {{< card link="02-el-vocabulario-manda" title="El vocabulario manda" subtitle="Las 400 etiquetas, hub.load, logits contra probabilidades, y el contraste entre una clase que existe y otra que no: 99,97 % de acierto frente a 98,13 % de falsa confianza" icon="document-text" >}}
  {{< card link="03-el-bug-del-preproceso" title="El bug del preproceso" subtitle="La actividad falla. H1 y H2: por qué $[0,1]$ invierte la predicción, el mecanismo de las BatchNorm heredadas y la validación contra GluonCV" icon="beaker" >}}
  {{< card link="04-donde-esta-la-evidencia" title="Dónde está la evidencia" subtitle="H3 y H4: el sujeto que se desplaza a la derecha y se aleja de la cámara, la ventana temporal que decide el resultado, y el 3-crop que empeora las cosas" icon="chart-bar" >}}
  {{< card link="05-invertir-el-tiempo" title="Invertir el tiempo" subtitle="H5, el resultado negativo: I3D no distingue subir de bajar. Qué significa para el sesgo de apariencia de Kinetics y la síntesis del laboratorio" icon="trending-down" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/inflado-de-convoluciones" title="Inflado de convoluciones" subtitle="El punto fijo del video aburrido, qué se infla y qué no — incluidas las BatchNorm que causan el bug de este lab" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-acciones" title="Reconocimiento de acciones" subtitle="Trimmed contra untrimmed, datasets y la evolución de los enfoques" icon="book-open" >}}
  {{< card link="/fundamentos/analisis-de-video" title="Análisis de video" subtitle="Video, movimiento, stream contra sequence" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer learning" subtitle="Feature extraction, fine-tuning y por qué la capa final hay que reemplazarla" icon="book-open" >}}
{{< /cards >}}

## Papers de este laboratorio

{{< cards >}}
  {{< card link="/papers/i3d-carreira-2017" title="I3D (2017)" subtitle="Carreira y Zisserman — el modelo del lab: el inflado, Kinetics y el 98,0 % en UCF101 que exige fine-tuning" icon="document-text" >}}
  {{< card link="/papers/kinetics-kay-2017" title="Kinetics (2017)" subtitle="Kay et al. — las 400 etiquetas que definen qué puede y qué no puede responder el modelo" icon="document-text" >}}
  {{< card link="/papers/something-something-goyal-2017" title="Something-Something (2017)" subtitle="Goyal et al. — el dataset creado precisamente porque Kinetics tiene sesgo de apariencia; el contrapunto a H5" icon="document-text" >}}
  {{< card link="/papers/c3d-tran-2015" title="C3D (2015)" subtitle="Tran et al. — los 78M de parámetros entrenados desde cero contra los 12M de I3D" icon="document-text" >}}
  {{< card link="/papers/two-stream-simonyan-2014" title="Two-Stream (2014)" subtitle="Simonyan y Zisserman — el flujo óptico precomputado fuera de la red" icon="document-text" >}}
  {{< card link="/papers/lrcn-donahue-2015" title="LRCN (2015)" subtitle="Donahue et al. — CNN 2D + LSTM: el movimiento local ya perdido antes del recurrente" icon="document-text" >}}
  {{< card link="/papers/large-scale-video-karpathy-2014" title="Sports-1M (2014)" subtitle="Karpathy et al. — CNN 2D + agrupación temporal, y el hallazgo de que un frame casi alcanzaba" icon="document-text" >}}
  {{< card link="/papers/slowfast-feichtenhofer-2019" title="SlowFast (2019)" subtitle="Feichtenhofer et al. — dos vías por framerate; el contexto del hallazgo sobre el stride temporal" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Clase 38 - Teoría](/clases/clase-38/teoria) · [Clase 38 - Profundización](/clases/clase-38/profundizacion) · [Clase 38 - Práctica](/clases/clase-38/practica) · [Lab 36 - Análisis de Video](/laboratorios/lab-36) (el mismo sesgo temporal, en una CNN 2D) · [Clase 36](/clases/clase-36) · Dominio [Video](/dominios/video).
