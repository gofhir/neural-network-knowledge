---
title: "Data Augmentation de Audio"
weight: 125
math: true
---

Cuando faltan datos, se **fabrican**: se crean variantes que suenan distinto pero **conservan la etiqueta**. La [data augmentation](/fundamentos/data-augmentation) es una técnica transversal del deep learning; este fundamento, que acompaña a la [Clase 37](/clases/clase-37), la especializa para audio —donde las transformaciones tienen semántica física (tono, velocidad, ruido) y donde la pregunta clave, antes de escribir código, es si la transformación es una **invariancia real del problema**.

---

## 1. El criterio: ¿es una invariancia real?

La idea de fondo: *un rock 10% más rápido sigue siendo rock*. Con variantes, el modelo debe aprender **qué hace que algo sea rock** en vez de memorizar cada grabación —y la brecha train-test se cierra.

{{< concept-alert type="clave" >}}
Antes de escribir código, la pregunta que ordena todo: **¿la transformación es una invariancia real del problema?** Un desplazamiento de ±4 semitonos vale para **clasificar género**, pero **destruye la tarea** si es detectar tonalidad o identificar al hablante. Una llamada más rápida sigue siendo el mismo reclamo (sí para análisis de sentimiento), pero **no** es la misma voz para la biometría del banco. La augmentation correcta depende de la etiqueta que quieres preservar.
{{< /concept-alert >}}

Otras dos reglas de oro: la augmentation va **solo en train** (el test se deja quieto para que la métrica sea comparable) y **distinta en cada época** (esa es la gracia: el modelo nunca ve dos veces el mismo ejemplo).

---

## 2. Sumar ruido: la perilla es el SNR

La augmentation más directa es agregar **otra señal de fondo**; la etiqueta no cambia. Pero la cantidad de ruido no se fija por amplitud, sino por la **relación señal-ruido** (SNR), un cociente de potencias en decibeles:

$$
\text{SNR} = 10 \log_{10}\!\left(\frac{P_{\text{señal}}}{P_{\text{ruido}}}\right) \; [\text{dB}], \qquad P = \text{promedio de amplitud}^2.
$$

Cada 10 dB es un factor 10 en potencia: $+20$ dB = la señal tiene 100× la potencia del ruido; $0$ dB = igual potencia; $-10$ dB = el ruido es 10× más potente. La imagen mental es conversar en una fiesta: la voz es la señal, el murmullo el ruido; en una pieza callada el SNR es alto, al lado del parlante es bajo o negativo.

En términos perceptuales: **20 dB** el ruido apenas se percibe, **10 dB** fondo claramente audible, **3 dB** casi tan fuerte como la señal. En entrenamiento, el SNR se **sortea en un rango** (p. ej., 10 a 20 dB) para que cada época suene distinto. `torchaudio.functional.add_noise(señal, ruido, snr)` escala el ruido al SNR pedido.

{{< concept-alert type="recordar" >}}
El **ruido blanco** ya regulariza, pero es mejor usar **grabaciones del ruido real** donde va a operar el modelo —que el ruido de train se parezca al de producción. Y en datos reales, el SNR se **estima** (silencios vía VAD, WADA-SNR, o SQUIM de `torchaudio.pipelines`).
{{< /concept-alert >}}

---

## 3. SpecAugment: tapar en vez de sumar

En vez de sumar ruido, se puede **tapar**. **[SpecAugment](/papers/specaugment-park-2019)** (Park et al., 2019) hace **cero** componentes del espectrograma al azar: se sortean **bandas de frecuencia** (filas) y **tramos de tiempo** (columnas) y se anulan. Si conoces **Cutout** en imágenes, es exactamente eso.

Por qué funciona: es **dropout con estructura** —se apagan regiones **contiguas**, no unidades sueltas. Si el modelo dependía de una sola banda, taparla entera lo obliga a **buscar evidencia en otra parte**. Y es casi gratis: opera sobre el espectrograma que **ya está en la GPU**, dos líneas en el loop de entrenamiento. Nació para reconocimiento de voz y hoy se usa en casi todo.

---

## 4. Cambiar tono o duración, por separado

Reproducir más rápido cambia **tono y duración a la vez**. Dos herramientas mueven **uno solo**:

- **`time_stretch`** — más lento o más rápido, **mismo tono**. Un rock 10% más rápido sigue siendo rock.
- **`pitch_shift`** — más agudo o más grave, **misma duración**. ±2 semitonos no cambian el género. (Ojo en voz: los **formantes** se corren y suena a otra persona —inútil, o nocivo, para biometría.)

Separar tono y duración requiere un algoritmo no trivial (**phase vocoder**) que cuesta ~1 s por archivo. Por eso **no va en el dataloader**: se precalcula una vez, offline.

---

## 5. Dónde va cada augmentation: el costo decide

{{< concept-alert type="clave" >}}
El **costo computacional** decide el lugar de cada augmentation en el pipeline:
{{< /concept-alert >}}

| Costo | Operaciones | Dónde va |
|---|---|---|
| **Barato, en CPU** | recorte aleatorio, ganancia, ruido (SNR) | en la `collate_fn` |
| **Gratis, en GPU** | SpecAugment (el Cutout del espectrograma) | en el loop de entrenamiento |
| **Caro (~1 s/archivo)** | `pitch_shift`, `time_stretch` | offline, precalculado |

Las de ruido son casi gratis —por eso son la **primera línea**. Las que separan tono/duración son caras —por eso se precalculan.

---

## 6. Relevancia para salud

En audio clínico, donde los datasets son casi siempre pequeños, la augmentation es especialmente valiosa —pero el criterio de "invariancia real" se vuelve crítico. Agregar **ruido a distintos SNR** entrena robustez a las condiciones reales de grabación (un estetoscopio digital en una sala ruidosa, un micrófono de teléfono para telemedicina), y **SpecAugment** regulariza casi gratis. Pero hay que ser cuidadoso: un `pitch_shift` puede ser una invariancia válida para detectar *si hay* tos, pero **destruir** la señal si la tarea es detectar una patología vocal cuya firma está justamente en el tono o los formantes. La regla es la misma que en el resto del campo: preguntar primero qué etiqueta se debe preservar, y usar **ruido real del entorno de despliegue** siempre que se pueda.

---

## Referencias

- Park, D. et al. (2019). *SpecAugment: A Simple Data Augmentation Method for ASR*. Interspeech. — [análisis](/papers/specaugment-park-2019)
- Fundamentos relacionados: [Data augmentation](/fundamentos/data-augmentation) · [Representación de audio](/fundamentos/representacion-de-audio) · [Datasets de audio](/fundamentos/datasets-de-audio).
- Aplicación práctica: el [Laboratorio 37](/laboratorios/lab-37/02-data-augmentation) verifica el SNR contra su definición, oye la "phasiness" de copiar las fases en vez de regenerarlas, y documenta **una regla que falta en la lista habitual**: aplicar la augmentación al 100% de las muestras de train deja de ser augmentación y se vuelve un cambio de dominio — medido en **11,7 puntos** de diferencia entre test limpio y test con ruido.
- Dominio: [Audio / Voz](/dominios/audio).
