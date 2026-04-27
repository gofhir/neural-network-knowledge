---
title: "Notas — MIT 6.S191 (2026) L3: Deep Computer Vision"
weight: 10
math: true
sidebar:
  open: true
---

> Recorrido temático de las 70 diapositivas del lecture **MIT 6.S191 (2026) Lecture 3 — Deep Computer Vision** (Alexander Amini, 6 de enero de 2026). Se omite la slide 70 (anuncios del curso, sin contenido técnico).
>
> [Video en YouTube](https://www.youtube.com/watch?v=pqIcoskUuWs) — [Slides oficiales (PDF)](/videos/mit-6s191-l3-2026/slides.pdf) — [PNGs individuales](/videos/mit-6s191-l3-2026/slides/)

---

## 1. Visión computacional: definición e impacto

La clase abre con una definición operativa de visión computacional atribuida a David Marr: *"to know what is where by looking"* — saber **qué** está **dónde** mirando *(slide 2)*. La extensión moderna añade dimensiones temporales y predictivas: *"discover from images what is present in the world, where things are, what actions are taking place, to predict and anticipate events"* *(slide 3)*. Visión, en este registro, no es solo reconocimiento estático: es percepción más predicción.

El recorrido motivacional repasa el impacto del campo *(slides 4-8)*:

- **Robótica:** percepción para manipulación y locomoción (Boston Dynamics).
- **Accesibilidad:** asistencia visual basada en cámaras (Google Project Guideline para corredores no videntes).
- **Biología y medicina:** detección de cáncer de mama, COVID-19 en radiografías, melanoma en piel; cobertura ya con resultados publicados en *Nature* (Esteva 2017, McKinney 2020, Wang 2020).
- **Conducción autónoma:** percepción + control end-to-end.
- **Reconocimiento facial:** detección de landmarks y reconstrucción 3D.

El objetivo del lecture es construir, a partir de primeros principios, las **redes convolucionales (CNNs)** que hicieron posible este salto de utilidad práctica.

---

## 2. Qué "ven" los computadores: imágenes como números

Para una máquina una imagen no es una percepción gestáltica sino una matriz numérica *(slides 10-12)*. Una imagen en escala de grises es una matriz de enteros en $[0, 255]$; una imagen RGB de tamaño $H \times W$ es un tensor $H \times W \times 3$. Para una imagen 1080×1080 RGB son $\approx 3.5\text{M}$ valores que codifican intensidad por canal y por píxel.

![Slide 12 — Images are Numbers: lo que el humano ve a la izquierda, lo que el computador "ve" a la derecha (matriz de intensidades).](/videos/mit-6s191-l3-2026/slides/slide-12.png)

La consecuencia inmediata: cualquier tarea de visión es, formalmente, una función $f: \mathbb{R}^{H \times W \times C} \to \mathcal{Y}$ que mapea ese tensor a un espacio de salida $\mathcal{Y}$.

Las dos familias canónicas de tareas son *(slide 13)*:

- **Regresión:** $\mathcal{Y} \subseteq \mathbb{R}^k$ — la salida es continua (e.g., posición de un objeto, ángulo de dirección de un auto).
- **Clasificación:** $\mathcal{Y}$ es un conjunto discreto de etiquetas; la red emite una distribución $p(y \mid x)$ sobre clases.

---

## 3. Por qué la extracción manual de features falla

Antes del deep learning, el pipeline clásico de visión era *(slides 15-17)*:

1. Aplicar **conocimiento de dominio** para decidir qué características importan (e.g., "para detectar una cara busca ojos, nariz y boca" *(slide 14)*).
2. **Definir** detectores manuales de esas features (filtros Haar, HOG, SIFT, descriptores de Gabor).
3. **Detectar** features y clasificar con un modelo lineal o un kernel SVM por encima.

Aunque elegante, este pipeline se quiebra ante la variabilidad real del mundo *(slide 16)*:

![Slide 16 — Fuentes de variación que invalidan features hechos a mano: viewpoint, escala, deformación, oclusión, background clutter, variación intra-clase.](/videos/mit-6s191-l3-2026/slides/slide-16.png)

| Fuente de variación | Ejemplo |
| --- | --- |
| **Viewpoint variation** | La misma cara desde tres ángulos genera tres patrones de píxeles muy distintos. |
| **Scale variation** | El mismo objeto a 2 m vs 20 m ocupa áreas radicalmente distintas. |
| **Deformation** | Cuerpos articulados (humanos, animales) cambian de pose. |
| **Occlusion** | Parte del objeto está oculta. |
| **Background clutter** | Texturas distractoras en el fondo (un guepardo entre hojas). |
| **Intra-class variation** | Una "silla" puede ser un banco, un sillón, una hamaca; comparten función pero no apariencia. |

Codificar manualmente todas estas invarianzas es intratable. La pregunta operativa que cierra esta sección: *"¿podemos **aprender** features directamente desde los datos en vez de diseñarlos a mano?"*.

---

## 4. Aprender una jerarquía de features

La respuesta es la idea central del deep learning aplicado a visión: las redes profundas aprenden una **jerarquía** de representaciones *(slide 18)*.

![Slide 18 — Jerarquía de features aprendidas: bordes y manchas (low), partes (mid), estructura facial (high). Imagen original: Lee+ ICML 2009.](/videos/mit-6s191-l3-2026/slides/slide-18.png)

- **Low-level features:** bordes, esquinas, manchas oscuras, gradientes locales.
- **Mid-level features:** combinaciones de bordes que forman partes — ojos, narices, ruedas.
- **High-level features:** composiciones de partes que forman objetos completos — caras, autos, casas.

Esta jerarquía no se programa: emerge naturalmente al apilar capas convolucionales con no-linealidades (Lee et al., ICML 2009). El resto del lecture se dedica a construir la maquinaria que la hace posible.

---

## 5. Por qué las redes fully-connected fallan en imágenes

El primer instinto sería aplicar la red densa de la clase 1 directamente a píxeles *(slides 20-22)*. Imaginemos una imagen 1080×1080×3 aplastada (*flattened*) a un vector de $p \approx 3.5\text{M}$ entradas. Conectar esa entrada a una capa oculta de, digamos, 1000 neuronas requiere $3.5 \times 10^9$ pesos solo en la primera capa. Dos problemas críticos surgen:

1. **No hay información espacial.** Aplastar destruye la estructura 2D: el píxel $(i, j)$ y el píxel $(i, j+1)$ pasan a ser dos entradas no-relacionadas. La red tiene que re-aprender desde cero la noción de vecindad.
2. **Demasiados parámetros.** $3.5 \times 10^9$ pesos por capa son inviables: el modelo sobre-ajusta de inmediato y exige cantidades obscenas de datos.

La pregunta de cierre del bloque *(slide 22)*: *"¿cómo usamos la **estructura espacial** del input para informar la arquitectura de la red?"*

---

## 6. Estructura espacial: del patch a la convolución

La idea fundadora *(slides 23-24)*: en vez de conectar cada píxel con cada neurona, conectar **parches** del input con neuronas en la capa siguiente, y deslizar (*sliding window*) el mismo parche a lo largo de la imagen.

Esto introduce dos compromisos cruciales:

1. **Localidad:** una neurona solo "ve" un parche pequeño del input, no la imagen completa. Justificación: las features de bajo nivel (bordes, esquinas) son locales por naturaleza.
2. **Compartición de pesos (*weight sharing*):** el mismo conjunto de pesos se aplica a todas las posiciones del input. Justificación: si un detector de borde es útil en la esquina superior izquierda, también lo es en la esquina inferior derecha; no necesitamos parámetros distintos por posición.

Ambos compromisos definen la operación de **convolución** *(slide 25)*:

![Slide 25 — Feature Extraction with Convolution: filtro 4×4 con 16 pesos, aplicado al mismo parche de entrada con stride 2.](/videos/mit-6s191-l3-2026/slides/slide-25.png)

Operativamente:

1. Se define un **filtro** (también llamado *kernel*) de tamaño $k \times k$ con $k^2$ pesos aprendibles.
2. Se aplica el filtro al parche $k \times k$ del input mediante **multiplicación elemento-a-elemento** seguida de **suma**.
3. Se desliza el filtro $s$ píxeles (*stride*) y se repite, generando una matriz de salida llamada **feature map**.
4. Se usan **múltiples filtros** en paralelo para detectar features distintas (uno detecta bordes verticales, otro horizontales, otro texturas, etc.).
5. Los pesos del filtro se **comparten espacialmente** entre todas las posiciones donde se aplica.

Estas tres propiedades — localidad, weight sharing, múltiples filtros — son la inducción de bias arquitectural que hace que las CNNs funcionen sobre imágenes mientras las MLPs fracasan.

En la siguiente sección formalizaremos la operación, definiremos stride, padding, profundidad de salida y campo receptivo, y veremos cómo se apilan capas convolucionales para producir la jerarquía de features de la sección 4.
