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

---

## 7. Caso de estudio: detectar una "X" deforme

Antes de formalizar, la clase usa un caso de estudio canónico (Rohrer, *How do CNNs work?*) para ganar intuición *(slides 27-29)*. La pregunta: dada una imagen binaria de una "X", ¿cómo clasificarla como X aunque esté **desplazada, escalada, rotada o deformada**? *(slide 27)*.

Respuesta: en vez de buscar la X completa, buscar **partes locales** que la caractericen *(slide 28)*: una esquina superior-izquierda con diagonal hacia adentro, una esquina superior-derecha con diagonal hacia adentro, y un cruce diagonal central. Cada parte vive en un parche de $3 \times 3$ píxeles.

![Slide 29 — Tres filtros 3×3 con pesos {+1, -1} que detectan las tres partes locales que caracterizan una X.](/videos/mit-6s191-l3-2026/slides/slide-29.png)

Cada filtro es una matriz de $3 \times 3$ con pesos en $\{+1, -1\}$. Aplicar el filtro a un parche significa multiplicar elemento-a-elemento y sumar: si el parche del input coincide perfectamente con el patrón del filtro, la suma se maximiza. Para un parche $3 \times 3$ que coincide exactamente con el filtro de pesos $\pm 1$, todos los productos son $+1$ y la suma vale $9$ *(slide 30)*.

Ese **escalar** es la respuesta de un detector local. Aplicar el filtro a todas las posiciones del input produce una matriz de respuestas: el **feature map**.

---

## 8. La operación de convolución, paso a paso

Las slides 31-40 desarrollan la convolución operativamente sobre un input $5 \times 5$ con un filtro $3 \times 3$ *(slides 31-40)*. Para cada posición $(p, q)$ de salida:

$$
y_{p,q} = \sum_{i=1}^{k} \sum_{j=1}^{k} w_{ij} \cdot x_{p+i-1,\, q+j-1}
$$

donde $k$ es el tamaño del filtro, $w_{ij}$ son los pesos del filtro, y $x_{\cdot,\cdot}$ es el input. Repetir para cada posición $(p, q)$ produce el feature map.

Tres ideas operativas se consolidan tras la animación slide-by-slide *(slide 42)*:

1. **Aplicar un conjunto de pesos** (un filtro) extrae *features locales* — una respuesta por parche, no una respuesta por imagen.
2. **Múltiples filtros** producen *múltiples feature maps*, cada uno especializado en una feature diferente (un detector de borde vertical, otro horizontal, otro de textura).
3. **Compartir parámetros espacialmente** (*spatial weight sharing*) significa que el mismo filtro se aplica a todas las posiciones — invarianza a traslación incorporada en la arquitectura.

La operación de convolución no es un invento del deep learning: filtros como Sobel para bordes, Gaussiano para suavizado, o Laplaciano para detección de bordes fuertes existen desde décadas en procesamiento de imágenes clásico *(slide 41)*.

![Slide 41 — Feature maps producidos por filtros conocidos sobre la imagen de Lena: sharpen, edge detect, strong edge detect.](/videos/mit-6s191-l3-2026/slides/slide-41.png)

La novedad de las CNNs no es la convolución en sí: es que los pesos del filtro se **aprenden por gradiente** desde los datos, en vez de diseñarlos a mano. Una CNN profunda puede aprender filtros de bordes en la primera capa que se parecen a los de Sobel, pero también filtros más exóticos en capas profundas que ningún humano hubiera escrito.

---

## 9. Arquitectura CNN para clasificación

Una CNN para clasificación apila tres tipos de operaciones *(slide 44)*:

![Slide 44 — Pipeline canónico: input → convolución (feature maps) → pooling → fully-connected → softmax.](/videos/mit-6s191-l3-2026/slides/slide-44.png)

1. **Convolución:** aplica filtros aprendibles para producir feature maps.
2. **No-linealidad** (típicamente ReLU): se aplica después de cada convolución.
3. **Pooling:** downsampling de cada feature map.

Estas tres se repiten varias veces (formando el "backbone" convolucional), y al final se conectan a una o varias capas fully-connected que producen la distribución sobre clases.

En código:

```python
# Keras / TensorFlow
import tensorflow as tf
tf.keras.layers.Conv2D(filters=d, kernel_size=(h, w), strides=s, activation="relu")
tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)

# PyTorch
import torch.nn as nn
nn.Conv2d(in_channels=c_in, out_channels=d, kernel_size=(h, w), stride=s)
nn.ReLU()
nn.MaxPool2d(kernel_size=(2, 2), stride=2)
```

---

## 10. Capas convolucionales: matemática y conectividad local

Cada neurona en una capa convolucional tiene tres propiedades clave *(slides 45-46)*:

- **Toma sus inputs de un parche** del feature map anterior, no de toda la entrada.
- **Calcula una combinación lineal ponderada** $\sum_{i,j} w_{ij} \, x_{i+p, j+q} + b$.
- **Aplica una función de activación no-lineal** (ReLU).

Para una capa con filtros de tamaño $k \times k$, la salida en la posición $(p, q)$ del feature map es:

$$
y_{p,q} = g\!\left(\sum_{i=1}^{k} \sum_{j=1}^{k} w_{ij} \cdot x_{i+p,\, j+q} + b\right)
$$

donde $g$ es la no-linealidad. La **misma matriz de pesos** $W \in \mathbb{R}^{k \times k}$ y **el mismo bias** $b$ se reutilizan en todas las posiciones $(p, q)$ — esto es el weight sharing que define la convolución.

---

## 11. Volúmenes 3D: profundidad y campo receptivo

Hasta aquí trabajamos con un solo filtro y un input de un solo canal. La realidad es 3D: imágenes RGB tienen 3 canales, y cada capa convolucional aplica $d$ filtros distintos, produciendo $d$ feature maps apilados *(slide 47)*.

![Slide 47 — Volumen de salida de una capa Conv: dimensiones $h \times w \times d$. Profundidad $d$ = número de filtros. Stride = paso del filtro. Receptive field = región del input que afecta a un nodo dado.](/videos/mit-6s191-l3-2026/slides/slide-47.png)

Tres dimensiones definen un volumen de feature maps:

- **Height ($h$) y width ($w$):** dimensiones espaciales — qué tan "grande" es el feature map.
- **Depth ($d$):** número de filtros aplicados, equivalente al número de feature maps producidos. Un input RGB tiene depth 3; una capa con 64 filtros produce un volumen de salida con depth 64.

Tres hiperparámetros controlan el shape de salida:

- **Kernel size $k$:** tamaño del filtro (típicamente 3, 5, 7).
- **Stride $s$:** cuántos píxeles avanza el filtro entre aplicaciones (1 = denso; 2 = downsample por 2).
- **Padding $p$:** cuántos píxeles de "ceros" se agregan al borde para controlar el shape de salida.

Para un input $H \times W$ con kernel $k$, padding $p$, stride $s$, la salida tiene tamaño:

$$
H_{\text{out}} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1
$$

(análogo para $W_{\text{out}}$).

El **receptive field** (campo receptivo) de un nodo en una capa profunda es la región del input original que efectivamente afecta su valor. Apilando capas con kernel $3 \times 3$ y stride 1 sin padding, el receptive field crece linealmente: capa 1 cubre $3 \times 3$, capa 2 cubre $5 \times 5$, capa 3 cubre $7 \times 7$. Con stride 2 o convoluciones dilatadas el crecimiento puede ser exponencial.

Esta capacidad de un nodo profundo de "ver" una porción cada vez mayor del input es lo que permite componer features de bajo nivel en features de alto nivel.

---

## 12. No-linealidad: ReLU

Se aplica una activación no-lineal después de cada convolución *(slide 48)*. El estándar de facto es **ReLU** (Rectified Linear Unit):

$$
g(z) = \max(0, z)
$$

Operativamente: cada valor negativo del feature map se reemplaza por cero; los positivos pasan sin cambio. Esto introduce no-linealidad (sin la cual una pila de capas convolucionales colapsaría a una sola convolución lineal) y mantiene el gradiente fluyendo en la mitad positiva, evitando la saturación que sufren sigmoid/tanh.

---

## 13. Pooling

Las CNNs intercalan capas de **pooling** entre convoluciones para reducir la dimensión espacial y agregar invarianza local *(slide 49)*:

![Slide 49 — Max pooling con filtros 2×2 y stride 2: cada bloque 2×2 del feature map se reemplaza por su valor máximo.](/videos/mit-6s191-l3-2026/slides/slide-49.png)

Operación más común: **max pooling**, $2 \times 2$ con stride 2. Por cada bloque $2 \times 2$ del feature map se queda solo el máximo, reduciendo $h$ y $w$ a la mitad pero conservando $d$.

Beneficios:

1. **Reducción de dimensionalidad:** $h \times w$ de salida es $\tfrac{1}{4}$ del de entrada, lo que acelera capas posteriores.
2. **Invarianza espacial local:** pequeños desplazamientos del input no cambian la salida del pool (si la feature está en cualquier lugar del bloque $2 \times 2$, el máximo es el mismo).

Alternativas: **average pooling** (promedio en vez de max), **global average pooling** (un solo escalar por feature map, usado al final del backbone en muchas arquitecturas modernas en vez de FC).

---

## 14. Representación aprendida: jerarquía emergente

Cuando se entrena una CNN profunda end-to-end con clasificación de imágenes, los filtros aprendidos en cada capa revelan exactamente la jerarquía que motivamos en la sección 4 *(slide 50)*:

![Slide 50 — Visualización de filtros aprendidos en capas Conv 1, 2 y 3 de una CNN profunda (Lee+ ICML 2009): bordes en Conv 1, partes faciales en Conv 2, caras completas en Conv 3.](/videos/mit-6s191-l3-2026/slides/slide-50.png)

- **Conv layer 1:** detectores de bordes orientados, manchas oscuras, gradientes locales.
- **Conv layer 2:** combinaciones de bordes que detectan partes — ojos, narices, esquinas de caras.
- **Conv layer 3:** detectores de objetos completos — caras enteras.

Esta jerarquía no se programa: emerge cuando se entrena una CNN profunda con un loss de clasificación end-to-end. Es una de las observaciones empíricas más importantes en deep learning, y el hilo que conecta la motivación inicial (¿podemos aprender features?) con la maquinaria desarrollada (convolución + pooling + ReLU apilados).

En la siguiente sección veremos qué aplicaciones se construyen sobre esta arquitectura — clasificación, detección, segmentación, generación, conducción autónoma — y cómo se modifica el "head" de la red según la tarea.

---

## 15. Pipeline completo: feature learning + clasificación

El pipeline canónico de una CNN para clasificación se descompone explícitamente en dos mitades *(slide 51)*:

![Slide 51 — Backbone convolucional (feature learning) + cabecera de clasificación. La parte convolucional aprende representaciones; la parte densa decide la clase.](/videos/mit-6s191-l3-2026/slides/slide-51.png)

1. **Feature learning (mitad convolucional):** apila bloques `Conv → ReLU → Pool` que aprenden la jerarquía de representaciones discutida en la sección 14.
2. **Clasificación:** un *flatten* aplana el último volumen de feature maps a un vector, seguido de una o más capas densas y una softmax que produce $p(y \mid x)$ *(slide 52)*:

$$
\text{softmax}(y_i) = \frac{e^{y_i}}{\sum_j e^{y_j}}
$$

La softmax convierte logits arbitrarios en una distribución de probabilidad sobre las clases. Se entrena con cross-entropy negativa.

---

## 16. Implementación práctica (TensorFlow y PyTorch)

La clase incluye implementaciones equivalentes en los dos frameworks dominantes *(slides 53-54)*. La estructura es idéntica; cambia solo el API.

```python
# TensorFlow / Keras
import tensorflow as tf

def generate_model():
    return tf.keras.Sequential([
        # primera capa convolucional
        tf.keras.layers.Conv2D(32, filter_size=3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        # segunda capa convolucional
        tf.keras.layers.Conv2D(64, filter_size=3, activation="relu"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
        # cabecera fully-connected
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
```

```python
# PyTorch
import torch.nn as nn

def generate_model():
    return nn.Sequential(
        # primera capa convolucional
        nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # segunda capa convolucional
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        # cabecera fully-connected
        nn.Flatten(),
        nn.Linear(64*6*6, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10),
    )
```

Notar que en PyTorch la activación es una capa explícita, mientras que Keras la pasa como argumento. La dimensión de entrada de la capa `Linear` (`64*6*6`) depende del shape espacial del último feature map, que en PyTorch debe calcularse manualmente; Keras lo infiere.

---

## 17. Una arquitectura, muchas aplicaciones

La idea más potente del lecture: el **mismo backbone convolucional** sirve para una gran variedad de tareas; solo cambia la cabecera *(slides 55-56)*.

![Slide 56 — El backbone aprende features genéricas; según la tarea (clasificación, detección, segmentación, control probabilístico), se intercambia la cabecera.](/videos/mit-6s191-l3-2026/slides/slide-56.png)

| Tarea | Cabecera | Output |
| --- | --- | --- |
| **Clasificación** | FC + softmax | distribución sobre clases |
| **Detección** | FC + regresión + softmax | bounding boxes con clase y posición |
| **Segmentación** | upsampling (decoder convolucional) | máscara $H \times W$ con etiqueta por píxel |
| **Control probabilístico** | regresión a parámetros de una distribución (e.g., $\mu, \sigma$) | comando de control con incertidumbre |

Esta modularidad — backbone reutilizable + cabecera específica por tarea — es la base del **transfer learning** moderno: se preentrena el backbone en ImageNet, y se reemplaza/fine-tunea la cabecera para cada tarea downstream.

---

## 18. Clasificación médica: cribado de cáncer de mama

La clase aterriza la utilidad práctica con un caso publicado en *Nature* *(slide 57)*: McKinney et al. 2020, "International evaluation of an AI system for breast cancer screening".

- El sistema basado en CNN **superó a radiólogos expertos** en sensibilidad/especificidad para cáncer de mama detectado en mamografías a 1 año y a 2 años, en cohortes del Reino Unido y Estados Unidos.
- En la inspección retrospectiva, la red detectó casos que los radiólogos humanos habían marcado como negativos (falsos negativos humanos detectados como verdaderos positivos por la red).

Esta es una aplicación canónica del pipeline `Conv → ReLU → Pool` apilado, con una cabecera de clasificación binaria (cancer / no-cancer) entrenada con cross-entropy. La diferencia respecto al caso académico no es la arquitectura: es la curaduría del dataset, el etiquetado por consenso de expertos, y el seguimiento longitudinal de pacientes.

---

## 19. Detección de objetos: del sliding window a R-CNN

Detección no es solo "qué hay en la imagen" sino también "**dónde** está y **cuántos** hay" *(slides 58-62)*.

![Slide 58 — Detección: por cada objeto se predice la clase y la bounding box $(x, y, w, h)$.](/videos/mit-6s191-l3-2026/slides/slide-58.png)

**Solución ingenua: sliding window** *(slide 60)*. Pasar una CNN clasificadora sobre cada parche de cada escala y posición posibles. Problema: combinatoriamente intratable — millones de pasadas por imagen.

**R-CNN (Girshick et al. 2014)** *(slide 61)*: dividir el problema en dos etapas:

![Slide 61 — Pipeline R-CNN: 1) input → 2) ~2000 region proposals (selective search) → 3) CNN sobre cada región (warpeada) → 4) clasificación.](/videos/mit-6s191-l3-2026/slides/slide-61.png)

1. **Input:** imagen completa.
2. **Region proposals:** un algoritmo no-aprendido (selective search) propone ~2000 regiones candidatas que podrían contener objetos.
3. **CNN features:** cada región se warpea a tamaño fijo y pasa por una CNN.
4. **Clasificación:** una cabecera por región decide la clase (o "no-objeto") y refina la bounding box.

Problemas reconocidos en la slide *(slide 61)*:

- **Lento:** 2000 forward passes de CNN por imagen → segundos por imagen, inviable en tiempo real.
- **Brittle:** las region proposals dependen de selective search, un algoritmo hecho a mano que no se entrena con el resto.

**Faster R-CNN (Ren et al. 2015)** *(slide 62)*: la red propone sus propias regiones. Una **Region Proposal Network (RPN)** aprende, end-to-end, dónde mirar. Una sola pasada convolucional sobre la imagen completa produce features compartidos que alimentan tanto la RPN como el clasificador. Resultado: orden de magnitud más rápido, end-to-end aprendible, y ya viable en tiempo casi-real.

Para máximas velocidades, los detectores **single-shot** (YOLO, Redmon et al. 2016; SSD, Liu et al. 2016) eliminan la etapa de propuesta y predicen clases + bounding boxes directamente desde una grid sobre la imagen — la clase no profundiza en estos pero los menciona como evolución natural.

---

## 20. Segmentación semántica: redes fully-convolutional

Si detección produce **bounding boxes**, segmentación produce una **etiqueta por píxel** *(slide 63)*. Para una imagen $H \times W$, la salida también es $H \times W$ pero con valores en $\{1, \dots, K\}$ — el "color" semántico de cada píxel.

![Slide 63 — Fully Convolutional Network (FCN, Long et al. 2015): encoder convolucional que downsamplea, decoder convolucional que upsamplea con transposed convolution, salida $H \times W$ por píxel.](/videos/mit-6s191-l3-2026/slides/slide-63.png)

La idea fundadora (Long et al. 2015): reemplazar las capas FC al final del backbone por más capas convolucionales, de modo que la salida también sea un mapa 2D — de ahí "fully convolutional". El reto es que el backbone reduce la resolución (por pooling), pero la salida tiene que volver a $H \times W$. Solución: capas de **transposed convolution** (a veces mal llamadas "deconvolution") que upsamplean los feature maps.

La arquitectura encoder-decoder simétrica **U-Net** (Ronneberger et al. 2015) añade *skip connections* que copian features de alta resolución del encoder al decoder, permitiendo recuperar detalles finos perdidos por el downsampling. U-Net es el caballo de batalla de segmentación médica desde 2015.

En código: `tf.keras.layers.Conv2DTranspose` y `torch.nn.ConvTranspose2d`.

---

## 21. Control continuo: visión para conducción autónoma

El último gran caso del lecture *(slides 64-66)* es la conducción autónoma como una tarea **end-to-end de visión a control**. En vez de la pipeline clásica (percepción → planificación → control con módulos separados), la idea es entrenar una sola red que, dada cámara + mapa, emita directamente comandos de dirección con incertidumbre.

![Slide 65 — Framework end-to-end (Amini et al. ICRA 2019): múltiples convolucionales sobre cámara y mapa coarse, fusión, regresión a parámetros $(\mu, \sigma)$ de una distribución de control.](/videos/mit-6s191-l3-2026/slides/slide-65.png)

La salida no es un escalar (ángulo de volante) sino los parámetros $(\mu_i, \sigma_i)$ de una mezcla gaussiana sobre comandos de control. La pérdida es la log-verosimilitud negativa:

$$
\mathcal{L}(\theta \mid I, M) = -\log p(\theta \mid I, M)
$$

donde $\theta$ es el comando de control deseado, e $I, M$ son la imagen y el mapa coarse. Modelar incertidumbre permite que el sistema reconozca "no sé" en vez de comprometerse confiadamente con un comando equivocado — propiedad crítica en seguridad.

La demo de la clase *(slide 66)* muestra un Toyota Prius modificado conduciendo en autopilot con esta política aprendida end-to-end (Amini et al. ICRA 2019).

---

## 22. Cierre: alcance e impacto

El lecture cierra con un collage del impacto del campo *(slide 67)* y un resumen de tres bloques *(slide 68)*:

| Bloque | Contenido |
| --- | --- |
| **Foundations** | ¿Por qué visión computacional? Imágenes como números. Convolución como extracción de features local. |
| **CNNs** | Arquitectura: Conv + ReLU + Pool apilados. Aplicación a clasificación, ImageNet como benchmark. |
| **Applications** | Segmentación, image captioning, control. Seguridad, medicina, robótica. |

La slide final *(slide 69)* enlaza al **Lab 2: Facial Detection Systems**, donde los estudiantes implementan una CNN de detección facial, evaluando sesgos demográficos del modelo entrenado.

---

## Atribución

> Material adaptado de **MIT 6.S191 (2026) Lecture 3: Deep Computer Vision**, Alexander Amini, 6 de enero de 2026.
> [Video](https://www.youtube.com/watch?v=pqIcoskUuWs) — [Slides oficiales](https://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L3.pdf) — [Sitio del curso](https://introtodeeplearning.com/).
> Notas en español como elaboración independiente. Sin afiliación oficial con MIT.
