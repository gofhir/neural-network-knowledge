---
title: "Profundización - Análisis de Video"
weight: 20
math: true
---

> **Desarrollo formal de la Clase 36.** La [teoría](/clases/clase-36/teoria) recorre el campo de forma narrativa; aquí se formalizan los métodos. Cinco partes: (1) el flujo óptico; (2) el 2D CNN por frame y su límite; (3) el 2D CNN + RNN (LRCN); (4) two-stream y convoluciones 3D; (5) la síntesis (I3D) y el muestreo esparcido (TSN).

---

## 1. Flujo óptico

El flujo óptico busca el campo de desplazamiento $(u, v)$ de cada píxel entre dos frames. Parte de la **constancia de brillo** —un punto conserva su intensidad al moverse—:

$$
I(x, y, t) = I(x + u\,dt, y + v\,dt, t + dt).
$$

Expandiendo en serie de Taylor y quedándose con el primer orden se obtiene la **ecuación de restricción del flujo óptico**:

$$
I_x\, u + I_y\, v + I_t = 0.
$$

Una ecuación, dos incógnitas por píxel: el **problema de apertura**. Los métodos clásicos lo resuelven con supuestos —suavidad global (Horn-Schunck) o flujo local constante (Lucas-Kanade). [FlowNet](/papers/flownet-dosovitskiy-2015) lo aprende con una CNN, usando una **capa de correlación** que compara features entre las dos imágenes y un dataset sintético (*Flying Chairs*) para sortear la falta de ground-truth. Detalle en el fundamento [Flujo óptico](/fundamentos/flujo-optico).

---

## 2. El 2D CNN por frame y su límite

El enfoque base aplica una CNN 2D $\phi$ a cada frame $x_t$ y agrega las predicciones (por promedio o *pooling*):

$$
\hat y = \text{agregar}\big(\phi(x_1), \phi(x_2), \dots, \phi(x_T)\big).
$$

El problema es que el promedio (o cualquier agregación **simétrica**) es **invariante al orden**: da el mismo resultado para $(x_1, \dots, x_T)$ y para cualquier permutación de los frames. Formalmente, la representación pierde toda información sobre la **secuencia** temporal —justo la que distingue "abrir una puerta" de "cerrar una puerta". Esta es la limitación que motiva todas las arquitecturas siguientes.

---

## 3. 2D CNN + RNN (LRCN)

La solución que propone la clase reemplaza la agregación simétrica por una **RNN** que procesa los features frame a frame respetando el orden. Con features $v_t = \phi(x_t)$, una **LSTM** mantiene un estado $h_t$ que resume la historia:

$$
h_t = \text{LSTM}(v_t, h_{t-1}), \qquad \hat y = \text{softmax}(W h_T + b).
$$

La LSTM, con sus compuertas (entrada, olvido, salida), captura **dependencias de largo alcance** —puede recordar lo que pasó al inicio del video al clasificar el final. Es el enfoque **[LRCN](/papers/lrcn-donahue-2015)** (Donahue, 2015), y es literalmente el "2D CNN + RNN" de la clase.

{{< concept-alert type="advertencia" >}}
El costo del RNN es su **secuencialidad**: $h_t$ depende de $h_{t-1}$, así que los frames deben procesarse **en orden, uno tras otro** —no se puede paralelizar el eje temporal. La parte CNN sí se paraleliza (los frames son independientes al extraer features), pero la recurrencia no. Esta es exactamente la desventaja que señala la slide, y una de las razones por las que arquitecturas posteriores buscaron alternativas paralelizables.
{{< /concept-alert >}}

---

## 4. Two-stream y convoluciones 3D

### 4.1 Two-stream: separar apariencia y movimiento

**[Two-Stream](/papers/two-stream-simonyan-2014)** (Simonyan & Zisserman, 2014) usa **dos** redes 2D en paralelo:

- **Stream espacial:** una CNN sobre frames **RGB** individuales, que captura apariencia (objetos, escena).
- **Stream temporal:** una CNN sobre un **stack de campos de flujo óptico** de varios frames, que captura movimiento explícitamente.

Sus predicciones se **fusionan** al final (promedio o SVM). La idea —inspirada en las dos vías del sistema visual humano— es que el flujo óptico entrega el movimiento "servido en bandeja", sin que la red tenga que inferirlo del RGB. Resuelve directamente la debilidad del 2D CNN por frame.

### 4.2 Convoluciones 3D: aprender el movimiento

**[C3D](/papers/c3d-tran-2015)** (Tran, 2015) generaliza la convolución 2D a **3D**: el kernel se extiende también en el **tiempo**. Una convolución 3D con kernel $d \times k \times k$ aplicada a un volumen de frames produce, en la posición $(x, y, t)$:

$$
(\,W * V\,)_{x,y,t} = \sum_{i}\sum_{j}\sum_{\tau} W_{i,j,\tau}\; V_{x+i,\; y+j,\; t+\tau}.
$$

Al convolucionar sobre el eje temporal $\tau$, la red aprende **features espacio-temporales** directamente —el movimiento emerge de los datos, sin flujo óptico externo. El paper mostró que un kernel temporal de **3** (kernels $3\times3\times3$) es el óptimo. Su costo: muchos parámetros y cómputo/memoria.

---

## 5. La síntesis (I3D) y el muestreo esparcido (TSN)

### 5.1 I3D: inflar y transferir

**[I3D](/papers/i3d-carreira-2017)** (Carreira & Zisserman, 2017) une las mejores ideas. Toma una CNN 2D probada (Inception) **pre-entrenada en ImageNet** e **infla** cada kernel 2D de $N \times N$ a 3D de $N \times N \times N$, **repartiendo** los pesos a lo largo del eje temporal y dividiendo por $N$ (para que una entrada constante en el tiempo —un "video aburrido"— produzca la misma respuesta que la red 2D sobre una imagen):

$$
W^{3D}_{i,j,\tau} = \frac{1}{N}\, W^{2D}_{i,j} \quad \forall \tau.
$$

Así el modelo 3D **hereda** el pre-entrenamiento de imágenes. Combinado con two-stream (RGB + flujo) y pre-entrenado en **[Kinetics](/papers/kinetics-kay-2017)**, I3D estableció el paradigma dominante: **pre-entrenar en Kinetics, transferir** a UCF101/HMDB, con grandes saltos de accuracy.

### 5.2 TSN: muestreo esparcido por segmentos

**[TSN](/papers/tsn-wang-2016)** (Wang, 2016) ataca un problema distinto: los frames consecutivos son **redundantes**, y procesarlos densos es caro y de rango corto. TSN divide el video en $K$ **segmentos** iguales, muestrea un *snippet* aleatorio $S_k$ de cada uno, los procesa con una red compartida $\phi$ y **agrega** con una función de consenso $g$:

$$
\text{TSN}(S_1, \dots, S_K) = h\Big(g\big(\phi(S_1), \dots, \phi(S_K)\big)\Big),
$$

donde $g$ suele ser un promedio y $h$ un softmax. El muestreo esparcido **cubre todo el video** —incluyendo su rango temporal largo— con muy poco cómputo. Es la idea que usa el [laboratorio de la clase](/laboratorios/lab-36): muestrear un número fijo de frames distribuidos y agregar sus predicciones.

---

## 6. Síntesis

El arco de la clase, en una línea: el 2D CNN por frame **descarta el orden temporal** (agregación simétrica); las arquitecturas posteriores lo recuperan de tres maneras complementarias —**secuencialmente** (RNN/LRCN), **con movimiento explícito** (two-stream + flujo óptico) o **con convolución en el tiempo** (C3D/I3D)— y las buenas prácticas de **muestreo** (TSN) y **transferencia** (pre-entrenar en Kinetics) hacen todo esto viable en la práctica. El movimiento, ignorado por el enfoque base, es el hilo que conecta cada avance.

---

**Ver también:** [Clase 36 - Teoría](/clases/clase-36/teoria) · [Clase 36 - Práctica](/clases/clase-36/practica) · Fundamentos: [Análisis de Video](/fundamentos/analisis-de-video) · [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) · [Flujo óptico](/fundamentos/flujo-optico).
