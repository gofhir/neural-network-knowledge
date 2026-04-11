---
title: "Teoria - Arquitecturas Profundas e Interpretabilidad"
weight: 10
math: true
---

## 1. Contexto

Dado el mismo dataset, distintas arquitecturas de CNN producen resultados muy diferentes. No existe "la mejor arquitectura": la eleccion depende del trade-off entre precision, velocidad y memoria.

| Arquitectura | Ano | Idea central | Parametros | Top-1 ImageNet |
|-------------|-----|-------------|-----------|----------------|
| AlexNet | 2012 | ReLU y Dropout | ~60M | ~56% |
| VGG-16 | 2014 | Profundidad con filtros 3x3 | 138M | ~74% |
| GoogLeNet | 2014 | Modulos Inception + 1x1 conv | ~6.8M | ~69% |
| ResNet-50 | 2016 | Conexiones residuales | ~25M | ~76% |
| ResNet-152 | 2016 | Residuales muy profundos | ~60M | ~77% |

---

## 2. Campo Receptivo (Receptive Field)

{{< concept-alert type="clave" >}}
El **campo receptivo** de una neurona es la porcion de la imagen de entrada que influencio su activacion. Apilar capas pequenas (3x3) aumenta el campo receptivo sin aumentar proporcionalmente los parametros. Este insight motiva el diseno de VGG.
{{< /concept-alert >}}

| Configuracion | Campo Receptivo | Parametros |
|--------------|----------------|------------|
| 1 capa con filtro 5x5 | 5x5 | 25 |
| 2 capas con filtros 3x3 | 5x5 | 18 (28% menos) |

---

## 3. VGG

> Paper: Simonyan & Zisserman (2014). *Very deep convolutional networks for large-scale image recognition.*

### Arquitectura VGG-16

```mermaid
graph TD
    IN["Entrada<br/>224x224x3 RGB"]:::input
    B1["Bloque 1: conv3-64 x2 + MaxPool<br/>112x112x64"]:::conv
    B2["Bloque 2: conv3-128 x2 + MaxPool<br/>56x56x128"]:::conv
    B3["Bloque 3: conv3-256 x3 + MaxPool<br/>28x28x256"]:::conv
    B4["Bloque 4: conv3-512 x3 + MaxPool<br/>14x14x512"]:::conv2
    B5["Bloque 5: conv3-512 x3 + MaxPool<br/>7x7x512"]:::conv2
    FC["FC-4096 — FC-4096 — FC-1000"]:::fc
    SM["Softmax"]:::output

    IN --> B1 --> B2 --> B3 --> B4 --> B5 --> FC --> SM

    classDef input fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef conv fill:#2563eb,color:#fff,stroke:#1e40af
    classDef conv2 fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef fc fill:#60a5fa,color:#fff,stroke:#3b82f6
    classDef output fill:#059669,color:#fff,stroke:#047857
```

### Key Insight: Filtros 3x3

El filtro 3x3 es el mas pequeno que captura las nociones espaciales esenciales (arriba, abajo, izquierda, derecha, centro). Dos capas de 3x3 logran el mismo campo receptivo que una capa de 5x5, con 28% menos parametros y una no-linealidad extra (ReLU entre capas).

---

## 4. Inception / GoogLeNet

> Paper: Szegedy et al. (2014). *Going deeper with convolutions.*

### Motivacion

Los objetos en una imagen pueden aparecer a distintas escalas. En lugar de elegir un tamano de filtro, Inception usa **todos al mismo tiempo** y concatena los resultados.

### Modulo Inception con reduccion

Se anaden **convoluciones 1x1 antes** de las 3x3 y 5x5 para reducir canales:

| Caso | Calculo | Parametros |
|------|---------|-----------|
| Sin 1x1 | 192 x 5 x 5 x 32 | 153,600 |
| Con 1x1 (16 intermedios) | 192x1x1x16 + 16x5x5x32 | 15,872 |

**Reduccion: ~90% menos parametros.**

### Average Pooling en lugar de FC

GoogLeNet reemplaza las capas densas finales por **Average Pooling global**: de ~50M parametros en FC a ~1M.

### Clasificadores auxiliares

Las redes profundas sufren de **vanishing gradient**. Inception agrega 2 clasificadores auxiliares en capas intermedias para inyectar gradiente. Solo se usa el clasificador final en inferencia.

---

## 5. ResNet

> Paper: He et al. (2016). *Deep residual learning for image recognition.* CVPR 2016.

### El problema: mas capas no siempre es mejor

Una red de 56 capas tiene **mayor error** que una de 20, tanto en train como en test. No es overfitting, es un problema de **optimizacion**.

### Residual Learning

{{< concept-alert type="clave" >}}
En lugar de aprender el mapeo $H(x)$ directamente, la red aprende el **residuo** $F(x) = H(x) - x$, de modo que $H(x) = F(x) + x$. Si la identidad es optima, es mas facil llevar $F(x)$ a cero que aprender la identidad completa.
{{< /concept-alert >}}

### Bloque Residual

```mermaid
graph TD
    X["x"]:::input --> W1["Weight layer"]:::layer
    X --> SK[ ]:::skip
    W1 --> R1["ReLU"]:::act
    R1 --> W2["Weight layer"]:::layer
    W2 --> ADD(("+")):::op
    SK --> ADD
    ADD --> R2["ReLU"]:::act
    R2 --> OUT["F(x) + x"]:::output

    linkStyle 1 stroke:#ef4444,stroke-width:2px

    classDef input fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef layer fill:#2563eb,color:#fff,stroke:#1e40af
    classDef act fill:#f59e0b,color:#fff,stroke:#d97706
    classDef op fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef output fill:#059669,color:#fff,stroke:#047857
    classDef skip fill:none,stroke:none
```

### Bottleneck Block (ResNet-50/101/152)

```text
256-d input
    |
  1x1, 64   <- reduce dimensionalidad
    |
  3x3, 64   <- convolucion espacial
    |
  1x1, 256  <- restaura dimensionalidad
    |
    + (skip connection)
```

### Decisiones de diseno

- **Batch Normalization** despues de cada capa convolucional
- **Sin Dropout**: la regularizacion la aportan los residuales y el BN

---

## 6. Interpretabilidad

### Por que es importante

Despues de entrenar una CNN, podemos preguntar: **que esta aprendiendo realmente la red?**

| Tecnica | Pregunta que responde |
|---------|----------------------|
| **Feature Visualization** | Que patrones de entrada activan maximamente una parte de la red? |
| **Attribution** | Que region de *esta* imagen es responsable de *esta* prediccion? |

### Feature Visualization

Las redes son diferenciables con respecto a su entrada. Podemos hacer **gradient ascent sobre el input**: actualizar la imagen para maximizar la activacion de un objetivo.

$$x^* = \arg\max_x \; \text{objetivo}(\text{red}(x))$$

Lo que aprende GoogLeNet capa a capa:

| Capas | Tipo de feature |
|-------|----------------|
| conv2d0-2 | Bordes y gradientes basicos |
| mixed3a-3b | Texturas (puntos, lineas) |
| mixed4a-4b | Patrones complejos |
| mixed4b-4c | Partes de objetos (ojos, patas) |
| mixed4d-4e | Objetos reconocibles |

### Attribution

Responde: *que region de esta imagen causo esta prediccion?*

**Caso real de bias:** una red clasificaba "caballo" basandose en el watermark de copyright, no en el caballo.

| Metodo | Tipo | Fortaleza |
|--------|------|-----------|
| Gradient | Backprop | Rapido, sensibilidad local |
| Guided Backprop | Backprop | Filtros mas limpios |
| Grad-CAM | Backprop | Mapas de calor semanticos |
| Occlusion | Perturbacion | Intuitivo, pero lento |
| RISE | Perturbacion | Robusto al ruido |
| Extremal Perturbation | Perturbacion | Mascaras precisas |

### Perturbacion Extremal

Aprender una mascara de tamano fijo $m$ que preserve maximamente la salida de la red:

$$\arg\max_m \; \Phi(m \otimes x) \quad \text{sujeto a: } \text{area}(m) = a$$

> "Si solo te dejo ver una pequena region de la imagen, que region elegiria la red para reconocer mejor el objeto?"
