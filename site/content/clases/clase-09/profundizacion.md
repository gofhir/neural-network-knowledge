---
title: "Profundizacion - Analisis de Arquitecturas"
weight: 20
math: true
---

## 1. El Problema que Motiva Todo

Toda arquitectura de red neuronal profunda intenta responder:

> Como construir una funcion lo suficientemente expresiva para representar conceptos complejos, sin que sea imposible de entrenar ni de usar en produccion?

| Tension | Lado A | Lado B |
|---------|--------|--------|
| Capacidad vs. eficiencia | Mas parametros = mas expresividad | Mas parametros = mas computo |
| Profundidad vs. entrenabilidad | Mas capas = mas abstraccion | Mas capas = vanishing gradient |
| Generalizacion vs. memorizacion | Buena accuracy en train | Mala accuracy en test |

### El benchmark: ImageNet (ILSVRC)

| Ano | Ganador | Top-5 Error | Innovacion clave |
|-----|---------|-------------|-----------------|
| 2010 | Hand-crafted | ~28% | -- |
| 2012 | **AlexNet** | **16.4%** | ReLU, Dropout, GPU |
| 2014 | **GoogLeNet** | **6.7%** | Inception modules |
| 2014 | **VGG** | **7.3%** | Profundidad + 3x3 |
| 2015 | **ResNet** | **3.57%** | Skip connections |
| (humano) | -- | ~5% | -- |

En 2015, ResNet supero el rendimiento humano en Top-5 error.

---

## 2. Campo Receptivo: Profundo

### Definicion precisa

El campo receptivo de una neurona en capa $n$ es el conjunto de pixeles de la imagen de entrada que influyen en ella. Para convoluciones con filtro $k$ y stride $s=1$:

$$RF_n = RF_{n-1} + (k-1)$$

Para filtros 3x3: $RF_n = (2n+1) \times (2n+1)$

### Tres estrategias para aumentar el campo receptivo

{{< concept-alert type="clave" >}}
Todas las innovaciones en arquitecturas CNN son, en el fondo, estrategias para **aumentar el campo receptivo de forma eficiente** mientras se mantiene el entrenamiento estable.
{{< /concept-alert >}}

1. **Filtros grandes (AlexNet):** 11x11 = 121 parametros. Costoso.
2. **Apilar filtros pequenos (VGG):** 2 capas de 3x3 = campo 5x5 con menos parametros + no-linealidad extra.
3. **Filtros en paralelo (Inception):** La red aprende que escala es relevante.
4. **Skip connections + profundidad (ResNet):** Profundidad arbitraria sin degradacion.

---

## 3. VGG en Profundidad

### La decision de usar solo filtros 3x3

**Argumento 1:** El filtro 3x3 cubre las 8 posiciones vecinas de cada pixel mas el centro. Es el mas pequeno con riqueza espacial completa.

**Argumento 2 (eficiencia de parametros):**

Para campo receptivo 5x5 con $C$ canales de entrada:

| Opcion | Parametros |
|--------|-----------|
| 1 capa 5x5 | $25C^2$ |
| 2 capas 3x3 | $18C^2$ (28% menos) |

Para campo receptivo 7x7:

| Opcion | Parametros |
|--------|-----------|
| 1 capa 7x7 | $49C^2$ |
| 3 capas 3x3 | $27C^2$ (45% menos) |

### Configuraciones evaluadas

| Config | Capas | Parametros |
|--------|-------|-----------|
| A | 11 | 133M |
| D (VGG-16) | 16 | **138M** |
| E (VGG-19) | 19 | **144M** |

---

## 4. Inception en Profundidad

### Por que usar filtros 1x1

Un filtro 1x1 actua como una **proyeccion lineal** sobre el eje de canales. Es analogo a un embedding: comprime la informacion sin perder informacion espacial.

### Ahorro de parametros

Primer bloque Inception (filtros 5x5, input 192 canales, output 32):

| Caso | Parametros |
|------|-----------|
| Sin 1x1 | 153,600 |
| Con 1x1 (16 intermedios) | 15,872 (~90% menos) |

### Clasificadores auxiliares y vanishing gradient

Las redes profundas sufren vanishing gradient. Inception inyecta gradiente en capas intermedias con clasificadores auxiliares. En inferencia solo se usa el final.

---

## 5. ResNet en Profundidad

### El experimento clave

Red plain de 56 capas tiene mayor error que la de 20 capas, tanto en train como en test. No es overfitting: es un problema de optimizacion.

### La hipotesis

> No todos los sistemas son igualmente faciles de optimizar. Es mas dificil aprender la funcion identidad directamente que aprender una perturbacion de cero.

### Bloque residual formal

Sea $H(x)$ el mapeo deseado. La red aprende:

$$F(x) := H(x) - x$$

Por lo tanto: $H(x) = F(x) + x$

Si la identidad es optima, basta con llevar los pesos de $F(x)$ a cero.

### Bottleneck (ResNet-50+)

Usa 3 capas (1x1 reduce, 3x3 convolucion, 1x1 restaura) para reducir costo computacional.

| Version | FLOPs |
|---------|-------|
| ResNet-18 | 1.8 x 10^9 |
| ResNet-50 | 3.8 x 10^9 |
| ResNet-152 | 11.3 x 10^9 |

---

## 6. Guia de Decision: Que Arquitectura Usar

| Criterio | Recomendacion |
|----------|--------------|
| Baseline simple | ResNet-18/34 |
| Maxima accuracy | ResNet-152 o Inception-v4 |
| Dispositivos moviles | MobileNet |
| Pocas operaciones | GoogLeNet |
| Transfer learning | ResNet-50 (preentrenado) |

---

## 7. Interpretabilidad: Analisis Profundo

### Feature Visualization: gradient ascent en el input

En lugar de actualizar pesos, actualizamos la imagen para maximizar activaciones. Problemas: imagenes ruidosas y patrones de tablero.

**Soluciones de regularizacion:**
- L1 regularization y Total Variation (penalizar ruido)
- Robustez a transformaciones (jitter, rotate, scale)
- Espacio decorrelado (optimizar en frecuencias de Fourier)

### Attribution: causalidad en redes neuronales

| Metodo | Tipo | Descripcion |
|--------|------|-------------|
| Gradient (Vanilla) | Backprop | $\partial\text{output}/\partial\text{input}$ directamente |
| Guided Backprop | Backprop | Solo propaga gradientes positivos por ReLUs |
| Occlusion | Perturbacion | Desliza parche negro y mide caida |
| Extremal Perturbation | Perturbacion | Aprende mascara optima por optimizacion |

{{< concept-alert type="clave" >}}
La interpretabilidad es critica para detectar **sesgos y shortcuts daninos**. Ejemplo real: una red clasificaba "caballo" basandose en el watermark de copyright, no en el animal. Attribution lo revelo claramente.
{{< /concept-alert >}}

### Filter-Concept Overlap

La relacion entre filtros y conceptos no es 1:1:
- Un filtro puede responder a multiples conceptos (polisemia)
- Un concepto puede activar multiples filtros
