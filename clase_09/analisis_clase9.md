# Clase 9 — Deep Network Architectures & Interpretability
**Profesor:** Miguel Fadic | **Escuela de Ingeniería, UC — Educación Profesional**

---

## Tabla de Contenidos

1. [Contexto: ¿Por qué importan las arquitecturas?](#1-contexto)
2. [Preliminar: Campo Receptivo (Receptive Field)](#2-campo-receptivo)
3. [VGG — La red más profunda de su época](#3-vgg)
4. [Inception / GoogLeNet](#4-inception--googlenet)
5. [ResNet — Aprendizaje Residual](#5-resnet)
6. [Comparativa final de arquitecturas](#6-comparativa-final)
7. [Interpretabilidad: ¿Qué está mirando la red?](#7-interpretabilidad)
8. [Feature Visualization — Visualización de Características](#8-feature-visualization)
9. [Attribution — Atribución](#9-attribution)
10. [Perturbación Extremal (Extremal Perturbation)](#10-perturbaci%C3%B3n-extremal)
11. [Referencias](#11-referencias)

---

## 1. Contexto

### El problema central

Dado el mismo dataset, distintas arquitecturas de CNN producen resultados muy diferentes. El gráfico comparativo presentado en clase (Canziani et al., 2016) ilustra esto claramente:

| Eje | Significado |
|-----|-------------|
| **X** | Operaciones computacionales (Giga-Ops) |
| **Y** | Exactitud Top-1 en ImageNet (%) |
| **Tamaño burbuja** | Número de parámetros (millones) |

**Observaciones clave del gráfico:**
- **VGG-16 y VGG-19** tienen ~138–144M de parámetros y requieren ~30–40 G-Ops, con ~74% de accuracy. Son precisas pero muy costosas.
- **ResNet-152** alcanza ~77% con solo ~11 G-Ops.
- **Inception-v4** logra ~80% con ~12 G-Ops.
- **MobileNets** están en la esquina inferior izquierda: pocos parámetros, pocas operaciones, menor exactitud — ideales para dispositivos móviles.
- **AlexNet** fue el pionero, pero hoy es considerado el menos eficiente.

> **Conclusión:** No existe "la mejor arquitectura" en absoluto. La elección depende del trade-off entre precisión, velocidad y memoria disponible.

---

## 2. Campo Receptivo

### Concepto fundamental

El **campo receptivo** (*receptive field*) de una neurona es la porción efectiva de la imagen de entrada que influenció su activación.

**Ejemplo visual del slide:**
- Una neurona en la capa 3 usa filtros de **3×3** en cada capa.
- Sin embargo, su campo receptivo con respecto a la **imagen de entrada** es de **5×5**.

```
Layer 1   →   Layer 2   →   Layer 3
  5×5    contiene  3×3    contiene  1 neurona
(en el input)  (en Layer 1)  (en Layer 2)
```

### ¿Por qué importa?

Apilar capas pequeñas aumenta el campo receptivo sin aumentar proporcionalmente el número de parámetros. Este es un insight clave que motiva el diseño de VGG.

---

## 3. VGG

> **Paper original:** Simonyan, K., & Zisserman, A. (2014). *Very deep convolutional networks for large-scale image recognition.* arXiv:1409.1556.

### Descripción general

VGG fue **la red más profunda de su época** al ganar ILSVRC 2014. Su arquitectura es conceptualmente simple: apilar muchas capas convolucionales pequeñas.

### Arquitectura VGG-16 (configuración D)

```
Entrada: 224×224×3 (RGB)

Bloque 1: conv3-64 × 2  → MaxPool  → 112×112×64
Bloque 2: conv3-128 × 2 → MaxPool  → 56×56×128
Bloque 3: conv3-256 × 3 → MaxPool  → 28×28×256
Bloque 4: conv3-512 × 3 → MaxPool  → 14×14×512
Bloque 5: conv3-512 × 3 → MaxPool  → 7×7×512

Capas Fully Connected:
FC-4096 → FC-4096 → FC-1000 → Softmax
```

### Key Insights de VGG

#### 1. Filtros 3×3

El filtro **3×3** es el **más pequeño** que captura las nociones espaciales esenciales:
- Arriba / Abajo
- Izquierda / Derecha
- Centro

#### 2. Composición de filtros pequeños = campo receptivo grande con menos parámetros

| Configuración | Campo Receptivo | Parámetros |
|--------------|----------------|------------|
| 1 capa con filtro 5×5 | 5×5 | 5×5 = **25** |
| 2 capas con filtros 3×3 | 5×5 | 3×3×2 = **18** |

> Con dos capas de 3×3 logramos el **mismo campo receptivo** que una capa de 5×5, pero con **28% menos parámetros**.

**Beneficios de menos parámetros:**
- Menor tiempo de predicción (inferencia más rápida)
- Menor riesgo de overfitting

#### 3. Configuraciones evaluadas

| Config | Capas | Parámetros |
|--------|-------|-----------|
| A | 11 | 133M |
| A-LRN | 11 | 133M |
| B | 13 | 133M |
| C | 16 | 134M |
| D (VGG-16) | 16 | **138M** |
| E (VGG-19) | 19 | **144M** |

La configuración **D** (VGG-16) es la más utilizada en la práctica.

---

## 4. Inception / GoogLeNet

> **Paper original:** Szegedy, C., et al. (2014). *Going deeper with convolutions.* arXiv:1409.4842.

### Motivación

VGG resolvió la profundidad, pero a un costo computacional enorme. Inception busca:
1. **Detección multi-escala**: objetos pequeños necesitan filtros pequeños; objetos grandes necesitan filtros grandes.
2. **Reducir el número de parámetros y operaciones**.

### El problema multi-escala

Los objetos en una imagen pueden aparecer a **distintas escalas**:
- Un objeto pequeño (ej. un pájaro lejano) requiere filtros pequeños.
- Un objeto grande (ej. un edificio) requiere filtros grandes o redes más profundas (más campo receptivo).

**Idea de Inception:** En lugar de elegir un tamaño de filtro, ¿por qué no usar **todos al mismo tiempo** y dejar que la red aprenda cuáles son más útiles?

### El Módulo Inception

#### Versión naïve

```
                  ┌─────────────────────────────────────┐
                  │         Filter Concatenation         │
                  └─────────┬──────┬────────┬────────────┘
                             │      │        │
                    1×1      3×3   5×5   3×3 MaxPool
                    conv     conv  conv  (padding)
                             │      │        │
                  └──────────┴──────┴────────┴────────────┘
                                Previous Layer
```

Todos los filtros operan en paralelo sobre la misma entrada y sus salidas se **concatenan** en el eje de canales.

**Problema:** Las convoluciones 5×5 sobre muchos canales son muy costosas.

#### Versión con reducción de dimensionalidad

Se añaden **convoluciones 1×1 antes** de las 3×3 y 5×5 para reducir la profundidad (número de canales):

```
Previous Layer
    ↓
1×1 conv (reduce canales)    → 3×3 conv
1×1 conv (reduce canales)    → 5×5 conv
3×3 MaxPool → 1×1 conv (proyección)
1×1 conv (directo)
    ↓ ↓ ↓ ↓
Filter Concatenation
```

### ¿Por qué usar filtros 1×1?

Un filtro **1×1** actúa como una **proyección lineal** sobre el eje de canales. Es análogo a un **embedding**: comprime la información en una representación más densa sin perder información espacial.

#### Ejemplo numérico de ahorro de parámetros

Primer bloque Inception (filtros 5×5):
- Input channels: **192**
- Output filters 5×5: **32**

| Caso | Cálculo | Parámetros |
|------|---------|-----------|
| Sin 1×1 | 192 × 5 × 5 × 32 | **153,600** |
| Con 1×1 (16 canales intermedios) | 192×1×1×16 + 16×5×5×32 | **15,872** |

**Reducción: ~90% menos parámetros** en esa capa.

### Average Pooling en lugar de capas FC

GoogLeNet reemplaza las capas densas al final por **Average Pooling global**:

| Alternativa | Parámetros |
|------------|-----------|
| 7×7×1024×1000 (FC clásico) | **50,176,000** |
| 1×1×1024×1000 (después de AvgPool) | **1,024,000** |

Reducción de **~49× en la última capa**.

### ¿Por qué hay 3 capas Softmax en GoogLeNet?

Las redes profundas sufren de **vanishing gradient**: el gradiente se desvanece al propagarse hacia atrás por muchas capas. La solución de Inception fue:

> Agregar **2 clasificadores auxiliares** en capas intermedias que también calculan la loss durante el entrenamiento, inyectando gradiente directamente en capas más tempranas.

Durante inferencia, solo se usa el clasificador final.

### Arquitectura GoogLeNet (tabla)

La red tiene 22 capas con parámetros, incluyendo:
- 2 capas convolucionales iniciales
- 9 módulos Inception (3a, 3b, 4a–4e, 5a, 5b)
- Average Pooling + Dropout(40%) + Linear

---

## 5. ResNet

> **Paper original:** He, Kaiming, et al. (2016). *Deep residual learning for image recognition.* CVPR 2016, p. 770-778.

### El problema: más capas ≠ mejor rendimiento

**Hipótesis intuitiva:** Una red de N+1 capas debería ser *al menos tan buena* como la de N capas. Si la capa extra no aporta, la red debería aprender la **función identidad** en esa capa.

**Realidad empírica (CIFAR-10):**

| Red | Error entrenamiento | Error test |
|-----|--------------------|-----------:|
| Plain 20-layer | Menor | Menor |
| Plain 56-layer | **Mayor** | **Mayor** |

La red de 56 capas tiene **mayor error que la de 20 capas**, tanto en train como en test. Esto no es overfitting (el error de entrenamiento también empeora). Es un problema de **optimización**.

### La hipótesis de ResNet

> No todos los sistemas son igualmente fáciles de optimizar. Es más difícil aprender la función identidad directamente que aprender una **perturbación de cero**.

### La solución: Residual Learning

**Definición formal:**
- Sea `H(x)` el mapeo deseado para una capa o bloque.
- En lugar de aprender `H(x)` directamente, la red aprende el **residuo**:
  ```
  F(x) := H(x) - x
  ```
- Por lo tanto: `H(x) = F(x) + x`

**¿Por qué es más fácil?**  
Si la identidad es la solución óptima, es más fácil llevar los pesos de `F(x)` a cero que aprender la función identidad completa desde cero.

### El Bloque Residual (Residual Block)

```
     x
     │
     ├────────────────────────┐
     │                        │ (shortcut / skip connection)
  weight layer                │
     │                        │
   relu                       │
     │                        │
  weight layer                │
     │                        │
     └────────────── + ───────┘
                     │
                   relu
                     │
               F(x) + x
```

La **shortcut connection** (conexión directa) suma la entrada `x` a la salida del bloque. No añade parámetros adicionales.

### Bottleneck Block (para ResNet-50/101/152)

Para redes más profundas, se usa una variante con 3 capas en lugar de 2:

```
256-d input
    │
  1×1, 64   ← reduce dimensionalidad (bottleneck)
    │
  3×3, 64   ← convolución espacial
    │
  1×1, 256  ← restaura dimensionalidad
    │
    + (skip connection)
    │
  relu
```

Esto reduce el costo computacional manteniendo el campo receptivo.

### Configuraciones de ResNet

| Versión | FLOPs | Bloques conv2_x | conv3_x | conv4_x | conv5_x |
|---------|-------|-----------------|---------|---------|---------|
| ResNet-18 | 1.8×10⁹ | ×2 | ×2 | ×2 | ×2 |
| ResNet-34 | 3.6×10⁹ | ×3 | ×4 | ×6 | ×3 |
| ResNet-50 | 3.8×10⁹ | ×3 (bottleneck) | ×4 | ×6 | ×3 |
| ResNet-101 | 7.6×10⁹ | ×3 | ×4 | **×23** | ×3 |
| ResNet-152 | 11.3×10⁹ | ×3 | ×8 | **×36** | ×3 |

### Otras decisiones de diseño

- **Batch Normalization** después de cada capa convolucional → estabiliza el entrenamiento
- **Sin Dropout** → la regularización la aportan los residuales y el BN

### Resultados de ResNet

Con residuales, la red de **34 capas supera a la de 18 capas**, demostrando que el aprendizaje residual permite escalar la profundidad de manera efectiva.

---

## 6. Comparativa Final

### Resumen de filosofías de diseño

| Arquitectura | Año | Idea central | Parámetros | Top-1 (ImageNet) |
|-------------|-----|-------------|-----------|-------------------|
| AlexNet | 2012 | Red profunda con ReLU y Dropout | ~60M | ~56% |
| VGG-16 | 2014 | Profundidad con filtros 3×3 | 138M | ~74% |
| GoogLeNet | 2014 | Módulos Inception + 1×1 conv | ~6.8M | ~69% |
| ResNet-50 | 2016 | Conexiones residuales | ~25M | ~76% |
| ResNet-152 | 2016 | Residuales muy profundos | ~60M | ~77% |
| Inception-v4 | 2017 | Inception + ResNet combinados | ~43M | ~80% |

### Principios que evolucionaron

1. **VGG:** Profundidad + filtros pequeños → mejor campo receptivo con menos parámetros por capa.
2. **Inception:** Paralelismo multi-escala + 1×1 conv → reduce operaciones drásticamente.
3. **ResNet:** Skip connections → permite entrenar redes arbitrariamente profundas.

---

## 7. Interpretabilidad

### ¿Por qué es importante?

Después de entrenar una CNN, podemos preguntarnos: **¿qué está aprendiendo realmente la red?**

Existen dos enfoques complementarios:

| Técnica | Pregunta que responde |
|---------|----------------------|
| **Feature Visualization** | ¿Qué patrones de entrada activan máximamente una parte de la red? |
| **Attribution** | ¿Qué región de *esta* imagen es responsable de *esta* predicción? |

### ¿Por qué preocuparse?

- **Detectar sesgos erróneos:** La red podría estar aprendiendo features irrelevantes (ej. el watermark de una imagen en lugar del contenido).
- **Verificar shortcuts dañinos:** Una red podría clasificar "caballo" basándose en el texto del copyright de la imagen.
- **Confianza en producción:** En dominios críticos (medicina, legal), se exige explicabilidad.

---

## 8. Feature Visualization

> **Referencia:** Olah, et al., "Feature Visualization", Distill, 2017.

### Principio fundamental

Las redes neuronales son **diferenciables con respecto a su entrada**. Esto nos permite hacer **gradient ascent sobre el input**: en lugar de actualizar los pesos, actualizamos la imagen para maximizar la activación de un objetivo.

```
x* = argmax_x  objetivo(red(x))
```

### Tipos de objetivos

| Objetivo | Notación | Qué visualiza |
|---------|----------|---------------|
| Neurona específica | layer_n[x,y,z] | El patrón que activa una neurona individual |
| Canal completo | layer_n[:,:,z] | El patrón promedio del canal z |
| Layer/DeepDream | layer_n[:,:,:]² | Amplifica cualquier patrón presente |
| Class Logits | pre_softmax[k] | La imagen "ideal" de la clase k (pre-softmax) |
| Class Probability | softmax[k] | La imagen que maximiza la probabilidad de k |

### Logits vs Probabilidad de clase

> Optimizar los **logits** produce imágenes de mejor calidad visual que optimizar la probabilidad softmax.

¿Por qué? Softmax tiene un efecto de competencia entre clases que puede llevar al optimizador a suprimir otras clases en lugar de potenciar la clase objetivo. Los logits son más directos.

### Problemas al optimizar directamente

#### 1. Imágenes ruidosas
La optimización sin restricciones genera imágenes con **ruido adversarial** de alta frecuencia que activan la red pero no son visualmente interpretables.

#### 2. Patrones de tablero (checkerboard)
Las capas con stride crean **patrones tipo tablero de ajedrez** en el gradiente durante la backpropagation. Esto contamina la visualización.

### Soluciones: Regularización

#### Penalización de frecuencia

Se penaliza la presencia de frecuencias altas (ruido). Técnicas:
- **L₁ regularization:** Penaliza la magnitud de los píxeles
- **Total Variation (TV):** Penaliza diferencias entre píxeles vecinos → suaviza la imagen
- **Blur:** Aplica suavizado gaussiano en cada paso

#### Robustez a transformaciones

En cada paso de optimización, se aplican **transformaciones aleatorias** a la imagen antes de evaluar la activación:
- **Jitter** (desplazamiento de 1px)
- **Rotate** (rotación de 5°)
- **Scale** (escala de 1.1×)

Esto fuerza a la imagen a activar la red independientemente de pequeñas perturbaciones, resultando en features más robustas y naturales.

#### Espacio decorrelado (Decorrelated Space)

En lugar de optimizar directamente en el espacio de píxeles (RGB correlacionado), se optimiza en un **espacio decorrelado** (similar a frecuencias de Fourier) y luego se transforma de vuelta. Esto mejora la calidad visual considerablemente.

| Métrica | Descripción |
|---------|-------------|
| L∞ metric | Gradiente regular → imágenes con ruido (usado en ataques adversariales) |
| L² metric | Gradiente regular en espacio Euclidiano → algo mejor |
| Espacio decorrelado | Cambia la parametrización de la imagen → mejor calidad visual |

### Resultado final: combinando todas las técnicas

Aplicar conjuntamente espacio decorrelado + robustez a transformaciones produce imágenes visualmente coherentes y significativas.

### Lo que aprende GoogLeNet capa a capa

| Capas | Tipo de feature |
|-------|----------------|
| conv2d0, conv2d1, conv2d2 | **Bordes** y gradientes básicos |
| mixed3a, mixed3b | **Texturas** (puntos, líneas, grillas) |
| mixed4a, mixed4b | **Patrones** complejos (flores, redes) |
| mixed4b, mixed4c | **Partes** de objetos (ojos, patas) |
| mixed4d, mixed4e | **Objetos** reconocibles |

Esta jerarquía confirma que las CNN aprenden representaciones de lo **simple a lo complejo**.

### Diversidad: el problema de una sola imagen

Visualizar un solo ejemplo no es suficiente. Una neurona puede activarse por **múltiples conceptos** diferentes.

**Ejemplo del slide:** El canal 143 de `mixed4a` parece activarse solo con "cabezas de perro". Pero con **diversidad en la optimización** (penalizando que múltiples soluciones sean similares), la misma neurona revela que también responde a otros patrones relacionados (orejas, snouts, etc.).

### Filter-Concept Overlap (Superposición Filtro-Concepto)

Un resultado sorprendente es que **la relación entre filtros y conceptos no es 1:1**:

- **Un filtro puede responder a múltiples conceptos** (polisemia de neuronas)
- **Un concepto puede activar múltiples filtros**

**Ejemplo:** El canal 55 de `mixed4e` visualmente parece detectar "gatos", pero con diversidad también detecta "zorros" y... "autos". El concepto de "auto" tiene overlap con el de "animal" en algunas neuronas.

---

## 9. Attribution

### Definición

La **atribución** responde a: *¿Qué región de esta imagen específica causó que la red produjera esta salida?*

Mientras Feature Visualization pregunta "¿qué busca la red en general?", Attribution pregunta "¿qué vio la red en *esta* imagen concreta?".

### Motivación

**Ejemplo del slide:** Una red clasifica una foto de un médico como "doctor". ¿Se basó en:
- El estetoscopio
- La ropa blanca
- La cara
- El fondo del hospital

Esto es crítico para detectar **sesgos y shortcuts**.

### Caso real de bias: el watermark del caballo

El ejemplo de debugging muestra que una red entrenada para clasificar caballos usaba el **watermark de copyright** en la esquina inferior izquierda de la imagen para hacer sus predicciones, no el caballo en sí. La atribución lo revela claramente: el mapa de calor está concentrado en ese texto invisible para el ojo humano.

### Dos ramas de métodos de atribución

#### 1. Backpropagation-based

Modifica las **reglas de backpropagation** para generar un mapa de saliencia de la pasada hacia atrás.

Ejemplos:
- **Gradient (Vanilla):** El gradiente `∂output/∂input` directamente
- **Guided Backpropagation:** Solo propaga gradientes positivos a través de ReLUs

#### 2. Perturbation-based

Cambia la entrada de manera controlada y observa cómo cambia la salida.

Ejemplos:
- **Occlusion:** Desliza un parche negro sobre la imagen y mide caída de activación
- **RISE:** Aplica máscaras aleatorias y promedia sus efectos
- **Extremal Perturbation:** Aprende la máscara óptima

### Comparación de métodos

| Método | Tipo | Fortaleza |
|--------|------|-----------|
| Gradient | Backprop | Rápido, indica sensibilidad local |
| Guided Backprop | Backprop | Filtros más limpios |
| Grad-CAM | Backprop | Mapas de calor de alta resolución semántica |
| Occlusion | Perturbación | Intuitivo, pero lento (O(N²)) |
| RISE | Perturbación | Robusto al ruido |
| Extremal Perturbation | Perturbación | Máscaras precisas y compactas |

---

## 10. Perturbación Extremal

> **Referencia:** Fong, R., Patrick, M., & Vedaldi, A. (2019). *Understanding deep networks via extremal perturbations and smooth masks.* ICCV 2019.

### Definición

Aprender una **máscara de tamaño fijo** `m` que, al aplicarse a la imagen `x`, **preserve máximamente la salida de la red** para una clase dada.

### Formulación matemática

```
argmax_m  Φ(m ⊗ x)
sujeto a: area(m) = a
```

Donde:
- `m` es la máscara binaria (o suave) de tamaño fijo
- `⊗` es la operación de perturbación (multiplicar y difuminar bordes)
- `Φ` es la red neuronal
- `a` es el área permitida (ej. 5%, 10%, 20% de la imagen)

### Intuición

> "Si solo te dejo ver una pequeña región de la imagen, ¿qué región elegiría la red para reconocer mejor el objeto?"

La máscara se **aprende por optimización** (gradient descent), no se define manualmente.

### Parámetro de área

El área define cuánta información puede revelar la máscara:

| Máscara | Área | Qué revela |
|---------|------|-----------|
| 5% | Muy pequeña | Solo el discriminador más puro |
| 10% | Pequeña | Región central del objeto |
| 20% | Mediana | Contexto adicional del objeto |

### Resultado en el ejemplo "chocolate sauce"

Comparando métodos:
- **Gradient / Guided:** Ruidosos, distribución difusa
- **Contrast Excitation / Grad-CAM / Occlusion:** Mapas de calor gruesos
- **Extremal Perturbation (máscara):** Región compacta y precisa alrededor del objeto

La máscara demuestra con números: pasar de 0.610 → 0.351 de probabilidad cuando se enmascara la región incorrecta, versus 0.610 → 0.015 cuando se enmascara la región correcta.

### Importancia práctica en modelos comparativos

Al comparar un modelo entrenado desde cero vs. un modelo fine-tuned:
- **Modelo base:** La máscara se concentra en el **fondo o contexto** (el modelo aprendió correlaciones espurias)
- **Modelo fine-tuned:** La máscara se concentra en el **objeto mismo** (el modelo aprendió la feature correcta)

---

## 11. Referencias

1. **SIMONYAN, Karen; ZISSERMAN, Andrew.** *Very deep convolutional networks for large-scale image recognition.* arXiv:1409.1556, 2014.

2. **SZEGEDY, C., et al.** *Going deeper with convolutions.* arXiv:1409.4842, 2014.

3. **HE, Kaiming, et al.** *Deep residual learning for image recognition.* CVPR 2016, p. 770-778.

4. **CANZIANI, A.; PASZKE, A.; CULURCIELLO, E.** *An analysis of deep neural network models for practical applications.* arXiv:1605.07678, 2016.

5. **OLAH, et al.** *Feature Visualization.* Distill, 2017.

6. **FONG, Ruth; PATRICK, Mandela; VEDALDI, Andrea.** *Understanding deep networks via extremal perturbations and smooth masks.* ICCV 2019, p. 2950-2958.

---

## Conceptos Clave para Recordar

| Concepto | Definición breve |
|---------|-----------------|
| Receptive field | Porción de la entrada que influyó en una neurona |
| Filtro 3×3 | El más pequeño con noción espacial; 2 capas = campo 5×5 con menos parámetros |
| Módulo Inception | Filtros paralelos de distintos tamaños concatenados |
| Convolución 1×1 | Reducción de canales = embedding convolucional |
| Residual block | F(x) + x; aprende perturbaciones respecto a la identidad |
| Vanishing gradient | El gradiente se desvanece en redes muy profundas |
| Batch Normalization | Normaliza activaciones de cada mini-batch; estabiliza el entrenamiento |
| Feature Visualization | Gradient ascent en el input para maximizar activaciones |
| Attribution | Identificar qué región del input causó una predicción |
| Extremal Perturbation | Máscara óptima de tamaño fijo que preserva la salida de la red |
