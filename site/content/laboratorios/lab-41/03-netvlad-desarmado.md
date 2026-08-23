---
title: "03 - NetVLAD desarmado"
weight: 30
math: true
---

> La identidad algebraica que convierte un softmax de distancias en un softmax lineal —y con eso desacopla los centroides de la asignación—, el broadcast de cinco dimensiones que el cálculo no necesita, la línea de 24 caracteres que es todo GhostVLAD, y la normalización que vuelve a VLAD ciego al número de descriptores.

---

## 1. La fórmula, y qué se descarta

$$v(j,k) = \sum_{i=1}^{N} \bar{a}_k(x_i) \cdot \big(x_i(j) - c_k(j)\big)$$

Para cada centroide `k` y cada dimensión `j`, se acumula el **residuo** `x_i(j) − c_k(j)` de todos los descriptores, ponderado por su pertenencia.

Lo esencial: **VLAD no guarda los descriptores ni su promedio. Guarda cómo se desvían de los centroides.** Si todos los descriptores cayeran exactamente sobre sus centroides, `V` sería la matriz cero y el audio no aportaría información. Lo informativo es el patrón de desvíos.

La salida es `J × K = 512 × 8 = 4.096` números, **independiente de cuántos descriptores entraron**. Ahí está resuelto el problema del largo variable: la suma sobre `i` absorbe la `N`.

Y una propiedad estructural que domina toda la [clase 41](/clases/clase-41): **la suma es invariante a permutaciones.** Reordena los descriptores en el tiempo y `V` no cambia. VLAD no sabe si dijiste «hola mundo» o «mundo hola». Para reconocimiento de hablante eso es exactamente lo que se quiere; para reconocimiento de habla es fatal.

---

## 2. Del `argmin` al softmax: una identidad, no una aproximación

[VLAD clásico](/papers/vlad-jegou-2010) usa **asignación dura**:

$$a_k(x_i) = \begin{cases} 1 & \text{si } k = \arg\min_{k'} \|x_i - c_{k'}\|^2 \\ 0 & \text{si no}\end{cases}$$

Un `argmin` **no es derivable**: gradiente cero en casi todas partes, inexistente en las fronteras. Con esa pertenencia no se puede entrenar nada por descenso de gradiente. Ese es *todo* el problema que [NetVLAD](/papers/netvlad-arandjelovic-2016) resuelve.

La solución natural es un softmax sobre distancias negativas. Pero la fórmula de la slide **no tiene distancias**: tiene un producto punto.

$$\bar{a}_k(x_i) = \frac{e^{\,w_k x_i + b_k}}{\sum_{k'} e^{\,w_{k'} x_i + b_{k'}}}$$

De dónde sale, expandiendo el cuadrado:

$$-\alpha\|x_i - c_k\|^2 = \underbrace{-\alpha\|x_i\|^2}_{\text{igual para todo } k} + \underbrace{2\alpha\, c_k^{\top} x_i}_{w_k^{\top}x_i} \underbrace{- \alpha\|c_k\|^2}_{b_k}$$

**El término `−α‖x_i‖²` no depende de `k`, así que se cancela en el softmax.** Queda exactamente un softmax lineal con `w_k = 2α·c_k` y `b_k = −α‖c_k‖²`.

Verificado numéricamente con 2.000 descriptores de 512-d y 10 centroides:

| α | max &#124;softmax de distancias − softmax lineal&#124; |
|---|---|
| 0,1 | 5,8×10⁻⁶ |
| 1,0 | 9,5×10⁻⁵ |
| 10,0 | 7,2×10⁻⁴ |

Idénticos hasta la precisión de `float32`. **No es una aproximación: es una identidad algebraica.**

### Y aquí está el truco de NetVLAD

Una vez reescrito así, `w_k`, `b_k` y `c_k` son **tres conjuntos de parámetros separados**. NetVLAD los **desacopla deliberadamente**: en lugar de exigir `w_k = 2α c_k`, los deja libres. El resultado es más expresivo que k-means — las fronteras de asignación ya no están obligadas a ser los bisectores perpendiculares entre centroides.

En el código, ese desacople es literal: `self.vlad_conv` produce los `w_k x_i + b_k` y `self.cluster` guarda los `c_k`, **sin ninguna relación entre ellos**.

### El α que desapareció

En la formulación desacoplada, α queda absorbido en la escala de `w_k`. Su rol es controlar la **dureza** de la asignación:

| α | masa en el centroide más cercano | entropía media (máx = 2,303) |
|---|---|---|
| 0,01 | 0,106 | 2,302 → asignación **uniforme** |
| 1,0 | 0,720 | 0,746 |
| 10 | 0,964 | 0,085 |
| 100 | **0,994** | 0,013 → asignación **dura** |

Con α → ∞ se recupera el `argmin` de Jégou. Con α → 0 todos los descriptores pertenecen a todos los clusters por igual y `V` colapsa a `x̄ − c̄`. **NetVLAD aprende dónde ubicarse en ese continuo**, a través de la magnitud de `w_k`. Es un caso de manual de «hacer diferenciable una operación discreta introduciendo una temperatura».

---

## 3. La bifurcación: dos ramas desde el mismo punto

```python
x = self.thin_resnet( x )                                    # (1, 512, 7, T/16)
x_fc       = self.block_1_activation( self.block_1( x ) )     # (1, 512, 1, T/16)
x_k_center = self.vlad_conv( x )                              # (1,  10, 1, T/16)
```

Lee la tercera línea con cuidado: **`vlad_conv` se aplica a `x`, no a `x_fc`.**

```
                       ┌─ block_1  (7×1, 512→512) + ReLU ─→ x_fc        (los descriptores x_i)
salida del resnet ─────┤
   (512, 7, T/16)      └─ vlad_conv (7×1, 512→10)          ─→ x_k_center (los scores w_k·x + b_k)
```

Corresponde a las dos etiquetas del diagrama de la clase: *Projection* y *Centroid Ownership*. Y el Keras original hace lo mismo (ambas `Conv2D((7,1))` parten de `x`).

{{< concept-alert type="clave" >}}
**Esto rompe la interpretación geométrica, y es el detalle más fino de la arquitectura.**

En NetVLAD «canónico», el soft assignment se calcula sobre **los mismos vectores** que se agregan, así que `w_k` y `c_k` viven en el mismo espacio y el desacople sigue siendo interpretable (`w_k ≈ 2α c_k` sería el caso geométrico).

Acá **no**: los `w_k` operan sobre la salida cruda del resnet (7×512 aplanado por el kernel), mientras los `c_k` viven en el espacio de `x_fc` (512-d, post-ReLU, no negativo). **Son espacios distintos.** Los `w_k` no son centroides reescalados de nada: son un clasificador de asignación libre, entrenado solo por la señal de la pérdida.

El diagrama de Voronoi que el notebook muestra en la celda 7 es la **metáfora pedagógica del ancestro** (VLAD 2010), no una descripción de lo que hace este código. Los `c_k` siguen siendo puntos en el espacio de descriptores, pero quién se asigna a ellos lo decide una función que ni los conoce.
{{< /concept-alert >}}

Ambos kernels son `(7, 1)`: cubren los 7 bins de frecuencia restantes y **un solo** frame temporal. Colapsan la frecuencia a 1 y no mezclan tiempo — funcionalmente, capas densas aplicadas de forma independiente a cada instante. Y ambas llevan `bias=True`, coherente con que ninguna vaya seguida de BatchNorm.

La `ReLU` de `block_1` hace que los descriptores `x_i` sean **no negativos**, lo que condiciona la geometría de los residuos `x_i − c_k`.

---

## 4. El softmax escrito a mano, y por qué la resta del máximo no es decorativa

```python
max_cluster_score, _ = cluster_score.max( dim = -1, keepdim = True )
exp_cluster_score = torch.exp( cluster_score - max_cluster_score )
A = exp_cluster_score / exp_cluster_score.sum( dim = -1, keepdim = True )
```

Tres líneas que son exactamente `F.softmax(cluster_score, dim=-1)` — verificado, error máximo **1,19×10⁻⁷**. Está escrito a mano por herencia del código Keras.

La resta del máximo es el **truco log-sum-exp**. Matemáticamente es una identidad: `exp(a−m)/Σexp(a'−m) = exp(a)·e⁻ᵐ/(e⁻ᵐ·Σexp(a')) = softmax(a)`. Numéricamente es otra cosa: `float32` desborda por encima de `3,4×10³⁸`, y `exp(89)` ya lo supera. Restar el máximo garantiza que el mayor exponente sea `exp(0) = 1`.

Medido con scores de magnitud extrema (`randn × 200`):

| | NaN producidos |
|---|---|
| sin restar el máximo | **381 de 1.200** |
| con el truco (como el lab) | **0** |

Los scores vienen de `vlad_conv` sin restricción de rango, así que la protección es pertinente. La única desventaja frente a `F.softmax` es que materializa tres tensores intermedios en lugar de usar el kernel fusionado de PyTorch.

---

## 5. El broadcast de cinco dimensiones, y cómo se hace bien

```python
A = A.unsqueeze( dim = -1 )                 # (1, 1, T/16, 10,   1)
feat_broadcast = feat.unsqueeze( dim = -2 ) # (1, 1, T/16,  1, 512)
feat_res = feat_broadcast - self.cluster    # (1, 1, T/16, 10, 512)  ← se materializa
weighted_res = torch.mul( A, feat_res )     # (1, 1, T/16, 10, 512)  ← y otro igual
```

Es la traducción **literal** de la fórmula, y didácticamente perfecta: se lee igual que la matemática. Pero materializa dos tensores de 5 dimensiones que el cálculo no necesita.

**La identidad que lo evita** — distribuir la suma, porque `c_k` no depende de `i`:

$$v_k = \sum_i a_{ik}(x_i - c_k) = \underbrace{\sum_i a_{ik} x_i}_{\text{un matmul}} - \underbrace{\Big(\sum_i a_{ik}\Big) c_k}_{\text{un producto exterior}}$$

Un producto matricial `Aᵀ·X` menos un reescalado de los centroides. **Ningún tensor de 5 dimensiones.** Es la implementación estándar y eficiente de NetVLAD.

| Descriptores | Error relativo | Tensor 5-D | Naive | Factorizada | Aceleración |
|---|---|---|---|---|---|
| 102 (audio típico) | 1,2×10⁻⁷ | 2,0 MB | 0,09 ms | 0,02 ms | **5,8×** |
| 800 | 2,3×10⁻⁷ | 15,6 MB | 0,37 ms | 0,02 ms | **14,9×** |
| 4.000 (audio largo) | 3,9×10⁻⁷ | **78,1 MB** | 3,10 ms | 0,10 ms | **32,5×** |

Numéricamente idénticas; la factorizada escala mucho mejor porque delega en BLAS. **En este lab no es el cuello de botella** (2 MB y décimas de milisegundo para el audio típico), pero es la clase de código que funciona perfecto con `batch=1` en inferencia y se vuelve insostenible en cuanto se escala a batches de entrenamiento.

---

## 6. La suma, y la línea que es todo GhostVLAD

```python
cluster_res = weighted_res.sum( dim = [1, 2] )
cluster_res = cluster_res[:, :self.k_centers, :]
```

**La primera línea es la `Σ` de la fórmula.** Colapsa `(1, 1, T/16, 10, 512)` en `(1, 10, 512)` sumando sobre frecuencia (tamaño 1) y tiempo. **Aquí desaparece el largo variable**: entren 25 descriptores o 4.000, la salida es siempre `10 × 512`.

**La segunda línea es GhostVLAD completo.** `[:, :8, :]` conserva las primeras 8 filas y **tira las 2 últimas**. Los clusters fantasma compitieron en el softmax —su masa de probabilidad salió del presupuesto de los 8 reales— y ahora sus residuos se descartan.

Un slice de 24 caracteres, que vale **0,35 puntos de EER** (3,57 % → 3,22 %) y es lo que permite al modelo prescindir de detección de actividad de voz.

El mecanismo, con precisión: si un descriptor de silencio se asigna con `ā = 0,9` a un fantasma, solo `0,1` de masa se reparte entre los 8 reales, y su contribución a `V` queda **atenuada en un 90 %**. El paper de [GhostVLAD](/papers/ghostvlad-zhong-2018) lo dice así:

> *"Este mecanismo permite a la red asignar descriptores no informativos a los clusters fantasma, disminuyendo así sus pesos de asignación hacia los clusters no fantasma."*

Y no hay supervisión de qué descartar: *"no forzamos explícitamente que las imágenes de baja calidad se asignen a los clusters fantasma, sino que [la ponderación por calidad] emerge automáticamente"*.

> **Un acoplamiento oculto:** `self.g_centers = 2` está **hardcodeado** en `VladPooling`, mientras `NetVlad` declara su propio `self.ghost_clusters = 2`. El número de fantasmas vive en dos lugares independientes. Cambiar solo uno —para reproducir la fila «NetVLAD, 3,57 %» del paper, por ejemplo— hace fallar el broadcast con un error de dimensiones. La forma limpia de hacer ese experimento sin editar dos celdas es restar 20 al bias de los fantasmas: `red.vlad_conv.bias[8:] -= 20.0` los saca de competencia (`e⁻²⁰ ≈ 2×10⁻⁹`).

---

## 7. La intra-normalización, y la invarianza que explica el espejado

```python
cluster_l2 = F.normalize( cluster_res, dim = -1, p = 2 )
```

`dim=-1` sobre `(1, 8, 512)` normaliza **cada uno de los 8 vectores de residuos por separado**. No es una normalización global del vector de 4.096: son 8 normalizaciones independientes.

Se llama **intra-normalization** y viene de Arandjelović & Zisserman 2013 (*All about VLAD*). El problema que resuelve: en VLAD crudo, un cluster que recibe muchos descriptores acumula una suma de norma grande y **domina** el vector concatenado, ahogando a los demás. Es el problema de los *burstiness* — un sonido repetitivo infla su cluster y aplasta la señal del resto. Normalizar cada cluster por separado iguala su voz: pasa a importar **la dirección** del residuo acumulado, no su magnitud.

Y tiene una consecuencia que cierra el hilo del espejado:

| | resultado |
|---|---|
| suma cruda: ‖v_duplicado‖ / ‖v_original‖ | **2,000000** — exactamente el doble |
| tras intra-normalizar: max &#124;diferencia&#124; | **2,2×10⁻⁸** — idénticos |

{{< concept-alert type="clave" >}}
**La intra-normalización hace a VLAD invariante al número de descriptores.** Duplicar el conjunto multiplica cada `v_k` por 2, y normalizar cancela el factor exactamente.

Eso explica de una vez tres mediciones del lab:
- el **espejado** de la celda 5 (2× descriptores) cambia el embedding un 0,02 %,
- el **`T/16` contra `T/32`** (2× descriptores) lo cambia un 0,07 %,
- y la duración del audio no sesga el descriptor, que es la propiedad que hace comparable un clip de 3 s con otro de 20.
{{< /concept-alert >}}

> **Lo que falta:** el VLAD canónico hace intra-normalización → concatenación → **L2 global**. Aquí no hay L2 global del vector de 4.096. El paper dice ambiguamente *"the final output is obtained by performing L2 normalisation and concatenation"*. En la práctica el efecto se recupera parcialmente con la `F.normalize` que sigue a `block_2`, pero con una `Linear` y una `ReLU` en medio no es lo mismo.

---

## 8. El cierre: ReLU + L2, y el rango que se pierde

```python
x = self.block_2_activation( self.block_2( x ) )   # 4096 → 512, + ReLU
x = F.normalize( x, dim = 1, p = 2 )               # norma 1
```

La slide 29 de la clase describe la cadena con precisión inusual:

> x = DimReduction(x) ⟹ xᵢ ∈ [−∞, ∞]
> x = ReLU(x) ⟹ xᵢ ∈ [0, ∞]
> x = L2_norm(x) ⟹ xᵢ ∈ [0, 1]

`block_2` comprime 4.096 → 512 con 2.097.664 parámetros. El paper: *"To keep computational and memory requirements low, dimensionality reduction is performed via a Fully Connected layer."* Para verificar 37.720 pares, 512-d en lugar de 4.096-d es 8× menos memoria y 8× menos multiplicaciones por comparación.

**La combinación ReLU + L2 es más consecuente de lo que parece.** La ReLU confina el embedding al **ortante positivo** de la esfera unitaria, lo que da la garantía anunciada: `score = v1·v2 ∈ [0,1]`, nunca negativo. El precio, medido:

| Embeddings de 512-d aleatorios | min | max | media |
|---|---|---|---|
| **con** ReLU (como el lab) | 0,2231 | 0,4348 | **0,3189** |
| sin ReLU | −0,1478 | 0,1510 | −0,0008 |

**El ángulo máximo entre dos embeddings es 90°, no 180°**: el modelo no puede expresar «opuesto», solo «ortogonal». Se pierde la mitad del rango angular a cambio de la interpretabilidad del rango.

Y esto genera la predicción de que el umbral de decisión no puede estar cerca de cero — que resultó **correcta en dirección y muy corta en magnitud**: los embeddings *entrenados* dan 0,647 entre voces distintas, no 0,32. Por qué, en [El EER y la dirección común](05-el-eer-y-la-direccion-comun).

> Detalle: `F.normalize(x, dim=1)` es correcto sobre `(batch, 512)`. Si dijera `dim=0` normalizaría *a través del batch*, y con `batch_size=1` **sería invisible**: cada componente saldría ±1, sin error ni excepción.

---

## 9. La fórmula, mapeada al código

| Elemento de la fórmula | Línea | Forma |
|---|---|---|
| `x_i(j)` — los descriptores | `feat` (tras el permute) | `(1, 1, T/16, 512)` |
| `c_k(j)` — los centroides | `self.cluster` | `(10, 512)` |
| `ā_k(x_i)` — el soft assignment | `A` (softmax de `cluster_score`) | `(1, 1, T/16, 10)` |
| `x_i(j) − c_k(j)` | `feat_res` | `(1, 1, T/16, 10, 512)` |
| `ā_k · (…)` | `weighted_res` | `(1, 1, T/16, 10, 512)` |
| `Σᵢ` | `.sum(dim=[1,2])` | `(1, 10, 512)` |
| *(fuera de la fórmula)* descartar fantasmas | `[:, :8, :]` | `(1, 8, 512)` |
| *(fuera de la fórmula)* intra-norm | `F.normalize(dim=-1)` | `(1, 8, 512)` |
| *(fuera de la fórmula)* concatenar | `.view(-1, 4096)` | `(1, 4096)` |

**Las tres últimas filas no están en la fórmula de la slide, y son las que hacen que esto funcione en la práctica.**

> Sobre los comentarios del código: dicen `bz x W x H x D`, pero tras el `permute(0,2,3,1)` el eje en posición «W» es la **frecuencia** (tamaño 1) y el de «H» es el **tiempo** (`T/16`) — al revés de la convención de PyTorch. No afecta la corrección, porque la línea que los consume los suma juntos, pero desorienta al leer. **Ese eje «H» es el que contiene los N descriptores de la fórmula.**

---

**Anterior:** [El Thin ResNet, la errata y el campo receptivo](02-el-thin-resnet-y-la-errata) · **Siguiente:** [El checkpoint abierto](04-el-checkpoint-abierto)
