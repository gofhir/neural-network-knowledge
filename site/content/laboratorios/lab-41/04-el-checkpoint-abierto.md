---
title: "04 - El checkpoint abierto"
weight: 40
math: true
---

> El lab trata al checkpoint como una caja negra: se descarga, se carga, se usa. Pero son 46 MB de tensores que se pueden abrir y mirar. Y lo que hay dentro contradice al paper en cuatro puntos: la mitad del backbone está apagada, los ocho centroides «discriminativos» son el mismo vector, dos de los diez nunca recibieron gradiente, y la capa más grande del modelo tiene un número de salidas que viene de un dataset de caras.

---

## 1. Qué hay en el archivo

```python
sd = torch.load('torch_weights.h5', map_location='cpu')
# tipo: OrderedDict | claves: 236 | elementos: 12.097.520
```

**12.097.520 = 12.083.466 parámetros + 14.054 buffers de BatchNorm.** Coincide exactamente con la definición de clases del notebook, y `load_state_dict` responde `<All keys matched successfully>`: ni una clave faltante ni sobrante. La reimplementación PyTorch del lab es estructuralmente idéntica al modelo que produjo estos pesos.

Las claves que no son del backbone:

```
vlad_pooling.cluster       (10, 512)          ← los centroides
block_1.weight             (512, 512, 7, 1)
vlad_conv.weight           (10, 512, 7, 1)    ← 8 reales + 2 fantasma
block_2.weight             (512, 4096)
dense_1.weight             (8631, 512)        ← ¡8631!
```

Y un detalle revelador antes de entrar en materia: **`num_batches_tracked = 0` en las 38 BatchNorms.** Ese contador solo avanza entrenando en PyTorch. Estar en cero es la firma de una **conversión**: los `running_mean`/`running_var` se copiaron desde los `moving_mean`/`moving_variance` de Keras y el contador quedó en su valor inicial. El modelo se entrenó en TensorFlow y se portó después — lo que a su vez explica la discrepancia del `eps` que aparece al final de esta página.

> **El archivo tampoco es lo que dice su extensión.** Los primeros bytes son `PK\x03\x04` seguidos de `archive/data.pkl`: es un **ZIP de `torch.save`**, no un HDF5 (que empezaría con `\x89HDF\r\n\x1a\n`). La extensión `.h5` es un residuo del port desde Keras. No causa ningún problema —`torch.load` no mira la extensión— pero hace perder una hora a quien intente abrirlo con `h5py`.

---

## 2. El 8631: rastreo de un número que no corresponde a nada del lab

`dense_1` es una `Linear(512, 8631)`. Pero:

- La slide de la clase dice: *"Model trained on VoxCeleb2 dataset (**5994 speakers**)"*.
- [El paper](/papers/utterance-level-xie-2019) dice lo mismo: VoxCeleb2 dev tiene 5.994 hablantes.
- **8.631 es el número de identidades de entrenamiento de VGGFace2**, el dataset de *caras* del mismo grupo de Oxford.

El rastreo tiene tres eslabones verificados:

1. El paper de [GhostVLAD](/papers/ghostvlad-zhong-2018) —caras, VGGFace2— menciona el número literalmente: *"we update the network weights based on the positive class and only 20 negative classes (**instead of 8631**)"*.
2. El repo oficial `VGG-Speaker-Recognition` de Weidi Xie tiene **`num_class=8631`** como valor por defecto en la firma de `model.py`. Es un default copiado del proyecto de caras que nunca se corrigió.
3. El lab lo heredó al portar el código.

**Y el checkpoint también.** Se podía predecir antes de abrirlo, solo con el tamaño del archivo:

| Componente | Bytes |
|---|---|
| 12.083.466 parámetros × 4 B | 48.333.864 |
| 38 buffers de BatchNorm (7.008 canales × 2 + 38 contadores) | 56.368 |
| **estimado** | **48.390.232** |
| **archivo real** | **48.474.142** |
| diferencia (overhead de zip + pickle) | 83.910 B = 82 KB |

Y el contrafactual cierra la pregunta: **si el checkpoint tuviera 5.994 clases pesaría 41,0 MB. Pesa 46,2 MB.** Al abrirlo, `dense_1.weight` es efectivamente `(8631, 512)`.

El modelo carga **3,07 millones de parámetros** correspondientes a 2.637 salidas que nunca representaron a ningún hablante. Y que además nunca se ejecutan.

---

## 3. Los 4,4 millones de parámetros que nunca corren

```python
self.dense_1 = nn.Linear( self.bottleneck_dim, self.num_class, bias = False )   # 512 → 8631
self.dense_1_activation = nn.Softmax( dim = -1 )
```

Búscalos en el `forward`. **No están.** Se declaran, se inicializan, se cargan del checkpoint, se transfieren a la GPU con `.to(device)`… y jamás se ejecutan.

| | Parámetros | % del total |
|---|---|---|
| Modelo completo | 12.083.466 | 100 % |
| `dense_1` (cabeza de clasificación) | **4.419.072** | **36,6 %** |
| Usado en inferencia | 7.664.394 | 63,4 % |

{{< concept-alert type="clave" >}}
**No es un descuido: es la tesis del lab hecha código.**

La clase abre con esta secuencia: *"How can we model our speaker recognizer? We can model it as a classifier… But how can we incorporate new people? **Our model must be trained entirely for each new speaker!**"* → *"A better idea would be to calculate a descriptor"*.

La solución a esa objeción es esta arquitectura: **entrenar un clasificador de 8.631 salidas y en inferencia cortarle la cabeza.** La clasificación no es el objetivo — es el *pretexto* para que el gradiente esculpa un espacio de 512 dimensiones donde vectores del mismo hablante quedan cerca. El paper: *"Discriminative representations emerge because the entire network is trained end-to-end for speaker identification."*

Y por eso funciona con hablantes nunca vistos: los 40 de VoxCeleb1-test no están entre las clases de entrenamiento (*"VoxCeleb1 and VoxCeleb2 are completely disjoint!"*). Al usar el embedding, la identidad se representa por **posición en el espacio**, no por **índice de clase** — y el espacio generaliza. Es el mismo patrón de FaceNet, de [x-vectors](/papers/x-vectors-snyder-2018) y de cualquier sistema de verificación moderno: **clasificar para aprender, medir distancias para desplegar.**
{{< /concept-alert >}}

> Nota menor: `nn.Softmax(dim=-1)` como capa explícita sería *incorrecto* de combinar con `nn.CrossEntropyLoss` de PyTorch, que espera **logits** y aplica log-softmax internamente. Es un residuo de la traducción desde Keras, donde `Dense(activation='softmax')` + `categorical_crossentropy` sí es el idioma correcto. Como nunca se ejecuta, es inocuo.

---

## 4. Los ocho centroides colapsaron

El paper describe la capa VLAD como *"trainable discriminative clustering: every frame-level descriptor will be softly assigned to different clusters"*. Ocho centroides que particionan el espacio.

Los cosenos entre los centroides entrenados:

| | c₀ | c₁ | c₂ | c₃ | c₄ | c₅ | c₆ | c₇ |
|---|---|---|---|---|---|---|---|---|
| **c₀** | 1,000 | 0,998 | 0,998 | 0,998 | 0,998 | 0,998 | 0,998 | 0,998 |
| **c₁** | 0,998 | 1,000 | 0,998 | 0,998 | 0,999 | 0,998 | 0,998 | 0,998 |
| **c₄** | 0,998 | 0,999 | 0,998 | 0,998 | 1,000 | 0,998 | 0,999 | 0,999 |
| … | … | … | … | … | … | … | … | … |

**Coseno medio entre los 8 centroides reales: 0,9983** (mínimo 0,9977). Sus normas son 14,03 / 14,03 / 14,03 / 14,03 / 14,07 / 14,03 / 14,11 / 14,11 — prácticamente idénticas. Y la **distancia media entre ellos es 0,82** frente a normas de 14,06: están todos dentro de una bola de radio 6 % de su propia longitud.

**Los 8 centroides son, esencialmente, el mismo vector.** No hay 8 regiones de Voronoi: hay un punto y ocho perturbaciones diminutas alrededor.

### Qué implica

Volviendo a la identidad de la agregación:

$$v_k = \sum_i a_{ik}\, x_i - \Big(\sum_i a_{ik}\Big)\,c_k$$

Si todos los `c_k ≈ c`, el segundo término es casi el mismo vector escalado por la masa de cada cluster. Entonces **lo único que distingue un cluster de otro es la distribución de atención `a_ik`**: qué descriptores pesa cada `k` y con cuánta masa.

{{< concept-alert type="clave" >}}
**El «clustering discriminativo entrenable» degeneró en un mecanismo de *attention pooling* con 8 cabezas**, más un término de sesgo común a todas. Los centroides no parten el espacio; las asignaciones sí distinguen, y son 8 formas distintas de ponderar los mismos descriptores.

No es un defecto del lab ni un error de nadie: es lo que el entrenamiento encontró. Pero reencuadra el diagrama de Voronoi de la celda 7 como lo que es —la intuición del ancestro de 2010— y sugiere que los 8 clusters se comportan más como las 8 cabezas de un multi-head attention que como 8 centroides de k-means.

**El experimento que lo pondría a prueba** (cuesta una segunda pasada de extracción): reemplazar los 8 por su promedio y recalcular el EER.
```python
red = copy.deepcopy(network_eval)
with torch.no_grad():
    c = red.vlad_pooling.cluster
    c[:8] = c[:8].mean(0, keepdim=True)      # los 8 reales, todos iguales a su media
```
Si el EER apenas se mueve, la degeneración queda demostrada de forma directa.
{{< /concept-alert >}}

---

## 5. Los centroides fantasma nunca recibieron gradiente

| | norma | media | std |
|---|---|---|---|
| c₀ … c₇ (reales) | **14,03 – 14,11** | +0,207 | 0,585 |
| **c₈ (fantasma)** | **1,000** | −0,0023 | 0,0442 |
| **c₉ (fantasma)** | **1,000** | +0,0012 | 0,0442 |

Norma **exactamente 1,000**, coseno entre ellos de **−0,000** (ortogonales) y casi ortogonales a los reales (−0,032 y +0,019). Es la firma inconfundible de la **inicialización ortogonal** de Keras (`kernel_initializer='orthogonal'`, que está en el repo original) **intacta**.

**Nunca se entrenaron, porque no hay camino desde ellos hasta la pérdida:**

```python
cluster_res = cluster_res[:, :self.k_centers, :]   # ← corta las filas 8 y 9
```

Sus residuos se descartan **antes** de la loss, así que su gradiente es cero. Son **1.024 parámetros muertos por construcción**.

Verificado de la forma más brutal posible:

| Perturbación | coseno con el embedding original |
|---|---|
| `c₈`, `c₉` ← ruido × 1000 | **0,99999994** (cambia en el 8.º decimal) |
| `c₀` ← ruido × 1000 | 0,99748522 (sí cambia) |

**Y el paper de GhostVLAD había especificado que no debían existir:**

> *"esto significa que $\{a_k\}$ y $\{b_k\}$ tienen K+G elementos cada uno, mientras que **$\{c_k\}$ sigue teniendo K**."*
> — [Zhong et al. 2018](/papers/ghostvlad-zhong-2018)

Los fantasmas existen solo en la etapa de asignación: tienen pesos y sesgo para competir en el softmax, pero **no tienen posición en el espacio de descriptores**, porque nunca se calcula un residuo contra ellos. El código del lab crea 10 centroides, computa 10 residuos y descarta 2. La medición empírica confirma exactamente la especificación del paper.

### Lo que sí se entrenó de los fantasmas

| k | ‖w_k‖ | b_k | |
|---|---|---|---|
| 0–5 | 0,180 – 0,206 | −0,088 … −0,140 | reales |
| 6–7 | 0,391 / 0,402 | −0,212 / −0,216 | reales |
| **8** | **1,079** | **+0,554** | **fantasma** |
| **9** | **0,759** | **+0,470** | **fantasma** |

Los ocho reales tienen **sesgo negativo** (media −0,127); los dos fantasmas, **positivo** (media +0,512). En un softmax, un sesgo más alto significa **ganar por defecto**: cuando el descriptor no activa fuertemente ningún `w_k`, la masa se va a los fantasmas. Y su `‖w_k‖` es 3–5× mayor, o sea que reaccionan con mucha más fuerza a la entrada.

**GhostVLAD, en el modelo entrenado, no es «dos clusters más que se descartan». Es un par de compuertas aprendidas con sesgo a favor**, cuyo único trabajo es absorber masa de probabilidad que de otro modo contaminaría los 8 clusters reales. El «centroide» del fantasma es decorativo; lo que importa es su capacidad de ganar el softmax.

Y eso es exactamente el detector de actividad de voz que el modelo necesita, porque —como se mide en [el dataloader](01-el-dataloader-y-la-normalizacion)— el preprocesamiento **amplifica el silencio 37×** y el paper es explícito en que no hay VAD.

---

## 6. La mitad del backbone está apagada

Al revisar los `running_var` aparece algo que no debería estar ahí: **3.519 de 7.008 canales tienen varianza exactamente 0**. La causa está en los filtros que los alimentan:

| | valor absoluto máximo del filtro | norma del filtro |
|---|---|---|
| canales «muertos» | **~2,0×10⁻³³** | ~1,3×10⁻³² |
| canales vivos | — | 0,759 (mediana) |

**No están en cero: están encogidos 32 órdenes de magnitud.** Es la firma inequívoca del **weight decay** actuando sobre parámetros que dejaron de recibir gradiente: `w ← w(1 − λη)` aplicado miles de veces converge exponencialmente a cero sin llegar nunca. El `kernel_regularizer=l2(weight_decay)` del repo original es el responsable.

> Y explica una contradicción de medición: la norma L2 en `float32` reporta exactamente `0.0` porque `(2×10⁻³³)² = 4×10⁻⁶⁶` está muy por debajo del mínimo representable (`1,2×10⁻³⁸`). Es **underflow**, no cero. Solo se ve pasando a `float64`.

La distribución por bloque es lo más informativo:

| Bloque | canales muertos | % |
|---|---|---|
| block1 | 13 / 64 | 20,3 % |
| block2 | 74 / 480 | 15,4 % |
| block3 | 259 / 1.088 | 23,8 % |
| block4 | 631 / 1.792 | 35,2 % |
| **block5** | **2.541 / 3.584** | **70,9 %** |
| **total** | **3.518 / 7.008** | **50,2 %** |

**El 71 % de `block5` está apagado.** Y `block5` concentra el **71 % de los parámetros** del backbone (2,63 M de 3,69 M). El bloque más caro del modelo tiene siete de cada diez canales muertos.

{{< concept-alert type="clave" >}}
**El «Thin» ResNet-34 es considerablemente más delgado de lo que declara.** El paper presume de 3 M de parámetros frente a los 22 M de un ResNet-34 estándar — pero **la mitad de esos 3 M no hace nada**.

Sugiere que la arquitectura está sobredimensionada para la tarea y que un modelo con la mitad de canales en `block5` rendiría igual. Y es un recordatorio de que el «tamaño» de un modelo entrenado y su **capacidad efectiva** pueden ser cosas muy distintas.

Qué hacen esos canales en inferencia: con filtro ≈ 0, la conv emite ~10⁻³³; BN calcula `(10⁻³³ − 0)/√(0 + eps)·γ + β`, que es indistinguible de `β`. **El canal emite una constante.** Si `β < 0`, la ReLU siguiente lo mata; si `β > 0`, aporta un sesgo fijo. En ningún caso transmite información de la entrada.
{{< /concept-alert >}}

---

## 7. El eps de BatchNorm, con el desglose que importa

Una estimación inicial con varianzas **simuladas** daba 0,77 % de error relativo por la diferencia entre el `eps=1e-3` de Keras y el `eps=1e-5` de PyTorch. Con los pesos **reales**:

| | resultado |
|---|---|
| coseno( emb[eps=1e-5] , emb[eps=1e-3] ) | **0,999188** |
| error relativo medio | **3,74 %** |

Casi 5× la estimación. Y el desglose por canal explica de dónde sale:

| Grupo | Cantidad | ¿le afecta el eps? |
|---|---|---|
| canales muertos (`var = 0`, filtro ≈ 10⁻³³) | **3.518** | **No.** El numerador es ~10⁻³³: indistinguible de 0 con cualquier eps |
| canales vivos con `var ≥ 1e-3` | 3.319 | Casi nada: `eps ≪ var` |
| **canales vivos con `var < 1e-3`** | **170 (4,87 % de los vivos)** | **Sí. El eps compite con la varianza** |

El 3,74 % de error viene de **menos del 5 % de los canales vivos**, amplificado a través de 38 capas. Es un efecto real, concentrado y sistemático.

**Y es el experimento propio más interesante que queda por hacer:**

```python
import copy
red = copy.deepcopy(network_eval)
for mod in red.modules():
    if isinstance(mod, torch.nn.BatchNorm2d): mod.eps = 1e-3   # el default de Keras
# reextraer features con `red` y recalcular el EER
```

Si el EER **mejora**, los pesos venían de Keras sin compensar el cambio de default y el port tiene un sesgo corregible. Si empeora o no cambia, la conversión está bien calibrada. **Es una pregunta genuinamente abierta**: depende de cómo se hizo la conversión, algo que el checkpoint no documenta.

---

## 8. El resumen: cuatro discrepancias con el paper

| Lo que dice el paper / el código | Lo que hay en el checkpoint |
|---|---|
| *"trainable discriminative clustering"*, 8 centroides | 8 centroides con coseno **0,9983**: el mismo vector |
| `{c_k}` tiene K = 8 elementos ([GhostVLAD](/papers/ghostvlad-zhong-2018)) | 10 centroides, **2 muertos** con la inicialización intacta |
| Thin ResNet-34 de 3 M de parámetros | **50,2 %** de los canales encogidos a 10⁻³³ |
| Entrenado en VoxCeleb2, **5.994** hablantes | `dense_1` de **8.631** salidas (VGGFace2) |

Ninguna de las cuatro impide que el modelo alcance **3,19 % de EER**. Es la lección del lab: el número es correcto, y la explicación de por qué funciona —tal como la cuenta el paper— no describe a este modelo.

---

**Anterior:** [NetVLAD desarmado](03-netvlad-desarmado) · **Siguiente:** [El EER, el umbral y la dirección común](05-el-eer-y-la-direccion-comun)
