---
title: "05 - El EER, el umbral y la dirección común"
weight: 50
math: true
---

> El resultado del lab —3,19 % de EER contra el 3,22 % publicado—, la función que lo calcula verificada contra fuerza bruta, y dos predicciones que fallaron: por qué el umbral está en 0,776 y no cerca de cero, y por qué la técnica estándar de la literatura no mejora nada acá.

---

## 1. El resultado

```
==> model : torch_weights.h5, threshold: 0.7756642103194826, EER: 0.0319194061507602
```

| | EER |
|---|---|
| **Esta ejecución** | **3,19 %** |
| [Xie et al. 2019](/papers/utterance-level-xie-2019), Thin ResNet-34 + GhostVLAD, VoxCeleb1-test | **3,22 %** |
| El mismo backbone con TAP (promedio temporal) | 10,48 % |
| NetVLAD sin ghost clusters | 3,57 % |

**Reproducido, y 0,03 puntos por debajo del paper.** Esa diferencia es ruido de implementación: la ventana Hann en lugar de la Hamming que declara el paper, el `eps` de BatchNorm que difiere entre frameworks, la aritmética de GPU. Lo relevante es que cae exactamente donde debe.

> El `eer` se imprime como **fracción**, no porcentaje: `0.0319` hay que leerlo como 3,19 % para compararlo con la tabla del paper.

---

## 2. Qué mide el EER, y qué esconde

El modelo produce un score continuo; la tarea pide una decisión binaria. Falta un umbral, y las slides 32–35 lo plantean así: *"What is the limit value to determine what is low and what is high? In other words, what is the best threshold?"*

Hay una trampa en la pregunta: **no existe «el mejor umbral» sin decidir antes qué error duele más.**

| Error | En verificación de hablante | Consecuencia |
|---|---|---|
| **Falso positivo** | aceptar a un impostor | brecha de seguridad |
| **Falso negativo** | rechazar al usuario legítimo | fricción |

El **EER** (*Equal Error Rate*) es el punto donde ambos se igualan:

$$\text{FPR} = \text{FNR} = 1 - \text{TPR}$$

Geométricamente, donde la ROC **cruza la antidiagonal**. Ese cruce siempre existe y es único para una ROC monótona, así que el EER resume el sistema en un número **sin requerir elegir un punto de operación ni asumir costos relativos**. Por eso es la métrica estándar en biometría.

### Traducir un EER a algo intuitivo

Si las dos distribuciones fueran gaussianas de igual varianza, el EER se relaciona con la **separabilidad** `d′ = (μ₊ − μ₋)/σ`:

$$\text{EER} = \Phi\!\left(-\frac{d'}{2}\right) \qquad\Longleftrightarrow\qquad d' = -2\,\Phi^{-1}(\text{EER})$$

Para el 3,19 % medido: **d′ = 3,71**. Las distribuciones de «mismo hablante» y «distinto hablante» están separadas 3,7 desviaciones estándar. Es una forma más informativa de pensar el número, y permite comparar mentalmente: pasar de TAP (10,48 %) a GhostVLAD (3,19 %) es pasar de `d′ = 2,51` a `d′ = 3,71`.

---

## 3. `calculate_eer`, verificada

```python
def calculate_eer(y, y_score):
  from scipy.optimize import brentq
  from sklearn.metrics import roc_curve
  from scipy.interpolate import interp1d
  fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
  eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
  thresh = interp1d(fpr, thresholds)(eer)
  return eer, thresh
```

**La ecuación.** Define `f(x) = 1 − x − TPR(x)` y busca su raíz: si `f(x) = 0`, entonces `TPR(x) = 1 − x`, que es exactamente la condición del EER con `x = FPR`.

**Por qué `brentq` puede funcionar.** El método de Brent exige cambio de signo:

- `f(0) = 1 − TPR(0) = 1 > 0` (umbral máximo: no se acepta a nadie)
- `f(1) = −TPR(1) = −1 < 0` (umbral mínimo: se acepta a todos)

Hay cambio de signo y `f` es continua y monótona decreciente → **la raíz existe, es única y Brent converge** (combina bisección con secante e interpolación cuadrática inversa: robustez de la bisección con convergencia superlineal).

**Por qué hace falta `interp1d`.** La ROC empírica es una **función escalonada**: con 37.720 muestras, FPR y TPR saltan de forma discreta, y la ecuación `TPR = 1 − FPR` **puede no tener solución exacta** — la curva puede saltar por encima de la antidiagonal sin tocarla. `interp1d` conecta los vértices con segmentos rectos y convierte la escalera en una función donde la raíz existe siempre. **Es lo que hace que el EER sea un número y no un intervalo.**

### La verificación

Simulando 18.860 positivos y 18.860 negativos y comparando la función contra un barrido por fuerza bruta de todos los umbrales:

| Método | EER | Umbral |
|---|---|---|
| `calculate_eer` (el del lab) | **3,2185 %** | 0,583016 |
| Fuerza bruta (buscar FPR = FNR) | **3,2185 %** | 0,583035 |
| diferencia | **0,00000 pts** | 1,9×10⁻⁵ |

En el umbral óptimo la fuerza bruta da `FPR = 3,2185 %` y `FNR = 3,2185 %` — iguales, como debe ser. **La función es correcta**, y la diferencia de 10⁻⁵ es el efecto de la interpolación lineal frente al valor discreto exacto.

> **Dos gotchas verificados.** (1) sklearn moderno devuelve `thresholds[0] = inf` y los primeros valores de `fpr` están **repetidos** (`[0., 0., 0.]`); `interp1d` con abscisas repetidas es ambiguo y devolvería `inf` si el EER cayera en ese primer tramo. No ocurre acá (el EER está en 0,032, lejísimo de FPR = 0), pero es lo que explicaría un umbral `inf` inexplicable en un sistema casi perfecto. (2) `roc_curve` usa `drop_intermediate=True` por defecto y reduce la curva de **37.721 a 1.876 puntos**. Para el EER da lo mismo; para graficar la cola de FPR bajo —la que importa en seguridad— conviene `drop_intermediate=False`.

---

## 4. El scoring: 512 multiplicaciones

```python
v1 = feats[ind1, 0]          # (512,)
v2 = feats[ind2, 0]
scores += [np.sum(v1*v2)]    # producto punto
```

**Esto es todo el modelo de decisión.** Después de 12 millones de parámetros, 34 capas convolucionales y una capa VLAD, comparar dos voces son 512 multiplicaciones y 511 sumas.

No hace falta dividir por las normas: la `F.normalize` del forward ya dejó ambos vectores con norma 1, así que `v1 · v2 = cos θ`. Y por la ReLU previa, ambos están en el ortante positivo → `cos θ ∈ [0, 1]`, la garantía que anuncia la slide 31.

**Que la comparación sea un producto punto es la propiedad que hace desplegable al sistema.** Si requiriera pasar los dos audios juntos por una red, verificar contra un millón de voces costaría un millón de forwards. Con embeddings precomputados es una multiplicación matriz-vector, y hay índices (FAISS, HNSW) que la resuelven en tiempo sublogarítmico. La arquitectura entera está diseñada alrededor de este momento — y es la misma distinción que separa un bi-encoder de un cross-encoder en recuperación de texto.

Es también lo que justifica la deduplicación de la celda 20: cada audio aparece en **16 pares** en promedio (75.440 rutas → 4.715 audios únicos), así que extraer features de los únicos en lugar de por par ahorra **16×** en la parte más costosa del lab. **El descriptor de un audio no depende de con quién se lo compara.**

---

## 5. Predicción fallida: el umbral

| Magnitud | Predicción | Real | |
|---|---|---|---|
| EER | 3,0 – 3,8 % | **3,19 %** | ✅ |
| d′ | ~3,7 | **3,71** | ✅ |
| **Umbral** | **0,45 – 0,70** | **0,7757** | ❌ |
| **Media de scores negativos** | **0,40 – 0,55** | **0,6474** | ❌ |

Las dos últimas fallaron por la misma razón. El razonamiento había sido: la ReLU confina los embeddings al ortante positivo, y **vectores no negativos aleatorios** de 512-d dan coseno ~0,32; ese debería ser el piso.

**El error: los embeddings entrenados no son aleatorios.** Los reales dan **0,6474** entre voces distintas — el doble.

### La geometría de lo que pasa realmente

Si todo embedding se descompone en una **dirección común** `u` con peso `α`, más una parte idiosincrática ortogonal:

$$v = \alpha\,u + \sqrt{1-\alpha^2}\;w_{\text{hablante}}$$

el coseno entre dos voces no relacionadas es `≈ α²`. Con `α² = 0,6474` → **α ≈ 0,805**.

**Y la medición directa lo confirma:** la norma de la media global de los 4.715 embeddings es **‖μ‖ = 0,8088** (sobre vectores de norma 1), o sea **65,4 % de energía común**.

{{< concept-alert type="clave" >}}
**El 65,4 % de la energía de cada embedding es una dirección compartida por las 4.715 grabaciones.** Solo el **34,6 %** restante distingue a los 40 hablantes. El sistema discrimina con un tercio de la norma de sus propios vectores.

Eso explica la estructura de los scores: las dos distribuciones viven **comprimidas en la mitad superior del rango**, y ninguna se acerca a 0. El umbral de 0,776 no está «alto»: está donde tiene que estar, entre 0,647 y 0,876.

| | media | desv. | min | p1 | p99 | max |
|---|---|---|---|---|---|---|
| **mismo hablante** | **0,8760** | 0,0496 | 0,6697 | 0,7533 | 0,9726 | 0,9982 |
| **distinto hablante** | **0,6474** | 0,0661 | 0,4653 | 0,5146 | 0,8017 | 0,8602 |
{{< /concept-alert >}}

---

## 6. Predicción fallida: el centrado

Si hay una dirección común que no lleva información de identidad, la técnica estándar es restarla. Es el primer paso del andamiaje **centering + LDA + PLDA** que usan los sistemas de [x-vectors](/papers/x-vectors-snyder-2018), y el lab no lo aplica.

```python
mu = E.mean(axis=0, keepdims=True)
Ec = (E - mu) / np.linalg.norm(E - mu, axis=1, keepdims=True)
```

Predicción: el EER **mejora**. Resultado:

| Transformación | EER | Δ | positivos | negativos | d′ |
|---|---|---|---|---|---|
| baseline | **3,192 %** | — | +0,8760 ± 0,0496 | +0,6474 ± 0,0661 | 3,910 |
| **centrado** | **3,266 %** | −0,074 pts | +0,6379 ± 0,1503 | **−0,0193** ± 0,1757 | **4,021** ⬆ |
| centrado + whitening diagonal | 3,600 % | −0,408 pts | — | — | — |

**Empeora.** Y el whitening empeora cinco veces más.

### La paradoja: d′ mejora y el EER empeora

Lo más interesante del experimento. El centrado **mejora la separabilidad** (d′ 3,910 → 4,021) y **empeora el error** (3,192 → 3,266 %). No es contradicción: **las dos métricas miden cosas distintas.**

- **d′ solo usa medias y varianzas.** Al centrar, la separación de medias se multiplica por 2,87 (0,2286 → 0,6572) y las desviaciones por ~3,0. El cociente mejora levemente.
- **El EER depende de dónde se cruzan las colas.** Y la fórmula `EER = Φ(−d′/2)` solo vale si ambas distribuciones son gaussianas **con la misma varianza**. Aquí las varianzas ya diferían (0,0496 vs 0,0661) y tras centrar difieren más (0,1503 vs 0,1757).

Comprobación: con d′ = 4,021 el EER *gaussiano* sería `Φ(−2,01) = 2,2 %`. El real es **3,27 %**. Ese punto de diferencia es exactamente la no-gaussianidad — colas más pesadas de lo que el segundo momento sugiere.

{{< concept-alert type="clave" >}}
**Lección metodológica: optimizar d′ —o cualquier margen basado en medias y varianzas— no es lo mismo que optimizar la tasa de error.** Es el motivo por el que en biometría se reportan EER y DCF, y no coeficientes de separabilidad.
{{< /concept-alert >}}

### ¿Y el cambio es siquiera real?

Con 18.860 pares positivos, el error estándar de un EER de ~3,2 % bajo independencia binomial es

$$\text{SE} \approx \sqrt{\frac{0{,}0319 \times 0{,}9681}{18860}} = 0{,}128 \text{ puntos}$$

| Experimento | Δ EER | en unidades de SE |
|---|---|---|
| centrado | −0,074 pts | **0,58 σ** → indistinguible de cero |
| whitening | −0,408 pts | 3,2 σ → probablemente real |

**El centrado no degrada: no hace nada.** Está dentro del ruido de estimación. Y hay un agravante que empuja en la misma dirección: **los 37.720 pares no son independientes** — cada audio aparece en 16 pares. Eso infla la varianza real por encima del binomial, así que el SE verdadero es **mayor** que 0,128 y hasta los 0,408 puntos del whitening quedan en duda. El bootstrap correcto remuestrea **audios**, no pares, precisamente por esa dependencia.

### Por qué el centrado no ayuda acá (y sí en x-vectors)

La diferencia está en **en qué espacio se entrenó el modelo**:

- Los **x-vectors** salen de una capa afín sin normalizar. Su geometría no está calibrada para coseno, y centering + PLDA es lo que la calibra.
- **Este modelo se entrena con softmax sobre embeddings ya L2-normalizados** (y el paper reporta también AM-Softmax, explícitamente angular). **El coseno ya es la métrica en la que se optimizó.** La dirección común de 65 % no es un artefacto a remover: es parte del sistema de coordenadas donde el entrenamiento colocó las clases.

Restarla saca a los embeddings del espacio donde fueron optimizados. Y el whitening es peor porque amplifica las dimensiones de baja varianza — que con [la mitad del backbone muerta](04-el-checkpoint-abierto) son muchas y son mayormente ruido.

> Nota metodológica: `μ` se estima **con los mismos audios que se evalúan**. Es una fuga pequeña (un vector estimado sobre 4.715 muestras, sin ver etiquetas), pero si el EER hubiera mejorado mucho, parte de la mejora podría venir de ahí. En un sistema real `μ` se estima sobre un conjunto de desarrollo aparte.

---

## 7. Cómo falla el sistema, y qué esconde el EER

Sobre una muestra de 4.999 pares reales (los que sobrevivieron al truncamiento de Colab, alineados al 100 % con el archivo de pares), con el umbral global de 0,7757:

| | |
|---|---|
| Falsos negativos | 48 (1,92 % de los positivos) |
| Falsos positivos | 70 (2,80 % de los negativos) |
| **Peor positivo** (mismo hablante, score mínimo) | **0,6697** |
| **Peor negativo** (distinto hablante, score máximo) | **0,8602** |
| **Zona de solape** `[0,670 , 0,860]` | contiene el **33,7 %** de todos los pares |

**Un tercio de los pares cae en el rango donde ambas clases coexisten.** El sistema no separa las distribuciones en dos montones limpios: las superpone en un tercio de su recorrido, y acierta el 96,8 % porque *dentro* de esa zona la densidad se inclina fuertemente hacia el lado correcto.

Los percentiles lo dicen mejor:

```
negativos:  p50=0,649   p90=0,735   p95=0,759   p99=0,802   p99,9=0,847
positivos:  p50=0,885   p10=0,823   p5=0,800    p1=0,753    p0,1=0,701
```

**El p99 de los negativos (0,802) es mayor que el p5 de los positivos (0,800).** El 1 % de impostores más convincentes puntúa por encima del 5 % de los pares legítimos más difíciles. Ahí vive el 3,19 %.

### Y el EER no es un punto de operación

Sobre un sistema simulado con el mismo EER:

| Punto de operación | TPR | Legítimos rechazados |
|---|---|---|
| FPR = 0,1 % | **72,79 %** | **27,21 %** |
| FPR = 1 % | 91,25 % | 8,75 % |
| FPR = 5 % | 97,85 % | 2,15 % |
| FPR = 10 % | 99,08 % | 0,92 % |
| *en el EER (3,22 %)* | 96,78 % | 3,22 % |

**El mismo sistema, con el mismo EER, rechaza al 27 % de los usuarios legítimos si se exige un FPR de 0,1 %.** Si se despliega autenticación por voz, el número que importa es el TPR al FPR tolerable, y puede ser mucho peor de lo que sugiere el EER.

Por eso el paper original de [VoxCeleb](/papers/voxceleb-nagrani-2017) no usa solo EER: reporta también la función de costo de detección `C_det = C_miss · P_miss · P_tar + C_fa · P_fa · (1 − P_tar)`, que **pondera explícitamente** los dos errores y la probabilidad a priori del target. Es la métrica de las evaluaciones NIST SRE. El lab se queda con el EER porque es el número que permite comparar con la tabla del paper — pero es un resumen, no una especificación.

Y aplicado al umbral concreto de este sistema: para operar a FPR = 0,1 % habría que subir el umbral a ~0,847 (el p99,9 de los negativos), y ahí se rechazaría bastante más del 3 % de los usuarios legítimos.

---

**Anterior:** [El checkpoint abierto](04-el-checkpoint-abierto) · **Siguiente:** [Los defectos del notebook](06-los-defectos-del-notebook)
