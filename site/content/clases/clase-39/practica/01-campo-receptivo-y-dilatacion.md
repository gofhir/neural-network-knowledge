---
title: "01 - Campo receptivo y dilatación"
weight: 10
math: true
---

> **Objetivo.** El slide 51 de la [Clase 39](/clases/clase-39) afirma que las convoluciones dilatadas permiten que, "tras pocas capas de profundidad, las neuronas cubran miles de timesteps". El slide 57 propone una arquitectura concreta —4 capas dilatadas con kernels $20, 10, 10, 5$— para hacerlo. Acá se verifica si esa arquitectura cumple la afirmación, midiendo el campo receptivo de tres maneras independientes.
>
> Todas las salidas de esta página son reales, producidas ejecutando el código sobre CPU con `torch 2.8.0`, `tensorflow 2.20.0` y `jax 0.4.30`.

---

## 1. La fórmula

Para una pila de $L$ capas donde la capa $l$ tiene kernel $k_l$, stride $s_l$ y dilatación $d_l$:

$$R_L = 1 + \sum_{l=1}^{L} (k_l - 1)\, d_l \prod_{i=1}^{l-1} s_i$$

En código son cinco líneas:

```python
def campo_receptivo(capas):
    """capas: lista de (k, s, d). Devuelve (R, stride_acumulado)."""
    R, S = 1, 1
    for k, s, d in capas:
        R += (k - 1) * d * S
        S *= s
    return R, S
```

Aplicada a los kernels del Ejemplo 2 con tres programas de dilatación distintos:

```python
EJEMPLO2 = [20, 10, 10, 5]

for nombre, dils in [("sin dilatar",                     [1, 1, 1, 1]),
                     ("duplicacion 1,2,4,8",             [1, 2, 4, 8]),
                     ("optimo sin huecos 1,20,200,2000", [1, 20, 200, 2000])]:
    capas = [(EJEMPLO2[i], 1, dils[i]) for i in range(4)]
    R, _ = campo_receptivo(capas)
    print(f"{nombre:34s} R = {R:6d} muestras = {1000*R/16000:8.2f} ms @16 kHz")
```

```text
sin dilatar                        R =     42 muestras =     2.62 ms @16 kHz
duplicacion 1,2,4,8                R =    106 muestras =     6.62 ms @16 kHz
optimo sin huecos 1,20,200,2000    R =  10000 muestras =   625.00 ms @16 kHz
```

{{< concept-alert type="advertencia" >}}
**Primera sorpresa: 6.6 milisegundos.** Con la progresión de dilataciones canónica —la que muestra la figura del slide 55, $1, 2, 4, 8$— las cuatro capas del Ejemplo 2 cubren **106 muestras**. Ni siquiera una ventana de análisis estándar de 25 ms. La promesa de "miles de timesteps" queda a dos órdenes de magnitud.

Con el programa $1, 20, 200, 2000$, en cambio, cubren 625 ms. La arquitectura del slide es perfectamente viable; lo que decide si sirve o no es el hiperparámetro que el slide deja sin especificar ("depende de la aplicación"). La sección 3 deriva de dónde sale ese programa.
{{< /concept-alert >}}

---

## 2. Medir el campo receptivo por gradiente

La fórmula puede estar mal aplicada. Hay una manera de medir el campo receptivo que no depende de haberla entendido bien: **propagar un impulso de gradiente**.

La idea: se construye la pila con todos los pesos en 1 (positivos, para que ninguna contribución se cancele con otra), se pasa una entrada de ceros, se toma una única posición de la salida y se le hace `backward()`. Las posiciones de la entrada que reciben gradiente no nulo son, por definición, las que influyen sobre esa salida.

```python
import torch, torch.nn as nn, numpy as np

def medir_rf_torch(kernels, dilataciones, T=40000):
    capas = []
    for k, d in zip(kernels, dilataciones):
        conv = nn.Conv1d(1, 1, k, dilation=d, bias=False)
        nn.init.constant_(conv.weight, 1.0)   # pesos positivos: sin cancelaciones
        capas.append(conv)
    red = nn.Sequential(*capas)

    x = torch.zeros(1, 1, T, requires_grad=True)
    y = red(x)
    centro = y.shape[-1] // 2
    y[0, 0, centro].backward()

    g = x.grad[0, 0].abs().numpy()
    activas = np.nonzero(g)[0]
    return len(activas), int(activas.max() - activas.min() + 1)
```

La función devuelve **dos** números, y ahí está lo interesante:

- `span` — la distancia entre la primera y la última posición que reciben gradiente. Es lo que la fórmula predice.
- `len(activas)` — cuántas posiciones **dentro** de ese tramo lo reciben efectivamente.

Si los dos coinciden, la pila consulta todas las posiciones de su campo receptivo. Si no, hay huecos.

```text
sin dilatar              span=    42 (formula     42 OK)  vistas=    42  MUERTAS=     0  cobertura=100.0%
duplicacion 1,2,4,8      span=   106 (formula    106 OK)  vistas=   106  MUERTAS=     0  cobertura=100.0%
optimo 1,20,200,2000     span= 10000 (formula  10000 OK)  vistas= 10000  MUERTAS=     0  cobertura=100.0%
excesivo 1,16,256,4096   span= 18852 (formula  18852 OK)  vistas=  8200  MUERTAS= 10652  cobertura= 43.5%
```

{{< concept-alert type="clave" >}}
**La última fila es el punto de todo el camino.** La configuración $1, 16, 256, 4096$ tiene el campo receptivo más grande de las cuatro —18.852 muestras, más de un segundo— y es la peor. De esas 18.852 posiciones, la red **solo mira 8.200**. Las otras **10.652 son invisibles**: ninguna ruta del grafo computacional las toca.

Es el artefacto de *gridding*, y explica por qué "más dilatación" no es "mejor". El campo receptivo mide hasta dónde llega la red; la cobertura mide qué fracción de eso realmente ve. La fórmula solo informa lo primero.
{{< /concept-alert >}}

---

## 3. La condición que evita los huecos

De dónde sale el programa $1, 20, 200, 2000$.

Tras la capa $l$, cada posición de salida resume un tramo contiguo de $R_l$ posiciones de la entrada original. La capa $l+1$ toma $k_{l+1}$ de esas posiciones, separadas por $d_{l+1}$. Para que los tramos resumidos por dos tomas consecutivas se toquen —y no dejen un vacío entre medio— basta con que la separación no supere el ancho de cada tramo:

$$d_{l+1} \le R_l$$

Tomar siempre el máximo permitido genera el crecimiento más rápido posible sin huecos:

```python
R, programa = 1, []
for k in [20, 10, 10, 5]:
    d = R                      # el maximo permitido
    programa.append(d)
    R = R + (k - 1) * d
```

```text
kernels      = [20, 10, 10, 5]
dilataciones = [1, 20, 200, 2000]
R final      = 10000 muestras = 625.0 ms @16 kHz
verificado: span=10000, vistas=10000, muertas=0
```

Y la misma regla, aplicada a kernel 2:

```text
Con kernel 2 la misma regla reproduce la progresion de WaveNet:
  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  ->  R = 1024
```

{{< concept-alert type="recordar" >}}
**La duplicación de WaveNet no es una convención estética: es el óptimo para kernel 2.** Cuando $k=2$, la regla $d_{l+1} = R_l$ produce exactamente $1, 2, 4, 8, \dots$ Por eso WaveNet no sufre gridding dentro de un bloque, y por eso su esquema se copió tanto.

El error está en copiarlo con kernels distintos. Con los kernels del Ejemplo 2, duplicar desperdicia dos órdenes de magnitud de cobertura: 106 muestras contra 10.000. La regla que hay que llevarse no es "duplica la dilatación", sino **"haz que la dilatación siga al campo receptivo acumulado"**.
{{< /concept-alert >}}

---

## 4. Lo mismo en TensorFlow

La medición por gradiente se traslada directamente con `GradientTape`. La única diferencia sintáctica relevante es que Keras usa `channels_last` — la entrada es $(B, T, C)$ y no $(B, C, T)$ — y que el parámetro se llama `dilation_rate`:

```python
import tensorflow as tf

def medir_rf_tf(kernels, dilataciones, T=40000):
    capas = [tf.keras.layers.Conv1D(1, k, dilation_rate=d, use_bias=False,
                                    kernel_initializer="ones")
             for k, d in zip(kernels, dilataciones)]
    red = tf.keras.Sequential(capas)

    x = tf.Variable(tf.zeros((1, T, 1)))
    with tf.GradientTape() as cinta:
        y = red(x)
        objetivo = y[0, y.shape[1] // 2, 0]

    g = cinta.gradient(objetivo, x).numpy()[0, :, 0]
    activas = np.nonzero(g)[0]
    return len(activas), int(activas.max() - activas.min() + 1)
```

```text
duplicacion 1,2,4,8      span=   106  vistas=   106  muertas=     0
optimo 1,20,200,2000     span= 10000  vistas= 10000  muertas=     0
```

---

## 5. Lo mismo en JAX

JAX no tiene capas: hay que llamar a `conv_general_dilated` directamente, lo que obliga a ser explícito sobre qué eje es cuál. Es más verboso y también más transparente — el argumento se llama `rhs_dilation` porque dilata el **operando derecho** (el kernel), y existe un `lhs_dilation` separado que dilata la entrada y sirve para implementar convoluciones transpuestas.

```python
import jax, jax.numpy as jnp

def medir_rf_jax(kernels, dilataciones, T=40000):
    pesos = [jnp.ones((k, 1, 1)) for k in kernels]

    def adelante(x):
        h = x[None, :, None]                       # (batch, T, canales)
        for w, d in zip(pesos, dilataciones):
            h = jax.lax.conv_general_dilated(
                h, w, window_strides=(1,), padding="VALID",
                rhs_dilation=(d,),                 # dilata el KERNEL
                dimension_numbers=("NWC", "WIO", "NWC"))
        return h[0, h.shape[1] // 2, 0]

    g = np.asarray(jax.grad(adelante)(jnp.zeros(T)))
    activas = np.nonzero(g)[0]
    return len(activas), int(activas.max() - activas.min() + 1)
```

```text
duplicacion 1,2,4,8      span=   106  vistas=   106  muertas=     0
optimo 1,20,200,2000     span= 10000  vistas= 10000  muertas=     0
```

Los tres frameworks dan el mismo resultado, que es lo que uno espera y conviene comprobar igual: el campo receptivo es una propiedad de la arquitectura, no de la implementación.

{{< concept-alert type="recordar" >}}
**Gotcha de JAX que vale para cualquier medición de gradiente.** `jax.grad` deriva respecto del **primer argumento** de la función por defecto, así que la función tiene que estar escrita con la entrada primero y los pesos capturados por clausura, o hay que pasar `argnums`. Y `dimension_numbers` no es opcional en la práctica: el default de JAX es el layout de imágenes, y usarlo con datos 1D produce errores de forma difíciles de leer.
{{< /concept-alert >}}

---

## 6. El costo de las tres estrategias

Ya sabemos cuántas capas necesita cada estrategia para cubrir un segundo a 16 kHz. Falta cuánto cuesta cada una. Suponiendo 64 canales por capa y kernel 3:

```python
def costo(capas, C=64, T=16000):
    params, macs, longitud = 0, 0, T
    for k, s, d in capas:
        params += k * C * C
        longitud = longitud // s
        macs += k * C * C * longitud
    return params, macs
```

```text
densa k=3 (7999 capas)    capas= 7999  R=   15999  params=  98.3M  MACs= 1572.7G  stride_final=1
dilatada k=3 exponencial  capas=   13  R=   16383  params=   0.2M  MACs=    2.6G  stride_final=1
stride/pool k=3, s=2      capas=   14  R=   32767  params=   0.2M  MACs=    0.2G  stride_final=16384
```

Tres lecturas:

**La convolución densa no es una opción.** 98 millones de parámetros y 1.5 teraMACs para ver un segundo de audio. No es que sea cara: es que no se puede.

**La dilatación es 500 veces más barata en parámetros y 600 veces en cómputo.** Y con la misma resolución de salida que la densa: el `stride_final` sigue siendo 1, así que emite una predicción por muestra de entrada. Es lo que WaveNet necesita.

**El stride es otras 13 veces más barato que la dilatación**, y ahí está el compromiso: su `stride_final` es 16.384. La salida tiene una posición por cada 16.384 de la entrada. Para un clasificador eso es exactamente lo que se quiere; para un generador de forma de onda es inservible.

{{< concept-alert type="clave" >}}
**El criterio de elección, en una línea: ¿la salida es densa en el tiempo o es una etiqueta?**

- **Densa** (generación, síntesis, separación de fuentes): no se puede submuestrear. Queda la dilatación. Es el caso de [WaveNet](/papers/wavenet-oord-2016).
- **Una etiqueta por clip** (clasificación, tagging): se puede colapsar el eje temporal entero sin pagar nada, y el stride es la herramienta más barata. Es el caso de la [familia M](/papers/raw-waveforms-dai-2017), que llega a 1.5 segundos de campo receptivo con 3.7 millones de parámetros.
- **Densa pero de menor resolución** (reconocimiento de voz, que emite unos pocos tokens por segundo): submuestrear hasta esa resolución y no más. Es lo que hacen los encoders convolucionales de wav2vec 2.0, HuBERT y [Conformer](/papers/conformer-gulati-2020) antes de entregarle la secuencia al Transformer.
{{< /concept-alert >}}

---

## Qué quedó demostrado

| Afirmación | Veredicto |
|---|---|
| "Las convoluciones dilatadas permiten campos receptivos grandes con pocas capas" (slide 55) | **Cierto.** 13 capas contra 7.999, y 500× menos parámetros |
| "El Ejemplo 2 cubre miles de timesteps" (slides 51 y 57) | **Falso con la progresión canónica**: cubre 106 muestras (6.6 ms). Cierto con $1,20,200,2000$: 10.000 muestras (625 ms) |
| "Más dilatación es mejor" (nadie lo dice, pero se asume) | **Falso.** La configuración con mayor campo receptivo tiene 43.5% de cobertura y 10.652 posiciones muertas |
| "La duplicación de dilataciones es la práctica estándar" | **Cierto solo para $k=2$**, donde coincide con el óptimo $d_{l+1} = R_l$ |

---

**Siguiente:** [02 - La CLDNN del Ejemplo 1](../02-la-cldnn-del-ejemplo-1) · **Ver también:** [Profundización, Parte I](/clases/clase-39/profundizacion) · [Convoluciones dilatadas](/fundamentos/convoluciones-dilatadas) · [WaveNet](/papers/wavenet-oord-2016) · [Familia M / Dai 2017](/papers/raw-waveforms-dai-2017).
