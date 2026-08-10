---
title: "02 - La CLDNN del Ejemplo 1"
weight: 20
math: true
---

> **Objetivo.** Construir la arquitectura del "Ejemplo 1" de la [Clase 39](/clases/clase-39) —conv → conv → reducción → LSTM → LSTM → FC → FC— en los tres frameworks, y verificar la afirmación del slide 41 de que la capa de reducción de dimensión "permite reducir parámetros sin pérdida de exactitud". El resultado es más fuerte de lo que el slide sugiere: esa capa decide si la red tiene 5 o 19 millones de parámetros.
>
> Todas las salidas de esta página son reales, producidas sobre CPU con `torch 2.8.0`, `tensorflow 2.20.0`, `jax 0.4.30` y `flax 0.8.5`.

---

## 1. La arquitectura en PyTorch

El slide especifica: entrada log-mel de 40 bandas, dos convoluciones de 256 mapas con kernels $9\times9$ y $4\times4$, max-pooling de 3 sin solape **solo en frecuencia**, una capa de reducción de dimensión, dos LSTM de 256 celdas y dos capas densas de 1.024.

Un detalle antes de escribirlo: el paper original usa $4\times3$ en la segunda convolución, no $4\times4$ — ver la [teoría, sección 5.2](/clases/clase-39/teoria). Acá se usa $4\times3$, que es lo que cabe.

```python
import torch, torch.nn as nn

T, F, N_CLASES = 20, 40, 10        # 20 tramas de contexto, 40 bandas log-mel

class CLDNN(nn.Module):
    def __init__(self, n_clases=N_CLASES, reduccion="lineal", celdas=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, (9, 9))
        self.pool  = nn.MaxPool2d((1, 3))          # SOLO en frecuencia
        self.conv2 = nn.Conv2d(256, 256, (4, 3))
        self.reduccion = reduccion
        if reduccion == "lineal":
            self.red = nn.Linear(256 * 9 * 8, 256)     # aplana TODO el bloque
            d_lstm = 256
        else:                                          # "conv1x1"
            self.red = nn.Conv2d(256, 32, 1)           # reduce mapas, preserva la grilla
            d_lstm = 32 * 9 * 8
        self.lstm   = nn.LSTM(d_lstm, celdas, num_layers=2, batch_first=True)
        self.fc1    = nn.Linear(celdas, 1024)
        self.fc2    = nn.Linear(1024, 1024)
        self.salida = nn.Linear(1024, n_clases)

    def forward(self, x):                              # x: (B, 1, T, F)
        h = torch.relu(self.conv1(x))
        h = self.pool(h)
        h = torch.relu(self.conv2(h))
        h = self.red(h.flatten(1)) if self.reduccion == "lineal" \
            else self.red(h).flatten(1)
        h, _ = self.lstm(h.unsqueeze(1))
        h = h[:, -1]
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        return self.salida(h)
```

Con `reduccion="lineal"`:

```text
salida=(4, 10)  parametros totales= 7,902,218
    conv1       :     20,992
    conv2       :    786,688
    reduccion   :  4,718,848
    lstm x2     :  1,052,672
    fc1+fc2     :  1,312,768
```

Con `reduccion="conv1x1"`:

```text
salida=(4, 10)  parametros totales= 5,288,746
    conv1       :     20,992
    conv2       :    786,688
    reduccion   :      8,224
    lstm x2     :  3,149,824
    fc1+fc2     :  1,312,768
```

---

## 2. La cuenta que justifica la capa de reducción

Y sin ninguna capa de reducción, entregándole al LSTM el bloque convolucional aplanado tal cual:

```text
Sin capa de reduccion (el LSTM come el bloque aplanado):
    LSTM x2 sobre 18432 entradas: 19,664,896
```

Diecinueve millones de parámetros, solo en el bloque recurrente. Más que toda la red convolucional que lo precede, por un factor de 24.

La razón es la fórmula del costo de un LSTM:

$$\text{params} = 4\,\big(d_{\text{in}}\, h + h^2 + h\big)$$

El término $d_{\text{in}} \cdot h$ es la matriz entrada→estado, y crece linealmente con lo que se le entregue. La salida de `conv2` es un bloque de $256 \times 9 \times 8 = 18\,432$ valores; multiplicado por 256 celdas, por 4 compuertas y por 2 capas, da los 19 millones.

| Configuración | Ancho al LSTM | Reducción | LSTM | **Total de la red** |
|---|---|---|---|---|
| Sin reducción | 18.432 | — | 19.664.896 | ~21.8M |
| Capa lineal $\to 256$ | 256 | 4.718.848 | 1.052.672 | **7.9M** |
| Conv $1\times1$, 32 mapas | 2.304 | 8.224 | 3.149.824 | **5.3M** |

{{< concept-alert type="clave" >}}
**El slide tiene razón, y por más margen del que dice.** "Reducir parámetros sin pérdida de exactitud" suena a una optimización menor. En realidad, la capa de reducción es lo que separa una red de 5 millones de parámetros de una de 22 — sobre un dataset de audio, esa diferencia decide si el modelo sobreajusta o no.

La lección transferible: **en cualquier arquitectura híbrida convolucional-recurrente, el cuello de botella de parámetros está en la interfaz entre ambas**, no en las convoluciones ni en las densas. Es el primer lugar donde mirar al hacer la contabilidad.
{{< /concept-alert >}}

{{< concept-alert type="advertencia" >}}
**Las dos reducciones no hacen lo mismo, aunque las dos ahorren.** La `conv1x1` termina siendo más barata en total (5.3M contra 7.9M), pero no son intercambiables:

- La **capa lineal** aplana la grilla tiempo × frecuencia × mapas y la proyecta a un vector de 256. El LSTM recibe un resumen global del bloque. Es lo que hace el paper de Sainath.
- La **conv $1\times1$** proyecta canal a canal y **preserva la grilla**: reduce de 256 mapas a 32, pero al aplanar quedan $9 \times 8 \times 32 = 2\,304$ valores con la estructura tiempo-frecuencia intacta.

El slide dice "convolución $1\times1$"; el paper dice capa lineal. Ambas son decisiones defendibles — solo conviene saber cuál se está tomando.
{{< /concept-alert >}}

---

## 3. Por qué el pooling va solo en frecuencia

El slide especifica *"max-pooling opcional en frecuencia solamente"*, sin decir por qué. La respuesta se ve en las formas:

```python
conv = nn.Conv2d(1, 256, (9, 9))
h = conv(torch.randn(4, 1, 20, 40))
print(nn.MaxPool2d((1, 3))(h).shape)   # solo frecuencia
print(nn.MaxPool2d((3, 3))(h).shape)   # tambien en tiempo
```

```text
tras conv 9x9                : (4, 256, 12, 32)   (B, mapas, tiempo, frecuencia)
pool (1,3) solo frecuencia   : (4, 256, 12, 10)   <- conserva las 12 tramas
pool (3,3) tambien en tiempo : (4, 256,  4, 10)   <- quedan 4 tramas para el LSTM
=> el pooling temporal destruye el 67% de la resolucion
   que el LSTM necesita como entrada.
```

Dos argumentos, y conviene tener los dos:

**El argumento arquitectónico.** La capa siguiente es un LSTM, cuyo trabajo entero es modelar la evolución temporal. Hacer pooling en tiempo antes de él es quitarle dos tercios de su entrada. En una CNN pura de clasificación el pooling temporal sería inofensivo —al final se colapsa todo igual—; en una arquitectura híbrida es destructivo.

**El argumento de la señal.** Los ejes de un espectrograma no son intercambiables. Desplazarse en tiempo preserva la etiqueta; desplazarse en frecuencia la transforma (transpone el sonido). La invarianza que se busca en frecuencia es **local**: compensar el jitter que introduce la anatomía de cada hablante sobre las formantes de un mismo fonema. Un pooling de 3 bins da exactamente eso. Se desarrolla en la [profundización, Parte III](/clases/clase-39/profundizacion).

---

## 4. TensorFlow / Keras

La misma red con el API funcional. Dos diferencias que importan: Keras usa `channels_last`, así que la entrada es $(B, T, F, 1)$ y el eje de canales va al final; y `LSTM` necesita `return_sequences=True` en todas las capas menos la última cuando se apilan.

```python
import tensorflow as tf
from tensorflow.keras import layers

def cldnn_tf(n_clases=10, celdas=256):
    entrada = tf.keras.Input(shape=(20, 40, 1))
    h = layers.Conv2D(256, (9, 9), activation="relu")(entrada)
    h = layers.MaxPooling2D((1, 3))(h)                 # SOLO en frecuencia
    h = layers.Conv2D(256, (4, 3), activation="relu")(h)
    h = layers.Flatten()(h)
    h = layers.Dense(256)(h)                           # capa de reduccion
    h = layers.Reshape((1, 256))(h)
    h = layers.LSTM(celdas, return_sequences=True)(h)
    h = layers.LSTM(celdas)(h)
    h = layers.Dense(1024, activation="relu")(h)
    h = layers.Dense(1024, activation="relu")(h)
    return tf.keras.Model(entrada, layers.Dense(n_clases)(h))
```

```text
salida=(4, 10)  parametros totales=7,900,170
    conv2d                :     20,992   (None, 12, 32, 256)
    conv2d_1              :    786,688   (None,  9,  8, 256)
    dense                 :  4,718,848   (None, 256)
    lstm                  :    525,312   (None,  1, 256)
    lstm_1                :    525,312   (None, 256)
    dense_1               :    263,168   (None, 1024)
    dense_2               :  1,049,600   (None, 1024)
    dense_3               :     10,250   (None, 10)
```

Las convoluciones y las densas coinciden exactamente con PyTorch. El total no: **7.900.170 contra 7.902.218**, una diferencia de 2.048.

{{< concept-alert type="recordar" >}}
**De dónde salen esos 2.048 parámetros de diferencia.** PyTorch parametriza el LSTM con **dos** vectores de sesgo por capa (`b_ih` y `b_hh`, uno para la transformación de la entrada y otro para la del estado); Keras usa **uno** solo. La diferencia es $2 \text{ capas} \times 4 \text{ compuertas} \times 256 \text{ celdas} = 2\,048$.

Matemáticamente son equivalentes —la suma de dos sesgos es un sesgo—, pero la duplicación de PyTorch viene de la implementación de cuDNN, que los mantiene separados. Es la razón número uno por la que dos implementaciones "idénticas" de un modelo recurrente reportan conteos de parámetros distintos, y vale conocerla antes de perder una tarde buscando el error.
{{< /concept-alert >}}

---

## 5. JAX / Flax

Flax exige declarar la estructura dentro de `@nn.compact` y separa los parámetros del módulo, lo que hace explícito algo que en los otros dos frameworks queda implícito.

```python
import jax, jax.numpy as jnp, flax.linen as fnn

class CLDNNFlax(fnn.Module):
    n_clases: int = 10
    celdas: int = 256

    @fnn.compact
    def __call__(self, x):                              # x: (B, T, F, 1)
        h = fnn.relu(fnn.Conv(256, (9, 9), padding="VALID")(x))
        h = fnn.max_pool(h, (1, 3), strides=(1, 3))     # SOLO en frecuencia
        h = fnn.relu(fnn.Conv(256, (4, 3), padding="VALID")(h))
        h = h.reshape(h.shape[0], -1)
        h = fnn.Dense(256)(h)                           # capa de reduccion
        celda = fnn.OptimizedLSTMCell(features=self.celdas)
        estado = celda.initialize_carry(jax.random.key(0), h.shape)
        for _ in range(2):
            estado, h = celda(estado, h)                # <-- cuidado aca
        h = fnn.relu(fnn.Dense(1024)(h))
        h = fnn.relu(fnn.Dense(1024)(h))
        return fnn.Dense(self.n_clases)(h)
```

```text
salida=(4, 10)  parametros totales=7,374,858
    Conv_0                :     20,992
    Conv_1                :    786,688
    Dense_0               :  4,718,848
    OptimizedLSTMCell_0   :    525,312
    Dense_1               :    263,168
    Dense_2               :  1,049,600
    Dense_3               :     10,250
```

{{< concept-alert type="advertencia" >}}
**Este modelo tiene una capa LSTM, no dos — y el bug es invisible salvo en el conteo.** El listado muestra un solo `OptimizedLSTMCell_0` con 525.312 parámetros, contra los dos `lstm` de Keras. La razón: en Flax, **instanciar un submódulo una vez y llamarlo dos veces comparte los pesos**. El bucle `for _ in range(2)` ejecuta la misma celda dos veces, lo que es un LSTM aplicado dos veces, no dos LSTM apilados.

El código corre, entrena y produce salidas del shape correcto. Lo único que delata el problema es que faltan 527.360 parámetros. Para tener dos capas independientes hay que instanciar dos celdas:

```python
celdas_lstm = [fnn.OptimizedLSTMCell(features=self.celdas) for _ in range(2)]
for celda in celdas_lstm:
    estado = celda.initialize_carry(jax.random.key(0), h.shape)
    estado, h = celda(estado, h)
```

Es el error más frecuente al portar arquitecturas apiladas a Flax, y el argumento más fuerte a favor de imprimir siempre el conteo de parámetros al traducir un modelo entre frameworks. Se deja aquí a propósito, porque encontrarlo es la lección.
{{< /concept-alert >}}

---

## 6. Resumen de la contabilidad

| Bloque | Parámetros | % del total (versión lineal) |
|---|---|---|
| conv1 $9\times9$, 256 mapas | 20.992 | 0.3% |
| conv2 $4\times3$, 256 mapas | 786.688 | 10.0% |
| **Capa de reducción** | **4.718.848** | **59.7%** |
| 2 × LSTM de 256 celdas | 1.052.672 | 13.3% |
| 2 × FC de 1.024 | 1.312.768 | 16.6% |
| Salida | 10.250 | 0.1% |

La primera convolución —la que "aprende los features locales", el bloque al que la clase dedica más justificación— es el **0.3%** de la red. Casi el 60% está en una capa que el slide menciona en una línea al pasar.

No es una crítica al diseño: la capa de reducción es cara precisamente porque está haciendo el trabajo de comprimir todo el bloque convolucional a un vector manejable, y sin ella la red costaría el triple. Es una observación sobre dónde mirar cuando hay que recortar un modelo.

---

## Qué quedó demostrado

| Afirmación del slide 41 | Veredicto |
|---|---|
| "La capa de reducción permite reducir parámetros sin pérdida de exactitud" | **Cierto, y por mucho.** Sin ella el bloque recurrente cuesta 19.7M; con ella, entre 1.1M y 3.1M |
| "Max-pooling opcional en frecuencia solamente" | **Justificado.** El pooling temporal destruiría el 67% de la resolución que el LSTM consume |
| Los tres frameworks producen la misma red | **Con dos salvedades**: los sesgos duplicados del LSTM en PyTorch (+2.048) y la compartición de pesos de Flax al reusar un módulo |

---

**Anterior:** [01 - Campo receptivo y dilatación](../01-campo-receptivo-y-dilatacion) · **Ver también:** [Profundización, Parte II](/clases/clase-39/profundizacion) · [CLDNN (Sainath 2015)](/papers/cldnn-sainath-2015) · [CRNN](/fundamentos/crnn) · [LSTM y GRU](/fundamentos/lstm-gru).
