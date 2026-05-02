---
title: "06b - Multi-Head Internals: ver los numeros reales"
weight: 65
math: true
---

En el capitulo anterior construimos multi-head attention y miramos su API. Funciona, las shapes calzan, el codigo corre. Pero quedan dos preguntas que muchos estudiantes arrastran como una sensacion de incomodidad:

- **Pregunta A**: cuando hablamos de "8 cabezas en paralelo", donde estan las 8 cabezas en el codigo? Yo solo veo tres `nn.Linear` y un `.view()`. Donde rayos estan los objetos `cabeza_1`, `cabeza_2`, ..., `cabeza_8`?
- **Pregunta B**: si dividimos el embedding en cabezas mas chiquitas, no se pierde informacion? Cada cabeza ve solo un cachito del input?

Las dos preguntas tienen la misma respuesta, y la mejor forma de verla es con numeros reales. En este capitulo vamos a poner $d_{model} = 4$, $h = 2$, $d_k = 2$ — todo chiquitisimo — y vamos a imprimir cada matriz, cada producto, cada reshape. Al final no va a quedar ninguna duda.

El script que acompana este capitulo es `clase_14/practica/03b_multi_head_internals.py`. Te recomiendo correrlo en paralelo a la lectura.

---

## 1. La pregunta A en una frase

Antes de meterse en el codigo, conviene tener clara la respuesta:

> **Las cabezas NO existen como objetos separados. Son slices conceptuales de las matrices grandes $W^Q$, $W^K$, $W^V$. El reshape "las separa" sin copiar datos.**

Cuando lees "8 cabezas", tu intuicion construye la imagen de 8 objetos distintos. Pero en el codigo no hay 8 nada. Hay tres matrices grandes y un `.view()` que las reinterpreta. Vamos a verlo.

{{< concept-alert type="clave" >}}
"Multi-head" es una forma de **mirar** las matrices, no una forma de **almacenarlas**. La misma matriz de 4x4 puede leerse como "1 cabeza con 4 dims de output" o como "2 cabezas con 2 dims cada una". El reshape implementa el cambio de lectura sin tocar los numeros.
{{< /concept-alert >}}

---

## 2. Setup tiny

Para poder seguir cada calculo a mano, vamos a usar dimensiones absurdamente chicas:

```
T = 3 tokens
d_model = 4
h = 2 cabezas
d_k = d_model / h = 2 dims por cabeza
```

Y embeddings inventados (no random) elegidos para que sea facil ver que pasa:

```python
x = torch.tensor([[
    [1.0, 0.0, 0.0, 0.0],   # token 0: vector "puro" en dim 0
    [0.0, 1.0, 0.0, 0.0],   # token 1: vector "puro" en dim 1
    [0.5, 0.5, 0.5, 0.5],   # token 2: equiparte
]])  # shape: (1, 3, 4)
```

Token 0 tiene un 1 en la primera dim y cero en las demas. Token 1 igual pero en la segunda. Token 2 reparte por igual. Esa eleccion no es magica — es solo para que cuando hagamos productos a mano, los ceros nos ahorren cuentas.

---

## 3. La matriz $W^Q$ completa

PyTorch construye una sola matriz de 4x4. Sin reshape, sin slices, sin nada raro:

```python
W_Q = nn.Linear(d_model, d_model, bias=False)
print(W_Q.weight.data)
```

Salida (con `torch.manual_seed(0)`):

```
W_Q.weight (4x4):
tensor([[-0.004,  0.268, -0.412, -0.368],
        [-0.193,  0.134, -0.010,  0.396],
        [-0.044,  0.132, -0.151, -0.098],
        [-0.478, -0.331, -0.206,  0.019]])
```

Eso es **una sola matriz** con 16 numeros. PyTorch la guarda como un tensor `(out, in)` de shape `(4, 4)`. No hay 2 matrices distintas para las 2 cabezas.

La interpretacion multi-head viene despues. Vamos a leerla asi:

- **Filas 0-1** (primera mitad de las filas): cabeza 0.
- **Filas 2-3** (segunda mitad): cabeza 1.

Esa division es **una eleccion de lectura**, no una propiedad de la matriz. La matriz no sabe nada de cabezas. Las "cabezas" emergen de como la usamos.

{{< concept-alert type="recordar" >}}
PyTorch guarda los pesos de `nn.Linear` como `(out_features, in_features)`. La operacion `y = W(x)` es `y = x @ W.weight.T`. Las "filas" de `W.weight` corresponden a las "columnas" del output conceptual. Para multi-head, **dividimos las filas** en grupos de `d_k`.
{{< /concept-alert >}}

---

## 4. Aplicar $W^Q$ al input completo

Ahora aplicamos la matriz completa al input completo, sin reshape todavia:

```python
Q = W_Q(x)            # shape: (1, 3, 4)
```

Resultado:

```
Q (sin reshape):
tensor([[[-0.004, -0.193, -0.044, -0.478],
         [ 0.268,  0.134,  0.132, -0.331],
         [-0.258, -0.166, -0.080, -0.499]]])
```

Cada fila es la "query" de un token, en 4 dimensiones. Verifiquemos manualmente el primer numero `Q[0, 0]`:

$$
Q[0, 0] = \sum_{j=0}^{3} x[0, j] \cdot W^Q[0, j]
$$

Con $x[0] = [1, 0, 0, 0]$ y $W^Q[0] = [-0.004, 0.268, -0.412, -0.368]$:

$$
Q[0, 0] = 1 \cdot (-0.004) + 0 \cdot 0.268 + 0 \cdot (-0.412) + 0 \cdot (-0.368) = -0.004
$$

```
Calculo manual: Q[0, 0] = sum_j x[0,0,j] * W_Q[0,j] = -0.0037
Valor de PyTorch: Q[0, 0] = -0.0037
Coinciden: True
```

**Nota importante**: este calculo uso **las 4 dims de $x$** y **toda la primera fila de $W^Q$** (4 numeros). No "la primera mitad" ni "la primera cabeza". Toda la fila. Recordemos esto para la pregunta B.

---

## 5. El reshape: aqui aparecen las cabezas

Ahora viene el momento clave:

```python
Q_reshaped = Q.view(1, T, h, d_k)
```

Antes y despues:

```
Q.shape antes:     (B, T, d_model) = (1, 3, 4)
Q.shape despues:   (B, T, h, d_k)  = (1, 3, 2, 2)
```

**El reshape NO copia datos**. PyTorch guarda los tensores como un bloque contiguo de memoria. `view()` solo cambia los **strides** — la regla para interpretar ese bloque. Los mismos 12 numeros que estaban ahi siguen estando ahi, en el mismo orden, en la misma RAM.

Lo que cambia es la "forma" de leerlos:

```
Q[0, 0] = [-0.004, -0.193, -0.044, -0.478]    (4 numeros, una fila)
                ↓ reshape ↓
Q_reshaped[0, 0] = [[-0.004, -0.193],          (2 cabezas con 2 numeros)
                    [-0.044, -0.478]]
                    ↑ cabeza 0      ↑ cabeza 1
```

Los primeros 2 numeros = cabeza 0. Los ultimos 2 = cabeza 1. Misma data, distinta lectura.

Despues hacemos un `transpose` para poner la dim de cabezas al frente, lo cual permite vectorizar las operaciones por cabeza:

```python
Q_heads = Q_reshaped.transpose(1, 2)   # (B, h, T, d_k)
```

```
Q_heads.shape = (1, 2, 3, 2) = (B, h, T, d_k)
```

Ahora `Q_heads[0, 0]` es "todo lo que la cabeza 0 ve" — los 3 tokens, cada uno con sus 2 dims:

```
Q_heads[0, 0] (CABEZA 0):
[[-0.004, -0.193],   <- query del token 0 (cabeza 0)
 [ 0.268,  0.134],   <- query del token 1 (cabeza 0)
 [-0.258, -0.166]]   <- query del token 2 (cabeza 0)

Q_heads[0, 1] (CABEZA 1):
[[-0.044, -0.478],
 [ 0.132, -0.331],
 [-0.080, -0.499]]
```

{{< concept-alert type="clave" >}}
El `view()` mas el `transpose()` reorganizan la presentacion del tensor sin tocar la memoria subyacente. La cabeza 0 "ve" los primeros `d_k` numeros de cada token, la cabeza 1 los siguientes `d_k`, etc. Es una vista, no una copia.
{{< /concept-alert >}}

---

## 6. Verificacion: la cabeza 0 usa solo filas 0-1 de $W^Q$

Aqui esta la prueba dura. Si las cabezas son slices de la matriz grande, deberia poder calcular el output de la cabeza 0 usando **solo las filas 0-1** de $W^Q$, y obtener exactamente el mismo resultado que el reshape.

Hagamoslo a mano:

```python
W_Q_cabeza0 = W_Q.weight[0:2, :]   # filas 0-1, shape (2, 4)
Q_cabeza0_manual = x[0] @ W_Q_cabeza0.T   # (T, 2)
```

`W_Q_cabeza0` es:

```
[[-0.004,  0.268, -0.412, -0.368],
 [-0.193,  0.134, -0.010,  0.396]]
```

Y el resultado:

```
Q_cabeza0_manual = x @ W_Q_cabeza0.T:
[[-0.004, -0.193],
 [ 0.268,  0.134],
 [-0.258, -0.166]]

Q obtenido por reshape (Q_heads[0, 0]):
[[-0.004, -0.193],
 [ 0.268,  0.134],
 [-0.258, -0.166]]

Max diferencia cabeza 0: 0.00e+00
```

**Identicos.** Cero diferencia. La cabeza 0 efectivamente usa solo las filas 0-1 de $W^Q$. Lo mismo pasa con la cabeza 1 y las filas 2-3.

Eso significa que la operacion multi-head es matematicamente equivalente a tener `h` matrices separadas de shape `(d_k, d_model)`, una por cabeza. La diferencia es puramente de implementacion: PyTorch guarda una matriz grande y la reinterpreta, en vez de guardar `h` matrices chicas. La GPU prefiere esa version porque le permite hacer un solo `matmul` en lugar de `h`.

{{< concept-alert type="recordar" >}}
"Una matriz grande con reshape" y "h matrices chicas" son **matematicamente equivalentes**. La eleccion de "una grande" es por eficiencia: una sola operacion `matmul` es mas rapida en GPU que `h` operaciones separadas.
{{< /concept-alert >}}

---

## 7. Pregunta B: ¿no se pierde informacion al dividir?

Aca llegamos a la duda profunda. Si cada cabeza tiene solo `d_k = d_model / h` dimensiones, no esta viendo "menos" del input?

**Respuesta: cada cabeza ve el $x$ completo. Lo que se "divide" es el output, no el input.**

Vuelvo al calculo del paso 6. Para producir el primer numero de Q de la cabeza 0 — el `Q_cabeza0[0, 0] = -0.0037` — usamos esto:

```
Q_cabeza0[0, 0] = x[0,0,0] * W[0,0]
                + x[0,0,1] * W[0,1]
                + x[0,0,2] * W[0,2]
                + x[0,0,3] * W[0,3]
```

Despleguemos los numeros:

```
  x[0,0,0] * W[0,0] = 1.000 * -0.004 = -0.0037
  x[0,0,1] * W[0,1] = 0.000 *  0.268 =  0.0000
  x[0,0,2] * W[0,2] = 0.000 * -0.412 =  0.0000
  x[0,0,3] * W[0,3] = 0.000 * -0.368 =  0.0000
  SUMA = -0.0037
```

**Las 4 dims de $x$ contribuyen al calculo.** En este caso particular, solo dim 0 era no-cero (porque token 0 era `[1, 0, 0, 0]`), asi que tres terminos dieron cero. Pero el calculo **lee los 4 valores** y los multiplica con los 4 pesos correspondientes. Si token 0 hubiera sido `[1, 1, 1, 1]`, los cuatro terminos habrian contribuido.

La cabeza 0 NO ve "solo las primeras 2 dims de $x$". Ve **todas las 4** y las **comprime a 2 dims** en su output. La compresion es del output (de `d_model = 4` a `d_k = 2` por cabeza), no del input.

---

## 8. Analogia: 4 fotografos con filtros distintos

Esta es la imagen mental que ayuda a fijar el concepto.

Imagina una escena: una calle con un perro, un auto y un arbol. Llegan 4 fotografos y se paran en el mismo lugar, mirando la misma escena. Pero cada uno tiene un filtro distinto en su lente:

- Fotografo 1: filtro de luz **roja**.
- Fotografo 2: filtro **infrarrojo**.
- Fotografo 3: filtro **polarizado**.
- Fotografo 4: filtro **UV**.

Los 4 toman una foto. Las 4 fotos son distintas. La del filtro infrarrojo muestra calor (el auto recien estacionado se ve brillante, el arbol oscuro). La del polarizado quita reflejos del vidrio. La UV muestra detalles invisibles al ojo.

**Los 4 vieron la escena completa.** Ninguno vio "solo un trozo". Pero cada filtro deja pasar distintas longitudes de onda, asi que cada foto resalta aspectos distintos.

Eso es multi-head. Cada cabeza:

- **Ve el $x$ completo** (todos los fotografos miran la misma escena).
- **Tiene su propio filtro** (las filas asignadas de $W^Q$, $W^K$, $W^V$, que son su "lente").
- **Produce una vista distinta** (su salida de `d_k` dims es la "foto" desde su filtro).

Distintos filtros sobre el mismo input. No hay perdida — hay **especializacion en paralelo**.

---

## 9. La matematica del "no pierde informacion"

Para sellar el argumento, comparemos parametros y dimensiones de salida entre single-head y multi-head con los mismos `d_model`:

| | Parametros totales | Output total | Distribuciones de atencion |
|---|---|---|---|
| Single-head, `d_k = 8` | $8 \times 8 = 64$ | $8$ dims | $1$ |
| Multi-head, `h = 4`, `d_k = 2` | $8 \times 8 = 64$ | $4 \times 2 = 8$ dims | $4$ |

Mismo numero de parametros. Misma dimension total de salida. La diferencia esta en **cuantas distribuciones de atencion produce el modelo**: $1$ vs $4$.

No es informacion lo que cambia — es **flexibilidad de combinacion**. Con una sola distribucion, cada token tiene una sola forma de mezclar el contexto. Con 4 distribuciones, tiene 4 formas distintas y simultaneas, cada una capturando un tipo de relacion.

{{< concept-alert type="clave" >}}
Multi-head no comprime informacion. Lo que hace es **factorizar** el espacio en `h` subespacios paralelos, cada uno con su propia atencion. Mismos parametros, mismas dimensiones totales, mas riqueza estructural.
{{< /concept-alert >}}

---

## 10. Cada cabeza con su propia atencion

Ahora veamos numericamente que cada cabeza efectivamente produce **una matriz de scores distinta**. Calculamos $K$ y $V$ con sus propias matrices, hacemos el reshape, y computamos los scores por cabeza:

```python
W_K = nn.Linear(d_model, d_model, bias=False)
W_V = nn.Linear(d_model, d_model, bias=False)

K = W_K(x).view(1, T, h, d_k).transpose(1, 2)   # (1, h, T, d_k)
V = W_V(x).view(1, T, h, d_k).transpose(1, 2)

scores = Q_heads @ K.transpose(-2, -1) / math.sqrt(d_k)
# scores shape: (1, 2, 3, 3) = (B, h, T, T)
```

Salida real:

```
Matriz de scores CABEZA 0 (3x3):
[[-0.025, -0.057, -0.059],
 [ 0.055,  0.096,  0.036],
 [-0.015, -0.006,  0.056]]

Matriz de scores CABEZA 1 (3x3):
[[ 0.109,  0.041,  0.035],
 [ 0.066,  0.035,  0.022],
 [ 0.115,  0.042,  0.037]]
```

**Numericamente distintas.** Cada cabeza tiene su propia geometria de comparacion entre tokens. Despues del softmax:

```
Weights cabeza 0:
[[0.339, 0.328, 0.327],
 [0.330, 0.344, 0.323],
 [0.328, 0.331, 0.341]]

Weights cabeza 1:
[[0.354, 0.331, 0.330],
 [0.339, 0.329, 0.327],
 [0.357, 0.328, 0.327]]
```

Dos distribuciones de pesos distintas. En este ejemplo random los numeros estan cerca de uniforme (los pesos no estan entrenados), pero la diferencia entre cabezas ya se ve. Despues de entrenar, esas diferencias se acentuan: cada cabeza aprende a atender a un tipo distinto de relacion.

---

## 11. La concatenacion final

Ya tenemos un output por cabeza. Cada uno tiene shape `(T, d_k) = (3, 2)`:

```
Output cabeza 0 (3 tokens, 2 dims):
[[-0.412,  0.142],
 [-0.401,  0.139],
 [-0.405,  0.140]]

Output cabeza 1:
[[-0.174, -0.212],
 [-0.171, -0.210],
 [-0.176, -0.213]]
```

Ahora los **concatenamos** para reconstruir un output con la dimension original `d_model = 4`:

```python
concat = head_outputs.transpose(1, 2).contiguous().view(1, T, d_model)
```

```
concat (1, 3, 4):
[[-0.412,  0.142, -0.174, -0.212],   <- cabeza0 | cabeza1
 [-0.401,  0.139, -0.171, -0.210],
 [-0.405,  0.140, -0.176, -0.213]]
```

Las primeras 2 dims son la cabeza 0, las ultimas 2 son la cabeza 1. La dimension total es la misma que la del input (`d_model = 4`), pero el contenido ahora es una **mezcla de 2 perspectivas distintas** del input.

Despues, una proyeccion final $W^O$ mezcla las cabezas (esto lo vimos en el capitulo 06). Pero ya con la concatenacion, lo importante esta hecho: el output tiene la misma forma que la entrada, listo para entrar en la siguiente capa del Transformer.

---

## 12. Pausa de verificacion

Antes de seguir, asegurate de poder responder estas preguntas con tus propias palabras:

1. **¿Las cabezas son objetos separados o slices conceptuales?**
   Slices conceptuales. PyTorch guarda una sola matriz grande $W^Q$ (4x4 en este ejemplo, 512x512 en uno real) y el reshape divide sus filas en grupos de $d_k$. La cabeza 0 = filas 0 a $d_k - 1$, cabeza 1 = filas $d_k$ a $2 d_k - 1$, etc.

2. **¿El reshape copia datos o solo reorganiza?**
   Solo reorganiza. `view()` cambia los strides del tensor (la regla para interpretar el bloque de memoria) pero no toca los numeros. Los mismos 12 numeros (en nuestro ejemplo) se leen ahora como `(B, T, h, d_k)` en vez de `(B, T, d_model)`.

3. **¿Cada cabeza ve el $x$ completo o un subset?**
   Ve el $x$ completo. La compresion sucede en el output (de `d_model` a `d_k` por cabeza), no en el input. Cada salida de la cabeza 0 es una combinacion lineal de **todas** las dims de $x$, ponderadas por una fila de $W^Q$ que tiene `d_model` numeros.

4. **¿Por que no se pierde informacion?**
   Porque mismos parametros totales y misma dimension de salida total. Lo que cambia es que en lugar de **una** distribucion de atencion, hay **`h`** distribuciones paralelas. Es factorizacion del espacio, no compresion de la senal.

---

## Codigo y siguiente capitulo

Codigo completo: `clase_14/practica/03b_multi_head_internals.py`

Volver al [hub de practica](..). Siguiente: [07 - Bloque Transformer](../07-transformer-block).
