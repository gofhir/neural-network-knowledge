---
title: "17 - SwiGLU: la FFN con gating"
weight: 170
math: true
---

En el escalon 16 reemplazamos LayerNorm por RMSNorm — un cambio de normalizacion que ahorra parametros y FLOPs sin perder calidad. Ese fue el primer paso de la transicion de mini-GPT a mini-LLaMA. Este escalon ataca la **segunda** modernizacion: el bloque feed-forward (FFN). Vaswani 2017 lo definio como dos capas lineales con una ReLU en el medio. BERT y GPT-2 cambiaron la ReLU por GELU (escalon 13) y dejaron todo lo demas igual. LLaMA, PaLM y la familia moderna dieron un paso mas: cambiaron la **estructura** misma del FFN, introduciendo un mecanismo de **gating** — la activacion ya no es una funcion fija coordenada-a-coordenada, sino una compuerta aprendida que decide cuanto de cada feature deja pasar. El resultado se llama **SwiGLU** y es, junto con RMSNorm y RoPE, la firma arquitectonica de la era LLaMA.

---

## 1. El problema sutil de la FFN tradicional

Recordemos la FFN del Transformer original:

$$
\text{FFN}(x) = \text{Linear}_2(\text{ReLU}(\text{Linear}_1(x)))
$$

Tres operaciones secuenciales: proyectar a una dimension mas grande ($d_{\text{model}} \to d_{\text{ff}}$, con $d_{\text{ff}} = 4 \cdot d_{\text{model}}$), aplicar una no-linealidad, y proyectar de vuelta ($d_{\text{ff}} \to d_{\text{model}}$). En esa estructura, **la unica forma que tiene el modelo de "decidir que pasa"** es la activacion. Y la activacion (ReLU, GELU, lo que sea) se aplica **a cada coordenada de manera independiente**, sin que la red pueda razonar sobre que features dejar pasar y cuales suprimir como funcion del propio input.

> Es un filtro coordenada-a-coordenada: "para esta neurona, ¿el numero es positivo? si -> deja pasar; no -> apaga". Aplicado feature por feature, sin contexto.

En el escalon 13 vimos que cambiar ReLU por GELU mejora el flujo de gradiente — es una transicion mas suave y deja pasar un poquito de senal para entradas levemente negativas. Pero la **estructura** sigue siendo la misma: una activacion fija por coordenada. El modelo no aprende **que** filtrar; solo aprende los pesos que entran y salen del filtro.

La pregunta natural es: ¿y si el filtro mismo fuera **aprendido y dependiente del input**?

---

## 2. La idea de "gating": dos caminos paralelos

La respuesta moderna se llama **gating**. En vez de UN camino con activacion fija, el FFN tiene DOS caminos paralelos: uno que produce **valores** y otro que produce **compuertas** (gates). La compuerta decide, coordenada por coordenada, cuanto de cada valor deja pasar.

```
                ┌─→ W_value(x) ────→ valor   ─┐
                │                              ├──→ producto elem-a-elem ─→ output
   x ───────────┤                              │
                │                              │
                └─→ W_gate(x) ─→ activacion ─→ ┘
                                  (gate)
```

- **Camino 1 (valor)**: que informacion quiero potencialmente transmitir.
- **Camino 2 (gate)**: cuanto de esa informacion realmente transmitir, coordenada por coordenada.

El producto elemento-a-elemento ($\odot$) combina los dos. La gracia es que el gate **es funcion del mismo input** $x$ — el modelo aprende a abrir o cerrar canales segun el contenido del token, no segun una regla fija.

Esta tecnica se llama **GLU (Gated Linear Unit)** y la introdujo Dauphin et al. en 2017 en el contexto de modelos convolucionales para lenguaje (paper "Language Modeling with Gated Convolutional Networks"). En su forma original, el gate usaba una sigmoide, que produce valores en $[0, 1]$ — literalmente "cuanto dejo pasar, entre 0% y 100%".

{{< concept-alert type="clave" >}}
La activacion tradicional decide pasa/no-pasa **por coordenada**, con una regla fija. El gating decide cuanto pasa **por coordenada**, con una regla **aprendida y condicionada al input**. Es la diferencia entre un filtro fijo y un filtro adaptativo.
{{< /concept-alert >}}

---

## 3. La evolucion historica

La cadena completa, en orden cronologico:

| Ano    | Mecanismo  | Activacion del gate / FFN                     | Donde aparecio      |
|--------|------------|-----------------------------------------------|---------------------|
| 2017   | ReLU FFN   | $\max(0, x)$ — fija, sin gate                 | Vaswani 2017        |
| 2018   | GELU FFN   | $x \cdot \Phi(x)$ — fija, sin gate, mas suave | BERT, GPT-2         |
| 2017   | GLU        | sigmoide en el gate (0 a 1)                   | Dauphin 2017        |
| 2020   | GeGLU      | GELU en el gate                               | Shazeer 2020 (T5)   |
| 2020   | **SwiGLU** | **Swish** en el gate                          | Shazeer 2020 / LLaMA|

Cada paso es una mejora marginal pero acumulativa. El paper clave es Shazeer 2020, "GLU Variants Improve Transformer", que comparo sistematicamente las variantes en T5-base y mostro que las versiones con gating (GLU, GeGLU, SwiGLU) ganan consistentemente a las versiones sin gating (ReLU, GELU). De entre las variantes con gating, **SwiGLU** quedo como la favorita: es la que adopto LLaMA en 2023, y de ahi se replico en Mistral, Mixtral, Qwen, DeepSeek, PaLM y casi toda la familia open-weight moderna.

---

## 4. Que es Swish

Swish (tambien llamada **SiLU**, Sigmoid Linear Unit) se define como:

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

donde $\sigma(x) = 1 / (1 + e^{-x})$ es la sigmoide clasica.

Comportamiento:

- Para $x$ muy negativo: $\sigma(x) \to 0$, entonces $\text{Swish}(x) \to 0$ (parecido a ReLU/GELU).
- Para $x$ muy positivo: $\sigma(x) \to 1$, entonces $\text{Swish}(x) \to x$ (igual que ReLU/GELU en el limite).
- En el medio: una transicion **suave** y **no monotonica** — la funcion baja un poquito por debajo de cero antes de subir, formando un valle pequeno cerca de $x \approx -1$.

```
   Swish(x)
       |
       |        /
       |       /
       |      /
       |     /
   ----+----·----
       |   /
       |  /
       | -      (valle pequeno bajo cero)
```

La parte clave: **deja pasar un poquito de gradiente para inputs ligeramente negativos**, igual que GELU. Pero ademas, Swish es ligeramente mas eficiente de computar (solo necesita una sigmoide, mientras que GELU exacta necesita la funcion error o una aproximacion con tanh). En la practica, SiLU/Swish y GELU dan resultados muy similares; la eleccion entre una y otra es mas convencion que ciencia dura.

> Que Swish sea no-monotonica es una **virtud**, no un defecto. La no-monotonicidad le da al gate la posibilidad de aprender filtros mas ricos: no solo "abrir" o "cerrar", sino tambien "invertir levemente" en regiones especificas.

---

## 5. La formula completa de SwiGLU

Combinando el esquema GLU con Swish en el gate, la formula completa del bloque SwiGLU es:

$$
\text{SwiGLU}(x) = W_{\text{down}} \big( \text{Swish}(W_{\text{gate}} \, x) \;\odot\; W_{\text{up}} \, x \big)
$$

donde $\odot$ es el producto elemento-a-elemento (Hadamard).

Tres matrices (vs dos de la FFN tradicional):

- $W_{\text{gate}}$: $d_{\text{model}} \to d_{\text{ff}}$ — produce el controlador.
- $W_{\text{up}}$: $d_{\text{model}} \to d_{\text{ff}}$ — produce los valores.
- $W_{\text{down}}$: $d_{\text{ff}} \to d_{\text{model}}$ — proyeccion final.

Comparada con FFN tradicional:

$$
\text{FFN}(x) = W_2 \big( \text{ReLU}(W_1 \, x) \big) \quad\text{vs}\quad \text{SwiGLU}(x) = W_{\text{down}} \big( \text{Swish}(W_{\text{gate}} \, x) \odot W_{\text{up}} \, x \big)
$$

La diferencia estructural se ve clara: SwiGLU rompe la cadena lineal "proyectar -> activar -> proyectar" en una **bifurcacion** "proyectar a dos caminos -> activar uno -> multiplicar -> proyectar de vuelta".

---

## 6. El trade-off: 3 matrices, menos $d_{\text{ff}}$

A primera vista SwiGLU parece mas caro: tres matrices en vez de dos. Si dejaramos $d_{\text{ff}} = 4 \cdot d_{\text{model}}$ como en la FFN tradicional, **la cantidad de parametros aumentaria 50%** (de 2 a 3 matrices del mismo tamano). Eso es inaceptable: estariamos comparando peras con manzanas.

LLaMA resuelve el trade-off compensando: **reduce $d_{\text{ff}}$** a $\frac{8}{3} \cdot d_{\text{model}}$ (aproximadamente $2.67 \cdot d_{\text{model}}$, en lugar de $4 \cdot d_{\text{model}}$).

Hagamos las cuentas. Tomemos $d = d_{\text{model}}$ y comparemos parametros (ignorando bias, que LLaMA omite igual):

**FFN tradicional** con $d_{\text{ff}} = 4d$:

$$
\underbrace{d \cdot 4d}_{W_1} + \underbrace{4d \cdot d}_{W_2} = 4d^2 + 4d^2 = 8 d^2
$$

**SwiGLU** con $d_{\text{ff}} = \frac{8}{3}d$:

$$
\underbrace{d \cdot \tfrac{8}{3}d}_{W_{\text{gate}}} + \underbrace{d \cdot \tfrac{8}{3}d}_{W_{\text{up}}} + \underbrace{\tfrac{8}{3}d \cdot d}_{W_{\text{down}}} = 3 \cdot \tfrac{8}{3} d^2 = 8 d^2
$$

**Mismos parametros, mismo presupuesto de FLOPs, mejor calidad.** La eleccion de $\frac{8}{3}$ no es magica: es justo el factor que iguala los conteos para que la comparacion sea limpia. En la practica, LLaMA redondea a multiplos de 256 para que las dimensiones jueguen bien con los kernels de GPU, asi que el numero exacto puede variar (por ejemplo, LLaMA-7B usa $d_{\text{ff}} = 11008$ con $d_{\text{model}} = 4096$, que es $\sim 2.69 \cdot d_{\text{model}}$, muy cerca de $8/3 \approx 2.667$).

{{< concept-alert type="clave" >}}
SwiGLU no es "agregar mas parametros y ganar calidad". Es "redistribuir el mismo presupuesto de parametros entre tres matrices mas pequenas, con una estructura de gating, y ganar calidad gratis". El truco esta en la **estructura**, no en la **escala**.
{{< /concept-alert >}}

---

## 7. Por que funciona mejor (intuicion)

Una manera de entender SwiGLU es pensar en lo que **puede** representar que la FFN tradicional no:

**FFN tradicional**: "transforma cada token via $W_1$, aplica una funcion no-lineal coordenada-a-coordenada, transforma de vuelta via $W_2$". El espacio de funciones que puede aprender es: combinaciones lineales de coordenadas, pasadas por una activacion fija, mezcladas linealmente de nuevo.

**SwiGLU**: "transforma cada token de DOS maneras simultaneas (gate y value), multiplica las dos coordenada a coordenada, y proyecta. La parte 'gate' aprende **que features son relevantes para este input**; la parte 'value' aprende **que info transmitir si las features fueran relevantes**". Multiplicar dos proyecciones lineales del mismo input introduce **terminos cuadraticos** en $x$ — el modelo gana expresividad de segundo orden gratis.

> Es una forma de **auto-atencion dentro del FFN**, aplicada al canal de features en vez de al canal de posiciones. La compuerta "atiende" a las coordenadas que importan en este input particular.

Otro angulo: en el limite donde el gate produce ~1 para todas las coordenadas, SwiGLU degenera en una FFN lineal (sin no-linealidad). En el limite donde el gate produce ~0, apaga todo. En el medio — que es donde vive en la practica — el gate aprende un **patron de saliencia** sobre las coordenadas del espacio intermedio, modulando la informacion que viaja al $W_{\text{down}}$.

Empiricamente, esa flexibilidad extra se traduce en ~3% de mejora consistente en perplejidad a igualdad de parametros.

---

## 8. Implementacion

En PyTorch, SwiGLU se escribe en pocas lineas. Esta es la version que usa el `mini_llama.py` del codigo de la clase:

```python
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)  # W_gate
        self.up   = nn.Linear(d_model, d_ff, bias=False)  # W_up
        self.down = nn.Linear(d_ff, d_model, bias=False)  # W_down

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

Tres detalles a notar:

1. **`F.silu` es la implementacion de Swish/SiLU en PyTorch**. Son sinonimos; PyTorch decidio usar el nombre SiLU.
2. **`bias=False` en las tres capas**. Es otra convencion de LLaMA: omitir biases en las capas lineales del modelo (en attention y FFN). Los biases agregan parametros que no aportan mucho una vez que el modelo es suficientemente grande, y las normalizaciones (RMSNorm) ya cumplen el rol de centrado. La maxima del estilo LLaMA es **simpler is better**.
3. **El producto `*` en PyTorch sobre tensores del mismo shape es elemento-a-elemento** — exactamente el $\odot$ que necesitamos.

Comparado con la FFN tradicional:

```python
class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
```

Tres lineas de diferencia: una matriz extra, una multiplicacion extra, y la activacion movida al gate.

---

## 9. Resultados empiricos (Shazeer 2020)

Shazeer 2020 entreno T5-base (un encoder-decoder de ~220M parametros) sobre el corpus C4 con distintas variantes de FFN, todas a igualdad de parametros (ajustando $d_{\text{ff}}$ como mostramos en la seccion 6). Los resultados de **eval loss** sobre el conjunto de validacion:

| Variante       | $d_{\text{ff}}$       | Eval loss |
|----------------|----------------------|-----------|
| FFN ReLU       | $4 d$                | 1.997     |
| FFN GELU       | $4 d$                | 1.983     |
| FFN Swish      | $4 d$                | 1.994     |
| GLU (sigmoide) | $\tfrac{8}{3} d$     | 1.953     |
| GeGLU          | $\tfrac{8}{3} d$     | 1.942     |
| **SwiGLU**     | $\tfrac{8}{3} d$     | **1.944** |

Las variantes con gating (GLU, GeGLU, SwiGLU) baten consistentemente a las sin gating (~3% relativo mejor en cross-entropy). Entre las variantes con gating, GeGLU y SwiGLU estan empatadas dentro del ruido — Shazeer no concluye que una sea mejor que la otra, pero el campo termino adoptando SwiGLU mas que GeGLU, en parte por inercia post-LLaMA.

A escala, esos ~3% de mejor cross-entropy se traducen en perplejidad significativamente mejor y, mas importante, en mejores benchmarks downstream. Por eso la lista de modelos modernos que usan SwiGLU es practicamente todo el campo open-weight: **LLaMA-1/2/3, Mistral-7B, Mixtral 8x7B, Qwen-2/3, DeepSeek-V2/V3, Yi, Gemma**. Modelos cerrados como PaLM, Gemini y (presumiblemente) Claude tambien lo usan, aunque los detalles exactos no son siempre publicos.

---

## 10. El "click"

```
ReLU/GELU FFN:    proyectar -> filtrar coord-a-coord (regla fija) -> proyectar
GLU/SwiGLU FFN:   proyectar a (gate, value) -> Swish(gate) ⊙ value -> proyectar
                                  ↑
                                  filtro condicionado: el modelo decide
                                  cuanto pasa de cada coordenada
                                  como funcion del input
```

El gate aprende un **patron adaptativo de seleccion** sobre las coordenadas del espacio intermedio. La FFN tradicional solo puede activar/desactivar coordenadas con una regla universal (la activacion); SwiGLU puede activar/desactivar coordenadas con una regla **especifica del input**.

Y todo eso sin agregar parametros, ajustando $d_{\text{ff}}$ de $4d$ a $\frac{8}{3}d$. Es un truco arquitectonico **elegante** en el sentido fuerte: mas expresividad, mismo presupuesto, una linea de codigo de diferencia. Eso es lo que lo hizo la opcion canonica de la era LLaMA.

{{< concept-alert type="recordar" >}}
Las modernizaciones de la familia LLaMA (RMSNorm, SwiGLU, RoPE, GQA) comparten un patron: **redistribuir parametros y operaciones para ganar calidad sin agregar costo**. No son "mas grande es mejor"; son "mejor estructurado a igual tamano es mejor". Es ingenieria fina sobre el Transformer original.
{{< /concept-alert >}}

---

## 11. Pausa de verificacion

Antes de pasar al siguiente escalon, asegurate de poder responder estas preguntas con confianza:

1. **¿Que hace el "gate" en SwiGLU que la activacion tradicional (ReLU/GELU) no hace?** El gate es una proyeccion lineal aprendida del input $x$, pasada por Swish. Multiplica elemento-a-elemento la otra proyeccion (`up`), modulando coordenada por coordenada cuanto se transmite. La activacion tradicional aplica una funcion fija (ReLU, GELU) coordenada por coordenada, sin conocer mas del input que el valor de esa coordenada en el espacio intermedio. El gate, en cambio, decide la modulacion como **funcion del input completo** — el modelo aprende **que** filtrar, no solo aplica un filtro fijo.

2. **¿Por que LLaMA usa $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$ en vez de $4 \cdot d_{\text{model}}$?** Porque SwiGLU usa **tres** matrices en vez de dos. Si mantuvieramos $d_{\text{ff}} = 4d$, tendriamos un 50% mas de parametros que la FFN tradicional, y no seria una comparacion justa. Reducir $d_{\text{ff}}$ a $\frac{8}{3} d$ iguala el conteo de parametros (8d² en ambas variantes) y aisla el efecto del gating como variable de interes.

3. **¿Por que Swish es preferido sobre la sigmoide pura (que era el gate original de GLU)?** La sigmoide sola produce valores en $[0, 1]$ pero **satura**: para $|x|$ grande, su derivada se hace muy chica y el gradiente se desvanece. Swish ($x \cdot \sigma(x)$) tiene la misma forma asintotica que ReLU/GELU para $|x|$ grande (lineal positiva, casi cero negativa), pero es suave y deja pasar gradiente bien en el regimen de transicion. Empiricamente, Shazeer 2020 mostro que GeGLU y SwiGLU (gates con activaciones tipo Swish/GELU) baten a GLU clasico (gate con sigmoide) por un margen pequeno pero consistente.

---

Codigo: `clase_14/practica/13_mini_llama.py` (clase `SwiGLU`).

Siguiente: [18 - RoPE](../18-rope).
