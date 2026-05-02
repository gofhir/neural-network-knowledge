---
title: "13 - GELU vs ReLU: por que los modernos usan GELU"
weight: 130
math: true
---

En el escalon 07 (bloque Transformer) y en el mini-GPT del escalon 08 usamos `F.relu` dentro del FFN, igual que el paper original de Vaswani 2017. Funciono. Pero si abres el codigo de GPT-2, BERT, GPT-3, ViT o LLaMA, la activacion **no** es ReLU. Es **GELU** (o variantes mas modernas como SwiGLU). Por que? Que tan grande es la diferencia? Vale la pena cambiar una linea de codigo? Este escalon responde esas preguntas con un experimento controlado: dos mini-GPTs identicos salvo por la activacion del FFN, misma seed, misma data, misma cantidad de iteraciones. Comparamos.

Codigo: `clase_14/practica/10_gelu_vs_relu.py`.

---

## 1. La diferencia

Vaswani et al. (2017), en el paper original del Transformer, usaron **ReLU** en la red feed-forward de cada bloque. La definicion es la mas simple posible:

$$
\text{ReLU}(x) = \max(0, x)
$$

Si $x$ es negativo, sale cero. Si $x$ es positivo, sale $x$. Es brutalmente eficiente: una comparacion y un branch.

Los modelos modernos (BERT 2018, GPT-2 2019, GPT-3 2020, ViT 2020, T5 2020, etc.) cambiaron a **GELU** (Gaussian Error Linear Unit), introducida por Hendrycks & Gimpel en 2016:

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

donde $\Phi(x)$ es la **CDF (funcion de distribucion acumulativa) de la distribucion normal estandar** $\mathcal{N}(0, 1)$. Es decir, la probabilidad de que una variable normal $Z$ sea $\le x$.

Intuitivamente: GELU "pondera" la entrada por la probabilidad de que sea positiva bajo una normal. Para $x$ muy positivo, $\Phi(x) \to 1$, entonces $\text{GELU}(x) \to x$. Para $x$ muy negativo, $\Phi(x) \to 0$, entonces $\text{GELU}(x) \to 0$. Cerca de cero, hay una **transicion suave**.

---

## 2. Forma visual

ReLU tiene un corte abrupto en $x = 0$. Para $x < 0$ devuelve exactamente 0. Para $x \ge 0$ devuelve exactamente $x$. Es lineal a trozos.

GELU es una "ReLU suavizada". Para $x$ muy negativo se acerca a 0, para $x$ muy positivo se acerca a $x$, **pero la transicion entre los dos regimenes es suave** y, ojo con esto, para $x$ ligeramente negativo todavia hay un pequeno valor no nulo (negativo, de hecho) — la funcion no se apaga de golpe.

```
   ReLU                GELU
    |                   |
    |    /              |    /
    |   /               |   /
    |  /                |  /
    | /                 | /
----+/----              -+------
    |                  /|
                      / |
                     -  |
```

La curva de GELU "baja" ligeramente bajo cero antes de aplastarse contra el eje. Ese pequeno "valle negativo" es importante: deja pasar gradiente para entradas levemente negativas, algo que ReLU corta.

---

## 3. Por que GELU funciona mejor

Tres razones, una mecanica y dos empiricas:

1. **Diferenciable en cero.** ReLU tiene un "kink" (un punto no diferenciable) exactamente en $x = 0$. La derivada salta de 0 a 1. En la practica los frameworks definen una subderivada (tipicamente 0), pero el resto del optimizer asume que la funcion es smooth. GELU es $C^{\infty}$ — infinitamente diferenciable — en todo $\mathbb{R}$. Eso le da al optimizer un paisaje mas amigable.

2. **No "muere" en $x$ ligeramente negativo.** ReLU tiene un problema famoso conocido como **dying ReLU**: si una neurona entra en regimen $x < 0$, su gradiente es **exactamente 0**, y ya no se actualiza nunca mas. Si el sesgo se mueve mal en una iteracion, la neurona puede quedarse "apagada" para siempre. GELU deja pasar un poquito de gradiente para $x$ ligeramente negativo — la neurona puede recuperarse. Esto es mas notorio en redes profundas, donde un porcentaje no trivial de neuronas pueden quedar muertas.

3. **Mejor inductive bias para deep learning.** Empiricamente, equipos que entrenaron BERT, GPT-2, ViT y otros encontraron que GELU converge mejor y a un loss final mas bajo a gran escala. La diferencia en modelos chicos es marginal; en modelos grandes se vuelve consistente.

{{< concept-alert type="clave" >}}
La elegancia matematica de ReLU (lineal a trozos, una operacion) tiene un costo: gradiente cero para entradas negativas, y un kink no diferenciable en cero. GELU sacrifica esa simplicidad — es una funcion no lineal "real" — para ganar smoothness y evitar neuronas muertas.
{{< /concept-alert >}}

---

## 4. El experimento

El script `10_gelu_vs_relu.py` entrena dos modelos identicos en absolutamente todo, **excepto** la activacion del FFN:

- **Modelo A**: FFN con `F.relu`.
- **Modelo B**: FFN con `F.gelu`.

Configuracion compartida (la misma del escalon 08):

```
d_model:    128
h:          4
n_layers:   4
d_ff:       512
block_size: 64
batch:      32
lr:         3e-4
max_iters:  3000
```

Y la pieza critica: **misma seed**.

```python
torch.manual_seed(1337)  # misma semilla -> misma init -> comparable
```

Eso garantiza que ambos modelos arrancan con **exactamente los mismos pesos iniciales**. Las diferencias que veamos al final solo pueden venir del cambio de activacion. Es un experimento controlado en el sentido cientifico mas estricto.

---

## 5. Resultado de loss

Promedio de los ultimos 50 steps (donde el loss ya esta estable):

```
ReLU loss final (promedio ultimos 50 steps):  1.6178
GELU loss final:                              1.6018
Diferencia:                                   -0.0160 (GELU mejor)
```

**Diferencia chica pero consistente.** Aproximadamente un 1% menor en cross-entropy. En perplexity:

| activacion | loss   | perplexity $e^{\text{loss}}$ |
|------------|--------|------------------------------|
| ReLU       | 1.6178 | 5.04                         |
| GELU       | 1.6018 | 4.96                         |

GELU le saca ~0.08 caracteres efectivos de ventaja. En un mini-GPT de 0.82M parametros entrenado 3000 iteraciones, eso es lo que esperariamos.

**El punto importante**: a esta escala la diferencia es marginal. Como veremos en la seccion 7, esa misma diferencia de ~1% se amplifica conforme escalas el modelo, hasta volverse decisiva en modelos de cientos de millones de parametros.

---

## 6. Generaciones lado a lado

Para que la diferencia se vea cualitativamente, generamos 200 caracteres con cada modelo, **mismo prompt y misma seed de sampleo** (`torch.manual_seed(42)` antes de cada generacion).

### Prompt `"ROMEO:"`

ReLU:

```
ROMEO:
Troy, make of the appects, thou hast with cousint
she their kispess, un we so Caunt to lang,
For she had forgest is of the not mide oldenes expuptes.
Savy, sharm, take to the ease.
```

GELU:

```
ROMEO:
Army dear come with us shoul call a goter,
But me solding a king, my borrant to some
That have coustion! To may consolf?

JULIZHERS OF SAY:
O, I ceasing your from to-lowerd: it not leaven thou be,
Our
```

Notar lo siguiente. GELU genero `JULIZHERS OF SAY:` — claramente un nombre de personaje en mayusculas seguido de dos puntos, el formato shakespeariano que vimos aprender en el escalon 08. ReLU se atasco en `Troy, make of the appects` — un patron repetitivo, sin cambio de hablante, sin pausa estructural.

GELU mantiene mejor la estructura de **dialogo entre personajes**, una caracteristica fundamental del corpus.

### Prompt `"JULIET:"`

ReLU:

```
JULIET:
Troy, make of the appects, thou with with chusinty spoin's me ...
```

ReLU genera **el mismo patron** que con el prompt anterior (`Troy, make of the appects`). Atascado en una atractor — un modo del modelo que se repite indefinidamente.

GELU:

```
JULIET:
Troy, my father bain these lackled that the call what their king,
So there of Clurence ...
```

Mas variedad lexica. Vocabulario distinto. Aparece `Clurence` que parece una mala ortografia de `Clarence` — un personaje real de Henry VI, que existe en el corpus de Shakespeare. GELU esta accediendo a regiones del espacio de hipotesis que ReLU no.

{{< concept-alert type="recordar" >}}
La diferencia cualitativa entre ReLU y GELU en este experimento no esta en la "calidad de cada frase aislada" — ambos generan oraciones similares de calidad media. Esta en la **diversidad y estructura**: GELU mantiene mejor el formato de dialogo y accede a mas vocabulario; ReLU se atasca en patrones repetidos.
{{< /concept-alert >}}

---

## 7. Por que el efecto se amplifica con escala

A modelos chicos (como nuestros 0.82M params), la capacidad del modelo es el cuello de botella principal. La diferencia entre activaciones es marginal porque hay tanta otra cosa que limita al modelo (capacidad, datos, iteraciones) que el cambio de no-linealidad se diluye.

A modelos grandes (cientos de millones, miles de millones de parametros), las cosas cambian:

- La **capacidad** ya no es el cuello de botella.
- El **flujo de gradiente** y la **eficiencia del entrenamiento** se vuelven dominantes.
- Una mejora del 1% en la calidad del paisaje de optimizacion se acumula a lo largo de muchas mas iteraciones.
- Las **neuronas muertas** (dying ReLU) se vuelven un problema serio en redes profundas: con 96 capas (GPT-3) y miles de neuronas por capa, tener un 5-10% de neuronas apagadas es un costo real de capacidad efectiva.

> Por eso casi todos los Transformers modernos usan GELU o sus variantes (SwiGLU en LLaMA, GeGLU en T5). La pequena ventaja se acumula. A la escala de GPT-3, el 1% de cross-entropy puede equivaler a millones de dolares de compute.

---

## 8. Familia de activaciones modernas

GELU no fue el final de la historia. Surgieron variantes con la familia de las **Gated Linear Units** (GLU) que funcionan aun mejor:

| Activacion | Formula                                  | Donde se usa                  |
|------------|------------------------------------------|-------------------------------|
| ReLU       | $\max(0, x)$                             | Vaswani 2017, modelos clasicos |
| **GELU**   | $x \cdot \Phi(x)$                        | BERT, GPT-2/3, ViT            |
| **SwiGLU** | $\text{Swish}(xW) \odot xV$              | LLaMA, PaLM                   |
| **GeGLU**  | $\text{GELU}(xW) \odot xV$               | T5, GLU variants              |

Las GLU tienen una idea distinta: en vez de aplicar la activacion a una sola proyeccion lineal, aplican **dos proyecciones** $xW$ y $xV$, pasan una por la activacion (Swish o GELU), y multiplican elemento a elemento ($\odot$) por la otra. La proyeccion $xV$ actua como una **compuerta** (gate) aprendida que decide cuanto deja pasar de la activacion.

Trade-off: las GLU tienen **mas parametros** (dos matrices $W, V$ en lugar de una) por lo cual son mas caras computacionalmente. Pero ese costo extra "rinde" mas calidad por parametro adicional que simplemente ensanchar la FFN.

LLaMA (2023) es el ejemplo mas conocido de SwiGLU en produccion. PaLM (Google, 2022) tambien lo usa. T5 uso GeGLU. Mixtral, Mistral y derivados igual.

---

## 9. Pausa de verificacion

Antes de pasar al siguiente escalon, asegurate de que estos puntos te quedaron claros:

1. **Por que ReLU "mata" gradientes para $x < 0$?** Porque la derivada $\frac{d}{dx}\text{ReLU}(x) = 0$ exactamente para $x < 0$. Si una neurona queda en regimen negativo durante muchos batches consecutivos, el optimizer no recibe senal para sacarla de ahi y queda permanentemente apagada.

2. **Por que GELU evita ese problema?** Porque su derivada nunca es exactamente cero. Para $x$ ligeramente negativo, GELU produce un valor pequeno pero no nulo, y su derivada es positiva pequena. Eso permite que la senal de gradiente fluya y que la neurona pueda "despertar" si los datos lo justifican.

3. **Por que la diferencia se amplifica con escala?** Porque a escala chica el modelo esta limitado por capacidad/datos, no por flujo de gradiente. A escala grande la dinamica de entrenamiento (cuanta senal de gradiente fluye, cuantas neuronas estan vivas) se vuelve el cuello de botella, y mejoras del 1% en eficiencia compuestas a lo largo de millones de iteraciones se vuelven decisivas.

---

## 10. Conexion con el codigo del mini-GPT

En el escalon 07 definimos el bloque Transformer con un FFN asi:

```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.ReLU(),
    nn.Linear(d_ff, d_model),
)
```

Para convertirlo al estilo "moderno" basta una linea:

```python
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.GELU(),                  # <-- aqui
    nn.Linear(d_ff, d_model),
)
```

Eso es exactamente lo que hace GPT-2 en `huggingface/transformers`. Una linea de codigo, ~1% de mejora a esta escala, mas a escalas grandes. El costo computacional extra de GELU (calcular $\Phi$ involucra una funcion error o una aproximacion con `tanh`) es despreciable comparado con el resto del forward pass.

{{< concept-alert type="clave" >}}
Una de las leyes empiricas de los Transformers modernos: **los detalles pequenos importan a escala grande**. ReLU vs GELU, LayerNorm vs RMSNorm, positional embeddings aprendidos vs RoPE, attention densa vs GQA — cada cambio aporta uno o dos puntos porcentuales. Acumulados, son la diferencia entre GPT-2 (2019) y LLaMA-3 (2024).
{{< /concept-alert >}}

---

Codigo: `clase_14/practica/10_gelu_vs_relu.py`

Siguiente: [14 - Top-k sampling](../14-topk-sampling).
