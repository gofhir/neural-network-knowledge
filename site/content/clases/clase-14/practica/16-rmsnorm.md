---
title: "16 - RMSNorm: el primer paso hacia LLaMA"
weight: 160
math: true
---

Hasta aqui tu mini-GPT es, esencialmente, **el Transformer de Vaswani et al. 2017** con algunos retoques minimos: GELU en vez de ReLU (escalon 13), top-k sampling para generacion (escalon 14), un poco mas de capacidad (escalon 11). Pero el **esqueleto arquitectonico** sigue siendo el del paper original: multi-head attention densa, FFN con una activacion entre dos lineales, **LayerNorm** entre bloques, positional embeddings aprendidos.

Eso fue el estado del arte en 2017. Hoy no lo es. Los LLMs modernos — LLaMA, Mistral, Falcon, Qwen, DeepSeek, Gemma — comparten un conjunto de **5 modernizaciones** que se acumularon entre 2018 y 2023. Cada una es individualmente sutil. **Acumuladas, son la diferencia entre texto vagamente gramatical y asistentes utilizables.**

En este escalon arrancamos con la primera y mas pequena: cambiar **LayerNorm** por **RMSNorm**.

Codigo: `clase_14/practica/13_mini_llama.py` (clase `RMSNorm`, lineas iniciales del archivo).

---

## 1. La fase 5: modernizar el Transformer hacia LLaMA

Esta fase del curso son **5 escalones de modernizacion incremental**. Cada uno cambia exactamente **un componente** del mini-GPT, manteniendo todo lo demas igual, para que puedas ver de donde viene cada mejora.

| Fase | Componente             | Reemplaza                                      |
|------|------------------------|------------------------------------------------|
| 16   | **RMSNorm**            | LayerNorm                                      |
| 17   | **SwiGLU**             | FFN con ReLU/GELU                              |
| 18   | **RoPE**               | Positional embeddings aprendidos               |
| 19   | **GQA**                | Multi-head attention estandar                  |
| 20   | **KV-cache**           | Sampling naive (regenerar todo el contexto)    |
| 21   | **Mini-LLaMA**         | Todo junto                                     |

Cada modernizacion responde a una observacion empirica concreta del campo entre 2019 y 2023. Ninguna es revolucionaria por si sola. Pero el efecto compuesto es enorme: **diferencia entre Vaswani 2017 (texto vagamente gramatical) y LLaMA 2024 (asistentes utilizables)**.

> Es la misma logica del escalon 13 (GELU vs ReLU): los detalles pequenos importan a escala grande. Cada cambio aporta uno o dos puntos porcentuales. Acumulados, son la diferencia generacional.

Empezamos por el cambio mas chico — apenas un par de lineas de codigo — para que el patron quede claro: **identificar el componente clasico, entender por que se quedo corto, presentar el reemplazo, ver por que funciona mejor**.

---

## 2. Recordatorio: que hace LayerNorm

LayerNorm aparece en cada bloque Transformer dos veces: una antes de attention, otra antes del FFN (en el patron "pre-norm" de GPT-2 y posteriores). Su trabajo es **estabilizar las activaciones** durante el entrenamiento — evitar que la magnitud de los vectores explote o se desvanezca a medida que pasan por capas profundas.

La formula clasica (Ba et al. 2016):

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

donde:

- $\mu = \frac{1}{d}\sum_i x_i$ es la **media** sobre las $d$ dimensiones del token.
- $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$ es la **varianza** sobre las mismas dimensiones.
- $\epsilon \approx 10^{-5}$ es un termino de estabilidad numerica.
- $\gamma \in \mathbb{R}^d$ es un parametro de **escala** aprendido.
- $\beta \in \mathbb{R}^d$ es un parametro de **shift** (bias) aprendido.

En palabras: para cada token, calcula la media y la desviacion estandar a lo largo de las $d_{\text{model}}$ dimensiones. **Centra** el vector restando la media. **Escala** dividiendo por la desviacion. Despues aplica una transformacion afin aprendida $(\gamma, \beta)$.

Resultado: cada token sale con media exactamente $\beta$ y desviacion estandar exactamente $\gamma$, **independientemente de lo que hizo la capa anterior**. Esa garantia de estabilidad es lo que hace entrenable un Transformer profundo.

---

## 3. La pregunta de Zhang & Sennrich 2019

En 2019, Biao Zhang y Rico Sennrich publicaron un paper corto y devastador: **"Root Mean Square Layer Normalization"**. Su pregunta era directa: ¿la **centrada** (restar la media) es realmente necesaria?

La intuicion clasica decia que si: centrar elimina sesgos en la distribucion y "ayuda al optimizer". Pero Zhang & Sennrich corrieron el experimento con cuidado en varias arquitecturas y encontraron lo siguiente.

Restar la media:

- Cuesta **una pasada extra** sobre el tensor: hay que calcular $\mu$, despues restar $\mu$ de cada coordenada. Eso es $O(d)$ adicional por token, por capa.
- Anade **un parametro extra** por dimension: el bias $\beta$. En un modelo con $d_{\text{model}} = 4096$ y 32 capas, son ~260K params solo en bias de LayerNorm.
- **Empiricamente no aporta calidad**. En los experimentos de Zhang & Sennrich (traduccion neural, language modeling), la centrada se podia eliminar sin cambio detectable en el loss final.

¿Y si solo normalizamos por la **magnitud** del vector, sin centrarlo? Eso es **RMSNorm**.

{{< concept-alert type="recordar" >}}
La pregunta de Zhang & Sennrich es un patron recurrente en deep learning: **¿este componente clasico es necesario o solo costumbre?** Si lo quitas y nada cambia, el componente era costumbre. La historia de la simplificacion de arquitecturas modernas — RMSNorm, no-bias en lineales, no-dropout, etc. — esta llena de versiones de esta misma pregunta.
{{< /concept-alert >}}

---

## 4. La formula de RMSNorm

$$\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\text{RMS}(x)}$$

donde el **Root Mean Square** del vector es:

$$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}$$

Tres diferencias frente a LayerNorm:

1. **No resta la media.** No hay $\mu$. El vector solo se divide por su magnitud RMS. Una pasada menos sobre el tensor.
2. **No tiene bias $\beta$.** Solo conserva la escala aprendida $\gamma$. La mitad de los parametros de norm desaparecen.
3. **Conserva la direccion del vector.** Como solo divide por un escalar (la RMS), la direccion de $x$ en $\mathbb{R}^d$ no cambia: solo se reescala su norma. LayerNorm, al restar la media, **traslada** el vector — y eso puede mover la direccion.

Esa tercera diferencia es la conceptual mas importante. La conviccion empirica detras de RMSNorm es que **la mayoria de la informacion util en un embedding esta en su direccion, no en su offset absoluto**. Centrar puede destruir parte de esa informacion.

---

## 5. La intuicion geometrica

Tomemos un vector chico para que se vea. Sea $x = [3, 4]$ en $\mathbb{R}^2$.

**LayerNorm (sin $\gamma, \beta$):**

- $\mu = (3 + 4)/2 = 3.5$
- $\sigma^2 = ((3-3.5)^2 + (4-3.5)^2)/2 = 0.25$, asi que $\sigma = 0.5$
- $\text{LayerNorm}(x) = [(3 - 3.5)/0.5, \, (4 - 3.5)/0.5] = [-1, \, 1]$

El vector salio **centrado en cero** (media 0) y con desviacion 1. Pero su **direccion cambio**: el vector original $[3, 4]$ apunta al primer cuadrante; el resultado $[-1, 1]$ apunta al segundo cuadrante. La traslacion lo movio.

**RMSNorm (sin $\gamma$):**

- $\text{RMS}(x) = \sqrt{(9 + 16)/2} = \sqrt{12.5} \approx 3.54$
- $\text{RMSNorm}(x) = [3/3.54, \, 4/3.54] \approx [0.85, \, 1.13]$

El vector salio en **la misma direccion** que el original (ambos apuntan al primer cuadrante), solo con magnitud reescalada. La proporcion entre coordenadas se mantuvo: en $[3, 4]$ la segunda coordenada es $4/3 \approx 1.33$ veces la primera; en $[0.85, 1.13]$ la segunda es $1.13/0.85 \approx 1.33$ veces la primera. **Identico.**

```
   LayerNorm                    RMSNorm
   x=[3,4]  ->  [-1, 1]         x=[3,4]  ->  [0.85, 1.13]
   (cambia direccion)           (preserva direccion)
```

> La mayor parte de la informacion semantica de un embedding vive en su **direccion** (el angulo en $\mathbb{R}^d$). El producto punto entre dos embeddings — que es lo que la attention usa para decidir similitud — depende fundamentalmente del angulo, no del offset absoluto. Centrar puede ensuciar esa senal. RMSNorm la preserva.

Esto conecta con todo lo que vimos en escalones tempranos: en `01-embeddings-y-dot-product` insistimos en que la similitud semantica se mide por **producto punto / coseno**, no por distancia euclidiana. RMSNorm es la operacion de normalizacion que respeta esa intuicion.

---

## 6. Implementacion (3 lineas utiles)

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))   # solo escala, sin beta

    def forward(self, x):
        # x: (B, T, d_model)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.gamma * x / rms
```

Linea por linea:

- `x.pow(2)`: eleva cada coordenada al cuadrado. Forma `(B, T, d_model)`.
- `.mean(dim=-1, keepdim=True)`: promedia sobre la ultima dimension (las $d_{\text{model}}$ coordenadas del token), dejando `(B, T, 1)`.
- `.add(self.eps).sqrt()`: suma $\epsilon$ y saca raiz. Eso es $\text{RMS}(x)$.
- `self.gamma * x / rms`: divide cada token por su RMS y reescala con el parametro aprendido.

Comparado con `nn.LayerNorm`, que internamente calcula media + varianza + dos parametros aprendidos + transformacion afin (entre 6 y 8 operaciones tensoriales segun la implementacion), **RMSNorm tiene 1 parametro aprendido y 4 operaciones**. La mitad de los flops.

Y la mitad de los parametros: para $d_{\text{model}} = 4096$, LayerNorm tiene $2 \times 4096 = 8192$ params por capa; RMSNorm tiene $4096$. En un LLaMA con 80 capas y dos normalizaciones por capa, son ~650K params menos. Chico en absoluto, pero gratis.

---

## 7. Por que LLaMA lo adopto

Resultados empiricos consistentes en la literatura post-2019:

- **Misma calidad final que LayerNorm**, dentro del ruido de seed-to-seed.
- **Aproximadamente 7-64% mas rapido en GPU**, dependiendo del modelo, el batch size y la implementacion del kernel. Los kernels fusionados de RMSNorm (en flash-attention, en triton) son particularmente eficientes porque hay una reduccion menos.
- **Implementacion mas simple**: menos lineas, menos bugs potenciales en kernels custom.
- **Menos parametros**: la mitad de params en cada capa de norm.

Resultado: el campo entero migro. **LLaMA (Meta, 2023), Mistral, Falcon, Qwen (Alibaba), DeepSeek, Gemma (Google) — todos usan RMSNorm.** Es el estandar moderno de facto. Si abres el codigo de cualquier LLM open-source post-2023, vas a ver `RMSNorm` (o un equivalente con otro nombre, como `T5LayerNorm` que tambien omite la centrada).

GPT-2/3/4 nominalmente siguen usando LayerNorm tradicional, pero la frontera de modelos open-source convergio a RMSNorm hace varios anos.

---

## 8. La math intuitiva: por que estabiliza

Una pregunta natural: si solo divides por la RMS, ¿que garantia tiene el modelo de que las activaciones quedan estables?

Veamos. Despues de aplicar RMSNorm (con $\gamma$ aprendido), la norma cuadrada del resultado es:

$$\|\text{RMSNorm}(x)\|^2 = \sum_i \left(\frac{\gamma_i \cdot x_i}{\text{RMS}(x)}\right)^2 = \frac{\sum_i \gamma_i^2 x_i^2}{\text{RMS}(x)^2}$$

Si los $\gamma_i$ son aproximadamente iguales (digamos todos $\approx \gamma$, lo tipico al inicio del entrenamiento donde se inicializan en 1):

$$\|\text{RMSNorm}(x)\|^2 \approx \gamma^2 \cdot \frac{\sum_i x_i^2}{\text{RMS}(x)^2} = \gamma^2 \cdot \frac{d \cdot \text{RMS}(x)^2}{\text{RMS}(x)^2} = d \cdot \gamma^2$$

Entonces $\|\text{RMSNorm}(x)\| \approx \sqrt{d} \cdot \gamma$.

**La salida de RMSNorm tiene magnitud predecible**, $\sqrt{d}$ veces el factor de escala aprendido — independientemente de cuan grande o chica era la entrada $x$. Esa es la propiedad clave que hace entrenable un Transformer profundo: cada bloque recibe vectores con magnitud controlada, sin importar lo que las capas previas hayan hecho con ellos.

LayerNorm da una garantia analoga (cada salida tiene varianza 1 y luego se escala por $\gamma$). RMSNorm la da con la mitad del trabajo.

---

## 9. Donde se aplica en el bloque

En el mini-GPT actual (escalon 07), el bloque Transformer usa **pre-norm** con LayerNorm:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, h)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

Para convertirlo a estilo LLaMA, el cambio es exactamente dos lineas:

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.ln1 = RMSNorm(d_model)        # <- aqui
        self.attn = MultiHeadAttention(d_model, h)
        self.ln2 = RMSNorm(d_model)        # <- aqui
        self.ffn = FeedForward(d_model, d_ff)
    # forward identico
```

Eso es todo. Sin cambios al optimizer, al scheduler, al training loop. La interfaz es la misma: recibe `(B, T, d_model)`, devuelve `(B, T, d_model)`. Solo cambia que adentro hace menos trabajo.

{{< concept-alert type="clave" >}}
La filosofia de la fase 5 — y de las modernizaciones modernas en general — es **substituibilidad limpia**. Cada nuevo componente respeta la interfaz del viejo, asi que se puede cambiar uno solo a la vez, medir el efecto aislado, y avanzar con confianza. Es el mismo principio del experimento controlado del escalon 13 (GELU vs ReLU): cambia una sola cosa, mantente todo lo demas igual.
{{< /concept-alert >}}

---

## 10. Pausa de verificacion

Antes de pasar a SwiGLU, asegurate de tener claro estos tres puntos:

1. **¿Que hace LayerNorm que RMSNorm no hace?** LayerNorm **resta la media** del vector (centrado) y tiene un parametro de **bias** $\beta$. RMSNorm omite ambas cosas: no centra y no tiene $\beta$. Solo divide por la RMS y multiplica por una escala $\gamma$.

2. **¿Por que la centrada no era necesaria empiricamente?** Porque la informacion semantica clave de un embedding esta en su **direccion**, no en su offset absoluto. La attention usa producto punto, que depende del angulo entre vectores. Trasladar el vector (restar la media) puede ensuciar esa senal sin aportar nada. Zhang & Sennrich 2019 mostraron que omitirlo no degrada la calidad.

3. **¿Cuanto mas rapido es RMSNorm en GPU?** Aproximadamente **7-64% mas rapido** dependiendo del setup. La diferencia viene de tener una reduccion menos (no calcular $\mu$) y la mitad de parametros para mover. En modelos grandes con cientos de capas, ese ahorro se acumula significativamente en tiempo de entrenamiento e inferencia.

---

## 11. Lo que aprendimos

Tres conclusiones para llevarte:

- **RMSNorm es LayerNorm sin la centrada y sin el bias.** Mismo proposito (estabilizar activaciones), menos trabajo, misma calidad final. Es uno de esos casos donde la version "mas simple" gana en todo: velocidad, parametros, claridad conceptual.
- **La direccion del vector es lo que importa.** RMSNorm preserva la direccion porque solo escala por un escalar. LayerNorm puede rotar el vector al trasladarlo. En un mundo donde la attention vive del producto punto, preservar direccion es respetar la semantica del modelo.
- **Es el primer paso de la modernizacion.** Solo. Una pieza chica. Pero el patron — identificar un componente clasico, preguntarse si todo lo que hace es necesario, simplificar — es el que vamos a repetir en los proximos 4 escalones. SwiGLU simplifica el FFN, RoPE simplifica los positional embeddings, GQA simplifica la attention multi-head, KV-cache simplifica el sampling. Cada uno es chico. Acumulados, son LLaMA.

> La distancia conceptual entre LayerNorm y RMSNorm es minuscula: borrar dos lineas y un parametro. La distancia entre "Transformer 2017" y "LLaMA 2023" es enorme. La razon de que esa distancia exista es que cinco simplificaciones minusculas, todas dirigidas a no hacer trabajo innecesario, se acumulan a lo largo de cientos de capas y miles de millones de parametros.

---

## Codigo y referencias

Codigo: `clase_14/practica/13_mini_llama.py` (clase `RMSNorm`, lineas iniciales del archivo).

Referencias:

- Zhang & Sennrich, **"Root Mean Square Layer Normalization"** (2019, NeurIPS).
- Ba, Kiros & Hinton, **"Layer Normalization"** (2016) — el paper original de LayerNorm.
- Touvron et al., **"LLaMA: Open and Efficient Foundation Language Models"** (Meta, 2023) — primera adopcion masiva de RMSNorm en un LLM frontera.

Siguiente: [17 - SwiGLU](../17-swiglu).
