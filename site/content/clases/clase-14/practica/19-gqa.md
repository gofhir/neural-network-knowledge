---
title: "19 - GQA: Grouped Query Attention"
weight: 190
math: true
---

En el escalon 06 construimos multi-head attention canonica: $h$ cabezas, cada una con sus propias $W_i^Q, W_i^K, W_i^V$. Es la version del paper de Vaswani 2017, la de BERT, la de GPT-2. Funciona, escala, y durante anos fue la receta. Pero cuando los modelos crecieron a la escala de **decenas de miles de millones de parametros y secuencias de miles de tokens**, una pieza que parecia inocente se volvio el cuello de botella: el **KV cache durante inferencia**.

Este escalon es sobre la solucion de ingenieria mas adoptada para ese problema. No agrega capacidad al modelo, no cambia la matematica fundamental de la attention, no introduce un truco de aprendizaje nuevo. Lo que hace es **reorganizar la geometria de las cabezas** para que el KV cache sea entre 4x y 8x mas chico, casi sin perder calidad. Se llama **Grouped Query Attention** (GQA), Ainslie et al. 2023, y es la attention que usan LLaMA 2 70B, LLaMA 3, Mistral, Mixtral, Qwen y Gemma. Es decir: casi todos los LLMs modernos.

Codigo: `clase_14/practica/13_mini_llama.py` (clase `GroupedQueryAttention`).

---

## 1. El problema: el KV cache es enorme en LLMs grandes

Para entender GQA primero hay que entender por que el KV cache existe. Durante **inferencia autoregresiva** — la generacion token a token que hace ChatGPT cuando responde — el modelo predice el token $t+1$ a partir de los tokens $1, 2, \ldots, t$ ya emitidos. Para cada nuevo token, recalcular self-attention sobre toda la secuencia desde cero seria $O(t^2)$ en cada paso, o $O(T^3)$ acumulado para una respuesta de longitud $T$. Es prohibitivo.

La optimizacion estandar es el **KV cache**: como las claves $K$ y los valores $V$ de los tokens previos no cambian al agregar un token nuevo (los pesos del modelo son fijos, los embeddings de los tokens viejos tampoco se mueven), se calculan una vez y se guardan. Cuando llega el token $t+1$, solo se calculan $K_{t+1}, V_{t+1}$ nuevos, se concatenan al cache, y se hace attention contra todo lo acumulado. El costo por token pasa de $O(t^2)$ a $O(t)$.

Esto suena gratis pero tiene un costo: **memoria**. El cache ocupa, por capa y por secuencia:

$$
\text{KV cache} = 2 \cdot n_{\text{layers}} \cdot h \cdot d_k \cdot T \cdot \text{bytes}
$$

donde el $2$ es por $K$ y $V$, $n_{\text{layers}}$ es la profundidad, $h$ el numero de cabezas, $d_k$ la dimension por cabeza, $T$ la longitud de secuencia, y `bytes` el tamano del tipo numerico (2 para float16, 4 para float32).

Para LLaMA 2 70B con secuencia de 4096 tokens, $n_{\text{layers}} = 80$, $h = 64$, $d_k = 128$, en float16:

$$
\text{KV cache} = 2 \cdot 80 \cdot 64 \cdot 128 \cdot 4096 \cdot 2 \approx 10.7 \text{ GB por secuencia}
$$

10 GB **solo en cache**, por una sola conversacion. Si servis 100 usuarios concurrentes en la misma GPU, son 1+ TB. Una H100 tiene 80 GB de VRAM. Los pesos del modelo (~140 GB en float16) ya no entran en una sola GPU; el cache de cada usuario ocupa el equivalente a multiples GPUs.

{{< concept-alert type="clave" >}}
A la escala de los LLMs modernos, **el KV cache se vuelve el cuello de botella, no los pesos del modelo**. Un sistema de produccion sirviendo 1000 usuarios concurrentes con secuencias largas dedica mas memoria al cache que a los pesos. Y a diferencia de los pesos (que se cargan una vez y se comparten entre usuarios), el cache es **per-secuencia**: cada conversacion paga su propia copia.
{{< /concept-alert >}}

Si el cache es el cuello de botella, la pregunta natural es: **podemos reducirlo sin matar la calidad?**

---

## 2. La idea de GQA: necesitamos $h$ K's distintos?

Antes de proponer una solucion, vale la pena cuestionar el supuesto. En multi-head canonico, cada cabeza tiene su propia tripleta $(W_i^Q, W_i^K, W_i^V)$. Eso da $h$ matrices $K$ y $h$ matrices $V$ distintas, y el cache guarda las $2h$ por token. Pero **realmente necesitamos $h$ K's distintos?**

Una serie de papers de "probing" en 2019 y 2020 encontraron pistas de que la respuesta es no:

- **Voita et al. (2019)** ("Analyzing Multi-Head Self-Attention") encontraron que en Transformers de NMT, una porcion grande de las cabezas son **redundantes**: se pueden podar sin perdida de calidad significativa. De ~96 cabezas en un modelo grande, ~30 hacian casi todo el trabajo util.
- **Michel et al. (2019)** ("Are Sixteen Heads Really Better than One?") mostraron que en BERT y MT models, la mayoria de las cabezas individuales se pueden eliminar tras entrenamiento sin afectar metricas downstream. En algunas capas, **una sola cabeza** alcanzaba para preservar el rendimiento.

Si las cabezas son redundantes a nivel de representacion completa, es razonable hipotetizar que tambien son redundantes a nivel de **claves y valores**. Es decir: aunque cada cabeza necesite su **propia query** para mantener su "lente" particular sobre la secuencia, puede ser suficiente con que **muchas cabezas compartan las mismas K, V**.

La intuicion fina: **lo que hace que una cabeza sea "distinta" es como mira (la query), no necesariamente sobre que mira (las claves) ni que extrae (los valores)**. Si dos cabezas tienen queries distintas pero comparten K y V, igual pueden producir distribuciones de atencion distintas — el softmax se calcula como $Q_i K^\top$, y cambiar $Q_i$ cambia el resultado aunque $K$ sea comun.

Esa hipotesis se traduce en una pregunta concreta de arquitectura: **podemos reducir el numero de cabezas K, V manteniendo $h$ cabezas Q?**

---

## 3. El espectro: MHA -> GQA -> MQA

La respuesta es si, y resulta que hay un espectro completo de configuraciones segun cuanto agresivo seas con el "compartir":

```
MHA (Vaswani 2017):
  h Q's, h K's, h V's
  cada cabeza tiene todo propio

MQA (Shazeer 2019):
  h Q's, 1 K, 1 V
  todas las cabezas comparten una unica K y V

GQA (Ainslie 2023):
  h Q's, g K's, g V's
  intermedio, con 1 < g < h
```

**MHA** (Multi-Head Attention) es lo de siempre: cada cabeza con su tripleta independiente. Es el extremo de "maxima expresividad, maximo cache".

**MQA** (Multi-Query Attention), introducida por Shazeer en 2019 ("Fast Transformer Decoding"), es el otro extremo: **una sola** K y **una sola** V para todas las cabezas. Esto reduce el cache por un factor de $h$ (en LLaMA 2 70B, 64x menos cache). Pero la perdida de calidad es notoria — alrededor de 2-3% en perplexity en modelos grandes — y los entrenamientos se vuelven inestables. MQA fue adoptada en algunos modelos (PaLM lo uso parcialmente) pero no se volvio universal.

**GQA** (Grouped Query Attention), Ainslie et al. 2023, es la propuesta intermedia: dividi las $h$ cabezas Q en $g$ grupos, y cada grupo comparte una unica K y V. El parametro $g$ controla el trade-off:

- $g = h$ es MHA (cada cabeza tiene su propia K, V).
- $g = 1$ es MQA (todas comparten una K, V).
- $g$ intermedio es el sweet spot.

LLaMA 2 70B usa $h = 64$ cabezas Q y $g = 8$ grupos KV, es decir 8 grupos de 8 cabezas Q cada uno. El cache se reduce a $1/8$ del MHA equivalente — de 10.7 GB a 1.3 GB por secuencia — perdiendo aproximadamente 0.5% de calidad. Casi gratis.

{{< concept-alert type="recordar" >}}
La regla de oro empirica: **MHA y GQA-8 son indistinguibles en calidad**, MQA pierde notablemente. El sweet spot es $g$ entre 4 y 8. Es la observacion central del paper de Ainslie 2023 y la razon por la cual GQA-8 es ahora el default en LLMs grandes.
{{< /concept-alert >}}

---

## 4. Visualizacion del agrupamiento

Hagamoslo concreto con un caso chico. Supongamos $h_q = 8$ cabezas Q y $h_{kv} = 2$ cabezas K, V (es decir, $g = 2$ grupos):

```
Q heads:   Q_0   Q_1   Q_2   Q_3   Q_4   Q_5   Q_6   Q_7
            \    \    /    /     \    \    /    /
              GRUPO 1               GRUPO 2
                |                     |
              K_0, V_0             K_1, V_1
```

Las cabezas Q 0, 1, 2, 3 forman el grupo 1 y todas usan $K_0, V_0$ para calcular su attention. Las cabezas Q 4, 5, 6, 7 forman el grupo 2 y usan $K_1, V_1$. Cada cabeza Q sigue siendo distinta — su matriz $W_i^Q$ es propia, su query es propia — pero las claves y valores con los que dialoga estan compartidos dentro del grupo.

La intuicion semantica es bonita: **el grupo define un "subespacio de attention"** (dado por sus K, V comunes), y dentro de ese subespacio, varias cabezas Q producen lentes distintas. Es como si las cabezas Q de un grupo fueran "preguntas diferentes sobre la misma base de datos".

Eso preserva la diversidad por dos razones:

1. **Las queries siguen siendo distintas** entre cabezas. Cada $W_i^Q$ tiene sus propios pesos aprendidos, asi que el patron de attention $\text{softmax}(Q_i K^\top / \sqrt{d_k})$ varia de cabeza en cabeza incluso con $K$ compartido.
2. **Los grupos permiten especializaciones gruesas**: el grupo 1 puede aprender una "vista" de la secuencia (por ejemplo, atencion local), el grupo 2 otra (por ejemplo, atencion global), y dentro de cada grupo las cabezas refinan distintos angulos de esa vista.

Para LLaMA 2 70B: $h_q = 64$, $h_{kv} = 8$, asi que cada grupo tiene 8 cabezas Q compartiendo una K y una V. Es como tener "8 sub-modelos de attention" corriendo en paralelo, cada uno con su propio espacio K/V y 8 queries distintas.

---

## 5. La matematica del ahorro

Vamos a contar parametros y memoria con numeros reales. Para una capa estilo LLaMA con $d_{model} = 4096$ y $h = 64$ cabezas (es decir, $d_k = 64$):

**MHA (todo propio):**

```
W_Q: (4096, 4096) = 16.8M parametros
W_K: (4096, 4096) = 16.8M
W_V: (4096, 4096) = 16.8M
W_O: (4096, 4096) = 16.8M
Total attention: 67.1M
```

**GQA con $h_{kv} = 8$ (LLaMA 2 70B):**

Aqui la matriz $W_K$ ya no proyecta de $d_{model}$ a $h \cdot d_k = d_{model}$, sino a $h_{kv} \cdot d_k = 8 \cdot 64 = 512$. Misma logica para $W_V$:

```
W_Q: (4096, 4096) = 16.8M
W_K: (4096, 512)  = 2.1M     <- 8x menos!
W_V: (4096, 512)  = 2.1M     <- 8x menos!
W_O: (4096, 4096) = 16.8M
Total attention: 37.8M
```

Reduccion en pesos de attention: ~44% (de 67M a 38M por capa). En un modelo de 80 capas como LLaMA 2 70B son ~2.3 GB menos en pesos. No es despreciable, pero tampoco es el premio principal.

**El premio real es el KV cache**, porque la formula del cache depende de $h_{kv}$ (el numero de cabezas K, V), no de $h_q$:

$$
\text{KV cache}_{\text{GQA}} = 2 \cdot n_{\text{layers}} \cdot h_{kv} \cdot d_k \cdot T \cdot \text{bytes}
$$

Para LLaMA 2 70B con $h_{kv} = 8$ vs $h = 64$:

```
MHA: 2 * 80 * 64 * 128 * 4096 * 2 = 10.7 GB por secuencia
GQA: 2 * 80 *  8 * 128 * 4096 * 2 =  1.3 GB por secuencia
```

**8x menos cache.** Eso significa que en una H100 podes:

- servir 8x mas usuarios concurrentes con la misma secuencia, o
- atender secuencias 8x mas largas con el mismo numero de usuarios, o
- cualquier combinacion intermedia.

Es la diferencia entre poder servir el modelo en produccion o no.

---

## 6. Implementacion

Vamos al codigo. La estructura sigue la version eficiente de multi-head que vimos en el escalon 06: una sola proyeccion grande, reshape, atencion vectorizada. La diferencia es que las matrices $W_K$ y $W_V$ son **mas pequenas** (proyectan a $h_{kv} \cdot d_k$ en lugar de $d_{model}$), y agregamos un paso para "replicar" las cabezas K, V hasta tener $h_q$ copias.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    GQA con h_q cabezas Q y h_kv < h_q cabezas K, V.
    Cada grupo de (h_q / h_kv) cabezas Q comparte una K, V.
    """
    def __init__(self, d_model, h_q, h_kv, block_size):
        super().__init__()
        assert h_q % h_kv == 0, "h_q debe ser divisible por h_kv"
        self.d_model = d_model
        self.h_q = h_q
        self.h_kv = h_kv
        self.d_k = d_model // h_q
        self.group_size = h_q // h_kv  # cuantas Q por cada K, V

        # Proyecciones: W_K y W_V son MAS CHICAS que en MHA
        self.W_Q = nn.Linear(d_model, h_q  * self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, h_kv * self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, h_kv * self.d_k, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Mascara causal (igual que en GPT)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, _ = x.shape

        # Proyectar
        Q = self.W_Q(x).view(B, T, self.h_q,  self.d_k).transpose(1, 2)
        # (B, h_q, T, d_k)
        K = self.W_K(x).view(B, T, self.h_kv, self.d_k).transpose(1, 2)
        # (B, h_kv, T, d_k)
        V = self.W_V(x).view(B, T, self.h_kv, self.d_k).transpose(1, 2)
        # (B, h_kv, T, d_k)

        # Replicar K, V para que matcheen las h_q cabezas de Q
        # Cada cabeza KV se duplica group_size veces
        K = K.repeat_interleave(self.group_size, dim=1)  # (B, h_q, T, d_k)
        V = V.repeat_interleave(self.group_size, dim=1)  # (B, h_q, T, d_k)

        # A partir de aqui, identico a MHA
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                      # (B, h_q, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_O(out)
```

Tres puntos clave:

- **`W_K` y `W_V` son mas chicas**: proyectan a `h_kv * d_k` en vez de `h_q * d_k = d_model`. Eso es donde se ahorran parametros y, lo mas importante, donde se ahorra **el shape del KV cache** durante inferencia.
- **`repeat_interleave(group_size, dim=1)`**: esta es la operacion de "broadcast" que replica cada cabeza KV `group_size` veces. Si tenes K de shape `(B, h_kv=8, T, d_k)` y `group_size=8`, despues de repeat_interleave queda `(B, 64, T, d_k)`. Cada bloque consecutivo de 8 cabezas tiene la misma K (y V). Conceptualmente, despues de esta operacion, el resto del codigo es identico a MHA.
- **En produccion no se materializa la replicacion**: `repeat_interleave` crea un tensor expandido, lo cual desperdicia memoria. Las implementaciones reales (Flash Attention, vLLM, xFormers) usan kernels especializados que tratan las cabezas KV como compartidas dentro del grupo y leen una sola vez de memoria por cada grupo. La logica matematica es identica; la implementacion es solo mas fina.

{{< concept-alert type="clave" >}}
La elegancia de GQA es que **se puede expresar como una pequena modificacion sobre MHA**: cambias dos shapes de matrices y agregas un repeat_interleave. Pero el efecto en memoria de inferencia es enorme. Es el tipo de cambio "barato en codigo, caro en impacto" que define la diferencia entre arquitecturas que escalan en produccion y arquitecturas que no.
{{< /concept-alert >}}

---

## 7. Trade-offs (memoria vs calidad)

El paper original (Ainslie et al. 2023, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints") corrio experimentos sistematicos sobre T5-Large y T5-XXL, comparando MHA, MQA y varias configuraciones de GQA. Los resultados resumidos:

| Variante  | Calidad (relativa a MHA) | KV cache memory |
|-----------|--------------------------|-----------------|
| MHA       | 100%                     | 100%            |
| GQA-8     | ~99.5%                   | 12.5%           |
| GQA-4     | ~99.0%                   | 6.25%           |
| GQA-2     | ~98%                     | 3.1%            |
| MQA       | ~97-98%                  | 1.6%            |

GQA-8 (8 grupos KV) es el sweet spot: pierdes ~0.5% de calidad medida en perplexity y benchmarks downstream, pero ganas 8x en memoria de cache. La diferencia entre MHA y GQA-8 cae **dentro del ruido** de variaciones de seed o de una corrida de entrenamiento mas larga.

A medida que reduces $g$ (mas cabezas Q por grupo, menos K, V), la calidad empieza a caer mas notablemente, hasta MQA que pierde ~2-3% — una diferencia chica pero ya consistentemente medible.

**Por que GQA-8 es casi gratis en calidad?** Hay dos hipotesis empiricas:

1. **Las K, V tienen "ancho de banda redundante"**: muchas cabezas en MHA aprenden K, V casi iguales (lo de Voita 2019). Compartir explicitamente entre 8 cabezas Q saca esa redundancia sin sacrificar lo unico.
2. **Las queries son lo verdaderamente diferenciador**: lo que da diversidad a las cabezas es como miran (la transformacion Q), no contra que comparan (K). Mantener $h$ Q's distintos preserva la mayoria de la expresividad.

**Por que MQA pierde mas?** Cuando todas las cabezas comparten una unica K, V, el modelo se vuelve mas estrecho: solo hay un "subespacio de attention" globalmente, y la diversidad de las queries queda limitada por esa unica perspectiva. La calidad sufre mas. Ademas, los entrenamientos se vuelven inestables — el paper de GQA documenta que MQA requiere truquitos extra de optimizacion que GQA no necesita.

---

## 8. Adopcion en modelos reales

GQA paso de ser una propuesta de paper a default de la industria en menos de un ano. Una mirada rapida a quien usa que:

| Modelo            | Configuracion              |
|-------------------|----------------------------|
| LLaMA 2 7B        | MHA ($h_q = h_{kv} = 32$)  |
| LLaMA 2 13B       | MHA ($h_q = h_{kv} = 40$)  |
| LLaMA 2 70B       | GQA-8 ($h_q = 64, h_{kv} = 8$) |
| LLaMA 3 (todas)   | GQA en todas las variantes |
| Mistral 7B        | GQA-8 ($h_q = 32, h_{kv} = 8$) |
| Mixtral 8x7B      | GQA                        |
| Qwen 2 (todas)    | GQA                        |
| Gemma             | GQA / MQA segun tamano     |
| GPT-4 (presunto)  | GQA o variantes (no publicado) |

Tres patrones a notar:

- **Modelos chicos suelen usar MHA**, modelos grandes GQA. La razon es que en modelos chicos el cache no es el cuello de botella (ya hay GPU memory de sobra), entonces no vale la pena el "0.5% que pierdes". A medida que el modelo crece, ese trade-off se invierte.
- **LLaMA 3 puso GQA hasta en sus variantes mas chicas (8B)**, indicando que la industria converge a GQA como default incluso cuando no es estrictamente necesario. El razonamiento es que si tu pipeline de inferencia ya esta optimizado para GQA, mejor usar GQA en toda la familia.
- **MQA quedo relegada a casos extremos**: solo tiene sentido cuando la memoria es absolutamente prioritaria sobre la calidad (modelos on-device, edge, mobile). En servers, GQA gana porque la calidad importa.

---

## 9. Por que GQA es revelacion para produccion

Hasta aqui mostramos que GQA reduce el cache. Pero vale la pena pensar en lo que eso significa **economicamente** para una empresa que sirve LLMs.

Antes de GQA, la economia de servir LLMs grandes era brutal:

- LLaMA 2 70B en MHA hipotetico tendria 10.7 GB de cache por usuario.
- En una H100 con 80 GB, despues de cargar los pesos del modelo (~140 GB con varios trucos de paralelizacion), te quedan ~20 GB para caches.
- Eso te da 1-2 usuarios concurrentes por GPU.

Despues de GQA-8:

- Cache por usuario: 1.3 GB.
- Mismos 20 GB para caches.
- 15-16 usuarios concurrentes por GPU.

La GPU no cambio. El modelo no cambio en parametros. Solo cambio la geometria de las cabezas K, V. Y la capacidad de servir aumento ~10x. Ese es el orden de magnitud que separa "viable comercialmente" de "no viable".

> Eso transformo los economics de servir LLMs. Mistral, Together, Anyscale, Anthropic, OpenAI — todos los proveedores de inferencia construyeron sus pipelines asumiendo GQA. El precio por token de los APIs cayo dramaticamente entre 2023 y 2024 en parte por GQA, en parte por quantization, en parte por mejoras de schedulers.

{{< concept-alert type="clave" >}}
GQA no es un truco de eficiencia "agradable de tener". Es lo que **permitio que serves LLMs de 70B parametros sea economicamente viable**. Sin GQA, la economia de la inferencia hubiera limitado a los proveedores a modelos mucho mas chicos o a precios mucho mas altos. Es uno de los cambios arquitecturales de mayor impacto economico de los ultimos anos en deep learning, y casi todo el efecto viene de **reorganizar tres matrices**.
{{< /concept-alert >}}

---

## 10. Conexion con Flash Attention y kernels especializados

Vale una nota tecnica sobre como se implementa GQA en sistemas reales. La version que mostramos arriba con `repeat_interleave` materializa fisicamente las K, V replicadas, lo cual es comodo para entender pero **desperdicia memoria GPU**: en el momento del attention tenes shape `(B, h_q, T, d_k)` para K aunque solo hay $h_{kv}$ K's reales.

En produccion se usan kernels que evitan esa materializacion:

- **Flash Attention 2 y 3** (Tri Dao et al.) tienen modo nativo para GQA: reciben K de shape `(B, h_kv, T, d_k)` directamente y, cuando recorren las queries para calcular el softmax, simplemente hacen `kv_idx = q_idx // group_size` para indexar la K, V correcta. No hay copia.
- **vLLM** maneja el cache de forma "paginada" (PagedAttention) y respeta la estructura de grupos: una sola pagina de K se usa para varias queries.
- **xFormers** y **TensorRT-LLM** tienen kernels equivalentes.

El resultado es que la implementacion real de GQA tiene **shapes de tensores mas chicos en GPU memory** que la version naive con `repeat_interleave`. Para LLaMA 2 70B en inferencia, el cache realmente ocupa los 1.3 GB que dijimos, no 10.7 GB.

Conexion con multi-head: igual que en el escalon 06, donde vimos que la version eficiente y la naive son **matematicamente equivalentes** y solo difieren en como se aprovecha la GPU, aqui pasa lo mismo. El `repeat_interleave` y los kernels nativos son computacionalmente identicos en lo que devuelven; lo que cambia es la huella de memoria y la velocidad.

---

## 11. Pausa de verificacion

Antes de pasar al siguiente escalon, asegurate de poder responder:

1. **Que es el KV cache y por que es problema?**
   Es el almacenamiento de las K y V de tokens previos durante inferencia autoregresiva, para evitar recalcularlas en cada paso. Crece linealmente con la profundidad, el numero de cabezas, la dimension por cabeza, y la longitud de secuencia. En LLMs grandes (70B+) con secuencias largas (4K+), facilmente supera los 10 GB por usuario, lo cual limita la cantidad de usuarios concurrentes que se pueden servir en una GPU.

2. **Que se comparte y que no en GQA?**
   Se **comparten K y V** dentro de cada grupo: $h_{kv}$ pares de matrices $(W^K_g, W^V_g)$ en lugar de $h_q$. Lo que **no se comparte** son las queries: cada una de las $h_q$ cabezas Q tiene su propia $W^Q_i$ y produce su propia query distinta. La diversidad estructural de las cabezas se preserva via las queries; el ahorro viene de las K, V.

3. **Por que GQA es "casi gratis" en calidad?**
   Por dos razones complementarias. Primera: trabajos de probing (Voita, Michel) mostraron que las K, V de muchas cabezas en MHA son redundantes — el modelo no usa toda la capacidad que le das. Compartir entre 8 cabezas saca esa redundancia sin sacrificar lo unico. Segunda: lo que diferencia funcionalmente a las cabezas es como miran (la query), no contra que (las claves). Mantener $h_q$ queries distintas preserva la mayor parte de la expresividad.

4. **Por que MQA pierde mas calidad que GQA-8?**
   Porque MQA reduce a **un solo subespacio de attention global** (una unica K, una unica V para todas las cabezas). Eso limita estructuralmente la diversidad de "vistas" que puede tener el modelo: aunque las queries sean distintas, todas operan sobre la misma base de claves y valores. GQA-8 mantiene 8 subespacios distintos (8 grupos, cada uno con su K, V), lo cual conserva mucha mas diversidad estructural mientras todavia ahorra 8x.

5. **Que cambia en el codigo entre MHA y GQA?**
   Tres cosas. Primero: las matrices `W_K` y `W_V` proyectan a `h_kv * d_k` en lugar de `d_model`, por lo cual son mas pequenas. Segundo: despues de hacer view/transpose, K y V tienen `h_kv` cabezas en lugar de `h_q`. Tercero: antes del matmul de attention, se hace `repeat_interleave(group_size, dim=1)` para que K y V matcheen el numero de cabezas de Q. El resto del codigo (softmax, mascara causal, projection final) es identico.

---

## Siguiente capitulo

[20 - KV-cache](../20-kv-cache): que es exactamente el cache, como se actualiza durante generacion, y por que GQA combinado con kv-cache es lo que hace viables las APIs de inferencia modernas.

Codigo: `clase_14/practica/13_mini_llama.py` (clase `GroupedQueryAttention`).

Volver al [hub de practica](..).
