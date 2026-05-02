---
title: "20 - KV-cache: generacion eficiente"
weight: 200
math: true
---

Hasta aqui todo lo que hemos visto en el bloque Transformer es entrenamiento: forward pass sobre toda la secuencia, calcular loss, backprop. Pero cuando un usuario interactua con ChatGPT, Claude o LLaMA en modo conversacion, el modelo no entrena: **genera**, token a token. Y resulta que la implementacion ingenua de la generacion — la misma que usamos en el escalon 08 con el mini-GPT — es catastroficamente ineficiente. Tan ineficiente que sin un truco especifico, **ChatGPT a 20 tokens por segundo seria fisicamente imposible**. El truco se llama **KV-cache** y es el primer ejemplo de un patron general en deep learning: cuando el algoritmo correcto cambia el orden de magnitud de complejidad, el sistema entero cambia.

Codigo: `clase_14/practica/13_mini_llama.py` (metodos del MiniLLaMA con `use_cache`).

---

## 1. La generacion naive es O(N²)

Recordar como genera el mini-GPT del escalon 08:

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]   # tomar contexto actual
        logits, _ = self(idx_cond)         # forward COMPLETO sobre todo el contexto
        logits = logits[:, -1, :]          # solo la ultima posicion importa
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

Mira la linea critica: `logits, _ = self(idx_cond)`. **Para generar el token N-esimo, hace un forward completo sobre los N tokens previos.** Toda la pila de Transformer — las atenciones, las FFN, las layernorms — corre sobre toda la secuencia, aunque al final solo nos interese la prediccion de la ultima posicion.

Cuenta de operaciones para generar una secuencia de longitud $N$ desde un prompt vacio:

$$
1 + 2 + 3 + \cdots + N = \frac{N(N+1)}{2} \approx \frac{N^2}{2}
$$

Es decir, **complejidad cuadratica $O(N^2)$ en el numero de forward passes** medido en tokens procesados. Para $N = 100$ tokens son ~5000 unidades de trabajo. Para $N = 2000$ son ~2 millones. Para $N = 100000$ (contextos largos modernos) son ~5 mil millones. **Inservible para produccion.**

Lo peor: las primeras $N-1$ posiciones estan haciendo, en cada paso, **exactamente las mismas operaciones que ya hicieron antes**. Computo desperdiciado.

---

## 2. El insight: K y V de tokens pasados NO cambian

Recordemos que en self-attention causal cada posicion $t$ produce tres vectores: una **query** $Q_t$, una **key** $K_t$ y un **value** $V_t$. La attention en la posicion $t$ es:

$$
\text{attn}_t = \text{softmax}\!\left(\frac{Q_t \cdot [K_1, K_2, \ldots, K_t]^\top}{\sqrt{d_k}}\right) \cdot [V_1, V_2, \ldots, V_t]
$$

Ahora observemos que pasa cuando generamos:

- **Token 1**: calculamos $Q_1, K_1, V_1$. Predecimos $T_2$.
- **Token 2**: calculamos $Q_1, K_1, V_1$ (otra vez), mas $Q_2, K_2, V_2$. Predecimos $T_3$.
- **Token 3**: calculamos $Q_1, K_1, V_1$, $Q_2, K_2, V_2$ (otra vez), mas $Q_3, K_3, V_3$. Predecimos $T_4$.

$K_1$ y $V_1$ son **identicos** cada vez que aparecen. ¿Por que? Porque dependen solo del embedding de $T_1$, los pesos $W_K, W_V$ y la posicion 1 — nada de eso cambio entre paso y paso.

{{< concept-alert type="clave" >}}
Las queries $Q_t$ se descartan despues de cada paso (solo se usaron para calcular la attention de su posicion). Pero las keys $K_t$ y values $V_t$ son referenciadas por **todas las posiciones futuras**. Si las guardamos, no necesitamos recalcularlas.
{{< /concept-alert >}}

Eso es **KV-cache**: una memoria que acumula $K$ y $V$ de cada paso, capa por capa, para reutilizarlos en pasos siguientes.

---

## 3. El nuevo flujo con cache

Con KV-cache el patron de generacion cambia radicalmente:

```
Token 1:
  Input: [T_1]                              (1 token)
  Calcular Q_1, K_1, V_1
  CACHEAR: K_cache=[K_1], V_cache=[V_1]
  Attention: Q_1 @ K_cache.T -> softmax -> @ V_cache
  Predecir T_2

Token 2:
  Input: SOLO [T_2]                         (1 token, no 2)
  Calcular Q_2, K_2, V_2
  K_cache = [K_1, K_2], V_cache = [V_1, V_2]
  Attention: Q_2 @ K_cache.T -> softmax -> @ V_cache
  Predecir T_3

...

Token N:
  Input: SOLO [T_N]                         (1 token, no N)
  Calcular Q_N, K_N, V_N
  K_cache = [K_1, ..., K_N], V_cache = [V_1, ..., V_N]
  Attention: Q_N @ K_cache.T -> softmax -> @ V_cache
  Predecir T_{N+1}
```

**Cada paso procesa solo el token nuevo.** Las matrices $W_Q, W_K, W_V, W_O$ se aplican a un unico vector por capa (no a $N$ vectores). El embedding lookup, las layernorms, las FFN — todo opera sobre **un solo token**.

Lo unico que crece linealmente es el cache de keys y values, contra el cual la query del paso actual hace producto interno.

---

## 4. Cuanto se ahorra

Para generar $N$ tokens (sin contar el prompt):

| Componente            | Sin cache         | Con cache         |
|-----------------------|-------------------|-------------------|
| Embedding + FFN + Norm| $O(N^2)$          | $O(N)$            |
| Proyecciones $Q, K, V$| $O(N^2)$          | $O(N)$            |
| Attention (Q·K, ·V)   | $O(N^2)$          | $O(N^2)$          |
| **Trabajo total**     | $O(N^2)$          | $O(N^2)$ pero con constante mucho menor |

La attention sigue siendo $O(N^2)$ porque cada query nueva debe atender a todo el cache acumulado, que crece. Pero el resto del modelo — que en un Transformer real es la **mayoria del compute** (FFN tipicamente representa ~2/3 del FLOPs por capa) — pasa de $O(N^2)$ a $O(N)$.

En la practica esto se traduce en **10x a 100x mas rapido** dependiendo del modelo y del largo de secuencia. Para LLaMA-2 7B con $N = 2048$, el speedup tipico es $\sim 50x$.

> Sin KV-cache, ChatGPT a 20 tokens/seg seria fisicamente imposible. Con KV-cache, una sola GPU H100 puede servir respuestas de varios miles de tokens en segundos.

---

## 5. La distincion: prefill vs decode

Cuando un usuario manda un prompt y espera una respuesta, el modelo pasa por **dos fases muy distintas**:

**Prefill** (procesar el prompt):

```
Input: oracion completa de N tokens del prompt
       (por ejemplo: "Explicame la fotosintesis en 3 lineas")
       |
       v
Un solo forward pass sobre toda la secuencia,
llenando el KV-cache con K, V de todos los tokens del prompt.
```

Eficiente y **paralelizable**: todas las posiciones se procesan en paralelo en la GPU. Esta fase domina el "tiempo al primer token" (TTFT).

**Decode** (generar respuesta token a token):

```
Cada paso: 1 token nuevo de input + atender a todo el cache acumulado
```

**Secuencial**: cada token depende del anterior. No se puede paralelizar dentro de una misma secuencia. Esta fase domina el "tiempo entre tokens" (TPOT, time per output token).

{{< concept-alert type="clave" >}}
Por eso las APIs de LLMs **cobran distinto** por input tokens (prefill, baratos, paralelizables) vs output tokens (decode, caros, secuenciales). Generar 1000 tokens de output cuesta tipicamente **3-5x mas** que procesar 1000 tokens de input. La asimetria es estructural, no comercial.
{{< /concept-alert >}}

---

## 6. Implementacion

La idea se traduce a PyTorch con muy poco codigo. Anadimos dos buffers `cache_k` y `cache_v` al modulo de attention y un flag `use_cache`:

```python
class CausalAttentionKVCache(nn.Module):
    def __init__(self, d_model, h, ...):
        super().__init__()
        # ... pesos W_Q, W_K, W_V, W_O como antes
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        B, T, _ = x.shape
        Q = self.W_Q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.h, self.d_k).transpose(1, 2)

        if use_cache:
            if self.cache_k is not None:
                # Concatenar el cache anterior con K, V del paso actual
                K = torch.cat([self.cache_k, K], dim=2)
                V = torch.cat([self.cache_v, V], dim=2)
            self.cache_k = K
            self.cache_v = V

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        # ... softmax + multiplicacion por V + W_O
        return out
```

Notar que dentro del modulo, $K$ y $V$ se reemplazan por la **version concatenada** (cache anterior mas el nuevo). La query $Q$ sigue siendo solo del token actual ($T = 1$ durante decode).

Y el `generate` se reescribe asi:

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens):
    self.reset_cache()

    # PREFILL: procesar todo el prompt en un solo forward
    self(idx, use_cache=True)

    # DECODE: generar uno por uno, alimentando solo el ultimo token
    for _ in range(max_new_tokens):
        last = idx[:, -1:]                         # SOLO el token mas reciente
        logits, _ = self(last, use_cache=True)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

Comparalo con el `generate` ingenuo de la seccion 1: la diferencia es que en cada paso le pasamos `idx[:, -1:]` (un solo token) en vez de `idx[:, -block_size:]` (todo el contexto). El cache se ocupa de mantener la historia.

---

## 7. La sutileza de la causal mask

Cuando hicimos self-attention causal en el escalon 06 introducimos la **mascara triangular** para impedir que cada posicion vea el futuro durante el entrenamiento. ¿Que pasa con esa mascara durante el decode con cache?

**No la necesitamos.** Solo procesamos **una query** (el token nuevo). Esa query atiende a todas las keys del cache, **todas las cuales son del pasado por construccion** — porque las metimos al cache en pasos anteriores. No hay futuro al que enmascarar; el cache ya filtro temporalmente por el orden en que lo construimos.

La mascara causal sigue siendo necesaria en otros dos contextos:

- **Training**: la secuencia entera entra al modelo y cada posicion debe predecir solo en base al pasado. Mascara obligatoria.
- **Prefill**: el prompt entero entra como una sola tanda; multiples queries atienden multiples keys, y dentro de ese batch hay que respetar el orden causal. Mascara obligatoria.
- **Decode con cache**: una sola query, todas las keys del cache son del pasado. **Mascara innecesaria.**

Esto es una de esas micro-optimizaciones que en un sistema real (vLLM, TGI, llama.cpp) se aprovechan para ahorrar operaciones.

---

## 8. Costo de memoria del cache

KV-cache no es gratis. Tiene un costo de memoria que crece con la longitud de la secuencia:

$$
\text{KV cache size} = 2 \cdot L \cdot h \cdot d_k \cdot N \cdot B \cdot \text{dtype\_bytes}
$$

donde $L$ es el numero de capas, $h$ el numero de heads, $d_k$ la dimension por head, $N$ la longitud de secuencia, $B$ el batch (numero de usuarios concurrentes) y `dtype_bytes` los bytes por escalar (2 para fp16/bf16). El factor $2$ es por las dos matrices $K$ y $V$.

Numeros concretos para **LLaMA-2 70B** con contexto 4096:

- $L = 80$ capas
- $h = 64$ heads
- $d_k = 128$
- $N = 4096$
- $B = 1$ (un solo usuario)
- dtype = fp16 (2 bytes)

$$
2 \cdot 80 \cdot 64 \cdot 128 \cdot 4096 \cdot 1 \cdot 2 \text{ bytes} \approx 10.7 \text{ GB}
$$

**10.7 GB de memoria GPU por cada conversacion concurrente, solo para el cache.** Y eso sin contar los pesos del modelo (que son ~140 GB en fp16, otros 70 GB en int8).

Para 100 usuarios concurrentes: 1 TB de cache. Por eso los proveedores de inferencia tienen ingenieria de servir **el problema central**: como cabe el cache en GPU, como compartirlo entre prompts similares (prefix caching), como evictarlo cuando una conversacion se vuelve inactiva.

> **Por eso GQA importa**: como vimos en el escalon anterior, GQA reduce el factor $h \cdot d_k$ del cache compartiendo $K, V$ entre grupos de queries. En LLaMA-2 70B, GQA usa 8 grupos en vez de 64 heads, **reduciendo el cache 8x** — de 10.7 GB a 1.3 GB por conversacion. Esa es la diferencia entre poder servir 8 usuarios o 1 con la misma GPU.

---

## 9. El "click"

Resumen del cambio mental que produce KV-cache:

```
Sin KV-cache:  cada paso procesa N tokens (todos)
Con KV-cache:  cada paso procesa 1 token (el nuevo)

Memoria adicional:  O(N) por capa por usuario
Tiempo por paso:    O(N) (vs O(N^2) acumulado de la version naive)
Speedup tipico:     10x a 100x dependiendo del seq_len
```

KV-cache es uno de esos casos donde **el algoritmo correcto cambia el orden de magnitud del problema**. No es una optimizacion micro; es un cambio estructural que separa "investigacion academica" de "sistema en produccion sirviendo millones de usuarios".

{{< concept-alert type="recordar" >}}
Patron general en sistemas: cuando un computo se repite identicamente paso tras paso, **cachealo**. Suena obvio. Pero en deep learning hay que reconocer cual es la unidad invariante (aqui: $K$ y $V$ por token), garantizar que no cambia (aqui: posiciones pasadas no se reescriben), y disenar la API para exponer ese caching limpiamente (aqui: el flag `use_cache` y el metodo `reset_cache`).
{{< /concept-alert >}}

---

## 10. Pausa de verificacion

Antes de pasar al siguiente escalon, asegurate de poder responder:

1. **¿Por que la generacion naive es O(N²)?** Porque en cada uno de los $N$ pasos de generacion el modelo hace un forward completo sobre los hasta-$N$ tokens previos. La suma $1 + 2 + \cdots + N$ es $\frac{N(N+1)}{2} \approx \frac{N^2}{2}$.

2. **¿Que se cachea exactamente y por que?** Las matrices $K_t$ y $V_t$ de cada token, en cada capa de cada head. No las queries (se descartan despues de cada paso) ni las activaciones intermedias del FFN (no se referencian desde el futuro). $K$ y $V$ se cachean porque son **invariantes**: dependen del embedding del token y los pesos, no del paso de generacion actual.

3. **¿Por que la causal mask se vuelve innecesaria con cache?** Porque durante decode procesamos **una sola query** (la del token nuevo), y todas las keys contra las que atiende ya estan en el cache, todas son del pasado por construccion. No hay futuro que enmascarar. La mascara sigue siendo necesaria en training y en prefill, donde hay multiples queries simultaneas.

---

Codigo: `clase_14/practica/13_mini_llama.py` (metodos del MiniLLaMA con `use_cache`).

Siguiente: [21 - Mini-LLaMA: todo junto](../21-mini-llama).
