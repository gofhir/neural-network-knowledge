---
title: "18 - RoPE: posicion via rotaciones geometricas"
weight: 180
math: true
---

Hasta aqui los Transformers que hemos construido inyectan informacion posicional **sumandola** al embedding al inicio del modelo: tomas el token-embedding, le sumas un position-embedding, y dejas que las 12 (o 32, o 96) capas de atencion se las arreglen para conservar esa senal. Funciona. GPT-2 funciona asi. BERT funciona asi. El paper original de Vaswani 2017 funciona asi.

Pero si abres el codigo de LLaMA, GPT-NeoX, PaLM, Mistral, Qwen, Gemma o practicamente cualquier LLM open-source moderno, **la posicion ya no se suma al embedding**. Se aplica de otra forma, en cada capa, **dentro** del mecanismo de atencion, y la operacion no es una suma sino una **rotacion**. Esa tecnica se llama **RoPE — Rotary Position Embedding** (Su et al., 2021).

Este capitulo es el mas matematicamente rico de la fase 5. RoPE es la modernizacion mas elegante del Transformer: una sola idea geometrica — rotar pares de dimensiones de Q y K en funcion de la posicion — resuelve simultaneamente tres problemas distintos de las codificaciones tradicionales. Sin agregar ni un solo parametro aprendible.

Codigo: `clase_14/practica/13_mini_llama.py` (funciones `precompute_rope` y `apply_rope`).

---

## 1. Las 3 limitaciones de las codificaciones posicionales tradicionales

Repasemos como inyectabamos posicion en los capitulos anteriores. En el mini-GPT del escalon 08, la primera linea del forward pass era:

```python
def forward(self, idx):
    B, T = idx.shape
    tok = self.tok_emb(idx)                              # (B, T, d_model)
    pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
    x = tok + pos                                        # info posicional SUMADA
    ...
```

La posicion entra al modelo **una sola vez**, sumada al embedding del token, justo despues del lookup table. De ahi en adelante, las 12 capas de atencion procesan el vector `x` sin recibir ningun recordatorio adicional de "este token estaba en la posicion 47".

Esto tiene tres problemas. Cada uno es serio por su cuenta; juntos son devastadores a las escalas de los LLMs modernos.

### 1.1 Problema 1: la senal posicional se diluye con la profundidad

Cuando sumas el position-embedding al token-embedding, estas mezclando dos tipos de informacion en el mismo vector. La capa 1 puede separarlos perfectamente — apenas vienen "frescos". Pero cada capa de atencion **transforma** el vector: lo proyecta a Q/K/V, hace mezclas ponderadas con todos los demas tokens, lo pasa por LayerNorm y por el FFN. Despues de 12 capas, la "porcion posicional" del vector original esta enterrada bajo capas y capas de transformaciones no lineales.

Estudios empiricos sobre BERT y GPT-2 muestran que en las capas profundas la informacion posicional se vuelve dificilmente recuperable. El modelo "siente" la posicion fuerte en las primeras capas y mas vagamente en las ultimas. Eso desperdicia capacidad: las capas profundas son justo las que toman las decisiones semanticas mas refinadas, y son las que tienen menos informacion posicional disponible.

### 1.2 Problema 2: no extrapola fuera de la longitud de entrenamiento

Si entrenas un modelo con `block_size = 1024`, tu `nn.Embedding(1024, d_model)` solo tiene parametros para las posiciones 0..1023. La posicion 5000 **literalmente no existe**: si en inferencia le pides al modelo que procese un contexto de 5000 tokens, hay que decidir que hacer con esa posicion fuera de rango. Las opciones son malas.

Las codificaciones **sinusoidales** de Vaswani (`sin/cos` de varias frecuencias) tecnicamente estan definidas para cualquier posicion entera, pero el modelo nunca vio durante entrenamiento las distribuciones de scores que produciria una posicion 5000. Fuera de distribucion. La calidad colapsa.

Las codificaciones **aprendidas** (las que usa GPT-2 con `nn.Embedding(block_size, d_model)`) son aun peores: la posicion 5000 ni siquiera tiene un vector valido. Hay que inventar uno.

A escalas modernas, donde Gemini 1.5 promete 2M tokens de contexto y los benchmarks de "needle in a haystack" exigen extrapolar a 100K+, esto es una limitacion fatal.

### 1.3 Problema 3: la posicion es absoluta, no relativa

Cuando sumas `pos_emb(47)` al embedding del token en posicion 47, le estas diciendo al modelo "este token esta en la posicion 47". Punto. No le estas diciendo "este token esta a 5 posiciones del que viene antes".

Pero el lenguaje no funciona en posiciones absolutas. Lo importante es la **distancia relativa**: que tan cerca esta un token del otro. La oracion "El gato negro duerme" tiene la misma estructura semantica si la pones al inicio del documento o en el medio del parrafo 47. El modelo deberia aprender que "negro" modifica a "gato" en base a que estan a una posicion de distancia, **no** en base a sus posiciones absolutas en el corpus.

Las codificaciones absolutas obligan al modelo a aprender la propiedad de "distancia" indirectamente, comparando dos vectores absolutos. Es como pedirle a alguien que te diga la distancia entre dos ciudades dandole solo sus coordenadas GPS, sin permitirle restarlas. Si claro, se puede, pero es una vuelta innecesaria.

{{< concept-alert type="recordar" >}}
Tres limitaciones de las codificaciones posicionales tradicionales: **(1)** se diluyen con la profundidad porque solo se inyectan al inicio; **(2)** no extrapolan bien fuera de la longitud de entrenamiento; **(3)** son absolutas en vez de relativas, forzando al modelo a derivar la nocion de distancia indirectamente. RoPE resuelve las tres con una sola idea geometrica.
{{< /concept-alert >}}

---

## 2. La idea de RoPE: rotar pares de dimensiones

La intuicion central de RoPE es radicalmente distinta. En vez de **sumar** un vector posicional al embedding al inicio del modelo, RoPE **rota** los vectores Q y K en cada capa de atencion, con un angulo proporcional a la posicion del token.

Veamoslo en pequeno. Supon que `d_k = 8`. RoPE divide ese vector en 4 pares de dimensiones: `(d_0, d_1)`, `(d_2, d_3)`, `(d_4, d_5)`, `(d_6, d_7)`. Cada par se trata como un **vector 2D** en su propio plano, y se rota por un angulo distinto que depende de la posicion del token.

```
Q de 8 dimensiones, posicion p:

   [ d_0, d_1 ]   <- rotado por angulo (p * theta_0)
   [ d_2, d_3 ]   <- rotado por angulo (p * theta_1)
   [ d_4, d_5 ]   <- rotado por angulo (p * theta_2)
   [ d_6, d_7 ]   <- rotado por angulo (p * theta_3)
```

Donde:

- $p$ es la posicion del token (0, 1, 2, ...).
- $\theta_i$ es la **frecuencia** del par $i$. Pares con $i$ chico tienen frecuencia alta (rotan rapido); pares con $i$ grande tienen frecuencia baja (rotan lento). Es la misma idea de "muchos relojes a distintas escalas" de las codificaciones sinusoidales.

La rotacion en 2D, recordatorio. Para rotar un vector $(x_1, x_2)$ por un angulo $\alpha$:

$$
\begin{bmatrix} x_1' \\ x_2' \end{bmatrix} = \begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} x_1 \cos\alpha - x_2 \sin\alpha \\ x_1 \sin\alpha + x_2 \cos\alpha \end{bmatrix}
$$

Notese: la rotacion **conserva la magnitud** del vector. $|R(\alpha) v| = |v|$. No estamos cambiando "cuanto vale" el embedding, solo "hacia donde apunta" en su plano 2D.

Para un vector $Q$ de dimension $d_k$ en posicion $p$, la operacion completa de RoPE es:

$$
\text{RoPE}_p(Q) = \begin{bmatrix} R(p \theta_0) & & & \\ & R(p \theta_1) & & \\ & & \ddots & \\ & & & R(p \theta_{d_k/2 - 1}) \end{bmatrix} \cdot Q
$$

Una matriz **bloque-diagonal** donde cada bloque es una rotacion 2D distinta. Es ortogonal por construccion (no cambia magnitudes), depende de la posicion via los angulos, y no tiene parametros aprendibles: las frecuencias $\theta_i$ son fijas, no se entrenan.

{{< concept-alert type="clave" >}}
RoPE no SUMA informacion posicional al embedding: ROTA el embedding en pares de dimensiones, con angulos proporcionales a la posicion. La operacion conserva magnitudes, no agrega parametros, y se aplica solo a Q y K (no a V) en cada capa de atencion.
{{< /concept-alert >}}

---

## 3. La magia: el dot product depende solo de la posicion RELATIVA

Aqui esta la parte magica de RoPE, la propiedad por la que vale la pena toda la maquinaria. Vamos a demostrar que, despues de aplicar RoPE, el producto punto $Q' \cdot K'$ (que es lo que entra al softmax de attention) **depende solo de la diferencia de posiciones $p_q - p_k$**, no de las posiciones absolutas $p_q$ y $p_k$.

Es decir, el modelo termina razonando en **distancias relativas** automaticamente, sin que tengamos que enseñarle.

### 3.1 La derivacion

Sean dos tokens en posiciones $p_q$ y $p_k$. Sus queries y keys, tras aplicar RoPE:

$$
Q'_{p_q} = R(\theta p_q) \cdot q \qquad K'_{p_k} = R(\theta p_k) \cdot k
$$

donde $R(\alpha)$ es la matriz de rotacion bloque-diagonal y $\theta$ es el vector de frecuencias. Para simplificar la notacion vamos a usar el caso de un solo par 2D — se generaliza pegando bloques.

El dot product en attention es:

$$
\text{score}(p_q, p_k) = (Q'_{p_q})^\top K'_{p_k} = (R(\theta p_q) q)^\top (R(\theta p_k) k)
$$

Aplicando la propiedad de la transpuesta del producto $(AB)^\top = B^\top A^\top$:

$$
= q^\top R(\theta p_q)^\top R(\theta p_k) k
$$

Ahora dos hechos clave de las matrices de rotacion:

1. **Componer rotaciones suma angulos**: $R(\alpha) R(\beta) = R(\alpha + \beta)$.
2. **Transponer una rotacion la invierte**: $R(\alpha)^\top = R(-\alpha)$. (Para matrices ortogonales, $R^\top = R^{-1}$.)

Aplicando ambos:

$$
q^\top R(\theta p_q)^\top R(\theta p_k) k = q^\top R(-\theta p_q) R(\theta p_k) k = q^\top R(\theta(p_k - p_q)) k
$$

Y ahi esta la magia:

$$
\boxed{(Q'_{p_q})^\top K'_{p_k} = q^\top R(\theta(p_k - p_q)) k}
$$

**El score depende solo de $(p_k - p_q)$**, la diferencia de posiciones, no de los valores absolutos $p_q$ y $p_k$. Las posiciones absolutas se **cancelan** algebraicamente.

### 3.2 Que significa eso semanticamente

Significa que **el modelo razona en terminos relativos**. Si tienes dos tokens A y B con score $s$ cuando estan en posiciones (10, 15), el score sera **identico** $s$ cuando esten en posiciones (1000, 1005), o (47, 52), o cualquier par cuya diferencia sea 5.

Compara con codificaciones absolutas: ahi `pos_emb(10)` y `pos_emb(1000)` son vectores completamente distintos, asi que la suma `tok + pos_emb(10)` y `tok + pos_emb(1000)` producen embeddings distintos, y los scores cambian incluso para tokens identicos a la misma distancia. Ese era el problema 3.

Con RoPE, el modelo aprende patrones del tipo "este verbo atiende fuerte al sujeto que esta a 1-3 tokens de distancia" sin tener que aprenderlo por separado para cada posicion absoluta. Generaliza inmediatamente.

{{< concept-alert type="clave" >}}
RoPE convierte la atencion en una funcion de la **posicion relativa** $(p_q - p_k)$ por construccion algebraica, no por aprendizaje. Las posiciones absolutas se cancelan en el dot product gracias a la propiedad $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$ de las matrices de rotacion.
{{< /concept-alert >}}

---

## 4. Visualizacion de la rotacion

Veamos un ejemplo concreto. Supon un par 2D del query con valor $q = [3, 2]$ y frecuencia $\theta = \pi/8$ (45 grados / 2). Veamos como va rotando segun la posicion:

| Posicion $p$ | Angulo $p \theta$ | Vector rotado |
|--------------|-------------------|---------------|
| 0            | 0   rad           | $[3.00, 2.00]$ |
| 1            | $\pi/8$ rad       | $[1.99, 2.99]$ |
| 2            | $\pi/4$ rad       | $[0.71, 3.54]$ |
| 3            | $3\pi/8$ rad      | $[-0.62, 3.61]$ |
| 4            | $\pi/2$ rad       | $[-2.00, 3.00]$ |

Los numeros calzan con la formula $(x \cos\alpha - y \sin\alpha,\ x \sin\alpha + y \cos\alpha)$. Lo importante: el vector **gira** en circulo, manteniendo magnitud constante $\sqrt{3^2 + 2^2} \approx 3.61$.

```
        Y
        |
   p=3  *           * p=2
        |
        |   * p=1
        |        * p=0
   ---- + -------------> X
        |
   p=4  *
        |
```

Cada token, segun su posicion, se "para" en un punto distinto del circulo de radio $|q|$. Tokens cercanos en posicion estan cercanos en el circulo. Tokens lejanos estan en angulos opuestos.

Ahora extiende esto a $d_k = 64$: el query tiene 32 pares 2D, cada uno girando a su propia velocidad. La posicion del token queda **codificada de forma distribuida** en los 32 angulos simultaneos.

---

## 5. Las frecuencias: los relojes a distintas escalas

Falta especificar las frecuencias $\theta_i$. RoPE usa la misma formula de las codificaciones sinusoidales de Vaswani:

$$
\theta_i = 10000^{-2i/d_k} \quad \text{para } i = 0, 1, \ldots, d_k/2 - 1
$$

Vamos a tomar $d_k = 64$ (un valor tipico de una cabeza de attention en LLaMA-7B con 32 cabezas):

| Par $i$ | Frecuencia $\theta_i$           | Periodo $2\pi/\theta_i$ (en posiciones) |
|---------|----------------------------------|------------------------------------------|
| 0       | $10000^0 = 1.00$                 | $\sim 6.28$ posiciones                   |
| 8       | $10000^{-0.25} \approx 0.10$     | $\sim 63$ posiciones                     |
| 16      | $10000^{-0.50} = 0.01$           | $\sim 628$ posiciones                    |
| 24      | $10000^{-0.75} \approx 0.001$    | $\sim 6280$ posiciones                   |
| 31      | $10000^{-31/32} \approx 0.0001$  | $\sim 60000$ posiciones                  |

El par 0 da una vuelta completa cada ~6 tokens. El par 31 da una vuelta completa cada ~60000 tokens. Distintos relojes a distintas escalas.

Por que esa formula con base 10000? Porque combina:

- **Frecuencias altas** (par 0) capturan relaciones de **corta distancia**: el modelo distingue claramente entre estar a 1 token vs estar a 2 tokens.
- **Frecuencias bajas** (par 31) capturan relaciones de **larga distancia**: el modelo distingue entre estar a 100 tokens vs estar a 1000 tokens, sin que la diferencia se vuelva trivial.

Es la misma logica de las codificaciones binarias de los numeros: el bit menos significativo cambia rapido, el mas significativo cambia lento, y juntos codifican el numero entero.

El 10000 es magico-ad-hoc: viene de Vaswani 2017, no de un argumento riguroso. RoPE lo hereda. Variantes recientes (NTK scaling, YaRN) ajustan ese valor para extender el contexto.

---

## 6. Implementacion

Aqui esta la implementacion compacta tal como aparece en `13_mini_llama.py`:

```python
import torch


def precompute_rope(d_k, max_seq_len, theta=10000.0):
    """
    Precomputa cos y sin de los angulos de RoPE.

    Returns:
        cos: (max_seq_len, d_k/2)
        sin: (max_seq_len, d_k/2)
    """
    # frecuencias: theta_i = 1 / (10000 ^ (2i/d_k)) para i = 0, ..., d_k/2 - 1
    freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))

    # posiciones: 0, 1, 2, ..., max_seq_len - 1
    positions = torch.arange(max_seq_len).float()

    # producto exterior: angulo[p, i] = p * theta_i
    angles = torch.outer(positions, freqs)         # (max_seq_len, d_k/2)

    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """
    Aplica RoPE a un tensor x.

    Args:
        x:   (..., seq_len, d_k)
        cos: (seq_len, d_k/2)
        sin: (seq_len, d_k/2)
    Returns:
        x rotado, mismo shape que x.
    """
    # split en dos mitades; convencion "split por chunks" — alternativas mas
    # adelante.
    x1, x2 = x.chunk(2, dim=-1)                    # cada uno: (..., seq_len, d_k/2)

    # rotacion 2D en cada par (x1[i], x2[i]) por angulo cos/sin precomputado
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    return torch.cat([rotated_x1, rotated_x2], dim=-1)
```

Linea por linea de `precompute_rope`:

- `torch.arange(0, d_k, 2)`: enteros pares 0, 2, 4, ..., d_k - 2. Hay `d_k/2` de ellos.
- `.float() / d_k`: los normaliza a `0/d_k, 2/d_k, ..., (d_k-2)/d_k`.
- `theta ** (...)`: eleva la base 10000 a esos exponentes. El `1.0 /` al frente da los $\theta_i$.
- `torch.outer(positions, freqs)`: matriz $(\text{max\_seq\_len}, d_k/2)$ donde la entrada $[p, i] = p \cdot \theta_i$ — los angulos que necesitamos.
- `cos()` y `sin()` precomputan los valores trigonometricos. Se hacen **una sola vez** al inicializar el modelo, no en cada forward.

Y `apply_rope`:

- Toma el tensor `x` (Q o K) y lo divide en dos mitades por la ultima dimension.
- Aplica la formula de rotacion 2D componente a componente.
- Reune las dos mitades de vuelta.

### Donde se aplica en el forward de attention

```python
# en lugar de
# Q = self.W_Q(x)
# K = self.W_K(x)
# scores = Q @ K.transpose(-2, -1) / sqrt(d_k)

# con RoPE
Q = self.W_Q(x)                                  # (B, T, d_k)
K = self.W_K(x)                                  # (B, T, d_k)
Q = apply_rope(Q, rope_cos[:T], rope_sin[:T])    # rotamos Q
K = apply_rope(K, rope_cos[:T], rope_sin[:T])    # rotamos K
# V queda intacto
V = self.W_V(x)
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
```

Las funciones `precompute_rope` se llaman al inicializar el modelo y producen tensores `cos` y `sin` de shape `(max_seq_len, d_k/2)`. En cada forward, se cortan al `T` actual y se aplican.

### Sobre la convencion del split

La convencion `chunk(2, dim=-1)` — partir el vector en mitad izquierda y mitad derecha — es la que usa LLaMA, GPT-NeoX y Hugging Face transformers. Hay otra convencion posible, "interleaved" (pares contiguos `(d_0, d_1), (d_2, d_3), ...`), que es la del paper original de Su et al. Las dos son matematicamente equivalentes — distintas permutaciones de las mismas dimensiones — pero **no son intercambiables** entre implementaciones. Si entrenas con una y haces inferencia con la otra, los pesos no calzan. Es una fuente clasica de bugs al portar checkpoints entre frameworks.

---

## 7. Por que solo Q y K, no V

Notese que en el forward de arriba, RoPE se aplica solo a `Q` y `K`. **El value `V` no se rota**. Por que?

Porque el rol de cada vector es distinto:

- `Q` y `K` se usan para **decidir relevancia**: cuanto atiende el token $i$ al token $j$. Es ahi donde la distancia importa: "que tan cerca esta este token del que estoy mirando ahora". Por eso los rotamos.
- `V` es **el contenido a transmitir** una vez que la relevancia esta decidida. No tiene que ser comparable con nada — solo tiene que ser informativo. La posicion no juega un rol en "que mensaje le entrego al modelo cuando me eligen".

Si rotaramos `V` tambien, estariamos diciendo "el contenido que entrega el token 47 es distinto del que entregaria si estuviera en posicion 48". Eso seria contraproducente: el contenido semantico de un token no deberia depender de su posicion absoluta. Lo que depende de la posicion es la **estructura** del matching (Q vs K), no el contenido a transmitir (V).

Esta separacion es muy pulcra: la asimetria identidad/contenido que ya vimos al introducir Q/K/V en el escalon 05 se profundiza con RoPE. K es la etiqueta posicionada con la que te encuentran; V es el mensaje atemporal que entregas.

{{< concept-alert type="clave" >}}
RoPE rota Q y K porque la posicion es relevante para el **matching** (decidir relevancia entre tokens). RoPE NO rota V porque V es el **contenido** a transmitir, y el contenido semantico de un token no debe depender de su posicion absoluta.
{{< /concept-alert >}}

---

## 8. Por que RoPE resuelve los 3 problemas

Volvamos a las tres limitaciones del comienzo y veamos como RoPE las elimina una por una.

### 8.1 No se diluye con la profundidad

Las codificaciones absolutas se aplican una sola vez, al inicio. RoPE se aplica **en cada capa** de attention, dentro de cada bloque de self-attention. Cada capa recibe una "inyeccion fresca" de informacion posicional sobre Q y K, antes de calcular los scores.

No importa que la representacion del token haya pasado por 32 capas: cuando llega a la capa 33, los Q y K se rotan de nuevo segun la posicion. La senal posicional **nunca** queda enterrada. Es accesible directamente en cada nivel de profundidad.

### 8.2 Extrapola a longitudes mucho mayores

La rotacion $R(p\theta)$ esta matematicamente bien definida para cualquier $p \in \mathbb{R}$. Si entrenaste con max\_seq\_len = 4096, no hay nada en la matematica que impida calcular $R(5000 \cdot \theta)$ — solo necesitas precomputar el cos/sin para esa posicion.

El modelo "entendera" parcialmente esa posicion porque las **frecuencias relativas** que ya aprendio siguen siendo validas. Tokens a distancia 5 son los mismos, esten en posiciones (10, 15) o en posiciones (4998, 5003).

Esto contrasta con codificaciones aprendidas, donde la posicion 5000 simplemente no tiene un vector definido. Y mejora sobre las sinusoidales, donde la senal esta presente pero diluida tras 32 capas.

En la practica, RoPE puede extrapolar 2-3x la longitud de entrenamiento sin tecnicas extra. Y con tecnicas como **NTK scaling**, **position interpolation** y **YaRN** (que veremos al final), puede llegar a 100x o mas: 4K -> 400K, 32K -> 2M. Por eso los modelos modernos de "long context" todos usan RoPE.

### 8.3 Es relativa por construccion

Como demostramos en la seccion 3, el dot product despues de RoPE depende solo de $(p_k - p_q)$, no de los valores absolutos. El modelo razona automaticamente en distancias relativas.

Eso significa que el patron "verbo atiende fuerte a su sujeto a 1-3 tokens de distancia" se aprende **una sola vez** y se aplica en cualquier posicion absoluta del corpus. Generalizacion masiva, gratis.

---

## 9. Por que LLaMA y todos los modernos lo adoptaron

Su et al. publicaron RoPE en 2021 ("RoFormer: Enhanced Transformer with Rotary Position Embedding"). Demostraron en benchmarks de NLP que RoPE supera consistentemente a las codificaciones absolutas y a otras codificaciones relativas (T5, ALiBi parcialmente):

- **Mejor perplexity** en language modeling.
- **Mejor extrapolacion** a longitudes superiores a las de entrenamiento.
- **Sin parametros adicionales** — el costo extra son los precomputos cos/sin, despreciables.
- **Compatible con la maquinaria existente** — no requiere reescribir el bloque de attention, solo modificar Q y K.

Cuando Meta entreno LLaMA en 2023, lo eligio sin discusion. Y de ahi el efecto cascada: GPT-NeoX, PaLM, Mistral, Mixtral, Qwen (Alibaba), Gemma (Google), Yi, DeepSeek, Phi-3, Llama-3 — **todos** usan RoPE. Es el estandar de facto de los Transformers decoder modernos.

GPT-2 (codificaciones aprendidas) y BERT (aprendidas) son parte del pasado. T5 (sesgos relativos) sobrevive en algunos contextos. RoPE domina el presente.

---

## 10. RoPE scaling: contextos > 100K

Una mencion breve, porque es importante para entender por que GPT-4-Turbo con 128K, Claude 3 con 200K, y Gemini 1.5 con 2M de contexto son posibles.

RoPE entrenado con max\_seq\_len = 4096 funciona "bien" hasta ~8K. Pero si quieres pasar a 128K, hay que **modificar las frecuencias** sin reentrenar todo desde cero. Tecnicas:

- **Position interpolation** (Chen et al., 2023): comprimir las posiciones por un factor (ej. dividir por 4) antes de aplicar RoPE. Tras unos miles de pasos de fine-tune, el modelo aprende a tratar 16K tokens como si fueran 4K reescalados. Funciona, pero pierde precision en distancias cortas.

- **NTK scaling** (sin paper formal, post de blog 2023): en lugar de comprimir uniforme, cambiar la base 10000 a un valor mayor. Esto comprime las frecuencias bajas (las que ya saturaban) y deja intactas las altas. Mejor preservacion de distancias cortas.

- **YaRN** (Peng et al., 2023): combinacion sofisticada de las dos anteriores con correcciones por frecuencia. Es lo que usa LLaMA-3.1 para llegar a 128K.

Lo crucial: ninguna de estas tecnicas funcionaria con codificaciones absolutas o aprendidas. Son posibles **porque RoPE codifica la posicion via angulos continuos, manipulables matematicamente**. Sin RoPE, no habria long context.

{{< concept-alert type="recordar" >}}
RoPE no solo es matematicamente mas elegante: es la pieza que hace posible long context. Las tecnicas modernas (interpolation, NTK, YaRN) que llevan los modelos a 100K-2M tokens dependen de poder manipular las frecuencias de RoPE — algo imposible con codificaciones aprendidas o sumadas.
{{< /concept-alert >}}

---

## 11. Pausa de verificacion

Antes de pasar al siguiente escalon (GQA), asegurate de que estas preguntas te quedan claras:

1. **Cual es la diferencia conceptual entre codificaciones sinusoidales tradicionales y RoPE?**
   Las sinusoidales se **suman** al embedding al inicio del modelo, una sola vez. RoPE **rota** los vectores Q y K dentro de cada capa de atencion, en cada bloque. La sinusoidal mezcla posicion y contenido en el mismo vector, y la senal se diluye con la profundidad. RoPE mantiene la posicion como una transformacion ortogonal aplicada repetidamente, y la senal no se diluye porque se reinyecta en cada capa. Ademas, RoPE produce un dot product que depende solo de la posicion relativa, mientras que las sinusoidales siguen siendo una codificacion absoluta.

2. **Por que el dot product despues de RoPE depende solo de la posicion relativa?**
   Porque las matrices de rotacion satisfacen $R(\alpha)^\top R(\beta) = R(\beta - \alpha)$. Cuando calculas el score entre $Q'_{p_q} = R(\theta p_q) q$ y $K'_{p_k} = R(\theta p_k) k$, en el producto $(Q')^\top K'$ aparece $R(\theta p_q)^\top R(\theta p_k)$ que se simplifica a $R(\theta(p_k - p_q))$. Las posiciones absolutas se cancelan algebraicamente, queda solo la diferencia.

3. **Por que solo Q y K se rotan, y no V?**
   Porque Q y K se usan para el **matching** (decidir cuanto atender entre tokens) y la posicion es relevante para el matching: que tan cerca estan los tokens. V es el **contenido** a transmitir cuando un token es seleccionado, y el contenido semantico no debe depender de la posicion absoluta. Rotar V seria distorsionar el mensaje a transmitir; no rotarlo preserva la separacion limpia entre "como te encuentran" (K, posicionada) y "que entregas" (V, atemporal).

---

## 12. Donde encaja RoPE en el panorama de modernizaciones

Recordando que estamos en la fase 5 (modernizaciones desde GPT-2 hasta LLaMA), RoPE es una de las tres piezas centrales junto con:

- **GELU/SwiGLU** (capitulo 13): mejor activacion en el FFN.
- **RMSNorm**: normalizacion mas barata que LayerNorm.
- **GQA** (capitulo 19, siguiente): atencion con menos memoria.

Cada una aporta un 1-3% de mejora individual a escala. Juntas convierten un mini-GPT estilo Vaswani-2017 en un mini-LLaMA estilo Meta-2023, y abren la puerta al long context que define los LLMs comerciales actuales.

RoPE es la mas matematicamente rica de todas. Es difici no admirarla: una sola idea geometrica (rotar pares 2D), una sola propiedad algebraica $(R^\top R = R(\beta - \alpha))$, sin parametros aprendibles, resuelve tres problemas distintos simultaneamente.

---

Codigo: `clase_14/practica/13_mini_llama.py` (funciones `precompute_rope` y `apply_rope`).

Siguiente: [19 - GQA](../19-gqa).
