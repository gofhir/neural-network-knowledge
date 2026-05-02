# Analisis del Paper: A simple neural network module for relational reasoning

**Autores**: Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap
**Institucion**: DeepMind, London, UK
**Publicado en**: NeurIPS 2017 (arXiv:1706.01427, junio 2017)
**Paginas**: 16 (9 paginas paper + 7 paginas supplementary)

> PDF: [relation-networks-santoro-2017.pdf](relation-networks-santoro-2017.pdf)

---

## 1. Resumen ejecutivo

Las **Relation Networks (RN)** son un modulo neural simple, plug-and-play, dedicado a razonamiento relacional. Su forma funcional es:

$$RN(O) = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j)\right)$$

donde $O$ es un conjunto de "objetos", $g_\theta$ es una MLP que computa la relacion entre cada par $(o_i, o_j)$, y $f_\phi$ es una MLP que agrega los resultados. Acoplado a CNN (vision) y LSTM (lenguaje), un modelo CNN+LSTM+RN logra:

- **CLEVR pixel**: 95.5% (super-humano, humano = 92.6%; SOTA previa = 76.6%).
- **CLEVR state**: 96.4%.
- **bAbI**: 18/20 tareas con >95% accuracy.
- **Sort-of-CLEVR relacional**: 94% (vs 63% de CNN+MLP).
- **Sistemas fisicos**: 93% en inferencia de conexiones, 95% en conteo de sistemas.

La contribucion central no es el desempeno -- es la **formalizacion del bias relacional** como building block reutilizable. RN es precursor conceptual de self-attention y un caso particular de Graph Neural Networks (message passing en grafo completo).

---

## 2. Motivacion

### 2.1. Razonamiento relacional como subsistema separable

Hipotesis: el razonamiento relacional debe ser un **modulo dedicado**, igual que la convolucion captura invariancia translacional o las RNN capturan dependencias secuenciales. La filosofia de diseno es **constreñir la forma funcional** de la red para que codifique un bias inductivo concreto -- en este caso, "considerar relaciones par-a-par".

### 2.2. Estado del arte previo (2016-2017)

- **CNNs/MLPs**: poderosos en percepcion local, pero fallan en preguntas que requieren comparar objetos arbitrarios (ej. CLEVR compare-attribute: ~52% solo con CNN+LSTM).
- **Memory Networks, DNC, Sparse DNC**: arquitecturas augmentadas con memoria externa. Resuelven bAbI parcialmente (14-19/20).
- **Stacked Attention Networks, attention modules**: ponderan regiones pero sin estructura par-a-par explicita.
- **Symbolic approaches**: precisos en relaciones pero sufren del symbol grounding problem.

RN se posiciona como un modulo **mas simple y mas general** que las anteriores: no requiere memoria externa ni atencion explicita.

---

## 3. Definicion formal

### 3.1. Forma simple

Para un conjunto de objetos $O = \{o_1, \dots, o_n\}$, $o_i \in \mathbb{R}^m$:

$$RN(O) = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j)\right)$$

- $g_\theta : \mathbb{R}^{2m} \to \mathbb{R}^{d_g}$: MLP compartida que mapea cada par a un vector "relacion".
- $f_\phi : \mathbb{R}^{d_g} \to \mathbb{R}^{d_{\text{out}}}$: MLP que produce la salida (ej. logits de respuesta).
- La suma es sobre **todos** los pares ordenados (incluyendo $i=j$).

### 3.2. Variante con condicionamiento

Cuando hay informacion adicional como una pregunta $q$:

$$RN(O, q) = f_\phi\left(\sum_{i,j} g_\theta(o_i, o_j, q)\right)$$

$g_\theta$ ahora recibe el par concatenado con $q$. Esto permite que la **misma escena** produzca respuestas distintas segun la pregunta.

### 3.3. Por que esta forma

**Tres consecuencias estructurales** del diseno:

1. **Inferir relaciones**: La red no recibe los pares relevantes; debe aprenderlos. $g_\theta$ produce vectores cercanos al cero para pares no informativos.
2. **Eficiencia de datos**: Una sola MLP $g_\theta$ se entrena con $n^2$ ejemplos por imagen (un mini-batch interno de pares). Una MLP global tendria que aprender $n^2$ funciones identicas embebidas en sus pesos.
3. **Permutation invariance**: $\sum$ es simetrica $\Rightarrow$ el orden de los objetos en $O$ no afecta el output.

---

## 4. CLEVR (visual QA con razonamiento 3D)

### 4.1. La tarea

CLEVR (Johnson et al. 2017) contiene 100K imagenes 3D-rendered con esferas, cubos y cilindros de distintos materiales, colores y tamanos, mas ~1M preguntas generadas programaticamente: query attribute, compare attribute, count, exist, compare numbers. Las preguntas relacionales son la mayoria.

### 4.2. Pipeline CNN+LSTM+RN

```text
Imagen 128x128
   │
   ▼
CNN (4 capas, 24 kernels c/u, ReLU + BatchNorm)
   │
   ▼
Feature maps d×d×24
   │
   ▼  (cada celda + coords (x,y) = "objeto" o_i)
   │
[o_1, o_2, ..., o_{d²}]
   │
   ▼  (par (o_i, o_j) || q)
   │
g_θ MLP (4 capas, 256 unidades, ReLU)
   │
   ▼
Σ_{i,j}
   │
   ▼
f_φ MLP (3 capas: 256, 256, 29 con dropout 50%)
   │
   ▼
softmax → respuesta

Pregunta → LSTM (128 units) → q
```

### 4.3. Resultados

| Modelo | Overall | Count | Exist | Compare Num. | Query Attr. | Compare Attr. |
|---|---|---|---|---|---|---|
| Human | 92.6 | 86.7 | 96.6 | 86.5 | 95.0 | 96.0 |
| Q-type baseline | 41.8 | 34.6 | 50.2 | 51.0 | 36.0 | 51.3 |
| CNN+LSTM | 52.3 | 43.7 | 65.2 | 67.1 | 49.3 | 53.0 |
| CNN+LSTM+SA | 76.6 | 64.4 | 82.7 | 77.4 | 82.6 | 75.4 |
| **CNN+LSTM+RN** | **95.5** | **90.1** | **97.8** | **93.6** | **97.9** | **97.1** |

Observaciones:

- RN supera a humanos en overall (95.5 vs 92.6).
- La mejora mas grande es en **compare attribute** (53 -> 97), que es la categoria mas relacional.
- El gap entre CNN+LSTM (52.3) y CNN+LSTM+RN (95.5) -- 43 puntos -- aisla la contribucion del modulo RN.
- Stacked Attention solo cierra parte del gap; explicit pair-wise relations cierran el resto.

### 4.4. CLEVR from state descriptions

Sin imagen: cada objeto se entrega como vector (3D coords, color, shape, material, size). RN logra **96.4%**, mostrando que el modulo es agnostico al input.

---

## 5. bAbI (text QA, 20 tareas)

### 5.1. Setup

bAbI es un suite de 20 tareas de razonamiento textual (deduccion, induccion, conteo, etc). Para cada pregunta, hay <=20 oraciones de support set.

**Codificacion como objetos**: cada oracion del support set se procesa con un LSTM (32 units) cuyo estado final es el "objeto". La pregunta se procesa con un LSTM separado para producir $q$. RN combina los $\le 20^2 = 400$ pares.

### 5.2. Resultados

| Modelo | Tareas pasadas (>95%) |
|---|---|
| Memory Networks | 14/20 |
| EntNet | 16/20 (joint training) |
| DNC | 18/20 |
| Sparse DNC | 19/20 |
| **RN** | **18/20** |

RN no falla catastroficamente en ninguna tarea (las dos que fallan son por margen pequeno: 3.1% y 11.5% bajo 95%). Notablemente, RN resuelve la tarea de induccion basica (2.1% error) que Sparse DNC, DNC y EntNet fallan (54-55% error).

---

## 6. Sort-of-CLEVR (control sintetico)

### 6.1. Diseno

Dataset construido para **aislar** la dificultad relacional vs no-relacional. 10000 imagenes 75x75 con 6 objetos 2D (cuadrados o circulos) en 6 colores distintos. 20 preguntas por imagen: 10 relacionales (closest-to, furthest-from, count) y 10 no-relacionales (query shape, query horizontal/vertical position).

### 6.2. Resultados

| Modelo | No-relacional | Relacional |
|---|---|---|
| CNN+RN | ~99% | **94%** |
| CNN+MLP | ~99% | 63% |

**Conclusion clave**: con la misma capacidad y mismo CNN, **CNN+MLP falla en relacionales**. El bias arquitectural -- pair-wise sum -- es lo que cierra la brecha.

---

## 7. Analisis: por que $\sum$ y no concatenacion

### 7.1. Permutation invariance

Concatenar todos los objetos $[o_1; o_2; \dots; o_n]$ en una MLP gigante introduce un orden arbitrario. La MLP debe **aprender** a ser invariante al orden -- gastando capacidad y datos. La suma garantiza la invariancia **por construccion**.

### 7.2. Deep Sets equivalent

Zaheer et al. 2017 ("Deep Sets") prueban que cualquier funcion permutation-invariant sobre conjuntos puede expresarse como $f(\sum_i \phi(o_i))$. La RN extiende esta forma a relaciones binarias: $f_\phi(\sum_{i,j} g_\theta(o_i, o_j))$. Por construccion, RN es un universal approximator sobre funciones simetricas binarias de conjuntos.

### 7.3. Compartir $g_\theta$

Una MLP global con todos los pares en input tendria que **replicar** $n^2$ veces la misma funcion de relacion en sus pesos. RN aprende una sola $g_\theta$ que se aplica $n^2$ veces. Eficiencia parametrica y de datos.

---

## 8. Conexion con Transformers

### 8.1. Self-attention reexpresado

La capa de self-attention (Vaswani et al. 2017) computa:

$$\text{Attn}(X)_i = \sum_j \alpha_{ij} (W^V x_j), \quad \alpha_{ij} = \text{softmax}_j\left(\frac{(W^Q x_i)^T (W^K x_j)}{\sqrt{d_k}}\right)$$

### 8.2. RN reexpresado

$$RN(X)_i = \sum_j g_\theta(x_i, x_j)$$

(version "por objeto", agregando solo sobre $j$).

### 8.3. RN como caso de self-attention con $g_\theta$ aprendido

Si definimos $g_\theta(x_i, x_j) := \alpha_{ij}(x_i, x_j) \cdot W^V x_j$, donde $\alpha$ es la softmax aprendida via $Q, K$, entonces self-attention **es** una RN con una forma particular de $g_\theta$. Diferencias:

| Aspecto | RN | Self-attention |
|---|---|---|
| Peso del par | Uniforme (todos cuentan igual) | $\alpha_{ij}$ aprendido |
| Forma de $g_\theta$ | MLP libre sobre $(x_i, x_j)$ | Inner-product $Q,K$ + value $V$ |
| Complejidad | $O(n^2)$ con MLP | $O(n^2)$ con producto matricial |
| Cabezas multiples | No | Multi-head |
| Profundidad | Una capa de relaciones | Apilable en stack |

**Conclusion**: la RN comparte el **bias relacional** (pair-wise sum) con self-attention; el Transformer escala esa idea con pesos aprendidos, multi-head y stacking.

---

## 9. Conexion con Graph Neural Networks

### 9.1. Message passing

GNNs computan, para cada nodo $v$:

$$h_v' = U\left(h_v, \sum_{u \in \mathcal{N}(v)} M(h_v, h_u)\right)$$

con $M$ funcion de mensaje y $U$ funcion de actualizacion.

### 9.2. RN como GNN sobre grafo completo

Si interpretamos los objetos como nodos y conectamos todos con todos ($\mathcal{N}(v) = $ todos), entonces:

- $g_\theta = M$ (funcion de mensaje aprendible).
- $\sum_{i,j}$ = agregacion de mensajes (suma sobre todas las aristas).
- $f_\phi$ = readout global (funcion de output del grafo).

RN = **GNN con grafo completamente conectado** y readout sumado. La diferencia con GNNs estandar (Kipf 2016, Battaglia 2016 Interaction Networks) es la asuncion de "todos con todos" en vez de un grafo dado a priori.

---

## 10. Lecciones transferibles

1. **Bias relacional explicito**: si el problema involucra relaciones par-a-par, no asumas que una MLP las aprendera. Inyectar $\sum_{i,j} g(o_i, o_j)$ es barato y efectivo.
2. **Permutation invariance por construccion**: usar $\sum$ + MLP compartida en lugar de concatenacion + MLP gigante.
3. **Composicion modular**: CNN para percepcion local + LSTM para lenguaje + RN para relaciones. Cada modulo aporta un bias inductivo distinto y se entrenan **end-to-end**.
4. **Induccion de objetos**: la CNN aguas arriba aprende, **sin supervision explicita**, a producir representaciones que la RN puede usar como objetos. La presion downstream del RN moldea la representacion upstream.

---

## 11. Limitaciones

### 11.1. Complejidad cuadratica

Para $n$ objetos hay $n^2$ pares. En CLEVR con feature map $8 \times 8 = 64$ celdas, eso son 4096 evaluaciones de $g_\theta$ por imagen. Para $n$ grande (cientos o miles de objetos) se vuelve intratable. Trabajos posteriores exploran sparsificar (Sparse RN, attention para filtrar pares).

### 11.2. Solo binario

$g_\theta$ ve **dos** objetos a la vez. Relaciones ternarias o de mayor aridad ("X esta entre Y y Z") deben ser inferidas indirectamente vias compositions de pares. No hay $g_\theta(o_i, o_j, o_k)$ explicito (escalaria a $n^3$).

### 11.3. Suma uniforme

Todas las relaciones contribuyen igual al pooling final. Self-attention, con pesos aprendidos $\alpha_{ij}$, es estrictamente mas expresivo en pooling.

### 11.4. Requiere "objetos"

En pixeles, el grid de la CNN hace las veces de objetos. Esto funciona en CLEVR (objetos discretos sobre fondo limpio) pero puede fallar en escenas naturales donde los objetos no estan alineados al grid. Trabajos posteriores (slot attention, object-centric learning) abordan esto explicitamente.

### 11.5. Falta profundidad relacional

RN tiene **una sola capa** de relaciones (un sumatorio + un MLP $f_\phi$). No itera. Transformers stackean atencion en multiples capas, refinando representaciones progresivamente; GNNs iteran message passing en multiples rondas. Esta limitacion explica parte del gap respecto a Transformers en tareas con cadenas largas de razonamiento.

---

## 12. Legado

```text
2017  ★ Relation Networks (Santoro et al.) ★
        Pair-wise sum como bias relacional explicito

2017  Attention Is All You Need (Vaswani et al.)
        Self-attention = RN con pesos aprendidos
        Multi-head + stacking + positional encoding

2017  Deep Sets (Zaheer et al.)
        Formaliza permutation invariance: f(Σ φ(x_i))

2018  Relational Inductive Biases (Battaglia et al.)
        Unifica RN, GNN, Interaction Networks bajo
        un framework de "graph network"

2019+ GNNs aplicadas a grafos moleculares,
      sociales, y razonamiento knowledge graph

2020+ Transformers dominan NLP, vision (ViT) y
      multimodal -- todos descendientes conceptuales
      del bias pair-wise + permutation invariance
```

RN es una pieza **conceptualmente fundamental** en la genealogia de las arquitecturas modernas centradas en relaciones. Aunque hoy se usa poco directamente -- self-attention lo subsume -- entender RN ayuda a entender **por que** los Transformers funcionan: el bias inductivo de "considerar todos los pares" estaba ya presente en este paper, y los Transformers simplemente lo escalaron con pesos aprendidos y profundidad.
