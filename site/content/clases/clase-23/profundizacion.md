---
title: "Profundizacion - VQA e Image Captioning"
weight: 20
math: true
---

> La [teoría](/clases/clase-23/teoria) recorrió las slides; aquí desarrollamos la matemática que sostiene cada bloque. Cinco partes: (I) la top-down attention de Pythia, (II) la fusión multimodal simple vs bilineal (MCB, MUTAN), (III) el entrenamiento de VQA como clasificación multi-etiqueta, (IV) beam search formal, y (V) BLEU paso a paso.

---

## Parte I — Top-Down Attention sobre regiones

### El sustrato: features de regiones

Tras pasar la imagen por el detector (Mask R-CNN + ResNet-101), Pythia dispone de un conjunto de $K$ features de regiones:

$$V = \{v_1, v_2, \ldots, v_K\}, \qquad v_i \in \mathbb{R}^{2048}$$

El número $K$ puede ser fijo (p.ej. 36 o 100 regiones) o variable según las detecciones que superan un umbral. Cada $v_i$ es un descriptor semántico de un objeto/región (no de un parche arbitrario de la grilla, como en los modelos pre-2018). Este es el aporte **bottom-up** de [Anderson et al. 2018](/papers/bottom-up-attention-anderson-2018).

### La pregunta como query

La pregunta, tras [GloVe](/fundamentos/glove) + [GRU](/fundamentos/redes-recurrentes), es un vector:

$$q \in \mathbb{R}^{512}$$

Este $q$ actúa como **query** que selecciona qué regiones son relevantes — la parte **top-down** (la tarea guía la atención).

### El mecanismo de atención

Para cada región $v_i$, se calcula un score de atención no normalizado combinando la región con la pregunta. Siguiendo la formulación de BUTD:

{{< math-formula title="Score de atención por región" >}}
a_i = w_a^\top \, f_a\!\left( W_v\, v_i \odot W_q\, q \right)
{{< /math-formula >}}

donde $W_v \in \mathbb{R}^{h\times 2048}$ y $W_q \in \mathbb{R}^{h\times 512}$ proyectan ambas modalidades a un espacio común de dimensión $h$, $\odot$ es el producto Hadamard (elemento a elemento), $f_a$ es una no linealidad (en Pythia, **ReLU** con weight normalization; en BUTD original, *gated tanh*), y $w_a \in \mathbb{R}^h$ colapsa a un escalar.

Los scores se normalizan con softmax sobre las $K$ regiones:

$$\alpha_i = \frac{\exp(a_i)}{\sum_{j=1}^{K} \exp(a_j)}, \qquad \sum_{i=1}^K \alpha_i = 1$$

Y la imagen atendida es la suma ponderada (la **weighted sum** de la slide 12):

{{< math-formula title="Vector visual atendido" >}}
\hat{v} = \sum_{i=1}^{K} \alpha_i \, v_i \in \mathbb{R}^{2048}
{{< /math-formula >}}

{{< concept-alert type="clave" >}}
La interpretación es directa: $\alpha_i$ es **cuánto mira el modelo el objeto $i$** para responder esta pregunta. Para *"What is this?"* sobre el tren, la región del tren recibe $\alpha$ alto y el fondo (rieles, campo) recibe $\alpha$ bajo. Esto hace la atención **interpretable**: se puede dibujar el mapa de calor sobre la imagen.
{{< /concept-alert >}}

### Single-hop vs multi-hop

[Stacked Attention (Yang 2016)](/papers/stacked-attention-yang-2016) mostró que un solo paso de atención no basta para preguntas complejas. La idea multi-hop refina la query iterativamente:

$$u^{(0)} = q, \qquad u^{(\ell)} = \hat{v}^{(\ell)} + u^{(\ell-1)}$$

donde en cada hop $\ell$ se recalcula la atención usando la query refinada $u^{(\ell-1)}$. La primera capa localiza candidatos amplios; las siguientes concentran la atención. Pythia usa esencialmente un hop sobre regiones (más rico que el grid de SAN), apostando a que las regiones detectadas ya proveen la estructura que SAN debía construir a fuerza de hops.

---

## Parte II — Fusión multimodal: de lo simple a lo bilineal

Una vez que tenemos el vector de pregunta $q$ y el vector visual atendido $\hat{v}$, debemos **fusionarlos** en una representación conjunta antes de predecir la respuesta. Esta es una decisión de diseño central en VQA.

### Fusión simple (Pythia): producto Hadamard

Pythia proyecta ambos a $\mathbb{R}^{512}$ y los fusiona por producto elemento a elemento:

$$h = (W_q' \, q) \odot (W_v' \, \hat{v}) \in \mathbb{R}^{512}$$

(la slide 13 lo llama *Dot-Product*; en la práctica es Hadamard seguido de proyecciones). La virtud, en palabras de la clase, es *"mezclar la información multimodal sin aumentar la dimensión del modelo"*: el resultado vive en $\mathbb{R}^{512}$, igual que las entradas.

### El ideal: fusión bilineal completa

La fusión más expresiva captura **todas** las interacciones multiplicativas entre cada feature de $q$ y cada feature de $\hat{v}$ — el producto externo:

$$z = q \otimes \hat{v} \in \mathbb{R}^{d_q \times d_v}, \qquad z_{jk} = q_j \, \hat{v}_k$$

seguido de una proyección lineal por un tensor $\mathcal{T}$:

{{< math-formula title="Interacción bilineal completa" >}}
y = \mathcal{T} \times_1 q \times_2 \hat{v}, \qquad \mathcal{T} \in \mathbb{R}^{d_q \times d_v \times d_o}
{{< /math-formula >}}

El problema es la **explosión de parámetros**: con $d_q = d_v = 2048$ y $d_o = 3000$, el tensor $\mathcal{T}$ tiene $\sim 1.3 \times 10^{10}$ parámetros. Intratable. Por eso surge una familia de aproximaciones.

### MCB — aproximación por Count Sketch + FFT

[MCB (Fukui 2016)](/papers/mcb-fukui-2016) aproxima el producto externo proyectándolo a un espacio de dimensión $d \ll d_q d_v$ mediante **Count Sketch** $\Psi$, usando la propiedad de que el producto externo en el dominio sketch equivale a una **convolución**:

$$\Psi(q \otimes \hat{v}) = \Psi(q) * \Psi(\hat{v})$$

y la convolución se computa eficientemente vía FFT (teorema de convolución):

{{< math-formula title="Compact Bilinear Pooling vía FFT" >}}
\Psi(q) * \Psi(\hat{v}) = \mathrm{FFT}^{-1}\!\left( \mathrm{FFT}(\Psi(q)) \odot \mathrm{FFT}(\Psi(\hat{v})) \right)
{{< /math-formula >}}

Esto reduce el costo de $O(d_q d_v)$ a $O(d + d\log d)$. MCB ganó el **VQA Challenge 2016**.

### MUTAN — descomposición de Tucker

[MUTAN (Ben-younes 2017)](/papers/mutan-ben-younes-2017) controla la complejidad descomponiendo el tensor $\mathcal{T}$ con **Tucker**:

{{< math-formula title="Descomposición de Tucker" >}}
\mathcal{T} = \mathcal{T}_c \times_1 W_q \times_2 W_v \times_3 W_o
{{< /math-formula >}}

donde $\mathcal{T}_c \in \mathbb{R}^{t_q \times t_v \times t_o}$ es un **core tensor** pequeño y $W_q, W_v, W_o$ son matrices de factores que proyectan a/desde espacios latentes de dimensión controlable ($t_q, t_v, t_o$). Una restricción de **rango** $R$ adicional sobre las slices del core acota aún más los parámetros. MUTAN **generaliza** MLB y MCB: ambos resultan casos particulares de elegir cierta estructura del core.

{{< concept-alert type="contexto" >}}
**El espectro de fusión:** Pythia elige la fusión simple (Hadamard) por eficiencia y porque, con buenas features de regiones, basta. MCB/MUTAN exploran el otro extremo (interacción bilineal rica) a costa de complejidad. La era de los **Transformers cross-modales** (LXMERT, ViLBERT) eventualmente reemplazó toda esta familia con **cross-attention**, que modela interacciones par a par de forma aprendible y escalable.
{{< /concept-alert >}}

| Método | Interacción | Parámetros | Venue |
|---|---|---|---|
| Concatenación / suma | aditiva | bajos | baseline |
| Hadamard (Pythia) | multiplicativa elemento a elemento | bajos | VQA Challenge 2018 |
| Bilinear completo | todas las multiplicativas | $\sim d_q d_v d_o$ (intratable) | — |
| MCB | bilinear aproximado (sketch+FFT) | medios | VQA Challenge 2016 |
| MUTAN | bilinear vía Tucker + rango | controlables | ICCV 2017 |
| Cross-attention | par a par aprendible | escalable | 2019+ |

---

## Parte III — VQA como clasificación multi-etiqueta

### El target soft

Para una pregunta, VQAv2 provee 10 respuestas humanas. En lugar de una etiqueta dura, Pythia entrena contra un **score blando** por respuesta candidata, derivado de la métrica de consenso:

{{< math-formula title="Score de consenso (VQA accuracy)" >}}
\text{acc}(a) = \min\!\left( \frac{\#\{\text{humanos que respondieron } a\}}{3}, \; 1 \right)
{{< /math-formula >}}

Es decir: si al menos 3 de los 10 humanos dieron la respuesta $a$, cuenta como totalmente correcta ($1.0$); si menos, cuenta parcialmente. Esta métrica (de [Antol 2015](/papers/vqa-antol-2015)) reconoce que las respuestas abiertas admiten variantes válidas y exige consenso, no unanimidad.

### La pérdida

Como varias respuestas pueden ser parcialmente correctas, se usa **binary cross-entropy** sobre cada respuesta del vocabulario $\mathcal{A}$ (de ahí la **Sigmoid**, no Softmax):

{{< math-formula title="Pérdida de Pythia (BCE multi-etiqueta)" >}}
\mathcal{L} = -\sum_{a \in \mathcal{A}} \Big[ s_a \log \sigma(\hat{y}_a) + (1 - s_a)\log(1 - \sigma(\hat{y}_a)) \Big]
{{< /math-formula >}}

donde $s_a = \text{acc}(a)$ es el score blando objetivo y $\hat{y}_a$ el logit del modelo para la respuesta $a$. El vocabulario $|\mathcal{A}|$ se restringe a las ~3000 respuestas más frecuentes del training set.

{{< concept-alert type="alerta" >}}
**Aquí nace la limitación de "respuestas limitadas" (slide 19)**: el modelo solo puede predecir respuestas dentro de $\mathcal{A}$. Una especie de pájaro rara, ausente del vocabulario, es inalcanzable. Es la diferencia estructural con un modelo **generativo** (BLIP), que produce tokens de un vocabulario abierto y puede componer respuestas nuevas.
{{< /concept-alert >}}

### Por qué persisten los language priors

Aun con VQAv2 balanceado, el modelo minimiza $\mathcal{L}$ sobre la distribución de entrenamiento. Si una pregunta tipo *"¿hay un...?"* tiene mayoritariamente respuesta "yes" en los datos, el gradiente empuja hacia "yes" incluso cuando la imagen dice lo contrario. El balanceo **reduce** pero no elimina el sesgo: solo garantiza que para cada $(I,Q,A)$ existe un $(I',Q,A')$ con $A'\neq A$, no que el modelo aprenda a distinguirlos perfectamente.

---

## Parte IV — Beam Search formal

### El objetivo de decodificación

Generar un caption es buscar la secuencia $\mathbf{y} = (y_1, \ldots, y_T)$ que maximiza la probabilidad condicionada en la imagen:

{{< math-formula title="Secuencia óptima" >}}
\mathbf{y}^* = \arg\max_{\mathbf{y}} \; \prod_{t=1}^{T} p(y_t \mid y_{<t}, I) = \arg\max_{\mathbf{y}} \sum_{t=1}^{T} \log p(y_t \mid y_{<t}, I)
{{< /math-formula >}}

El espacio de secuencias es $|V|^T$ — exponencial. Buscar el óptimo exacto es intratable, así que se aproxima.

### Greedy como caso degenerado

Greedy (slide 24) toma el $\arg\max$ local en cada paso:

$$y_t = \arg\max_{w \in V} \; p(w \mid y_{<t}, I)$$

Esto es óptimo localmente pero no globalmente: $\prod_t \max_w p(w\mid\cdot) \neq \max_{\mathbf{y}} \prod_t p(\cdot)$. Una palabra subóptima ahora puede abrir una continuación mucho mejor.

### Beam search

Beam search (slides 25-26) mantiene un conjunto $\mathcal{B}_t$ de las $k$ secuencias parciales (hipótesis) más probables en cada paso $t$. El parámetro $k$ es el **beam width**. El algoritmo:

1. Inicializar $\mathcal{B}_0 = \{\langle \text{START} \rangle\}$.
2. En cada paso $t$: para cada hipótesis $\mathbf{y}_{<t} \in \mathcal{B}_{t-1}$ y cada palabra $w \in V$, calcular el score acumulado:
   $$\text{score}(\mathbf{y}_{<t} \oplus w) = \sum_{\tau \le t} \log p(y_\tau \mid y_{<\tau}, I)$$
3. Conservar las $k$ extensiones de mayor score → $\mathcal{B}_t$.
4. Repetir hasta `<END>`; devolver la hipótesis completa de mayor score.

El ejemplo de la slide 26 factoriza el caso de dos palabras:

$$P(y^1, y^2 \mid x) = p(y^1 \mid x) \cdot p(y^2 \mid x, y^1)$$

En lugar de fijar el mejor $y^1$ (greedy elegiría una sola rama), beam con $k=3$ conserva *"I"*, *"My"*, *"We"* como inicios, expande cada uno, y conserva las 3 secuencias globales más probables: *"I am"*, *"I will"*, *"My parents"*. Las ramas descartadas (la "X roja" del diagrama) habrían sido tomadas erróneamente por greedy.

### El sesgo de longitud

Como cada paso suma un $\log p \le 0$, las secuencias largas acumulan score más negativo. Beam search tiende entonces a **preferir captions cortos**. Se corrige normalizando por longitud:

$$\text{score}_{\text{norm}}(\mathbf{y}) = \frac{1}{T^\beta} \sum_{t=1}^{T} \log p(y_t \mid y_{<t}, I)$$

con $\beta \in [0.6, 1.0]$ un exponente de penalización de longitud. Ver el [fundamento de Decoding Strategies](/fundamentos/decoding-strategies) para sampling estocástico (top-k, nucleus, temperatura).

---

## Parte V — BLEU paso a paso

### Modified n-gram precision

El núcleo de BLEU ([Papineni 2002](/papers/bleu-papineni-2002)) es la **precisión de n-gramas modificada**. La precisión ingenua (fracción de n-gramas del candidato presentes en la referencia) se rompe: el candidato *"the the the the"* contra la referencia *"the cat is on the mat"* tendría precisión $4/4=1$. El **clipping** lo arregla: cada n-grama cuenta como máximo las veces que aparece en la referencia.

{{< math-formula title="Modified n-gram precision" >}}
p_n = \frac{\displaystyle\sum_{C \in \text{cand}} \;\sum_{\text{ng} \in C} \min\big(\text{Count}(\text{ng}),\; \text{Count}_{\text{ref}}^{\max}(\text{ng})\big)}{\displaystyle\sum_{C \in \text{cand}} \;\sum_{\text{ng} \in C} \text{Count}(\text{ng})}
{{< /math-formula >}}

**Ejemplo** ("the the the the" vs "the cat is on the mat"): "the" aparece 4 veces en el candidato, pero solo 2 en la referencia → cuenta clippeada $= 2$. Total candidato $= 4$. Entonces $p_1 = 2/4 = 0.5$ (no $1.0$).

### Combinación geométrica

BLEU combina $p_1, p_2, p_3, p_4$ con un **promedio geométrico** ponderado (típicamente $w_n = 1/4$):

$$\exp\left( \sum_{n=1}^{4} w_n \log p_n \right) = \left( p_1 p_2 p_3 p_4 \right)^{1/4}$$

¿Por qué geométrico y no aritmético? Porque la precisión decae exponencialmente con $n$ (es más difícil acertar 4-gramas que unigramas), y el promedio geométrico es **severo con los ceros**: si algún $p_n = 0$, todo el producto se anula. Esto fuerza al candidato a acertar n-gramas de todos los órdenes.

### Brevity Penalty

La precisión no penaliza candidatos **demasiado cortos** (no hay recall con múltiples referencias). El candidato *"the cat"* contra *"the cat is on the mat"* tendría $p_1 = p_2 = 1$ — perfecto, pero incompleto. La **brevity penalty** lo castiga:

{{< math-formula title="Brevity Penalty" >}}
BP = \begin{cases} 1 & \text{si } c > r \\ e^{\,1 - r/c} & \text{si } c \le r \end{cases}
{{< /math-formula >}}

donde $c$ es la longitud del candidato y $r$ la longitud de referencia efectiva. Si el candidato es más corto que la referencia ($c \le r$), $BP < 1$ lo penaliza exponencialmente.

### La fórmula final

{{< math-formula title="BLEU" >}}
\text{BLEU} = BP \cdot \exp\left( \sum_{n=1}^{N} w_n \log p_n \right)
{{< /math-formula >}}

**Ejemplo completo.** Candidato *"the cat sat"* ($c=3$), referencia *"the cat sat on the mat"* ($r=6$). Supongamos $p_1=1$, $p_2=1$, $p_3=1$, $p_4$ indefinido (usemos $N=3$, $w_n=1/3$):

- Promedio geométrico: $(1\cdot1\cdot1)^{1/3} = 1$.
- $BP = e^{1-6/3} = e^{-1} \approx 0.368$.
- $\text{BLEU} = 0.368 \cdot 1 = 0.368$.

El caption es preciso pero corto, y BLEU lo penaliza correctamente.

### Corpus-level vs sentence-level

BLEU se diseñó para calcularse **a nivel de corpus**: se acumulan los numeradores y denominadores de todas las oraciones antes de dividir, no se promedia el BLEU por oración. A nivel de oración individual, BLEU es ruidoso (un solo n-grama faltante puede dar $p_n=0$ y anular todo), por lo que se usan variantes con *smoothing*.

{{< concept-alert type="alerta" >}}
**Limitaciones de BLEU** (la slide 27: *"no evalúa la calidad de la traducción"*): mide coincidencia superficial de n-gramas, no significado. Penaliza paráfrasis válidas (*"el auto"* vs *"el coche"*), es insensible al orden más allá del n-grama, y no captura adecuación ni fluidez semántica. En captioning premia repetir palabras frecuentes de COCO. Esto motivó **CIDEr** (pondera n-gramas por TF-IDF e incorpora consenso), **METEOR** (alineación con sinónimos y stems) y **SPICE** (grafos de escena semánticos), además de métricas neuronales como BERTScore. Ver el contraste con [ROUGE](/fundamentos/rouge-metric) (recall-oriented) en el [fundamento de BLEU](/fundamentos/bleu-metric).
{{< /concept-alert >}}

---

## Síntesis matemática

| Bloque | Operación clave | Resultado |
|---|---|---|
| Top-down attention | $\alpha = \text{softmax}(a)$, $\hat{v} = \sum_i \alpha_i v_i$ | imagen vista a través de la pregunta |
| Fusión simple | $h = (W_q' q) \odot (W_v' \hat{v})$ | representación conjunta sin explosión dimensional |
| Fusión bilineal | Tucker / Count Sketch+FFT | interacción rica controlando parámetros |
| Clasificación VQA | BCE multi-etiqueta + Sigmoid | scores blandos sobre vocabulario cerrado |
| Beam search | mantener top-$k$ por $\sum \log p$ | mejor secuencia que greedy |
| BLEU | $BP \cdot \exp(\sum_n w_n \log p_n)$ | similitud de n-gramas con referencias |

## Para seguir

- [Teoría de la clase](/clases/clase-23/teoria) — el recorrido conceptual de las slides.
- [Fundamento: Visual Question Answering](/fundamentos/visual-question-answering), [Image Captioning](/fundamentos/image-captioning), [BLEU Metric](/fundamentos/bleu-metric).
- Papers: [Pythia](/papers/pythia-jiang-2018), [Bottom-Up/Top-Down](/papers/bottom-up-attention-anderson-2018), [MCB](/papers/mcb-fukui-2016), [MUTAN](/papers/mutan-ben-younes-2017), [BLEU](/papers/bleu-papineni-2002).
- [Laboratorio 23](/laboratorios/lab-23) — práctica con BLIP.
