# Analisis: Attention Is All You Need (Vaswani et al. 2017)

**Autores**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
**Instituciones**: Google Brain, Google Research, University of Toronto
**Publicado en**: 31st Conference on Neural Information Processing Systems (NeurIPS 2017), Long Beach, CA
**arXiv**: 1706.03762 (subido en junio 2017)

> PDF: [attention-is-all-you-need-vaswani-2017.pdf](attention-is-all-you-need-vaswani-2017.pdf)

---

## Resumen Ejecutivo

El paper introduce el **Transformer**, una arquitectura encoder-decoder sin recurrencia ni convolucion, basada exclusivamente en mecanismos de atencion. Sus contribuciones tecnicas son: (i) **scaled dot-product attention** con factor de escala $1/\sqrt{d_k}$, (ii) **multi-head attention** que proyecta a $h$ subespacios paralelos, (iii) **positional encoding sinusoidal** que inyecta orden sin parametros entrenables, y (iv) un **stack de $N=6$ capas** con residuales y layer normalization. El modelo obtiene 28.4 BLEU en WMT'14 EN-DE y 41.0 BLEU en EN-FR, batiendo a todos los baselines (incluidos ensembles) entrenando en 3.5 dias sobre 8 GPUs P100. Es el paper fundacional de la era moderna de foundation models: BERT, GPT, T5, ViT y AlphaFold descienden directamente de esta arquitectura.

---

## 1. Aporte Central

### 1.1. Que propone

Reemplazar **completamente** los bloques recurrentes (LSTM, GRU) y convolucionales (ConvS2S, ByteNet) por bloques de **self-attention**. La hipotesis fuerte es: si la atencion ya esta haciendo el trabajo pesado en seq2seq+attention, la recurrencia es prescindible.

### 1.2. Por que es innovador en 2017

El estado del arte previo (GNMT, ConvS2S, MoE) anadia atencion **encima** de un backbone recurrente o convolucional. Vaswani et al. dieron el paso conceptual: la atencion no es un complemento, es **el bloque elemental**. Esto desbloqueo:

| Propiedad | RNN/LSTM | CNN | Transformer |
|---|---|---|---|
| Paralelizable en train | NO (secuencial en $t$) | SI | SI |
| Distancia efectiva token-a-token | $O(n)$ | $O(\log_k n)$ | $O(1)$ |
| Dependencias largas | Dificil (vanishing) | Dificil (kernel chico) | Trivial |
| Saturacion de aceleradores | Pobre | Buena | Excelente |

El "$O(1)$ path length" es el punto fundamental: cualquier par de tokens se conecta directamente en una sola capa, lo que facilita el aprendizaje de dependencias largas y elimina el vanishing gradient propio de las RNNs profundas.

---

## 2. Arquitectura Detallada

### 2.1. Hiperparametros del modelo base

| Simbolo | Valor | Significado |
|---|---|---|
| $N$ | 6 | numero de capas en encoder y decoder |
| $d_{model}$ | 512 | dimension de embeddings y residual stream |
| $d_{ff}$ | 2048 | dimension interna del FFN |
| $h$ | 8 | numero de cabezas de atencion |
| $d_k = d_v$ | 64 | dimension por cabeza ($d_{model}/h$) |
| $P_{drop}$ | 0.1 | dropout |
| $\epsilon_{ls}$ | 0.1 | label smoothing |
| Parametros totales | 65 M | base model |

Big model: $d_{model}=1024$, $d_{ff}=4096$, $h=16$, $P_{drop}=0.3$, 213 M parametros.

### 2.2. Encoder layer

```text
x -> [LayerNorm(x + MultiHeadSelfAttn(x))] -> [LayerNorm(. + FFN(.))] -> output
     |__________sub-capa 1__________|       |________sub-capa 2_______|
```

- **Sub-capa 1**: self-attention donde $Q = K = V = $ output de la capa anterior.
- **Sub-capa 2**: FFN posicional.
- Cada sub-capa con conexion residual + layer norm.
- Todas las representaciones internas son de dimension $d_{model} = 512$ para que las residuales sumen.

### 2.3. Decoder layer

Tres sub-capas:

1. **Masked self-attention**: causal mask que pone $-\infty$ en las posiciones $j > i$ antes del softmax, garantizando autoregresion.
2. **Cross-attention encoder-decoder**: $Q$ del decoder, $K, V$ del output del encoder.
3. **FFN posicional**.

### 2.4. Tres usos de atencion en el modelo

| Donde | Q | K, V | Comentario |
|---|---|---|---|
| Encoder self-attn | encoder layer prev | encoder layer prev | bidireccional, no enmascarada |
| Decoder self-attn | decoder layer prev | decoder layer prev | enmascarada (causal) |
| Cross-attn | decoder layer prev | output del encoder final | clasica seq2seq |

---

## 3. Math Derivada Paso a Paso

### 3.1. Por que $\sqrt{d_k}$: derivacion de varianza

**Setup**: asumimos componentes de $q, k \in \mathbb{R}^{d_k}$ con media 0 y varianza 1, independientes.

El producto punto es:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

**Esperanza**: $\mathbb{E}[q \cdot k] = \sum_i \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$.

**Varianza**: usando independencia,

$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_i \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] = d_k \cdot 1 \cdot 1 = d_k$$

Es decir, la desviacion estandar crece como $\sqrt{d_k}$.

**Problema para softmax**: si los logits tienen magnitud $\sim \sqrt{d_k}$, el softmax saturara a una distribucion casi delta, y su gradiente:

$$\frac{\partial \text{softmax}(x)_i}{\partial x_j} = \text{softmax}(x)_i (\delta_{ij} - \text{softmax}(x)_j)$$

se vuelve practicamente cero (un termino cerca de 1, los otros cerca de 0). Esto mata el aprendizaje.

**Solucion**: dividir por $\sqrt{d_k}$ restablece varianza 1:

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

Por eso la formula del paper es:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Sin el escalado, el modelo no entrena para $d_k$ grande (lo confirma la nota de pie de pagina 4 del paper).

### 3.2. Multi-head: por que proyectar a subespacios

Una sola atencion sobre vectores $d_{model}$-dimensionales **promedia** linealmente los values segun la distribucion softmax. Si una sola distribucion debe capturar simultaneamente "concordancia sujeto-verbo" + "anafora" + "estructura sintactica" + "semantica lexica", el promedio diluye toda la informacion.

La solucion: aprender $h$ proyecciones $W_i^Q, W_i^K, W_i^V$ a un subespacio de dimension $d_k = d_{model}/h$, atender en paralelo, concatenar y proyectar de vuelta.

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \in \mathbb{R}^{n \times d_v}$$

$$\text{MultiHead}(Q,K,V) = [\text{head}_1; \ldots; \text{head}_h] W^O$$

**Costo computacional**: cada cabeza opera en dimension $d_k = d_{model}/h$, asi que el costo total es:

$$h \cdot O(n^2 \cdot d_k) = h \cdot O(n^2 \cdot d_{model}/h) = O(n^2 \cdot d_{model})$$

Es decir, **el mismo costo que una atencion de cabeza unica** con dimension $d_{model}$. La factorizacion en cabezas es gratis en compute pero gana expresividad.

Las visualizaciones del apendice (Figuras 3-5) muestran cabezas que se especializan: una en dependencias sintacticas (sujeto-verbo), otra en anafora (resolucion de "its"), otra en delimitacion de frases.

### 3.3. Positional encoding sinusoidal: prueba de linealidad

La eleccion de senos y cosenos no es estetica. Los autores argumentan: "para cualquier offset fijo $k$, $PE_{pos+k}$ puede representarse como una funcion lineal de $PE_{pos}$".

**Demostracion**. Para un par de coordenadas $(2i, 2i+1)$ con frecuencia $\omega_i = 1/10000^{2i/d_{model}}$:

$$PE_{pos} = \begin{pmatrix} \sin(\omega_i \cdot pos) \\ \cos(\omega_i \cdot pos) \end{pmatrix}$$

Por las identidades trigonometricas:

$$\sin(\omega_i (pos + k)) = \sin(\omega_i pos)\cos(\omega_i k) + \cos(\omega_i pos)\sin(\omega_i k)$$

$$\cos(\omega_i (pos + k)) = \cos(\omega_i pos)\cos(\omega_i k) - \sin(\omega_i pos)\sin(\omega_i k)$$

Que en forma matricial:

$$PE_{pos+k} = \underbrace{\begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix}}_{R_{\omega_i k}} \cdot PE_{pos}$$

Es decir, **avanzar $k$ posiciones es una rotacion** en el plano $(2i, 2i+1)$ por un angulo proporcional a $k$. La transformacion es lineal y depende solo de $k$ (no de $pos$).

Esto significa que un modelo lineal sobre $PE$ puede aprender trivialmente a "atender al token a $k$ posiciones de distancia" -- no necesita memorizar embeddings posicionales separados para cada posicion absoluta.

(Esta misma idea es la semilla intelectual de RoPE, Rotary Position Embeddings, Su et al. 2021, que la lleva a las queries/keys directamente en lugar de a los embeddings.)

---

## 4. Tabla 1 del Paper: Complejidades Comparadas

| Layer Type | Complexity per Layer | Sequential Ops | Max Path Length |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-Attention (restricted, ventana $r$) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

**Lectura clave**:

- Self-attention es **cuadratica en $n$** pero **lineal en $d$**. Recurrent es lineal en $n$ pero **cuadratica en $d$**. En NMT tipico $n \sim 100$ y $d = 512$, asi que $n^2 d = 5 \cdot 10^6$ vs $n d^2 = 2.6 \cdot 10^7$ -- self-attention es **5x mas barata** en este regimen.
- "Sequential ops" mide cuantos pasos no paralelizables hay. RNN: $O(n)$ (devastador para GPU). Atencion: $O(1)$.
- "Max path length" mide cuantas hops debe dar una senal entre dos tokens cualesquiera. RNN: $O(n)$ (de aqui el vanishing). Atencion: $O(1)$.

---

## 5. Tabla 2 del Paper: Resultados WMT'14

| Modelo | BLEU EN-DE | BLEU EN-FR | FLOPs EN-DE | FLOPs EN-FR |
|---|---|---|---|---|
| ByteNet | 23.75 | -- | -- | -- |
| Deep-Att + PosUnk | -- | 39.2 | -- | $1.0 \cdot 10^{20}$ |
| GNMT + RL | 24.6 | 39.92 | $2.3 \cdot 10^{19}$ | $1.4 \cdot 10^{20}$ |
| ConvS2S | 25.16 | 40.46 | $9.6 \cdot 10^{18}$ | $1.5 \cdot 10^{20}$ |
| MoE | 26.03 | 40.56 | $2.0 \cdot 10^{19}$ | $1.2 \cdot 10^{20}$ |
| GNMT + RL Ensemble | 26.30 | 41.16 | $1.8 \cdot 10^{20}$ | $1.1 \cdot 10^{21}$ |
| ConvS2S Ensemble | 26.36 | **41.29** | $7.7 \cdot 10^{19}$ | $1.2 \cdot 10^{21}$ |
| **Transformer (base)** | 27.3 | 38.1 | $\mathbf{3.3 \cdot 10^{18}}$ | $\mathbf{3.3 \cdot 10^{18}}$ |
| **Transformer (big)** | **28.4** | **41.0** | $2.3 \cdot 10^{19}$ | $2.3 \cdot 10^{19}$ |

**Observaciones**:

- **Base** ya supera a todos los modelos individuales con **3-30x menos compute**.
- **Big** supera a todos los ensembles, igualando a ConvS2S Ensemble en EN-FR con **50x menos compute**.
- La eficiencia es la historia tan importante como el BLEU. En 2017 esto significo "puedes entrenar el SOTA en 3.5 dias en una sola maquina con 8 GPUs P100", democratizando NMT investigacionalmente.

---

## 6. Tabla 3: Ablations

Ablation systematica sobre WMT'14 EN-DE dev (newstest2013):

| Variante | Cambio | PPL dev | BLEU dev | Comentario |
|---|---|---|---|---|
| **base** | $N=6, d=512, d_{ff}=2048, h=8, d_k=d_v=64$ | 4.92 | 25.8 | referencia |
| (A) $h=1$ | una sola cabeza | 5.29 | 24.9 | -0.9 BLEU, perder cabezas duele |
| (A) $h=4$ | 4 cabezas | 5.00 | 25.5 | mejor que single-head |
| (A) $h=16$ | 16 cabezas | 4.91 | 25.8 | sweet spot ancho |
| (A) $h=32$ | 32 cabezas, $d_k=16$ | 5.01 | 25.4 | demasiadas cabezas pequenas degradan |
| (B) $d_k=16$ | reducir dim por cabeza | 5.16 | 25.1 | hurts quality -- compatibility no es facil |
| (B) $d_k=32$ | $d_k=32$ | 5.01 | 25.4 | sigue por debajo |
| (C) $N=2$ | menos capas | 6.11 | 23.7 | depth ayuda mucho |
| (C) $N=8$ | mas capas | 4.88 | 25.5 | retornos decrecientes |
| (C) $d_{model}=256$ | menor ancho | 5.75 | 24.5 | width tambien importa |
| (C) $d_{model}=1024$ | mayor ancho | 4.66 | 26.0 | mejora pero +2.5x params |
| (C) $d_{ff}=1024$ | FFN mas chica | 5.12 | 25.4 | FFN aporta capacidad |
| (C) $d_{ff}=4096$ | FFN mas grande | 4.75 | 26.2 | mejora monotonica |
| (D) $P_{drop}=0$ | sin dropout | 5.77 | 24.6 | overfit severo |
| (D) $P_{drop}=0.2$ | mas dropout | 4.95 | 25.5 | optimum cerca de 0.1 |
| (D) $\epsilon_{ls}=0$ | sin label smoothing | 4.67 | 25.3 | mejor PPL pero peor BLEU |
| (D) $\epsilon_{ls}=0.2$ | mas LS | 5.47 | 25.7 | demasiado regulariza |
| (E) PE aprendido | embeddings posicionales | 4.92 | 25.7 | identico a sinusoidal! |
| **big** | $N=6, d=1024, d_{ff}=4096, h=16, P_{drop}=0.3, 300K$ steps | **4.33** | **26.4** | mejor configuracion |

**Lecciones de la ablation**:

1. **Multi-head importa, pero no demasiado**: ir de $h=1$ a $h=8$ gana 0.9 BLEU; de $h=8$ a $h=16$ apenas se mueve.
2. **$d_k$ pequeno hiere la calidad** (B): el producto punto necesita suficiente dimensionalidad. Sugerencia abierta del paper: "una funcion de compatibilidad mas sofisticada que dot-product podria ayudar". Esto germinaria en investigaciones sobre kernels alternativos.
3. **Ancho > profundidad** marginalmente, en este regimen: pasar de $d_{model}=512$ a $1024$ aporta tanto como pasar de $N=6$ a $N=8$.
4. **FFN aporta capacidad**: hoy sabemos que una fraccion grande de los parametros del modelo (~2/3) y de la "memoria" factual viven en el FFN.
5. **Dropout y label smoothing son obligatorios** (D): sin regularizacion, el modelo overfit-ea.
6. **PE aprendido vs sinusoidal: empate** (E). El paper eligio sinusoidal por extrapolacion a longitudes mayores que las vistas en train. Hoy las opciones modernas (RoPE, ALiBi) superan a ambas.

---

## 7. Trabajos Previos Referenciados

| Trabajo | Anho | Aporte que Vaswani usa |
|---|---|---|
| Bahdanau, Cho, Bengio | 2015 | Mecanismo de atencion (additive) |
| Cho et al. | 2014 | Encoder-decoder con GRU |
| Sutskever, Vinyals, Le | 2014 | Seq2Seq con LSTM |
| Wu et al. (GNMT) | 2016 | NMT en escala de produccion, ensembles |
| Gehring et al. (ConvS2S) | 2017 | Encoder-decoder convolucional, paralelizable |
| Kalchbrenner et al. (ByteNet) | 2017 | Convoluciones dilatadas, $O(\log n)$ path |
| Shazeer et al. (MoE) | 2017 | Mixture of Experts |
| Cheng, Dong, Lapata | 2016 | Self-attention para reading comprehension |
| Parikh et al. | 2016 | "A decomposable attention model" -- precursor directo de self-attn |
| Ba, Kiros, Hinton | 2016 | Layer Normalization |
| He et al. | 2016 | Residual connections |
| Srivastava et al. | 2014 | Dropout |
| Szegedy et al. | 2016 | Label smoothing |
| Sennrich et al. | 2015 | Byte-pair encoding (BPE) |
| Press, Wolf | 2016 | Tied input/output embeddings |
| Kingma, Ba | 2015 | Adam optimizer |

El paper se apoya con elegancia en muchos componentes de la era 2014-2017 (residual + layer norm + dropout + label smoothing + Adam + BPE). La novedad arquitectonica especifica es la combinacion de scaled dot-product + multi-head + positional encoding + stack profundo.

---

## 8. Sucesores Directos

```text
2017 ─── ATTENTION IS ALL YOU NEED (Vaswani et al.)
            │
            ├── 2018 ─── BERT (Devlin)         encoder-only, MLM, transferencia masiva
            │           │
            │           ├── RoBERTa, ALBERT, ELECTRA, DistilBERT
            │           └── XLM, XLM-R (multilingue)
            │
            ├── 2018 ─── GPT-1 (Radford)       decoder-only, autoregresivo
            │           │
            │           ├── GPT-2 (2019)        scale up, zero-shot
            │           ├── GPT-3 (2020)        in-context learning, 175B params
            │           ├── GPT-4 (2023)        multimodal
            │           └── ChatGPT, Claude, Gemini, LLaMA, Mistral, ...
            │
            ├── 2019 ─── T5 (Raffel)            encoder-decoder unificado, text-to-text
            │
            ├── 2020 ─── ViT (Dosovitskiy)      transformer puro en vision
            │           │
            │           ├── DeiT, Swin, MAE, DINO
            │           └── CLIP (vision-language)
            │
            ├── 2020 ─── DETR (Carion)          deteccion de objetos como set prediction
            │
            ├── 2021 ─── AlphaFold 2 (Jumper)   transformer sobre residuos, proteinas
            │
            ├── 2022 ─── Whisper, MusicLM       audio
            │           Stable Diffusion, DALL-E 2 (componente cross-attn)
            │
            ├── 2023 ─── RoPE (Su)              positional encoding rotatorio
            │           ALiBi (Press)            atencion con bias linear
            │           FlashAttention (Dao)     atencion exacta y eficiente en memoria
            │
            └── 2024 ─── Mamba (Gu)              state-space alternativo (no transformer)
                        Sora (video)             scaling continuado
```

Cada uno de estos modelos hereda directamente la arquitectura Transformer; varios solo difieren en si usan encoder, decoder o ambos, en el preentrenamiento, y en escala.

---

## 9. Insights y Observaciones Tecnicas

### 9.1. Lo elegante

- **Atencion como primitiva universal**. La formula $\text{softmax}(QK^T/\sqrt{d_k})V$ se aplica identicamente para self-attn y cross-attn. Una sola operacion para varios usos.
- **Residual + LayerNorm como "highway"**. Permite que el gradiente fluya sin atenuarse en stacks profundos. Hoy sabemos que la norma del residual stream es la metrica clave de salud del modelo.
- **Multi-head es factorizacion gratuita**. Mismo costo total, mas expresividad. Las cabezas se especializan emergentemente sin supervision adicional.
- **Sinusoides como pseudo-rotaciones**. La derivacion linear-en-$k$ es matematicamente bonita y permite extrapolacion limitada.
- **Decoder mask como triangular inferior**. Codigo de 1 linea (`mask = torch.triu(..., diagonal=1)`) que implementa la causalidad sin cambiar la operacion.

### 9.2. Lo pragmatico

- **Warmup del learning rate** (formula $d_{model}^{-0.5} \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$): truco crucial para estabilizar Adam con LayerNorm. Posteriormente RMSNorm + LR cosine schedule lo simplifico.
- **Label smoothing** ($\epsilon_{ls}=0.1$): sacrifica perplexity pero **mejora BLEU**. Esto es un recordatorio importante de que la metrica de optimizacion no es la metrica de evaluacion.
- **Adam con $\beta_2=0.98$** en lugar del 0.999 tipico: es un heuristico para training rapido, atribuido a observaciones empiricas.
- **Tied embeddings** (input emb = output emb = pre-softmax): reduce parametros y mejora generalizacion, multiplicado por $\sqrt{d_{model}}$ para que la varianza coincida.
- **BPE / wordpiece**: sin esto el vocabulario seria intratable. Es ortogonal al transformer pero indispensable para que funcione en NLP.

### 9.3. Lo que se descubrio luego que es suboptimo

- **Post-LN (Post Layer Norm)**: el orden $\text{LN}(x + \text{Sublayer}(x))$ del paper original sufre inestabilidades en stacks muy profundos. **Pre-LN** ($x + \text{Sublayer}(\text{LN}(x))$, usado en GPT-2 en adelante) es mas estable y elimina la necesidad de warmup tan agresivo.
- **Sinusoidal PE** no extrapola tan bien como prometia. RoPE (Su 2021) y ALiBi (Press 2022) son superiores en practica.
- **Atencion cuadratica** es el cuello de botella estructural. Ha sido objeto de centenares de papers (Linformer, Reformer, Performer, Longformer, Big Bird, FlashAttention, RetNet, Mamba, RWKV).
- **Activacion ReLU** en el FFN: superada por GeLU (BERT, GPT-2) y SwiGLU (LLaMA, PaLM). El paper original usaba ReLU por simplicidad.
- **LayerNorm vs RMSNorm**: RMSNorm (Zhang & Sennrich 2019) elimina el centrado de la media, es mas barato y empiricamente equivalente. LLaMA y modernos lo prefieren.
- **8 cabezas de 64 dim**: investigaciones posteriores (Michel et al. 2019) muestran que **muchas cabezas son podables** sin perdida -- gran parte del trabajo lo hacen unas pocas cabezas dominantes. Esto sugiere que el numero optimo depende mas del compute que de la expresividad.

### 9.4. Lo que el paper anuncio sin saberlo

- "**The Transformer can be trained significantly faster**" (conclusion). Lo que parecia 3.5 dias para SOTA, en realidad era una primera prueba de que la arquitectura **escala maravillosamente**. Las scaling laws (Kaplan 2020) lo confirmaron 3 anos despues.
- "**We are excited about... extending the Transformer to problems involving input and output modalities other than text**". ViT (2020), DALL-E (2021), Whisper (2022), AlphaFold (2021). El paper lo predijo.
- "**Making generation less sequential is another research goals of ours**". Sigue siendo problema abierto en 2026: speculative decoding, parallel decoding, diffusion-LM, etc.

---

## 10. Lecciones Transferibles para Diseno de Arquitecturas

### 10.1. Cuestiona los sesgos inductivos heredados

El campo asumia que **secuencia => recurrencia**. Vaswani et al. cuestionaron eso. Lecciones generales:

- Si una primitiva (atencion) ya domina el rendimiento, considera elevarla a bloque elemental.
- Los sesgos inductivos correctos son los minimos necesarios para que el modelo aprenda; sesgos extra (recurrencia, localidad convolucional) limitan la flexibilidad cuando hay datos suficientes.

### 10.2. Optimiza para el hardware, no para la teoria

- **Paralelizacion masiva** justifica cuadraticidad en secuencia. En GPUs/TPUs, $O(n^2)$ paralelo es mejor que $O(n)$ secuencial.
- **Operaciones matriciales densas** son lo que los aceleradores hacen mejor. Atencion es matmul + softmax + matmul; nada exotico.
- Esto presagia el "**hardware lottery**" (Sara Hooker 2020): las arquitecturas que ganan son las que se alinean con el hardware disponible.

### 10.3. Residual + Norm es la base de toda red profunda moderna

Sin estos dos ingredientes (heredados de ResNet 2016 y LayerNorm 2016), el Transformer no entrenaria. Cualquier arquitectura profunda actual los usa por defecto.

### 10.4. Las visualizaciones no son adornos

Las Figuras 3-5 del apendice (cabezas que aprenden anafora, sintaxis, etc.) **convencen** al lector de que el modelo no es magia negra. Una arquitectura que produce representaciones interpretables genera confianza.

### 10.5. Las ablations son la forma de defender un diseno

La Tabla 3 es ejemplar: cada hiperparametro variado por separado, con metricas claras. No es decorativo -- es como se justifica cientificamente que **cada decision** del diseno aporta.

### 10.6. Eficiencia de training es tan importante como SOTA

El Transformer no solo gano en BLEU. Gano **en costo**. Esto fue lo que catalizo el escalado masivo posterior: sin eficiencia, GPT-3 no habria sido economicamente factible.

### 10.7. La eleccion correcta de primitiva habilita escalado

Las scaling laws funcionan tan bien en transformers porque la atencion es uniforme, paralelizable y sin cuellos secuenciales. RNNs nunca habrian escalado a billones de parametros con la misma facilidad.

---

## 11. Resumen en una pagina

```text
PROBLEMA:  En 2017, los modelos seq2seq dominantes (LSTM con
           atencion, ConvS2S) eran lentos de entrenar (recurrencia
           secuencial) y limitados en dependencias largas. Atencion
           era un anadido sobre RNN/CNN, no la primitiva.

PROPUESTA: El Transformer -- arquitectura encoder-decoder basada
           UNICAMENTE en atencion. Sin RNN, sin CNN. Stack de N=6
           capas con multi-head self-attention + FFN posicional,
           residuales + LayerNorm, positional encoding sinusoidal.

INNOVACIONES TECNICAS:
  1. Scaled dot-product:  softmax(QK^T / sqrt(d_k)) V
     → escalado evita saturacion del softmax para d_k grande
  2. Multi-head: h=8 cabezas paralelas en subespacios d_k=64
     → factorizacion gratuita en compute, mas expresividad
  3. Positional encoding sinusoidal:
     → PE_{pos+k} es funcion lineal de PE_pos (rotacion)
  4. Tres usos de atencion:
     - encoder self-attn (bidirectional)
     - decoder self-attn (causal mask)
     - encoder-decoder cross-attn

CONFIGURACION BASE:
  N=6, d_model=512, d_ff=2048, h=8, d_k=d_v=64, 65M params
  Big: d_model=1024, d_ff=4096, h=16, 213M params

RESULTADOS WMT'14:
  EN-DE: 28.4 BLEU  (vs ConvS2S 26.36, GNMT 26.30)
  EN-FR: 41.0 BLEU  (vs ConvS2S Ens 41.29 con 50x mas compute)
  Training: 3.5 dias en 8 P100 (vs semanas para baselines)
  Tambien generaliza a parsing constituyente: 91.3 F1 en WSJ.

POR QUE FUNCIONA:
  - Path length O(1) entre cualquier par de tokens
  - Paralelizable en el eje temporal (saturando GPU/TPU)
  - Mismo backbone para muchas modalidades

LIMITACIONES:
  - Complejidad O(n^2) en longitud (problema central post-2017)
  - Sin sesgo de localidad: requiere mucha data
  - Generacion sigue siendo secuencial

LEGADO:
  Es LA arquitectura de los foundation models 2018+:
  BERT, GPT-1/2/3/4, T5, ViT, AlphaFold, CLIP, Whisper,
  Stable Diffusion, ChatGPT, Claude, Gemini, LLaMA.
  Habilito el escalado (scaling laws) y la era de los LLMs.

CITAS: >150,000 (Google Scholar, 2026) -- uno de los papers
       mas influyentes en la historia del Deep Learning.
```
