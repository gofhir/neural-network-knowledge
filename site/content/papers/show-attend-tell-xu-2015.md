---
title: "Show, Attend and Tell"
weight: 210
math: true
---

{{< paper-card
    title="Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
    authors="Xu, Ba, Kiros, Cho, Courville, Salakhutdinov, Zemel, Bengio"
    year="2015"
    venue="ICML 2015"
    pdf="/papers/show-attend-tell-xu-2015.pdf"
    arxiv="1502.03044" >}}
Extiende el modelo NIC (image captioning basico, Vinyals 2015) con un **mecanismo de atencion visual** que permite al decoder LSTM enfocarse dinamicamente sobre distintas regiones de la imagen mientras genera cada palabra. Introduce dos variantes -- **soft attention** (determinista, diferenciable) y **hard attention** (estocastica, entrenada con REINFORCE / variational lower bound). Establece el nuevo estado del arte en Flickr8k, Flickr30k y MS COCO, y sus visualizaciones de atencion son un icono visual del deep learning moderno.
{{< /paper-card >}}

---

## Contexto

[Vinyals et al. 2015 (NIC)](/papers/show-and-tell-vinyals-2015) demostro que CNN encoder + LSTM decoder podia generar captions de imagenes end-to-end. Pero la imagen se comprimia en un solo vector (output de la CNN FC), que se alimentaba al LSTM al inicio. El decoder no podia **mirar distintas partes** de la imagen al generar cada palabra.

Paralelamente, [Bahdanau 2015](/papers/bahdanau-attention-2015) habia introducido attention para NMT. La pregunta natural: podemos aplicar attention al eje **espacial** de las features de una imagen?

Este paper (Toronto + Montreal) responde afirmativamente y establece el patron de **visual attention** que domino la vision + language hasta los Transformers multimodales.

---

## Ideas principales

### 1. Features visuales como secuencia

En vez de usar la salida fully-connected de una CNN (un solo vector 4096-dim, como en NIC), usan un **feature map intermedio**:

$$a = \{a_1, a_2, \ldots, a_L\}, \quad a_i \in \mathbb{R}^D$$

En particular: $14 \times 14 \times 512$ del **4th conv layer de VGG** preentrenada → 196 annotation vectors de 512-dim.

Cada $a_i$ corresponde a una **region espacial** de la imagen. Esto permite atencion sobre ubicaciones especificas.

### 2. LSTM decoder condicionado en context vector

El LSTM recibe en cada paso un context vector $\hat{z}_t$ que **cambia por paso** segun la atencion:

$$\hat{z}_t = \phi(\{a_i\}, \{\alpha_{t,i}\})$$

Con $\alpha_{t,i}$ computada via alignment model entre el estado previo del LSTM $h_{t-1}$ y la annotation $a_i$:

$$e_{ti} = f_{\text{att}}(a_i, h_{t-1})$$
$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_k \exp(e_{tk})}$$

### 3. Dos variantes de atencion

#### Soft attention (deterministica)

$$\hat{z}_t = \sum_{i=1}^{L} \alpha_{ti} a_i$$

El context vector es un **promedio ponderado** de todas las regiones. Diferenciable → backprop estandar, training facil. Es el workhorse en la practica.

#### Hard attention (estocastica)

En cada paso, **muestrear una unica region** segun $p(s_{t,i} = 1 \mid a) = \alpha_{t,i}$ (multinoulli). El context es entonces el $a$ de esa region sola:

$$\hat{z}_t = \sum_i s_{t,i} a_i \quad \text{(con } s_t \text{ one-hot)}$$

No diferenciable → requiere **REINFORCE** (policy gradient) o variational lower bound. Training inestable, mejora marginal sobre soft. Menos usado en la practica.

### 4. Doubly stochastic attention (regularizacion)

Para evitar que el modelo solo mire una pequena region durante toda la generacion, se agrega un termino de regularizacion que **empuja** a que $\sum_t \alpha_{t,i} \approx 1$ para cada ubicacion $i$:

$$L_d = -\log P(y \mid x) + \lambda \sum_{i}^{L} \left(1 - \sum_{t}^{C} \alpha_{ti}\right)^2$$

Esto obliga al modelo a "visitar" todas las regiones de la imagen al menos una vez a lo largo de la oracion. Mejora calidad de captions.

### 5. Output con deep output layer

En lugar de solo softmax(Wh_t), usan un deep output (Pascanu 2014):

$$p(y_t \mid a, y_{1}^{t-1}) \propto \exp(L_o (E y_{t-1} + L_h h_t + L_z \hat{z}_t))$$

combinando embedding del token previo, estado LSTM y context vector, para mayor expresividad.

---

## Resultados experimentales

BLEU scores en tres datasets estandar:

| Modelo | Flickr8k | Flickr30k | MS COCO (BLEU-4) |
|---|---|---|---|
| Google NIC (Vinyals 2015) | 63 | 66.3 | 27.7 |
| **Soft attention (este)** | **67** | **66.7** | **24.3** |
| **Hard attention (este)** | **67** | **66.9** | **25.0** |

Notable: soft y hard attention obtienen resultados comparables; **soft es mas facil de entrenar**. La mejora sobre NIC es modesta en numeros, pero **cualitativamente** (visualizaciones) es dramatica.

### Figura iconica: attention maps

La Figura 3 del paper muestra como el modelo mira la region correcta al generar cada palabra:

- "A **bird** flying over a body of **water**" → atencion al pajaro al generar "bird", al agua al generar "water".
- "A **dog** is standing on a hardwood floor" → atencion al perro.
- "A **stop sign** is on a road with a mountain" → atencion al signo de pare.

Failure cases (Figura 5) tambien son informativos: cuando el modelo se equivoca, **la visualizacion explica por que** -- estaba mirando la region equivocada.

---

## Por que importa hoy

- **Paper visual signature**: las visualizaciones de attention maps se volvieron icono del deep learning moderno. Antes de Show-Attend-Tell, los modelos eran cajas negras; despues, habia una forma visual de "ver" lo que miraban.
- **Patron universal**: el uso de un feature map intermedio como secuencia de tokens visuales es la base conceptual de **Vision Transformer** (Dosovitskiy 2020), modelos multimodales como **CLIP**, **BLIP**, **LLaVA**, **GPT-4V**.
- **Interpretabilidad en vision**: inspiro Grad-CAM, Integrated Gradients y otras tecnicas de attribution.
- **Soft attention como default**: la inviabilidad practica de hard attention (REINFORCE inestable) consolido soft attention como el estandar industrial.
- **Precursor del bottom-up attention** (Anderson 2018): mejoro las regiones usando Faster R-CNN en vez de un grid uniforme.

---

## Limitaciones

- **Grid uniforme**: el 14×14 grid de VGG no respeta los objetos reales en la imagen. [Anderson 2018](/papers/bottom-up-attention-anderson-2018) resolvio esto con Faster R-CNN.
- **CNN fija**: VGG preentrenada en ImageNet no se fine-tunea. Con CNN fine-tuned + mejor arquitectura (ResNet, Inception), los numeros mejoran.
- **Hard attention poco practico**: el training con REINFORCE es fragil; en la practica todo el mundo usa soft.
- **Complejidad cuadratica**: atencion sobre $196 \times T_y$ regiones para cada paso.
- **Limitaciones en razonamiento multi-object**: captions complejos con multiples objetos interactuando aun requieren bottom-up attention + VQA models.

---

## Notas y enlaces

- La **Figura 4** (diagrama LSTM con input_modulator, memory_cell) es una version pedagogica de LSTM con forget gate.
- Las **Figuras 2, 3, 5** son referentes visuales muy citadas en charlas y tutoriales.
- Codigo oficial: [kelvinxu/arctic-captions](https://github.com/kelvinxu/arctic-captions) (Theano).
- Follow-ups:
  - **Lu et al. 2017** "Knowing When to Look" -- adaptive attention (decide cuando mirar la imagen vs generar del lenguaje).
  - **Anderson et al. 2018** [Bottom-Up and Top-Down Attention](/papers/bottom-up-attention-anderson-2018) -- regiones de Faster R-CNN.
  - **Tan & Bansal 2019** "LXMERT" -- attention multimodal para VQA.

Ver fundamentos: [Mecanismo de Atencion](/fundamentos/mecanismo-atencion) · [Sequence to Sequence](/fundamentos/seq2seq) · [Redes Convolucionales](/fundamentos/redes-convolucionales) · [Transfer Learning](/fundamentos/transfer-learning).
