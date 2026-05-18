# Mikolov et al. 2010 — Recurrent Neural Network Based Language Model

| Campo | Valor |
|---|---|
| **Autores** | Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan "Honza" Černocký, Sanjeev Khudanpur |
| **Afiliación** | Speech@FIT, Brno University of Technology + Johns Hopkins (CLSP) |
| **Venue** | INTERSPEECH 2010, Makuhari, Chiba, Japan |
| **Fecha** | 26-30 septiembre 2010 |
| **Pdf** | `Mikolov-RNN-LM-2010.pdf` (4 páginas) |
| **Citaciones** | >5.000 |
| **URL** | https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf |

> *"Results indicate that it is possible to obtain around 50% reduction of perplexity by using mixture of several RNN LMs, compared to a state of the art backoff language model."*

Este es **el primer paper de Tomáš Mikolov** (junto a otros sobre RNN-LM 2011 y la tesis 2012) que precede directamente a Word2Vec. Es un paper corto (4 páginas) pero **demuele la creencia de que los n-gramas son insuperables**, abriendo el camino a toda la era neuronal en NLP. La slide 22 de la clase 18 muestra el diagrama RNN-LM derivado de este paper.

---

## 1. Contexto — n-gramas vs neural networks en 2010

### 1.1 El estado del arte en LMs alrededor de 2010

- **Trigramas / 5-gramas con suavizado Kneser-Ney modificado** (Chen & Goodman 1998): el SOTA estadístico, dominante en speech recognition y MT desde finales de los 90s.
- **NPLM** (Bengio 2003): mejora marginal sobre KN5, pero costo prohibitivo (3 semanas en cluster para 1M palabras). Schwenk lo aplicó a speech con éxito limitado por el costo.
- **Class-based models**: Brown 1992. Mejora moderada, complementario a n-gramas.
- **Cache models**: aumentar probabilidad de palabras vistas recientemente. Mejora moderada en perplejidad.

### 1.2 Diagnóstico de Mikolov

Mikolov es directo en la introducción:

> *"Models coming from research tend to be complex and often work well only for systems based on very limited amounts of training data. In fact, most of the proposed advanced language modeling techniques provide only tiny improvements over simple baselines, and are rarely used in practice."*

Su crítica al NPLM de Bengio es específica: **contexto fijo**. NPLM ve solo $n-1$ palabras (típicamente 5-10). Pero el lenguaje humano usa dependencias mucho más largas.

### 1.3 La propuesta

Reemplazar la ventana fija del NPLM por una **RNN simple (Elman)** que codifica el contexto en un estado oculto recurrente. El contexto efectivo es **ilimitado** (al menos en teoría).

---

## 2. Arquitectura — Simple Recurrent Network (SRN) / Elman

### 2.1 Notación

- $\mathbf{w}(t)$: vector one-hot de la palabra en tiempo $t$.
- $\mathbf{s}(t)$: estado oculto (también llamado *context layer*).
- $\mathbf{y}(t)$: vector de output (distribución sobre vocab).
- $\mathbf{x}(t)$: input concatenado.

### 2.2 Ecuaciones

**Concatenación del input:**
$$
\mathbf{x}(t) = \mathbf{w}(t) + \mathbf{s}(t-1). \quad (1)
$$

(Nota: el `+` es **concatenación**, no suma. El input es $|V| + H$ dimensiones, con $|V|$ size del vocab y $H$ size de la hidden layer.)

**Hidden layer con sigmoid:**
$$
s_j(t) = f\left( \sum_i x_i(t) u_{ji} \right), \quad f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}. \quad (2, 4)
$$

**Output con softmax:**
$$
y_k(t) = g\left( \sum_j s_j(t) v_{kj} \right), \quad g(z_m) = \frac{e^{z_m}}{\sum_k e^{z_k}}. \quad (3, 5)
$$

### 2.3 Diagrama (figura 1 del paper, reproducida en slide 22 de la clase)

```
                        ┌──────────────────────┐
                        │     OUTPUT(t)        │  ← y(t), softmax sobre |V|
                        │   ─────────────────  │
                        └──────────────────────┘
                                  ↑
                  ┌───────────────────────────────┐
INPUT(t) ────────►│       CONTEXT(t)              │  ← s(t), hidden state
[one-hot w(t)]    │   ─────────────────────       │     (función de w(t) y s(t-1))
                  └───────────────────────────────┘
                              ↑
                  ┌───────────────────────────────┐
                  │      CONTEXT(t-1)             │  ← s(t-1), estado previo
                  └───────────────────────────────┘
```

**Lectura**: el estado oculto $\mathbf{s}(t)$ depende de la palabra actual $\mathbf{w}(t)$ y del estado previo $\mathbf{s}(t-1)$. La predicción $\mathbf{y}(t)$ es la distribución sobre $\mathbf{w}(t+1)$ dado todo el contexto pasado **comprimido en $\mathbf{s}(t)$**.

### 2.4 Tamaños

| Parámetro | Valor típico |
|---|---|
| $|V|$ | 30k - 200k |
| Hidden size $H$ | 30 - 500 (paper) |
| # parámetros $U$ (input → hidden) | $(|V| + H) \times H$ |
| # parámetros $V$ (hidden → output) | $H \times |V|$ |
| Total | $|V| \cdot H + (|V| + H) \cdot H$ |

Para $|V| = 50k$ y $H = 100$: ~10M parámetros. Comparable a NPLM de Bengio.

---

## 3. Entrenamiento

### 3.1 Inicialización

- Pesos: ruido gaussiano $\mathcal{N}(0, 0.1^2)$.
- $\mathbf{s}(0)$: vector de valores pequeños (e.g., todo 0.1). El paper menciona que cuando se procesa mucho data, la inicialización no es crucial — el estado se "olvida" en pocos pasos.

### 3.2 Stochastic Gradient Descent

- Learning rate inicial $\alpha = 0.1$.
- Tras cada epoch: evaluar en validation. Si mejora, continuar; si no, halve $\alpha$. Después de no mejora dos veces, parar.
- Convergencia típica: 10-20 epochs.

### 3.3 Error y backprop

Cross-entropy contra one-hot:
$$
\text{error}(t) = \text{desired}(t) - \mathbf{y}(t). \quad (6)
$$

Backprop estándar sobre los pesos. **Nota importante**: el paper inicial usa **truncated BPTT con $\tau = 1$** — solo retropropaga error a un paso atrás. Esto simplifica el entrenamiento pero limita la capacidad de capturar dependencias largas.

En trabajos posteriores (Mikolov 2011 *Extensions of RNN-LM*, Mikolov 2012 PhD thesis), se usa **BPTT más profundo** ($\tau = 5$ o $\tau = 10$) con mejoras notables.

### 3.4 Regularización mínima

> *"In our experiments, networks do not overtrain significantly, even if very large hidden layers are used - regularization of networks to penalize large weights did not provide any significant improvements."*

Esto es contraintuitivo desde la perspectiva moderna (dropout, weight decay son canónicos). Mikolov reporta que con suficientes datos, el modelo no overfittea.

---

## 4. Innovaciones de optimización

### 4.1 Rare words → token único

Para reducir el tamaño efectivo del vocabulario y acelerar:

> *"We merge all words that occur less often than a threshold (in the training text) into a special rare token."*

Probabilidad asignada a una palabra rara $w_i$:
$$
P(w_i \mid \text{contexto}) = \frac{y_{\text{rare}}(t)}{C_{\text{rare}}}
$$

donde $C_{\text{rare}}$ es el número de palabras consolidadas en `<rare>` y $y_{\text{rare}}(t)$ es la masa que el modelo asigna a esa clase. Distribución uniforme dentro de `<rare>`. Esta clase es predecesor conceptual del unknown token `<UNK>` que usarán BPE/WordPiece.

### 4.2 Dynamic model — online learning durante test

Una idea radical para 2010: **continuar entrenando durante la inferencia**.

> *"The network should continue training even during testing phase. We refer to such model as dynamic. For dynamic model, we use fixed learning rate $\alpha = 0.1$."*

**Motivación**: en speech recognition, si una persona menciona repetidamente un nombre propio (e.g., "Khudanpur"), un LM estático asignará baja probabilidad cada vez. Un LM dinámico aprende sobre la marcha.

**Conexión con cache models**: cache n-gram aumenta probabilidad de palabras vistas. Dynamic RNN-LM hace algo similar pero **en espacio continuo**: si "dog" aparece frecuente en test data, también sube la probabilidad de "cat" (porque están cerca en el embedding implícito).

**Limitación**: en speech recognition, la "historia" contiene errores del recognizer. Cache models tienden a degradar en este escenario. Dynamic RNN-LM degrada menos por la similitud continua.

### 4.3 Speedup en entrenamiento

> *"For comparison, it takes around 6 hours for our basic implementation to train RNN model based on Brown corpus (800K words, 100 hidden units and vocabulary threshold 5), while Bengio reports 113 days for basic implementation and 26 hours with importance sampling."*

**6 horas vs 113 días** — Mikolov demuestra que las RNN-LM son **órdenes de magnitud más eficientes** que NPLM con feedforward. Esto es porque:
1. No hay capa de proyección concatenada de $(n-1)m$ dim — solo una matriz $(|V| + H) \times H$.
2. La recurrencia comparte parámetros entre posiciones.
3. Usa BLAS optimizado.

---

## 5. Experimentos

### 5.1 Wall Street Journal (WSJ)

Tarea estándar de speech recognition. Training data: 6.4M palabras del NYT section de Gigaword.

**Tabla 1** — Efecto del tamaño de training data:

| Model | # words | PPL | WER |
|---|---|---|---|
| KN5 LM | 200K | 336 | 16.4 |
| KN5 LM + RNN 90/2 | 200K | 271 | 15.4 |
| KN5 LM | 1M | 287 | 15.1 |
| KN5 LM + RNN 90/2 | 1M | 225 | 14.0 |
| KN5 LM | 6.4M | 221 | 13.5 |
| KN5 LM + RNN 250/5 | 6.4M | **156** | **11.7** |

**Lectura**: incluso con baseline KN5 fuerte, agregar un RNN reduce perplejidad 30-40% y WER ~13%.

**Tabla 2** — Configuraciones de RNN:

| Model | PPL (RNN solo) | PPL (RNN+KN) | WER (RNN solo) | WER (RNN+KN) |
|---|---|---|---|---|
| KN5 baseline | — | 221 | — | 13.5 |
| RNN 60/20 | 229 | 186 | 13.2 | 12.6 |
| RNN 90/10 | 202 | 173 | 12.8 | 12.2 |
| RNN 250/5 | 173 | 155 | 12.3 | 11.7 |
| RNN 400/10 | 171 | 152 | 12.5 | 12.1 |
| **3xRNN static** | 151 | 143 | 11.6 | 11.3 |
| **3xRNN dynamic** | **128** | **121** | **11.3** | **11.1** |

**Observaciones**:
- RNNs solas superan a KN5 (PPL 171 vs 221, WER 12.5 vs 13.5).
- **Combinación lineal RNN+KN5** (interpolación 0.75/0.25) supera a cualquiera individualmente — son **complementarios**.
- 3 RNN's distintas (diferente init y tamaño) en ensemble ganan más.
- Dynamic RNN gana ~3 puntos perpleja sobre static.

### 5.2 Comparación con SOTA en WSJ

**Tabla 3** — Mikolov vs sistemas competidores:

| Model | DEV WER | EVAL WER |
|---|---|---|
| Lattice 1-best (no LM rescoring) | 12.9 | 18.4 |
| Baseline KN5 (37M words) | 12.2 | 17.2 |
| Discriminative LM Xu (37M) | 11.5 | 16.9 |
| Joint LM Filimonov (70M) | — | 16.7 |
| **Static 3xRNN + KN5 (37M)** | **11.0** | **15.5** |

**Lectura**: Mikolov reduce WER de 17.2 → 15.5 (relativa 10%), superando todos los baselines incluyendo modelos discriminativos y joint LMs entrenados con **70M palabras**, usando RNN entrenado solo en **6.4M palabras**.

### 5.3 NIST RT05 — Meeting Speech

Tarea más difícil (speech espontáneo, multi-hablante).

**Tabla 4** — RNN en setting low-resource:

| Model | WER static | WER dynamic |
|---|---|---|
| RT05 LM (1.3G words) | 24.5 | — |
| RT09 LM baseline | 24.1 | — |
| KN5 in-domain (5.4M) | 25.7 | — |
| RNN 500/10 in-domain (5.4M) | 24.2 | 24.1 |
| **RNN + RT09 LM** | **23.3** | **23.2** |
| 3xRNN + RT09 LM | 23.3 | 22.8 |

**Conclusión devastadora**: un RNN entrenado en **5.4M words in-domain** supera al backoff trained en **1.3 GIGAWORDS** (240× más data). Este resultado **rompió el mito** de que "más data > mejor modelo" para LMs.

---

## 6. Limitaciones reconocidas

1. **Truncated BPTT con $\tau = 1$**: no captura dependencias muy largas. El paper menciona que "no parece que RNN simples puedan capturar contexto realmente largo", aunque trabajos posteriores con BPTT más profundo y LSTM (Sundermeyer 2012) lo mejoraron.
2. **Sin capa de embedding explícita**: el word vector es one-hot en el input, no hay matriz de embedding entrenable separada como en NPLM. Esto Mikolov lo corregirá en Word2Vec.
3. **Softmax sobre $|V|$**: bottleneck. Resuelto en Mikolov 2011 con hierarchical softmax / classes.
4. **Sin paralelismo natural**: las RNN secuenciales no se paralelizan tan bien como los Transformers.
5. **Dynamic learning durante test es no-estándar**: rompe la suposición usual de modelos congelados, complica el deployment.

---

## 7. Impacto

### 7.1 Impacto inmediato (2010-2013)

- **2010**: este paper.
- **2011** (Mikolov, ICASSP) — *Extensions of RNN-LM* con classes y BPTT profundo.
- **2012** (Mikolov PhD) — sistema completo, batería de experimentos.
- **2012** (Sundermeyer): LSTM para LM. Mejora vanishing gradients.
- **2013** (Mikolov): **Word2Vec**. Aplicar lecciones del RNN-LM al problema de aprender solo embeddings.

### 7.2 Impacto a largo plazo

Este paper inicia la **era neural en NLP**. Sin RNN-LM no hay:

- Word2Vec (2013) — Mikolov venía del RNN-LM, reemplazó la recurrencia por ventana móvil para escalar.
- Seq2seq (2014) — Sutskever et al. usaron LSTM, no SRN, pero la idea de **encoder-decoder recurrente** es directamente derivada de RNN-LM.
- ELMo (2018) — biLSTM language model preentrenado como source de embeddings contextuales.
- ULMFiT (2018) — transfer learning con LMs.
- GPT (2018) — Transformer-LM. Mismo objetivo que RNN-LM, diferente arquitectura.

### 7.3 Frase clave del paper

> *"Obtained results are breaking myth that language modeling is just about counting n-grams, and that the only reasonable way how to improve results is by acquiring new training data."*

Esta frase **marca el cambio de paradigma**. En 2010 era una afirmación arriesgada — los n-gramas reinaban en producción y la mayoría de papers de LM se enfocaban en mejor suavizado, no en arquitectura. Tres años después, Word2Vec haría obvia esta observación para toda la comunidad.

---

## 8. Conexión con la clase 18

**Slide 22** de la clase 18 muestra:

> *"No es obligatorio que la red neuronal usada sea una feed forward. Otra opción puede ser una red neuronal recurrente (RNN)."*

Con un diagrama de RNN procesando secuencialmente palabras "The cat sat on a ..." → "cat sat on a mat", donde cada celda es un "context vector" que se actualiza.

Esto es **literalmente** el diagrama del SRN de Mikolov 2010 (figura 1 del paper). La slide elide los detalles técnicos pero captura la idea esencial: **reemplazar la ventana fija del NPLM por un estado recurrente para capturar contexto ilimitado**.

**Conexión con clases previas del curso IA UC**:
- Clase 11-12 (RNN, LSTM, GRU): Mikolov 2010 es la primera aplicación exitosa de RNNs simples a NLP en gran escala.
- Clase 12 cubre LSTM/GRU como mejora sobre SRN para mitigar vanishing gradients — Mikolov adoptó LSTM en trabajos posteriores.

---

## 9. Cita BibTeX

```bibtex
@inproceedings{mikolov2010rnnlm,
  title={Recurrent neural network based language model},
  author={Mikolov, Tomas and Karafi{\'a}t, Martin and Burget, Luk{\'a}s and Cernock{\`y}, Jan and Khudanpur, Sanjeev},
  booktitle={Interspeech},
  volume={2},
  pages={1045--1048},
  year={2010},
  organization={Makuhari}
}
```

---

## 10. Frase para recordar

> *"Recurrence breaks the fixed-context curse."* — Mikolov 2010 demuestra que reemplazar la ventana fija del NPLM por una RNN simple ya da una mejora dramática. Es el ancestro directo de toda la era de language models recurrentes (LSTM-LM, biLM-ELMo) y eventualmente Transformer-LMs.

---

## 11. Comparación rápida con NPLM (Bengio 2003)

| | NPLM (Bengio 2003) | RNN-LM (Mikolov 2010) |
|---|---|---|
| Contexto | Fijo, $n-1$ palabras | Ilimitado (vía recurrencia) |
| Embeddings | Tabla $C$ explícita | Implícita en pesos $U$ |
| Capa hidden | 1 capa tanh | 1 capa sigmoid recurrente |
| Activación | tanh | sigmoid |
| Speedup vs n-gram | Modesto | 50% PPL reduction posible |
| Tiempo de entrenamiento | Semanas en cluster | Horas en CPU |
| Application en producción | Limitado | Speech recognition AMI/AMI ICSI |

Mikolov 2010 captura tres lecciones del NPLM y las amplifica:
1. Embeddings densos (aquí implícitos en $U$).
2. Autosupervisión (predicción del siguiente token).
3. Cross-entropy + softmax.

Y agrega tres innovaciones propias:
1. **Recurrencia** para contexto ilimitado.
2. **Dynamic learning** durante test.
3. **Rare token consolidation** para vocab manageability.

---

## 12. Lecciones que migran a Word2Vec

Mikolov mismo escribió Word2Vec 3 años después. ¿Qué lecciones del RNN-LM se ven en Word2Vec?

| RNN-LM (2010) | Word2Vec (2013) |
|---|---|
| Vocabulary > 200k es factible | Vocab de 1M+ |
| Cross-entropy + softmax | Softmax (luego negative sampling) |
| Embeddings implícitos útiles | Embeddings **explícitos** como producto principal |
| Speed via BLAS + rare tokens | Speed via negative sampling + subsampling |
| Sin parameter sharing complicado | Sin hidden layer en absoluto |
| Captura dependencias hasta el sentence | Captura solo ventana fija (sacrificio para escalar) |

Word2Vec es esencialmente **RNN-LM simplificado al mínimo viable que conserva embeddings útiles**. Mikolov se da cuenta de que la recurrencia, aunque buena para perplejidad, no es necesaria para aprender embeddings — y la **abandona deliberadamente** para escalar a 30B palabras.

Esta lección se repite en NLP moderno: los **mejores embeddings no necesariamente vienen del mejor LM**. SimCSE, Sentence-BERT, etc. usan objetivos contrastivos que no son strict LM. La intuición se remonta a este paper de 2010.

---

## 13. Aspectos no cubiertos por la clase 18

La clase 18 menciona RNN-LM brevemente en slide 22 como ejemplo de "otra red neuronal" después de NPLM. Lo que vale la pena enfatizar en `profundizacion.md` y `teoria.md` de la clase:

1. **RNN-LM fue un puente clave**: NPLM (2003) → RNN-LM (2010) → Word2Vec (2013) → ELMo (2018) → GPT (2018). Cada paso elimina un cuello de botella del anterior.
2. **Dynamic learning** es una idea elegante que la era moderna ha olvidado — pero conceptualmente es lo que hacen los **continual learning** y **in-context learning** modernos.
3. **Mikolov mismo es figura clave**: trabajó en RNN-LM, luego Word2Vec, luego se fue a Facebook y trabajó en FastText. Su trayectoria define la era 2010-2017 de NLP estadístico/neural.

---

## 14. Apéndice — Implementación PyTorch

Reimplementación moderna del SRN-LM de Mikolov 2010 (didáctica, no para producción):

```python
import torch
import torch.nn as nn

class MikolovRNNLM(nn.Module):
    """
    Simple Recurrent Network LM (Mikolov 2010).
    Note: usa one-hot input + sigmoid, no embedding layer ni LSTM.
    Es la traducción literal del paper.
    """
    def __init__(self, vocab_size: int, hidden_size: int = 100):
        super().__init__()
        # U: pesos input + recurrent → hidden
        self.U = nn.Linear(vocab_size + hidden_size, hidden_size, bias=False)
        # V: pesos hidden → output
        self.V = nn.Linear(hidden_size, vocab_size, bias=False)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        return torch.full((batch_size, self.hidden_size), 0.1)

    def forward(self, token_ids: torch.Tensor):
        # token_ids: [B, T]
        B, T = token_ids.shape
        s = self.init_hidden(B).to(token_ids.device)
        logits = []
        for t in range(T):
            # One-hot encode
            w_t = torch.zeros(B, self.vocab_size, device=token_ids.device)
            w_t.scatter_(1, token_ids[:, t:t+1], 1.0)
            # Concatenate input
            x = torch.cat([w_t, s], dim=-1)
            # Update hidden
            s = torch.sigmoid(self.U(x))
            # Output
            y = self.V(s)  # logits, softmax aplica en loss
            logits.append(y)
        return torch.stack(logits, dim=1)  # [B, T, V]


def train_step(model, batch, optimizer, criterion):
    """Train un step, equivale a truncated BPTT con τ = T."""
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]
    logits = model(input_ids)
    loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

**Diferencia con LSTM moderno**:
- LSTM tiene gates (forget, input, output) que mitigan vanishing gradient.
- LSTM usa embedding layer densa, no one-hot.
- LSTM se entrena con BPTT más profundo (full sequence o $\tau = 50+$).

El SRN de Mikolov es el ancestro directo, simple pero ya **suficiente para superar n-gramas con suficiente datos**.

---

## 15. Tres lecciones que el paper deja para 2026

1. **Más datos no es la única respuesta**: arquitectura importa, especialmente arquitectura que captura dependencias largas. La frase final del paper sigue siendo verdadera: el campo necesitaba modelos mejores, no solo más data.

2. **La autosupervisión a escala vence al feature engineering**: KN5 con suavizado refinado era el estado del arte después de **dos décadas de research en n-gram smoothing**. Un RNN simple sin features lingüísticas lo supera.

3. **El "online learning" es una idea pendiente**: dynamic RNN-LM anticipó conceptualmente el in-context learning, prompt tuning, y memory-augmented LLMs. Es un area donde la práctica moderna podría aprender del pasado.

Mikolov mismo capturó la motivación general: *"Sequential data prediction is considered by many as a key problem in machine learning and artificial intelligence."* — una afirmación que en 2026, con GPT, Claude, Gemini, parece evidente. En 2010 era visionaria.
