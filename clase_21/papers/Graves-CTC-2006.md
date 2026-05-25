---
title: "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"
authors:
  - Alex Graves
  - Santiago Fernández
  - Faustino Gomez
  - Jürgen Schmidhuber
year: 2006
venue: ICML 2006
slug: ctc-graves-2006
tags:
  - ctc
  - sequence-learning
  - rnn
  - asr
  - ocr
  - htr
  - alineamiento
  - forward-backward
  - dynamic-programming
related_papers:
  - shi-crnn-2017
  - hochreiter-lstm-1997
related_fundamentals:
  - ctc-loss
  - rnn-lstm
  - sequence-to-sequence
clase: clase-21
---

## Resumen ejecutivo

El paper de Graves, Fernández, Gomez y Schmidhuber (ICML 2006) introduce **Connectionist Temporal Classification (CTC)**: una función de pérdida y un esquema de decodificación que permiten entrenar redes recurrentes para etiquetar secuencias **sin segmentación previa** entre los frames de entrada $\mathbf{x}$ y los símbolos del target $\mathbf{l}$. La contribución técnica central es triple: (i) ampliar el alfabeto con un símbolo *blank* $\epsilon$, lo que define una distribución sobre paths de longitud $T$ con softmax per-frame; (ii) un mapeo determinista $\mathcal{B}$ que colapsa repeticiones y elimina blanks para recuperar la transcripción; y (iii) un **algoritmo forward-backward** análogo al de los HMM Baum-Welch que computa $p(\mathbf{l}\mid\mathbf{x})$ y sus derivadas en $O(T\,|\mathbf{l}|)$. CTC es discriminativo, end-to-end, y entrena con backpropagation through time estándar. En TIMIT (phoneme recognition con BLSTM) supera tanto al HMM context-dependent como al híbrido HMM-RNN. Es la base de DeepSpeech, CRNN, Wav2Letter, Wav2Vec 2.0 y prácticamente todos los sistemas streaming de ASR/OCR/HTR modernos.

## 1. Contexto histórico (pre-2006)

Para entender por qué CTC fue revolucionario hay que situarse en el panorama del *sequence labelling* hacia 2005-2006.

### 1.1 HMMs como framework dominante

El framework de los **Hidden Markov Models** (Rabiner, 1989) había definido la práctica de reconocimiento de voz durante dos décadas. Un HMM modela una secuencia oculta de estados discretos con transiciones $a_{ij}$ y emisiones $b_j(x_t)$, típicamente Gaussian Mixture Models (GMM) sobre vectores MFCC. El pipeline típico para ASR pre-2006 era:

1. **Forced alignment** sobre el corpus de entrenamiento usando un modelo seed para obtener correspondencia frame-a-state.
2. **Decision tree state tying** para compartir parámetros entre estados acústicamente similares (Young et al. 1994).
3. **Baum-Welch (EM)** para refinar los GMMs.
4. **Decoding** con Viterbi + lexicon FST + language model n-gram.

Los HMMs sufren de tres asunciones cuestionables que el propio paper enumera:

- **Conocimiento experto a priori**: la topología de estados (típicamente HMMs left-to-right de 3 estados por fonema), el tying, y el lexicon requieren ingeniería pesada.
- **Observation independence**: $p(x_t \mid \text{state}_t)$ se asume independiente del resto, lo cual es físicamente falso (los espectros adyacentes están altamente correlacionados).
- **Entrenamiento generativo**: maximizan $p(\mathbf{x}, \mathbf{l})$ aunque la tarea es discriminativa ($p(\mathbf{l}\mid\mathbf{x})$). MMI (Maximum Mutual Information) y MPE (Minimum Phone Error) intentaban paliarlo, pero a costo de complejidad.

### 1.2 Hybrid HMM-ANN

El enfoque **híbrido HMM-MLP** (Bourlard & Morgan, 1994) reemplazaba las emisiones GMM por un MLP que predecía $p(\text{state}_j \mid x_t)$. Para usarlo en el HMM se invertía con regla de Bayes: $p(x_t \mid \text{state}_j) \propto p(\text{state}_j \mid x_t) / p(\text{state}_j)$. El **híbrido HMM-RNN** de Robinson (1994) sustituía el MLP por una recurrent net. Bengio (1999) sistematizó esta familia.

Limitaciones del híbrido:

- Requiere **forced alignment** previo (Viterbi forced) para obtener targets frame-level $\hat y_t$ con los que entrenar la red. La calidad del alignment limita el techo del sistema.
- El HMM impone una **topología discreta** y un *insertion penalty* que hay que tunear.
- La red aprende **clasificación per-frame**, no etiquetado de secuencia: su objetivo es local.

### 1.3 CRFs y discriminative sequence models

**Conditional Random Fields** (Lafferty, McCallum, Pereira, 2001) modelaban directamente $p(\mathbf{l}\mid\mathbf{x})$ con un potential global. Eran discriminativos pero requerían:

- **Feature engineering** explícito (no aprenden representaciones).
- **Markov assumption** sobre las labels (orden $k$).
- Una **segmentación implícita** o explícita del input para definir los potenciales.

### 1.4 RNNs en 2006

Las RNN existían desde Elman (1990), Jordan (1986) y Werbos (1990) había definido BPTT. Hochreiter & Schmidhuber (1997) habían publicado **LSTM** y Schuster & Paliwal (1997) **bidirectional RNNs**. Graves & Schmidhuber (2005) acababan de demostrar que BLSTM superaba MLP/BRNN/LSTM en framewise phoneme classification sobre TIMIT. Pero:

- La función de pérdida era cross-entropy **frame-a-frame**, lo que exige segmentación.
- La inferencia daba una secuencia de softmaxes que había que post-procesar heurísticamente (votar, agrupar repeticiones, etc.).

**El problema central**: no había manera de entrenar una RNN directamente para producir una transcripción cuya alineación con el input es desconocida. Esa es la brecha que CTC cierra.

## 2. El problema formal

Sea $\mathcal{S} \subset \mathcal{D}_{\mathcal{X}\times\mathcal{Z}}$ un conjunto de entrenamiento muestreado iid. El espacio de inputs es $\mathcal{X} = (\mathbb{R}^m)^*$ (secuencias de longitud variable de vectores reales $m$-dimensionales). El espacio de targets es $\mathcal{Z} = L^*$ sobre un alfabeto finito $L$.

Cada par $(\mathbf{x}, \mathbf{l})$ cumple:

- $\mathbf{x} = (x_1, \ldots, x_T)$ con $x_t \in \mathbb{R}^m$ (e.g. MFCC frames).
- $\mathbf{l} = (l_1, \ldots, l_U)$ con $l_u \in L$ (e.g. fonemas) y crucialmente $U \le T$.
- **No conocemos el alineamiento**: no sabemos qué frames del input corresponden a qué label del output.

El objetivo es aprender $h: \mathcal{X} \to \mathcal{Z}$ que minimice un error de transcripción. El paper usa **Label Error Rate (LER)**:

$$
\mathrm{LER}(h, \mathcal{S}') = \frac{1}{Z} \sum_{(\mathbf{x}, \mathbf{l}) \in \mathcal{S}'} \mathrm{ED}\bigl(h(\mathbf{x}), \mathbf{l}\bigr)
$$

donde $\mathrm{ED}$ es la **edit distance** (Levenshtein) y $Z$ el total de labels en el test set. Esta métrica es el progenitor de WER (word error rate) y CER (character error rate) usados hoy en ASR/OCR.

Tareas que encajan en este molde:

- **ASR**: $\mathbf{x}$ = MFCC frames, $\mathbf{l}$ = fonemas/caracteres/wordpieces.
- **HTR (handwriting)**: $\mathbf{x}$ = trayectoria online o columnas de imagen, $\mathbf{l}$ = caracteres.
- **OCR / STR (scene text)**: $\mathbf{x}$ = columnas de CNN feature map, $\mathbf{l}$ = caracteres (esto será exactamente CRNN, Shi et al. 2017).
- **Music / gesture / sign language**.

## 3. Solución CTC

### 3.1 Augmented alphabet

CTC introduce un alfabeto extendido:

$$
L' = L \cup \{\epsilon\}
$$

donde $\epsilon$ es el símbolo **blank** (también escrito `-` o `_`). El blank no significa "silencio acústico" sino "ningún label emitido en este frame".

La red tiene una capa softmax de tamaño $|L'| = |L| + 1$. Para una entrada $\mathbf{x}$ de longitud $T$, la red produce $\mathbf{y} = \mathcal{N}_w(\mathbf{x}) \in (\mathbb{R}^{|L'|})^T$ y se interpreta $y_k^t$ como $p(\text{símbolo } k \text{ en frame } t \mid \mathbf{x})$.

### 3.2 Paths

Un **path** $\pi \in L'^T$ es una secuencia frame-a-frame:

$$
p(\pi \mid \mathbf{x}) = \prod_{t=1}^{T} y_{\pi_t}^t
$$

Asunción crucial: **conditional independence** entre frames dado el estado interno de la red. Para que esto sea coherente, no puede haber feedback del output layer hacia sí mismo o hacia la red. Esta asunción simplificadora es lo que después motivará el RNN-Transducer.

### 3.3 Mapping $\mathcal{B}$

Se define el map many-to-one $\mathcal{B}: L'^T \to L^{\le T}$ en dos pasos:

1. **Colapsar repeticiones consecutivas** del mismo símbolo.
2. **Eliminar blanks**.

Ejemplos canónicos:

- $\mathcal{B}(a\,\epsilon\,a\,b\,\epsilon) = aab$
- $\mathcal{B}(\epsilon\,a\,a\,\epsilon\,\epsilon\,a\,b\,b) = aab$
- $\mathcal{B}(c\,c\,\epsilon\,a\,t) = cat$
- $\mathcal{B}(c\,a\,t) = cat$

**Por qué importa el blank**: sin él, $\mathcal{B}(aa) = a$ por la regla de colapso. Para producir `aa` (doble letra real) hace falta un blank en medio: $\mathcal{B}(a\,\epsilon\,a) = aa$. El blank es el separador que permite distinguir character boundaries cuando el mismo símbolo aparece dos veces seguidas.

### 3.4 Likelihood

La probabilidad de un labelling $\mathbf{l} \in L^{\le T}$ es la suma sobre todas sus pre-imágenes bajo $\mathcal{B}$:

$$
p(\mathbf{l} \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} p(\pi \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} y_{\pi_t}^t
$$

El número de paths que mapean a $\mathbf{l}$ crece exponencialmente con $T$, por lo que la suma directa es intratable. El forward-backward la resuelve en tiempo polinomial.

La pérdida CTC es la **negative log-likelihood**:

$$
\mathcal{L}_{\mathrm{CTC}}(\mathcal{S}, \mathcal{N}_w) = -\sum_{(\mathbf{x}, \mathbf{l}) \in \mathcal{S}} \ln p(\mathbf{l} \mid \mathbf{x})
$$

## 4. Forward-Backward algorithm

### 4.1 Extended sequence

Para manejar blanks insertables entre labels y al principio/fin, se define:

$$
\mathbf{l}' = (\epsilon, l_1, \epsilon, l_2, \epsilon, \ldots, \epsilon, l_U, \epsilon)
$$

de longitud $|\mathbf{l}'| = 2U + 1$. Cada posición $s \in \{1, \ldots, 2U+1\}$ de $\mathbf{l}'$ es o bien un blank ($s$ impar) o el $u$-ésimo label ($s$ par).

### 4.2 Forward variables

Se define $\alpha_t(s)$ como la probabilidad total de **todos los paths** que en el frame $t$ están en la posición $s$ del extended sequence:

$$
\alpha_t(s) = \sum_{\substack{\pi \in L'^T : \\ \mathcal{B}(\pi_{1:t}) = \mathbf{l}_{1:\lceil s/2 \rceil}}} \prod_{t'=1}^{t} y_{\pi_{t'}}^{t'}
$$

**Inicialización** ($t = 1$):

$$
\alpha_1(1) = y_\epsilon^1, \qquad \alpha_1(2) = y_{l_1}^1, \qquad \alpha_1(s) = 0 \ \forall s > 2
$$

Solo se permite empezar en blank o en $l_1$ (no en blanks intermedios o labels posteriores).

**Recursión**. Sea $\bar\alpha_t(s) = \alpha_{t-1}(s) + \alpha_{t-1}(s-1)$. Entonces:

$$
\alpha_t(s) =
\begin{cases}
\bar\alpha_t(s) \cdot y_{l'_s}^t & \text{si } l'_s = \epsilon \ \text{ o }\ l'_{s-2} = l'_s \\
\bigl(\bar\alpha_t(s) + \alpha_{t-1}(s-2)\bigr) \cdot y_{l'_s}^t & \text{en otro caso}
\end{cases}
$$

Interpretación de los tres orígenes:

- $\alpha_{t-1}(s)$: el path estaba en $s$ y repite (consume frame extra sin avanzar).
- $\alpha_{t-1}(s-1)$: el path estaba en $s-1$ (un blank o un label distinto) y transiciona.
- $\alpha_{t-1}(s-2)$: el path estaba en $s-2$ y **salta** el blank entre ambos labels. Solo permitido si $l'_s$ no es blank y $l'_{s-2} \ne l'_s$ (si fueran iguales, saltar el blank los colapsaría en uno).

Restricciones de tracking eficiente:

- $\alpha_t(s) = 0$ si $s < |\mathbf{l}'| - 2(T-t) - 1$: no hay frames suficientes para completar el resto.
- $\alpha_t(s) = 0$ si $s < 1$.

**Probabilidad total**:

$$
p(\mathbf{l} \mid \mathbf{x}) = \alpha_T(|\mathbf{l}'|) + \alpha_T(|\mathbf{l}'|-1)
$$

(terminamos o en el blank final o en $l_U$).

### 4.3 Backward variables

Dualmente, $\beta_t(s)$ = probabilidad de los paths que en $t$ están en $s$ y completan correctamente la cola del labelling. Sea $\bar\beta_t(s) = \beta_{t+1}(s) + \beta_{t+1}(s+1)$:

$$
\beta_T(|\mathbf{l}'|) = y_\epsilon^T, \qquad \beta_T(|\mathbf{l}'|-1) = y_{l_U}^T, \qquad \beta_T(s) = 0 \ \forall s < |\mathbf{l}'|-1
$$

$$
\beta_t(s) =
\begin{cases}
\bar\beta_t(s) \cdot y_{l'_s}^t & \text{si } l'_s = \epsilon \ \text{ o }\ l'_{s+2} = l'_s \\
\bigl(\bar\beta_t(s) + \beta_{t+1}(s+2)\bigr) \cdot y_{l'_s}^t & \text{en otro caso}
\end{cases}
$$

### 4.4 Complejidad y rescaling

Cada paso de recursión tiene $O(|\mathbf{l}'|) = O(U)$ operaciones, y se ejecuta para $T$ frames: total $O(T\,U)$ por sample. Esto es **idéntica complejidad** al forward-backward de un HMM con $2U+1$ estados, lo que era un argumento políticamente importante en 2006: CTC no es asintóticamente más caro que el baseline.

**Underflow**: los productos $\alpha_t(s) = \prod_{t'} y_{\pi_{t'}}^{t'}$ son productos de $T$ probabilidades, lo que para $T \sim 100$ ya colapsa a cero en float32. El paper sigue la receta clásica (Rabiner 1989):

$$
C_t = \sum_s \alpha_t(s), \qquad \hat\alpha_t(s) = \alpha_t(s)/C_t
$$

y análogamente $D_t, \hat\beta_t$. La log-verosimilitud queda como:

$$
\ln p(\mathbf{l} \mid \mathbf{x}) = \sum_{t=1}^{T} \ln C_t
$$

### 4.5 Conexión con HMM Baum-Welch

El algoritmo es estructuralmente idéntico al forward-backward para HMMs:

- En HMM, $\alpha_t(j) = p(x_1, \ldots, x_t, \text{state}_t = j)$.
- En CTC, $\alpha_t(s)$ es la suma de paths que llegan a la posición $s$ del extended sequence en el frame $t$.

La diferencia clave: en HMM las "transiciones" son parámetros aprendidos $a_{ij}$, mientras que en CTC las transiciones son **estructurales** (determinadas por la topología del extended sequence y el blank). En CTC, **toda la modelación está en la red**; el grafo de transiciones es fijo y mínimo.

## 5. Gradiente

### 5.1 De $p(\mathbf{l}\mid\mathbf{x})$ a derivadas

El paper establece la identidad clave:

$$
\alpha_t(s)\,\beta_t(s) = \sum_{\substack{\pi \in \mathcal{B}^{-1}(\mathbf{l}) \\ \pi_t = l'_s}} y_{l'_s}^t \prod_{t'=1}^{T} y_{\pi_{t'}}^{t'}
$$

Esto es la probabilidad de **todos los paths que pasan por $l'_s$ en frame $t$**. Reagrupando:

$$
\frac{\alpha_t(s)\,\beta_t(s)}{y_{l'_s}^t} = \sum_{\substack{\pi \in \mathcal{B}^{-1}(\mathbf{l}) \\ \pi_t = l'_s}} p(\pi \mid \mathbf{x})
$$

Sumando sobre $s$ recuperamos:

$$
p(\mathbf{l} \mid \mathbf{x}) = \sum_{s=1}^{|\mathbf{l}'|} \frac{\alpha_t(s)\,\beta_t(s)}{y_{l'_s}^t} \qquad \forall t
$$

(esta identidad debe valer para cualquier $t$; en la práctica se usa el $t$ con mejor estabilidad numérica).

### 5.2 Derivada respecto a softmax output

Definiendo $\mathrm{lab}(\mathbf{l}, k) = \{s : l'_s = k\}$ (las posiciones del extended sequence donde aparece el símbolo $k$, que pueden ser varias):

$$
\frac{\partial p(\mathbf{l} \mid \mathbf{x})}{\partial y_k^t} = \frac{1}{(y_k^t)^2} \sum_{s \in \mathrm{lab}(\mathbf{l}, k)} \alpha_t(s)\,\beta_t(s)
$$

Y usando $\partial \ln p / \partial y = (1/p)\,\partial p / \partial y$:

$$
\frac{\partial \ln p(\mathbf{l} \mid \mathbf{x})}{\partial y_k^t} = \frac{1}{p(\mathbf{l}\mid\mathbf{x})\,(y_k^t)^2} \sum_{s \in \mathrm{lab}(\mathbf{l}, k)} \alpha_t(s)\,\beta_t(s)
$$

### 5.3 Error signal pre-softmax

Sea $u_k^t$ el output **pre-softmax** (logit). Por la derivada del softmax $\partial y_k/\partial u_j = y_k(\delta_{kj} - y_j)$ y usando los rescaled $\hat\alpha, \hat\beta$, el paper deriva la forma compacta:

$$
\boxed{\frac{\partial \mathcal{L}_{\mathrm{CTC}}}{\partial u_k^t} = y_k^t - \frac{1}{y_k^t\,Z_t} \sum_{s \in \mathrm{lab}(\mathbf{l}, k)} \hat\alpha_t(s)\,\hat\beta_t(s)}
$$

con $Z_t = \sum_{s} \hat\alpha_t(s)\hat\beta_t(s) / y_{l'_s}^t$.

Este "error signal" es el ingrediente que se inyecta al BPTT de la red recurrente. Interpretación: el target en frame $t$ para la unidad $k$ es la **posterior probability** marginal de emitir $k$ en $t$ dado el labelling completo, computada por el forward-backward; el gradiente es la diferencia entre el output actual $y_k^t$ y ese target soft.

La Figura 4 del paper visualiza la evolución del error signal: al inicio (red random) los blanks ganan; conforme entrena, los spikes correctos emergen y el error se localiza alrededor de ellos; al converger los spikes son altos y el error se desvanece.

## 6. Decoding

Dado un modelo entrenado, queremos $h(\mathbf{x}) = \arg\max_{\mathbf{l}} p(\mathbf{l}\mid\mathbf{x})$. El paper reconoce que **no se conoce un algoritmo general tractable** para este problema (en efecto, posteriormente se demostraría que es NP-hard exacto sobre un grafo arbitrario).

### 6.1 Best path (greedy) decoding

$$
\pi^* = \arg\max_{\pi \in L'^T} p(\pi \mid \mathbf{x}) = \Bigl(\arg\max_{k} y_k^t\Bigr)_{t=1}^{T}, \qquad h(\mathbf{x}) \approx \mathcal{B}(\pi^*)
$$

- **Complejidad**: $O(T \cdot |L'|)$.
- **No óptimo**: el path más probable no necesariamente corresponde al labelling más probable, porque un labelling puede tener muchos paths cada uno de baja probabilidad cuya suma supera al path "monolítico" del greedy.

Sin embargo, en redes bien entrenadas con **peaky outputs** (que es lo que típicamente se observa), best path es excelente aproximación.

### 6.2 Prefix search decoding

Aprovecha que el forward-backward modificado puede computar $p(\text{labelling con prefijo } \mathbf{p} \mid \mathbf{x})$. Se construye un árbol de prefijos, expandiendo en cada iteración el prefijo más probable restante (Figura 2 del paper). Termina cuando un labelling completo supera la probabilidad de cualquier prefijo restante.

- **Garantía teórica**: dado tiempo suficiente, encuentra el labelling óptimo.
- **Complejidad práctica**: exponencial en el peor caso. El paper introduce una heurística: dividir el output en secciones por umbral de blank ($p(\epsilon) > 0.9999$) y aplicar prefix search en cada sección.

### 6.3 Beam search con language model

Aunque no está en el paper original, pronto se popularizó la extensión que combina CTC con un language model:

$$
\hat{\mathbf{l}} = \arg\max_{\mathbf{l}}\Bigl[\ln p_{\mathrm{CTC}}(\mathbf{l}\mid\mathbf{x}) + \alpha \ln p_{\mathrm{LM}}(\mathbf{l}) + \beta\,|\mathbf{l}|\Bigr]
$$

(con $\alpha$ = LM weight, $\beta$ = word insertion bonus). Esta forma es la usada en DeepSpeech, Wav2Letter, etc. Hannun et al. (2014) publicaría un beam search específico para CTC con shallow LM fusion que se volvería estándar.

### 6.4 NP-hardness del decoding óptimo

Aunque el paper no lo demuestra, posteriormente se estableció que decodificar exactamente $\arg\max_{\mathbf{l}} p(\mathbf{l}\mid\mathbf{x})$ en CTC con LM externo de orden arbitrario es NP-hard. Las soluciones prácticas son: best path, prefix search con heurísticas, o beam search aproximado.

## 7. Experimentos: TIMIT phoneme recognition

### 7.1 Setup

- **Corpus**: TIMIT, 4620 utterances de entrenamiento, 1680 de test, 184 (5%) validation. 61 fonemas distintos.
- **Features**: ventanas de 10 ms con 5 ms overlap, 12 MFCCs + log-energy + first derivatives = 26 coeficientes por frame, normalizados a media 0 y std 1.
- **Arquitectura CTC**: BLSTM con 100 LSTM blocks por dirección, peepholes + forget gates (Gers et al. 2002), tanh para input/output cell, sigmoide para gates. Input 26 → BLSTM(100+100) → softmax 62 (61 + blank). Total: 114,662 weights.
- **Training**: BPTT online (update tras cada utterance), lr=$10^{-4}$, momentum=0.9, Gaussian input noise std=0.6 (data augmentation/regularización), weights init $U(-0.1, 0.1)$. Early stopping sobre validation.
- **Prefix search threshold**: $p(\epsilon) > 0.9999$.

### 7.2 Baselines

- **HMM context-independent**: 3-state left-to-right por fonema, GMM emissions, HTK Toolkit, >900,000 parámetros (con CD).
- **HMM context-dependent**: con triphones.
- **Híbrido BLSTM-HMM**: mismo BLSTM (sin blank, 61 outputs) entrenado con Viterbi forced alignment. lr=$10^{-5}$, noise std=0.5. 114,461 weights + 183 del HMM.
- **Weighted error BLSTM-HMM**: scaling del error para igualar contribución de fonemas largos y cortos.

### 7.3 Resultados

| Sistema                          | LER                 |
|----------------------------------|---------------------|
| HMM context-independent          | 38.85 %             |
| HMM context-dependent            | 35.21 %             |
| BLSTM/HMM híbrido                | 33.84 ± 0.06 %      |
| Weighted error BLSTM/HMM         | 31.57 ± 0.06 %      |
| **CTC (best path)**              | 31.47 ± 0.21 %      |
| **CTC (prefix search)**          | **30.51 ± 0.19 %**  |

Lecturas:

- CTC supera al híbrido BLSTM-HMM con la misma arquitectura RNN. Esto es la prueba decisiva: no es la arquitectura, es la objetivo.
- CTC supera al CD-HMM con 8× más parámetros, sin usar lexicon, sin LM, sin context-dependent state tying.
- Prefix search da una mejora pequeña pero consistente sobre best path (~1 pp).
- El weighted-error trick es necesario para el híbrido pero **innecesario** para CTC, porque la pérdida CTC no depende de duración/segmentación.
- CTC fue más sensible al input noise y se benefició de un nivel más alto (0.6 vs 0.5 del híbrido).

## 8. Propiedades teóricas y limitaciones

### 8.1 Conditional independence

La asunción $p(\pi\mid\mathbf{x}) = \prod_t y_{\pi_t}^t$ trata cada frame independiente dado el estado interno de la red. Para una RNN bidireccional, el estado interno **sí captura contexto pasado y futuro**, así que en términos de la red la asunción es razonable. Pero en términos del output layer: dos timesteps consecutivos no están correlacionados condicional al input, lo cual es una **simplificación deliberada**. Las consecuencias:

- CTC **no modela explícitamente dependencias entre labels**. No hay un equivalente a la transition matrix de HMM.
- Para incorporar dependencias label-level se necesita un LM externo (shallow fusion en decoding) o cambiar de modelo (RNN-T).

### 8.2 Implicit label dependency

Pese a lo anterior, el paper observa que CTC **modela implícitamente** dependencias inter-label vía el estado interno de la RNN. Ejemplo en Figura 1: el cluster `dcl + d` (closure + stop) aparece como un *double spike* porque la red aprendió esa co-ocurrencia frecuente del inglés.

### 8.3 Blank dominance / peaky outputs

Una propiedad empírica robusta: tras entrenar, el output de una CTC suele estar dominado por blanks con spikes muy puntuales en los frames donde el modelo está confiado de emitir. La distribución no es uniforme: es **peaky**. Esto:

- Hace que best path decoding sea suficiente en la mayoría de casos.
- Permite la heurística de segmentación por umbral de blank para prefix search.
- Pero implica que CTC **no es un buen modelo de duración/segmentación fina**: los spikes no se alinean con boundaries acústicos verdaderos (no son timestamps confiables sin post-procesamiento).

Trabajos posteriores (Sak et al. 2015, Senior et al. 2015) caracterizaron formalmente la peakiness.

### 8.4 Length constraint

CTC asume $|\mathbf{l}| \le T$. En la práctica con el blank obligatorio entre repeticiones, la constraint efectiva es $|\mathbf{l}'| = 2|\mathbf{l}|+1 \le T$, es decir $|\mathbf{l}| \le (T-1)/2$. Esto es problemático cuando:

- El input frame rate es bajo (e.g. downsampling agresivo en encoders modernos).
- El target es muy largo respecto al input (e.g. summarization, traducción).

Para ASR con frames de 10 ms y characters, $T \approx 100\,\text{frames/seg}$ y caracteres $\approx 15$/seg en habla normal: hay margen sobrado. Para wordpieces o words ya empieza a apretar y muchos encoders modernos usan stride 2 en lugar de 4.

### 8.5 Otras limitaciones

- **No segmentation output**: si la tarea requiere boundaries (e.g. forced alignment para evaluación), CTC no los entrega de forma confiable.
- **Sensibilidad a class imbalance**: el blank domina; sin cuidado el modelo puede colapsar a emitir solo blanks.
- **Sin condicionamiento autoregresivo del output**: cada frame es independiente, lo que limita la expresividad en tareas con fuerte estructura output-side.

## 9. Variantes y extensiones

### 9.1 RNN-Transducer (Graves 2012, 2013)

**RNN-T** elimina la conditional independence ampliando el output: en lugar de un softmax per frame, define un **joint network** sobre $(T \times U)$ que combina un encoder (sobre $\mathbf{x}$) y un *prediction network* autoregresivo (sobre $\mathbf{l}_{1:u-1}$). Hay un nuevo forward-backward sobre lattice 2D. Ventajas:

- Modela $p(\mathbf{l}\mid\mathbf{x})$ sin asumir independencia entre símbolos.
- Sigue siendo **streaming** (causal): el prediction network solo ve labels pasados.
- Es la arquitectura **estándar de production ASR en Google, Amazon, Microsoft, Meta** desde ~2019.

### 9.2 Attention-based seq2seq (LAS, Chan et al. 2016)

**Listen-Attend-Spell** abandona el blank y usa attention softmax sobre frames para cada paso del decoder autoregresivo. Ventajas: no asume monotonicidad ni la length constraint de CTC. Desventajas: no es streaming nativo, requiere todo el input para empezar a decodificar.

### 9.3 CTC + Attention hybrid (Watanabe et al. 2017)

ESPnet popularizó el **hybrid CTC/attention**: comparte encoder, suma una loss CTC y una attention loss con peso $\lambda$:

$$
\mathcal{L} = \lambda \mathcal{L}_{\mathrm{CTC}} + (1-\lambda) \mathcal{L}_{\mathrm{att}}
$$

CTC actúa como regularizador monotonic (fuerza alineación lineal) y acelera la convergencia. En decoding ambos scores se combinan. Esta receta supera a cada componente por separado y es estándar hoy.

### 9.4 Wav2Vec 2.0 + CTC fine-tuning

Modelos foundation de speech como **Wav2Vec 2.0** (Baevski et al. 2020) y **HuBERT** (Hsu et al. 2021) se pre-entrenan con contrastive/masked tasks self-supervised y se **fine-tunean con CTC head** sobre transcripciones supervisadas. CTC sigue siendo el output layer de fine-tuning más simple y competitivo. Whisper (Radford 2022) se aleja a seq2seq decoder, pero los modelos production-friendly (latencia, streaming) siguen con CTC o RNN-T.

### 9.5 Otras variantes

- **Maximum Mutual Information CTC** (Povey et al. 2016): combina CTC con MMI sequence training.
- **Margin-based CTC, Focal CTC**: ajustan la loss para combatir blank dominance.
- **EnCodec / Soft alignment CTC**: relajan la asunción de independencia.

## 10. Aplicaciones masivas

### 10.1 DeepSpeech (Hannun et al. 2014, Baidu/Mozilla)

Primer sistema end-to-end de ASR a escala publicado: arquitectura simple (5 capas FC + BiRNN), CTC loss, beam search con LM externo. Demostró que con suficientes datos y compute, CTC + RNN simple supera HMM con décadas de ingeniería. Mozilla DeepSpeech (open source) y Facebook **Wav2Letter** son sus descendientes directos.

### 10.2 CRNN (Shi, Bai, Yao 2017)

**CRNN** (Convolutional Recurrent Neural Network) aplica CTC a **scene text recognition**: una CNN extrae features de la imagen del texto, las columnas del feature map se pasan a una BiLSTM y un CTC head decodifica caracteres. Es el ejemplar canónico de CTC en visión y la baseline obligatoria de cualquier paper de STR. Está directamente cubierto en clase 21.

### 10.3 Multidimensional CTC en handwriting

El propio Graves extendió CTC a **multidimensional RNN + CTC** para handwriting offline (escaneado), donde el input es una imagen 2D. Logró state-of-the-art en IAM y dominó las competencias ICDAR HTR durante años.

### 10.4 Otras aplicaciones

- **Music transcription** (Sigtia et al.): polyphonic music → notas.
- **Gesture/sign language recognition**.
- **Lip reading** (LipNet, Assael et al. 2016).
- **Action segmentation** en video.
- **Bio-signal labelling** (EEG, ECG arrhythmia detection).

CTC sigue siendo competitivo en **streaming low-latency ASR** (Google Assistant, dictado en mobile) precisamente porque su asunción de independencia entre frames lo hace naturalmente causal y compatible con outputs en tiempo real.

## 11. Conexión con la clase 21

La clase 21 cubre **Scene Text Recognition (STR)**. La slide "Text Recognition Stages" del PDF de clase descompone un sistema STR en cuatro etapas: Transformation, Feature Extraction, Sequence Modeling, Prediction. La etapa **Prediction** se divide canónicamente en dos opciones:

1. **CTC**: la opción de CRNN (Shi 2017). Asume alineación monotonic implícita. Más rápida y simple. Cubierta exhaustivamente por el presente paper.
2. **Attention**: la opción de ASTER, MORAN, ABCNet (Liu 2020). Más flexible para texto irregular (curvo, perspectivado).

El contraste pedagógico entre ambas es central:

- **CTC ventajas**: simple, rápido, streaming-friendly, beam search ligero, gradient bien definido para texto recto o moderadamente curvo.
- **CTC limitaciones**: monotonic only (problemático para texto curvado con rotaciones >90°, vertical, o reverso), no condicionamiento autoregresivo entre caracteres.
- **Attention ventajas**: maneja texto irregular, condicionamiento autoregresivo full.
- **Attention desventajas**: más caro, propenso a errores de alineación en texto largo, no streaming.

Entender CTC permite entender por qué attention se vuelve preferible en STR irregular y por qué hybrids CTC+attention son comunes hoy. La conexión se concreta también con el fundamento transversal `ctc-loss.md` que profundiza el forward-backward con ejemplos numéricos.

## 12. Timeline de dominio

| Año | Hito | Comentario |
| --- | --- | --- |
| 1989 | HMM tutorial (Rabiner) | Framework dominante de sequence labelling |
| 1994 | HMM-RNN hybrid (Bourlard, Robinson) | Mejor que GMM pero requiere forced alignment |
| 1997 | LSTM (Hochreiter & Schmidhuber) | Soluciona vanishing gradient en RNN |
| 1997 | BRNN (Schuster & Paliwal) | Contexto bidireccional |
| 2001 | CRF (Lafferty et al.) | Sequence labelling discriminativo, pero feature-engineered |
| 2005 | BLSTM framewise (Graves & Schmidhuber) | Estado del arte en classification per-frame |
| **2006** | **CTC (Graves et al.)** | **End-to-end RNN sequence labelling sin alignment** |
| 2009 | Multidimensional CTC (Graves) | Handwriting offline SOTA |
| 2012 | RNN-Transducer (Graves) | Quita conditional independence, autoregresivo en label-side |
| 2014 | DeepSpeech (Hannun et al.) | ASR end-to-end a escala con CTC |
| 2015 | Attention seq2seq (Bahdanau) | Alternativa a CTC |
| 2016 | LAS (Chan et al.) | Attention en ASR |
| 2017 | CRNN (Shi et al.) | CTC en scene text recognition |
| 2017 | CTC/Attention hybrid (Watanabe) | ESPnet standard |
| 2020 | Wav2Vec 2.0 + CTC fine-tuning | Foundation model + CTC head |
| 2022 | Whisper (Radford et al.) | Seq2seq decoder gana en ASR offline, pero CTC sigue en streaming |
| 2024 | RNN-T en producción Google/Amazon | CTC influye toda la familia transducer |

## 13. Worked example: forward-backward sobre "CAT"

Para fijar intuición, sigamos el ejemplo de la Figura 3 del paper: target $\mathbf{l} = \text{CAT}$, $|\mathbf{l}| = 3$. El extended sequence es:

$$
\mathbf{l}' = (\epsilon, C, \epsilon, A, \epsilon, T, \epsilon), \qquad |\mathbf{l}'| = 7
$$

Supongamos $T = 6$ frames. El grafo de transiciones permitidas tiene $|\mathbf{l}'| \times T = 7 \times 6 = 42$ celdas, pero muchas son inalcanzables.

**Restricciones de alcanzabilidad**:

- Posiciones iniciales accesibles en $t=1$: solo $s=1$ (blank inicial) o $s=2$ (la C).
- Para terminar correctamente en $t=T$: solo $s=6$ (T) o $s=7$ (blank final) pueden cerrar.
- Las "celdas no conectadas" arriba-derecha y abajo-izquierda de la Figura 3 corresponden a:
  - **Arriba-derecha**: estados muy avanzados en $\mathbf{l}'$ con $t$ pequeño (no se llegó tan lejos).
  - **Abajo-izquierda**: estados muy atrasados con $t$ grande (no quedaría tiempo).

**Transiciones permitidas desde $(t, s)$ hacia $(t+1, s')$**:

- $s' = s$: quedarse (emite el mismo símbolo o blank).
- $s' = s+1$: avanzar al siguiente símbolo del extended sequence.
- $s' = s+2$: saltar el blank intermedio, **solo si** $l'_{s+2} \ne l'_s$ y $l'_{s+2}$ no es blank.

En CAT todos los caracteres son distintos, así que los skips de 2 están todos permitidos. Si el target fuera $\mathbf{l} = \text{CC}$, $\mathbf{l}' = (\epsilon, C, \epsilon, C, \epsilon)$ y NO se permite saltar de $s=2$ a $s=4$ (ambos son C: colapsarían en una sola C, lo que correspondería al labelling "C" en vez de "CC"). Este es el rol estructural del blank obligatorio.

**Conteo de paths para CAT con $T=6$**:

Un análisis combinatorio: cada path debe pasar por C, A, T en orden, con blanks/repeticiones libres entre medio. El número de paths es el coeficiente combinatorio de distribuir $T - |\mathbf{l}| = 3$ frames "extra" entre las posiciones permitidas, considerando blanks insertables. Para CAT con $T=6$ son del orden de varias docenas; para $T=100$ ya son astronómicos. **Esta explosión combinatoria es lo que justifica el forward-backward**.

Cada celda $\alpha_t(s)$ acumula la masa de todos esos paths que pasan por $(s, t)$. El producto $\alpha_t(s)\,\beta_t(s)/y_{l'_s}^t$ es la masa de paths que **pasan por $l'_s$ exactamente en $t$**, y sumar sobre $s$ recupera $p(\mathbf{l}\mid\mathbf{x})$.

## 14. Detalles de implementación práctica

Para quien implementa CTC desde cero (no usando `nn.CTCLoss`), los puntos sutiles son:

### 14.1 Padding de secuencias variables

En batches, los inputs $\mathbf{x}^{(b)}$ tienen longitudes $T_b$ distintas y los targets $\mathbf{l}^{(b)}$ longitudes $U_b$ distintas. PyTorch resuelve esto con `input_lengths` y `target_lengths` por sample: el forward-backward se computa sobre la región válida de cada uno. Crucial: **no incluir padding en el cómputo** o la gradiente se contamina.

### 14.2 Log-space computation

Lo más numéricamente estable es trabajar enteramente en log-space:

$$
\log\alpha_t(s) = \log y_{l'_s}^t + \mathrm{logsumexp}(\log\alpha_{t-1}(s),\ \log\alpha_{t-1}(s-1),\ \log\alpha_{t-1}(s-2))
$$

con el `logsumexp` truco $\log(\sum_i e^{x_i}) = \max_i x_i + \log\sum_i e^{x_i - \max}$. Las implementaciones modernas (PyTorch, TensorFlow) usan log-space por defecto y no el rescaling tradicional.

### 14.3 Blank index convention

PyTorch usa `blank=0` por defecto: el índice 0 del vocabulario es blank y los caracteres reales empiezan en 1. Otras implementaciones usan `blank=|L|` (el último índice). **Bug común**: confundir convenciones produce gradientes incorrectos y entrenamiento que no converge.

### 14.4 Initial learning rate y warmup

CTC tiene una dinámica peculiar de entrenamiento: las primeras épocas la red colapsa a emitir solo blanks (porque es el "atractor" trivial: predecir blank siempre da $p \approx 0$ pero no es $-\infty$). Solo cuando los gradientes de los símbolos no-blank empiezan a dominar, los spikes emergen. Recetas comunes:

- **Warmup learning rate**: arrancar con lr baja, subir gradualmente.
- **Curriculum**: empezar con secuencias cortas, alargar.
- **Label smoothing sobre blank**: penalizar suavemente la confianza en blank.
- **Focal CTC**: bajar el peso de blanks dominantes.

### 14.5 Receptive field y stride

Si el encoder tiene stride $s$ (e.g. CNN con strided convolutions o pooling), el frame-rate efectivo del output es $T/s$. Hay que asegurar $|\mathbf{l}'| = 2U+1 \le T/s$. Para STR con texto largo en imágenes pequeñas, el stride debe calibrarse cuidadosamente.

### 14.6 GPU implementation

WarpCTC (Baidu, 2016) introdujo un kernel CUDA que paraleliza el forward-backward sobre el batch y sobre las posiciones del extended sequence. Es decenas de veces más rápido que implementaciones naive en Python. PyTorch lo integró nativamente en `torch.nn.CTCLoss`.

## 15. Lecciones generales del paper

Más allá del aporte técnico específico, CTC es ejemplar metodológicamente:

1. **Separar el modelo (RNN) de la objective (loss)**: la RNN no cambia, solo cambia cómo se computa la pérdida y el gradiente. Esto es la receta general "diferentiable surrogate loss" que aparece en políticas de RL, en attention, en CRF neural.
2. **Marginalizar latentes con DP**: cuando una variable latente (aquí el alineamiento) tiene estructura aprovechable, marginalizarla con dynamic programming es preferible a sample-based estimators (EM hard, REINFORCE) cuando la suma es tratable.
3. **Discriminative end-to-end**: en 2006 era radical sustituir un pipeline modular bien entendido (HMM-MFCC-GMM-LM) por una red entrenada end-to-end. CTC plantó la semilla que florecería en DeepSpeech, en seq2seq attention, en transformer ASR.
4. **Asunciones simplificadoras útiles**: la conditional independence entre frames no es realista, pero es **suficiente para extraer la señal** y permite el algoritmo polinomial. Subsecuentes refinamientos (RNN-T) la relajan, pero solo cuando vale la pena el costo.

## 16. Comparación cuantitativa CTC vs alternativas modernas

Para situar a CTC respecto a su descendencia, conviene una comparación de propiedades:

| Propiedad | CTC | RNN-T | Attention seq2seq (LAS) | Hybrid CTC/Att |
| --- | --- | --- | --- | --- |
| Condicional independence frames | Sí | No | No | Mixto |
| Alineación implícita | Monotonic | Monotonic | Soft (no monotonic) | Monotonic dominante |
| Streaming-friendly | Sí | Sí | No | Parcial |
| Decoding autoregresivo en labels | No | Sí | Sí | Sí (att rama) |
| Necesita LM externo para SOTA | Frecuente | Beneficial | Beneficial | Beneficial |
| Length constraint sobre target | Sí (estricto) | No (más flexible) | No | Sí (por CTC arm) |
| Costo training por sample | O(T·U) | O(T·U·V) | O(T·U) | O(T·U) |
| Memoria training | Baja | Alta (joint network) | Media | Media-alta |
| Convergencia desde scratch | Buena | Difícil (necesita CTC warmup) | Difícil | Excelente |
| Madurez en producción (2024) | Muy alta | Muy alta | Media (offline) | Alta |

Observación clave para production: RNN-T es preferido en mobile/edge ASR por su buen trade-off latencia/calidad, pero **suele inicializarse con CTC warmup** porque entrenar RNN-T desde scratch es notoriamente difícil. CTC sigue siendo el "primer paso" del pipeline incluso cuando no es la objective final.

## 17. Pitfalls comunes al implementar CTC

Recopilo errores frecuentes que vale la pena conocer antes de implementar o debuggear CTC:

1. **Blank index mismatch**: usar `blank=0` en el cómputo de loss pero ordenar el vocabulario asumiendo que el blank está al final (o viceversa). Síntoma: entrenamiento que diverge o se estanca en blanks.
2. **Targets con padding incluido**: si el batch tiene targets de distinta longitud y se pasan rellenados con un padding-id que coincide con el blank, el forward-backward computa mal. Hay que pasar `target_lengths` correctamente.
3. **$T < |\mathbf{l}'| = 2U+1$**: si algún sample del batch no satisface la constraint, la loss devuelve `inf`. Solución: filtrar samples problemáticos o aumentar el frame rate del encoder (menos stride).
4. **CNN stride excesivo**: en STR/HTR, si la CNN reduce el ancho de la imagen a menos de $2U+1$ "columnas", CTC no puede entrenar. Cuidado con downsampling agresivo.
5. **Confundir log-softmax con softmax al input de `nn.CTCLoss`**: PyTorch espera log-probabilities. Pasar probabilities lineales o logits crudos produce gradientes incorrectos.
6. **No reset de hidden state entre batches**: en BLSTM/LSTM streaming, olvidar reset hace que el estado del sample anterior contamine el siguiente.
7. **Mezclar GPU determinism off con CTC**: el kernel CUDA de CTC es no determinístico por defecto. Para reproducibilidad estricta, hay que activar `torch.use_deterministic_algorithms(True)` con la penalización de velocidad asociada.
8. **Best-path decoding asumiendo timestamps confiables**: los spikes de CTC NO son timestamps acústicos verdaderos por la peakiness. Para forced alignment, usar el forward-backward y elegir el Viterbi argmax sobre la lattice, no el best path frame-level.

## 18. Influencia en otros dominios del curso IA UC

CTC no es solo un truco de speech: su patrón conceptual aparece transversalmente:

- **Clase 14 Transformers**: el masked language modeling de BERT comparte la idea de marginalizar sobre estructuras latentes (qué tokens están enmascarados). El forward-backward exacto da paso a aproximaciones estocásticas, pero el principio es el mismo.
- **Clase 17 Pose Recognition**: OpenPose y PifPaf usan loss functions que marginalizan sobre asignaciones latentes (qué keypoints pertenecen a qué persona). El paralelismo con CTC es que la asignación es latente y la suma se hace con DP.
- **Clase 18 Word Embeddings**: word2vec con negative sampling es un approximation a una marginalización sobre el vocabulario. CTC no negativa-samplea pero comparte el espíritu de marginalizar lo latente.
- **Clase 20 ELMo/BERT/GPT**: el RLHF de ChatGPT marginaliza implícitamente sobre preferencias humanas latentes. La maquinaria es distinta pero la mentalidad de "los humanos no anotan alineamiento, infiérelo" es genealógicamente cercana a CTC.

La transferibilidad del CTC pattern es enorme: cualquier task de **alineamiento monotonic con longitudes desiguales** es candidato natural. En el dominio FHIR/MDM de Roberto, ejemplos plausibles:

- Alineación de campos demográficos con campos canónicos cuando el orden se preserva pero hay omisiones (CTC degenera a edit distance pero con representaciones aprendidas).
- Mapeo de notas clínicas libres a códigos ICD-10/CIE-10 estructurados: una secuencia de tokens médicos → una secuencia más corta de códigos, alineación monotonic, sin supervisión de alignment.
- Annotation automática de timelines clínicos: eventos en orden cronológico mapeados a fases/diagnósticos.

## 19. Recursos complementarios

- Distill.pub: "Sequence Modeling with CTC" (Hannun, 2017): explicación visual ejemplar con animaciones de las celdas $\alpha_t(s)$.
- Graves, "Supervised Sequence Labelling with Recurrent Neural Networks" (2012): el libro que sistematiza CTC, RNN-T, multidimensional RNN.
- PyTorch `torch.nn.CTCLoss` y `torchaudio.functional.forced_align`: implementaciones canónicas.
- WarpCTC (Baidu): CUDA kernel altamente optimizado, ahora integrado en PyTorch.
- ESPnet: framework de ASR con CTC, attention y hybrid out-of-the-box.

## 14. Cierre

CTC resuelve un problema fundacional —entrenar redes recurrentes para etiquetar secuencias sin alineamiento previo— con un mecanismo matemáticamente elegante: ampliar el alfabeto con un blank, definir una distribución sobre paths, marginalizar con forward-backward. La idea es transferible a cualquier dominio donde input y output sean secuencias monotonicamente alineadas pero no isomorfas en longitud.

Veinte años después de su publicación, CTC sigue presente en producción: en el head de fine-tuning de Wav2Vec 2.0, en streaming ASR, en CRNN para OCR, en sistemas low-latency donde el costo de un decoder autoregresivo no se puede pagar. Su influencia teórica es aún mayor: estableció el patrón **"marginalizar sobre alineamientos con dynamic programming"** que reaparece en RNN-T, en CTC-segmental, en pointer networks, y conceptualmente en las masked-prediction objectives de los foundation models.

Para Roberto, ingeniero FHIR construyendo pipelines de matching, la lección transferible no está solo en ASR: cualquier task donde tengas secuencias de longitud variable sin alineación supervisada (e.g. matching nombres con typos, alineación de códigos clínicos a guidelines, sequence labelling sobre notas clínicas) puede modelarse con CTC. El truco mental clave: si tu output es una secuencia más corta que tu input y la alineación es monotonic pero desconocida, CTC es la herramienta correcta.
