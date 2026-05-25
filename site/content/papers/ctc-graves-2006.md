---
title: "CTC (Connectionist Temporal Classification)"
weight: 107
math: true
---

{{< paper-card
    title="Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"
    authors="Graves, Fernández, Gomez, Schmidhuber"
    year="2006"
    venue="ICML 2006"
    pdf="/papers/ctc-graves-2006.pdf"
    arxiv="" >}}
Paper fundacional que entrena RNNs sobre secuencias sin alineamiento previo entre frames de entrada y labels de salida. Introduce un símbolo "blank" en el alfabeto, define una distribución sobre paths frame-a-frame y marginaliza sobre todos los alineamientos con un algoritmo forward-backward al estilo HMM Baum-Welch. Es la base de DeepSpeech, CRNN, Wav2Letter y los heads de fine-tuning de Wav2Vec 2.0.
{{< /paper-card >}}

---

## El problema

Hacia 2006 las RNN ya existían (Elman 1990, Werbos BPTT 1990) y LSTM había resuelto el vanishing gradient (Hochreiter & Schmidhuber 1997, Schuster & Paliwal BRNN 1997). El estado del arte previo en sequence labelling (ASR, OCR, handwriting) eran los **HMM context-dependent** con emisiones GMM y los **híbridos HMM-ANN/HMM-RNN** (Bourlard & Morgan 1994, Robinson 1994).

Las limitaciones eran sistémicas:

- Los híbridos requerían **forced alignment** previo (Viterbi forced) sobre el corpus para generar targets frame-level. La calidad del alignment limitaba el techo del sistema.
- Los HMM imponían una topología discreta, lexicon explícito y observation independence cuestionable.
- El entrenamiento era generativo (maximizan $p(\mathbf{x}, \mathbf{l})$) cuando la tarea es discriminativa.
- Las RNN entrenadas con cross-entropy frame-a-frame producían softmaxes que había que post-procesar heurísticamente.

**La brecha**: no había forma de entrenar una RNN directamente para producir una transcripción cuya alineación con el input es desconocida. Tareas afectadas: ASR (frames acústicos → fonemas/caracteres), OCR/STR (columnas de imagen → caracteres), handwriting (trayectorias → caracteres), todas con $|\mathbf{l}| \le T$ y sin supervisión de alignment.

El paper se evalúa sobre **TIMIT phoneme recognition**: 4620 utterances de train, 1680 test, 61 fonemas, features MFCC + log-energy + derivadas (26 coeficientes por frame). El baseline son HMM context-dependent (>900k parámetros) y un híbrido BLSTM-HMM con la misma arquitectura recurrente que CTC. La métrica es **Label Error Rate (LER)**:

$$
\text{LER}(h, \mathcal{S}') = \frac{1}{Z} \sum_{(\mathbf{x}, \mathbf{l}) \in \mathcal{S}'} \text{ED}\bigl(h(\mathbf{x}), \mathbf{l}\bigr)
$$

donde ED es edit distance (Levenshtein). Esta métrica es el progenitor directo de WER (word error rate) y CER (character error rate) usados hoy en ASR/OCR. Resultados clave: CTC con prefix search alcanza **30.51% LER**, superando al híbrido BLSTM-HMM (33.84%) con la misma arquitectura RNN y al CD-HMM (35.21%) con 8× más parámetros. La conclusión decisiva: no es la arquitectura, es la objective.

---

## Solución CTC

CTC introduce tres piezas que en conjunto permiten entrenamiento end-to-end.

### Augmented alphabet

Se extiende el alfabeto target con un símbolo **blank**:

$$
\mathcal{L}' = \mathcal{L} \cup \{\epsilon\}
$$

El blank no significa "silencio acústico" sino "ningún label emitido en este frame". La red termina en una softmax de tamaño $|\mathcal{L}'| = |\mathcal{L}| + 1$. Para una entrada $\mathbf{x}$ de largo $T$, la red produce $\mathbf{y} \in (\mathbb{R}^{|\mathcal{L}'|})^T$ donde $y_k^t = p(\text{símbolo } k \text{ en frame } t \mid \mathbf{x})$.

### Path

Un **path** es una secuencia frame-a-frame sobre el alfabeto extendido:

$$
\pi \in \mathcal{L}'^T
$$

### Mapping $\mathcal{B}$

El operador $\mathcal{B}: \mathcal{L}'^T \to \mathcal{L}^{\le T}$ colapsa cada path a un labelling en dos pasos: (1) colapsar repeticiones consecutivas del mismo símbolo, (2) eliminar blanks.

Ejemplos canónicos:

- $\mathcal{B}(\text{-aa-bb-cc-}) = abc$
- $\mathcal{B}(\text{a-ab-}) = aab$
- $\mathcal{B}(\text{cc-at}) = cat$
- $\mathcal{B}(\text{cat}) = cat$

El blank es estructuralmente necesario para distinguir dobles letras reales: $\mathcal{B}(aa) = a$, mientras que $\mathcal{B}(a\epsilon a) = aa$. Es el separador que permite emitir el mismo símbolo dos veces seguidas.

---

## Likelihood y loss

Asumiendo **conditional independence** entre frames dado el estado interno de la red (la red puede tener feedback interno, pero el output layer no se realimenta), la probabilidad de un path se factoriza:

$$
p(\pi \mid \mathbf{x}) = \prod_{t=1}^{T} y_{\pi_t}^t
$$

La probabilidad de un labelling $\mathbf{l} \in \mathcal{L}^{\le T}$ es la suma sobre todas sus pre-imágenes bajo $\mathcal{B}$:

$$
p(\mathbf{l} \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} p(\pi \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} y_{\pi_t}^t
$$

La pérdida CTC es la negative log-likelihood sobre el conjunto de entrenamiento:

$$
\mathcal{L}_{\text{CTC}} = -\sum_{(\mathbf{x}, \mathbf{l}) \in \mathcal{S}} \ln p(\mathbf{l} \mid \mathbf{x})
$$

El número de paths en $\mathcal{B}^{-1}(\mathbf{l})$ crece exponencialmente con $T$, así que la suma directa es intratable. Aquí entra el forward-backward.

---

## Forward-backward

### Extended sequence

Para tracking eficiente se construye $\mathbf{l}'$ intercalando blanks:

$$
\mathbf{l}' = (\epsilon, l_1, \epsilon, l_2, \epsilon, \ldots, \epsilon, l_U, \epsilon), \qquad |\mathbf{l}'| = 2U + 1
$$

### Recurrencia forward

La variable forward $\alpha_t(s)$ es la masa total de paths que en frame $t$ están en la posición $s$ del extended sequence. Inicialización en $t=1$: solo se permite empezar en blank inicial ($s=1$) o en el primer label ($s=2$):

$$
\alpha_1(1) = y_\epsilon^1, \quad \alpha_1(2) = y_{l_1}^1, \quad \alpha_1(s) = 0 \ \forall s > 2
$$

Recursión con $\bar\alpha_t(s) = \alpha_{t-1}(s) + \alpha_{t-1}(s-1)$:

$$
\alpha_t(s) =
\begin{cases}
\bar\alpha_t(s)\,y_{l'_s}^t & \text{si } l'_s = \epsilon \text{ o } l'_{s-2} = l'_s \\[2pt]
\bigl(\bar\alpha_t(s) + \alpha_{t-1}(s-2)\bigr)\,y_{l'_s}^t & \text{en otro caso}
\end{cases}
$$

El skip de 2 está prohibido cuando $l'_{s-2} = l'_s$ porque eso colapsaría dos labels iguales (rompe el rol estructural del blank obligatorio entre repeticiones). La probabilidad total se cierra sumando los dos estados terminales:

$$
p(\mathbf{l} \mid \mathbf{x}) = \alpha_T(|\mathbf{l}'|) + \alpha_T(|\mathbf{l}'|-1)
$$

La recurrencia backward $\beta_t(s)$ es dual. El gradiente respecto a los logits pre-softmax tiene la forma compacta:

$$
\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial u_k^t} = y_k^t - \frac{1}{y_k^t \cdot p(\mathbf{l}\mid\mathbf{x})} \sum_{s \in \text{lab}(\mathbf{l}, k)} \alpha_t(s)\,\beta_t(s)
$$

Interpretación: el target en frame $t$ para el símbolo $k$ es la posterior marginal de emitir $k$ en $t$ dado el labelling completo, computada por el forward-backward.

### Complejidad

Cada paso tiene $O(|\mathbf{l}'|) = O(U)$ operaciones por $T$ frames: total **$O(T \cdot U)$** por sample. Idéntica complejidad asintótica al forward-backward HMM con $2U+1$ estados, lo que en 2006 fue argumento decisivo: CTC no es más caro que el baseline. En implementación real se trabaja en log-space con `logsumexp` para evitar underflow numérico (los productos de $T \sim 100$ probabilidades colapsan a cero en float32).

---

## Decoding

Dado un modelo entrenado, queremos $h(\mathbf{x}) = \arg\max_{\mathbf{l}} p(\mathbf{l} \mid \mathbf{x})$. No se conoce un algoritmo general tractable (se demostraría más tarde que es NP-hard cuando se combina con LM externo arbitrario). Se usan aproximaciones:

| Estrategia | Idea | Pros | Contras |
| --- | --- | --- | --- |
| Best path (greedy) | $\pi^* = \arg\max_\pi p(\pi\mid\mathbf{x})$, luego $\mathcal{B}(\pi^*)$ | $O(T\cdot\|\mathcal{L}'\|)$, trivial | No óptimo: ignora la marginalización |
| Prefix search | Expandir árbol de prefijos con forward-backward modificado | Óptimo dado tiempo suficiente | Exponencial en peor caso |
| Beam search + LM | $\arg\max_\mathbf{l}\bigl[\ln p_{\text{CTC}} + \alpha\ln p_{\text{LM}} + \beta\|\mathbf{l}\|\bigr]$ | Calidad alta, estándar en ASR moderno | Requiere LM y tunear $\alpha,\beta$ |
| Lexicon-constrained | Beam search restringido a vocabulario | Cero out-of-vocab errors | Limitado a OOV |

Hannun et al. (2014) publicaría la receta canónica de beam search con shallow LM fusion que se vuelve estándar en DeepSpeech.

---

## Propiedades importantes

**Conditional independence**: dos timesteps consecutivos no están correlacionados condicionalmente al input. En la práctica esto es una simplificación deliberada (las representaciones de una BLSTM sí capturan contexto), pero implica que CTC **no modela dependencias label-level** directamente. Incorporarlas exige LM externo o cambiar de modelo (RNN-Transducer).

**Peaky outputs**: tras entrenar, la distribución frame-level se vuelve dominada por blanks con spikes puntuales en los frames donde el modelo está confiado. Esto hace que best-path decoding sea típicamente suficiente, pero implica que **los spikes no son timestamps acústicos confiables** sin post-procesamiento (Sak 2015, Senior 2015).

**Length constraint**: CTC asume $|\mathbf{l}| \le T$, y con blanks obligatorios entre repeticiones la constraint efectiva es $|\mathbf{l}'| = 2|\mathbf{l}|+1 \le T$. Encoders con stride agresivo (típico downsampling $\times 4$ o $\times 8$) pueden violarla en targets largos.

**Monotonic alignment**: el path debe avanzar o quedarse en $\mathbf{l}'$, nunca retroceder. Esto encaja con ASR/OCR/HTR (el orden de fonemas/caracteres es el orden temporal/espacial), pero no con tareas no monotónicas como traducción.

**Blank dominance**: al inicio del entrenamiento la red colapsa a predecir solo blanks (atractor trivial). Solo cuando los gradientes de los símbolos no-blank dominan emergen los spikes. Requiere warmup, curriculum o focal CTC para acelerar la convergencia.

**Implicit label dependency vía estado interno**: pese a la conditional independence en el output layer, la RNN bidireccional acumula contexto pasado y futuro en su estado oculto, y CTC modela implícitamente dependencias inter-label vía ese estado. El paper observa que clusters como `dcl + d` (closure + stop en inglés) aparecen como *double spikes* porque la red aprendió esa co-ocurrencia frecuente. Es decir: las dependencias label-level existen, pero están embebidas en las representaciones, no expuestas en la lattice de transiciones.

---

## Aplicaciones masivas

- **DeepSpeech** (Hannun et al. 2014, Baidu/Mozilla): primer ASR end-to-end a escala. Arquitectura simple (5 FC + BiRNN), CTC loss, beam search con LM. Demostró que con datos y compute, CTC + RNN supera HMM con décadas de ingeniería. Mozilla DeepSpeech y Facebook **Wav2Letter** son descendientes directos.
- **CRNN** (Shi, Bai, Yao 2017): el ejemplar canónico de CTC en visión. CNN extrae features de la imagen del texto, las columnas del feature map pasan a BiLSTM y un CTC head decodifica caracteres. Es la baseline obligatoria de cualquier paper de scene text recognition. Cubierto directamente en clase 21. Ver [CRNN paper](/papers/crnn-shi-2017).
- **Wav2Vec 2.0** (Baevski et al. 2020) y **HuBERT** (Hsu et al. 2021): foundation models de speech pre-entrenados self-supervised y fine-tuneados con CTC head sobre transcripciones. CTC sigue siendo el output layer de fine-tuning más simple y competitivo.
- **Whisper** (Radford et al. 2022): se aleja a seq2seq decoder para offline ASR, pero los sistemas production-friendly (mobile, edge, streaming) siguen con CTC o RNN-T.
- **Handwriting OCR** offline e online (IAM benchmark): el propio Graves extendió CTC a multidimensional RNN para imágenes 2D, dominando ICDAR HTR durante años.
- **LipNet** (Assael 2016) para lip-reading, music transcription, sign language, action segmentation, EEG/ECG labelling.

---

## Variantes y sucesores

| Modelo | Año | Cambio respecto a CTC |
| --- | --- | --- |
| Multidimensional CTC | 2009 | Extiende a inputs 2D para handwriting offline |
| **RNN-Transducer (RNN-T)** | 2012 | Agrega prediction network autoregresiva sobre labels: elimina conditional independence. Forward-backward sobre lattice 2D. Estándar production ASR en Google/Amazon/Microsoft. |
| Attention seq2seq (LAS) | 2016 | Abandona blank, usa attention soft sobre frames. Más flexible (no monotonic) pero no streaming. |
| CTC/Attention hybrid | 2017 | ESPnet: $\mathcal{L} = \lambda \mathcal{L}_{\text{CTC}} + (1-\lambda)\mathcal{L}_{\text{att}}$ con encoder compartido. CTC regulariza monotonicidad. |
| PARSeq | 2022 | Permutation language modeling para STR; combina ideas de attention y permutaciones tipo XLNet. |

Observación clave: RNN-T es preferido en mobile/edge por trade-off latencia/calidad, pero **suele inicializarse con CTC warmup** porque entrenar RNN-T desde scratch es notoriamente difícil. CTC sigue siendo el "primer paso" del pipeline incluso cuando no es la objective final.

---

## Implementación

PyTorch ofrece `torch.nn.CTCLoss` con kernel CUDA optimizado (WarpCTC, Baidu 2016, integrado nativamente). Puntos críticos:

```python
import torch
import torch.nn as nn

ctc = nn.CTCLoss(blank=0, zero_infinity=True)

# log_probs: (T, N, C) en log-space (log_softmax aplicado)
# targets:  (N, S) concatenados sin padding o (sum(target_lengths),)
# input_lengths: (N,) frames válidos por sample
# target_lengths: (N,) labels válidos por sample
loss = ctc(log_probs, targets, input_lengths, target_lengths)
```

Detalles que importan:

- **Blank index**: PyTorch usa `blank=0` por defecto. Confundir convenciones (blank al final vs al principio) produce gradientes incorrectos y entrenamiento que no converge.
- **Log-space input**: pasar `log_softmax`, no softmax ni logits crudos. Errror frecuente.
- **`zero_infinity=True`**: cuando $T < 2U+1$ la loss devuelve `inf`. Esta opción la pone a 0 (y enmascara el gradiente) en lugar de propagar `nan` al batch entero.
- **Padding**: nunca incluir tokens de padding en `targets`. Usar `target_lengths` para definir la región válida.
- **Stride del encoder**: si la CNN reduce el ancho a menos de $2U+1$ "columnas" efectivas, CTC no puede entrenar. Calibrar downsampling cuidadosamente.
- **Reproducibilidad**: el kernel CUDA es no determinístico. Para experimentos estrictos, `torch.use_deterministic_algorithms(True)` con costo de velocidad.

Para una explicación visual ejemplar, ver [Hannun, "Sequence Modeling with CTC" en Distill.pub (2017)](https://distill.pub/2017/ctc/).

---

## Conexión con clase 21

La clase 21 cubre **Scene Text Recognition (STR)**. La slide *Text Recognition Stages* del PDF descompone el pipeline en cuatro etapas (Transformation, Feature Extraction, Sequence Modeling, Prediction). La etapa **Prediction** se divide canónicamente en dos opciones:

1. **CTC**: la opción de CRNN (Shi 2017). Asume alineación monotonic implícita. Más rápida, simple, streaming-friendly, gradient bien definido.
2. **Attention**: la opción de ASTER, MORAN, [ABCNet](/papers/abcnet-liu-2020) (Liu 2020). Más flexible para texto irregular (curvo, perspectivado, rotado).

El contraste pedagógico CTC vs attention es central en STR:

| | CTC | Attention |
| --- | --- | --- |
| Alineación | Monotonic implícita | Soft, no monotonic |
| Velocidad inferencia | Alta (one-shot frame-level) | Más lenta (autoregresiva) |
| Streaming | Nativo | Necesita input completo |
| Texto curvado/rotado | Limitado | Maneja bien |
| Beam search | Ligero | Más caro |
| Condicionamiento label-side | Sin (necesita LM externo) | Autoregresivo full |

Entender CTC permite entender por qué attention se vuelve preferible en STR irregular y por qué los híbridos CTC+attention dominan benchmarks recientes. ABCNet (Liu 2020) representa la rama attention con manejo explícito de texto curvo via Bezier alignment, y se compara directamente con CRNN/CTC en la clase.

---

## Notas y enlaces

**Fundamentos transversales**:

- [CTC loss](/fundamentos/ctc-loss): forward-backward con ejemplos numéricos paso a paso.
- [Scene Text Recognition](/fundamentos/scene-text-recognition): el pipeline de 4 etapas.
- [LSTM y GRU](/fundamentos/lstm-gru): la arquitectura recurrente que CTC asume en el encoder.
- [Redes recurrentes](/fundamentos/redes-recurrentes): contexto general de RNN, BPTT, BRNN.

**Papers relacionados**:

- [CRNN (Shi, Bai, Yao 2017)](/papers/crnn-shi-2017): el ejemplar canónico de CTC en visión, baseline obligatoria de STR.
- [ABCNet (Liu et al. 2020)](/papers/abcnet-liu-2020): comparación pedagógica con la rama attention de scene text.
- [LSTM (Hochreiter & Schmidhuber 1997)](/papers/lstm-hochreiter-1997): la arquitectura recurrente sin la cual CTC no funciona en la práctica.
- [Text Recognition in the Wild (Chen et al. 2020)](/papers/text-recognition-wild-chen-2020): survey que sitúa CTC frente a alternativas modernas en STR.

**Clase**: [Clase 21 — Scene Text Recognition](/clases/clase-21).

**Timeline de dominio**:

| Año | Hito |
| --- | --- |
| 1989 | HMM tutorial (Rabiner) — framework dominante de sequence labelling |
| 1994 | HMM-RNN hybrid (Bourlard, Robinson) — mejor que GMM pero requiere forced alignment |
| 1997 | LSTM (Hochreiter & Schmidhuber) + BRNN (Schuster & Paliwal) |
| 2005 | BLSTM framewise (Graves & Schmidhuber) — SOTA en classification per-frame |
| **2006** | **CTC (Graves et al.)** — end-to-end RNN sequence labelling sin alignment |
| 2012 | RNN-Transducer (Graves) — quita conditional independence |
| 2014 | DeepSpeech (Hannun et al.) — ASR end-to-end a escala con CTC |
| 2017 | CRNN (Shi et al.) + CTC/Attention hybrid (Watanabe, ESPnet) |
| 2020 | Wav2Vec 2.0 + CTC fine-tuning — foundation model + CTC head |
| 2022 | Whisper (Radford et al.) — seq2seq decoder gana offline, CTC sigue en streaming |

---

CTC resuelve un problema fundacional —entrenar redes recurrentes para etiquetar secuencias sin alineamiento previo— con un mecanismo matemáticamente elegante: ampliar el alfabeto con un blank, definir una distribución sobre paths, marginalizar con forward-backward. Veinte años después sigue presente en producción: en el head de fine-tuning de Wav2Vec 2.0, en streaming ASR, en CRNN para OCR, en sistemas low-latency donde el costo de un decoder autoregresivo no se puede pagar. Su influencia teórica es aún mayor: estableció el patrón **"marginalizar sobre alineamientos con dynamic programming"** que reaparece en RNN-T, en CTC-segmental, en pointer networks, y conceptualmente en los objectives masked-prediction de los foundation models.
