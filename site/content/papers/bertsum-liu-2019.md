---
title: "BERTSum (Fine-tune BERT for Extractive Summarization)"
weight: 111
math: true
---

{{< paper-card
    title="Fine-tune BERT for Extractive Summarization"
    authors="Yang Liu"
    year="2019"
    venue="arXiv 1903.10318 (EMNLP 2019)"
    pdf="/papers/bertsum-liu-2019.pdf"
    arxiv="1903.10318" >}}
Primera adaptacion exitosa de BERT a resumen extractivo a nivel de documento. La receta es minimal: insertar un token `[CLS]` **antes de cada oracion**, alternar **interval segment embeddings** $E_A/E_B$ segun paridad, usar el vector del `[CLS]` de la oracion $i$ como su representacion, apilar una **summary layer** (la mejor: un Transformer inter-oracional de 2 capas), y clasificar binariamente contra un **oracle greedy** que maximiza ROUGE-2 frente al resumen humano. En inferencia se ranquean oraciones y se aplica **trigram blocking** para evitar redundancia. Sobre CNN/DailyMail establece R-1 = 43.25, R-2 = 20.24, R-L = 39.63 -- superando al SOTA previo (NeuSum, REFRESH, PGN, DCA) por ~1.65 puntos R-L.
{{< /paper-card >}}

---

## El problema

BERT (Devlin 2018) fue disenado para producir representaciones contextuales **a nivel de token** sobre **una o dos oraciones** (sentence A / sentence B). El resumen extractivo, en cambio, necesita representaciones **a nivel de oracion** sobre **m oraciones** del documento (tipicamente $m \in [20, 50]$ en CNN/DM). Aplicar BERT vanilla rompe en tres puntos:

1. **Output a nivel de token, no de oracion**. Concatenar todo el documento como `[CLS] doc [SEP]` produce un unico vector `[CLS]` que comprime el documento entero -- no $m$ vectores por oracion.
2. **Segment embeddings binarios**. BERT solo distingue $E_A$ vs $E_B$. No tiene mecanismo para distinguir oracion 1, 2, 3, ..., m.
3. **Limite de 512 tokens**. Documentos noticiosos pueden exceder 700-1500 tokens, forzando truncamiento.

Liu ataca (1) y (2) con cambios minimos sobre la entrada de BERT y acepta (3) como limitacion conocida.

---

## Idea central -- BERT modificado

### Multi-`[CLS]`

En vez de un solo `[CLS]` al inicio, se inserta uno **antes de cada oracion** y un `[SEP]` despues:

```
[CLS] sent_1 [SEP] [CLS] sent_2 [SEP] [CLS] sent_3 [SEP] ... [CLS] sent_m [SEP]
```

Cada oracion queda flanqueada por un `[CLS]` y un `[SEP]`. El vector de salida del `[CLS]` que precede a la oracion $i$ se reinterpreta como **embedding de la oracion $i$**:

$$T_i = \text{BERT}(\text{document})[\text{posicion del i-esimo [CLS]}] \in \mathbb{R}^{768}$$

Es contextual: depende de las demas oraciones del documento via la atencion global de BERT.

### Interval segment embeddings

Para que el modelo distinga oraciones consecutivas, Liu introduce los **interval segment embeddings**: a la oracion $i$ se le asigna

$$\text{segment}(\text{sent}_i) = \begin{cases} E_A & \text{si } i \text{ es impar} \\ E_B & \text{si } i \text{ es par} \end{cases}$$

Un documento $[s_1, s_2, s_3, s_4, s_5]$ recibe embeddings $[E_A, E_B, E_A, E_B, E_A]$. La alternancia $E_A/E_B$ no identifica univocamente cada oracion -- es solo paridad -- pero permite reusar la arquitectura BERT existente **sin agregar parametros** y conservar el beneficio del pretraining.

### Input total

Igual que BERT vanilla, el input es la suma de tres embeddings:

$$\text{input}_j = \text{TokenEmb}(t_j) + \text{IntervalSegEmb}(t_j) + \text{PosEmb}(j)$$

Position embeddings llegan hasta 512. Documentos mas largos se truncan.

---

## Arquitectura -- Summary Layer

Sobre los embeddings de oracion $\{T_1, ..., T_m\}$ se apila una capa que produce un score $\hat{Y}_i \in [0, 1]$ por oracion. Liu evalua tres variantes:

### Simple Classifier (lineal + sigmoid)

$$\hat{Y}_i = \sigma(W_o T_i + b_o)$$

Cada $T_i$ se proyecta independientemente. CNN/DM: R-1 = 43.23, R-2 = 20.22, R-L = 39.60.

### Inter-sentence Transformer (la ganadora)

Un Transformer encoder estandar de $L = 2$ capas operando **sobre la secuencia de embeddings de oracion**:

$$
\begin{aligned}
h^0 &= \text{PosEmb}(T) \\
\tilde{h}^{\,l} &= \text{LN}(h^{l-1} + \text{MHAtt}(h^{l-1})) \\
h^l &= \text{LN}(\tilde{h}^{\,l} + \text{FFN}(\tilde{h}^{\,l})) \\
\hat{Y}_i &= \sigma(W_o h^L_i + b_o)
\end{aligned}
$$

con $T \in \mathbb{R}^{m \times 768}$ y un **nuevo** conjunto de position embeddings que codifican posicion **de la oracion** dentro del documento (no del token). Liu prueba $L \in \{1, 2, 3\}$ y reporta $L = 2$ como optimo. CNN/DM: R-1 = 43.25, R-2 = 20.24, R-L = 39.63.

El cosa importante: este Transformer opera sobre ~30 elementos (oraciones), no sobre los 500+ tokens del documento. El computo es barato.

### LSTM

LSTM unidireccional con per-gate layer normalization. CNN/DM: R-1 = 43.22, R-2 = 20.17, R-L = 39.59.

### Sintesis

| Capa | R-1 | R-2 | R-L |
|------|-----|-----|-----|
| Simple Classifier | 43.23 | 20.22 | 39.60 |
| Inter-sentence Transformer ($L=2$) | **43.25** | **20.24** | **39.63** |
| LSTM | 43.22 | 20.17 | 39.59 |

Las tres variantes difieren en <0.1 ROUGE. **El pretraining de BERT hace el trabajo pesado**; la summary layer es casi cosmetica, pero el Transformer queda como referencia por estabilidad y elegancia.

---

## Oracle target

CNN/DailyMail y NYT vienen con resumenes humanos **abstractivos** -- texto nuevo, parafraseado, no subconjuntos de oraciones del documento. Pero el modelo necesita labels $y_i \in \{0, 1\}$ por oracion. La solucion estandar (heredada de Nallapati 2017) es construir un **oracle** mediante busqueda greedy que maximiza ROUGE-2 contra el resumen humano:

```
oracle = []
best_rouge = 0
loop:
    para cada oracion s no en oracle:
        r = ROUGE-2(oracle + [s], gold_summary)
    si la mejor candidata mejora best_rouge:
        agregar a oracle, actualizar best_rouge
    si no: break
```

Las oraciones seleccionadas reciben $y_i = 1$; el resto $y_i = 0$.

El oracle es un **techo** para cualquier extractivo: sobre CNN/DM alcanza R-1 = 52.59, R-2 = 31.24, R-L = 48.87. BERTSum llega a R-L = 39.63, dejando ~9 puntos de gap explotables. No es un techo absoluto de la tarea -- un abstractivo podria superarlo via parafraseo -- pero acota la familia extractiva.

El oracle greedy es ruidoso: puede asignar $y_i = 1$ a oraciones que coinciden lexicamente con el resumen sin ser semanticamente las mas importantes. MatchSum (Zhong 2020) intenta corregir esto reformulando la tarea a summary-level matching.

---

## Training

**Loss**: binary cross-entropy promediada sobre oraciones,

$$\mathcal{L} = -\sum_{i=1}^{m} \left[ y_i \log \hat{Y}_i + (1 - y_i) \log(1 - \hat{Y}_i) \right]$$

donde $y_i$ proviene del oracle. BERT + summary layer se **fine-tunean juntos**.

**Optimizador**: Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$) con schedule de Vaswani con warmup:

$$\text{lr} = 2 \times 10^{-3} \cdot \min(\text{step}^{-0.5}, \; \text{step} \cdot \text{warmup}^{-1.5})$$

10000 pasos de warmup. 50000 steps totales sobre 3 GPUs GTX 1080 Ti, gradient accumulation cada 2 steps, batch efectivo ~36.

**Seleccion de checkpoint**: cada 1000 steps se evalua loss en val. Se promedian los **top-3 checkpoints por loss** (model averaging) para el reporte final.

---

## Inference -- Trigram Blocking

Una vez entrenado, BERTSum produce scores $\hat{Y}_i$ por oracion. La seleccion final es:

1. **Rankear** oraciones por $\hat{Y}_i$ descendente.
2. **Trigram blocking**: iterar la lista; agregar una oracion al resumen $S$ **si y solo si no comparte ningun trigrama con $S$**.
3. **Length cap**: detenerse cuando $|S| = 3$ (ajustable por dataset).

```
S = []
para candidate en oraciones rankeadas:
    si len(S) >= 3: break
    si trigramas(candidate) ∩ trigramas(S) == ∅:
        S.append(candidate)
return S
```

Es una version simple de **Maximal Marginal Relevance** (Carbonell & Goldstein 1998): seleccionar la siguiente oracion solo si aporta informacion no cubierta.

La ablacion muestra que **trigram blocking aporta mas que cambiar la summary layer**:

| Variante | $\Delta$R-1 | $\Delta$R-2 | $\Delta$R-L |
|---|---|---|---|
| sin interval segments | -0.02 | -0.05 | -0.03 |
| sin trigram blocking | **-0.66** | **-0.26** | **-0.56** |

Es el truco con mayor return-on-effort del paper.

---

## Resultados

### CNN/DailyMail

| Model | R-1 | R-2 | R-L |
|---|---|---|---|
| LEAD-3 | 40.42 | 17.62 | 36.67 |
| PGN (See 2017) | 39.53 | 17.28 | 37.98 |
| DCA (Celikyilmaz 2018) | 41.69 | 19.47 | 37.92 |
| REFRESH (Narayan 2018) | 41.0 | 18.8 | 37.7 |
| NeuSum (Zhou 2018) | 41.59 | 19.01 | 37.98 |
| Transformer from-scratch (no BERT) | 40.90 | 18.02 | 37.17 |
| **BERTSum + Classifier** | 43.23 | 20.22 | 39.60 |
| **BERTSum + Transformer** | **43.25** | **20.24** | **39.63** |
| **BERTSum + LSTM** | 43.22 | 20.17 | 39.59 |
| ORACLE (greedy) | 52.59 | 31.24 | 48.87 |

Observaciones:

- **El gap BERT vs no-BERT lo explica todo**. Un Transformer entrenado from-scratch sobre summarization (40.90 / 18.02 / 37.17) queda por debajo del LEAD-3 trivial en R-1 y R-L. **El pretraining es lo que mueve la aguja**.
- **+1.65 R-L** sobre el SOTA previo. En la era pre-GPT, esto era una mejora significativa.
- **Equivalencia entre summary layers**: el techo no esta en la cabeza, sino en cuanto BERT puede extraer del documento.

### NYT50

Evaluacion limited-length recall (predicciones truncadas a longitud del gold):

| Model | R-1 | R-2 | R-L |
|---|---|---|---|
| First-$k$ words | 39.58 | 20.11 | 35.78 |
| Durrett 2016 | 42.2 | 24.9 | -- |
| Deep Reinforced (Paulus 2018) | 42.94 | 26.02 | -- |
| **BERTSum + Classifier** | **46.66** | **26.35** | **42.62** |

BERTSum sostiene el lead con +3.7 R-1 sobre Durrett.

### XSum

En el paper extendido (Liu & Lapata, EMNLP 2019, arXiv 1908.08345) se evalua XSum (resumenes de una sola oracion, "extreme summarization"). El paradigma extractivo pierde por construccion: las oraciones del documento rara vez sintetizan en una sola frase la idea central. BERTSumAbs (la variante abstractiva) supera ampliamente a BERTSumExt en XSum.

---

## BERTSumAbs -- la extension abstractiva

La version extendida con Mirella Lapata introduce **BERTSumAbs**:

- **Encoder**: BERTSum (multi-`[CLS]` + interval segments + inter-sentence Transformer), reusado.
- **Decoder**: Transformer de 6 capas entrenado **from scratch**.

La innovacion clave es el **two-stage fine-tuning con dos optimizadores Adam separados**, cada uno con su propio schedule:

$$\text{lr}_E = 2 \times 10^{-3} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot 20000^{-1.5})$$

$$\text{lr}_D = 0.1 \cdot \min(\text{step}^{-0.5}, \text{step} \cdot 10000^{-1.5})$$

El encoder corre con lrs efectivos ~$2\text{e-}5$ (tipico de fine-tuning BERT, evita desestabilizar el pretraining); el decoder con lrs ~$1\text{e-}3$ (tipico de Transformer aleatorio, necesita amplitud para aprender). Sin esta separacion el encoder se rompe o el decoder no converge.

CNN/DM:

| Variante | R-1 | R-2 | R-L |
|---|---|---|---|
| BERTSumExt | 43.25 | 20.24 | 39.63 |
| BERTSumAbs | 41.72 | 19.39 | 38.76 |
| BERTSumExtAbs (pre-train Ext, luego Abs) | 42.13 | 19.60 | 39.18 |

El extractivo gana en R-2 porque la rigidez del oracle preserva n-gramas literales del documento (citas, frases hechas). En XSum la historia se invierte.

---

## Limitaciones

- **Limite de 512 tokens**: documentos largos se truncan. Sucesores como **Longformer** (Beltagy 2020) y **BigBird** (Zaheer 2020) introducen atencion sparse para 4096+ tokens.
- **Oracle aproximado**: ground truth ruidoso por la naturaleza greedy y la dependencia exclusiva de ROUGE como funcion objetivo. False positives lexicos y false negatives semanticos.
- **Rigidez extractiva**: BERTSum no puede parafrasear, resolver anaforas en el output, ni sintetizar varias oraciones en una. Resumenes coherentes localmente pero entrecortados en el flujo.
- **Lead-bias**: CNN/DailyMail tiene fuerte sesgo posicional (las primeras oraciones suelen ser el "lead" del periodismo inverted-pyramid). El modelo lo aprende; puede no generalizar a dominios sin lead-bias.
- **ROUGE como proxy**: maximizar ROUGE-2 sobre un solo resumen humano sobreajusta a expresiones del anotador. Metricas semanticas (BERTScore, MoverScore) o de factualidad (NLI-based) no se evaluaron en el original.

---

## Sucesores

- **MatchSum** (Zhong 2020): reformula la seleccion como **summary-level matching** -- genera candidatos de resumen (subsets) y entrena un modelo que matchea documento vs candidato via cosine en embeddings BERT. R-L = 40.55 en CNN/DM (+0.92 sobre BERTSum).
- **HiBERT** (Zhang 2019): hierarchical BERT explicito -- primero codifica oraciones, luego documento.
- **PEGASUS** (Zhang 2020): cambia el **pretraining objective** a Gap Sentence Generation (predecir oraciones enteras enmascaradas) -- practicamente "pretrain para summarization". R-2 = 21.47 en CNN/DM.
- **BART** (Lewis 2020) y **T5** (Raffel 2020): encoder-decoders preentrenados con denoising. Fine-tune sobre CNN/DM da R-L = 40.90 (BART) y R-L = 39.75 (T5-large). Ver [paper-card de T5](/papers/t5-raffel-2020).
- **LLMs zero-shot** (era 2022+): GPT-3, ChatGPT, Claude, Llama, Mistral. ROUGE no siempre supera a BERTSum, pero las evaluaciones humanas tienden a preferirlos por fluidez. La summarization extractiva clasica sigue vigente cuando la fidelidad estricta, latencia/costo, o interpretabilidad importan mas que la fluidez (legal, medico, cientifico, RAG retrievers).

---

## Conexion con la clase 22

La clase 22 del curso (Procesamiento de Lenguaje Natural: Generacion) dedica una seccion entera al resumen automatico. La estructura del PDF:

- **Slides 1-16**: introduccion, paradigmas extractivo vs abstractivo, **Pointer-Generator Network** (See 2017) -- ver [paper-card de PGN](/papers/pointer-generator-see-2017).
- **Slides 17-32**: **Extractive Model** -- implementa directamente BERTSum: encoder BERT con multi-`[CLS]` (slides 21-24), interval segment embeddings (25-28), inter-sentence Transformer como summary layer (29-30), loss BCE contra oracle (31), inferencia con trigram blocking (32).
- **Slides 33-56**: ROUGE, evaluacion, oracle como techo, LEAD-3.
- **Slide 57**: appendix con referencias -- cita explicita a Liu (2019) "Fine-tune BERT for Extractive Summarization" como **fuente del extractive model**.

El paper de Liu es la **referencia primaria** del modulo de resumen extractivo. Las decisiones de diseno que el curso ensena -- modificar la entrada de BERT en vez de la arquitectura, construir labels via oracle greedy, separar scoring de selection con trigram blocking, aceptar limitaciones (512 tokens, rigidez extractiva, ROUGE como proxy) -- vienen directamente de este trabajo.

Conexiones con otras clases del modulo:

- **Clase 19 (Transformers)**: el inter-sentence Transformer es un Vanilla Transformer encoder de 2 capas operando sobre embeddings de oracion en vez de tokens.
- **Clase 20 (BERT y modelos preentrenados)**: BERTSum es la primera aplicacion canonica de fine-tuning de BERT a una tarea estructural (no clasificacion simple); introduce el patron de **modificar la entrada de BERT** para tareas no contempladas en el pretraining.
- **Clase 21 (Generacion abstractiva)**: BERTSum es el contrapunto extractivo a Pointer-Generator Network.

---

## Notas y enlaces

- El paper original (Liu solo, 2019) tiene 5 paginas; la version extendida con Lapata (EMNLP 2019, arXiv 1908.08345) anade BERTSumAbs y experimentos en XSum.
- Codigo oficial: [nlpyang/PreSumm](https://github.com/nlpyang/PreSumm) (PyTorch).
- Implementaciones modernas con HuggingFace: ver `BertModel` mas una cabeza de clasificacion custom; el patron multi-`[CLS]` se implementa al construir el input.

Ver fundamentos: [Resumen Automatico](/fundamentos/text-summarization) - [Pre-training y BERT](/fundamentos/pretraining-bert) - [BERT](/fundamentos/bert) - [Metrica ROUGE](/fundamentos/rouge-metric).

Ver papers: [BERT (Devlin 2018)](/papers/bert-devlin-2018) - [Pointer-Generator (See 2017)](/papers/pointer-generator-see-2017) - [T5 (Raffel 2020)](/papers/t5-raffel-2020) - [ROUGE (Lin 2004)](/papers/rouge-lin-2004).

Ver clase: [Clase 22 -- Generacion: Resumen Automatico](/clases/clase-22).
