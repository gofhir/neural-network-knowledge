---
title: "Pointer-Generator (Summarization)"
weight: 220
math: true
---

{{< paper-card
    title="Get To The Point: Summarization with Pointer-Generator Networks"
    authors="See, Liu, Manning"
    year="2017"
    venue="ACL 2017"
    pdf="/papers/pointer-generator-see-2017.pdf"
    arxiv="1704.04368" >}}
Resuelve tres problemas cronicos de los modelos seq2seq con atencion aplicados a **summarization abstractiva**: errores factuales, palabras fuera de vocabulario (OOV) y **repeticion** de frases. La solucion combina dos ideas -- una **pointer-generator network** que permite **copiar** palabras directamente del texto fuente (resolviendo OOV y factualidad), y un **coverage mechanism** que trackea atencion acumulada para evitar re-atender las mismas posiciones (eliminando repeticion). Mejora state-of-art abstractivo en CNN/Daily Mail por ~2 ROUGE points.
{{< /paper-card >}}

---

## Contexto

En 2015-2016, los modelos **abstractive summarization** basados en seq2seq + attention (Nallapati 2016, Rush 2015) mostraban promesa pero tenian tres problemas que limitaban su utilidad practica:

1. **Errores factuales**: reproducian mal detalles especificos (fechas, numeros, nombres).
2. **OOV words**: palabras raras del articulo (nombres propios, cifras) no estaban en el vocabulario del decoder → se convertian en `[UNK]`.
3. **Repetition**: generaban la misma frase o expresion varias veces a lo largo del resumen.

Este paper (Stanford + Google Brain) propuso un modelo hibrido que mitiga los tres simultaneamente.

---

## Ideas principales

### 1. Baseline: Seq2Seq con atencion

Modelo de referencia similar a Nallapati 2016:
- **Encoder**: BiLSTM sobre tokens del articulo → estados $h_i$.
- **Decoder**: LSTM unidireccional con atencion Bahdanau:

$$e_i^t = v^T \tanh(W_h h_i + W_s s_t + b_{\text{attn}})$$

$$a^t = \text{softmax}(e^t)$$

$$h_t^* = \sum_i a_i^t h_i \quad \text{(context vector)}$$

Distribucion sobre vocabulario:

$$P_{\text{vocab}}(w) = \text{softmax}(V'(V[s_t, h_t^*] + b) + b')$$

Loss: NLL del token correcto en cada paso.

### 2. Pointer-Generator Network

**Idea**: el modelo puede **copiar** una palabra del articulo fuente o **generar** una del vocabulario. Se introduce una probabilidad de decision $p_{\text{gen}} \in [0, 1]$:

$$p_{\text{gen}} = \sigma(w_{h^*}^T h_t^* + w_s^T s_t + w_x^T x_t + b_{\text{ptr}})$$

donde $x_t$ es el input del decoder en el paso $t$.

Distribucion final combinada:

$$P(w) = p_{\text{gen}} P_{\text{vocab}}(w) + (1 - p_{\text{gen}}) \sum_{i: w_i = w} a_i^t$$

- Si $p_{\text{gen}} \approx 1$: generar del vocabulario.
- Si $p_{\text{gen}} \approx 0$: copiar palabra del articulo usando la atencion como pointer.

Crucialmente, **incluso palabras OOV** (no en el vocabulario) pueden ser producidas si aparecen en el articulo -- se "copian" via el mecanismo de atencion. Esto resuelve OOV y reduce errores factuales (los detalles exactos se copian literalmente).

Solo 1153 parametros adicionales vs el baseline (para $w_{h^*}, w_s, w_x, b_{\text{ptr}}$).

### 3. Coverage Mechanism

**Problema**: seq2seq tiende a **repetir** frases, especialmente en textos largos.

**Solucion**: mantener un **coverage vector** $c^t$ que suma las distribuciones de atencion de todos los pasos previos:

$$c^t = \sum_{t'=0}^{t-1} a^{t'}$$

Intuitivamente, $c_i^t$ indica "cuanta atencion ha recibido la posicion $i$ hasta ahora". Se incorpora al alignment model:

$$e_i^t = v^T \tanh(W_h h_i + W_s s_t + w_c c_i^t + b_{\text{attn}})$$

El termino $w_c c_i^t$ permite que el modelo aprenda a **desviar** atencion de posiciones ya visitadas.

### 4. Coverage Loss

Se agrega una perdida auxiliar que penaliza atender repetidamente a las mismas posiciones:

$$\text{covloss}_t = \sum_i \min(a_i^t, c_i^t)$$

Intuitivamente, si $c_i^t$ ya es alto (la posicion recibio mucha atencion), el modelo es penalizado por seguir atendiendola.

Loss total:

$$\text{loss}_t = -\log P(w_t^*) + \lambda \sum_i \min(a_i^t, c_i^t)$$

con $\lambda = 1$ en experimentos.

Solo 512 parametros extra ($w_c$).

---

## Resultados experimentales

Dataset: **CNN/Daily Mail** (287K pares articulo-resumen, ~780 tokens/articulo, ~56 tokens/resumen).

Metricas: ROUGE-1/2/L F1, METEOR.

| Modelo | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|---|---|---|---|---|
| Abstractive (Nallapati 2016) | 35.46 | 13.30 | 32.65 | -- |
| Seq2Seq + atten (50k vocab) | 31.33 | 11.81 | 28.83 | 12.03 |
| **Pointer-Generator** | 36.44 | 15.66 | 33.42 | 15.35 |
| **Pointer-Generator + Coverage** | **39.53** | **17.28** | **36.38** | **17.32** |
| Lead-3 baseline (extractive) | 40.34 | 17.70 | 36.57 | 20.48 |

**Observaciones**:

- Pointer-generator solo: +5 ROUGE-1 sobre baseline seq2seq.
- Coverage agrega +3 ROUGE-1 adicionales.
- El modelo final **abstractivo** alcanza casi el mismo nivel que Lead-3 (una baseline extractiva que simplemente toma los primeros 3 sentences del articulo).
- Coverage **elimina repeticiones practicamente por completo** (la Figura 1 ilustra este efecto con ejemplos cualitativos).

---

## Por que importa hoy

- **Pointer-generator pattern**: se volvio un componente estandar en summarization, dialog systems, code generation. Cualquier tarea donde copiar entrada literal sea util (identificadores, numeros, nombres propios) se beneficia.
- **Coverage mechanism**: adaptable a NMT, caption generation, chatbots para evitar loops de repeticion.
- **Pre-Transformer era summarization**: este es el paper de referencia de esa era. Transformers (T5, BART, Pegasus) los reemplazaron, pero con mucha de la intuicion de See/Liu/Manning embebida.
- **Production use**: durante ~2 anos (2017-2019) fue la arquitectura practica para summarization en produccion, antes de la era BERT/T5.
- **Metrica y dataset**: CNN/Daily Mail sigue siendo el benchmark estandar en summarization abstractiva; el split de este paper es el usado universalmente.

---

## Limitaciones

- **Casi-extractivo en la practica**: el modelo frecuentemente elige copy ($p_{\text{gen}} \to 0$) para palabras de contenido, lo que produce resumenes que son mas concatenaciones inteligentes que reformulaciones abstractivas. El analisis cualitativo del paper (Seccion 7) lo admite.
- **Vocabulario aun limitado**: 50K palabras significa que sustantivos muy raros todavia se copian en vez de normalizarse.
- **Entrenamiento dos-fases**: requiere entrenar primero sin coverage, luego anadir coverage loss. Un solo entrenamiento con coverage desde el inicio no converge bien.
- **No paralelo**: como todos los seq2seq LSTM-based, el decoder es secuencial. T5 y Pegasus (Transformer-based) son mas rapidos y performantes.
- **Superado**: BART (2019), Pegasus (2020), T5 (2020) mejoran significativamente (ROUGE-2 > 21 en CNN/DM).

---

## Notas y enlaces

- Las **Figuras 2 y 3** son excelentes visualizaciones de la arquitectura baseline vs pointer-generator.
- La Seccion 7 (Analysis) es particularmente pedagogica: analizan novel vs copy n-grams, patrones de repeticion, etc.
- Codigo oficial: [abisee/pointer-generator](https://github.com/abisee/pointer-generator) (TensorFlow).
- Blog post de la autora: [abigailsee.com](https://www.abigailsee.com/) con excelente explicacion visual.
- Follow-ups:
  - **Paulus et al. 2018** "A Deep Reinforced Model for Abstractive Summarization" -- usa RL (ROUGE como reward) junto con pointer-gen.
  - **Lewis et al. 2020** BART -- Transformer seq2seq preentrenado, estandar moderno.
  - **Zhang et al. 2020** PEGASUS -- pretraining especifico para summarization (gap sentence prediction).

Ver fundamentos: [Mecanismo de Atencion](/fundamentos/mecanismo-atencion) · [Sequence to Sequence](/fundamentos/seq2seq) · [LSTM y GRU](/fundamentos/lstm-gru).
