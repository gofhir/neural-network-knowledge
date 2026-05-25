---
title: "NLLB - No Language Left Behind"
weight: 167
math: true
---

{{< paper-card
    title="No Language Left Behind: Scaling Human-Centered Machine Translation"
    authors="NLLB Team (Meta AI)"
    year="2022"
    venue="arXiv preprint"
    pdf="/papers/nllb-team-2022.pdf"
    arxiv="2207.04672" >}}
Esfuerzo masivo de Meta AI para escalar la traducción automática neural de ~100 a **202 idiomas**, cubriendo familias low-resource históricamente excluidas (acholi, tumbuka, igbo, fon, kanuri, banjar, bhojpuri…). Combina **FLORES-200** (benchmark con 40.602 direcciones de traducción), **NLLB-200** (modelo Mixture-of-Experts de 54.5B parámetros) y un pipeline completo de data mining más toxicity detection. Su versión distilada de 600M es la que se usa en el lab 16.
{{< /paper-card >}}

---

## Contexto

La traducción automática (Machine Translation, MT) recorrió este camino:

```
1940s-1990s    1990s-2010s    2010s-2017       2017-2020      2020-2022          2022+
Reglas         SMT            RNN/LSTM         Transformer    Multilingual MT    MoE @ 200 idiomas
+ diccionarios IBM models     seq2seq          (Vaswani'17)   M2M-100 (Meta'20)  NLLB-200 (Meta'22)
ALPAC report   (Brown'93)     attention                       54B params, 100 lang
(1966)         moses          Bahdanau'15
```

Hasta 2022, los grandes sistemas de MT multilingüe cubrían **~100 idiomas**, todos high-resource o mid-resource. El internet mostraba un sesgo brutal: el 63.7% de los sitios web están en inglés, y **solo el 25.9% de los usuarios de internet hablan inglés como L1**. Idiomas como catalán, asamés, ligurio o kinyarwanda quedaban fuera de la conversación digital.

NLLB se planteó duplicar la cobertura de **100 a 200 idiomas**, incluyendo varios extremadamente low-resource. El proyecto se enmarca en *Value Sensitive Design* (Friedman & Hendry, 2019): empieza con **44 entrevistas a hablantes nativos** de idiomas low-resource antes de decidir arquitectura.

La frase con la que se abre el paper, citando a Jack Vance (*The Eyes of the Overworld*, 1977):
> *"In order to facilitate your speech, I endow you with this instrument which relates all possible vocables to every conceivable system of meaning."*

El dispositivo mágico de Cugel es, hoy, Machine Translation. NLLB es el intento más ambicioso de construirlo para todo idioma humano.

---

## Ideas principales

NLLB es un esfuerzo masivo que combina **datos, modelos, evaluación y consideraciones éticas**. Las contribuciones se agrupan en cuatro pilares (Figura 1 del paper).

### 1. Estudios humanos con hablantes low-resource

44 entrevistas semi-estructuradas con hablantes nativos de 36 idiomas (5 NA, 8 SA, 4 Europa, 12 África, 7 Asia). Reveló preocupaciones sobre:

- Declive cultural y económico de idiomas locales.
- Coverage incompleto en sistemas comerciales (ej. árabe con múltiples languoids —marroquí, egipcio, etc.— tratados como uno solo).
- Calidad pobre actual disuade el uso ("traducir y luego editar toma más tiempo que traducir manualmente").
- Toxicidad y traducciones crudas que erosionan confianza.

### 2. Datasets creados profesionalmente

- **FLORES-200**: dataset de evaluación con **204 idiomas**, traducido profesionalmente. Cada frase está en TODOS los idiomas, lo que permite evaluar **40.602 direcciones de traducción** (202 × 201).
- **NLLB-Seed**: datos seed de entrenamiento en 39 idiomas low-resource.
- **NLLB-MD** (Multi-Domain): datos seed en 6 idiomas para evaluar generalización a dominios distintos (chat, salud, viajes, noticias, novela, Wikipedia).
- **Toxicity-200**: listas de palabras tóxicas en 200 idiomas para detectar y prevenir traducciones tóxicas o alucinadas.

### 3. Herramientas y pipelines de minado de datos

- **Language Identification** para 200+ idiomas (necesario para limpiar crawls web masivos).
- **LASER3**: encoder de sentencias para identificar bitext alineado en 148 idiomas; extracción automática de pares paralelos desde corpus monolingüe masivo.
- **stopes**: librería de data mining para procesar y alinear datos monolingüe → bitext.

### 4. Modelos

- **NLLB-200**: modelo **Sparsely Gated Mixture-of-Experts** de **54.5B parámetros**, soporta **202 idiomas, 40.602 direcciones de traducción**.
- **Variantes densas**: 3.3B y 1.3B Transformer denso.
- **Modelos distilados**: 1.3B y **600M** distilados desde NLLB-200, para deployment más liviano.

El modelo usado en el lab 16 (`facebook/nllb-200-distilled-600M`) es la versión distilada de 600M parámetros: pesos compactos para que corra en una notebook Colab gratuita.

---

## Arquitectura

### Setup general

NLLB-200 es un Transformer encoder-decoder estándar (Vaswani et al. 2017) con varias modificaciones críticas:

$$P(T \mid S, \ell_s, \ell_t)$$

condiciona en source, source-lang y target-lang.

- **Source language** se inyecta como prefix en la secuencia (el token `eng_Latn` o `spa_Latn` aparece al inicio del input). Decisión clave: prefixar con **source lang** en lugar de **target lang** mejora el rendimiento zero-shot para pares no vistos durante el training. Esto contrasta con la línea anterior de Arivazhagan et al. 2019 y Johnson et al. 2017, que prefixaban con target.
- **Target language** se pasa al decoder como token de inicio (BOS embedding).

### SentencePiece tokenizer

- Un único SentencePiece (Kudo & Richardson 2018) entrenado sobre **100M sentencias** muestreadas de todos los idiomas con sampling temperature $\tau=5$ (upsamplea low-resource).
- **Vocabulario: 256.000 subword units.** Tamaño masivo necesario para representar adecuadamente 200+ idiomas con scripts variados (latín, cirílico, árabe, devanagari, etíope, chino, etc.).

Esta vocab de 256k es lo que hace que cargar NLLB-200 ocupe tanta RAM solo en embeddings ($|V| \times d_{\text{model}}$).

### Transformer + Pre-LayerNorm

Cada capa Transformer:

$$Z = X + \text{self-attention}(\text{norm}(X))$$

$$Y = Z + \text{feed-forward}(\text{norm}(Z))$$

Pre-LayerNorm (norm antes de cada sublayer) es más estable durante entrenamiento que Post-LN (Xiong et al. 2020). Esto es relevante para entrenar modelos del tamaño de NLLB-200 estables a 100k+ updates.

### Sparsely Gated Mixture of Experts (MoE)

**El cambio arquitectónico clave** que permite escalar de 1.3B a 54B parámetros sin volverse computacionalmente prohibitivo.

#### Idea base

En vez de un FFN denso de tamaño $d_{\text{ffn}} = 8192$ en cada capa Transformer, se tienen **E expertos paralelos** (cada uno un FFN propio), y un **gating network** decide qué experto(s) procesa cada token.

#### Formalmente (ecs. 6-8 del paper)

$$\text{FFN}_e(x_t) = W_o^{(e)} \cdot \text{ReLU}(W_i^{(e)} \cdot x_t)$$

$$G_t = \text{softmax}(W_g \cdot x_t)$$

$$\mathcal{G}_t = \text{Top-k-Gating}(G_t)$$

$$\text{MoE}(x_t) = \sum_e \mathcal{G}_{te} \cdot \text{FFN}_e(x_t)$$

- $E = 64$ expertos por capa MoE.
- Top-2 gating: cada token se rutea a sus 2 expertos mejor-scored.
- $f_{\text{MoE}} = 2$: se inserta MoE en cada capa Transformer **alternada** (capas 2, 4, 6, …).
- Capacity factor: cada experto procesa máximo $2T/E$ tokens del mini-batch (forzar load balance).
- **Load balancing loss** (ec. 9): término auxiliar que empuja a distribución uniforme entre expertos.

Esto es **conditional compute**: el modelo tiene 54B parámetros pero **solo activa ~1B por token**, manteniendo el costo de FLOPs comparable a un modelo denso de 1.3B.

#### Por qué MoE para multilingual MT

Hipótesis: con 200+ idiomas hay tanta diversidad que comprimir todo en parámetros compartidos causa **interferencia** entre idiomas no relacionados. Expertos especializados pueden capturar familias lingüísticas (románicas, eslavas, sino-tibetanas, …) o tareas (idiomas con scripts no latinos, idiomas tonales, …).

Análisis empírico (sección 6.2.4): se hace forward pass y se mide qué expertos se activan por idioma. Encuentra que en capas tempranas del encoder el routing es más "language-agnostic"; en capas profundas se especializa.

### Regularización: el problema del overfitting en low-resource

Vanilla MoE + low-resource = overfitting catastrófico. El paper documenta tres técnicas:

| Técnica | Mecanismo |
|---|---|
| **Overall dropout** ($p_{\text{drop}}$) | Dropout uniforme en todo el modelo. Mejora respecto a sin-dropout. |
| **MoE Expert Output Masking (EOM)** | Para una fracción aleatoria de tokens, *enmascarar el output del experto* antes de combinar. Esto fuerza al residual a transportar más señal y reduce co-adaptation entre expertos top-2. Es **MoE-específico**. |
| **Conditional MoE Routing (CMR)** | Añade una rama densa paralela $\text{FFN}_{\text{shared}}$. Un gate binario decide cuándo usar el MoE pesado vs el shared dense. Útil para tokens "simples" que no necesitan capacidad MoE. |

Tabla 16 del paper (ablation): **EOM con $p_{\text{drop}}=0.3$, $p_{\text{eom}}=0.1$ es la mejor configuración global**, ganándole a vanilla MoE +0.4 chrF++ across-the-board y **+0.6 a +0.9 chrF++ en very low-resource**.

### Curriculum Learning

Para pares low-resource que overfittean en pocos updates, introducen los pares por *phases*: empezar entrenando solo con high-resource e introducir progresivamente buckets de low-resource más cerca del final del entrenamiento. Esto limita updates innecesarios sobre pares pequeños y reduce overfitting.

### Self-supervision y back-translation

Para los idiomas extremadamente low-resource (<100k pares paralelos):

- **Back-translation:** generar pseudo-paralelos invirtiendo dirección con un modelo intermedio.
- **Self-supervision** (denoising objectives en datos monolingüe).
- **Bootstrapping con NLLB-Seed:** la pequeña cantidad de datos profesionales sirve como ancla de calidad.

---

## Resultados

### Métrica principal: chrF++

Se usa **chrF++** (Popović 2017) en lugar de solo BLEU porque:

- BLEU funciona mal para idiomas morfológicamente ricos o de scripts no-latinos.
- chrF++ usa precisión y recall de **character n-grams** + word n-grams. Más robusto across-scripts.

Para 40.602 direcciones de traducción, calcular BLEU sería costoso y poco informativo.

### Headline result

> *"Our model achieves an improvement of 44% BLEU relative to the previous state-of-the-art."*

Comparado con M2M-100 (Fan et al. 2020), su predecesor que cubría 100 idiomas, NLLB-200 logra +44% BLEU promedio. Para low-resource el incremento es aún mayor.

### Evaluación humana

Además de métricas automáticas (chrF++, spBLEU), realizan evaluación humana usando XSTS (Cross-lingual Semantic Textual Similarity): anotadores nativos comparan traducciones contra referencia humana en escala 1-5. La correlación entre chrF++ y XSTS varía por idioma, lo que justifica reportar ambas.

### Detección de toxicidad

Toxicity-200 detecta cuándo el modelo genera contenido tóxico no presente en el source: fenómeno conocido como *added toxicity* / *hallucinated toxicity*. Crítico para deployment: una traducción que invente insultos puede causar daño grave.

### Distribuciones disponibles

| Modelo | Parámetros | Disco | Hardware |
|---|---|---|---|
| `nllb-200-54B` (MoE) | 54.5B | ~120 GB | GPU 80GB+ (A100) |
| `nllb-200-3.3B` (denso) | 3.3B | ~15 GB | GPU 24GB+ |
| `nllb-200-1.3B` (denso) | 1.3B | ~6 GB | GPU 12GB+ |
| `nllb-200-distilled-1.3B` | 1.3B | ~5 GB | GPU 8GB+ |
| **`nllb-200-distilled-600M`** | 600M | **~2.4 GB** | **CPU posible** |

---

## Limitaciones

Discutidas explícitamente en el paper (sección 9):

1. **Cobertura ≠ paridad de calidad.** Un idioma puede estar "soportado" con BLEU 15 mientras inglés-español está a BLEU 45. Inclusión nominal no es lo mismo que utilidad real.
2. **El modelo hereda sesgos de la web.** Los corpora minados de internet sobre-representan dominios formales (Wikipedia, gobierno, religión) y no capturan el español/portugués/etc. coloquial real.
3. **El modelo alucina.** Tablas en el paper muestran ejemplos de "hallucinated tokens": el modelo genera output gramaticalmente correcto pero semánticamente disociado del input. Más frecuente en low-resource.
4. **Janus-faced digital participation** (sección 9.2): traer un idioma low-resource al internet vía MT no es neutro. Puede:
   - Empoderar a la comunidad (acceso a Wikipedia, oportunidades económicas).
   - Acelerar la pérdida de matiz cultural si MT homogeneiza el discurso.
5. **Energía y huella ambiental.** El paper incluye sección 8.8 sobre impacto ambiental: entrenar 54B con MoE consume megavatios-hora.

---

## Por qué importa hoy

NLLB es el **statement-of-the-art moral** del MT moderno: no se trata solo de mejorar el BLEU, sino de **decidir conscientemente qué idiomas merecen ser tratados con calidad humana**. Algunos efectos:

- **Deployment público**: Meta integra NLLB en Facebook, Instagram y WhatsApp para traducir posts de idiomas low-resource. Editores de Wikipedia usan NLLB para generar borradores en idiomas sin artículos.
- **Open source completo**: todo el código (fairseq/nllb), modelos en HuggingFace Hub, datasets en GitHub. **Esto es excepcional** en una empresa Big Tech.
- **FLORES-200 se vuelve el benchmark estándar.** Cualquier paper de MT multilingüe post-2022 reporta resultados en FLORES-200.
- **Modelos distilled** (600M, 1.3B) permiten deployment en edge devices. El `nllb-200-distilled-600M` del lab corre en una Colab gratuita con ~3GB de RAM.
- **Versión Nature 2024**: la versión peer-reviewed en Nature (junio 2024) consolida el trabajo y le da peso institucional adicional.

A mayo de 2026, **~3500 citas en Google Scholar** y creciendo rápidamente. El modelo distilled de 600M tiene **~5M descargas/mes** en HuggingFace.

La filosofía "cubrir todo idioma humano con MT de calidad" se generalizó a otros dominios (speech, OCR, ASR multilingüe) y empujó la conversación sobre **equidad lingüística** en sistemas de IA.

---

## Lecturas relacionadas

**Predecesores directos:**

- Vaswani et al. (2017), *Attention is All You Need* — Transformer base.
- Johnson et al. (2017), *Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation* — el primer multilingual MT de Google con language tokens.
- Arivazhagan et al. (2019), *Massively Multilingual Neural Machine Translation in the Wild* — el primer modelo de Google que escala a ~100 lenguas.
- Fan et al. (2020), *Beyond English-Centric Multilingual Machine Translation* (M2M-100) — predecesor inmediato de NLLB de Meta.

**MoE foundations:**

- Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* — la idea original de MoE en deep learning.
- Lepikhin et al. (2020), GShard — escalado MoE a billions de parámetros para MT.
- Du et al. (2021), GLaM — MoE para language modeling de Google.

**Toxicity y safety en MT:**

- Costa-jussà et al. (2023), *Toxicity in Multilingual Machine Translation at Scale* — detalle sobre Toxicity-200.

---

## Notas y enlaces

- **Clase asociada**: [Clase 16 - NLP clásico, NLTK, BoW, embeddings](/clases/clase-16).
- **Laboratorio asociado**: [Lab 16 - Pipeline NLP con NLTK/spaCy/NLLB/VADER](/laboratorios/lab-16).
- **Fundamento relacionado**: [Seq2seq](/fundamentos/seq2seq).
- **Cita BibTeX**:

```bibtex
@article{nllb2022,
  title={No Language Left Behind: Scaling Human-Centered Machine Translation},
  author={{NLLB Team} and Costa-juss{\`a}, Marta R and Cross, James and {\c{C}}elebi, Onur and Elbayad, Maha and Heafield, Kenneth and others},
  journal={arXiv preprint arXiv:2207.04672},
  year={2022}
}
```
