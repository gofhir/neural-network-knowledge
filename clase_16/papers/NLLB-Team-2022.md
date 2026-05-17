# No Language Left Behind: Scaling Human-Centered Machine Translation — NLLB Team / Meta AI (2022)

**Autores:** NLLB Team (Meta AI), UC Berkeley, Johns Hopkins University. Líderes de investigación: Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, et al. Corresponding: Angela Fan.
**Publicación:** Pre-print arXiv 2207.04672 (Julio 2022). Versión publicada en *Nature* (Junio 2024) como "Scaling neural machine translation to 200 languages".
**PDF local:** `NLLB-Team-2022.pdf` (192+ páginas incluyendo apéndices)
**Conexión con el laboratorio:** El bloque 4 del Práctico 16 (celdas 53-62) usa **NLLB-200** vía HuggingFace `transformers` para traducir entre inglés y español. La celda 71 (Actividad 4) lo combina con VADER para hacer sentiment analysis multilingüe vía traducción a inglés.

---

## 1. Contexto histórico

La traducción automática (Machine Translation, MT) recorrió este camino:

```
1940s-1990s    1990s-2010s    2010s-2017       2017-2020      2020-2022          2022+
Reglas         SMT            RNN/LSTM         Transformer    Multilingual MT    MoE @ 200 idiomas
+ diccionarios IBM models     seq2seq          (Vaswani'17)   M2M-100 (Meta'20)  NLLB-200 (Meta'22)
ALPAC report   (Brown'93)     attention                       54B params, 100 lang
(1966) ☠       moses          Bahdanau'15
```

**Por qué NLLB es importante en esta línea:** hasta 2022, los grandes sistemas de MT multilingüe cubrían **~100 idiomas**, todos high-resource o mid-resource. El internet mostraba un sesgo brutal: el 63.7% de los sitios web están en inglés, y **solo el 25.9% de los usuarios de internet hablan inglés como L1**. Idiomas como Catalán, Asamés, Ligurio, Kinyarwanda quedaban fuera de la conversación digital.

NLLB se planteó duplicar la cobertura de **100 a 200 idiomas**, incluyendo varios extremadamente low-resource (Acholi, Tumbuka, Igbo, Fon, Kanuri, Banjar, Bhojpuri…). El proyecto se enmarca en *Value Sensitive Design* (Friedman & Hendry, 2019): empieza con **44 entrevistas a hablantes nativos** de idiomas low-resource antes de decidir arquitectura.

La frase con la que se abre el paper, citando a Jack Vance (*The Eyes of the Overworld*, 1977):
> *"In order to facilitate your speech, I endow you with this instrument which relates all possible vocables to every conceivable system of meaning."*

El dispositivo mágico de Cugel es, hoy, Machine Translation. NLLB es el intento más ambicioso de construirlo para todo idioma humano.

---

## 2. Contribución central

NLLB es un esfuerzo masivo que combina **datos, modelos, evaluación y consideraciones éticas**. Las contribuciones se agrupan en cuatro pilares (Figura 1 del paper):

### 2.1 Estudios humanos con hablantes low-resource
44 entrevistas semi-estructuradas con hablantes nativos de 36 idiomas (5 NA, 8 SA, 4 Europa, 12 África, 7 Asia). Reveló preocupaciones sobre:
- Decline cultural y económico de idiomas locales.
- Coverage incompleto en sistemas comerciales (ej. Arabic con múltiples languoids — Moroccan, Egyptian, etc. — tratados como uno solo).
- Calidad pobre actual disuade el uso ("traducir y luego editar toma más tiempo que traducir manualmente").
- Toxicidad / traducciones crudas que erosionan confianza.

### 2.2 Datasets creados profesionalmente
- **FLORES-200**: dataset de evaluación con **204 idiomas**, traducido profesionalmente. Cada frase está en TODOS los idiomas → permite evaluar **40,602 direcciones de traducción** (202 × 201).
- **NLLB-Seed**: datos seed de entrenamiento en 39 idiomas low-resource.
- **NLLB-MD** (Multi-Domain): datos seed en 6 idiomas para evaluar generalización a dominios distintos (chat, salud, viajes, noticias, novela, Wikipedia).
- **Toxicity-200**: listas de palabras tóxicas en 200 idiomas para detectar y prevenir traducciones tóxicas/alucinadas.

### 2.3 Herramientas y pipelines de minado de datos
- **Language Identification** para 200+ idiomas (necesario para limpiar crawls web masivos).
- **LASER3**: encoder de sentencias para identificar bitext alineado en 148 idiomas — extracción automática de pares paralelos desde corpus monolingüe masivo.
- **stopes**: librería de data mining para procesar y alinear datos monolingüe → bitext.

### 2.4 Modelos
- **NLLB-200**: modelo **Sparsely Gated Mixture-of-Experts** de **54.5B parámetros**, soporta **202 idiomas, 40,602 direcciones de traducción**.
- **Variantes densas**: 3.3B y 1.3B Transformer denso.
- **Modelos distilados**: 1.3B y **600M** distilados desde NLLB-200, para deployment más liviano.

**El modelo usado en el lab** (celda 58, `facebook/nllb-200-distilled-600M`) es la versión distilada de 600M parámetros — pesos compactos para que corra en una notebook Colab gratuita.

---

## 3. Arquitectura

### 3.1 Setup general

NLLB-200 es Transformer encoder-decoder estándar (Vaswani et al. 2017) con varias modificaciones críticas:

```
P(T | S, ℓ_s, ℓ_t)   # condiciona en source, source-lang, target-lang
```

- **Source language** se inyecta como prefix en la secuencia (el token `eng_Latn` o `spa_Latn` aparece al inicio del input). Decisión clave: prefixar con **source lang** en lugar de **target lang** mejora el rendimiento zero-shot para pares no vistos durante el training. Esto contrasta con la línea anterior de Arivazhagan et al. 2019 y Johnson et al. 2017 que prefixaban con target.
- **Target language** se pasa al decoder como token de inicio (BOS embedding).

### 3.2 SentencePiece tokenizer

- Un único SentencePiece (Kudo & Richardson 2018) entrenado sobre **100M sentencias** muestreadas de todos los idiomas con sampling temperature τ=5 (upsamplea low-resource).
- **Vocabulario: 256,000 subword units.** Tamaño masivo necesario para representar adecuadamente 200+ idiomas con scripts variados (latín, cirílico, árabe, devanagari, etíope, chino, etc.).

Esta vocab de 256k es lo que hace que cargar NLLB-200 ocupe tanta RAM solo en embeddings (vocab × d_model).

### 3.3 Transformer + Pre-LayerNorm

Cada capa Transformer:
```
Z = X + self-attention(norm(X))     # Pre-LN, no Post-LN
Y = Z + feed-forward(norm(Z))
```

Pre-LayerNorm (norm antes de cada sublayer) es más estable durante entrenamiento que Post-LN (Xiong et al. 2020). Esto es relevante para entrenar modelos del tamaño de NLLB-200 estables a 100k+ updates.

### 3.4 Sparsely Gated Mixture of Experts (MoE)

**El cambio arquitectónico clave** que permite escalar de 1.3B a 54B parámetros sin volverse computacionalmente prohibitivo.

#### Idea base
En vez de un FFN denso de tamaño d_ffn = 8192 en cada capa Transformer, se tienen **E expertos paralelos** (cada uno un FFN propio), y un **gating network** decide qué experto(s) procesar cada token.

#### Formalmente (ecs. 6-8 del paper):
```
FFN_e(x_t) = W_o^(e) · ReLU(W_i^(e) · x_t)         # expert e
G_t = softmax(W_g · x_t)                            # gating logits
𝒢_t = Top-k-Gating(G_t)                             # quedarse con los top-2
MoE(x_t) = Σ 𝒢_te · FFN_e(x_t)                     # combinar expertos elegidos
```

- E = 64 expertos por capa MoE.
- Top-2 gating: cada token se rutea a sus 2 expertos mejor-scored.
- f_MoE = 2: insertamos MoE en cada capa Transformer **alternada** (capas 2, 4, 6, …).
- Capacity factor: cada experto procesa máximo 2T/E tokens del mini-batch (forzar load balance).
- **Load balancing loss** (ec. 9): término auxiliar que empuja a distribución uniforme entre expertos.

Esto es **conditional compute**: el modelo tiene 54B parámetros pero **solo activa ~1B por token**, manteniendo el costo de FLOPs comparable a un modelo denso de 1.3B.

#### Por qué MoE para multilingual MT
Hipótesis: con 200+ idiomas, hay tanta diversidad que comprimir todo en parámetros compartidos causa **interferencia** entre idiomas no relacionados. Expertos especializados pueden capturar familias lingüísticas (románicas, eslavas, sino-tibetanas, …) o tareas (idiomas con scripts no latinos, idiomas tonales, …).

Análisis empírico (sección 6.2.4): hace forward pass y mide qué expertos se activan por idioma. Encuentra que en capas tempranas del encoder el routing es más "language-agnostic"; en capas profundas se especializa.

### 3.5 Regularización: el problema del overfitting en low-resource

Vanilla MoE + low-resource = overfitting catastrófico. El paper documenta tres técnicas:

| Técnica | Mecanismo |
|---|---|
| **Overall dropout** (p_drop) | Dropout uniforme en todo el modelo. Mejora respecto a sin-dropout. |
| **MoE Expert Output Masking (EOM)** | Para una fracción aleatoria de tokens, *enmascarar el output del experto* antes de combinar. Esto fuerza al residual a transportar más señal y reduce co-adaptation entre expertos top-2. Es **MoE-específico**. |
| **Conditional MoE Routing (CMR)** | Añade una rama densa paralela `FFN_shared`. Un gate binario decide cuándo usar el MoE pesado vs el shared dense. Útil para tokens "simples" que no necesitan capacidad MoE. |

Tabla 16 del paper (ablation): **EOM con p_drop=0.3, p_eom=0.1 es la mejor configuración global**, ganándole a vanilla MoE +0.4 chrF++ across-the-board y **+0.6 a +0.9 chrF++ en very low-resource**.

### 3.6 Curriculum Learning

Para pares low-resource que overfittean en pocos updates, introducen los pares por *phases*: empezar entrenando solo con high-resource, e introducir progresivamente buckets de low-resource más cerca del final del entrenamiento. Esto limita updates innecesarios sobre pares pequeños y reduce overfitting.

### 3.7 Self-supervision y back-translation

Para los idiomas extremadamente low-resource (<100k pares paralelos):
- **Back-translation:** generar pseudo-paralelos invirtiendo dirección con un modelo intermedio.
- **Self-supervision** (denoising objectives en datos monolingüe).
- **Bootstrapping con NLLB-Seed:** la pequeña cantidad de datos profesionales sirve como ancla de calidad.

---

## 4. Resultados clave

### 4.1 Métrica principal: chrF++

Se usa **chrF++** (Popović 2017) en lugar de solo BLEU porque:
- BLEU funciona mal para idiomas morfológicamente ricos o de scripts no-latinos.
- chrF++ usa precisión y recall de **character n-grams** + word n-grams. Más robusto across-scripts.

Para 40,602 direcciones de traducción, calcular BLEU sería costoso y poco informativo.

### 4.2 Headline result

> *"Our model achieves an improvement of 44% BLEU relative to the previous state-of-the-art."*

Comparado con M2M-100 (Fan et al. 2020), su predecesor que cubría 100 idiomas, NLLB-200 logra +44% BLEU promedio. Para low-resource el incremento es aún mayor.

### 4.3 Evaluación humana
Además de métricas automáticas (chrF++, spBLEU), realizan evaluación humana usando XSTS (Cross-lingual Semantic Textual Similarity) — anotadores nativos comparan traducciones contra referencia humana en escala 1-5. La correlación entre chrF++ y XSTS varía por idioma, lo cual justifica reportar ambas.

### 4.4 Detección de toxicidad

Toxicity-200 detecta cuándo el modelo genera contenido tóxico no presente en el source — fenómeno conocido como *added toxicity* / *hallucinated toxicity*. Crítico para deployment: una traducción que invente insultos puede causar daño grave.

---

## 5. Limitaciones y consideraciones éticas

Discutidas explícitamente en el paper (sección 9):

1. **Cobertura ≠ paridad de calidad.** Un idioma puede estar "soportado" con BLEU 15 mientras inglés-español está a BLEU 45. Inclusión nominal no es lo mismo que utilidad real.
2. **El paper hereda sesgos de la web.** Los corpora minados de internet sobre-representan dominios formales (Wikipedia, gobierno, religión) y no capturan el español/portugués/etc. coloquial real.
3. **El modelo halucinar.** Tablas en el paper muestran ejemplos de "hallucinated tokens" — el modelo genera output gramaticalmente correcto pero semánticamente disociado del input. Más frecuente en low-resource.
4. **Janus-faced digital participation** (sección 9.2): traer un idioma low-resource al internet vía MT no es neutro. Puede:
   - Empoderar a la comunidad (acceso a Wikipedia, oportunidades económicas).
   - Acelerar la pérdida de matiz cultural si MT homogeneiza el discurso.
5. **Energía y huella ambiental.** El paper incluye sección 8.8 sobre impacto ambiental — entrenar 54B con MoE consume megavatios-hora.

---

## 6. Impacto y legado

NLLB es el **statement-of-the-art moral** del MT moderno: no se trata solo de hacer más BLEU, sino de **decidir conscientemente qué idiomas merecen ser tratados con calidad humana**. Algunos efectos:

- **Deployment público**: Meta integra NLLB en Facebook, Instagram, WhatsApp para traducir posts de idiomas low-resource. Wikipedia editors usan NLLB para generar borradores en idiomas sin artículos.
- **Open source completo**: todo el código (fairseq/nllb), modelos en HuggingFace Hub, datasets en GitHub. **Esto es excepcional** en una empresa Big Tech.
- **FLORES-200 se vuelve el benchmark estándar.** Cualquier paper de MT multilingüe post-2022 reporta resultados en FLORES-200.
- **Modelos distilled** (600M, 1.3B) permiten deployment en edge devices. El `nllb-200-distilled-600M` del lab corre en una Colab gratuita con ~3GB de RAM.
- **Versión Nature 2024**: la versión peer-reviewed en Nature (Junio 2024) consolida el trabajo y le da peso institucional adicional.

A mayo de 2026, **~3500 citas en Google Scholar** y creciendo rápidamente. El modelo distilled de 600M tiene **~5M descargas/mes** en HuggingFace.

---

## 7. Conexión directa con el Práctico 16

| Celda del lab | Concepto del paper |
|---|---|
| 56 | Link a `flores200/README.md` — el paper introduce **FLORES-200** como benchmark. Los códigos BCP-47 tipo `eng_Latn`, `spa_Latn`, `fra_Latn`, `cmn_Hans` provienen de aquí. |
| 57 | `pip install transformers==4.56.1` — versión fija porque APIs de tokenizer NLLB cambiaron en versiones recientes. |
| 58 | `AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")` y `AutoModelForSeq2SeqLM.from_pretrained(...)` — carga el modelo de 600M distilled descrito en sección 8.6 *Making Large Models More Accessible through Distillation*. |
| 58 | `pipeline("translation", ..., src_lang="eng_Latn", tgt_lang="spa_Latn")` — usa el patrón de prefix language token descrito en §6.1.1. |
| 60-62 | Traducción inglés→español y vuelta. Permite ver cómo NLLB-200 maneja un round-trip (translate to + translate back). Si la salida final no coincide con el input original, sirve como demostración informal de pérdida de información en traducción neural. |

**Una verificación que podés hacer:** Tomá una frase ambigua o cultural en español ("nos vemos al tiro" — chilenismo) y mirá cómo NLLB la traduce a inglés. Su entrenamiento sobre Wikipedia + crawls genéricos sesgará hacia el español "neutro" / formal.

---

## 8. Lecturas relacionadas

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

**Para el patrón "translate-then-analyze" del lab:**
- VADER (Hutto & Gilbert 2014) → solo inglés → traducir con NLLB es el patrón canónico para sentiment multilingüe rápido sin entrenar modelos por idioma. Ver `VADER-Hutto-Gilbert-2014.md`.
