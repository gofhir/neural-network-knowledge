---
title: "Texto / NLP"
weight: 1
sidebar:
  open: true
---

# Texto / NLP

## El problema central

El lenguaje natural es **discreto, composicional y ambiguo**. A diferencia de una imagen — donde los píxeles vecinos están altamente correlacionados — en texto la unidad mínima (la palabra o el token) no tiene una métrica natural de "cercanía": *gato* y *perro* son tokens completamente distintos a nivel de símbolo, aunque semánticamente parecidos. Las dependencias importantes pueden estar a una palabra de distancia o a 500 palabras, y el significado de una palabra cambia con el contexto.

Esto fuerza tres decisiones arquitectónicas que vertebran toda la historia del NLP neuronal: (1) cómo representar palabras como vectores densos que capturen similitud semántica, (2) cómo modelar dependencias de largo alcance entre tokens, y (3) cómo entrenar a escala con texto sin etiquetar — porque el texto etiquetado nunca alcanza, pero texto crudo hay infinito.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era pre-neural" years="1948-2014" >}}
    {{< hito year="1948" name="n-gramas (Shannon)" status="minimal" >}}
      Modelos de lenguaje basados en frecuencias de secuencias cortas. **Por qué importó:** estableció el problema de predecir la siguiente palabra y la métrica de perplexity.
    {{< /hito >}}
    {{< hito year="1954" name="Hipótesis distribucional (Harris)" status="minimal" >}}
      *"The meaning of a word is given by the company it keeps."* Fundamento conceptual del Bag of Words y, mucho después, de word2vec y embeddings distribuidos. **Por qué importó:** estableció la idea de que la semántica emerge de la co-ocurrencia.
    {{< /hito >}}
    {{< hito year="1980" name="Porter Stemmer" status="covered" link="/papers/porter-stemmer-1980" >}}
      Algoritmo simple en 5 pasos para reducir palabras inglesas a su raíz por eliminación iterativa de sufijos. **Por qué importó:** baseline obligatoria del NLP clásico durante 40+ años; sigue corriendo en producción dentro de Lucene, Elasticsearch, NLTK.
    {{< /hito >}}
    {{< hito year="1988" name="TF-IDF (Salton & Buckley)" status="minimal" >}}
      Pondera Bag of Words castigando palabras frecuentes y amplificando las raras y discriminativas. **Por qué importó:** la representación textual estándar de Information Retrieval por décadas; sigue siendo baseline competitiva en clasificación.
    {{< /hito >}}
    {{< hito year="1995" name="WordNet" status="covered" link="/papers/wordnet-miller-1995" >}}
      Lexicón inglés organizado por relaciones semánticas (synsets, hipónimos, meronimia) en lugar de orden alfabético. **Por qué importó:** el recurso léxico más influyente del NLP. Base de NLTK lemmatizer, scispaCy, BabelNet.
    {{< /hito >}}
    {{< hito year="2001" name="NLTK toolkit" status="covered" link="/papers/nltk-bird-loper-2006" >}}
      Suite Python con interfaces uniformes, 15+ corpora preempaquetados, demos GUI. **Por qué importó:** empujó la adopción de Python como lengua franca del NLP, precediendo a scikit-learn (2007), gensim (2009), spaCy (2015) y Transformers (2018).
    {{< /hito >}}
    {{< hito year="2003" name="Bengio NNLM" status="minimal" >}}
      Primera red neuronal que aprende simultáneamente embeddings densos y un modelo de lenguaje probabilístico end-to-end. **Por qué importó:** validó la idea de representaciones distribuidas aprendidas, base conceptual de toda la era siguiente.
    {{< /hito >}}
    {{< hito year="2006" name="Punkt sentence tokenizer" status="covered" link="/papers/punkt-kiss-strunk-2006" >}}
      Algoritmo no supervisado para segmentar texto en oraciones, basado en detectar abreviaciones como collocations. **Por qué importó:** sentence tokenizer default de NLTK; sigue corriendo en miles de pipelines NLP pre-Transformer.
    {{< /hito >}}
    {{< hito year="2011" name="TweetTokenizer / Twitter POS" status="covered" link="/papers/twitter-pos-gimpel-2011" >}}
      Tokenizer y POS tagger especializados para texto de redes sociales (emoticons, hashtags, mentions, URLs). **Por qué importó:** primer tagset estándar para Twitter; la herramienta de facto para tokenizar tweets en Python.
    {{< /hito >}}
    {{< hito year="2014" name="VADER (rule-based sentiment)" status="covered" link="/papers/vader-hutto-gilbert-2014" >}}
      Modelo de sentiment analysis con lexicón de 7,500 entradas + 5 reglas heurísticas. F1=0.96 en tweets, superando humanos individuales. **Por qué importó:** demostró que rule-based pragmático puede competir con ML supervisado; sigue siendo baseline obligatoria de sentiment en redes sociales.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de embeddings" years="2013-2017" >}}
    {{< hito year="2013" name="word2vec" status="covered" link="/fundamentos/embeddings-distribuidos" >}}
      Skip-gram y CBOW: embeddings entrenables a escala que capturan analogías ("rey - hombre + mujer ≈ reina").
    {{< /hito >}}
    {{< hito year="2014" name="GloVe" status="minimal" >}}
      Embeddings basados en factorización de la matriz de coocurrencias global. **Por qué importó:** alternativa a word2vec con mejor uso de estadísticas globales del corpus.
    {{< /hito >}}
    {{< hito year="2016" name="FastText" status="minimal" >}}
      Embeddings que descomponen palabras en n-gramas de caracteres. **Por qué importó:** maneja palabras fuera de vocabulario y morfología rica.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era recurrente y seq2seq" years="1997-2016" >}}
    {{< hito year="1997" name="LSTM (Hochreiter & Schmidhuber)" status="deep" link="/fundamentos/lstm-gru" >}}
      Redes con memoria de largo plazo capaces de modelar dependencias entre tokens distantes. La aplicación masiva a NLP llega ~17 años después con seq2seq.
    {{< /hito >}}
    {{< hito year="2014" name="Seq2Seq (Sutskever)" status="deep" link="/fundamentos/seq2seq" >}}
      Encoder-decoder con LSTMs: el primer modelo que traducía oraciones completas extremo a extremo.
    {{< /hito >}}
    {{< hito year="2015" name="Bahdanau attention" status="deep" link="/fundamentos/mecanismo-atencion" >}}
      Atención sobre el encoder: rompe el cuello de botella del vector de contexto fijo y permite oraciones largas.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de la atención" years="2017-2020" >}}
    {{< hito year="2017" name="Transformer" status="deep" link="/fundamentos/transformer" >}}
      *Attention is all you need*: self-attention pura, sin recurrencias. Paralelismo masivo en training.
    {{< /hito >}}
    {{< hito year="2018" name="ELMo" status="deep" link="/papers/elmo-peters-2018" >}}
      Deep Contextualized Word Representations (Peters et al., NAACL 2018 Best Paper). Char-CNN + 2 BiLSTM con forward y backward LMs entrenados conjuntamente, combinación lineal task-specific. **Por qué importó:** introdujo *embeddings contextuales* — el vector de cada palabra depende de su oración. Mejoró SOTA en SQuAD, SNLI, SRL, Coref, NER, SST-5.
    {{< /hito >}}
    {{< hito year="2018" name="GPT-1" status="covered" link="/papers/gpt-1-radford-2018" >}}
      Decoder-only Transformer entrenado con generative pre-training en BookCorpus, fine-tuneado por tarea con input transformations (Radford et al. 2018). **Por qué importó:** estableció el patrón *pretrain + fine-tune* en arquitectura decoder y descubrió zero-shot behaviors emergentes.
    {{< /hito >}}
    {{< hito year="2018" name="BERT" status="deep" link="/fundamentos/bert" >}}
      Pretraining bidireccional con MLM + NSP: el primer modelo que volvió obsoleto entrenar desde cero para cada tarea. BERT-large 82.1 vs SOTA previa 74.0 en GLUE.
    {{< /hito >}}
    {{< hito year="2019" name="GPT-2" status="covered" link="/papers/gpt-2-radford-2019" >}}
      *Language Models are Unsupervised Multitask Learners* (Radford et al. 2019). 1.5B params en WebText (40GB Reddit-filtered) evaluado zero-shot — SOTA en 7 de 8 datasets sin fine-tuning. **Por qué importó:** validó la idea de que un LM grande aprende implícitamente muchas tareas vía prompting natural.
    {{< /hito >}}
    {{< hito year="2019" name="RoBERTa" status="minimal" >}}
      BERT-large entrenado con más datos, más cómputo, sin NSP, mejor búsqueda de hyperparams (Liu et al. 2019). **Por qué importó:** mostró que la arquitectura BERT estaba sub-entrenada, anticipando scaling laws.
    {{< /hito >}}
    {{< hito year="2019" name="BETO" status="minimal" >}}
      BERT-base entrenado en español por el grupo de Jorge Pérez en DCC UChile (Cañete et al. 2020). **Por qué importó:** referencia de NLP en español; corre en producción dentro de pipelines clínicos y de gobierno en Chile.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de los LLMs" years="2020-presente" >}}
    {{< hito year="2020" name="GPT-3" status="deep" link="/papers/gpt-3-brown-2020" >}}
      175B parámetros, few-shot in-context learning (Brown et al., NeurIPS 2020 Best Paper). 300B tokens entrenados sobre Common Crawl filtered + WebText2 + Books + Wikipedia. **Por qué importó:** la escala desbloqueó capacidades cualitativamente nuevas (razonamiento, programación, aritmética) sin fine-tuning; pavimentó la era de prompt engineering.
    {{< /hito >}}
    {{< hito year="2021" name="BERTIN" status="minimal" >}}
      Proyecto comunitario que entrenó BERT en español sobre mC4-es con distintas estrategias de muestreo (gaussian, stepwise, random). **Por qué importó:** demostró que la comunidad open-source puede producir modelos competitivos sin compute industrial.
    {{< /hito >}}
    {{< hito year="2022" name="InstructGPT / SFT + RLHF" status="deep" link="/papers/instructgpt-ouyang-2022" >}}
      Alineamiento por feedback humano formalizado en pipeline 3-pasos (SFT → Reward Model → PPO). Ouyang et al. 2022 mostró que InstructGPT 1.3B vence a GPT-3 175B en preferencia humana. **Por qué importó:** la receta técnica detrás de ChatGPT (noviembre 2022) y de toda la familia chat-tuned (Claude, LLaMA-chat, Mistral-Instruct).
    {{< /hito >}}
    {{< hito year="2022" name="ChatGPT launch (noviembre 2022)" status="minimal" >}}
      OpenAI libera ChatGPT como producto público gratuito sobre GPT-3.5. Alcanza 100M usuarios en 2 meses — el producto digital con adopción más rápida de la historia. **Por qué importó:** marca el punto donde los LLMs entran a la conciencia masiva y se vuelven una expectativa de usabilidad estándar.
    {{< /hito >}}
    {{< hito year="2022" name="NLLB-200 (No Language Left Behind)" status="covered" link="/papers/nllb-team-2022" >}}
      Transformer Mixture-of-Experts de Meta para traducción entre 200 idiomas (40,602 direcciones). **Por qué importó:** primer modelo MT con cobertura masiva en idiomas low-resource (quechua, kinyarwanda, etc.); statement-of-the-art moral del MT moderno con open source completo.
    {{< /hito >}}
    {{< hito year="2023" name="DPO" status="deep" link="/fundamentos/dpo" >}}
      Direct Preference Optimization: alineamiento sin RL, equivalente teórico a RLHF pero más simple y estable.
    {{< /hito >}}
    {{< hito year="2023-2025" name="LLMs frontier" status="covered" link="/fundamentos/foundation-models" >}}
      GPT-4/5, Claude, Gemini, LLaMA: razonamiento extendido, herramientas, contexto largo, multimodalidad.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}

## Era 1 — Pre-neural (1948-2010)

### Problema heredado

Antes de las redes neuronales, modelar lenguaje era un problema de conteo. Shannon (1948) había mostrado que el lenguaje tiene estructura estadística predecible, y los modelos de n-gramas estimaban directamente $P(w_n \mid w_{n-k}, ..., w_{n-1})$ contando frecuencias en corpus. Funcionaban razonablemente para tareas locales (autocompletado, ASR) pero sufrían **dispersidad** — la mayoría de las secuencias de 4 palabras nunca aparecen en el corpus — y no capturaban similitud semántica: para un n-grama, *gato* y *felino* son tokens completamente distintos.

### Idea clave

Usar redes neuronales para **representar palabras como vectores densos en un espacio continuo** donde la cercanía geométrica captura cercanía semántica. El paper de Bengio et al. (2003) introdujo este principio en su *Neural Probabilistic Language Model*: una red feedforward aprendía simultáneamente los embeddings de las palabras y un modelo de lenguaje sobre ellas.

### Qué la destronó

La era pre-neural no terminó por una arquitectura mejor sino por un cambio de **disponibilidad de cómputo**. Cuando entrenar embeddings sobre miles de millones de palabras se volvió viable, la representación distribuida ganó por knockout.

## Era 2 — Embeddings distribuidos (2013-2017)

### Problema heredado

Bengio había probado que los embeddings funcionaban, pero su modelo era costoso: la red feedforward predecía la palabra siguiente con una softmax sobre el vocabulario completo. Para corpus realistas (miles de millones de tokens) era inviable.

### Idea clave

**Aprender embeddings sin modelar la distribución completa.** word2vec (Mikolov, 2013) reformuló el problema: en lugar de predecir la palabra siguiente, entrena con dos tareas mucho más baratas — predecir contexto desde una palabra (skip-gram) o palabra desde contexto (CBOW) — usando **negative sampling** para esquivar la softmax cara. El resultado: embeddings entrenables sobre billones de palabras en horas.

### Qué la destronó

Los embeddings de word2vec son **estáticos**: la palabra *banco* tiene un único vector, independiente de si es entidad financiera o asiento. Para tareas que requieren resolver ambigüedad por contexto — y eso es prácticamente todo NLP serio — esto es una limitación dura. Las RNN y luego los Transformers prometían **embeddings contextuales** que resuelven el sentido en función de la oración completa.

## Era 3 — Recurrente y seq2seq (1997-2016)

### Problema heredado

Las redes feedforward y los embeddings estáticos tratan tokens en aislamiento. El lenguaje es secuencial: *"el gato persigue al ratón"* y *"el ratón persigue al gato"* tienen los mismos tokens y embeddings promedio idénticos, pero significan cosas opuestas.

### Idea clave

Procesar el texto **token por token** en orden, manteniendo un **estado oculto** que se actualiza en cada paso y resume todo lo visto hasta ahora. Las RNN clásicas tenían el problema del gradiente que se desvanece sobre secuencias largas; **LSTM (Hochreiter y Schmidhuber, 1997) y GRU** resolvieron esto con compuertas que aprenden cuándo retener y cuándo olvidar. La aplicación masiva al NLP no llegó hasta ~17 años después: Sutskever et al. (2014) llevaron la idea a su punto natural con **Seq2Seq**: un encoder LSTM resume la oración fuente en un vector y un decoder LSTM genera la traducción token a token.

Bahdanau et al. (2015) agregaron **atención** sobre el encoder: en cada paso del decoder, el modelo aprende dónde mirar en la fuente, eliminando el cuello de botella del vector único de contexto.

### Qué la destronó

Las RNN son **secuencialmente irreductibles**: hay que procesar el token $t$ antes de procesar el $t+1$. Esto las hace lentas en GPUs modernas, que están hechas para paralelismo masivo. Y la atención de Bahdanau era un parche sobre la recurrencia. La pregunta natural era: *¿y si quitamos la recurrencia y dejamos solo la atención?*

## Era 4 — Atención pura y pretraining (2017-2020)

### Problema heredado

Las RNN no paralelizaban y la atención existía solo como complemento. El campo necesitaba una arquitectura que pudiera aprovechar GPU y TPU al máximo, y que escalara a contextos cada vez más largos.

### Idea clave

**Self-attention en lugar de recurrencia.** El Transformer (Vaswani et al., 2017) reemplaza por completo las RNN: cada token atiende a todos los demás en una sola operación matricial paralelizable. La información posicional, que la recurrencia provee implícitamente, se inyecta vía positional encoding.

El segundo gran salto fue desacoplar **arquitectura** de **régimen de entrenamiento**. BERT (Devlin et al., 2018) propuso entrenar un Transformer encoder en **Masked Language Modeling** sobre Wikipedia + BookCorpus, y luego fine-tunear sobre tareas específicas. El resultado: un único modelo pre-entrenado destronaba a soluciones especializadas en docenas de benchmarks.

GPT-1 y GPT-2 exploraron la versión decoder-only del mismo principio, entrenada en next-token prediction sobre texto crudo masivo.

### Qué la destronó

BERT y los modelos encoder-only sobreviven en producción para clasificación, búsqueda y embedding. Pero la dirección que terminó dominando fue la **decoder-only autoregresiva escalada**: el camino GPT.

## Era 5 — LLMs y alineamiento (2020-presente)

### Problema heredado

GPT-2 había mostrado que un decoder Transformer entrenado en next-token prediction generaba texto sorprendentemente coherente. Pero seguía siendo un *modelo de lenguaje* — no un asistente. Y nadie había probado qué pasaba al escalarlo cien veces más.

### Idea clave

Tres ideas que se combinaron:

1. **Escala bruta.** GPT-3 (Brown et al., 2020) llevó el tamaño a 175B parámetros y entrenó sobre cientos de miles de millones de tokens. Capacidades cualitativamente nuevas — razonamiento, programación, traducción — emergieron sin entrenamiento específico, vía *in-context learning*.
2. **Alineamiento por feedback humano.** InstructGPT (2022) mostró que un GPT-3 fine-tuneado primero con SFT (datos de demostración) y luego con RLHF (modelo de recompensa entrenado sobre preferencias humanas) se vuelve dramáticamente más útil y seguro. Es lo que separó a GPT-3 de ChatGPT.
3. **Alineamiento sin RL.** DPO (Rafailov et al., 2023) demostró que el objetivo de RLHF se puede reescribir como una pérdida supervisada directa sobre pares de preferencias, eliminando la necesidad del modelo de recompensa y de PPO.

### Qué viene

Esta es la era actual. Las direcciones activas — razonamiento extendido (chain-of-thought, o1, agentes), contexto largo, multimodalidad nativa, herramientas, modelos pequeños competitivos — se desarrollan en paralelo, sin un sucesor claro todavía.

## Estado del arte hoy

{{< callout type="info" >}}

**Frontier LLMs (2024-2025).** Los modelos punteros combinan escala (∼1T parámetros), entrenamiento sobre billones de tokens curados, RLHF/DPO y técnicas de razonamiento extendido.

- **GPT-5** — OpenAI. Razonamiento por defecto, contexto extendido, capacidades multimodales nativas (texto + imagen + audio).
- **Claude Opus 4.7** — Anthropic. Contexto de 1M tokens, foco en razonamiento sostenido y uso de herramientas en tareas largas.
- **Gemini 2.5** — Google DeepMind. Multimodal nativo desde el pretraining, integración profunda con búsqueda y herramientas.
- **LLaMA 4** — Meta. Open weights, pesos abiertos competitivos a frontera cerrada en muchos benchmarks.
- **DeepSeek-R1** — DeepSeek. Modelo de razonamiento abierto entrenado con RL puro sobre cadenas de pensamiento.

{{< /callout >}}

## Casos de uso reales

- **Asistentes conversacionales** (ChatGPT, Claude, Gemini): productividad general, redacción, programación.
- **Búsqueda semántica y RAG**: Google AI Overviews, Perplexity, asistentes corporativos sobre documentación interna.
- **Generación de código**: GitHub Copilot, Cursor, Claude Code — completación, refactor y agentes que ejecutan tareas extremo a extremo.
- **Traducción automática**: DeepL, Google Translate (modelos NMT actuales son Transformers descendientes directos de Seq2Seq + atención).
- **Extracción de información estructurada**: del texto libre a JSON conforme a esquema — facturas, historias clínicas, contratos.
- **Moderación y clasificación a escala**: filtros de spam, detección de toxicidad, triaje de tickets de soporte.

## Qué viene

Las apuestas activas hoy — sin un ganador claro — incluyen: **razonamiento explícito** (modelos como o-series y R1 que producen chain-of-thought antes de responder), **agentes** (LLMs que ejecutan secuencias largas de acciones con herramientas), **contextos extra-largos** (millones de tokens, memoria persistente entre conversaciones), **modelos pequeños competitivos** (eficiencia por destilación e instrucción cuidada), y **arquitecturas más allá del Transformer** (Mamba, RWKV, mezclas de expertos a gran escala). Cuál de estas líneas marca el siguiente salto cualitativo es la pregunta abierta de 2025.

## Recursos relacionados

**Fundamentos:**
- [Embeddings distribuidos](/fundamentos/embeddings-distribuidos) — word2vec, GloVe, embeddings contextuales.
- [Redes recurrentes](/fundamentos/redes-recurrentes) y [LSTM/GRU](/fundamentos/lstm-gru).
- [Seq2Seq](/fundamentos/seq2seq) y [mecanismo de atención](/fundamentos/mecanismo-atencion).
- [Self-attention](/fundamentos/self-attention) y [Transformer](/fundamentos/transformer).
- [Positional encoding](/fundamentos/positional-encoding).
- [BPE — tokenización](/fundamentos/bpe).
- [BERT](/fundamentos/bert) y [pretraining BERT](/fundamentos/pretraining-bert).
- [SFT](/fundamentos/sft), [DPO](/fundamentos/dpo) y [KL implícito](/fundamentos/kl-implicito).
- [Foundation models](/fundamentos/foundation-models).

**Papers:**
- [Attention is All You Need (Vaswani 2017)](/papers/attention-is-all-you-need-vaswani-2017).
- [BERT (Devlin 2018)](/papers/bert-devlin-2018).
- [Seq2Seq (Sutskever 2014)](/papers/seq2seq-sutskever-2014).
- [Bahdanau attention (2015)](/papers/bahdanau-attention-2015).
- [LSTM (Hochreiter 1997)](/papers/lstm-hochreiter-1997).
- [GRU (Cho 2014)](/papers/gru-cho-2014).

**Clases del diplomado:**
- Clase 13 — RNNs, seq2seq y atención.
- Clase 14 — Transformer, GPT, BERT, alineamiento.

---

*Última actualización: 2026-05-03.*
