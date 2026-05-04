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
  {{< era name="Era pre-neural" years="1948-2010" >}}
    {{< hito year="1948" name="n-gramas (Shannon)" status="minimal" >}}
      Modelos de lenguaje basados en frecuencias de secuencias cortas. **Por qué importó:** estableció el problema de predecir la siguiente palabra y la métrica de perplexity.
    {{< /hito >}}
    {{< hito year="2003" name="Bengio NNLM" status="minimal" >}}
      Primera red neuronal que aprende simultáneamente embeddings densos y un modelo de lenguaje probabilístico end-to-end. **Por qué importó:** validó la idea de representaciones distribuidas aprendidas, base conceptual de toda la era siguiente.
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
    {{< hito year="2018" name="BERT" status="deep" link="/fundamentos/bert" >}}
      Pretraining bidireccional con MLM: el primer modelo que volvió obsoleto entrenar desde cero para cada tarea.
    {{< /hito >}}
    {{< hito year="2018-2019" name="GPT-1 / GPT-2" status="minimal" >}}
      Decoder-only autoregresivo entrenado en texto crudo. **Por qué importó:** mostró que la generación de texto coherente emerge solo con escala y next-token prediction.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de los LLMs" years="2020-presente" >}}
    {{< hito year="2020" name="GPT-3" status="minimal" >}}
      175B parámetros, few-shot in-context learning. **Por qué importó:** la escala desbloqueó capacidades cualitativamente nuevas (razonamiento, programación) sin fine-tuning.
    {{< /hito >}}
    {{< hito year="2022" name="InstructGPT / SFT + RLHF" status="deep" link="/fundamentos/sft" >}}
      Alineamiento por feedback humano: convierte un modelo de lenguaje en un asistente útil y seguro.
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
