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
