---
title: "49 - Comparativa final: BERT vs GPT — cierre Camino 4"
weight: 490
math: true
---

## 1. Apertura

Once capitulos despues del cap 38, el Camino 4 cierra. Mini-BERT existe: 952K parametros, MLM pretraining sobre Shakespeare+Quijote, fine-tuning a deteccion de idioma EN/ES con accuracy 0.998. La pregunta inevitable al cierre del camino: **¿en que se diferencia esto de Mini-GPT?** Y mas profundo: **¿cuando usar uno u otro en la practica?**

Este capitulo no agrega codigo. Es una sintesis comparativa de los tres modelos del curso (Mini-GPT, Mini-LLaMA, Mini-BERT), una guia para elegir arquitectura segun tarea, y un repaso historico de como los encoders dominaron NLP entre 2019 y 2022 para luego ser desplazados por los decoders desde 2022 en adelante — sin desaparecer, sino encontrando un segundo nicho en sistemas de busqueda y re-ranking.

---

## 2. Tabla tripartita: Mini-GPT / Mini-LLaMA / Mini-BERT

| Dimension | Mini-GPT | Mini-LLaMA | Mini-BERT |
|---|---|---|---|
| **Tipo arquitectonico** | Decoder-only | Decoder-only | Encoder-only |
| **Direccionalidad** | Causal (izquierda → derecha) | Causal (izquierda → derecha) | Bidireccional |
| **Mascara de atencion** | Triangular inferior | Triangular inferior | Sin mascara |
| **Posicion** | Embedding aprendido | RoPE (rotatorio) | Embedding aprendido |
| **Norm** | Post-LayerNorm | Pre-RMSNorm | Post-LayerNorm |
| **Activacion FFN** | GELU | SwiGLU | GELU |
| **Atencion** | MHA estandar | GQA (grouped-query) | MHA estandar |
| **Cache** | KV-cache | KV-cache | (no aplica) |
| **Special tokens** | (ninguno) | (ninguno) | `[CLS]` `[SEP]` `[MASK]` |
| **Objetivo de pretrain** | Next-token prediction | Next-token prediction | Masked Language Modeling |
| **Loss masking** | (todos los tokens) | SFT: solo respuesta | MLM: 15% de tokens enmascarados |
| **Generacion** | Si (auto-regresiva) | Si (auto-regresiva) | No (representacional) |
| **Output principal** | Logits sobre vocab | Logits sobre vocab | Vector `[CLS]` o por-token |
| **Parametros** | ~600K | ~880K | 952K |
| **Caps del curso** | 1-15 | 16-21 | 38-48 |

Las tres arquitecturas comparten el mismo nucleo (embeddings + bloques transformer + atencion) pero divergen en tres ejes que cambian todo: **direccionalidad de la atencion**, **objetivo de pretraining**, y **forma de uso** (generacion vs representacion).

---

## 3. Cuando usar encoder-only vs decoder-only vs encoder-decoder

### Encoder-only (BERT, RoBERTa, DistilBERT, DeBERTa)

**Usar cuando:** la salida es una etiqueta, una clasificacion, una puntuacion, un vector — no texto generado.

Casos canonicos:
- Clasificacion de texto (sentimiento, intencion, idioma — exactamente lo que hizo Mini-BERT en cap 47)
- Named Entity Recognition (NER): etiquetar tokens individuales
- Question answering extractivo: predecir spans `(start, end)` sobre un parrafo
- Inference de oraciones: entailment, contradiction, neutral
- Embeddings semanticos: convertir texto en vectores para busqueda y similaridad

La bidireccionalidad es la ventaja: cada token "ve" todo el contexto. Para una sola tarea de clasificacion sobre una sola secuencia, eso es ideal.

### Decoder-only (GPT, LLaMA, Claude, Mistral)

**Usar cuando:** la salida es texto generado, una continuacion, una respuesta abierta, una conversacion.

Casos canonicos:
- Chat conversacional
- Generacion de codigo
- Resumen abstractive
- Traduccion (aunque tradicionalmente fue encoder-decoder)
- Completion de texto en cualquier dominio
- Tareas zero-shot via prompting (la propiedad emergente de los LLMs)

La auto-regresion es la propiedad clave: el modelo genera token por token, condicionando en lo que ya genero. Esto permite escalar a salidas de longitud arbitraria. Y la propiedad emergente — que con suficiente escala el modelo aprende tareas sin fine-tuning — es exclusiva de los decoders entrenados a gran escala con next-token prediction.

### Encoder-decoder (T5, BART, FLAN-T5, mBART)

**Usar cuando:** la entrada y la salida son ambas texto, pero estructuralmente distintas.

Casos canonicos:
- Traduccion automatica (entrada: ingles; salida: espanol — distintos)
- Resumen (entrada: documento largo; salida: resumen corto — distintos)
- Question answering generativo (entrada: contexto+pregunta; salida: respuesta libre)
- Reformulacion: parafrasis, simplificacion, correccion gramatical

El encoder procesa la entrada bidireccionalmente; el decoder genera la salida atendiendo a la entrada via cross-attention. Es la arquitectura mas flexible pero con el doble de parametros y mayor costo de inferencia.

**Decision rapida:**
- Salida es etiqueta o vector → encoder-only
- Salida es texto libre, mismo dominio que entrada → decoder-only
- Salida es texto libre, dominio distinto → encoder-decoder

En 2026 la frontera real esta entre encoder-only y decoder-only — los encoder-decoder han perdido protagonismo frente a decoders grandes que pueden hacer cualquier transformacion via prompting.

---

## 4. La historia: tres eras de NLP moderno

### Era 1 (2018-2022): BERT domina

BERT (Devlin et al., 2018) aparecio como "el momento ImageNet de NLP". Antes de BERT, casi todas las tareas de NLP se resolvian con arquitecturas especificas y embeddings preentrenados (Word2Vec, GloVe). BERT propuso un cambio: **un solo encoder preentrenado con MLM + fine-tuning ligero por tarea** rompio benchmarks por 10-20 puntos.

Entre 2018 y 2022, casi todos los papers serios de NLP usaban BERT, RoBERTa (Liu et al., 2019) o variantes (DistilBERT, ALBERT, ELECTRA, DeBERTa). Hugging Face crecio principalmente como hub de modelos BERT-like. Los benchmarks GLUE y SuperGLUE eran liderados por encoders.

### Era 2 (2022-presente): los decoders escalan y desplazan

GPT-3 (Brown et al., 2020) ya habia mostrado que los decoders a gran escala (175B parametros) podian hacer tareas zero-shot via prompting — sin fine-tuning, sin etiquetas, solo contexto en el prompt. Pero fue ChatGPT (noviembre 2022) lo que cambio la percepcion publica e industrial.

Para 2024, los decoders grandes podian hacer clasificacion de texto, NER, QA, sentiment analysis — todas las tareas de BERT — simplemente con prompts. La fine-tunear un BERT para sentimiento dejo de tener sentido cuando un Claude o un GPT-4o lo hacen mejor sin entrenamiento. Los benchmarks tradicionales de NLP empezaron a ser saturados por LLMs.

### Era 3 (presente): encoders en su segunda vida

Los encoders no desaparecieron. Encontraron dos nichos donde siguen siendo superiores:

**1. Embeddings densos.** Para sistemas de busqueda semantica, recuperacion (retrieval) y clustering, los encoders bidireccionales producen embeddings de mejor calidad por dolar que los decoders. Modelos como `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5` y `intfloat/e5-large-v2` son todos encoders BERT-like, fine-tuneados con contrastive learning.

**2. Cross-encoders y re-ranking.** En sistemas RAG (Retrieval-Augmented Generation), la primera etapa usa embeddings rapidos para recuperar candidatos. La segunda etapa usa un cross-encoder BERT-like para re-rankear con mas precision. El cross-encoder concatena `[CLS] query [SEP] candidate [SEP]` y predice un score de relevancia. Es lento — uno por par — pero mucho mas preciso que la similitud de embeddings independientes.

**El balance actual:** los decoders dominan generacion y razonamiento. Los encoders dominan representacion y ranking. La mayoria de pipelines de produccion usan ambos: BERT-like para retrieval, GPT-like para razonar sobre lo recuperado.

---

## 5. Sentence-Transformers: el encoder como motor de busqueda

Sentence-Transformers (Reimers & Gurevych, 2019) reformulo el uso de BERT. En lugar de usar `[CLS]` para clasificacion, lo uso para producir embeddings de oraciones. Fine-tuneando con un objetivo contrastivo (pares positivos atraen, negativos repelen), los autores convirtieron BERT en un modelo de embeddings de calidad superior a Word2Vec promediado.

La arquitectura "siamese" es clave: dos encoders identicos (mismo peso) procesan dos oraciones en paralelo, y la distancia entre sus vectores `[CLS]` es la similitud semantica. Una vez entrenado, encoder cada documento de un corpus de millones produce un indice que permite busqueda en milisegundos via FAISS o Pinecone.

En el ecosistema actual, Sentence-Transformers (con MiniLM, MPNet, E5) es la columna vertebral de:
- Buscadores semanticos
- Sistemas de FAQ automatizados
- Detector de duplicados en bases de conocimiento
- Recomendadores de contenido textual
- Pipelines RAG (etapa 1: retrieval)

Mini-BERT en este curso comparte el ADN con Sentence-Transformers: mismo encoder, mismo `[CLS]`, distinta cabeza. Lo que aprendiste en cap 45 es exactamente la base de la familia entera de modelos de embedding. Cambia el fine-tuning (contrastive en lugar de cross-entropy) y tienes un modelo de busqueda.

---

## 6. Cross-encoders en RAG: la segunda vida de BERT

En un sistema RAG moderno, BERT vive en dos lugares:

```
Usuario hace pregunta
       |
       v
[Query encoder]  ← Sentence-Transformer (BERT-like)
       |
       v
Vector de query  →  busqueda en FAISS/Pinecone
                     (recupera top-100 documentos)
       |
       v
[Cross-encoder]  ← BERT-like (re-ranker)
       |
       v
Top-5 documentos relevantes
       |
       v
[LLM grande]  ← decoder (GPT/Claude/LLaMA)
       |
       v
Respuesta generada con citas
```

El cross-encoder hace algo que el encoder de embeddings no puede: **mira pregunta y candidato juntos**. La concatenacion `[CLS] pregunta [SEP] candidato [SEP]` permite que la atencion bidireccional cruce informacion entre los dos textos. La cabeza de clasificacion predice un score de relevancia.

Esta es la razon de la segunda vida: para precision maxima, cross-encoders BERT-like vencen a embeddings independientes en re-ranking. La penalizacion en latencia (procesar uno por uno cientos de pares) se compensa porque solo se usan sobre los top-100 ya filtrados.

Modelos canonicos: `cross-encoder/ms-marco-MiniLM-L-12-v2`, `BAAI/bge-reranker-large`, `cohere/rerank-english-v3.0` (este ultimo cerrado pero documentado como BERT-like).

El paradigma BERT — pretrain MLM + fine-tune por tarea — sigue siendo la base. Solo cambia la tarea: en lugar de clasificacion de idioma, es ranking de relevancia.

---

## 7. Preguntas finales del Camino 4

**1. ¿Por que BERT no genera texto y GPT no produce embeddings de calidad?**

BERT no genera porque su atencion bidireccional no respeta la causalidad: el modelo "ve" el futuro al hacer predicciones, lo que hace imposible la generacion auto-regresiva. Si intentaras generar token por token con BERT, tendrias que re-procesar toda la secuencia en cada paso, perdiendo eficiencia y consistencia.

GPT si puede producir embeddings — extraer el ultimo hidden state como vector — pero su atencion causal hace que solo el ultimo token "vea" todo el contexto, mientras que los primeros solo ven hasta su posicion. Esto produce vectores asimetricos: el inicio de la secuencia tiene representaciones empobrecidas. Para embeddings de oracion, donde necesitas capturar el significado global de manera uniforme, los encoders bidireccionales son superiores. La excepcion son los modelos GPT-grandes fine-tuneados para embeddings (como `text-embedding-3-large` de OpenAI), que compensan la asimetria con escala.

**2. Si ChatGPT puede clasificar idioma sin entrenamiento, ¿para que aprender BERT?**

Tres razones practicas:

- **Costo:** un BERT fine-tuneado para deteccion de idioma corre en milisegundos en CPU. Un LLM grande cuesta 10-100x mas por consulta. A escala (millones de clasificaciones/dia), la diferencia es decisiva.
- **Latencia:** retrieval semantico necesita encoders rapidos. Indexar 100M de documentos con un decoder grande es prohibitivo. Con BERT-like es trivial.
- **Especializacion:** para tareas estrechas (clasificar tickets, etiquetar entidades en un dominio especifico, scoring), un encoder fine-tuneado supera a un LLM zero-shot — porque tiene mas senal del dominio en su unica tarea.

Pedagogicamente, ademas, BERT ensena algo que GPT no puede: **representacion sin generacion**. Es la base conceptual de embeddings, retrieval, ranking, y de muchas arquitecturas multimodales (CLIP, SigLIP) donde la fase de imagen y texto son ambas encoders.

**3. ¿Cual es el siguiente paso natural despues de Mini-BERT?**

Tres direcciones complementarias:

- **Sentence-Transformers**: fine-tunear Mini-BERT con un objetivo contrastivo (pares positivos/negativos) para producir embeddings de oraciones. Cambia la cabeza y la loss; la arquitectura permanece.
- **Modelos multimodales (Camino 5: ViT)**: el mismo encoder, pero la entrada son patches de imagen en lugar de tokens. CLIP y SigLIP son la version cross-modal: dos encoders (uno texto, uno imagen) entrenados para alinear sus embeddings.
- **Sequence-to-sequence**: combinar Mini-BERT (encoder) con Mini-GPT (decoder) y cross-attention entre ambos. Es la arquitectura T5/BART. El encoder bidireccional "lee" la entrada; el decoder causal genera la salida.

El Camino 5 del curso explora la primera de estas direcciones aplicada a vision: Vision Transformer (ViT) reutiliza la arquitectura encoder de BERT pero opera sobre imagenes. La transferencia conceptual es directa: el patch de imagen es el "token" de BERT.

---

## 8. Cierre del Camino 4

Mini-GPT (Camino 1), Mini-LLaMA (Camino 2.5), Mini-BERT (Camino 4): tres arquitecturas, una familia. El bloque transformer es el comun denominador; las decisiones de diseno producen modelos que generan, modelos que representan, o modelos que hacen ambas cosas.

El Camino 4 cubrio la mitad representacional de la familia. Los encoders no son obsoletos — son complementarios. En cualquier sistema NLP moderno serio (busqueda, RAG, recomendacion, clasificacion a escala), conviven encoders rapidos para representar y decoders grandes para razonar. Saber construir ambos es saber el campo.

**Caminos pendientes:**
- **Camino 5: Vision Transformer (ViT)** — el encoder BERT aplicado a parches de imagen.

Mini-BERT vive en `clase_14/practica/`. El checkpoint fine-tuneado, el dataset bilingue, y los 11 scripts ejecutables permanecen como referencia. El paradigma se transfiere a ViT casi sin cambios — eso lo veremos en el Camino 5.
