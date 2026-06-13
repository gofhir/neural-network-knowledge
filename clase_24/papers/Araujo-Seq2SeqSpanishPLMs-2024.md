# Sequence-to-Sequence Spanish Pre-trained Language Models

> Análisis técnico exhaustivo para el curso IA UC (Diplomado en Inteligencia Artificial, PUC Chile). Clase 24 — Question Answering y Machine Reading Comprehension. Este paper es el sustento directo del Laboratorio 24 Parte 2 (QA generativo con BARTO y T5S sobre SQAC).

## 1. Metadata

| Campo | Valor |
|---|---|
| Título | Sequence-to-Sequence Spanish Pre-trained Language Models |
| Autores | Vladimir Araujo, Maria Mihaela Trusca, Rodrigo Tufiño, Marie-Francine Moens |
| Afiliación | LIIR Lab, KU Leuven (Bélgica); IDEIAGEOCA, Universidad Politécnica Salesiana, Quito (Ecuador) |
| Venue | LREC-COLING 2024 (Joint International Conference on Computational Linguistics, Language Resources and Evaluation) |
| arXiv | arXiv:2309.11259v2 [cs.CL], 21 de marzo de 2024 |
| Modelos liberados | BARTO (BART en español), T5S (T5 en español) y variantes BERT2BERT-style |
| Disponibilidad | https://github.com/vgaraujov/Seq2Seq-Spanish-PLMs — modelos públicos en Hugging Face Hub (`vgaraujov/bart-base-spanish`, `vgaraujov/t5-base-spanish`) |
| Palabras clave | generative models, pre-trained language models, sequence-to-sequence models, transformer |

Es un paper de *recursos lingüísticos* (de ahí su hogar natural en LREC): su contribución no es una arquitectura nueva sino un conjunto de modelos preentrenados que llenan un hueco concreto del ecosistema del español. El primer autor, Vladimir Araujo, es además el autor del notebook del Laboratorio 24 Parte 2 del curso, lo que vuelve a este paper especialmente relevante: el lab no usa modelos genéricos, usa exactamente los modelos que se presentan aquí. La senior author, Marie-Francine Moens, dirige el LIIR Lab de KU Leuven.

## 2. Contexto: el hueco que el paper llena

La historia del NLP en español venía corriendo en paralelo a la del inglés, pero con un retraso estructural y, sobre todo, con una asimetría de cobertura arquitectónica. Para 2023 el español ya tenía modelos **encoder-only** sólidos y modelos **decoder-only** decentes, pero carecía de modelos **encoder-decoder (seq2seq)** preentrenados de forma nativa.

El paper sitúa con precisión las tres familias del Transformer (Vaswani et al., 2017):

**(i) Encoder-only.** Modelos tipo BERT, especializados en *comprensión* (natural language understanding). En español: **BETO** (Cañete et al., 2020), un BERT entrenado sobre el corpus SUC; **ALBETO/DistilBETO** (Cañete et al., 2022), versiones ligeras; **BERTIN** (De la Rosa et al., 2022), un RoBERTa entrenado sobre la porción española de mC4 con muestreo por perplejidad; **MarIA** (Gutiérrez-Fandiño et al., 2022), familia de RoBERTa y GPT-2 entrenados sobre el crawl de la Biblioteca Nacional de España; y **RigoBERTa** (Serrano et al., 2022), basado en DeBERTa. Modelos regionalizados sobre Twitter (Tellez et al., 2023) completan el panorama.

**(ii) Decoder-only.** Modelos tipo GPT, especializados en *generación* autorregresiva. En español el GPT-2 de MarIA es el representante principal.

**(iii) Encoder-decoder.** Modelos que mapean una secuencia de entrada a una secuencia de salida de longitud distinta y condicionada a la entrada (Sutskever et al., 2014): el encoder procesa toda la entrada en paralelo, el decoder genera la salida de forma autorregresiva atendiendo a las representaciones del encoder vía *cross-attention*. Son la herramienta natural para *summarization*, *traducción*, *diálogo*, *split-and-rephrase* y *QA generativo*. En inglés esta familia estaba madura desde 2019-2020 con **MASS** (Song et al., 2019), **BART** (Lewis et al., 2020a) y **T5** (Raffel et al., 2020). Otras lenguas ya habían recibido sus versiones: BARThez para francés, GreekBART, IndicBART, AraBART, BARTpho (vietnamita); y T5 en portugués (PTT5), italiano (IT5), árabe (AraT5), indic, etc.

El español era la excepción notable: a pesar de ser una de las lenguas más habladas del mundo, **no tenía un BART ni un T5 entrenados exclusivamente sobre corpus en español**. El único recurso disponible eran los modelos multilingües (mBART, mT5), que reparten su capacidad entre decenas o cien idiomas y suelen quedar por debajo de los modelos monolingües específicos. El paper cita el patrón consolidado de los BERT específicos por idioma (CamemBERT francés, RobBERT neerlandés, FinBERT finés, GePpeTto italiano): todos superan a sus contrapartes multilingües, lo que justifica el esfuerzo de entrenar desde cero.

La tesis de motivación es entonces directa y de servicio: **democratizar los modelos seq2seq para la comunidad NLP hispanohablante**, entregando modelos abiertos, reproducibles y competitivos, junto con un benchmark curado de tareas seq2seq en español que antes no existía de forma consolidada.

## 3. Contribución central: BARTO y T5S

El núcleo del paper son dos modelos preentrenados desde cero sobre corpus en español, más una familia de baselines.

**BARTO** — la versión española de BART. Sigue la arquitectura BART-base: encoder y decoder de **6 capas cada uno**, **12 cabezas de atención**, **768 dimensiones ocultas**. El nombre es un juego: BART + O (la O de la terminación masculina del español, igual que BETO).

**T5S** — la versión española de T5. Sigue la variante **T5.1.1-base**: encoder y decoder de **12 capas cada uno**, 12 cabezas, 768 dimensiones ocultas. T5.1.1 introduce mejoras sobre el T5 original (GEGLU en lugar de ReLU en las capas feed-forward, sin compartir parámetros entre embeddings y capa de salida, sin dropout durante el preentrenamiento).

**BERT2BERT-style** — baselines monolingües construidos sin preentrenamiento seq2seq propio, siguiendo Rothe et al. (2020): se inicializa un encoder-decoder a partir de checkpoints encoder-only ya existentes. El decoder hereda los pesos del checkpoint pero sus capas de cross-attention se inicializan al azar (BERT no las tiene). Se construyen cuatro: **BETO2BETO** y **BETOShare** (a partir del checkpoint de BETO), y **RoBERTa2RoBERTa** y **RoBERTaShare** (a partir de RoBERTa-BNE de MarIA). Las variantes "Share" comparten parámetros entre encoder y decoder. Estos modelos no requieren preentrenamiento adicional: se fine-tunean directamente en cada tarea.

### 3.1 Datos de preentrenamiento

El paper es meticuloso con el corpus, porque la calidad de los datos determina el resultado (Liu et al., 2019; Raffel et al., 2020). Combina tres fuentes:

| Corpus | Tamaño | Descripción |
|---|---|---|
| **OSCAR 21.09** (Abadji et al., 2022) | ~160 GB | Conjunto deduplicado del español del crawl multilingüe OSCAR |
| **mC4-es** (Xue et al., 2021) | ~500 GB | Subconjunto español de mC4 con muestreo por perplejidad gaussiana de De la Rosa et al. (2022) |
| **SUC** (corpus de BETO) | ~14 GB | Texto crudo de fuentes diversas; se excluye su Wikipedia y se reemplaza por un dump actualizado (~10 GB) |

El **pipeline de preprocesamiento** tiene cuatro etapas:

1. **Formato a nivel de documento.** Cada instancia es un documento con varias oraciones contiguas y coherentes. Liu et al. (2019) mostraron que restringir las secuencias a un único documento (en vez de empaquetar fragmentos de varios) rinde algo mejor y captura dependencias contextuales amplias.
2. **Filtrado de calidad.** Se eliminan documentos muy cortos, texto con caracteres repetidos o especiales atípicos del español, páginas con código y contenido sensible. Se usa el clasificador **fastText** (Joulin et al., 2017) para excluir documentos con menos de 98% de confianza de ser español — un umbral deliberadamente no perfecto, para retener la pequeña proporción de otras lenguas que se entremezclan naturalmente con el español contemporáneo (este detalle será clave para explicar el desempeño en traducción).
3. **Deduplicación.** Con la librería `text-dedup` sobre todos los corpus. Deduplicar reduce el tamaño, acelera el entrenamiento y mejora el resultado (Lee et al., 2022).
4. **Corrección de encoding.** Con `ftfy`, garantizando UTF-8 y normalización NFKC al final del pipeline.

El corpus resultante supera los **120 GB** de texto sin comprimir, una escala comparable a la usada para entrenar RoBERTa y BART en inglés.

### 3.2 Objetivos de preentrenamiento y configuración

Aquí está la diferencia conceptual fundamental entre los dos modelos, que conviene tener clarísima porque es el corazón de "BART vs T5":

**BARTO — denoising autoencoder.** BART se preentrena *reconstruyendo texto corrompido*. Se aplica una transformación de ruido al documento de entrada y el modelo aprende a regenerar el documento original completo. BARTO usa la combinación recomendada por Lewis et al. (2020a): **text infilling** (se enmascara el **30%** de los tokens, reemplazando spans contiguos por un único token `[MASK]`, de modo que el modelo debe inferir tanto el contenido como *cuántos* tokens faltaban) más **sentence permutation** (se permutan todas las oraciones del documento). El decoder predice la salida completa, lo que entrena explícitamente su capacidad de modelado de lenguaje.

**T5S — span corruption.** T5 se preentrena con el objetivo de *rellenar spans eliminados* (fill-in dropped-out spans). Se corrompe el **15%** de los tokens con una **longitud media de span de 3**; cada span eliminado se reemplaza en la entrada por un token centinela único (`<extra_id_0>`, `<extra_id_1>`, …) y el objetivo es generar *solo* los spans faltantes, separados por esos mismos centinelas — no el documento completo. Es un objetivo más "económico" en cómputo del decoder que el de BART.

| Hiperparámetro | BARTO | T5S |
|---|---|---|
| Arquitectura | BART-base (6+6 capas) | T5.1.1-base (12+12 capas) |
| Objetivo | text infilling (30%) + sentence permutation | span corruption (15%, span medio 3) |
| Tokenizer | SentencePiece BPE, **50.264** tokens | SentencePiece unigram, **32.000** tokens |
| Librería | fairseq | nanoT5 |
| Pasos de entrenamiento | 100.000 | 130.000 |
| Hardware | 8× NVIDIA A100 | 4× NVIDIA A100 |
| Longitud de entrada | 1024 | 1024 |
| Batch size | 2048 | 320 |
| Optimizador | Adam | AdamW |
| Warm-up | 10.000 pasos | 10.000 pasos |
| Dropout | 0.1 | 0 |
| Precisión | FP16 | BF16 |

El tokenizer merece atención: cada modelo tiene su propio vocabulario subword construido sobre texto español. BARTO usa BPE de ~50k (como BART/RoBERTa), T5S usa unigram de 32k (como T5). Un vocabulario nativo en español es justamente lo que evita la fragmentación excesiva que sufren los tokenizers multilingües al codificar tildes, ñ y morfología rica del español.

## 4. Arquitectura: BART vs T5, recordatorio

Para leer los resultados conviene recordar por qué estas dos arquitecturas, ambas encoder-decoder, se comportan distinto:

- **BART** es esencialmente BERT (encoder bidireccional) + GPT (decoder autorregresivo) pegados, entrenados como un *autoencoder de denoising*. Su decoder ve la salida entera durante el entrenamiento, lo que refuerza su modelado de lenguaje generativo. Es robusto para generación de texto largo y fluido.
- **T5** trata *toda* tarea de NLP como un problema text-to-text: entrada de texto → salida de texto, incluso para clasificación (donde la "respuesta" es la etiqueta escrita como string). Su preentrenamiento por span corruption lo hace fuerte en tareas con alto solapamiento entre entrada y salida.

La adaptación al español no cambió estas arquitecturas: cambió los **datos** (corpus español de 120 GB), el **tokenizer** (vocabulario nativo) y nada más en la estructura. Es preentrenamiento *from scratch*, no continuación de un checkpoint inglés. Eso es lo que distingue a BARTO/T5S de los baselines BERT2BERT (que reciclan checkpoints) y de los multilingües (que diluyen su capacidad).

## 5. Experimentos

El paper curó un benchmark de tareas seq2seq en español, dividido en **tareas generativas** (el foco) y **tareas discriminativas** (para situar las debilidades). Todos los modelos se fine-tunean en un RTX 3090 con la librería `transformers` (PyTorch), con búsqueda de hiperparámetros sobre batch size ∈ {4, 8, 16}, learning rate (AdamW) ∈ {3e-5, 5e-5} y épocas ∈ {3, 6}. Los baselines multilingües son **mT5-base** (101 idiomas) y **mBART-large** (mBART-50, que incluye español y portugués; no existe mBART-base comparable, de ahí que se use la versión large, que es más grande y juega con ventaja).

### 5.1 Tareas generativas, datasets y métricas

| Tarea | Dataset(s) | Métrica |
|---|---|---|
| Summarization abstractivo | MLSUM (noticias, ~900 tok), WikiLingua (guías, ~500 tok) | ROUGE-1/2/L |
| Summarization de texto largo | XL-Sum (artículos ~1200 tok), EUR-Lex-Sum (legal, ~19000 tok in, ~1200 tok out) | ROUGE-1/2/L |
| Split-and-rephrase | BiSECT-es (~290k instancias) | SARI, BLEU |
| **QA generativo** | **MLQA, SQAC** | ROUGE-1/2/L |
| Diálogo | MIAM-es (corpus de actos de diálogo, adaptado a pares contexto-respuesta) | F1, METEOR |
| Traducción | Fapesp-v2 (PT↔ES), WMT13 (EN↔ES) | BLEU |

**Resultado general:** **BARTO y T5S son los mejores en prácticamente todas las tareas generativas**, superando tanto a los BERT2BERT-style monolingües como a los multilingües mBART/mT5 — pese a que mBART-large es un modelo *grande* y los suyos son base. La excepción sistemática es EUR-Lex-Sum (legal), donde los multilingües ganan por su exposición previa a documentos jurídicos durante el preentrenamiento multilingüe.

**Summarization (Tabla 1, dev/test).** En MLSUM y WikiLingua, **T5S es el mejor** con un promedio de **26.54 ROUGE**, seguido de BARTO con **25.49**. El mejor BERT2BERT (BETO2BETO) promedia 24.39. T5S supera a los multilingües salvo mBART en WikiLingua, donde mBART le gana por 0.45 ROUGE. Números concretos de T5S en MLSUM: R1 30.14/29.44, R2 12.27/11.56, RL 24.62/23.88.

**Summarization largo (Tabla 2).** En XL-Sum, T5S (22.06 prom.) supera ligeramente a BARTO (21.58). En EUR-Lex-Sum, BARTO (56.82) supera a T5S (56.42), pero aquí los multilingües se imponen: mBART-large llega a R1 ~68 / R2 ~52 / RL ~58, muy por encima. Los autores lo atribuyen al formato a nivel de documento durante el preentrenamiento (que ayuda con secuencias largas) y, en el caso legal, a la diversidad de tipos de texto vistos por los multilingües.

**QA generativo (Tabla 3) — la tarea más relevante para la Clase 24.** En SQAC y MLQA, **BARTO y T5S dominan ampliamente**, con **T5S como el mejor, alcanzando 67.92 ROUGE de promedio**. La brecha frente a los BERT2BERT-style es notablemente grande (el mejor del grupo, BETOShare, promedia ~27 ROUGE). En SQAC, T5S obtiene **R1 80.68/78.80, R2 60.39/59.33, RL 80.64/78.64**; BARTO le sigue con R1 77.92/77.00. Ambos superan incluso al mBART-large. Los autores atribuyen la ventaja al objetivo autosupervisado de BART y T5, que transfiere mejor a esta tarea, y resaltan que esto "evidencia la superior capacidad de comprensión lectora" de los modelos. Un matiz honesto: SQAC y MLQA fueron diseñados como tareas *extractivas* (span-based); aquí se usan *generativamente* (el modelo genera la respuesta como texto en lugar de predecir posiciones de inicio/fin), y los resultados muestran que funcionan bien como benchmark de QA generativo aunque no fueran concebidos para eso.

**Split-and-rephrase (Tabla 4).** T5S gana en ambas métricas: **56.37 SARI y 43.27 BLEU** de promedio. BARTO queda segundo en BLEU pero flojo en SARI. El éxito de T5S se atribuye a que genera secuencias que conservan el orden de palabras de la entrada (lo que SARI premia) y a que su objetivo de span-filling facilita el "partir" oraciones.

**Diálogo (Tabla 5).** Aquí cambia el patrón: **BARTO lidera con 34.30 F1**, seguido de cerca por BETOShare (32.45 F1); en METEOR, BETOShare gana (27.33) y BARTO va segundo (26.95). T5S queda cuarto (28.80 F1, 22.24 METEOR). La explicación: T5S brilla cuando hay alto solapamiento entrada-salida (summarization, split-and-rephrase), pero el diálogo tiene bajo solapamiento, por lo que sufre. Los multilingües se comportan parecido a los monolingües, con mBART por encima de mT5, sugiriendo que el conocimiento más amplio del multilingüe ayuda en diálogo.

**Traducción (Tabla 6).** BARTO y T5S rinden similar entre sí y son competitivos con los multilingües. En Fapesp-v2 PT→ES y WMT13 EN→ES, T5S y BARTO superan claramente al mejor BERT2BERT (mBERT2mBERT). El hallazgo notable: aunque se entrenaron *casi exclusivamente* en español (umbral fastText del 98%), traducen bien **tanto generando español como generando portugués/inglés**. La hipótesis: ese 2% de texto no-español filtrado dentro del corpus, más las raíces lingüísticas compartidas con el portugués (se detectan diacríticos portugueses ç, ã, ü en el tokenizer), bastan para dotar a los modelos de conocimiento translingüístico utilizable. Es decir, un seq2seq preentrenado *desde cero en español con un mínimo de otra lengua* puede abordar traducción de forma efectiva.

### 5.2 Tareas discriminativas (Tabla 7)

Para situar las debilidades, se evalúan tareas de clasificación del benchmark **GLUES** (Cañete et al., 2020): MLDoc (clasificación de documentos largos), PAWS-X (paráfrasis), XNLI (inferencia), STS-es y SemRel2024 (similitud semántica), y SQAC en formato de clasificación a nivel de token (span-based clásico). Aquí los baselines son los encoder-only puros BETO y RoBERTa-BNE.

- En **MLDoc**, BARTO y T5S *superan* a los encoder-only, gracias a su capacidad para secuencias largas (BARTO 96.70/96.20, T5S 96.90/96.63 accuracy).
- En **PAWS-X y XNLI**, son competitivos pero no siempre ganan.
- En **STS-es y SemRel2024** (similitud semántica), los encoder-only ganan claramente, por su especialización en representación de oraciones — replicando el comportamiento conocido de BART/T5 en inglés.
- En **SQAC token-level (span-based)**, los encoder-only tienen ligera ventaja, y **T5S muestra el peor desempeño** (F1 58.46/62.29 vs ~79-80 de BETO/RoBERTa/BARTO). T5S opera aquí en formato "text-to-text" generativo, y la métrica F1 span-based no captura bien su naturaleza generativa — es una discrepancia de *medición*, no de capacidad: la sección 5.1 ya mostró que T5S genera respuestas excelentes.

Este contraste es pedagógicamente valioso: **el mismo modelo (T5S) es el mejor en QA generativo y el peor en QA extractivo medido con F1 span-based**, lo que ilustra que la formulación de la tarea y la métrica determinan tanto como el modelo.

### 5.3 Análisis cualitativo (Tabla 8)

El paper inspecciona predicciones reales. En QA generativo sobre SQAC, ante la pregunta "¿Cuál es el anillo más interno y grueso de los anillos de Júpiter?" (target: "halo"), T5S responde "El anillo halo" y BARTO "anillo halo" (ambos correctos y más completos que el target), mientras RoBERTaShare degenera en "el anillo halo: Iris (Aright (Award (Aver" — un colapso típico de un decoder con cross-attention mal inicializada. En summarization, BARTO se ciñe al target; T5S a veces *alucina* ("Escribe tu ensayo en voz alta", ausente del original), lo que explica la ligera superioridad de BARTO en esa tarea. En traducción, mBERT2mBERT alucina "mejorando la información". El veredicto cualitativo: BARTO y T5S generan respuestas naturales, coherentes y contextualmente relevantes, con un dominio sólido del español.

## 6. Limitaciones reconocidas

El paper incluye un *Ethics Statement and Limitations* explícito y honesto:

- **Modelos base, no grandes.** Son arquitecturas *base*; pueden no servir para tareas que exijan capacidades emergentes. La tendencia hacia modelos grandes (que mejoran con escala) marca el trabajo futuro: preentrenar arquitecturas mayores, tipo Llama (Touvron et al., 2023), para chatbots.
- **QA generativo evaluado con datasets extractivos.** Al usar SQAC/MLQA (span-based) de forma generativa, el modelo tiende a *reproducir información exacta* del texto fuente en vez de parafrasear genuinamente. No existe aún un dataset de QA *verdaderamente abstractivo* en español; crearlo y evaluarlo es trabajo pendiente. (El propio notebook del lab abre con un ejemplo de Quito que ilustra justamente esa aspiración abstractiva.)
- **Riesgos de uso.** Como cualquier LM, podrían usarse para construir sistemas maliciosos o sesgados; los datos no implican violación de privacidad.
- **Comparación monolingüe vs multilingüe incompleta.** Una comparación sistemática (al estilo Agerri y Agirre, 2023) ofrecería más claridad sobre fortalezas y límites.

## 7. Impacto y relevancia

El impacto del paper es de *infraestructura*: pone a disposición de toda la comunidad hispanohablante dos modelos seq2seq abiertos y competitivos que antes no existían, junto con un benchmark curado y reproducible (scripts en GitHub, modelos en Hugging Face Hub). Antes de este trabajo, quien quisiera hacer summarization, QA generativo o traducción en español tenía dos opciones malas: usar un multilingüe (que reparte capacidad entre cien idiomas) o reciclar un BERT en un BERT2BERT (con cross-attention al azar y rendimiento inferior). BARTO y T5S cierran ese hueco con modelos nativos que, siendo *base*, superan incluso a un mBART-large en la mayoría de tareas generativas.

La relevancia para el curso es directa: la Clase 24 trata Question Answering, y este paper provee el camino *generativo* del QA en español, complementando el camino *extractivo* (SQuAD/BERT) que vimos con Rajpurkar et al. (2016). Para un practicante en contextos clínicos como Roberto, la lección transferible es doble: (1) para lenguas distintas del inglés, un modelo monolingüe entrenado desde cero suele batir al multilingüe pese a tener menos parámetros — relevante si alguna vez se evalúa NLP sobre texto clínico en español; y (2) la elección entre formulación extractiva y generativa, y entre métrica span-based y ROUGE, cambia drásticamente qué modelo "gana", lo que obliga a alinear métrica con el objetivo real del sistema.

## 8. Conexión con el Laboratorio 24 Parte 2

El Laboratorio 24 tiene dos partes que encarnan los dos paradigmas de QA, y la **Parte 2 usa directamente los modelos de este paper**:

| | **Parte 1 — Extractivo** | **Parte 2 — Generativo** |
|---|---|---|
| Notebook | `QA_BERT_Spanish.ipynb` | `QA_EncoderDecoder_Spanish.ipynb` |
| Paradigma | Extractive QA: la respuesta es un *span* del contexto | Generative QA: el modelo *genera* la respuesta token a token |
| Arquitectura | Encoder-only (BERT/BETO) | Encoder-decoder (BARTO, T5S) — este paper |
| Salida del modelo | Logits de posición de inicio y fin del span | Secuencia generada autorregresivamente por el decoder |
| Cabeza | `AutoModelForQuestionAnswering` | `AutoModelForSeq2SeqLM` |
| Dataset | SQuAD-es / SQAC | SQAC (`avacaondata/sqac_fixed`) |
| Métrica | Exact Match, F1 span-based | ROUGE |

La Parte 2 del lab (cuyo autor es el propio Vladimir Araujo) clona el repositorio **`Seq2Seq-Spanish-PLMs`** y entrena con el script **`scripts/generativeqa/run_generativeqa.py`**, invocándolo sobre el dataset SQAC. Los model IDs concretos son **`vgaraujov/t5-base-spanish`** (T5S) y **`vgaraujov/bart-base-spanish`** (BARTO), exactamente los modelos de este paper. El comando pasa `--context_column context --question_column question --answer_column answers`, `--max_source_length 480`, `--max_target_length 32`, `--predict_with_generate` — es decir, la entrada es la **concatenación de pregunta y contexto** (siguiendo el fine-tuning de BART de Lewis et al., 2020a) y la salida es la respuesta *generada*, con `predict_with_generate` activando la decodificación autorregresiva. Para predicción rápida sin entrenar, el notebook usa checkpoints ya fine-tuneados (`mrm8488/spanish-t5-small-sqac-for-qa`, `vgaraujov/bart-base-spanish-sqac`).

El contraste pedagógico es el corazón de la clase. En la **Parte 1**, sobre el mismo pasaje, BETO localiza el span "halo" dentro del texto prediciendo dos índices (inicio y fin); la respuesta *está* literalmente en el contexto y el modelo solo la *señala*. En la **Parte 2**, T5S/BARTO *escriben* la respuesta ("El anillo halo") condicionados en pregunta+contexto; el decoder puede reformular, expandir o sintetizar, y por eso responde con frases más naturales que el span crudo del target. Extracción = *seleccionar* un fragmento existente (rápido, restringido a lo que está en el texto, evaluable con EM/F1 exactos). Generación = *producir* texto nuevo (flexible, capaz de parafrasear, pero susceptible de alucinar y evaluable solo con métricas de solapamiento como ROUGE). Que SQAC fuera diseñado como dataset extractivo y aquí se use generativamente es justamente lo que conecta ambas partes del lab y las dos lecturas de la Clase 24.

## 9. Notas y enlaces

- **Paper:** Araujo, V., Trusca, M. M., Tufiño, R., Moens, M.-F. (2024). *Sequence-to-Sequence Spanish Pre-trained Language Models.* LREC-COLING 2024. arXiv:2309.11259.
- **Repositorio + modelos:** https://github.com/vgaraujov/Seq2Seq-Spanish-PLMs — checkpoints `vgaraujov/bart-base-spanish` (BARTO) y `vgaraujov/t5-base-spanish` (T5S) en Hugging Face Hub.
- **Modelos base de referencia (inglés):** BART (Lewis et al., 2020a, denoising autoencoder seq2seq), T5 (Raffel et al., 2020, text-to-text con span corruption), MASS (Song et al., 2019), BERT2BERT (Rothe et al., 2020).
- **Modelos previos en español (encoder/decoder-only):** BETO y ALBETO (Cañete et al., 2020, 2022), BERTIN (De la Rosa et al., 2022), MarIA con RoBERTa-BNE y GPT-2 (Gutiérrez-Fandiño et al., 2022), RigoBERTa (Serrano et al., 2022).
- **Corpus de preentrenamiento:** OSCAR 21.09 (Abadji et al., 2022), mC4-es (Xue et al., 2021; muestreo De la Rosa et al., 2022), SUC (Cañete et al., 2020).
- **Datasets de evaluación:** MLSUM, WikiLingua, XL-Sum, EUR-Lex-Sum (summarization); BiSECT (split-and-rephrase); **SQAC** (Gutiérrez-Fandiño et al., 2022) y MLQA (Lewis et al., 2020b) para QA; MIAM (diálogo); Fapesp-v2 y WMT13 (traducción); GLUES, STS-es, SemRel2024 (discriminativas).
- **Métricas usadas:** ROUGE (Lin, 2004), SARI (Xu et al., 2016), BLEU (Post, 2018), METEOR (Banerjee y Lavie, 2005), F1, Exact Match.
- **Baselines multilingües:** mT5-base (Xue et al., 2021), mBART-50 / mBART-large (Tang et al., 2020).
- **Conexión con la Clase 24:** Parte 1 del lab = extractivo (BETO sobre SQAC, cabeza de span); Parte 2 = generativo (BARTO/T5S sobre SQAC vía `run_generativeqa.py`). Complementa el paper extractivo fundacional Rajpurkar-SQuAD-2016.
