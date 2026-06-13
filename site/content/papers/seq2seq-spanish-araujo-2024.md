---
title: "Seq2Seq Spanish PLMs: BARTO y T5S (2024)"
weight: 122
math: true
---

{{< paper-card
    title="Sequence-to-Sequence Spanish Pre-trained Language Models"
    authors="Araujo, Trusca, Tufiño, Moens"
    year="2024"
    venue="LREC-COLING 2024 / arXiv"
    pdf="/papers/seq2seq-spanish-araujo-2024.pdf"
    arxiv="2309.11259" >}}
El paper que llena un hueco concreto del ecosistema NLP en español: para 2023 el español ya tenia modelos **encoder-only** solidos (BETO) y **decoder-only** decentes (GPT-2 de MarIA), pero **no tenia modelos encoder-decoder (seq2seq) preentrenados de forma nativa**. En ingles esa familia llevaba madura desde 2019-2020 con BART y T5; en español el unico recurso eran los multilingues (mBART, mT5), que reparten su capacidad entre cien idiomas. Los autores preentrenan **desde cero** dos modelos sobre ~120 GB de texto español con tokenizers propios: **BARTO** (BART en español, denoising / text infilling) y **T5S** (T5 en español, span corruption), mas una familia de baselines BERT2BERT-style. Sobre un benchmark curado de seis tareas seq2seq en español, BARTO y T5S superan tanto a los baselines monolingues como a mBART/mT5 -- pese a ser modelos *base* y mBART-large ser *grande*. Es un paper de *recursos linguisticos* (de ahi su hogar en LREC): su contribucion no es una arquitectura nueva sino infraestructura abierta y reproducible para la comunidad hispanohablante. El primer autor, **Vladimir Araujo**, es ademas el profesor de la Clase 24 del curso y el autor del notebook del Laboratorio 24 Parte 2 -- que usa exactamente estos modelos.
{{< /paper-card >}}

---

## El hueco que el paper llena

La historia del NLP en español venia corriendo en paralelo a la del ingles, pero con una asimetria de cobertura arquitectonica. El paper situa con precision las tres familias del Transformer (Vaswani et al., 2017):

**(i) Encoder-only** -- modelos tipo BERT, especializados en *comprension*. En español: BETO (Cañete et al., 2020), ALBETO/DistilBETO, BERTIN, MarIA (RoBERTa-BNE) y RigoBERTa. Familia madura.

**(ii) Decoder-only** -- modelos tipo GPT, especializados en *generacion* autorregresiva. En español el GPT-2 de MarIA es el representante principal.

**(iii) Encoder-decoder (seq2seq)** -- mapean una secuencia de entrada a una de salida de longitud distinta (Sutskever et al., 2014): el encoder procesa la entrada completa en paralelo, el decoder genera la salida autorregresivamente atendiendo al encoder via *cross-attention*. Son la herramienta natural para summarization, traduccion, dialogo, split-and-rephrase y QA generativo. En ingles esta familia estaba madura con MASS (2019), **BART** (Lewis et al., 2020) y **T5** (Raffel et al., 2020), y otras lenguas ya tenian sus versiones (BARThez frances, IndicBART, AraBART, PTT5 portugues, IT5 italiano).

El español era la excepcion notable: a pesar de ser una de las lenguas mas habladas del mundo, **no tenia un BART ni un T5 entrenados exclusivamente sobre corpus español**. El patron consolidado de los BERT monolingues (CamemBERT, RobBERT, FinBERT) muestra que la version especifica por idioma supera a la multilingue, lo que justifica el esfuerzo de entrenar desde cero. La tesis de motivacion es directa y de servicio: **democratizar los modelos seq2seq para la comunidad NLP hispanohablante**.

---

## Contribucion central -- BARTO y T5S

El nucleo del paper son dos modelos preentrenados desde cero sobre corpus español, mas baselines.

**BARTO** -- version española de BART-base: encoder y decoder de **6 capas cada uno**, 12 cabezas, 768 dimensiones. El nombre es un juego: BART + O (terminacion masculina del español, igual que BETO).

**T5S** -- version española de la variante **T5.1.1-base**: encoder y decoder de **12 capas cada uno**, 12 cabezas, 768 dimensiones. T5.1.1 introduce GEGLU en lugar de ReLU, no comparte parametros entre embeddings y capa de salida, y no usa dropout en el preentrenamiento.

**BERT2BERT-style** -- baselines monolingues sin preentrenamiento seq2seq propio, siguiendo Rothe et al. (2020): se inicializa un encoder-decoder a partir de checkpoints encoder-only ya existentes (BETO, RoBERTa-BNE), pero las capas de cross-attention del decoder se inicializan al azar (BERT no las tiene). Se construyen BETO2BETO, BETOShare, RoBERTa2RoBERTa y RoBERTaShare. Las variantes "Share" comparten parametros entre encoder y decoder.

### BART vs T5: el corazon conceptual

Esta es la diferencia fundamental entre los dos modelos:

- **BART** es esencialmente BERT (encoder bidireccional) + GPT (decoder autorregresivo) pegados y entrenados como un **denoising autoencoder**: se corrompe el documento de entrada y el modelo aprende a regenerar el documento *completo*. BARTO usa la receta de Lewis et al. (2020): **text infilling** (enmascara el **30%** de los tokens, reemplazando spans contiguos por un unico `[MASK]`, de modo que el modelo debe inferir tanto el contenido como *cuantos* tokens faltaban) mas **sentence permutation**. El decoder ve la salida entera, lo que refuerza su modelado de lenguaje generativo.
- **T5** trata *toda* tarea como text-to-text: entrada de texto → salida de texto, incluso para clasificacion. Se preentrena con **span corruption**: se corrompe el **15%** de los tokens (longitud media de span 3), cada span eliminado se reemplaza por un token centinela (`<extra_id_0>`, …) y el objetivo es generar *solo* los spans faltantes, no el documento completo. Es fuerte cuando hay alto solapamiento entre entrada y salida.

La adaptacion al español no cambio las arquitecturas: cambio los **datos** (corpus español) y el **tokenizer** (vocabulario nativo). Es preentrenamiento *from scratch*, no continuacion de un checkpoint ingles -- eso distingue a BARTO/T5S de los baselines BERT2BERT (que reciclan pesos) y de los multilingues (que diluyen su capacidad).

### Datos y configuracion

El corpus combina tres fuentes: **OSCAR 21.09** (~160 GB, español deduplicado del crawl OSCAR), **mC4-es** (subconjunto español de mC4 con muestreo por perplejidad) y **SUC** (~14 GB, corpus de BETO, con su Wikipedia reemplazada por un dump actualizado). El pipeline aplica formato a nivel de documento, filtrado de calidad (clasificador fastText con umbral del 98% de confianza de ser español -- deliberadamente no perfecto, para retener un ~2% de otras lenguas), deduplicacion con `text-dedup` y correccion de encoding con `ftfy`. El corpus resultante supera los **120 GB** de texto, escala comparable a la del RoBERTa/BART en ingles.

| Hiperparametro | BARTO | T5S |
|---|---|---|
| Arquitectura | BART-base (6+6 capas) | T5.1.1-base (12+12 capas) |
| Objetivo | text infilling (30%) + sentence permutation | span corruption (15%, span medio 3) |
| Tokenizer | SentencePiece BPE, **50.264** tokens | SentencePiece unigram, **32.000** tokens |
| Pasos / hardware | 100.000 / 8× A100 | 130.000 / 4× A100 |
| Precision | FP16 | BF16 |

Cada modelo tiene su propio vocabulario subword sobre texto español, lo que evita la fragmentacion excesiva que sufren los tokenizers multilingues al codificar tildes, ñ y la morfologia rica del español.

---

## Experimentos

El paper curo un benchmark de tareas seq2seq en español. Todos los modelos se fine-tunean en un RTX 3090 con `transformers` (PyTorch). Los baselines multilingues son **mT5-base** (101 idiomas) y **mBART-large** (mBART-50; no existe mBART-base comparable, asi que el multilingue juega con ventaja de tamaño).

| Tarea | Dataset(s) | Metrica |
|---|---|---|
| Summarization abstractivo | MLSUM, WikiLingua | ROUGE-1/2/L |
| Summarization largo | XL-Sum, EUR-Lex-Sum (legal) | ROUGE |
| Split-and-rephrase | BiSECT-es | SARI, BLEU |
| **QA generativo** | **MLQA, SQAC** | ROUGE |
| Dialogo | MIAM-es | F1, METEOR |
| Traduccion | Fapesp-v2 (PT↔ES), WMT13 (EN↔ES) | BLEU |

**Resultado general:** BARTO y T5S son los mejores en practicamente todas las tareas generativas, superando a los BERT2BERT-style monolingues y a los multilingues mBART/mT5 -- pese a ser *base* y mBART-large ser *grande*. La excepcion sistematica es EUR-Lex-Sum (legal), donde los multilingues ganan por su exposicion previa a documentos juridicos.

**Summarization.** En MLSUM y WikiLingua, **T5S es el mejor con 26.54 ROUGE de promedio**, seguido de BARTO (25.49); el mejor BERT2BERT (BETO2BETO) promedia 24.39.

**QA generativo (la tarea mas relevante para la Clase 24).** En SQAC y MLQA, BARTO y T5S dominan ampliamente, con **T5S como el mejor: 67.92 ROUGE de promedio**. En SQAC, T5S obtiene **R1 80.68/78.80, R2 60.39/59.33, RL 80.64/78.64**; BARTO le sigue (R1 77.92/77.00). La brecha frente al mejor BERT2BERT (BETOShare, ~27 ROUGE) es enorme, y ambos superan incluso a mBART-large. Los autores atribuyen la ventaja al objetivo autosupervisado de BART/T5, que transfiere mejor a esta tarea. Matiz honesto: SQAC y MLQA fueron diseñados como tareas *extractivas* (span-based); aqui se usan *generativamente* (el modelo genera la respuesta como texto en vez de predecir posiciones inicio/fin), y funcionan bien aunque no fueran concebidos para eso.

**Split-and-rephrase.** T5S gana en ambas metricas (56.37 SARI, 43.27 BLEU), porque conserva el orden de palabras de la entrada (lo que SARI premia) y su objetivo de span-filling facilita "partir" oraciones.

**Dialogo.** Aqui cambia el patron: **BARTO lidera (34.30 F1)**, seguido de BETOShare; T5S queda cuarto. La razon: T5S brilla con alto solapamiento entrada-salida (summarization, split-and-rephrase), pero el dialogo tiene bajo solapamiento, asi que sufre.

**Traduccion.** Hallazgo notable: aunque se entrenaron *casi exclusivamente* en español, BARTO y T5S traducen bien tanto generando español como portugues/ingles. La hipotesis: ese ~2% de texto no-español filtrado, mas las raices compartidas con el portugues (diacriticos ç, ã detectados en el tokenizer), bastan para dotar a los modelos de conocimiento translinguistico utilizable.

**Tareas discriminativas (clasificacion).** En MLDoc (documentos largos) BARTO y T5S superan a los encoder-only puros, pero en similitud semantica (STS-es) los encoder-only ganan, replicando el comportamiento conocido de BART/T5 en ingles. Contraste pedagogicamente valioso: **el mismo T5S es el mejor en QA generativo y el peor en QA extractivo medido con F1 span-based** (F1 ~58-62 vs ~79-80 de BETO) -- una discrepancia de *medicion*, no de capacidad. La formulacion de la tarea y la metrica determinan tanto como el modelo.

---

## Limitaciones reconocidas

El paper incluye un *Ethics Statement and Limitations* explicito:

- **Modelos base, no grandes.** Son arquitecturas *base*; pueden no servir para tareas que exijan capacidades emergentes. El trabajo futuro apunta a preentrenar arquitecturas mayores tipo Llama para chatbots.
- **QA generativo evaluado con datasets extractivos.** Al usar SQAC/MLQA (span-based) de forma generativa, el modelo tiende a *reproducir informacion exacta* del texto en vez de parafrasear genuinamente. No existe aun un dataset de QA *verdaderamente abstractivo* en español; crearlo es trabajo pendiente.
- **Comparacion monolingue vs multilingue incompleta.** Una comparacion mas sistematica ofreceria mas claridad sobre fortalezas y limites.

---

## Por que importa hoy

El impacto es de *infraestructura*: pone a disposicion de toda la comunidad hispanohablante dos modelos seq2seq abiertos y competitivos que antes no existian, junto con un benchmark curado y reproducible (scripts en GitHub, modelos en Hugging Face Hub). Antes de este trabajo, quien quisiera hacer summarization, QA generativo o traduccion en español tenia dos opciones malas: usar un multilingue (que reparte capacidad entre cien idiomas) o reciclar un BERT en un BERT2BERT (con cross-attention al azar). BARTO y T5S cierran ese hueco con modelos nativos que, siendo *base*, superan incluso a mBART-large en la mayoria de tareas generativas.

La leccion transferible para sistemas reales es doble: (1) para lenguas distintas del ingles, un modelo monolingue entrenado desde cero suele batir al multilingue pese a tener menos parametros -- relevante si alguna vez se evalua NLP sobre texto clinico en español; y (2) la eleccion entre formulacion extractiva y generativa, y entre metrica span-based y ROUGE, cambia drasticamente que modelo "gana", lo que obliga a alinear la metrica con el objetivo real del sistema.

---

## Conexion con el Laboratorio 24 Parte 2

El [Laboratorio 24](/laboratorios/lab-24) tiene dos partes que encarnan los dos paradigmas de QA, y la **Parte 2 usa directamente los modelos de este paper**.

| | **Parte 1 -- Extractivo** | **Parte 2 -- Generativo** |
|---|---|---|
| Paradigma | la respuesta es un *span* del contexto | el modelo *genera* la respuesta token a token |
| Arquitectura | Encoder-only (BERT/BETO) | Encoder-decoder (BARTO, T5S) -- este paper |
| Cabeza | `AutoModelForQuestionAnswering` | `AutoModelForSeq2SeqLM` |
| Dataset | SQuAD-es / SQAC | SQAC (`avacaondata/sqac_fixed`) |
| Metrica | Exact Match, F1 span-based | ROUGE |

La Parte 2 (cuyo autor es el propio Vladimir Araujo, profesor de la clase) clona el repositorio **`Seq2Seq-Spanish-PLMs`** y entrena con el script **`scripts/generativeqa/run_generativeqa.py`** sobre SQAC. Los model IDs son **`vgaraujov/t5-base-spanish`** (T5S) y **`vgaraujov/bart-base-spanish`** (BARTO), exactamente los modelos de este paper. El comando pasa `--max_source_length 480 --max_target_length 32 --predict_with_generate`: la entrada es la **concatenacion de pregunta y contexto** (siguiendo el fine-tuning de BART de Lewis et al., 2020) y la salida es la respuesta *generada* autorregresivamente.

El contraste pedagogico es el corazon de la clase. En la **Parte 1**, sobre el mismo pasaje, BETO localiza el span "halo" prediciendo dos indices (inicio y fin): la respuesta *esta* literalmente en el contexto y el modelo solo la *señala*. En la **Parte 2**, T5S/BARTO *escriben* la respuesta ("El anillo halo") condicionados en pregunta+contexto; el decoder puede reformular, expandir o sintetizar. Extraccion = *seleccionar* un fragmento existente (rapido, restringido a lo que esta en el texto, evaluable con EM/F1 exactos). Generacion = *producir* texto nuevo (flexible, capaz de parafrasear, pero susceptible de alucinar y evaluable solo con metricas de solapamiento como ROUGE). Este camino *generativo* del QA en español complementa el camino *extractivo* que vimos con [SQuAD (Rajpurkar et al., 2016)](/papers/squad-rajpurkar-2016).

---

## Notas y enlaces

- **Paper:** Araujo, V., Trusca, M. M., Tufiño, R., Moens, M.-F. (2024). *Sequence-to-Sequence Spanish Pre-trained Language Models.* LREC-COLING 2024. arXiv:2309.11259.
- **Repositorio + modelos:** https://github.com/vgaraujov/Seq2Seq-Spanish-PLMs -- checkpoints `vgaraujov/bart-base-spanish` (BARTO) y `vgaraujov/t5-base-spanish` (T5S) en Hugging Face Hub.
- **Modelos base de referencia (ingles):** BART (Lewis et al., 2020, denoising autoencoder seq2seq), [T5 (Raffel et al., 2020, text-to-text con span corruption)](/papers/t5-raffel-2020), MASS (Song et al., 2019), BERT2BERT (Rothe et al., 2020).
- **Modelos previos en español:** BETO y ALBETO (Cañete et al., 2020, 2022), BERTIN, MarIA (RoBERTa-BNE y GPT-2), RigoBERTa.
- **Datasets de evaluacion:** MLSUM, WikiLingua, XL-Sum, EUR-Lex-Sum (summarization); BiSECT (split-and-rephrase); **SQAC** y MLQA (QA); MIAM (dialogo); Fapesp-v2 y WMT13 (traduccion).
- **Baselines multilingues:** mT5-base (Xue et al., 2021), mBART-50 / mBART-large (Tang et al., 2020).

Ver fundamentos: [Question Answering](/fundamentos/question-answering) - [Machine Reading Comprehension](/fundamentos/machine-reading-comprehension) - [T5 y la arquitectura encoder-decoder](/fundamentos/t5-encoder-decoder).

Ver papers: [SQuAD (Rajpurkar 2016)](/papers/squad-rajpurkar-2016) - [T5 (Raffel 2020)](/papers/t5-raffel-2020).

Ver clase: [Clase 24 -- Question Answering y Machine Reading Comprehension](/clases/clase-24) - [Laboratorio 24](/laboratorios/lab-24) - [Dominio Texto](/dominios/texto).
