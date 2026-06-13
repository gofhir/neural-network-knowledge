---
title: "P2 — Actividad y comparación de paradigmas"
weight: 6
---

> **Celdas 30-35 del notebook (Parte 2).** El experimento propio que cierra el lab: el **mismo contexto FHIR** de la Parte 1, ahora con el modelo generativo, para enfrentar los dos paradigmas cara a cara. Y las tres Verdadero/Falso conceptuales.

## A. Experimento propio (FHIR): la comparación de paradigmas

Esta es la actividad central y el cierre del laboratorio. Para comparar ambos paradigmas de forma **controlada**, corrí exactamente el mismo contexto de dominio técnico (HL7 FHIR) que en la [Parte 1 con BERT extractivo](../lab-24/), ahora con el modelo **generativo T5S** (`mrm8488/spanish-t5-small-sqac-for-qa`). Si cambiara el contexto, no sabría si las diferencias vienen del modelo o del texto; al fijar el contexto, lo único que varía es el paradigma.

```python
context = ("El estándar HL7 FHIR define recursos para representar información clínica. "
           "El recurso Patient almacena datos demográficos del paciente, mientras que "
           "el recurso Observation registra mediciones como signos vitales o resultados "
           "de laboratorio. FHIR fue publicado por primera vez en 2014 por la "
           "organización Health Level Seven International.")

questions = ["¿Qué recurso almacena los datos demográficos del paciente?",  # respondible directa
             "¿Quién publicó FHIR?",                                        # respondible (agente)
             "¿Qué registra el recurso Observation?",                       # reformulación esperada
             "¿Cuál es el recurso para agendar citas médicas?"]             # TRAMPA: no está en el contexto
```

Las cuatro preguntas están diseñadas a propósito: una extracción de entidad precisa, una con respuesta clara, una que invita a reformular, y una **pregunta-trampa** cuya respuesta no aparece en el contexto (para medir abstención vs. alucinación).

### Resultados reales del generativo (T5S)

| Pregunta | Respuesta del modelo | Veredicto |
|----------|----------------------|-----------|
| ¿Qué recurso almacena los datos demográficos del paciente? | `el estándar hl7 fhir` | ❌ Incorrecto (esperado: Patient) |
| ¿Quién publicó FHIR? | `la organización health level seven international` | ✅ Correcto |
| ¿Qué registra el recurso Observation? | `mediciones como signos vitales o resultados de laboratorio` | ✅ Correcto |
| ¿Cuál es el recurso para agendar citas médicas? *(trampa)* | `hl7 fhir` | ❌ Alucinación (no abstención) |

### Comparación directa con la Parte 1 (mismo contexto)

Esta tabla es el corazón del experimento: **misma entrada, dos paradigmas**.

| Pregunta | BERT extractivo (P1) | T5 generativo (P2) |
|----------|----------------------|--------------------|
| ¿Qué recurso almacena los datos demográficos? | `Patient` ✅ | `el estándar hl7 fhir` ❌ |
| ¿Quién publicó FHIR? | `organización Health Level Seven International` ✅ | `la organización health level seven international` ✅ |
| Pregunta-trampa sin respuesta | `empty` (se abstuvo) ✅ | alucina (`hl7 fhir`) ❌ |

### Hallazgos

**1. El generativo no es estrictamente mejor: falló en extracción de entidad precisa.**
Ante "¿qué recurso almacena los datos demográficos?", el extractivo devolvió el token exacto `Patient`, mientras que el generativo derivó hacia la frase más saliente y más "en español" del contexto (`el estándar hl7 fhir`), **errando la granularidad**. La razón es estructural: el extractivo solo debe *señalar* una posición de inicio y fin en el texto; el generativo debe *producir* la palabra token a token, y al hacerlo tiende a lo fluido y frecuente en vez del término técnico preciso. Donde el extractivo apunta con el dedo, el generativo redacta — y al redactar, suaviza.

**2. El sesgo del idioma de entrenamiento aflora.**
El T5S fue pre-entrenado y fine-tuneado en español nativo (SQAC), por lo que es **frágil con términos técnicos en inglés** ("Patient", "Observation") incrustados en texto español. Aquí surge una paradoja interesante: el modelo entrenado con datos nativos en español maneja *peor* el vocabulario técnico en inglés que el extractivo BETO, porque BETO no necesita "entender" `Patient` para copiarlo — basta con localizarlo. El generativo, en cambio, tiene que reconstruir esa palabra rara desde su vocabulario, y prefiere la salida española más probable.

**3. La firma generativa: reformulación fluida donde acierta.**
En las preguntas que respondió bien, el modelo generó frases naturales y bien construidas: "la organización health level seven international" (Q2) y la frase completa "mediciones como signos vitales o resultados de laboratorio" (Q3). Esta fluidez es justamente la ventaja del paradigma generativo: produce lenguaje, no un recorte literal.

**4. Sin abstención → alucinación.**
La pregunta-trampa es la prueba más reveladora. El extractivo respondió `empty` (se abstuvo correctamente: SQuAD v2 lo entrenó con preguntas sin respuesta). El generativo, en cambio, **alucinó** `hl7 fhir`: inventó una respuesta plausible aunque el contexto no la contiene. SQAC no incluye preguntas *unanswerable*, así que el T5S nunca aprendió a callarse. Genera siempre.

### Conclusión: ningún paradigma domina; perfiles de error opuestos

El experimento no produce un ganador, produce un **mapa de fortalezas y debilidades complementarias**:

| Dimensión | Extractivo (BERT) | Generativo (T5) |
|-----------|-------------------|-----------------|
| Precisión de entidad | Alta (token exacto) | Baja (suaviza la granularidad) |
| Vocabulario en otro idioma | Robusto (solo copia) | Frágil (debe reconstruirlo) |
| Abstención | Sabe decir "no sé" | Alucina |
| Fluidez / reformulación | Rígido (copia literal) | Flexible y natural |

Para dominios donde la **trazabilidad** es innegociable —clínico/FHIR, legal, financiero— el anclaje al texto y la capacidad de abstención del extractivo son ventajas decisivas: toda respuesta es un span verificable del documento fuente. El generativo gana cuando se necesita **reformular o sintetizar** y el riesgo de alucinación es tolerable o se puede mitigar (por ejemplo con [RAG](../../fundamentos/dense-retrieval/), que ancla la generación en pasajes recuperados, o con técnicas de *grounding*).

Esta tensión es exactamente la que aparece en el [lab-23 (BLIP)](../../laboratorios/lab-23/) entre VQA y captioning, y la base teórica está en [Question Answering](../../fundamentos/question-answering/).

## B. Verdadero / Falso (celdas 33-35)

Las tres afirmaciones con respuesta y justificación.

| # | Afirmación | Respuesta | Justificación |
|---|------------|-----------|---------------|
| 34 | *Experts annotated the Spanish SQuAD v2 dataset* | **False** | El SQuAD v2 en español fue **traducido automáticamente** del inglés y realineado, no anotado por expertos. El dataset nativo en español anotado por humanos es **SQAC** (Spanish Question Answering Corpus). |
| 35 | *BART and BERT use exactly the same underlying transformer architecture* | **False** | BERT es **encoder-only**; BART es **encoder-decoder**. Ambos usan bloques Transformer, pero no la misma arquitectura: BART añade un decoder autorregresivo que BERT no tiene. |
| 36 | *Encoder-decoder models are pre-trained to reconstruct the input text* | **True** | BART es un **denoising autoencoder** (text infilling: enmascara spans y los reconstruye); T5 usa **span corruption**. En ambos casos el pre-entrenamiento consiste en reconstruir texto corrompido. |

La clave transversal de las tres: distinguir **cómo se construye el dato** (traducción vs. anotación nativa), **qué forma tiene la arquitectura** (encoder-only vs. encoder-decoder) y **qué objetivo de pre-entrenamiento** caracteriza a cada familia (reconstrucción/denoising en los seq2seq).

---

**Anterior:** [Inferencia generativa](inferencia-generativa) · [Volver al lab](../)
