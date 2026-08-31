---
title: "BETO: Spanish Pre-Trained BERT Model and Evaluation Data (2020)"
weight: 298
math: true
---

{{< paper-card
    title="Spanish Pre-Trained BERT Model and Evaluation Data"
    authors="José Cañete, Gabriel Chaperon, Rodrigo Fuentes, Jou-Hui Ho, Hojin Kang, Jorge Pérez (DCC y EE, Universidad de Chile / IMFD)"
    year="2020"
    venue="PML4DC @ ICLR 2020"
    pdf="/papers/beto-canete-2020.pdf" >}}
El BERT en español que el [Laboratorio 20](/laboratorios/lab-20) usa en su segunda actividad, y probablemente el modelo de lenguaje más usado del ecosistema hispanohablante entre 2020 y 2023. No propone arquitectura nueva —y esa es una decisión deliberada—: el aporte está en el **corpus**, el **vocabulario dedicado**, el **régimen de entrenamiento** y **GLUES**, un benchmark de evaluación en español. Producido en la Universidad de Chile con TPUs donadas por el programa TFRC de Google.
{{< /paper-card >}}

---

## El vacío que llena

Entre 2018 y 2020, quien quisiera un modelo pre-entrenado para español tenía dos opciones malas: **mBERT** —multilingüe, con un vocabulario de 119.547 subwords repartido entre 104 idiomas— o traducir el texto al inglés y usar BERT. La primera reparte capacidad entre idiomas que compiten; la segunda pierde en la traducción.

BETO es la tercera opción: mismo BERT-base, entrenado solo en español, con un vocabulario que solo tiene que cubrir un idioma.

## Tres precisiones sobre el paper

Este es el tipo de paper donde la lectura atenta corrige lo que circula informalmente.

{{< concept-alert type="atencion" >}}
**1 · La configuración declarada no cuadra.** El paper dice (§3): *"12 capas de auto-atención con **16 cabezas** cada una, usando **1024** como hidden size. En total 110M de parámetros."*

Esos números no corresponden a BERT-base (12 capas, 768, 12 cabezas) ni a BERT-large (24, 1024, 16), y **no cierran aritméticamente**: 12 capas con hidden 1024 darían ~160M de parámetros, no 110M.

El `config.json` publicado en HuggingFace dice **12 capas, hidden 768, 12 cabezas, FFN 3072, 109M** — o sea BERT-base exacto. Lo más probable es un error de escritura al comparar notas con BERT-large. **Para implementar, vale el config.json, no el paper.**

**2 · El tokenizador es SentencePiece BPE, no WordPiece.** §3: *"construimos un vocabulario de 31K subwords usando el algoritmo **byte pair encoding** provisto por la librería **SentencePiece**"*. BERT original usa WordPiece, que es un algoritmo distinto (maximiza la verosimilitud de un LM unigrama en lugar de fusionar el par más frecuente).

El matiz práctico: los modelos publicados se cargan con `BertTokenizer`, que usa la convención WordPiece con `##` como marcador de continuación. El vocabulario producido por SentencePiece fue convertido al formato WordPiece antes del release, para mantener compatibilidad *drop-in* con `BertModel`.

**3 · Son seis autores, no dos.** José Cañete, Gabriel Chaperon, Rodrigo Fuentes, Jou-Hui Ho, Hojin Kang y Jorge Pérez — dos de ellos estudiantes de pregrado de Ingeniería Eléctrica. Las referencias informales suelen citar solo a "Cañete y Pérez", omitiendo a los cuatro del medio.
{{< /concept-alert >}}

## Por qué no innovar en arquitectura

BETO replica BERT-base sin modificaciones: mismo encoder, mismo MLM + NSP, mismas convenciones de tokens especiales. Es una decisión defendible y explícita — el objetivo era un reemplazo **directo** de mBERT en cualquier pipeline existente:

```python
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
```

Cualquier modificación arquitectónica habría agregado fricción de adopción sin beneficio proporcional. El aporte está en otro lado.

## El vocabulario dedicado

| Modelo | Vocabulario | Idiomas |
|---|---:|---:|
| BERT-base inglés (uncased) | 30.522 | 1 |
| **BETO** | **31.002** (+1.002 placeholders) | **1** |
| mBERT | 119.547 | 104 |
| XLM-R | 250.002 | 100 |

Un vocabulario de 31 K dedicado al español representa las palabras completas mucho más a menudo que uno de 119 K repartido entre 104 idiomas. Menos fragmentación significa secuencias más cortas, menos posiciones consumidas y mejor uso del contexto.

Los **1.002 tokens placeholder** son un detalle de diseño con visión práctica: dejan espacio para que quien haga fine-tuning agregue vocabulario de dominio —terminología clínica, entidades corporativas, notación química— **sin redimensionar la matriz de embeddings**.

## Régimen de entrenamiento

Toma prestado lo que [RoBERTa](/papers/roberta-liu-2019) había establecido meses antes —enmascarado dinámico, batches grandes— y agrega **Whole Word Masking**, la técnica de [Cui et al.](/papers/whole-word-masking-cui-2019): cuando una palabra se fragmenta en varios subwords, se enmascaran **todos** sus pedazos o ninguno.

Entrenado 2M pasos sobre TPU v3-8 *preemptible* donadas por el programa TFRC de Google — una restricción de recursos que el paper documenta, y que explica varias de sus decisiones.

## GLUES: la segunda contribución

El otro aporte, menos citado y quizá más duradero, es un benchmark de evaluación en español que hasta entonces no existía de forma consolidada:

| Tarea | Dataset | Métrica |
|---|---|---|
| XNLI | XNLI ES | Accuracy |
| PAWS-X | PAWS-X ES | Accuracy |
| NER | CoNLL-2002 ES | F1 |
| POS | Universal Dependencies v1.4 ES | Accuracy |
| MLDoc | MLDoc ES | Accuracy |
| Dependency parsing | UD v2.2 ES (AnCora + GSD) | UAS / LAS |
| QA | MLQA, XQuAD, TAR | F1 / EM |

Resultados contra el mejor mBERT:

| Modelo | XNLI | PAWS-X | NER | POS | MLDoc |
|---|---:|---:|---:|---:|---:|
| Mejor mBERT | 78,50 | 89,00 | 87,38 | 97,10 | 95,70 |
| BETO uncased | 80,15 | 89,55 | 82,67 | 98,44 | **96,12** |
| **BETO cased** | **82,01** | 89,05 | **88,43** | **98,97** | 95,60 |

El patrón vale la pena notar: **`cased` gana en NER y POS**, tareas donde la mayúscula es información (nombres propios, inicio de oración), y **`uncased` gana en clasificación de documentos**, donde no lo es.

## Limitaciones reconocibles

- **Sin comparación con modelos monolingües contemporáneos.** La evaluación es contra mBERT; no hay comparación con otras iniciativas de BERT-español de la época.
- **El corpus mezcla registros** —Wikipedia, OpenSubtitles, ParaCrawl— sin análisis del sesgo dialectal resultante. El español de España, México y Chile no están representados por igual, y el paper no lo cuantifica.
- **Los recursos limitaron el diseño experimental.** Con TPUs *preemptible* no hubo margen para ablaciones sistemáticas; varias decisiones se heredan de RoBERTa sin verificarse en español.
- **NSP se conserva** aunque RoBERTa ya había mostrado que quitarlo mejora.

## En el laboratorio

El [Lab 20](/laboratorios/lab-20) usa `dccuchile/bert-base-spanish-wwm-cased` en su segunda actividad, y una de las preguntas —*¿qué modelo usaría para BERT en español?*— se responde con este checkpoint. La respuesta interesante es la asimetría que revela: **la comunidad hispanohablante invirtió mucho más en modelos encoder-only que en decoder-only**. Para BERT en español hay opciones maduras; para GPT-2 en español las alternativas son bastante más escasas.

Un detalle operativo que aparece al usarlo: el token `[CLS]` **tiene ID 4**, no 101 como en `bert-base-uncased`. Los identificadores dependen del vocabulario, no del nombre del token.

---

**Ver también:** [BERT](/papers/bert-devlin-2018) · [RoBERTa](/papers/roberta-liu-2019) · [Whole Word Masking](/papers/whole-word-masking-cui-2019) · [SentencePiece](/papers/sentencepiece-kudo-2018) · [Clase 20](/clases/clase-20) · [Lab 20](/laboratorios/lab-20).
