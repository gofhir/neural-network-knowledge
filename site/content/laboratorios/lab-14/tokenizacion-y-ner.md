---
title: "Tokenizacion + NER con BETO"
weight: 10
math: true
---

Esta primera seccion del laboratorio establece **el setup base** para todo lo que viene: cargar un Transformer ya entrenado (BETO fine-tuned para NER en espanol), entender como tokeniza con WordPiece, ver que tokens especiales agrega al input, y observar las predicciones de NER con `displacy`. El objetivo no es entrenar nada — el modelo viene pre-entrenado y fine-tuned. **Es un lab de interpretabilidad, no de implementacion.**

La leccion clave que emerge en esta seccion: las predicciones de NER **no son propiedades intrinsecas de las palabras** sino del **rol contextual** que cumplen en la frase — y eso se manifiesta en errores pedagogicamente reveladores como `Espana → ORG`.

## Setup del modelo

El notebook carga *(parte 1, celda 12)* el modelo `mrm8488/bert-spanish-cased-finetuned-ner` desde el Hub de Hugging Face. Hay tres piezas que conviene desglosar:

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_type = 'bert'
model_version = 'mrm8488/bert-spanish-cased-finetuned-ner'

tokenizer = AutoTokenizer.from_pretrained(model_version)
model = AutoModelForTokenClassification.from_pretrained(model_version, output_attentions=True)
model.to(device)

special_tokens = tokenizer.special_tokens_map.values()
```

| Pieza | Que es |
| --- | --- |
| **BETO** | BERT base entrenado por la U. de Chile (Canete et al. 2020) sobre un corpus de espanol peninsular y latinoamericano. ~110M parametros, 12 capas, 12 cabezas |
| **`mrm8488/...-finetuned-ner`** | El BETO base fue **fine-tuned** por el usuario `mrm8488` sobre el dataset CoNLL-2002 espanol para NER. Es lo que entrena la "cabeza" que clasifica cada token en 9 clases BIO |
| **`cased`** | Preserva mayusculas. Critico para NER porque `Madrid` vs `madrid` son tokens distintos en el vocab y dan embeddings distintos |
| **`output_attentions=True`** | El flag clave del lab. Por default el modelo solo retorna logits para no desperdiciar memoria. Con esta flag, ademas devuelve las **matrices de atencion de las 12 capas**, lo que permite visualizarlas con `bertviz` mas adelante |

Al ejecutar la celda aparecen warnings del tipo:

```
bert.pooler.dense.bias   | UNEXPECTED |
bert.pooler.dense.weight | UNEXPECTED |
```

**Estos warnings son esperados y correctos.** El checkpoint del modelo trae los pesos del **pooler** (la capa que produce el resumen del token `[CLS]` para clasificacion de secuencia), pero `AutoModelForTokenClassification` no instancia ese modulo — porque NER es **clasificacion token-a-token**, no clasificacion de secuencia. Cada uno de los embeddings finales pasa por una capa lineal `768 → 9 clases` independientemente. El pooler simplemente se descarta.

Si vieras el warning opuesto — `MISSING: classifier.weight` — eso si seria un problema: significaria que la cabeza de NER no esta inicializada y estarias prediciendo con pesos aleatorios.

## Tokenizacion: WordPiece y tokens especiales

Para entender que ve el modelo, hay que descomponer el camino del texto al input numerico.

### El truco del round-trip

El notebook usa *(parte 1, celda 13)* un patron que parece redundante pero tiene una razon pedagogica:

```python
text = 'Eduardo Vargas le metio un gol a Espana en el mundial de Brasil.'
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
```

| Paso | Operacion | Salida |
| --- | --- | --- |
| `encode(text)` | Texto → IDs numericos, **agregando** `[CLS]` al inicio y `[SEP]` al final | `[4, 8421, 12943, ..., 5]` |
| `decode(...)` | IDs → string plano, con tokens especiales como texto | `'[CLS] Eduardo Vargas le metio... [SEP]'` |
| `tokenize(...)` | String → lista de tokens WordPiece, incluyendo los especiales | `['[CLS]', 'Eduardo', 'Vargas', 'le', 'metio', ..., '[SEP]']` |

Sin el round-trip, `tokenizer.tokenize(text)` daria los WordPieces **sin** los tokens especiales. Con el round-trip, conseguimos la lista completa que el modelo realmente vera.

### Tokens especiales `[CLS]` y `[SEP]`

Al ejecutar `tokenizer.decode(tokenizer.encode(text))` *(parte 1, celda 15)*:

```
[CLS] Eduardo Vargas le metio un gol a Espana en el mundial de Brasil. [SEP]
```

| Token | Posicion | Funcion |
| --- | --- | --- |
| `[CLS]` | Inicio | Durante el pre-entrenamiento, su embedding final se usa para **NSP** (Next Sentence Prediction). En fine-tuning de clasificacion de secuencia, se usa como vector resumen. En NER **no se usa**, pero igual va siempre porque BERT fue entrenado esperandolo |
| `[SEP]` | Final de cada frase | Marca fin de oracion. Si hay dos frases en el input, va entre ambas y al final |

Los pesos de BERT aprendieron durante el pre-entrenamiento que esos tokens **siempre estan ahi**. Quitarlos cambia la distribucion estadistica del input y degrada las representaciones — es parte del "contrato" de entrada de BERT.

### WordPiece: que palabras se rompen

WordPiece (Wu et al. 2016) es el algoritmo de tokenizacion sub-palabra de BERT. La idea: vocabulario fijo (~30k tokens para BETO), palabra completa si esta en vocab, si no, romperla en piezas — la primera sin prefijo, las siguientes con `##`.

Inspeccionando la primera frase *(parte 1, celda 19)*:

```python
['[CLS]', 'Eduardo', 'Vargas', 'le', 'metio', 'un', 'gol', 'a',
 'Espana', 'en', 'el', 'mundial', 'de', 'Brasil', '.', '[SEP]']
```

**Sin sub-palabras**: BETO conoce todas estas palabras enteras. Tiene sentido — son palabras frecuentes en el corpus espanol con el que entrenaron BETO.

Inspeccionando una frase con dos oraciones unidas *(parte 1, celdas 22 y 23 round-trip)*, la cosa cambia:

```python
['[CLS]', 'Eduardo', 'Vargas', 'le', 'metio', 'un', 'gol', 'a', 'Espana',
 'en', 'el', 'mundial', 'de', 'Brasil', '.', '[SEP]', 'Y', 'Alex', '##is',
 'Sanchez', 'a', 'Brasil', 'en', 'octavo', '##s', ',', 'en', 'Bel', '##o',
 'Hor', '##izon', '##te', '[SEP]']
```

Aparecen 5 sub-piezas extras en la segunda frase:

| Palabra | WordPieces | Por que se rompe |
| --- | --- | --- |
| `Alexis` | `Alex` + `##is` | Nombre propio menos frecuente que `Eduardo`. BETO conoce "Alex" pero no "Alexis" completo |
| `Sanchez` | (entero) | Apellido comun en espanol — esta en el vocab |
| `octavos` | `octavo` + `##s` | BETO conoce la raiz singular pero no este plural especifico como token unico |
| `Belo` | `Bel` + `##o` | Ciudad brasilena, rara en corpus espanol |
| `Horizonte` | `Hor` + `##izon` + `##te` | Roto en **tres** piezas — muy rara en el corpus de pre-entrenamiento |

> **Observacion interesante:** la pieza mas "exotica" para BETO no es el nombre `Alexis` (2 piezas) sino el toponimo brasileno `Horizonte` (3 piezas). Esto refleja que BETO fue entrenado mayormente con texto en espanol peninsular y latinoamericano hispano, no portugues ni topomimos brasilenos. Es un sesgo del corpus.

### `token_type_ids` (segment IDs)

Cuando se pasa una sola frase al tokenizer *(parte 1, celda 21)*:

```python
sample_input = tokenizer(text, return_tensors="pt")
sample_input['token_type_ids']
# tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
```

Con **dos frases** *(parte 1, celda 22)*:

```python
sample_input2 = tokenizer(text, text2, return_tensors="pt")
sample_input2['token_type_ids']
# tensor([[0,0,...,0, 1,1,...,1]])  ← 16 ceros + 17 unos
```

| Valor | Significado |
| --- | --- |
| `0` | Token pertenece a la **frase A** (incluye `[CLS]` y el primer `[SEP]`) |
| `1` | Token pertenece a la **frase B** (incluye el segundo `[SEP]`) |

Este vector se suma a los embeddings antes del primer Transformer block:

$$\text{embedding}_{i} = \text{word\_emb}_i + \text{position\_emb}_i + \text{segment\_emb}_i$$

donde `segment_emb` es una matriz `2 × 768` aprendida durante pre-entrenamiento. Le permite a BERT distinguir tokens de A vs B en tareas como QA, NLI o similitud que requieren pares de frases.

> Para NER no necesitamos dos frases (es tarea sobre una sola oracion), pero el lab muestra el formato porque es parte del contrato de BERT. GPT no tiene `token_type_ids` porque no fue entrenado con NSP — esa es una diferencia arquitectural sutil pero relevante.

## NER: prediccion + visualizacion

### Forward pass

```python
inputs = tokenizer(text, return_tensors="pt")
inputs.to(device)

outputs = model(**inputs)[0]
predictions = torch.argmax(outputs, dim=2)
predictions = predictions[0].tolist()
```

| Linea | Que hace |
| --- | --- |
| `model(**inputs)` | Desempaca el dict (`input_ids`, `attention_mask`, `token_type_ids`) y hace el forward |
| `[0]` | Toma el **primer** elemento del output del modelo — los **logits** de shape `(1, seq_len, 9)`. Con `output_attentions=True`, **`[-1]`** seria la tupla de matrices de atencion (la usaremos en la siguiente seccion) |
| `argmax(dim=2)` | Para cada token, la clase con mayor logit. Shape `(1, seq_len)` |
| `predictions[0].tolist()` | Quita la dim de batch y pasa a lista Python |

NER aqui es **clasificacion token-a-token independiente**: cada token recibe 9 logits y se elige el maximo. No hay decoder ni softmax conjunto sobre la secuencia. Eso puede generar inconsistencias BIO (un `I-PER` sin `B-PER` previo). Modelos NER mas sofisticados ponen un **CRF** encima de los logits para forzar consistencia BIO, pero este modelo simple no lo tiene.

### Las 9 clases BIO

```python
LABEL_LIST = ["B-LOC", "B-MISC", "B-ORG", "B-PER",
              "I-LOC", "I-MISC", "I-ORG", "I-PER", "O"]
```

| Prefijo | Significado |
| --- | --- |
| `B-XXX` | **Beginning** — primer token de una entidad de tipo XXX |
| `I-XXX` | **Inside** — continuacion de una entidad |
| `O` | **Outside** — no es entidad |

Tipos: `LOC` (lugar), `PER` (persona), `ORG` (organizacion), `MISC` (misceláneo — el cajón de sastre).

### Visualizacion con `displacy`

El notebook construye un objeto `Doc` de spaCy y lo pinta con `displacy.render` *(parte 1, celdas 25 y 26)*. spaCy aqui es **puro render layer** — no esta haciendo NLP, solo dibujando cajas de colores sobre los tokens. El trabajo real (prediccion) ya lo hizo BERT. Para esto las funciones auxiliares de la celda 10 hacen el plumbing necesario: convertir WordPieces de BERT en tokens spaCy (sacando los `##`), decidir donde poner espacios, agrupar entidades BIO contiguas en spans.

Resultado sobre la frase de ejemplo (representacion textual del output HTML de displacy):

```text
[Eduardo Vargas]PER le metio un gol a [Espana]ORG en el mundial de [Brasil]LOC .
```

Tres entidades reconocidas:

| Texto | Etiqueta predicha | Correcto |
| --- | --- | --- |
| Eduardo Vargas | **PER** | Si — persona |
| Espana | **ORG** | **No** — deberia ser **LOC** |
| Brasil | **LOC** | Si — lugar |

## El error `Espana → ORG`: leccion conceptual

El error no es un bug, es **comportamiento tipico** de modelos NER y es **el momento didactico** mas importante de esta seccion. Por que pasa:

1. **Ambiguedad contextual.** En frases como "Espana gano el mundial", "Espana ficho a un jugador", "Espana celebra", la palabra "Espana" funciona como **seleccion nacional** (organizacion deportiva), no como pais geografico. El modelo aprendio que en contextos deportivos, los paises a veces se usan **metonimicamente** como organizaciones.

2. **El contexto de la frase es deportivo.** "Eduardo Vargas le **metio un gol** a X" → BETO ve "metio un gol a X" y aprende que X suele ser una **seleccion/equipo** (ORG), no un territorio (LOC).

3. **Brasil escapa al error** probablemente porque la frase termina con "en el mundial de Brasil" — aqui "Brasil" actua como **sede** del torneo, contexto claramente geografico.

### La idea profunda

Las etiquetas de NER **no son propiedades intrinsecas de las palabras**, sino del **rol contextual** que cumplen:

- "Visite Espana" → LOC (territorio)
- "Espana gano 3-0" → ORG (seleccion)
- "Soy de Espana" → LOC

BERT contextualiza embeddings, asi que el mismo token tiene representaciones distintas segun el contexto y eso lleva a etiquetas distintas. **Esto es feature, no bug** — pero el modelo a veces se equivoca en casos de borde como este.

## Validacion empirica del sesgo

Probando tres frases distintas *(parte 1, celda 28)*, el patron se confirma:

**Frase 1 — contexto turistico/cotidiano:**

> *"Visite Espana el verano pasado y comi paella en Madrid."*
> → `Espana` y `Madrid` ambos como **LOC**. Confirma la hipotesis: sin verbos deportivos, el modelo lee Espana como territorio.

**Frase 2 — contexto deportivo (clubes):**

> *"El Real Madrid vencio al Barcelona en el Bernabeu."*
> → `Real Madrid` y `Barcelona` ambos como **ORG** (clubes), `Bernabeu` como **LOC** (estadio).

Notable nivel de matizacion: el modelo usa el verbo "vencio" para deducir que "Barcelona" aqui es el club FC Barcelona, no la ciudad. Distinguir "Barcelona-ciudad" de "Barcelona-club" requiere context-aware embeddings — BERT lo hace bien aqui porque el verbo "vencio" solo aplica a entidades animadas/organizacionales.

**Frase 3 — dominio especializado (salud digital):**

> *"Salud digital en Chile: HL7 Chile lidera la implementacion de FHIR."*
> → `Chile` como **MISC** (deberia ser LOC), `HL7 Chile` no se reconoce como entidad, `FHIR` como **MISC**.

Aqui el modelo **falla y sirve de leccion**:

| Entidad | Predicho | Esperado | Razon del error |
| --- | --- | --- | --- |
| HL7 Chile | (no aparece) | ORG | El modelo no conoce HL7. Tokeniza como `HL` + `##7` + `Chile` y el clasificador no agrupa esas piezas como una entidad coherente |
| Chile (primer) | **MISC** | LOC | El contexto "Salud digital en X" no tiene precedentes en el corpus de entrenamiento (CoNLL-2002 es periodistico, no tecnico) |
| FHIR | **MISC** | MISC | Cuando no es PER/LOC/ORG, MISC es el cajon de sastre. FHIR como estandar tecnico encaja |

## Implicaciones para dominios especializados

Este experimento revela las limitaciones reales de modelos NER pre-entrenados:

1. **Sesgo de corpus.** BETO + CoNLL-2002 vio principalmente **textos periodisticos generales** en espanol. Salud digital, terminologia clinica, estandares HL7/FHIR — todo es **out-of-distribution**.

2. **Tokenizacion degrada en dominios especializados.** "HL7" probablemente queda como `HL` + `##7` (porque "HL7" no esta en el vocab de 30k). El modelo trata las sub-piezas inconexamente y pierde la entidad.

3. **Para un caso real (matching de pacientes FHIR, por ejemplo), un NER off-the-shelf no sirve.** Habria que:
   - **Fine-tuning** con un dataset clinico en espanol (escaso, sensible a privacidad)
   - O un modelo medico ya fine-tuned: `PlanTL-GOB-ES/roberta-base-bne-capitel-ner-plus`, **xMEN** (multilingual entity normalization in medicine), o **BioMedRoBERTa-es**
   - Para HL7/FHIR especificamente, anotar dataset propio o combinar reglas + LLM

> Es **exactamente la razon** por la que arquitecturas MDM serias apuestan a un GBM scorer con features explicitas (UCUM, CQL, demograficos) y bi-encoder solo como blocker — no a un NER pipeline. Los embeddings tipo BERT pierden senal en jerga clinica.

## Lo que viene en la siguiente seccion

Esta seccion uso BERT como caja negra: paso texto, salieron etiquetas. **Las siguientes secciones abren el capo** y miran como BERT distribuye atencion internamente — primero a nivel de capas y cabezas (head_view, model_view), despues a nivel de vectores Q y K dimension por dimension (neuron_view). El patron `output_attentions=True` que activamos aqui es lo que habilita todo eso.
