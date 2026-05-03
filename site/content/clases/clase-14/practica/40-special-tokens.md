---
title: "40 - Special tokens: [CLS], [MASK], [SEP]"
weight: 400
math: true
---

## 1. Tokens que no existen en el texto natural

BERT introduce 3 tokens especiales que no existen en el vocab BPE — estan fuera del espacio de texto natural. Son instrucciones para el modelo.

Un tokenizador BPE aprende su vocabulario a partir del corpus. Si entrenas sobre el Quijote y Shakespeare, los tokens del vocab son fragmentos de esas obras: caracteres, silabas, palabras frecuentes. Ninguna frase del Quijote empieza con `[CLS]` ni contiene `[MASK]`. Esos tokens no emergen del corpus — se insertan de forma artificial por disenio del arquitecto.

Este es el punto clave: cuando el tokenizador tiene 1112 tokens aprendidos del corpus, agregar `[CLS]`, `[SEP]` y `[MASK]` extiende el vocab a 1115 con tres entradas que tienen IDs consecutivos pero ningun texto de entrenamiento BPE los genero. El modelo aprende a usarlos unicamente por su posicion y funcion durante el pretraining.

---

## 2. Rol de cada token

### [CLS] — Classification Token

`[CLS]` ocupa la **posicion 0** de cada input que entra a BERT. Es el primer token, siempre, sin excepcion.

Su funcion no es representar contenido. Cuando la secuencia pasa por los $N$ bloques de atencion, el vector en la posicion 0 acumula informacion de toda la secuencia via self-attention. Al final del ultimo bloque, ese vector — que llamamos $\mathbf{h}_{[CLS]}$ — se usa como representacion de toda la oracion para tareas de clasificacion.

**Por que [CLS] no tiene contenido semantico propio:** Al inicio del training, el embedding de `[CLS]` es un vector aleatorio. No corresponde a ninguna palabra, no tiene significado linguistico previo. Lo que ocurre durante el pretraining MLM es que el modelo aprende, de forma implicita, a concentrar en ese vector informacion util sobre el contexto completo. No hay ninguna supervision directa que le diga "en la posicion 0 pon un resumen de la frase" — ese comportamiento emerge como efecto secundario de que todas las otras posiciones atienden a `[CLS]` y viceversa.

En fine-tuning, se agrega una capa lineal sobre $\mathbf{h}_{[CLS]}$:

$$\hat{y} = \text{softmax}(W_c \cdot \mathbf{h}_{[CLS]} + b_c)$$

El vector de `[CLS]` es el punto de contacto entre el encoder y la cabeza de clasificacion.

### [SEP] — Separator Token

`[SEP]` marca el fin de un segmento. En el caso de una sola frase, aparece al final. En el caso de dos frases (por ejemplo, para la tarea de Next Sentence Prediction), separa la primera de la segunda:

```
[CLS] frase_A [SEP] frase_B [SEP]
```

En nuestros experimentos con una sola frase, el patron es simplemente:

```
[CLS] tokens... [SEP]
```

`[SEP]` no tiene ninguna funcion especial en la salida — no hay una cabeza conectada a ese vector. Su rol es estructural: delimita donde termina un segmento para que el modelo pueda distinguir entre partes del input.

### [MASK] — Mask Token

`[MASK]` reemplaza tokens durante el pretraining con Masked Language Modeling (MLM). El procedimiento en BERT original:

1. Seleccionar al azar el 15% de los tokens del input.
2. De esos, el 80% se reemplaza con `[MASK]`, el 10% con un token aleatorio del vocab, el 10% se deja igual.
3. El modelo debe predecir el token original en cada posicion enmascarada.

Esto obliga al modelo a aprender representaciones contextuales ricas: para predecir el token oculto debe leer toda la secuencia.

---

## 3. Por que [CLS] en lugar del promedio de todos los tokens

Una alternativa obvia seria usar el promedio de los vectores de salida de todos los tokens como representacion de la secuencia. iPor que BERT usa un token dedicado en vez de pooling?

Hay tres razones de disenio:

**1. Separacion de roles.** Los otros tokens tienen una tarea propia: reconstruirse a si mismos en MLM. Si el promedio se usa tambien para clasificacion, los gradientes de ambas tareas compiten. `[CLS]` es un token sin tarea MLM — sus gradientes en pretraining vienen solo de la perdida de clasificacion (o de la tarea NSP en BERT original), lo que lo especializa libremente.

**2. Asimetria de atencion.** En self-attention, `[CLS]` puede atender a todos los tokens de la secuencia desde la posicion 0. El promedio de salidas no tiene esta propiedad: es solo un agregado post-hoc, no un vector que participo activamente en los calculos de atencion.

**3. Flexibilidad para dos frases.** Con `[CLS]` al inicio, BERT puede procesar pares de frases (`[CLS] A [SEP] B [SEP]`) sin cambiar la arquitectura. El token `[CLS]` sigue siendo el punto de clasificacion en ambos casos. Un esquema de promedio tendria que decidir sobre que tokens promediar en presencia de dos frases.

El promedio de tokens (mean pooling) se usa en modelos de sentence embeddings como Sentence-BERT (Reimers y Gurevych, 2019), pero requiere ajuste fino especifico para esa tarea. El `[CLS]` de BERT raw sin ajuste fino raramente produce buenos sentence embeddings.

---

## 4. El script

`clase_14/practica/40_special_tokens.py`:

```python
"""40_special_tokens.py - Cap 40: [CLS], [MASK], [SEP] en accion."""
from pathlib import Path
from _bpe import BPETokenizer

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

print("=== Special tokens BERT ===\n")
print(f"[CLS]  id={tok.cls_id}  — Classification token (inicio de secuencia)")
print(f"[SEP]  id={tok.sep_id}  — Separator token (fin de segmento)")
print(f"[MASK] id={tok.mask_id} — Mask token (reemplaza tokens en MLM)")
print(f"\nVocab size antes: 1112  | despues: {tok.vocab_size}")

sentences = [
    "To be or not to be",
    "En un lugar de la Mancha",
]
print("\n=== encode_bert vs encode regular ===\n")
for s in sentences:
    regular = tok.encode(s)
    bert = tok.encode_bert(s)
    print(f"Texto:   {s!r}")
    print(f"Regular: {regular[:5]}... ({len(regular)} tokens)")
    print(f"BERT:    {bert[:5]}... ({len(bert)} tokens)  <- +2 ([CLS] y [SEP])")
    print(f"Decode:  {tok.decode(bert)!r}\n")

print("=== Rol de cada token ===")
print("""
[CLS] — Classification Token:
  Posicion 0 de CADA input BERT.
  El vector de salida de [CLS] despues de pasar por los N bloques
  representa TODA la secuencia. Es este vector el que va a la
  cabeza de clasificacion en fine-tuning. No tiene contenido
  semantico propio — aprende a ser un "resumen" del input.

[SEP] — Separator Token:
  Indica el fin del input (o separacion entre dos frases en BERT original).
  En nuestro caso de una sola frase: marca el fin.

[MASK] — Mask Token:
  Reemplaza tokens durante pretraining MLM.
  El modelo aprende a predecir el token original dado el contexto.
  NUNCA aparece en fine-tuning — es exclusivo del pretraining.
""")
```

---

## 5. Output del script

```
=== Special tokens BERT ===

[CLS]  id=1112  — Classification token (inicio de secuencia)
[SEP]  id=1113  — Separator token (fin de segmento)
[MASK] id=1114 — Mask token (reemplaza tokens en MLM)

Vocab size antes: 1112  | despues: 1115

=== encode_bert vs encode regular ===

Texto:   'To be or not to be'
Regular: [49, 131, 234, 268, 200]... (7 tokens)
BERT:    [1112, 49, 131, 234, 268]... (9 tokens)  <- +2 ([CLS] y [SEP])
Decode:  '[CLS]To be or not to be[SEP]'

Texto:   'En un lugar de la Mancha'
Regular: [34, 294, 285, 1, 69]... (14 tokens)
BERT:    [1112, 34, 294, 285, 1]... (16 tokens)  <- +2 ([CLS] y [SEP])
Decode:  '[CLS]En un lugar de la Mancha[SEP]'

=== Rol de cada token ===

[CLS] — Classification Token:
  Posicion 0 de CADA input BERT.
  El vector de salida de [CLS] despues de pasar por los N bloques
  representa TODA la secuencia. Es este vector el que va a la
  cabeza de clasificacion en fine-tuning. No tiene contenido
  semantico propio — aprende a ser un "resumen" del input.

[SEP] — Separator Token:
  Indica el fin del input (o separacion entre dos frases en BERT original).
  En nuestro caso de una sola frase: marca el fin.

[MASK] — Mask Token:
  Reemplaza tokens durante pretraining MLM.
  El modelo aprende a predecir el token original dado el contexto.
  NUNCA aparece en fine-tuning — es exclusivo del pretraining.
```

---

## 6. Analisis del output

**IDs consecutivos al final del vocab:** `[CLS]=1112`, `[SEP]=1113`, `[MASK]=1114`. El tokenizador BPE tenia 1112 tokens aprendidos del corpus (IDs 0 a 1111). Los tres tokens especiales se agregan en orden al final, con `vocab_size` pasando de 1112 a 1115.

**encode_bert agrega exactamente 2 tokens:** "To be or not to be" tiene 7 tokens en encode regular y 9 en encode_bert — la diferencia es exactamente 2 (uno `[CLS]` al inicio, uno `[SEP]` al final). Lo mismo para "En un lugar de la Mancha": 14 tokens regulares, 16 con BERT. Esto confirma que `encode_bert` implementa literalmente `[cls_id] + encode(text) + [sep_id]`.

**El primer ID de BERT siempre es 1112:** Observa la linea `BERT: [1112, 49, 131, ...]`. El ID 1112 es `[CLS]`, y aparece en la posicion 0 en ambas frases. Esto es invariante: toda secuencia BERT empieza con 1112.

**decode(bert_ids) funciona correctamente:** El decode de los IDs BERT produce `'[CLS]To be or not to be[SEP]'`. El tokenizador sabe decodificar los tokens especiales porque `id_to_token[1112] = "[CLS]"`, etc. La funcion decode es simplemente una lookup en el diccionario inverso del vocab — funciona identicamente para tokens especiales y tokens BPE.

**Frases en ingles vs espanol tienen distinta cantidad de tokens:** "To be or not to be" (6 palabras) se tokeniza en 7 tokens con BPE, mientras que "En un lugar de la Mancha" (6 palabras) se tokeniza en 14 tokens. El tokenizador fue entrenado sobre Shakespeare (ingles) y el Quijote (espanol). La diferencia en eficiencia refleja la distribucion del corpus de training BPE: el ingles de Shakespeare es mas comprimible para este tokenizador especifico porque el vocabulario de merges fue aprendido tambien de texto espanol (y no solo de uno de los dos).

---

## 7. NOTA IMPORTANTE: [MASK] nunca aparece en fine-tuning

Esta es una de las discordancias mas importantes del diseno BERT original, y fue criticada en trabajos posteriores (XLNet, RoBERTa).

Durante el **pretraining**, el 15% de los tokens se reemplaza con `[MASK]` (entre otras estrategias). El modelo ve inputs del tipo:

```
[CLS] the cat [MASK] on the mat [SEP]
```

y aprende a predecir que el token en la posicion 4 era `sat`.

Durante el **fine-tuning** (por ejemplo, clasificacion de sentimiento), el input es:

```
[CLS] this movie is great [SEP]
```

El token `[MASK]` **nunca aparece**. El modelo fue entrenado durante cientos de miles de pasos con `[MASK]` en el input, pero en inferencia y fine-tuning ese token nunca esta presente. Esto crea una discordancia entre la distribucion de pretraining y la de fine-tuning.

La propia solucion parcial de BERT original es que solo el 80% de los tokens seleccionados se reemplaza con `[MASK]`; el 10% se reemplaza con un token aleatorio y el 10% se deja igual. Esto obliga al modelo a mantener representaciones utiles para todos los tokens, no solo los enmascarados.

Trabajos posteriores abordaron esto de distintas formas:
- **XLNet** (Yang et al., 2019): usa permutation language modeling en lugar de masking — no hay token `[MASK]` en absoluto.
- **RoBERTa** (Liu et al., 2019): mantiene el esquema BERT pero con masking dinamico (las mascaras se generan en cada epoch, no una sola vez).
- **ELECTRA** (Clark et al., 2020): reemplaza MLM con una tarea discriminativa donde el modelo detecta tokens "reemplazados" — tampoco usa `[MASK]` en el mismo sentido.

La discordancia pretraining/fine-tuning de `[MASK]` es un compromiso de disenio aceptado en BERT, no un error — el paper lo reconoce y justifica empiricamente. Pero es importante tenerlo presente al usar BERT en produccion.

---

## 8. Preguntas de verificacion

**1.** El output muestra que `[CLS]` tiene ID 1112, `[SEP]` tiene ID 1113 y `[MASK]` tiene ID 1114. iQue pasaria si llamaras a `add_special_tokens()` dos veces? iCambiarian los IDs? Lee el codigo de `add_special_tokens()` en `_bpe.py` y explica por que (o por que no).

**2.** "En un lugar de la Mancha" produce 14 tokens en encode regular pero 16 en encode_bert. Si codificaras el par de frases `"To be or not to be"` + `"En un lugar de la Mancha"` en formato BERT two-sentence (`[CLS] A [SEP] B [SEP]`), icuantos tokens tendria el resultado total? Escribe el codigo para verificarlo.

**3.** El `[MASK]` nunca aparece en fine-tuning. Sin embargo, su embedding (el vector en la tabla de embeddings para el ID 1114) existe y tiene valores aprendidos. iEse embedding se usa en fine-tuning? iQue pasa con esos pesos durante el ajuste fino?
