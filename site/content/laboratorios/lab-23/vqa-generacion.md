---
title: "VQA como generación"
weight: 2
math: true
---

> **Celdas 7-24 del notebook.** Cargar imágenes reales, construir las entradas con el `processor` y dejar que BLIP **responda preguntas generando texto**. Tres aciertos (uno sorprendente) y un límite estructural: el conteo. De paso, un bug real en la celda 24 que delata cómo está armado el experimento.

## VQA como generación, no como clasificación

El cambio mental respecto a [Pythia](../../papers/pythia-jiang-2018) es el corazón de esta clase. Pythia trata VQA como **clasificación**: un `argmax` sobre un vocabulario fijo de ~3000 respuestas frecuentes. BLIP, en cambio, **genera** la respuesta token a token:

```python
out = model.generate(**inputs)
```

`generate` es **decodificación autoregresiva**: el decoder emite un token, lo realimenta, emite el siguiente, hasta producir `[SEP]`. Por defecto, **greedy decoding** (en cada paso toma el token de mayor probabilidad). Internamente el flujo es:

1. El **ViT** codifica la imagen en una secuencia de embeddings de parches.
2. La **pregunta** entra al text encoder, que aplica **cross-attention** sobre las features visuales (el texto "mira" la imagen).
3. El **decoder** genera la respuesta condicionada en esa representación image-grounded.

No hay lista cerrada de respuestas: el modelo puede emitir cualquier cadena que su vocabulario permita. Ver [Vision-Language Models](../../fundamentos/vision-language-models) y el paper [BLIP (Li et al., 2022)](../../papers/blip-li-2022).

## El pipeline del processor (celda 12)

```python
inputs = processor(image, q1, return_tensors="pt")   # q1 = "What is this?"
```

El `processor` es un objeto compuesto que abre **dos ramas** en paralelo:

| Rama | Qué hace | Salida |
|---|---|---|
| **Imagen** | resize a 384×384, normalización ImageNet | `pixel_values` `[1, 3, 384, 384]` |
| **Texto** | tokenizer estilo BERT (WordPiece) | `input_ids`, `attention_mask` |

`return_tensors="pt"` pide tensores de PyTorch (con batch dimension). Y antes de todo, en la carga:

```python
image = Image.open("demo.jpg").convert('RGB')
```

`.convert('RGB')` fuerza **3 canales**. Es un *gotcha* clásico: un PNG con transparencia llega como RGBA (4 canales) y una imagen médica o un escaneo puede llegar en escala de grises (1 canal); el ViT espera exactamente 3 y revienta si no.

Para leer la salida hay que **decodificar**:

```python
processor.decode(out[0], skip_special_tokens=True)
```

Sin `skip_special_tokens=True` verías la respuesta envuelta en `[CLS] ... [SEP]`. `out[0]` son IDs de tokens, no texto.

## "What is this?" → "dog and beach" (celda 13)

La imagen demo de Salesforce muestra una **mujer sentada en la playa con su perro**. BLIP responde:

> **dog and beach**

Es **correcta pero parcial**: ignora a la mujer, que es el sujeto más prominente. Greedy decoding eligió lo más **saliente y frecuente** en su distribución, y se detuvo. Además, `blip-vqa-base` está fine-tuneado sobre **VQAv2**, donde las respuestas son **telegráficas** (1-3 palabras): el modelo aprendió a no elaborar. No es que "no vea" a la mujer; es que el estilo de respuesta de VQAv2 premia lo corto.

## "is there a girl?" → "yes" (celda 14)

```python
inputs2 = processor(image, q2, return_tensors="pt")   # q2 = "is there a girl?"
out2 = model.generate(**inputs2)   # → "yes"
```

Las preguntas **sí/no** son la categoría más fácil de VQAv2 (accuracy ~80%). Aquí la mujer que el modelo "omitió" en la respuesta anterior sí está representada internamente: cuando la pregunta apunta directo a ella, la cross-attention la recupera. El contraste con "dog and beach" confirma que el problema anterior era de **estilo de generación**, no de percepción.

## "olives" — la respuesta impresionante (celdas 16-18)

La imagen es una **ensalada** (demo de PnP-VQA). La pregunta:

> What is the black objects on the salad called? → **olives**

Esto es mucho más difícil de lo que parece. Para responder "olives" el modelo tiene que encadenar:

1. **Localizar** los objetos negros dentro del plato (percepción).
2. **Razonar con conocimiento del mundo**: negro + redondo + sobre una ensalada ⇒ aceitunas.
3. **Generar** la palabra correcta.

Ningún clasificador de 3000 clases hace esto sin que "olives" esté en su vocabulario. Esta respuesta es la vitrina del **pre-entrenamiento masivo vision-language**: BLIP vio millones de pares imagen-texto de la web y de ahí salió esa asociación. Es el argumento de venta de los VLM generativos.

## El conteo de jirafas — el límite estructural (celdas 20-24)

Imagen COCO con **dos jirafas**. Primera pregunta:

```python
inputs4 = processor(image2, "How many giraffes are there?", return_tensors="pt")
out4 = model.generate(**inputs4)   # → "1"
```

Responde **"1"**. Hay dos. Es **erróneo**, y no por azar: el conteo es de las **peores categorías** de VQAv2 (~50% de accuracy vs ~80% en yes/no). Tres razones estructurales explican el fallo:

1. **El conteo no es composicional para un decoder de lenguaje.** En VQAv2 hay un **sesgo de prior** hacia números bajos ("1", "2", "3" dominan las respuestas de conteo); el modelo apuesta a lo frecuente en vez de enumerar.
2. **El ViT colapsa la imagen en parches 16×16 sin noción de "objetidad".** Un parche es una grilla, no un objeto. Esto contrasta de raíz con Pythia, donde la entrada son **regiones Mask R-CNN** y *cada región es, por construcción, un objeto candidato* — contar es casi natural.
3. **La cross-attention agrega/promedia, no enumera.** El mecanismo está hecho para fusionar evidencia visual en una representación, no para llevar un contador discreto de instancias.

| | Pythia | BLIP |
|---|---|---|
| Entrada visual | **Regiones** (Mask R-CNN), ~100 objetos | **Parches** 16×16 de un grid |
| Unidad mínima | un objeto | un trozo de imagen sin objetidad |
| Salida | argmax sobre ~3000 clases | generación autoregresiva |
| Conteo | más natural (regiones = objetos) | débil (parches no enumerables) |

El poder generativo de BLIP ("olives") y su debilidad en conteo ("1") son **dos caras de la misma arquitectura**: cambiar regiones por parches y clasificación por generación gana flexibilidad y conocimiento del mundo, pero pierde la enumeración explícita de objetos. Ver [Pythia (Jiang et al., 2018)](../../papers/pythia-jiang-2018).

## ⚠️ El bug de la celda 24

La segunda pregunta sobre las jirafas debería ser sí/no:

```python
inputs5 = processor(image2, "Are there 2 giraffes?", return_tensors="pt")   # question_2
```

Pero la celda 24 imprime, etiquetando la salida como "Are there 2 giraffes?":

```python
out5 = model.generate(**inputs4)   # ⚠️ usa inputs4 (la de conteo), NO inputs5
print(processor.decode(out5[0], skip_special_tokens=True))   # → "1"
```

El **tell** es la respuesta: a una pregunta **sí/no** ("Are there 2 giraffes?") el modelo "responde" **"1"**, un número. Eso es imposible si realmente le hubieras pasado la pregunta sí/no — BLIP habría dicho "yes" o "no". Lo que pasó es que se reutilizó `inputs4` (la pregunta de conteo) por copy-paste, y `inputs5` **nunca se usa**. El experimento muestra dos veces la misma inferencia con etiquetas distintas.

**Fix:**

```python
out5 = model.generate(**inputs5)   # ahora sí la pregunta sí/no
print(processor.decode(out5[0], skip_special_tokens=True))   # → "yes" (probablemente)
```

Es un recordatorio práctico: cuando una salida **no tiene la forma esperada** (un número donde debía ir yes/no), antes de culpar al modelo conviene revisar qué tensor se le pasó.

---

**Anterior:** [Arquitectura: BLIP y el MED](arquitectura-blip) · **Siguiente:** [Modos de fallo](modos-de-fallo)
