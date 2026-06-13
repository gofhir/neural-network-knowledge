---
title: "Image Captioning con BLIP"
weight: 4
---

> **Celdas 30-34 del notebook.** El segundo gran experimento del lab: dejar que el modelo **describa una imagen por su cuenta**, sin pregunta. Otra cabeza (`BlipForConditionalGeneration`), otro checkpoint, y la misma lección de fondo que en VQA — el modelo brilla en lo prototípico y alucina en lo raro.

## De responder a describir

El [VQA](/fundamentos/vqa) parte de un par *(imagen, pregunta)* y produce una respuesta corta y condicionada. El **image captioning** elimina la pregunta: la única entrada es la imagen, y la tarea es **generar una descripción autónoma** de lo que hay en ella. Ver el [fundamento Image Captioning](/fundamentos/image-captioning).

| | Entrada | Salida | Cabeza HF |
|---|---|---|---|
| **VQA** (celdas previas) | imagen **+** pregunta | respuesta corta | `BlipForQuestionAnswering` |
| **Captioning** (celdas 30-34) | imagen sola | frase descriptiva | `BlipForConditionalGeneration` |

## La diferencia clave en el código

En VQA el `processor` recibía dos argumentos:

```python
inputs = processor(image, question, return_tensors="pt")   # VQA: condicionado por texto
```

En captioning, el texto desaparece:

```python
inputs = processor(image, return_tensors="pt")             # captioning INCONDICIONAL
out = model.generate(**inputs, max_length=20)
```

Sin argumento de texto, BLIP hace **captioning incondicional**: genera la frase desde cero usando solo las *features* visuales. (BLIP también soporta captioning **condicional**, donde uno pasa un prefijo como `"a photography of"` y el modelo lo continúa; aquí no se usa — eso se explora en la página de [decodificación y robustez](decoding-y-robustez).)

## Otra cabeza, otro checkpoint — y un gotcha

La celda 31 carga un modelo **distinto** al de VQA:

```python
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")  # ~990 MB
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
```

`BlipForConditionalGeneration` ≠ `BlipForQuestionAnswering`: distinta cabeza generativa, distinto checkpoint pre-entrenado.

> **⚠️ Gotcha de estado.** La celda 31 **reasigna** `model` y `processor`, pisando los de VQA. Si después vuelves a ejecutar una celda de VQA sin recargar el checkpoint de QA, estarás corriendo el modelo equivocado y obtendrás resultados sin sentido. En un notebook con variables globales mutables, el orden de ejecución importa.

El helper queda así:

```python
def show_and_caption(url):
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_length=20)
    print(processor.decode(out[0], skip_special_tokens=True))
```

`max_length=20` corta el caption a 20 tokens. Es el **primer parámetro de decodificación explícito del lab**: sin un tope, los modelos generativos tienden a divagar o repetirse hasta el límite duro. Es el mismo control que veremos extendido en [decodificación y robustez](decoding-y-robustez).

## Aclaración importante: BLIP-1 no usa Q-Former

El markdown del curso (celda 30) afirma que BLIP usa **ViT → Q-Former → Language Model**. **Esto es incorrecto para el modelo de este lab.**

- El **Q-Former** (Querying Transformer) pertenece a **BLIP-2** (Li et al., 2023, arXiv 2301.12597), donde un módulo ligero conecta un *encoder* visual congelado con un LLM congelado.
- El **BLIP original** — el de este lab, `blip-image-captioning-base` (Li et al., 2022) — usa el **MED (Multimodal mixture of Encoder-Decoder)**. Para generar captions opera en modo **image-grounded text decoder**: *self-attention* causal sobre el texto generado + *cross-attention* a las *features* de la imagen, con objetivo de Language Modeling (predecir el siguiente token).

Es una confusión muy frecuente porque ambos modelos se llaman "BLIP" y comparten autores. Pero arquitectónicamente son distintos: aquí **no hay Q-Former**. Ver el [paper de BLIP](/papers/blip-li-2022).

## Los tres casos

```python
show_and_caption("https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg")  # celda 32
show_and_caption("<url ornitorrinco>")                                                    # celda 33
show_and_caption("<url grupo de jóvenes>")                                                # celda 34
```

| Imagen | Caption generado | Veredicto |
|---|---|---|
| **Perro** (celda 32) | *"a white dog sitting in the grass"* | ✅ acierto preciso |
| **Ornitorrinco** (celda 33) | *"a baby bird is held in a box"* | ❌ alucinación total |
| **Grupo de jóvenes** (celda 34) | *"group of young people standing in front of white brick wall"* | ✅ acierto complejo (pero evita contar) |

**Perro — el caso prototípico.** Una imagen sobre-representada en los datos de pre-entrenamiento (perros en pasto son cientos de miles de ejemplos). El modelo no solo acierta el sujeto, sino el color, la postura y el fondo. Cuando la imagen cae en el centro de la distribución de entrenamiento, el captioning es excelente.

**Ornitorrinco — la alucinación.** Es exactamente la **misma imagen** que en VQA hizo decir al modelo *"monkey"*. Ahora, sin una pregunta que lo ancle, no solo se equivoca de animal ("bird") sino que **fabrica una escena completa**: *"is held in a box"* — algo que no está en la imagen en absoluto. Dos fenómenos se combinan:

1. **Alucinación dependiente de la tarea:** la obligación de generar una frase completa (no solo una palabra) le da más espacio para inventar.
2. **Exposure bias:** una vez que el modelo emite *"a baby bird"*, el resto de la frase se condiciona a ese error y construye una narrativa coherente *con la equivocación*, no con la imagen.

**Grupo de jóvenes — bueno, pero esquiva el número.** Acierta una escena compleja (personas, postura, fondo de ladrillo blanco), pero dice *"group of"* en vez de un cardinal exacto ("five people", "seven people"). Es **el mismo patrón que vimos con las jirafas en VQA**: el modelo prefiere un cuantificador vago antes que arriesgar un conteo. Conéctalo con la pregunta 5 de la Actividad y con los [modos de fallo](modos-de-fallo).

## Síntesis

El captioning incondicional muestra la **misma causa raíz** que el VQA, vista desde otro ángulo:

- **Brilla en lo prototípico** (el perro) porque esas escenas dominan la distribución de pre-entrenamiento.
- **Colapsa en lo raro** (el ornitorrinco) porque debe generar *algo* con confianza aunque la imagen esté fuera de distribución — y la generación autoregresiva amplifica el error inicial.
- **Evade el conteo** (el grupo) por la misma razón que en VQA: contar es frágil, y un cuantificador vago casi nunca está "mal".

La obligación de producir una salida fluida y segura es a la vez la fortaleza del modelo en lo común y su talón de Aquiles en lo infrecuente.

---

**Anterior:** [Modos de fallo](modos-de-fallo) · **Siguiente:** [Decodificación y robustez](decoding-y-robustez)
