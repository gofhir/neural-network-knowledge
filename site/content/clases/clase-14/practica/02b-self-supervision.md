---
title: "02b - Self-supervision: el dataset es su propio target"
weight: 25
math: true
---

En el capitulo 02 viste como se mide el error con cross-entropy. Pero quedo una pregunta abierta: **¿de donde sale la "respuesta correcta" contra la que se compara la prediccion?** Sin un target, no hay loss, no hay aprendizaje.

Este capitulo responde esa pregunta y explica el truco que hizo posible la era de los LLMs: **self-supervision**.

---

## 1. Backup: el target en cualquier entrenamiento

Para que un modelo aprenda, cada ejemplo de training debe ser un par:

$$\text{ejemplo} = (\text{input}, \text{target\_correcto})$$

- **Input** = lo que el modelo ve.
- **Target** = la respuesta que se supone que debe dar.

Sin un target, no hay forma de medir error, no hay loss, no hay aprendizaje.

---

## 2. Supervisado clasico (lo que probablemente ya conoces)

Ejemplo: clasificar correos como spam / no_spam.

```
correo                                        target (etiqueta humana)
──────────────────────────────────────────────────────────────────────
"Hola, te envio un saludo"                    "no_spam"
"Felicidades! ganaste 1M, click aqui"         "spam"
"Reunion mañana 10am"                         "no_spam"
"Compra Viagra al 90% off!!"                  "spam"
```

**¿Quien decidio la etiqueta?** Un humano se sento y etiqueto cada correo. Esto se llama **supervision humana** o **labeling**.

Para entrenar un modelo decente necesitas decenas o cientos de miles de ejemplos etiquetados.

{{< concept-alert type="recordar" >}}
**Costo del supervisado clasico:** etiquetar manualmente es caro y lento. Un equipo de 10 anotadores trabajando todo un ano produce ~100,000 ejemplos. Los modelos de lenguaje modernos necesitan **billones**. Imposible.
{{< /concept-alert >}}

---

## 3. La gran pregunta de NLP

Para entrenar un buen modelo de lenguaje necesitas **billones** de ejemplos. Imposible que humanos etiqueten tanto.

**Pregunta:** ¿hay alguna forma de generar pares `(input, target)` automaticamente, **sin humanos**?

**Respuesta: si, usa el texto mismo como su propio target.**

Eso es **self-supervision**.

---

## 4. Self-supervision con GPT (next-token prediction)

Toma cualquier oracion. Por ejemplo:

```
"El cielo es azul"
```

GPT "barre" la oracion generando pares automaticamente, donde el target es siempre **la siguiente palabra** del texto original.

```
input                  target_correcto
─────────────────────────────────────
"El"                   →  "cielo"
"El cielo"             →  "es"
"El cielo es"          →  "azul"
"El cielo es azul"     →  <fin>
```

**¿Quien dijo que el target era "azul" despues de "El cielo es"?** El propio texto. Antes de armar este ejemplo de training, sabiamos que "azul" venia despues porque **estaba escrito en la oracion original**.

**No hubo humano etiquetando.** La oracion misma genero 4 ejemplos de training automaticamente.

---

## 5. Self-supervision con BERT (masked language model)

BERT lo hace ligeramente distinto. Dada una oracion:

```
"El gato come pescado"
```

BERT enmascara una palabra al azar y le pide al modelo predecir cual era:

```
input al modelo:        "El gato [MASK] pescado"
target (separado):      "come"
```

El modelo ve `[MASK]` en lugar de `come`. No sabe cual era la palabra. Despues de predecir, se compara con la palabra original que **sabiamos** que estaba ahi.

---

## 6. La pregunta natural: ¿no es trampa?

> "Si el modelo tiene acceso al texto completo, ¿no es trampa?"

**No, porque el modelo NO ve el texto completo durante la prediccion.**

Mira el flujo en detalle, paso por paso:

```
PASO 1: Preparar el ejemplo
  Texto original (existe en el dataset): "El cielo es azul"
  Vamos a entrenar al modelo a predecir "azul" dado "El cielo es"

  input_para_el_modelo = "El cielo es"   ← solo 3 tokens
  target_para_el_loss  = "azul"          ← se guarda APARTE, fuera del modelo

PASO 2: Forward pass — el modelo SOLO ve el input
  output = model("El cielo es")

  El modelo NUNCA ve "azul" en este paso.
  Produce: distribucion sobre todo el vocabulario.
    P("azul")     = 0.45
    P("oscuro")   = 0.20
    P("verde")    = 0.10
    ...

PASO 3: Comparar con el target — ahora SI usamos "azul"
  loss = -log(P("azul")) = -log(0.45) = 0.80
  El "azul" se uso solo para CALCULAR el loss. Nunca entro al modelo.

PASO 4: Backprop ajusta los pesos para que la proxima vez
        P("azul") sea mayor.
```

{{< concept-alert type="clave" >}}
El modelo recibe `"El cielo es"`. Tiene que adivinar `"azul"` **sin verlo**. La palabra `"azul"` solo se usa **despues**, para calcular el error. Es como un examen: el profesor sabe la respuesta correcta porque la escribio, pero el alumno responde **sin ver la respuesta**, y solo despues se comparan.
{{< /concept-alert >}}

---

## 7. El dataset NUNCA se modifica

Esto es un punto donde mucha gente se confunde:

> **El dataset original (Wikipedia, libros, CommonCrawl) esta en disco como texto plano sin tocar.** NUNCA se enmascara permanentemente. NUNCA se le quitan palabras. Esta completo.

Lo que se enmascara es **una copia temporal en memoria** que se genera **al vuelo** (on-the-fly) cuando se esta armando un batch de training.

```python
# El dataset en disco no cambia jamas:
dataset = ["El gato come pescado", "El perro corre", ...]

for epoch in range(num_epochs):
    for sentence in dataset:
        # AQUI se genera la version enmascarada en memoria
        masked_input, target = mask_random_token(sentence)

        # Forward: el modelo SOLO ve el masked_input
        output = model(masked_input)

        # Loss: comparar con el target (que sabemos cual era)
        loss = cross_entropy(output, target)

        loss.backward()
        optimizer.step()
```

La funcion `mask_random_token` hace algo asi:

```python
def mask_random_token(sentence):
    tokens = tokenize(sentence)
    # ["el", "gato", "come", "pescado"]

    # Elegir un indice al azar (digamos que sale 2)
    idx = random.randint(0, len(tokens) - 1)

    # Guardar la palabra original como target
    target_word = tokens[idx]  # "come"

    # Reemplazar por [MASK] solo en la copia
    masked = tokens.copy()
    masked[idx] = "[MASK]"
    # ["el", "gato", "[MASK]", "pescado"]

    return masked, target_word
```

**La oracion original sigue intacta en `dataset`.** Solo se modifico una **copia local** que se le pasa al modelo.

---

## 8. La misma oracion se usa muchas veces, con mascaras distintas

Como `mask_random_token` elige al azar, cada vez que la oracion aparece en training se enmascara una palabra **distinta**:

```
Iteracion 1:  "[MASK] gato come pescado"   target: "El"
Iteracion 2:  "El gato come [MASK]"        target: "pescado"
Iteracion 3:  "El [MASK] come pescado"     target: "gato"
Iteracion 4:  "El gato [MASK] pescado"     target: "come"
Iteracion 5:  "El [MASK] come pescado"     target: "gato"   (puede repetirse)
...
```

**La misma oracion → muchas mascaras distintas → muchos ejemplos de training.** El modelo eventualmente "ve" la oracion con casi todas las posiciones enmascaradas en algun momento.

Esto se llama **dynamic masking** (introducido por RoBERTa, una mejora sobre el masking estatico de BERT original).

---

## 9. En GPT, el "masking" es causal

Para GPT (next-token prediction) la "mascara" es distinta: en vez de ocultar una palabra al azar, **siempre se ocultan todas las palabras a la derecha** del token actual.

Y hay un truco genial: **una sola pasada del modelo procesa toda la oracion y genera N ejemplos de training en paralelo**, gracias al **causal masking** (la triangular superior con $-\infty$ en los scores de atencion).

```
Dataset: "El cielo es azul"

Forward pass UNICO con causal masking:
                           El   cielo   es   azul
  Posicion 0 (vio: "El")              → predice "cielo"
  Posicion 1 (vio: "El cielo")        → predice "es"
  Posicion 2 (vio: "El cielo es")     → predice "azul"
  Posicion 3 (vio: "El cielo es azul")→ predice <fin>

Targets correctos (de la oracion original):
  ["cielo", "es", "azul", "<fin>"]

Loss: promedio de cross-entropy en las 4 posiciones
```

**Clave:** el causal masking dentro del modelo garantiza que **cada posicion solo vea las anteriores**. Aunque el modelo "lee" toda la oracion a la vez, **internamente cada token solo puede atender a su pasado**.

Con un solo forward pass, una oracion de 4 palabras genera 4 ejemplos de training. Una oracion de 100 palabras genera 100. **Eficiencia masiva.**

---

## 10. ¿Por que no es solo memorizacion?

Pregunta natural: si el modelo procesa millones de oraciones, ¿no termina simplemente memorizandolas?

**No, por dos razones:**

### Razon 1: el modelo es demasiado chico para memorizar todo

GPT-3 tiene **175 mil millones de parametros**. Suena gigante. Pero el dataset (CommonCrawl + libros + Wikipedia) tiene **trillones de tokens**. Los parametros no alcanzan para memorizar token por token. **Esta obligado a aprender patrones que comprimen.**

### Razon 2: lo que el modelo aprende son patrones generalizables

Despues de ver millones de oraciones del tipo "el cielo es azul", "el cielo es claro", "el cielo es nublado", "el cielo es gris", el modelo aprende que **despues de "el cielo es" la palabra siguiente suele ser un adjetivo de color o estado climatico**.

Cuando le das una oracion **nueva** que nunca vio antes ("Mira al horizonte, el cielo es..."), puede predecir bien porque **aprendio el patron general**, no la oracion especifica.

> Esto es el principio de **generalizacion** en machine learning. El modelo destila patrones que se aplican mas alla de los ejemplos especificos.

---

## 11. La precision: el modelo no "sabe" nada

Un punto sutil pero importante:

> El **modelo** no sabe cual es el target. El que sabe es el **training loop** (el codigo alrededor del modelo).

```python
for batch in dataset:
    # EL TRAINING LOOP (el codigo) conoce la verdad absoluta:
    masked_input, target_correcto = prepare_batch(batch)

    # EL MODELO solo ve el input. No tiene idea del target.
    output = model(masked_input)

    # EL TRAINING LOOP compara y calcula el loss.
    # El modelo recibe el gradiente, pero nunca "ve" el target directamente.
    loss = compare(output, target_correcto)
    loss.backward()
    optimizer.step()
```

El modelo es solo una **funcion matematica**: recibe numeros, produce numeros, recibe ajustes a sus pesos. No "sabe" en sentido cognitivo. La inteligencia aparente que despues exhibe (predecir bien) es resultado de millones de ajustes acumulados, no de "saber" la respuesta durante training.

---

## 12. Diagrama final

```
┌────────────────────────────────────────────────────────┐
│ DATASET EN DISCO (NUNCA SE MODIFICA)                   │
│                                                        │
│   "El gato come pescado"                               │
│   "El perro corre rapido"                              │
│   "Las aves vuelan alto"                               │
│   ... millones de oraciones ...                        │
└────────────────────────────────────────────────────────┘
                       │
                       │ (data loader lee batch)
                       ▼
┌────────────────────────────────────────────────────────┐
│ DATA LOADER (en RAM, durante training)                 │
│                                                        │
│   Toma una oracion, aplica masking al vuelo:           │
│                                                        │
│   "El gato come pescado"                               │
│         ↓ random mask                                  │
│   masked: "El [MASK] come pescado"                     │
│   target:      "gato"                                  │
└────────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ MODELO (forward pass, sin ver el target)               │
│                                                        │
│   input:  "El [MASK] come pescado"                     │
│   output: distribucion sobre vocab                     │
│           P("gato") = 0.40                             │
│           P("perro") = 0.10                            │
│           ...                                          │
└────────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│ TRAINING LOOP compara:                                 │
│                                                        │
│   loss = -log(P("gato")) = -log(0.40) = 0.92           │
│   loss.backward() → ajustar pesos                      │
└────────────────────────────────────────────────────────┘
```

---

## 13. La razon por la que self-supervision lo cambio todo

Compara las dos formas de generar training data:

### Supervisado clasico
```
1 oracion + 1 etiqueta humana = 1 ejemplo de training
1 ano de etiquetadores trabajando = ~100,000 ejemplos
```

### Self-supervised (GPT)
```
1 oracion de 100 palabras = 100 ejemplos de training
1 libro de 100,000 palabras = 100,000 ejemplos
1TB de texto de internet = miles de millones de ejemplos
                           ...todos GRATIS, sin humanos
```

Por eso GPT-3 se entreno con **570GB de texto**. Ningun humano podria etiquetar tanto. **La self-supervision es lo que hizo posible la era de los LLMs.**

---

## 14. El patron general: este loop es TODO el ML

Lo que acabas de entender no es solo de BERT/GPT. Es **el esquema universal del entrenamiento supervisado**:

```
1. Dataset tiene la verdad absoluta (input, target).
2. Training loop le pasa solo el input al modelo.
3. Modelo predice algo (probabilidades, clases, numeros).
4. Training loop compara prediccion vs target → loss.
5. Backprop convierte el loss en gradientes que ajustan los pesos.
6. Repetir hasta que el modelo prediga bien.
```

Lo unico que cambia entre tareas es:
- **De donde sale el target.** Humano (supervisado clasico) vs el texto mismo (self-supervised) vs reward de un agente (RL).
- **Que predice el modelo.** Una clase, un numero, una secuencia, una imagen.
- **Que loss se usa.** Cross-entropy, MSE, contrastive, etc.

Pero **el bucle es el mismo siempre**.

---

## 15. Pausa de verificacion

Antes de seguir al capitulo 03 (gradient descent), asegurate de poder responder:

1. **¿Que es self-supervision?** (El target sale del texto mismo, no de un humano. La supervision es "auto" en el sentido de que el dataset es su propio supervisor.)

2. **¿El dataset en disco se modifica durante training?** (No. Sigue intacto. El masking se aplica a una copia en RAM.)

3. **¿El modelo ve el target durante el forward pass?** (No. Recibe solo el input. El target se usa despues, en el training loop, para calcular el loss.)

4. **¿La misma oracion puede ser ejemplo de training varias veces?** (Si. En BERT con mascaras distintas cada vez (dynamic masking). En GPT con causal masking que genera multiples ejemplos en una sola pasada.)

5. **¿Por que el modelo no termina solo memorizando?** (Porque tiene menos parametros que tokens en el dataset. Esta forzado a comprimir info en patrones generalizables.)

6. **¿Quien sabe la respuesta correcta durante training, el modelo o el codigo?** (El codigo / training loop. El modelo es ciego — solo procesa input y recibe gradientes.)

Cuando estas seis hagan click solido, ya tienes **el fundamento universal del entrenamiento supervisado**. Lo que viene son detalles arquitectonicos.

---

## Siguiente capitulo

[03 - Gradient descent y autograd](../03-gradient-descent): como el loss se traduce en ajustes a los pesos del modelo.
