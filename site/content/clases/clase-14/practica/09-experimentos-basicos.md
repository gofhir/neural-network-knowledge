---
title: "09 - Experimentos basicos: temperatura, prompts, modelo micro"
weight: 90
math: true
---

Hasta aqui entrenamos un mini-GPT y vimos que produce algo "Shakespeare-ish". Pero falto una parte esencial: **jugar con el modelo**. Manipularlo. Pasarle distintos prompts, subir y bajar la temperatura, comparar contra una version chica para ver cuanto se gana con capacidad. Esa fase exploratoria es donde se construye la intuicion de **que palancas controlan la generacion** en un modelo de lenguaje. Las mismas palancas que usas en Claude, GPT, Gemini, todas. Solo que aqui las podemos mover sobre un modelo cuyas tripas conocemos por dentro.

El script de referencia es `clase_14/practica/06_experimentos.py`. Toma alrededor de 2-3 minutos en MPS. Entrena dos modelos de tamaños distintos y luego corre tres bloques de experimentos sobre ellos.

---

## 1. La fase 4: experimentos sobre el mini-GPT

El script entrena **dos** mini-GPT en paralelo y luego los inspecciona:

- **MICRO**: `d_model=32`, `n_layers=2`, `h=2`, `d_ff=128`. Total: **31,425 parametros**.
- **ESTANDAR**: `d_model=128`, `n_layers=4`, `h=4`, `d_ff=512`. Total: **816,065 parametros**.

Ambos entrenan **2000 iteraciones** sobre Tiny Shakespeare con el mismo learning rate (3e-4), el mismo batch size (32) y el mismo block size (64). Lo unico que cambia entre los dos es la **capacidad** del modelo. El estandar es **26x mas grande** que el micro.

Con esos dos modelos en mano, el script corre tres experimentos:

1. **Temperatura**: mismo modelo, mismo prompt, distinta temperatura de sampling. Como cambia la generacion?
2. **Prompts**: mismo modelo, distintos prompts. Como condiciona el contexto la salida?
3. **Tamaño**: mismo prompt, distintos modelos. Cuanto importa la capacidad?

Cada experimento aisla **una variable**. Es la forma honesta de medir efectos: si cambias varias cosas a la vez, no sabes a que atribuir la diferencia.

---

## 2. Experimento 1: efecto de la TEMPERATURA

La temperatura aparece en una sola linea del codigo de `generate`:

```python
logits = logits[:, -1, :] / temperature
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
```

Dividir los logits por $T$ antes del softmax cambia la forma de la distribucion de probabilidades:

- **$T < 1$**: los logits crecen en magnitud relativa, el softmax se vuelve mas "pico". El sampling se concentra en el caracter mas probable. Texto **mas determinista, mas seguro, mas repetitivo**.
- **$T = 1$**: distribucion natural del modelo, sin manipular.
- **$T > 1$**: los logits se aplanan, el softmax se acerca a uniforme. Mas variedad de caracteres ganan probabilidad. Texto **mas creativo, con mas errores tipograficos, palabras inventadas**.

Para verlo en vivo, fijamos seed = 42 y prompt = `"ROMEO:"`. Variamos solo $T$. El modelo es el ESTANDAR.

### 2.1 Temperatura 0.5 (conservadora)

```
ROMEO:
What is the more the staring of the senerey,
And the seet the seet the seet the seet the seet
And the seet the seet the seet
```

Estructura clara, pero el modelo cae en **bucles**. La frase "the seet" se repite porque a $T=0.5$ el caracter mas probable gana con muchisima ventaja, y una vez que entras en un patron periodico no sales. La salida es "segura" pero **patologicamente determinista**.

### 2.2 Temperatura 1.0 (natural)

```
ROMEO:
And see the heart your wounds we'll be the strights
That were the more boy he was thee with my chief and
What hath you to thee a duke?
```

Texto **balanceado**. Conserva estructura (mayusculas, signos, cadencia poetica) pero introduce variacion lexica. "Strights" es palabra inventada, pero plausible isabelina. Las preguntas con signo apropiado, los "thee" y "thou" en su lugar. Esta es la temperatura por defecto cuando el modelo "respira normal".

### 2.3 Temperatura 1.5 (creativa, ruidosa)

```
ROMEO:
What gxtad heart ye glowTbid Calas swer'k
joy! ofy that, ce shis muunchy to-y,
And foll wuth aged lear of my lipt foullen
```

Aqui el ruido se nota. Aparecen monstruos lexicos: **"gxtad"**, **"glowTbid"**, **"Calas"**, **"swer'k"**, **"muunchy"**. El modelo todavia respeta vagamente la estructura, pero los caracteres improbables (mayusculas en medio de palabras, combinaciones imposibles tipo "gx", "Tb") ahora reciben suficiente probabilidad como para colarse en el muestreo.

### 2.4 Resumen de la palanca

```
T = 0.5  ->  determinista, repetitivo, bucles
T = 1.0  ->  natural, balanceado
T = 1.5  ->  creativo, errores tipograficos, palabras inventadas
```

En la practica con LLMs comerciales, **$T \in [0.7, 1.0]$** es lo tipico para creative writing. **$T \in [0, 0.3]$** para tareas que requieren precision (codigo, traduccion, extraccion estructurada). $T = 0$ desactiva el muestreo completamente y produce greedy decoding determinista.

---

## 3. Experimento 2: PROMPTS variados

Con el mismo modelo (ESTANDAR) y la misma temperatura ($T = 0.8$), el script prueba cinco prompts distintos. La idea es ver como **el contexto inicial condiciona** todo lo que sigue. Es **in-context learning en miniatura**: el modelo no se reentrena, pero sus primeras predicciones cambian dependiendo de lo que tiene a la izquierda.

### 3.1 Prompt `"ROMEO:"`

```
ROMEO:
And so see the heart you well have toon dispute
With the fortune so to be a strange...
```

El modelo aprendio que despues de un nombre en mayusculas seguido de `:` viene un **parlamento poetico**. Continua en cadencia de pentametro yambico, vocabulario de alto registro.

### 3.2 Prompt `"JULIET:"`

```
JULIET:
I would not say the world had her too much
And let me see the way of love in this...
```

Mismo patron estructural (parlamento), pero observa que cambia el **tono**. Las palabras "love", "world", "see" aparecen con mas frecuencia. El modelo aprendio una distribucion condicional ligeramente distinta para Juliet que para Romeo, porque sus parlamentos en el corpus son sistematicamente diferentes.

### 3.3 Prompt `"To be or not to "`

```
To be or not to be the more they here,
And what is more than to know the heart...
```

Notar: **el modelo completa "to be or not to be"** correctamente. No por memorizacion exacta, sino porque la secuencia es lo suficientemente frecuente en Shakespeare como para que las probabilidades condicionales hagan que la "b" sea muy probable despues de "to ", y la "e" muy probable despues de "to b". El completion canonico emerge de la estadistica.

### 3.4 Prompt `"HAMLET:\nO that this too too solid "`

```
HAMLET:
O that this too too solid heart with a heart
Of love that is the heart of his soul...
```

El prompt continua una de las frases mas famosas del soliloquio de Hamlet ("O that this too too solid flesh would melt"). El modelo no produce **"flesh"** exactamente, pero **mantiene el registro retorico**: "heart", "soul", construcciones de genitivo encadenadas ("heart of his soul"). Es un Hamlet plausible que diverge del original.

### 3.5 Prompt `"Friends, Romans, "`

```
Friends, Romans, and the country shall be so
That we shall not be the man of the death...
```

Frase de apertura del discurso de Marco Antonio en *Julius Caesar* ("Friends, Romans, countrymen, lend me your ears"). El modelo no completa con "countrymen", pero produce un **discurso politico** con vocabulario apropiado: "country", "death", "man". El estilo cambia respecto a un parlamento de Romeo. Es prosa retorica, no lirica.

### 3.6 La leccion

**El prompt no es solo el comienzo de la generacion. Es la condicion bajo la cual se calcula todo lo que sigue.** Cada caracter futuro depende de la distribucion $P(x_{t+1} \mid x_1, \ldots, x_t)$, y esa distribucion cambia drasticamente segun el contexto inicial.

Esto es exactamente lo que un LLM moderno hace cuando le das un system prompt: condiciona toda la cascada de probabilidades. **No hay "memoria interna" especial**: solo el contexto y los pesos.

---

## 4. Experimento 3: MICRO (31K) vs ESTANDAR (816K)

Aqui el experimento se vuelve dramatico. Mismo prompt (`"ROMEO:\n"`), mismo seed (42), misma temperatura (0.8). Lo unico que cambia es **cual de los dos modelos genera**.

### 4.1 MICRO — 31,425 parametros

```
ROMEO:
Lroy ther cou Custhau cus al chl mathaterd
hou aler tot fer mat e the the t hath
to a cor my mal be the s me
```

El modelo MICRO **aprendio algo, pero poco**. Captura:

- Mayuscula despues del salto de linea ("Lroy").
- La presencia de "the" como palabra frecuente.
- Espacios entre tokens.
- Vagamente, longitudes de palabra cortas-medianas.

Pero produce **ensaladas de letras**: "Custhau", "mathaterd", "chl", "tot fer". El modelo no tiene capacidad suficiente para memorizar el vocabulario isabelino. Sus 31 mil parametros se reparten entre embeddings, atencion, FFN y output, y cada uno queda demasiado debil para representar la riqueza del corpus.

### 4.2 ESTANDAR — 816,065 parametros

```
ROMEO:
Lroy, I know you, and come thinle a hate;
Hencome so off it king, and I will be the man
Of the world that is the heart of my love.
```

El modelo ESTANDAR, con **26x mas parametros**, produce algo radicalmente distinto:

- **Vocabulario reconocible**: "I know you", "come", "hate", "king", "man", "world", "heart", "love".
- **Sintaxis completa**: oracion con sujeto, verbo modal ("I will be"), preposiciones encadenadas correctamente.
- **Puntuacion correcta**: comas en pausas internas, punto y coma para transicion.
- **Solo dos palabras inventadas**: "thinle" y "Hencome". El resto es ingles legible.

La diferencia no es de grado. Es de **categoria**. El MICRO produce ruido con estructura; el ESTANDAR produce ingles isabelino reconocible con detalles equivocados.

### 4.3 Cuanta capacidad necesitas?

La pregunta es importante porque cuesta dinero. Cuantos parametros minimos necesitas para que el modelo deje de balbucear y empiece a generar lenguaje?

Empiricamente, en datasets de tamaño Tiny Shakespeare:

- **< 50 K params**: balbuceo, captura solo regularidades superficiales (mayusculas, espacios, frecuencia de "the").
- **50 K - 500 K params**: estructura, pero vocabulario pobre y ensalada lexica.
- **500 K - 5 M params**: ingles legible con errores menores. Aqui esta tu mini-GPT estandar.
- **5 M - 50 M params**: estilo Shakespeare reconocible con detalles realistas. Personajes correctos, pentametro estable.
- **> 50 M params**: para Tiny Shakespeare, ya empieza el overfitting porque el corpus es solo 1.1 MB.

Tu mini de 816 K esta justo en el sweet spot para este corpus.

---

## 5. Las 3 palancas que controlas en LLMs

Lo que viste son **las tres palancas universales** de generacion en modelos autoregresivos. Cada una tiene su uso:

### 5.1 Temperatura — controla el sampling

- Bajar para **precision** (codigo, traduccion, extraccion estructurada).
- Subir para **creatividad** (poesia, brainstorming, ficcion).
- $T = 0$ -> greedy, determinista (mismo input siempre da mismo output).
- $T \in [0.7, 1.0]$ -> default razonable.
- $T > 1.5$ -> casi siempre demasiado ruidoso para uso practico.

### 5.2 Prompts — controla el contexto

- El prompt **es** la programacion del modelo en runtime.
- Cambiar el prompt cambia la distribucion condicional de todos los tokens siguientes.
- Lo que llamamos "prompt engineering" es exactamente eso: encontrar el contexto que hace que la cadena de probabilidades caiga donde quieres.
- En modelos grandes ($\geq 6B$ params) el prompt habilita **few-shot learning**: pones 3 ejemplos y el modelo generaliza. En tu mini-GPT eso no pasa — no hay capacidad suficiente.

### 5.3 Tamaño — controla la capacidad

- Mas parametros -> mas patrones absorbidos -> texto mas reconocible.
- La curva no es lineal; tiene **saltos cualitativos** (ver el escalon 11 sobre scaling laws).
- Esta es la palanca que **no puedes mover en runtime**. Se elige cuando entrenas.
- En LLMs comerciales, tu eliges entre Haiku/Sonnet/Opus, GPT-4o-mini/GPT-4o, etc. Esa eleccion es esta palanca.

Las tres palancas combinadas son lo que define la experiencia de usar un LLM. Tu mini-GPT te dejo manipular las tres explicitamente. En un LLM cerrado, dos las controlas tu (temperatura, prompts) y una la elige el proveedor (modelo y tamaño, aunque tu pagas por el).

---

## 6. Pausa de verificacion

Antes de seguir, asegurate de tener clara la intuicion sobre las tres palancas:

1. **¿Por que con temperatura 0.5 el modelo cae en bucles?**
   Pista: a baja temperatura, el caracter mas probable gana con mucha ventaja. Una vez que el modelo entra en un patron donde "X -> Y -> X -> Y" es la cadena mas probable, no hay suficiente ruido para escapar. Por eso "the seet the seet the seet" se repite indefinidamente.

2. **¿Que diferencia hay entre cambiar el prompt y cambiar los pesos del modelo?**
   Pista: el prompt cambia la distribucion **condicional** $P(x_{t+1} \mid \text{contexto})$ sin alterar el modelo. Los pesos definen la funcion completa. En inference, solo puedes mover el contexto. Eso es prompt engineering. Para mover los pesos hay que reentrenar (fine-tuning).

3. **¿Por que el MICRO (31 K params) produce balbuceo y el ESTANDAR (816 K) produce ingles?**
   Pista: capacidad. El MICRO no tiene parametros suficientes para representar el vocabulario completo del corpus ni las dependencias largas de Shakespeare. Aprende solo lo mas frecuente y superficial. El ESTANDAR tiene capacidad para memorizar patrones de varias palabras y recombinarlos. Es el mismo principio que separa un Haiku de un Opus, escalado abajo.

4. **¿Que pasa si subes la temperatura del MICRO a 1.5?**
   Pista: el balbuceo se vuelve aun mas catastrofico. La temperatura amplifica el ruido del modelo subyacente. Si el modelo ya es debil, mas temperatura no lo hace mejor — lo hace peor.

---

## 7. Lo que aprendiste en este escalon

Tres ideas concretas para llevarte:

- **La temperatura es una manija de creatividad vs precision.** Bajita para tareas precisas, alta para creativas. Encima de 1.5 casi siempre rompes la salida.
- **El prompt es el lenguaje real con el que programas un LLM.** No hay setting interno mas poderoso que el contexto inicial. Cambiar el prompt cambia todo lo que sigue.
- **El tamaño define el techo del modelo.** Un MICRO de 31K nunca va a generar Shakespeare reconocible, por mas que ajustes temperatura o prompt. La capacidad la fijaste cuando elegiste la arquitectura.

Estas tres palancas son las mismas que tienes con cualquier LLM comercial. Lo unico distinto es que aqui las moviste sobre un modelo cuya estructura entendes completamente: 4 bloques transformer, 4 cabezas, 128 dimensiones internas. No es magia. Es estadistica condicional con muchos parametros.

---

## Codigo y referencias

Codigo: `clase_14/practica/06_experimentos.py`

Siguiente: [10 - Train longer](../10-train-longer).
