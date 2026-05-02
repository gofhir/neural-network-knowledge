---
title: "12 - Cambiar el dataset: Don Quijote en español"
weight: 120
math: true
---

Hasta ahora todo lo que hemos hecho fue con texto de Shakespeare. Pero hay una pregunta que vale la pena tomarse en serio: ¿el Transformer es realmente universal, o esta sutilmente "casado" con el ingles isabelino? La unica forma honesta de responder esto es **cambiar el dataset y no tocar nada mas**. Si el modelo aprende otro idioma con la misma arquitectura, los mismos hyperparametros y el mismo codigo — entonces la universalidad no es una promesa de marketing, es algo que vimos con nuestros propios ojos.

El script de este capitulo es `clase_14/practica/09_dataset_quijote.py`. La unica cosa que cambia respecto al mini-GPT del escalon 08 es de donde se baja el texto. Todo lo demas — `MiniGPT`, `CharTokenizer`, `train`, `sample` — es el mismo codigo, importado tal cual desde `_models.py`.

---

## 1. El experimento

¿Y si cambiamos COMPLETAMENTE el dataset? Misma arquitectura, mismo codigo, mismos hyperparametros — pero entrenamos en **Don Quijote** (Cervantes, español del Siglo de Oro) en vez de Shakespeare (ingles isabelino).

La idea es minimalista a proposito. No vamos a tocar el modelo. No vamos a cambiar el tokenizador a algo "mejor para español". No vamos a ajustar el learning rate ni la cantidad de capas. Solo cambia el archivo de texto que entra. Si el modelo aprende, la conclusion es directa: **lo unico que el Transformer "ve" del mundo es la secuencia de tokens**, y le da exactamente igual de donde vienen.

---

## 2. La hipotesis

Si el Transformer es **universal**, deberia aprender español sin que cambiemos una linea de codigo del modelo. **Solo cambia el input.**

Esta hipotesis es fuerte: implica que la arquitectura no codifica conocimiento sobre el ingles, ni sobre el alfabeto latino "estandar", ni sobre los espacios entre palabras como los conocemos. Lo que codifica son **mecanismos generales** — atender a tokens del pasado, aprender embeddings desde gradientes, refinar representaciones capa por capa. Todo lo concreto sobre el lenguaje sale del corpus.

Si la hipotesis fuera falsa, esperariamos ver que el modelo se queda atascado, que el loss no baja como con Shakespeare, o que la salida nunca llega a parecer español. Vamos a ver que pasa.

---

## 3. La data

Don Quijote desde Project Gutenberg (`https://www.gutenberg.org/cache/epub/2000/pg2000.txt`). Tomamos los primeros 500K caracteres para que sea comparable con tinyshakespeare (~1.1MB).

```python
QUIJOTE_URL = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
LOCAL = "quijote.txt"
if not os.path.exists(LOCAL):
    urllib.request.urlretrieve(QUIJOTE_URL, LOCAL)

with open(LOCAL, "r", encoding="utf-8") as f:
    text = f.read()

# Limpieza basica del header de Gutenberg
if "*** START" in text:
    text = text.split("*** START", 1)[1].split("\n", 1)[1]
if "*** END" in text:
    text = text.split("*** END", 1)[0]

text = text[:500_000]
```

500K chars es la mitad de tinyshakespeare. Lo elegimos asi para que sea comparable y entrenable en pocos minutos en una Mac. Don Quijote completo son ~2 MB, asi que estamos usando aproximadamente la primera parte (mas o menos hasta los molinos de viento).

```
Vocab size: 86 caracteres
Caracteres unicos: '\n !"\'(),-.0146:;?ABCDEFGHIJLMNOPQRSTUVWXYZ]abcdefghijlmnopqrstuvxyz¡«»¿ÁÉÍÑÓÚáéíïñóúü—'
```

Notar la presencia de **caracteres especificos del español**: `áéíóú¡¿ñ«»—`. Sin que nadie le diga, el tokenizer detecta y maneja estos caracteres. El `CharTokenizer` simplemente hace `sorted(set(text))`: lo que aparece en el corpus, va al vocabulario. Si hubiera japones, entrarian los kanji. Si hubiera emojis, entrarian los emojis. El tokenizer es agnostico al alfabeto.

{{< concept-alert type="recordar" >}}
El vocabulario crece de 65 (Shakespeare) a 86 (Quijote) porque el español tiene mas caracteres distintos: tildes, eñe, signos de apertura `¡¿`, comillas latinas `« »`, raya de dialogo `—`. El modelo no los conoce a priori — los va a tener que aprender desde cero, igual que aprendio las letras del ingles.
{{< /concept-alert >}}

---

## 4. La curva de loss

Misma config que en Shakespeare: `d_model=128, h=4, n_layers=4, d_ff=512, block_size=64, batch_size=32, lr=3e-4`. Mismas 3000 iteraciones.

```
step     0: loss = 4.62
step   300: loss = 2.31
step   900: loss = 1.96
step  1500: loss = 1.71
step  2100: loss = 1.61
step  2999: loss = 1.52
```

Loss inicial: $-\log(1/86) \approx 4.45$, un poco mayor que el de Shakespeare (4.17) porque el vocabulario es mas grande. Loss final: **1.52**, muy similar al de Shakespeare. **La arquitectura es indiferente al idioma.** Solo aprende patrones del corpus.

La curva tiene exactamente la misma forma que con Shakespeare: caida abrupta en los primeros cientos de pasos (aprende la distribucion marginal de caracteres), luego descenso mas suave (aprende secuencias y palabras), luego retornos decrecientes (aprende detalles finos del estilo). El modelo no "sabe" que cambio de idioma. Solo ve secuencias y se ajusta.

---

## 5. La salida real

**Prompt vacio:**
```
cuánto mercede a dar que mala buena de las puedias alcasiguas de adonantes? Pero con la
consólera que la conertad, el cual, mal dio que el pasago a hicdo que
el tato había caballero, a la porque los imos y a mozces que no le
confilino, era que me la micho tan bien alcancialo:

— Hice del haré de aco
```

Notar: aparecen palabras españolas: `cuánto`, `mala buena`, `caballero`, `confilino`. Y la **estructura de dialogo con guion largo** `—`. El modelo aprendio que el español usa `—` para marcar parlamentos (en lugar del `:` tras nombre como Shakespeare). Eso lo capto solo, sin que nadie le explique la convencion tipografica.

Las tildes aparecen en lugares plausibles (`cuánto`, `consólera`, `había`). Las palabras inventadas como `confilino` o `alcasiguas` tienen morfologia castellana correcta — terminaciones en `-ino`, `-aguas`, distribucion de vocales y consonantes que suena a español.

**Prompt 'En un lugar de ' (la frase mas famosa del Quijote):**
```
En un lugar de las sobres
El viérele la ropíse, y, viendo mesmo silano que de aquel pasaran en
son su abrito de la venira, y aquel muy dieron que sobre,
que se hacían tanto a murce de su muy será de monos de la pasadura
```

Continua en estilo barroco: `viendo mesmo silano`, `abrito de la venira`. Aunque el modelo no sabe español "real", capturo la **musica del lenguaje del Siglo de Oro**. La palabra `mesmo` (forma arcaica de "mismo") aparece naturalmente — el modelo la encontro mil veces en el corpus y la reproduce. La cadencia de las frases largas separadas por comas, con varios complementos antes del verbo, es tipica de Cervantes.

**Prompt 'Don Quijote ':**
```
Don Quijote que altrón trato,
acomodo bocargando facarla que buscanse la hombridad hacicia del mesmo contición.
... — Pero don Quinote con las castado poco le haya Sancho, son de su comundo en
la caba
```

¡El modelo escribe `Sancho`! Aprendio que `Sancho` co-ocurre con `Don Quijote`. Esto es notable: nadie le dijo "Sancho es el escudero de Don Quijote". Lo dedujo de la estadistica de co-ocurrencia. En las primeras 500K chars de la novela, Sancho aparece muchisimas veces cerca del nombre del protagonista, y el modelo internalizo esa asociacion. Es la base de todas las "memorias asociativas" de los LLMs grandes.

Tambien notar `don Quinote` con falta de ortografia — el modelo no es perfecto, confunde `j` con `n`. Pero la idea de "despues de don, viene un nombre propio que empieza con Q" la tiene.

**Prompt 'Sancho dijo:':**
```
Sancho dijo:

— ¿Capíta yo que le su
caballeros de las pechas peligastas, lleguó por él, diciéndole el cauto con
la cueria que, y de sientimos en la quitieran al la ha del aquella
gerarde los otros bius; tes que fuérmos, famos, dijo:

— Como quiero si se lo quiero todo esto le puedes desbiado lo mis menos.
```

Dialogo entre personajes con `—`. Aparecen palabras españolas: `caballeros`, `peligastas`, `quiero`. Y algo aun mas fino: el modelo abre con `¿` (signo de interrogacion espanol), y luego pone otra raya `—` para introducir un segundo parlamento. Es decir, internalizo que "Sancho dijo:" es seguido de una replica, y que las replicas en español usan raya inicial. La estructura tipografica del dialogo cervantino, capturada gratis.

---

## 6. La leccion: universalidad

> **El Transformer es indiferente al idioma o tipo de texto.** Le pones Shakespeare → aprende ingles. Le pones Don Quijote → aprende español. Le pones codigo Python → aprenderia codigo. La arquitectura no tiene "preferencia" por ningun lenguaje.

Eso es lo que permite que **el mismo modelo arquitectonico** entrene:

- ChatGPT (multiples idiomas)
- Claude (cientos de idiomas)
- AlphaFold (aminoacidos)
- Codex (codigo)
- Whisper (audio)
- ViT (imagenes)

> Mientras puedas convertir tu dominio a una secuencia de tokens, el Transformer funciona.

Esta universalidad no es accidental — es consecuencia directa de como esta diseñada la arquitectura. La self-attention solo necesita que sus inputs sean vectores; le da igual si esos vectores codifican letras, parches de imagen, frecuencias de audio o residuos de proteina. Las matrices Q, K, V se aprenden desde gradientes, no estan "preconfiguradas" para ningun dominio.

{{< concept-alert type="clave" >}}
La universalidad del Transformer es el unico motivo por el que tiene sentido hablar de "modelos fundacionales": un solo modelo que se adapta a multiples dominios. Si la arquitectura estuviera optimizada para ingles, no podria entrenarse en codigo. Si estuviera optimizada para texto, no podria entrenarse en imagenes. La indiferencia al input es la feature, no un bug.
{{< /concept-alert >}}

---

## 7. Que pasaria con dataset mezclado

Si entrenas en Shakespeare + Don Quijote + Wikipedia espanola + codigo Python juntos, el modelo aprende los **4 estilos** y los puede generar segun el prompt. Eso es exactamente como funcionan los LLMs reales: entrenados con texto MULTILINGUE y MULTI-DOMINIO.

El truco es que el modelo aprende a usar el **prompt como condicionante de estilo**. Si le das un prompt que parece codigo Python (`def `), va a continuar generando codigo. Si le das un prompt que parece dialogo cervantino (`Sancho dijo:`), va a continuar en español del Siglo de Oro. La misma red, el mismo conjunto de pesos, distintos modos de operacion segun el contexto.

Esto se llama **in-context learning** y es la propiedad que hace que GPT-3 y modelos posteriores puedan resolver tareas nuevas sin reentrenamiento. El "switch de modo" lo hace el prompt, no un cambio en los parametros.

---

## 8. Pausa de verificacion

1. ¿Cuantas lineas de codigo del modelo cambiamos para entrenar en español?
2. ¿Que es lo unico que hace falta para que el modelo "aprenda" un nuevo dominio?
3. Si entrenaramos en codigo Python, ¿que aprenderia el modelo?

---

Codigo: `clase_14/practica/09_dataset_quijote.py`

Siguiente: [13 - GELU vs ReLU](../13-gelu-vs-relu).
