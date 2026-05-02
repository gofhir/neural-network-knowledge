---
title: "11 - Modelo XL: 6x mas grande"
weight: 110
math: true
---

Hasta aqui jugamos con el mini-GPT estandar: `d_model=128`, `n_layers=4`, `h=4`, **0.82 M parametros**. Es chiquito a proposito — entrena en un Mac en menos de un minuto y ya ves Shakespeare emergiendo. Pero queda flotando una pregunta inevitable: **que pasa si lo hago mucho mas grande**? No "un poquito mas". Mucho mas. Mismo dataset, mismo numero de iteraciones, misma receta — solo subir capacidad. Eso es lo que hace este experimento.

El script de referencia es `clase_14/practica/08_model_xl.py`. Te recomiendo correrlo despues del estandar (escalon 08) para ver la comparacion en vivo.

---

## 1. La pregunta

¿Que pasa si hacemos el modelo MUCHO mas grande, manteniendo todo lo demas igual?

Mismo dataset (Shakespeare, 1.1 MB). Mismas iteraciones (3000). Mismo learning rate (3e-4). Mismo optimizer (AdamW). Mismo `block_size` (64). **Solo cambian las dimensiones internas de la red.**

Esa es la pregunta central de la decada en deep learning: si todo lo demas es igual y solo subes parametros, **cuanto mejora el modelo**? La respuesta corta es "mucho, y de forma sorprendentemente predecible". La respuesta larga es lo que vamos a ver en este capitulo.

---

## 2. La configuracion

```
ESTANDAR (escalon 8):  d_model=128, n_layers=4, h=4   -> 0.82 M
XL (este experimento): d_model=256, n_layers=6, h=8   -> 4.78 M
```

Tres cambios simultaneos:

- **`d_model` 128 -> 256**: cada token vive en un espacio el doble de ancho. Embeddings mas ricos, mayor capacidad por capa.
- **`n_layers` 4 -> 6**: una pila 50% mas profunda. Cada bloque agrega un "paso de razonamiento" sobre la representacion.
- **`h` 4 -> 8**: el doble de cabezas de atencion. Mas vias paralelas para que el modelo se mire a si mismo desde distintos angulos.
- **`d_ff` 512 -> 1024**: el FFN interno crece a la par con $d_{model}$ (sigue siendo $4 \times d_{model}$).

El conteo combinado da **~4.78 M parametros**, aproximadamente **6x mas** que el estandar. Pero — y esto es importante — el numero de **iteraciones de entrenamiento es el mismo: 3000**. No le damos mas tiempo. No le damos mas datos. Solo mas capacidad.

```python
# 08_model_xl.py
config = dict(d_model=256, h=8, n_layers=6, d_ff=1024, block_size=64)
model = MiniGPT(vocab_size=tokenizer.vocab_size, **config).to(device)
```

{{< concept-alert type="recordar" >}}
"Mismo training, distinto tamaño" es el experimento canonico para medir el efecto puro de la **capacidad** del modelo. Aisla la variable: lo unico que cambia es cuantos parametros hay para aprender los patrones del corpus.
{{< /concept-alert >}}

---

## 3. La curva de loss

```
step     0: loss = 4.43
step   300: loss = 2.21  <- mas rapido al inicio
step   900: loss = 1.68
step  1500: loss = 1.50
step  2400: loss = 1.43
step  2999: loss = 1.39  <- vs 1.63 del estandar
```

Final: **1.39** vs **1.63** del estandar. **El modelo grande aprende mas rapido al inicio y llega mas bajo en el mismo numero de iteraciones.**

Vale la pena traducir esa diferencia a perplexity ($e^{\text{loss}}$):

| modelo | params | loss final | perplexity |
|--------|--------|-----------|------------|
| Estandar | 0.82 M | 1.63 | 5.10 |
| XL       | 4.78 M | 1.39 | 4.01 |

El estandar elige efectivamente entre ~5 caracteres con probabilidad relevante; el XL entre ~4. **No suena dramatico**, pero esa diferencia de 1 caracter de incertidumbre por posicion se acumula a lo largo de centenares de generaciones, y se nota mucho en la calidad cualitativa del texto. Lo vemos en la siguiente seccion.

Tambien fijate en algo del comportamiento: **a step 300 el XL ya esta en 2.21**, mientras que el estandar a step 500 estaba en 2.28. El XL aprende mas rapido por iteracion. Tiene mas capacidad disponible para absorber los patrones desde el primer batch.

---

## 4. La salida real

Las generaciones del modelo XL son **notablemente** mejores. Vamos a mirar tres prompts.

### 4.1 Prompt `ROMEO:`

```
ROMEO:
No, she do I for here lead your piewly:
All know I well be thee revenge, art I
do beseech thee your honour himpedy way and deed here.
Alikes, father lihed their were fore? Nay, by,
All strength'd use to lack and thoughts honour
What he strate we to proget where, by shiends
She they had by the hurtin
```

Notar especialmente: **"art I do beseech thee your honour"**, **"thoughts honour"**. Son frases genuinamente Shakespearianas. "Beseech thee" es un giro tipico del ingles isabelino — no un ingles random, sino el subdialecto especifico que usa Shakespeare. El modelo lo absorbio del corpus.

Tambien el formato esta impecable: nombre en mayuscula + dos puntos + dialogo, exactamente como en el corpus. La distribucion de mayusculas y puntuacion es muy estable.

### 4.2 Prompt `JULIET:`

```
JULIET:
I not think you have will say thou hast love
The manies on of their brother quieen's and hell
And her to perdone the pretty and their fatal and
sullen'd with me, and a man your very his day.
Finds are alute: take the gods of death
To corry prince.

MISTRESS OVERDONE:
We will be lordshed in the downe
```

Hay un detalle aqui que merece detenerse. **Aparece "MISTRESS OVERDONE"** como personaje. **Mistress Overdone es un personaje REAL** — la madama del burdel en *Measure for Measure*, una de las obras de Shakespeare. El modelo no lo invento al azar: aprendio que en el corpus existen personajes con ese nombre, y los introduce como hablantes nuevos.

Eso es importante por una razon sutil. El modelo no tiene una "lista de personajes" como un campo separado. Tampoco le dijimos quien existe en el corpus. Simplemente, durante el entrenamiento, cada vez que aparecia "MISTRESS OVERDONE:" en el texto, el patron se reforzaba. El modelo termino con una distribucion de probabilidad que asigna masa a esa secuencia exacta de caracteres. Cuando muestrea, a veces sale.

**Memorizo nombres del corpus** sin que se lo dijeramos. Es una forma de "recall" emergente.

### 4.3 Prompt `What light through ` (la famosa frase de Romeo)

```
What light through their particular.
I say, afray thee hands of the fair say one.
So money's nay, yet did she we are with me
Come and too leave the showe to person,
And tender which likes here all looks with incannor
And seoul king, I serving to be queen
In the fortune so hards of the remember him:
```

La frase original de Shakespeare es "*But, soft! What light through yonder window breaks?*" — el famoso monologo de Romeo en el balcon. Le damos al modelo el inicio "What light through ", y continua:

- En **estilo poetico apropiado**: cadencia de pentametro yambico, vocabulario de alto registro ("tender", "fortune", "remember him"), invocacion de figuras nobles ("king", "queen").
- Con **palabras inventadas que respetan la morfologia**: "afray", "showe", "incannor", "hards" suenan plausiblemente isabelinas aunque no existan.
- Con **estructura sintactica coherente**: hay verbos, sujetos, modales, conjunciones. Las oraciones tienen forma.

El modelo no esta repitiendo Shakespeare. Esta **continuando** Shakespeare en estilo, con su propio "estilo Shakespeare destilado".

---

## 5. Scaling laws (Kaplan 2020, Chinchilla 2022)

Lo que viste arriba — modelo mas grande, mismo training, mejor loss y mejor calidad — no es un accidente de tu mini-GPT. **Es un fenomeno universal**, formalizado en dos papers que cambiaron como la industria gasta sus dolares.

**Kaplan et al., "Scaling Laws for Neural Language Models" (2020, OpenAI):** demostraron empiricamente que el loss de un Transformer entrenado bien sigue una ley de potencias en funcion de tres variables:

$$\text{Loss} \propto N^{-\alpha} \cdot D^{-\beta} \cdot C^{-\gamma}$$

donde:

- $N$ = numero de **parametros** del modelo.
- $D$ = numero de **tokens** vistos en entrenamiento (datos x epochs).
- $C$ = **compute total** (FLOPs).

Y los exponentes son chicos pero positivos:

- $\alpha \approx 0.07$ a $0.09$ (efecto de mas params).
- $\beta \approx 0.07$ a $0.09$ (efecto de mas datos).
- $\gamma$ similar para compute.

El hecho de que sean **leyes de potencias** (lineales en log-log) significa que **escalar predice mejoras** de forma sorprendentemente regular. Doblas los parametros, el loss baja una cantidad calculable. Doblas otra vez, baja la misma cantidad otra vez. Hasta tamaños muy grandes.

**Chinchilla (Hoffmann et al., DeepMind, 2022)** refino la idea: para un compute fijo $C$, la mejor manera de gastarlo no es solo subir parametros — hay un **balance optimo** entre $N$ y $D$. Concretamente, los modelos pre-Chinchilla (GPT-3 incluido) estaban **sobreparametrizados y subentrenados**. La regla optima de Chinchilla: aproximadamente **20 tokens de entrenamiento por cada parametro**.

Esto explica por que **el campo entero se obsesiono con escalar entre 2018 y 2024**: la formula predice mejoras consistentes, y la geometria del problema esta razonablemente entendida. Si tienes presupuesto $C$, sabes mas o menos donde poner $N$ y $D$ para sacarle el maximo.

```
GPT-2 (1.5B params, 2019)
   |
   v   100x mas params
GPT-3 (175B params, 2020)
   |
   v   ~10x mas params
GPT-4 (~1.76T params estimado, 2023)
```

Cada salto de orden de magnitud trae **capacidades emergentes** (lo vemos en la siguiente seccion).

> Nuestro mini de 0.82 M -> XL de 4.78 M es un mini-salto de ~6x. Y ya viste la diferencia. Imagina escalar **100,000x** mas. Eso es la distancia entre tu XL y un Claude 3 Opus.

{{< concept-alert type="clave" >}}
Las **scaling laws** son la razon economica por la que los frontier labs gastan miles de millones de dolares en entrenar un solo modelo. La curva de retorno es predecible, no especulativa. Si pones $C$ dolares de compute, sabes mas o menos cuanto va a bajar el loss.
{{< /concept-alert >}}

---

## 6. Capacidades emergentes

Esto es lo mas raro y lo mas interesante. A escalas suficientemente grandes (tipicamente >10 B parametros), aparecen **capacidades que no estan presentes en modelos chicos** — no como mejora gradual, sino como aparicion abrupta. La curva de skill vs scale tiene un "codo".

Fenomenos documentados (papers como "Emergent Abilities of Large Language Models", Wei et al. 2022):

- **Few-shot learning / in-context learning**: el modelo resuelve tareas nuevas con solo ver 2-5 ejemplos en el prompt. No es fine-tuning — es inferencia pura. Aparece alrededor de 6-10 B parametros, no antes.
- **Chain-of-thought reasoning**: si pides al modelo "piensa paso a paso", produce razonamiento intermedio que mejora drasticamente la respuesta final en problemas matematicos y logicos. Aparece a partir de cierta escala.
- **Codigo**: escribir y debuggear programas. En modelos chicos no funciona; en modelos grandes (Codex, Copilot, Claude Code) si.
- **Traduccion zero-shot**: traducir entre idiomas sin haber sido entrenado especificamente para traduccion.
- **Matematica simbolica**: resolver ecuaciones, manipular algebra, hacer demostraciones.

**Lo desconcertante**: todo esto **emerge de la misma arquitectura** que tu mini-GPT. Mismo Transformer, mismo objective de cross-entropy de next-token, mismo entrenamiento por gradient descent. **Solo cambia la escala.**

Hay debate cientifico sobre si las "capacidades emergentes" son realmente fenomenos de fase (saltos abruptos) o si son ilusiones de medir con metricas binarias (paper "Are Emergent Abilities a Mirage?", Schaeffer et al. 2023). Pero la observacion practica es solida: cosas que un modelo de 1 B no puede hacer, un modelo de 100 B si.

Tu mini-GPT XL muestra una version de esto a escala diminuta — el salto de 0.82 M a 4.78 M ya cambia la calidad del texto de "casi-Shakespeare" a "Shakespeare reconocible". Imagina ese salto repetido 5 veces mas.

---

## 7. Pausa de verificacion

Antes de seguir, asegurate de que tienes claro tres puntos:

1. **¿Por que el modelo XL llega a loss mas bajo en el mismo numero de iteraciones?** Pista: capacidad. El modelo grande tiene mas parametros disponibles para absorber regularidades del corpus. Cada batch le aporta mas informacion utilizable. Por iteracion, baja mas rapido.

2. **¿Que predicen las "scaling laws"?** Pista: ley de potencias. El loss escala como $N^{-\alpha}$, $D^{-\beta}$, $C^{-\gamma}$, con exponentes chicos pero positivos. Doblar parametros (o datos, o compute) baja el loss en una cantidad consistente y predecible.

3. **¿Que diferencia hay con los LLMs comerciales actuales?** Pista: ~6 ordenes de magnitud. Tu XL tiene 4.78 M parametros; GPT-3 tiene 175 B. Eso es **un factor de ~37,000**. Y esta entrenado con ~570,000 veces mas datos. Misma arquitectura. Solo escala.

---

## 8. Lo que aprendimos

Tres conclusiones para llevarte:

- **Mismo training, mas parametros = mejor modelo.** No siempre. Pero hasta cierto punto, casi siempre. Y la mejora es sorprendentemente predecible.
- **La calidad cualitativa cambia**, no solo el numero. Pasamos de "casi-Shakespeare" a "Shakespeare reconocible con personajes reales". Esa transicion no es continua; tiene saltos cualitativos.
- **El secreto de los LLMs modernos no es magia, es escala**. Misma arquitectura que tu mini-GPT — mismo Transformer, mismo objective, mismo entrenamiento. Lo que cambia son los ceros: en parametros, en datos, en compute.

> Si tu XL de 4.78 M te impresiona, recuerda: estas a 6 ordenes de magnitud de la frontera. Cada orden de magnitud trae nuevas capacidades. **No hay razon teorica conocida por la que la curva se aplane**. Por eso el campo sigue escalando.

---

## Codigo y referencias

Codigo: `clase_14/practica/08_model_xl.py`

Referencias:

- Kaplan et al., **"Scaling Laws for Neural Language Models"** (2020, OpenAI).
- Hoffmann et al., **"Training Compute-Optimal Large Language Models"** ("Chinchilla", 2022, DeepMind).
- Wei et al., **"Emergent Abilities of Large Language Models"** (2022, Google).
- Schaeffer et al., **"Are Emergent Abilities of Large Language Models a Mirage?"** (2023).

Siguiente: [12 - Don Quijote](../12-dataset-quijote).
