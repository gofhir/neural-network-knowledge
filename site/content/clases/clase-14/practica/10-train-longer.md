---
title: "10 - Entrenar mas tiempo (6000 iteraciones)"
weight: 100
math: true
---

Hasta este capitulo siempre paramos en **3000 iteraciones** — el escalon 8, el "estandar" de toda la serie. Es un numero pragmatico: entrena en pocos minutos, llega a un loss razonable y produce texto que se reconoce como Shakespeare. Pero hay una pregunta evidente que no respondimos: **¿que pasa si no paramos? ¿que pasa si lo dejamos correr el doble**? Esa es la pregunta de este experimento. Mismo modelo, misma data, misma receta — solo cambiamos `max_iters=3000` por `max_iters=6000`.

El script de referencia es `clase_14/practica/07_train_longer.py`. Es literalmente el escalon 8 con un solo numero modificado.

---

## 1. La pregunta: ¿y si entrenamos el doble?

Cuando frenamos en step 3000 estamos tomando una decision implicita: "asumo que el modelo ya extrajo la mayor parte de los patrones del corpus, y seguir entrenando no vale la pena". Esa decision es **arbitraria**. La fijamos al inicio para que el experimento sea reproducible y rapido, no porque el optimizador hubiera convergido.

Si miras la curva de loss en el escalon 8, no esta plana. Sigue bajando. Lentamente, pero baja. Eso significa que **hay rendimiento que estamos dejando sobre la mesa**. Cuanto exactamente? La unica forma de saberlo es correr el experimento.

Asi que tomamos el modelo identico — `d_model=128, h=4, n_layers=4, d_ff=512`, los mismos **0.82 M parametros** — y lo entrenamos por **6000 iteraciones** en lugar de 3000. Mismo dataset, mismo learning rate, mismo optimizer, mismo `block_size`. La unica variable que cambia es el tiempo de entrenamiento.

```python
# 07_train_longer.py
config = dict(d_model=128, h=4, n_layers=4, d_ff=512, block_size=64)
model = MiniGPT(vocab_size=tokenizer.vocab_size, **config).to(device)
train(model, get_batch, max_iters=6000, label="modelo entrenado 6000 iters")
```

Este es el experimento mas barato del set: no cambias arquitectura, no cambias hiperparametros, no cambias datos. Solo dejas el loop corriendo el doble.

{{< concept-alert type="recordar" >}}
"Mismo modelo, mas iteraciones" aisla la variable **tiempo de entrenamiento**. Cualquier mejora que veas viene exclusivamente del optimizador habiendo hecho mas pasos sobre el loss landscape — no de mas capacidad ni mas datos.
{{< /concept-alert >}}

---

## 2. La curva de loss real

Esto es lo que produce el script al correr en CPU/MPS:

```
step     0: loss = 4.31
step   500: loss = 2.18
step  1000: loss = 1.92
step  2000: loss = 1.74
step  3000: loss = 1.62  <- aqui hubieramos parado en escalon 8
step  4000: loss = 1.55
step  5000: loss = 1.51
step  6000: loss = 1.49
```

Final: **1.49** vs **1.62** del corte estandar. Doble de iteraciones, **0.13 menos de loss**. ¿Es mucho o poco?

Traducido a perplexity ($e^{\text{loss}}$):

| corte | iters | loss | perplexity |
|-------|-------|------|------------|
| Estandar | 3000 | 1.62 | 5.05 |
| Largo    | 6000 | 1.49 | 4.44 |

A step 3000 el modelo elige efectivamente entre ~5 caracteres con probabilidad relevante; a step 6000 entre ~4.4. Una diferencia de **medio caracter de incertidumbre por posicion**. Pequeña por cada token, pero acumulada a lo largo de 200 generaciones cambia notoriamente la coherencia del texto.

Tambien fijate en la **forma de la curva**: la mayor parte del progreso ocurre en las primeras 1000 iteraciones (de 4.31 a 1.92). Despues la curva se aplana, y entre 3000 y 6000 solo bajamos 0.13. Esto es **rendimientos decrecientes**, exactamente lo que predicen las scaling laws para datos/compute fijos.

---

## 3. Las generaciones reales

Vamos a mirar dos prompts.

### 3.1 Prompt `ROMEO:`

```
ROMEO:
For that seen, what stands your nighKing,
Hake my free abless to should with his gentle
A revenge comes of in pression:
```

Detalles que vale la pena resaltar:

- **"what stands your"** es una construccion sintactica isabelina valida ("what does your X stand for"). El modelo aprendio el patron interrogativo de Shakespeare.
- **"my free abless"**, **"his gentle"**, **"A revenge comes"** — frases nominales con la estructura adjetivo + sustantivo correctamente formateadas.
- **"pression"** y **"abless"** son palabras inventadas, pero respetan la morfologia inglesa: terminacion en `-ion`, prefijo `a-`. Suenan plausiblemente shakespearianas aunque no existan.
- El formato dialogo (nombre + dos puntos + linea nueva) esta impecable. Eso es estable desde el escalon 8, no es ganancia nueva.

Lo que cambio respecto a 3000 iteraciones es **la densidad de "casi-palabras" reales**. Hay menos basura puramente aleatoria. Mas pedazos de oracion que casi tienen sentido.

### 3.2 Prompt `JULIET:\nO Romeo, `

Aqui aparece algo mas interesante:

```
JULIET:
O Romeo, what hath the man and farewell.

CLARENCE:
The vault is now and beauty have no more.

BRUTUS:
By that hour we know to die our heads.

DUKE OF EDWARD:
Speak on, my lord, in the field of arms.
```

Tres detalles cualitativos:

- **CLARENCE, BRUTUS, DUKE OF EDWARD** son **personajes reales del corpus**. Clarence aparece en *Richard III*. Brutus es el conspirador de *Julius Caesar*. "Duke of Edward" es una variante posible de los multiples Dukes en las obras historicas (*Henry VI*, *Edward IV*, etc).
- **El modelo memorizo nombres** sin que se lo dijeramos. Cada vez que entrenamos sobre "BRUTUS:" en el corpus, el patron se reforzo. Ahora, cuando muestrea, asigna probabilidad alta a esa secuencia exacta de caracteres en posicion de hablante.
- **"By that hour we know to die our heads"** — sintacticamente coherente, vocabulario tragico apropiado, cadencia que recuerda al pentametro. No tiene sentido literal pero **suena a Shakespeare**.

Comparando con la version de 3000 iteraciones: alli el modelo a veces inventaba nombres que no existian ("HOREEK:", "MARCIBAL:"). A 6000 iteraciones la frecuencia de nombres reales sube. **El modelo se acerca mas a la distribucion de personajes real del corpus.**

---

## 4. Por que mas iteraciones funciona

La intuicion mecanica: **gradient descent es un proceso estocastico**. Cada batch es una muestra ruidosa del corpus, y cada paso de actualizacion mueve los pesos en una direccion aproximada. Con 3000 pasos el optimizador ya bajo a una region de loss bajo, pero **dentro de esa region hay subregiones aun mas bajas** que no exploro todavia.

Tres efectos especificos suceden al pasar de 3000 a 6000 iteraciones:

1. **Mas exploracion del loss landscape.** Cada minibatch perturba los pesos y los empuja en una direccion ligeramente distinta. Mas iteraciones = mas oportunidades de encontrar mejores valles. La superficie de loss en redes neuronales tiene muchos minimos locales casi-equivalentes; mas pasos suelen llegar a uno mejor.

2. **Mas vueltas sobre los mismos datos.** El corpus de Shakespeare es pequeño (1.1 MB, ~300K tokens en BPE; en char-level mas). A 3000 iters con `batch_size=32` y `block_size=64`, el modelo vio aproximadamente 6 M tokens — **muchas pasadas por el corpus completo**. A 6000 iters duplicas eso. Cada pasada adicional refina patrones que solo aparecen en contextos infrecuentes. Por eso aparecen personajes secundarios (Clarence) que en pocas pasadas no se habian estabilizado.

3. **Refinamiento fino de pesos.** En las primeras iteraciones el modelo aprende lo grueso: "hay mayusculas + dos puntos al inicio de cada turno", "los espacios siguen ciertas reglas", "estas letras son frecuentes". En las iteraciones tardias refina lo fino: "despues de 'thee' es probable 'art' o 'hast'", "antes de un nombre propio suele haber 'O' o 'My'". Estas correlaciones de orden 2, 3, 4 se afinan en los pasos 3000-6000.

Matematicamente: la **norma del gradiente** baja, pero no se anula. Mientras siga apuntando en una direccion util, el optimizador sigue encontrando mejoras. Y como la loss es no-convexa, esa "direccion util" puede mantenerse por miles de iteraciones mas.

---

## 5. Los limites: el techo determinado por la capacidad

Ahora la mala noticia: **mas iteraciones no escalan infinitamente**. Hay un techo. Y ese techo lo determina el numero de parametros del modelo.

```
loss
  |
4 |\
  | \
3 |  \
  |   \____
2 |        \_____
  |              \____________
1 |                            \________________ <- techo del modelo de 0.82 M
  |
  +------------------------------------------------> iters
        1k   3k   6k   10k  20k  50k  100k
```

Si dejaras el script corriendo por **10000, 20000, 100000 iteraciones**, no veras el loss bajar a 1.0 ni a 0.5. Veras una asintota cerca de un valor — para un modelo de 0.82 M parametros sobre Shakespeare, ese valor probablemente esta entre **1.30 y 1.40**. Mas alla, el modelo ya no tiene capacidad para representar mejores patrones. Sus pesos saturan.

Esto se llama el **regimen "data/training-bound" vs "capacity-bound"**:

- En las primeras iteraciones el modelo es **data/training-bound**: cada pasada por los datos le aporta informacion nueva utilizable.
- A partir de cierto punto se vuelve **capacity-bound**: ya extrajo del corpus toda la informacion que sus 0.82 M parametros pueden codificar. Mas tiempo no ayuda.

En la practica esto se ve como:

- **Train loss sigue bajando** (el modelo memoriza ruido especifico).
- **Validation loss se estanca** o incluso empieza a subir (overfitting).

A 6000 iteraciones todavia no llegamos al techo de este modelo. Probablemente podriamos ganar 0.05-0.10 mas de loss yendo a 12000. Pero el retorno por iteracion adicional baja monotonicamente.

{{< concept-alert type="clave" >}}
**Para un dataset y arquitectura dados, el loss tiene un piso.** Mas iteraciones te acercan al piso, pero no lo cruzan. Para bajar mas: hay que crecer el modelo (proximo capitulo) o agregar datos.
{{< /concept-alert >}}

---

## 6. Conexion con scaling laws

Esto que viste en pequeño — mas iteraciones bajan loss con rendimientos decrecientes — es **una rebanada de las scaling laws de Kaplan 2020**.

Recordemos la ley:

$$\text{Loss} \propto N^{-\alpha} \cdot D^{-\beta} \cdot C^{-\gamma}$$

donde $D$ es **tokens vistos en entrenamiento** (data x epochs). Mas iteraciones con el mismo corpus = mas $D$ efectivo. La ley predice que el loss baja como $D^{-\beta}$, con $\beta \approx 0.07$-$0.09$.

En numeros concretos: doblar $D$ (de 3000 a 6000 iters) deberia bajar el loss en aproximadamente:

$$\Delta \log \text{Loss} \approx -\beta \cdot \log 2 \approx -0.06$$

Eso predice que el loss baja un factor de $e^{-0.06} \approx 0.94$. Si partimos de 1.62 y lo multiplicamos por 0.94, predecimos **~1.52**. Lo observado: **1.49**. La ley empirica funciona razonablemente bien incluso a esta escala diminuta.

A escala industrial, este mismo principio rige los entrenamientos modernos:

- **GPT-3 (175 B params)**: entrenado por aproximadamente **34 dias en ~10000 GPUs A100**, viendo ~300 B tokens. Eso son cerca de **3.14 \times 10^{23}$ FLOPs.
- **GPT-4** (estimado): training de meses en miles de H100s.
- **Claude 3 Opus / Gemini Ultra**: ordenes similares.

Lo que para tu modelo son 6000 iteraciones de unos minutos, para los frontier labs son **trillones de iteraciones de meses**. La logica es identica: gradient descent sobre cross-entropy de next-token. **Solo cambian los ceros**.

Y la regla **Chinchilla** (Hoffmann 2022) dice algo importante para nuestra situacion: para un compute fijo, hay un **balance optimo** entre $N$ (parametros) y $D$ (tokens entrenados). La regla aproximada: **20 tokens por parametro**. Para nuestro modelo de 0.82 M, eso predice que el optimo esta cerca de **16 M tokens** vistos. A 6000 iters con 32 batches de 64 tokens vemos ~12 M tokens, todavia debajo del optimo Chinchilla. Esto sugiere que **hay margen** para mas iteraciones — pero el retorno marginal seguira bajando.

---

## 7. Pausa de verificacion

Antes de pasar al siguiente experimento (modelo XL), asegurate de tener claro:

1. **¿Por que mas iteraciones bajan el loss?** Pista: gradient descent es estocastico. Mas pasos = mas exploracion del loss landscape, mas pasadas por el corpus, mas refinamiento de correlaciones finas entre tokens.

2. **¿Por que ese efecto es decreciente?** Pista: las primeras iteraciones aprenden lo grueso (mucha informacion por paso). Las tardias afinan detalles (poca informacion adicional por paso). Y eventualmente, el modelo satura al techo determinado por su capacidad.

3. **¿Que diferencia hay entre "entrenar mas" y "modelo mas grande"?** Pista: entrenar mas explota mejor la capacidad existente. Modelo mas grande **agrega capacidad nueva**. Ambos bajan el loss, pero atacan variables distintas en las scaling laws ($D$ vs $N$).

4. **¿Cuando dejaria de ayudar entrenar mas?** Pista: cuando train loss siga bajando pero val loss se estanque (capacity-bound + overfitting). En ese punto, la unica salida es crecer el modelo.

---

Siguiente: [11 - Modelo XL](../11-model-xl).

Codigo: `clase_14/practica/07_train_longer.py`
