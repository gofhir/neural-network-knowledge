---
título: "Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering"
autores: "Yash Goyal*, Tejas Khot*, Douglas Summers-Stay, Dhruv Batra, Devi Parikh"
afiliaciones: "Virginia Tech, Army Research Laboratory, Georgia Institute of Technology"
venue: "CVPR 2017 (IEEE Conference on Computer Vision and Pattern Recognition)"
año: 2017
arxiv: "1612.00837"
link: "https://arxiv.org/abs/1612.00837"
proyecto: "https://visualqa.org/"
clase: 23
tema: "Visual Question Answering + Image Captioning"
---

# Making the V in VQA Matter — VQA v2.0 (Goyal et al., 2017)

> **Cita.** Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, D. Parikh. *Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering*. CVPR 2017. arXiv:1612.00837. Proyecto: https://visualqa.org/. (Los dos primeros autores contribuyeron por igual.)

---

## 1. Resumen ejecutivo y por qué importa

Este paper hace una intervención quirúrgica sobre un problema embarazoso de toda una subdisciplina. Visual Question Answering (VQA) se vendía como la tarea multimodal por excelencia: muéstrale a un modelo una imagen, hazle una pregunta en lenguaje natural ("¿de qué color es el plátano?", "¿cuántas personas hay?") y exígele una respuesta. La promesa era que para responder bien el modelo tenía que *entender la imagen*. El golpe de Goyal et al. es demostrar, con evidencia empírica concreta y un dataset rediseñado, que **los modelos VQA de 2015-2016 respondían correctamente sin mirar la imagen**, explotando regularidades estadísticas del lenguaje (los *language priors* o sesgos de lenguaje).

El título es un juego de palabras deliberado: "Make the V in VQA matter" — que la **V** (Vision) de VQA *importe*. La tesis del paper es que en VQA v1 (Antol et al., 2015) la V era casi decorativa. La contribución central no es un modelo nuevo ni una arquitectura: es un **rediseño del benchmark**. Construyen VQA v2.0, un dataset **balanceado por construcción**, donde para cada par (pregunta, respuesta) existe un *par* de imágenes muy parecidas que, ante la misma pregunta, dan respuestas *distintas*. Esto destruye el atajo del lenguaje: un modelo que solo lee la pregunta queda matemáticamente atascado, porque las dos respuestas posibles son indistinguibles desde el texto.

Para Roberto, que viene del mundo de *patient matching* y MDM en FHIR, la analogía es directa y útil: VQA v1 es como evaluar un *record linkage* sobre un dataset donde el 90% de los pares positivos comparten el mismo `family.name` exacto. Un clasificador que solo mira el apellido luce excelente en ese test, pero no aprendió a *matchear*, aprendió el sesgo del conjunto de evaluación. VQA v2.0 es el equivalente a forzar *hard negatives*: pares casi idénticos en las features fáciles, que solo se distinguen mirando la señal difícil. Es exactamente la misma idea de minería de negativos difíciles que aparece en blocking/scoring de entidades.

El paper aporta tres contribuciones concretas: (1) **balancean** el VQA dataset recolectando imágenes complementarias, duplicando aproximadamente su tamaño (~1.1 millones de pares imagen-pregunta, ~13 millones de respuestas asociadas); (2) **re-evalúan** modelos VQA del estado del arte y muestran caídas grandes de accuracy, confirmando que explotaban priors; y (3) proponen un modelo de **explicación por contraejemplos** que, además de responder, muestra una imagen "negativa" donde la respuesta sería distinta, como mecanismo de interpretabilidad y confianza.

---

## 2. Contexto — el problema de los language priors en VQA v1

VQA como tarea fue introducida por Antol et al. (ICCV 2015, referencia [3] en este paper). El dataset original (lo que llamaremos VQA v1) acopla imágenes reales de COCO con preguntas de forma libre recolectadas por crowdsourcing y respuestas anotadas por humanos. Rápidamente se volvió el benchmark de referencia para "vision + language". El problema, que el propio paper documenta con cifras devastadoras, es que **el lenguaje es un prior tremendamente fuerte**: la estructura del mundo real y los sesgos del lenguaje hacen que la pregunta sola, sin imagen, ya determine la respuesta con altísima probabilidad.

Los autores citan ejemplos concretos del VQA v1:

- "tennis" es la respuesta correcta para el **41%** de las preguntas que empiezan con "What sport is".
- "2" es la respuesta correcta para el **39%** de las preguntas que empiezan con "How many".
- Las preguntas que empiezan con "Is there a clock" tienen respuesta "yes" el **98%** de las veces.
- Las preguntas que empiezan con "Is the man standing" tienen respuesta "no" el **69%** de las veces.

El caso más perverso que mencionan: para las preguntas que empiezan con el n-grama "Do you see a ...", responder ciegamente "yes" sin leer el resto de la pregunta ni mirar la imagen da una **accuracy de VQA del 87%**. Es decir, un sistema que ni siquiera procesa la pregunta completa, mucho menos la imagen, alcanza casi 9 de cada 10 correctas en ese subconjunto.

¿Por qué pasa esto? Hay dos fenómenos entrelazados:

1. **El prior del mundo y del lenguaje.** La gente no pregunta cosas al azar. Si pregunto "¿de qué color es el plátano?" es porque los plátanos suelen ser amarillos; la distribución $P(A)$ y especialmente la distribución condicional al n-grama de la pregunta $P(A \mid \text{n-grama}(Q))$ están enormemente concentradas. El modelo puede memorizar esa distribución marginal/condicional y nunca consultar los píxeles.

2. **El *visual priming bias*.** Zhang et al. (referencia [47]) lo identificaron: los anotadores *vieron* la imagen mientras escribían la pregunta. Por eso la gente solo pregunta "¿hay una torre de reloj en la imagen?" cuando la imagen efectivamente contiene una torre de reloj. La mera existencia de la pregunta filtra información sobre la respuesta. El acto de preguntar está condicionado a la imagen, lo que sesga el dataset hacia respuestas afirmativas.

La consecuencia metodológica es grave: estos *language priors* **dan una falsa impresión de progreso**. Un modelo puede reportar accuracy alta y parecer que "entiende imágenes" cuando en realidad solo aprendió la estadística del texto. Eso frena el avance real de la IA multimodal porque optimizamos contra una métrica que premia el atajo equivocado. El paper lo dice explícitamente: este hallazgo provee "la primera evidencia empírica concreta de lo que parecía ser una sospecha cualitativa entre los practicantes".

Un punto técnico crucial que los autores subrayan: **no basta con uniformizar $P(A)$** (la distribución marginal de respuestas) en el dataset. Aunque hicieras que "yes" y "no" aparecieran 50/50 globalmente, los modelos siguen explotando la *correlación* entre los n-gramas de la pregunta y las respuestas, es decir $P(A \mid \text{n-grama}(Q))$. Lo que se necesita no es solo mayor entropía en $P(A)$, sino **mayor entropía en $P(A \mid Q)$**, de modo que la imagen $I$ tenga forzosamente que jugar un rol en determinar $A$. Esto motiva un balanceo **a nivel de cada pregunta individual**, no global. Esta distinción es la clave técnica de todo el diseño.

---

## 3. La idea central — balancear con imágenes complementarias

La idea es elegante y se puede enunciar en una frase. Para cada triplete $(I, Q, A)$ del VQA dataset original —imagen, pregunta, respuesta— se busca, con ayuda de un humano, **otra imagen $I'$ que sea similar a $I$ pero para la cual la misma pregunta $Q$ tenga una respuesta $A'$ distinta** ($A' \neq A$).

Formalmente, el dataset balanceado contiene, para cada pregunta, *al menos un par* de la forma:

$$(I, Q, A) \quad \text{y} \quad (I', Q, A'), \qquad A' \neq A, \quad I' \approx I$$

donde $I' \approx I$ significa que $I'$ es semánticamente cercana a $I$ (en el espacio de features de una CNN).

¿Por qué esto funciona? La hipótesis de los autores es que el balanceo **fuerza a los modelos a enfocarse en la información visual**. Considera un modelo que solo procesa el lenguaje: ve $(Q, I)$ y $(Q, I')$. Como en ambos casos la pregunta $Q$ es idéntica, y el modelo ignora la imagen, *no tiene absolutamente ninguna base para diferenciar los dos casos*. Producirá la misma respuesta para ambos y, por construcción, una de las dos estará mal. El atajo del lenguaje colapsa: ya no puede ganar accuracy memorizando $P(A \mid Q)$, porque ahora $P(A \mid Q)$ tiene alta entropía (las dos respuestas son igualmente probables dado solo el texto).

Hay un matiz adicional que eleva la dificultad y que los autores destacan como característica deseada, no como bug: el par complementario $I'$ es **cercano a $I$ en el espacio semántico de la penúltima capa (fc7) de VGGNet**. Esto significa que las dos imágenes no solo dan respuestas distintas, sino que además se *parecen mucho* en el espacio de representaciones que las propias CNN aprenden. Por lo tanto, incluso un modelo que sí mira la imagen tiene que captar **diferencias sutiles** entre las dos imágenes para responder bien en ambas. No basta con "ver que hay una imagen distinta"; hay que entender *qué* cambió. Esto convierte el balanceo en un test de razonamiento visual fino, no solo de no-ignorar-la-imagen.

La diferencia con trabajos previos es importante. Zhang et al. (ref. [47]) habían estudiado el balanceo, pero en un escenario muy restringido: solo preguntas binarias (sí/no) sobre **escenas de clipart** (escenas sintéticas, parte del VQA abstract scenes). El clipart permite editar el contenido para forzar el cambio de respuesta. Goyal et al. generalizan esto a (1) **imágenes reales** de COCO, donde no puedes editar píxeles a voluntad, (2) **todas las preguntas**, no solo binarias, (3) benchmarking de modelos del estado del arte, y (4) el modelo de explicación por contraejemplos. La novedad metodológica es la **interfaz de recolección de imágenes complementarias** aplicada a fotografías reales.

---

## 4. Construcción del dataset balanceado

El proceso de construcción es de dos etapas y descansa enteramente en Amazon Mechanical Turk (AMT). Vale la pena entenderlo en detalle porque ahí está el aporte de ingeniería del paper.

### 4.1. Punto de partida

Se construye **encima** del VQA dataset de Antol et al. (ref. [3]). Ese dataset de imágenes reales contiene:

| Magnitud | VQA v1 (real images) |
|---|---|
| Imágenes (de COCO) | ~204K |
| Preguntas de forma libre | 614K (≈3 por imagen) |
| Respuestas de forma libre | >6 millones (10 por pregunta) |

Las 10 respuestas por pregunta provienen de 10 anotadores distintos; esto importa para la métrica de evaluación (sección 8).

### 4.2. Etapa 1 — recolectar imágenes complementarias

Para cada triplete $(I, Q, A)$ se quiere encontrar $I'$. El procedimiento:

1. **Candidatos por vecinos cercanos.** Se calculan los **24 vecinos más cercanos** de $I$. Para ello se representa cada imagen con las activaciones de la penúltima capa ('fc7') de una CNN profunda —en particular **VGGNet** (ref. [37])— y se usan **distancias $\ell_2$** para encontrar los vecinos. Esto garantiza que los candidatos sean visualmente/semánticamente parecidos a $I$.

2. **Interfaz en AMT (Figura 3).** A cada trabajador de AMT se le muestran las 24 imágenes vecinas de $I$, junto con la pregunta $Q$ y la respuesta original $A$. La tarea: **elegir una imagen $I'$ de las 24 para la cual $Q$ "tenga sentido" y la respuesta a $Q$ sea distinta de $A$** (es decir, "is NOT $A$").

3. **Que la pregunta "tenga sentido".** Esto es delicado y los autores lo cuidan con tests de calificación previos. Le explican a los trabajadores que cualquier *premisa* asumida por la pregunta debe ser verdadera en la imagen elegida. Ejemplo: la pregunta "What is the woman doing?" (¿qué está haciendo la mujer?) presupone que hay una mujer visible. No tiene sentido elegir una imagen sin mujer. Esto evita pares degenerados donde la respuesta cambia solo porque la pregunta se volvió inaplicable.

4. **Opción "not possible".** A veces ninguno de los 24 vecinos sirve como imagen complementaria, por dos razones: (a) la pregunta no tiene sentido en *ninguno* de los 24 (ej. "What is the woman doing?" y ninguno contiene mujer), o (b) la pregunta sí aplica pero la respuesta sigue siendo $A$ en todos (ej. te piden una imagen donde la respuesta a "¿de qué color es el plátano?" NO sea "amarillo", y eso es raro). La interfaz permite marcar "not possible". **En total, las selecciones "not possible" constituyen el 22% de todas las preguntas del VQA dataset.** Los autores especulan que una interfaz que permitiera scrollear más allá de 24 vecinos reduciría esa fracción, pero (1) probablemente no llegaría a 0 (puede que no exista en COCO ninguna imagen donde "¿está volando la mujer?" sea "no" relevante... el ejemplo que dan es que difícilmente no exista, pero el costo sube), y (2) la tarea sería mucho más pesada y cara para los trabajadores.

5. **Causas analizadas de "not possible".** Los autores analizan estos casos y encuentran que ocurren típicamente cuando (1) el objeto del que habla la pregunta es demasiado pequeño en la imagen original, de modo que los vecinos cercanos —globalmente similares— no necesariamente lo contienen; o (2) el concepto de la pregunta es raro (ej. pedir una imagen donde "¿de qué color es el plátano?" NO sea "amarillo").

### 4.3. Etapa 2 — recolectar respuestas nuevas

Una vez elegida $I'$, hay que saber cuál es la respuesta $A'$. No se asume: se mide. Se muestra la imagen elegida $I'$ junto con la pregunta $Q$ a **10 nuevos trabajadores de AMT** y se recolectan **10 respuestas ground-truth** (igual que en el VQA original, ref. [3]). **La respuesta más común entre las 10 es la nueva respuesta $A'$.**

Esto introduce un detalle honesto: aunque es improbable, el voto mayoritario de las 10 respuestas nuevas *podría* coincidir con $A$, ya sea por desacuerdo entre humanos o porque el trabajador que eligió $I'$ se equivocó. Los autores encuentran que esto ocurre —es decir, $A = A'$— en **aproximadamente el 9% de las preguntas**. Son los casos donde el balanceo no logró producir una respuesta efectivamente distinta.

### 4.4. Estadísticas finales

El proceso de dos etapas produce **pares de imágenes complementarias $(I, I')$** semánticamente similares pero con respuestas distintas $(A, A')$ a la misma $Q$.

| Magnitud | VQA v2.0 (balanceado) |
|---|---|
| Pares (imagen, pregunta) totales | **~1.1 millones** (casi el doble de v1) |
| Respuestas asociadas | **~13 millones** |
| Imágenes base (COCO) | ~200K |
| Imágenes complementarias — **train** | ~195K |
| Imágenes complementarias — **val** | ~93K |
| Imágenes complementarias — **test** | ~191K |
| Preguntas con complementaria recolectada | ~135K (aprox.) en train |
| "(question, image)" extra añadidos al test | ~18K (para detectar tendencias anómalas) |

El dataset completo balanceado contiene **más de 443K (train) + 214K (val) + 453K (test)** pares (pregunta, imagen). Siguiendo el VQA original (ref. [3]), el test se divide en **4 splits**: test-dev, test-standard, test-challenge y test-reserve. El dataset se publicó como parte de la **2ª iteración del Visual Question Answering Dataset and Challenge (VQA v2.0)** en https://visualqa.org/.

Un matiz que conviene tener claro: la recolección de imágenes complementarias se hizo para train, val **y** test. Pero como el 22% es "not possible" y el 9% termina con $A=A'$, el dataset **no queda perfectamente balanceado** — queda *significativamente más* balanceado que v1, no perfectamente 50/50.

---

## 5. Análisis del balanceo — cuánto se redujo el sesgo

La Figura 4 del paper compara la distribución de respuestas por tipo de pregunta para una muestra aleatoria de **60K preguntas**, en v1 (arriba) vs v2.0 (abajo). Los hallazgos cualitativos y cuantitativos:

**Preguntas binarias (yes/no).** Es donde el efecto es más visible. Tipos como "is the", "is this", "is there", "are", "does" tienen en v2.0 una distribución **significativamente más balanceada** entre "yes" y "no" que en v1. En v1 estaban fuertemente sesgadas hacia "yes" (recordemos: "Is there a clock" → "yes" 98% del tiempo). El balanceo aplana esto.

**Tipos de respuesta con colas más pesadas.** En v1, "tennis" dominaba "What sport". En v2.0, "baseball" se vuelve ligeramente más popular que "tennis" bajo "what sport", y en general "baseball" y "tennis" dejan de dominar tan abrumadoramente; deportes como "frisbee", "skiing", "soccer", "skateboarding", "snowboard" y "surfing" se vuelven más visibles. Lo mismo para colores, animales y números: las colas de la distribución se engrosan, es decir, las respuestas menos frecuentes ganan masa de probabilidad.

**Métrica cuantitativa de sesgo: la entropía.** El indicador medible que reportan es que la **entropía de las distribuciones de respuesta, promediada sobre tipos de pregunta (ponderada por frecuencia de cada tipo), aumenta en un 56% tras el balanceo**. Esto es la confirmación numérica directa de "colas más pesadas" / menor concentración: un dataset menos predecible desde el solo texto. Mayor entropía en $P(A \mid \text{tipo de }Q)$ es exactamente el objetivo que se planteó en la sección 3 (recordar: queremos alta entropía en $P(A\mid Q)$, no solo en $P(A)$).

En resumen: el balanceo no es perfecto, pero es *medible* y *grande* — 56% más de entropía es una reducción sustancial del prior explotable.

---

## 6. Experimentos — re-evaluar modelos VQA v1 sobre el dataset balanceado

Aquí está el corazón empírico. La pregunta es: si los modelos del estado del arte de v1 realmente explotaban *language priors*, deberían **caer** al evaluarse sobre el dataset balanceado. Y eso es exactamente lo que ocurre.

### 6.1. Modelos evaluados

- **d-LSTM+n-I (Deeper LSTM Question + norm Image)** (ref. [24]): el modelo VQA "estándar" introducido junto al dataset de Antol et al. Usa un embedding CNN de la imagen, un embedding LSTM de la pregunta, los combina por **multiplicación punto a punto (point-wise)**, y un perceptrón multicapa predice una distribución sobre las **1000 respuestas más frecuentes**.
- **HieCoAtt (Hierarchical Co-attention)** (ref. [25]): modelo basado en atención que "co-atiende" imagen y pregunta. Modela la pregunta jerárquicamente (nivel de palabra, de frase, y de pregunta completa) y combina los niveles recursivamente. Distribución sobre las 1000 respuestas más frecuentes.
- **MCB (Multimodal Compact Bilinear Pooling)** (ref. [9]): la **entrada ganadora** del track de imágenes reales del **VQA Challenge 2016**. Usa pooling bilineal compacto para atender sobre features de imagen y combinarlas con las del lenguaje, luego una capa fully-connected predice sobre las **3000 respuestas más frecuentes**. Importante: MCB usa features de **ResNet** (ref. [12]), una CNN más potente, mientras d-LSTM+n-I y HieCoAtt usan **VGGNet**.

### 6.2. Baselines de referencia

- **Prior:** predecir siempre la respuesta más común del train set. En ambos conjuntos (balanceado y no balanceado) esa respuesta es "yes".
- **Language-only:** misma arquitectura que d-LSTM+n-I pero **sin la rama de imagen** — solo recibe la pregunta. Esta ablación cuantifica cuánto puede lograrse *sin imagen alguna*. Comparar los modelos completos contra este baseline mide cuánto realmente aprovechan la imagen.

### 6.3. Protocolo de notación

Los experimentos cruzan entrenamiento y evaluación entre datasets, con una notación compacta de dos letras (primera = train, segunda = test; U = Unbalanced, B = Balanced):

- **UU**: entrenar en no balanceado, testear en no balanceado (el setting clásico de v1).
- **UB**: entrenar en no balanceado, testear en **balanceado**.
- **B$_\text{half}$B**: entrenar en un train balanceado *de tamaño similar al original* (para comparación justa) y testear en balanceado.
- **BB**: entrenar en el train balanceado **completo** (~2× el original) y testear en balanceado.

### 6.4. Tabla 1 — el resultado central

> Performance de modelos VQA entrenados/testeados sobre datasets no balanceados/balanceados.

| Approach | UU | UB | B$_\text{half}$B | BB |
|---|---|---|---|---|
| Prior | 27.38 | 24.04 | 24.04 | 24.04 |
| Language-only | 48.21 | 41.40 | 41.47 | 43.01 |
| d-LSTM+n-I [24] | 54.40 | 47.56 | 49.23 | 51.62 |
| HieCoAtt [25] | 57.09 | 50.31 | 51.88 | 54.57 |
| MCB [9] | 60.36 | 54.22 | 56.08 | 59.14 |

Lecturas clave:

1. **La caída UU → UB.** Todos los modelos caen significativamente al pasar de testear en no balanceado (UU) a balanceado (UB), *manteniendo el mismo entrenamiento*. MCB cae de 60.36 a 54.22 (~6 puntos); HieCoAtt de 57.09 a 50.31 (~6.8 puntos); d-LSTM de 54.40 a 47.56 (~6.8 puntos). Esto **confirma la hipótesis**: los modelos habían aprendido sesgos de lenguaje del dataset, y esos sesgos *también existían en el val set no balanceado*, por eso lucían bien en UU. Al evaluar en el val balanceado (sesgos reducidos), el rendimiento se desploma.

2. **El baseline Language-only es revelador.** Sin imagen alguna, alcanza 48.21 en UU. Que un modelo *ciego* (que nunca ve píxeles) llegue a ~48% de accuracy de VQA es la prueba más limpia de cuánta señal hay en el solo lenguaje. Y como se esperaba, cae a ~41% en UB: el modelo solo-lenguaje sufre todavía más con el balanceo, confirmando que la palanca era el prior textual.

3. **Re-entrenar en balanceado ayuda (UB → B$_\text{half}$B).** Si entrenas sobre datos balanceados (incluso del mismo tamaño), el rendimiento sube respecto a entrenar en no balanceado. El modelo aprende a usar más la imagen porque el train ya no premia el atajo.

4. **Más datos balanceados ayudan más (B$_\text{half}$B → BB).** Pasar al train balanceado completo (~2× tamaño) mejora **2-3 puntos** adicionales. Los autores interpretan que los modelos están *data-starved* (hambrientos de datos) y se beneficiarían de datasets aún más grandes. Aun así, ningún modelo recupera el nivel de UU: BB de MCB (59.14) sigue por debajo de UU (60.36), pero ahora ese número refleja entendimiento visual real, no prior.

### 6.5. Tabla 2 — números oficiales sobre test-standard

Para que otros papers reporten comparables, entrenan también sobre **VQA v2.0 train+val** y reportan en **test-standard**. Los autores recomiendan que todo paper que use VQA v2.0 reporte estos números de test-standard.

> Performance sobre VQA v2.0 test-standard (entrenado en train+val v2.0).

| Approach | All | Yes/No | Number | Other |
|---|---|---|---|---|
| Prior | 25.98 | 61.20 | 00.36 | 01.17 |
| Language-only | 44.26 | 67.01 | 31.55 | 27.37 |
| d-LSTM+n-I [24] | 54.22 | 73.46 | 35.18 | 41.83 |
| MCB [9] | 62.27 | 78.82 | 38.28 | 53.36 |

MCB queda como el mejor modelo, con **62.27%** de accuracy global sobre test-standard de VQA v2.0. Nótese cuánto cuesta "Number" (conteo): ~38% incluso para MCB; es el tipo de pregunta más difícil junto con la cola larga de "Other".

---

## 7. El "counterexample" como interpretabilidad

Esta es la tercera contribución y conecta el balanceo con explicabilidad. La intuición: si tengo, para cada pregunta, una imagen $I$ y una complementaria $I'$ donde la respuesta cambia, puedo enseñar a un modelo a **explicar su respuesta exhibiendo un contraejemplo**.

La idea de producto: para "What color is the fire-hydrant?" → "red", un modelo VQA será percibido como más confiable si, además de decir "red", agrega "a diferencia de esta" y muestra una imagen de un hidrante que *no* es rojo. Es una **explicación por negativos / hard negatives**: "esto es lo que creo que es similar pero con respuesta distinta". Construye confianza ("trust") porque sugiere que el modelo entiende el concepto que se le pregunta, y permite al usuario detectar fallas.

### 7.1. Modelo de dos pasos

En test, el modelo opera en dos pasos:

1. **Responder.** Como un VQA convencional, recibe $(Q, I)$ y predice $A_\text{pred}$.
2. **Explicar.** Usa $A_\text{pred}$ junto con $Q$ para recuperar una imagen similar a $I$ pero con respuesta distinta. Elige una de las $K$ vecinas más cercanas $I_{NN} = \{I_1, \dots, I_K\}$ como contraejemplo.

El dato supervisado para entrenar esto viene "gratis" del proceso de balanceo: $I'$ (la imagen complementaria elegida por humanos) **es, por definición, un buen contraejemplo** — $Q$ es relevante para $I'$, $I'$ tiene respuesta distinta de $A$, e $I'$ es similar a $I$. Entonces tenemos datos supervisados donde $I'$ es el contraejemplo correcto entre los $K=24$ vecinos.

### 7.2. Arquitectura (dos cabezas sobre un tronco compartido)

- **Base compartida.** Red de 2 canales: embedding CNN de la imagen en una rama, embedding LSTM de la pregunta en otra, combinados por **multiplicación punto a punto** → embedding conjunto $QI$ (similar a ref. [24]). Se pasan por aquí la imagen original $I$ más las 24 candidatas, total **25 imágenes**.
- **Cabeza de respuesta.** Capa fully-connected + softmax sobre las respuestas, alimentada solo por el $QI$ de la imagen original. Entrenada con **cross-entropy**.
- **Cabeza de explicación.** Transforma linealmente el $QI$ conjunto y la respuesta a explicar $A$ a un espacio común, computa un **producto interno** que da un score escalar $S(I_i)$ por cada candidata. Las $K$ candidatas se ordenan por ese score; las de mayor score son los mejores contraejemplos. Se entrena con **pérdida de ranking por bisagra (hinge) pairwise** que empuja el score de la imagen humana $I'$ a estar por encima de las demás por un margen $M$:

$$S(I') - S(I_i) > M - \epsilon, \quad I_i \in \{I_1, \dots, I_K\} \setminus \{I'\}$$

con la forma estándar de hinge $\max\bigl(0,\, M - (S(I') - S(I_i))\bigr)$. La pérdida combinada es:

$$\mathcal{L} = -\log P(A \mid I, Q) + \lambda \sum_i \max\bigl(0,\, M - (S(I') - S(I_i))\bigr)$$

donde el primer término es la cross-entropy de la cabeza de respuesta sobre $(I, Q)$, el segundo término es la suma de hinge losses que premia rankear alto la $I'$ humana, y $\lambda$ es el peso de compromiso (trade-off) entre las dos pérdidas.

### 7.3. Resultados de explicación (Tabla 4)

Se evalúa con **Recall@5**: con qué frecuencia la $I'$ elegida por humanos está entre las top-5 candidatas que ordena el modelo (análogo al top-5 de ImageNet, robusto a que la "mejor" complementaria no sea única). Baselines: **Random** (orden aleatorio), **Distance** (orden por cercanía a $I$; la más similar = contraejemplo más probable), **VQA** [3] (ordenar por $P(A \mid Q, I_i)$ ascendente: la candidata *menos* probable de tener $A$ como respuesta = contraejemplo más probable).

| | Random | Distance | VQA [3] | Ours |
|---|---|---|---|---|
| Recall@5 | 20.79 | 42.84 | 21.65 | **43.39** |

El modelo propuesto (43.39) supera a Random y al baseline VQA, y queda apenas por encima de **Distance** (42.84), que resulta ser un baseline sorprendentemente fuerte. Los autores son honestos: identificar el contraejemplo correcto entre los vecinos cercanos es una tarea difícil, y el hecho de que Distance sea tan competitivo "sugiere de nuevo que los modelos de entendimiento visual capaces de extraer detalles finos siguen siendo elusivos". La Figura 5 muestra ejemplos cualitativos sensatos (ej. "Which way is its head turned? left" con cebras mirando a otro lado; "What color is the plate? blue" con platos de otro color).

---

## 8. Resultados numéricos — desglose por tipo de respuesta

La métrica de VQA usa las **10 respuestas ground-truth por pregunta** y el script de evaluación público del VQA original (ref. [3]). La fórmula estándar es:

$$\text{Acc}(a) = \min\!\left(\frac{\#\{\text{humanos que respondieron } a\}}{3}, \; 1\right)$$

es decir, una respuesta del modelo cuenta como totalmente correcta si al menos 3 de los 10 anotadores la dieron, con crédito parcial si menos. Por eso se recolectaron 10 respuestas también para cada imagen complementaria, para ser consistentes con v1.

### 8.1. Tabla 3 — desglose por tipo de respuesta (MCB y HieCoAtt)

> Accuracy por tipo de respuesta, entrenado/testeado en no balanceado/balanceado.

| Approach | Ans Type | UU | UB | B$_\text{half}$B | BB |
|---|---|---|---|---|---|
| **MCB** [9] | Yes/No | 81.20 | 70.40 | 74.89 | 77.37 |
| | Number | 34.80 | 31.61 | 34.69 | 36.66 |
| | Other | 51.19 | 47.90 | 47.43 | 51.23 |
| | All | 60.36 | 54.22 | 56.08 | 59.14 |
| **HieCoAtt** [25] | Yes/No | 79.99 | 67.62 | 70.93 | 71.80 |
| | Number | 34.83 | 32.12 | 34.07 | 36.53 |
| | Other | 45.55 | 41.96 | 42.11 | 46.25 |
| | All | 57.09 | 50.31 | 51.88 | 54.57 |

Hallazgos:

1. **Yes/No es donde más cae (UU → UB).** MCB cae ~10.8 puntos (81.20 → 70.40) y HieCoAtt ~12.4 puntos (79.99 → 67.62) en yes/no. Esto demuestra que los modelos estaban **explotando fuertemente los sesgos de lenguaje en las preguntas binarias**: en v1, la alta accuracy en yes/no venía de que el val no balanceado *también* contenía esos sesgos. Al testear en el val balanceado (sesgos reducidos en yes/no), el desempeño se desploma. Es la firma inequívoca del prior.

2. **Mayor fuente de mejora al balancear (UB → B$_\text{half}$B):** de nuevo yes/no (~4.5 puntos MCB, ~3 puntos HieCoAtt) y number (~3 MCB, ~2 HieCoAtt). Re-entrenar en balanceado recupera sobre todo lo perdido en binarias.

### 8.2. La observación más punzante: el benchmark "borraba" las diferencias entre modelos

Este es uno de los aportes conceptuales más finos. Los autores observan que yes/no y number son justamente los tipos donde los enfoques publicados mostraban **mejoras mínimas entre sí**. En los resultados del VQA Real Open Ended Challenge 2016, el gap de accuracy entre los **top-4** approaches era de apenas **0.15%** en yes/no (y 3.48% entre los top-10). En number, las accuracies variaban solo **1.51%** (top-4) y **2.64%** (top-10). El grueso de las diferencias entre modelos venía del tipo "other" (gaps de 7.03% top-4 y 10.58% top-10).

La conclusión es demoledora para la metodología de v1: como los language priors llevan a *todos* los modelos a accuracies similares en yes/no y number, esos tipos de pregunta vuelven a los modelos **virtualmente indistinguibles entre sí**. El benchmark no podía discriminar entre un buen modelo (con los *inductive biases* correctos: atención, composicionalidad) y uno que simplemente es un clasificador de alta capacidad ajustándose a los sesgos del dataset. **Benchmarkear sobre el dataset balanceado, con priors reducidos, finalmente permite distinguir modelos buenos de modelos que solo memorizan sesgos.**

### 8.3. El gap humano-máquina y el techo

Aunque los números absolutos en valor parecen razonables (MCB ~62% en test-standard), el paper enfatiza que hay **mucho espacio de mejora** para construir modelos de entendimiento visual capaces de extraer detalle fino. El razonamiento visual sobre pares de imágenes similares con respuestas distintas es intrínsecamente difícil: el modelo HieCoAtt entrenado en balanceado responde *ambas* imágenes del par correctamente en solo **17.7%** de los pares (vs. 13.5% cuando se entrena en no balanceado — una mejora de 4.2 puntos), y predice respuestas idénticas para ambas en 10.5% menos de pares (de 59.9% a 49.4%). O sea: incluso el mejor modelo, en más del 80% de los pares complementarios, **no logra responder bien las dos imágenes**. Eso confirma que el balanceo dejó al descubierto cuánto razonamiento visual fino sigue faltando.

---

## 9. Limitaciones

El propio paper, y la lectura crítica, dejan ver varias limitaciones:

1. **No es perfectamente balanceado.** El 22% de "not possible" y el 9% de casos donde $A=A'$ implican que el dataset es *significativamente* más balanceado, no balanceado al 100%. Persiste algo de prior explotable (esto será central en la conexión con Pythia, sección 11).

2. **Restringido a 24 vecinos.** La elección de $I'$ se limita a los 24 vecinos más cercanos por $\ell_2$ sobre fc7 de VGGNet. Una ventana más amplia reduciría "not possible" pero encarece y complica la tarea. Además, atar la similitud a VGGNet-fc7 hereda los sesgos de esa representación particular.

3. **Solo balancea sobre el conjunto de preguntas *existentes* de v1.** No genera preguntas nuevas; reusa las preguntas de Antol et al. y les añade imágenes complementarias. Los sesgos en el *tipo* de preguntas que la gente formula permanecen.

4. **El modelo de explicación es modesto.** El Recall@5 (43.39) apenas supera al baseline Distance (42.84). Identificar el contraejemplo correcto sigue siendo difícil; el modelo no demuestra entendimiento visual fino robusto.

5. **El contraejemplo "humano" no es único ni necesariamente óptimo.** Los humanos solo eligieron *alguna* imagen donde $Q$ tiene sentido y la respuesta no es $A$; no había criterio de "mejor" contraejemplo. Esto añade ruido a la supervisión de la cabeza de explicación. Una mejora futura sería incorporar un componente de *question relevance* (ref. [30]) para elegir mejores contraejemplos y evitar imágenes confusas (ej. para "What is the woman doing?", un minado de negativos puro podría elegir una imagen sin mujer).

6. **El conteo ("Number") sigue siendo pésimo.** ~36-38% incluso para el mejor modelo. El balanceo expone la dificultad pero no la resuelve.

---

## 10. Impacto y legado

VQA v2.0 se convirtió en **el** benchmark estándar de Visual Question Answering, desplazando por completo a v1.

- **VQA Challenge 2017 en adelante.** A partir de 2017, el VQA Challenge oficial (en CVPR workshops) corre sobre VQA v2.0. Reportar sobre v2.0 test-standard se vuelve obligatorio para publicar.
- **Pythia.** El framework Pythia (Facebook AI Research / Jiang et al., 2018), que **ganó el VQA Challenge 2018**, se entrena y evalúa sobre VQA v2.0. Pythia (luego absorbido en MMF, Multimodal Framework) se convirtió en una base de referencia para investigación VQA. Es el sistema que la clase 23 usa como ejemplo práctico (ver sección 11).
- **Base de evaluación de los VLMs modernos.** Desde los modelos de atención (Bottom-Up Top-Down de Anderson et al., 2018, también ganador con features de detección de objetos), pasando por LXMERT, ViLBERT, UNITER, y hasta los grandes modelos visión-lenguaje (VLMs) contemporáneos como BLIP-2, Flamingo, LLaVA, GPT-4V/o y Gemini, **VQA v2.0 sigue siendo un benchmark reportado** para medir capacidad de razonamiento visual sobre preguntas abiertas. Su accuracy "VQA score" con la fórmula de 3/10 anotadores es todavía el estándar de facto.
- **Cambio cultural.** Más allá del dataset, el legado conceptual es metodológico: instaló en la comunidad la conciencia de que **un benchmark mal balanceado infla las capacidades** y que el balanceo por hard negatives es una herramienta de diseño de datasets. Inspiró trabajos posteriores sobre *dataset bias*, *shortcut learning* y splits diagnósticos (GQA, VQA-CP — VQA under Changing Priors, que lleva esta idea aún más lejos creando splits train/test con priors *deliberadamente distintos*).

---

## 11. Conexión con la clase 23

Este paper es **exactamente** el dataset descrito en las slides 7-8 de la clase 23. Aquellas slides enuncian: "204K imágenes de COCO, 614K preguntas (3 por imagen), 6M respuestas (10 por pregunta), conjunto balanceado: para cada triplete (I,Q,A) identifican otra imagen cercana a I que da una respuesta diferente a Q". Cada una de esas cifras y la mecánica del balanceo provienen literalmente de este artículo (las cifras 204K/614K/6M son las del VQA v1 sobre el que se construye; el balanceo con imagen complementaria cercana es la contribución de Goyal et al.).

**Por qué la profesora enfatiza "evitar sesgos de lenguaje".** Toda la motivación pedagógica de mostrar VQA v2.0 en clase es justamente la lección central del paper: si entrenas un sistema multimodal sobre un dataset sesgado, obtendrás un modelo que parece entender imágenes pero que en realidad explota la estadística del texto. Para un curso de IA aplicada, es una advertencia metodológica de primer orden — vale para VQA, para captioning, y por extensión para cualquier sistema que combine modalidades (incluido, Roberto, el *patient matching*: si tu test set tiene atajos, tu F1 miente).

**Cómo Pythia entrena sobre VQAv2.** Pythia —el sistema que la clase usa para demostrar VQA en la práctica— se entrena sobre VQA v2.0 precisamente porque es el benchmark estándar y balanceado. Pythia toma features de regiones de objetos (Bottom-Up attention sobre Faster R-CNN) más el embedding de la pregunta, los fusiona con atención, y predice sobre el vocabulario de respuestas frecuentes — exactamente el linaje d-LSTM+n-I → MCB → Bottom-Up que este paper benchmarkea, escalado y optimizado.

**El punto más importante para conectar con las slides 14-19 (los problemas de Pythia).** Las slides 14-19 muestran que **los language priors persisten incluso con un dataset balanceado**. Esto no contradice el paper — *lo confirma desde la honestidad del propio paper*. Recordemos las limitaciones de la sección 9: VQA v2.0 está *significativamente más* balanceado, no *perfectamente* balanceado (22% "not possible", 9% con $A=A'$, balanceo solo sobre preguntas preexistentes). Por lo tanto:

- Pythia, entrenado sobre VQA v2.0, **aún puede explotar el prior residual** que sobrevive al balanceo. Cuando en clase se muestra a Pythia respondiendo "amarillo" a "¿de qué color es el plátano?" sobre un plátano verde, o dando respuestas plausibles-pero-ciegas, estás viendo el prior de lenguaje *que el balanceo no eliminó del todo*.
- La lección de las slides 14-19 es la madurez del campo: el balanceo de Goyal et al. fue un avance enorme (entropía +56%, modelos distinguibles de nuevo), pero **no es una bala de plata**. El *shortcut learning* es persistente. Por eso surgieron benchmarks aún más agresivos como VQA-CP (priors invertidos entre train y test), donde modelos como Pythia caen catastróficamente — demostrando que incluso sobre v2.0 todavía dependían de priors.
- Conexión técnica concreta: el balanceo reduce $P(A\mid Q)$ a algo de mayor entropía, pero los modelos de alta capacidad aún capturan correlaciones residuales. Mostrar la imagen complementaria $I'$ (el contraejemplo del paper) es justamente la herramienta de diagnóstico que revela cuándo Pythia "no está mirando": si responde igual a $I$ y a $I'$, no entendió.

En síntesis para la clase: VQA v2.0 es la respuesta de la comunidad al pecado original de VQA (el atajo del lenguaje), Pythia es el caballo de batalla práctico que se entrena sobre él, y los fallos de Pythia que se muestran en clase son la evidencia viva de que el problema que este paper atacó **sigue parcialmente vigente**. El paper no "arregla" VQA; lo hace *honesto y medible*, que es exactamente lo que un buen benchmark debe hacer.

---

## 12. Notas y enlaces

- **Paper:** arXiv:1612.00837 — https://arxiv.org/abs/1612.00837 (v3, 15 May 2017). Publicado en CVPR 2017.
- **Proyecto y dataset:** https://visualqa.org/ — descarga de VQA v2.0, splits train/val/test (test-dev, test-standard, test-challenge, test-reserve), script de evaluación oficial.
- **Challenge:** https://visualqa.org/challenge.html — VQA Challenge, corre sobre v2.0 desde 2017.

**Referencias clave citadas en el paper (mapeo de números internos):**

| Ref. | Trabajo | Relevancia |
|---|---|---|
| [3] | Antol et al., *VQA: Visual Question Answering*, ICCV 2015 | El dataset v1 sobre el que se construye |
| [24] | Deeper LSTM Q + norm Image (d-LSTM+n-I) | Modelo VQA baseline benchmarkeado |
| [25] | Hierarchical Co-attention (HieCoAtt) | Modelo de atención benchmarkeado |
| [9] | Fukui et al., MCB (Multimodal Compact Bilinear Pooling), EMNLP 2016 | Ganador VQA Challenge 2016; mejor modelo en este paper |
| [47] | Zhang et al., balanceo en clipart (abstract scenes) | Antecedente directo del balanceo, en escenas sintéticas |
| [12] | He et al., ResNet, CVPR 2016 | Backbone CNN usado por MCB |
| [37] | Simonyan & Zisserman, VGGNet | CNN para vecinos cercanos (fc7) y baselines |
| [23] | Lin et al., COCO | Fuente de las ~200K imágenes |
| [14] | Hodosh & Hockenmaier, forced-choice captioning | Antecedente conceptual de "hard negatives" |
| [30] | Ray et al., question relevance | Mejora futura para mejores contraejemplos |

**Glosario rápido para Roberto:**

- **Language prior / sesgo de lenguaje:** $P(A\mid Q)$ concentrada; el texto solo casi determina la respuesta. Análogo a un *shortcut feature* en record linkage.
- **Imagen complementaria $I'$ / contraejemplo:** *hard negative* visual — imagen vecina en fc7 con respuesta distinta. Misma idea que minería de negativos difíciles en blocking/scoring.
- **VQA score:** $\min(\#\text{anotadores con esa respuesta}/3, 1)$ sobre 10 anotadores — métrica robusta a desacuerdo humano.
- **Entropía +56%:** medida cuantitativa de reducción de sesgo tras balancear; mayor entropía en $P(A\mid \text{tipo de }Q)$.
- **UU / UB / B$_\text{half}$B / BB:** notación train-test (Unbalanced/Balanced) de la Tabla 1.
