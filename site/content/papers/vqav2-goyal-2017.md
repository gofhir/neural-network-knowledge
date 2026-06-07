---
title: "Making the V in VQA Matter (VQAv2)"
weight: 241
math: true
---

{{< paper-card
    title="Making the V in VQA Matter: Elevating the Role of Image Understanding in VQA"
    authors="Goyal, Khot, Summers-Stay, Batra, Parikh"
    year="2017"
    venue="CVPR 2017"
    pdf="/papers/vqav2-goyal-2017.pdf"
    arxiv="1612.00837" >}}
Demuestra empíricamente que los modelos de Visual Question Answering de 2015-2016 respondían bien **sin mirar la imagen**, explotando los *language priors* (sesgos del lenguaje). La contribución no es un modelo sino un **rediseño del benchmark**: VQA v2.0, un dataset balanceado por construcción donde cada pregunta tiene un *par* de imágenes complementarias casi idénticas que dan respuestas distintas. Esto destruye el atajo del texto y obliga a que la **V** (Vision) de VQA realmente importe. Es el [fundamento Visual Question Answering](/fundamentos/visual-question-answering) hecho honesto y medible, y el benchmark estándar que aún reportan los VLMs modernos.
{{< /paper-card >}}

---

## Contexto: los language priors en VQA v1

VQA fue introducido por [VQA original (Antol 2015)](/papers/vqa-antol-2015): se acoplan imágenes reales de [COCO (Lin 2014)](/papers/coco-lin-2014) con preguntas de forma libre y respuestas anotadas por humanos. La promesa era que para responder bien el modelo tenía que *entender la imagen*. El problema, que este paper documenta con cifras devastadoras, es que **el lenguaje es un prior tremendamente fuerte**: la pregunta sola, sin imagen, ya determina la respuesta con altísima probabilidad.

Ejemplos concretos del VQA v1:

- "tennis" es la respuesta correcta para el **41%** de las preguntas que empiezan con "What sport is".
- "2" es correcta para el **39%** de las que empiezan con "How many".
- "Is there a clock" → "yes" el **98%** de las veces.
- "Is the man standing" → "no" el **69%** de las veces.

El caso más perverso: para las preguntas que empiezan con "Do you see a ...", responder ciegamente "yes" sin leer el resto de la pregunta ni mirar la imagen da **87% de accuracy de VQA**.

Hay dos fenómenos entrelazados. Primero, el **prior del mundo y del lenguaje**: la gente no pregunta al azar, así que $P(A \mid Q)$ está enormemente concentrada y el modelo puede memorizarla sin consultar los píxeles. Segundo, el **visual priming bias** (Zhang et al.): los anotadores *vieron* la imagen al escribir la pregunta, por lo que solo preguntan "¿hay un reloj?" cuando la imagen tiene un reloj; el acto mismo de preguntar filtra información sobre la respuesta y sesga el dataset hacia el "sí".

La consecuencia metodológica es grave: estos priors **dan una falsa impresión de progreso**. Un modelo reporta accuracy alta y parece "entender imágenes" cuando solo aprendió la estadística del texto. El paper aporta "la primera evidencia empírica concreta de lo que era una sospecha cualitativa entre los practicantes".

Un punto técnico crucial: **no basta con uniformizar $P(A)$** (la distribución marginal de respuestas). Aunque "yes" y "no" aparecieran 50/50 globalmente, los modelos siguen explotando la correlación $P(A \mid \text{n-grama}(Q))$. Lo que se necesita es **mayor entropía en $P(A \mid Q)$**, de modo que la imagen $I$ tenga forzosamente que jugar un rol. Esto motiva un balanceo **a nivel de cada pregunta individual**, no global. Esa es la clave de todo el diseño.

---

## Ideas principales

### El dataset balanceado con imágenes complementarias

La idea se enuncia en una frase. Para cada triplete $(I, Q, A)$ del VQA original se busca, con ayuda de un humano, **otra imagen $I'$ similar a $I$ pero para la cual la misma pregunta $Q$ tenga una respuesta $A'$ distinta**:

$$(I, Q, A) \quad \text{y} \quad (I', Q, A'), \qquad A' \neq A, \quad I' \approx I$$

¿Por qué funciona? Considera un modelo que solo procesa el lenguaje: ve $(Q, I)$ y $(Q, I')$. Como la pregunta es idéntica y el modelo ignora la imagen, *no tiene ninguna base para diferenciar los dos casos*. Producirá la misma respuesta para ambos y, por construcción, una de las dos estará mal. El atajo del lenguaje colapsa: $P(A \mid Q)$ ahora tiene alta entropía.

Hay un matiz que eleva la dificultad y que los autores destacan como característica deseada: $I'$ es **cercana a $I$ en el espacio semántico de la penúltima capa (fc7) de VGGNet**. Las dos imágenes no solo dan respuestas distintas, sino que se *parecen mucho* en el espacio de representaciones que las CNN aprenden. Por lo tanto, incluso un modelo que sí mira la imagen tiene que captar **diferencias sutiles** para responder bien en ambas. Esto convierte el balanceo en un test de razonamiento visual fino, no solo de "no ignorar la imagen".

La diferencia con trabajos previos importa. Zhang et al. habían estudiado el balanceo, pero solo en preguntas binarias sobre **escenas de clipart** (sintéticas, editables a voluntad). Goyal et al. generalizan a (1) **imágenes reales** de COCO, donde no puedes editar píxeles; (2) **todas las preguntas**, no solo binarias; (3) benchmarking del estado del arte; y (4) el modelo de explicación por contraejemplos.

### El proceso de construcción

El proceso descansa enteramente en Amazon Mechanical Turk (AMT) y tiene dos etapas. Se construye **encima** del VQA v1 de Antol et al., que contiene ~204K imágenes de COCO, 614K preguntas (≈3 por imagen) y >6 millones de respuestas (10 por pregunta).

**Etapa 1 — recolectar imágenes complementarias.** Para cada $(I, Q, A)$ se calculan los **24 vecinos más cercanos** de $I$, representando cada imagen con las activaciones fc7 de **VGGNet** y usando distancias $\ell_2$. A un trabajador se le muestran las 24 vecinas junto con $Q$ y $A$, y debe **elegir una $I'$ donde $Q$ "tenga sentido" y la respuesta sea distinta de $A$**. Que la pregunta "tenga sentido" significa que cualquier *premisa* que asume debe ser verdadera (la pregunta "What is the woman doing?" exige una imagen con mujer). Existe la opción **"not possible"** cuando ninguno de los 24 vecinos sirve, que constituye el **22%** de todas las preguntas, típicamente cuando el objeto es muy pequeño o el concepto es raro.

**Etapa 2 — recolectar respuestas nuevas.** Una vez elegida $I'$, se muestra a **10 nuevos trabajadores** y se recolectan **10 respuestas ground-truth**; la más común es $A'$. En **~9%** de los casos el voto mayoritario coincide con $A$ (desacuerdo humano o error del trabajador): ahí el balanceo no produjo una respuesta efectivamente distinta.

**Estadísticas finales:**

| Magnitud | VQA v2.0 (balanceado) |
|---|---|
| Pares (imagen, pregunta) totales | **~1.1 millones** (casi el doble de v1) |
| Respuestas asociadas | **~13 millones** |
| Imágenes base (COCO) | ~200K |
| Splits | train / val / test (test-dev, test-standard, test-challenge, test-reserve) |

Por el 22% de "not possible" y el 9% de $A=A'$, el dataset queda **significativamente más** balanceado, no perfectamente 50/50. La métrica cuantitativa de éxito: la **entropía de las distribuciones de respuesta, promediada por tipo de pregunta, aumenta un 56%** tras el balanceo. Un dataset mucho menos predecible desde el solo texto.

### Los contraejemplos como interpretabilidad

La tercera contribución conecta el balanceo con explicabilidad. Si para cada pregunta tengo $I$ y su complementaria $I'$ donde la respuesta cambia, puedo enseñar a un modelo a **explicar su respuesta exhibiendo un contraejemplo**: para "What color is the fire-hydrant? → red", el modelo agrega "a diferencia de esta" y muestra un hidrante que *no* es rojo. Es una explicación por *hard negatives* que construye confianza.

El modelo opera en dos pasos: (1) responde como un VQA convencional; (2) usa $A_\text{pred}$ y $Q$ para recuperar, entre los $K=24$ vecinos, una imagen similar pero con respuesta distinta. La supervisión viene "gratis" del balanceo: la $I'$ elegida por humanos *es* el contraejemplo correcto. La cabeza de explicación se entrena con una pérdida de ranking por bisagra que empuja el score de $I'$ por encima de las demás, combinada con la cross-entropy de la respuesta:

$$\mathcal{L} = -\log P(A \mid I, Q) + \lambda \sum_i \max\bigl(0,\, M - (S(I') - S(I_i))\bigr)$$

Evaluado con Recall@5 (frecuencia con que $I'$ aparece en el top-5), el modelo logra **43.39**, apenas por encima del baseline Distance (42.84) y muy por encima de Random (20.79). Los autores son honestos: identificar el contraejemplo correcto sigue siendo difícil, lo que "sugiere que los modelos de entendimiento visual capaces de extraer detalles finos siguen siendo elusivos".

---

## Resultados experimentales

El corazón empírico: si los modelos de v1 explotaban priors, deberían **caer** al evaluarse sobre el dataset balanceado. Se re-evalúan **d-LSTM+n-I** (modelo estándar de Antol et al.), **HieCoAtt** (co-atención jerárquica) y **MCB** (Multimodal Compact Bilinear, ganador del VQA Challenge 2016, con features de ResNet). La notación cruza train-test: U = Unbalanced, B = Balanced (primera letra = train, segunda = test).

| Approach | UU | UB | B$_\text{half}$B | BB |
|---|---|---|---|---|
| Prior | 27.38 | 24.04 | 24.04 | 24.04 |
| Language-only | 48.21 | 41.40 | 41.47 | 43.01 |
| d-LSTM+n-I | 54.40 | 47.56 | 49.23 | 51.62 |
| HieCoAtt | 57.09 | 50.31 | 51.88 | 54.57 |
| MCB | 60.36 | 54.22 | 56.08 | 59.14 |

Lecturas clave:

1. **La caída UU → UB** (mismo entrenamiento, cambia solo el test a balanceado) confirma la hipótesis: MCB cae ~6 puntos (60.36 → 54.22), HieCoAtt ~6.8, d-LSTM ~6.8. Los modelos habían aprendido sesgos que el val no balanceado *también* contenía; al quitarlos, el rendimiento se desploma.
2. **El baseline Language-only es revelador**: un modelo *ciego* que nunca ve píxeles alcanza ~48% en UU. Es la prueba más limpia de cuánta señal hay en el solo lenguaje, y cae a ~41% en UB.
3. **Re-entrenar en balanceado ayuda** (UB → B$_\text{half}$B): el train ya no premia el atajo. Y **más datos balanceados ayudan más** (B$_\text{half}$B → BB, ~2-3 puntos): los modelos están hambrientos de datos. Aun así, ningún modelo recupera el nivel de UU, pero ahora ese número refleja entendimiento visual real.

### Caídas v1 → v2 por tipo de pregunta

El desglose por tipo de respuesta (UU → UB) muestra dónde estaba el prior:

| Modelo | Tipo | UU | UB | Caída |
|---|---|---|---|---|
| MCB | Yes/No | 81.20 | 70.40 | **−10.8** |
| MCB | Number | 34.80 | 31.61 | −3.2 |
| MCB | Other | 51.19 | 47.90 | −3.3 |
| HieCoAtt | Yes/No | 79.99 | 67.62 | **−12.4** |
| HieCoAtt | Number | 34.83 | 32.12 | −2.7 |
| HieCoAtt | Other | 45.55 | 41.96 | −3.6 |

**Yes/No es donde más cae** (−10.8 a −12.4 puntos): los modelos explotaban fuertemente los sesgos de las preguntas binarias. Es la firma inequívoca del prior.

La observación más fina: en el VQA Challenge 2016, el gap entre los **top-4** approaches era de apenas **0.15%** en yes/no y **1.51%** en number. Como los priors llevan a *todos* los modelos a accuracies similares en esos tipos, el benchmark v1 los volvía **virtualmente indistinguibles**. **Balancear permite por fin distinguir un buen modelo de uno que solo memoriza sesgos.** Sobre test-standard de v2.0, MCB queda como mejor modelo con **62.27%** global (78.82% Yes/No, 38.28% Number, 53.36% Other) — nótese que "Number" sigue siendo durísimo.

Sobre el techo de dificultad: HieCoAtt entrenado en balanceado responde *ambas* imágenes de un par complementario correctamente en solo **17.7%** de los pares (vs 13.5% entrenado en no balanceado). En más del 80% de los pares no logra acertar las dos: el balanceo dejó al descubierto cuánto razonamiento visual fino sigue faltando.

---

## Limitaciones

1. **No es perfectamente balanceado.** El 22% de "not possible" y el 9% de $A=A'$ dejan prior residual explotable. Esto será central en la conexión con Pythia.
2. **Restringido a 24 vecinos** por $\ell_2$ sobre fc7 de VGGNet: una ventana mayor reduciría "not possible" pero encarece la tarea, y hereda los sesgos de esa representación.
3. **Solo balancea sobre las preguntas existentes de v1**: no genera preguntas nuevas, así que los sesgos en el *tipo* de preguntas que la gente formula permanecen.
4. **El modelo de explicación es modesto**: Recall@5 de 43.39 apenas supera a Distance (42.84).
5. **El conteo sigue pésimo** (~36-38% incluso para el mejor modelo): el balanceo expone la dificultad pero no la resuelve.

---

## Por qué importa hoy: el benchmark estándar

VQA v2.0 se convirtió en **el** benchmark estándar, desplazando por completo a v1.

- **VQA Challenge 2017 en adelante** corre sobre v2.0; reportar sobre v2.0 test-standard se vuelve obligatorio para publicar.
- **[Pythia (Jiang 2018)](/papers/pythia-jiang-2018)**, que ganó el VQA Challenge 2018, se entrena y evalúa sobre VQA v2.0. Es el caballo de batalla práctico de la clase, en el linaje d-LSTM+n-I → MCB → Bottom-Up que este paper benchmarkea.
- **Base de evaluación de los VLMs modernos**: desde Bottom-Up/Top-Down (Anderson 2018), LXMERT, ViLBERT y UNITER, hasta BLIP-2, Flamingo, LLaVA y GPT-4V, VQA v2.0 sigue siendo un benchmark reportado, con el VQA score $\min(\#\text{anotadores}/3, 1)$ sobre 10 anotadores como estándar de facto.
- **Cambio cultural**: instaló la conciencia de que **un benchmark mal balanceado infla las capacidades** y que el balanceo por hard negatives es una herramienta de diseño de datasets. Inspiró trabajos sobre *dataset bias*, *shortcut learning* y splits diagnósticos (GQA, VQA-CP, que invierte deliberadamente los priors entre train y test).

### Conexión con la Clase 23

Este paper es **exactamente** el dataset descrito en las slides 7-8 de la [Clase 23](/clases/clase-23): "204K imágenes de COCO, 614K preguntas (3 por imagen), 6M respuestas (10 por pregunta), conjunto balanceado: para cada triplete (I,Q,A) identifican otra imagen cercana a I que da una respuesta diferente a Q". Cada cifra y la mecánica del balanceo provienen literalmente de aquí.

La motivación pedagógica de mostrar VQA v2.0 en clase es la lección central: si entrenas un sistema multimodal sobre un dataset sesgado, obtendrás un modelo que *parece* entender imágenes pero explota la estadística del texto. Y las slides 14-19 muestran que **los language priors persisten incluso con un dataset balanceado** — lo que no contradice el paper, sino que lo confirma desde su propia honestidad. Como v2.0 está *significativamente más* balanceado pero no perfectamente, Pythia aún explota el prior residual: cuando en clase responde "amarillo" a "¿de qué color es el plátano?" sobre un plátano verde, estás viendo el prior que el balanceo no eliminó. El balanceo de Goyal et al. fue un avance enorme (entropía +56%, modelos distinguibles de nuevo) pero **no es una bala de plata**: por eso surgieron benchmarks aún más agresivos como VQA-CP. El paper no "arregla" VQA; lo hace *honesto y medible*, que es exactamente lo que un buen benchmark debe hacer.

---

## Notas y enlaces

- **Paper:** arXiv:1612.00837 (v3, 15 May 2017), CVPR 2017.
- **Proyecto y dataset:** [visualqa.org](https://visualqa.org/) — descarga de VQA v2.0, splits y script de evaluación oficial.
- **Antecedentes y linaje:** [VQA original (Antol 2015)](/papers/vqa-antol-2015) (el dataset v1 sobre el que se construye) · [COCO (Lin 2014)](/papers/coco-lin-2014) (fuente de las imágenes) · MCB (Fukui 2016, ganador del Challenge 2016).
- **Sigue:** [Pythia (Jiang 2018)](/papers/pythia-jiang-2018), Bottom-Up/Top-Down (Anderson 2018), y la familia de VLMs modernos.
- Ver el [fundamento Visual Question Answering](/fundamentos/visual-question-answering) y el [dominio Multimodal](/dominios/multimodal).
