# Show Your Work: Scratchpads for Intermediate Computation with Language Models — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Show Your Work: Scratchpads for Intermediate Computation with Language Models*.
- **Autores:** Maxwell Nye (MIT / Google Research), Anders Johan Andreassen, Guy Gur-Ari, Aitor Lewkowycz (Google Research, Blueshift Team), Jacob Austin, David Bieber, David Luan, David Dohan, Henryk Michalewski, Charles Sutton, Maarten Bosma, Augustus Odena (Google Research, Brain Team).
- **Preprint:** arXiv:2112.00114v1 (30 de noviembre de 2021), [arxiv.org/abs/2112.00114](https://arxiv.org/abs/2112.00114).
- **Año:** 2021.
- **Linaje:** se apoya directamente en *Program Synthesis with Large Language Models* (Austin et al., 2021), del cual reutiliza los modelos, el dataset MBPP y la tarea de "predicción directa de ejecución" que aquí sirve de línea base a superar.

El paper parte de una observación incómoda. Los grandes modelos de lenguaje pre-entrenados rinden asombrosamente bien en tareas que pueden resolverse "de una sola pasada" (*in one pass*) —generar texto realista, sintetizar programas cortos—, pero **fracasan en tareas que requieren cómputo multi-paso no acotado**: sumar enteros largos o predecir el resultado de ejecutar un programa. La propuesta es tan simple como poderosa: en vez de exigirle al modelo que emita la respuesta final directamente, se le permite **producir una secuencia arbitraria de tokens intermedios —un "scratchpad" o bloc de notas— antes de la respuesta**. En ese bloc el modelo escribe los pasos intermedios del algoritmo (los acarreos de una suma, los términos de un polinomio evaluados uno a uno, la traza línea por línea de un programa). Entrenando Transformers para que emitan estos pasos con supervisión estándar de máxima verosimilitud, el desempeño en cómputo multi-paso **mejora drásticamente**.

Los autores validan la idea en una serie de tareas de dificultad creciente: **adición de enteros largos**, **evaluación de polinomios** y, como caso más general que en cierto sentido subsume a los anteriores, **la ejecución paso a paso de programas Python arbitrarios**. En todos ellos el scratchpad supera de forma consistente a la predicción directa, tanto en el régimen de *few-shot* como en el de *fine-tuning*.

Para la **Clase 34 (Razonamiento)** este paper es una pieza de bisagra. Es citado explícitamente (slide 27) como "LLMs con bloc de notas": la idea de guiar al modelo para que escriba los pasos intermedios de un cálculo. Es el **precursor directo de Chain-of-Thought** (Wei et al., 2022). La diferencia esencial —que desarrollaremos al final— es de método: Nye et al. logran el comportamiento **entrenando** con trazas explícitas, mientras que Chain-of-Thought lo consigue un año después **solo por prompting**, sin tocar los pesos. Ambos comparten la misma intuición central: darle al modelo espacio para "mostrar su trabajo" (*show your work*, como pide el título) convierte problemas imposibles de una sola pasada en problemas resolubles.

## 2. Contexto: por qué los LLMs fallan en cómputo multi-paso

Antes de la solución conviene entender bien el problema. Los Transformers grandes exhiben capacidades impresionantes, incluida la de escribir código que resuelve problemas de programación simples (Chen et al., 2021; Austin et al., 2021). Pero tropiezan de forma sistemática con cálculos algorítmicos multi-paso, especialmente los que exigen razonamiento preciso y cómputo no acotado. El paper enumera varios síntomas concretos:

- **GPT-3 no puede sumar en few-shot** números de más de tres dígitos (Brown et al., 2020).
- Los modelos grandes **no logran predecir el resultado de ejecutar código Python**, ni siquiera código que ellos mismos son capaces de escribir como solución a un problema (Austin et al., 2021).
- Las redes recurrentes y de grafos estándar **no generalizan sistemáticamente** al predecir la salida de programas simples con bucles (Bieber et al., 2020).

De aquí sale una observación filosóficamente interesante: los modelos pueden **escribir** código, pero no parecen **representar la semántica** del código que escriben, porque no pueden predecir su ejecución. Saben la forma pero no el significado.

¿Por qué ocurre esto? La hipótesis de los autores es arquitectónica, no de capacidad bruta. El problema está en **cómo se aplica** el Transformer a estas tareas: se le pide realizar el cómputo **en una sola pasada hacia adelante** (*a single forward pass*). Con un número fijo de capas y una cantidad fija de cómputo, el modelo **no puede adaptar la cantidad de cómputo que dedica a un problema según su dificultad** antes de producir la salida. Un problema de suma de 3 dígitos y otro de 30 dígitos reciben exactamente el mismo presupuesto de cómputo. En una nota al pie, los autores precisan el argumento teórico: un Transformer realiza un cómputo cuadrático en la longitud de la secuencia de entrada, de modo que en teoría no puede simular perfectamente algoritmos de complejidad temporal mayor que $O(n^2)$ —aunque advierten que no está claro cuán relevante es esta cota en la práctica, dado que la predicción neuronal es aproximada y los modelos pueden ser lo bastante grandes como para memorizar soluciones correctas en un subespacio relevante de entradas.

Trabajos previos habían atacado la limitación **modificando la arquitectura**: redes que permiten elegir dinámicamente cuánto tiempo de cómputo dedicar a distintas sub-tareas (Adaptive Computation Time de Graves, 2016; PonderNet de Banino et al., 2021; Universal Transformers de Dehghani et al., 2018). La apuesta de este paper es la contraria y mucho más pragmática: **no tocar el modelo ni el procedimiento de entrenamiento, sino rediseñar la tarea**. Aprovechar los Transformers existentes y los grandes modelos capaces de few-shot, cambiando únicamente el formato del objetivo.

## 3. Contribución central

La propuesta se resume en una frase: **permitir que el modelo produzca una secuencia arbitraria de tokens intermedios —el scratchpad— antes de la respuesta final**. Para entrenar el modelo, se codifican como texto los pasos intermedios del algoritmo y se usa entrenamiento supervisado estándar.

Esta simple modificación tiene varias ventajas potenciales que el paper articula con precisión:

1. **Cómputo adaptativo (*adaptive computation time*).** Al poder escribir un scratchpad de longitud arbitraria, el modelo procesa la información durante tanto tiempo como haga falta, dependiendo de la complejidad del problema. Un problema difícil genera un scratchpad más largo, y por tanto recibe más pasos de cómputo. Esto resuelve directamente la limitación diagnosticada en la sección anterior.
2. **Memoria externalizada.** El modelo puede almacenar el estado intermedio de su cómputo en el buffer del scratchpad y **volver a consultarlo atendiendo a su propio contexto**. Esto elimina la necesidad de guardar todo el estado intermedio en las activaciones internas de la red.
3. **Cuantización de errores.** Al forzar al modelo a emitir estados intermedios concretos muestreando del modelo generativo, se busca **reducir la propagación y acumulación de errores pequeños**, porque los estados quedan cuantizados a embeddings de tokens. Los errores compuestos son un problema típico de métodos como las Neural Turing Machines (Graves et al., 2014), que usan recurrencia con estados continuos para sostener cómputos extendidos.
4. **Interpretabilidad y depuración.** Examinar la salida del scratchpad **ayuda a identificar errores comunes y corregirlos revisando el formato**. Los autores reportan que esta capacidad de interpretar errores les resultó útil durante el propio trabajo.

Las contribuciones concretas, tarea por tarea, son: introducir la noción de scratchpad (Sección 2); mostrar que ayuda en la **adición larga** en régimen de fine-tuning, mejorando en particular la generalización **fuera de distribución** a instancias más grandes (Sección 3); mostrar que ayuda en la **evaluación de polinomios**, tanto en few-shot como en fine-tuning (Sección 4); y, en el contexto más general, mostrar que entrenar Transformers para emitir **trazas completas de programa línea por línea, anotadas con las variables locales**, mejora drásticamente su capacidad de predecir el resultado de ejecutar un programa dado (Sección 5).

## 4. Método

### 4.1. Dos problemas: inducción de algoritmos y aprender a ejecutar

El paper enmarca su trabajo en dos problemas relacionados. En ambos el objetivo es que la red aprenda a emular una función $f$ "algorítmica" (representable por un programa corto, como la suma o la evaluación de polinomios) a partir de su comportamiento entrada-salida.

- **Inducción de algoritmos (*algorithm induction*):** el objetivo es aprender **un solo** algoritmo. Cada ejemplo da una entrada y la salida deseada como strings. Los datos de entrenamiento son $D = \{x_i, f(x_i)\}_{i=1}^{N}$.
- **Aprender a ejecutar (*learning to execute*):** se quiere que el modelo produzca el resultado de **un programa cualquiera**, representado como código fuente, sobre alguna entrada. Si $\pi_i$ es el código fuente de un programa $f_i$, los datos son $D = \{(\pi_i, x_i, f_i(x_i))\}_{i=1}^{N}$.

La idea principal es la misma en ambos casos: **codificar los pasos intermedios del algoritmo como texto y entrenar al modelo para que los emita a un buffer que llamamos scratchpad**. En el caso de "aprender a ejecutar" simplemente se antepone el código fuente $\pi_i$ a la entrada, el scratchpad y la salida deseada.

En tiempo de entrenamiento el modelo recibe la entrada más el objetivo, con entrenamiento estándar basado en verosimilitud. En tiempo de prueba recibe **solo la entrada** y debe predecir el objetivo, por ejemplo mediante *beam search* o muestreo con temperatura. En principio cualquier modelo de secuencia serviría; los autores eligen **Transformers decoder-only**, aunque señalan que modelos encoder-decoder o recurrentes también podrían funcionar.

### 4.2. El formato del scratchpad, ilustrado

El formato es lo suficientemente concreto como para mostrarlo. Para enseñar a un modelo a sumar 29 y 57, un ejemplo de entrenamiento se ve así (los comentarios marcados con `#` se añaden por claridad y **no** forman parte del objetivo):

```
Input:
2 9 + 5 7
Target:
<scratch>
2 9 + 5 7 , C: 0
2 + 5 , 6 C: 1   # sumó 9 + 7 = 6 lleva 1
, 8 6 C: 0       # sumó 2 + 5 + 1 = 8 lleva 0
0 8 6
</scratch>
8 6
```

El scratchpad, delimitado por `<scratch>` … `</scratch>`, contiene los resultados intermedios del algoritmo escolar de suma larga: el acarreo (*carry*) se registra en el dígito que sigue a `C:`. Un detalle de implementación importante: se **introducen espacios entre los dígitos** para asegurar que cada dígito se mapee a un token separado, evitando que el tokenizador agrupe dígitos y arruine la aritmética.

Para la evaluación de polinomios, cada término se computa por separado y luego se suman. Evaluar $-7x^2 + 7x + 5$ en $x = 1$ produce un scratchpad donde `-7*x**2: -7`, `7*x: 7`, `5: 5` y finalmente `total: 5`.

Para la ejecución de programas, el objeto clave es la **traza (*trace*)**: una secuencia alternada de 1) la secuencia ordenada de líneas de código fuente ejecutadas, y 2) el estado de las variables locales después de ejecutar cada línea. La traza captura tanto el **flujo de control** (qué operaciones se ejecutan y en qué orden) como **cómo cambia el estado** tras cada operación. Se representa como string, con la línea de código reproducida directamente y el estado como un diccionario JSON. La salida correcta del programa aparece en la última línea de la traza, asignada a la variable `output`. Nótese que **todos los bucles se despliegan (*unrolled*) a lo largo del tiempo**: cada iteración aparece explícitamente en la traza, lo que es precisamente lo que da al modelo el cómputo adaptativo que necesita.

### 4.3. Modelos y configuración

Todos los experimentos usan **Transformers densos decoder-only pre-entrenados**, con tamaños que van desde **2 millones hasta 137 mil millones de parámetros**. Fueron pre-entrenados en documentos web y datos de diálogo, y corresponden a los modelos usados en Austin et al. (2021). Para los experimentos de código Python se usó el modelo de **~137 mil millones de parámetros**, con una ventana de contexto de 1024 tokens y un límite de 512 tokens de generación; salvo indicación contraria, el fine-tuning usó batch de 8192 tokens, learning rate $3 \times 10^{-5}$ y decodificación *greedy* (temperatura $T = 0$).

## 5. Resultados

### 5.1. Adición de enteros largos

Se entrenaron varios modelos en sumas de 1 a 8 dígitos (100 000 ejemplos, 5000 pasos, batch 32), evaluándose luego dentro de distribución (hasta 8 dígitos) y **fuera de distribución** (9 y 10 dígitos). La línea base incluye la entrada y el resultado, pero sin pasos intermedios.

Los hallazgos son nítidos. **Más allá de un tamaño crítico de modelo**, los modelos con scratchpad resuelven la tarea de suma, mientras que los modelos entrenados sin scratchpad fracasan incluso a la mayor escala probada. En las tareas fuera de distribución (9 y 10 dígitos), los modelos sin scratchpad **fracasan por completo**, mientras que los que usan scratchpad muestran mejora consistente en función del tamaño del modelo. Es decir, el scratchpad no solo mejora el desempeño dentro de distribución sino que habilita la **generalización a longitudes no vistas** —justamente lo que uno esperaría de un modelo que ha aprendido el algoritmo en lugar de memorizar respuestas.

Un ablation revelador (Apéndice B): si se toma el modelo con scratchpad y luego se le hace fine-tuning para que produzca la respuesta directamente (sin scratchpad), **no hay mejora significativa** respecto de la línea base directa. Esto indica que la información extra vista en tiempo de entrenamiento **no es por sí sola** la responsable del mejor desempeño; es el uso del scratchpad **en tiempo de inferencia** lo que importa.

### 5.2. Evaluación de polinomios

Se generaron polinomios de grado $\leq 3$, con coeficientes enteros, entradas en $[-10, 10]$ y salidas restringidas a $[-1000, 1000]$ (10 000 de entrenamiento, 2000 de test). Se evaluó en dos regímenes: **few-shot** con el modelo de 137B (usando $n = 4$ ejemplos en el prompt) y **fine-tuning** con un modelo de 8B (2000 pasos). Los resultados:

| Método | Few-shot | Fine-tuning |
|---|---|---|
| Predicción directa | 8.8 % | 31.8 % |
| Scratchpad | **20.1 %** | **50.7 %** |

El scratchpad supera a la predicción directa de forma significativa en **ambos** regímenes: más que duplica el desempeño en few-shot (de 8.8 % a 20.1 %) y lo eleva en más de 18 puntos en fine-tuning (de 31.8 % a 50.7 %).

### 5.3. Ejecución de programas Python

Aquí está el resultado más ambicioso, porque **subsume** a los anteriores: en vez de diseñar a mano los estados intermedios de cada algoritmo, se pide al modelo aprender a implementar cualquier algoritmo **ejecutando código arbitrario**. La línea base es la "predicción directa de ejecución" de Austin et al. (2021): mostrar el código de una función y pedir la salida sobre una entrada específica. La alternativa es la **traza vía scratchpad**: predecir primero la secuencia de estados intermedios computados durante la ejecución.

**Programas sintéticos** (prueba de concepto, dataset modificado de Bieber et al. 2020, con enteros pequeños, bucles `while` e `if`; 400 entrenamiento / 100 validación / 200 test):

| Método | Few-shot | Fine-tuned |
|---|---|---|
| Predicción directa | 11 % | 20 % |
| Scratchpad | **26.5 %** | **41.5 %** |

(El criterio de precisión en el caso few-shot con scratchpad se ajustó ligeramente: el modelo tendía a mantener el nombre de variable `v0` en vez de reasignar a `output` en la línea final, un error consistente y puramente de formato; bajo un scoring ingenuo la precisión few-shot sería cercana a cero.)

**Programas reales** (dataset MBPP, 1000 problemas de programación con especificación en lenguaje natural, programa de referencia y tres casos de test; se reportan sobre el subconjunto de 212 tareas cuya traza cabe en la ventana de generación):

- **Régimen de muy pocos datos.** Entrenando solo con los 374 problemas de MBPP (1122 ejemplos), ni el modelo de scratchpad ni el de ejecución directa rinden bien (5 % y 10 % de precisión de salida, respectivamente); aquí la **ejecución directa incluso supera** al scratchpad. La traza es más difícil de aprender con tan pocos datos.
- **Aumento de datos con programas muestreados (MBPP-aug).** Se genera un dataset mucho mayor haciendo que el modelo pre-entrenado de 137B sintetice 80 programas candidatos por tarea ($T = 0.5$); cada candidato ejecutado sobre las entradas originales produce nuevos ejemplos (se descartan los que dan error), resultando en 17 000 programas nuevos. El efecto es asimétrico y muy ilustrativo: **este dato adicional daña** a la ejecución directa, pero **mejora enormemente** al scratchpad, que pasa a resolver más del triple de tareas que con solo los datos originales. La razón conceptual: los programas candidatos, aun cuando produzcan salidas distintas de las originales o incluso sean incorrectos, siguen siendo **trazas de ejecución válidas** y por tanto señal de entrenamiento útil para aprender la semántica del lenguaje.
- **Datos adicionales (single-line y CodeNet).** Añadir un dataset de ~9 millones de transformaciones Python de una sola línea (recolectado por Fraser Greenlee) y 670 904 trazas extraídas de Project CodeNet (Puri et al., 2021) —incluyendo programas con errores, cuya traza termina con el mensaje de error— mejora aún más. El mejor modelo combinado (**MBPP-aug + CodeNet + single-line**) alcanza **26.6 %** de tareas ejecutadas correctamente y traza perfectamente casi un cuarto de las tareas (**24.6 %**). A nivel de ejemplos individuales, la decodificación greedy produce la traza exactamente correcta para **casi el 42 %** de todas las trazas. Que la mejora escale al añadir datos ligeramente fuera de distribución sugiere que la técnica **escala bien con más datos**.

Un control importante (Apéndice A): el fine-tuning en la tarea de traza **no destruye** la capacidad del modelo de sintetizar programas. El modelo entrenado con trazas logra 54 % de precisión de síntesis few-shot en MBPP, frente al 62 % del modelo original —una caída moderada, no un colapso.

## 6. Limitaciones

- **Tamaño de la ventana de contexto.** Todos los experimentos se limitan a problemas cuyo scratchpad cabe en la ventana de generación (512 tokens). Muchos problemas reales requieren scratchpads mucho más largos, de modo que realizar plenamente el potencial de la técnica exige mejoras en el tamaño de la ventana de generación del Transformer —un área activa de investigación en NLP (Tay et al., 2020). En MBPP, esta restricción reduce el conjunto evaluable de 500 a 212 tareas.
- **Necesidad de supervisión con trazas.** En la inducción de algoritmos hay que **diseñar a mano** los estados intermedios de cada tarea nueva, lo que los propios autores reconocen como sub-óptimo. La ejecución de programas alivia esto (las trazas se generan automáticamente con un intérprete), pero el patrón general sigue dependiendo de datos de traza.
- **Aprender a usar el scratchpad sin supervisión.** Un siguiente paso claro es que el modelo aprenda a usar el scratchpad **sin supervisión directa**, por ejemplo con aprendizaje por refuerzo: recompensar respuestas correctas con una recompensa inversamente proporcional al número de tokens de scratchpad usados. La esperanza es que usar el scratchpad sea una **habilidad transferible** —que el algoritmo aprendido para la suma larga ayude en la evaluación de polinomios. (Este es, de hecho, el germen de la investigación posterior en razonamiento auto-inducido.)

## 7. Conexión con la Clase 34 y con Chain-of-Thought

En la Clase 34 (Razonamiento), el profesor Amenábar presenta este paper con la metáfora del **mono recolectando cocos**: así como no basta con premiar al mono por el resultado, sino que conviene guiarlo para que aprenda el **procedimiento** de recolección, aquí no le pedimos al modelo la respuesta final de golpe, sino que **lo guiamos para que emita el algoritmo paso a paso**. El scratchpad es exactamente eso: un andamiaje que fuerza al modelo a exteriorizar el cómputo en vez de intentar comprimirlo todo en una sola pasada opaca. El título del paper —*Show Your Work*, "muestra tu trabajo"— es la instrucción que todo profesor de matemáticas le da a un alumno: no me des solo el número, muéstrame cómo llegaste. Resulta que a los Transformers les pasa lo mismo que a los alumnos: mostrar el trabajo no es solo para que el profesor verifique, sino que **el acto mismo de escribir los pasos hace posible resolver el problema**.

La conexión con **Chain-of-Thought (Wei et al., 2022)** es la del precursor con su sucesor. Ambos comparten la tesis central: darle al modelo espacio para generar pasos de razonamiento intermedios antes de la respuesta transforma problemas irresolubles de una sola pasada en problemas resolubles. La diferencia es de **método de obtención del comportamiento**:

| Eje | Scratchpad (Nye et al., 2021) | Chain-of-Thought (Wei et al., 2022) |
|---|---|---|
| Cómo se obtiene | **Fine-tuning** con trazas explícitas | **Prompting** con ejemplos few-shot que muestran el razonamiento |
| Toca los pesos | Sí (entrenamiento supervisado) | No (solo el prompt) |
| Dominio | Cómputo algorítmico (aritmética, ejecución de código) | Razonamiento en lenguaje natural (aritmético, sentido común, simbólico) |
| Formato del paso intermedio | Estructurado (acarreos, trazas JSON) | Cadena de razonamiento en prosa |
| Requisito | Datos de traza etiquetados | Un modelo suficientemente grande (capacidad emergente) |

Chain-of-Thought es, en un sentido preciso, "el scratchpad sin fine-tuning": el mismo comportamiento de escribir pasos intermedios, pero disparado solo con el prompt, aprovechando que los modelos suficientemente grandes ya lo tienen latente. Esta línea desemboca directamente en el paradigma de razonamiento moderno —*zero-shot CoT* ("pensemos paso a paso"), *self-consistency*, y en última instancia los **modelos de razonamiento** que dedican cómputo de inferencia extendido antes de responder. Todos son descendientes de la intuición que este paper formalizó primero: **el cómputo intermedio explícito es lo que separa a un LLM que adivina de uno que razona.**

## 8. Nota final: relevancia para salud

En pipelines clínicos la diferencia entre una respuesta directa y una respuesta con cómputo intermedio explícito no es cosmética, es de seguridad. Un modelo que calcula una dosis por peso, ajusta por función renal, deriva un puntaje de riesgo o interpreta una regla de decisión clínica **no debe entregar el número final de golpe**: debe mostrar la derivación paso a paso —igual que el scratchpad despliega los acarreos de una suma o la traza de un programa— para que cada estado intermedio sea **auditable** por un profesional de salud y verificable contra el algoritmo clínico de referencia. Esto ataca dos riesgos concretos que este paper ya anticipa en su dominio: la **acumulación de errores** (un desliz en un paso intermedio que se propaga silenciosamente hasta una conclusión peligrosa) y la **opacidad** (un veredicto sin cadena de razonamiento es imposible de trazar cuando falla). El scratchpad, además, cuantiza los estados intermedios a tokens concretos, lo que en un contexto de historia clínica o de motor de reglas equivale a dejar un registro discreto e inspeccionable de cómo se llegó a cada valor —un requisito no negociable en cualquier sistema de apoyo a la decisión clínica que aspire a ser confiable y regulable.
