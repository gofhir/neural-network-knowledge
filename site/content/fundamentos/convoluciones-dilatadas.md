---
title: "Convoluciones Dilatadas"
weight: 127
math: true
---

Una **convolución dilatada** —también llamada *atrous* (del francés *à trous*, "con agujeros")— es una convolución cuyo filtro toma muestras separadas por $d$ posiciones en lugar de contiguas. El filtro conserva exactamente los mismos $k$ pesos, pero abarca un tramo $d$ veces más largo. Es una de las herramientas más económicas del deep learning: **no agrega un solo parámetro, no cambia el costo por posición de salida, y no reduce la resolución** — y aun así permite que el campo receptivo de una red crezca exponencialmente con la profundidad en vez de linealmente.

Nació en el procesamiento de señales como el algoritmo *à trous* de la transformada wavelet, y llegó al deep learning por dos caminos casi simultáneos en 2016: [WaveNet](/papers/wavenet-oord-2016) la usó para generar audio muestra a muestra, y Yu y Koltun la usaron para segmentación semántica de imágenes, donde el problema era el opuesto pero simétrico — ahí el pooling destruía la resolución espacial que la tarea necesitaba. Hoy aparece en las redes convolucionales temporales (TCN), en la familia DeepLab, en ByteNet y en cualquier arquitectura que necesite contexto largo sobre una salida densa.

Este fundamento cubre el operador de forma autónoma: la definición, la aritmética del campo receptivo, la condición que separa una progresión de dilataciones sana de una que deja huecos, y el criterio para decidir cuándo la dilatación es la herramienta correcta y cuándo no.

---

## 1. El problema: el campo receptivo no escala

Una convolución mira un vecindario local. Para que una neurona de la última capa vea un tramo grande de la entrada, hay tres caminos, y los tres tienen un costo.

Sea una pila de $L$ capas donde la capa $l$ tiene kernel $k_l$, stride $s_l$ y dilatación $d_l$. El **campo receptivo** —el número de posiciones de la entrada que influyen sobre una posición de la salida— es

$$R_L = 1 + \sum_{l=1}^{L} (k_l - 1)\, d_l \prod_{i=1}^{l-1} s_i$$

Cada capa aporta $(k_l - 1)$ posiciones, amplificadas por su propia dilatación y por el stride acumulado de todas las anteriores. De ahí salen las tres estrategias:

| Estrategia | Qué crece | Qué cuesta |
|---|---|---|
| **Kernel más grande** | $k_l$ | Parámetros y cómputo, ambos lineales en $k$ |
| **Más profundidad** | El número de sumandos | Parámetros, cómputo y dificultad de optimización |
| **Stride o pooling** | $\prod s_i$ | **Resolución de la salida** |
| **Dilatación** | $d_l$ | Nada directo — pero deja huecos |

La cuarta fila es la que llama la atención, y por eso conviene entender exactamente qué se paga.

{{< concept-alert type="clave" >}}
**La cuenta que motiva todo, con números de audio.** Para cubrir un segundo de señal a 16 kHz —16.000 muestras— con kernels de tamaño 3 sin stride ni dilatación hacen falta

$$L = \frac{16\,000 - 1}{2} = 7\,999 \text{ capas}$$

Con dilatación exponencial ($d_l = 2^{l-1}$), **13**. La diferencia no es de grado: una red de ocho mil capas no es cara, es imposible. En audio crudo, donde la señal tiene tres o cuatro órdenes de magnitud más de muestras por unidad de contenido que el texto o la imagen, la dilatación no es una optimización sino un habilitador.
{{< /concept-alert >}}

---

## 2. La definición

Para una señal 1D $x$ y un filtro $w$ de tamaño $k$, la convolución dilatada con factor $d$ es

$$(x *_d w)[i] \;=\; \sum_{j=0}^{k-1} x[i + d \cdot j]\, w[j]$$

Con $d = 1$ se recupera la convolución estándar. El caso 2D es idéntico aplicando la dilatación a cada eje, y no tiene por qué ser la misma en ambos: en audio es habitual dilatar solo el eje temporal y dejar el de frecuencia intacto, por la misma razón por la que el pooling va solo en frecuencia (ver [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel) y la [profundización de la Clase 39](/clases/clase-39/profundizacion)).

Tres propiedades que conviene tener claras:

**No cambia el número de parámetros.** El filtro sigue teniendo $k$ pesos. Un kernel de 3 con dilatación 512 tiene tres pesos, igual que uno con dilatación 1.

**No cambia el costo por posición de salida.** Se hacen las mismas $k$ multiplicaciones. Lo que cambia es *de dónde* se leen los operandos, lo que sí tiene efecto en la localidad de memoria: una convolución muy dilatada tiene peor comportamiento de caché que una densa equivalente, aunque el conteo de FLOPs sea idéntico.

**No reduce la resolución.** A diferencia del stride, la salida tiene tantas posiciones como la entrada (con padding adecuado). Esta es la propiedad que la hace insustituible cuando la salida debe ser densa.

### Implementación

Todos los frameworks lo exponen como un argumento, no como una capa aparte:

```python
# PyTorch
nn.Conv1d(canales_in, canales_out, kernel_size=3, dilation=8)

# TensorFlow / Keras
tf.keras.layers.Conv1D(filtros, 3, dilation_rate=8)

# JAX (lax directamente)
jax.lax.conv_general_dilated(x, w, window_strides=(1,), padding="SAME",
                             rhs_dilation=(8,),          # dilata el KERNEL
                             dimension_numbers=("NWC", "WIO", "NWC"))
```

{{< concept-alert type="advertencia" >}}
**El gotcha de JAX: `rhs_dilation` y `lhs_dilation` son cosas distintas.** `rhs_dilation` dilata el operando derecho —el kernel— y es la convolución dilatada de la que trata este fundamento. `lhs_dilation` dilata la **entrada**, insertando ceros entre sus posiciones, y sirve para implementar convoluciones transpuestas (*upsampling*). Confundirlas produce una red que corre sin errores y hace algo completamente distinto.
{{< /concept-alert >}}

---

## 3. El crecimiento exponencial

El caso más usado es la dilatación que se duplica: $d_l = 2^{l-1}$. Con kernel $k$ y $L$ capas, el campo receptivo es una serie geométrica que colapsa a una expresión notablemente limpia:

$$R_L = 1 + (k-1)\sum_{l=1}^{L} k^{l-1} = 1 + (k-1)\cdot\frac{k^L - 1}{k-1} = k^L$$

cuando la base de la progresión es el propio $k$. Con $k = 2$ y dilataciones $1, 2, 4, \dots, 512$ (diez capas), el campo receptivo es exactamente $2^{10} = 1\,024$ muestras. Ese es el bloque de WaveNet, y a 16 kHz corresponde a 64 milisegundos.

| Capas | Dilataciones | $R$ con $k=2$ | @16 kHz |
|---|---|---|---|
| 10 | $1 \dots 512$ | 1.024 | 64 ms |
| 20 | dos bloques | 2.047 | 128 ms |
| 30 | tres bloques | 3.070 | 192 ms |
| 40 | cuatro bloques | 4.093 | 256 ms |

Nótese que **repetir el bloque suma, no multiplica**: cuatro bloques de diez capas dan 4.093, no $1024^4$. Reiniciar la dilatación a 1 al empezar cada bloque es una decisión deliberada, y la sección siguiente explica por qué.

---

## 4. El costo real: los huecos

Un filtro dilatado no mira las posiciones intermedias. Apilando capas con dilataciones mal elegidas se llega a una situación en la que **hay posiciones de la entrada que ninguna ruta del grafo computacional consulta**. El artefacto se conoce como *gridding* o *checkerboard*, y en imágenes produce patrones de tablero visibles en los mapas de activación.

La condición que lo evita se deriva en dos pasos. Tras la capa $l$, cada posición de salida resume un tramo contiguo de $R_l$ posiciones. La capa $l+1$ toma $k_{l+1}$ de esas posiciones separadas por $d_{l+1}$. Para que los tramos resumidos por dos tomas consecutivas se toquen y no dejen vacío entre medio:

$$\boxed{\;d_{l+1} \;\le\; R_l\;}$$

Tomar siempre el máximo permitido da el crecimiento más rápido posible sin huecos, y en ese caso cada capa multiplica el campo receptivo por su kernel:

$$R_{l+1} = R_l + (k_{l+1}-1)R_l = k_{l+1}\,R_l$$

{{< concept-alert type="clave" >}}
**La duplicación de WaveNet es el óptimo, pero solo para $k=2$.** Con kernel 2, la regla $d_{l+1} = R_l$ produce exactamente $1, 2, 4, 8, \dots$ Esa coincidencia explica dos cosas: por qué WaveNet no sufre gridding dentro de un bloque, y por qué su esquema se copió tanto sin cuestionarlo.

El error aparece al trasladar la duplicación a kernels distintos. Con kernels $20, 10, 10, 5$ —los del "Ejemplo 2" de la [Clase 39](/clases/clase-39)— duplicar da un campo receptivo de 106 muestras, mientras que el programa óptimo $1, 20, 200, 2000$ da **10.000**. Dos órdenes de magnitud de diferencia, con los mismos parámetros y el mismo cómputo.

La regla que hay que llevarse no es "duplica la dilatación" sino **"haz que la dilatación siga al campo receptivo acumulado"**.
{{< /concept-alert >}}

Violar la condición tiene un efecto medible. Con los mismos cuatro kernels y dilataciones $1, 16, 256, 4096$, el campo receptivo nominal es de 18.852 muestras, pero solo 8.200 de ellas reciben gradiente: **10.652 posiciones son invisibles para la red, un 56.5% del tramo**. La medición y el código que la produce están en la [práctica de la Clase 39](/clases/clase-39/practica/01-campo-receptivo-y-dilatacion).

### Las dos mitigaciones estándar

**Repetir bloques.** En vez de una sola progresión $1, 2, 4, \dots, 2^{L-1}$, se apilan varios bloques cortos que reinician en 1. Cada bloque nuevo consulta, con dilatación baja, las posiciones que el anterior salteó. Es lo que hace WaveNet, y la razón por la que su campo receptivo suma en vez de multiplicar: el precio de la cobertura.

**Dilataciones coprimas o híbridas (HDC).** Elegir factores sin divisores comunes —$1, 2, 5$ en vez de $1, 2, 4$— de modo que las grillas de muestreo de capas sucesivas no se alineen. Es la solución habitual en segmentación semántica.

---

## 5. Cuándo usarla y cuándo no

La pregunta que decide es simple: **¿la salida debe ser densa en el eje que se está convolucionando?**

| Tipo de salida | Ejemplos | Herramienta correcta |
|---|---|---|
| **Densa, misma resolución que la entrada** | Generación de forma de onda, síntesis, separación de fuentes, segmentación semántica | **Dilatación.** El stride está vedado |
| **Densa, menor resolución** | Reconocimiento de voz (unos pocos tokens por segundo), detección de eventos con marcas de tiempo | Submuestrear hasta esa resolución, y dilatar de ahí en adelante |
| **Una etiqueta por ejemplo** | Clasificación de audio, tagging, clasificación de imágenes | **Stride y pooling.** Más barato, y la resolución no importa |

El contraste entre [WaveNet](/papers/wavenet-oord-2016) y la [familia M de Dai et al.](/papers/raw-waveforms-dai-2017) ilustra la regla exactamente. Ambos procesan onda cruda y ambos necesitan campos receptivos de cientos de milisegundos. WaveNet **no puede** submuestrear —su salida es una muestra de audio por cada muestra de entrada— así que la dilatación es su única herramienta. La familia M clasifica, así que puede colapsar el eje temporal por un factor de 1.024 con stride y pooling, y llega a 1.5 segundos de campo receptivo con 3.7 millones de parámetros y sin una sola convolución dilatada.

Comparando el costo de las tres estrategias para cubrir un segundo a 16 kHz con 64 canales por capa:

| Estrategia | Capas | Parámetros | MACs | Stride final |
|---|---|---|---|---|
| Densa, $k=3$ | 7.999 | 98.3M | 1.572,7 G | 1 |
| Dilatada exponencial, $k=3$ | 13 | 0.2M | 2,6 G | 1 |
| Stride 2, $k=3$ | 14 | 0.2M | 0,2 G | 16.384 |

La dilatación es 500 veces más barata en parámetros que la densa, con la misma resolución de salida. El stride es otras 13 veces más barato que la dilatación, pero paga con toda la resolución.

---

## 6. Dónde más aparece

**Segmentación semántica (DeepLab, 2015-2017).** El problema simétrico al del audio: las CNN de clasificación reducen la resolución espacial por un factor de 32 con pooling y stride, pero segmentar exige una etiqueta por píxel. La solución de DeepLab es reemplazar los últimos pooling por convoluciones dilatadas —conservando el campo receptivo sin perder resolución— y agregar el módulo ASPP, que aplica varias dilataciones en paralelo para capturar objetos a múltiples escalas.

**Redes convolucionales temporales (TCN, 2018).** La generalización del bloque de WaveNet a cualquier tarea de secuencia: convoluciones causales dilatadas con conexiones residuales, propuestas como alternativa a las RNN. Su argumento es el mismo de los Transformers pero por otra vía — el entrenamiento es paralelizable y la longitud del camino entre posiciones es $O(\log T)$ en vez de $O(T)$. Ver [redes recurrentes](/fundamentos/redes-recurrentes) para el contraste.

**Modelos de secuencia en texto.** ByteNet y las primeras arquitecturas convolucionales de traducción, antes de que la [self-attention](/fundamentos/self-attention) desplazara a ambas familias.

{{< concept-alert type="recordar" >}}
**Cómo se compara con la atención.** Los tres operadores que pueden dar contexto largo se ordenan así por longitud máxima del camino entre dos posiciones —la métrica de Vaswani et al. 2017:

| Operador | Camino máximo | Costo por capa | Operaciones secuenciales |
|---|---|---|---|
| Recurrente | $O(T)$ | $O(T d^2)$ | $O(T)$ |
| Convolución densa | $O(T/k)$ | $O(k T d^2)$ | $O(1)$ |
| **Convolución dilatada** | $O(\log_k T)$ | $O(k T d^2)$ | $O(1)$ |
| Self-attention | $O(1)$ | $O(T^2 d)$ | $O(1)$ |

La atención gana en camino y pierde en costo, y el cruce está en $T = d$. Por eso los modelos de audio con Transformer no atienden sobre la onda cruda: primero la reducen con convoluciones hasta unas 50 tramas por segundo, y recién ahí atienden. La dilatación y la atención no compiten — se usan en etapas distintas del mismo modelo.
{{< /concept-alert >}}

---

## 7. Errores frecuentes

**Copiar la duplicación con kernels grandes.** Ya cubierto: es el que más cobertura desperdicia y el más difícil de notar, porque la red entrena igual.

**Olvidar que el padding cambia.** Con dilatación $d$ y kernel $k$, el padding necesario para preservar la longitud es $\frac{(k-1)d}{2}$, no $\frac{k-1}{2}$. Los frameworks con `padding="same"` lo calculan solo; los que piden un entero, no.

**Usar dilatación cuando el problema era el stride.** Si la salida es una etiqueta, la dilatación resuelve un problema que no se tiene y cuesta más que el pooling.

**Apilar muchas capas con la misma dilatación alta.** Es la receta directa del gridding: sin variación, las grillas se alinean perfectamente y las posiciones intermedias nunca se consultan.

---

## Ver también

**Papers:** [WaveNet (2016)](/papers/wavenet-oord-2016) · [Very Deep CNN para formas de onda crudas (2017)](/papers/raw-waveforms-dai-2017) · [Conformer (2020)](/papers/conformer-gulati-2020).

**Fundamentos:** [Redes Convolucionales](/fundamentos/redes-convolucionales) · [Redes Recurrentes](/fundamentos/redes-recurrentes) · [Self-attention](/fundamentos/self-attention) · [Representación de audio](/fundamentos/representacion-de-audio) · [Inflado de Convoluciones](/fundamentos/inflado-de-convoluciones).

**Clases:** [Clase 39 - Modelos de DL para audio](/clases/clase-39) · [Profundización, Parte I](/clases/clase-39/profundizacion) · [Práctica: campo receptivo y dilatación](/clases/clase-39/practica/01-campo-receptivo-y-dilatacion).
