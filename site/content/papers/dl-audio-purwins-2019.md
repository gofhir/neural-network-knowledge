---
title: "Deep Learning for Audio Signal Processing (2019)"
math: true
weight: 431
---

{{< paper-card
    title="Deep Learning for Audio Signal Processing"
    authors="Hendrik Purwins, Bo Li, Tuomas Virtanen, Jan Schlüter, Shuo-Yiin Chang, Tara Sainath (Aalborg University Copenhagen, Google, Tampere University, OFAI)"
    year="2019"
    venue="IEEE Journal of Selected Topics in Signal Processing / arXiv:1905.00078"
    pdf="/papers/dl-audio-purwins-2019.pdf" >}}
Es el survey que puso **voz, música y sonido ambiental lado a lado** en un mismo marco, precisamente para exponer sus similitudes, sus diferencias y el potencial de fertilización cruzada entre comunidades que hasta entonces hablaban idiomas distintos sobre los mismos problemas. Su arquitectura es una matriz: primero las columnas transversales —una **taxonomía de tareas en dos ejes** (cuántas etiquetas × de qué tipo), un inventario de **representaciones** que va del MFCC a la onda cruda, un inventario de **modelos** (CNN, convolución dilatada, RNN/LSTM, CRNN, seq2seq, GAN), los datos y la evaluación— y recién después las filas: ASR, *music information retrieval*, sonidos ambientales, localización, separación de fuentes, realce y síntesis. Sus dos veredictos centrales son que el **log-mel espectrograma desplazó a los MFCC** en deep learning (porque la DCT que los define era un parche para el supuesto de covarianza diagonal de los GMM, y con redes neuronales "elimina información y destruye relaciones espaciales") y que entre CNN, RNN y CRNN **no hay preferencia clara**, algo que atribuye con inusual honestidad al conocimiento tácito de cada grupo de investigación. Su diagnóstico de época —"todas las tareas de todos los dominios de audio enfrentan datasets relativamente pequeños" y "no existe un dataset comparable a ImageNet, ni modelos preentrenados sobre él, para el dominio del audio"— acertó el hueco y erró la lista de candidatos para llenarlo: propuso semi-supervisado, activo y *few-shot*, y la historia respondió con auto-supervisión. Con manuscrito recibido en **octubre de 2018**, es una fotografía nítida de un campo justo antes de su cambio de paradigma. Y es, con altísima probabilidad, la **fuente estructural de la [Clase 39](/clases/clase-39)**.
{{< /paper-card >}}

---

## Contexto y alcance

El paper se publicó en el *IEEE Journal of Selected Topics in Signal Processing*, vol. 13, n.º 2, mayo de 2019, pp. 206–219. Pero el dato que más condiciona su lectura no está en la portada sino en la nota al pie: **el manuscrito se recibió el 11 de octubre de 2018**. El corte bibliográfico real es ese, no mayo de 2019. Todo lo que hoy parece una omisión rara —que no mencione SpecAugment, que no discuta el Transformer, que trate la separación de fuentes como un problema de máscaras en tiempo-frecuencia— se explica por esa fecha y no por descuido de los autores.

La lista de autores importa para calibrar el texto. **Hendrik Purwins** firma desde el Departamento de Arquitectura, Diseño y Tecnología de Medios de la Universidad de Aalborg Copenhague (no desde Sonos, una atribución frecuente y errónea: trabajó allí en otro período, pero el paper no lo dice). **Bo Li**, **Shuo-yiin Chang** y **Tara Sainath** están en **Google**. **Tuomas Virtanen** en Tampere University. **Jan Schlüter** en CNRS LIS / Université de Toulon y en el OFAI de Viena. Un asterisco marca contribución igualitaria de Purwins, Li, Virtanen y Schlüter.

{{< concept-alert type="recordar" >}}
**Tara Sainath, coautora de este survey, es la primera autora de la [CLDNN](/papers/cldnn-sainath-2015)** (Sainath, Vinyals, Senior y Sak, ICASSP 2015). Esto explica por qué la descripción de la CLDNN dentro del survey es tan precisa y por qué ocupa el lugar de arquitectura canónica en la sección de voz: es la referencia [93] del paper y la fuente directa del "Ejemplo 1" de la [Clase 39](/clases/clase-39/teoria).
{{< /concept-alert >}}

**Qué lo distingue.** En 2018 ya existían buenas revisiones de deep learning para ASR, para MIR y para detección de eventos acústicos, pero cada una vivía en su propia comunidad con su propio vocabulario para el mismo problema. Purwins et al. se toman el trabajo de construir un vocabulario común antes de bajar a los dominios: un mismo eje de categorización de tareas, un mismo inventario de representaciones, un mismo inventario de modelos. Esa es la contribución organizadora, y es la que envejeció mejor.

El segundo rasgo distintivo es la justificación de por qué el audio merece un survey propio y no es un corolario de visión. La introducción empaqueta tres afirmaciones en un párrafo: (a) el audio nativo es **unidimensional**, una serie temporal de muestras; (b) se lo transforma en representaciones bidimensionales tiempo-frecuencia, pero **los dos ejes no son homogéneos** como el horizontal y el vertical de una imagen —desplazarse en frecuencia significa algo físicamente distinto que desplazarse en tiempo, de modo que la invarianza traslacional del kernel 2D es legítima en un eje y discutible en el otro—; y (c) las imágenes son instantáneas que se analizan enteras o por parches con pocas restricciones de orden, mientras que **el audio debe estudiarse secuencialmente en orden cronológico**. De esas tres se derivan casi todas las decisiones de diseño que el survey cataloga después.

**Estructura material.** 153 referencias, **2 figuras** y —dato que conviene tener presente— **ninguna tabla**. La Figura 1 es la taxonomía de tareas en dos ejes: media página, ninguna fórmula, y sin embargo la contribución conceptual más reutilizable del paper. La Figura 2 ordena todo el zoológico de arquitecturas en un solo eje —*¿cómo hace tu modelo para mirar hacia atrás y hacia adelante en el tiempo?*— con cinco paneles: **A** convolución 1D, **B** convolución 1D dilatada, **C** capa recurrente, **D** capa recurrente bidireccional, **E** atención.

El tercer rasgo, y el más simpático: el survey es **honestamente inconcluso**. Su sección de discusión sobre modelos termina admitiendo que no hay preferencia clara entre CNN, RNN y CRNN, y que la razón probable es sociológica: *"esto puede deberse al conocimiento informal especializado de cada grupo de investigación sobre cómo diseñar y ajustar efectivamente un tipo particular de arquitectura"*. Es una de las frases más francas que se leen en un survey de la época.

## Representaciones de audio

Esta es la sección más valiosa del paper y también donde es notoriamente más preciso que cualquier resumen de clase. Abre con la tesis que hace que todo lo demás tenga sentido: construir una representación de features y diseñar un clasificador para esas features se trataron históricamente como **problemas separados**, con el inconveniente de que las features diseñadas a mano pueden no ser óptimas para el objetivo de clasificación; las redes profundas, en cambio, pueden pensarse como algo que **extrae features conjuntamente con la optimización del objetivo**. La evidencia concreta que cita es de Mohamed et al.: las activaciones de las capas bajas de una DNN funcionan como features **adaptadas al hablante**, mientras que las capas altas hacen discriminación por clase. La red reimplementa por su cuenta la etapa que el pipeline clásico hacía a mano —normalización de hablante tipo VTLN o fMLLR— y la optimiza para la tarea final.

Eso define un continuo, y el survey lo recorre entero, de más "hecho a mano" a menos:

$$\text{MFCC} \;\to\; \text{log-mel} \;\to\; \text{espectro de magnitud completo} \;\to\; \text{filtros aprendidos sobre la onda} \;\to\; \text{onda cruda sin filtro alguno}$$

Con una nota al pie metodológicamente importante: *"si bien la señal de audio a menudo se procesará hacia una secuencia de features, consideramos esa parte de la solución, no de la tarea"*. Elegir log-mel ya es una decisión de modelado, no un prerrequisito neutral. Los [fundamentos de representación de audio](/fundamentos/representacion-de-audio) desarrollan la escalera completa.

### La onda cruda y el problema de la fase

El survey identifica dos problemas concretos de trabajar directamente sobre las muestras.

El primero es el **tamaño del campo receptivo**. Para entradas de onda cruda con tasa de muestreo alta, alcanzar un campo receptivo suficiente puede implicar un número enorme de parámetros y un costo computacional alto. A 16 kHz, un contexto de un segundo son 16 000 muestras: o filtros gigantes o muchísimas capas. De ahí la convolución dilatada.

El segundo es la **invarianza de fase**, y es la observación específica de audio que casi nadie explica bien:

> "Al usar la onda cruda como representación de entrada, para una tarea de análisis, una de las dificultades es que **sonidos perceptual y semánticamente idénticos pueden aparecer con desplazamientos de fase distintos**, de modo que usar una representación invariante a pequeños desplazamientos de fase es crítico."

Traducido a lo operativo: dos grabaciones del mismo fonema, del mismo golpe de tambor, del mismo motor, son *el mismo objeto perceptual* pero **vectores completamente distintos** en el dominio del tiempo, porque el instante en que se abrió la ventana desplaza la fase de cada componente. El espectrograma de magnitud resuelve esto de un plumazo: descarta la fase. Una red sobre onda cruda tiene que **aprender** la invarianza, y el survey cataloga las dos maneras conocidas: capas convolucionales que hacen *pooling en el tiempo* —el max-pooling temporal sobre la salida de un filtro es literalmente un detector de energía invariante al desplazamiento dentro de la ventana, o sea la reconstrucción aprendida de $|X(f)|$— o capas densas con unidades ocultas grandes, potencialmente sobrecompletas, *"capaces de capturar la misma forma de filtro en una variedad de fases"*.

Y el corolario: la onda cruda como entrada se usa frecuentemente en **tareas de síntesis**, por ejemplo con modelos autorregresivos como [WaveNet](/papers/wavenet-oord-2016). En síntesis la fase no es un estorbo: es el producto.

### Espectrograma de magnitud vía STFT

Tres puntos que rara vez llegan a una lámina de clase.

**El compromiso de la ventana.** El tamaño de ventana intercambia resolución temporal (ventanas cortas) contra resolución frecuencial (ventanas largas): el principio de incertidumbre de Gabor. Y la observación fina: *sí* se pueden usar ventanas más cortas en frecuencias altas, tanto para log-mel como para constant-Q, "pero esto produce **espectrogramas borrosos de forma no homogénea, inadecuados para modelos espacialmente locales**". Es decir, la solución que suena obvia rompe el supuesto que hace válido usar una CNN 2D. La alternativa que el survey recomienda es calcular espectros con **distintos largos de ventana**, proyectarlos a las **mismas bandas de frecuencia** y tratarlos como **canales separados**. Multi-resolución como canales, no como resolución variable.

**Las bandas no son comparables entre sí.** "A diferencia de las imágenes, las distribuciones de valores difieren significativamente entre bandas de frecuencia. Para contrarrestarlo, **los espectrogramas pueden estandarizarse por separado en cada banda**." Es el consejo práctico más accionable de la sección y el que más se olvida: no se normaliza un espectrograma como si fuera una imagen, con media y desviación globales; se normaliza **por banda**, porque la energía de la banda de 100 Hz y la de 7 kHz difieren en órdenes de magnitud en casi cualquier señal natural.

**Los armónicos rompen la localidad espacial.** Por la física de la producción del sonido hay correlaciones adicionales entre frecuencias que son múltiplos de una misma fundamental. Una nota de piano en $f_0 = 220$ Hz enciende bins en 220, 440, 660, 880 Hz. Un kernel 2D de $3\times3$ ve a lo más tres bins contiguos: **jamás** puede relacionar el fundamental con su tercer armónico. La solución que el survey cita es agregar una **tercera dimensión que entregue directamente las magnitudes de la serie armónica**, apilando como canales las versiones del espectrograma desplazadas a $f_0, 2f_0, 3f_0, \dots$ para que el armónico $k$ caiga en el mismo píxel del canal $k$. Es el equivalente de la dilatación, pero en el eje de frecuencia y con espaciado multiplicativo.

### Escala mel, log-mel y por qué los MFCC caen en desuso

El survey da la definición operacional de los MFCC en una sola frase densa: son espectros de magnitud **proyectados a un conjunto reducido de bandas de frecuencia**, convertidos a **magnitudes logarítmicas** y aproximadamente **blanqueados y comprimidos con una transformada discreta del coseno (DCT)**. Cuatro pasos. El **log-mel es exactamente eso menos el cuarto**.

El banco mel está "inspirado en el sistema auditivo humano y en hallazgos fisiológicos sobre percepción del habla". El survey no da la fórmula; la conversión estándar es

$$m = 2595 \, \log_{10}\!\left(1 + \frac{f}{700}\right)$$

aproximadamente lineal bajo ~1 kHz y logarítmica arriba. El efecto neto es reducir dimensionalidad concentrando resolución donde el oído la tiene: un espectro STFT de 512 bins se comprime a 40–80 bandas. El logaritmo, por su parte, cumple dos funciones que conviene separar: comprime el rango dinámico enorme del audio a algo que una red pueda digerir, y convierte la **multiplicación** por la función de transferencia de la sala o el micrófono en una **suma** constante por banda, que la normalización por banda puede después restar. El log es lo que hace que la normalización cancele el canal. El detalle completo está en [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel).

Ahora, el punto que más veces se explica mal. La frase exacta del survey:

> "Con modelos de deep learning, se ha demostrado que esto último [la DCT] es **innecesario o indeseable, dado que elimina información y destruye relaciones espaciales**. Omitirlo produce el espectro log-mel, una feature popular en todos los dominios del audio."

Y el cierre de la discusión: "mientras que los MFCC son la representación más común en el procesamiento tradicional de señales de audio, **los espectrogramas log-mel son la feature dominante en deep learning**, seguidos por ondas crudas o espectrogramas complejos".

**El argumento histórico completo**, que el survey comprime en dos líneas, es el siguiente. El pipeline clásico de ASR era GMM-HMM: cada estado de trifono tenía asociada una mezcla de gaussianas sobre el vector de features. Por costo computacional y por cantidad de datos, esas gaussianas se entrenaban con **matriz de covarianza diagonal**; estimar la covarianza plena de un vector de 40 dimensiones son $40 \times 41 / 2 = 820$ parámetros por componente, por cada uno de miles de estados, lo que era inviable. Pero una gaussiana diagonal asume **independencia entre dimensiones**, y las bandas log-mel están fuertísimamente correlacionadas: la energía en la banda 12 predice muy bien la de la banda 13. Alimentar un GMM diagonal con log-mel es violar su supuesto central de la peor manera posible.

Ahí entra la DCT. La DCT sobre las log-energías de banda es una aproximación fija —independiente de los datos— a la transformación de Karhunen-Loève del espectro log-mel: **decorrelaciona aproximadamente** las dimensiones. Además concentra la energía en los primeros coeficientes, lo que permite quedarse con 12 o 13 y descartar el resto.

{{< concept-alert type="clave" >}}
Los MFCC no son "mejores features": son **log-mel torcidos para que quepan en el supuesto de covarianza diagonal de un GMM**. Son un parche de modelado que se disfrazó de feature perceptual durante treinta años. Cuando el clasificador pasa a ser una red neuronal, el parche deja de tener sentido y pasa a hacer daño.
{{< /concept-alert >}}

Las dos razones que el survey nombra merecen desarrollarse por separado:

1. **"Elimina información."** Truncar a 13 coeficientes bota deliberadamente la estructura fina del espectro, y en particular la estructura armónica, que vive en los coeficientes cepstrales altos (la *quefrencia* del pitch). Una DNN sí puede aprovecharla; un GMM diagonal no. La DCT truncada es un cuello de botella diseñado para un modelo que ya no usamos.
2. **"Destruye relaciones espaciales."** Y esta es la razón decisiva para cualquier clase que muestre el espectrograma como imagen. Una CNN funciona porque asume que **vecindad en el índice implica vecindad semántica**: los bins log-mel $k$ y $k+1$ son frecuencias adyacentes, y un kernel local sobre ellos captura una estructura espectral real —un formante, un ataque. Los coeficientes cepstrales $c_k$ y $c_{k+1}$ **no son adyacentes en ningún sentido físico**: son proyecciones sobre dos cosenos de frecuencias distintas, cada uno con soporte sobre *todo* el espectro. Convolucionar sobre el eje cepstral es aplicar un kernel local a un vector cuya estructura es global. **La DCT destruye exactamente la propiedad que hace que valga la pena usar una CNN.**

Un matiz de honestidad: "los MFCC están obsoletos" es una simplificación del veredicto. Lo que el survey dice es que la DCT es innecesaria o indeseable **con modelos de deep learning**, y que en procesamiento tradicional los MFCC siguen siendo lo más común. Siguen vivos donde su decorrelación importa: modelos con supuestos de independencia, clustering —[HuBERT](/papers/hubert-hsu-2021) los usa como target de la primera iteración de k-means—, features de baja dimensión para clasificadores clásicos y sistemas embebidos con presupuesto mínimo.

### Constant-Q: por qué la música necesita otro eje

> "Para algunas tareas es preferible usar una representación que capture **las transposiciones como traslaciones**. Transponer un tono consiste en escalar la frecuencia base y los armónicos por un factor común, lo que se vuelve un **desplazamiento en una escala logarítmica de frecuencia**. El espectro constant-Q logra esa escala con un banco de filtros adecuado."

El razonamiento en detalle: transponer una nota un semitono multiplica $f_0$ y **todos** sus armónicos por $2^{1/12}$. En un eje lineal (STFT) o casi lineal en la parte baja (mel), esa multiplicación es un **estiramiento no uniforme** —el fundamental se mueve 13 Hz y el quinto armónico se mueve 65 Hz—, con lo cual el patrón armónico **cambia de forma**. En un eje logarítmico, en cambio,

$$\log\!\left(2^{1/12} \cdot k f_0\right) = \log(k f_0) + \tfrac{1}{12}\log 2$$

todos los componentes se desplazan **la misma cantidad**: el patrón se **traslada rígidamente**. Y una CNN es equivariante a traslaciones por construcción. Sobre constant-Q, **un solo kernel aprendido reconoce un acorde mayor en las doce tonalidades**; sobre STFT o mel habría que aprender doce plantillas. Eso es un factor 12 de eficiencia estadística en un dominio donde los datasets etiquetados son chicos.

El *Q* del nombre es el factor de calidad $Q = f_k / \Delta f_k$, constante por construcción: cada filtro tiene ancho de banda proporcional a su frecuencia central, o sea ancho constante **en semitonos**. La consecuencia práctica es resolución temporal peor en graves y mejor en agudos, que es exactamente lo que la música pide. El survey confirma el patrón empírico: onset detection sobre constant-Q log-magnitud, reconocimiento de acordes sobre constant-Q lineal, estimación de tonalidad sobre espectrogramas de log-frecuencia. **Todas las tareas de armonía usan eje logarítmico.**

### Onda cruda contra features: el veredicto real

Aquí conviene ser quirúrgico, porque es donde más se distorsiona la posición del paper. El veredicto es **condicional a la tarea**.

Para tareas de **análisis** (ASR, MIR, reconocimiento de sonido ambiente): "los espectrogramas log-mel proveen una representación más compacta, y los métodos que usan estas features **usualmente necesitan menos datos y menos entrenamiento para alcanzar resultados que, en el estado del arte actual, son comparables en desempeño de clasificación** a una configuración donde se usa audio crudo". Léase con cuidado: el survey **no** dice que log-mel gane en precisión; dice que **empatan** y que log-mel llega ahí **con menos datos y menos entrenamiento**. La onda cruda "evita features diseñadas a mano, lo que debería permitir explotar mejor la capacidad de modelado del deep learning… sin embargo, esto incurre en costos computacionales y requisitos de datos mayores, y **los beneficios pueden ser difíciles de materializar en la práctica**".

Para tareas de **síntesis** (separación de fuentes, realce, TTS, transformación de timbre): "usar espectrogramas de magnitud (log-mel) plantea el desafío de reconstruir la fase. En ese caso, **las ondas crudas o los espectrogramas complejos son generalmente preferidos**".

{{< concept-alert type="clave" >}}
El criterio limpio es **el problema de la fase**. Si la salida es una etiqueta, tira la fase y usa log-mel. Si la salida es audio, no puedes tirarla. Citar el survey como "dice que la onda cruda es mejor" o como "dice que log-mel es mejor", sin la condición, es citarlo mal.
{{< /concept-alert >}}

La evidencia empírica concreta que el survey reporta en música es instructiva y merece la tabla:

| Trabajo | Entrada | Diseño de la primera capa | Resultado según el survey |
|---|---|---|---|
| Dieleman y Schrauwen | log-mel de 3 s | convoluciones 1D cortas sobre el tiempo | referencia |
| Dieleman y Schrauwen | muestras crudas | filtro dimensionado **para igualar un frame de espectrograma** | "obtiene peores resultados" |
| Lee et al. | muestras crudas | filtros muy cortos (tamaño **2 a 4**) intercalados con max-pooling | "iguala el desempeño de los espectrogramas log-mel" |

La lectura que el survey deja implícita: **la onda cruda funciona cuando no se intenta imitar el espectrograma**. Dieleman eligió el filtro de primera capa para calzar con un frame típico de STFT y perdió; Lee eligió filtros de 2 a 4 muestras apilados con pooling —una jerarquía genuinamente aprendida, sin prior de STFT— y empató.

El camino intermedio es el que el survey trata como la línea más prometedora: "algunos intentan encontrar un punto medio diseñando o inicializando las primeras capas de un sistema de deep learning para que imiten representaciones diseñadas a mano". Uno de esos trabajos es de la propia Sainath, aprendiendo el *front-end* del habla con CLDNN sobre onda cruda: primeras capas que imitan la computación del log-mel pero con todos los parámetros del filtro aprendidos de los datos. Y en el extremo, WaveNet, donde "la noción de banco de filtros se descarta".

## Los modelos

### MLP

El survey casi no le dedica espacio como familia propia: "para audio, se apilan usualmente múltiples capas feedforward, convolucionales y recurrentes (por ejemplo LSTM) para aumentar la capacidad de modelado". Su rol funcional aparece por dos vías: la evidencia de Mohamed et al. (capas bajas ≈ features adaptadas al hablante, capas altas ≈ discriminación por clase) y la histórica (las DNN feedforward con millones de parámetros sobre miles de horas fueron las que en 2012 redujeron el WER dramáticamente). En las arquitecturas concretas, el MLP aparece siempre **al final**, como cabeza clasificadora.

### CNN, convolución dilatada y WaveNet

La caracterización del survey distingue **tres regímenes**, no uno: "en el caso de features espectrales se adopta comúnmente **una convolución temporal 1D o una convolución 2D tiempo-frecuencia**, mientras que para entradas de onda cruda se aplica **una convolución 1D en el dominio del tiempo**". La convolución 1D temporal sobre espectrograma —convolucionar solo en $t$, tratando las frecuencias como canales— es una opción de primera clase que se olvida con frecuencia.

Sobre el campo receptivo, definido explícitamente como "el número de muestras o espectros involucrados en calcular una predicción", el survey observa que está fijado por la arquitectura y puede aumentarse con kernels más grandes o más capas, con el costo ya mencionado. La alternativa es la **convolución dilatada** (también llamada *atrous*, o convolución con agujeros), que "aplica el filtro convolucional sobre un área mayor que su largo **insertando ceros entre los coeficientes del filtro**. Una pila de convoluciones dilatadas permite a las redes obtener campos receptivos muy grandes con apenas unas pocas capas, **preservando la resolución de entrada además de la eficiencia computacional**".

La aritmética que el survey no escribe: con $L$ capas de kernel $k$ y dilatación duplicándose $1, 2, 4, \dots, 2^{L-1}$, el campo receptivo crece como

$$R = 1 + (k-1)\sum_{l=0}^{L-1} 2^{l} = 1 + (k-1)\left(2^{L}-1\right)$$

es decir **exponencial en la profundidad**, contra el crecimiento **lineal** $R = 1 + L(k-1)$ de la convolución estándar. Con $k=2$ y $L=10$ ya son 1024 muestras.

{{< concept-alert type="advertencia" >}}
El detalle que casi todas las presentaciones pierden es **"preservando la resolución de entrada"**. La dilatación amplía el contexto **sin hacer submuestreo**, y eso es precisamente lo que la vuelve indispensable en tareas de salida densa: sample a sample en síntesis, frame a frame en detección de eventos. El survey lo repite en la sección de sonidos ambientales: para emitir un vector de actividad de eventos con resolución temporal suficiente, el grado de max-pooling o de stride temporal no debe ser grande; si se desea un campo receptivo amplio, se usan **convolución y pooling dilatados**. La dilatación no es un truco de eficiencia: es lo que resuelve el conflicto entre contexto amplio y resolución de salida fina.
{{< /concept-alert >}}

Y una admisión que rara vez se repite: "**no hay disponibles, al momento de escribir, teorías operativas y validadas sobre cómo determinar la arquitectura CNN óptima** (tamaño de kernels, pooling y mapas de features, número de canales y capas consecutivas) para una tarea dada. Actualmente, por lo tanto, la arquitectura de una CNN se elige en gran medida experimentalmente en base a un error de validación, lo que ha llevado a algunas **reglas de dedo**, tales como menos parámetros para menos datos, **aumentar el número de canales a medida que decrece el tamaño de los mapas de features** en capas convolucionales sucesivas, considerar el tamaño necesario de contexto temporal, y el diseño relacionado con la tarea".

**WaveNet** aparece como el caso paradigmático y se cita catorce veces a lo largo del texto. Los rasgos que el survey destaca: descarta la noción de banco de filtros, aprendiendo "un modelo de regresión causal de las muestras de la onda en el dominio del tiempo **sin conocimiento previo humano**"; plantea la predicción autorregresiva del sample **como un problema de clasificación**, con la amplitud cuantizada logarítmicamente en clases discretas —a 8 bits, 256 clases—; admite condicionamiento **global** (identidad del hablante) o **variable en el tiempo** ($f_0$, espectros mel); y "los modelos basados en WaveNet para síntesis de voz superan a los sistemas del estado del arte por un margen amplio, pero **su entrenamiento es computacionalmente costoso**".

### RNN y LSTM

"El tamaño de contexto efectivo que pueden modelar las CNN es limitado, incluso usando convoluciones dilatadas. Las RNN siguen un enfoque distinto: calculan la salida de un paso temporal a partir tanto de la entrada en ese paso como de su estado oculto en el paso previo. Esto modela inherentemente la dependencia temporal en las entradas y **permite que el campo receptivo se extienda indefinidamente hacia el pasado**. Para aplicaciones offline, las RNN bidireccionales emplean una segunda recurrencia en orden inverso, **extendiendo el campo receptivo hacia el futuro**."

El argumento formal de capacidad, poco citado y muy bueno: a diferencia de los HMM convencionales, con crecimiento lineal del número de unidades ocultas recurrentes, **el número de estados representables crece exponencialmente, mientras que el tiempo de entrenamiento o inferencia crece a lo sumo cuadráticamente**. Un HMM con $N$ estados necesita una matriz $N \times N$; una RNN con $H$ unidades puede representar del orden de $2^H$ configuraciones con $O(H^2)$ pesos. Ese es el argumento de por qué la RNN reemplazó al HMM.

Dos variantes específicas de audio que el survey introduce y que casi nadie conoce: la **F-LSTM** (*Frequency LSTM*), que recurre **sobre el eje de frecuencia** en lugar del tiempo —"a diferencia de las CNN, las F-LSTM capturan invarianza traslacional a través de filtros locales y conexiones recurrentes; no requieren operaciones de pooling y son más adaptables a un rango de tipos de features de entrada"—, y la **TF-LSTM** (*Time-Frequency LSTM*), desenrollada sobre ambos ejes. El veredicto sobre esta última es la frase que hay que retener: "**las TF-LSTM superan a las CNN en ciertas tareas, pero son menos paralelizables y por lo tanto más lentas**". En 2018, la alternativa que ganaba en precisión perdía en paralelismo. Es exactamente el trade-off que el Transformer vino a romper.

### CRNN

El survey la formula en dos líneas: "alternativamente, las RNN pueden procesar la salida de una CNN, formando una **red neuronal convolucional recurrente (CRNN)**. En este caso, **las capas convolucionales extraen información local y las capas recurrentes la combinan sobre un contexto temporal más largo**". Es la formulación de la que deriva el fundamento [CRNN](/fundamentos/crnn).

Y la cierra en la discusión con la comparación completa, que vale citar entera porque es el pasaje más importante del paper y el que más se suaviza al resumirlo:

> "A lo largo de los dominios, CNN, RNN y CRNN se emplean exitosamente, **sin preferencia clara**. Las tres pueden modelar secuencias temporales y resolver tareas de clasificación, etiquetado y transducción de secuencias. **Las CNN tienen un campo receptivo fijo, lo que limita el contexto temporal considerado para una predicción, pero al mismo tiempo hace muy fácil ampliar o angostar el contexto usado. Las RNN pueden, teóricamente, basar sus predicciones en un contexto temporal ilimitado, pero primero necesitan aprender a hacerlo**, lo que puede requerir adaptaciones al modelo (como la LSTM) e **impide el control directo sobre el tamaño del contexto**. Además, requieren procesar la entrada secuencialmente, haciéndolas **más lentas de entrenar y evaluar en hardware moderno** que las CNN. **Las CRNN ofrecen un compromiso intermedio, heredando las ventajas y desventajas de ambas.**"

Cuatro matices que suelen perderse al condensar este párrafo: (1) el "sin preferencia clara" es explícito, y en música el survey es aún más enfático —"ni dentro ni entre tareas hay consenso sobre qué representación de entrada usar ni qué arquitectura emplear"—; (2) el campo receptivo fijo de la CNN es también **una ventaja**, porque es controlable por diseño, mientras que la RNN tiene contexto teóricamente infinito pero **no controlable**; (3) la RNN *puede* descubrir relaciones distantes *si logra aprenderlas*, y por eso hizo falta inventar la LSTM; (4) la recurrencia tiene un **costo estructural de paralelismo**, que es exactamente el argumento que explica por qué el campo se movió a Transformers y Conformers en los dos años siguientes.

El ejemplo canónico de CRNN que el survey destaca es McFee y Bello para reconocimiento de acordes: una convolución 2D que aprende features espectrotemporales, seguida de una convolución 1D que integra información a lo largo de las frecuencias, seguida de una GRU bidireccional, con **170 clases de acordes** y *side targets* para incorporar las relaciones entre ellas.

### Sequence-to-sequence

El survey plantea el problema de fondo: por la complejidad de las tareas de audio, los sistemas convencionales dividen la tarea en subtareas y resuelven cada una independientemente. En ASR eso significa componentes separados de modelo **acústico, de pronunciación y de lenguaje**, normalmente entrenados por separado. El argumento end-to-end es que los sistemas seq2seq se entrenan optimizando criterios **relacionados con la métrica final de evaluación**; son completamente neuronales y no usan transductores de estados finitos, léxico ni módulos de normalización de texto; no requieren *bootstrapping* desde árboles de decisión ni alineamientos temporales generados por otro sistema; y simplifican la decodificación.

| Modelo | Mecanismo |
|---|---|
| **CTC** | Introduce un símbolo *blank* para igualar largos e **integra sobre todas las formas de insertar blanks**, optimizando la secuencia de salida en vez de cada etiqueta individual |
| **RNN-T** | Extensión de CTC con un **componente de modelo de lenguaje recurrente separado** |
| **Atención** | Aprende alineamientos entre entrada y salida **conjuntamente** con la optimización del objetivo |
| **LAS** (*Listen, Attend and Spell*) | Encoder ≈ modelo acústico, módulo de atención ≈ modelo de alineamiento, decoder ≈ modelo de lenguaje, todo en una sola red |

El survey señala que "entre varios modelos seq2seq, **LAS ofreció mejoras sobre los otros**".

### GAN y VAE

El survey es notoriamente frío con las GAN en audio: "**a pesar del éxito de las GAN en síntesis de imágenes, su uso en el dominio del audio ha sido limitado**". Las lista en separación de fuentes, transformación de instrumentos musicales y realce de voz. Y reporta el resultado más medido, que además viene de sus propios coautores: SEGAN "produce mejoras en métricas perceptuales de calidad de voz sobre los datos ruidosos y sobre un baseline de realce tradicional", **pero** cuando se usa voz realzada por GAN para ASR "**no se encuentra mejora** comparada con un realce que usa un enfoque de regresión más simple". Mejora la métrica perceptual, no mejora la tarea aguas abajo.

Los **VAE** aparecen en dos lugares: en el diseño de funciones de pérdida —el caso de Piano Genie, donde una pérdida se personalizó para mantener las variables latentes dentro de un rango definido y otra para que los cambios en el espacio de control se reflejaran en el audio generado— y en síntesis por bloques, donde "el sonido se sintetiza a menudo desde una representación latente de baja dimensión, desde la cual necesita ser sobremuestreado hasta el sonido de alta resolución. **Los artefactos inducidos por las distintas resoluciones de capa pueden atenuarse mediante perturbación aleatoria de fase en distintas capas**". Ese artefacto —el *checkerboard* del sobremuestreo, audible como un zumbido tonal fijo— es un problema específico de audio que rara vez se menciona.

### Dos secciones que suelen ignorarse: pérdidas y fase

Sobre **funciones de pérdida**, la observación de fondo que un ingeniero debería tener grabada: "comparar dos señales de audio tomando el **MSE entre las muestras en el dominio del tiempo no es una medida robusta**. Por ejemplo, la pérdida para **dos señales sinusoidales de la misma frecuencia dependería enteramente de la diferencia entre sus fases**". Dos señales perceptualmente idénticas pueden tener MSE máximo. Es el problema de la fase otra vez, ahora del lado de la pérdida. Las alternativas que cataloga: MSE entre log-mel (compara envolventes espectrales), MSE entre espectrogramas log-mel (agrega estructura temporal), **soft-DTW diferenciable** para tolerar deformaciones temporales no lineales, y **distancia de Wasserstein**. Más pérdidas específicas de tarea: en separación de fuentes se puede diseñar una pérdida diferenciable a partir de **experimentos psicoacústicos de inteligibilidad**, maximizando directamente una medida tipo STOI.

Sobre **modelado de fase**, el inventario completo de opciones es el mejor resumen breve del tema que existe: **Griffin-Lim** para estimar la fase desde la magnitud, con la advertencia de que "la precisión de la fase estimada es **insuficiente para producir audio de alta calidad**"; **vocoder neural**, entrenando una red tipo WaveNet para generar la señal temporal desde log-mel; **espectro complejo como entrada**, con magnitud y fase como features; **targets complejos** (*complex ratio masking*); y **extender toda la red al dominio complejo**, con convolución, pooling y activaciones definidas sobre números complejos.

## Las tareas

El survey recorre siete familias. La tabla resume el planteo en su propia nomenclatura, la representación típica, la arquitectura dominante en 2018 y el benchmark de referencia.

{{< concept-alert type="advertencia" >}}
**La columna de benchmark requiere una aclaración.** El survey **no reporta ni un solo resultado numérico por tarea** y nombra apenas siete datasets en total. Los marcados con ✱ sí aparecen en el texto; los demás se agregan aquí como referencia de campo para orientar la búsqueda, no como afirmaciones del paper.
{{< /concept-alert >}}

| Tarea | Planteo (nomenclatura del survey) | Representación típica | Arquitectura dominante en 2018 | Benchmark |
|---|---|---|---|---|
| **ASR** | *sequence transduction* (audio → palabras) | log-mel; también onda cruda con front-end aprendido | CLDNN → seq2seq (CTC, RNN-T, LAS) | LDC ✱; LibriSpeech, Switchboard |
| **Identificación de hablante / idioma** | *sequence classification* | onda cruda (SincNet), log-mel | CNN / DNN | VoxCeleb |
| **Music tagging** | *multi-label sequence classification* global | log-mel de 3 s o 29 s; muestras crudas | CNN 1D, FCN, CNN sample-level | Million Song Dataset ✱; MagnaTagATune |
| **Transcripción de notas** | *sequence labeling* / *transduction* | constant-Q, log-frecuencia | CNN, CRNN | MusicNet ✱ |
| **Reconocimiento de acordes** | *sequence labeling* multiclase | constant-Q, magnitud lineal | CNN; CNN + chroma; CRNN de 170 clases | Isophonics Beatles ✱ |
| **Onset / beat / downbeat** | *event detection* | constant-Q log-magnitud; espectrograma | MLP → BLSTM → CNN; CNN+HMM; RNN+DBN; CRNN | Ballroom, GTZAN-rhythm |
| **Tempo** | *sequence regression* (o clasificación discretizada) | espectrograma, excerpts de 12 s | CNN directa | — |
| **Clasificación de escena acústica** | *sequence classification* multinomial | log-mel | CNN | DCASE ✱ |
| **Detección de eventos acústicos** | *sequence labeling* polifónico | log-mel | RNN; CNN con convolución y pooling dilatados | DCASE ✱ |
| **Audio tagging** | *multi-label sequence classification* global, sin timing | log-mel | CNN | AudioSet ✱ (>2M snippets) |
| **Localización / DOA** | *multi-label classification* sobre grilla de direcciones, o *regression* | espectro de fase, magnitud, GCC entre canales | CNN con kernels que abarcan canales; CRNN | — |
| **Separación de fuentes** | *regression per time step* (máscara T-F) | STFT complejo o de magnitud | CNN, RNN, *deep clustering*, *deep attractor network* | wsj0-2mix |
| **Realce de voz** | *regression per time step* | STFT | denoising autoencoder, CNN, RNN, SEGAN | — |
| **Síntesis / TTS** | *sequence transduction* autorregresiva | onda cruda; log-mel como condicionamiento | WaveNet, SampleRNN, WaveRNN, Parallel WaveNet | — |

### Reconocimiento de voz

El arco histórico que traza: el GMM-HMM de estados de trifono dominó "durante décadas", con virtudes reales —"elegancia matemática, que conduce a muchas soluciones principiadas para problemas prácticos como la adaptación a hablante o a tarea"—; hacia 1990 el entrenamiento discriminativo superó a máxima verosimilitud; se propusieron híbridos con redes (la TDNN de Waibel, las RNN de Robinson); y **en 2012 las DNN con millones de parámetros sobre miles de horas bajaron el WER dramáticamente**. Después: "además del gran éxito de las redes profundas feedforward y convolucionales, **se ha mostrado que las LSTM y GRU superan a las DNN feedforward**. Más tarde, una cascada de capas convolucionales, LSTM y feedforward, es decir el modelo **CLDNN**, mostró además **superar a los modelos con solo LSTM**".

La descripción de la CLDNN dentro del survey es literalmente el "Ejemplo 1" de la Clase 39:

> "En las CLDNN, una ventana de frames de entrada es **primero procesada por dos capas convolucionales con capas de max-pooling para reducir la varianza en frecuencia de la señal**, luego **proyectada hacia un espacio de features de menor dimensión** para que las siguientes capas LSTM modelen las correlaciones temporales, y finalmente **pasada por unas pocas capas feedforward y una capa softmax de salida**."

Los tres roles están nombrados con precisión funcional, y conviene notar el matiz: la CNN **reduce varianza en frecuencia** —no "aprende features locales" en abstracto, sino que normaliza la variabilidad de altura tonal entre hablantes—; la proyección lineal reduce dimensionalidad antes de la LSTM; la LSTM modela correlaciones temporales; el MLP discrimina. Los números concretos están en el [paper original de la CLDNN](/papers/cldnn-sainath-2015).

El giro hacia seq2seq lo enuncia así: "con la adopción de RNN para modelado de voz, **el supuesto de independencia condicional de los targets de salida que impone el modelado tradicional de estados de fonema basado en HMM deja de ser necesario**, y el campo de investigación se desplazó hacia modelos completos de secuencia a secuencia".

### Music information retrieval

El survey abre con la diferencia estructural respecto de la voz: "comparadas con el habla, **las grabaciones musicales típicamente contienen una variedad más amplia de fuentes sonoras de interés**. En muchos tipos de música, su ocurrencia sigue restricciones comunes en tiempo y frecuencia, **creando dependencias complejas dentro y entre fuentes**".

Eso es exactamente lo que hace la música más difícil que la voz en separación: "en el habla se asume que la señal es dispersa y que las distintas fuentes son independientes entre sí. En sonidos ambientales, la independencia usualmente puede asumirse. **En música hay una alta dependencia entre fuentes simultáneas**, así como dependencias temporales específicas a lo largo del tiempo, en la onda y en las repeticiones estructurales de largo plazo". Los instrumentos de un acorde no son fuentes independientes: comparten fundamental, se solapan en armónicos y están sincronizados rítmicamente. Todo lo que la separación ciega asume, la música lo viola.

Y el hallazgo transversal más fácil de pasar por alto, en el párrafo sobre detección de eventos: "comparando enfoques, tanto CNN con contexto temporal de tamaño fijo como RNN con contexto potencialmente ilimitado se usan exitosamente para detección de eventos. **Curiosamente, para las primeras parece crítico difuminar los targets de entrenamiento en el tiempo**". Tres trabajos independientes —onsets, onsets otra vez, y fronteras estructurales— encontraron lo mismo. La razón: la anotación humana de un onset tiene jitter de decenas de milisegundos, y un target one-hot exacto castiga con pérdida máxima una predicción que está a un frame de distancia y es esencialmente correcta. Difuminar el target convierte un problema mal condicionado en uno aprendible. Es *label smoothing* en el eje temporal, y es un truco práctico de primera línea.

### Sonidos ambientales

Es la sección más relevante para la Clase 39, que se centra sobre todo en sonidos generales. El survey divide el campo en exactamente tres, y son las tres categorías del fundamento [clasificación de audio](/fundamentos/clasificacion-de-audio):

- **Clasificación de escena acústica.** "Apunta a etiquetar una grabación completa con una única etiqueta de escena", con etiquetas definidas de antemano como "casa", "calle", "en el auto", "restaurante". Es un **problema de clasificación multinomial**: una etiqueta global y exclusiva.
- **Detección de eventos acústicos.** "Apunta a estimar los **tiempos de inicio y fin** de eventos sonoros individuales tales como pasos, señalización acústica de semáforos o ladridos de perro, y asignarles una etiqueta." La implementación práctica que recomienda: "una forma simple y eficiente de aplicar aprendizaje supervisado para hacer detección es **predecir la actividad de cada clase de evento en segmentos temporales cortos**".
- **Tagging.** "Apunta a predecir la actividad de múltiples clases sonoras (posiblemente simultáneas), **sin información temporal**."

Sobre el contexto, la observación práctica: el clasificador usualmente usará información contextual, es decir features acústicas calculadas **fuera del segmento a clasificar**. La forma simple es **concatenar features de varios frames de contexto** alrededor del frame objetivo, como hacía el método baseline de DCASE 2016; la alternativa es usar arquitecturas que modelen información temporal, mapeando una secuencia de features frame a frame a una secuencia de vectores binarios de actividad.

Sobre polifonía, el hallazgo empírico: "se ha encontrado que **usar un clasificador multi-etiqueta para predecir conjuntamente la actividad de múltiples clases a la vez produce mejores resultados** que usar clasificadores de una sola clase por separado. Esto podría deberse, por ejemplo, a que el clasificador multiclase es capaz de **modelar la interacción de clases simultáneamente activas**". La consecuencia de implementación, que el survey no explicita pero se deduce: la salida es **sigmoide por clase con entropía cruzada binaria**, no softmax. Un softmax fuerza $\sum_c p_c = 1$, es decir competencia entre clases, que es lo contrario de lo que se quiere en polifonía.

Y el cierre, que justifica directamente el bloque de aumentación de datos: "dado que el análisis de sonidos ambientales es un **campo de investigación menos establecido** en comparación con voz y música, el tamaño y la diversidad de los datasets disponibles es más limitado… **Debido al tamaño limitado de los datasets ambientales anotados, la aumentación de datos es una técnica comúnmente usada en el campo, y se ha encontrado altamente efectiva.**"

### Separación de fuentes

La formalización. La mezcla en el micrófono $m$ es

$$x_m(n) = \sum_{i=1}^{I} s_{m,i}(n)$$

donde $s_{m,i}(n)$ es la **imagen espacial** de la fuente $i$ en el micrófono $m$ —no la fuente "seca": ya incluye la respuesta de la sala hasta ese micrófono. El enmascaramiento en el dominio tiempo-frecuencia es

$$\hat{S}_{m,i}(f,t) = M_{m,i}(f,t)\, X_m(f,t)$$

Las **tres razones** por las que se trabaja en tiempo-frecuencia son el mejor párrafo del survey sobre por qué el espectrograma no es solo una conveniencia:

1. "La estructura de las fuentes sonoras naturales **es más prominente en el dominio tiempo-frecuencia**, lo que permite modelarlas más fácilmente que las señales en el dominio del tiempo."
2. "**El mezclado convolutivo**, que involucra una función de transferencia acústica desde la fuente al micrófono, **puede aproximarse como mezclado instantáneo en el dominio de la frecuencia**, simplificando el procesamiento."
3. "Las fuentes sonoras naturales son **dispersas en el dominio tiempo-frecuencia**, lo que facilita su separación en ese dominio."

La razón (2) merece énfasis: la propagación en una sala es una **convolución** con la respuesta impulsiva; en frecuencia, y suponiendo ventanas más largas que esa respuesta, se convierte en una **multiplicación escalar por bin**. Un problema de deconvolución se vuelve uno de escalado. La razón (3) es la que hace que enmascarar funcione: si dos fuentes rara vez ocupan el mismo bin con energía comparable, una máscara binaria puede separarlas casi perfectamente.

Y la elección de la STFT sobre alternativas invierte el argumento clásico: se usa "porque puede implementarse eficientemente con la FFT, y también porque **la STFT puede invertirse fácilmente**. El uso de otras representaciones tiempo-frecuencia también es posible, como constant-Q o mel. **Su uso se ha vuelto sin embargo menos común, dado que reducen la calidad de salida, y el deep learning no requiere una representación de entrada compacta** como la que ellas proveerían en comparación con la STFT". Mel existía para comprimir; con deep learning la compresión dejó de ser necesaria y su costo pasó a dominar.

Los métodos que destaca: **deep clustering**, que "usa aprendizaje profundo supervisado para estimar vectores de embedding para cada punto tiempo-frecuencia, que luego se agrupan de manera no supervisada. **Este enfoque permite separar fuentes que no estaban presentes en el conjunto de entrenamiento**" —resuelve el problema de permutación y generaliza a hablantes no vistos—, y su extensión, la **deep attractor network**, basada en estimar un único vector atractor por fuente.

En **realce de voz**, la ventaja estructural en una línea: "los enfoques convencionales de denoising, como los métodos de Wiener, usualmente **asumen ruido estacionario, mientras que los enfoques de deep learning pueden modelar ruido variable en el tiempo**".

## Etiquetas fuertes y débiles

{{< concept-alert type="advertencia" >}}
**Precisión de atribución, primero.** Verificado sobre el texto completo del PDF: el survey **no usa la terminología "weak/strong labeling" de forma sistemática**. La expresión "weakly-labelled" aparece **dos veces**: una describiendo "el dataset AudioSet débilmente etiquetado", sin definir el término, y otra dentro del título de una referencia de Schlüter. La expresión "strong labels" **no aparece nunca**. Y **"multiple instance learning" no aparece ni una sola vez**. La sustancia de la taxonomía sí está —en la Figura 1 y en la sección sobre sonidos ambientales—, pero atribuirle al survey un tratamiento del *multiple instance learning* sería atribuirle algo que no dice. Lo que sigue separa explícitamente lo que el paper afirma de lo que este análisis agrega.
{{< /concept-alert >}}

### Lo que el survey sí dice: los dos ejes de la Figura 1

La figura cruza dos ejes independientes. El primero, **cuántas etiquetas hay que predecir**:

| Caso | Nombre en el survey | Ejemplos del survey |
|---|---|---|
| Una etiqueta global para toda la secuencia | *sequence classification* | idioma, hablante, tonalidad musical, escena acústica |
| Una etiqueta por paso de tiempo | *sequence labeling* / *event detection* | anotación de acordes, detección de actividad vocal, cambios de hablante, onsets de nota |
| Una secuencia de etiquetas de largo libre | *sequence transduction* | speech-to-text, transcripción musical, traducción |

El segundo, **de qué tipo es cada etiqueta**:

| Caso | Nombre en el survey | Ejemplos |
|---|---|---|
| Una sola clase | clasificación | escena acústica, tonalidad |
| Un conjunto de clases | *multi-label* | eventos acústicos simultáneos (AudioSet), conjunto de alturas musicales |
| Un valor numérico | *regression* | tempo, distancia a una fuente en movimiento, pitch, el siguiente sample de audio |

Con dos observaciones del propio texto: "los problemas de regresión siempre pueden discretizarse y convertirse en problemas de clasificación" —el caso de la cuantización a 8 bits, que vuelve la predicción del siguiente sample una clasificación de 256 clases— y que la clasificación multi-etiqueta "puede ser particularmente eficiente **cuando las clases dependen unas de otras**".

### El producto cartesiano, y dónde encaja la terminología del campo

Cruzar ambos ejes produce exactamente las cuatro casillas que la Clase 39 presenta como "single/multiple × global/local":

| Casilla | Fig. 1 del survey (eje 1 × eje 2) | Terminología estándar del campo |
|---|---|---|
| Etiqueta única **global** | global × clase única = *sequence classification* | **acoustic scene classification**, clasificación a nivel de clip |
| Etiqueta única **local** | por paso × clase única = *sequence labeling* | clasificación a nivel de frame, segmentación |
| Múltiples etiquetas **globales** | global × conjunto de clases = *multi-label sequence classification* | **audio tagging** con **etiquetas débiles** |
| Múltiples etiquetas **locales** | por paso × conjunto de clases = *sequence labeling* multi-etiqueta | **sound event detection** polifónica con **etiquetas fuertes** |

La distinción sustantiva, que el survey deja implícita:

- **Etiquetas fuertes**: la anotación incluye **tiempos de inicio y fin** de cada evento. Permite entrenar un detector directamente, porque cada frame tiene su vector binario de target. Son **caras**: anotar onsets y offsets en un clip de diez segundos toma minutos de trabajo humano experto.
- **Etiquetas débiles**: la anotación dice **qué clases están presentes en el clip**, sin decir cuándo. Es lo que AudioSet ofrece para más de dos millones de fragmentos de diez segundos: barato de anotar —un humano marca casillas— e imposible de usar directamente para entrenar un detector por frame.

{{< concept-alert type="recordar" >}}
**Aporte de este análisis, no del survey.** El puente entre ambos regímenes es el *multiple instance learning*, que el paper **no menciona**. La formulación estándar trata cada clip como una **bolsa** de instancias (frames), con etiqueta de bolsa positiva si **al menos una** instancia lo es. Se entrena una red que produce predicciones por frame $p_c(t)$ y se agregan a nivel de clip con un *pooling* diferenciable —max, promedio o *attention pooling*— para comparar contra la etiqueta débil. El resultado es un detector con resolución temporal entrenado **solo** con etiquetas de clip. En la literatura posterior a 2019 esto es el estándar (PANNs, [AST](/papers/ast-gong-2021), las tareas de DCASE que combinan datos débilmente etiquetados con datos sintéticos fuertemente etiquetados). El survey tiene un solo punto de contacto: cita el trabajo de Schlüter sobre *localizar* la voz cantada a partir de ejemplos débilmente etiquetados —que es MIL puro—, pero lo cita como ejemplo de **interpretabilidad**, no de aprendizaje débil. Es una oportunidad perdida y vale registrarla.
{{< /concept-alert >}}

La consecuencia práctica es de búsqueda bibliográfica: los términos "single vs multiple labels, global vs local" no devuelven nada. Los que sirven son `audio tagging` + `weakly labeled`, `sound event detection` + `strongly labeled`, `acoustic scene classification`, `multiple instance learning` + `audio`, y `DCASE` para el challenge que define tareas y métricas oficiales.

## Data augmentation

El survey define el tema en una línea que la clase reproduce casi textual: la aumentación de datos "**genera datos de entrenamiento adicionales manipulando ejemplos existentes** para cubrir un rango más amplio de entradas posibles". Su catálogo completo, con el dominio donde lo reporta y la invarianza que impone:

| Técnica | Dominio donde el survey la reporta | Qué invarianza impone |
|---|---|---|
| **Pitch shifting** (en ASR: *vocal tract length perturbation*) | ASR; reconocimiento de acordes; detección de voz cantada; reconocimiento de instrumento | Invarianza al largo del tracto vocal / a la tonalidad |
| **Time stretching** | ASR; detección de voz cantada; reconocimiento de instrumento | Invarianza a la velocidad de habla / al tempo |
| **Simulación de sala (reverberación) y multicanal ruidoso** | ASR de campo lejano (generación de utterances para Google Home) | Invarianza a la acústica de la sala y a la posición del micrófono |
| **Filtrado espectral** | detección de voz cantada; reconocimiento de instrumento | Invarianza a la ecualización y a la respuesta del micrófono |
| **Combinación lineal de ejemplos con sus etiquetas** (*between-class learning*) | sonidos ambientales | Regularización del espacio entre clases; "mejora la generalización" |
| **Mezcla de pistas separadas para sintetizar mezclas** | separación de fuentes | Genera pares (mezcla, fuentes) exactos y gratis |
| **Generación sintética completa con parámetros conocidos** | general | "Un aumento gradual y controlado de la complejidad de los datos generados facilita entender, depurar y mejorar los métodos" |

Con la advertencia central: "**el desempeño de un algoritmo sobre datos reales puede ser pobre si se entrena solo con datos generados**". Y con el refuerzo de dominio ya citado: en sonidos ambientales la aumentación "se ha encontrado altamente efectiva".

Tres ausencias verificadas que conviene tener presentes. El survey **no lista "agregar ruido"** como técnica de aumentación —lo más cercano es la simulación de sala, que produce voz "ruidosa y reverberante" multicanal—; **no menciona SpecAugment**, y no podía, porque es de abril de 2019 y el manuscrito se recibió en octubre de 2018; y **no menciona Scaper**, la librería de síntesis de paisajes sonoros que implementa justamente la "generación sintética con parámetros conocidos" que sí propone.

### El criterio que falta: cuándo la transformación destruye la etiqueta

Este es el punto que ni el survey ni la clase explicitan, y es el más importante en la práctica. La regla general: **una aumentación es válida si y solo si la transformación preserva la etiqueta**, es decir, si la invarianza que impone es una invarianza real de la tarea. El desarrollo completo está en [data augmentation de audio](/fundamentos/data-augmentation-de-audio); el resumen:

| Transformación | Válida en | Destruye la etiqueta en |
|---|---|---|
| **Pitch shifting** | ASR (el contenido léxico no depende del pitch); detección de voz cantada; instrumento, con cuidado | **Identificación de hablante** (el pitch *es* parte de la identidad); **estimación de $f_0$ y transcripción de notas** (mueve el target); **detección de tonalidad musical**; instrumento si el desplazamiento excede su rango real |
| **Time stretching** | ASR; clasificación de escena; tagging | **Estimación de tempo** y **beat tracking** (mueve el target directamente); cualquier tarea con target temporal absoluto sin reescalar las anotaciones |
| **Adición de ruido** | casi todo lo de reconocimiento; robustez | **Realce y separación** si el ruido se agrega también a la referencia limpia; **estimación de SNR** |
| **Reverberación simulada** | ASR de campo lejano; reconocimiento robusto | **Estimación de distancia a la fuente**; **localización** si la simulación no actualiza la geometría |
| **Filtrado espectral / EQ aleatoria** | tagging, escena, voz cantada | **Identificación de instrumento** con filtros agresivos (el timbre *es* la envolvente espectral); clasificación de vocales y análisis de formantes |
| **Mixup / between-class** | clasificación multi-etiqueta y mono-etiqueta con targets blandos | **Detección de eventos con targets temporales exactos** (superponer dos clips desalinea los onsets) |
| **Enmascaramiento en tiempo o frecuencia** | ASR, tagging, escena: regularización general | **Separación de fuentes** (la máscara borra información que el target sí contiene) y tareas de reconstrucción |
| **Inversión temporal** | prácticamente nada en audio | **Todo**: el habla invertida es ininteligible y un ataque de piano invertido suena a órgano |

{{< concept-alert type="clave" >}}
La última fila es la diferencia más limpia entre audio e imagen. En visión, el *flip* horizontal es el augmentation por defecto, gratis y universal. En audio **no hay equivalente**: el eje temporal tiene flecha —causalidad, ataque y decaimiento— y el eje de frecuencia tiene semántica absoluta. Por eso todas las técnicas del catálogo son transformaciones **físicas** (pitch, tiempo, sala, canal), no geométricas.
{{< /concept-alert >}}

## Datos y evaluación

### Los datasets, y su escasez

La lista completa que el survey nombra es **corta**, y esa cortedad es en sí misma un dato:

| Dataset | Dominio | Cómo lo describe el survey |
|---|---|---|
| **ImageNet** | contraste con visión | "14 millones (2019) de imágenes etiquetadas a mano", "un factor mayor" en el salto del deep learning en visión |
| **LDC** | voz | "Para reconocimiento de voz hay **datasets grandes**, en particular para el inglés" |
| **Million Song Dataset** | música | "para clasificación de secuencias musicales o similitud musical" |
| **MusicNet** | música | "aborda el **etiquetado de secuencias nota a nota**" |
| **Isophonics / anotaciones de referencia de los Beatles** | música | "Los datasets para etiquetado musical de más alto nivel, como acordes, beats o análisis estructural, son a menudo **mucho más pequeños**" |
| **AudioSet** | sonido ambiental | "más de **2 millones de fragmentos de audio**", descrito como débilmente etiquetado |
| **DCASE** | sonido ambiental | "La mayoría de los datos abiertos se ha publicado en el contexto de los challenges anuales DCASE" |

Y el diagnóstico transversal: "**no existe un dataset bien etiquetado que pueda compartirse entre dominios**, incluyendo voz, música y sonidos ambientales". Reforzado por la que probablemente es la frase más importante del paper:

> "Con la posible excepción del reconocimiento de voz, en la industria, para los idiomas más difundidos, **todas las tareas en todos los dominios del audio enfrentan datasets relativamente pequeños**, poniendo un límite al tamaño y complejidad de los modelos de deep learning entrenados sobre ellos."

Nótese lo cuidadosamente calificado que está: la única excepción es ASR, **en industria**, **para los idiomas más difundidos**. Todo lo demás —música, sonido ambiental, ASR académico, ASR de idiomas de bajos recursos— vive en escasez.

### Las métricas, qué miden y cuándo engañan

**WER (Word Error Rate)**, para ASR. El survey lo define como la fracción de errores de palabra tras alinear referencia e hipótesis, compuesta por tasas de inserción, borrado y sustitución divididas por el número de palabras de referencia:

$$\text{WER} = \frac{S + D + I}{N}$$

*Cuándo engaña:* no está acotada por 1 —como $I$ no está limitado por $N$, un modelo que alucina puede superar el 100%—; trata todos los errores como iguales, de modo que confundir "no" por "know" cuenta lo mismo que perder una negación que invierte el sentido clínico de una frase; es sensible a la normalización de texto, y dos sistemas pueden diferir dos puntos solo por cómo escriben "veinte" contra "20"; y depende del idioma, por lo que para lenguas aglutinantes conviene el CER.

**Accuracy.** "Tanto en música como en clasificación de escena acústica, la accuracy es una métrica comúnmente usada." *Cuándo engaña:* con clases desbalanceadas. En detección de eventos raros, un modelo que siempre predice "ausente" puede alcanzar 99% y ser inútil. Por eso en escena acústica —clases balanceadas por diseño de DCASE— es razonable, y en detección de eventos no lo es.

**AUROC.** "Para evaluar clasificación binaria **sin un umbral fijo**, el área bajo la curva ROC es una alternativa a la accuracy." Mide la probabilidad de que un positivo aleatorio reciba mayor score que un negativo aleatorio. *Cuándo engaña:* es **optimista bajo desbalance severo**. Con un positivo por cada diez mil negativos, un AUROC de 0.99 puede corresponder a una precisión inutilizable, porque el eje de falsos positivos se normaliza por el enorme número de negativos. En ese régimen la métrica correcta es el average precision.

**F-score y EER** para detección de eventos. "En detección de eventos, el desempeño se mide típicamente usando **tasa de error igual o F-score**, donde los verdaderos positivos, falsos positivos y falsos negativos se calculan **ya sea en segmentos de largo fijo o por evento**." Esa cláusula final hay que desempaquetarla, porque es la fuente número uno de comparaciones inválidas en SED:

- **Por segmento**: se discretiza el tiempo en segmentos fijos —típicamente 1 s o 10 ms— y se compara la actividad de cada clase en cada segmento. Es tolerante a errores de límites.
- **Por evento**: se comparan eventos completos con una tolerancia de colisión en el onset, típicamente 200 ms, y opcionalmente en el offset. Es mucho más estricto: un evento detectado partido en dos cuenta como un acierto **más** un falso positivo.

Los mismos sistemas pueden reordenarse completamente entre ambos criterios. La referencia que el survey cita para esto (Mesaros, Heittola y Virtanen, con Virtanen como coautor del survey) define además el **error rate** de SED en analogía al WER, con $N$ el número de eventos de referencia; conviene saber que la métrica oficial de varias tareas DCASE ha sido ER y no F-score. *Cuándo engañan:* el F-score depende del umbral; el EER solo tiene sentido si el operativo real es simétrico en costos, y rara vez lo es —en vigilancia acústica un falso negativo cuesta mucho más que un falso positivo—; y ambos, en polifonía, se pueden promediar por clase (macro) o por instancia (micro), con resultados muy distintos bajo desbalance.

**SDR / SIR / SAR**, para separación de fuentes. El framework `BSS_EVAL` descompone la señal estimada en cuatro componentes ortogonales por proyección:

$$\hat{s} = s_{\text{target}} + e_{\text{interf}} + e_{\text{noise}} + e_{\text{artif}}$$

y define razones de energía en decibeles:

$$\text{SDR} = 10\log_{10}\frac{\lVert s_{\text{target}}\rVert^2}{\lVert e_{\text{interf}} + e_{\text{noise}} + e_{\text{artif}}\rVert^2}, \qquad
\text{SIR} = 10\log_{10}\frac{\lVert s_{\text{target}}\rVert^2}{\lVert e_{\text{interf}}\rVert^2}, \qquad
\text{SAR} = 10\log_{10}\frac{\lVert s_{\text{target}} + e_{\text{interf}} + e_{\text{noise}}\rVert^2}{\lVert e_{\text{artif}}\rVert^2}$$

El **SIR** mide cuánta *otra fuente* quedó filtrada; el **SAR**, cuánto artefacto introdujo el algoritmo (distorsión, *musical noise*, huecos espectrales); el **SDR** es la métrica global que combina ambos. *Cuándo engañan:* hay un **trade-off directo entre SIR y SAR** —una máscara binaria agresiva sube el SIR y hunde el SAR—, de modo que reportar solo SDR oculta de qué lado del trade-off está el sistema; el framework permite un filtro de distorsión al calcular la proyección, lo que lo vuelve invariante a filtrados que el oído sí escucha; **es degenerado cuando la fuente está en silencio**, porque $\lVert s_{\text{target}}\rVert^2 \to 0$ manda el SDR a $-\infty$; por eso la comunidad migró al **SI-SDR**; y lo que suele reportarse es el **SDRi**, la mejora respecto de la mezcla sin procesar, no el SDR absoluto, así que confundir ambos hace comparaciones sin sentido.

**MOS (Mean Opinion Score).** "Un test **subjetivo** para evaluar la calidad del audio sintetizado, en particular de voz." *Cuándo engaña:* **no es comparable entre estudios**. Depende del pool de evaluadores, del set de referencia incluido en el test, de las instrucciones y del hardware de reproducción. Un "MOS 4.2" de un paper y un "MOS 4.1" de otro no se pueden comparar; solo son válidas las comparaciones dentro del mismo test, con intervalos de confianza. Y hay efectos de anclaje: incluir grabaciones reales comprime los scores de todo lo demás.

**Test de Turing.** "Pedirle a un humano que distinga entre ejemplos de audio reales y sintetizados es un test duro para un modelo, dado que pasarlo requiere que **no haya diferencia perceptible**."

Una nota de atribución: **el survey no menciona la mAP** en ninguna parte. Se volvió la métrica estándar de AudioSet a partir de los trabajos posteriores de Google, no de este paper.

## Los desafíos abiertos de 2019, y cuáles se resolvieron

La sección de discusión del survey enumera cinco frentes.

**Features.** Estado: log-mel domina, con onda cruda y espectro complejo siguiéndolo. Y cuatro preguntas textuales: ¿son los espectrogramas mel realmente la mejor representación para análisis de audio? ¿Bajo qué circunstancias es mejor usar la onda cruda? ¿Podemos hacerlo mejor explorando el punto medio, **un espectrograma con hiperparámetros aprendibles**? Si aprendemos una representación desde la onda cruda, ¿generaliza aún entre tareas o dominios?

**Modelos.** Nota histórica: en ASR, MIR y sonido ambiental los modelos profundos reemplazaron a las **máquinas de vectores de soporte** para clasificación de secuencias y a los **GMM-HMM** para transducción; en realce y separación, resolvieron tareas antes abordadas con **factorización de matrices no negativas** y **métodos de Wiener**; en síntesis, la **síntesis concatenativa** fue reemplazada por WaveNet, SampleRNN y WaveRNN. Y el desafío: "**es una pregunta de investigación abierta qué modelo es superior en qué contexto**. A partir de la literatura existente esto es muy difícil de responder, dado que distintos grupos de investigación obtienen resultados de estado del arte con distintos modelos".

**Requisitos de datos.** Escasez generalizada, ausencia de un "ImageNet del audio" y de modelos preentrenados sobre él, y tres preguntas: ¿cuál sería una tarea equivalente para el dominio del audio? ¿Puede haber un dataset de audio que cubra voz, música y sonidos ambientales, usado para transferencia, que resuelva un amplio rango de problemas de clasificación? ¿Cómo pueden los modelos preentrenados adaptarse flexiblemente a nuevas tareas con una cantidad mínima de datos? Con una reserva escéptica —"es muy posible que esto deba responderse **por separado para cada dominio**"— y un plan B: "**si la transferencia resulta ser la dirección equivocada para el audio**, la investigación necesita explorar otros paradigmas para aprender modelos complejos desde datos etiquetados escasos, tales como **aprendizaje semi-supervisado, aprendizaje activo o few-shot learning**".

**Complejidad computacional.** Las redes del estado del arte requieren más poder de cómputo y más datos; las CPU son inadecuadas y hacen falta GPU y TPU. "Las aplicaciones con límites estrictos de recursos computacionales, como **teléfonos móviles o audífonos**, requieren modelos más pequeños… puede valer la pena explorar opciones para los requisitos específicos del procesamiento de audio en tiempo real."

**Interpretabilidad y adaptabilidad.** "La conexión entre los parámetros de las capas y la tarea real es **difícil de interpretar**." Dos líneas de ataque: relacionar las activaciones de neuronas con la tarea, e investigar en qué parte de la entrada se basa la predicción. Con el objetivo de "mejorar la estructura del modelo para **atender casos de falla**".

### Qué pasó entre 2019 y hoy

| Desafío del survey | Estado hoy | Cómo se resolvió |
|---|---|---|
| Aprender con pocas etiquetas | **Resuelto**, y no por los caminos que el survey listó | **Aprendizaje auto-supervisado**, no semi-supervisado / activo / few-shot |
| "No existe un ImageNet del audio" ni modelos preentrenados | **Resuelto** | AudioSet + preentrenados públicos: VGGish → PANNs → AST → BEATs; wav2vec 2.0 / HuBERT / WavLM para voz |
| Generación de alta calidad y en tiempo real | **Resuelto** | Vocoders GAN, modelos de difusión, codecs neuronales y modelos de lenguaje sobre tokens de codec |
| Modelar contexto largo sin pagar el costo secuencial de la RNN | **Resuelto** | Transformer y Conformer |
| El problema de la fase en síntesis | **Resuelto en la práctica** | Vocoders neuronales; codecs que operan en el dominio del tiempo |
| "Qué modelo es superior en qué contexto" | **Resuelto para ASR, parcial en el resto** | Conformer como respuesta consensuada; en MIR y sonido ambiental sigue habiendo pluralidad |
| La representación óptima y el "punto medio con hiperparámetros aprendibles" | **Parcial** | SincNet y LEAF existen pero no desplazaron al log-mel, que sigue siendo el default de facto |
| Cómputo y tiempo real en dispositivos restringidos | **Parcial** | Streaming, destilación, cuantización, ASR on-device; pero los modelos grandes crecieron más rápido que las optimizaciones |
| Interpretabilidad | **Abierto** | Sin solución satisfactoria; el campo se movió a evaluación de comportamiento y sondas |

**Aprendizaje con pocas etiquetas.** El survey listó semi-supervisado, activo y few-shot. Lo que ocurrió fue **auto-supervisión sobre audio no etiquetado a escala**. **wav2vec 2.0** (2020) combina un encoder convolucional sobre **onda cruda**, enmascaramiento de las representaciones latentes, un objetivo contrastivo contra latentes cuantizados y un Transformer sobre el contexto; su resultado emblemático es alcanzar WER competitivo en LibriSpeech con **diez minutos** de audio etiquetado y 53 000 horas sin etiquetar, lo que responde de frente la tercera pregunta del survey sobre adaptación con datos mínimos. [HuBERT](/papers/hubert-hsu-2021) (2021) reemplaza el objetivo contrastivo por **predicción enmascarada de targets discretos** obtenidos por k-means, primero sobre MFCC y luego sobre las propias representaciones del modelo: es literalmente BERT sobre audio, con el paso extra de fabricar el vocabulario. La ironía vale registrarla: **el primer paso de HuBERT clusteriza MFCC**, la feature que el survey declaró en desuso, porque para clustering la decorrelación de la DCT sí ayuda. WavLM agrega denoising al preentrenamiento y generaliza a diarización y separación. Y Whisper (2022) toma el camino opuesto y también funciona: 680 000 horas de supervisión débil raspada de la web con un encoder-decoder Transformer estándar sobre log-mel de 80 bandas. La simetría con el survey es notable: la respuesta vino de replicar el paradigma de ELMo y BERT que el propio paper señaló como lo que le faltaba al audio. **Vio el hueco con precisión y erró la lista de candidatos para llenarlo.**

**Preentrenamiento y transferencia.** AudioSet, que el survey cita, resultó ser exactamente el "ImageNet del audio" que decía que no existía. La cadena de modelos: [VGGish](/papers/vggish-hershey-2017) (2017), CNN entrenada sobre YouTube-100M que produce embeddings de 128 dimensiones y se volvió el extractor genérico por defecto; **PANNs** (2020), con CNN14 sobre AudioSet y transferencia sistemática a media docena de tareas; [AST](/papers/ast-gong-2021) (2021), Transformer puro sobre parches del espectrograma, sin convolución alguna, inicializado desde un ViT preentrenado en ImageNet; y **BEATs** (2022), con preentrenamiento auto-supervisado y un tokenizador acústico auto-destilado.

**Generación de alta calidad.** El survey dejaba a WaveNet como estado del arte con la queja de su costo de entrenamiento. Después llegaron **HiFi-GAN** y vocoders GAN similares, que dieron calidad WaveNet a cientos de veces tiempo real; **DiffWave** y **WaveGrad**, que trajeron difusión al dominio de la onda; **SoundStream** y **EnCodec**, que establecieron los **codecs neuronales** —cuantización vectorial residual que convierte audio en secuencias de tokens discretos—; y sobre esos tokens se montaron modelos de lenguaje: AudioLM, VALL-E, MusicGen, AudioGen, más difusión latente para texto-a-audio. El requisito de **controlabilidad** que el survey listaba se resolvió por una vía que no anticipaba: **condicionamiento por texto libre**. Y con eso el problema de la fase quedó resuelto de facto: nadie invierte espectrogramas con Griffin-Lim en producción; se predice mel y se sintetiza con vocoder neural, o se trabaja directamente sobre tokens de codec.

**Contexto largo.** El survey identificó el problema exacto —la RNN da contexto ilimitado pero procesa secuencialmente, la CNN es paralelizable pero tiene campo receptivo fijo— y el **Transformer** rompió el dilema: contexto global **y** paralelo. Más interesante todavía para esta clase es el **Conformer** (2020), que **fusiona convolución y self-attention dentro del mismo bloque**, con módulos feed-forward tipo *macaron* a ambos lados. Su versión grande alcanza 2.1% / 4.3% de WER en LibriSpeech test-clean/test-other sin modelo de lenguaje, y su ablación muestra que **quitar el bloque convolucional degrada el resultado**: la convolución sigue aportando aun teniendo self-attention.

{{< concept-alert type="clave" >}}
El Conformer es **la misma receta local + global + clasificación**, con self-attention en lugar de la RNN. La tesis de complementariedad entre operadores locales y globales no envejeció; lo que envejeció es el operador elegido para la parte global. **No es que el Transformer haya reemplazado a la CNN en audio: reemplazó a la RNN.**
{{< /concept-alert >}}

Lo que sigue abierto: la interpretabilidad, que es el desafío que menos se movió; la representación óptima, donde el log-mel sigue ganando por inercia y eficiencia y los front-ends aprendidos funcionan sin haberse impuesto; el cómputo en dispositivos restringidos, donde la brecha entre el estado del arte y lo que corre en un audífono es hoy **mayor** que en 2019; los datos anotados para música y sonido ambiental, donde la comunidad DCASE sigue dependiendo de datos sintéticos y débilmente etiquetados; y la separación de fuentes musicales con dependencias fuertes, el punto que el survey identificó como estructuralmente más difícil y que sigue siéndolo.

## Limitaciones del survey

**Fecha de corte.** Manuscrito recibido en octubre de 2018. Casi todo lo que un lector de hoy echa en falta cae después: SpecAugment (2019), Conv-TasNet en su versión de journal (2019), wav2vec 2.0 (2020), Conformer (2020), HiFi-GAN (2020), AST y HuBERT (2021), codecs neuronales (2021-2022), Whisper (2022). No es una crítica: es un requisito de lectura.

**Sesgo hacia el ASR y hacia Google.** Tres de los seis autores están en Google trabajando en reconocimiento de voz, y se nota. La sección de voz es la más desarrollada y la única con un arco histórico completo; la sección de sequence-to-sequence está escrita **enteramente desde ASR** (CTC, RNN-T, LAS), sin una palabra sobre seq2seq en música o en sonido ambiental; la ilustración de madurez industrial cita "Google Home, Amazon Alexa y Microsoft Cortana"; y el ejemplo de simulación de sala es literalmente el paper de generación de utterances para Google Home. Esto no invalida el contenido —los autores escriben sobre lo que mejor conocen y lo hacen bien—, pero el lector debe calibrar: la sección de voz es un estado del arte, las de música y sonido ambiental son buenos mapas con menos profundidad, y el balance del survey refleja la composición del equipo autoral más que la importancia relativa de los tres campos.

{{< concept-alert type="advertencia" >}}
**Restricción de verificabilidad.** El survey **no tiene tablas** y **no reporta una sola cifra de desempeño**: ni un WER, ni una mAP, ni una accuracy. Cualquier afirmación del tipo "según Purwins et al., el WER de X es Y" es una invención. Las únicas cifras del paper son los 14 millones de imágenes de ImageNet, los más de 2 millones de fragmentos de AudioSet, las 256 clases de la cuantización a 8 bits, y los tamaños de excerpts o campos receptivos de algunos trabajos de música (200 ms, 15 frames, 3 s, 29 s, 12 s, 60 s). Cuando el survey dice "las TF-LSTM superan a las CNN en ciertas tareas", **no es posible verificarlo desde el propio texto**: hay que ir a la referencia. Eso lo hace envejecer bien como mapa conceptual y mal como referencia de estado del arte.
{{< /concept-alert >}}

**Cobertura muy delgada de datasets.** Siete en total. Para un survey que declara la escasez de datos como el desafío central, es poco: no aparecen ESC-50, UrbanSound8K, GTZAN, MagnaTagATune, LibriSpeech, TIMIT, VoxCeleb, Common Voice ni Freesound. Quien busque una referencia de datasets debe ir a otra parte.

**Poca cobertura de multimodalidad y de traducción de voz**, y **ninguna discusión de despliegue** —nada sobre streaming contra offline más allá de mencionar RNN bidireccionales, latencia o la ingeniería de un sistema real— ni **de aspectos éticos**: nada sobre sesgos de reconocimiento por acento, dialecto o género, un tema ya documentado en 2018, ni sobre privacidad de la captura de audio siempre encendida, ni sobre deepfakes de voz, que el survey tiene a mano al describir el test de Turing como criterio de éxito de la síntesis.

**Las dos sentencias que peor envejecieron.** La primera: que "**no existe una tarea y un dataset comparables —ni modelos preentrenados sobre ellos— para el dominio del audio**". Era discutible al escribirse: VGGish existía desde 2017 y hacía exactamente eso, y el survey no lo cita. La segunda: que "los métodos de separación de fuentes del estado del arte **típicamente toman la ruta de estimar operaciones de enmascaramiento en el dominio tiempo-frecuencia**". **Conv-TasNet** invirtió eso en cuestión de meses, con un encoder-decoder convolucional aprendido en el **dominio del tiempo** y una TCN dilatada, superando incluso a las máscaras oracle ideales de tiempo-frecuencia. El survey se cubre parcialmente —reconoce que "hay enfoques que operan directamente sobre señales en el dominio del tiempo"—, pero el peso de la afirmación quedó del lado equivocado.

## Qué decía sobre atención y Transformers

Esta sección requiere el máximo rigor, porque la tentación de leer el survey a la luz de lo que vino después es fuerte. La auditoría completa de todas las menciones a atención en el texto son **cinco**:

| # | Ubicación | Texto |
|---|---|---|
| 1 | Sección de seq2seq | "**Los modelos basados en atención, que aprenden alineamientos entre las secuencias de entrada y salida conjuntamente con la optimización del objetivo, se han vuelto cada vez más populares.**" |
| 2 | Sección de seq2seq | "Entre varios modelos de secuencia a secuencia, **listen, attend and spell (LAS) ofreció mejoras sobre los demás**." |
| 3 | Fig. 2, panel E | "**La atención puede usarse para transducción de secuencias.** El encoder y el decoder de la red incluyen respectivamente una capa recurrente… El contexto $c_t$ es una suma ponderada de los embeddings del encoder, donde los pesos se calculan entre el embedding del decoder y todos los embeddings del encoder." |
| 4 | Sección de voz | "El modelo LAS es una única red neuronal que incluye un encoder análogo a un modelo acústico convencional, **un módulo de atención que actúa como modelo de alineamiento**, y un decoder análogo al modelo de lenguaje." |
| 5 | Referencias | **Vaswani et al., "Attention is all you need" (2017), citado una sola vez, dentro de una lista de tres referencias en la mención #1.** |

Y los hechos verificados sobre lo que **no** está: la palabra **"Transformer" no aparece nunca en el cuerpo del texto** —solo dentro del título de la referencia a BERT—; la expresión **"self-attention" no aparece nunca**; el inventario de modelos que "se emplean exitosamente, sin preferencia clara" es **CNN, RNN y CRNN**, y los modelos de atención no están en esa lista; y **el survey no expresa escepticismo alguno** hacia la atención: no dice que sea inviable, ni que le falten datos, ni que no modele dependencias largas.

{{< concept-alert type="clave" >}}
**El survey de 2019 no era escéptico: era agnóstico.** Su posición es precisa y limitada: la atención, entendida como mecanismo de **alineamiento dentro de un encoder-decoder recurrente** (estilo Bahdanau o LAS), es una técnica exitosa y en ascenso para transducción de secuencias en ASR. Nada más y nada menos. En la Figura 2E el encoder y el decoder son **capas recurrentes** y la atención es el puente entre ellas: eso es 2015-2016, no 2017. El survey cita a Vaswani pero no lo discute, no describe self-attention, y no contempla eliminar la recurrencia.
{{< /concept-alert >}}

Lo notable es otra cosa: **el survey nombró el diagnóstico sin dar el salto a la cura**. En su discusión de modelos dice que las RNN "requieren procesar la entrada secuencialmente, haciéndolas más lentas de entrenar y evaluar en hardware moderno que las CNN", y en su inventario de variantes dice que las TF-LSTM ganan en precisión pero "son menos paralelizables y por lo tanto más lentas". Ese es, literalmente, el argumento de apertura de *Attention is all you need*. El survey tenía el diagnóstico completo en la mano. Es un excelente caso de estudio de cómo se ve un campo justo antes de un cambio de paradigma.

| Dimensión | Survey (2019) | Realidad posterior a 2020 |
|---|---|---|
| La atención como mecanismo | "cada vez más popular" para transducción; LAS "ofreció mejoras" | Ubicua |
| El Transformer como arquitectura | Citado una vez sin discutirlo; ausente del inventario de modelos | Arquitectura dominante: Conformer, AST, Whisper, wav2vec 2.0 |
| Datos | "todas las tareas enfrentan datasets relativamente pequeños"; propone semi-supervisado, activo, few-shot | Resuelto por auto-supervisión sobre audio no etiquetado |
| Costo secuencial de la RNN | Identificado explícitamente como desventaja en hardware moderno | Es la razón por la que ganó el Transformer |
| Dependencias largas | RNN con contexto ilimitado "pero primero necesitan aprender a usarlo"; convolución dilatada como alternativa | Self-attention es *la* solución a dependencias largas; su problema es el costo $O(n^2)$, no la capacidad |

## En la clase 39: la fuente estructural

La [Clase 39](/clases/clase-39) lista este survey entre sus referencias. La lectura del material junto al paper confirma algo más fuerte: **el survey es su fuente estructural**. Coincide en la taxonomía tripartita de aplicaciones (sonidos generales / voz / música), en el debate features contra onda cruda, en la receta CNN + RNN + capas densas, y en el bloque de aumentación de datos. Y hay **dos pasajes del material de la clase que están parafraseados casi palabra por palabra del survey**: la barra lateral de reglas prácticas del diagrama del "Ejemplo 1" y la lámina sobre convolución dilatada.

| Bloque de la clase | Sección del survey que lo respalda | Relación |
|---|---|---|
| **Taxonomía de aplicaciones**: sonidos generales / voz / música | Las tres subsecciones de análisis: voz, música, sonidos ambientales | **Reproduce** la partición tripartita, incluso el orden. El survey la justifica explícitamente ("lado a lado, para señalar similitudes y diferencias"); la clase la presenta como natural, sin argumento |
| **Clasificación**: single/multiple × global/local | Categorización de problemas + **Figura 1** | **Reproduce** el cruce de los dos ejes. **Simplifica**: pierde el tercer tipo de etiqueta (valor numérico → toda la familia de regresión) y el tercer número de etiquetas (*sequence transduction*), y no usa la nomenclatura estándar de tagging / SED ni de etiquetas débiles y fuertes |
| **El espectrograma como imagen** ("Audio vs Image Data") | Introducción (ejes no homogéneos, orden cronológico) + sección de features (normalización por banda, armónicos, ventana) | **Se queda corto.** La clase señala que "hay diferencias relevantes entre datos de audio y visuales" sin enumerarlas; el survey enumera **cuatro**: ejes no homogéneos, orden cronológico obligatorio, distribuciones distintas por banda (→ normalizar por banda) y correlaciones armónicas no locales (→ tercera dimensión de armónicos). El survey es mucho más rico aquí |
| **Receta CNN + RNN + MLP** ("propiedades complementarias") | CRNN + la discusión de modelos | **Reproduce el contenido pero endurece la conclusión.** El survey dice explícitamente "**sin preferencia clara**" entre CNN, RNN y CRNN, y que en música "ni dentro ni entre tareas hay consenso". La clase presenta la combinación como *la* respuesta. También pierde cuatro matices: el campo receptivo fijo como ventaja, que la RNN debe **aprender** a usar su contexto, la imposibilidad de controlar ese contexto, y el costo de paralelismo de la recurrencia |
| **"Ejemplo 1"**: log-mel 40D + 2 conv + LSTM + capas densas | Descripción de la CLDNN en la sección de voz | **Reproduce** la CLDNN, que es la referencia [93] del survey y cuya autora es coautora del paper. Los detalles numéricos exactos están en el [paper original](/papers/cldnn-sainath-2015) |
| **Barra lateral del Ejemplo 1**: elegir por error de validación, menos parámetros con menos datos, reducir tamaño de filtro y aumentar canales | Párrafo de reglas de dedo en la sección de CNN | **Paráfrasis casi literal.** Es la prueba más directa de que el material se construyó sobre este paper. Con una distorsión sutil: el survey dice "aumentar el número de canales **a medida que decrece el tamaño de los mapas de features**" —el *feature map* es el mapa de activaciones, cuya resolución baja por el pooling, no el filtro— mientras que la clase dice "reducir el tamaño del filtro y aumentar los canales". Son dos reglas distintas fusionadas en una |
| **Audio crudo y dilatación** | Campo receptivo y convolución dilatada + fase + el veredicto de features | **Paráfrasis casi literal** en la lámina de dilatación: "alcanzar un campo receptivo suficiente lleva a un gran número de parámetros y alta complejidad computacional" y "permite campos receptivos muy grandes con apenas unas pocas capas de profundidad" son ambas del survey. **Omite** lo esencial en dos frentes: en dilatación, "**preservando la resolución de entrada**", que es la propiedad clave; y en audio crudo, el problema de la **invarianza de fase** y el veredicto condicional (log-mel empata en precisión con menos datos; onda cruda o espectro complejo cuando hay que reconstruir fase). Las tasas de muestreo que la clase menciona son aporte del profesor: el survey **no da tasas de muestreo** |
| **"Ejemplo 2"**: audio crudo + 4 conv dilatadas + 2 LSTM + 2 capas densas | Sin correspondencia directa | **Se aparta.** No corresponde a ninguna arquitectura concreta del survey: no es WaveNet (que no tiene LSTM ni MLP) ni ninguno de los trabajos de onda cruda que cita. Es una receta genérica compuesta, consistente con el espíritu del survey pero sin fuente en él |
| **Transformers**: tres problemas, "no muy populares para audio" | Sin correspondencia | **Se aparta frontalmente.** El survey trata la atención como "cada vez más popular" y cita a Vaswani sin objeciones |
| **Data augmentation**: modificar pitch, agregar ruido, time stretching, preentrenar en idiomas ricos, síntesis con reserva | Sección de datos | **Paráfrasis casi literal** de tres frases: la definición, el punto de transferencia entre idiomas, y la reserva sobre datos generados. **Omite** cuatro técnicas del catálogo: simulación de sala, filtrado espectral, mezcla de pistas para separación, y combinación lineal de ejemplos con sus etiquetas. **Agrega** "agregar ruido", que el survey no lista como tal |

El punto más importante de toda esta comparación es el último renglón sobre Transformers, y conviene dejarlo enunciado sin ambigüedad.

{{< concept-alert type="advertencia" >}}
**Las tres objeciones sobre Transformers que aparecen en el material de clase no provienen de este survey.** El material sostiene que faltan datasets masivos de audio, que self-attention opera sobre secuencias finitas de entidades discretas mientras que el audio no se segmenta trivialmente, y que los Transformers no son buenos para modelar dependencias largas. El survey **no dice nada parecido**: era agnóstico respecto del Transformer y es de 2018-2019. Son afirmaciones que corresponden a la lectura del docente, no al paper, y conviene no atribuírselas. Sobre el fondo del asunto: la primera objeción confunde datos con datos **etiquetados** —wav2vec 2.0 usa 53 000 horas sin etiquetar y Whisper 680 000 con supervisión débil—; la segunda tuvo al menos cuatro respuestas distintas y todas funcionan (parches de espectrograma en AST, cuantización de latentes en wav2vec 2.0, k-means en HuBERT, cuantización vectorial residual en los codecs neuronales); y la tercera probablemente conflaciona el **costo cuadrático** $O(n^2)$, que en audio sí duele porque las secuencias son larguísimas, con incapacidad de modelar contexto largo, que es lo opuesto de la propiedad definitoria de self-attention: conectar cualquier par de posiciones en **un solo salto**.
{{< /concept-alert >}}

Dicho eso, la tesis central de la clase —que los operadores locales, los globales y los clasificadores tienen propiedades complementarias y conviene combinarlos— **no envejeció**. El Conformer es esa misma receta con self-attention en el lugar de la RNN, y su propia ablación muestra que la convolución sigue aportando. Lo que cambió es el operador elegido para la parte global, no la idea de que hacen falta ambos.

## Notas y enlaces

- **Lectura mínima si el tiempo es escaso.** Las secciones de categorización de problemas, de features y la discusión final. Son unas cinco páginas y contienen la mayor parte del valor duradero del survey. La sección de aplicaciones por dominio es un catálogo útil para localizar literatura por tarea, pero envejeció más.
- **Cómo citarlo bien.** El survey no reporta cifras: no se le pueden atribuir números de desempeño. No dice que la onda cruda sea mejor ni que el log-mel lo sea: su veredicto es **condicional a la tarea**. No dice que los MFCC estén obsoletos: dice que la DCT es innecesaria o indeseable **con modelos de deep learning**. Y no usa la terminología de etiquetas débiles y fuertes ni menciona el *multiple instance learning*.
- **Discrepancia menor.** El cuerpo del texto data el algoritmo del perceptrón en 1957 y su propia referencia lo data en 1958. Ambas fechas circulan: el reporte técnico del Cornell Aeronautical Laboratory es de 1957 y el artículo publicado en *Psychological Review* es de 1958.
- **Fundamentos relacionados:** [representación de audio](/fundamentos/representacion-de-audio), [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel), [clasificación de audio](/fundamentos/clasificacion-de-audio), [CRNN](/fundamentos/crnn), [data augmentation de audio](/fundamentos/data-augmentation-de-audio).
- **Papers relacionados:** [CLDNN](/papers/cldnn-sainath-2015) (la arquitectura del "Ejemplo 1", con Sainath como autora en ambos), [WaveNet](/papers/wavenet-oord-2016) (el caso paradigmático de onda cruda y dilatación), [VGGish](/papers/vggish-hershey-2017) (el "ImageNet del audio" que el survey no citó), [HuBERT](/papers/hubert-hsu-2021) y [AST](/papers/ast-gong-2021) (las dos respuestas a los desafíos que el survey dejó abiertos).
- **En el site:** [Clase 39](/clases/clase-39) y su [teoría](/clases/clase-39/teoria); el recorrido histórico completo está en el [dominio de audio](/dominios/audio).
