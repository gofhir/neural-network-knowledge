---
title: "Teoría - Modelos de Deep Learning para Audio"
weight: 10
math: true
---

> **Recorrido de la Clase 39** del Diplomado IA UC (Gabriel Sepúlveda, DCC PUC). Tercera clase del hilo de audio: la [Clase 35](/clases/clase-35) cubrió la naturaleza de la señal (Fourier, muestreo, STFT, MFCC) y la [Clase 37](/clases/clase-37) el ciclo de vida del dato (formatos, librerías, datasets, augmentation). El propio slide 2 lo dice: *"So far, we have discussed... **This class: DL models to process audio**"*. Acá se responde la pregunta que quedaba: **qué modelo se pone encima del espectrograma, y qué pasa si se prescinde del espectrograma**.

{{< concept-alert type="clave" >}}
**El hilo que hay que seguir.** La clase tiene dos mitades y una coda. La primera mitad dice: *el espectrograma es una imagen, usa una CNN 2D, y combínala con una RNN porque cada una aporta algo distinto*. La segunda dice: *también puedes tirar el espectrograma y trabajar sobre la onda cruda, pero entonces el problema pasa a ser el **campo receptivo**, y la respuesta es la convolución dilatada*. La coda descarta los Transformers.

Las dos mitades son la misma pregunta —**cómo cubrir suficiente contexto temporal sin que la red explote**— resuelta con dos presupuestos distintos. Y la coda es la parte que hay que leer con más cuidado, porque en 2024 ya no se sostenía.
{{< /concept-alert >}}

---

## 1. Dónde estamos: el mapa de las tres clases de audio

El slide 2 traza el recorrido explícitamente:

| Clase | Qué cubrió | Qué dejó pendiente |
|---|---|---|
| [35](/clases/clase-35) | La naturaleza del sonido: superposición de componentes de frecuencia, features hechos a mano (espectrograma, log-mel) | Cómo se modela eso |
| [37](/clases/clase-37) | Herramientas: estructura de los archivos de audio, librerías, datasets | Cómo se modela eso |
| **39** | **Modelos de deep learning** | El habla y el hablante, que van a la [Clase 41](/clases/clase-41) |

La clase declara además su alcance: **se concentra en sonidos ambientales (general sounds)**, deja el habla y la voz para la clase siguiente, y trata la música muy brevemente. Ese reparto explica por qué el ejemplo del laboratorio es UrbanSound8K y no LibriSpeech.

---

## 2. La taxonomía de aplicaciones

Según la naturaleza de la señal, el slide 4 divide el campo en tres familias:

| Familia | Ejemplos que da el slide |
|---|---|
| **General sounds** | Clasificación de sonido ambiental, audio tagging |
| **Speech** | Reconocimiento de voz, traducción del habla, identificación de hablante |
| **Music** | Reconocimiento de canciones, identificación de instrumentos |

La división no es cosmética: cada familia tiene su propia escala de tiempo característica, su propio conjunto de datasets y su propia comunidad de investigación. Un fonema dura decenas de milisegundos; un evento sonoro urbano, segundos; una estructura musical, minutos. Esa disparidad de escalas es la que vuelve al campo receptivo el parámetro de diseño dominante en audio, y es el hilo que atraviesa toda la clase.

### 2.1 Realce y síntesis

Antes de la clasificación, el slide enumera dos familias de tareas donde la salida **también es audio**:

- **Realce (sound enhancement)**: reducción o eliminación de ruido, reconstrucción, **separación de fuentes**, transformación de fuente.
- **Síntesis**: de sonido, de habla (TTS), creación musical.

Las dos comparten una propiedad que las separa de todo lo demás: no producen una etiqueta sino una forma de onda, así que ni la métrica ni la arquitectura se parecen a las de un clasificador. La separación de fuentes se evalúa con SDR/SIR/SAR, y la síntesis con juicios humanos (MOS) porque no hay métrica automática que capture la naturalidad. [WaveNet](/papers/wavenet-oord-2016) —la primera referencia del slide final— pertenece a esta familia, y [SV2TTS](/papers/sv2tts-jia-2018) también.

### 2.2 Clasificación: la matriz que vale la pena retener

Los slides 8 y 9 presentan la distinción más útil de toda la primera parte, aunque sin nombrarla con la terminología del campo. Son **dos ejes independientes**:

|  | **Etiqueta global** (todo el clip) | **Etiquetas locales** (con marca de tiempo) |
|---|---|---|
| **Una etiqueta** | "John está hablando" | "John dice: *you* / *must* / *know* / *AI*" |
| **Múltiples etiquetas** | "En la cocina hay: gente, radio, pasos" | Tabla temporal: ID5:John, puerta, gente, pasos / ID7:Laura, radio, bicicleta, puerta, pasos |

El ejemplo del slide 9 es una tabla de dos habitaciones (cocina y ventana) con dos hablantes y eventos superpuestos, que ilustra las tres cosas a la vez: **etiquetado temporal de secuencias**, **identificación de múltiples hablantes** y **reconocimiento a distintos niveles de granularidad** ("un auto" frente a "una parte específica de un auto").

{{< concept-alert type="recordar" >}}
**La terminología estándar del campo.** Lo que el slide llama etiqueta global frente a local se conoce en la literatura como **audio tagging** frente a **sound event detection (SED)**, y la diferencia entre ambos es la de trabajar con **etiquetas débiles** (weak labels: se sabe qué suena en el clip, no cuándo) o **etiquetas fuertes** (strong labels: con inicio y fin). Es la distinción que organiza los desafíos DCASE, y la razón de ser de herramientas como [Scaper](/papers/scaper-salamon-2017): anotar límites temporales a mano es carísimo e inconsistente entre anotadores, así que se sintetizan paisajes sonoros donde las marcas de tiempo son exactas por construcción.

Vale usar los términos del campo al buscar literatura: "audio classification" devuelve todo mezclado, "weakly labeled sound event detection" devuelve el problema concreto. Se desarrolla en el fundamento [Clasificación de audio](/fundamentos/clasificacion-de-audio).
{{< /concept-alert >}}

### 2.3 Detección de eventos

El slide 10 la separa como tarea propia: **detectar un sonido específico dentro de una secuencia larga**. Los ejemplos son reveladores del rango de aplicación — "alguien está abriendo una puerta", "hay una anomalía en el latido cardíaco" — y el slide señala que pueden detectarse varios eventos simultáneos.

El segundo ejemplo merece atención en un contexto clínico: la auscultación automatizada es un caso donde el problema no es la exactitud media sino el **desbalance extremo de clases** y el costo asimétrico del error. Un detector de anomalías cardíacas que acierta el 99% pero se pierde el 1% de eventos raros es inservible, y esa es exactamente la situación en la que la métrica agregada engaña. Es el mismo argumento que ordena las métricas de recuperación en [ranking](/fundamentos/ranking-metrics).

El slide enlaza a **SoundNet** (`soundnet.csail.mit.edu`) como ejemplo. Vale saber qué es: SoundNet (Aytar, Vondrick y Torralba, 2016) entrenó una red de audio **sin etiquetas de audio**, transfiriendo conocimiento desde redes de visión ya entrenadas sobre los fotogramas del mismo vídeo. Es un antecedente directo del [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) multimodal que la [Clase 28](/clases/clase-28) cubrió, y el ancestro conceptual de [CLAP](/fundamentos/vision-language-models) y los modelos audio-texto actuales.

### 2.4 Habla y música, en dos líneas

El slide 12 lista para **habla**: reconocimiento de voz, subtitulado de vídeo, separación de audio por hablante, traducción. Todo eso va a la [Clase 41](/clases/clase-41).

El slide 14 lista para **música**: reconocimiento de canciones, similitud de estilo, detección de instrumentos, transcripción musical. La clase lo deja explícitamente de lado; el análisis de [musicnn](/papers/musicnn-pons-2019) llena ese hueco, y tiene un argumento que conviene tener a mano antes de leer la sección siguiente: **en música, la forma del filtro convolucional no debería ser cuadrada**.

---

## 3. El espectrograma como imagen — y las cuatro diferencias

El slide 16 hace el movimiento fundacional de toda la clase:

> *"En principio, la representación 2D tiempo-frecuencia (espectrograma) de una señal de audio puede interpretarse como una imagen. De este modo podemos usar CNN 2D para procesar señales de audio. Si bien esto es posible, **hay diferencias relevantes entre los datos de audio y los visuales que es importante considerar**."*

La primera parte es cierta y está empíricamente validada a gran escala: [Hershey et al. 2017](/papers/vggish-hershey-2017) tomó AlexNet, VGG, Inception-V3 y ResNet-50 tal cual, las alimentó con espectrogramas log-mel y las entrenó sobre 70 millones de vídeos de YouTube. Funcionaron. De ahí salió VGGish, el extractor de embeddings de audio que el laboratorio de esta clase usa.

La segunda parte —las "diferencias relevantes"— el slide **no la desarrolla**. Vale la pena hacerlo, porque explica varias decisiones de diseño que aparecen después sin justificación:

**(a) Los dos ejes no son intercambiables.** En una imagen, $x$ e $y$ son ambos espaciales: un gato desplazado veinte píxeles a la derecha sigue siendo el mismo gato, y desplazado veinte píxeles hacia arriba también. En un espectrograma, desplazarse en el eje temporal preserva el objeto (el mismo sonido, más tarde) pero **desplazarse en el eje de frecuencia lo transforma**: es transponer el sonido. Un ladrido desplazado una octava hacia arriba no es el mismo ladrido. La invarianza a traslación de la convolución, que en visión es un regalo, en el eje de frecuencia es una suposición que hay que decidir si se quiere.

**(b) La localidad no es simétrica.** Un kernel $3\times3$ asume que lo relevante está cerca. En frecuencia eso es falso para sonidos armónicos: la energía de una voz o de un violín está en bandas separadas por múltiplos de la frecuencia fundamental, es decir, **distribuida y periódica a lo largo de todo el eje**. Ningún kernel local ve esa estructura completa; hace falta profundidad, un kernel muy alto, o atención global.

**(c) Las fuentes se suman, no se ocluyen.** Dos objetos visuales que se superponen se **ocluyen**: el de adelante tapa al de atrás y sus píxeles son mutuamente excluyentes. Dos sonidos simultáneos se **suman**: el espectrograma resultante contiene ambos, mezclados en cada celda. Por eso la separación de fuentes es un problema tratable en audio y no tiene análogo directo en visión, y por eso el etiquetado multi-clase es la norma y no la excepción.

**(d) El eje de frecuencia ya viene deformado.** La escala mel es logarítmica por diseño perceptual — se construyó para imitar la resolución del oído humano, que discrimina mucho mejor en graves que en agudos. Un espectrograma log-mel no es una medición neutra de la señal: es una medición ya sesgada hacia lo que un humano oiría. Ver [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel).

{{< concept-alert type="advertencia" >}}
**La consecuencia de diseño más visible: el pooling solo en frecuencia.** El "Ejemplo 1" de la clase (slide 41) especifica *"max-pooling opcional **en frecuencia solamente**"*. Ahora se entiende por qué. Hacer pooling en frecuencia da invarianza a pequeños desplazamientos del espectro — que es deseable, porque el tracto vocal de cada persona desplaza levemente las formantes. Hacer pooling en tiempo destruiría la resolución temporal que el LSTM de la capa siguiente necesita. Es la asimetría (a) convertida en hiperparámetro, y es el detalle del slide que más información contiene.
{{< /concept-alert >}}

---

## 4. El menú de modelos y su reducción

Los slides 17 a 31 son un build incremental que enumera todo lo visto en el diplomado —MLP, CNN, RNN, GAN, Transformers y modelos relacionales, aprendizaje reforzado, aprendizaje por imitación, razonamiento neuro-simbólico— para después reducirlo, en los slides 32 a 38, a **tres**:

$$\text{modelo de audio} = \underbrace{\text{CNN}}_{\text{local}} + \underbrace{\text{RNN}}_{\text{temporal}} + \underbrace{\text{MLP}}_{\text{clasificación}}$$

Con la justificación de las **propiedades complementarias**:

| Bloque | Qué explota | Qué descubre |
|---|---|---|
| **CNN** | (i) el span local de cada filtro, (ii) la invarianza a traslación del operador convolucional | Features **locales** relevantes |
| **RNN** | El operador recurrente | Relaciones temporales **distantes** (globales) |
| **MLP** | Features ya informativos | Un mapeo a un espacio **discriminativo** entre clases |

{{< concept-alert type="clave" >}}
**Esta tesis es correcta y sigue vigente. Lo que cambió es quién ocupa la casilla del medio.** El argumento de que un modelo de audio necesita un operador **local** y un operador **global**, y que ninguno reemplaza al otro, es exactamente la tesis de [Conformer](/papers/conformer-gulati-2020) (2020), la arquitectura estándar del reconocimiento de voz moderno. Conformer conserva la convolución para lo local y **reemplaza la RNN por self-attention** para lo global, fusionándolas dentro de un mismo bloque.

O sea: la clase acierta en el diagnóstico y se queda corta en el remedio. Vale tener esto presente al llegar a la sección 10.
{{< /concept-alert >}}

---

## 5. Ejemplo 1: la CLDNN sin nombrarla

Los slides 39 a 44 especifican una arquitectura completa. Es la **CLDNN de [Sainath et al. 2015](/papers/cldnn-sainath-2015)**, aunque el slide no la nombra en ese punto (sí aparece en las referencias finales) y aunque tres de sus números **no coinciden** con el paper. Primero la versión del slide; las discrepancias vienen después.

```
   X_{t-n}, ..., X_t          log-mel 40D, ventanas de 10-20 ms, solape 5-10 ms
          │
          ▼
   ┌─────────────┐
   │ Convolución │            256 filtros, 9x9
   └─────────────┘            max-pool opcional SOLO en frecuencia (ventanas de 3, sin solape)
          │                   batch norm opcional
          ▼
   ┌─────────────┐
   │ Convolución │            256 filtros, 4x4
   └─────────────┘
          │
          ▼
   ┌─────────────────────┐
   │ Reducción de dim.   │    convolución 1x1
   └─────────────────────┘
          │
          ▼
   ┌─────────────┐
   │    LSTM     │            256 celdas
   └─────────────┘            (normalizar el largo de secuencia dentro del minibatch)
          │
          ▼
   ┌─────────────┐
   │    LSTM     │            256 celdas
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │  FC (MLP)   │            1.024 unidades
   └─────────────┘
          │
          ▼
   ┌─────────────┐
   │  FC (MLP)   │            1.024 unidades
   └─────────────┘
          │
          ▼
       Softmax                (o sigmoides, si las etiquetas son múltiples)
```

### 5.1 Tres detalles del slide 41 que no son decorativos

**La capa de reducción de dimensión.** El slide dice que *"permite reducir parámetros sin pérdida de exactitud"*, y ahí acierta de lleno. La salida de la segunda convolución tiene 256 mapas sobre una grilla tiempo-frecuencia, y aplanar eso para entregárselo al LSTM produce un vector de entrada enorme cuyo costo se paga entero en la matriz de pesos entrada→estado del LSTM — la matriz más cara de la red. Interponer una proyección que baje esa dimensión antes del LSTM cuesta casi nada y ahorra muchísimo. La [profundización](profundizacion) hace la contabilidad exacta.

**El softmax *o* las sigmoides.** Es la matriz de la sección 2.2 convertida en función de pérdida. Una etiqueta por clip: softmax con entropía cruzada categórica, y las probabilidades suman 1. Múltiples etiquetas simultáneas: una sigmoide independiente por clase con entropía cruzada binaria, porque "hay un perro" y "hay una sirena" no compiten entre sí. Confundirlas es el error más común al empezar en audio tagging: un softmax fuerza al modelo a repartir masa de probabilidad entre eventos que ocurren a la vez.

**"Normalizar el largo de la secuencia en el minibatch".** Es el problema del batching de largo variable que la [Clase 37](/clases/clase-37) trató con las `collate_fn`. El audio no viene en clips de largo fijo, y un LSTM sobre un tensor rectangular necesita que todo el batch tenga la misma longitud: se rellena y se enmascara, o se agrupan ejemplos de largo parecido.

### 5.2 Dónde el slide se aparta del paper

Tres números del Ejemplo 1 no son los de Sainath et al. Ninguno cambia la idea, pero conviene saberlo antes de intentar reproducir la arquitectura o de citarla:

| Elemento | Slide 41 | Paper (verificado en el PDF) |
|---|---|---|
| Segundo kernel convolucional | $4\times4$ | **$4\times3$** |
| Celdas por capa LSTM | 256 | **832 celdas + proyección de 512** |
| Capa de reducción | "convolución $1\times1$" | **capa lineal** (aplana $1792 \to 256$) |

La tercera es la que más importa, porque las dos operaciones no hacen lo mismo. Una convolución $1\times1$ proyecta canal a canal **preservando** los ejes de tiempo y frecuencia: reduce el número de mapas, pero al aplanar el resultado siguen quedando tiempo × frecuencia × mapas. La capa lineal del paper aplana **todo** el bloque de una vez y lo proyecta a un vector de 256. Las dos reducen la dimensión que llega al LSTM —la [profundización](profundizacion) cuantifica cuánto ahorra cada una—, pero solo la segunda colapsa la estructura de la grilla, que es lo que el paper describe.

Lo que sí coincide exactamente: los 40 log-mel, el salto de 10 ms, los dos bloques convolucionales de 256 mapas, el $9\times9$ del primero, el pooling de 3 sin solape aplicado solo en frecuencia, y las dos capas FC de 1.024.

{{< concept-alert type="advertencia" >}}
**"Invarianza a traslación" es vocabulario de libro de texto, no del paper.** La palabra *invariance* no aparece ni una vez en Sainath et al. 2015. El paper habla de **reducir la varianza espectral** (*"reduce frequency variance"*), y su argumento es más concreto que el genérico: la CNN cumple el rol que en el pipeline clásico de reconocimiento de voz cumplía la **normalización de la longitud del tracto vocal (VTLN)**, que compensa el hecho de que un mismo fonema aparece a frecuencias distintas según la anatomía de quien habla. Eso también explica el orden C→L→D: en la receta GMM/HMM, la normalización iba antes del modelo de secuencia.

La formulación del slide no es incorrecta, pero pierde la razón específica por la que el pooling va en frecuencia y no en tiempo.
{{< /concept-alert >}}

**El softmax *o* las sigmoides.** Es la matriz de la sección 2.2 convertida en función de pérdida. Una etiqueta por clip: softmax con entropía cruzada categórica, y las probabilidades suman 1. Múltiples etiquetas simultáneas: una sigmoide independiente por clase con entropía cruzada binaria, porque "hay un perro" y "hay una sirena" no compiten entre sí. Confundirlas es el error más común al empezar en audio tagging: un softmax fuerza al modelo a repartir masa de probabilidad entre eventos que ocurren a la vez.

**"Normalizar el largo de la secuencia en el minibatch".** Es el problema del batching de largo variable que la [Clase 37](/clases/clase-37) trató con las `collate_fn`. El audio no viene en clips de largo fijo, y un LSTM sobre un tensor rectangular necesita que todo el batch tenga la misma longitud: se rellena y se enmascara, o se agrupan ejemplos de largo parecido.

El slide 44 cierra con cuatro reglas prácticas, todas honestas:

- Los detalles de la estructura **se eligen experimentalmente** según el error de validación.
- La frecuencia de muestreo y la aplicación determinan la ventana de contexto temporal.
- Menos parámetros cuando hay pocos datos, para evitar sobreajuste.
- **Reducir el tamaño del filtro y aumentar el número de canales en las capas profundas.**

La última es la regla de diseño heredada de VGG, y la razón por la que el Ejemplo 1 pasa de $9\times9$ a $4\times4$.

---

## 6. ¿Se puede usar la onda cruda? Los cuatro problemas

Los slides 45 a 53 hacen el giro. El planteo es en cinco pasos, cada uno un build:

1. Los espectrogramas y el log-mel son **features hechos a mano**.
2. ¿Se puede usar deep learning para aprender los features directamente del dato crudo?
3. **Sí, pero hay que considerar algunos problemas.**
4. Para no perder información hay que muestrear a **15-20 kHz** (44.1 kHz para música). Eso implica muchísimas muestras por segundo. *¿Algún problema?*
5. Con una arquitectura convolucional harían falta **filtros enormes o una estructura muy profunda**. *¿Por qué?*
6. Se puede aumentar el campo receptivo de las neuronas intermedias usando **filtros de convolución dilatados**.

La pregunta retórica del paso 5 tiene una respuesta cuantitativa contundente. Una convolución con kernel $k=3$ apilada $L$ veces, sin stride ni dilatación, tiene un campo receptivo de $1 + 2L$ muestras. Para cubrir **un segundo** de audio a 16 kHz:

$$1 + 2L = 16\,000 \quad\Longrightarrow\quad L = 7\,999 \text{ capas}$$

Ocho mil capas para ver un segundo. Ese es el problema, en una línea.

{{< concept-alert type="recordar" >}}
**La familia M del laboratorio es la respuesta "estructura muy profunda", sin dilatación.** [Dai et al. 2017](/papers/raw-waveforms-dai-2017) —el paper del práctico de esta clase— ataca el mismo problema con la otra herramienta: **stride y pooling agresivos**. Su primera capa usa un kernel de **80** con stride 4 seguido de max-pooling 4, lo que reduce la entrada por un factor de ~16 de una sola vez, y el resto de la red usa kernels de 3 al estilo VGG. El kernel de 80 no es arbitrario: a 8 kHz, 80 muestras son **10 ms**, exactamente la escala de la ventana estándar de los MFCC. Los autores observan que lo que aprende esa primera capa se parece a un banco de filtros.

Con esa receta, M11 y M18 alcanzan campos receptivos de ~880 ms y ~1.5 s. Sin una sola convolución dilatada. Ver la [profundización](profundizacion) para la aritmética completa de las tres estrategias.
{{< /concept-alert >}}

---

## 7. Convolución dilatada

Los slides 54 y 55 introducen el operador con dos figuras: la grilla 2D con dilatación 1, 2 y 3, y el diagrama en árbol de WaveNet con capas de dilatación 1, 2, 4 y 8. Los tres puntos del slide:

- Para una CNN, alcanzar un campo receptivo suficiente lleva a **muchos parámetros y alta complejidad computacional**.
- Las convoluciones dilatadas permiten **campos receptivos muy grandes con pocas capas de profundidad**.
- En audio crudo, tras pocas capas las neuronas pueden cubrir **miles de timesteps** manteniendo una eficiencia razonable.

La idea es simple: en vez de tomar muestras contiguas, el filtro toma muestras separadas por $d$ posiciones. El número de parámetros **no cambia** —el filtro sigue teniendo $k$ pesos— pero el tramo que abarca crece por un factor $d$. Apilando dilataciones que se duplican, el campo receptivo crece de forma **exponencial** con la profundidad en vez de lineal:

$$R = 1 + \sum_{l=1}^{L} (k_l - 1)\, d_l$$

Con $k=3$ y $d_l = 2^{l-1}$, la misma cobertura de un segundo a 16 kHz que exigía ocho mil capas densas se consigue con **13**. Ese contraste —7.999 contra 13— es el argumento entero de la sección.

El costo: los huecos. Un filtro dilatado no mira las posiciones intermedias, y apilar varias capas con la misma dilatación produce el artefacto de *gridding* (posiciones de entrada que nunca se consultan). WaveNet lo mitiga repitiendo bloques de dilataciones $1,2,4,\dots,512$ varias veces, de modo que cada bloque nuevo rellena lo que el anterior salteó. Se desarrolla en el fundamento [Convoluciones dilatadas](/fundamentos/convoluciones-dilatadas).

---

## 8. Ejemplo 2 — y la cuenta que no cierra

Los slides 56 a 60 dan la contraparte del Ejemplo 1, ahora sobre onda cruda:

| Bloque | Especificación del slide |
|---|---|
| **Entrada** | Audio crudo, 15.000-20.000 muestras por segundo, serie temporal 1D |
| **CNN** | **4 capas convolucionales dilatadas** (el factor de dilatación "depende de la aplicación"), con 128, 128, 256 y 256 filtros y kernels de $20\times1$, $10\times1$, $10\times1$ y $5\times1$. Max-pooling y batch norm opcionales |
| **RNN** | 2 capas LSTM de 256 celdas. Normalizar el largo de secuencia en el minibatch. El alcance del *unrolling* depende de la aplicación |
| **MLP** | 2 capas FC de 1.024 unidades. Salida softmax o sigmoides |

Vale tomarse en serio la promesa del slide anterior —"miles de timesteps tras pocas capas"— y verificarla sobre este ejemplo concreto. Aplicando la fórmula del campo receptivo a los kernels $20, 10, 10, 5$:

| Configuración | Campo receptivo | En tiempo (@16 kHz) |
|---|---|---|
| Sin dilatación ni stride | 42 muestras | **2.6 ms** |
| Dilataciones $1, 2, 4, 8$ | 106 muestras | **6.6 ms** |
| Dilataciones $1, 4, 16, 64$ | 456 muestras | 28.5 ms |
| Dilataciones $1, 8, 64, 512$ | 2.716 muestras | 169.8 ms |
| Dilataciones $1, 16, 256, 4096$ | 18.852 muestras | 1.178 ms |
| Dilataciones $1,2,4,8$ + max-pool 4 tras cada conv | 2.716 muestras | 169.8 ms |
| **Dilataciones $1, 20, 200, 2000$** (el máximo sin huecos) | **10.000 muestras** | **625 ms** |

{{< concept-alert type="advertencia" >}}
**El Ejemplo 2, tal como está escrito, no cumple la promesa del Ejemplo anterior.** Con dilatación exponencial de base 2 —la configuración canónica, la que muestra la figura del slide 55— esas cuatro capas cubren **6.6 milisegundos**. No "miles de timesteps": ciento seis. Ni siquiera alcanza para una ventana de análisis estándar de 25 ms.

El principio del slide 51 es correcto; el ejemplo del slide 57 no lo instancia. Para que esas cuatro capas cubran un segundo hacen falta dilataciones de base 16 ($1, 16, 256, 4096$), y ahí el *gridding* es severo: la última capa consulta 5 muestras separadas por 4.096 posiciones cada una, es decir, un cuarto de segundo de silencio entre pesos consecutivos.

Esto no invalida la arquitectura: el slide dice que el factor de dilatación "depende de la aplicación" y agrega max-pooling opcional, y con pooling 4 entre capas se llega a ~170 ms, que ya es una ventana útil. Pero muestra que **la dilatación sola no basta con cuatro capas**: o se apilan muchas más (WaveNet usa 30), o se combina con reducción de resolución (la familia M usa stride y pooling), o se acepta un campo receptivo pequeño y se delega el contexto largo al LSTM que viene después. Lo tercero es, probablemente, lo que el ejemplo tiene en mente — y es una decisión legítima que el slide no explicita.
{{< /concept-alert >}}

La última fila de la tabla merece explicación, porque es el mejor resultado posible y no es obvio. Existe una condición que evita los huecos: para que la capa $l+1$ no saltee posiciones que la capa $l$ nunca consultó, basta que su dilatación no supere el campo receptivo acumulado hasta ahí, $d_{l+1} \le R_l$. Tomando siempre el máximo permitido con los kernels del slide se obtiene el programa $1, 20, 200, 2000$, que cubre **10.000 muestras (625 ms a 16 kHz) sin un solo hueco**. Es más de tres veces lo que da la duplicación clásica con pooling, y sin pooling.

Es un resultado que el slide no menciona y que vale como regla práctica general: **la progresión de dilataciones no debería duplicarse por costumbre, sino seguir al campo receptivo**. Con kernel 2 —el caso de WaveNet— esa regla da exactamente la duplicación $1, 2, 4, 8, \dots$, que resulta ser el crecimiento máximo posible sin gridding. Con kernels grandes, duplicar desperdicia casi todo el potencial. La derivación completa está en la [profundización](profundizacion).

La [profundización](profundizacion) desarrolla las tres estrategias —profundidad densa, dilatación y stride/pooling— con la aritmética completa y la comparación de costos.

---

## 9. Audio y Transformers: la sección que hay que auditar

El slide 61 es la coda de la clase, y es donde más ha envejecido el material. Afirma que ha habido trabajos previos en audio con arquitecturas Transformer, pero que existen **tres problemas relevantes**:

1. *"En el contexto del audio, todavía hay una falta de datasets de audio masivos."*
2. *"El mecanismo de self-attention opera sobre una secuencia finita de entidades discretas. En texto la segmentación en oraciones es trivial, pero para audio no lo es."*
3. *"Los Transformers no son buenos para modelar dependencias largas en secuencias."*

Y concluye: *"Como consecuencia, los Transformers no son actualmente muy populares para aplicaciones de audio."*

El PDF de la clase es de **abril de 2024**. Vale contrastar cada afirmación con lo que ya existía.

### 9.1 La objeción 1: faltan datasets masivos

Fue un problema real, y es la objeción que mejor envejeció — pero se resolvió **cambiando la pregunta**. El aprendizaje autosupervisado no necesita datos etiquetados: necesita datos. [HuBERT](/papers/hubert-hsu-2021) (2021) se pre-entrenó sobre **60.000 horas** de Libri-Light sin una sola transcripción, y luego alcanza resultados usables con diez minutos de audio etiquetado. [wav2vec 2.0](/papers/wav2vec2-baevski-2020) (2020) había hecho el mismo movimiento un año antes, y [Whisper](/papers/whisper-radford-2022) (2022) lo escaló a 680.000 horas de supervisión débil.

[AST](/papers/ast-gong-2021) resolvió el mismo problema por otra vía todavía más económica: **transferencia cross-modal**. Toma un Vision Transformer entrenado en ImageNet y le transfiere los pesos a un modelo que consume espectrogramas, adaptando el embedding posicional con un mecanismo de corte e interpolación. Es decir: el dataset masivo que faltaba en audio se tomó prestado de visión.

{{< concept-alert type="recordar" >}}
**La objeción 1 era real, y el propio AST lo demuestra.** Su ablation más contundente: entrenado sobre los 2 millones de clips de AudioSet **sin** pre-entrenamiento de ImageNet, AST cae de 0.459 a 0.366 de mAP — y con eso pierde contra las CNN de la época (PANNs 0.439, PSLA 0.444). Sin datos prestados, el Transformer puro no ganaba.

Así que la objeción del slide identificaba un problema genuino. Lo que no registra es que ese problema **ya tenía dos soluciones desplegadas** —la transferencia cross-modal y el pre-entrenamiento autosupervisado—, y que ninguna consistió en conseguir el dataset etiquetado que faltaba.
{{< /concept-alert >}}

### 9.2 La objeción 2: el audio no se segmenta en entidades discretas

Es una observación acertada sobre el problema, y por eso mismo fue el punto de partida de la investigación que la resolvió. Los propios autores de HuBERT enuncian esa dificultad —las unidades sonoras tienen longitud variable y no vienen segmentadas— como el problema a atacar, y lo resuelven **fabricando** las unidades: clustering k-means sobre features acústicos genera un vocabulario discreto artificial sobre el que se aplica una pérdida de predicción enmascarada estilo BERT. Su hallazgo más interesante es que las unidades del maestro **no necesitan ser correctas**, solo consistentes.

AST lo resuelve de una manera aún más simple, y esa simplicidad es el argumento: **la tokenización no necesita ser semántica**. Corta el espectrograma en parches de $16\times16$ solapados, igual que ViT hace con las imágenes. Un parche no corresponde a ningún fonema ni a ningún evento; es una grilla regular. Y funciona.

### 9.3 La objeción 3: los Transformers no modelan bien las dependencias largas

Esta es la que no se sostiene, porque **invierte la motivación original de la self-attention**. El argumento de Vaswani et al. (2017) para reemplazar la recurrencia fue precisamente el opuesto: en una RNN, la señal entre dos posiciones separadas por $n$ pasos recorre un camino de longitud $O(n)$, lo que degrada el gradiente; en un Transformer ese camino es de longitud $O(1)$, porque toda posición atiende a toda posición en una sola operación. La longitud máxima del camino es la métrica que la tabla de complejidad de ese paper usa para justificar el diseño.

La limitación real de los Transformers con secuencias largas es **el costo cuadrático** $O(T^2)$ en cómputo y memoria, no la capacidad de modelar la dependencia. Son cosas distintas: una es un problema de presupuesto y la otra sería un problema de expresividad. Ver [Self-attention](/fundamentos/self-attention).

La evidencia empírica más limpia está dentro de un solo paper, sobre un solo dataset y en una sola condición experimental — la Tabla 2 de Conformer, sobre LibriSpeech test/test-other con modelo de lenguaje:

| Modelo | Operador global | Parámetros | WER test / test-other |
|---|---|---|---|
| LAS | LSTM | 360M | 2.2 / 5.2 |
| **Conformer S** | **self-attention** | **10.3M** | **2.1 / 5.0** |

Treinta y cinco veces menos parámetros, y mejor en las dos columnas. Si los Transformers modelaran peor las dependencias largas, este resultado no debería existir: el reconocimiento de voz es precisamente la tarea donde el contexto largo decide.

### 9.4 El estado real del campo en abril de 2024

| Modelo | Año | Qué hace | Qué objeción refuta |
|---|---|---|---|
| [wav2vec 2.0](/papers/wav2vec2-baevski-2020) | 2020 | SSL contrastivo sobre audio sin etiquetar | 1 |
| [Conformer](/papers/conformer-gulati-2020) | 2020 | Convolución + self-attention en un bloque; supera a un Transformer Transducer con una fracción de los parámetros | 3 |
| [AST](/papers/ast-gong-2021) | 2021 | ViT sobre parches de espectrograma; estado del arte en AudioSet, ESC-50 y Speech Commands | 1 y 2 |
| [HuBERT](/papers/hubert-hsu-2021) | 2021 | Unidades discretas por clustering + predicción enmascarada | 1 y 2 |
| [Whisper](/papers/whisper-radford-2022) | 2022 | Transformer encoder-decoder, 680k h, ASR multilingüe robusto | 1 |

{{< concept-alert type="clave" >}}
**Lo que sí conviene rescatar de esta sección.** No que los Transformers sean inadecuados para audio —no lo son—, sino que **el audio impone restricciones que el texto no**, y que esas restricciones exigen adaptaciones concretas. La secuencia de audio es larguísima y continua, así que todos los Transformers de audio la reducen antes de atender: AST con parches solapados, wav2vec 2.0 y HuBERT con un encoder convolucional que baja a 50 tramas por segundo, Conformer con un stem convolucional de submuestreo, Whisper con dos convoluciones iniciales.

Y esa reducción es, otra vez, **una convolución**. La intuición del profesor sobre la complementariedad entre operador local y operador global no solo sigue viva: es la arquitectura de todos los modelos de la tabla de arriba. Lo que cambió es que la RNN dejó de ser el operador global.
{{< /concept-alert >}}

---

## 10. Data augmentation

El slide 62 cierra con la generación de datos de entrenamiento manipulando los ejemplos existentes. Las estrategias que lista:

- **Modificar el pitch**, **agregar ruido**, **hacer time stretching**.
- En reconocimiento de voz, **pre-entrenar en idiomas con más datos transcritos** y adaptar después a uno de bajos recursos o a un dominio nuevo.
- **Usar técnicas de síntesis de sonido** — con la advertencia de que *"el desempeño sobre datos reales puede ser pobre si el modelo se entrena solo con datos sintéticos: hace falta fine-tuning sobre datos reales"*.

Los dos primeros bloques ya se cubrieron en la [Clase 37](/clases/clase-37) y en el fundamento [Data Augmentation de Audio](/fundamentos/data-augmentation-de-audio), incluyendo [SpecAugment](/papers/specaugment-park-2019), que aplica el enmascaramiento directamente sobre el espectrograma y que es la técnica que se volvió estándar.

El tercero es el que aporta algo nuevo, y remite a las dos últimas referencias del slide final. [Scaper](/papers/scaper-salamon-2017) sintetiza paisajes sonoros completos: en vez de transformar un ejemplo existente, **compone ejemplos nuevos con sus anotaciones exactas**, porque quien genera la mezcla sabe con precisión de milisegundo cuándo empieza y termina cada evento. Y [SV2TTS](/papers/sv2tts-jia-2018) sintetiza habla de identidades arbitrarias a partir de segundos de audio de referencia.

{{< concept-alert type="advertencia" >}}
**Cuidado con las transformaciones que destruyen la etiqueta.** El pitch shifting es un aumentador excelente para clasificación de sonido ambiental y desastroso para identificación de hablante, reconocimiento de instrumentos o detección de tonalidad musical: transpone justamente la propiedad que se quiere predecir. El time stretching es inofensivo para detectar una sirena y destructivo para clasificar tempo musical. La regla general: una transformación es válida como augmentation si y solo si preserva la etiqueta, y eso **depende de la tarea, no de la señal**.

Y sobre la advertencia del slide respecto de los datos sintéticos: la razón es la **brecha de dominio**. Una mezcla aditiva de eventos sobre un fondo ignora la acústica de la sala (reverberación, respuesta del espacio), la respuesta del micrófono y la correlación real entre eventos — en el mundo, qué sonidos co-ocurren no es uniforme. Un modelo entrenado solo con eso puede aprender los artefactos de la síntesis en lugar de la señal.
{{< /concept-alert >}}

---

## 11. Las referencias del slide final

El slide 63 lista siete trabajos. Vale saber qué es cada uno y por qué está ahí:

| Referencia | Qué aporta a la clase |
|---|---|
| [WaveNet](/papers/wavenet-oord-2016) (van den Oord et al., 2016) | El origen de la convolución dilatada en audio y la figura del slide 55 |
| [CLDNN](/papers/cldnn-sainath-2015) (Sainath et al., 2015) | El "Ejemplo 1", capa por capa |
| [Deep Learning for Audio Signal Processing](/papers/dl-audio-purwins-2019) (Purwins et al., 2019) | El survey del que sale la estructura de toda la clase. Uno de sus coautores es Tara Sainath |
| [musicnn](/papers/musicnn-pons-2019) (Pons y Serra, 2019) | La familia "música", tratada brevemente |
| [AudioSet](/papers/audioset-gemmeke-2017) (Gemmeke et al., 2017) | La ontología y el dataset sobre el que se pre-entrena VGGish |
| [SV2TTS](/papers/sv2tts-jia-2018) (Jia et al., 2018) | Síntesis de habla como augmentation, y transferencia entre tareas de audio |
| [Scaper](/papers/scaper-salamon-2017) (Salamon et al., 2017) | Síntesis de paisajes sonoros con anotación exacta |

{{< concept-alert type="recordar" >}}
**Dos erratas menores del material.** El slide cita SV2TTS como "Jia et al., 2019"; el trabajo es de **2018** (NeurIPS 2018, preprint de junio de ese año). Y el notebook del laboratorio enlaza VGGish al preprint `arXiv:1610.00087`, que en realidad es el paper de **Dai et al. sobre raw waveforms**; VGGish es `arXiv:1609.09430` (Hershey et al.). Vale tenerlo presente al buscar las fuentes.
{{< /concept-alert >}}

---

## 12. Qué se lleva uno de esta clase

Tres cosas, en orden de utilidad:

**El pooling solo en frecuencia.** Es el detalle que resume por qué un espectrograma no es una imagen. Los ejes no son intercambiables, y la arquitectura debe reflejarlo. Todo lo demás de la asimetría audio/imagen se deduce de ahí.

**El campo receptivo como parámetro de diseño dominante.** En visión uno rara vez calcula el campo receptivo; en audio crudo es lo primero que hay que calcular, porque la señal tiene tres o cuatro órdenes de magnitud más de muestras por unidad de contenido. Las tres estrategias para crecerlo —profundidad densa, dilatación, stride y pooling— tienen costos distintos y se combinan. La aritmética está en la [profundización](profundizacion) y el código en la [práctica](practica).

**La tesis de la complementariedad, con el remedio actualizado.** Un modelo de audio necesita un operador local y uno global. El local sigue siendo la convolución. El global dejó de ser la RNN en 2020.

---

## Para seguir

- [Profundización](profundizacion) — la aritmética del campo receptivo, la contabilidad de parámetros del Ejemplo 1, por qué el espectrograma no es una imagen, y el costo comparado de convolución, recurrencia y atención.
- [Práctica desde 0](practica) — construir y **medir** el campo receptivo de una pila dilatada, y armar la CLDNN del Ejemplo 1, en PyTorch, TensorFlow y JAX.
- [Clase 41](/clases/clase-41) — habla y hablante, la continuación de este hilo.
- [Dominio: Audio / Voz](/dominios/audio) — la línea de tiempo completa.
