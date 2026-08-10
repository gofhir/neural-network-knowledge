---
title: "Scaper: síntesis y augmentation de paisajes sonoros (2017)"
weight: 429
math: true
---

{{< paper-card
    title="Scaper: A Library for Soundscape Synthesis and Augmentation"
    authors="Justin Salamon, Duncan MacConnell, Mark Cartwright, Peter Li, Juan Pablo Bello (MARL, New York University)"
    year="2017"
    venue="WASPAA 2017"
    pdf="/papers/scaper-salamon-2017.pdf" >}}
El paper parte de una asimetría brutal: para detectar eventos sonoros hace falta saber **cuándo** empieza y termina cada sonido, y en 2017 el dataset público de referencia con ese tipo de anotación —el *development set* de TUT Sound Events 2016— duraba **78 minutos**, mientras ImageNet llevaba cinco años entrenando visión con 1.2 millones de imágenes. La respuesta de Scaper es un cambio de categoría más que una mejora incremental: **si tú compones la mezcla, la anotación no se estima, se lee del generador**. Scaper es una librería Python que, dado un *soundbank* de eventos aislados organizados en carpetas de primer plano y de fondo, actúa como un **secuenciador de audio controlado probabilísticamente**: el usuario declara una *event specification* donde cada propiedad (etiqueta, archivo fuente, tiempo de inicio, duración, SNR, *pitch shift*, *time stretch*) es una **distribución** en vez de un valor, y de esa única especificación se instancian infinitos paisajes distintos, cada uno con su anotación fuerte exacta y con la **receta completa en formato JAMS** que permite regenerarlo o editarlo. Con ella los autores construyen **URBAN-SED** (10 000 paisajes de 10 s, casi 30 horas, ~50 000 eventos anotados, más de 20× el dataset real de su momento) sobre los clips de [UrbanSound8K](/papers/urbansound8k-salamon-2014), y ejecutan dos barridos controlados —desempeño frente a **polifonía máxima** y frente a **SNR**— que muestran algo que ninguna métrica agregada revela: una F-measure plana esconde una precisión que cae 14 puntos y un recall que sube 11. El paper también acota su propio alcance sin defensas: lo sintético no reemplaza a las grabaciones reales anotadas si lo que se quiere es estimar el desempeño en producción.
{{< /paper-card >}}

---

## Contexto: dos tareas que se confunden todo el tiempo

Casi todo el paper sale de una distinción que conviene fijar antes de seguir.

| | **Audio tagging** (etiquetas débiles) | **Sound event detection** (etiquetas fuertes) |
|---|---|---|
| Pregunta | ¿Qué fuentes están presentes en este clip? | ¿Qué fuentes, y **cuándo** empieza y termina cada instancia? |
| Salida | Vector multi-etiqueta por clip | Secuencia de tripletas $(\text{onset}, \text{offset}, \text{label})$ |
| Anotación | Marcar presencia/ausencia | Marcar **límites temporales** de cada evento |
| Costo por hora | Bajo | Alto, y crece con la densidad de eventos |
| Métrica típica | mAP, AUC por clase | F-measure por segmentos o por eventos (`sed_eval`) |

El paper define **sound event detection (SED)** como "la tarea de identificar automáticamente la fuente y la ubicación en el tiempo de distintos sonidos a medida que ocurren en un flujo de audio continuo", y acto seguido enuncia el problema: la mayoría de los modelos de SED requieren etiquetas fuertes. Con un matiz que suele pasarse por alto:

> "Incluso los modelos que **pueden** entrenarse con datos débilmente etiquetados requieren datos fuertemente etiquetados **para evaluar** su desempeño a resoluciones temporales más finas."

La supervisión débil libera del costo de anotación en el conjunto de **entrenamiento**, pero no en el de **evaluación**: si la métrica es temporal, hace falta verdad de terreno temporal. Y las aplicaciones que el paper enumera —monitoreo de ruido urbano, bioacústica, autos autónomos, vigilancia, indexación multimedia— comparten justamente la necesidad de saber *cuándo*: un sensor que reporta "hubo un martillo neumático en algún momento de esta hora" no sirve para nada regulatorio, porque lo que importa es la duración acumulada de exposición. La cifra que el paper usa como ancla es la que duele: el *development set* de **TUT Sound Events 2016 dura 78 minutos**. Esa desproporción no es contexto decorativo: **es la razón de existir de Scaper**.

### Por qué la anotación fuerte es tan cara

El paper dice que anotar límites temporales es "laborioso y consume mucho tiempo", pero el problema es peor que el costo por hora-humano: **los límites temporales de un evento sonoro no son un hecho objetivo bien definido**, y por lo tanto no existe un anotador perfecto al que aspirar. Basta recorrer las clases urbanas de UrbanSound8K. Un **disparo** es casi un impulso, con onset inequívoco al milisegundo, pero su offset depende de si se corta al decaer el transitorio o al perderse la cola reverberante bajo el fondo. Un **aire acondicionado** o un **motor en ralentí** son estacionarios y difusos: no tienen onset perceptual, se van imponiendo, y dos anotadores competentes pueden diferir en varios segundos sin que ninguno esté equivocado. Un **martillo neumático** es una secuencia de impactos: ¿un evento con estructura interna o veinte eventos? La respuesta cambia la anotación de un clip de un registro a veinte. **Niños jugando** o **música callejera** son texturas compuestas cuyo límite es semántico, no acústico.

De ahí salen tres consecuencias que ponen un **techo duro** a los datasets reales:

1. **El desacuerdo entre anotadores es irreducible para las clases difusas.** No es ruido que se promedie a cero con más anotadores, porque no hay un valor verdadero latente al que converger: hay una decisión de convención, y el protocolo que la impone pasa a ser parte de la definición de la tarea.
2. **El desempeño reportado está acotado por el acuerdo entre anotadores.** Si dos humanos concuerdan a 0.75 de F-measure, un modelo que reporta 0.78 no está superando a los humanos: está sobreajustando las idiosincrasias del anotador que produjo el *ground truth*.
3. **La métrica hereda la ambigüedad.** Por eso `sed_eval` ofrece evaluación *segment-based* además de *event-based*: cuantizar el tiempo en segmentos de 1 s amortigua el desacuerdo en los límites. El paper elige exactamente esa métrica, y no por casualidad.

### Por qué el augmentation clásico no alcanza

Aquí el paper hace la observación que es la bisagra entre "augmentation" y "síntesis":

> "Para SED, dado que los datos de entrenamiento están compuestos de paisajes sonoros que contienen múltiples eventos, las aumentaciones aplicadas al paisaje sonoro **como un todo** ciertamente pueden ayudar, pero están limitadas en que las características del paisaje sonoro —como **el timing de los eventos, el grado de solapamiento y la SNR**— permanecerán **sin cambios** aun después de la transformación."

Si a un paisaje real de 10 s con cuatro eventos se le aplica *pitch shift*, *time stretch* o ruido, se obtiene un ejemplo nuevo, pero la **estructura de la escena** es la misma: los mismos cuatro eventos, el mismo orden, el mismo patrón de solapamiento. La aumentación clásica genera variación **dentro de** una escena; no genera **escenas nuevas**. Y para SED, la estructura de la escena es precisamente lo que el modelo tiene que aprender a desenredar.

## La idea central: si generas el paisaje sonoro, las etiquetas vienen gratis

El razonamiento es casi tautológico y por eso mismo es fuerte. Un paisaje sintetizado es una suma de fuentes cuyos tiempos de inicio y duraciones **son parámetros del proceso de síntesis**; los parámetros son conocidos porque fueron elegidos; luego la anotación fuerte no se estima, **se lee del generador**. Si el paisaje se construye como

$$x(t) = b(t) + \sum_{i=1}^{N} g_i \cdot e_i(t - \tau_i)$$

con $b$ el fondo, $e_i$ el $i$-ésimo evento ya recortado y transformado, $\tau_i$ su tiempo de inicio y $g_i$ su ganancia, la anotación fuerte es exactamente $\mathcal{A} = \{(\tau_i,\; \tau_i + d_i,\; \ell_i)\}_{i=1}^{N}$, con $d_i$ la duración post-transformación. No hay estimación, no hay anotador, no hay desacuerdo, no hay error: la anotación es exacta *por construcción*, y el costo marginal de una hora anotada pasa de decenas de dólares y horas-humano a segundos de CPU.

El paper lleva ese razonamiento a su conclusión lógica en una decisión pequeña y muy reveladora. Para el fondo de URBAN-SED usa **ruido browniano sintético**, el mismo clip para los 10 000 paisajes, porque "al usar un fondo puramente sintetizado tenemos la garantía de que **no contiene ningún evento sonoro espurio que no estaría incluido en la anotación**". Un fondo urbano real casi seguro contiene una bocina lejana, un perro, un frenazo — eventos de las propias clases objetivo, presentes en el audio y **ausentes de la anotación**: cada uno es un falso negativo forzado en el *ground truth*, donde el modelo detecta correctamente algo que la etiqueta niega y la métrica lo castiga. El ruido browniano es acústicamente pobre, pero es el único fondo que se puede garantizar limpio. Es integridad del dato por sobre realismo, y dado el objetivo está bien elegido.

### El segundo beneficio, que es el importante

La anotación gratis no es la única ganancia, y a largo plazo la segunda es más valiosa: el **control sistemático sobre las variables de la mezcla**. Con datos reales, la SNR de un evento respecto del fondo, la densidad de eventos, el grado de solapamiento y el tipo de fondo son **covariables observacionales**: se pueden medir *a posteriori* (con dificultad, porque harían falta las fuentes separadas), pero no se puede **intervenir** sobre ellas. Y están confundidas entre sí: en una grabación urbana real la hora del día correlaciona simultáneamente con la densidad de eventos, con el nivel de fondo y con la distribución de clases, así que si un modelo falla más en los clips de las 8 de la mañana no se sabe si es por polifonía, por SNR o porque a esa hora hay más motos.

Scaper convierte esas covariables en **variables manipulables**, y con eso mueve el estudio de un modelo desde el diseño observacional al **diseño experimental**. El paper ejecuta la idea de la forma más limpia posible:

> "A priori esto parece complicado, dado que cada paisaje sonoro contiene múltiples eventos con valores de SNR heterogéneos. Scaper ofrece una solución sencilla: tomamos las **2000 anotaciones JAMS del conjunto de test, las editamos de modo que todos los eventos de un paisaje tengan la misma SNR, y luego re-generamos los archivos de audio a partir de las anotaciones modificadas.** Repetimos este proceso ocho veces [...] Esto resulta en **ocho versiones del conjunto de test que tienen características idénticas con la excepción de la SNR**."

Ocho conjuntos de test que son **el mismo conjunto de test** —los mismos archivos fuente, los mismos tiempos de inicio, las mismas duraciones, las mismas transformaciones, el mismo fondo— excepto por una única variable. Un experimento *ceteris paribus* perfecto sobre un dataset de audio, algo que con grabaciones reales no existe.

{{< concept-alert type="clave" >}}
La puerta de entrada a esa capacidad no es el sintetizador, sino el **formato de anotación editable y re-ejecutable**. La reproducibilidad no es aquí una buena práctica de higiene: es la *feature*. Es exactamente la diferencia entre guardar los resultados de una consulta y guardar la consulta.
{{< /concept-alert >}}

Por rigor, y porque es donde se abusa del paper: la anotación es exacta **respecto del modelo generativo**, no respecto de la percepción. Un evento sintetizado a $-3$ dB bajo un fondo denso está en la anotación durante 2.3 s, pero un oyente humano —o un micrófono real— podría no percibir nada. La etiqueta es correcta por definición y perceptualmente falsa; es la razón por la que URBAN-SED acota la SNR a $[6, 30]$ y no incluye eventos enterrados, aunque el paper no discute el punto.

## El modelo de generación probabilística

**La ontología.** La premisa perceptual que estructura el diseño es que los sonidos de un paisaje se agrupan en dos categorías: **eventos de primer plano**, salientes y reconocibles, y **sonidos de fondo**, tratados como "un único sonido holístico, más distante, ambiguo y similar a una textura". Las citas de respaldo son de psicoacústica y no de ML —Maffiolo, Guastavino, y McDermott et al. sobre estadísticas de resumen en percepción auditiva—, y son las que justifican tratar el fondo como objeto único y estacionario en vez de como una colección de eventos. De ahí la arquitectura: **un paisaje se genera como la suma de eventos de primer plano y una grabación de fondo**. La curaduría queda delegada al usuario, que organiza su *soundbank* con **una subcarpeta por clase**: `fg/<label>/*.wav`, `bg/<label>/*.wav`. El sistema de archivos *es* el esquema — la misma decisión que hace `ImageFolder` de PyTorch, que minimiza la fricción de adopción a costa de no poder expresar relaciones complejas. La consecuencia declarada es que Scaper es **agnóstico al contenido**: sirve igual para paisajes urbanos, bioacústica, casas inteligentes o vigilancia.

**La event specification** almacena todas las propiedades de un evento que Scaper puede controlar:

| Propiedad | Qué controla |
|---|---|
| `label` | La clase del evento; determina la subcarpeta del soundbank |
| `source_file` | El clip específico, entre los que corresponden a la etiqueta |
| `source_time` | Dónde empieza el recorte **dentro del clip fuente** |
| `event_time` | Cuándo empieza el evento **en el paisaje generado**; es el onset de la anotación |
| `event_duration` | Cuánto dura; con el anterior determina el offset |
| `snr` | Nivel respecto del fondo, medido en LUFS |
| `role` | `foreground` o `background` |
| `pitch_shift` | En semitonos. **No afecta la duración** |
| `time_stretch` | Como factor de la duración. **No afecta el pitch** |

Las dos últimas aclaraciones son técnicamente importantes: significan que Scaper usa transformaciones **desacopladas** (vía SoX, a través de `pysox`), no un simple *resampling*. Un cambio de velocidad de reproducción altera pitch y duración de forma ligada; SoX implementa *pitch shift* preservando duración y *stretch* preservando pitch mediante procesamiento de fase. Las dos perillas quedan **ortogonales**, y para un diseño experimental esa ortogonalidad es todo.

**La distribution tuple** es la abstracción que hace todo el trabajo: "para cada propiedad en una *event specification* el usuario provee una **distribution tuple**, que define una distribución de la cual muestrear el valor de la propiedad".

| Tupla | Semántica | Ejemplo |
|---|---|---|
| `('const', v)` | Valor constante | `('const', 0)` para `source_time` |
| `('choose', [a, b, c])` | Uniforme sobre una lista discreta; lista vacía = todo lo disponible | `('choose', [])` para `source_file` |
| `('uniform', min, max)` | $\mathcal{U}(\text{min}, \text{max})$ continua | `('uniform', 0.5, 4.0)` para `event_duration` |
| `('normal', mu, sigma)` | $\mathcal{N}(\mu, \sigma)$ | `('normal', 5, 2)` para `event_time` |
| `('truncnorm', mu, sigma, min, max)` | Normal truncada al intervalo | `('truncnorm', 3, 1, 0.5, 5)` |

El paper añade que "se pueden agregar fácilmente distribuciones adicionales" — promesa que la librería cumplió: la versión actual incluye `choose_weighted`, que muestrea con probabilidades no uniformes y permite construir soundbanks con prior de clase realista. El rango de expresividad que esto abre es un **continuo**: con todo `const` la especificación es una receta determinista y Scaper se comporta como un renderizador de mezclas; con todo distribucional es un **generador de una familia de paisajes**, del que "el usuario puede generar infinitas instanciaciones".

### Por qué la indirección entre especificación e instancia es la decisión correcta

Scaper mantiene **dos objetos distintos**: la **especificación probabilística** (una descripción declarativa de una *familia* de paisajes, con distribuciones en vez de valores) y la **especificación instanciada** (la misma estructura con valores concretos ya muestreados, que el paper llama literalmente una "receta"). El audio es el tercer objeto, derivado del segundo. La separación produce cinco propiedades que un generador monolítico no tiene:

- **Una descripción compacta genera un dataset arbitrariamente grande.** La especificación de URBAN-SED cabe en unas veinte líneas de Python; el dataset pesa casi 30 horas. La descripción es lo que se versiona y se comparte; el audio es caché.
- **La instancia es inspeccionable y editable antes de renderizar.** Es lo que hace posible el barrido de SNR: se editan 2000 instancias en **un campo** y se re-renderiza. Sin la capa intermedia habría que re-muestrear, y al re-muestrear cambiarían también todas las demás variables — el experimento controlado se perdería.
- **El muestreo se separa del rendering.** Uno es barato y determinista dado un `random_state`; el otro es caro (SoX, I/O de disco). El API lo expone con el flag `no_audio=True`.
- **La reproducibilidad es exacta y no probabilística.** Guardar la semilla reproduce *si* la versión de la librería, el orden de las llamadas y el soundbank no cambian; guardar la **instancia** reproduce aunque el generador cambie, porque ya no hay nada que muestrear.
- **La misma abstracción cubre generación y aumentación.** Agregar un evento a un paisaje existente es agregar una especificación a una instancia y re-renderizar; no hace falta un segundo camino de código.

Para quien viene de sistemas de datos, esto es la distinción entre DDL y datos, o entre un *query plan* y su *result set*, o —el paralelo más cercano— entre un **generador de fixtures parametrizado** y los fixtures concretos que produce. Lo interesante es que en ML esa lección todavía no está internalizada: la enorme mayoría de los datasets se distribuye como bolsas de archivos sin la receta que los produjo, y por eso son inmodificables, no auditables y no extensibles.

El pipeline completo tiene cinco etapas: `SELECT PARAMETERS` (muestrear de las distribuciones), `TRIM` (recortar cada fuente), `TRANSFORM` (pitch y stretch **por evento**), `NORMALIZE` (ajustar niveles según la SNR) y `COMBINE` (sumar en la línea de tiempo). Y **cada salida es un par (audio, anotación)**, nunca audio suelto. Un sintetizador de audio es un generador; un generador de datasets es un generador **más un anotador acoplado**. La diferencia es de plomería, y la plomería es la contribución.

## Reproducibilidad y el archivo JAMS

Scaper emite la anotación en dos formatos. El primero es **texto plano tabular** con tres columnas —onset, offset, etiqueta— justificado explícitamente porque "puede cargarse directamente en software como Audacity para ver las etiquetas junto con el archivo de audio": que un investigador arrastre el `.txt` sobre la forma de onda y **vea** las anotaciones alineadas en tres segundos convierte la verificación visual en un gesto y no en un script. El segundo es **JAMS**, el formato serio.

**JAMS** = *JSON Annotated Music Specification* (Humphrey, Salamon, Nieto, Forsyth, Bittner y Bello, ISMIR 2014). Nótese la superposición de autores: Salamon y Bello firman ambos papers, así que Scaper no adopta un formato externo sino **su propio formato previo**, lo que explica lo bien que encajan. Es un contenedor JSON para anotaciones de audio con tres propiedades relevantes: admite **múltiples anotaciones sobre el mismo audio coexistiendo**, cada una con su `namespace`, su `annotation_metadata` (quién o qué la produjo, con qué herramienta y versión) y sus observaciones $(\text{time}, \text{duration}, \text{value}, \text{confidence})$ — que el modelo de datos permita **varios anotadores discrepando sobre el mismo archivo** es exactamente lo que se necesita en un dominio donde el desacuerdo es estructural; soporta **metadata estructurada e ilimitada**; y es **JSON**, o sea legible, versionable en git, diffable y parseable en cualquier lenguaje.

Lo que Scaper mete adentro es la clave:

> "Scaper explota esto para almacenar **tanto la especificación probabilística como la instanciada** de cada evento sonoro. Esto significa que (asumiendo que se tiene acceso al *soundbank* original) **Scaper puede reconstruir completamente el audio de un paisaje sonoro a partir de su anotación JAMS**."

Es decir, tres niveles en un solo archivo: la especificación probabilística (para saber de qué familia salió el ejemplo y generar hermanos distintos), la instanciada (para regenerar *este* ejemplo exacto o editarlo) y la anotación $(\text{onset}, \text{offset}, \text{label})$ (para entrenar y evaluar). A eso se suma una covariable derivada que el paper computa y persiste explícitamente: "la polifonía máxima es **calculada automáticamente por Scaper durante la generación y almacenada en la anotación JAMS**. Esto nos permitirá evaluar fácilmente el desempeño del modelo como función de la polifonía máxima". Ese es el patrón completo: **precomputar en el momento de generación las covariables que después se van a querer para estratificar el análisis, y persistirlas junto al dato**.

### Por qué esto importa fuera del audio

El estado habitual de un dataset sintético en la literatura es: alguien escribe un script *ad hoc*, lo corre, publica los archivos resultantes, y el script se pierde. El propio paper lo señala con nombre y apellido al comparar con TUT-SED synthetic 2016: "hasta donde sabemos, **fue generado usando scripts ad-hoc que no están disponibles públicamente**". Ese dataset es perfectamente utilizable para entrenar y evaluar; lo que no se puede hacer con él es **preguntarle nada**.

| Nivel | Qué se publica | Qué se puede hacer |
|---|---|---|
| 0 | Solo el audio | Entrenar y evaluar. Nada más |
| 1 | Audio + anotaciones | Lo anterior, y verificar métricas |
| 2 | + el script generador | Regenerar si sobrevive el entorno y el soundbank. Frágil |
| 3 | + la **instancia serializada por ejemplo** (JAMS) | Regenerar exactamente, **editar una variable y re-renderizar**, auditar la distribución real |
| 4 | + la **especificación probabilística** | Todo lo anterior, y **generar hermanos nuevos** de la misma familia |

Scaper opera en el nivel 4, y URBAN-SED se publica con el audio, las anotaciones, los scripts de generación **y** los de los experimentos de ML. El barrido de SNR demuestra que el nivel 3 no es cosmético: **habilita una clase de experimento que sin él no existe**. Traducido a otro vocabulario, la anotación JAMS es a un paisaje sonoro lo que una **migración versionada** es a un esquema de base de datos, o lo que un **manifiesto de build reproducible** es a un binario: el artefacto derivado deja de ser la fuente de verdad, lo es la receta, y por lo tanto **se puede tirar el artefacto y reconstruirlo** — lo que convierte 30 horas de WAV en caché en vez de en patrimonio.

La única dependencia que rompe el cierre es la que el propio paper señala entre paréntesis: *"assuming one has access to the original soundbank"*. El JAMS referencia archivos fuente **por ruta**, no por hash de contenido; si el soundbank cambia o desaparece, la reconstrucción falla. Es la misma fragilidad que un `package-lock.json` sin *registry*, y la solución canónica —fijar el soundbank con un identificador de contenido— no está ni en el paper ni en la librería.

## Los detalles de la mezcla

### La SNR: por qué LUFS y no picos

Este es el detalle técnico más cuidado del paper y el que más se cita mal:

> "La **simple normalización por pico no garantiza que dos sonidos normalizados al mismo nivel se perciban como igualmente sonoros**. Para sortear esto, Scaper usa **Loudness Units relative to Full Scale (LUFS)**, una medida estándar de sonoridad percibida usada en radio, televisión y transmisión por Internet. Así, si un evento se especifica con una SNR de 6, significa que estará **6 LUFS por encima del nivel del fondo**."

La normalización por pico fija $\max_t |x(t)|$: es trivial de computar y es lo que hace la mayoría del código casero, pero el pico no tiene relación estable con la sonoridad percibida, y la desconexión es brutal justamente para las clases de UrbanSound8K. Un `gun_shot` es un impulso —pico altísimo, energía total baja, duración de decenas de milisegundos— y normalizado por pico suena **más débil** de lo que su número sugiere; un `air_conditioner` es casi estacionario, con factor de cresta bajo y energía sostenida, y normalizado por pico suena **mucho más fuerte**. La diferencia de sonoridad percibida al mismo pico puede ser de 15–20 dB, así que una "SNR" definida sobre picos no es comparable entre clases y **no es una variable experimental utilizable**: al barrerla se estaría barriendo simultáneamente la composición de clases.

LUFS (recomendación ITU-R BS.1770) lo resuelve con tres piezas: un **filtro de ponderación K** que aproxima la respuesta del sistema auditivo y el efecto de la cabeza, medición de **energía media sobre bloques solapados de 400 ms** en vez del pico, y un **gating** que descarta los bloques silenciosos. El resultado es una escala logarítmica donde una diferencia de 6 LUFS entre dos señales cualesquiera significa aproximadamente lo mismo perceptualmente, sin importar su contenido espectral o temporal. **Esa invarianza es lo que convierte la SNR en una variable manipulable de verdad.**

En URBAN-SED el fondo se normaliza a $-50$ LUFS y las SNR se muestrean en $[6, 30]$, o sea los eventos viven entre $-44$ y $-20$ LUFS. De $\text{SNR}_i = L_{e_i} - L_b$, la ganancia lineal es $g_i = 10^{\,(L_b + \text{SNR}_i - L_{e_i}^{\text{orig}})/20}$ con $L_{e_i}^{\text{orig}}$ la sonoridad medida del evento ya recortado y transformado. El paper no escribe esta fórmula; es la lectura directa de lo que describe en prosa.

### Polifonía y solapamiento

Scaper **no modela el solapamiento como una restricción**: lo deja emerger del muestreo de los tiempos de inicio, con suma aditiva y sin tratamiento especial. La métrica que lo resume es la **polifonía máxima**, "la mayor polifonía observada en cualquier punto del tiempo del paisaje", es decir $\max_t |\{i : \tau_i \le t < \tau_i + d_i\}|$, contando también eventos de la misma clase. Y la forma en que URBAN-SED induce variedad es indirecta y elegante: en vez de controlar el solapamiento, controla la **distribución de los tiempos de inicio**.

| Distribución de `event_time` | Forma | Efecto sobre la polifonía |
|---|---|---|
| $\mathcal{U}(0, 10)$ | Uniforme sobre los 10 s | Eventos dispersos, poco solapamiento |
| $\mathcal{N}(5, 2)$ | Unimodal centrada | Aglomeración en el medio, más solapamiento |
| $\tfrac{1}{2}\mathcal{N}(3,2) + \tfrac{1}{2}\mathcal{N}(7,2)$ | Bimodal | Dos racimos |

Con el número de eventos muestreado de una uniforme discreta en $[1, 9]$, la polifonía máxima observada abarca **de 1 a 7**. El paper reconoce la limitación: "actualmente Scaper **no soporta controlar explícitamente ciertas características de la escena** como la polifonía máxima (o promedio)". La polifonía es una **variable de salida**, no de entrada: se mide *a posteriori* (por eso se guarda en el JAMS), pero no se puede pedir "genérame 500 paisajes con polifonía máxima exactamente 4" salvo generando de más y filtrando — rechazo por muestreo, funcional pero ineficiente y con la distribución condicional sesgada de forma difícil de razonar.

### Las transformaciones por evento

`pitch_shift` y `time_stretch` se aplican **a cada evento individualmente**, y esa individualidad es una de las tres diferencias declaradas frente al antecedente de Lafay et al. (2016), "aumentando significativamente el rango posible y la variabilidad de los paisajes generados". El argumento combinatorio es directo: con $M$ clips y transformaciones globales hay $M \times K$ variantes de material; aplicadas por evento y componiendo $N$ eventos por escena, el espacio crece como $(MK)^N$ por las combinaciones de tiempos, duraciones y niveles — con la salvedad importante de que **la diversidad combinatoria no es diversidad acústica**. Hay además un efecto que el paper no comenta: `time_stretch` altera la duración y por lo tanto **el offset de la anotación**, que debe reflejar la duración *post-stretch*. Scaper lo hace correctamente, pero es exactamente el tipo de bug silencioso que un script *ad hoc* introduce sin que nadie lo note: un desfase sistemático entre etiqueta y audio que degrada el entrenamiento sin producir ningún error visible.

### Normalización y clipping: lo que el paper deja abierto

Hay que ser honesto sobre el alcance del texto. El paper **sí especifica** que existe una etapa `NORMALIZE` antes de `COMBINE` y que la normalización de niveles se hace en LUFS respecto del fondo. **No especifica** qué ocurre cuando la suma excede el rango $[-1, 1]$. El riesgo es aritméticamente inevitable: con nueve eventos simultáneos a SNR alta la suma puede saturar, y el *clipping* no es cosmético — introduce distorsión armónica de banda ancha que es **exactamente el tipo de artefacto que una CNN puede aprender a usar como atajo**: si los paisajes de polifonía alta saturan más, el modelo puede inferir la polifonía desde la distorsión en vez de desde el contenido. La mitigación de facto en URBAN-SED es el generoso *headroom*: sumar nueve señales incoherentes a $-20$ LUFS da del orden de $-20 + 10\log_{10}(9) \approx -10.5$ LUFS, todavía bajo el techo.

{{< concept-alert type="advertencia" >}}
La librería **sí** resolvió el problema después: el `generate()` actual expone `fix_clipping` y `peak_normalization` como parámetros explícitos. Ninguno aparece en el paper de 2017 — son agregados posteriores, presumiblemente motivados justo por este problema. Es evolución de la librería más allá del paper, no algo que el paper diga.
{{< /concept-alert >}}

El texto tampoco especifica **fundidos en los bordes de los eventos** (un recorte abrupto produce un clic, y si no hay ventanas de fundido cada evento trae un transitorio artificial en su onset y su offset — otro atajo perfecto, porque el clic marca exactamente el límite que se quiere predecir), ni la **frecuencia de muestreo** y política de resampleo, ni el manejo de **canales** (mono/estéreo, panorámica), ni la **reverberación**, que en 2017 no se menciona en absoluto.

## Los experimentos

{{< concept-alert type="advertencia" >}}
**Verificabilidad de las cifras.** El paper **no publica ninguna tabla numérica** en sus cinco páginas. Todas las cifras de desempeño que siguen son **lecturas aproximadas de gráficos de barras** (Figuras 2 y 3), y no pueden darse con más precisión que la que permite el eje. La única afirmación cuantitativa escrita en prosa es cualitativa: "los dos modelos rinden de forma comparable". Cualquier documento que cite "la F-measure fue de 0.573" para URBAN-SED está reportando una precisión que el paper nunca publicó.
{{< /concept-alert >}}

### URBAN-SED

Se usa [UrbanSound8K](/papers/urbansound8k-salamon-2014) como soundbank (~1000 clips por cada una de 10 fuentes urbanas, un solo evento por clip), respetando sus **10 folds** oficiales: folds 1–6 → 6000 paisajes de entrenamiento, 7–8 → 2000 de validación, 9–10 → 2000 de test. Cada paisaje dura 10 s sobre ruido browniano a $-50$ LUFS, con número de eventos $\sim\mathcal{U}\{1,9\}$, `source_time` siempre 0 ("para asegurar que no perdamos el onset de un evento"), `event_time` de una de las tres distribuciones de la tabla anterior, `event_duration` $\sim\mathcal{U}(0.5, 4.0)$ s, `snr` $\sim\mathcal{U}(6, 30)$, `pitch_shift` $\sim\mathcal{U}(-3, 3)$ semitonos y `time_stretch` $\sim\mathcal{U}(0.8, 1.2)$.

Resultado: **10 000 paisajes**, **casi 30 horas**, **~50 000 eventos anotados**, polifonías máximas de 1 a 7. En su momento, "el dataset con etiquetas fuertes más grande disponible para SED, **aunque por supuesto podríamos hacerlo arbitrariamente más grande o más pequeño**". La coletilla es la tesis del paper en cuatro palabras: el tamaño dejó de ser una propiedad del dataset para ser un **parámetro**. La comparación de escala es contundente: 30 horas frente a las 9.5 h de TUT-SED synthetic 2016 y a los 78 minutos de TUT Sound Events 2016 real — **más de 20×** el dataset real más usado de entonces, con anotaciones exactas en vez de aproximadas.

### Los dos modelos, y el control que los hace comparables

Se comparan un **CRNN** (Çakir et al., 743k parámetros, sin modificaciones; los autores verificaron su implementación reproduciendo los resultados publicados sobre TUT-SED-2016) y una **CNN** adaptada de Salamon y Bello 2017 (softmax → sigmoides para pasar de multi-clase a multi-etiqueta, 64 filtros por capa para igualar capacidad, *batch normalization*, entrada reducida a 1 s, pooling a (2,2); de 241k a **720k** parámetros). El CRNN opera a nivel de frame, la CNN a 1 s. Entrenamiento con Keras, Adam, entropía cruzada binaria, 300 épocas con parada temprana a 100 sin mejora, evaluando F-measure por segmentos de 1 s con `sed_eval`.

La resolución de 1 s se justifica de forma pragmática ("sería suficiente para aplicaciones de monitoreo urbano [...] y resulta en un modelo significativamente más rápido de entrenar"), pero el control está en una nota al pie que es el tipo de verificación que debería ser obligatoria y casi nunca se hace: evaluar el CRNN a 100 ms y 20 ms **no produjo F-measures más altas** que a 1 s. El *handicap* aparente no lo es; sin esa nota, toda la comparación sería impugnable.

### Resultado 1: el empate global esconde dos perfiles de error

| Clase | CRNN (≈) | CNN (≈) | | Clase | CRNN (≈) | CNN (≈) |
|---|---|---|---|---|---|---|
| `air_conditioner` | 0.47 | 0.34 | | `engine_idling` | 0.59 | 0.51 |
| `car_horn` | 0.52 | 0.69 | | `gun_shot` | 0.69 | 0.50 |
| `children_playing` | 0.54 | 0.49 | | `jackhammer` | 0.58 | 0.75 |
| `dog_bark` | 0.57 | 0.52 | | `siren` | 0.63 | 0.64 |
| `drilling` | 0.56 | 0.58 | | `street_music` | 0.56 | 0.59 |
| | | | | **Global** | **≈0.57** | **≈0.57** |

El paper concluye que los dos modelos rinden de forma comparable, con el CRNN notablemente mejor en aire acondicionado y disparos y la CNN mejor en bocinas y martillos neumáticos. Hay un patrón que no comenta: el CRNN gana en `gun_shot` (impulsivo, brevísimo) y en `air_conditioner` (estacionario largo y difuso), **los dos extremos de la escala temporal** — uno porque un evento de 200 ms cae dentro de un único segmento de 1 s y se diluye, el otro porque exige integrar evidencia sobre varios segundos. La CNN gana donde la textura espectral es fuerte y cabe en una ventana de 1 s.

### Resultado 2: degradación con la polifonía

| Polifonía máxima | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Precision (≈) | 0.68 | 0.67 | 0.70 | 0.70 | 0.69 | 0.71 | **0.78** |
| Recall (≈) | 0.64 | 0.54 | 0.50 | 0.46 | 0.42 | 0.42 | **0.41** |
| F-measure (≈) | 0.66 | 0.60 | 0.58 | 0.56 | 0.52 | 0.52 | 0.54 |

> "Como uno podría esperar, notamos que la F-measure declina gradualmente a medida que aumenta la polifonía máxima, pero **más interesante todavía**, vemos que es porque **el recall declina, mientras que la precisión permanece estable (e incluso sube)**. Esto sugiere que a medida que más eventos sonoros se solapan, el modelo es cada vez más propenso a **detectar solo un subconjunto** de los eventos, sin embargo permanece igualmente preciso."

Que la F-measure baje con la polifonía es trivialmente esperable; **la descomposición no lo es**. La caída podía venir de dos mecanismos con implicancias opuestas: **confusión** (el modelo alucina clases que no están porque los espectros se superponen → cae la *precisión*) o **enmascaramiento** (el modelo pierde eventos porque unos tapan a otros, pero lo que detecta lo detecta bien → cae el *recall*). Los datos apoyan inequívocamente la segunda, y con un refinamiento: la precisión no solo se mantiene, **sube** hacia el extremo, coherente con un modelo que se vuelve conservador bajo incertidumbre.

Las consecuencias de ingeniería son concretas y ninguna se derivaría de una métrica agregada. Si la aplicación es monitoreo de ruido y se necesita cobertura, el modo de falla dominante es la **omisión** y bajar el umbral es la palanca correcta (el margen de precisión sugiere que hay espacio). Si es vigilancia y no tolera falsas alarmas, el modelo ya opera en el régimen deseado. Y el problema a atacar arquitectónicamente no es la discriminación entre clases sino la **separación de fuentes solapadas**. Hay que insistir además en que **el eje de este gráfico solo existe porque la polifonía se conoce con exactitud**: con datos reales habría que estimarla desde una anotación humana que, precisamente en escenas densas, es la menos confiable — el instrumento de medición se degradaría junto con lo medido.

### Resultado 3: el barrido de SNR

Ocho versiones del conjunto de test, idénticas salvo por la SNR, obtenidas editando los JAMS y re-renderizando.

| Rango de SNR | 6–9 | 9–12 | 12–15 | 15–18 | 18–21 | 21–24 | 24–27 | 27–30 |
|---|---|---|---|---|---|---|---|---|
| Precision (≈) | **0.77** | 0.76 | 0.74 | 0.72 | 0.70 | 0.68 | 0.66 | 0.63 |
| Recall (≈) | 0.43 | 0.46 | 0.48 | 0.50 | 0.51 | 0.52 | 0.53 | **0.54** |
| F-measure (≈) | 0.55 | 0.57 | 0.58 | 0.59 | 0.59 | 0.59 | 0.59 | 0.58 |

> "A medida que la SNR aumenta, la precisión y el recall exhiben **comportamientos opuestos**: el recall sube [...] pero esto lleva a un número aumentado de falsos positivos, bajando la precisión. El efecto neto es una **F-measure relativamente estable**, pero a través de este experimento **ahora sabemos que esa estabilidad esconde un comportamiento del modelo bastante distinto** a medida que la SNR cambia."

Esa última frase es la moraleja del paper entero: **una métrica plana no es evidencia de comportamiento estable**. La F-measure se mueve entre 0.55 y 0.59 sobre un rango de 24 dB — visualmente, una línea horizontal. Debajo, la precisión cae ~14 puntos y el recall sube ~11: son **dos modelos distintos** en los extremos del rango, y el escalar agregado no lo registra. Para quien diseña protocolos de evaluación, es la demostración empírica de por qué el **desglose estratificado por covariable controlada** debería ser el estándar, y por qué la síntesis con receta es la única forma barata de conseguir esas covariables.

### El cuarto experimento: Scaper como instrumento para medir humanos

El caso de uso más elegante y el que menos se cita. El paper resume resultados de un estudio propio de *crowdsourcing* (Cartwright et al., *"Seeing sound"*, PACM HCI 2017) cuyo planteo es circular: "dado que el objetivo del experimento era **evaluar la calidad de las etiquetas humanas, no es posible usar paisajes sonoros etiquetados por humanos como estímulos**". Para medir cuán bien anota un humano hace falta una referencia; si la referencia la produjo otro humano, se está midiendo **acuerdo entre anotadores**, no exactitud. Scaper rompe la circularidad porque **proporciona una referencia que no es una opinión**. La escala del estudio es en sí misma una demostración del argumento: 90 eventos curados → 3000 paisajes generados → 60 seleccionados para cubrir un rango de complejidades.

El hallazgo es que los anotadores humanos exhiben **el mismo comportamiento que los modelos**: al subir la polifonía cae la F-measure, principalmente por caída del recall, mientras la precisión permanece alta. Conclusión: "las etiquetas humanas para paisajes sonoros densos pueden considerarse **confiables, aunque incompletas**". El diagnóstico cambia la estrategia de anotación — si el error humano fuera de *precisión* el dataset estaría contaminado y no habría cómo arreglarlo salvo re-anotando; siendo de *recall*, el dataset es correcto pero incompleto, lo cual es tratable con supervisión parcial, *positive-unlabeled learning*, o unión de anotaciones de varios anotadores independientes. El segundo hallazgo es de HCI y tiene consecuencia directa: comparando forma de onda, espectrograma y ausencia de visualización, **el espectrograma produjo una mejora estadísticamente significativa en la exactitud temporal** de las anotaciones.

Este uso de Scaper es el más subestimado: no como generador de datos de entrenamiento, sino como **instrumento de calibración del proceso de anotación humana**. Si se van a anotar 500 horas de audio real, gastar antes una semana en generar estímulos sintéticos, medir a los anotadores contra ellos y ajustar protocolo e interfaz tiene un retorno evidente. Es control de calidad con patrón de referencia, como se calibra cualquier instrumento físico.

## La brecha entre lo sintético y lo real

El paper es explícito y no defensivo: "los paisajes sonoros generados, **incluso si suenan bastante realistas en algunos casos, no pueden abarcar la riqueza y complejidad de los paisajes sonoros reales**. [...] Scaper **no puede usarse como reemplazo de grabaciones del mundo real anotadas manualmente**, si queremos estimar cuán bien rendirá un modelo en un ambiente real". "No abarca la riqueza y complejidad" es honesto pero vago; vale descomponer **exactamente qué falta**, porque cada componente tiene una mitigación distinta.

### La ausencia de acústica del espacio

El modelo generativo de Scaper es una **suma de señales**, $x(t) = b(t) + \sum_i g_i\, e_i(t - \tau_i)$. Lo que un micrófono captura en un espacio real es

$$x(t) = \sum_i \big(h_i * s_i\big)(t - \tau_i) + n(t)$$

donde $h_i$ es la **respuesta al impulso** (RIR) del camino acústico entre la fuente $i$ y el micrófono. Esa $h_i$ codifica el **sonido directo** (atenuado por $1/r_i$ y retardado por $r_i/c$), las **reflexiones tempranas** de los primeros ~50–80 ms —el patrón discreto de ecos de fachadas, suelo y mobiliario, principal pista perceptual de dónde está la fuente y qué tamaño tiene el espacio— y la **cola reverberante**, caracterizada por el $T_{60}$, que en una calle urbana con fachadas puede ir de 0.5 a 2 s. Tres consecuencias concretas de que $h_i$ no exista:

1. **Los offsets sintéticos son abruptos; los reales no.** Un evento real no termina cuando la fuente deja de emitir, sino cuando su cola reverberante decae bajo el fondo. Un detector entrenado con offsets limpios aprende una noción de "fin de evento" que en audio real no ocurre nunca — y como la métrica de SED evalúa offsets, es un desajuste directo entre objetivo de entrenamiento y tarea real.
2. **La reverberación es el "pegamento" perceptual de una escena.** Cuando todas las fuentes comparten la misma $h$, el oyente las integra en una escena coherente; cuando cada fuente trae la reverberación de un espacio distinto —el caso de un soundbank compilado de grabaciones heterogéneas— la mezcla suena a *collage*, y eso no es un juicio estético: es información espectral y temporal explotable por una red.
3. **La reverberación degrada la separabilidad, y esa degradación es la tarea.** Al alargar cada fuente y difuminar sus transitorios, aumenta el solapamiento efectivo mucho más allá del solapamiento nominal de los onsets. Un modelo entrenado sobre mezclas anecoicas nunca vio ese régimen.

La librería actual tiene un parámetro `reverb` que aplica el efecto de SoX: una reverberación **algorítmica y global** sobre la mezcla, no una convolución por fuente con una RIR medida. Ayuda con (1) y algo con (2), pero no reproduce la relación entre reverberación y **posición de la fuente**, que es lo que da coherencia espacial.

### La respuesta del micrófono y la cadena de captura

Un paisaje real pasa por una cadena física que Scaper no modela: respuesta en frecuencia y directividad del transductor, ganancia y ruido del preamplificador, AGC, códec con pérdida, cuantización. Importa por una razón concreta: **en despliegue real, esa cadena es fija**. Un sensor de la red de monitoreo urbano de NYU tiene un micrófono MEMS específico, con una coloración específica, en una carcasa específica, a una altura específica, y todo el audio de producción comparte esa firma. El audio sintético hereda en cambio la firma **heterogénea de Freesound**, unión de cientos de cadenas de captura distintas: el modelo aprende a ser invariante a un conjunto de coloraciones que **no incluye** la del sensor de despliegue. Es *domain shift* de manual, de la variedad *covariate shift*. Un detalle corta en dirección contraria y vale reconocerlo: **la diversidad de fuentes heterogéneas también actúa como regularizador**, porque un modelo forzado a reconocer un `car_horn` grabado con veinte micrófonos distintos aprende una representación menos dependiente del transductor. Cuál de los dos efectos domina es empírico.

### Los eventos aislados ya vienen con su propia acústica

Este es el punto más sutil y el más dañino. Los clips de UrbanSound8K provienen de Freesound: grabaciones de campo hechas por personas distintas, con equipos distintos, en lugares distintos. Cuando Scaper los recorta y los suma, **cada evento trae adherida la acústica de su grabación original** —ruido de fondo residual, reverberación, coloración, piso de ruido—, así que un paisaje sintético de nueve eventos es una superposición de **nueve ambientes acústicos distintos** más un ruido browniano.

Y eso deja una firma detectable. Un evento recortado de una grabación de campo casi nunca es puro: en los milisegundos previos a su onset y posteriores a su offset hay ruido de fondo *de su grabación original*, distinto del browniano. Al insertarlo, en el instante $\tau_i$ el nivel de piso de ruido y su color espectral **cambian abruptamente**, y vuelven a cambiar en $\tau_i + d_i$. Una CNN sobre espectrograma puede aprender a detectar **exactamente esa discontinuidad de fondo** —un salto en las bandas donde no hay energía del evento— y usarla como predictor del onset. Y funcionaría perfecto, con F-measure alta y estable, sobre datos sintéticos. Y colapsaría por completo sobre audio real, donde el fondo es continuo.

{{< concept-alert type="advertencia" >}}
Este es el modo de falla más peligroso del entrenamiento sintético, porque **es invisible en validación**: el conjunto de validación tiene el mismo artefacto que el de entrenamiento, así que la métrica no lo detecta, y solo aparece al evaluar sobre datos reales. Es la versión auditiva del clásico de visión donde el clasificador de tanques había aprendido a detectar el clima. Súmale posibles clics en los bordes de recorte, artefactos de fase del *pitch shift* y el *time stretch*, y la eventual distorsión por *clipping*: cada uno es un canal lateral.
{{< /concept-alert >}}

### La independencia entre eventos no refleja el mundo

En URBAN-SED cada evento se muestrea de forma **independiente e idénticamente distribuida**, y nada de eso es cierto en el mundo:

- **Co-ocurrencia entre clases.** `jackhammer` y `drilling` van juntos: son la misma faena. `siren` implica tráfico, luego `engine_idling` y `car_horn`. `children_playing` ocurre en plazas con `dog_bark` y **casi nunca** con `jackhammer`. La matriz de co-ocurrencia real es fuertemente estructurada; la sintética es producto de marginales uniformes.
- **Correlación temporal.** Los eventos reales tienen estructura secuencial y causal: un auto frena, toca la bocina, acelera; una sirena se acerca y se aleja con Doppler y curva de nivel; un martillo neumático opera en ráfagas periódicas. Scaper tira tiempos de inicio de una distribución sin memoria: no hay *proceso*, solo *puntos*.
- **Correlación entre etiqueta y parámetros.** Una sirena de emergencia es intrínsecamente más fuerte que un ladrido; en URBAN-SED comparten la misma $\mathcal{U}(6,30)$.
- **Prior de clase.** Las 10 clases son equiprobables. En una calle real `engine_idling` y `air_conditioner` están casi permanentemente presentes y `gun_shot` es rarísimo; el desbalance real es de órdenes de magnitud.

Por qué esto degrada el desempeño real: **un modelo aprende y explota las correlaciones de sus datos de entrenamiento aunque nadie se lo pida**. Si en el entrenamiento las clases son independientes, el modelo trata cada clase por separado y **no aprovecha el contexto**; sobre datos reales, donde el contexto es informativo, deja información sobre la mesa. Es simétrico al problema de las etiquetas: la síntesis da control perfecto sobre las marginales y a cambio obliga a especificar la estructura conjunta, que es justamente lo que nadie sabe especificar bien. El `choose_weighted` actual permite al menos priors de clase realistas; las correlaciones condicionales requerirían muestrear la especificación desde un modelo de escena (una cadena de Markov, un modelo gráfico de co-ocurrencia estimado de datos reales), lo cual el API permite pero no facilita.

### Por qué el fine-tuning mitiga el problema

Lo que aprende un detector de eventos sonoros se descompone en dos partes: **un banco de representaciones acústicas** (qué patrones tiempo-frecuencia corresponden a un ladrido, a una sirena, a un martillo: conocimiento sobre las **fuentes**) y **un modelo de la condición de observación** (cómo se ve una fuente reverberada, atenuada, mezclada, capturada por *este* micrófono: conocimiento sobre el **canal**). El dato sintético es **excelente para lo primero** —las fuentes de UrbanSound8K son grabaciones reales de bocinas reales, el conocimiento de fuente es genuino— y **sistemáticamente equivocado para lo segundo**.

El fine-tuning con datos reales es entonces una operación quirúrgica: se preservan las capas bajas y medias que codifican las representaciones acústicas —que costaron 30 horas sintéticas y ninguna anotación humana— y se reajustan las capas altas y la calibración de salida para el canal, que es lo que las pocas horas de datos reales anotados pueden enseñar. Es la misma economía de transferencia que hace funcionar ImageNet → tarea específica, y explica por qué **la proporción importa**: para reajustar el canal no hacen falta 30 horas reales, sino las suficientes para cubrir la variabilidad del canal de despliegue, que si el sensor es fijo es mucho menor que la variabilidad de las fuentes.

### Las otras estrategias

| Estrategia | Qué hace | Costo | Qué no arregla |
|---|---|---|---|
| **Fine-tuning sobre datos reales** | Reajusta el modelo con un conjunto pequeño anotado del dominio destino | Poca anotación fuerte real | Nada, si el conjunto real es demasiado pequeño o poco diverso |
| **Convolución con RIRs medidas** | Convolucionar cada fuente con una respuesta al impulso real antes de sumar | Un banco de RIRs (BUT ReverbDB, ACE Challenge, MIT IR Survey) | La coloración del micrófono, la co-ocurrencia |
| **Simulación de sala** (image-source, ray tracing) | Generar RIRs sintéticas para geometrías y posiciones arbitrarias (`pyroomacoustics`, `gpuRIR`) | Cómputo; modelos de absorción simplificados | Escenas exteriores urbanas, difíciles de modelar como salas |
| **Aumentación de canal** | Filtros aleatorios, códecs, ruido, AGC simulado sobre la mezcla | Barato | La estructura de la escena |
| **Adaptación de dominio no supervisada** | Alinear distribuciones de features sintético/real sin etiquetas reales (DANN, CORAL, MMD, *mean teacher*) | Audio real sin anotar, que sí abunda | Puede alinear features sin alinear la tarea |
| **Randomización de dominio** | Aleatorizar los parámetros no realistas para que el dominio real caiga dentro de la envolvente sintética | Que el simulador exponga esos parámetros | Los sesgos estructurales que no son parámetros |

La convergencia práctica en SED, visible en la evolución de DCASE, fue la combinación: **entrenar con sintético + real débilmente etiquetado + real sin etiquetar**, con un esquema de consistencia tipo *mean teacher*. La comunidad no resolvió la brecha eliminando lo sintético, sino **poniéndolo en su lugar dentro de un régimen mixto**.

Un contrapunto necesario, porque el discurso "lo sintético no sirve" es tan equivocado como el opuesto: la brecha es un problema **cuando la cifra sintética se usa como estimador del desempeño en producción**, y no lo es cuando se usa para comparar modelos entre sí bajo condiciones idénticas, caracterizar el comportamiento de un modelo como función de una variable, depurar un pipeline (si el modelo falla donde la anotación es exacta por construcción, el problema es del modelo), pre-entrenar representaciones transferibles, o calibrar anotadores humanos. El propio paper traza la línea con el condicional exacto: no es reemplazo de grabaciones reales **"si queremos estimar cuán bien rendirá un modelo en un ambiente real"**.

## Scaper como herramienta viva

**URBAN-SED, el caso canónico.** Es el dataset que el propio paper construye para demostrar la herramienta: 10 000 paisajes urbanos sintéticos de 10 s con anotación fuerte exacta, compuestos a partir de los clips aislados de [UrbanSound8K](/papers/urbansound8k-salamon-2014) sobre una cama de ruido browniano, respetando los 10 folds oficiales para armar los splits. Lo que lo hace *canónico* no es que sea el dataset más usado —no lo es—, sino que es **la demostración de que un dataset puede publicarse junto con su generador**: se distribuye con el audio, las anotaciones, los scripts de generación, los scripts de los experimentos de ML, y un JAMS por paisaje con la especificación probabilística y la instanciada. Con eso, cualquiera puede reproducir el dataset exacto, generar una versión de 100 000 paisajes, generar una versión con las mismas escenas y otra SNR, o auditar si hubo sesgo en el muestreo. Ninguna de esas cosas es posible con la mayoría de los datasets publicados, ni entonces ni ahora.

**DCASE.** La adopción más consecuente. Los desafíos **DCASE** (*Detection and Classification of Acoustic Scenes and Events*) incorporaron datasets sintetizados con Scaper en sus tareas de detección de eventos, de forma prominente en la **Tarea 4** (*Sound Event Detection in Domestic Environments*), cuyo conjunto sintético fuertemente etiquetado se genera con la librería a partir de aislados de Freesound. El diseño de esa tarea es la respuesta institucional a la brecha: a los participantes se les entrega **real débilmente etiquetado**, **real sin etiquetar** y **sintético fuertemente etiquetado**, y se evalúa sobre **real fuertemente etiquetado**. Los sintéticos aportan la supervisión temporal que los reales no pueden aportar a escala; los reales aportan el realismo acústico que los sintéticos no tienen.

**Separación de fuentes.** Scaper se usó también fuera de SED, en datasets de **separación de fuentes universal** (el ecosistema WHAM/WHAMR/FUSS). La conexión es natural: si ya existe un motor que mezcla fuentes aisladas con parámetros controlados, los *targets* de separación salen gratis, porque **las fuentes individuales pre-mezcla son la verdad de terreno de la separación**. La librería lo soporta hoy con `save_isolated_events=True`, que escribe cada evento renderizado por separado junto con la mezcla. Ese flag **no existe en el paper de 2017**: es un buen ejemplo de cómo una abstracción correcta admite usos que su autor no anticipó, porque la especificación instanciada ya contenía la información necesaria y solo faltaba exponer los estemas.

### El API actual, comentado

```python
import os, numpy as np, scaper

# El soundbank es una convención de sistema de archivos, no una base de datos:
#   foreground/<label>/*.wav   y   background/<label>/*.wav   -> la subcarpeta ES la etiqueta.
audio_root = os.path.expanduser("~/audio")

sc = scaper.Scaper(
    duration=10.0,                                    # 10 s, como URBAN-SED
    fg_path=os.path.join(audio_root, "foreground"),
    bg_path=os.path.join(audio_root, "background"),
    protected_labels=[],                              # etiquetas exentas de transformaciones
    random_state=20170415,                            # reproducibilidad del MUESTREO
)
sc.ref_db = -50          # nivel del fondo en LUFS; todas las SNR se miden RESPECTO de este valor

# El fondo: uno por paisaje, tratado como textura holística.
# Cada argumento es una DISTRIBUTION TUPLE, no un valor. ('choose', []) = uniforme sobre
# todos los archivos de la etiqueta.
sc.add_background(label=("const", "noise"),        # URBAN-SED usa ruido browniano sintético,
                  source_file=("choose", []),      # elegido justamente para garantizar que el
                  source_time=("const", 0))        # fondo NO contenga eventos espurios sin anotar

# Un evento fijado a mano: el extremo determinista del continuo de control.
sc.add_event(label=("const", "siren"), source_file=("choose", []), source_time=("const", 0),
             event_time=("const", 2.0), event_duration=("const", 3.0),
             snr=("const", 12),                    # 12 LUFS sobre el fondo -> -38 LUFS
             pitch_shift=None, time_stretch=None)  # None = sin transformación

# La receta de URBAN-SED, completamente probabilística.
rng = np.random.RandomState(20170415)
for _ in range(rng.randint(1, 10)):                # n.º de eventos ~ Uniforme {1..9}
    sc.add_event(
        label=("choose", []),                      # cualquiera de las 10 clases urbanas
        source_file=("choose", []),
        source_time=("const", 0),                  # "para no perder el onset del evento"
        event_time=("normal", 5.0, 2.0),           # controla la polifonía INDIRECTAMENTE
        event_duration=("uniform", 0.5, 4.0),      # los clips de UrbanSound8K duran <= 4 s
        snr=("uniform", 6, 30),
        pitch_shift=("uniform", -3, 3),            # semitonos; NO altera la duración
        time_stretch=("uniform", 0.8, 1.2),        # factor;    NO altera el pitch
    )

# Instanciar y renderizar. Cada generate() muestrea de nuevo:
# UNA especificación -> INFINITAS instanciaciones.
sc.generate(
    audio_path="soundscape_0001.wav",
    jams_path="soundscape_0001.jams",   # <- LA RECETA COMPLETA
    txt_path="soundscape_0001.txt",     # <- onset/offset/label, cargable en Audacity
    allow_repeated_label=True, allow_repeated_source=True,
    reverb=None,                        # POSTERIOR al paper: reverb algorítmica de SoX
    fix_clipping=False,                 # POSTERIOR al paper
    peak_normalization=False,           # POSTERIOR al paper
    save_isolated_events=False,         # POSTERIOR: estemas para separación de fuentes
    no_audio=False,                     # True = instanciar sin renderizar (barato)
)

# La operación que hace único a Scaper: EDITAR LA RECETA Y RE-RENDERIZAR. Es el mecanismo
# del barrido de SNR: se homogeneiza la SNR de los 2000 JAMS de test y se regenera el audio
# -> 8 conjuntos de test idénticos salvo por UNA variable.
import jams
jam = jams.load("soundscape_0001.jams")
for obs in jam.annotations.search(namespace="scaper")[0].data:
    if obs.value["role"] == "foreground":
        obs.value["snr"] = 9.0
jam.save("soundscape_0001_snr9.jams")

scaper.generate_from_jams("soundscape_0001_snr9.jams",
                          audio_outfile="soundscape_0001_snr9.wav")
# fg_path / bg_path son opcionales aquí: si el soundbank se movió, se re-apunta. Esa es la
# dependencia que rompe el cierre — el JAMS referencia fuentes por RUTA, no por hash.
```

Tres cosas a notar, que son las tres decisiones de diseño del paper hechas API. **Cada parámetro es una tupla, no un valor**: la firma *fuerza* la indirección especificación→instancia, porque no hay forma de llamar `add_event(snr=12)` sino `snr=("const", 12)` — fricción deliberada y buena, que garantiza que la ruta determinista y la probabilística sean el mismo código. **`generate()` se puede llamar $N$ veces sobre el mismo objeto** y produce $N$ paisajes distintos: la especificación es el objeto persistente, el paisaje es efímero. Y **`generate_from_jams()` es una función de módulo, no un método**: reconstruir desde una receta no requiere el objeto que la creó, la receta es autosuficiente salvo por el soundbank, y eso es lo que la hace compartible.

**Qué es del paper y qué es posterior:** el paper de 2017 lista `const`, `choose`, `uniform`, `normal` y `truncnorm`, y no menciona reverberación, clipping ni estemas aislados. `choose_weighted`, `reverb`, `fix_clipping`, `peak_normalization` y `save_isolated_events` son adiciones posteriores. Atribuirle al paper capacidades de la documentación actual es un anacronismo fácil de cometer.

## Limitaciones

**Las que el paper reconoce.** (i) La **riqueza acústica**: los paisajes generados no pueden abarcar la complejidad de los reales y no reemplazan a las grabaciones reales anotadas si el objetivo es estimar el desempeño en producción. (ii) **La plausibilidad queda en manos del usuario**: "es posible generar paisajes sonoros que no son plausibles". Scaper es una herramienta **sin opinión** —no sabe nada de acústica ni del mundo, y no impedirá poner ocho sirenas simultáneas sobre un fondo de biblioteca—, así que la carga de la validez ecológica se transfiere íntegramente al usuario, que no tiene ninguna forma automática de verificarla: no hay validador de plausibilidad ni advertencias. (iii) **No hay control explícito de la polifonía**, que es variable de salida y no de entrada. (iv) **La aumentación por recombinación está propuesta, no demostrada**: el paper sugiere extraer los eventos no solapados de un dataset real y usarlos como soundbank, y añade con encomiable honestidad que "**esto queda por demostrarse**".

**Las que el paper no reconoce:**

1. **La calidad del soundbank es el techo de todo, y no se audita.** Si un clip etiquetado `dog_bark` contiene además una bocina de fondo, esa bocina entra al paisaje **sin aparecer en la anotación**, y el falso negativo forzado que el paper evitó cuidadosamente en el fondo reaparece por la puerta de los eventos. La garantía de anotación exacta es, con precisión, exactitud respecto del **modelo generativo**, no exactitud acústica.
2. **Diversidad combinatoria ≠ diversidad acústica.** "Infinitas instanciaciones" es literalmente cierto y epistemológicamente engañoso: los 10 000 paisajes de URBAN-SED se construyen sobre ~10 000 clips fuente y contienen ~50 000 eventos, así que **cada clip aparece en promedio unas cinco veces**, cambiado de pitch y de tiempo. Un modelo con suficiente capacidad puede memorizar los clips fuente y resolver la tarea por identificación de grabación en vez de por reconocimiento de clase. Y falta la ablación obvia: ¿cuánto mejora un modelo entrenado con 1000 vs. 10 000 vs. 100 000 paisajes del mismo soundbank?
3. **Sin modelado acústico del espacio ni dimensión espacial.** No hay posición de la fuente, distancia, azimut, Doppler ni salida multicanal. Para SED monoaural es aceptable; para las tareas de **localización y detección** (SELD, que se volvió tarea DCASE propia) es descalificante.
4. **Independencia i.i.d. entre eventos:** la estructura conjunta del mundo no es expresable en el modelo de especificación tal como está.
5. **El riesgo de aprender los artefactos de la síntesis**, que es la limitación más peligrosa porque **es invisible en validación** — la única forma de detectarla es evaluar sobre datos reales, que es exactamente lo que la síntesis intenta evitar.
6. **Nada sobre el costo computacional ni la escala de generación.** No se reporta cuánto tarda generar 10 000 paisajes ni si el proceso paraleliza; como cada llamada a SoX es un subproceso con I/O de disco, es plausible que el cuello de botella sea sustancial a escala de cientos de miles de ejemplos.
7. **Ninguna evaluación perceptual del realismo.** "Suenan bastante realistas en algunos casos" es una afirmación empírica sin evidencia: no hay test de escucha ni métrica objetiva de distancia entre distribuciones. Dado que los autores tenían montada toda la infraestructura de *crowdsourcing*, la ausencia es notoria.
8. **La reproducibilidad depende de la estabilidad del entorno.** El JAMS captura la receta, pero el audio depende de la versión de SoX, del medidor de LUFS y de la versión de Scaper: un JAMS de 2017 re-renderizado hoy produce audio *casi* idéntico, no idéntico.

## Por qué importa hoy

Ocho años después, Scaper sigue siendo la herramienta estándar para sintetizar paisajes sonoros con anotación fuerte: DCASE la adoptó institucionalmente, la librería siguió evolucionando (`choose_weighted`, `fix_clipping`, `peak_normalization`, `save_isolated_events`, `generate_from_jams` con re-apuntado de rutas) sin romper la abstracción original —la mejor señal de que la abstracción era correcta— y se extendió a separación de fuentes, bioacústica y escenas domésticas. El mérito de fondo no es técnico sino **conceptual**: normalizó la idea de que **un dataset es un programa, no una carpeta**.

La pregunta obvia hoy es otra: si existen modelos generativos de audio de calidad —AudioLDM, AudioGen, Stable Audio, Make-An-Audio— que producen audio arbitrario desde texto, ¿para qué sirve un secuenciador que recorta y suma WAVs?

| | **Scaper (composición)** | **Generativos de audio (síntesis neuronal)** |
|---|---|---|
| Qué produce | Mezclas de grabaciones **reales** existentes | Audio **nuevo**, nunca grabado |
| Diversidad de fuentes | Acotada al soundbank | En principio ilimitada |
| Realismo del evento aislado | **Perfecto** (es una grabación real) | Variable; artefactos, "sonido de generador" |
| Etiquetas fuertes | **Exactas por construcción** | **No las hay** |
| Control de la mezcla | Total y explícito (SNR, tiempos, polifonía) | Implícito, vía prompt; poco controlable |
| Reproducibilidad | Receta serializada, regeneración exacta | Semilla + modelo + versión + prompt; frágil |
| Costo por hora generada | Segundos de CPU | GPU, órdenes de magnitud más |
| Auditable | Sí: se sabe qué archivo fuente entró | No: el modelo es opaco |

**La asimetría decisiva es la fila de las etiquetas fuertes.** Si a un modelo texto→audio se le pide "un perro ladrando mientras pasa una sirena, en una calle", entrega diez segundos de audio plausible — y **no dice en qué milisegundo empieza el ladrido**. Para obtener la anotación fuerte habría que anotarlo a mano, y se vuelve al punto de partida, con el agravante de que ahora el audio ni siquiera es real. Los problemas nuevos que traen los generativos son cuatro más: **sesgos del generador** (un modelo texto→audio produce la *moda* de su distribución de entrenamiento, así que mil "ladridos de perro" son mil variantes de un ladrido prototípico y no la diversidad real de razas, distancias y acústicas — el dataset resultante tiene una **cola corta artificialmente**, y las colas son donde los modelos fallan); **colapso al entrenar con datos generados**, porque si el generativo se entrenó con [AudioSet](/papers/audioset-gemmeke-2017) y con él se generan datos para entrenar un detector que después se evalúa en AudioSet, el circuito es cerrado y el detector aprende la aproximación que el generativo hizo de la distribución, degradando progresivamente las colas al iterar; **artefactos aprendibles**, agravados, porque el audio neuronal tiene firmas propias (del vocoder, del decodificador latente) que una red discriminativa detecta con facilidad, y entrenar sobre él puede enseñar a detectar el generador en vez del evento; y **procedencia y licencia**, ya que Scaper dice exactamente qué archivo fuente entró en cada paisaje mientras un modelo generativo no dice de qué se acuerda.

Existe además una vía híbrida que es la que tiene sentido: **generar los eventos aislados con un modelo generativo y componerlos con Scaper**. El generativo resuelve la escasez de fuentes —mil variantes de bocina sin salir a grabar, que es exactamente el problema de calidad del soundbank—; Scaper resuelve la composición y la anotación. Operan en niveles distintos y son complementarios, no competidores. Algo parecido ocurre en el extremo opuesto del espectro: [SV2TTS](/papers/sv2tts-jia-2018) sintetiza voz nueva desde un embedding de hablante, con un realismo que Scaper jamás alcanzaría — pero sin ninguna anotación temporal derivada del proceso.

{{< concept-alert type="clave" >}}
**Scaper no fue superado, fue complementado.** Su ventaja competitiva —el vínculo determinista y exacto entre el proceso generativo y la anotación— no es algo que un modelo generativo pueda ofrecer, porque un generativo no *construye* la escena a partir de partes identificables: la *muestrea* como un todo. La lección general trasciende el audio: **cuando el cuello de botella es la anotación y no el dato, la solución no es generar más dato realista sino generar dato cuya anotación sea una consecuencia del proceso de generación.** Es el mismo principio detrás de los simuladores en robótica (la pose del objeto la sabe el simulador), del renderizado sintético en visión (la máscara de segmentación sale del *z-buffer*, gratis y perfecta) y del *fuzzing* basado en gramáticas (la entrada válida se construye desde la gramática que la define).
{{< /concept-alert >}}

## Erratas y matices

1. **El rango de SNR del barrido está mal enumerado en el texto.** La Sección 4 dice "repetimos este proceso **ocho veces**, fijando la SNR de los eventos en el rango 6–9, 9–12, 12–15, 15–18, **18–24**, 24–27 y 27–30": son **siete** rangos listados. La Figura 3 muestra inequívocamente **ocho** barras (`6–9, 9–12, 12–15, 15–18, 18–21, 21–24, 24–27, 27–30`). La lectura correcta es la de la figura; el texto colapsó por error "18–21, 21–24" en "18–24".
2. **Las unidades de la SNR se mezclan entre secciones.** La Sección 2 define la SNR en **LUFS** de forma explícita y razonada; la Sección 3 dice "6–30 **dB**" y la Figura 3 rotula su eje como `SNR (db)`. LUFS es una escala en decibeles relativa a *full scale*, así que las magnitudes son consistentes, pero la unidad correcta según la propia definición del paper es LUFS, y la notación `db` en minúsculas es descuidada.
3. **Hay un typo en la Figura 1:** el bloque central del diagrama dice **"SOUNDCSAPE INSTANTIATION & GENERATION"** en vez de "SOUNDSCAPE".
4. **El slide de la clase escribe mal el apellido del primer autor:** la referencia dice *"Scaper: a library for soundscape synthesis and augmentation. **Salaman** et al., 2017"*. Es **Salamon**, con o. La cita correcta es Salamon, MacConnell, Cartwright, Li y Bello, WASPAA 2017.
5. **"Scaper es una herramienta de data augmentation" es a lo sumo media verdad.** La palabra está en el título, pero **el paper no demuestra ningún resultado de aumentación**: los usos evaluados son generar un dataset desde cero y generar estímulos para un experimento con humanos, y el uso aumentativo aparece solo en la discusión, con la frase textual "esto queda por demostrarse".
6. **"URBAN-SED es el dataset de SED más grande" era cierto en 2017 y ya no lo es.** La afirmación es explícitamente relativa a su momento y viene relativizada por los propios autores; repetirla sin la fecha ni la coletilla es una tergiversación menor pero frecuente.
7. **Scaper no es un sintetizador de audio, es un secuenciador y mezclador.** No genera muestras nuevas: recorta, transforma y suma grabaciones existentes. El título dice "soundscape synthesis" y es correcto —sintetiza *paisajes*, no *sonidos*—, pero la ambigüedad hace que a veces se lo confunda con un modelo generativo.
8. **Los números de las Figuras 2 y 3 no están tabulados:** no hay ninguna tabla numérica en las cinco páginas, y todas las cifras de esta página son lecturas aproximadas de gráficos de barras.
9. **El paper de 2017 no habla de reverberación, clipping ni estemas aislados**, y no lista `choose_weighted`: todos son agregados posteriores de la librería.
10. **El formato de texto: "separado por espacios" vs. tabulaciones.** El paper describe la salida como *space-separated*; la implementación actual usa tabulaciones por defecto (`txt_sep='\t'`), que es lo que Audacity espera. Relevante si alguien escribe un parser basándose en el texto.
11. **La CNN evaluada no es la CNN de Salamon y Bello 2017 tal cual**, sino una adaptación sustancial (sigmoides, 64 filtros, *batch norm*, entrada de 1 s, pooling (2,2)) que pasa de 241k a 720k parámetros: es un pariente, no el original.
12. **El experimento de crowdsourcing no es de este paper:** la sección 5 resume resultados de Cartwright et al., *"Seeing sound"* (PACM HCI 2017), y los hallazgos sobre el espectrograma y sobre el comportamiento de los anotadores deben citarse a ese trabajo.

## En la clase 39 y su laboratorio

### El linaje Salamon: UrbanSound8K → Scaper → URBAN-SED

Este es el punto que hay que dejar explícito, porque es literalmente el mismo autor cerrando un círculo de tres años. **Justin Salamon** es primer autor de [UrbanSound8K](/papers/urbansound8k-salamon-2014) (Salamon, Jacoby y Bello, ACM Multimedia 2014) —el dataset que el laboratorio de la [Clase 39](/clases/clase-39) usa con sus 10 folds oficiales—, de la CNN con augmentation para clasificación de sonido ambiental (Salamon y Bello, IEEE SPL 2017) que el propio paper de Scaper adapta como uno de sus dos modelos evaluados, y de **Scaper**. Juan Pablo Bello firma los tres.

| Año | Trabajo | Qué problema resuelve | Qué deja abierto |
|---|---|---|---|
| 2014 | **UrbanSound8K** | No había dataset ni taxonomía para sonido urbano: se construye uno de 8732 clips ≤4 s, 10 clases, con 10 folds estratificados | Es clasificación de clips: **una clase por clip, sin tiempos**. No sirve para SED |
| 2017 | **CNN + augmentation** | ¿Cómo entrenar una CNN con tan poco dato? Con aumentación (deformación temporal, pitch, ruido, compresión dinámica) | La aumentación no crea escenas nuevas ni etiquetas temporales |
| 2017 | **Scaper + URBAN-SED** | Cómo obtener **etiquetas fuertes a escala**: componiendo escenas a partir de los clips aislados de 2014 | La brecha con lo real |

**El vínculo es literal, no temático: URBAN-SED se construye usando UrbanSound8K como soundbank.** Los clips de 2014 son las piezas, Scaper es el motor de ensamblaje, URBAN-SED es el producto. Lo que en 2014 era un dataset de clasificación se convierte en 2017 en la materia prima de un dataset de detección, sin una sola anotación humana adicional.

### La filosofía de diseño que comparten: los 10 folds

Esta es la continuidad más concreta, y la que el laboratorio toca directamente. UrbanSound8K se distribuye **pre-dividido en 10 folds estratificados**, y su documentación es célebremente enfática en que hay que usar los folds oficiales y reportar validación cruzada de 10 pliegues sobre ellos, en vez de re-barajar aleatoriamente. La razón: los 8732 clips provienen de 1302 grabaciones de campo de Freesound, y varios clips pueden salir de la misma grabación original, así que un *shuffle* aleatorio pondría fragmentos de la misma grabación en entrenamiento y en test y el modelo, en vez de reconocer la clase, **reconocería la grabación** — con una inflación de la métrica que suele ser de varios puntos. Los folds oficiales garantizan que todos los clips de una misma grabación caigan en el mismo fold: es control de fuga a nivel de **grupo**, no de instancia.

URBAN-SED **hereda esa disciplina**: folds 1–6 para los 6000 paisajes de entrenamiento, 7–8 para los 2000 de validación, 9–10 para los 2000 de test. La propiedad preservada es que **ningún clip fuente que aparece en un paisaje de entrenamiento aparece en uno de test**. Sin esa herencia, la síntesis habría destruido justamente la garantía que costó construir en 2014, porque un mismo `siren.wav` podría reaparecer en test bajo otro pitch, otro tiempo de inicio y otra SNR.

{{< concept-alert type="recordar" >}}
Cuando en el laboratorio de la [Clase 39](/clases/clase-39) se use UrbanSound8K con sus 10 folds y se resista la tentación de hacer `train_test_split(shuffle=True)`, se está aplicando exactamente el mismo principio que Salamon aplicó al construir URBAN-SED tres años después. **La unidad de división no es el ejemplo: es la fuente de la que el ejemplo deriva.** Es la misma lección de cualquier dominio con datos agrupados — pacientes en datos clínicos, usuarios en logs, repositorios en datasets de código.
{{< /concept-alert >}}

### Dónde encaja Scaper en el mapa de data augmentation de audio

La [Clase 37](/clases/clase-37) cubrió el fundamento de [data augmentation de audio](/fundamentos/data-augmentation-de-audio): mezcla de ruido a SNR controlada, SpecAugment, *pitch shifting* y *time stretching*. La Clase 39 vuelve sobre el tema y agrega la advertencia sobre la síntesis. Scaper opera en un **nivel distinto** de todas esas técnicas, y ubicarlo bien es lo que da coherencia al conjunto.

| Nivel | Dónde opera | Ejemplos | ¿Crea ejemplos nuevos? | ¿Crea etiquetas nuevas? |
|---|---|---|---|---|
| **Representación** | Sobre el espectrograma, después del *feature extraction* | **SpecAugment** (máscaras de tiempo y frecuencia), mixup de features | No: perturba uno existente | No: hereda la etiqueta |
| **Forma de onda** | Sobre la señal, antes del *feature extraction* | **Pitch shift**, **time stretch**, ruido a **SNR** controlada, compresión dinámica, filtrado, códec | No: perturba uno existente | No: hereda la etiqueta |
| **Escena** | Sobre la **composición** de la escena | **Scaper** | **Sí**: compone una escena que no existía | **Sí, y exactas**: onsets, offsets y polifonía se derivan del proceso |

La distinción operacional es nítida. Las técnicas de los dos primeros niveles toman **un ejemplo etiquetado** $(x, y)$ y producen $(T(x), y)$: necesitan que el ejemplo etiquetado exista, la etiqueta se hereda sin cambios, y **no aumentan la cantidad de anotación**, solo su rendimiento. Scaper toma **un conjunto de fuentes etiquetadas por clase** y produce $(x_{\text{nuevo}}, y_{\text{nuevo}})$ donde $y_{\text{nuevo}}$ es **estructuralmente más rica** que las etiquetas de entrada: entra supervisión débil (una etiqueta por clip, como en UrbanSound8K o [ESC-50](/papers/esc50-piczak-2015)) y sale supervisión fuerte (onsets y offsets). **No transforma un ejemplo existente: compone ejemplos nuevos junto con sus etiquetas**, y ese es el truco que ninguna de las otras técnicas hace. Es exactamente la observación del paper que citamos al principio: la aumentación aplicada al paisaje completo deja intactos "el timing de los eventos, el grado de solapamiento y la SNR", y Scaper es la herramienta que ataca esas tres variables — las que definen la tarea de SED.

Y nótese que **los tres niveles se componen**. Scaper mismo usa pitch shift y time stretch (nivel forma de onda) **por evento**, dentro de la composición. Un pipeline maduro sería: Scaper para componer escenas con etiquetas fuertes → augmentation de forma de onda sobre la mezcla resultante para simular variación de canal → SpecAugment sobre el espectrograma durante el entrenamiento. Tres niveles, tres tipos de invarianza, ningún conflicto.

### La advertencia del slide 62, cerrada

El slide de la Clase 39 dice, textualmente:

> "Use of sound synthesis techniques is a convenient strategy. However, performance on real data might be poor if a model is trained using only synthetic data. **Need for finetuning on real data.**"

Y el paper que lo respalda dice, en su discusión, que los paisajes generados "no pueden abarcar la riqueza y complejidad de los paisajes sonoros reales" y que Scaper "no puede usarse como reemplazo de grabaciones del mundo real anotadas manualmente, si queremos estimar cuán bien rendirá un modelo en un ambiente real". Las dos frases dicen lo mismo, y el paper llegó primero. Lo notable es que la advertencia **está en el paper que propone la técnica**, no en una crítica posterior: los autores conocían la limitación, la enunciaron con precisión, y aun así el trabajo fue valioso — porque **acotaron correctamente el alcance de su propia herramienta**.

La síntesis de datos es una estrategia legítima y potente cuando se sabe qué compra y qué no: compra **volumen y exactitud de anotación**, no compra **realismo acústico**. Y como esas dos cosas son separables, la arquitectura de solución correcta es mixta — pre-entrenamiento masivo con sintético y ajuste fino con real, o el régimen de tres fuentes que DCASE institucionalizó. Ese es el mensaje que el slide comprime en tres líneas y que este paper desarrolla en cinco páginas.

## Notas y enlaces

- **Paper:** Justin Salamon, Duncan MacConnell, Mark Cartwright, Peter Li y Juan Pablo Bello, *"Scaper: A Library for Soundscape Synthesis and Augmentation"*, IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, octubre de 2017. Financiado por NSF (awards 1544753 y 1633259) y un Google Faculty Award. [PDF local](/papers/scaper-salamon-2017.pdf).
- **Código:** `github.com/justinsalamon/scaper` — la librería sigue mantenida; el API actual difiere del paper en varios parámetros (ver erratas).
- **Dataset:** URBAN-SED (`urbansed.weebly.com`), publicado junto con los scripts de generación y los de los experimentos de ML.
- **Formato de anotación:** JAMS (*JSON Annotated Music Specification*), Humphrey, Salamon, Nieto, Forsyth, Bittner y Bello, ISMIR 2014. **Métrica:** `sed_eval` (Mesaros, Heittola y Virtanen).
- **Papers relacionados:** [UrbanSound8K](/papers/urbansound8k-salamon-2014) (el soundbank y el dataset del laboratorio), [ESC-50](/papers/esc50-piczak-2015) (el otro dataset canónico de sonido ambiental con etiquetas débiles), [AudioSet](/papers/audioset-gemmeke-2017) (la escala ontológica del audio tagging), [SV2TTS](/papers/sv2tts-jia-2018) (el otro extremo: síntesis neuronal de audio, donde la anotación no se deriva del proceso).
- **Fundamentos:** [data augmentation de audio](/fundamentos/data-augmentation-de-audio), [clasificación de audio](/fundamentos/clasificacion-de-audio), [datasets de audio](/fundamentos/datasets-de-audio).
- **En el curso:** [Clase 39](/clases/clase-39) (el paper y su laboratorio sobre UrbanSound8K), [Clase 37](/clases/clase-37) (datasets y herramientas de audio, donde se cubrió el augmentation clásico) y el [dominio de audio](/dominios/audio) para la línea de tiempo completa.
