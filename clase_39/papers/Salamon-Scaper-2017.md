# Scaper: A Library for Soundscape Synthesis and Augmentation — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Justin Salamon$^{1,2}$, Duncan MacConnell$^1$, Mark Cartwright$^1$, Peter Li$^1$ y Juan Pablo Bello$^1$.
  - $^1$ **Music and Audio Research Laboratory (MARL)**, New York University, NYC, NY, USA.
  - $^2$ **Center for Urban Science and Progress (CUSP)**, New York University, NYC, NY, USA.
  - Correspondencia: `justin.salamon@nyu.edu` (nota al pie de la primera página).
- **Venue:** *2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics* (**WASPAA 2017**), 15–18 de octubre de 2017, New Paltz, NY.
- **Financiamiento:** NSF awards **1544753** y **1633259**, y un **Google Faculty Award** (nota al pie, primera página).
- **Código:** `https://github.com/justinsalamon/scaper` (nota al pie 1). El dataset derivado, URBAN-SED, en `http://urbansed.weebly.com/` (nota al pie 3); los scripts de generación y de los experimentos de ML en `https://git.io/v9GEM` (nota al pie 4).
- **Índice de términos declarado:** *Soundscape, synthesis, sound event detection.*

**Qué es Scaper, en una frase precisa:** una librería Python de código abierto que, dado un *soundbank* de eventos sonoros aislados organizados en carpetas de primer plano y de fondo, actúa como un **secuenciador de audio de alto nivel controlado probabilísticamente**, capaz de instanciar y renderizar un número arbitrario de paisajes sonoros distintos a partir de una **única especificación** — entregando, junto con cada audio, la **anotación fuerte exacta** (onset, offset y etiqueta de cada evento) y la receta completa que permite reconstruirlo.

El paper es corto —cinco páginas de contenido en formato WASPAA— y su tesis cabe en una línea: **si tú generas la mezcla, la anotación no hay que estimarla, se conoce**. Todo lo demás del trabajo es la consecuencia de tomarse esa observación en serio como problema de ingeniería de datos: cómo se especifica una familia de mezclas, cómo se muestrea de ella, cómo se serializa la receta, y qué experimentos controlados se vuelven posibles cuando cada variable de la mezcla es una perilla y no un accidente del mundo.

Las tres piezas que el paper aporta, en orden de importancia decreciente para la práctica:

1. **La librería** y su modelo de especificación probabilística (Sección 2).
2. **URBAN-SED** (Sección 3): 10.000 paisajes sonoros de 10 s, casi 30 horas, cerca de 50.000 eventos anotados, con polifonías máximas entre 1 y 7. En el momento de publicación, *el dataset con etiquetas fuertes más grande disponible para SED*.
3. **Dos estudios controlados** (Secciones 4 y 5) que son la demostración del argumento: un barrido de desempeño de dos modelos de SED como función de la **polifonía máxima** y de la **SNR**, y un experimento de *crowdsourcing* donde los estímulos sintéticos sirven de referencia perfecta para medir a los anotadores humanos.

Un detalle de calibración antes de seguir: este es un *tool paper*. No propone una arquitectura, no reporta estado del arte, no compite en un benchmark. Su unidad de contribución es **infraestructura**, y el criterio con el que hay que juzgarlo es el que se aplica a una librería: ¿resuelve un problema real, la abstracción es la correcta, y sigue viva ocho años después? La respuesta a las tres es sí, y la sección 11 desarrolla por qué.

---

## 2. Contexto: el problema del dato en sound event detection

### 2.1. Dos tareas que se confunden todo el tiempo

Conviene separarlas con precisión porque de la distinción sale todo el paper.

| | **Audio tagging** (etiquetas débiles) | **Sound event detection** (etiquetas fuertes) |
|---|---|---|
| Pregunta | ¿Qué fuentes están presentes en este clip? | ¿Qué fuentes, y **cuándo** empieza y termina cada instancia? |
| Salida | Vector multi-etiqueta por clip | Secuencia de tripletas $(\text{onset}, \text{offset}, \text{label})$ |
| Anotación requerida | Marcar presencia/ausencia | Marcar **límites temporales** de cada evento |
| Costo de anotar 1 h de audio | Bajo | Alto, y crece con la densidad de eventos |
| Métrica típica | mAP, AUC por clase | F-measure basada en segmentos o en eventos (`sed_eval`) |

El paper define SED en la primera línea de la Sección 1: "la tarea de identificar automáticamente **la fuente y la ubicación en el tiempo** de distintos sonidos a medida que ocurren en un flujo de audio continuo". Y acto seguido establece la asimetría que motiva todo: aunque existe trabajo sobre entrenamiento con etiquetas débiles (cita [6], Su et al., ICASSP 2017), "la mayoría de los modelos propuestos hasta la fecha para SED requieren datos con etiquetas fuertes". Más aún —y este es el punto que suele pasarse por alto—:

> "Incluso los modelos que **pueden** entrenarse con datos débilmente etiquetados requieren datos fuertemente etiquetados **para evaluar** su desempeño a resoluciones temporales más finas." (Sección 1)

Es decir: la supervisión débil te puede liberar del costo de anotación en el conjunto de **entrenamiento**, pero **no** en el de **evaluación**. Si tu métrica es temporal, necesitas verdad de terreno temporal, punto. El cuello de botella del dato en SED es irreducible por el lado del entrenamiento débil.

Las aplicaciones que el paper enumera para justificar el interés —monitoreo de ruido en ciudades inteligentes [1], monitoreo bioacústico de especies y migraciones [2], autos autónomos [3], vigilancia [4], indexación multimedia a gran escala [5]— comparten una propiedad: todas necesitan saber *cuándo*. Un sensor de ruido urbano que reporta "hubo un martillo neumático en algún momento de esta hora" no sirve para nada regulatorio; lo que importa es la duración acumulada de exposición.

### 2.2. La cifra que duele

El paper ancla la escasez con un número concreto en la Sección 1:

> "los datasets anotados manualmente para SED con etiquetas fuertes son muy limitados en tamaño (p. ej., el *development set* de **TUT Sound Events 2016 dura 78 minutos**)."

**Setenta y ocho minutos.** Ese era el orden de magnitud del recurso público disponible para entrenar detectores temporales de eventos sonoros en 2017, en un momento en que ImageNet tenía 1.2 millones de imágenes etiquetadas y las CNN de visión llevaban cinco años escalando sobre esa base. La desproporción no es un detalle de contexto: **es la razón de existir de Scaper**.

### 2.3. Por qué la anotación fuerte es tan cara: la ambigüedad de los límites

El paper dice que generar anotaciones fuertes es "laborioso y consume mucho tiempo" (Sección 1), pero el problema es peor que el costo por hora-humano. La dificultad estructural es que **los límites temporales de un evento sonoro no son un hecho objetivo bien definido**, y por lo tanto no existe un anotador perfecto al que aspirar.

Piensa en los diez tipos de sonido del taxonómico urbano (los mismos de UrbanSound8K, que la Sección 3 usa como *soundbank*):

- Un **disparo** (`gun_shot`) es casi un impulso: el onset es inequívoco al milisegundo. Pero el offset ya no lo es — ¿termina cuando el transitorio decae, o cuando la cola reverberante se pierde en el fondo? La respuesta depende del umbral de audibilidad respecto del ruido de fondo, que a su vez depende de la mezcla.
- Un **aire acondicionado** (`air_conditioner`) o un **motor en ralentí** (`engine_idling`) son estacionarios y difusos. No tienen onset perceptual: se van imponiendo. Dos anotadores competentes pueden diferir en varios segundos, y ninguno de los dos está equivocado.
- Un **martillo neumático** (`jackhammer`) es una secuencia de impactos. ¿Es un evento con estructura interna, o veinte eventos? La respuesta cambia la anotación de un clip de 4 s de uno a veinte registros.
- **Niños jugando** (`children_playing`) o **música callejera** (`street_music`) son texturas compuestas. El límite del "evento" es un límite semántico, no acústico.

De aquí salen tres consecuencias que ponen un **techo duro** a los datasets reales:

1. **El desacuerdo entre anotadores es irreducible para las clases difusas.** No es ruido de anotación que se pueda promediar hasta cero con más anotadores, porque no hay un valor verdadero latente al que converger: hay una decisión de convención. Cualquier acuerdo se logra imponiendo un protocolo, y el protocolo se vuelve parte de la definición de la tarea.
2. **El desempeño reportado de un modelo está acotado por el acuerdo entre anotadores.** Si dos humanos concuerdan al 0.75 de F-measure sobre un dataset, un modelo que reporta 0.78 no está superando a los humanos: está sobreajustando las idiosincrasias del anotador particular que produjo el *ground truth*. Sin una estimación del techo humano, las cifras de un benchmark de SED son incomparables entre datasets.
3. **La métrica hereda la ambigüedad.** Por eso `sed_eval` (Mesaros et al. [30]) ofrece evaluación *segment-based* además de *event-based*: cuantizar el tiempo en segmentos de 1 s amortigua el desacuerdo en los límites. El paper elige exactamente esa métrica a 1 s (Sección 4), y no por casualidad.

Este punto —la ambigüedad de los límites como techo epistémico— es lo que convierte la Sección 5 del paper (el experimento de *crowdsourcing*) en algo más que una anécdota: es la única forma de medir el techo, y requiere estímulos con anotación perfecta. Volveremos sobre eso.

### 2.4. Por qué el data augmentation clásico no alcanza para SED

El paper hace aquí una observación fina y específica del dominio, que es la bisagra conceptual entre "augmentation" y "síntesis" (Sección 1):

> "Para SED, dado que los datos de entrenamiento están compuestos de **paisajes sonoros que contienen múltiples eventos**, las aumentaciones aplicadas al paisaje sonoro **como un todo** ciertamente pueden ayudar, pero están limitadas en que las características del paisaje sonoro — como **el timing de los eventos, el grado de solapamiento y la SNR** — permanecerán **sin cambios** aun después de la transformación."

Léelo dos veces. Si tomas un paisaje sonoro real de 10 s con cuatro eventos y le aplicas *pitch shift*, *time stretch* o le sumas ruido, obtienes un ejemplo nuevo — pero la **estructura de la escena** es la misma: los mismos cuatro eventos, el mismo orden, el mismo patrón de solapamiento, las mismas relaciones de nivel entre ellos. La aumentación clásica genera variación **dentro de** una escena; no genera **escenas nuevas**.

Y para SED, la estructura de la escena es precisamente lo que el modelo tiene que aprender a desenredar. Un detector que solo ha visto cuatro configuraciones de solapamiento no va a generalizar a la quinta, por mucho que le hayas variado el pitch de cada una.

La segunda mitad del argumento apunta a la **evaluación**, y es el que a mí me parece más valioso:

> "Para evaluación, una limitación de los datasets basados en grabaciones reales es que **no es posible controlar las distintas características acústicas**, lo que podría ser útil para dar visión sobre las diferencias entre distintos modelos de SED." (Sección 1)

Con datos reales tú puedes medir *que* el modelo A es mejor que el modelo B. No puedes medir *por qué*, ni *bajo qué condiciones deja de serlo*, porque las condiciones no son manipulables — vienen empaquetadas en la grabación.

### 2.5. Qué había antes

El paper posiciona su trabajo contra dos frentes (Sección 1):

- **Síntesis de escenas por otras razones:** síntesis binaural/espacial [10] (Zotkin et al.), síntesis de eventos acústicos [11] (Verron et al., un sintetizador inmersivo 3D de sonidos ambientales) y síntesis de texturas [12] (Schwarz, *state of the art in sound texture synthesis*). El diagnóstico es preciso: esos sistemas "típicamente **no están diseñados con el objetivo** de entrenar/evaluar el desempeño de SED de máquinas (y humanos)", lo que implica tres carencias concretas: (i) no reproducen necesariamente los tipos de escena sobre los que se evalúan esos modelos, (ii) **no generan anotaciones que correspondan al audio sintetizado**, y (iii) no están pensados para procesamiento por lotes ni para integrarse con pipelines de ML.

  La carencia (ii) es la decisiva. Un sintetizador de audio es un generador; un generador de datasets es un generador **más un anotador acoplado**. La diferencia es de plomería, y la plomería es la contribución.

- **El único antecedente diseñado para SED: Lafay et al. [13]**, *"A morphological model for simulating acoustic scenes and its application to sound event detection"* (IEEE/ACM TASLP, vol. 24, n.º 10, 2016). Scaper se diferencia en tres ejes, textualmente (Sección 1):
  1. **Nivel de control.** El sistema de Lafay "solo provee controles de alto nivel para generar paisajes sonoros basados en **distribuciones fijas**", mientras Scaper usa la noción de *event specification* acoplada a **múltiples distribuciones**, lo que da un rango continuo de control "desde una definición probabilística de alto nivel hasta especificar cada detalle de cada evento sonoro".
  2. **Transformaciones por evento.** Scaper aplica *pitch shifting* y *time stretching* **individualmente a cada evento**, no a la escena.
  3. **Formato de anotación.** Genera anotaciones en **JAMS** [14], que almacenan metadata estructurada y permiten reconstruir el paisaje completo desde la anotación, o manipularla para generar variantes.

  A eso se suma un argumento de ecosistema: Python, sin software propietario (el paper nombra explícitamente Matlab), fácil de integrar con scikit-learn [15], TensorFlow [16], Keras [17], Essentia [18] y Librosa [19].

Ese último punto no es marketing. Una herramienta de generación de datos que vive fuera del lenguaje del pipeline de entrenamiento se usa una vez y se abandona; una que vive dentro se usa en cada corrida. La decisión de implementar en Python, sobre `pysox` [23], y de emitir un formato de anotación ya estándar en la comunidad MIR, es lo que hizo a Scaper sobrevivir.

---

## 3. La idea central: si generas el paisaje sonoro, las etiquetas vienen gratis

### 3.1. El argumento en su forma mínima

El razonamiento es casi tautológico y por eso mismo es fuerte:

> Un paisaje sonoro sintetizado es una **suma de fuentes cuyos tiempos de inicio y duración son parámetros del proceso de síntesis**. Los parámetros son conocidos porque fueron elegidos. Por lo tanto la anotación fuerte no se estima: **se lee del generador**.

Formalmente, si el paisaje sonoro se construye como
$$x(t) = b(t) + \sum_{i=1}^{N} g_i \cdot e_i(t - \tau_i)$$
donde $b$ es el fondo, $e_i$ el $i$-ésimo evento (ya recortado y transformado), $\tau_i$ su tiempo de inicio y $g_i$ su ganancia, entonces la anotación fuerte es exactamente
$$\mathcal{A} = \{(\tau_i,\; \tau_i + d_i,\; \ell_i)\}_{i=1}^{N}$$
con $d_i$ la duración post-transformación y $\ell_i$ la etiqueta. **No hay estimación, no hay anotador, no hay desacuerdo, no hay error.** La anotación es exacta *por construcción*.

Esto no es una mejora incremental sobre la anotación manual: es un cambio de categoría. La anotación manual tiene un error irreducible (Sección 2.3 de este análisis); la anotación sintética tiene error **cero respecto de su propia definición generativa**. El costo marginal de una hora anotada pasa de decenas de dólares y horas-humano a segundos de CPU.

El paper lo aprovecha inmediatamente y con una elegancia que vale destacar. En la Sección 3, para el fondo de URBAN-SED:

> "Usamos el mismo archivo de audio de fondo para todos los paisajes sonoros, un clip de 10 segundos de **ruido browniano**, que se asemeja al 'zumbido' típico que se escucha en ambientes urbanos. **Al usar un fondo puramente sintetizado tenemos la garantía de que no contiene ningún evento sonoro espurio que no estaría incluido en la anotación.**"

Ese es el razonamiento llevado a su conclusión lógica. Si tú grabas un fondo urbano real y lo usas como cama, es **casi seguro** que contiene una bocina lejana, un perro, un frenazo — eventos de tus propias clases objetivo, presentes en el audio y **ausentes de la anotación**. Cada uno de ellos es un falso negativo forzado en el *ground truth*: el modelo detecta correctamente algo que la etiqueta niega, y la métrica lo castiga. El ruido browniano es acústicamente pobre, sí, pero **es el único fondo que puedes garantizar limpio**. Es una decisión de integridad del dato por sobre el realismo, y está bien tomada dado el objetivo.

(Sobre la elección del ruido browniano —densidad espectral $\propto 1/f^2$, es decir, con la energía concentrada en bajas frecuencias— el paper solo justifica que "se asemeja al zumbido típico" urbano. Es una elección razonable: el tráfico rodante y la maquinaria producen precisamente ese perfil de espectro descendente. Pero es una aproximación estacionaria de algo que en la realidad no lo es.)

### 3.2. El segundo beneficio, que es el importante

El paper es explícito en que la anotación gratis no es la única ganancia, y en mi lectura el segundo beneficio es **más valioso a largo plazo**: el **control sistemático sobre las variables de la mezcla**.

Con datos reales, la SNR de un evento respecto del fondo, la densidad de eventos por minuto, el grado de solapamiento y el tipo de fondo son **covariables observacionales**: puedes medirlas *a posteriori* (con dificultad, porque necesitarías las fuentes separadas), pero no puedes **intervenir** sobre ellas. Y están **confundidas entre sí**: en una grabación urbana real, la hora del día correlaciona simultáneamente con la densidad de eventos, con el nivel de fondo y con la distribución de clases. Si un modelo falla más en los clips de las 8 de la mañana, tú no sabes si es por polifonía, por SNR o porque a esa hora hay más motos.

Scaper convierte esas covariables en **variables manipulables**. Eso mueve el estudio del modelo desde el diseño observacional al **diseño experimental**, con toda la potencia inferencial que eso implica: puedes fijar todo y variar una sola cosa.

El paper ejecuta esa idea de la forma más limpia posible en la Sección 4, y el mecanismo merece atención porque es donde la reproducibilidad deja de ser una virtud abstracta y se vuelve una capacidad operativa:

> "A priori esto parece complicado, dado que cada paisaje sonoro contiene múltiples eventos con valores de SNR heterogéneos. Scaper ofrece una solución sencilla: tomamos las **2000 anotaciones JAMS del conjunto de test, las editamos de modo que todos los eventos de un paisaje sonoro tengan la misma SNR, y luego re-generamos los archivos de audio a partir de las anotaciones JAMS modificadas.** Repetimos este proceso ocho veces [...] Esto resulta en **ocho versiones del conjunto de test que tienen características idénticas con la excepción de la SNR**, permitiendo un experimento altamente controlado que no sería posible de otro modo." (Sección 4)

Detente en eso. Ocho conjuntos de test que son **el mismo conjunto de test** —los mismos archivos fuente, los mismos tiempos de inicio, las mismas duraciones, las mismas transformaciones, el mismo fondo, la misma semilla efectiva— **excepto por una única variable**. Es un experimento *ceteris paribus* perfecto sobre un dataset de audio, algo que en el mundo de las grabaciones reales simplemente no existe.

Y la puerta de entrada a esa capacidad no es el sintetizador: es el **formato de anotación editable y re-ejecutable**. La reproducibilidad no es aquí una buena práctica de higiene; es la *feature*. Es exactamente la diferencia entre guardar los resultados de una consulta y guardar la consulta.

### 3.3. Lo que este argumento **no** dice

Por rigor, y porque es donde se abusa de este paper: la anotación es exacta **respecto del modelo generativo**, no respecto de la percepción. Si un evento se sintetiza a SNR $-3$ dB bajo un fondo denso, la anotación dice que ese evento está presente durante 2.3 s, pero un oyente humano —o un micrófono real— podría no percibir absolutamente nada. La etiqueta es correcta por definición y perceptualmente falsa.

Esto no es una objeción menor: define un régimen donde el *ground truth* sintético se vuelve inalcanzable, y donde un modelo entrenado sobre él aprende a perseguir una señal que no existe. Es la razón por la que URBAN-SED acota la SNR al rango $[6, 30]$ dB (Sección 3) y no incluye eventos enterrados. El paper no discute explícitamente este punto — es una limitación no reconocida que retomo en la Sección 10.

---

## 4. El modelo de generación probabilística

### 4.1. La ontología: foreground y background

La Sección 2 abre con la premisa perceptual que estructura todo el diseño:

> "SED se basa en la noción de que los sonidos en un paisaje sonoro pueden agruparse a grandes rasgos en dos categorías: **eventos sonoros de primer plano**, que son salientes y reconocibles, y **sonidos de fondo**, a menudo considerados como un **único sonido holístico** que es más distante, ambiguo y **similar a una textura**."

Las citas de respaldo son de psicoacústica y de calidad sonora urbana, no de ML: Maffiolo [20] (tesis doctoral sobre caracterización semántica y acústica de la calidad sonora urbana), Guastavino [21] (*The ideal urban soundscape*, Acta Acustica 2006) y McDermott, Schemitsch y Simoncelli [22] (*Summary statistics in auditory perception*, Nature Neuroscience 2013). La última es la más profunda: sostiene que la percepción de texturas auditivas se apoya en **estadísticas de resumen** temporales, no en el detalle instante a instante — lo que justifica tratar el fondo como un objeto único y estacionario en vez de como una colección de eventos.

De ahí la decisión de arquitectura:

> "un paisaje sonoro se genera como **la sumatoria de eventos de primer plano y una grabación de fondo**."

Y la delegación explícita de la curaduría al usuario:

> "Le corresponde al usuario **curar** un *soundbank* de su elección y organizar los sonidos en carpetas de **foreground** y **background**, con una **subcarpeta por cada clase de sonido (etiqueta)**."

El sistema de archivos *es* el esquema. `fg/<label>/*.wav` y `bg/<label>/*.wav`. No hay base de datos, no hay manifiesto, no hay formato de metadata que aprender. Para alguien que trabaja con sistemas de datos, esta es una decisión de diseño reconocible y buena: **la convención de nombres como esquema implícito** minimiza la fricción de adopción a costa de no poder expresar relaciones complejas. Es el mismo tipo de compromiso que hace `ImageFolder` de PyTorch.

La consecuencia declarada:

> "Scaper es **agnóstico al contenido** y puede aplicarse directamente a una variedad de dominios de audio incluyendo paisajes urbanos y rurales, grabaciones bioacústicas, ambientes interiores (p. ej. casas inteligentes) y grabaciones de vigilancia."

### 4.2. La event specification

El bloque de construcción central. Una *event specification* almacena todas las propiedades de un evento que Scaper puede controlar (Sección 2):

| Propiedad | Qué controla | Nota del paper |
|---|---|---|
| `label` | La clase del evento | Determina la subcarpeta del soundbank |
| `source_file` | El clip específico a usar | Dentro de los que corresponden a la etiqueta |
| `source_time` | Dónde empieza el recorte **dentro del clip fuente** | Permite tomar un fragmento interno |
| `event_time` | Cuándo empieza el evento **en el paisaje generado** | El onset de la anotación |
| `event_duration` | Cuánto dura | Junto al anterior determina el offset |
| `snr` | Nivel respecto del fondo | Ver Sección 6 de este análisis |
| `role` | `foreground` o `background` | Distingue eventos de cama |
| `pitch_shift` | En semitonos | "**no afecta la duración**" |
| `time_stretch` | Como factor de la duración | "**no afecta el pitch**" |

Las dos aclaraciones entre paréntesis son técnicamente importantes: significan que Scaper usa transformaciones **desacopladas** (vía SoX, a través de `pysox` [23]), no un simple *resampling*. Un cambio de velocidad de reproducción altera pitch y duración conjuntamente y de forma ligada; SoX implementa *pitch shift* preservando duración y *tempo/stretch* preservando pitch mediante procesamiento de fase. Esto importa porque significa que las dos perillas son **ortogonales**: puedes barrer una sin contaminar la otra. Para un diseño experimental, esa ortogonalidad es todo.

Un paisaje sonoro queda entonces definido por:

$$\text{Soundscape} = \underbrace{\{\text{spec}_1, \dots, \text{spec}_N\}}_{\text{foreground specification}} \;\cup\; \underbrace{\{\text{spec}_{bg}\}}_{\text{background specification}} \;+\; (\text{duración}, \text{nivel de referencia})$$

El usuario fija una duración deseada, un nivel de referencia de sonoridad para el fondo, y va agregando *event specifications*.

### 4.3. La distribution tuple

Aquí está la abstracción que hace todo el trabajo. Textual (Sección 2):

> "Para cada propiedad en una *event specification* el usuario provee una **distribution tuple**, que define una distribución de la cual muestrear el valor de la propiedad."

Las distribuciones soportadas según el paper:

| Tupla | Semántica | Ejemplo |
|---|---|---|
| `('const', v)` | Valor constante | `('const', 0)` para `source_time` |
| `('choose', [a, b, c])` | Selección uniforme de una lista discreta; lista vacía = todo lo disponible | `('choose', [])` para `source_file` |
| `('uniform', min, max)` | $\mathcal{U}(\text{min}, \text{max})$ continua | `('uniform', 0, 9)` para `event_time` |
| `('normal', mu, sigma)` | $\mathcal{N}(\mu, \sigma)$ | `('normal', 10, 3)` para `snr` |
| `('truncnorm', mu, sigma, min, max)` | Normal truncada al intervalo | `('truncnorm', 3, 1, 0.5, 5)` para `event_duration` |

Y el paper añade que "**se pueden agregar fácilmente distribuciones adicionales**" — promesa que la librería cumplió: la versión actual incluye además `choose_weighted`, que permite muestrear de una lista discreta con probabilidades no uniformes (verificado en `SUPPORTED_DIST` de `scaper/core.py`, rama `master`). Ese agregado importa más de lo que parece: permite construir *soundbanks* con prior de clase realista en vez de uniforme, que es una de las críticas de realismo que desarrollo en la Sección 8.

El rango de expresividad que esto abre está explicitado en el paper:

> "el usuario tiene control sobre **cuán detallada es la especificación**: desde definir con precisión cada propiedad de cada evento usando constantes, hasta una especificación probabilística de alto nivel que solo especifica una distribución de la cual muestrear para cada propiedad de cada evento."

Es decir, un **continuo** entre dos extremos:

- **Todo `const`** → la especificación es una receta determinista. Scaper se comporta como un renderizador de mezclas. Útil para reproducir un estímulo exacto.
- **Todo distribucional** → la especificación es un **generador de una familia de paisajes sonoros**, y "el usuario puede generar **infinitas instanciaciones** de paisajes sonoros" (Sección 2).

### 4.4. Por qué la indirección entre especificación e instancia es la decisión correcta

Este es, para mí, el aporte de diseño del paper, y merece desarrollarse porque es exactamente la clase de decisión que se reconoce desde la ingeniería de software.

Scaper mantiene **dos objetos distintos**:

1. La **especificación probabilística**: una descripción de una *familia* de paisajes sonoros. Un objeto pequeño, declarativo, con distribuciones en vez de valores.
2. La **especificación instanciada**: la misma estructura con **valores concretos ya muestreados** para todas las propiedades. El paper la llama textualmente una **"receta" (recipe) para generar el audio** (Sección 2).

Y el audio es el tercer objeto, derivado de (2).

La separación produce cinco propiedades que un generador monolítico no tiene:

- **Una descripción compacta genera un dataset arbitrariamente grande.** La especificación de URBAN-SED cabe en unas veinte líneas de Python; el dataset pesa casi 30 horas de audio. La descripción es el artefacto que se versiona, se revisa y se comparte; el audio es caché.
- **La instancia es inspeccionable y editable antes de renderizar.** Es lo que hace posible el truco del barrido de SNR de la Sección 4: se toman 2000 instancias, se edita **un campo** en cada una, se re-renderiza. Sin la capa intermedia habría que re-muestrear, y al re-muestrear cambiarían también todas las demás variables — el experimento controlado se perdería.
- **El muestreo se separa del rendering.** Uno es barato y determinista dado un `random_state`; el otro es caro (SoX, I/O de disco). Se pueden instanciar millones de recetas, filtrarlas o auditarlas, y renderizar solo las que interesan. El API expone esto con el flag `no_audio=True` de `generate()`.
- **La reproducibilidad es exacta y no probabilística.** Guardar la semilla te da reproducibilidad *si* la versión de la librería, el orden de las llamadas y el contenido del soundbank no cambian. Guardar la **instancia** te da reproducibilidad aunque el generador cambie, porque ya no hay nada que muestrear.
- **La misma abstracción cubre generación y aumentación.** Agregar un evento a un paisaje existente es agregar una `event specification` a una instancia y re-renderizar. No hace falta un segundo código-camino.

Si tú vienes del mundo de los sistemas de datos, esto es la distinción entre **DDL y datos**, o entre un *query plan* y su *result set*, o —el paralelo más cercano— entre un **generador de datos de prueba parametrizado** y los *fixtures* concretos que produce. Todo el mundo que ha mantenido *fixtures* a mano sabe por qué la parametrización gana. Lo interesante es que en ML esa lección aún no está internalizada: la enorme mayoría de los datasets se distribuyen como bolsas de archivos sin la receta que los produjo, y por eso son inmodificables, no auditables y no extensibles.

El diagrama de bloques (Figura 1) resume el pipeline con cinco etapas dentro del bloque de instanciación y generación:

```
SOUNDBANK (foreground / background)  ─┐
                                      ├──▶  SELECT PARAMETERS ──▶ TRIM ──▶ TRANSFORM ──▶ NORMALIZE ──▶ COMBINE
EVENT SPECIFICATIONS (distribuciones) ─┘                                                                  │
                                                                                                          ├──▶ Soundscape 1 + Annotation 1
                                                                                                          ├──▶ Soundscape 2 + Annotation 2
                                                                                                          └──▶ ...   Soundscape N + Annotation N
```

El panel de `EVENT SPECIFICATION` de la Figura 1 muestra el ejemplo canónico, con la notación de conjuntos y distribuciones:

```
label         ∈ {car_horn, jackhammer, …}
source file   ∈ {1.wav, 2.wav, …}
source time   ∈ {0}
event time    ∈ N(5, 2)
event duration∈ U(0.5, 4.0)
SNR           ∈ U(6, 30)
pitch shift   ∈ U(-3, 3)
time stretch  ∈ U(0.8, 1.2)
```

Nótese que **cada salida es un par (audio, anotación)**, nunca audio suelto. Es la plomería de la que hablaba en la Sección 2.5.

---

## 5. La reproducibilidad y el archivo JAMS

### 5.1. Los dos formatos de salida

Scaper emite la anotación en **dos** formatos (Sección 2), y la coexistencia es deliberada:

**(a) Texto plano tabular.** "Un archivo de texto simple **separado por espacios** con tres columnas para el onset, offset y etiqueta de cada evento sonoro." Justificación explícita: "Este formato es útil para inspeccionar rápidamente los eventos de un paisaje sonoro y **puede cargarse directamente en software como Audacity** para ver las etiquetas junto con el archivo de audio."

Es el formato de *labels* de Audacity, un estándar de facto en la comunidad. Que un investigador pueda arrastrar el `.txt` sobre la forma de onda y **ver** las anotaciones alineadas en tres segundos es una decisión de usabilidad barata y de alto retorno: convierte la verificación visual en un gesto, no en un script.

(Detalle menor de discrepancia: el paper dice "separado por espacios", pero la implementación actual usa `txt_sep='\t'` por defecto en `generate()`, es decir tabulaciones — que es lo que Audacity efectivamente espera. Ver Sección 13.)

**(b) JAMS.** El formato serio.

### 5.2. Qué es JAMS

**JAMS** = *JSON Annotated Music Specification*, de Humphrey, Salamon, Nieto, Forsyth, Bittner y Bello, ISMIR 2014 [14]. Nótese la superposición de autores: Salamon y Bello firman ambos papers. Scaper no adopta un formato externo; adopta **su propio formato previo**, lo que explica el ajuste tan bueno entre ambos.

JAMS es, en esencia, un contenedor JSON para anotaciones de audio con tres propiedades relevantes aquí:

1. **Múltiples anotaciones sobre el mismo audio, coexistiendo.** Un archivo JAMS contiene una lista de `annotations`, cada una con su propio `namespace` (el tipo de anotación: acordes, *beats*, `sound_event`, etc.), su `annotation_metadata` (quién o qué la produjo, con qué herramienta, versión, corrector) y sus `data` (la lista de observaciones $(\text{time}, \text{duration}, \text{value}, \text{confidence})$). Que el modelo de datos admita **varios anotadores discrepando sobre el mismo archivo** es exactamente lo que se necesita en un dominio donde el desacuerdo es estructural (Sección 2.3).
2. **Metadata estructurada y sin límite.** El paper: JAMS "soporta almacenar metadata de archivo **ilimitada y estructurada**". Cada anotación tiene un campo `sandbox` de forma libre.
3. **JSON.** Legible, versionable en git, parseable en cualquier lenguaje sin una librería especializada, y diffable. No es poco: un formato binario habría matado la mitad de las ventajas.

### 5.3. Lo que Scaper mete adentro

La frase clave del paper (Sección 2):

> "Scaper explota esto para almacenar **tanto la especificación probabilística como la instanciada** de cada evento sonoro. Esto significa que (asumiendo que se tiene acceso al *soundbank* original) **Scaper puede reconstruir completamente el audio de un paisaje sonoro a partir de su anotación JAMS**."

Y en la Sección 1:

> "facilitando una reconstrucción completa del paisaje sonoro desde su anotación JAMS, así como **soportando la manipulación de la anotación JAMS para generar variantes de paisajes sonoros existentes**."

Lo que se guarda, entonces, es una jerarquía de tres niveles en un solo archivo:

| Nivel | Contenido | Para qué sirve |
|---|---|---|
| Especificación probabilística | Las *distribution tuples* originales | Saber de qué familia salió este ejemplo; regenerar hermanos distintos |
| Especificación instanciada | Los valores concretos muestreados | Regenerar **este** ejemplo exacto; editarlo |
| Anotación | $(\text{onset}, \text{offset}, \text{label})$ por evento | Entrenar y evaluar |

Y una propiedad derivada que el paper computa y guarda explícitamente (Sección 3):

> "La polifonía máxima es **calculada automáticamente por Scaper durante la generación y almacenada en la anotación JAMS**. Esto nos permitirá evaluar fácilmente el desempeño del modelo como función de la polifonía máxima."

Ese es el patrón completo: **precomputar en el momento de generación las covariables que después vas a querer para estratificar el análisis, y persistirlas junto al dato**. Calcular la polifonía máxima *a posteriori* desde el audio sería imposible; desde la anotación sería posible pero requeriría un pase adicional y una definición que podría divergir de la del generador. Guardarla al generar la vuelve canónica y gratis.

### 5.4. Por qué esto importa fuera del audio

Aquí es donde el paper deja de ser sobre audio. El patrón que implementa es reconocible en cualquier disciplina que produzca datos sintéticos, y está mal resuelto en casi todas ellas.

El estado habitual de un dataset sintético en la literatura es: alguien escribe un script *ad hoc*, lo corre, publica los archivos resultantes, y el script se pierde o queda sin publicar. El propio paper lo señala con nombre y apellido al comparar con el segundo dataset más grande de su categoría (Sección 3):

> "El segundo más grande, **TUT-SED synthetic 2016** (también sintetizado) [26], dura aproximadamente 9.5 horas y, hasta donde sabemos, **fue generado usando scripts ad-hoc que no están disponibles públicamente**."

Ese dataset es perfectamente utilizable para entrenar y evaluar. Lo que no puedes hacer con él es **preguntarle nada**: no puedes generar más, no puedes generar una variante con una sola condición alterada, no puedes auditar si hay un sesgo en cómo se eligieron las fuentes, no puedes reproducirlo si el link se cae. Es un artefacto opaco.

Scaper hace explícito el contraste (Sección 3):

> "URBAN-SED está disponible gratuitamente en línea, y **por reproducibilidad todos los scripts usados para generarlo, así como para correr los experimentos de machine learning** reportados en la sección siguiente, **también están disponibles en línea**."

La jerarquía de reproducibilidad que se desprende, ordenada de peor a mejor:

| Nivel | Qué se publica | Qué puedes hacer |
|---|---|---|
| 0 | Solo el audio | Entrenar y evaluar. Nada más. |
| 1 | Audio + anotaciones | Lo anterior, y verificar métricas. |
| 2 | + el script generador | Regenerar si sobrevive el entorno y el soundbank. Frágil. |
| 3 | + la **instancia serializada por ejemplo** (JAMS) | Regenerar **exactamente**, **editar una variable y re-renderizar**, auditar la distribución real. |
| 4 | + la **especificación probabilística** | Todo lo anterior, y **generar hermanos nuevos** de la misma familia. |

Scaper opera en el nivel 4, y el experimento de SNR de la Sección 4 del paper es la demostración de que el nivel 3 no es cosmético: **habilita una clase de experimento que sin él no existe**.

Traducido a un vocabulario más familiar: la anotación JAMS de Scaper es a un paisaje sonoro lo que una **migración versionada** es a un esquema de base de datos, o lo que un **manifiesto de build reproducible** es a un binario. El artefacto derivado deja de ser la fuente de verdad; lo es la receta. Y la consecuencia práctica es la misma en los tres casos: **puedes tirar el artefacto y reconstruirlo**, lo que convierte 30 horas de WAV en caché en vez de en patrimonio.

La única dependencia externa que rompe el cierre es la que el propio paper señala entre paréntesis: *"assuming one has access to the original soundbank"*. El JAMS referencia archivos fuente por ruta; si el soundbank cambia o desaparece, la reconstrucción falla. Es la misma clase de fragilidad que un `package-lock.json` sin *registry*, o un `Dockerfile` cuya imagen base fue borrada. La solución canónica —fijar el soundbank con un identificador de contenido, un hash— no está en el paper ni en la librería.

---

## 6. Los detalles de la mezcla

### 6.1. El pipeline de renderizado

Según la Figura 1, la instanciación y generación consta de cinco etapas ordenadas:

1. **SELECT PARAMETERS** — muestrear de las distribuciones para producir la instancia.
2. **TRIM** — recortar cada archivo fuente según `source_time` y `event_duration`.
3. **TRANSFORM** — aplicar `pitch_shift` y `time_stretch` a cada evento **individualmente**.
4. **NORMALIZE** — ajustar los niveles según la SNR especificada.
5. **COMBINE** — sumar todo en la línea de tiempo.

Todo el procesamiento de audio se hace con **`pysox`** [23] (Bittner, Humphrey y Bello, ISMIR 2016 *Late Breaking Demo*), es decir, envolviendo SoX. Otra decisión de linaje interno del MARL, y una decisión sólida: SoX es una implementación madura y probada de resampling, pitch shift, stretch y normalización, y no hay razón para reimplementarla.

### 6.2. La SNR: por qué LUFS y no picos

Este es el detalle técnico más cuidado del paper, y el que más se cita mal. Textual (Sección 2):

> "Un aspecto de la generación que requiere cuidado especial es el manejo de los valores de SNR. En particular, la **simple normalización por pico no garantiza que dos sonidos normalizados al mismo nivel se perciban como igualmente sonoros**. Para sortear esto, Scaper usa **Loudness Units relative to Full Scale (LUFS)** [24], una medida estándar de sonoridad percibida usada en radio, televisión y transmisión por Internet. Así, si un evento se especifica con una SNR de 6, significa que estará **6 LUFS por encima del nivel del fondo**."

Desarrollemos por qué esto importa, porque es la diferencia entre una SNR que significa algo y una que no.

La normalización por pico fija $\max_t |x(t)|$. Es trivial de computar y es lo que hace la mayoría del código casero. El problema es que el pico no tiene relación estable con la sonoridad percibida, y la desconexión es brutal precisamente para las clases de UrbanSound8K:

- Un `gun_shot` es un impulso: pico altísimo, energía total baja, duración de decenas de milisegundos. Normalizado por pico, suena **más débil** de lo que su número sugiere.
- Un `air_conditioner` es casi estacionario: factor de cresta bajo, energía sostenida. Normalizado por pico, suena **mucho más fuerte**.

Normalizados ambos al mismo pico, la diferencia de sonoridad percibida puede ser de 15–20 dB. Una "SNR" definida sobre picos no es comparable entre clases y, por lo tanto, **no es una variable experimental utilizable**: al barrerla estarías barriendo simultáneamente la composición de clases.

LUFS (definido por la recomendación ITU-R BS.1770, cuya adopción europea es la referencia [24] de Grimm, Van Everdingen y Schöpping en el SMPTE Motion Imaging Journal) resuelve esto con tres piezas:

1. Un **filtro de ponderación K** aplicado a la señal, que aproxima la respuesta en frecuencia del sistema auditivo y el efecto de la cabeza: atenúa graves y realza agudos respecto de la energía cruda.
2. Medición de **energía media** sobre bloques solapados de 400 ms, no del pico.
3. Un **gating** que descarta los bloques silenciosos, para que los silencios no diluyan la medida.

El resultado es una escala logarítmica (en dB respecto de *full scale*) que correlaciona bien con la sonoridad percibida, y donde **una diferencia de 6 LUFS entre dos señales cualesquiera significa aproximadamente lo mismo perceptualmente**, sin importar su contenido espectral o temporal. Esa invarianza es la propiedad que convierte la SNR en una variable manipulable de verdad.

En URBAN-SED, el fondo se normaliza a **$-50$ LUFS** (Sección 3) y las SNR de los eventos se muestrean uniformemente en $[6, 30]$ dB, es decir, los eventos viven entre $-44$ y $-20$ LUFS. El *headroom* respecto de 0 dBFS es amplio, lo que reduce la probabilidad de saturación al sumar.

Formalmente, la ganancia aplicada a cada evento se deduce de
$$\text{SNR}_i = L_{e_i} - L_{b}$$
donde $L$ denota sonoridad integrada en LUFS. Fijado $L_b = -50$ y sorteado $\text{SNR}_i$, la ganancia lineal es
$$g_i = 10^{(L_b + \text{SNR}_i - L_{e_i}^{\text{orig}})/20}$$
con $L_{e_i}^{\text{orig}}$ la sonoridad medida del evento ya recortado y transformado. El paper no escribe esta fórmula; es la lectura directa de lo que describe en prosa.

**Un matiz de notación que el propio paper mezcla:** en la Sección 2 la SNR se define en LUFS ("6 LUFS por encima del nivel del fondo"); en la Sección 3 se dice que "la SNR se muestrea uniformemente entre **6–30 dB**". Ambas escalas son logarítmicas y la magnitud es la misma, pero la unidad correcta según la propia definición del paper es LUFS. Es una inconsistencia de redacción, no de implementación (ver Sección 13).

### 6.3. Polifonía y solapamiento

Scaper **no modela el solapamiento como una restricción**: lo deja emerger del muestreo de los tiempos de inicio. Si dos eventos se sortean con `event_time` cercanos y duraciones que se cruzan, se solapan; punto. La suma es aditiva y no hay ningún tratamiento especial.

El paper introduce la métrica que resume esto (Sección 3):

> "Definimos la **polifonía máxima** de un paisaje sonoro como la **mayor polifonía de eventos sonoros observada en cualquier punto del tiempo** del paisaje."

Es decir, $\max_t |\{i : \tau_i \le t < \tau_i + d_i\}|$. La cuenta incluye eventos de la misma clase.

Y la forma en que URBAN-SED induce variedad de polifonía es indirecta y elegante (Sección 3): en vez de controlar el solapamiento, se controla la **distribución de los tiempos de inicio**, muestreando cada paisaje de una de tres opciones:

| Distribución de `event_time` | Forma | Efecto sobre la polifonía |
|---|---|---|
| $\mathcal{U}(0, 10)$ | Uniforme sobre los 10 s | Eventos dispersos, poco solapamiento |
| $\mathcal{N}(5, 2)$ | Unimodal centrada | Aglomeración en el medio, más solapamiento |
| $\tfrac{1}{2}\mathcal{N}(3,2) + \tfrac{1}{2}\mathcal{N}(7,2)$ | Bimodal | Dos racimos |

Textual: "Al usar estas distribuciones para los tiempos de inicio de los eventos obtenemos una variedad de paisajes sonoros, algunos en los que los eventos están dispersos y otros en los que tienden a estar más agrupados en el tiempo, llevando en consecuencia a un **mayor grado de solapamiento**".

Combinado con un número de eventos muestreado de una uniforme discreta en $[1, 9]$, el resultado observado es una polifonía máxima que abarca **de 1 a 7** en el dataset final (Sección 3).

**El paper reconoce esto como una limitación explícita** (Sección 6):

> "Actualmente Scaper **no soporta controlar explícitamente ciertas características de la escena** como la polifonía máxima (o promedio), y planeamos agregar esta funcionalidad en el futuro."

Es una limitación real y bien identificada. La polifonía es una **variable de salida** del proceso, no de entrada: tú la mides *a posteriori* (por eso se guarda en el JAMS), pero no puedes pedir "genérame 500 paisajes con polifonía máxima exactamente 4". Para conseguirlo hay que generar de más y filtrar por la propiedad guardada — rechazo por muestreo, funcional pero ineficiente y con la distribución condicional sesgada de forma difícil de razonar.

### 6.4. Las transformaciones por evento

`pitch_shift` (semitonos, no afecta duración) y `time_stretch` (factor, no afecta pitch) se aplican **a cada evento individualmente**, y esa individualidad es una de las tres diferencias declaradas respecto de Lafay et al. (Sección 1):

> "Scaper soporta aplicar transformaciones de audio como *pitch shifting* y *time stretching* **individualmente a cada evento sonoro**, aumentando significativamente el rango posible y la variabilidad de los paisajes sonoros generados."

El argumento combinatorio es directo. Con un *soundbank* de $M$ clips y transformaciones aplicadas globalmente, tienes $M \times K$ variantes de material. Aplicadas por evento y componiendo $N$ eventos por escena, el espacio de escenas distintas crece como $(M K)^N$ multiplicado por las combinaciones de tiempos, duraciones y niveles. Un soundbank pequeño alcanza para un dataset grande — con la salvedad importante de que **la diversidad combinatoria no es diversidad acústica** (ver Sección 10).

Hay además un efecto no comentado por el paper: `time_stretch` altera la duración del evento, y por lo tanto **el offset de la anotación**. La anotación debe reflejar la duración *post-stretch*, no la original. Scaper lo hace correctamente (la duración instanciada es la que se anota), pero es exactamente el tipo de bug silencioso que un script *ad hoc* introduce sin que nadie lo note: un desfase sistemático entre etiqueta y audio que degrada el entrenamiento sin producir ningún error visible. Que la librería lo maneje de forma centralizada y probada es parte de su valor.

### 6.5. Normalización y clipping: lo que el paper deja abierto

Aquí hay que ser honesto sobre el alcance del texto. El paper:

- **Sí especifica** que existe una etapa `NORMALIZE` antes de `COMBINE` (Figura 1).
- **Sí especifica** que la normalización de niveles se hace en LUFS respecto del fondo (Sección 2).
- **No especifica** qué ocurre cuando la suma $b(t) + \sum_i g_i e_i(t-\tau_i)$ excede el rango $[-1, 1]$.

El riesgo es real y aritméticamente inevitable: con 9 eventos simultáneos a SNR alta sobre un fondo, la suma puede saturar. Y el *clipping* no es un detalle cosmético — introduce distorsión armónica de banda ancha que es **exactamente el tipo de artefacto sistemático que un CNN puede aprender a usar como atajo**: si los paisajes de polifonía alta saturan más, el modelo puede inferir la polifonía desde la distorsión en vez de desde el contenido.

El paper no lo discute. La mitigación de facto en URBAN-SED es el generoso *headroom* del diseño: fondo a $-50$ LUFS y eventos a lo sumo a $-20$ LUFS, lo que deja 20 dB antes de fondo de escala. Sumar nueve señales incoherentes a $-20$ LUFS da del orden de $-20 + 10\log_{10}(9) \approx -10.5$ LUFS de sonoridad integrada, todavía bajo el techo, aunque los picos instantáneos pueden acercarse más.

La librería sí resolvió el problema después: el `generate()` actual expone `fix_clipping=False` y `peak_normalization=False` como parámetros explícitos (verificado en `scaper/core.py` de la rama `master`). Ninguno de los dos aparece en el paper de 2017 — son agregados posteriores, presumiblemente motivados justo por este problema. Lo señalo como una **evolución de la librería más allá del paper**, no como algo que el paper diga.

Otros aspectos que el paper deja sin especificar y que conviene tener presentes al usar la herramienta:

- **Fundidos (fades) en los bordes de los eventos.** Un recorte abrupto en `source_time` o al final de `event_duration` produce un clic (discontinuidad en la forma de onda). El paper no menciona ventanas de fundido. Si no las hay, cada evento del dataset trae un transitorio artificial en su onset y su offset — que es, otra vez, un atajo perfecto para un detector de eventos, porque el clic marca exactamente el límite que se quiere predecir.
- **Frecuencia de muestreo y resampleo.** El paper no dice a qué $f_s$ opera. UrbanSound8K trae clips con las tasas nativas heterogéneas de Freesound, así que necesariamente hay resampleo en alguna parte; SoX lo maneja, pero la política no está documentada en el texto.
- **Canales.** No se menciona mono vs estéreo, ni panorámica. La ausencia de posicionamiento espacial es una limitación de fondo (Sección 8).
- **Reverberación.** El paper de 2017 **no la menciona en absoluto**. La librería actual expone un parámetro `reverb` en `generate()`, que aplica el efecto `reverb` de SoX — una reverberación algorítmica global, no una convolución con una respuesta al impulso medida. Detalle importante para la Sección 8.

---

## 7. Los experimentos del paper

El paper presenta dos casos de uso. El primero (Secciones 3 y 4) es el que más se cita; el segundo (Sección 5) es el más conceptualmente interesante.

### 7.1. URBAN-SED: el dataset

Construcción, paso a paso (Sección 3):

| Elemento | Especificación |
|---|---|
| Soundbank | **UrbanSound8K** [25], ~1000 clips por cada una de 10 fuentes urbanas; cada clip contiene una sola fuente |
| Splits | Se respetan los **10 folds estratificados** de UrbanSound8K: **folds 1–6** → 6000 paisajes de entrenamiento; **folds 7–8** → 2000 de validación; **folds 9–10** → 2000 de test |
| Duración | 10 s por paisaje |
| Fondo | Clip de 10 s de **ruido browniano**, el mismo para todos, normalizado a **$-50$ LUFS** |
| Número de eventos | Uniforme discreta en $[1, 9]$ |
| `label` | Elegida al azar entre las 10 clases |
| `source_file` | Al azar entre los clips que corresponden a la etiqueta |
| `source_time` | **Siempre 0**, "para asegurar que no perdamos el onset de un evento" |
| `event_time` | Una de tres: $\mathcal{U}(0,10)$, $\mathcal{N}(5,2)$, o bimodal $\mathcal{N}(3,2)/\mathcal{N}(7,2)$ |
| `event_duration` | $\mathcal{U}(0.5, 4.0)$ s (todos los clips de UrbanSound8K duran a lo sumo 4 s), acotada a la duración de la fuente si esta es menor |
| `snr` | $\mathcal{U}(6, 30)$ |
| `pitch_shift` | $\mathcal{U}(-3, 3)$ semitonos |
| `time_stretch` | $\mathcal{U}(0.8, 1.2)$ |

Resultado (Sección 3):

- **10.000 paisajes sonoros**
- **casi 30 horas** de audio
- **cerca de 50.000 eventos anotados**
- **polifonías máximas entre 1 y 7**
- "Esto lo convierte en el **dataset con etiquetas fuertes más grande disponible para SED**, aunque por supuesto podríamos hacerlo arbitrariamente más grande o más pequeño."

La coletilla final es la tesis del paper en cuatro palabras. El tamaño dejó de ser una propiedad del dataset para ser un **parámetro**.

La comparación de escala que ofrece el paper es contundente: 30 horas frente a las **9.5 horas** de TUT-SED synthetic 2016 [26], y frente a los **78 minutos** de TUT Sound Events 2016 real (Sección 1). Es decir, **más de 20× el dataset real más usado de su momento**, con anotaciones exactas en vez de aproximadas.

Nota sobre la herencia de los folds: al derivar los splits de los folds de UrbanSound8K, URBAN-SED hereda la garantía de que **ningún clip fuente aparece en dos splits**. Sin eso, el mismo `dog_bark.wav` podría aparecer en un paisaje de entrenamiento y en uno de test bajo distinto pitch y offset, y el modelo podría memorizar la grabación específica. Es una decisión correcta y silenciosa que un script *ad hoc* olvidaría con facilidad.

### 7.2. Los dos modelos comparados

| | **CRNN** | **CNN** |
|---|---|---|
| Origen | Çakir et al. [26], la arquitectura que sus autores identificaron como la mejor sobre el *development set* de TUT-SED-2016 | Adaptación de Salamon y Bello [27] (*Deep CNNs and data augmentation for environmental sound classification*, IEEE SPL 2017) |
| Parámetros | **743k** | Original: **241k**. Adaptado: **720k** |
| Adaptaciones | Ninguna; los autores verificaron su implementación entrenando/evaluando sobre TUT-SED-2016, obteniendo "resultados casi idénticos a los reportados en el paper" | Softmax final → **sigmoides** (multi-clase → multi-etiqueta); filtros convolucionales aumentados a **64** por capa para igualar la capacidad del CRNN; **batch normalization** [28] a la salida de las capas convolucionales; entrada reducida a **1 s**; max pooling posterior a las convoluciones reducido a **(2,2)** |
| Resolución de salida | **Nivel de frame** (p. ej. 20 ms) | **1 s** |

La justificación de la resolución de 1 s para el CNN (Sección 4) es pragmática y honesta: "una resolución temporal de 1 s sería suficiente para aplicaciones de monitoreo de sonido urbano, ha sido usada en trabajo reciente con resultados prometedores [5], y resulta en un modelo **significativamente más rápido de entrenar**". La referencia [5] es Hershey et al., *CNN architectures for large-scale audio classification*, ICASSP 2017 — el paper de AudioSet/VGGish.

Y el control correspondiente está en la **nota al pie 5**, que es exactamente el tipo de verificación que debería ser obligatoria y casi nunca se hace:

> "Evaluar el CRNN a resoluciones temporales más finas (100 ms y 20 ms) **no resultó en F-measures globales ni por clase más altas** comparado con 1 s."

Es decir: la comparación a 1 s no perjudica al modelo que podría operar más fino. El *handicap* aparente no lo es. Sin esa nota, toda la Figura 2 sería impugnable.

**Entrenamiento** (Sección 4): Keras [17], optimizador **Adam** [29], pérdida de **entropía cruzada binaria**, **300 épocas** con criterio de parada temprana de **100 épocas sin mejora** en la F-measure basada en segmentos [30] calculada a **1 s** con `sed_eval` [30]. Una época = un pase completo sobre el conjunto de entrenamiento (6000 paisajes; validación y test, 2000 cada uno).

### 7.3. Resultado 1: comparación global y por clase (Figura 2)

La Figura 2 muestra la F-measure basada en segmentos de 1 s para ambos modelos, desglosada por clase y en global. **Las cifras siguientes son lecturas aproximadas del gráfico de barras** — el paper no publica una tabla numérica, así que no puedo darlas con más precisión de la que permite el eje:

| Clase | CRNN (≈) | CNN (≈) |
|---|---|---|
| `air_conditioner` | 0.47 | 0.34 |
| `car_horn` | 0.52 | 0.69 |
| `children_playing` | 0.54 | 0.49 |
| `dog_bark` | 0.57 | 0.52 |
| `drilling` | 0.56 | 0.58 |
| `engine_idling` | 0.59 | 0.51 |
| `gun_shot` | 0.69 | 0.50 |
| `jackhammer` | 0.58 | 0.75 |
| `siren` | 0.63 | 0.64 |
| `street_music` | 0.56 | 0.59 |
| **OVERALL** | **≈0.57** | **≈0.57** |

La lectura del paper (Sección 4): "en general los dos modelos rinden de forma comparable, con el **CRNN notablemente mejor en aire acondicionado y disparos**, mientras el **CNN rinde mejor en bocinas de auto y martillos neumáticos**."

Vale la pena notar el patrón, que el paper no comenta: el CRNN gana en `gun_shot` (evento impulsivo, brevísimo) y en `air_conditioner` (estacionario largo y difuso) — los dos extremos de la escala temporal. Ambos son casos donde la resolución de frame y la memoria recurrente ayudan: uno porque un evento de 200 ms cae dentro de un único segmento de 1 s del CNN y se diluye, el otro porque requiere integrar evidencia sobre varios segundos. El CNN gana en `jackhammer` y `car_horn`, que son eventos con textura espectral fuerte y bien delimitada dentro de una ventana de 1 s. Es una lectura consistente, aunque el paper no la hace.

El punto metodológico es que **el empate global (≈0.57 vs ≈0.57) oculta dos perfiles de error muy distintos**. Ese es, en el fondo, el argumento del paper: los agregados esconden estructura, y la síntesis controlada es lo que permite recuperarla.

### 7.4. Resultado 2: degradación con la polifonía (Figura 3, arriba)

El barrido estrella. Se toma el modelo CNN (elegido porque los dos rinden comparable) y se agrupa el conjunto de test por polifonía máxima, de 1 a 7. Lecturas aproximadas de la Figura 3 (arriba):

| Polifonía máxima | Precision (≈) | Recall (≈) | F-measure (≈) |
|---|---|---|---|
| 1 | 0.68 | 0.64 | 0.66 |
| 2 | 0.67 | 0.54 | 0.60 |
| 3 | 0.70 | 0.50 | 0.58 |
| 4 | 0.70 | 0.46 | 0.56 |
| 5 | 0.69 | 0.42 | 0.52 |
| 6 | 0.71 | 0.42 | 0.52 |
| 7 | **0.78** | **0.41** | 0.54 |

La interpretación del paper es precisa (Sección 4):

> "Como uno podría esperar, notamos que la F-measure **declina gradualmente** a medida que aumenta la polifonía máxima, pero **más interesante todavía**, vemos que es porque **el recall declina, mientras que la precisión permanece estable (e incluso sube)**. Esto sugiere que a medida que más eventos sonoros se solapan, el modelo es **cada vez más propenso a detectar solo un subconjunto** de los eventos, sin embargo **permanece igualmente preciso**."

Este es el resultado que justifica todo el paper, y conviene entender por qué.

Que la F-measure baje con la polifonía es trivialmente esperable — cualquiera lo habría apostado. **La descomposición no lo es.** La caída podría haber venido de cualquiera de dos mecanismos con implicancias opuestas:

- **Hipótesis A (confusión):** con más solapamiento el modelo se confunde y **alucina** clases que no están, porque los espectros se superponen y aparecen patrones espurios. Predicción: la **precisión** cae.
- **Hipótesis B (enmascaramiento):** con más solapamiento el modelo **pierde** eventos, porque unos enmascaran a otros, pero lo que sí detecta lo detecta bien. Predicción: el **recall** cae y la precisión se mantiene.

Los datos apoyan inequívocamente B, y con un refinamiento: la precisión no solo se mantiene, **sube monótonamente** hacia el extremo (≈0.68 en polifonía 1 → ≈0.78 en polifonía 7). Eso es coherente con un modelo que se vuelve más conservador bajo incertidumbre: dispara solo sobre la evidencia más fuerte, que es también la más confiable.

Las consecuencias de ingeniería son concretas y accionables, y ninguna se habría podido derivar de una métrica agregada:

- Si tu aplicación es **monitoreo de ruido** y necesitas cobertura, el modo de falla dominante en escenas densas es la **omisión**. Bajar el umbral de decisión es la palanca correcta, y el margen de precisión disponible (0.78) sugiere que hay espacio para hacerlo.
- Si tu aplicación tolera omisiones pero no falsas alarmas (**vigilancia**), el modelo ya está operando en el régimen deseado sin ajuste.
- El problema a atacar arquitectónicamente no es la discriminación entre clases, sino la **separación de fuentes solapadas** — lo que apunta a modelos con capacidad de desenredado, no a más capacidad de clasificación.

Y hay que insistir: **el eje de este gráfico solo existe porque la polifonía se conoce con exactitud**. Con datos reales tendrías que estimarla desde una anotación humana que, precisamente en escenas densas, es la menos confiable (lo que la Sección 5 del paper demuestra). El instrumento de medición se degradaría junto con lo medido.

### 7.5. Resultado 3: el barrido de SNR (Figura 3, abajo)

El experimento *ceteris paribus* descrito en la Sección 3.2 de este análisis. Ocho versiones del conjunto de test, idénticas salvo por la SNR, obtenidas editando los JAMS y re-renderizando. Lecturas aproximadas de la Figura 3 (abajo):

| Rango de SNR | Precision (≈) | Recall (≈) | F-measure (≈) |
|---|---|---|---|
| 6–9 | **0.77** | 0.43 | 0.55 |
| 9–12 | 0.76 | 0.46 | 0.57 |
| 12–15 | 0.74 | 0.48 | 0.58 |
| 15–18 | 0.72 | 0.50 | 0.59 |
| 18–21 | 0.70 | 0.51 | 0.59 |
| 21–24 | 0.68 | 0.52 | 0.59 |
| 24–27 | 0.66 | 0.53 | 0.59 |
| 27–30 | 0.63 | **0.54** | 0.58 |

La lectura del paper (Sección 4):

> "Vemos un efecto interesante: a medida que la SNR aumenta, la precisión y el recall del modelo exhiben **comportamientos opuestos**: el recall sube, lo que significa que el modelo detecta correctamente más de los eventos, pero esto lleva a un **número aumentado de falsos positivos**, bajando la precisión. El efecto neto es una **F-measure relativamente estable**, pero a través de este experimento **ahora sabemos que esa estabilidad esconde un comportamiento del modelo bastante distinto** a medida que la SNR cambia."

La última frase es la moraleja del paper entero, y la citaría en cualquier discusión sobre protocolos de evaluación: **una métrica plana no es evidencia de comportamiento estable**. La F-measure se mueve entre 0.55 y 0.59 sobre un rango de 24 dB de SNR — visualmente, una línea horizontal. Debajo, la precisión cae ~14 puntos y el recall sube ~11. Son **dos modelos distintos** en los extremos del rango, y el escalar agregado no lo registra.

Para alguien que diseña protocolos de evaluación, esto es la demostración empírica de por qué las métricas escalares agregadas son insuficientes y por qué el **desglose estratificado por covariable controlada** debería ser el estándar. Y por qué la síntesis con receta es la única forma barata de conseguir esas covariables.

(Anoto de paso que el mecanismo del aumento de falsos positivos con SNR alta es interesante y el paper no lo explica: probablemente eventos fuertes que "sangran" hacia segmentos adyacentes donde ya no están presentes —el detector sigue disparando por un segundo de más— o que enmascaran e inducen predicciones erróneas de clases con espectro similar. Es una hipótesis mía, no del paper.)

### 7.6. El cuarto experimento: Scaper como instrumento de medición de humanos (Sección 5)

Este caso de uso es el más conceptualmente elegante y el que menos se cita. El paper resume resultados de un estudio propio sobre *crowdsourcing* de anotaciones [31] (Cartwright, Seals, Salamon, Williams, Mikloska, MacConnell, Law, Bello y Nov, *"Seeing sound: Investigating the effects of visualizations and complexity on crowdsourced audio annotations"*, PACM HCI 2017).

El planteo del problema es circular y el paper lo desarma con precisión (Sección 5):

> "Dado que el objetivo del experimento era **evaluar la calidad de las etiquetas humanas, no es posible usar paisajes sonoros etiquetados por humanos como estímulos**."

Ahí está la trampa: para medir cuán bien anota un humano necesitas una referencia. Si la referencia la produjo otro humano, estás midiendo **acuerdo entre anotadores**, no exactitud, y no puedes distinguir un error del sujeto de una idiosincrasia del anotador de referencia. El paper lo dice tal cual: usando Scaper "aseguramos que cualquier efecto observado en el experimento se deba a **las habilidades de anotación de los sujetos y a las intervenciones experimentales**, y **no a una discrepancia subjetiva entre los sujetos y un 'anotador de referencia'**".

**Scaper rompe la circularidad**: proporciona una referencia que no es una opinión.

Escala del experimento (Sección 5): un soundbank de **90 eventos sonoros cuidadosamente curados** → **3000 paisajes sonoros** generados → **60 seleccionados** para cubrir un rango de complejidades → usados como estímulos. La razón 3000:60 es en sí misma una demostración del argumento: se genera 50× lo que se necesita y se selecciona el subconjunto que cubre uniformemente el espacio de condiciones. Con grabaciones reales, conseguir 60 clips que cubran uniformemente un eje de complejidad requeriría anotar cientos de horas primero.

Resultados (Figura 4), con la polifonía máxima agrupada en tres niveles: **nivel 0** = polifonía máxima 1, **nivel 1** = 2, **nivel 2** = 3 o 4. Métricas de precisión, recall y F-measure basadas en segmentos, calculadas a **100 ms** de resolución. El hallazgo (Sección 5):

> "Interesantemente, vemos que los anotadores humanos exhiben un **comportamiento similar a los modelos de machine learning**: a medida que el nivel de polifonía aumenta la F-measure decrece, **principalmente debido a una caída en el recall, mientras que la precisión permanece alta**. Los sujetos humanos, como la máquina al parecer, **pierden más eventos a medida que aumenta el grado de solapamiento, pero anotan con exactitud los eventos que sí reconocen**."

Y la conclusión que de ahí se saca:

> "Este es un resultado muy prometedor, que sugiere que **las etiquetas humanas para paisajes sonoros densos pueden considerarse confiables, aunque incompletas**."

Piensa en lo que esto significa para el diseño de un dataset. Si el error humano fuera de **precisión** (etiquetas incorrectas), el dataset estaría contaminado y no habría cómo arreglarlo salvo re-anotando. Si el error es de **recall** (eventos omitidos), el dataset es **correcto pero incompleto** — y eso es tratable: se puede modelar como supervisión parcial / *positive-unlabeled learning*, o mitigar agregando anotadores independientes (la unión de anotaciones aumenta el recall sin dañar la precisión). **Es un diagnóstico que cambia la estrategia de anotación**, y solo se pudo obtener porque hubo una referencia perfecta contra la cual medir.

El segundo hallazgo, sobre la interfaz: se compararon tres visualizaciones —**forma de onda**, **espectrograma** y **sin visualización**— y "comparando la distribución de los onsets y offsets anotados por humanos contra las anotaciones de Scaper, pudimos mostrar que **una visualización de espectrograma resultó en una mejora estadísticamente significativa en la exactitud temporal** de las anotaciones humanas" (Sección 5).

Resultado de HCI con consecuencia directa: la herramienta de anotación que use tu equipo debe mostrar el espectrograma. Y nuevamente, solo medible con *ground truth* exacto — la desviación de un onset humano solo tiene sentido respecto de un onset verdadero.

Este es el uso de Scaper que a mí me parece más subestimado: no como generador de datos de entrenamiento, sino como **instrumento de calibración del proceso de anotación humana**. Si vas a invertir en anotar 500 horas de audio real, gastar primero una semana en generar estímulos sintéticos, medir a tus anotadores contra ellos, y ajustar el protocolo y la interfaz, tiene un retorno evidente. Es control de calidad con patrón de referencia, exactamente como se calibra un instrumento de medición físico.

---

## 8. La brecha entre lo sintético y lo real

Esta es la sección de análisis propio, y la que conecta con la advertencia literal del slide de la clase:

> "Use of sound synthesis techniques is a convenient strategy. However, **performance on real data might be poor if a model is trained using only synthetic data. Need for finetuning on real data.**"

El paper es explícito y no defensivo al respecto (Sección 6):

> "Primero y principal, los paisajes sonoros generados, **incluso si suenan bastante realistas en algunos casos, no pueden abarcar la riqueza y complejidad de los paisajes sonoros reales**. Esto significa que si bien Scaper es útil para generar datasets tanto para entrenar modelos como para comparar su desempeño en función de características acústicas controladas, **no puede usarse como reemplazo de grabaciones del mundo real anotadas manualmente**, si queremos estimar cuán bien rendirá un modelo en un ambiente real."

Ahora bien, "no abarca la riqueza y complejidad" es una frase honesta pero vaga. Vale la pena descomponer **exactamente qué falta**, porque cada componente tiene una mitigación distinta.

### 8.1. La ausencia de acústica del espacio

El modelo generativo de Scaper es
$$x(t) = b(t) + \sum_i g_i\, e_i(t - \tau_i)$$
Es decir, **suma de señales**. Pero lo que un micrófono captura en un espacio real es
$$x(t) = \sum_i \big(h_i * s_i\big)(t - \tau_i) + n(t)$$
donde $h_i$ es la **respuesta al impulso** (RIR) del camino acústico entre la fuente $i$ y el micrófono, y $*$ es convolución.

Esa $h_i$ codifica:

- **El sonido directo**, atenuado por $1/r_i$ con la distancia y retardado por $r_i/c$.
- **Las reflexiones tempranas** (primeros ~50–80 ms): el patrón discreto de ecos de superficies cercanas —fachadas, suelo, mobiliario— que es la principal pista perceptual de **dónde está la fuente y qué tamaño tiene el espacio**.
- **La cola reverberante**: el decaimiento difuso, caracterizado por el $T_{60}$ (tiempo hasta $-60$ dB), que en una calle urbana con fachadas puede ir de 0.5 a 2 s y en un interior reflectante mucho más.

Tres consecuencias concretas de que $h_i$ no exista en un paisaje de Scaper:

1. **Los offsets sintéticos son abruptos; los reales no.** Un evento real no termina cuando la fuente deja de emitir, sino cuando su cola reverberante decae bajo el fondo. Un detector entrenado con offsets limpios aprende una noción de "fin de evento" que en audio real no ocurre nunca. Y como la métrica de SED evalúa offsets, esto es un desajuste directo entre el objetivo de entrenamiento y la tarea real.
2. **La reverberación es el "pegamento" perceptual de una escena.** Cuando todas las fuentes comparten la misma $h$ (mismo espacio, posiciones similares), el oyente las integra en una escena coherente. Cuando cada fuente trae la reverberación de un espacio distinto —que es exactamente el caso de un soundbank compilado de grabaciones heterogéneas—, la mezcla suena a *collage*. Y ese "sonar a collage" no es un juicio estético: es **información espectral y temporal explotable** por una red, discutida en 8.3.
3. **La reverberación degrada la separabilidad, y esa degradación es la tarea.** Al alargar cada fuente en el tiempo y difuminar sus transitorios, la reverberación aumenta el solapamiento efectivo mucho más allá del solapamiento nominal de los onsets. Un modelo entrenado sobre mezclas anecoicas nunca vio ese régimen.

**Lo que existe hoy y el paper no usa:** la librería actual tiene un parámetro `reverb` en `generate()` que aplica el efecto `reverb` de SoX. Es una reverberación **algorítmica y global** —un modelo paramétrico de sala aplicado a la mezcla— no una convolución por fuente con una RIR medida. Ayuda con (1) y algo con (2), pero no reproduce la relación entre reverberación y **posición de la fuente**, que es lo que da coherencia espacial. El paper de 2017 no menciona reverberación en ninguna parte.

### 8.2. La respuesta del micrófono y la cadena de captura

Un paisaje real pasa por una cadena física que Scaper no modela: la respuesta en frecuencia y la directividad del transductor, la ganancia y el ruido del preamplificador, el AGC (control automático de ganancia) si lo hay, el códec de compresión con pérdida, y la cuantización.

Esto importa por una razón muy concreta: **en despliegue real, esa cadena es fija**. Un sensor de la red de monitoreo urbano de NYU —el trabajo de Mydlarz, Salamon y Bello [1], citado en el propio paper— tiene un micrófono MEMS específico, con una coloración específica, en una carcasa específica, montado a una altura específica. Todo el audio de producción comparte esa firma. El audio sintético de Scaper hereda la firma **heterogénea de Freesound**, que es la unión de cientos de cadenas de captura distintas.

Un modelo entrenado con Scaper aprende a ser invariante a un conjunto de coloraciones que **no incluye** la del sensor de despliegue. Es *domain shift* de manual, de la variedad covariate shift.

Y aquí hay un detalle que corta en la dirección contraria y que vale reconocer: **la diversidad de fuentes heterogéneas también actúa como regularizador**. Un modelo forzado a reconocer un `car_horn` grabado con veinte micrófonos distintos aprende una representación menos dependiente del transductor que uno entrenado con un solo sensor. La heterogeneidad del soundbank es simultáneamente un problema de realismo y una forma de aumentación de dominio. Cuál de los dos efectos domina es empírico.

### 8.3. Los eventos aislados ya vienen con su propia acústica

Este es el punto más sutil y el más dañino, y merece detenerse.

Los clips de UrbanSound8K provienen de **Freesound**, es decir, son grabaciones de campo hechas por personas distintas, con equipos distintos, en lugares distintos. Cuando Scaper los recorta y los suma, **cada evento trae adherida la acústica de su grabación original**: su propio ruido de fondo residual, su propia reverberación, su propia coloración de micrófono, su propio nivel de piso de ruido.

La consecuencia es que un paisaje sintético de nueve eventos es una superposición de **nueve ambientes acústicos distintos** más un ruido browniano. Y eso deja una firma detectable.

Piensa en el mecanismo específico. Un evento recortado de una grabación de campo casi nunca es puro: en los milisegundos previos a su onset y posteriores a su offset hay **ruido de fondo de su grabación original**, distinto del ruido browniano. Al insertarlo en el paisaje, en el instante $\tau_i$ el nivel de piso de ruido y su color espectral **cambian abruptamente**, y vuelven a cambiar en $\tau_i + d_i$. Una CNN sobre espectrograma puede aprender a detectar **exactamente esa discontinuidad de fondo** —un salto en las bandas donde no hay energía del evento— y usarla como predictor del onset.

Y funcionaría perfecto. Con una F-measure alta y estable. Sobre datos sintéticos. Y colapsaría por completo sobre audio real, donde no existe tal discontinuidad porque el fondo es continuo.

Este es el modo de falla más peligroso del entrenamiento sintético, porque **es invisible en validación**: el conjunto de validación tiene el mismo artefacto que el de entrenamiento, así que la métrica no lo detecta. Solo aparece al evaluar sobre datos reales. Es la versión auditiva del clásico de visión donde el clasificador de tanques aprendió a detectar el clima.

Súmale los artefactos de la síntesis misma: posibles clics en los bordes de recorte (Sección 6.5), artefactos de fase del *pitch shift* y el *time stretch* de SoX (que en factores extremos producen el característico *smearing* del vocoder de fase), y la eventual distorsión por *clipping*. Cada uno es un canal lateral.

### 8.4. La independencia entre eventos no refleja el mundo

En URBAN-SED, cada evento se muestrea de forma **independiente e idénticamente distribuida**: etiqueta uniforme entre 10 clases, tiempo de inicio de una distribución fija, duración independiente, SNR independiente. Ninguna de estas cosas es cierta en el mundo:

- **Correlación entre clases (co-ocurrencia).** `jackhammer` y `drilling` van juntos: son la misma faena de construcción. `siren` implica tráfico, luego `engine_idling` y `car_horn`. `children_playing` ocurre en plazas, con `dog_bark`, y **casi nunca** con `jackhammer` (nadie lleva niños a jugar junto a una demolición). La matriz de co-ocurrencia real es fuertemente estructurada; la sintética es producto de marginales uniformes.
- **Correlación temporal.** Los eventos reales tienen **estructura secuencial y causal**: un auto frena, luego toca la bocina, luego acelera. Una sirena se acerca y se aleja (con efecto Doppler y curva de nivel). Un martillo neumático opera en ráfagas periódicas con pausas. Scaper tira tiempos de inicio de una distribución sin memoria; no hay *proceso*, solo *puntos*.
- **Correlación entre etiqueta y parámetros.** La SNR de un evento no es independiente de su clase: una sirena de emergencia es intrínsecamente más fuerte que un ladrido, y en la realidad se percibe a mayor distancia. En URBAN-SED, sirena y ladrido comparten la misma $\mathcal{U}(6,30)$.
- **Prior de clase.** Las 10 clases son equiprobables en URBAN-SED. En una calle real, `engine_idling` y `air_conditioner` están presentes casi permanentemente y `gun_shot` es rarísimo. El desbalance real es de órdenes de magnitud.

¿Por qué esto degrada el desempeño real? Porque **un modelo aprende y explota las correlaciones de sus datos de entrenamiento aunque nadie se lo pida**. Si en el entrenamiento las clases son independientes, el modelo aprende un detector que trata cada clase por separado y **no aprovecha el contexto**. Sobre datos reales, donde el contexto es informativo (oír un martillo neumático debería subir el prior de un taladro), ese modelo está dejando información sobre la mesa. Peor: si el modelo *sí* aprendió alguna correlación espuria de la síntesis, la aplicará donde no corresponde.

Es simétrico al problema de las etiquetas: la síntesis te da control perfecto sobre las marginales, y a cambio te obliga a especificar la estructura conjunta — que es justamente lo que nadie sabe cómo especificar bien.

**Mitigación disponible:** el `choose_weighted` de la librería actual permite al menos priors de clase realistas. Las correlaciones condicionales requerirían muestrear la especificación desde un modelo de escena (una cadena de Markov sobre estados de escena, por ejemplo, o un modelo gráfico de co-ocurrencia estimado de datos reales), lo cual el API permite —la especificación se construye programáticamente— pero no facilita. Es un espacio de trabajo abierto.

### 8.5. Por qué el fine-tuning mitiga esto

La advertencia del slide —"need for finetuning on real data"— tiene una explicación mecánica que vale la pena hacer explícita.

Descompón mentalmente lo que aprende un detector de eventos sonoros en dos partes:

1. **Un banco de representaciones acústicas**: qué patrones tiempo-frecuencia corresponden a un ladrido, a una sirena, a un martillo. Es conocimiento sobre las **fuentes**.
2. **Un modelo de la condición de observación**: cómo se ve una fuente cuando está reverberada, atenuada, mezclada con otras, capturada por *este* micrófono. Es conocimiento sobre el **canal**.

El dato sintético es **excelente para (1)** y **sistemáticamente equivocado para (2)**. Las fuentes de UrbanSound8K son grabaciones reales de bocinas reales: el conocimiento de fuente es genuino. Lo que es falso es el canal.

El fine-tuning con datos reales, en ese marco, es una operación quirúrgica: se preservan las capas bajas y medias que codifican (1) —que costaron 30 horas de datos sintéticos y ninguna anotación humana— y se reajustan las capas altas y la calibración de salida para (2), que es lo que las pocas horas de datos reales anotados pueden enseñar. Es la misma economía de transferencia que hace funcionar a ImageNet → tarea específica: lo caro y genérico se aprende donde hay datos, lo barato y específico se ajusta donde no los hay.

Esto también explica por qué **la proporción importa**: para reajustar el canal no necesitas 30 horas reales, necesitas suficientes como para cubrir la variabilidad del canal de despliegue — que, si el sensor es fijo, es mucho menor que la variabilidad de las fuentes. Es un argumento a favor de la estrategia, no una excusa.

### 8.6. Las otras estrategias

Ordenadas por invasividad creciente:

| Estrategia | Qué hace | Costo | Qué no arregla |
|---|---|---|---|
| **Fine-tuning sobre datos reales** | Reajusta el modelo con un conjunto pequeño anotado del dominio destino | Requiere anotación fuerte real, poca | Nada si el conjunto real es demasiado pequeño o poco diverso |
| **Convolución con RIRs medidas** | Convolucionar cada fuente con una respuesta al impulso real del espacio destino, antes de sumar | Requiere un banco de RIRs (existen públicos: BUT ReverbDB, ACE Challenge, MIT IR Survey) | La coloración del micrófono, la co-ocurrencia |
| **Simulación de sala (image-source, ray tracing)** | Generar RIRs sintéticas para geometrías y posiciones arbitrarias (`pyroomacoustics`, `gpuRIR`) | Cómputo, y modelos de absorción simplificados | Escenas exteriores urbanas, difíciles de modelar como salas |
| **Aumentación de canal** | Filtros aleatorios, códecs, ruido, AGC simulado, aplicados a la mezcla | Barato | La estructura de la escena |
| **Adaptación de dominio no supervisada** | Alinear las distribuciones de features de sintético y real sin etiquetas reales (DANN con gradient reversal, CORAL, MMD, *self-training* con pseudo-etiquetas, *mean teacher*) | Requiere audio real **sin anotar**, que sí es abundante | Puede alinear features sin alinear la tarea si el desajuste es de etiquetas |
| **Sim-to-real por randomización de dominio** | Aleatorizar agresivamente todos los parámetros no realistas del simulador para que el dominio real caiga *dentro* de la envolvente sintética | Requiere que el simulador exponga esos parámetros | Los sesgos estructurales (co-ocurrencia) que no son parámetros |

La convergencia práctica en SED, visible en la evolución de los desafíos DCASE, fue la combinación: **entrenar con sintético + real débilmente etiquetado + real sin etiquetar**, con un esquema *mean teacher* de consistencia. La tarea 4 de DCASE (*Sound Event Detection in Domestic Environments*) institucionalizó exactamente esa receta de tres fuentes, y el conjunto sintético de esa tarea se genera **con Scaper**. Es decir: la comunidad no resolvió la brecha eliminando lo sintético, sino **poniéndolo en su lugar dentro de un régimen mixto**, que es precisamente lo que dice el slide.

### 8.7. Cuándo la brecha no importa

Un contrapunto necesario, porque el discurso "lo sintético no sirve" es tan equivocado como el opuesto.

La brecha sintético→real es un problema **cuando la cifra sintética se usa como estimador del desempeño en producción**. No lo es cuando se usa para:

- **Comparar modelos entre sí** bajo condiciones idénticas (el uso de la Sección 4 del paper).
- **Caracterizar el comportamiento** de un modelo como función de una variable (los barridos de polifonía y SNR).
- **Depurar** un pipeline: si tu modelo falla sobre datos donde la anotación es exacta por construcción, el problema es tuyo, no del *ground truth*.
- **Pre-entrenar** representaciones que después se transfieren.
- **Calibrar anotadores humanos** (Sección 5 del paper).

El propio paper traza la línea con precisión: Scaper "es útil para generar datasets tanto para entrenar modelos como para comparar su desempeño en función de características acústicas controladas", pero "no puede usarse como reemplazo de grabaciones del mundo real anotadas manualmente **si queremos estimar cuán bien rendirá un modelo en un ambiente real**" (Sección 6). El condicional final es todo.

---

## 9. Scaper como herramienta viva

### 9.1. URBAN-SED, el caso canónico

Ya descrito en detalle en la Sección 7.1. Lo que hay que retener es qué lo hace *canónico*: no es que sea el dataset más usado (no lo es), sino que es **la demostración de que un dataset puede publicarse junto con su generador**. URBAN-SED se distribuye con:

- El audio y las anotaciones (`http://urbansed.weebly.com/`).
- Los scripts de generación **y** los de los experimentos de ML (`https://git.io/v9GEM`).
- Un JAMS por paisaje con la especificación probabilística y la instanciada.

Con eso, cualquiera puede: reproducir el dataset exacto, generar una versión de 100.000 paisajes en vez de 10.000, generar una versión con las mismas escenas pero otra SNR, o auditar si hay algún sesgo en el muestreo. Ninguna de esas cosas es posible con la mayoría de los datasets publicados, ni entonces ni ahora.

### 9.2. DCASE

La adopción más consecuente. Los desafíos **DCASE** (*Detection and Classification of Acoustic Scenes and Events*), organizados anualmente por la comunidad, incorporaron datasets sintetizados con Scaper en sus tareas de detección de eventos sonoros — de forma prominente en la **Tarea 4** (*Sound Event Detection in Domestic Environments*), cuyo conjunto sintético fuertemente etiquetado se genera con la librería a partir de aislados de Freesound.

El diseño de esa tarea es la respuesta institucional a la brecha de la Sección 8: se entrega a los participantes **datos reales débilmente etiquetados**, **datos reales sin etiquetar** y **datos sintéticos fuertemente etiquetados**, y se evalúa sobre **datos reales fuertemente etiquetados**. Los sintéticos aportan la supervisión temporal que los reales no pueden aportar a escala; los reales aportan el realismo acústico que los sintéticos no tienen. Es exactamente la arquitectura de solución que el slide de la clase describe en una línea.

Que un desafío internacional recurrente construya sus datos sobre una librería es la mejor evidencia de madurez que puede tener una herramienta de este tipo.

### 9.3. Separación de fuentes

Scaper también se usó fuera de SED, en la creación de datasets para **separación de fuentes universal**, notablemente en la familia de conjuntos derivados del ecosistema WHAM/WHAMR/FUSS. La conexión es natural: si ya tienes un motor que mezcla fuentes aisladas con parámetros controlados, tienes gratis los *targets* de separación — porque **las fuentes individuales pre-mezcla son la verdad de terreno de la separación**.

La librería lo soporta explícitamente hoy: `generate()` acepta `save_isolated_events=True` e `isolated_events_path`, que escriben cada evento renderizado por separado junto con la mezcla (verificado en `scaper/core.py`). Ese flag **no existe en el paper de 2017**; es una adición posterior motivada justo por este caso de uso. Es un buen ejemplo de cómo una abstracción correcta admite usos que su autor no anticipó: la especificación instanciada ya contenía toda la información necesaria, solo faltaba exponer los estemas.

Vale una nota de precisión: **no puedo verificar desde este paper** qué datasets específicos de la familia WHAM/FUSS usan Scaper y en qué grado; lo anoto como conexión de ecosistema conocida, no como cifra citable del paper.

### 9.4. Ejemplo de código del API actual

El siguiente ejemplo está construido sobre las firmas verificadas en `scaper/core.py` (rama `master`) y sobre el tutorial oficial. Los comentarios marcan qué es del paper y qué es posterior.

```python
import os
import scaper

# ─────────────────────────────────────────────────────────────────────────────
# 1. El Scaper: duración del paisaje + rutas del soundbank + semilla
# ─────────────────────────────────────────────────────────────────────────────
# El soundbank es una convención de sistema de archivos, no una base de datos:
#   foreground/<label>/*.wav    y    background/<label>/*.wav
# La subcarpeta ES la etiqueta. (Sección 2 del paper.)
audio_root = os.path.expanduser("~/audio")

sc = scaper.Scaper(
    duration=10.0,                                    # 10 s, como URBAN-SED
    fg_path=os.path.join(audio_root, "foreground"),
    bg_path=os.path.join(audio_root, "background"),
    protected_labels=[],                              # etiquetas exentas de transformaciones
    random_state=20170415,                            # reproducibilidad del MUESTREO
)

# Nivel de referencia del fondo, en LUFS. URBAN-SED usa -50.
# Todas las SNR de los eventos se miden RESPECTO de este nivel. (Sección 2 y 3.)
sc.ref_db = -50

# ─────────────────────────────────────────────────────────────────────────────
# 2. El fondo. Un solo background por paisaje, tratado como textura holística.
# ─────────────────────────────────────────────────────────────────────────────
# Cada argumento es una DISTRIBUTION TUPLE, no un valor: (nombre, *parámetros).
# ('choose', []) = elegir uniformemente entre TODOS los archivos disponibles
# para la etiqueta ya fijada.
sc.add_background(
    label=("const", "noise"),        # URBAN-SED: ruido browniano sintético,
    source_file=("choose", []),      # elegido justamente para garantizar que
    source_time=("const", 0),        # el fondo NO contenga eventos espurios.
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Los eventos de primer plano.
# ─────────────────────────────────────────────────────────────────────────────
# Evento 1: una sirena con parámetros fijados a mano (extremo determinista
# del continuo de control que describe la Sección 2).
sc.add_event(
    label=("const", "siren"),
    source_file=("choose", []),
    source_time=("const", 0),                # empezar en el onset del clip fuente
    event_time=("const", 2.0),               # aparece en t=2 s del paisaje
    event_duration=("const", 3.0),
    snr=("const", 12),                       # 12 LUFS sobre el fondo -> -38 LUFS
    pitch_shift=None,                        # None = sin transformación
    time_stretch=None,
)

# Eventos 2..N: la receta de URBAN-SED (Sección 3), completamente probabilística.
# Cantidad de eventos ~ Uniforme discreta {1..9}.
import numpy as np
rng = np.random.RandomState(20170415)
n_events = rng.randint(1, 10)

for _ in range(n_events):
    sc.add_event(
        label=("choose", []),                    # cualquiera de las 10 clases urbanas
        source_file=("choose", []),              # cualquier clip de esa clase
        source_time=("const", 0),                # "para no perder el onset del evento"
        event_time=("normal", 5.0, 2.0),         # una de las 3 distribuciones del paper;
                                                 # controla la polifonía INDIRECTAMENTE
        event_duration=("uniform", 0.5, 4.0),    # clips de UrbanSound8K duran <= 4 s
        snr=("uniform", 6, 30),                  # rango de SNR de URBAN-SED
        pitch_shift=("uniform", -3, 3),          # semitonos; NO altera la duración
        time_stretch=("uniform", 0.8, 1.2),      # factor;   NO altera el pitch
    )

# ─────────────────────────────────────────────────────────────────────────────
# 4. Instanciar y renderizar. Cada llamada a generate() muestrea de nuevo:
#    UNA especificación -> INFINITAS instanciaciones.
# ─────────────────────────────────────────────────────────────────────────────
sc.generate(
    audio_path="soundscape_0001.wav",
    jams_path="soundscape_0001.jams",       # <- LA RECETA COMPLETA
    txt_path="soundscape_0001.txt",         # <- onset/offset/label, cargable en Audacity
    allow_repeated_label=True,              # permitir dos eventos de la misma clase
    allow_repeated_source=True,             # permitir reusar el mismo archivo fuente
    reverb=None,                            # POSTERIOR al paper: reverb algorítmica de SoX
    fix_clipping=False,                     # POSTERIOR al paper: ver Sección 6.5
    peak_normalization=False,               # POSTERIOR al paper
    save_isolated_events=False,             # POSTERIOR: estemas para separación de fuentes
    disable_sox_warnings=True,
    no_audio=False,                         # True = instanciar sin renderizar (barato)
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. La operación que hace único a Scaper: EDITAR LA RECETA Y RE-RENDERIZAR.
#    Es el mecanismo del barrido de SNR de la Sección 4 del paper: se toman las
#    2000 anotaciones JAMS del test, se homogeneiza la SNR de todos sus eventos,
#    y se regenera el audio -> 8 conjuntos de test idénticos salvo por UNA variable.
# ─────────────────────────────────────────────────────────────────────────────
import jams

jam = jams.load("soundscape_0001.jams")
ann = jam.annotations.search(namespace="scaper")[0]
for obs in ann.data:
    if obs.value["role"] == "foreground":
        obs.value["snr"] = 9.0               # fijar TODOS los eventos a la misma SNR
jam.save("soundscape_0001_snr9.jams")

scaper.generate_from_jams(
    "soundscape_0001_snr9.jams",
    audio_outfile="soundscape_0001_snr9.wav",
    # fg_path / bg_path opcionales: si el soundbank se movió, se re-apunta aquí.
    # Esta es la dependencia externa que rompe el cierre: el JAMS referencia
    # archivos fuente por RUTA, no por hash de contenido.
)
```

Tres cosas a notar en ese código, que son las tres decisiones de diseño del paper hechas API:

1. **Cada parámetro es una tupla, no un valor.** La firma del API *fuerza* la indirección especificación→instancia. No hay forma de llamar `add_event(snr=12)`; hay que decir `snr=("const", 12)`. Es fricción deliberada, y buena: garantiza que la ruta determinista y la probabilística sean el mismo código.
2. **`generate()` se puede llamar $N$ veces sobre el mismo objeto** y produce $N$ paisajes distintos. La especificación es el objeto persistente; el paisaje es efímero.
3. **`generate_from_jams()` es una función de módulo, no un método.** Reconstruir desde una receta no requiere el objeto que la creó — la receta es autosuficiente (salvo por el soundbank). Eso es lo que hace que la receta sea el artefacto compartible.

---

## 10. Limitaciones

### 10.1. Las que el paper reconoce (Sección 6)

1. **La riqueza acústica.** "Los paisajes sonoros generados, incluso si suenan bastante realistas en algunos casos, no pueden abarcar la riqueza y complejidad de los paisajes sonoros reales." Y la consecuencia operativa: no es reemplazo de grabaciones reales anotadas si el objetivo es estimar el desempeño en producción.

2. **La plausibilidad queda en manos del usuario.** "Dado que la especificación depende completamente del usuario, **es posible generar paisajes sonoros que no son plausibles**, y por lo tanto los parámetros del paisaje deben elegirse **concienzudamente y en función de la aplicación de dominio específica**."

   Esta es una limitación honesta y más profunda de lo que aparenta: Scaper es una herramienta **sin opinión**. No sabe nada de acústica ni del mundo; no te impedirá poner ocho sirenas simultáneas a $-3$ dB de SNR sobre un fondo de biblioteca. **La carga de la validez ecológica se transfiere íntegramente al usuario**, y el usuario no tiene ninguna forma automática de verificarla. No hay validador de plausibilidad, ni siquiera advertencias.

3. **No hay control explícito de la polifonía.** "Actualmente Scaper no soporta controlar explícitamente ciertas características de la escena como la polifonía máxima (o promedio), y planeamos agregar esta funcionalidad en el futuro." Ver Sección 6.3.

4. **La aumentación por recombinación está propuesta, no demostrada.** El paper sugiere un uso interesante: "Dado un dataset para SED, uno también podría **extraer todos los eventos sonoros que no se solapan con otros** y usarlos como *soundbank* para generar paisajes completamente nuevos, como forma de aumentación de datos". Y luego, con encomiable honestidad: "Aunque el material fuente no es nuevo, al aplicar transformaciones de audio y generar polifonías nunca vistas los datos aumentados podrían potencialmente mejorar la generalizabilidad del modelo — **esto queda por demostrarse**."

   Vale subrayarlo: **el paper no demuestra que Scaper mejore la generalización de ningún modelo**. Lo propone como hipótesis. La palabra "augmentation" está en el título, pero el uso aumentativo es el único de los propuestos que queda sin evaluar.

### 10.2. Las que el paper no reconoce

1. **La calidad del soundbank es el techo de todo, y no se audita.** Scaper hereda todo defecto de sus fuentes: si un clip etiquetado `dog_bark` contiene además una bocina de fondo, esa bocina entra al paisaje **sin aparecer en la anotación**, y el falso negativo forzado que el paper evitó cuidadosamente en el fondo (usando ruido browniano) reaparece por la puerta de los eventos. UrbanSound8K está bien curado, pero "bien curado" no es "sin eventos secundarios". Ningún mecanismo de la librería detecta esto. La garantía de anotación exacta es, con precisión, **exactitud respecto del modelo generativo**, no exactitud acústica.

2. **Diversidad combinatoria ≠ diversidad acústica.** "Infinitas instanciaciones" es literalmente cierto y epistemológicamente engañoso. Los 10.000 paisajes de URBAN-SED se construyen sobre ~10.000 clips fuente; con ~50.000 eventos, **cada clip aparece en promedio unas cinco veces** en el dataset, cambiado de pitch y de tiempo. Un modelo con suficiente capacidad puede memorizar los clips fuente y resolver la tarea por identificación de grabación específica en vez de por reconocimiento de clase — y el conjunto de test, aunque usa folds distintos (lo que evita la fuga directa), sigue siendo un espacio de fuentes reducido. La curva de retorno decreciente respecto del tamaño generado no se estudia en el paper: no hay ablación de "¿cuánto mejora un modelo entrenado con 1.000 vs 10.000 vs 100.000 paisajes del mismo soundbank?", que es la pregunta central para justificar la generación masiva.

3. **Sin modelado acústico del espacio.** Desarrollado en la Sección 8.1. La suma sustituye a la convolución. La `reverb` posterior es global y algorítmica, no por fuente ni medida.

4. **Sin dimensión espacial.** No hay posición de la fuente, ni distancia, ni azimut, ni Doppler, ni panorámica estéreo, ni salida multicanal. Para SED monoaural es aceptable; para las tareas de **localización y detección** (SELD, que se volvió una tarea DCASE propia) es descalificante, y motivó herramientas de generación espacial separadas.

5. **Independencia i.i.d. entre eventos.** Desarrollado en la Sección 8.4. La estructura conjunta del mundo no es expresable en el modelo de especificación tal como está.

6. **El riesgo de aprender los artefactos de la síntesis.** Desarrollado en la Sección 8.3. Es la limitación más peligrosa porque es **invisible en validación** — el conjunto de validación comparte el artefacto. La única forma de detectarla es evaluar sobre datos reales, que es exactamente lo que la síntesis intenta evitar. Hay una circularidad incómoda ahí que el paper no discute.

7. **Nada sobre el costo computacional ni la escala de generación.** No se reporta cuánto tarda generar 10.000 paisajes, ni si el proceso paraleliza. Como cada llamada a SoX es un subproceso con I/O de disco, es plausible que el cuello de botella sea sustancial a escala de cientos de miles de ejemplos. Es información que uno querría al planificar.

8. **Ninguna evaluación perceptual del realismo.** El paper dice que los paisajes "suenan bastante realistas en algunos casos", lo cual es una afirmación empírica sin evidencia. No hay un test de escucha, ni una métrica objetiva de distancia entre distribuciones sintética y real (algo tipo FAD, *Fréchet Audio Distance*, que apareció después). Dado que los autores tenían montada toda la infraestructura de *crowdsourcing* de la Sección 5, la ausencia de un experimento de realismo percibido es notoria.

9. **La reproducibilidad depende de la estabilidad del entorno.** El JAMS captura la receta, pero el audio resultante depende de la versión de SoX, de la implementación del medidor de LUFS y de la versión de Scaper. Un JAMS de 2017 re-renderizado hoy produce audio *casi* idéntico, no idéntico. Ni el paper ni la librería fijan un hash del soundbank ni una versión del motor de audio dentro de la anotación — que es el paso que faltaría para cerrar la cadena de reproducibilidad de verdad.

---

## 11. Impacto y legado

### 11.1. Qué pasó con Scaper

Ocho años después, Scaper sigue siendo la herramienta estándar para sintetizar paisajes sonoros con anotación fuerte. La evidencia:

- **DCASE la adoptó institucionalmente** para generar los conjuntos sintéticos de sus tareas de SED (Sección 9.2). Un desafío internacional recurrente construido sobre tu librería es la forma más fuerte de validación que existe para infraestructura.
- **La librería siguió evolucionando** más allá del paper: `choose_weighted`, `fix_clipping`, `peak_normalization`, `quick_pitch_time`, `save_isolated_events`, `generate_from_jams` con re-apuntado de rutas, `protected_labels`. Todas son respuestas a problemas encontrados en uso real, y todas caben dentro de la abstracción original sin romperla — la mejor señal de que la abstracción era correcta.
- **Se extendió a dominios nuevos**: separación de fuentes (estemas aislados), bioacústica, escenas domésticas.

El mérito de fondo no es técnico sino **conceptual**: Scaper normalizó la idea de que **un dataset es un programa, no una carpeta**. Esa idea excede al audio.

### 11.2. La comparación con los generativos modernos

La pregunta obvia en 2026: si existen modelos generativos de audio de calidad —AudioLDM, AudioGen, Stable Audio, Make-An-Audio— que producen audio arbitrario desde texto, ¿para qué sirve un secuenciador que recorta y suma WAVs?

La respuesta requiere separar dos capacidades que se confunden:

| | **Scaper (composición)** | **Generativos de audio (síntesis neuronal)** |
|---|---|---|
| Qué produce | Mezclas de grabaciones **reales** existentes | Audio **nuevo**, nunca grabado |
| Diversidad de fuentes | Acotada al soundbank | En principio ilimitada |
| Realismo del evento aislado | **Perfecto** (es una grabación real) | Variable; artefactos, "sonido de generador" |
| Etiquetas fuertes | **Exactas por construcción** | **No las hay** |
| Control de la mezcla | Total y explícito (SNR, tiempos, polifonía) | Implícito y vía prompt; poco controlable |
| Reproducibilidad | Receta serializada, regeneración exacta | Semilla + modelo + versión + prompt; frágil |
| Costo por hora generada | Segundos de CPU | GPU, órdenes de magnitud más |
| Auditable | Sí: se sabe qué archivo fuente entró | No: el modelo es opaco |

**La asimetría decisiva es la fila de las etiquetas fuertes.** Si le pides a un modelo texto→audio "un perro ladrando mientras pasa una sirena, en una calle", te entrega diez segundos de audio plausible — y **no te dice en qué milisegundo empieza el ladrido**. Para obtener la anotación fuerte tendrías que anotarlo a mano, y volvimos al punto de partida de la Sección 2, con el agravante de que ahora el audio ni siquiera es real.

Existe una vía híbrida que es la que tiene sentido: **generar los eventos aislados con un modelo generativo y componerlos con Scaper**. El generativo resuelve la escasez de fuentes (puedes producir mil variantes de bocina sin salir a grabar); Scaper resuelve la composición y la anotación. Las dos herramientas operan en niveles distintos y son complementarias, no competidoras. La calidad del soundbank —la limitación 10.2.1— es exactamente el problema que un generativo puede aliviar.

Los problemas nuevos que traen los generativos, y que conviene tener presentes:

1. **No hay etiquetas fuertes.** Ya discutido. Es el problema estructural.
2. **Sesgos del generador.** Un modelo texto→audio produce la *moda* de su distribución de entrenamiento. Pídele "ladrido de perro" mil veces y obtendrás mil variantes de un ladrido prototípico, no la diversidad real de razas, distancias, estados emocionales y acústicas. La diversidad aparente es alta; la diversidad efectiva, mucho menor. Un dataset generado así tiene una **cola corta artificialmente**, y las colas es donde los modelos fallan.
3. **Colapso al entrenar con datos generados (*model collapse*).** Si el generativo se entrenó con AudioSet, y tú generas datos con él para entrenar un detector que después evalúas en AudioSet, estás en un circuito cerrado: el detector aprende la aproximación que el generativo hizo de la distribución, no la distribución. Iterar el proceso —entrenar generativos sobre datos generados— degrada progresivamente las colas de la distribución, un fenómeno ya bien documentado en texto e imagen y sin razón para no aplicar a audio.
4. **Artefactos aprendibles, otra vez.** El mismo problema de la Sección 8.3, agravado: el audio neuronal tiene firmas propias (del vocoder, del decodificador latente) que una red discriminativa detecta con facilidad. Entrenar sobre él puede enseñar a detectar el generador, no el evento.
5. **Procedencia y licencia.** Scaper te dice exactamente qué archivo fuente, con qué licencia, entró en cada paisaje — la trazabilidad está en el JAMS. Un modelo generativo no te dice de qué se acuerda.

Mi lectura: **Scaper no fue superado, fue complementado.** Su ventaja competitiva —el vínculo determinista y exacto entre el proceso generativo y la anotación— no es algo que un modelo generativo pueda ofrecer, porque un generativo no *construye* la escena a partir de partes identificables: la *muestrea* como un todo. Mientras la tarea sea de localización temporal, la composición explícita seguirá siendo la única fuente barata de verdad de terreno exacta.

Y hay una lección más general, que trasciende el audio: **cuando el cuello de botella es la anotación y no el dato, la solución no es generar más dato realista sino generar dato cuya anotación sea una consecuencia del proceso de generación**. Es el mismo principio detrás de los simuladores en robótica (donde la pose del objeto la sabe el simulador), del renderizado sintético en visión (donde la máscara de segmentación sale del *z-buffer*, gratis y perfecta), y del *fuzzing* basado en gramáticas en testing (donde la entrada válida se construye desde la gramática que la define). Scaper es la instancia auditiva de una familia de ideas.

---

## 12. Conexión con la clase 39 y el laboratorio

### 12.1. El linaje Salamon: UrbanSound8K → Scaper → URBAN-SED

Este es el punto que hay que dejar explícito, porque es literalmente el mismo autor cerrando un círculo de tres años.

**Justin Salamon** es primer autor de:

1. **"A dataset and taxonomy for urban sound research"** — Salamon, Jacoby y Bello, ACM Multimedia 2014. Es la referencia **[25]** del propio paper de Scaper, y es **UrbanSound8K**: el dataset del laboratorio de esta clase.
2. **"Deep convolutional neural networks and data augmentation for environmental sound classification"** — Salamon y Bello, IEEE Signal Processing Letters, vol. 24, n.º 3, 2017. Referencia **[27]**, y es la CNN que el paper adapta como uno de los dos modelos evaluados (Sección 4).
3. **Scaper** — este paper, WASPAA 2017.

Y **Juan Pablo Bello** firma los tres. La secuencia es una trayectoria de investigación coherente:

| Año | Trabajo | Qué problema resuelve | Qué problema deja abierto |
|---|---|---|---|
| 2014 | **UrbanSound8K** | No hay dataset ni taxonomía para sonido urbano. Se construye uno de 8732 clips ≤4 s, 10 clases, con **10 folds estratificados** | Es clasificación de clips: **una clase por clip, sin tiempos**. No sirve para SED |
| 2017 | **CNN + augmentation (SPL)** | ¿Cómo se entrena una CNN con tan poco dato? Con aumentación (deformación temporal, pitch, ruido de fondo, compresión dinámica) | La aumentación no crea escenas nuevas ni etiquetas temporales |
| 2017 | **Scaper + URBAN-SED** | Cómo obtener **etiquetas fuertes a escala**: componiendo escenas a partir de los clips aislados de 2014 | La brecha con lo real (Sección 8) |

**Y el vínculo es literal, no temático: URBAN-SED se construye usando UrbanSound8K como soundbank.** Los clips de 2014 son las piezas; Scaper es el motor de ensamblaje; URBAN-SED es el producto. Lo que en 2014 era un dataset de clasificación se convierte en 2017 en la materia prima de un dataset de detección, sin una sola anotación humana adicional.

### 12.2. El diseño de dataset que comparten: los 10 folds

Esta es la continuidad de diseño más concreta y la que el laboratorio va a tocar directamente.

UrbanSound8K se distribuye **pre-dividido en 10 folds estratificados**, y su documentación es enfática (célebremente enfática, de hecho) en que **hay que usar los folds oficiales y reportar validación cruzada de 10 pliegues sobre ellos**, en vez de re-barajar aleatoriamente. La razón: los 8732 clips provienen de **1302 grabaciones de campo** de Freesound, y varios clips pueden salir de la misma grabación original. Un *shuffle* aleatorio pondría fragmentos de la misma grabación en entrenamiento y en test, y el modelo, en vez de reconocer la clase, **reconocería la grabación** — con una inflación de la métrica que es a menudo de varios puntos.

Los folds oficiales garantizan que **todos los clips de una misma grabación caigan en el mismo fold**. Es control de fuga de datos a nivel de grupo, no de instancia.

URBAN-SED **hereda esa disciplina** (Sección 3): "UrbanSound8K está pre-ordenado en 10 folds estratificados, y por lo tanto usamos los **folds 1–6** para generar 6000 paisajes de entrenamiento, **7–8** para 2000 de validación y **9–10** para 2000 de test."

La propiedad que esto preserva: **ningún clip fuente que aparece en un paisaje de entrenamiento aparece en uno de test**. Sin esa herencia, la síntesis habría destruido justamente la garantía que costó construir en 2014 — porque un mismo `siren.wav` podría reaparecer en test bajo otro pitch, otro tiempo de inicio y otra SNR, y el modelo lo reconocería.

**Este es el punto de diseño que hay que llevarse al laboratorio.** Cuando en el lab de esta clase se use UrbanSound8K con sus 10 folds y se resista la tentación de hacer `train_test_split(shuffle=True)`, se está aplicando exactamente el mismo principio que Salamon aplicó al construir URBAN-SED tres años después. **La unidad de división no es el ejemplo: es la fuente de la que el ejemplo deriva.** Es la misma lección que en cualquier dominio con datos agrupados — pacientes en datos clínicos, usuarios en logs, repositorios en datasets de código.

### 12.3. Dónde encaja Scaper en el mapa de data augmentation de audio

La clase 37 del mismo diplomado ("Datasets y Herramientas para Audio") cubrió el fundamento de augmentation de audio: SNR y mezcla de ruido, SpecAugment, pitch shifting, time stretching. La clase 39 vuelve sobre el tema en su slide de *Data Augmentation*, enumerando "modificar el pitch, agregar ruido, hacer time stretching" y agregando la advertencia sobre la síntesis.

Scaper opera en un **nivel distinto** de todas esas técnicas, y ubicarlo bien en el mapa es lo que da coherencia al conjunto:

| Nivel | Dónde opera | Ejemplos | ¿Crea ejemplos nuevos? | ¿Crea etiquetas nuevas? |
|---|---|---|---|---|
| **Representación** | Sobre el espectrograma, después del feature extraction | **SpecAugment** (máscaras de tiempo y frecuencia), mixup en el dominio de features | No: perturba uno existente | No: hereda la etiqueta |
| **Forma de onda** | Sobre la señal, antes del feature extraction | **Pitch shift**, **time stretch**, adición de ruido a **SNR** controlada, compresión dinámica, filtrado, códec | No: perturba uno existente | No: hereda la etiqueta |
| **Escena** | Sobre la **composición** de la escena | **Scaper** | **Sí**: compone una escena que no existía | **Sí, y exactas**: onsets, offsets y polifonía se derivan del proceso |

La distinción operacional es nítida:

- Las técnicas de los dos primeros niveles toman **un ejemplo etiquetado** $(x, y)$ y producen $(T(x), y)$. Necesitan que el ejemplo etiquetado exista, y la etiqueta se hereda sin cambios. **No aumentan la cantidad de anotación**, solo su rendimiento.
- Scaper toma **un conjunto de fuentes etiquetadas por clase** y produce $(x_{\text{nuevo}}, y_{\text{nuevo}})$ donde $y_{\text{nuevo}}$ es **estructuralmente más rica** que las etiquetas de entrada: parte de clases sueltas y llega a una anotación temporal completa. **Sube el nivel de supervisión**: entra supervisión débil (una etiqueta por clip), sale supervisión fuerte (onsets y offsets). Ese es el truco que ninguna de las otras técnicas hace.

Es exactamente la observación que el paper hace en la Sección 1 y que cité en la Sección 2.4 de este análisis: la aumentación aplicada al paisaje completo deja intactos "el timing de los eventos, el grado de solapamiento y la SNR". Scaper es la herramienta que ataca **esas tres** variables, que son las que definen la tarea de SED.

Y nótese que **los tres niveles se componen**. Scaper mismo usa pitch shift y time stretch (nivel forma de onda) **por evento**, dentro de la composición. Un pipeline maduro sería: Scaper para componer escenas con etiquetas fuertes → augmentation de forma de onda sobre la mezcla resultante para simular variación de canal → SpecAugment sobre el espectrograma durante el entrenamiento. Tres niveles, tres tipos de invarianza, ningún conflicto.

### 12.4. La advertencia del slide, cerrada

El slide de la clase 39 dice, textualmente:

> "Use of sound synthesis techniques is a convenient strategy. However, performance on real data might be poor if a model is trained using only synthetic data. Need for finetuning on real data."

Y el paper que lo respalda dice, en su Sección 6:

> "los paisajes sonoros generados [...] no pueden abarcar la riqueza y complejidad de los paisajes sonoros reales [...] **no puede usarse como reemplazo de grabaciones del mundo real anotadas manualmente**, si queremos estimar cuán bien rendirá un modelo en un ambiente real."

Las dos frases dicen lo mismo, y el paper llegó primero. Lo notable es que la advertencia **está en el paper que propone la técnica**, no en una crítica posterior. Los autores conocían la limitación, la enunciaron con precisión, y aun así el trabajo fue valioso — porque **acotaron correctamente el alcance de su propia herramienta**.

La síntesis de datos es una estrategia legítima y potente cuando se sabe qué compra y qué no compra: compra **volumen y exactitud de anotación**, no compra **realismo acústico**. Y como esas dos cosas son separables, la arquitectura de solución correcta es mixta — el pre-entrenamiento masivo con sintético y el ajuste fino con real, o el régimen de tres fuentes que DCASE institucionalizó. Ese es el mensaje que el slide comprime en tres líneas y que este paper desarrolla en cinco páginas.

---

## 13. Erratas, matices y cosas que se citan mal

**1. El rango de SNR del barrido está mal enumerado en el texto.** La Sección 4 dice: "Repetimos este proceso **ocho veces**, fijando la SNR de los eventos en el rango **6–9, 9–12, 12–15, 15–18, 18–24, 24–27 y 27–30**". Son **siete** rangos listados, y el listado salta de 18 a 24 en un solo tramo. La Figura 3 (abajo) muestra inequívocamente **ocho** barras con las etiquetas `6–9, 9–12, 12–15, 15–18, 18–21, 21–24, 24–27, 27–30`. La lectura correcta es la de la figura: el texto colapsó por error "18–21, 21–24" en "18–24". Verificado sobre la Figura 3 del PDF.

**2. Las unidades de la SNR se mezclan entre secciones.** La Sección 2 define la SNR en **LUFS** de forma explícita y razonada ("si un evento se especifica con una SNR de 6, significa que estará **6 LUFS** por encima del nivel del fondo"). La Sección 3 dice que "la SNR se muestrea uniformemente entre 6–30 **dB**", y la Figura 3 rotula su eje como `SNR (db)`. LUFS es una escala en decibeles relativa a *full scale*, así que las magnitudes son consistentes, pero la unidad correcta según la propia definición del paper es LUFS y la notación `db` (además en minúsculas, cuando lo correcto sería `dB`) es descuidada.

**3. Hay un typo tipográfico en la Figura 1.** El bloque central del diagrama de la Figura 1 dice **"SOUNDCSAPE INSTANTIATION & GENERATION"** en vez de "SOUNDSCAPE". Verificado visualmente sobre el PDF, página 2. Es intrascendente, pero si alguien reproduce la figura, ahí está.

**4. El slide de la clase 39 escribe mal el apellido del primer autor.** La referencia del slide dice *"Scaper: a library for soundscape synthesis and augmentation. **Salaman** et al., 2017"*. Es **Salamon**, con o. La forma correcta de citar es Salamon, MacConnell, Cartwright, Li y Bello, WASPAA 2017.

**5. "Scaper es una herramienta de data augmentation" es a lo sumo media verdad.** La palabra está en el título, pero **el paper no demuestra ningún resultado de aumentación**. Los usos evaluados son (a) generar un dataset de entrenamiento/evaluación desde cero y (b) generar estímulos para un experimento con humanos. El uso aumentativo —agregar eventos a paisajes existentes, o recombinar eventos extraídos de un dataset real— aparece solo en la Sección 6 (Discusión) como propuesta, con la frase textual "**esto queda por demostrarse**" (*this remains to be shown*). Citar Scaper como evidencia de que la aumentación por síntesis mejora la generalización es citar mal el paper.

**6. "URBAN-SED es el dataset de SED más grande" era cierto en 2017 y ya no lo es.** La afirmación del paper es explícitamente relativa a su momento ("Esto lo convierte en el dataset con etiquetas fuertes más grande **disponible** para SED"), y viene inmediatamente relativizada por los autores: "aunque por supuesto podríamos hacerlo arbitrariamente más grande o más pequeño". Repetir la frase sin la fecha ni la coletilla es una tergiversación menor pero frecuente.

**7. Scaper no es un sintetizador de audio.** Es un **secuenciador y mezclador**. No genera muestras de audio nuevas: recorta, transforma y suma grabaciones existentes. El título dice "soundscape synthesis" y es correcto —sintetiza *paisajes*, no *sonidos*— pero la ambigüedad hace que a veces se lo confunda con un modelo generativo. La distinción es la de la Sección 11.2 y no es cosmética: de ella depende cuál es su ventaja competitiva.

**8. Los números de las Figuras 2 y 3 no están tabulados en el paper.** No hay ninguna tabla numérica en las cinco páginas. Todas las cifras de desempeño que cito en la Sección 7 de este análisis son **lecturas aproximadas de gráficos de barras** y las marco como tales. Cualquier documento que cite "la F-measure fue de 0.573" para URBAN-SED está reportando una precisión que el paper no publica. La única cifra de desempeño escrita en prosa es cualitativa ("los dos modelos rinden de forma comparable").

**9. El paper de 2017 no habla de reverberación, ni de clipping, ni de estemas aislados.** Los parámetros `reverb`, `fix_clipping`, `peak_normalization` y `save_isolated_events` de `generate()` son **adiciones posteriores** a la librería. Atribuirle al paper capacidades de la librería actual es un anacronismo fácil de cometer al leer la documentación en vez del PDF. Del mismo modo, la distribución `choose_weighted` no aparece en el paper: las distribuciones listadas en la Sección 2 son `const`, `choose`, `uniform`, `normal` y `truncnorm`.

**10. El formato de texto: "separado por espacios" vs tabulaciones.** La Sección 2 describe la salida de texto como "space-separated"; la implementación actual usa `txt_sep='\t'` por defecto. Detalle menor, pero relevante si alguien escribe un parser basándose en el paper.

**11. La CNN evaluada no es la CNN de Salamon y Bello 2017 tal cual.** Es una adaptación sustancial: activación final cambiada de softmax a sigmoides, filtros aumentados de la configuración original a 64 por capa, batch normalization agregada, entrada reducida a 1 s y max pooling cambiado a (2,2). El conteo de parámetros pasa de **241k a 720k** — casi el triple. Citar "la CNN de Salamon y Bello obtiene 0.57 en URBAN-SED" sin esas salvedades es incorrecto: el modelo evaluado es un pariente, no el original.

**12. El experimento de crowdsourcing no es de este paper.** La Sección 5 es un **resumen de resultados** de Cartwright et al. [31] (*"Seeing sound"*, PACM HCI 2017), y el propio texto remite al lector a esa publicación para los detalles ("For further details and results from the crowdsourcing experiments the reader is referred to [31]"). Los hallazgos sobre la superioridad del espectrograma como visualización y sobre el comportamiento de los anotadores humanos deben citarse a ese trabajo, no a Scaper.

**13. Sobre lo que no pude verificar.** No puedo confirmar desde este PDF: (a) qué datasets específicos de la familia WHAM/WHAMR/FUSS usan Scaper y en qué versión; (b) el detalle exacto de qué ediciones de DCASE y qué tareas lo emplean; (c) el conteo de citas o métricas de adopción. Todo lo que digo sobre esos puntos en las Secciones 9 y 11 es conocimiento de ecosistema, marcado como tal, y no debe atribuirse al paper.
