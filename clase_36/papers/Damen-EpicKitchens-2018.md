# Scaling Egocentric Vision: The EPIC-KITCHENS Dataset — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Scaling Egocentric Vision: The EPIC-KITCHENS Dataset*.
- **Autores:** Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price y Michael Wray.
- **Afiliaciones:** Universidad de Bristol (Reino Unido), Universidad de Catania (Italia) y Universidad de Toronto (Canadá).
- **Venue:** *European Conference on Computer Vision (ECCV 2018)*.
- **Año:** 2018. **Preprint:** arXiv:1804.02748v2 (31 de julio de 2018).
- **Palabras clave:** visión egocéntrica, dataset, benchmarks, visión en primera persona, detección de objetos egocéntrica, reconocimiento y anticipación de acciones.

Este paper no propone un modelo ni un algoritmo: propone un **dataset**, y con él redefine la escala a la que se puede estudiar la visión egocéntrica. EPIC-KITCHENS es un benchmark de video en primera persona grabado por **32 participantes** en sus propias cocinas, en cuatro ciudades de Norteamérica y Europa. Los participantes pertenecen a **10 nacionalidades distintas**, lo que produce una diversidad de estilos de cocina que ningún dataset previo había capturado. El corpus reúne **55 horas de video** que suman **11,5 millones de frames**, densamente etiquetados con **39,6 mil segmentos de acción** y **454,3 mil bounding boxes de objetos**.

La contribución metodológica más original es la **anotación narrada**: en lugar de contratar observadores externos, los autores pidieron a los propios participantes que, después de grabar, **narraran en voz alta las acciones que realizaban**, como un comentario en vivo de su propio video. Esa narración —en el idioma nativo de cada persona— se transcribió, se alineó temporalmente y se convirtió en etiquetas verbo-sustantivo. Sobre ese material se definen **tres desafíos** con baselines y leaderboards públicos: detección de objetos, reconocimiento de acciones y anticipación de acciones, evaluados en dos particiones de test —cocinas **vistas** (*seen*) y **no vistas** (*unseen*)—.

Para la **Clase 36 (Introduction to Video Analysis)** este trabajo importa porque encarna dos ideas que son el corazón del análisis de video realista: la **visión egocéntrica** (el video capturado por una cámara wearable que ve el mundo desde los ojos de quien actúa) y el video **untrimmed** (grabación continua y sin recortar, donde las acciones no vienen aisladas en clips de pocos segundos sino entrelazadas en el flujo natural de la actividad). Ambas condiciones hacen el problema mucho más difícil —y mucho más parecido a lo que un sistema tendría que resolver en el mundo real— que los benchmarks de clips cortos que dominaban el campo.

## 2. Contexto: visión egocéntrica frente a tercera persona; untrimmed frente a trimmed

### 2.1. Qué es la visión egocéntrica

La **visión en primera persona** o **egocéntrica** se refiere al video capturado por una cámara **wearable** —montada en la cabeza o el pecho de una persona— que registra el mundo tal como lo ve quien lleva la cámara. Se contrapone a la visión en **tercera persona**, donde una cámara externa observa a los sujetos desde afuera. La diferencia no es solo de ángulo: el punto de vista egocéntrico ofrece una perspectiva única sobre la **interacción de las personas con los objetos, su atención e incluso su intención**. Cuando alguien lava los platos con una GoPro en la cabeza, la cámara ve exactamente lo que las manos manipulan, hacia dónde se dirige la mirada y qué objeto viene después. Ese registro es rico porque refleja los objetivos y la motivación de la persona, su capacidad de hacer varias cosas a la vez y las muchas maneras de ejecutar tareas cotidianas y mundanas.

El paper argumenta que el progreso en este dominio había sido lento por una razón concreta: **la falta de datasets suficientemente grandes**. Mientras la clasificación de imágenes, la detección de objetos, el *captioning* y el *visual question answering* avanzaban gracias a benchmarks masivos, los datasets egocéntricos habían sido **mucho más pequeños que sus equivalentes en tercera persona**, y frecuentemente estaban capturados en un **único entorno**. El video egocéntrico de interacciones cotidianas, tomado con cámaras wearables, apenas estaba disponible en línea, lo que lo convertía en una fuente de información prácticamente inaccesible.

### 2.2. Untrimmed frente a trimmed: por qué es más realista y más difícil

El segundo eje conceptual es la distinción entre video **trimmed** (recortado) y **untrimmed** (sin recortar). La mayoría de los benchmarks de clasificación de acciones que existían contenían videos **muy cortos** —de apenas unos segundos— centrados en **una sola acción**. Son clips ya recortados: alguien decidió dónde empieza y dónde termina la acción, y el clip contiene poco más que esa acción. Ese formato simplifica el aprendizaje, pero es artificial.

EPIC-KITCHENS hace lo contrario. A cada participante simplemente se le pidió que **empezara a grabar cada vez que entrara a su cocina** y que registrara secuencias sin importar su duración. El resultado es video **untrimmed**: una grabación continua de todo lo que ocurre en la cocina, donde las acciones no están aisladas sino **encadenadas y superpuestas**. Esta grabación muestra el **multitasking natural** —lavar unos platos en medio de la cocción, buscar un utensilio, cambiar de idea, reaccionar a un imprevisto—, interacciones con objetivos paralelos que no aparecen en los datasets existentes. Esto lo hace a la vez **más realista y más difícil**.

El paper contrasta explícitamente su enfoque con el de datasets como Charades, donde los videos fueron grabados de manera **scriptada**: se pidió a trabajadores de Amazon Mechanical Turk que actuaran un guion frente a la cámara. Eso produce acciones que a menudo lucen poco naturales y que carecen de la progresión y el multitasking de la vida real. Todos los datasets egocéntricos previos usaban también actividades scriptadas —se le decía a la gente qué acciones ejecutar—. Cuando alguien sigue instrucciones, realiza los pasos en un orden secuencial, a diferencia de los escenarios reales que involucran hacer varias cosas a la vez, buscar un objeto, pensar qué hacer a continuación o cambiar de opinión. EPIC-KITCHENS captura video **no scriptado** de **actividades diarias no ensayadas**, y ese es precisamente su valor.

## 3. Contribución central

La contribución de EPIC-KITCHENS es haber construido **el dataset egocéntrico más grande y variado hasta la fecha**, capturado en los entornos nativos de los participantes, y haberlo dotado de una anotación densa que habilita múltiples tareas de comprensión fina de video. Sus aportes concretos son:

1. **Escala sin precedentes en visión egocéntrica.** Con 11,5 millones de frames frente al millón de ADL —el dataset egocéntrico previo más comparable, también grabado en entornos nativos—, EPIC-KITCHENS ofrece unas **90 veces más segmentos de acción anotados** y **4 veces más bounding boxes de objetos**, convirtiéndose en el dataset en primera persona más grande de su momento.
2. **Diversidad genuina de entornos.** 32 participantes implican **32 cocinas distintas**, en cuatro ciudades y con 10 nacionalidades. Esto permite, por primera vez, evaluar la **generalización a entornos nuevos** de manera rigurosa, separando cocinas vistas de cocinas no vistas en el test.
3. **Anotación por narración del propio actor.** El pipeline de etiquetado arranca con un **comentario en vivo** grabado por los mismos participantes, lo que refleja la **intención verdadera** de quien ejecutó la acción, algo que un observador externo no puede recuperar con la misma fidelidad.
4. **Tres benchmarks con baselines y leaderboards.** Detección de objetos, reconocimiento de acciones y anticipación de acciones, con protocolos de evaluación cuidadosos y ground-truth reservado para mantener la competencia justa.

## 4. Recolección y anotación

### 4.1. Recolección de los datos

El dataset fue grabado por 32 individuos entre **mayo y noviembre de 2017**, distribuidos así: **15 en Bristol (Reino Unido), 8 en Toronto (Canadá), 8 en Catania (Italia) y 1 en Seattle (Estados Unidos)**. A cada participante se le pidió capturar todas sus visitas a la cocina durante **tres días consecutivos**, empezando a grabar inmediatamente antes de entrar y deteniéndose solo al salir. Participaron de forma **voluntaria y sin remuneración**. Se les pidió estar solos en la cocina durante las grabaciones —para capturar actividad de una sola persona— y retirar objetos que revelaran su identidad, como retratos o espejos.

La captura se hizo con una **GoPro montada en la cabeza**, con un soporte ajustable para controlar el punto de vista según la altura de cada persona y su cocina. Antes de cada grabación, los participantes verificaban la batería y el encuadre con la app GoPro Capture, de modo que sus manos extendidas quedaran aproximadamente en el centro del frame. La cámara se configuró con **campo de visión lineal, 59,94 fps y resolución Full HD de 1920×1080**, aunque algunos sujetos hicieron cambios menores (FOV ancho o ultra-ancho, u otras resoluciones y tasas de cuadros) al encender y apagar el dispositivo a lo largo de varios días. En promedio, cada persona grabó **1,7 horas** (máximo 4,6 horas) y produjo **13,6 secuencias**. Cocinar una sola comida puede abarcar múltiples secuencias, según si la persona permanece en la cocina o sale y vuelve más tarde.

### 4.2. Narración: el actor como anotador

Como hacer *crowd-sourcing* de anotaciones sobre videos tan largos es muy costoso, los autores diseñaron una **primera anotación gruesa a cargo de los propios participantes**. Después de terminar todas sus grabaciones, cada persona veía sus videos y **narraba en voz alta las acciones ejecutadas**, usando un dispositivo de grabación de audio. Se optó por audio en vez de subtítulos escritos porque es mucho más rápido para el participante, que así estaba más dispuesto a colaborar. Estas narraciones son análogas a un **comentario en vivo** del video.

Las instrucciones de narración eran deliberadamente laxas: usar las palabras que se prefiera, en tiempo presente, con pares verbo-objeto (por ejemplo, *"wash carrot"*), pudiendo omitir artículos y pronombres (*"cut kiwi"* en vez de *"I cut the kiwi"*), usar preposiciones cuando hagan falta (*"pour water into kettle"*), la conjunción *"and"* para acciones simultáneas (*"hold mug and pour water"*) y narrar de nuevo si una acción se prolonga (*"still stirring soup"*). Cada participante narró en **inglés** si tenía suficiente fluidez, o en su **idioma nativo**. En total se usaron **cinco idiomas**: 17 narraron en inglés, 7 en italiano, 6 en español, 1 en griego y 1 en chino.

La decisión de recoger las narraciones de los **propios participantes** responde a que ellos son los más calificados para etiquetar la actividad —fueron quienes la ejecutaron—, a diferencia de un observador independiente. Y la narración **posterior** a la grabación permite que la persona realice sus actividades sin ser molestada, sin preocuparse por etiquetar mientras cocina.

Como las APIs automáticas de audio-a-texto fallaron —esperan corpus coherentes y oraciones completas—, las transcripciones se obtuvieron manualmente vía Amazon Mechanical Turk (AMT), usando la herramienta de alineación de subtítulos automáticos de YouTube para lograr timings precisos; para las narraciones no inglesas, trabajadores de AMT también tradujeron las frases. Para asegurar consistencia, cada tarea (HIT) se enviaba tres veces y se seleccionaban las transcripciones con **distancia de edición 0** respecto de al menos otra. En total se recolectaron **39.596 narraciones de acción**, es decir, una narración cada **4,9 segundos** de video, con un promedio de **2,8 palabras por frase**.

Los autores son honestos respecto de las limitaciones de las narraciones como ground-truth: (i) son **incompletas**, porque los participantes fueron selectivos —etiquetaron más las acciones de *abrir* que las de *cerrar*, porque su atención ya se había movido al siguiente objetivo—; (ii) están **retrasadas temporalmente**, ocurren después de la acción; y (iii) usan **vocabulario libre**, lo que se resuelve agrupando verbos y sustantivos en clases.

### 4.3. Segmentos de acción y el algoritmo de acuerdo

Para cada frase narrada se ajustan los tiempos de inicio y fin de la acción mediante AMT. Cada HIT contiene hasta 10 frases narradas consecutivas, y los anotadores marcan $A_i = [t^s_i, t^e_i]$ como el inicio y fin de la $i$-ésima acción, con dos restricciones para reducir ruido: la acción debe durar al menos 0,5 segundos, y no puede empezar antes que la anterior. Se piden $K_a = 4$ anotadores por HIT. Dada una anotación $A_i(j)$, se calcula el acuerdo como el promedio de solapamiento (IoU) con las demás:

$$\alpha_i(j) = \frac{1}{K_a} \sum_{k=1}^{K_a} \mathrm{IoU}\big(A_i(j), A_i(k)\big)$$

Se elige al anotador de máximo acuerdo, $\hat{j} = \arg\max_j \alpha_i(j)$, y a su mejor par, $\hat{k} = \arg\max_k \mathrm{IoU}(A_i(\hat{j}), A_i(k))$. El segmento ground-truth se define combinando ambas anotaciones cuando concuerdan fuertemente:

$$A_i = \begin{cases} \mathrm{Union}\big(A_i(\hat{j}), A_i(\hat{k})\big), & \text{si } \mathrm{IoU}(A_i(\hat{j}), A_i(\hat{k})) > 0{,}5 \\ A_i(\hat{j}), & \text{en caso contrario} \end{cases}$$

Se recolectaron así etiquetas para **39.564 segmentos de acción** (duración media $\mu = 3{,}7$ s, desviación $\sigma = 5{,}6$ s), el **99,9 % de los segmentos narrados**.

### 4.4. Bounding boxes de objetos activos

Los sustantivos narrados corresponden a los **objetos relevantes para la acción**. Para cada objeto asociado a un segmento $A_i = [t^s_i, t^e_i]$, se consideran los frames en la ventana ampliada $[t^s_i - 2\text{s}, \, t^e_i + 2\text{s}]$ como candidatos a anotar la caja delimitadora. Cada HIT busca anotar un objeto durante hasta 25 segundos (50 frames consecutivos a 2 fps). La calidad se controla exigiendo un $\mathrm{IoU} \geq 0{,}7$ contra anotaciones patrón al inicio de cada HIT. Se piden $K_o = 3$ trabajadores por HIT y se selecciona al de máximo acuerdo $\beta$:

$$\beta(q) = \sum_f \max_{j \neq q} \max_{k,l} \mathrm{IoU}\big(BB(j, f, k), BB(q, f, l)\big)$$

Los empates se resuelven a favor de quien entrega cajas más ajustadas. El **77 %** de las anotaciones solicitadas produjo al menos una caja, para un total de **454.255 bounding boxes** ($\mu = 1{,}64$ cajas por frame, $\sigma = 0{,}92$).

### 4.5. Clases de verbos y sustantivos

Como los participantes anotaron en texto libre y en varios idiomas, se recogió una gran variedad de verbos y sustantivos que hubo que **agrupar en clases** con solapamiento semántico mínimo, para acomodar los enfoques típicos de detección y reconocimiento multiclase donde cada ejemplo pertenece a una sola clase. Se estimó la categoría gramatical (POS) con el modelo de SpaCy: se toma el primer verbo de la oración y todos los sustantivos, resolviendo pronombres (*"it"*) con el sustantivo de la narración inmediatamente anterior. El clustering automático con WordNet, Word2Vec y el algoritmo de Lesk produjo demasiados grupos sin sentido por el contexto limitado, así que los verbos se agruparon **manualmente** y los sustantivos de forma **semiautomática**. El resultado son **125 clases de verbos** ($C_V$) y **331 clases de sustantivos** ($C_N$), organizadas en **19 supercategorías** —de las cuales 9 son comidas y bebidas, y el resto, elementos esenciales de cocina—.

La calidad se auditó sobre 300 muestras aleatorias con estas tasas de error: límites de segmentos de acción **5,7 %**, bounding boxes de objetos **6,3 %**, clases de verbos **3,3 %** y clases de sustantivos **6,0 %** —comparables a las de datasets recientes—.

## 5. Estadísticas

El siguiente cuadro resume las cifras que definen la escala de EPIC-KITCHENS:

| Dimensión | Valor |
|---|---|
| Participantes / cocinas | 32 |
| Ciudades (Norteamérica y Europa) | 4 |
| Nacionalidades | 10 |
| Horas de video | 55 |
| Frames | 11,5 M |
| Narraciones de acción | 39.596 |
| Segmentos de acción anotados | 39.564 |
| Bounding boxes de objetos | 454.255 |
| Clases de verbos ($C_V$) | 125 |
| Clases de sustantivos ($C_N$) | 331 |
| Supercategorías de objetos | 19 (9 de comida/bebida) |
| Idiomas de narración | 5 |
| Resolución / tasa de cuadros | 1920×1080 / 59,94 fps |

Frente al dataset egocéntrico previo más cercano (ADL), EPIC-KITCHENS multiplica por **90** los segmentos de acción y por **4** las bounding boxes, y es varias veces mayor que datasets de tercera persona centrados en actividades de interacción con objetos.

## 6. Los desafíos que define

Los autores reservaron el ground-truth del **27 %** de los datos para test y estructuraron dos particiones que apuntan al problema central de la **generalización a entornos nuevos**:

- **Cocinas vistas (S1):** cada cocina aparece tanto en entrenamiento como en test (aproximadamente 80 % de secuencias en train, 20 % en test), sin dividir secuencias individuales.
- **Cocinas no vistas (S2):** todas las secuencias de una misma cocina van completas a entrenamiento o a test; se reservan las 4 cocinas de 4 participantes para test. Aunque S2 es solo el 7 % del dataset en cantidad de frames, el desafío sigue siendo considerable.

Sobre estas particiones se definen tres benchmarks:

**6.1. Detección de objetos.** El objetivo es detectar todas las clases $C_N$, restringiendo la evaluación por clase a las imágenes donde el objeto fue anotado (recordando que solo se anotaron objetos **activos**: pre-, durante y post-interacción). El baseline es **Faster R-CNN** con base ResNet-101 preentrenada en MS-COCO, midiendo mAP a distintos umbrales de IoU (0,05, 0,5 y 0,75). Con **202 clases many-shot** ($\geq 100$ cajas en train) y **88 few-shot** ($\geq 10$ y $< 100$), los resultados muestran que los objetos de EPIC-KITCHENS son **más difíciles de detectar** que en datasets existentes: el desempeño a IoU $> 0{,}5$ queda por debajo del 40 %, y el régimen few-shot rinde mucho peor que el many-shot. Curiosamente, el desempeño en cocinas vistas y no vistas es comparable, lo que indica buena capacidad de generalización entre entornos para los objetos.

**6.2. Reconocimiento de acciones.** Dado un segmento $A_i = [t^s_i, t^e_i]$, se clasifica en su clase de acción, definida como el par $(c_v, c_n)$ con $c_v \in C_V$ y $c_n \in C_N$. El baseline es la **Temporal Segment Network (TSN)** con arquitectura Inception, ajustada para predecir verbo y sustantivo de forma conjunta con pérdidas independientes, entrenando streams espacial (RGB) y temporal (flujo óptico TV-L1). El desafío de acertar **verbo y sustantivo a la vez** sigue siendo significativo: top-1 de acción de **20,5 %** en cocinas vistas y **10,9 %** en no vistas. Es decir, para muchos ejemplos el modelo acierta solo una de las dos etiquetas, y **generalizar a entornos no vistos es más difícil para las acciones que para los objetos**.

**6.3. Anticipación de acciones.** Aquí el objetivo es **pronosticar la próxima acción antes de que ocurra**. Definiendo $\tau_a$ como el "tiempo de anticipación" (cuánto antes se reconoce la acción) y $\tau_o$ como el "tiempo de observación" (cuánto video previo se observa), se predice la clase de acción $C_a$ observando el segmento $[t^s_i - (\tau_a + \tau_o), \, t^s_i - \tau_a]$. Con $\tau_a = 1$ s y $\tau_o = 1$ s, y usando de nuevo TSN, el desempeño **cae respecto del reconocimiento**, como era de esperar. El modelo tiende a **sobre-predecir "put"** como próxima acción: una vez que un objeto se levanta, aprende a creer que lo siguiente será dejarlo. Los autores señalan que harían falta métodos que entiendan el **objetivo de largo plazo** y usen historia multiescala para superar esa tendencia. La anticipación tiene, además, implicaciones directas en la **vida asistida** (*assertive living*): un sistema wearable que anticipe la próxima acción del usuario podría, por ejemplo, activar electrodomésticos inteligentes de forma anticipada.

## 7. Impacto

EPIC-KITCHENS se convirtió en el **benchmark de referencia para la visión egocéntrica**. Al ofrecer escala, diversidad de entornos y anotación densa de acciones y objetos, habilitó una agenda de investigación que va mucho más allá de los tres desafíos iniciales. Los propios autores anticiparon extensiones hacia **localización temporal de acciones**, *parsing* de video, diálogo visual, *goal completion* y **determinación de habilidad** (cuán bien ejecuta alguien una tarea, por ejemplo preparar el desayuno). La existencia de leaderboards con ground-truth reservado —y el énfasis en el desempeño **en tiempo real**, crucial en el dominio wearable— convirtió al dataset en un motor de competencia para la comunidad. La partición vistas/no vistas, en particular, instaló la **generalización a entornos nuevos** como criterio de evaluación estándar, más exigente que medir precisión sobre datos del mismo entorno de entrenamiento.

## 8. Limitaciones

El propio paper reconoce varias limitaciones inherentes a su diseño:

- **Narraciones incompletas y sesgadas.** Los participantes narraron selectivamente, favoreciendo ciertas acciones (*abrir* sobre *cerrar*), lo que obliga a evaluar solo acciones narradas y deja fuera parte de lo que realmente ocurre en el video.
- **Solo objetos activos.** Las bounding boxes cubren únicamente objetos involucrados en la interacción, no todos los objetos presentes en la escena, lo que restringe la evaluación de detección a las imágenes donde el objeto fue anotado.
- **Vocabulario libre agrupado a mano.** El clustering automático de verbos y sustantivos falló por falta de contexto, y hubo que agrupar manual y semiautomáticamente, un proceso costoso y sujeto a decisiones subjetivas.
- **Actividad de una sola persona.** Se pidió a los participantes estar solos, de modo que el dataset **excluye interacciones interpersonales**, un aspecto importante de la vida diaria que queda fuera de alcance.
- **Baselines lejos de resolver las tareas.** Los métodos evaluados están todavía muy lejos de tackear estos desafíos con alta precisión —especialmente en cocinas no vistas y en anticipación—, lo que confirma la dificultad del dataset pero también que las soluciones estaban por construirse.

## 9. Conexión con la Clase 36 (Introduction to Video Analysis)

La Clase 36 introduce el análisis de video como un dominio con desafíos propios: reconocimiento de acciones, detección, anticipación, adaptación de dominio y recuperación, tanto en video **trimmed** como **untrimmed**. EPIC-KITCHENS es el ejemplo canónico de **por qué el análisis de video realista es difícil**, y aporta a la clase dos ejes que conviene que el estudiante internalice:

1. **La perspectiva egocéntrica cambia el problema.** El video en primera persona no es simplemente "video con otro ángulo": trae oclusiones frecuentes por las manos, movimiento de cámara acoplado a la cabeza, objetos que entran y salen del campo de visión, y una relación estrecha entre lo que se ve y la **intención** de quien actúa. El *egocentric action recognition* es un subcampo con dinámicas distintas al reconocimiento de acciones en tercera persona.

2. **Untrimmed es el escenario real.** Los benchmarks de clips cortos ocultan la parte difícil del problema —¿dónde empieza y termina cada acción en un flujo continuo?—. EPIC-KITCHENS mantiene la grabación completa, con multitasking y acciones encadenadas, lo que conecta directamente con las tareas de **localización temporal** y **anticipación** que la clase presenta como frontera del video analysis.

El descenso de desempeño entre cocinas vistas (top-1 de acción 20,5 %) y no vistas (10,9 %) es, además, una ilustración concreta del problema de **domain adaptation** en video: un modelo entrenado en unos entornos no transfiere trivialmente a entornos nuevos, y medir esa brecha es parte esencial del análisis riguroso.

**Enlaces internos:**

- Clase: [/clases/clase-36](/clases/clase-36) — Introduction to Video Analysis.
- Tema afín: reconocimiento de acciones y anticipación en video egocéntrico y untrimmed.

---

**Nota sobre relevancia para salud.** El video egocéntrico untrimmed que EPIC-KITCHENS formaliza tiene aplicaciones directas en salud y cuidado. Una cámara wearable en primera persona que reconozca y anticipe **actividades de la vida diaria** —cocinar, comer, tomar medicamentos, higiene— es la base tecnológica para sistemas de **asistencia a personas mayores** y de vida asistida (*assertive living*), capaces de detectar omisiones (por ejemplo, que un adulto mayor no haya tomado su medicación o no haya comido), de estimar el nivel de independencia funcional a partir de cómo se ejecutan las tareas cotidianas, o de anticipar la próxima acción para activar dispositivos del hogar antes de que se necesiten. La anotación densa de interacciones mano-objeto y el énfasis del paper en la **anticipación en tiempo real** anticipan usos clínicos y de monitoreo domiciliario: cámaras wearables que documentan la ejecución de rutinas de rehabilitación, seguimiento de deterioro cognitivo mediante cambios en patrones de actividad, o soporte a pacientes con demencia mediante recordatorios contextuales gatillados por lo que la persona está a punto de hacer. La misma dificultad que hace difícil al dataset —variabilidad de entornos, oclusiones, acciones entrelazadas— es la que un sistema clínico real debe enfrentar, por lo que EPIC-KITCHENS ofrece un banco de pruebas realista para esa clase de tecnología asistencial.
