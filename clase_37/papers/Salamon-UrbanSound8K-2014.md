# A Dataset and Taxonomy for Urban Sound Research (UrbanSound / UrbanSound8K) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *A Dataset and Taxonomy for Urban Sound Research*.
- **Autores:** Justin Salamon, Christopher Jacoby, Juan Pablo Bello. Music and Audio Research Laboratory (MARL) y Center for Urban Science and Progress (CUSP), New York University.
- **Venue:** *Proceedings of the 22nd ACM International Conference on Multimedia* (ACM MM 2014), Orlando, Florida, 3–7 de noviembre de 2014.
- **DOI:** [10.1145/2647868.2655045](http://dx.doi.org/10.1145/2647868.2655045).
- **Financiamiento:** una beca semilla del Center for Urban Science and Progress (CUSP) de NYU.

El paper identifica **dos barreras** que frenaban la investigación en clasificación automática de sonido urbano: (1) la **falta de una taxonomía común** para nombrar y organizar las fuentes de sonido, y (2) la **escasez de datos anotados, grandes y del mundo real**. Para atacarlas, los autores aportan tres piezas encadenadas: una **taxonomía de sonidos urbanos** que organiza jerárquicamente las fuentes acústicas de una ciudad; un **dataset, UrbanSound**, con 27 horas de audio de campo, de las cuales 18.5 horas corresponden a ocurrencias de eventos sonoros anotadas manualmente en 10 clases; y un **subconjunto, UrbanSound8K**, diseñado específicamente para entrenar y comparar clasificadores, con 8.732 fragmentos (*slices*) de a lo más 4 segundos, ya repartidos en **10 folds predefinidos** para validación cruzada reproducible. Sobre este material corren una batería de experimentos con un sistema de referencia (*baseline*) basado en **MFCC + clasificadores clásicos** para caracterizar las dificultades del problema, no para maximizar la exactitud.

Para la **Clase 37 (Datasets y Herramientas para Audio)** este trabajo es el ejemplo canónico de cómo se construye un **benchmark de clasificación de sonido ambiental (ESC, *Environmental Sound Classification*)**. UrbanSound8K es citado en la clase como dataset didáctico —"8.732 clips de hasta 4 s, 10 clases urbanas, en 10 folds"— y su valor pedagógico no está solo en las cifras, sino en las **decisiones de diseño**: por qué se necesita una taxonomía, cómo se anota con marcas de tiempo y saliencia, y por qué los 10 folds vienen fijados de fábrica y **no deben re-barajarse**.

## 2. Contexto: por qué faltaban datos y una taxonomía común

La clasificación automática de **sonido ambiental** venía creciendo como campo, con aplicaciones a la indexación y recuperación de contenido multimedia a gran escala. El análisis sonoro de entornos urbanos, en particular, ganaba interés impulsado por las **redes de sensores multimedia** y por la enorme cantidad de contenido urbano subido a repositorios en línea. Sin embargo, comparado con áreas vecinas y maduras como el habla, la música o la bioacústica, el trabajo sobre acústica urbana era relativamente escaso. Peor aún: cuando existía, solía enfocarse en clasificar el **tipo de escena auditiva** (calle, parque) en lugar de **identificar las fuentes de sonido** dentro de esas escenas (bocina de auto, motor al ralentí, canto de pájaro). El paper apunta a lo segundo, que es lo verdaderamente útil.

Los autores diagnostican dos obstáculos concretos:

1. **Falta de datos etiquetados.** El trabajo previo se apoyaba en audio de películas o programas de TV cuidadosamente producidos, en entornos muy específicos (ascensores, oficinas), o en datasets comerciales/propietarios inaccesibles. El enorme esfuerzo de anotar manualmente grabaciones de campo reales hacía que esos conjuntos fueran pequeños; como referencia, el dataset de detección de eventos del *IEEE AASP Challenge* tenía apenas **24 grabaciones por cada una de 17 clases**.

2. **Falta de un vocabulario común.** Sin una taxonomía compartida, cada estudio agrupaba los sonidos en categorías semánticas distintas, lo que hacía **imposible comparar resultados** entre trabajos. La categorización taxonómica de sonidos ambientales es de hecho un primer paso habitual en clasificación, y había sido muy estudiada en la investigación perceptual de *soundscapes*; pero los esfuerzos específicos sobre sonido urbano solían ser subconjuntos de taxonomías más amplias de entornos acústicos y no cubrían las necesidades de un análisis urbano sistemático.

La tesis del paper es que **ambos problemas se resuelven juntos**: una taxonomía da el marco conceptual y las etiquetas; un dataset grande, real y libre da la evidencia empírica. Solo teniendo las dos cosas la comunidad puede acumular resultados comparables.

## 3. Contribución central

El aporte se organiza en tres contribuciones que se sostienen mutuamente:

- **La Taxonomía de Sonidos Urbanos**: una jerarquía de fuentes de sonido pensada para ofrecer un marco común de investigación, anclada tanto en la literatura de *soundscapes* como en datos reales de quejas por ruido.
- **UrbanSound**: el —según los autores— **dataset libre más grande de eventos sonoros urbanos etiquetados** disponible para investigación hasta ese momento, con 27 horas de grabaciones de campo y 18.5 horas de ocurrencias anotadas en 10 clases.
- **UrbanSound8K**: un subconjunto de fragmentos cortos ($\leq 4$ s) diseñado para el entrenamiento y, sobre todo, para la **comparación reproducible** vía 10 folds predefinidos.

A esto se suma el estudio experimental con un *baseline* de MFCC, cuyo objetivo declarado no es récord de exactitud sino **entender las características y desafíos del dataset**.

## 4. Método

### 4.1. La taxonomía de sonidos urbanos

Los autores fijan tres requisitos de diseño para la taxonomía: (1) debe **incorporar** taxonomías previas; (2) debe ser **tan detallada como sea posible**, llegando a fuentes de bajo nivel como "bocina de auto" (en vez de "transporte") o "martillo neumático" (en vez de "construcción"); y (3) debe, en su primera iteración, **priorizar los sonidos relevantes** para la investigación urbana, especialmente los que contribuyen a la **contaminación acústica**.

Para el requisito (1) parten del subconjunto dedicado al entorno acústico urbano de una taxonomía de *soundscapes* previa, y definen **cuatro grupos de nivel superior** comunes a la mayoría de las taxonomías propuestas:

$$\text{human} \quad\bullet\quad \text{nature} \quad\bullet\quad \text{mechanical} \quad\bullet\quad \text{music}$$

Cada grupo se ramifica en subgrupos y finalmente en **hojas** que corresponden a fuentes de sonido concretas. Por ejemplo: *human* se abre en voz (habla, risa, grito, llanto, tos, canto, infante, niños) y movimiento (pasos); *nature* en elementos (viento, agua, trueno), animales (ladrido y aullido de perro, trino de pájaro) y vegetación (hojas); *mechanical* en construcción (martillo neumático, martilleo, perforación, sierra, explosión, motor), ventilación (aire acondicionado), transporte motorizado (carretera, riel, aéreo, marítimo) y no motorizado (bicicleta, patineta); y *music* en amplificada/no amplificada, en vivo/grabada (fiesta, club, radio de auto, camión de helados, parlantes). En la figura de la taxonomía, los rectángulos redondeados representan clases semánticas de alto nivel y los rectángulos de esquinas rectas son las hojas (fuentes concretas como sirena o pasos); por concisión, una hoja puede ser compartida por varias clases de alto nivel.

Para el requisito (3), los autores hacen algo distintivo: examinan **todas las quejas por ruido presentadas al servicio 311 de la ciudad de Nueva York** entre 2010 y la fecha del trabajo —**más de 370.000 quejas**— y construyen la taxonomía en torno a las categorías y fuentes **más frecuentemente denunciadas**: construcción (p. ej. martillo neumático), ruido de tráfico (bocinas de autos y camiones, motores al ralentí), música fuerte, aires acondicionados y ladridos de perro. Como el número de fuentes de sonido posibles en una ciudad es enorme (potencialmente infinito), los autores tratan la taxonomía como un **trabajo en progreso permanente**, extensible mediante colaboración de la comunidad.

### 4.2. Las 10 clases y la recolección desde Freesound

Del conjunto de hojas de la taxonomía, seleccionan **10 clases de bajo nivel** para anotar:

**aire acondicionado, bocina de auto, niños jugando, ladrido de perro, perforación (drilling), motor al ralentí (engine idling), disparo (gun shot), martillo neumático (jackhammer), sirena y música callejera (street music)**.

Con excepción de "niños jugando" y "disparo" —añadidas por variedad—, todas fueron escogidas por su **alta frecuencia en las quejas por ruido**. Se limitó deliberadamente el número a 10 clases por el costo de anotación manual, considerándolas un buen punto de partida.

Antes de compilar el dataset fijaron tres metas: que contuviera sonidos de un entorno urbano, que **todas las grabaciones fueran de campo reales**, y que el conjunto fuera suficientemente grande y variado —en sonidos y condiciones de grabación— como para entrenar algoritmos escalables capaces de analizar datos reales de redes de sensores o repositorios multimedia. Para lograrlo recurrieron a **Freesound**, un repositorio en línea con más de 160.000 grabaciones subidas por usuarios bajo licencia *Creative Commons*, rico en grabaciones de campo urbanas. Vía la **API de Freesound** buscaron y descargaron un subconjunto del repositorio, aprovechando los **metadatos aportados por los usuarios** (título, descripción y *tags*) para acelerar la anotación.

El flujo de recolección fue:

1. **Descarga por clase.** Para cada clase descargaron todos los sonidos que devolvía el buscador de Freesound al usar el nombre de la clase como consulta (p. ej. "jackhammer"), obteniendo **más de 3.000 grabaciones** que sumaban **poco más de 60 horas** de audio.
2. **Filtrado manual.** Escucharon cada grabación e inspeccionaron sus metadatos, conservando solo las que eran **grabaciones de campo reales** y donde la clase de interés estaba efectivamente presente en algún punto. Tras este filtro quedaron **1.302 grabaciones**, poco más de **27 horas**.

### 4.3. Anotación con marcas de tiempo y saliencia

Sobre esas 1.302 grabaciones, y usando el editor **Audacity**, etiquetaron los **tiempos de inicio y fin de cada ocurrencia** del sonido de interés dentro de cada grabación. A cada ocurrencia añadieron además una descripción de **saliencia**: si subjetivamente se percibía en **primer plano (foreground, FG)** o en **fondo (background, BG)** de la grabación. La etiqueta BG se usa también cuando hay otras fuentes igualmente salientes.

El resultado fue un total de **3.075 ocurrencias etiquetadas** que suman **18.5 horas de audio etiquetado**. La colección completa —1.302 grabaciones de longitud original con sus anotaciones de ocurrencia y saliencia— constituye **UrbanSound**, disponible libremente para investigación. El audio se entrega en el mismo formato en que fue subido a Freesound. Un detalle importante para el modelado: la duración de las ocurrencias **varía mucho**, desde 1–2 s (p. ej. disparos) hasta más de 30 s (sonidos continuos como martillos neumáticos o motores al ralentí).

### 4.4. UrbanSound8K: fragmentos cortos y los 10 folds

Para la investigación en **identificación de fuentes de sonido**, los autores derivan un subconjunto de fragmentos cortos, **UrbanSound8K**. La justificación del límite de 4 segundos viene de un trabajo previo (Chu et al.) que, mediante una prueba de escucha, encontró que **4 segundos bastaban para que sujetos humanos identificaran sonidos ambientales con 82% de exactitud**. Siguiendo ese hallazgo, fijan una **duración máxima de ocurrencia de 4 s** y **segmentan las ocurrencias más largas en fragmentos de 4 s usando una ventana deslizante con salto (*hop*) de 2 s**. Para evitar grandes desbalances entre clases, imponen un **tope de 1.000 fragmentos por clase**, lo que da un total de **8.732 fragmentos etiquetados (8.75 horas)**. La distribución de fragmentos por clase, desglosada por saliencia FG/BG, se reporta en el paper.

La pieza que convierte a UrbanSound8K en un **benchmark** es la **división en 10 folds**. Los autores explican con cuidado el riesgo que evitan. Como muchos fragmentos provienen de **una misma grabación original de Freesound**, si los folds se generaran de manera **completamente aleatoria**, podrían terminar fragmentos de la *misma grabación* tanto en el conjunto de entrenamiento como en el de prueba. Eso produce una **fuga de información** que infla artificialmente la exactitud: el clasificador reconoce la grabación específica en vez de generalizar a la clase. Para impedirlo, diseñaron un proceso de asignación aleatoria de fragmentos a folds con dos restricciones:

$$
\textbf{(R1) } \text{todos los fragmentos de una misma grabación} \rightarrow \text{el mismo fold}
$$
$$
\textbf{(R2) } \text{balancear el número de fragmentos por fold para cada clase}
$$

El subconjunto publicado ya trae los fragmentos **agrupados en 10 folds generados con esta metodología**. Así, cualquier investigador que evalúe con esos folds obtiene resultados **insesgados y directamente comparables** con el *baseline* del paper y entre estudios. De aquí se desprende la regla de oro operativa: **no re-barajar los datos**. Rehacer los folds aleatoriamente rompe R1, reintroduce la fuga por grabación y produce cifras optimistas que no son comparables con la literatura.

## 5. Baseline y resultados

El sistema de referencia extrae **MFCC (Mel-Frequency Cepstral Coefficients)** con la librería **Essentia**. Los MFCC se usan ampliamente en análisis de sonido ambiental y sirven como *baseline* competitivo para comparar técnicas nuevas. La extracción es por *frame*, con ventana de **23.2 ms y 50% de solapamiento**, calculando **40 bandas Mel entre 0 y 22050 Hz** y quedándose con los **primeros 25 coeficientes MFCC** (sin pre-énfasis ni *liftering*). Los valores por *frame* de cada coeficiente se resumen a lo largo del tiempo con estadísticas: mínimo, máximo, mediana, media, varianza, *skewness*, *kurtosis* y la media y varianza de la primera y segunda derivadas. Esto da un **vector de características de dimensión 225 por fragmento** —un enfoque *bag-of-frames*.

Los experimentos se corren con la suite **Weka**, siempre con **validación cruzada de 10 folds** (los folds predefinidos) y selección de atributos por correlación dentro de cada fold para evitar sobreajuste; todos los clasificadores usan parámetros por defecto, y se reporta la **exactitud promedio sobre los 10 folds**. Se comparan cinco algoritmos: árbol de decisión (J48), k-NN ($k=5$), *random forest* (500 árboles), **SVM con kernel RBF** y un clasificador de voto mayoritario (ZeroR) como piso.

Hallazgos principales:

- **Duración del fragmento.** Generando 10 copias de UrbanSound8K con duración máxima variando de 10 s a 1 s, el desempeño se mantiene estable de 10 a 6 s y luego **decae gradualmente**. Para el mejor clasificador (SVM) **no hay diferencia estadísticamente significativa entre 6 s y 4 s**, mientras que por debajo de 4 s la caída sí es significativa. Esto respalda empíricamente la elección de 4 s.
- **Por clase.** Las clases con **eventos rápidos** —disparo, sirena— son claramente identificables en escalas temporales cortas y casi no se afectan por la duración; en cambio, **música callejera y niños jugando** caen de forma casi monótona al acortar el fragmento, lo que sugiere que necesitan **escalas temporales más largas** y que un análisis multi-escala podría ser una vía relevante.
- **Confusiones (matriz de confusión, SVM).** El clasificador confunde sobre todo **tres pares** de clases: aires acondicionados con motores al ralentí, martillos neumáticos con perforaciones, y niños jugando con música callejera. Tiene sentido: el timbre de cada par es muy parecido (en el último par, por la presencia de tonos armónicos complejos). El enfoque *bag-of-frames* con MFCC **falla especialmente en sonidos continuos, de banda ancha y tipo ruido**.
- **Saliencia.** Como se esperaba, hay una diferencia considerable de desempeño entre fragmentos FG y BG (los MFCC son sensibles al ruido). La excepción es la **sirena**, probablemente porque su contenido en frecuencia no se solapa con el de otras fuentes (por diseño). Esto señala un desafío central del dataset: **identificar fuentes en presencia de ruido de fondo real**.

## 6. Impacto

UrbanSound8K se convirtió en el **benchmark estándar de facto para clasificación de sonido ambiental urbano**. Su combinación de tamaño razonable, licencia libre, 10 clases bien definidas y —crucialmente— **folds predefinidos** lo transformó en el conjunto de referencia sobre el que se midieron durante años los avances del área, especialmente cuando el *deep learning* llegó al audio. El propio Salamon lo usó después como base de trabajos muy influyentes sobre **CNN para clasificación de sonido ambiental** y **aumento de datos (*data augmentation*)** para audio, y la comunidad lo adoptó como el punto de comparación obligado. Junto con ESC-50 y, más tarde, AudioSet, UrbanSound8K forma parte del canon de datasets que estructuran la enseñanza y la evaluación en ESC. Su legado no es solo el dataset: es haber **normalizado la práctica de publicar folds fijos** para garantizar comparaciones honestas, una lección metodológica que trasciende el audio.

## 7. Limitaciones

- **Solo 10 clases.** El número se limitó por el costo de anotación manual; los propios autores lo plantean como punto de partida, no como cobertura exhaustiva del paisaje sonoro urbano.
- **Saliencia subjetiva.** Las etiquetas FG/BG son juicios subjetivos de los anotadores, por lo que el efecto de la interferencia de fondo **no puede cuantificarse rigurosamente** desde los experimentos.
- **Baseline deliberadamente simple.** El enfoque *bag-of-frames* con MFCC ignora la **dinámica temporal** de la energía y el timbre, y por eso falla en los sonidos continuos tipo ruido; los autores lo señalan como línea de trabajo futuro (modelar mejor la evolución temporal).
- **Sesgo de origen.** Al provenir de Freesound y de las quejas del servicio 311 de Nueva York, tanto la selección de clases como las condiciones de grabación llevan un sesgo hacia ese contexto urbano particular.
- **Tope de 1.000 fragmentos por clase.** Evita desbalances groseros pero no garantiza balance perfecto ni representa la frecuencia real de cada sonido en la ciudad.

## 8. Conexión con la Clase 37 (Datasets y Herramientas para Audio)

La Clase 37 presenta UrbanSound8K como **dataset didáctico** para clasificación de sonido ambiental, y este paper es la fuente primaria que explica de dónde salen las cifras que la clase cita: **8.732 clips de $\leq 4$ s, 10 clases urbanas, 10 folds**. Tres ideas del paper son las que conviene que el estudiante internalice:

1. **Un benchmark es más que audio: es taxonomía + anotación + protocolo de evaluación.** La secuencia taxonomía → recolección desde Freesound → anotación con marcas de tiempo y saliencia → fragmentación en 4 s → folds fijos es una **plantilla reutilizable** para construir cualquier dataset de audio serio.

2. **Respetar los folds no es una formalidad, es correctitud experimental.** La regla "todos los fragmentos de una misma grabación al mismo fold" existe para impedir fuga de información. Si un estudiante re-baraja UrbanSound8K con un *split* aleatorio, obtendrá exactitudes **artificialmente altas** e **incomparables** con la literatura. La lección operativa —**usar los 10 folds tal como vienen**— es exactamente la que la clase enfatiza.

3. **El baseline MFCC fija el punto de partida.** Antes de las CNN espectrográficas, MFCC + SVM/random forest era el estado del arte práctico; conocerlo permite dimensionar cuánto aportó el *deep learning* sobre el mismo benchmark y con el mismo protocolo.

**Enlaces internos sugeridos:**

- Clase: [/clases/clase-37](/clases/clase-37) — Datasets y Herramientas para Audio.
- Fundamento transversal: clasificación de sonido ambiental (ESC), MFCC y representaciones espectrográficas.

---

**Nota sobre relevancia para salud.** La clasificación de sonidos ambientales que este dataset popularizó es directamente aplicable al **monitoreo acústico del hogar y a la asistencia a personas** en contextos de salud. Un sistema entrenado para reconocer fuentes de sonido cotidianas puede detectar **sonidos de alerta** —una alarma, una sirena que se aproxima, un vidrio que se rompe, una caída, tos persistente o llanto— y disparar avisos en escenarios de teleasistencia para adultos mayores o personas con discapacidad auditiva, o alimentar sistemas de vida asistida por el entorno (*ambient assisted living*). La metodología del paper importa aquí tanto como la aplicación: la **anotación con saliencia** (primer plano vs. fondo) anticipa el desafío real de distinguir el evento clínicamente relevante del ruido doméstico de fondo, y la disciplina de **folds sin fuga** es la que evita sobreestimar el desempeño de un clasificador de salud antes de desplegarlo, donde un falso negativo puede tener consecuencias reales.
