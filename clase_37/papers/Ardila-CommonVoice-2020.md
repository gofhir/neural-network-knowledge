# Common Voice: A Massively-Multilingual Speech Corpus — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Common Voice: A Massively-Multilingual Speech Corpus*.
- **Autores:** Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M. Tyers, Gregor Weber. Afiliaciones: **Mozilla** (la mayoría), Indiana University (Tyers) y Artie, Inc. (Meyer).
- **Venue:** *Proceedings of the 12th Language Resources and Evaluation Conference* (**LREC 2020**).
- **Año:** 2020. **Preprint:** arXiv:1912.06670v2 (5 mar 2020), [arxiv.org/abs/1912.06670](https://arxiv.org/abs/1912.06670).
- **Palabras clave declaradas:** *spoken corpus, Automatic Speech Recognition, low-resource languages*.

El paper presenta **Common Voice**, un corpus de voz transcrita **masivamente multilingüe** pensado para investigación y desarrollo en tecnologías del habla, principalmente **reconocimiento automático del habla (ASR, *Automatic Speech Recognition*)**, aunque los autores señalan que también sirve para otros dominios como identificación de idioma. La tesis central es de infraestructura, no de arquitectura: para que el ASR deje de ser un privilegio de un puñado de idiomas de altos recursos hace falta **datos abiertos a gran escala en muchos idiomas**, y la forma sostenible de producirlos es el **crowdsourcing**. Common Voice construye ese dato mediante una plataforma web donde voluntarios **graban** oraciones leídas en pantalla y **validan** las grabaciones de otros por votación.

Las cifras que reporta el paper: la versión más reciente al momento de escribir incluye **29 idiomas**, y **hasta noviembre de 2019 hay 38 idiomas recolectando datos**. Más de **50.000 personas** han participado, produciendo **2.500 horas** de audio recolectado. Los autores afirman que, a su conocimiento, es **el mayor corpus de audio en dominio público** para reconocimiento del habla, tanto en número de horas como en número de idiomas. Como caso de uso de ejemplo, entrenan modelos ASR con el toolkit **DeepSpeech** de Mozilla y, aplicando *transfer learning* desde un modelo fuente en inglés, obtienen una mejora promedio de **5.99 ± 5.48** puntos en *Character Error Rate* para doce idiomas objetivo (alemán, francés, italiano, turco, catalán, esloveno, galés, irlandés, bretón, tártaro, chuvasio y cabilio). Para la mayoría de esos idiomas, estos son **los primeros resultados publicados de ASR end-to-end**.

Para la **Clase 37 (Datasets y Herramientas para Audio)** este paper es el arquetipo del **dataset de escala web crowdsourced con licencia abierta**: se cita junto a GigaSpeech y Emilia como ejemplo de corpus de "cientos de miles de horas", se distribuye en **MP3**, y encarna directamente el eje que la clase subraya —**sesgo y representación**: ¿el dataset cubre acentos, géneros e idiomas?

## 2. Contexto: el ASR concentrado en pocos idiomas y la escasez de datos abiertos

El punto de partida del paper es un diagnóstico sobre el estado de las tecnologías del habla: los datos de entrenamiento son **o prohibitivamente caros o directamente inexistentes** para la mayoría de los idiomas del mundo. El reconocimiento del habla comercial y académico funciona bien en inglés, mandarín y un pequeño grupo de idiomas de **altos recursos** porque para ellos existen corpus grandes; para los **miles de idiomas restantes** —los llamados *low-resource*— no hay dato suficiente, y sin dato no hay modelo. Los autores enmarcan esto como un problema no solo técnico sino de **valores**: sostienen que la tecnología del habla, como toda tecnología, "debería ser abierta y descentralizada".

El problema tiene dos caras. La primera es de **cobertura**: la investigación se concentra donde ya hay datos, lo que refuerza el desequilibrio. La segunda es de **licencia**: incluso cuando existen corpus multilingües de calidad, muchos no están disponibles bajo licencia abierta. El paper revisa los antecedentes más notorios y muestra que cada uno falla en al menos un eje:

- **Babel** (Gales et al., 2014) contiene datos de alta calidad de 22 idiomas minoritarios, pero **no se libera bajo licencia abierta**.
- **VoxForge** es el más parecido a Common Voice —también *community-driven*, multilingüe (17 idiomas) y con licencia abierta (GPL)—, pero **no tiene un pipeline de recolección sostenible** ni **paso de validación** de los datos.
- **M-AILABS** cubre 9 variedades lingüísticas con una licencia BSD modificada, pero **carece del componente comunitario**.

Contra este panorama, Common Voice se posiciona como una alternativa **sostenible y abierta** que permite recolectar por igual idiomas minoritarios y mayoritarios. La palabra "sostenible" es clave: el diseño busca que el corpus **crezca orgánicamente** a medida que las comunidades usan las herramientas provistas para traducir la interfaz, aportar oraciones de texto y, finalmente, grabar y validar voces en su idioma. El proyecto arrancó enfocado en inglés en **julio de 2017** y en **junio de 2018** se abrió a cualquier idioma.

## 3. Contribución central

La contribución de Common Voice es **doble** y es más de infraestructura que de modelo:

1. **Un corpus masivamente multilingüe en dominio público.** Todos los datos de voz se liberan bajo licencia **Creative Commons CC0**, es decir, **dominio público** sin restricciones de atribución ni de uso. Esto lo convierte, según los autores, en el mayor corpus de dominio público diseñado para ASR.
2. **Una plataforma de crowdsourcing sostenible.** No solo se publica un dataset estático: se publica el **mecanismo** que lo produce y lo hace escalar a nuevos idiomas —el sitio web/app de grabación, el sistema de validación por votos, el *Sentence Collector* para reunir texto, y la traducción de la interfaz vía Pontoon. El corpus es, en palabras del paper, "un proyecto vivo".

Como demostración de utilidad —no como aporte metodológico— el paper añade un conjunto de **experimentos de ASR multilingüe con transfer learning**, que muestran que el corpus sirve para entrenar modelos reales y que copiar capas desde un modelo fuente en inglés acelera y mejora el aprendizaje en idiomas objetivo con pocos datos.

## 4. Método: el modelo de crowdsourcing

El corazón del paper es el **proceso de creación del corpus**, dividido en dos actos que cualquier voluntario puede realizar desde el navegador: **grabar** y **validar**.

### 4.1. Grabación

Usando el sitio web de Common Voice o la app de iPhone, los contribuyentes **graban su voz leyendo oraciones** que aparecen en pantalla. La interfaz permite además **saltar** una oración o **reportarla como problemática** (por ejemplo, si el texto tiene errores o el audio es defectuoso). El dato producido es un par **⟨audio, transcripción⟩**: el audio es la lectura del voluntario, y la transcripción es la oración exacta que se le mostró —lo que hace que el etiquetado sea gratuito y confiable, porque el texto se conoce de antemano.

### 4.2. Validación por votación

Las grabaciones se verifican **por otros contribuyentes** mediante un sistema de votos simple. En la interfaz de validación, cada voluntario escucha un par ⟨audio, transcripción⟩ y lo marca como **correcto (*up-vote*)** o **incorrecto (*down-vote*)**. Las reglas son:

- Un máximo de **tres contribuyentes** escuchan cualquier clip.
- Si el clip recibe primero **dos *up-votes***, se marca como **válido**.
- Si recibe primero **dos *down-votes***, se marca como **inválido**.
- Un voluntario puede alternar libremente entre grabar y validar.

Solo los clips marcados como **válidos** entran en los conjuntos oficiales de entrenamiento, desarrollo y prueba de cada idioma. Los clips que al momento del *release* no acumularon suficientes votos para ser validados o invalidados se publican bajo la categoría **"other"**. Este diseño de **doble crowdsourcing** —la multitud produce el dato y la misma multitud lo controla— es lo que distingue a Common Voice de VoxForge, que carecía de paso de validación.

### 4.3. Particionado sin fuga de hablantes

Los conjuntos train/dev/test se arman de modo que **un hablante aparezca en uno solo** de ellos. Esto garantiza que los locutores vistos en entrenamiento no reaparezcan en prueba, lo que sesgaría los resultados; es una **evaluación honesta de la generalización a hablantes nuevos**. Además se eliminan repeticiones de oraciones entre los tres conjuntos. El número de clips por partición se decide con un **análisis de potencia estadística**: dado el total de clips validados de un idioma, el conjunto de prueba tiene el tamaño necesario para alcanzar un **nivel de confianza del 99% con un margen de error del 1%** relativo al de entrenamiento, y lo mismo para el conjunto de desarrollo.

### 4.4. Formato de audio: MP3 a 48 kHz

Los clips se liberan como **mono-canal, 16 bits, archivos MPEG-3 (MP3) a 48 kHz de muestreo**. La decisión de usar MP3 —un formato **con pérdida**— en lugar de uno sin pérdida como WAV es deliberada y responde a la **naturaleza web** de la plataforma: MP3 es el formato de audio **más universalmente soportado** por navegadores y dispositivos, lo que lo hace el mecanismo más confiable de grabación y reproducción. Los autores argumentan que, en la práctica, **la calidad es adecuada para aplicaciones de habla**. Este detalle es exactamente el que la Clase 37 destaca al describir cómo se distribuyen los corpus de escala web: la compresión con pérdida es un compromiso consciente entre fidelidad y accesibilidad masiva.

### 4.5. Metadatos demográficos opcionales

Cada idioma se descarga como un directorio comprimido con **seis archivos TSV** (valores separados por tabuladores) y un subdirectorio `clips` con todo el audio. Los seis TSV comparten las mismas columnas:

`[client_id, path, sentence, up_votes, down_votes, age, gender, accent]`

Las tres primeras columnas identifican al hablante (**ID anonimizado**), la ubicación del archivo de audio y el texto leído. Las dos siguientes registran cómo los oyentes juzgaron el par ⟨audio, transcripción⟩. Y las **tres últimas** —**edad, género y acento**— son **datos demográficos autorreportados de forma opcional** por el hablante. Esta opcionalidad es central para la discusión de sesgo: son metadatos valiosísimos para auditar representación, pero como no son obligatorios, **su cobertura es parcial**.

### 4.6. Agregar un idioma nuevo

Escalar a un idioma nuevo requiere dos pasos, ambos comunitarios:

1. **Traducir la interfaz** de la web-app al idioma objetivo. Al momento de escribir el paper hay **610 cadenas de texto** en la interfaz, gestionadas por la plataforma de traducción **Pontoon**, donde la comunidad propone traducciones y los moderadores del idioma las aprueban o rechazan.
2. **Reunir oraciones para leer en voz alta.** Estos textos **no se traducen** —eso sería lento y no escalable— sino que se recolectan desde cero por idioma. Para idiomas con más de **500.000 artículos de Wikipedia**, las oraciones se extraen de Wikipedia con reglas provistas por la comunidad. Cualquier comunidad puede además aportar oraciones vía el **Sentence Collector**, con validaciones automáticas (longitud, alfabetos extranjeros, números) y aprobación de **dos de cada tres revisores**. Una vez traducida la interfaz y reunidas al menos **5.000 oraciones**, el idioma se habilita para grabar voces.

## 5. Estadísticas del corpus

La Tabla 1 del paper resume el estado de los datos (release del 12 de junio de 2019, con algunas cifras en cursiva aún no publicadas). Los idiomas listados van desde el **abjasio, árabe, vasco, bretón, catalán, chino (China y Taiwán), chuvasio, dhivehi, neerlandés, inglés, esperanto, estonio, francés, alemán, hakha chin, indonesio, interlingua, irlandés, italiano, japonés, cabilio, kiñaruanda, kirguís, letón, mongol, persa, portugués, ruso, sakha, esloveno, español, sueco, tamil, tártaro, turco, vótico** hasta el **galés**. Los totales agregados del cuadro:

- **58.250 voces** (hablantes) en total.
- **2.508 horas** de audio total.
- **2.019 horas validadas**.

El desequilibrio entre idiomas es enorme y explícito en los datos. El **inglés** domina con **39.577 voces** y del orden de **mil horas** validadas, mientras muchos idiomas tienen **menos de una hora** validada (marcada como "<1" en la tabla) y apenas un puñado de voces. El **alemán** aporta más de **65.000 clips** en los experimentos, frente a **menos de 1.000 clips para el esloveno**. Esta *cola larga* de idiomas con datos mínimos es a la vez la mayor virtud del corpus (cubre idiomas que nadie más cubre) y su limitación central (para muchos de ellos el dato es insuficiente para entrenar solo).

## 6. La discusión de sesgo y representación

Este es el eje que la Clase 37 pone en primer plano, y el paper lo aborda de forma tanto explícita como estructural.

**Desbalance entre idiomas.** Como muestran las estadísticas, la distribución de horas es profundamente desigual: el inglés y un grupo de idiomas europeos concentran la mayor parte del audio, mientras los idiomas minoritarios quedan con fracciones de hora. Esto reproduce, dentro del propio corpus abierto, el mismo sesgo de altos-vs-bajos recursos que el proyecto busca combatir. El paper es consciente de ello y lo compensa parcialmente con **transfer learning** (Sección 5 del paper): copiar capas de un modelo inglés bien entrenado permite que idiomas con muy pocos datos alcancen resultados razonables, un reconocimiento implícito de que **el dato por sí solo no basta** para los idiomas de la cola.

**Representación demográfica: edad, género y acento.** Al incluir metadatos de **edad, género y acento** en cada TSV, Common Voice **habilita la auditoría de representación** que la clase exige: se puede preguntar cuántas mujeres vs. hombres, qué rangos etarios y qué acentos están representados en cada idioma. Pero el diseño tiene una tensión inherente: como estos campos son **autorreportados y opcionales**, quedan frecuentemente vacíos, y la propia recolección crowdsourced tiende a **sobrerrepresentar el perfil típico del voluntario de Mozilla** —tecnófilo, con acceso a internet y a un dispositivo, y sesgado hacia ciertos géneros y grupos de edad. En otras palabras, la herramienta para medir el sesgo existe, pero el sesgo de participación sigue ahí: **quién decide grabar** determina qué acentos y qué voces terminan en el corpus.

**Sesgo de particionado como preocupación de calidad.** El diseño de splits sin fuga de hablantes (§4.3) es también una postura sobre sesgo evaluativo: al forzar que ningún hablante cruce de train a test, los autores evitan el sesgo optimista de "reconocer voces ya vistas", a costa de dejar algunos conjuntos de entrenamiento con **muy pocos hablantes**, lo que hace la tarea más difícil pero la métrica más honesta.

El mensaje transversal es que **la apertura y la escala no eliminan el sesgo por sí solas**: un corpus CC0 gigante puede seguir siendo demográficamente estrecho. Common Voice ofrece los metadatos para detectarlo, pero deja la corrección —conseguir hablantes de acentos y géneros subrepresentados— como un problema abierto de organización comunitaria.

## 7. Impacto

Common Voice se convirtió en **infraestructura fundacional del ASR multilingüe moderno**. Su combinación de escala, apertura CC0 y cobertura de idiomas de bajos recursos lo volvió una fuente estándar para entrenar y evaluar sistemas de habla. Dos de los sistemas de reconocimiento del habla más importantes de la generación siguiente lo usan directamente:

- **Whisper** (OpenAI, 2022), el modelo de ASR multilingüe robusto de referencia, incorpora datos y evaluación en el linaje de corpus abiertos multilingües del que Common Voice es pilar.
- **MMS (Massively Multilingual Speech)** (Meta, 2023), que empuja el reconocimiento y síntesis del habla a más de mil idiomas, se apoya explícitamente en la existencia de recursos abiertos multilingües como Common Voice para las lenguas que sí tienen dato transcrito.

Más allá de estos, Common Voice popularizó un **modelo de producción de datasets** —crowdsourcing con validación por pares, licencia de dominio público, crecimiento comunitario— que otros proyectos de datos abiertos han replicado. Cada nuevo *release* anual amplía idiomas y horas, de modo que el impacto no es el de un dataset congelado sino el de una **plataforma que sigue creciendo**.

## 8. Limitaciones

- **Calidad variable.** Al ser grabado por voluntarios con dispositivos y entornos heterogéneos, y comprimido en MP3, el audio tiene calidad desigual: ruido de fondo, micrófonos dispares, lecturas con errores de pronunciación. La validación por votos filtra lo peor, pero no homogeneiza la calidad.
- **Desbalance severo entre idiomas.** Como se detalló, un puñado de idiomas concentra casi todas las horas; para la mayoría de los idiomas de la cola el dato es insuficiente para entrenar un ASR competitivo sin transfer learning u otras muletas.
- **Sesgo de participación demográfica.** Los metadatos de edad/género/acento son opcionales y muchas veces ausentes, y la población de voluntarios no es representativa de los hablantes reales de cada idioma.
- **Habla leída, no espontánea.** El corpus se construye leyendo oraciones en pantalla, por lo que captura **habla leída** y no conversacional o espontánea; los modelos entrenados solo con Common Voice pueden generalizar peor a habla natural, disfluencias y diálogo real.
- **Formato con pérdida.** La elección de MP3 sobre WAV prioriza la compatibilidad web sobre la fidelidad acústica; para tareas sensibles a la señal fina, la compresión puede ser una restricción.

## 9. Conexión con la Clase 37 (Datasets y Herramientas para Audio)

La Clase 37 —segunda de cinco sobre audio— trata **de dónde salen los datos** con que se entrenan los sistemas de habla, y Common Voice es su caso de estudio canónico en tres frentes:

1. **Escala web.** La clase lo agrupa con GigaSpeech y Emilia como corpus de "cientos de miles de horas". Common Voice mostró que el crowdsourcing masivo puede producir miles de horas transcritas sin comprar dato, y su modelo de crecimiento comunitario es precisamente el que habilita esa escala.
2. **Sesgo y representación.** La pregunta guía de la clase —"¿el dataset cubre acentos, géneros e idiomas?"— es literalmente la que Common Voice hace operable: incluye columnas de **edad, género y acento**, expone su **desbalance entre idiomas** en la Tabla 1, y deja en evidencia que un corpus abierto y grande **no es automáticamente representativo**. Es el ejemplo perfecto para discutir por qué la apertura es necesaria pero no suficiente.
3. **Licencia y formato.** El corpus es **CC0 (dominio público)**, el grado máximo de apertura, y se distribuye en **MP3**, ilustrando el compromiso práctico entre fidelidad y compatibilidad que la clase señala al hablar de cómo se empaquetan los datasets de audio.

**Nota de relevancia para salud.** El sesgo de representación de Common Voice no es una curiosidad académica: es el mismo riesgo que amenaza a cualquier **ASR clínico**. Un sistema de dictado médico o de transcripción de la relación médico-paciente entrenado sobre datos que sobrerrepresentan ciertos acentos, géneros y grupos etarios **fallará justo con los pacientes más vulnerables** —hablantes de lenguas minoritarias, migrantes con acentos no cubiertos, adultos mayores, poblaciones rurales— que son quienes más dependen de que la tecnología no los deje afuera. La equidad en salud digital exige preguntar, antes de desplegar un ASR clínico, exactamente lo que la Clase 37 y Common Voice enseñan a preguntar: *¿este dataset se parece a la población real de pacientes que va a atender?* Un modelo que no cubre el habla de sus usuarios reales no es solo menos preciso: es una fuente activa de inequidad.
