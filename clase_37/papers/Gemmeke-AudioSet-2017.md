# Audio Set: An Ontology and Human-Labeled Dataset for Audio Events — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Audio Set: An Ontology and Human-Labeled Dataset for Audio Events*.
- **Autores:** Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal, Marvin Ritter. Todos en **Google, Inc.** (Mountain View, CA, y Nueva York).
- **Venue:** *IEEE International Conference on Acoustics, Speech and Signal Processing* (**ICASSP 2017**), Nueva Orleans.
- **Año:** 2017.
- **Recursos liberados:** la ontología como archivo JSON y el *Audio Set YouTube Corpus* como archivo CSV, ambos publicados en `g.co/audioset`.

El paper describe la creación de **Audio Set**, un recurso doble: (1) una **ontología jerárquica de 632 categorías de eventos de audio** construida a partir de la literatura y curación manual, y (2) un **dataset a gran escala** de segmentos de 10 segundos extraídos de videos de YouTube, etiquetados por anotadores humanos con una o más categorías de esa ontología. El resultado que se libera contiene **1 789 621 segmentos (4971 horas)**, con al menos 100 instancias para 485 de las 632 categorías. Cada segmento puede portar múltiples etiquetas (en promedio **2.7 etiquetas por segmento**), lo que hace de Audio Set un problema **multi-etiqueta** por diseño.

La motivación es explícita y central para entender el trabajo: el **reconocimiento de eventos de audio** —la capacidad humana de identificar y relacionar sonidos— es un problema "naciente" en percepción de máquinas, y carecía de un recurso comparable a **ImageNet**. Los autores enmarcan Audio Set como el intento de **cerrar la brecha de disponibilidad de datos** entre la investigación en imágenes y en audio, aspirando a proveer cobertura de sonidos del mundo real "a escala tipo ImageNet".

Para la **Clase 37 (Datasets y Herramientas para Audio)** este paper es doblemente relevante. Primero, porque Audio Set es el **benchmark de escala** que ancla el campo del *audio tagging* y la clasificación de eventos sonoros a gran escala. Segundo, y más sutil, porque su **modo de distribución** —links de YouTube en lugar de archivos de audio— es el ejemplo canónico del **eje de disponibilidad** que la clase discute: un dataset gigantesco pero frágil, sujeto a *link rot* (videos que se caen), que se contrasta con datasets como FSD50K que sí entregan el audio directamente.

## 2. Contexto: existía ImageNet para imágenes, pero nada equivalente para eventos de audio

El paralelo con visión no es decorativo: es el argumento estructural del paper. Los autores recuerdan que los "resultados asombrosos" en reconocimiento de imágenes (AlexNet, GoogLeNet, ResNet) descansan sobre **ImageNet**, que provee "más de un millón de imágenes etiquetadas con 1000 categorías de objetos". ImageNet parece haber sido "un factor mayor" impulsando esos avances — "y sin embargo nada de esta escala existe para fuentes de sonido". La tesis operativa es que **la escala del dataset, no solo la arquitectura, fue lo que desbloqueó visión**, y que replicar esa escala para audio podría producir un salto análogo.

El paper recorre los antecedentes, sistemáticamente **de dominio limitado o de escala pequeña**:

- **Perspectiva perceptual:** Ballas estudió la identificación de 41 sonidos breves; Gygi, Kidd y Watson usaron *multidimensional scaling* sobre 50 sonidos ambientales; Lemaitre y Heller propusieron una taxonomía objetos/acciones, mostrando que los oyentes prefieren un "rango medio" de abstracción.
- **Taxonomías de ingeniería:** Gaver diseñó efectos de sonido por factores perceptuales; Nakatani y Okuno idearon una ontología de sonido para análisis de escenas auditivas; Burger et al. desarrollaron 42 *noisemes* (por analogía con fonemas) para anotar 5.6 horas de audio web.
- **Datasets previos:** Salamon et al. liberaron 18.5 horas de sonidos urbanos de `freesound.org` con 10 categorías de una taxonomía urbana de 101; Säger et al. construyeron **AudioSentiBank**, 1267 horas con 1123 pares adjetivo/verbo-sustantivo, pero con etiquetas a nivel de clip completo (hasta 15 min) y sin garantía de que los pares correspondan (el célebre "talking bird" rara vez contiene un pájaro hablando).
- **Evaluaciones de detección:** **DCASE 2013** atrajo 7 sistemas para 16 eventos de "oficina", con F-measures **por debajo de 0.2**; DCASE 2016 anotó 90 minutos con 13 categorías.

El punto que los autores subrayan es que, **a diferencia de todo lo anterior, Audio Set considera *todos* los eventos sonoros** en lugar de un dominio acotado, apostando a que una tarea a gran escala —en categorías *y* en datos— habilitará técnicas de aprendizaje más potentes y un salto cualitativo en la calidad de los sistemas.

## 3. Contribución central

La contribución es doble y ambas partes son inseparables:

1. **La ontología de eventos de audio:** una jerarquía de **632 categorías** de sonido, diseñada para cubrir de forma exhaustiva las distinciones acústicas que hace un oyente "típico", estructurada para servir tanto al entrenamiento de clasificadores como al proceso de etiquetado humano.
2. **El dataset a escala:** **1 789 621 segmentos** de 10 segundos de YouTube, verificados por humanos, multi-etiqueta, publicados como un corpus con particiones de entrenamiento y evaluación balanceadas.

A esto se suma un **baseline** que da una idea del desempeño alcanzable y, crucialmente para la Clase 37, una **decisión de distribución**: dado que los clips provienen de YouTube y no pueden redistribuirse por copyright, lo que se libera son **identificadores de YouTube + timestamps + etiquetas** (más la ontología), no los archivos de audio. Esta decisión, racional desde lo legal, es la fuente del problema de disponibilidad que veremos en la sección 8.

## 4. Método

### 4.1. Diseño de la ontología: por qué jerárquica y no una lista plana

Los autores argumentan que una **lista plana** sería inadecuada, y que la jerarquía cumple funciones concretas:

- **En entrenamiento**, indica clases con relaciones **no exclusivas**: no queremos que el clasificador intente separar "sonidos de perro" de "ladrido", porque uno contiene al otro.
- **En reconocimiento**, permite *backing-off*: un sonido reconocido ambiguamente como "gruñido", "ladrido" y "aullido" puede replegarse a "sonidos de perro".
- **En etiquetado**, ayuda al anotador a encontrar rápido los términos que mejor describen un sonido, y a los diseñadores a añadir categorías sin solapamiento.

Los **principios de diseño** declarados son notables por su orientación perceptual:

- Las categorías deben ser un conjunto **comprensivo** que describa los eventos del mundo real.
- La categoría debe corresponder a **la idea que le viene inmediatamente a la mente** a un oyente que escucha el sonido.
- Las categorías deben ser **distinguibles por un oyente "típico"**: si dos sonidos no pueden distinguirse fácil o confiablemente, las categorías se fusionan. Distinciones que solo un experto haría (especies de aves, matices finos entre instrumentos) **no** se separan — una condición natural para que el conjunto no se vuelva inmanejable.
- Idealmente las categorías son distintas **por su sonido solo**, sin apoyarse en información visual o de contexto: un sonido es un "*thump*", no "un pie descalzo pisando un piso de madera".
- La jerarquía **no debe ser demasiado profunda**, y el número de hijos de un nodo rara vez debe superar 10, para permitir escaneo rápido.

### 4.2. Construcción de la ontología: siembra por texto web + ensamblaje manual

Para **no sesgar** el conjunto de categorías hacia la orientación de un investigador o la diversidad limitada de un dataset particular, los autores parten de un análisis **neutral y a escala web** de texto. Sembraron el léxico de eventos de audio con una forma modificada de los **"patrones de Hearst"** para identificar hipónimos de "sonido" — términos que aparecen frecuentemente en construcciones como "*...sonidos, tales como X e Y...*" o "*X, Y, y otros sonidos...*". Aplicando estas reglas sobre texto a escala web se obtiene un conjunto muy grande de términos, que luego se ordenan según cuán bien representan sonidos (combinando frecuencia de aparición con qué tan exclusivamente se identifican como hipónimos de "sonido"). Esto dio una **lista inicial de más de 3000 términos**, resueltos además contra los identificadores de máquina (**MIDs**) de Freebase / Knowledge Graph, que se usan como identificadores estables.

Partiendo del tope de la lista ordenada, los autores **ensamblaron manualmente** la jerarquía, deteniéndose cuando la lista empezó a entregar términos "oscuros o mal definidos" (ejemplos citados: "Wilhelm scream", "The Oak Ridge Boys", "Earcon", "Whump"). La estructura resultante **no es una jerarquía estricta**: hay nodos que aparecen en varios lugares — por ejemplo, "Hiss" figura bajo "Cat", "Steam" y "Onomatopoeia" — con un total de **33 categorías que aparecen más de una vez**.

El conjunto se **refinó** comparándolo con taxonomías previas; la esperanza era que la lista inicial las subsumiera, y aunque expuso numerosos vacíos, eventualmente casi toda clase de otros conjuntos quedó cubierta. Algunas clases externas se descartaron por ser **demasiado específicas o dependientes del contexto**: la taxonomía urbana de Salamon incluye "Car radio" (tiene sentido al etiquetar ambiente de calle, pero es demasiado contextual), y ejemplos como "Trumpet jazz solo" o "Walking on leaves" se colapsan en Audio Set a simplemente "Trumpet" y "Walk, footsteps".

La **lista final tiene 632 categorías** en una jerarquía con **profundidad máxima de 6 niveles**. Un ejemplo de nodo de nivel 6 (hay ocho) es la cadena:

$$\text{Sounds of things} \rightarrow \text{Vehicle} \rightarrow \text{Motor vehicle} \rightarrow \text{Emergency vehicle} \rightarrow \text{Siren} \rightarrow \text{Ambulance (siren)}$$

Los **50 nodos de nivel 1 y 2** organizan el mundo sonoro en siete grandes familias: **Human sounds** (voz, silbido, sonidos respiratorios, locomoción, digestivos, manos, latidos cardíacos, emisión otoacústica, acciones grupales), **Animal sounds** (animales domésticos, ganado y de trabajo, salvajes), **Natural sounds** (viento, tormenta, agua, fuego), **Music** (instrumento, género, conceptos, rol, estado de ánimo), **Sounds of things** (vehículos, motores, sonidos domésticos, campanas, alarmas, mecanismos, herramientas, explosiones, madera, vidrio, líquidos, impactos específicos), **Source-ambiguous sounds** (impactos genéricos, contacto de superficies, onomatopeya, silencio) y **Channel, environment and background** (ambiente acústico, ruido, reproducción de sonido).

### 4.3. Formato de la ontología

La ontología se libera como JSON, con estos campos por categoría:

- **ID:** el MID de Knowledge Graph que mejor corresponde al sonido o fuente (identificador primario).
- **Display name:** nombre breve de una o dos palabras, a veces con alternativas ("Burst, pop") o desambiguación entre paréntesis ("Fill (with liquid)").
- **Description:** descripción más larga, típicamente basada en Wikipedia o WordNet (con URIs de cita), adaptada para enfatizar el uso como evento de audio.
- **Examples:** al menos un ejemplo del sonido, provisto como **URL a un fragmento corto de un video público de YouTube**.
- **Children:** los MIDs de los hijos inmediatos (codifica la jerarquía).
- **Restrictions:** de las 632 categorías, **56 están en lista negra** (no se exponen a los anotadores por ser oscuras —"Alto saxophone"— o confusas —"Sounds of things"—) y **22 están marcadas como "Abstract"** (existen solo como nodos intermedios de estructura, p. ej. "Onomatopoeia", y no se usan como etiquetas).

### 4.4. Nominación de candidatos + verificación humana

El dataset se construye en dos etapas que la clase resume como **nominación + verificación**.

**Nominación de segmentos candidatos.** Para obtener suficientes ejemplos positivos con esfuerzo de anotación moderado, hay que enviar a etiquetar solo segmentos con buena probabilidad de contener el evento. Se usaron varios métodos: cerca de la mitad de los eventos correspondía a etiquetas ya predichas por un **sistema interno de anotación automática a nivel de video** (basado en metadatos, texto ancla de enlaces entrantes, comentarios y señales de interacción), restringido a videos con **al menos 1000 vistas**; y búsquedas por título y metadatos con un enfoque de *ranking* que pondera la coincidencia con el *display name* del evento y, con peso decreciente, con sus ancestros en la ontología (incluir "sound" como raíz mejoró la precisión). La selección temporal solía tomar el segmento a partir de los 30 segundos, para evitar intros o *branding* de canal.

**Verificación humana.** Cada segmento de 10 segundos se presenta a los anotadores **con video y audio** (no con el título ni metainformación del video). Se experimentó con segmentos más cortos y con presentación **solo-audio**, pero los anotadores encontraron esas condiciones mucho más difíciles, posiblemente por la naturaleza fina de la ontología. Para cada segmento, los anotadores califican de forma independiente la presencia de cada etiqueta como **"present", "not present" o "unsure"**. Cada segmento es evaluado por **tres anotadores** y se requiere **voto mayoritario**; por velocidad, la tercera calificación no se recolecta si las dos primeras coinciden en todas las etiquetas.

### 4.5. Construcción del dataset balanceado

El dataset liberado es un **subconjunto** del material: solo se representan calificaciones **"present"**. Como los segmentos pueden llevar múltiples eventos (incluidas "Speech" y "Music", siempre calificadas), ciertas etiquetas aparecen mucho más. Se buscó **maximizar el balance** añadiendo iterativamente segmentos para la clase menos representada, con preferencia por los de más etiquetas, y **evitando más de un segmento por video** para no correlacionar ejemplos. Se proveen subconjuntos **balanceados de entrenamiento y test** (de videos disjuntos), con al menos 50 positivos por clase donde fue posible; aun así, "Music" terminó con más de 5000.

## 5. Estadísticas, calidad y baseline

**Acuerdo entre anotadores.** Los anotadores fueron **unánimes en el 76.2%** de las votaciones; la calificación "unsure" fue rara (**0.5%** de las respuestas), de modo que los votos por mayoría 2:1 explican el **23.6%** de las decisiones. Categorías con mayor acuerdo: "Christmas music", "Accordion", "Babbling" (> 0.92). Con menor acuerdo: "Basketball bounce", "Boiling", "Bicycle" (< 0.17). Una verificación por muestreo reveló un número pequeño de errores, atribuidos a etiquetas confusas, error humano, y diferencias en la detección de eventos tenues o poco salientes. Como chequeo adicional se analizaron correlaciones entre etiquetas "present" y palabras en los metadatos del video, lo que expuso etiquetas comúnmente mal interpretadas que luego se removieron de la ontología.

**Eficacia de la nominación.** Los segmentos nominados por el sistema de anotación automática rindieron mejor (**49% calificados como present**) frente al enfoque por metadatos (**41%**); para clases no incluidas en la anotación automática, el **36%** de los segmentos por metadatos fue calificado present.

**Escala y distribución de etiquetas.** El dataset liberado incluye **1 789 621 segmentos (4971 horas)**, con al menos 100 instancias para **485 categorías**. El conjunto de entrenamiento no balanceado tiene **1 771 873 segmentos** y el de evaluación **17 748**. El promedio es de **2.7 etiquetas por segmento** y la distribución es fuertemente desigual: las clases más pobladas son **Music (1 006 882)**, **Speech (893 911)**, **Vehicle (80 422)**, **Musical Instrument (74 729)** y **Guitar (30 466)**; en la cola aparecen clases con poco más de 100 ejemplos ("Squeak" 127, "Shatter" 126, "Slap" 123, "Screech" 121). "Music" está presente en el **56%** de los segmentos. Es notable, para el interés clínico, que categorías como **"Heart sounds, heartbeat" (963)** aparezcan explícitamente en el conteo.

**Baseline.** Para dar una idea del desempeño alcanzable, se entrenó un sistema simple: usando la representación de la **capa de embedding** de un clasificador profundo entrenado sobre etiquetas genéricas de tópicos de video (referencia [22], Hershey et al., ICASSP 2017), se entrenó una red *fully-connected* poco profunda para las 485 categorías. Aplicando el clasificador a marcos de 1 segundo, promediando *scores* y rankeando, se obtuvo un **mean Average Precision balanceado de 0.314** y un **AUC promedio de 0.959** (equivalente a un *d-prime* de separación de clases de 2.452). La mejor categoría fue "Music" (AP/AUC/d-prime de 0.896 / 0.951 / 2.34, reflejando su alto prior); la peor fue "Rattle" (0.020 / 0.796 / 1.168). El *mAP* de 0.314 deja claro que la tarea, lejos de estar resuelta, deja mucho margen — que era justamente el punto.

## 6. El problema de distribución: links de YouTube, copyright y link rot

Aquí está el rasgo que la Clase 37 destaca. El *Audio Set YouTube Corpus* se distribuye como un **archivo CSV** que contiene **identificadores de YouTube, tiempo de inicio, tiempo de fin y una o más etiquetas** — no archivos de audio. Los ejemplos de la ontología también son **URLs a fragmentos de YouTube**. La razón es de copyright: Google no puede redistribuir el contenido de audio/video de terceros, así que en su lugar redistribuye **punteros** al contenido más las anotaciones humanas. Junto al corpus se liberaron además **features precomputadas** (embeddings derivados del clasificador de la referencia [22], el linaje directo de lo que la comunidad conoce como **VGGish**), lo que permite entrenar modelos sobre representaciones ya extraídas sin descargar cada video.

Esta arquitectura de distribución tiene una consecuencia estructural: **link rot**. Un puntero a un video de YouTube deja de resolver cuando el video se elimina, se hace privado, se bloquea por región o cae por reclamo de copyright. Con el tiempo, una fracción creciente de los ~1.79 millones de segmentos se vuelve **irrecuperable**, y no hay dos investigadores que necesariamente descarguen el *mismo* subconjunto disponible. El paper mismo no cuantifica el *link rot* (era un recurso recién liberado), pero la decisión de diseño que lo causa está explícitamente en el texto: audio no redistribuible, solo identificadores. Para un curso sobre **datasets y herramientas**, Audio Set es el caso de estudio perfecto de la tensión entre **escala** (solo YouTube ofrece millones de horas etiquetables) y **reproducibilidad** (solo el audio en mano garantiza que el dataset no se erosione).

## 7. Impacto

Audio Set se convirtió en el **benchmark de referencia** para clasificación de eventos de audio a gran escala y *audio tagging* multi-etiqueta, cumpliendo su ambición de ser "el ImageNet del audio". Su influencia se materializó en varias direcciones:

- **VGGish.** El baseline se apoya en el clasificador de Hershey et al. (referencia [22], mismo grupo y ICASSP), cuyas *embeddings* —conocidas como **VGGish**— se liberaron junto a Audio Set y se volvieron una representación estándar de audio *off-the-shelf*, análoga a usar features de una CNN preentrenada en ImageNet.
- **PANNs y arquitecturas posteriores.** Audio Set habilitó modelos entrenados directamente sobre la escala del corpus (*Pretrained Audio Neural Networks*, y luego transformers de audio), que reportan su *mAP* sobre la partición de evaluación de Audio Set — la métrica que este paper inauguró con su 0.314.
- **La ontología como vocabulario compartido.** Las 632 categorías y sus MIDs se adoptaron como vocabulario común para etiquetar sonido, más allá del dataset.

## 8. Limitaciones

- **Link rot y reproducibilidad.** Como se discutió, distribuir punteros a YouTube en vez de audio erosiona el dataset con el tiempo y hace que distintos equipos entrenen sobre subconjuntos distintos. Es la limitación práctica más citada del recurso.
- **Ruido de etiquetas.** Aunque el acuerdo unánime fue alto (76.2%), un 23.6% de las decisiones se resolvió por mayoría 2:1, y ciertas categorías tienen acuerdo muy bajo (< 0.17). Los propios autores reconocen errores residuales atribuidos a etiquetas confusas, error humano y eventos poco salientes; dada la escala y el alto acuerdo mayoritario, **no se tomaron más acciones correctivas**, de modo que el ruido de etiquetas es un residuo aceptado del diseño.
- **Sesgo de YouTube y de la nominación.** El corpus hereda los sesgos de YouTube (contenido popular, con ≥ 1000 vistas, cargado por ciertas poblaciones) y del sistema de nominación (que favorece lo que la anotación automática ya sabía etiquetar). Esto se refleja en el desbalance extremo: "Music" y "Speech" dominan, y la cola larga de eventos raros apenas alcanza 100 ejemplos.
- **Cobertura incompleta.** Solo **485 de 632** categorías llegan a 100 instancias; el resto está excluido (lista negra/abstract) o es difícil de poblar — trabajo en progreso, según los autores.
- **Verificación con video.** El etiquetado se hizo **con video visible** (solo-audio resultaba demasiado difícil): las etiquetas verifican que "el sonido está presente en un clip donde el anotador *también vio* la fuente", lo que puede no coincidir con lo que un sistema puramente acústico debería inferir del audio aislado.

## 9. Conexión con la Clase 37 (Datasets y Herramientas para Audio)

Audio Set encarna dos de los ejes que organizan la clase.

**El eje de la escala web.** La clase presenta Audio Set y **FSD50K** como los dos grandes benchmarks de eventos de audio. Audio Set representa la vía de **escala máxima vía web**: aprovechar YouTube para reunir ~1.79 millones de clips etiquetados, algo imposible con grabación curada. Esa escala es lo que hizo viable entrenar modelos de audio "estilo ImageNet".

**El eje de la disponibilidad.** Aquí la clase hace la advertencia explícita: a veces un dataset "solo da un link de YouTube que se cae (Audio Set)", frente a datasets que entregan el audio directo. El contraste con **FSD50K** es didáctico y directo:

| Eje | Audio Set (Gemmeke et al., 2017) | FSD50K |
|---|---|---|
| Fuente del audio | Videos de YouTube (terceros) | Freesound (audio con licencias abiertas) |
| Qué se distribuye | Identificadores de YouTube + timestamps + etiquetas + features | **Los archivos de audio** directamente |
| Escala | ~1.79 M clips de 10 s (4971 h) | Decenas de miles de clips |
| Ontología | 632 categorías propias (jerárquicas) | Subconjunto de la ontología de **Audio Set** |
| Riesgo de link rot | Alto (los videos se caen) | Nulo (audio en mano) |
| Reproducibilidad | Erosiona con el tiempo | Estable |

La lección de diseño es que **escala y reproducibilidad están en tensión**: la elección de Audio Set —punteros por copyright— compró la mayor escala del campo al costo de la fragilidad. FSD50K, notablemente, **reutiliza la ontología de Audio Set** (validando su contribución taxonómica) pero resuelve la disponibilidad al construirse sobre audio de licencia abierta. Entender por qué existen ambos, y cuándo conviene cada uno, es lo que la Clase 37 quiere que el estudiante sepa evaluar al elegir datos y herramientas para un proyecto de audio.

**Enlaces internos:**

- Fundamento transversal: [/fundamentos/clasificacion-de-audio](/fundamentos/clasificacion-de-audio) — eventos de audio, *audio tagging* multi-etiqueta, métricas (mAP, AUC).
- Clase: [/clases/clase-37](/clases/clase-37) — Datasets y Herramientas para Audio.
- Paper compañero: Hershey et al. (2017), *CNN architectures for large-scale audio classification* — el origen de las *embeddings* VGGish usadas como baseline.

## Nota final: relevancia para salud

La ontología de Audio Set ofrece un **marco reutilizable para clasificar sonidos clínicos**: su rama de *Human sounds* ya incluye nodos directamente médicos —"Respiratory sounds", "Heart sounds, heartbeat", "Digestive"—, y el conteo del dataset lista "Heart sounds, heartbeat" con 963 ejemplos, lo que sugiere que la auscultación (crepitaciones, sibilancias, soplos, ruidos intestinales) puede formalizarse como **detección de eventos de audio multi-etiqueta** sobre una taxonomía estructurada, en lugar de resolverse ad hoc por patología. Al mismo tiempo, Audio Set es una **advertencia de reproducibilidad**: un corpus distribuido como punteros frágiles se erosiona y deja de ser el mismo dataset con el tiempo; en un dominio donde la evidencia debe ser auditable, conviene priorizar la **entrega del audio de facto** (el modelo FSD50K) sobre la máxima escala obtenida a costa de la disponibilidad — máxime cuando en salud el copyright se suma a restricciones de privacidad y consentimiento que agudizan esa tensión.
