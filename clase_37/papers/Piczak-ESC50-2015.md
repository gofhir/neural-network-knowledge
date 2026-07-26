# ESC: Dataset for Environmental Sound Classification — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *ESC: Dataset for Environmental Sound Classification*.
- **Autor:** Karol J. Piczak, Institute of Electronic Systems, Warsaw University of Technology (Varsovia, Polonia).
- **Venue:** *Proceedings of the 23rd ACM International Conference on Multimedia (MM '15)*, Brisbane, Australia, 26–30 de octubre de 2015.
- **DOI:** [10.1145/2733373.2806390](http://dx.doi.org/10.1145/2733373.2806390).
- **Recursos:** dataset publicado bajo licencia Creative Commons no comercial en Harvard Dataverse ([10.7910/DVN/YDEPUT](http://dx.doi.org/10.7910/DVN/YDEPUT)); notebook IPython (Jupyter) con el análisis completo y el código de replicación en [github.com/karoldvl/paper-2015-esc-dataset](https://github.com/karoldvl/paper-2015-esc-dataset).

Este trabajo introduce la colección **ESC (Environmental Sound Classification)**, pensada para paliar un problema concreto del área: la escasez de conjuntos de datos abiertos, balanceados y comparables para clasificar **sonidos ambientales** —esa categoría heterogénea de eventos de audio cotidiano que no son ni habla ni música—. El aporte se articula en tres piezas: **ESC-50** (2.000 clips etiquetados de 5 segundos en 50 clases), **ESC-10** (un subconjunto de 10 clases más sencillo, pensado como prueba de concepto) y **ESC-US** (250.000 clips **sin etiquetar** para preentrenamiento no supervisado). Todos los clips se construyeron a partir de grabaciones disponibles públicamente en el proyecto **Freesound**.

Más allá de liberar los datos, el paper hace algo metodológicamente valioso: mide el **techo humano** de desempeño mediante crowdsourcing (≈81,3 % de exactitud en ESC-50 y ≈95,7 % en ESC-10) y lo contrasta con **clasificadores base** clásicos —features MFCC y tasa de cruces por cero alimentando k-NN, random forest y SVM— que quedan muy por debajo (44,3 % en ESC-50 con el mejor de ellos). Esa **brecha humano-máquina** es el argumento central del trabajo: hay mucho margen de mejora, y el autor apunta explícitamente a que las redes profundas —en particular las convolucionales, exploradas en un trabajo hermano suyo— son el camino para cerrarla.

Para la **Clase 37 (Datasets y Herramientas para Audio)** este paper es doblemente pertinente. Primero, porque ESC-50 se ha convertido en el dataset **didáctico** por excelencia para clasificación de sonido ambiental: pequeño, balanceado, con folds predefinidos y trivial de descargar. Segundo, porque ilustra una práctica de buen diseño de benchmark que trasciende el audio: acompañar un dataset con una **estimación del desempeño humano** que sirva de referencia superior contra la cual medir el progreso de las máquinas.

## 2. Contexto: la falta de un benchmark abierto y balanceado

Piczak abre situando el problema en perspectiva. Los años previos habían traído avances espectaculares en percepción por máquina, sobre todo en **visión**, donde el auge del deep learning ya permitía a veces superar capacidades humanas. En audio, sin embargo, el esfuerzo se había concentrado casi exclusivamente en **habla y música**. El análisis de sonidos ambientales quedó rezagado en la adopción de esas mejoras, pese a numerosas aplicaciones posibles: sistemas de vigilancia acústica, audífonos, monitoreo inteligente de habitaciones y generación de resúmenes de video.

El diagnóstico del autor es que uno de los impedimentos objetivos del campo es su **fragmentación**: la mayoría de los estudios se habían realizado sobre conjuntos de datos muy específicos, pequeños o (semi)propietarios, y muchas veces sin acceso al código original, lo que hacía la **reproducibilidad** mucho más difícil de lo razonable. Piczak lo contrasta de forma directa con la visión por computador, donde corpora como **MNIST** y **CIFAR** funcionaban como estándar de facto para comparaciones base. En audio ambiental no existía un equivalente ampliamente adoptado. Solo iniciativas recientes como el proyecto **UrbanSound** (Salamon, Jacoby y Bello, 2014), centrado en entornos urbanos, empezaban a cambiar el panorama —pero, en palabras del autor, "la situación seguía siendo más bien desalentadora".

De ahí que el objetivo declarado del paper sea **facilitar la investigación abierta** en clasificación de sonido ambiental mediante cuatro contribuciones concretas: aportar un dataset público de grabaciones ambientales; presentar estimaciones de la exactitud humana sobre él; comparar esas cifras con el desempeño base de los clasificadores de machine learning más comunes; y proveer un notebook Jupyter con un análisis más detallado y el código para replicar los resultados.

## 3. Contribución central: ESC-50, ESC-10 y ESC-US

La colección se compone de tres partes complementarias, todas construidas a partir de grabaciones de Freesound y en un formato de clip corto unificado (5 segundos, 44,1 kHz, mono, Ogg Vorbis a 192 kbit/s).

### 3.1. ESC-50 — el conjunto etiquetado principal

**ESC-50** contiene **2.000 grabaciones ambientales etiquetadas**, perfectamente balanceadas entre **50 clases** (exactamente **40 clips por clase**). Por conveniencia, las clases se agrupan en **5 categorías** mayores (10 clases cada una), definidas de manera "laxa":

- **sonidos de animales** (p. ej. perro, gallo, cerdo, vaca, rana, gato, insectos, oveja, cuervo, gallina);
- **paisajes sonoros naturales y sonidos de agua** (lluvia, olas de mar, fuego crepitante, grillos, aves, gotas de agua, viento, agua vertiéndose, tormenta, cadena de baño);
- **sonidos humanos no vocales** (bebé llorando, estornudo, aplausos, respiración, tos, pasos, risa, cepillado de dientes, ronquido, sorbos al beber);
- **sonidos interiores / domésticos** (golpeteo en puerta, clic de ratón, tecleo, crujidos de madera, apertura de latas, lavadora, aspiradora, reloj despertador, tictac de reloj, vidrio rompiéndose);
- **ruidos exteriores / urbanos** (helicóptero, motosierra, sirena, bocina, motor, tren, campanas de iglesia, avión, fuegos artificiales, serrucho manual).

El proceso de extracción buscó mantener el evento sonoro **expuesto en primer plano**, con ruido de fondo limitado cuando fuera posible; aun así, el autor advierte que las grabaciones de campo distan de ser estériles y que algunos clips conservan solapamiento auditivo de fondo. El dataset ofrece deliberadamente una mezcla de dificultades: fuentes muy comunes (risa, maullido, ladrido), otras muy distintivas (vidrio rompiéndose, cepillado de dientes) y algunas con diferencias sutiles y potencialmente confundibles (ruido de helicóptero frente al de avión).

La **limitación reconocida** de ESC-50 es el número reducido de clips por clase. El propio autor la atribuye al alto costo de la anotación y extracción manual y a la decisión de mantener un balance estricto entre clases pese a la disponibilidad desigual de grabaciones para eventos más exóticos.

### 3.2. ESC-10 — el subconjunto de prueba de concepto

**ESC-10** es una selección de **10 clases** tomadas del conjunto mayor, concebida inicialmente como dataset de prueba de concepto y como un problema **más fácil** para comenzar. Sus 10 clases representan tres grupos generales de sonido:

- **transitorios / percusivos**, a veces con patrones temporales muy significativos (estornudo, ladrido de perro, tictac de reloj);
- **eventos con fuerte contenido armónico** (bebé llorando, canto de gallo);
- **ruido / paisajes sonoros más o menos estructurados** (lluvia, olas de mar, fuego crepitante, helicóptero, motosierra).

Al restringir el problema a un conjunto pequeño de clases muy distinguibles y con ambigüedad limitada, ESC-10 fija un **listón muy alto** para la exactitud esperada de un sistema automático: como clasificarlas es trivial para un humano, un modelo debería aspirar a un desempeño cercano a lo perfecto. El autor señala que este subconjunto plantea un problema cualitativamente distinto al de ESC-50 y podría favorecer clases de modelos diferentes.

### 3.3. ESC-US — el conjunto no etiquetado para preentrenamiento

El número limitado de instancias etiquetadas hace a ESC-50/ESC-10 inadecuados por sí solos para enfoques que **aprenden representaciones a partir de los datos**. Para mitigarlo, Piczak provee **ESC-US**, un conjunto adicional de **250.000 grabaciones** extraídas de archivos de Freesound etiquetados como *"field recording"*, en el mismo formato de clip corto de 5 segundos. Es el material pensado para procedimientos de **preentrenamiento no supervisado** y modelos generativos.

A diferencia de la parte etiquetada, ESC-US se presenta **sin anotación manual verificada**: no fue revisado individualmente por el autor y se apoya únicamente en el control de calidad de Freesound mediante moderación colaborativa. Sí incluye los metadatos (tags y descripciones) que los autores originales enviaron con cada grabación. Por eso el autor sugiere que, además de clustering y aprendizaje de variedades (*manifold learning*), ESC-US podría usarse en regímenes de **aprendizaje débilmente supervisado**, donde las etiquetas están parcialmente ausentes o no son suficientemente específicas.

## 4. Método: recolección desde Freesound y folds predefinidos

Las tres partes se construyeron a partir de **Freesound** (Font, Roma y Serra, 2013), una base colaborativa de grabaciones de campo. Para la parte etiquetada, el procedimiento fue el siguiente:

1. **Selección de clases.** Las clases se eligieron de forma arbitraria pero con el objetivo de mantener **balance entre los grandes tipos de eventos sonoros**, considerando a la vez las limitaciones de cantidad y diversidad de las grabaciones fuente y la utilidad y distinción percibidas de cada clase.
2. **Consulta y anotación.** Se consultó la base de Freesound con términos comunes relacionados con las clases; el autor evaluó y verificó individualmente los resultados de búsqueda, anotando los fragmentos que contenían eventos de la clase correspondiente.
3. **Extracción.** A partir de esas anotaciones se extrajeron grabaciones de **5 segundos** de los eventos de audio; los eventos más cortos se rellenaron con silencio según fuera necesario.
4. **Normalización de formato.** Las muestras se reconvirtieron a un formato unificado: **44,1 kHz, un solo canal, compresión Ogg Vorbis a 192 kbit/s**.
5. **Partición en folds.** Los conjuntos etiquetados se organizaron en **5 folds de validación cruzada de tamaño uniforme**, con una regla crucial: los clips provenientes de un **mismo archivo fuente** quedan siempre en un **único fold**.

Esa última decisión es la sofisticación metodológica que hace de ESC un benchmark honesto. Si dos clips recortados de la misma grabación original cayeran en folds distintos —uno en entrenamiento y otro en prueba—, el modelo podría reconocer características idiosincrásicas de esa grabación concreta (fondo, micrófono, sala) en lugar de la clase de sonido, inflando artificialmente la exactitud por fuga de información. Al confinar cada archivo fuente a un solo fold, los folds predefinidos garantizan que la evaluación mida **generalización a fuentes nuevas**, y —al venir fijados con el dataset— aseguran que distintos trabajos reporten cifras **comparables** entre sí. Este esquema de "artist/source filtering" es hoy una buena práctica estándar en audio.

## 5. Benchmark humano y baselines clásicos

### 5.1. Exactitud humana como techo de referencia

Para estimar el desempeño humano, el autor recurrió a la plataforma de crowdsourcing **CrowdFlower**. A los participantes se les presentaban grabaciones y debían elegir la etiqueta correcta de una lista de 10 o de 50 categorías, según el dataset; se les pagaba una tarifa fija por unidad de trabajo (clasificar 10 grabaciones). El control de calidad se apoyó en los procedimientos internos de CrowdFlower —preselección de participantes y monitoreo continuo mediante preguntas de prueba con respuesta esperada insertadas al azar—, y el autor descartó posteriormente un pequeño número de entradas atípicas.

En total se recolectaron **alrededor de 4.000 juicios por dataset**: en promedio una docena de clasificaciones humanas por clip en ESC-10 y **dos** por clip en ESC-50. El autor es explícito en que un montaje así no permite una interpretación estadística formal, pero sí ofrece una **estimación aproximada** de las capacidades humanas.

Los resultados: exactitud media de **95,7 % en ESC-10** y **81,3 % en ESC-50**. El recall por clase varió enormemente entre tipos de evento: desde **34,1 % para el ruido de lavadora** hasta casi **100 % para bebés llorando y ladridos de perro**. El autor agrupa los eventos en tres niveles de dificultad:

- **fáciles:** la mayoría de sonidos humanos, animales y fuentes muy distintivas (sirena, gotas de agua, vidrio rompiéndose);
- **medios:** todo lo intermedio;
- **difíciles:** sobre todo paisajes sonoros y diversos ruidos mecánicos.

Una observación fina: al aumentar el número de categorías, se vuelve más difícil para participantes no entrenados **abarcar mentalmente** todas las posibilidades y distinciones semánticas. Con 50 clases todavía era posible presentarlas en una vista coherente (dividida en 5 grupos para orientarse más rápido) sin recurrir a taxonomías anidadas, pero —dice el autor— se estaba "al borde de lo verificable" en un experimento de este tipo. Por eso proyecta que **oyentes entrenados y atentos** probablemente puntuarían de forma impecable en ESC-10 y alcanzarían niveles del orden del **90 %** en ESC-50, con algún margen de error en los ruidos mecánicos y paisajes sonoros más ambiguos.

### 5.2. Baselines de features clásicas + clasificadores

El segundo objetivo fue verificar qué se logra con enfoques base, no construir el sistema más robusto posible sino explorar las dificultades del dataset. De cada clip se extrajeron dos tipos de **features**:

- **tasa de cruces por cero** (*zero-crossing rate*), un descriptor muy simple pero útil;
- **coeficientes cepstrales en escala mel (MFCC)**, ubicuos en procesamiento de habla y análisis de contenido armónico.

Los MFCC se calcularon con la librería **librosa (v0.3.1)** con ajustes por defecto, resultando en tramas (*frames*) de **11,6 ms**. Descartando el coeficiente 0, se tomaron los **primeros 12 MFCC** y la tasa de cruces por cero, y cada clip se resumió con la **media y la desviación estándar** de esas features a lo largo de las tramas. El vector de características así formado alimentó tres clasificadores: **k-vecinos más cercanos (k-NN)**, **random forest** (ensamble) y **máquina de soporte vectorial (SVM)** con kernel lineal. El aprendizaje se realizó sobre ambos datasets con el régimen de **validación cruzada de 5 folds** predefinidos.

Formalmente, un MFCC resume la envolvente espectral de corto plazo: se agrupa el espectro de potencia en un banco de filtros triangulares espaciados en la **escala mel** $m = 2595 \, \log_{10}(1 + f/700)$, se toma el logaritmo de la energía de cada banda y se aplica una transformada de coseno discreta; los primeros coeficientes de esa DCT son los MFCC. La escala mel imita la resolución no lineal del oído humano en frecuencia, razón por la cual estas features siguen siendo un punto de partida razonable para audio.

Los resultados quedan **muy por debajo del humano**:

| Modelo | ESC-10 | ESC-50 |
|---|---|---|
| k-NN | 66,7 % | 32,2 % |
| SVM (kernel lineal) | 67,5 % | 39,6 % |
| Random forest | **72,7 %** | **44,3 %** |
| **Humano** | **95,7 %** | **81,3 %** |

En ESC-10 el random forest fue el mejor (72,7 %), con dispersión notable entre folds atribuible a su pequeño tamaño absoluto. En ESC-50 hubo menos variabilidad entre folds pero una superioridad más marcada del random forest (44,3 %) sobre SVM (39,6 %) y k-NN (32,2 %). Una tendencia que resalta el autor es la **caída pronunciada del k-NN**, el modelo más simple, al pasar a ESC-50: sugiere que las dependencias entre features son más intrincadas en el dataset grande y se capturan mejor con modelos más complejos.

El análisis por clase (figura 2 del paper) revela un patrón interesante y algo paradójico: muchas grabaciones de paisajes sonoros y ruido de fondo son **ambiguas para los oyentes humanos** pero, casualmente, puntúan relativamente **alto** con los sistemas automáticos. Y el SVM superó al random forest en sonidos de animales, lo que —aunque podría ser un artefacto de los datos— insinúa que **modelos especializados por grupos** de sonido (una clasificación jerárquica multietapa) podrían ser una vía viable.

## 6. La brecha humano-máquina como motivación

El corazón argumentativo del paper es el **contraste de las dos tablas**. En ESC-50, el humano acierta 81,3 % y el mejor baseline apenas 44,3 %: una diferencia de casi **37 puntos porcentuales**. En ESC-10 la brecha es de unos 23 puntos (95,7 % frente a 72,7 %). Piczak establece primero la cifra objetivo —lo que un sistema con capacidad casi humana debería alcanzar— y luego muestra lo lejos que están los enfoques rudimentarios de ese objetivo.

Esa distancia es exactamente el valor del dataset: define un **problema abierto con techo conocido**. El autor deja claro que los métodos base presentados son deliberadamente simples y remite a un **trabajo más reciente suyo** —clasificación de sonido ambiental con **redes neuronales convolucionales** (Piczak, MLSP 2015)— como la evaluación de enfoques más robustos. Es decir, ESC-50 no se libera como un fin en sí mismo sino como el **terreno de juego** sobre el cual demostrar que el deep learning puede cerrar la brecha con el desempeño humano, del mismo modo que las CNN lo habían hecho en visión. El techo humano cumple aquí el rol que en visión cumplían las tasas de error humano en ImageNet: una referencia tangible que convierte "mejorar la exactitud" en "acercarse a —y eventualmente superar— al oyente humano".

## 7. Impacto

ESC-50 y ESC-10 se convirtieron rápidamente en el **benchmark estándar** para clasificación de sonido ambiental, cumpliendo con creces el objetivo declarado del autor de dar al campo un equivalente abierto a MNIST/CIFAR. Sus virtudes —tamaño manejable, balance perfecto entre clases, folds predefinidos con filtrado por fuente, licencia abierta y distribución trivial— lo hicieron ideal tanto para investigación como para docencia. En los años siguientes, prácticamente todo trabajo sobre *environmental sound classification* reportó resultados en ESC-50, lo que permitió por fin las comparaciones directas que Piczak echaba en falta.

La propia predicción del autor se cumplió: las redes profundas cerraron la brecha. Los sistemas basados en CNN sobre espectrogramas, y más tarde los modelos preentrenados a gran escala, terminaron **superando el techo humano** de 81,3 % en ESC-50, alcanzando exactitudes por encima del 90 % y luego del 95 %. ESC-50 fue el instrumento que hizo ese progreso medible y verificable. El conjunto ESC-US, por su parte, anticipó la lógica del **preentrenamiento no supervisado / auto-supervisado** en audio, que años después se volvería dominante.

## 8. Limitaciones

El propio Piczak es transparente sobre los límites del trabajo:

- **Pocos clips por clase.** 40 grabaciones por clase en ESC-50 es escaso para métodos que aprenden representaciones ricas; es la razón de ser de ESC-US, pero limita lo que se puede hacer solo con la parte etiquetada.
- **Selección de clases subjetiva.** Las clases se eligieron de forma arbitraria según utilidad y distinción percibidas y la disponibilidad de fuentes, sin criterio taxonómico formal.
- **Solapamiento de fondo.** Las grabaciones de campo no son estériles; algunos clips conservan ruido o eventos secundarios de fondo pese al esfuerzo por dejar el evento en primer plano.
- **Estimación humana informal.** El montaje de crowdsourcing —con solo ~2 juicios por clip en ESC-50 y participantes no entrenados— no admite interpretación estadística formal; es una estimación aproximada, no una medición rigurosa del límite humano.
- **Baselines deliberadamente débiles.** Los clasificadores base no buscan ser competitivos; la evaluación de métodos robustos (CNN) se delega a otro trabajo.
- **ESC-US sin verificar.** El conjunto no etiquetado se apoya solo en la moderación colaborativa de Freesound, sin revisión individual del autor.

## 9. Conexión con la Clase 37 (Datasets y Herramientas para Audio)

La Clase 37, segunda del bloque de audio, cita ESC-50 como **dataset didáctico** de referencia: "2.000 clips de 5 s, 50 clases de sonidos ambientales". Este paper es la fuente primaria de esa descripción y aporta el contexto que la vuelve pedagógicamente útil. Tres ideas conviene que el estudiante internalice:

1. **Anatomía de un buen benchmark.** ESC-50 ejemplifica qué hace confiable a un dataset: balance estricto entre clases, formato unificado, **folds predefinidos con filtrado por fuente** (para evitar fuga de información) y distribución abierta. Estas decisiones son transferibles a cualquier dominio, no solo al audio.
2. **El techo humano como referencia.** Medir la exactitud humana (≈81 % en ESC-50, ≈95 % en ESC-10) y publicarla junto al dataset convierte una métrica abstracta en una meta concreta. Es el mismo principio que guio el progreso en visión y una práctica que todo diseñador de sistemas debería adoptar.
3. **La brecha que motiva el deep learning.** El salto entre los baselines clásicos (MFCC + k-NN/RF/SVM, ~44 % en ESC-50) y el humano (~81 %) es la justificación empírica del uso de redes profundas. ESC-50 fue el terreno donde esa promesa se verificó y, con el tiempo, se superó al oyente humano.

**Nota final — relevancia para salud.** El techo humano de referencia que ESC-50 populariza tiene un valor directo en aplicaciones clínicas de audio. En el **monitoreo ambiental de pacientes** —por ejemplo, detectar tos, estornudos, caídas, ronquidos, respiración anómala o alarmas dentro de una habitación mediante clasificación de sonido— importa saber no solo qué exactitud logra un modelo, sino **cuán bien lo haría un clínico atento** escuchando lo mismo. Publicar el desempeño humano junto al de la máquina permite calibrar expectativas, definir umbrales de alerta clínicamente aceptables y decidir cuándo un sistema automático es lo bastante confiable para asistir (o descargar) al personal de salud; la lección metodológica de Piczak —un benchmark abierto, balanceado y con techo humano explícito— es tan aplicable a los sonidos de una sala de hospital como a los ladridos y sirenas de ESC-50.

---

**Enlaces internos:**

- Clase: [/clases/clase-37](/clases/clase-37) — Datasets y Herramientas para Audio.
- Fundamento transversal: [/fundamentos/clasificacion-de-audio](/fundamentos/clasificacion-de-audio) — features espectrales, MFCC, benchmarks de sonido ambiental.
