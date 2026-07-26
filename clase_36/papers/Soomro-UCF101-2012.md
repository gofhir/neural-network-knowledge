# UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild*.
- **Autores:** Khurram Soomro, Amir Roshan Zamir y Mubarak Shah. Los tres del **Center for Research in Computer Vision (CRCV)**, University of Central Florida (UCF), Orlando, Florida.
- **Formato:** reporte técnico **CRCV-TR-12-01**, noviembre de 2012. **Preprint:** arXiv:1212.0402v1 (3 dic 2012), [arxiv.org/abs/1212.0402](https://arxiv.org/abs/1212.0402).
- **Sitio del dataset:** [crcv.ucf.edu/data/UCF101.php](http://crcv.ucf.edu/data/UCF101.php).
- **Palabras clave declaradas:** Action Dataset, UCF101, UCF50, Action Recognition.
- **Linaje:** cuarto y mayor eslabón de la familia de datasets de acción compilados por UCF —**UCF Sports → UCF11 → UCF50 → UCF101**—, donde cada uno incluye a su precursor.

Este trabajo no propone un modelo ni una arquitectura: propone un **dataset**, y con él redefine el listón de dificultad del reconocimiento de acciones (*action recognition*) durante la primera mitad de la década de 2010. UCF101 reúne **101 clases de acciones humanas**, **13 320 clips** y **27 horas de video** descargados de YouTube. Su rasgo distintivo es que se trata de videos **subidos por usuarios reales** ("in the wild"), con movimiento de cámara, fondos abarrotados (*cluttered*), oclusión parcial, iluminación variable y baja calidad de cuadro. En el momento de su publicación era, según los autores, **el dataset de acciones más grande y más desafiante** que existía, tanto por su número de clases como por la naturaleza no controlada de sus clips.

Además del corpus, el paper aporta un **baseline** de reconocimiento con el método estándar de la época —*bag of words* sobre descriptores espacio-temporales— que alcanza un **44,5 % de exactitud (accuracy) global**, y recomienda un **protocolo de evaluación** (validación cruzada de 25 folds, *leave-one-group-out*) para que los resultados reportados por distintos grupos sean comparables entre sí.

Para la **Clase 36 (Introduction to Video Analysis)** este paper importa porque es el **benchmark canónico** sobre el que se midieron casi todos los enfoques de reconocimiento de acciones que la clase discute —desde *bag of words* clásico hasta las CNN 2D, las arquitecturas 2D CNN + RNN y, después, las CNN 3D y los *two-stream networks*—. El **laboratorio de la clase usa UCF11**, el abuelo directo de UCF101 en el mismo linaje UCF, de modo que entender UCF101 es entender el estándar hacia el que ese linaje evolucionó.

## 2. Contexto: por qué el campo necesitaba un dataset "in the wild"

Para dimensionar la contribución hay que mirar el estado de los datasets de acción hacia 2012. El paper identifica **dos deficiencias sistemáticas** en los benchmarks existentes.

**Primera deficiencia: muy pocas clases.** La riqueza de acciones que los humanos realizan en la realidad es enorme, pero los datasets de referencia cubrían apenas un puñado de categorías. Los autores lo cuantifican con crudeza: **KTH** incluía 6 clases, **Weizmann** 9, **UCF Sports** 9 e **IXMAS** 11. Varios trabajos habían mostrado que el **número de clases juega un rol crucial** al evaluar un método de reconocimiento —un clasificador que separa 6 acciones no dice casi nada sobre cómo escalará a decenas—. Incluso el dataset más grande disponible no superaba las 51 acciones (**HMDB51** con 51 acciones y 6766 clips; **UCF50** con 50 acciones y 6681 clips).

**Segunda deficiencia: entornos irrealmente controlados.** Los benchmarks clásicos se grababan en condiciones de laboratorio que no se parecen al video que una aplicación real encontraría:

- **KTH, Weizmann e IXMAS** eran *actor staged*: actores ejecutando la acción frente a una cámara fija, con fondo estático y sin movimiento de cámara. Son datasets limpios, casi de "figura sobre fondo".
- **HOHA (Hollywood Human Actions)** y **UCF Sports** provenían de clips de cine y televisión, capturados por equipos de filmación profesionales, con encuadre y estabilización cuidados.

Ninguna de esas condiciones representa el video que un sistema debe procesar en producción, donde la cámara tiembla, el fondo está saturado de objetos y personas, y la iluminación es la que haya. El campo ya había empezado a atacar la **segunda** deficiencia usando **video web**: datasets como UCF11 (Liu, Luo y Shah, 2009), UCF50 y HMDB51 incorporaban clips subidos por usuarios o extraídos de películas. Pero la **primera** deficiencia —el techo bajo de clases— seguía sin resolverse.

UCF101 ataca **ambas a la vez**: casi **duplica** el número de clases del dataset más grande de la época (de ~50 a 101) y **duplica** el número de clips (de ~6700 a 13 320), manteniéndose enteramente en el terreno del video no controlado de YouTube. Esa combinación —muchas clases *y* condiciones realistas— es lo que lo volvió el benchmark de referencia.

### 2.1. El eje "controlado" vs. "in the wild"

Vale la pena hacer explícito por qué la variabilidad de un video de YouTube lo hace más difícil —y más útil— que un video de laboratorio. En KTH o Weizmann, la señal de movimiento del actor es casi la única fuente de variación: el fondo es constante, la cámara no se mueve y la iluminación está fijada. Un descriptor de movimiento razonablemente bueno resuelve la tarea. En un video "in the wild" conviven varias fuentes de variabilidad que **compiten con la señal de la acción**:

- **Movimiento de cámara.** El flujo óptico (*optical flow*) que un método mide ya no corresponde solo al sujeto: mezcla el movimiento del objeto con el paneo, el zoom y el temblor de la cámara. Separar movimiento de cámara de movimiento de acción es, en sí mismo, un problema difícil que la Clase 36 discute.
- **Fondo abarrotado (*cluttered*).** Objetos y personas irrelevantes ocupan la escena y generan movimiento y textura que distraen al clasificador.
- **Iluminación variable, oclusión parcial y cuadros de baja calidad.** Condiciones que degradan cualquier descriptor de apariencia.

El propio baseline del paper lo confirma: las clases de **deportes** obtienen la exactitud más alta (50,54 %) precisamente porque suelen tener movimientos distintivos y fondos **menos** saturados, mientras que las de **interacción humano-objeto** rinden bajo (38,52 %) por tener fondos muy abarrotados y porque el movimiento informativo ocupa solo una pequeña porción del clip. La dificultad, en otras palabras, es una propiedad medible del régimen "in the wild".

## 3. Contribución central

La contribución de UCF101 tiene tres componentes:

1. **El corpus.** 101 clases, 13 320 clips, 27 horas (≈1600 minutos) de video de YouTube, todos en condiciones no controladas. Cerca del doble del dataset previo más grande en clases y en clips.
2. **La organización interna.** Una taxonomía de **cinco tipos** de acción y una estructura de **grupos** que codifica deliberadamente la correlación entre clips del mismo grupo, para que la evaluación no sea engañosamente fácil (ver §4).
3. **El baseline y el protocolo.** Un resultado de referencia con *bag of words* (44,5 %) y una receta de validación cruzada de 25 folds *leave-one-group-out* recomendada como estándar, para que los reportes sobre UCF101 sean directamente comparables.

Es útil recordar que la contribución de un buen dataset es **estructural**, no algorítmica: fija qué significa "hacerlo bien" para toda una comunidad durante años. UCF101 tuvo exactamente ese efecto.

## 4. Composición del dataset

### 4.1. Los cinco grupos de acciones

Las 101 clases se reparten en **cinco tipos**, una taxonomía que organiza la variedad de acciones humanas cubiertas:

1. **Human-Object Interaction** (interacción humano-objeto): p. ej. *Apply Eye Makeup*, *Brushing Teeth*, *Cutting In Kitchen*, *Hammering*, *Knitting*, *Mopping Floor*, *Typing*, *Writing On Board*.
2. **Body-Motion Only** (solo movimiento corporal): p. ej. *Baby Crawling*, *Body Weight Squats*, *Handstand Pushups*, *Jumping Jack*, *Lunges*, *Pull Ups*, *Push Ups*, *Rock Climbing Indoor*.
3. **Human-Human Interaction** (interacción humano-humano): un grupo pequeño (clases 97–101 en la matriz de confusión), p. ej. *Band Marching*, *Haircut*, *Head Massage*, *Military Parade*, *Salsa Spin*.
4. **Playing Musical Instruments** (tocar instrumentos musicales): p. ej. *Drumming*, *Playing Cello*, *Playing Daf*, *Playing Dhol*, *Playing Guitar*, *Playing Piano*, *Playing Sitar*, *Playing Tabla*, *Playing Violin*, *Playing Flute*.
5. **Sports** (deportes): el grupo más numeroso (clases 1–50 en la matriz de confusión), p. ej. *Archery*, *Baseball Pitch*, *Basketball*, *Diving*, *Fencing*, *Golf Swing*, *High Jump*, *Skiing*, *Surfing*, *Sumo Wrestling*, *Volleyball Spiking*.

En la Figura 2 del paper cada clase se muestra con un cuadro cuyo **color de borde** indica a qué tipo pertenece. La matriz de confusión (Figura 5) ordena las clases por tipo y marca las fronteras: **1–50 Sports, 51–60 Playing Musical Instrument, 61–80 Human-Object Interaction, 81–96 Body-Motion Only, 97–101 Human-Human Interaction**.

### 4.2. El linaje UCF11 → UCF50 → UCF101

UCF101 es literalmente una **extensión** de UCF50: hereda sus 50 clases (*Baseball Pitch, Basketball Shooting, Bench Press, Biking, Billiards Shot, Breaststroke, Clean and Jerk, Diving, Drumming, …, Walking with a Dog, Yo Yo*) y les **suma 51 clases nuevas** (*Apply Eye Makeup, Apply Lipstick, Archery, Baby Crawling, Balance Beam, Band Marching, Basketball Dunk, Blow Drying Hair, …, Uneven Bars, Wall Pushups, Writing On Board*). Sobre la porción heredada de UCF50 se hicieron **dos modificaciones menores** para homogeneizar la estructura: fijar el número de grupos en **25** para todas las acciones, y limitar cada grupo a **hasta 7 clips**.

El paper enmarca esto explícitamente: *UCF Sports, UCF11, UCF50 y UCF101 son los cuatro datasets de acción compilados por UCF en orden cronológico, cada uno incluyendo a su precursor.* Es una familia con herencia acumulativa, y **UCF11 —el que usa el laboratorio de la Clase 36— es el eslabón temprano** de esa cadena (11 clases, 1168 clips, ya de YouTube, con fondo dinámico y movimiento de cámara según la Tabla 2 del paper).

### 4.3. Estructura de grupos y por qué importa

Los clips de **cada** clase se dividen en **25 grupos** de **4 a 7 clips** cada uno. La clave es lo que comparten los clips de un mismo grupo: **características comunes como el fondo o los actores** (por ejemplo, provienen del mismo video fuente o de una misma sesión). Esta no es una decisión cosmética. Si el entrenamiento y la prueba compartieran clips del mismo grupo, un clasificador podría reconocer la acción **memorizando el fondo o al actor** en vez de aprender la acción misma —una fuga de información que inflaría artificialmente la exactitud—. Por eso el protocolo (§5) evalúa dejando **grupos completos fuera**.

### 4.4. Estadísticas y convención de nombres

La Tabla 1 del paper resume las características del dataset:

| Característica | Valor |
|---|---|
| Acciones (clases) | 101 |
| Clips | 13 320 |
| Grupos por acción | 25 |
| Clips por grupo | 4–7 |
| Duración media de clip | 7,21 s |
| Duración total | 1600 min (≈27 h) |
| Duración mínima de clip | 1,06 s |
| Duración máxima de clip | 71,04 s |
| Frame rate | 25 fps |
| Resolución | 320 × 240 |
| Audio | Sí (para las 51 acciones nuevas) |

Los videos se descargaron de YouTube y los irrelevantes se **removieron manualmente**. Todos los clips tienen frame rate y resolución **fijos** (25 fps, 320 × 240), se guardan en archivos `.avi` comprimidos con códec DivX, y el **audio se preserva** en los clips de las 51 acciones nuevas. La distribución de clips por clase (Figura 3) y la duración total y media por clase (Figura 4) muestran que las clases no son perfectamente uniformes en tamaño ni en duración.

La **convención de nombres** codifica la estructura de grupos en el propio nombre de archivo:

```
v_X_gY_cZ.avi
```

donde `X` es la etiqueta de clase, `Y` el número de grupo y `Z` el número de clip. Por ejemplo, `v_ApplyEyeMakeup_g03_c04.avi` es el clip 4 del grupo 3 de la clase *ApplyEyeMakeup*. Esta convención permite implementar el *leave-one-group-out* simplemente parseando nombres.

## 5. Baseline y protocolo de evaluación

El paper reporta un **baseline** con el pipeline de reconocimiento de acciones más aceptado de la época, *bag of words* sobre características espacio-temporales:

1. **Detección de puntos de interés.** De cada clip se extraen **esquinas Harris3D** (puntos de interés espacio-temporales, *space-time interest points*, STIP).
2. **Descripción local.** Para cada punto se computa un descriptor **HOG/HOF** de **162 dimensiones** (gradientes orientados + flujo óptico orientado, es decir, apariencia + movimiento).
3. **Construcción del codebook.** Se agrupan con **k-means** unos **100 000** STIP seleccionados al azar para formar un vocabulario visual de tamaño **k = 4000**, valor que rinde bien en un amplio rango de datasets.
4. **Cuantización.** Cada descriptor se asigna a su palabra visual más cercana (vecino más cercano) y cada clip queda representado por un **histograma de 4000 dimensiones** de sus palabras visuales.
5. **Clasificación.** Un **SVM no lineal multiclase** con **kernel de intersección de histogramas** y 101 salidas (una por acción) se entrena sobre los histogramas de entrenamiento; el video de consulta se clasifica con el mismo esquema.

El **protocolo recomendado** es una **validación cruzada de 25 folds bajo esquema *leave-one-group-out***: en cada fold se deja fuera uno de los 25 grupos para prueba y se entrena con el resto. Como los clips de un grupo comparten fondo y actores, dejar el grupo entero fuera **impide la fuga de información** descrita en §4.3. Los autores recomiendan explícitamente este montaje "usando todos los videos del dataset" para **mantener la consistencia** de los tests reportados sobre UCF101 —una preocupación que anticipa el problema de la reproducibilidad en benchmarks compartidos—.

**Resultado.** La exactitud global es **44,5 %**. Desglosada por tipo de acción:

| Tipo de acción | Exactitud |
|---|---|
| Sports | 50,54 % |
| Human-Human Interaction | 44,14 % |
| Human-Object Interaction | 38,52 % |
| Playing Musical Instrument | 37,42 % |
| Body-Motion Only | 36,26 % |

Los deportes lideran por sus movimientos distintivos y fondos menos abarrotados; la interacción humano-objeto queda al fondo porque combina fondos muy saturados con movimientos informativos que ocupan solo una fracción del clip. La matriz de confusión completa (Figura 5) muestra el patrón de errores entre las 101 clases.

Conviene subrayar el punto pedagógico: un baseline de 44,5 % sobre 101 clases —contra el 100 % casi trivial que los métodos de la época lograban en KTH— **no es un fracaso, es el objetivo del diseño**. UCF101 fue construido para ser difícil, para que hubiera espacio de mejora que empujara a la comunidad hacia mejores modelos.

## 6. Impacto

UCF101 se convirtió en el **benchmark estándar de reconocimiento de acciones durante buena parte de la década de 2010**. Casi todos los hitos de *deep learning* para video que la Clase 36 recorre reportaron su desempeño sobre UCF101, lo que lo volvió la vara común de comparación:

- Las primeras **CNN 2D** aplicadas cuadro a cuadro y agregadas temporalmente (el punto de partida que la clase discute) se midieron aquí.
- Las arquitecturas **2D CNN + RNN**, que pasan features por cuadro a una recurrente para modelar la dinámica temporal —el otro enfoque central de la clase— usaron UCF101 como banco de pruebas.
- Los **two-stream networks** (apariencia + flujo óptico), las **CNN 3D** (C3D, I3D) y modelos posteriores reportaron sistemáticamente sobre UCF101, muchas veces junto a HMDB51.

Su longevidad como benchmark hizo que la exactitud sobre UCF101 se saturara con los años (los mejores modelos superaron el 95 %), lo que a su vez motivó datasets aún más grandes —**Sports-1M**, **Kinetics**, **ActivityNet**— para volver a abrir espacio de mejora. Pero incluso después de esa saturación, UCF101 siguió usándose como *dataset de sanidad*, para **pre-entrenamiento/fine-tuning** y como referencia histórica. Su estructura de grupos y su protocolo de folds influyeron en cómo se diseñaron splits en datasets posteriores.

## 7. Limitaciones

Con la perspectiva del tiempo, UCF101 tiene límites que la Clase 36 debería nombrar:

- **Clips cortos y recortados (*trimmed*).** Cada clip contiene **una sola acción ya segmentada temporalmente** (duración media 7,21 s). El dataset resuelve **clasificación** de acción, no **detección/localización temporal** de acciones en video largo sin recortar (*untrimmed*), que es el problema realista de "¿qué acción ocurre y *cuándo*?". Datasets posteriores (ActivityNet, THUMOS) atacaron ese vacío.
- **Sesgo de YouTube.** Todo el corpus proviene de video subido a YouTube y filtrado manualmente. Eso introduce sesgos de **selección** (qué se sube y se vuelve popular), de **producción** (encuadres, edición, categorías sobre-representadas como deportes) y **culturales/geográficos** en qué acciones y contextos aparecen. Las 101 clases son un recorte particular, no una muestra representativa del espacio de acciones humanas.
- **Baja resolución y compresión.** 320 × 240 a 25 fps con códec DivX es modesto incluso para 2012; detalles finos de movimiento se pierden.
- **Desbalance moderado.** El número de clips y la duración por clase no son uniformes (Figuras 3 y 4), y los cinco tipos tienen tamaños muy distintos (Sports domina con 50 clases; Human-Human Interaction apenas 5).
- **Baseline de época.** El 44,5 % con *bag of words* es representativo de 2012, no del potencial del dataset; su valor es servir de piso, no de techo.

## 8. Conexión con la Clase 36 y con el laboratorio

La **Clase 36 (Introduction to Video Analysis)** cubre la definición de video, *object tracking*, *optical flow* y *action recognition* —sus tareas, datasets y desafíos— y los primeros enfoques de *deep learning* (CNN 2D, CNN 2D + RNN). UCF101 aparece en la clase como **uno de los grandes datasets de reconocimiento de acciones**, y encaja en varios de sus ejes:

- **Tarea.** UCF101 es el ejemplo canónico de **clasificación de acciones sobre clips recortados**, la formulación más simple de *action recognition* con la que la clase abre el tema.
- **Desafíos.** Movimiento de cámara, fondo abarrotado, iluminación variable y oclusión —exactamente los desafíos que la clase enumera— son constitutivos del régimen "in the wild" de UCF101, y el baseline los cuantifica (deportes fácil, interacción humano-objeto difícil).
- **Optical flow.** El descriptor HOF del baseline y, más tarde, los *two-stream networks* evaluados sobre UCF101 hacen del flujo óptico —tema propio de la clase— la señal de movimiento central. UCF101 es donde se midió cuánto aporta el flujo frente a la sola apariencia.
- **Enfoques de deep learning.** UCF101 es el banco donde se compararon CNN 2D por cuadro y CNN 2D + RNN, las dos familias que la clase presenta.

Y el vínculo más directo: **el laboratorio de la clase usa UCF11**, el eslabón temprano del mismo linaje UCF (UCF11 → UCF50 → UCF101). Trabajar el lab sobre UCF11 y leer este paper permite ver **el mismo problema a dos escalas**: 11 clases y 1168 clips en el lab, 101 clases y 13 320 clips en el benchmark de referencia. La estructura (video de YouTube, grupos con fondo/actor compartidos, clips cortos) es la misma; lo que cambia es la magnitud y, con ella, la dificultad.

**Enlaces internos:**

- Clase: [/clases/clase-36](/clases/clase-36) — Introduction to Video Analysis (object tracking, optical flow, action recognition).
- Laboratorio: usa **UCF11**, precursor directo de UCF101 en el linaje UCF.
- Datasets hermanos citados en el paper: KTH, Weizmann, UCF Sports, IXMAS, UCF11, HOHA, Olympic, UCF50, HMDB51 (Tabla 2 del paper).

---

**Nota sobre relevancia clínica.** El salto de datasets de acción "de laboratorio" (KTH, Weizmann) a datasets "in the wild" (UCF101) reproduce el desafío exacto del **análisis de video clínico**: los sistemas que reconocen acciones en cirugía (fases quirúrgicas, gestos instrumentales), en **rehabilitación** (calidad y repeticiones de ejercicios de fisioterapia) o en **monitoreo de pacientes** (detección de caídas, agitación, convulsiones, movilidad en cama) deben operar sobre video real de pabellón o de sala —con cámaras que se mueven o se ocluyen, iluminación desigual, campos abarrotados de personal y equipos, y sujetos parcialmente tapados por sábanas o instrumental—. La lección de UCF101 es que **el cuello de botella no es solo el modelo, sino el *ground-truth***: construir el equivalente clínico de UCF101 exige que expertos anoten temporalmente miles de clips (¿qué acción, en qué segundo?), una tarea costosa, sujeta a variabilidad inter-observador y atravesada por restricciones de privacidad del paciente que YouTube nunca impuso. Por eso el video clínico rara vez cuenta con benchmarks públicos del tamaño de UCF101, y por eso las estrategias que la Clase 36 y los papers de la era discuten —**pre-entrenar** en datasets grandes y genéricos como UCF101/Kinetics y luego hacer *fine-tuning* con los escasos datos etiquetados del dominio— son la vía práctica dominante para llevar el reconocimiento de acciones al hospital.
