# HMDB: A Large Video Database for Human Motion Recognition — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *HMDB: A Large Video Database for Human Motion Recognition*.
- **Autores:** Hildegard Kuehne (Karlsruhe Institute of Technology), Hueihan Jhuang, Estíbaliz Garrote, Tomaso Poggio (Massachusetts Institute of Technology) y Thomas Serre (Brown University).
- **Venue:** *International Conference on Computer Vision (ICCV 2011)*.
- **Año:** 2011.
- **Recurso:** el dataset se distribuye públicamente desde el sitio del Serre Lab, `serre-lab.clps.brown.edu/resources/HMDB/`.
- **Linaje:** nace del grupo de neurociencia computacional de Poggio y Serre en el MIT, cuyo modelo del córtex visual (features C2, streams ventral/dorsal) es a la vez uno de los baselines evaluados y la motivación biológica del trabajo. HMDB51 es contemporáneo de UCF50 (2010) y anterior a UCF101 (2012); juntos definieron el benchmark estándar de action recognition pre-deep-learning.

El paper describe la construcción y evaluación de **HMDB51**, en su momento la mayor base de datos de acción disponible: **51 categorías** con al menos 101 clips cada una, para un total de **6.766 clips** (cerca de 7.000) anotados manualmente y extraídos de fuentes diversas —películas digitalizadas, el archivo Prelinger, videos de internet, YouTube y Google Videos—. Cada clip fue validado por al menos dos observadores humanos y anotado con **meta-etiquetas** ricas: parte del cuerpo visible/oclusiones, movimiento de cámara, punto de vista, número de personas y calidad del video. Sobre esta base los autores generaron tres particiones balanceadas y evaluaron dos sistemas representativos con representaciones clásicas.

El argumento de fondo es que los datasets de acción de la época —KTH, Weizmann, IXMAS— eran **pequeños y controlados** (6-11 acciones, un solo actor escenificado, sin oclusión, iluminación y cámara limitadas), y el desempeño sobre ellos había llegado casi al techo (12 de 21 sistemas superaban 90% en KTH; en Weizmann 3 de 16 alcanzaban 100%). HMDB51 buscó deliberadamente **romper ese techo** con un benchmark diverso, realista y difícil, donde los mejores sistemas apenas rondaban el **23% de accuracy** (azar 2%).

Para la **Clase 36 (Introduction to Video Analysis)** este paper importa porque HMDB51 es uno de los datasets fundacionales de reconocimiento de acciones que la clase cita como básico: junto a UCF101 sirvió de campo de pruebas para la primera generación de arquitecturas de video profundas (two-stream, C3D, I3D). Además introduce dos ideas metodológicas que siguen vigentes: la **estabilización de video** para separar el movimiento de la cámara del movimiento del sujeto, y las **meta-anotaciones de calidad y condiciones** que permiten diseccionar dónde y por qué falla un sistema.

## 2. Contexto: datasets pequeños y controlados, y la necesidad de un benchmark diverso

Hacia 2011 la visión por computador ya disponía de grandes bases de imágenes estáticas realistas —ImageNet, PASCAL VOC, LabelMe, 80 Million Tiny Images, SUN— con miles de categorías. Los datasets de acción iban muy por detrás. Los más populares —**KTH** (6 acciones), **Weizmann** (9) e **IXMAS** (11)— compartían un patrón: un clip típico contenía **un único actor escenificado, sin oclusión, con muy poco desorden de fondo (clutter)**, iluminación uniforme y cámara fija; no eran representativos de la complejidad de los videos reales.

La consecuencia directa era la **saturación**: 12 de 21 sistemas superaban el 90% en KTH, y en Weizmann 14 de 16 alcanzaban 90% o más, 8 pasaban de 95% y 3 lograban un 100% perfecto. Cuando un benchmark se resuelve casi por completo, deja de discriminar entre métodos. Hacía falta un dataset nuevo, más grande y más difícil.

Existían esfuerzos previos en esa dirección —**Hollywood** y **Hollywood2** (clips de películas reales), **UCF Sports**, **UCF YouTube**, **Olympic Sports** y **UCF50** (50 categorías, mayormente deportes de YouTube)—. Estos datasets ya eran más desafiantes por sus grandes variaciones de movimiento de cámara, apariencia, posición, escala y punto de vista, más el desorden de fondo. Pero los autores identifican un **sesgo crítico** en varios de ellos: en los datasets de deportes las acciones son fácilmente distinguibles **por señales estáticas de forma o de escena**, sin necesidad de analizar el movimiento.

Lo demuestran con dos experimentos elegantes. Primero, **las poses estáticas bastan en UCF Sports**: anotando 14 ubicaciones de articulaciones (joints) por frame en las 9 categorías, un clasificador basado solo en las posiciones de joints en frames individuales alcanza **más del 98%** (azar 11%) — la cinemática del movimiento resulta innecesaria, en contradicción con los resultados clásicos de Johansson sobre percepción de movimiento biológico. Segundo, **la escena delata la acción**: como muchos deportes son específicos de una locación (pelota en césped, natación en agua, esquí en nieve), el descriptor global de escena **gist** predice la categoría mejor que las features espacio-temporales de nivel medio; se clasifica el fondo, no la acción.

En contraste, el experimento análogo sobre HMDB51 —10 categorías comparables a UCF50, joints anotados en más de 1.100 clips— arroja solo **35%** con poses estáticas (azar 10%), muy por debajo del **54%** obtenido con features de movimiento sobre el clip completo. Esto valida que **las categorías de HMDB51 se distinguen principalmente por el movimiento, no por la pose estática**, y lo convierte en un banco de pruebas legítimo para estudiar la contribución relativa de las señales de movimiento (motion) frente a las de forma (shape).

## 3. Contribución: HMDB51 y sus meta-anotaciones

La contribución central es el propio dataset. HMDB51 contiene **51 categorías de acción distintas**, cada una con **al menos 101 clips**, para un total de **6.766 clips** extraídos de un rango amplio de fuentes. En su momento era, según los autores, el dataset de acción más grande y quizás el más realista disponible. Las categorías se agrupan en **cinco tipos**:

1. **Acciones faciales generales:** sonreír, reír, masticar, hablar.
2. **Acciones faciales con manipulación de objetos:** fumar, comer, beber.
3. **Movimientos corporales generales:** rueda de carro, aplaudir, escalar, subir escaleras, zambullirse, caer al piso, flic-flac, pararse de manos, saltar, dominadas, flexiones, correr, sentarse, voltereta, ponerse de pie, girar, caminar, saludar.
4. **Movimientos corporales con interacción de objetos:** cepillarse el pelo, atrapar, desenvainar espada, driblar, golf, patear pelota, recoger, verter, andar en bici, montar a caballo, disparar arco/arma, batear béisbol, arrojar, entre otras.
5. **Movimientos corporales para interacción humana:** esgrima, abrazar, patear a alguien, besar, golpear con el puño, dar la mano, pelea de espadas.

El conjunto se generó pidiendo a estudiantes que revisaran videos de internet y películas digitalizadas y anotaran cualquier segmento que representara **una única acción humana no ambigua**, con estándares mínimos: una acción por clip, al menos 60 píxeles de altura para el actor principal, contraste mínimo, duración mínima de 1 segundo y artefactos de compresión aceptables. Se partió de un conjunto con **más de 60 categorías**, reducido a 51 reteniendo solo aquellas con al menos 101 clips.

Lo que distingue a HMDB51 de sus predecesores no es solo el tamaño, sino la **capa de meta-información** por clip, que permite seleccionar subconjuntos con precisión y diseñar experimentos flexibles. Los campos de meta-etiquetas son:

- **Parte del cuerpo visible / oclusiones:** si es visible la cabeza, el cuerpo superior, el inferior o el completo (full 56,3%, upper 30,5%, head 12,3%, lower 0,8%).
- **Movimiento de cámara:** si la cámara se mueve (59,9%) o está estática (40,1%) — aproximadamente **dos tercios** de los clips tienen movimiento de cámara.
- **Punto de vista de la cámara** respecto del actor: frente 40,8%, izquierda 22,1%, derecha 19,0%, atrás 18,2%.
- **Número de personas** involucradas (una, dos o múltiples).
- **Calidad del clip**, en tres niveles con definición operacional: **alta** (dedos y ojos del actor identificables, blur y compresión limitados; 17,1%), **media** (partes grandes del cuerpo identificables; 62,1%) y **baja** (partes grandes no identificables por blur y compresión; 20,8%).

Estas meta-etiquetas convierten a HMDB51 en un **instrumento de diagnóstico**: permiten preguntar cuánto cae el desempeño con movimiento de cámara o al bajar la calidad — exactamente el análisis que el paper realiza en su evaluación.

## 4. Composición, normalización y estabilización de video

### 4.1. Splits de entrenamiento y prueba

Para la evaluación se generaron **tres particiones distintas** de entrenamiento/prueba. Cada categoría aporta **70 clips de entrenamiento y 30 de prueba** (balance 70/30), con dos restricciones cuidadosamente diseñadas:

1. **No fuga entre train y test:** clips provenientes del **mismo video fuente** no pueden aparecer a la vez en entrenamiento y prueba. Esta es una salvaguarda contra el sobreajuste al contexto: dos clips del mismo film comparten escena, iluminación y a menudo actor, de modo que mezclarlos inflaría artificialmente el desempeño.
2. **Balance de meta-etiquetas:** las proporciones de posición de cámara, calidad, movimiento, etc. se distribuyen de forma pareja entre train y test, para que el conjunto de prueba sea representativo.

Como no siempre es posible satisfacer perfectamente ambas condiciones, los autores generaron un gran número de splits aleatorios y seleccionaron los tres mejores. Para que las tres particiones **no estuvieran correlacionadas entre sí**, usaron un enfoque codicioso (greedy): primero el split más balanceado, luego el segundo y tercero que menos se correlacionaran con los previos según la **distancia de Hamming normalizada**.

### 4.2. Normalización de video

Como las fuentes variaban en tamaño y frame rate, se homogeneizó el dataset: **altura escalada a 240 píxeles** (ancho proporcional para preservar el aspect ratio), **frame rate a 30 fps** y compresión con códec DivX 5.0 vía ffmpeg.

### 4.3. Estabilización de video: separar el movimiento de la cámara del movimiento del sujeto

Este es uno de los aportes metodológicos más citados del paper. Un desafío mayor de los clips del mundo real es el **movimiento de cámara significativo**, que —según las meta-etiquetas— afecta a **aproximadamente dos tercios** de los clips. El problema es conceptual: si el objetivo es reconocer una acción por el movimiento del sujeto, el movimiento de la cámara **contamina el cómputo de movimiento local** — un descriptor de flujo óptico no distingue por sí solo si el patrón proviene del brazo del actor o de un paneo. Por eso el movimiento de cámara debe corregirse, lo que hace de la estabilización un **paso de preprocesamiento clave**.

El procedimiento usa técnicas estándar de *image stitching* para alinear los frames de un clip, estimando un **plano de fondo** entre frames adyacentes:

1. Se detectan y emparejan features salientes en dos frames vecinos, con una medida de distancia que combina las diferencias absolutas de píxeles y la distancia euclidiana de los puntos.
2. Se usa el algoritmo **RANSAC** para estimar la transformación geométrica entre cada par de frames vecinos, de forma independiente.
3. Con esa transformación, todos los frames se deforman (warp) y combinan para producir un clip estabilizado.

La inspección visual mostró que el stitching funcionaba "sorprendentemente bien". Un detalle de diseño experimental: el desempeño se reporta **tanto para los clips originales como para los estabilizados**, para medir si eliminar el movimiento de cámara ayuda al reconocimiento — una pregunta cuya respuesta resultó contraintuitiva (Sección 5).

## 5. Baselines con representaciones clásicas

El paper evalúa dos sistemas representativos, ambos anteriores a la era del aprendizaje profundo de extremo a extremo.

### 5.1. Comparación con features de bajo nivel (color, gist) y de nivel medio (HOG/HOF)

Antes de los baselines completos, los autores analizan el **sesgo del dataset** comparando features de bajo nivel frente a features de movimiento de nivel medio. La lógica: en un dataset ideal y sin sesgo, **el color no debería predecir la categoría de acción**. Se consideraron **color+gris+PCA** (color medio HSV sobre una grilla 12×16 por frame), **gist** (representación holística de la escena basada en orientaciones, Oliva y Torralba) y **HOG/HOF** (bag-of-words espacio-temporal de nivel medio, referencia de "features de movimiento"). Una caída pequeña al pasar de HOG/HOF a una feature de bajo nivel significa que esta funciona casi tan bien como el movimiento — señal de sesgo. Los resultados (Tabla 2) son reveladores:

| Dataset | N | Color+Gris+PCA | Gist | HOG/HOF |
|---|---|---|---|---|
| Hollywood | 8 | 26,9% | 27,4% | 32,3% |
| UCF Sports | 9 | 47,7% | 60,0% | 58,6% |
| UCF YouTube | 11 | 38,3% | 53,8% | 58,9% |
| Hollywood2 | 12 | 16,2% | 21,8% | 51,7% |
| UCF50 | 50 | 41,3% | 38,8% | 47,9% |
| **HMDB51** | **51** | **8,8%** | **13,4%** | **20,2%** |

En **UCF Sports** gist (60,0%) supera incluso a HOG/HOF (58,6%): la escena predice la acción mejor que el movimiento. En UCF YouTube y UCF50 las features de bajo nivel son bastante predictivas, por sesgos de los videos amateur (puntos de vista y posiciones de cámara preferidos). En cambio, en **HMDB51** el color solo alcanza 8,8% y gist 13,4%, muy por debajo del 20,2% de HOG/HOF, lo que confirma que HMDB51 tiene **mucho menos sesgo de bajo nivel** y exige de verdad modelar el movimiento. El caso de Hollywood, donde bajo y medio nivel rinden parecido, se explica por el bajo número de películas fuente (12), que hace que clips del mismo film compartan escena.

### 5.2. HOG/HOF (bag-of-words espacio-temporal de Laptev)

El primer baseline completo es el sistema de Laptev y colegas, el enfoque dominante de la época basado en **información espacio-temporal local**. Se detectan esquinas 3D de Harris (Harris3D) en cada clip y se computa un descriptor local como concatenación de **HOG** (histograma de gradientes orientados, señal de forma) y **HOF** (histograma de flujo orientado, versión 3D basada en flujo de HOG, señal de movimiento) alrededor de cada esquina.

Sobre estos descriptores se construye un **bag-of-words**: se muestrean 100.000 descriptores del conjunto de entrenamiento y se aplica **k-means** con $k \in \{2000, 4000, 6000, 8000\}$ palabras visuales; cada clip se representa por el histograma de índices de codebook, un vector de dimensión $k$. La clasificación final usa una **SVM con kernel RBF** $K(u, v) = \exp(-\gamma \lVert u - v \rVert^2)$, cuyos hiperparámetros (costo $C$ y ancho de banda $\gamma$) se optimizan con validación cruzada de 5 folds. El mejor resultado para clips originales se obtuvo con $k = 8000$ y para estabilizados con $k = 2000$. Como validación de la reimplementación, sobre KTH se reprodujeron los resultados reportados: HOG 81,4% y HOF 90,7%.

### 5.3. Features C2 (modelo biológico del córtex visual)

El segundo baseline proviene del propio grupo del MIT: un modelo jerárquico inspirado en la **organización del córtex visual de primates**, con dos streams. El **stream ventral** (forma, reconocimiento de objetos invariante a escala y posición) parte de una pirámide de **filtros de Gabor** (unidades S1, análogas a las células simples de V1); la capa **C1** modela las células complejas por pooling de S1 en una región espacial local y a través de escalas, ganando tolerancia a traslación y tamaño; la capa **S2** empareja las entradas de C1 con un diccionario de $n$ prototipos aprendidos; y la capa **C2** produce, por pooling del máximo de las respuestas S2, un vector de dimensión $n$. A diferencia del bag-of-words, que hace cuantización vectorial (solo el índice del prototipo más cercano), C2 **retiene la similitud** (entre 0 y 1) con cada prototipo.

Jhuang et al. extendieron este modelo al **stream dorsal** (movimiento): sus unidades S1 tienen campos receptivos espacio-temporales no separables y responden mejor a **direcciones de movimiento** mediante filtros de Gabor espacio-temporales, con el resto de la arquitectura (C1, S2, C2) análoga y las unidades S2 sintonizadas a patrones de flujo óptico. En este trabajo se computan **ambos tipos de C2 (forma y movimiento) de forma independiente y se concatenan**.

### 5.4. Resultados

Ambos sistemas rinden de forma comparable, apenas por encima del **20%** (azar 2%). Los números clave (Tabla 3):

| Sistema | Clips originales | Clips estabilizados |
|---|---|---|
| HOG/HOF | 20,44% | 21,96% |
| C2 (movimiento+forma) | 22,83% | 23,18% |

Tres hallazgos merecen destacarse:

- **La estabilización ayuda solo marginalmente.** El resultado "más sorprendente" es que ambos sistemas mejoran apenas tras estabilizar el movimiento de cámara (+1,5 puntos HOG/HOF, +0,35 C2). Aunque el movimiento de cámara *debería* contaminar el cómputo de movimiento, corregirlo no rinde la mejora esperada — un resultado contraintuitivo sobre cuánto de la señal útil vive realmente en el movimiento del sujeto frente al contexto.
- **La calidad del clip es el factor dominante.** Ni las oclusiones ni la posición de cámara influyen significativamente. El factor mayor es la **calidad del video**: de alta a baja calidad ambos sistemas caen unos **10 puntos** (de 27,90%/28,62% a 17,18%/17,54% para HOG/HOF y C2). Una regresión logística lo confirma: son casi el doble de probables de acertar en video de alta frente a media calidad.
- **El movimiento de cámara a veces ayuda a C2.** Contraintuitivamente, C2 *mejora* con movimiento de cámara (25,20% vs. 19,13%), mientras HOG/HOF se mantiene estable; los autores conjeturan que ese movimiento incrementa la respuesta de los detectores S1 de bajo nivel. Descartan que prediga la acción por sí solo: clasificar únicamente los parámetros de movimiento estimados por el estabilizador arroja solo 5,29%.

El análisis de **forma vs. movimiento** (Tabla 6) muestra que las señales de movimiento solas (HOF 17,95%, C2-Motion 21,96%) superan claramente a las de forma solas (HOG 15,01%, C2-Shape 13,40%), y su combinación mejora muy moderadamente. Esto matiza el resultado previo de Schindler y Van Gool ("la forma y el flujo local de un solo frame bastan para reconocer acciones"): puede valer para acciones simples como las de KTH, pero **para las acciones complejas de HMDB51 el movimiento es más poderoso que la forma**. Finalmente, los autores verifican que el bajo desempeño no es solo consecuencia del mayor número de clases: al comparar 10 categorías comunes entre UCF50 y HMDB51 con HOG/HOF, hay una caída suave de **66,3% a 54,3%** con igual azar (10%), lo que confirma que HMDB51 es genuinamente más difícil por su clutter e intra-clase variación, no solo por tener más categorías.

## 6. Impacto

HMDB51 se convirtió, junto con **UCF101** (2012, del mismo linaje que UCF50), en el **par de benchmarks estándar del reconocimiento de acciones en la era pre-deep-learning y durante la transición a ella**. El comentario de los autores —que con apenas ~23% el dataset "es probablemente un buen lugar para empezar", recordando que Caltech-101 también comenzó en torno al 16%— resultó profético: HMDB51 tenía justo el margen de dificultad para impulsar años de progreso. Cuando llegaron las arquitecturas profundas de video —**two-stream networks**, **C3D**, **I3D** y sucesores—, HMDB51 y UCF101 fueron los datasets de evaluación por defecto. Su tamaño modesto los hizo especialmente útiles para medir **transferencia**: I3D, por ejemplo, mostró el valor del pre-entrenamiento en Kinetics reportando saltos de desempeño al hacer fine-tuning sobre HMDB51 y UCF101. Las tres particiones balanceadas y la restricción de no-fuga entre train y test, diseñadas en 2011, siguieron siendo el protocolo de evaluación citado durante más de una década.

## 7. Limitaciones

Los propios autores son explícitos sobre los límites del trabajo:

- **Sigue lejos de la complejidad real.** Con 51 categorías y poco menos de 7.000 clips, HMDB51 "aún está lejos" de capturar la riqueza de los videos reales. Es grande para su época, pero pequeño para los estándares posteriores (Kinetics tendría cientos de miles de clips).
- **Baselines de baja capacidad representativa.** Aunque los métodos son "razonablemente robustos" frente a degradaciones de bajo nivel (posición/movimiento de cámara, oclusiones), permanecen **limitados en su poder representativo**; el ~23% es un diagnóstico de la debilidad de las features clásicas tanto como de la dificultad del dataset.
- **La estabilización no rinde lo esperado.** Que corregir el movimiento de cámara mejore tan poco sugiere que, o bien el stitching no elimina toda la interferencia, o bien las features de movimiento no aprovechan la señal limpia. Queda como pregunta abierta.
- **Sesgos residuales.** HMDB51 está mucho menos sesgado que los datasets de YouTube, pero no es completamente insesgado; algunas categorías aún pueden apoyarse en el contexto de escena.

## 8. Conexión con la Clase 36 (Introduction to Video Analysis)

La Clase 36 introduce el análisis de video y presenta los datasets fundacionales de reconocimiento de acciones. HMDB51 es uno de los básicos que la clase cita explícitamente (51 acciones, ~6.849 videos, con estabilización de video). Tres ideas del paper conviene que el estudiante internalice:

1. **Un buen benchmark de video debe forzar el modelado del movimiento, no del contexto.** El experimento de poses estáticas (98% en UCF Sports vs. 35% en HMDB51) enseña a desconfiar de datasets que se resuelven con señales de forma o de escena. Un dataset donde el color predice la acción está midiendo otra cosa.
2. **El movimiento de cámara es un factor de confusión (confounder) de primer orden en video, y la estabilización es el intento clásico de controlarlo.** Aunque en este paper la estabilización rindió poco, la separación conceptual entre movimiento del sujeto y movimiento del sensor es central en todo el análisis de video posterior (motion compensation, ego-motion, flujo óptico compensado).
3. **Las meta-anotaciones convierten un dataset en un instrumento de diagnóstico.** Poder desglosar el error por calidad, oclusión, punto de vista y movimiento de cámara es lo que permite descubrir que la calidad del clip —no el movimiento de cámara— era el factor dominante. Sin esas etiquetas, ese hallazgo habría sido invisible.

**Nota final — relevancia para video clínico.** Las dos ideas metodológicas de HMDB51 se trasladan casi literalmente al análisis de video médico. Primero, **controlar el movimiento de cámara / estabilización** es crítico donde el video proviene de fuentes móviles: endoscopía, laparoscopía, ecografía de mano, análisis de marcha o monitoreo de pacientes. Si el movimiento del endoscopio o de la sonda se confunde con el de la anatomía o del gesto que se quiere cuantificar —un temblor, una convulsión, un patrón de marcha—, el sistema medirá el artefacto en lugar del signo clínico; la separación entre ego-motion del instrumento y movimiento del sujeto que HMDB51 aborda con stitching y RANSAC es exactamente el problema a resolver. Segundo, las **meta-anotaciones de calidad** son igualmente valiosas: en video clínico la calidad varía enormemente (iluminación, motion blur, oclusión por instrumental o tejido, compresión), y el hallazgo de que la calidad del clip domina el desempeño advierte que un modelo médico debe evaluarse estratificado por calidad de adquisición y, idealmente, abstenerse o señalar baja confianza sobre un video degradado en lugar de emitir una predicción igualmente segura.

**Enlaces internos:**

- Clase: [/clases/clase-36](/clases/clase-36) — Introduction to Video Analysis (reconocimiento de acciones, datasets, features espacio-temporales).
- Dataset hermano: UCF101 — el otro benchmark estándar pre-deep-learning con el que HMDB51 formó par.
