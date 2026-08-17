---
title: "Teoría - Tracking de objetos en video"
weight: 10
math: true
---

> **Recorrido de la Clase 42** del Diplomado IA UC (Carlos Aspillaga, DCC PUC). Noventa y cinco diapositivas organizadas en cinco secciones: introducción, la división offline/online, SORT y DeepSORT, modelos integrados, y el trabajo práctico. El profesor declara sus fuentes al comenzar: Tomás Vergara Brown, Álvaro Soto y los papers originales de cada modelo.

---

## 1. El objetivo (diapositivas 5-7)

La clase abre mostrando dos frames de una calle con peatones encajonados y numerados —129, 180, 159, 165, 137— y los mismos números sobre las mismas personas en el frame siguiente.

> *"Nuestro objetivo es trackear objetos en un video."*

Y define qué significa hacerlo, en dos viñetas:

- **reconocer objetos**;
- **mantener identidades de los objetos**.

La primera ya la resuelve un detector. La segunda es la clase entera.

Luego viene la oposición que ordena todo lo demás, presentada en dos columnas:

| **Razonamiento espacial** | **Razonamiento espacio-temporal** |
|---|---|
| **Detección de objetos**: localización espacial sin componente temporal. | **Object tracking**: asociación continua de detecciones en el tiempo. |
| Comprende snapshots aisladas, objetivos estáticos e inferencia directa. No hay consciencia de la componente temporal. | Requiere preservación de identidad, comprensión de dinámicas de movimiento, y recuperación de la asociación ante oclusión. |

{{< concept-alert type="clave" >}}
Todo lo que sigue en la clase se puede leer como respuestas sucesivas a las tres exigencias de la columna derecha. **Preservación de identidad** → el algoritmo húngaro. **Dinámicas de movimiento** → el filtro de Kalman. **Recuperación tras oclusión** → los descriptores de apariencia. Cada componente de SORT y DeepSORT ataca exactamente una de ellas.
{{< /concept-alert >}}

## 2. Por qué querríamos trackear objetos (8-15)

Ocho diapositivas casi sin texto, solo imágenes, con la misma pregunta como subtítulo. El catálogo:

- **Deporte**: jugadores de la NBA seguidos por nombre a través de cuatro frames distantes (#51, #127, #188, #276); y un mapa de calor de acciones de Lionel Messi sobre la cancha — el producto que solo existe si alguien mantuvo la identidad durante noventa minutos.
- **Conducción**: un asistente de detección de vehículos en el parabrisas, y la salida de un sistema Mobileye con peatones y autos encajonados y sus distancias estimadas.
- **Retail**: Amazon Go — *"no lines, no checkout, just grab and go"*. Un sistema que cobra sin cajas es, en el fondo, un tracker que no puede perder a nadie.
- **Robótica**: un brazo manipulador que sigue un objeto que una persona le acerca.
- **VFX y postproducción**: *motion tracking* en Cinema 4D para insertar objetos sintéticos coherentes con el movimiento de cámara.

Cierra con una línea sobria: *"Trackear objetos puede ser importante, útil e interesante."*

## 3. Una tarea, o muchas (16-18)

> *"Formulamos el trackeo de objetos como un gran problema, pero eso no es la realidad."*

Y enumera las variaciones:

- un objeto contra **múltiples objetos**;
- una cámara contra **múltiples cámaras**;
- cámara **dinámica** contra cámara **estática**.

Vale hacer explícito lo que la clase deja implícito, porque afecta cómo leer sus últimas diapositivas: el primer eje separa dos literaturas casi disjuntas. **SOT** (*single object tracking*) recibe una caja en el primer frame y solo re-localiza; **MOT** (*multiple object tracking*) descubre objetos que entran y salen y gestiona un número variable de identidades. SORT y DeepSORT son MOT; [SUTrack](/papers/sutrack-chen-2024), que aparece en la sección de modelos integrados, es SOT.

## 4. Los desafíos (19-31)

Trece diapositivas bajo el mismo encabezado —*"Trackear objetos es difícil"*— con una lista y luego una imagen por desafío:

- cambios en la iluminación,
- variaciones en la pose,
- **oclusiones**,
- variaciones en escala,
- deformaciones,
- variaciones intra-clase,
- restricciones de tiempo real,
- muchos objetos.

Las imágenes son deliberadamente concretas: ocho miniaturas de un mismo soldado de plomo en poses distintas; un corredor a dos escalas en el mismo cuadro; seis sillas que no se parecen en nada entre sí; una avenida de Bangkok con cientos de vehículos superpuestos.

Y luego el remate, en tres diapositivas encadenadas:

> *"Y lo peor… ¡es que todo puede ocurrir al mismo tiempo!"*
> *"Parece imposible de resolver. Sin embargo..."*
> *"Los seres humanos no tenemos ningún problema trackeando objetos, y efectivamente ya hay métodos que lo resuelven relativamente bien."*

De la lista, el que estructura la literatura es la **oclusión**, porque es el único que elimina la evidencia en lugar de degradarla. Los demás ensucian la señal; la oclusión obliga a sostener una identidad sin ninguna observación durante $k$ frames. Casi todo lo que la clase presenta después —$T_{\text{lost}}$, $A_{\max}$, la galería de apariencia, la cascada— son decisiones sobre qué hacer durante ese hueco.

## 5. Online contra offline (32-34)

> **Online Tracking**: tracking en tiempo real. No se pueden utilizar frames del *futuro* para hacer el tracking.
> **Offline Tracking**: tenemos acceso a todo un video, y queremos hacer tracking de este.

Y la hoja de ruta: *"Vamos a empezar con offline tracking, pero después nos vamos a enfocar más con online tracking, pues tiene más casos de uso."*

## 6. Offline tracking (35-42)

El argumento se construye en pasos cortos:

**Primero**, la reducción: *"el tracking de objetos es un subproblema del reconocimiento de objetos. Entonces, podemos usar los mismos modelos de reconocimiento de objetos."*

**Segundo**, lo que queda pendiente una vez que se tiene el detector: *"Tenemos el reconocimiento de objetos listo. ¿Qué problemas faltan resolver?"* — **múltiples objetos** y **mantener las identidades**.

**Tercero**, la formulación: *"Queremos encontrar los caminos de los objetos en un video."* El problema se parte en dos:

1. encontrar los objetos en cada frame;
2. hacer asociaciones entre objetos en distintos frames para formar *caminos*.

**Cuarto**, la ilustración: una escena con cuatro peatones vista desde arriba, y debajo dos grillas. En la izquierda, una nube de puntos morados sin estructura —las detecciones sueltas—. En la derecha, los mismos puntos coloreados y unidos por curvas —las trayectorias—. Entre una y otra, una flecha con la frase que abre la parte central de la clase:

> *"Necesitamos una buena manera de asociar puntos."*

Y la respuesta: *"Podríamos **aprender** cómo asociar puntos correctamente. ¿cómo? Aprendiendo una métrica de distancia."*

{{< concept-alert type="recordar" >}}
Nótese lo que acaba de ocurrir. En el marco offline el problema se convirtió en un **grafo**: los nodos son detecciones, las aristas son asociaciones candidatas, una trayectoria es un camino. Lo que hay que aprender no es el camino sino el **costo de las aristas** — una distancia entre dos recortes de imagen que valga poco cuando son el mismo objeto y mucho cuando no.
{{< /concept-alert >}}

## 7. Paréntesis: aprender una métrica de distancia (43-55)

La clase abre un paréntesis explícito, con una pregunta y dos pares de fotos: *"¿Son estos pares de imágenes de la misma persona?"* — dos fotos de Tiger Woods, dos de Keanu Reeves.

### Red siamesa

Dos ramas idénticas con **pesos compartidos** (*tied weights*) que procesan cada imagen y producen features; una función de pérdida opera sobre ambos. Los objetivos:

- que las imágenes de la **misma** clase estén **cerca**;
- que las imágenes de clases **diferentes** estén **lejos**.

El *pairwise ranking loss* que la clase escribe:

$$L(f(I_1), f(I_2)) := |f(I_1) - f(I_2)| \qquad \text{si son de la misma clase}$$

$$L(f(I_1), f(I_2)) := \max\{0,\; m - |f(I_1) - f(I_2)|\} \qquad \text{si son de clases diferentes}$$

### Por qué es inestable

La clase plantea la pregunta —*"pero la red siamesa es bastante inestable de entrenar (¿por qué?)"*— y responde: **hay shortcuts**.

- Si se usan imágenes de la misma clase, **tira todo al mismo punto**.
- Si se usan imágenes de distintas clases, **simplemente aleja todo del centro**.

Cada término, aislado, tiene una solución degenerada trivial. El objetivo solo es correcto si ambos están equilibrados dentro del mismo lote, y eso depende del muestreo, no de la pérdida.

### Triplet network

La corrección: tres ramas con pesos compartidos —ancla, positivo, negativo— y una pérdida donde los dos términos **compiten en la misma expresión**:

$$L(f(I_1), f(I_2), f(I_3)) := \max\{0,\; m - |f(I_1)-f(I_3)| + |f(I_1)-f(I_2)|\}$$

Ahora el colapso deja de ser solución: si todo va al mismo punto, ambas distancias son cero y la pérdida vale $m > 0$.

La clase cierra el paréntesis con dos imágenes. Una, el espacio de features aprendido, con las fotos de Tiger Woods agrupadas a la izquierda y las de Keanu Reeves a la derecha. Otra, la misma idea con monos, sillas y payasos, bajo la anotación: **"Esto es generalizable a cualquier tipo de objetos."**

Ver [Re-identificación](/fundamentos/re-identificacion), [Triplet Loss](/fundamentos/triplet-loss) y [Metric Learning](/fundamentos/metric-learning).

## 8. De vuelta al offline (56-60)

Con la métrica en mano, la respuesta al problema del grafo: *"Podemos usar una red siamesa/triplet network para tener features de los objetos y conectar los más similares."*

Y una variante que la clase señala como interesante: **múltiples cámaras**. Muestra un sistema con dos vistas de un pasillo y un mapa de calor de ocupación, con la anotación: *"Podemos seguir usando los features de la red siamesa/triplet network para conectar puntos."*

Es correcto y además es el escenario que originó [IDF1](/papers/idf1-ristani-2016): sin solapamiento entre campos de visión no hay continuidad geométrica que explotar, y la apariencia es lo único que queda.

## 9. Online tracking: la anatomía (61-67)

> *"A diferencia de offline tracking, ahora no podemos acceder a frames futuros. Eso hace más difícil la detección, pues tenemos que **predecir** las nuevas localizaciones."*
> *"Esto se puede resolver con modelos de movimiento."*

Y el esqueleto que ordena el resto de la clase:

**1. Detección de objetos**
  - 1.1 **Localización en el espacio**: usando un modelo preentrenado.
  - 1.2 **Representación del objeto**: podríamos representarlo como el *bounding box* obtenido.

**2. Búsqueda de objetos**
  - 2.1 **Asociación de datos (modelo de movimiento)**: podemos estimar la nueva localización del objeto en base a sus últimas localizaciones.
  - 2.2 **Medida de similaridad**: podríamos medir qué tanto se asemejan la *bounding box* predicha con la de los objetos detectados en el nuevo frame.

Las cuatro diapositivas ilustran cada paso sobre la misma foto de una persona caminando en una duna: primero la caja roja del detector, luego varias cajas amarillas tenues —las posiciones posibles según el modelo de movimiento—, y finalmente una caja verde, la elegida.

{{< concept-alert type="clave" >}}
Este esqueleto de cuatro casillas es el mapa de toda la literatura de MOT. Cada método posterior es una elección distinta en una o dos de ellas:

| | SORT | DeepSORT | ByteTrack | Tracktor |
|---|---|---|---|---|
| 1.1 Localización | Faster R-CNN | Faster R-CNN | YOLOX | Faster R-CNN |
| 1.2 Representación | la caja | caja + embedding 128-D | la caja | la caja |
| 2.1 Movimiento | Kalman v. constante | Kalman v. constante | Kalman v. constante | **el regresor del detector** |
| 2.2 Similaridad | IoU | coseno + compuerta Mahalanobis | IoU en **dos rondas** por score | ninguna |
{{< /concept-alert >}}

## 10. SORT (69-77)

> **SORT**: *Simple Online Realtime Tracking*. Fue un paper muy influyente en el área, que con un algoritmo sencillo tenía buenos resultados.

**Detección.** *"Usando Faster R-CNN obtienen detecciones de objetos en un frame."*

**Estado.** Inicializan una identificación del objeto mediante una estimación del *bounding box* y su velocidad:

$$x = [u,\, v,\, s,\, r,\, \dot{u},\, \dot{v},\, \dot{s}]$$

donde $u,v$ es la ubicación del centro, $s$ el área, $r$ la razón ancho/alto, y los puntos las velocidades de cada medición. *"Todas las velocidades se inicializan en 0, y se calculan después a partir del frame anterior."*

Nótese que **no hay $\dot{r}$**: la razón de aspecto se trata como constante. La clase reproduce el estado exactamente como está en el paper.

**Asociación.** Dos pasos:

- calcular la métrica de costo de **Intersection over Union (IoU)** entre la caja predicha y todas las detecciones nuevas;
- usar el **algoritmo húngaro** para hacer asignaciones óptimas que minimicen este costo.

La clase explica el húngaro con una matriz de $3\times 3$: *"Tenemos que asignar $n$ objetos a otros $n$ objetos, intentando minimizar un costo. El algoritmo toma la matriz, y retorna las asignaciones óptimas de tal forma de minimizar el costo."* Ver [Asignación Húngara](/fundamentos/asignacion-hungara).

**Los detalles del ciclo de vida.** Dos, que la clase menciona al pasar y que son más determinantes de lo que parecen:

- se establece un **mínimo de intersección** necesaria entre la caja predicha y la nueva;
- si el objeto no tiene *match* por más de $n$ frames, se considera que desapareció.

{{< concept-alert type="advertencia" >}}
Ese $n$ del segundo punto vale **1** en todos los experimentos de SORT. Una trayectoria muere tras un solo frame sin detección. Los autores lo justifican explícitamente: el modelo de velocidad constante es mal predictor, y la re-identificación está fuera del alcance del trabajo. Si el objeto reaparece, *"el seguimiento se reanudará implícitamente bajo una identidad nueva"* — SORT **acepta** el cambio de identidad en vez de evitarlo.

En DeepSORT ese mismo parámetro pasa a $A_{\max}=30$. Buena parte de la diferencia entre las métricas de ambos sistemas viene de ahí, no del descriptor.
{{< /concept-alert >}}

**El balance.** *"Una gran ventaja es que SORT es extremadamente directo de implementar, y muy rápido en la práctica, por lo que tuvo mucho impacto. Sin embargo, SORT no es muy bueno para manejar largas oclusiones, y se propusieron nuevos algoritmos para mejorarlo."*

Hay un resultado del paper que la clase no menciona y que es el más transferible: **cambiar el detector, dejando el tracker intacto, mueve MOTA de 15,1 a 34,0**. Ver [SORT (2016)](/papers/sort-bewley-2016).

## 11. DeepSORT (78-87)

> **DeepSORT**: una ligera modificación de SORT que agrega features *aprendidos* para mantener mejores identificaciones de objetos.

**El diagnóstico de la clase.** *"Un problema con SORT es que el modelo de movimiento es demasiado simple: calculan una velocidad y así aproximan la nueva localización. Eso no considera las incertezas existentes en las mediciones. Por ejemplo, objetos de alta velocidad van a tener mayor incerteza en su nueva localización."*

**Mahalanobis.** *"En lugar de usar IOU, usan la distancia de Mahalanobis, que mide cuántas desviaciones estándar está la localización predicha en comparación a cada bounding box detectado."* La ilustración es la clásica: una nube de puntos elíptica con sus componentes principales, y un punto exterior a distancia euclídea corta pero Mahalanobis larga.

Y la glosa: *"Se puede pensar como si estuviera estimando regiones de probabilidad de la siguiente localización, en lugar de un lugar en particular."*

**Embedding.** *"Además, consideran usar un vector de embedding aprendido para identificar un objeto."*

**La combinación.** *"Entonces, al calcular la similaridad entre un objeto existente y una nueva detección, mezclan: la distancia Mahalanobis, la similitud entre los embeddings."*

**El balance.** *"En la práctica DeepSORT es más robusto que SORT para mantener identidades, y sigue siendo muy eficiente y relativamente sencillo de implementar."*

{{< concept-alert type="advertencia" >}}
Tres precisiones sobre esta sección, todas verificables contra el paper y desarrolladas en la [profundización](profundizacion):

**Sobre la incertidumbre.** SORT **ya corre un filtro de Kalman completo** y por lo tanto ya propaga covarianzas. Lo que le falta no es el modelo de incertidumbre sino una métrica que lo consulte: IoU es ciega a $S$, Mahalanobis la usa. La diferencia no es tener incertidumbre, es usarla al asociar.

**Sobre la mezcla.** La fórmula $c_{ij} = \lambda d^{(1)} + (1-\lambda) d^{(2)}$ está en el paper, pero **los experimentos usan $\lambda = 0$**: el costo es puramente apariencia y Mahalanobis actúa solo como compuerta binaria con umbral $\chi^2_{0{,}95;4}=9{,}4877$.

**Sobre el diagrama.** La figura de arquitectura que aparece en esta sección —con los bloques *Regression / Classification / Detection*, la pregunta *"Kill $b^k_t$?"* y el bloque *"Init new $b^k_t$"*— no es de DeepSORT: es la de [Tracktor](/papers/tracktor-bergmann-2019) (Bergmann et al., 2019), un método con un principio opuesto, que elimina la asociación en vez de refinarla.
{{< /concept-alert >}}

**El límite.** La clase cierra la sección con la objeción correcta:

> *"Pero este approach sigue teniendo problemas. Supone que el objeto se movió poco. ¿Es razonable suponer eso? Tal vez… pero ¿qué si la cámara se mueve mucho? ¿O si hay periodos grandes de oclusión?"*

Y ofrece dos caminos de mejora del modelo de movimiento —**un modelo que compense el movimiento de la cámara** y **un modelo que considere zonas ciegas**— más una tercera vía: *"otra técnica es usar algoritmos de re-identificación de objetos."*

Esa pregunta es exactamente la que [OC-SORT](/papers/oc-sort-cao-2022) responde seis años después, y su respuesta es un cuarto camino que la clase no menciona: **no mejorar el modelo de movimiento sino dejar de confiar en él** durante la oclusión, y reconstruirlo a partir de las observaciones que la rodean.

## 12. Modelos integrados (89-93)

La clase salta de 2017 a 2024-2025.

### SUTrack (2024)

> En SUTrack se mueven hacia la multimodalidad (RGB, profundidad, termal, eventos, *language*) con una tokenización universal.
> - Consolida **5 tareas** en un solo Transformer.
> - Usan un ***soft token type embedding*** para conseguir tracking agnóstico a la modalidad.
> - El paradigma integrado supera a modelos especializados (**sinergia multimodal**).

El lema de la figura: *One Model, One Training, Five Tasks*. Conviene tener presente que [SUTrack](/papers/sutrack-chen-2024) es **SOT**: lo que unifica son modalidades de entrada, no MOT y SOT.

### Ideas para dominios específicos

Tres líneas, una por dominio:

- **UAV aéreos**: *SocialTrack* usa las dinámicas de movimiento de grupos.
- **Vehículos autónomos 3D**: *MCTrack* proyecta la velocidad de los píxeles a la velocidad real 3D.
- **Biología**: *Cell-TRACTR* segmenta células de forma *end-to-end*.

El patrón común es informativo: los tres reemplazan el supuesto genérico de velocidad constante por una **restricción del dominio** —la coherencia del grupo, la geometría métrica de la escena, la dinámica de división celular—.

### YOLO11 a YOLO26

- **Remueven el NMS** (*Non-Maxima Suppression*).
- Mecanismo **STAL** (*Small-Target-Aware-Labeling*): mantiene persistencia del tracking cuando los objetos se pierden en la distancia.

Quitar la supresión no máxima importa para tracking más de lo que sugiere: el NMS es un post-proceso no diferenciable que decide, sin contexto temporal, cuál de dos cajas superpuestas sobrevive. En una oclusión parcial puede eliminar justamente la caja del objeto ocluido — el mismo problema que [ByteTrack](/papers/bytetrack-zhang-2021) ataca desde el umbral de confianza.

### SAM 3 (2025)

> Agregan una cabecera de **"Presencia"**. Detector y tracker con arquitecturas separadas, que permite hacer tracking de conceptos complejos mediante lenguaje natural.
> *"Track all yellow buses"* — SAM 3 asigna IDs a cada instancia que coincide con el prompt semántico.
> Obtiene 22 % de mejora en LVIS Zero-Shot AP vs. el mejor modelo anterior.

Una diapositiva adicional muestra la optimización *SAM 3.1 Object Multiplexing*: memoria compartida para multi-tracking conjunto, 50 % menos de VRAM, 30 ms por imagen con 100+ detecciones, hasta 7× de aceleración.

Sobre la cifra del 22 %: el paper publicado reporta **48,8 de mask AP zero-shot en LVIS contra 38,5** del mejor modelo previo — 10,3 puntos absolutos, 26,8 % relativo. Ver [SAM 3 (2025)](/papers/sam3-meta-2025).

---

## Lo que la clase deja fuera

Cuatro huecos que conviene tener presentes al estudiar, todos desarrollados en la [profundización](profundizacion):

1. **Las métricas.** La clase compara SORT con DeepSORT cualitativamente (*"más robusto para mantener identidades"*) sin mencionar MOTA, IDF1 ni HOTA. Como MOTA está dominada por errores de detección, la mejora real de DeepSORT —45 % menos ID switches— es casi invisible en la métrica con que se rankeaban los benchmarks de la época.

2. **Los años 2018-2023.** Entre DeepSORT y SUTrack hay un salto de siete años que incluye a [Tracktor](/papers/tracktor-bergmann-2019), [FairMOT](/papers/fairmot-zhang-2020), [ByteTrack](/papers/bytetrack-zhang-2021) y [OC-SORT](/papers/oc-sort-cao-2022) — y el resultado incómodo de que en 2022 el estado del arte seguía siendo Kalman más húngaro.

3. **La cascada de matching**, que es la contribución algorítmica más original de DeepSORT y un ejemplo notable de métrica estadísticamente correcta con incentivo perverso.

4. **El umbral de detección como decisión del tracker.** En la clase aparece enterrado en el paso 1 y nunca se revisa; ByteTrack demuestra que es una de las palancas más rentables.

---

**Siguiente:** [Profundización](profundizacion) — la derivación del filtro de Kalman en el estado de SORT, por qué Mahalanobis premia la incertidumbre, la aritmética de MOTA, y qué mide realmente cada métrica. Después, la [práctica](practica): SORT implementado desde cero y las tres métricas verificadas, en triple framework.
