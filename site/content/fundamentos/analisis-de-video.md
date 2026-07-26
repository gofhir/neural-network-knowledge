---
title: "Análisis de Video"
weight: 120
math: true
---

El **análisis de video** extiende la visión por computador de una imagen estática a una **secuencia ordenada de imágenes**. El salto no es trivial: agrega una dimensión —el **tiempo**— que trae consigo el rasgo más distintivo del video, el **movimiento**, pero también un costo computacional que crece con la duración. Este fundamento acompaña a la [Clase 36](/clases/clase-36), la introducción al análisis de video: define qué es un video, por qué el movimiento lo cambia todo, y las dos grandes áreas del campo —el **seguimiento de objetos** y el **reconocimiento de acciones**.

---

## 1. ¿Qué es un video?

Un **video** es un **conjunto ordenado de frames** (imágenes). Lo caracterizan:

- **Duración** (longitud fija, en segundos).
- **Resolución** (360p, 1080p, 4k…), constante a lo largo de la secuencia.
- **Color** (escala de grises o RGB).
- **Cantidad de frames** por segundo (fps).

Según cómo se accede a él, se distingue:

- **Video stream** (feed en vivo): solo se dispone del frame actual y los anteriores. Es el escenario de las aplicaciones en tiempo real.
- **Video sequence** (video de longitud fija): se tiene acceso completo, del primer al último frame.

---

## 2. Imagen vs. video: el movimiento lo cambia todo

Por definición, un video es una secuencia de imágenes cuyos frames están relacionados **espacial y temporalmente**. La diferencia clave con el análisis de imágenes es el **movimiento**:

{{< concept-alert type="clave" >}}
El **movimiento** es la característica más poderosa del análisis de video. Dos acciones pueden verse casi idénticas frame a frame —*correr* y *trotar* tienen los mismos píxeles promedio— pero se distinguen por su **dinámica temporal**. Analizar un video cuadro por cuadro, como si fueran imágenes independientes, **descarta** justamente esta información. Modelar el movimiento es el problema central del campo.
{{< /concept-alert >}}

Otras dos diferencias importan:

- **Multimodalidad.** Un video incluye al menos imagen y **audio**, y frecuentemente texto (subtítulos). El análisis de video puede aprovechar todas estas modalidades —un puente natural con el [dominio de audio](/dominios/audio) y los modelos multimodales.
- **Tamaño.** Un video necesita mucho más almacenamiento y capacidad de procesamiento que una imagen: un clip de 16 frames a 224×224 son ~16× los píxeles de una imagen, y ese factor crece linealmente con la duración.

Aunque el análisis de video no es nuevo (ya en **1878** Muybridge usó múltiples cámaras para capturar 24 imágenes del galope de un caballo y responder si levanta las cuatro patas del suelo), su automatización con deep learning es reciente y tiene muchos problemas sin resolver.

---

## 3. Las dos grandes áreas

### 3.1 Seguimiento visual de objetos (VOT)

El **Visual Object Tracking** consiste en **localizar un objeto en todos los frames** de un video, dado únicamente su ubicación en el **primer frame**. Detalles definitorios:

- **No** hace falta saber *qué* es el objeto (puede ser cualquier cosa o persona).
- Se usa típicamente el enfoque **stream** (solo frames anteriores), para *tracking* de corto plazo.
- Es posible seguir **múltiples objetos** a la vez (MOT, *Multiple Object Tracking*).

Sus **desafíos**: la carga de cómputo (aplicaciones en tiempo real), los cambios de **apariencia** del objeto (dinámica, iluminación, punto de vista), la **interacción entre objetos** (oclusión, similitud con otros) y el **movimiento** (que se aborda estimando el flujo). El [flujo óptico](/fundamentos/flujo-optico) es la herramienta clave para modelar ese movimiento.

### 3.2 Reconocimiento de acciones

El **reconocimiento de acciones** analiza un video para identificar las **acciones o eventos** que ocurren en él. Es el corazón del campo —las acciones humanas son el contenido principal de la mayoría de los videos— y se desarrolla en detalle en el fundamento [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones). A veces depende del seguimiento de objetos.

---

## 4. Por qué importa

El análisis de video es importante por varias razones que la clase subraya:

- **Automatización.** La mayoría de las cámaras solo *graban*, con poca o ninguna automatización. Automatizar el análisis (detección de eventos, vigilancia inteligente) tiene enorme valor.
- **Explorar el mundo.** Robots, autos autónomos y agentes que deben detectar objetos y ubicarse dependen del análisis de video en tiempo real.
- **Mucho por resolver.** El análisis de video sigue siendo desafiante, con numerosos problemas abiertos.

Sus aplicaciones abarcan vigilancia, análisis deportivo, recuperación de video (buscar un clip por una consulta), navegación de contenido (encontrar un paso de una receta) y más.

---

## 5. Relevancia para salud y video clínico

El análisis de video tiene un campo de aplicación clínico en rápido crecimiento. El **seguimiento de objetos** se usa en el análisis de **marcha** (rastrear articulaciones a lo largo del tiempo), el seguimiento de instrumentos en **video quirúrgico** y el monitoreo de movimiento en camas de UCI. El **reconocimiento de acciones** habilita el análisis de **actividades de la vida diaria** (para asistencia a personas mayores), la fase de un **procedimiento quirúrgico** o de rehabilitación, y la detección de eventos anómalos (caídas, convulsiones). En todos, los dos desafíos centrales del campo —modelar el **movimiento** y manejar el **costo computacional** de secuencias largas— son restricciones de diseño reales, y la multimodalidad (video + señales fisiológicas) abre posibilidades que una sola modalidad no alcanza.

---

## Referencias

- Fundamentos relacionados: [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) · [Flujo óptico](/fundamentos/flujo-optico) · [Redes convolucionales](/fundamentos/redes-convolucionales).
- Dominio: [Video](/dominios/video).
