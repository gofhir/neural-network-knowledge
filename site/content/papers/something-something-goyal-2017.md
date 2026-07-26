---
title: "Something-Something: sentido común temporal (2017)"
weight: 399
math: true
---

{{< paper-card
    title="The 'Something Something' Video Database for Learning and Evaluating Visual Common Sense"
    authors="Raghav Goyal et al. (TwentyBN)"
    year="2017"
    venue="ICCV 2017 / arXiv:1706.04261"
    pdf="/papers/something-something-goyal-2017.pdf" >}}
**Something-Something** es un dataset de video diseñado para forzar el modelado del **tiempo**. En lugar de etiquetas atómicas usa **plantillas de acción con ranuras** —"Putting [something] onto [something]", "Dropping [something] into [something]"— donde cada `[something]` es un objeto que el propio trabajador elige. Como la misma plantilla se instancia con miles de objetos distintos, **el objeto deja de ser informativo** y solo el patrón de movimiento distingue una clase de otra: el modelo se ve **obligado a modelar la dinámica temporal**. La versión descrita tiene **108.499 videos** (de 2 a 6 s, promedio 4,03 s) en **174 clases**. Es la prueba experimental de que el tiempo importa, y expone la debilidad de cualquier modelo que lo ignore —el argumento central del [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) en la [Clase 36](/clases/clase-36).
{{< /paper-card >}}

---

## Contexto: el atajo de la apariencia

La visión por computador despegó con **ImageNet** y la clasificación de objetos en imágenes fijas, pero las redes entrenadas sobre imágenes estáticas nunca observan cómo **cambia** un objeto —su pose, posición, distancia—, y esos cambios son los que revelan propiedades físicas: geometría 3D, rigidez, deformabilidad, *affordances*, gravedad, permanencia del objeto. Ese "sentido común físico" es lo que las imágenes fijas codifican, en el mejor de los casos, solo de forma indirecta.

El paper hace una crítica precisa al paradigma de reconocimiento de acciones de la época: en muchos casos **la etiqueta puede predecirse a partir de un solo frame** recortado del video. Una cancha de tenis, un ring, una piscina: el fondo y los objetos delatan la acción. Se obtiene buen rendimiento por **agregación frame a frame** de features de una CNN pre-entrenada en ImageNet, sin que la red aprenda nada sobre el **movimiento**. Esto es el **atajo de la apariencia**: cuando el dataset permite resolver la tarea mirando la escena estática, el modelo aprende a mirar la escena estática y el eje temporal se vuelve prescindible. Es un caso de **sesgo de dataset** (*dataset bias*): la red "hace trampa" explotando correlaciones espurias fáciles de leer.

## Composición: plantillas y crowdsourcing invertido

En lugar de recolectar videos existentes y etiquetarlos, los autores **invierten el proceso** (inspirados en Charades / *Hollywood in Homes*): dan al trabajador una plantilla de acción y le piden que **grabe él mismo** un clip que la ejecute, eligiendo los objetos. El corazón conceptual son las **plantillas con placeholders**: una clase no es "poner una taza sobre una mesa", sino la plantilla abstracta **"Putting [something] onto [something]"**. Como la etiqueta es **agnóstica al objeto**, lo único que distingue esa clase de "Putting [something] next to [something]" es el **patrón de movimiento y la relación espacial que evoluciona en el tiempo**.

La ingeniería anti-atajo más deliberada son los **grupos de acción**: conjuntos de acciones muy similares con diferencias visuales mínimas, que solo un entendimiento fino de la actividad permite separar. Por ejemplo:

- "Putting something on top of / next to / behind something".
- "Poking something so lightly that it does not move / so it slightly moves / so that it falls over".

Algunos grupos incluyen **acciones fingidas** ("Poking something" vs. "Pretending to poke something"): distinguir la genuina de la simulada obliga a **observar de cerca el objeto** —si realmente se movió— en vez de mirar la posición de la mano. Dar los grupos al trabajador lo incentiva a ejecutar las variantes **con el mismo objeto**, anulando aún más el objeto como pista.

**Estadísticas.** 108.499 videos, 174 clases, duración media 4,03 s, ~620 videos por clase (mín. 77, máx. 986), **23.137 nombres de objeto distintos** aportados por **1.133 trabajadores**. Los splits train/val/test (8:1:1) se construyen de modo que **todos los videos de un mismo trabajador queden en un solo split**, evitando que el modelo memorice su estilo, fondo u objetos y los reencuentre en test.

## Impacto: por qué exige modelado temporal

Los baselines miden la dificultad con varios codificadores: **2D-CNN + Avg** (VGG-16 promediando features de todos los frames, lo que **destruye el orden temporal**), **Pre-2D-CNN + LSTM**, **3D-CNN + Stack** (convoluciona sobre el eje temporal) y combinaciones. El hallazgo central es explícito: las redes **3D-convolucionales superan en general a las 2D**, y su combinación funciona mejor todavía. Es la evidencia empírica de la tesis del dataset: **cuando la apariencia no basta, el modelo que procesa el tiempo gana**. Una CNN 2D con promedio de frames es una "bolsa de fotogramas" que no distingue "levantar" de "dejar caer", porque ambas contienen los mismos frames en orden inverso.

Cuantitativamente, sobre **10 clases** los errores top-1 van de ~44,9% (2D+3D-CNN) a ~76,5% (2D-CNN + Avg desde cero); sobre las **174 clases** completas el error es **top-1: 88,5%** y **top-5: 70,3%**. La dificultad **crece al aumentar el número de clases** pese a crecer también el conjunto de entrenamiento, y muchas de las distinciones sutiles elegidas para endurecer la tarea apenas son separables con arquitecturas estándar —un problema "extraordinariamente difícil" que reclama modelos más sofisticados.

## Limitaciones

- **Etiquetas ambiguas.** Describir acciones físicas finas con lenguaje es intrínsecamente ambiguo; se mitiga con top-K y agrupación de clases, no se elimina.
- **Señal de aprendizaje débil.** Las descripciones en lenguaje natural aportan supervisión más débil que las etiquetas one-hot; esta versión se restringe a captions simples (verbos, sustantivos, preposiciones).
- **Trampa residual.** Aunque los grupos y las acciones fingidas bloquean atajos, las redes pueden seguir explotando pistas indirectas: el sesgo de dataset se mitiga, no desaparece.
- **Sesgo del setup de grabación.** Videos grabados a propósito (primeros planos de manos sobre una mesa): dominio visual acotado que difiere del video "en el mundo real".
- **Colección en curso.** El paper describe un esfuerzo evolutivo ("como enseñar a un niño de un año"); versiones posteriores expandirían escala y complejidad.

## Por qué importa para la Clase 36

La [Clase 36](/clases/clase-36) (Introduction to Video Analysis) abre preguntando qué añade el tiempo respecto de la imagen fija, y por qué una CNN 2D no basta. Something-Something es el **caso de estudio canónico** para responder:

- **Diagnostica el atajo de la apariencia.** Datasets donde el fondo o el objeto delatan la acción permiten que una CNN 2D sobre frames aislados ya rinda bien. Las plantillas agnósticas al objeto neutralizan ese atajo: **solo el movimiento** distingue una clase de otra.
- **Justifica el salto de la 2D a la 3D-CNN.** El resultado 3D-CNN > 2D-CNN es el argumento empírico para introducir las arquitecturas espacio-temporales (convoluciones 3D, redes de dos flujos, recurrentes) frente al *frame-wise pooling* del [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).
- **Define qué es "razonamiento temporal".** Distinguir "levantar" de "dejar caer", "empujar levemente" de "empujar hasta que cae", o una acción real de una fingida, exige entender **causalidad y orden** en la secuencia —el "sentido común visual" que da nombre al dataset.

Para el video clínico la lección es directa: en un procedimiento quirúrgico la diferencia entre una maniobra correcta y una incorrecta suele estar en la **secuencia** de pasos (irrigar antes de suturar, retirar el instrumento antes de cerrar), no en la apariencia de un fotograma aislado. Un sistema ciego al tiempo reproduciría exactamente el fracaso que Something-Something fue diseñado para exponer: acertar lo trivial y errar donde el orden temporal es lo que importa clínicamente.
