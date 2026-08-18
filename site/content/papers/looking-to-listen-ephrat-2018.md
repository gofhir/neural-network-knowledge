---
title: "Looking to Listen at the Cocktail Party (2018)"
weight: 467
math: true
---

{{< paper-card
    title="Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation"
    authors="Ariel Ephrat, Inbar Mosseri, Oran Lang, Tali Dekel, Kevin Wilson, Avinatan Hassidim, William T. Freeman, Michael Rubinstein (Google Research / Universidad Hebrea de Jerusalén)"
    year="2018"
    venue="SIGGRAPH 2018 / arXiv:1804.03619"
    arxiv="1804.03619"
    pdf="/papers/looking-to-listen-ephrat-2018.pdf" >}}
Aislar la voz de una persona en un video donde varias hablan a la vez, usando **su cara** como referencia. La contribución conceptual es que la modalidad visual no aporta información acústica: aporta **una estructura de la que el audio carece**. Al condicionar cada salida a un rostro detectado, el célebre *problema de la permutación* —que una red con varias salidas no sabe en qué orden ponerlas— simplemente deja de existir. Se entrenó sobre **AVSpeech**, un dataset construido filtrando 290 000 charlas de YouTube para conservar solo segmentos con un hablante visible y audio limpio, que luego se mezclan sintéticamente.
{{< /paper-card >}}

---

## El problema de la permutación, y cómo se disuelve

Separar dos voces de una grabación monocanal tiene una dificultad que no es de capacidad sino de **especificación**: si la red produce dos salidas, ¿cuál corresponde a cuál hablante? Cualquier asignación es arbitraria, y una pérdida que compare salida 1 con hablante 1 castiga a la red por una decisión que no tiene forma de tomar bien. Los gradientes de ejemplos distintos se cancelan.

Las soluciones conocidas hasta entonces atacaban el síntoma: *deep clustering* (aprender un espacio donde agrupar, que no tiene orden) o *permutation invariant training* (probar todas las asignaciones y quedarse con la mejor).

Este trabajo lo elimina en el planteo: **la salida $i$ es, por construcción, la voz de la persona cuyo rostro se pasó como entrada $i$**. No hay nada que ordenar.

{{< concept-alert type="clave" >}}
Vale insistir en el punto, porque es el que se transfiere: **la cara no dice nada sobre el timbre de la voz que el audio no contenga ya**. Lo que aporta es una **etiqueta estable** —esta señal pertenece a esta persona— que convierte un problema de agrupamiento sin supervisión en uno de regresión condicionada.

Es el mismo patrón que la [Clase 43](/clases/clase-43) muestra en la fusión audiovisual y que [SoundNet](/papers/soundnet-aytar-2016) usa para entrenar: la segunda modalidad no siempre agrega información, a veces agrega **estructura**.
{{< /concept-alert >}}

## Arquitectura

Dos flujos que convergen:

- **Visual**: se detectan los rostros, se siguen a lo largo del video, y se extrae un *embedding* facial por cuadro con una red preentrenada. Se usan *embeddings* faciales, no píxeles crudos ni landmarks de labios — lo que hace al modelo independiente del hablante.
- **Audio**: espectrograma de la mezcla, procesado con **convoluciones dilatadas**, que amplían el campo receptivo temporal sin multiplicar parámetros.

Ambos se concatenan y pasan por una **BiLSTM** y capas densas que producen una **máscara compleja** por hablante, la cual multiplica el espectrograma de la mezcla. La máscara compleja —no solo de magnitud— permite corregir también la fase.

## AVSpeech

El dataset es parte sustancial de la contribución. Se parte de 290 000 charlas y conferencias de YouTube y se filtran automáticamente los segmentos donde hay **exactamente un hablante visible y audio limpio**, resultando en miles de horas de material.

El detalle metodológico que conviene retener: los ejemplos de entrenamiento se fabrican **mezclando sintéticamente** esos segmentos limpios, porque así se conoce la separación verdadera. Es la práctica estándar del área y tiene un sesgo conocido — las mezclas artificiales no reproducen la reverberación de una sala real, el movimiento de las personas ni el efecto Lombard. Ver [Separación de Fuentes](/fundamentos/separacion-de-fuentes).

## Por qué importa para la Clase 44

Es la tercera de las siete aplicaciones de la clase, y la que tiene aplicaciones prácticas más inmediatas: audífonos que aíslan al interlocutor, videoconferencia, preprocesamiento para transcripción automática en ambientes ruidosos.

Y encaja en el hilo que une toda la clase: **la correspondencia audiovisual gratuita del video**, usada aquí para resolver un problema que el audio solo no puede resolver bien — no por falta de capacidad del modelo, sino por falta de una referencia que diga quién es quién.

---

**Ver también:** [Learning to Separate Object Sounds (2018)](/papers/separating-object-sounds-gao-2018) · [Speech2Face (2019)](/papers/speech2face-oh-2019) · [Separación de Fuentes](/fundamentos/separacion-de-fuentes) · [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual) · [Clase 43](/clases/clase-43)
