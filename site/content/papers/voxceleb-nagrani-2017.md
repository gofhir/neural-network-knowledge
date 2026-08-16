---
title: "VoxCeleb (2017)"
weight: 442
math: true
---

{{< paper-card
    title="VoxCeleb: a large-scale speaker identification dataset"
    authors="Arsha Nagrani, Joon Son Chung, Andrew Zisserman (VGG, University of Oxford)"
    year="2017"
    venue="INTERSPEECH 2017 / arXiv:1706.08612"
    pdf="/papers/voxceleb-nagrani-2017.pdf" >}}
El dataset que sacó al reconocimiento de hablante del laboratorio. Hasta 2017 los corpus disponibles se grababan en condiciones controladas —habla leída, micrófono cercano, salas acústicamente tratadas— y se anotaban a mano, lo que los mantenía chicos. VoxCeleb reúne **más de 100 000 enunciados de 1 251 celebridades** extraídos de entrevistas subidas a YouTube: ruido de fondo, risas, superposición de voces, calidad de grabación variable y acentos de todo el mundo. La clave metodológica es que la recolección es **completamente automática**, y curiosamente el que hace el trabajo pesado no es un sistema de audio sino uno de **visión por computador**: se detectan y siguen caras, se verifica con un modelo audiovisual que la persona en pantalla es la que habla, y se confirma su identidad comparando contra imágenes de referencia. Es el conjunto de prueba sobre el que se evalúa el modelo de la [Clase 41](/clases/clase-41).
{{< /paper-card >}}

---

## Contexto: el problema de la etiqueta

Reconocer hablantes necesita, por definición, saber quién habla. Y eso es caro: alguien tiene que escuchar el audio y anotarlo. Esa restricción mantenía a los datasets del área en un régimen incompatible con el aprendizaje profundo — pocos hablantes, pocas horas, y condiciones de grabación tan controladas que los modelos entrenados en ellos se degradaban al salir al mundo.

La comparación que el paper presenta es elocuente: los corpus previos venían de discursos oficiales, llamadas telefónicas o laboratorios de acústica. VoxCeleb aporta habla **"in the wild"**, con todo lo que eso implica de ruido y variabilidad, y a una escala dos órdenes de magnitud mayor.

## Método: un pipeline de visión para etiquetar audio

La idea que hace posible la escala es aprovechar que en un video de entrevista **la cara y la voz están sincronizadas**. El pipeline encadena:

1. **Descarga** de videos de entrevistas de YouTube para una lista de personas de interés.
2. **Detección y seguimiento de caras**, agrupando detecciones en *face tracks* con un tracker por posición.
3. **Verificación de hablante activo** con SyncNet, una CNN de dos corrientes que estima la correlación entre el movimiento de los labios y la señal de audio. Este es el paso decisivo: descarta los segmentos donde la persona aparece en pantalla pero **no** es quien habla — voces en off, entrevistador fuera de cuadro, doblajes.
4. **Verificación de identidad** comparando la cara contra imágenes de referencia de esa celebridad.
5. **Filtrado** de los segmentos que no superan los umbrales.

Ningún humano escucha nada. El costo de anotación pasa de horas-persona a horas-GPU.

## Composición

| | VoxCeleb |
|---|---|
| Personas | 1 251 |
| Enunciados | 153 516 |
| Videos | 22 496 |
| Enunciados por persona (máx / mediana / mín) | 250 / 123 / 45 |
| Duración de los enunciados en segundos (máx / mediana / mín) | 145,0 / 8,2 / 4,0 |
| Nacionalidades | 36 |

El género está aproximadamente balanceado, y las nacionalidades cubren un rango amplio de acentos — un punto que importa porque el acento es una variable de confusión clásica en verificación de hablante.

El dataset soporta **dos tareas**: identificación (clasificación entre las 1 251 personas) y verificación (decidir si dos enunciados vienen del mismo hablante, con listas de pares provistas). La segunda es la que se usa como benchmark, porque no presupone un conjunto cerrado de identidades.

## Limitaciones

- **Sesgo de celebridad.** Son personas entrevistadas en televisión: hablan en registro público, con dicción cuidada, casi siempre en estudios. No es representativo de conversación telefónica, habla espontánea o entornos verdaderamente adversos.
- **Las etiquetas son automáticas y por lo tanto ruidosas.** Los propios autores de [Xie et al. (2019)](/papers/utterance-level-xie-2019) encontraron errores en las listas de verificación extendidas y publicaron versiones corregidas.
- **La verificación audiovisual sesga hacia caras visibles y bien iluminadas**, lo que descarta sistemáticamente cierto tipo de material.
- **Enunciados relativamente largos** (mediana de 8,2 s). El propio Xie et al. muestran que el rendimiento depende fuertemente de la duración: con 2 s el EER casi se duplica respecto de 6 s.

## Por qué importa para la Clase 41

La clase menciona VoxCeleb1 y VoxCeleb2 en una sola diapositiva —*"Model trained on VoxCeleb2 (5994 speakers), tested on VoxCeleb1, VoxCeleb1 and VoxCeleb2 are completely disjoint!"*— y ese signo de exclamación merece desarrollo.

Que los conjuntos sean **disjuntos en identidades** es lo que convierte la evaluación en una prueba de la representación y no de la memorización. El modelo nunca vio a ninguna de las personas del test: no puede haber aprendido "esta voz es de Fulano", solo "estas dos voces son de la misma persona". Es la diferencia entre un clasificador cerrado y un **descriptor** — que es exactamente el argumento con el que la clase abre su parte de speaker recognition, cuando descarta el enfoque de clasificador porque *"our model must be trained entirely for each new speaker"*.

La familia VoxCeleb, junto con la métrica EER y la curva ROC, forma el marco de evaluación estándar del área. Ver [Reconocimiento de hablante](/fundamentos/reconocimiento-de-hablante).
