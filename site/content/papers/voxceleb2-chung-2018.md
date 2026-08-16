---
title: "VoxCeleb2 (2018)"
weight: 443
math: true
---

{{< paper-card
    title="VoxCeleb2: Deep Speaker Recognition"
    authors="Joon Son Chung, Arsha Nagrani, Andrew Zisserman (VGG, University of Oxford)"
    year="2018"
    venue="INTERSPEECH 2018 / arXiv:1806.05622"
    pdf="/papers/voxceleb2-chung-2018.pdf" >}}
La segunda iteración de [VoxCeleb](/papers/voxceleb-nagrani-2017), un orden de magnitud más grande: **más de un millón de enunciados de 6 112 hablantes** —contra los 153 516 de 1 251— recogidos con el mismo pipeline audiovisual automático. Dos decisiones de diseño le dan su valor como recurso de entrenamiento. La primera es la **diversidad**: 145 nacionalidades frente a las 36 de VoxCeleb1, lo que amplía sustancialmente la cobertura de acentos. La segunda es que la partición de desarrollo —**5 994 hablantes**— es **completamente disjunta** de VoxCeleb1, de modo que un modelo entrenado acá puede evaluarse allá sin contaminación. Esa combinación es la que la [Clase 41](/clases/clase-41) usa como protocolo: entrenar en VoxCeleb2, medir en VoxCeleb1.
{{< /paper-card >}}

---

## Contexto: para entrenar hacen falta hablantes, no horas

VoxCeleb1 resolvió el problema de tener habla real etiquetada a escala, pero 1 251 identidades siguen siendo pocas para entrenar una representación que deba generalizar a personas nunca vistas. La variable crítica en reconocimiento de hablante no es la cantidad de audio sino el **número de hablantes distintos**: es lo que determina cuántos ejemplos de "variación entre personas" ve el modelo, que es justamente lo que tiene que aprender a capturar.

## Composición

| | VoxCeleb1 | VoxCeleb2 |
|---|---|---|
| Personas | 1 251 | **6 112** |
| Videos | 22 496 | 150 480 |
| Enunciados | 153 516 | **1 128 246** |
| Nacionalidades | 36 | **145** |

Y la partición de VoxCeleb2:

| Partición | Personas | Enunciados |
|---|---|---|
| Desarrollo | 5 994 | 1 092 009 |
| Prueba | 118 | 36 237 |

El pipeline de recolección es el de VoxCeleb1 con ajustes: se buscan videos añadiendo la palabra *"interview"* al nombre de cada persona, se procesan los 100 primeros resultados, y el modelo de reconocimiento facial se amplía para tolerar **poses no frontales** —lo que aumenta el material aprovechable.

Las **etiquetas de nacionalidad** se obtienen automáticamente y son un aporte propio: permiten construir listas de evaluación controladas por acento, que es lo que hace posible el conjunto **VoxCeleb1-H** (*hard*), donde los pares a discriminar comparten género y nacionalidad. Ese es el escenario realmente difícil, porque elimina los dos atajos más obvios que un modelo puede tomar.

Junto al dataset, el paper propone una arquitectura basada en ResNet con *temporal average pooling*, entrenada con softmax y pérdida contrastiva, que fija el estado del arte del momento: **4,19 % de EER** en el test original de VoxCeleb1 con ResNet-50, y 7,33 % en VoxCeleb1-H.

## Limitaciones

- **Hereda los sesgos de VoxCeleb1**: celebridades, entrevistas, registro público. Más grande no significa más representativo del habla cotidiana.
- **Las etiquetas siguen siendo automáticas.** [Xie et al. (2019)](/papers/utterance-level-xie-2019) documentan errores en las listas de evaluación extendidas y publican versiones limpias.
- **Las etiquetas de nacionalidad son un proxy imperfecto del acento.** El paper lo asume explícitamente —la nacionalidad *"is often more indicative of accent"*— pero una persona puede tener nacionalidad de un país y acento de otro.
- **El desbalance por hablante persiste**: la cantidad de enunciados por persona varía mucho, lo que sesga el entrenamiento hacia los más representados.

## Por qué importa para la Clase 41

Aporta las dos mitades del protocolo experimental que la clase presenta:

**El conjunto de entrenamiento.** Los 5 994 hablantes de la partición de desarrollo son la fuente de variación entre personas con la que se aprende el descriptor. La slide los menciona por número exacto.

**La garantía de disyunción.** Que ninguna de las 1 251 identidades de VoxCeleb1 aparezca en VoxCeleb2-dev es lo que hace que la evaluación mida la **capacidad de generalizar a hablantes nuevos** en vez de la memorización. Sin esa propiedad, el argumento con que la clase abre —que un clasificador cerrado obliga a reentrenar por cada persona nueva, y que un descriptor no— no sería verificable experimentalmente.

Su baseline con temporal average pooling también funciona como punto de comparación: cuando [Xie et al.](/papers/utterance-level-xie-2019) cambian **solo la agregación**, manteniendo backbone y datos, bajan de 4,19 % a 3,22 % de EER con **menos parámetros** (10 millones contra 26). Ese contraste es el argumento central de la parte de speaker recognition de la clase.
