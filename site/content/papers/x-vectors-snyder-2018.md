---
title: "x-vectors (2018)"
weight: 444
math: true
---

{{< paper-card
    title="X-Vectors: Robust DNN Embeddings for Speaker Recognition"
    authors="David Snyder, Daniel Garcia-Romero, Gregory Sell, Daniel Povey, Sanjeev Khudanpur (Johns Hopkins University)"
    year="2018"
    venue="ICASSP 2018"
    pdf="/papers/x-vectors-snyder-2018.pdf" >}}
El sistema que se volvió la línea base obligatoria del reconocimiento de hablante moderno, y el nombre con el que se conoce a toda una familia de descriptores neuronales. Una red **TDNN** entrenada para discriminar entre hablantes mapea enunciados de longitud variable a vectores de dimensión fija —los **x-vectors**— mediante una capa de **statistics pooling** que agrega media y desviación estándar sobre el tiempo. El aporte que le da el nombre al paper es otro: mostrar que la **aumentación de datos** con ruido y reverberación es la palanca más barata para hacerlos robustos. Comparados contra i-vectors sobre Speakers in the Wild y NIST SRE 2016, los x-vectors aprovechan mucho mejor los datos de entrenamiento a gran escala. Es el enfoque contra el que compite el modelo de la [Clase 41](/clases/clase-41).
{{< /paper-card >}}

---

## Contexto: del modelo generativo al discriminativo

Durante una década, el estado del arte en reconocimiento de hablante fueron los **i-vectors**: un enfoque generativo que modela la distribución de features acústicos con un GMM universal y extrae un vector de variabilidad total, seguido de PLDA para el scoring. Funcionaba bien y no necesitaba grandes cantidades de datos etiquetados.

El problema de los i-vectors es que no escalan con los datos: al ser un modelo generativo con supuestos fuertes, agregar cien veces más habla no mejora proporcionalmente. Las redes neuronales entrenadas **discriminativamente** —para separar hablantes— tienen la propiedad opuesta: piden muchos datos y los aprovechan.

## Método: TDNN, statistics pooling, y ruido a propósito

**La arquitectura.** Una red de retardo temporal (TDNN) procesa los features acústicos frame a frame, con capas cuyo campo receptivo temporal crece con la profundidad. Es una convolución 1D sobre el tiempo, en la práctica.

**El pooling de estadísticas.** El paso crítico, porque es donde una secuencia de largo variable se vuelve un vector fijo. En vez de promediar los frames, se concatenan **media y desviación estándar** de las activaciones a lo largo del tiempo:

$$\text{stats}(h_{1:T}) = \big[\;\mu(h_{1:T})\;\|\;\sigma(h_{1:T})\;\big]$$

La desviación estándar es el aporte respecto del promedio simple: captura cuánto **varía** la voz a lo largo del enunciado, no solo dónde está su centro. Sobre esa representación van un par de capas totalmente conectadas, y el x-vector se extrae de una de ellas.

**La aumentación.** Aquí está el título del paper. Etiquetar más hablantes es caro; **degradar** los que ya se tienen es gratis. Se generan copias de cada enunciado con ruido aditivo (murmullo de fondo, música, ruido ambiente) y **reverberación** simulada con respuestas impulsivas de salas. El efecto es doble: multiplica el volumen de datos y fuerza al modelo a representar al hablante y no al canal de grabación.

Un hallazgo matizado que el paper reporta con honestidad: la aumentación ayuda de forma clara en el entrenamiento del **backend PLDA**, y su beneficio en el entrenamiento de la red misma es más dependiente de la configuración.

## Resultados

Sobre Speakers in the Wild y NIST SRE 2016 Cantonese, los x-vectors superan a los i-vectors, y la ventaja **crece con la cantidad de datos de entrenamiento** — que es la propiedad estructural que interesa: el enfoque neuronal escala donde el generativo satura.

## Limitaciones

- **La statistics pooling trata todos los frames por igual.** Media y desviación estándar sobre la secuencia completa incluyen silencios, ruido y voces de fondo con el mismo peso que la voz del hablante. Es el hueco que atacan primero el *attentive pooling* y después la agregación por diccionario de [NetVLAD](/papers/netvlad-arandjelovic-2016) y GhostVLAD.
- **Necesita un backend separado.** El x-vector se extrae de la red, pero el scoring lo hace un PLDA entrenado aparte: el sistema no es end-to-end y hay dos objetivos distintos en juego.
- **La TDNN tiene campo receptivo limitado**, lo que restringe el contexto temporal disponible frente a arquitecturas más profundas.
- **La aumentación simula degradaciones**; el ruido real puede tener estructura que las respuestas impulsivas y los bancos de ruido no reproducen.

## Por qué importa para la Clase 41

La clase no menciona a los x-vectors, pero están presentes en la tabla de resultados del paper que sí presenta. En la comparación de [Xie et al. (2019)](/papers/utterance-level-xie-2019), las filas de Okabe et al. corresponden a *"TDNN (x-vector)"* con tres variantes de agregación: TAP 4,70 %, SAP 4,19 % y **ASP 3,85 %** de EER — esta última, *attentive statistics pooling*, era el estado del arte previo sobre VoxCeleb1.

Su valor conceptual para la clase es que ilustra el **paso intermedio** entre los dos extremos que el material contrapone. La clase plantea la disyuntiva "clasificador cerrado contra descriptor comparable por similitud"; los x-vectors ya son un descriptor —esa parte estaba resuelta desde 2018— y lo que queda por resolver es **cómo se agregan los frames**. Ahí es donde entra VLAD:

| Agregación | Qué calcula | EER en VoxCeleb1 |
|---|---|---|
| Temporal average pooling | media sobre el tiempo | 4,19 – 10,48 % |
| Statistics pooling (x-vector) | media + desviación estándar | ~4,70 % |
| Attentive statistics pooling | media + std ponderadas por atención | 3,85 % |
| **GhostVLAD** | residuos respecto de un diccionario aprendido, con clusters sumidero | **3,22 %** |

La progresión es una sola idea llevada cada vez más lejos: **cuánta estructura se le permite al paso de agregación**. Ver [Reconocimiento de hablante](/fundamentos/reconocimiento-de-hablante) y [Agregación VLAD](/fundamentos/agregacion-vlad).
