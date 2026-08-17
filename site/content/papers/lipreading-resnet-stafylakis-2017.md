---
title: "ResNet + LSTM para lectura de labios (2017)"
weight: 461
math: true
---

{{< paper-card
    title="Combining Residual Networks with LSTMs for Lipreading"
    authors="Themos Stafylakis, Georgios Tzimiropoulos (University of Nottingham)"
    year="2017"
    venue="Interspeech 2017 / arXiv:1703.04105"
    arxiv="1703.04105"
    pdf="/papers/lipreading-resnet-stafylakis-2017.pdf" >}}
La arquitectura de tres bloques que se volvió estándar en lectura de labios: **frente convolucional espacio-temporal 3D**, **ResNet-34 aplicada a cada paso temporal**, y **BiLSTM** como *back-end*, entrenados conjuntamente end-to-end. Alcanza **83,0 % de exactitud de palabra en LRW**, una mejora absoluta de **6,8 puntos** sobre el estado del arte previo. Es la referencia directa del flujo visual de [E2E-AVSR](/papers/e2e-avsr-petridis-2018) —el paper de la Clase 43 la cita como `[13]` y adopta su diseño— y, notablemente, la única línea de su tabla de resultados que E2E-AVSR **no logra superar**.
{{< /paper-card >}}

---

## La arquitectura

Tres bloques que resuelven tres problemas distintos:

**1. Frente espacio-temporal.** Una capa convolucional con **64 núcleos 3D**, seguida de max-pooling que reduce el tamaño espacial. Su función es capturar la dinámica de **corto plazo** del movimiento de la boca — la apertura y cierre de los labios ocurre en decenas de milisegundos y esa información está en la relación entre fotogramas contiguos, no en ninguno por separado.

**2. Backbone espacial.** Una **ResNet-34** en su versión de mapeo identidad, aplicada de forma independiente a cada paso temporal, que colapsa progresivamente la dimensión espacial hasta un vector por fotograma.

**3. Back-end temporal.** Una **BiLSTM** que modela la dinámica de largo plazo. El softmax se aplica a todos los pasos y las pérdidas por instante se agregan; el sistema completo se entrena conjuntamente.

{{< concept-alert type="clave" >}}
La contribución de diseño es la **separación de escalas temporales**: el frente 3D se ocupa de los milisegundos, la ResNet de la apariencia por fotograma, y la BiLSTM de la palabra completa. Los autores encuentran —y E2E-AVSR lo confirma— que el frente 3D **sigue aportando aunque haya un recurrente detrás**, que es el resultado contraintuitivo: si la LSTM ya modela el tiempo, ¿para qué convolucionar en el tiempo? Porque operan en escalas distintas y la recurrente sola no captura bien la dinámica de milisegundos.
{{< /concept-alert >}}

## Resultado

**83,0 % en LRW**, contra el 76,2 % de Chung y Zisserman (2016), la referencia previa: **+6,8 puntos absolutos**. Y lo consigue sin usar información de audio en ninguna etapa.

## Por qué importa para la Clase 43

Es el eslabón que la [Clase 43](/clases/clase-43) usa sin nombrar. El flujo visual de [E2E-AVSR](/papers/e2e-avsr-petridis-2018) —convolución espacio-temporal, ResNet-34, recurrente bidireccional de 2 capas— **es esta arquitectura**, con GRU en lugar de LSTM, y su procedimiento de entrenamiento en tres pasos también viene de aquí.

Y aparece en la tabla de resultados de la clase como `V [13]* — 83,0`, una línea por encima del `V (End-to-End) — 82,0` del propio sistema. La nota al pie explica la diferencia: este trabajo extrae el recorte de boca **siguiendo puntos faciales**, mientras que E2E-AVSR usa una **caja fija de 96×96** para todos los videos. Un punto de exactitud que se pierde en el preprocesamiento, no en la arquitectura — un recordatorio de que en visión la etapa previa al modelo suele valer tanto como el modelo.

---

**Ver también:** [E2E-AVSR (2018)](/papers/e2e-avsr-petridis-2018) · [LipNet (2016)](/papers/lipnet-assael-2016) · [AV-HuBERT (2022)](/papers/av-hubert-shi-2022) · [Lectura de Labios](/fundamentos/lectura-de-labios) · [ResNet (2015)](/papers/resnet-he-2015) · [LSTM y GRU](/fundamentos/lstm-gru)
