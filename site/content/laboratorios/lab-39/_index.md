---
title: "Lab 39 - CNN sobre onda cruda y VGGish preentrenado"
weight: 390
sidebar:
  open: true
---

**Profesores:** Juan Pablo de Vicente · Gabriel Sepúlveda · Álvaro Soto
**Módulo:** Audio — modelos convolucionales y transferencia
**Notebook origen:** `clase_39/material/Laboratorio/Practico_3_Audio_DINTA_alumnos_v3.ipynb`
**Notebook ejecutado:** [lab39.ipynb](/notebooks/lab39.ipynb) · [HTML](/notebooks-html/lab39.html)

## Encuadre

La contraparte práctica de la [clase 39](/clases/clase-39), y la aplicación directa de su segunda mitad: **qué pasa si se prescinde del espectrograma**. El lab tiene dos partes que responden lo mismo con presupuestos opuestos. La primera entrena desde cero la familia de redes **M** —M3, M5, M11, M18— de [Dai et al. (2016)](/papers/raw-waveforms-dai-2017) sobre la **onda cruda** de UrbanSound8K. La segunda toma **VGGish** de [Hershey et al. (2017)](/papers/vggish-hershey-2017), preentrenado sobre 70 millones de videos de YouTube, y le hace fine-tuning de una sola capa lineal.

El práctico pide cinco actividades. Ninguna de ellas es lo que hace interesante al laboratorio.

Lo que lo hace interesante es que **tres defectos del notebook conspiran para producir números que parecen correctos**: una fuga de datos que mete el conjunto de test completo dentro del de entrenamiento, un learning rate que invierte el orden entre arquitecturas, y un `ReLU` mal ubicado que deja tres de las diez clases estructuralmente inalcanzables. Los tres se detectan midiendo, y los tres se corrigen en una línea.

El caso más instructivo es el primero. Con el bug de fuga activo, M3 obtiene **56.13 %** contra los **56.12 %** que reporta el paper — coincidencia hasta la segunda cifra decimal. Esa coincidencia no validaba nada: era **la cancelación de dos sesgos de signo opuesto**, +11.00 puntos por la fuga y −10.99 por el preprocesamiento. Las dos señales que se usan habitualmente para dar por bueno un experimento —"el resultado calza con la referencia" y "train y test van parejos"— apuntaban ambas en la dirección equivocada.

![Descomposición del rendimiento: con fuga, sin fuga y contra el paper](/laboratorios/lab-39/fuga-de-datos.jpg)

## Resultados consolidados (medidos en el notebook)

### Parte 1 — la familia M sobre onda cruda

| Modelo | Params | Campo receptivo | Test (notebook) | Test (split limpio) | Paper |
|---|---|---|---|---|---|
| M3 | 221 194 | **19.5 ms** | 56.13 % | **45.13 %** | 56.12 % |
| M5 | 559 114 | **200.9 ms** | 76.63 % | **52.12 %** | 63.42 % |
| M18 | 3 683 786 | **1358.3 ms** | 60.82 % | — | 71.68 % |
| M3 preentrenado 50 épocas (del curso) | 221 194 | 19.5 ms | 68.61 % | — | — |

La ganancia por profundidad M3 → M5 aparece como **+20.50 puntos** con el split roto y como **+6.99** con el split corregido, contra los **+7.30** del paper: una diferencia de 0.31 puntos. El efecto arquitectónico se reproduce con precisión una vez eliminada la fuga; con el bug activo aparecía inflado casi tres veces.

### El learning rate decide el orden entre arquitecturas

| Modelo | capas | `lr = 0.01` (notebook) | `lr = 0.001` (default de Adam) | ganancia |
|---|---|---|---|---|
| M5 | 5 | 76.63 % | 81.10 % | **+4.47** |
| M18 | 18 | 60.82 % | **83.85 %** | **+23.02** |

Con el learning rate del notebook, M18 rinde **15.8 puntos peor** que M5, contradiciendo al paper. Con el default de Adam vuelve a ser el mejor modelo de la Parte 1. La conclusión "la profundidad no ayuda" que sale del notebook tal cual es el artefacto de un hiperparámetro, y el control con M5 —que gana 5.2 veces menos con el mismo cambio— es lo que permite atribuir el efecto a la profundidad.

### Parte 2 — VGGish y el `ReLU` sobre los logits

| Salida | `lr` | Test (por parche) | Clases muertas | Logit mínimo | % logits en cero |
|---|---|---|---|---|---|
| `ReLU` | 0.01 | 64.53 % | **3** | 0.000 | **84.9 %** |
| `ReLU` | 0.001 | 71.71 % | 1 | 0.000 | 71.8 % |
| **`Identity`** | **0.01** | **91.45 %** | **0** | −38.36 | 0 % |
| `Identity` | 0.001 | 86.52 % | 0 | −20.60 | 0 % |

Al reemplazar `net.fc[-2]` por la capa de 10 clases, el `ReLU` de `fc[-1]` queda aplicado **sobre los logits**. El 84.9 % de ellos quedaba saturado en cero —8.5 de cada 10 por ejemplo— y tres clases (`children_playing`, `drilling`, `gun_shot`, el 26.9 % del test) nunca se predecían. Borrar esa línea vale **+26.92 puntos**.

### El veredicto del transfer learning

| Estrategia de agregación | Accuracy |
|---|---|
| por parche de 0.96 s | 91.45 % |
| voto mayoritario | 93.47 % |
| suma de logits | 96.91 % |
| **promedio de softmax** | **97.14 %** |

Y la comparación final, todos con folds disjuntos y métrica por clip:

| Modelo | Test | Parámetros entrenados | Épocas |
|---|---|---|---|
| M3 desde cero | 45.13 % | 221 194 | 20 |
| M5 desde cero | 52.12 % | 559 114 | 20 |
| **VGGish fine-tuning** | **84.65 %** | **40 970** | **3** |

VGGish supera a M5 por **32.53 puntos** entrenando **14 veces menos parámetros** durante 3 épocas en lugar de 20. El 97.14 % de la tabla anterior no era un resultado: implicaba fallar en 25 de 873 clips, por encima de lo publicado para este dataset con evaluación honesta. El 84.65 % sí lo es.

## Bloques del lab

{{< cards >}}
  {{< card link="01-la-fuga-de-folds" title="La fuga de folds" subtitle="Los corchetes que glob interpreta como clase de caracteres, el 100 % del test dentro del train, y por qué el sesgo escala con la capacidad del modelo en lugar de ser constante" icon="beaker" >}}
  {{< card link="02-el-preprocesamiento" title="El preprocesamiento y sus once puntos" subtitle="Diez frecuencias de muestreo decimadas por igual, el aliasing de [::5], el recorte central que se come el 58 % de algunos clips y el déficit constante contra el paper" icon="adjustments" >}}
  {{< card link="03-familia-m-y-campo-receptivo" title="La familia M y el campo receptivo" subtitle="De 19.5 ms a 1358 ms capa a capa, el global average pooling que diluye los eventos breves, y las clases sumidero que se mudan en vez de desaparecer" icon="chart-bar" >}}
  {{< card link="04-learning-rate-y-profundidad" title="El learning rate decide el orden" subtitle="M18 pierde contra M5 por un hiperparámetro. El control que separa profundidad de preferencia general, y la interacción que invierte el signo del efecto" icon="trending-down" >}}
  {{< card link="05-el-relu-sobre-los-logits" title="El ReLU sobre los logits" subtitle="Tres clases muertas, el 84.9 % de los logits en cero y los 38 puntos de rango dinámico truncado. Un factorial 2x2 para separar la causa del disparador" icon="code" >}}
  {{< card link="06-agregacion-y-transfer-learning" title="Agregación y transfer learning" subtitle="Por qué el voto mayoritario pierde contra la suma de logits, el 97 % que no era creíble, y la magnitud real de traer AudioSet a un dataset de 8732 clips" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/clasificacion-de-audio" title="Clasificación de audio" subtitle="Etiqueta global contra local, audio tagging y detección de eventos" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de audio" subtitle="Onda cruda, espectrograma y qué se gana y se pierde en cada una" icon="book-open" >}}
  {{< card link="/fundamentos/mfcc-y-escala-mel" title="MFCC y escala mel" subtitle="Por qué VGGish usa log-mel y no MFCC: la DCT que destruye la estructura local" icon="book-open" >}}
  {{< card link="/fundamentos/datasets-de-audio" title="Datasets de audio" subtitle="UrbanSound8K, AudioSet y la razón de ser de los folds oficiales" icon="book-open" >}}
  {{< card link="/fundamentos/digitalizacion-de-audio" title="Digitalización de audio" subtitle="Muestreo, Nyquist y el aliasing que produce decimar sin filtrar" icon="book-open" >}}
  {{< card link="/fundamentos/transfer-learning" title="Transfer learning" subtitle="Feature extraction contra fine-tuning, y por qué hay que reemplazar la capa final" icon="book-open" >}}
{{< /cards >}}

## Papers de este laboratorio

{{< cards >}}
  {{< card link="/papers/raw-waveforms-dai-2017" title="Redes M (2016)" subtitle="Dai et al. — las cinco arquitecturas del lab, el kernel de 80 como banco de filtros y el 71.68 % de M18 sobre onda cruda" icon="document-text" >}}
  {{< card link="/papers/vggish-hershey-2017" title="VGGish (2017)" subtitle="Hershey et al. — 70 millones de videos, los parches log-mel de 96x64 y el embedding de 128 dimensiones que este lab reutiliza" icon="document-text" >}}
  {{< card link="/papers/urbansound8k-salamon-2014" title="UrbanSound8K (2014)" subtitle="Salamon et al. — el dataset del lab: las 10 clases, los folds oficiales y por qué existen" icon="document-text" >}}
  {{< card link="/papers/audioset-gemmeke-2017" title="AudioSet (2017)" subtitle="Gemmeke et al. — la ontología de 527 clases sobre la que se preentrenó VGGish; incluye Gunshot y Drilling" icon="document-text" >}}
  {{< card link="/papers/dl-audio-purwins-2019" title="DL for Audio Signal Processing (2019)" subtitle="Purwins et al. — el panorama que sitúa onda cruda contra espectrograma" icon="document-text" >}}
  {{< card link="/papers/cldnn-sainath-2015" title="CLDNN (2015)" subtitle="Sainath et al. — la receta CNN+RNN+MLP de la clase, el contrapunto arquitectónico a las redes M" icon="document-text" >}}
  {{< card link="/papers/wavenet-oord-2016" title="WaveNet (2016)" subtitle="van den Oord et al. — la otra respuesta al campo receptivo sobre onda cruda: convoluciones dilatadas" icon="document-text" >}}
  {{< card link="/papers/ast-gong-2021" title="AST (2021)" subtitle="Gong et al. — el sucesor de VGGish en AudioSet, y la refutación de la coda de la clase" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Clase 39 - Teoría](/clases/clase-39/teoria) · [Clase 39 - Profundización](/clases/clase-39/profundizacion) · [Clase 39 - Práctica](/clases/clase-39/practica) · [Lab 37 - Datasets y Herramientas para Audio](/laboratorios/lab-37) (el lab anterior de audio, donde aparecen las *collate functions* que esta Parte 2 reutiliza) · [Lab 35 - Análisis de Audio](/laboratorios/lab-35) (Fourier, Nyquist y el muestreo que explica el aliasing de este lab) · Dominio [Audio](/dominios/audio).
