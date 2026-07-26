---
title: "MusicNet: música clásica etiquetada nota a nota (2017)"
weight: 418
math: true
---

{{< paper-card
    title="Learning Features of Music from Scratch"
    authors="John Thickstun, Zaid Harchaoui, Sham Kakade (UW)"
    year="2017"
    venue="ICLR 2017 / arXiv:1611.09827"
    pdf="/papers/musicnet-thickstun-2017.pdf" >}}
MusicNet es un dataset público de música clásica de licencia libre con etiquetas alineadas **nota a nota**: **330 grabaciones** de 10 compositores y 11 instrumentos, con **más de 1 millón de etiquetas temporales** (exactamente **1.299.329**) sobre **34 horas** de interpretaciones de cámara. Cada nota lleva instrumento, altura y tiempos de inicio y fin, obtenidos sin anotación manual masiva sino por **alineación audio-partitura**. Sobre ese corpus el paper define la tarea de *note prediction* (clasificación multi-etiqueta) y compara features de espectrograma contra redes *end-to-end* sobre la waveform. El hallazgo que da título al trabajo: las redes entrenadas para predecir notas **aprenden por sí solas filtros selectivos en frecuencia** —sinusoides moduladas tipo Fourier—, redescubriendo el análisis espectral desde cero. Para la [Clase 37](/clases/clase-37) es el caso canónico de cómo se **fabrica supervisión** en audio y del debate entre features hechas a mano y aprendidas.
{{< /paper-card >}}

---

## Contexto: la escasez de datasets de música con etiquetas densas

En 2016 la informática musical ya usaba aprendizaje automático para recomendación y generación, pero **no existía un dataset público a gran escala con etiquetas** para predecir notas en música clásica. Los recursos disponibles eran diminutos: el *MIREX MultiF0 Development Set* y *Bach10* juntos suman **menos de 7 minutos** de música etiquetada, y estaban diseñados para **evaluar** métodos, no para **entrenarlos**. El paper contrasta con la visión por computador, donde **ImageNet** habilitó arquitecturas end-to-end cuyas representaciones aprendidas superaron a las features hechas a mano. Anotar música nota a nota a mano es prohibitivo: una grabación de cámara de seis minutos contiene miles de notas solapadas (polifonía) de varios instrumentos, cada una con inicio y fin precisos. Los datasets ricos previos eran pequeños o restringidos —**MAPS** y **Mazurka** son solo de piano, **RWC** es de acceso costoso—. MusicNet resuelve simultáneamente escala, licencia libre y diversidad instrumental.

## Composición: alineación audio-MIDI por DTW

La clave para producir un millón de etiquetas sin anotación manual es la **alineación audio-partitura**. Las grabaciones provienen de archivos de licencia libre; las etiquetas, de **partituras digitales MIDI**. El procedimiento asocia cada evento de la partitura (nota, instrumento, altura) a un tiempo de la interpretación mediante **Dynamic Time Warping (DTW)**: partitura e interpretación son dos secuencias de la misma pieza, desalineadas porque el intérprete toma libertades de tempo, y DTW encuentra la correspondencia temporal óptima. Formalmente, para una partitura $Y \in E \times K$ y una interpretación $X \in \mathbb{R}^{T \times p}$ con costo local $C$, el problema es

$$\min_{t \in \mathbb{Z}} \sum_{i=1}^{n} C(X_{t_i}, Y_i) \quad \text{s.a.} \quad t_0 = 0,\; t_n = m,\; t_i \le t_j \text{ si } i < j,$$

resuelto **exactamente en tiempo y espacio $O(mn)$** por programación dinámica; las restricciones fuerzan una alineación monótona de principio a fin. Como partitura e interpretación viven en espacios distintos, se **sintetiza el MIDI en audio** y se comparan ambos por sus log-espectrogramas, usando la norma $\ell_2$ solo sobre las **50 dimensiones de más baja frecuencia** (~1 kHz), donde la síntesis es fiel. Las grabaciones promedian 6 minutos (de 55 s a casi 18 min) y cada nota conserva instrumento, altura, tiempos de inicio/fin y metadatos musicales.

**Validación del etiquetado.** Como las etiquetas se generan automáticamente, se validan con una **prueba auditiva**: se mezcla sobre la grabación una sinusoide con la frecuencia que la partitura indica en el instante alineado; si la alineación es correcta los tonos se superponen, si no, suena disonante. Dos autores con formación musical analizaron 28 clips de 30 s ralentizados y midieron una **tasa de error promedio de 4.0%**; la causa más común de rechazo son las **repeticiones** musicales.

## La tarea de note prediction y el hallazgo de features tipo Fourier

La *note prediction* se modela como **clasificación multi-etiqueta**: a cada segmento se le asigna $y \in \{0,1\}^{128}$ (códigos MIDI), con $y_n=1$ si la nota $n$ suena **en el punto medio**. Se entrena una regresión lineal sobre un mapa de features $f(x)$ y se predice si $\hat{y}_n > c$, con el umbral $c$ elegido para maximizar F1. El paper compara tres familias: **MLP sobre waveform**, **espectrograma** y **CNN sobre waveform**.

El hallazgo central: los pesos de la capa inferior de las redes end-to-end convergen a **sinusoides moduladas que decaen en los bordes**, análogas a los filtros de Gabor de la visión. La atenuación en los extremos es una **función de ventana** aprendida (tipo Hann): como el segmento se etiqueta con la nota de su centro, la información de los bordes importa menos. La red **redescubre el análisis de Fourier** sin que se le imponga —lo que explica por qué el espectrograma funciona— y, además, aprende pesos cuya distribución de frecuencias **imita la distribución de notas en MusicNet**, concentrando resolución donde los datos la necesitan. La **escala del dataset es esencial**: sobre un subconjunto de 65.000 puntos las features son ruidosas, pero una red **sin regularizar** sobre el MusicNet completo aprende features limpias sin sobreajustar.

## Impacto

Las features aprendidas **superan modestamente** al espectrograma para ventanas comparables: con 2048 muestras el MLP alcanza **56.2%** de precisión promedio frente al **48.8%** del log-espectrograma; con 16.384 muestras la CNN llega a **67.8%**, el mejor resultado del paper, frente al **45.5%** del espectrograma equivalente. Más allá de las cifras, MusicNet estableció un **benchmark reproducible** para transcripción polifónica y una receta —alinear una señal continua contra una referencia estructurada para generar etiquetas densas automáticas— que trasciende la música. Es el argumento paralelo al de ImageNet en visión, trasladado al audio, y anticipa la línea de modelos que operan directamente sobre la waveform.

## Limitaciones

- **Dominio acotado.** Es exclusivamente música clásica de cámara de dominio público; alinear música pop es más difícil porque los sintetizadores reproducen mal timbres vocales y percusión inarmónica.
- **Sesgos del corpus.** Está sesgado hacia **Beethoven** y hacia el **piano solo** (por la abundancia de partituras digitales); flauta y oboe están subrepresentados.
- **Errores de alineación.** El 4.0% no captura todo: los *offsets* se verifican peor que los *onsets*, y los acordes "rolados" de cuerdas quedan mal fechados.
- **Modelo de tarea simple.** La note prediction usa regresión lineal con umbral fijo y no modela la estructura temporal secuencial; es un banco de pruebas para *representaciones*, no un transcriptor de última generación.

## Por qué importa para la Clase 37

La [Clase 37](/clases/clase-37) trata de dónde salen los datos de audio, y MusicNet encarna dos de sus ejes centrales. Primero, como [dataset de audio](/fundamentos/datasets-de-audio) con **etiquetas alineadas nota a nota**: este paper explica *cómo* se fabrica esa supervisión a escala cuando etiquetar a mano es inviable —proyectando la partitura MIDI sobre la grabación vía DTW—. Segundo, como demostración limpia de la tensión entre features hechas a mano y aprendidas: la red redescubre el [análisis de Fourier](/fundamentos/analisis-de-fourier) desde cero, lo que a la vez explica el éxito del espectrograma y muestra que aprender la representación permite superarlo concentrando resolución donde los datos lo exigen.

**Relevancia para salud.** El mecanismo de MusicNet —alinear temporalmente una señal continua contra una referencia estructurada para generar etiquetas automáticas— es directamente análogo a un problema recurrente en señales biomédicas: anotar complejos QRS en un ECG, espículas en un EEG o eventos en audio respiratorio exige un etiquetado experto instante a instante tan costoso como lo era la anotación nota a nota. Usar DTW para alinear un registro fisiológico con una plantilla anotada y **propagar** etiquetas es transferible al dominio clínico; y el hallazgo de features aprendidas sugiere que modelos end-to-end sobre la señal cruda pueden descubrir por sí mismos las bandas de frecuencia clínicamente relevantes.
