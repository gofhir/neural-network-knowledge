---
title: "GTZAN: clasificación de géneros musicales (2002)"
weight: 410
math: true
---

{{< paper-card
    title="Musical Genre Classification of Audio Signals"
    authors="George Tzanetakis, Perry Cook"
    year="2002"
    venue="IEEE Trans. Speech and Audio Processing"
    pdf="/papers/gtzan-tzanetakis-2002.pdf" >}}
Uno de los papers fundacionales del **music information retrieval (MIR)** moderno. Formaliza la **clasificación automática de géneros musicales** a partir de la señal de audio cruda y propone tres familias de descriptores diseñados específicamente para música: **timbral texture** (con [MFCC](/fundamentos/mfcc-y-escala-mel), spectral centroid, rolloff, flux, zero-crossings), **rhythmic content** (beat histogram) y **pitch content** (pitch histogram). Sobre un conjunto propio de **diez géneros musicales**, el sistema alcanza **61% de exactitud** en clasificación whole-file y **44%** en real-time por frame —cerca del techo humano en una tarea de fronteras difusas. Su legado más duradero fue **el dataset**: 1.000 fragmentos de 30 s en 10 géneros (100 por clase), que la comunidad bautizó **GTZAN** por las iniciales del autor y convirtió en el benchmark didáctico de facto —el "MNIST del audio musical" y el dataset del laboratorio de la [Clase 37](/clases/clase-37).
{{< /paper-card >}}

---

## Contexto: la explosión de la música digital

El paper se escribe en pleno estallido de la distribución digital (con Napster como síntoma de la época). Cuando el catálogo pasa de cientos de discos a millones de archivos, **organizar y estructurar automáticamente** el acervo deja de ser un lujo. Los **géneros** son las etiquetas categóricas con que los humanos describen ese universo; el paper es honesto sobre que **no tienen fronteras estrictas** —surgen de una interacción de público, marketing e historia— pero sus miembros comparten rasgos observables de **instrumentación**, **estructura rítmica** y **contenido armónico**. Esas tres dimensiones motivan las tres familias de features.

La crítica central de Tzanetakis y Cook al trabajo previo —proveniente del reconocimiento de voz (MFCC de Davis y Mermelstein, discriminación música/voz de Scheirer y Slaney)— es que esos descriptores *no modelan directamente la señal musical*: no capturan, por ejemplo, la estructura rítmica. De ahí la necesidad de features nuevas, específicamente musicales.

## Método: tres familias de features

Toda la maquinaria opera sobre audio mono, 16 bits, **22.050 Hz**. El análisis de corto plazo usa **ventanas de análisis de 23 ms (512 muestras)**, pero la *textura* sonora vive en el patrón de muchos espectros sucesivos: por eso se introduce la **ventana de textura (1 s, 43 ventanas)**, que calcula **medias y varianzas móviles** de las features —el equivalente conceptual a agregar features en el tiempo.

**Timbral texture (19 dims).** Basadas en la STFT y por frame: **spectral centroid** (centro de gravedad del espectro),

$$C_t = \frac{\sum_{n=1}^{N} n\, M_t[n]}{\sum_{n=1}^{N} M_t[n]},$$

**spectral rolloff** (frecuencia bajo la cual cae el 85% de la magnitud), **spectral flux** (cambio espectral local), **zero-crossings** (ruidosidad), **low-energy** (porcentaje de ventanas con RMS bajo la media) y los **[MFCC](/fundamentos/mfcc-y-escala-mel)** —de los cuales los autores encontraron que **los primeros cinco** dan el mejor desempeño para género.

**Rhythmic content (6 dims).** Un **beat histogram** construido con la transformada wavelet (DWT, filtros DAUB4): se extrae la envolvente de cada banda de octava, se suman y se calcula la **autocorrelación realzada**; los picos dominantes (bins de 40 a 200 bpm) se acumulan sumando su amplitud, de modo que un beat fuerte produce picos más altos. Las features son amplitudes relativas (A0, A1), su razón (RA), periodos (P1, P2) y la suma del histograma (SUM).

**Pitch content (5 dims).** Un **pitch histogram** por detección multi-pitch; las frecuencias se convierten a notas MIDI,

$$c = 12 \log_2\!\left(\frac{f}{440}\right) + 69,$$

y se genera una versión plegada a una octava (chroma) con $c_{\text{folded}} = c_{\text{unfolded}} \bmod 12$, remapeada a un **círculo de quintas** para expresar relaciones tonales.

El vector completo tiene **30 dimensiones**. Se clasifica con **reconocimiento estadístico de patrones**: gaussiano simple (GS), mezcla de gaussianas (**GMM**, inicializada con k-means y ajustada con EM) y **k-NN**, evaluados con validación cruzada de diez pliegues.

## Resultados

Con el vector combinado de 30 dimensiones y clasificación whole-file se alcanza **61%** de exactitud en los diez géneros (**classical, country, disco, hiphop, jazz, rock, blues, reggae, pop, metal**); la versión real-time por frame (solo timbral, 19 dims) llega a **44%**, frente al 10% del azar. Clasificadores auxiliares dan **86%** en música/voz y **74%** en voz de tres clases.

Las **confusiones se parecen a las humanas**: la clásica se confunde con jazz en piezas de fuerte ritmo (26% de la clásica etiquetada como jazz), y el **rock** es el género con **peor exactitud** por su naturaleza amplia. En importancia relativa, las features **timbrales (STFT + MFCC) son las más informativas**; pitch y beat rinden peor por separado, aunque todas superan el azar. Comparado con humanos (Perrot y Gjerdigen: 53% tras 250 ms, 70% tras 3 s), el desempeño automático **no está lejos** —ambos reflejan la naturaleza difusa de las fronteras de género.

## Limitaciones y los defectos conocidos del dataset

El paper reconoce límites de método: las features rítmicas y de pitch se calculan sobre el archivo completo, válido solo si es homogéneo; combinarlas no siempre incrementa la exactitud de forma significativa.

La limitación más citada, sin embargo, llegó con la **literatura posterior** que auditó GTZAN una vez que se volvió estándar: se documentaron **repeticiones** (mismo tema o artista varias veces, a veces en distintos géneros), **errores de etiquetado** y **distorsiones** en varios clips. Un modelo puede "hacer trampa" reconociendo artistas o grabaciones en vez de aprender género, y las cifras sobre GTZAN sin control de estas fugas tienden a estar **optimistamente sesgadas**. Pese a todo el conjunto sobrevivió —por su tamaño manejable, su formato simple (WAV de 30 s) y sus diez géneros balanceados— consolidándose como benchmark didáctico. Conviene presentarlo con esa doble cara: **excelente para aprender, imperfecto para publicar**.

## Por qué importa para la Clase 37

Este paper es la **piedra angular histórica** de la [Clase 37](/clases/clase-37), "Datasets y Herramientas para Audio", por dos razones. Primero, **define el vocabulario de features** que sigue vigente: cuando el [laboratorio](/laboratorios/lab-37) extrae MFCC, spectral centroid, rolloff, flux y zero-crossing rate de un WAV, está usando exactamente el timbral texture set que Tzanetakis y Cook sistematizaron. Segundo, **aporta el dataset del laboratorio**: los "1.000 temas de 30 s en 10 géneros" de la clase *son* GTZAN, uno de los [datasets de audio](/fundamentos/datasets-de-audio) canónicos.

El flujo del laboratorio replica en miniatura el pipeline del paper: cada WAV de GTZAN → extracción de features (típicamente con librosa: MFCC, centroide, rolloff, flux, ZCR, agregadas en medias y varianzas —la ventana de textura del paper) → un clasificador (k-NN o GMM como el original, o una red sobre espectrogramas) que predice uno de los diez géneros. Entender el paper le da al estudiante el *porqué* de cada paso: por qué se promedian las features en el tiempo, por qué los [MFCC](/fundamentos/mfcc-y-escala-mel) dominan, por qué el rock será el peor clasificado, y por qué un ~61% ya es un buen resultado —no un fracaso— dado que roza el techo humano en una tarea intrínsecamente difusa. Y deja una lección metodológica: **auditar el dataset** (repeticiones, mislabels) antes de creer ciegamente en la métrica.
