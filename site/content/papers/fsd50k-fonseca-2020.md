---
title: "FSD50K: dataset abierto de eventos de audio (2020)"
weight: 416
math: true
---

{{< paper-card
    title="FSD50K: An Open Dataset of Human-Labeled Sound Events"
    authors="Eduardo Fonseca et al. (UPF)"
    year="2020"
    venue="IEEE/ACM TASLP 2022 / arXiv:2010.00475"
    pdf="/papers/fsd50k-fonseca-2020.pdf" >}}
FSD50K (Freesound Dataset 50k) es un dataset abierto de eventos sonoros etiquetados por humanos: **51 197 clips (108.3 horas)**, **multi-etiqueta**, con etiquetas débiles y **200 clases tomadas de la ontología de [AudioSet](/papers/audioset-gemmeke-2017)**. Los clips provienen de **Freesound** bajo licencias **Creative Commons**, lo que permite redistribuir el dataset **incluyendo las formas de onda**. Esa es su tesis central, y la razón por la que la [Clase 37](/clases/clase-37) lo pone frente a AudioSet: la primera pregunta pragmática al elegir un dataset de audio es *"¿te dan el audio, o solo un enlace de YouTube que se cae?"*. FSD50K es la respuesta del lado "te dan el audio" —estable, versionado en Zenodo y flexible—, y resuelve de raíz el *link rot* de AudioSet reutilizando su vocabulario pero cambiando la fuente y el modo de distribución.
{{< /paper-card >}}

---

## Contexto: por qué AudioSet no basta como benchmark abierto

El reconocimiento de eventos sonoros (SER) consiste en identificar qué sonidos ocurren en un audio, asignándoles etiquetas de un vocabulario. Como en visión, **los datasets son un insumo crítico** para los métodos hambrientos de datos. En 2017 AudioSet transformó el campo con ≈2.1 millones de clips y 527 clases, pero —según los autores de FSD50K— arrastra un problema mayor: **no es un dataset abierto**. Sus clips son fragmentos de videos de YouTube, no redistribuibles por los Términos de Servicio de la plataforma, de modo que la *release* oficial entrega **features precalculados** a 960 ms en vez de formas de onda. Quien quiera trabajar con la señal cruda simplemente no puede a partir de esa *release*.

La alternativa —bajar el audio de YouTube— tiene un costo que el paper documenta con números concretos: el ***link rot***. En un intento de descarga (11 de mayo de 2020) los autores obtuvieron **18 205 de 20 371** segmentos de evaluación y **19 862 de 22 160** del *balanced train* —pérdidas de **10.6%** y **10.4%**—, y esa cantidad **decrece con el tiempo y de forma desigual** entre particiones: dos equipos que "usan AudioSet" en fechas distintas literalmente no evalúan sobre el mismo conjunto. FSD50K nace para cerrar esa brecha con un benchmark abierto, estable y redistribuible.

## Composición: cómo se construyó FSD50K

El proceso encadena tres cimientos —**Freesound** (repositorio colaborativo del MTG, todo bajo licencias CC), la **ontología de AudioSet** (632 clases, profundidad 6) y el **Freesound Annotator**— con varias etapas de filtrado:

- **Nominación de candidatos.** Se pobló cada clase haciendo *matching* entre los *tags* de los usuarios y *keywords* por clase (con *stemming* de Porter), asociando más de 300 000 clips.
- **Validación (control de calidad).** Los candidatos se validaron manualmente con mecanismos de calidad clave: **FAQs por clase**, la distinción **PP/PNP** ("present and predominant" vs. "present but not predominant"), clips de verificación, acuerdo inter-anotador (hasta que dos coinciden), espectrogramas y normalización de sonoridad **EBU R-128**. Participaron más de 350 anotadores.
- **División de datos.** Para evitar el **"efecto uploader"** (contaminación por clips del mismo autor en train y test), todo el contenido de un *uploader* va entero a *development* o a *evaluation*, con los *uploaders* pequeños reservados para evaluación. Se ordenaron con un puntaje que premia la diversidad:

$$\text{score}_u = n\_labels^{\max}_u + \frac{1}{K_u}\sum_{k=1}^{K_u} n\_labels_{u c_k}$$

- **Refinamiento del *eval*.** El conjunto de evaluación se re-anotó **exhaustivamente** (2 a 5 anotadores por etiqueta, al menos uno experto), de modo que en *eval* **la ausencia de etiqueta significa ausencia de evento sonoro** (salvo error humano). El *development* queda como "correcto pero potencialmente incompleto" (CpI).

El resultado son **51 197 clips** en 200 clases (144 hojas + 56 nodos intermedios), 80% en *development* y 20% en *evaluation*, entregados como **audio PCM sin comprimir de 16 bits, 44.1 kHz, mono**, de 0.3 a 30 s. Sobre licencias, **CC0 y CC-BY suman el 84.7%** del dataset (el 15.3% restante incluye la restricción *NonCommercial*). Un mérito del paper es **cuantificar su propio ruido de etiquetas**: en *dev* las etiquetas son **94.3% correctas** pero potencialmente incompletas (**50.9%** de clips recibieron alguna etiqueta faltante); en *eval*, correctas y completas.

## El eje de disponibilidad y licencia: CC vs. enlaces de YouTube

Este es el corazón del paper. La comparación punto por punto con AudioSet gira sobre dos filas decisivas: **qué se distribuye y de dónde viene**. AudioSet entrega *features* a 960 ms (no el audio), cuyo original vive en YouTube, es **inestable** por *link rot* y su uso choca con políticas de copyright. FSD50K entrega **las formas de onda completas** bajo licencias CC, descargables como ZIP desde una página de Zenodo **estable y versionada**. En términos de la pregunta de la clase —"¿te dan el audio o solo un enlace que se cae?"— FSD50K te da el audio, con tres consecuencias prácticas:

1. **Reproducibilidad exacta:** dos equipos que descargan FSD50K en fechas distintas obtienen el mismo conjunto; en AudioSet, no.
2. **Flexibilidad metodológica:** con la forma de onda se puede aprender desde el audio crudo, recalcular cualquier representación o hacer *data augmentation* sobre la señal —todo imposible desde *features* congelados.
3. **Redistribución legal:** se pueden armar y compartir subconjuntos, algo que las licencias CC permiten y los Términos de Servicio de YouTube prohíben.

Como bonus, el audio de Freesound tiene mejor SNR (media 26 dB vs. 14 dB de AudioSet): al grabarse con la intención de capturar sonido tiende a ser más limpio que el audio incidental de videos. Aun así, los autores concluyen que ambos datasets son **recursos complementarios** (AudioSet aporta más volumen y diversidad de condiciones del mundo real).

## Baselines

El *pipeline* de referencia usa log-mel espectrogramas de 96 bandas, parches de 1 s, pérdida de entropía cruzada binaria (adecuada para multi-etiqueta) y tres métricas independientes del umbral (**mAP**, **$d'$** y **lωlrap**). El hallazgo interesante: el mejor modelo es el **VGG-like**, el más liviano (0.27 M pesos, mAP 0.434), por encima de ResNet-18 (11.3 M, mAP 0.373) y DenseNet-121, arquitecturas mucho más pesadas tomadas "tal cual" de visión. La lectura de los autores: **a esta escala de datos, modelos pequeños con diseño informado por audio superan a arquitecturas grandes de visión sin ajuste** —un contraste con AudioSet, donde las ResNets sí rinden en el estado del arte gracias a su mucho mayor volumen de datos.

## Limitaciones

- **Ruido de etiquetas residual:** persisten etiquetas faltantes (sobre todo en *dev*) e incorrectas ocasionales, ambas *class-conditional*.
- **Desbalance de datos:** por distribución no uniforme de clases, longitud variable de clips y la propia jerarquía de la ontología.
- **Sesgo en el *development*:** al reservar los *uploaders* pequeños para *eval*, algunas clases del *dev* quedan dominadas por pocos *uploaders* grandes.
- **Falta de especificidad del vocabulario:** varios nodos hoja con pocos datos se fusionaron con su nodo padre, reduciendo el detalle.
- **Etiquetas débiles y longitud variable:** la debilidad de la etiqueta varía con la duración (*label density noise*), lo que impone decisiones de diseño.
- **Grabaciones no siempre "en el mundo real":** parte de Freesound son grabaciones tipo *foley* o generadas a propósito, con posible *acoustic mismatch*.

## Por qué importa para la Clase 37

La [Clase 37](/clases/clase-37) presenta FSD50K junto a [AudioSet](/papers/audioset-gemmeke-2017) para enseñar un criterio de selección de [datasets de audio](/fundamentos/datasets-de-audio) que **va más allá del tamaño: la disponibilidad**. El estudiante debería salir con tres ideas ancladas en este paper:

1. **Tamaño no es todo.** AudioSet es ≈40× más grande, pero su *release* como features congelados y su *link rot* (10.6% de pérdida en *eval*, creciente) lo hacen frágil como benchmark reproducible. FSD50K, más chico, es estable, redistribuible y flexible porque entrega la forma de onda bajo licencia CC.
2. **La licencia es un criterio técnico, no solo legal.** Que el 84.7% sea CC0/CC-BY define qué puedes hacer (redistribuir, derivar, usar comercialmente): en audio, la licencia condiciona directamente la reproducibilidad de un experimento y la posibilidad de desplegar un producto.
3. **La calidad del *eval* importa más que la del *train*.** El diseño prioriza un conjunto de evaluación exhaustivamente etiquetado y libre del "efecto uploader" —una lección transversal sobre cómo particionar datos para no inflar métricas.

En investigación clínica el eje "disponibilidad + licencia clara" no es burocracia sino un requisito de reproducibilidad: un modelo de detección de eventos sonoros clínicos —tos, sibilancias, ronquidos, sonidos respiratorios o de alarma en UCI— solo es auditable si el dataset de referencia puede redistribuirse íntegramente y re-descargarse idéntico. FSD50K muestra el estándar deseable: audio real, licencias explícitas por clip, versionado estable en Zenodo y una estimación honesta del ruido de etiquetas.
