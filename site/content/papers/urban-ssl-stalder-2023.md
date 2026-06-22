---
title: "SSL unveils change in urban housing (2023)"
weight: 325
math: true
---

{{< paper-card
    title="Self-supervised learning unveils change in urban housing from street-level images"
    authors="Stalder, Volpi, et al."
    year="2023"
    venue="arXiv"
    pdf="/papers/urban-ssl-stalder-2023.pdf"
    arxiv="2309.11354" >}}
Paper de aplicación —no de algoritmo— que cierra la [Clase 28](/clases/clase-28). Stalder et al. adaptan **Barlow Twins** ([aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo)) a imágenes street-level con un giro mínimo pero decisivo en la tarea *pretext*: las dos "vistas" que el modelo debe alinear no son distorsiones artificiales de una foto, sino **dos fotografías reales del mismo punto geográfico tomadas en años distintos**. El resultado es **Street2Vec**, un espacio de representaciones donde la distancia coseno entre la foto de 2008 y la de 2018 de un lugar *es* una métrica de cambio urbano —aprendida sin una sola etiqueta—. Aplicado a 15,3 millones de imágenes de Londres, detecta dónde cambió el parque de vivienda y separa cambio mayor de cambio menor.
{{< /paper-card >}}

---

## Contexto: medir el mundo desde la calle, pero sin la dimensión temporal

Las imágenes a nivel de calle (Google Street View, Mapillary, Baidu, Tencent) capturan el entorno urbano **tal como lo viven sus residentes** y, en la última década, alimentaron una línea madura de investigación con deep learning supervisado: estimar composición sociodemográfica de barrios, desigualdades de salud y ambientales, precios de vivienda, criminalidad, contaminación, percepción de seguridad, caminabilidad. Ya sabíamos "medir variables socioeconómicas desde fotos urbanas".

El cuello de botella que ataca este paper es otro. Los proveedores archivan imágenes **multi-anuales** de muchas ciudades desde hace más de una década, pero casi todos esos estudios eran **transversales** (cross-sectional): una foto del estado del mundo en un instante, no su evolución. ¿Por qué? Porque lo supervisado exige etiquetas de alta calidad, y conseguir etiquetas **temporalmente coherentes y espacialmente densas a escala** es muy difícil: no existe un dataset de "cuánto cambió la vivienda en cada esquina de Londres entre 2008 y 2018". La dimensión temporal —el mayor activo de estas imágenes para estudiar el cambio— quedaba subexplotada.

El problema de fondo es de política pública: las ciudades enfrentan una crisis de vivienda asequible, pero nuestra capacidad de **monitorear el progreso** es limitada. El censo es completo pero se levanta cada diez años; las encuestas de hogares son frecuentes pero carecen de granularidad espacial; los permisos de construcción capturan solo parte del cambio (excluyen demoliciones, renovaciones, regeneración) y están fragmentados entre actores públicos y privados. El [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) aparece como la salida natural: extraer información de grandes conjuntos **estructurados pero sin etiquetar**, optimizando una tarea auxiliar que expone la estructura intrínseca de los datos.

## Contribución central

El aporte se descompone en tres afirmaciones verificables:

1. **Primera aplicación de SSL temporal a street-level para medir cambio urbano.** La métrica de cambio **no requiere etiquetas**: solo usa las imágenes y su metadata de fábrica (año de captura y ubicación).
2. **Street2Vec: rediseño del pretext de Barlow Twins usando el tiempo como señal.** En lugar de distorsiones sintéticas, las dos vistas son dos fotos del mismo punto en años distintos. El modelo aprende a representar lo estable (estructura urbana) y a descartar lo efímero (estación, luz, autos, personas).
3. **Una métrica de distancia interpretable.** El grado de cambio en un punto es la **distancia coseno** entre el embedding de su imagen de 2008 y el de 2018. Embeddings cercanos = sin cambio relevante; lejanos = cambio estructural probable.

La elegancia está en el punto 2: el método transforma una propiedad del *dato* (el mundo cambia lento; entre dos fotos del mismo lugar lo que más varía suele ser irrelevante) en una *señal de entrenamiento*. Se asume que, en promedio, dos imágenes del mismo punto comparten estructura y difieren solo en lo accesorio; el modelo aprende esa invarianza y, por contraste, se vuelve **sensible a los casos minoritarios donde sí hubo cambio estructural** —que son justamente los de interés—.

## Método

### Los datos

15.335.000 imágenes en 3.833.750 ubicaciones de Londres entre 2008 y 2021, vía la API de Street View. El muestreo: una grilla de 50 m sobre la red vial de OpenStreetMap dentro del Greater London; por cada punto, las panorámicas cercanas, usando **cuatro imágenes** de 600×600 px (las orientaciones 0°, 90°, 180°, 270°). Para detectar cambio se usaron los **329.031 puntos** con imágenes en 2008 y en 2018 —el mayor lapso con gran solapamiento espacial—.

### Barlow Twins, el método base

Conviene fijar el original antes de adaptarlo. Barlow Twins (Zbontar et al., 2021) genera dos versiones distorsionadas de cada imagen, las pasa por una ResNet-50 y un proyector MLP, y obtiene dos batches de embeddings $Z^A$ y $Z^B$. Tras estandarizar, calcula la **matriz de correlación cruzada** $C$ entre dimensiones y minimiza:

$$\mathcal{L}_{BT} = \sum_i (1 - C_{ii})^2 + \lambda \sum_i \sum_{j \neq i} C_{ij}^2$$

El primer término empuja la diagonal a 1: cada dimensión debe ser **invariante** a la distorsión (las dos vistas dan el mismo valor). El segundo empuja lo de fuera de la diagonal a 0: las dimensiones deben estar **decorrelacionadas** (cada una aporta algo nuevo, reduciendo redundancia). Esa combinación —invarianza + redundancia mínima— evita el colapso trivial **sin pares negativos explícitos**, lo que distingue a Barlow Twins de los contrastivos clásicos tipo SimCLR.

### La adaptación: Street2Vec

El cambio es conceptualmente pequeño pero semánticamente decisivo. En vez de distorsiones predefinidas, **las dos vistas son dos imágenes reales del mismo lugar en años distintos**: en cada paso, el segundo batch se toma de las **mismas ubicaciones pero en otro año**, cuando hay al menos dos años disponibles. Si solo existe una imagen para una coordenada, se recurre a un jitter de color suave como respaldo —recuperando el esquema original solo en ese caso raro—.

El supuesto explícito: en promedio, dos fotos del mismo punto tendrán fuertes variaciones de apariencia que **no** son el foco (iluminación, estación, gente, autos) pero poco o ningún cambio estructural. No se puede descartar el cambio estructural entre dos fotos —de hecho es el objetivo detectarlo—, pero se postula que esos casos son mucho menos frecuentes. Así el modelo aprende representaciones **invariantes al cambio irrelevante y sensibles a lo estructural**, sin etiquetas que lo marquen.

Detalles: entrenamiento en una GPU, un solo epoch (~30 horas; más no mejoraba). Las cuatro orientaciones se concatenan en una vista panorámica de 128×512 px. Embedding de dimensión **1024**, $\lambda = 0{,}005$. En inferencia, el cambio en un punto es la distancia coseno entre sus dos embeddings:

$$d_{cos}(x, y) = 1 - \frac{x^\top y}{\lVert x \rVert_2 \lVert y \rVert_2}$$

Eligen coseno y no euclidiana deliberadamente: con embeddings de alta dimensión y decorrelacionados, prefieren capturar cambio moderado repartido en muchas dimensiones antes que cambio grande en unas pocas.

## Resultados

No hay ground truth temporal —esa es la premisa—, así que la validación es indirecta, en tres frentes.

**(a) Opportunity Areas (OAs).** Zonas que el Alcalde de Londres designó desde 2004 como clave para nueva vivienda e infraestructura, con incentivos activos. Hipótesis: deberían mostrar **más cambio**. Confirmado ($p < 0{,}01$): el método destacó transformaciones reales (King's Cross, St. Pancras, Tottenham Court Road) y áreas sobre nueva infraestructura de transporte (extensión de la Northern Line, Elizabeth Line en Battersea y Woolwich), e identificó OAs que **no** lograron el desarrollo esperado pese a los incentivos.

**(b) Cambio sutil vs. desarrollo mayor (1.449 pares etiquetados).** Etiquetaron a mano 1.449 pares 2008/2018 en una escala ordinal de cinco clases y compararon contra el mismo ResNet-50 preentrenado solo en ImageNet:

| | Clase 1 (mínimo) | Clase 2 (irrelevante) | Clase 3 (menor) | Clase 4 (mayor) | Clase 5 (anómalo) |
|---|---|---|---|---|---|
| **Street2Vec** | 0,090 | 0,205 | 0,424 | 0,592 | 0,838 |
| **Baseline (ImageNet)** | 0,151 | 0,201 | 0,228 | 0,275 | 0,685 |

El baseline respeta el orden pero **no tiene spread** (entre clase 2 y 4 va solo de 0,201 a 0,275). Street2Vec separa con amplitud: aprendió features de cambio urbano *relevante* y descartó lo visualmente irrelevante.

**(c) Clustering con UMAP.** Proyectando 10.000 panorámicas a 2D, y **sin haber entregado nunca coordenadas geográficas al modelo**, el coloreo revela patrones espaciales coherentes: el centro se agrupa y transita gradualmente hacia los suburbios; los puntos celestes siguen las autopistas. La geometría del espacio aprendido codifica semántica urbana.

## Limitaciones reconocidas

- **No todo cambio de vivienda deja señal visual externa:** mejoras de eficiencia energética, aumento de capacidad interna o cambios de uso pueden no alterar la fachada. Lo ideal es combinar imagen con otras fuentes.
- **Anomalías y eventos raros** (nieve en Londres) influyen en la detección y deben tenerse en cuenta.
- **Dependencia de proveedores privados** (Google, Baidu, Bing) que imponen crecientes restricciones de acceso, incluso para uso no comercial de interés público.
- **Sesgos de cobertura y privacidad:** la cobertura es problemática en países con alto escrutinio de privacidad y en asentamientos informales; además Londres tiene de las mejores iniciativas de datos geoespaciales del mundo, lo que dificulta extrapolar.
- **Sin ground truth a nivel de punto:** la validación se apoya en proxies (OAs) y una muestra etiquetada a mano.

## El mensaje de cierre de la Clase 28

Más allá de lo técnico, este paper demuestra que el SSL puede generar **herramientas de medición de bajo costo y gran escala** para problemas de política pública donde las etiquetas son inalcanzables —un insumo posible para rastrear el avance hacia vivienda adecuada y asequible, uno de los Objetivos de Desarrollo Sostenible de la ONU—.

Pero el aporte más interesante para el cierre de la [Clase 28](/clases/clase-28) (*Aprendizaje Autosupervisado*) es de **diseño**: la creatividad del pretext. Barlow Twins fue concebido con augmentaciones sintéticas; Stalder et al. notaron que el dominio ofrece una augmentación *natural y semánticamente más rica* —el paso del tiempo en un lugar fijo— y la usaron como señal. Cambiar de dónde viene la "segunda vista" cambia por completo qué aprende el modelo: ya no invarianza a recortes, sino invarianza a lo efímero del mundo y sensibilidad a lo estructural.

Esa es la lección transferible que el curso retiene: **el algoritmo de SSL es genérico; el ingenio está en encontrar, dentro de la estructura del dominio, la señal supervisora gratuita que codifica exactamente la invarianza que uno quiere**. El tiempo regala la segunda vista gratis. En un dominio médico o de registros, el equivalente sería preguntarse qué "dos vistas del mismo objeto" provee gratis la estructura de los datos —dos exámenes del mismo paciente, dos registros del mismo individuo— para aprender qué debe permanecer invariante y qué señala un cambio relevante.

## Enlaces

- Fundamento transversal: [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado) — el paradigma SSL, sus pretext tasks y la distinción frente a los métodos no supervisados clásicos.
- Fundamento transversal: [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) — métodos contrastivos y de redundancia (SimCLR, Barlow Twins), invarianza, decorrelación y la noción de "dos vistas".
- Clase: [Clase 28](/clases/clase-28) — *Aprendizaje Autosupervisado*; este paper ilustra el slide de contrastive learning en otros dominios y el mensaje de cierre.
- Código: `gitlab.renkulab.io/deeplnafrica/Street2Vec`. Preprint: arXiv:2309.11354v2 [cs.CV].
