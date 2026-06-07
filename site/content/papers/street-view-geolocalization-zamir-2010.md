---
title: "Geolocalización con Google Street View (UCF)"
weight: 100
math: true
---

{{< paper-card
    title="Accurate Image Localization Based on Google Maps Street View"
    authors="Zamir, Shah"
    year="2010"
    venue="ECCV 2010"
    pdf="/papers/street-view-geolocalization-zamir-2010.pdf" >}}
Aborda el problema de estimar la **ubicación GPS exacta** de una imagen con precisión comparable a un GPS de mano. Usa un dataset estructurado de **~100.000 imágenes de Google Maps Street View** como referencia. Indexa los descriptores **SIFT** de las imágenes de referencia en un árbol; para localizar una *query*, consulta el árbol con sus SIFT, aplica un **pruning basado en GPS-tag** que elimina descriptores poco fiables, y un esquema de **voting** donde cada descriptor vota por la ubicación de su vecino más cercano. Define la **Confidence of Localization** (basada en la curtosis de la distribución de votos) y un método jerárquico para localizar **grupos de imágenes**. Es el origen del *Google Street View Dataset* de UCF que el Laboratorio 21 reaprovecha para OCR geolocalizado.
{{< /paper-card >}}

> **Nota de versiones.** El paper original es de **ECCV 2010**. Una versión extendida posterior (Zamir & Shah, IEEE TPAMI 2014, *"Image Geo-Localization Based on Multiple Nearest Neighbor Feature Matching Using Generalized Graphs"*) introdujo el matching por **GMCP** (Generalized Minimum Clique Problem) y consolidó el dataset conocido como `GMCP_Geolocalization` — el subconjunto de Pittsburgh con archivo `GPS_Long_Lat_Compass.mat` que se usa en el lab.

---

## El problema

Determinar la ubicación GPS exacta de una imagen es un problema clásico de visión con muchas aplicaciones (organización de fotos, realidad aumentada, reconstrucción 3D urbana). El reto: la inmensa mayoría de las imágenes en repositorios online (Flickr, Panoramio) **no están geo-etiquetadas**. La idea del paper es usar como **referencia** un corpus masivo de imágenes que *sí* tienen GPS preciso —las de Google Street View, capturadas sistemáticamente a lo largo de las calles— y localizar una imagen nueva por **matching de apariencia** contra ese corpus.

A diferencia del reconocimiento de lugares basado en clasificación (¿en cuál de K ciudades está?), aquí se busca **precisión métrica**: un GPS comparable al de un dispositivo de mano, no solo la ciudad o el barrio.

## El dataset

- **~100.000 imágenes** de Google Maps Street View como conjunto de referencia, cada una con su **GPS** asociado.
- Estructura por **posiciones (placemarks)** a lo largo de las calles: cada punto geográfico se captura con **múltiples vistas** (distintas orientaciones de cámara), cubriendo el entorno alrededor del punto. En el subconjunto que usa el lab esto se materializa como **6 fotos por posición GPS**.
- El archivo `GPS_Long_Lat_Compass.mat` codifica, por posición, la **latitud, longitud y orientación de brújula** (compass). El subconjunto de Pittsburgh del lab usa las primeras 1099 posiciones (6594 fotos).

## El método

El pipeline de localización de una imagen *query*:

1. **Indexación SIFT.** Se detectan puntos de interés **SIFT** en las imágenes de referencia y sus descriptores se indexan en un **árbol** (búsqueda eficiente de vecinos más cercanos).
2. **Consulta.** Para la imagen query se calculan sus SIFT y se consulta el árbol buscando los descriptores de referencia más parecidos.
3. **Pruning por GPS-tag.** Un método novedoso descarta los descriptores **poco fiables** (los que aparecen dispersos en muchas ubicaciones, poco discriminativos) aprovechando la información GPS de las referencias.
4. **Voting + smoothing.** Cada descriptor de la query **vota** por la ubicación a la que pertenece su vecino más cercano; un paso de suavizado agrega los votos espacialmente.
5. **Confidence of Localization.** Un parámetro basado en la **curtosis** de la distribución de votos mide cuán fiable es la localización: una distribución muy concentrada (curtosis alta) indica alta confianza; una dispersa, baja.
6. **Localización de grupos.** Para conjuntos de imágenes, se localiza cada una; luego las demás se emparejan contra el vecindario del primer match fiable, y la ubicación final se decide por la Confidence of Localization. Esto permite geolocalizar queries muy ambiguas que serían imposibles de localizar de forma aislada.

## Resultados clave

El método alcanza una precisión de localización **comparable a la de dispositivos GPS de mano** sobre el conjunto de prueba de imágenes urbanas, superando a los baselines de matching directo gracias al pruning por GPS-tag (que reduce los falsos matches de descriptores repetitivos como ventanas o texturas de fachada) y al esquema de voting con confianza. La localización jerárquica de grupos extiende el alcance a imágenes individualmente no localizables.

## Limitaciones

- Depende de la **cobertura** de Street View: zonas sin captura no son localizables.
- SIFT es sensible a cambios fuertes de iluminación, estación y a la **dinámica urbana** (obras, vehículos, vegetación) entre la referencia y la query.
- Los descriptores repetitivos de entornos urbanos (ventanas idénticas, señalética estándar) generan ambigüedad — justo lo que el pruning busca mitigar.

## Conexión con el laboratorio 21

El [Laboratorio 21](/laboratorios/lab-21) **reaprovecha el dataset** (no el método de localización): en lugar de estimar dónde se tomó una foto, usa [ABCNet](/papers/abcnet-liu-2020) para **leer el texto** de las fotos de calle y, cruzando con el GPS que el dataset ya provee, **mapear geográficamente** dónde aparece cada palabra (función `draw_in_map`). Es minería de información geoespacial desde texto en imágenes.

La estructura de **6 vistas por posición GPS** es central en el lab: la función `draw_in_map` mapea cada foto a su punto con `image_id // 6`. Además, el lab descubre empíricamente un problema de calidad de datos —muchas capturas incluyen el **overlay de la interfaz de Google Maps** (controles, watermark "© 2009")—, que el modelo de OCR lee como si fuera texto de la escena; se mitiga con un filtrado geométrico por zonas (`get_mask`). Ver la página [App 2 · Google Street View](/laboratorios/lab-21/app-streetview).
