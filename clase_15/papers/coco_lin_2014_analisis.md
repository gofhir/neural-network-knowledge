# Análisis: Microsoft COCO (Lin et al., 2014)

> **Cita completa**
> Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., & Dollár, P. (2014). *Microsoft COCO: Common Objects in Context*. ECCV 2014.
>
> arXiv: [1405.0312](https://arxiv.org/abs/1405.0312) (v3, 21 febrero 2015)
> Sitio: <https://cocodataset.org>
> Citas (2026): ~50.000 (paper de dataset más citado en visión).

PDF local: [coco_lin_2014.pdf](coco_lin_2014.pdf)

---

## 1. Contexto y motivación

A 2014, los datasets dominantes en visión tenían cada uno una limitación distinta:

| Dataset | Tamaño | Categorías | Problema |
|---------|--------|-----------|----------|
| MNIST | 70k | 10 | Solo dígitos sobre fondo blanco |
| CIFAR-10/100 | 60k | 10/100 | Imágenes tiny (32×32 px) |
| Caltech 101/256 | ~10k | 101/256 | Imágenes "icónicas", un objeto centrado |
| **PASCAL VOC** | ~11k | **20** | Pocas categorías, ~2-3 objetos por imagen |
| **ImageNet** | ~14M | **22k** (clasificación) / **200** (detección) | Imágenes "icónicas", un objeto principal |
| SUN | ~131k | 908 escenas | Pocas instancias por categoría |

El problema común: las imágenes eran **icónicas** (objeto centrado, sin contexto, sin oclusiones, sin clutter). Esto producía modelos que funcionaban en benchmarks pero fallaban en escenas reales.

> *"For many categories of objects, there exists an iconic view [...] We posit that current recognition systems perform fairly well on iconic views, but struggle to recognize objects otherwise — in the background, partially occluded, amid clutter."*

---

## 2. Contribución central de COCO

COCO se diseñó explícitamente para resolver tres problemas que los datasets previos no abordaban:

1. **Detección de objetos no-icónicos**: vistas no canónicas, objetos parcialmente ocluidos, en contexto real.
2. **Razonamiento contextual entre objetos**: múltiples objetos por imagen (7.7 en promedio vs 3.0 en ImageNet y 2.3 en VOC).
3. **Localización 2D precisa**: segmentación a nivel de píxel (instancia), no solo bounding boxes.

### Comparación visual (Figura 1 del paper)

```
(a) Image classification    (b) Object localization      (c) Semantic segmentation    (d) COCO (instance segmentation)
   "person, sheep, dog"      [cajas: persona, oveja...]   pixeles: persona, oveja...    instancias: persona 1,
                                                                                         oveja 1, oveja 2, ...
```

COCO es el primer dataset masivo que ofrece **segmentación a nivel de instancia** (no solo semántica), lo que es la base de tareas como Mask R-CNN.

---

## 3. Estadísticas del dataset

### Volumen (versión 2014, ampliada en 2017)

- **328.000 imágenes** (versión 2014).
  - 165k en train + 81k val + 81k test.
- **~2.5 millones de instancias etiquetadas**.
- **91 categorías propuestas**, **82 con más de 5000 instancias** (luego se descartaron a 80 reales).
- Promedio: **7.7 objetos por imagen** (vs 3.0 ImageNet, 2.3 VOC, 17 SUN).

### Versión COCO 2017 (la que usa el lab)

Re-split del mismo conjunto de imágenes:
- **train2017**: 118.287 imágenes.
- **val2017**: 5.000 imágenes.
- **test-dev2017**: 20.288 imágenes (submission a servidor).
- **80 categorías reales**, agrupadas en **11 super-categorías**.

### Por qué 91 categorías originalmente vs 80 reales

La lista original (Figura 5a del paper) tenía 91 categorías candidatas. Después del proceso de etiquetado se descartaron 11 que no alcanzaron volumen suficiente (>5000 instancias) o tenían problemas de anotación:

- "hat", "shoe", "eye glasses" (objetos demasiado pequeños/parciales).
- "street sign", "plate", "mirror", "window", "desk", "door", "blender", "hair brush" (ambiguos o redundantes con otras categorías).

Pero los **IDs nunca se renumeraron**. Por eso `CATEGORY_NAMES` en el lab tiene 91 entradas con 11 `'N/A'`.

---

## 4. Diseño del dataset — decisiones de método

### Selección de categorías

El paper detalla un proceso meticuloso:

1. **Combinaron categorías de PASCAL VOC** + subset de las 1200 palabras más frecuentes para objetos visualmente identificables.
2. **Niños de 4-8 años nombraron objetos** que veían en sus entornos cotidianos → 272 candidatos.
3. **Los autores votaron 1-5** considerando: frecuencia, utilidad práctica, diversidad relativa a otras categorías.
4. **Balance entre super-categorías** (animales, vehículos, muebles, etc.) para no sobre-representar ninguna.
5. **Backwards compatibility con PASCAL VOC** (las 20 categorías de VOC están incluidas).
6. **Solo "things", no "stuff"**: incluyeron objetos discretos (persona, silla, auto) pero no materiales/superficies (cielo, calle, pasto). "Stuff" carece de límites de instancia claros, y el dataset es de **instance segmentation**.

### Recolección de imágenes — el truco para evitar imágenes icónicas

Si simplemente buscas "perro" en Google, obtienes **fotos icónicas** (perro centrado, fondo limpio). Para evitar esto:

> *"We did not search for object categories in isolation. A search for 'dog' will tend to return iconic images of large, centered dogs. However, if we searched for pairwise combinations of object categories, such as 'dog + car', we found many more non-iconic images."*

Es decir, **buscaron pares de categorías** en Flickr. Esto fuerza imágenes con composición natural donde los objetos coexisten sin ser el sujeto principal.

### Pipeline de anotación (Figura 3 del paper)

Tres etapas con Amazon Mechanical Turk (AMT):

1. **Category labeling** (etiquetar qué categorías hay):
   - Categorías agrupadas en 11 super-categorías; el trabajador decide primero si hay alguna super-categoría presente.
   - Luego decide cuáles sub-categorías concretas hay.
   - **8 trabajadores por imagen** para alta recall.
   - **~20.000 horas-trabajador** en total.

2. **Instance spotting** (marcar dónde están las instancias):
   - Por cada categoría identificada, marcar una cruz sobre cada instancia.
   - **8 trabajadores por imagen** para encontrar todas las instancias (incluso las pequeñas u ocluidas).
   - ~10.000 horas-trabajador.

3. **Instance segmentation** (segmentar a nivel de píxel):
   - **22 horas por 1000 instancias**.
   - Training task obligatorio para los trabajadores antes de poder anotar.
   - Verificación posterior (3-5 trabajadores votan si la segmentación es buena).
   - ~55.000 horas-trabajador en total para 2.5M instancias.

**Total: ~85.000 horas-trabajador de anotación = ~10 años-persona a tiempo completo.** Costo estimado en millones de dólares.

---

## 5. Métricas de evaluación — el "mAP COCO"

Antes de COCO, las competencias usaban **mAP@0.5** (mean Average Precision con IoU threshold 0.5). COCO introdujo una métrica más estricta:

### mAP@[.5, .95]

Promedia mAP sobre 10 thresholds de IoU: $[0.5, 0.55, 0.6, ..., 0.95]$.

- Premia **localización precisa**: una caja con IoU=0.6 con la GT es solo "ok", pero ya no perfecta como en mAP@0.5.
- Detectores que solo "cazan" objetos sin localizarlos bien pierden mucho.

### Métricas adicionales

- **AP_S**: AP para objetos pequeños (área < 32²).
- **AP_M**: AP para objetos medianos (32² ≤ área < 96²).
- **AP_L**: AP para objetos grandes (área ≥ 96²).

→ Permite analizar dónde fallan los detectores. Faster R-CNN sin FPN tenía AP_S muy bajo (los objetos pequeños se perdían); FPN lo arregló.

---

## 6. Impacto e influencia

### COCO se volvió el estándar de evaluación de detección

- **2015**: ImageNet Detection sigue siendo el benchmark principal. COCO emerge.
- **2016 en adelante**: COCO eclipsa a ImageNet Detection. Todos los papers de detección reportan mAP@[.5, .95] en COCO.
- **2017–2025**: COCO sigue siendo el benchmark canónico, aunque hay datasets más grandes y especializados.

### Datasets que vinieron después

- **Visual Genome** (2017): grafos de relaciones entre objetos.
- **Open Images V4-V6** (2018–2020): más grande (~9M imágenes) pero más ruidoso.
- **LVIS** (2019): 1200 categorías con long tail (extiende COCO).
- **Objects365** (2019): 365 categorías, 2M imágenes.

Ninguno ha reemplazado a COCO porque la combinación de calidad de anotación + tamaño + métricas estandarizadas + comunidad establecida es difícil de replicar.

### Influencia en el modelo

- Faster R-CNN, Mask R-CNN, RetinaNet, FCOS, DETR, Co-DETR... todos se entrenan y evalúan principalmente en COCO.
- Los pesos pre-entrenados de `torchvision.models.detection` están entrenados en **COCO train2017**.

---

## 7. Conexión con el laboratorio

El lab usa:

- **Modelo**: Faster R-CNN pre-entrenado en COCO 2017.
- **Lista `CATEGORY_NAMES`**: las 91 entradas (80 reales + 11 'N/A') de la API original de COCO.
- **Imágenes de prueba** (zebras, oficina, comida): elegidas para contener categorías que están en COCO. Si pasaras una imagen con una **iguana** o un **dragón**, el modelo no devolvería nada (esas clases no existen en COCO).

Esto motiva la **Parte 2 del lab**: para detectar **mapaches** (clase **no presente en COCO**), hay que hacer fine-tuning. El modelo aprende a usar todos los features pre-entrenados pero reemplaza el clasificador final.

### Detalle práctico

El paper reporta que las categorías con **más instancias por imagen** son:
- person (~272k instancias totales)
- chair, car, dining table, cup, bottle...

Las con **menos instancias**:
- toaster, hair drier, bear, fire hydrant...

Esto significa que el modelo es **más confiable** para clases comunes (persona, auto) que para raras (toaster). Cuando ejecutes la inferencia, espera más falsos positivos / negativos en las categorías raras.

---

## 8. Aspecto a destacar — la métrica del 4-year-old

Una frase del abstract que vale la pena retener:

> *"Our dataset contains photos of 91 objects types that would be easily recognizable by a 4 year old."*

Es decir: las categorías son **entry-level**. No incluye razas específicas de perros (eso lo hace el dataset Stanford Dogs), ni modelos específicos de autos (eso lo hace Cars-196). COCO es deliberadamente **práctico** para detección general.

Esto contrasta con ImageNet, que tiene 200 razas de perros distintas pero solo "dog" en general también. COCO eligió la categoría general → más útil para aplicaciones reales.
