---
title: "Microsoft COCO"
weight: 54
math: true
---

{{< paper-card
    title="Microsoft COCO: Common Objects in Context"
    authors="Lin, Maire, Belongie, Bourdev, Girshick, Hays, Perona, Ramanan, Zitnick, Dollar"
    year="2014"
    venue="ECCV 2014"
    pdf="/papers/coco-lin-2014.pdf"
    arxiv="1405.0312" >}}
El dataset estandar de deteccion, segmentacion de instancias y captioning desde 2014. **328k imagenes, 2.5M instancias anotadas, 80 categorias reales** en escenas no-iconicas con multiples objetos por imagen. Introduce ademas la metrica **mAP@[.5:.95]** que premia localizacion precisa.
{{< /paper-card >}}

---

## Motivacion

Antes de COCO, los datasets dominantes tenian limitaciones complementarias:

| Dataset | Tamano | Clases | Problema |
| --- | --- | --- | --- |
| PASCAL VOC | ~11k | 20 | Pocas clases, ~2-3 objetos por imagen |
| ImageNet | ~14M | 1000 (cls) / 200 (det) | Imagenes **iconicas**, un objeto central |
| SUN | ~131k | 908 escenas | Pocas instancias por categoria |

El problema comun: imagenes **iconicas** (objeto centrado, sin contexto, sin oclusiones). Esto producia modelos que funcionaban en benchmarks pero fallaban en escenas reales.

## Ideas principales

- **Imagenes no-iconicas**: buscaron pares de categorias en Flickr ("dog + car") en vez de una sola, forzando composiciones naturales.
- **Multiples objetos por imagen**: promedio de **7.7 objetos por imagen** (vs 3.0 ImageNet, 2.3 VOC).
- **Segmentacion a nivel de instancia**: no solo bounding box, sino mascara poligonal por objeto individual (la base de Mask R-CNN).
- **91 categorias propuestas, 80 reales**: 11 fueron descartadas durante anotacion por baja calidad (hat, shoe, plate, mirror, window, etc.). Los IDs **nunca se renumeraron** -> los datasets tienen "huecos" hasta el ID 90.
- **Categorias "entry-level"**: las que un nino de 4 anos reconoceria. No incluye razas especificas de perros ni modelos de autos. Cobertura: 11 super-categorias (persona, vehiculos, animales, comida, muebles, etc.).
- **Anotacion masiva con AMT**: 85.000 horas-trabajador (= ~10 anos-persona a tiempo completo). Pipeline de 3 etapas: category labeling, instance spotting, instance segmentation.

## La metrica COCO

Antes de COCO la metrica estandar era **mAP@0.5** (solo IoU 0.5). COCO introdujo:

$$\text{mAP}_{[.5:.95]} = \frac{1}{10} \sum_{\tau \in \{0.5, 0.55, \ldots, 0.95\}} \text{mAP}_\tau$$

Promedia sobre 10 thresholds de IoU. **Penaliza imprecision en la caja**, no solo en la clase. Un detector con IoU=0.6 ahora cuenta menos que uno con IoU=0.9.

Adicionales:

- **AP_S**, **AP_M**, **AP_L**: separan accuracy por tamano de objeto. Permite ver donde fallan los detectores (los pequenos sufren mas).

## Splits oficiales

| Split | Imagenes | Uso |
| --- | --- | --- |
| train2017 | 118.287 | Entrenamiento |
| val2017 | 5.000 | Validacion publica |
| test-dev2017 | 20.288 | Test sin etiquetas, submission a servidor |
| test-challenge2017 | 20.000 | Reservado para competencia anual |

## Impacto

- **Benchmark de facto** de deteccion desde 2016. Todo paper de deteccion reporta mAP@[.5:.95] en COCO.
- Pesos pre-entrenados de `torchvision.models.detection` se entrenan en **COCO train2017**.
- Datasets posteriores (Open Images, LVIS, Objects365) son mas grandes pero ninguno ha desplazado a COCO como benchmark canonico por la combinacion de calidad + tamano + metricas estandarizadas.

## Conexion con el laboratorio

El lab usa `fasterrcnn_resnet50_fpn(pretrained=True)` que descarga pesos entrenados en COCO 2017. La lista `CATEGORY_NAMES` tiene **91 entradas**:

- Indice 0: `__background__`
- Indices 1-90: las 80 categorias reales + 11 `'N/A'` (los huecos descartados).

La motivacion de la **Parte 2** del lab (fine-tuning para mapaches) es precisamente que **raccoon no esta en COCO**. Cuando se le pasa una imagen de mapache al modelo COCO, el detector con threshold 0.5 predice `bear:0.5x` (la categoria mas parecida fenotipicamente). Esto motiva reemplazar el clasificador final para anadir "raccoon" como clase.
