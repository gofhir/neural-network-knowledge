---
title: "Vocabulario abierto y segmentación"
weight: 6
---

La tercera fase del laboratorio cambia de paradigma: en lugar de un detector de 80 clases fijas, uno que decide qué buscar a partir de **texto**; y en lugar de cajas, **máscaras a nivel de píxel**.

```python
yolo_world = YOLOWorld("yolov8s-worldv2.pt")
yolo_world.set_classes(["person"])
sam_model = SAM("sam2.1_t.pt")
```

Es la versión con componentes de generación anterior del paradigma con que cierra la clase: [SAM 3](/papers/sam3-meta-2025) haciendo seguimiento de conceptos definidos en lenguaje natural —*"track all yellow buses"*—, con detector y tracker de arquitecturas separadas.

Dos hallazgos: el vocabulario restringido resulta ser mucho más que una comodidad de interfaz, y el modelo de video más sofisticado del pipeline **no usa nada de lo que lo hace un modelo de video**.

## Qué hace `set_classes()` realmente

YOLO-World no procesa el texto en cada frame. Corre el encoder de texto de CLIP **una sola vez**, obtiene los embeddings de los conceptos pedidos, y los **hornea dentro de los pesos** de la capa final como una convolución 1×1. Es la reparametrización que le permite ser de vocabulario abierto y correr en tiempo real a la vez.

Tres consecuencias:

- **En inferencia no hay costo de lenguaje.** Sobre 596 frames, el texto se procesó una vez, no 596.
- **El vocabulario queda congelado.** Después de `set_classes(["person"])` el modelo es, literalmente, un detector de personas de vocabulario cerrado.
- **Es una escritura sobre los pesos**, y por lo tanto sensible al estado del proceso. Tras varias corridas apareció un `RuntimeError: Inference tensors do not track version counter`: los tensores creados bajo `torch.inference_mode()` no llevan contador de versión y no pueden usarse como pesos de una convolución en un contexto que sí lo espera. La solución práctica es instanciar un `YOLOWorld` nuevo por cada conjunto de prompts.

Un detalle sutil que importa para la actividad: con **un solo prompt** la clasificación por similitud tiene un único candidato y no hay competencia — cualquier región cuya similitud con el embedding supere el umbral se reporta. Con **varios prompts** las regiones compiten entre sí, y ese es un régimen de decisión distinto, no simplemente "lo mismo con más clases".

## El resultado principal

Los cuatro casos ejecutados, con el mismo tracker y la misma configuración que la Actividad 1:

| Caso | Video | Prompts | IDs | Quemados | Dup. | Multiclase | conf mín | Llenado |
|---|---|---|---|---|---|---|---|---|
| **2a** objeto único | `one-by-one-person` | `["person"]` | 2 | 1 | 0 | 0 | 0,379 | 0,536 |
| **2b** objeto múltiple | `store-aisle` (~15 fps) | `["person"]` | 7 | **0** | 0 | 0 | 0,296 | 0,526 |
| **2c** multiprompt | `person-bicycle-car` | `["person","bicycle","car"]` | 4 | **0** | 0 | 0 | 0,306 | 0,570 |
| **bonus** idénticos | `bottle-detection` | `["bottle"]` | 4 | **0** | 0 | 0 | **0,696** | **0,761** |

*(«Llenado» = área de la máscara ÷ área de la caja. Una persona de pie llena típicamente entre 30 % y 50 %.)*

La comparación que importa es contra la Actividad 1 sobre **la misma escena**, cambiando únicamente el detector:

| | `person-bicycle-car` con YOLO26 (80 clases) | con YOLO-World (3 prompts) |
|---|---|---|
| Identidades emitidas | 15 | **4** |
| Nacimientos espurios | 36 | **0** |
| Cajas duplicadas | 13 | **0** |
| Tracks con clase inestable | 5 | **0** |
| Cobertura de los tracks | 84,6 – 100 % | **100 % en los cuatro** |
| Clases alucinadas | `boat`, `cell phone`, `sports ball`, `tennis racket` | **ninguna** |

![Varias personas simultáneas en un pasillo de tienda, cada una con máscara e identificador propios](/laboratorios/lab-42/store-multiple.jpg)

Y la descomposición semántica sale correcta: `id:39 bicycle` (f313-350) e `id:40 person` (f327-349) **coexisten** en tiempo y espacio. La bicicleta y su ciclista quedan como dos identidades separadas, mientras YOLO26 los colapsaba en un único track que oscilaba entre tres etiquetas.

## La métrica que lo explica: la confianza mínima

`model.track()` fuerza `conf = 0,1` en ambos pipelines. Solo uno llena ese rango.

| Pipeline | conf mínima observada |
|---|---|
| YOLO26 (los tres videos de la Actividad 1) | 0,100 – 0,108 |
| YOLO-World 2a / 2b / 2c | **0,379 / 0,296 / 0,306** |
| YOLO-World bonus | **0,696** |

{{< concept-alert type="clave" >}}
**YOLO-World nunca baja de ~0,30.** Clasificar por similitud coseno contra un embedding de texto produce puntajes mucho más estables en el régimen bajo que un softmax sobre 80 clases con fondo: la similitud entre un parche de piso vacío y el concepto «persona» es baja *y constante*, mientras un clasificador multiclase puede producir picos espurios en cualquiera de sus 80 salidas.

Menos ruido en el rango bajo → menos nacimientos espurios → menos competidores capaces de expropiar un track. Es la misma cadena causal del [ID switch](../02-anatomia-de-un-id-switch), atacada desde el otro extremo.
{{< /concept-alert >}}

## SAM 2 usado como si fuera SAM 1

![Las máscaras de SAM sobre people-detection: siluetas precisas, incluida la falda a rayas](/laboratorios/lab-42/sam-people.jpg)

Las máscaras son excelentes. En **1.790 detecciones** no hubo **ninguna máscara vacía ni ninguna pobre** (por debajo del 15 % de llenado), y el llenado medio está en rango sano en los cuatro casos. La falda a rayas del frame 200 aparece recortada con precisión de píxel, incluyendo el hueco entre las piernas.

Y sin embargo, el pipeline desaprovecha justo lo que hace especial a este modelo.

SAM 2 es explícitamente un modelo de **video**: sobre SAM 1 agrega un *memory encoder*, un *memory bank*, *memory attention* y una **cabeza de oclusión**, con los que propaga una máscara a lo largo del video desde un único prompt inicial. En `ultralytics/models/sam/model.py`:

```python
@property
def task_map(self):
    return {"segment": {"predictor": SAM2Predictor if self.is_sam2 else ...}}
```

`SAM2Predictor` es el predictor **de imágenes**. `SAM2VideoPredictor` —la clase que implementa el banco de memoria— **nunca se selecciona por esa ruta**; requiere instanciación explícita. Como el laboratorio llama al modelo con un frame suelto y cajas nuevas en cada iteración, **cada frame se segmenta como una fotografía aislada, sin ninguna noción de que existió un frame anterior**.

{{< concept-alert type="clave" >}}
**El memory bank, la memory attention y la cabeza de oclusión no se ejecutan en ningún momento del laboratorio.**

El seguimiento —la asignación y el mantenimiento de identidades— lo hace íntegramente BoT-SORT, con el mismo `botsort.yaml` por defecto, el mismo `with_reid: False` y el mismo `new_track_thresh: 0.25` que produjeron el ID switch. SAM 2 solo pinta píxeles dentro de cajas que otro decidió.

La definición de seguimiento que da la clase pone en el centro la **recuperación de la asociación ante oclusión**. En este pipeline esa recuperación sigue recayendo sobre un filtro de Kalman.
{{< /concept-alert >}}

La evidencia empírica lo confirma: el falso negativo de 11 frames en `people-detection` —una persona plenamente visible desde el frame 344 que no se detecta hasta el 355— ocurre **en el mismo punto exacto** con YOLO26 y con YOLO-World + SAM 2. Dos detectores sin relación arquitectónica, mismo error. Ninguno de los dos pipelines tiene forma de sostener el objeto durante la oclusión más allá de la extrapolación lineal.

![El falso negativo: la persona del sombrero, plenamente visible en f352, no recibe caja hasta f360](/laboratorios/lab-42/falso-negativo.jpg)

Un pipeline que sí usara SAM 2 sería otro paradigma: **seguimiento por propagación de máscara**, sin detector por frame, sin asociación por IoU y sin filtro de Kalman. Más cercano a los modelos integrados con que la clase cierra.

## El costo de segmentar

Los tiempos parciales de la ejecución son irregulares —12 s, +5, +11, +13, +4, +8— y esa irregularidad es informativa: los tramos rápidos son los frames vacíos. Ajustando un modelo de dos costos (frames con objetos frente a frames sin ellos) a los seis tramos medidos:

| Tramo (frames) | Vacíos | Con objetos | Tiempo real | Modelo |
|---|---|---|---|---|
| 0–100 | 44 | 56 | 12,0 s | 12,4 s |
| 100–200 | 81 | 19 | 5,0 s | 5,0 s |
| 200–300 | 49 | 51 | 11,0 s | 11,4 s |
| 300–400 | 38 | 62 | 13,0 s | 13,6 s |
| 400–500 | 86 | 14 | 4,0 s | 4,0 s |
| 500–596 | 62 | 34 | 7,9 s | 7,9 s |

$$\text{frame vacío} \approx 12\ \text{ms} \qquad\qquad \text{frame con objetos} \approx 212\ \text{ms}$$

**SAM 2 cuesta unas 17 veces lo que cuesta el detector.** El reparto encaja con la arquitectura: el encoder de imagen corre una vez por frame a 1024 × 1024 —Ultralytics fuerza `imgsz: 1024` y `retina_masks: True`— y es la parte dominante; el decodificador de máscaras corre una vez por caja y es barato. Por eso el costo escala con *si hay objetos* y no tanto con *cuántos*.

Traducido a la restricción de tiempo real que la clase enumera entre los desafíos: **4,7 fps con segmentación contra ~40 fps sin ella**. La máscara a nivel de píxel es lo que saca al pipeline del tiempo real.

## Un detalle de renderizado con consecuencias

El código del laboratorio dibuja así:

```python
annotated_frame = sam_results.plot(boxes=False)          # máscaras
annotated_frame = yolo_results.plot(img=annotated_frame) # cajas e identificadores encima
```

`sam_results` **no tiene identificadores**: SAM recibió un arreglo de cajas y devolvió un arreglo de máscaras, sin saber nada de identidades. Al colorear, `plot()` asigna color **por índice de máscara**, y ese índice viene del orden en que el detector entregó las detecciones — ordenado por puntaje descendente.

La consecuencia es falsable: **si dos personas intercambian su orden de confianza entre dos frames, sus máscaras intercambian color aunque sus identificadores no hayan cambiado.** Un parpadeo visual que parece un ID switch y no lo es. El arreglo es colorear por `track_id` en vez de por índice, que es lo que hace la versión instrumentada del notebook.

## El caso de los objetos idénticos

![Tres botellas idénticas con identificadores estables 41, 42 y 43](/laboratorios/lab-42/botellas.jpg)

El caso bonus invierte uno de los desafíos que la clase enumera. Entre ellos están las *variaciones intra-clase*; aquí la variación es prácticamente **cero**: tres botellas del mismo modelo, dos con líquido azul y una transparente. Ningún descriptor de apariencia podría separarlas de forma fiable.

Resultado: **4 identidades, 0 duplicados, 0 nacimientos espurios, 100 % de cobertura**, con la confianza mínima más alta de todo el trabajo (0,696) y las mejores máscaras (llenado 0,761).

Conviene, sin embargo, ser preciso sobre lo que este caso demuestra. En el tramo procesado **las botellas están prácticamente estáticas** sobre una superficie: no hay movimiento que predecir, y el modelo de velocidad constante tiene el trabajo más fácil posible. Que no haya errores de identidad es lo esperable, no una victoria del algoritmo.

Lo que sí demuestra es la otra mitad: la **detección** es impecable en este régimen —confianza mínima de 0,696 frente a los 0,10 de YOLO26 en la Actividad 1— y la **segmentación** también. Es el caso de control que muestra cómo se ve el pipeline cuando ninguna de sus etapas está bajo presión.

Para poner a prueba de verdad la hipótesis de que la geometría basta sin apariencia haría falta el mismo tipo de objetos **en movimiento y cruzándose**, que es el escenario que la clase menciona al hablar de oclusiones prolongadas.

---

**Siguiente:** [Los defectos del notebook](../07-los-defectos-del-notebook) — nueve cosas que romperían la actividad, y sus arreglos.
