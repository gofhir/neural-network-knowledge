---
title: "Actividades: conceptual y multimodal"
weight: 4
---

Las actividades cierran el lab: dos preguntas conceptuales sobre los límites de los LLMs, y una parte práctica con un modelo de **visión-lenguaje** (Qwen3-VL) que culmina en extracción documental — el patrón más cercano al trabajo con FHIR.

## Actividad 1 — preguntas conceptuales

**Pregunta 1: otros 2 problemas de los contextos muy grandes.**

El mecanismo de self-attention compara cada token con todos los demás → una matriz de atención $n \times n$. De ahí salen dos problemas acoplados (además del "lost in the middle" que da el enunciado):

- **Costo computacional cuadrático ($O(n^2)$).** El cómputo crece con el cuadrado del largo: duplicar el contexto cuadruplica las operaciones. Con cientos de miles de tokens la inferencia se vuelve prohibitiva.
- **Memoria cuadrática / crecimiento del KV-cache.** Materializar la matriz $n \times n$ y mantener las claves/valores de todos los tokens agota la VRAM antes del límite teórico de contexto.

Ambos comparten raíz: la matriz $n \times n$. Por eso la **atención sparse** (calcular solo algunas interacciones) y la **atención lineal** ($O(n)$) se usan como mitigación.

**Pregunta 2: utilidad de deshabilitar el modo de razonamiento.**

El razonamiento tiene un costo que no toda tarea justifica:
- **Latencia y costo**: la traza `<think>` consume muchos tokens; para tareas simples es un desperdicio.
- **Evitar el over-thinking**: un modelo puede paralizarse deliberando (como se vio con el modelo base en el [bloque de tool use](01-uso-de-herramientas)).
- **Control y previsibilidad**: en producción, las trazas son ruido que hay que filtrar.

Los modelos híbridos permiten **elegir el esfuerzo según la dificultad** — razonar en tareas difíciles, responder directo en las simples.

## Actividad 2 — visión-lenguaje con Qwen3-VL

Un **Vision-Language Model** procesa imágenes y texto juntos: un encoder de visión convierte la imagen en tokens visuales que entran al mismo LLM. El patrón de uso es el **mensaje con contenido mixto** (`{"type": "image"}` + `{"type": "text"}`), procesado por el `AutoProcessor`.

### Sub-tarea 1: descripción de imagen

El modelo describió una imagen de comida en detalle y estructura, aunque con **imprecisiones** (confundió naranjas con "cerezas", inventó "lentejas") — esperable en un modelo de 2B. Cumple la tarea.

### Sub-tarea 2: detección con bounding boxes

El modelo `-Thinking` razonó sobre los objetos (deliberó si la cuchara es objeto separado) y generó **8 cajas** en formato `{"bbox_2d": [x1,y1,x2,y2], "label": ...}` con coordenadas normalizadas a **escala 0–1000** (el formato nativo de Qwen3-VL).

![Imagen de comida con 8 bounding boxes dibujadas: siete bowls de comida (arroz, sopa, naranjas, salteado, tomate con huevo, salsa roja) y una cuchara, cada uno con su rectángulo de color y etiqueta](/laboratorios/lab-34/bounding-boxes-comida.jpg)

{{< callout type="warning" >}}
**Trampa de formato:** el docstring de `plot_bounding_boxes` dice "[y1 x1 y2 x2]" pero el código real usa **[x1,y1,x2,y2] en escala 0–1000**. Es el error #1 en detección con VLMs: si las cajas salen descuadradas, casi siempre es confusión de orden o escala de coordenadas.
{{< /callout >}}

### Sub-tarea 3: extraer una boleta a JSON

La joya de la actividad, y la más cercana al trabajo con FHIR: **imagen de documento → prompt de extracción → JSON estructurado → validar**. El modelo leyó una boleta chilena real:

![Boleta electrónica chilena con RUT, número 120, fecha, y dos artículos (Coca-Cola y Fanta) con cantidades y precios](/laboratorios/lab-34/boleta.png)

Y produjo JSON válido:

```json
{
  "numero_boleta": "120",
  "fecha_venta": "2020-09-28",
  "articulos": [
    {"nombre": "COCA-COLA ORIGINAL 3", "unidades": 3.0, "precio_unitario": 2.1},
    {"nombre": "FANTA 15", "unidades": 1.0, "precio_unitario": 1.8}
  ]
}
```

## El hallazgo: error de separador de miles

Hay un **error sutil pero grave** en esa extracción. El modelo leyó `precio_unitario: 2.1` y `1.8`, pero la boleta dice **`2.100` y `1.800`** — que en Chile son **2100 y 1800 pesos** (el punto es **separador de miles**, no decimal). El total real es **$8.100** (3×2100 + 1×1800), no 8.1.

El modelo, con su lectura anglosajona (punto = decimal), "cuadra" a 8.1 (3×2.1 + 1×1.8) — internamente consistente, pero **con la escala 1000× equivocada**.

{{< callout type="warning" >}}
**Oro para el trabajo con documentos clínicos.** Es el error clásico de extracción con LLMs: **fallan en convenciones numéricas locales** (separador de miles/decimales, formatos de fecha, unidades). Un LLM entrenado mayormente en inglés asume la convención anglosajona. En extracción de documentos clínicos chilenos esto puede convertir "2.100 mg" en "2.1 mg" — un error de dosificación de 1000×. La lección: la extracción con LLM **siempre necesita validación de rangos/unidades post-hoc**, nunca confiar ciegamente en los números.
{{< /callout >}}

### La corrección: prompt + validación

La solución no es post-procesar con regex (frágil), sino **atacar la raíz por dos capas**:

1. **Instruir la convención local en el prompt**: *"en Chile el punto es separador de miles; '2.100' significa 2100, no 2.1"*. Conecta con el bloque de [optimización de prompt](03-optimizacion-de-prompt): mejorar el comportamiento sin tocar el modelo.
2. **Validación de sanidad**: comprobar que $\sum(\text{unidades} \times \text{precio}) = \text{total}$. Con la boleta: 3×2100 + 1×1800 = 8100 = total → ✅. Es la red de seguridad que detecta si la escala está mal.

Este es el **ciclo completo de ingeniería** que el lab quiere enseñar: extracción ingenua → detección del error → corrección vía prompt → validación automática. Vale más que un JSON "perfecto" de una pasada — es exactamente el flujo que aplicarías en FHIR: primera extracción, detectas escala rara en dosis/unidades, refinas el prompt con las convenciones del dominio, validas contra una regla de negocio.
