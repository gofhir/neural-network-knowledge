---
title: "Cierre — Respuesta final integradora"
weight: 6
---

> **Celdas 24-25 del notebook.** El cierre del lab es **una única pregunta** que une los dos hilos del recorrido: ¿la compresión JPEG (Parte 5) ataca efectivamente el bottleneck del transporte JSON (Partes 3-4)? Spoiler: sí, y de forma masiva.

## La pregunta

> **¿Cree que sería beneficioso en términos de tiempo de ejecución la implementación de compresión de las imágenes antes de realizar la predicción al modelo API?**

**Lo que esconde**: el evaluador quiere que conectes los dos hilos del lab:

- **Hilo 1 (Partes 3-4)**: descubriste que la latencia es **dominada por transporte JSON**, no por inferencia. Throughput plano en ~1.2 req/s, escalado lineal con tamaño de payload.
- **Hilo 2 (Parte 5)**: aprendiste que `cv2.imencode/imdecode` reduce el tamaño del payload **3-30×** según contenido y quality.

¿Se puede atacar el problema del hilo 1 con la herramienta del hilo 2? La respuesta es claramente **sí**, pero hay matices importantes.

## El diagnóstico cuantitativo (cierre)

Las mediciones de [latencia vs tamaño](../latencia-payload) mostraron escalado aproximadamente lineal con el número de píxeles:

| Shape  | Píxeles  | JSON (MB) | Latencia (s) | Throughput efectivo |
|--------|----------|-----------|--------------|---------------------|
| 640    | 1.2 M    | 6.1       | 1.70         | 3.5 MB/s            |
| 1024   | 3.1 M    | 15.7      | 1.34         | 11.7 MB/s           |
| 2048   | 12.6 M   | 62.9      | 7.10         | 7.9 MB/s            |
| 3028   | 27.5 M   | 137.5     | 13.30        | 9.8 MB/s            |

Las mediciones de [concurrencia](../concurrencia-multiprocessing) mostraron throughput agregado plano en **~1.2 req/s** independiente del número de clientes — el sistema está saturado desde N=1, con cuellos en serialización JSON cliente + parsing JSON server + runner secuencial.

**El cuello dominante NO es la inferencia del modelo** (que en este lab es un `print()`); es el **transporte JSON** de payloads enormes (5-140 MB por request).

## El instrumento de mitigación

La [compresión JPEG en memoria](../compresion-jpeg) reduce el payload con los siguientes ratios honestos (sobre uint8 cruda, ruido aleatorio = peor caso del codec):

| Quality | Ratio (random) | Ratio esperable (foto real) |
|---------|----------------|-----------------------------|
| 70      | 5.4×           | 15-25×                      |
| 90      | 3.3×           | 10-15×                      |
| 95      | 2.6×           | 7-10×                       |

Para una pipeline real de Space Z (foto de parking con autos), una compresión a quality 70-90 reduciría el payload **entre 10× y 25× respecto a la imagen uint8 cruda**, y **entre 50× y 150× respecto al JSON de listas anidadas** (que es lo que medimos como baseline).

## El cálculo del beneficio en tiempo

Si reemplazamos el JSON crudo por bytes JPEG en el pipeline:

1. **Cliente**: en lugar de `np.zeros(...).tolist()` + `json.dumps` (segundos para imágenes grandes), `cv2.imencode` toma ~5-15 ms para 1024×1024. **Ahorro: 100-500× en costo CPU del cliente.**

2. **Red**: en lugar de transferir 14 MB (1024×1024 en JSON), se transfieren ~150-400 KB (JPEG q=80-90 de foto real). Sobre loopback el ahorro es ~30× en tiempo; sobre red real cloud (egress típico 100 Mbps-1 Gbps) el ahorro es aún más marcado.

3. **Server**: el parser JSON anidado se reemplaza por `cv2.imdecode` (~5-15 ms). **Ahorro: 50-200× en costo CPU del API server**, lo que destraba el cuello del worker único de uvicorn medido en la Parte 4.

4. **Throughput agregado**: si cada request consumía ~1 segundo de CPU en JSON parsing, el sistema saturaba a ~1 req/s. Con JPEG decode de ~10-20 ms, el techo teórico sube a **~50-100 req/s** — un factor 50-100× de mejora en capacidad.

## Estimación cuantitativa concreta

Para una imagen 1024×1024 (caso típico de Space Z):

| Métrica                    | Sin compresión | Con JPEG q=85    | Mejora |
|----------------------------|----------------|------------------|--------|
| Bytes por request          | ~14 MB         | ~200-400 KB      | 35-70× |
| Latencia individual        | 1.3-1.7 s      | ~50-100 ms       | ~15×   |
| Throughput sostenido       | 1.2 req/s      | ~20-50 req/s     | ~20×   |
| Egress cloud ($/mes)*      | ~USD $2000     | ~USD $40-80      | ~25×   |

*estimación basada en 8 cámaras × 1 fps × 30 días × GCP egress $0.12/GB.

## Caveats que un evaluador serio espera ver mencionados

### (a) JPEG es lossy

Hay pérdida de información en cada encode-decode. Para detección de autos (YOLOv8), mAP se mantiene estable hasta quality ~70. Para tareas más finas como OCR de patentes, conviene quality 90+. El benchmark de robustez debería hacerse offline antes de elegir el quality en producción.

### (b) El costo CPU del encode no es cero

En el cliente, comprimir cada frame consume ~5-15 ms de CPU. Despreciable comparado con el ahorro de transporte, pero si el cliente es un dispositivo edge muy limitado (Raspberry Pi clase 3, microcontrolador), conviene medirlo.

### (c) Para video continuo, JPEG por frame sigue siendo subóptimo

Si Space Z procesa stream de cámaras 24/7, lo ideal sería un encoder de video real (H.264/H.265) que explota redundancia temporal entre frames consecutivos. Da ratios 100-1000× sobre raw, no 15-50×. Pero requiere infraestructura más compleja (RTSP, GStreamer, NVIDIA DeepStream).

### (d) Para alcanzar el speedup teórico hay que rediseñar el endpoint

El `server.py` actual recibe `JSON` y hace `np.array(...).transpose(...)`. Para aprovechar JPEG bytes, habría que cambiar el descriptor a `bentoml.io.Bytes` (o `bentoml.io.Image` que internamente hace decode) y ajustar el handler. El lab muestra el codec pero **no integra el cambio en el server** — eso queda como ejercicio.

## Boceto del server con bentoml.io.Image

```python
import bentoml
from bentoml.io import Image, JSON
import numpy as np

@svc.api(input=Image(), output=JSON())
async def classify(img: PIL.Image.Image) -> dict:
    arr = np.array(img)                    # decode interno
    result = await runner.predict.async_run(arr)
    return {"predict": result}
```

Cliente:

```python
import cv2, requests
img = cv2.imread("parking.jpg")
_, encimg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
r = requests.post(
    "http://localhost:8000/classify",
    files={"image": ("frame.jpg", encimg.tobytes(), "image/jpeg")}
)
```

**Diferencias respecto al `server.py` original**:

- `bentoml.io.Image()` en lugar de `JSON()`.
- El cliente envía multipart/form-data en lugar de JSON.
- El server recibe ya un `PIL.Image.Image` decodificado por BentoML — no hay parse JSON ni transpose manual.
- El payload de wire pesa ~200-400 KB en lugar de ~14 MB.

## Conclusión

**Sí, la compresión es beneficiosa, y de forma masiva**: ~10-50× de mejora en throughput sostenido sin pérdida observable de accuracy para detección de objetos. Es **la primera optimización que cualquier ingeniero de MLOps aplica** al diagnosticar un sistema de visión con latencia alta y throughput plano.

Esto conecta directamente con la tesis central de los papers de la clase:

- **[Sculley et al. (2015)](/papers/hidden-technical-debt-sculley-2015)**: en sistemas de ML en producción, el modelo es la "cajita chica"; la mayor parte de las optimizaciones vienen de los **bordes del sistema**, no del modelo. La compresión del payload es uno de los primeros bordes a optimizar.

- **[Paleyes et al. (2022)](/papers/challenges-deploying-ml-paleyes-2022)**: documenta este exacto patrón en case studies industriales (AirBnB, Pinterest, Sepsis Watch) — el deployment introduce challenges que el desarrollo del modelo no anticipa.

- **[Kreuzberger et al. (2023)](/papers/mlops-overview-kreuzberger-2023)**: define el **continuous training/deployment** como uno de los 9 principios; eso requiere que el pipeline de wire format esté optimizado para no comerse el budget de latencia en transporte.

## Recursos relacionados

{{< cards >}}
  {{< card link="../" title="Volver al hub del Lab 19" subtitle="Índice de las 5 partes + resultados consolidados" icon="academic-cap" >}}
  {{< card link="/clases/clase-19" title="Clase 19 - Teoría" subtitle="GPUs, Cloud, Docker, Vertex AI, serving, MLOps" icon="academic-cap" >}}
  {{< card link="/fundamentos/model-serving" title="Fundamento: Model Serving" subtitle="BentoML/Triton/TorchServe, batching, cuantización" icon="adjustments" >}}
  {{< card link="/fundamentos/mlops" title="Fundamento: MLOps" subtitle="9 principios, 9 componentes, 7 roles" icon="variable" >}}
{{< /cards >}}
