---
title: "Parte 2 — Servidor BentoML"
weight: 2
---

> **Celdas 6-11 del notebook.** Construir el endpoint HTTP que la opción (b) de la Parte 1 requiere. BentoML como stack de serving sobre Starlette/Uvicorn con runners desacoplados por IPC.

## La idea central de BentoML

BentoML separa **dos preocupaciones** que en FastAPI puro están mezcladas:

- El **API server** (Starlette + Uvicorn) es **I/O-bound** y asíncrono. Maneja HTTP, JSON parsing, requests concurrentes.
- Los **Runners** son **CPU/GPU-bound** y secuenciales por defecto. Cargan el modelo en memoria, ejecutan el forward pass.

Los conecta vía **IPC** (Unix domain socket o TCP local). Esta separación permite escalar I/O y compute por separado: 1 API server + N runners en GPUs distintas.

Mira el [fundamento Model Serving](/fundamentos/model-serving) para la comparativa con TorchServe, TensorFlow Serving, Triton, KServe.

## El archivo `server.py`

La celda 8 escribe a disco un `server.py` con la definición del Service. Lo desglosamos por piezas.

### Imports

```python
import bentoml
from bentoml.io import JSON
import numpy as np
import torch
import cv2
import typing as t
```

- `bentoml`: el namespace principal (Service, Runnable, Runner).
- `bentoml.io.JSON`: el descriptor de I/O. Le dice a BentoML "este endpoint recibe JSON y devuelve JSON". Otros descriptors: `NumpyNdarray`, `Image`, `PandasDataFrame`, `File`.
- `numpy`: para convertir el JSON entrante a array.
- `torch`, `cv2`: **importados pero NO usados** — herencia de un template. Cargan ~1 GB de RAM por proceso del runner sin razón. En producción real esto sería un code smell.

### Modelo dummy como Runnable

```python
class modelo(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False
    def __init__(self):
        print("")
    @bentoml.Runnable.method(batchable=False)
    def predict(self, image):
        print("prediccion")
```

- **`Runnable`**: la abstracción que envuelve la lógica de inferencia. Se ejecuta en un **proceso aparte** del API server.
- **`SUPPORTED_RESOURCES = ("cpu",)`**: declara dónde puede correr. Para GPU sería `("nvidia.com/gpu",)`.
- **`SUPPORTS_CPU_MULTI_THREADING = False`**: indica que no es thread-safe → BentoML serializa llamadas dentro del proceso.
- **`__init__`**: se llama **una vez por runner process**, aquí cargarías los pesos del modelo. En el dummy es `print("")`.
- **`@Runnable.method(batchable=False)`**: registra el método. Con `batchable=True` BentoML agruparía requests automáticamente — optimización gigante para modelos en GPU (5-10× throughput). Para el dummy con `print` no aplica.

> **El predict dummy es deliberado**: el lab no quiere enseñar PyTorch (eso ya se vio antes), quiere enseñar **el envoltorio de serving**. Toda la latencia que mediremos en las partes siguientes viene del **stack HTTP + serialización**, no del cómputo del modelo. Es el "control" experimental.

### Wiring: Runner + Service

```python
runner = t.cast("RunnerImpl", bentoml.Runner(modelo, name="modelo"))
svc = bentoml.Service("example", runners=[runner])
```

- `bentoml.Runner(modelo, name="modelo")`: instancia la fachada cliente que el API server usa para hablar con el proceso del runnable por IPC.
- `bentoml.Service("example", runners=[runner])`: define el servicio. Un servicio puede tener múltiples runners (ej: OCR + NER en pipeline).
- El `t.cast("RunnerImpl", ...)` es un hint para type-checker que no hace nada en runtime — herencia de templates con typing estricto.

### Endpoint HTTP

```python
@svc.api(input=JSON(), output=JSON())
async def classify(input_series) -> np.ndarray:
    image = np.array(input_series["image"], dtype=np.float32).transpose(1, 2, 0)
    runner.predict.async_run(image)
    return {"predict": 200}
```

- `@svc.api(input=JSON(), output=JSON())`: registra `classify` como endpoint en `POST /classify`.
- `async def`: el handler corre dentro del event loop de Starlette. **Crítico**: no llamar funciones bloqueantes dentro.
- `np.array(...).transpose(1,2,0)`: reconstruye el array desde la lista Python anidada y **transpone CHW → HWC**. Esto sugiere que el cliente envía en **CHW** (convención PyTorch).
- `runner.predict.async_run(image)`: despacha el cómputo al runner por IPC sin bloquear el event loop.

> **Bug funcional en este código**: `async_run` devuelve un coroutine que **no es awaitado**. En el dummy no importa (el modelo es `print()`), pero en producción real cambiar `runner.predict.async_run(image)` por `result = await runner.predict.async_run(image)` y devolver `result` es la diferencia entre "endpoint funcional" y "endpoint que descarta la predicción".

> **Inconsistencia**: el type hint dice `-> np.ndarray` pero el return es `{"predict": 200}`. BentoML no hace runtime type checking sobre el return. El status 200 hardcodeado es **sentinel de éxito**, no resultado del modelo.

## Lanzamiento del server

```python
get_ipython().system_raw('BENTOML_PORT=8000 bentoml serve server:svc &')
```

- `system_raw`: variante "cruda" de `system()` — NO captura stdout (lo que evita saturar la celda con logs del server) y soporta `&` para background limpio.
- `BENTOML_PORT=8000`: env var inline shell. Cambia el default `3000` a `8000`.
- `bentoml serve server:svc`: invoca el CLI. La notación `server:svc` significa "importa el módulo `server` y busca `svc` dentro".
- `&`: fork al background. El shell desconecta el proceso del terminal.

### Qué pasa internamente al ejecutar `bentoml serve`

1. Carga el módulo `server.py` (los imports — torch, cv2 — toman ~1-3s).
2. Para cada runner spawnea un **proceso hijo** que carga el runnable. Llama `modelo.__init__`.
3. Levanta el **API server** usando Uvicorn (servidor ASGI) sobre Starlette en `BENTOML_PORT`.
4. Establece canal IPC entre API server y runner (Unix socket en Linux/Mac).
5. Registra endpoints estándar además del tuyo:
   - `POST /classify` (el nuestro).
   - `GET /` — Swagger UI con la doc OpenAPI.
   - `GET /healthz` — liveness probe (responde 200 OK).
   - `GET /livez` — readiness probe.
   - `GET /metrics` — métricas Prometheus (latencia P50/P99, request rate, GPU util).
6. Empieza el event loop.

**Lo que viene "gratis"** (`healthz`, `metrics`, Swagger, batching, runner IPC) es el **valor real** de BentoML sobre FastAPI puro. Armarlo a mano son tardes de código con bugs en el shutdown ordenado.

> **Gotcha**: este patrón **solo funciona limpio en Colab**, donde el kernel mantiene un shell de fondo. En JupyterLab moderno con clean shutdown, el proceso `bentoml serve` puede ser matado cuando termina el handler. En local: corre `bentoml serve` en otro terminal aparte.

## Test inicial

```python
from google.colab import output
time.sleep(10)
output.serve_kernel_port_as_iframe(8000)
pred = requests.post(
    "http://localhost:8000/classify/",
    json={"image": np.zeros((3, 640, 640)).tolist()}
)
```

- **`time.sleep(10)`**: espera arranque del server. **Magic number defensivo**. Mejor en producción: polling de `/healthz`:
  ```python
  for _ in range(20):
      try:
          if requests.get("http://localhost:8000/healthz", timeout=1).status_code == 200: break
      except Exception: pass
      time.sleep(0.5)
  ```
- **`output.serve_kernel_port_as_iframe(8000)`**: utilidad de Colab para mostrar la Swagger UI embebida.
- **El POST**: envía `(3, 640, 640)` de ceros en convención CHW como JSON.

## Resultado del test

| Métrica | Valor |
|---------|-------|
| HTTP status | 200 ✅ |
| Response body | `{"predict": 200}` |
| Latencia end-to-end | **334 ms** |

**Lectura de los 334 ms**:

- El modelo tarda **cero** (es un `print`). Los 334 ms son **puramente overhead de transporte**.
- Descomposición aproximada:
  - `np.zeros(...).tolist()` cliente: ~50-100 ms (1.2M ceros → listas anidadas).
  - `json.dumps` interno de `requests`: ~80-150 ms (~5 MB de texto).
  - TCP + HTTP a localhost: ~1-5 ms.
  - `np.array(...).transpose(...)` server: ~50-100 ms.
  - IPC al runner + return: ~5-10 ms.

**~80% del tiempo se va en serialización/deserialización JSON, no en cómputo.** Esa intuición es la que las partes 3-5 confirmarán y atacarán.

## Arquitectura final mental

```
┌────────────────────────────┐
│   Cliente (notebook)       │
│   requests.post(...)       │
└─────────────┬──────────────┘
              │ HTTP POST /classify
              │ Body: JSON
              ▼
┌────────────────────────────┐
│   API Server (Uvicorn)     │ ← event loop async
│   handler classify()       │
│   JSON deserialization     │
│   np.array + transpose     │
└─────────────┬──────────────┘
              │ IPC (Unix socket / TCP local)
              │ args pickled
              ▼
┌────────────────────────────┐
│   Runner Process           │ ← proceso separado, CPU-bound
│   modelo.predict(image)    │
│   (carga modelo en __init__)│
└────────────────────────────┘
```

## Siguiente

{{< cards >}}
  {{< card link="../latencia-payload" title="Parte 3 - Latencia vs payload" subtitle="Benchmark con 4 shapes × 3 trials, escalado lineal con píxeles" icon="academic-cap" >}}
{{< /cards >}}
