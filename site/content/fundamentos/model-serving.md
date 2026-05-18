---
title: "Model Serving"
weight: 298
math: true
---

**Model serving** es la disciplina de **exponer modelos ML a usuarios reales** — humanos via web/app, o sistemas via APIs. Es la frontera entre el modelo entrenado y el valor de negocio. Las decisiones de serving (latencia, throughput, costo, formato) son tan determinantes para el exito como la calidad del modelo mismo.

Este fundamento cubre: patrones de serving (embedded/online/async/streaming), optimizacion de modelos (cuantizacion, pruning, distillation, ONNX, TensorRT), comparativa de frameworks (FastAPI, TF Serving, TorchServe, Triton, BentoML, KServe), y trade-offs latencia vs throughput.

---

## 1. Patrones de serving — taxonomia

| Patron | Latencia objetivo | Throughput | Caso de uso |
|---|---|---|---|
| **Embedded** | <1 ms | N/A (1 user) | Mobile (TFLite, CoreML), edge (Jetson, OAK) |
| **Online sync** (REST/gRPC) | 10-500 ms | 100-10k QPS | Chat, recommendation realtime, search |
| **Async with queue** | seg-min | alto | Generacion de imagen/video, batch scoring largo |
| **Batch offline** | min-h | masivo (millones) | Scoring nocturno, embeddings masivos |
| **Streaming** | continuo | continuo | Fraud detection, IoT, log analysis |

{{< concept-alert type="clave" >}}
**Latencia vs throughput no es trade-off lineal.** Batching (procesar varias requests juntas) **aumenta throughput** pero **incrementa latencia** de la primera request del batch. Es por eso que Triton tiene dynamic batching configurable con `max_queue_delay_microseconds`.
{{< /concept-alert >}}

---

## 2. Optimizacion de modelos

Hacer el modelo mas rapido/pequeno **sin entrenar de nuevo**:

| Tecnica | Reduccion size | Speedup | Impacto accuracy |
|---|---|---|---|
| **Cuantizacion INT8** | 4× | 2-4× | <1% |
| **Cuantizacion FP16** | 2× | 1.5-2× | ~0% |
| **Pruning estructurado 50%** | 2× | 1.5-2× | 1-3% |
| **Distillation** (student smaller) | 5-10× | 5-10× | 1-5% |
| **Compilacion ONNX + TensorRT** | 1× | 2-5× | 0% (mismo modelo) |

### 2.1 Cuantizacion — la mas usada

**Idea:** representar pesos y activaciones con menos bits (FP32 → INT8). Reduce memoria 4× y acelera inferencia en hardware con soporte INT8 (Tensor cores).

**Dos modalidades:**

- **Post-Training Quantization (PTQ)**: cuantizar despues de entrenar. Simple, sin re-entrenar. Pierde accuracy.
- **Quantization-Aware Training (QAT)**: entrenar simulando INT8. Mejor accuracy. Mas trabajo.

PyTorch dynamic quantization (PTQ simple):
```python
import torch.ao.quantization as quant
model_int8 = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

TFLite INT8 con representative dataset:
```python
converter = tf.lite.TFLiteConverter.from_saved_model('model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = repr_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()
```

JAX via [AQT (Accurate Quantized Training)](https://github.com/google/aqt):
```python
from aqt.jax.v2 import config
quant_cfg = config.fully_quantized(fwd_bits=8, bwd_bits=8)
```

### 2.2 Knowledge distillation

Entrenar un **modelo chico (student)** para imitar las predicciones de un **modelo grande (teacher)**. La idea original es de **Hinton et al. (2015)**, "Distilling the Knowledge in a Neural Network".

Loss tipica:
$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{CE}}(y_{\text{student}}, y_{\text{true}}) + (1 - \alpha) \cdot T^2 \cdot \text{KL}(p_{\text{student}}^T \| p_{\text{teacher}}^T)
$$

donde $T$ es la **temperature** que suaviza los logits del teacher.

Casos famosos: **DistilBERT** (40% mas chico que BERT, 60% mas rapido, 97% del performance), **TinyBERT**, **MobileBERT**.

### 2.3 Compilacion a formatos eficientes

**PyTorch → ONNX → TensorRT** es el pipeline canonico:

```python
# PyTorch → ONNX
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}},
    opset_version=17,
)

# ONNX → TensorRT (CLI)
# trtexec --onnx=model.onnx --fp16 --saveEngine=model.trt
```

**TensorFlow → SavedModel → TensorRT** equivalente.

Formatos de runtime:

| Formato | Hardware | Frameworks |
|---|---|---|
| **TorchScript** | CPU/GPU NVIDIA | PyTorch |
| **SavedModel** | CPU/GPU NVIDIA, TPU | TensorFlow |
| **ONNX** | CPU/GPU multi-vendor | Cross-framework |
| **TensorRT** | NVIDIA GPUs | Optimized inference |
| **OpenVINO** | Intel CPU/iGPU/VPU | Intel hardware |
| **CoreML** | Apple Neural Engine | iOS/macOS |
| **TFLite** | Mobile/embedded | Android, edge devices |

---

## 3. Frameworks de serving

### 3.1 Comparativa

| Framework | Strengths | Weaknesses | Cuando |
|---|---|---|---|
| **FastAPI + torch** | Maximo control, Python | Tu manejas batching, scaling | Prototipos, casos custom |
| **TF Serving** | Battle-tested, gRPC, versioning | Solo TF SavedModel | Stack TF |
| **TorchServe** | PyTorch nativo, batching, metricas | Menos maduro | Stack PyTorch |
| **NVIDIA Triton** | Multi-framework (TF/PT/ONNX/TRT/JAX), GPU optimizado, **dynamic batching** | Config compleja | Production seria, GPU |
| **BentoML** | Pythonic, facil empaquetar, multi-cloud | Capa adicional, menos perf que Triton | Iteracion rapida, MLOps |
| **KServe** (KFServing) | K8s-native, autoscaling, canary | Requiere K8s know-how | K8s nativo |
| **Ray Serve** | Composicion de modelos, scaling fino | Curva Ray | Pipelines complejos |
| **Seldon Core** | K8s-native, model graphs, explainability | Enterprise feel | Plataformas grandes |
| **MLflow Models** | Standard de packaging | Serving basico | Empaquetar, no servir alto QPS |

### 3.2 FastAPI + PyTorch — "hola mundo" production

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = torch.jit.load('model.pt', map_location='cuda').eval()
    yield
    del app.state.model
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class Req(BaseModel):
    instances: list[list[float]]

@app.post('/predict')
@torch.inference_mode()
async def predict(req: Req):
    x = torch.tensor(req.instances, device='cuda')
    logits = app.state.model(x)
    probs = torch.softmax(logits, dim=-1)
    return {
        'predictions': probs.argmax(dim=-1).cpu().tolist(),
        'probabilities': probs.cpu().tolist(),
    }

@app.get('/health')
def health():
    return {'cuda': torch.cuda.is_available()}
```

### 3.3 NVIDIA Triton — production GPU

`models/resnet50/config.pbtxt`:
```
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 64
input  [{ name: "input",  data_type: TYPE_FP32, dims: [3, 224, 224] }]
output [{ name: "output", data_type: TYPE_FP32, dims: [1000] }]
dynamic_batching {
  preferred_batch_size: [16, 32, 64]
  max_queue_delay_microseconds: 100
}
instance_group [{ count: 2, kind: KIND_GPU }]
```

**Dynamic batching** es la feature killer de Triton: coalescer requests asincronos en batches grandes — clave para alto QPS en GPU.

### 3.4 BentoML — el mas rapido para prototipar

```python
import bentoml
from bentoml.io import NumpyNdarray
import numpy as np

bentoml.pytorch.save_model('classifier', model,
                            signatures={'__call__': {'batchable': True}})

runner = bentoml.pytorch.get('classifier:latest').to_runner()
svc = bentoml.Service('my-classifier', runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def predict(x: np.ndarray) -> np.ndarray:
    return await runner.async_run(x)
```

```bash
bentoml serve service:svc --port 3000
bentoml build && bentoml containerize my-classifier:latest
```

Auto-genera Dockerfile + image lista para deployar.

---

## 4. Escalabilidad

### 4.1 Vertical vs horizontal

| | Vertical (scale up) | Horizontal (scale out) |
|---|---|---|
| Cambia | Tamano de instancia | Numero de instancias |
| Tope | GPU mas grande disponible | Solo presupuesto |
| Failover | No (single point) | Si |
| Estado | Stateful posible | Idealmente stateless |
| Para ML | GPU mas grande | Mas replicas detras de LB |

### 4.2 Autoscaling — senales

1. **CPU/GPU utilization** — simple, ruidoso para ML.
2. **Request rate (RPS) / concurrency** — usado por Cloud Run.
3. **Custom metric: queue depth** — el estandar oro.

K8s HPA con custom metric (ejemplo) en [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) seccion III.3.

### 4.3 Cold start — el problema serverless

Cloud Run / Lambda escalan a cero cuando no hay trafico → ahorro de costo. Pero cuando llega una request a cero replicas, el cold start agrega 1-30 s.

**Mitigaciones:**
- `min-instances=1` (cuesta plata).
- Lazy async load del modelo en `startup`.
- Modelo en imagen vs en GCS (imagen grande, GCS small image + descarga).
- Warm-up endpoint que precarga al deploy.

### 4.4 Online vs batch — costo

| | Online | Batch |
|---|---|---|
| GPU encendida | 24/7 | solo cuando corre el job |
| Latencia | ms | min-h |
| Costo/prediccion | alto | bajo (~100×) |

**Regla:** si **no necesitas** respuesta sincrona, **batch es 100× mas barato**. Ejemplo: scoring de embeddings de un catalogo de 10M productos → batch nocturno, no API REST.

---

## 5. Monitoring de serving

Que medir post-deploy:

### 5.1 Metricas operacionales
- **Latency** (p50, p95, p99) — los p99 cuentan.
- **Throughput** (RPS) — capacidad real.
- **Error rate** (5xx, timeouts).
- **GPU utilization, memory usage**.

### 5.2 Metricas de modelo
- **Prediction distribution** — alerta si cambia (predicting all 0s, etc.).
- **Confidence histogram** — drift de calibracion.
- **Feature distribution** (data drift via PSI/KL) — ver [Fundamento: MLOps](/fundamentos/mlops).
- **Label drift** cuando llegan labels reales con delay.

### 5.3 Stack tipico
- **Prometheus + Grafana** — metricas operacionales.
- **Evidently / WhyLabs / Arize / Fiddler** — drift y model performance.
- **Sentry / Cloud Error Reporting** — exceptions.
- **OpenTelemetry** — distributed tracing.

---

## 6. Conexion con el resto del site

Este fundamento se cita en:

- [Clase 19 - Entrenamiento, Deployment y MLOps](/clases/clase-19) — slides 35-46.
- [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) — Parte III (5 patrones de codigo).
- [Fundamento: Cloud Computing](/fundamentos/cloud-computing) — Cloud Run vs Vertex AI Endpoints.
- [Fundamento: Docker y Containers](/fundamentos/docker-containers) — serving frameworks usan containers.
- [Fundamento: MLOps](/fundamentos/mlops) — serving es un componente del paradigma MLOps.

Y se relaciona con conceptos previos:

- [Foundation Models (Bommasani 2021)](/papers/foundation-models-bommasani-2021) — serving LLMs es problema diferente al de modelos clasicos.

---

## Lecturas

- **Hinton, Vinyals, Dean (2015)** — *Distilling the Knowledge in a Neural Network*. El paper origen de distillation.
- **Jacob et al. (2018)** — *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. Cuantizacion formal.
- **NVIDIA Triton Inference Server documentation**.
- **BentoML documentation** (docs.bentoml.org) — guia de getting started.
- **KServe** (kserve.github.io) — Kubernetes-native serving.
- **Chip Huyen, "Designing ML Systems" (O'Reilly 2022)** — capitulos 7-8 dedicados a deployment.
