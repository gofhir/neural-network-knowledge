---
title: "Profundizacion - Math + Codigo + Arquitectura"
weight: 20
math: true
---

> Profundizacion riguroso de la clase 19. Cuatro partes: **(I) anatomia GPU + memoria**, **(II) distributed training** con codigo en PyTorch/TensorFlow/JAX, **(III) inferencia y deployment** con cinco patrones de serving, **(IV) MLOps formalizado** — pipelines KFP, drift detection con PSI/KL, retraining triggers. Es el complemento matematico-codigo al recorrido conceptual en [teoria](teoria).

---

## Parte I — Hardware GPU y presupuesto de memoria

### I.1 Jerarquia de una GPU NVIDIA moderna

Una GPU moderna se organiza jerarquicamente. Para una H100:

```
H100 (Hopper)
├── 132 SMs (Streaming Multiprocessors)
│   Cada SM contiene:
│   ├── 128 CUDA cores (FP32/INT32 SIMT)
│   ├── 4 Tensor cores Gen 4 (FP8/FP16/BF16/TF32 matmul)
│   ├── Warp schedulers (32 threads/warp, 4 warps activos)
│   ├── 256 KB Shared memory / L1 cache (configurable)
│   └── Register file (65.536 registros 32-bit)
├── 50 MB L2 cache global
├── 80 GB HBM3 a 3 TB/s
└── NVLink 4.0 (900 GB/s entre GPUs)
```

Total H100: 132 × 128 = **16.896 CUDA cores físicos** (el numero 14.592 del slide del prof corresponde a una variante con SMs desactivados; la SXM5 completa expone 16.896).

{{< concept-alert type="recordar" >}}
**SIMT (Single Instruction Multiple Threads):** los 32 threads de un warp ejecutan la **misma instruccion** sobre datos distintos. Branches divergentes serializan ejecucion → divergencia destruye performance.
{{< /concept-alert >}}

### I.2 Generaciones y formatos de precision

| Gen | Anio | Modelos | Precisiones nuevas | Para que |
|---|---|---|---|---|
| Pascal | 2016 | P100 | FP16 | Legacy |
| Volta | 2017 | V100 | Tensor cores FP16 (Gen 1) | Legacy training |
| Turing | 2018 | T4 | INT8 inferencia | Inference barata |
| Ampere | 2020 | A100, A10, A30 | **TF32, BF16**, sparsity 2:4 | Training estandar |
| Ada Lovelace | 2022 | L4, L40, L40S, RTX 4090 | **FP8** (E4M3, E5M2) | Inference LLM |
| Hopper | 2022 | H100, H200 | FP8 + **Transformer Engine** | Training LLM fronterizo |
| Blackwell | 2024 | B100, B200, GB200 | **FP4** + TE gen 2 | Training/inference fronterizo |

**Formatos numericos clave:**

| Formato | Bits exp | Bits mantisa | Rango | Cuando usar |
|---|---|---|---|---|
| FP32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | Baseline correcto, loss accumulation |
| TF32 | 8 | 10 | $\pm 3.4 \times 10^{38}$ | Drop-in FP32 sin loss scaling, ~2× speedup matmul |
| FP16 | 5 | 10 | $\pm 6.5 \times 10^{4}$ | Requiere loss scaling (underflow gradients) |
| **BF16** | 8 | 7 | $\pm 3.4 \times 10^{38}$ | **Mismo rango FP32, sin scaling**. Estandar moderno |
| FP8 (E4M3) | 4 | 3 | $\pm 448$ | Forward H100+ con Transformer Engine |
| FP8 (E5M2) | 5 | 2 | $\pm 5.7 \times 10^{4}$ | Backward H100+ |

### I.3 Memoria GPU durante training — la formula

Cuatro componentes contribuyen al consumo:

$$
M_{\text{train}} = \underbrace{4P}_{\text{weights}} + \underbrace{8P}_{\text{optim state (Adam)}} + \underbrace{2 \cdot B \cdot A \cdot 4}_{\text{activations + grads}} + \underbrace{B \cdot I}_{\text{input batch}} \; \text{[bytes]}
$$

donde:
- $P$ = numero de parametros del modelo
- $B$ = batch size
- $A$ = numero de activations intermedias (proporcional a depth × hidden_dim)
- $I$ = tamano en bytes de un input

**Ejemplo del prof Soto (slide 8 ImplementationTips.pdf):**

- $P = 100\text{M}$, $A = 10\text{M}$, $B = 128$, $I = 1\text{MB}$

$$
M_{\text{train}} = 100\text{M} \cdot 4 + 100\text{M} \cdot 4 \cdot 2 + 128 \cdot 10\text{M} \cdot 4 \cdot 2 + 128 \cdot 1\text{MB} \approx 11.5 \text{ GB}
$$

vs en **inferencia** (sin grads, sin optim state, batch=1):

$$
M_{\text{inf}} = 100\text{M} \cdot 4 + 1\text{MB} \approx 101 \text{ MB}
$$

Ratio training/inferencia: **~100×**. Es la razon por la que un modelo que entrena en H100 (80 GB) corre inferencia en L4 (24 GB) o incluso CPU.

{{< concept-alert type="clave" >}}
**Memoria de Adam = 2× memoria de weights.** Adam guarda primer y segundo momento (m y v) por parametro. Un modelo de 7 B params en BF16 (14 GB de weights) consume 56 GB solo de optim state en FP32. Por eso AdamW + gradient checkpointing + FSDP son obligatorios para LLMs.
{{< /concept-alert >}}

### I.4 ROI on-prem vs cloud — calculo concreto

**Comprar un H100 SXM** (Q1 2026):

- Hardware: 30.000 USD
- Infraestructura (servidor, cooling, redundancia): 5.000 USD
- **CapEx total: 35.000 USD**
- Energia: 700 W × 8.760 h × 0,15 USD/kWh = **920 USD/ano**
- Amortizacion: 3 anos → **costo total 3 anos ~37.700 USD**

**Misma carga en cloud** (GCP A3 con 8× H100):

| Modalidad | USD/hora por nodo | USD/ano nodo (100% uso) |
|---|---|---|
| On-demand | ~88 | ~770.000 |
| Committed 1 ano | ~50 | ~440.000 |
| Committed 3 anos | ~30 | ~263.000 |
| Spot/preemptible (fraccional 1 H100) | ~3 | ~26.000 |

**Regla practica:** on-prem gana si utilizacion sostenida > 30% y horizonte > 12 meses. Cloud gana en burst, experimentacion, presupuesto limitado.

Ver [Fundamento: GPU Hardware para ML](/fundamentos/gpu-hardware-ml) para detalles de cada generacion.

---

## Parte II — Distributed training

Cuando un modelo no cabe en una GPU o el training es lento, hay tres paradigmas:

| Paradigma | Que divide | Cuando |
|---|---|---|
| **Data parallelism (DP/DDP)** | El batch | Modelo cabe en 1 GPU |
| **Tensor/Model parallelism** | Pesos por layer | Modelo NO cabe en 1 GPU |
| **Pipeline parallelism** | Layers entre GPUs | Modelo enorme, batch pequeno |
| **ZeRO / FSDP** | Pesos + grads + optim states shardeados | LLM, mejor que DDP |

### II.1 Mixed precision en los 3 frameworks

**PyTorch (`torch.amp`):**
```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # solo para fp16; bf16 no necesita

for x, y in loader:
    x, y = x.cuda(), y.cuda()
    optimizer.zero_grad()
    with autocast(dtype=torch.bfloat16):
        logits = model(x)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**TensorFlow / Keras:**
```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_bfloat16')

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.AdamW(1e-4),
    loss='sparse_categorical_crossentropy',
)
model.fit(ds, epochs=10)
```

**JAX (Flax):**
```python
import jax, jax.numpy as jnp
from flax import linen as nn
import optax

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        x = nn.relu(x)
        return nn.Dense(10, dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)

@jax.jit
def loss_fn(params, x, y):
    logits = model.apply(params, x).astype(jnp.float32)  # upcast loss
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
```

JAX expone explicitamente `dtype` (compute) vs `param_dtype` (storage) — el patron canonico es **params en FP32, compute en BF16**.

### II.2 Data Parallel (DDP) en PyTorch

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = MyModel().cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

for epoch in range(epochs):
    sampler.set_epoch(epoch)  # importante para shuffle determinista entre procesos
    for x, y in loader:
        # training step normal
        ...
```

Lanzar:
```bash
torchrun --nproc_per_node=8 --nnodes=1 train.py
```

### II.3 FSDP — Fully Sharded Data Parallel (PyTorch)

Para LLMs grandes, DDP no alcanza porque cada GPU mantiene una copia completa de weights + grads + optim state. FSDP **shardea** todos esos componentes entre las GPUs.

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    model,
    auto_wrap_policy=transformer_auto_wrap_policy,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalente
    device_id=torch.cuda.current_device(),
)
```

FSDP = ZeRO-3 de DeepSpeed reimplementado natively en PyTorch.

### II.4 TensorFlow MultiWorkerMirroredStrategy

```python
import tensorflow as tf

strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.NCCL,
    )
)

with strategy.scope():
    model = build_model()
    model.compile(optimizer='adamw', loss='sparse_categorical_crossentropy')

model.fit(ds, epochs=10)
```

Configuracion via `TF_CONFIG` env var en cada worker:
```json
{"cluster": {"worker": ["host1:12345", "host2:12345"]},
 "task": {"type": "worker", "index": 0}}
```

### II.5 JAX `pjit` / `shard_map`

JAX brilla aqui: el sharding es **declarativo** via `PartitionSpec`, no imperativo.

```python
import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

# Topologia 8 GPUs como 2 ejes: 4 data × 2 model parallel
devices = mesh_utils.create_device_mesh((4, 2))
mesh = Mesh(devices, axis_names=('data', 'model'))

# Anota como se shardean params y data
params_sharding = NamedSharding(mesh, P(None, 'model'))    # weights por eje 'model'
data_sharding   = NamedSharding(mesh, P('data', None))     # batch por eje 'data'

@jax.jit
def train_step(params, batch):
    def loss_fn(p):
        return model.apply(p, batch['x']).mean()
    grads = jax.grad(loss_fn)(params)
    return jax.tree.map(lambda p, g: p - 1e-3 * g, params, grads)

# Coloca tensors con su sharding
params = jax.device_put(params, params_sharding)
batch  = jax.device_put(batch, data_sharding)
new_params = train_step(params, batch)
```

### II.6 Spot/preemptible + checkpoint resilience

Spot VMs son 60-91% mas baratas pero pueden ser preempted con 30 s de aviso. La estrategia: **checkpointear a GCS cada N pasos**.

```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

ckpt = ModelCheckpoint(
    dirpath='gs://my-bucket/exp42/ckpts/',  # bucket directamente
    every_n_train_steps=1000,
    save_top_k=3,
    monitor='val_loss',
)
trainer = pl.Trainer(callbacks=[ckpt], max_epochs=100)
trainer.fit(model, datamodule, ckpt_path='last')  # auto-resume desde ultimo ckpt
```

### II.7 Vertex AI Custom Training SDK

```python
from google.cloud import aiplatform

aiplatform.init(
    project='mi-proyecto',
    location='us-central1',
    staging_bucket='gs://mi-bucket',
)

job = aiplatform.CustomContainerTrainingJob(
    display_name='resnet50-imagenet-h100',
    container_uri='us-central1-docker.pkg.dev/mi-proyecto/imgs/train:v3',
    model_serving_container_image_uri='us-central1-docker.pkg.dev/mi-proyecto/imgs/serve:v3',
)

model = job.run(
    args=['--epochs=90', '--lr=0.1', '--data=gs://mi-bucket/imagenet/'],
    replica_count=1,
    machine_type='a3-highgpu-8g',           # 8× H100
    accelerator_type='NVIDIA_H100_80GB',
    accelerator_count=8,
    base_output_dir='gs://mi-bucket/runs/exp42/',
    enable_web_access=True,
)
```

---

## Parte III — Inferencia y deployment

### III.1 Optimizacion de modelo

Cuatro familias de tecnicas para hacer un modelo mas rapido/pequeno **sin entrenar de nuevo**:

| Tecnica | Reduccion tamano | Speedup | Impacto accuracy |
|---|---|---|---|
| Cuantizacion INT8 | 4× | 2-4× | <1% |
| Pruning estructurado 50% | 2× | 1.5-2× | 1-3% |
| Distillation (student smaller) | 5-10× | 5-10× | 1-5% |
| Compilacion a TensorRT | 1× | 2-5× | 0% |

**Cuantizacion dinamica en PyTorch:**
```python
import torch.ao.quantization as quant
model_int8 = quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

**Cuantizacion estatica (mejor accuracy, requiere calibracion):**
```python
model.qconfig = quant.get_default_qconfig('fbgemm')
prepared = quant.prepare(model)
for batch in calibration_loader:
    prepared(batch)
quantized = quant.convert(prepared)
```

**TFLite int8 con representative dataset:**
```python
converter = tf.lite.TFLiteConverter.from_saved_model('model/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = repr_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()
```

**PyTorch → ONNX → TensorRT:**
```python
torch.onnx.export(
    model, dummy_input, 'model.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}},
    opset_version=17,
)
# Luego trtexec --onnx=model.onnx --fp16 --saveEngine=model.trt
```

Ver [Fundamento: Model Serving](/fundamentos/model-serving) para profundizar.

### III.2 Cinco patrones de serving — codigo

#### A. FastAPI + PyTorch (el "hola mundo" production)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga modelo una vez al startup
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
    return {'predictions': probs.argmax(dim=-1).cpu().tolist(),
            'probabilities': probs.cpu().tolist()}

@app.get('/health')
def health():
    return {'status': 'ok', 'cuda': torch.cuda.is_available()}
```

```bash
uvicorn server:app --host 0.0.0.0 --port 8080 --workers 1
# 1 worker porque cada uno carga su copia del modelo en GPU
```

#### B. TF Serving (zero codigo server-side)

```python
model.save('models/my_model/1/')  # versionado por carpeta numerica

# Run server con Docker:
# docker run -p 8501:8501 -v $(pwd)/models:/models \
#   -e MODEL_NAME=my_model tensorflow/serving:latest-gpu

# Cliente:
import requests
r = requests.post(
    'http://localhost:8501/v1/models/my_model:predict',
    json={'instances': [[1.0, 2.0, 3.0]]},
)
```

#### C. JAX server con AOT compile

```python
import jax, jax.numpy as jnp
from flax.training import checkpoints
from fastapi import FastAPI

params = checkpoints.restore_checkpoint('ckpt/', target=None)
model = MyModel()

@jax.jit
def predict_fn(params, x):
    return jax.nn.softmax(model.apply(params, x))

# Warmup para forzar compilacion AOT con shape estable
_ = predict_fn(params, jnp.ones((1, 224, 224, 3))).block_until_ready()

app = FastAPI()

@app.post('/predict')
async def predict(req: dict):
    x = jnp.array(req['instances'])
    probs = predict_fn(params, x)
    return {'probabilities': probs.tolist()}
```

#### D. BentoML (el mas rapido para prototipar)

```python
# service.py
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

#### E. NVIDIA Triton (production seria)

`models/resnet50/config.pbtxt`:
```
name: "resnet50"
platform: "onnxruntime_onnx"
max_batch_size: 64
input [{ name: "input", data_type: TYPE_FP32, dims: [3, 224, 224] }]
output [{ name: "output", data_type: TYPE_FP32, dims: [1000] }]
dynamic_batching {
  preferred_batch_size: [16, 32, 64]
  max_queue_delay_microseconds: 100
}
instance_group [{ count: 2, kind: KIND_GPU }]
```

Triton coalescer requests asincronos en batches grandes (dynamic batching) y maneja multiples modelos por server. Es lo que GCP/AWS usan internamente para serving GPU optimizado.

### III.3 Autoscaling y cold start

#### Kubernetes HPA con custom metric

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Pods
      pods:
        metric:
          name: nvidia_gpu_utilization
        target:
          type: AverageValue
          averageValue: "70"
    - type: Pods
      pods:
        metric:
          name: inference_queue_depth
        target:
          type: AverageValue
          averageValue: "10"
```

Senales de escalado, en orden de utilidad creciente para ML:

1. **CPU/GPU utilization** — simple pero ruidoso (1 request larga → 100% utilization).
2. **Request rate (RPS) / concurrency** — usado por Cloud Run.
3. **Custom metric: queue depth** — el estandar oro. Cola crece → escala.

#### Cold start: 4 mitigaciones

```python
# 1. NO bloquear startup con carga sincrona del modelo
@app.on_event('startup')
async def startup():
    loop = asyncio.get_event_loop()
    app.state.model = await loop.run_in_executor(None, load_model)

# 2. Min instances para evitar scale-to-zero
# gcloud run deploy my-svc --min-instances=1

# 3. Model en imagen vs en GCS
# - En imagen: cold start ~5 s, image gigante
# - En GCS al startup: cold start ~30 s, image chica

# 4. Warm-up endpoint que precarga al deploy
@app.get('/warmup')
def warmup():
    dummy = torch.zeros((1, 3, 224, 224), device='cuda')
    _ = app.state.model(dummy)
    return {'status': 'warm'}
```

---

## Parte IV — MLOps formalizado

### IV.1 Google MLOps maturity levels

| Nivel | Descripcion | Estado tipico |
|---|---|---|
| **0** | Manual | Jupyter → email → ingeniero deploya. La mayoria de equipos. |
| **1** | ML pipeline automation | Pipeline automatizado, retraining trigger-eable. |
| **2** | CI/CD pipeline automation | Cambios de codigo → CI testea → CD redeploya. Production seria. |

Alineado con los 9 principios de [Kreuzberger 2023](/papers/mlops-overview-kreuzberger-2023): nivel 0 implementa 0 principios; nivel 1 implementa P2 + P6; nivel 2 implementa P1 + P2 + P6 + P9 al menos.

### IV.2 Drift detection — formal

[Sculley 2015](/papers/hidden-technical-debt-sculley-2015) lo plantea cualitativamente. [Paleyes 2022](/papers/challenges-deploying-ml-paleyes-2022) lo profundiza con case studies. La formalizacion estadistica:

**Tres tipos de drift:**

| Tipo | Que cambia | Notacion |
|---|---|---|
| **Covariate shift (data drift)** | $P(X)$ | $P_{\text{train}}(X) \neq P_{\text{prod}}(X)$ |
| **Concept drift** | $P(Y\|X)$ | La relacion misma cambia |
| **Label shift (prior shift)** | $P(Y)$ | La distribucion de clases cambia |

#### Population Stability Index (PSI) — estandar industrial

$$
\text{PSI} = \sum_{i=1}^{B} (a_i - e_i) \cdot \ln\left(\frac{a_i}{e_i}\right)
$$

donde $e_i, a_i$ son las fracciones esperadas (training) y actuales (produccion) en el bin $i$.

**Interpretacion:**
- $\text{PSI} < 0.1$: sin cambio significativo.
- $0.1 \leq \text{PSI} < 0.25$: cambio menor, monitorear.
- $\text{PSI} \geq 0.25$: cambio mayor, **retraining recomendado**.

```python
import numpy as np

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breaks = np.linspace(expected.min(), expected.max(), bins + 1)
    exp_pct = np.histogram(expected, breaks)[0] / len(expected)
    act_pct = np.histogram(actual,   breaks)[0] / len(actual)
    # evitar log(0)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
```

#### KL divergence

$$
D_{\text{KL}}(P \| Q) = \sum_x P(x) \ln \frac{P(x)}{Q(x)}
$$

Donde $P$ = produccion, $Q$ = training. **Asimetrica**. KS test es alternativa no parametrica.

### IV.3 Retraining triggers — tres patrones

| Patron | Senal | Pros | Cons |
|---|---|---|---|
| **Schedule-based** | "Cada 7 dias" | Simple, predecible | Malgasta compute si no hay drift; tarda si drift acelera |
| **Performance-based** | Metrica online < umbral | Reactivo a degradacion real | Requiere ground truth; lag hasta saber |
| **Drift-based** | PSI/KL > umbral | Proactivo, sin ground truth | Falsos positivos posibles |

Production seria combina los tres con votacion mayoritaria.

### IV.4 Pipeline KFP completo

Kubeflow Pipelines (KFP) ejecutado sobre Vertex AI Pipelines. Codigo end-to-end:

```python
from kfp import dsl, compiler
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics

@component(
    base_image='python:3.11',
    packages_to_install=['pandas==2.2.0', 'google-cloud-storage==2.14.0'],
)
def get_data(output_dataset: Output[Dataset], data_uri: str):
    import pandas as pd
    df = pd.read_csv(data_uri)
    df.to_parquet(output_dataset.path)
    output_dataset.metadata['n_rows'] = len(df)

@component(
    base_image='pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime',
    packages_to_install=['scikit-learn'],
)
def train(
    dataset: Input[Dataset], model: Output[Model], metrics: Output[Metrics],
    lr: float = 1e-3, epochs: int = 10,
):
    import torch, pandas as pd
    df = pd.read_parquet(dataset.path)
    # ... entrenamiento ...
    torch.save(net.state_dict(), model.path)
    metrics.log_metric('val_loss', val_loss)
    metrics.log_metric('val_acc', val_acc)

@component
def evaluate(model: Input[Model], dataset: Input[Dataset]) -> float:
    # devuelve metrica para condicion
    return 0.95

@component
def deploy(model: Input[Model], endpoint_name: str):
    from google.cloud import aiplatform
    aiplatform.Model.upload(
        display_name=endpoint_name,
        artifact_uri=model.uri,
        serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/pytorch-gpu.2-3:latest',
    ).deploy(
        machine_type='n1-standard-4',
        accelerator_type='NVIDIA_TESLA_T4',
        accelerator_count=1,
    )

@pipeline(name='train-eval-deploy', pipeline_root='gs://my-bucket/kfp/')
def my_pipeline(
    data_uri: str, lr: float = 1e-3, accuracy_threshold: float = 0.9,
):
    get_op = get_data(data_uri=data_uri)
    train_op = train(dataset=get_op.outputs['output_dataset'], lr=lr)
    eval_op = evaluate(
        model=train_op.outputs['model'],
        dataset=get_op.outputs['output_dataset'],
    )

    with dsl.Condition(eval_op.output >= accuracy_threshold, name='check-accuracy'):
        deploy(model=train_op.outputs['model'], endpoint_name='production')

# Compile + submit
compiler.Compiler().compile(my_pipeline, 'pipeline.yaml')

from google.cloud import aiplatform
aiplatform.PipelineJob(
    display_name='train-eval-deploy',
    template_path='pipeline.yaml',
    parameter_values={'data_uri': 'gs://my-bucket/data.csv', 'lr': 1e-4},
).submit()
```

Reproduce **exactamente** el screenshot del slide 62 del prof.

### IV.5 Experiment tracking inline (MLflow)

```python
import mlflow

mlflow.set_tracking_uri('http://mlflow.internal:5000')
mlflow.set_experiment('resnet50-imagenet')

with mlflow.start_run(run_name='lr-1e-3-bs-256'):
    mlflow.log_params({'lr': 1e-3, 'batch_size': 256, 'optimizer': 'adamw'})

    for epoch in range(epochs):
        train_loss = train_one_epoch(...)
        val_loss, val_acc = evaluate(...)
        mlflow.log_metrics({'train_loss': train_loss,
                            'val_loss': val_loss, 'val_acc': val_acc}, step=epoch)

    mlflow.pytorch.log_model(model, 'model', registered_model_name='resnet50')
    mlflow.log_artifact('confusion_matrix.png')
```

### IV.6 Stack MLOps moderno por capa

```
┌─────────────────────────────────────────────────────────────┐
│ Experiment tracking:  MLflow, W&B, Neptune, Comet, ClearML  │
├─────────────────────────────────────────────────────────────┤
│ Data versioning:      DVC, LakeFS, Pachyderm, Delta Lake    │
├─────────────────────────────────────────────────────────────┤
│ Feature store:        Feast, Tecton, Vertex Feature Store   │
├─────────────────────────────────────────────────────────────┤
│ Model registry:       MLflow Registry, Vertex Model Reg.    │
├─────────────────────────────────────────────────────────────┤
│ Orchestration:        Kubeflow, Airflow, Flyte, Prefect,    │
│                       Dagster, Argo Workflows               │
├─────────────────────────────────────────────────────────────┤
│ Serving:              Triton, TorchServe, TF Serving, KServe│
├─────────────────────────────────────────────────────────────┤
│ Monitoring:           Evidently, WhyLabs, Arize, Prometheus │
├─────────────────────────────────────────────────────────────┤
│ CI/CD:                GitHub Actions, GitLab CI, Tekton     │
└─────────────────────────────────────────────────────────────┘
```

Ver [Fundamento: MLOps](/fundamentos/mlops) para profundizar cada capa.

---

## Lecturas

- **Sculley et al. 2015** — el manifesto. [analisis local](/papers/hidden-technical-debt-sculley-2015).
- **Kreuzberger et al. 2023** — la arquitectura formal con 9 principios + 9 componentes. [analisis local](/papers/mlops-overview-kreuzberger-2023).
- **Paleyes et al. 2022** — survey con case studies industriales. [analisis local](/papers/challenges-deploying-ml-paleyes-2022).
- **Huyen, "Designing Machine Learning Systems"** (O'Reilly 2022) — libro de referencia practica.
- **Zinkevich, "Rules of ML"** (Google 2017) — 43 reglas operacionales derivadas del espiritu Sculley.
