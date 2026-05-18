---
title: "GPU Hardware para ML"
weight: 295
math: true
---

Una **GPU (Graphics Processing Unit)** es un procesador masivamente paralelo originalmente disenado para renderizar graficos. Desde 2012 (AlexNet entrenada en 2× GTX 580) se volvio el **acelerador estandar** para deep learning. Entender su anatomia, jerarquia de memoria y formatos de precision es la base para razonar sobre training a escala y deployment eficiente.

Este fundamento cubre: anatomia de una GPU NVIDIA moderna, generaciones (Pascal a Blackwell), formatos numericos, presupuesto de memoria, CPU vs GPU vs TPU.

---

## 1. Anatomia de una GPU NVIDIA

Una GPU se organiza jerarquicamente:

```
GPU
├── SMs (Streaming Multiprocessors) — la unidad de computo fisica
│   Cada SM contiene:
│   ├── CUDA cores                   — FP32/INT32 SIMT, 1 op/ciclo
│   ├── Tensor Cores                 — matmul fused (4×4 → 4×4)
│   ├── RT Cores                     — ray tracing (irrelevante ML)
│   ├── Warp schedulers (32 threads) — agrupa hilos para SIMT
│   ├── Shared memory / L1 cache
│   └── Register file
├── L2 cache global
├── HBM / GDDR (VRAM)
└── NVLink / PCIe                    — interconexion multi-GPU
```

### 1.1 CUDA cores vs Tensor cores

La distincion mas importante:

| Atributo | CUDA core | Tensor Core |
|---|---|---|
| Operacion | `a*b + c` escalar (FMA) | matriz pequena GEMM fused |
| Throughput | 1 op/ciclo | 64-256 ops/ciclo (segun gen) |
| Precision | FP64/FP32/FP16/INT32 | FP16/BF16/TF32/FP8/INT8/INT4 |
| Uso tipico | kernels custom | matmul, conv, attention |

{{< concept-alert type="clave" >}}
En un Transformer, >90% de los FLOPs son matmuls que viven en Tensor Cores. Un H100 con 14.592 CUDA cores triplica a un A100 con 6.920 CUDA — pero la razon principal es que sus Tensor Cores (456 vs 422) son **generacionalmente mas potentes**, no el conteo de CUDA cores.
{{< /concept-alert >}}

### 1.2 SIMT y warps

**SIMT (Single Instruction Multiple Threads):** los 32 threads de un warp ejecutan la **misma instruccion** sobre datos distintos. Si un branch hace que la mitad de threads tome `if` y la otra mitad `else`, ambas ramas se ejecutan secuencialmente — **divergence destruye performance**.

Implicancia para ML: las operaciones tensoriales (matmul, conv, elementwise) son **uniformes** y mapean perfecto a SIMT. Operaciones con control flow dependiente del dato (e.g., variable-length attention sin masking proper) son patologicas.

---

## 2. Generaciones NVIDIA

```
2016 Pascal  ──→ P100 ──→ FP16 basico
2017 Volta   ──→ V100 ──→ Tensor Cores Gen 1 (FP16)
2018 Turing  ──→ T4, RTX 20XX ──→ INT8 inferencia
2020 Ampere  ──→ A100, A10, A30 ──→ TF32, BF16, sparsity 2:4
2022 Ada     ──→ L4, L40, L40S, RTX 4090 ──→ FP8
2022 Hopper  ──→ H100, H200 ──→ FP8 + Transformer Engine, DPX
2024 Blackwell ──→ B100, B200, GB200 ──→ FP4 + TE Gen 2
```

### 2.1 Tabla comparativa (referencia Q1 2026)

Las cinco GPUs mas comunes en cloud:

| GPU | Gen | CUDA cores | Tensor cores | VRAM | Power | Precio USD | Cuando usar |
|---|---|---|---|---|---|---|---|
| T4 | Turing | 2.500 | 320 | 15 GB GDDR6 | 70 W | ~1.100 | Inferencia barata |
| L4 | Ada | 7.680 | 240 | 24 GB GDDR6 | 72 W | ~2.600 | Inferencia LLM |
| L40 | Ada | 18.176 | 568 | 48 GB GDDR6 | 300 W | ~8.400 | Inferencia grande, training mediano |
| A100 | Ampere | 6.920 | 422 | 80 GB HBM2e | 400 W | ~12.000 | Training general |
| H100 | Hopper | 14.592 | 456 | 80 GB HBM3 | 350 W | ~30.000 | Training LLM fronterizo |

---

## 3. Formatos numericos

| Formato | Bits exp | Bits mantisa | Rango | Cuando usar |
|---|---|---|---|---|
| FP64 | 11 | 52 | $\pm 1.8 \times 10^{308}$ | Cientifico, no ML |
| **FP32** | 8 | 23 | $\pm 3.4 \times 10^{38}$ | Baseline; loss accumulation |
| **TF32** | 8 | 10 | $\pm 3.4 \times 10^{38}$ | Drop-in FP32, 2× speedup matmul |
| FP16 | 5 | 10 | $\pm 6.5 \times 10^{4}$ | Requiere loss scaling (underflow) |
| **BF16** | 8 | 7 | $\pm 3.4 \times 10^{38}$ | **Sin scaling**. Estandar moderno |
| FP8 (E4M3) | 4 | 3 | $\pm 448$ | Forward Hopper+ con Transformer Engine |
| FP8 (E5M2) | 5 | 2 | $\pm 5.7 \times 10^{4}$ | Backward Hopper+ |
| INT8 | — | — | $[-128, 127]$ | Cuantizacion inferencia |
| FP4 | 2 | 1 | $\pm 12$ | Blackwell, inference extrema |

### 3.1 Reglas practicas

- **FP32** — baseline correctness, debugging.
- **TF32** (Ampere+) — activar con `torch.backends.cuda.matmul.allow_tf32 = True` para ~2× speedup matmul sin tocar el codigo.
- **FP16 mixed precision** — speedup 2-3× pero requiere **GradScaler** porque gradients pequenos underflowean. Riesgo: NaN.
- **BF16** — mismo rango que FP32 con menos precision en mantisa. **No requiere loss scaling**. Recomendado en Ampere+ y TPUs.
- **FP8** — solo Hopper+ con Transformer Engine de NVIDIA. Reduce memoria 50% vs BF16.

Codigo en PyTorch / TF / JAX en [Clase 19 - Profundizacion](/clases/clase-19/profundizacion).

---

## 4. Presupuesto de memoria GPU

Cuatro componentes contribuyen al consumo en training:

$$
M_{\text{train}} = \underbrace{4P}_{\text{weights}} + \underbrace{8P}_{\text{Adam optim state}} + \underbrace{8 \cdot B \cdot A}_{\text{activations + grads}} + \underbrace{B \cdot I}_{\text{input batch}}
$$

donde $P$ = params, $B$ = batch size, $A$ = activations intermedias, $I$ = tamano de input.

**Ejemplo (slide 8 Soto 2020):** $P = 100\text{M}$, $A = 10\text{M}$, $B = 128$, $I = 1\text{MB}$:

$$
M_{\text{train}} = 400 \text{ MB} + 800 \text{ MB} + 10.24 \text{ GB} + 128 \text{ MB} \approx 11.5 \text{ GB}
$$

vs **inferencia** (sin grads ni optim state, batch=1):

$$
M_{\text{inf}} = 400 \text{ MB} + 1 \text{ MB} \approx 101 \text{ MB}
$$

**Ratio training/inferencia: ~100×.** Por eso un modelo entrenado en H100 (80 GB) puede correr inferencia en L4 (24 GB) e incluso CPU.

### 4.1 Memoria de Adam es 2× memoria de weights

Adam mantiene **primer momento** (`m`) y **segundo momento** (`v`) por parametro, ambos en FP32:

$$
\text{mem}_{\text{Adam}} = P \cdot 4 \text{ bytes} \cdot 2 = 8P \text{ bytes}
$$

Un Llama 7B en BF16 tiene 14 GB de weights pero **56 GB de optim state**. Por eso entrenar 7B params naively requiere ~80 GB → necesitas H100 o **gradient checkpointing + FSDP**.

---

## 5. CPU vs GPU vs TPU

| Aspecto | CPU | GPU | TPU |
|---|---|---|---|
| Cores | 8-128 | 2.000-16.000 | matrix multiply units |
| Mem bandwidth | ~100 GB/s (DDR5) | 1-3 TB/s (HBM) | ~1 TB/s (HBM) |
| FP32 TFLOPS | 1-5 | 30-80 | 30-100 |
| Mejor para | logica condicional, IO | matmul, conv, parallel | matmul a escala TPU pod |
| Cuando usar ML | inferencia chica, preprocessing | training+inferencia general | training LLM, TPU pods |

**TPUs (Google):** chips ASIC para matmul. Optimas para TensorFlow / JAX. La ventaja real es el **TPU pod** — 1024+ TPUs interconectadas con red propia, para training de modelos masivos.

---

## 6. Edge / embedded GPUs

Para deployment fuera del data center:

| Device | Memoria | Power | Precio | Caso de uso |
|---|---|---|---|---|
| **NVIDIA Jetson Nano** | 2-4 GB | 5-10 W | $129 | Robotica hobbyist |
| **NVIDIA Jetson TX2** | 8 GB | 10-20 W | $400 | Dispositivos industriales |
| **NVIDIA Jetson Xavier** | 32 GB | 10-30 W | $900-1000 | Vision embarcada seria |
| **Google Coral USB** | 8 MB SRAM | 2 W | $60 | TensorFlow Lite int8 |
| **Intel Movidius (Neural Compute Stick)** | — | 1 W | $80 | Raspberry Pi + OpenVINO |
| **OpenCV AI Kit (OAK-D)** | — | 4-5 W | $149 | Stereo depth + DL |

Para mobile: **Apple Neural Engine** (iPhone/iPad), **Qualcomm Adreno** (Android), **ARM Mali**. Frameworks: CoreML (iOS), TFLite (Android), ONNX Runtime Mobile.

**Estrategia comun:** entrenar en cloud GPU → cuantizar a INT8 → deployar a edge con TFLite/CoreML/ONNX.

---

## 7. Conexion con el resto del site

Este fundamento se cita en:

- [Clase 19 - Entrenamiento, Deployment y MLOps](/clases/clase-19) — slide 7-8 (tabla GPUs).
- [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) — Parte I (memoria + ROI).
- [Fundamento: Cloud Computing](/fundamentos/cloud-computing) — pricing por hora segun GPU.
- [Fundamento: Model Serving](/fundamentos/model-serving) — eleccion de GPU para serving.

Y se relaciona con conceptos previos del diplomado:

- [Fundamento: Foundation Models](/fundamentos/foundation-models) — escala de modelos modernos.
- [AlexNet (Krizhevsky 2012)](/papers/alexnet-krizhevsky-2012) — el primer training serio multi-GPU.
- [Large-batch SGD (Goyal 2017)](/papers/large-minibatch-sgd-goyal-2017) — escalar training a multi-GPU.

---

## Lecturas

- **NVIDIA H100 Tensor Core GPU Architecture** (whitepaper oficial NVIDIA, 2022).
- **NVIDIA Ada Lovelace Architecture Whitepaper** (2022).
- **NVIDIA Blackwell Architecture Whitepaper** (2024).
- **Krizhevsky, Sutskever, Hinton (2012)** — *ImageNet Classification with Deep CNNs*. El paper que justifico GPUs para ML.
- **Jouppi et al. (2017)** — *In-Datacenter Performance Analysis of a TPU*. Justificacion arquitectonica de TPU.
- **Strubell, Ganesh, McCallum (2019)** — *Energy and Policy Considerations for Deep Learning in NLP*. Impacto ambiental.
