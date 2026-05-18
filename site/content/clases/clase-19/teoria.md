---
title: "Teoria - Entrenamiento, Deployment y MLOps"
weight: 10
math: true
---

Recorrido de las 59 diapositivas de la clase del prof Javier Rojas. La clase es el cierre del bloque de ingenieria del diplomado: cubre la cadena completa que va desde el **hardware** (que GPU comprar) hasta **MLOps** (como mantener un modelo en produccion sin que el sistema se degrade silenciosamente).

A diferencia de las clases previas, casi nada aqui es "matematica de aprendizaje". Es **ingenieria de sistemas para IA**: contenedores, redes, escalado, monitoreo, cultura de equipo. La cadena conceptual es:

```
Hardware → Cloud → Entrenamiento gestionado → Inferencia → Deployment escalable → MLOps
   (GPU)   (GCP)      (Vertex AI)            (API REST)    (Cloud Run)       (pipelines)
```

{{< concept-alert type="clave" >}}
**Tesis central de la clase:** un modelo entrenado en un notebook **no es un producto**. Llevarlo a produccion exige infraestructura, deployment escalable y un paradigma operacional (MLOps) para mantenerlo confiable mientras el mundo cambia.
{{< /concept-alert >}}

---

## 1. Motivacion y conceptos clave (slides 4-12)

### 1.1 El ML pipeline completo (slide 5)

La clase 19 se posiciona explicitamente sobre los **dos ultimos pasos** del pipeline ML:

```
Data collection → Data prep → Choosing/Training → Optimization → Evaluation → ┃ Deployment ┃
                                                                              ┗━━━━━━━━━━━━┛
                                                                                  ↑
                                                                       AQUI EMPIEZA LA CLASE
```

Las clases previas del diplomado cubrieron exhaustivamente *choosing/training*. Hoy nos enfocamos en **training a escala** (que requiere infra cloud) y **deployment** (que requiere serving + escalabilidad + monitoreo).

### 1.2 Anatomia GPU (slide 7-8)

El prof presenta una **tabla comparativa** de GPUs NVIDIA modernas que conviene memorizar:

| GPU | CUDA cores | Tensor cores | VRAM | Power | Precio USD |
|---|---|---|---|---|---|
| **T4** | 2.500 | 320 | 15 GB | 70 W | ~1.100 |
| **L4** | 7.680 | 240 | 24 GB | 72 W | ~2.600 |
| **L40** | 18.176 | 568 | 48 GB | 300 W | ~8.400 |
| **A100** | 6.920 | 422 | 80 GB | 400 W | ~12.000 |
| **H100** | 14.592 | 456 | 80 GB | 350 W | ~30.000 |

**Lectura correcta de la tabla:** la metrica que mas importa para LLM no son los CUDA cores sino los **Tensor cores** (matrix-matrix fused operations). Por eso H100 con "solo" 14.5 k CUDA triplica a A100 (6.9 k CUDA) en entrenamiento de transformers — el 90%+ de los FLOPs de un transformer son matmuls que viven en tensor cores.

{{< concept-alert type="recordar" >}}
**CUDA cores vs Tensor cores:** un CUDA core ejecuta una operacion FMA escalar por ciclo. Un Tensor core ejecuta una **matmul 4×4 fused** por ciclo (64-256 ops/ciclo segun generacion). Workloads ML modernas viven en tensor cores; las CUDA cores son backup para ops no estandar.
{{< /concept-alert >}}

Ver [Fundamento: GPU Hardware para ML](/fundamentos/gpu-hardware-ml) para la jerarquia completa (SMs, warps, memory hierarchy), las generaciones Pascal→Volta→Turing→Ampere→Ada→Hopper→Blackwell, y los formatos de precision (FP32/TF32/FP16/BF16/FP8/INT8).

### 1.3 NVIDIA throughput vs CPU (slide 7)

El benchmark clasico de NVIDIA muestra **V100 = 47× CPU** en inferencia ResNet-50, P100 = 15× CPU. Es marketing pero captura algo real: para deep learning, **GPU no es opcional** salvo en modelos muy pequenos.

### 1.4 ¿Vale la pena comprar hardware? (slides 9-12)

El prof plantea la pregunta y la responde matizadamente: "**ninguna opcion es mejor que otra, depende de las necesidades**" (slide 21). Pero el calculo cuantitativo concreto que el prof omite y vale la pena tener internalizado:

| Escenario | Cuando gana on-prem | Cuando gana cloud |
|---|---|---|
| Utilizacion sostenida | >30% del tiempo | <30%, picos esporadicos |
| Horizonte | >12 meses de uso predecible | Experimentacion, no se sabe que HW necesitas |
| Capital | Tienes CapEx | Prefiere OpEx |
| Equipo | Tienes SREs | Equipo chico, no quieres operar HW |

**3 desventajas reales de comprar GPU** (slide 10):
1. Inversion alta al corto plazo (~35 k USD por H100 + infra).
2. Obsolescencia rapida (3 anos antes de que la generacion siguiente la duplique).
3. Autogestion: tu eres el sysadmin.

---

## 2. Cloud computing (slides 13-22)

### 2.1 Definicion (slide 14)

> "Disponibilidad de recursos o servicios informaticos de acceso a traves de internet. No requiere hardware fisico y se cobra por tiempo de uso."

Esta definicion captura **IaaS** (Infrastructure-as-a-Service) basicamente. La realidad moderna es mas rica — ver [Fundamento: Cloud Computing](/fundamentos/cloud-computing) para la taxonomia completa **IaaS → PaaS → FaaS → MLaaS → SaaS** que el prof omite pero que es esencial para entender por que mas adelante aparecen Vertex AI (MLaaS) y Cloud Run (PaaS).

### 2.2 Cuatro ventajas (slide 15)

1. **Baja inversion al corto plazo** — convierte CapEx en OpEx.
2. **Administracion/mantencion** la realiza el proveedor — no eres sysadmin.
3. **Servicios integrados** — storage, compute, ML, networking, IAM, todo conectado.
4. **Facilidad para automatizacion y escalamiento** — el feature **decisivo** para ML. Ver seccion 5.

### 2.3 Proveedores (slide 16)

Mercado en orden de market share (Q2 2024):

| Proveedor | Share | Fortalezas para ML |
|---|---|---|
| **AWS** | ~32% | SageMaker maduro, region us-east-1 enorme |
| **Azure** | ~22% | Integracion Office, partnership OpenAI |
| **GCP** | ~12% | Vertex AI integrado, TPUs propias, fuerte en Latam |
| **Alibaba/Huawei/Oracle/IBM** | resto | regionales |

El prof se enfoca en GCP por el resto de la clase (slide 17). Razones implicitas: presencia en Chile, free tier, Vertex AI como producto bandera.

### 2.4 Tres servicios GCP fundamentales (slides 18-20)

| Servicio | Que es | Analogo AWS / Azure |
|---|---|---|
| **GCS (Google Cloud Storage)** | Object storage (buckets) | S3 / Blob Storage |
| **GCE (Google Compute Engine)** | Maquinas virtuales | EC2 / Virtual Machines |
| **Vertex AI** | Plataforma ML gestionada (training + serving + pipelines + monitoring) | SageMaker / Azure ML |

Estos tres bloquecitos arman el **esquema basico** que el prof presenta en la siguiente seccion.

---

## 3. Entrenamiento en la nube (slides 23-34)

### 3.1 Esquema basico: MV + GCS (slides 24-25)

El primer approach naive:

```
   ┌───────────┐                  ┌────────────────────┐
   │   GCE     │ ───── data ────→ │  Google Cloud      │
   │   (VM)    │ ←── checkpoints  │  Storage (GCS)     │
   │ + Python  │                  │  (bucket)          │
   │ + PyTorch │                  └────────────────────┘
   └───────────┘
```

SSH a la VM, instalas dependencies, descargas data desde GCS, entrenas, subes checkpoints a GCS. Funciona. Para un proyecto solo es razonable.

### 3.2 Limitaciones del esquema basico (slide 26)

El prof identifica dos:
- **Configuracion del ambiente manual**: cada VM requiere instalacion de Python, CUDA, drivers, framework, code, data. Tedioso, propenso a errores.
- **Monitoreo y control manuales**: no hay dashboards, no hay restart automatico ante fallas, no hay version tracking.

A esto hay que sumar el **costo oculto**: una VM con GPU on-demand corriendo 24/7 mientras esta idle (esperando a que ejecutes algo) **te factura igual**. Por eso aparece Docker + Vertex AI en las siguientes secciones.

### 3.3 Docker — la solucion al "configuracion manual" (slides 28-30)

[Docker](https://www.docker.com/) empaqueta tu ambiente completo (codigo + dependencias + OS + runtime) en una **imagen** reproducible. La imagen se ejecuta como **container** que es identico en cualquier host con Docker instalado.

```
┌───────────────┐    docker     ┌──────────────────┐    docker     ┌──────────────────┐
│  Dockerfile   │ ───────────→  │  Docker Image    │ ───────────→  │ Docker Container │
│  (receta)     │    build      │  (artefacto      │     run       │  (proceso        │
│               │               │   inmutable)     │               │   ejecutandose)  │
└───────────────┘               └──────────────────┘               └──────────────────┘
                                        │
                                        │ docker push
                                        ↓
                                ┌──────────────────┐
                                │  Registry        │
                                │  (Docker Hub,    │
                                │   Artifact Reg.) │
                                └──────────────────┘
```

Tres conceptos del slide 30 que conviene fijar:

- **Docker Client**: tu `docker build / pull / run` desde la terminal.
- **Docker Daemon (Host)**: el proceso que ejecuta containers y maneja images.
- **Docker Registry**: el almacen de imagenes (Docker Hub, **GCP Artifact Registry**, AWS ECR, Azure ACR).

Ver [Fundamento: Docker y Containers](/fundamentos/docker-containers) para Dockerfile production-ready, multi-stage builds, BuildKit cache mounts, NVIDIA Container Toolkit y comparativa registries.

### 3.4 Artifact Registry — el registro GCP (slide 31)

Repositorio gestionado de imagenes Docker en GCP. Cuando empujas tu imagen (`docker push us-central1-docker.pkg.dev/proyecto/repo/imagen:tag`), Vertex AI puede tirar de ahi para correr tu codigo en GPUs gestionadas.

### 3.5 Vertex AI Training (slides 32-34)

[Vertex AI](https://cloud.google.com/vertex-ai) es la plataforma MLaaS de GCP. La parte de **training** del producto te permite:

1. Empaquetar tu codigo en una imagen Docker (build local + push a Artifact Registry).
2. Enviar un **Custom Training Job** especificando: maquina (A100/H100), region, hyperparams, output GCS.
3. Vertex AI **provisiona** la VM con GPU, descarga la imagen, ejecuta, sube logs/checkpoints, **apaga la VM cuando termina**.

Beneficios sobre el esquema basico:
- Pay-per-use real (no pagas idle time).
- Spot/preemptible VMs con descuento 60-91% si toleras interrupciones — clave si checkpoint-eas a GCS.
- Hyperparameter tuning como Job dedicado.
- Distributed training entre multiples VMs automatizado.

Codigo concreto en [Profundizacion](profundizacion).

---

## 4. Inferencia y deployment (slides 35-46)

### 4.1 ¿Que es inferencia? (slides 36-37)

Definicion del prof:

> "Proceso de **ejecutar** datos en un modelo de aprendizaje automatico para calcular un resultado."

Las dos preguntas guia que abre el prof (`¿Quienes?` `¿Como?`) tienen respuestas distintas segun el patron de uso:

| Patron | Quienes | Como | Latencia | Throughput |
|---|---|---|---|---|
| **Embedded** | App movil/edge device | Modelo en el device (TFLite, CoreML) | <1 ms | N/A |
| **Online sync** | Web/app via HTTP | API REST/gRPC | 10-500 ms | 100-10k QPS |
| **Async batch** | Cron job nocturno | Worker consume cola | minutos | masivo |
| **Streaming** | IoT, logs | Pipeline Kafka/Pub-Sub | continuo | continuo |

Ver [Fundamento: Model Serving](/fundamentos/model-serving) para profundizacion en cada patron + comparativa de frameworks de serving.

### 4.2 Esquema cliente-modelo (slides 38-39)

```
┌─────────────┐       comunicacion        ┌─────────────┐
│  Cliente 1  │ ─────────────────────────→│             │
├─────────────┤                           │   Modelo    │
│  Cliente 2  │ ─────────────────────────→│  (predict)  │
├─────────────┤                           │             │
│  Cliente N  │ ─────────────────────────→│             │
└─────────────┘                           └─────────────┘
   web/movil                                  ¿como?
   consola                                  predict.py
```

Las dos preguntas en el slide 39 (`¿como se comunican?` y `¿que se ejecuta?`) son el corazon del deployment.

### 4.3 Definicion de deployment (slide 40)

> "Disponibilizar modelos para el uso real de usuarios."

La palabra **real** carga toda la complejidad. Implica disponibilidad, latencia, autenticacion, escalabilidad, monitoreo, costo. La clase 19 trata exactamente sobre operacionalizar esa palabra.

### 4.4 Modelo como API REST (slide 41)

El patron mas comun en 2026:

```
┌──────────┐                           ┌──────────────────────────┐
│ Cliente  │ ──── HTTP POST /predict ─→│   API REST               │
│ (web,    │      Content-Type: JSON   │   ┌────────────────────┐ │
│  movil,  │      {"x": [1.2, 3.4]}    │   │  model.py          │ │
│  cli)    │ ←──── JSON response ──────│   │  (loads pesos.pt   │ │
└──────────┘      {"prediction": 7}    │   │   on startup)      │ │
                                       │   └────────────────────┘ │
                                       └──────────────────────────┘
```

El modelo se carga **una sola vez** al iniciar el proceso, queda en memoria/GPU, y atiende requests concurrentes.

### 4.5 Frameworks de serving (slide 42)

El prof menciona 4:

| Framework | Strengths | Weaknesses |
|---|---|---|
| **NVIDIA Triton** | Multi-framework (TF, PT, ONNX, TRT, JAX), GPU optimizado, dynamic batching | Configuracion compleja |
| **BentoML** | Pythonic, facil empaquetar, multi-cloud deploy | Capa adicional, menos performance que Triton |
| **MLEM** | Standard packaging, plug-in con MLflow | Menos maduro en 2026 |
| **TorchServe** | PyTorch nativo, batching, metricas | Menos maduro |

A esta lista conviene anadir **TF Serving** (TF nativo, gRPC), **KServe** (K8s-native con autoscaling y canary), **Ray Serve** (composicion de modelos). Ver [Fundamento: Model Serving](/fundamentos/model-serving).

### 4.6 Deployment en VM y sus limites (slides 43-46)

El primer approach es replicar el esquema training pero con la API REST en una VM:

```
┌──────────┐  ┌──────────┐  ┌──────────┐         ┌──────────────────┐
│ Cliente1 │  │ Cliente2 │  │ ClienteN │ ───────→│  VM (GCE)        │
└──────────┘  └──────────┘  └──────────┘         │  ┌────────────┐  │
                                                 │  │ API REST   │  │
                                                 │  │ model.pt   │  │
                                                 │  └────────────┘  │
                                                 └──────────────────┘
```

**El problema** (slides 45-46, con la imagen del PC ardiendo): si N crece, la VM se sobrecarga. Una sola maquina = un solo punto de fallo y un cuello de botella fijo.

---

## 5. Deployment escalable (slides 47-49)

### 5.1 La solucion: multiples nodos (slide 49)

```
                                       ┌────────────────────────┐
┌──────────┐                            │                        │
│ Cliente1 │ ──→ ┐                      │  ┌─────────────────┐   │
└──────────┘     │                      │  │  Nodo 1         │   │
┌──────────┐     │                      │  │  API REST model │   │
│ Cliente2 │ ──→ │  ┌─────────────┐     │  └─────────────────┘   │
└──────────┘     │  │ Load        │     │                        │
┌──────────┐     ├─→│ Balancer    │ ───→│  ┌─────────────────┐   │
│ Cliente3 │ ──→ │  │ (Cloud Run, │     │  │  Nodo 2         │   │
└──────────┘     │  │  Vertex AI) │     │  │  API REST model │   │
┌──────────┐     │  └─────────────┘     │  └─────────────────┘   │
│ ClienteN │ ──→ ┘                      │                        │
└──────────┘                            │  ┌─────────────────┐   │
                                        │  │  Nodo N         │   │
                                        │  │  API REST model │   │
                                        │  └─────────────────┘   │
                                        └────────────────────────┘
                                          autoscaling: M nodos
                                          segun trafico
```

### 5.2 Cloud Run vs Vertex AI Endpoints

GCP tiene **dos productos** que el prof menciona en el slide 48:

| | **Cloud Run** | **Vertex AI Endpoints** |
|---|---|---|
| Tipo | PaaS / serverless container | MLaaS gestionado |
| Cold start | si (1-30 s) | si (depende del modelo) |
| GPU | si (L4, T4 desde 2024) | si (full lineup) |
| Scale-to-zero | si | si (con min=0) |
| Monitoreo nativo | basico (Cloud Monitoring) | drift detection, feature attribution |
| Cuando elegir | API simple, picos, presupuesto | modelos estandar con monitoring incluido |

{{< concept-alert type="advertencia" >}}
**Cold start** en serverless puede matar latencia para modelos grandes. Cargar un checkpoint de 5 GB toma 10-30 s. Mitigaciones: `min-instances=1` (cuesta plata pero mantiene container caliente), modelo en imagen vs en GCS, lazy loading asincrono. Ver detalles en [profundizacion](profundizacion).
{{< /concept-alert >}}

---

## 6. MLOps (slides 50-62)

### 6.1 El planteo socratico del prof (slides 51-54)

El prof construye la motivacion con tres preguntas escalonadas:

1. **¿Cuando un producto fisico se considera terminado?** Cuando sale de fabrica, se entrega.
2. **¿Cuando un producto de software se considera terminado?** Cuando se despliega y los usuarios lo usan — aunque con bugfixes y features posteriores.
3. **¿Cuando un producto de IA se considera terminado?** **Nunca**. Porque:
   - Los **datos cambian con el tiempo** (concept drift).
   - Aparecen **versiones del modelo mas novedosas** (un mejor checkpoint, un mejor algoritmo).

Esta es la justificacion filosofica de MLOps: el producto de IA es **vivo**, requiere mantenimiento continuo.

### 6.2 Definicion (slide 55)

> **MLOps:** "paradigma repetible que tiene como objetivo implementar y mantener modelos de aprendizaje automatico en produccion de manera confiable y eficiente."

Es la **misma definicion** que aparece, mas formalmente, en [Kreuzberger et al. 2023](/papers/mlops-overview-kreuzberger-2023):

> "MLOps is a paradigm, including aspects like best practices, sets of concepts, as well as a development culture when it comes to the end-to-end conceptualization, implementation, monitoring, deployment, and scalability of machine learning products."

### 6.3 El diagrama de 3 lobulos (slide 56)

```
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
   │                  │    │                  │    │                  │
   │   ML Design      │    │ Model Development│    │   Operations     │
   │                  │ ⇄  │                  │ ⇄  │                  │
   │ • Requirements   │    │ • Data prep      │    │ • Deployment     │
   │ • Use cases      │    │ • Feature eng    │    │ • CI/CD          │
   │ • Business       │    │ • Training       │    │ • Monitoring     │
   │ • Data acq.      │    │ • Evaluation     │    │ • Triggering     │
   └──────────────────┘    └──────────────────┘    └──────────────────┘
```

Es el diagrama **infinity loop** de [Kreuzberger 2023](/papers/mlops-overview-kreuzberger-2023). Los flujos van en ambos sentidos: drift detectado en Operations triggea retraining en Development, que puede a su vez reabrir cuestiones de Design (el problema cambio, hay que redefinir).

### 6.4 Pipelines (slides 57-59)

> "Una Pipeline es un flujo de trabajo conformado por uno o varios componentes y sus interacciones a traves de entradas y salidas."

```
       ┌─────────────┐
       │ Load Model  │ ──┐
       └─────────────┘   │     ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
                         ├───→│ Train Model │ ──→│ Test Model  │ ─→│ Deploy model│
       ┌─────────────┐   │    └─────────────┘    └─────────────┘   └─────────────┘
       │Load Dataset │ ──┘
       └─────────────┘
```

Una pipeline es un **DAG (Directed Acyclic Graph)** donde nodos son tasks y aristas son dependencias de I/O. Es la mejor abstraccion para evitar el anti-patron "**pipeline jungle**" identificado en [Sculley 2015](/papers/hidden-technical-debt-sculley-2015): scripts ad-hoc encadenados con cron y bash.

### 6.5 Frameworks para MLOps (slides 60-61)

**Open-source orchestrators:**

| Framework | Origen | Cuando usar |
|---|---|---|
| **Kubeflow** | Google, K8s-native | Si ya estas en K8s, multi-framework |
| **Apache Airflow** | Airbnb (2014) | DAGs generales, no solo ML; muy maduro |
| **Flyte** | Lyft | Type-safe, K8s-native, distinguish task vs workflow |
| **Prefect** | (no en slide) | Pythonic, mas moderno que Airflow |
| **Dagster** | (no en slide) | Asset-oriented, lineage automatico |

**Cloud-managed (MLaaS):**

| Cloud | Servicio | Notas |
|---|---|---|
| AWS | SageMaker (incluye Pipelines, Studio) | El mas maduro del mercado |
| Azure | Azure Machine Learning Studio | Integracion fuerte con MS stack |
| GCP | **Vertex AI** | Pipelines basado en Kubeflow + monitoring nativo |

### 6.6 Pipeline real Vertex AI (slide 62)

El screenshot del prof muestra un pipeline real con nodos:
- `get-model` (skywarrd/model_component)
- `getdata` (us-docker.pkg.dev/zippedi-pr)
- `train` (pytorch/pytorch:1.9.1-cuda11)
- `test` (python:3.8)
- `condition-1` (rama condicional segun metrica)

Cada nodo es un **component** en lenguaje Kubeflow Pipelines (KFP) — basicamente un container que toma inputs/outputs versionados.

Codigo completo de un pipeline KFP en [profundizacion](profundizacion).

---

## Cierre

La clase 19 conecta una cadena conceptual completa:

```
GPU                  → habilita training a escala
Cloud                → habilita pagar solo por lo que usas
Docker               → habilita reproducibilidad y deploy
Vertex AI            → abstrae operaciones de training/serving
API REST             → expone el modelo a usuarios reales
Cloud Run/Endpoints  → escala horizontalmente bajo demanda
MLOps                → mantiene el sistema vivo en el tiempo
```

Para profundizar:
- [Profundizacion](profundizacion) — math de memoria GPU, codigo DDP/FSDP/JAX pmap, autoscaling, cuantizacion, drift formal.
- [Fundamento: GPU Hardware para ML](/fundamentos/gpu-hardware-ml).
- [Fundamento: Cloud Computing](/fundamentos/cloud-computing).
- [Fundamento: Docker y Containers](/fundamentos/docker-containers).
- [Fundamento: Model Serving](/fundamentos/model-serving).
- [Fundamento: MLOps](/fundamentos/mlops).
- [Sculley 2015](/papers/hidden-technical-debt-sculley-2015) — el paper origen.
- [Kreuzberger 2023](/papers/mlops-overview-kreuzberger-2023) — la arquitectura canonica.
- [Paleyes 2022](/papers/challenges-deploying-ml-paleyes-2022) — case studies industriales.
