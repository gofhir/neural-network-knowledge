---
title: "Docker y Containers"
weight: 297
math: true
---

**Docker** (y, mas generalmente, los **containers** OCI) son la abstraccion estandar para empaquetar aplicaciones — incluido codigo ML — junto con todas sus dependencias en un artefacto portable y reproducible. Para ML moderno son **infraestructura no-negociable**: Vertex AI, SageMaker, Cloud Run, Kubernetes — todos consumen containers como unidad de deployment.

Este fundamento cubre: anatomia (image, container, layer, registry), Dockerfile production-ready para ML, multi-stage builds, NVIDIA Container Toolkit, comparativa de registries.

---

## 1. Conceptos basicos

### 1.1 La cadena conceptual

```
┌───────────────┐    docker     ┌──────────────────┐    docker     ┌──────────────────┐
│  Dockerfile   │ ───────────→  │  Docker Image    │ ───────────→  │ Docker Container │
│  (receta)     │    build      │  (artefacto      │     run       │  (proceso vivo)  │
│  texto plano  │               │   inmutable)     │               │  con filesystem  │
│               │               │   identificado   │               │  escribible      │
│               │               │   por hash       │               │  on top          │
└───────────────┘               └──────────────────┘               └──────────────────┘
                                        │
                                docker push
                                        ↓
                                ┌──────────────────┐
                                │  Registry        │
                                │  (Docker Hub,    │
                                │   GCP Artifact,  │
                                │   AWS ECR, ...)  │
                                └──────────────────┘
```

### 1.2 Glosario

- **Dockerfile** — receta texto plano con instrucciones (`FROM`, `RUN`, `COPY`, `CMD`, etc.) que define como construir una imagen.
- **Layer** — cada instruccion del Dockerfile crea una **capa inmutable** cacheable. Las capas son compartidas entre imagenes → ahorro disco y red.
- **Image** — arbol de layers + metadata. Identificada por hash SHA-256. Inmutable.
- **Container** — instancia ejecutable de una image, con una capa **escribible** encima (los cambios al filesystem desaparecen cuando el container muere salvo que uses volumes).
- **Registry** — almacen versionado de images (Docker Hub, GCP Artifact Registry, AWS ECR, Azure ACR).
- **OCI (Open Container Initiative)** — el estandar abierto. Docker, containerd, podman lo implementan.

### 1.3 Container vs VM

| | VM | Container |
|---|---|---|
| Aislamiento | Hypervisor (Type 1/2) | Kernel namespaces + cgroups |
| Tamano | GB (incluye OS completo) | MB (comparte kernel host) |
| Boot | minutos | segundos |
| Overhead | alto (kernel propio) | bajo (proceso normal +) |
| Aislamiento seguridad | fuerte (hardware) | menor (kernel compartido) |

{{< concept-alert type="recordar" >}}
**Containers no son VMs ligeras.** Comparten el kernel del host — un container Linux no corre en Windows nativamente (Docker Desktop levanta una VM Linux por dentro en hosts Windows/Mac). Esto es lo que les permite ser tan rapidos pero tambien lo que limita su uso para multi-tenant security-critical.
{{< /concept-alert >}}

---

## 2. Dockerfile production-ready para ML

### 2.1 Patron multi-stage (PyTorch training image)

```dockerfile
# syntax=docker/dockerfile:1.6
# Reduce imagen final ~70% via multi-stage

# ===== Stage 1: builder =====
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# BuildKit cache mount: pip cache persiste entre builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# ===== Stage 2: runtime =====
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app
COPY src/ ./src/
COPY train.py .

# Usuario no-root por seguridad
RUN useradd -m -u 1000 trainer && chown -R trainer:trainer /app
USER trainer

# Health check para orchestrators
HEALTHCHECK --interval=30s --timeout=10s CMD \
  python -c "import torch; assert torch.cuda.is_available()"

ENTRYPOINT ["python", "train.py"]
```

**Por que multi-stage:** stage 1 (`builder`) contiene compilador, headers, git → ~12 GB. Stage 2 (`runtime`) solo runtime + paquetes → ~4 GB. Critico para cold-start de Cloud Run y costo de Artifact Registry.

### 2.2 Buenas practicas

| Practica | Por que |
|---|---|
| Usar imagen base **especifica** (no `latest`) | Reproducibilidad |
| Combinar `RUN` con `&&` y limpiar caches | Reduce layers + tamano |
| Copiar `requirements.txt` antes que el codigo | Cache layer de deps |
| `.dockerignore` para excluir `__pycache__`, `.git`, datasets | Reduce contexto build |
| Usuario no-root | Seguridad (least privilege) |
| Healthcheck | Orchestrators detectan containers no sanos |
| Multi-stage | Imagen runtime chica |
| BuildKit cache mounts | Builds repetidos rapidos |

### 2.3 `.dockerignore` tipico ML

```
__pycache__/
*.pyc
.git/
.venv/
node_modules/
*.ipynb_checkpoints/
data/
datasets/
checkpoints/
runs/
wandb/
.DS_Store
```

Sin esto, copiar 100 GB de datasets a la build context.

---

## 3. NVIDIA Container Toolkit

Sin esto un container **no ve la GPU del host**. Instalacion + uso:

```bash
# Una sola vez en el host (Ubuntu/Debian)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --gpus all nvidia/cuda:12.4.1-base nvidia-smi

# Correr con GPUs especificas
docker run --gpus '"device=0,1"' my-training-image
```

En cluster Kubernetes equivalente: **NVIDIA GPU Operator** instala drivers + toolkit + device plugin.

---

## 4. Registries — comparativa

| Registry | Cuando elegir |
|---|---|
| **Docker Hub** | Open-source publico, hobby. 1 repo privado gratis. |
| **GCP Artifact Registry** | GCP, integracion con Vertex AI, Cloud Build, GKE |
| **AWS ECR** | AWS, integracion con ECS, EKS, Lambda |
| **Azure ACR** | Azure, integracion con ACI, AKS, App Service |
| **GitHub Container Registry (ghcr.io)** | Open-source o codigo en GitHub, vinculado al repo |
| **Quay.io (Red Hat)** | Open-source o enterprise, security scanning fuerte |
| **Self-hosted** (Harbor) | On-prem, control total |

### 4.1 Tag conventions

```bash
# Inmutable tag (preferido para deploy)
gcr.io/proyecto/imagen:v3.2.1
gcr.io/proyecto/imagen:sha-a1b2c3d  # commit hash

# Mutable tags (NUNCA para deploy production)
gcr.io/proyecto/imagen:latest
gcr.io/proyecto/imagen:dev
```

**Regla:** tag inmutable + version semantica. `:latest` solo para experimentacion local.

---

## 5. Docker Compose para desarrollo local

Para ML un setup tipico de dev con DB + cache + serving:

```yaml
# docker-compose.yml
version: '3.9'

services:
  model-server:
    build: .
    ports: ["8080:8080"]
    environment:
      - MODEL_URI=gs://my-bucket/model.pt
      - GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json
    volumes:
      - ./secrets:/secrets:ro
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

`docker compose up` levanta todo. Ideal para reproducir entorno de produccion localmente.

---

## 6. Container orchestration: Kubernetes

Para >1 container o necesidad de scale/HA, **Kubernetes (K8s)** es el estandar. Conceptos minimos:

- **Pod** — unidad atomica de deployment (1 o mas containers que comparten red).
- **Deployment** — define cuantas replicas de pods deseas.
- **Service** — expone pods via DNS interno o load balancer.
- **Ingress** — expone services a internet con routing por host/path.
- **HPA (Horizontal Pod Autoscaler)** — escala replicas segun metricas.
- **ConfigMap / Secret** — config externa al container.

Cloud equivalents managed: **GKE** (GCP), **EKS** (AWS), **AKS** (Azure).

Ver [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) Parte III.3 para HPA + custom metrics.

---

## 7. Containers vs alternativas

| Tecnologia | Cuando |
|---|---|
| **Containers** (Docker, OCI) | Default en cloud, K8s, Vertex AI, etc. |
| **WebAssembly (WASM)** | Edge, sandboxing extremo, portabilidad maxima. Limitado para ML. |
| **AWS Lambda layers** | Functions cortos con dependencies pesadas. Limites de tamano. |
| **VM images (AMI)** | Cuando necesitas el OS completo o latency cold-start importa mas que portabilidad. |

---

## 8. Conexion con el resto del site

Este fundamento se cita en:

- [Clase 19 - Entrenamiento, Deployment y MLOps](/clases/clase-19) — slides 28-30.
- [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) — Parte II (NVIDIA Container Toolkit).
- [Fundamento: Cloud Computing](/fundamentos/cloud-computing) — Artifact Registry / ECR / ACR.
- [Fundamento: Model Serving](/fundamentos/model-serving) — Triton, BentoML, KServe — todos basados en containers.
- [Fundamento: MLOps](/fundamentos/mlops) — pipelines KFP/Airflow ejecutan steps como containers.

---

## Lecturas

- **Docker official documentation** (docs.docker.com).
- **The Kubernetes Book** (Nigel Poulton) — introduccion practica.
- **Site Reliability Engineering** (Google, 2016) — origen filosofico de containers en operations.
- **The Phoenix Project** (Gene Kim) — narrativa DevOps que motiva containers.
