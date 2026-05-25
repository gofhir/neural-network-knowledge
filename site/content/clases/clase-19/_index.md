---
title: "Clase 19 - Entrenamiento, Deployment y MLOps"
weight: 190
sidebar:
  open: true
---

**Profesor:** Javier Rojas
**Fecha:** 2026-05-14

Cierre del bloque de ingenieria del diplomado. Recorrido por la cadena completa que va desde el hardware (GPU, comparativa T4/L4/L40/A100/H100) hasta el deployment escalable en la nube, pasando por contenedores Docker, entrenamiento en Vertex AI, inferencia y deployment de modelos como API REST, deployment escalable con Cloud Run y, finalmente, MLOps como paradigma para mantener modelos en produccion de forma confiable y eficiente.

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 59 diapositivas: GPUs, Cloud, Docker, Vertex AI, serving, MLOps" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Memoria GPU, autoscaling, FSDP/pmap, cuantizacion, drift, KFP" icon="beaker" >}}
  {{< card link="/laboratorios/lab-19" title="Laboratorio 19 - BentoML + benchmark" subtitle="Construir el endpoint, medir latencia/throughput, atacar el bottleneck con compresion JPEG" icon="academic-cap" >}}
  {{< card link="/clases/clase-20" title="Clase siguiente: ELMo, BERT, GPT, ChatGPT" subtitle="Cierre del diplomado con LLMs" icon="arrow-right" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/gpu-hardware-ml" title="Fundamento: GPU Hardware para ML" subtitle="CUDA cores, Tensor cores, FP16/BF16/FP8, generaciones Pascal-Blackwell" icon="chip" >}}
  {{< card link="/fundamentos/cloud-computing" title="Fundamento: Cloud Computing" subtitle="IaaS/PaaS/FaaS/MLaaS, pricing, spot/preemptible, GCP/AWS/Azure" icon="book-open" >}}
  {{< card link="/fundamentos/docker-containers" title="Fundamento: Docker y Containers" subtitle="Dockerfile, layers, registries, NVIDIA Container Toolkit" icon="cube-transparent" >}}
  {{< card link="/fundamentos/model-serving" title="Fundamento: Model Serving" subtitle="Cuantizacion, ONNX/TensorRT, FastAPI/Triton/TorchServe/BentoML" icon="adjustments" >}}
  {{< card link="/fundamentos/mlops" title="Fundamento: MLOps" subtitle="9 principios, 9 componentes, 7 roles, drift, retraining" icon="variable" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/hidden-technical-debt-sculley-2015" title="Sculley et al. (2015) - Hidden Technical Debt in ML" subtitle="El paper que origino el campo MLOps. CACE, glue code, pipeline jungles, 5% ML code / 95% infra" icon="document-text" >}}
  {{< card link="/papers/mlops-overview-kreuzberger-2023" title="Kreuzberger et al. (2023) - MLOps: Overview, Definition, Architecture" subtitle="La definicion academica formal: 9 principios + 9 componentes + 7 roles + arquitectura end-to-end" icon="document-text" >}}
  {{< card link="/papers/challenges-deploying-ml-paleyes-2022" title="Paleyes et al. (2022) - Challenges in Deploying ML" subtitle="Survey ACM CSUR con case studies industriales: AirBnB, Booking, Pinterest, ISS, Sepsis Watch" icon="document-text" >}}
{{< /cards >}}

## Dominio

{{< cards >}}
  {{< card link="/dominios/ingenieria-ml" title="Dominio: Ingenieria de ML" subtitle="Linea de tiempo de produccionizacion: parameter server, Docker, K8s, MLflow, MLOps moderno" icon="book-open" >}}
{{< /cards >}}
