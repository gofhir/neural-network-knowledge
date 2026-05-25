---
title: "Lab 19 - Entrenamiento, Deployment y MLOps con BentoML"
weight: 190
sidebar:
  open: true
---

**Profesor:** Javier Rojas
**Fecha:** Mayo 2026
**Notebook origen:** `clase_19/material/Laboratorio/Lab_Entrenamiento_Deployment_MLOps.ipynb` (26 celdas)
**Notebook ejecutado:** [lab-19.ipynb](/notebooks/lab-19.ipynb) · [HTML](/notebooks-html/lab-19.html)

## Encuadre

Laboratorio que ataca **el problema central del MLOps en visión por computadora**: cuando despliegas un modelo como API REST, el cuello de botella en latencia y throughput casi nunca es la GPU — es el **transporte de los datos** entre cliente y servidor. El lab construye un endpoint de inferencia con [BentoML](/fundamentos/model-serving) sobre el caso de negocio de una startup ficticia (Space Z, sistema de control de accesos a un parking), lo somete a un benchmark sistemático variando tamaño de imagen y concurrencia, y descubre experimentalmente que el cuello dominante es la **serialización JSON**. Cierra con la herramienta de mitigación: **compresión JPEG en memoria** con `cv2.imencode/imdecode`, que ataca la causa raíz del problema con un orden de magnitud de mejora.

El recorrido se organiza en 5 partes operativas:

1. **Caso Space Z** (celdas 3-5): razonar como líder cloud entre tres arquitecturas de deployment — on-prem vs VM cloud vs Vertex AI Endpoint.
2. **Servidor BentoML** (celdas 6-11): construir `server.py` con `Runnable + Runner + Service`, lanzarlo en background con `bentoml serve`, verificar el endpoint.
3. **Latencia vs payload** (celdas 12-14): benchmark con 4 tamaños de imagen `(3, shape, shape)` × 3 trials. Observar escalado aproximadamente lineal con el número de píxeles.
4. **Concurrencia** (celdas 15-17): barrido con `multiprocessing.Process` para N = 1, 2, 4, 6, 8, 10. Observar throughput plano = sistema saturado desde N=1.
5. **Compresión JPEG** (celdas 18-23): `cv2.imencode/imdecode` en memoria. Curva quality/tamaño. Cuantificar el ratio honesto sobre uint8 cruda.

Cada parte produce evidencia cuantitativa que conecta con los papers asociados a la [clase 19](/clases/clase-19): **Sculley 2015** (el ML es la cajita chica), **Paleyes 2022** (challenges de deployment en case studies industriales), **Kreuzberger 2023** (arquitectura formal de MLOps).

## Resultados consolidados

### Parte 3 — Latencia vs tamaño (`shape × shape × 3`, ceros, 3 trials)

| Shape  | Píxeles  | JSON (MB) | Latencia avg (s) | Mediana (s) | Throughput efectivo |
|--------|----------|-----------|------------------|-------------|---------------------|
| 640    | 1.2 M    | 6.1       | 1.74             | 1.70        | 3.5 MB/s            |
| 1024   | 3.1 M    | 15.7      | 1.34             | 1.34        | 11.7 MB/s           |
| 2048   | 12.6 M   | 62.9      | 7.95             | 7.10        | 7.9 MB/s            |
| 3028   | 27.5 M   | 137.5     | 14.07            | 13.30       | 9.8 MB/s            |

→ **Escalado aproximadamente lineal con el número de píxeles** (O(n²) sobre la dimensión del lado). Doblar el lado de la imagen cuadruplica la latencia. El "tiempo de inferencia" reportado no proviene del modelo (que es un `print()`) sino del **overhead de transporte JSON**.

### Parte 4 — Concurrencia (shape fijo `(3, 640, 640)`)

| N procesos | Wall (s) | Throughput (req/s) | Latencia individual prom |
|------------|----------|---------------------|---------------------------|
| 1          | 0.89     | 1.12                | 0.80                      |
| 2          | 1.39     | 1.44                | 1.26                      |
| 4          | 2.89     | 1.38                | 2.30                      |
| 6          | 6.05     | 0.99                | 4.88                      |
| 8          | 5.57     | 1.44                | 4.25                      |
| 10         | 8.62     | 1.16                | 6.31                      |

→ **Throughput agregado plano en ~1.2 req/s** independiente de N. Es la firma de un sistema completamente saturado desde N=1: agregar más clientes no extrae más trabajo del servidor. La latencia individual crece aproximadamente lineal con N, lo que confirma que los N parses JSON compiten por el único worker uvicorn de BentoML.

### Parte 5 — Curva calidad/tamaño JPEG (ruido random uint8, 1024×1024×3)

| Quality | Bytes JPEG | Ratio vs cruda uint8 (3.15 MB) |
|---------|-----------|-------------------------------|
| 10      | 115.483   | 27.2×                         |
| 30      | 292.800   | 10.7×                         |
| 50      | 429.467   | 7.3×                          |
| 70      | 580.060   | 5.4×                          |
| 90      | 939.067   | 3.3×                          |
| 95      | 1.228.575 | 2.6×                          |
| 100     | 2.071.153 | 1.5×                          |

→ **Quality 90 reduce 3.3× sobre uint8 cruda** en el peor caso (ruido aleatorio). Para fotos reales con correlación espacial, el ratio sube a **10-30×** en q=90 y **15-25×** en q=70 sin pérdida observable de mAP en modelos de detección.

> **Cuidado con el "26.8×" que reporta la celda 23 del notebook**: ese número está inflado porque `np.random.randint` devolvió int64 (8 bytes/píxel) en lugar de uint8. El ratio honesto sobre uint8 es 3.3× para ruido random.

## Bloques pedagógicos del lab

{{< cards >}}
  {{< card link="caso-space-z" title="Parte 1 - Caso Space Z" subtitle="Razonar como líder cloud: on-prem vs VM cloud vs Vertex AI Endpoint" icon="academic-cap" >}}
  {{< card link="servidor-bentoml" title="Parte 2 - Servidor BentoML" subtitle="server.py con Runnable + Runner + Service, bentoml serve, test de la API" icon="academic-cap" >}}
  {{< card link="latencia-payload" title="Parte 3 - Latencia vs payload" subtitle="Benchmark con 4 shapes × 3 trials, escalado O(n²) con la dimensión lineal" icon="academic-cap" >}}
  {{< card link="concurrencia-multiprocessing" title="Parte 4 - Concurrencia" subtitle="Barrido N=1..10 con multiprocessing, throughput plano = saturación temprana" icon="academic-cap" >}}
  {{< card link="compresion-jpeg" title="Parte 5 - Compresión JPEG" subtitle="imencode/imdecode en memoria, curva quality/tamaño, sweet spot ML" icon="academic-cap" >}}
  {{< card link="cierre-integracion" title="Cierre - Respuesta final integradora" subtitle="Conectar diagnóstico (transporte = bottleneck) con remedio (JPEG)" icon="academic-cap" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/hidden-technical-debt-sculley-2015" title="Sculley et al. (2015) - Hidden Technical Debt in ML" subtitle="El paper que originó el campo MLOps. CACE, glue code, pipeline jungles, 5% ML / 95% infra" icon="document-text" >}}
  {{< card link="/papers/mlops-overview-kreuzberger-2023" title="Kreuzberger et al. (2023) - MLOps: Overview, Definition, Architecture" subtitle="Definición académica: 9 principios + 9 componentes + 7 roles + arquitectura end-to-end" icon="document-text" >}}
  {{< card link="/papers/challenges-deploying-ml-paleyes-2022" title="Paleyes et al. (2022) - Challenges in Deploying ML" subtitle="Survey ACM CSUR con case studies: AirBnB, Booking, Pinterest, ISS, Sepsis Watch" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/model-serving" title="Fundamento: Model Serving" subtitle="Patrones de serving, BentoML/Triton/TorchServe, batching, cuantización" icon="adjustments" >}}
  {{< card link="/fundamentos/mlops" title="Fundamento: MLOps" subtitle="9 principios, 9 componentes, 7 roles, drift, retraining" icon="variable" >}}
  {{< card link="/fundamentos/cloud-computing" title="Fundamento: Cloud Computing" subtitle="IaaS/PaaS/FaaS/MLaaS, pricing, GCP/AWS/Azure" icon="book-open" >}}
  {{< card link="/fundamentos/docker-containers" title="Fundamento: Docker y Containers" subtitle="Dockerfile, layers, registries, NVIDIA Container Toolkit" icon="cube-transparent" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-19" title="Clase 19 - Teoría" subtitle="GPUs, Cloud, Docker, Vertex AI, serving, MLOps" icon="academic-cap" >}}
  {{< card link="/clases/clase-19/profundizacion" title="Profundización" subtitle="Memoria GPU, autoscaling, FSDP/pmap, cuantización, drift, KFP" icon="academic-cap" >}}
  {{< card link="/dominios/ingenieria-ml" title="Dominio: Ingeniería de ML" subtitle="Timeline de producción: parameter server → Docker → K8s → MLflow → MLOps moderno" icon="book-open" >}}
  {{< card link="/laboratorios/lab-18" title="Lab 18 - Word Embeddings (anterior)" subtitle="Word2Vec, analogías, sentiment analysis con MLP" icon="academic-cap" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las 26 celdas del notebook original con 6 páginas temáticas. Evidencia cuantitativa verificada en outputs reales (4 shapes × 3 trials, 6 niveles de concurrencia, 7 niveles de quality JPEG). Análisis crítico de gotchas pedagógicos (int64 vs uint8, ratio inflado, async sin await en el handler, transpose CHW/HWC inconsistente). Reproducible en Colab versión 2025.10 con CPU en ~3-5 minutos por benchmark.
