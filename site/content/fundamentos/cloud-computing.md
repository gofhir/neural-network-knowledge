---
title: "Cloud Computing"
weight: 296
math: true
---

**Cloud computing** es la entrega de recursos de computo (CPU, GPU, storage, networking, servicios gestionados) a traves de internet con un modelo de pago por uso. Para ML moderno es practicamente sinonimo de **donde corren los workloads** — desde notebooks de exploracion hasta training distribuido en miles de GPUs hasta serving de modelos a millones de usuarios.

Este fundamento cubre la **taxonomia de servicios** (IaaS/PaaS/FaaS/MLaaS/SaaS), los **modelos de pricing** (on-demand, reserved, spot), la **comparativa GCP/AWS/Azure** para ML, y los **patrones de costo** que conviene memorizar.

---

## 1. Taxonomia de modelos de servicio

```
On-premise → IaaS → PaaS → FaaS → SaaS/MLaaS
   ↑                                       ↓
   tu manejas todo            el proveedor maneja todo
```

| Nivel | Tu manejas | Proveedor maneja | Ejemplo GCP |
|---|---|---|---|
| **On-prem** | Hardware, energia, networking, OS, runtime, app, datos | nada | tu data center |
| **IaaS** | OS, runtime, app, datos | Hardware, virtualizacion, red | Compute Engine, Cloud Storage |
| **PaaS** | App, datos | OS, runtime, escalado | Cloud Run, App Engine |
| **FaaS** | Funcion + datos | todo lo demas, scale-to-zero | Cloud Functions |
| **MLaaS** | Datos, hyperparams | Training, deployment, monitoring | Vertex AI |
| **SaaS** | Datos (a veces) | todo | BigQuery ML, Gemini API |

{{< concept-alert type="recordar" >}}
**Regla:** a mas alto en la pila, **menos control + menos operacion + mas dependencia del vendor**. La eleccion correcta depende del trade-off entre **velocidad de delivery** y **portabilidad**.
{{< /concept-alert >}}

---

## 2. Proveedores principales (Q2 2024 market share)

| Proveedor | Share | Fortalezas para ML |
|---|---|---|
| **AWS** | ~32% | SageMaker maduro, region us-east-1 enorme, mayor catalogo |
| **Azure** | ~22% | Office integration, partnership OpenAI, Azure ML |
| **GCP** | ~12% | Vertex AI integrado, **TPUs propias**, fuerte LATAM |
| **Alibaba Cloud** | ~4% | Lider en China |
| **Otros** (Oracle, IBM, Huawei, Tencent) | resto | regionales o legacy |

### 2.1 Servicios equivalentes — la "Rosetta Stone" cloud

| Funcion | GCP | AWS | Azure |
|---|---|---|---|
| Object storage | Cloud Storage (GCS) | S3 | Blob Storage |
| Maquinas virtuales | Compute Engine (GCE) | EC2 | Virtual Machines |
| Container registry | Artifact Registry | ECR | ACR |
| Container managed | Cloud Run | App Runner, Fargate | Container Apps |
| Kubernetes managed | GKE | EKS | AKS |
| Serverless functions | Cloud Functions | Lambda | Azure Functions |
| ML platform | **Vertex AI** | **SageMaker** | **Azure ML Studio** |
| ML pipelines | Vertex AI Pipelines (KFP) | SageMaker Pipelines | Azure ML Pipelines |
| Feature store | Vertex AI Feature Store | SageMaker Feature Store | Azure ML Feature Store |
| Notebook hosted | Vertex Workbench, Colab Enterprise | SageMaker Studio | Azure ML Notebooks |
| API LLM | Gemini API, Vertex Generative AI | Bedrock | Azure OpenAI Service |

---

## 3. Modelos de pricing

### 3.1 Tres modalidades fundamentales

| Modalidad | Descuento | Trade-off |
|---|---|---|
| **On-demand** | 0% (precio base) | Sin compromiso, maxima flexibilidad. Caro. |
| **Reserved / Committed Use** | 30-60% | Compromiso 1 o 3 anos. Ideal para baseline predecible. |
| **Spot / Preemptible** | 60-91% | El proveedor puede matarlas con 30 s aviso. Ideal training con checkpoints. |

### 3.2 Ejemplo concreto (GCP, 2024-2026)

VM con 1× A100 80 GB en `us-central1`:

| Modalidad | USD/hora |
|---|---|
| On-demand | ~3.67 |
| Committed 1 ano | ~2.30 |
| Committed 3 anos | ~1.40 |
| Spot/preemptible | ~1.10 |

Sumar: 1× T4 on-demand ~0.35 USD/h, 1× L4 ~0.71 USD/h, 1× H100 ~10 USD/h en SXM.

### 3.3 Estrategia tipica training

```
1. Notebook exploration       → on-demand, instancias chicas
2. Hyperparameter sweep       → spot, distintos jobs paralelos
3. Final training run         → spot + checkpointing cada N pasos a GCS
4. Production serving         → on-demand o committed (no spot, downtime no aceptable)
```

Codigo de checkpoint-resilient training en [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) seccion II.6.

---

## 4. Patrones de uso ML

### 4.1 Training

- **VM + GCS** (IaaS) — flexible, requires sysadmin work. Bueno para investigacion.
- **Vertex AI Training Job** (MLaaS) — empaquetas container, GCP provisiona, ejecuta, apaga.
- **Vertex AI Pipelines** — multiples jobs orquestados como DAG, soportan KFP.

### 4.2 Serving

- **Cloud Run** (PaaS) — container con autoscaling, scale-to-zero, GPU opcional desde 2024. Cold start 1-30 s.
- **Vertex AI Endpoints** (MLaaS) — modelo + monitoring + drift detection.
- **GKE** (IaaS-cluster) — control total con Kubernetes para casos complejos.
- **Cloud Functions / Lambda** (FaaS) — solo para modelos muy chicos en CPU.

### 4.3 Storage

- **Cloud Storage** (object storage) — buckets para datos, modelos, artifacts. Pricing por GB-mes + egress.
- **Persistent Disk** — discos para VMs, SSD o HDD.
- **BigQuery** — data warehouse SQL, soporta ML inline (BigQuery ML).
- **Firestore / Bigtable** — bases de datos NoSQL para features online.

### 4.4 Networking

- **VPC** (Virtual Private Cloud) — red privada virtual entre tus recursos.
- **Cloud Load Balancing** — distribuye trafico entre replicas de serving.
- **Cloud CDN** — cache geografico, util para modelos chicos servidos a nivel global.

---

## 5. Costo total (TCO) — pensar mas alla del precio hora

Cuatro costos invisibles que aparecen en facturas:

1. **Egress** (salida de datos del cloud) — GCS a internet cuesta 0.08-0.12 USD/GB. Trasvasar 1 TB = 80-120 USD.
2. **Idle compute** — VMs encendidas pero sin carga te facturan igual. Causa #1 de cuentas hinchadas.
3. **Snapshots y backups** — pequenos pero acumulativos.
4. **Logs y monitoring** — Cloud Logging cobra por GB ingestado mas alla del free tier.

### 5.1 Cuando on-prem gana

| Factor | On-prem gana si | Cloud gana si |
|---|---|---|
| Utilizacion sostenida | > 30% del tiempo | < 30%, picos esporadicos |
| Horizonte | > 12 meses predecibles | Experimentacion o crecimiento variable |
| Capital | Tienes CapEx | Prefieres OpEx |
| Equipo | Tienes SREs | Equipo chico |
| Compliance | Datos sensibles que no pueden salir | Sin restriccion |

Calculo concreto H100 cloud vs comprar en [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) seccion I.4.

---

## 6. Multi-cloud y vendor lock-in

**Vendor lock-in** es el riesgo de que migrar a otro proveedor sea prohibitivo. Mitigaciones:

- Usar **Kubernetes** en vez de Cloud Run / App Runner / Container Apps — portable entre clouds.
- Usar **Terraform** para infraestructura como codigo — mismo language en GCP/AWS/Azure.
- Usar **ONNX** como formato de modelo — portable entre frameworks y clouds.
- Evitar servicios propietarios sin equivalente abierto (e.g., DynamoDB) salvo razon fuerte.

**Multi-cloud** real es **caro** — duplicas operacion + costos de egress entre clouds. La estrategia comun es **primary + DR (Disaster Recovery)**, no balanceo activo.

---

## 7. Conexion con el resto del site

Este fundamento se cita en:

- [Clase 19 - Entrenamiento, Deployment y MLOps](/clases/clase-19) — slides 13-22.
- [Clase 19 - Profundizacion](/clases/clase-19/profundizacion) — codigo Vertex AI SDK.
- [Fundamento: Docker y Containers](/fundamentos/docker-containers) — sobre Artifact Registry.
- [Fundamento: MLOps](/fundamentos/mlops) — sobre Vertex AI Pipelines.
- [Fundamento: Model Serving](/fundamentos/model-serving) — Cloud Run vs Vertex AI Endpoints.

---

## Lecturas

- **Google Cloud Architecture Framework** — guia oficial (cloud.google.com/architecture/framework).
- **AWS Well-Architected Framework** — el equivalente AWS, 6 pilares (operational excellence, security, reliability, performance, cost, sustainability).
- **Cloud Native Computing Foundation** (cncf.io) — landscape de tools cloud-native open-source.
- **The Phoenix Project / The Unicorn Project** (Gene Kim) — narrativa de transformacion DevOps que sustenta el porque del cloud.
