---
title: "Parte 1 — Caso Space Z: arquitectura cloud"
weight: 1
---

> **Celdas 3-5 del notebook.** Razonamiento como líder cloud antes de tocar código. El lab te pone en el rol de arquitecto de una startup que necesita desplegar un detector de autos en parking.

## El enunciado

> Imagine que usted es líder cloud de un equipo de IA. La Startup "Space Z" se contacta con usted para realizar un proyecto. El objetivo principal es desarrollar una plataforma para contar tiempos de espera de los vehículos en las entradas/salidas de un parking a través de cámaras instaladas en cada una de ellas. Al ser una startup, tienen un presupuesto limitado de gastos al mes.
>
> La solución planteada por su equipo es utilizar un modelo de reconocimiento de objetos ya entrenado en automóviles, el cual es ligero y rápido.
>
> **Requisitos:**
> 1. No cuentan con servidores propios.
> 2. El parking tiene 2 entradas.
> 3. El modelo puede ser ejecutado en 3 cámaras simultáneamente utilizando una GPU NVIDIA Tesla T4.

![Sistema de control de accesos a parking](/laboratorios/lab-19/control-accesos-parking.jpg)

## Las tres opciones

El lab pide elegir entre tres arquitecturas de deployment realistas — cada una representa un patrón distinto del [fundamento Cloud Computing](/fundamentos/cloud-computing):

| Opción | Resumen | Patrón |
|--------|---------|--------|
| **(a)** | Comprar un servidor on-prem con Tesla T4 | On-premise / self-hosted |
| **(b)** | Una VM en la nube con Tesla T4 (GCE `n1-standard-4 + T4`, AWS `g4dn.xlarge`, Azure `NC4as_T4`) | IaaS / single-instance cloud |
| **(c)** | Endpoint escalable de Vertex AI con T4 | PaaS / managed inference |

## Análisis crítico

### Opción (a) — Servidor propio: **descarte**

El argumento del enunciado ("gastos una sola vez = ahorro a largo plazo") es engañoso por tres razones:

1. **Viola explícitamente el requisito 1** ("no cuentan con servidores propios").
2. **CapEx vs OpEx para una startup**. El sticker de una Tesla T4 (16 GB GDDR6, 8.1 TFLOPS FP32, 65W TDP) en una workstation con caja-redundante-disco-RAID-ECC RAM cuesta entre **USD 5.000-9.000**. Para una startup con presupuesto limitado, soltar 6 mil dólares de golpe duele mucho más que pagar 350 USD/mes durante 18 meses. Las startups optimizan **runway**, no TCO a 5 años.
3. **Costos ocultos**: conectividad de subida grande (cámaras envían video al server), o el server **debe estar físicamente en el parking** (rayos, cortes de luz, polvo, robo); energía y aire acondicionado; mantenimiento físico; **single point of failure** sin redundancia.

### Opción (b) — VM cloud con T4: **óptima para el alcance actual**

A favor:
- Cumple con "no servidores propios": el hardware lo opera el cloud provider.
- **OpEx puro**: ~USD 0.50/h a 24/7 = ~USD 360/mes en GCE `n1-standard-4 + T4`. Si la startup quiebra, apagas y dejas de pagar al día siguiente.
- **Time-to-market rápido**: en horas tienes la VM levantada, BentoML corriendo en ella.
- **Carga ya dimensionada**: 2 cámaras + GPU soporta 3 simultáneas con margen.
- **Control completo**: tú decides el stack ([BentoML](/fundamentos/model-serving), Triton, FastAPI). Si mañana cambias de proveedor, te llevas tu Dockerfile.

En contra:
- **Tú operas la VM**: parches OS, drivers NVIDIA, restart cuando crashea.
- **HA es manual**: si quiebra el VM → quiebra el servicio. Tienes que arquitectar redundancia tú.
- **Escalabilidad futura es manual**: si mañana son 8 parkings, escalas verticalmente o pones un load balancer.

### Opción (c) — Vertex AI Endpoint: **viable si hay expansión proyectada**

A favor:
- Cumple "no servidores propios" sin reparo.
- **Autoscaling out-of-the-box**: `min_replica=1, max_replica=N`, Vertex agrega/quita instancias según QPS o utilización GPU.
- **HA managed**: rolling updates, health checks, restart automático.
- **A/B testing y traffic splitting**: deploy v2 con 5% de tráfico, comparar métricas, promover si gana.
- **Model registry, versioning, monitoring** integrados sin escribir código.

En contra:
- **Sobrecargado para 2 cámaras**. Es traer un martillo neumático para un clavo. Vertex AI cobra incluso por nodo idle.
- **Sobrecosto del 10-30%** sobre la VM equivalente.
- **Lock-in fuerte a GCP**: bundle del modelo va a Vertex Model Registry, monitoring en Vertex, pipeline en Vertex Pipelines. Migrar a SageMaker después es reescribir buena parte.
- **YAGNI**: el requisito 3 dice "3 cámaras simultáneas en 1 T4". El sistema **no necesita escalar todavía**. Optimizar para crecimiento futuro que aún no existe es uno de los peores antipatrones en startups con runway corto.

## Veredicto

Las opciones **viables son (b) y (c)**. Descarte de (a) por requisito violado + CapEx incompatible con startup.

Entre (b) y (c), **(b) es la óptima para el estado actual**, **(c) es la óptima si Space Z ya tiene roadmap firmado de 5+ parkings**.

**Recomendación de líder cloud**: empezar con **(b)** ahora, diseñando la app de forma container-portable (Dockerfile con el bento exportado, desacoplado de la VM específica). De esa forma, migrar a (c) cuando lo justifique el crecimiento del negocio es subir el bundle a Vertex Model Registry, no reescribir la solución. **Capta lo mejor de ambos mundos: simplicidad y bajo costo hoy, opción real de escalar sin deuda técnica mañana.**

## Conexión con el resto del lab

La Parte 2 en adelante **construye literalmente la opción (b)**: BentoML levantando un endpoint HTTP en localhost, simulando lo que correría en una VM cloud. No es coincidencia — el lab te muestra que el "MVP de MLOps" para una startup **es exactamente lo que vas a programar a continuación**.

## Conexión con los papers

- [Paleyes et al. (2022)](/papers/challenges-deploying-ml-paleyes-2022) identifica el deployment de modelos como uno de los stages donde más fricción aparece en case studies reales. La elección entre on-prem / VM / managed es uno de los primeros forks que toma una org.
- [Kreuzberger et al. (2023)](/papers/mlops-overview-kreuzberger-2023) propone "MLOps maturity" — startups operan en niveles 0-1 (un modelo, un endpoint, deployment manual). (b) es nivel 1. (c) implica saltar a nivel 2-3 con CI/CD de modelos.
- [Sculley et al. (2015)](/papers/hidden-technical-debt-sculley-2015) advierte sobre la **glue code** entre el modelo y el resto del sistema. (a) maximiza glue code (operar OS + drivers + serving + monitoring tú). (c) minimiza glue code pero introduce **lock-in como deuda equivalente**.

## Siguiente

{{< cards >}}
  {{< card link="../servidor-bentoml" title="Parte 2 - Servidor BentoML" subtitle="Construir el endpoint que la opción (b) requiere" icon="academic-cap" >}}
{{< /cards >}}
