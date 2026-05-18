---
title: "Hidden Technical Debt in ML Systems (Sculley)"
weight: 195
math: true
---

{{< paper-card
    title="Hidden Technical Debt in Machine Learning Systems"
    authors="Sculley, Holt, Golovin, Davydov, Phillips, Ebner, Chaudhary, Young, Crespo, Dennison"
    year="2015"
    venue="NeurIPS 2015 (Google)"
    pdf="/papers/hidden-technical-debt-sculley-2015.pdf" >}}
El paper que **origino el campo MLOps**. Argumenta que los sistemas ML acumulan una clase especifica de deuda tecnica que no se detecta a nivel de codigo sino de **sistema**. Catalogo de 7 categorias de deuda + figura canonica "5% ML code, 95% glue code" + principio CACE (Changing Anything Changes Everything) que se volvio jerga estandar. Sin algoritmos nuevos: es un manifesto/taxonomia operacional.
{{< /paper-card >}}

---

## Contexto

Para 2015 el ML ya estaba migrando del laboratorio a la produccion masiva en Google, Facebook, Microsoft. Los autores son el equipo que opera modelos publicitarios de Google (cita el paper canonico complementario: **McMahan et al. 2013** "Ad click prediction: a view from the trenches"). La observacion incomoda:

> "developing and deploying ML systems is relatively fast and cheap, but maintaining them over time is difficult and expensive."

El paper toma la metafora de **technical debt** (Ward Cunningham 1992) y argumenta que los sistemas ML acumulan **deudas adicionales a nivel de sistema** que el debt clasico no captura. Esa deuda es **silenciosa** porque vive en interacciones entre componentes, datos y mundo externo.

El paper **no propone algoritmos**. Es taxonomia + ejemplos + advertencias. Su impacto historico es enorme: catalizo el campo que despues se llamaria MLOps.

## Ideas principales

### La Figura 1 — "5% ML code, 95% infrastructure"

La figura mas citada del paper: en un sistema ML real, el codigo de modelo es un cuadrado pequeno rodeado de una vasta infraestructura.

```
            ╔═════════════╗
            ║   ML code   ║
            ╚═════════════╝
   ┌─────────┐ ┌─────────┐ ┌──────────────┐
   │ Config  │ │ Data    │ │ Feature      │
   └─────────┘ │ collect │ │ extraction   │
   ┌─────────┐ └─────────┘ └──────────────┘
   │ Data    │ ┌─────────┐ ┌──────────────┐
   │ verif.  │ │ Process │ │ Machine      │
   └─────────┘ │ mgmt    │ │ resource mgmt│
   ┌─────────┐ └─────────┘ └──────────────┘
   │ Analysis│ ┌─────────┐   ┌──────────┐
   │ tools   │ │ Serving │   │Monitoring│
   └─────────┘ │ infra   │   └──────────┘
               └─────────┘
```

### Las 7 categorias de deuda

#### Complex Models Erode Boundaries

- **Entanglement / CACE (Changing Anything Changes Everything):** modificar un feature, hiperparam o sampling cambia importancias del resto. **No hay aislamiento real** en ML.
- **Correction cascades:** modelo `m'` que corrige output de `m` crea improvement deadlock — mejorar un componente puede empeorar el sistema.
- **Undeclared consumers:** outputs del modelo consumidos via files/logs por otros sistemas sin acuerdo formal. **Visibility debt**.

#### Data Dependencies Cost More than Code Dependencies

- **Unstable data dependencies:** features que vienen de otros modelos o sistemas que cambian. Incluso "mejoras" del signal upstream pueden romper downstream.
- **Underutilized data dependencies:** legacy features, bundled features, $\epsilon$-features que aportan poco pero crean vulnerabilidad.
- **Static analysis tooling falta** — necesitamos herramientas equivalentes a compilers para data deps.

Caso canonico: "old vs new product numbering scheme: a year later, the code that stops populating the database with old numbers is deleted. This will not be a good day for the maintainers."

#### Feedback Loops

- **Direct:** el modelo influye su propia training data future (e.g., un recomendador aprende de clicks que el mismo genera).
- **Hidden:** dos sistemas se influyen a traves del mundo (e.g., productos + reviews de la misma pagina).

#### ML-System Anti-Patterns

- **Glue code:** codigo que envuelve frameworks generales (TensorFlow, sklearn). "5% ML, 95% glue" en muchos sistemas maduros.
- **Pipeline jungles:** scripts ad-hoc encadenados que crecen organicamente.
- **Dead experimental codepaths:** branches if/else acumulando complejidad. Ejemplo: **Knight Capital** perdio USD 465M en 45 minutos por codigo experimental obsoleto.
- **Abstraction debt:** ML carece de abstracciones tan robustas como la base de datos relacional.
- **Code smells:** Plain-Old-Data Type Smell, Multiple-Language Smell, Prototype Smell.

#### Configuration Debt

En sistemas ML maduros las **lineas de config superan a las de codigo**. Cada linea es potencial bug. Necesita ser version-controlled, code-reviewed, auto-verified.

#### Dealing with External World Changes

- **Fixed thresholds in dynamic systems:** modelos retrainados pero thresholds manualmente fijados se vuelven invalidos.
- **Monitoring 3 puntos minimos:** prediction bias, action limits, up-stream producers.

#### Other Areas

Data testing debt, reproducibility debt, process management debt, **cultural debt** (separacion research vs engineering).

## Resultados experimentales

El paper **no presenta experimentos cuantitativos**. Es un position paper con ejemplos cualitativos derivados de la experiencia operacional de Google. Su rigor viene de:
- Anecdotas concretas verificables (Knight Capital, product numbering scheme).
- Categorizacion sistematica derivada de literatura SE clasica ([13] Morgenthaler et al. 2012 sobre "build debt").
- Tres mitigaciones concretas por categoria de deuda.

## Limitaciones reconocibles

- **No cuantifica** la deuda. Es taxonomia cualitativa, no metrica.
- **Sesgo Google** — autores todos de Google, ejemplos de ads/search masivo. Equipos chicos pueden tener prioridades distintas.
- **Pre-deep-learning-LLM** — no aborda problemas especificos de modelos pretrained gigantes (data contamination, prompt engineering).
- **Pocas recetas concretas** — mas problemas que soluciones. Mitigaciones genericas ("invest in tooling", "shift culture").
- **No discute observabilidad moderna** — Prometheus, distributed tracing, OpenTelemetry no eran ubicuos.

## Por que importa hoy

Este paper es el **ancestro comun** de practicamente toda la literatura MLOps. Su vocabulario es jerga estandar:

| Termino acunado | Status hoy |
|---|---|
| CACE | mantra estandar |
| 5% ML code / 95% glue | citado en intro de toda charla MLOps |
| Glue code / Pipeline jungles | vocabulario tecnico estandar |
| Hidden feedback loops | tema central en RecSys, RLHF |
| Configuration debt | motivo OmegaConf, Hydra |
| Undeclared consumers | "visibility debt" |
| Cultural debt | argumento para el rol MLOps Engineer cross-functional |

Sus mitigaciones inspiraron herramientas:
- **TFX (TensorFlow Extended)** — respuesta Google formal.
- **MLflow, W&B** — versioning como respuesta al config debt.
- **Feast, Tecton** — feature stores contra unstable data deps.
- **DVC, Pachyderm, LakeFS** — data versioning.
- **Evidently, WhyLabs, Arize** — monitoring de prediction bias.
- **Kubeflow, Airflow, Flyte** — orquestadores que reemplazan pipeline jungles.

## Notas y enlaces

- **Conferencia:** SE4ML Workshop, NIPS 2014 (version corta) → NIPS/NeurIPS 2015 (version completa).
- **Autores:** equipo Google que operaba modelos publicitarios. Mismo equipo que McMahan et al. 2013 (KDD).
- **Citaciones (~2026):** >7.000, uno de los papers mas citados sobre ML en produccion.
- **Trabajos derivados directos:**
  - Polyzotis et al. 2018 (SIGMOD): "Data lifecycle challenges in production ML".
  - Amershi et al. 2019 (ICSE): "Software engineering for ML: a case study at Microsoft".
  - Breck et al. 2017: "The ML test score" — operacionaliza Sculley en 0-28 puntos.
  - [Paleyes et al. 2022](/papers/challenges-deploying-ml-paleyes-2022): survey ACM CSUR.
  - [Kreuzberger et al. 2023](/papers/mlops-overview-kreuzberger-2023): definicion arquitectonica formal.

## Conexion con el diplomado

Este paper es la **justificacion filosofica de la clase 19**. El prof Javier Rojas plantea en el slide 51-54 "¿cuando un producto de IA esta terminado?" — Sculley es la respuesta extendida: **nunca**, por las 7 deudas.

- [Clase 19 - Entrenamiento, Deployment y MLOps](/clases/clase-19) — toda la seccion 6 (MLOps) se sustenta en este paper.
- [Fundamento: MLOps](/fundamentos/mlops) — el principio CACE y los 7 anti-patterns son referencias canonicas.

Lectura complementaria:

- [Kreuzberger et al. 2023](/papers/mlops-overview-kreuzberger-2023) — la solucion arquitectonica (componentes, roles, principios) ocho anos despues.
- [Paleyes et al. 2022](/papers/challenges-deploying-ml-paleyes-2022) — case studies industriales que confirman empiricamente los problemas que Sculley plantea.
