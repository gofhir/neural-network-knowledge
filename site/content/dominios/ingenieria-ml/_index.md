---
title: "Ingeniería de ML"
weight: 8
sidebar:
  open: true
---

# Ingeniería de Machine Learning

## El problema central

A diferencia de los otros dominios — que se organizan alrededor de **modalidades de datos** (texto, vision, audio, video, multimodal, robotica, estructurados) — este dominio es **transversal**: no estudia una clase de problemas sino una **disciplina de practica**. La pregunta guia es: *¿como pasa un modelo de ser un notebook a ser un sistema productivo confiable?*

La respuesta corta es **que no es facil**. El paper canonico [Sculley et al. 2015](/papers/hidden-technical-debt-sculley-2015) lo cuantifica con su figura mas citada: en sistemas ML reales, el codigo de modelo es **5% del sistema**; el otro 95% es infraestructura — config, data verification, feature extraction, serving, monitoring. Operar bien ese 95% es lo que se llamo, anos despues, **MLOps**.

Este dominio recorre la **historia tecnologica** que hizo posible operar ML a escala: desde los primeros sistemas distribuidos pre-deep-learning (parameter server, Hadoop), pasando por el boom de containers (Docker 2013, Kubernetes 2014), la formalizacion de MLOps como disciplina (Kubeflow, MLflow, Vertex AI, SageMaker) hasta el paradigma LLMOps emergente en 2023+.

## Por que este dominio existe

Las otras vertientes del diplomado responden "*como modelo X*". Este dominio responde "*como llevo X a produccion sin que se rompa*". Hay tres tensiones que vertebran su historia:

1. **Reproducibilidad** — el mismo codigo + mismos datos + misma seed deberia dar el mismo modelo. Imposible sin versionado de los tres.
2. **Escala** — entrenar un modelo en un laptop es distinto a entrenarlo en 8 GPUs, distinto a entrenarlo en 1024 TPUs. Distintas abstracciones para distintos regimenes.
3. **Mantenimiento** — un modelo entrenado hoy se degrada manana porque el mundo cambia (concept drift). El producto IA es **vivo**, requiere operacion continua.

MLOps es el campo que opera estas tres tensiones simultaneamente.

## Línea de tiempo

{{< timeline >}}

  {{< era name="Era pre-MLOps (infraestructura distribuida)" years="2003-2013" >}}

    {{< hito year="2003" name="Hadoop / HDFS" status="minimal" >}}
Google publica el paper de MapReduce. Pronto Doug Cutting open-sourcea Hadoop. **Por que importo:** primer paradigma masivamente distribuido para procesar datos. Sentó la base operacional sobre la que ML a escala se construyo (Mahout 2009, Spark 2011).
    {{< /hito >}}

    {{< hito year="2006" name="AWS lanza EC2" status="minimal" >}}
Amazon Web Services hace publico Elastic Compute Cloud. **Por que importo:** la primera vez que se podia rentar computo industrial por hora, sin CapEx. Cambio el modelo economico de la computacion intensiva (incluida ML).
    {{< /hito >}}

    {{< hito year="2011" name="Apache Spark + MLlib" status="minimal" >}}
Berkeley AMPLab publica Spark. **Por que importo:** mucho mas rapido que Hadoop para iterativo (algoritmos ML). MLlib se vuelve el estandar pre-deep-learning para ML distribuido en empresas (clasificacion, clustering, recomendacion).
    {{< /hito >}}

    {{< hito year="2012" name="AlexNet entrena en 2 GPUs" status="covered" link="/papers/alexnet-krizhevsky-2012" >}}
Krizhevsky, Sutskever, Hinton entrenan AlexNet en **2× GTX 580** durante una semana, ganan ImageNet por ~10 puntos. **Por que importo:** primer training "serio" multi-GPU. Justifico GPUs como hardware estandar para deep learning. Antes era visto como overkill.
    {{< /hito >}}

    {{< hito year="2013" name="Docker hace publicos los containers" status="covered" link="/fundamentos/docker-containers" >}}
Solomon Hykes presenta Docker en PyCon. **Por que importo:** containers OCI se vuelven la unidad estandar de deployment. Sin Docker no hay Vertex AI, no hay SageMaker, no hay Cloud Run, no hay MLOps como lo conocemos. **El sustrato.**
    {{< /hito >}}

    {{< hito year="2014" name="Parameter Server (Li et al. OSDI)" status="minimal" >}}
Mu Li et al. publican "Scaling distributed machine learning with the parameter server". **Por que importo:** primera abstraccion ampliamente aceptada para distributed training. Citada por [Sculley 2015](/papers/hidden-technical-debt-sculley-2015) como rara isla de consenso arquitectonico en una literatura ML caotica.
    {{< /hito >}}

  {{< /era >}}

  {{< era name="Era proto-MLOps (kubernetes y frameworks)" years="2014-2017" >}}

    {{< hito year="2014" name="Kubernetes (Google → CNCF)" status="covered" link="/fundamentos/docker-containers" >}}
Google open-sourcea Borg como Kubernetes. **Por que importo:** se vuelve el sustrato de orquestacion estandar. Casi todo MLOps moderno (Kubeflow, KServe, Vertex AI) corre encima de K8s. **El segundo sustrato.**
    {{< /hito >}}

    {{< hito year="2014" name="AWS Lambda — primer serverless" status="minimal" >}}
**Por que importo:** abre el paradigma FaaS (Functions-as-a-Service). Anios despues, Cloud Run + Cloud Functions traen lo mismo a serving ML.
    {{< /hito >}}

    {{< hito year="2015" name="TensorFlow open-source (Google)" status="minimal" >}}
Google libera TensorFlow. **Por que importo:** primer framework deep learning open-source diseñado **explicitamente para production** (vs Theano/Torch que eran research-first). Mismo mes que [Sculley 2015](/papers/hidden-technical-debt-sculley-2015).
    {{< /hito >}}

    {{< hito year="2015" name="Sculley et al. — el paper origen MLOps" status="deep" link="/papers/hidden-technical-debt-sculley-2015" >}}
"Hidden Technical Debt in Machine Learning Systems" (NeurIPS). **Por que importo:** primera articulacion academica de **por que** ML en produccion es dificil. Catalogo de 7 deudas, principio CACE, anti-patterns (glue code, pipeline jungles). Cataliza el campo que se llamaria MLOps. El paper ancestro de todo el dominio.
    {{< /hito >}}

    {{< hito year="2016" name="PyTorch open-source (Facebook)" status="minimal" >}}
**Por que importo:** competidor research-friendly de TensorFlow. Eventualmente domina research; TensorFlow domina produccion. Esta dualidad fuerza la emergencia de **ONNX** (2017) como formato intercambio.
    {{< /hito >}}

    {{< hito year="2017" name="Uber Michelangelo (paper interno)" status="minimal" >}}
Hermann & Del Balso publican "Meet Michelangelo: Uber's Machine Learning Platform". **Por que importo:** primera plataforma ML interna corporativa publicada. Demuestra que MLOps **se puede ingenierar**. Inspiro Kubeflow, SageMaker, Vertex AI.
    {{< /hito >}}

    {{< hito year="2017" name="Zinkevich (Google) — 'Rules of ML'" status="minimal" >}}
Whitepaper Google con 43 reglas operacionales derivadas del espiritu Sculley. **Por que importo:** primer manual operacional concreto de ML productivo.
    {{< /hito >}}

  {{< /era >}}

  {{< era name="Era MLOps formal (frameworks, tools y vocabulario)" years="2018-2022" >}}

    {{< hito year="2018" name="Kubeflow + Kubeflow Pipelines" status="covered" link="/fundamentos/mlops" >}}
Google + comunidad publican Kubeflow. **Por que importo:** primera implementacion open-source de **ML como DAG sobre K8s**. KFP se vuelve el estandar de pipelines portables. Vertex AI Pipelines lo soporta nativamente.
    {{< /hito >}}

    {{< hito year="2018" name="MLflow (Databricks)" status="covered" link="/fundamentos/mlops" >}}
Databricks open-sourcea MLflow. **Por que importo:** estandar de **experiment tracking + model registry + project packaging**. Resuelve "reproducibilidad" y "versioning" del paper Sculley.
    {{< /hito >}}

    {{< hito year="2018" name="TFX (TensorFlow Extended)" status="minimal" >}}
Modi et al. publican "TFX: a TensorFlow-Based Production-Scale ML Platform". **Por que importo:** primera plataforma E2E publica que implementa todos los componentes de Kreuzberger antes de que existieran como categoria.
    {{< /hito >}}

    {{< hito year="2018" name="DVC + Pachyderm — data versioning" status="minimal" >}}
**Por que importo:** primer tooling para versionar **datasets** como se versionan codigo. Resuelve "unstable data dependencies" de Sculley.
    {{< /hito >}}

    {{< hito year="2018" name="Feast — primer feature store open-source" status="minimal" >}}
GoJek + Google open-sourcean Feast. **Por que importo:** primer implementacion publica de **feature store** (offline + online). Resuelve training/serving skew, el componente C4 de Kreuzberger.
    {{< /hito >}}

    {{< hito year="2019" name="Vertex AI / SageMaker — MLaaS maduros" status="covered" link="/fundamentos/mlops" >}}
GCP renombra AI Platform a Vertex AI; AWS expande SageMaker; Azure lanza ML Studio. **Por que importo:** las tres hyperscalers consolidan sus plataformas MLOps E2E, alineadas a los 9 componentes que Kreuzberger formalizaria 4 anios despues.
    {{< /hito >}}

    {{< hito year="2020" name="NVIDIA Triton Inference Server GA" status="covered" link="/fundamentos/model-serving" >}}
**Por que importo:** primer serving framework GPU-optimized multi-framework (TF, PT, ONNX, TRT) con **dynamic batching**. Estandar para inferencia high-QPS en GPU.
    {{< /hito >}}

    {{< hito year="2021" name="Datasheets for Datasets + Model Cards" status="minimal" >}}
Gebru et al. y Mitchell et al. proponen documentos estandar para datasets/modelos. **Por que importo:** primera "data governance" formal. Anti-pattern Sculley "undeclared consumers" empieza a tener antidote.
    {{< /hito >}}

    {{< hito year="2022" name="Paleyes et al. — survey ACM CSUR" status="deep" link="/papers/challenges-deploying-ml-paleyes-2022" >}}
"Challenges in Deploying ML: a Survey of Case Studies". **Por que importo:** primer survey academico riguroso que mapea los **problemas reales** documentados en 159 referencias. Confirma empiricamente Sculley y agrega ethics/law/trust/security como cross-cutting first-class.
    {{< /hito >}}

  {{< /era >}}

  {{< era name="Era MLOps consolidada y LLMOps emergente" years="2023-2026" >}}

    {{< hito year="2023" name="Kreuzberger et al. — la arquitectura formal" status="deep" link="/papers/mlops-overview-kreuzberger-2023" >}}
"MLOps: Overview, Definition, and Architecture" (IEEE Access). **Por que importo:** la definicion academica consensuada. 9 principios + 9 componentes + 7 roles + arquitectura E2E de 4 zonas. El vocabulario P1-P9 / C1-C9 / R1-R7 se vuelve referencia citable. **El paper que cierra una decada de evolucion.**
    {{< /hito >}}

    {{< hito year="2023" name="LLMOps emerge como sub-disciplina" status="minimal" >}}
Con la explosion de GPT-4, ChatGPT y open-source LLMs, emerge **LLMOps** con vocabulario propio: vector DB, prompt registry, eval frameworks (Promptfoo, OpenAI Evals, LangSmith). **Por que importo:** los principios MLOps siguen aplicando pero se extienden con prompt engineering, fine-tuning (LoRA, QLoRA), serving especifico LLM (vLLM, TGI).
    {{< /hito >}}

    {{< hito year="2024" name="Feature Stores + Vector DBs + RAG MLOps" status="minimal" >}}
Pinecone, Weaviate, Qdrant compiten en vector storage. **Por que importo:** el feature store clasico (C4) se complementa con **vector storage** para embeddings de RAG. Stack MLOps se bifurca: classical ML vs LLM/RAG.
    {{< /hito >}}

    {{< hito year="2024" name="W&B Weave, LangSmith — Observability LLM" status="minimal" >}}
**Por que importo:** primera generacion de tools especializados en observabilidad de **agentes LLM** — trazas de tool calls, prompts, alucinaciones. Extension natural de monitoring clasico (C9).
    {{< /hito >}}

    {{< hito year="2017-2022" name="Aprendizaje continuo (respuesta algorítmica al drift)" status="covered" link="/fundamentos/aprendizaje-continuo" >}}
Cuando un modelo en producción se degrada por *concept drift*, el reentrenamiento desde cero es caro y puede no tener acceso a los datos antiguos. El **aprendizaje continuo** —[EWC](/papers/ewc-kirkpatrick-2017) (regularización), [GEM](/papers/gem-lopez-paz-2017)/[iCaRL](/papers/icarl-rebuffi-2017) (replay), [L2P](/papers/l2p-wang-2022) (prompts)— es la respuesta algorítmica: incorporar datos nuevos sin **olvido catastrófico**. **Por qué importó:** complementa el monitoring y el reentrenamiento del MLOps clásico con métodos que actualizan el modelo de forma incremental. Cubierto en la [Clase 32](/clases/clase-32).
    {{< /hito >}}

  {{< /era >}}

{{< /timeline >}}

## Conexion con otros dominios

A diferencia de los demas dominios — que estudian **arquitecturas por modalidad** — este dominio es transversal: **cualquier sistema ML productivo**, sin importar la modalidad, atraviesa el pipeline MLOps. Es la "plomeria" comun.

Conexiones relevantes:

- [Texto / NLP](/dominios/texto) — los LLMs modernos viven de operacionalizacion MLOps a escala extrema. LLMOps es subdominio.
- [Visión](/dominios/vision) — vision en produccion (clinica, autonomo, retail) requiere serving especializado y monitoring de drift visual.
- [Audio](/dominios/audio) — pipelines de Whisper en produccion requiere optimizacion (cuantizacion) + serving streaming.
- [Robótica / RL](/dominios/robotica) — robot foundation models en produccion son frontera maxima de MLOps (real-time, embedded, safety-critical).

## Lecturas obligadas

- **[Sculley et al. (2015)](/papers/hidden-technical-debt-sculley-2015)** — el manifesto origen.
- **[Kreuzberger et al. (2023)](/papers/mlops-overview-kreuzberger-2023)** — la arquitectura formal.
- **[Paleyes et al. (2022)](/papers/challenges-deploying-ml-paleyes-2022)** — survey de challenges industriales.
- **Zinkevich, "Rules of ML"** (Google 2017) — 43 reglas operacionales.
- **Huyen, "Designing Machine Learning Systems"** (O'Reilly 2022).
- **Burkov, "Machine Learning Engineering"** (2020).

## Fundamentos del dominio

- [GPU Hardware para ML](/fundamentos/gpu-hardware-ml).
- [Cloud Computing](/fundamentos/cloud-computing).
- [Docker y Containers](/fundamentos/docker-containers).
- [Model Serving](/fundamentos/model-serving).
- [MLOps](/fundamentos/mlops).

## Clase asociada del diplomado

- [Clase 19 — Entrenamiento, Deployment y MLOps](/clases/clase-19).
