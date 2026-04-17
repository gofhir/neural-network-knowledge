---
title: "Foundation Models"
weight: 86
math: true
---

**Foundation models** son modelos entrenados sobre **datos amplios** (generalmente con **self-supervision** a gran escala) que pueden ser **adaptados a una amplia variedad de tareas downstream**. El termino fue acunado en el reporte de Stanford CRFM (Bommasani et al. 2021) y captura un cambio de paradigma en como construimos sistemas de IA.

Ejemplos: **BERT, GPT-3, DALL-E, CLIP, LLaMA, GPT-4, Claude**.

---

## 1. Definicion

Un foundation model es un modelo que cumple tres caracteristicas:

1. **Entrenado sobre datos amplios y diversos** (texto de internet, imagenes web, datasets multimodales).
2. **Entrenado con self-supervision** a escala (no requiere labels humanos).
3. **Adaptable** a muchas tareas downstream mediante fine-tuning, prompting o retrieval.

### La intuicion de Bommasani et al.

Dos propiedades definen a los foundation models:

- **Emergence**: capacidades que emergen **implicitamente** de la escala, no son programadas explicitamente. GPT-3 hace aritmetica, traduccion y code completion sin ser entrenado para ninguna de esas tareas.
- **Homogenization**: los mismos modelos base se usan para **todas** las tareas. BERT es el backbone de clasificacion, QA, NER y traduccion. Esto da leverage (una mejora en el modelo base beneficia a todos los productos downstream) pero crea **single points of failure**.

{{< concept-alert type="clave" >}}
**La historia de la IA ha sido una trayectoria creciente de emergence y homogenization**:

| Era | Que emerge | Que se homogeniza |
|---|---|---|
| Machine Learning (90s) | El "como" resolver tareas (aprendido de datos) | Algoritmos de aprendizaje (regresion logistica, SVMs) |
| Deep Learning (2010s) | Features high-level | Arquitecturas (CNN, RNN, luego Transformer) |
| Foundation Models (2020s) | Funcionalidades enteras (razonamiento, in-context learning) | El modelo mismo (GPT-3, CLIP) |

(Tabla basada en la Figura 1 de Bommasani et al. 2021.)
{{< /concept-alert >}}

---

## 2. Como Surgen: Los 3 Ingredientes

La emergencia de foundation models requirio tres avances simultaneos:

### 2.1 Hardware

- **GPUs** con throughput 10x en 4 anos (A100 → H100 → B100).
- **TPUs** de Google especificamente disenadas para deep learning.
- **Memoria de alto ancho de banda** (HBM, HBM2, HBM3).
- Escalado a **10,000+ GPUs** en data centers para entrenamiento distribuido.

### 2.2 Arquitectura: Transformer

[Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) introdujeron el **Transformer**, que es:

- **Paralelizable** (a diferencia de RNN/LSTM).
- **Escalable** casi linealmente hasta ordenes de magnitud impensables antes.
- **Expresivo**: mecanismo de atencion captura dependencias arbitrarias.
- **Universal**: funciona para texto, imagenes, audio, video, codigo, DNA.

### 2.3 Datos + self-supervision

- **Self-supervision**: tareas derivadas automaticamente de datos no etiquetados.
  - **Masked Language Modeling** (BERT): predecir palabras ocultas.
  - **Next Token Prediction** (GPT): predecir la siguiente palabra.
  - **Contrastive learning** (CLIP): emparejar imagen con texto.
  - **Denoising** (diffusion): recuperar imagen original de ruido.
- **Escala masiva de datos**: Common Crawl (cientos de TB), LAION-5B (5B imagenes con captions).

{{< concept-alert type="clave" >}}
Transfer learning + escala = foundation models. Pretrain en una tarea pretexto sobre datos masivos sin labels, luego adaptar a tareas downstream con datos escasos. Ver [transfer learning](transfer-learning) para el framework general.
{{< /concept-alert >}}

---

## 3. Capacidades Emergentes

Con escala, los modelos adquieren capacidades **no explicitamente entrenadas**:

### In-context learning (GPT-3)

Sin gradient updates, GPT-3 aprende de ejemplos en el prompt:

```
Q: Translate "Hello" to French.
A: Bonjour

Q: Translate "Good morning" to French.
A: Bonjour

Q: Translate "Thank you" to French.
A: [model generates: Merci]
```

### Chain-of-thought reasoning

Prompts del tipo "Let's think step by step" desbloquean razonamiento multi-paso.

### Zero-shot generalization

CLIP clasifica imagenes de categorias **nunca vistas** durante entrenamiento, usando descripciones naturales.

### Instruction following

GPT-4 y Claude siguen instrucciones expresadas en lenguaje natural sin training especifico.

---

## 4. Ejemplos Canonicos

| Modelo | Ano | Dominio | Contribucion |
|---|---|---|---|
| **BERT** | 2018 | Texto | Masked language modeling, bidireccional, fine-tuning universal |
| **GPT-2** | 2019 | Texto | Generacion a gran escala |
| **T5** | 2019 | Texto | Text-to-text para toda tarea |
| **GPT-3** | 2020 | Texto | 175B params, in-context learning |
| **CLIP** | 2021 | Vision-Lenguaje | Contrastive, zero-shot classification |
| **DALL-E** | 2021 | Generacion imagen | Texto a imagen |
| **Codex** | 2021 | Codigo | Copilot backbone |
| **LLaMA** | 2023 | Texto | Foundation model abierto |
| **GPT-4** | 2023 | Multimodal | Razonamiento general |
| **Claude 3** | 2024 | Multimodal | Context 200K, razonamiento |
| **Claude Opus 4.7** | 2026 | Multimodal | Context 1M, agentes autonomos |

Para Espanol: **BETO** (DCCUChile) -- variante BERT entrenada en corpus en Espanol.

---

## 5. Adaptacion: Como Usar un Foundation Model

Tres paradigmas principales:

### 5.1 Fine-tuning

Actualizar (algunos o todos) los pesos del modelo en la tarea objetivo. Cuando hay datos suficientes (miles de ejemplos), produce el mejor rendimiento. Ver [transfer learning](transfer-learning).

### 5.2 Prompting

Describir la tarea en lenguaje natural. Sin gradient updates. Ideal para:

- Datasets muy pequenos (< 100 ejemplos).
- Prototipado rapido.
- Tareas diversas con un solo modelo.

### 5.3 Retrieval-Augmented Generation (RAG)

Combinar el foundation model con una base de datos externa. El modelo genera, pero primero **recupera** documentos relevantes para contextualizar la respuesta. Standard en aplicaciones empresariales.

### 5.4 Parameter-efficient fine-tuning (PEFT)

| Tecnica | Idea |
|---|---|
| **LoRA** | Fine-tune solo low-rank updates a las matrices de atencion |
| **Adapters** | Insertar pequenas MLPs entre capas, fine-tune solo esas |
| **Prompt tuning** | Aprender embeddings de prompts (continuous prompts) |
| **Prefix tuning** | Aprender un prefix key-value en atencion |

Permiten fine-tunear modelos de 70B+ parametros con **GPUs consumer** (ej. RTX 4090).

---

## 6. Oportunidades

Bommasani et al. identifican beneficios masivos:

### 6.1 Capacidades
- **Lenguaje**: comprension, generacion, traduccion en cientos de idiomas.
- **Vision**: clasificacion zero-shot, deteccion, generacion.
- **Robotica**: policy learning con demonstrations en lenguaje.
- **Razonamiento y busqueda**: resolver problemas multi-paso.
- **Interaccion**: dialogo natural, asistentes.

### 6.2 Aplicaciones
- **Salud y biomedicina**: diagnostico asistido, diseno de farmacos.
- **Legal**: analisis de contratos, busqueda legal.
- **Educacion**: tutores personalizados, grading automatico.

---

## 7. Riesgos

El mismo reporte dedica **40+ paginas** a riesgos:

### 7.1 Homogenization como vulnerabilidad

Si todos los modelos derivan de GPT-4, **un sesgo o error en GPT-4** se propaga a todos los productos downstream. Single point of failure.

### 7.2 Inequity y fairness

- Los modelos heredan sesgos de los datos de internet.
- Estereotipos de genero, raza, cultura embebidos en generaciones.
- Representation inequal en idiomas (ingles dominante).

### 7.3 Misuse

- Desinformacion a escala (deepfakes, noticias falsas generadas).
- Spear phishing automatizado.
- Asistencia a actores malicioss (malware, bioweapons).

### 7.4 Environment

- Entrenar GPT-3 costo ~1,287 MWh = emisiones de 5 carros en su vida util.
- GPT-4 y posteriores: ordenes de magnitud mas.

### 7.5 Economia y trabajo

- Automatizacion de tareas cognitivas (writing, coding, analysis).
- Concentracion de poder en pocas organizaciones con recursos.

### 7.6 Legalidad

- Copyright: el modelo memoriza y reproduce contenido protegido?
- Liability: quien responde por decisiones del modelo?
- Privacy: datos personales en el training set.

### 7.7 Seguridad y alineamiento

- Jailbreaks: prompts que evaden safety guardrails.
- Deceptive alignment: modelo parece alineado en entrenamiento, desalineado en deployment.
- AI safety research: investigacion activa con recursos crecientes.

---

## 8. Relacion con Transfer Learning

Foundation models son el **caso extremo** de transfer learning:

| Aspecto | Transfer Learning tradicional | Foundation Models |
|---|---|---|
| **Source task** | ImageNet classification (1K clases) | Next-token prediction en Common Crawl |
| **Source data** | 1.2M imagenes etiquetadas | Trillions de tokens sin etiquetar |
| **Tamano modelo** | ResNet-50 (25M params) | GPT-4 (~1.7T params estimado) |
| **Adaptacion** | Fine-tune ultimas capas | Fine-tune, prompt, retrieve, PEFT |
| **Target tasks** | Decenas (por dominio) | Miles (cualquier tarea NL) |

El principio es el mismo: **aprovechar representaciones pretrained**. La diferencia es **escala** y **universalidad**.

---

## 9. Futuro (2026 snapshot)

Tendencias activas al momento de escribir:

- **Modelos de razonamiento** (o1, o3, Claude con extended thinking): orquestacion multi-step, self-consistency.
- **Agentes autonomos**: foundation models que planean y ejecutan en entornos reales (Claude Code, Devin, AutoGPT).
- **Multimodalidad nativa**: texto + imagen + audio + video + code en un solo modelo.
- **Context windows extensos**: 1M+ tokens (Claude Opus 4.7, Gemini).
- **Open weights**: LLaMA, Mistral, Qwen bajan la barrera de entrada.
- **Specialized foundation models**: BioBERT para biomedicina, AlphaFold para proteinas, Claude-for-legal.
- **Efficiency**: mixture of experts (MoE), quantization, distillation.

---

## 10. Resumen

- Un **foundation model** es un modelo **pretrained a escala** sobre datos amplios con self-supervision, adaptable a muchas tareas.
- Se caracterizan por **emergence** (capacidades no programadas) y **homogenization** (un modelo base para todas las aplicaciones).
- Son posibles por la convergencia de **hardware + Transformer + datos masivos + self-supervision**.
- Adaptacion via **fine-tuning, prompting, RAG** o **PEFT**.
- Oportunidades enormes en capacidades y aplicaciones, **riesgos serios** en fairness, misuse, environment, concentracion de poder.
- Son el **limite natural** del paradigma de transfer learning.

Ver tambien: [Transfer Learning](transfer-learning) · [Data Augmentation](data-augmentation) · [Paper Bommasani 2021](/papers/foundation-models-bommasani-2021).

### Enlaces externos

- [Center for Research on Foundation Models (CRFM), Stanford](https://crfm.stanford.edu/)
- [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) -- el paper completo (214 pp).
- [Hugging Face Hub](https://huggingface.co/) -- model zoo mas grande.
- [Papers with Code](https://paperswithcode.com/methods/category/language-models) -- tracking de benchmarks.
