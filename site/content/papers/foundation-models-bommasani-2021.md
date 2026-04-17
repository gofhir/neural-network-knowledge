---
title: "Foundation Models (Bommasani)"
weight: 190
math: true
---

{{< paper-card
    title="On the Opportunities and Risks of Foundation Models"
    authors="Bommasani, Hudson, Adeli, Altman, Arora, ... (100+ authors)"
    year="2021"
    venue="Stanford CRFM Report"
    pdf="/papers/foundation-models-bommasani-2021.pdf"
    arxiv="2108.07258" >}}
Reporte de 214 paginas de 100+ autores del Stanford Center for Research on Foundation Models (CRFM). Define y caracteriza los **foundation models** -- modelos entrenados a gran escala con self-supervision, adaptables a multiples tareas downstream -- y ofrece un analisis exhaustivo de sus capacidades, aplicaciones, tecnologia subyacente y riesgos sociales. Es el documento de referencia que formaliza el paradigma GPT-3, BERT, DALL-E, CLIP.
{{< /paper-card >}}

---

## Contexto

Entre 2018 y 2021, la IA paso por un cambio cualitativo. Modelos como **BERT** (2018), **GPT-2/3** (2019-2020), **DALL-E** (2021) y **CLIP** (2021) mostraron capacidades **emergentes**: comportamientos complejos (in-context learning, zero-shot classification, image generation) no explicitamente programados. Al mismo tiempo, los mismos modelos base se usaban para **todas** las tareas, creando una homogeneizacion sin precedentes.

El grupo de Stanford CRFM convoca a 100+ investigadores para articular este paradigma emergente, estandarizar terminologia y enumerar oportunidades y riesgos. El termino **"foundation model"** fue acunado aqui.

---

## Ideas principales

### 1. Definicion

Un **foundation model** es un modelo que cumple:

1. **Entrenado sobre datos amplios y diversos** (tipicamente scraped de la web o datasets masivos curados).
2. **Entrenado con self-supervision** a gran escala (masked language modeling, next-token prediction, contrastive learning).
3. **Adaptable** a muchas tareas downstream mediante fine-tuning, prompting, retrieval o parameter-efficient methods.

Ejemplos: BERT, GPT-3, DALL-E, CLIP, T5, Codex.

### 2. Dos propiedades definitorias

#### Emergence

Capacidades que **emergen implicitamente** del training a escala, no son programadas. Ejemplos:

- GPT-3 hace aritmetica, traduccion y code completion sin ser entrenado explicitamente para ellas.
- CLIP clasifica imagenes de clases nunca vistas durante entrenamiento.
- GPT-4 razona multi-paso con prompts tipo "Let's think step by step".

Emergence es **fuente de entusiasmo y de ansiedad**: abre posibilidades, pero tambien crea incertidumbre sobre que mas puede emerger que no hemos detectado.

#### Homogenization

Los **mismos modelos base** se usan para todas las aplicaciones:

- BERT es el backbone de clasificacion, QA, NER, traduccion en miles de productos.
- GPT-4 alimenta ChatGPT, Copilot, asistentes medicos, tutores educativos.

Ventaja: una mejora al modelo base se propaga a todos los productos. Desventaja: **single point of failure**. Un sesgo o error del modelo base afecta a todos.

### 3. Trayectoria historica

El reporte ilustra la IA como trayectoria de crecientes emergence y homogenization:

| Era | Emerge | Se homogeniza |
|---|---|---|
| ML (1990s) | El "como" (algoritmo) | Algoritmos de aprendizaje |
| Deep Learning (2010s) | Features de alto nivel | Arquitecturas (CNN, Transformer) |
| Foundation Models (2020s) | Funcionalidades enteras | El modelo mismo |

### 4. Los tres ingredientes habilitadores

Los foundation models requirieron convergencia de:

- **Hardware**: GPU throughput 10x en 4 anos, TPUs especializadas.
- **Arquitectura Transformer** (Vaswani 2017): paralelizable, escalable, universal.
- **Datos + self-supervision**: masked LM, next token prediction, contrastive learning sobre cientos de TB de datos sin etiquetar.

### 5. Capacidades cubiertas (Parte 2)

- **Lenguaje**: comprension, generacion, traduccion.
- **Vision**: clasificacion, deteccion, generacion.
- **Robotica**: policy learning con demonstrations.
- **Reasoning & search**: multi-step problem solving.
- **Interaction**: dialogo natural.
- **Philosophy of understanding**: que significa "entender" para un modelo.

### 6. Aplicaciones (Parte 3)

- **Healthcare & biomedicine**: diagnostico asistido, drug discovery (AlphaFold).
- **Law**: analisis de contratos, search legal.
- **Education**: tutores personalizados, grading.

### 7. Tecnologia (Parte 4)

Subsecciones sobre: modeling, training, adaptation, evaluation, systems, data, security & privacy, robustness to distribution shifts, AI safety and alignment, theory, interpretability.

### 8. Society & Risks (Parte 5)

La parte mas extensa (50+ paginas):

- **Inequity and fairness**: sesgos heredados de datos de internet, disparidades de representacion.
- **Misuse**: desinformacion, spear phishing automatizado, asistencia a actores maliciosos.
- **Environment**: costo energetico enorme (GPT-3 = ~1287 MWh).
- **Legality**: copyright, liability, privacy.
- **Economics**: automatizacion cognitiva, concentracion de poder.
- **Ethics of scale**: cuando el scale mismo genera obligaciones nuevas.

---

## Conclusiones principales

{{< concept-alert type="clave" >}}
1. **Foundation models son un nuevo paradigma**, no solo "modelos mas grandes". La escala cualitativa cambia la forma en que se construye IA.
2. **Emergence y homogenization son simultaneas**, crean oportunidades y riesgos acoplados.
3. **Transfer learning + scale** es el mecanismo habilitador.
4. **Riesgos requieren atencion interdisciplinaria**: technical researchers, ethicists, policy makers, social scientists.
5. **Carencia de entendimiento**: no sabemos exactamente por que estos modelos funcionan, cuando fallan, o que pueden hacer -- eso debe cambiar antes de despliegues masivos.
{{< /concept-alert >}}

---

## Por que importa hoy

- **Estandariza el vocabulario**: "foundation model", "emergence", "homogenization" son ahora terminos canonicos.
- **Organizacion de riesgos**: la taxonomia del reporte es usada por reguladores (EU AI Act, US Executive Orders) y companies AI-safety focused.
- **Fundamento del discurso publico** sobre LLMs, alignment, x-risk.
- **Influye en research priorities**: las areas identificadas como criticas (interpretability, robustness, safety) recibieron financiamiento y talento masivos post-2021.

---

## Limitaciones

- **Longitud**: 214 paginas dificulta la lectura integral. Mas citado por secciones que como obra completa.
- **Foco en estado 2021**: muchos analisis estan desactualizados (previo a GPT-4, Claude, Gemini, modelos de razonamiento).
- **Perspectiva academica**: menos atencion a consideraciones de producto, UX, monetizacion.
- **Falta de prescripcion**: enumera problemas pero pocas soluciones concretas.

---

## Notas y enlaces

- **Navegacion recomendada**: leer Introduccion (pp 1-20), luego secciones especificas segun interes. La Introduccion sola contiene el 80% del valor conceptual.
- **Follow-ups relevantes**:
  - Bender et al. (2021) "On the Dangers of Stochastic Parrots" -- critica paralela.
  - Wei et al. (2022) "Emergent Abilities of Large Language Models".
  - Kaplan et al. (2020) "Scaling Laws for Neural Language Models" -- base empirica.
- [Stanford CRFM website](https://crfm.stanford.edu/)
- [Full text on arXiv](https://arxiv.org/abs/2108.07258)

Ver fundamentos: [Foundation Models](/fundamentos/foundation-models) · [Transfer Learning](/fundamentos/transfer-learning).
