---
title: "Lab 34 - Razonamiento: tool use, LoRA y optimización de prompt"
weight: 340
sidebar:
  open: true
---

**Profesor:** Sebastián Amenábar
**Curso 3 / Tópicos de profundización**
**Notebook origen:** `clase_34/material/Laboratorio/DiplomadoRazonamientoV18.ipynb`
**Notebook ejecutado:** [lab34.ipynb](/notebooks/lab34.ipynb) · [HTML](/notebooks-html/lab34.html)

## Encuadre

La contraparte práctica de la [clase 34](/clases/clase-34): **cómo hacer que un LLM razone y actúe mejor en una tarea específica, sin entrenar un modelo gigante desde cero**. El lab presenta tres palancas —de la más barata a la más costosa— y las aplica a un mismo problema: **traducir del inglés al Lakota**, una lengua de bajos recursos donde el modelo "de fábrica" alucina.

La lógica es una **escalera de intervención**: intenta resolver la tarea sin tocar el modelo (herramientas o mejor prompt), y solo si no basta, modifica sus pesos.

| Palanca | Qué modifica | Costo |
|---|---|---|
| **Tool use** | nada del modelo — le das una herramienta externa (un diccionario) | inferencia |
| **PEFT / LoRA** | pocos pesos nuevos (adaptadores de bajo rango) | entrenamiento ligero |
| **Optimización de prompt** (GEPA) | nada del modelo — optimiza la instrucción | búsqueda evolutiva |

## Resultados consolidados (medidos en el notebook)

| Bloque | Hallazgo |
|---|---|
| **Tool use (base)** | El modelo base **se paralizó**: 0 tool calls, adivinó palabras, sin traducción |
| **LoRA** | Post-fine-tuning: **7 tool calls correctas**, tradujo; 6.67% de params, loss 2.58→1.95 sin overfitting |
| **GEPA** | Prompt de 1 línea → detallado automáticamente; **84.3%** con modelo de 3B |
| **Actividad 2 (boleta)** | Extracción a JSON correcta en estructura, **error de separador de miles** (2.100 → 2.1) |

### Las lecciones del lab

1. **Dar una herramienta ≠ usarla bien.** El modelo base, con el diccionario disponible, se paralizó razonando y no lo usó. El fine-tuning (LoRA) fue lo que cimentó el comportamiento correcto.
2. **La escalera de intervención.** El mismo fracaso motivó *dos* soluciones desde ángulos distintos: fine-tuning (tocar pesos) y optimización de prompt (tocar la instrucción). Ambas funcionaron.
3. **El over-thinking es real.** Un modelo con razonamiento activo puede deliberar hasta paralizarse — el lado oscuro del test-time compute.
4. **La extracción con LLM falla en convenciones locales.** El VLM leyó "2.100" pesos como 2.1 (decimal anglosajón). La corrección: instruir la convención en el prompt + validar que las cuentas cuadren.
5. **Todo mapea a FHIR.** Tool use = validador/terminología consultable; LoRA = adaptar a terminología clínica; boleta→JSON = extraer documentos a recursos estructurados.

## Bloques del lab

{{< cards >}}
  {{< card link="01-uso-de-herramientas" title="Uso de herramientas (tool use)" subtitle="ReAct, el diccionario Lakota con fuzzy matching, el system prompt anti-alucinación, y el fracaso instructivo del modelo base (0 tool calls)" icon="cube-transparent" >}}
  {{< card link="02-peft-lora" title="Fine-tuning eficiente con LoRA" subtitle="Descomposición de bajo rango, por qué ahorra memoria, el entrenamiento (6.67% params, sin overfitting) y el contraste dramático: 0 → 7 tool calls" icon="adjustments" >}}
  {{< card link="03-optimizacion-de-prompt" title="Optimización de prompt (DSPy / GEPA)" subtitle="Programar en vez de promptear, signatures tipadas (Literal = value set), reflexión en lenguaje natural, y el prompt evolucionado (84.3%)" icon="sparkles" >}}
  {{< card link="04-actividades-multimodal" title="Actividades: conceptual y multimodal" subtitle="Contextos largos y modo razonamiento (Act 1), Qwen3-VL: descripción, bounding boxes y boleta→JSON con el hallazgo del separador de miles (Act 2)" icon="photograph" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/razonamiento" title="Razonamiento" subtitle="El fundamento transversal: sistemas 1/2, causalidad, los límites del deep learning asociativo" icon="book-open" >}}
  {{< card link="/fundamentos/chain-of-thought" title="Chain-of-Thought" subtitle="Razonar paso a paso: el mecanismo detrás del modo -Thinking y ReAct" icon="book-open" >}}
  {{< card link="/fundamentos/test-time-compute" title="Test-time compute" subtitle="Gastar cómputo en inferencia para razonar mejor — y su lado oscuro (over-thinking)" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-34" title="Clase 34 - Teoría" subtitle="Razonamiento: causalidad, sistemas 1/2, Chain-of-Thought, self-consistency, test-time compute, DeepSeek-R1" icon="academic-cap" >}}
  {{< card link="/clases/clase-34/profundizacion" title="Profundización" subtitle="Math: GRPO, self-consistency, escalera de causalidad de Pearl" icon="beaker" >}}
  {{< card link="/clases/clase-34/practica" title="Práctica de clase" subtitle="Self-Consistency y GRPO en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-33" title="Lab 33 - Imitación y DAGGER (anterior)" subtitle="El lab previo del Curso 3" icon="arrow-left" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda de las 41 celdas + las actividades resueltas (Act 1 conceptual, Act 2 multimodal con Qwen3-VL). Notebook ejecutado en Colab: tool use (fracaso del base), LoRA (0→7 tool calls, 6.67% params), GEPA (84.3%, no ejecutado localmente — requiere API key), y extracción de boleta a JSON con el hallazgo del separador de miles y su corrección. Sin papers ni fundamentos nuevos (todos de la clase 34).
