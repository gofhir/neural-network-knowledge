---
title: "Uso de herramientas (tool use)"
weight: 1
---

El primer bloque del lab ataca un problema real: un LLM traduce **mal** a una lengua de bajos recursos porque apenas la vio en su entrenamiento. La solución sin tocar el modelo: darle una **herramienta** —un diccionario consultable— y enseñarle a usarla.

## El problema: el Lakota y la alucinación

La tarea que hilvana todo el lab es traducir del inglés al **Lakota** (lengua del pueblo Lakȟóta). Se elige a propósito porque es una lengua **de bajos recursos**: los LLMs vieron poquísimo Lakota, así que "de fábrica" traducen mal e **inventan** palabras plausibles pero inexistentes. Es un problema genuino —a diferencia de traducir a un idioma común, donde el modelo ya sería bueno y no se vería la mejora.

Este es exactamente el escenario del [razonamiento aumentado con acciones](/clases/clase-34): en vez de que el modelo dependa de su memoria congelada, le damos acceso a **conocimiento externo consultable**.

## Qué es el tool use (function calling)

Un LLM es un predictor de texto: todo su "conocimiento" está congelado en sus pesos. El **tool use** le da acceso a **funciones externas** —que sí pueden buscar, calcular, consultar— y el modelo aprende a **decidir cuándo llamarlas y cómo usar sus resultados**.

El patrón canónico es **ReAct** (Reason + Act): el modelo alterna pensar y actuar.

```
1. THOUGHT   → "Necesito traducir 'fox'. No estoy seguro. Debería buscar en el diccionario."
2. ACTION    → emite <tool_call><function=search_dictionary><parameter=query>fox</parameter>...
3. [pausa]   → el SISTEMA detecta la llamada, ejecuta la búsqueda real
4. OBSERVATION → "search_dictionary returned: šuŋǧíla (n) = red fox..."
5. THOUGHT   → "Bien, 'fox' es šuŋǧíla. Ahora 'jumps'..." → vuelve al paso 1
```

{{< callout type="info" >}}
**El punto sutil: el modelo NO ejecuta la herramienta.** Solo genera *texto* que parece una llamada a función (los tags `<tool_call>`). Es el código que rodea al modelo (la función `translate` del notebook) el que detecta ese texto, lo parsea, **ejecuta la función Python real**, e **inyecta el resultado** de vuelta en el prompt. Esa danza —el modelo pide, el sistema ejecuta y responde— es el corazón del tool use.
{{< /callout >}}

## La herramienta: un diccionario con fuzzy matching

El diccionario Lakota viene en formato lingüístico crudo (marcadores `\lx` lexeme, `\de` definición, `\xv` ejemplo). La clase `LakotaDictionaryTool` lo parsea y expone un `search()` que:

1. **Normaliza el texto** (paso clave): mapea `ŋ→n`, descompone diacríticos con `unicodedata.normalize("NFD", ...)` y elimina las marcas → "Wíčhaša" y "wichasa" colapsan a la misma cadena comparable.
2. **Puntúa por campo con pesos**: match exacto con el lexeme = 3.0, con la definición = 2.5, con ejemplos = 2.3–2.8. Es un ranking de evidencia.
3. **Umbraliza y rankea**: devuelve solo el top-5 sobre un threshold de relevancia.

{{< callout type="info" >}}
**La normalización es tu problema de record linkage.** El truco `unicodedata.normalize("NFD", s)` + eliminar categoría `Mn` (marcas diacríticas) es *la* técnica canónica para normalizar nombres: "José"/"Jose", "Muñoz"/"Munoz" colapsan a la misma forma. Aquí aplicado a una lengua indígena, pero transferible tal cual a matching de nombres de pacientes en FHIR MDM. Y el motor de búsqueda ponderado (normalizar → puntuar por campo → umbralizar → rankear) es el mismo *blocking + scoring* de tu arquitectura de matching.
{{< /callout >}}

## El contrato: el system prompt

El modelo sabe qué herramientas tiene por el **system prompt**, que lista las funciones (`TOOLS`, un esquema JSON estándar) y da 10 reglas de comportamiento. Las más importantes:

- **Regla 1**: usa `search_dictionary` al menos una vez antes de traducir → fuerza la fundamentación.
- **Reglas 6 y 10**: *"no inventes evidencia"* / *"tu conocimiento interno no es confiable, apóyate enteramente en el diccionario"* → **anti-alucinación explícita**.

{{< callout type="info" >}}
**Directo a FHIR.** El esquema de herramienta (`TOOLS`) es **JSON Schema** — el mismo estándar con el que defines perfiles y validaciones FHIR. Y las reglas 6/10 son justo lo que necesitas para que un LLM no alucine códigos LOINC/SNOMED: *"no inventes códigos, consúltalos siempre en la herramienta de terminología"*. Tu experiencia con validadores FHIR es ventaja directa: una herramienta bien descrita en JSON Schema es la diferencia entre que el modelo la use bien o mal.
{{< /callout >}}

## El resultado real: el modelo base FALLA

Aquí está el hallazgo didáctico del bloque. Al ejecutar la traducción de *"The quick brown fox jumps over the lazy dog"* con el modelo **base** (sin fine-tunear), el resultado fue un **fracaso instructivo**:

- El modelo entró en modo razonamiento y **se obsesionó con una ambigüedad del prompt** (la regla de "máximo 2 turnos" vs "sin límite de tool calls"), deliberando ~7.700 caracteres sobre qué significaba "2 turnos".
- **Empezó a adivinar palabras** ("Fox: `wiya` o `wica`?"), violando la regla anti-alucinación.
- **Nunca emitió una llamada válida** en el formato correcto → el bucle terminó con `Done (no tool calls)`, **sin ejecutar una sola búsqueda ni producir traducción**.

Las tres lecciones de este fracaso:

1. **Over-thinking** — el lado oscuro del test-time compute: más razonamiento no siempre es mejor; puede degenerar en parálisis (motiva la [Actividad 1, pregunta 2](04-actividades-multimodal)).
2. **La ambigüedad del prompt causa fallos** → motiva el bloque de [optimización de prompt](03-optimizacion-de-prompt).
3. **El modelo base no usa bien la herramienta** → motiva el [fine-tuning con LoRA](02-peft-lora).

{{< callout type="warning" >}}
**La lección para tu trabajo.** Dar acceso a una herramienta **no garantiza** que el modelo la use bien. Es exactamente lo que temes en un sistema FHIR: un LLM que, en vez de consultar el validador/terminología, se paraliza razonando o inventa códigos. El tool use es poderoso pero no es "plug and play" — este fracaso lo demuestra, y es lo que la siguiente parte (LoRA) resuelve.
{{< /callout >}}
