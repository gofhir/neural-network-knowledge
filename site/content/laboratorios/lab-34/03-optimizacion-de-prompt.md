---
title: "Optimización de prompt (DSPy / GEPA)"
weight: 3
---

La tercera palanca cierra la escalera de intervención volviendo a **no tocar el modelo** — pero ahora mejorando automáticamente la **instrucción**. Es la sistematización del *prompt engineering*.

> **Nota:** el lab indica que esta sección **no es necesario ejecutarla** (requiere una API key de OpenRouter y mucho cómputo; la demo corrió en un servidor externo). El código está para leerlo y entender el flujo; los resultados están capturados en el notebook.

## El problema: el prompt engineering es artesanal

El comportamiento de un LLM es exquisitamente sensible al prompt — se vio crudo en el [bloque de tool use](01-uso-de-herramientas), donde una instrucción ambigua ("2 turnos") paralizó al modelo. Pero escribir prompts es hoy un arte manual: subjetivo, no escalable, frágil. La pregunta: *¿y si pudiéramos **optimizar** el prompt automáticamente*, como optimizamos los pesos de una red?

## DSPy: "programar, no promptear"

**DSPy** (Stanford) trata los prompts como **parámetros optimizables**. En vez de escribir el prompt a mano, defines una **firma (signature)**: qué entra y qué sale.

```python
class FacilitySupportAnalyzerUrgency(dspy.Signature):
    """Read the provided message and determine the urgency."""
    message: str = dspy.InputField()
    urgency: Literal['low', 'medium', 'high'] = dspy.OutputField()
```

- El **docstring** es el prompt inicial (aquí, trivial).
- **`Literal['low','medium','high']`** restringe la salida a un conjunto cerrado y la valida.

{{< callout type="info" >}}
**`Literal` = value set de FHIR.** Ese `Literal[...]` es idéntico a un *binding* a un value set en FHIR: restringes un campo a códigos válidos. DSPy te da validación de salida estructurada y tipada sobre un LLM — justo lo que necesitas para que un modelo produzca un `status` que solo puede ser `active|inactive|entered-in-error`.
{{< /callout >}}

## GEPA: optimización evolutiva con reflexión

**GEPA** (Genetic-Pareto) es el optimizador. Su mecánica:

```
1. Evalúa un prompt sobre ejemplos de entrenamiento → score con una métrica.
2. Un LLM "reflexivo" (más potente) MIRA los fallos y su feedback textual.
3. Propone un prompt MODIFICADO basado en esa reflexión (mutación).
4. Evalúa el nuevo prompt. Si es mejor, lo conserva. Repite.
```

Tres ideas clave:

1. **Reflexión en lenguaje natural (no gradientes).** El texto no es diferenciable; GEPA usa un LLM potente que *lee los errores y razona* cómo mejorar el prompt. En vez de "el gradiente apunta acá", es *"estos ejemplos fallaron porque el prompt no aclaraba X"*.
2. **El feedback textual es el corazón.** No solo un score (1/0), sino un mensaje que **explica el error**: *"clasificaste como 'low' pero era 'high'; piensa cómo habrías razonado"*. Es "backpropagation en lenguaje natural" — semánticamente rico, eficiente en muestras.
3. **Dos modelos, roles distintos.** Un modelo **objetivo** chico (`ministral-3b`, ejecuta la tarea) y un modelo **reflexivo** grande (`gemini-2.5-flash`, propone las mejoras). El grande *enseña* al chico escribiéndole mejores instrucciones.

## El resultado: el prompt evolucionado

El contraste es tan dramático como el de LoRA. El prompt **inicial** era una línea:

> *"Read the provided message and determine the urgency."*

El prompt **optimizado por GEPA** es una instrucción detallada que el sistema descubrió solo:

> *"...**High Urgency Indicators**: 'urgent', 'immediate', 'critical', 'emergency', 'safety concern'... **Medium**: 'persistent issue', 'minor leak', 'malfunction'... **Low**: 'inquiry', 'information request'..."*

GEPA **descubrió automáticamente** qué palabras clave señalan cada nivel — algo que un humano tendría que deducir a mano. El programa optimizado (con las signatures envueltas en `dspy.ChainOfThought`) alcanza **84.3%** de precisión en el test set, con un modelo de solo 3B.

{{< callout type="info" >}}
**La síntesis de los tres bloques.** GEPA logró con *optimización de prompt* (barato, sin tocar pesos) una mejora comparable al *fine-tuning*. Esto refuerza la **escalera de intervención**: muchas veces el prompt es la palanca, y optimizarlo automáticamente evita el costo del fine-tuning. Para tu trabajo: antes de fine-tunear un modelo para extraer FHIR, valdría probar si GEPA encuentra un prompt suficientemente bueno — mucho más barato de mantener que un adaptador LoRA.
{{< /callout >}}
