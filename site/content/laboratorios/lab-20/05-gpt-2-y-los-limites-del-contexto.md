---
title: "GPT-2 y los límites del contexto"
weight: 5
math: true
---

La última sección del laboratorio cambia de familia: de encoders bidireccionales a un decoder causal. Dos celdas de generación y tres preguntas conceptuales que apuntan a los dos límites duros de [GPT-2](/papers/gpt-2-radford-2019) — cuánto texto puede ver, y qué puede hacer sin instruction tuning.

## Las dos celdas

```python
# [52]
from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
```

```python
# [54] beam search
beam_output = model.generate(input_ids, attention_mask=..., max_length=100,
                             num_beams=4, early_stopping=True)

# [56] beam search + bloqueo de n-gramas
beam_output = model.generate(input_ids, attention_mask=..., max_length=100,
                             num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
```

La diferencia entre ambas es la línea que importa.

**`pad_token_id=tokenizer.eos_token_id`** en la carga es un parche necesario: GPT-2 **no tiene token de padding**. Fue entrenado sobre flujo continuo de texto, sin lotes de secuencias de distinta longitud que rellenar. Reutilizar el token de fin de secuencia como padding es la convención estándar, y sin ella la generación emite advertencias en cada llamada.

## `no_repeat_ngram_size` y por qué hace falta

Sin esa restricción, la búsqueda por haces produce texto que **se repite**. Es un comportamiento documentado y bien entendido: beam search maximiza la probabilidad de la secuencia completa, y en un modelo de lenguaje las secuencias de alta probabilidad son las repetitivas.

$$\arg\max_{y} \; P(y \mid x) \quad \longrightarrow \quad \text{texto probable} \ne \text{texto natural}$$

El habla humana no es la secuencia más probable: es sorprendente de forma sistemática, con una tasa de información aproximadamente uniforme. Un decodificador que busca el máximo produce el equivalente textual del gris promedio.

`no_repeat_ngram_size=3` prohíbe repetir cualquier trigrama ya emitido. Es un parche efectivo y también un instrumento romo: en un texto largo hay trigramas que **deben** repetirse —nombres propios, términos técnicos— y prohibirlos fuerza al modelo a parafrasear donde no debería.

La alternativa que la práctica adoptó después es el **muestreo con núcleo** —*nucleus sampling*, con `top_p`—, que ataca la causa en lugar del síntoma: no busca el máximo, muestrea de la masa de probabilidad. Aparece en el [Lab 22](/laboratorios/lab-22/decodificacion).

## Pregunta 11 · `max_length=10000`

El enunciado propone generar varias páginas de texto sobre "New York":

```python
model.generate(input, attention_mask=..., max_length=10000,
               num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
```

**GPT-2 tiene 1024 posiciones.**

$$\texttt{n\_positions} = 1024 \quad \ll \quad \texttt{max\_length} = 10000$$

No es un límite de memoria ni de tiempo: la matriz de embeddings posicionales tiene exactamente 1024 filas, y **la posición 1025 no está definida**. No hay vector que sumar. Según la versión de la librería, el resultado es un `IndexError` o un truncado silencioso.

{{< concept-alert type="atencion" >}}
Vale la pena distinguir tres límites que suelen confundirse:

| Límite | De dónde viene | Qué pasa al excederlo |
|---|---|---|
| **Posiciones** (1024 en GPT-2) | Filas de la matriz de embeddings posicionales | No existe el vector — error o truncado |
| **Memoria** | Atención $O(N^2)$ en las activaciones | OOM |
| **Calidad** | Cuánto contexto largo vio en entrenamiento | Degradación gradual |

El de esta pregunta es el primero, y es el único de los tres que es **discreto**: no se degrada, simplemente no existe.
{{< /concept-alert >}}

**Soluciones concretas:**

- **Ventana deslizante.** Generar en tramos de ~512 tokens y usar la cola del tramo anterior como prompt del siguiente. Funciona, con una limitación real: la coherencia es local. Sobre varias páginas el texto **deriva temáticamente**, porque nada garantiza que el tramo 8 recuerde lo que dijo el tramo 1.
- **Un modelo con contexto mayor.** Es la solución de fondo, y la que la historia tomó: los modelos posteriores ampliaron el contexto por órdenes de magnitud, hasta volver la pregunta obsoleta.

## Pregunta 10 · Lo que un modelo sin instruction tuning puede hacer

> *Los GPT completan texto según lo que estiman probable. ¿Es esto un impedimento para tareas dirigidas como summarization?*

De las tres opciones, la correcta es **"es un impedimento en algunos casos, pero no en otros"**, y la razón es más interesante que la respuesta.

GPT-2 **sí** puede resumir con el truco de agregar `TL;DR:` al final del texto. El propio paper lo reporta como capacidad zero-shot. Pero el mecanismo no es el que sugiere la palabra "instrucción":

**`TL;DR:` aparece en el corpus de entrenamiento seguido de resúmenes reales.** El modelo no interpreta una orden — reconoce un patrón textual y continúa la distribución que aprendió. Es completado de texto todo el tiempo; lo que cambia es que el prompt activa una regularidad útil.

De ahí se sigue el límite: **funciona para las tareas cuyo formato quedó representado en el corpus, y no para las demás**. Y aun en las que funciona, es inconsistente — la misma instrucción con otras palabras puede no activar nada.

Cerrar esa brecha es exactamente el aporte de [InstructGPT](/papers/instructgpt-ouyang-2022): entrenar explícitamente para seguir instrucciones, con demostraciones humanas y RLHF. Por eso ChatGPT aparece al final de la [clase 20](/clases/clase-20) y no al principio: no es un GPT-2 más grande, es un GPT alineado a una tarea distinta.

## Preguntas 12–13 · La asimetría del español

Llevar ambas arquitecturas al español, con recursos acotados:

**BERT — directo.** Cambiar el identificador del checkpoint por [BETO](/papers/beto-canete-2020). La arquitectura es idéntica, así que todo el código sigue funcionando. El laboratorio ya lo hizo en las preguntas 5-8.

**GPT-2 — posible, con menos opciones.** `PlanTL-GOB-ES/gpt2-base-bne` o `DeepESP/gpt2-spanish`. Ambos más pequeños y entrenados sobre corpus considerablemente menores que sus equivalentes en inglés.

La asimetría tiene una explicación económica. Un encoder de 110 M de parámetros está al alcance de un grupo universitario —BETO se entrenó con TPUs donadas por el programa TFRC de Google—. Un decoder generativo competitivo exige un orden de magnitud más de cómputo y datos, y en 2020 el retorno académico de publicarlo era menor que el de publicar un encoder que la comunidad iba a usar de inmediato en tareas de clasificación.

Esa asimetría marcó el ecosistema hispanohablante durante años, y solo empezó a cerrarse cuando los modelos multilingües grandes hicieron irrelevante la pregunta.

## El cierre del laboratorio

Las dos familias que el notebook recorre —encoders bidireccionales y decoders causales— se separaron en 2018 y volvieron a converger después. Los encoders ganaron las tareas de comprensión; los decoders, las de generación. Hoy los modelos que dominan ambas son decoders escalados con instruction tuning.

El laboratorio se detiene justo antes de esa convergencia, y eso lo hace un buen retrato del momento: cinco modelos, cuatro convenciones de tokenización incompatibles, dos paradigmas de entrenamiento y ninguna forma obvia de decidir cuál iba a ganar.

---

**Volver al** [índice del laboratorio](/laboratorios/lab-20).
