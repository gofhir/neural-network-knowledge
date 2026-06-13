---
title: "P2 — Inferencia generativa: cuando el modelo no sabe abstenerse"
weight: 5
---

> **Celdas 28-29 del notebook (Parte 2).** Probar `spanish-t5-small-sqac-for-qa` sobre un párrafo de Wikipedia acerca de Bélgica. Cuatro preguntas: tres responsables y una trampa. El modelo nunca se queda callado — y ahí está la lección.

## El contexto y las preguntas

```
context = "Bélgica, oficialmente Reino de Bélgica... población de 11.754.004
Su capital y la conurbación más poblada es Bruselas, mientras que su ciudad
(municipio) más poblada es Amberes."
```

| # | Pregunta | Respuesta del modelo | |
|---|---|---|---|
| 1 | ¿Cuál es la población de Bélgica? | `11.754.004` | ✅ |
| 2 | ¿En qué parte de Europa esta ubicado? | `en el noroeste europeo` | ✅ |
| 3 | ¿Cuál es la la ciudad más poblada? | `amberes` | ✅ |
| 4 | ¿Cuál es la cápital de Alemania? | `11.754.004` | ❌ **alucinación** |

Tres aciertos y un fallo. Pero los aciertos no son extracciones limpias y el fallo no es un fallo cualquiera: cada respuesta revela algo sobre cómo *genera* un modelo seq2seq.

## La firma generativa (pregunta 2)

A *"¿En qué parte de Europa esta ubicado?"* el modelo respondió **`en el noroeste europeo`** — con la preposición `en el` incluida.

Ese detalle es la **firma del paradigma generativo**. Un modelo extractivo (como el BERT de la [Parte 1](inferencia-extractiva)) sólo puede devolver un *span* literal del texto: predice un índice de inicio y uno de fin sobre los tokens del contexto, así que su respuesta máxima posible sería `noroeste europeo` — el sustantivo desnudo. No tiene forma de inventar el `en el`.

T5, en cambio, no apunta a posiciones: **decodifica texto token a token** desde su vocabulario. Por eso puede envolver el hecho en una frase fluida, agregar conectores, conjugar verbos. Aquí la diferencia es cosmética y casi imperceptible, pero es la *misma maquinaria* que más abajo le permitirá inventar una respuesta entera de la nada. Generar texto fluido y alucinar son la misma capacidad vista desde dos lados.

## Comprensión real, no saliencia (pregunta 3)

A *"¿Cuál es la la ciudad más poblada?"* respondió **`amberes`**, y eso es más fino de lo que parece.

El párrafo menciona dos ciudades con dos roles distintos: **Bruselas** es "la capital y la conurbación más poblada", **Amberes** es "su ciudad (municipio) más poblada". Bruselas es la respuesta más *saliente* — aparece primero, está adornada con "capital", suena más importante. Un modelo que respondiera por reflejo de saliencia habría dicho Bruselas. T5 distinguió que la pregunta apunta a *ciudad (municipio) más poblada* y entregó Amberes: leyó el matiz correcto.

> La minúscula (`amberes` en vez de `Amberes`) no es un error de comprensión: es la **normalización del tokenizer**, que pasó el texto a minúsculas. El contenido es exacto.

## La alucinación estrella (pregunta 4)

Aquí está la trampa. Pregunté por **la capital de Alemania** — un país que **no aparece en el contexto**, que habla sólo de Bélgica. La respuesta correcta era abstenerse: "el texto no lo dice". El modelo respondió **`11.754.004`**.

Eso no es la capital de nada. Es la **población de Bélgica** — el número de la pregunta 1. El modelo no sólo inventó: inventó una respuesta que ni siquiera es del *tipo* correcto. A una pregunta de "¿cuál es la capital?" devolvió una cifra de habitantes. Es un fallo más grave que la alucinación de [BLIP en el lab anterior](/laboratorios/lab-23/modos-de-fallo) (que ante un ornitorrinco respondió "monkey"): allá al menos el tipo de respuesta era correcto — un animal por otro animal. Aquí la categoría está rota.

**Hipótesis: pattern-matching de la plantilla sintáctica.** Mira las preguntas 1 y 4 lado a lado:

- P1: *¿Cuál es la **población** de **Bélgica**?*
- P4: *¿Cuál es la **cápital** de **Alemania**?*

Tienen la **forma idéntica** — `¿Cuál es la ___ de ___?`. La P1 sí era anclable en el contexto y el modelo la respondió con `11.754.004`. Ante la P4, que *no* puede anclarse en ningún span del texto, el modelo parece haberse aferrado a la **estructura superficial** de la pregunta en lugar de a su contenido, y reprodujo la respuesta de la única pregunta estructuralmente idéntica que sí había podido contestar. Un atajo espurio: igualó *forma* con *forma* porque no tenía cómo igualar *significado* con *contenido*, y por diseño nunca tuvo la opción de no responder.

## El contraste central del laboratorio

Las dos partes del lab plantean la misma trampa — una pregunta deliberadamente no respondible — y producen comportamientos opuestos:

| | **Parte 1** (extractivo) | **Parte 2** (generativo) |
|---|---|---|
| Modelo | BERT span prediction | T5 seq2seq |
| Dataset de fine-tuning | SQuAD **v2.0** (con *unanswerable*) | **SQAC** (sin preguntas sin respuesta) |
| Pregunta trampa | "¿comida típica en Ecuador?" | "¿capital de Alemania?" |
| Respuesta | `empty` (**abstención**) | `11.754.004` (**alucinación**) |
| Por qué | Aprendió a predecir "sin respuesta" | Nunca vio un ejemplo de "no sé" |

Y la trampa de la Parte 2 es **más adversarial**: "¿capital de Alemania?" usa palabras que *sí* están en el universo del contexto (capital, un país europeo), pero del **país equivocado**. Tiene todas las pistas léxicas correctas apuntando al objetivo incorrecto — mucho más difícil de detectar que una pregunta sobre un tema totalmente ajeno como la comida ecuatoriana.

## La lección

El QA generativo **no sabe abstenerse**: genera *siempre* algo, exactamente como BLIP en el [lab-23](/laboratorios/lab-23/modos-de-fallo). La causa raíz es de datos, no de arquitectura: **SQAC no contiene preguntas sin respuesta**, así que el modelo nunca aprendió a decir "no sé". SQuAD v2.0 sí las contiene, y por eso el BERT de la Parte 1 pudo devolver `empty`.

Lo más importante es que las dos caras son **inseparables**. La misma flexibilidad que le permitió a T5 reformular ("en el noroeste europeo") y entender un matiz fino (Amberes ≠ Bruselas) es la que lo deja sin frenos ante una pregunta imposible. No se puede tener la flexibilidad sin la propensión a alucinar — **son la misma capacidad** en este diseño. Si el dominio es crítico, la abstención no es un lujo: es un requisito que hay que *entrenar*, eligiendo un dataset con ejemplos sin respuesta (la sección de abstención en [/fundamentos/question-answering](/fundamentos/question-answering)) y midiéndola explícitamente (ver [/fundamentos/qa-evaluation-metrics](/fundamentos/qa-evaluation-metrics)).

---

**Anterior:** [Arquitectura generativa](arquitectura-generativa) · **Siguiente:** [Actividad y comparación de paradigmas](actividades-generativo)
