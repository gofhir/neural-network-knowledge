---
title: "P2 — Parámetros de decodificación (Actividad 3)"
weight: 7
---

> **Celdas 28-56 del notebook (Parte 2).** Explorar cómo `num_beams`, `do_sample`, `top_p` y `temperature` afectan la generación, usando la misma noticia de COVID para aislar cada efecto. Son los slides 44-47 de la [clase 22](/clases/clase-22), ahora en código. Ver el [fundamento Decoding Strategies](/fundamentos/decoding-strategies).

## 1. Número de beams (`num_beams` 5 vs 20)

| | `num_beams=5` | `num_beams=20` |
|---|---|---|
| Arranque | "the president predicts some states will reopen this month" | "the u.s. has over 637,000... deaths" |
| Saliencia | ✅ lidera con el hecho central (reapertura) | ❌ lidera con las cifras de muertes |

**Hallazgo contraintuitivo: subir de 5 a 20 beams empeoró la saliencia.** Con más beams, beam search converge hacia la secuencia de **mayor probabilidad global** — que aquí abre con el bloque de cifras (frecuente en el estilo de CNN/DM). El de 5 beams, al explorar menos, se quedó con una estructura más natural.

> Es la lección del slide 45: beam search optimiza probabilidad, no calidad percibida. "Más beams" tiene **retornos decrecientes —o negativos—**. Y la diversidad entre los 5 devueltos es baja en ambos casos: beam search no genera diversidad.

## 2. Sampling (`do_sample` False vs True)

Con `do_sample=True`, en cada paso no se elige el token más probable, sino que se **muestrea** según las probabilidades del modelo.

- `do_sample=False` (celda 39): **idéntico** al `num_beams=5` — es el grupo de control determinista.
- `do_sample=True` (celda 41): apareció una cita nueva — *"we'll be the comeback kids"* — que no salía en ninguna versión determinista. El sampling tomó caminos menos probables y trajo material nuevo del texto.

**Pero la diversidad fue modesta**, no explosiva. Dos razones:
1. `do_sample=True` + `num_beams=5` es un modo híbrido (beam-search multinomial) que mantiene la estructura de beam search.
2. La distribución de T5 para este input es muy **puntiaguda** (el modelo está muy seguro), así que samplear casi siempre cae en el token top.

> Activar el azar **no basta** para diversidad fuerte. Hace falta además **aplanar la distribución** — eso hacen `top_p` y `temperature`.

## 3. Nucleus sampling (`top_p` 0.95 vs 0.9)

`top_p` limita el conjunto de candidatos al **núcleo**: los tokens más probables cuya suma acumulada alcanza $p$, descartando la cola. El tamaño del núcleo es **dinámico** (paper de [Holtzman 2020](/papers/nucleus-sampling-holtzman-2020)).

⚠️ **Comparación contaminada (gotcha del notebook):** las celdas cambiaron **dos parámetros a la vez**:

| | Celda 45 | Celda 47 |
|---|---|---|
| `top_p` | 0.95 | 0.90 |
| `num_beams` | **5** | **20** |

Es como un experimento clínico donde se cambia el fármaco *y* la dieta a la vez — no se puede atribuir el resultado a ninguno. La diferencia observada (uno parte por la reapertura, otro por las cifras) **se debe al `num_beams`** (ya sabíamos que 5 vs 20 hace eso), no al `top_p`. El único efecto atribuible al `top_p=0.95` (núcleo más amplio) fue un fragmento descolocado: *"all of us"* pegado fuera de contexto al final de un resumen.

Conceptualmente: `top_p` más bajo → núcleo más chico → más conservador y coherente; `top_p` más alto → núcleo más amplio → más variedad pero más riesgo. El efecto fue leve porque 0.9 y 0.95 son ambos amplios y la distribución es puntiaguda.

## 4. Temperatura (`temperature` 0.6 vs 1.5)

La temperatura $T$ divide los logits antes del softmax:

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T < 1$ (0.6): **agudiza** la distribución → conservador, casi determinista.
- $T > 1$ (1.5): **aplana** la distribución → los tokens improbables ganan chance → diverso.

| | `temperature=0.6` | `temperature=1.5` |
|---|---|---|
| Cómo dice "EE.UU." | "the **u.s.** has" | "the **country** has" (en los 5) |
| Citas | solo "we want to get our country back" | + "we'll be the comeback kids, all of us" |

**El que más diversidad introdujo.** Con `temp=1.5`, los 5 resúmenes cambiaron *"the u.s. has"* por *"the country has"* — la **única paráfrasis sistemática de todo el experimento** (el token "country" ganó chance al aplanarse la distribución). Y trajo más variedad de citas. Costo: un pequeño desliz (Resumen 4 atribuyó una cita a "tells governors").

> **Por qué no rompió todo:** sigue combinada con beam search (`num_beams=20`), que "sujeta" el resultado y mantiene coherencia. Con `temperature=1.5` y `num_beams=1` (sampling puro) sí se verían frases rotas.

## Síntesis: ¿cuál notó mayor diversidad? (celda 55)

**La temperatura.** Al subirla a 1.5 fue cuando se notó la mayor variedad: el modelo reemplazó palabras por sinónimos ("u.s." → "country") y usó citas distintas, algo que `num_beams`, `do_sample` y `top_p` no lograron. Conecta con la conclusión de la clase: *"greedy/beam para constrained, top-p para open-ended, temperature para diversidad"*.

---

**Anterior:** [generación cualitativa (Act. 2)](generacion-cualitativa) · **Siguiente:** [evaluación ROUGE (Act. 4)](evaluacion-rouge)
