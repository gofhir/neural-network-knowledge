---
title: "Actividad resuelta (7 preguntas)"
weight: 6
---

> **Celdas 35-49 del notebook.** Las siete preguntas de opción múltiple que cierran el lab, resueltas con la justificación de la correcta y el motivo por el que cada distractor falla.

## Pregunta 1 — ¿Qué es Visual Question Answering (VQA)?

✅ **b) Un modelo que responde preguntas sobre una imagen.**

VQA recibe **dos entradas** (una imagen y una pregunta en lenguaje natural) y produce una respuesta. Es la combinación de visión y lenguaje en la dirección imagen+texto → texto.

- **a) Un modelo que genera imágenes a partir de texto.** Es la tarea *inversa* (text-to-image, tipo DALL·E). No hay pregunta ni respuesta.
- **c) Un modelo que clasifica imágenes en categorías.** Clasificación pura: ignora la pregunta, que es justamente lo que define a VQA.
- **d) Una métrica para evaluar captions.** Confunde la tarea con una métrica; eso describe a BLEU.

## Pregunta 2 — ¿Qué característica tiene VQAv2 para evitar sesgos de lenguaje?

✅ **b) Se incluyen imágenes con preguntas similares pero respuestas distintas.**

Es el **balanceo** de [Goyal et al. 2017](/papers/vqav2-goyal-2017): para cada pregunta se buscan **dos imágenes que producen respuestas distintas** (el plátano amarillo vs. el plátano verde). Así un modelo que solo memorice el prior lingüístico ("los plátanos son amarillos") falla en la mitad de los casos y se ve **obligado a mirar la imagen**.

- **a) Las preguntas tienen una longitud máxima.** La longitud es irrelevante para el sesgo; no es una propiedad de VQAv2.
- **c) Solo se permiten respuestas de tipo sí/no.** Falso: VQAv2 mezcla yes/no, conteo (número) y respuestas abiertas.
- **d) Se evalúa únicamente con BLEU.** VQA se evalúa con **accuracy** (consenso de anotadores), no con BLEU.

## Pregunta 3 — ¿Qué problemas tienen modelos VQA como Pythia?

✅ **b) Falta de composicionalidad y dependen de sesgos de lenguaje.**

[Pythia](/papers/pythia-jiang-2018) y compañía detectan objetos y conceptos sueltos, pero les cuesta la **composicionalidad**: combinar conceptos y razonar relaciones (contar, posición espacial, comparaciones). Cuando la imagen no basta, recaen en el prior del lenguaje (el mismo sesgo que VQAv2 intenta romper).

- **a)**, **c)** y **d)** describen limitaciones inventadas que no corresponden a la arquitectura ni a la evidencia empírica del modelo.

## Pregunta 4 — ¿Qué técnica genera captions más variados que Greedy Search?

✅ **a) Beam Search.**

Greedy Search elige **siempre el token más probable** en cada paso: es la opción más conservadora y repetitiva. [Beam Search](/fundamentos/decoding-strategies) mantiene $k$ hipótesis en paralelo y explora más del espacio de secuencias, produciendo captions menos rígidos.

> **Matiz honesto:** para diversidad *pura*, los métodos de **sampling** (nucleus / temperature) son aún más variados que beam search — de hecho BLIP usa nucleus sampling en su pipeline CapFilt. Pero **entre las opciones dadas**, beam search es la única respuesta correcta.

- **b) Token embedding.** Es una representación vectorial de tokens, no una estrategia de decodificación.
- **c) Top-down attention.** Es el mecanismo de atención del encoder de Pythia, no un método de generación.
- **d) BLEU.** Es una métrica de evaluación, no una técnica de decodificación.

## Pregunta 5 — Caption "a baby bird is held in a box" para un ornitorrinco

✅ **a) BLIP tiene problemas de alucinación y confunde animales poco frecuentes con categorías más comunes en su entrenamiento.**

El ornitorrinco es un caso **out-of-distribution (OOD)**: BLIP casi no lo vio en el entrenamiento, así que lo **proyecta a la clase más cercana que sí conoce** ("bird") y **fabrica contexto plausible** ("held in a box"). Es el patrón clásico de [alucinación en modelos visión-lenguaje](/fundamentos/vision-language-models).

- **b) BLIP siempre genera inglés gramaticalmente perfecto.** Aunque la gramática sea correcta, no explica el error de contenido; es irrelevante.
- **c) BLIP nunca comete errores en sus descripciones.** Este caption es precisamente el **contraejemplo** que refuta la afirmación.
- **d) El problema es solo la resolución de la imagen.** No es un tema de píxeles, sino de **distribución de entrenamiento**: la clase no estaba representada.

## Pregunta 6 — ¿Qué mide BLEU en captioning?

✅ **b) La similitud entre el caption generado y uno o varios captions de referencia.**

[BLEU](/fundamentos/bleu-metric) ([Papineni et al. 2002](/papers/bleu-papineni-2002)) mide el **solapamiento de n-gramas** entre la hipótesis y una o más referencias, con un *brevity penalty* que castiga las salidas demasiado cortas. Es **precision-oriented** (la contraparte de ROUGE, que es recall-oriented).

- **a) La capacidad de detectar objetos en la imagen.** Eso es detección de objetos, no una métrica de texto.
- **c) El accuracy de la clasificación.** BLEU no es accuracy de clasificación.
- **d) La atención del Vision Transformer.** Confunde una métrica de salida con un mecanismo interno del ViT.

## Pregunta 7 — BLIP responde "yes" a "Is the dog in front of the chair?" cuando el perro está debajo o al lado

✅ **b) El modelo se confunde con la relación espacial entre objetos.**

BLIP **detecta correctamente la presencia** del perro y la silla, pero falla en la **geometría relativa** entre ellos ("in front of"). El ViT tiende a **colapsar la profundidad 3D** de la escena, por lo que las relaciones espaciales precisas se le escapan.

- **a) No logra detectar al perro en la imagen.** Falso: sí lo detecta, el problema es la relación, no la presencia.
- **c) No puede responder preguntas binarias (sí/no).** Falso: responde "yes", o sea sí maneja el formato binario.
- **d) El error se debe a la baja resolución.** No es resolución: es **razonamiento espacial**.

## Tabla resumen

| # | Tema | Respuesta |
|---|------|-----------|
| 1 | Definición de VQA | **b** |
| 2 | Balanceo de VQAv2 contra sesgos | **b** |
| 3 | Limitaciones de Pythia | **b** |
| 4 | Decodificación más variada que greedy | **a** |
| 5 | Alucinación OOD (ornitorrinco) | **a** |
| 6 | Qué mide BLEU | **b** |
| 7 | Fallo en relación espacial | **b** |

---

**Anterior:** [Decodificación, BLEU y robustez](decoding-y-robustez) · [Volver al lab](../)
