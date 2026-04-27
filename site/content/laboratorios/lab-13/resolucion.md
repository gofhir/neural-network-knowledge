---
title: "Resolución"
weight: 50
math: true
---

> Resolucion razonada de las Actividades 1.1 y 1.2 del notebook 3 + insights consolidados de las 3 partes. Los enunciados literales estan en [ejercicios](ejercicios).

---

## Actividad 1.1

**Enunciado:** ¿Comparando con modelo Seq2Seq sin atencion, esta metodologia (teacher forcing) hace que el entrenamiento converja mas rapido o mas lento?

**Respuesta:**

> [Resolucion pendiente — se completara en Fase 3 cuando Roberto entregue las curvas de loss ejecutadas en Colab para el modelo con teacher forcing y el modelo sin teacher forcing. La respuesta esperable, segun la teoria, es que **converja mas rapido** porque teacher forcing alimenta al decoder con tokens de ground-truth durante el entrenamiento, evitando el compounding de errores tempranos. Pero la afirmacion debe sustentarse con la evidencia visual de las curvas reales del notebook.]

---

## Actividad 1.2

**Enunciado:** Explique en palabras simples a que se debe el cambio visto en la velocidad de convergencia. *(Hint: el modo en que se entrega el input al decoder.)*

**Respuesta:**

> [Resolucion pendiente — se completara en Fase 3. La explicacion esperable, alineada con el hint del enunciado, es: durante el entrenamiento sin teacher forcing el decoder recibe en cada paso su propia prediccion del paso anterior (modo autoregresivo). Si el decoder esta poco entrenado y se equivoca temprano, los errores se acumulan a lo largo de la secuencia y el modelo nunca aprende a generar secuencias coherentes. Con teacher forcing, en cambio, el decoder recibe el token de ground-truth del paso anterior, lo que le permite enfocarse en aprender la transicion correcta a partir de un input "limpio". Esto produce gradientes mas informativos y por lo tanto convergencia mas rapida. La contraparte es el problema de **exposure bias** durante inference, cuando el ground-truth ya no esta disponible.]

---

## Insights consolidados de las 3 partes

[Pendiente Fase 3.]

- **Parte 1 (Seq2Seq basico):** [pendiente — se completa con observaciones reales de las curvas y ejemplos de traduccion del modelo sin attention. Hipotesis a confirmar: el modelo logra traducciones razonables en oraciones cortas pero degrada notablemente en oraciones largas, evidenciando el bottleneck del context vector unico.]

- **Parte 2 (Seq2Seq + Attention):** [pendiente — se completa con observaciones reales del modelo con Bahdanau attention. Hipotesis a confirmar: el modelo mejora significativamente en oraciones largas y el attention heatmap muestra alineamientos linguisticamente sensatos entre tokens fuente y destino.]

- **Parte 3 (Teacher Forcing):** [pendiente — se completa con la comparacion real de las curvas de loss con/sin teacher forcing. Hipotesis a confirmar: con teacher forcing converge mas rapido pero potencialmente sufre exposure bias en inference; sin teacher forcing converge mas lento pero la distribucion train/inference esta mejor alineada.]

- **Conexion con la teoria:** [pendiente — referenciar las secciones especificas de la [clase 13 - teoria](/clases/clase-13/teoria) y de la [profundizacion](/clases/clase-13/profundizacion) que respaldan los hallazgos de cada parte. Particularmente: secciones 7-9 de la teoria para Bahdanau attention; secciones de la profundizacion sobre teacher forcing y scheduled sampling.]
