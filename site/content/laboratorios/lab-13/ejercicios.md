---
title: "Ejercicios Prácticos"
weight: 40
math: true
---

> Enunciados literales de las actividades del notebook 3 (`Practico_clase_13_parte_3.ipynb`). Las respuestas razonadas estan en [resolucion](resolucion).

Las dos actividades evaluadas del lab-13 estan en el tercer notebook (Teacher Forcing). Ambas son preguntas conceptuales que el alumno responde tras ejecutar el notebook completo y observar la diferencia de comportamiento entre el modelo Seq2Seq con y sin teacher forcing.

---

## Actividad 1.1

**Enunciado** *(notebook 3, cell 34)*:

> Comparando con modelo Seq2Seq **SIN** atención, ¿esta metodología hace que el entrenamiento converja más rápido o más lento?

Para responder esta pregunta el alumno necesita:

1. Haber ejecutado el modelo Seq2Seq sin attention (parte 1) y haber observado su curva de loss durante el entrenamiento.
2. Haber ejecutado el modelo Seq2Seq con teacher forcing (parte 3) y haber observado su curva de loss.
3. Comparar cualitativamente las dos curvas de loss en las primeras epochs y justificar la respuesta.

---

## Actividad 1.2

**Enunciado** *(notebook 3, cell 36)*:

> Explique en palabras simples a que se debe el cambio visto en la velocidad de converjencia.
>
> *HINT: Base su respuesta en el modo en que se entrega el input al decoder.*

La pista del enunciado apunta directamente al mecanismo de teacher forcing: durante el entrenamiento, el decoder recibe como input el **token de ground-truth** del paso anterior en lugar de su propia prediccion. La respuesta debe explicar como esto afecta la senal de gradiente y la calidad de los inputs al decoder a lo largo del entrenamiento.

---

> **Nota:** las respuestas razonadas estan en [resolucion](resolucion). Los outputs concretos del notebook (curvas de loss, accuracy, ejemplos de traduccion) que sustentan las respuestas se integran en Fase 2 cuando Roberto ejecute el notebook en Colab.
