---
title: "Audio Super-Resolution con redes neuronales (2017)"
weight: 469
math: true
---

{{< paper-card
    title="Audio Super-Resolution Using Neural Nets"
    authors="Volodymyr Kuleshov, S. Zayd Enam, Stefano Ermon (Universidad de Stanford)"
    year="2017"
    venue="ICLR 2017 Workshop / arXiv:1708.00853"
    arxiv="1708.00853"
    pdf="/papers/audio-superres-kuleshov-2017.pdf" >}}
Aumentar la tasa de muestreo de una señal de audio prediciendo las muestras que faltan, *"en un proceso de interpolación similar a la super-resolución de imágenes"*. La arquitectura son bloques convolucionales de submuestreo y sobremuestreo con conexiones de salto, entrenados sobre pares de audio de baja y alta calidad. Es la aplicación de audio de lo que la [Clase 44](/clases/clase-44) llama, con precisión, un ***informed guess***.
{{< /paper-card >}}

---

## Qué se inventa exactamente

Bajar la tasa de muestreo elimina, por el teorema de muestreo, todo el contenido por encima de la frecuencia de Nyquist. Esa información **no está** en la señal de baja resolución: reconstruirla es generarla.

Lo que el modelo aprende es la estructura estadística del habla y la música — que los armónicos de una voz siguen relaciones predecibles, que una consonante fricativa tiene cierta forma espectral. Con ese prior, propone contenido de alta frecuencia plausible dado lo observado. Es exactamente la situación que analiza [Super-resolución](/fundamentos/super-resolucion): el problema está mal planteado y lo que decide la salida es el prior, no los datos.

{{< concept-alert type="recordar" >}}
La diferencia con la super-resolución de imágenes no es técnica sino **perceptual y ética**. Un armónico inventado suena razonable y nadie resulta perjudicado; un rasgo facial inventado cambia quién parece ser una persona.

Por eso el intercambio distorsión-percepción, que en imagen tiene consecuencias serias cuando el resultado se usa como evidencia, en audio es un problema mayormente estético. La excepción son los contextos forenses de reconocimiento de hablante, donde vale la misma advertencia: lo que sale es la reconstrucción más probable bajo el prior del modelo, no la señal original.
{{< /concept-alert >}}

## Por qué importa para la Clase 44

Es la sexta de las siete aplicaciones. La clase la ilustra con DLSS de NVIDIA —super-resolución en tiempo real para videojuegos, hoy parte del hardware de consumo— y con este trabajo para el caso de audio, y agrega una diapositiva con la única definición que hace falta: **informed guess**. La [práctica](/clases/clase-44/practica) de la clase toma esa frase y la mide.

---

**Ver también:** [Super-resolución](/fundamentos/super-resolucion) · [Modelos de Difusión](/fundamentos/modelos-de-difusion) · [Representación de Audio](/fundamentos/representacion-de-audio) · [Clase 44 — Práctica](/clases/clase-44/practica)
