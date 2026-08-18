---
title: "Vid2Speech: reconstruir habla desde video mudo (2017)"
weight: 468
math: true
---

{{< paper-card
    title="Vid2Speech: Speech Reconstruction from Silent Video"
    authors="Ariel Ephrat, Shmuel Peleg (Universidad Hebrea de Jerusalén)"
    year="2017"
    venue="ICASSP 2017 / arXiv:1701.00495"
    arxiv="1701.00495"
    pdf="/papers/vid2speech-ephrat-2017.pdf" >}}
Generar una señal acústica **inteligible** a partir de video mudo de alguien hablando, sin pasar por texto. El modelo es una CNN que produce características de sonido para cada cuadro a partir de sus cuadros vecinos, y de esas características se sintetiza la forma de onda. Es la primera de las siete aplicaciones que presenta la [Clase 44](/clases/clase-44), y su interés está en que **evita el paso por el lenguaje**: no transcribe para después hablar, va directo de píxeles a sonido.
{{< /paper-card >}}

---

## Por qué no pasar por texto

La ruta obvia sería leer los labios ([reconocimiento visual de habla](/fundamentos/lectura-de-labios)), obtener texto, y sintetizar voz con un TTS. Vid2Speech no lo hace, y la razón es informativa: **al pasar por texto se pierde todo lo que el texto no codifica** — la prosodia, la entonación, el ritmo, la identidad del hablante. Predecir directamente características acústicas conserva parte de esa información, que está visible en el movimiento facial.

El costo es el techo que impone el canal visual, y es el mismo que discute la [Clase 43](/clases/clase-43): los visemas colapsan fonemas que no se distinguen desde afuera. Vid2Speech opera sobre GRID, un corpus de estudio con gramática restringida, y la restricción del vocabulario es parte de lo que hace viable el resultado.

## Contexto

La clase menciona junto a este trabajo su continuación, *Vocoder-Based Speech Synthesis from Silent Videos* (Michelsanti et al., 2020), con arquitectura de convolución 3D más GRU, que reemplaza la síntesis directa de la forma de onda por la predicción de parámetros de un **vocoder** — separando explícitamente la excitación glotal de la envolvente espectral, lo que produce audio notablemente más natural.

Junto con [Speech2Face](/papers/speech2face-oh-2019), que va en la dirección inversa, forma el par simétrico con que la clase abre: **de video a audio y de audio a video**, ambos aprendidos de la correspondencia gratuita del video.

---

**Ver también:** [Speech2Face (2019)](/papers/speech2face-oh-2019) · [E2E-AVSR (2018)](/papers/e2e-avsr-petridis-2018) · [Lectura de Labios](/fundamentos/lectura-de-labios) · [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual)
