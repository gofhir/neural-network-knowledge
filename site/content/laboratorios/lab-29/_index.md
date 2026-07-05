---
title: "Lab 29 - Modelos Generativos en Visión: Stable Diffusion con diffusers"
weight: 290
sidebar:
  open: true
---

**Profesora:** Francisca Cattan
**Fecha:** Junio 2026
**Notebook origen:** `clase_29/material/Laboratorio/Práctico_Modelos_Generativos_en_Computer_Vision.ipynb`
**Notebook ejecutado:** [lab29.ipynb](/notebooks/lab29.ipynb) · [HTML](/notebooks-html/lab29.html)

## Encuadre

La contraparte práctica de la [clase 29](/clases/clase-29): usar la librería [`diffusers`](https://huggingface.co/docs/diffusers) de HuggingFace para manipular **Stable Diffusion** (SDXL / SD 1.5) y entender sus parámetros y modos de condicionamiento. El lab tiene dos secciones:

1. **Exploración (no evaluada):** generación text-to-image y sus perillas (`num_inference_steps`, noise schedulers, `guidance_scale`), más los modos alternativos (Img2Img, Inpainting, ControlNet).
2. **Cuestionario (evaluado):** tres preguntas conceptuales — el *generative learning trilemma*, la difusión latente, y una aplicación al ámbito propio.

`diffusers` separa tres componentes que estructuran todo el lab:

| Componente | Qué hace |
|---|---|
| **Diffusion Pipeline** | la arquitectura completa (Stable Diffusion): text encoder CLIP + U-Net + VAE |
| **Noise Scheduler** | el algoritmo de muestreo (cómo se hace el denoising paso a paso) |
| **Modelo pre-entrenado** | los pesos ya entrenados, intercambiables |

La difusión genera **partiendo de ruido puro y quitándolo iterativamente**, guiada por el prompt. Stable Diffusion lo hace en un **espacio latente comprimido** (difusión latente), lo que lo vuelve eficiente — es lo que permitió correr todo el lab en una GPU de Colab.

## Las lecciones del lab

1. **`num_inference_steps` = calidad vs. tiempo con retornos decrecientes.** Con 5 pasos la imagen sale borrosa e incompleta; con ~50-100 se estabiliza; de 100 a 200 casi no cambia. Además, cambiar el número de pasos no solo refina: altera la trayectoria de denoising y puede cambiar la composición.
2. **Los noise schedulers convergen a ~50 pasos.** LMS, Euler, KDPM2 y UniPC dieron resultados casi idénticos — la elección de scheduler importa con **pocos** pasos (eficiencia), no con muchos.
3. **`guidance_scale` es el trade-off fidelidad↔calidad (CFG).** Con 0 el prompt se diluye (imagen realista genérica); ~7.5-10 es el punto dulce; con 40 la imagen se "quema" (saturación, artefactos).
4. **Los tres modos de condicionamiento desacoplan qué se conserva:** Img2Img hereda la composición completa, Inpainting edita solo una región enmascarada, ControlNet impone la estructura (bordes) manteniendo el resto libre.

## Bloques del lab

{{< cards >}}
  {{< card link="01-stable-diffusion-parametros" title="Stable Diffusion y sus parámetros" subtitle="text2img, num_inference_steps (5/100/200), noise schedulers, guidance_scale (0/10/40) con evidencia visual" icon="photograph" >}}
  {{< card link="02-modos-alternativos" title="Modos alternativos: Img2Img, Inpainting, ControlNet" subtitle="Condicionar con imagen, editar regiones enmascaradas, control estructural con bordes Canny" icon="photograph" >}}
  {{< card link="03-cuestionario" title="Cuestionario resuelto (evaluado)" subtitle="Generative learning trilemma, difusión latente, aplicación a datos clínicos/MDM" icon="academic-cap" >}}
{{< /cards >}}

## Papers y fundamentos relacionados

{{< cards >}}
  {{< card link="/papers/latent-diffusion-rombach-2022" title="Latent Diffusion (2022)" subtitle="Rombach et al. — la arquitectura de Stable Diffusion; difusión en espacio latente (Pregunta 2)" icon="document-text" >}}
  {{< card link="/papers/diffusion-gan-xiao-2021" title="Denoising Diffusion GANs (2021)" subtitle="Xiao et al. — el generative learning trilemma (Pregunta 1)" icon="document-text" >}}
  {{< card link="/papers/ddpm-ho-2020" title="DDPM (2020)" subtitle="Ho et al. — el proceso de difusión forward/reverse que gobierna los pasos" icon="document-text" >}}
  {{< card link="/fundamentos/modelos-de-difusion" title="Modelos de Difusión" subtitle="Forward/reverse, denoising, schedulers, guidance" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-29" title="Clase 29 - Teoría" subtitle="VAE, GAN, difusión, Stable Diffusion, el trilemma" icon="academic-cap" >}}
  {{< card link="/clases/clase-29/profundizacion" title="Profundización" subtitle="Math de VAE (ELBO), GAN (minimax), difusión (forward/reverse), LDM" icon="beaker" >}}
  {{< card link="/fundamentos/modelos-generativos" title="Modelos Generativos" subtitle="El panorama: VAE, GAN, flujos, difusión" icon="book-open" >}}
  {{< card link="/laboratorios/lab-28" title="Lab 28 - Aprendizaje Autosupervisado (anterior)" subtitle="UDA semi-supervisado sobre IMDB" icon="arrow-left" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Recorrido celda a celda del notebook (50 celdas) de la sección de exploración de Stable Diffusion, con evidencia visual generada propia (progresión de pasos, guidance 0→40, los tres modos de condicionamiento). Las tres preguntas del cuestionario resueltas y respaldadas con los experimentos. Notebook ejecutado en Colab (GPU) con imágenes embebidas.
