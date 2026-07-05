---
title: "Cuestionario resuelto (evaluado)"
weight: 3
---

Las tres preguntas evaluadas (10 pts), respaldadas con los experimentos de la sección de exploración.

## Pregunta 1 (4 pts) — El "Generative Learning Trilemma"

El **generative learning trilemma** ([Xiao, Kreis & Vahdat, NVIDIA 2021](/papers/diffusion-gan-xiao-2021), *"Tackling the Generative Learning Trilemma with Denoising Diffusion GANs"*) plantea que todo modelo generativo idealmente busca **tres propiedades a la vez**, pero las familias clásicas solo alcanzan **dos de las tres**:

1. **Alta calidad de muestras** — realistas y nítidas, fieles a la distribución real (sin borrosidad ni artefactos).
2. **Cobertura de modos / diversidad** — capturar toda la variedad de la distribución, sin colapsar a pocos ejemplos (*mode collapse*).
3. **Muestreo rápido** — generar barato (idealmente una pasada, no cientos de iteraciones).

| Familia | Calidad | Diversidad | Velocidad | Sacrifica |
|---|---|---|---|---|
| **GANs** | Alta | Baja (mode collapse) | Rápida (1 forward) | **diversidad** (+ entrenamiento inestable) |
| **VAEs / Normalizing Flows** | Menor (borrosas) | Buena | Rápida | **calidad** |
| **Difusión (DDPM)** | Alta | Alta | Lenta (cientos de pasos) | **velocidad** |

Ninguna familia clásica logra los tres vértices simultáneamente. La difusión resolvió el problema histórico de calidad + diversidad de GANs/VAEs, pero a costa de ser lenta.

**Evidencia en el lab:** la calidad de Stable Diffusion depende de [`num_inference_steps`](../01-stable-diffusion-parametros) — con 5 pasos la imagen sale borrosa, y recién con ~50-100 se estabiliza. Ese costo iterativo **es** la debilidad del vértice "velocidad". Los noise schedulers (que a 50 pasos convergen) y los modelos destilados atacan ese vértice. El paper del trilemma propone las **Denoising Diffusion GANs**: modelar cada paso de denoising con un GAN para dar pasos grandes (rápido) sin perder calidad ni diversidad.

## Pregunta 2 (4 pts) — Difusión latente

La **difusión latente** ([Rombach et al. 2022](/papers/latent-diffusion-rombach-2022), la arquitectura base de Stable Diffusion) aplica el proceso de difusión **no sobre píxeles**, sino sobre un **espacio latente comprimido** aprendido por un autoencoder (VAE):

1. Un **encoder** VAE comprime la imagen: `512×512×3` (~786k valores) → latente `64×64×4` (~16k valores, **~48× menor**).
2. La **difusión** (forward + reverse con el U-Net) opera **en ese latente**, no en píxeles.
3. Un **decoder** VAE reconstruye la imagen final de alta resolución.

**Ventajas:**

1. **Eficiencia computacional y de memoria** — el U-Net procesa ~48× menos datos por paso. Es lo que hace viable correr Stable Diffusion en GPUs de consumo (todo este lab corrió en una GPU de Colab precisamente por esto; difundir en píxeles no cabría).
2. **Velocidad** — cada paso de denoising sobre `64×64` es mucho más rápido que sobre `512×512`.
3. **Separación percepción/semántica** — el VAE maneja los detalles perceptuales de alta frecuencia; la difusión se concentra en la estructura semántica en el latente.
4. **Democratización** — frente a modelos que difunden en píxeles (DALL·E 2, Imagen), que requerían clusters, la difusión latente llevó la generación de alta calidad al hardware accesible.

El costo es una ligera pérdida de reconstrucción del VAE (detalles muy finos), un trade-off ampliamente favorable.

## Pregunta 3 (2 pts) — Modelos generativos en el ámbito propio

En el área de **datos e interoperabilidad clínica** (estándar FHIR) de un centro oncológico, incluyendo un sistema **MDM** (Master Data Management) que unifica registros duplicados de pacientes, los modelos generativos tienen dos aplicaciones justificadas:

**1. Datos sintéticos que preservan la privacidad.** Los datos de pacientes tienen fuertes restricciones legales y éticas. Un modelo generativo crea datos sintéticos (imágenes médicas o registros tabulares/FHIR) que mantienen la distribución estadística real **sin corresponder a ningún paciente real**, habilitando entrenar modelos, probar pipelines y compartir datos sin exponer PII. En el sistema MDM, permite generar **registros sintéticos con variaciones controladas** (errores de tipeo, formatos distintos de nombres/direcciones, duplicados parciales) como banco de pruebas realista y seguro para evaluar el matching y el blocking, sin datos productivos.

**2. Aumentación de datos para casos escasos.** En oncología, muchas patologías son poco frecuentes → datasets pequeños y desbalanceados. Los modelos generativos aumentan esos datos con variaciones sintéticas realistas de los casos raros, para entrenar clasificadores/detectores sin sobreajustar.

**Justificación de la elección de modelo:** para estos usos, un modelo de **difusión** antes que un GAN, por la razón del trilemma (Pregunta 1): la difusión ofrece mejor **cobertura de modos / diversidad**. En un contexto clínico no se puede permitir que el generador colapse e ignore variantes raras de una patología o de un registro; se necesita capturar toda la diversidad real. Además, el **inpainting** visto en el lab aplica directo a la **anonimización** de imágenes médicas (reemplazar regiones con identificadores) sin alterar el resto.

---

**Volver al** [índice del lab](../) **o a la** [clase 29 (teoría)](/clases/clase-29).
