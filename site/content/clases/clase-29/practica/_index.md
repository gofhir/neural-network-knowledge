---
title: "Practica desde 0 - Modelos Generativos en Visión"
weight: 30
sidebar:
  open: true
---

La clase 29 recorre las familias de **modelos generativos** para visión: VAE, GAN, difusión y Latent/Stable Diffusion. Esta práctica implementa los **tres paradigmas fundacionales** en **mínima escala** y sin librerías especializadas (nada de `diffusers` ni `torchgan`), para entender por dentro qué los distingue: el **VAE** (autoencoder probabilístico), la **GAN** (juego adversarial) y un **modelo de difusión DDPM** (predecir el ruido). Cada camino replica el mismo modelo en **triple framework** (PyTorch, TensorFlow y JAX) para ver cómo cada uno expresa las mismas ideas.

## Caminos

{{< cards >}}
  {{< card link="01-vae-desde-cero" title="01 - VAE desde cero" subtitle="Encoder→(μ,σ), reparameterization, ELBO (reconstrucción + KL) en PyTorch, TensorFlow y JAX" icon="code" >}}
  {{< card link="02-gan-desde-cero" title="02 - GAN desde cero" subtitle="Generador vs discriminador, entrenamiento alternado, truco non-saturating, mode collapse" icon="code" >}}
  {{< card link="03-difusion-ddpm-desde-cero" title="03 - Difusión (DDPM) desde cero" subtitle="Schedule β_t, forward q_sample, predecir el ruido ε_θ, sampling reverse en triple framework" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 23 - VQA/CLIP](../../clase-23): CLIP es el text encoder de Stable Diffusion.
- [Clase 09 - CNN](../../clase-09) y nociones de autoencoders.
- Python intermedio y NumPy; PyTorch básico. Útil: nociones de TensorFlow/Keras y JAX.
- GPU **no necesaria**: los toy datasets (MNIST pequeño, two moons 2D) corren en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - VAE | PyTorch 2.x | TensorFlow 2.x, JAX |
| 02 - GAN | PyTorch 2.x | TensorFlow 2.x, JAX |
| 03 - Difusión (DDPM) | PyTorch 2.x | TensorFlow 2.x, JAX |

## El hilo conductor

Tres formas distintas de aprender a generar:

1. **VAE**: comprime a un latente probabilístico y reconstruye; entrena maximizando el ELBO. Muestras diversas pero borrosas.
2. **GAN**: un generador y un discriminador compiten; sin verosimilitud explícita. Muestras nítidas pero con riesgo de *mode collapse*.
3. **Difusión (DDPM)**: añade ruido gradualmente y aprende a invertirlo prediciendo el ruido. Calidad y cobertura altas, pero *sampling* lento.

Es exactamente el **trilema generativo** (calidad / velocidad / cobertura) que la clase usa para comparar las familias.

---

**Ver tambien:** [Clase 29 - Teoria](../teoria) · [Clase 29 - Profundizacion](../profundizacion) · Fundamentos: [Modelos Generativos](/fundamentos/modelos-generativos) · [Modelos de Difusión](/fundamentos/modelos-de-difusion).
