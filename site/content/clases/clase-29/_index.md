---
title: "Clase 29 - Modelos Generativos en Visión"
weight: 290
sidebar:
  open: true
---

**Profesora:** Francisca Cattan
**Curso 3:** Relacional, GANs, RL, Meta-Learning, Razonamiento y Memoria

Clase sobre **modelos generativos en visión por computador**: modelos que aprenden la **distribución de probabilidad** de los datos para **muestrear** instancias nuevas y realistas (en vez de solo clasificar). La clase recorre las cuatro familias fundamentales —**Autoencoders / VAE**, **GANs**, **modelos de difusión** y **Latent / Stable Diffusion**—, las compara con el **trilema generativo** (calidad / velocidad / cobertura) y la métrica **FID**, y cierra con sus usos en la industria (data augmentation con datos sintéticos, detección de anomalías, restauración, generación por texto). El hilo culminante es **Stable Diffusion** = VAE (compresión) + U-Net con cross-attention (difusión en el latente) + text encoder CLIP, hoy el generador de imágenes más usado del mundo.

La clase integra el curso: autoencoders y CNN, la [U-Net](/papers/unet-ronneberger-2015) (de la segmentación médica) como denoiser, [CLIP (Clase 23)](/clases/clase-23) como codificador de texto, y el [mecanismo de atención (Clase 15)](/clases/clase-15) en el cross-attention de Stable Diffusion.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las 47 diapositivas: motivación, Autoencoders/VAE, GANs, difusión, Latent/Stable Diffusion, FID, industria" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: ELBO y reparameterization, minimax y Jensen-Shannon, DDPM (ε-prediction), score matching, cross-attention y guidance, FID" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="VAE, GAN y difusión DDPM desde cero en triple framework (PyTorch, TensorFlow, JAX)" icon="code" >}}
  {{< card link="/clases/clase-30" title="Clase siguiente: Modelos con memoria externa" subtitle="Memory Networks, NTM, memoria explícita" icon="arrow-right" >}}
  {{< card link="/clases/clase-28" title="Clase anterior: Aprendizaje Autosupervisado" subtitle="SSL, contrastivo, MAE, UDA" icon="arrow-left" >}}
  {{< card link="/clases/clase-23" title="Base: VQA e Image Captioning (CLIP)" subtitle="CLIP, el text encoder de Stable Diffusion" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/modelos-generativos" title="Modelos Generativos" subtitle="Las familias (VAE/GAN/difusión), el trilema, evaluación (FID), aplicaciones" icon="book-open" >}}
  {{< card link="/fundamentos/modelos-de-difusion" title="Modelos de Difusión" subtitle="Forward/reverse, ε-prediction, U-Net, score matching, guidance, latent diffusion" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-contrastivo" title="Aprendizaje Contrastivo" subtitle="CLIP, el codificador de texto que condiciona Stable Diffusion" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atención" subtitle="El cross-attention que condiciona la difusión por texto" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/vae-kingma-2013" title="VAE (2013)" subtitle="Kingma & Welling — autoencoder variacional, ELBO, reparameterization" icon="document-text" >}}
  {{< card link="/papers/gan-goodfellow-2014" title="GAN (2014)" subtitle="Goodfellow et al. — el juego adversarial generador vs discriminador" icon="document-text" >}}
  {{< card link="/papers/ddpm-ho-2020" title="DDPM (2020)" subtitle="Ho et al. — modelos de difusión que predicen el ruido" icon="document-text" >}}
  {{< card link="/papers/latent-diffusion-rombach-2022" title="Latent / Stable Diffusion (2022)" subtitle="Rombach et al. — difusión en el latente, el modelo más usado" icon="document-text" >}}
  {{< card link="/papers/unet-ronneberger-2015" title="U-Net (2015)" subtitle="Ronneberger et al. — el denoiser de la difusión (origen médico)" icon="document-text" >}}
  {{< card link="/papers/fid-heusel-2017" title="FID / TTUR (2017)" subtitle="Heusel et al. — la métrica estándar de calidad generativa" icon="document-text" >}}
  {{< card link="/papers/diffusion-gan-xiao-2021" title="Diffusion GANs / trilema (2021)" subtitle="Xiao et al. — el trilema generativo (citado en la clase)" icon="document-text" >}}
  {{< card link="/papers/datasetgan-zhang-2021" title="DatasetGAN (2021)" subtitle="Zhang et al. — fábrica de datos etiquetados (citado en la clase)" icon="document-text" >}}
{{< /cards >}}

## Papers canónicos (complementarios)

{{< cards >}}
  {{< card link="/papers/dcgan-radford-2015" title="DCGAN (2015)" subtitle="Radford et al. — GANs convolucionales que funcionan en imágenes" icon="document-text" >}}
  {{< card link="/papers/stylegan-karras-2019" title="StyleGAN (2019)" subtitle="Karras et al. — control de estilo, caras hiperrealistas, FFHQ" icon="document-text" >}}
  {{< card link="/papers/vq-vae-oord-2017" title="VQ-VAE (2017)" subtitle="van den Oord et al. — latentes discretos, base de DALL-E/VQ-GAN" icon="document-text" >}}
  {{< card link="/papers/score-based-song-2019" title="Score-Based / NCSN (2019)" subtitle="Song & Ermon — la otra estirpe de la difusión (score matching)" icon="document-text" >}}
  {{< card link="/papers/classifier-free-guidance-ho-2022" title="Classifier-Free Guidance (2022)" subtitle="Ho & Salimans — la guidance scale de todo text-to-image" icon="document-text" >}}
  {{< card link="/papers/clip-radford-2021" title="CLIP (2021)" subtitle="Radford et al. — el text encoder de Stable Diffusion" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/vision" title="Dominio: Visión" subtitle="Línea de tiempo: la era generativa, de las GAN a la difusión" icon="globe-alt" >}}
{{< /cards >}}
