---
title: "Practica desde 0 - Aprendizaje Autosupervisado"
weight: 30
sidebar:
  open: true
---

La clase 28 cubre el **aprendizaje autosupervisado (SSL)**: aprender representaciones sin etiquetas, generando el objetivo a predecir automáticamente desde el propio dato. Esta práctica implementa los **tres mecanismos centrales** de la clase en **mínima escala** y sin librerías especializadas, para entender por dentro qué los distingue: la pérdida **contrastiva** (SimCLR/NT-Xent), el **masked autoencoder** (MAE) y el **consistency training** semi-supervisado (UDA, el método del [Laboratorio 28](/laboratorios/lab-28)). Cada camino replica el mismo núcleo en **triple framework** (PyTorch, TensorFlow y JAX) para ver cómo cada uno expresa las mismas ideas.

## Caminos

{{< cards >}}
  {{< card link="01-simclr-desde-cero" title="01 - SimCLR / NT-Xent desde cero" subtitle="La pérdida contrastiva (positivos del mismo, negativos del batch) en PyTorch, TensorFlow y JAX" icon="code" >}}
  {{< card link="02-masked-autoencoder-desde-cero" title="02 - Masked Autoencoder desde cero" subtitle="Patchify, máscara 75%, encoder asimétrico y reconstrucción en triple framework" icon="code" >}}
  {{< card link="03-uda-consistency-desde-cero" title="03 - UDA / Consistency Training desde cero" subtitle="Pérdida supervisada + KL de consistencia con confidence masking — el método del lab" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 20 - BERT/GPT](../../clase-20): el masked language modeling de BERT es el primo lingüístico del MAE.
- [Clase 23 - VQA/CLIP](../../clase-23): el aprendizaje contrastivo imagen-texto (CLIP).
- Python intermedio y NumPy; PyTorch básico. Útil: nociones de TensorFlow/Keras y JAX.
- GPU **no necesaria**: los toy datasets (embeddings sintéticos, MNIST, two moons) corren en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - SimCLR / NT-Xent | PyTorch 2.x | TensorFlow 2.x, JAX |
| 02 - Masked Autoencoder | PyTorch 2.x | TensorFlow 2.x, JAX |
| 03 - UDA / Consistency | PyTorch 2.x | TensorFlow 2.x, JAX + Flax/optax |

## El hilo conductor

Los tres caminos son tres formas distintas de **fabricar la supervisión gratis**:

1. **Contrastivo (SimCLR)**: la señal es la *identidad de instancia* — dos vistas aumentadas de la misma imagen deben acercarse; las demás, alejarse.
2. **Generativo (MAE)**: la señal es la *reconstrucción* — esconder el 75% de la imagen y reconstruirlo obliga a entenderla.
3. **Consistency (UDA)**: la señal es la *invarianza* — el modelo debe predecir igual para un dato y su versión aumentada, usando datos sin etiquetar para regularizar.

---

**Ver tambien:** [Clase 28 - Teoria](../teoria) · [Clase 28 - Profundizacion](../profundizacion) · Fundamentos: [Aprendizaje Autosupervisado](/fundamentos/aprendizaje-autosupervisado) · [Aprendizaje Contrastivo](/fundamentos/aprendizaje-contrastivo) · [Aprendizaje Semi-Supervisado](/fundamentos/aprendizaje-semi-supervisado).
