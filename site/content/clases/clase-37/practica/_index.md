---
title: "Practica desde 0 - Datasets y Herramientas para Audio"
weight: 30
sidebar:
  open: true
---

La Clase 37 es sobre el **ciclo de vida del dato de audio**: del archivo en disco al tensor, la augmentation, y el batch que entra al modelo. Esta práctica construye ese pipeline **desde cero**, sin `torchaudio` ni `librosa` (solo NumPy y las operaciones tensoriales), para que cada paso quede claro. El primer camino va **del WAV al tensor**: espectrograma Mel, sumar ruido calibrado por SNR, y SpecAugment. El segundo resuelve el problema que la clase subraya: **armar un batch con audios de largo variable** con una `collate_fn`. Cada uno en **triple framework** (PyTorch, TensorFlow, JAX).

## Caminos

{{< cards >}}
  {{< card link="01-del-wav-al-tensor-mel-snr-y-specaugment" title="01 - Del WAV al tensor: Mel, SNR y SpecAugment" subtitle="Espectrograma Mel, sumar ruido por SNR y SpecAugment desde cero (las 3 representaciones)" icon="code" >}}
  {{< card link="02-collate-fn-y-batching-de-largo-variable" title="02 - collate_fn y batching de largo variable" subtitle="Por qué falla el batch directo y cómo el padding lo arregla, en PyTorch, TensorFlow y JAX" icon="code" >}}
{{< /cards >}}

## Requisitos previos

- [Clase 37 - Teoría](/clases/clase-37/teoria) y [Profundización](/clases/clase-37/profundizacion).
- [Clase 35](/clases/clase-35) (Fourier, STFT, MFCC) — la teoría de señales que aquí se usa.
- Python intermedio y NumPy; PyTorch básico. Útil: TensorFlow/Keras y JAX.
- GPU **no necesaria**: todo corre en CPU en segundos.

## Tecnologias usadas

| Camino | Stack principal | Frameworks secundarios |
|--------|------------------|------------------------|
| 01 - WAV al tensor | NumPy | PyTorch / TensorFlow / JAX |
| 02 - collate_fn | PyTorch 2.x | TensorFlow 2.x, JAX |

## El hilo conductor

1. **Del WAV al tensor**: partimos de una señal y calculamos su **espectrograma Mel** (la representación que entra al modelo), luego la aumentamos: **sumar ruido** calibrando la relación señal-ruido (SNR) y **SpecAugment** (tapar bandas y tramos del espectrograma). Verificamos las dimensiones del tensor con la fórmula de la clase.
2. **collate_fn y batching**: el audio no viene de tamaño fijo, así que armar un batch directo **falla**. Implementamos la `collate_fn` que rellena (padding) al largo del más largo del batch —la pieza que hace posible entrenar con audio real.

---

**Ver tambien:** [Clase 37 - Teoria](/clases/clase-37/teoria) · [Clase 37 - Profundizacion](/clases/clase-37/profundizacion) · Fundamentos: [Representación de audio](/fundamentos/representacion-de-audio) · [Data augmentation de audio](/fundamentos/data-augmentation-de-audio).
