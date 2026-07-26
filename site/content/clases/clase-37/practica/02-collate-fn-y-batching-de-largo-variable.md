---
title: "collate_fn y batching de largo variable desde cero"
weight: 2
math: true
---

El [camino 01](/clases/clase-37/practica/01-del-wav-al-tensor-mel-snr-y-specaugment) convirtió el audio en un tensor y lo aumentó. Falta el último eslabón que la Clase 37 subraya: **armar un batch**. A diferencia de las imágenes (todas del mismo tamaño), el audio **no viene de largo fijo** —cada clip dura distinto—, y PyTorch **no puede** apilar tensores de formas distintas. Este capítulo muestra por qué el batch directo **falla** y cómo la **`collate_fn`** lo arregla con *padding*, en triple framework.

> **Lecturas de apoyo:** el fundamento [Representación de audio](/fundamentos/representacion-de-audio) (sección "detalles que conviene cuidar").

---

## 1. El problema: el batch directo falla

Simulemos un dataset de audios de duración variable —lo normal en el mundo real.

```python
import numpy as np
rng = np.random.default_rng(0)

# 4 "clips" de largo distinto (número de muestras)
clips = [rng.standard_normal(n).astype(np.float32) for n in [16000, 24000, 12000, 20000]]
labels = np.array([0, 1, 0, 1])
print("largos:", [len(c) for c in clips])       # [16000, 24000, 12000, 20000]
```

Intentar apilarlos directamente en un tensor falla, porque las formas no coinciden:

```python
import torch
try:
    batch = torch.stack([torch.tensor(c) for c in clips])   # ✗ formas distintas
except RuntimeError as e:
    print("falla:", str(e)[:60])                # "stack expects each tensor to be equal size..."
```

{{< concept-alert type="advertencia" >}}
Como dice la clase: *"PyTorch no arma el batch"* cuando los ejemplos tienen largo variable. El `DataLoader` intenta apilar los tensores del batch en uno solo y **explota** si no tienen la misma forma. La solución es una **`collate_fn`**: una función que le enseña al `DataLoader` **cómo** combinar una lista de ejemplos en un batch.
{{< /concept-alert >}}

---

## 2. La solución: padding al más largo del batch

La `collate_fn` más común **rellena** (padding con ceros) cada clip hasta el largo del más largo del batch, y devuelve además las **longitudes reales** (para que el modelo pueda ignorar el relleno, p. ej. con `pack_padded_sequence` o una máscara).

```python
def collate_pad(batch):
    """batch: lista de (clip, label). Devuelve tensores apilados con padding."""
    clips, labels = zip(*batch)
    lengths = torch.tensor([len(c) for c in clips])
    max_len = lengths.max().item()
    padded = torch.zeros(len(clips), max_len)             # [B, max_len] de ceros
    for i, c in enumerate(clips):
        padded[i, :len(c)] = torch.tensor(c)             # copia el clip; el resto queda en 0
    return padded, torch.tensor(labels), lengths

# uso con un DataLoader real
dataset = list(zip(clips, labels))
loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_pad)
padded, y, lengths = next(iter(loader))
print(padded.shape, "longitudes reales:", lengths.tolist())   # torch.Size([4, 24000]) [16000, 24000, 12000, 20000]
```

Ahora sí hay un batch rectangular `[4, 24000]` que el modelo puede procesar, y `lengths` recuerda dónde termina cada clip real y empieza el relleno.

{{< concept-alert type="recordar" >}}
Guardar las **longitudes reales** es clave: sin ellas, el modelo trataría los ceros de relleno como audio de verdad. Con ellas, se puede **enmascarar** el padding (en una RNN, `pack_padded_sequence`; en un Transformer, una *attention mask*) para que no contamine el cálculo.
{{< /concept-alert >}}

En la práctica, la `collate_fn` también es un buen lugar para las **augmentations baratas** (recorte aleatorio, ganancia, ruido por SNR), que corren en CPU mientras el `DataLoader` prepara el batch —tal como indica la tabla de la clase.

---

## 3. La misma idea en triple framework

El padding de secuencias de largo variable es una operación estándar en los tres frameworks.

### PyTorch

```python
from torch.nn.utils.rnn import pad_sequence

def collate_torch(batch):
    clips, labels = zip(*batch)
    seqs = [torch.tensor(c) for c in clips]
    padded = pad_sequence(seqs, batch_first=True)        # rellena al más largo
    lengths = torch.tensor([len(c) for c in clips])
    return padded, torch.tensor(labels), lengths
```

### TensorFlow

```python
import tensorflow as tf

# tf.data agrupa y rellena con padded_batch
def make_tf_dataset(clips, labels):
    ds = tf.data.Dataset.from_generator(
        lambda: ((c, l) for c, l in zip(clips, labels)),
        output_signature=(tf.TensorSpec([None], tf.float32), tf.TensorSpec([], tf.int64)))
    return ds.padded_batch(4, padded_shapes=([None], []))   # <- padding automático al más largo
```

### JAX

```python
import jax.numpy as jnp

def collate_jax(clips, labels):
    max_len = max(len(c) for c in clips)
    padded = jnp.stack([jnp.pad(jnp.asarray(c), (0, max_len - len(c))) for c in clips])
    lengths = jnp.array([len(c) for c in clips])
    return padded, jnp.asarray(labels), lengths
```

En los tres, la receta es la misma: **encontrar el largo máximo del batch y rellenar con ceros hasta ahí**, recordando las longitudes reales. TensorFlow lo tiene integrado (`padded_batch`); en PyTorch y JAX se hace explícito con la `collate_fn`.

---

## 4. Qué nos llevamos

- El audio **no viene de largo fijo**, así que apilar un batch directamente **falla**.
- La **`collate_fn`** le enseña al `DataLoader` a combinar ejemplos: rellena (padding) al más largo del batch y guarda las **longitudes reales**.
- Las longitudes permiten **enmascarar** el relleno para que no contamine el modelo (RNN packing, attention mask).
- La `collate_fn` es también el lugar de las augmentations baratas en CPU (recorte, ganancia, ruido por SNR).
- Es una operación estándar en los tres frameworks (`pad_sequence`, `padded_batch`, `jnp.pad`).

Con esto, el ciclo de vida del dato de audio queda completo: del **WAV** ([camino 01](/clases/clase-37/practica/01-del-wav-al-tensor-mel-snr-y-specaugment)) al **tensor aumentado** y al **batch** listo para entrenar —el pipeline que el [laboratorio](/laboratorios/lab-37) lleva a un clasificador de géneros real sobre GTZAN.

---

**Ver también:** [Clase 37 - Teoría](/clases/clase-37/teoria) · [Clase 37 - Profundización](/clases/clase-37/profundizacion) · [Camino 01: Del WAV al tensor](/clases/clase-37/practica/01-del-wav-al-tensor-mel-snr-y-specaugment) · [Laboratorio](/laboratorios/lab-37).
