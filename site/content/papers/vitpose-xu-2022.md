---
title: "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation"
weight: 82
math: true
---

{{< paper-card
    title="ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation"
    authors="Xu, Zhang, Zhang, Tao"
    year="2022"
    venue="NeurIPS 2022"
    pdf="/papers/vitpose-xu-2022.pdf"
    arxiv="2204.12484" >}}
Demuestra que un **Vision Transformer plain** como backbone + un decoder lightweight (2 deconvs o incluso 1 bilinear + 1 conv) basta para alcanzar **SOTA en COCO Keypoint** (80.9 AP test-dev con ViTPose-G de 1B parámetros). Establece la era del *backbone transformer puro* en pose estimation, rompiendo la hipótesis de HRNet de que se necesitan feature maps multi-resolución.
{{< /paper-card >}}

---

## Contexto

En 2022, HRNet y su familia multi-resolución dominaban COCO Keypoint con AP ~76-77. HRFormer (NeurIPS 2021) trajo transformers pero conservaba la complejidad multi-resolución de HRNet. TokenPose, TransPose y PRTR usaban CNN como backbone + transformer como decoder/refiner. **Ninguno** exploraba "ViT puro como backbone". La pregunta de ViTPose: ¿qué tan bien funciona un ViT plain, sin trucos arquitecturales?

## Ideas principales

### Arquitectura ViTPose

```
Image → Patch Embed (16×16, stride 16) → L × Transformer Block → F_out → Lightweight Decoder → Heatmaps
```

**Entrada**: persona crop a 256×192 (top-down, requiere person detector externo).

**Encoder estándar ViT**:

$$
F'_{i+1} = F_i + \text{MHSA}(\text{LN}(F_i)) \qquad F_{i+1} = F'_{i+1} + \text{FFN}(\text{LN}(F'_{i+1}))
$$

**Dos decoders posibles**:

*Classic* (igual a SimpleBaseline):

$$
K = \text{Conv}_{1 \times 1}(\text{Deconv}_2(\text{Deconv}_2(F_\text{out})))
$$

*Simple*:

$$
K = \text{Conv}_{3 \times 3}(\text{Bilinear}_{4 \times}(\text{ReLU}(F_\text{out})))
$$

Resultado sorprendente: en ViT-B/L/H, ambos decoders dan AP ~idéntico — el ViT pretrained *carga* todo el peso representacional. Pero ResNet pierde 18 AP con el simple decoder.

### Pretraining flexibility

ViTPose pre-entrena con **MAE (Masked Autoencoder)**: enmascara 75% de patches y reconstruye. **Hallazgo clave**: pre-entrenar MAE sobre las propias imágenes de pose (COCO + AI Challenger, 500K) da 75.8 AP — *idéntico* a usar ImageNet-1K (1M). El cuello de botella no es la cantidad sino la auto-supervisión sobre el dominio.

### Knowledge token distillation

Innovación sobre output distillation tradicional:

$$
t^* = \arg\min_t \text{MSE}(T(\{t; X\}), K_{gt})
$$

donde $T$ es el teacher congelado (ViTPose-L) y $X$ son los tokens de imagen. Una vez aprendido $t^*$, se concatena con los tokens del student durante finetuning. Token + output distillation transfiere ViTPose-L → ViTPose-B con +0.8 AP.

### Cuatro propiedades

1. **Simplicidad**: ViT plain + decoder mínimo.
2. **Escalabilidad**: 86M → 1B parámetros cubriendo Pareto throughput-vs-AP.
3. **Flexibilidad**: tolerante a pretraining, resoluciones, atención (full/window/shift/pool), finetuning parcial, multi-dataset.
4. **Transferibilidad**: knowledge token + output distillation.

## Resultados experimentales

### COCO Keypoint val (Tabla 9)

| Modelo | Backbone | Params | Speed (fps) | AP val |
|---|---|---|---|---|
| SimpleBaseline | ResNet-152 | 60M | 829 | 73.5 |
| HRNet-W48 | HRNet | 64M | 309 | 76.3 |
| HRFormer-B | HRFormer | 43M | 158 | 75.6 |
| **ViTPose-B** | ViT-B | 86M | **944** | **75.8** |
| ViTPose-L | ViT-L | 307M | 411 | 78.3 |
| ViTPose-H | ViT-H | 632M | 241 | 79.1 |
| **ViTPose-G** | ViTAE-G | 1024M | — | **80.9** |

ViTPose-B es **3× más rápido** que HRNet-W48 con AP comparable. ViTPose-G destrona el SOTA.

### Decoder simple vs classic (Tabla 2)

| Backbone | Decoder | AP val |
|---|---|---|
| ResNet-152 | Classic | 73.5 |
| ResNet-152 | Simple | 55.3 (**-18**) |
| **ViTPose-B** | Classic | 75.8 |
| **ViTPose-B** | Simple | 75.5 (-0.3) |

ResNet colapsa con decoder simple. ViTPose es robusto — la representación del ViT es lo suficientemente rica.

### Finetuning parcial (Tabla 6)

| MHSA | FFN | AP val |
|---|---|---|
| ✓ | ✓ | 75.8 |
| ✓ | frozen | 72.8 (**-3.0**) |
| frozen | ✓ | 75.1 (-0.7) |

**FFN es más específico de tarea que MHSA**. Frenar FFN cuesta 3 AP; frenar MHSA solo 0.7.

## Limitaciones reconocidas

1. **Top-down only**: requiere detector externo, hereda problemas con bboxes intersectados.
2. **Sin decoders elaborados**: futuras mejoras con FPN, dilated attention, skip connections quedan abiertas.
3. **Solo human pose 2D**: animal pose / facial keypoints / 3D quedan como future work.
4. **Costo de pretraining**: MAE en 500K × cientos de epochs, ViTPose-G en 8 A100.
5. **Sin estudio explícito de oclusiones**.

## Por qué importa hoy

- **Nuevo SOTA en COCO Keypoint** (80.9 AP test-dev) — destrona HRNet/HRFormer.
- Confirma **"scale > arquitectura"** también en pose.
- **mmpose** lo integra como baseline de referencia.
- Inspira ViTPose+ (multi-task), DWPose (distillation), EdgePose (mobile via knowledge token).
- Concretiza la afirmación del profesor: *"el SOTA en object detection se hace con Vision Transformers, no CNNs"*.

## Conexión con la clase 17

ViTPose es el **disclaimer final** que el profesor introduce en los slides 51-52. Cierra el arco **CNN → Transformer** que vimos en la clase de Transformers y demuestra que los conceptos de pose (top-down, keypoints, heatmaps) **siguen siendo válidos** — solo cambia el backbone.

## Notas y enlaces

- Código: https://github.com/ViTAE-Transformer/ViTPose
- mmpose: https://mmpose.readthedocs.io/
- Análisis interno con código PyTorch/TF/JAX en el repositorio del curso.
