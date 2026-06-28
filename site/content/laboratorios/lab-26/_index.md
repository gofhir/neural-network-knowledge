---
title: "Lab 26 - Meta-aprendizaje: MAML y Prototypical Networks"
weight: 260
sidebar:
  open: true
---

**Profesor:** Pablo Messina
**Fecha:** Junio 2026
**Notebooks origen:** `clase_26/material/Laboratorio/Practico_Metaaprendizaje_(Parte_1).ipynb` (MAML) y `Practico_Metaaprendizaje_(Parte_2).ipynb` (Prototypical)
**Notebooks ejecutados:** Parte 1 — [lab26-parte1.ipynb](/notebooks/lab26-parte1.ipynb) · [HTML](/notebooks-html/lab26-parte1.html) · Parte 2 — [lab26-parte2.ipynb](/notebooks/lab26-parte2.ipynb) · [HTML](/notebooks-html/lab26-parte2.html)

## Encuadre

La contraparte práctica de la [clase 26](/clases/clase-26): implementar **dos algoritmos de meta-aprendizaje de filosofías opuestas** sobre los mismos benchmarks few-shot, y comparar su comportamiento.

- **Parte 1 — MAML** (Model-Agnostic Meta-Learning, *optimization-based*): aprende una **inicialización** de pesos tal que pocos pasos de gradiente adaptan a una tarea nueva. Optimización binivel (bucle interno de adaptación + bucle externo de meta-update).
- **Parte 2 — Prototypical Networks** (*metric-based*): aprende un **espacio de embeddings** donde cada clase es un **prototipo** (centroide) y clasificar es medir distancia. Sin adaptación por gradiente en test.

Ambas partes usan la librería [learn2learn](/papers/learn2learn-arnold-2020) y los dos benchmarks canónicos del few-shot: **Omniglot** (caracteres manuscritos 1×28×28, fácil) y **Mini-ImageNet** (fotos RGB 84×84, difícil; su split 64/16/20 viene de [Ravi & Larochelle 2017](/papers/ravi-optimization-fewshot-2017)). El protocolo es **N-way K-shot**: clasificar N clases con K ejemplos de support por clase.

| Pieza | Implementación en el lab |
|---|---|
| Muestreo de tareas | `l2l.vision.benchmarks.get_tasksets(...)` — episodios N-way K-shot, splits con clases disjuntas |
| MAML | `l2l.algorithms.MAML` con `clone()` / `adapt()`, FOMAML (`first_order=True`) |
| Prototypical | encoder + prototipos (promedio de support) + distancia euclidiana |
| Modelos | `OmniglotFC`, `OmniglotCNN`, `MiniImagenetCNN`, `ResNet12` |
| Evaluación | Meta Test Accuracy sobre 2000 tareas de clases no vistas |

## Resultados consolidados

### MAML vs Prototypical (mejor configuración de cada uno)

| Problema | MAML (mejor) | Prototypical (mejor) |
|---|---|---|
| Omniglot 4-way 1-shot | 0.877 (CNN, 400 iters, 5 hiperparámetros afinados) | **0.934** (CNN, 80 épocas) |
| Mini-ImageNet 4-way 1-shot | 0.324 | **0.377** |
| Mini-ImageNet 4-way 5-shot | 0.491 | **0.632** |

→ Prototypical **iguala o supera** a MAML en todo el espacio, con un método estructuralmente más simple (sin segundo orden, sin bucle interno, inferencia más barata).

### Las cuatro lecciones del lab

1. **El cuello de botella depende del dato.** En Omniglot (simple) manda la **optimización** (más iteraciones de meta-entrenamiento); en Mini-ImageNet (complejo) manda la **información** (más SHOTS). No hay receta universal de hiperparámetros.
2. **Prototypical es robusto al número de clases** en Omniglot (accuracy crudo ~constante de 2 a 8 ways) mientras MAML se desploma (0.932 → 0.657) — pero esa robustez **depende de la calidad de los embeddings** (desaparece en Mini-ImageNet 1-shot).
3. **El overfitting depende del dataset**, no del método: aparece en Mini-ImageNet (memorización de las 64 clases de train) y es nulo en Omniglot.
4. **Más capacidad no es más desempeño:** ni la CNN sin iteraciones suficientes (MAML), ni ResNet12 sin augmentation (Prototypical) ayudaron — la capacidad necesita los recursos que la activan.

## Bloques del lab

{{< cards >}}
  {{< card link="maml-fundamentos" title="MAML: optimización binivel en código" subtitle="Muestreo episódico, fast_adapt_MAML, run_MAML, bucle interno/externo, clone(), FOMAML" icon="academic-cap" >}}
  {{< card link="experimentos-maml" title="Experimentos con MAML (Act. 1-3)" subtitle="Ablation de hiperparámetros en Omniglot, ejes WAYS/SHOTS, información vs optimización en Mini-ImageNet" icon="beaker" >}}
  {{< card link="prototypical-fundamentos" title="Prototypical Networks: clasificar por distancia" subtitle="pairwise_distances_logits, prototipos, run_Protonet, encoder independiente de WAYS, contraste con MAML" icon="academic-cap" >}}
  {{< card link="experimentos-prototypical" title="Experimentos con Prototypical (Act. 4-6)" subtitle="Encoder CNN, robustez al WAYS, límites en Mini-ImageNet, ResNet12 que no ayuda" icon="beaker" >}}
  {{< card link="comparacion-y-teoria" title="Comparación MAML vs Prototypical y teoría" subtitle="Síntesis de los dos paradigmas + Actividad 7 (MAML vs fine-tuning, Siamese→Matching→Prototypical) + conexión FHIR" icon="document-text" >}}
{{< /cards >}}

## Papers relacionados

{{< cards >}}
  {{< card link="/papers/maml-finn-2017" title="MAML (2017)" subtitle="Finn et al. — la inicialización adaptable, el método estrella optimization-based de la Parte 1" icon="document-text" >}}
  {{< card link="/papers/prototypical-networks-snell-2017" title="Prototypical Networks (2017)" subtitle="Snell et al. — prototipos como centroides, el método de la Parte 2" icon="document-text" >}}
  {{< card link="/papers/learn2learn-arnold-2020" title="learn2learn (2020)" subtitle="Arnold et al. — la librería sobre la que se construye todo el lab" icon="document-text" >}}
  {{< card link="/papers/ravi-optimization-fewshot-2017" title="Optimization as a Model for Few-Shot Learning (2017)" subtitle="Ravi & Larochelle — el split de Mini-ImageNet y el predecesor conceptual de MAML" icon="document-text" >}}
  {{< card link="/papers/matching-networks-vinyals-2016" title="Matching Networks (2016)" subtitle="Vinyals et al. — entrenamiento episódico y attention, eslabón Siamese→Prototypical" icon="document-text" >}}
  {{< card link="/papers/siamese-networks-koch-2015" title="Siamese Networks (2015)" subtitle="Koch et al. — torres gemelas, el origen del deep metric learning" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/meta-aprendizaje" title="Meta-aprendizaje" subtitle="Aprender a aprender: las tres familias de métodos" icon="book-open" >}}
  {{< card link="/fundamentos/few-shot-learning" title="Few-shot Learning" subtitle="N-way K-shot, support/query, benchmarks" icon="book-open" >}}
  {{< card link="/fundamentos/optimizacion-binivel" title="Optimización bi-nivel" subtitle="El marco matemático de MAML: inner/outer loop, meta-gradiente, FOMAML" icon="book-open" >}}
  {{< card link="/fundamentos/metric-learning" title="Metric Learning" subtitle="Aprender un espacio donde la distancia codifica similitud (Prototypical)" icon="book-open" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-26" title="Clase 26 - Teoría" subtitle="Meta-aprendizaje: MAML, MANN, métodos no-paramétricos, aplicaciones médicas" icon="academic-cap" >}}
  {{< card link="/clases/clase-26/profundizacion" title="Profundización" subtitle="Math: optimización bi-nivel, meta-gradiente de 2º orden, FOMAML/Reptile, Bregman" icon="beaker" >}}
  {{< card link="/dominios/vision" title="Dominio: Visión" subtitle="Era Meta-aprendizaje y Few-shot (2015-2025)" icon="globe-alt" >}}
  {{< card link="/laboratorios/lab-25" title="Lab 25 - Recomendación multimodal (anterior)" subtitle="Content-based multimodal con imágenes y texto" icon="academic-cap" >}}
  {{< card link="/laboratorios/lab-27" title="Lab 27 - Redes Neuronales de Grafos (siguiente)" subtitle="GNN con PyTorch Geometric: clasificación de nodos y grafos" icon="arrow-right" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las dos partes del práctico (MAML y Prototypical Networks) con 5 páginas temáticas y las 7 actividades resueltas. Incluye los resultados medidos (MAML Omniglot 0.699→0.877 sin overfitting; Prototypical Omniglot 0.934; ambos métodos en Mini-ImageNet con SHOTS como palanca dominante), el diseño factorial 2×2 de WAYS/SHOTS, y el hallazgo de la robustez al número de clases de Prototypical condicionada a la calidad de los embeddings. Notebooks ejecutados en Colab.
