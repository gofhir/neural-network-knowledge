---
title: "Profundizacion - Vicinal Risk Minimization, Mixup y Transfer Learning"
weight: 20
math: true
---

> Este documento profundiza en los fundamentos teoricos detras de la Clase 12.
> Cubre el marco formal de data augmentation via Vicinal Risk Minimization,
> la derivacion y analisis de Mixup, el marco experimental de Yosinski (2014)
> para transferibilidad layer-by-layer, las matematicas de finetuning con
> discriminative learning rates, y la transicion conceptual a foundation models.

---

# Parte I: Marco Formal de Data Augmentation

---

## 1. Empirical Risk Minimization (ERM) vs Vicinal Risk Minimization (VRM)

### 1.1 ERM

En supervised learning buscamos $f \in \mathcal{F}$ que minimiza el **riesgo esperado**:

$$R(f) = \int \ell(f(x), y) \, dP(x, y)$$

Como $P(x, y)$ es desconocida, usamos la **distribucion empirica**:

$$P_\delta(x, y) = \frac{1}{n} \sum_{i=1}^{n} \delta(x = x_i, y = y_i)$$

y aproximamos con el **riesgo empirico**:

$$R_\delta(f) = \frac{1}{n} \sum_{i=1}^{n} \ell(f(x_i), y_i)$$

Minimizar $R_\delta$ es **ERM**. Para redes con muchos parametros, ERM tiene una patologia: puede **memorizar** el training set (Zhang et al. 2017). La red alcanza training error cero incluso con labels aleatorios.

### 1.2 VRM

Chapelle et al. (2000) propusieron aproximar $P$ con una **distribucion de vecindad**:

$$P_\nu(\tilde{x}, \tilde{y}) = \frac{1}{n} \sum_{i=1}^{n} \nu(\tilde{x}, \tilde{y} \mid x_i, y_i)$$

donde $\nu$ define un "entorno" alrededor de cada ejemplo. El riesgo vicinal empirico:

$$R_\nu(f) = \frac{1}{m} \sum_{i=1}^{m} \ell(f(\tilde{x}_i), \tilde{y}_i)$$

con $(\tilde{x}_i, \tilde{y}_i) \sim P_\nu$.

### 1.3 Ejemplos de distribuciones vecinales

| $\nu$ | Resultado |
|---|---|
| Gaussian noise: $\nu = \mathcal{N}(x - x_i, \sigma^2) \delta(\tilde{y} = y_i)$ | Augmentation con ruido aditivo |
| Geometric: flips, crops, rotaciones | Augmentation clasica de imagenes |
| **Mixup**: convex combinations de pares | Augmentation con labels interpolados |

{{< concept-alert type="clave" >}}
La distincion clave entre ERM y VRM es que **ERM asume que solo los ejemplos del training set tienen probabilidad no-cero**. VRM extiende la probabilidad a un **entorno** alrededor de cada ejemplo -- este entorno captura que variaciones preservan la clase.
{{< /concept-alert >}}

---

# Parte II: Mixup en Detalle

---

## 2. Formulacion de Mixup

Mixup (Zhang et al. 2017) define la siguiente distribucion vecinal:

$$\nu_{\text{mixup}}(\tilde{x}, \tilde{y} \mid x_i, y_i) = \frac{1}{n} \sum_{j=1}^{n} \mathbb{E}_\lambda \left[ \delta(\tilde{x} = \lambda x_i + (1-\lambda) x_j, \, \tilde{y} = \lambda y_i + (1-\lambda) y_j) \right]$$

donde $\lambda \sim \text{Beta}(\alpha, \alpha)$.

### 2.1 Propiedades de $\lambda \sim \text{Beta}(\alpha, \alpha)$

La distribucion Beta simetrica $\text{Beta}(\alpha, \alpha)$ tiene densidad:

$$p(\lambda; \alpha) = \frac{\lambda^{\alpha-1} (1-\lambda)^{\alpha-1}}{B(\alpha, \alpha)}, \quad \lambda \in [0, 1]$$

| $\alpha$ | Comportamiento |
|---|---|
| $\alpha \to 0$ | Concentra masa en $\lambda \in \{0, 1\}$ -- recupera ERM (no mezcla) |
| $\alpha = 0.2$ | Valor tipico -- mayoria de $\lambda$ cerca de 0 o 1, mezcla ocasional fuerte |
| $\alpha = 1$ | Uniforme en $[0, 1]$ |
| $\alpha \to \infty$ | Concentra masa en $\lambda = 0.5$ -- mezcla maxima |

{{< concept-alert type="recordar" >}}
**Rango recomendado**: $\alpha \in [0.1, 0.4]$ para imagenes naturales. Mayor $\alpha$ puede empeorar si el modelo no tiene capacidad suficiente para aprender las interpolaciones fuertes.
{{< /concept-alert >}}

### 2.2 Por que one-hot labels importan

Las labels deben ser **one-hot vectors** para que la interpolacion tenga sentido:

$$y_i = [0, 0, \ldots, 1, \ldots, 0], \quad y_j = [0, \ldots, 1, 0, \ldots, 0]$$

Entonces $\tilde{y} = \lambda y_i + (1-\lambda) y_j$ es una **distribucion de probabilidad valida** sobre clases, con masa $\lambda$ en clase $i$ y $(1-\lambda)$ en clase $j$.

El loss con softmax cross-entropy:

$$L = - \tilde{y}^T \log \hat{p} = -\lambda \log \hat{p}_i - (1-\lambda) \log \hat{p}_j$$

Equivalente a **loss ponderado**: $\lambda \cdot L_i + (1-\lambda) \cdot L_j$ con labels duras $y_i$ y $y_j$.

### 2.3 Implementacion eficiente

En lugar de dos data loaders, se usa una sola minibatch mezclada consigo misma permutada:

```python
def mixup_batch(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mixed, y_a, y_b, lam

# En el training loop
x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=0.2)
logits = model(x_mixed)
loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
```

La formulacion con loss ponderado evita expandir las labels a one-hot.

---

## 3. Por que Mixup Funciona: Analisis

### 3.1 Regularizacion implicita hacia linealidad

Sea $f$ el modelo. Mixup penaliza desviaciones de linealidad entre pares:

$$f(\lambda x_i + (1-\lambda) x_j) \approx \lambda f(x_i) + (1-\lambda) f(x_j)$$

Este es un **prior de Occam fuerte**: modelos lineales entre ejemplos de entrenamiento tienen menor complejidad efectiva. Reduce la capacidad de memorizar ruido.

### 3.2 Normas de gradientes menores

El paper muestra empiricamente (Figura 2b) que modelos entrenados con mixup tienen **normas de gradientes input-side menores** en la region entre ejemplos:

$$\| \nabla_x \hat{p}(x = \lambda x_i + (1-\lambda) x_j) \|$$

es significativamente menor con mixup que con ERM para $\lambda \in (0, 1)$. Esta suavidad implica robustez a:

- **Adversarial examples**: perturbaciones pequenas cambian las predicciones poco.
- **Distribution shift**: comportamiento mas estable fuera del training set.

### 3.3 Decision boundaries suaves

En el toy problem de la Figura 1b, mixup produce decision boundaries que **transicionan linealmente** entre clases, mientras que ERM genera boundaries abruptas que hacen "hard classification" cerca de los bordes entre clases.

---

## 4. Variantes de Mixup

### 4.1 CutMix (Yun et al. 2019)

En vez de interpolar pixel-wise, **cortar un parche rectangular** de una imagen y pegarlo sobre otra:

$$\tilde{x} = M \odot x_i + (1 - M) \odot x_j$$

donde $M \in \{0, 1\}^{H \times W}$ es una mascara rectangular binaria. La etiqueta se mezcla proporcionalmente:

$$\tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda = \frac{\text{area}(M)}{H \cdot W}$$

**Ventaja**: preserva estructura local, evita imagenes "fantasma" de mixup estandar.

### 4.2 Manifold Mixup (Verma et al. 2019)

Mezclar en **features intermedias** en vez de en pixels:

$$\tilde{h}^{(l)} = \lambda h^{(l)}(x_i) + (1-\lambda) h^{(l)}(x_j)$$

para una capa $l$ aleatoria en cada batch. Fuerza que las representaciones intermedias sean convexas entre ejemplos.

### 4.3 PuzzleMix (Kim et al. 2020)

Mixup guiado por **saliency**: mezcla regiones relevantes de cada imagen segun un detector de saliency. Mejor preservacion de features discriminativos.

---

# Parte III: Analisis de Yosinski (2014)

---

## 5. Formalizacion de Transferibilidad

### 5.1 Pregunta fundamental

Dados:

- Una red con $L$ capas entrenada en tarea $A$: $f_A = g_L \circ g_{L-1} \circ \cdots \circ g_1$.
- Una nueva tarea $B$.

**Cuan buenas son las primeras $n$ capas $g_1, \ldots, g_n$ como extractor de features para $B$?**

### 5.2 Experimentos de Yosinski

Para $n \in \{1, \ldots, 7\}$, construir 4 variantes:

1. **Selffer BnB**: copiar $g_1, \ldots, g_n$ de baseB, congelar, entrenar $g_{n+1}, \ldots, g_8$ en B.
2. **Selffer BnB+**: igual pero fine-tune todas las capas.
3. **Transfer AnB**: copiar $g_1, \ldots, g_n$ de baseA, congelar, entrenar $g_{n+1}, \ldots, g_8$ en B.
4. **Transfer AnB+**: igual pero fine-tune.

Comparar top-1 accuracy de cada variante con la de **baseB** (entrenada desde cero en B).

### 5.3 Metricas

- **Specificity degradation**: $\text{acc}(BnB) - \text{acc}(baseB)$ -- cuanto se pierde al congelar capas propias.
- **Transfer degradation**: $\text{acc}(AnB) - \text{acc}(baseB)$ -- cuanto se pierde al transferir de A.
- **Fine-tuning recovery**: $\text{acc}(AnB^+) - \text{acc}(AnB)$ -- cuanto recupera fine-tune.

### 5.4 Dos mecanismos distintos

Yosinski separa rigurosamente:

| Mecanismo | Causa | Region dominante |
|---|---|---|
| **Fragile co-adaptation** | Features de capas adyacentes aprenden a coordinar de manera compleja; congelar rompe la coordinacion | Capas 3-5 |
| **Representation specificity** | Features se especializan a la tarea source | Capas 6-7 |

Con fine-tuning, el primer mecanismo **se elimina** (las capas descongeladas recuperan co-adaptacion). El segundo **se mitiga pero persiste**.

---

## 6. Implicaciones Matematicas

### 6.1 Cuando fine-tune ayuda mas

Sea $L$ el loss post-transfer y $\Delta \theta$ el ajuste de pesos durante fine-tuning. El beneficio de fine-tuning se puede modelar como:

$$\mathbb{E}[\Delta L] \approx -\eta \cdot \| \nabla_\theta L \|^2$$

donde $\eta$ es el learning rate. La magnitud del beneficio depende de **cuanto hay que ajustar**:

- Capas 1-2: features universales, $\nabla_\theta L \approx 0$, fine-tune no ayuda mucho.
- Capas 3-5: fragile co-adaptation, $\nabla_\theta L$ mediana, fine-tune recupera co-adaptacion.
- Capas 6-7: especificidad, $\nabla_\theta L$ grande, fine-tune especializa.

### 6.2 Distancia entre tareas

Sea $d(A, B)$ una metrica de distancia entre tareas (ej. Kolmogorov-Smirnov sobre distribuciones de features). Yosinski muestra empiricamente:

$$\text{acc}(AnB^+) \approx \text{acc}(baseB) - \gamma(n) \cdot d(A, B)$$

con $\gamma(n)$ creciente en $n$ (capas mas profundas mas sensibles a distancia de tarea).

---

# Parte IV: Finetuning en Detalle

---

## 7. Discriminative Learning Rates

La observacion de Yosinski -- capas bajas mas transferibles que altas -- motiva **learning rates diferenciados**: menor para capas bajas, mayor para capas altas.

### 7.1 Formulacion

Dividir la red en $K$ grupos. Asignar:

$$\eta_k = \eta_0 \cdot \gamma^{K-k}$$

donde $k = 1$ es el grupo mas profundo (clasificador), $k = K$ el mas superficial. Con $\gamma \in [0.1, 0.5]$, las primeras capas reciben un LR **10x-100x menor** que las ultimas.

### 7.2 Ejemplo practico

ResNet-50, con $\eta_0 = 10^{-3}, \gamma = 0.1$:

| Grupo | Componentes | LR |
|---|---|---|
| $k=7$ (clasificador) | `fc` | $10^{-3}$ |
| $k=6$ | `layer4` | $10^{-4}$ |
| $k=5$ | `layer3` | $10^{-5}$ |
| $k=4$ | `layer2` | $10^{-6}$ |
| $k=3$ | `layer1` | $10^{-7}$ |
| $k \leq 2$ | `conv1`, `bn1` | $10^{-8}$ |

### 7.3 Implementacion

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import torch.optim as optim

optimizer = optim.Adam([
    {'params': model.conv1.parameters(),  'lr': 1e-6},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},
])
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import optax

def discriminative_optimizer(params):
    def mask_for_group(group_name):
        return jax.tree_util.tree_map_with_path(
            lambda path, _: group_name in str(path), params)

    return optax.chain(
        optax.masked(optax.adam(1e-6), mask_for_group('conv1')),
        optax.masked(optax.adam(1e-5), mask_for_group('layer1')),
        optax.masked(optax.adam(1e-5), mask_for_group('layer2')),
        optax.masked(optax.adam(1e-4), mask_for_group('layer3')),
        optax.masked(optax.adam(1e-4), mask_for_group('layer4')),
        optax.masked(optax.adam(1e-3), mask_for_group('classifier')),
    )
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf

# TF/Keras requires per-layer compile o usar tf.keras.optimizers.experimental
# con multiple_optimizers via tfa.optimizers.MultiOptimizer

import tensorflow_addons as tfa

optimizers_and_layers = [
    (tf.keras.optimizers.Adam(1e-6), model.get_layer('conv1')),
    (tf.keras.optimizers.Adam(1e-5), model.get_layer('layer1')),
    (tf.keras.optimizers.Adam(1e-5), model.get_layer('layer2')),
    (tf.keras.optimizers.Adam(1e-4), model.get_layer('layer3')),
    (tf.keras.optimizers.Adam(1e-4), model.get_layer('layer4')),
    (tf.keras.optimizers.Adam(1e-3), model.get_layer('classifier')),
]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
```
{{< /tab >}}
{{< /tabs >}}

---

## 8. Gradual Unfreezing (ULMFiT)

Howard & Ruder (2018) propusieron **gradual unfreezing**: descongelar capas una a la vez, desde la mas profunda a la mas superficial.

### 8.1 Procedimiento

Epoch 1: solo clasificador entrena.
Epoch 2: descongelar ultima capa del encoder + clasificador.
Epoch 3: descongelar dos ultimas capas + clasificador.
...
Epoch $K$: toda la red entrena.

### 8.2 Por que funciona

Combina los insights de Yosinski:

- Evita el **shock de fine-tuning**: no se descongelan capas criticas con pesos aun ruidosos de las capas superiores.
- Respeta la jerarquia **generic → specific**: las capas genericas se tocan al final, cuando la red ya converge en las especificas.
- **Regulariza implicitamente**: cada etapa es mas restringida, reduce overfitting en target pequeno.

---

# Parte V: Foundation Models como Transfer Learning Extremo

---

## 9. Escalado del Paradigma

Foundation models (BERT, GPT-3, CLIP) son el **caso extremo** de transfer learning:

| Aspecto | Transfer Learning clasico | Foundation Models |
|---|---|---|
| Source task | ImageNet classification | Next-token prediction, masked LM |
| Source data | 1.2M imagenes etiquetadas | Trillions de tokens sin etiquetar |
| Tamano del modelo | ~25M params (ResNet-50) | ~1.7T (GPT-4 estimado) |
| Adaptacion | Fine-tune ultimas capas | Fine-tune, prompt, RAG, PEFT |
| Target tasks | Decenas | Miles |

### 9.1 Scaling laws

Kaplan et al. (2020) mostraron que el loss sigue **leyes de escala predecibles**:

$$L(N) \propto N^{-\alpha}$$

donde $N$ es el numero de parametros y $\alpha \approx 0.076$ para language modeling. Duplicar el modelo reduce loss ~5%. Esta predictibilidad justifica inversiones masivas en scaling.

### 9.2 Emergent capabilities

A cierta escala, capacidades **emergen** no linealmente:

- In-context learning (GPT-3 a ~10B+ params).
- Chain-of-thought reasoning (GPT-3.5 a ~100B+ params).
- Instruction following robusto (post-RLHF).

Wei et al. (2022) documentan muchas: tareas que son **aleatoriamente mal** hasta cierto umbral de scale, y luego **substancialmente bien**.

### 9.3 Implicacion para transfer learning

En foundation models, **no transferimos una capa** -- transferimos **toda la red**. La adaptacion puede ser:

- **Fine-tune** (todas o algunas capas): mejor calidad, requiere compute.
- **Prompt engineering**: adaptar via texto en el input, sin gradient updates.
- **Retrieval-Augmented Generation (RAG)**: inyectar contexto externo.
- **PEFT** (LoRA, adapters, prefix tuning): fine-tunear pocos parametros adicionales.

---

# Parte VI: Evaluacion y Best Practices

---

## 10. Matriz de Decision para Transfer Learning

Combinando Yosinski + practica industrial:

### 10.1 Algoritmo de decision

```
dataset_size = len(target_dataset)
similarity = estimate_similarity(source, target)  # 0=disimilar, 1=identica

if dataset_size < 1000:
    if similarity > 0.7:
        # Feature extraction pura
        freeze_all_except_classifier()
        lr = 1e-3  # solo clasificador
    else:
        # Fine-tune capas medias
        freeze_early_layers(1, 2)  # capas genericas
        freeze_late_layers(6, 7)   # features muy especificos
        lr_classifier = 1e-3
        lr_middle = 1e-4

elif dataset_size < 50000:
    # Fine-tune completo con discriminative LRs
    for layer in network:
        set_lr(layer, base_lr * gamma**(max_layer - layer.depth))
    use_data_augmentation = True

else:  # dataset_size >= 50000
    # Fine-tune completo + augmentation agresiva
    all_layers_trainable()
    lr = 1e-4
    use_heavy_augmentation = True
    consider_train_from_scratch_as_baseline()
```

### 10.2 Checklist pre-produccion

- [ ] Usar **normalizacion correcta** (mean/std de ImageNet si preentrenado en ImageNet).
- [ ] LR **10x menor** que entrenar desde cero.
- [ ] **Data augmentation** apropiada al dominio.
- [ ] **Early stopping** con validation loss.
- [ ] Verificar que **el overfitting se reduzca** respecto a training desde cero.
- [ ] Si usas BERT/LLM: respetar **tokenizer** y **longitud maxima** del modelo source.
- [ ] Para modelos muy grandes: considerar **PEFT** (LoRA) en vez de fine-tune completo.

---

# Resumen Ejecutivo

1. **VRM generaliza ERM** reemplazando la distribucion empirica por una distribucion de vecindad. Data augmentation es VRM.
2. **Mixup** define una distribucion vecinal via convex combinations de pares + labels interpolados. Regulariza hacia comportamiento lineal entre ejemplos.
3. **Yosinski 2014** cuantifica transferibilidad layer-by-layer y separa dos mecanismos de degradacion: **fragile co-adaptation** (capas medias) y **representation specificity** (capas profundas).
4. **Transfer + fine-tune supera a training desde cero** incluso en datasets grandes (~1.6-2.1% mejor). Es **free lunch**.
5. **Discriminative learning rates** aplican menores $\eta$ a capas generales y mayores a capas especificas. **Gradual unfreezing** es la extension temporal.
6. **Foundation models** llevan transfer learning al extremo de escala: modelos multi-B de parametros preentrenados sobre trillions de tokens, adaptables via prompting/RAG/PEFT.

---

## Referencias

- Zhang, Cisse, Dauphin, Lopez-Paz (2017). [mixup: Beyond Empirical Risk Minimization](/papers/mixup-zhang-2017). *ICLR 2018*.
- Yosinski, Clune, Bengio, Lipson (2014). [How transferable are features in deep neural networks?](/papers/transferable-features-yosinski-2014) *NeurIPS*.
- Krizhevsky, Sutskever, Hinton (2012). [ImageNet Classification with Deep CNNs](/papers/alexnet-krizhevsky-2012). *NeurIPS*.
- Bommasani et al. (2021). [On the Opportunities and Risks of Foundation Models](/papers/foundation-models-bommasani-2021). *Stanford CRFM*.
- Chapelle, Weston, Bottou, Vapnik (2000). Vicinal Risk Minimization. *NIPS*.
- Howard & Ruder (2018). Universal Language Model Fine-tuning for Text Classification (ULMFiT). *ACL*.
- Yun et al. (2019). CutMix. *ICCV*.
- Kaplan et al. (2020). Scaling Laws for Neural Language Models. *arXiv 2001.08361*.
- Wei et al. (2022). Emergent Abilities of Large Language Models. *TMLR*.

Volver a [Teoria](teoria) | Hub de la [Clase 12](/clases/clase-12).
