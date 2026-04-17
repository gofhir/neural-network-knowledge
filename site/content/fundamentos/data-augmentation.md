---
title: "Data Augmentation"
weight: 83
math: true
---

**Data augmentation** es el conjunto de tecnicas que **sintetizan nuevos ejemplos de entrenamiento** aplicando transformaciones que preservan la etiqueta. Es una de las herramientas mas efectivas para combatir el **overfitting** y **mejorar la generalizacion** en problemas con datos limitados, especialmente en vision, audio y NLP.

---

## 1. El Problema: Small Data

La realidad de deep learning aplicado es que **la mayoria de los problemas reales tienen datos pequenos**. Un equipo de producto no cuenta con los 1.2M imagenes de ImageNet -- probablemente tiene 500, 5,000 o con suerte 50,000 ejemplos.

Las consecuencias:

- **Overfitting severo**: una red con millones de parametros memoriza el dataset en vez de aprender patrones.
- **Brechas train/test** del 30-40%.
- **Sensibilidad a variaciones menores** (rotaciones, ruido, cambios de iluminacion).
- **Adversarial examples**: perturbaciones imperceptibles cambian drasticamente las predicciones.

{{< concept-alert type="clave" >}}
Data augmentation **no crea informacion nueva**, pero **expande la distribucion de entrenamiento** hacia la distribucion real del problema. Es un prior estructural fuerte: le decimos a la red "todas estas transformaciones conservan la clase", reduciendo el espacio de hipotesis plausibles.
{{< /concept-alert >}}

---

## 2. Principio Fundamental

Sea $(x, y)$ un ejemplo de entrenamiento y $\mathcal{T}$ una familia de transformaciones. Data augmentation expande el dataset con:

$$D_{\text{aug}} = \{(T(x), y) : T \in \mathcal{T}, (x, y) \in D\}$$

**Condicion critica**: la transformacion $T$ **no debe afectar la relacion entrada-etiqueta**. Si rotamos una imagen de un "6" a 180 grados se convierte en un "9" -- esa rotacion **rompe la semantica** y no es valida.

### Marco formal: Vicinal Risk Minimization (VRM)

Chapelle et al. (2000) formalizaron data augmentation como aproximar la distribucion real $P(x, y)$ mediante una **distribucion de vecindad** $P_\nu$:

$$P_\nu(\tilde{x}, \tilde{y}) = \frac{1}{n} \sum_{i=1}^{n} \nu(\tilde{x}, \tilde{y} \mid x_i, y_i)$$

donde $\nu$ define el entorno alrededor de cada ejemplo. Minimizar el **riesgo vicinal empirico** reemplaza a la minimizacion de riesgo empirico (ERM) estandar.

Para imagenes, $\nu$ tipicamente incluye flips horizontales, rotaciones suaves y escalas leves.

---

## 3. Taxonomia de Tecnicas en Imagenes

### 3.1 Transformaciones geometricas

| Transformacion | Descripcion | Aplicable a |
|---|---|---|
| **Cropping** | Recortar regiones aleatorias | Casi todo (evitar perder el objeto clave) |
| **Flips horizontales** | Espejo horizontal | Naturaleza, objetos (no numeros, texto) |
| **Flips verticales** | Espejo vertical | Imagenes satelitales, microscopia |
| **Rotaciones** | Angulos arbitrarios | Imagenes con orientacion irrelevante |
| **Scaling** | Zoom in/out | Robustez a escala |
| **Shearing / affine** | Distorsiones afines | Reconocimiento de documentos, OCR |
| **Translation** | Shifts | Objetos pequenos en canvas grande |

### 3.2 Transformaciones foto-metricas

| Transformacion | Descripcion |
|---|---|
| **Brightness / contrast** | Cambios de intensidad |
| **Saturation / hue** | Cambios de color |
| **Color jittering** | Combinacion aleatoria de los anteriores |
| **PCA color augmentation** (AlexNet) | Perturbar canales RGB con multiplos de las componentes principales del dataset |
| **Gaussian noise** | Anadir ruido pixelwise |
| **Gaussian blur** | Desenfoque suave |
| **Cutout / erasing** | Borrar parches rectangulares |

### 3.3 Augmentations compuestas (modernas)

| Tecnica | Idea | Referencia |
|---|---|---|
| **Mixup** | $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, $\tilde{y} = \lambda y_i + (1-\lambda) y_j$ | [Zhang 2017](/papers/mixup-zhang-2017) |
| **CutMix** | Reemplazar parches entre imagenes, mezclar labels | Yun et al. 2019 |
| **CutOut** | Borrar parches cuadrados aleatoriamente | DeVries & Taylor 2017 |
| **AutoAugment** | Buscar la mejor politica de augmentation con RL | Cubuk et al. 2018 |
| **RandAugment** | Politica simple parametrizada, sin buscar | Cubuk et al. 2020 |
| **AugMix** | Combinar multiples transformaciones con consistency loss | Hendrycks et al. 2019 |

---

## 4. Mixup en Detalle

Mixup (Zhang et al. 2017, ICLR 2018) es una de las tecnicas mas simples y efectivas de augmentation moderna:

$$
\begin{aligned}
\tilde{x} &= \lambda \, x_i + (1 - \lambda) \, x_j \\
\tilde{y} &= \lambda \, y_i + (1 - \lambda) \, y_j \\
\lambda &\sim \text{Beta}(\alpha, \alpha)
\end{aligned}
$$

donde $y_i, y_j$ son **one-hot labels** y $\alpha \in [0.1, 0.4]$ controla la fuerza de la mezcla.

### Intuicion

Mixup fuerza al modelo a **comportarse linealmente entre ejemplos de distintas clases**. Esto:

- Suaviza las fronteras de decision.
- Reduce la confianza excesiva fuera del soporte de entrenamiento.
- Hace al modelo menos sensible a perturbaciones adversariales.
- Estabiliza el entrenamiento de GANs.

### Resultados (Mixup paper)

| Modelo | ERM | Mixup ($\alpha=0.2$) |
|---|---|---|
| ResNet-50 (ImageNet, 90 ep) | 23.5 | **23.3** |
| ResNet-101 (ImageNet, 90 ep) | 22.1 | **21.5** |
| PreAct ResNet-18 (CIFAR-10) | 5.6 | **4.2** |
| DenseNet-BC-190 (CIFAR-100) | 19.0 | **16.8** |

La implementacion cabe en 10 lineas:

{{< tabs >}}
{{< tab name="PyTorch" >}}
```python
import numpy as np

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
{{< /tab >}}
{{< tab name="JAX" >}}
```python
import jax
import jax.numpy as jnp

def mixup_batch(key, x, y, alpha=0.2):
    k1, k2 = jax.random.split(key)
    lam = jax.random.beta(k1, alpha, alpha)
    idx = jax.random.permutation(k2, x.shape[0])
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_mixed = lam * y + (1 - lam) * y[idx]  # y en one-hot
    return x_mixed, y_mixed

def loss_fn(params, batch):
    x_mixed, y_mixed = mixup_batch(batch['key'], batch['x'], batch['y'])
    logits = model.apply(params, x_mixed)
    return -jnp.mean(jnp.sum(y_mixed * jax.nn.log_softmax(logits), axis=-1))
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
import tensorflow as tf
import tensorflow_probability as tfp

@tf.function
def mixup_batch(x, y, alpha=0.2):
    batch_size = tf.shape(x)[0]
    lam = tfp.distributions.Beta(alpha, alpha).sample()
    idx = tf.random.shuffle(tf.range(batch_size))
    x_mixed = lam * x + (1 - lam) * tf.gather(x, idx)
    y_mixed = lam * y + (1 - lam) * tf.gather(y, idx)
    return x_mixed, y_mixed
```
{{< /tab >}}
{{< /tabs >}}

Ver el [paper original](/papers/mixup-zhang-2017) para la derivacion completa y las conexiones con Vicinal Risk Minimization.

---

## 5. Data Augmentation en Texto

Mucho mas dificil que en imagenes porque **cambios locales frecuentemente cambian la semantica**. Las estrategias comunes:

| Tecnica | Descripcion | Riesgo |
|---|---|---|
| **Synonym replacement** | Reemplazar palabras por sinonimos (WordNet) | Puede cambiar el sentimiento |
| **Random deletion** | Borrar palabras aleatorias con probabilidad $p$ | Puede borrar informacion critica |
| **Random insertion** | Insertar sinonimos en posiciones aleatorias | Puede introducir ruido |
| **Random swap** | Intercambiar palabras | Baja el mejor resultado |
| **Back-translation** | Traducir a otro idioma y de vuelta | Parafrasis natural, costoso |
| **Word embedding similarity** | Reemplazar por vecinos en word2vec / BERT | Mejor que sinonimos literales |
| **Contextual augmentation** | Usar un LM para predecir reemplazos | Muy efectivo con BERT |
| **EDA (Easy Data Augmentation)** | Combinacion de synonym+insert+swap+delete | Framework estandar |

Para **generacion condicionada**, la augmentation suele empeorar resultados -- prefiere back-translation o paraphrase generation dedicada.

---

## 6. Data Augmentation en Series Temporales

| Tecnica | Descripcion |
|---|---|
| **Time flipping** | Invertir direccion del tiempo |
| **Window cropping** | Usar ventanas variables |
| **Window warping** | Comprimir/expandir un rango temporal |
| **Frequency domain augmentation** | Aplicar ruido en espacio Fourier |
| **Magnitude warping** | Escalar la amplitud con curvas suaves |
| **Jittering / noise** | Ruido gaussiano pixelwise |

Caveats: en series temporales muchas transformaciones **si afectan la semantica** (ej. un ECG invertido es patologicamente distinto). Aplicar con conocimiento del dominio.

---

## 7. AlexNet Data Augmentation (Precedente Historico)

Krizhevsky, Sutskever y Hinton (2012) usaron dos formas de augmentation en [AlexNet](/papers/alexnet-krizhevsky-2012):

### 7.1 Random crops + horizontal flips

- Extraer crops aleatorios de **224×224** desde imagenes **256×256**.
- Cada flip horizontal es un nuevo ejemplo.
- Factor de aumento: **2048x** el dataset original.
- En inferencia: promediar predicciones sobre **10 crops** (4 esquinas + centro, mas sus flips).

### 7.2 PCA color augmentation

- Calcular PCA sobre los pixeles RGB del training set.
- En cada imagen, anadir multiplos aleatorios de las componentes principales:

$$[\Delta R, \Delta G, \Delta B]^T = [p_1, p_2, p_3][\alpha_1 \lambda_1, \alpha_2 \lambda_2, \alpha_3 \lambda_3]^T$$

donde $p_i$ son las componentes principales, $\lambda_i$ los eigenvalues y $\alpha_i \sim \mathcal{N}(0, 0.1)$.

Reduce top-1 error en **~1%**. Captura la intuicion de que la identidad del objeto es invariante a cambios en la **intensidad** y **color** de la iluminacion.

---

## 8. Consideraciones Importantes

{{< concept-alert type="advertencia" >}}
**La transformacion debe ser relevante para la tarea** y **no corromper la semantica del ejemplo**. Una augmentation util en ImageNet puede ser destructiva en diagnostico medico.
{{< /concept-alert >}}

Principios para disenar augmentations:

1. **Conservar la clase**: si $T$ cambia la etiqueta, no es augmentation -- es ruido.
2. **Generar nueva varianza**: si todas las transformaciones son casi identidad, no aportan.
3. **No corromper ejemplos**: un rotacion de 170 grados de una cara frontal produce algo que nunca veriamos en produccion.
4. **Considerar el dominio**: en medicina, la orientacion importa; en productos de retail, no.
5. **Balance**: demasiado fuerte empeora resultados (modelo no converge), muy debil no ayuda.

### Cuando la augmentation ayuda mas

| Situacion | Beneficio de augmentation |
|---|---|
| Dataset pequeno (< 10K ejemplos) | **Masivo** |
| Clases desbalanceadas | **Alto** (augmentar las minoritarias) |
| Test distribution difiere del train | **Alto** |
| Dataset enorme (> 1M ejemplos) | Bajo |
| Modelo ya regularizado agresivamente | Menor (pero aun ayuda) |

---

## 9. Tecnicas Experimentales / Avanzadas

### GAN / Adversarial-based

Entrenar un generador que sintetiza ejemplos de la clase objetivo:

- **SMOTE en deep**: generar embeddings intermedios.
- **Adversarial training** (Goodfellow 2014): generar ejemplos adversariales y entrenar para ser robusto.

### Reinforcement Learning-based

**AutoAugment** (Cubuk 2018): formular la busqueda de augmentations como RL, entrenando una politica que maximiza accuracy en un validation set. Encontro politicas mejores que las hand-crafted.

**RandAugment** (Cubuk 2020): simplificar drasticamente AutoAugment con 2 hiperparametros (numero de transformaciones aplicadas, magnitud). Casi tan bueno con mucho menos computo.

### Meta-learning

Aprender augmentations que se adapten al modelo y al dataset. Mas complejo, mas costoso, usualmente pequenas mejoras.

---

## 10. Combinacion con Transfer Learning

Data augmentation y [transfer learning](transfer-learning) son **complementarios**:

- Transfer learning aporta features preentrenadas (resuelve el problema de datos escasos en capas bajas).
- Data augmentation mejora las capas superiores (clasificador) fine-tuneadas.

**Receta estandar en produccion**:

1. Partir de un modelo preentrenado (ImageNet para vision, BERT para texto).
2. Aplicar augmentation agresiva durante fine-tuning.
3. Usar learning rate bajo (1/10 del original).
4. Early stopping en validation.

---

## 11. Resumen

- Data augmentation **expande la distribucion de entrenamiento** con transformaciones label-preserving.
- Para **imagenes**: geometricas (crops, flips, rotaciones), foto-metricas (color jitter), compuestas (Mixup, CutMix, AutoAugment).
- Para **texto**: dificil -- back-translation y synonym replacement son los mas comunes.
- Para **series temporales**: con cuidado del dominio (flipping, warping, jittering).
- **Mixup** es la augmentation moderna mas citada: convex combinations de pares + labels interpolados.
- **Aumenta si**: dataset pequeno, clases desbalanceadas, test difiere de train.
- **Combinar** con [transfer learning](transfer-learning) y [regularizacion](regularizacion) para resultados maximos.

Ver tambien: [Transfer Learning](transfer-learning) · [Regularizacion](regularizacion) · [Foundation Models](foundation-models) · [Paper Mixup](/papers/mixup-zhang-2017) · [Paper AlexNet](/papers/alexnet-krizhevsky-2012) · [Clase 12](/clases/clase-12).
