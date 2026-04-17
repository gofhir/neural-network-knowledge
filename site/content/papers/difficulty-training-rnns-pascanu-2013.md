---
title: "Difficulty Training RNNs (Vanishing/Exploding)"
weight: 140
math: true
---

{{< paper-card
    title="On the difficulty of training Recurrent Neural Networks"
    authors="Pascanu, Mikolov, Bengio"
    year="2013"
    venue="ICML 2013"
    pdf="/papers/difficulty-training-rnns-pascanu-2013.pdf"
    arxiv="1211.5063" >}}
Provee el **analisis formal moderno** del problema de vanishing y exploding gradients en RNNs vanilla, dando condiciones suficientes y necesarias en terminos del mayor valor singular de la matriz recurrente. Propone **gradient norm clipping** como solucion practica para exploding gradients y discute soluciones para vanishing. Es el paper de referencia que cita cualquier trabajo serio sobre RNNs.
{{< /paper-card >}}

---

## Contexto

Bengio, Simard y Frasconi (1994) habian mostrado **empiricamente** que las RNNs sufren de gradients que decaen o explotan con la longitud de la secuencia. Pero la formalizacion en terminos de **valores singulares de $W_{rec}$** y la propuesta sistematica de **clipping** y otras soluciones quedo en este paper de Pascanu, Mikolov y Bengio (2013). Es el documento que profesionaliza el tratamiento del problema.

---

## Ideas principales

### 1. Analisis formal

Para una RNN $h_t = \sigma(W_{rec} h_{t-1} + W_{in} u_t + b)$, el gradiente del error en el paso $t$ respecto al estado en el paso $k < t$ se descompone como:

$$\frac{\partial \mathcal{E}_t}{\partial \theta} = \sum_{1 \leq k \leq t} \left( \frac{\partial \mathcal{E}_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial^+ h_k}{\partial \theta} \right)$$

donde el factor critico $\frac{\partial h_t}{\partial h_k} = \prod_{t \geq i > k} W_{rec}^T \, \text{diag}(\sigma'(h_{i-1}))$ es un producto de Jacobianas.

### 2. Condiciones formales

Sea $\lambda_1$ el mayor valor singular de $W_{rec}$ y $\gamma$ una cota de $|\sigma'|$ (1 para tanh, 1/4 para sigmoide):

| Condicion | Consecuencia |
|---|---|
| $\lambda_1 < \frac{1}{\gamma}$ | **Suficiente** para vanishing gradient (componentes a largo plazo se anulan exponencialmente) |
| $\lambda_1 > \frac{1}{\gamma}$ | **Necesaria** para exploding gradient |

Esto da una caracterizacion precisa: para tanh, **vanishing es seguro** cuando los valores singulares de $W_{rec}$ son menores que 1; **exploding es posible** cuando alguno excede 1.

### 3. Vista de sistemas dinamicos

El paper interpreta los gradientes desde la teoria de **sistemas dinamicos**:

- Los estados de la RNN convergen a uno o varios **atractores**.
- Las **bifurcaciones** (puntos donde cambia la estructura de atractores) crean **paredes empinadas** en el paisaje de error.
- Cruzar una bifurcacion genera saltos catastroficos -- esto es exploding gradient en accion.

```
Error landscape:    ___           ___
                   /   \   wall   /   \
                  /     \  |  |  /     \
   SGD step ───→ /       \_|  |_/       \
                                ^
                          aqui SGD salta lejos
```

### 4. Solucion: Gradient Norm Clipping

Algoritmo extremadamente simple, propuesto formalmente en este paper:

```text
g ← ∂L/∂θ
if ||g|| ≥ threshold:
    g ← (threshold / ||g||) · g
```

Preserva la **direccion** del gradiente, escala solo la **magnitud** cuando excede un umbral. Tipicamente threshold = 1 a 10.

Justificacion: en una pared empinada, el gradiente apunta correctamente lejos de la pared, pero su magnitud es enorme. Clipping permite tomar un paso bounded en la direccion correcta sin saltar fuera del valle.

### 5. Solucion: regularizacion de norma para vanishing

Para vanishing, el paper propone una regularizacion soft que penaliza cuando el factor temporal $\frac{\partial h_{t+1}}{\partial h_t}$ tiene norma menor que la del error que viene del futuro -- empuja a $W_{rec}$ a preservar normas. En la practica, **LSTM/GRU resuelven vanishing mas elegantemente** y han sido la solucion adoptada.

---

## Resultados

Experimentos en el **adding problem** (suma diferida de dos numeros en una secuencia) y modelado de lenguaje con varios tamanos de RNN. Conclusiones:

- Sin clipping: las RNNs vanilla **divergen** o quedan atrapadas en plateaus en secuencias largas.
- Con clipping: convergen consistentemente y aprenden dependencias de hasta cientos de pasos.
- LSTM (sin clipping) tambien funciona, pero clipping ayuda incluso a LSTMs cuando hay residuos de exploding.

---

## Por que importa hoy

- **Gradient clipping** es **estandar en cualquier entrenamiento de RNN, LSTM, Transformer o modelo grande**. Las APIs de PyTorch (`utils.clip_grad_norm_`) y TensorFlow (`clipnorm` en optimizers) lo implementan directamente.
- Los **LLMs modernos** (GPT, LLaMA, Claude) usan clipping rutinariamente durante el pre-entrenamiento.
- Las **condiciones espectrales** sobre $W_{rec}$ inspiraron la inicializacion ortogonal y los Echo State Networks.
- El insight de **paredes y bifurcaciones** explica por que entrenamientos a veces "se rompen" subitamente y motivo investigacion en optimizadores adaptativos (Adam, RMSprop) que escalan automaticamente.

---

## Notas y enlaces

- La derivacion completa esta en la Seccion 2 ("Exploding and Vanishing Gradients") y el Algoritmo 1 (clipping) en Seccion 3.2.
- Ver tambien: **Hochreiter 1991** (tesis original que identifica el problema), **Bengio et al. 1994** (analisis empirico previo), **LSTM (Hochreiter & Schmidhuber 1997)** como solucion arquitectural alternativa.
- Recurso adicional: PyTorch tutorial "Sequence Models and Long Short-Term Memory Networks" usa clipping explicitamente.

Ver fundamentos: [Backpropagation Through Time](/fundamentos/backpropagation-through-time) · [Redes Recurrentes](/fundamentos/redes-recurrentes).
