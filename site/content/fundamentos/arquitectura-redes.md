---
title: "Arquitectura de Redes Neuronales"
weight: 20
math: true
---

Una red neuronal es un sistema de funciones compuestas, organizado en **capas**, que transforma datos de entrada en predicciones. Esta pagina cubre el modelo matematico de la neurona, las funciones de activacion y como se combinan en redes profundas.

---

## 1. El Modelo de la Neurona

Una neurona artificial recibe multiples entradas, las combina linealmente y aplica una funcion de activacion no-lineal:

{{< math-formula title="Modelo de una neurona" >}}
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(W^T x + b)
{{< /math-formula >}}

Donde:
- $x_i$ son las entradas
- $w_i$ son los **pesos** (parametros aprendibles)
- $b$ es el **bias**
- $f$ es la **funcion de activacion**

Sin la funcion de activacion, multiples capas lineales colapsarian en una sola transformacion lineal. La no-linealidad es lo que permite a las redes profundas aproximar funciones arbitrariamente complejas.

{{< concept-alert type="clave" >}}
**Teorema de Aproximacion Universal (Cybenko, 1989):** Una red con una sola capa oculta y suficientes neuronas puede aproximar cualquier funcion continua con precision arbitraria. Sin embargo, la profundidad permite representaciones exponencialmente mas compactas.
{{< /concept-alert >}}

---

## 2. Capas y Arquitectura

### Perceptron Multicapa (MLP)

Un MLP organiza las neuronas en capas completamente conectadas:

```mermaid
graph LR
    subgraph Entrada["Capa entrada"]
        x1([x1]):::input
        x2([x2]):::input
        x3([x3]):::input
    end
    subgraph Oculta1["Capa oculta 1"]
        h1a([h1]):::hidden
        h2a([h2]):::hidden
        h3a([h3]):::hidden
    end
    subgraph Oculta2["Capa oculta 2"]
        h1b([h1]):::hidden2
        h2b([h2]):::hidden2
        h3b([h3]):::hidden2
    end
    subgraph Salida["Capa salida"]
        yy([y]):::output
    end

    x1 --> h1a & h2a & h3a
    x2 --> h1a & h2a & h3a
    x3 --> h1a & h2a & h3a
    h1a --> h1b & h2b & h3b
    h2a --> h1b & h2b & h3b
    h3a --> h1b & h2b & h3b
    h1b --> yy
    h2b --> yy
    h3b --> yy

    classDef input fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef hidden fill:#2563eb,color:#fff,stroke:#1e40af
    classDef hidden2 fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef output fill:#059669,color:#fff,stroke:#047857
```

### Forward Pass

El forward pass es la evaluacion secuencial de cada capa:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = f(z^{(l)})$$

Donde $a^{(0)} = x$ es la entrada y $a^{(L)}$ es la salida de la red.

---

## 3. Funciones de Activacion

### Sigmoide

$$\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))$$

- Rango: $(0, 1)$. Util como salida para clasificacion binaria.
- **Problema:** Gradiente maximo de 0.25, causa vanishing gradient en redes profundas.

### Tangente Hiperbolica (Tanh)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

- Centrada en cero (mejora sobre Sigmoide), pero aun sufre vanishing gradient.

### ReLU (Rectified Linear Unit)

{{< math-formula title="ReLU" >}}
\text{ReLU}(x) = \max(0, x), \quad \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
{{< /math-formula >}}

{{< concept-alert type="clave" >}}
**ReLU es la activacion por defecto** para capas ocultas. Converge hasta 6x mas rapido que Sigmoide (Krizhevsky et al., 2012) porque su gradiente es constante (1) en la zona positiva. Su unico problema: neuronas que reciben siempre entradas negativas "mueren" permanentemente.
{{< /concept-alert >}}

### Leaky ReLU / PReLU

$$\text{PReLU}(x) = \max(\alpha x, x)$$

Con $\alpha$ pequeno (ej. 0.01), las neuronas nunca dejan de aprender.

### Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Convierte logits en probabilidades que suman 1. Se usa en la capa de salida para clasificacion multiclase.

### Tabla Comparativa

| Funcion | Rango | Centrada en 0 | Vanishing Gradient | Uso tipico |
|---------|-------|---------------|-------------------|------------|
| Sigmoide | $(0, 1)$ | No | Si | Salida binaria |
| Tanh | $(-1, 1)$ | Si | Si | Capas ocultas (RNN) |
| ReLU | $[0, +\infty)$ | No | Solo negativos | Capas ocultas (CNN, MLP) |
| Leaky ReLU | $(-\infty, +\infty)$ | No | No | Capas ocultas |
| Softmax | $(0, 1)$ | N/A | N/A | Salida multiclase |

---

## 4. Inicializacion de Pesos

En redes profundas, una mala inicializacion produce vanishing o exploding signals:

$$y = W^{[L]} \cdot W^{[L-1]} \cdots W^{[1]} \cdot x$$

Si los pesos son consistentemente $< 1$, la senal se desvanece. Si son $> 1$, explota.

{{< math-formula title="Inicializacion Xavier/Glorot" >}}
\text{Var}(W) = \frac{2}{\text{fan\_in} + \text{fan\_out}}
{{< /math-formula >}}

| Inicializacion | Formula | Activacion recomendada |
|---|---|---|
| **Xavier/Glorot** (2010) | $\text{Var} = 2/(\text{fan\_in} + \text{fan\_out})$ | Sigmoid, Tanh |
| **He/Kaiming** (2015) | $\text{Var} = 2/\text{fan\_in}$ | ReLU y variantes |

---

## 5. El Pipeline Completo

Una red neuronal tipica sigue estos pasos:

```text
1. DATOS:        Cargar + normalizar + armar batches
2. RED:          Definir capas (Linear, Conv2d, etc.)
3. LOSS:         Funcion que mide el error
4. OPTIMIZADOR:  Algoritmo que ajusta pesos
5. ENTRENAR:     forward -> loss -> backward -> update
6. EVALUAR:      Probar con datos nunca vistos
```

Cada uno de estos componentes se cubre en detalle en las paginas siguientes de esta seccion.

---

## Para Profundizar

- [Clase 06 - Grafos de Computo, Activaciones e Inicializacion](/clases/clase-06/) -- Funciones de activacion, Xavier, grafos
- [Clase 07 - Conceptos y Definiciones](/clases/clase-07/) -- Frameworks, normalizations, CNN pipelines
