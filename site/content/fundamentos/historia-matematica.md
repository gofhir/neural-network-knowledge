---
title: "Historia Matematica"
weight: 90
math: true
---

La historia del deep learning es una convergencia de dos lineas independientes: la **matematica pura de optimizacion** y las **redes neuronales artificiales**. Cada idea nacio como solucion a un problema concreto de su epoca, y cada generacion construyo sobre los fundamentos de sus predecesores.

Para el tratamiento completo, ver [Clase 10 - Historia Matematica](/clases/clase-10/historia-matematica/).

---

## Linea de Tiempo

```mermaid
timeline
    title De Newton a Transformer
    1669 : Newton — Busqueda iterativa de raices
    1744 : Euler — Ecuacion de Euler-Lagrange
    1809 : Gauss — Metodo de minimos cuadrados
    1847 : CAUCHY — GRADIENT DESCENT
    1943 : McCulloch-Pitts — Primera neurona matematica
    1949 : Hebb — Primera regla de aprendizaje
    1951 : Robbins-Monro — Fundamentos de SGD
    1958 : Rosenblatt — Perceptron
    1964 : POLYAK — MOMENTUM (Heavy Ball)
    1969 : Minsky-Papert — 1er Invierno IA
    1974 : Werbos — Backpropagation
    1982 : Hopfield — Redes de energia
    1983 : NESTEROV — GRADIENTE ACELERADO O(1/k2)
    1986 : Rumelhart+ — Backprop popularizado
    1989 : Cybenko — Aprox. Universal
         : LeCun — CNNs para digitos
    2006 : Hinton — Deep Belief Networks
    2011 : DUCHI+ — ADAGRAD
    2012 : Hinton — RMSPROP
         : Krizhevsky — AlexNet
    2014 : KINGMA y BA — ADAM
    2015 : He et al. — ResNets
    2017 : Vaswani et al. — Transformer
```

---

## Hitos Fundamentales

### Cauchy y el Gradient Descent (1847)

En solo 3 paginas, Cauchy formulo el primer algoritmo explicito de optimizacion iterativa:

$$x^{(k+1)} = x^{(k)} - \alpha \nabla F(x^{(k)})$$

{{< concept-alert type="clave" >}}
**La ecuacion de Cauchy de 1847 es exactamente lo que `loss.backward(); optimizer.step()` hace en PyTorch hoy.** La matematica no cambio -- lo que cambio es la escala.
{{< /concept-alert >}}

### Robbins-Monro y los Cimientos de SGD (1951)

Establecieron las condiciones formales de convergencia: $\sum_n a_n = \infty$ (poder alcanzar el optimo) y $\sum_n a_n^2 < \infty$ (el ruido se promedia). Los learning rate schedules modernos estan motivados por estas condiciones.

### Rumelhart, Hinton, Williams (1986)

Reemplazaron la funcion escalon con el **sigmoid** diferenciable, haciendo la red entrenable de extremo a extremo con backpropagation. El XOR -- imposible para un perceptron -- se resolvio trivialmente con una red 2-2-1.

### De Polyak a Adam (1964-2015)

La evolucion de los optimizadores sigue un patron claro:

| Problema identificado | Solucion |
|---|---|
| GD es lento | **Momentum** (Polyak, 1964) -- inercia |
| Momentum es ciego | **Nesterov** (1983) -- mirar adelante |
| LR unico para todos los pesos | **AdaGrad** (2011) -- LR adaptativo |
| AdaGrad decae a cero | **RMSProp** (2012) -- media movil exponencial |
| Combinar lo mejor | **Adam** (2015) -- momentum + adaptividad |

---

## El Arco Narrativo

```mermaid
graph TD
    subgraph Mat["Matematica Pura"]
        direction TB
        M1["Euler 1744"]:::math --> M2["Cauchy 1847"]:::math
        M2 --> M3["Polyak 1964"]:::math
        M3 --> M4["Nesterov 1983"]:::math
        M4 --> M5["Duchi 2011 — AdaGrad"]:::math
        M5 --> M6["Hinton 2012 — RMSProp"]:::math
        M6 --> M7["Kingma 2014 — Adam"]:::math
    end

    subgraph NN["Redes Neuronales"]
        direction TB
        N1["McCulloch-Pitts 1943"]:::nn --> N2["Hebb 1949"]:::nn
        N2 --> N3["Rosenblatt 1958"]:::nn
        N3 --> N4["Minsky-Papert 1969<br/><small>1er Invierno IA</small>"]:::winter
        N4 --> N5["Werbos 1974"]:::nn
        N5 --> N6["Rumelhart 1986"]:::nn
        N6 --> N7["Cybenko y LeCun 1989"]:::nn
        N7 --> N8["2do Invierno<br/><small>~1988-2006</small>"]:::winter
        N8 --> N9["Hinton 2006 / AlexNet 2012"]:::nn
    end

    M4 -->|CONVERGENCIA| N7
    M7 -->|FUSION| DL["Deep Learning moderno"]:::output
    N9 --> DL

    classDef math fill:#2563eb,color:#fff,stroke:#1e40af
    classDef nn fill:#f59e0b,color:#fff,stroke:#d97706
    classDef winter fill:#ef4444,color:#fff,stroke:#dc2626
    classDef output fill:#059669,color:#fff,stroke:#047857
```

{{< concept-alert type="recordar" >}}
La historia muestra un patron recurrente: (1) la teoria matematica establece lo posible, (2) las limitaciones practicas se identifican, (3) breakthroughs algoritmicos las superan. Cada generacion construyo directamente sobre sus predecesores.
{{< /concept-alert >}}

---

## Los Breakthroughs que Terminaron el Invierno

| Ano | Innovacion | Contribucion |
|---|---|---|
| 2006 | Deep Belief Networks (Hinton) | Pretraining no-supervisado |
| 2010 | ReLU (Nair & Hinton) | Derivada = 1, sin vanishing gradient |
| 2012 | AlexNet | CNN profunda en GPUs gano ImageNet |
| 2015 | Batch Normalization | Permite learning rates mas altos |
| 2015 | ResNets (He et al.) | Skip connections para profundidad arbitraria |
| 2015 | Adam (Kingma & Ba) | Momentum + adaptividad |
| 2017 | Transformer (Vaswani et al.) | "Attention is all you need" |

---

## Para el Tratamiento Completo

- [Clase 10 - Historia Matematica: De Cauchy a Adam](/clases/clase-10/historia-matematica/) -- Newton-Raphson, Euler-Lagrange, Gauss, metodos de segundo orden, McCulloch-Pitts, Perceptron, Hopfield, demostraciones formales
