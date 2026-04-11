---
title: "Backpropagation"
weight: 80
math: true
---

{{< paper-card
    title="Learning representations by back-propagating errors"
    authors="Rumelhart, Hinton, Williams"
    year="1986"
    venue="Nature"
    pdf="/papers/1_Backpropagation_Rumelhart1986.pdf" >}}
Algoritmo para calcular eficientemente el gradiente del error respecto a cada peso en una red neuronal multicapa, permitiendo aprender representaciones internas utiles. El paper mas influyente en la historia del Deep Learning.
{{< /paper-card >}}

---

## El problema del credit assignment

Antes de 1986, las redes neuronales estaban limitadas al **Perceptron** (una sola capa de pesos ajustables). Minsky y Papert (1969) demostraron que el Perceptron no puede resolver problemas **no linealmente separables** (como XOR). La solucion era agregar capas ocultas, pero no existia un metodo practico para ajustar sus pesos.

{{< concept-alert type="clave" >}}
**El credit assignment problem**: en la capa de salida conocemos el error directo, pero en las capas ocultas no hay target -- como saber que "culpa" tiene cada neurona oculta en el error final?
{{< /concept-alert >}}

## Forward Pass y funcion de error

Para cada neurona $j$, la activacion se calcula en dos pasos:

1. **Entrada neta**: $\text{net}_j = \sum_i w_{ji} \cdot y_i + b_j$
2. **Activacion sigmoide**: $y_j = \sigma(\text{net}_j) = \frac{1}{1 + e^{-\text{net}_j}}$

La funcion de error es el **error cuadratico medio**:

$$E = \frac{1}{2} \sum_c \sum_j (t_{cj} - y_{cj})^2$$

## La regla de backpropagation

{{< math-formula title="Regla de actualizacion de pesos" >}}
\Delta w_{ji} = \eta \cdot \delta_j \cdot y_i
{{< /math-formula >}}

donde $\eta$ es el learning rate y $\delta_j$ es la senal de error de la neurona $j$.

**Para neuronas de salida**:

$$\delta_j = (t_j - y_j) \cdot y_j \cdot (1 - y_j)$$

**Para neuronas ocultas** (la clave del algoritmo):

{{< math-formula title="Propagacion del error hacia atras" >}}
\delta_j = y_j (1 - y_j) \sum_k \delta_k \cdot w_{kj}
{{< /math-formula >}}

El error de una neurona oculta es la suma ponderada de los errores de las neuronas a las que esta conectada, multiplicada por la derivada de su activacion. Los errores se **propagan hacia atras** desde la salida hasta la entrada.

## Momentum

Los autores proponen un termino de momentum para acelerar la convergencia:

$$\Delta w_{ji}(t) = \eta \cdot \delta_j \cdot y_i + \alpha \cdot \Delta w_{ji}(t-1)$$

donde $\alpha \approx 0.9$ acelera en direcciones consistentes y amortigua oscilaciones.

## Resultados experimentales

- **XOR**: resuelve el problema imposible para el Perceptron
- **Encoder 8-3-8**: la red descubre independientemente la codificacion binaria al comprimir 8 patrones en 3 neuronas ocultas
- **Family Trees**: las neuronas ocultas aprenden features como generacion, genero y rama familiar sin supervision explicita
- **Generalizacion**: la red clasifica correctamente patrones no vistos durante el entrenamiento

{{< concept-alert type="clave" >}}
Backpropagation es simplemente una aplicacion eficiente de la **regla de la cadena** del calculo. Su complejidad es $O(W)$ por patron (igual que el forward pass), lo que lo hace practico para redes grandes.
{{< /concept-alert >}}

## Legado

| Tecnica moderna | Raiz en este paper |
|---|---|
| SGD | Online backpropagation |
| Adam, RMSprop | Extensiones del momentum |
| Autoencoders | Experimento encoder 8-3-8 |
| Word Embeddings | Representaciones distribuidas aprendidas |
| PyTorch autograd / TF GradientTape | Implementaciones de backpropagation |

Sin este paper, no existirian CNNs, RNNs, Transformers, GPT, BERT ni ninguna de las tecnologias de IA modernas. Es el algoritmo que hace posible la inteligencia artificial moderna.
