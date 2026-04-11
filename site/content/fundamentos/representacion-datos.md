---
title: "Representacion de Datos"
weight: 10
math: true
---

Una red neuronal nunca ve fotos, palabras ni sonidos. **Solo ve arrays de numeros con una forma (shape) especifica.** Antes de que cualquier modelo pueda aprender, los datos del mundo real deben transformarse en tensores numericos que la red pueda procesar.

---

## 1. Tensores: la Estructura Fundamental

Un **tensor** es la generalizacion de escalares, vectores y matrices a dimensiones arbitrarias:

| Dimensiones | Nombre | Ejemplo |
|-------------|--------|---------|
| 0 | Escalar | `5.0` |
| 1 | Vector | `[1, 2, 3]` |
| 2 | Matriz | `[[1, 2], [3, 4]]` |
| 3+ | Tensor | Imagen RGB: alto x ancho x 3 canales |

En PyTorch, los tensores tienen dos superpoderes: pueden ejecutarse en **GPU** y pueden rastrear operaciones para calcular **gradientes automaticamente** (autograd).

---

## 2. Tipos de Datos de Entrada

Cada tipo de dato tiene un shape estandar y un preproceso particular:

| Tipo | Shape tipico | Preproceso |
|------|-------------|-----------|
| Tabular (CSV) | `(batch, features)` | Ya son numeros |
| Imagenes | `(batch, canales, alto, ancho)` | Dividir por 255, normalizar |
| Texto | `(batch, tokens, embedding_dim)` | Tokenizar + Embedding |
| Audio | `(batch, 1, frecuencias, tiempo)` | Muestreo + Espectrograma |

{{< concept-alert type="clave" >}}
**Imagenes como tensores:** Una imagen RGB de 224x224 se representa como un tensor de shape `(3, 224, 224)` -- 3 canales (rojo, verde, azul), 224 filas, 224 columnas. Cada valor es un numero entre 0 y 255 que se normaliza a [0, 1] dividiendo por 255.
{{< /concept-alert >}}

---

## 3. Normalizacion de Datos

La normalizacion asegura que todas las features tengan escalas comparables, lo cual es esencial para que el entrenamiento converja de forma estable.

### Z-score (estandarizacion)

{{< math-formula title="Normalizacion Z-score" >}}
x_{\text{norm}} = \frac{x - \mu}{\sigma}
{{< /math-formula >}}

Transforma los datos para que tengan media 0 y varianza 1.

### Min-Max

$$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Transforma los datos al rango $[0, 1]$.

{{< concept-alert type="recordar" >}}
**Las imagenes se normalizan por canal.** En ImageNet, la normalizacion estandar usa `mean = [0.485, 0.456, 0.406]` y `std = [0.229, 0.224, 0.225]` para los canales RGB respectivamente. Esto garantiza que cada canal tenga distribucion similar.
{{< /concept-alert >}}

---

## 4. Grafos de Computo

Los frameworks de deep learning representan las operaciones sobre tensores como **grafos de computo**: grafos dirigidos donde los nodos son operaciones y las aristas representan el flujo de datos.

```mermaid
graph LR
    x([x]):::input --> add(("+")):::op
    y([y]):::input --> add
    add --> mul(("*")):::op
    z([z]):::input --> mul
    mul --> g([g]):::output

    classDef input fill:#2563eb,color:#fff,stroke:#1e40af
    classDef op fill:#f59e0b,color:#fff,stroke:#d97706
    classDef output fill:#059669,color:#fff,stroke:#047857
```

Estos grafos son fundamentales porque permiten:

1. **Forward pass**: evaluar la funcion en orden topologico
2. **Backward pass**: calcular gradientes automaticamente recorriendo el grafo en orden inverso (ver [Backpropagation](/fundamentos/backpropagation/))

PyTorch construye el grafo dinamicamente durante cada forward pass, lo que facilita el debugging y permite arquitecturas condicionales.

---

## 5. De Datos Crudos a Batches

El flujo tipico de preparacion de datos es:

```mermaid
graph LR
    A["Datos crudos<br/><small>archivos</small>"]:::step1 --> B["Preproceso<br/><small>normalizar, redimensionar</small>"]:::step2
    B --> C["Dataset<br/><small>pares x_i, y_i</small>"]:::step3
    C --> D["DataLoader<br/><small>agrupa en mini-batches</small>"]:::step4
    D --> E["Batches<br/><small>tensores listos</small>"]:::step5

    classDef step1 fill:#1e40af,color:#fff,stroke:#1e3a8a
    classDef step2 fill:#2563eb,color:#fff,stroke:#1e40af
    classDef step3 fill:#3b82f6,color:#fff,stroke:#2563eb
    classDef step4 fill:#60a5fa,color:#fff,stroke:#3b82f6
    classDef step5 fill:#059669,color:#fff,stroke:#047857
```

El **batch size** determina cuantos ejemplos se procesan juntos en cada iteracion. Esto impacta directamente la eficiencia computacional y la dinamica del entrenamiento (ver [Optimizadores](/fundamentos/optimizadores/) y [Learning Rate](/fundamentos/learning-rate/)).

---

## Para Profundizar

- [Clase 07 - Conceptos y Definiciones](/clases/clase-07/) -- Tipos de datos, normalizaciones, frameworks
- [Clase 06 - Grafos de Computo](/clases/clase-06/) -- Grafos computacionales y tensores en PyTorch
