---
title: "Capas Convolucionales"
weight: 10
math: true
---

## 1. Por que no usar MLPs para imagenes

Las capas lineales (fully connected) de un MLP tienen un problema fundamental cuando trabajan con imagenes: cada neurona se conecta con **todos** los pixeles de la entrada. Esto implica dos limitaciones graves.

### Patrones globales vs locales

Una capa lineal aprende patrones **globales** — un template completo de la imagen. Si una cara aparece en una posicion diferente, el MLP la trata como un patron completamente distinto. En vision, lo que importa son los patrones **locales**: bordes, texturas, partes de objetos. Estos patrones se repiten en distintas posiciones de la imagen, y queremos que la red los detecte independientemente de donde aparezcan.

### Explosion de parametros

La cantidad de parametros de una capa lineal es $\text{dim\_in} \times \text{dim\_out}$. Para una imagen de $224 \times 224 \times 3 = 150{,}528$ pixeles con una capa oculta de 100 neuronas, ya tenemos $\sim 15$ millones de parametros. Si la imagen es de $1024 \times 1024 \times 3$, los parametros superan los $300$ millones solo en la primera capa.

Las capas convolucionales resuelven ambos problemas: aprenden patrones locales con filtros pequenos, y comparten pesos en toda la imagen.

{{< concept-alert type="clave" >}}
Las capas convolucionales son una forma alternativa de computar capas ocultas de una red neuronal. Requieren menos parametros que una capa lineal equivalente y son buenas para aprender **patrones locales** en las entradas.
{{< /concept-alert >}}

---

## 2. La operacion de convolucion

En vez de organizar los parametros en una matriz grande, los organizamos en un conjunto de **filtros** (tambien llamados **kernels**). Cada filtro es tipicamente mucho mas pequeno que la entrada — por ejemplo, $3 \times 3$, $5 \times 5$ u $11 \times 11$ pixeles.

El filtro se desliza sobre la imagen de entrada, y en cada posicion se calcula el **producto punto** entre el filtro y la region local de la imagen. Cada resultado es una **neurona** del mapa de activacion de salida.

### Ejemplo: filtro $3 \times 3$ sobre entrada $5 \times 5$

Supongamos una entrada de $5 \times 5$ y un filtro de $3 \times 3$:

$$
\text{salida}(i, j) = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} \text{entrada}(i+m, j+n) \cdot \text{filtro}(m, n)
$$

El filtro se posiciona en la esquina superior izquierda, calcula el producto punto, avanza una posicion a la derecha, calcula de nuevo, y asi sucesivamente. El conjunto de todos los resultados forma el **mapa de activacion** (o **feature map**).

---

## 3. Hiperparametros de la convolucion

### Filtros (kernels)

Cada filtro detecta un tipo de patron. Un filtro puede aprender a detectar bordes horizontales, otro bordes verticales, otro texturas. El numero de filtros determina cuantos patrones diferentes puede detectar una capa. Este numero es el parametro `out_channels` en PyTorch.

### Stride

El **stride** es el numero de pixeles que el filtro avanza en cada paso. Con stride 1, el filtro se mueve de pixel en pixel. Con stride 2, salta de dos en dos, lo que reduce la dimension de salida a la mitad (aproximadamente).

### Padding

El **padding** agrega pixeles (generalmente ceros) alrededor de la entrada antes de aplicar la convolucion. Esto permite controlar la dimension de salida. Con el padding adecuado, la salida puede tener el mismo tamano que la entrada.

---

## 4. Formula de dimension de salida

Este es uno de los calculos mas importantes al disenar una CNN. Dados:

- $H, W$: alto y ancho de la entrada
- $K_H, K_W$: tamano del kernel
- $S_H, S_W$: stride
- $P$: padding

La dimension de salida es:

$$O_H = \left\lfloor\frac{H - K_H + 2P}{S_H}\right\rfloor + 1$$

$$O_W = \left\lfloor\frac{W - K_W + 2P}{S_W}\right\rfloor + 1$$

### Ejemplo: primera capa de AlexNet

Entrada $224 \times 224$, kernel $11 \times 11$, stride $4$, padding $2$:

$$O = \left\lfloor\frac{224 - 11 + 2 \cdot 2}{4}\right\rfloor + 1 = \left\lfloor\frac{217}{4}\right\rfloor + 1 = 54 + 1 = 55$$

La salida es $55 \times 55$ por cada filtro.

---

## 5. Feature maps y canales

La salida de una capa convolucional tiene tres dimensiones: $C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}$, donde $C_{\text{out}}$ es el numero de filtros. Cada filtro produce un feature map 2D, y todos juntos forman un volumen 3D.

La cantidad de **neuronas** a la salida de una capa convolucional es:

$$\text{neuronas} = C_{\text{out}} \cdot H_{\text{out}} \cdot W_{\text{out}}$$

---

## 6. Weight sharing (pesos compartidos)

Un mismo filtro se aplica en **todas** las posiciones de la imagen. Esto significa que los pesos del filtro se comparten a lo largo de toda la entrada. Este mecanismo tiene dos consecuencias:

1. **Reduce drasticamente la cantidad de parametros** — un filtro de $3 \times 3$ tiene solo 9 pesos, sin importar si la imagen es de $32 \times 32$ o de $1024 \times 1024$.
2. **Invarianza a traslaciones** — si el filtro aprende a detectar un borde, lo detectara en cualquier posicion de la imagen.

---

## 7. Calculo de parametros

La cantidad de parametros de una capa convolucional es:

$$\text{params} = C_{\text{in}} \cdot C_{\text{out}} \cdot K_H \cdot K_W + B$$

Donde $B = C_{\text{out}}$ si la capa usa bias (lo habitual), y $B = 0$ en caso contrario.

### Ejemplo: primera capa de AlexNet

- $C_{\text{in}} = 3$ (RGB)
- $C_{\text{out}} = 96$ filtros
- $K = 11 \times 11$

$$\text{params} = 3 \times 96 \times 11 \times 11 + 96 = 34{,}944$$

Compare esto con una capa lineal equivalente: una entrada de $224 \times 224 \times 3 = 150{,}528$ neuronas conectada a $96 \times 55 \times 55 = 290{,}400$ neuronas de salida requeriria $\sim 43$ **mil millones** de parametros. La capa convolucional logra lo mismo con menos de $35{,}000$.

{{< concept-alert type="clave" >}}
La eficiencia de las capas convolucionales viene de dos propiedades: **localidad** (cada filtro solo mira una region pequena) y **weight sharing** (el mismo filtro se aplica en toda la imagen). Juntas reducen los parametros en ordenes de magnitud respecto a capas lineales.
{{< /concept-alert >}}

---

## 8. Implementacion en PyTorch

```python
import torch.nn as nn

# Capa convolucional basica
conv = nn.Conv2d(
    in_channels=3,       # canales de entrada (RGB)
    out_channels=96,     # numero de filtros
    kernel_size=(11, 11),# tamano del kernel
    stride=(4, 4),       # stride
    padding=(2, 2)       # padding
)

# Verificar dimensiones
import torch
x = torch.randn(1, 3, 224, 224)  # batch=1, canales=3, 224x224
y = conv(x)
print(y.shape)  # torch.Size([1, 96, 55, 55])

# Contar parametros
total = sum(p.numel() for p in conv.parameters())
print(f"Parametros: {total}")  # 34944
```

### Bloque convolucional tipico

En la practica, una capa convolucional casi siempre va acompanada de una funcion de activacion y, opcionalmente, una operacion de pooling:

```python
bloque_conv = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=96,
              kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
    nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
    nn.ReLU()
)
```

Este patron — convolucion, pooling, activacion — se repite en las arquitecturas clasicas de CNNs como AlexNet.
