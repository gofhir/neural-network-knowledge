# Documentacion Teorica - Practico Clase 6

## Grafo de Computo, Inicializacion de Pesos y Funciones de Activacion

**Diplomado IA: Inteligencia Artificial I - Parte 2**
Profesores: Carlos Aspillaga, Gabriel Sepulveda | Ayudante: Bianca del Solar

---

## Tabla de Contenidos

1. [Algoritmo de Aprendizaje y Grafos de Computo](#1-algoritmo-de-aprendizaje-y-grafos-de-computo)
2. [Funciones de Activacion](#2-funciones-de-activacion)
3. [Inicializacion de Pesos](#3-inicializacion-de-pesos)
4. [Conceptos Clave de PyTorch](#4-conceptos-clave-de-pytorch)
5. [Explicacion Detallada del Codigo del Practico](#5-explicacion-detallada-del-codigo-del-practico)
6. [Resultados Experimentales](#6-resultados-experimentales)
7. [Guia de Actividades y Respuestas](#7-guia-de-actividades-y-respuestas)

---

## 1. Algoritmo de Aprendizaje y Grafos de Computo

### 1.1 Pasos de Entrenamiento

El entrenamiento de una red neuronal consta de 3 pasos fundamentales que se repiten ciclicamente:

1. **Forward (Propagacion hacia adelante):** Los datos de entrada pasan a traves del modelo, que genera una prediccion.
2. **Backward (Retropropagacion):** Se compara la prediccion con la etiqueta real mediante una funcion de perdida, y se propaga el error hacia atras para calcular los gradientes.
3. **Weights Update (Actualizacion de pesos):** Los pesos del modelo se actualizan segun el error cometido.

> **Idea clave:** Los pesos se actualizan segun el error cometido. Para ello necesitamos cuantificar el error **y su gradiente**.

### 1.2 Descenso del Gradiente (Regla Delta)

La actualizacion de pesos se realiza mediante la **Regla Delta**:

```
delta_w = -eta * (dE(w) / dw)
w <- w + delta_w
```

Donde:
- `E(w)` es la funcion de error (loss)
- `eta` (n) es el **learning rate** (tasa de aprendizaje)
- `dE(w)/dw` es el gradiente del error respecto a los pesos

### 1.3 Entrenamiento de un Perceptron

Para un perceptron simple con funcion de activacion `f`:

```
y_hat = f(sum(w_i * x_i) + w_0)
```

La funcion de error (MSE) es:

```
E(w) = (1/2) * (y - y_hat)^2
```

El gradiente del error respecto a cada peso es:

```
dE(w) / dw_i = (y - y_hat) * (df/dz) * x_i
```

Y la actualizacion de cada peso:

```
delta_w_i = -eta * dE(w) / dw_i
w_i <- w_i + delta_w_i
```

### 1.4 Perceptron Multicapa (MLP)

- Los MLPs combinan multiples perceptrones en varias capas
- Se conocen como **Deep Feed Forward Networks** (DFF)
- Permiten resolver problemas de mayor complejidad
- En teoria, pueden aproximar cualquier funcion matematica (Teorema de Aproximacion Universal)
- **Problema:** Como entrenar los pesos de las capas intermedias (hidden layers)?

### 1.5 Backpropagation

El algoritmo de Backpropagation resuelve el problema de entrenar capas ocultas:

- Aplica la **Regla de la Cadena** de forma recursiva
- Puede ser usado en redes de tamano arbitrario
- Funciona con cualquier tipo de red y funcion diferenciable
- **El rendimiento no esta garantizado**

### 1.6 Grafos de Computo

Los grafos de computo son la forma moderna de implementar backpropagation:

- Son **representaciones que permiten expresar y evaluar funciones matematicas**
- Consisten en **grafos dirigidos** donde:
  - Los **nodos** corresponden a operaciones matematicas
  - Las **aristas** representan el flujo de variables
- Los frameworks mas populares (TensorFlow, PyTorch, etc.) los utilizan para implementar backpropagation de forma automatica

**Ejemplo:** Para `g = (x + y) * z`:

```
x ---\
      (+) ---\
y ---/        (*) ---> g
z -----------/
```

**Ventaja:** No necesitas derivar manualmente las ecuaciones de backpropagation. El framework lo hace automaticamente a traves del grafo de computo.

---

## 2. Funciones de Activacion

### 2.1 Rol en la Red Neuronal

La arquitectura de una red neuronal se define por:
1. **Estructura:** numero de capas y neuronas por capa
2. **Metodo de calculo de pesos** W_ij que interconectan neuronas
3. **Funcion de Activacion:** modula la senal enviada desde una neurona a la siguiente

En un perceptron: `salida = f(sum(w_i * x_i) + b)`, donde `f` es la funcion de activacion.

### 2.2 Funcion Sigmoide

```
sigma(x) = 1 / (1 + e^(-x))
```

**Derivada:**
```
d(sigma)/dx = sigma(x) * (1 - sigma(x))
```

**Buenas propiedades:**
- Suave y derivable
- Lleva la salida a valores extremos (0 y 1), util para clasificacion

**Malas propiedades:**
1. **Neuronas saturadas tienen gradiente cercano a cero.** Cuando la entrada es muy positiva o muy negativa, la derivada se aproxima a 0. Esto causa:
   - Convergencia lenta
   - **Vanishing gradient** (desvanecimiento del gradiente): problema critico en redes profundas y redes recurrentes (RNNs)
2. **La salida no esta centrada en cero.** Los gradientes son todos positivos o todos negativos, lo que produce un efecto de **zig-zag** en la convergencia (actualizaciones ineficientes)

### 2.3 Funcion Tangente Hiperbolica (Tanh)

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = 2*sigma(2x) - 1
```

**Buenas propiedades:**
- Suave y derivable
- Lleva salida a valores entre (-1, 1)
- **Centrada en cero** (mejora respecto a Sigmoide)

**Malas propiedades:**
- Neuronas en saturacion tienen gradiente cercano a cero
- Convergencia lenta y problema de **vanishing gradient**
- Extremadamente sensible para valores de salida cercanos a cero

### 2.4 ReLU (Rectified Linear Unit)

```
ReLU(x) = max(0, x)
```

**Buenas propiedades:**
- No se satura para entradas positivas
- **Convergencia mucho mas rapida** que usando sigmoide (ej. 6x mas rapido segun Krizhevsky et al., 2012, en CIFAR-10)
- Eficiente computacionalmente (es una funcion lineal)

**Malas propiedades:**
1. No es derivable en cero
2. **Se satura para entradas negativas** (gradiente = 0, la neurona "muere")
3. No es util directamente para problemas de clasificacion (en capa de salida)

**Derivada:**
- Si x > 0: gradiente = 1
- Si x <= 0: gradiente = 0

> **Esto es clave para la Actividad 2.2 del practico:** ReLU aprende mas rapido que Sigmoide porque su gradiente en la zona positiva es constante (= 1), mientras que el gradiente de Sigmoide es como maximo 0.25 y disminuye hacia los extremos.

### 2.5 Leaky ReLU / Parametric ReLU (PReLU)

```
PReLU(x) = max(alpha * x, x)
```

Donde `alpha` es un parametro pequeno (ej. 0.01).

**Buenas propiedades:**
- No se satura (las neuronas no dejan de aprender, ni siquiera para valores negativos)
- Convergencia rapida (similar a ReLU)
- Eficiente computacionalmente

**Malas propiedades:**
- No es util directamente para clasificacion (en capa de salida)

### 2.6 SoftMax

```
softmax(z_i) = e^(z_i) / sum_j(e^(z_j))
```

- Usada normalmente en **capas de salida** para clasificacion
- Convierte la salida de las neuronas en valores interpretables como **probabilidad** (suman 1)
- Es derivable
- Funciona como un "maximizador suave": amplifica valores altos y atenua valores bajos
- Incentiva la competencia entre salidas (*the winner takes it all*)

### 2.7 Tabla Comparativa

| Funcion    | Rango        | Centrada en 0 | Vanishing Gradient | Velocidad | Uso tipico             |
|------------|--------------|---------------|--------------------|-----------|------------------------|
| Sigmoide   | (0, 1)       | No            | Si                 | Lenta     | Salida binaria         |
| Tanh       | (-1, 1)      | Si            | Si                 | Lenta     | Capas ocultas (RNN)    |
| ReLU       | [0, +inf)    | No            | Solo negativos     | Rapida    | Capas ocultas (CNN)    |
| Leaky ReLU | (-inf, +inf) | No            | No                 | Rapida    | Capas ocultas          |
| SoftMax    | (0, 1)       | N/A           | N/A                | N/A       | Capa salida (multiclase)|

---

## 3. Inicializacion de Pesos

### 3.1 El Problema

En redes profundas existen 4 grandes problemas relacionados con la inicializacion:

**En el Forward (senal de entrada):**
- **Vanishing input signal:** Si los pesos son < 1, la senal se atenua exponencialmente capa tras capa
- **Exploding input signal:** Si los pesos son > 1, la senal crece exponencialmente

**En el Backward (propagacion de gradientes):**
- **Vanishing gradient:** Los gradientes se hacen tan pequenos que los pesos de las primeras capas no se actualizan
- **Exploding gradient:** Los gradientes crecen exponencialmente, causando inestabilidad

### 3.2 Demostracion Matematica

Para una red de L capas sin funcion de activacion, la salida es:

```
y = w^[L] * w^[L-1] * ... * w^[2] * w^[1] * x
```

**Caso vanishing (pesos < 1):**
Si `w^[l] = [[0.5, 0], [0, 0.5]]`, entonces:
```
y = [[0.5, 0], [0, 0.5]]^(L-1) * x  -->  tiende a 0 cuando L es grande
```

**Caso exploding (pesos > 1):**
Si `w^[l] = [[1.5, 0], [0, 1.5]]`, entonces:
```
y = [[1.5, 0], [0, 1.5]]^(L-1) * x  -->  tiende a infinito cuando L es grande
```

Lo mismo aplica para los gradientes en el backward:
```
dy/dx_i = f'^[1] * f'^[2] * ... * f'^[L] * g
```

### 3.3 Inicializacion con Valor Constante (Experimento del Practico)

El practico demuestra estos problemas con una red de 10 capas lineales (sin activacion):

- **Pesos = 0.1 (< 1):** Se observa **vanishing gradient**. Los gradientes son extremadamente pequenos y el modelo no aprende eficientemente.
- **Pesos = 1.5 (> 1):** Se observa **exploding gradient**. Los gradientes son enormes e inestables, causando oscilaciones en el entrenamiento.

### 3.4 Inicializacion de Xavier Glorot

**Objetivo:** Que las senales fluyan apropiadamente en ambas direcciones (forward y backward).

**Estrategia:** Hacer que la **varianza de entrada y salida sean iguales** en cada capa.

**Formula (Glorot & Bengio, 2010):**

Los pesos se inicializan aleatoriamente desde una distribucion gaussiana con:
- **Valor medio:** 0 (cero)
- **Varianza:**

```
Var(W_i) = 2 / (fan_in + fan_out)
```

Donde:
- `fan_in`: numero de entradas a la capa
- `fan_out`: numero de salidas de la capa (hacia la capa siguiente)

**Efecto:** Al igualar las varianzas, se evita que la senal se amplifique o se atenue al pasar por cada capa, mitigando tanto el vanishing como el exploding gradient.

### 3.5 Implementacion en PyTorch

Funciones de inicializacion disponibles:
- `torch.nn.init.ones_(tensor)` - Inicializa con unos
- `torch.nn.init.zeros_(tensor)` - Inicializa con ceros
- `torch.nn.init.constant_(tensor, val)` - Inicializa con valor constante
- `torch.nn.init.xavier_uniform_(tensor, gain=1.0)` - **Inicializacion de Xavier (uniforme)**

> **Nota importante:** La inicializacion de pesos es una gran ayuda para evitar vanishing/exploding gradient, pero **no es una solucion definitiva**. Otras tecnicas como Batch Normalization, residual connections, etc., tambien son necesarias en redes muy profundas.

---

## 4. Conceptos Clave de PyTorch

### 4.1 Tensores

Un **tensor** es una estructura de datos que generaliza escalares, vectores y matrices a cualquier numero de dimensiones:

| Dimensiones | Nombre  | Ejemplo                                              |
|-------------|---------|------------------------------------------------------|
| 0           | Escalar | `5.0`                                                |
| 1           | Vector  | `[1, 2, 3]`                                         |
| 2           | Matriz  | `[[1, 2], [3, 4]]`                                  |
| 3+          | Tensor  | Una "caja" de numeros (ej. imagen RGB: alto x ancho x 3 canales) |

En PyTorch, `torch.Tensor` es el equivalente a un array de NumPy, pero con dos superpoderes:

1. **Puede ejecutarse en GPU** para calculos mas rapidos.
2. **Puede rastrear operaciones para calcular gradientes automaticamente.** Esto es lo que hace posible el backpropagation sin derivar a mano.

```python
# Tensor simple (no rastrea gradientes)
x = torch.Tensor([2.0])

# Tensor que SI rastrea gradientes (necesario para entrenar)
x = torch.tensor([2.0], requires_grad=True)
```

### 4.2 Variable (torch.autograd.Variable)

`Variable` es un **wrapper** (envoltorio) antiguo de los tensores que les agregaba la capacidad de rastrear gradientes.

```python
# Forma antigua (aparece en el practico)
from torch.autograd import Variable
x = Variable(torch.Tensor([2.0]), requires_grad=True)

# Forma moderna (equivalente, PyTorch actual)
x = torch.tensor([2.0], requires_grad=True)
```

**Son exactamente lo mismo.** Desde PyTorch 0.4, `Variable` ya no es necesaria porque los tensores absorbieron esa funcionalidad directamente. Si ves `requires_grad=True` en un tensor, ya hace todo lo que `Variable` hacia.

**Para que sirve `requires_grad=True`?** Le dice a PyTorch: "registra todas las operaciones que hagas con este tensor para que despues pueda calcular gradientes con `.backward()`". Es lo que construye el **grafo de computo** automaticamente.

### 4.3 matplotlib.pyplot (plt)

`matplotlib.pyplot` (importada como `plt`) es la libreria estandar de Python para crear graficos y visualizaciones.

Funciones comunes usadas en el practico:

| Funcion                  | Que hace                                   |
|--------------------------|---------------------------------------------|
| `plt.scatter(x, y)`     | Grafico de puntos                           |
| `plt.plot(x, y)`        | Grafico de linea                            |
| `plt.xlabel("texto")`   | Etiqueta eje x                              |
| `plt.ylabel("texto")`   | Etiqueta eje y                              |
| `plt.title("texto")`    | Titulo del grafico                          |
| `plt.figure(figsize=(9,6))` | Crear figura con tamano personalizado   |
| `plt.show()`            | Mostrar el grafico en pantalla              |

### 4.4 Optimizador Adam

**Adam** (Adaptive Moment Estimation) es un **optimizador**, el algoritmo que decide *como* actualizar los pesos usando los gradientes.

El optimizador mas basico es el descenso del gradiente clasico:

```
w = w - lr * gradiente
```

Adam es una version mas inteligente que adapta el learning rate para cada peso individualmente. Combina dos ideas:

1. **Momentum:** tiene "memoria" de la direccion en la que venia actualizando (como una bola rodando cuesta abajo, no cambia de direccion bruscamente)
2. **Adaptativo:** si un peso tiene gradientes consistentemente grandes, le reduce el paso; si tiene gradientes pequenos, le aumenta el paso

En el codigo del practico:

```python
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Crea el optimizador
# ...
optimizer.zero_grad()   # Limpia gradientes del paso anterior
loss_value.backward()   # Calcula los gradientes nuevos
optimizer.step()        # Aplica Adam para actualizar los pesos
```

Para el practico no necesitas entender la matematica interna de Adam. Solo saber que **toma los gradientes calculados por `.backward()` y actualiza los pesos del modelo**.

---

## 5. Explicacion Detallada del Codigo del Practico

### 5.1 Graficar una Funcion de Activacion

```python
import matplotlib.pyplot as plt
import numpy

X = []
Fx = []

for x in numpy.arange(-10, 10, 0.1):
    X.append(x)
    x = torch.Tensor([x])
    result = activation_function(x)
    Fx.append(float(result))

plt.scatter(X, Fx)
plt.show()
```

**Paso a paso:**
- `X` guarda los valores de entrada (eje x del grafico)
- `Fx` guarda los resultados de aplicar la funcion de activacion (eje y)
- `numpy.arange(-10, 10, 0.1)` genera: -10.0, -9.9, -9.8, ..., 9.8, 9.9
- En cada iteracion:
  - `X.append(x)` guarda el valor original para el eje x
  - `x = torch.Tensor([x])` convierte el numero a tensor de PyTorch (necesario para que `activation_function` lo acepte)
  - `result = activation_function(x)` aplica la funcion (ReLU, Sigmoid, etc.) al tensor
  - `Fx.append(float(result))` convierte el tensor resultado de vuelta a float y lo guarda
- `plt.scatter(X, Fx)` dibuja los ~200 puntos como grafico de dispersion

### 5.2 Calcular y Graficar el Gradiente de una Funcion de Activacion

```python
from torch.autograd import Variable

x = torch.tensor([2.0], requires_grad=True)
result = activation_function(x)
result.backward()
print(x.grad)  # El gradiente (derivada) evaluado en x=2
```

**Paso a paso:**
1. Se crea el tensor con `requires_grad=True` - PyTorch empieza a registrar toda operacion
2. `result = activation_function(x)` - Forward pass (ej: ReLU(2.0) = 2.0)
3. `result.backward()` - PyTorch recorre el grafo de computo en reversa y calcula: "cuanto cambia `result` si cambio `x`?" (la derivada)
4. `x.grad` contiene el valor del gradiente

**Para graficar la derivada en un rango completo:**

```python
X = []
gradiente = []

for x in numpy.arange(-10, 10, 0.1):
    X.append(x)
    x = Variable(torch.Tensor([x]), requires_grad=True)
    result = activation_function(x)
    result.backward()
    gradiente.append(float(x.grad))

plt.scatter(X, gradiente)
plt.show()
```

Es el mismo patron pero en vez de guardar `result`, guarda `x.grad` (el gradiente).

**Para ReLU el grafico de la derivada es un escalon:**
- x < 0: gradiente = 0 (ReLU es plana, no hay cambio)
- x > 0: gradiente = 1 (ReLU es recta con pendiente 1)

### 5.3 Funciones Auxiliares: plot_variable y fit

```python
def plot_variable(grads, varname):
    plt.figure(figsize=(9, 6))
    ax = plt.axes()
    ax.set_xlabel('Training step')
    ax.set_ylabel(varname)
    ax.plot(grads)
    ax.update_datalim(list(zip(range(len(grads)), grads)))
    ax.autoscale()
```

Simplemente grafica una lista de valores a lo largo del entrenamiento:
- Eje x: paso de entrenamiento (0, 1, 2, ...)
- Eje y: el valor que le pases (gradiente o loss)

```python
def fit(model):
    mse_lossfunc = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    x = torch.Tensor([0.5, 0.5])
    y = torch.Tensor([1])
    gradients = list()
    loss_seq = list()
    for i in range(200):
        p = model(x)                    # 1. Forward: predice con entrada [0.5, 0.5]
        loss_value = mse_lossfunc(p, y) # 2. Calcula el error (prediccion vs 1)
        optimizer.zero_grad()           # 3. Limpia gradientes del paso anterior
        loss_value.backward()           # 4. Backward: calcula gradientes
        optimizer.step()                # 5. Actualiza pesos usando Adam
        gradients.append(model.layer_0.weight.grad[0,:])
        loss_seq.append(loss_value.item())
    gradients = torch.stack(gradients)
    loss_seq = torch.Tensor(loss_seq)
    plot_variable(gradients[:,0], 'Gradient')
    plot_variable(loss_seq, 'Loss')
```

Es el **loop de entrenamiento completo**. Entrena el modelo 200 pasos con un unico ejemplo trivial: entrada `[0.5, 0.5]`, salida esperada `1`.

Despues de cada paso guarda:
- `gradients` - el gradiente de la **primera capa** (`layer_0`), para ver como llega la senal de error hasta el inicio de la red
- `loss_seq` - el valor del error, para ver si el modelo esta aprendiendo

**Por que se monitorea `layer_0`?** Porque es la capa mas alejada de la salida. Si hay vanishing gradient, es la primera capa donde se nota: los gradientes llegan tan debilitados que los pesos casi no se actualizan.

Al final genera **2 graficos:**
1. **Gradient vs Training step** - Muestra si los gradientes son estables, se desvanecen o explotan
2. **Loss vs Training step** - Muestra si el modelo converge (loss baja) o no

### 5.4 La Red Neuronal DeepNN

```python
class DeepNN(nn.Module):
    def __init__(self, weights_initial_value):
        super(DeepNN, self).__init__()
        self.layer_0 = torch.nn.Linear(2, 2)
        # ... capas 1 a 8: Linear(2, 2)
        self.layer_9 = torch.nn.Linear(2, 1)
        torch.nn.init.constant_(self.layer_0.weight.data, weights_initial_value)
        # ... lo mismo para todas las capas

    def forward(self, x):
        x = self.layer_0(x)
        # ... pasa por capas 1 a 8
        x = self.layer_9(x)
        return x
```

Es una red de **10 capas sin funcion de activacion**, disenada para demostrar los problemas de inicializacion. Tiene 3 partes:

**`class DeepNN(nn.Module)`** - Hereda de `nn.Module`, la clase base de PyTorch para todos los modelos. Le da al modelo la capacidad de rastrear parametros y calcular gradientes.

**`__init__`** (constructor) - Crea las 10 capas:
- `nn.Linear(2, 2)` es una capa **sin activacion** que hace: `salida = W * x + b`, donde W es una matriz de pesos 2x2 y b es el bias
- `layer_0` a `layer_8`: 2 entradas, 2 salidas (matrices 2x2)
- `layer_9`: 2 entradas, 1 salida (capa final)
- Luego inicializa TODOS los pesos al mismo valor constante con `init.constant_`

**`forward`** - La propagacion hacia adelante. Los datos pasan por las 10 capas en secuencia, sin ninguna funcion de activacion entre ellas. Esto es intencional: aisla el efecto de los pesos.

**La arquitectura completa:**
```
entrada [2] -> capa0 [2] -> capa1 [2] -> ... -> capa8 [2] -> capa9 [1] -> salida
```

**Creacion e inicializacion:**
```python
model = DeepNN(0.1)
print(model.layer_0.weight.data)
# tensor([[0.1000, 0.1000],
#         [0.1000, 0.1000]])
```

La matriz es 2x2 porque `layer_0 = nn.Linear(2, 2)` (2 entradas, 2 neuronas). Cada valor es 0.1 como se pidio:

```
         entrada_0  entrada_1
neurona_0 [0.1000,   0.1000]
neurona_1 [0.1000,   0.1000]
```

Cuando la entrada `[0.5, 0.5]` pasa por esta capa:
```
neurona_0: 0.1*0.5 + 0.1*0.5 + bias = ~0.1
neurona_1: 0.1*0.5 + 0.1*0.5 + bias = ~0.1
```

Y esa salida `[~0.1, ~0.1]` entra a la siguiente capa, que tambien multiplica por 0.1, dando valores aun mas chicos. Asi 10 capas seguidas: la senal se achica exponencialmente.

### 5.5 Red con Xavier Glorot (Actividad 3.2)

Solo se cambia `init.constant_` por `init.xavier_uniform_` y se elimina el parametro del constructor:

```python
class DeepNNXavier(nn.Module):
    def __init__(self):
        super(DeepNNXavier, self).__init__()
        self.layer_0 = torch.nn.Linear(2, 2)
        self.layer_1 = torch.nn.Linear(2, 2)
        self.layer_2 = torch.nn.Linear(2, 2)
        self.layer_3 = torch.nn.Linear(2, 2)
        self.layer_4 = torch.nn.Linear(2, 2)
        self.layer_5 = torch.nn.Linear(2, 2)
        self.layer_6 = torch.nn.Linear(2, 2)
        self.layer_7 = torch.nn.Linear(2, 2)
        self.layer_8 = torch.nn.Linear(2, 2)
        self.layer_9 = torch.nn.Linear(2, 1)
        torch.nn.init.xavier_uniform_(self.layer_0.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_1.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_2.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_3.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_4.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_5.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_6.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_7.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_8.weight.data)
        torch.nn.init.xavier_uniform_(self.layer_9.weight.data)

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        return x

model = DeepNNXavier()
fit(model)
```

---

## 6. Resultados Experimentales

### 6.1 Pesos = 0.1 (Vanishing Gradient)

**Grafico de Gradient:**
- Valores del orden de **1e-7** (0.00000015), practicamente cero
- La escala del eje y muestra `1e-7`, lo que significa que todos los valores se multiplican por 0.0000001
- Los gradientes llegan tan debilitados a `layer_0` que no logran mover los pesos

**Grafico de Loss:**
- Baja de ~0.363 a ~0.315 en 200 pasos
- Parece que aprende pero es **extremadamente lento**
- La caida es casi una linea recta: avanza a paso constante y minimo
- Adam logra algo de progreso porque adapta el learning rate internamente, pero no puede compensar del todo el vanishing gradient

### 6.2 Pesos = 1.5 (Exploding Gradient)

**Grafico de Gradient:**
- Valores del orden de **1e8** (280,000,000), enormes
- Comparado con vanishing (1e-7), hay una diferencia de **15 ordenes de magnitud**

**Grafico de Loss:**
- Empieza en **810,000,000** (8.1e8). El objetivo es predecir `1`, asi que una loss de 800 millones significa que la prediccion esta absurdamente lejos
- Baja lentamente gracias a Adam, pero despues de 200 pasos sigue en ~600 millones

### 6.3 Pesos con Xavier Glorot (Estable)

**Grafico de Gradient:**
- Valores en el rango **~0.038 a 0.044**, un rango razonable y estable
- Ni demasiado chicos (vanishing) ni demasiado grandes (exploding)

**Grafico de Loss:**
- Baja de **1.85 a 1.45** de forma consistente
- El modelo esta aprendiendo efectivamente
- No hay oscilaciones violentas ni estancamiento

### 6.4 Tabla Comparativa de los 3 Experimentos

| Metrica    | Pesos = 0.1        | Pesos = 1.5        | Xavier Glorot   |
|------------|--------------------|--------------------|-----------------|
| Gradientes | ~1e-7 (casi 0)    | ~1e8 (enormes)     | ~0.04 (estable) |
| Loss       | 0.36 (casi plana)  | ~8e8 (enorme)      | 1.85 -> 1.45    |
| Fenomeno   | Vanishing gradient | Exploding gradient | Estable         |
| Aprende?   | Casi nada          | No (inestable)     | Si, efectivamente |

---

## 7. Guia de Actividades y Respuestas

### Actividad 2.1 - Graficos de la Sigmoide

**Que se pide:**
1. Graficar la funcion Sigmoid en el rango [-10, 10]
2. Graficar la derivada de la Sigmoid en el mismo rango

**Codigo para grafico de la funcion:**
```python
activation_function = torch.nn.Sigmoid()

X = []
Fx = []

for x in numpy.arange(-10, 10, 0.1):
    X.append(x)
    x = torch.Tensor([x])
    result = activation_function(x)
    Fx.append(float(result))

plt.scatter(X, Fx)
plt.title('Funcion Sigmoid')
plt.show()
```

**Codigo para grafico de la derivada:**
```python
X = []
gradiente = []

for x in numpy.arange(-10, 10, 0.1):
    X.append(x)
    x = Variable(torch.Tensor([x]), requires_grad=True)
    result = activation_function(x)
    result.backward()
    gradiente.append(float(x.grad))

plt.scatter(X, gradiente)
plt.title('Derivada Sigmoid')
plt.show()
```

**Que se espera ver:**
- Funcion: curva en forma de "S" que va de 0 a 1
- Derivada: campana con pico en x=0 donde el gradiente maximo es ~0.25, cayendo a ~0 en los extremos

### Actividad 2.2 - Comparacion ReLU vs Sigmoid

**Pregunta:** Entre un modelo con ReLU y otro con Sigmoid, cual aprenderia mas rapido?

**Respuesta:** **ReLU aprende mas rapido** porque:
- Su gradiente es **1** (constante) para toda entrada positiva, lo que significa actualizaciones de pesos fuertes y consistentes
- Sigmoid tiene gradiente maximo de **~0.25** (en x=0) y para valores alejados de 0 el gradiente se acerca a 0 (saturacion)
- Recordando la regla de actualizacion: `w_nuevo = w_viejo - eta * gradiente`. Con Sigmoid el gradiente es como maximo 0.25, con ReLU es 1, asi que ReLU ajusta los pesos al menos **4x mas rapido**
- En redes profundas, los gradientes se multiplican capa por capa (regla de la cadena). Si cada capa aporta un factor de 0.25 (Sigmoid), despues de pocas capas el gradiente se desvanece (vanishing gradient). Con ReLU el factor es 1 y el gradiente se mantiene estable

### Actividad 3.1 - Identificar Fenomenos

**Pesos < 1:** **Vanishing Gradient** (desvanecimiento del gradiente). Los pesos menores a 1 se multiplican capa tras capa y el valor se reduce exponencialmente. En los graficos los gradientes eran del orden de 1e-7, practicamente cero, y la loss apenas bajaba.

**Pesos > 1:** **Exploding Gradient** (explosion del gradiente). Los pesos mayores a 1 se multiplican capa tras capa y el valor crece exponencialmente. En los graficos los gradientes eran del orden de 1e8 (cientos de millones) y la loss estaba en 800 millones.

### Actividad 3.2 - Xavier Glorot

**Que efecto se observa:**
1. Los gradientes se mantienen en un rango razonable (~0.04), ni se desvanecen (como con pesos 0.1 donde eran 1e-7) ni explotan (como con pesos 1.5 donde eran 1e8)
2. La loss desciende de forma consistente (de 1.85 a 1.45), indicando que el modelo efectivamente esta aprendiendo
3. El entrenamiento es estable, sin oscilaciones violentas ni estancamiento

**A que se deben los efectos:**
Xavier Glorot inicializa los pesos con varianza `2 / (fan_in + fan_out)`, lo que asegura que la **varianza de la senal se mantenga igual entre la entrada y la salida de cada capa**.
- **No hay vanishing** porque los pesos no son sistematicamente menores a 1, asi que la senal no se atenua al multiplicarse capa tras capa
- **No hay exploding** porque los pesos no son sistematicamente mayores a 1, asi que la senal no se amplifica exponencialmente
- Al mantener la varianza constante en ambas direcciones (forward y backward), los gradientes llegan con magnitudes utiles a todas las capas, permitiendo que los pesos se actualicen de forma efectiva
