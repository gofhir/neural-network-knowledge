# Laboratorio 7: PyTorch - Diplomado IA (Aprendizaje Profundo I)

**Profesores:** Alain Raymond, Gabriel Sepulveda, Miguel Fadic, Alvaro Soto  
**Ayudante:** Andres Villa

---

## Descripcion General

Este laboratorio es una introduccion practica a **PyTorch**, cubriendo desde los conceptos mas basicos (tensores) hasta un flujo completo de entrenamiento de una red convolucional (AlexNet) para clasificacion de imagenes de flores (102 clases). El laboratorio se divide en 7 secciones principales y finaliza con 4 actividades a resolver.

---

## 1. El Tensor: la unidad fundamental

### Que es un tensor

Un tensor es una **matriz de n-dimensiones**, la estructura de datos basica en PyTorch. Es el equivalente al `ndarray` de NumPy pero con soporte para GPU.

- **Imagenes:** tensores 4D → `(batch, canales, alto, ancho)`
- **Texto:** tensores 3D → `(batch, palabras, embedding)`
- **Video:** tensores 5D → agregan dimension temporal

### Crear tensores

```python
import torch

# Tensor aleatorio de 3 dimensiones
tensor_ejemplo = torch.randn((2, 4, 3)).float()
print(tensor_ejemplo.shape)  # torch.Size([2, 4, 3])

# Desde una lista de Python
arreglo = [[1, 2, 3], [4, 5, 6]]
tensor = torch.tensor(arreglo)
```

**Explicacion:** `torch.randn` genera valores aleatorios con distribucion normal. `.float()` convierte a float32. `torch.tensor()` convierte listas de Python a tensores.

### Operaciones basicas

```python
tensor_1 = torch.randn((10, 2, 3))
tensor_2 = torch.randn((10, 2, 3))
tensor_3 = tensor_1 * tensor_2  # Multiplicacion elemento a elemento
```

### Indexacion

```python
tensor_indexado = tensor_1[0:2, 0, 0]  # Primeros 2 elementos del batch, fila 0, columna 0
```

**Explicacion:** funciona igual que NumPy. Se pueden usar slices en cada dimension.

### Dimension batch

- La dimension batch es **siempre la primera** dimension en PyTorch.
- Permite entrenar multiples ejemplos en paralelo en GPU.
- Incluso un solo ejemplo debe tener dimension batch: `(1, C, H, W)`.
- En imagenes, el orden es: `(batch, canales, alto, ancho)` — el canal va **antes** que las dimensiones espaciales.

### Dispositivo (CPU vs GPU)

```python
tensor_nuevo = torch.randn((1, 2, 3))       # En CPU por defecto
tensor_gpu = tensor_nuevo.cuda()             # Copia a GPU
otra_forma = tensor_nuevo.to("cuda")         # Alternativa equivalente
```

**Explicacion:** para aprovechar la GPU, los tensores deben moverse explicitamente con `.cuda()` o `.to("cuda")`. Todos los tensores que interactuen entre si deben estar en el mismo dispositivo.

---

## 2. Definicion del Modelo

### Estructura de un modelo en PyTorch

Todo modelo hereda de `torch.nn.Module` e implementa dos metodos:

1. **`__init__(self)`**: define las capas y componentes de la red.
2. **`forward(self, x)`**: define como fluyen los datos a traves de las capas. Recibe un tensor y retorna un tensor.

### Elementos arquitectonicos principales (en `torch.nn`)

| Capa | Descripcion | Parametros clave |
|------|-------------|-----------------|
| `Linear` | Capa fully connected | `in_features`, `out_features` |
| `Conv2d` | Convolucion 2D | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` |
| `ReLU` | Activacion ReLU | Sin parametros |
| `Sigmoid` | Activacion Sigmoid | Sin parametros |
| `Softmax` | Convierte salida a distribucion de probabilidad | `dim` |
| `MaxPool2d` | Max Pooling 2D | `kernel_size`, `stride` |
| `Dropout` | Apaga neuronas con probabilidad p | `p` |
| `BatchNorm2d` | Normaliza por canal en el batch | `num_features` |
| `LayerNorm` | Normaliza por ejemplo | `normalized_shape` |
| `Sequential` | Agrupa modulos en secuencia | Lista de modulos |

### Ejemplos de cada componente

#### Capa Lineal

```python
from torch.nn import Linear
capa_lineal = Linear(5, 8)          # 5 entradas → 8 salidas
tensor_nuevo = capa_lineal(tensor)  # Aplica W*x + b
```

**Explicacion:** transforma un tensor de dimension 5 a dimension 8 mediante una multiplicacion matricial mas un bias.

#### Funciones de Activacion

```python
# ReLU: max(0, x) — elimina valores negativos
relu = ReLU()
resultado = relu(tensor)

# Sigmoid: mapea a [0, 1]
s = Sigmoid()
resultado = s(tensor)

# Softmax: convierte a distribucion de probabilidad (suma = 1)
soft = Softmax(dim=1)
resultado = soft(tensor)  # Cada fila suma 1
```

**Explicacion:**
- **ReLU** es la activacion mas usada en deep learning; pone en 0 los negativos.
- **Sigmoid** comprime valores al rango [0,1], util para probabilidades binarias.
- **Softmax** normaliza para que los valores sumen 1 a lo largo de `dim`, ideal para clasificacion multiclase.

#### Sequential

```python
seq = Sequential(capa_lineal, relu, soft)
resultado = seq(tensor)  # Aplica las 3 capas en secuencia
```

**Explicacion:** `Sequential` encadena modulos. La salida de uno es la entrada del siguiente.

#### Dropout

```python
drop = Dropout(p=0.1)  # 10% de probabilidad de poner en 0
resultado = drop(tensor)
```

**Explicacion:** durante entrenamiento, pone aleatoriamente un porcentaje de las neuronas en 0 para prevenir **overfitting**. Durante evaluacion no hace nada.

#### BatchNorm1d / BatchNorm2d

```python
# Para datos 1D (ej. salida de Linear)
bn1d = BatchNorm1d(5)  # 5 features
resultado = bn1d(tensor)

# Para datos 2D (imagenes)
bn2d = BatchNorm2d(3)  # 3 canales
resultado = bn2d(tensor_imagenes)
```

**Explicacion:** normaliza las activaciones por **canal** usando la media y varianza del batch actual. Estabiliza y acelera el entrenamiento. Mantiene estadisticas corrientes (`running_mean`, `running_var`) para usarlas durante evaluacion.

#### LayerNorm

```python
ln = LayerNorm(100)  # Normaliza vectores de dimension 100
resultado = ln(tensor)
```

**Explicacion:** a diferencia de BatchNorm, normaliza por **cada ejemplo individual** (no por batch). Tiene parametros aprendibles `weight` (escala) y `bias` (desplazamiento) que permiten que la red ajuste la media y varianza optimas.

---

## 3. Implementacion de AlexNet (MiAlexNet)

### Arquitectura completa

```python
class MiAlexNet(nn.Module):
    def __init__(self):
        super(MiAlexNet, self).__init__()
        # === BLOQUES CONVOLUCIONALES ===
        # Conv1: 3x224x224 → 96x55x55 → MaxPool → 96x27x27
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        # Conv2: 96x27x27 → 256x27x27 → MaxPool → 256x13x13
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        # Conv3: 256x13x13 → 384x13x13
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Conv4: 384x13x13 → 384x13x13
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Conv5: 384x13x13 → 256x13x13 → MaxPool → 256x6x6
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.flat = nn.Flatten()  # 256x6x6 = 9216

        # === BLOQUES FULLY CONNECTED ===
        self.fc6 = nn.Sequential(nn.Linear(9216, 4096), nn.ReLU())
        self.fc7 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU())
        self.fc8 = nn.Sequential(nn.Linear(4096, 102))  # 102 clases de flores

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flat(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x
```

### Explicacion paso a paso del flujo de datos

| Etapa | Entrada | Operacion | Salida |
|-------|---------|-----------|--------|
| Conv1 | `3 x 224 x 224` | Conv(11x11, stride=4, pad=2) + MaxPool(3x3, stride=2) + ReLU | `96 x 27 x 27` |
| Conv2 | `96 x 27 x 27` | Conv(5x5, stride=1, pad=2) + MaxPool(3x3, stride=2) + ReLU | `256 x 13 x 13` |
| Conv3 | `256 x 13 x 13` | Conv(3x3, stride=1, pad=1) + ReLU | `384 x 13 x 13` |
| Conv4 | `384 x 13 x 13` | Conv(3x3, stride=1, pad=1) + ReLU | `384 x 13 x 13` |
| Conv5 | `384 x 13 x 13` | Conv(3x3, stride=1, pad=1) + MaxPool(3x3, stride=2) + ReLU | `256 x 6 x 6` |
| Flatten | `256 x 6 x 6` | Aplanar | `9216` |
| FC6 | `9216` | Linear + ReLU | `4096` |
| FC7 | `4096` | Linear + ReLU | `4096` |
| FC8 | `4096` | Linear | `102` (clases) |

**Nota:** la salida final tiene 102 neuronas porque el dataset Flowers tiene 102 clases.

### Verificacion

```python
m = MiAlexNet()
# Ver todos los parametros
for n, p in m.named_parameters():
    print(n, p.shape)

# Prueba con datos ficticios
fotos_de_mentira = torch.randn(15, 3, 224, 224)
print(m(fotos_de_mentira).shape)  # torch.Size([15, 102])
```

---

## 4. Manejo de Datos (Dataset y DataLoader)

### Concepto

PyTorch maneja datos grandes mediante dos abstracciones:

- **`Dataset`**: interfaz para acceder a los datos individuales.
- **`DataLoader`**: agrupa datos del Dataset en **batches** para el modelo.

### Datasets predefinidos

```python
from torchvision.datasets import MNIST, CIFAR10

mnist_train = MNIST(root=".", train=True, download=True)
cifar = CIFAR10(root=".", train=True, download=True)

print(len(mnist_train))  # 60000
print(mnist_train[459])  # (imagen PIL, etiqueta)
```

**Explicacion:** PyTorch viene con datasets clasicos listos para usar. Cada elemento es una tupla `(dato, etiqueta)`.

### Transformaciones (PIL → Tensor)

```python
from torchvision.transforms import ToTensor
mnist_train = MNIST(root=".", train=True, download=True, transform=ToTensor())
# Ahora mnist_train[459] devuelve (tensor, etiqueta) en vez de (PIL, etiqueta)
```

**Explicacion:** `transform=ToTensor()` convierte automaticamente cada imagen PIL a un tensor de PyTorch al acceder al dato.

### Crear un Dataset personalizado

Para el dataset **Flowers** (102 clases de flores), se crea una clase que hereda de `torch.utils.data.Dataset`:

```python
class Flowers(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imagenes, self.imgs_to_class = self.armar_indices(root)

    def armar_indices(self, root):
        # Recorre carpetas (cada carpeta = una clase)
        # Construye listas de nombres de archivo y sus clases
        lista_imagenes = []
        imgs_to_class = []
        for clase in listdir(root):
            directorio = join(root, clase)
            for archivo in listdir(directorio):
                lista_imagenes.append(archivo)
                imgs_to_class.append(int(clase))
        return lista_imagenes, imgs_to_class

    def __getitem__(self, idx):
        # Carga una imagen y su clase dado un indice
        nombre_archivo = self.imagenes[idx]
        clase = self.imgs_to_class[idx]
        ruta_img = join(self.root, str(clase), nombre_archivo)
        img = Image.open(ruta_img)
        if self.transform is not None:
            img = self.transform(img)
        return img, clase

    def __len__(self):
        return len(self.imagenes)
```

**Explicacion paso a paso:**

1. **`__init__`**: recibe la ruta al directorio y las transformaciones. Llama a `armar_indices` para construir un indice de todas las imagenes.
2. **`armar_indices`**: recorre la estructura de carpetas donde cada subcarpeta es una clase (ej. `0/`, `1/`, ..., `101/`). Guarda el nombre del archivo y la clase correspondiente.
3. **`__getitem__`**: dado un indice, carga la imagen desde disco, aplica la transformacion (resize + toTensor) y devuelve `(tensor_imagen, clase)`.
4. **`__len__`**: retorna el total de imagenes.

### Uso con transformaciones

```python
transforms = Compose([Resize((224, 224)), ToTensor()])
f = Flowers('flowers_dataset/train', transform=transforms)
```

**Explicacion:** `Compose` encadena transformaciones: primero redimensiona a 224x224 (lo que espera AlexNet), luego convierte a tensor.

### DataLoader: iterar por batches

```python
from torch.utils.data import DataLoader
train_dl = DataLoader(f, batch_size=128, shuffle=True)

for n_batch, (x, target) in enumerate(train_dl):
    print(f"Batch {n_batch}: input={x.shape}, labels={target.shape}")
```

**Explicacion:**
- `batch_size=128`: agrupa de a 128 imagenes por batch.
- `shuffle=True`: mezcla los datos en cada epoca (importante para entrenamiento).
- Un recorrido completo del DataLoader = **1 epoca**.

---

## 5. Optimizacion (Entrenamiento)

### Funcion de perdida

```python
from torch.nn import CrossEntropyLoss
loss_function = CrossEntropyLoss()
loss = loss_function(output, target)
```

**Explicacion:**
- **CrossEntropyLoss** es la funcion estandar para clasificacion multiclase.
- Recibe la salida cruda del modelo (logits, sin softmax) y las etiquetas verdaderas.
- Internamente aplica softmax + log + negative log-likelihood.

### Optimizador

```python
from torch.optim import SGD, Adam
optimizer = Adam(model.parameters(), lr=0.001)
```

**Explicacion:** el optimizador gestiona la actualizacion de los pesos. Se le vinculan los parametros del modelo. **Adam** es una variante avanzada de SGD que adapta el learning rate por parametro.

### Loop de entrenamiento (1 iteracion)

```python
optimizer.zero_grad()                   # 1. Limpiar gradientes previos
output = model(input)                   # 2. Forward pass
loss = loss_function(output, target)    # 3. Calcular perdida
loss.backward()                         # 4. Backpropagation (calcula gradientes)
optimizer.step()                        # 5. Actualizar pesos
```

**Explicacion detallada de cada paso:**

1. **`zero_grad()`**: los gradientes se acumulan por defecto en PyTorch. Si no los limpiamos, se sumarian con los del batch anterior.
2. **`model(input)`**: pasa los datos a traves de todas las capas definidas en `forward()`.
3. **`loss_function(output, target)`**: compara la prediccion con la verdad y calcula un escalar de error.
4. **`loss.backward()`**: calcula automaticamente los gradientes de la perdida respecto a cada parametro (backpropagation via autograd).
5. **`optimizer.step()`**: usa los gradientes calculados para actualizar los pesos segun el algoritmo de optimizacion (Adam, SGD, etc.).

---

## 6. Evaluacion de Rendimiento

```python
output = model(input)                       # Salida: (batch_size, n_clases)
preds = output.argmax(dim=1)                # Indice de la clase con mayor valor
n_correctas = (preds == targets).sum()      # Comparar con ground truth
total = targets.shape[0]
accuracy = n_correctas / total
```

**Explicacion:** `argmax(dim=1)` toma el indice de mayor valor a lo largo de la dimension de clases. Comparar con las etiquetas verdaderas nos da un tensor booleano; `.sum()` cuenta los `True`.

---

## 7. Flujo Completo de Entrenamiento y Evaluacion

### Entrenamiento

```python
model = MiAlexNet().cuda()
transforms = Compose([Resize((224, 224)), ToTensor()])
ds_train = Flowers("flowers_dataset/train", transform=transforms)
train_dl = DataLoader(ds_train, batch_size=128, shuffle=True)
optimizer = Adam(model.parameters(), lr=0.001)
loss_function = CrossEntropyLoss()

model.train()  # Modo entrenamiento (Dropout y BatchNorm activos)
for epoch in range(1, 11):
    total_correctas = 0.0
    total_muestras = 0.0
    for x, target in train_dl:
        optimizer.zero_grad()
        x, target = x.cuda(), target.cuda()
        output = model(x)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        preds = output.argmax(dim=1)
        total_correctas += (preds == target).sum()
        total_muestras += target.shape[0]
    accuracy = total_correctas / total_muestras
    print(f"Epoca {epoch}: Accuracy={100*accuracy:.2f}%")
```

**Explicacion:**
- `model.train()`: activa comportamiento de entrenamiento (Dropout descarta neuronas, BatchNorm usa estadisticas del batch).
- El loop externo recorre epocas; el interno recorre batches.
- Se acumulan las predicciones correctas para reportar accuracy por epoca.

### Evaluacion en Test

```python
ds_test = Flowers("flowers_dataset/test", transform=transforms)
test_dl = DataLoader(ds_test, batch_size=1024)

model.eval()  # Modo evaluacion
total_correctas = 0.0
total_muestras = 0.0
for x, target in test_dl:
    with torch.no_grad():  # No calcular gradientes (ahorra memoria y tiempo)
        x, target = x.cuda(), target.cuda()
        output = model(x)
        preds = output.argmax(dim=1)
        total_correctas += (preds == target).sum()
        total_muestras += target.shape[0]
accuracy = total_correctas / total_muestras
print(f"Test Accuracy: {100*accuracy:.2f}%")
```

**Diferencias clave con entrenamiento:**
- `model.eval()`: desactiva Dropout y usa running stats en BatchNorm.
- `torch.no_grad()`: no se calculan gradientes (no hay backpropagation en evaluacion).
- No se llama a `optimizer.zero_grad()`, `loss.backward()`, ni `optimizer.step()`.

---

## 8. Guardar y Cargar el Modelo

```python
# Guardar pesos
torch.save(model.state_dict(), "pesos_modelo_entrenado.pth")

# Cargar pesos
modelo = MiAlexNet()
pesos = torch.load("pesos_modelo_entrenado.pth")
modelo.load_state_dict(pesos)
```

**Explicacion:**
- `state_dict()` contiene todos los pesos aprendidos del modelo como un diccionario.
- Se guarda solo los pesos (no la arquitectura), por lo que al cargar necesitas crear primero una instancia del modelo con la misma arquitectura.

---

## 9. Actividades a Resolver

### Actividad 1: Entrenar MiAlexNet por 10 epocas
> Entrenar el modelo base y reportar **accuracy en train y test**.

**Lo que se espera:** ejecutar el codigo de entrenamiento tal cual esta (10 epocas) y luego evaluar en test. Reportar ambos numeros.

### Actividad 2: Agregar Dropout antes de FC6 y FC7
> Modificar `MiAlexNet` agregando `nn.Dropout()` antes de las capas FC6 y FC7. Entrenar 10 epocas y comparar.

**Lo que se debe hacer:** en el `__init__` de MiAlexNet, agregar Dropout a fc6 y fc7:

```python
self.fc6 = nn.Sequential(nn.Dropout(), nn.Linear(9216, 4096), nn.ReLU())
self.fc7 = nn.Sequential(nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU())
```

**Efecto esperado:** Dropout deberia **reducir overfitting** — el accuracy en train podria bajar pero el accuracy en test podria mejorar o mantenerse.

### Actividad 3: Agregar BatchNorm2d antes de Conv3, Conv4 y Conv5
> Agregar capas de `BatchNorm2d` antes de las convoluciones 3, 4 y 5. Entrenar 10 epocas.

**Lo que se debe hacer:** agregar `nn.BatchNorm2d(num_canales_entrada)` al inicio de los bloques conv3, conv4 y conv5:

```python
self.conv3 = nn.Sequential(
    nn.BatchNorm2d(256),  # Nuevo
    nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
    nn.ReLU()
)
self.conv4 = nn.Sequential(
    nn.BatchNorm2d(384),  # Nuevo
    nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
    nn.ReLU()
)
self.conv5 = nn.Sequential(
    nn.BatchNorm2d(384),  # Nuevo
    nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.ReLU()
)
```

**Efecto esperado:** BatchNorm deberia **estabilizar y acelerar el entrenamiento**. Se espera ver que la loss converge mas rapido y el accuracy suba mas consistentemente.

### Actividad 4: Usar AlexNet preentrenado en ImageNet (Transfer Learning)
> Reemplazar `MiAlexNet()` por el modelo `alexnet(pretrained=True)` de torchvision, congelando las capas convolucionales y reentrenando solo el clasificador.

**Codigo proporcionado:**

```python
from torchvision.models import alexnet
model = alexnet(pretrained=True)
model.features.requires_grad_(False)  # Congela capas convolucionales
model.classifier = nn.Sequential(     # Redefine el clasificador
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 102),
)
```

**Explicacion:**
- `pretrained=True` carga pesos entrenados en ImageNet (1.2M imagenes, 1000 clases).
- `requires_grad_(False)` congela las capas convolucionales para no modificar sus pesos.
- Se reemplaza el clasificador original (1000 clases) por uno nuevo de 102 clases.
- Solo se entrenan las capas del clasificador.

**Efecto esperado:** el **transfer learning deberia mejorar significativamente el rendimiento**, ya que las features convolucionales preentrenadas en ImageNet son mucho mejores que las entrenadas desde cero en un dataset pequeno de flores.

---

## Resumen de Conceptos Clave

| Concepto | Para que sirve |
|----------|---------------|
| **Tensor** | Estructura de datos fundamental, similar a ndarray de NumPy |
| **nn.Module** | Clase base para definir modelos |
| **forward()** | Define el flujo de datos en la red |
| **Dataset** | Interfaz para acceder a datos individuales |
| **DataLoader** | Agrupa datos en batches para entrenamiento |
| **CrossEntropyLoss** | Funcion de perdida para clasificacion |
| **Adam/SGD** | Algoritmos de optimizacion |
| **loss.backward()** | Calcula gradientes via backpropagation |
| **optimizer.step()** | Actualiza los pesos del modelo |
| **model.train()** | Modo entrenamiento (Dropout/BN activos) |
| **model.eval()** | Modo evaluacion (Dropout/BN desactivados) |
| **torch.no_grad()** | Desactiva calculo de gradientes (para evaluacion) |
| **state_dict()** | Diccionario con todos los pesos del modelo |
| **Transfer Learning** | Reusar pesos preentrenados de otro modelo/dataset |
