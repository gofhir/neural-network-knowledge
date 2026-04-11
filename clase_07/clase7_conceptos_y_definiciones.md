# Clase 7 — Conceptos y Definiciones: Guia de Estudio

Resumen de todos los conceptos explorados en el taller practico sobre normalizaciones, redes neuronales y procesamiento de datos.

---

## 1. Normalizaciones: BatchNorm y LayerNorm

### 1.1. Que problema resuelven

En una red profunda, las activaciones de cada capa cambian de escala constantemente a medida que los pesos se actualizan. Esto se llama **Internal Covariate Shift**.

```text
Red profunda, iteracion 1:
  Capa 1 produce: [0.5, 1.2, 0.8]     ← valores "pequenos"
  Capa 2 se adapta a recibir valores entre 0 y 1.5

Red profunda, iteracion 100:
  Capa 1 produce: [15.3, 42.7, 28.1]  ← ahora valores "grandes"!
  Capa 2 esta confundida: esperaba valores pequenos

  → La capa 2 gasta tiempo adaptandose a la nueva escala
    en vez de aprender patrones utiles
```

**La solucion**: despues de cada capa, normalizar las activaciones para que SIEMPRE tengan media ≈ 0 y varianza ≈ 1.

### 1.2. Que es normalizar (Z-score)

```text
         x - μ
x_norm = ─────
           σ

Ejemplo:
  Valores: [2.0, 8.0, 4.0, 6.0]
  Media μ = 5.0
  Std σ = 2.24
  Normalizado: [-1.34, +1.34, -0.45, +0.45]  ← media=0, var=1
```

### 1.3. Gamma y Beta: no perder expresividad

Despues de normalizar, la red aplica dos parametros **aprendibles**:

```text
y = γ * x_norm + β

  γ (gamma) = cuanto estirar o comprimir (escala)
  β (beta)  = cuanto desplazar (centro)
```

- Se inicializan en γ=1, β=0 (no cambian nada al inicio)
- El optimizador los ajusta durante el entrenamiento
- Son una "valvula de escape": si normalizar no ayuda en una capa, la red puede deshacerlo aprendiendo γ=σ_original, β=μ_original

**Nombres en cada framework**:

| Concepto | PyTorch | TensorFlow | JAX/Flax |
|---|---|---|---|
| gamma | `.weight` | `.gamma` | `params['scale']` |
| beta | `.bias` | `.beta` | `params['bias']` |
| desactivar | `affine=False` | `scale=False, center=False` | `use_scale=False, use_bias=False` |

### 1.4. BatchNorm: normalizar por COLUMNA

Calcula media y varianza de cada **feature** a traves de todas las muestras del batch.

```text
         Feature 1   Feature 2   Feature 3
Muestra 1:   2.0        10.0        0.5
Muestra 2:   8.0        20.0        1.5
Muestra 3:   4.0        30.0        0.8
Muestra 4:   6.0        40.0        1.2
              ↕           ↕           ↕
         normaliza    normaliza    normaliza
         esta col.    esta col.    esta col.
```

**En inferencia** usa running mean/var acumuladas durante el entrenamiento (no las del batch actual). Por eso `model.eval()` es critico.

### 1.5. LayerNorm: normalizar por FILA

Calcula media y varianza de cada **muestra** a traves de todos sus features.

```text
         Feature 1   Feature 2   Feature 3
Muestra 1:   2.0        10.0        0.5    ← normaliza esta fila
Muestra 2:   8.0        20.0        1.5    ← normaliza esta fila
Muestra 3:   4.0        30.0        0.8    ← normaliza esta fila
```

- No depende del batch (funciona con batch=1)
- Comportamiento identico en entrenamiento e inferencia
- No tiene running stats

### 1.6. Cuando usar cual

| | BatchNorm | LayerNorm |
|---|---|---|
| **Ideal para** | Imagenes (CNNs) | Texto (Transformers) |
| **Normaliza** | Por canal/feature (columna) | Por muestra (fila) |
| **Depende del batch** | Si (necesita batches grandes) | No (funciona con batch=1) |
| **train vs eval** | Distintos (running stats) | Iguales |
| **En PyTorch** | `nn.BatchNorm1d`, `nn.BatchNorm2d` | `nn.LayerNorm` |

### 1.7. Running Stats (solo BatchNorm)

BatchNorm necesita media y varianza para normalizar. Pero en inferencia puede llegar **una sola muestra**, y no se puede calcular media de 1 valor.

```text
ENTRENAMIENTO (muchas muestras por batch):
  Batch 1:  media_feature_1 = 2.1  → running_mean se actualiza
  Batch 2:  media_feature_1 = 1.8  → running_mean se actualiza
  Batch 3:  media_feature_1 = 2.3  → running_mean se actualiza
  ...
  → Va acumulando un PROMEDIO MOVIL de las estadisticas

INFERENCIA (puede ser 1 sola muestra):
  Feature 1: [3.5]  ← un solo numero, no puedo calcular media
  → Usa la running_mean acumulada (-1.50) para normalizar
```

LayerNorm NO tiene running stats porque no las necesita: calcula media/varianza sobre los features de la muestra, no sobre el batch. Siempre tiene suficientes datos.

```text
BatchNorm en inferencia:           LayerNorm en inferencia:
  Feature 1: [3.5] ← 1 dato         Features: [3.5, -1.2, 0.8, ...]
  ¿Media? No se puede calcular      ← 784 datos en UNA muestra
  → Necesita running stats          ¿Media? 0.43 ✓
                                     → Calcula en el momento, sin problemas
```

Las running stats se guardan en el archivo `.pth` junto con los pesos, pero **no son parametros entrenables** (no se ajustan con backpropagation, solo se acumulan).

### 1.8. Donde poner la normalizacion en la red

```text
Orden comun: Capa Lineal → Normalizacion → Activacion (ReLU) → Dropout

No es una regla fija, hay variantes:
  Clasico:          Linear → BatchNorm → ReLU → Dropout
  Pre-Norm (GPT):   LayerNorm → Attention → Dropout
  Post-Norm (BERT): Attention → LayerNorm → Dropout
  Simple:           Linear → ReLU (sin norm ni dropout)

NUNCA en la ultima capa (queremos la prediccion sin normalizar).
```

**Por que la capa de salida va sola** (sin norm, ReLU ni Dropout):

```text
La ultima capa produce la RESPUESTA FINAL de la red.

  Si aplicas BatchNorm: distorsiona las magnitudes
  Si aplicas ReLU:      elimina valores negativos (pierde informacion)
  Si aplicas Dropout:   apaga clases al azar (prediccion inconsistente)

  Capas intermedias: "Estoy procesando" → norm, ReLU, Dropout ayudan
  Capa final:        "Esta es mi respuesta" → no tocar
```

---

## 2. Dropout: Regularizacion

### 2.1. Que es

En cada iteracion de entrenamiento, cada neurona tiene una probabilidad **p** de ser "apagada" (su valor se pone en 0). Fuerza a que TODAS las neuronas aprendan a ser utiles, no solo algunas.

### 2.2. Tipos de Dropout

```text
Dropout(p=0.5):    apaga NEURONAS individuales → para capas lineales
Dropout2d(p=0.25): apaga CANALES completos     → para capas convolucionales
```

### 2.3. Inverted Dropout (como funciona en la practica)

Si en entrenamiento solo esta activo el 50% de neuronas, en inferencia (100% activas) la capa produce el doble de informacion. PyTorch resuelve esto automaticamente con **inverted dropout**:

```text
ENTRENAMIENTO (p=0.5):
  Valores:         [2.0, 4.0, 1.0, 3.0]
  Mascara:         [1,   0,   1,   0  ]   ← apaga al azar
  Aplicar mascara: [2.0, 0.0, 1.0, 0.0]
  Dividir por 0.5: [4.0, 0.0, 2.0, 0.0]   ← escala AQUI para compensar

INFERENCIA:
  Valores:         [2.0, 4.0, 1.0, 3.0]   ← no hace NADA, ya esta escalado
```

Esto lo hace `nn.Dropout` internamente. No requiere running stats ni recordar nada.

### 2.4. Train vs Eval

- **Entrenamiento**: Dropout activo (apaga neuronas)
- **Inferencia**: Dropout desactivado (usa todas las neuronas, ya escaladas por inverted dropout)
- PyTorch: `model.train()` / `model.eval()`
- TensorFlow: `training=True` / `training=False`

### 2.5. Dropout, BatchNorm e Inverted Dropout son independientes

Son 3 mecanismos que resuelven 3 problemas distintos. No dependen entre si:

```text
Dropout:          "Apago neuronas para no memorizar" (regularizacion)
                  En inferencia se apaga. No necesita recordar nada.

BatchNorm:        "Normalizo activaciones para estabilizar" (estabilidad)
                  En inferencia usa running stats porque no tiene batch.

Inverted Dropout: "Escalo durante entrenamiento para que las magnitudes
                   no cambien en inferencia" (compensacion)
                  Lo hace PyTorch automaticamente dentro de nn.Dropout.
```

---

## 3. Frameworks: PyTorch vs TensorFlow vs JAX

### 3.1. Comparacion general

| | PyTorch | TensorFlow/Keras | JAX/Flax |
|---|---|---|---|
| **Estilo** | Orientado a objetos | Orientado a objetos | Funcional puro |
| **Construir red** | Clase con `forward()` | `Sequential([...])` | Clase con `@nn.compact` |
| **Estado** | Interno al modelo | Interno a la capa | Externo (dict de params) |
| **Train/Eval** | `model.train()`/`eval()` | `training=True/False` | `use_running_average=True/False` |
| **Quien lo usa** | Industria + academia | Industria (Google) | Investigacion (DeepMind) |

### 3.2. Crear un tensor

```python
# Los 3 crean lo mismo: un array de numeros
# PyTorch
x = torch.tensor([1.0, 2.0, 3.0])     # mutable

# TensorFlow
x = tf.constant([1.0, 2.0, 3.0])      # inmutable
x = tf.Variable([1.0, 2.0, 3.0])      # mutable

# JAX
x = jnp.array([1.0, 2.0, 3.0])        # inmutable (estilo funcional)
```

### 3.3. Flax

JAX solo es un motor de calculo numerico (como NumPy con autograd). **Flax** (`flax.linen`) es la libreria que le agrega las capas de redes neuronales (`nn.Dense`, `nn.BatchNorm`, etc.). Es la relacion entre TensorFlow y Keras.

---

## 4. Tipos de datos de entrada

### 4.1. Principio fundamental

**La red nunca ve fotos, palabras ni sonidos. Solo ve arrays de numeros con distinta forma (shape).**

### 4.2. Datos tabulares (CSV)

```text
Ya son numeros. Son lo mas directo.

  Edad=25, Peso=70, Presion=120 → [25.0, 70.0, 120.0]
  Shape: (batch, features)
```

### 4.3. Imagenes

```text
Una foto es una grilla de pixeles. Cada pixel ya es un numero (0-255).
La camara hizo la conversion de luz a numeros.

  Foto B/N:    (batch, 1, 28, 28)     ← 1 canal
  Foto color:  (batch, 3, 224, 224)   ← 3 canales (R, G, B)

  Preproceso: dividir por 255 → valores [0.0, 1.0]
  
  TF pone canales al FINAL:  (batch, 28, 28, 1)
  PyTorch los pone PRIMERO:  (batch, 1, 28, 28)
```

### 4.4. Texto

```text
Las palabras NO son numeros. Conversion en 2 pasos:

  Paso 1 - Tokenizar: palabra → ID numerico
    "el gato come" → [2, 5, 8]

  Paso 2 - Embedding: ID → vector de numeros
    2 → [0.23, -0.51, 0.89, ...]   (vector de embedding_dim numeros)
    5 → [1.12,  0.34, 0.02, ...]

  Shape final: (batch, tokens, embedding_dim)
```

### 4.5. Audio

```text
El sonido es una onda. El microfono la muestrea miles de veces por segundo.

  Paso 1 - Muestreo: onda continua → array de numeros
    16,000 muestras por segundo → [0.02, -0.01, 0.05, ...]

  Paso 2 - Espectrograma (STFT): array 1D → "imagen" 2D
    Eje X = tiempo, Eje Y = frecuencia, Color = intensidad
    Se puede procesar con una CNN como si fuera una foto

  Shape final: (batch, 1, frecuencias, tiempo)
```

---

## 5. Vocabulario de Texto

### 5.1. Tokenizacion

Convertir palabras a IDs numericos usando un vocabulario:

```text
vocab = {"el": 2, "gato": 5, "come": 8, ...}
"el gato come" → [2, 5, 8]
```

### 5.2. Tokens especiales

| Token | Significado | Para que sirve |
|---|---|---|
| `<pad>` (ID=0) | Relleno (padding) | Igualar el largo de todas las frases en un tensor |
| `<unk>` (ID=1) | Desconocido (unknown) | Reemplazar palabras que no estan en el vocabulario |

```text
<pad> es necesario porque los tensores deben ser RECTANGULARES:

  "me encanta"                → [27, 13,  0,  0,  0]
  "la pelicula es muy buena"  → [24, 33, 14, 30,  7]
                                              ^^^^
                                              padding

  No es por la normalizacion, es por como funcionan los tensores.
  La red IGNORA los tokens de padding al calcular.
```

### 5.3. Embedding

Tabla de vectores que convierte cada ID en un vector denso. Las dimensiones del embedding (32, 768, 12288) son **arbitrarias** y no tienen nombre ni significado predefinido.

```text
  embedding_dim = 32     nuestro ejemplo (40 palabras, tarea simple)
  embedding_dim = 768    BERT (30,000 tokens)
  embedding_dim = 12288  GPT-3 (50,000 tokens)
```

- Los vectores se **aprenden** durante el entrenamiento
- Despues de entrenar, palabras similares tienen vectores cercanos
- Mas palabras y mas complejidad → necesitas mas dimensiones
- Los numeros clasicos (64, 128, 256, 768) son por eficiencia de GPU, no por matematica

---

## 6. MNIST

Dataset clasico de 70,000 imagenes de digitos escritos a mano (0-9), de 28x28 pixeles cada una. Es el "Hola Mundo" del machine learning.

- 60,000 imagenes de entrenamiento
- 10,000 imagenes de prueba
- Viene incluido en PyTorch y TensorFlow
- Se descarga con una linea de codigo

---

## 7. Batch (tamano del lote)

### 7.1. Que es

Cuantas muestras procesas **al mismo tiempo** en una iteracion.

```text
batch=1:   procesa 1 imagen por iteracion (lento)
batch=64:  procesa 64 imagenes juntas (rapido, en paralelo)
```

### 7.2. Relacion con BatchNorm

```text
Batch grande (128): estadisticas estables → BatchNorm funciona bien
Batch pequeno (1):  la "media" es un solo valor → BatchNorm no sirve
```

Por eso GPT (que genera token por token, batch=1 en inferencia) usa LayerNorm.

---

## 8. Redes Convolucionales (CNN)

### 8.1. El problema de Flatten + Linear

```text
Flatten destruye la estructura 2D de la imagen.
La red no sabe que un pixel esta al lado de otro.
```

### 8.2. Que es una convolucion

Un **filtro** (tipicamente 3x3) que recorre la imagen detectando patrones locales:

```text
Filtro de bordes verticales:    Filtro de bordes horizontales:
  [-1  0  1]                      [-1 -1 -1]
  [-1  0  1]                      [ 0  0  0]
  [-1  0  1]                      [ 1  1  1]
```

### 8.3. MaxPool

Reduce el tamano de la imagen a la mitad, tomando el valor maximo de cada zona 2x2:

```text
[1 3 | 5 2]        [4 | 6]
[2 4 | 0 6]   →    [7 | 9]
[7 1 | 3 8]
[0 2 | 1 9]

28x28 → 14x14 → 7x7
```

### 8.4. Arquitectura CNN tipica

```text
Conv2d → BatchNorm2d → ReLU → MaxPool → Dropout2d  (entiende la imagen)
Conv2d → BatchNorm2d → ReLU → MaxPool → Dropout2d  (patrones mas complejos)
Flatten                                              (aplana al final)
Linear → BatchNorm1d → ReLU → Dropout               (clasifica)
Linear                                               (salida)
```

### 8.5. Comparacion

```text
                    Red Plana         CNN
MNIST (28x28):      ~98%              ~99%    ← diferencia chica
ImageNet (224x224):  ~30%              ~96%    ← diferencia ENORME
```

### 8.6. Tipos de normalizacion en CNN

```text
BatchNorm2d: para capas convolucionales (normaliza por canal)
BatchNorm1d: para capas lineales despues de Flatten
Dropout2d:   apaga canales completos (no neuronas individuales)
Dropout:     apaga neuronas individuales (capas lineales)
```

---

## 9. Entrenamiento de una red neuronal

### 9.1. Los 7 pasos

1. **DATOS**: cargar + normalizar + armar batches
2. **RED**: definir capas (Linear, BatchNorm, ReLU, Dropout)
3. **LOSS**: funcion que mide el error (CrossEntropyLoss para clasificacion)
4. **OPTIMIZADOR**: algoritmo que ajusta los pesos (Adam, lr=0.001)
5. **ENTRENAR**: repetir por cada batch: forward → loss → backward → update
6. **EVALUAR**: probar con datos nunca vistos
7. **GUARDAR**: `torch.save()` para usar despues

### 9.2. El loop de entrenamiento

```text
Para cada epoca:
  Para cada batch:
    1. FORWARD:  prediccion = model(imagenes)
    2. LOSS:     error = loss_fn(prediccion, etiqueta_real)
    3. BACKWARD: error.backward()        ← calcula gradientes
    4. UPDATE:   optimizer.step()        ← ajusta pesos
```

### 9.3. Funcion de perdida (Loss Function)

Un calculo matematico que compara la prediccion con la respuesta correcta y devuelve **un solo numero**: que tan equivocado estas.

```text
Loss alto (2.5):  la red predijo mal   → hay que ajustar mucho
Loss bajo (0.01): la red predijo bien  → casi no hay que ajustar
```

Cada funcion de perdida sirve para un tipo de tarea distinto:

| Tarea | Funcion | Salida de la red | Ejemplo |
|---|---|---|---|
| Clasificar en N clases | `nn.CrossEntropyLoss` | N numeros (uno por clase) | digito 0-9, tipo de animal |
| Si o no (binario) | `nn.BCELoss` | 1 numero (0 a 1) | ¿es spam?, ¿tiene cancer? |
| Predecir un valor | `nn.MSELoss` | 1 numero (cualquiera) | precio de casa, temperatura |
| Predecir valor (robusto) | `nn.L1Loss` | 1 numero (cualquiera) | ventas, distancia |

**CrossEntropyLoss** (la que usamos en los ejemplos):

```text
Internamente hace 2 cosas:
  1. Softmax: convierte números crudos en probabilidades que suman 1
     [0.1, 0.2, 0.1, 8.5, ...] → [0.01, 0.01, 0.01, 0.95, ...]
  
  2. -log(probabilidad de la clase correcta):
     Si la clase correcta es 3 y la red le dio 0.95:
     loss = -log(0.95) = 0.05  ← bajo, acertó!
     
     Si la red le dio 0.01:
     loss = -log(0.01) = 4.6   ← alto, falló!
```

**MSELoss** (Mean Squared Error, para regresion):

```text
  Prediccion: 245,000
  Real:       250,000
  Loss: (245,000 - 250,000)² = 25,000,000,000
  → Penaliza MAS los errores grandes (por el cuadrado)
```

**L1Loss** (para regresion, menos sensible a outliers):

```text
  Prediccion: 245,000
  Real:       250,000
  Loss: |245,000 - 250,000| = 5,000
  → Penaliza errores proporcionalmente (sin cuadrado)
```

Todas vienen incluidas en PyTorch (`torch.nn`). No hay que implementar la matematica:

```python
loss_fn = nn.CrossEntropyLoss()                      # crear
loss = loss_fn(prediccion_de_la_red, etiqueta_real)   # usar
loss.backward()                                       # backpropagation
```

### 9.4. Epoca

Una **epoca** = pasar por TODAS las imagenes de entrenamiento una vez. Se entrena multiples epocas.

```text
Pocas epocas (underfitting):   la red no aprendio lo suficiente
Punto justo:                   la red aprendio y generaliza bien
Demasiadas epocas (overfitting): la red memoriza, test accuracy baja
```

En la practica se usa **early stopping**: si el test accuracy no mejora por N epocas seguidas, se para y se guarda el mejor modelo.

Rangos tipicos:

```text
MNIST (simple):          5-20 epocas
Clasificar texto:        10-50 epocas
ImageNet (fotos reales): 90-300 epocas
GPT-3:                   ~1 epoca (dataset tan grande que no necesita repetir)
```

### 9.4. Datos de entrenamiento vs prueba

```text
ENTRENAMIENTO (train): la red aprende con estos datos
  → Como los ejercicios de practica para una prueba

PRUEBA (test): evaluamos si la red aprendio de verdad
  → Como la prueba final. La red NUNCA los vio antes
```

### 9.5. model.train() vs model.eval()

```text
model.train():  Dropout ACTIVO, BatchNorm usa stats del batch actual
model.eval():   Dropout DESACTIVADO, BatchNorm usa running stats

SIEMPRE llamar model.eval() antes de predecir.
```

---

## 10. Imagenes vs Texto: resumen de diferencias

| | Imagenes | Texto |
|---|---|---|
| **Entrada** | Pixeles [0-255] | Palabras |
| **Preproceso** | Dividir por 255 | Tokenizar + Embedding |
| **Normalizacion** | BatchNorm (por canal) | LayerNorm (por token) |
| **Red** | CNN (Conv2d) | Linear o Transformer |
| **Por que esa normalizacion** | Todas las imagenes mismo tamano, canales con significado fijo | Frases de largo variable, cada token independiente |
| **Batch en inferencia** | Puede ser grande | Frecuentemente 1 |

---

## 11. Estructura de los ejemplos

Todos los ejemplos se ejecutan con Docker:

```bash
# Construir la imagen (solo una vez, o despues de cambios)
docker build -t clase7-pytorch .

# Correr un ejemplo
docker run --rm clase7-pytorch python -u ejemplos/pytorch/01_normalizacion_manual_pytorch.py

# Correr ejemplos con graficos (montar carpeta output)
docker run --rm -v $(pwd)/output:/app/output clase7-pytorch \
  python -u ejemplos/transformaciones/14_entrenar_mnist_paso_a_paso.py
```

### Estructura de archivos

```text
clase7_ejemplos/
├── pytorch/          ← 10 ejemplos en PyTorch
├── tensorflow/       ← 10 ejemplos en TensorFlow
├── jax/              ← 10 ejemplos en JAX/Flax
└── transformaciones/ ← ejemplos avanzados con graficos
    ├── 11_transformar_imagen.py
    ├── 12_transformar_texto.py
    ├── 13_transformar_audio.py
    ├── 14_entrenar_mnist_paso_a_paso.py     (red plana)
    ├── 15_entrenar_mnist_cnn.py             (red convolucional)
    └── 16_entrenar_texto_paso_a_paso.py     (LayerNorm para texto)
```
