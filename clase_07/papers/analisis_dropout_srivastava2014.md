# Analisis del Paper: Dropout — A Simple Way to Prevent Neural Networks from Overfitting

**Autores**: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
**Institucion**: Department of Computer Science, University of Toronto
**Publicado en**: Journal of Machine Learning Research 15 (2014) 1929-1958
**Editor**: Yoshua Bengio

> PDF descargado en: [papers/srivastava2014_dropout.pdf](srivastava2014_dropout.pdf)

---

## Datos Clave del Paper

| Aspecto | Detalle |
|---|---|
| **Ano** | 2014 (enviado Nov 2013, publicado Jun 2014) |
| **Citas** | Uno de los papers mas citados en Deep Learning (>50,000 citas) |
| **Autores notables** | Geoffrey Hinton (padre del Deep Learning), Ilya Sutskever (co-fundador OpenAI), Alex Krizhevsky (creador de AlexNet) |
| **Idea central** | Apagar neuronas al azar durante entrenamiento previene overfitting |
| **Impacto** | Se volvio tecnica estandar en practicamente toda red neuronal profunda |

---

## 1. Problema que Resuelve

Las redes neuronales profundas con muchos parametros son modelos muy expresivos, pero con datos de entrenamiento limitados tienden a **memorizar** los datos (overfitting). Muchas de las relaciones complejas que aprenden son producto del **ruido en los datos**, no de patrones reales.

Metodos existentes en 2014:
- Early stopping (parar cuando el error de validacion sube)
- Regularizacion L1 y L2 (penalizar pesos grandes)
- Weight sharing

**Limitacion**: Con computacion ilimitada, lo ideal seria promediar las predicciones de todos los posibles modelos ponderados por su probabilidad posterior (inferencia Bayesiana). Pero esto es computacionalmente imposible para redes grandes.

**Solucion propuesta**: Dropout — una aproximacion eficiente que combina exponencialmente muchas redes "delgadas" (thinned networks) con parametros compartidos.

---

## 2. Motivacion Biologica (Seccion 2 del paper)

Los autores presentan una motivacion fascinante inspirada en la **reproduccion sexual en la evolucion**:

```text
REPRODUCCION ASEXUAL:
  → Copia los genes del padre con pequenas mutaciones
  → Optimiza "fitness individual" del conjunto de genes
  → Los genes se "co-adaptan" (dependen unos de otros)

REPRODUCCION SEXUAL:
  → Mezcla genes de DOS padres al azar
  → Rompe las co-adaptaciones complejas
  → Un gen tiene que ser util POR SI MISMO, sin depender
    de un compañero especifico
  → Resultado: genes mas robustos e independientes
```

**Analogia directa con Dropout**:
- Sin Dropout: Las neuronas se "co-adaptan" — desarrollan dependencias complejas entre ellas que funcionan en entrenamiento pero no generalizan
- Con Dropout: Cada neurona debe aprender a ser util independientemente, porque no puede depender de que sus "compañeras" esten presentes

**La analogia de las conspiraciones** (textual del paper):
> "Diez conspiraciones de cinco personas cada una probablemente causan mas estragos que una gran conspiracion que requiere que cincuenta personas desempeñen su papel correctamente."

Las co-adaptaciones complejas (conspiraciones grandes) funcionan bien en datos de entrenamiento pero son fragiles ante datos nuevos. Multiples patrones simples e independientes (conspiraciones pequenas) son mas robustos.

---

## 3. Modelo Formal (Seccion 4)

### 3.1. Red estandar (sin Dropout)

Para una red con L capas ocultas, la operacion feed-forward en la capa l+1 es:

```text
z_i^(l+1) = w_i^(l+1) * y^l + b_i^(l+1)    (combinacion lineal)
y_i^(l+1) = f(z_i^(l+1))                     (funcion de activacion)

donde:
  y^(l) = salidas de la capa l
  w_i^(l+1) = pesos de la neurona i en la capa l+1
  b_i^(l+1) = bias
  f = funcion de activacion (e.g., sigmoid, ReLU)
```

### 3.2. Red con Dropout

Se agrega un paso extra — una **mascara Bernoulli**:

```text
r_j^(l)  ~ Bernoulli(p)              ← vector de 0s y 1s aleatorios
ỹ^(l)    = r^(l) * y^(l)             ← apagar neuronas (producto elemento a elemento)
z_i^(l+1) = w_i^(l+1) * ỹ^l + b_i^(l+1)   ← usar salidas "thinned"
y_i^(l+1) = f(z_i^(l+1))
```

**Paso a paso con ejemplo**:

```text
Capa l tiene 5 neuronas.    p = 0.6 (prob. de mantener)
Salidas: y^(l) = [2.1, 0.8, 1.5, 3.0, 0.4]

1) Generar mascara Bernoulli con p=0.6:
   r^(l) = [1, 0, 1, 1, 0]     ← cada valor es 1 con prob 0.6

2) Aplicar mascara:
   ỹ^(l) = [2.1, 0.8, 1.5, 3.0, 0.4] * [1, 0, 1, 1, 0]
   ỹ^(l) = [2.1, 0.0, 1.5, 3.0, 0.0]
                 ^^^              ^^^
                 apagadas

3) Pasar ỹ^(l) como entrada a la capa l+1
   (las neuronas apagadas no contribuyen)
```

### 3.3. En test time (inferencia)

Los pesos se escalan multiplicandolos por p:

```text
W_test^(l) = p * W^(l)

Intuicion: Si durante entrenamiento cada neurona estaba presente
solo el 60% del tiempo (p=0.6), en test la mantenemos siempre
pero reducimos su peso al 60%.

Esto hace que el valor ESPERADO de la salida sea igual
en entrenamiento y en test.
```

### 3.4. Interpretacion como ensamble

```text
Una red con n neuronas y Dropout puede verse como:

  2^n posibles "sub-redes" (cada neurona esta o no esta)
  ^^^^
  Con 1000 neuronas → 2^1000 sub-redes posibles

Todas comparten pesos (weight sharing).
En cada iteracion de entrenamiento, se samplea UNA sub-red.
En test, el escalado por p aproxima el PROMEDIO de todas.

  Total de parametros: O(n^2)  (no 2^n)
  → Costo computacional razonable
```

---

## 4. Entrenamiento de Redes con Dropout (Seccion 5)

### 4.1. Backpropagation con Dropout

Identico al SGD estandar, con una diferencia:

```text
Para cada minibatch:
  1. Samplear una sub-red (generar mascaras Bernoulli)
  2. Forward pass solo por la sub-red
  3. Backpropagation solo por la sub-red
  4. Actualizar pesos

Los parametros que no participaron en la sub-red
reciben gradiente = 0 (no se actualizan en esa iteracion).
```

### 4.2. Max-norm Regularization (complemento importante)

Los autores encontraron que Dropout funciona mejor combinado con **max-norm regularization**:

```text
Restriccion: ||w||_2 <= c   para cada neurona

Si los pesos de una neurona crecen mas alla de c,
se proyectan sobre la esfera de radio c.

Valores tipicos: c = 3 a 4

Por que ayuda?
  - Dropout introduce ruido → los pesos pueden "explotar"
  - Max-norm los mantiene acotados
  - Permite usar learning rates MAS ALTOS
  - El ruido de Dropout + learning rate alto permite
    EXPLORAR mas el espacio de pesos
```

### 4.3. Pretraining con Dropout

- Se puede aplicar Dropout al hacer fine-tuning de redes preentrenadas (RBMs, autoencoders)
- Los pesos del pretraining se escalan por 1/p antes de aplicar Dropout
- Se debe usar un learning rate MAS BAJO para no destruir la informacion preentrenada

---

## 5. Resultados Experimentales (Seccion 6)

### 5.1. Datasets utilizados

| Dataset | Dominio | Dimensiones | Train | Test |
|---|---|---|---|---|
| MNIST | Vision (digitos) | 784 (28x28 gris) | 60K | 10K |
| SVHN | Vision (numeros calle) | 3072 (32x32 color) | 600K | 26K |
| CIFAR-10/100 | Vision (objetos) | 3072 (32x32 color) | 60K | 10K |
| ImageNet | Vision (1000 clases) | 65536 (256x256 color) | 1.2M | 150K |
| TIMIT | Habla | 2520 (120-dim, 21 frames) | 1.1M frames | 58K frames |
| Reuters-RCV1 | Texto | 2000 | 200K | 200K |
| Alt. Splicing | Genetica | 1014 | 2932 | 733 |

### 5.2. Resultados en MNIST

| Metodo | Error % |
|---|---|
| Red estandar (sin dropout) | 1.60 |
| Dropout NN (Logistic) | 1.35 |
| Dropout NN (ReLU) | 1.25 |
| Dropout + max-norm (ReLU, 1024 units) | 1.06 |
| Dropout + max-norm (ReLU, 8192 units) | **0.95** |
| DBM + dropout finetuning | **0.79** |

**Observaciones clave**:
- Solo agregar Dropout reduce el error de 1.60% a 1.35% (mejora del 15%)
- Dropout + ReLU es mejor que Dropout + Logistic (1.25 vs 1.35)
- La red con 8192 unidades y Dropout (65M parametros en 60K datos) **no overfittea** — algo impensable sin Dropout
- Combinar pretraining + Dropout da el mejor resultado (0.79%)

### 5.3. Resultados en SVHN (Street View House Numbers)

| Metodo | Error % |
|---|---|
| Conv Net + max-pooling (sin dropout) | 3.95 |
| + dropout en fully connected | 3.02 |
| + dropout en TODAS las capas | 2.55 |
| + maxout (Goodfellow et al.) | **2.47** |
| Rendimiento humano | 2.0 |

**Hallazgo importante**: Agregar Dropout a las capas convolucionales (no solo las FC) da una mejora adicional significativa (3.02% → 2.55%). Esto fue inesperado porque las capas convolucionales tienen menos parametros.

**Valores de p por capa**: p = (0.9, 0.75, 0.75, 0.5, 0.5, 0.5) desde la entrada hasta las capas FC. Las capas mas cercanas a la entrada retienen mas neuronas.

### 5.4. Resultados en CIFAR-10 y CIFAR-100

| Metodo | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Conv Net (sin dropout) | 15.60 | 43.48 |
| + dropout solo en FC | 14.32 | 41.26 |
| + dropout en todas las capas | **12.61** | **37.20** |

Reduccion del error del 14% en CIFAR-10 y 14.5% en CIFAR-100.

### 5.5. Resultados en ImageNet (ILSVRC-2012)

| Modelo | Top-1 (val) | Top-5 (val) | Top-5 (test) |
|---|---|---|---|
| Metodos tradicionales (SIFT, etc.) | - | - | ~26% |
| Conv Net + dropout (Krizhevsky/AlexNet) | 40.7 | 18.2 | - |
| Avg de 5 Conv Nets + dropout | 38.1 | 16.4 | **16.4** |

**Este fue el modelo que gano ILSVRC-2012** (AlexNet), un momento watershed en Deep Learning. La diferencia con metodos clasicos fue abrumadora (~26% → 16.4% en top-5 error).

### 5.6. Resultados en Speech (TIMIT)

| Metodo | Phone Error Rate % |
|---|---|
| NN 6 capas (sin dropout) | 23.4 |
| Dropout NN 6 capas | 21.8 |
| DBN-pretrained 4 capas + dropout | **19.7** |

### 5.7. Resultados en Texto (Reuters-RCV1)

- Sin dropout: 31.05% error
- Con dropout: 29.62% error
- Mejora mas modesta que en vision/habla, probablemente porque el dataset es grande (200K ejemplos) y el overfitting no es tan severo

### 5.8. Comparacion con Redes Bayesianas

En el dataset de Alternative Splicing (genetica, datos escasos):

| Metodo | Code Quality (bits) |
|---|---|
| Neural Network (early stopping) | 440 |
| SVM + PCA | 487 |
| Neural Network + Dropout | 567 |
| Bayesian Neural Network | **623** |

Dropout pierde contra redes Bayesianas (que son la referencia teorica ideal), pero la diferencia es sorprendentemente pequena, y Dropout es mucho mas facil y rapido de entrenar.

### 5.9. Comparacion con Otros Regularizadores (MNIST)

| Metodo | Error % |
|---|---|
| L2 | 1.62 |
| L2 + L1 | 1.60 |
| L2 + KL-sparsity | 1.55 |
| Max-norm | 1.35 |
| Dropout + L2 | 1.25 |
| **Dropout + Max-norm** | **1.05** |

Dropout supera a todos los otros regularizadores. La combinacion Dropout + Max-norm es la mejor.

---

## 6. Analisis de Propiedades (Seccion 7)

### 6.1. Efecto en la calidad de features

El paper muestra visualmente las features aprendidas por autoencoders en MNIST:

```text
SIN Dropout:
  - Las features son "ruidosas" y poco interpretables
  - Cada neurona por si sola no detecta nada claro
  - Las neuronas se co-adaptan para reconstruir la imagen
  - Los features parecen "estatica de TV"

CON Dropout (p=0.5):
  - Las features son NITIDAS y significativas
  - Cada neurona detecta bordes, trazos o puntos especificos
  - Las neuronas son independientemente utiles
  - Se ven claramente patrones tipo "detector de bordes"
```

**Conclusion**: Dropout rompe las co-adaptaciones y fuerza a cada neurona a aprender algo util por si misma.

### 6.2. Efecto en la sparsity (dispersion)

```text
Sin Dropout:
  - Activacion media de las neuronas ≈ 2.0
  - Histograma de activaciones: muchas neuronas con activacion alta
  - Red "densa" — todas las neuronas estan activas

Con Dropout:
  - Activacion media ≈ 0.7  (reduccion del 65%)
  - Histograma: pico pronunciado en activaciones cercanas a 0
  - Red "sparse" — pocas neuronas muy activas, la mayoria apagadas
```

Dropout automaticamente induce **sparsity** sin necesidad de un regularizador explicito. Esto es deseable porque las representaciones sparse son mas interpretables y generalizan mejor.

### 6.3. Efecto de la tasa de Dropout (p)

Los autores variaron p de 0.0 a 1.0 en MNIST:

```text
  Error %
  3.5│ *
     │  *
  3.0│
     │    *
  2.5│
     │
  2.0│
     │
  1.5│           *
     │        *     *   *   *
  1.0│              *  * * *
     │                        *
  0.5│
     └──────────────────────────
     0.0  0.2  0.4  0.6  0.8  1.0
                    p (prob. de retener)

  - p muy bajo (0.1-0.2): Demasiadas neuronas apagadas → underfitting
  - p = 0.4 a 0.8: Zona optima, error minimo y plano
  - p = 1.0: Sin dropout, error mayor
  - SWEET SPOT: p ≈ 0.5 para capas ocultas
```

### 6.4. Efecto del tamano del dataset

```text
  Error %
  30│  *
    │
  25│     *
    │  * (sin dropout, siempre peor)
  20│        *
    │
  15│     *
    │           *
  10│        *
    │              *
   5│           *     *     *
    │              *     *     *  (con dropout)
   0│
    └──────────────────────────────
     100  500  1K   5K   10K  50K
              Tamano del dataset

  - Con 100-500 datos: Dropout NO AYUDA (muy pocos datos,
    el modelo memoriza con el ruido incluido)
  - Con 1K-50K datos: Dropout MEJORA significativamente
  - A medida que crece el dataset, la ventaja se reduce
    (menos overfitting natural)
  - Hay un "sweet spot" en datasets medianos
```

### 6.5. Monte-Carlo vs Weight Scaling

```text
Se puede hacer inferencia de dos formas:

1. Weight Scaling (metodo propuesto):
   → Multiplicar pesos por p
   → Rapido: una sola pasada

2. Monte-Carlo Model Averaging:
   → Hacer K forward passes CON dropout activado
   → Promediar las K predicciones
   → Mas costoso pero mas "correcto"

Resultado: Con K ≈ 50, Monte-Carlo iguala a Weight Scaling.
Weight Scaling es una aproximacion excelente y mucho mas rapida.
```

---

## 7. Variante: Dropout Gaussiano Multiplicativo (Seccion 10)

El Dropout clasico (Bernoulli) multiplica activaciones por 0 o 1. Los autores proponen una alternativa:

```text
Bernoulli Dropout:
  h_i * r   donde r ~ Bernoulli(p)
  r es 0 o 1 → la neurona esta completamente ON u OFF

Gaussian Dropout:
  h_i * r   donde r ~ N(1, σ²)     con σ² = (1-p)/p
  r es un valor continuo cercano a 1
  → Perturba la neurona en vez de apagarla

Ambos tienen el mismo valor esperado E[r] = 1
y la misma varianza Var[r] = (1-p)/p
```

| Dataset | Bernoulli | Gaussiano |
|---|---|---|
| MNIST | 1.08 +/- 0.04 | **0.95 +/- 0.04** |
| CIFAR-10 | 12.6 +/- 0.1 | **12.5 +/- 0.1** |

El Dropout Gaussiano funciona ligeramente mejor en algunos casos.

---

## 8. Guia Practica (Appendix A del paper)

Los autores incluyen un apendice con recomendaciones practicas:

### 8.1. Tamano de red

```text
Regla: Si la red optima SIN dropout tiene n neuronas por capa,
la red CON dropout debe tener al menos n/p neuronas.

Ejemplo:
  Red sin dropout optima: 512 neuronas/capa
  Con p=0.5: necesitas al menos 512/0.5 = 1024 neuronas/capa

Razon: En promedio solo p*n neuronas estan activas con Dropout.
Para mantener la misma capacidad efectiva, hay que compensar
con mas neuronas.
```

### 8.2. Learning rate y momentum

```text
Con Dropout los gradientes son MAS RUIDOSOS
(cada iteracion entrena una sub-red diferente)

→ Learning rate: Usar 10-100x mas alto que sin dropout
→ Momentum: Usar 0.95-0.99 (vs 0.9 sin dropout)

El learning rate alto + ruido de Dropout permite EXPLORAR
mas el espacio de pesos y encontrar mejores minimos.
```

### 8.3. Max-norm regularization

```text
||w||_2 <= c  para cada neurona
Valores tipicos: c = 3 a 4

Evita que los pesos "exploten" por el learning rate alto.
Mejora consistentemente los resultados.
```

### 8.4. Dropout rate por tipo de capa

```text
Capas de entrada:
  p = 0.8 (retener 80%, apagar 20%)
  Mas conservador — no destruir demasiada informacion

Capas ocultas:
  p = 0.5 (retener 50%, apagar 50%)
  Valor por defecto que funciona bien en general

Capas convolucionales:
  p = 0.75 (retener 75%)
  Menos agresivo porque tienen menos parametros

→ El valor optimo de p depende del tamano de la red
  p bajo + red grande = p alto + red pequena
  (lo que importa es p*n, el numero efectivo de neuronas activas)
```

---

## 9. Conexion Matematica: Dropout como Regularizacion (Seccion 9)

### Para regresion lineal

Los autores demuestran que para el caso simple de regresion lineal, **Dropout es equivalente a una forma especial de regularizacion L2 (Ridge)**:

```text
Sin dropout: minimizar ||y - Xw||²

Con dropout marginalizado:
  minimizar ||y - Xw̃||² + (1-p)/p * ||Γw̃||²

donde:
  w̃ = p*w
  Γ = diag(sqrt(diag(X^T * X)))

→ Es Ridge regression, pero el costo de regularizacion
  de cada peso w_i se ESCALA por la desviacion estandar
  de la feature i correspondiente.

→ Features con alta varianza → mas regularizadas
→ Features con baja varianza → menos regularizadas
→ Dropout es un regularizador ADAPTATIVO
```

### Para redes profundas

No existe una forma cerrada tan limpia, pero la intuicion se mantiene:
- Dropout actua como regularizacion adaptativa
- La "fuerza" del regularizador depende de (1-p)/p
  - p cercano a 1 → poca regularizacion
  - p cercano a 0 → mucha regularizacion

---

## 10. Conclusiones del Paper

1. **Dropout es una tecnica general** que funciona en vision, habla, texto y biologia computacional
2. **State-of-the-art** en SVHN, ImageNet, CIFAR-100 y MNIST al momento de publicacion
3. **La idea central**: tomar un modelo grande que overfittea facilmente, y samplear repetidamente sub-modelos para entrenarlos
4. **Trade-off**: El entrenamiento toma **2-3x mas tiempo** porque los gradientes son ruidosos (cada iteracion entrena una arquitectura diferente)
5. **Equivalencia matematica**: Para regresion lineal, Dropout es una forma adaptativa de L2

### Limitaciones mencionadas

- Entrenamiento mas lento (2-3x)
- No ayuda mucho con datasets muy pequenos (< 500 ejemplos)
- No ayuda mucho con datasets muy grandes (donde overfitting no es problema)
- La marginalizacion exacta para redes profundas es intratable

---

## 11. Impacto Historico y Legado

```text
2012: AlexNet gana ImageNet con Dropout
      → Momento decisivo que catalizo la era del Deep Learning

2014: Este paper formaliza y analiza Dropout
      → Se convierte en tecnica ESTANDAR

2016+: Variantes de Dropout surgen:
  - DropConnect (Wan et al., 2013): apaga PESOS en vez de neuronas
  - Spatial Dropout: apaga canales completos en CNNs
  - Variational Dropout: aprende p por neurona
  - DropBlock (Ghiasi et al., 2018): apaga regiones contiguas
  - MC Dropout (Gal & Ghahramani, 2016): usa Dropout en inferencia
    para estimar incertidumbre (Bayesian Deep Learning)

Hoy (2026):
  - Dropout sigue usandose en practicamente toda red neuronal
  - PyTorch lo incluye como nn.Dropout (version Bernoulli)
    y nn.AlphaDropout (version para redes SELU)
  - Los Transformers modernos usan Dropout (tipicamente p=0.1)
```

---

## 12. Resumen en Una Pagina

```text
PROBLEMA:  Redes profundas con muchos parametros memorizan los datos
SOLUCION:  Apagar neuronas al azar durante entrenamiento

COMO:
  Entrenamiento: Cada neurona se apaga con probabilidad p
  Inferencia:    Todas activas, pesos multiplicados por p
  (PyTorch usa "inverted dropout": escala en entrenamiento)

POR QUE FUNCIONA:
  1. Ensamble implicito de 2^n sub-redes
  2. Rompe co-adaptaciones entre neuronas
  3. Cada neurona aprende features utiles independientemente
  4. Induce sparsity automaticamente
  5. Equivalente a regularizacion L2 adaptativa (caso lineal)

CONFIGURACION RECOMENDADA:
  - Capas de entrada: p = 0.8
  - Capas ocultas FC:  p = 0.5
  - Capas conv:        p = 0.75
  - Red mas grande que sin dropout (n/p neuronas)
  - Learning rate 10-100x mayor + momentum 0.95-0.99
  - Combinar con max-norm regularization (c = 3-4)

RESULTADOS:
  Mejora en TODOS los datasets probados
  - Vision:  hasta 25% reduccion de error
  - Habla:   mejora de 23.4% a 19.7% en TIMIT
  - Texto:   mejora modesta (~5%)
  - Genetica: cercano a redes Bayesianas

LIMITACION: 2-3x mas lento en entrenamiento
```
