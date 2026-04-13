---
title: "Resolución del Laboratorio"
weight: 50
math: true
---

Resolución completa de las 4 actividades practicas del laboratorio.

{{< notebook-viewer src="/notebooks-html/lab10.html" >}}

---

## Actividad Practica 1 — Parametros de SGD

### Enunciado

Revise la documentacion de PyTorch para `torch.optim.SGD` y comente los diferentes parametros.

### Respuesta

La firma de SGD es:

```python
torch.optim.SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

**`params`** (iterable, requerido): los parametros del modelo — sus pesos $W$ y biases $b$. Son los valores que se actualizan en cada iteracion siguiendo la regla $w_i^{new} = w_i^{old} - \eta \frac{\partial L}{\partial w_i}$. Se pasan con `model.parameters()`.

**`lr`** (float, requerido): el learning rate $\eta$. Como vimos en clase, es el factor que multiplica al gradiente y determina el tamano del paso en cada actualizacion. Es el dilema central de la optimizacion:

- Muy alto: los pasos sobredimensionados saltan sobre el minimo y el modelo **diverge** (el loss sube en vez de bajar)
- Muy bajo: converge muy lentamente, puede tardar demasiadas epocas
- Valor adecuado: baja rapido y se estabiliza en un buen minimo

Valores tipicos: 0.001 a 0.1. En SGD basico el learning rate es **constante** durante todo el entrenamiento, a diferencia de metodos adaptativos como Adam.

**`momentum`** (float, default=0): factor de momentum $\rho$. Como vimos en clase, el gradiente en SGD depende solo del batch actual, lo que genera variabilidad — diferentes batches producen gradientes distintos. Momentum resuelve esto incorporando la **historia** del movimiento, analogo al momentum fisico:

$$v_t = \rho \cdot v_{t-1} + \nabla L(\theta)$$
$$\theta = \theta - \eta \cdot v_t$$

Con $\rho = 0.9$ (valor tipico), el 90% de la velocidad anterior se conserva. Esto suaviza el camino hacia el minimo y ayuda a escapar de saddle points y minimos locales — dos de los tres problemas principales de SGD vistos en clase.

**`dampening`** (float, default=0): factor de amortiguamiento que reduce la contribucion del gradiente actual al termino de momentum. Rara vez se modifica en la practica.

**`weight_decay`** (float, default=0): regularizacion L2. Corresponde al termino $\alpha \Omega(W)$ de la funcion objetivo vista en clase:

$$L(W) = \mathcal{L}(f(x), y; W) + \alpha \Omega(W)$$

Agrega una penalizacion proporcional a la magnitud de los pesos, lo que incentiva pesos pequenos y previene overfitting. Valores tipicos: 1e-4 a 1e-2.

**`nesterov`** (bool, default=False): activa momentum de Nesterov. Como vimos en clase, la diferencia con momentum clasico es que primero se aplica el momentum para proyectar la posicion futura, y **luego** se calcula el gradiente en esa posicion proyectada. Es una "mirada al futuro" que generalmente produce una convergencia mas rapida y estable. Requiere `momentum > 0`.

**`maximize`** (bool, default=False): si es True, maximiza la funcion objetivo en vez de minimizarla. Es decir, se mueve **en la misma direccion** del gradiente en vez de en la contraria. Util en contextos como entrenamiento adversarial o reinforcement learning, pero en clasificacion estandar siempre se deja en False.

**`foreach`** (bool, opcional): controla si se usa la implementacion vectorizada del optimizador. Cuando es None (default), PyTorch usa automaticamente la version `foreach` en CUDA porque es significativamente mas rapida — actualiza todos los parametros en una sola operacion del kernel en vez de iterar uno por uno.

**`differentiable`** (bool, default=False): permite que autograd calcule gradientes **a traves** del paso de optimizacion. Normalmente `optimizer.step()` se ejecuta dentro de `torch.no_grad()` porque no necesitamos gradientes del propio paso de actualizacion. Se activa en casos avanzados como meta-learning donde se necesita diferenciar el proceso de entrenamiento completo.

**`fused`** (bool, opcional): usa una implementacion fusionada (solo CUDA) que combina multiples operaciones en un solo kernel de GPU. Soporta float64, float32, float16 y bfloat16. Es una optimizacion de rendimiento — no cambia el resultado matematico, solo lo computa mas rapido.

---

## Actividad Practica 2 — Learning rates altos

### Enunciado

1. Explique el grafico obtenido con learning rates `[0.0001, 0.001, 0.01, 0.1]`. Por que hay tanta diferencia?
2. Entrene el modelo con learning rates `[1, 10, 100]`.
3. Comente los resultados comparando ambos graficos.

### Respuesta

#### Parte 1 — Grafico con lr bajos `[0.0001, 0.001, 0.01, 0.1]`

El grafico muestra cuatro curvas de loss a lo largo de 20 epocas, con comportamientos radicalmente distintos segun el learning rate:

- **lr=0.0001**: baja muy lentamente. El loss apenas se mueve en 20 epocas porque los pasos son demasiado pequenos — el modelo converge, pero necesitaria muchas mas epocas para llegar al minimo.
- **lr=0.001**: curva descendente clara y estable. Es el punto dulce para este modelo: baja rapido sin oscilar. Converge bien dentro de las 20 epocas.
- **lr=0.01**: baja mas rapido que 0.001 al inicio pero puede mostrar algo mas de ruido. Sigue siendo un lr valido — llega a valores de loss similares o algo mejores, pero con menos estabilidad.
- **lr=0.1**: baja muy rapido inicialmente pero oscila o incluso sube en algunas epocas. El paso es tan grande que "salta" sobre el minimo y le cuesta estabilizarse.

La diferencia es tan grande porque el learning rate $\eta$ escala directamente la magnitud de cada actualizacion:

$$w^{new} = w^{old} - \eta \cdot \nabla L(w)$$

Un factor de 10x en $\eta$ significa pasos 10 veces mas grandes — la diferencia entre converger suavemente, oscilar, o diverger completamente.

#### Parte 2 — Entrenamiento con lr altos `[1, 10, 100]`

```python
import torch.optim as optim
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
epochs = 20

fig = plt.figure(figsize=(10, 6))

for lr in [1, 10, 100]:
    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    loss = train(net, trainloader, optimizer, criterion, lr, epochs=epochs)
    test(testloader, net)
    plt.plot(loss, label=lr, linewidth=4)

plt.xlabel("Epochs", fontsize=16)
plt.xticks(fontsize=14)
plt.ylabel("Loss", fontsize=16)
plt.yticks(fontsize=14)
plt.legend(prop={'size': 14})
plt.show()
```

**Resultados obtenidos:**

| Learning Rate | Accuracy (test) | Comportamiento del loss |
|---------------|-----------------|------------------------|
| lr = 1        | ~10%            | Colapsa a 0 (NaN / inestabilidad numerica) |
| lr = 10       | ~10%            | Colapsa a 0 (NaN) |
| lr = 100      | ~10%            | Explosion catastrofica hasta ~$1.3 \times 10^{11}$ en epoca 1, luego colapsa |

El 10% de accuracy en los tres casos corresponde exactamente a la precision de un clasificador aleatorio en CIFAR10 (10 clases, 1/10 = 10%) — el modelo no aprende absolutamente nada.

#### Parte 3 — Comparacion de ambos graficos

La comparacion entre los dos experimentos ilustra perfectamente el efecto del learning rate sobre la optimizacion:

**Con lr bajos `[0.0001 — 0.1]`:** el loss desciende de forma controlada. La diferencia entre ellos es de velocidad de convergencia y estabilidad, pero todos aprenden algo util. Es el regimen normal de entrenamiento.

**Con lr altos `[1 — 100]`:** ocurre divergencia total. Los gradientes en cada batch son, digamos, de magnitud $\sim 1$. Con lr=100, cada paso actualiza los pesos en $\sim 100$ unidades — los pesos explotan a valores gigantescos, las activaciones se saturan, los gradientes se vuelven NaN y el modelo queda inutilizable. La "inestabilidad numerica" de lr=1 y lr=10 (loss=0) no significa que el loss sea realmente cero, sino que los pesos llegaron a valores tan extremos que los calculos de punto flotante producen NaN o Inf, que PyTorch reporta como 0.

Formalmente, la condicion de convergencia de SGD exige que el learning rate sea lo suficientemente pequeno respecto a la curvatura de la funcion de loss (Lipschitz constant $L$):

$$\eta < \frac{2}{L}$$

Con learning rates de 1, 10 o 100, esta condicion se viola masivamente. Como se vio en clase, el dilema del learning rate es precisamente este: muy alto diverge, muy bajo converge lento — el arte esta en encontrar el valor adecuado (o usar metodos adaptativos como Adam que ajustan $\eta$ automaticamente).

---

## Actividad Practica 3 — Mejor lr para Adam

### Enunciado

1. Encuentre un mejor valor de learning rate para Adam (puede cambiar los betas si lo cree necesario).
2. Comente los resultados y explique la diferencia con SGD.

### Respuesta

#### Parte 1 — Busqueda del mejor learning rate para Adam

El learning rate por defecto de Adam es `lr=0.001`. Para encontrar el valor optimo, se comparan varios candidatos:

```python
import torch.optim as optim
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()
epochs = 20

fig = plt.figure(figsize=(10, 6))

for lr in [0.0001, 0.001, 0.01, 0.1]:
    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss = train(net, trainloader, optimizer, criterion, lr, epochs=epochs)
    acc = test(testloader, net)
    plt.plot(loss, label=f"Adam lr={lr}", linewidth=2)

plt.xlabel("Epochs", fontsize=16)
plt.xticks(fontsize=14)
plt.ylabel("Loss", fontsize=16)
plt.yticks(fontsize=14)
plt.legend(prop={'size': 11})
plt.title("Adam — comparacion de learning rates", fontsize=14)
plt.show()
```

**Resultados obtenidos:**

| Learning Rate | Accuracy (test) | Observacion |
|---------------|-----------------|-------------|
| lr = 0.0001   | 48%             | Convergencia lenta — pasos demasiado pequenos para 20 epocas |
| lr = 0.001    | **61%**         | **Mejor resultado** — balance optimo velocidad/estabilidad |
| lr = 0.01     | 57%             | Converge rapido pero oscila al final, accuracy inferior a 0.001 |
| lr = 0.1      | 10%             | Divergencia total — mismo comportamiento que SGD con lr alto |

**Conclusion:** `lr=0.001` es el mejor valor para Adam en este experimento (61%), consistente con las recomendaciones de Kingma & Ba (2015) en el paper original. No es necesario cambiar los betas — `beta1=0.9, beta2=0.999` ya estan bien calibrados para este tipo de problema.

#### Parte 2 — Diferencias entre Adam y SGD

La diferencia fundamental entre Adam y SGD radica en como se calcula el paso de actualizacion de cada parametro:

**SGD con momentum:**
$$v_t = \rho \cdot v_{t-1} + \nabla L(\theta)$$
$$\theta = \theta - \eta \cdot v_t$$

Todos los parametros se actualizan con el **mismo** learning rate $\eta$. Si un parametro tiene gradientes muy grandes y otro muy pequenos, ambos reciben el mismo factor — lo que es suboptimo.

**Adam:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta) \quad \text{(momento de orden 1)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) [\nabla L(\theta)]^2 \quad \text{(momento de orden 2)}$$
$$\theta = \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

Adam divide el gradiente por $\sqrt{\hat{v}_t}$ — la raiz cuadrada del promedio de los gradientes al cuadrado. Esto tiene un efecto clave: **cada parametro tiene su propio learning rate efectivo** que se adapta automaticamente:

- Parametros con gradientes historicamente grandes → learning rate efectivo pequeno (no sobreactualiza)
- Parametros con gradientes historicamente pequenos → learning rate efectivo grande (no se estanca)

**Resultados reales comparando los tres optimizadores (20 epocas, CIFAR10):**

| Optimizador | Accuracy (test) |
|-------------|-----------------|
| Adagrad     | 50%             |
| SGD (momentum=0.9, lr=0.001) | 59% |
| **Adam (lr=0.001)** | **61%** |

Adam supera a SGD en 2 puntos con el mismo learning rate y las mismas epocas. La clave esta en la adaptacion por parametro: donde SGD usa el mismo $\eta$ para todos los pesos, Adam ajusta el paso de cada parametro segun su historial de gradientes. Esto hace que converja mas eficientemente en el mismo numero de epocas.

Adagrad queda por debajo (50%) porque acumula los gradientes al cuadrado de forma monotonica — el learning rate efectivo solo puede decrecer, nunca recuperarse. Con 20 epocas, los parametros mas activos terminan con un lr efectivo casi cero. Adam resuelve esto con $\beta_2$ que usa una media movil exponencial, "olvidando" gradientes muy antiguos y manteniendo el lr efectivo activo durante todo el entrenamiento.

---

## Actividad Practica 4 — Schedulers

### Enunciado

Entrenar SGD, Adam y Adagrad con dos estrategias de ajuste del learning rate durante el entrenamiento:
- **4a:** StepLR — reduce el lr por un factor cada N epocas
- **4b:** ReduceLROnPlateau — reduce el lr cuando el loss se estanca

### Respuesta

| Optimizador | Sin scheduler | StepLR (4a) | ReduceLROnPlateau (4b) |
|-------------|--------------|-------------|----------------------|
| SGD         | 59%          | 58%         | 57%                  |
| Adam        | 61%          | 57%         | 59%                  |
| Adagrad     | 50%          | 50%         | 43%                  |

**StepLR** reduce el lr por 0.1 en la epoca 10. La accuracy no mejora — con solo 20 epocas, reducir el lr a mitad del entrenamiento deja poco margen para seguir aprendiendo.

**ReduceLROnPlateau** reduce el lr cuando el loss se estanca (`patience=10`). Con 20 epocas casi no alcanza a actuar. Adagrad cae a 43% porque el scheduler reduce un lr que ya era muy bajo por su acumulacion monotonica.

**Conclusion:** los schedulers necesitan mas epocas (50-100+) para mostrar su valor. Con 20 epocas el entrenamiento es demasiado corto para beneficiarse de la reduccion del lr.
