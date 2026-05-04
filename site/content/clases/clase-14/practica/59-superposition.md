---
title: "59 - Superposition: por que las neuronas son polisemanticas"
weight: 590
math: true
---

## 1. Apertura: el problema de empacar muchas cosas en pocas dimensiones

Hasta ahora hablamos de "el residual stream tiene `d_model = 128` dimensiones" como si fuera obvio que cada dimension representa algo. Pero los modelos modernos suelen necesitar representar muchos mas conceptos que dimensiones tienen. ¿Como un modelo con 128 dimensiones puede representar 1000+ conceptos distintos?

La respuesta es **superposition**, formalizada por Anthropic en "Toy Models of Superposition" (Elhage et al. 2022). Cuando el numero de features que el modelo necesita representar excede su dimensionalidad, el modelo aprende a **comprimir** features en direcciones NO ortogonales del espacio. El precio: las neuronas individuales se vuelven **polisemanticas** — cada una responde a multiples conceptos no relacionados.

Este capitulo demuestra superposition con un toy model minimal: 5 features comprimidas en 2 dimensiones. Si las features fueran ortogonales, solo cabrian 2. Vamos a ver como el modelo "rompe" la ortogonalidad y que pasa cuando lo hace.

---

## 2. El toy model: 5 features → 2 dim → 5

Setup minimo del paper de Anthropic (versions reducidas):

- **Input**: vector $x \in \mathbb{R}^5$ con sparsity 70% (la mayoria de entradas son 0)
- **Encoder**: $h = x W$ con $W \in \mathbb{R}^{5 \times 2}$. Comprime a 2 dimensiones.
- **Decoder**: $\hat{x} = \text{ReLU}(h W^T)$. Tied weights — usa la misma $W$.
- **Loss**: $\text{MSE}(\hat{x}, x)$

```python
class Toy(nn.Module):
    def __init__(self, n_features=5, d_model=2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_features, d_model) * 0.1)

    def forward(self, x):
        h = x @ self.W
        recon = torch.relu(h @ self.W.T)
        return recon, h
```

La matriz aprendida $W$ tiene una row por feature (un vector 2D que representa esa feature). Si las rows son ortogonales, el modelo SOLO puede representar 2 features distintas; las otras 3 se confunden con las primeras dos. Si las rows son **angularmente distribuidas** alrededor del plano (idealmente con angulos de $360/5 = 72°$ entre vecinas), el modelo puede representar las 5 con minimo overlap.

### Por que sparsity importa

Con sparsity=0 (todas las features siempre activas), el modelo no puede comprimir 5 en 2 sin colision destructiva. Con sparsity alta (70% de las features son 0), pares de features rara vez co-ocurren — entonces el modelo puede darle direcciones similares a features no co-ocurrentes sin perder informacion.

**Sparsity hace posible la superposition.** El paper de Anthropic muestra que features con probabilidad de aparicion `p` pueden codificarse en menos dimensiones cuando `p` es chico.

---

## 3. Script

```python
"""59_superposition.py - Cap 59: superposition demo (Anthropic Toy Models 2022)."""
import math, torch
import torch.nn as nn

torch.manual_seed(1337)
N_FEATURES, D_MODEL = 5, 2
SPARSITY = 0.7

def generate_data(n_samples, n_features, sparsity):
    mask = torch.rand(n_samples, n_features) > sparsity
    values = torch.rand(n_samples, n_features)
    return values * mask.float()

class Toy(nn.Module):
    def __init__(self, n_features, d_model):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_features, d_model) * 0.1)
    def forward(self, x):
        h = x @ self.W
        return torch.relu(h @ self.W.T), h

model = Toy(N_FEATURES, D_MODEL)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for step in range(2000):
    x = generate_data(1024, N_FEATURES, SPARSITY)
    recon, _ = model(x)
    loss = ((x - recon) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

W = model.W.detach()
# Analizar angulos, normas, gram matrix
```

---

## 4. Output literal

```
Toy model de superposition: 5 features -> 2 dim -> 5
Sparsity=0.7, samples=8192

  step    0: loss=0.09457
  step  100: loss=0.04694
  step  500: loss=0.03835
  step 1000: loss=0.03773
  step 1999: loss=0.03622

Matriz aprendida W (shape (5, 2)):
[[-0.5236744  -0.25304157]
 [-0.51290053 -0.23174687]
 [-0.4533195   0.92178226]
 [-0.5204153  -0.25767472]
 [ 0.42323226 -0.9239351 ]]

Norma de cada feature vector:
  feature 0: ||W_0|| = 0.582
  feature 1: ||W_1|| = 0.563
  feature 2: ||W_2|| = 1.027
  feature 3: ||W_3|| = 0.581
  feature 4: ||W_4|| = 1.016

Angulo de cada feature vector (en grados, vs eje x):
  feature 0: angulo=-154.21°
  feature 1: angulo=-155.68°
  feature 2: angulo=+116.19°
  feature 3: angulo=-153.66°
  feature 4: angulo= -65.39°

Diferencias angulares (esperamos ~72° si distribucion uniforme):
  feature  1 (-155.68°) -> feature  0 (-154.21°): diff=1.47°
  feature  0 (-154.21°) -> feature  3 (-153.66°): diff=0.55°
  feature  3 (-153.66°) -> feature  4 ( -65.39°): diff=88.27°
  feature  4 ( -65.39°) -> feature  2 (+116.19°): diff=181.58°
  feature  2 (+116.19°) -> feature  1 (-155.68°): diff=88.13°
```

Plot ASCII (cada digito es la feature correspondiente, lineas desde el origen):

```
             2                             
              2                            
                2                          
                 2                         
                   2                       
                     2                     
                    134                    
                  133 4                    
               1133   4                    
             1133      4                   
                        4                  
                         4                 
                           4               
                            4              
                             4             
                              4
```

---

## 5. Analisis: superposition imperfecta — dos clusters

Lo que esperabamos (paper de Anthropic): 5 features distribuidas casi uniformemente alrededor del plano, con ~72° de separacion entre vecinas. Eso seria un pentagono regular en 2D — la solucion teoricamente optima.

**Lo que encontramos**: dos clusters.

- **Cluster 1**: features 0, 1, 3 — todas en el angulo ~-155°. Diferencias angulares < 1.5° entre ellas.
- **Cluster 2 (anti-pareados)**: features 2 (+116°) y 4 (-65°). Diferencia 181.6° entre ambas — son casi exactamente opuestas en el plano (anti-features).

### Por que esto es correcto y no un bug

Un pentagono regular requeriria que las 5 features tuvieran identical importancia y costos identicos de confusion entre ellas. Pero nuestro toy model NO impone eso — todas las features tienen la misma probabilidad de activarse, pero el optimizador encuentra una solucion mas eficiente:

1. **Anti-pareo**: features 2 y 4 estan opuestas (+116° y -65°, diferencia 181°). Como ReLU corta valores negativos, dos features anti-pareadas pueden coexistir SIN colision: cuando feature 2 es positiva, decoda como feature 2 con peso alto y feature 4 con peso 0 (porque ReLU). Lo mismo en reverso.

2. **Cluster colapsado**: features 0, 1, 3 estan todas en la misma direccion. Esto implica que el modelo NO LAS DISTINGUE — cuando una de las tres aparece, el decoder produce activacion en las tres por igual. Es **polisemanticidad pura**: una direccion del espacio representa 3 features distintas.

¿Por que el modelo aprende esto en vez de distribuir uniformemente? Porque el optimizador encontro un minimo local que es facil de aprender. Anthropic muestra en su paper que la solucion del pentagono regular existe pero requiere inicializaciones especificas o entrenamiento mas largo. Aqui con 2000 pasos y semilla 1337 caemos en el cluster + anti-pareo.

### Confirmacion via Gram matrix

```
W @ W^T (productos punto entre features):
  +0.338  +0.327  +0.004  +0.338  +0.012
  +0.327  +0.317  +0.019  +0.327  -0.003
  +0.004  +0.019  +1.055  -0.002  -1.044
  +0.338  +0.327  -0.002  +0.337  +0.018
  +0.012  -0.003  -1.044  +0.018  +1.033
```

- Features 0, 1, 3: productos punto entre si ~0.32-0.34 (alta similaridad — no son distinguibles).
- Features 2 y 4: producto punto -1.044 (ANTI-CORRELADAS — opuestas en el plano).
- Features 2,4 vs el cluster (0,1,3): productos ~0 (ortogonales al cluster).

La estructura es: dos features ortogonales al "cluster" + tres features que comparten una direccion. Eficiente para sparsity alta porque colisiones del cluster son raras.

---

## 6. Implicaciones para Mini-LLaMA

Mini-LLaMA tiene `d_model=128` dimensiones del residual stream. Si el modelo necesita codificar 500 features distintas (sustantivos, verbos, posicion en linea, mood del texto, identidad de speaker, etc.), las 128 dimensiones NO bastan ortogonalmente. El modelo debe usar superposition.

Consecuencia: las **neuronas son polisemanticas**. Si miras la activacion de la dimension 17 del residual stream sobre 1000 prompts distintos, vas a encontrar que se activa fuertemente en contextos heterogeneos — mayusculas, principio de oracion, entradas de Brutus, sustantivos. Cada uno es un feature distinto que comparte la dimension 17 con otros (en superposition).

Esto es por que NO podemos interpretar Mini-LLaMA mirando dimensiones individuales del residual stream. Necesitamos descomponer el espacio en features monosemanticas — features que cada una representa UN concepto. Esa descomposicion es el rol de los **Sparse Autoencoders** del cap 60.

---

## 7. Por que SAEs deshacen superposition

La idea de los Sparse Autoencoders (Bricken et al. 2023):

- **Encoder**: `Linear(d_model=128 -> d_features=512)` con ReLU + sparsity penalty L1
- **Decoder**: `Linear(d_features=512 -> d_model=128)`
- **Loss**: `MSE(reconstruction) + lambda * L1(features)`

El SAE aprende a re-representar el residual stream del modelo en un espacio MAS GRANDE pero con activacion ESPARSA. La penalizacion L1 obliga a que solo unas pocas features esten activas a la vez. Con suficiente sparsity, las features tienden a ser monosemanticas — cada una representa un concepto distinto.

La intuicion: cuando expandes a 4× mas dimensiones (128 → 512), las features ya tienen "espacio" para ser ortogonales. La sparsity penalty penaliza activaciones distribuidas (que serian polisemanticas). El optimo bajo estas condiciones es: cada feature representa una sola cosa, y solo unas pocas estan activas en cualquier input.

Cap 60 entrena este SAE sobre las activaciones del residual stream de Mini-LLaMA. Cap 61 inspecciona las features aprendidas para ver si son monosemanticas.

---

## 8. Lo honesto sobre este toy model

El experimento **valida superposition**: con `n_features > d_model` y sparsity alta, el modelo aprende representaciones no-ortogonales. Pero:

- **No produjo el pentagono regular** — el optimizador eligio una solucion alternativa con cluster colapsado. Esto es esperable y aparece en el paper original con ciertas configuraciones.
- **2D es chico**: en `d_model=128` con 500+ features, los efectos son mas sutiles y no podemos visualizarlos como angulos en el plano.
- **El experimento es pedagogico, no diagnostic**: confirma el principio pero no nos dice nada concreto sobre Mini-LLaMA.

Para diagnosticar Mini-LLaMA necesitamos el SAE — esa es la herramienta de produccion. Cap 59 establece la teoria; caps 60-61 la aplican.

---

## 9. Preguntas de verificacion

**1. ¿Por que el modelo NO aprendio el pentagono regular?**

El pentagono regular (5 features distribuidas a 72° cada una) es un minimo de la loss BAJO ciertas condiciones: importancia uniforme entre features, suficientes pasos de entrenamiento, inicializacion adecuada. En la practica, los optimizadores caen en minimos locales — y el cluster + anti-pareo es un minimo local valido. El paper de Anthropic muestra que con MAS pasos y barridos sobre sparsity, distintas configuraciones emergen, incluyendo pentagonos regulares en algunos casos. Aqui con 2000 pasos y semilla 1337 caimos en una solucion alternativa pero igualmente valida.

**2. ¿Que cambia si reduces la sparsity de 0.7 a 0.0?**

Con sparsity=0.0, todas las features se activan en cada sample. Esto rompe la posibilidad de superposition — el modelo no puede "compartir direcciones" porque las co-activaciones serian frecuentes y destructivas. El optimo se vuelve aprender SOLO 2 features (las que mas contribuyen al loss) y descartar las otras 3 (norma cero). El modelo perdera capacidad — no es solo un cambio de geometria sino una reduccion del numero de features representables. La sparsity es lo que permite la compresion.

**3. ¿Como se relaciona superposition con la capacidad de los modelos modernos?**

GPT-4 tiene `d_model` del orden de 12000-18000. Sin superposition, podria representar como mucho ~15000 conceptos ortogonalmente. Pero estimaciones (Sparse Autoencoders sobre Anthropic Claude) sugieren que estos modelos representan **millones** de features distintas. La unica forma de empacar millones de features en miles de dimensiones es superposition masiva. Cada neurona individual es polisemantica — responde a docenas de conceptos no relacionados. Esto es por que la interpretabilidad neurona-por-neurona NO funciona en LLMs grandes; necesitas SAEs para descomponer el espacio en features monosemanticas. El campo entero de mech interp moderna (2023-2026) esta basado en este hecho.
