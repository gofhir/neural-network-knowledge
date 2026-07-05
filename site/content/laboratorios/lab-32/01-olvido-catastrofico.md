---
title: "Olvido catastrófico y Permuted MNIST"
weight: 1
---

Antes de combatir el olvido hay que **verlo y medirlo**. Esta primera parte monta el instrumento de medición (Permuted MNIST + una CNN chica) y produce la demostración cruda del fenómeno.

## Por qué el aprendizaje secuencial es distinto

En el aprendizaje estándar (i.i.d.) mezclas todos los datos en batches aleatorios. En [aprendizaje continuo](/fundamentos/aprendizaje-continuo) los datos llegan **no-i.i.d.**: primero ves toda la Tarea A, luego toda la B, y cuando entrenas B ya no tienes acceso (o muy limitado) a A. Ahí aparece el enemigo: el **olvido catastrófico** (McCloskey & Cohen, 1989) — la red sobreescribe los pesos útiles para A mientras optimiza para B. Es *catastrófico* porque es abrupto y casi total, no una erosión suave.

## El benchmark: Permuted MNIST

Cada tarea es MNIST con **una permutación fija de los 784 píxeles**, aplicada idéntica a todas sus imágenes. La Tarea 0 usa la identidad (MNIST original); la Tarea 1 la permutación `seed=1`; la Tarea 2 la `seed=2`.

**Dígitos originales** (Tarea 0) — estructura espacial intacta, legibles:

![Cuatro dígitos MNIST originales: 0, 4, 1, 9](/laboratorios/lab-32/mnist-digitos.png)

**Original vs permutado** (`seed=0`) — el mismo "0" a la izquierda, y a la derecha su versión con los píxeles barajados: ruido ininteligible para un humano:

![Comparación: dígito 0 original vs permutado, que parece nieve](/laboratorios/lab-32/permuted-seed0.png)

Por qué es un benchmark tan limpio para continual learning:

- **Dificultad idéntica entre tareas.** La permutación es una biyección fija: conserva toda la información. Un clasificador puede aprender cualquier permutación exactamente igual de bien. Así, **cualquier caída de accuracy se debe puramente al olvido**, no a que una tarea sea más difícil.
- **Mismo espacio de entrada/salida, distribuciones independientes.** Todas las tareas tienen entrada `[1,28,28]` y 10 clases → una sola red single-head. Pero lo que la red aprendió (qué píxeles importan) queda **inútil** para la siguiente permutación → máximo conflicto → máximo olvido.
- **Escenario Domain-IL.** Cambia el dominio de entrada, no las etiquetas. No hace falta saber qué tarea es en test.

**El matiz profundo:** una CNN asume localidad espacial. Al permutar, esa localidad se destruye — pero como la permutación es *fija y consistente*, la red aprende un patrón espacial artificial perfectamente aprendible. No "ve" un dígito; ve una firma estadística de posiciones activas, y eso basta.

## El modelo: una LeNet minimalista

Una CNN deliberadamente simple (~21.840 parámetros). No busca SOTA — es un instrumento para que el olvido se vea sin que otros efectos lo enmascaren.

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)   # 28→24
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 12→8
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)                  # 320 = 20·4·4
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))                    # [N,10,12,12]
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))   # [N,20,4,4]
        x = x.view(-1, 320)                                            # aplanado
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x                                                       # logits crudos
```

Flujo dimensional: `[N,1,28,28] → conv1+pool → [N,10,12,12] → conv2+pool → [N,20,4,4] → view → [N,320] → fc1 → [N,50] → fc2 → [N,10]`. El `320` está hardcodeado — solo funciona con entrada 28×28. Devuelve **logits crudos**; el `log_softmax` lo aplica internamente `F.cross_entropy` (más estable numéricamente vía log-sum-exp).

El entrenamiento usa **SGD con momentum 0.9**, no Adam: en continual learning se prefiere SGD porque su dinámica es más predecible y los buffers adaptativos de Adam interactúan de forma confusa con el cambio de tarea.

## La demostración del olvido

Se entrena la Tarea 0 (4 épocas → **96.33%**), luego la Tarea 2 (permutada) durante solo 2 épocas, midiendo ambas tareas antes y después:

| Momento | Tarea 1 (original) | Tarea 2 (permutada) |
|---|---|---|
| Modelo solo sabe T1 | **96.33%** ✅ | **10.10%** ❌ (azar) |
| Modelo ya aprendió T2 | **33.82%** 💥 | **82.60%** ✅ |

**La lectura:**

- El punto de partida es limpio: T2 en 10.10% = exactamente el azar (10 clases). La permutación es un "idioma" que la red nunca vio, sin nada transferible.
- La Tarea 2 se aprende rápido (10% → 83% en 2 épocas): la **plasticidad** está intacta.
- La Tarea 1 se **desploma de 96.33% → 33.82%**: una caída de **62.5 puntos**, o sea la red retuvo solo ~35% de lo que sabía. Eso es lo "catastrófico": no es erosión, es demolición en poquísimas actualizaciones de gradiente.
- El test loss de T1 se multiplicó **~30×** (0.0005 → 0.0154): la red no solo acierta menos, está **segura y equivocada** — su función de decisión se reconfiguró entera para T2.

Es el **dilema estabilidad-plasticidad** en su forma más cruda: SGD sin restricciones está 100% del lado de la plasticidad. Cada gradiente de T2 mueve *cualquier* peso, incluidos los críticos para T1. La red no tiene forma de saber que ciertos pesos "no se deben tocar".

Ese 33.82% es el enemigo. Las [tres estrategias](02-tres-estrategias) intentan subirlo sin bajar el 83%.
