"""
Demo de gradient descent puro, sin red neuronal.

Objetivo: ver con tus ojos como funciona el mecanismo "ajustar un numero
para minimizar otro numero". Esto es backpropagation en su forma mas
elemental: una sola variable, una sola operacion.

Vamos a minimizar la funcion f(x) = x^2, que tiene su minimo en x = 0.
Empezamos en x = 5 (lejos del minimo) y veremos como gradient descent
nos lleva a 0 paso a paso.
"""

import math
import torch

torch.set_printoptions(precision=4, sci_mode=False)


# ---------------------------------------------------------------------------
# La funcion a minimizar: f(x) = x^2
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 0: La funcion que vamos a minimizar")
print("=" * 70)

print("""
    f(x) = x^2

         loss
          │   .                      .
          │    .                    .
         16│     .                .
          │       .              .
          │         .          .
          │           .      .
         0│              ____             <- minimo en x = 0
          └─────────────┬──────────────  x
              -4   -2   0   2   4

  Si estamos en x = 5, queremos llegar a x = 0 (donde f es minimo).
  Si estamos en x = -3, tambien queremos llegar a x = 0.
""")


# ---------------------------------------------------------------------------
# El gradiente de f(x) = x^2 es f'(x) = 2x
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 1: El gradiente (la pendiente de la funcion)")
print("=" * 70)

print("""
La derivada de x^2 respecto a x es 2x. Eso es el gradiente.
En palabras: "si aumento x un poquito, f(x) aumenta a razon de 2x".

  En x = 5:    gradiente = 2*5 = 10    (pendiente alta hacia arriba)
  En x = 1:    gradiente = 2*1 = 2     (pendiente moderada)
  En x = 0:    gradiente = 2*0 = 0     (estamos en el minimo, sin pendiente)
  En x = -3:   gradiente = 2*(-3) = -6 (pendiente hacia arriba en sentido inverso)

Regla de gradient descent:
    nuevo_x = x_actual - learning_rate * gradiente

Si gradiente > 0 (subimos a la derecha): restamos -> vamos a la izquierda.
Si gradiente < 0 (subimos a la izquierda): restamos -> vamos a la derecha.
En ambos casos: vamos hacia el minimo.
""")


# ---------------------------------------------------------------------------
# Gradient descent manual paso a paso
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 2: Gradient descent manual con learning_rate = 0.1")
print("=" * 70)

x = 5.0  # punto de partida
learning_rate = 0.1
n_pasos = 20

print(f"\n{'paso':>4} | {'x':>8} | {'f(x)':>10} | {'gradiente':>10} | {'nuevo_x':>10}")
print("-" * 60)
for paso in range(n_pasos):
    fx = x ** 2          # valor de la funcion
    grad = 2 * x         # gradiente: derivada de x^2 es 2x
    nuevo_x = x - learning_rate * grad
    print(f"{paso:>4} | {x:>8.4f} | {fx:>10.4f} | {grad:>10.4f} | {nuevo_x:>10.4f}")
    x = nuevo_x

print(f"\nDespues de {n_pasos} pasos: x = {x:.6f}")
print(f"Distancia al minimo (x=0): {abs(x):.6f}")
print(f"Convergiendo hacia 0 exponencialmente.")


# ---------------------------------------------------------------------------
# Lo mismo con PyTorch autograd
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 3: Lo mismo, pero PyTorch calcula el gradiente automatico")
print("=" * 70)

print("""
En vez de calcular el gradiente a mano (2x), le pedimos a PyTorch que lo
haga. La clave es marcar el tensor con requires_grad=True para que PyTorch
"trackee" todas las operaciones y pueda derivar despues.
""")

x = torch.tensor(5.0, requires_grad=True)
learning_rate = 0.1

print(f"{'paso':>4} | {'x':>8} | {'f(x)':>10} | {'x.grad':>10}")
print("-" * 45)
for paso in range(n_pasos):
    fx = x ** 2          # forward: calcular f(x)
    fx.backward()        # backward: PyTorch calcula x.grad = df/dx
    grad = x.grad.item()

    print(f"{paso:>4} | {x.item():>8.4f} | {fx.item():>10.4f} | {grad:>10.4f}")

    # Actualizacion manual del peso. with torch.no_grad() le dice a PyTorch
    # "no trackees esta operacion en el grafo de computo, es housekeeping".
    with torch.no_grad():
        x -= learning_rate * x.grad
        x.grad.zero_()  # limpiar el gradiente para la siguiente iteracion

print(f"\nResultado final con autograd: x = {x.item():.6f}")
print("Mismo resultado que el calculo manual: PyTorch hizo la derivada por nosotros.")


# ---------------------------------------------------------------------------
# Experimento: learning rate muy chico
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 4: Que pasa con learning_rate muy chico (0.01)")
print("=" * 70)

x = 5.0
learning_rate = 0.01
n_pasos = 20

print(f"\nAhora con learning_rate = 0.01 (10x mas chico que antes):")
print(f"{'paso':>4} | {'x':>10} | {'f(x)':>12}")
print("-" * 35)
for paso in range(n_pasos):
    fx = x ** 2
    grad = 2 * x
    x = x - learning_rate * grad
    if paso % 4 == 0 or paso == n_pasos - 1:
        print(f"{paso:>4} | {x:>10.4f} | {fx:>12.4f}")

print(f"\nDespues de {n_pasos} pasos: x = {x:.4f}")
print("Nota: el modelo todavia esta lejos de 0. Con lr chico converge lento.")


# ---------------------------------------------------------------------------
# Experimento: learning rate muy grande
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 5: Que pasa con learning_rate muy grande (1.1)")
print("=" * 70)

x = 5.0
learning_rate = 1.1
n_pasos = 10

print(f"\nAhora con learning_rate = 1.1 (paso enorme):")
print(f"{'paso':>4} | {'x':>12} | {'f(x)':>15}")
print("-" * 40)
for paso in range(n_pasos):
    fx = x ** 2
    grad = 2 * x
    x = x - learning_rate * grad
    print(f"{paso:>4} | {x:>12.4f} | {fx:>15.4f}")

print(f"\nDespues de {n_pasos} pasos: x = {x:.4f}")
print("EL MODELO DIVERGE. Cada paso es tan grande que se pasa al otro lado")
print("del minimo y termina mas lejos. Es la version matematica de patear")
print("la pelota tan fuerte que se sale del campo.")


# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)

print("""
Esto que viste es exactamente lo que hace una red neuronal cuando entrena,
pero con MILLONES de variables a la vez en vez de una sola:

  - x = un peso del modelo (en la red real son millones)
  - f(x) = x^2 = el LOSS (en la red real es cross-entropy)
  - 2x = el gradiente (en la red real lo calcula backprop con regla de la cadena)
  - x = x - lr * 2x = la actualizacion del peso (igual que en la red)

El learning_rate es CRITICO:
  - Muy chico: converge pero lento.
  - Justo:    converge rapido al minimo.
  - Muy grande: diverge, el modelo nunca aprende.

En Transformers el learning_rate tipico es 1e-4 (0.0001) y se usa Adam,
que adapta lr por peso. Pero la idea base es la misma que viste aqui.
""")

print("=" * 70)
print("FIN demo gradient descent.")
print("=" * 70)
