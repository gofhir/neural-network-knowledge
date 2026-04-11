# Clase 8 — Analisis del Laboratorio

Desglose pregunta por pregunta del Practico 8: Funciones de Perdida, Regularizacion y Tareas Auxiliares.
Incluye resultados reales de la ejecucion del notebook y las respuestas para cada celda.

---

## Experimento 1: Funciones de Perdida

### Contexto

Se entrena una CNN en **CIFAR-10** (imagenes a color de 10 clases) con dos funciones de perdida:
- Cross-Entropy (output_dim=10, un valor por clase)
- MSE (output_dim=1, un solo valor numerico para la clase)

### Regla de decision

```text
  ¿Que tipo de valor predice el modelo?

  A) Un NUMERO continuo (precio, temperatura, score, coordenadas)
     → MSE

  B) Una CATEGORIA (gato/perro, spam/no-spam, digito 0-9)
     → Cross-Entropy

  C) Una PROBABILIDAD de si/no (¿compra? ¿enfermo?)
     → BCE (Binary Cross-Entropy)

  La diferencia entre A y B es si el ORDEN importa.
  Si predices 8 y el real es 9, ¿eso es "casi bien"?
    Si → MSE
    No (porque 8="barco" y 9="camion" no tienen cercania) → Cross-Entropy
```

### Preguntas: ¿En cuales usaria MSE en lugar de Cross-Entropy?

El lab pide marcar True/False en cada pregunta (True = usaria MSE).

---

**1) Estimar probabilidad de compra de pasajes (0% a 100%)**

→ **Respuesta = False** (NO es MSE, es BCE)

```text
  Parece un numero continuo, pero no lo es.
  El resultado final es compra o no compra (si/no).
  El porcentaje es la probabilidad de un evento binario.

  → BCE, porque es un clasificador binario con salida probabilistica.
  Esta es la "trampa": el 0%-100% parece un numero,
  pero en realidad es la probabilidad de un evento binario.
```

---

**2) Score de sentimiento continuo entre -1 y 1**

→ **Respuesta = True** (SI es MSE)

```text
  El output es un VALOR CONTINUO entre -1 y 1.
  -0.3 es mas negativo que +0.2. Hay orden y distancia.
  Predecir 0.8 cuando el real es 0.9 es CASI CORRECTO.

  Nota: si fuera clasificar en 3 clases (positivo/neutro/negativo)
  se usaria Cross-Entropy. La diferencia es CONTINUO vs CATEGORICO.
```

---

**3) Estimar precio del Dolar a partir de tweets**

→ **Respuesta = True** (SI es MSE)

```text
  $823 es un NUMERO CONTINUO. Equivocarse por $2 es mejor
  que equivocarse por $50. Hay distancia numerica real.
  Es regresion pura.
```

---

**4) Cantidad de alumnos que asistiran (entero entre 1 y 40)**

→ **Respuesta = True** (SI es MSE)

```text
  Aunque es un entero, 14 esta mas cerca de 15 que de 5.
  El ORDEN numerico importa.

  Con MSE:
    error(14, 15) = (14-15)² = 1      ← chico (casi acerto)
    error(5, 15)  = (5-15)²  = 100    ← grande (muy lejos)

  Con Cross-Entropy (40 clases):
    Predecir "14" cuando es "15" seria TAN malo como predecir "1".
    No aprovecha que 14 esta CERCA de 15.
```

---

**5) Predecir color de cada pixel (RGB, 3 valores entre 0-255)**

→ **Respuesta = True** (SI es MSE)

```text
  Cada canal (R, G, B) es un NUMERO CONTINUO de 0 a 255.
  Rojo=195 cuando el real es 200 es un error chico.
  Tres regresiones independientes (una por canal).
```

---

**6) Aproximar funcion continua altamente no lineal**

→ **Respuesta = True** (SI es MSE)

```text
  f(x) → y, donde y es un NUMERO CONTINUO.
  Caso textbook de regresion.
```

---

### Resumen Experimento 1

```text
  Pregunta   Respuesta   ¿Que predice?              ¿Orden importa?
  ────────   ─────────   ──────────────             ───────────────
  1          False       si/no (compra)              N/A (binario) → BCE
  2          True        numero entre -1 y 1         SI → MSE
  3          True        precio ($)                  SI → MSE
  4          True        entero 1-40                 SI → MSE
  5          True        3 valores 0-255             SI → MSE
  6          True        valor continuo f(x)         SI → MSE

  5 de 6 son True (MSE). Solo la pregunta 1 es False (BCE).
```

---

## Experimento 2: Regularizacion

### Contexto

Red grande para datos 2D simples. Se compara:
- Sin regularizacion: `weight_decay=0.0`
- Con regularizacion L2: `weight_decay=0.2`

### Resultados reales de la ejecucion

```text
  Sin regularizacion (weight_decay=0.0):
    Train: loss 0.0143, acc 100.00%  (epoca 150)
    Test:  loss 0.2130, acc 95.58%

  Con regularizacion L2 (weight_decay=0.2):
    Train: loss 0.2215, acc 95.83%  (epoca 150)
    Test:  loss 0.2130, acc 95.58%

  Con regularizacion L2 (weight_decay=0.9):
    Train: loss 0.5354, acc 94.44%  (epoca 150)
    Test:  loss 0.5248, acc 96.13%

  Con regularizacion L2 (weight_decay=100000):
    Train: loss 0.6931, acc 52.08%  (epoca 150)
    Test:  loss 0.6931, acc 54.70%
```

### Preguntas y respuestas

---

**1. ¿Cual es la principal diferencia entre los limites de clasificacion sin vs con regularizacion?**

→ Respuesta (texto):

```text
Sin regularizacion los limites de clasificacion son complejos y ruidosos,
se ajustan a cada punto individual. Con regularizacion L2 los limites son
suaves y simples, capturando el patron general.
```

Explicacion detallada:

```text
  Sin regularizacion:
    El limite se curva y retuerce para pasar por TODOS los puntos.

    ┌─────────────────────┐
    │ o · o ·    ··  o    │
    │  ╱──╲    ╱──╲       │  ← curvas muy complejas
    │ · o  ╲──╱  o  · o   │
    │    ╱──╲   ╱──╲      │
    │ o ╱  o ╲─╱ o   · o  │
    └─────────────────────┘

  Con regularizacion L2:
    El limite es suave, captura el patron general.

    ┌─────────────────────┐
    │ o o o         · ·   │
    │ o o    ╲      · · · │  ← curva suave
    │  o o    ╲    · ·    │
    │   o      ╲  · · ·  │
    │ o o       ╲  · ·   │
    └─────────────────────┘
```

---

**2. A nivel del modelo, ¿que explica esta diferencia?**

→ Respuesta (texto):

```text
La regularizacion L2 penaliza los pesos grandes (agrega λ*Σw² al loss).
Sin L2 los pesos crecen libremente generando curvas complejas.
Con L2 los pesos se mantienen chicos, produciendo limites de decision mas suaves.
```

Explicacion detallada:

```text
  Matematicamente:
    Loss = CrossEntropy + λ * Σ(w²)
    Si w=50: el penalty es 0.2 * 2500 = 500 → MUY caro
    Si w=1:  el penalty es 0.2 * 1 = 0.2   → barato

  Sin L2:  pesos = [50.3, -30.1, 0.01, 85.2, -42.0, ...]  → curvas extremas
  Con L2:  pesos = [1.2, -0.8, 0.5, -0.3, 0.9, -0.6, ...] → curvas suaves
```

---

**3. ¿Cual es mejor en el conjunto de ENTRENAMIENTO?**

→ Respuesta:
- Mejor_Modelo: **Sin Regularizacion**
- Accuracy: **1.0 (100%)**

```text
  Sin regularizacion llega a 100% en train (desde la epoca ~81).
  Con regularizacion se queda en 95.83%.

  Sin regularizacion "gana" en train porque MEMORIZO cada dato.
  Es como un alumno que se aprende las respuestas de la guia de memoria:
  en la guia (train) saca 100%, pero en la prueba (test) no necesariamente.
```

---

**4. ¿Cual es mejor en el conjunto de TEST?**

→ Respuesta:
- Mejor_Modelo: **Son iguales**
- Accuracy: **0.9558 (95.58%)**

```text
  En esta ejecucion ambos dan exactamente lo mismo en test: 95.58%.
  Esto puede pasar cuando el dataset es relativamente simple.
  Aunque en teoria la regularizacion mejora el test,
  en esta ejecucion el resultado es identico.
```

---

**5. ¿Que estrategia recomendaria?**

→ Respuesta:
- Modelo_seleccionado: **Con Regularizacion**

```text
  Aunque en este caso el test es igual, con regularizacion es la
  estrategia mas segura: el modelo es mas simple, generaliza mejor
  en general, y no depende de haber memorizado los datos.
```

---

**6. ¿Que linea cambiar para weight_decay=0.9?**

→ Respuesta (codigo):

```python
optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 0.9)
```

---

**7. ¿Que efecto tiene weight_decay=0.9?**

→ Respuesta (texto):

```text
Con weight_decay=0.9 la curva que separa las clases es mas suave que con
weight_decay=0.2. El train accuracy baja de 95.83% a 94.44% porque el
modelo tiene menos libertad para ajustarse a los datos. Sin embargo, el
test accuracy sube ligeramente de 95.58% a 96.13%, lo que indica que la
regularizacion mas fuerte ayuda a generalizar un poco mejor en este caso.
El loss de entrenamiento es mas alto (0.535 vs 0.221) porque el termino
de penalizacion L2 es mayor.
```

---

**8. ¿Que pasa con weight_decay=100000?**

→ Respuesta (tres campos):

```text
  Efecto_test:
    El accuracy de test cae a 54.70%, practicamente aleatorio.
    El modelo no aprendio nada util.

  Efecto_grafico:
    El limite de clasificacion desaparece o es una linea recta
    que no separa las clases correctamente.

  Explicacion:
    Con weight_decay=100000 el termino de regularizacion domina
    completamente el loss. Los pesos son forzados a ser practicamente
    cero, por lo que la red no puede aprender. El loss se estanca en
    0.6931 (que es exactamente ln(2), el loss de predecir 50/50 para
    ambas clases), confirmando que el modelo esta adivinando al azar.
```

### Resumen visual de weight_decay

```text
  weight_decay    Train Acc    Test Acc    Loss final    Comportamiento
  ────────────    ─────────    ────────    ──────────    ──────────────
  0.0             100.00%      95.58%      0.0143        Overfitting (memoriza)
  0.2              95.83%      95.58%      0.2215        Regularizado (suave)
  0.9              94.44%      96.13%      0.5354        Mas suave (mejor test)
  100000           52.08%      54.70%      0.6931        No aprende nada

  λ=0:      sin freno → overfitting
  λ=0.2:    freno suave → generaliza bien
  λ=0.9:    freno fuerte → aun mejor test en este caso
  λ=100000: freno de mano → no se mueve (accuracy aleatorio)
```

---

## Experimento 3: Tareas Auxiliares

### Contexto

Dataset CelebA (caras de celebridades, 40x40 pixeles). La CNN tiene capas compartidas y opcionalmente una cabeza auxiliar.

### ¿Que es una tarea auxiliar?

```text
  Normalmente una red aprende UNA tarea:
    Imagen → Red → ¿Sonrie? (si/no)

  Con tarea auxiliar, aprende DOS tareas al mismo tiempo:
                                         ┌→ ¿Sonrie?  (tarea principal)
    Foto → Conv1 → Conv2 → FC1 → FC2 ──┤
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ └→ ¿Joven?   (tarea auxiliar)
           estas capas aprenden features
           que sirven para AMBAS tareas

  ¿Por que ayuda?
    Las capas compartidas aprenden features MAS RICAS porque tienen
    que servir para AMBAS tareas. Es como estudiar para dos pruebas
    a la vez: terminas entendiendo el tema mas profundamente.

  Lo importante: al modelo SOLO le importa la tarea principal al final.
  La auxiliar es un "truco" de entrenamiento para que las capas
  compartidas aprendan mejores representaciones.
```

### Los 3 escenarios del lab

```text
  Escenario   Principal    Auxiliar                  λ
  ─────────   ─────────    ────────                  ───
  1           ¿Sonrie?     ninguna                   —
  2           ¿Sonrie?     ¿Joven? (binaria)         0.2
  3           ¿Joven?      Landmarks (regresion)     0.1
```

### El loss combinado

```text
  Loss = Loss_principal + λ * Loss_auxiliar

  λ controla cuanta importancia le das a la auxiliar:
    λ = 0    → ignoras la auxiliar (como no tenerla)
    λ = 0.5  → la auxiliar tiene la mitad de importancia
    λ = 1.0  → ambas tienen la misma importancia
    λ muy grande → la auxiliar domina y la principal empeora

  IMPORTANTE: la tarea auxiliar puede usar una loss DISTINTA a la principal.
    Principal binaria → BCE (Cross-Entropy binaria)
    Auxiliar binaria  → BCE
    Auxiliar Landmarks → MSE (porque es regresion de coordenadas x,y)
```

### Preguntas y respuestas

---

**1. Si la loss auxiliar es 1000x mayor que la principal, ¿que haria para que ambas contribuyan 50/50?**

→ Respuesta (texto):

```text
Usaria un lambda pequeno para equilibrar las escalas. Como la loss auxiliar
es 1000 veces mayor, elegiria λ = 0.001. De esta forma ambas perdidas
contribuyen con magnitudes similares al loss total y el modelo no ignora
la tarea principal.
```

Explicacion detallada:

```text
  Problema:
    main_loss ≈ 100
    aux_loss  ≈ 100000 (1000 veces mayor)

    Con λ=1: Loss = 100 + 1 * 100000 = 100100
    → 99.9% del loss viene de la auxiliar. La principal se ignora.

  Solucion: λ = main_loss / aux_loss = 100 / 100000 = 0.001

    Con λ=0.001: Loss = 100 + 0.001 * 100000 = 100 + 100 = 200
    → Ambas contribuyen 50% cada una. ✓

  Demostracion con distintos λ:
    λ=1.000: Loss=100100  (main=0.1%, aux=99.9%)
    λ=0.100: Loss=10100   (main=1%,   aux=99%)
    λ=0.010: Loss=1100    (main=9%,   aux=91%)
    λ=0.001: Loss=200     (main=50%,  aux=50%)  ← equilibrado
```

---

**2. 1 principal + 4 auxiliares, todas con escala similar. Principal al 50%, las 4 auxiliares repartiendo el otro 50%.**

→ Respuesta (texto):

```text
Asignaria λ = 0.25 a cada una de las 4 tareas auxiliares. Asi la suma de
los pesos auxiliares es 0.25*4 = 1.0, igual al peso de la principal,
logrando que cada grupo aporte 50% del loss total.
```

Explicacion detallada:

```text
  Loss = main_loss + 0.25*aux1 + 0.25*aux2 + 0.25*aux3 + 0.25*aux4

  Verificacion:
    Peso principal:  1.0           → 1.0/2.0 = 50%  ✓
    Peso auxiliares: 0.25*4 = 1.0  → 1.0/2.0 = 50%  ✓
    Cada auxiliar:   0.25/2.0 = 12.5%                 ✓

  En codigo:
    loss = main_loss
    for aux_loss in [aux1, aux2, aux3, aux4]:
        loss = loss + 0.25 * aux_loss

  OJO: esto asume que todas las losses tienen escalas similares.
  Si alguna es 1000x mayor, ajustar su λ individual.
```

---

## Resumen del Laboratorio

```text
Experimento 1 (Funciones de Perdida):
  - MSE para regresion (valores continuos con orden)
  - Cross-Entropy para clasificacion (categorias sin orden)
  - BCE para probabilidad de evento binario
  - MSE NO funciona para clasificacion (prefiere predecir la clase del medio)
  - 5 de 6 preguntas son MSE (True), solo la 1 es BCE (False)

Experimento 2 (Regularizacion L2):
  - weight_decay=0.0: sin regularizacion → overfitting (train 100%, test 95.58%)
  - weight_decay=0.2: pesos chicos → limites suaves (train 95.83%, test 95.58%)
  - weight_decay=0.9: aun mas suave (train 94.44%, test 96.13%)
  - weight_decay=100000: no aprende nada (train 52.08%, test 54.70%, loss=ln(2))
  - La linea clave: optimizer = optim.Adam(..., weight_decay=X)
  - Siempre preferir con regularizacion para produccion

Experimento 3 (Tareas Auxiliares):
  - Capas compartidas + cabezas separadas por tarea
  - Loss = main_loss + λ * aux_loss
  - λ debe calibrarse segun la escala de los losses
  - Si aux_loss es 1000x mayor → λ = 0.001
  - 1 principal + 4 auxiliares al 50/50 → λ = 0.25 cada una
  - La auxiliar es un truco para que las capas compartidas aprendan mejores features
```
