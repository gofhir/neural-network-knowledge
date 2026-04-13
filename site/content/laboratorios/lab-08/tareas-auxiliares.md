---
title: "Tareas Auxiliares"
weight: 30
math: true
---

## Motivacion

A veces la tarea principal es dificil y la red no tiene suficiente "senal" para aprender buenas representaciones internas. Una **tarea auxiliar** es una tarea adicional que se entrena al mismo tiempo para forzar a la red a aprender features mas ricos y generales.

```text
Solo tarea principal (Smiling):
  La red tiene que aprender todo sobre caras
  a partir de una sola senal binaria (smile / no smile)

Con tarea auxiliar (Smiling + Young):
  La red aprende a detectar sonrisas Y juventud al mismo tiempo.
  Para predecir ambas cosas, las capas compartidas aprenden
  representaciones mas ricas de la cara (bordes, texturas, formas)
  que ayudan a AMBAS tareas.

  -> La tarea principal mejora porque las features son mejores
```

Esta idea se conoce como **multi-task learning**: entrenar un modelo en multiples tareas simultaneamente para que las representaciones compartidas se beneficien de todas las senales de supervision.

---

## Representaciones compartidas

La arquitectura tipica tiene capas compartidas (que aprenden features comunes) y **cabezas separadas** para cada tarea:

```text
                  Capas compartidas                Cabeza principal
Imagen -> Conv1 -> Conv2 -> FC1 -> FC2 ----------> FC3 -> Smiling?
                                    |
                                    +------------> FC4 -> Young?
                                                   Cabeza auxiliar
```

Las capas compartidas (Conv1, Conv2, FC1, FC2) aprenden representaciones utiles para **ambas** tareas. Esto es mas eficiente que entrenar dos redes separadas, y produce features mas robustos porque reciben supervision desde multiples angulos.

---

## CombinedLoss: como combinar las perdidas

Cada tarea genera su propia perdida. Se combinan con un peso $\lambda$:

$$L_{\text{total}} = L_{\text{principal}} + \lambda \cdot L_{\text{auxiliar}}$$

El parametro $\lambda$ controla la importancia relativa de la tarea auxiliar:

| $\lambda$ | Efecto |
|---|---|
| 0.0 | La tarea auxiliar no tiene efecto (se ignora) |
| 0.2 | La tarea auxiliar tiene poca influencia (valor tipico) |
| 1.0 | Ambas tareas tienen la misma importancia |

### Implementacion en PyTorch

```python
class CombinedLoss(nn.Module):
    def __init__(self, auxiliary_task, auxiliary_weight):
        super().__init__()
        self.auxiliary_task = auxiliary_task
        self.aux_weight = auxiliary_weight  # esto es lambda

    def forward(self, main_pred, aux_pred, main_labels, aux_labels):
        if aux_labels is None:
            # Sin tarea auxiliar: solo cross-entropy principal
            return F.binary_cross_entropy_with_logits(main_pred, main_labels)
        else:
            # Con tarea auxiliar: combinar ambas
            main_loss = F.binary_cross_entropy_with_logits(main_pred, main_labels)

            if self.auxiliary_task == 'Landmarks':
                aux_loss = F.mse_loss(aux_pred, aux_labels)  # regresion
            else:
                aux_loss = F.binary_cross_entropy_with_logits(aux_pred, aux_labels)

            return main_loss + self.aux_weight * aux_loss
```

{{< concept-alert type="clave" >}}
La tarea auxiliar puede usar una funcion de perdida **distinta** a la tarea principal. En el laboratorio, la tarea principal (Smiling) usa Binary Cross-Entropy porque es clasificacion binaria, y la tarea auxiliar de Landmarks usa MSE porque predecir coordenadas $(x, y)$ es regresion.
{{< /concept-alert >}}

---

## Dataset: CelebA

El laboratorio usa el dataset **CelebA** (rostros de celebridades) con multiples atributos anotados:

```text
Atributos disponibles (40 en total):
  Smiling, Young, Male, Eyeglasses, Bald, Bangs,
  Big_Nose, Black_Hair, Blond_Hair, Heavy_Makeup, ...

Landmarks: posicion (x, y) de 5 puntos clave
  (ojos izquierdo y derecho, nariz, extremos de la boca)
```

Para acelerar los experimentos, el laboratorio pre-filtra las imagenes a 10,000 por set (entrenamiento, validacion y test) y las reduce a 40x40 pixeles.

---

## Arquitectura del modelo

```python
class FaceModel(nn.Module):
    def __init__(self, auxiliary_task_dim=10):
        super().__init__()
        # Capas compartidas (aprenden features comunes)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)

        # Cabeza principal (tarea: Smiling)
        self.fc3 = nn.Linear(84, 1)

        # Cabeza auxiliar (tarea: Young o Landmarks)
        if auxiliary_task_dim is not None:
            self.fc4 = nn.Linear(84, auxiliary_task_dim)

    def forward(self, x):
        # Capas compartidas
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Dos salidas separadas
        main_task = self.fc3(x)
        auxiliary_task = self.fc4(x) if self.fc4 else None
        return main_task, auxiliary_task
```

La arquitectura es intencionalmente casi identica a la del Experimento 1 (CIFAR-10). La diferencia principal es la segunda cabeza de salida `fc4` que produce la prediccion auxiliar.

---

## Los tres experimentos del laboratorio

### Experimento sin tarea auxiliar (baseline)

```python
auxiliary_task = None
# Solo entrena la cabeza principal (Smiling)
# Sirve como baseline para comparar
```

### Tarea auxiliar binaria (Smiling + Young)

```python
auxiliary_task = 'Young'
auxiliary_weight = 0.2
# La tarea auxiliar es similar: clasificar un atributo binario de la cara
# Ambas tareas comparten estructura (entender rostros)
```

### Tarea auxiliar de regresion (Young + Landmarks)

```python
primary_task = 'Young'
auxiliary_task = 'Landmarks'
auxiliary_weight = 0.1
# La tarea auxiliar predice coordenadas (x, y) de puntos faciales
# Fuerza a la red a entender la GEOMETRIA de la cara
```

---

## Balanceo de pesos entre tareas

Un problema critico surge cuando las tareas tienen **ordenes de magnitud distintos** en su loss:

```text
main_loss ≈ 0.5    (cross-entropy, valores entre 0 y ~5)
aux_loss  ≈ 500.0  (MSE de landmarks, valores grandes)

Con lambda = 0.2:
  Loss = 0.5 + 0.2 * 500 = 100.5
  -> La tarea auxiliar DOMINA completamente
  -> La red ignora la tarea principal

Solucion: usar lambda MUY chico
  Loss = 0.5 + 0.001 * 500 = 1.0   <- balanceado
```

{{< concept-alert type="clave" >}}
Si la loss auxiliar es $K$ veces mayor que la principal y se quiere un balance 50%-50%, se necesita $\lambda = 1/K$. Por ejemplo, si la auxiliar vale 1000 veces mas, usar $\lambda = 0.001$ para que ambas contribuyan de forma equilibrada al loss total.
{{< /concept-alert >}}

### Multiples tareas auxiliares

Si hay mas de una tarea auxiliar, cada una recibe su propio peso:

$$L_{\text{total}} = L_{\text{principal}} + \lambda_1 L_{\text{aux1}} + \lambda_2 L_{\text{aux2}} + \lambda_3 L_{\text{aux3}}$$

Para un balance 50%-50% entre la tarea principal y todas las auxiliares con ordenes de magnitud similares, asignar $\lambda_i = 1/n$ donde $n$ es el numero de tareas auxiliares. Con 4 tareas auxiliares, $\lambda_i = 0.25$ cada una, de modo que la suma de los pesos auxiliares ($0.25 \times 4 = 1.0$) iguala al peso de la tarea principal.

---

## Cuando usar tareas auxiliares

**Util cuando:**

- La tarea principal tiene pocos datos o es dificil
- Existen tareas relacionadas con datos disponibles
- Las tareas comparten estructura (por ejemplo, ambas necesitan entender caras)
- Se quiere mejorar las representaciones internas del modelo

**NO util cuando:**

- Las tareas no estan relacionadas (detectar sonrisas + predecir clima)
- La tarea auxiliar es mucho mas facil o dificil que la principal
- $\lambda$ esta mal calibrado (muy alto: la auxiliar domina; muy bajo: no tiene efecto)

---

## Conexion con regularizacion

Las tareas auxiliares actuan como una forma de **regularizacion implicita**. Al forzar a las capas compartidas a ser utiles para multiples tareas, se previene la especializacion excesiva en la tarea principal. Esto reduce el overfitting de manera similar a L2 o Dropout, pero a traves de un mecanismo distinto: en vez de restringir los pesos directamente, se restringen las representaciones para que sean mas generales.
