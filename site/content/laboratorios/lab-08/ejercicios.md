---
title: "Ejercicios"
weight: 40
math: true
---

El laboratorio contiene 3 experimentos, cada uno con actividades practicas. Las respuestas deben ser respondidas en las celdas del notebook dispuestas para ello.

---

## Experimento 1: Funciones de Perdida

Se entrena una CNN en CIFAR-10 con Cross-Entropy y con MSE, y se comparan los resultados.

### Actividad: MSE vs Cross-Entropy

Para cada uno de los siguientes problemas, indicar si usaria MSE o Cross-Entropy y justificar:

1. **Un sistema que estima la probabilidad de que un cliente compre pasajes de avion dentro de los proximos 30 dias.**

2. **Un sistema que predice que tan positivo o negativo es un comentario en un foro mediante un score continuo entre -1 y 1.**

3. **Un sistema que estima el precio del dolar a partir del analisis de tweets relacionados con Donald Trump.**

4. **Un sistema que aproxima una funcion discreta que mide la cantidad de alumnos que asistiran a la proxima clase (un numero entero entre 1 y 40).**

5. **Un sistema que para cada pixel de una imagen en blanco y negro predice 3 valores entre 0 y 255 (R, G, B) para colorearla.**

6. **Un sistema que aproxima una funcion continua altamente no lineal.**

{{< concept-alert type="clave" >}}
La clave para decidir entre MSE y Cross-Entropy es preguntarse si la salida es un **valor continuo** (regresion, usar MSE) o una **categoria discreta** (clasificacion, usar Cross-Entropy). Cuando la salida es un numero con orden y distancia significativa, MSE es apropiado.
{{< /concept-alert >}}

---

## Experimento 2: Regularizacion

Se entrena una red grande (1000 neuronas por capa) en un dataset 2D de clasificacion, primero sin regularizacion y luego con regularizacion L2. Se visualizan los limites de clasificacion.

### Actividad 1: Analisis de limites de clasificacion

**1. Explique brevemente: cual es la principal diferencia entre la forma de los limites de clasificacion del modelo sin regularizacion vs con regularizacion?**

```python
# Pista: observar el grafico de decision boundary de ambos modelos
# Sin regularizacion: weight_decay=0.0
# Con regularizacion: weight_decay=0.2
```

**2. A nivel del modelo mismo, que explica esta diferencia por el hecho de agregar o quitar la regularizacion?**

### Actividad 2: Comparacion de accuracy

**3. Al comparar el modelo sin regularizacion vs el modelo con regularizacion:**
  - Cual es mejor en el conjunto de entrenamiento?
  - Cuanto accuracy obtuvo en el conjunto de entrenamiento?

**4. Al comparar ambos modelos:**
  - Cual es mejor en el conjunto de test?
  - Cuanto accuracy obtuvo en el conjunto de test?

**5. Unicamente considerando lo analizado en las preguntas 3 y 4: que estrategia recomendaria usar para generar el modelo definitivo?**

### Actividad 3: Ajuste de lambda

**6. Revisando el codigo anterior, que linea tendria que editar si usted quisiera aumentar a 0.9 la importancia de la regularizacion L2? Copie y pegue la linea de codigo modificada.**

```python
# Linea original:
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.2)

# Linea modificada:
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=______)
```

**7. Ejecute el codigo con esta regularizacion L2 ponderada por 0.9. Que efecto tiene en la curva que separa las clases?**

### Actividad 4: Regularizacion extrema

**8. Repita la actividad 7, pero esta vez con regularizacion L2 ponderada por 100000.**
  - Que efecto tiene en el accuracy de test?
  - Que efecto tiene en el grafico que separa las clases?
  - Que explica este fenomeno?

*Hint: puede servir intentar explicarlo en base a la formula de la funcion de perdida que recibe el modelo:*

$$L_{\text{total}} = L_{\text{original}} + 100000 \sum_{i} w_i^2$$

*Cuando $\lambda$ es tan grande, que le conviene al optimizador: mejorar la prediccion o reducir los pesos?*

---

## Experimento 3: Tareas Auxiliares

Se usa el dataset CelebA con una CNN para clasificar atributos faciales, comparando entrenamientos con y sin tarea auxiliar.

### Actividad 1: Balanceo de escalas

**1. Supongamos que la loss asociada a la tarea auxiliar es 1000 veces mayor que la de la tarea principal (cuando la loss principal vale 100, la auxiliar vale 100000). Que tendria que hacer para que el modelo considere ambas perdidas de manera equilibrada (50%-50%)?**

```python
# Pista: si aux_loss ≈ 1000 * main_loss, entonces:
# Loss = main_loss + lambda * aux_loss
# Para que main_loss ≈ lambda * aux_loss:
lambda_value = ______
```

### Actividad 2: Multiples tareas auxiliares

**2. Supongamos que tenemos una tarea principal y 4 tareas auxiliares, todas con ordenes de magnitud similar. Que tendria que hacer para que el modelo considere la perdida de la tarea principal en un 50% y la perdida de todas las demas homogeneamente completando el otro 50%?**

```python
# Loss = main_loss + lambda_1 * aux1 + lambda_2 * aux2 + lambda_3 * aux3 + lambda_4 * aux4
# Para balance 50%-50% con magnitudes similares:
lambda_1 = ______
lambda_2 = ______
lambda_3 = ______
lambda_4 = ______
```
