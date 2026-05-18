---
title: "Actividad evaluable: pipeline running vs riding_a_bike"
weight: 60
math: true
---

La actividad evaluable combina **todo lo aprendido** en un pipeline end-to-end aplicado a una imagen nueva. Tres sub-tareas con **2 puntos cada una** + 1 punto base por entrega completa = **7 puntos totales**.

## El pipeline completo

```
Imagen Honbike (multi-persona)
   │
   ▼ OpenPifPaf → predictions (9 esqueletos detectados)
   │
   ▼ MLP entrenado sobre running vs. riding_a_bike
   │
   ▼ is_running(pred) → True/False por persona
   │
   ▼ Visualización: rojo (corredor) vs azul (ciclista)
```

## La imagen objetivo

```python
image_response = requests.get('https://www.honbike.com/cdn/shop/articles/benefits-of-biking-and-running.png')
pil_image = PIL.Image.open(io.BytesIO(image_response.content)).convert('RGB')
display(pil_image)
```

Foto del blog "Benefits of Biking and Running" de [Honbike](https://www.honbike.com) (tienda de e-bikes). **Escena multi-persona** con corredores y ciclistas mezclados — escenario ideal para validar el pipeline.

**Detalle implícito**: la variable `pil_image` se **sobrescribe** (era la foto COCO del demo). El alumno hereda ese nombre estándar para reusar código de las celdas 11-12 directamente.

![Imagen Honbike base](/laboratorios/lab-17/honbike-base.jpg)

## Sub-tarea 1 — Entrenamiento del MLP (2 puntos)

```python
LABELS = ['riding_a_bike', 'running']
data = prepare_pifpaf_data_for_a_subset_of_labels(LABELS)
```

El profesor provee el filtrado de `data` ya hecho. El alumno solo construye los tensores, hace el split y entrena.

### El bug sutil en el orden de LABELS

`LABELS = ['riding_a_bike', 'running']` — **`riding_a_bike` en índice 0, `running` en índice 1**.

Pero la consigna pide colorear **rojo para corredores** (palette index 0) y **azul para ciclistas** (index 1). Hay una **inversión silenciosa** entre el orden de `LABELS` y el orden esperado de colores.

**Fix robusto**: usar nombres explícitos en la función `is_running`, no asumir el orden.

### El training loop con early stopping

Mejora opcional sobre el código sugerido por el profesor — usa **early stopping con best checkpoint**:

```python
import copy

model = MLP(input_size=X_train.shape[1], hidden_dim=128, output_size=y_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.BCEWithLogitsLoss()

epochs = 100
patience = 15
best_accuracy = 0
best_model_state = None
epochs_without_improvement = 0

training_losses = []
testing_accuracies = []

for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    training_losses.append(loss.item())
    current_accuracy = calculate_accuracy(model, test_loader)
    testing_accuracies.append(current_accuracy)

    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping en epoch {epoch + 1}. Mejor accuracy: {best_accuracy:.2f}%")
        break

model.load_state_dict(best_model_state)
```

**Por qué importa el checkpoint**: con dataset pequeño (~25-50 muestras por clase), la accuracy de test **oscila ±5-10 puntos** entre epochs. Sin checkpoint, `model` queda con los pesos de la última epoch (que pueden ser inferiores al peak). Con checkpoint, captura el **mejor estado visto durante training**.

### Resultados — gráficos con early stopping

![Pérdida actividad](/laboratorios/lab-17/actividad-mlp-loss.png)

![Accuracy actividad](/laboratorios/lab-17/actividad-mlp-accuracy.png)

**Early stopping cortó en epoch 20** capturando el mejor checkpoint (epoch ~5-8). Accuracy plateau en **~50-65%** debido al dataset reducido — las clases `running` y `riding_a_bike` empiezan con R, cercanas al corte alfabético de `MAX_SAMPLES = 2000`, lo que deja **muy pocas muestras por clase**.

## Sub-tarea 2 — Reconocimiento de poses en la imagen (2 puntos)

Inferir esqueletos sobre la imagen Honbike — copia del demo de la celda 12 con `pil_image` actualizada:

```python
data = openpifpaf.datasets.PilImageList([pil_image])
loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=True)

for images_batch, _, __ in loader:
    images_batch = images_batch.cuda()
    fields_batch = processor.fields(images_batch)
    predictions = processor.annotations(fields_batch[0])

    keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)
    with openpifpaf.show.image_canvas(pil_image) as ax:
        keypoint_painter.annotations(ax, predictions)
```

**`predictions` queda en el scope global** del notebook para que la sub-tarea 3 la consuma. Es **importante mantener el nombre estándar** — la celda 46 espera leer `predictions`.

![Inferencia OpenPifPaf sobre imagen Honbike](/laboratorios/lab-17/actividad-pifpaf-honbike.jpg)

**9 esqueletos detectados**: 2 corredores grandes en primer plano + 3-4 caminantes del centro + 2 ciclistas a la derecha. **OpenPifPaf detecta incluso las personas pequeñas del fondo** — fortaleza del paper para baja resolución.

## Sub-tarea 3 — Clasificación con colores diferenciados (2 puntos)

### La función `is_running`

```python
model.eval()

def is_running(prediction):
    """Clasifica una predicción de OpenPifPaf como corredor (True) o ciclista (False)."""
    flat = prediction.data.reshape(-1)
    tensor = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)

    predicted_index = out.argmax(dim=1).item()
    return predicted_index == LABEL_TO_INDEX['running']
```

**5 líneas que compactan el patrón canónico de inferencia con un clasificador downstream**:

1. **Flatten** keypoints `(17, 3) → (51,)`.
2. **To tensor** + `.unsqueeze(0)` para agregar dimensión de batch.
3. **Forward pass** dentro de `torch.no_grad()` (sin gradientes).
4. **Argmax** sobre los 40 outputs.
5. **Compara** con `LABEL_TO_INDEX['running']` (mapeo explícito, no índice ciego).

### Por qué `LABEL_TO_INDEX['running']` y no `argmax() == 1`

Aunque solo entrenaste 2 clases, el MLP produce **40 logits** (uno por clase de Stanford 40). El índice del argmax corresponde al label dentro de las 40 clases, no dentro de las 2 del subset. Usar `LABEL_TO_INDEX['running']` es **robusto** porque mapea el nombre del label a su índice numérico de manera explícita.

Si usaras `argmax() == 1` (asumiendo que `running` está en posición 1 de `LABELS`), el código rompería si alguien cambiara el orden de `LABELS`.

### La visualización colorizada

```python
fig, ax = plt.subplots(figsize=(15, 8))
ax.imshow(pil_image)
keypoint_painter = openpifpaf.show.KeypointPainter(linewidth=3)

for i, pred in enumerate(predictions):
    if is_running(pred):
        color = 'red'
    else:
        color = 'blue'

    keypoint_painter.annotations(ax, [pred], color=color)

plt.axis('off')
plt.show()

n_running = sum(is_running(pred) for pred in predictions)
n_biking = len(predictions) - n_running
print(f"Detectados: {n_running} corredores y {n_biking} ciclistas")
```

### Detalles críticos

1. **`KeypointPainter(linewidth=3)` sin `color_connections=True`**: clave para que el argumento `color` se respete. Si pusieras `color_connections=True`, OpenPifPaf seguiría asignando colores fijos por limb e ignoraría tu argumento.

2. **`[pred]` con corchetes** en `keypoint_painter.annotations(ax, [pred], color=color)`: la función espera una **lista**, no un elemento individual.

3. **Bug típico que el profesor dejó en la plantilla**: el comentario sugiere `if is_running(predictions):` (la lista entera) en lugar de `if is_running(pred):` (el iterando). El alumno tiene que corregirlo o falla con `AttributeError`.

### Resultado final

![Visualización final con colores diferenciados](/laboratorios/lab-17/actividad-final-clasificada.jpg)

**Output**: `Detectados: 7 corredores y 2 ciclistas`

Lectura:

| Persona | Realidad | Predicción | Acierto |
|---|---|---|---|
| 2 corredores grandes (verde+blanco) | running | rojo 🔴 | ✅ |
| Ciclista del fondo (más a la derecha) | riding_a_bike | azul 🔵 | ✅ |
| Caminantes del centro (3-4) | ambiguos | rojo 🔴 | ✅ defensible |
| Ciclista del frente (derecha cercano) | riding_a_bike | rojo 🔴 | ❌ |
| Persona con cochecito (centro) | ambiguo | azul 🔵 | ⚠️ confuso |

**~6/9 correctos = 67%** — consistente con el accuracy ~65% del MLP medido durante training.

## Cross-links

{{< cards >}}
  {{< card link="../clasificador-pifpaf" title="Clasificador PifPaf" subtitle="Patrón base del MLP que reusas" icon="academic-cap" >}}
  {{< card link="../analisis-resultados" title="Análisis de resultados y errores" subtitle="Qué falló y por qué" icon="light-bulb" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="Paper PifPaf" subtitle="Modelo de inferencia upstream" icon="document-text" >}}
{{< /cards >}}
