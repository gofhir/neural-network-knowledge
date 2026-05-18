---
title: "Análisis de resultados y errores"
weight: 70
---

Análisis crítico del output final del pipeline en la actividad evaluable, identificando aciertos, errores y limitaciones estructurales del experimento.

## Resumen cuantitativo

| Métrica | Valor |
|---|---|
| Personas detectadas por OpenPifPaf | 9 |
| Clasificadas como corredor (rojo) | 7 |
| Clasificadas como ciclista (azul) | 2 |
| Accuracy del MLP en test (durante training) | 65.62% |
| Epochs ejecutadas (early stopping, patience=15) | 20 |
| Aciertos visuales estimados | ~6/9 (67%) |

**Consistencia operacional**: el accuracy del pipeline en la imagen real (~67%) coincide con el accuracy medido durante training (~65%). El modelo se comporta como dice el set de test.

## Aciertos del pipeline

### Los 2 corredores grandes en primer plano

Camisetas verde y blanca. Clasificados correctamente como `running`. Sus poses son **prototípicas**:

- Brazos asimétricos en movimiento.
- Una pierna levantada, otra apoyada.
- Torso ligeramente inclinado adelante.

El MLP captura esta configuración sin esfuerzo porque vio muchos ejemplos similares durante training.

### El ciclista del fondo (más a la derecha)

Clasificado correctamente como `riding_a_bike`. Pose clarísima:

- Sentado, manos en manubrio.
- Piernas dobladas a 90° (pedales).
- Esqueleto visualmente "compacto" comparado con corredores.

### Los 3-4 caminantes del centro

Clasificados como `running` — **defensible aunque visualmente caminan más que corren**. Justificación:

- Están **de pie**, en posición vertical.
- No están sentados sobre nada.
- Más cercanos a `running` (pose vertical) que a `riding_a_bike` (pose sentada).

**Si el MLP solo conoce 2 clases**, asignar `running` a alguien parado es la **elección menos errónea** disponible.

## Errores observados

### Error 1 — Ciclista del frente (a la derecha cercano)

- **Realidad**: sentado sobre bicicleta, pose claramente de ciclismo.
- **Predicción**: rojo (running).
- **Hipótesis del error**:
  1. OpenPifPaf pudo no detectar con claridad los keypoints de las piernas (dobladas detrás del marco de la bicicleta), produciendo un vector de features con **ceros donde debería haber información de pedaleo**.
  2. El MLP interpretó la **ausencia de "piernas dobladas"** como ausencia de pose de ciclismo, defaulteando a `running`.
  3. Postura ligeramente diferente al ciclista del fondo (más erguido, manos en otra posición) que el MLP nunca vio con suficientes ejemplos.

### Error 2 — Persona del centro (probablemente con cochecito)

- **Realidad**: probablemente caminante empujando algo.
- **Predicción**: azul (riding_a_bike) — sin bicicleta visible.
- **Hipótesis del error**: la pose de **"manos al frente sosteniendo algo"** se parece geométricamente a **"manos sobre manubrio"**. Sin acceso al contexto (el objeto sostenido), el MLP solo ve **brazos extendidos hacia adelante** y lo asocia con su prior de `riding_a_bike`.

## Limitaciones estructurales del experimento

### Limitación 1 — Tamaño reducido del dataset

`MAX_SAMPLES = 2000` con orden alfabético no-shuffleado deja **muy pocas muestras** de las clases `running` (R...) y `riding_a_bike` (también R), porque están cerca del corte alfabético de las primeras 2000 imágenes.

Verificación típica:

```python
print(f"Total ejemplos en data: {len(data)}")
# Probable: ~25-50 ejemplos
```

Con tan pocas muestras, el MLP no puede generalizar bien. **El techo del modelo es estructural, no algorítmico**.

**Fix profesional**: agregar `random.shuffle` antes del slice en `Stanford40Dataset.__init__`:

```python
filenames = [f for f in os.listdir(directory) if f.endswith('.jpg')]
random.seed(42)
random.shuffle(filenames)
self.image_filenames = filenames[:MAX_SAMPLES]
```

### Limitación 2 — Vocabulario binario sin categoría "ambiguo"

El MLP solo conoce 2 clases. Cualquier persona en pose **intermedia** (caminando, parada, empujando algo) debe ser forzada a una de las dos categorías. Esto produce clasificaciones técnicamente incorrectas pero **defensibles** dado el setup.

**Solución profesional**: agregar una tercera clase `other` o `unknown` que capture poses fuera del scope. Tendría que estar representada con ejemplos en el training set (otras clases de Stanford 40, e.g., `walking_the_dog`, `applauding`, `gardening`).

### Limitación 3 — Pérdida de estructura espacial en el flatten

El `prediction.data.reshape(-1)` aplana `(17, 3) → (51,)` perdiendo la **estructura espacial del esqueleto**. El MLP nunca sabe que esos 51 números son "keypoints" — son solo un vector para él.

**Arquitecturas que preservarían la estructura**:

- **GNN sobre el grafo del esqueleto**: cada keypoint es un nodo, las conexiones son aristas. Captura dependencias estructurales (ej. "si el brazo está doblado y el hombro está alto, es más probable que sea X").
- **Transformer con position embeddings por keypoint**: trata los 17 keypoints como una secuencia de tokens con sus posiciones aprendidas.

Ambos serían **overkill** para un lab pedagógico pero más informativos en producción.

### Limitación 4 — Sin random_state fijo

Los splits `train_test_split` en este lab **no son reproducibles** entre ejecuciones. Cada vez que ejecutas:

- El split cambia.
- La accuracy del MLP varía ±2-5%.
- El experimento PifPaf vs. OpenPose tiene splits **distintos entre sí**, añadiendo varianza al A/B test.

**Fix**:

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_int_labels)
```

### Limitación 5 — Una imagen puede contribuir múltiples ejemplos al training

Si una imagen tiene 3 personas, **las 3 contribuyen ejemplos al training set con el mismo label de la imagen**. Esto puede ser ruido si solo una persona realmente realiza la acción etiquetada.

**Fix profesional**: usar las bounding boxes del Stanford 40 para identificar **qué persona específica** hace la acción, e ignorar las demás detecciones de OpenPifPaf en esa imagen.

## Lo que esto nos dice sobre el modelo

El MLP captura bien las poses **extremas y prototípicas**:

- Corredor con brazos en movimiento + piernas extendidas.
- Ciclista sentado con manos al manubrio + piernas dobladas.

Falla en:

- **Poses ambiguas** (caminantes, gente parada).
- **Oclusiones parciales** donde keypoints clave no se detectan.
- **Variaciones de la pose** que no estuvieron en el training set.

Esto es **textbook overfitting con dataset pequeño**: el modelo memoriza prototipos en lugar de generalizar principios.

## Lo que esto nos dice sobre el ecosistema

Más allá del experimento concreto, el lab te enseñó implícitamente:

1. **Pretrain + cabeza ligera es el patrón dominante**. PifPaf preentrenado en COCO + MLP ligero para tu tarea — exactamente cómo se construye la mayoría de apps de visión en producción.

2. **El SOTA académico no siempre gana**. PifPaf supera marginalmente a OpenPose en este task, no dramáticamente. Para 4 clases bien-separadas en alta resolución, ambos modelos son **suficientemente buenos**.

3. **Friction de adopción importa más que accuracy**. OpenPose tiene buena precisión pero su licencia restrictiva + Caffe legacy lo excluyeron de muchas adopciones industriales. PifPaf (Apache 2.0 + `pip install`) ganó por ergonomía. MediaPipe ganó por producto.

4. **Disciplina experimental no es opcional**. random_state, stratify, mismos splits — estos son los detalles que en un paper son **obligatorios** y en este lab se sacrifican por simplicidad. Para tus experimentos en producción, **siempre los aplicas**.

## Conclusión

A pesar de las limitaciones reconocidas, el **pipeline end-to-end funciona** como prueba de concepto:

- OpenPifPaf detecta correctamente las 9 personas en la imagen.
- El MLP clasifica ~67% correctamente, consistente con su accuracy de training.
- La visualización con colores diferenciados comunica claramente la salida del modelo.

Para producción, sería necesario:

- **Aumentar el dataset** (eliminar `MAX_SAMPLES` o aumentarlo a ~10000).
- **Agregar shuffle** antes del slice.
- **Considerar tercera clase** o regla de confianza ("si ambas probabilidades < 0.6, marcar como ambiguo").
- **Explorar arquitecturas** que preserven la estructura del esqueleto (GNN, Transformer).
- **Validar con cross-validation k-fold** en lugar de un único train/test split.

## Cross-links

{{< cards >}}
  {{< card link="../actividad-running-vs-bike" title="Actividad evaluable" subtitle="El pipeline que produjo estos resultados" icon="check-circle" >}}
  {{< card link="../clasificador-pifpaf" title="Clasificador PifPaf" subtitle="El patrón base del experimento principal" icon="academic-cap" >}}
  {{< card link="/fundamentos/pose-estimation" title="Fundamento: Pose Estimation" subtitle="Marco teórico" icon="book-open" >}}
  {{< card link="/clases/clase-17" title="Clase 17 - Teoría" subtitle="Slides de la clase" icon="academic-cap" >}}
{{< /cards >}}
