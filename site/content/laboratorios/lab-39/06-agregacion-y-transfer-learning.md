---
title: "Agregación y transfer learning"
weight: 6
---

Corregido [el `ReLU`](/laboratorios/lab-39/05-el-relu-sobre-los-logits), VGGish llega a 91.45 %. Pero ese número todavía no es comparable con nada de la Parte 1, y además no es creíble. Las dos cosas se arreglan en esta página.

## La métrica no mide lo mismo en las dos partes

El `AudioDataset` de la Parte 2 corta cada clip en **tres parches de 0.96 s**, y la `collate_fn` replica la etiqueta:

```python
def make_batch(batch):
    tensors, labels = [], []
    for tensor, label in batch:
        bs, *_ = tensor.shape          # bs = 3
        tensors.append(tensor)
        for i in range(bs):
            labels.append(label)       # la etiqueta se repite 3 veces
    return torch.cat(tensors, dim=0).float().unsqueeze(1), torch.tensor(labels).long()
```

Es la misma técnica de *collate function* que el [lab 37](/laboratorios/lab-37) introdujo para manejar entradas de largo variable, aplicada aquí a un fin distinto: cada parche se convierte en un ejemplo independiente **que se clasifica y se evalúa por separado**.

La consecuencia es que todas las métricas de la Parte 2 son **por parche** sobre 2619 filas (3 × 873), mientras que las de la Parte 1 son **por clip** sobre 873. El propio notebook lo insinúa en su celda de cierre —*"al hacer la agregación por 3 debería ser aún mejor"*— y no la implementa.

## Cuatro formas de combinar tres parches

| Estrategia | Accuracy | Δ |
|---|---|---|
| por parche (0.96 s) | 91.45 % | — |
| voto mayoritario | 93.47 % | +2.02 |
| suma de logits | 96.91 % | +5.46 |
| **promedio de softmax** | **97.14 %** | **+5.69** |

**El voto mayoritario es el peor de los tres agregadores**, y por un margen grande: 3.4 puntos bajo la suma de logits. La razón es que reduce cada parche a un voto binario y **descarta la confianza**.

El escenario concreto: dos parches caen sobre ambiente y dicen `air_conditioner` con confianza tibia; el tercero contiene el evento y dice `gun_shot` con confianza altísima. El voto pierde 2 a 1; la suma de logits acierta. Con clips donde el evento ocupa una fracción del tiempo —exactamente el problema de dilución que hundía a `car_horn` en [la familia M](/laboratorios/lab-39/03-familia-m-y-campo-receptivo)— eso ocurre seguido.

{{< concept-alert type="clave" >}}
**Agregar predicciones es más robusto que agregar features.** Las redes M promedian *representaciones* sobre el clip completo con `AvgPool1d`, y ahí un evento breve queda diluido antes de que el clasificador lo vea. VGGish clasifica cada ventana por separado y combina *decisiones*, de modo que un parche muy informativo puede dominar a dos parches ambiguos.

Es la misma diferencia que en video separa el *average temporal pooling* de las estrategias de agregación por segmento, y explica por qué `car_horn` pasa de recall 0.61 en M5 a 0.95 en VGGish.
{{< /concept-alert >}}

## El 97 % no era un resultado

UrbanSound8K es un benchmark con más de una década de historia. Los resultados publicados con evaluación honesta van aproximadamente del **73 %** de las primeras CNN sobre espectrograma al **~88 %** de los modelos recientes preentrenados en AudioSet. Un 97.14 % implicaría fallar en **25 clips de 873** — menos errores que la propia tasa de ruido de etiquetado del dataset.

La señal de alarma más clara está en el desglose: `gun_shot` con precision y recall 1.00, siendo la clase que en la Parte 1 era el sumidero universal y que tiene apenas 35 ejemplos.

La causa ya se conoce: la Parte 2 arrastra [la misma fuga de folds](/laboratorios/lab-39/01-la-fuga-de-folds), confirmada por el contador `26196 = 3 × 8732`. Y acá el efecto es máximo, porque la combinación es la peor posible: embeddings muy expresivos, una capa lineal sobre 4096 features y los mismos clips vistos tres veces por época.

Verificarlo no exige recalcular los log-mel —basta filtrar del cache los índices del fold 1, ya que la precarga se hizo con `shuffle=False` sobre el mismo objeto:

```python
mask = torch.tensor([os.path.basename(os.path.dirname(p)) != 'fold1' for p in ds_tr.audio_paths])
A, L = train_cached_vgg.tensors
train_fix_vgg = torch.utils.data.TensorDataset(A[mask], L[mask])   # 7859 clips
```

| | Por parche | Por clip (softmax) |
|---|---|---|
| Split original (fuga) | 91.45 % | **97.14 %** |
| Split corregido | 76.90 % | **84.65 %** |
| Efecto de la fuga | −14.55 | **−12.49** |

El 84.65 % sí es consistente con lo publicado para transfer learning desde AudioSet sobre este dataset.

## El veredicto

Todos los modelos con folds disjuntos y métrica por clip:

| Modelo | Test | Parámetros entrenados | Épocas |
|---|---|---|---|
| M3 desde cero | 45.13 % | 221 194 | 20 |
| M5 desde cero | 52.12 % | 559 114 | 20 |
| **VGGish fine-tuning** | **84.65 %** | **40 970** | **3** |

**+32.53 puntos sobre M5, entrenando 14 veces menos parámetros durante 3 épocas en lugar de 20.** La afirmación del notebook —que un modelo preentrenado *"ya es competitivo con haber entrenado desde 0"* en la primera época— se queda corta: no es competitivo, es de otra categoría. Con el `ReLU` corregido, ya en la primera época supera a cualquier red M entrenada las 20 completas.

El desglose por clase del modelo honesto:

| Clase | P | R | F1 |
|---|---|---|---|
| **gun_shot** | **1.00** | **1.00** | **1.00** |
| air_conditioner | 0.88 | 0.97 | 0.92 |
| street_music | 0.96 | 0.89 | 0.92 |
| dog_bark | 0.88 | 0.92 | 0.90 |
| siren | 0.83 | 0.91 | 0.87 |
| children_playing | 0.81 | 0.92 | 0.86 |
| car_horn | 0.93 | 0.75 | 0.83 |
| engine_idling | 0.91 | 0.71 | 0.80 |
| jack_hammer | 0.80 | 0.75 | 0.77 |
| **drilling** | **0.68** | **0.71** | **0.69** |

**`gun_shot` mantiene precision y recall de 1.00 incluso con el split limpio**: 35 de 35, sin un solo falso positivo. Es la clase que en la Parte 1 era el destino por defecto de todos los transitorios impulsivos, con precision de 0.37 en M3.

La explicación está en [AudioSet](/papers/audioset-gemmeke-2017): su ontología de 527 clases incluye *"Gunshot, gunfire"* como categoría propia, entrenada sobre miles de ejemplos de YouTube. El embedding **ya trae ese detector**. El fine-tuning no tuvo que aprenderlo, solo enrutarlo hacia la salida correcta. Eso es lo que significa transferir: no acelerar el aprendizaje de la tarea, sino no tener que aprender la parte difícil.

Y las debilidades que persisten son acústicamente razonables: **`drilling` (F1 0.69) y `jack_hammer` (0.77)**, el par taladro/martillo neumático que se confundía en todos los modelos anteriores del lab. Dos herramientas eléctricas de percusión con espectros de banda ancha y patrones de repetición similares siguen siendo el límite, con AudioSet o sin él.

## Síntesis del laboratorio

El práctico pide entrenar M5, opcionalmente M18, y hacer fine-tuning de VGGish. Cumplido eso, lo que queda es un caso de estudio sobre **cómo se acumulan los errores de medición**:

| Efecto | Magnitud | Se corrige con |
|---|---|---|
| Fuga de folds (`glob` con `str(lista)`) | +11 a +24 puntos, escala con la capacidad | una línea de `glob` |
| Preprocesamiento (rates, aliasing, recorte, normalización) | −11 puntos, constante entre modelos | resamplear antes de decimar |
| `lr = 0.01` sobre 18 capas | −23 puntos, escala con la profundidad | el default de Adam |
| `ReLU` sobre los logits | −27 puntos, mata el 27 % de las clases | `nn.Identity()` |

Ninguno es visible desde la métrica global, y dos de ellos —la fuga y el learning rate— **invierten conclusiones**: hacen que M18 parezca peor que M5, y que M3 parezca reproducir el paper cuando en realidad está once puntos por debajo y once por encima al mismo tiempo.

---

**Volver al** [índice del lab 39](/laboratorios/lab-39) · **Clase relacionada:** [Clase 39 - Modelos de Deep Learning para Audio](/clases/clase-39)
