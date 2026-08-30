---
title: "La arqueología del checkpoint"
weight: 3
---

El laboratorio descarga tres archivos de pesos y los carga con una función de siete líneas que imprime `*** model has been successfully loaded! ***`. Ese mensaje no prueba nada — y auditar lo que realmente ocurre convierte los checkpoints en la evidencia física del procedimiento de entrenamiento descrito en el paper.

## La función que no puede fallar

```python
def reload_model(model, path=""):
  model_dict = model.state_dict()
  pretrained_dict = torch.load(path)
  pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
  print('*** model has been successfully loaded! ***')
  return model
```

La tercera línea **filtra silenciosamente toda clave del checkpoint que no exista en el modelo**. El propósito legítimo es tolerar diferencias menores. El modo de falla es brutal:

{{< concept-alert type="cuidado" >}}
Si los nombres de las claves no coincidieran en absoluto —por ejemplo, un checkpoint guardado desde `nn.DataParallel`, que antepone `module.` a todo— el diccionario filtrado quedaría **vacío**. `model_dict.update({})` no haría nada. `load_state_dict` cargaría el modelo **con sus pesos aleatorios de inicialización**. Y la función imprimiría, impasible, que la carga fue exitosa.

El resultado sería una accuracy de ~0,2 % —el azar sobre 500 clases— **sin una sola excepción en todo el notebook**.
{{< /concept-alert >}}

Y el `strict=True` que trae `load_state_dict` por defecto no ayuda: como `model_dict` conserva todas sus claves originales, nunca falta ninguna y la validación siempre pasa. **La red de seguridad está desactivada por construcción.**

## La auditoría

```
audio   | 120/156 claves cargadas | 36 descartadas | 20 sin cargar |  12,500,340 params
video   | 204/240 claves cargadas | 36 descartadas | 37 sin cargar |  29,025,460 params
fusión  |  18/ 18 claves cargadas |  0 descartadas |  0 sin cargar |  13,107,700 params
```

Comparando además tensor a tensor contra el archivo: `True` en los tres. **La carga fue real**, y por tanto la evaluación posterior es válida.

### Las "sin cargar" son benignas, y la cuenta lo demuestra

Todas son `num_batches_tracked`, un buffer agregado a `BatchNorm` en **PyTorch 0.4.1**, posterior a estos checkpoints. Su única función es llevar un contador para el promedio acumulativo cuando se usa `momentum=None`; con el `momentum=0.1` por defecto **no se consulta jamás**, y en `eval()` BatchNorm usa `running_mean` y `running_var`, que sí se cargaron.

Que sean exactamente esas y ninguna más lo confirma el conteo de capas:

| Modelo | Capas BatchNorm en el código | "Sin cargar" |
|---|---|---|
| **Audio** | 1 (`frontend1D`) + 16 (8 bloques × 2) + 3 (downsample de layers 2–4) | **20** ✓ |
| **Video** | 1 (`frontend3D`) + 32 (16 bloques × 2) + 3 (downsample) + 1 (`bnfc`) | **37** ✓ |

Cuadra al dígito. No falta ningún peso real.

## Las 36 claves descartadas

Exactamente 36 sobran en el checkpoint de audio y exactamente 36 en el de video. Que el número coincida en dos arquitecturas distintas —ResNet-18 1D contra ResNet-34 2D— significa que lo que sobra **no depende del stream: es una estructura común adosada a ambos**.

Al imprimirlas, aparecen **dos** estructuras muertas, no una.

### 1. El backend temporal-convolucional de la fase 1

```
backend_conv1.0.weight   (1024, 512, 5)   ← Conv1d, kernel temporal 5
backend_conv1.4.weight   (2048, 1024, 5)  ← Conv1d, kernel temporal 5
backend_conv2.0.weight   (512, 2048)      ← Linear
backend_conv2.3.weight   (500, 512)       ← clasificador a 500 clases
```

Dos convoluciones 1D sobre el eje temporal seguidas de dos capas densas que terminan en 500 clases. Es palabra por palabra la sección 4.3.1 del paper:

> *"Initially, a **temporal convolutional back-end** is used instead of the 2-layer BGRU. The combination of ResNet and temporal convolution (**together with a softmax output layer**) is trained until there is no improvement in the classification rate on the validation set for more than 5 epochs. Then the temporal convolutional back-end is **removed** and the BGRU back-end is attached."*

**"Removed" del grafo de cómputo, pero nunca del archivo.**

### 2. Un BiLSTM de una versión anterior del código

```
lstm.forwardModule1.weight_ih_l0    (2048, 512)
lstm.backwardModule1.weight_ih_l0   (2048, 512)
lstm.forwardModule2 ...  lstm.backwardModule2 ...
lstm.fc.weight                      (500, 1024)
```

La forma lo prueba: **2048 = 4 × 512**, o sea **cuatro compuertas** por unidad — input, forget, cell, output. Un GRU tendría 3 × 512 = 1536. Es inequívocamente LSTM.

Esto explica un detalle del notebook que parecía un descuido: la celda de carga imprime `print("reload LSTM model")` para un objeto que es un `GRU`. **No era un error de copia-pega — era literalmente correcto en la versión anterior, y el modelo entero sigue guardado en el archivo.**

Hay además un detalle de topología. Mirando las formas del stream de video:

```
forwardModule1.weight_ih_l0   (2048, 256)   ← entrada: los 256 de la ResNet-34
forwardModule2.weight_ih_l0   (2048, 512)   ← entrada: 512, NO 1024
fc.weight                     (500, 1024)   ← salida: 1024 = 512 + 512
```

Si fuera un `nn.LSTM(bidirectional=True)` de PyTorch, la segunda capa recibiría **1024** — la concatenación de ambas direcciones de la capa 1. Recibe 512. Eran **dos pilas independientes de 2 capas**, una hacia adelante y otra hacia atrás, que solo se juntan al final. Es una topología distinta a la del `nn.GRU` bidireccional del código actual, donde las direcciones se mezclan entre capas.

**Cambiaron de LSTM a GRU y de topología, y el paper no menciona ninguna de las dos cosas.**

## El 39 % del ZIP es peso muerto

| Checkpoint | Parámetros vivos | Parámetros muertos | % muerto |
|---|---|---|---|
| Audio | 12,5 M | **23,3 M** | **65 %** |
| Video | 29,0 M | 11,5 M | 28 % |
| Fusión | 13,1 M | 0 | 0 % |
| **Total** | 54,6 M | **34,9 M** | **39 %** |

El checkpoint de audio es el caso extremo: **casi dos tercios del archivo son pesos que la red nunca toca**. El culpable principal es `backend_conv1.4`, una `Conv1d(1024 → 2048, kernel=5)` que por sí sola son 10,5 M de parámetros.

Y el `concat_model` con **0 % muerto** es coherente con la historia: es el único de los tres que nunca tuvo una fase previa. Nació ya como BiGRU de fusión, en la tercera etapa.

## Por qué esto responde la Actividad 2

La pregunta del laboratorio es *"¿por qué el entrenamiento del modelo se hace por etapas?"*, y con esto no hace falta citar el paper — se puede mostrar el `state_dict`.

El procedimiento completo es:

1. **Fase 1** — cada ResNet se entrena con el backend temporal-convolucional y una capa softmax, hasta que la validación deja de mejorar.
2. **Fase 2** — se remueve ese backend, se conecta el BiGRU y se entrena solo el BiGRU por 5 épocas, con la ResNet congelada.
3. **Fase 3** — se destraba todo y se entrena end-to-end con Adam (lr 3e-4, batch 36 por stream; lr 1e-4, batch 18 para el modelo audiovisual).

La razón de fondo es de optimización: **una recurrente montada sobre una ResNet sin entrenar converge mal**, porque el gradiente que llega a las capas convolucionales debe atravesar 29 pasos temporales de compuertas. Un backend convolucional es mucho más fácil de optimizar, así que primero se estabiliza el extractor de features y después se sustituye el backend. El paper lo dice sin rodeos: *"Directly training end-to-end each stream leads to suboptimal performance."*

**El archivo `.pt` conserva la arqueología de su propio entrenamiento.**

## Dos incompatibilidades de entorno

`torch.load(path)` se llama sin argumentos, y dos cosas cambiaron desde 2018:

- **`weights_only`** — desde **PyTorch 2.6** el valor por defecto pasó de `False` a `True`, porque `torch.load` usa `pickle` y puede ejecutar código arbitrario al deserializar. Estos archivos son `state_dict` de tensores puros y cargan sin problema; un checkpoint con el modelo completo serializado fallaría con `UnpicklingError`.
- **`map_location`** — no se especifica, así que los tensores van al device donde estaban al guardarse. Un checkpoint de GPU cargado en una sesión de CPU daría `RuntimeError`.

---

**Anterior:** [Los dos streams](02-los-dos-streams) · **Siguiente:** [El vocabulario y los 29 errores](04-el-vocabulario-y-los-29-errores)
