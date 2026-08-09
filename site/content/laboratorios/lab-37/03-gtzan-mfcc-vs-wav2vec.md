---
title: "GTZAN: MFCC vs wav2vec"
weight: 3
math: true
---

Acá se junta todo. La Parte 2 entrena un clasificador de géneros musicales sobre **GTZAN** y plantea la pregunta central del laboratorio: ¿rinde mejor un modelo alimentado con **features de fórmula** (MFCC) o con **embeddings preentrenados** (wav2vec 2.0)?

El notebook lo anuncia con una advertencia: *"que gane el preentrenado no está garantizado: wav2vec aprendió de habla, y esto es música"*. Los números dicen que wav2vec gana — y al mirar por qué, la razón no es la que sugiere el enunciado.

## El pipeline, y los cuatro matices del audio

La Parte 2 usa el mismo esqueleto de siempre —`Dataset`, `DataLoader`, modelo, loop— con cuatro particularidades:

| Matiz | Dónde aparece |
|---|---|
| Los archivos hay que **decodificarlos** | El `Dataset` de GTZAN, en cada `__getitem__` |
| Los largos **varían**, el batching necesita ayuda | El error de `stack` y el `collate_fn` |
| La señal cruda pasa por una **transform** antes del modelo | La función `extraer`, que corre en GPU |
| El `collate_fn` suma **ruido solo en entrenamiento** | El flag `train=True` |

### El dataset, y lo que dicen sus números

```python
d_train = GTZAN(root=".", folder_in_archive="genres_original", subset="training")
# train: 443 | val: 197 | test: 290
# total: 930 de las 1000 canciones del dataset
```

| Split | Clips | % | Clips por género |
|---|---|---|---|
| train | 443 | 47,6% | **44,3** |
| val | 197 | 21,2% | 19,7 |
| test | 290 | 31,2% | 29,0 |

Tres lecturas que condicionan todo lo que sigue:

- **44 clips por género para entrenar.** Es un régimen de muy pocos datos, y explica por qué el notebook insiste tanto con augmentation. La referencia al leer la accuracy no es "90% o fracasé": es **10% de azar**.
- **El split es 48/21/31**, no el habitual 80/10/10. Menos de la mitad de los datos se usan para entrenar.
- **Se descartaron 70 canciones (7%).** torchaudio usa el split "filtered", diseñado para mitigar los duplicados conocidos de GTZAN — un dataset tan usado como criticado, con clips repetidos, algunos mal etiquetados y artefactos de grabación que correlacionan con el género.

{{< callout type="warning" >}}
**`d_val` se crea y no se usa nunca más.** Ninguna celda posterior lo menciona: el entrenamiento evalúa contra `d_test` **en cada época** e imprime la curva. Con 10 épocas fijas no hay selección de modelo, así que técnicamente no se elige nada mirando test — pero si a partir de esa curva alguien decidiera "me quedo con la época 7", estaría haciendo selección de modelo sobre el conjunto de prueba. El `d_val` existe precisamente para eso y quedó sin usar.
{{< /callout >}}

### El batch que falla

```python
data = DataLoader(d_train, batch_size=20, shuffle=True)
next(iter(data))
# RuntimeError: stack expects each tensor to be equal size,
#               but got [1, 661794] at entry 0 and [1, 661504] at entry 1
```

Una celda que **falla a propósito**, y el `try/except` está ahí porque el error *es* el contenido. Sin `collate_fn`, el `DataLoader` usa `default_collate` → `torch.stack`, que exige formas idénticas.

Los dos largos difieren en **290 muestras = 13 milisegundos**. Trece milésimas de segundo bastan: `stack` no admite ninguna diferencia. Es la manifestación concreta de la asimetría de la Parte 1 — las filas de un espectrograma son un hiperparámetro, las columnas dependen del audio. Un dataset de imágenes se arregla con un `Resize`; en audio no hay equivalente natural, porque recortar cambia el contenido y estirar cambia el tempo, que en música *es* información.

### El `collate_fn`

```python
def new_collate(batch, max_values=660000, train=False):
    for s, sr_b, label in batch:
        w = s[0][:max_values]                              # (1,N) -> (N,), truncado
        if train:
            snr = torch.empty(1).uniform_(10, 20)          # sorteado POR MUESTRA
            w = Fa.add_noise(w.unsqueeze(0), torch.randn(1, w.shape[0]), snr=snr)[0]
        samples.append(w); labels.append(a_indice[label])  # "country" -> 2
    return torch.stack(samples), torch.tensor(srs), torch.tensor(labels)
```

Resuelve tres problemas de una vez: **truncar** al largo común (660.000 muestras = 29,93 s, descartando 81 ms), **mapear** las etiquetas de texto a índices porque `CrossEntropyLoss` quiere enteros, y **aumentar** solo en entrenamiento.

{{< callout type="warning" >}}
**Trunca pero no rellena.** Si algún clip midiera menos de 660.000 muestras, `stack` volvería a fallar igual. Funciona porque GTZAN es homogéneo, no porque el código sea robusto. Compáralo con el `collate` de Speech Commands en la celda opcional, que sí maneja los dos casos:

```python
x = torch.nn.functional.pad(w[0], (0, max(0, LARGO_SC - w.shape[1])))[:LARGO_SC]
#     ^ rellena si falta                                              ^ trunca si sobra
```
{{< /callout >}}

Un detalle: `srs` se devuelve y **nunca se usa** — en el loop de entrenamiento se desempaqueta y se ignora. El modelo nunca ve el sample rate.

### El modelo

```python
class ModeloRNN(nn.Module):
    def __init__(self, input_size, hidden_size=100, n_hidden_layers=2, n_classes=10):
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=n_hidden_layers,
                          batch_first=True, dropout=0.5)
        self.cls = nn.Linear(hidden_size, n_classes)
    def forward(self, x):
        output, h = self.rnn(x)
        return self.cls(h[-1])
```

| Entrada | Parámetros |
|---|---|
| MFCC (`input_size=40`) | **104.210** |
| wav2vec (`input_size=768`) | **322.610** |

Con 443 ejemplos son ~235 parámetros por ejemplo: el sobreajuste no es un riesgo, es el escenario por defecto.

{{< callout type="info" >}}
**La decisión discutible: `h[-1]`.** La GRU devuelve `output` con el estado en **cada** paso (`(batch, 162, 100)`) y `h` con el estado **final** de cada capa. El modelo clasifica con `h[-1]`, o sea **descarta 161 de los 162 estados**. Todo el clip de 30 segundos tiene que caber en el vector que la GRU produce en el último paso — y las RNN olvidan lo lejano, así que ese vector está sesgado hacia el final de la canción.

Para clasificar género eso es especialmente cuestionable: **el género es una propiedad global del clip**, no algo que ocurra en el segundo 30. La alternativa es una línea —`self.cls(output.mean(dim=1))`— no agrega parámetros y en clasificación de audio suele dar varios puntos.
{{< /callout >}}

### La ventana de 372 ms, y por qué

```python
t_mfcc = TA_MFCC(sample_rate=22050, n_mfcc=40,
                 melkwargs={"win_length": 8192, "n_fft": 8192}).to(device)
```

| | Este MFCC | Estándar de voz |
|---|---|---|
| Ventana | **372 ms** | 25 ms |
| Hop | 186 ms | 10 ms |
| Resolución frecuencial | **2,69 Hz** | ~40 Hz |
| Pasos para la GRU | **162** | ~2.993 |

Hay una justificación musical —2,69 Hz permite distinguir notas vecinas incluso en la región grave, donde un semitono son ~6 Hz— pero la razón determinante está en la última fila. Con ventanas de 25 ms, un clip de 30 segundos daría **casi 3.000 pasos de secuencia**, inviable para una GRU. **La ventana grande está elegida para que la secuencia quepa en la RNN**: el compromiso de Gabor usado como herramienta de ingeniería.

Y ahí se entiende el comentario del propio notebook: *"la próxima clase lo trataremos como imagen, con CNNs"*. Una CNN sobre el espectrograma procesa los 3.000 frames en paralelo, como píxeles. La ventana de 8192 es una concesión a la arquitectura recurrente.

## Las dos corridas

```
MFCC                                wav2vec
Epoca 01 | loss 2.333 | 8.97%       Epoca 01 | loss 2.309 | 13.45%
Epoca 05 | loss 2.243 | 15.86%      Epoca 05 | loss 2.204 | 27.59%
Epoca 10 | loss 2.152 | 19.31%      Epoca 10 | loss 1.995 | 28.62%
```

![Curva de accuracy en test a lo largo de 10 épocas para wav2vec, ascendente desde 13% hasta cerca de 29% con oscilaciones en las últimas épocas](/laboratorios/lab-37/curva-wav2vec.png)

**La referencia que hace legibles estos números: la loss de azar con 10 clases es $\ln(10) = 2{,}3026$.** El MFCC **arranca en 2,333 —por encima del azar—** y termina en 2,152: baja apenas **7,8% en 10 épocas**. Esa loss final equivale a una perplejidad de $e^{2{,}152} = 8{,}6$: el modelo se comporta como si eligiera entre 8,6 clases de las 10.

Con 443 clips y batch de 20 son 23 batches por época: **230 pasos de gradiente en total** con `lr=1e-4`. Es un presupuesto de optimización muy chico, y explica el estancamiento de la curva desde la época 6 (18,97 → 16,90 → 19,66 → 18,62 → 19,31: oscilación sin tendencia).

## La matriz que separa dos efectos

Medir train y test **en las dos condiciones** —con y sin ruido— cambia por completo el diagnóstico:

|  | MFCC limpio | MFCC + ruido | wav2vec limpio | wav2vec + ruido |
|---|---|---|---|---|
| **Train** | 20,09% | 39,95% | **35,21%** | 39,28% |
| **Test** | **19,31%** | 31,03% | **28,62%** | 31,38% |
| Brecha train−test | **+0,78 pp** | +8,92 pp | **+6,59 pp** | +7,90 pp |

{{< callout type="error" >}}
**Con MFCC no hay sobreajuste, y la brecha de 2× que reporta el notebook es un artefacto de medición.** La celda de cierre hace `test(model, train_loader)`, y ese loader tiene la augmentación **activada**, mientras el test se evalúa limpio. Son dos distribuciones distintas, no train contra test.

Medidos en la misma condición: **20,09% en train limpio contra 19,31% en test — 0,78 puntos.** Son el mismo número. El modelo generaliza perfectamente; lo que no hace es aprender.
{{< /callout >}}

**El efecto dominante es el dominio, no el sobreajuste.** Pasar de audio limpio a audio con ruido vale entre 12 y 20 puntos en MFCC; pasar de train a test vale entre 0,8 y 8,9. La pregunta de la actividad apunta al efecto menor.

Y hay un detalle fino: **el sobreajuste solo aparece donde el modelo funciona.** En audio limpio el MFCC está a 1,93-2,01× el azar en ambos conjuntos — no puede haber brecha donde no hay desempeño que perder. En el dominio que sí conoce (4,0× azar en train) aparecen los 8,9 puntos.

## La causa: la GRU arranca saturada

Al medir qué llega efectivamente a la red, aparece la explicación de todo:

|  | MFCC | wav2vec |
|---|---|---|
| Norma del vector de entrada | 204,7 (limpio) / 267,3 (ruido) | **9,3 / 9,1** |
| Cambio por el ruido | **+31%** | **−2%** |
| $c_0$ medio (la energía) | **133 – 258** | 0,0 – 0,2 |
| Preactivación de la GRU | **11,8 – 15,4** | **0,5** |
| Derivada de $\tanh$ | $10^{-10}$ – $10^{-13}$ | **0,79** |

`nn.GRU` inicializa sus pesos en $U(-0{,}1;\ 0{,}1)$, calibrados para entradas de magnitud ~1. Con MFCC sin normalizar la preactivación llega a ~15, donde $\tanh$ y sigmoide están completamente planas: **la derivada es de orden $10^{-13}$ y prácticamente no fluye gradiente**. La red arranca en una región donde no puede aprender, y las 230 actualizaciones se gastan en salir de la saturación.

El culpable principal es $c_0$, el primer coeficiente cepstral, que es esencialmente la **energía global del frame**: su media es 6,4× la de $c_1$ y 150× la de $c_2$. Es la razón por la que el preprocesamiento estándar en reconocimiento de voz (**CMVN**, cepstral mean and variance normalization) lo normaliza o directamente lo descarta — y este notebook no lo hace.

{{< callout type="info" >}}
**Una sola causa explica tres observaciones:**

1. **wav2vec aprende más** (train limpio 35,21% contra 20,09%): sus embeddings vienen normalizados por LayerNorm, caen en la zona lineal y el gradiente fluye.
2. **wav2vec sufre 5× menos desajuste de dominio** (+2,1 pp por el ruido en test, contra +11,7 pp): el ruido cambia la norma de sus features solo −2%, contra +31% en MFCC. Un modelo saturado es hipersensible a la escala de su entrada.
3. **Con wav2vec aparece sobreajuste real** (6,59 pp entre train y test limpios) — no porque generalice peor, sino porque **hay algo aprendido que sobreajustar**.
{{< /callout >}}

Una hipótesis que se probó y **se refutó**: que el ruido ayudara porque comprimía el rango dinámico del log-mel y reducía la saturación. Los números dicen lo contrario — el ruido **sube** la norma de 204,7 a 267,3 y **empeora** la saturación. Lo que queda es adaptación de dominio pura: el modelo entrenó con features de magnitud ~267 y en test recibe ~205, y operando saturado, lo que determina su salida es *qué* unidades saturan y en qué dirección.

## Actividad 4

> Entrenen el modelo por 10 épocas y reporten sus resultados: ¿cómo es el rendimiento en train y test? ¿Hay sobreajuste? ¿Qué señales lo indican? Luego cambien a `wav2vec` y reentrenen. ¿Mejora o empeora?

**Parte 1 — MFCC.** No hay sobreajuste apreciable (0,78 pp entre train y test limpios). El problema dominante es el **contrario**: subentrenamiento. Señales: la loss arranca por encima del azar y baja 7,8% en 10 épocas, la perplejidad final es 8,6 de 10 clases, la accuracy se estanca desde la época 6, y son solo 230 pasos de gradiente con `lr=1e-4`. Segundo hallazgo: un **desajuste de dominio** de +11,7 puntos inducido por aplicar la augmentación al 100% de las muestras.

**Parte 2 — wav2vec mejora, en +9,31 pp de test.** Pero la causa medida es la **escala de las features**, no el preentrenamiento: los MFCC saturan la GRU desde la inicialización y los embeddings no.

{{< callout type="warning" >}}
**La comparación no es controlada: el toggle mueve tres variables a la vez.**

| | MFCC | wav2vec |
|---|---|---|
| Parámetros de la GRU | 104.210 | **322.610** (3,1×) |
| Audio que ve | **30 s** | **15 s** (`samples[:, :330000]`) |
| Pasos de secuencia | 162 | 187 |

El primer factor favorece a wav2vec, el segundo lo perjudica. La conclusión sobre la escala de features se apoya en la **medición directa de las preactivaciones**, no en la diferencia de accuracy.
{{< /callout >}}

Y un gotcha del propio notebook que vale conocer: **los resultados que trae guardados son de wav2vec, aunque el código muestre `FEATURES = "MFCC"`.** Los `#@param` de Colab son widgets que reescriben la línea del source, y el archivo quedó guardado con un valor distinto del que se ejecutó. Al comparar contra la referencia del profesor hay que tenerlo presente.

## Mejoras identificadas

Ordenadas por relación entre impacto esperado y esfuerzo:

| Mejora | Costo | Por qué |
|---|---|---|
| **Normalizar los MFCC (CMVN)** | 2 líneas | Elimina la saturación, que es la causa medida de la diferencia |
| **`output.mean(1)` en vez de `h[-1]`** | 1 línea | El género es global; usar solo el último estado descarta 161 de 162 |
| **Augmentación con probabilidad 0,5** | 1 línea | Cierra los 11,7 puntos de desajuste de dominio |
| **Trocear en fragmentos de 3 s** | ~10 líneas | Convierte 443 ejemplos en ~4.400, sin descargar nada |
| **Subir el `lr` a 1e-3** | 1 carácter | Es el que usa la celda opcional; 230 pasos a 1e-4 no alcanzan |
| **Usar SpecAugment** | 2 líneas | El notebook lo enseña y no lo usa en el entrenamiento |

## Qué nos llevamos

- **La escala de las features decide si la red puede aprender.** Preactivación 15 contra 0,5 es la diferencia entre gradiente nulo y gradiente sano.
- **Medir train y test en la misma condición**, o la brecha que reportas es un artefacto.
- **El sobreajuste solo es medible donde el modelo funciona.**
- **La ventana del MFCC estaba elegida por la arquitectura**, no por la señal: 372 ms para que la secuencia quepa en la GRU.
- **`h[-1]` descarta el 99% de los estados** de la RNN, y para una etiqueta global eso es una decisión, no un detalle.

---

**Ver tambien:** [Lab 37 — hub](/laboratorios/lab-37) · Anterior: [Data augmentation](02-data-augmentation) · Siguiente: [Transferencia y dominio](04-transferencia-y-dominio) · Papers: [GTZAN](/papers/gtzan-tzanetakis-2002) · [wav2vec 2.0](/papers/wav2vec2-baevski-2020) · Fundamentos: [Datasets de audio](/fundamentos/datasets-de-audio) · [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel) · [LSTM y GRU](/fundamentos/lstm-gru).
