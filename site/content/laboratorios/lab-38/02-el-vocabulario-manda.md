---
title: "El vocabulario manda"
weight: 2
---

Las celdas 27 a 42 cargan el modelo y lo ejecutan sobre dos videos. El código es trivial —tres líneas— pero el contraste entre los dos resultados es el hallazgo central de la parte guiada del lab: **un modelo pre-entrenado sólo puede responder dentro de su vocabulario, y no tiene forma de avisar cuando la respuesta correcta no está ahí.**

## Las 400 etiquetas y su vínculo puramente posicional

```python
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
  labels = [line.decode("utf-8").strip() for line in obj.readlines()]
print("Found %d labels." % len(labels))     # Found 400 labels.
```

`labels` es una **lista**, y la correspondencia entre etiqueta y clase es exclusivamente **posicional**: `labels[i]` nombra la clase cuyo logit ocupa la posición `i`. No hay ningún metadato en el `SavedModel` que valide esa correspondencia.

Si el archivo estuviera ordenado distinto que el orden de entrenamiento, el modelo funcionaría **perfectamente** y todas las predicciones saldrían **mal etiquetadas**, sin ningún error visible. Es un fallo silencioso clásico en despliegue de modelos. El orden real es alfabético, lo que da una verificación gratis:

```python
labels[0], labels[-1], labels == sorted(labels)     # 'abseiling', 'zumba', True
```

Ese `labels[0] == 'abseiling'` importa: el video de la actividad se llama `abseiling_k400.mp4`, así que la clase correcta es el **índice 0**.

## `hub.load` y qué hay dentro

```python
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']
```

Descarga ~50 MB, los descomprime en `/tmp/tfhub_modules/` y extrae la `ConcreteFunction` exportada:

```
entrada : float32  [batch, frames, 224, 224, 3]
salida  : dict     {'default': float32 [batch, 400]}   ← logits, no probabilidades
```

Lo que hay dentro es la **rama RGB** de I3D: una Inception-v1 inflada, **~12,1 M de parámetros**. La comparación que la [clase 38](/clases/clase-38/teoria) desarrolla en detalle: C3D necesitaba **78 M** entrenados desde cero. I3D es 6,5 veces más chico y mucho mejor, y esa es toda la tesis del [inflado](/fundamentos/inflado-de-convoluciones).

{{< callout type="warning" >}}
**Nota de entorno.** `tfhub.dev` dejó de alojar modelos el 15-nov-2023 y redirige a Kaggle Models; los assets no migrados se borraron el 18-mar-2024. `i3d-kinetics-400` sí está migrado y la URL antigua sigue resolviendo desde Colab. Si fallara, el síntoma no es un 404 limpio sino `tarfile.ReadError: not a gzip file`, porque `hub.load` intenta descomprimir una página HTML. Las alternativas son la URL de Kaggle (`kaggle.com/models/deepmind/i3d-kinetics/frameworks/tensorFlow1/variations/400/versions/1`) o `kagglehub.model_download(...)` + `tf.saved_model.load`.

Aparte: `tensorflow_hub 0.16.1` **exige** el paquete `tf-keras` instalado y falla en el propio `import` si no está, pese a que este lab no usa Keras en absoluto.
{{< /callout >}}

## `predict`: dónde ocurre realmente el promedio

```python
def predict(sample_video):
  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]
  logits = i3d(model_input)['default'][0]
  probabilities = tf.nn.softmax(logits)
  for i in np.argsort(probabilities)[::-1][:5]:
    print(f"  {labels[i]:22}: {probabilities[i] * 100:5.2f}%")
```

**`[tf.newaxis, ...]`** agrega el eje de batch: `(164, 224, 224, 3)` → `(1, 164, 224, 224, 3)`, el layout `NDHWC`. I3D es totalmente convolucional en espacio **y en tiempo**, así que acepta 100, 164 o 250 frames sin cambiar nada: la red reduce el eje temporal por un factor de 8 y un *average pooling* colapsa lo que quede. El piso son ~9-10 frames, por debajo de los cuales la dimensión temporal se agota tras los strides.

**El softmax es casi decorativo.** Es estrictamente monótono, así que el orden del top-5 sería idéntico sin él; sólo existe para reportar "98,13 %" en vez de "logit 12,4". Y ese porcentaje no debe leerse como confianza calibrada — las dos secciones siguientes lo demuestran con datos.

**El promedio temporal es de logits, no de probabilidades.** Ocurre dentro de la red, antes de la capa final:

$$\text{softmax}\left(\frac{1}{T}\sum_t z_t\right) \neq \frac{1}{T}\sum_t \text{softmax}(z_t)$$

Promediar logits equivale a una media *geométrica* de probabilidades: un solo segmento muy negativo puede vetar una clase. Es más conservador que promediar probabilidades, y explica por qué la ventana temporal pesa tanto en [dónde está la evidencia](../04-donde-esta-la-evidencia).

## Caso 1: la clase no existe

![Frame del video ApplyEyeMakeup de UCF101](/laboratorios/lab-38/applyeyemakeup.jpg)

```
Top 5 actions:
  filling eyebrows      : 98.13%
  applying cream        :  1.57%
  waxing eyebrows       :  0.17%
  playing harmonica     :  0.07%
  brush painting        :  0.04%
```

**Kinetics-400 no tiene la clase "applying eye makeup".** Lo más cercano que existe entre las 400 es `filling eyebrows` (126), `applying cream` (4), `waxing eyebrows` (388), `curling hair` (80), `doing nails` (97). El modelo **no puede acertar**.

Y sin embargo responde con **98,13 %** de confianza. Ese es el punto: el softmax obliga a que las 400 probabilidades sumen 1, así que toda la masa se va al vecino semántico más próximo, sin ninguna señal de que la respuesta correcta no estaba disponible.

En descargo del modelo: `filling eyebrows` es funcionalmente casi la misma acción visual —primer plano de una cara, una mano con un aplicador cerca del ojo, movimientos pequeños y repetidos—. El modelo **no está viendo mal**. Está viendo bien y reportando la etiqueta más cercana que su espacio de salida permite. La falla no es de percepción sino de **vocabulario**, y no se arregla con ningún ajuste numérico: hace falta reemplazar la capa final y hacer fine-tuning. Es literalmente lo que hace *[Quo Vadis](/papers/i3d-carreira-2017)* —cambiar 400 salidas por 101— antes de reportar su 98,0 % en UCF101.

## Caso 2: la clase existe

![Frame del video archery](/laboratorios/lab-38/archery.jpg)

```
Top 5 actions:
  archery                     : 99.28%
  throwing axe                :  0.32%
  flying kite                 :  0.11%
  catching or throwing frisbee:  0.05%
  pole vault                  :  0.04%
```

`archery` es el índice 5 de Kinetics-400, y el modelo lo clava. Con el preproceso corregido y el video completo sube a **99,97 %**.

Lo interesante es la **estructura de los perseguidores**: `throwing axe`, `flying kite`, `catching or throwing frisbee`, `pole vault`. Las cuatro comparten la misma gramática de movimiento —brazos extendidos, objeto alargado, gesto de tensar o lanzar—. El espacio de representación agrupa acciones por **patrón cinemático**, no por contexto ni por objeto. Es lo que se espera de filtros espacio-temporales aprendidos end-to-end, y lo que los enfoques previos no lograban: la CNN 2D con pooling de [Karpathy](/papers/large-scale-video-karpathy-2014) ignora el orden, [Two-Stream](/papers/two-stream-simonyan-2014) precomputa el flujo fuera de la red, y [LRCN](/papers/lrcn-donahue-2015) colapsa cada frame a un vector antes de que el LSTM lo vea.

Nótese además que este 99,28 % salió con los **primeros 100 de 300 frames** y perdiendo el 25 % del ancho por el crop. La evidencia era tan robusta que el preproceso subóptimo no la tocó — algo que no se repetirá en la actividad.

## Las dos predicciones, lado a lado

| | `ApplyEyeMakeup` | `archery` |
|---|---|---|
| ¿La clase está en Kinetics-400? | **no** | **sí** (índice 5) |
| Top-1 | `filling eyebrows` 98,13 % | `archery` 99,28 % |
| Entropía | 0,104 nats | 0,060 nats |
| ¿Es correcta? | **imposible** | sí |

**Las dos predicciones son igual de seguras. Una es correcta y la otra no puede serlo.** El modelo no tiene forma de expresar "esta acción no está en mi vocabulario", y ninguna métrica derivada del softmax lo delata: la entropía de `ApplyEyeMakeup` (0,104) es la *segunda más baja* de todo el lab.

Es fallo de calibración bajo cambio de dominio, y tiene consecuencia operativa directa: un filtro de "aceptar si la confianza supera el 95 %" dejaría pasar este caso sin objeción. El mismo patrón que el [lab 30](/laboratorios/lab-30) documentó en KV-MemNN y el [lab 24](/laboratorios/lab-24) en QA generativo.

---

**Siguiente:** [El bug del preproceso](../03-el-bug-del-preproceso) — la actividad falla y el diagnóstico encuentra un desajuste en el tutorial oficial.
