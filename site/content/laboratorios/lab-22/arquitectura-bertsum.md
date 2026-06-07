---
title: "P1 — Arquitectura extractiva (BertSum)"
weight: 1
---

> **Celdas 5-30 del notebook (Parte 1).** Cómo BERT —diseñado para una o dos oraciones— se modifica para producir representaciones a nivel de oración sobre un documento entero, y el andamiaje para correr el repo original en Colab.

## El problema que BertSum resuelve

El resumen **extractivo** se modela como una **clasificación binaria por oración**: para cada oración $i$ del documento, predecir $y_i \in \{0,1\}$ — ¿va al resumen o no? La salida final es el subconjunto de oraciones con mayor score. El modelo es [BertSum](/papers/bertsum-liu-2019) (Yang Liu, 2019).

El obstáculo: **BERT directo no sirve** para esto, y entenderlo es la clave de la arquitectura.

- BERT produce representaciones contextuales **a nivel de token**, sobre **una o dos** oraciones (la entrada es *sentence A* / *sentence B* con un solo `[CLS]` al inicio).
- El resumen extractivo necesita una representación **a nivel de oración**, sobre **muchas** oraciones a la vez (en CNN/DailyMail un artículo tiene ~30-50 oraciones; medimos un promedio de 969 tokens).

## La solución de Liu (Figura 1 del paper)

1. Insertar un token **`[CLS]` antes de cada oración** (no solo uno al inicio del documento).
2. Insertar **`[SEP]` después de cada oración**.
3. Usar **interval segment embeddings** $E_A, E_B$ que **alternan por paridad** de la oración (impares → $E_A$, pares → $E_B$), para que el modelo distinga dónde empieza y termina cada una.
4. El vector de salida de cada `[CLS]` **es** la representación de su oración.
5. Apilar una **capa de resumen** encima de esos vectores `[CLS]`. El lab usa la variante `classifier` (lineal + sigmoide); la mejor variante del paper es un **Transformer inter-oraciones de 2 capas**.
6. **Loss BCE** contra un *oracle* (ver [estrategia de entrenamiento](entrenamiento-bertsum)).
7. En inferencia: rankear por score y aplicar **trigram blocking** (ver [inferencia](inferencia-extractiva)).

Con esto BertSum logró ROUGE-1/2/L = **43.25 / 20.24 / 39.63** en CNN/DailyMail, batiendo el estado del arte previo (~38 en ROUGE-L). Conecta con el [fundamento Text Summarization](/fundamentos/text-summarization) y el detalle de [BERT](/fundamentos/bert).

> **Gotcha histórico:** el lab clona un **fork** del repo oficial ([`fdelrio89/BertSum`](https://github.com/fdelrio89/BertSum), del profesor), no el original de nlpyang. El fork tiene parches de compatibilidad para Colab moderno.

## Setup del entorno (celdas 7-18)

El preámbulo reconstruye un entorno de 2019 sobre Colab actual, lo que explica casi toda la fragilidad de instalación:

```python
import warnings
warnings.filterwarnings('ignore')   # silencia los DeprecationWarning de las libs viejas
```

```python
!pip install -qqq --force-reinstall urllib3 folium   # workaround cosmético de Colab
!pip install -qqq pytorch_pretrained_bert tensorboardX pyrouge
```

| Paquete | Qué es | Por qué aquí |
|---|---|---|
| `pytorch_pretrained_bert` | El **ancestro** de `transformers` (v0.x, ~2018) | BertSum se escribió contra esta API, antes de que existiera HuggingFace `transformers` |
| `tensorboardX` | Puente TensorBoard para PyTorch antiguo | Logging de entrenamiento |
| `pyrouge` | Wrapper Python sobre el script **Perl** ROUGE-1.5.5 | Cálculo "oficial" de ROUGE. Notoriamente frágil (la Parte 2 usa `evaluate` en Python puro) |

El `--force-reinstall` de urllib3/folium genera un `ERROR: pip's dependency resolver...` con conflictos de `requests`/`numpy` — **benigno**: pip resuelve lo pedido y solo avisa que las versiones no satisfacen todas las restricciones cruzadas del entorno Colab. No afecta al modelo.

### Clonado idempotente

```bash
![ ! -d "/content/BertSum" ] && git clone --progress https://github.com/fdelrio89/BertSum.git
```

El patrón `[ ! -d ... ] && comando` es **idempotente**: si re-ejecutas la celda, no intenta volver a clonar. Aparece repetido en todo el preámbulo (descomprimir, descargar).

### Descarga del checkpoint con balanceo de carga casero

```python
if not os.path.exists('/content/lab4.zip'):
    if random.random() > 0.5:
        !gdown --no-cookies 18ecT1r9jFGyChATPwTdRgvj9dKctd3sS -O /content/lab4.zip
    else:
        !gdown --no-cookies 1tGez67YBFkZqY3qyCCul1Wex96PKkeqo -O /content/lab4.zip
```

El `random.random() > 0.5` elige entre **dos espejos idénticos** del mismo archivo (`lab4.zip`, ~2.08 GB, contiene `model.pt`). ¿Por qué? Google Drive impone **cuota de descarga por archivo**: si 40 alumnos bajan el mismo a la vez, lo bloquea 24h. Repartir el tráfico 50/50 **duplica la cuota efectiva**. Es load balancing artesanal con una moneda. `gdown` (no `wget`) maneja el handshake de confirmación de virus de Drive (`&confirm=t`).

El `model.pt` se mueve a `BertSum/models/` (donde el repo lo espera) y `bertsum_data.zip` trae los datos **ya preprocesados**: oraciones tokenizadas con `[CLS]`/`[SEP]`, segment embeddings asignados y **etiquetas oracle** ya calculadas (ver [entrenamiento](entrenamiento-bertsum)).

## La clase `Args` (celda 20)

```python
os.chdir('/content/BertSum/src/')   # crítico: el repo usa rutas relativas ../bert_data, ../models
```

El código original de BertSum se ejecuta desde **línea de comandos** y parsea opciones con `argparse`. En un notebook no hay CLI, así que se **emula** ese objeto con una clase de atributos:

```python
class Args():
    encoder='classifier'        # la variante más simple: lineal + sigmoide por [CLS]
    mode='test'                 # solo inferencia, no entrenamiento
    use_interval=True           # activa los interval segment embeddings E_A/E_B
    hidden_size=128; ff_size=512; heads=4; inter_layers=2   # capa de resumen
    block_trigram=True          # anti-redundancia en inferencia
    visible_gpus=-1
    ...
```

| Param clave | Valor | Significado |
|---|---|---|
| `encoder` | `'classifier'` | Capa de resumen más simple (lineal + sigmoide sobre cada `[CLS]`) |
| `mode` | `'test'` | Solo inferencia — coherente con cargar `model.pt` pre-entrenado |
| `block_trigram` | `True` | Activa el trigram blocking en inferencia |
| `use_interval` | `True` | Activa los segment embeddings alternados por paridad |
| `batch_size` | 1000 | Se mide en **tokens**, no documentos |

Los parámetros de optimización (`lr=1`, `optim='adam'`, `warmup_steps=8000`) están presentes pero **dormidos** porque `mode='test'`. El `lr=1` es solo el multiplicador base del scheduler Noam (ver [entrenamiento](entrenamiento-bertsum)).

> **El punto de toda la celda:** estamos en modo **`test` + `classifier` + `block_trigram=True`** — la variante más simple, solo inferencia, con anti-redundancia activada.

## Cargar y sincronizar el checkpoint (celdas 29-35)

```python
model_flags = ['hidden_size','ff_size','heads','inter_layers','encoder','ff_actv','use_interval','rnn_size']
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=lambda storage, loc: storage)
opt = vars(checkpoint['opt'])
for k in opt.keys():
    if k in model_flags:
        setattr(args, k, opt[k])   # sobrescribe SOLO los flags de arquitectura con los del checkpoint
```

El patrón: **confía en el checkpoint para la *forma* del modelo, confía en tu `Args` para *cómo lo ejecutas*.** Solo los 8 `model_flags` (que determinan la forma de los tensores) se sincronizan; los paths y `block_trigram` se conservan.

- **`weights_only=False`**: permite deserializar `checkpoint['opt']` (objeto, no solo tensores). El default cambió a `True` en PyTorch 2.6 por seguridad; ponerlo explícito revela que el notebook fue **actualizado** para PyTorch moderno — la costura entre "código 2019" y "Colab 2025".
- **`map_location=lambda storage, loc: storage`**: fuerza carga en CPU primero, ignorando que el checkpoint se guardó en GPU. Patrón defensivo estándar.

Finalmente, construir la estructura y cargar pesos:

```python
config = BertConfig.from_json_file(args.bert_config_path)
model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
model.load_cp(checkpoint)
```

`load_pretrained_bert=False` evita descargar el BERT genérico: vamos a cargar **nuestro** checkpoint, que ya tiene los pesos de BERT *fine-tuneados para summarization*. Estructura vacía (con la forma correcta gracias a los `model_flags`) + inyección de pesos = modelo funcional.

---

**Siguiente:** [estrategia de entrenamiento de BertSum](entrenamiento-bertsum) · [inferencia y trigram blocking](inferencia-extractiva)
