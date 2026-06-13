---
title: "Decodificación, BLEU y robustez en Colab"
weight: 5
math: true
---

> **Dos historias en una página.** (A) El gotcha que ocurrió de verdad al correr el notebook: imágenes de prueba que fallan al cargarse con `UnidentifiedImageError`, y por qué la causa raíz no estaba en PIL sino en la **reputación de IP de Colab**. (B) Los conceptos de *decoding* y BLEU que el lab **menciona** (en la Actividad) pero **nunca ejecuta** en código. Esta segunda parte conecta con la [clase 23](/clases/clase-23) y con los fundamentos de generación.

---

## Parte A — Robustez de carga de imágenes (gotcha real)

El notebook carga cada imagen de prueba con una sola línea:

```python
from PIL import Image
import requests

image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
```

En Colab, **varias URLs fallan** con un error críptico:

```
PIL.UnidentifiedImageError: cannot identify image file <...>
```

Lo confuso es que esas mismas URLs **abren bien en el navegador local**. El error de PIL apunta al archivo, pero el archivo no es el problema.

### Causa raíz verificada

El CDN comercial `i0.wp.com` (el proxy de imágenes de WordPress / Jetpack) responde distinto **según quién pregunta**:

| Quién pide | Respuesta de `i0.wp.com` |
|---|---|
| IP residencial (tu casa, el navegador) | `HTTP 200` + JPEG válido ✅ |
| IP de datacenter (la VM de Colab/Google Cloud) | `HTTP 404` ❌ |

Es **hotlink protection / filtrado por reputación de IP**: el CDN discrimina contra rangos de datacenter. Cuando PIL recibe el cuerpo del `404` (que es **HTML de error**, no una imagen) intenta abrirlo como imagen y lanza `UnidentifiedImageError`. El mensaje no menciona el `404` porque `requests` con `stream=True` **nunca verificó el código de estado**. Mismo riesgo con `gstatic.com` y `freepik`.

### Por qué `.raw` es frágil

Más allá del 404, el patrón `requests.get(...).raw` tiene problemas estructurales:

| Riesgo | Qué pasa con `.raw` |
|---|---|
| `Content-Encoding: gzip` | `.raw` entrega **bytes comprimidos** sin descomprimir → PIL ve basura |
| *Seekability* | El stream crudo no es un archivo *seekable* fiable; PIL necesita rebobinar para detectar el formato |
| Redirects | Un `301/302` puede dejar `.raw` apuntando al cuerpo del redirect, no a la imagen final |
| Diagnóstico | Nunca se llamó a `raise_for_status()` → un `404` se silencia y reaparece como error de PIL |

### Solución robusta

Un helper que descarga el contenido completo a memoria, valida el estado y el tipo, y recién entonces lo entrega a PIL:

```python
from io import BytesIO
from PIL import Image
import requests

def load_image(url):
    headers = {
        # Algunos CDN bloquean clientes sin User-Agent de navegador
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0 Safari/537.36")
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()  # 404/403 lanzan un error LEGIBLE, no críptico
    ctype = resp.headers.get("Content-Type", "")
    if "image" not in ctype:
        raise ValueError(f"La URL no devolvió una imagen (Content-Type: {ctype!r})")
    return Image.open(BytesIO(resp.content)).convert("RGB")
```

Comparación directa de los dos enfoques:

| Aspecto | `.raw` (sin headers) | `BytesIO(resp.content)` + headers |
|---|---|---|
| `Content-Encoding: gzip` | entrega bytes comprimidos ❌ | `requests` descomprime; `.content` ya viene plano ✅ |
| *Seekability* | stream no fiable ❌ | `BytesIO` es totalmente *seekable* ✅ |
| Redirects | puede romperse ❌ | `requests` los sigue antes de `.content` ✅ |
| Hotlink / 404 | el cuerpo del 404 llega a PIL ❌ | `raise_for_status()` lo corta antes ✅ |
| Mensaje de error | `UnidentifiedImageError` críptico ❌ | `HTTPError: 404` legible ✅ |

### Regla práctica: de dónde tomar imágenes de prueba

Para imágenes de prueba, **prefiere fuentes que no discriminan por IP**: `upload.wikimedia.org` o `raw.githubusercontent.com` sobre CDNs comerciales (`i0.wp.com`, `gstatic.com`, `freepik`).

Hay una trampa adicional con Wikimedia: **la ruta se deriva del hash MD5 del nombre del archivo**, no es adivinable. La imagen real del ornitorrinco vive en:

```
https://upload.wikimedia.org/wikipedia/commons/f/f2/Platypus.jpg
```

Los segmentos `f/f2` son los dos primeros dígitos hexadecimales del MD5 de `Platypus.jpg`, **no** `8/82` ni ningún otro par "lógico". Inventar la ruta da `404`.

Esto explica el patrón observado en el lab: la celda del **perro** (servida desde `raw.githubusercontent.com`) funcionó **sin tocar nada**, mientras que las de `gstatic.com` y `i0.wp.com` fallaron — no por el código, sino por la fuente.

---

## Parte B — Decoding y BLEU (lo que el lab promete pero no ejecuta)

A diferencia del [lab 22](/laboratorios/lab-22/decodificacion), donde se barren `num_beams`, `top_p` y `temperature`, el lab 23 **siempre usa greedy**: llama a `generate` sin `num_beams`, sin `do_sample`. El único parámetro de decoding visible es `max_length=20` en captioning. La Actividad pregunta por *beam search* (P4) y por *BLEU* (P6), pero el notebook nunca los corre. Aquí está lo que faltaría.

### Generación autoregresiva

Tanto el caption como la respuesta de VQA se construyen **token a token**, condicionando cada paso en lo ya generado, la imagen y (en VQA) la pregunta:

$$P(\text{respuesta}) = \prod_t P(y_t \mid y_{<t}, \text{imagen}, \text{pregunta})$$

La **estrategia de decoding** es cómo se elige cada $y_t$ a partir de esa distribución. Detalle en el [fundamento Decoding Strategies](/fundamentos/decoding-strategies).

### Greedy

Toma el **argmax** en cada paso: el token más probable, sin mirar atrás ni hacia adelante.

$$y_t = \arg\max_{w} P(w \mid y_{<t}, \dots)$$

Es **determinista, rápido y conservador**, pero **miope**: una elección localmente óptima puede cerrar el camino a una secuencia globalmente mejor. Esto explica directamente las **respuestas cortas y secas** de VQA en el lab ("yes", "2", "white") — greedy no se arriesga a frases largas.

### Beam search (`num_beams=k`)

Mantiene $k$ secuencias candidatas (los *beams*) ordenadas por **log-probabilidad acumulada**, expandiéndolas en paralelo, y al final devuelve la de mayor probabilidad **global**:

$$\text{score}(y_{1:T}) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, \dots)$$

Es lo que pide la **pregunta 4 de la Actividad**. *Trade-off:* es más caro (computa $k$ ramas) y tiende a salidas algo **genéricas** — optimiza probabilidad, no calidad percibida (la lección del lab 22: más beams puede empeorar la saliencia). Detalle en [Decoding Strategies](/fundamentos/decoding-strategies).

### Sampling + temperature + top-p (nucleus)

En lugar de elegir el más probable, se **muestrea** de la distribución. Dos perillas controlan cuán arriesgado es:

- **Temperatura $T$**: divide los logits antes del softmax.

$$P(x_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

  $T < 1$ **afila** la distribución (más conservador); $T > 1$ la **aplana** (más diverso).

- **Top-p (nucleus)**: se queda con el **núcleo** — el conjunto más pequeño de tokens cuya probabilidad acumulada alcanza $\geq p$ — y descarta la cola. El tamaño del núcleo es **dinámico** según cuán segura esté la distribución.

BLIP usa precisamente **nucleus sampling** en su etapa de *CapFilt* (el *Captioner* genera captions sintéticos) para conseguir **diversidad** de descripciones. Ver el paper de [Holtzman 2020](/papers/nucleus-sampling-holtzman-2020).

### BLEU

El lab **nunca calcula BLEU**, aunque la **pregunta 6 de la Actividad** evalúa la calidad del caption con esta métrica. BLEU mide:

- **Precisión de n-gramas modificada**: qué fracción de los n-gramas del caption generado aparece en la(s) referencia(s), recortando por la cuenta máxima en la referencia (evita inflar repitiendo una palabra).
- **Brevity penalty**: penaliza captions más cortos que la referencia, para que el modelo no haga trampa generando salidas mínimas.

$$\text{BLEU} = \underbrace{\text{BP}}_{\text{brevity penalty}} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

Un caption correcto contra la referencia real da **BLEU alto**; el caption **alucinado** del ornitorrinco — *"a baby bird is held in a box"* — da **BLEU ≈ 0** frente a la referencia verdadera (que habla de un *platypus*, no de un *bird* ni de una *box*): casi ningún n-grama coincide. Detalle en el [fundamento BLEU](/fundamentos/bleu-metric).

### Captioning condicional

BLIP también admite **sesgar el caption** con un prefijo de texto:

```python
inputs = processor(image, text="a photography of", return_tensors="pt")
out = model.generate(**inputs)
```

El modelo continúa a partir de *"a photography of …"*, lo que orienta el estilo y el comienzo del caption. **No se usa en el lab base** (que hace captioning incondicional), pero es la palanca natural para guiar la descripción.

---

## Qué se podría agregar al notebook

Todas estas piezas son mejoras directas sobre el notebook actual:

- **BLEU con `nltk`** (`nltk.translate.bleu_score`) para puntuar los captions contra referencias y responder la P6 con números, no a ojo.
- **Comparar greedy / beam / nucleus** sobre la misma imagen para ver el efecto en VQA y captioning (P4).
- **Captioning condicional** con distintos prefijos.
- **GPU + bucle de evaluación** sobre un set de imágenes con su referencia, para medir calidad de forma sistemática en vez de inspección visual caso a caso.

---

**Anterior:** [Image Captioning con BLIP](image-captioning-blip) · **Siguiente:** [Actividad resuelta](actividad)
