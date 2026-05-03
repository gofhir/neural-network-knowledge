---
title: "45 - [CLS] como clasificador"
weight: 450
math: true
---

## 1. Apertura: del encoder a la clasificacion

El encoder MiniBERT produce un tensor de forma `(B, T, d_model)`. Para clasificar una oracion completa, necesitamos un unico vector que la represente. BERT eligio una solucion elegante: la posicion 0 de la secuencia, ocupada por el token especial `[CLS]`.

La idea es simple pero poderosa. En vez de comprimir todos los tokens en un vector (por ejemplo promediandolos), BERT reserva una posicion dedicada cuyo unico rol es acumular informacion de toda la oracion. Sobre ese vector se coloca una cabeza de clasificacion minima: una sola capa lineal `Linear(d_model, n_classes)`.

Este es el mecanismo que permite el fine-tuning eficiente de BERT en tareas de clasificacion de texto. El encoder hace el trabajo pesado de representacion; la cabeza es casi gratis.

---

## 2. El token [CLS] como resumen aprendido

El token `[CLS]` no tiene contenido semantico propio. En el vocabulario BPE es simplemente un identificador especial, y en la secuencia de entrada ocupa siempre la posicion 0 antes de cualquier texto. Eso es exactamente lo que lo hace util.

Durante el pretraining MLM, el modelo no predice el token en la posicion 0 (esa posicion no se enmascara). El `[CLS]` esta presente en todas las secuencias pero nunca es objetivo de ninguna prediccion. Sus pesos de embedding existen, pero no reciben gradiente directo del objetivo MLM.

Cuando llega el fine-tuning de clasificacion, la cabeza lineal se conecta al vector `[CLS]` y si recibe gradiente. El modelo aprende a usar esa posicion como espacio de representacion para la tarea especifica. En pocas iteraciones de fine-tuning, `[CLS]` se convierte en un "resumen" de la oracion optimizado para la clasificacion que le pedimos.

La formula es:

$$\hat{y} = \text{softmax}\left( W_c \cdot h_{\text{[CLS]}} + b_c \right)$$

donde $h_{\text{[CLS]}} = h[:, 0, :]$ es el vector en la posicion 0, $W_c \in \mathbb{R}^{n\_\text{classes} \times d\_\text{model}}$ y $b_c \in \mathbb{R}^{n\_\text{classes}}$.

---

## 3. Minimalismo de la cabeza: 258 parametros

La cabeza de clasificacion para la tarea EN vs ES (2 clases, d_model=128) consiste en:

- Pesos: $128 \times 2 = 256$ parametros
- Bias: $2$ parametros
- Total: **258 parametros**

El encoder MiniBERT tiene cientos de miles de parametros. La cabeza tiene 258. Esta asimetria es intencional: todo el conocimiento linguistico esta en el encoder preentrenado; la cabeza solo aprende a proyectar ese conocimiento a las clases de la tarea.

Esto es lo que hace que el paradigma pretraining + fine-tuning sea tan eficiente. No necesitamos entrenar millones de parametros desde cero para cada tarea nueva. El encoder se reutiliza; solo se ajusta con una cabeza minima y pocas iteraciones sobre datos etiquetados.

```python
class ClassificationHead(nn.Module):
    """Toma el vector [CLS] (posicion 0) y proyecta a n_classes."""
    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h[:, 0, :])
```

La linea clave es `h[:, 0, :]`: toma la dimension de secuencia en indice 0, dejando `(B, d_model)`. Sobre ese tensor se aplica la proyeccion lineal para obtener `(B, n_classes)`.

---

## 4. Script completo

```python
"""45_cls_head.py - Cap 45: [CLS] como vector clasificador."""
import torch
from _bpe import BPETokenizer
from _models import MiniBERT, ClassificationHead, get_device

device = get_device()
tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device, weights_only=False)
cfg  = ckpt["config"]
model = MiniBERT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# Cabeza de clasificacion: 128 → 2 (EN=0, ES=1)
cls_head = ClassificationHead(d_model=128, n_classes=2).to(device)

print("=== [CLS] como clasificador ===\n")
print(f"ClassificationHead: Linear(128, 2)")
n_params = sum(p.numel() for p in cls_head.parameters())
print(f"Params de la cabeza: {n_params} (minimos!)\n")

examples = [
    ("To be or not to be", "EN", 0),
    ("The king is dead", "EN", 0),
    ("En un lugar de la Mancha", "ES", 1),
    ("No hay mal que por bien no venga", "ES", 1),
]
print("CLS vectors antes de fine-tuning (clasificacion aleatoria):")
print(f"{'Texto':<40} {'Idioma'} {'Logit EN':>10} {'Logit ES':>10}")
for text, lang, _ in examples:
    ids = torch.tensor([tok.encode_bert(text)[:cfg["max_seq_len"]]],
                       dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids)
        logits = cls_head(h)
    print(f"{text:<40} {lang}     {logits[0,0].item():>10.3f}  {logits[0,1].item():>10.3f}")

print("\nLos logits son aleatorios (cabeza no entrenada) — fine-tuning en cap 47.")
print("\n=== Por que [CLS] y no promedio de todos los tokens? ===")
print("""
BERT podria usar promedio de todos los tokens como representacion.
Usar [CLS] es una decision de diseno:
  1. [CLS] es un token sin contenido propio — aprende libremente a ser 'resumen'
  2. Permite arquitecturas de dos-torres (cross-encoder) eficientes
  3. El promedio puede mezclar señales de tokens no relevantes
  4. En practica: ambos funcionan; [CLS] es el estandar BERT
""")
```

---

## 5. Output del script

```
=== [CLS] como clasificador ===

ClassificationHead: Linear(128, 2)
Params de la cabeza: 258 (minimos!)

CLS vectors antes de fine-tuning (clasificacion aleatoria):
Texto                                    Idioma   Logit EN   Logit ES
To be or not to be                       EN         -0.257       0.572
The king is dead                         EN         -0.083       0.700
En un lugar de la Mancha                 ES         -1.130       0.036
No hay mal que por bien no venga         ES         -1.076       0.309

Los logits son aleatorios (cabeza no entrenada) — fine-tuning en cap 47.

=== Por que [CLS] y no promedio de todos los tokens? ===

BERT podria usar promedio de todos los tokens como representacion.
Usar [CLS] es una decision de diseno:
  1. [CLS] es un token sin contenido propio — aprende libremente a ser 'resumen'
  2. Permite arquitecturas de dos-torres (cross-encoder) eficientes
  3. El promedio puede mezclar señales de tokens no relevantes
  4. En practica: ambos funcionan; [CLS] es el estandar BERT
```

---

## 6. Analisis: logits aleatorios como punto de partida

Los logits muestran algo interesante: los cuatro ejemplos tienen Logit ES > Logit EN. Esto no es porque el modelo distinga EN de ES — es simplemente la inicializacion aleatoria de la cabeza `Linear(128, 2)`, que con los pesos de PyTorch por defecto produce valores sistematicamente sesgados hacia ES.

Lo que importa notar:

- Los logits son pequenos (entre -1.13 y 0.70), consistente con pesos de cabeza recien inicializados (escala ~ $1/\sqrt{128} \approx 0.088$).
- No hay separacion entre ejemplos EN y ES: tanto "To be or not to be" como "En un lugar de la Mancha" tienen Logit ES dominante. La cabeza no distingue idiomas.
- El encoder si produce vectores `[CLS]` distintos para cada oracion (de lo contrario todos los logits serian iguales). La diversidad viene del encoder preentrenado.

El mensaje es claro: el encoder esta listo para ser usado como extractor de caracteristicas. La cabeza necesita entrenamiento supervisado. Cap 47 construira el dataset etiquetado y ejecutara el fine-tuning para convertir estos logits aleatorios en clasificacion util.

---

## 7. Por que [CLS] y no promedio de todos los tokens

El script incluye cuatro razones. Vamos a expandirlas:

### Razon 1: [CLS] aprende libremente su rol

Un token de contenido como `"king"` ya tiene un significado semantico consolidado — sus embeddings y pesos de atencion estan sesgados por las coocurrencias de la palabra "king" en el corpus. `[CLS]` no tiene ese sesgo: es un token artificial, sin ocurrencias en texto natural, que el modelo solo ve como marcador de inicio de secuencia. Cuando se aplica fine-tuning, sus representaciones son "terreno virgen" que el modelo puede moldear libremente para la tarea.

### Razon 2: Arquitecturas de dos-torres (cross-encoder)

En tareas de similitud de pares (por ejemplo: "¿son estas dos oraciones semanticamente equivalentes?"), se pueden pasar dos oraciones juntas: `[CLS] oracion_A [SEP] oracion_B [SEP]`. El vector `[CLS]` al final contiene la representacion de la interaccion completa entre ambas oraciones. Esta arquitectura es la base de los cross-encoders usados en retrieval y ranking. El promedio de tokens no tiene una posicion natural para acumular informacion de dos oraciones separadas.

### Razon 3: El promedio puede diluir la señal

Imagina una oracion como "La pelicula fue absolutamente horrible, aunque los efectos eran increibles". El promedio de todos los tokens mezcla el sentimiento negativo de "horrible" con el positivo de "increibles". Si queremos clasificar el sentimiento global, el promedio puede producir un vector ambiguo. `[CLS]`, al ser una posicion dedicada que atiende a toda la secuencia con sus propios pesos, puede aprender a ponderar tokens segun su relevancia para la tarea de clasificacion.

### Razon 4: Es el estandar establecido

El paper original de BERT (Devlin et al. 2018) usa `[CLS]` para todas las tareas de clasificacion de sentencias. La comunidad ha reproducido, ajustado y verificado este comportamiento en miles de experimentos. Hay tooling, checkpoints y benchmarks construidos sobre esta convencion. Cambiar a promedio seria posible — y en algunos casos funciona igual de bien — pero romper el estandar sin ganancia clara tiene costo de compatibilidad.

La eleccion correcta depende de la tarea y la arquitectura. Para clasificacion de oracion unica con fine-tuning BERT, `[CLS]` es la opcion probada. Para sentence embeddings sin fine-tuning (por ejemplo, similitud semantica zero-shot), el promedio de tokens o la media ponderada por atencion a veces supera a `[CLS]`.

---

## 8. Preguntas de verificacion

**1. ¿Cuantos parametros tiene una cabeza de clasificacion `Linear(128, 10)` para 10 clases?**

$128 \times 10 + 10 = 1290$ parametros. Los 10 del bias son uno por clase. La formula general es $d\_\text{model} \times n\_\text{classes} + n\_\text{classes}$.

**2. Los logits del script muestran Logit ES > Logit EN para todos los ejemplos, incluyendo los ingleses. ¿Eso significa que la cabeza ya clasifica incorrectamente?**

No exactamente: "clasificar incorrectamente" implica que el modelo deberia estar distinguiendo idiomas pero falla. Aqui la cabeza no ha sido entrenada en ningun dato etiquetado. El sesgo sistematico hacia ES es un artefacto de la inicializacion aleatoria de los pesos, no una decision semantica. Despues del fine-tuning, los logits tendran significado.

**3. ¿Por que el script usa `weights_only=False` en `torch.load`?**

El checkpoint `mini_bert_pretrained.pt` guarda un diccionario Python con campos como `"config"` y `"model"`. `weights_only=True` (el default seguro en PyTorch moderno) rechaza objetos arbitrarios de Python por seguridad — solo acepta tensores y tipos basicos. Como el checkpoint incluye el dict de configuracion, se necesita `weights_only=False`. En produccion esto requiere confiar en la fuente del checkpoint; aqui es local y seguro.
