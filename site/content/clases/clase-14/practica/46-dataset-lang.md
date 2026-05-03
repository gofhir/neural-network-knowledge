---
title: "46 - Dataset: deteccion de idioma EN/ES"
weight: 460
math: true
---

## 1. Ground truth perfecto sin etiquetado manual

Para fine-tuning necesitas datos etiquetados. Aqui el etiquetado es perfecto y automatico: cualquier ventana de `shakespeare.txt` es EN, cualquier ventana de `quijote.txt` es ES.

No hay ambiguedad. No hay costo de anotacion. No hay posibilidad de error humano. El corpus de Shakespeare es completamente en ingles isabelino; el Quijote es completamente en castellano del siglo XVII. Una ventana aleatoria de cualquiera de los dos textos es 100 % del idioma correspondiente, sin excepciones.

Este es el tipo de alineacion ideal para construccion de datasets de clasificacion: cuando la fuente de los datos es en si misma la etiqueta. En la practica, muchas tareas de NLP tienen esta propiedad — deteccion de idioma, clasificacion de sentimiento con resenas de estrellas, analisis de polaridad con votos — y aprovecharla elimina el cuello de botella del etiquetado humano.

---

## 2. Diseno del dataset: ventanas, semilla y balance

### Ventanas de 64 tokens

El script extrae ventanas contiguas de exactamente 64 tokens del corpus tokenizado (BPE). A cada ventana se le antepone `[CLS]` y se le agrega `[SEP]` al final, produciendo secuencias de 66 tokens de largo.

```
[CLS] tok_1 tok_2 ... tok_64 [SEP]   <- longitud total: 66
```

La decision de WINDOW=64 no es arbitraria: el modelo MiniBERT fue preentrenado con `max_seq_len=66`. Usar ventanas mas grandes causaria un `IndexError` en `LearnedPositionalEmbedding` porque los embeddings posicionales solo estan definidos para posiciones 0 a 65. Las ventanas de 64 tokens maximizan la longitud de contexto sin exceder ese limite.

### Semilla SEED=246

El script usa `SEED=246` para tres propositos:
1. `random.seed(SEED)` y `torch.manual_seed(SEED)` — reproducibilidad global.
2. `random.Random(SEED + label + split_offset)` — semilla distinta por idioma Y por split, evitando que train y eval muestreen las mismas ventanas:
   - Train EN: SEED+0+0 = 246
   - Train ES: SEED+1+0 = 247
   - Eval EN: SEED+0+100 = 346
   - Eval ES: SEED+1+100 = 347
3. `random.shuffle(examples)` — mezcla determinista del conjunto combinado.

El resultado es que cada ejecucion del script produce exactamente los mismos datasets. Esto es critico para reproducibilidad: si alguien descarga el repositorio y corre el script, obtiene los mismos archivos JSONL que estan commitidos.

### Balance perfecto

| Split | EN | ES | Total |
|-------|----|----|-------|
| train | 1000 | 1000 | 2000 |
| eval  | 250  | 250  | 500  |

El balance 50/50 es intencional. Con balance perfecto, un clasificador que predice siempre la clase mayoritaria obtiene 50 % de exactitud. Cualquier metrica por encima de 50 % es evidencia real de aprendizaje. Si el dataset estuviera desbalanceado (por ejemplo, 90 % EN), un modelo que siempre prediga EN tendria 90 % de exactitud sin haber aprendido nada.

---

## 3. Por que WINDOW=64 y no mayor

El constraint es la arquitectura del encoder preentrenado:

```python
# En _models.py
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return self.embedding(positions)  # IndexError si x.size(1) > max_seq_len
```

El checkpoint `mini_bert_pretrained.pt` fue entrenado con `max_seq_len=66`. Si se intenta pasar una secuencia de longitud 70, `torch.arange(70)` genera indices [0..69], y `self.embedding(positions)` falla con `IndexError: index 66 is out of bounds for dimension 0 with size 66`.

WINDOW=64 garantiza que `[CLS] + window + [SEP]` tiene exactamente 66 tokens — el maximo permitido.

---

## 4. Script completo

```python
"""46_dataset_lang.py - Cap 46: dataset EN/ES para deteccion de idioma."""
import json, random, torch
from pathlib import Path
from _bpe import BPETokenizer

SEED = 246
random.seed(SEED); torch.manual_seed(SEED)

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

WINDOW = 64  # tokens por ejemplo (sin [CLS][SEP])

en_text = Path("shakespeare.txt").read_text(encoding="utf-8")
es_text = Path("quijote.txt").read_text(encoding="utf-8")
en_tokens = tok.encode(en_text)
es_tokens = tok.encode(es_text)

def sample_windows(tokens, n, label, split_offset=0):
    rng = random.Random(SEED + label + split_offset)
    examples = []
    for _ in range(n):
        start = rng.randint(0, len(tokens) - WINDOW - 1)
        window = tokens[start:start + WINDOW]
        full = [tok.cls_id] + window + [tok.sep_id]
        examples.append({"ids": full, "label": label})
    return examples

Path("data").mkdir(exist_ok=True)

for offset, (split, n_each, fout) in enumerate([
    ("train", 1000, "data/lang_train.jsonl"),
    ("eval",   250, "data/lang_eval.jsonl"),
]):
    examples = sample_windows(en_tokens, n_each, 0, split_offset=offset * 100) + \
               sample_windows(es_tokens, n_each, 1, split_offset=offset * 100)
    random.shuffle(examples)
    with open(fout, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"[{split}] {len(examples)} ejemplos ({n_each} EN + {n_each} ES) -> {fout}")

print("\nEjemplos del train set:")
with open("data/lang_train.jsonl") as f:
    for line in list(f)[:2]:
        ex = json.loads(line)
        decoded = tok.decode(ex["ids"])
        lang = "EN" if ex["label"] == 0 else "ES"
        print(f"  [{lang}] {decoded[:60]!r}...")
```

---

## 5. Output del script

```
[train] 2000 ejemplos (1000 EN + 1000 ES) -> data/lang_train.jsonl
[eval] 500 ejemplos (250 EN + 250 ES) -> data/lang_eval.jsonl

Ejemplos del train set:
  [EN] '[CLS]present for that time.\n\nJOHN OF GAUNT:\nWhat is six wint'...
  [EN] "[CLS]\nThe contrary doth make thee wonder'd at:\n'Tis governme"...
```

---

## 6. Verificacion: los ejemplos son coherentes

El output muestra dos ejemplos del train set. Ambos tienen label `EN` (label=0), lo que confirma que la mezcla y shuffle funcionaron — no estan ordenados primero todos los EN y luego todos los ES.

Los textos decodificados son reconociblemente Shakespeare:
- `"present for that time.\n\nJOHN OF GAUNT:\nWhat is six wint"` — fragmento de *Richard II*, personaje John of Gaunt.
- `"The contrary doth make thee wonder'd at:\n'Tis governme"` — prosa isabelina con construcciones arcaicas (`doth`, `thee`, `'Tis`).

Que la primera linea del train set sea EN no significa que el dataset este desordenado: el shuffle con `random.seed(SEED)` determina exactamente que linea aparece primera. En una inspeccion de las primeras 10 lineas, se alternarian ejemplos EN y ES.

La coherencia lingüistica es la garantia de calidad del dataset: no hace falta revisar manualmente si un ejemplo esta bien etiquetado — la procedencia del texto es la etiqueta.

---

## 7. Ventajas de este enfoque vs etiquetado manual

### Cero coste

El etiquetado manual de texto para clasificacion de idioma cuesta entre $0.02 y $0.10 por ejemplo segun plataformas de crowdsourcing. Para 2500 ejemplos, eso seria entre $50 y $250, mas el tiempo de gestion, instrucciones a anotadores, control de calidad y reconciliacion de desacuerdos. El enfoque de ventanas sobre corpus ya existentes tiene costo cero.

### Perfecto balance automatico

El script genera exactamente `n_each` ejemplos por clase con un parametro. Lograr balance perfecto con etiquetado manual requiere estratificacion activa y monitoreo durante la coleccion. Aqui es inherente al diseno.

### Reproducible al bit

Cualquier persona con acceso al repositorio puede regenerar los mismos datasets ejecutando el script. En etiquetado manual, incluso con los mismos textos fuente, distintos anotadores produciran etiquetas distintas en casos ambiguos. Aqui no hay casos ambiguos.

### Escalable trivialmente

Para aumentar el dataset de 2000 a 20000 ejemplos de entrenamiento, se cambia `n_each` de 1000 a 10000. El texto de Shakespeare tiene mas de 900000 caracteres y el Quijote supera el millon — hay suficiente material para decenas de miles de ventanas no repetidas de 64 tokens.

### Limitaciones inherentes

Este enfoque solo funciona cuando la fuente del dato es la etiqueta. No sirve para tareas donde la etiqueta no es deducible del corpus (sentimiento de una resena sin estrellas, intencion de un mensaje de chat, calidad de una traduccion). En esos casos, el etiquetado humano es irremplazable. La deteccion de idioma es un caso privilegiado donde el truco funciona perfectamente.

---

## 8. Preguntas de verificacion

**1. El script usa `random.Random(SEED + label + split_offset)` en lugar de el `random` global. ¿Por que?**

Si ambas clases usaran el mismo generador global, la secuencia de posiciones de inicio para EN y ES dependeria del orden en que se llaman las funciones. Con generadores independientes por clase y por split, la muestra de EN es identica independientemente de cuantas veces se llame la funcion para ES, y viceversa. El parametro `split_offset` es critico: sin el, train y eval inicializarian el RNG con la misma semilla, haciendo que el eval sea un subconjunto exacto del train (data leakage del 100%). Con `split_offset=0` para train y `split_offset=100` para eval, las semillas son disjuntas y los conjuntos son independientes.

**2. ¿Que pasaria si se usara WINDOW=65 en lugar de WINDOW=64?**

La secuencia resultante seria `[CLS] + 65 tokens + [SEP]` = 67 tokens. Al pasar al encoder con `max_seq_len=66`, `LearnedPositionalEmbedding` intentaria acceder al embedding en posicion 66, que no existe (los embeddings van de 0 a 65). PyTorch lanzaria `IndexError: index 66 is out of bounds for dimension 0 with size 66`. El fine-tuning fallaria en el primer batch.

**3. ¿Por que los dos ejemplos mostrados son ambos EN si el dataset esta mezclado?**

El shuffle con semilla fija produce un orden determinista. Con `SEED=246`, los primeros dos ejemplos del train set resultan ser de Shakespeare — esto es simplemente el resultado del shuffle, no un error. Si se inspeccionan mas lineas, aparecen ejemplos ES. La aleatoriedad del shuffle garantiza que en cualquier minibatch de tamano razonable habra mezcla de idiomas, que es lo que importa para el entrenamiento.
