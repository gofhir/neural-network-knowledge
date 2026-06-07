---
title: "P1 — Inferencia y trigram blocking"
weight: 3
---

> **Celdas 22, 31-37 del notebook (Parte 1).** Cargar datos, hacer el forward pass, rankear oraciones, aplicar trigram blocking y analizar predicciones reales sobre CNN/DailyMail.

## Funciones auxiliares (celda 22)

Dos grupos de funciones. El primero, formato de terminal con códigos ANSI:

```python
def bold(text):      return '\033[1m' + text + '\033[0m'   # negrita
def highlight(text): return '\033[43m' + text + '\033[0m'  # fondo amarillo
```

`highlight` se usa para **resaltar qué oraciones del texto fuente fueron elegidas** por el modelo — un truco visual que encarna el resumen extractivo dentro del artículo. La función `display_translation` imprime, por ejemplo, Source Text / Gold Standard / Predicted, resaltando en amarillo cada oración del source que aparezca en la predicción.

El segundo grupo es la **lógica de trigram blocking** (parte del modelo, no solo visual):

```python
def _get_ngrams(n, text):     # conjunto de n-gramas de una lista de tokens
    return set(tuple(text[i:i+n]) for i in range(len(text)-n+1))

def _block_tri(c, p):         # ¿bloquear el candidato c dadas las oraciones p ya elegidas?
    tri_c = _get_ngrams(3, c.split())
    for s in p:
        if len(tri_c.intersection(_get_ngrams(3, s.split()))) > 0:
            return True       # comparte un trigrama → bloquear
    return False
```

Es directamente del paper: si dos oraciones comparten **cualquier trigrama**, probablemente dicen lo mismo → quédate con la primera. Es **Maximal Marginal Relevance** en versión barata. Quitar el trigram blocking baja ROUGE notablemente (ablación del paper).

## Cargar datos de test (celdas 32-33)

```python
test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=True),
                                   args.batch_size, device, shuffle=False, is_test=True)
batch = next(iter(test_iter))
display_translation(batch.src_str, batch.tgt_str)
```

`is_test=True` hace que el batch conserve los **strings originales** (`src_str`, `tgt_str`) además de los tensores, permitiendo la visualización legible. El gold usa el separador `<q>` entre sus oraciones (convención de BertSum). Mostrar los datos *antes* del modelo deja ver que el **gold es abstractivo**: sus oraciones no aparecen literales en el source — el problema que motiva el oracle.

## El corazón: inferencia + selección (celda 37)

```python
model.eval()
with torch.no_grad():
    sent_scores, mask = model(src, segs, clss, mask, mask_cls)   # score por oración
    sent_scores = sent_scores + mask.float()                     # empuja oraciones reales sobre el padding
    selected_ids = np.argsort(-sent_scores, 1)                   # ranking descendente

    for i, idx in enumerate(selected_ids):
        _pred = []
        for j in selected_ids[i][:len(batch.src_str[i])]:
            candidate = batch.src_str[i][j].strip()
            if args.block_trigram:
                if not _block_tri(candidate, _pred):   # trigram blocking
                    _pred.append(candidate)
            else:
                _pred.append(candidate)
            if len(_pred) == 3:                         # length cap (CNN/DM)
                break
```

El procedimiento es exactamente el pseudo-código de inferencia del paper:
1. **Forward pass:** BERT procesa el documento, se extraen los vectores `[CLS]`, el clasificador produce un **score $\hat{Y}_i \in [0,1]$ por oración**.
2. **Ranking** por score descendente (`argsort` del negativo).
3. **Trigram blocking:** agrega una oración solo si no comparte trigrama con las ya elegidas.
4. **Length cap:** se detiene en **3 oraciones**.

Detalles: `model.eval()` desactiva dropout; `torch.no_grad()` ahorra memoria; `sent_scores + mask.float()` garantiza que el padding nunca se seleccione. `batch.labels` se extrae pero **no se usa** (son las etiquetas oracle, irrelevantes en inferencia).

## Análisis de predicciones reales

Dos ejemplos reales del test set ilustran un **éxito** y una **falla parcial** — el contraste perfecto.

### Sample "rugby" (Jonny May) — el modelo acertó 3/3 hechos ✅

| Hecho del Gold | Oración elegida | ¿Cubierto? |
|---|---|---|
| "gloucester beat exeter 30-19...challenge cup final" | #4 "...gloucester's **30-19 victory against exeter**" | ✅ |
| "cherry and whites face edinburgh at twickenham stoop on may 1" | #2 "...the **cherry and whites** now face **edinburgh at the twickenham stoop on may 1**" | ✅ |
| "jonny may overlooked for the last three rounds of the rbs 6 nations" | #3 "...**overlooked for the last three rounds of the rbs 6 nations**" | ✅ |

El modelo eligió {4, 2, 3} y cubrió los tres hechos. La oración #2 es **densa** — empaqueta dos sub-hechos.

### Sample "tenis" (Bedene) — el modelo acertó solo 2/3 ❌

El gold tenía tres hechos: Bedene perdió 6-1 6-4, cambió de Eslovenia a GB, y **Klizan venció a Almagro 6-4 7-6**. El modelo eligió {2, 4, 3} — **las tres del primer tercio del artículo** — y se perdió la oración #10 (Klizan), que estaba **al final** pero era un hecho del gold casi textual.

> **Sesgo LEAD en vivo:** el modelo concentró su atención al inicio y "no vio" la oración 10. Gastó **dos** de sus tres cupos en el mismo subtema (Bedene/Vesely), cuando el gold quería diversidad. Es una falla de **cobertura**.

### Tres defectos teóricos, observados los tres

1. **Captions como ruido:** en *ambos* samples, el modelo eligió como #1 una **caption** (pie de foto; en el de tenis literalmente "(not pictured)"). Las captions son densas en n-gramas clave → el oracle las etiqueta con label=1 → el modelo aprende a elegirlas. Confirma empíricamente el ruido del oracle.
2. **Límite del trigram blocking:** en el sample de rugby, las oraciones #2 y #4 son **semánticamente redundantes** ("gloucester venció 30-19 a exeter") pero **no se bloquearon** porque no comparten 3 palabras consecutivas idénticas. El método detecta redundancia *léxica exacta*, no *parafraseada*.
3. **Sesgo LEAD:** el baseline LEAD-3 (tomar las 3 primeras oraciones) es fuerte en CNN/DM (~ROUGE-L 36) porque las noticias usan pirámide invertida. El sesgo está horneado en los datos y BertSum lo hereda.

> **Síntesis:** BertSum es bueno cuando los hechos importantes están al inicio (rugby) y falla cuando un hecho clave está enterrado al final (tenis, la oración de Klizan). Los tres "defectos teóricos" —sesgo LEAD, ruido de captions, límite del trigram blocking— **aparecieron los tres en dos ejemplos**. No son hipótesis: son comportamiento observable.

---

**Anterior:** [estrategia de entrenamiento](entrenamiento-bertsum) · **Siguiente:** [actividades del extractivo](actividades-extractivo)
