---
title: "Construcción de la KB y blocking"
weight: 2
---

El núcleo conceptual del lab no es el modelo, sino **cómo se convierte texto de Wikipedia en una memoria (key, value)** y cómo se reduce esa memoria a un puñado de candidatos por pregunta. Tres etapas: ventanas duales → índice invertido → candidate generation.

## Etapa 1 — Ventanas key/value duales

Para cada **entidad** que aparece en un artículo de Wikipedia, se toma una ventana de **±3 tokens** y se generan **dos entradas duales**:

```python
TOKENS_PER_SIDE = 3
for i, center_token in enumerate(article['body']):
    if center_token in entity_replacements:        # solo entidades
        window = list(map(lambda x: '1:' + x, article['body'][start:end+1]))  # prefijo key
        window[TOKENS_PER_SIDE] = center_token       # el centro queda SIN prefijo (rol value)
        # entrada WINDOW_CENTER → value = la entidad central
        keys_txt.append(['__window_center__', '1:' + article['title']] + window)
        values_txt.append(center_token)
        # entrada MOVIE → value = el título de la película
        keys_txt.append(['__movie__'] + window)
        values_txt.append(article['title'])
```

Ejemplo, con body `... ( film ) marathon_man is ...` y centro `film`:

```
key = [__window_center__, 1:marathon_man, 1:(, film, 1:), 1:marathon_man, 1:is]   value = film
key = [__movie__,         1:(, film, 1:), 1:marathon_man, 1:is]                    value = marathon_man
```

**El porqué de las dos vistas.** Cada porción de texto sirve para dos direcciones de consulta:

| Tipo de entrada | Pregunta que ayuda a responder | Lógica |
|---|---|---|
| `__movie__` (value = película) | *"¿En qué película aparece X?"* | la ventana lleva a la **película** |
| `__window_center__` (value = entidad) | *"¿Qué/quién es la entidad central en la película Y?"* | película + ventana llevan a la **entidad** |

Es la separación key/value del paper: **la key se diseña para matchear la pregunta, el value para ser la respuesta**.

## El truco del prefijo `1:`

Todos los tokens de la ventana llevan `1:` (rol **key**), **excepto el central**, que va sin prefijo (rol **value**). Resultado: `1:movie` y `movie` reciben **embeddings distintos** en la matriz compartida.

**El porqué.** En vez de mantener dos matrices físicas de embedding (una para keys, otra para values), el modelo usa **una sola tabla** y duplica el vocabulario con el prefijo. Por eso `n_embed = 186841` ≈ vocabulario × 2 + tokens especiales (`__movie__`, `__window_center__`, `<NULL>`, `UNK`). Es lo que permite los [embeddings compartidos](../01-arquitectura-kvmemnn) del modelo.

## Etapa 2 — Índice invertido (el blocker)

La KB completa tiene ~800.000 entradas. Comparar la pregunta contra todas con atención sería carísimo. Solución: un **índice invertido** token → entradas que lo contienen.

```python
wiki_hash = defaultdict(set)
for i, key in enumerate(wiki_k):
    for token in key:
        if freqs[token] < 1000:     # ignora stopwords (tokens muy frecuentes)
            wiki_hash[token].add(i)
```

**El filtro `freqs[token] < 1000` es la clave.** Los tokens hiperfrecuentes (`the`, `,`, `film`, `is`) aparecen en casi todas las entradas; si se indexaran, `wiki_hash['the']` tendría millones de entradas y el blocking no filtraría nada. Saltarlos = tratarlos como **stopwords**. Es el problema de los blocking keys de baja entropía en record linkage: un blocking key demasiado común no particiona el espacio.

> **Conexión con MDM/FHIR.** Este `wiki_hash` es un **blocker / inverted index** para candidate generation — igual que el bi-encoder blocker de una arquitectura de matching. Primero se reduce el espacio de candidatos con un índice barato, después se scorea fino con atención. Y `freqs < 1000` es el equivalente a no bloquear por campos de baja entropía (género, país) y sí por campos discriminativos (RUT, fecha de nacimiento).

## Etapa 3 — Candidate generation por pregunta

Para cada pregunta se recuperan las entradas que comparten ≥1 token, y se arma el ejemplo de entrenamiento:

```python
def gen_kb_candidates(question):
    cand_indices = set()
    for token in question:
        if token not in wiki_hash: continue        # stopword → saltar
        cand_indices.update(wiki_hash[token])      # entradas que contienen el token
    return [(wiki_k[i], wiki_v[i]) for i in cand_indices]

# ... en gen_candidates:
first_answer = answer[0]                            # solo la 1ra respuesta si hay varias
kb_values = [c[1][0] for c in kb_candidates]
if first_answer not in kb_values: continue          # descarta el ejemplo si no es respondible
ans_index = kb_values.index(first_answer)           # la etiqueta es un ÍNDICE
```

Tres decisiones de diseño que cambian todo:

1. **La etiqueta es un índice, no un token del vocabulario.** El modelo clasifica sobre los **candidatos recuperados para esa pregunta** (de tamaño variable), y la respuesta es la **posición** del value correcto. El techo de accuracy lo pone la **cobertura del recall del blocker**.
2. **Ejemplos no respondibles se descartan** (`if first_answer not in kb_values: continue`). Consecuencia: el accuracy reportado es **sobre los ejemplos respondibles**, no sobre todas las preguntas — la pérdida de recall del blocker se "esconde".
3. **Solo la primera respuesta** (`first_answer = answer[0]`). WikiMovies tiene preguntas multi-respuesta ("¿qué películas dirigió X?"); colapsarlas a la primera es la causa raíz del modo de fallo `describe X` analizado en [Experimentos](../03-experimentos-y-analisis).

Esto reduce de ~800.000 entradas a ~3.000 por pregunta (preprocesamiento didáctico) o ~1.000 (preprocesamiento óptimo, con un filtro adicional top-1000 por similaridad de un modelo entrenado).

## El padding (un detalle que confunde)

```python
UNK_TOKEN = 2
MAX_KEY_WORDS = 9
# cada key se trunca/rellena a 9 tokens con UNK_TOKEN...
# ...luego pad_sequence apila todas las keys del ejemplo con padding_value=0
```

Hay **dos padding values distintos** (`UNK_TOKEN=2` dentro de cada key, `0` entre keys), pero el segundo nunca se dispara porque todas las keys ya miden 9. El objetivo es **evitar máscaras**: si todas las secuencias tienen el mismo largo, no hace falta enmascarar en la agregación BoW. El costo es ruido mínimo del padding.

## Los dos preprocesamientos (gotcha de evaluación)

El notebook muestra un preprocesamiento **didáctico** (entidad = 1 token, ~3000 candidatos, **61%** accuracy) pero luego carga `.pkl` ya hechos con el preprocesamiento **óptimo** (palabras separadas, top-1000 por similaridad, **69%**). Son incompatibles entre sí: los IDs de token difieren, y el checkpoint `best_state.pt` está alineado con el óptimo. **Hay que cargar los `.pkl` (celdas 38-39)** o la evaluación con pesos pre-entrenados da basura.

---

**Siguiente:** [Experimentos propios y análisis](../03-experimentos-y-analisis) — qué reveló sondear el modelo entrenado.
