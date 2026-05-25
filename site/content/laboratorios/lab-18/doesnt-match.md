---
title: "Bloque 2 — doesnt_match (outliers por centroide)"
weight: 20
math: true
---

Recorrido del bloque `doesnt_match` (Celdas 24-35 del notebook). El lab te entrega 5 ejemplos guía y te pide generar 3 propios + comentario (Actividad 2).

## La función `doesnt_match`

A diferencia de las analogías (3CosMul), `doesnt_match` usa un mecanismo mucho más simple — **distancia al centroide**:

```python
def doesnt_match(words):
    vectors = [normalize(model[w]) for w in words]
    centroid = sum(vectors) / len(vectors)
    centroid = centroid / norm(centroid)  # re-normalizar
    similarities = [cosine(v, centroid) for v in vectors]
    return words[argmin(similarities)]
```

→ Devuelve la palabra **más lejana** del centroide normalizado del grupo.

## Diferencia conceptual vs analogías

| Aspecto | `most_similar_cosmul` | `doesnt_match` |
|---|---|---|
| Pregunta | "¿Cuál cumple A:B :: C:?" | "¿Cuál NO pertenece al grupo?" |
| Inputs | Listas `positive` + `negative` | Una lista única |
| Fórmula | 3CosMul (Levy-Goldberg 2014) | Distancia al centroide |
| Output | Lista top-N con scores | Una sola palabra |
| Sensibilidad a polisemia | **Alta** (vimos los problemas con Santiago) | **Moderada** (el promedio suaviza) |
| Paper canónico | Sí, Levy-Goldberg 2014 CoNLL | No (es propiedad emergente) |

## Los 5 ejemplos guía del lab

| Grupo | Outlier | Tipo de test |
|---|---|---|
| `white, blue, red, Chile` | **Chile** | Color + país (dominios completamente disjuntos) |
| `Sun, Moon, lunch, Jupiter` | **lunch** | Astros + comida (disjunto trivial) |
| `April, May, September, Tuesday, July` | **Tuesday** | Meses + día (sub-dominios del mismo dominio temporal) |
| `Monday, Tuesday, September, Thursday, Friday` | **September** | Días + mes (test de consistencia) |
| `Lima, Paris, London, Madrid` | **Lima** | Capitales europeas + sudamericana (geografía continental fina) |

## Observación crítica: robustez a polisemia

La Celda 28 prueba `[April, May, September, Tuesday, July]`. Notable: **`May` está en NLTK stopwords y es polisémica** (mes + verbo modal "may"). Sin embargo, el modelo correctamente detecta `Tuesday` como outlier, **no May**.

→ **El centroide es robusto a la polisemia individual** porque promedia 5 vectores. Esto contrasta con las analogías donde un atractor polisémico (Santiago) destruye el resultado.

## Los 3 ejemplos propios (Actividad 2)

### Ejemplo 1 — Trivial: música + intrusa

```python
google_wordvecs.doesnt_match(['guitar', 'piano', 'violin', 'baseball'])
# → 'baseball'
```

✅ Resultado esperado. Dominios completamente disjuntos.

### Ejemplo 2 — Intermedio: dentro del dominio deportes USA

```python
google_wordvecs.doesnt_match(['Lakers', 'Yankees', 'Celtics', 'Bulls'])
# → 'Yankees'
```

✅ El modelo distingue **NBA vs MLB** dentro del super-dominio "equipos deportivos estadounidenses". Tres dimensiones convergen apuntando a Yankees: (a) liga distinta (MLB), (b) ciudad distinta (NY), (c) deporte distinto (béisbol).

### Ejemplo 3 — Difícil: big tech con dominios solapados

```python
google_wordvecs.doesnt_match(['Microsoft', 'Apple', 'Google', 'Amazon'])
# → 'Amazon'
```

✅ El modelo capturó la dimensión **"retail vs tech platform"**. En el corpus pre-2013, Amazon era predominantemente "the online retailer". AWS lanzó en 2006 pero su cobertura mediática masiva no llegó hasta 2015-2017, después del entrenamiento. Microsoft/Apple/Google forman cluster "tech platforms"; Amazon vive en cluster "retail/e-commerce".

→ **Confirmación cruzada con Plot 3 de Actividad 3**: Amazon aparece como outlier en PC2 también, validando este hallazgo geométricamente.

## ¿Por qué `doesnt_match` es más estable?

Tres razones convergentes:

1. **Promedio de vectores cancela ruido**: la polisemia individual de una palabra se diluye al promediarse con 4-5 otras palabras del cluster.
2. **No depende de aritmética relacional**: no necesita resolver "A:B :: C:?", solo medir distancias absolutas.
3. **Robusto a frecuencia**: el centroide ignora si una palabra es rara o común — solo importa la dirección semántica.

## Aplicaciones prácticas más allá del lab

| Aplicación | Cómo usa `doesnt_match` |
|---|---|
| Detección de spam | Palabras del email que no encajan con el tema → señal de spam injectado |
| Limpieza de listas | Detectar entradas mal categorizadas en taxonomías |
| QA de datasets | Encontrar ejemplos mal etiquetados en sets de entrenamiento |
| Análisis de jerga | Detectar palabras "fuera de jerga" en un dominio técnico |

## Cross-links

{{< cards >}}
  {{< card link="../" title="← Lab 18 - Hub" subtitle="Volver al índice del lab" icon="academic-cap" >}}
  {{< card link="../analogias" title="Bloque 1 - Analogías" subtitle="3CosMul con polisemia" icon="academic-cap" >}}
  {{< card link="../visualizacion-pca" title="Bloque 3 - PCA →" subtitle="Visualizar la geometría" icon="academic-cap" >}}
{{< /cards >}}
