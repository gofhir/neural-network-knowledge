---
title: "Bloque 3 — Visualización PCA 2D"
weight: 30
math: true
---

Recorrido del bloque de visualización (Celdas 36-54 del notebook). El lab te da 5 plots guía y te pide generar 3 propios + comentario (Actividad 3).

## El problema: 300 dimensiones son invisibles

Los embeddings de Google News viven en ℝ³⁰⁰. Para tener intuición geométrica necesitamos proyectar a 2D (o 3D). El lab usa **PCA** (Principal Component Analysis) — método lineal que encuentra los $k$ ejes ortogonales que máximamente preservan la varianza.

```python
vectors_2d = PCA(n_components=2).fit_transform(google_wordvecs.vectors)
# (100000, 300) → (100000, 2)
```

## La advertencia gigante: PCA es lossy

| Componentes | Varianza explicada típica |
|---|---|
| PC1 | ~4.5% |
| PC2 | ~2.9% |
| **PC1 + PC2** | **~7%** |
| PC1-100 | ~70-80% |
| PC1-300 | 100% |

→ **2 componentes capturan solo ~7% de la varianza total**. El 93% se pierde al visualizar en 2D. Esto tiene **consecuencias críticas** para interpretar los plots.

## Los 5 plots guía del lab

### Plot canónico: king/queen/man/woman

![PCA king queen man woman](/laboratorios/lab-18/pca-king-queen-man-woman.png)

**Análisis geométrico cuantitativo**:

| Palabra | PC1 | PC2 |
|---|---|---|
| king | 0.456 | 0.143 |
| man | 0.456 | 0.121 |
| queen | 0.359 | 0.072 |
| woman | 0.345 | 0.038 |

**Vectores diferencia**:

- `queen − king = (−0.097, −0.071)`, magnitud **0.120**
- `woman − man = (−0.111, −0.083)`, magnitud **0.139**
- **Coseno entre vectores ≈ 1.000** (perfectamente paralelos)
- **ζ = 0.139 / 0.120 ≈ 1.16** (no exactamente 1)

→ Confirmación empírica del **Teorema 1 de [Ri-Lee-Verma 2023](/papers/contrastive-analogies-ri-lee-verma-2023)**: la geometría real es **líneas paralelas con factor ζ ≠ 1**, no paralelogramos exactos.

### Plot animales vs vehículos — **el mito de los clusters limpios**

![PCA animals vs vehicles](/laboratorios/lab-18/pca-animals-vehicles.png)

**Sorpresa**: las 8 palabras NO forman dos clusters claramente separados:

- `dog`, `cat`, `animal`, `car`, `truck`, `vehicle` se **mezclan en una región central** (distancias muy pequeñas entre sí).
- `lion` y `tiger` (dos felinos salvajes semánticamente casi idénticos) aparecen en **cuadrantes opuestos** como outliers extremos.

**Distancias cuantitativas a `animal`**:
- `dog`: 0.051 ✅
- `car`: **0.061** ← un vehículo está MÁS CERCA del hyperónimo "animal" que `tiger`
- `tiger`: 0.121
- `lion`: 0.367 ← muy lejos

**Causa**: polisemia masiva. `lion` está sesgado a "Detroit Lions / Lion King". `tiger` está sesgado a "Tiger Woods / Detroit Tigers". Sus vectores se alejan del cluster animal por contaminación de nombres propios.

→ **Lección crítica**: los componentes principales PC1+PC2 capturan dimensiones distribucionales genéricas (frecuencia, formalidad), NO categorías taxonómicas.

### Plot países-capitales — paralelismo solo parcial

![PCA countries-capitals](/laboratorios/lab-18/pca-countries-capitals.png)

10 pares país-capital. Mi predicción era ver el paralelogramo canónico de Mikolov 2013. **Falló parcialmente**:

- **Pares que apuntan arriba-derecha** (esperado): Chile→Santiago, Greece→Athens, France→Paris, Italy→Rome.
- **Pares en dirección OPUESTA**: `Spain→Madrid`, `China→Beijing`, `Japan→Tokyo` van hacia ABAJO. Esto **rompe el paralelogramo**.

**Explicación**: `Madrid` se asocia a `Real Madrid` (club de fútbol), `Tokyo` y `Beijing` a contextos urbanos específicos (Olympics, pollution, earthquakes), no como "capital nacional". Polisemia funcional otra vez.

## Los 3 plots propios (Actividad 3)

### Plot 1 — Instrumentos por familia ✅

![PCA instrumentos](/laboratorios/lab-18/pca-instruments.png)

**Resultado**: separación clara por familia en PC2. Cuerdas (guitar, violin, cello, piano) abajo, vientos (trumpet, saxophone, flute, clarinet) arriba. Gap de ~0.10 entre clusters. **Trumpet es outlier en PC1** por polisemia funcional (uso militar + verbal metafórico + jazz icónico).

→ **Caso de éxito**: PCA 2D PUEDE capturar distinciones semánticas finas cuando las palabras son monosémicas.

### Plot 2 — Equipos NBA por conferencia ⚠️

![PCA NBA teams](/laboratorios/lab-18/pca-nba-teams.png)

**Resultado**: separación parcial. Los 3 equipos del Oeste sin polisemia (Lakers, Suns, Clippers) se agrupan en PC2 medio. Los 3 del Este (Knicks, Celtics, Pistons) en PC2 abajo. **Warriors y Bulls son outliers superiores** por polisemia: `Bulls` tiene sentido financiero fuerte ("bull market"), `Warriors` aparece como sustantivo genérico ("weekend warriors").

### Plot 3 — Empresas tech con líneas conectoras ★

![PCA tech companies con líneas](/laboratorios/lab-18/pca-tech-companies.png)

**Resultado visual**: las 4 líneas empresa→producto se ven paralelas. Pero **el análisis cuantitativo 2D vs 300D revela un hallazgo crítico**:

| Par 1 | Par 2 | Coseno 2D | Coseno 300D |
|---|---|---|---|
| Microsoft→Windows | Apple→iPhone | 0.978 | **0.251** |
| Microsoft→Windows | Google→Android | 0.935 | **0.392** |
| Microsoft→Windows | Amazon→Kindle | 0.706 | **0.216** |
| Apple→iPhone | Google→Android | 0.989 | **0.321** |
| Apple→iPhone | Amazon→Kindle | 0.839 | **0.239** |
| Google→Android | Amazon→Kindle | 0.911 | **0.249** |
| **Promedio** | | **0.893** | **0.278** |

**El paralelismo visual en 2D (0.89) es 3.2× más alto que en 300D (0.28)**.

→ **PCA preserva varianza global pero NO preserva ángulos entre vectores**. Las visualizaciones canónicas muestran una versión MÁS LIMPIA de la geometría real. La propiedad de paralelogramo opera en 300D, no en proyecciones 2D.

Pero el coseno 300D promedio (0.28) **no es ruido aleatorio**: para vectores aleatorios en ℝ³⁰⁰, la desviación estándar del coseno es ≈ $1/\sqrt{300} ≈ 0.058$. Cosenos de 0.21-0.39 están a 3-7 desviaciones estándar sobre aleatorio, lo que indica que **el "eje empresa→producto" existe como correlación débil pero estadísticamente significativa**.

## Tres conclusiones sobre PCA 2D para embeddings

1. **PUEDE capturar distinciones semánticas finas** cuando las palabras son monosémicas (Plot 1 instrumentos: separación cuerdas/vientos).
2. **FALLA cuando hay polisemia** porque las palabras viven en clusters mixtos (Plot 2 NBA: Warriors/Bulls outliers).
3. **INFLA artificialmente el paralelismo** comparado con el espacio 300D original — útil para intuición, engañoso para afirmaciones cuantitativas.

## Cross-links

{{< cards >}}
  {{< card link="../" title="← Lab 18 - Hub" subtitle="Volver al índice del lab" icon="academic-cap" >}}
  {{< card link="../doesnt-match" title="Bloque 2 - doesnt_match" subtitle="Lo mismo geométricamente algebraico" icon="academic-cap" >}}
  {{< card link="../sentiment-analysis" title="Bloque 4 - Sentiment →" subtitle="Aplicación downstream" icon="academic-cap" >}}
  {{< card link="/papers/contrastive-analogies-ri-lee-verma-2023" title="Teorema 1 (Ri-Lee-Verma)" subtitle="La teoría que valida el hallazgo ζ=1.16" icon="document-text" >}}
{{< /cards >}}
