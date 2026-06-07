---
title: "Lab 21 - Scene Text Recognition: ABCNet end-to-end"
weight: 210
sidebar:
  open: true
---

**Profesor:** Miguel Fadic
**Fecha:** Mayo 2026
**Notebook origen:** `clase_21/material/Laboratorio/Laboratorio Clase 21 - Scene Text Recognition.ipynb` (102 celdas)
**Notebook ejecutado:** [lab-21.ipynb](/notebooks/lab-21.ipynb) · [HTML](/notebooks-html/lab-21.html)

## Encuadre

Laboratorio práctico sobre [ABCNet](/papers/abcnet-liu-2020) (Liu et al., CVPR 2020), el primer *scene text spotter* **end-to-end** que representa texto curvo con **curvas Bézier cúbicas**. El lab tiene tres movimientos: (1) instalar y correr el modelo preentrenado, (2) **diseccionar su salida bit a bit** —desde los puntos de control Bézier y los índices de caracteres hasta el texto legible—, y (3) construir **dos aplicaciones reales** que reaprovechan el modelo sin reentrenarlo.

El eje conceptual es que ABCNet preentrenado en [TotalText](/papers/total-text-chng-2017) (carteles, señales) es **directamente útil** en dominios que nunca vio, porque aprendió a reconocer **formas de glifos latinos**, no un dataset ni un idioma concreto:

| | App 1 — Freiburg Groceries | App 2 — Google Street View |
|---|---|---|
| Dataset | [Freiburg Groceries](/papers/freiburg-groceries-jund-2016) (5000 imágenes, productos) | [Street View (UCF)](/papers/street-view-geolocalization-zamir-2010) (~6000 fotos + GPS) |
| Tarea reaprovechada | OCR de marcas / inventario | minería geoespacial de texto |
| Concepto clave | transfer *zero-shot* cross-idioma (inglés→alemán) | georreferenciación + diagnóstico de ruido |
| Técnica nueva | [fuzzy matching](/fundamentos/fuzzy-string-matching) (Levenshtein) | filtrado por zonas + `extent` geográfico |

El recorrido sigue las 102 celdas del notebook:

1. **Instalación y stack** (celdas 1-8): Detectron2 → AdelaiDet → ABCNet, parches de compatibilidad, arqueología de dependencias.
2. **Demo end-to-end** (celdas 9-18): pesos preentrenados, TotalText, el script `demo.py` sobre 300 imágenes.
3. **Disección del output** (celdas 19-37): `instances.beziers` (N,16) + `instances.recs` (N,25), el charset `chr(32+i)`, el glifo 口 hardcodeado, y la decodificación hecha a mano y verificada.
4. **App 1 — Groceries** (celdas 38-61): `build_model` + batching, OCR de productos alemanes, `Counter`, fuzzy matching de marcas.
5. **App 2 — Street View** (celdas 62-97): ruido de la UI de Maps, `get_mask` (filtrado por zonas), GPS y la función `draw_in_map`.
6. **Actividades** (celdas 98-101): las 3 preguntas, con el experimento real de `food` (15 matches analizados imagen por imagen).

## Resultados consolidados

### Demo sobre TotalText (300 imágenes de test)

| Métrica | Valor medido |
|---|---|
| Throughput | ~2.5 img/s (T4 GPU) |
| Latencia primera imagen | 2.25 s (*warmup* de kernels CUDA) |
| Latencia siguientes | ~0.17–0.5 s |
| Detecciones por imagen | de **0** (0000022) a **60** (0000056) |

### Disección de `0000089.jpg` — del tensor al texto

`instances.beziers` → `(12, 16)` · `instances.recs` → `(12, 25)`. Charset: `CTLABELS[i] = chr(32+i)` para `i∈[0,94]`; índice **95 = 口** (carácter desconocido, hardcodeado), índice **96 = blank/padding**. Las 12 palabras decodificadas a mano coincidieron **exactamente** con `_decode_recognition`:

```
['TURN','AHEAD','COFFEE','REAL','Cafe','ALLEY','Tea','RIGHT','SMALL','METERS','10','IN']
```

"COFFEE" y "REAL" tienen curvas Bézier arqueadas (la `y` del borde superior varía de 21 a 139 px) — el caso que una caja rectangular no captura y que motivó ABCNet.

### App 1 — Freiburg Groceries (OCR de productos alemanes)

| Top palabras (de 5000 imágenes) | Frecuencia |
|---|---|
| bio · real · milch · reis · honig | 254 · 165 · 103 · 86 · 79 |
| Marca "nestle" — match **exacto** | 17 |
| Marca "nestle" — con **fuzzy** (threshold 80) | +3 (≈20 total) |

→ El modelo, entrenado en inglés, leyó decenas de palabras alemanas limpias (transfer cross-idioma). Los caracteres especiales del alemán (ä, ö, ü) se colapsan a su letra base por estar fuera del charset: `Müsli → musli`, `Äpfel → apfel`.

### App 2 — Street View (efecto del filtrado por zonas)

| | Palabra más frecuente | Diagnóstico |
|---|---|---|
| **Antes del filtro** | "54" → **1096** ocurrencias | ruido del overlay de la UI de Google Maps |
| **Después del filtro** (`get_mask`) | top → **46** | texto real de calle (Penn Ave, stop, food) emerge |

→ Caída de **1096 → 46** (~24×). El experimento de `draw_in_map('food', ..., True)` devolvió 15 matches con diversidad real: George Aiken's ("Prepared Foods"), "Food Mart", "Food Court", "Fifth Avenue Place" y "DAIRY FOODS" — **no todos** son establecimientos de comida.

## Bloques del lab

{{< cards >}}
  {{< card link="instalacion-stack" title="Instalación y stack" subtitle="Detectron2 → AdelaiDet → ABCNet, parches de compatibilidad, arqueología de dependencias" icon="academic-cap" >}}
  {{< card link="demo-abcnet" title="Demo end-to-end" subtitle="Pesos preentrenados, TotalText, demo.py sobre 300 imágenes, análisis del log" icon="academic-cap" >}}
  {{< card link="diseccion-output" title="Disección del output (la joya)" subtitle="beziers (N,16) + recs (N,25), charset chr(32+i), glifo 口, decodificación a mano verificada" icon="beaker" >}}
  {{< card link="app-groceries" title="App 1 · Freiburg Groceries" subtitle="build_model + batching, OCR cross-idioma alemán, Counter, fuzzy matching de marcas" icon="academic-cap" >}}
  {{< card link="app-streetview" title="App 2 · Google Street View" subtitle="Ruido de la UI, get_mask por zonas, GPS, georreferenciación con draw_in_map" icon="academic-cap" >}}
  {{< card link="actividades" title="Actividades (3 preguntas)" subtitle="Palabras vs frases, transfer cross-idioma, y el experimento real de food (15 imágenes)" icon="academic-cap" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/abcnet-liu-2020" title="Liu et al. (2020) - ABCNet" subtitle="Scene text spotter end-to-end, curvas Bézier cúbicas, BezierAlign, recognizer attention" icon="document-text" >}}
  {{< card link="/papers/total-text-chng-2017" title="Ch'ng & Chan (2017) - TotalText" subtitle="Benchmark de texto curvo, anotación poligonal, dataset de fine-tuning del modelo" icon="document-text" >}}
  {{< card link="/papers/freiburg-groceries-jund-2016" title="Jund et al. (2016) - Freiburg Groceries" subtitle="5000 imágenes, 25 categorías de productos, dataset de la aplicación 1" icon="document-text" >}}
  {{< card link="/papers/street-view-geolocalization-zamir-2010" title="Zamir & Shah (2010) - Street View Geolocalization" subtitle="Google Street View + GPS, SIFT + voting, dataset de la aplicación 2 (UCF)" icon="document-text" >}}
  {{< card link="/papers/faster-rcnn-ren-2015" title="Ren et al. (2015) - Faster R-CNN" subtitle="Detector de objetos propuesto en la actividad 3 para filtrar por contexto" icon="document-text" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/scene-text-recognition" title="Fundamento: Scene Text Recognition" subtitle="STR vs OCR clásico, pipeline en stages, detección + reconocimiento" icon="book-open" >}}
  {{< card link="/fundamentos/bezier-curves" title="Fundamento: Curvas Bézier" subtitle="Bernstein, control points, parametrización del texto curvo" icon="variable" >}}
  {{< card link="/fundamentos/fuzzy-string-matching" title="Fundamento: Fuzzy String Matching" subtitle="Levenshtein, ratio de similaridad, record linkage, precision/recall" icon="adjustments" >}}
  {{< card link="/fundamentos/ctc-loss" title="Fundamento: CTC Loss" subtitle="Alineación implícita en reconocimiento de secuencias (alternativa al head de atención)" icon="variable" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-21" title="Clase 21 - Teoría" subtitle="Scene Text Recognition: STR vs OCR, stages, datasets, evaluation, ABCNet" icon="academic-cap" >}}
  {{< card link="/clases/clase-21/profundizacion" title="Profundización" subtitle="Math: curvas Bézier, BezierAlign, CTC vs attention, FCOS, Levenshtein" icon="beaker" >}}
  {{< card link="/dominios/vision" title="Dominio: Visión" subtitle="Timeline: del scene text spotting a los modelos end-to-end" icon="book-open" >}}
  {{< card link="/laboratorios/lab-22" title="Lab 22 - Summarization (siguiente)" subtitle="BertSum extractivo + T5 abstractivo, ROUGE" icon="academic-cap" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las 102 celdas del notebook con 6 páginas temáticas. Evidencia cuantitativa verificada en outputs reales (decodificación de 0000089 confirmada contra `_decode_recognition`; top 40 de ambos datasets; caída de ruido 1096→46 tras filtrado; 15 matches de `food` analizados imagen por imagen). Incluye el experimento real de la Actividad 3 (recuperación del dataset GSV vía copia a Drive tras bloqueo de cuota) y las respuestas a las 3 preguntas. Notebook ejecutado en Colab con T4 GPU.
