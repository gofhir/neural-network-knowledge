---
title: "Lab 17 - Pose Recognition: comparación de modelos + clasificación de acciones"
weight: 170
sidebar:
  open: true
---

**Profesor:** Tomás Vergara Browne
**Fecha:** Mayo 2026
**Notebook origen:** `clase_17/material/Laboratorio/Lab_17.ipynb`

## Encuadre

Laboratorio que recorre **el ciclo completo de selección y aplicación de un modelo de pose**, organizado en dos bloques:

- **Bloque enseñado** — comparación cuantitativa de **OpenPifPaf vs. OpenPose** como feature extractors para clasificación downstream de acciones (4 clases del Stanford 40 Actions Dataset). Cada modelo alimenta un MLP simple de 3 capas que aprende a mapear configuraciones de keypoints a categorías de acción.
- **Actividad evaluable** — pipeline end-to-end aplicado a una imagen nueva: entrenar MLP sobre `running` vs. `riding_a_bike`, correr OpenPifPaf sobre una imagen multi-persona, clasificar cada esqueleto detectado y colorearlo con rojo/azul según la predicción.

El lab te enseña, implícitamente, **dos lecciones operacionales clave**:

1. **Pretrain + cabeza ligera es el patrón estándar de ML aplicado**: el modelo de pose es un *feature extractor congelado* y un MLP pequeño aprende la tarea downstream. Cubre el 80% de los casos productivos.
2. **El mejor modelo depende del caso de uso, no del leaderboard**: la comparación se evalúa por **accuracy del clasificador downstream**, no por PCK directo sobre los keypoints.

Para la teoría detrás de los modelos ver la [clase 17](/clases/clase-17).

## Tres modelos, tres paradigmas, tres APIs

El lab instala y prueba **tres librerías de pose** sobre la misma imagen, exponiendo la heterogeneidad del ecosistema:

| Librería | Modelo | Paradigma | Personas | Instalación |
|---|---|---|---|---|
| **OpenPifPaf** | [PifPaf](/papers/pifpaf-kreiss-2019) | Bottom-up moderno | Multi | `pip install openpifpaf==0.10.1` |
| **MediaPipe** | [BlazePose](/papers/blazepose-bazarevsky-2020) | Detector-tracker single-person | 1 | `pip install mediapipe==0.10.13` |
| **OpenPose** | [OpenPose](/papers/openpose-cao-2017) | Bottom-up legacy | Multi | `git clone` + `wget` + `sys.path hack` |

La fricción de instalación es **inversamente proporcional a la modernidad del modelo**. OpenPose, pese a su impacto histórico, exige scaffolding manual porque su código original es Caffe (framework muerto desde 2018). El alumno aprende empíricamente por qué la industria prefiere PifPaf/MediaPipe sobre el research code académico.

![Comparación visual: imagen base COCO procesada por las 3 librerías](/laboratorios/lab-17/demo-pifpaf-coco.jpg)

> Output del demo de OpenPifPaf sobre la imagen base — multi-persona detectado con colores diferenciados por limb.

## Experimento principal: A/B test PifPaf vs OpenPose

Estructura del experimento, alineada con el principio de **comparación justa**:

- **Mismo dataset**: Stanford 40 Actions, subset de 4 clases (`playing_guitar`, `climbing`, `riding_a_horse`, `cutting_vegetables`).
- **Mismo MLP**: 3 capas ocultas con 128 unidades, BCEWithLogitsLoss, Adam lr=1e-3, 100 epochs.
- **Mismo training/test split**: 80/20.
- **Única variable**: el modelo de pose (PifPaf vs. OpenPose).

| Etapa | PifPaf | OpenPose |
|---|---|---|
| Feature size | 51 (17×3) | 54 (18×3) |
| Parámetros del MLP | 44,840 | 45,224 |
| Loss final (training) | ~0.05 | ~0.05 |
| Accuracy final (test) | depende del split aleatorio (~75-85%) | depende del split aleatorio (~70-80%) |

**Caveats experimentales** que el lab no mitiga pero conviene reconocer:

- Sin `random_state` fijo, los splits PifPaf y OpenPose son **distintos** entre sí → la diferencia residual de accuracy tiene ±5% de ruido.
- Sin `stratify`, las clases pueden quedar desbalanceadas en train/test.
- `MAX_SAMPLES = 2000` con orden alfabético no-shuffleado limita la representación de clases que empiezan con letras tardías del alfabeto.

## Actividad evaluable: pipeline end-to-end

La actividad combina las piezas en un pipeline completo aplicado a una **imagen nueva** (foto del blog de [Honbike](https://www.honbike.com) sobre running vs. biking):

```
Imagen Honbike (multi-persona)
   │
   ▼ OpenPifPaf → predicciones (9 esqueletos detectados)
   │
   ▼ MLP entrenado sobre running vs. riding_a_bike
   │
   ▼ is_running(pred) → True/False por persona
   │
   ▼ Visualización: rojo (corredor) vs azul (ciclista)
```

### Resultados finales

Con un MLP entrenado con early stopping (Best checkpoint en epoch ~5 sobre dataset reducido):

| Métrica | Valor |
|---|---|
| Mejor accuracy en test (durante training) | **65.62%** |
| Epochs ejecutadas (early stopping con patience=15) | ~20 |
| Personas detectadas por OpenPifPaf | 9 |
| Clasificadas como corredor (rojo) | 7 |
| Clasificadas como ciclista (azul) | 2 |

![Resultado final con colores diferenciados por clase](/laboratorios/lab-17/actividad-final-clasificada.jpg)

> Visualización final: 9 esqueletos detectados por OpenPifPaf, coloreados según predicción del MLP. Los 2 ciclistas grandes a la derecha son los azules; los 7 rojos incluyen los corredores en primer plano y caminantes del centro. El ciclista del frente (clasificado erróneamente como corredor) es uno de los errores esperables dado el accuracy ~65% del clasificador.

## Recursos del lab — Bloque enseñado

{{< cards >}}
  {{< card link="instalacion-tres-modelos" title="Instalación de las 3 librerías" subtitle="OpenPifPaf, MediaPipe, OpenPose — tres patrones de friction" icon="academic-cap" >}}
  {{< card link="demos-tres-librerias" title="Demos sobre la misma imagen" subtitle="Inferencia + visualización de PifPaf, MediaPipe, OpenPose" icon="academic-cap" >}}
  {{< card link="dataset-stanford40" title="Stanford 40 Actions Dataset" subtitle="Descarga, Stanford40Dataset PyTorch, subset de 4 clases" icon="academic-cap" >}}
  {{< card link="clasificador-pifpaf" title="Clasificador MLP con representaciones PifPaf" subtitle="Pipeline completo: features 51-D → MLP → accuracy" icon="academic-cap" >}}
  {{< card link="clasificador-openpose" title="Clasificador MLP con representaciones OpenPose" subtitle="Mismo MLP, features 54-D, comparación final" icon="academic-cap" >}}
{{< /cards >}}

## Recursos del lab — Actividad evaluable

{{< cards >}}
  {{< card link="actividad-running-vs-bike" title="Pipeline end-to-end running vs. riding_a_bike" subtitle="MLP + OpenPifPaf + visualización condicional sobre imagen Honbike" icon="check-circle" >}}
  {{< card link="analisis-resultados" title="Análisis de resultados y errores" subtitle="Aciertos, errores y limitaciones del experimento" icon="light-bulb" >}}
{{< /cards >}}

## Notebook (Colab + descarga)

{{< cards >}}
  {{< card link="/notebooks/lab17.ipynb" title="Notebook ejecutado" subtitle="Lab_17_rae.ipynb con outputs completos (.ipynb descargable, ~10 MB)" icon="document" >}}
  {{< card link="/notebooks-html/lab17.html" title="Render HTML" subtitle="Notebook ejecutado en HTML con imágenes embebidas (~9.7 MB)" icon="document-text" >}}
{{< /cards >}}

## Papers de esta clase

{{< cards >}}
  {{< card link="/papers/openpose-cao-2017" title="OpenPose (Cao 2017)" subtitle="Primer bottom-up multi-persona en tiempo real, ganador COCO 2016" icon="document-text" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="PifPaf (Kreiss 2019)" subtitle="Bottom-up moderno para baja resolución y oclusión" icon="document-text" >}}
  {{< card link="/papers/blazepose-bazarevsky-2020" title="BlazePose (Bazarevsky 2020)" subtitle="Detector-tracker single-person mobile, MediaPipe Pose" icon="document-text" >}}
{{< /cards >}}

## Cross-links

{{< cards >}}
  {{< card link="/clases/clase-17" title="Clase 17 - Teoría" subtitle="Recorrido de las 59 diapositivas de Pose Recognition" icon="academic-cap" >}}
  {{< card link="/fundamentos/pose-estimation" title="Fundamento: Pose Estimation 2D" subtitle="Bottom-up vs. top-down, heatmaps, OKS/PCK" icon="book-open" >}}
  {{< card link="/laboratorios/lab-15" title="Lab 15 - Faster R-CNN" subtitle="Detección de objetos como precursor de pose (cabezas adicionales)" icon="academic-cap" >}}
{{< /cards >}}

---

> **Estado:** Lab completo. Cubre las 47 celdas del notebook original con 7 páginas temáticas, resultados reales del entrenamiento integrados (PifPaf + OpenPose + actividad), análisis crítico de errores en la visualización final. Reproducible end-to-end en Colab versión 2025.10 con GPU T4 en ~15-30 minutos.
