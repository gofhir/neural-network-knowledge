---
title: "ByteTrack: asociar todas las cajas de detección (2021)"
weight: 453
math: true
---

{{< paper-card
    title="ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
    authors="Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang (HUST / HKU / ByteDance)"
    year="2021"
    venue="ECCV 2022 / arXiv:2110.06864"
    arxiv="2110.06864"
    pdf="/papers/bytetrack-zhang-2021.pdf" >}}
Una idea de una línea con un efecto desproporcionado. Todo tracker descarta las detecciones cuyo score cae bajo un umbral, porque suelen ser ruido. ByteTrack observa que **una detección de score bajo a veces es un objeto ocluido**, y que descartarla produce fragmentaciones irreversibles. Su método, BYTE, asocia en **dos rondas**: primero las cajas de score alto contra las trayectorias, y después las cajas de score **bajo** contra las trayectorias que quedaron huérfanas. El contexto de la trayectoria hace de discriminador — si una caja mala coincide con la predicción de un objeto que se estaba siguiendo, es el objeto; si no, es fondo. Aplicado sobre nueve trackers distintos mejora IDF1 entre 1 y 10 puntos, y su propia versión alcanza **80,3 MOTA, 77,3 IDF1 y 63,1 HOTA** en MOT17 a 30 FPS.
{{< /paper-card >}}

---

## El argumento

El pipeline estándar toma una decisión temprana e irreversible: umbralizar el score de detección. Bajo ese umbral, la caja desaparece del sistema. El paper cuestiona esa decisión con un caso concreto: una persona que se ocluye progresivamente ve caer su score de 0,9 a 0,4 y luego a 0,1. Con umbral 0,5, la trayectoria se corta en el frame donde empieza la oclusión y se reinicia con ID nuevo cuando reaparece — dos errores de asociación producidos por una decisión del detector.

La observación clave es que **el score de detección y la existencia del objeto no son lo mismo**. Un score bajo puede significar dos cosas muy distintas: fondo, o un objeto real difícil de ver. El detector no puede distinguirlas mirando un solo frame. El tracker sí, porque tiene la predicción de dónde debería estar el objeto.

## BYTE

El algoritmo separa las detecciones en $D_{\text{high}}$ y $D_{\text{low}}$ según un umbral, y asocia en dos etapas:

1. **Primera asociación**: trayectorias contra $D_{\text{high}}$, con la similitud habitual (IoU sobre la predicción del [filtro de Kalman](/fundamentos/filtro-de-kalman), o distancia de re-ID).
2. **Segunda asociación**: las trayectorias que quedaron sin par contra $D_{\text{low}}$, usando **solo similitud de movimiento**.
3. Las cajas de score bajo que no se asociaron a nada se descartan como fondo. **Nunca inician trayectorias nuevas.**

Esa última restricción es la que hace que el método no explote en falsos positivos: las detecciones dudosas solo pueden *continuar* una identidad existente, jamás crear una.

{{< concept-alert type="clave" >}}
**Por qué la segunda ronda usa movimiento y no apariencia.** Los autores lo explican: las cajas de score bajo suelen corresponder a objetos **severamente ocluidos o borrosos**, cuyos features de apariencia son poco confiables — justo el escenario donde un descriptor de re-ID se equivoca. El modelo de movimiento, en cambio, se comporta de forma más fiable ahí. Es el reverso exacto del argumento de [DeepSORT](/papers/deepsort-wojke-2017), y ambos son correctos en sus regímenes: la apariencia sirve para *reencontrar* tras una oclusión larga, el movimiento para *sostener* durante una oclusión corta.
{{< /concept-alert >}}

## Resultados

ByteTrack = **YOLOX-X** como detector + BYTE como asociación. En MOT17 (detecciones privadas):

| Tracker | HOTA↑ | MOTA↑ | IDF1↑ | IDs↓ |
|---|---|---|---|---|
| FairMOT | 59,3 | 73,7 | 72,3 | 3303 |
| TransMOT | 61,7 | 76,7 | 75,1 | 2346 |
| **ByteTrack** | **63,1** | **80,3** | **77,3** | 2196 |

Y el resultado más importante no es esa tabla sino la genericidad: aplicado como reemplazo del módulo de asociación de **nueve trackers distintos**, BYTE mejora IDF1 entre 1 y 10 puntos en todos. No es una arquitectura, es un componente reutilizable.

Los autores también observan que, en su configuración, **el filtro de Kalman simple basta para asociación de largo alcance** y da mejor IDF1 y menos ID switches que la re-identificación cuando las cajas de detección son de buena calidad — un resultado que va a contramano de la dirección que había tomado el área desde DeepSORT.

## Limitaciones

- **Depende de un detector muy fuerte.** Con YOLOX-X entrenado sobre CrowdHuman más los datos del benchmark, las detecciones de score alto ya son excelentes. Con un detector mediocre, la segunda ronda tiene mucho más ruido que filtrar.
- **Hereda los supuestos del filtro de Kalman.** En DanceTrack —movimiento no lineal, apariencia poco discriminativa— ByteTrack obtiene 47,3 de HOTA, por **debajo** del SORT original (47,9). El problema ahí no es qué cajas se descartan sino que la predicción de movimiento es incorrecta, y BYTE no lo toca.
- **Dos umbrales más que ajustar**, y el rendimiento es sensible a ellos.

## Por qué importa para la Clase 42

La [Clase 42](/clases/clase-42) llega hasta DeepSORT y salta directamente a los modelos integrados de 2024-2025. ByteTrack ocupa el hueco, y aporta la lección más transferible de todo el bloque: **el umbral de detección es una decisión del tracker, no del detector**. En el pipeline de la clase esa decisión está enterrada en el paso 1 (*"solo pasamos detecciones con probabilidad mayor a 50 %"*, en SORT) y nunca se revisa.

También reordena la narrativa de la clase. La secuencia SORT → DeepSORT sugiere que el progreso viene de agregar apariencia. ByteTrack, cinco años después, obtiene el estado del arte **sin ningún modelo de apariencia**, con un mejor detector y una mejor política sobre qué detecciones conservar.

---

**Ver también:** [SORT (2016)](/papers/sort-bewley-2016) · [DeepSORT (2017)](/papers/deepsort-wojke-2017) · [OC-SORT (2022)](/papers/oc-sort-cao-2022) · [FairMOT (2020)](/papers/fairmot-zhang-2020) · [Asignación Húngara](/fundamentos/asignacion-hungara)
