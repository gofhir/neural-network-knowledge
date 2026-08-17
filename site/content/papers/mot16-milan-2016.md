---
title: "MOT16: A Benchmark for Multi-Object Tracking (2016)"
weight: 449
math: true
---

{{< paper-card
    title="MOT16: A Benchmark for Multi-Object Tracking"
    authors="Anton Milan, Laura Leal-Taixé, Ian Reid, Stefan Roth, Konrad Schindler"
    year="2016"
    venue="arXiv:1603.00831"
    arxiv="1603.00831"
    pdf="/papers/mot16-milan-2016.pdf" >}}
El benchmark que le dio al seguimiento multi-objeto lo que ImageNet le dio a la clasificación: un conjunto de secuencias fijo, un protocolo de anotación estricto, detecciones públicas compartidas y un servidor de evaluación con el *ground truth* de test oculto. Antes de MOTChallenge, cada paper elegía sus propias secuencias, su propio detector y su propia implementación de las métricas, y los números de la literatura eran incomparables entre sí. MOT16 son **14 secuencias** con 292 733 cajas anotadas y anotación de nivel de visibilidad por objeto. Su decisión más consecuente fue **rankear por MOTA**, lo que orientó una década de investigación hacia el eje que esa métrica premia.
{{< /paper-card >}}

---

## El problema que resuelve

El diagnóstico del paper sobre el estado del área en 2015 es duro y específico:

- **No hay conjuntos de datos comunes.** Cada trabajo elige sus secuencias, y las decisiones sobre train/test son ad hoc.
- **El protocolo de anotación no es consistente** entre secuencias, y algunas anotaciones son de calidad dudosa.
- **Muchas secuencias son fáciles**, y una vez saturadas los participantes migran a otras, lo que impide medir progreso.
- Las métricas se implementan de formas distintas y producen números distintos sobre los mismos datos.

MOTChallenge (2015) y su sucesor MOT16 responden con tres decisiones de diseño: secuencias fijas con división train/test declarada, **detecciones públicas** provistas por los organizadores para que todos partan del mismo punto, y evaluación **centralizada en un servidor** con el *ground truth* de test retenido.

{{< concept-alert type="clave" >}}
Las **detecciones públicas** son la decisión más importante y la más ignorada al leer tablas. Al fijar la entrada del tracker, aíslan la contribución de la asociación de la del detector. Un resultado con detecciones **privadas** (el equipo entrena su propio detector) no es comparable con uno público: la diferencia puede valer decenas de puntos de MOTA, como demuestra la ablación de [SORT](/papers/sort-bewley-2016).
{{< /concept-alert >}}

## Qué contiene

- **14 secuencias** (siete de entrenamiento, siete de test) con escenas más pobladas que MOT15, distintos puntos de vista, movimiento de cámara y condiciones climáticas variadas.
- **292 733 cajas** anotadas en total, sobre 2430 trayectorias de peatones.
- Anotación no solo de peatones sino también de **vehículos, objetos ocluyentes y otras clases**, y el **nivel de visibilidad** de cada caja individual — información granular que permite analizar el rendimiento en función del grado de oclusión, algo que ningún benchmark previo permitía.
- Detecciones pre-computadas y código de evaluación común.

## Las métricas y su consecuencia

MOT16 adopta las *CLEAR MOT metrics* (Bernardin y Stiefelhagen, 2008) con **MOTA** como criterio de ranking:

$$\mathrm{MOTA} = 1 - \frac{|\mathrm{FN}| + |\mathrm{FP}| + |\mathrm{IDSW}|}{|\mathrm{gtDet}|}$$

acompañada de MOTP, MT, ML, FP, FN, ID switches y fragmentaciones.

La elección tuvo un efecto de largo plazo que solo se hizo visible años después. Como MOTA suma errores de detección y de asociación con el mismo peso, y los primeros son dos órdenes de magnitud más numerosos, **el ranking del benchmark premia mejorar la detección mucho más que mejorar la asociación** — que es justamente el problema que el benchmark existe para medir. [HOTA](/papers/hota-luiten-2020) documenta este sesgo cuatro años después y se convierte en la métrica principal del challenge. El desarrollo está en el fundamento [Métricas de Tracking](/fundamentos/metricas-de-tracking).

## Por qué importa para la Clase 42

Todos los números que la [Clase 42](/clases/clase-42) menciona al comparar [SORT](/papers/sort-bewley-2016) con [DeepSORT](/papers/deepsort-wojke-2017) vienen de este benchmark. Sin él, la afirmación *"en la práctica DeepSORT es más robusto que SORT para mantener identidades"* no sería verificable: sería una impresión.

Y es el que permite hacer la lectura crítica que la clase no hace. Que DeepSORT reduzca los ID switches un 45 % mientras MOTA sube 1,6 puntos no es un detalle de implementación; es la métrica del benchmark diciendo que le importa poco lo que DeepSORT mejora.

---

**Ver también:** [HOTA (2020)](/papers/hota-luiten-2020) · [IDF1 (2016)](/papers/idf1-ristani-2016) · [SORT (2016)](/papers/sort-bewley-2016) · [Métricas de Tracking](/fundamentos/metricas-de-tracking) · [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos)
