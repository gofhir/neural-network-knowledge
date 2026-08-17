---
title: "Métricas de Tracking"
weight: 135
math: true
---

Evaluar un tracker es más difícil que evaluar un detector porque hay **dos cosas que pueden fallar por separado**: encontrar los objetos y mantener sus identidades. Un sistema puede detectar perfectamente y asignar identidades al azar; otro puede seguir con identidades impecables la mitad de los objetos e ignorar el resto. Ninguna métrica escalar los ordena bien a la vez, y la historia del área es la historia de ese conflicto.

Este fundamento acompaña a la [Clase 42](/clases/clase-42) y explica MOTA, MOTP, IDF1 y HOTA: qué mide cada una, dónde falla y por qué a veces se contradicen.

---

## 1. El vocabulario base

Antes de cualquier métrica hay que **emparejar** la salida del tracker con el *ground truth*. Para eso se define una similitud espacial $S$ entre cajas (normalmente IoU) y un umbral $\alpha$ (clásicamente 0,5). De ahí salen:

| Símbolo | Significado |
|---|---|
| **TP** | detección predicha emparejada con una del *ground truth* |
| **FP** | detección predicha sin correspondencia (predicción de más) |
| **FN** | detección del *ground truth* no cubierta (objeto perdido) |
| **IDSW** | un TP cuyo ID predicho difiere del ID predicho del TP anterior con el mismo ID verdadero |
| **Frag** | trayectoria interrumpida por una detección faltante y luego retomada |

**Dónde ocurre el emparejamiento** es la decisión de diseño que separa a las tres métricas grandes: MOTA empareja **por detección, frame a frame**; IDF1 empareja **por trayectoria completa**; HOTA empareja por detección pero puntúa por trayectoria.

## 2. MOTA — la métrica de la detección

Propuesta por Bernardin y Stiefelhagen (2008) como parte de las *CLEAR MOT metrics*, fue durante quince años **la** métrica del área:

$$\mathrm{MOTA} = 1 - \frac{|\mathrm{FN}| + |\mathrm{FP}| + |\mathrm{IDSW}|}{|\mathrm{gtDet}|}$$

Suma los tres tipos de error, los normaliza por el número de detecciones verdaderas y lo resta de 1.

Va acompañada de **MOTP**, que mide solo localización:

$$\mathrm{MOTP} = \frac{1}{|\mathrm{TP}|}\sum_{\mathrm{TP}} S$$

{{< concept-alert type="advertencia" >}}
**MOTA está dominada por la detección.** Los tres términos del numerador se suman con peso 1, pero sus órdenes de magnitud no son comparables. En MOT16, un tracker típico tiene decenas de miles de FN y FP y **unos cientos** de ID switches. Los errores de asociación aportan menos del 2 % del numerador: MOTA es, en la práctica, una métrica de detección con una corrección marginal.

Además **no está acotada por abajo**: si el sistema produce más falsos positivos que objetos verdaderos, MOTA es negativa.
{{< /concept-alert >}}

Un caso concreto que ilustra el problema: [DeepSORT](/papers/deepsort-wojke-2017) reduce los ID switches de 1423 a 781 respecto de [SORT](/papers/sort-bewley-2016) —una mejora del 45 % en lo que la métrica dice medir— y MOTA solo sube de 59,8 a 61,4. Peor: como DeepSORT mantiene vivas las trayectorias hasta 30 frames, sus falsos positivos suben de 8698 a 12852, lo que *resta* más de lo que aportaron los ID switches recuperados.

## 3. IDF1 — la métrica de la identidad

[Ristani et al. (2016)](/papers/idf1-ristani-2016) la introdujeron para seguimiento multi-cámara, donde lo que importa no es contar cajas sino saber si el sistema entendió que la persona de la cámara 3 es la misma de la cámara 1. El emparejamiento es **global, entre trayectorias completas**, resuelto con el [algoritmo húngaro](/fundamentos/asignacion-hungara) minimizando errores de identidad. De ahí salen IDTP, IDFP e IDFN, y:

$$\mathrm{IDF1} = \frac{|\mathrm{IDTP}|}{|\mathrm{IDTP}| + 0{,}5\,|\mathrm{IDFN}| + 0{,}5\,|\mathrm{IDFP}|}$$

que es el F1 estándar sobre detecciones correctamente **identificadas**.

Corrige el sesgo de MOTA hacia la detección, pero se pasa al otro extremo: sobrepondera la asociación y exhibe comportamiento **no monótono respecto de la detección** — agregar detecciones correctas puede *bajar* IDF1, si esas detecciones caen en una trayectoria que el emparejamiento global no eligió.

## 4. HOTA — separar los dos ejes

[Luiten et al. (2020)](/papers/hota-luiten-2020) parten del diagnóstico de que MOTA e IDF1 no son dos métricas rivales sino **dos proyecciones de un espacio de dos dimensiones**, y proponen medir las dos explícitamente.

Para cada TP $c$ se definen los conjuntos de asociación: **TPA** (los TP que comparten el mismo ID verdadero *y* el mismo ID predicho que $c$), **FNA** (los que comparten el ID verdadero pero no el predicho) y **FPA** (los que comparten el ID predicho pero no el verdadero). Con ellos:

$$A(c) = \frac{|\mathrm{TPA}(c)|}{|\mathrm{TPA}(c)| + |\mathrm{FNA}(c)| + |\mathrm{FPA}(c)|}$$

$$\mathrm{DetA}_\alpha = \frac{|\mathrm{TP}|}{|\mathrm{TP}| + |\mathrm{FN}| + |\mathrm{FP}|}, \qquad \mathrm{AssA}_\alpha = \frac{1}{|\mathrm{TP}|}\sum_{c\in \mathrm{TP}} A(c)$$

$$\mathrm{HOTA}_\alpha = \sqrt{\frac{\sum_{c \in \mathrm{TP}} A(c)}{|\mathrm{TP}| + |\mathrm{FN}| + |\mathrm{FP}|}} = \sqrt{\mathrm{DetA}_\alpha \cdot \mathrm{AssA}_\alpha}$$

Es una **doble Jaccard**: un índice de Jaccard sobre TP/FP/FN donde cada TP del numerador está pesado por *otro* índice de Jaccard, esta vez sobre los conjuntos de asociación. Y la métrica final integra sobre umbrales de localización:

$$\mathrm{HOTA} = \int_0^1 \mathrm{HOTA}_\alpha\, d\alpha \approx \frac{1}{19}\sum_{\alpha \in \{0{,}05,\,0{,}10,\,\dots,\,0{,}95\}} \mathrm{HOTA}_\alpha$$

con lo que incorpora también el error de **localización**, que ni MOTA ni IDF1 consideran (MOTP lo mide aparte y nadie lo reporta como criterio de ranking).

{{< concept-alert type="clave" >}}
**HOTA en una frase**, según sus autores: *mide qué tan bien se alinean las trayectorias de las detecciones emparejadas, promediado sobre todas las detecciones emparejadas, penalizando además las detecciones que no se emparejan.*

Ser **media geométrica** no es un detalle: $\sqrt{\mathrm{DetA}\cdot\mathrm{AssA}}$ castiga el desbalance. Un tracker con DetA = 1,0 y AssA = 0,25 obtiene 0,50, exactamente lo mismo que uno con 0,50 y 0,50. Una media aritmética habría premiado al primero.
{{< /concept-alert >}}

## 5. El ejemplo que las separa

La figura de apertura del paper de HOTA construye tres trackers sobre el mismo *ground truth*, con detección creciente y asociación decreciente:

| Tracker | DetA | AssA | **MOTA** | **IDF1** | **HOTA** |
|---|---|---|---|---|---|
| A | 50 % | 50 % | 50 % | **67 %** | 50 % |
| B | 70 % | 35 % | 69 % | 52 % | 50 % |
| C | 100 % | 25 % | **97 %** | 25 % | 50 % |

Las tres métricas ven los mismos tres sistemas y producen **tres órdenes distintos**: MOTA dice C > B > A, IDF1 dice A > B > C, HOTA los declara empatados. Ninguna está equivocada; miden cosas distintas. Lo que el ejemplo demuestra es que **reportar solo una es una decisión editorial**, no una medición neutral.

## 6. Qué reportar

La práctica actual en los benchmarks serios ([MOTChallenge](/papers/mot16-milan-2016), KITTI):

- **HOTA** como métrica principal de ranking, con **DetA** y **AssA** al lado. Con esas tres se entiende qué hace el sistema.
- **MOTA** por continuidad histórica y comparabilidad con la literatura previa.
- **IDF1** cuando la aplicación es de identidad (multi-cámara, re-identificación, conteo de personas únicas).
- **FP, FN, IDSW, Frag** en bruto: son los que permiten diagnosticar. Una subida de MOTA con IDSW constante es una mejora de detector, no de tracker.

{{< concept-alert type="recordar" >}}
Antes de comparar dos números de la literatura, hay que verificar tres cosas: **qué detecciones** usó cada sistema (públicas o privadas — la diferencia puede valer 20 puntos de MOTA), **qué split** (train, val o test server) y **qué umbral $\alpha$**. La mayoría de las comparaciones informales del área fallan en al menos una.
{{< /concept-alert >}}

---

## Ver también

- [HOTA (Luiten et al., 2020)](/papers/hota-luiten-2020) — la métrica actual y su análisis de las anteriores.
- [IDF1 (Ristani et al., 2016)](/papers/idf1-ristani-2016) — la métrica de identidad.
- [MOT16 (Milan et al., 2016)](/papers/mot16-milan-2016) — el benchmark que impuso MOTA.
- [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos) — los tres tipos de error que estas métricas reparten.
- [Ranking Metrics](/fundamentos/ranking-metrics) — el análogo en recomendación, donde el mismo problema de "una métrica no basta" reaparece.
- [Clase 42 — Práctica](/clases/clase-42/practica) — las tres métricas implementadas y el ejemplo de la tabla reproducido.
