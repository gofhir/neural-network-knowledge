---
title: "OC-SORT: Observation-Centric SORT (2022)"
weight: 454
math: true
---

{{< paper-card
    title="Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking"
    authors="Jinkun Cao, Jiangmiao Pang, Xinshuo Weng, Rawal Khirodkar, Kris Kitani (CMU / Shanghai AI Lab / NVIDIA)"
    year="2022"
    venue="CVPR 2023 / arXiv:2203.14360"
    arxiv="2203.14360"
    pdf="/papers/oc-sort-cao-2022.pdf" >}}
Seis años después de [SORT](/papers/sort-bewley-2016), este paper demuestra que **un filtro de Kalman básico todavía alcanza el estado del arte** si se corrige el ruido que acumula durante las oclusiones. El diagnóstico tiene tres partes: SORT es sensible al ruido de las estimaciones de estado, acumula error con el tiempo, y —el punto conceptual— es **centrado en la estimación**: cuando no hay observaciones, confía en sus propias predicciones y las usa para actualizarse, realimentando el error. OC-SORT invierte la prioridad: usa las **observaciones** reales que rodean una oclusión para construir una trayectoria virtual y re-actualizar el filtro hacia atrás. Corre a **700+ FPS en una sola CPU** y gana en MOT17, MOT20, KITTI y especialmente en DanceTrack, donde el movimiento es fuertemente no lineal.
{{< /paper-card >}}

---

## Las tres limitaciones de SORT

**1. Sensibilidad al ruido de estimación.** El filtro de Kalman estima la velocidad a partir de la diferencia entre posiciones consecutivas. Como el desplazamiento entre frames es pequeño y el ruido de la caja no lo es, la relación señal-ruido de la velocidad es mala. El paper muestra que el error de dirección de movimiento resultante puede ser tan grande como el propio tamaño del objeto.

**2. Amplificación temporal del error.** Durante una oclusión, ese error se integra. Sin correcciones del detector, la posición estimada se aleja linealmente y la varianza crece en cada paso. Cuando el objeto reaparece, la predicción puede estar tan desviada que la IoU con la detección correcta sea cero.

**3. Ser centrado en la estimación.** La convención estándar cuando no hay medición es *confiar en la estimación a priori para la actualización a posteriori*. Es decir: el filtro se actualiza con lo que él mismo predijo. Formalmente es lo correcto bajo los supuestos del modelo; en la práctica significa que **el error se realimenta** y que la trayectoria se extiende con pura extrapolación.

{{< concept-alert type="clave" >}}
El diagnóstico central: **los errores no ocurren por la oclusión ni por la no linealidad por separado, sino cuando ocurren juntas**. Una oclusión corta con movimiento lineal la resuelve el filtro. Movimiento no lineal sin oclusión lo corrige el detector en cada frame. Es la combinación —el objeto gira mientras está oculto— la que rompe el sistema.
{{< /concept-alert >}}

## Los tres remedios

**ORU — *Observation-centric Re-Update*.** Cuando una trayectoria perdida se recupera en el frame $t_2$ tras haberse perdido en $t_1$, se construye una **trayectoria virtual** interpolando entre las dos observaciones **reales** $z_{t_1}$ y $z_{t_2}$, y se re-ejecuta el ciclo predicción-actualización del filtro a lo largo de ese tramo. En vez de heredar los parámetros contaminados por la extrapolación ciega, el filtro se reconstruye a partir de datos observados. Es un *smoothing* dirigido a la ventana de oclusión.

**OCM — *Observation-Centric Momentum*.** Se agrega a la matriz de costo un término de **consistencia de dirección**: se compara la dirección de movimiento implícita entre observaciones separadas por $\Delta t$ frames con la dirección que implicaría asociar la detección candidata.

$$C(\hat{X}, Z) = C_{\mathrm{IoU}}(\hat{X}, Z) + \lambda\, C_v(\mathcal{Z}, Z)$$

Usar observaciones separadas por $\Delta t$ (y no frames consecutivos) es lo que reduce el ruido de la estimación de dirección; hay un compromiso, porque un $\Delta t$ grande baja el ruido pero asume linealidad sobre un intervalo mayor.

**OCR — *Observation-Centric Recovery*.** Un segundo intento de asociación entre las trayectorias no emparejadas y las detecciones sobrantes, esta vez usando la **última observación real** de la trayectoria en vez de la predicción del filtro. Recupera objetos que se detuvieron o que reaparecen cerca de donde se los vio por última vez.

Las tres correcciones comparten el mismo principio: **cuando el filtro y la observación discrepan, creerle a la observación**.

## Resultados

MOT17 test, detecciones compartidas con ByteTrack:

| Tracker | HOTA↑ | MOTA↑ | IDF1↑ | FP (10⁴)↓ | IDs↓ | AssA↑ |
|---|---|---|---|---|---|---|
| ByteTrack | 63,1 | **80,3** | 77,3 | 2,55 | 2196 | 62,0 |
| **OC-SORT** | **63,2** | 78,0 | **77,5** | **1,51** | **1950** | **63,2** |

Empate técnico en HOTA con perfiles opuestos: ByteTrack detecta más (MOTA), OC-SORT asocia mejor y con **41 % menos falsos positivos**.

Donde la diferencia es real es en **DanceTrack** —bailarines con vestuario similar, oclusión severa y movimiento fuertemente no lineal—:

| Tracker | HOTA↑ | DetA↑ | AssA↑ | MOTA↑ | IDF1↑ |
|---|---|---|---|---|---|
| SORT (2016) | 47,9 | 72,0 | 31,2 | 91,8 | 50,8 |
| DeepSORT (2017) | 45,6 | 71,0 | 29,7 | 87,8 | 47,9 |
| ByteTrack (2021) | 47,3 | 71,6 | 31,4 | 89,5 | 52,5 |
| **OC-SORT** | **54,6** | **80,4** | **40,2** | 89,6 | **54,6** |

{{< concept-alert type="advertencia" >}}
**El dato incómodo de esta tabla: DeepSORT (45,6) queda por debajo de SORT (47,9).**

Es exactamente el escenario donde su ventaja se anula: los bailarines visten igual, así que el descriptor de apariencia —que en DeepSORT es *todo* el costo de asociación, con $\lambda=0$— no discrimina. ByteTrack tampoco supera a SORT aquí, porque el problema no es qué detecciones se descartan sino que la predicción de movimiento está mal.

La conclusión no es que DeepSORT sea malo, sino que **el orden de mérito entre trackers depende del dataset de un modo que las tablas de MOT17 esconden**. La progresión SORT → DeepSORT → ByteTrack → OC-SORT no es una escalera; cada método domina un régimen distinto.
{{< /concept-alert >}}

## Por qué importa para la Clase 42

La [Clase 42](/clases/clase-42) cierra la sección de SORT/DeepSORT observando que el enfoque *"supone que el objeto se movió poco"* y preguntando qué pasa con el movimiento de cámara y las oclusiones largas. OC-SORT es la respuesta más directa a esa pregunta, y su valor didáctico está en que **no cambia de paradigma para responderla**: sigue siendo Kalman más húngaro, corriendo en CPU, seis años después. Lo que cambia es a quién se le cree cuando el modelo y los datos discrepan.

Es también el contraejemplo útil frente a la narrativa de que el progreso en MOT vino de arquitecturas más grandes. La versión de 2022 con el estado del arte es el algoritmo de 2016 con tres correcciones de un par de decenas de líneas.

---

**Ver también:** [SORT (2016)](/papers/sort-bewley-2016) · [DeepSORT (2017)](/papers/deepsort-wojke-2017) · [ByteTrack (2021)](/papers/bytetrack-zhang-2021) · [Filtro de Kalman](/fundamentos/filtro-de-kalman) · [Métricas de Tracking](/fundamentos/metricas-de-tracking)
