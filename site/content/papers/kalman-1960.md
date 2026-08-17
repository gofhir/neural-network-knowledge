---
title: "Filtro de Kalman: A New Approach to Linear Filtering (1960)"
weight: 448
math: true
---

{{< paper-card
    title="A New Approach to Linear Filtering and Prediction Problems"
    authors="Rudolf E. Kálmán (Research Institute for Advanced Study, Baltimore)"
    year="1960"
    venue="Transactions of the ASME — Journal of Basic Engineering, 82(D), 35–45" >}}
El paper que reformuló el problema clásico de filtrado de Wiener-Kolmogorov en el lenguaje del **espacio de estados** y lo resolvió de manera **recursiva**. En vez de una integral sobre toda la historia de la señal, un par de ecuaciones que actualizan una media y una covarianza con cada nueva observación. Eso hizo el filtrado óptimo computable en máquinas de la época y con memoria acotada — la razón por la que voló al programa Apollo y por la que, sesenta años después, sigue siendo el modelo de movimiento de [SORT](/papers/sort-bewley-2016) y [DeepSORT](/papers/deepsort-wojke-2017). Es un paper de teoría de control, no de visión, y llega al seguimiento de objetos por adopción.
{{< /paper-card >}}

*El artículo original es de ASME (1960) y no está disponible en acceso abierto por vías estándar; la copia de referencia habitual se distribuye desde [la página de Greg Welch en UNC](https://www.cs.unc.edu/~welch/kalman/kalmanPaper.html).*

---

## Qué resolvió

El problema de Wiener (1949) —estimar una señal a partir de observaciones ruidosas minimizando el error cuadrático medio— tenía solución conocida, pero en forma de una ecuación integral (Wiener-Hopf) que exigía la función de autocorrelación completa del proceso y suponía estacionariedad. Era teóricamente satisfactoria e prácticamente incómoda.

Kálmán la reformula sobre tres decisiones que en conjunto cambian el problema:

1. **Modelo de estado.** El proceso se describe por un vector de estado que evoluciona con una ecuación de diferencias lineal, en vez de por su función de correlación. Esto admite procesos **no estacionarios** de forma natural.
2. **Recursividad.** El estimador solo necesita la estimación anterior y la observación actual. **No hay que guardar la historia**: toda la información pasada está resumida en la media y la covarianza actuales.
3. **Proyección ortogonal.** La derivación se apoya en que, bajo hipótesis gaussianas, la esperanza condicional es la proyección ortogonal sobre el subespacio generado por las observaciones — lo que da la optimalidad en error cuadrático medio con un argumento geométrico limpio.

## Las ecuaciones

Con transición $x_t = F x_{t-1} + w_t$ y observación $z_t = H x_t + v_t$, ruidos gaussianos de covarianzas $Q$ y $R$:

$$\hat{x}_{t|t-1} = F\hat{x}_{t-1|t-1}, \qquad P_{t|t-1} = FP_{t-1|t-1}F^\top + Q$$
$$K_t = P_{t|t-1}H^\top\left(HP_{t|t-1}H^\top + R\right)^{-1}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t\left(z_t - H\hat{x}_{t|t-1}\right), \qquad P_{t|t} = (I - K_tH)P_{t|t-1}$$

La **ganancia** $K_t$ es el objeto central: pondera automáticamente cuánto creerle al modelo y cuánto al sensor según la razón entre sus incertidumbres, sin ningún parámetro que ajustar a mano. El desarrollo completo está en el fundamento [Filtro de Kalman](/fundamentos/filtro-de-kalman).

## Supuestos, y qué pasa cuando fallan

La optimalidad vale bajo tres condiciones: **linealidad** de transición y observación, ruido **gaussiano** de **media cero**, y matrices $Q$, $R$ **conocidas**. Fuera de ellas, el filtro sigue siendo el mejor estimador *lineal* pero deja de ser el mejor estimador.

En seguimiento de objetos las tres se violan a la vez, y cada violación tiene su literatura:

- La linealidad falla con movimiento no rectilíneo — de ahí el filtro extendido (EKF), el *unscented* (UKF) y los filtros de partículas.
- $Q$ y $R$ nunca se conocen: en MOT se ajustan a mano y se asumen constantes, cuando el ruido real de una detección depende de su tamaño, su nivel de oclusión y la confianza del detector.
- La media cero falla con movimiento de cámara, que introduce un desplazamiento sistemático de todas las cajas.

## Por qué importa para la Clase 42

La [Clase 42](/clases/clase-42) llega a Kálmán por la puerta de atrás. SORT lo cita como una de sus dos piezas prestadas, y la clase describe su papel de forma implícita al decir que DeepSORT *"estima regiones de probabilidad de la siguiente localización, en lugar de un lugar en particular"*. Esa frase es exactamente la descripción de lo que hace el filtro: propagar una distribución, no un punto.

El matiz que conviene tener presente al leer la clase es que **ese objeto ya existía en SORT**. Ambos trackers corren el mismo filtro y ambos disponen de la covarianza $S$. Lo que DeepSORT agrega no es la incertidumbre sino una métrica que la consulta —[Mahalanobis](/fundamentos/filtro-de-kalman)— en lugar de una que la ignora —IoU—. Ver la [profundización](/clases/clase-42/profundizacion) de la clase.

---

**Ver también:** [Filtro de Kalman](/fundamentos/filtro-de-kalman) · [SORT (2016)](/papers/sort-bewley-2016) · [DeepSORT (2017)](/papers/deepsort-wojke-2017) · [OC-SORT (2022)](/papers/oc-sort-cao-2022) · [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos)
