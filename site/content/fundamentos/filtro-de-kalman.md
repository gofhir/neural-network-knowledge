---
title: "Filtro de Kalman"
weight: 133
math: true
---

El **filtro de Kalman** es el estimador recursivo óptimo para un sistema lineal con ruido gaussiano. En [seguimiento de objetos](/fundamentos/seguimiento-de-objetos) cumple el rol del "modelo de movimiento": dado dónde estuvo un objeto, predice dónde estará en el próximo frame **y con cuánta incertidumbre**. Esa segunda mitad —la incertidumbre— es la que suele olvidarse y la que separa a [SORT](/papers/sort-bewley-2016) de [DeepSORT](/papers/deepsort-wojke-2017).

Este fundamento acompaña a la [Clase 42](/clases/clase-42).

---

## 1. El problema

Se quiere estimar el estado $x_t \in \mathbb{R}^n$ de un sistema que no se observa directamente. Lo que se observa es una medición ruidosa $z_t \in \mathbb{R}^m$. El modelo asume dos ecuaciones lineales:

$$x_t = F\,x_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q) \quad \text{(transición)}$$

$$z_t = H\,x_t + v_t, \qquad v_t \sim \mathcal{N}(0, R) \quad \text{(observación)}$$

donde $F$ es la matriz de transición, $H$ la de observación, $Q$ la covarianza del ruido de proceso (cuánto se aparta la realidad del modelo) y $R$ la del ruido de medición (cuánto miente el sensor).

Bajo estos supuestos, la distribución posterior del estado dado todas las mediciones hasta $t$ sigue siendo gaussiana, y basta con propagar su media $\hat{x}$ y su covarianza $P$. Ese es el resultado de [Kalman (1960)](/papers/kalman-1960): el estimador óptimo es recursivo y de forma cerrada, no hace falta guardar la historia.

## 2. Las dos fases

**Predicción** (*a priori*, sin mirar la nueva medición):

$$\hat{x}_{t|t-1} = F\,\hat{x}_{t-1|t-1}$$
$$P_{t|t-1} = F\,P_{t-1|t-1}\,F^{\top} + Q$$

**Corrección** (*a posteriori*, incorporando $z_t$):

$$\tilde{y}_t = z_t - H\,\hat{x}_{t|t-1} \qquad \text{(innovación)}$$
$$S_t = H\,P_{t|t-1}\,H^{\top} + R \qquad \text{(covarianza de la innovación)}$$
$$K_t = P_{t|t-1}\,H^{\top} S_t^{-1} \qquad \text{(ganancia de Kalman)}$$
$$\hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t\,\tilde{y}_t$$
$$P_{t|t} = (I - K_t H)\,P_{t|t-1}$$

{{< concept-alert type="clave" >}}
La **ganancia de Kalman** $K_t$ es un promedio ponderado entre lo que dice el modelo y lo que dice el sensor, y el peso lo fija la razón entre incertidumbres. Si $R \gg P$ (sensor malo), $K \to 0$ y el filtro ignora la medición. Si $P \gg R$ (modelo malo), $K \to H^{-1}$ y el filtro le cree al sensor. No hay que ajustarlo a mano: sale de la propagación de covarianzas.
{{< /concept-alert >}}

La consecuencia que importa para tracking: **si no hay medición, la fase de corrección no se ejecuta**. El estado se propaga solo con $F$ y la covarianza crece monótonamente, $P_{t|t-1} = F P F^\top + Q$, sumando $Q$ en cada paso. Un objeto ocluido durante 30 frames acumula 30 veces $Q$ en su incertidumbre.

## 3. La parametrización en tracking

En MOT el "sistema" es una caja delimitadora y el modelo de movimiento es **velocidad constante**. Las dos parametrizaciones canónicas:

**SORT** — estado de 7 dimensiones:

$$x = [u,\; v,\; s,\; r,\; \dot{u},\; \dot{v},\; \dot{s}]^{\top}$$

con $(u,v)$ el centro de la caja, $s$ su **área**, $r$ su razón de aspecto. Nótese que **no hay $\dot{r}$**: SORT trata la razón de aspecto como constante y no le asigna velocidad. Es una decisión deliberada, no un descuido: un peatón cambia de tamaño al acercarse pero mantiene su proporción.

**DeepSORT** — estado de 8 dimensiones:

$$x = [u,\; v,\; \gamma,\; h,\; \dot{u},\; \dot{v},\; \dot{\gamma},\; \dot{h}]^{\top}$$

con $\gamma$ la razón de aspecto y $h$ la **altura** en vez del área. DeepSORT sí modela $\dot{\gamma}$. El cambio de $s$ (área) a $h$ (altura) no es cosmético: la altura de un peatón en la imagen es aproximadamente proporcional a su distancia inversa, mientras que el área es cuadrática en esa misma cantidad — la altura se comporta de forma más lineal, que es exactamente lo que el modelo asume.

En ambos casos la observación es la caja detectada, $H$ selecciona las cuatro componentes de posición y descarta las velocidades:

$$H = \begin{bmatrix} I_4 & 0_{4\times 3}\end{bmatrix} \quad\text{(SORT)}, \qquad H = \begin{bmatrix} I_4 & 0_{4\times 4}\end{bmatrix} \quad\text{(DeepSORT)}$$

y la transición es la cinemática de velocidad constante con $\Delta t = 1$ frame:

$$F = \begin{bmatrix} I_4 & I_4 \\ 0 & I_4 \end{bmatrix}$$

(en SORT, la fila y columna correspondientes a $r$ se ajustan porque no tiene velocidad asociada).

**Inicialización.** Cuando nace una trayectoria solo se observa una caja, así que las velocidades se fijan en 0 — pero su covarianza se inicializa con **valores grandes**, porque no se sabe nada de ellas. Esa asimetría es la que permite que el filtro aprenda la velocidad en los primeros frames sin quedar anclado al cero.

## 4. Del filtro a la métrica: la distancia de Mahalanobis

Aquí está el punto que la [Clase 42](/clases/clase-42/profundizacion) desarrolla. El filtro no entrega un punto sino una **distribución** $\mathcal{N}(H\hat{x}, S)$ sobre la posición esperada. La distancia natural de una detección $d_j$ a esa distribución es la de Mahalanobis:

$$d^{(1)}(i,j) = (d_j - y_i)^{\top} S_i^{-1} (d_j - y_i)$$

que mide **en cuántas desviaciones estándar** está la detección del centro predicho, corrigiendo por la forma de la elipse de incertidumbre. Es la distancia euclídea aplicada tras blanquear el espacio con $S^{-1/2}$.

Bajo el supuesto gaussiano, esta cantidad se distribuye $\chi^2$ con $m$ grados de libertad, lo que da un umbral con interpretación estadística: para un espacio de medición de 4 dimensiones, el cuantil 0,95 es

$$t^{(1)} = \chi^2_{0{,}95;\,4} = 9{,}4877$$

y es exactamente el número que usa DeepSORT para descartar asociaciones improbables.

{{< concept-alert type="advertencia" >}}
**El malentendido frecuente.** Suele decirse que SORT "no modela la incertidumbre" y que DeepSORT la agrega. No es así: **ambos corren el mismo filtro de Kalman** y ambos tienen $S$. La diferencia está en la **métrica de asociación**: SORT compara con IoU, que solo mira el solapamiento geométrico y es ciego a $S$; DeepSORT compara con Mahalanobis, que la usa. La incertidumbre estaba ahí desde SORT — lo que cambia es si la asociación la consulta.
{{< /concept-alert >}}

## 5. Las dos patologías

**Mahalanobis premia la incertidumbre.** Como divide por $S$, una trayectoria que lleva 20 frames sin detección (y por lo tanto con $S$ inflada) tiene distancias *menores* a cualquier detección que una trayectoria recién actualizada. Cuando dos trayectorias compiten por la misma caja, gana la más incierta — exactamente al revés de lo deseable. DeepSORT lo contrarresta con una **cascada de matching** que resuelve primero las trayectorias de menor edad. Es una corrección de ingeniería a un defecto de la métrica.

**Amplificación temporal del error.** Durante una oclusión, el filtro se actualiza con sus propias predicciones. Un pequeño error de estimación de la velocidad, sin observaciones que lo corrijan, se integra linealmente en la posición y cuadráticamente en la varianza. [OC-SORT](/papers/oc-sort-cao-2022) muestra que tras una oclusión larga la dirección de movimiento estimada puede quedar completamente desviada, y propone re-actualizar el filtro *hacia atrás* con una trayectoria virtual construida entre las dos observaciones reales que rodean la oclusión, en vez de confiar en la propagación ciega.

## 6. Cuándo el supuesto se rompe

El filtro de Kalman es óptimo **bajo sus supuestos**: linealidad y ruido gaussiano. En video se rompen de tres formas:

1. **Movimiento no lineal.** Una persona que gira, un bailarín, un deportista que cambia de dirección. En intervalos cortos (un frame) la aproximación lineal es aceptable; en oclusiones largas no. El dataset DanceTrack existe justamente para exponer esto.
2. **Movimiento de cámara.** Un desplazamiento del sensor mueve todas las cajas a la vez sin que ningún objeto se haya movido. El modelo lo interpreta como aceleración de todos los objetos. La solución estándar es la **compensación de movimiento de cámara** (estimar la homografía entre frames y aplicarla al estado antes de predecir).
3. **El detector no es un sensor con $R$ conocida.** $R$ se ajusta a mano y se asume constante, cuando en realidad el ruido de una detección depende del tamaño de la caja, del nivel de oclusión y de la confianza del detector.

Las variantes no lineales —filtro de Kalman extendido (EKF), *unscented* (UKF), filtros de partículas— existen para el primer punto, pero en MOT rara vez se usan: la evidencia empírica es que arreglar la asociación rinde más que sofisticar el modelo dinámico.

---

## Ver también

- [Kalman (1960)](/papers/kalman-1960) — el paper original.
- [SORT](/papers/sort-bewley-2016) y [DeepSORT](/papers/deepsort-wojke-2017) — las dos parametrizaciones canónicas.
- [OC-SORT](/papers/oc-sort-cao-2022) — la crítica al uso ciego del filtro durante oclusiones.
- [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos) — dónde encaja este componente.
- [Asignación Húngara](/fundamentos/asignacion-hungara) — qué se hace con la matriz de costo que produce.
- [Clase 42 — Práctica](/clases/clase-42/practica) — implementación desde cero, en triple framework.
