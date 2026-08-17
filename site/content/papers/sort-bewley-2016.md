---
title: "SORT: Simple Online and Realtime Tracking (2016)"
weight: 446
math: true
---

{{< paper-card
    title="Simple Online and Realtime Tracking"
    authors="Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft (QUT / University of Sydney)"
    year="2016"
    venue="ICIP 2016 / arXiv:1602.00763"
    arxiv="1602.00763"
    pdf="/papers/sort-bewley-2016.pdf" >}}
El paper que fijó la línea base de todo el seguimiento multi-objeto moderno, y lo hizo **quitando** componentes en vez de agregarlos. SORT combina dos técnicas de los años cincuenta y sesenta —el [filtro de Kalman](/papers/kalman-1960) y el [algoritmo húngaro](/fundamentos/asignacion-hungara)— sobre las detecciones de un [Faster R-CNN](/papers/faster-rcnn-ren-2015), sin modelo de apariencia, sin re-identificación y sin ningún tratamiento explícito de la oclusión. Con eso alcanza **34,0 de MOTA** en el benchmark MOT —el mejor entre los trackers online de su momento y comparable a métodos batch mucho más complejos— corriendo a **260 Hz** en un solo núcleo de CPU, más de 20 veces más rápido que sus competidores. Su hallazgo central es incómodo para el área: **cambiar el detector, dejando el tracker intacto, mueve MOTA de 15,1 a 34,0**. Buena parte de lo que se creía un problema de seguimiento era un problema de detección.
{{< /paper-card >}}

---

## Contexto: por qué un paper minimalista

En 2015-2016, las primeras posiciones del MOTChallenge estaban ocupadas por métodos de asociación de datos maduros y caros — *Multiple Hypothesis Tracking* (MHT) y *Joint Probabilistic Data Association* (JPDA)— cuya complejidad combinatoria es exponencial en el número de objetos y que, además, retrasan las decisiones difíciles hasta tener más evidencia. Eso los vuelve inaplicables en línea.

Los autores hacen dos observaciones sobre esa tabla de resultados. La primera es que el compromiso entre exactitud y velocidad estaba muy marcado: los trackers más precisos eran demasiado lentos para tiempo real. La segunda, y más punzante: **el único tracker que no usaba el detector ACF era también el mejor rankeado**, lo que sugería que la calidad de la detección estaba conteniendo a todos los demás.

De ahí la pregunta del paper, formulada explícitamente: *¿cuán simple puede ser el MOT y cuán bien puede funcionar?* La respuesta se construye con la navaja de Occam — se ignora todo lo que no sea la asociación frame a frame.

## Método

**Detección.** [Faster R-CNN](/papers/faster-rcnn-ren-2015) con los parámetros por defecto de PASCAL VOC; solo se pasan al tracker las detecciones de clase *persona* con probabilidad mayor a 50 %.

**Modelo de estado.** Cada objeto se representa con siete variables:

$$x = [u,\; v,\; s,\; r,\; \dot{u},\; \dot{v},\; \dot{s}]^{\top}$$

centro $(u,v)$, área $s$, razón de aspecto $r$, y las velocidades de las tres primeras. **La razón de aspecto se considera constante** y por eso no tiene velocidad asociada. Cuando una detección se asocia a una trayectoria, el estado se corrige con el [filtro de Kalman](/fundamentos/filtro-de-kalman); cuando no, se propaga la predicción sin corregir.

**Asociación de datos.** Se predice la caja de cada trayectoria en el frame actual, se construye la matriz de costo con la **distancia IoU** entre cada predicción y cada detección nueva, y se resuelve con el [algoritmo húngaro](/fundamentos/asignacion-hungara). Se impone un mínimo $\mathrm{IoU}_{\min}$ por debajo del cual la asignación se rechaza.

**Ciclo de vida.** Una detección que no se asocia a nada inicia una trayectoria con velocidad cero y **covarianza de velocidad muy alta**, reflejando que esa componente aún no se observó. La trayectoria pasa por un periodo de prueba antes de reportarse, para no seguir falsos positivos. Y se elimina si pasa $T_{\text{lost}}$ frames sin detección.

{{< concept-alert type="advertencia" >}}
**$T_{\text{lost}} = 1$.** En todos los experimentos del paper, una trayectoria se mata tras **un solo** frame sin detección. Los autores justifican el valor con dos razones honestas: el modelo de velocidad constante es un mal predictor de la dinámica real, y la re-identificación está declaradamente fuera del alcance del trabajo. Si el objeto reaparece, *"el seguimiento se reanudará implícitamente bajo una identidad nueva"* — es decir, SORT **asume el ID switch** en vez de intentar evitarlo.

Este parámetro es la diferencia de diseño más grande con [DeepSORT](/papers/deepsort-wojke-2017), que lo lleva a $A_{\max}=30$.
{{< /concept-alert >}}

## El hallazgo: la detección domina

El experimento más influyente del paper es una ablación de dos filas. Manteniendo el tracker fijo y cambiando solo el detector, sobre las secuencias de validación:

| Tracker | Detector | Recall | Precisión | ID Sw | **MOTA** |
|---|---|---|---|---|---|
| MDP | ACF | 36,6 | 75,8 | 222 | 24,0 |
| MDP | FrRCNN (ZF) | 46,2 | 67,2 | 245 | 22,6 |
| MDP | FrRCNN (VGG16) | 50,1 | 76,0 | 178 | **33,5** |
| SORT | ACF | 33,6 | 65,7 | 224 | 15,1 |
| SORT | FrRCNN (ZF) | 41,3 | 72,4 | 347 | 24,0 |
| SORT | FrRCNN (VGG16) | 49,5 | 77,5 | 274 | **34,0** |

**+18,9 puntos de MOTA sin tocar una línea del algoritmo de seguimiento.** Y un detalle que suele pasarse por alto: los ID switches **suben** (224 → 274) con el mejor detector, porque hay más trayectorias vivas que confundir. La métrica global mejora mientras el componente que el tracker sí controla empeora.

## Resultados

En el servidor de test del MOT benchmark:

| Método | Tipo | MOTA↑ | MOTP↑ | MT↑ | ML↓ | ID sw↓ |
|---|---|---|---|---|---|---|
| NOMT | Batch | 33,7 | 71,9 | 12,2 % | 44,0 % | **442** |
| TDAM | Online | 33,0 | 72,8 | 13,3 % | 39,1 % | 464 |
| MDP | Online | 30,3 | 71,3 | 13,0 % | 38,4 % | 680 |
| **SORT** | Online | **33,4** | 72,1 | 11,7 % | **30,9 %** | 1001 |

SORT es el mejor tracker **online** en MOTA y queda a 0,3 puntos de NOMT, un método batch considerablemente más complejo que además consulta frames futuros. Tiene el menor porcentaje de objetos *mostly lost* de la tabla. Y paga el precio esperado por su diseño: **más del doble de ID switches** que NOMT, consecuencia directa de $T_{\text{lost}}=1$.

En velocidad, el componente de seguimiento corre a **260 Hz** en un núcleo de un Intel i7 de 2,5 GHz. El detector, naturalmente, queda fuera de esa cifra.

## Limitaciones

Las que el propio paper declara, sin adornos:

- **La oclusión no se trata.** Se la considera lo bastante infrecuente como para que su manejo explícito no compense la complejidad que introduce. Los ID switches lo pagan.
- **No hay modelo de apariencia.** Solo posición y tamaño de la caja. Dos personas que se cruzan con trayectorias compatibles pueden intercambiar identidades sin que nada lo impida.
- **Velocidad constante e independencia.** El modelo de movimiento ignora la aceleración, las interacciones entre objetos y el movimiento de cámara.
- **Solo peatones.** Aunque el diseño es agnóstico a la clase, la evaluación se limita a personas.

Los autores lo dicen con claridad al cerrar: la simplicidad del marco lo hace **adecuado como línea base**, y su expectativa explícita es que los métodos siguientes se concentren en la re-identificación para manejar la oclusión de largo plazo. Eso es literalmente lo que hace [DeepSORT](/papers/deepsort-wojke-2017) un año después, con uno de los autores en común.

## Por qué importa para la Clase 42

SORT es el eje de la [Clase 42](/clases/clase-42). La clase lo presenta como *"un paper muy influyente en el área, que con un algoritmo sencillo tenía buenos resultados"*, y desarrolla sus cuatro piezas —estado, predicción, IoU, húngaro— en el orden del paper.

Hay dos puntos donde conviene leer el paper junto a la clase. El primero: la clase describe el modelo de movimiento de SORT como *"calcular una velocidad y así aproximar la nueva localización"*, y contrasta eso con DeepSORT, que *"sí considera las incertezas"*. En rigor, **SORT ya corre un filtro de Kalman completo** y por lo tanto propaga covarianzas; lo que ocurre es que su métrica de asociación —IoU— **no las consulta**. La diferencia entre ambos no es tener o no tener incertidumbre, sino usarla o no en la asociación. Está desarrollado en la [profundización](/clases/clase-42/profundizacion).

El segundo: el número que la clase no menciona y que es el hallazgo más transferible del paper — que en *tracking-by-detection*, **mejorar el detector rinde más que mejorar el tracker**. Es la primera pregunta que conviene hacerse antes de invertir en asociación sofisticada.

---

**Ver también:** [DeepSORT (2017)](/papers/deepsort-wojke-2017) · [ByteTrack (2021)](/papers/bytetrack-zhang-2021) · [OC-SORT (2022)](/papers/oc-sort-cao-2022) · [Filtro de Kalman](/fundamentos/filtro-de-kalman) · [Asignación Húngara](/fundamentos/asignacion-hungara) · [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos)
