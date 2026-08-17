---
title: "DeepSORT: SORT with a Deep Association Metric (2017)"
weight: 447
math: true
---

{{< paper-card
    title="Simple Online and Realtime Tracking with a Deep Association Metric"
    authors="Nicolai Wojke, Alex Bewley, Dietrich Paulus (Universität Koblenz-Landau / QUT)"
    year="2017"
    venue="ICIP 2017 / arXiv:1703.07402"
    arxiv="1703.07402"
    pdf="/papers/deepsort-wojke-2017.pdf" >}}
La extensión de [SORT](/papers/sort-bewley-2016) que ataca su debilidad declarada: los cambios de identidad tras una oclusión. La receta mantiene la filosofía del original —todo lo caro se paga **offline**— y agrega un descriptor de apariencia de 128 dimensiones aprendido sobre un dataset de re-identificación de personas. Durante el seguimiento no se aprende nada: se hacen consultas de vecino más cercano en el espacio de apariencia. El resultado es una reducción de los **ID switches de 1423 a 781 (−45 %)** en MOT16, con MOTA subiendo apenas de 59,8 a 61,4. Aporta además dos ideas que sobreviven a la arquitectura: la **compuerta de Mahalanobis** con umbral $\chi^2$ y la **cascada de matching**, que corrige una patología contraintuitiva del filtro de Kalman.
{{< /paper-card >}}

---

## El diagnóstico

El paper es explícito sobre qué falla en SORT y por qué: *"la métrica de asociación empleada solo es precisa cuando la incertidumbre de la estimación de estado es baja"*. La distancia IoU compara solapamientos geométricos; si la predicción se desvía —porque el objeto estuvo ocluido varios frames, o porque la cámara se movió—, el solapamiento cae a cero y la asociación se rompe aunque el objeto esté claramente ahí.

La solución no es un mejor modelo de movimiento sino una **métrica más informada**, que combine movimiento y apariencia.

## Método

**Estado.** Ocho dimensiones, un cambio sutil pero significativo respecto de SORT:

$$x = (u,\; v,\; \gamma,\; h,\; \dot{x},\; \dot{y},\; \dot{\gamma},\; \dot{h})$$

centro $(u,v)$, razón de aspecto $\gamma$, **altura** $h$ y las cuatro velocidades. SORT usaba el área $s$ y trataba la razón de aspecto como constante; DeepSORT usa la altura y sí le da velocidad a $\gamma$.

**Métrica de movimiento.** La distancia de Mahalanobis entre la predicción proyectada y cada detección:

$$d^{(1)}(i,j) = (d_j - y_i)^{\top} S_i^{-1} (d_j - y_i)$$

Con ella se define una **compuerta binaria** usando el cuantil 0,95 de la $\chi^2$ con 4 grados de libertad (la dimensión del espacio de medición):

$$b^{(1)}_{i,j} = \mathbb{1}\left[d^{(1)}(i,j) \leq t^{(1)}\right], \qquad t^{(1)} = 9{,}4877$$

**Métrica de apariencia.** Una CNN pre-entrenada produce, para cada detección, un descriptor $r_j$ con $\lVert r_j \rVert = 1$. Cada trayectoria conserva una **galería** de los últimos $L_k = 100$ descriptores asociados, y la distancia es el mínimo coseno sobre esa galería:

$$d^{(2)}(i,j) = \min\{\,1 - r_j^\top r_k^{(i)} \;\mid\; r_k^{(i)} \in \mathcal{R}_i \,\}$$

**Combinación.** Suma ponderada de las dos, con una asociación admisible solo si pasa **ambas** compuertas:

$$c_{i,j} = \lambda\, d^{(1)}(i,j) + (1-\lambda)\, d^{(2)}(i,j), \qquad b_{i,j} = \prod_{m=1}^{2} b^{(m)}_{i,j}$$

{{< concept-alert type="advertencia" >}}
**El detalle que casi nunca se menciona: en los experimentos del paper, $\lambda = 0$.**

Los autores lo declaran sin rodeos: *"encontramos que fijar $\lambda = 0$ es una elección razonable cuando hay movimiento de cámara sustancial. En ese ajuste, solo se usa información de apariencia en el término de costo de asociación. Sin embargo, la compuerta de Mahalanobis sigue usándose para descartar asignaciones inviables."*

Es decir: la ecuación de mezcla existe, pero la configuración evaluada **no mezcla nada**. El costo de asociación es puramente apariencia; el movimiento entra solo como filtro binario. La lectura habitual de DeepSORT —"promedia Mahalanobis con coseno"— describe la fórmula, no el sistema que produjo los números.
{{< /concept-alert >}}

## La cascada de matching

La contribución algorítmica más fina del paper, y la que rara vez se cita. El argumento parte de una observación contraintuitiva:

> Cuando un objeto está ocluido mucho tiempo, las predicciones sucesivas del filtro de Kalman **aumentan** la incertidumbre de su posición. Intuitivamente, la métrica de asociación debería aumentar la distancia para reflejar esa dispersión. **Contraintuitivamente, la distancia de Mahalanobis favorece la incertidumbre mayor**, porque reduce efectivamente la distancia en desviaciones estándar de cualquier detección hacia la media proyectada.

Dicho de otro modo: $S^{-1}$ crece cuando $S$ crece, así que una trayectoria con covarianza inflada obtiene distancias *menores* a todo. Cuando dos trayectorias compiten por la misma detección, **gana la más incierta** — exactamente al revés de lo deseable, y el resultado son fragmentaciones y trayectorias inestables.

La solución es no resolver un problema de asignación global sino una **secuencia** de problemas, ordenados por edad:

```
para n = 1 … A_max:
    T_n ← trayectorias que llevan exactamente n frames sin asociarse
    resolver asignación húngara entre T_n y las detecciones aún libres
    retirar las detecciones asignadas
```

Las trayectorias vistas más recientemente eligen primero. Y al final se ejecuta una pasada extra de **IoU al estilo SORT** sobre las trayectorias no confirmadas y las de edad 1, para absorber cambios bruscos de apariencia por oclusión parcial con geometría estática.

## El descriptor

| Propiedad | Valor |
|---|---|
| Arquitectura | *wide residual network*: 2 convoluciones + 6 bloques residuales |
| Parámetros | 2 800 864 |
| Salida | 128-D, con *batch* y normalización $\ell_2$ → hiperesfera unitaria |
| Dataset de entrenamiento | MARS: 1 100 000 imágenes de 1 261 peatones |
| Latencia | ~30 ms por lote de 32 cajas en una GTX 1050 móvil |

La normalización $\ell_2$ final es lo que hace compatible la salida con la distancia coseno. No hay *metric learning* durante el seguimiento — la red se entrena una vez y se congela.

## Resultados

MOT16, comparando contra SORT re-ejecutado sobre **exactamente las mismas detecciones** (las de Yu et al., POI), con $\lambda=0$ y $A_{\max}=30$:

| Método | Tipo | MOTA↑ | MOTP↑ | MT↑ | ML↓ | **ID↓** | FM↓ | FP↓ | FN↓ |
|---|---|---|---|---|---|---|---|---|---|
| SORT | Online | 59,8 | **79,6** | 25,4 % | 22,7 % | 1423 | **1835** | **8698** | 63245 |
| **DeepSORT** | Online | **61,4** | 79,1 | **32,8 %** | **18,2 %** | **781** | 2008 | 12852 | **56668** |

Lo que la tabla dice de verdad:

- **ID switches −45 %.** Es el objetivo declarado y se cumple limpiamente.
- **MOTA sube solo 1,6 puntos.** La razón está en el [fundamento de métricas](/fundamentos/metricas-de-tracking): MOTA suma FN, FP e IDSW con el mismo peso, y hay decenas de miles de los dos primeros contra cientos del tercero. Ahorrar 642 ID switches es invisible en esa escala.
- **Los falsos positivos suben 48 %** (8698 → 12852). Es el costo directo de $A_{\max}=30$: mantener trayectorias vivas medio segundo sin evidencia significa que las respuestas espurias del detector sobre geometría estática se consolidan en trayectorias estables. Los autores lo inspeccionan visualmente y confirman el diagnóstico: no son trayectorias saltando entre falsas alarmas, sino trayectorias estacionarias falsas.
- **Las fragmentaciones suben** (1835 → 2008), efecto secundario de sostener identidades a través de oclusiones.

Sobre velocidad, el paper es internamente inconsistente: la tabla reporta 40 Hz, y el texto dice que *"nuestra implementación corre a aproximadamente 20 Hz, con más o menos la mitad del tiempo dedicado a la generación de features"*. La cifra medida es la del texto.

## Limitaciones

- **Requiere GPU.** La mitad del cómputo se va en el extractor de apariencia. Se pierde la propiedad más atractiva de SORT —260 Hz en una CPU— y con ella los escenarios *edge*.
- **El descriptor está atado a su dominio.** Entrenado sobre peatones de vigilancia; transferirlo a vehículos, células o animales exige re-entrenar.
- **Sigue suponiendo movimiento moderado.** La compuerta de Mahalanobis se calcula sobre un filtro de velocidad constante sin compensación de cámara. Con movimiento de cámara fuerte, la compuerta puede descartar la asociación correcta — y por eso los autores terminan con $\lambda=0$, desconfiando de su propio término de movimiento.
- **Más falsos positivos.** El diseño intercambia recall de identidad por precisión de detección, y en MOTA ese intercambio casi no rinde.

## Por qué importa para la Clase 42

La [Clase 42](/clases/clase-42) presenta DeepSORT como *"una ligera modificación de SORT que agrega features aprendidos"*, y recorre la distancia de Mahalanobis con la imagen clásica de la elipse frente a la distancia euclídea. Tres precisiones que el paper permite hacer y que la [profundización](/clases/clase-42/profundizacion) desarrolla:

1. **La mezcla que la clase describe es la ecuación (5); la configuración evaluada usa $\lambda=0$.** Mahalanobis funciona como compuerta, no como costo.
2. **La cascada de matching no aparece en la clase**, y es la contribución algorítmica más original del paper — además de un ejemplo precioso de cómo una métrica estadísticamente correcta puede tener un incentivo perverso.
3. **El diagrama que la clase muestra en la sección de DeepSORT** —con las cajas *Regression / Classification / Detection*, *Kill $b_t^k$?* e *Init new $b_t^k$*— no proviene de este paper: es la figura de arquitectura de [Tracktor](/papers/tracktor-bergmann-2019) (Bergmann et al., 2019), un método posterior con un principio distinto.

---

**Ver también:** [SORT (2016)](/papers/sort-bewley-2016) · [Tracktor (2019)](/papers/tracktor-bergmann-2019) · [FairMOT (2020)](/papers/fairmot-zhang-2020) · [ByteTrack (2021)](/papers/bytetrack-zhang-2021) · [Re-identificación](/fundamentos/re-identificacion) · [Filtro de Kalman](/fundamentos/filtro-de-kalman)
