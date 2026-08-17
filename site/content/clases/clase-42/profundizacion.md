---
title: "Profundización - Lo que la incertidumbre hace, y lo que la métrica no ve"
weight: 20
math: true
---

> La [teoría](teoria) presentó dos algoritmos y una progresión: SORT asocia con IoU, DeepSORT agrega apariencia y Mahalanobis, y el segundo es "más robusto". Esta página desarma esa frase en cinco partes. Se deriva qué hace realmente el filtro de Kalman dentro de SORT; se muestra —con números— que la distancia de Mahalanobis tiene un incentivo perverso que obliga a la cascada de matching; se reconstruye la aritmética de MOTA para descubrir cuánto vale de verdad la mejora de DeepSORT; se mide cuándo el algoritmo húngaro le gana al codicioso; y se cierra con lo que pasó entre 2018 y 2023, que la clase salta.
>
> Todas las cifras marcadas como **medidas** provienen de código ejecutado; el mismo que se desarrolla en la [práctica](practica).

---

## Parte I — La incertidumbre ya estaba en SORT

### I.1. Lo que la clase afirma, y lo que dice el paper

La clase presenta la transición así:

> *"Un problema con SORT es que el modelo de movimiento es demasiado simple: calculan una velocidad y así aproximan la nueva localización. Eso no considera las incertezas existentes en las mediciones."*

La primera mitad es correcta —velocidad constante—; la segunda no lo es. El paper de [SORT](/papers/sort-bewley-2016) es explícito: *"las componentes de velocidad se resuelven óptimamente vía un marco de filtro de Kalman"*. Y un filtro de Kalman **es** un modelo de incertidumbre: propaga una covarianza $P$ en cada paso y la usa para calcular la ganancia.

De hecho SORT usa la incertidumbre en un lugar visible: al inicializar una trayectoria, las velocidades se ponen en cero pero *"la covarianza de la componente de velocidad se inicializa con valores grandes, reflejando esta incertidumbre"*. Sin ese modelo, la velocidad quedaría anclada al cero de inicialización.

### I.2. Dónde está entonces la diferencia

En **qué se hace con $S$**, la covarianza de la innovación.

$$S_t = H P_{t|t-1} H^\top + R$$

- **SORT** construye su matriz de costo con $1 - \mathrm{IoU}$. El IoU es una función puramente geométrica de dos rectángulos: **no ve $S$ en absoluto**. La elipse de incertidumbre existe, se calcula, y se descarta.
- **DeepSORT** construye su compuerta con $d^{(1)} = (d_j - y_i)^\top S_i^{-1}(d_j-y_i)$, que **es $S$ aplicada**.

{{< concept-alert type="clave" >}}
La formulación correcta de la diferencia no es *"SORT no modela la incertidumbre y DeepSORT sí"*, sino:

**Ambos la modelan; solo uno la consulta al asociar.**

El matiz importa porque explica por qué la mejora es más chica de lo esperado: si SORT hubiera sido ciego a la incertidumbre, agregársela debería haber cambiado todo. Como ya la tenía, lo que DeepSORT aporta es una métrica que la lee — y, sobre todo, un descriptor de apariencia y un $A_{\max}$ treinta veces mayor.
{{< /concept-alert >}}

### I.3. La aritmética de la oclusión

Durante una oclusión no hay corrección, y la covarianza crece sin freno:

$$P_{t|t-1} = F P_{t-1|t-1} F^\top + Q$$

Con velocidad constante en una dimensión, $F = \begin{bmatrix}1 & 1\\ 0 & 1\end{bmatrix}$ y $Q = I$, partiendo de $P_0 = I$ (**medido**):

| frames sin detección | 1 | 2 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| $\mathrm{var}(\text{pos})$ | 3,0 | 8,0 | 61,0 | 396,0 | 2891,0 | 9486,0 |
| $\sigma(\text{pos})$ [px] | 1,73 | 2,83 | 7,81 | 19,90 | 53,77 | **97,40** |

El crecimiento es **cúbico en el tiempo**: cada paso agrega $Q$ y además el término $F P F^\top$ mezcla la varianza de velocidad en la de posición proporcionalmente a $\Delta t^2$. A los 30 frames —el $A_{\max}$ de DeepSORT— la desviación estándar de la posición es de unos **97 px**, del orden del ancho de una persona en un plano medio.

**Consecuencia sobre la compuerta $\chi^2$.** DeepSORT descarta asociaciones con $d^{(1)} > t^{(1)} = \chi^2_{0{,}95;4} = 9{,}4877$. Con covarianza isotrópica, eso define un radio admisible $r = \sqrt{t^{(1)}\sigma^2}$ (**medido**):

| frames sin detección | 0 | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| radio admisible [px] | 3,08 | 5,34 | 24,06 | 61,30 | 165,62 | **300,00** |

Al llegar a $A_{\max}$, la compuerta acepta cualquier detección en un radio de 300 px. En un cuadro de 1920×1080 eso ya casi no filtra nada. **La compuerta se autodesactiva justo cuando más falta hace.**

## Parte II — Por qué Mahalanobis necesita la cascada

### II.1. El incentivo perverso

El paper de [DeepSORT](/papers/deepsort-wojke-2017) formula la objeción con precisión y la califica de contraintuitiva:

> *"Contraintuitivamente, la distancia de Mahalanobis favorece la incertidumbre mayor, porque reduce efectivamente la distancia en desviaciones estándar de cualquier detección hacia la media proyectada de la trayectoria."*

Es aritmética directa: $d^{(1)} \propto \lVert d - y\rVert^2 / \sigma^2$. Duplicar $\sigma$ divide la distancia por cuatro. La misma detección, sin moverse un píxel, se vuelve cuatro veces "más cercana" a una trayectoria que se volvió el doble de incierta (**medido**, con la detección fija a 10 px del centro predicho):

| $\sigma$ [px] | 1,0 | 2,0 | 5,0 | 10,0 |
|---|---|---|---|---|
| $d^{(1)}$ | 100,00 | 25,00 | 4,00 | **1,00** |

### II.2. Cuándo se rompe, exactamente

El escenario crítico es el de dos trayectorias compitiendo por la misma detección. Sea:

- **Track A**: visto hace 1 frame, predicción a **5 px** de la detección.
- **Track B**: visto hace $k$ frames, predicción a **40 px** de la detección.

La respuesta correcta es siempre A: está ocho veces más cerca y su predicción es confiable. Lo que decide Mahalanobis (**medido**):

| edad de B | $\sigma_A$ | $\sigma_B$ | $d^{(1)}_A$ | $d^{(1)}_B$ | gana |
|---|---|---|---|---|---|
| 5 frames | 1,73 | 7,81 | 8,33 | 26,23 | A ✓ |
| **10 frames** | 1,73 | 19,90 | 8,33 | **4,04** | **B ✗** |
| 25 frames | 1,73 | 74,51 | 8,33 | 0,29 | B ✗ |
| 40 frames | 1,73 | 148,93 | 8,33 | 0,07 | B ✗ |

**El punto de quiebre está entre los 5 y los 10 frames sin detección.** A partir de ahí, una trayectoria perdida hace un tercio de segundo se roba sistemáticamente las detecciones de las trayectorias que se están siguiendo bien. El resultado, en palabras del paper, es *"fragmentaciones aumentadas y trayectorias inestables"*.

### II.3. La cascada como parche estructural

La solución de DeepSORT no toca la métrica: cambia el **orden de resolución**. En vez de un problema de asignación global sobre todas las trayectorias, resuelve una secuencia de problemas por edad creciente:

$$\text{para } n = 1, \dots, A_{\max}: \quad \mathcal{T}_n \leftarrow \{i : a_i = n\}, \quad \text{asignar}(\mathcal{T}_n,\, \mathcal{U})$$

Las trayectorias jóvenes eligen primero y retiran sus detecciones del conjunto disponible. Para cuando le toca a $\mathcal{T}_{25}$, la detección de la tabla anterior ya no está.

{{< concept-alert type="clave" >}}
Esto es más interesante que un truco de implementación: es un caso de **métrica estadísticamente correcta con incentivo perverso bajo asignación competitiva**. $d^{(1)}$ responde bien la pregunta que le corresponde —*¿es esta detección compatible con esta trayectoria?*— pero no la que el sistema le hace —*¿cuál de estas dos trayectorias es la dueña de esta detección?*—. Comparar verosimilitudes de modelos con distinta dispersión sin normalizar por la dispersión es el mismo error que comparar $p$-valores de tests con distinto poder.

La corrección "correcta" sería incluir el término $\log\det S$ de la verosimilitud gaussiana completa,

$$-\log p(d \mid y, S) = \tfrac{1}{2}d^{(1)} + \tfrac{1}{2}\log\det S + \text{cte}$$

que penaliza explícitamente las distribuciones dispersas. DeepSORT no lo hace: usa la solución heurística de ordenar por edad. Funciona, y es más barata.
{{< /concept-alert >}}

## Parte III — Cuánto vale realmente la mejora de DeepSORT

### III.1. Reconstruir MOTA

La tabla de MOT16 del paper de DeepSORT reporta, sobre las mismas detecciones:

| | MOTA | FP | FN | IDSW |
|---|---|---|---|---|
| SORT | 59,8 | 8698 | 63245 | 1423 |
| DeepSORT | 61,4 | 12852 | 56668 | 781 |

De $\mathrm{MOTA} = 1 - (\mathrm{FN}+\mathrm{FP}+\mathrm{IDSW})/|\mathrm{gtDet}|$ se puede **despejar el denominador**, que el paper no reporta:

$$|\mathrm{gtDet}|_{\text{SORT}} = \frac{73366}{1 - 0{,}598} = 182\,502, \qquad |\mathrm{gtDet}|_{\text{DeepSORT}} = \frac{70301}{1 - 0{,}614} = 182\,127$$

Los dos despejes coinciden dentro del error de redondeo de los MOTA publicados. Tomando $|\mathrm{gtDet}| = 182\,326$, la fórmula reproduce ambas filas (**medido**): **59,76** y **61,44**, contra 59,8 y 61,4 reportados. La aritmética cierra, lo que permite hacer contrafactuales.

### III.2. Los contrafactuales

**¿Cuánto MOTA vale reducir los ID switches un 45 %?** Se toman los FP y FN de SORT y se le regalan los 781 ID switches de DeepSORT:

$$\mathrm{MOTA} = 1 - \frac{8698 + 63245 + 781}{182326} = 60{,}11$$

**+0,35 puntos** (medido). Eliminar 642 cambios de identidad —el logro central del paper, su contribución declarada— vale **un tercio de punto** en la métrica con que se rankeaba el benchmark.

**¿Cuánto cuestan los falsos positivos de $A_{\max}=30$?** Se le dan a DeepSORT los FP de SORT:

$$\mathrm{MOTA} = 1 - \frac{8698 + 56668 + 781}{182326} = 63{,}72$$

**+2,28 puntos** (medido). Los 4154 falsos positivos adicionales que introduce mantener trayectorias vivas medio segundo cuestan **6,5 veces más MOTA de lo que aporta arreglar todos los ID switches**.

{{< concept-alert type="advertencia" >}}
De aquí se sigue algo incómodo. **La mayor parte de los 1,6 puntos de MOTA que DeepSORT gana no vienen de su contribución declarada.** Vienen de reducir los falsos negativos (63245 → 56668, −6577), que es consecuencia de mantener las trayectorias vivas más tiempo — y ese efecto lo produce el parámetro $A_{\max}$, no el descriptor de apariencia.

Los ID switches aportan 0,35 puntos. Los FP restan 2,28. Los FN aportan el resto.

Esto no significa que DeepSORT no funcione: reducir un 45 % los cambios de identidad es una mejora real y la aplicación la nota. Significa que **MOTA no la mide**, y que evaluar este trabajo con MOTA era el instrumento equivocado. Es exactamente el argumento que [HOTA](/papers/hota-luiten-2020) formaliza tres años después.
{{< /concept-alert >}}

### III.3. Por qué MOTA hace esto

Sus tres términos entran con el mismo peso pero no con la misma magnitud. En estas dos filas, la fracción del numerador que aportan los ID switches (**medido**) es **1,94 %** en SORT y **1,11 %** en DeepSORT. El otro 98 % son errores de detección.

Toda métrica que sume errores heterogéneos sin normalizar por su frecuencia termina midiendo el término más numeroso. La corrección de HOTA no es cambiar los pesos sino **separar los ejes** y combinarlos con una media geométrica:

$$\mathrm{HOTA}_\alpha = \sqrt{\mathrm{DetA}_\alpha\cdot\mathrm{AssA}_\alpha}$$

que es invariante a la escala relativa de los conteos porque cada factor se normaliza dentro de su propio eje. Ver [Métricas de Tracking](/fundamentos/metricas-de-tracking).

### III.4. El ejemplo de HOTA, reconstruido

El paper de HOTA presenta tres trackers sobre un mismo *ground truth* de 100 detecciones de un solo objeto: A predice una trayectoria de 50; B, dos de 35; C, cuatro de 25. Reimplementando las tres métricas desde su definición (**medido**):

| Tracker | DetA | AssA | MOTA | IDF1 | HOTA | IDSW |
|---|---|---|---|---|---|---|
| A — 1 track de 50 | 50,0 | 50,0 | 50,0 | 66,7 | 50,0 | 0 |
| B — 2 tracks de 35 | 70,0 | 35,0 | 69,0 | **41,2** | 49,5 | 1 |
| C — 4 tracks de 25 | 100,0 | 25,0 | 97,0 | 25,0 | 50,0 | 3 |

La reconstrucción reproduce **exactamente** DetA, AssA, MOTA y HOTA de los tres trackers, y el IDF1 de A (66,7 ≈ 67) y de C (25,0). Difiere en el IDF1 de B: 41,2 % contra el 52 % que reporta el paper. La figura del artículo es esquemática y no fija todos los detalles de la construcción —cualquier reparto de las 70 detecciones de B en trayectorias de longitudes distintas cambia IDF1 sin cambiar DetA—, así que la discrepancia es de la reconstrucción, no del paper.

El punto cualitativo se sostiene, y con la reconstrucción queda **más marcado**: MOTA ordena C > B > A (97 > 69 > 50), IDF1 ordena A > B > C (66,7 > 41,2 > 25,0), y HOTA los declara empatados. Tres métricas, tres órdenes, los mismos tres sistemas.

## Parte IV — Cuándo importa el algoritmo húngaro

La clase presenta el húngaro como el mecanismo de asignación óptima. La pregunta práctica es cuánto se pierde con la alternativa codiciosa, que es trivial de implementar y más barata.

Que el codicioso **puede** fallar es fácil de mostrar. Con

$$C = \begin{bmatrix} 1 & 2 \\ 3 & 100\end{bmatrix}$$

el codicioso toma el mínimo global (1) y queda obligado a pagar 100: total **101**. El húngaro elige 2 y 3: total **5** (**medido**). Un factor 20 de diferencia.

Pero eso no dice con qué frecuencia ocurre en escenas reales. Midiéndolo sobre escenas sintéticas de objetos que se cruzan, con 20 semillas por configuración (**medido**):

| Escena | Método | MOTA | IDF1 | HOTA | ID switches | semillas donde difieren |
|---|---|---|---|---|---|---|
| 12 objetos, $\sigma=1$ px | húngaro | 99,93 | 99,95 | 99,93 | 0,40 | **1 / 20** |
| | codicioso | 99,89 | 99,91 | 99,89 | 0,65 | |
| 25 objetos, $\sigma=6$ px | húngaro | **67,20** | **67,05** | **67,33** | **120,35** | **20 / 20** |
| | codicioso | 62,80 | 62,40 | 63,16 | 145,50 | |

{{< concept-alert type="recordar" >}}
**El húngaro es irrelevante en escenas fáciles y decisivo en escenas difíciles.** Con objetos bien separados y detecciones limpias, la compuerta de IoU deja tan pocos candidatos por trayectoria que la asignación es casi forzada, y las dos estrategias coinciden en 19 de 20 semillas. Con 25 objetos y ruido de 6 px, difieren siempre, y el húngaro gana **4,4 puntos de MOTA y 25 ID switches** en promedio.

La implicación operativa: la elección solo se puede evaluar en el régimen de densidad al que se va a desplegar el sistema. Medida en una escena fácil, la conclusión "da lo mismo" es correcta y engañosa a la vez.
{{< /concept-alert >}}

## Parte V — Los siete años que la clase salta

La clase pasa de DeepSORT (2017) a SUTrack (2024) sin escalas. Lo que ocurrió en el medio contradice la lectura de que el progreso vino de agregar apariencia y modelos más grandes.

### V.1. Tracktor (2019): eliminar la asociación

[Tracktor](/papers/tracktor-bergmann-2019) observa que la cabeza de regresión de un detector ya sabe ajustar una caja a un objeto, así que se le puede pasar la caja del frame anterior sobre la imagen actual y obtener la posición nueva **con la identidad intacta**. Sin filtro de Kalman, sin IoU, sin húngaro, sin entrenamiento sobre datos de seguimiento.

Y su análisis con oráculos concluye que *"ninguno de los métodos dedicados de seguimiento es considerablemente mejor manejando escenarios complejos"* — lo que los métodos sofisticados ganaban eran los casos fáciles.

Es también el origen del diagrama que la clase muestra en la sección de DeepSORT.

### V.2. ByteTrack (2021): el umbral es del tracker

[ByteTrack](/papers/bytetrack-zhang-2021) ataca una decisión que en la clase queda enterrada en el paso 1 —*"solo pasamos detecciones con probabilidad mayor a 50 %"*— y nunca se revisa. Su observación: **una detección de score bajo puede ser un objeto ocluido**, y descartarla produce una fragmentación irreversible.

La solución es asociar en dos rondas: score alto primero, y luego las trayectorias huérfanas contra el score bajo, usando solo movimiento. Con eso obtiene 63,1 de HOTA en MOT17 **sin ningún modelo de apariencia**.

### V.3. OC-SORT (2022): dejar de creerle al filtro

[OC-SORT](/papers/oc-sort-cao-2022) identifica el defecto estructural del uso de Kalman en MOT: cuando no hay observaciones, el filtro se actualiza con sus propias predicciones, **realimentando el error**. Su corrección es reconstruir el filtro hacia atrás a partir de las observaciones reales que rodean la oclusión.

### V.4. El resultado que reordena todo

En **DanceTrack** —bailarines con vestuario similar, oclusión severa, movimiento fuertemente no lineal—:

| Tracker | Año | HOTA | AssA |
|---|---|---|---|
| SORT | 2016 | **47,9** | 31,2 |
| DeepSORT | 2017 | **45,6** | 29,7 |
| ByteTrack | 2021 | 47,3 | 31,4 |
| OC-SORT | 2022 | 54,6 | 40,2 |

**DeepSORT queda por debajo de SORT**, y ByteTrack tampoco lo supera. La razón es estructural: en DanceTrack la apariencia no discrimina —todos visten igual—, y como DeepSORT evalúa con $\lambda=0$, su costo de asociación *es* la apariencia. Un descriptor que no distingue nada es peor que ninguno, porque introduce ruido donde SORT tenía geometría.

{{< concept-alert type="clave" >}}
**La progresión SORT → DeepSORT → ByteTrack → OC-SORT no es una escalera.** Cada método domina un régimen distinto, definido por qué señal es informativa en ese dataset:

- apariencia discriminativa + oclusiones largas → **DeepSORT** y descendientes;
- detector fuerte + escenas densas → **ByteTrack**;
- movimiento no lineal + apariencia inútil → **OC-SORT**;
- vocabulario abierto y *prompts* en lenguaje → **SAM 3**.

El ranking en MOT17 esconde esto porque MOT17 es un dataset de peatones con apariencia variada y movimiento aproximadamente lineal — favorable a todos por igual. Antes de elegir un tracker, la pregunta no es cuál puntúa más alto sino **qué señal es informativa en el video propio**.
{{< /concept-alert >}}

---

## Resumen de lo verificado

| Afirmación | Resultado |
|---|---|
| $\chi^2_{0{,}95;4} = 9{,}4877$ | confirmado, es el umbral de DeepSORT |
| La fórmula de MOTA reproduce las dos filas de MOT16 | 59,76 y 61,44 contra 59,8 y 61,4 |
| Reducir los ID switches un 45 % vale, en MOTA | **+0,35 puntos** |
| Los FP extra de $A_{\max}=30$ cuestan, en MOTA | **−2,28 puntos** |
| Los ID switches como fracción del numerador de MOTA | 1,94 % (SORT), 1,11 % (DeepSORT) |
| Mahalanobis con $\sigma$ de 1 → 10 px, misma detección | la distancia cae de 100 a 1 |
| Punto de quiebre de la patología de Mahalanobis | entre 5 y 10 frames de edad |
| Radio admisible de la compuerta $\chi^2$ a los 30 frames | ~300 px |
| $\sigma$ de posición tras 30 frames de oclusión | 97,40 px |
| Húngaro contra codicioso, 12 objetos limpios | difieren en 1 / 20 semillas |
| Húngaro contra codicioso, 25 objetos con ruido | difieren en 20 / 20; **+4,4 MOTA** |
| El ejemplo de HOTA reconstruido | DetA, AssA, MOTA y HOTA exactos; IDF1 de B difiere |

---

**Siguiente:** la [práctica](practica) — SORT implementado desde cero, las tres métricas desde su definición, y todos estos experimentos reproducibles en triple framework.
