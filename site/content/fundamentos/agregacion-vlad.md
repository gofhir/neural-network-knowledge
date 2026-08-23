---
title: "Agregación VLAD"
weight: 131
math: true
---

**VLAD** (*Vector of Locally Aggregated Descriptors*) es un método para resumir un conjunto de descriptores locales de cardinalidad variable en un **vector de dimensión fija**, conservando mucha más información que un promedio. El problema que resuelve aparece en todas partes: una imagen produce cientos de descriptores locales, un enunciado produce cientos de vectores de frame, un video produce cientos de features por cuadro — y casi siempre hace falta un solo vector por objeto para poder compararlos. La respuesta ingenua es promediar. VLAD hace algo distinto: aprende un **diccionario de prototipos** y acumula, para cada uno, cuánto se **desvían** de él los descriptores que le fueron asignados. Este fundamento presenta el mecanismo, su versión diferenciable ([NetVLAD](/papers/netvlad-arandjelovic-2016)) y por qué la diferencia con el promedio no es cosmética.

---

## 1. El problema: de un conjunto a un vector

Formalmente: dado un conjunto $\{x_1, \dots, x_N\}$ de descriptores en $\mathbb{R}^d$, con $N$ **variable**, producir un único vector $v$ de dimensión fija tal que objetos parecidos den vectores parecidos.

Las opciones habituales, en orden de estructura:

| Método | Qué produce | Dimensión | Qué conserva |
|---|---|---|---|
| Average pooling | $\frac{1}{N}\sum_i x_i$ | $d$ | el centro de masa |
| Max pooling | $\max_i x_i$ (componente a componente) | $d$ | los extremos |
| Statistics pooling | $[\mu \,\|\, \sigma]$ | $2d$ | centro y dispersión global |
| *Bag of features* | histograma de asignaciones | $k$ | **ocupación** por celda |
| **VLAD** | residuos acumulados por celda | $k \times d$ | **desplazamiento** dentro de cada celda |

El salto conceptual está en las dos últimas filas, y se entiende mejor por contraste. *Bag of features* aprende $k$ prototipos y cuenta cuántos descriptores cayeron cerca de cada uno: dice **qué estructuras aparecen**. VLAD conserva los mismos prototipos pero, en vez de contar, acumula las diferencias: dice **cómo se desvían** los descriptores del prototipo que les tocó.

---

## 2. El mecanismo clásico

Los cinco pasos, tal como los enumera la [Clase 41](/clases/clase-41):

1. **Proyectar** el conjunto de entrenamiento en el espacio de features.
2. **Aprender $k$ centroides** $c_1, \dots, c_k$, típicamente con k-means.
3. **Asignar** cada descriptor a su centroide más cercano:
   $$c(x) = \arg\min_{c_i} \; \lVert c_i - x \rVert^2$$
4. **Calcular el residuo** $x - c_i$.
5. **Agregar los residuos** por centroide.

El resultado es una matriz de $k \times d$ que se aplana y se normaliza en L2:

$$v(j,k) = \sum_{i=1}^{N} a_k(x_i)\,\big(x_i(j) - c_k(j)\big)$$

donde $a_k(x_i) \in \{0,1\}$ indica si $x_i$ fue asignado al centroide $k$. Los índices $j = 1 \dots d$ recorren las componentes del descriptor y $k = 1 \dots K$ los centroides.

{{< concept-alert type="clave" >}}
**Un vector VLAD nulo no significa "no hay nada": significa que los descriptores están centrados en los prototipos.** El vector codifica desviaciones, no presencia. Es la razón por la que VLAD y *bag of features* capturan información complementaria — y por la que en la práctica se suele usar un diccionario chico: con $k$ grande cada celda tiene pocos descriptores y los residuos se vuelven ruidosos.
{{< /concept-alert >}}

---

## 3. Por qué el promedio no alcanza: un ejemplo mínimo

El argumento de que "VLAD conserva más que el promedio" se puede volver concreto con un caso construido.

Sean dos fuentes de descriptores en $\mathbb{R}^2$, con un diccionario de dos prototipos en $c_1 = (-2, 0)$ y $c_2 = (2, 0)$:

- **Fuente A**: la mitad de sus descriptores cerca de $c_1 + (0, 0{,}6)$, la otra mitad cerca de $c_2 - (0, 0{,}6)$.
- **Fuente B**: exactamente al revés — $c_1 - (0, 0{,}6)$ y $c_2 + (0, 0{,}6)$.

Por construcción **ambas tienen la misma media global**: $(0,0)$. Con 400 descriptores por enunciado y dos enunciados por fuente:

```
media global (mean pooling) — idéntica por construcción:
   A1: [0. 0.]      A2: [-0. -0.]
   B1: [-0. -0.]    B2: [-0. -0.]

método            mismo hablante   distinto    margen  dim
mean pooling              0.0000    -0.0000    0.0000    2
VLAD (hard)               0.9999    -0.9999    1.9998    4
NetVLAD (soft)            0.9999    -0.9999    1.9998    4
```

El promedio colapsa las cuatro muestras al vector nulo: **no puede distinguir nada**, porque la información no está en el centro de masa sino en cómo se distribuye la masa alrededor de los prototipos. VLAD las separa perfectamente, y los vectores resultantes son opuestos:

```
A: [-0.005,  0.707,  0.005, -0.707]
B: [-0.001, -0.707,  0.001,  0.707]
```

Las componentes 1 y 3 (eje $x$) son ~0 en ambos: en esa dirección los descriptores sí están centrados en sus prototipos. Toda la información discriminativa vive en las componentes 2 y 4 (eje $y$), con signos invertidos.

El caso es artificial —en datos reales las medias raramente coinciden con esa exactitud— pero aísla el mecanismo: **VLAD codifica estructura interna que el promedio destruye por construcción**. Es la versión geométrica del argumento de [Xie et al. (2019)](/papers/utterance-level-xie-2019), donde cambiar promedio por VLAD, con todo lo demás igual, lleva el EER de 10,48 % a 3,57 %.

---

## 4. El problema de la derivabilidad

Los pasos 2 y 3 dejan a VLAD fuera del aprendizaje end-to-end:

- Los centroides salen de **k-means**, cuyo criterio es de reconstrucción, no de discriminación. Nada garantiza que las celdas separen lo que interesa separar.
- La asignación usa **`argmin`**, que es constante a trozos: su derivada es cero en casi todas partes y no existe en las fronteras. **No hay gradiente que propagar.**

Mientras eso siga así, VLAD es un postprocesamiento: se calculan descriptores con una red, se agregan con VLAD, y las dos etapas no se hablan.

---

## 5. NetVLAD: reemplazar argmin por softmax

[Arandjelović et al. (2016)](/papers/netvlad-arandjelovic-2016) resuelven el bloqueo cambiando la pertenencia dura por una **blanda**:

$$\bar{a}_k(x_i) = \frac{e^{\,w_k^\top x_i + b_k}}{\sum_{k'} e^{\,w_{k'}^\top x_i + b_{k'}}}$$

y la agregación queda

$$V(j,k) = \sum_{i=1}^{N} \bar{a}_k(x_i)\,\big(x_i(j) - c_k(j)\big)$$

Tres consecuencias:

1. **Es derivable en todas partes**: el gradiente llega a los descriptores, y por lo tanto al backbone.
2. **Cada descriptor contribuye a todos los clusters**, con peso decreciente. No hay saltos en las fronteras.
3. **$\{w_k\}$, $\{b_k\}$ y $\{c_k\}$ son parámetros independientes y entrenables.** Este desacople es más sutil de lo que parece: en VLAD clásico el criterio de asignación y el centro del residuo son el mismo objeto; acá pueden divergir, y la capa aprende un criterio de asignación distinto de la posición del prototipo.

**NetVLAD generaliza VLAD.** El softmax con temperatura $\tau$ converge al `argmin` cuando $\tau \to 0$, y se puede comprobar numéricamente:

```
tau=5.0    cos(NetVLAD_tau, VLAD_hard) = 0.960035
tau=1.0    cos(NetVLAD_tau, VLAD_hard) = 1.000000
tau=0.3    cos(NetVLAD_tau, VLAD_hard) = 1.000000
```

(La convergencia es rápida acá porque los dos prototipos están muy separados; con un diccionario denso hace falta $\tau$ más chico.)

---

## 6. GhostVLAD: clusters que se tiran

Una limitación de la asignación blanda es que **todo descriptor contribuye a algo**. El ruido, los silencios y las voces de fondo no se descartan: se reparten entre los clusters reales.

**GhostVLAD** agrega $G$ clusters adicionales —los *fantasma*— que participan del softmax pero cuyos residuos **se descartan** de la concatenación final. El efecto es que los descriptores irrelevantes pueden depositar su peso ahí, atenuando su contribución a los clusters que sí se usan.

Es una forma elegante de aprender a descartar: la red no necesita un módulo que decida qué ignorar ni una etiqueta de "esto es ruido" — le basta con tener un sumidero disponible y una pérdida que premie no usar el ruido. En [Xie et al. (2019)](/papers/utterance-level-xie-2019) se usan 8 clusters reales y 2 fantasma, y el aporte sobre NetVLAD es modesto pero consistente (3,22 % contra 3,57 % de EER).

---

## 7. Dónde se usa

La estructura del problema —**un conjunto de descriptores locales de cardinalidad variable que hay que resumir en un vector fijo**— aparece en dominios que no se hablan entre sí, y la técnica viajó por todos:

| Dominio | Descriptores locales | Trabajo |
|---|---|---|
| Búsqueda de imágenes | SIFT sobre puntos de interés | [Jégou et al. (2010)](/papers/vlad-jegou-2010) |
| Reconocimiento de lugares | posiciones de un mapa de activaciones CNN | [NetVLAD (2016)](/papers/netvlad-arandjelovic-2016) |
| Reconocimiento de hablante | frames de un espectrograma | [Xie et al. (2019)](/papers/utterance-level-xie-2019) |
| Reconocimiento de acciones | features por cuadro | ActionVLAD |

En todos, el diagnóstico previo es el mismo: se estaba usando promedio o máximo, heredados de la clasificación de imágenes, sin preguntarse si eran adecuados para agregar conjuntos.

---

## 8. Costos y consideraciones prácticas

- **La dimensión de salida es $K \times d$**, que crece rápido: 64 clusters sobre features de 512 dimensiones dan 32 768 valores. Una capa lineal de reducción posterior es prácticamente obligatoria (Xie et al. reducen a 512).
- **Normalización.** La práctica estándar es normalizar en L2 **dentro de cada cluster** y después globalmente, lo que evita que un cluster muy poblado domine el vector.
- **Inicialización.** Arrancar los centroides con k-means sobre descriptores de un backbone preentrenado funciona mucho mejor que inicializar al azar.
- **Robustez al número de clusters.** En el caso del habla, el barrido entre 8 y 14 clusters mueve el EER entre 3,22 % y 3,37 %: el método no es sensible a ese hiperparámetro.
- **Memoria en entrenamiento.** Hay que materializar los residuos de todos los descriptores contra todos los clusters, un tensor de $N \times K \times d$. Es evitable: como $c_k$ no depende de $i$, la suma se distribuye en $\sum_i a_{ik}x_i - (\sum_i a_{ik})c_k$ — un producto matricial más un producto exterior, sin tensor de tres ejes. Medido en el [Lab 41](/laboratorios/lab-41): idéntico numéricamente y hasta **32× más rápido** con 4.000 descriptores, ahorrando un tensor de 78 MB.

## Qué pasa con los centroides en un modelo ya entrenado

La teoría describe los centroides como un vocabulario que particiona el espacio. Al abrir el checkpoint entrenado del [Lab 41](/laboratorios/lab-41) —Thin ResNet + GhostVLAD sobre VoxCeleb2— aparece algo distinto:

- **Los 8 centroides tienen coseno 0,9983 entre sí.** Normas casi idénticas (14,03–14,11) y distancia media entre ellos de 0,82. No hay 8 regiones de Voronoi: hay un punto y ocho perturbaciones diminutas. Como $v_k = \sum_i a_{ik}x_i - (\sum_i a_{ik})c_k$, con todos los $c_k$ iguales lo único que distingue un cluster de otro es **la distribución de asignación**. El *trainable discriminative clustering* degeneró en **attention pooling de 8 cabezas** con un sesgo común.
- **Los 2 centroides fantasma conservan su inicialización** (norma exactamente 1,000, ortogonales entre sí) y perturbarlos ×1000 no mueve la salida. Su gradiente es cero porque sus residuos se descartan antes de la pérdida. El paper de [GhostVLAD](/papers/ghostvlad-zhong-2018) ya lo había especificado: *"$\{a_k\}$ y $\{b_k\}$ tienen K+G elementos, mientras que $\{c_k\}$ sigue teniendo K"*. Los fantasmas viven solo en la asignación.
- **Lo que sí se entrena de los fantasmas son sus compuertas.** Sesgo **positivo** (+0,55 y +0,47) contra el sesgo negativo de los 8 reales, y $\|w_k\|$ 3–5× mayor: ganan el softmax por defecto y absorben la masa que de otro modo contaminaría los clusters reales.

La lección transversal: **la intra-normalización por cluster** —que es lo que vuelve al descriptor invariante al número de descriptores agregados, verificado con factor de escala exactamente 2,000000 al duplicar el conjunto— hace que buena parte de la geometría del vocabulario deje de importar. Lo que discrimina es cómo se reparte la atención, no dónde están los prototipos.

---

## Referencias

- Fundamentos relacionados: [Reconocimiento de hablante](/fundamentos/reconocimiento-de-hablante) · [Metric learning](/fundamentos/metric-learning) · [Bag of words](/fundamentos/bag-of-words) (el análogo textual del *bag of features*) · [Representación de datos](/fundamentos/representacion-datos).
- Papers: [VLAD (2010)](/papers/vlad-jegou-2010) · [NetVLAD (2016)](/papers/netvlad-arandjelovic-2016) · [GhostVLAD (2018)](/papers/ghostvlad-zhong-2018) · [Utterance-level Aggregation (2019)](/papers/utterance-level-xie-2019) · [x-vectors (2018)](/papers/x-vectors-snyder-2018).
- Clases: [Clase 41](/clases/clase-41) y su [práctica](/clases/clase-41/practica/02-agregacion-vlad), donde el ejemplo de la sección 3 se implementa y se extiende en triple framework.
- Laboratorios: [Lab 41](/laboratorios/lab-41), donde la capa se desarma línea por línea y los centroides entrenados se abren y se miden.
- Dominio: [Audio](/dominios/audio).
