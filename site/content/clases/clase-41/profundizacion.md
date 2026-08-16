---
title: "Profundización - Alineación, agregación y lo que cada una destruye"
weight: 20
math: true
---

> La [teoría](teoria) presentó dos problemas y dos soluciones: el habla desalineada se resuelve con un token *blank*, y los frames de un enunciado se resumen con residuos respecto de un diccionario. Esta página deriva ambos mecanismos, verifica numéricamente lo que se puede verificar, y muestra que **cada uno destruye deliberadamente la información que la otra tarea necesita**. Cinco partes: la suma sobre alineaciones de CTC, su costo oculto, la aritmética del pooling temporal, el gradiente que bloquea el `argmin`, y la geometría de por qué el promedio pierde.

---

## Parte I — CTC: sumar sobre un número exponencial de alineaciones

### I.1. La función de colapso

Sea $\mathcal{A} = \{a, b, \dots\} \cup \{\varnothing\}$ el alfabeto de salida con el símbolo *blank*. La función de colapso $\mathcal{B}$ toma una secuencia de largo $T$ y produce una de largo $\leq T$ mediante dos operaciones, **en este orden**:

1. colapsar repeticiones consecutivas;
2. eliminar los blanks.

$$\mathcal{B}(\texttt{c-a-a-}\varnothing\texttt{-s-s-a}) = \texttt{casa}$$

El orden importa y es lo que hace posible representar letras dobles: para escribir `casa` con dos `a` seguidas hay que separarlas con un blank, porque `aa` colapsa a `a`. Esa es la razón de existir del blank más allá de la "pausa" que menciona la clase — sin él, ninguna transcripción con letras repetidas sería representable.

### I.2. La probabilidad marginal

CTC no elige una alineación: las suma todas.

$$P(y \mid x) \;=\; \sum_{\pi \,\in\, \mathcal{B}^{-1}(y)} \; \prod_{t=1}^{T} P(\pi_t \mid x)$$

La preimagen $\mathcal{B}^{-1}(y)$ es enorme. Para una transcripción sin símbolos repetidos consecutivos, su tamaño resulta ser exactamente

$$\big|\mathcal{B}^{-1}(y)\big| \;=\; \binom{T+U}{2U}$$

con $U = |y|$. Verificado por enumeración exhaustiva para la palabra `casa` ($U=4$):

| $T$ | 6 | 8 | 10 | 12 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| alineaciones | 45 | 495 | 3 003 | 12 870 | 735 471 | $1{,}04 \times 10^9$ | $2{,}58 \times 10^{11}$ |

Un segundo de audio son unos 100 frames. **Doscientos cincuenta mil millones de alineaciones para transcribir cuatro letras.**

Cuando $y$ **sí** tiene símbolos repetidos, el conteo baja: la repetición obliga a un blank intermedio, lo que elimina alineaciones. Medido: `[1,1]` con $T=7$ admite **70** alineaciones, mientras que la fórmula sin repeticiones daría $\binom{9}{4} = 126$.

### I.3. La programación dinámica

La suma se calcula en $O(T \cdot U)$ construyendo la **secuencia extendida** $l$, que intercala blanks alrededor de cada símbolo:

$$l = (\varnothing,\, y_1,\, \varnothing,\, y_2,\, \varnothing,\, \dots,\, y_U,\, \varnothing), \qquad |l| = 2U+1$$

y definiendo $\alpha_t(s)$ como la probabilidad total de todos los prefijos de largo $t$ que terminan en la posición $s$ de $l$:

$$\alpha_t(s) = \Big[\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \underbrace{\alpha_{t-1}(s-2)}_{\text{solo si } l_s \neq \varnothing \;\wedge\; l_s \neq l_{s-2}}\Big] \cdot P(l_s \mid x_t)$$

Los tres términos son las tres formas de llegar a la posición $s$ en el paso $t$: quedarse, avanzar uno, o **saltarse un blank**. Ese tercer término está condicionado, y la condición $l_s \neq l_{s-2}$ es exactamente la que impide saltarse el blank obligatorio entre dos símbolos iguales.

La probabilidad total es $\alpha_T(2U+1) + \alpha_T(2U)$: la secuencia puede terminar en el último símbolo o en el blank final.

{{< concept-alert type="clave" >}}
**Verificación numérica.** Implementando el DP y la enumeración exhaustiva por separado, y comparándolos sobre casos con $T$ chico:

```
  T          y   #alin    fuerza bruta     forward DP      |dif|
  4        [1]      10     0.253750768    0.253750768   0.00e+00
  5     [1, 2]      35     0.188044421    0.188044421   5.55e-17
  6     [1, 2]      70     0.124306657    0.124306657   2.78e-17
  7     [1, 1]      70     0.044837265    0.044837265   2.08e-17
  8  [1, 2, 1]     462     0.067375506    0.067375506   5.55e-17
```

Coinciden hasta el épsilon de máquina, incluido el caso con símbolo repetido. Una recursión de tres términos calcula en $O(TU)$ lo que la definición pide sumar sobre $\binom{T+U}{2U}$ términos.
{{< /concept-alert >}}

---

## Parte II — Lo que CTC destruye

La factorización de CTC esconde un supuesto que define sus límites:

$$P(\pi \mid x) = \prod_{t=1}^{T} P(\pi_t \mid x)$$

Cada símbolo depende **solo de $x$**, nunca de los símbolos vecinos. Es **independencia condicional** entre salidas dado el audio, y tiene una consecuencia concreta: el modelo no aprende lenguaje. Nada en la función de pérdida penaliza que `qasa` sea ortográficamente imposible en español, siempre que la acústica lo sugiera.

Compárese con el decodificador autorregresivo de un modelo con atención:

$$P(y \mid x) = \prod_{i=1}^{U} P(y_i \mid x,\, y_{<i})$$

Acá cada salida ve todas las anteriores. El modelo aprende, dentro de la misma red, que después de `q` viene `u` — y por eso [LAS](/papers/las-chan-2016) puede prescindir de un modelo de lenguaje externo mientras que los sistemas CTC lo necesitan.

| | CTC | Autorregresivo con atención |
|---|---|---|
| Factorización | $\prod_t P(\pi_t \mid x)$ | $\prod_i P(y_i \mid x, y_{<i})$ |
| Dependencia entre salidas | ninguna | total |
| Modelo de lenguaje | **externo, necesario** | interno |
| Inferencia | paralela sobre $t$ | secuencial sobre $i$ |
| Monotonía | garantizada | no garantizada |
| Streaming | sí | no |

Ninguna domina a la otra, y por eso los sistemas modernos usan pérdidas híbridas: el término CTC impone monotonía y acelera la convergencia; la atención aporta el modelado del lenguaje.

---

## Parte III — La aritmética del pooling temporal

La clase presenta el *pooling over time* junto a CTC, como si fueran alternativas. Son cosas distintas, y la cuenta lo muestra.

**El costo de la atención.** Para emitir $U$ símbolos consultando $T$ posiciones, el mecanismo evalúa el puntaje

$$e_{i,t} = \text{score}(s_{i-1}, h_t)$$

para cada par $(i,t)$: **$T \times U$ evaluaciones**. Con 10 segundos de audio a 100 frames/s y 100 caracteres de transcripción, son $10^5$ evaluaciones del puntaje por enunciado, cada una con su producto matricial.

**Qué hace el pooling.** Reducir la resolución temporal del encoder por un factor $r$ deja $T/r$ posiciones:

$$\text{costo} \;=\; \frac{T}{r} \times U$$

Con la pirámide de [LAS](/papers/las-chan-2016) —factor 2 por capa, tres capas— $r = 8$ y el costo baja un orden de magnitud.

**Por qué se puede.** A 100 frames por segundo con ventanas de 25 ms, dos frames consecutivos **comparten 15 ms de señal**: su solape es del 60 %. La secuencia está masivamente sobremuestreada respecto de la tasa a la que cambia el tracto vocal, así que descartar la mitad pierde poca información.

$$\text{solape} = \frac{25 - 10}{25} = 60\,\%$$

**Y el límite.** El factor no puede crecer indefinidamente: hay que conservar al menos un vector por unidad de salida, así que $T/r \geq U$. Con habla a ~13 fonemas por segundo y frames cada 10 ms, $r = 8$ deja unos 12 vectores por segundo — al borde. Es la razón por la que los sistemas usan factores de 4 a 8 y no 32.

{{< concept-alert type="nota" >}}
CTC y el pooling temporal atacan problemas distintos: CTC resuelve **la alineación**, el pooling resuelve **el costo de la atención**. La prueba de que son ortogonales es que se usan juntos — los sistemas actuales aplican subsampling convolucional en el encoder *y* una pérdida CTC auxiliar *y* atención en el decodificador.
{{< /concept-alert >}}

---

## Parte IV — El gradiente que bloquea el argmin

En VLAD clásico, la pertenencia de un descriptor a un centroide es

$$a_k(x) = \mathbb{1}\Big[k = \arg\min_j \lVert x - c_j \rVert^2\Big]$$

Una función indicadora. Su comportamiento como objeto derivable:

$$\frac{\partial a_k(x)}{\partial x} = 0 \quad \text{en casi todas partes}, \qquad \text{no definida en las fronteras}$$

Las fronteras entre celdas de Voronoi son un conjunto de medida nula, así que el gradiente es **cero en casi todo el espacio** y **no existe** justo donde la función cambia. En cualquiera de los dos casos, no hay señal que propagar: ni los centroides ni el extractor de features pueden aprenderse para la tarea.

### La relajación

[NetVLAD](/papers/netvlad-arandjelovic-2016) reemplaza el indicador por un softmax:

$$\bar{a}_k(x) = \frac{e^{\,w_k^\top x + b_k}}{\sum_{k'} e^{\,w_{k'}^\top x + b_{k'}}}$$

Con $w_k = 2c_k/\tau$ y $b_k = -\lVert c_k \rVert^2/\tau$ se recupera exactamente el softmax sobre distancias negativas escaladas por $\tau$, y en el límite

$$\lim_{\tau \to 0} \bar{a}_k(x) = a_k(x)$$

**NetVLAD generaliza VLAD**: el caso duro es el límite de temperatura cero. Verificado numéricamente sobre un diccionario de dos centroides:

```
tau=5.0    cos(NetVLAD_tau, VLAD_hard) = 0.960035
tau=1.0    cos(NetVLAD_tau, VLAD_hard) = 1.000000
tau=0.3    cos(NetVLAD_tau, VLAD_hard) = 1.000000
```

Y el paso adicional de NetVLAD, que es lo que la hace más que una relajación: **desacoplar los parámetros**. En VLAD, el criterio de asignación y el centro del residuo son el mismo $c_k$. En NetVLAD, $\{w_k, b_k\}$ y $\{c_k\}$ son independientes, así que la capa puede aprender a asignar según un criterio distinto de la posición del prototipo — un grado de libertad que el VLAD clásico no tiene.

---

## Parte V — Por qué el promedio pierde

### V.1. El promedio es un VLAD degenerado

Con un solo centroide en el origen ($K=1$, $c_1 = 0$), la fórmula de VLAD se reduce a

$$v = \sum_{i=1}^{N} (x_i - 0) = N \cdot \bar{x}$$

que tras la normalización L2 **es exactamente el promedio normalizado**. Verificado:

```
VLAD con K=1 y c=0 (normalizado) == media normalizada: True
```

Es decir: *average pooling* no es una alternativa a VLAD sino su **caso más pobre** — un diccionario de un solo elemento colocado en el origen. Toda la capacidad adicional viene de tener varios prototipos y de que estén donde los datos están.

### V.2. La información que el promedio destruye

Un promedio es una proyección de $\mathbb{R}^{N \times d}$ a $\mathbb{R}^d$: descarta todo salvo el primer momento. Dos conjuntos con la misma media son **indistinguibles** para él, por definición.

Construyamos ese caso. Diccionario de dos prototipos en $c_1 = (-2,0)$ y $c_2 = (2,0)$, y dos fuentes:

- **A**: mitad de sus descriptores en $c_1 + (0, 0{,}6)$, mitad en $c_2 - (0, 0{,}6)$
- **B**: mitad en $c_1 - (0, 0{,}6)$, mitad en $c_2 + (0, 0{,}6)$

Por construcción ambas tienen media global exactamente $(0,0)$. Con 400 descriptores por muestra:

```
media global (mean pooling) — idéntica por construcción:
   A1: [0. 0.]      A2: [-0. -0.]
   B1: [-0. -0.]    B2: [-0. -0.]

método            mismo hablante   distinto    margen  dim
mean pooling              0.0000    -0.0000    0.0000    2
VLAD (hard)               0.9999    -0.9999    1.9998    4
NetVLAD (soft)            0.9999    -0.9999    1.9998    4
```

El promedio colapsa las cuatro muestras al vector nulo y **no puede distinguir nada**. VLAD las separa con margen máximo, y sus vectores son opuestos:

$$v_A = (-0{,}005,\; \mathbf{0{,}707},\; 0{,}005,\; \mathbf{-0{,}707}), \qquad v_B = (-0{,}001,\; \mathbf{-0{,}707},\; 0{,}001,\; \mathbf{0{,}707})$$

Las componentes del eje $x$ son ~0 en ambos —en esa dirección los descriptores sí están centrados en sus prototipos— y toda la información discriminativa vive en las del eje $y$, con signo invertido.

### V.3. Qué tiene que ver con hablantes

El caso es construido, pero el mecanismo es el que opera en el problema real. La explicación de [Xie et al. (2019)](/papers/utterance-level-xie-2019) sobre por qué el promedio temporal falla:

> *"the features from TAP are typically good at optimizing the inter-class difference (i.e., separating different speakers), but not good at reducing the intra-class variation (i.e. making features of the same speaker compact)."*

Traducido a la geometría de arriba: promediar sobre un enunciado con ruido, silencios y voces ajenas produce un centro de masa que **se mueve mucho** entre dos grabaciones de la misma persona, porque depende de qué proporción de basura tocó en cada una. Los residuos respecto de prototipos aprendidos son más estables: la basura se acumula en sus propias celdas —o, con GhostVLAD, en clusters que se descartan— y las celdas que codifican voz se ven menos afectadas.

Es la diferencia entre 10,48 % y 3,57 % de EER con el mismo backbone.

---

## Las dos mitades, en una línea

Cada tarea define su representación por **lo que decide destruir**:

$$\underbrace{\text{ASR}: \; x_{1:T} \mapsto y_{1:U}}_{\text{conserva el tiempo, descarta al hablante}} \qquad\qquad \underbrace{\text{Speaker}: \; x_{1:T} \mapsto v \in \mathbb{R}^{512}}_{\text{descarta el tiempo, conserva al hablante}}$$

CTC marginaliza sobre alineaciones para no tener que saber **cuándo** ocurrió cada fonema, pero conserva el orden en la salida. VLAD acumula residuos precisamente para **olvidar cuándo** ocurrió cada frame: su salida es invariante a permutaciones de la entrada — dos enunciados con los mismos frames en distinto orden dan el mismo descriptor. Esa invarianza, que sería fatal en ASR, es exactamente lo que se busca en reconocimiento de hablante.

---

## Ver también

- [Teoría](teoria) — el recorrido de las 88 diapositivas.
- [Práctica desde 0](practica) — el código que produce todas las verificaciones de esta página, en triple framework.
- [Fundamento: Reconocimiento de voz](/fundamentos/reconocimiento-de-voz) · [Reconocimiento de hablante](/fundamentos/reconocimiento-de-hablante) · [Agregación VLAD](/fundamentos/agregacion-vlad) · [CTC Loss](/fundamentos/ctc-loss).
