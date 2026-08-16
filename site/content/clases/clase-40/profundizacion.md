---
title: "Profundización - La matemática del desplazamiento temporal"
weight: 20
math: true
---

> La [teoría](teoria) presentó el desplazamiento temporal como "la convolución son dos pasos, hacemos el barato y el otro lo absorbe la red". Esta página verifica esa afirmación, y en el camino encuentra que **no es exactamente cierta**: lo que TSM implementa no es una convolución temporal cualquiera sino una **estructuralmente restringida**, y esa restricción es justamente lo que la hace gratis. Cinco partes: la equivalencia formal y su letra chica, la aritmética del fold, el alcance temporal efectivo de una pila de módulos, la invarianza al orden que TSM viene a romper, y el costo real medido en bytes en vez de FLOPs.

---

## Parte I — Qué implementa exactamente el desplazamiento

### I.1. La convolución temporal general

Una convolución temporal de kernel 3 aplicada a una secuencia de mapas de características $x_t \in \mathbb{R}^{C_{\text{in}} \times H \times W}$, sin mezclar posiciones espaciales (kernel $3\times1\times1$), es

$$y_t \;=\; W^{(-1)} x_{t-1} \;+\; W^{(0)} x_{t} \;+\; W^{(+1)} x_{t+1}$$

con tres matrices $W^{(\tau)} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$. El costo es de $3 \, C_{\text{in}} C_{\text{out}}$ parámetros y $3 \, C_{\text{in}} C_{\text{out}} HW$ multiplicaciones por frame.

### I.2. Lo que produce TSM seguido de una convolución $1\times1$

Definamos una partición de los canales en tres bloques disjuntos, con $F = \lfloor C_{\text{in}} / d \rfloor$ y $d$ el `fold_div`:

$$\mathcal{A} = [0, F), \qquad \mathcal{B} = [F, 2F), \qquad \mathcal{C} = [2F, C_{\text{in}})$$

El desplazamiento produce un tensor $\tilde{x}_t$ cuyo canal $c$ vale

$$\tilde{x}_t[c] \;=\; \begin{cases} x_{t+1}[c] & c \in \mathcal{A} \\[2pt] x_{t-1}[c] & c \in \mathcal{B} \\[2pt] x_{t}[c] & c \in \mathcal{C} \end{cases}$$

Al aplicarle una convolución $1\times1$ con matriz $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}}}$:

$$y_t \;=\; W \tilde{x}_t \;=\; \sum_{c \in \mathcal{A}} W_{:,c}\, x_{t+1}[c] \;+\; \sum_{c \in \mathcal{B}} W_{:,c}\, x_{t-1}[c] \;+\; \sum_{c \in \mathcal{C}} W_{:,c}\, x_{t}[c]$$

que se reescribe exactamente en la forma de la convolución temporal general:

$$y_t \;=\; W^{(-1)} x_{t-1} + W^{(0)} x_{t} + W^{(+1)} x_{t+1}, \qquad \text{con} \quad W^{(\tau)} = W \, P_\tau$$

donde $P_{+1}, P_{-1}, P_{0}$ son las proyecciones diagonales sobre $\mathcal{A}, \mathcal{B}, \mathcal{C}$ respectivamente.

### I.3. La letra chica: los soportes son disjuntos

La equivalencia es cierta, pero con una restricción que conviene enunciar porque explica todo lo demás:

$$P_{+1} + P_{-1} + P_{0} = I \qquad \text{y} \qquad P_\tau P_{\tau'} = 0 \;\; \text{para } \tau \neq \tau'$$

Es decir: **las tres matrices $W^{(\tau)}$ tienen columnas de soporte disjunto**. Cada canal de entrada contribuye a **un solo instante temporal**. Una convolución $3\times1\times1$ general permite que el canal $c$ aporte simultáneamente desde $t-1$, $t$ y $t+1$ con tres pesos independientes; TSM le asigna un instante y sanseacabó.

Esto se traduce directo al conteo:

| | Parámetros | Multiplicaciones/frame |
|---|---|---|
| Convolución $3\times1\times1$ general | $3\,C_{\text{in}}C_{\text{out}}$ | $3\,C_{\text{in}}C_{\text{out}}HW$ |
| TSM + la $1\times1$ que ya existía | $C_{\text{in}}C_{\text{out}}$ | $C_{\text{in}}C_{\text{out}}HW$ |

{{< concept-alert type="clave" >}}
**El desplazamiento no hace gratis a la convolución temporal: la reemplaza por una versión con un tercio de los grados de libertad, que resulta ser exactamente el presupuesto que la red ya tenía.** La frase "cero parámetros y cero FLOPs" es literalmente cierta —el módulo no agrega nada— pero lo que se obtiene no es la convolución 3D que se evitó, sino una restricción estructurada de ella. Que esa restricción baste es un hallazgo empírico, no una consecuencia matemática.
{{< /concept-alert >}}

Esta lectura explica el experimento de control `I3D_replace` del paper: reemplazar cada TSM por una convolución $3\times1\times1$ genuina —tres veces más parámetros, capacidad estrictamente mayor— resulta **más lento y menos preciso**. Más lento es esperable; menos preciso indica que los grados de libertad extra no se aprovechan y sí encarecen la optimización.

### I.4. La proporción reaparece como una decisión de asignación

Con la partición explícita, el hiperparámetro $d$ deja de ser "cuánto se desplaza" y pasa a ser **cómo se reparte un presupuesto fijo de $C_{\text{in}}$ canales entre tres tareas**:

$$\underbrace{F}_{\text{ver el futuro}} \;+\; \underbrace{F}_{\text{ver el pasado}} \;+\; \underbrace{C_{\text{in}} - 2F}_{\text{ver el presente}} \;=\; C_{\text{in}}$$

Ahí está la tensión que el paper resuelve empíricamente en $2F/C_{\text{in}} = 1/4$: cada canal asignado al pasado o al futuro es un canal **restado** a la representación espacial del instante actual. No hay canales nuevos; hay una reasignación. Con $d = 2$ el bloque $\mathcal{C}$ se vacía y **ningún canal ve el presente** — que es exactamente el régimen donde el [Laboratorio 40](/laboratorios/lab-40/04-la-curva-de-proporcion) mide un colapso a 0,52 % de confianza.

---

## Parte II — Aritmética del fold en una ResNet-50

El módulo envuelve la `conv1` de cada bloque bottleneck, de modo que $C$ es el número de canales de **entrada** a esa convolución. Para la ResNet-50 del laboratorio:

| Etapa | Bloques | $C_{\text{in}}$ de `conv1` | $F = C/8$ | Desplazado |
|---|---|---|---|---|
| `layer1[0]` | 1 | 64 | 8 | 16 / 64 |
| `layer1[1:3]` | 2 | 256 | 32 | 64 / 256 |
| `layer2[0]` | 1 | 256 | 32 | 64 / 256 |
| `layer2[1:4]` | 3 | 512 | 64 | 128 / 512 |
| `layer3[0]` | 1 | 512 | 64 | 128 / 512 |
| `layer3[1:6]` | 5 | 1024 | 128 | 256 / 1024 |
| `layer4[0]` | 1 | 1024 | 128 | 256 / 1024 |
| `layer4[1:3]` | 2 | 2048 | 256 | 512 / 2048 |

Todos los anchos son divisibles por 8, así que la proporción es exactamente 1/4 en los 16 módulos. En arquitecturas con anchos no divisibles —MobileNet, por ejemplo— el piso de la división deja un residuo que simplemente no se desplaza; el sesgo es hacia menos desplazamiento, nunca hacia más.

**El caso degenerado que habilita la ablación.** Si $d > C_{\text{in}}$ entonces $F = 0$, los bloques $\mathcal{A}$ y $\mathcal{B}$ quedan vacíos y $\mathcal{C} = [0, C_{\text{in}})$: el módulo es la identidad. Como el mínimo $C_{\text{in}}$ de la tabla es 64, basta $d > 2048$ para anular los 16 módulos de una vez. Esa es la manipulación que el laboratorio usa para medir la contribución del desplazamiento sin reentrenar.

**La densidad de inserción.** La implementación de referencia reduce la densidad cuando `layer3` tiene $\geq 23$ bloques, insertando el módulo en uno de cada dos. Aplicado a las arquitecturas usuales:

| Backbone | Bloques en `layer3` | `n_round` | Módulos insertados |
|---|---|---|---|
| ResNet-50 | 6 | 1 | 16 de 16 |
| ResNet-101 | 23 | 2 | 17 de 33 |

---

## Parte III — El alcance temporal efectivo

La clase afirma que "para cada TSM insertado, el campo receptivo temporal se ampliará en 2". Es correcto como **cota**: con $L$ módulos en serie y sin pooling temporal, la información puede recorrer hasta $L$ frames en cada dirección, de modo que el campo receptivo teórico es

$$R_{\text{teórico}} = 2L + 1 \;=\; 33 \text{ frames para } L = 16$$

sobre una entrada de 8 segmentos. Es decir, saturación: el campo receptivo teórico cubre el clip varias veces.

Pero "puede recorrer" no es "recorre". Vale preguntarse **cuánta** información efectivamente viaja esa distancia.

### Un modelo idealizado

Considérese un canal individual y sígase su contenido a través de la pila. En cada módulo, ese canal cae en $\mathcal{A}$, $\mathcal{B}$ o $\mathcal{C}$, de modo que la información que transporta se corre $+1$, $-1$ o $0$ frames. Si se modela la asignación como aleatoria e independiente en cada módulo —lo que **no** es literalmente cierto, ver la advertencia abajo— con probabilidades $p_{+1} = p_{-1} = 1/8$ y $p_0 = 3/4$, el desplazamiento acumulado tras $L$ módulos es una suma de $L$ variables i.i.d. con

$$\mathbb{E}[\Delta_1] = 0, \qquad \operatorname{Var}[\Delta_1] = p_{+1}(+1)^2 + p_{-1}(-1)^2 = \tfrac{1}{4}$$

y por lo tanto

$$\operatorname{Var}[\Delta_L] = \frac{L}{4}, \qquad \sigma_L = \frac{\sqrt{L}}{2}$$

Para $L = 16$: $\sigma_{16} = 2$ frames.

{{< concept-alert type="nota" >}}
**El alcance efectivo crece como $\sqrt{L}$, no como $L$.** El campo receptivo teórico es de $\pm 16$ frames, pero la masa de información se concentra en $\pm 2$. Duplicar la profundidad de la red multiplica el alcance efectivo por $1{,}41$, no por 2. Es la diferencia entre "hasta dónde puede llegar una señal" y "dónde está la señal".
{{< /concept-alert >}}

**Por qué el modelo es idealizado.** Entre módulo y módulo hay convoluciones que mezclan canales, así que la información no viaja por un canal fijo: se redistribuye. La caminata aleatoria es una heurística sobre el transporte, no una descripción del cómputo. Sirve para el orden de magnitud y para la dependencia en $\sqrt{L}$; no para predecir valores concretos.

Aun así, la conclusión cualitativa concuerda con lo que se observa: TSM es fuerte en **movimiento local** —lo que ocurre entre segmentos vecinos— y no construye dependencias de largo alcance sobre la línea de tiempo completa. Para eso la cobertura la aporta el otro mecanismo, el muestreo por segmentos, que estira el paso entre frames hasta abarcar todo el video.

### Los bordes

El relleno con ceros implica que en el frame $t = 0$ el bloque $\mathcal{B}$ (pasado) llega nulo, y en $t = T-1$ el bloque $\mathcal{A}$ (futuro) también. Con $T = 8$ eso afecta a $2/8 = 25\,\%$ de los segmentos, y ocurre en cada uno de los 16 módulos. La fracción de la entrada anulada en un frame de borde es $F/C = 1/8$ por módulo.

En clips cortos el efecto es proporcionalmente mayor: con $T = 4$, la mitad de los segmentos son borde. Es un argumento —no discutido en el paper— a favor de usar $T$ grande cuando el clip lo permite.

---

## Parte IV — Por qué el consenso de TSN es ciego al orden

La [teoría](teoria) afirma que el promedio de TSN es invariante al orden. La demostración es de una línea, pero vale hacerla explícita porque delimita exactamente qué agrega TSM.

Sea $f_\theta$ la CNN 2D compartida y $\pi$ una permutación de $\{1, \dots, N\}$. La predicción de TSN es

$$\text{TSN}(x_1, \dots, x_N) \;=\; \sigma\!\left(\frac{1}{N}\sum_{k=1}^{N} f_\theta(x_k)\right)$$

Como la suma es conmutativa,

$$\text{TSN}(x_{\pi(1)}, \dots, x_{\pi(N)}) \;=\; \sigma\!\left(\frac{1}{N}\sum_{k=1}^{N} f_\theta(x_{\pi(k)})\right) \;=\; \text{TSN}(x_1, \dots, x_N)$$

para **toda** permutación $\pi$. En particular para la reversión temporal. TSN no puede distinguir "abrir una puerta" de "cerrar una puerta", y no por falta de capacidad del backbone sino por la estructura de la agregación: la información de orden se destruye en el promedio, después de que cada frame ya fue procesado por separado.

### Dónde se rompe la simetría en TSM

Con desplazamiento, la entrada efectiva al bloque $k$ deja de ser $x_k$ y pasa a ser una función de la terna $(x_{k-1}, x_k, x_{k+1})$:

$$\text{TSM}(x_1, \dots, x_N) \;=\; \sigma\!\left(\frac{1}{N}\sum_{k=1}^{N} g_\theta(x_{k-1}, x_k, x_{k+1})\right)$$

El promedio exterior sigue siendo conmutativo, pero **cada término ya no depende solo de su frame**: permutar la secuencia cambia las ternas y por lo tanto los sumandos. La invarianza se rompe dentro de $g_\theta$, no en la agregación.

Un caso vale examinarlo aparte porque es tentador. Bajo **reversión** temporal, los roles de $\mathcal{A}$ (futuro) y $\mathcal{B}$ (pasado) se intercambian. ¿Es TSM invariante a reversión? No, y la razón es que los dos bloques ocupan **rangos de canales distintos**: el peso $W_{:,c}$ que aprendió a leer el futuro está en columnas que, tras la reversión, reciben el pasado. La red solo sería invariante si $W$ fuera simétrica bajo el intercambio $\mathcal{A} \leftrightarrow \mathcal{B}$, lo que nada en el entrenamiento impone.

Que la simetría esté rota en la arquitectura no garantiza que el modelo **use** el orden: el [Laboratorio 38](/laboratorios/lab-38/05-invertir-el-tiempo) midió que I3D —que también tiene la simetría rota— predice prácticamente lo mismo con el video invertido, porque Kinetics no lo obliga a distinguirlo. Es la diferencia entre poder y necesitar.

---

## Parte V — El costo real: bytes, no FLOPs

El argumento de eficiencia del paper es que el desplazamiento cuesta cero FLOPs pero sí movimiento de datos, y que por eso hay que desplazar poco. Vale hacer la cuenta.

### Intensidad aritmética

La métrica que decide si una operación está limitada por cómputo o por memoria es la razón entre operaciones y bytes transferidos:

$$I \;=\; \frac{\text{FLOPs}}{\text{bytes movidos}}$$

Para una convolución $1\times1$ sobre un tensor de $C \times H \times W$: FLOPs $\approx 2 C_{\text{in}} C_{\text{out}} HW$, bytes $\approx 4(C_{\text{in}} + C_{\text{out}})HW$, de modo que $I$ es del orden de cientos. Para el desplazamiento: **FLOPs $= 0$**, bytes $> 0$, y por lo tanto

$$I_{\text{shift}} = 0$$

El desplazamiento es una operación *memory-bound* pura: en un diagrama roofline no está cerca del techo de cómputo, está pegada al eje. Su tiempo lo determina exclusivamente el ancho de banda. Por eso su costo relativo **crece** cuando el hardware es más rápido en cómputo, y por eso el paper mide overheads del 12-14 % para algo que "no cuesta nada".

### Cuánto se mueve realmente

Acá aparece un detalle de implementación que conviene mirar de cerca. El código de referencia es:

```python
out = torch.zeros_like(x)                          # (1) escribe C·T·H·W ceros
out[:, :-1, :fold]        = x[:, 1:, :fold]        # (2) copia F canales
out[:, 1:, fold:2*fold]   = x[:, :-1, fold:2*fold] # (3) copia F canales
out[:, :,  2*fold:]       = x[:, :,  2*fold:]      # (4) copia C−2F canales
```

Las líneas (2) y (3) mueven $2F$ canales, que es lo que el *partial shift* pretende minimizar. Pero la línea (1) escribe el tensor **completo** y la línea (4) copia los $C - 2F$ canales restantes. Sumando, el tráfico total es del orden de

$$\underbrace{C}_{\text{(1) escritura}} + \underbrace{C}_{\text{(2)(3)(4) lectura}} + \underbrace{C}_{\text{(2)(3)(4) escritura}} \;=\; 3C \quad \text{canales de tráfico, independiente de } F$$

{{< concept-alert type="cuidado" >}}
**En esta implementación, el ahorro del partial shift no se materializa en movimiento de datos.** El tráfico es proporcional a $C$, no a $F$: reducir la fracción desplazada no reduce los bytes movidos, porque el tensor se copia entero de todos modos.

El ahorro que el paper reporta corresponde a una implementación *in-place* que solo toca los canales desplazados —presente en el repositorio como la clase `InplaceShift`— pero **deshabilitada** con un `raise NotImplementedError`, por errores de orden en ejecución paralela. Es decir: el código que se ejecuta en el laboratorio conserva el beneficio de precisión del partial shift, pero no su beneficio de latencia.

Esta es una lectura del código, no una medición. Es falsable de forma directa: cronometrar la inferencia con `fold_div = 2`, `8` y $10^9$; si los tres tiempos coinciden dentro del ruido, la latencia es independiente de la fracción desplazada.
{{< /concept-alert >}}

### Dónde duele más

El tráfico por módulo es $\propto C \cdot T \cdot H \cdot W$. En una ResNet-50 con entrada de $224\times224$ y $T = 8$, ese producto es notablemente **constante** entre etapas, porque cada duplicación de canales viene con una reducción a la cuarta parte del área espacial:

| Etapa | $C$ | $H \times W$ | $C \cdot H \cdot W$ | Tensor (fp32, $T=8$) |
|---|---|---|---|---|
| `layer1` | 256 | $56 \times 56$ | 802 816 | 25,7 MB |
| `layer2` | 512 | $28 \times 28$ | 401 408 | 12,8 MB |
| `layer3` | 1024 | $14 \times 14$ | 200 704 | 6,4 MB |
| `layer4` | 2048 | $7 \times 7$ | 100 352 | 3,2 MB |

El costo se concentra en las **etapas tempranas**: un módulo en `layer1` mueve ocho veces más datos que uno en `layer4`. Y como en las primeras capas las convoluciones son comparativamente baratas, ahí es donde el desplazamiento pesa más en proporción. Un diseño consciente del ancho de banda insertaría menos módulos abajo y más arriba — lo contrario de lo que hace la implementación de referencia, que los pone en los 16 bloques por igual.

---

## Resumen de lo que se deriva

| Afirmación de la clase | Lo que dice la matemática |
|---|---|
| "La convolución es desplazamiento + multiplicación" | Cierto, pero TSM implementa una convolución temporal **con soportes de columna disjuntos**: un tercio de los grados de libertad de una $3\times1\times1$ genuina |
| "Cero costo computacional" | Cero FLOPs. Intensidad aritmética cero ⇒ *memory-bound* puro; el overhead medido es de 3-14 % según la fracción |
| "El campo receptivo temporal se amplía en 2 por módulo" | Cota correcta. El alcance **efectivo** crece como $\sqrt{L}/2$: con 16 módulos, $\pm 2$ frames, no $\pm 16$ |
| "1/8 del mapa de características" | 1/8 **por dirección** en el modo bidireccional; 1/4 en total. El 1/8 literal corresponde al modo online |
| "Partial shift reduce el movimiento de datos" | Cierto en la implementación *in-place*, que está deshabilitada. En la que se ejecuta, el tráfico es $\propto C$ e independiente de la fracción |

---

## Ver también

- [Teoría](teoria) — el recorrido de las 29 diapositivas.
- [Práctica desde 0](practica) — implementar y verificar numéricamente la equivalencia de la Parte I, en triple framework.
- [Laboratorio 40](/laboratorios/lab-40) — la verificación empírica: la ablación, la curva de proporción y el modo online sobre un checkpoint entrenado.
- [Fundamento: Desplazamiento Temporal](/fundamentos/desplazamiento-temporal) — el mecanismo presentado de forma autónoma.
- [Clase 38 - Profundización](/clases/clase-38/profundizacion) — la matemática de la estrategia opuesta: el punto fijo del video aburrido.
