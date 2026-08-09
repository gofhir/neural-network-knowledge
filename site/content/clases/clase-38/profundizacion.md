---
title: "Profundización - Inflado, factorización y el costo del tiempo"
weight: 20
math: true
---

> **Matemática de la Clase 38.** La [teoría](teoria) recorre la escalera de arquitecturas; acá se derivan las cuatro cuentas que la sostienen. **Parte I**: la condición de punto fijo del video aburrido —qué exige exactamente, por qué la división por $N$ es una solución y no *la* solución, y qué pasa con BatchNorm, max-pooling y el padding. **Parte II**: contabilidad de parámetros y FLOPs, incluyendo de dónde salen realmente los 79M de C3D contra los 25M de I3D. **Parte III**: la factorización $(2+1)$D y la derivación del hiperparámetro que hace justa la comparación con 3D. **Parte IV**: el campo receptivo temporal, y una jerarquía formal de sensibilidad al orden. **Parte V**: por qué tratar el tiempo y el espacio simétricamente es un error de diseño.
>
> La [Clase 36](/clases/clase-36/profundizacion) ya derivó el flujo óptico, LRCN y la convolución 3D básica. Acá no se repiten: el foco es el mecanismo de transferencia de pesos, que es el tema propio de esta clase.

---

# Parte I — El punto fijo del video aburrido

El slide 23 de la clase enuncia el truco de I3D en una frase: *"gracias a la linealidad, repetir los pesos de los filtros 2D $N$ veces a lo largo de la dimensión temporal y escalarlos dividiendo por $N$"*. Vale derivarlo, porque la frase esconde una condición más general y tres excepciones prácticas.

## 1.1 La condición de equivalencia

Sea una capa convolucional 2D con pesos $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$ y sesgo $b \in \mathbb{R}^{C_{\text{out}}}$. Sobre una imagen $x \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ produce

$$y[c_o, i, j] \;=\; \sum_{c_i=1}^{C_{\text{in}}} \sum_{u=1}^{k} \sum_{v=1}^{k} W[c_o, c_i, u, v]\; x[c_i,\, i+u,\, j+v] \;+\; b[c_o]$$

Ahora construyamos el **video aburrido**: la misma imagen repetida a lo largo del tiempo,

$$\tilde{x}[c_i, t, i, j] \;=\; x[c_i, i, j] \qquad \forall\, t \in \{1,\dots,T\}$$

y hagámoslo pasar por una capa convolucional 3D con pesos $\widetilde{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times N \times k \times k}$, donde $N$ es la extensión temporal del kernel:

$$\tilde{y}[c_o, t, i, j] \;=\; \sum_{c_i} \sum_{\tau=1}^{N} \sum_{u,v} \widetilde{W}[c_o, c_i, \tau, u, v]\; \tilde{x}[c_i,\, t+\tau,\, i+u,\, j+v] \;+\; b[c_o]$$

Como $\tilde{x}$ **no depende de $t$**, el índice temporal de la entrada es irrelevante y la suma sobre $\tau$ se puede factorizar:

$$\tilde{y}[c_o, t, i, j] \;=\; \sum_{c_i} \sum_{u,v} \underbrace{\left(\sum_{\tau=1}^{N} \widetilde{W}[c_o, c_i, \tau, u, v]\right)}_{\text{peso efectivo}} x[c_i,\, i+u,\, j+v] \;+\; b[c_o]$$

Comparando con la ecuación 2D, la equivalencia $\tilde{y}[c_o,t,i,j] = y[c_o,i,j]$ se cumple si y solo si

$$\boxed{\;\sum_{\tau=1}^{N} \widetilde{W}[c_o, c_i, \tau, u, v] \;=\; W[c_o, c_i, u, v]\;}$$

{{< concept-alert type="clave" >}}
**La condición es sobre la suma, no sobre cada peso.** Lo que el punto fijo exige es que los pesos inflados **sumen** el peso 2D original a lo largo del eje temporal. El reparto uniforme $\widetilde{W}[\cdot,\tau,\cdot] = W/N$ satisface la condición, pero es **una** solución entre infinitas. La división por $N$ del slide es la elección de I3D, no una necesidad matemática.
{{< /concept-alert >}}

## 1.2 Dos inicializaciones válidas y por qué difieren

Dos soluciones de la condición aparecen en implementaciones reales:

**Reparto uniforme** (el de I3D):

$$\widetilde{W}[c_o,c_i,\tau,u,v] \;=\; \frac{1}{N} W[c_o,c_i,u,v] \qquad \forall \tau$$

**Inicialización central** (delta de Dirac temporal), usada en varias implementaciones de inflado de ResNet:

$$\widetilde{W}[c_o,c_i,\tau,u,v] \;=\; \begin{cases} W[c_o,c_i,u,v] & \text{si } \tau = \lceil N/2 \rceil \\ 0 & \text{en otro caso}\end{cases}$$

Ambas producen exactamente la misma salida sobre un video aburrido. **Difieren en video real**, y la diferencia es interpretable:

| Inicialización | Comportamiento inicial sobre video real | Sensible al padding temporal | Riesgo |
|---|---|---|---|
| Uniforme ($W/N$) | Promedia temporalmente antes de convolucionar: actúa como un filtro pasa-bajos en el tiempo | **Sí** (ver 1.4) | Suaviza el movimiento; arranca ciego a cambios rápidos |
| Central (delta) | Ignora los frames vecinos: es exactamente la red 2D aplicada frame a frame | **No** | Arranca sin ninguna sensibilidad temporal, con gradiente que debe "descubrir" los taps vecinos desde cero |

La uniforme le da al modelo un sesgo inicial hacia la estabilidad temporal; la central lo deja idéntico a un modelo por frame. En la práctica ambas funcionan y la elección se trata como hiperparámetro; conviene saber que existen porque los checkpoints inflados no son intercambiables entre convenciones.

## 1.3 Por qué "gracias a la linealidad": la propagación capa a capa

La equivalencia de una capa no basta — hace falta que la propiedad **se propague por toda la red**. El argumento es inductivo. Digamos que un tensor de activaciones es *temporalmente constante* si no depende de $t$. Entonces:

1. **Capa convolucional 3D inflada**: si la entrada es temporalmente constante, la salida es temporalmente constante (lo acabamos de demostrar) y coincide con la 2D.
2. **No linealidad puntual** ($\mathrm{ReLU}$, sigmoide): se aplica elemento a elemento, así que preserva la constancia y conmuta con la repetición: $\sigma(\tilde{y})[t] = \sigma(y)$.
3. **Average pooling inflado**: el promedio de $N$ copias idénticas es la copia. Preserva.
4. **Stride temporal $s > 1$**: submuestrear una señal constante da una señal constante. Preserva.
5. **Global average pooling final**: colapsa el eje temporal promediando valores idénticos, devolviendo el valor 2D.

Por inducción sobre la profundidad, la red inflada entera evaluada sobre un video aburrido reproduce exactamente el logit de la red 2D sobre la imagen. Ese es el contenido preciso de la frase del slide: la **linealidad** de la convolución es lo que permite factorizar la suma sobre $\tau$ en el paso 1, y la **puntualidad** de las activaciones es lo que permite el paso 2.

## 1.4 Tres excepciones que hay que tratar a mano

{{< concept-alert type="advertencia" >}}
**Max-pooling no se divide por $N$.** El razonamiento de la división aplica a operaciones que **suman** sobre el eje temporal. Un max-pooling toma el máximo, y el máximo de $N$ copias idénticas ya es la copia:
$$\max_{\tau=1..N} \tilde{x}[t+\tau] = x$$
Inflar un max-pool consiste simplemente en extender la ventana al eje temporal, sin reescalar nada. Dividir los valores por $N$ ahí rompería el punto fijo en lugar de preservarlo. Es el error más frecuente al implementar el inflado a mano.
{{< /concept-alert >}}

**BatchNorm.** Los parámetros aprendidos ($\gamma, \beta$) y las estadísticas acumuladas (media y varianza por canal) se copian sin modificación, y eso es correcto: sobre video aburrido las pre-activaciones son idénticas a las 2D, así que las estadísticas heredadas son exactamente las válidas. El detalle práctico es que **sobre video real dejan de serlo** —la distribución de activaciones cambia cuando la entrada tiene movimiento—, así que las estadísticas se re-estiman durante el fine-tuning. Ver [Regularización](/fundamentos/regularizacion).

**Padding temporal.** La derivación de 1.1 supone que la entrada es constante en toda la ventana del kernel. En los **bordes temporales** del clip, el padding con ceros inyecta valores que no son la imagen, así que el punto fijo se cumple exactamente en el interior y solo de forma aproximada en el primer y último $\lfloor N/2 \rfloor$ frames: la activación de borde queda escalada por un factor $\frac{N-1}{N}$.

Esto tiene una consecuencia que la fórmula no anticipa y que la [práctica](practica) mide. Como el clasificador termina con un global average pooling que promedia **todos** los frames, el error de borde contamina el logit final, y su magnitud decae como $1/T$: sobre una ResNet-18 inflada con el modo uniforme, la desviación respecto de la red 2D fue de $5{,}30$ logits con $T=4$ y bajó a $0{,}35$ con $T=64$ (un número fijo de frames corruptos diluido por un promedio cada vez más largo). El modo de **delta central es inmune**: reproduce el logit 2D a precisión de máquina en todos los frames, porque un tap que solo mira el frame central nunca toca el padding. Es un argumento práctico a favor de la delta que no aparece en el paper de I3D.

---

# Parte II — La contabilidad: parámetros y FLOPs

## 2.1 El factor que agrega la tercera dimensión

Para una capa convolucional con $C_{\text{in}}$ canales de entrada, $C_{\text{out}}$ de salida y kernel espacial $k \times k$:

$$P_{\text{2D}} = C_{\text{in}} \, C_{\text{out}} \, k^2 \qquad\qquad P_{\text{3D}} = C_{\text{in}} \, C_{\text{out}} \, k^2 \, t$$

El costo en operaciones, evaluado sobre un mapa de salida de $H \times W$ (y $T$ pasos temporales en el caso 3D):

$$F_{\text{2D}} = C_{\text{in}} \, C_{\text{out}} \, k^2 \, H W \qquad\qquad F_{\text{3D}} = C_{\text{in}} \, C_{\text{out}} \, k^2 \, t \cdot T H W$$

Los parámetros crecen por un factor $t$ (la extensión temporal del kernel); los FLOPs crecen por $t \cdot T$ — el kernel es más grande **y** hay que evaluarlo en más posiciones. Con $t=3$ y $T=64$, eso es un factor $192$ sobre la misma capa 2D.

## 2.2 De dónde salen realmente los 79M de C3D

La tabla del slide 26 dice que C3D tiene 79M de parámetros y I3D 25M, y el slide de I3D lo lista como ventaja ("reduce el número de parámetros"). Es tentador atribuirlo al inflado, pero la cuenta muestra otra cosa. Recorramos [C3D](/papers/c3d-tran-2015) con la arquitectura del slide 20 (kernels $3\times3\times3$ en todas las convoluciones):

| Capa | $C_{\text{in}} \to C_{\text{out}}$ | Parámetros |
|---|---|---|
| Conv1a | $3 \to 64$ | $3 \cdot 64 \cdot 27 \approx 5{,}2\text{K}$ |
| Conv2a | $64 \to 128$ | $\approx 221\text{K}$ |
| Conv3a, Conv3b | $128 \to 256$, $256 \to 256$ | $\approx 884\text{K} + 1{,}77\text{M}$ |
| Conv4a, Conv4b | $256 \to 512$, $512 \to 512$ | $\approx 3{,}54\text{M} + 7{,}08\text{M}$ |
| Conv5a, Conv5b | $512 \to 512$ | $\approx 7{,}08\text{M} + 7{,}08\text{M}$ |
| **Subtotal convoluciones** | | $\approx 27{,}7\text{M}$ |
| fc6 | $8192 \to 4096$ | $\approx 33{,}6\text{M}$ |
| fc7 | $4096 \to 4096$ | $\approx 16{,}8\text{M}$ |
| **Subtotal densas** | | $\approx 50{,}3\text{M}$ |
| **Total** | | $\approx 78\text{M}$ |

{{< concept-alert type="clave" >}}
**Casi dos tercios de C3D son sus dos capas densas, no sus convoluciones 3D.** De los ~78M, unos 50M viven en `fc6` y `fc7`. I3D no es más liviano porque el inflado ahorre parámetros —el inflado, de hecho, **multiplica** los pesos convolucionales por $N$— sino porque **Inception-v1 reemplaza las capas densas gigantes por global average pooling**. La ventaja del slide es real, pero su causa es la topología heredada, no la técnica de inflado. Es el mismo hallazgo que hizo [GoogLeNet](/papers/googlenet-szegedy-2014) en imágenes, cobrado de nuevo en video.
{{< /concept-alert >}}

## 2.3 Por qué las capas bajas son las caras

La fórmula de FLOPs contiene el producto $H W$, y en una CNN la resolución espacial **decrece** con la profundidad mientras los canales crecen. Eso produce una asimetría contraintuitiva: una convolución 3D en una capa temprana, con pocos canales pero resolución alta, puede costar más operaciones que una convolución 3D en una capa profunda con muchos canales y resolución baja.

Tomemos una entrada de $64 \times 112 \times 112$ (tiempo × alto × ancho) tras el primer stride, contra un mapa profundo de $8 \times 7 \times 7$:

$$\frac{F_{\text{temprana}}}{F_{\text{profunda}}} \;=\; \frac{C_{\text{in}}^{(1)} C_{\text{out}}^{(1)} \cdot 64 \cdot 112^2}{C_{\text{in}}^{(5)} C_{\text{out}}^{(5)} \cdot 8 \cdot 7^2}$$

El factor espacio-temporal solo, $\frac{64 \cdot 12544}{8 \cdot 49} \approx 2048$, tiene que ser compensado por el producto de canales para que la capa profunda resulte más costosa. Con $64 \times 64$ canales arriba y $512 \times 512$ abajo, el cociente de canales es $64$ — insuficiente para cerrar la brecha de 2048.

Esta es la razón cuantitativa detrás del hallazgo *top-heavy* de [S3D](/papers/s3d-xie-2018): **poner las convoluciones 3D solo en las capas profundas** conserva la capacidad de modelar movimiento donde los features son semánticos, y elimina precisamente las convoluciones 3D más caras. Es contraintuitivo porque uno esperaría que el movimiento se capture cerca de los píxeles; el resultado empírico dice que no hace falta.

---

# Parte III — La factorización $(2+1)$D

## 3.1 Descomponer el cubo

En lugar de un kernel $t \times k \times k$, se aplican en secuencia un kernel **espacial** $1 \times k \times k$ y un kernel **temporal** $t \times 1 \times 1$, con una no linealidad entre ambos. Con $M$ canales intermedios:

$$\underbrace{C_{\text{in}} \xrightarrow{\;1 \times k \times k\;} M}_{\text{espacial}} \quad \xrightarrow{\;\mathrm{ReLU}\;} \quad \underbrace{M \xrightarrow{\;t \times 1 \times 1\;} C_{\text{out}}}_{\text{temporal}}$$

Los parámetros pasan de $C_{\text{in}} C_{\text{out}} k^2 t$ a

$$P_{(2+1)\text{D}} \;=\; \underbrace{C_{\text{in}} \, M \, k^2}_{\text{espacial}} \;+\; \underbrace{M \, C_{\text{out}} \, t}_{\text{temporal}}$$

## 3.2 El hiperparámetro que hace justa la comparación

Acá está la elegancia del diseño de [R(2+1)D](/papers/r2plus1d-tran-2018): en lugar de presumir que factorizar es mejor porque ahorra parámetros, los autores eligen $M$ **para igualar exactamente el conteo de parámetros** del bloque 3D, de modo que cualquier diferencia de precisión no se pueda atribuir a capacidad. Igualando:

$$C_{\text{in}} M k^2 + M C_{\text{out}} t \;=\; C_{\text{in}} C_{\text{out}} k^2 t$$

$$\boxed{\;M \;=\; \frac{t \, k^2 \, C_{\text{in}} \, C_{\text{out}}}{k^2 \, C_{\text{in}} + t \, C_{\text{out}}}\;}$$

Con el caso típico $k=3$, $t=3$ y $C_{\text{in}} = C_{\text{out}} = C$:

$$M \;=\; \frac{3 \cdot 9 \cdot C^2}{9C + 3C} \;=\; \frac{27C^2}{12C} \;=\; 2{,}25\,C$$

El bloque factorizado usa $2{,}25$ veces más canales en su capa intermedia que el bloque 3D equivalente, y con eso empata en parámetros. Lo que gana no es capacidad:

1. **El doble de no linealidades.** Donde el bloque 3D tiene un $\mathrm{ReLU}$, el factorizado tiene dos. Para una red de $L$ bloques, eso duplica la cantidad de regiones lineales que la función puede representar — es capacidad expresiva a costo cero en parámetros.
2. **Optimización más fácil.** Este es el argumento más fino del paper, y el más fácil de malinterpretar. Los autores muestran que R(2+1)D alcanza **menor error de entrenamiento** que R3D, no solo menor error de test. Si el beneficio fuera regularización, esperaríamos lo contrario: más error de entrenamiento y menos de test. Que baje el de entrenamiento significa que la factorización hace el problema de optimización **más tratable**, presumiblemente porque desacopla dos tipos de estructura que el kernel cúbico tiene que aprender entrelazados.

{{< concept-alert type="advertencia" >}}
**S3D y R(2+1)D parecen no coincidir, y la lectura fina importa.** Los dos papers son contemporáneos (2018) y llegan a la separabilidad de forma independiente, pero sus conclusiones sobre *dónde* poner las convoluciones 3D parecen opuestas: S3D concluye que conviene 2D abajo y 3D arriba (*top-heavy*, sobre Inception), mientras que R(2+1)D reporta que las convoluciones mixtas con 3D abajo (MC$x$) superan a las invertidas (rMC$x$).

Al mirar los números, la contradicción casi se disuelve. En R(2+1)D la ventaja de MC$x$ sobre rMC$x$ es de **0.4 a 0.9 puntos a nivel de clip**, y **a nivel de video con 16 fotogramas se evapora**: rMC3 obtiene 65.0 contra 64.7 de MC3, y rMC4 empata con MC4 y MC5 en 65.1. El único perdedor claro es rMC5 (63.1), el caso degenerado de poner 3D solo en el último grupo. El resultado que sí es sólido en ese paper es **económico**: MC$x$ iguala a R3D con 11.4M de parámetros contra 33.4M, es decir 2.93× menos. El titular no es "3D abajo es mejor" sino "3D abajo es *suficiente*".

La lectura prudente: la separabilidad es un resultado robusto en ambos papers, mientras que la distribución óptima de 3D en la profundidad depende del backbone y del protocolo, y cualquier afirmación fuerte del tipo "el movimiento es una feature de bajo nivel" está sobre-interpretando diferencias menores a un punto.
{{< /concept-alert >}}

## 3.3 Cómo se infla un kernel separable

Una pregunta natural: si el kernel ya no es cúbico, ¿sobrevive el truco del inflado? Sí, y de forma más limpia. Para el par $(1 \times k \times k) \to (t \times 1 \times 1)$:

- El kernel **espacial** $1 \times k \times k$ tiene extensión temporal $N = 1$, así que la condición de la Parte I se satisface con $\widetilde{W} = W$ **sin división alguna**: se copian los pesos de ImageNet tal cual.
- El kernel **temporal** $t \times 1 \times 1$ no tiene análogo 2D del cual heredar. Se inicializa con la delta central (identidad temporal): peso $1$ en $\tau = \lceil t/2 \rceil$ y $0$ en el resto.

Con esa combinación, la red separable inflada **arranca siendo exactamente la red 2D aplicada frame a frame**, y el entrenamiento solo tiene que aprender los taps temporales. Es una inicialización mejor condicionada que el reparto uniforme de I3D, donde el gradiente debe primero *romper* la simetría entre los $N$ taps idénticos antes de poder especializarlos.

Conviene aclarar el estatus de esta receta: es la lectura mecánica de la condición de la Parte I aplicada a un kernel separable, **no un procedimiento que los papers de S3D o R(2+1)D especifiquen**. Ninguno de los dos detalla cómo inicializar el kernel temporal nuevo. La observación empírica que la respalda de forma indirecta es la figura de S3D sobre los pesos aprendidos: en las capas bajas, los taps temporales $t \neq 0$ terminan centrados en cero — la red **apaga sola** la dimensión temporal abajo, que es justamente el estado en que la delta central la deja al inicializar.

---

# Parte IV — Campo receptivo temporal y sensibilidad al orden

## 4.1 Cómo crece el campo receptivo del slide 22

El slide de la arquitectura Inflated Inception-V1 anota los campos receptivos en la forma (temporal, vertical, horizontal): parte en $7{,}11{,}11$, pasa por $11{,}27{,}27$, $23{,}75{,}75$, $59{,}219{,}219$ y termina en $99{,}539{,}539$. Esos números siguen la recurrencia estándar del campo receptivo:

$$R_l \;=\; R_{l-1} \;+\; (k_l - 1) \prod_{m < l} s_m$$

donde $k_l$ es el tamaño del kernel de la capa $l$ y $s_m$ los strides acumulados. Cada capa suma su extensión de kernel escalada por todo el submuestreo previo, lo que hace que el crecimiento sea **multiplicativo en la profundidad**.

{{< concept-alert type="clave" >}}
**El campo receptivo temporal final (99) es mayor que el clip de entrenamiento (64).** I3D entrena con clips de 64 frames, pero sus últimas capas tienen un campo receptivo temporal de 99 frames. Es decir: **en la parte profunda de la red, cada activación ve el clip completo**. A 25 fps, 99 frames son unos 4 segundos. Esta es la explicación técnica de por qué I3D "exprime más el stream de flujo" que las demás arquitecturas —tiene 64 frames de huella temporal en entrenamiento contra los 10 del Two-Stream original— y de por qué sus features transfieren mejor a UCF-101 y HMDB-51.
{{< /concept-alert >}}

Nótese también el desbalance deliberado: el campo receptivo espacial (539) crece mucho más rápido que el temporal (99). No es un descuido — es la asimetría de la Parte V aplicada al diseño.

## 4.2 Una jerarquía de sensibilidad al orden

Las cinco arquitecturas de la clase se pueden ordenar formalmente por cuánta información de orden temporal pueden usar. Sea $\pi$ una permutación de los índices de frame.

| Arquitectura | Relación con $\pi$ | Consecuencia |
|---|---|---|
| CNN2D + promedio | $\hat{y}(x_{\pi}) = \hat{y}(x)$ **para toda** $\pi$ | Invariante al orden por construcción; no puede distinguir *abrir* de *cerrar* |
| CNN2D + LSTM | $\hat{y}(x_{\pi}) \neq \hat{y}(x)$, pero sobre features ya colapsados espacialmente | Sensible al orden **de eventos**; ciega al movimiento de bajo nivel |
| Two-Stream | Sensible al orden dentro de la ventana de flujo (10 frames); invariante fuera | Movimiento fino sí, estructura de largo alcance no |
| C3D / I3D | Sensible al orden dentro del campo receptivo temporal $R_l$, que crece con $l$ | Movimiento fino en capas bajas, estructura de evento en capas profundas |

La demostración de la primera fila está en la [teoría](teoria); su verificación empírica, en el [Laboratorio 36](/laboratorios/lab-36), donde muestrear 4 frames rindió igual que 8 — la firma inequívoca de un modelo que trata los frames como votos independientes.

La lectura que ordena la tabla: **el progreso de la escalera consiste en mover la sensibilidad al orden desde el final de la red hacia el interior de la red**. El promedio y la LSTM la ponen después de todo el procesamiento espacial; la convolución 3D la distribuye en cada capa. Eso es lo que "entretejer el módulo temporal" significa en concreto.

---

# Parte V — La asimetría espacio-temporal

## 5.1 Por qué un kernel cúbico es una mala apuesta

Inflar $k \times k$ a $k \times k \times k$ es la extensión más simple, y precisamente por simple asume algo falso: que el eje temporal tiene la misma estructura estadística que los ejes espaciales. No la tiene.

En una imagen natural, los dos ejes espaciales son **intercambiables**: la estadística de correlación entre píxeles es aproximadamente isótropa, y una rotación de 90° produce otra imagen natural plausible. El eje temporal no es intercambiable con ninguno de los dos. Tiene:

- **Una escala física distinta.** Un desplazamiento de un frame corresponde a $1/\text{fps}$ segundos, y la correlación entre frames consecutivos es típicamente **mucho mayor** que entre píxeles vecinos: el mundo cambia lento respecto de la tasa de muestreo.
- **Una dirección privilegiada.** El tiempo tiene flecha; el espacio no. Invertir el eje temporal cambia la clase (*sentarse* $\to$ *pararse*); invertir un eje espacial casi nunca lo hace, razón por la cual el flip horizontal es una augmentación estándar y el flip temporal no lo es. Ver [Data Augmentation](/fundamentos/data-augmentation).
- **Una resolución mucho menor.** Un clip típico tiene $224 \times 224$ píxeles espaciales y 8 a 64 frames. Decimar el tiempo con la misma agresividad que el espacio lo agota en dos o tres capas — de ahí que I3D use `stride 1,2,2` en sus primeros pooling: **no toca el tiempo**.

## 5.2 El framerate como tasa de muestreo

Hay un puente útil con la [Clase 35](/clases/clase-35) que vale explicitar: el framerate **es una frecuencia de muestreo**, y todo lo que el teorema de muestreo de Nyquist-Shannon dice sobre audio aplica al eje temporal del video. Un movimiento cuya frecuencia excede $\text{fps}/2$ produce **aliasing temporal** — el efecto de la rueda que parece girar al revés. Ver [Digitalización de Audio](/fundamentos/digitalizacion-de-audio).

Esto reencuadra el problema del diseño de arquitecturas de video: elegir cuántos frames muestrear no es solo un trade-off de cómputo, es elegir **qué banda de movimiento es representable**. Un modelo que muestrea 8 frames de un clip de 10 segundos opera a $0{,}8$ fps efectivos y no puede, en principio, representar nada más rápido que $0{,}4$ Hz. Ninguna capacidad de la red compensa información que no entró.

## 5.3 Separar por framerate en lugar de por modalidad

[SlowFast](/papers/slowfast-feichtenhofer-2019) toma esta asimetría como principio de diseño. En lugar de dos corrientes que se distinguen por **modalidad** (RGB contra flujo óptico), usa dos vías que se distinguen por **resolución temporal**:

- Una vía **Slow**: framerate bajo (stride temporal $\tau$), muchos canales. Resuelve apariencia y semántica, donde la resolución espacial importa.
- Una vía **Fast**: framerate alto (stride $\tau/\alpha$), pocos canales (una fracción $\beta$ de los del Slow). Resuelve movimiento, donde importa la resolución temporal y no el detalle.

El costo relativo de la vía rápida se puede estimar con la fórmula de FLOPs de la Parte II. Los canales entran al cuadrado (afectan $C_{\text{in}}$ y $C_{\text{out}}$) y el framerate linealmente:

$$\frac{F_{\text{Fast}}}{F_{\text{Slow}}} \;\approx\; \beta^2 \cdot \alpha$$

Con los valores típicos $\beta = 1/8$ y $\alpha = 8$: $\;\frac{1}{64} \cdot 8 = \frac{1}{8}$. Es la observación de diseño clave — **una vía puede ser 8 veces más rápida temporalmente y costar una fracción del total**, porque adelgazar los canales ahorra cuadráticamente mientras densificar el tiempo cuesta linealmente. La cifra exacta que reporta el paper es algo mayor que esta estimación, porque la vía Fast no aplica submuestreo temporal en ninguna etapa y las conexiones laterales agregan cómputo propio; el análisis detallado está en [el paper](/papers/slowfast-feichtenhofer-2019).

{{< concept-alert type="clave" >}}
**El cierre del arco de la clase.** La escalera CNN2D → RNN → Two-Stream → C3D → I3D es una historia sobre cómo conseguir capacidad temporal sin perder el pre-entrenamiento de ImageNet. Lo que muestran S3D, R(2+1)D y SlowFast es que la pregunta correcta no era *"cómo hago 3D lo que funcionaba en 2D"* sino *"cuánta capacidad 3D necesito realmente, y dónde"*. La respuesta —menos de la que parecía, y no en las capas bajas— es lo que volvió práctico el análisis de video.
{{< /concept-alert >}}

---

## Síntesis

| Cuenta | Resultado | Dónde importa |
|---|---|---|
| Condición de punto fijo | $\sum_{\tau} \widetilde{W}[\cdot,\tau,\cdot] = W$ | Cualquier reparto que sume el peso 2D sirve; uniforme y delta central son ambos válidos |
| Parámetros 3D vs 2D | Factor $t$ | El inflado **multiplica** pesos; el ahorro de I3D viene de Inception, no del inflado |
| FLOPs 3D vs 2D | Factor $t \cdot T$ | Las capas de alta resolución espacial son las caras: base del diseño *top-heavy* |
| Canales de $(2+1)$D a igual parámetros | $M = \dfrac{t k^2 C_{\text{in}} C_{\text{out}}}{k^2 C_{\text{in}} + t C_{\text{out}}} = 2{,}25\,C$ típico | Hace la comparación con 3D justa; el beneficio es no linealidades + optimización |
| Campo receptivo temporal de I3D | 99 frames > 64 de entrada | Explica su ventaja en transferencia y en explotar el flujo óptico |
| Costo relativo de la vía Fast | $\approx \beta^2 \alpha$ | Adelgazar canales ahorra cuadráticamente; densificar tiempo cuesta lineal |

La implementación de estas cuentas en PyTorch, TensorFlow y JAX está en la [práctica desde 0](practica).
