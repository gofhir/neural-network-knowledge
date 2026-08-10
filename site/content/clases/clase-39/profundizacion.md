---
title: "Profundización - Campo receptivo, dilatación y el costo del contexto"
weight: 20
math: true
---

> **Matemática de la Clase 39.** La [teoría](teoria) recorre las diapositivas; acá se derivan las cuentas que las sostienen. **Parte I**: el campo receptivo, la fórmula general, las tres estrategias para crecerlo y la condición que evita los huecos — con la aritmética completa del "Ejemplo 2" del slide. **Parte II**: la contabilidad de parámetros del "Ejemplo 1", y por qué la capa de reducción es la pieza que decide el tamaño de la red. **Parte III**: por qué un espectrograma no es una imagen, formalizado como una pregunta sobre qué grupo de transformaciones deja invariante a la etiqueta. **Parte IV**: el costo comparado de convolución, recurrencia y atención sobre longitudes reales de audio, que es donde la objeción 3 de la clase se puede evaluar con números. **Parte V**: causalidad, latencia y el factor 2 que se gana al no necesitarla.
>
> Las cifras de este documento se calcularon ejecutando el código, no de memoria. Los cálculos son reproducibles con lo que está en la [práctica](practica).

---

# Parte I — El campo receptivo

En visión uno rara vez calcula el campo receptivo de una red: una imagen de $224\times224$ se cubre entera con una arquitectura estándar, y el asunto no aparece. En audio crudo es lo primero que hay que calcular, porque **la señal tiene tres o cuatro órdenes de magnitud más de muestras por unidad de contenido semántico**. Una palabra ocupa unos pocos caracteres en texto y unas 8.000 muestras en audio a 16 kHz. Toda la segunda mitad de la clase es una respuesta a ese hecho.

## 1.1 La fórmula general

Sea una pila de $L$ capas, donde la capa $l$ tiene kernel de tamaño $k_l$, stride $s_l$ y dilatación $d_l$. El **campo receptivo** $R_L$ —el número de posiciones de la entrada que influyen sobre una única posición de la salida— es

$$\boxed{\;R_L \;=\; 1 \;+\; \sum_{l=1}^{L} (k_l - 1)\, d_l \prod_{i=1}^{l-1} s_i\;}$$

y el **stride acumulado** (el factor de submuestreo total) es $S_L = \prod_{l=1}^{L} s_l$.

La estructura de la fórmula contiene ya todo el argumento de la clase. Cada capa aporta $(k_l - 1)$ posiciones, y ese aporte se amplifica por dos factores independientes:

- La **dilatación** $d_l$, que separa las tomas del propio filtro. Amplifica **esa** capa.
- El **stride acumulado de las capas anteriores** $\prod_{i<l} s_i$, porque cada posición de entrada a la capa $l$ ya representa $\prod_{i<l} s_i$ posiciones originales. Amplifica **todas las capas siguientes**.

De ahí salen las tres estrategias. Y de ahí sale también la asimetría que decide cuál conviene: **la dilatación es gratis en parámetros y en resolución; el stride es gratis en parámetros pero cuesta resolución; la profundidad cuesta parámetros y cómputo pero no resolución.**

## 1.2 Las tres estrategias, con números

El objetivo de referencia: cubrir **un segundo de audio a 16 kHz**, es decir $R \ge 16\,000$.

| Estrategia | Configuración | Capas necesarias | Qué cuesta |
|---|---|---|---|
| **Profundidad densa** | $k=3$, $s=1$, $d=1$ | $\dfrac{16\,000-1}{2} = 7\,999$ | Absurdo en parámetros y cómputo |
| **Dilatación exponencial** | $k=3$, $s=1$, $d_l = 2^{l-1}$ | $\lceil \log_2(8\,000+1)\rceil = 13$ | Huecos (gridding) |
| **Stride / pooling** | $k=3$, $s=2$ | $\approx 13$ | Resolución temporal de la salida |

El contraste entre 7.999 y 13 es el argumento entero del slide 55, y es correcto. Lo que el slide no dice es que la tercera columna existe: la dilatación no es la única salida, y la familia M del laboratorio —[Dai et al. 2017](/papers/raw-waveforms-dai-2017)— usa la tercera sin tocar la segunda.

Para la dilatación exponencial con kernel $k$ y $d_l = k^{l-1}$ (el caso general del esquema de WaveNet), el campo receptivo es una serie geométrica:

$$R_L = 1 + (k-1)\sum_{l=1}^{L} k^{l-1} = 1 + (k-1)\cdot\frac{k^L - 1}{k - 1} = k^L$$

Es decir, **exactamente $k^L$**: el crecimiento es exponencial en la profundidad y la base es el tamaño del kernel. Con $k=2$ y $L=10$, el campo receptivo es 1.024 muestras. Ese es el bloque de WaveNet.

## 1.3 La condición que evita los huecos

Una convolución dilatada no mira las posiciones intermedias. Si se apilan capas con dilataciones mal elegidas, hay posiciones de la entrada que **ninguna** ruta del grafo consulta: el artefacto conocido como *gridding*. La condición que lo evita es simple de derivar.

Tras la capa $l$, cada posición de salida resume un tramo contiguo de $R_l$ posiciones de la entrada original. La capa $l+1$ toma $k_{l+1}$ de esas posiciones, separadas por $d_{l+1}$. Para que los tramos que resumen dos tomas consecutivas **se toquen o se solapen** —y no dejen un vacío entre medio— hace falta que la separación no supere el ancho de cada tramo:

$$\boxed{\;d_{l+1} \;\le\; R_l\;}$$

Tomar siempre el máximo permitido, $d_{l+1} = R_l$, da el crecimiento más rápido posible **sin dejar huecos**. Con esa elección, cada capa multiplica el campo receptivo por $k_{l+1}$:

$$R_{l+1} = R_l + (k_{l+1}-1)\,R_l = k_{l+1}\, R_l$$

{{< concept-alert type="clave" >}}
**Por qué duplicar es lo correcto en WaveNet y un error en el Ejemplo 2.** Con kernel $k=2$, la regla $d_{l+1} = R_l$ produce $1, 2, 4, 8, \dots$ — exactamente el esquema de WaveNet. Es decir: **la duplicación clásica no es una convención, es el óptimo para kernel 2**, y es la razón por la que WaveNet no sufre gridding dentro de un bloque.

Con kernels grandes, en cambio, duplicar desperdicia casi todo el potencial. Los kernels del Ejemplo 2 del slide son $20, 10, 10, 5$: la regla óptima da $d = 1, 20, 200, 2000$, mientras que duplicar da $d = 1, 2, 4, 8$. La diferencia entre ambos programas es de dos órdenes de magnitud en cobertura.
{{< /concept-alert >}}

## 1.4 El Ejemplo 2 del slide, auditado

El slide 57 especifica 4 capas convolucionales dilatadas 1D con 128, 128, 256 y 256 filtros y kernels $20\times1$, $10\times1$, $10\times1$ y $5\times1$, dejando el factor de dilatación como "depende de la aplicación". Aplicando la fórmula:

| Programa de dilataciones | $R$ (muestras) | @16 kHz | @20 kHz | ¿Huecos? |
|---|---|---|---|---|
| $1,1,1,1$ (sin dilatar) | 42 | 2.6 ms | 2.1 ms | No |
| $1,2,4,8$ (duplicación) | 106 | 6.6 ms | 5.3 ms | No |
| $1,4,16,64$ | 456 | 28.5 ms | 22.8 ms | No |
| $1,8,64,512$ | 2.716 | 169.8 ms | 135.8 ms | No |
| **$1,20,200,2000$ (óptimo)** | **10.000** | **625 ms** | **500 ms** | **No** |
| $1,16,256,4096$ | 18.852 | 1.178 ms | 942 ms | **Sí** |
| $1,2,4,8$ + max-pool 4 tras cada conv | 2.716 | 169.8 ms | 135.8 ms | No |

Dos lecturas:

**La configuración canónica no alcanza.** Con duplicación, esas cuatro capas cubren 6.6 ms. Ni siquiera una ventana de análisis estándar de 25 ms. La promesa del slide 51 —"tras pocas capas las neuronas pueden cubrir miles de timesteps"— no se cumple con este ejemplo bajo la lectura natural de la figura del slide 55.

**Con el programa correcto sí alcanza, y sin huecos.** El programa $1, 20, 200, 2000$ da 625 ms: una ventana perfectamente útil para clasificar un evento sonoro urbano. Y es el máximo alcanzable sin gridding. La arquitectura del slide es viable; lo que falta es la especificación del hiperparámetro que la hace viable, que es justamente el que el slide deja abierto.

Nótese que la penúltima fila —$1,16,256,4096$— cubre más pero **viola la condición**: $d_4 = 4096 > R_3 = 2\,516$. La última capa consulta 5 muestras separadas por un cuarto de segundo cada una. Cubre más territorio y ve menos.

## 1.5 La otra vía: la familia M

[Dai et al. 2017](/papers/raw-waveforms-dai-2017) llega a campos receptivos comparables **sin una sola convolución dilatada**. Su receta: una primera capa con kernel 80 y stride 4 seguida de max-pooling 4 —que reduce la entrada por un factor de 16 de una sola vez— y después kernels de 3 al estilo VGG con pooling intercalado.

| Red | Capas conv | Parámetros | Campo receptivo | @8 kHz |
|---|---|---|---|---|
| M3 | 2 | 221.194 | 172 | 21.5 ms |
| M5 | 4 | 559.114 | 1.772 | 221 ms |
| M11 | 10 | 1.786.442 | 7.052 | 881 ms |
| M18 | 17 | 3.683.786 | 11.980 | **1.50 s** |

El kernel de 80 de la primera capa no es arbitrario: a 8 kHz son exactamente **10 ms**, la escala de la ventana estándar de los MFCC. Los autores observan que lo que aprende esa capa se parece a un banco de filtros — es decir, la red redescubre por gradiente lo que el pipeline clásico ponía a mano.

{{< concept-alert type="recordar" >}}
**Cuándo conviene cada estrategia: la pregunta es si se puede destruir resolución.** WaveNet **no puede** submuestrear, porque su salida es la forma de onda: necesita una predicción por muestra, así que el stride le está vedado y la dilatación es su única herramienta. Un clasificador **sí puede**: su salida es una etiqueta por clip, así que puede colapsar el eje temporal sin pagar nada. Por eso la familia M usa stride agresivo y es barata, y por eso los encoders de wav2vec 2.0 y HuBERT bajan de 16.000 a 50 tramas por segundo antes de que el Transformer vea nada.

La regla: **si la salida es densa en el tiempo, dilata; si la salida es una etiqueta, submuestrea.** Y si es densa pero de menor resolución que la entrada —como el reconocimiento de voz, que emite unos pocos tokens por segundo— submuestrea hasta esa resolución y no más.
{{< /concept-alert >}}

---

# Parte II — La contabilidad del Ejemplo 1

El slide 41 especifica la CLDNN y afirma que la capa de reducción de dimensión *"permite reducir parámetros sin pérdida de exactitud"*. Vale verificar cuánto, porque la respuesta explica el diseño entero.

## 2.1 Las formas, paso a paso

Tomando un contexto de 20 tramas de 40 coeficientes log-mel (una instancia concreta; el paper original procesa trama a trama con contexto asimétrico):

| Operación | Salida | Parámetros |
|---|---|---|
| Entrada | $(1, 1, 20, 40)$ | — |
| Conv $9\times9$, 256 mapas | $(1, 256, 12, 32)$ | 20.992 |
| Max-pool $(1,3)$ — **solo frecuencia** | $(1, 256, 12, 10)$ | 0 |
| Conv $4\times3$, 256 mapas | $(1, 256, 9, 8)$ | 786.688 |
| Aplanado | $(1, 18\,432)$ | — |

Los 18.432 valores del aplanado son el problema. Ese vector es la **entrada** del LSTM, y el costo de un LSTM crece linealmente con el ancho de su entrada:

$$\text{params}(\text{LSTM}) = 4\,\big(d_{\text{in}}\, h + h^2 + h\big) \quad\text{con } h \text{ celdas}$$

## 2.2 Cuánto ahorra la capa de reducción

| Configuración | Ancho al LSTM | Params de la reducción | Params del LSTM (256 celdas) | **Total** |
|---|---|---|---|---|
| Sin reducción | 18.432 | — | 19.138.560 | **19.1M** |
| Capa lineal $18\,432 \to 256$ | 256 | 4.718.848 | 526.336 | **5.2M** |
| Conv $1\times1$, $256 \to 32$ mapas | 2.304 | 8.224 | 2.623.488 | **2.6M** |

El slide tiene razón: sin la capa de reducción, el bloque recurrente solo cuesta 19 millones de parámetros — más que toda la red convolucional que lo precede, y por lejos la pieza más cara. Con reducción baja a 5.2 millones (capa lineal) o 2.6 millones (convolución $1\times1$).

{{< concept-alert type="advertencia" >}}
**Las dos reducciones no son la misma operación, y la del slide no es la del paper.** El slide dice "convolución $1\times1$"; el paper de Sainath dice **capa lineal** que aplana el bloque completo y lo proyecta a 256.

- La **capa lineal** colapsa la grilla tiempo × frecuencia × mapas en un vector. El LSTM recibe 256 números que resumen todo el bloque.
- La **convolución $1\times1$** proyecta canal a canal y **preserva la grilla**: reduce mapas de 256 a 32, pero al aplanar quedan $9 \times 8 \times 32 = 2\,304$ valores, con la estructura tiempo-frecuencia intacta.

En esta instancia la $1\times1$ resulta más barata, así que la sustitución del slide no es un error de eficiencia. Es un cambio de qué se le entrega al LSTM: un resumen global contra una representación estructurada. Ambas son decisiones defendibles; solo hay que saber cuál se está tomando.
{{< /concept-alert >}}

## 2.3 El otro número que el slide cambia

El slide dice "celdas LSTM de 256D"; el paper dice **832 celdas con una proyección de 512**. La proyección —la técnica de Sak et al. 2014— es una matriz que reduce el estado oculto antes de realimentarlo, y existe precisamente para desacoplar la capacidad del costo:

| Configuración | Parámetros de las 2 capas LSTM |
|---|---|
| 2 × LSTM(256 celdas) — el slide | 1.052.672 |
| 2 × LSTM(832 celdas, proyección 512) — el paper | 6.829.056 |
| 2 × LSTM(832 celdas, sin proyección) | 7.255.040 |

La proyección ahorra un 6% frente a no usarla, pero el punto no es ese: es que **el bloque recurrente del paper es 6.5 veces más grande de lo que el slide sugiere**. Alguien que implemente el Ejemplo 1 al pie de la letra obtiene una red con un cuello de botella recurrente mucho más angosto que el original.

---

# Parte III — Por qué un espectrograma no es una imagen

El slide 16 dice que hay "diferencias relevantes" y no las desarrolla. Vale formalizarlas, porque la pregunta tiene una forma precisa: **¿bajo qué grupo de transformaciones la etiqueta es invariante?** Una convolución es equivariante a traslaciones por construcción; el pooling convierte esa equivariancia en invariancia local. Si la etiqueta *no* es invariante a esa transformación, la arquitectura está imponiendo un supuesto falso.

## 3.1 La asimetría de los ejes

Sea $X[f, t]$ un espectrograma. Considérense dos traslaciones:

$$\text{(tiempo)}\quad X'[f,t] = X[f, t-\tau] \qquad\qquad \text{(frecuencia)}\quad X''[f,t] = X[f-\phi, t]$$

En una imagen ambas preservan la etiqueta. En audio:

- $X'$ **preserva** la etiqueta: el mismo sonido, más tarde. Un ladrido sigue siendo un ladrido.
- $X''$ **no** la preserva en general. Sobre un eje de frecuencia lineal, desplazar es un cambio inarmónico que destruye la estructura de la señal. Sobre un eje **logarítmico** —que es el caso del mel en su parte alta y de la escala musical entera— desplazar equivale a **multiplicar todas las frecuencias por una constante**, es decir, a transponer:

$$\log(\alpha f) = \log f + \log \alpha$$

Un desplazamiento constante en el eje log-frecuencia es un cambio de tono. Y si la etiqueta es "nota Do" o "identidad del hablante", ese desplazamiento la cambia. Si la etiqueta es "sirena", no.

{{< concept-alert type="clave" >}}
**Por eso el pooling va solo en frecuencia, y por eso su ventana es pequeña.** Parece contradictorio con lo anterior, pero es exactamente lo que se quiere: la invarianza deseada en frecuencia es **local, no global**. El tracto vocal de cada persona desplaza las formantes de un mismo fonema en una fracción pequeña — el paper de Sainath enmarca la CNN como una **normalización de longitud del tracto vocal (VTLN) aprendida**. Un pooling de 3 bins da invarianza a ese jitter y a nada más. Un pooling de 20 borraría la identidad del sonido.

Y el pooling **no** va en tiempo porque la capa siguiente es un LSTM, cuyo trabajo es justamente modelar la evolución temporal: destruir la resolución temporal antes de entregársela sería quitarle la entrada.
{{< /concept-alert >}}

## 3.2 La localidad rota por los armónicos

Un sonido tonal con frecuencia fundamental $f_0$ tiene energía en $f_0, 2f_0, 3f_0, \dots$ — es decir, en bandas **separadas y periódicas** a lo largo de todo el eje de frecuencia. Un kernel de $9\times9$ sobre un espectrograma de 40 bandas cubre menos de un cuarto del eje: no puede ver, en una sola operación, que las bandas 5, 10, 15 y 20 están correlacionadas.

Hay tres salidas, y cada una es una arquitectura conocida:

| Salida | Quién la usa |
|---|---|
| Profundidad: apilar capas hasta que el campo receptivo cubra el eje | Las CNN de audio clásicas (VGGish) |
| Kernels con forma de dominio: altos y angostos para timbre, largos y bajos para ritmo | [musicnn](/papers/musicnn-pons-2019) |
| Atención global: cada posición ve todo el espectro desde la primera capa | [AST](/papers/ast-gong-2021) |

La tercera es la que resuelve el problema de raíz, y es un argumento a favor de los Transformers en audio que la clase no considera: no es solo que la atención modele bien el tiempo largo, es que **modela bien la frecuencia distribuida**, que es donde la localidad convolucional falla peor.

## 3.3 Suma contra oclusión

Dos objetos visuales que se superponen se **ocluyen**: los píxeles pertenecen a uno o al otro. Dos sonidos simultáneos se **suman**:

$$X_{\text{mezcla}}[f,t] \approx X_1[f,t] + X_2[f,t]$$

(aproximadamente, porque la magnitud del espectrograma descarta la fase y la suma exacta es compleja). Tres consecuencias de diseño:

1. **La separación de fuentes es tratable.** Existe una descomposición aditiva que recuperar; en visión, lo que está detrás del objeto ocluso simplemente no fue medido.
2. **El etiquetado multi-clase es la norma.** De ahí las sigmoides en vez del softmax que el slide 43 menciona al pasar.
3. **La mezcla es una augmentation válida y barata** — sumar dos clips con una SNR controlada produce un ejemplo nuevo con las dos etiquetas. Es el principio de [Scaper](/papers/scaper-salamon-2017) y de mixup aplicado a audio.

---

# Parte IV — El costo del contexto: convolución, recurrencia y atención

La objeción 3 del slide 61 —"los Transformers no son buenos para modelar dependencias largas"— se puede evaluar con la tabla de complejidad de Vaswani et al. (2017), que es de donde salió el argumento original en la dirección contraria.

## 4.1 Las tres métricas

| Capa | Complejidad por capa | Operaciones secuenciales | **Longitud máxima del camino** |
|---|---|---|---|
| Self-attention | $O(T^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrente | $O(T \cdot d^2)$ | $O(T)$ | $O(T)$ |
| Convolución densa, kernel $k$ | $O(k \cdot T \cdot d^2)$ | $O(1)$ | $O(T/k)$ |
| Convolución dilatada | $O(k \cdot T \cdot d^2)$ | $O(1)$ | $O(\log_k T)$ |

La tercera columna es la que responde la objeción. **La longitud del camino entre dos posiciones cualesquiera mide cuántas operaciones tiene que atravesar la señal —y el gradiente— para conectarlas.** En una RNN es $O(T)$: la información del instante 1 al instante 1000 pasa por mil transformaciones sucesivas, cada una con su multiplicación y su no linealidad, y por eso el gradiente se degrada. En self-attention es $O(1)$: una sola operación conecta cualquier par.

Ese fue el argumento de Vaswani para abandonar la recurrencia. La objeción del slide lo invierte.

## 4.2 Lo que sí es un problema: la primera columna

El costo cuadrático es real, y sobre audio es brutal. Con $d = 512$:

| Representación | $T$ | Atención $T^2d$ | Recurrencia $Td^2$ | Razón |
|---|---|---|---|---|
| Onda cruda, 10 s @16 kHz | 160.000 | 13.1 T | 41.9 G | **312×** |
| Onda cruda, 1 s @16 kHz | 16.000 | 131.1 G | 4.2 G | 31× |
| Parches AST, 10 s | 1.212 | 752 M | 318 M | 2.4× |
| Log-mel, 10 s, salto 10 ms | 1.000 | 512 M | 262 M | 2.0× |
| Encoder conv wav2vec 2.0 / HuBERT, 10 s @50 Hz | 500 | 128 M | 131 M | **1.0×** |
| Frase de texto | 30 | 461 K | 7.9 M | 0.1× |

El punto de equilibrio está en $T = d$: por debajo de 512 posiciones, la atención cuesta **menos** que la recurrencia.

{{< concept-alert type="clave" >}}
**La última fila es la que explica la arquitectura de todos los modelos de audio modernos.** El encoder convolucional de wav2vec 2.0 y HuBERT baja de 16.000 muestras por segundo a **50 tramas por segundo** — un factor de 320. A esa tasa, diez segundos de audio son 500 posiciones, y la atención cuesta exactamente lo mismo que una capa recurrente, con camino $O(1)$ en vez de $O(T)$ y sin operaciones secuenciales.

Es decir: **la convolución en los Transformers de audio no está ahí principalmente para "extraer features locales". Está ahí para bajar $T$ hasta el régimen donde la atención es asequible.** Y esa es, cuantificada, la razón por la que la intuición del profesor sobre la complementariedad entre operador local y operador global sigue siendo correcta — aunque la conclusión que extrae de ella no lo sea.

Todos los modelos de audio con Transformer hacen esta reducción: AST con parches solapados de $16\times16$, wav2vec 2.0 y HuBERT con siete capas convolucionales, [Conformer](/papers/conformer-gulati-2020) con un stem de submuestreo, Whisper con dos convoluciones iniciales.
{{< /concept-alert >}}

## 4.3 La evidencia empírica

La comparación más limpia entre recurrencia y atención como operador global está dentro de un solo paper, sobre el mismo dataset y en la misma condición — la Tabla 2 de Conformer, LibriSpeech con modelo de lenguaje:

| Modelo | Operador global | Parámetros | WER test / test-other |
|---|---|---|---|
| LAS | LSTM | 360M | 2.2 / 5.2 |
| Conformer S | self-attention + conv | 10.3M | **2.1 / 5.0** |
| Conformer M | self-attention + conv | 30.7M | 2.0 / 4.3 |
| Conformer L | self-attention + conv | 118.8M | **1.9 / 3.9** |

Treinta y cinco veces menos parámetros y mejor en las dos columnas. Si la atención modelara peor las dependencias largas, este resultado no debería existir en la tarea donde el contexto largo más decide.

---

# Parte V — Causalidad, latencia y el factor 2

Un detalle que la clase no menciona y que cambia por un factor 2 todas las cuentas de la Parte I.

## 5.1 El campo receptivo causal

WaveNet es **autorregresivo**: predice $x_t$ a partir de $x_1, \dots, x_{t-1}$, así que sus convoluciones deben ser **causales** — el filtro solo puede mirar hacia atrás. Se implementa rellenando por la izquierda y recortando el final. Un filtro de tamaño $k$ y dilatación $d$ abarca entonces $(k-1)d$ posiciones **hacia el pasado**, y el campo receptivo total es asimétrico: todo hacia atrás, nada hacia adelante.

Un **clasificador no necesita causalidad**. Si el objetivo es etiquetar un clip completo, el filtro puede mirar en ambas direcciones, y el mismo kernel abarca $\frac{(k-1)d}{2}$ hacia cada lado. El campo receptivo total es idéntico —la fórmula de la Parte I no cambia— pero **la posición de salida queda centrada** en vez de al final del tramo.

La consecuencia práctica: para que una posición de salida vea un evento centrado en ella, un modelo causal necesita el doble de campo receptivo que uno no causal. Cuando se toman prestadas las cifras de WaveNet para una arquitectura de clasificación, esa diferencia es un factor 2 a favor.

## 5.2 El costo real de la causalidad: la latencia

Lo que la causalidad compra no es exactitud sino **operación en tiempo real**. Un modelo no causal necesita el clip entero antes de emitir su primera salida; un modelo causal puede emitir en cuanto llega cada muestra.

Es el mismo compromiso que aparece en el Ejemplo 1: el slide 44 dice que *"la frecuencia de muestreo y la aplicación determinan la ventana de contexto temporal"*. Traducido: cuánto contexto **futuro** se permite es una decisión de latencia, no de exactitud. El paper de CLDNN usa contexto asimétrico —bastante pasado, poco futuro— por esa razón exacta, y los Conformer de streaming usan atención con máscara causal y ventanas limitadas por lo mismo.

| Arquitectura | ¿Causal? | Latencia mínima | Uso |
|---|---|---|---|
| WaveNet | Sí | Una muestra | Generación en línea |
| CLDNN con contexto futuro $n$ | Casi | $n$ tramas | ASR en producción |
| Familia M, VGGish, AST | No | El clip completo | Clasificación offline |
| Conformer streaming | Sí, con ventana | Cientos de ms | ASR en vivo |

---

## Resumen operativo

1. **El campo receptivo se calcula, no se estima.** $R_L = 1 + \sum_l (k_l-1)d_l\prod_{i<l}s_i$. En audio crudo es el primer número que hay que mirar en cualquier arquitectura.
2. **La progresión de dilataciones debe seguir a $d_{l+1} \le R_l$.** Duplicar es el óptimo solo cuando $k=2$. Con kernels grandes, duplicar desperdicia dos órdenes de magnitud de cobertura.
3. **Si la salida es una etiqueta, submuestrea; si es densa, dilata.** WaveNet no puede submuestrear; un clasificador sí, y por eso es más barato.
4. **La capa de reducción antes del bloque recurrente decide el tamaño de la red**: 19M de parámetros contra 3-5M en el Ejemplo 1.
5. **Los ejes del espectrograma no son intercambiables.** Pooling en frecuencia, con ventana pequeña, y nunca en tiempo si viene un modelo de secuencia después.
6. **La atención tiene camino $O(1)$ y costo $O(T^2 d)$.** Sobre audio, la solución universal es bajar $T$ con convoluciones antes de atender — que es la tesis de la complementariedad de la clase, con el operador global actualizado.

---

## Para seguir

- [Teoría](teoria) — el recorrido de las diapositivas.
- [Práctica desde 0](practica) — medir el campo receptivo empíricamente y construir la CLDNN, en PyTorch, TensorFlow y JAX.
- [Convoluciones dilatadas](/fundamentos/convoluciones-dilatadas) — el operador, más allá del audio.
- [Clasificación de audio](/fundamentos/clasificacion-de-audio) — tagging, detección de eventos y etiquetas débiles.
