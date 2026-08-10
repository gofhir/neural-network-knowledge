---
title: "WaveNet: A Generative Model for Raw Audio (2016)"
weight: 426
math: true
---

{{< paper-card
    title="WaveNet: A Generative Model for Raw Audio"
    authors="Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu (DeepMind)"
    year="2016"
    venue="arXiv:1609.03499"
    pdf="/papers/wavenet-oord-2016.pdf" >}}
Hasta 2016 nadie generaba la **forma de onda cruda** con una red neuronal: a 16 kHz, un segundo de habla son 16 000 muestras que un modelo autorregresivo debe emitir una tras otra, y el consenso era que el contexto necesario —cientos de milisegundos, es decir **miles de muestras**— quedaba fuera del alcance tanto de una RNN como de una CNN. WaveNet traslada la receta de **PixelCNN** al audio y resuelve el problema del contexto con dos decisiones: una pila de **convoluciones causales dilatadas**, cuyo campo receptivo crece **exponencialmente** con la profundidad mientras el costo crece solo linealmente ($R = 2^L$ con kernel 2, o sea 1024 muestras con 10 capas), y una salida **softmax categórica de 256 clases** sobre audio comprimido con $\mu$-law, que no impone ninguna forma a una distribución condicional fuertemente multimodal. El resultado desmonta veinte años de síntesis de voz: **MOS 4.21 en inglés y 4.08 en mandarín**, contra 3.86 del mejor concatenativo y 3.79 del mejor paramétrico, cerrando **más del 50% de la brecha** con el habla natural sin ningún post-procesamiento. Como subproducto entrega **18.8 PER en TIMIT** entrenando directamente sobre audio crudo. Es el paper que la [Clase 39](/clases/clase-39) usa como justificación de las [convoluciones dilatadas](/fundamentos/convoluciones-dilatadas), y su costo de inferencia —**una pasada completa de la red por cada muestra**— es también el motivo por el que casi nadie usa hoy este modelo exacto.
{{< /paper-card >}}

---

## Contexto: por qué generar la onda parecía inviable

Un sistema TTS clásico se parte en dos mitades. El **análisis de texto** (normalización, POS tagging, grafema-a-fonema) produce una secuencia de fonemas con contextos lingüísticos. La **síntesis de voz** convierte esa secuencia en una onda, y ahí competían dos escuelas.

La **concatenativa** (*unit selection*, Hunt & Black 1996) pega fragmentos de habla real grabada, eligiendo unidades de una base de datos grande con un costo de objetivo y un costo de concatenación. Como cada trozo es audio genuino, la calidad segmental es excelente. Sus límites son estructurales: exige una base de datos enorme, es rígida —para cambiar la voz, el estilo o la emoción hay que grabar de nuevo— y las junturas producen discontinuidades audibles cuando la base no cubre bien el contexto.

La **paramétrica estadística** (SPSS) extrae del habla una secuencia de parámetros de vocoder $\mathbf{o}$ —uno cada 5 ms: cepstra para el tracto vocal, $F_0$ y aperiodicidad para la fuente glotal—, entrena $\hat{\Lambda} = \arg\max_\Lambda p(\mathbf{o}\mid \mathbf{l},\Lambda)$ dadas las features lingüísticas $\mathbf{l}$, y en síntesis genera $\hat{\mathbf{o}}$ y reconstruye la onda con un vocoder. Es compacta y flexible, pero suena peor: Zen et al. (2009) atribuyeron el déficit a la calidad del vocoder, a la precisión del modelo generativo y al **sobresuavizado**, que es lo que produce ese timbre amortiguado característico.

El diagnóstico de fondo del apéndice —escrito casi con seguridad por Heiga Zen, autor de la mayor parte de esa literatura y de los baselines que aquí se derrotan— es que el enfoque paramétrico es una **optimización en dos pasos y por lo tanto subóptima**: primero se ajusta un modelo generativo de la señal para extraer parámetros, después se modela la trayectoria de esos parámetros con un modelo *distinto*. Y los modelos convencionales de audio arrastran tres supuestos, los tres falsos:

1. **Ventana de análisis de longitud fija.** Se asume estacionariedad dentro de ventanas de 20–30 ms con salto de 5–10 ms. Pero hay fonos —las oclusivas— que duran **menos de 20 ms**. La ventana es más larga que el evento que pretende describir.
2. **Filtro lineal.** Modelos LTI dentro de la ventana, cuando la relación entre muestras sucesivas es fuertemente no lineal.
3. **Proceso gaussiano.** Desde el modelo fuente-filtro, equivale a asumir excitación glotal gaussiana; junto con la linealidad, implica que el habla es normalmente distribuida. No lo es.

La predicción lineal clásica es el caso canónico: $x_t = \sum_{p=1}^{P} a_p\, x_{t-p} + \epsilon_t$ con $\epsilon_t \sim \mathcal{N}(0,G^2)$. Es decir, **un modelo autorregresivo de la onda cruda ya existía desde los años setenta** — lineal, gaussiano, de orden pequeño. WaveNet es literalmente el mismo objeto con las tres restricciones levantadas: no lineal, categórico, con memoria de miles de muestras.

¿Y por qué no se había hecho antes? Por aritmética. El habla de banda ancha necesita al menos **16 000 muestras por segundo**; la música, 44 100. Bajo esa cifra conviven dos problemas distintos que suelen confundirse:

- **El problema de contexto.** Para modelar prosodia hacen falta cientos de milisegundos, es decir **miles de muestras de memoria**. Una RNN sobre 16 000 pasos por segundo implica un horizonte de BPTT brutal; una CNN densa necesitaría profundidad o filtros absurdos. **Este es el problema que WaveNet resuelve**, y lo resuelve con dilatación.
- **El problema de generación.** Las 16 000 predicciones son estrictamente secuenciales. Este problema WaveNet **no lo resuelve**: lo hereda y lo empeora.

La distinción es la clave para leer el paper honestamente. El abstract dice que el modelo "puede ser **entrenado eficientemente** con datos de decenas de miles de muestras por segundo". Dice *entrenado*, no *ejecutado*.

Y sí hubo intentos previos: Tokuda y Zen (2015, 2016) integraron un proceso gaussiano no estacionario de la señal con un generador LSTM-RNN optimizados conjuntamente, y su naturalidad segmental resultó **significativamente peor** que la del modelo no integrado. Ese fracaso es el contexto inmediato: no bastaba con poner una red sobre la onda; había que cambiar la **forma de la distribución de salida**.

## La factorización autorregresiva y la softmax de 256 valores

La factorización es la de siempre en [modelos generativos](/fundamentos/modelos-generativos) autorregresivos, aplicada a la muestra de audio:

$$p(\mathbf{x}) = \prod_{t=1}^{T} p\left(x_t \mid x_1, \dots, x_{t-1}\right)$$

con $T = 16\,000$ para un segundo de habla. No hay variable latente, no hay marginalización aproximada, no hay cota inferior: la verosimilitud es **exacta**, no una ELBO. Y la **factorización es la red misma** — gracias a la causalidad, un solo pase hacia adelante sobre la secuencia completa produce simultáneamente los $T$ factores condicionales, de modo que entrenar es una única pasada de *teacher forcing* con cross-entropy sobre todos los timesteps **en paralelo**. Eso permite ajustar hiperparámetros en validación y detectar sobreajuste sin evaluación subjetiva, que en un dominio cuya métrica real es un MOS con ocho humanos por estímulo es lo que hace posible iterar. Con una salvedad que el propio paper ilustra sin decirla: **la tractabilidad se usa como herramienta interna, no como resultado**. No reporta un solo valor de log-verosimilitud en ninguno de sus cuatro experimentos; todo el argumento empírico descansa en MOS y pruebas de preferencia.

### Por qué categórica y no continua

Esta es la decisión de diseño más importante después de la dilatación. La alternativa natural para una señal continua es una densidad: una *mixture density network* o una mezcla de gaussianas condicionales. El paper elige lo contrario, apoyado en el hallazgo de PixelRNN de que "una distribución softmax tiende a funcionar mejor, **incluso cuando los datos son implícitamente continuos**", con la razón declarada de que una categórica "**no hace ningún supuesto sobre la forma** de la distribución".

Traducido al dominio: la condicional de $x_t$ dado el pasado es **fuertemente multimodal**. En un silencio previo a una oclusiva, la muestra siguiente es casi determinista y cercana a cero; en el inicio de una explosión de /p/ o /t/ salta con signo y amplitud que dependen de detalles no observables; durante un fricativo es esencialmente ruido de banda ancha. Una gaussiana condicional está obligada a resumir todo eso en media y varianza: si la verdadera condicional es bimodal, pone masa en el valle entre modos, que es exactamente donde no hay señal real. **Eso es el sobresuavizado, reformulado como problema de familia paramétrica.** Una mezcla de $K$ componentes lo mitiga pero exige elegir $K$ y sufre colapso de componentes. La categórica de 256 clases es una **mezcla no paramétrica de 256 deltas**: representa bimodalidad, asimetría, colas pesadas y masas puntuales sin ningún hiperparámetro de forma.

{{< concept-alert type="advertencia" >}}
El costo de la decisión es que la softmax **descarta el orden**. Para la pérdida, confundir el nivel 130 con el 131 cuesta exactamente lo mismo que confundirlo con el 5: el modelo debe **aprender la métrica del espacio de amplitudes desde los datos** en vez de recibirla gratis. Es un desperdicio de estructura, y funciona igual — un patrón que se repite en toda la familia PixelCNN. La literatura posterior lo revirtió: PixelCNN++ (2017) y luego Parallel WaveNet reemplazaron la categórica por una **mezcla de logísticas discretizadas**, que recupera el orden ordinal y permite subir a 16 bits sin explotar el número de clases.
{{< /concept-alert >}}

### La cuantización $\mu$-law

El audio crudo típico es PCM lineal de 16 bits: $2^{16} = 65\,536$ valores por muestra. Una softmax de ese tamaño por timestep era inviable en 2016 —la capa de salida domina los parámetros y el cómputo, y la mayoría de las clases se observaría un número irrisorio de veces—. La solución es la **compansión $\mu$-law** (ITU-T G.711) seguida de cuantización uniforme a 256 niveles:

$$f(x_t) = \operatorname{sign}(x_t)\,\frac{\ln\!\left(1 + \mu\,|x_t|\right)}{\ln\!\left(1 + \mu\right)}, \qquad -1 < x_t < 1,\quad \mu = 255$$

Conviene notar que la fórmula del paper describe **solo la compansión**: mapea $[-1,1]$ en $[-1,1]$ de forma monótona, impar y no lineal. La cuantización propiamente tal es el paso siguiente, que el paper deja en prosa:

$$q_t = \operatorname{clip}\!\left(\left\lfloor \frac{f(x_t)+1}{2}\cdot 256 \right\rfloor,\ 0,\ 255\right), \qquad
x = \operatorname{sign}(y)\,\frac{1}{\mu}\left[(1+\mu)^{|y|} - 1\right],\quad y = \frac{2q+1}{256} - 1$$

El valor $\mu = 255$ no es arbitrario: es $2^8 - 1$, el estándar de G.711 para telefonía de 8 bits, y controla la agresividad de la compresión logarítmica. La derivada lo cuantifica:

$$f'(x) = \frac{\mu}{(1+\mu|x|)\,\ln(1+\mu)}$$

Con $\mu = 255$ resulta $\ln 256 = 5.545$, de modo que $f'(0) = 45.99$ y $f'(1) = 0.180$: **la razón es exactamente $1+\mu = 256$**. El paso de cuantización en el dominio original es 256 veces más fino cerca del silencio que cerca de la saturación. En cifras concretas, con paso uniforme $\Delta y = 2/256$ en el dominio comprimido:

| Región | $f'(x)$ | $\Delta x$ en el dominio original | Resolución efectiva |
|---|---|---|---|
| Cerca de cero | 45.99 | $1.70\times10^{-4}$ | ~13.5 bits lineales |
| Mitad de escala ($x=0.5$) | 0.354 | $2.21\times10^{-2}$ | ~6.5 bits |
| Escala completa ($x=1$) | 0.180 | $4.34\times10^{-2}$ | ~5.5 bits |

Con 8 bits de índice, $\mu$-law entrega **~13.5 bits de resolución efectiva donde la señal es débil**. La cuantización lineal de 8 bits daría $\Delta x = 7.8\times10^{-3}$ constante: **46 veces más grueso en la zona de baja amplitud**. Que eso sea lo correcto para audio se sostiene en tres razones convergentes, desarrolladas en el fundamento de [digitalización de audio](/fundamentos/digitalizacion-de-audio): el oído discrimina intensidad de forma aproximadamente logarítmica (ley de Weber), de modo que un paso proporcional a la amplitud produce **SNR de cuantización aproximadamente constante** en todo el rango dinámico; la distribución de amplitudes del habla está fuertemente concentrada cerca de cero, por su alto factor de cresta, y la compansión asigna niveles donde está la masa de probabilidad; y —efecto de segundo orden, específico de este modelo— como la salida es **categórica**, $\mu$-law **ecualiza el uso de las 256 clases**, mientras que con cuantización lineal la mayor parte del vocabulario de salida quedaría prácticamente sin entrenar.

Vale la pena registrar una nota del propio paper que suele omitirse: WaveNet "incorpora casi ningún conocimiento previo sobre las señales de audio, **excepto la elección del campo receptivo y la codificación $\mu$-law**". Los dos únicos sesgos inductivos declarados del modelo son precisamente los dos temas centrales de esta lectura.

## Convoluciones causales

Una convolución es **causal** cuando la salida en el instante $t$ depende solo de entradas en instantes $\le t$. El paper lo formula como garantía: "el modelo no puede violar el orden en el que modelamos los datos: la predicción $p(x_{t+1}\mid x_1,\dots,x_t)$ emitida en el timestep $t$ no puede depender de ninguno de los timesteps futuros".

No es un detalle de notación, es un **contrato**. Si la red que computa $p(x_t \mid x_{<t})$ tuviera acceso a $x_t$, aprendería la identidad, la log-verosimilitud sería casi perfecta y el muestreo produciría basura, porque en generación ese acceso no existe. El fallo es silencioso: la pérdida de entrenamiento **y la de validación** caen a casi cero, así que el diagnóstico habitual no lo detecta; la única señal es que las muestras no suenan a nada.

{{< concept-alert type="advertencia" >}}
La fuga de causalidad es el bug número uno de cualquier WaveNet casera, y ninguna curva de aprendizaje lo revela. El único diagnóstico que funciona es un test explícito: perturbar violentamente la mitad futura de la entrada y verificar que la salida en las posiciones pasadas **no cambia en absoluto**. Si cambia, el padding es simétrico o el recorte está del lado equivocado.
{{< /concept-alert >}}

La implementación es más simple de lo que sugiere el nombre. El paper es explícito y pragmático: para imágenes el equivalente es una **convolución enmascarada**, pero "para datos 1-D como audio uno puede implementar esto más fácilmente **desplazando la salida de una convolución normal unos pocos timesteps**". La forma estándar es el **padding asimétrico**: para kernel $k$ y dilatación $d$, se rellenan $(k-1)\,d$ ceros **solo a la izquierda** y se aplica la convolución sin padding. La salida conserva la longitud de la entrada y cada posición $t$ ve exactamente $\{t-(k-1)d,\dots,t\}$. Un matiz sobre la primera capa: la que toca directamente $x$ debe excluir $x_t$, y la convención es alimentar la red con la señal desplazada un paso, o usar la máscara tipo "A" de PixelRNN en la primera capa y tipo "B" en las superiores. Las capas ocultas *sí* pueden ver su propia posición, porque esa posición ya solo contiene información de $x_{<t}$.

El paper también deja claro que **no hay pooling** en la red generativa y que "la salida del modelo tiene la misma dimensionalidad temporal que la entrada". Eso es lo que permite que un solo pase produzca los $T$ condicionales alineados.

### Por qué no una RNN

El paper lo despacha en una frase —"como los modelos con convoluciones causales no tienen conexiones recurrentes, típicamente son más rápidos de entrenar que las RNN, especialmente en secuencias muy largas"— pero conviene ser preciso sobre en qué régimen aplica la ventaja:

| | Convolución causal | RNN (LSTM/GRU) |
|---|---|---|
| **Entrenamiento** | Todos los timesteps **en paralelo**: un pase, $O(1)$ pasos secuenciales | **Secuencial**: $O(T)$ pasos dependientes, BPTT sobre 16 000 por segundo |
| **Generación** | **Secuencial**, una pasada de *toda la red* por muestra | **Secuencial**, un paso de celda por muestra |
| **Memoria** | Estado explícito y finito: el campo receptivo | Estado latente comprimido de tamaño fijo |
| **Contexto** | Acotado y **declarado por diseño** | Ilimitado en principio, limitado por gradientes en la práctica |
| **Gradiente** | Camino logarítmico en el campo receptivo | Camino de $O(T)$ pasos |

Las dos primeras filas son la asimetría clave. **En entrenamiento la convolución causal es paralela y la RNN no**, porque en *teacher forcing* la ground truth completa es conocida y la convolución solo lee entradas, nunca su propia salida anterior; la RNN, aunque conozca todas las entradas, no puede computar $h_t$ sin haber computado $h_{t-1}$. Sobre 16 000 pasos por segundo, esa diferencia decide si el experimento es factible. **En generación, en cambio, no hay ventaja alguna sino desventaja**: la RNN gasta un paso de celda por muestra, WaveNet gasta **una pasada de decenas de capas**.

La cuarta fila se suele leer al revés. Que el contexto sea **acotado** parece una desventaja frente a la memoria "ilimitada" de una LSTM, pero es lo que hace el diseño analizable: el campo receptivo es un número que se calcula, se elige y se justifica. Y la evidencia del propio paper es que la LSTM no aprovechaba su horizonte teórico —"las capas de convoluciones dilatadas permiten que el campo receptivo crezca de manera **mucho más barata** que usando unidades LSTM"—. El límite de la convolución causal *simple* es lo que motiva la sección siguiente: cuatro capas con kernel 2 dan un campo receptivo de **5 muestras**, es decir **0.3 milisegundos** a 16 kHz.

## Convoluciones dilatadas

Esta es la contribución que sobrevivió a todo lo demás.

Una convolución dilatada —también llamada *à trous*, "con agujeros", término heredado de la literatura de wavelets (Holschneider 1989; Dutilleux 1989)— aplica el filtro sobre un área mayor que su longitud, saltándose entradas con un paso fijo. En su versión causal 1-D, con kernel de longitud $K$ y factor $d$:

$$(x *_d k)[i] \;=\; \sum_{j=0}^{K-1} k[j]\; x\!\left[i - d\cdot j\right]$$

Con $d=1$ se recupera la convolución causal estándar. Dos caracterizaciones equivalentes, ambas útiles: como **filtro inflado con ceros**, el filtro efectivo tiene longitud $1+(K-1)d$ pero solo $K$ pesos no nulos —y la implementación real nunca materializa los ceros, reindexa la entrada—; y como **operación a escala más gruesa**, "similar a pooling o a convoluciones con stride, **pero aquí la salida tiene el mismo tamaño que la entrada**". Esta última cláusula es todo el punto: se gana escala **sin perder resolución temporal**, que es exactamente lo que un modelo autorregresivo muestra a muestra necesita, porque debe emitir una predicción en cada timestep.

### La aritmética del campo receptivo

Para una pila de $L$ capas con kernels $k_l$ y dilataciones $d_l$, sin stride:

$$R \;=\; 1 + \sum_{l=1}^{L} (k_l - 1)\, d_l$$

Con kernel uniforme $k$ y dilataciones que se duplican, $d_l = 2^{l-1}$:

$$R \;=\; 1 + (k-1)\left(2^{L} - 1\right) \quad\xrightarrow{\ k=2\ }\quad \boxed{R = 2^{L}}$$

El campo receptivo es **exponencial en la profundidad**. Diez capas con dilataciones $1,2,4,\dots,512$ dan $R = 1 + (1+2+4+\cdots+512) = 1024$, que es precisamente lo que afirma el paper: cada bloque de ese tipo "tiene un campo receptivo de tamaño 1024, y puede verse como una contraparte más eficiente y discriminativa (no lineal) de una convolución $1\times1024$".

El esquema propuesto duplica la dilatación hasta un límite y luego **reinicia**:

$$1, 2, 4, \dots, 512,\quad 1, 2, 4, \dots, 512,\quad 1, 2, 4, \dots, 512$$

Cada bloque de 10 capas aporta $\sum d = 1023$ muestras, de modo que con $B$ bloques y $k=2$ resulta $R = 1 + B\cdot 1023$:

| Bloques | Capas dilatadas | $R$ (muestras) | $R$ @16 kHz | $R$ @44.1 kHz |
|---|---|---|---|---|
| 1 | 10 | 1 024 | 64.0 ms | 23.2 ms |
| 2 | 20 | 2 047 | 127.9 ms | 46.4 ms |
| 3 | 30 | 3 070 | 191.9 ms | 69.6 ms |
| 4 | 40 | 4 093 | 255.8 ms | 92.8 ms |
| 5 | 50 | 5 116 | 319.8 ms | 116.0 ms |

### Denso contra dilatado: el número que justifica todo

Fijemos un objetivo modesto de $R \ge 1024$ muestras —64 ms a 16 kHz, apenas el orden de un fonema corto— y comparemos las maneras de conseguirlo, contando parámetros con $C$ canales por capa:

| Estrategia | $k$ | Dilataciones | Capas | $R$ | Parámetros | Factor |
|---|---|---|---|---|---|---|
| Filtro único gigante | 1024 | — | 1 | 1024 | $1024\,C^2$ | 51× |
| Pila densa, kernel 2 | 2 | todas 1 | 1023 | 1024 | $2046\,C^2$ | 102× |
| Pila densa, kernel 3 | 3 | todas 1 | 512 | 1025 | $1536\,C^2$ | 77× |
| **Dilatada, kernel 2** | 2 | $1,2,\dots,512$ | **10** | **1024** | $\mathbf{20\,C^2}$ | **1×** |

Y escalando el objetivo a $R\approx 4096$ (256 ms, el orden del campo receptivo real del paper), la brecha se abre todavía más: un filtro único necesitaría $4096\,C^2$ parámetros (171×), una pila densa de kernel 2 necesitaría **4095 capas** y $8190\,C^2$ (341×), mientras que **cuatro bloques dilatados de 10 capas** lo consiguen con 40 capas y $80\,C^2$. Con 30 capas dilatadas WaveNet cubre 3070 muestras; conseguir lo mismo con convolución densa de kernel 2 requeriría **3069 capas**. La relación es exactamente la que corresponde: **crecimiento lineal frente a crecimiento exponencial del campo receptivo con la profundidad**.

Lo que hace que esto no sea un truco contable es que la dilatación **no cuesta nada por capa**. Una convolución dilatada de kernel $k$ tiene exactamente los mismos $k$ pesos que su versión no dilatada y realiza exactamente las mismas multiplicaciones-acumulaciones por posición de salida:

$$\text{Parámetros} = k\,C_{in}C_{out}, \qquad \text{MACs} = T\,k\,C_{in}C_{out}$$

**ambos independientes de $d$**. Esta es la propiedad que hace que "campo receptivo exponencial a costo lineal" sea literalmente cierto y no una aproximación.

### Lo que sí cuesta: gridding y la condición que WaveNet satisface con igualdad

Tres costos reales, en orden de importancia práctica. Primero, **localidad de memoria**: con $d=512$ la capa lee elementos separados por 512 posiciones, destruyendo la localidad de caché; las implementaciones recurren a `space_to_batch`/`batch_to_space` o a kernels especializados, pero el throughput por FLOP es peor que en una convolución densa. Segundo, **huecos en la cobertura** o *gridding*. Tercero, **profundidad no lineal desacoplada del campo receptivo**: un bloque de 10 capas cubre 1024 muestras pero solo aplica 10 no linealidades, así que se puede tener un alcance enorme con un modelo poco expresivo.

El segundo merece desarrollo porque contiene el punto más elegante y menos citado del diseño. Una sola capa con $d=512$ mira dos muestras separadas por 512 y **no ve nada de lo que hay en medio**. Si se apilaran capas dilatadas sin dilataciones pequeñas que las complementen, la red vería un **retículo disperso** de la entrada y sería ciega a la estructura fina en escalas intermedias — el fenómeno que Yu y Koltun (2016) y Wang et al. (2018) documentaron en segmentación semántica como artefactos de tablero de ajedrez. La condición para que una pila dilatada cubra **sin huecos** su campo receptivo es que la dilatación de cada capa no exceda el campo receptivo ya acumulado:

$$d_{l+1} \;\le\; R_l \;=\; 1 + \sum_{i=1}^{l}(k_i-1)\,d_i$$

{{< concept-alert type="clave" >}}
Con $k = 2$ y $d_l = 2^{l-1}$ tenemos $R_l = 2^l$ y $d_{l+1} = 2^{l}$: la condición se satisface **con igualdad exacta en cada capa**. Es decir, el esquema de duplicación con kernel 2 es precisamente **el crecimiento más rápido posible sin dejar agujeros**. Un factor de 3 con kernel 2 ($1,3,9,\dots$) ya lo violaría desde la segunda capa. La estructura resultante es un **árbol binario perfecto**: cada una de las $2^L$ entradas del campo receptivo contribuye a la salida por exactamente un camino, sin redundancia y sin omisión. Por eso WaveNet nunca reportó artefactos de retículo pese a usar dilataciones de 512.
{{< /concept-alert >}}

Queda por explicar el reinicio de bloques. El paper solo dice que "apilar estos bloques aumenta aún más la capacidad del modelo y el tamaño del campo receptivo", pero el reinicio a $d=1$ tiene tres efectos concretos: **reconstruye la cobertura fina**, porque cada bloque nuevo vuelve a mirar vecinos inmediatos y convierte el árbol binario de caminos únicos en un grafo denso multiescala; **triplica la profundidad no lineal** (con 3 bloques) para un crecimiento del campo receptivo que es solo lineal en el número de bloques, que es la palanca correcta —pagar profundidad donde se necesita expresividad, no donde se necesita alcance—; y **estabiliza el entrenamiento**, porque 30 capas con dilatación monótonamente creciente hasta $2^{29}$ serían inútiles mientras que 30 capas en tres bloques residuales son perfectamente entrenables.

## Bloques residuales y activación con compuerta

WaveNet no usa ReLU. Usa la **gated activation unit** heredada de Gated PixelCNN:

$$\mathbf{z} \;=\; \tanh\!\left(W_{f,k} * \mathbf{x}\right) \;\odot\; \sigma\!\left(W_{g,k} * \mathbf{x}\right)$$

donde $*$ es convolución, $\odot$ producto elemento a elemento, y los subíndices $f$ y $g$ denotan *filter* y *gate*. Ambas convoluciones tienen la misma dilatación y la misma entrada; en la práctica se implementan como **una sola convolución con el doble de canales de salida** que luego se parte por la mitad. La evidencia del paper es escueta y no cuantificada: esta no linealidad "funcionaba **significativamente mejor** que la rectificada lineal para modelar señales de audio". Las razones hay que reconstruirlas, y convergen:

1. **Interacción multiplicativa.** ReLU es aditiva y por coordenada: un canal no puede modular a otro dentro de la misma activación. El producto $\tanh(\cdot)\odot\sigma(\cdot)$ introduce una **interacción de segundo orden** entre dos proyecciones distintas de la misma entrada, y permite aprender condicionalidades del tipo "usa este detector de formante **solo si** esta otra proyección indica que hay sonoridad". Con ReLU eso exige capas adicionales.
2. **Es el mecanismo de una LSTM sin la recurrencia.** $\sigma$ produce una compuerta en $[0,1]$ que atenúa contenido acotado en $[-1,1]$. Dado que el paper argumenta contra las RNN por su costo de entrenamiento, es coherente que preserve lo que sí funcionaba de ellas.
3. **Salida acotada y sin unidades muertas.** $z \in [-1,1]$. Con 30 o más capas residuales apiladas y **sin ninguna normalización** —el paper no menciona batch norm ni layer norm—, acotar la activación es un mecanismo de estabilidad barato; ReLU no está acotada por arriba y las sumas residuales acumulan escala. Además, la conmutación abrupta se representa mejor con interacciones multiplicativas, y eso es exactamente lo que hace el habla al pasar de silencio a explosión de oclusiva.

El bloque completo, leído desde la Figura 4: convolución causal inicial no dilatada → $k$ bloques residuales → suma de todas las skip → `ReLU` → `1×1` → `ReLU` → `1×1` → `Softmax`. Y dentro de cada bloque: convolución dilatada, unidad con compuerta, proyección $1\times1$, y dos destinos —**suma residual** con la entrada del bloque, que alimenta al bloque siguiente, y **skip connection** que se suma al bus que va a la salida—.

La división del trabajo entre las dos rutas es el punto interesante, y el paper no lo explica. La **ruta residual** es el camino de la información hacia arriba, y su función es la de siempre en un ResNet: garantizar que el gradiente llegue a las capas bajas sin atenuarse a través de 30 no linealidades acotadas, cuyo jacobiano tiene norma típicamente menor que 1. La **ruta skip** es el camino hacia la salida: cada bloque deposita su contribución directamente en el bus final, de modo que la predicción se computa como una **suma de representaciones de todas las escalas temporales simultáneamente** — el bloque con $d=1$ aporta la microestructura de la onda (fase, amplitud instantánea) y el bloque con $d=512$ aporta contexto de decenas de milisegundos. Sin skip connections, la información de escala fina tendría que sobrevivir el viaje por todos los bloques posteriores, que están dilatados y por lo tanto la comprimirían. **Las skip son lo que evita que el modelo tenga que elegir entre resolución y contexto.** Nótese que son *parametrizadas*: no son atajos identidad, pasan por una proyección $1\times1$ aprendida, y en las implementaciones estándar los canales residuales son pocos —para que la pila sea barata— y los canales skip muchos, para que el bus de salida tenga capacidad.

## Condicionamiento global y local

Sin condicionamiento, WaveNet es un modelo del habla *en general*: genera balbuceo con la fonotáctica y la prosodia correctas del idioma, pero sin contenido. Para que sirva de algo hay que condicionarlo:

$$p(\mathbf{x}\mid \mathbf{h}) = \prod_{t=1}^{T} p\!\left(x_t \mid x_1,\dots,x_{t-1},\mathbf{h}\right)$$

El paper distingue dos modos según la **estructura temporal del condicionante**.

**Condicionamiento global.** Un único vector $\mathbf{h}$ que influye en todos los timesteps; el caso canónico es la identidad del hablante. La activación se modifica añadiendo un sesgo dentro de cada capa:

$$\mathbf{z} \;=\; \tanh\!\left(W_{f,k} * \mathbf{x} + V_{f,k}^{T}\mathbf{h}\right)\;\odot\;\sigma\!\left(W_{g,k} * \mathbf{x} + V_{g,k}^{T}\mathbf{h}\right)$$

donde $V_{*,k}^{T}\mathbf{h}$ se difunde sobre la dimensión temporal. Mecánicamente es un **sesgo aditivo dependiente del condicionante, aplicado antes de las no linealidades**: sesgar el argumento de la sigmoide desplaza el punto de operación de la compuerta, de modo que con el hablante A ciertos canales quedan abiertos y con el hablante B cerrados. La identidad del hablante **reconfigura qué detectores están activos en toda la pila**, no solo reescala la salida — y como se reinyecta en *cada* capa, no es una condición que deba sobrevivir el viaje por la red.

En el experimento sobre VCTK, $\mathbf{h}$ es el one-hot de 109 hablantes, y el resultado es notable: "agregar hablantes resultó en **mejor desempeño en validación** comparado con entrenar sobre un solo hablante. Esto sugiere que la representación interna estaba **compartida entre múltiples hablantes**". Un modelo de 44 horas repartidas entre 109 voces aprende mejor que uno de 24 horas de una sola voz, porque casi toda la estructura de la onda del habla —armónicos, formantes, transiciones— es independiente del hablante y solo un residuo lo distingue. Con un efecto lateral que anticipa la literatura de clonación de voz, incluida la de [SV2TTS](/papers/sv2tts-jia-2018): el modelo "también imitó la acústica y la calidad de grabación, así como la respiración y los movimientos de boca" de los hablantes. **WaveNet modela la señal completa, incluido el canal y los artefactos del micrófono: es un modelo generativo del audio, no de la voz.**

**Condicionamiento local.** Una segunda serie temporal $h_t$ con frecuencia de muestreo mucho menor. Para TTS son las features lingüísticas derivadas cada **5 ms**, o sea a 200 Hz contra las 16 000 muestras por segundo del audio: hay un factor de **80×** que salvar. El mecanismo elegido es una **red convolucional transpuesta** (*upsampling* aprendido) que mapea $\mathbf{h}$ a una serie $\mathbf{y} = f(\mathbf{h})$ a resolución de audio, que luego entra en cada bloque por una convolución $1\times1$:

$$\mathbf{z} \;=\; \tanh\!\left(W_{f,k} * \mathbf{x} + V_{f,k} * \mathbf{y}\right)\;\odot\;\sigma\!\left(W_{g,k} * \mathbf{x} + V_{g,k} * \mathbf{y}\right)$$

La alternativa —repetir los valores de $\mathbf{h}$ a lo largo del tiempo— funcionó "ligeramente peor". La diferencia es el perfil de transición: la repetición produce una señal en escalones, constante durante 80 muestras y luego un salto discontinuo, mientras que la convolución transpuesta aprende **cómo interpolar**. Que la diferencia sea "ligera" es un dato útil de implementación: empezar con repetición no es un error grave.

### El $F_0$: sacar la dependencia larga fuera del modelo

Aquí está el hallazgo experimental más instructivo del paper. Con condicionamiento solo lingüístico —**WaveNet (L)**— el modelo lograba "calidad segmental natural pero a veces prosodia poco natural, acentuando palabras equivocadas en una oración", y el diagnóstico es explícitamente de campo receptivo: "esto podría deberse a la dependencia de largo plazo de los contornos de $F_0$: el tamaño del campo receptivo, **240 milisegundos, no era suficientemente largo**". La solución fue condicionar además sobre el $\log F_0$ predicho por un modelo externo —**WaveNet (L+F)**— y la explicación es limpia: ese modelo externo "corre a una frecuencia mucho menor (**200 Hz**) así que puede aprender dependencias de largo alcance que existen en los contornos de $F_0$".

{{< concept-alert type="clave" >}}
El principio es transferible y vale más que el caso particular: **si el campo receptivo del modelo de forma de onda no alcanza para una dependencia, se saca esa dependencia del modelo y se la entrega precomputada por otro modelo que opere en una escala temporal más gruesa.** 240 ms son 3840 muestras a 16 kHz pero solo 48 pasos a 200 Hz: la misma memoria efectiva cuesta 80 veces menos en la serie lenta. El condicionamiento local es un mecanismo de **descomposición jerárquica de escalas temporales** disfrazado de detalle de implementación.
{{< /concept-alert >}}

### WaveNet es un vocoder, no un TTS

Este es el matiz que más se pierde en las citas de segunda mano. **WaveNet no es un sistema TTS end-to-end.** Las features lingüísticas —identidad de fono, acento silábico, número de sílabas de la palabra, posición de la sílaba en la frase, features de posición de frame y duración— provienen del formato de etiquetas dependientes de contexto de Zen (2006), es decir del **mismo front-end de análisis de texto que alimenta a los sistemas HMM clásicos**, y se asocian al habla cada 5 ms por alineamiento forzado. La **duración de cada fono** la predice un LSTM-RNN externo. El **$\log F_0$** lo predice una CNN autorregresiva externa. Ambos externos se entrenan con MSE.

WaveNet reemplaza exactamente **una** pieza del pipeline clásico: el **vocoder**. Todo lo que está aguas arriba sigue siendo el sistema de producción de Google. Y eso es lo que hace la comparación honesta, no lo que la debilita: como se usaron "los mismos datasets y las mismas features lingüísticas" para los baselines y para WaveNet, la única variable que cambia es el generador de onda, y los 0.35 puntos de MOS son atribuibles a esa pieza y solo a esa pieza.

También explica el camino que tomó la literatura posterior. **Tacotron 2** (2018) reemplazó todo el front-end por un seq2seq con atención que produce un **mel-espectrograma**, y usó una WaveNet condicionada localmente sobre ese mel en lugar de sobre features lingüísticas. La interfaz de condicionamiento local es lo que permitió ese reemplazo sin tocar el vocoder, y "mel-espectrograma en, forma de onda out" sigue siendo la interfaz estándar diez años después.

## Resultados

Cuatro dominios. El paper enumera tres —generación multi-hablante, TTS y música— y añade un cuarto casi de pasada.

**TTS.** Bases de datos mono-hablante de producción de Google: **24.6 horas de inglés norteamericano y 34.8 de mandarín**, ambas de hablantes femeninas profesionales, a 16 kHz. Campo receptivo declarado de **240 ms**. Los baselines son sistemas reales, no reimplementaciones: concatenativo *unit selection* dirigido por HMM (Gonzalvo et al. 2016) y paramétrico basado en LSTM-RNN (Zen et al. 2016). La evaluación es ciega y *crowdsourced* sobre **100 frases no vistas**, con 8 sujetos por estímulo, sujetos pagados y nativos, y un detalle de higiene experimental que en 2016 no era habitual: **se descartó cerca del 40% de las evaluaciones por no haberse usado audífonos**.

| Sistema | Inglés norteamericano | Chino mandarín |
|---|---|---|
| LSTM-RNN paramétrico | 3.67 ± 0.098 | 3.79 ± 0.084 |
| Concatenativo *unit selection* | 3.86 ± 0.137 | 3.47 ± 0.108 |
| **WaveNet (L+F)** | **4.21 ± 0.081** | **4.08 ± 0.085** |
| Natural, 8-bit $\mu$-law | 4.46 ± 0.067 | 4.25 ± 0.082 |
| Natural, 16-bit linear PCM | 4.55 ± 0.075 | 4.21 ± 0.071 |

Son, en palabras del paper, "los valores de MOS más altos jamás reportados con estos datasets y estas frases de prueba". El cierre de brecha con el habla natural va de **0.69 a 0.34 (51%)** en inglés y de **0.42 a 0.13 (69%)** en mandarín, calculado contra el mejor baseline **de cada idioma** — que no es el mismo sistema en los dos casos: en inglés el concatenativo aplasta al paramétrico (3.86 contra 3.67) y en mandarín ocurre lo inverso (3.79 contra 3.47). Y sin ningún post-procesamiento: los sistemas paramétricos de la época dependían de post-filtros de modulación espectral para compensar el sobresuavizado, y WaveNet no necesitó ninguno.

{{< concept-alert type="advertencia" >}}
**El techo real de WaveNet no es 4.55, es 4.46.** Como el modelo genera en el dominio comprimido de 8 bits $\mu$-law, no puede superar por construcción la fila *Natural (8-bit µ-law)*. Su 4.21 está a **0.25 de su techo alcanzable**, no a 0.34 del habla humana: parte de la deficiencia que el paper se atribuye es del esquema de cuantización, no del modelo. El apéndice agrega el dato que hace justa la comparación: los baselines se construyeron desde audio en **16-bit PCM lineal** mientras que WaveNet se entrenó desde **8-bit $\mu$-law**, o sea que compite con una mano atada. Nótese además una anomalía que casi nunca se comenta: en mandarín el audio natural de 8 bits puntuó **4.25**, por encima del PCM de 16 bits (4.21). Los intervalos se solapan, así que es ruido — pero indica que la resolución del instrumento está en el límite de las diferencias que se pretende detectar.
{{< /concept-alert >}}

**Preferencias pareadas.** La tabla completa es más informativa que la figura que suele reproducirse. El resultado más contundente es WaveNet (L+F) **82.0%** contra el LSTM paramétrico **7.6%** en inglés, más de diez a uno. La comparación con el concatenativo es mucho más ajustada: 49.3% contra 20.1%, con **30.6% de indiferencia**; el concatenativo era un rival mucho más duro que el paramétrico, exactamente como sostenía la literatura de TTS. Y hay una fila que desaparece de todos los resúmenes:

| Comparación | LSTM | WaveNet (L) | Sin preferencia | $p$ |
|---|---|---|---|---|
| Mandarín, LSTM vs WaveNet (L) | 25.0 | 23.3 | 51.8 | **0.476** |

Es la **única comparación no significativa de todo el paper**. WaveNet condicionada solo por features lingüísticas **no fue mejor que el baseline paramétrico en mandarín**. Importa porque implica que el condicionamiento por $F_0$ no era un extra opcional sino **necesario para ganar en ese idioma** — lo cual tiene todo el sentido en una lengua tonal, donde el contorno de $F_0$ es contrastivo a nivel léxico y no meramente prosódico.

**Generación multi-hablante libre.** Sobre VCTK (44 horas, 109 hablantes), condicionada solo por identidad de hablante y sin texto, la red genera "palabras inexistentes pero similares a lenguaje humano de manera fluida, con entonaciones que suenan realistas". El valor real de esta sección es el diagnóstico cuantitativo: "la falta de coherencia de largo alcance se debe en parte al tamaño limitado del campo receptivo (**unos 300 milisegundos**), lo que significa que solo puede recordar **los últimos 2–3 fonemas** que produjo". Es el único lugar donde el paper traduce campo receptivo a una unidad lingüística, y es la calibración más útil del documento: **300 ms ≈ 2–3 fonemas**.

**Música.** Sobre MagnaTagATune (~200 horas, clips de 29 segundos con tags de género, instrumentación, tempo y ánimo) y un dataset propio de ~60 horas de piano solo de YouTube. **Sin evaluación cuantitativa alguna**: ni MOS, ni log-verosimilitud, ni estudio con sujetos. Es la sección más débil. Los hallazgos declarados son que "agrandar el campo receptivo era crucial para obtener muestras que sonaran musicales", que las muestras "eran a menudo armónicas y estéticamente agradables incluso producidas por modelos incondicionales", y —lo importante— que **incluso con un campo receptivo de varios segundos los modelos no imponían consistencia de largo alcance**, produciendo variaciones de segundo a segundo en género, instrumentación, volumen y calidad de sonido. El condicionamiento por tags funciona con la misma maquinaria global del embedding de hablante.

**TIMIT: el resultado que se olvida, y el más relevante para la clase.** Es el uso **discriminativo** de la misma arquitectura. El paper motiva el experimento con una afirmación estructural: las LSTM-RNN "han sido un componente clave en estos nuevos pipelines de clasificación de habla porque permiten construir modelos con contextos de largo alcance. Con las WaveNets hemos mostrado que **las capas de convoluciones dilatadas permiten que el campo receptivo crezca de manera mucho más barata que usando unidades LSTM**". Las modificaciones son tres, y las tres importan:

1. **Mean-pooling después de las convoluciones dilatadas**, agregando a frames de **10 ms** — un submuestreo de **160×** a 16 kHz.
2. Tras el pooling, "**unas pocas convoluciones no causales**". La causalidad se abandona explícitamente en cuanto la tarea deja de ser generativa.
3. **Dos términos de pérdida**: predecir la muestra siguiente y clasificar el frame. El modelo "generalizó mejor que con una sola pérdida".

Resultado: **18.8 PER**, calificado con precisión quirúrgica como "el mejor puntaje obtenido por un modelo entrenado **directamente sobre audio crudo** en TIMIT". Es un récord dentro de una categoría restringida, no el estado del arte absoluto de TIMIT, que en 2016 estaba en el rango de 16–18 PER con features log-mel. Ese tercer punto —el objetivo generativo funcionando como regularizador auxiliar del discriminativo— era casi anecdótico en 2016 y es hoy el fundamento del preentrenamiento autosupervisado de wav2vec 2.0 y HuBERT. La línea de clasificación directa sobre la onda que este experimento inaugura es la que recoge, un año después, [Dai et al. con sus CNN muy profundas sobre formas de onda crudas](/papers/raw-waveforms-dai-2017).

## Limitaciones

**La inferencia es secuencial y lentísima.** Domina todas las demás, y el paper no la discute: no aparece la palabra latencia ni tiempo real, ni una sola medición de velocidad de generación. La aritmética es implacable. Generar **un segundo de audio a 16 kHz requiere 16 000 pasadas completas hacia adelante**, cada una recorriendo decenas de capas residuales con su convolución dilatada, su compuerta, sus dos proyecciones $1\times1$ y la softmax final de 256 clases. Las 16 000 pasadas son estrictamente dependientes: **no hay paralelismo temporal disponible en absoluto**. Y el batch de trabajo por pasada es de una sola posición temporal, así que las convoluciones se reducen a productos matriz-vector — una GPU está diseñada para lo contrario, y la utilización es catastrófica.

El orden de magnitud del daño no está en este paper sino en el trabajo que lo resolvió: **Parallel WaveNet** (2018) reporta que el modelo autorregresivo original generaba del orden de **172 muestras por segundo** contra las 16 000 necesarias para tiempo real, es decir unas **90 veces más lento que tiempo real** — producir un segundo de habla tomaba del orden de un minuto y medio. La ironía estructural es completa: **la arquitectura fue elegida por ser paralela en entrenamiento, y es exactamente esa elección la que la hace ineficiente en generación.** Una convolución causal sin recurrencia obliga, en inferencia ingenua, a recomputar todo el campo receptivo en cada paso — trabajo $O(R)$ por muestra donde una RNN gasta $O(1)$. La mitigación estándar (*fast WaveNet inference*, Paine et al. 2016) cachea las activaciones intermedias en colas circulares de longitud $d$ por capa y reduce el trabajo por muestra a $O(L)$; da órdenes de magnitud, pero no cambia la naturaleza secuencial.

**Falta de estructura de largo plazo.** Documentada por el propio paper en tres dominios y siempre atribuida al campo receptivo: 300 ms y "2–3 fonemas" en habla libre; 240 ms y prosodia equivocada en TTS, parchado con el $F_0$ externo; y varios segundos sin consistencia en música. El caso de la música es el diagnóstico más honesto e incómodo: **aumentar el campo receptivo un orden de magnitud no resolvió el problema, solo lo desplazó**. La estructura musical vive en escalas de decenas de segundos —frases, secciones, forma— y ninguna dilatación razonable llega ahí. El paper propone *context stacks* como remedio —una pila separada y más pequeña que procesa un tramo largo y condiciona localmente a una WaveNet grande— pero **no reporta ningún experimento que los use**. La línea posterior demostró que el problema no era de campo receptivo sino de **representación**.

**El paper no publica su arquitectura.** Este punto merece enunciarse sin rodeos, porque afecta a todo lo que se pueda decir del modelo:

{{< concept-alert type="advertencia" >}}
**WaveNet nunca publica el número de capas, el número de bloques, el tamaño de los kernels ni el ancho de los canales residuales, de compuerta o de skip.** La secuencia $1,2,4,\dots,512$ repetida tres veces aparece precedida de un literal "e.g.": es una ilustración del principio de diseño, no la configuración declarada de ningún experimento. Tampoco reporta optimizador, learning rate, schedule, batch, longitud de secuencia, número de pasos, hardware, tiempo de entrenamiento, ninguna log-verosimilitud —pese a argumentar que su tractabilidad es una ventaja central— ni ablación cuantificada alguna: las ventajas del gated activation sobre ReLU ("significativamente mejor") y de la convolución transpuesta sobre la repetición ("ligeramente peor") son afirmaciones sin números.

El único hiperparámetro arquitectónico publicado es el campo receptivo en milisegundos: **240 ms** para TTS y **~300 ms** para el multi-hablante. A 16 kHz eso son 3840 y 4800 muestras, y **ninguna de las dos corresponde a un número entero de bloques de 10 capas con kernel 2** (que dan 1024, 2047, 3070, 4093, 5116). Como la arquitectura no está publicada, no es posible reconciliar las cifras: traducir milisegundos a capas es **reconstrucción, no cita**. Toda reproducción que afirme "el WaveNet de TTS tenía 30 capas" está tomando el número de implementaciones de terceros — y además 30 capas dan 191.9 ms, no 240.
{{< /concept-alert >}}

El contraste con el rigor del protocolo de evaluación subjetiva es violento: hay dos culturas experimentales conviviendo en el mismo documento. Que sea un **preprint de arXiv que nunca pasó por revisión por pares** —uno de los papers más citados de la década que jamás vio un comité de programa— ayuda a explicarlo: no hubo revisores que exigieran esos detalles.

Vale también corregir dos citas erróneas frecuentes. **WaveNet no superó al habla humana**: 4.21 contra 4.55 y 4.46 en inglés, 4.08 contra 4.21 y 4.25 en mandarín. En ningún idioma ni contra ninguna referencia natural lo igualó; el paper solo afirma haber reducido la brecha en más del 50%. Y **el MOS de 4.21 corresponde a WaveNet (L+F)**, no a "WaveNet": la tabla tiene una sola fila del modelo, la condicionada sobre features lingüísticas *y* $\log F_0$, mientras que **WaveNet (L) no tiene MOS reportado en ninguna parte** — solo aparece en las pruebas de preferencia.

## Por qué importa hoy

**El despliegue fue rápido, y no fue este modelo.** En octubre de 2017, poco más de un año después del preprint, DeepMind anunció WaveNet sirviendo tráfico real en el Google Assistant en inglés norteamericano y japonés. Es uno de los ciclos investigación-a-producción más rápidos de la década, y prácticamente el momento en que el público general notó que las voces sintéticas habían dejado de sonar sintéticas. Con dos precisiones que el entusiasmo suele borrar: lo que se desplegó fue **Parallel WaveNet**, porque a ~90× más lento que tiempo real este modelo habría tomado minutos por respuesta; y se subió la calidad de la señal a **24 kHz y 16 bits**, lo que obligó a abandonar la softmax categórica de 256 clases en favor de una mezcla de logísticas discretizadas. La decisión de diseño conceptualmente más elegante del paper fue de las primeras en descartarse al llevarlo a producción.

**La línea que resolvió la velocidad**, en cuatro movimientos que atacan la misma restricción desde ángulos distintos:

| | Paradigma | Paralelo al generar | Necesita maestro | Costo relativo |
|---|---|---|---|---|
| WaveNet (2016) | Autorregresivo, verosimilitud exacta | No | — | 1× (~90× peor que tiempo real) |
| Parallel WaveNet (2018) | IAF + destilación de densidad | Sí | Sí | ~3000× más rápido |
| WaveRNN (2018) | Autorregresivo eficiente + esparsidad | No | No | Tiempo real en CPU móvil |
| WaveGlow (2019) | Flujo normalizante | Sí | No | Tiempo real en GPU |
| HiFi-GAN (2020) | GAN | Sí | No | Cientos de veces tiempo real |

**Parallel WaveNet** entrena un estudiante que es un *Inverse Autoregressive Flow*: parte de ruido logístico y produce todas las muestras en paralelo con $x_t = z_t\,s(z_{<t}) + \mu(z_{<t})$. Un IAF es rapidísimo para **muestrear** y lentísimo para **evaluar** verosimilitud de datos externos — el perfil exactamente opuesto al de WaveNet, y la destilación explota esa complementariedad: el estudiante muestrea en paralelo y el maestro, ya entrenado, puntúa esas muestras minimizando $D_{KL}$; el maestro nunca genera nada. **WaveRNN** ataca la constante en vez del orden: una única GRU con salida factorizada en dos softmax de 8 bits (16 bits de resolución con dos softmax de 256 en lugar de una de 65 536), más poda por esparsidad y *subscaling*; su contribución conceptual es que el cuello de botella era la **profundidad por muestra**, no la autorregresión en sí. **WaveGlow** es la simplificación de ingeniería: un flujo invertible entrenado por máxima verosimilitud, una sola red y una sola pérdida. **HiFi-GAN** abandona la verosimilitud por completo, con un generador convolucional adversarial cuya innovación clave es el *multi-period discriminator*, que evalúa la señal reordenada en periodicidades primas (2, 3, 5, 7, 11) para capturar la estructura periódica del habla; sigue siendo el vocoder de referencia por relación calidad-costo.

**La herencia de la convolución dilatada es el legado más duradero.** WaveNet **no la inventó** —el propio paper la atribuye a Holschneider y Dutilleux en procesamiento de señales, y a Chen et al. y Yu & Koltun en segmentación de imágenes— pero fue el trabajo que la convirtió en **el mecanismo estándar para conseguir campos receptivos grandes en secuencias**. En segmentación semántica el problema es dual al del audio: la *atrous convolution* de **DeepLab** permite quitar los strides y compensar el campo receptivo con dilatación, manteniendo la resolución de píxel, y **ASPP** apila varias ramas dilatadas en paralelo para agregar contexto multiescala. Cronológicamente DeepLab precede a WaveNet y el flujo de influencia fue de imágenes a audio; lo que WaveNet aportó de vuelta fue el **esquema de bloques repetidos con duplicación**, que la comunidad de visión adoptó después para combatir el gridding (*Dilated Residual Networks*, *Hybrid Dilated Convolution*). En traducción automática, **ByteNet** —de Nal Kalchbrenner, coautor de WaveNet, publicado semanas después— aplicó convoluciones causales dilatadas con el mismo argumento de paralelismo, y fue el paso intermedio entre las RNN seq2seq y el Transformer: mismo diagnóstico, distinta solución. Y en series temporales, Bai, Kolter y Koltun tomaron la receta —convoluciones causales dilatadas más bloques residuales, sin compuertas ni salida categórica—, la bautizaron **TCN** y mostraron que igualaba o superaba a LSTM/GRU en un abanico amplio de benchmarks. Por eso la TCN es hoy línea base estándar en pronóstico, detección de anomalías y modelado de señales fisiológicas, con el campo receptivo tratado como hiperparámetro de primera clase y calculado con la misma fórmula de arriba.

**El paradigma probabilístico, en cambio, fue reemplazado dos veces.** Los modelos de **difusión** sobre la onda o el espectrograma —WaveGrad, DiffWave— modelan $p(x)$ como denoising iterativo: en vez de $T=16\,000$ pasos secuenciales usan $N\approx 6$–50 pasos sobre tensores completos, cada uno **paralelo en el tiempo**. Como la generación ya no es autorregresiva, la causalidad deja de ser necesaria — y sin embargo DiffWave conserva la pila de residuales dilatados con activación con compuerta de WaveNet, simplemente **bidireccional**, como red de predicción de ruido. Es la mejor evidencia de que la contribución arquitectónica es más robusta que la probabilística: la estructura sobrevivió, la causalidad no. La otra línea son los **codecs neuronales**: **SoundStream** y **EnCodec** son autoencoders convolucionales con cuantización vectorial residual que comprimen audio a 1.5–24 kbps con calidad superior a Opus, y el cambio conceptual es profundo — en vez de modelar 16 000 muestras por segundo, se modela una secuencia de **tokens discretos a ~50–75 Hz por codebook**, reduciendo la longitud en dos órdenes de magnitud y habilitando un Transformer autorregresivo encima. AudioLM, VALL-E y MusicGen son exactamente eso.

Visto con perspectiva, el arco es nítido y algo irónico. WaveNet mostró que **se podía** modelar audio crudo autorregresivamente y que el resultado superaba a cualquier alternativa. La comunidad tardó unos seis años en concluir que **se podía pero no convenía**: lo correcto es aprender una representación discreta comprimida —con una red que hereda las convoluciones dilatadas de WaveNet— y hacer la autorregresión ahí, donde la secuencia es doscientas veces más corta. Y el argumento de fondo del paper sobrevive intacto en ese traslado: discretizar el audio y modelarlo con una distribución categórica **sigue siendo la respuesta**. Lo único que cambió es el objeto que se discretiza — de la amplitud instantánea a un vector latente que resume 20 ms.

## En la clase 39

La [Clase 39](/clases/clase-39) invoca WaveNet como justificación de las [convoluciones dilatadas](/fundamentos/convoluciones-dilatadas), con esta cadena: el audio crudo exige 15–20 kHz (44.1 kHz para música) → muchísimas muestras por segundo → una arquitectura convolucional necesitaría filtros enormes o una red muy profunda para cubrir contexto suficiente → **la salida son los filtros dilatados**, que tras pocas capas cubren miles de timesteps manteniendo eficiencia computacional. Cada eslabón está respaldado:

| Afirmación de la clase | Respaldo en el paper |
|---|---|
| El audio crudo exige 15–20 kHz | "al menos 16 000 muestras por segundo"; los sistemas se construyeron a 16 kHz |
| Convolución densa → filtros enormes o mucha profundidad | Campo receptivo = capas + longitud del filtro − 1. Cuatro capas $k=2$ → 5 muestras = 0.3 ms |
| La dilatación cubre miles de timesteps con pocas capas | $R = 2^L$ con $k=2$; 30 capas → 3070 muestras, contra 3069 capas densas |
| Manteniendo eficiencia computacional | Parámetros y MACs por capa **independientes de $d$** |

El único matiz que agregaría al slide es que la eficiencia es **de entrenamiento y de campo receptivo por parámetro**, no de generación: ninguna cantidad de dilatación arregla que generar un segundo de audio requiera 16 000 pasadas secuenciales.

### El "Ejemplo 2": los números no dan si se duplica

El **Ejemplo 2** de la clase propone, sobre audio crudo a 15–20 kHz, cuatro capas de convolución 1D dilatada con kernels **20, 10, 10 y 5** (128, 128, 256 y 256 filtros), seguidas de dos LSTM de 256 y dos capas totalmente conectadas de 1024. Aplicando $R = 1 + \sum_l (k_l-1)\,d_l$ a 16 kHz:

| Escenario | $(d_1,d_2,d_3,d_4)$ | $R$ (muestras) | @16 kHz | @20 kHz | Veredicto |
|---|---|---|---|---|---|
| **A** — sin dilatación | 1, 1, 1, 1 | **42** | 2.6 ms | 2.1 ms | Inútil: por debajo de un periodo glotal |
| **B** — duplicación | 1, 2, 4, 8 | **106** | **6.6 ms** | 5.3 ms | Insuficiente: menos de un periodo de voz masculina |
| **C** — cuadruplicación | 1, 4, 16, 64 | **456** | 28.5 ms | 22.8 ms | Marginal: ~3 periodos glotales, media vocal |
| **D** — potencias de 10 | 1, 10, 100, 1000 | **5 010** | 313 ms | 250 ms | Suficiente: escala de 2–3 fonemas |
| **E** — óptimo sin huecos | 1, 20, 200, 2000 | **10 000** | **625 ms** | 500 ms | Amplio: escala de palabra |

Los cálculos, explícitos:

- **A:** $1 + 19 + 9 + 9 + 4 = 42$.
- **B:** $1 + 19(1) + 9(2) + 9(4) + 4(8) = 1 + 19 + 18 + 36 + 32 = 106$.
- **C:** $1 + 19(1) + 9(4) + 9(16) + 4(64) = 456$.
- **D:** $1 + 19(1) + 9(10) + 9(100) + 4(1000) = 5010$.
- **E:** $1 + 19(1) + 9(20) + 9(200) + 4(2000) = 1 + 19 + 180 + 1800 + 8000 = 10\,000$.

{{< concept-alert type="advertencia" >}}
Con el esquema "razonable" en el sentido de WaveNet —dilataciones que se duplican, $1,2,4,8$— la pila del Ejemplo 2 tiene un campo receptivo de **106 muestras, o sea 6.6 milisegundos a 16 kHz**. Eso está a tres órdenes de magnitud de los "miles de timesteps" que promete el slide, y es **acústicamente ciego**: no alcanza a cubrir *un solo* periodo glotal de una voz masculina ($F_0 \approx 100$ Hz, 160 muestras). Una neurona con esa ventana no puede distinguir sonoro de sordo, porque la periodicidad que define esa distinción vive en una escala mayor que su campo receptivo, y no puede estimar $F_0$ en absoluto.
{{< /concept-alert >}}

La razón del desajuste es que la regla de duplicación está calibrada para **kernel 2**, y los kernels del Ejemplo 2 son 20, 10, 10 y 5. Con kernels grandes el crecimiento por capa ya es fuerte y conviene un factor de dilatación mucho mayor. La condición de cobertura sin huecos, $d_{l+1} \le R_l$, da la respuesta de diseño correcta sin adivinar nada:

- $R_1 = 20 \Rightarrow d_2 \le 20$; con $d_2 = 20$: $R_2 = 20 + 9(20) = 200$
- $\Rightarrow d_3 \le 200$; con $d_3 = 200$: $R_3 = 200 + 9(200) = 2000$
- $\Rightarrow d_4 \le 2000$; con $d_4 = 2000$: $R_4 = 2000 + 4(2000) = 10\,000$

El escenario **E** es, por lo tanto, **exactamente el máximo crecimiento sin gridding** para esos kernels: cuatro capas, cero huecos, **10 000 muestras = 625 ms a 16 kHz** (500 ms a 20 kHz). Esa es la elección que corresponde defender para el Ejemplo 2. Y nótese la simetría con WaveNet: la misma regla que aquí produce $1, 20, 200, 2000$ es la que, aplicada a kernels de tamaño 2, produce **exactamente** $1, 2, 4, 8, \dots$ — es decir, la progresión de WaveNet no es una convención feliz sino **el crecimiento máximo posible sin dejar huecos** para su tamaño de kernel.

Vale registrar además una coincidencia didácticamente valiosa: apilar cuatro capas con **stride 2** y sin dilatación produce **el mismo $R = 106$** que el escenario B. La diferencia es que el stride **descarta** resolución temporal —la salida tiene $T/16$ posiciones— mientras la dilatación la conserva. Es literalmente la frase del paper: "similar a pooling o convoluciones con stride, **pero aquí la salida tiene el mismo tamaño que la entrada**".

### Generativo y causal contra discriminativo y no causal

La diferencia más importante entre WaveNet y el Ejemplo 2 no es de dilataciones sino de **régimen**.

WaveNet **debe** ser causal porque la factorización autorregresiva se rompe si el modelo ve el futuro: es un contrato con la distribución que está modelando. El Ejemplo 2 es un **clasificador**: se le entrega una ventana de audio completa y se le pide una etiqueta. No hay ningún orden de generación que respetar, y **nada obliga a que el filtro mire solo hacia atrás**. El propio WaveNet abandona la causalidad en cuanto cambia de tarea — en el experimento de TIMIT, el pooling va "seguido de unas pocas convoluciones **no causales**".

La consecuencia hay que enunciarla con precisión, porque la formulación coloquial ("no ser causal duplica el campo receptivo") es imprecisa. El número total de muestras que influye en una salida **es el mismo**, $R = 1 + \sum(k_l-1)d_l$, en ambos casos. Lo que cambia es **dónde está la ventana** respecto del punto de interés:

| | Convolución causal | Convolución centrada (no causal) |
|---|---|---|
| Ventana que ve la salida en $t$ | $[\,t-R+1,\; t\,]$ | $[\,t-\tfrac{R-1}{2},\; t+\tfrac{R-1}{2}\,]$ |
| Contexto pasado | $R-1$ muestras | $\tfrac{R-1}{2}$ muestras |
| Contexto futuro | **0** | $\tfrac{R-1}{2}$ muestras |

El mismo presupuesto de kernel se reparte hacia ambos lados. Para obtener $W$ muestras de contexto **simétrico** basta $R = 2W+1$ centrado; un modelo causal necesitaría $R = 4W+1$ para tener $W$ de futuro además de $W$ de pasado, y aun así no puede conseguirlo por construcción, porque nunca mira adelante. El factor 2 es real, pero está en la **utilidad** de la ventana, no en su tamaño.

Y acústicamente importa mucho: para decidir si un instante pertenece a una oclusiva hay que ver **la explosión que viene después**, no solo el silencio previo; para clasificar una transición de formante hay que ver a dónde va. La coarticulación es bidireccional — un fono se ve afectado tanto por el que lo precede como por el que lo sigue. Un modelo causal es estructuralmente peor para clasificación de habla, y por eso todos los reconocedores serios usan contexto bidireccional (BiLSTM, convoluciones centradas, atención completa) salvo cuando la latencia de streaming lo prohíbe. **Con el escenario E y filtros centrados, cada salida del Ejemplo 2 vería ±312 ms alrededor de su posición**: cubre la sílaba con holgura y llega al orden de palabra. Es un front-end acústico perfectamente razonable.

Dos corolarios de ingeniería. Primero, en el Ejemplo 2 la pila dilatada **no es todo el modelo**: hay dos LSTM después, lo que redistribuye el trabajo — las convoluciones construyen una representación local rica por frame (periodicidad, estructura formántica, transitorios) y las LSTM aportan el contexto largo sobre una secuencia mucho más corta. Eso hace del Ejemplo 2, estrictamente, una **CLDNN** con front-end convolucional dilatado, no una WaveNet discriminativa — y es la arquitectura de Sainath et al. (2015) que el propio WaveNet cita como parte del movimiento hacia el audio crudo. Segundo, y es el detalle que decide si el modelo entrena o no: para que las LSTM sean viables, la secuencia que reciben **no puede tener 16 000 pasos por segundo**. Tiene que haber stride o pooling en la pila convolucional. WaveNet lo hizo exactamente así en su experimento discriminativo —**mean-pooling a frames de 10 ms, submuestreo de 160×**, dejando 100 frames por segundo—, y ese es el orden de magnitud al que hay que llegar.

## Notas y enlaces

- La [Clase 39](/clases/clase-39) usa este paper como justificación de las [convoluciones dilatadas](/fundamentos/convoluciones-dilatadas); la [profundización](/clases/clase-39/profundizacion) desarrolla la aritmética del campo receptivo, la condición $d_{l+1}\le R_l$ y el cálculo completo del Ejemplo 2 bajo los cinco esquemas de dilatación.
- La compansión $\mu$-law, el teorema de muestreo y la relación entre bits, rango dinámico y SNR de cuantización están en [digitalización de audio](/fundamentos/digitalizacion-de-audio); la factorización autorregresiva y su lugar frente a VAEs, GANs y difusión, en [modelos generativos](/fundamentos/modelos-generativos).
- El experimento de TIMIT es la puerta de entrada al uso **discriminativo** de la onda cruda, que [Dai et al. (2017)](/papers/raw-waveforms-dai-2017) llevan a CNN muy profundas sin ninguna maquinaria generativa. El condicionamiento **global** por identidad de hablante, y la observación de que un solo modelo cubre 109 voces mejor que 109 modelos separados, es el germen directo de [SV2TTS](/papers/sv2tts-jia-2018) y de la clonación de voz por embedding.
- El recorrido completo del [dominio audio](/dominios/audio) sitúa a WaveNet como el punto de quiebre entre el vocoder paramétrico y el vocoder neuronal, y traza la línea posterior hacia Parallel WaveNet, WaveRNN, WaveGlow y HiFi-GAN.
- Al citar el paper conviene sostener tres precisiones: WaveNet **no superó al habla humana** (4.21 contra 4.55 y 4.46); el MOS corresponde a **WaveNet (L+F)**, que depende de un modelo externo de $F_0$; y **no es un TTS end-to-end** sino un vocoder neuronal condicionado, que hereda el front-end lingüístico, las duraciones y el contorno de pitch de tres modelos externos.
- Para señales fisiológicas y series clínicas la transferencia es directa: la TCN es el descendiente civil de esta arquitectura, y su campo receptivo se calcula con la misma fórmula. Antes de entrenar conviene hacer el cálculo al revés —qué escala temporal tiene el fenómeno que se quiere detectar, cuántas muestras son a la frecuencia de muestreo disponible— y elegir el programa de dilataciones que la cubra sin huecos. Es un ejercicio de cinco minutos que evita entrenar durante días un modelo estructuralmente ciego al fenómeno.
