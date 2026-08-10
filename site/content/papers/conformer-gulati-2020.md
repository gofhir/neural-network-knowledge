---
title: "Conformer: Convolution-augmented Transformer for Speech Recognition (2020)"
weight: 433
math: true
---

{{< paper-card
    title="Conformer: Convolution-augmented Transformer for Speech Recognition"
    authors="Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang (Google)"
    year="2020"
    venue="Interspeech 2020 / arXiv:2005.08100"
    pdf="/papers/conformer-gulati-2020.pdf" >}}
El paper parte de una observación sobre dos familias que estaban ganando el ASR *end-to-end* por caminos opuestos, y que fallaban en cosas distintas. Los **Transformers** modelan bien el contexto global —camino de longitud $O(1)$ entre cualquier par de posiciones— pero, en palabras del propio paper, *"are less capable to extract fine-grained local feature patterns"*: sin sesgo posicional, la self-attention trata la secuencia como una bolsa y no tiene noción privilegiada de "la trama de al lado", que es justo donde viven los eventos acústicos que definen un fonema. Las **CNN** tienen esa localidad gratis, pero su campo receptivo crece linealmente con la profundidad, de modo que necesitan muchas capas o muchos parámetros para alcanzar contexto global. La propuesta de Conformer es combinarlas **dentro del mismo bloque** —no en ramas paralelas, no en etapas separadas— con una estructura *macaron*: media capa feed-forward, self-attention multi-cabeza con **codificación posicional relativa** de Transformer-XL, un **módulo de convolución** con GLU y convolución depthwise, otra media capa feed-forward, y layernorm. La afirmación fuerte no es "combinar mejora" sino que combinar es **más eficiente en parámetros**, porque cada operador deja de pagar con capacidad lo que el otro ya resuelve estructuralmente. Los números lo respaldan sobre LibriSpeech: **2.1% / 4.3% de WER sin modelo de lenguaje y 1.9% / 3.9% con él** en test-clean/test-other, y un modelo mediano de **30.7M parámetros que supera al Transformer Transducer de 139M**. El bloque Conformer se volvió, en un par de años, el encoder acústico por defecto del ASR moderno.
{{< /paper-card >}}

---

## Contexto: el ASR en 2019-2020, con tres familias y tres debilidades

En 2019-2020 el ASR *end-to-end* estaba repartido en tres familias, y cada una fallaba en algo distinto. Cada pieza del bloque Conformer es una respuesta a una de esas debilidades.

**Las RNN eran la opción por defecto.** El paper lo dice sin adornos en su primera línea: *"Recurrent neural networks (RNNs) have been the de-facto choice for ASR as they can model the temporal dependencies in the audio sequences effectively"*. La referencia clave es Graves 2012, que introduce el **RNN-T**: la pérdida que permite entrenar un transductor de secuencias sin alineamiento previo y que hace posible el *streaming*. La línea de Google —Chiu et al. 2018, He et al. 2019, Sainath et al. 2020— era encoder LSTM + *prediction network* LSTM + RNN-T, corriendo en teléfonos: descendencia directa de la [CLDNN de 2015](/papers/cldnn-sainath-2015).

El problema de esta familia no era la calidad sino algo estructural, y son dos cosas distintas que conviene no confundir. Primero, **el entrenamiento no paraleliza sobre el tiempo**: una [LSTM](/fundamentos/lstm-gru) necesita $O(n)$ operaciones estrictamente secuenciales para procesar $n$ tramas, y con audio a 100 tramas por segundo, una *utterance* de 10 segundos son 1000 pasos que no se pueden solapar —ni con más GPUs ni con más memoria, porque es una dependencia de datos. Segundo, **el camino entre dos posiciones distantes es $O(n)$**: la información entre la trama 10 y la trama 900 atraviesa 890 aplicaciones de la celda recurrente, y las compuertas mitigan el desvanecimiento del gradiente pero no lo eliminan.

**Los Transformers llegaron y ganaron.** El Speech-Transformer (Dong et al., 2018) y sobre todo el **Transformer Transducer** (Zhang et al., 2020) mostraron que reemplazar el encoder LSTM por un encoder [Transformer](/fundamentos/transformer) mejoraba. El paper resume la razón en dos propiedades: *"its ability to capture long distance interactions and the high training efficiency"* —camino $O(1)$ entre cualquier par de posiciones, y todo el eje temporal procesado en paralelo dentro de una capa.

Lo que les faltaba está en la primera frase del segundo párrafo: *"While Transformers are good at modeling long-range global context, they are less capable to extract fine-grained local feature patterns."* La [self-attention](/fundamentos/self-attention) es un operador de agregación ponderada por contenido **sobre un conjunto**: sin sesgo inductivo posicional, no distingue estructuralmente al vecino inmediato del vecino a 300 tramas. En audio eso duele, porque los eventos acústicos que definen un fonema —una explosión, una transición formántica, un *onset*— viven en ventanas de 20 a 100 milisegundos y son **estructura local con desplazamiento arbitrario**. Aprender ese detector desde atención pura significa gastar capacidad reaprendiendo la equivarianza traslacional que una convolución trae gratis.

**Las CNN también funcionaban:** Jasper, QuartzNet, ContextNet, y antes Sainath et al. 2013 y Abdel-Hamid et al. 2014. Su mecanismo, según el paper, *"capture local context progressively via a local receptive field layer by layer"*. El campo receptivo de una pila de $L$ convoluciones de kernel $k$ crece **linealmente**, $\approx L(k-1)+1$: para cubrir 250 tramas con kernel 31 hacen falta unas 8 capas; para cubrir 2500, unas 83. De ahí la cita: *"One limitation of using local connectivity is that you need many more layers or parameters to capture global information."*

**El intento de arreglarlo desde el lado CNN, y por qué no basta.** El rival directo es **ContextNet**, del mismo grupo de Google, publicado apenas 13 días antes, que inserta un módulo *squeeze-and-excitation* en cada bloque residual para inyectar contexto largo. La crítica de Conformer es el argumento que sostiene todo el paper:

> *"However, it is still limited in capturing **dynamic** global context as it only applies a **global averaging** over the entire sequence."*

*Squeeze-and-excitation* comprime toda la secuencia a **un solo vector** por promedio, lo pasa por un MLP y reescala canales. Es contexto global, sí, pero con resolución posicional nula: **todas las posiciones reciben la misma modulación**. La self-attention calcula un contexto **distinto para cada posición**, condicionado a lo que esa posición está preguntando. Esa es la diferencia entre "global" y "dinámicamente global", y es lo que justifica por qué la atención no es reemplazable por un truco barato de *pooling*.

**El trabajo concurrente.** Bello et al. 2019 (*Attention Augmented CNNs*) ya había mostrado en visión que combinar convolución y self-attention supera a usarlas por separado. Y sobre todo **Lite Transformer** (Wu et al., 2020), que parte la entrada en **dos ramas paralelas** —una de atención, una convolucional— y concatena las salidas. Conformer se declara *"inspired by Wu et al."* pero toma la decisión opuesta en el punto clave: **secuencial en vez de paralelo**, y lo justifica con un ablation.

## La tesis: local y global no son sustitutos, son complementos

La formulación del paper:

> *"We hypothesize that both global and local interactions are important for being parameter efficient. [...] self-attention learns the global interaction whilst the convolutions efficiently capture the relative-offset-based local correlations."*

Hay dos afirmaciones ahí y conviene separarlas, porque solo una es interesante.

**La afirmación débil: combinar ayuda.** Trivialmente cierta y ya demostrada por Bello et al.

**La afirmación fuerte, que es la que sostiene el paper: combinar es *más eficiente en parámetros*.** No dice "combinar da mejor WER a igual costo"; dice que la combinación es lo que permite llegar al mismo WER con **menos capacidad**. Eso sí lo demuestran los resultados: 30.7M contra 139M, 10.3M contra 360M.

El argumento mecánico es que cada operador tiene un sesgo inductivo que le sale gratis y que el otro tendría que **comprar con parámetros**:

| | Sesgo inductivo que trae gratis | Lo que tiene que comprar con parámetros |
|---|---|---|
| **Convolución** | equivarianza traslacional, localidad, pesos compartidos sobre el eje temporal | contexto global: crece $O(L)$ en profundidad |
| **Self-attention** | camino $O(1)$ a cualquier posición, mezcla condicionada al contenido | localidad y noción de vecindad: hay que aprenderlas desde datos |

{{< concept-alert type="clave" >}}
La eficiencia en parámetros no viene de que dos módulos "sumen capacidad". Viene de que **cada uno deja de pagar por lo que el otro ya resuelve estructuralmente**. Ese es todo el paper en una frase.
{{< /concept-alert >}}

## El bloque Conformer: la estructura *macaron*

El encoder completo es:

```
audio (tramas cada 10 ms)
  → SpecAugment
  → Convolution Subsampling      [10 ms → 40 ms, es decir 100 fps → 25 fps]
  → Linear
  → Dropout
  → N × Conformer Block
```

y cada bloque es la estructura **macaron**: media FFN → self-attention multi-cabeza → módulo de convolución → media FFN → layernorm. Las ecuaciones del paper, para la entrada $x_i$ al bloque $i$:

$$\tilde{x}_i = x_i + \frac{1}{2}\,\mathrm{FFN}(x_i)$$

$$x_i' = \tilde{x}_i + \mathrm{MHSA}(\tilde{x}_i)$$

$$x_i'' = x_i' + \mathrm{Conv}(x_i')$$

$$y_i = \mathrm{Layernorm}\!\left(x_i'' + \frac{1}{2}\,\mathrm{FFN}(x_i'')\right)$$

Cuatro detalles que se pasan por alto al leer rápido:

**(a) Solo las FFN llevan el factor $\tfrac{1}{2}$.** Los residuales de la atención y de la convolución son de paso completo. El medio paso es una propiedad específica del **par** macaron, no una convención general del bloque. Es uno de los errores de reimplementación más comunes.

**(b) Todos los módulos son *pre-norm*.** El paper lo dice para la atención (*"We use pre-norm residual units with dropout"*) y para las FFN, y el módulo de convolución también arranca con LayerNorm. La normalización va **dentro** de cada rama residual, y la ruta identidad queda limpia de extremo a extremo. Eso es lo que permite entrenar 17 capas con 10k pasos de *warm-up* sin que el entrenamiento explote.

**(c) Pero además hay un LayerNorm *post* al final del bloque.** Esto es inusual. Un Transformer pre-norm canónico pone un único LayerNorm al final de toda la pila; Conformer pone uno **por bloque**, después del último residual. En la práctica reescala la salida de cada bloque a norma controlada antes de entregarla al siguiente, lo que evita la deriva de escala típica del pre-norm profundo. Es un híbrido pre-norm/post-norm y el paper no lo argumenta: solo lo dibuja y lo escribe en la ecuación.

**(d) El dropout va en la salida de cada módulo, antes del residual**, con tasa $P_{drop} = 0.1$.

### De dónde sale el medio paso

La estructura macaron viene de **Macaron-Net** (Lu et al., 2019, *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View*), y su argumento original no es empírico sino numérico. La idea es leer una capa Transformer como un **paso de integración numérica** de un sistema dinámico de múltiples partículas, donde la self-attention hace de término de **difusión** (interacción entre partículas) y la FFN de término de **convección** (evolución individual). Un bloque Transformer estándar, que aplica primero atención y después FFN, corresponde a una separación de operadores de **Lie-Trotter**, con error local de **primer orden**. La alternativa clásica en análisis numérico es la separación de **Strang-Marchuk**: medio paso del primer operador, un paso completo del segundo, medio paso del primero. Esa es de **segundo orden**.

Aplicado al Transformer, Strang-Marchuk da exactamente media FFN → atención → media FFN. De ahí el nombre "macarón" (dos galletas iguales alrededor de un relleno), y de ahí los factores $\tfrac{1}{2}$: **no son un hiperparámetro, son la constante que sale del esquema de integración**. Conformer toma la estructura pero mete **dos** operadores en el relleno: atención *y* convolución.

### Por qué ese orden y no otro

Que la atención vaya **antes** que la convolución no es arbitrario: sale de un barrido (cifras en dev, sin modelo de lenguaje):

| Arquitectura | dev-clean | dev-other |
|---|---|---|
| **Conformer (MHSA → Conv)** | **1.9** | **4.4** |
| Convolución antes de MHSA | 1.9 | 4.5 |
| MHSA y convolución en paralelo, salidas concatenadas | 2.0 | 4.9 |
| Depthwise conv reemplazada por *lightweight conv* | 2.0 | 4.8 |

Conclusión del paper: *"convolution module stacked after the self-attention module works best for speech recognition"*. La lectura mecánica es que la atención produce una representación **ya contextualizada globalmente**, y la convolución después refina localmente **sobre ese contexto**; al revés, la atención tendría que operar sobre features locales crudos. La diferencia es chica (0.1 en dev-other) pero consistente.

Lo más informativo de esa tabla es la fila del paralelo: **0.5 puntos peor en dev-other**. Esa es precisamente la configuración de Lite Transformer, el trabajo que Conformer cita como inspiración. La razón plausible es que en paralelo cada rama ve la mitad de los canales y las salidas se combinan por concatenación —una operación lineal—, mientras que en serie ambos módulos ven el ancho completo y la composición es no lineal. Vale la pena retener esta fila: Branchformer la revirtió dos años después.

### Dónde vive el presupuesto de parámetros

El paper no lo desglosa. Estimando para dimensión $d$, expansión FFN de 4 y kernel $k$:

| Módulo | Parámetros (aprox., sin sesgos ni normalizaciones) | Con $d=512$, $k=32$ |
|---|---|---|
| FFN × 2 | $2 \times (d \cdot 4d + 4d \cdot d) = 16d^2$ | 4.19 M |
| MHSA (con posición relativa: $W_q, W_k, W_v, W_o$ + $W_{k,R}$) | $\approx 5d^2$ | 1.31 M |
| Convolución (pointwise $d\!\to\!2d$, depthwise $k$, pointwise $d\!\to\!d$) | $3d^2 + kd$ | 0.80 M |
| **Total por bloque** | $\approx 24d^2 + kd$ | **6.30 M** |

Multiplicado por las 17 capas del modelo grande da unos 107M; sumando decoder y *subsampling* se llega al orden de los 118.8M reportados, lo que valida el desglose.

El dato interesante: **las dos medias FFN son unos dos tercios de los parámetros del bloque, y el módulo de convolución apenas el 13%**. Y sin embargo, cuando el ablation quita la convolución, el WER en dev-other se degrada 0.4 puntos. Es la contribución con mejor relación WER/parámetro del bloque, por lejos. Es literalmente la tesis de la eficiencia en parámetros convertida en número.

## El módulo de convolución, capa por capa

```
x → LayerNorm
  → Pointwise Conv (d → 2d)
  → GLU
  → Depthwise Conv 1D (kernel 32)
  → BatchNorm
  → Swish
  → Pointwise Conv (d → d)
  → Dropout
  → (+ residual)
```

**LayerNorm de entrada.** Es el pre-norm del residual: estabiliza la escala antes de que empiece la expansión de canales.

**Pointwise conv con factor de expansión 2.** Una convolución $1\times1$ sobre el eje de canales, es decir una capa lineal aplicada posición a posición. Duplica los canales de $d$ a $2d$, y su única función es preparar la entrada del GLU, que consume dos mitades.

**GLU (Gated Linear Unit).** De Dauphin et al., ICML 2017. Parte el tensor en dos mitades $a, b \in \mathbb{R}^{d}$ y calcula

$$\mathrm{GLU}(a, b) = a \odot \sigma(b)$$

volviendo a $d$ canales. Lo que aporta es una **interacción multiplicativa condicionada a los datos**: la rama $b$ decide, canal por canal y posición por posición, cuánto de la rama $a$ pasa. Es un mecanismo de compuerta, hermano de las *gates* de una LSTM pero sin recurrencia y sin estado. Su valor es doble: le da al módulo la capacidad de **suprimir selectivamente** canales o instantes irrelevantes *antes* de la convolución temporal —silencio, ruido, regiones sin contenido fonético—, algo que una no linealidad puntual como ReLU no puede hacer; y mantiene una **ruta lineal**, porque $a$ se multiplica sin pasar por una saturación, lo que preserva el gradiente.

**Depthwise conv 1D.** Aquí está el corazón del módulo y la razón de que sea barato. Una convolución 1D **completa** con kernel $k$ sobre $d$ canales de entrada y $d$ de salida cuesta

$$P_{\text{full}} = k \cdot d^2$$

Una **depthwise** aplica un filtro independiente de largo $k$ a cada canal, sin mezclar canales:

$$P_{\text{dw}} = k \cdot d$$

La razón es exactamente $d$. Con $d = 512$ y $k = 32$: **8 388 608 parámetros contra 16 384**. Un factor de 512, y el costo en FLOPs baja en la misma proporción. La mezcla entre canales, que la depthwise sacrifica, la restituyen las dos pointwise que la rodean: es la factorización de Xception/MobileNet, trasladada al eje temporal —separar "mezclar en el tiempo" de "mezclar entre canales".

{{< concept-alert type="clave" >}}
Sin la factorización depthwise el módulo no cabría en el presupuesto. Con $k=32$ y $d=512$, una convolución completa costaría 8.4M de parámetros **por bloque**, más que las dos FFN juntas, y el argumento de eficiencia en parámetros del paper se caería entero. **La depthwise es lo que hace viable la tesis, no un detalle de implementación.**
{{< /concept-alert >}}

**BatchNorm.** El paper la justifica en una línea: *"Batchnorm is deployed just after the convolution to aid training deep models."* Nótese la asimetría: el resto del bloque usa LayerNorm y solo aquí aparece BatchNorm, una herencia del mundo CNN. Es también la decisión más frágil del diseño, porque calcula estadísticas sobre el eje de *batch* **y** el eje temporal: en un lote de *utterances* de largo variable mezcla tramas reales con tramas de *padding* salvo que se enmascare con cuidado, y en *streaming* introduce discrepancia entre entrenamiento (estadísticas del lote) e inferencia (estadísticas acumuladas). Por eso muchas implementaciones posteriores la reemplazan: `torchaudio` expone un flag `use_group_norm` para cambiarla por GroupNorm.

**Swish.** $\mathrm{Swish}(x) = x \cdot \sigma(\beta x)$; con $\beta = 1$ es SiLU. El paper reporta que *"using swish activations led to faster convergence in the Conformer models"*: habla de **velocidad de convergencia**, no de WER final. El ablation lo confirma —cambiar Swish por ReLU deja dev-clean y dev-other idénticos, mejora test-clean una décima y empeora test-other dos. Es ruido. Swish está ahí por optimización, no por calidad.

**Pointwise conv de salida y dropout.** Proyecta de vuelta a $d$ y regulariza antes del residual.

### El tamaño de kernel, y qué revela su barrido

Los tres modelos usan **kernel 32**. El barrido se hizo sobre el modelo grande, con el mismo kernel en todas las capas, y es la única tabla del paper con dos decimales:

| Kernel | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| 3 | 1.88 | 4.41 | 1.99 | 4.39 |
| 7 | 1.88 | 4.30 | 2.02 | 4.44 |
| 17 | 1.87 | 4.31 | 2.04 | 4.38 |
| **32** | **1.83** | **4.30** | 2.03 | **4.29** |
| 65 | 1.89 | 4.47 | 1.98 | 4.46 |

Tres lecturas:

1. **Hay un óptimo interior.** El rendimiento mejora hasta 17-32 y **empeora en 65**. Un kernel de 65 tramas, a 40 ms por trama, abarca **2.6 segundos**. Eso ya no es "contexto local": es contexto de frase, territorio de la atención. La convolución con kernel gigante paga parámetros y campo receptivo para hacer peor el trabajo que el módulo de al lado hace mejor. La existencia de este óptimo interior es evidencia empírica **directa** de la tesis de complementariedad: si local y global fueran sustitutos, el WER debería mejorar monótonamente con el kernel.
2. **El rango útil es amplio y la ganancia es chica.** De kernel 3 a kernel 32 la mejora en test-other es 4.39 → 4.29: una décima. Comparado con el 0.4-0.6 que cuesta **quitar el módulo entero**, la conclusión es que lo que importa es *tener* convolución local, no afinar su alcance.
3. **La selección se hizo en dev, correctamente.** El paper explica: *"On comparing the second decimal in dev WER, we find kernel size 32 to perform better than rest."* Y hace falta el segundo decimal porque kernel 7 y kernel 32 empatan en dev-other a 4.30; desempata dev-clean (1.88 vs 1.83). En test-clean, kernel 32 (2.03) es de hecho **peor** que kernel 65 (1.98) y kernel 3 (1.99): elegir mirando test habría dado otra respuesta, y por eso está bien no haberlo hecho.

## Self-attention con embedding posicional relativo

El módulo de atención es corto —LayerNorm → multi-head attention → dropout → residual— y lo único no estándar es el esquema posicional. Pero es donde el paper es más económico y, a la vez, más importante:

> *"We employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL, the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention module to generalize better on different input length and the resulting encoder is more robust to the variance of the utterance length."*

### Qué reemplaza exactamente

En Vaswani et al. 2017 la posición entra **sumada a la entrada**: $x_i \leftarrow e_i + p_i$, con $p_i$ una sinusoide de la **posición absoluta** $i$. Expandiendo el score de atención antes del softmax:

$$A_{ij}^{\text{abs}} = (e_i + p_i)^\top W_q^\top W_k (e_j + p_j) = \underbrace{e_i^\top W_q^\top W_k e_j}_{\text{contenido-contenido}} + \underbrace{e_i^\top W_q^\top W_k p_j}_{\text{contenido-posición}} + \underbrace{p_i^\top W_q^\top W_k e_j}_{\text{posición-contenido}} + \underbrace{p_i^\top W_q^\top W_k p_j}_{\text{posición-posición}}$$

Transformer-XL (Dai et al., 2019) reescribe esto sustituyendo cada aparición de la posición absoluta de la clave $p_j$ por una sinusoide de la **distancia relativa** $R_{i-j}$, y las apariciones de la posición absoluta de la consulta $p_i$ por vectores de sesgo **globales aprendidos** $u$ y $v$:

$$A_{ij}^{\text{rel}} = \underbrace{e_i^\top W_q^\top W_{k,E}\, e_j}_{\text{(a) contenido-contenido}} + \underbrace{e_i^\top W_q^\top W_{k,R}\, R_{i-j}}_{\text{(b) sesgo posicional según contenido}} + \underbrace{u^\top W_{k,E}\, e_j}_{\text{(c) sesgo de contenido global}} + \underbrace{v^\top W_{k,R}\, R_{i-j}}_{\text{(d) sesgo posicional global}}$$

con dos matrices de proyección de clave separadas, $W_{k,E}$ para contenido y $W_{k,R}$ para posición. El costo extra es una matriz $d \times d$ más dos vectores por cabeza.

### Por qué esto importa específicamente en audio

Aquí está el argumento que convierte a Conformer en una arquitectura **de audio** y no en un Transformer genérico al que le pegaron una convolución.

**La propiedad clave es la equivarianza traslacional del patrón de atención.** Con codificación relativa, el score entre dos tramas depende solo de su contenido y de la **distancia** entre ellas, no de dónde caen dentro de la *utterance*. Desplazar la señal completa 200 ms a la derecha no cambia la matriz de atención. Con codificación absoluta sí la cambia: el mismo fonema en la trama 50 y en la trama 500 recibe modulaciones posicionales completamente distintas, y el modelo debe aprender por separado a reconocerlo en cada rango de posiciones. Es exactamente la propiedad que hace útiles a las convoluciones, importada al operador global.

**Y el audio castiga la falta de esa propiedad más que el texto**, por tres razones:

1. **Variabilidad extrema de longitud.** LibriSpeech tiene *utterances* de 1 a 35 segundos; a 25 tramas/s tras el submuestreo, eso va de 25 a 875 posiciones: un rango de 35×, algo que una frase de texto rara vez alcanza en tokens. Con codificación absoluta, un modelo entrenado mayoritariamente con *utterances* de 8 segundos ve posiciones mayores a 200 pocas veces y su representación de ellas queda mal estimada; en inferencia, una *utterance* larga cae fuera de distribución en el eje posicional.
2. **No hay unidad natural de segmentación.** En NLP la oración tiene fronteras y la posición 0 significa algo ("inicio de oración"). En audio, el corte de una *utterance* es un artefacto del detector de actividad vocal, del turno de conversación o del formato del dataset: **la posición absoluta 0 no codifica ninguna información lingüística estable**, es donde el segmentador decidió cortar. Al depender solo de $i-j$, el encoder queda **invariante a dónde se puso el corte**.
3. **La estructura acústica relevante es relativa por naturaleza.** "La trama de hace 3 pasos", "el *onset* 200 ms atrás", "la vocal anterior a esta consonante": las relaciones que importan son desplazamientos, no coordenadas.

El precio es real —la codificación relativa impide precomputar la matriz posicional una sola vez, requiere el truco de *relative shift* para no materializar un tensor $T \times T \times d$, y agrega términos al kernel de atención—, pero el ablation dice que vale la pena: quitarla es el paso **más caro** de toda la tabla de degradación.

## Los tres tamaños

El método de selección es explícito: *"found via sweeping different combinations and choosing the best performing models within the parameter limits"*. Es decir, se fijaron presupuestos de 10M, 30M y 118M y se buscó dentro de cada uno.

| | **Conformer (S)** | **Conformer (M)** | **Conformer (L)** |
|---|---|---|---|
| Parámetros (M) | 10.3 | 30.7 | 118.8 |
| Capas del encoder | 16 | 16 | 17 |
| Dimensión del encoder | 144 | 256 | 512 |
| Cabezas de atención | 4 | 4 | 8 |
| Tamaño de kernel de convolución | 32 | 32 | 32 |
| Capas del decoder | 1 | 1 | 1 |
| Dimensión del decoder | 320 | 640 | 640 |

Lo que se lee entre líneas:

**El escalamiento es casi puramente en ancho.** De S a M la profundidad no cambia (16 capas) y solo sube la dimensión de 144 a 256; de M a L sube una sola capa y se duplica la dimensión. Como los parámetros por bloque escalan $\approx 24d^2$, duplicar $d$ cuadruplica el bloque: $16 \times 24 \times 144^2 \approx 8.0$M, $16 \times 24 \times 256^2 \approx 25.2$M, $17 \times 24 \times 512^2 \approx 107$M, lo que reproduce bien la progresión 10.3 / 30.7 / 118.8 una vez sumados decoder y *subsampling*. **Ancho, no profundidad.** Es coherente con que la atención global ya resuelva el alcance temporal: no hace falta profundidad para agrandar el campo receptivo, como sí la necesitaría una CNN pura.

**El kernel es constante en 32 para los tres tamaños**, aunque el barrido se hizo **solo sobre el modelo grande**. Es una extrapolación no verificada: nada garantiza que 32 sea óptimo con $d=144$.

**Las cabezas: 4 / 4 / 8.** Dimensión por cabeza: $144/4 = 36$, $256/4 = 64$, $512/8 = 64$. El modelo chico se queda con cabezas anchas en relación a su tamaño. Y aquí hay una inconsistencia con el ablation, que discutimos más abajo.

**El decoder es una LSTM de una sola capa** en los tres modelos. Este es el detalle irónico del paper y merece un párrafo. Conformer **no elimina la recurrencia del sistema**: la elimina del **encoder acústico** y la conserva en la *prediction network* del transductor RNN-T. Y tiene todo el sentido: esa red modela la secuencia de **etiquetas** ya emitidas, que es corta (decenas de *word-pieces*, no cientos de tramas), estrictamente autorregresiva, y donde la latencia por token pesa más que el paralelismo.

{{< concept-alert type="recordar" >}}
La conclusión honesta de leer la tabla de configuraciones no es "las RNN murieron", sino **"las RNN dejaron de ser el operador correcto para modelar la señal acústica"**. Es una afirmación más precisa y mucho más defendible.
{{< /concept-alert >}}

## Resultados

### Setup

- **Datos:** LibriSpeech, descrito por el paper como *"970 hours of labeled speech"*, más un corpus de texto de 800M tokens para el modelo de lenguaje.
- **Features:** filterbanks de 80 canales, ventana de 25 ms, salto de 10 ms → 100 tramas por segundo a la entrada.
- **Aumentación:** [SpecAugment](/papers/specaugment-park-2019) con $F=27$, **diez** máscaras temporales y ratio máximo de máscara $p_S = 0.05$ —máscaras **proporcionales** a la longitud, la variante para datasets grandes.
- **Regularización:** dropout $P_{drop}=0.1$ en cada unidad residual, ruido variacional sobre los pesos y regularización $\ell_2$ con peso $10^{-6}$.
- **Optimización:** Adam con $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$; *schedule* de Transformer con 10k pasos de *warm-up* y learning rate pico $0.05/\sqrt{d}$ (para el modelo grande, $\approx 2.2\times10^{-3}$).
- **Modelo de lenguaje:** LSTM de 3 capas y ancho 4096, con perplejidad a nivel de palabra de 63.9 sobre dev, integrada por *shallow fusion*.
- **Framework:** Lingvo.

### LibriSpeech, tabla completa

WER en porcentaje; menor es mejor. El paper advierte que *"all our evaluation results round up to 1 digit after decimal point"*, de ahí que los resultados de terceros aparezcan con dos decimales y los de Google con uno.

| Familia | Método | Params (M) | Sin LM: test-clean | Sin LM: test-other | Con LM: test-clean | Con LM: test-other |
|---|---|---|---|---|---|---|
| Hybrid | Transformer (Wang et al.) | – | – | – | 2.26 | 4.85 |
| CTC | QuartzNet | 19 | 3.90 | 11.28 | 2.69 | 7.25 |
| LAS | Transformer (Synnaeve et al.) | 270 | 2.89 | 6.98 | 2.33 | 5.17 |
| LAS | Transformer (Karita et al.) | – | 2.2 | 5.6 | 2.6 | 5.7 |
| LAS | LSTM | 360 | 2.6 | 6.0 | 2.2 | 5.2 |
| Transducer | Transformer Transducer | 139 | 2.4 | 5.6 | 2.0 | 4.6 |
| Transducer | ContextNet (S) | 10.8 | 2.9 | 7.0 | 2.3 | 5.5 |
| Transducer | ContextNet (M) | 31.4 | 2.4 | 5.4 | 2.0 | 4.5 |
| Transducer | ContextNet (L) | 112.7 | 2.1 | 4.6 | 1.9 | 4.1 |
| **Transducer** | **Conformer (S)** | **10.3** | **2.7** | **6.3** | **2.1** | **5.0** |
| **Transducer** | **Conformer (M)** | **30.7** | **2.3** | **5.0** | **2.0** | **4.3** |
| **Transducer** | **Conformer (L)** | **118.8** | **2.1** | **4.3** | **1.9** | **3.9** |

### Las comparaciones que valen

**(a) Conformer M contra el Transformer Transducer: el dato central.** El paper lo afirma en la introducción: *"Our medium 30M parameters-sized model already outperforms transformer transducer published in [7] which uses 139M model parameters."*

| | Params | sin LM clean | sin LM other | con LM clean | con LM other |
|---|---|---|---|---|---|
| Transformer Transducer | 139 M | 2.4 | 5.6 | 2.0 | 4.6 |
| Conformer (M) | 30.7 M | **2.3** | **5.0** | 2.0 (empate) | **4.3** |

La afirmación es **correcta en 3 de las 4 columnas**, y en test-clean con modelo de lenguaje hay **empate exacto** (2.0 vs 2.0), no victoria. Conviene enunciarla así y no como un barrido de las cuatro. El margen sólido está en test-other: **5.6 → 5.0 sin LM, un 10.7% relativo con 4.5 veces menos parámetros**. Ese es el dato contundente del paper y el que hay que citar.

**(b) Conformer S contra ContextNet S.** 10.3M contra 10.8M, y 7.0 → 6.3 en test-other sin LM: 0.7 absoluto, 10% relativo. Con LM, Conformer S gana también en clean (2.1 vs 2.3) y en other (5.0 vs 5.5).

**(c) Conformer L contra ContextNet L: la comparación más dura, y la más reveladora.**

| | Params | sin LM clean | sin LM other | con LM clean | con LM other |
|---|---|---|---|---|---|
| ContextNet (L) | 112.7 M | 2.1 | 4.6 | 1.9 | 4.1 |
| Conformer (L) | 118.8 M | 2.1 | **4.3** | 1.9 | **3.9** |

**En test-clean empatan exactamente**, en las dos condiciones, y Conformer L usa 6M de parámetros **más**. Toda la ventaja está en test-other, el *split* difícil, con locutores y grabaciones de peor calidad. El paper no destaca esto, y es importante para no sobrevender el resultado: contra una CNN bien diseñada con *squeeze-and-excitation*, la ventaja de la atención **no aparece en habla limpia y fácil**; aparece cuando la señal local es ambigua y hay que desambiguar con contexto largo. Que es, mecánicamente, exactamente lo que uno esperaría. Es una confirmación bonita de la tesis, no una debilidad.

**(d) La comparación más limpia entre recurrencia y atención.**

| | Operador global | Params | test-clean (con LM) | test-other (con LM) |
|---|---|---|---|---|
| LAS con encoder LSTM | recurrencia | 360 M | 2.2 | 5.2 |
| **Conformer (S)** | self-attention + convolución | **10.3 M** | **2.1** | **5.0** |

Mismo dataset, misma métrica, misma condición, misma tabla del mismo paper. Un encoder basado en atención con **35 veces menos parámetros** le gana a la mejor arquitectura LSTM reportada, en ambos *splits*. Volveremos a este número al final.

## Ablations: la sección más valiosa del paper

Aquí está el verdadero contenido científico, y el resultado principal es más matizado de lo que dice el texto.

### Desmontar el Conformer hacia un Transformer

El diseño experimental es el correcto: se parte del bloque Conformer y se le quitan diferencias una por una hasta llegar a un bloque Transformer estándar, **manteniendo constante el número total de parámetros**. Es **acumulativo** —cada fila incluye todas las remociones anteriores— y todo se evalúa **sin modelo de lenguaje externo**.

| Arquitectura | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| Conformer | 1.9 | 4.4 | 2.1 | 4.3 |
| – Swish + ReLU | 1.9 | 4.4 | 2.0 | 4.5 |
| &nbsp;&nbsp;– Módulo de convolución | 2.1 | 4.8 | 2.1 | 4.9 |
| &nbsp;&nbsp;&nbsp;&nbsp;– Macaron FFN | 2.1 | 5.1 | 2.1 | 5.0 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;– Embedding posicional relativo | 2.3 | 5.8 | 2.4 | 5.6 |

Los deltas paso a paso, que el paper no tabula:

| Paso | Δ dev-clean | Δ dev-other | Δ test-clean | Δ test-other |
|---|---|---|---|---|
| Swish → ReLU | 0.0 | **0.0** | −0.1 | +0.2 |
| Quitar la convolución | +0.2 | **+0.4** | +0.1 | +0.4 |
| Macaron → FFN única | 0.0 | **+0.3** | 0.0 | +0.1 |
| Posición relativa → absoluta | +0.2 | **+0.7** | +0.3 | +0.6 |
| **Total Conformer → Transformer** | **+0.4** | **+1.4** | **+0.3** | **+1.3** |

**El resultado global:** llegar del bloque Conformer al bloque Transformer estándar, a igual número de parámetros, cuesta **1.4 puntos de WER en dev-other (4.4 → 5.8, un 32% relativo)** y 1.3 en test-other. Con los parámetros congelados. Ese es el valor de la arquitectura, aislado del tamaño, y es una de las ablaciones más limpias que se ven en un paper de arquitectura.

### La discrepancia sobre cuál componente es el más importante

El paper concluye: *"Among all differences, convolution sub-block is the most important feature, while having a Macaron-style FFN pair is also more effective than a single FFN of the same number of parameters."*

Pero **la tabla no dice eso si se miden los deltas**. El paso más caro es el último: quitar el embedding posicional relativo cuesta **+0.7 en dev-other**, casi el doble que quitar la convolución (+0.4). Ordenando por daño:

1. Embedding posicional relativo: **+0.7** dev-other / +0.6 test-other
2. Módulo de convolución: **+0.4** / +0.4
3. Macaron FFN: +0.3 / +0.1
4. Swish: 0.0 / +0.2 (ruido)

{{< concept-alert type="advertencia" >}}
Hay que presentar **las dos lecturas**. A favor del paper: la ablación es **acumulativa**, así que el efecto de la codificación relativa se mide sobre un modelo que **ya perdió la convolución y el macaron**; es plausible que en el Conformer completo su remoción cueste menos, porque el módulo de convolución también aporta información posicional relativa —una convolución *es* un operador de desplazamiento relativo— y cuando se quita uno, el otro se vuelve más crítico. En contra: el paper **no corre el ablation en el otro orden**, así que la afirmación literal "la convolución es la característica más importante" **no está sostenida por los datos publicados**. Lo defendible es: *la convolución y la codificación posicional relativa son los dos componentes cuya remoción más daña, y juntos explican +1.1 de los +1.4 de degradación total en dev-other.*
{{< /concept-alert >}}

Y hay una lectura más profunda que unifica ambos componentes: **los dos cuya remoción más duele son los dos que inyectan estructura *relativa* en el modelo.** La convolución la inyecta como operador (pesos compartidos sobre desplazamientos); la codificación posicional relativa la inyecta en el kernel de atención. El Transformer vanilla no tiene ninguno de los dos. La conclusión del paper, reformulada: **lo que le falta a un Transformer para hacer ASR no es capacidad, es equivarianza traslacional**, y se la puedes dar por dos vías complementarias.

### El número de cabezas de atención

Sobre el modelo grande ($d = 512$), mismo número de cabezas en todas las capas:

| Cabezas | Dim. por cabeza | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|---|
| 4 | 128 | 1.9 | 4.6 | 2.0 | 4.5 |
| 8 | 64 | 1.9 | 4.4 | 2.1 | **4.3** |
| 16 | 32 | 2.0 | **4.3** | 2.2 | 4.4 |
| 32 | 16 | 1.9 | 4.4 | 2.1 | 4.5 |

El paper afirma que *"increasing attention heads up to 16 improves the accuracy, especially over the dev-other datasets"*. Pero el efecto es **pequeño y no monótono**: todo el rango 8-32 cae entre 4.3 y 4.4 en dev-other, y solo 4 cabezas es claramente peor (4.6). La lectura razonable es que **hay un mínimo de cabezas necesario para el modelo grande y a partir de ahí da bastante igual**. Con 4 cabezas y dimensión 128 por cabeza, el modelo tiene pocas "vistas" independientes de la secuencia; con 32 y dimensión 16, cada cabeza es demasiado angosta para representar una consulta útil. El óptimo plano entre medias es el patrón habitual.

### Macaron contra FFN única

| Arquitectura | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| **Conformer (macaron, medio paso)** | 1.9 | **4.4** | 2.1 | **4.3** |
| Macaron con residuales de paso completo | 1.9 | 4.5 | 2.1 | 4.5 |
| Una sola FFN (estilo Transformer) | 1.9 | 4.5 | 2.1 | 4.5 |

Traducción honesta: **0.1 en dev-other y 0.2 en test-other**, la contribución más chica de todas las que ablaciona el paper. El propio texto se pasa al describirla como *"provides a significant improvement"*: en 0.2 de WER absoluto, "significativo" es generoso cuando no hay barras de error, y el paper no reporta ninguna.

Dos cosas interesantes sí salen de ahí. Primero, **el medio paso importa tanto como el par**: "residuales de paso completo" y "una sola FFN" dan exactamente lo mismo (4.5/4.5), o sea que partir la FFN en dos **sin** el factor $\tfrac{1}{2}$ no aporta nada. Toda la ganancia está en la combinación par + medio paso, tal como predice Strang-Marchuk. Segundo, **hay interacción con la convolución**: quitar el macaron desde el Conformer completo cuesta +0.1 en dev-other, pero quitarlo **después** de haber quitado ya la convolución cuesta +0.3. Vale tres veces más cuando la convolución no está —plausiblemente porque el par de FFN, al distribuir capacidad no lineal a ambos lados del bloque, compensa en parte lo que aportaba la convolución. El paper no comenta esta interacción.

### Cómo combinar convolución y atención

Recapitulando el orden de mérito en dev-other de la tabla ya vista: Conformer (4.4) < convolución antes de atención (4.5) < *lightweight conv* (4.8) < paralelo (4.9). Dos conclusiones:

- **La depthwise conv no es reemplazable por *lightweight conv*.** La *lightweight convolution* de Wu et al. 2019 es una depthwise con pesos compartidos entre grupos de canales y **normalizados por softmax sobre el eje del kernel**. Perder 0.4 en dev-other al cambiar una por otra sugiere que esa normalización —que fuerza pesos positivos que suman 1, convirtiendo el filtro en un promedio ponderado— es una restricción demasiado fuerte. Una convolución de audio necesita pesos **negativos** para actuar como detector diferencial de *onsets* y transiciones.
- **El paralelo pierde.** Es la refutación experimental de Lite Transformer en el dominio ASR, hecha por el paper que dice inspirarse en él.

## Limitaciones

El paper no tiene sección de limitaciones —es un Interspeech de cinco páginas—, así que estas salen de leerlo críticamente.

### El costo cuadrático, cuantificado

La atención es $O(T^2)$ en tiempo y memoria. Con los números del paper se puede ver **exactamente** cuán grave es, y la respuesta es interesante: menos de lo que se teme en cómputo, más de lo que se teme en memoria.

El pipeline entrega filterbanks cada 10 ms (100 tramas/s), y el *convolution subsampling* lleva la tasa de 10 ms a **40 ms**: un submuestreo de 4× que deja **25 tramas por segundo**. Entonces:

| Duración de la *utterance* | Tramas $T$ tras submuestreo | Entradas de la matriz de atención $T^2$ |
|---|---|---|
| 2 s | 50 | 2 500 |
| 10 s | 250 | 62 500 |
| 30 s | 750 | 562 500 |
| 60 s | 1 500 | 2 250 000 |

**Sin el submuestreo 4×, esos números se multiplicarían por 16.** El *stem* convolucional no es un detalle de ingeniería: es lo que hace tratable la atención sobre audio, y es por sí mismo una segunda instancia de "usar convolución para lo local".

**En cómputo, el término cuadrático no es el cuello de botella.** El costo de atención por capa es $\approx 2T^2 d$ MACs; el de las partes lineales del bloque, $\approx T \cdot P_{\text{bloque}}$ con $P_{\text{bloque}} \approx 6.3$M para $d = 512$. La atención domina cuando

$$2T^2 d > T \cdot P_{\text{bloque}} \quad\Longleftrightarrow\quad T > \frac{P_{\text{bloque}}}{2d} = \frac{6.3\times10^6}{1024} \approx 6150 \text{ tramas} \approx 4 \text{ minutos}$$

A 10 segundos ($T = 250$), la atención es del orden del 4% del cómputo del encoder. Esto matiza el discurso habitual: en ASR, el problema del $O(T^2)$ no es el tiempo.

**La memoria sí es el problema.** Los pesos de atención hay que guardarlos para el *backward*. Para Conformer L, con 8 cabezas y 17 capas, una *utterance* de 30 s ($T = 750$):

$$750^2 \times 8 \times 17 = 76.5\times10^6 \text{ escalares} \approx 306\ \text{MB en fp32, por muestra del lote}$$

Multiplicado por el tamaño del lote, eso es lo que fuerza a truncar *utterances*, ordenarlas por longitud, usar *gradient checkpointing* o pasar a atención por *chunks*. Y crece cuadráticamente: a 60 s serían ~1.2 GB por muestra. A esto se suma un tercer costo que el paper no cuantifica: la codificación posicional relativa agrega el término $QR^\top$ y el truco de *relative shift*, lo que aumenta tanto el tiempo como la memoria del bloque de atención respecto a la atención absoluta.

### Conformer, tal como está publicado, no es causal ni *streamable*

Esto es central y el paper **ni lo menciona**. Es un modelo *full-context*, offline, pensado para transcribir *utterances* completas. Hay dos fuentes de no causalidad, ambas cuantificables:

**(a) La self-attention es bidireccional.** Cada trama atiende a toda la *utterance*, pasada y futura, sin máscara causal y sin límite de contexto derecho. Un modelo así no puede emitir nada hasta que la *utterance* terminó.

**(b) La convolución depthwise es centrada.** Kernel 32 con *padding* simétrico implica unas 15-16 tramas de contexto futuro **por capa**. A 40 ms por trama son unos 600-640 ms de *lookahead* por bloque, y se **acumula linealmente con la profundidad**:

$$17 \times 16 \text{ tramas} \times 40 \text{ ms} \approx 10.9 \text{ segundos de contexto derecho}$$

Ese es el *lookahead* solo del camino convolucional, ignorando la atención. Para un sistema de dictado en tiempo real es inutilizable. Las variantes causales posteriores atacan esto con convolución causal (*padding* solo a la izquierda, a costa de la mitad del campo receptivo efectivo), atención con contexto limitado o por *chunks* (que acota el *lookahead* a una constante independiente de la profundidad y hace la memoria lineal en $T$), y encoders en cascada donde un encoder causal emite hipótesis con baja latencia y un segundo encoder no causal las corrige.

{{< concept-alert type="advertencia" >}}
La comparación contra el Transformer Transducer no es del todo simétrica: el título de aquel paper es literalmente *"a **streamable** speech recognition model"*. Conformer le gana en WER, pero **no es *streamable***. Esa dimensión no aparece en ninguna tabla del paper.
{{< /concept-alert >}}

### Dependencia de SpecAugment y de la pila de regularización

Conformer L son 118.8M de parámetros entrenados sobre unas 960 horas de audio: un régimen fuertemente propenso al sobreajuste. El paper lo enfrenta apilando **cuatro** regularizadores simultáneos —SpecAugment con diez máscaras temporales, dropout 0.1 en todos los residuales, ruido variacional sobre los pesos y $\ell_2$ de $10^{-6}$— y **no ablaciona ninguno**, así que no sabemos cuánto del 4.3 de test-other se debe a la arquitectura y cuánto a la receta. Hay además una asimetría de comparación: SpecAugment con máscaras proporcionales a la longitud no es necesariamente lo que usaban todos los *baselines*, de modo que parte de la ventaja podría venir de la receta y no del bloque. El único control disponible es el ablation arquitectónico, que mantiene todo constante salvo el bloque y muestra +1.4 en dev-other; eso sí es atribuible a la arquitectura.

### Alcance experimental estrecho

- **Un solo dataset, un solo idioma, un solo dominio.** LibriSpeech es inglés, leído, de audiolibros, con buena relación señal-ruido y dicción clara. Nada sobre habla espontánea, conversacional, con ruido de fondo, con acentos fuera de distribución ni multilingüe. Y **una sola pérdida**: todo es RNN-T, sin CTC ni *encoder-decoder* con atención.
- **Ninguna medida de eficiencia real.** El paper reclama *parameter efficiency* en el abstract pero **no reporta FLOPs, latencia, memoria, tiempo de entrenamiento ni throughput**. Los parámetros son un proxy pobre: la atención tiene pocos parámetros y mucho cómputo dependiente de $T$. Con este paper no se puede saber si Conformer M es más rápido que ContextNet M en inferencia. Es la omisión más seria.
- **Los detalles del *stem* convolucional no se especifican**: se indica la tasa (10 ms → 40 ms) pero no cuántas capas, qué kernel ni qué canales, en un componente que afecta el costo de todo lo que sigue.
- **Sin varianza.** Un solo entrenamiento por configuración. Diferencias de 0.1-0.2 de WER —la escala de los ablations de macaron y de cabezas— no son distinguibles del ruido de semilla.

## Por qué importa hoy

Conformer se convirtió, en el lapso de un par de años, en **la arquitectura por defecto del encoder acústico**. Entre 2021 y 2024, si abrías un sistema de ASR de investigación o de producción, lo más probable era encontrar un encoder Conformer o un descendiente directo.

**Dónde está.** En **NVIDIA NeMo**, Conformer-CTC y Conformer-Transducer son los modelos base, y su evolución **FastConformer** —submuestreo 8× con convoluciones separables, que reduce a la mitad las tramas que ve la atención— es la base de Parakeet y Canary. En **torchaudio**, `torchaudio.models.Conformer` está en la API estable. En **ESPnet** el encoder Conformer es el estándar de facto de las recetas de ASR desde 2020-2021, y **WeNet**, **SpeechBrain** y **k2/icefall** implementan el bloque o un descendiente. En producción, el **Universal Speech Model (USM)** de Google usa un encoder Conformer de unos 2B de parámetros preentrenado sobre millones de horas, y **Chirp** —el ASR de Google Cloud— deriva de él. Más allá del ASR, el bloque se usa en separación de fuentes, detección de eventos sonoros, *keyword spotting*, mejora de habla, traducción de voz y como encoder de audio en modelos multimodales: el sesgo inductivo "local + global sobre una secuencia densa y larga" no tiene nada de específico del reconocimiento de habla.

### Dónde *no* está: Whisper

Este punto conviene dejarlo preciso porque se cita mal con frecuencia.

{{< concept-alert type="advertencia" >}}
**[Whisper](/papers/whisper-radford-2022) NO usa Conformer.** Usa un **Transformer encoder-decoder estándar** al estilo Vaswani: el encoder toma un espectrograma log-Mel de 30 segundos de duración fija, lo pasa por **dos convoluciones 1D** (kernel 3, la segunda con stride 2, activación GELU) como *stem* de submuestreo, suma **embeddings posicionales sinusoidales absolutos**, y de ahí en adelante son bloques Transformer convencionales. Sin módulo de convolución dentro del bloque, sin codificación posicional relativa, sin macaron.
{{< /concept-alert >}}

Es una decisión deliberada y muy informativa: el paper de Whisper argumenta explícitamente que quería una arquitectura **conocida y sin novedades** para que las mejoras fueran atribuibles a la escala de datos —680 000 horas de supervisión débil— y no al modelo. Y funcionó.

**La lectura correcta de esta divergencia es un argumento sobre el régimen de datos.** La arquitectura importa mucho en el régimen de ~1000 horas etiquetadas —donde vive Conformer, y donde los sesgos inductivos compran generalización— e **importa menos** en el régimen de 100 000+ horas, donde el modelo puede aprender la localidad desde los datos. Es la misma historia que el [Vision Transformer](/fundamentos/vision-transformer) contra las CNN en visión, y la misma que ordena la relación entre [AST](/papers/ast-gong-2021) y los modelos convolucionales de audio. Conformer sigue siendo la elección correcta si entrenas con datos limitados y quieres eficiencia en parámetros; Whisper demuestra que con suficiente audio un Transformer plano alcanza. Nótese que **incluso Whisper conserva el *stem* convolucional**: nadie alimenta atención directamente con 3000 tramas de espectrograma.

Un tercer punto del mismo mapa: [HuBERT](/papers/hubert-hsu-2021) y la familia de preentrenamiento autosupervisado atacan el problema por el eje de los datos *no etiquetados*, no por el de la arquitectura. Son estrategias ortogonales, y de hecho se combinan —w2v-BERT usa bloques Conformer como encoder para preentrenamiento autosupervisado.

### Los sucesores y qué corrigieron

| Modelo | Venue | Qué cambió respecto a Conformer |
|---|---|---|
| **Squeezeformer** (Kim et al.) | NeurIPS 2022 | Estructura **U-Net temporal**: submuestrea a la mitad en las capas intermedias y vuelve a subir, porque la resolución de 40 ms es innecesariamente fina en el medio de la red. Abandona el macaron y vuelve al macro-bloque estilo Transformer. Post-LN con escalado aprendido. Mejor relación WER/FLOPs. |
| **Branchformer** (Peng et al.) | ICML 2022 | Vuelve a la estructura **paralela** que el ablation de Conformer había descartado: una rama de atención y una rama cgMLP (*convolutional gating MLP*), fusionadas. Al hacerlo bien —ramas de ancho completo, fusión aprendida— resulta competitivo o mejor. |
| **E-Branchformer** (Kim et al.) | SLT 2022 | Mejora la fusión de Branchformer con una convolución depthwise sobre las salidas concatenadas y reincorpora las FFN macaron. Superó a Conformer en varios *benchmarks* de ESPnet. |
| **Zipformer** (Yao et al.) | ICLR 2024 | Encoder **multi-tasa** con estructura U-Net, **BiasNorm** en lugar de LayerNorm, activaciones SwooshR/SwooshL, **reutilización de los pesos de atención** entre módulos, y el optimizador ScaledAdam. Mejor WER con menos parámetros y menos memoria. Es el encoder de icefall/k2. |
| **FastConformer** (NVIDIA) | 2023 | Submuestreo 8× con convoluciones separables en el *stem*: menos tramas para la atención, 2-3× más rápido, sin pérdida de WER. |

El patrón común de los sucesores es revelador: **ninguno cuestiona la tesis local+global**. Todos la aceptan. Lo que atacan es la **resolución temporal uniforme** —Conformer procesa sus 17 capas a 25 tramas/s, lo cual es un desperdicio— y el **costo de la atención**. La idea central sobrevivió intacta; lo que se optimizó fue su implementación.

## En la clase 39: la tesis correcta con el operador actualizado

La [Clase 39](/clases/clase-39) sostiene que un modelo de audio conviene armarlo combinando **CNN** (features locales) + **RNN** (relaciones temporales distantes) + **MLP** (clasificación), porque los tres tienen propiedades complementarias, y presenta como *Ejemplo 1* la [CLDNN de Sainath et al. (2015)](/papers/cldnn-sainath-2015): convolución → LSTM → capas totalmente conectadas. Al final, la clase concluye que los Transformers *"no son actualmente muy populares para aplicaciones de audio"*.

Conformer es la refutación más elegante posible de esa conclusión, y lo es **precisamente porque acepta la premisa**. No discute que haya que combinar, no discute la partición local/global/clasificación, no discute que los tres roles sean distintos. Mantiene exactamente la tesis de la complementariedad y solo reemplaza la RNN por self-attention, fusionándola con la convolución dentro de un mismo bloque.

| Rol funcional | [CLDNN](/papers/cldnn-sainath-2015) (2015) | Conformer (2020) |
|---|---|---|
| Extractor de patrones locales | capa convolucional sobre el espectrograma | módulo de convolución (depthwise 1D, kernel 32) **dentro de cada bloque** |
| Modelador de dependencias distantes | 2 capas LSTM | multi-head self-attention con embedding posicional relativo |
| Transformación no lineal / clasificación | 2 capas totalmente conectadas | **dos medias** capas feed-forward (macaron), una a cada lado |
| Forma de combinar | **por etapas**: primero todo lo local, después todo lo global, después la clasificación | **intercalada y repetida**: los tres roles conviven en cada uno de los 16-17 bloques |

Esa última fila importa más de lo que parece. En CLDNN la LSTM ve features locales de un solo nivel de abstracción; en Conformer la atención de la capa 12 opera sobre representaciones que ya pasaron por 11 rondas de refinamiento local, y la convolución de la capa 12 opera sobre representaciones que ya fueron contextualizadas globalmente 11 veces. **La composición es multiplicativa, no aditiva.** Es la misma diferencia que separa a una [CRNN](/fundamentos/crnn) clásica de un modelo con bloques híbridos apilados.

### (a) La tesis del profesor es correcta, y Conformer la valida

El abstract de Conformer podría ser una diapositiva de la clase: *"Transformer models are good at capturing content-based global interactions, while CNNs exploit local features effectively. In this work, we achieve the best of both worlds."* Y el paper lo demuestra **tres veces, en tres direcciones distintas**:

1. **Quitar lo local de un modelo global daña.** Quitar el módulo de convolución de un Conformer, a parámetros constantes, cuesta **+0.4 de WER en dev-other**.
2. **Un modelo puramente local, bien diseñado, no alcanza en lo difícil.** ContextNet L —una CNN con *squeeze-and-excitation*, 112.7M— empata a Conformer L en test-clean pero pierde en test-other (4.6 vs 4.3 sin LM, 4.1 vs 3.9 con LM). Lo local basta para habla limpia; falla donde hay que desambiguar con contexto.
3. **Estirar lo local hasta cubrir lo global no funciona.** Agrandar el kernel de convolución de 32 a 65 tramas —2.6 segundos— **empeora** el WER. La convolución no puede reemplazar a la atención simplemente creciendo.

Ese tercer punto es el más fuerte y el más fácil de pasar por alto: es la demostración experimental de que los dos operadores **no son sustitutos**. Si lo fueran, la curva del kernel sería monótona.

### (b) Lo que cambió es cuál es el mejor operador global

Conformer conserva el rol que la LSTM ocupaba en CLDNN y cambia la pieza que lo llena. Las razones son estructurales y se pueden enunciar con precisión.

**Razón 1: longitud del camino.** En una RNN, propagar información entre las posiciones $i$ y $j$ requiere $|i - j|$ aplicaciones de la celda recurrente: camino $O(n)$. En self-attention, cualquier par de posiciones está conectado por **una sola** operación: camino $O(1)$. Con audio a 25 tramas/s tras el submuestreo, una dependencia a 8 segundos de distancia son **200 pasos recurrentes** contra **un producto punto**. El gradiente que debe viajar 200 aplicaciones de una LSTM se atenúa; el que viaja por un peso de atención, no.

La tabla canónica de Vaswani et al., con $n$ = longitud de secuencia, $d$ = dimensión, $k$ = kernel:

| Capa | Complejidad por capa | Operaciones secuenciales | Longitud máxima de camino |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrente | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolucional (kernel $k$) | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ dilatada / $O(n/k)$ |

**Razón 2: paralelización del entrenamiento.** La columna "operaciones secuenciales" es la que decide qué modelos se pueden entrenar. Una LSTM sobre 250 tramas necesita 250 pasos que **no se pueden solapar** —ni con más GPUs ni con más memoria, porque es una dependencia de datos. Una capa de atención procesa las 250 tramas en un par de multiplicaciones de matrices. Sobre hardware que es esencialmente un multiplicador de matrices, la diferencia de *throughput* es de un orden de magnitud. Y la consecuencia va más allá de la velocidad: **hace entrenables modelos que antes no lo eran**. El encoder de 2B parámetros de USM sobre millones de horas no es una LSTM porque no podría serlo.

**Razón 3: el contexto es dinámico, no comprimido.** Una RNN comprime todo el pasado en un vector de estado de tamaño fijo: todo lo que la trama 250 sabe de la trama 10 tuvo que sobrevivir a 240 escrituras sobre el mismo estado. La atención no comprime: **consulta**. Cada posición formula una *query* y recupera lo que necesita del conjunto completo, con un patrón distinto por posición y por cabeza. Es exactamente el mismo argumento que Conformer usa contra el *squeeze-and-excitation* de ContextNet —*"it is still limited in capturing dynamic global context as it only applies a global averaging"*— y vale igual contra el estado oculto de una RNN.

**El precio de haber cambiado, para ser justos.** La atención pagó por esas tres ventajas: complejidad $O(n^2 d)$ contra $O(n d^2)$, la pérdida del *streaming* natural que la recurrencia da gratis, y la pérdida del sesgo posicional implícito —que es exactamente lo que la codificación relativa y el módulo de convolución vienen a restituir.

{{< concept-alert type="clave" >}}
Leído así, **el bloque Conformer es el precio que hay que pagar para usar atención en audio**. No es un Transformer con un extra pegado: es un Transformer al que le devolvieron las dos propiedades que la recurrencia y la convolución tenían y la atención pura no.
{{< /concept-alert >}}

### (c) "Los Transformers no modelan bien dependencias largas": contrastado con datos

Aquí hay dos afirmaciones distintas y ambas se pueden contrastar directamente.

**Sobre "no modelan bien dependencias largas":** es lo opuesto de lo que dice el paper. El abstract describe a los Transformers como *"good at capturing content-based global interactions"* y la introducción como buenos *"at modeling long-range global context"*. La limitación que Conformer les atribuye es **exactamente la contraria**: *"they are less capable to extract fine-grained local feature patterns"*. La debilidad del Transformer en audio es **lo local**, no lo distante, y el módulo de convolución existe precisamente para eso.

**Sobre "no son populares en audio":** en 2020 era discutible; hoy es insostenible. Los encoders acústicos de NeMo, ESPnet, torchaudio, SpeechBrain, WeNet, icefall, Whisper, wav2vec 2.0, HuBERT, USM y prácticamente todo lo que se publica en Interspeech/ICASSP son basados en atención. El bloque Conformer es la arquitectura de audio más replicada de la década.

**Qué es comparable y qué no.** El contraste directo CLDNN (2015) contra Conformer (2020) es tentador pero **no es una comparación válida**, y hay que decirlo:

- CLDNN se evaluó sobre tareas internas de Google (*Voice Search*, habla espontánea de dominio abierto); Conformer sobre LibriSpeech (audiolibros leídos). **Datasets distintos, dominios distintos: los WER no son comparables entre sí.**
- Median cinco años de diferencia en optimizadores, aumentación de datos (SpecAugment no existía en 2015), tokenización, pérdida de entrenamiento (entropía cruzada con alineamiento contra RNN-T) e infraestructura. Atribuir cualquier diferencia solo a la arquitectura sería un error.

**Lo que sí es comparable, y es suficiente,** está dentro de la misma tabla del mismo paper, sobre el mismo dataset, con la misma métrica y en la misma condición:

| | Operador global | Params | test-clean (con LM) | test-other (con LM) |
|---|---|---|---|---|
| LAS con encoder **LSTM** | recurrencia | 360 M | 2.2 | 5.2 |
| **Conformer (S)** | self-attention + convolución | **10.3 M** | **2.1** | **5.0** |
| **Conformer (L)** | self-attention + convolución | 118.8 M | **1.9** | **3.9** |

Un modelo basado en atención con **35 veces menos parámetros** le gana a la mejor arquitectura LSTM de la tabla, en ambos *splits*. Y el modelo grande, con un tercio de los parámetros de la LSTM, baja el WER de test-other de 5.2 a 3.9: **25% relativo**. Esto no prueba que las RNN sean inútiles —siguen ahí, en el decoder del propio Conformer, modelando la secuencia de etiquetas, y siguen siendo la elección natural cuando el *streaming* con estado acotado es un requisito duro. Prueba algo más acotado y más cierto: **para modelar la señal acústica, la self-attention con sesgos posicionales relativos desplazó a la recurrencia, y lo hizo en la misma tarea de ASR donde nació la CLDNN, con una fracción de los parámetros.**

{{< concept-alert type="recordar" >}}
El cierre, que es lo que hace elegante a Conformer como respuesta a la clase: **el razonamiento del profesor es correcto y el diseño también.** La estructura CNN + [operador global] + MLP sigue siendo la arquitectura correcta para audio. Conformer es esa misma estructura con la casilla del medio actualizada. La [profundización de la clase 39](/clases/clase-39/profundizacion) desarrolla el paso de la recurrencia a la atención en el [dominio del audio](/dominios/audio).
{{< /concept-alert >}}

## Erratas y matices

### La confusión más frecuente: con o sin modelo de lenguaje

Se cita habitualmente "Conformer: 2.1/4.3 en LibriSpeech" o "Conformer: 1.9/3.9" sin especificar nada. Ambos son de Conformer **L** y corresponden a **condiciones distintas**:

| Cifra (test-clean / test-other) | Modelo | ¿LM externo? |
|---|---|---|
| **2.1 / 4.3** | Conformer L (118.8M) | **No** |
| **1.9 / 3.9** | Conformer L (118.8M) | **Sí** (LSTM 3×4096, *shallow fusion*) |
| **2.7 / 6.3** | Conformer S (10.3M) | **No** |
| **2.1 / 5.0** | Conformer S (10.3M) | **Sí** |
| **2.3 / 5.0** | Conformer M (30.7M) | **No** |
| **2.0 / 4.3** | Conformer M (30.7M) | **Sí** |

Errores concretos que circulan: atribuir 1.9/3.9 al modelo sin LM (sin LM el mejor es 2.1/4.3); citar 2.7/6.3 como "el modelo pequeño con LM" (es sin LM; con LM es 2.1/5.0); y decir "Conformer llega a 1.9% de WER en LibriSpeech" sin más, cuando eso es test-**clean** con LM externo y test-other es el doble. Además, **todos los ablations están sin LM externo**: por eso el Conformer de referencia en esas tablas aparece como 2.1/4.3 en test y 1.9/4.4 en dev.

### Cabezas de atención: la tabla de configuraciones y el ablation no concuerdan

El paper afirma que subir a **16 cabezas** mejora, y el ablation lo confirma en dev-other (4.3 con 16 contra 4.4 con 8). Pero la tabla de configuraciones lista **8 cabezas** para Conformer L, y el *baseline* de todos los demás ablations (dev 1.9/4.4) corresponde a la fila de 8: **el modelo publicado no usa la configuración que el propio ablation identifica como mejor en dev-other**. La explicación probable es que 8 cabezas gana en test-other (4.3 vs 4.4) y en dev-clean (1.9 vs 2.0) —pero eso implicaría haber seleccionado mirando test—, o simplemente que la configuración se fijó antes de correr el ablation. El paper no lo aclara, y en cualquier caso las diferencias están dentro del ruido de una sola semilla.

### "970 hours" de LibriSpeech

El paper describe el dataset como *"which consists of 970 hours of labeled speech"*. LibriSpeech-960, el conjunto de entrenamiento estándar, tiene **960.85 horas**; sumando dev y test (~21 horas) se llega a ~981. La cifra de 970 no corresponde a ninguna partición canónica y es casi seguro un redondeo laxo: lo que se entrena es LibriSpeech-960, como confirma el propio paper al describir el LM (*"tokenized with the 1k WPM built from LibriSpeech 960h"*).

### La fila de Karita et al., donde el LM empeora el WER

En la tabla de resultados, `LAS / Transformer` de Karita et al. reporta **2.2/5.6 sin LM** y **2.6/5.7 con LM**: el modelo de lenguaje externo **empeora** el WER en ambos *splits*. Eso no tiene sentido físico —el *shallow fusion* con $\lambda$ ajustado en dev puede no ayudar, pero degradar 0.4 puntos en clean es raro. Es probablemente un error de transcripción o una inversión de columnas. Conviene no citar esa fila.

### Kernel 32 es par

Una convolución "same" con kernel par **no admite *padding* simétrico**: con $\text{pad}=(k-1)/2 = 15$ a cada lado, la salida tiene $T-1$ tramas. El paper no discute cómo lo maneja (*padding* asimétrico 16/15 es lo natural). Prácticamente todas las reimplementaciones usan **kernel 31**, y `torchaudio.models.Conformer` directamente **exige kernel impar** con un *assert*. Si alguien reporta "Conformer con kernel 31", está reproduciendo el paper correctamente, no desviándose.

Hay además tres erratas menores de redacción: *"Sections 2.1, **1**, and 2.3"* por 2.2, *"As in **Macron**-Net"* por Macaron-Net, y *"LibriSpeech **langauge** model corpus"*.

### Lo que Conformer *no* afirma y a veces se le atribuye

- **No dice que las RNN sean inútiles.** Su propio decoder es una LSTM de una capa, en los tres tamaños.
- **No es un modelo de *streaming*.** Las variantes causales son trabajo posterior de otros papers.
- **No propone una nueva forma de atención.** Toma la de Transformer-XL tal cual.
- **No inventa la combinación convolución + atención.** Cita a Bello et al., QANet, Lite Transformer y Yang et al. como antecedentes. Su aporte es la **organización específica** —secuencial, convolución después de atención, envuelta en macaron— más el ablation que la justifica.

## Notas y enlaces

Tres trampas de reimplementación, en orden de gravedad:

1. **`nn.MultiheadAttention` de PyTorch no sirve** para reproducir el paper: implementa atención con posición absoluta o sin posición, mientras que la codificación relativa de Transformer-XL requiere las dos proyecciones de clave, los sesgos $u, v$ y el *relative shift*. Usarla y llamarlo Conformer es reproducir la fila "– Relative Pos. Emb." del ablation: unos 0.6-0.7 puntos de WER de más en los *splits* difíciles.
2. **La máscara de *padding* importa dos veces** dentro del módulo de convolución: para no arrastrar posiciones inexistentes a través del kernel de 31 tramas, y para no contaminar las estadísticas de BatchNorm.
3. **Los factores $\tfrac{1}{2}$ van solo en las dos FFN.** La atención y la convolución llevan residual de paso completo.

**`torchaudio.models.Conformer`** trae una implementación mantenida, pero es **solo el encoder**: no incluye el *convolution subsampling*, ni el decoder LSTM, ni la pérdida RNN-T. Su API expone como *flags* justamente las dos decisiones que el paper ablacionó —`convolution_first` (la peor fila del barrido de orden) y `use_group_norm` (reemplazar BatchNorm)—, lo que dice bastante sobre cuáles fueron los puntos de fricción reales en las reimplementaciones. Y **para producción hoy** la elección razonable no es este bloque tal cual sino un descendiente: FastConformer si importa el *throughput*, Zipformer si importan la memoria y el WER, E-Branchformer si el ecosistema es ESPnet. Todos comparten la tesis; ninguno comparte la implementación literal de 2020.

**Enlaces relacionados:** [Clase 39](/clases/clase-39) · [profundización de la clase 39](/clases/clase-39/profundizacion) · [CLDNN (Sainath, 2015)](/papers/cldnn-sainath-2015) · [AST (Gong, 2021)](/papers/ast-gong-2021) · [HuBERT (Hsu, 2021)](/papers/hubert-hsu-2021) · [Whisper (Radford, 2022)](/papers/whisper-radford-2022) · [SpecAugment (Park, 2019)](/papers/specaugment-park-2019) · [Self-attention](/fundamentos/self-attention) · [Transformer](/fundamentos/transformer) · [LSTM y GRU](/fundamentos/lstm-gru) · [CRNN](/fundamentos/crnn) · [Dominio: audio](/dominios/audio)
