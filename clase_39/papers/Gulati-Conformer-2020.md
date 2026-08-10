# Conformer: Convolution-augmented Transformer for Speech Recognition — Análisis interno

> Nota de método: todas las cifras marcadas con referencia a Tabla/Sección provienen del PDF `Gulati-Conformer-2020.pdf` (arXiv:2005.08100v1). Lo que viene de fuera del paper está marcado explícitamente como conocimiento externo. Donde no pude verificar algo contra el PDF, lo digo en vez de rellenarlo.

## 1. Metadata y resumen ejecutivo

- **Título:** *Conformer: Convolution-augmented Transformer for Speech Recognition*
- **Autores:** Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang. **Todos en Google Inc.**
- **Venue:** Interspeech 2020. Preprint arXiv:2005.08100v1, 16 de mayo de 2020.
- **Toolkit:** Lingvo (Sección 3.2), el framework de secuencia a secuencia de Google.
- **Dato de linaje:** Niki Parmar es coautora de *Attention Is All You Need*; Yonghui Wu venía de GNMT; ContextNet (referencia [10] del paper, el rival más duro de la Tabla 2) es del mismo grupo y salió apenas 13 días antes (arXiv:2005.03191). Esto no es una comparación externa: es una auditoría interna del mismo laboratorio contra su propia arquitectura convolucional.

**La tesis en una línea:** los Transformers modelan bien el contexto global pero son menos capaces de extraer patrones locales de grano fino; las CNN explotan lo local pero necesitan muchas capas o parámetros para alcanzar contexto global; combinarlos **dentro del mismo bloque** —no en ramas paralelas, no en etapas separadas— da lo mejor de ambos con muchos menos parámetros.

Cifras ancla en LibriSpeech test-clean / test-other (Tabla 2), WER en porcentaje, menor es mejor:

| Método | Familia | #Params (M) | Sin LM (clean/other) | Con LM (clean/other) |
|---|---|---|---|---|
| QuartzNet | CTC | 19 | 3.90 / 11.28 | 2.69 / 7.25 |
| Transformer (Synnaeve et al.) | LAS | 270 | 2.89 / 6.98 | 2.33 / 5.17 |
| LSTM | LAS | 360 | 2.6 / 6.0 | 2.2 / 5.2 |
| Transformer Transducer | Transducer | **139** | 2.4 / 5.6 | 2.0 / 4.6 |
| ContextNet (S) | Transducer | 10.8 | 2.9 / 7.0 | 2.3 / 5.5 |
| ContextNet (M) | Transducer | 31.4 | 2.4 / 5.4 | 2.0 / 4.5 |
| ContextNet (L) | Transducer | 112.7 | 2.1 / 4.6 | 1.9 / 4.1 |
| **Conformer (S)** | Transducer | **10.3** | 2.7 / 6.3 | 2.1 / 5.0 |
| **Conformer (M)** | Transducer | **30.7** | 2.3 / 5.0 | 2.0 / 4.3 |
| **Conformer (L)** | Transducer | **118.8** | **2.1 / 4.3** | **1.9 / 3.9** |

Las tres lecturas que importan, todas verificables en esa tabla:

1. **Conformer M, con 30.7M parámetros, supera al Transformer Transducer de 139M** en las cuatro columnas donde ambos reportan: 2.3/5.0 contra 2.4/5.6 sin LM, y 2.0/4.3 contra 2.0/4.6 con LM (empate en test-clean con LM). Son **4.5 veces menos parámetros**.
2. **Conformer S, con 10.3M parámetros y LM, da 2.1/5.0; el LAS con LSTM de 360M da 2.2/5.2.** Un modelo **35 veces más chico** basado en atención le gana a la mejor RNN de la tabla. Este es el dato que más directamente toca el argumento de la clase 39, y volveremos a él en la Sección 13.
3. El titular del abstract —**2.1%/4.3% sin LM y 1.9%/3.9% con LM**— corresponde exclusivamente a **Conformer L (118.8M)**. La mejora relativa de 15% que anuncia la introducción es sobre test-other **con LM externo**: $(4.6 - 3.9)/4.6 = 15.2\%$ contra el Transformer Transducer.

## 2. Contexto: el ASR en 2019-2020

Para entender por qué Conformer aterriza como aterriza hay que ver el mapa de fuerzas del momento. En 2019-2020 el ASR end-to-end estaba repartido en tres familias, y cada una fallaba en algo distinto.

**Las RNN eran la opción por defecto.** El paper lo dice sin adornos en la introducción: *"Recurrent neural networks (RNNs) have been the de-facto choice for ASR [1,2,3,4] as they can model the temporal dependencies in the audio sequences effectively"*. La referencia [5] es Graves 2012, *Sequence Transduction with Recurrent Neural Networks*: el RNN-T, la pérdida que permite entrenar un transductor de secuencias sin alineamiento previo y que hace posible el streaming. La línea de Google —Chiu et al. 2018, Rao et al. 2017, He et al. 2019, Sainath et al. 2020, todas citadas— era LSTM encoder + LSTM prediction network + RNN-T loss, corriendo en teléfonos.

El problema de esta familia no es la calidad: es estructural.

- **El entrenamiento no paraleliza sobre el tiempo.** Una LSTM necesita $O(n)$ operaciones secuenciales para procesar $n$ tramas. Con audio a 100 tramas por segundo, una utterance de 10 segundos son 1000 pasos que no se pueden solapar. Esto acota el tamaño de modelo entrenable con un presupuesto de GPU dado.
- **El camino entre dos posiciones distantes es $O(n)$.** La información entre la trama 10 y la trama 900 tiene que atravesar 890 aplicaciones de la celda recurrente. Las gates de la LSTM mitigan el desvanecimiento del gradiente, no lo eliminan.

**Los Transformers llegaron y ganaron.** El Speech-Transformer (Dong et al., ICASSP 2018, referencia [24]) y sobre todo el **Transformer Transducer** (Zhang et al., ICASSP 2020, referencia [7]) mostraron que se podía reemplazar el encoder LSTM por un encoder Transformer y ganar. El paper resume la razón: *"due to its ability to capture long distance interactions and the high training efficiency"*. Ambas cosas: camino $O(1)$ entre cualquier par de posiciones, y todo el eje temporal procesado en paralelo dentro de una capa.

Lo que les faltaba está en la primera frase de la Sección 1, párrafo 2: *"While Transformers are good at modeling long-range global context, they are less capable to extract fine-grained local feature patterns."* La self-attention es un operador de **agregación ponderada por contenido sobre un conjunto**. Sin sesgo inductivo posicional, una capa de atención trata la secuencia como una bolsa: no tiene noción privilegiada de "la trama de al lado". En audio eso duele, porque los eventos acústicos que definen un fonema —una explosión, una transición formántica, un onset— viven en ventanas de 20 a 100 milisegundos y son **estructura local con desplazamiento arbitrario**. Aprender ese detector desde atención pura significa gastar capacidad reaprendiendo la equivarianza traslacional que una convolución tiene gratis.

**Las CNN también funcionaban.** Jasper, QuartzNet, ContextNet, y antes Sainath et al. 2013 y Abdel-Hamid et al. 2014 (referencias [8-12]). El paper describe su mecanismo: *"capture local context progressively via a local receptive field layer by layer"*. El campo receptivo de una pila de $L$ convoluciones de kernel $k$ crece linealmente, $\approx L(k-1)+1$. Para cubrir una utterance de 250 tramas con kernel 31 hacen falta ~8 capas; para cubrir 2500, ~83. La cita textual: *"One limitation of using local connectivity is that you need many more layers or parameters to capture global information."*

**El intento de arreglarlo desde el lado CNN: ContextNet.** El paper le dedica un párrafo específico porque es su rival directo. ContextNet mete el módulo **squeeze-and-excitation** (Hu et al., CVPR 2018) en cada bloque residual para inyectar contexto largo. La crítica de Conformer es precisa y vale la pena entenderla:

> *"However, it is still limited in capturing **dynamic** global context as it only applies a **global averaging** over the entire sequence."*

Squeeze-and-excitation comprime toda la secuencia a **un vector** por promedio, lo pasa por un MLP y usa el resultado para reescalar canales. Es contexto global, sí, pero **de un solo bit de resolución posicional: ninguna**. Todas las posiciones reciben la misma modulación. La self-attention, en cambio, calcula un contexto **distinto para cada posición**, condicionado a lo que esa posición está preguntando. Esa es la diferencia entre "global" y "dinámicamente global", y es exactamente el argumento que Conformer necesita para justificar por qué la atención no es reemplazable por un truco barato de pooling.

**El trabajo concurrente que Conformer reconoce.** Bello et al. 2019 (*Attention Augmented CNNs*, [14]) ya había mostrado que combinar convolución y self-attention mejora sobre usarlas por separado, en visión. Yang et al. y QANet ([15,16]) habían aumentado la atención con información de posición relativa. Y sobre todo **Lite Transformer** (Wu et al., 2020, [17]), que parte la entrada en **dos ramas paralelas** —una de atención, una convolucional— y concatena las salidas. Conformer se declara *"Inspired by Wu et al. [17,18]"* pero toma la decisión opuesta en el punto clave: **secuencial en vez de paralelo**. Y lo justifica con un ablation (Sección 10).

## 3. La tesis: local y global no son sustitutos, son complementos

La formulación del paper, Sección 1:

> *"We hypothesize that both global and local interactions are important for being parameter efficient. [...] self-attention learns the global interaction whilst the convolutions efficiently capture the relative-offset-based local correlations."*

Hay dos afirmaciones ahí y conviene separarlas.

**Afirmación 1 (la débil): combinar ayuda.** Trivialmente cierta y ya demostrada por Bello et al.

**Afirmación 2 (la fuerte, y la que sostiene el paper): combinar es *más eficiente en parámetros*.** No dice "combinar da mejor WER a igual costo"; dice que la combinación es la que permite llegar al mismo WER con menos capacidad. Y eso sí lo demuestra la Tabla 2: 30.7M contra 139M, 10.3M contra 360M. El argumento mecánico es que cada operador tiene un sesgo inductivo que le sale gratis y que el otro tendría que **comprar con parámetros**:

| | Sesgo inductivo que trae gratis | Lo que tiene que comprar con parámetros |
|---|---|---|
| **Convolución** | equivarianza traslacional, localidad, pesos compartidos sobre el tiempo | contexto global: crece $O(L)$ en profundidad |
| **Self-attention** | camino $O(1)$ a cualquier posición, mezcla condicionada al contenido | localidad y noción de vecindad: hay que aprenderla desde datos |

La eficiencia en parámetros no viene de que dos módulos "sumen capacidad"; viene de que **cada uno deja de pagar por lo que el otro ya resuelve estructuralmente**.

### Dónde coincide y dónde difiere con la tesis de la clase

La clase 39 sostiene que para audio conviene combinar **CNN (features locales) + RNN (relaciones temporales distantes) + MLP (clasificación)** porque tienen propiedades complementarias, y presenta como Ejemplo 1 la arquitectura **CLDNN** de Sainath et al. (ICASSP 2015): conv → LSTM → fully connected.

**Conformer acepta la premisa completa.** No discute que haya que combinar; no discute la partición local/global/clasificación; no discute que los tres roles sean distintos. Mapeando componente a componente:

| Rol funcional | CLDNN (2015) | Conformer (2020) |
|---|---|---|
| Extractor de patrones locales | capa convolucional sobre el espectrograma | módulo de convolución (depthwise 1D, kernel 32) **dentro de cada bloque** |
| Modelador de dependencias distantes | 2 capas LSTM | multi-head self-attention con embedding posicional relativo |
| Transformación no lineal / clasificación | 2 capas fully connected | **dos medias** capas feed-forward (macaron), una a cada lado |
| Combinación | **secuencial y global**: una etapa conv, después una etapa LSTM, después una etapa FC | **intercalada y repetida**: los tres roles conviven en cada uno de los 16-17 bloques |

Las dos diferencias reales:

1. **Qué operador cumple el rol global.** La RNN se reemplaza por self-attention. Este es el cambio sustantivo y el hilo conductor del análisis.
2. **La granularidad de la combinación.** CLDNN combina *por etapas*: primero todo lo local, después todo lo global. Conformer combina *por bloque*, y repite la secuencia local→global 16 o 17 veces. Esto importa: en CLDNN la LSTM ve features locales de un solo nivel de abstracción; en Conformer la atención de la capa 12 opera sobre representaciones que ya pasaron por 11 rondas de refinamiento local, y la convolución de la capa 12 opera sobre representaciones que ya fueron contextualizadas globalmente 11 veces. La composición es multiplicativa, no aditiva.

Es importante ser justo: **la tesis del profesor no queda refutada por Conformer, queda confirmada**. Lo que queda desactualizado es la instanciación concreta del rol global (la RNN) y la conclusión final del slide sobre la irrelevancia de los Transformers en audio. Volvemos a eso con cifras en la Sección 13.

## 4. El bloque Conformer en detalle

El encoder completo (Figura 1) es:

```
audio (tramas cada 10 ms)
  → SpecAugment
  → Convolution Subsampling      [10 ms → 40 ms, es decir 100 fps → 25 fps]
  → Linear
  → Dropout
  → N × Conformer Block
```

y cada bloque es la estructura **macaron**: media FFN → MHSA → Conv → media FFN → LayerNorm.

Las ecuaciones exactas del paper (Ecuación 1, Sección 2.4), para la entrada $x_i$ al bloque $i$:

$$\tilde{x}_i = x_i + \frac{1}{2}\,\mathrm{FFN}(x_i)$$

$$x_i' = \tilde{x}_i + \mathrm{MHSA}(\tilde{x}_i)$$

$$x_i'' = x_i' + \mathrm{Conv}(x_i')$$

$$y_i = \mathrm{Layernorm}\!\left(x_i'' + \frac{1}{2}\,\mathrm{FFN}(x_i'')\right)$$

Cuatro detalles que se pasan por alto al leer rápido:

**(a) Solo las FFN llevan el factor $\tfrac{1}{2}$.** Los residuales de MHSA y Conv son de paso completo. El medio paso es una propiedad específica del par macaron, no una convención general del bloque.

**(b) Todos los módulos son *pre-norm*.** El paper lo dice para MHSA (Sección 2.1: *"We use pre-norm residual units [21,22] with dropout"*) y para FFN (Sección 2.3: *"we follow pre-norm residual units and apply layer normalization within the residual unit and on the input before the first linear layer"*), y la Figura 2 confirma que el módulo de convolución también arranca con LayerNorm. Es decir: la normalización está **dentro** de cada rama residual, y la ruta identidad va limpia de extremo a extremo. Eso es lo que permite entrenar 17 capas con warm-up de 10k pasos sin que explote.

**(c) Pero además hay un LayerNorm *post* al final del bloque.** Esto es inusual. Un Transformer pre-norm canónico pone un único LayerNorm al final de toda la pila. Conformer pone uno **por bloque**, después del último residual. En la práctica esto reescala la salida de cada bloque a norma controlada antes de entregarla al siguiente, lo que evita la deriva de escala típica del pre-norm profundo (donde la varianza de la ruta residual crece con la profundidad). Es un híbrido pre-norm/post-norm y el paper no lo argumenta, solo lo dibuja en la Figura 1 y lo escribe en la Ecuación 1.

**(d) El dropout va en la salida de cada módulo, antes del residual.** Sección 3.2: *"we apply dropout in each residual unit of the conformer, i.e, to the output of each module, before it is added to the module input. We use a rate of $P_{drop} = 0.1$."*

### Por qué ese orden y no otro

El orden **MHSA antes que Conv** no es arbitrario: es resultado de un barrido (Tabla 4, Sección 3.4.2, todas cifras en dev):

| Arquitectura | dev-clean | dev-other |
|---|---|---|
| **Conformer (MHSA → Conv)** | **1.9** | **4.4** |
| Convolución antes de MHSA | 1.9 | 4.5 |
| MHSA y Convolución en paralelo, salidas concatenadas | 2.0 | 4.9 |
| Depthwise conv reemplazada por lightweight conv | 2.0 | 4.8 |

Conclusión del paper: *"convolution module stacked after the self-attention module works best for speech recognition"*. La lectura mecánica: la atención produce una representación **ya contextualizada globalmente**, y la convolución después refina localmente **sobre ese contexto**. Al revés, la atención tendría que operar sobre features locales crudos. La diferencia es chica (0.1 en dev-other), pero es consistente.

Lo más informativo de esa tabla es la fila del paralelo: **0.5 puntos peor en dev-other**. Esa es la configuración de Lite Transformer, el trabajo que Conformer cita como inspiración. La razón plausible es que en paralelo cada rama ve **la mitad de los canales** y las salidas se combinan por concatenación (una operación lineal), mientras que en serie ambos módulos ven el ancho completo y la composición es no lineal. Guarda esta fila: en la Sección 12 veremos que Branchformer la revirtió dos años después.

### Presupuesto de parámetros del bloque

El paper no lo desglosa, así que lo calculo yo. Para dimensión de modelo $d$, expansión FFN de 4 (Figura 4) y kernel de convolución $k$:

| Módulo | Parámetros (aprox., sin sesgos ni normalizaciones) | Con $d=512$, $k=32$ |
|---|---|---|
| FFN × 2 | $2 \times (d \cdot 4d + 4d \cdot d) = 16d^2$ | 4.19 M |
| MHSA (con rel-pos: $W_q,W_k,W_v,W_o$ + $W_{k,R}$) | $\approx 5d^2$ | 1.31 M |
| Convolución (pointwise $d\!\to\!2d$, depthwise $k$, pointwise $d\!\to\!d$) | $2d^2 + kd + d^2 = 3d^2 + kd$ | 0.80 M |
| **Total por bloque** | $\approx 24d^2 + kd$ | **6.30 M** |

Multiplicado por las 17 capas de Conformer L da ~107M, más el decoder LSTM y el subsampling llegamos al orden de los 118.8M reportados. La estimación cierra, lo cual valida que el desglose es correcto.

El dato interesante: **las dos medias FFN son ~2/3 de los parámetros del bloque, y el módulo de convolución apenas el ~13%**. Pero cuando el ablation quita la convolución (Sección 10), el WER en dev-other se degrada 0.4 puntos. Es la contribución con mejor relación WER/parámetro del bloque, por lejos. Eso es literalmente la tesis de "eficiencia en parámetros" hecha número.

## 5. El módulo de convolución

La Figura 2 lo define capa por capa:

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

Pieza por pieza.

**LayerNorm de entrada.** Es el pre-norm del residual. Estabiliza la escala antes de que empiece la expansión de canales.

**Pointwise conv con factor de expansión 2.** Una convolución $1\times1$ sobre el eje de canales, es decir una capa lineal aplicada posición a posición. Duplica los canales de $d$ a $2d$. Su función es puramente preparar la entrada del GLU, que consume dos mitades.

**GLU (Gated Linear Unit).** De Dauphin et al., ICML 2017 (referencia [23]). Parte el tensor en dos mitades $a, b \in \mathbb{R}^{d}$ y calcula

$$\mathrm{GLU}(a, b) = a \odot \sigma(b)$$

Vuelve a $d$ canales. Lo que aporta es una **interacción multiplicativa condicionada a los datos**: la rama $b$ decide, canal por canal y posición por posición, cuánto de la rama $a$ pasa. Es un mecanismo de compuerta, hermano de las gates de una LSTM pero sin recurrencia y sin estado. Su valor real es doble: (i) da al módulo la capacidad de **suprimir selectivamente** canales o instantes irrelevantes antes de la convolución temporal —silencio, ruido, regiones sin contenido fonético—, algo que una no linealidad puntual como ReLU no puede hacer; (ii) mantiene una **ruta lineal** ($a$ se multiplica sin pasar por una saturación), lo que preserva el gradiente. Ese fue el argumento original de Dauphin et al. frente a las LSTM.

El paper es explícito en que esto viene de Lite Transformer: *"Inspired by [17], the convolution module starts with a gating mechanism [23]—a pointwise convolution and a gated linear unit (GLU)."*

**Depthwise conv 1D.** Aquí está el corazón del módulo y la razón de que sea barato. Una convolución 1D **completa** con kernel $k$ sobre $d$ canales de entrada y $d$ de salida cuesta

$$P_{\text{full}} = k \cdot d^2$$

Una **depthwise** aplica un filtro independiente de largo $k$ a cada canal, sin mezclar canales:

$$P_{\text{dw}} = k \cdot d$$

La razón es exactamente $d$. Con $d = 512$ y $k = 32$: **8,388,608 parámetros contra 16,384**. Un factor de 512. El costo en FLOPs baja en la misma proporción. La mezcla entre canales, que la depthwise sacrifica, la restituyen las dos pointwise que la rodean —es la factorización de Xception/MobileNet, trasladada al eje temporal: separar "mezclar en el tiempo" de "mezclar entre canales".

Sin esa factorización el módulo de convolución no cabría en el presupuesto. Con $k=32$ y $d=512$, una convolución completa costaría 8.4M parámetros **por bloque**, más que las dos FFN juntas, y el argumento de eficiencia en parámetros del paper se caería entero. **La depthwise es lo que hace viable la tesis, no un detalle de implementación.**

**BatchNorm.** El paper la justifica en una línea (Sección 2.2): *"Batchnorm is deployed just after the convolution to aid training deep models."* Nótese la asimetría: el resto del bloque usa LayerNorm; solo aquí aparece BatchNorm. Es una herencia del mundo CNN. Y es, en mi lectura, la decisión más frágil del diseño: BatchNorm calcula estadísticas sobre el eje de batch **y** el eje temporal, lo que en un batch de utterances de largo variable **mezcla estadísticas de tramas reales con tramas de padding** salvo que se enmascare cuidadosamente; y en streaming introduce discrepancia entre entrenamiento (estadísticas de batch) e inferencia (estadísticas acumuladas). Por eso muchas implementaciones posteriores la reemplazan —`torchaudio` expone directamente un flag `use_group_norm` para cambiarla por GroupNorm.

**Swish.** $\mathrm{Swish}(x) = x \cdot \sigma(\beta x)$, con $\beta = 1$ es SiLU. De Ramachandran, Zoph y Le, 2017 (referencia [25]). El paper la usa aquí y en las FFN. Su efecto medido está en la Sección 3.4.1: *"Using swish activations led to faster convergence in the Conformer models."* Ojo con esa frase: habla de **velocidad de convergencia**, no de WER final. Y la Tabla 3 lo confirma: cambiar Swish por ReLU deja dev-clean y dev-other **idénticos** (1.9/4.4), mejora test-clean (2.1 → 2.0) y empeora test-other (4.3 → 4.5). Es ruido. Swish está ahí por optimización, no por calidad.

**Pointwise conv de salida y dropout.** Proyecta de vuelta a $d$ y regulariza antes del residual.

### El kernel size, y qué dice el ablation

Los tres modelos usan **kernel 32** (Tabla 1). El barrido está en la Tabla 7, Sección 3.4.5, sobre el modelo grande, con el mismo kernel en todas las capas. Es la única tabla del paper con dos decimales:

| Kernel | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| 3 | 1.88 | 4.41 | 1.99 | 4.39 |
| 7 | 1.88 | 4.30 | 2.02 | 4.44 |
| 17 | 1.87 | 4.31 | 2.04 | 4.38 |
| **32** | **1.83** | **4.30** | 2.03 | **4.29** |
| 65 | 1.89 | 4.47 | 1.98 | 4.46 |

Tres observaciones:

1. **Hay un óptimo interior.** El rendimiento mejora hasta 17-32 y **empeora en 65**. Un kernel de 65 tramas a 40 ms por trama abarca **2.6 segundos**. Eso ya no es "contexto local": es contexto de frase, territorio de la atención. La convolución con kernel gigante paga parámetros y campo receptivo para hacer peor el trabajo que el módulo de al lado hace mejor. La existencia de este óptimo interior es evidencia empírica **directa** de la tesis de complementariedad: si local y global fueran sustitutos, el WER debería mejorar monótonamente con el kernel.
2. **El rango útil es amplio y la ganancia es chica.** De kernel 3 a kernel 32 la mejora en test-other es 4.39 → 4.29, una décima. Comparado con el 0.4-0.6 que cuesta **quitar el módulo entero**, la conclusión es que lo que importa es *tener* convolución local, no afinar su alcance.
3. **La selección se hizo en dev, correctamente.** El paper: *"On comparing the second decimal in dev WER, we find kernel size 32 to perform better than rest."* Y hace falta el segundo decimal porque kernel 7 y kernel 32 empatan en dev-other a 4.30; desempata dev-clean (1.88 vs 1.83). En test-clean, kernel 32 (2.03) es de hecho **peor** que kernel 65 (1.98) y kernel 3 (1.99) — elegir por test habría dado otra respuesta, y por eso está bien no haberlo hecho.

**Nota de implementación importante:** kernel 32 es **par**. Una convolución "same" con kernel par no admite padding simétrico: con $\text{pad}=(k-1)/2 = 15$ a cada lado la salida tiene $T-1$ tramas. El paper no discute cómo maneja esto (padding asimétrico 16/15 es lo natural). Prácticamente todas las reimplementaciones usan **kernel 31**; `torchaudio.models.Conformer` directamente **exige kernel impar** con un assert. Ver Sección 15.

## 6. La self-attention con embedding posicional relativo

El módulo (Figura 3) es corto:

```
x → LayerNorm → Multi-Head Attention con Relative Positional Embedding → Dropout → (+ residual)
```

Lo único no estándar es el esquema posicional, y es donde el paper es más económico y más importante. Sección 2.1:

> *"We employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL [20], the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention module to generalize better on different input length and the resulting encoder is more robust to the variance of the utterance length."*

### Qué reemplaza exactamente

En Vaswani et al. 2017 la posición entra **sumada a la entrada**: $x_i \leftarrow e_i + p_i$, con $p_i$ una sinusoide de la **posición absoluta** $i$. Expandiendo el score de atención antes del softmax:

$$A_{ij}^{\text{abs}} = (e_i + p_i)^\top W_q^\top W_k (e_j + p_j) = \underbrace{e_i^\top W_q^\top W_k e_j}_{\text{contenido-contenido}} + \underbrace{e_i^\top W_q^\top W_k p_j}_{\text{contenido-posición}} + \underbrace{p_i^\top W_q^\top W_k e_j}_{\text{posición-contenido}} + \underbrace{p_i^\top W_q^\top W_k p_j}_{\text{posición-posición}}$$

Transformer-XL (Dai et al., 2019, referencia [20]) reescribe esto sustituyendo cada aparición de la posición absoluta de la clave $p_j$ por una sinusoide de la **distancia relativa** $R_{i-j}$, y las apariciones de la posición absoluta de la consulta $p_i$ por vectores de sesgo **globales aprendidos** $u$ y $v$:

$$A_{ij}^{\text{rel}} = \underbrace{e_i^\top W_q^\top W_{k,E}\, e_j}_{(a)\ \text{contenido-contenido}} + \underbrace{e_i^\top W_q^\top W_{k,R}\, R_{i-j}}_{(b)\ \text{sesgo posicional dependiente del contenido}} + \underbrace{u^\top W_{k,E}\, e_j}_{(c)\ \text{sesgo de contenido global}} + \underbrace{v^\top W_{k,R}\, R_{i-j}}_{(d)\ \text{sesgo posicional global}}$$

con dos matrices de proyección de clave separadas: $W_{k,E}$ para contenido y $W_{k,R}$ para posición. El costo extra es una matriz $d \times d$ más dos vectores por cabeza. Es el $5d^2$ que puse en la tabla de la Sección 4.

### Por qué esto importa específicamente en audio

Aquí está el argumento que hay que entender, porque es el que hace que Conformer sea una arquitectura de **audio** y no un Transformer genérico al que le pegaron una convolución.

**La propiedad clave es la equivarianza traslacional del patrón de atención.** Con codificación relativa, el score entre dos tramas depende solo de su contenido y de la **distancia** entre ellas, no de dónde caen en la utterance. Desplazar la señal completa 200 ms a la derecha no cambia la matriz de atención. Con codificación absoluta sí la cambia: el mismo fonema en la trama 50 y en la trama 500 recibe modulaciones posicionales completamente distintas, y el modelo tiene que aprender por separado a reconocerlo en cada rango de posiciones. Esto es exactamente la propiedad que hace útiles a las convoluciones, importada al operador global.

**Y el audio castiga la falta de esa propiedad más que el texto**, por tres razones concretas:

1. **Variabilidad extrema de longitud.** LibriSpeech tiene utterances de 1 a 35 segundos; a 25 tramas/s eso va de 25 a 875 posiciones, un rango de 35×. Una frase de texto rara vez varía tanto en tokens. Con codificación absoluta, un modelo entrenado mayoritariamente con utterances de 8 segundos ve posiciones $>$ 200 pocas veces, y su representación de esas posiciones queda mal estimada. En inferencia, una utterance larga cae fuera de distribución en el eje posicional.
2. **No hay unidad natural de segmentación.** En NLP la oración tiene fronteras, y la posición 0 significa algo ("inicio de oración"). En audio, el corte de una utterance es un artefacto del VAD, del turno de conversación o del formato del dataset. La posición absoluta 0 en audio **no codifica ninguna información lingüística estable**: es donde el segmentador decidió cortar. Anclar la representación a ella es anclarla a ruido de pipeline. Esta es exactamente la objeción sobre segmentación del audio que aparece en la clase 39, y la codificación relativa es la respuesta arquitectónica a ella: al depender solo de $i-j$, el encoder es **invariante a dónde se puso el corte**.
3. **La estructura acústica relevante es relativa por naturaleza.** "La trama de hace 3 pasos", "el onset 200 ms atrás", "la vocal anterior a esta consonante": todas las relaciones que importan son desplazamientos, no coordenadas.

El precio es real: la codificación relativa **rompe la posibilidad de precomputar** la matriz posicional una sola vez, requiere el truco de *relative shift* de Transformer-XL para no materializar un tensor $T \times T \times d$, y agrega términos al kernel de atención. En la práctica cuesta entre 10% y 30% de tiempo por capa según la implementación (esto es conocimiento externo, el paper no reporta tiempos). El ablation dice que vale la pena: quitarla es el paso **más caro** de toda la Tabla 3, como veremos.

## 7. Las medias capas feed-forward (macaron)

El módulo FFN individual (Figura 4) es estándar salvo detalles:

```
x → LayerNorm → Linear (d → 4d) → Swish → Dropout → Linear (4d → d) → Dropout → (+ ½ residual)
```

Factor de expansión **4** (caption de la Figura 4), Swish en vez de ReLU, pre-norm, y dropout **dos veces**: después de la activación y después de la segunda proyección.

Lo distintivo es que hay **dos** de estos módulos, uno antes y uno después del par MHSA+Conv, cada uno con residual de **medio paso**.

### De dónde viene la idea

De **Macaron-Net**: Lu et al., 2019, *Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View* (referencia [18]). El argumento original no es empírico sino numérico, y vale la pena porque explica el $\tfrac{1}{2}$.

La idea de Lu et al. es leer una capa Transformer como un **paso de integración numérica** de un sistema dinámico de múltiples partículas, donde la self-attention hace de término de **difusión** (interacción entre partículas) y la FFN de término de **convección** (evolución individual de cada partícula). Un bloque Transformer estándar, que aplica primero atención y después FFN, corresponde a un esquema de separación de operadores de **Lie-Trotter**, que tiene error local de **primer orden**. La alternativa clásica en análisis numérico es la separación de **Strang-Marchuk**: medio paso del primer operador, un paso completo del segundo, medio paso del primero. Esa es de **segundo orden**.

Aplicado al Transformer, Strang-Marchuk da exactamente: media FFN → atención completa → media FFN. De ahí el nombre "macaron" (dos galletas iguales alrededor de un relleno) y de ahí los factores $\tfrac{1}{2}$ en la Ecuación 1, que **no son un hiperparámetro sino la constante que sale del esquema de integración**.

Conformer toma la estructura pero mete **dos** operadores en el relleno: atención **y** convolución.

### Qué dice el ablation

Tabla 5, Sección 3.4.3, sin LM externo, sobre el modelo grande:

| Arquitectura | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| **Conformer (macaron, medio paso)** | 1.9 | **4.4** | 2.1 | **4.3** |
| Macaron con residuales de paso completo | 1.9 | 4.5 | 2.1 | 4.5 |
| Una sola FFN (estilo Transformer) | 1.9 | 4.5 | 2.1 | 4.5 |

Traducción honesta: **0.1 en dev-other y 0.2 en test-other**, en ambos casos. Es la contribución más chica de todas las que ablaciona el paper, y el propio texto se pasa un poco al describirla como *"provides a significant improvement"* (Sección 2.4). En 0.2 de WER absoluto sobre test-other, "significativo" es generoso sin barras de error, y el paper no reporta ninguna.

Dos cosas interesantes que sí salen de ahí:

- **El medio paso importa tanto como el par.** "Full step residuals" y "single FFN" dan exactamente lo mismo (4.5/4.5). O sea: partir la FFN en dos **sin** el factor $\tfrac{1}{2}$ no aporta nada. La ganancia entera está en la combinación par + medio paso, tal como predice el argumento de Strang-Marchuk. Es un punto elegante a favor de la teoría de Lu et al.
- **Hay interacción con la convolución.** En la Tabla 5, quitar el macaron desde el Conformer completo cuesta **+0.1** en dev-other (4.4 → 4.5). En la Tabla 3, quitar el macaron **después** de haber quitado ya la convolución cuesta **+0.3** (4.8 → 5.1). La estructura macaron vale tres veces más cuando el módulo de convolución no está. Interpretación plausible: el par de FFN, al distribuir capacidad no lineal a ambos lados del bloque, compensa parcialmente lo que aportaba la convolución. El paper no comenta esta interacción; es lectura mía cruzando las dos tablas.

## 8. Los tres tamaños de modelo

Tabla 1. El caption es explícito sobre el método: *"found via sweeping different combinations and choosing the best performing models within the parameter limits"* — es decir, se fijaron presupuestos de 10M, 30M y 118M y se buscó dentro de cada uno.

| | **Conformer (S)** | **Conformer (M)** | **Conformer (L)** |
|---|---|---|---|
| Num Params (M) | 10.3 | 30.7 | 118.8 |
| Encoder Layers | 16 | 16 | 17 |
| Encoder Dim | 144 | 256 | 512 |
| Attention Heads | 4 | 4 | 8 |
| Conv Kernel Size | 32 | 32 | 32 |
| Decoder Layers | 1 | 1 | 1 |
| Decoder Dim | 320 | 640 | 640 |

Lo que se lee entre líneas:

**El escalamiento es casi puramente en ancho.** De S a M la profundidad no cambia (16 capas) y solo sube la dimensión de 144 a 256; de M a L sube una sola capa (17) y se duplica la dimensión a 512. Como los parámetros por bloque escalan $\approx 24d^2$, duplicar $d$ cuadruplica el bloque: $16 \times 24 \times 144^2 \approx 8.0$M, $16 \times 24 \times 256^2 \approx 25.2$M, $17 \times 24 \times 512^2 \approx 107$M. Eso reproduce bien la progresión 10.3 / 30.7 / 118.8 una vez sumados decoder y subsampling. **Ancho, no profundidad.** Es coherente con que la atención global ya resuelve el alcance temporal: no se necesita profundidad para agrandar el campo receptivo, como sí necesitaría una CNN pura.

**El kernel es constante en 32 para los tres tamaños**, aunque el barrido de la Tabla 7 se hizo **solo sobre el modelo grande**. Es una extrapolación no verificada: nada garantiza que 32 sea óptimo con $d=144$.

**Las cabezas: 4 / 4 / 8.** Dimensión por cabeza: $144/4 = 36$, $256/4 = 64$, $512/8 = 64$. El modelo chico se queda con cabezas anchas relativas al modelo. Y aquí hay una inconsistencia con el ablation que discuto en la Sección 14.

**El decoder es una LSTM de una capa.** Sección 3.2: *"We use a single-LSTM-layer decoder in all our models."* Este es el detalle irónico del paper y merece un párrafo. Conformer **no elimina la recurrencia del sistema**: la elimina del **encoder acústico** y la conserva en la *prediction network* del transductor RNN-T. Y tiene sentido: la prediction network modela la secuencia de **etiquetas** ya emitidas, que es corta (decenas de word-pieces, no cientos de tramas), estrictamente autorregresiva, y donde la latencia por token es más crítica que el paralelismo. La conclusión honesta de leer la Tabla 1 no es "las RNN murieron", es **"las RNN dejaron de ser el operador correcto para modelar la señal acústica"**. Es una afirmación más precisa y más defendible.

Nótese también que el decoder no es despreciable en el modelo chico: una LSTM de dimensión 320 con embedding y capa de joint aporta un par de millones sobre los 10.3M totales.

## 9. Experimentos y resultados

### Setup (Secciones 3.1 y 3.2)

- **Datos:** LibriSpeech (Panayotov et al., 2015). El paper dice *"970 hours of labeled speech"* más un corpus de texto de **800M tokens** para el LM. (Sobre las 970 horas, ver Sección 14.)
- **Features:** filterbanks de **80 canales**, ventana de **25 ms**, salto de **10 ms** → 100 tramas por segundo a la entrada.
- **Aumentación:** SpecAugment con $F = 27$, **diez** máscaras temporales, ratio máximo de máscara temporal $p_S = 0.05$ (el largo máximo de cada máscara es $p_S$ por el largo de la utterance — máscaras **proporcionales**, no absolutas, que es la variante de "SpecAugment on Large Scale Datasets", referencia [28]).
- **Regularización:** dropout $P_{drop} = 0.1$ en cada unidad residual, **ruido variacional** sobre los pesos, y regularización $\ell_2$ con peso $10^{-6}$ sobre todos los parámetros entrenables.
- **Optimización:** Adam con $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$; schedule de Transformer con **10k pasos de warm-up** y learning rate pico $0.05/\sqrt{d}$, con $d$ la dimensión del encoder. Para L eso da $0.05/\sqrt{512} \approx 2.2\times10^{-3}$.
- **Modelo de lenguaje:** LSTM de **3 capas, ancho 4096**, entrenada sobre el corpus de LM de LibriSpeech **más** las transcripciones de LibriSpeech-960h, tokenizada con un **WPM de 1k** construido sobre LibriSpeech 960h. **Perplejidad a nivel de palabra 63.9** sobre las transcripciones de dev. Se integra por **shallow fusion** con peso $\lambda$ ajustado en dev por búsqueda en grilla.
- **Framework:** Lingvo.

El paper **no reporta**: número de GPUs/TPUs, tiempo de entrenamiento, número de pasos, tamaño de batch, ni FLOPs de inferencia. Vuelve en la Sección 11.

### Resultados en LibriSpeech (Tabla 2)

Reproduzco la tabla completa tal cual, incluyendo las filas de otros métodos. Advertencia del paper: *"All our evaluation results round up to 1 digit after decimal point"* — de ahí que QuartzNet y los LAS de la literatura aparezcan con dos decimales y todo lo de Google con uno.

| Familia | Método | #Params (M) | Sin LM: test-clean | Sin LM: test-other | Con LM: test-clean | Con LM: test-other |
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

**(a) Conformer M vs Transformer Transducer — el dato central, verificado.** El paper lo afirma en la introducción: *"Our medium 30M parameters-sized model already outperforms transformer transducer published in [7] which uses 139M model parameters."*

| | Params | sin LM clean | sin LM other | con LM clean | con LM other |
|---|---|---|---|---|---|
| Transformer Transducer | 139 M | 2.4 | 5.6 | 2.0 | 4.6 |
| Conformer (M) | 30.7 M | **2.3** | **5.0** | 2.0 (empate) | **4.3** |

**Verificado y correcto**, con un matiz: en test-clean con LM hay **empate** (2.0 vs 2.0), no victoria. La afirmación se sostiene en las otras tres columnas, y la ganancia mayor está en test-other sin LM: **5.6 → 5.0, un 10.7% relativo con 4.5× menos parámetros**. Es el dato más contundente del paper y el que hay que citar.

**(b) Conformer S vs ContextNet S.** El paper afirma *"0.7% better on testother"* con 10.3M vs 10.8M: 7.0 → 6.3 sin LM. **Verificado**, es 0.7 absoluto (10% relativo). Y con LM Conformer S gana también en clean (2.1 vs 2.3) y other (5.0 vs 5.5).

**(c) Conformer L vs ContextNet L — la comparación más dura, y la más reveladora.**

| | Params | sin LM clean | sin LM other | con LM clean | con LM other |
|---|---|---|---|---|---|
| ContextNet (L) | 112.7 M | 2.1 | 4.6 | 1.9 | 4.1 |
| Conformer (L) | 118.8 M | 2.1 | **4.3** | 1.9 | **3.9** |

**En test-clean empatan exactamente**, en las dos condiciones. **Toda la ventaja de Conformer está en test-other**, el split difícil (locutores y grabaciones de peor calidad). Y Conformer L usa **6M parámetros más** que ContextNet L. Este es un resultado que el paper no destaca y que es importante para no sobrevender: contra una CNN bien diseñada con squeeze-and-excitation, la ventaja de la atención **no aparece en habla limpia y fácil**; aparece cuando la señal local es ambigua y hay que desambiguar con contexto largo. Que es, mecánicamente, exactamente lo que uno esperaría. Es una confirmación bonita de la tesis, no una debilidad.

**(d) El punto que más importa para la clase 39: Conformer S vs LAS LSTM.**

| | Params | con LM clean | con LM other |
|---|---|---|---|
| LAS con LSTM | 360 M | 2.2 | 5.2 |
| **Conformer (S)** | **10.3 M** | **2.1** | **5.0** |

Mismo dataset, misma métrica, misma condición (con LM externo), misma tabla del mismo paper. Un encoder basado en atención con **35 veces menos parámetros** le gana a la mejor arquitectura LSTM reportada. Este número, y no otro, es el que hay que llevar a la Sección 13.

**(e) El estado del arte.** Conformer L con LM (1.9/3.9) es el mejor número de toda la tabla en ambas columnas. El paper: *"our model achieves the lowest word error rate among all the existing models. This clearly demonstrates the effectiveness of combining Transformer and convolution in a single neural network."*

## 10. Ablations: la sección más valiosa

Aquí está el verdadero contenido científico del paper. Vale la pena leerla con cuidado porque el resultado principal es más matizado de lo que dice el texto.

### 10.1. Desmontar el Conformer hacia un Transformer (Tabla 3)

El diseño experimental es el correcto: se parte del bloque Conformer y se le quitan diferencias una por una hasta llegar a un bloque Transformer estándar, **manteniendo constante el número total de parámetros** (caption: *"while keeping the total number of parameters unchanged"*). Es **acumulativo**: cada fila incluye todas las remociones anteriores. Todo evaluado **sin LM externo**.

| Arquitectura | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|
| Conformer Model | 1.9 | 4.4 | 2.1 | 4.3 |
| – SWISH + ReLU | 1.9 | 4.4 | 2.0 | 4.5 |
| &nbsp;&nbsp;– Convolution Block | 2.1 | 4.8 | 2.1 | 4.9 |
| &nbsp;&nbsp;&nbsp;&nbsp;– Macaron FFN | 2.1 | 5.1 | 2.1 | 5.0 |
| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;– Relative Pos. Emb. | 2.3 | 5.8 | 2.4 | 5.6 |

Los deltas paso a paso (lo que el paper no tabula):

| Paso | Δ dev-clean | Δ dev-other | Δ test-clean | Δ test-other |
|---|---|---|---|---|
| Swish → ReLU | 0.0 | **0.0** | −0.1 | +0.2 |
| Quitar convolución | +0.2 | **+0.4** | +0.1 | +0.4 |
| Macaron → FFN única | 0.0 | **+0.3** | 0.0 | +0.1 |
| Rel-pos → posicional absoluto | +0.2 | **+0.7** | +0.3 | +0.6 |
| **Total Conformer → Transformer** | **+0.4** | **+1.4** | **+0.3** | **+1.3** |

**El resultado global:** llegar del bloque Conformer al bloque Transformer estándar, a igual número de parámetros, cuesta **1.4 puntos de WER en dev-other (4.4 → 5.8, un 32% relativo)** y 1.3 en test-other. Con parámetros congelados. Ese es el valor de la arquitectura, aislado del tamaño. Es una de las ablaciones más limpias que he visto en un paper de arquitectura.

**El ordenamiento por importancia — y una discrepancia honesta.** El paper concluye (Sección 3.4.1): *"Among all differences, convolution sub-block is the most important feature, while having a Macaron-style FFN pair is also more effective than a single FFN of the same number of parameters."*

Pero **la tabla no dice eso si se miden los deltas**. El paso más caro es el último: **quitar el embedding posicional relativo cuesta +0.7 en dev-other**, casi el doble que quitar la convolución (+0.4). Ordenando por daño:

1. **Embedding posicional relativo: +0.7 dev-other / +0.6 test-other**
2. **Módulo de convolución: +0.4 / +0.4**
3. Macaron FFN: +0.3 / +0.1
4. Swish: 0.0 / +0.2 (ruido)

¿Es una contradicción? No exactamente, y hay que ser preciso:

- La ablación es **acumulativa**, así que el efecto de la codificación relativa se mide sobre un modelo que **ya perdió la convolución y el macaron**. Es plausible que en el Conformer completo su remoción cueste menos, porque el módulo de convolución también aporta información posicional relativa (una convolución **es** un operador de desplazamiento relativo). Es decir, hay **redundancia parcial** entre los dos mecanismos, y cuando se quita uno el otro se vuelve más crítico. El paper no corre el ablation en el otro orden, así que esto no se puede zanjar con los datos publicados.
- Pero con los datos que **sí** están, la afirmación literal "la convolución es la característica más importante" **no está sostenida por la Tabla 3**. Lo defendible es: *"la convolución y la codificación posicional relativa son los dos componentes cuya remoción más daña, y juntos explican +1.1 de los +1.4 de degradación en dev-other"*.

Y hay una lectura más profunda que unifica ambos: **los dos componentes cuya remoción más duele son los dos que inyectan estructura *relativa* en el modelo.** La convolución la inyecta como operador (pesos compartidos sobre desplazamientos); la codificación posicional relativa la inyecta en el kernel de atención. El Transformer vanilla no tiene ninguno de los dos. La conclusión limpia del paper, reformulada: **lo que le falta a un Transformer para hacer ASR no es capacidad, es equivarianza traslacional**, y se la puedes dar por dos vías complementarias.

### 10.2. Cómo combinar convolución y atención (Tabla 4)

Ya la mostré en la Sección 4. Recapitulando el orden de mérito en dev-other: Conformer (4.4) < conv antes de MHSA (4.5) < lightweight conv (4.8) < paralelo (4.9).

Dos conclusiones:

- **La depthwise conv no es reemplazable por lightweight conv.** La lightweight convolution de Wu et al. 2019 (referencia [35]) es una depthwise con pesos **compartidos entre grupos de canales y normalizados por softmax sobre el eje del kernel**. Perder 0.4 en dev-other al cambiar una por otra sugiere que la normalización softmax sobre el kernel —que fuerza que los pesos del filtro sean positivos y sumen 1, convirtiéndolo en un promedio ponderado— es una restricción demasiado fuerte. Una convolución de audio necesita pesos **negativos** para actuar como detector diferencial de onsets y transiciones. Esta es interpretación mía; el paper solo reporta la cifra.
- **El paralelo pierde.** Ya comentado. Es la refutación experimental de Lite Transformer en el dominio ASR, hecha por el paper que dice inspirarse en él.

### 10.3. Macaron vs FFN única (Tabla 5)

Ya está en la Sección 7: 0.1-0.2 de mejora, y el medio paso es necesario para que el par sirva.

### 10.4. Número de cabezas de atención (Tabla 6)

Sobre el modelo grande ($d = 512$), mismo número de cabezas en todas las capas:

| Cabezas | Dim por cabeza | dev-clean | dev-other | test-clean | test-other |
|---|---|---|---|---|---|
| 4 | 128 | 1.9 | 4.6 | 2.0 | 4.5 |
| 8 | 64 | 1.9 | 4.4 | 2.1 | **4.3** |
| 16 | 32 | 2.0 | **4.3** | 2.2 | 4.4 |
| 32 | 16 | 1.9 | 4.4 | 2.1 | 4.5 |

El paper: *"We find that increasing attention heads up to 16 improves the accuracy, especially over the devother datasets."*

Observaciones:

- **El efecto es pequeño y no monótono.** Todo el rango 8-32 cae entre 4.3 y 4.4 en dev-other. Solo 4 cabezas es claramente peor (4.6). La lectura razonable: **hay un mínimo de cabezas necesario para el modelo grande y a partir de ahí da bastante igual**. Con 4 cabezas y dim 128 por cabeza, el modelo tiene pocas "vistas" independientes de la secuencia; con 32 y dim 16, cada cabeza es demasiado angosta para representar una consulta útil. El óptimo plano entre medias es el patrón habitual.
- **Inconsistencia con la Tabla 1.** El mejor dev-other es 16 cabezas (4.3), pero **Conformer L usa 8** (Tabla 1). Ver Sección 14.

### 10.5. Kernel de convolución (Tabla 7)

Ya desarrollado en la Sección 5. El punto conceptual: **óptimo interior en 17-32, degradación en 65**, que es evidencia de que la convolución debe quedarse en su nicho local y no invadir el territorio de la atención.

## 11. Limitaciones

El paper no tiene sección de limitaciones (es un Interspeech de 5 páginas). Estas salen de leerlo críticamente.

### 11.1. El costo cuadrático de la atención, cuantificado

La atención es $O(T^2)$ en tiempo y memoria. Con los números del paper se puede ver **exactamente** cuán grave es, y la respuesta es interesante: menos de lo que se teme en cómputo, más de lo que se teme en memoria.

El pipeline: filterbanks cada **10 ms** → 100 tramas/s. El *convolution subsampling* lleva de la tasa de 10 ms a la de **40 ms** (anotaciones de la Figura 1), o sea **4× de submuestreo → 25 tramas por segundo**. Entonces:

| Duración de la utterance | Tramas $T$ tras submuestreo | Entradas de la matriz de atención $T^2$ |
|---|---|---|
| 2 s | 50 | 2,500 |
| 10 s | 250 | 62,500 |
| 30 s | 750 | 562,500 |
| 60 s | 1,500 | 2,250,000 |

**Sin el submuestreo 4×, esos números se multiplicarían por 16.** El stem convolucional no es un detalle: es lo que hace tratable la atención sobre audio, y es por sí mismo una segunda instancia de "usar convolución para lo local".

**Cómputo.** El costo de atención por capa es $\approx 2T^2 d$ MACs (los productos $QK^\top$ y $AV$). El costo de las partes lineales del bloque es $\approx T \cdot P_{\text{bloque}}$, con $P_{\text{bloque}} \approx 6.3$M para $d=512$. La atención domina cuando

$$2T^2 d > T \cdot P_{\text{bloque}} \quad\Longleftrightarrow\quad T > \frac{P_{\text{bloque}}}{2d} = \frac{6.3\times10^6}{1024} \approx 6150 \text{ tramas} \approx 4 \text{ minutos}$$

O sea: para utterances de longitud normal en ASR, **el término cuadrático no es el cuello de botella de FLOPs**. A 10 segundos ($T=250$), la atención es del orden del 4% del cómputo del encoder. Esto matiza el discurso habitual: el problema del $O(T^2)$ en ASR no es el tiempo.

**Memoria, que sí es el problema.** Los pesos de atención hay que guardarlos para el backward. Para Conformer L, con 8 cabezas y 17 capas, una utterance de 30 s ($T = 750$):

$$750^2 \times 8 \times 17 = 76.5\times10^6 \text{ escalares} \approx 306\ \text{MB en fp32, por muestra del batch}$$

Multiplicado por el tamaño de batch, eso es lo que fuerza a truncar utterances, ordenar por longitud, usar *gradient checkpointing* o pasar a atención por chunks. Y crece cuadráticamente: a 60 s serían ~1.2 GB por muestra.

**Un tercer costo que se omite:** la codificación posicional relativa de Transformer-XL agrega un término $QR^\top$ y el truco de *relative shift*, lo que en la práctica aumenta tanto el tiempo como la memoria del bloque de atención respecto a atención absoluta. El paper no cuantifica esto.

### 11.2. Conformer, tal como está publicado, no es causal ni streamable

Esto es central y el paper **ni lo menciona**. Es un modelo *full-context*, offline, para transcribir utterances completas.

Dos fuentes de no causalidad, ambas cuantificables:

**(a) La self-attention es bidireccional.** Cada trama atiende a toda la utterance, pasada y futura. Sin máscara causal, sin límite de contexto derecho. Un modelo así no puede emitir nada hasta que la utterance terminó.

**(b) La convolución depthwise es centrada.** Kernel 32 con padding simétrico implica ~**15-16 tramas de contexto futuro por capa**. A 40 ms por trama son **~600-640 ms de lookahead por bloque**. Y se **acumula linealmente con la profundidad**: con 17 capas,

$$17 \times 16 \text{ tramas} \times 40 \text{ ms} \approx 10.9 \text{ segundos de contexto derecho}$$

Ese es el lookahead solo del camino convolucional, ignorando la atención. Para un sistema de dictado en tiempo real es inutilizable.

**Qué hacen las variantes causales/streaming** (esto es conocimiento externo; el paper de 2020 no lo cubre):

- **Convolución causal:** padding solo a la izquierda ($k-1$ a la izquierda, 0 a la derecha). Elimina el lookahead convolucional a costa de la mitad del campo receptivo efectivo.
- **Atención con contexto limitado / por chunks:** máscaras que restringen cada trama a $L$ tramas de contexto izquierdo y $R$ de contexto derecho, o partición de la secuencia en chunks con atención completa dentro del chunk y acceso al estado de chunks previos. Esto acota el lookahead a una constante independiente de la profundidad y hace la memoria lineal en $T$.
- **Encoders en cascada** (línea de Narayanan et al. y sucesores en Google): un encoder causal emite hipótesis con baja latencia y un segundo encoder no causal, con contexto derecho acotado, las corrige. Permite un único modelo que sirve streaming y offline.
- **BatchNorm**, ya mencionado, es problemático en streaming por la discrepancia entre estadísticas de batch y acumuladas; las variantes de producción típicamente la reemplazan.

El punto para el análisis: **la comparación de la Tabla 2 contra el Transformer Transducer no es del todo simétrica**, porque el título de aquel paper es literalmente *"a **streamable** speech recognition model"*. Conformer le gana en WER, pero no es streamable. Esa dimensión no aparece en ninguna tabla.

### 11.3. Dependencia de SpecAugment y de la pila de regularización

Conformer L es un modelo de 118.8M parámetros entrenado en ~960 horas de audio. Eso es un régimen fuertemente propenso al sobreajuste, y el paper lo enfrenta apilando **cuatro** regularizadores simultáneos: SpecAugment con diez máscaras temporales, dropout 0.1 en todos los residuales, ruido variacional sobre los pesos, y $\ell_2$ de $10^{-6}$.

**No hay ablation de ninguno.** No sabemos cuánto de los 4.3 de test-other se debe a la arquitectura y cuánto a la receta de regularización. Y hay una asimetría de comparación: SpecAugment con máscaras proporcionales a la longitud (referencia [28], "on large scale datasets") no es lo que usaban necesariamente todos los baselines de la Tabla 2. Es plausible que parte de la ventaja sea de la receta y no del bloque. El único control disponible es la Tabla 3, que mantiene todo constante salvo la arquitectura, y que muestra +1.4 en dev-other — eso sí es atribuible al bloque, y es sustancial.

### 11.4. Alcance experimental estrecho

- **Un solo dataset, un solo idioma, un solo dominio.** LibriSpeech es inglés, leído, de audiolibros, con relación señal-ruido alta y locutores que leen con dicción clara. Nada sobre habla espontánea, conversacional, con ruido de fondo, con acentos fuera de distribución, ni multilingüe. Que Conformer haya escalado después a todo eso (Sección 12) es historia posterior; el paper no lo demuestra.
- **Una sola pérdida.** Todo es RNN-T. No hay CTC, no hay attention-based encoder-decoder, no hay evaluación del encoder como extractor de features. Que el bloque sirva más allá del transductor es, otra vez, historia posterior.
- **Ninguna medida de eficiencia real.** El paper reclama "parameter efficiency" en el abstract y en la introducción, pero **no reporta FLOPs, ni latencia, ni memoria, ni tiempo de entrenamiento, ni throughput**. Los parámetros son un proxy pobre: la atención tiene pocos parámetros y mucho cómputo dependiente de $T$; la depthwise conv tiene poquísimos parámetros y también poco cómputo. Un lector no puede saber, con este paper, si Conformer M es más rápido que ContextNet M en inferencia. Esta es la omisión más seria.
- **Los detalles del stem convolucional no se especifican.** La Figura 1 dice "Convolution Subsampling" y las anotaciones de tasa (10 ms → 40 ms), pero no cuántas capas, qué kernel, qué canales. Es un componente no trivial que afecta directamente el costo de todo lo que sigue.
- **Sin varianza.** Un solo entrenamiento por configuración, sin desviaciones estándar. Diferencias de 0.1-0.2 de WER —que es la escala de los ablations de macaron y de cabezas— no son distinguibles de ruido de semilla sin repeticiones.

## 12. Impacto y legado

> Toda esta sección es **conocimiento externo al paper**, contrastado hasta donde puedo sin acceso a los PDFs originales. Lo marco donde la certeza es menor.

Conformer se convirtió, en el lapso de un par de años, en **la arquitectura por defecto del encoder acústico**. Es difícil exagerarlo: entre 2021 y 2024, si abrías un sistema de ASR de investigación o de producción, lo más probable era encontrar un encoder Conformer o un descendiente directo.

### Dónde está

- **NVIDIA NeMo.** Conformer-CTC y Conformer-Transducer son los modelos base del toolkit. **FastConformer**, su evolución interna, cambia el subsampling a 8× con convoluciones separables en profundidad (en vez de 4×), lo que reduce a la mitad las tramas que ve la atención y acelera significativamente. Las familias **Parakeet** y **Canary** de NVIDIA están construidas sobre FastConformer.
- **torchaudio.** `torchaudio.models.Conformer` está en la API estable, junto con recetas de Conformer RNN-T. Detalles en la Sección 15.
- **ESPnet.** El encoder Conformer es el estándar de facto de las recetas de ASR desde ~2020-2021 (Guo et al., *Recent developments on ESPnet toolkit boosted by Conformer*, ICASSP 2021).
- **WeNet, SpeechBrain, k2/icefall.** Todos lo implementan o implementan un descendiente.
- **Google en producción.** El **Universal Speech Model (USM)** de Google (2023) usa un encoder Conformer de ~2B parámetros preentrenado con **BEST-RQ** sobre millones de horas; **Chirp**, el modelo de ASR de Google Cloud, deriva de USM. Antes, **w2v-BERT** (Chung et al., 2021) ya usaba bloques Conformer como encoder para preentrenamiento autosupervisado. Estoy razonablemente seguro de esto, aunque no tengo los PDFs a mano para verificar conteos exactos.
- **Más allá de ASR.** El bloque se ha usado en separación de fuentes, detección de eventos sonoros, keyword spotting, mejora de habla, traducción de voz y como encoder de audio en modelos multimodales. La razón es que el sesgo inductivo "local + global sobre una secuencia densa y larga" no tiene nada de específico del reconocimiento de habla.

### Dónde **no** está: Whisper

Este punto hay que dejarlo preciso porque se cita mal con frecuencia.

**Whisper (Radford et al., OpenAI, 2022) NO usa Conformer.** Usa un **Transformer encoder-decoder estándar** al estilo Vaswani: el encoder toma un log-Mel spectrogram de 30 segundos de duración fija, lo pasa por **dos convoluciones 1D** (kernel 3, la segunda con stride 2, activación GELU) como stem de submuestreo, suma **embeddings posicionales sinusoidales absolutos**, y de ahí en adelante son bloques Transformer convencionales. Sin módulo de convolución dentro del bloque, sin codificación posicional relativa, sin macaron.

Es una decisión deliberada y muy informativa: el paper de Whisper argumenta explícitamente que quería una arquitectura **conocida y sin novedades** para que las mejoras fueran atribuibles a la escala de datos (680,000 horas de supervisión débil) y no al modelo. Y funcionó.

**La lectura correcta de esta divergencia** es que la arquitectura importa mucho en el régimen de 1,000 horas de datos etiquetados —donde vive Conformer, y donde los sesgos inductivos compran generalización— y **importa menos** en el régimen de 100,000+ horas, donde el modelo puede aprender la localidad desde datos. Es la misma historia que ViT contra CNN en visión. Conformer sigue siendo la elección correcta si entrenas con datos limitados y quieres eficiencia en parámetros; Whisper demuestra que si tienes suficiente audio, un Transformer plano alcanza. Nótese que **incluso Whisper conserva el stem convolucional**: nadie alimenta atención directamente con 3000 tramas de espectrograma.

### Los sucesores y qué corrigieron

| Modelo | Venue | Qué cambió respecto a Conformer |
|---|---|---|
| **Squeezeformer** (Kim et al.) | NeurIPS 2022 | Estructura **U-Net temporal**: submuestrea a la mitad en las capas intermedias y vuelve a subir, porque la resolución de 40 ms es innecesariamente fina en el medio de la red. Abandona el macaron y vuelve al macro-bloque estilo Transformer (MHSA → FFN → Conv → FFN). Post-LN con escalado aprendido. Subsampling con convolución separable en profundidad. Mejor relación WER/FLOPs. |
| **Branchformer** (Peng et al.) | ICML 2022 | Vuelve a la estructura **paralela** que la Tabla 4 de Conformer había descartado: una rama de atención y una rama **cgMLP** (convolutional gating MLP), fusionadas. Al hacerlo bien —ramas de ancho completo, fusión aprendida— resulta competitivo o mejor. Es la refutación del ablation 3.4.2 de Conformer. |
| **E-Branchformer** (Kim et al.) | SLT 2022 | Mejora la fusión de Branchformer con una convolución depthwise sobre las salidas concatenadas y reincorpora las FFN macaron. Superó a Conformer en varios benchmarks de ESPnet y se volvió una alternativa estándar. |
| **Zipformer** (Yao et al.) | ICLR 2024 | Encoder **multi-tasa** con estructura U-Net (varias resoluciones temporales simultáneas), **BiasNorm** en lugar de LayerNorm, activaciones **SwooshR/SwooshL**, **reutilización de los pesos de atención** entre módulos para no recomputarlos, y el optimizador **ScaledAdam**. Mejor WER con menos parámetros y menos memoria. Es el encoder de icefall/k2. |
| **FastConformer** (NVIDIA) | 2023 | Submuestreo 8× con convoluciones separables en el stem: menos tramas para la atención, ~2-3× más rápido, sin pérdida de WER. Base de Parakeet/Canary. |

El patrón común de los sucesores es revelador: **ninguno cuestiona la tesis local+global**. Todos la aceptan. Lo que atacan es (a) la **resolución temporal uniforme** —Conformer procesa las 17 capas a 25 tramas/s, lo cual es un desperdicio— y (b) el **costo de la atención**. La idea central de Conformer sobrevivió intacta; lo que se optimizó fue su implementación.

## 13. Conexión con la clase 39

Aquí está el hilo conductor completo, con las cifras verificadas de las secciones anteriores.

### (a) La tesis de la complementariedad es correcta, y Conformer la valida

La clase afirma que para audio conviene combinar operadores con **propiedades complementarias**: convolución para lo local, un mecanismo de secuencia para lo distante, capas densas para la transformación no lineal. Conformer no discute nada de eso. Lo suscribe y lo pone en el abstract: *"Transformer models are good at capturing content-based global interactions, while CNNs exploit local features effectively. In this work, we achieve the best of both worlds."*

Y lo demuestra tres veces, en tres direcciones distintas:

1. **Quitar lo local de un modelo global daña.** Tabla 3: quitar el módulo de convolución de un Conformer, a parámetros constantes, cuesta **+0.4 de WER en dev-other**.
2. **Un modelo puramente local, bien diseñado, no alcanza en lo difícil.** ContextNet L (CNN con squeeze-and-excitation, 112.7M) empata a Conformer L en test-clean pero pierde en test-other (4.6 vs 4.3 sin LM, 4.1 vs 3.9 con LM). Lo local basta para habla limpia; falla donde hay que desambiguar con contexto.
3. **Estirar lo local hasta cubrir lo global no funciona.** Tabla 7: agrandar el kernel de convolución de 32 a 65 tramas (2.6 segundos) **empeora** el WER. La convolución no puede reemplazar a la atención simplemente creciendo.

Ese tercer punto es el más fuerte y el más fácil de pasar por alto: es la demostración experimental de que los dos operadores **no son sustitutos**. Si lo fueran, la curva del kernel sería monótona.

### (b) Lo que cambió es cuál es el mejor operador global

Conformer conserva el rol de la LSTM en CLDNN y cambia la pieza que lo ocupa. Las tres razones son estructurales y se pueden enunciar con precisión.

**Razón 1: longitud del camino.** En una RNN, propagar información entre las posiciones $i$ y $j$ requiere $|i - j|$ aplicaciones de la celda recurrente: camino $O(n)$. En self-attention, cualquier par de posiciones está conectado por **una sola** operación: camino $O(1)$. Con audio a 25 tramas/s tras el submuestreo, una dependencia a 8 segundos de distancia son **200 pasos recurrentes** contra **1 producto punto**. El gradiente que debe viajar 200 aplicaciones de una LSTM se atenúa; el que viaja por un peso de atención, no.

La tabla canónica (Vaswani et al. 2017, Tabla 1), con $n$ = longitud de secuencia, $d$ = dimensión, $k$ = kernel:

| Capa | Complejidad por capa | Operaciones secuenciales | Longitud máxima de camino |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrente | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolucional (kernel $k$) | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ (dilatada) / $O(n/k)$ |

**Razón 2: paralelización del entrenamiento.** La columna "operaciones secuenciales" es la que decide qué modelos se pueden entrenar. Una LSTM sobre 250 tramas necesita 250 pasos que **no se pueden solapar**, ni con más GPUs ni con más memoria: es una dependencia de datos. Una capa de atención procesa las 250 tramas en un par de multiplicaciones de matrices. Sobre hardware que es esencialmente un multiplicador de matrices, la diferencia de throughput es de un orden de magnitud. Y esto tiene una consecuencia que va más allá de la velocidad: **hace entrenables modelos que antes no lo eran**. El encoder de 2B parámetros de USM sobre millones de horas no es una LSTM porque no podría serlo.

**Razón 3: el contexto es dinámico, no comprimido.** Una RNN comprime todo el pasado en un vector de estado de tamaño fijo. Todo lo que la trama 250 sabe de la trama 10 tiene que haber sobrevivido a 240 escrituras sobre el mismo estado. La atención no comprime: **consulta**. Cada posición formula una query y recupera lo que necesita del conjunto completo, con un patrón distinto por posición y por cabeza. Este es también el argumento que Conformer usa contra el squeeze-and-excitation de ContextNet, que promedia toda la secuencia a un vector: *"it is still limited in capturing dynamic global context as it only applies a global averaging"*. Vale igual contra el estado oculto de una RNN.

**El costo de haber cambiado.** Para ser justos, la atención pagó por esas tres ventajas: complejidad $O(n^2 d)$ contra $O(n d^2)$, la pérdida del streaming natural que la recurrencia da gratis, y la pérdida del sesgo posicional implícito —que es exactamente lo que la codificación relativa y el módulo de convolución vienen a restituir. **El bloque Conformer, leído así, es el precio que hay que pagar para usar atención en audio.** No es un Transformer con un extra: es un Transformer al que le devolvieron las dos propiedades que la recurrencia y la convolución tenían y la atención pura no.

### (c) La afirmación del slide, contrastada con datos

La clase cierra afirmando que los Transformers "no son actualmente muy populares para aplicaciones de audio", y sugiere que no modelan bien dependencias largas. Ambas cosas se pueden contrastar directamente.

**Sobre "no modelan bien dependencias largas":** es lo opuesto de lo que dice el paper y de lo que muestra la tabla. El abstract de Conformer describe a los Transformers como *"good at capturing content-based global interactions"* y la introducción como buenos *"at modeling long-range global context"*. La limitación que Conformer les atribuye es **exactamente la contraria**: *"they are less capable to extract fine-grained local feature patterns"*. La debilidad del Transformer en audio es **lo local**, no lo distante. Y el módulo de convolución de Conformer existe precisamente para eso.

**Sobre "no son populares en audio":** en 2020 la afirmación era discutible; hoy es insostenible. Los encoders acústicos de NeMo, ESPnet, torchaudio, SpeechBrain, WeNet, icefall, Whisper, wav2vec 2.0, USM y prácticamente todo lo que se publica en Interspeech/ICASSP son basados en atención. El bloque Conformer es la arquitectura de audio más replicada de la década.

**La cuantificación rigurosa, y sus límites.** El contraste directo CLDNN (2015) vs Conformer (2020) es tentador pero **no es una comparación válida**, y hay que decirlo:

- CLDNN se evaluó sobre tareas internas de Google (Voice Search de 2000 horas, habla espontánea de dominio abierto, según recuerdo del paper de Sainath et al., ICASSP 2015 — **no tengo ese PDF aquí y no verifiqué sus cifras**). Conformer se evaluó sobre LibriSpeech (audiolibros leídos, 960 horas). **Datasets distintos, dominios distintos, WER no comparables entre sí.**
- Median cinco años de diferencia en optimizadores, aumentación de datos (SpecAugment no existía en 2015), tokenización, pérdida de entrenamiento (CE con alineamiento vs RNN-T) e infraestructura. Atribuir cualquier diferencia solo a la arquitectura sería un error.

**Lo que sí es comparable, y es suficiente,** está dentro de la misma Tabla 2 del mismo paper, sobre el mismo dataset, con la misma métrica y en la misma condición:

| | Operador global | Params | test-clean (con LM) | test-other (con LM) |
|---|---|---|---|---|
| LAS con encoder **LSTM** | recurrencia | 360 M | 2.2 | 5.2 |
| **Conformer (S)** | self-attention + convolución | **10.3 M** | **2.1** | **5.0** |
| **Conformer (L)** | self-attention + convolución | 118.8 M | **1.9** | **3.9** |

**Un modelo basado en atención con 35 veces menos parámetros le gana a la mejor arquitectura LSTM de la tabla, en ambos splits.** Y el modelo grande, con un tercio de los parámetros de la LSTM, baja el WER de test-other de 5.2 a 3.9: **25% relativo**.

Ese es el contraste honesto. No prueba que las RNN sean inútiles —siguen ahí, en el decoder del propio Conformer, modelando la secuencia de etiquetas, y siguen siendo la elección natural cuando el streaming con estado acotado es un requisito duro. Prueba algo más acotado y más cierto: **para modelar la señal acústica, la self-attention con sesgos posicionales relativos desplazó a la recurrencia, y lo hizo con un margen que no admite discusión.**

Y el cierre, que es lo que hace elegante a Conformer como respuesta: **el profesor tiene razón en el diseño y en el razonamiento. La estructura CNN + [operador global] + MLP sigue siendo la arquitectura correcta para audio en 2025. Conformer es esa misma estructura con la casilla del medio actualizada.**

## 14. Erratas, matices y cosas que se citan mal

### 14.1. La confusión más frecuente: qué cifra corresponde a qué

Se cita habitualmente "Conformer: 2.1/4.3 en LibriSpeech" o "Conformer: 1.9/3.9" sin especificar nada. Ambos son de Conformer **L** y son condiciones distintas:

| Cifra | Modelo | ¿LM externo? | Fuente |
|---|---|---|---|
| **2.1 / 4.3** | Conformer L (118.8M) | **NO** | Abstract, Tabla 2 |
| **1.9 / 3.9** | Conformer L (118.8M) | **SÍ** (LSTM 3×4096, shallow fusion) | Abstract, Tabla 2 |
| **2.7 / 6.3** | Conformer **S** (10.3M) | **NO** | Abstract, Tabla 2 |
| **2.1 / 5.0** | Conformer S (10.3M) | **SÍ** | Tabla 2 |
| **2.3 / 5.0** | Conformer M (30.7M) | **NO** | Tabla 2 |
| **2.0 / 4.3** | Conformer M (30.7M) | **SÍ** | Tabla 2 |

Errores concretos que he visto:
- Atribuir 1.9/3.9 al modelo sin LM. **No.** Sin LM el mejor es 2.1/4.3.
- Citar 2.7/6.3 como "el modelo pequeño con LM". **No.** Es sin LM; con LM es 2.1/5.0.
- Decir "Conformer llega a 1.9% de WER en LibriSpeech" sin más. Es test-**clean** con LM externo; test-other es el doble.

**Además: todos los ablations (Tablas 3-7) están sin LM externo.** El caption de la Tabla 3 lo dice explícitamente. Por eso el Conformer de referencia en esas tablas aparece como 2.1/4.3 en test y 1.9/4.4 en dev, no como 1.9/3.9.

### 14.2. "Conformer M supera a un Transformer Transducer de 139M": verificado, con matiz

La afirmación es correcta en 3 de 4 columnas y **empata** en test-clean con LM (2.0 vs 2.0). No es una victoria en las cuatro. El margen real y sólido está en test-other: 5.6 → 5.0 sin LM y 4.6 → 4.3 con LM.

### 14.3. "La convolución es el componente más importante": no está sostenido por la Tabla 3

Desarrollado en la Sección 10.1. El paso de mayor degradación en el ablation acumulativo es **quitar la codificación posicional relativa** (+0.7 en dev-other) y no quitar la convolución (+0.4). La ablación es acumulativa, lo que puede explicar la diferencia, pero el paper afirma lo primero sin el experimento que lo demostraría. Formulación defendible: *convolución y codificación posicional relativa son los dos componentes críticos, y juntos explican +1.1 de los +1.4 de degradación total en dev-other*.

### 14.4. Cabezas de atención: la Tabla 1 y la Tabla 6 no concuerdan

La Sección 3.4.4 dice que subir a **16 cabezas** mejora, y la Tabla 6 lo confirma en dev-other (4.3 con 16 vs 4.4 con 8). Pero la Tabla 1 lista **8 cabezas** para Conformer L, y el baseline de las Tablas 3, 4 y 5 (dev 1.9/4.4) corresponde a la fila de 8 cabezas. Es decir: **el modelo publicado no usa la configuración que el propio ablation identifica como mejor en dev-other**. La explicación probable es que 8 cabezas gana en test-other (4.3 vs 4.4) y en dev-clean (1.9 vs 2.0), pero eso implicaría haber seleccionado mirando test, o simplemente que la Tabla 1 se fijó antes de correr el ablation. El paper no lo aclara. En cualquier caso, las diferencias están dentro del ruido esperable de una sola semilla.

### 14.5. "970 hours" de LibriSpeech

La Sección 3.1 dice *"which consists of 970 hours of labeled speech"*. LibriSpeech-960, el conjunto de entrenamiento estándar, tiene **960.85 horas**; sumando dev-clean, dev-other, test-clean y test-other (~21 horas en total) se llega a ~981. La cifra de 970 no corresponde exactamente a ninguna partición canónica. Es casi seguro un redondeo laxo. Lo que se entrena es LibriSpeech-960; lo confirma el propio paper cuando describe el LM: *"trained on the LibriSpeech language model corpus with the LibriSpeech960h transcripts added, tokenized with the 1k WPM built from LibriSpeech 960h"*.

### 14.6. Una fila rara en la Tabla 2

`LAS / Transformer [19]` (Karita et al.) reporta **2.2/5.6 sin LM** y **2.6/5.7 con LM**: el modelo de lenguaje externo **empeora** el WER en ambos splits. Eso no tiene sentido físico —shallow fusion con $\lambda$ ajustado en dev puede no ayudar, pero degradar 0.4 puntos en clean es raro. Es probablemente un error de transcripción de la tabla original o una inversión de columnas. Conviene no citar esa fila.

### 14.7. Kernel 32 es par

Discutido en la Sección 5. Una convolución "same" con kernel par requiere padding asimétrico y el paper no lo menciona. La mayoría de las reimplementaciones usan **31**; `torchaudio` lo exige impar por assert. Si alguien reporta "Conformer con kernel 31", está reproduciendo el paper correctamente, no desviándose.

### 14.8. Erratas menores de redacción

- Sección 2: *"Sections 2.1, **1**, and 2.3 introduce the self-attention, convolution, and feed-forward modules"* — debería decir 2.2.
- Sección 2.4: *"As in **Macron**-Net"* — es Macaron-Net.
- Sección 3.2: *"LibriSpeech **langauge** model corpus"*.
- Sección 3.2: la fórmula del learning rate aparece como `0.05/√d` con el símbolo de raíz desplazado en el texto extraído; en el PDF es $0.05/\sqrt{d}$.
- Sección 3.2: la constante de Adam aparece como ` = 10−9`, es $\epsilon = 10^{-9}$.

### 14.9. Lo que Conformer **no** afirma y a veces se le atribuye

- **No dice que las RNN sean inútiles.** Su propio decoder es una LSTM de una capa, en los tres tamaños.
- **No es un modelo de streaming.** Ver Sección 11.2. Las variantes causales son trabajo posterior de otros papers.
- **No propone una nueva forma de atención.** Toma la de Transformer-XL tal cual.
- **No inventa la combinación conv+atención.** Cita a Bello et al., QANet, Lite Transformer y Yang et al. como antecedentes. Su aporte es la **organización específica** —secuencial, conv después de atención, envuelta en macaron— más el ablation que la justifica.

## 15. Cómo se ve hoy

### 15.1. El módulo de convolución en PyTorch

Implementación directa de la Figura 2, sin dependencias más allá de `torch.nn`.

```python
import torch
import torch.nn as nn


class ConvolutionModule(nn.Module):
    """Módulo de convolución del bloque Conformer (Gulati et al., 2020, Fig. 2).

    LayerNorm -> Pointwise(d->2d) -> GLU -> Depthwise(k) -> BatchNorm
              -> Swish -> Pointwise(d->d) -> Dropout
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        # El paper usa kernel 32 (par), que no admite padding "same" simetrico.
        # Las reimplementaciones usan 31; torchaudio directamente lo exige impar.
        assert kernel_size % 2 == 1, "usa kernel impar para padding simetrico"

        self.layer_norm = nn.LayerNorm(d_model)                       # pre-norm del residual
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)     # factor de expansion 2
        self.glu = nn.GLU(dim=1)                                      # 2d -> d, compuerta multiplicativa
        self.depthwise_conv = nn.Conv1d(                              # k*d params en vez de k*d^2
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,                                           # <- esto es lo que la hace depthwise
        )
        self.batch_norm = nn.BatchNorm1d(d_model)                     # el paper: "to aid training deep models"
        self.activation = nn.SiLU()                                   # SiLU == Swish con beta=1
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)         # vuelve a mezclar canales
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D). pad_mask: (B, T) con True en las posiciones de padding.
        x = self.layer_norm(x)
        x = x.transpose(1, 2)                                          # (B, D, T) para Conv1d

        # Critico: anular el padding ANTES de convolucionar y de BatchNorm.
        # Si no, el kernel de 31 tramas arrastra basura hacia adentro y las
        # estadisticas de BatchNorm se contaminan con posiciones inexistentes.
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.glu(self.pointwise_conv1(x))                          # (B, 2D, T) -> (B, D, T)
        x = self.depthwise_conv(x)
        x = self.activation(self.batch_norm(x))
        x = self.pointwise_conv2(x)
        return self.dropout(x.transpose(1, 2))                         # (B, T, D)
```

### 15.2. El bloque completo

```python
class FeedForwardModule(nn.Module):
    """FFN de Conformer (Fig. 4): pre-norm, expansion 4, Swish, dropout doble."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    """Bloque Conformer completo. Implementa la Ecuacion (1) del paper.

        x~  = x  + 1/2 FFN(x)
        x'  = x~ +     MHSA(x~)
        x'' = x' +     Conv(x')
        y   = LayerNorm(x'' + 1/2 FFN(x''))
    """

    def __init__(self, d_model=512, num_heads=8, kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, dropout=dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        # En el paper esto es MHSA con embedding posicional RELATIVO (Transformer-XL).
        # nn.MultiheadAttention no lo implementa: es un placeholder estructural.
        # Sustituir por una RelPositionMultiHeadedAttention (ESPnet/torchaudio/NeMo).
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)                       # post-LN por bloque (inusual)

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = x + 0.5 * self.ffn1(x)                                    # medio paso (Strang-Marchuk)

        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, key_padding_mask=pad_mask, attn_mask=attn_mask)
        x = x + self.attn_dropout(h)                                  # residual de paso COMPLETO

        x = x + self.conv(x, pad_mask)                                # residual de paso COMPLETO

        x = x + 0.5 * self.ffn2(x)                                    # medio paso
        return self.final_norm(x)
```

Tres cosas que este esqueleto deja explícitas y que se pierden al leer el diagrama:

- **Los $\tfrac12$ están solo en las FFN.** MHSA y Conv llevan residual de paso completo. Es un error frecuente ponerlos en los cuatro.
- **`nn.MultiheadAttention` no sirve para reproducir el paper.** Implementa atención con posición absoluta o sin posición; la codificación relativa de Transformer-XL requiere una implementación propia con las dos proyecciones de clave, los sesgos $u,v$ y el *relative shift*. Y según la Tabla 3, es el componente cuya ausencia más degrada. Usar `nn.MultiheadAttention` y llamarlo Conformer es reproducir la fila "– Relative Pos. Emb." del ablation.
- **La máscara de padding importa dos veces** en el módulo de convolución: para no arrastrar posiciones inexistentes por el kernel de 31 tramas, y para no contaminar las estadísticas de BatchNorm.

### 15.3. `torchaudio.models.Conformer`

`torchaudio` trae una implementación mantenida:

```python
from torchaudio.models import Conformer

encoder = Conformer(
    input_dim=512,
    num_heads=8,
    ffn_dim=2048,                  # = 4 * input_dim, la expansion del paper
    num_layers=17,                 # Conformer L
    depthwise_conv_kernel_size=31, # impar por assert; el paper usa 32
    dropout=0.1,
    use_group_norm=False,          # True cambia BatchNorm por GroupNorm
    convolution_first=False,       # True pone Conv antes de MHSA (la fila peor de la Tabla 4)
)

x = torch.randn(2, 250, 512)                 # 250 tramas a 40 ms = 10 s
lengths = torch.tensor([250, 180])           # longitudes reales, para el enmascarado
y, out_lengths = encoder(x, lengths)         # (2, 250, 512)
```

Notas de uso, importantes si el objetivo es reproducir el paper:

- **Es solo el encoder.** No incluye el *convolution subsampling* (el 10 ms → 40 ms), ni el decoder LSTM, ni la pérdida RNN-T. Para el sistema completo hay que mirar `torchaudio.models.RNNTBeamSearch`, `torchaudio.models.emformer_rnnt_*` o las recetas de `torchaudio` / NeMo / ESPnet.
- **`depthwise_conv_kernel_size` debe ser impar.** El 32 del paper no se puede pasar.
- **La API expone las dos decisiones que el paper ablacionó:** `convolution_first` y `use_group_norm`. Que estén ahí como flags dice bastante sobre cuáles fueron los puntos de fricción reales en las reimplementaciones.
- **Verificar la variante de atención posicional.** Si el objetivo es reproducir los números del paper, hay que confirmar que la implementación use codificación posicional relativa y no absoluta; según la Tabla 3, la diferencia son ~0.6-0.7 puntos de WER en los splits difíciles, que es más que toda la contribución del módulo de convolución.
- **Para producción hoy**, la elección razonable no es este bloque tal cual sino un descendiente: FastConformer si importa el throughput, Zipformer si importa memoria y WER, E-Branchformer si el ecosistema es ESPnet. Todos comparten la tesis; ninguno comparte la implementación literal de 2020.
