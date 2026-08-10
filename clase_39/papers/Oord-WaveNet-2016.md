# WaveNet: A Generative Model for Raw Audio — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** Aäron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu.
- **Afiliación:** **Google DeepMind, Londres**. Heiga Zen figura con daga (†) en **Google, Londres** — no en DeepMind. Ese detalle no es cosmético: Zen es el autor de casi toda la literatura de *statistical parametric speech synthesis* que el paper usa como baseline (Zen et al. 2009, 2013, 2016), y su presencia explica por qué la evaluación de TTS está hecha con el rigor de un equipo de producción de síntesis y no con el rigor típico de un paper de generative models.
- **Estado editorial:** **preprint de arXiv, `arXiv:1609.03499v2 [cs.SD]`, 19 de septiembre de 2016**. Nunca se publicó en una conferencia con revisión por pares. Es uno de los papers más citados de la década que jamás pasó por un comité de programa — hecho relevante para leer sus omisiones metodológicas (ver §13).
- **Linaje arquitectónico:** WaveNet es explícitamente "un modelo generativo de audio basado en la arquitectura **PixelCNN**" (§1). Cuatro de los nueve autores firman también PixelRNN (van den Oord et al. 2016a) o Gated PixelCNN (2016b). Esto es traslado de una receta probada en imágenes al dominio del audio, no una invención desde cero.

**Qué hace.** Modela la forma de onda cruda, muestra a muestra, con una pila de **convoluciones causales dilatadas** y una salida **softmax categórica de 256 valores** sobre audio cuantizado con **compansión $\mu$-law**. Sin vocoder, sin ventana de análisis, sin supuesto gaussiano, sin filtro lineal.

**Cifras ancla (Tabla 1, MOS de naturalidad en escala 1–5, 100 frases no vistas, evaluación ciega y *crowdsourced*, 8 sujetos por estímulo):**

| Sistema | Inglés norteamericano | Chino mandarín |
|---|---|---|
| LSTM-RNN paramétrico (Zen et al. 2016) | 3.67 ± 0.098 | 3.79 ± 0.084 |
| Concatenativo *unit selection* dirigido por HMM (Gonzalvo et al. 2016) | 3.86 ± 0.137 | 3.47 ± 0.108 |
| **WaveNet (L+F)** | **4.21 ± 0.081** | **4.08 ± 0.085** |
| Natural, 8-bit $\mu$-law | 4.46 ± 0.067 | 4.25 ± 0.082 |
| Natural, 16-bit linear PCM | 4.55 ± 0.075 | 4.21 ± 0.071 |

**Cierre de la brecha con el habla natural (§3.2, texto):** de **0.69 a 0.34 (51%)** en inglés norteamericano y de **0.42 a 0.13 (69%)** en mandarín. Ambas cifras se calculan contra la fila *Natural (16-bit linear PCM)* y contra el **mejor baseline de cada idioma**, que no es el mismo sistema en los dos casos (concatenativo en inglés, paramétrico en mandarín). El pie de la Tabla 1 lo resume como "reduciendo la brecha en más del 50%".

**Los otros tres resultados**, que casi nadie recuerda: generación libre multi-hablante sobre VCTK con 109 hablantes en un solo modelo (§3.1); **18.8 PER en TIMIT** entrenando directamente sobre audio crudo (§3.4); y modelado de música sobre MagnaTagATune y un dataset propio de piano de YouTube (§3.3).

---

## 2. Contexto: el TTS antes de 2016

El Apéndice A del paper —escrito casi con seguridad por Zen— es la mejor síntesis breve del estado del arte pre-WaveNet que existe, y conviene leerlo como el verdadero planteamiento del problema.

Un TTS clásico se parte en dos: **análisis de texto** (segmentación de oraciones y palabras, normalización, POS tagging, grafema-a-fonema) que produce una secuencia de fonemas con contextos lingüísticos; y **síntesis de voz**, que toma esa secuencia y produce una onda. La segunda mitad tenía dos escuelas.

**Concatenativa (*unit selection*).** Construye la elocución pegando fragmentos de habla real grabada, eligiendo unidades de una base de datos grande según un costo de objetivo y un costo de concatenación (Hunt & Black 1996; Moulines & Charpentier 1990). Como cada trozo es audio genuino, la calidad segmental es excelente. Sus límites son estructurales: (i) necesita una base de datos enorme y por lo tanto una huella de memoria enorme; (ii) es rígida — para cambiar la voz, el estilo o la emoción hay que grabar de nuevo; (iii) las junturas entre unidades producen discontinuidades audibles cuando la base no cubre bien el contexto.

**Paramétrica estadística (SPSS).** Extrae del habla una secuencia de parámetros de vocoder $\mathbf{o} = \{o_1,\dots,o_N\}$ —típicamente uno cada 5 ms, con cepstra o pares de líneas espectrales para el tracto vocal, más $F_0$ y aperiodicidad para la fuente glotal—, entrena un modelo generativo $\hat{\Lambda} = \arg\max_\Lambda p(\mathbf{o}\mid \mathbf{l},\Lambda)$ sobre esos parámetros dadas las features lingüísticas $\mathbf{l}$, y en síntesis genera $\hat{\mathbf{o}} = \arg\max_{\mathbf{o}} p(\mathbf{o}\mid \mathbf{l},\hat\Lambda)$ y reconstruye la onda con un vocoder. Es compacta y flexible, pero suena peor. Zen et al. (2009) identificaron tres causas: **calidad del vocoder** (produce artefactos), **precisión del modelo generativo** y el **sobresuavizado** (estas dos producen el sonido "amortiguado", *muffled*).

El diagnóstico de fondo, en el Apéndice A: el enfoque paramétrico es una **optimización en dos pasos y por lo tanto subóptima** — primero se ajusta un modelo generativo de la señal para extraer parámetros, después se modela la trayectoria de esos parámetros con un modelo *distinto*. Y los modelos generativos convencionales de audio arrastran tres supuestos, los tres falsos:

1. **Ventana de análisis de longitud fija.** Se asume proceso estocástico estacionario dentro de ventanas de 20–30 ms con salto de 5–10 ms. Pero hay fonos —las oclusivas— que duran **menos de 20 ms** (Rabiner & Juang 1993). La ventana es más larga que el evento que pretende describir.
2. **Filtro lineal.** Modelos LTI dentro de la ventana, cuando la relación entre muestras sucesivas puede ser fuertemente no lineal.
3. **Proceso gaussiano.** Desde el modelo fuente-filtro, equivale a asumir que la excitación glotal es una muestra gaussiana; junto con la linealidad, implica que el habla es normalmente distribuida. No lo es.

La predicción lineal clásica es el caso canónico: $x_t = \sum_{p=1}^{P} a_p x_{t-p} + \epsilon_t$ con $\epsilon_t \sim \mathcal{N}(0,G^2)$ (ecuaciones 6 y 7 del apéndice). Un modelo autorregresivo de la onda cruda **ya existía desde los años 70** — lineal, gaussiano, de orden $P$ pequeño. WaveNet es literalmente el mismo objeto con las tres restricciones levantadas: no lineal, categórico, con memoria de miles de muestras.

**¿Por qué se consideraba inviable generar la onda directamente?** Por aritmética. El habla de banda ancha necesita al menos **16.000 muestras por segundo** (§1); música, 44.100. Un modelo autorregresivo sobre la onda debe encadenar **16.000 predicciones por segundo de audio**, cada una condicionada a todas las anteriores. Dos problemas distintos se confunden bajo esa cifra:

- **El problema de contexto.** Para modelar prosodia hacen falta cientos de milisegundos, es decir **miles de muestras de memoria**. Una RNN sobre 16.000 pasos por segundo es un horizonte de BPTT brutal; una CNN densa necesitaría profundidad o filtros absurdos. Este es el problema que WaveNet resuelve, y lo resuelve con dilatación.
- **El problema de generación.** Las 16.000 predicciones son estrictamente secuenciales. Este problema WaveNet **no lo resuelve**; lo hereda y lo empeora (§10).

La distinción es la clave del paper: el abstract dice que el modelo "puede ser **entrenado eficientemente** con datos de decenas de miles de muestras por segundo". Dice *entrenado*, no *ejecutado*.

Hay que registrar también que había intentos previos de modelar la onda con redes neuronales. Tokuda & Zen (2015, 2016) integraron un proceso gaussiano no estacionario de la señal con un generador LSTM-RNN y los optimizaron conjuntamente por backprop. Podían aproximar habla natural, pero su naturalidad segmental resultó **significativamente peor** que el modelo no integrado, por sobregeneralización y sobreestimación de las componentes de ruido. Ese fracaso es el contexto inmediato: no bastaba con "poner una red sobre la onda"; había que cambiar la forma de la distribución de salida.

---

## 3. La contribución central: un modelo autorregresivo sobre la muestra cruda

La factorización es la de siempre en modelos autorregresivos, aplicada a la muestra de audio (ecuación 1, §2):

$$p(\mathbf{x}) = \prod_{t=1}^{T} p(x_t \mid x_1, \dots, x_{t-1})$$

Con $T = 16000$ para un segundo de habla. No hay variable latente, no hay marginalización aproximada, no hay cota inferior.

**Qué significa "totalmente probabilístico y tratable".** El paper subraya en §2 que "como las log-verosimilitudes son **tratables**, ajustamos hiperparámetros en un conjunto de validación y podemos medir fácilmente si el modelo está sobreajustando o subajustando". Vale la pena desempacar la afirmación, porque en 2016 era el argumento de venta de la familia PixelRNN/PixelCNN frente a VAEs y GANs:

- La verosimilitud es **exacta**, no una ELBO. No hay brecha entre lo que se optimiza y lo que se quiere.
- La **factorización es la red misma**: gracias a la causalidad, un solo pase hacia adelante sobre la secuencia completa produce simultáneamente los $T$ factores condicionales. Entrenar es una única pasada de *teacher forcing* con cross-entropy sobre todos los timesteps en paralelo.
- Se puede **comparar modelos** con un número, y detectar sobreajuste sin evaluación subjetiva. En un dominio donde la métrica real es un MOS con 8 humanos por estímulo, tener una señal de validación barata es lo que hace posible iterar.

Lo que no significa: la verosimilitud tratable **no** es una buena predictora de la calidad perceptual. El paper nunca reporta un número de log-verosimilitud, ni en bits por muestra ni en nats, en ninguno de los cuatro experimentos. Todo el argumento empírico descansa en MOS y pruebas de preferencia. La tractabilidad se usa como herramienta interna de desarrollo, no como resultado.

**Por qué softmax categórica y no una salida continua.** Esta es la decisión de diseño más importante después de la dilatación, y el paper la justifica en §2.2. La alternativa natural es una densidad continua: una *mixture density network* (Bishop 1994) o una mezcla de gaussianas condicionales con escala (MCGSM, Theis & Bethge 2015). El argumento es que van den Oord et al. (2016a) mostraron que "una distribución softmax tiende a funcionar mejor, **incluso cuando los datos son implícitamente continuos**", y la razón declarada es:

> "una distribución categórica es más flexible y puede modelar más fácilmente distribuciones arbitrarias porque **no hace ningún supuesto sobre su forma**".

Traducido al dominio: la distribución de $x_t$ dado el pasado en una señal de voz es **fuertemente multimodal**. En un instante de silencio antes de una oclusiva, la muestra siguiente es prácticamente determinista y concentrada cerca de cero; en el inicio de una explosión de /p/ o /t/, la señal salta con signo y amplitud que dependen de detalles no observables; durante un fricativo, la muestra siguiente es esencialmente ruido de banda ancha. Una gaussiana condicional está obligada a resumir todo eso en media y varianza: si la verdadera condicional es bimodal, la gaussiana pone masa en el valle entre modos, que es exactamente donde no hay señal real. Eso es el sobresuavizado del párrafo anterior, formulado como problema de familia paramétrica.

Una mezcla de $K$ componentes mitiga el problema pero exige elegir $K$, sufre de colapso de componentes y de inestabilidades numéricas en las varianzas. La categórica de 256 clases es una **mezcla no paramétrica de 256 deltas**: puede representar bimodalidad, asimetría, colas pesadas y masas puntuales sin ningún hiperparámetro de forma.

El costo de la decisión: la softmax **descarta el orden** de los valores. Para la función de pérdida, confundir el nivel 130 con el 131 cuesta exactamente lo mismo que confundirlo con el 5. El modelo tiene que **aprender la métrica del espacio de amplitudes desde los datos** en vez de recibirla gratis. Es un desperdicio de estructura, y funciona igual — un patrón que se repite en toda la familia PixelCNN. La familia posterior de trabajos revirtió parcialmente esta decisión: PixelCNN++ (Salimans et al. 2017) y luego Parallel WaveNet reemplazaron la categórica por una **mezcla de logísticas discretizadas**, que recupera el orden ordinal y permite subir a 16 bits sin explotar el número de clases.

---

## 4. La cuantización $\mu$-law

Audio crudo típico es PCM lineal de **16 bits**: $2^{16} = 65.536$ valores posibles por muestra. Una softmax sobre 65.536 clases, por timestep, sobre secuencias de decenas de miles de pasos, es inviable en 2016 por tres razones acumuladas: la capa de salida $C_{\text{skip}} \times 65536$ domina el conteo de parámetros; el softmax y su gradiente dominan el cómputo por muestra; y la mayoría de las 65.536 clases se observa un número irrisorio de veces en el corpus, con lo cual la estimación es pésima.

La solución del paper (§2.2) es **compansión $\mu$-law** (ITU-T G.711, 1988) seguida de cuantización uniforme a 256 niveles:

$$f(x_t) = \operatorname{sign}(x_t)\,\frac{\ln\!\left(1 + \mu\,|x_t|\right)}{\ln\!\left(1 + \mu\right)}, \qquad -1 < x_t < 1,\ \ \mu = 255$$

La fórmula tal como aparece en el paper describe solo la **compansión**: mapea $[-1,1]$ en $[-1,1]$ de forma monótona, impar y no lineal. La cuantización propiamente tal es el paso siguiente, que el paper describe en prosa ("y luego la cuantizamos a 256 valores posibles") sin escribir:

$$q_t = \operatorname{clip}\left(\left\lfloor \frac{f(x_t)+1}{2}\cdot 256 \right\rfloor,\ 0,\ 255\right)$$

y la inversa, necesaria para reconstruir la onda desde los índices generados:

$$x = \operatorname{sign}(y)\,\frac{1}{\mu}\left[(1+\mu)^{|y|} - 1\right], \qquad y = \frac{2q+1}{256} - 1$$

**El rol de $\mu = 255$.** No es un número arbitrario: es $2^8 - 1$, el estándar de G.711 para codificación telefónica a 8 bits. Controla la **agresividad de la compresión logarítmica**. Con $\mu \to 0$ la transformación tiende a la identidad (cuantización lineal); con $\mu$ grande, la curva se aplana cerca de la saturación y se empina cerca de cero. La derivada lo cuantifica exactamente:

$$f'(x) = \frac{\mu}{(1+\mu|x|)\,\ln(1+\mu)}$$

Con $\mu = 255$: $\ln(256) = 5.545$, de modo que $f'(0) = 255/5.545 = 45.99$ y $f'(1) = 255/(256 \cdot 5.545) = 0.180$. **La razón es exactamente $1 + \mu = 256$.** Esto es lo que hace la compansión: el paso de cuantización en el dominio original es **256 veces más fino cerca del silencio que cerca de la saturación**.

Las cifras concretas, que valen más que la intuición. El paso uniforme en el dominio companded es $\Delta y = 2/256 = 1/128$. Traducido al dominio original:

| Región | $f'(x)$ | $\Delta x = \Delta y / f'(x)$ | Resolución efectiva (bits lineales) |
|---|---|---|---|
| Cerca de cero ($x \approx 0$) | 45.99 | $1.70\times10^{-4}$ | $\log_2(2/\Delta x) \approx 13.5$ |
| Mitad de escala ($x = 0.5$) | 0.354 | $2.21\times10^{-2}$ | $\approx 6.5$ |
| Escala completa ($x = 1$) | 0.180 | $4.34\times10^{-2}$ | $\approx 5.5$ |

Con 8 bits de índice, $\mu$-law entrega **~13.5 bits de resolución efectiva donde la señal es débil** y ~5.5 bits donde es fuerte. La cuantización lineal de 8 bits entregaría 8 bits uniformes en todas partes, es decir $\Delta x = 2/256 = 7.8\times10^{-3}$ constante — **46 veces más grueso que $\mu$-law en la zona de baja amplitud**.

**Por qué eso es lo correcto para audio y no para, digamos, coordenadas geométricas.** Tres razones que se refuerzan:

1. **El oído es aproximadamente logarítmico.** La discriminación de intensidad sigue la ley de Weber: el mínimo cambio detectable es proporcional al nivel, no absoluto. Un error de cuantización de magnitud $\epsilon$ es inaudible sobre una vocal fuerte y perfectamente audible sobre una consonante débil. Un cuantizador con paso proporcional a la amplitud produce **SNR de cuantización aproximadamente constante** en todo el rango dinámico, que es la propiedad perceptual que se quiere.
2. **La distribución de amplitudes del habla está fuertemente concentrada cerca de cero.** El habla tiene un factor de cresta alto: silencios, pausas entre palabras, cierres oclusivos y colas de decaimiento ocupan la mayor parte del tiempo. Un cuantizador lineal de 8 bits asigna la mitad de sus niveles a amplitudes que casi nunca ocurren; $\mu$-law asigna los niveles donde está la masa de probabilidad. En términos de teoría de la información, la compansión aproxima la cuantización óptima para una fuente de distribución no uniforme.
3. **Efecto de segundo orden sobre el aprendizaje.** Como el modelo predice una **categórica**, la distribución marginal sobre las 256 clases importa. Con cuantización lineal, la mayoría de las muestras caería en un puñado de bins centrales y la mayor parte del vocabulario de salida quedaría prácticamente sin entrenar. $\mu$-law **ecualiza el uso de las clases**, lo que mejora la estimación de la softmax.

**Qué distorsión introduce.** El paper afirma que "esta cuantización no lineal produce una reconstrucción significativamente mejor que un esquema de cuantización lineal simple. Especialmente para habla, encontramos que la señal reconstruida sonaba **muy similar** a la original". Pero el propio paper mide el costo, y es la fila más subestimada de la Tabla 1: **Natural (8-bit $\mu$-law) = 4.46 vs Natural (16-bit linear PCM) = 4.55** en inglés. La compansión cuesta **0.09 MOS**. Ese es el **techo real** de WaveNet: como el modelo genera en el dominio de 8 bits $\mu$-law, no puede superar 4.46 por construcción. Su 4.21 está a **0.25 de su techo alcanzable**, no a 0.34 del habla humana. Y en mandarín la anomalía es aún más llamativa: 8-bit $\mu$-law puntuó **4.25** contra 4.21 del 16-bit PCM — es decir, la versión degradada obtuvo mejor MOS que el original. Dentro del ruido de los intervalos de confianza (±0.082 y ±0.071), pero suficiente para tomar con pinzas cualquier lectura fina de esas dos filas.

El Apéndice B agrega el detalle que hace justa la comparación: los baselines LSTM y concatenativo se construyeron desde los datasets en **16-bit linear PCM**, mientras que WaveNet se entrenó desde los mismos datasets en **8-bit $\mu$-law**. WaveNet compite con una mano atada, y aun así gana.

Y una nota del propio paper que suele omitirse: WaveNet "incorpora casi ningún conocimiento previo sobre las señales de audio, **excepto la elección del campo receptivo y la codificación $\mu$-law de la señal**" (Apéndice A). Los dos únicos sesgos inductivos declarados del modelo son precisamente los dos temas de este análisis.

---

## 5. Convoluciones causales

Una convolución es **causal** cuando la salida en el instante $t$ depende solo de entradas en instantes $\le t$. En el paper (§2.1): "usando convoluciones causales nos aseguramos de que el modelo no pueda violar el orden en el que modelamos los datos: la predicción $p(x_{t+1}\mid x_1,\dots,x_t)$ emitida en el timestep $t$ no puede depender de ninguno de los timesteps futuros".

**Por qué no es negociable.** La factorización de la ecuación (1) no es un detalle de notación: es un **contrato**. Si la red que computa el factor $p(x_t \mid x_{<t})$ tuviera acceso a $x_t$, el modelo aprendería la identidad, la log-verosimilitud de entrenamiento sería perfecta y el muestreo produciría basura, porque en generación ese acceso no existe. El fallo es silencioso: la pérdida de entrenamiento *y* la de validación caen a casi cero, así que el diagnóstico habitual no lo detecta. La única señal es que las muestras no suenan a nada. En la práctica de implementación este es el bug número uno de cualquier WaveNet casera, y se detecta con un test específico (ver §14).

**Cómo se implementa.** El paper es explícito y pragmático: "Para imágenes, el equivalente de una convolución causal es una **convolución enmascarada**, que puede implementarse construyendo un tensor de máscara y multiplicándolo elemento a elemento con el kernel antes de aplicarlo. Para datos 1-D como audio uno puede implementar esto más fácilmente **desplazando la salida de una convolución normal unos pocos timesteps**".

La forma concreta y estándar es el **padding asimétrico**: para un kernel de tamaño $k$ y dilatación $d$, se rellenan $(k-1)\cdot d$ ceros **solo a la izquierda** de la secuencia y se aplica la convolución sin padding. La salida tiene la misma longitud que la entrada y cada posición $t$ ve exactamente $\{t-(k-1)d,\dots,t\}$. Equivalentemente, se puede aplicar `padding=(k-1)*d` simétrico de PyTorch y luego recortar los últimos $(k-1)d$ elementos — que es literalmente "desplazar la salida". Ambas son la misma operación; la primera evita computar y descartar.

Un matiz importante sobre la **primera capa**: la convolución causal que toca directamente $x$ debe excluir $x_t$ de la predicción de $x_t$. La convención usual es alimentar la red con $x$ desplazada un paso (predecir $x_t$ desde $x_{<t}$) o hacer la primera capa estrictamente causal con máscara tipo "A" de PixelCNN, dejando las capas superiores en máscara tipo "B" — la distinción viene directo de PixelRNN. Las capas ocultas *sí* pueden ver su propia posición, porque esa posición ya solo contiene información de $x_{<t}$.

**El paper también dice explícitamente que no hay pooling y que la resolución se preserva** (§2): "No hay capas de pooling en la red, y la salida del modelo tiene la misma dimensionalidad temporal que la entrada". Esto es lo que permite que un solo pase produzca los $T$ condicionales alineados.

**Por qué no una RNN.** El paper da la razón en una frase (§2.1): "Como los modelos con convoluciones causales no tienen conexiones recurrentes, típicamente son **más rápidos de entrenar que las RNN, especialmente cuando se aplican a secuencias muy largas**". Vale la pena desarrollar por qué, y ser preciso sobre en qué régimen la ventaja aplica:

| | Convolución causal | RNN (LSTM/GRU) |
|---|---|---|
| **Entrenamiento** | Todos los timesteps en **paralelo**: un solo pase, $O(1)$ pasos secuenciales, $O(T)$ trabajo total | **Secuencial** por construcción: $O(T)$ pasos dependientes, BPTT sobre $T=16.000$ por segundo |
| **Generación** | **Secuencial**, un pase de red por muestra | **Secuencial**, un paso de celda por muestra |
| **Memoria** | Estado explícito y finito: el campo receptivo | Estado latente comprimido de tamaño fijo |
| **Contexto** | Acotado y **declarado por diseño** (el campo receptivo) | En principio ilimitado, en la práctica limitado por gradientes |
| **Gradiente** | Camino de $O(\log_{\text{base}} R)$ capas hasta cualquier entrada | Camino de $O(T)$ pasos; el problema clásico de dependencias largas |

Las dos primeras filas son la asimetría clave. **En entrenamiento la convolución causal es paralela y la RNN no**, porque en teacher forcing la ground truth completa es conocida, y la convolución solo lee entradas, nunca su propia salida anterior. La RNN, aunque conozca todas las entradas, no puede computar $h_t$ sin haber computado $h_{t-1}$. Sobre 16.000 pasos por segundo de audio, esa diferencia es la que decide si el experimento es factible.

**En generación no hay ventaja alguna.** El paper lo admite en la misma sección: "Cuando se genera con el modelo, las predicciones son secuenciales: después de que cada muestra es predicha, se realimenta a la red para predecir la siguiente". Peor todavía: la RNN gasta un paso de celda por muestra, mientras WaveNet gasta **una pasada de toda la red** (decenas de capas) por muestra. En inferencia ingenua la convolución es estrictamente peor. Ese es el tema de §10.

La cuarta fila es un punto que se suele leer al revés. Que el contexto sea **acotado** parece una desventaja frente a la memoria "ilimitada" de una LSTM, pero es lo que hace el diseño analizable: el campo receptivo es un número que se calcula, se elige y se justifica. Y la evidencia empírica del propio paper es que la LSTM no aprovechaba su horizonte teórico — §3.4 lo dice sin rodeos: "con WaveNets hemos mostrado que las capas de convoluciones dilatadas permiten que el campo receptivo crezca de manera **mucho más barata** que usando unidades LSTM".

El límite de la convolución causal simple, y la transición al tema siguiente, está en la frase del paper: "Uno de los problemas de las convoluciones causales es que requieren **muchas capas, o filtros grandes**, para aumentar el campo receptivo. Por ejemplo, en la Fig. 2 el campo receptivo es solo 5 (= #capas + longitud del filtro − 1)". Cuatro capas convolucionales con kernel 2 dan un campo receptivo de 5 muestras: **0.3 milisegundos** a 16 kHz.

---

## 6. Convoluciones dilatadas

Esta es la sección que justifica el slide de la clase, así que la desarrollo con detalle.

### 6.1. Definición

Una convolución dilatada —también llamada *à trous* ("con agujeros"), término de la literatura de wavelets (Holschneider et al. 1989; Dutilleux 1989)— aplica el filtro sobre un área mayor que su longitud, saltándose valores de entrada con un paso fijo. En su versión causal 1-D, con kernel $k$ de longitud $K$ y factor de dilatación $d$:

$$(x *_d k)[i] \;=\; \sum_{j=0}^{K-1} k[j]\; x\!\left[i - d\cdot j\right]$$

Con $d = 1$ se recupera la convolución causal estándar; el paper lo señala explícitamente ("como caso especial, la convolución dilatada con dilatación 1 produce la convolución estándar"). La Figura 3 del paper visualiza una pila con dilataciones 1, 2, 4 y 8.

Dos caracterizaciones equivalentes, ambas útiles:

- **Como filtro inflado con ceros.** Es "equivalente a una convolución con un filtro más grande derivado del filtro original **dilatándolo con ceros**, pero es significativamente más eficiente". El filtro efectivo tiene longitud $1 + (K-1)d$ pero solo $K$ pesos no nulos. La implementación real nunca materializa los ceros: reindexa la entrada.
- **Como operación a escala más gruesa.** "Permite efectivamente que la red opere a una escala más gruesa que con una convolución normal. Esto es similar a *pooling* o a convoluciones con stride, **pero aquí la salida tiene el mismo tamaño que la entrada**." Esta última cláusula es todo el punto: se gana escala **sin perder resolución temporal**, que es exactamente lo que un modelo autorregresivo muestra-a-muestra necesita, porque tiene que emitir una predicción en cada timestep.

### 6.2. La aritmética del campo receptivo

Para una pila de $L$ capas con tamaños de kernel $k_1,\dots,k_L$ y dilataciones $d_1,\dots,d_L$ (sin stride), el campo receptivo en muestras es:

$$R \;=\; 1 + \sum_{l=1}^{L} (k_l - 1)\, d_l$$

Con kernel uniforme $k$ y dilataciones que se duplican, $d_l = 2^{l-1}$ para $l = 1,\dots,L$:

$$R \;=\; 1 + (k-1)\sum_{l=1}^{L} 2^{l-1} \;=\; 1 + (k-1)\left(2^{L} - 1\right)$$

**Caso $k = 2$** (el de WaveNet, según la Figura 3 y la aritmética del paper):

$$\boxed{R = 2^{L}}$$

El campo receptivo es **exactamente $2^L$**: exponencial en la profundidad. Diez capas con dilataciones $1,2,4,\dots,512$ dan $R = 1024$, que es precisamente lo que afirma el paper: "cada bloque $1,2,4,\dots,512$ tiene un campo receptivo de tamaño **1024**, y puede verse como una contraparte más eficiente y discriminativa (no lineal) de una convolución $1\times1024$".

Verificación con la otra fórmula: $1 + 1\cdot(1+2+4+\cdots+512) = 1 + 1023 = 1024$. ✓

### 6.3. El ejemplo del paper: bloques repetidos

El paper describe el esquema concreto (§2.1): "la dilatación se duplica en cada capa hasta un límite y luego se repite: por ejemplo

$$1, 2, 4, \dots, 512,\quad 1, 2, 4, \dots, 512,\quad 1, 2, 4, \dots, 512$$

Nótese el "**e.g.**": el paper lo presenta como ilustración del principio de diseño, no como la configuración declarada de sus experimentos (ver §13).

Un bloque de 10 capas contribuye $\sum d = 1023$ al campo receptivo. Con $B$ bloques y $k=2$:

$$R = 1 + B \cdot 1023$$

| Bloques | Capas dilatadas | $R$ (muestras) | $R$ @16 kHz | $R$ @44.1 kHz |
|---|---|---|---|---|
| 1 | 10 | 1.024 | 64.0 ms | 23.2 ms |
| 2 | 20 | 2.047 | 127.9 ms | 46.4 ms |
| 3 | 30 | 3.070 | 191.9 ms | 69.6 ms |
| 4 | 40 | 4.093 | 255.8 ms | 92.8 ms |
| 5 | 50 | 5.116 | 319.8 ms | 116.0 ms |

Estas cifras hay que contrastarlas con las dos que el paper sí declara: **campo receptivo de 240 ms para los WaveNets de TTS** (§3.2) y **"unos 300 milisegundos"** para el modelo multi-hablante (§3.1). A 16 kHz —el sampling declarado en el Apéndice B— 240 ms son **3.840 muestras** y 300 ms son **4.800**. Ninguna de las dos coincide exactamente con un número entero de bloques de 10 capas; 300 ms cae entre 4 y 5 bloques, y 240 ms entre 3 y 4. La conclusión honesta es que **el paper nunca publica el número de capas, el número de bloques ni el ancho de los canales**, así que la traducción de milisegundos a arquitectura es reconstrucción, no cita. Lo que sí es citable y verificable es la relación $R = 1 + B\cdot 1023$ y el valor 1024 por bloque, que el paper afirma explícitamente.

### 6.4. La justificación del slide: denso vs dilatado

El slide de la clase 39 afirma que la dilatación permite que "tras pocas capas de profundidad las neuronas puedan cubrir miles de timesteps manteniendo eficiencia computacional". Los números lo respaldan de forma contundente. Fijemos un objetivo de $R \ge 1024$ muestras (64 ms a 16 kHz — apenas el orden de un fonema corto) y comparemos las cuatro maneras de conseguirlo, contando parámetros con $C$ canales de entrada y salida por capa:

| Estrategia | $k$ | Dilataciones | Capas | $R$ | Parámetros | MACs por muestra de salida | Factor vs dilatada |
|---|---|---|---|---|---|---|---|
| Filtro único gigante | 1024 | — | 1 | 1024 | $1024\,C^2$ | $1024\,C^2$ | **51×** |
| Pila densa, kernel 2 | 2 | todas 1 | 1023 | 1024 | $2046\,C^2$ | $2046\,C^2$ | **102×** |
| Pila densa, kernel 3 | 3 | todas 1 | 512 | 1025 | $1536\,C^2$ | $1536\,C^2$ | **77×** |
| **Dilatada, kernel 2** | 2 | $1,2,\dots,512$ | **10** | **1024** | $\mathbf{20\,C^2}$ | $\mathbf{20\,C^2}$ | **1×** |
| Dilatada, kernel 3 | 3 | $1,3,9,\dots,729$ | 7 | 2187 | $21\,C^2$ | $21\,C^2$ | 0.98× (con $R$ 2× mayor) |

Y con el objetivo escalado a $R \approx 4096$ (256 ms, el orden del campo receptivo real del paper):

| Estrategia | Capas necesarias | Parámetros | Factor |
|---|---|---|---|
| Filtro único $k=4096$ | 1 | $4096\,C^2$ | 171× |
| Pila densa $k=2$ | 4095 | $8190\,C^2$ | 341× |
| Pila densa $k=3$ | 2048 | $6144\,C^2$ | 256× |
| **Dilatada $k=2$, 4 bloques de 10** | **40** | $\mathbf{80\,C^2}$ | **1×** |

La afirmación del slide queda verificada con margen. Con **30 capas dilatadas** WaveNet cubre 3.070 muestras (191.9 ms a 16 kHz); conseguir lo mismo con convolución densa de kernel 2 requeriría **3.069 capas**. La relación es exactamente la que corresponde: **crecimiento lineal versus crecimiento exponencial del campo receptivo con la profundidad**. Y "miles de timesteps" no es hipérbole: son literalmente miles a partir del segundo bloque.

### 6.5. El compromiso: la dilatación no es gratis pero casi

**Lo que la dilatación no cuesta.** Una convolución dilatada de kernel $k$ tiene **exactamente los mismos $k$ pesos** que su versión no dilatada, y realiza **exactamente el mismo número de multiplicaciones-acumulaciones por posición de salida**. Los "agujeros" no se computan: la implementación reindexa la entrada con paso $d$. Formalmente, para una capa con $C_{in}$ canales de entrada, $C_{out}$ de salida y longitud $T$:

$$\text{Parámetros} = k\,C_{in}C_{out}, \qquad \text{MACs} = T\,k\,C_{in}C_{out}$$

**ambos independientes de $d$**. Esta es la propiedad que hace que la afirmación "campo receptivo exponencial a costo lineal" sea literalmente cierta y no una aproximación.

**Lo que sí cuesta.** Tres cosas, en orden de importancia práctica:

1. **Localidad de memoria.** Con $d = 512$, la capa lee elementos separados por 512 posiciones en el eje temporal. El patrón de acceso destruye la localidad de caché. En la práctica las implementaciones usan `space_to_batch` / `batch_to_space` (reordenamiento a bloques contiguos, la estrategia de TensorFlow) o kernels especializados, pero el throughput real por FLOP es peor que en una convolución densa.
2. **Huecos en la cobertura (*gridding*).** Una sola capa con $d = 512$ mira 2 muestras separadas por 512: no ve nada de lo que hay en medio. Si se apilaran capas dilatadas **sin** dilataciones pequeñas que las complementen, la red vería un **retículo disperso** de la entrada y sería ciega a la estructura fina en las escalas intermedias. Este es el fenómeno que Yu & Koltun (2016) —citado por WaveNet— y Wang et al. (2018) documentaron en segmentación semántica como artefactos de tablero de ajedrez.
3. **Profundidad no lineal desacoplada del campo receptivo.** Un bloque de 10 capas cubre 1024 muestras pero solo aplica 10 no linealidades. Se puede tener un campo receptivo enorme con un modelo poco expresivo. Este es el motivo real, y menos citado, de la repetición de bloques.

**Por qué el esquema del paper no sufre gridding.** Vale la pena hacer la verificación formal, porque es un punto elegante que casi nunca se explicita. La condición para que una pila dilatada cubra **sin huecos** su campo receptivo es que la dilatación de cada capa no exceda el campo receptivo acumulado por las capas anteriores:

$$d_{l+1} \;\le\; R_l \;=\; 1 + \sum_{i=1}^{l}(k_i-1)\,d_i$$

Con $k = 2$ y $d_l = 2^{l-1}$, tenemos $R_l = 2^l$ y $d_{l+1} = 2^{l}$. La condición se satisface **con igualdad exacta en cada capa**. Es decir: el esquema de duplicación con kernel 2 es precisamente **el crecimiento más rápido posible sin dejar agujeros**. Un factor de duplicación de 3 con kernel 2 ($1,3,9,\dots$) violaría la condición desde la segunda capa y produciría gridding. La estructura resultante es la de un **árbol binario perfecto**: cada una de las $2^L$ entradas del campo receptivo contribuye a la salida por **exactamente un camino**, sin redundancia y sin omisión. Muy limpio, y explica por qué WaveNet nunca reportó artefactos de retículo pese a usar dilataciones de 512.

**Por qué repetir bloques.** El paper da una justificación en dos partes (§2.1): "Primero, aumentar exponencialmente el factor de dilatación resulta en crecimiento exponencial del campo receptivo con la profundidad. [...] Segundo, apilar estos bloques **aumenta aún más la capacidad del modelo y el tamaño del campo receptivo**". El reinicio a $d=1$ al comenzar cada bloque tiene tres efectos concretos:

- **Reconstruye la cobertura fina.** Cada bloque nuevo vuelve a mirar vecinos inmediatos, densificando la conectividad efectiva. Los caminos ya no son únicos: cada entrada alcanza la salida por múltiples rutas de distintas escalas, lo que convierte el árbol binario en un grafo denso multiescala.
- **Triplica la profundidad no lineal** (con 3 bloques) para un crecimiento del campo receptivo que es solo lineal en el número de bloques. Es la palanca correcta: pagar profundidad donde se necesita expresividad, no donde se necesita alcance.
- **Estabiliza el entrenamiento.** Combinado con las conexiones residuales (§7), 30 capas son perfectamente entrenables; 30 capas de dilatación monótonamente creciente hasta $2^{29}$ serían inútiles.

### 6.6. Veredicto sobre la afirmación del slide

El slide dice: *"tras pocas capas de profundidad las neuronas pueden cubrir miles de timesteps manteniendo eficiencia computacional"*. Es exacto, y ahora con los números:

- **"Pocas capas":** 10 capas para 1.024 muestras; 30 para 3.070. Frente a las 3.069 capas que necesitaría una pila densa de kernel 2 para lo mismo.
- **"Miles de timesteps":** 3.070 muestras con 30 capas. A 16 kHz, 192 ms — el orden de 2–3 fonemas, que es exactamente lo que el paper reporta para su modelo multi-hablante ("solo puede recordar los últimos 2–3 fonemas que produjo", §3.1).
- **"Manteniendo eficiencia computacional":** literalmente cierto por capa. Los parámetros y las MACs son independientes de $d$. El costo total es lineal en la profundidad, mientras el alcance es exponencial.

La única precisión que agregaría al slide es que la eficiencia es **de entrenamiento y de campo receptivo por parámetro**, no de generación. Ninguna cantidad de dilatación arregla que generar un segundo de audio requiera 16.000 pasadas secuenciales.

---

## 7. Bloques residuales, skip connections y la activación con compuerta

### 7.1. La unidad de activación con compuerta

WaveNet no usa ReLU. Usa la misma **gated activation unit** del Gated PixelCNN (van den Oord et al. 2016b), ecuación (2) del paper:

$$\mathbf{z} \;=\; \tanh\!\left(W_{f,k} * \mathbf{x}\right) \;\odot\; \sigma\!\left(W_{g,k} * \mathbf{x}\right)$$

donde $*$ es convolución, $\odot$ producto elemento a elemento, $\sigma$ la sigmoide, $k$ el índice de capa, y los subíndices $f$ y $g$ denotan *filter* y *gate*. Ambas convoluciones tienen la misma dilatación y operan sobre la misma entrada; en la práctica se implementan como **una sola convolución con el doble de canales de salida**, que luego se parte por la mitad.

El paper reporta la evidencia empírica sin adornos (§2.3): "En nuestros experimentos iniciales, observamos que esta no linealidad funcionaba **significativamente mejor** que la función de activación rectificada lineal para modelar señales de audio". Eso es todo lo que dice. No hay ablación cuantificada. Las razones hay que reconstruirlas, y son varias y convergentes:

1. **Interacción multiplicativa.** ReLU es una función *aditiva y por coordenada*: el valor de un canal no puede modular el de otro dentro de la misma activación. El producto $\tanh(\cdot)\odot\sigma(\cdot)$ introduce una **interacción de segundo orden** entre dos proyecciones lineales distintas de la misma entrada. Una capa puede aprender a decir "usa este detector de formante **solo si** esta otra proyección indica que hay voicing". Con ReLU esa condicionalidad requiere capas adicionales.
2. **Es exactamente el mecanismo de una LSTM.** $\sigma$ produce una compuerta en $[0,1]$ que atenúa o deja pasar un contenido acotado en $[-1,1]$. WaveNet importa el mecanismo de compuerta de las RNN sin importar la recurrencia. Dado que el paper argumenta contra las RNN por su costo de entrenamiento, es coherente que preserve lo que sí funcionaba de ellas.
3. **Salida acotada.** $\tanh \in [-1,1]$ y $\sigma \in [0,1]$ implican $z \in [-1,1]$. Con 30 o más capas residuales apiladas y **sin batch normalization** (el paper no menciona normalización alguna), acotar la activación es un mecanismo de estabilidad barato. ReLU no está acotada por arriba y las sumas residuales acumulan escala.
4. **Sin unidades muertas.** ReLU tiene gradiente exactamente cero en el semieje negativo; una unidad que se sesga hacia allá deja de aprender. Sobre 30 capas eso importa.
5. **Multimodalidad.** El paper eligió una salida categórica precisamente porque las condicionales de audio son multimodales. Una arquitectura interna con interacciones multiplicativas es más apta para representar funciones con conmutación abrupta —que es lo que hace el habla al pasar de silencio a explosión de oclusiva— que una pila puramente aditiva.

Vale registrar el contrapunto histórico: en PixelCNN el *gating* fue introducido justamente para recuperar el desempeño que PixelCNN perdía frente a PixelRNN, y ahí la ganancia estaba medida en NLL. En WaveNet la evidencia es cualitativa ("funcionaba significativamente mejor") y no reproducible desde el paper.

### 7.2. Flujo residual y skip

La Figura 4 muestra el bloque y la arquitectura completa. La estructura, leída desde la figura:

**Entrada global** → una **convolución causal** inicial (no dilatada) que proyecta de la representación de entrada a los canales residuales → **$k$ bloques residuales apilados** → suma de todas las skip → `ReLU` → `1×1` → `ReLU` → `1×1` → `Softmax` → salida.

Y dentro de cada bloque residual:

1. `Dilated Conv` sobre la entrada del bloque.
2. La unidad con compuerta: rama $\tanh$, rama $\sigma$, producto $\times$.
3. Una **convolución $1\times1$** sobre $\mathbf{z}$.
4. Dos destinos: (a) **suma residual** con la entrada del bloque, que se convierte en la entrada del bloque siguiente; (b) **skip connection** que se suma al bus de skip que va a la salida.

El paper es escueto (§2.4): "Tanto conexiones residuales (He et al. 2015) como conexiones skip parametrizadas se usan a lo largo de toda la red, **para acelerar la convergencia y permitir entrenar modelos mucho más profundos**". La palabra clave es *parametrizadas*: las skip no son atajos identidad, pasan por una proyección $1\times1$ aprendida. En las implementaciones estándar (y en la Figura 4, que muestra un solo `1×1` del que salen dos flechas) hay dos proyecciones separadas, una hacia residual y otra hacia skip, con anchos distintos: los canales residuales suelen ser pocos (para que la pila sea barata) y los canales skip muchos (para que el bus de salida tenga capacidad).

**La división del trabajo entre las dos rutas es el punto interesante, y el paper no lo explica:**

- La **ruta residual** es el camino de la información *hacia arriba*. Su función principal es la de siempre en ResNet: garantizar que el gradiente llegue a las capas bajas sin atenuarse a través de 30 no linealidades acotadas. Sin ella, el $\tanh\odot\sigma$ —cuyo jacobiano tiene norma típicamente $< 1$— haría que el gradiente se desvaneciera mucho antes del bloque 1.
- La **ruta skip** es el camino de la información *hacia la salida*. Cada bloque deposita directamente su contribución en el bus final. Esto significa que la predicción se computa a partir de **una suma de representaciones de todas las escalas temporales simultáneamente**: el bloque con $d=1$ aporta la microestructura de la forma de onda (fase, amplitud instantánea), el bloque con $d=512$ aporta contexto de decenas de milisegundos. Es una arquitectura **explícitamente multiescala**, y las skip son el mecanismo que hace que ninguna escala tenga que sobrevivir el viaje por todas las capas superiores para ser usada.

Sin skip connections, la información de escala fina tendría que propagarse a través de todos los bloques posteriores, que están dilatados y por lo tanto la comprimirían. Las skip son lo que evita que el modelo tenga que elegir entre resolución y contexto.

Un detalle práctico ausente del paper pero universal en las implementaciones: la suma residual suele escalarse por $\sqrt{0.5}$ para que la varianza no crezca linealmente con la profundidad. No aparece en el texto ni en la Figura 4.

---

## 8. Condicionamiento global y local

Sin condicionamiento, WaveNet es un modelo del habla *en general*: genera balbuceo con la fonotáctica y la prosodia correctas del idioma pero sin contenido (§3.1). Para que sirva de algo hay que condicionarlo. Ecuación (3):

$$p(\mathbf{x}\mid \mathbf{h}) = \prod_{t=1}^{T} p\!\left(x_t \mid x_1,\dots,x_{t-1},\mathbf{h}\right)$$

El paper distingue dos modos según la **estructura temporal del condicionante**.

### 8.1. Condicionamiento global

Un **único vector latente $\mathbf{h}$** que influye sobre la distribución de salida en **todos** los timesteps. El caso canónico es el embedding de hablante. La activación se modifica así:

$$\mathbf{z} \;=\; \tanh\!\left(W_{f,k} * \mathbf{x} + V_{f,k}^{T}\mathbf{h}\right)\;\odot\;\sigma\!\left(W_{g,k} * \mathbf{x} + V_{g,k}^{T}\mathbf{h}\right)$$

donde $V_{*,k}$ es una proyección lineal aprendida y el vector $V_{*,k}^{T}\mathbf{h}$ se **difunde (broadcast) sobre la dimensión temporal**.

Mecánicamente es un **sesgo aditivo dependiente del condicionante, aplicado dentro de cada capa antes de las no linealidades**. Sesgar el argumento de la sigmoide desplaza el punto de operación de la compuerta: con el hablante A ciertos canales quedan abiertos y con el hablante B cerrados. Es decir, la identidad del hablante **reconfigura qué detectores están activos en toda la pila**, no solo reescala la salida. Y como se inyecta en *cada* capa (índice $k$ en $V_{*,k}$), no es una condición que deba sobrevivir el viaje por la red: se reinyecta a cada nivel.

En el experimento de VCTK (§3.1) $\mathbf{h}$ es simplemente el **one-hot de 109 hablantes**, con lo que $V^T\mathbf{h}$ es una lectura de tabla de embeddings. El resultado observado es notable: "agregar hablantes resultó en **mejor desempeño en el conjunto de validación** comparado con entrenar sobre un solo hablante. Esto sugiere que la representación interna de WaveNet estaba **compartida entre múltiples hablantes**". Un modelo de 44 horas repartidas entre 109 voces aprende mejor que uno de 24 horas de una sola voz — porque casi toda la estructura de la forma de onda del habla (armónicos, formantes, transiciones) es independiente del hablante, y solo un residuo lo distingue.

Y un efecto lateral que anticipa toda la literatura posterior de clonación de voz: "el modelo también captó otras características del audio además de la voz misma. Por ejemplo, también **imitó la acústica y la calidad de grabación, así como la respiración y los movimientos de boca** de los hablantes". WaveNet modela la señal completa, incluyendo el canal y los artefactos del micrófono. Es un modelo generativo del audio, no de la voz.

### 8.2. Condicionamiento local

Una **segunda serie temporal $h_t$**, típicamente con **frecuencia de muestreo mucho menor** que el audio. Para TTS son las features lingüísticas, derivadas cada **5 ms** (Apéndice B) — es decir a 200 Hz, contra las 16.000 muestras por segundo del audio. Hay un factor de **80×** entre ambas resoluciones, y hay que salvarlo.

El paper propone dos mecanismos y declara cuál ganó:

**(a) Convolución transpuesta (upsampling aprendido) — el que usaron.** Se transforma $\mathbf{h}$ con "una red convolucional transpuesta (*upsampling* aprendido) que la mapea a una nueva serie temporal $\mathbf{y} = f(\mathbf{h})$ **con la misma resolución que la señal de audio**", y luego:

$$\mathbf{z} \;=\; \tanh\!\left(W_{f,k} * \mathbf{x} + V_{f,k} * \mathbf{y}\right)\;\odot\;\sigma\!\left(W_{g,k} * \mathbf{x} + V_{g,k} * \mathbf{y}\right)$$

"donde $V_{f,k}*\mathbf{y}$ es ahora una convolución $1\times1$". Es decir: el upsampling es una red aparte que corre **una sola vez**, y su salida a resolución de audio se proyecta con un $1\times1$ dentro de cada bloque.

**(b) Repetición.** "Como alternativa a la red convolucional transpuesta, también es posible usar $V_{f,k}*\mathbf{h}$ y **repetir estos valores a lo largo del tiempo**. Vimos que esto funcionaba **ligeramente peor** en nuestros experimentos."

La diferencia es el perfil de transición. La repetición produce una señal de condicionamiento en escalones: constante durante 80 muestras y luego un salto discontinuo. La convolución transpuesta aprende **cómo interpolar** entre valores consecutivos, y puede aprender transiciones suaves donde la fonética las requiere y abruptas donde hay una frontera de fono. Que la diferencia sea "ligera" es un dato útil: si se está implementando esto, empezar con repetición no es un error grave.

**El $F_0$ y el rol del condicionamiento en la prosodia.** Aquí está el hallazgo experimental más instructivo de la sección. Con condicionamiento solo lingüístico —**WaveNet (L)**— el paper reporta (§3.2) que el modelo "podía sintetizar muestras con **calidad segmental natural** pero a veces tenía **prosodia poco natural**, acentuando palabras equivocadas en una oración". El diagnóstico es explícitamente de campo receptivo:

> "Esto podría deberse a la dependencia de largo plazo de los contornos de $F_0$: el tamaño del campo receptivo de la WaveNet, **240 milisegundos, no era suficientemente largo** para capturar tal dependencia de largo plazo."

La solución fue condicionar también sobre el $\log F_0$ —**WaveNet (L+F)**— predicho por un modelo externo. Y la explicación es limpia: "el modelo externo de predicción de $F_0$ corre a una frecuencia mucho menor (**200 Hz**) así que **puede aprender dependencias de largo alcance que existen en los contornos de $F_0$**".

Esto es un principio de diseño transferible: **si el campo receptivo del modelo de forma de onda no alcanza para una dependencia, se saca esa dependencia del modelo y se le entrega precomputada por un modelo que opere en una escala temporal más gruesa**. 240 ms a 200 Hz son 48 pasos; a 16 kHz son 3.840. La misma memoria efectiva cuesta 80 veces menos en la serie lenta. El condicionamiento local es un mecanismo de **descomposición jerárquica de escalas temporales** disfrazado de detalle de implementación.

### 8.3. Por qué esto convierte a WaveNet en un vocoder, no en un TTS

Este es el matiz que más se pierde en las citas de segunda mano. **WaveNet no es un sistema TTS end-to-end.** El Apéndice B lo detalla:

- Las **features lingüísticas** (identidad de fono, acento silábico, número de sílabas en la palabra, posición de la sílaba en la frase, más features de posición de frame y duración de fono) provienen del formato de etiquetas dependientes de contexto de Zen (2006) — es decir, del **mismo front-end de análisis de texto** que alimenta a los sistemas HMM clásicos. Se derivan y asocian al habla cada 5 ms mediante **alineamiento forzado a nivel de fono** en el entrenamiento.
- La **duración de cada fono** la predice un modelo **LSTM-RNN externo**.
- El **$\log F_0$** lo predice una **CNN autorregresiva externa**.
- Ambos externos se entrenan minimizando MSE.

WaveNet reemplaza exactamente **una** pieza del pipeline clásico: el **vocoder**, es decir el generador de forma de onda a partir de parámetros. Todo lo que está aguas arriba —normalización de texto, G2P, contextos lingüísticos, duraciones, contorno de pitch— sigue siendo el sistema de producción de Google.

Y eso es lo que hace la comparación honesta, no lo que la debilita: "Como se usaron los **mismos datasets y las mismas features lingüísticas** para entrenar tanto los baselines como las WaveNets, estos sintetizadores podían ser comparados de forma justa" (§3.2). La única variable que cambia entre WaveNet y los baselines es el generador de onda. Los 0.35 puntos de MOS son atribuibles a esa pieza y solo a esa pieza.

También explica el camino que tomó la literatura posterior: **Tacotron 2** (Shen et al., 2018) reemplazó todo el front-end por un seq2seq con atención que produce un **mel-espectrograma**, y usó una WaveNet modificada condicionada localmente sobre ese mel en vez de sobre features lingüísticas. El interfaz de condicionamiento local de §2.5 es lo que permitió ese reemplazo sin tocar el vocoder. "Mel-espectrograma en, forma de onda out" sigue siendo la interfaz estándar en 2026.

Un último dato del Apéndice B que dice mucho sobre la calidad del resultado: **"es importante notar que no se aplicó ningún post-procesamiento a las señales de audio generadas por las WaveNets"**. Los sistemas paramétricos de la época dependían de post-filtros de modulación espectral (Takamichi et al. 2016) para compensar el sobresuavizado. WaveNet no necesitó ninguno.

---

## 9. Experimentos y resultados

Cuatro dominios. El paper enumera tres en §3 ("generación de habla multi-hablante, TTS, y modelado de audio musical") y añade el cuarto en §3.4.

### 9.1. Generación multi-hablante libre (§3.1)

**Datos:** corpus multi-hablante en inglés del **CSTR Voice Cloning Toolkit (VCTK)** (Yamagishi 2012), **44 horas, 109 hablantes**. Condicionamiento **solo por hablante**, one-hot. Sin texto.

**Resultado cualitativo:** genera "palabras inexistentes pero similares a lenguaje humano de manera fluida, con entonaciones que suenan realistas". La comparación que hace el paper es con modelos generativos de lenguaje o imágenes, "donde las muestras parecen realistas a primera vista pero son claramente antinaturales al inspeccionarlas de cerca".

**El diagnóstico cuantitativo, que es el valor real de esta sección:** "La falta de coherencia de largo alcance se debe en parte al tamaño limitado del campo receptivo del modelo (**unos 300 milisegundos**), lo que significa que solo puede recordar **los últimos 2–3 fonemas** que produjo." Este es el único lugar donde el paper traduce campo receptivo a una unidad lingüística, y es la calibración más útil del documento: **300 ms ≈ 2–3 fonemas**. Cualquier evaluación de si un campo receptivo alcanza debe pasar por esa escala.

Tres hallazgos adicionales:

- Un solo modelo captura los 109 hablantes "con igual fidelidad" y conmuta entre ellos por condicionamiento.
- Agregar hablantes **mejora** el desempeño en validación respecto de entrenar con uno solo → representación interna compartida.
- El modelo imita también acústica de sala, calidad de grabación, respiración y movimientos de boca.

### 9.2. TTS (§3.2 y Apéndice B)

**Datos:** las mismas bases de datos mono-hablante desde las que se construyen los sistemas TTS de producción de Google. **Inglés norteamericano: 24.6 horas. Mandarín: 34.8 horas.** Ambas de hablantes femeninas profesionales. Muestreo a **16 kHz** para el sistema concatenativo y para WaveNet; los LSTM-RNN se entrenaron a 22.05 kHz y se remuestreó a 16 kHz en runtime con el vocoder Vocaine.

**Campo receptivo de las WaveNets de TTS: 240 ms.**

**Baselines:** concatenativo *unit selection* dirigido por HMM (Gonzalvo et al. 2016) y paramétrico estadístico basado en LSTM-RNN (Zen et al. 2016). Ambos son sistemas reales de producción, no reimplementaciones de paper.

**Protocolo de evaluación (Apéndice B):** pruebas ciegas y *crowdsourced*, **100 frases fuera del entrenamiento**. Cada sujeto evaluó hasta 8 estímulos en inglés y hasta 63 en mandarín. Estímulos elegidos y presentados al azar. **8 sujetos por par** en la prueba de preferencia y **8 sujetos por estímulo** en el MOS. Sujetos pagados y hablantes nativos. Y un detalle de higiene experimental que hoy sería obligatorio y en 2016 no lo era: **se descartaron las evaluaciones en que no se usaron audífonos, que fueron cerca del 40%**.

**MOS (Tabla 1):** ver §1. Lo esencial: WaveNet (L+F) alcanza **4.21 ± 0.081** en inglés y **4.08 ± 0.085** en mandarín, ambos por encima de 4.0, y el paper afirma que "fueron los valores de MOS más altos jamás reportados con estos datasets de entrenamiento y estas frases de prueba".

**Preferencias pareadas (Tabla 2, Apéndice B; la Figura 5 muestra un subconjunto).** Tabla completa, en porcentaje, con el valor $p$ del test:

| Idioma | LSTM | Concat | WaveNet (L) | WaveNet (L+F) | Sin preferencia | $p$ |
|---|---|---|---|---|---|---|
| Inglés NA | 23.3 | **63.6** | | | 13.1 | $10^{-9}$ |
| Inglés NA | 18.7 | | **69.3** | | 12.0 | $10^{-9}$ |
| Inglés NA | 7.6 | | | **82.0** | 10.4 | $10^{-9}$ |
| Inglés NA | | 32.4 | **41.2** | | 26.4 | 0.003 |
| Inglés NA | | 20.1 | | **49.3** | 30.6 | $10^{-9}$ |
| Inglés NA | | | 17.8 | **37.9** | 44.3 | $10^{-9}$ |
| Mandarín | **50.6** | 15.6 | | | 33.8 | $10^{-9}$ |
| Mandarín | 25.0 | | 23.3 | | 51.8 | **0.476** |
| Mandarín | 12.5 | | | **29.3** | 58.2 | $10^{-9}$ |
| Mandarín | | 17.6 | **43.1** | | 39.3 | $10^{-9}$ |
| Mandarín | | 7.6 | | **55.9** | 36.5 | $10^{-9}$ |
| Mandarín | | | 10.0 | **25.5** | 64.5 | $10^{-9}$ |

Lecturas que la Figura 5 no deja ver:

- **El resultado más contundente del paper está en la tercera fila: WaveNet (L+F) 82.0% vs LSTM paramétrico 7.6% en inglés.** Una relación de más de 10 a 1.
- **La comparación con el concatenativo es mucho más ajustada:** 49.3% vs 20.1% con 30.6% de indiferencia. WaveNet gana, pero un tercio de los oyentes no distingue. El concatenativo era un rival mucho más duro que el paramétrico, que es exactamente lo que la literatura de TTS sostenía.
- **El ordenamiento de los baselines se invierte entre idiomas.** En inglés el concatenativo aplasta al LSTM (63.6 vs 23.3); en mandarín el LSTM aplasta al concatenativo (50.6 vs 15.6). Consistente con la Tabla 1 (Concat: 3.86 en inglés, 3.47 en mandarín). El "mejor baseline" **no es el mismo sistema en los dos idiomas**.
- **La única comparación no significativa del paper:** en mandarín, LSTM vs WaveNet (L) dio 25.0 vs 23.3 con **$p = 0.476$**. WaveNet condicionada solo por features lingüísticas **no fue mejor que el baseline paramétrico en mandarín**. El $F_0$ no era un extra opcional: era necesario para ganar en ese idioma. Coherente con que el mandarín es una lengua tonal, donde el contorno de $F_0$ es **contrastivo a nivel léxico** y no meramente prosódico.
- **La proporción de "sin preferencia" crece sistemáticamente en mandarín** (33.8% a 64.5%) frente a inglés (10.4% a 44.3%). Puede ser el protocolo —hasta 63 estímulos por sujeto en mandarín contra 8 en inglés, con la fatiga que eso implica— o menor separación real entre sistemas.

### 9.3. Música (§3.3)

**Datos:** **MagnaTagATune** (Law & Von Ahn 2009), ~200 horas, clips de 29 segundos etiquetados con tags de un conjunto de 188 que describen género, instrumentación, tempo, volumen y ánimo; y un **dataset propio de piano de YouTube**, ~60 horas de piano solo, "considerablemente más fácil de modelar" por estar restringido a un instrumento.

**Sin evaluación cuantitativa.** El paper lo admite: "aunque es difícil evaluar cuantitativamente estos modelos, es posible una evaluación subjetiva escuchando las muestras que producen". No hay MOS, no hay log-verosimilitud, no hay estudio con sujetos. Esta es la sección más débil del paper.

Los hallazgos declarados:

- "Encontramos que **agrandar el campo receptivo era crucial** para obtener muestras que sonaran musicales."
- "Incluso con un campo receptivo de **varios segundos**, los modelos **no imponían consistencia de largo alcance**, lo que resultaba en variaciones de segundo a segundo en género, instrumentación, volumen y calidad de sonido."
- "Sin embargo, las muestras eran a menudo armónicas y estéticamente agradables, incluso producidas por modelos incondicionales."
- El condicionamiento por tags funciona insertando sesgos que dependen de un vector binario de tags por clip — es condicionamiento **global**, la misma maquinaria del embedding de hablante. Requirió limpiar los tags de MagnaTagATune ("relativamente ruidosos y con muchas omisiones") fusionando tags similares y eliminando los de pocos clips.

Que los modelos de música usaran "varios segundos" de campo receptivo mientras los de TTS usaban 240 ms es el único indicio del paper de que la arquitectura se escaló según dominio.

### 9.4. Reconocimiento de voz en TIMIT (§3.4) — el resultado que se olvida

El experimento que casi nunca se cita, y el más relevante para la clase 39, porque es **el uso discriminativo de la misma arquitectura**.

**Contexto declarado por el paper:** el reconocimiento de voz se había apoyado históricamente en energías de banco de filtros log-mel o MFCC, pero "se ha estado moviendo hacia audio crudo recientemente" (Palaz et al. 2013; Tüske et al. 2014; Hoshen et al. 2015; **Sainath et al. 2015**). Y la afirmación estructural, que es exactamente la tesis del slide:

> "Las redes neuronales recurrentes como las LSTM-RNN han sido un componente clave en estos nuevos pipelines de clasificación de habla, porque permiten construir modelos con contextos de largo alcance. Con las WaveNets hemos mostrado que **las capas de convoluciones dilatadas permiten que el campo receptivo crezca de manera mucho más barata que usando unidades LSTM**."

**Dataset:** **TIMIT** (Garofolo et al. 1993).

**Modificaciones a la arquitectura:**

1. Se agregó una capa de **mean-pooling después de las convoluciones dilatadas**, que agrega las activaciones a **frames más gruesos de 10 milisegundos** — es decir, un **submuestreo de 160×** (a 16 kHz, 160 muestras = 10 ms).
2. Tras el pooling, "**unas pocas convoluciones no causales**". La causalidad se abandona explícitamente en cuanto la tarea deja de ser generativa.
3. **Dos términos de pérdida**: uno para predecir la muestra siguiente (el objetivo generativo original) y otro para clasificar el frame. El paper reporta que "el modelo **generalizó mejor que con una sola pérdida**".

**Resultado: 18.8 PER (phone error rate) en el conjunto de test.** El paper califica el resultado con precisión quirúrgica: "que es, según nuestro conocimiento, **el mejor puntaje obtenido por un modelo entrenado directamente sobre audio crudo en TIMIT**". Es un récord dentro de una categoría restringida, no el estado del arte absoluto de TIMIT — que en 2016 estaba en el rango de 16–18 PER con modelos sobre features log-mel.

Este experimento anticipa exactamente la arquitectura del "Ejemplo 2" de la clase: convoluciones dilatadas sobre audio crudo → pooling a resolución de frame → capas no causales → clasificación. Ver §12.

---

## 10. Limitaciones

### 10.1. La inferencia es secuencial y lentísima

Esta domina todas las demás, y el paper no la discute. Ni en la conclusión ni en ninguna sección aparece la palabra "latencia", "tiempo real", ni una sola medición de velocidad de generación. La única mención está en §2.1, presentada como un hecho neutro: "Cuando se genera con el modelo, las predicciones son secuenciales: después de que cada muestra es predicha, se realimenta a la red para predecir la siguiente".

La aritmética del problema:

- Generar **1 segundo de audio a 16 kHz requiere 16.000 pasadas completas hacia adelante** por la red.
- Cada pasada recorre las decenas de capas residuales, cada una con su convolución dilatada, su compuerta y sus dos proyecciones $1\times1$, más la softmax de 256 clases al final.
- Las 16.000 pasadas son **estrictamente dependientes**: la muestra $t$ requiere haber muestreado la $t-1$. **No hay paralelismo temporal disponible en absoluto.**
- El batch de trabajo por pasada es de **una sola posición temporal**, así que las convoluciones se reducen a productos matriz-vector. Una GPU está diseñada para lo contrario. La utilización es catastrófica: el hardware pasa el tiempo esperando memoria, no multiplicando.

El orden de magnitud del daño, con cifras que **no están en este paper** sino en el trabajo que lo resolvió: el paper de **Parallel WaveNet** (van den Oord et al., 2018) reporta que el WaveNet autorregresivo original generaba del orden de **172 muestras por segundo**, contra las **16.000 muestras por segundo** que se necesitan para tiempo real. Es decir, **aproximadamente 90 veces más lento que tiempo real**: producir un segundo de habla tomaba del orden de un minuto y medio. Parallel WaveNet reportó del orden de **500.000 muestras por segundo** con la misma calidad perceptual. Consigno que estas dos cifras son externas a este paper y provienen del trabajo sucesor; no pude verificarlas contra el PDF de 1609.03499, que no reporta ninguna medición de velocidad.

La ironía estructural es completa: **la arquitectura fue elegida por ser paralela en entrenamiento, y es exactamente esa elección la que la hace ineficiente en generación**. Una convolución causal sin recurrencia obliga, en inferencia ingenua, a recomputar todo el campo receptivo en cada paso — trabajo $O(R)$ por muestra donde una RNN gasta $O(1)$. La mitigación estándar es el **fast WaveNet inference** (Paine et al., 2016), que cachea las activaciones intermedias en colas circulares de longitud $d$ por capa, reduciendo el trabajo por muestra a $O(L)$. Eso da órdenes de magnitud, pero no cambia la naturaleza secuencial.

### 10.2. Falta de estructura de largo plazo

Documentada por el propio paper en dos dominios, y en ambos atribuida al campo receptivo:

- **Habla libre (§3.1):** campo receptivo de ~300 ms → "solo puede recordar los últimos 2–3 fonemas" → genera palabras inexistentes, sin sintaxis ni semántica.
- **TTS (§3.2):** 240 ms → prosodia poco natural, acentúa palabras equivocadas. Se parchó con el $F_0$ externo.
- **Música (§3.3):** "incluso con un campo receptivo de varios segundos, los modelos no imponían consistencia de largo alcance, lo que resultaba en variaciones de segundo a segundo en género, instrumentación, volumen y calidad de sonido".

El caso de la música es el diagnóstico más honesto y el más incómodo: aumentar el campo receptivo de 300 ms a varios segundos —un orden de magnitud— **no resolvió el problema, solo lo desplazó**. La estructura musical vive en escalas de decenas de segundos (frases, secciones, forma), y ninguna dilatación razonable llega ahí. El paper propone los **context stacks** (§2.6) como remedio complementario —una pila separada, más pequeña, que procesa un tramo largo de señal y condiciona localmente una WaveNet grande que procesa un tramo corto, con pooling para correr a menor frecuencia— pero **no reporta ningún experimento que los use**. Es una sección de diseño sin evaluación.

Lo que la línea posterior demostró es que el problema no era de campo receptivo sino de **representación**: la solución fue la jerarquía discreta (VQ-VAE → Jukebox → AudioLM), donde un modelo de lenguaje opera sobre tokens que ya resumen decenas de milisegundos, y el modelo de forma de onda solo tiene que rellenar el detalle local.

### 10.3. Costo y opacidad del entrenamiento

Este paper es **notablemente poco reproducible**, cosa que su condición de preprint nunca revisado ayuda a explicar. No reporta:

- Número de capas, número de bloques, tamaño del kernel, ni el ancho de los canales residuales, de compuerta o de skip. Las dilataciones $1,\dots,512 \times 3$ se dan como "e.g.".
- Optimizador, learning rate, schedule, tamaño de batch, longitud de las secuencias de entrenamiento, número de pasos, tiempo de entrenamiento, hardware.
- Cualquier valor de log-verosimilitud, pese a argumentar que su tractabilidad es una ventaja central.
- Ablaciones cuantificadas. La ventaja del gated activation sobre ReLU ("significativamente mejor") y la de la convolución transpuesta sobre la repetición ("ligeramente peor") son afirmaciones sin números.
- Cualquier medición de tiempo o memoria.

El único hiperparámetro arquitectónico publicado es el campo receptivo en milisegundos (240 y ~300). Esto es un contraste violento con el rigor del protocolo de evaluación subjetiva del Apéndice B — hay dos culturas experimentales distintas conviviendo en el mismo documento.

### 10.4. La línea que resolvió el problema de velocidad

Cuatro salidas, en orden cronológico. Cada una ataca la misma restricción desde un ángulo distinto.

**Parallel WaveNet (van den Oord et al., ICML 2018) — destilación de densidad de probabilidad.** La idea es entrenar una red *estudiante* que es un **Inverse Autoregressive Flow (IAF)**: parte de ruido blanco $z \sim \text{Logistic}(0,1)$ de longitud $T$ y aplica transformaciones que producen todas las muestras **en paralelo**, con $x_t = z_t\cdot s(z_{<t}) + \mu(z_{<t})$. Un IAF es rapidísimo para **muestrear** (una pasada paralela) pero lentísimo para **evaluar** la verosimilitud de datos externos — exactamente el perfil opuesto al de WaveNet, que es lento para muestrear y rápido para evaluar. La destilación explota esa complementariedad: se muestrea del estudiante en paralelo, y el WaveNet *maestro* —ya entrenado— evalúa esas muestras y provee la señal de entrenamiento minimizando $D_{KL}(P_{\text{estudiante}}\,\|\,P_{\text{maestro}})$. El maestro nunca genera nada; solo puntúa. Se agregan pérdidas auxiliares (potencia, perceptual, contraste) para evitar el colapso de modos. **Esta es la variante que se desplegó en producción.**

**WaveRNN (Kalchbrenner et al., ICML 2018) — atacar la constante, no el orden.** Acepta que la generación es secuencial y reduce el costo de cada paso: una **única capa GRU** con salida factorizada en dos softmax de 8 bits (byte alto y byte bajo, "dual softmax"), lo que da 16 bits de resolución con dos softmax de 256 en vez de una de 65.536. Sobre eso aplica **poda por esparsidad** (hasta 96% de pesos en cero, con un modelo denso equivalente más grande) y **subscaling** (partir la secuencia en $B$ subsecuencias intercaladas que se generan en paralelo con dependencias aproximadas). Resultado: generación **más rápida que tiempo real en una CPU móvil**. La contribución conceptual es que el cuello de botella de WaveNet era la profundidad por muestra, no la autorregresión en sí.

**WaveGlow (Prenger, Valle & Catanzaro, ICASSP 2019) — flujo normalizante sin destilación.** Combina Glow con WaveNet: un **flujo invertible** entrenado directamente con máxima verosimilitud, **una sola red y una sola pérdida**, sin maestro, sin estudiante y sin las pérdidas auxiliares que hacían frágil el entrenamiento de Parallel WaveNet. La generación es una pasada paralela por el flujo invertido. Es la simplificación de ingeniería de la idea de Parallel WaveNet, a costa de un modelo grande (~90M de parámetros).

**HiFi-GAN (Kong, Kim & Bae, NeurIPS 2020) — abandonar la verosimilitud.** Un generador puramente convolucional (upsampling transpuesto + bloques residuales de campo receptivo múltiple) entrenado de forma adversarial, con la innovación clave en los discriminadores: un **multi-period discriminator** que evalúa la señal reordenada en varias periodicidades primas (2, 3, 5, 7, 11) para capturar la estructura periódica del habla, más un multi-scale discriminator. Sin autorregresión, sin flujo, sin verosimilitud. Genera **cientos de veces más rápido que tiempo real en GPU** y más rápido que tiempo real en CPU, con MOS comparable al de los autorregresivos. En 2026 sigue siendo el vocoder de referencia por relación calidad/costo, y la mayoría de los TTS de producción usa HiFi-GAN o un descendiente suyo.

| | Paradigma | Paralelo en generación | Necesita maestro | Costo relativo |
|---|---|---|---|---|
| WaveNet (2016) | Autorregresivo, verosimilitud exacta | No | — | 1× (referencia, ~90× peor que tiempo real) |
| Parallel WaveNet (2018) | IAF + destilación de densidad | Sí | Sí | ~3.000× más rápido |
| WaveRNN (2018) | Autorregresivo eficiente + esparsidad | No | No | Tiempo real en CPU móvil |
| WaveGlow (2019) | Flujo normalizante | Sí | No | Tiempo real en GPU |
| HiFi-GAN (2020) | GAN | Sí | No | Cientos de veces tiempo real |

---

## 11. Impacto y legado

### 11.1. El despliegue real

WaveNet pasó de preprint a producción en **poco más de un año**. En **octubre de 2017** DeepMind anunció que WaveNet estaba sirviendo tráfico real en el **Google Assistant**, en inglés norteamericano y japonés, con voces generadas a **24 kHz y 16 bits**. Es uno de los ciclos investigación-a-producción más rápidos de la década, y prácticamente el momento en que el público general notó que las voces sintéticas habían dejado de sonar sintéticas.

Con dos precisiones que el hype suele borrar:

- **Lo que se desplegó fue Parallel WaveNet, no este modelo.** El modelo de este paper no era desplegable: a ~90× más lento que tiempo real, servir una sola respuesta del Assistant habría tomado minutos.
- **Se subió la calidad de la señal en el camino:** de 16 kHz / 8 bits $\mu$-law a **24 kHz / 16 bits**, lo cual exigió abandonar la softmax categórica de 256 clases en favor de una mezcla de logísticas discretizadas. La decisión de diseño de §2.2 —la más elegante conceptualmente del paper— fue de las primeras en descartarse al llevar el modelo a producción.

### 11.2. La herencia de la convolución dilatada

Aquí está el legado más duradero y el que más importa para la clase. La convolución dilatada **no la inventó WaveNet** —el propio paper la atribuye a Holschneider et al. (1989) y Dutilleux (1989) en procesamiento de señales, y a Chen et al. (2015) y Yu & Koltun (2016) en segmentación de imágenes—, pero WaveNet fue el trabajo que la convirtió en **el mecanismo estándar para conseguir campos receptivos grandes en secuencias**.

**Segmentación semántica: *atrous convolution* en DeepLab.** El problema en segmentación es dual al de audio: las CNN de clasificación reducen la resolución agresivamente con pooling y strides, pero la segmentación necesita una salida a resolución de píxel. La *atrous convolution* de DeepLab (Chen et al., desde ICLR 2015 hasta DeepLabv3+) permite quitar los strides y compensar el campo receptivo con dilatación, manteniendo la resolución. DeepLabv2 introdujo el **ASPP (Atrous Spatial Pyramid Pooling)**: varias ramas dilatadas en paralelo con tasas distintas, agregando contexto multiescala. Cronológicamente DeepLab precede a WaveNet; el flujo de influencia fue de imágenes a audio, y WaveNet lo reconoce. Lo que WaveNet aportó de vuelta fue el **esquema de bloques repetidos con duplicación**, que es lo que la comunidad de visión adoptó después para combatir el gridding (**Dilated Residual Networks**, Yu, Koltun & Funkhouser 2017; **Hybrid Dilated Convolution**, Wang et al. 2018).

**Traducción automática: ByteNet.** Nal Kalchbrenner, coautor de WaveNet, publicó **ByteNet** (*Neural Machine Translation in Linear Time*, 2016) semanas después: un decoder de convoluciones causales dilatadas sobre un encoder dilatado, con el mismo argumento de paralelismo en entrenamiento. Fue el paso intermedio entre las RNN seq2seq y el Transformer: mismo diagnóstico (la recurrencia impide el paralelismo), distinta solución (dilatación en vez de atención). Que el Transformer ganara no invalida el diagnóstico — lo confirma.

**Series temporales: TCN.** Bai, Kolter y Koltun (*An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, 2018) tomaron la receta de WaveNet —convoluciones causales dilatadas + bloques residuales, sin las compuertas ni la salida categórica— la bautizaron **Temporal Convolutional Network** y mostraron que igualaba o superaba a LSTM/GRU en un abanico amplio de benchmarks de secuencias. Ese paper es la razón por la que "TCN" es hoy una línea base estándar en pronóstico de series temporales, detección de anomalías y modelado de señales fisiológicas, con el campo receptivo tratado como hiperparámetro de primera clase. **Si en algún momento te toca modelar series temporales de eventos clínicos, la TCN es el descendiente directo de este paper y su campo receptivo se calcula con la misma fórmula de §6.2.**

**Y dentro del audio, la arquitectura sobrevivió a los cambios de paradigma.** El backbone de convoluciones dilatadas con activación con compuerta sigue vivo en modelos que ya no son autorregresivos: **DiffWave** (Kong et al., ICLR 2021) usa una pila de residuales dilatados WaveNet-style, pero **bidireccional** (sin causalidad, porque el modelo de difusión denoisea la señal completa en paralelo) como red de predicción de ruido. La estructura sobrevivió; la causalidad no.

### 11.3. La generación siguiente: difusión y codecs neuronales

Dos líneas reemplazaron el paradigma de WaveNet, y ninguna es "una WaveNet mejor".

**Difusión sobre la forma de onda o sobre el espectrograma.** **WaveGrad** (Chen et al., 2020) y **DiffWave** (2021) modelan $p(x)$ como un proceso de denoising iterativo: se parte de ruido y se refina en $N$ pasos, cada uno **paralelo en el tiempo**. El intercambio es explícito y ajustable: en vez de $T = 16.000$ pasos secuenciales se usan $N \approx 6$–50 pasos secuenciales sobre tensores completos. Como la generación ya no es autorregresiva, la causalidad deja de ser necesaria y el campo receptivo puede ser bidireccional — pero el resto de la maquinaria (dilatación exponencial, gated units, skip connections) se conserva casi intacta. Es la mejor evidencia de que la contribución arquitectónica de WaveNet es más robusta que su contribución probabilística.

**Codecs neuronales y modelado sobre tokens.** **SoundStream** (Zeghidour et al., 2021) y **EnCodec** (Défossez et al., 2022) son autoencoders convolucionales con **cuantización vectorial residual (RVQ)** que comprimen audio a tasas de 1.5–24 kbps con calidad superior a Opus, entrenados con pérdidas adversariales y espectrales. El cambio conceptual es profundo: en vez de modelar 16.000 muestras por segundo, se modela una **secuencia de tokens discretos a ~50–75 Hz por codebook**. Eso reduce la longitud de secuencia en dos órdenes de magnitud y permite aplicarle directamente un Transformer autorregresivo. **AudioLM** (Borsos et al., 2022), **VALL-E** (Wang et al., 2023) y **MusicGen** (Copet et al., 2023) son exactamente eso.

Visto desde 2026, el arco es nítido y algo irónico. WaveNet mostró que **se podía** modelar audio crudo autorregresivamente y que el resultado era mejor que cualquier alternativa. La comunidad tardó unos seis años en concluir que **se podía pero no convenía**: la forma correcta es aprender una representación discreta comprimida —con una red que hereda las convoluciones dilatadas de WaveNet— y hacer la autorregresión ahí, donde la secuencia es 200 veces más corta. WaveNet estableció el techo de calidad y, al mismo tiempo, demostró por reducción al absurdo por qué había que cambiar de dominio.

Y la sección §2.2 del paper —"una categórica es más flexible porque no hace supuestos sobre la forma de la distribución"— es, mirada de cerca, el mismo argumento que sostiene hoy a los tokens de RVQ. Discretizar el audio y modelarlo con una distribución categórica sigue siendo la respuesta. Cambió el objeto que se discretiza: de la amplitud instantánea a un vector latente que resume 20 ms.

---

## 12. Conexión con la clase 39

### 12.1. El planteamiento del slide, verificado

El slide plantea la cadena: audio crudo exige 15–20 kHz (44.1 kHz para música) → muchísimas muestras por segundo → una arquitectura convolucional necesitaría filtros enormes o una red muy profunda para cubrir contexto suficiente → **la salida son los filtros de convolución dilatados**, que tras pocas capas cubren miles de timesteps manteniendo eficiencia computacional.

Cada eslabón está respaldado por el paper y por la aritmética de §6:

| Afirmación del slide | Respaldo |
|---|---|
| El audio crudo exige 15–20 kHz | §1: "al menos 16.000 muestras por segundo". Apéndice B: los sistemas se construyeron a 16 kHz |
| Convolución densa → filtros enormes o mucha profundidad | §2.1: campo receptivo = #capas + longitud del filtro − 1. Cuatro capas $k=2$ → 5 muestras = 0.3 ms |
| La dilatación cubre miles de timesteps con pocas capas | §6.2–6.4: $R = 2^L$ con $k=2$; 30 capas → 3.070 muestras vs 3.069 capas densas |
| Manteniendo eficiencia computacional | §6.5: parámetros y MACs por capa **independientes de $d$** |

Un matiz que vale la pena agregar al slide: la eficiencia es **de entrenamiento y de campo receptivo por parámetro**. La dilatación no hace nada por la velocidad de generación, que fue el problema real de WaveNet en producción.

### 12.2. El "Ejemplo 2": campo receptivo de la pila propuesta

El Ejemplo 2 de la clase propone, sobre audio crudo a 15–20 kHz:

| Capa | Tipo | Filtros | Kernel |
|---|---|---|---|
| 1 | Conv 1D dilatada | 128 | 20×1 |
| 2 | Conv 1D dilatada | 128 | 10×1 |
| 3 | Conv 1D dilatada | 256 | 10×1 |
| 4 | Conv 1D dilatada | 256 | 5×1 |
| 5–6 | LSTM | 256 | — |
| 7–8 | Fully connected | 1024 | — |

Aplicando $R = 1 + \sum_l (k_l - 1)d_l$ con $k = [20, 10, 10, 5]$, y a 16 kHz:

| Escenario | Dilataciones $(d_1,d_2,d_3,d_4)$ | $R$ (muestras) | $R$ @16 kHz | $R$ @20 kHz | Veredicto |
|---|---|---|---|---|---|
| **A** — sin dilatación | 1, 1, 1, 1 | **42** | 2.6 ms | 2.1 ms | Inútil: por debajo de un periodo glotal |
| **B** — duplicación | 1, 2, 4, 8 | **106** | 6.6 ms | 5.3 ms | Insuficiente: menos de un periodo de voz masculina |
| **C** — cuadruplicación | 1, 4, 16, 64 | **456** | 28.5 ms | 22.8 ms | Marginal: cubre ~3 periodos glotales, media vocal |
| **D** — potencias de 10 | 1, 10, 100, 1000 | **5.010** | 313 ms | 250 ms | Suficiente: escala de 2–3 fonemas |
| **E** — $d_l = \prod_{i<l} k_i$ | 1, 20, 200, 2000 | **10.000** | 625 ms | 500 ms | Amplio: escala de palabra |
| **F** — stride 2, sin dilatación | 1, 1, 1, 1 con $s=2$ | **106** | 6.6 ms | 5.3 ms | Igual que B pero con pérdida de resolución |

Cálculos explícitos:

- **A:** $1 + 19 + 9 + 9 + 4 = 42$.
- **B:** $1 + 19(1) + 9(2) + 9(4) + 4(8) = 1 + 19 + 18 + 36 + 32 = 106$.
- **C:** $1 + 19(1) + 9(4) + 9(16) + 4(64) = 1 + 19 + 36 + 144 + 256 = 456$.
- **D:** $1 + 19(1) + 9(10) + 9(100) + 4(1000) = 1 + 19 + 90 + 900 + 4000 = 5010$.
- **E:** $1 + 19(1) + 9(20) + 9(200) + 4(2000) = 1 + 19 + 180 + 1800 + 8000 = 10000$.
- **F:** con stride, el "salto" acumula: $j_1=1$, $R=20$; $j_2=2$, $R = 20 + 9(2) = 38$; $j_3=4$, $R = 38 + 9(4) = 74$; $j_4=8$, $R = 74 + 4(8) = 106$.

**La coincidencia entre B y F no es casual y es didácticamente valiosa:** apilar strides de 2 y apilar dilataciones que se duplican producen **exactamente el mismo campo receptivo**. La diferencia es que el stride **descarta** resolución temporal (la salida tiene $T/16$ posiciones) mientras la dilatación la conserva. Esa es literalmente la frase del paper: "similar a pooling o convoluciones con stride, **pero aquí la salida tiene el mismo tamaño que la entrada**".

### 12.3. ¿Alcanza para una ventana acústicamente útil?

Depende de qué escala se necesite. Las escalas relevantes del habla, con su equivalente en muestras a 16 kHz:

| Fenómeno | Escala temporal | Muestras @16 kHz |
|---|---|---|
| Periodo glotal, voz masculina ($F_0 \approx 100$ Hz) | 10 ms | **160** |
| Periodo glotal, voz femenina ($F_0 \approx 200$ Hz) | 5 ms | **80** |
| Ventana de análisis STFT típica | 20–30 ms | 320–480 |
| Oclusiva (fono más corto; Apéndice A) | < 20 ms | < 320 |
| Fonema típico | 50–100 ms | 800–1.600 |
| Sílaba | 150–250 ms | 2.400–4.000 |
| 2–3 fonemas (calibración de §3.1) | ~300 ms | **4.800** |
| Palabra | 300–600 ms | 4.800–9.600 |

Cruzando ambas tablas:

- **Escenarios A y B (42 y 106 muestras, 2.6 y 6.6 ms) son inservibles.** Ni siquiera cubren **un** periodo glotal de una voz masculina, que necesita 160 muestras. Una neurona con ese campo receptivo no puede distinguir sonoro de sordo, porque la periodicidad que define esa distinción vive en una escala mayor que su ventana. Y no puede estimar $F_0$ en absoluto. En términos de Fourier: una ventana de 42 muestras a 16 kHz tiene resolución frecuencial de $16000/42 \approx 380$ Hz — no resuelve armónicos individuales ni el primer formante con precisión.
- **Escenario C (456 muestras, 28.5 ms)** es el mínimo defendible: equivale a una ventana de análisis STFT convencional y cubre ~3 periodos glotales masculinos, lo suficiente para estimar periodicidad y estructura formántica. Sigue sin cubrir un fonema completo.
- **Escenarios D y E (5.010 y 10.000 muestras)** cubren la escala de 2–3 fonemas que el propio paper identifica como el mínimo para coherencia fonética, y llegan a la escala de palabra.

**El veredicto para el Ejemplo 2 depende críticamente del factor de dilatación que se elija, y el slide no lo especifica.** Con dilataciones "razonables" en el sentido de WaveNet (duplicación), la pila es **acústicamente ciega**: 6.6 ms. Con kernels grandes como los del Ejemplo 2 (20 y 10, no 2), el crecimiento por capa ya es fuerte y conviene un factor de dilatación mucho mayor. La regla de cobertura sin huecos de §6.5 —$d_{l+1} \le R_l$— da la respuesta de diseño correcta:

- $R_1 = 20$ → $d_2 \le 20$
- con $d_2 = 20$: $R_2 = 20 + 9(20) = 200$ → $d_3 \le 200$
- con $d_3 = 200$: $R_3 = 200 + 9(200) = 2000$ → $d_4 \le 2000$
- con $d_4 = 2000$: $R_4 = 2000 + 4(2000) = 10000$

Es decir, **el escenario E es exactamente el máximo crecimiento sin gridding** para esa configuración de kernels, y produce 625 ms a 16 kHz. Esa es la elección que yo defendería para el Ejemplo 2: cuatro capas, cero huecos, 625 ms de contexto por neurona de salida.

### 12.4. Las dos diferencias esenciales con WaveNet

**Diferencia 1: las LSTM cambian el rol del campo receptivo convolucional.**

En WaveNet la pila dilatada es **todo el modelo**: no hay nada después que pueda recuperar contexto perdido, y por eso el campo receptivo *es* la memoria del modelo. En el Ejemplo 2 hay **2 LSTM de 256 unidades** después de las convoluciones. Eso cambia la división del trabajo por completo:

- Las convoluciones dilatadas construyen una **representación local rica por frame**: periodicidad, estructura formántica, transitorios. Su campo receptivo solo necesita cubrir la escala del fenómeno acústico local — el orden de 20–50 ms, es decir el escenario C o algo entre C y D.
- Las **LSTM aportan el contexto largo**, con memoria en principio ilimitada, operando sobre una secuencia mucho más corta.

Esto es exactamente la arquitectura **CLDNN** (Convolutional, Long short-term memory, Deep neural network) de Sainath et al., *Learning the speech front-end with raw waveform CLDNNs*, Interspeech 2015 — **que WaveNet cita explícitamente en §3.4** como parte del movimiento hacia el audio crudo. El Ejemplo 2 no es una WaveNet discriminativa; es una CLDNN con el front-end convolucional dilatado.

**Corolario de ingeniería, y es el punto más importante en la práctica:** para que las LSTM sean viables, la secuencia que reciben **no puede tener 16.000 pasos por segundo**. Una LSTM de 256 unidades sobre 16.000 timesteps por segundo de audio es intratable en entrenamiento (BPTT sobre 16.000 pasos) e innecesaria. Tiene que haber **stride o pooling** en la pila convolucional. El propio WaveNet lo hizo así en el experimento discriminativo de §3.4: **mean-pooling a frames de 10 ms, submuestreo de 160×**, dejando 100 frames por segundo. Ese es el número al que hay que llegar. Si el Ejemplo 2 no especifica strides, hay que asumirlos o agregar pooling; es el detalle que decide si el modelo entrena o no.

**Diferencia 2: en clasificación la causalidad no hace falta, y la ventana se puede centrar.**

WaveNet **debe** ser causal porque la factorización autorregresiva se rompe si el modelo ve el futuro (§5). El Ejemplo 2 es un clasificador: se le entrega una ventana de audio completa y se le pide una etiqueta. No hay ningún orden de generación que respetar, y **nada obliga a que el filtro mire solo hacia atrás**. El propio WaveNet abandona la causalidad en cuanto cambia de tarea: en §3.4 el pooling va "seguido de **unas pocas convoluciones no causales**".

La consecuencia hay que enunciarla con precisión, porque la formulación coloquial ("duplica el campo receptivo") es imprecisa. El número total de muestras que influye en una salida **es el mismo**: $R = 1 + \sum(k_l-1)d_l$ en ambos casos. Lo que cambia es **dónde está la ventana respecto del punto de interés**:

| | Convolución causal | Convolución centrada (no causal) |
|---|---|---|
| Ventana que ve la salida en $t$ | $[t - R + 1,\; t]$ | $[t - \tfrac{R-1}{2},\; t + \tfrac{R-1}{2}]$ |
| Contexto pasado | $R-1$ muestras | $\tfrac{R-1}{2}$ muestras |
| Contexto futuro | **0** | $\tfrac{R-1}{2}$ muestras |
| $R$ necesario para $\pm W$ de contexto | $2W+1$ (y la mitad se desperdicia) | $W\cdot 2 + 1$ usado simétricamente |

La forma correcta de decirlo: **para obtener $W$ muestras de contexto simétrico a cada lado del instante que se está clasificando, la convolución centrada necesita $R = 2W+1$; la causal necesitaría también $R=2W+1$ pero todo hacia atrás, y por lo tanto necesitaría $R = 4W+1$ para tener $W$ de contexto futuro además de $W$ de pasado — lo cual no puede conseguir por construcción, porque nunca mira adelante.** El factor 2 es real, pero está en la **utilidad** de la ventana, no en su tamaño.

Por qué eso importa acústicamente: para decidir si un instante pertenece a una oclusiva hay que ver **la explosión que viene después**, no solo el silencio previo. Para clasificar una transición de formante hay que ver a dónde va. La coarticulación es bidireccional: un fono se ve afectado tanto por el que lo precede como por el que lo sigue. Un modelo causal es estructuralmente peor para clasificación de habla, y por eso todos los reconocedores serios usan contexto bidireccional (BiLSTM, convoluciones centradas, atención completa) salvo cuando la latencia de streaming lo prohíbe.

Con el escenario E y filtros centrados, cada salida del Ejemplo 2 vería **±312 ms alrededor de su posición** a 16 kHz. Eso cubre la sílaba con holgura y llega al orden de palabra. Es un front-end acústico perfectamente razonable.

**Diferencia 3, menor pero conviene registrarla: la salida.** WaveNet emite una categórica de 256 clases **por muestra** y se entrena con máxima verosimilitud sobre la señal. El Ejemplo 2 emite una etiqueta **por ventana** tras dos capas FC de 1024, y se entrena con cross-entropy sobre la clase. La pila convolucional dilatada es idéntica en espíritu; todo lo demás cambia. Y hay un puente que el paper mismo tendió: el experimento de TIMIT usó **dos pérdidas simultáneas** —predecir la muestra siguiente y clasificar el frame— y reportó que "el modelo generalizó mejor que con una sola pérdida". Es decir: **el objetivo generativo funciona como regularizador auxiliar del objetivo discriminativo**. Esa idea, en 2016 casi anecdótica, es hoy el fundamento del preentrenamiento autosupervisado de wav2vec 2.0 y HuBERT.

---

## 13. Erratas, matices y cosas que se citan mal

**1. "WaveNet superó al habla humana." Falso.** Tabla 1: WaveNet (L+F) obtuvo 4.21 en inglés contra 4.55 del PCM lineal de 16 bits y 4.46 del $\mu$-law de 8 bits. En mandarín, 4.08 contra 4.21 y 4.25. **En ningún idioma ni contra ninguna referencia natural WaveNet igualó al habla real.** El paper solo afirma haber reducido la brecha "en más del 50%".

**2. El MOS de 4.21 corresponde a WaveNet (L+F), no a "WaveNet".** La Tabla 1 tiene **una sola fila de WaveNet**, la condicionada sobre features lingüísticas **y** $\log F_0$. **WaveNet (L) no tiene MOS reportado en ninguna parte del paper**: solo aparece en las pruebas de preferencia (Tabla 2). Citar "WaveNet obtuvo MOS 4.21" sin la calificación L+F omite que ese número depende de un modelo externo de predicción de $F_0$.

**3. El "mejor baseline" es un sistema distinto en cada idioma.** En inglés es el concatenativo (3.86 > 3.67 del LSTM); en mandarín es el paramétrico LSTM (3.79 > 3.47 del concatenativo). El cálculo del cierre de brecha —51% y 69%— usa el mejor de cada idioma, lo cual es correcto pero hace que los dos porcentajes no sean directamente comparables entre sí.

**4. El techo real de WaveNet no es 4.55, es 4.46.** Como genera en 8 bits $\mu$-law, su máximo alcanzable es la fila *Natural (8-bit $\mu$-law)*. Su distancia al techo alcanzable es **0.25**, no 0.34. El cálculo del paper contra el PCM de 16 bits es defendible (es la referencia de calidad absoluta) pero atribuye a WaveNet una deficiencia que en parte es del esquema de cuantización.

**5. Anomalía en la Tabla 1 que nadie comenta: en mandarín, el audio natural de 8 bits $\mu$-law (4.25) puntuó más alto que el de 16 bits PCM (4.21).** Los intervalos se solapan (±0.082 y ±0.071), así que es ruido, pero indica que la resolución del instrumento de medición está en el límite de las diferencias que se quieren detectar. Cualquier lectura fina de esas dos filas debe hacerse con esa reserva.

**6. $\mu = 255$, no 256.** El paper es explícito: "$-1 < x_t < 1$ y $\mu = 255$". Es $2^8 - 1$, siguiendo G.711. Aparece mal transcrito con frecuencia. La razón de fondo es que $\ln(1+\mu) = \ln(256)$, lo que hace que $f(1) = 1$ exactamente y que la razón de pasos de cuantización entre el extremo y el cero sea exactamente $1+\mu = 256$.

**7. La fórmula del paper es la compansión, no la cuantización completa.** $f(x_t)$ mapea $[-1,1]\to[-1,1]$; la cuantización a 256 niveles es un paso adicional que el paper describe en prosa. Reproducir solo la fórmula sin el paso de cuantización deja el pipeline incompleto.

**8. WaveNet no es un TTS end-to-end.** Usa el front-end lingüístico completo de los sistemas HMM de Google (formato de etiquetas de Zen 2006), un **modelo LSTM-RNN externo para las duraciones de fono** y una **CNN autorregresiva externa para el $\log F_0$**, ambos entrenados con MSE (Apéndice B). Es un **vocoder neuronal condicionado**. El primer TTS neuronal razonablemente end-to-end fue Tacotron (2017), y aun así usaba un vocoder aparte.

**9. Las dilataciones "$1,2,4,\dots,512$ repetidas 3 veces" son un ejemplo, no una configuración declarada.** El paper escribe literalmente "e.g." antes de la secuencia. **El paper nunca publica el número de capas, de bloques, ni los anchos de canal de ninguno de sus modelos.** Todas las reproducciones de WaveNet que usan "30 capas, 3 bloques, 256 canales skip" toman esos números de implementaciones de terceros, no del paper.

**10. Las cifras de campo receptivo son 240 ms (TTS, §3.2) y "unos 300 ms" (multi-hablante, §3.1); para música, "varios segundos" sin número.** A 16 kHz eso es 3.840 y 4.800 muestras. **Ninguna de las dos corresponde a un número entero de bloques de 10 capas con $k=2$** (que dan 1.024, 2.047, 3.070, 4.093, 5.116). Como el paper no publica la arquitectura, no es posible reconciliar las cifras: la traducción de milisegundos a capas es reconstrucción, no cita. Consigno esto explícitamente porque he visto reproducciones que afirman "el WaveNet de TTS tenía 30 capas" — eso da 191.9 ms, no 240.

**11. Los 18.8 PER de TIMIT no son estado del arte de TIMIT.** El paper es cuidadoso: es "el mejor puntaje obtenido por un modelo entrenado **directamente sobre audio crudo** en TIMIT". Los sistemas sobre features log-mel de la época estaban por debajo. Citarlo como "WaveNet fue estado del arte en reconocimiento de voz" es una tergiversación.

**12. La preferencia mandarín entre WaveNet (L) y el LSTM paramétrico no fue significativa ($p = 0.476$, Tabla 2).** Es la única comparación no significativa del paper y desaparece de casi todos los resúmenes. Importa porque implica que el condicionamiento sobre $F_0$ era **necesario**, no opcional, para ganar en mandarín — lo cual tiene sentido en una lengua tonal.

**13. WaveNet no usa pooling en la pila generativa.** §2: "No hay capas de pooling en la red, y la salida del modelo tiene la misma dimensionalidad temporal que la entrada". El mean-pooling de 160× aparece **solo** en el experimento de reconocimiento de §3.4, junto con convoluciones no causales. Confundir ambos es confundir el modo generativo con el discriminativo.

**14. El paper no reporta ninguna log-verosimilitud**, pese a que §2 argumenta que su tractabilidad es una ventaja central. Toda la evidencia empírica es MOS y preferencias subjetivas, salvo el PER de TIMIT.

**15. El paper no reporta ninguna medición de velocidad de generación.** Las cifras de "~172 muestras/segundo" que circulan provienen del paper de Parallel WaveNet (2018), no de este. Este documento no menciona latencia ni tiempo real en ninguna parte.

**16. Lo que se desplegó en el Google Assistant en 2017 fue Parallel WaveNet, no este modelo**, y a 24 kHz / 16 bits, lo que implica que la softmax categórica de 256 clases sobre $\mu$-law fue reemplazada en producción.

**17. La convolución dilatada no la inventó WaveNet.** El paper la atribuye correctamente a Holschneider et al. (1989) y Dutilleux (1989) —el "algorithme à trous" de la transformada wavelet— y a Chen et al. (2015) y Yu & Koltun (2016) en segmentación de imágenes. La contribución de WaveNet es el **esquema de bloques con duplicación repetida** y la demostración de que resuelve el problema de contexto en audio.

**18. Sobre "una contraparte más eficiente y discriminativa de una convolución $1\times1024$" (§2.1):** la equivalencia es de **campo receptivo**, no de conectividad. Una conv de $1\times1024$ tiene 1024 pesos por par de canales y conecta densamente; el bloque dilatado tiene 20 pesos por par de canales, aplica 10 no linealidades y su grafo de conectividad es un árbol binario. El paper es honesto al llamarla "no lineal" y "más discriminativa", pero la frase se cita a veces como si fueran operaciones equivalentes.

**19. Este paper nunca fue publicado en una conferencia.** Es `arXiv:1609.03499v2`, del 19 de septiembre de 2016, y ahí se quedó. Vale tenerlo presente al evaluar las omisiones metodológicas de §10.3: no hubo revisores que las exigieran.

---

## 14. Cómo se ve hoy

Implementación de un bloque residual dilatado con activación con compuerta, con condicionamiento global y local, más el cálculo del campo receptivo. Es la traducción directa de las ecuaciones (2), del condicionamiento de §2.5 y de la Figura 4.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualBlock(nn.Module):
    """Bloque residual de WaveNet (Fig. 4 del paper).

    Flujo: conv causal dilatada -> gated activation (ec. 2) -> dos
    proyecciones 1x1, una hacia la rama residual y otra hacia el bus skip.

    Notas de implementacion no explicitas en el paper:
      - filtro y compuerta se computan con UNA sola conv de 2*gate_channels
        y se parten por la mitad; es matematicamente identico a tener W_f y
        W_g separadas, y aprovecha mejor el GEMM.
      - la suma residual se escala por sqrt(0.5) para que la varianza no
        crezca linealmente con la profundidad. El paper no lo menciona.
    """

    def __init__(self, residual_channels=64, gate_channels=64,
                 skip_channels=256, kernel_size=2, dilation=1,
                 local_cond_channels=None, global_cond_channels=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Padding causal: SOLO a la izquierda. Es lo unico que separa un
        # modelo autorregresivo valido de uno que aprende la identidad.
        self.left_pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(residual_channels, 2 * gate_channels,
                              kernel_size, dilation=dilation)

        # Condicionamiento local (seccion 2.5): serie temporal ya
        # upsampleada a la resolucion del audio; entra por una conv 1x1.
        self.cond_local = (
            nn.Conv1d(local_cond_channels, 2 * gate_channels, 1, bias=False)
            if local_cond_channels else None)

        # Condicionamiento global: proyeccion lineal V^T h, broadcast en t.
        self.cond_global = (
            nn.Linear(global_cond_channels, 2 * gate_channels, bias=False)
            if global_cond_channels else None)

        self.res_proj = nn.Conv1d(gate_channels, residual_channels, 1)
        self.skip_proj = nn.Conv1d(gate_channels, skip_channels, 1)

    def forward(self, x, y_local=None, h_global=None):
        # x: (B, residual_channels, T)
        h = F.pad(x, (self.left_pad, 0))          # padding causal
        h = self.conv(h)                          # (B, 2*gate, T)

        if self.cond_local is not None:
            h = h + self.cond_local(y_local)      # y_local: (B, C_loc, T)
        if self.cond_global is not None:
            bias = self.cond_global(h_global)     # (B, 2*gate)
            h = h + bias.unsqueeze(-1)            # broadcast sobre el tiempo

        f, g = h.chunk(2, dim=1)                  # filtro y compuerta
        z = torch.tanh(f) * torch.sigmoid(g)      # ecuacion (2)

        skip = self.skip_proj(z)                  # va directo a la salida
        out = (self.res_proj(z) + x) * (0.5 ** 0.5)
        return out, skip


def receptive_field(kernel_sizes, dilations, strides=None):
    """Campo receptivo, en muestras de entrada, de una pila 1D.

        R = 1 + sum_l (k_l - 1) * d_l * j_l ,   j_l = prod_{i<l} s_i

    El factor de salto j_l es lo que hace que el stride tambien amplie el
    campo receptivo: cada capa posterior a un stride ve unidades que ya
    cubren varias muestras de la entrada original.
    """
    if strides is None:
        strides = [1] * len(kernel_sizes)
    rf, jump = 1, 1
    for k, d, s in zip(kernel_sizes, dilations, strides):
        rf += (k - 1) * d * jump
        jump *= s
    return rf


def wavenet_receptive_field(n_blocks=3, layers_per_block=10, kernel_size=2,
                            initial_causal_kernel=2, sample_rate=16000):
    """Campo receptivo del esquema del paper: dilataciones 1,2,4,...,2^(n-1)
    repetidas en n_blocks bloques, precedidas de una conv causal no dilatada.
    Con kernel_size=2 cada bloque aporta 2^layers_per_block - 1 muestras.
    """
    ks = [initial_causal_kernel]
    ds = [1]
    for _ in range(n_blocks):
        for i in range(layers_per_block):
            ks.append(kernel_size)
            ds.append(2 ** i)
    rf = receptive_field(ks, ds)
    return rf, 1000.0 * rf / sample_rate  # (muestras, milisegundos)


def max_dilation_without_gridding(kernel_sizes):
    """Programa de dilataciones de crecimiento maximo SIN dejar huecos.

    Condicion: d_{l+1} <= R_l, el campo receptivo acumulado hasta la capa l.
    Con kernel 2 esto reproduce exactamente 1, 2, 4, 8, ... (igualdad en
    cada paso), que es por que el esquema del paper no sufre gridding.
    """
    dilations, rf = [], 1
    for k in kernel_sizes:
        d = rf                       # el maximo admisible
        dilations.append(d)
        rf += (k - 1) * d
    return dilations, rf


if __name__ == "__main__":
    # 1. Verificacion de la afirmacion del paper: un bloque 1..512 -> RF 1024
    ks = [2] * 10
    ds = [2 ** i for i in range(10)]
    assert receptive_field(ks, ds) == 1024

    # 2. Esquema completo del paper (3 bloques) a 16 kHz
    for b in (1, 2, 3, 4, 5):
        rf, ms = wavenet_receptive_field(n_blocks=b)
        print(f"{b} bloque(s): {rf:5d} muestras = {ms:6.1f} ms @16 kHz")
    # 1 bloque(s):  1025 muestras =   64.1 ms @16 kHz
    # 3 bloque(s):  3071 muestras =  191.9 ms @16 kHz
    # 5 bloque(s):  5116 muestras =  319.8 ms @16 kHz  <- ~300 ms de la sec 3.1

    # 3. Ejemplo 2 de la clase 39, bajo distintos supuestos de dilatacion
    ej2 = [20, 10, 10, 5]
    for nombre, d in [("sin dilatacion", [1, 1, 1, 1]),
                      ("duplicacion",    [1, 2, 4, 8]),
                      ("cuadruplicacion",[1, 4, 16, 64]),
                      ("potencias de 10",[1, 10, 100, 1000]),
                      ("prod. kernels",  [1, 20, 200, 2000])]:
        rf = receptive_field(ej2, d)
        print(f"{nombre:>17}: {rf:6d} muestras = {1000*rf/16000:7.1f} ms")
    #    sin dilatacion:     42 muestras =     2.6 ms
    #       duplicacion:    106 muestras =     6.6 ms
    #   cuadruplicacion:    456 muestras =    28.5 ms
    #   potencias de 10:   5010 muestras =   313.1 ms
    #     prod. kernels:  10000 muestras =   625.0 ms

    # 4. El programa optimo sin gridding para esos kernels
    print(max_dilation_without_gridding(ej2))   # ([1, 20, 200, 2000], 10000)

    # 5. Test de causalidad. Perturbar el futuro NO debe cambiar el pasado.
    torch.manual_seed(0)
    blk = DilatedResidualBlock(residual_channels=8, gate_channels=8,
                               skip_channels=8, kernel_size=2, dilation=4)
    x = torch.randn(1, 8, 64)
    y1, _ = blk(x)
    x2 = x.clone(); x2[:, :, 40:] += 100.0       # destrozar el futuro
    y2, _ = blk(x2)
    assert torch.allclose(y1[:, :, :40], y2[:, :, :40], atol=1e-5), \
        "fuga de informacion del futuro: el bloque no es causal"
    print("test de causalidad: OK")
```

Tres comentarios sobre el código, que son los tres lugares donde una implementación de WaveNet se rompe en la práctica:

**El test de causalidad del punto 5 es obligatorio.** Es el único diagnóstico que detecta el bug más común de estas arquitecturas —padding simétrico en vez de padding izquierdo, o un `crop` que recorta del lado equivocado—, porque ese bug produce pérdidas de entrenamiento *y de validación* excelentes y muestras generadas que son ruido. Ninguna curva de aprendizaje lo revela.

**El `receptive_field` con `jump` es la versión completa.** La fórmula sin strides es un caso particular. En arquitecturas discriminativas como el Ejemplo 2, donde casi con seguridad hay stride o pooling para bajar la tasa de frames antes de las LSTM, ignorar el factor de salto subestima gravemente el campo receptivo: cada capa posterior a un stride ve unidades que ya agregan varias muestras de la entrada.

**`max_dilation_without_gridding` convierte en código la condición $d_{l+1}\le R_l$ de §6.5.** Ejecutado sobre kernels de tamaño 2 devuelve exactamente $1, 2, 4, 8, \dots$, lo que demuestra que el esquema del paper es el crecimiento máximo sin huecos. Ejecutado sobre los kernels del Ejemplo 2 devuelve $1, 20, 200, 2000$ y $R = 10.000$ — que es la recomendación de diseño de §12.3, derivada y no adivinada.

Lo que falta en este código para tener un WaveNet completo: el bus de skip (sumar las skip de todos los bloques, `ReLU → 1×1 → ReLU → 1×1 → Softmax`), la codificación/decodificación $\mu$-law, la red de upsampling por convolución transpuesta para el condicionamiento local, y —lo más importante para que sea usable— la **generación incremental con colas circulares** que evita recomputar todo el campo receptivo en cada muestra. Ese último punto no es una optimización opcional: es la diferencia entre generar un segundo de audio en minutos o en horas.
