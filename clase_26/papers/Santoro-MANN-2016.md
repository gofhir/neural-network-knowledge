# One-shot Learning with Memory-Augmented Neural Networks (MANN) — Análisis interno exhaustivo

## 1. Metadata y resumen ejecutivo

**Cita completa.** Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, Timothy Lillicrap. *One-shot Learning with Memory-Augmented Neural Networks.* arXiv:1605.06065v1 [cs.LG], 19 de mayo de 2016. Google DeepMind (Bartunov adicionalmente afiliado a la National Research University Higher School of Economics, HSE). El trabajo fue presentado en ICML 2016.

**Posicionamiento.** Este paper es, junto con Matching Networks (Vinyals et al., 2016) y la línea de optimization-as-a-model (Ravi & Larochelle, 2017), uno de los textos fundacionales del *meta-learning* moderno con redes profundas. Su contribución central no es un nuevo benchmark sino una *hipótesis arquitectónica*: separar explícitamente dos escalas temporales de aprendizaje. Por un lado, un aprendizaje lento, gradual, que vive en los pesos $\theta$ entrenados por descenso de gradiente y que captura conocimiento transversal a las tareas (cómo extraer buenas representaciones de píxeles, cómo "atar" representaciones a etiquetas). Por otro lado, un aprendizaje rápido, episódico, que vive en una **memoria externa direccionable** $M_t$ y que almacena información específica del episodio actual tras una sola presentación. Los autores acuñan el uso del término MANN (memory-augmented neural network) para referirse específicamente a la clase de redes con memoria externa, en oposición a redes con memoria "interna" como las LSTM.

**Dos aportes técnicos concretos.**

1. **Un setup de meta-learning episódico** en el que las etiquetas se presentan con *offset temporal* —la red ve $(x_t, y_{t-1})$, no $(x_t, y_t)$— y en el que clases, etiquetas y muestras se barajan entre episodios. Esta combinación hace que memorizar en los pesos sea inútil y *fuerza* a la red a usar la memoria externa como mecanismo de binding rápido.

2. **El módulo LRUA (Least Recently Used Access)**, un mecanismo de escritura en memoria *puramente basado en contenido* que reemplaza el direccionamiento por ubicación (location-based addressing) de las Neural Turing Machines originales. LRUA escribe en la posición menos usada o en la más recientemente leída, interpolando entre ambas con una compuerta sigmoidea aprendible $\sigma(\alpha)$.

**Resultado estrella.** En clasificación one-shot sobre Omniglot (5 clases por episodio, etiquetas one-hot), el MANN alcanza una precisión por instancia de **36.4% (1.ª) / 82.8% (2.ª) / 91.0% (3.ª) / 92.6% (4.ª) / 94.9% (5.ª) / 98.1% (10.ª)**, superando a un humano (34.5 / 57.3 / 70.1 / 71.8 / 81.4 / 92.4) en *todas* las instancias y dejando muy atrás a un LSTM puro y a una red feedforward. El salto de ~36% a ~83% entre la primera y la segunda presentación de una clase es la firma del aprendizaje one-shot: la red ve un ejemplo, lo ata a su etiqueta, lo guarda, y al reencontrar la clase lo recupera.

**Relevancia para Roberto.** Más allá de Omniglot, MANN es el ancestro conceptual directo de la atención *key-value* que hoy domina los Transformers, y de los sistemas RAG y de memoria que aparecen en producción. La idea de "binding rápido en memoria externa, aprendizaje lento en pesos" se traduce casi literalmente a escenarios de salud con pocos ejemplos: clasificación de enfermedades raras, adaptación a un paciente nuevo con un puñado de observaciones, o sistemas de record-linkage donde cada par de registros es un "episodio" con estructura compartida pero contenido nuevo.

## 2. Contexto: NTM, memoria externa direccionable y el problema del one-shot learning

**El cuello de botella del deep learning clásico.** El paper abre constatando que el éxito del deep learning depende de aplicar optimización por gradiente a modelos de alta capacidad sobre grandes conjuntos de datos, con entrenamiento incremental y extenso: clasificación de imágenes (He et al., 2015), reconocimiento de voz (Yu & Deng, 2012), juegos (Mnih et al., 2015; Silver et al., 2016). El problema es que muchas tareas reales requieren *inferencia rápida desde pocos datos*. En el límite del **one-shot learning**, una sola observación debería producir un cambio abrupto y correcto de comportamiento.

Para una red entrenada por gradiente, el camino obvio ante pocos ejemplos nuevos —reentrenar los parámetros desde cero con los datos del momento— conduce a aprendizaje pobre y a **interferencia catastrófica** (catastrophic interference): los nuevos gradientes sobrescriben representaciones útiles ya aprendidas. Por eso los métodos no paramétricos (p. ej. k-NN) suelen considerarse mejor adaptados a este régimen: no "olvidan" porque simplemente almacenan.

**Meta-learning como estrategia.** La alternativa que el paper recoge es el meta-learning (Thrun 1998; Vilalta & Drissi 2002), un escenario donde el agente aprende en dos niveles asociados a escalas temporales distintas. El aprendizaje *rápido* ocurre dentro de una tarea (clasificar bien dentro de un dataset particular); el aprendizaje *lento* ocurre a través de muchas tareas y captura cómo varía la estructura de tarea entre dominios. Esta organización de dos niveles es lo que coloquialmente se llama "learning to learn". Hochreiter et al. (2001) ya habían mostrado que una LSTM entrenada para meta-aprender puede aprender funciones cuadráticas nunca vistas con pocos datos, usando su estado recurrente como memoria.

**Por qué la memoria interna no escala.** El argumento crítico del paper: la memoria implícita en arquitecturas recurrentes "no estructuradas" (LSTM) probablemente no escale a tareas donde cada episodio nuevo requiere codificar rápidamente *mucha* información. Una solución escalable necesita dos propiedades que las LSTM no ofrecen de forma natural:

1. La información debe almacenarse en una representación a la vez **estable** (recuperable de forma fiable) y **direccionable elemento a elemento** (acceso selectivo a piezas relevantes).
2. **El número de parámetros no debe estar atado al tamaño de la memoria**.

En una LSTM, ampliar la "memoria" significa ampliar el estado oculto, lo que infla el número de parámetros cuadráticamente y mezcla todo en un único vector denso difícil de direccionar selectivamente.

**Neural Turing Machines (Graves et al., 2014).** La NTM es la pieza que cumple ambos requisitos y es el antecedente directo del MANN. Es una implementación *totalmente diferenciable* de una MANN: un **controlador** (feedforward o LSTM) interactúa con un **módulo de memoria externa** —una matriz $M_t \in \mathbb{R}^{N\times m}$ de $N$ slots de tamaño $m$— mediante cabezales de **lectura y escritura** (read/write heads). La codificación y recuperación son rápidas: vectores entran y salen de memoria potencialmente en cada paso temporal. Crucialmente, el tamaño de $M_t$ es independiente del número de parámetros del controlador: agrandar la memoria no agranda la red. Esto convierte a la NTM en un candidato natural para one-shot: largo plazo vía actualizaciones lentas de pesos, corto plazo vía la memoria externa. El paper también menciona las Memory Networks (Weston et al., 2014) como otra arquitectura que cumple los criterios.

## 3. El problema formal del meta-learning episódico

**Objetivo de optimización.** En aprendizaje supervisado clásico se eligen parámetros $\theta$ que minimizan un costo $L$ sobre *un* dataset $D$. En meta-learning, en cambio, se minimiza el costo *esperado* sobre una **distribución de datasets** $p(D)$:

$$\theta^{*} = \arg\min_{\theta}\; \mathbb{E}_{D\sim p(D)}\big[L(D;\theta)\big]. \tag{1}$$

El cambio es sutil pero decisivo: $\theta$ ya no busca resolver un problema, sino *resolver problemas de una familia*. El conocimiento que se acumula en $\theta$ es meta-conocimiento sobre la estructura compartida de la familia, no sobre el contenido de ningún dataset particular.

**Episodio.** Una tarea o episodio es la presentación de un dataset $D = \{d_t\}_{t=1}^{T} = \{(x_t, y_t)\}_{t=1}^{T}$. Para clasificación, $y_t$ es la etiqueta de la imagen $x_t$; para regresión, $y_t = f(x_t)$ es el valor de una función oculta evaluada en $x_t$.

**El offset temporal de las etiquetas.** Aquí está el truco de diseño que define el paper. La etiqueta $y_t$ es *a la vez* el objetivo a predecir en el paso $t$ *y* una entrada que se le presenta a la red en el paso *siguiente*, con un desfase temporal. La secuencia de entrada que ve la red es:

$$(x_1, \text{null}),\; (x_2, y_1),\; (x_3, y_2),\; \ldots,\; (x_T, y_{T-1}).$$

Es decir, en el paso $t$ la red recibe la nueva consulta $x_t$ junto con la *etiqueta correcta del ejemplo anterior* $y_{t-1}$, y debe producir la etiqueta de $x_t$ (es decir, $y_t$). En la práctica el controlador recibe el vector concatenado $(x_t, y_{t-1})$.

**Por qué este offset es esencial.** Si la red recibiera $(x_t, y_t)$ simultáneamente, la tarea sería trivial y degenerada: bastaría copiar $y_t$ a la salida. El offset rompe ese atajo. En el paso $t$, la red ve $x_t$ pero *todavía no sabe* su etiqueta; debe arriesgar una predicción. Solo en el paso $t+1$ recibe $y_t$, y ese es el momento en que puede *atar* (bind) la representación de $x_t$ con su etiqueta verdadera y guardar el binding en memoria. Cuando más tarde aparece otra muestra de la misma clase, debe *recuperar* el binding y acertar.

**Por qué se barajan clases, etiquetas y muestras entre episodios.** Las etiquetas se barajan de dataset a dataset (label shuffling): la misma clase visual de Omniglot puede ser "etiqueta 3" en un episodio y "etiqueta 1" en otro. Esto impide que la red aprenda lentamente *bindings muestra-clase en sus pesos*. Si las etiquetas fueran consistentes entre episodios, $\theta$ podría memorizar la asociación "este carácter → clase 3" y la memoria externa sería innecesaria. Al barajar, la única asociación estable es *estructural* ("ata lo que veas a la etiqueta que venga después y recupéralo"), no de contenido. Esto es exactamente el meta-conocimiento explotable que se quiere forzar.

**Consecuencia sobre el rendimiento ideal.** Para un episodio dado, la conducta óptima es: *adivinar al azar* en la primera presentación de cada clase (la etiqueta no puede inferirse de episodios anteriores por el barajado) y luego usar la memoria para alcanzar precisión perfecta. El sistema modela la distribución predictiva

$$p\big(y_t \mid x_t, D_{1:t-1};\, \theta\big),$$

induciendo una pérdida en cada paso temporal. La estructura de la tarea incorpora meta-conocimiento explotable: un modelo que meta-aprende aprenderá a atar representaciones a sus etiquetas *independientemente del contenido concreto* de la representación o la etiqueta, y empleará un esquema general para mapear esos bindings a clases o valores de función.

## 4. Arquitectura del controlador

El controlador es el cerebro que decide qué leer y qué escribir; la memoria es solo un sustrato pasivo direccionable. El paper experimenta con dos tipos de controlador: feedforward y LSTM. El mejor rendimiento se obtiene con un **LSTM de 200 unidades ocultas**.

**Dinámica del LSTM controlador.** En cada paso, el controlador recibe la entrada concatenada $(x_t, y_{t-1})$ y actualiza su estado según las ecuaciones estándar de LSTM (numeración del suplemento del paper):

$$\hat{g}_f, \hat{g}_i, \hat{g}_o, \hat{u} = W_{xh}(x_t, y_{t-1}) + W_{hh}h_{t-1} + b_h, \tag{9}$$

$$g_f = \sigma(\hat{g}_f), \quad g_i = \sigma(\hat{g}_i), \quad g_o = \sigma(\hat{g}_o), \tag{10–12}$$

$$u = \tanh(\hat{u}), \tag{13}$$

$$c_t = g_f \odot c_{t-1} + g_i \odot u, \tag{14}$$

$$h_t = g_o \odot \tanh(c_t), \tag{15}$$

$$o_t = (h_t, r_t). \tag{16}$$

Aquí $g_f, g_i, g_o$ son las compuertas de olvido, entrada y salida; $c_t$ el estado de celda; $h_t$ el estado oculto; $r_t$ el vector leído de memoria; $\odot$ producto elemento a elemento; $(\cdot,\cdot)$ concatenación. $W_{xh}$ son los pesos de entrada al estado oculto y $W_{hh}$ los pesos recurrentes entre estados ocultos a través del tiempo.

**El punto clave de la ecuación (16).** La salida del controlador $o_t$ es la *concatenación* del estado oculto $h_t$ con el vector leído de memoria $r_t$. Es decir, el LSTM no es un clasificador autosuficiente: su predicción depende explícitamente de lo recuperado de la memoria externa. Esto crea el canal de gradiente que, durante el backprop a través del tiempo (BPTT), enseña al controlador *qué* clave generar para leer y escribir de forma útil.

**El doble rol del estado de celda.** En este diseño, el estado de celda del controlador sirve como *query* para la memoria. La clave $k_t$ que se usa para direccionar memoria se deriva del controlador. Esto significa que el LSTM aprende dos cosas simultáneamente: cómo procesar la secuencia (su rol recurrente clásico) y cómo emitir claves que indexen la memoria externa de forma semánticamente coherente.

## 5. Acceso a memoria: direccionamiento por contenido

La **lectura** se hace por content-based addressing. Dada la clave $k_t$ producida por el controlador, se calcula la **similitud coseno** entre $k_t$ y cada fila $M_t(i)$ de la memoria:

$$K\big(k_t, M_t(i)\big) = \frac{k_t \cdot M_t(i)}{\lVert k_t \rVert\, \lVert M_t(i)\rVert}. \tag{2/17}$$

Estas similitudes se normalizan mediante **softmax** para producir el vector de pesos de lectura $w_t^{r}$:

$$w_t^{r}(i) \leftarrow \frac{\exp\!\big(K(k_t, M_t(i))\big)}{\sum_j \exp\!\big(K(k_t, M_t(j))\big)}. \tag{3/18}$$

Y la memoria recuperada $r_t$ es la combinación convexa de las filas, ponderada por los pesos de lectura:

$$r_t \leftarrow \sum_i w_t^{r}(i)\, M_t(i). \tag{4/19}$$

**Lectura como atención suave.** Quien venga de Transformers reconocerá inmediatamente este patrón: $k_t$ es la query, las filas $M_t(i)$ son simultáneamente keys y values, la similitud coseno + softmax produce la distribución de atención, y $r_t$ es el contexto atendido. La única diferencia con la atención escalada de Vaswani et al. (2017) es el uso de coseno en lugar de producto punto escalado por $\sqrt{d_k}$, y que aquí $M_t$ es un estado *persistente y escribible*, no recalculado desde cero cada paso. MANN es, en este sentido, atención key-value con memoria de escritura.

**Múltiples lecturas.** El número de lecturas es un hiperparámetro libre. Se probaron 1 y 4 lecturas; **4 lecturas** fue lo elegido para los resultados reportados. Cada lectura adicional se implementa como *concatenación* adicional al vector de salida $o_t$, no como combinación o interpolación. Cuatro lecturas dan al controlador cuatro "ventanas" simultáneas a regiones distintas de la memoria, análogo a las cabezas múltiples de la multi-head attention.

## 6. LRUA (Least Recently Used Access): el módulo de escritura propuesto

Aquí está la contribución arquitectónica original del paper. La pregunta es: cuando llega información nueva que hay que guardar, **¿en qué slot de memoria escribir?**

**Por qué no el direccionamiento por ubicación de la NTM.** En la NTM original (Graves et al., 2014), la memoria se direcciona por contenido *y por ubicación*. El location-based addressing promueve pasos iterativos (avanzar por la "cinta") y saltos de larga distancia, lo que es ventajoso para tareas *secuenciales* donde el orden importa (copiar una secuencia, ordenar). Pero en one-shot la información tiene una **codificación conjuntiva independiente de la secuencia**: lo que importa es atar muestra↔etiqueta, no recordar en qué posición temporal apareció. El direccionamiento por ubicación es subóptimo aquí porque introduce un sesgo posicional irrelevante.

**La idea de LRUA.** LRUA es un escritor *puramente* basado en contenido que escribe la memoria nueva en una de dos posiciones:

- la posición **menos usada** (least-used), preservando así la información reciente codificada en otros slots; o
- la posición **leída más recientemente** (most recently used), que funciona como una *actualización* del slot con información nueva y posiblemente más relevante.

La distinción entre ambas opciones se logra interpolando entre los pesos de lectura previos y unos pesos escalados según el uso.

**Usage weights.** Se mantienen pesos de uso $w_t^{u}$ que registran qué posiciones fueron leídas o escritas recientemente. Se actualizan decayendo el uso previo y sumando los pesos de lectura y escritura actuales:

$$w_t^{u} \leftarrow \gamma\, w_{t-1}^{u} + w_t^{r} + w_t^{w}, \tag{5/20}$$

donde $\gamma$ es un parámetro de decaimiento (fijado en $0.99$ en los experimentos; el grid search lo lista como "usage decay of the write weights"). $w_t^{r}$ se calcula como en (3).

**Least-used weights.** Definida la notación $m(v, n)$ como el $n$-ésimo elemento más pequeño del vector $v$, los pesos de menor uso $w_t^{lu}$ son una máscara binaria:

$$w_t^{lu}(i) = \begin{cases} 0 & \text{si } w_t^{u}(i) > m(w_t^{u}, n) \\ 1 & \text{si } w_t^{u}(i) \le m(w_t^{u}, n) \end{cases}, \tag{6/21}$$

donde $n$ se fija igual al **número de lecturas a memoria** (es decir, 4 en los experimentos principales). En palabras: $w_t^{lu}$ marca con 1 los $n$ slots menos usados —los candidatos a ser sobrescritos sin pérdida valiosa.

**Write weights vía la compuerta $\sigma(\alpha)$.** Los pesos de escritura $w_t^{w}$ son una combinación *convexa* entre los pesos de lectura previos y los least-used previos, modulada por una compuerta sigmoidea aprendible:

$$w_t^{w} \leftarrow \sigma(\alpha)\, w_{t-1}^{r} + \big(1 - \sigma(\alpha)\big)\, w_{t-1}^{lu}, \tag{7/22}$$

con $\sigma(x) = \frac{1}{1+e^{-x}}$ y $\alpha$ un parámetro escalar de compuerta (dinámico/aprendible). La interpretación es elegante:

- Si $\sigma(\alpha) \to 1$: se escribe en la posición **leída más recientemente** ($w_{t-1}^{r}$), actualizando información reciente.
- Si $\sigma(\alpha) \to 0$: se escribe en la posición **menos usada** ($w_{t-1}^{lu}$), preservando todo lo demás y depositando la novedad en un slot "libre".

El gate $\alpha$ se aprende por gradiente, de modo que la red *descubre* la política de escritura óptima para la familia de tareas.

**Escritura efectiva.** Antes de escribir, la posición menos usada calculada desde $w_{t-1}^{u}$ se pone a cero (se borra). Luego se escribe sumando la clave ponderada por los pesos de escritura:

$$M_t(i) \leftarrow M_{t-1}(i) + w_t^{w}(i)\, k_t, \quad \forall i. \tag{8/23}$$

Así, las memorias pueden escribirse en el slot ya puesto a cero o en el previamente usado; en este último caso, las memorias menos usadas simplemente se borran. Nótese que la escritura es *aditiva* (suma a $M_{t-1}$), no un reemplazo destructivo, salvo por el zeroing previo del slot LRU.

**Por qué LRUA supera al direccionamiento por ubicación aquí.** El experimento de la Tabla 2 lo confirma cuantitativamente: con 15 clases y controlador LSTM, **MANN (LRUA)** alcanza 62.6 / 79.3 / 86.6 / 88.7 / 95.3 en las instancias 2–10, mientras que **MANN (NTM)** —con el módulo de acceso location-based estándar— se queda en 35.4 / 61.2 / 71.7 / 77.7 / 88.4. La brecha en la 2.ª instancia (62.6 vs 35.4) es enorme: LRUA atrapa el binding correcto casi al doble de tasa. La razón es que LRUA dedica toda su capacidad de direccionamiento al *contenido* relevante y gestiona la asignación de slots por *recencia de uso*, en lugar de gastar grados de libertad modelando una estructura secuencial que esta tarea no tiene.

## 7. El pipeline de un episodio: bind-and-encode y retrieve

El ciclo de vida de la información en un episodio MANN tiene dos fases entrelazadas:

**Fase bind-and-encode (atar y codificar).** En el paso $t$ la red ve $(x_t, y_{t-1})$. El controlador procesa $x_t$, genera una clave $k_t$ y lee la memoria (puede arriesgar una predicción para $x_t$). En el paso $t+1$, cuando llega $y_t$ junto con la nueva consulta $x_{t+1}$, el sistema ya dispone de la información necesaria para *atar* la representación de $x_t$ con su etiqueta verdadera $y_t$ y *escribir* ese binding en un slot de memoria vía LRUA. El backprop de la señal de error de la predicción modela los pesos de los pasos anteriores para promover esta estrategia de binding: el gradiente que llega desde "fallé al predecir $x_t$" empuja al controlador a haber generado una clave $k_t$ más distintiva y a haber escrito el binding de forma más recuperable.

**Fase retrieve (recuperar).** Cuando más tarde en el episodio aparece otra muestra de una clase ya vista, el controlador genera una clave que, por similitud coseno, alcanza un peso de lectura alto sobre el slot que contiene el binding correspondiente. La lectura $r_t = \sum_i w_t^r(i) M_t(i)$ trae de vuelta esa información, se concatena con $h_t$ y el clasificador emite la etiqueta correcta. Esta es la fase que produce el salto de precisión de la 2.ª instancia en adelante.

**Wiping entre episodios.** La estrategia impuesta por defecto es *borrar* la memoria externa de episodio a episodio. Como cada episodio tiene clases y etiquetas únicas, cualquier información que persista actúa como interferencia. La sección 4.2.1 estudia qué pasa cuando *no* se borra (ver §11).

## 8. Experimentos en Omniglot

**El dataset.** Omniglot consiste en más de **1600 clases** de caracteres con muy pocos ejemplos por clase, lo que le valió el apodo de "el transpuesto de MNIST" (Lake et al., 2015): donde MNIST tiene 10 clases con miles de ejemplos, Omniglot tiene miles de clases con decenas de ejemplos. Para reducir el sobreajuste se hizo data augmentation por traslación y rotación aleatorias, y se crearon clases nuevas por rotaciones de 90°, 180° y 270°. El entrenamiento usó las **1200 clases originales** (más augmentations); las **423 clases restantes** (clases 1201–1623) se reservaron para test. Las imágenes se reescalaron a **20×20**.

**Protocolo.** El MANN se entrenó con representaciones one-hot como etiquetas (Figura 2). Tras **100 000 episodios** con 5 clases elegidas al azar y etiquetas elegidas al azar, se evaluó en episodios de test *sin más aprendizaje* (pesos congelados), sobre clases nunca vistas del conjunto disjunto. La red mostró alta precisión ya en la segunda presentación de una clase dentro del episodio (82.8%), llegando a 94.9% en la quinta y 98.1% en la décima.

**Tabla 1 — Precisión por instancia (one-hot, 5 clases/episodio).**

| Modelo | 1.ª | 2.ª | 3.ª | 4.ª | 5.ª | 10.ª |
|---|---|---|---|---|---|---|
| Human | 34.5 | 57.3 | 70.1 | 71.8 | 81.4 | 92.4 |
| Feedforward | 24.4 | 19.6 | 21.1 | 19.9 | 22.8 | 19.5 |
| LSTM | 24.4 | 49.5 | 55.3 | 61.0 | 63.6 | 62.5 |
| **MANN** | **36.4** | **82.8** | **91.0** | **92.6** | **94.9** | **98.1** |

Lectura de la tabla. El feedforward es incapaz de aprender: ~20–24% en todas las instancias (azar es 20% para 5 clases), porque sin recurrencia ni memoria no puede acumular información dentro del episodio. El LSTM aprende algo (sube de 24.4 a ~62) pero se *satura* muy por debajo del techo: su estado oculto no escala como almacén direccionable de 5 bindings arbitrarios y barajados. El MANN no solo supera al humano en todas las instancias, sino que muestra el patrón cualitativo correcto: salto fuerte 1.ª→2.ª y aproximación asintótica a ~98%.

**Educated guessing.** Un detalle fascinante: el MANN supera el azar en la *primera* instancia (36.4% > 20%). Los autores lo explican como una estrategia de "adivinanza educada": si una muestra produce una clave que es *mal* match con todos los bindings ya almacenados, la red infiere que probablemente es una clase nueva y *evita* las etiquetas ya asignadas, aumentando su probabilidad de acertar la clase nueva. Los participantes humanos reportaron cualitativamente una estrategia similar.

**Curvas de aprendizaje.** La Figura 2 contrasta LSTM vs MANN en dos regímenes (5 clases one-hot; 15 clases con etiquetas string). El MANN exhibe un *spike* característico y rápido de precisión que el LSTM no logra. En el caso de 15 clases con strings (Fig. 2d), la precisión de 2.ª instancia se acerca a 80% durante el entrenamiento, y el rendimiento en clases de test retenidas es comparable al de entrenamiento —evidencia de meta-generalización, no de memorización.

**Etiquetas string para escalar.** Aprender clasificadores con vectores one-hot grandes se vuelve difícil al escalar el número de clases. Para permitir episodios con muchas más clases, los autores usaron etiquetas de **strings de cinco caracteres**, cada carácter tomando uno de cinco valores del conjunto {a,b,c,d,e}, p. ej. 'ecdba'. Representadas como cinco one-hot concatenados, los vectores son de longitud 25 (cinco posiciones a 1). Esto da $5^5 = 3125$ etiquetas posibles, casi el doble del número de clases del dataset, reduciendo enormemente la probabilidad de que una clase reciba la misma etiqueta en dos episodios (y, de paso, aboliendo en gran medida la estrategia de educated guessing, pues hay que adivinar entre 3125 strings).

**Tabla 2 — Precisión con etiquetas string, 100 000 episodios.**

| Modelo | Controlador | #Clases | 1.ª | 2.ª | 3.ª | 4.ª | 5.ª | 10.ª |
|---|---|---|---|---|---|---|---|---|
| kNN (raw pixels) | – | 5 | 4.0 | 36.7 | 41.9 | 45.7 | 48.1 | 57.0 |
| kNN (deep features) | – | 5 | 4.0 | 51.9 | 61.0 | 66.3 | 69.3 | 77.5 |
| Feedforward | – | 5 | 0.0 | 0.2 | 0.0 | 0.2 | 0.0 | 0.0 |
| LSTM | – | 5 | 0.0 | 9.0 | 14.2 | 16.9 | 21.8 | 25.5 |
| MANN | Feedforward | 5 | 0.0 | 8.0 | 16.2 | 25.2 | 30.9 | 46.8 |
| **MANN** | **LSTM** | **5** | **0.0** | **69.5** | **80.4** | **87.9** | **88.4** | **93.1** |
| kNN (raw pixels) | – | 15 | 0.5 | 18.7 | 23.3 | 26.5 | 29.1 | 37.0 |
| kNN (deep features) | – | 15 | 0.4 | 32.7 | 41.2 | 47.1 | 50.6 | 60.0 |
| Feedforward | – | 15 | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| LSTM | – | 15 | 0.0 | 2.2 | 2.9 | 4.3 | 5.6 | 12.7 |
| MANN (LRUA) | Feedforward | 15 | 0.1 | 12.8 | 22.3 | 28.8 | 32.2 | 43.4 |
| **MANN (LRUA)** | **LSTM** | **15** | **0.1** | **62.6** | **79.3** | **86.6** | **88.7** | **95.3** |
| MANN (NTM) | LSTM | 15 | 0.0 | 35.4 | 61.2 | 71.7 | 77.7 | 88.4 |

Lecturas clave de la Tabla 2. (1) El controlador *importa*: MANN-LSTM aplasta a MANN-feedforward, lo que confirma que la recurrencia interna y la memoria externa son complementarias, no redundantes. (2) LRUA > NTM (62.6 vs 35.4 en 2.ª instancia con 15 clases), validando el módulo de escritura propuesto. (3) MANN-LSTM > kNN incluso cuando el kNN usa deep features de un autoencoder y tiene **memoria ilimitada** con almacenamiento/recuperación automáticos de *todos* los ejemplos vistos. El autoencoder (encoder/decoder de dos capas de 200 unidades, leaky ReLU, cuello de botella de 32 unidades) tenía significativamente *más* parámetros que el MANN y entrenó con el triple de datos aumentados, y aun así fue claramente superado. Las primeras instancias del kNN son $1/N^2$ (4% para 5 clases, 0.4% para 15) por la combinación de azar inicial y la imposibilidad de clasificar clases aún no observadas.

**Curriculum training.** Para escalar más, se usó un régimen de currículum: empezar con 15 clases por episodio y, cada 10 000 episodios, incrementar en uno el máximo de clases (Figura 4), con longitud de episodio escalada a 10× el máximo. La red mantuvo alta precisión al subir el número de clases. Tras entrenar (en el marco de 100 000 episodios, ya con 25 clases), se evaluó en episodios con hasta 50 clases, escalando luego hasta 100. El rendimiento decae gradualmente hacia las 100 clases pero el límite de entrenamiento parecía no haberse alcanzado.

## 9. Experimentos de regresión y clases nuevas held-out

**Regresión sobre funciones de un GP.** Para mostrar que MANN genera una estrategia *general* de meta-learning y no un truco de clasificación, se probó en regresión sobre funciones nunca vistas. Se generaron funciones muestreando de un **prior de proceso gaussiano (GP)** con hiperparámetros fijos, entrenando con funciones únicas por episodio. Cada episodio presenta x-values (1, 2 o 3-dimensionales) junto con valores de función con offset temporal $f(x_{t-1})$. La estrategia exitosa es atar x-values con sus valores de función y guardar los bindings en memoria.

**Diferencia con clasificación: lectura mezclada.** Como cada x-value se presenta una sola vez por episodio, la predicción exitosa requiere un *look-up por contenido de información proximal* en memoria. A diferencia de la clasificación, esta tarea demanda una **lectura más amplia/mezclada** de memoria: la red debe *interpolar* entre puntos previamente vistos, lo que probablemente implica una lectura más "blended" de varios slots. Esa estrategia interpolativa es menos obvia (y probablemente innecesaria) en clasificación de imágenes.

**Comparación con el GP verdadero.** El rendimiento se comparó con predicciones GP exactas sobre las mismas muestras en el mismo orden. El GP tiene una ventaja estructural: puede hacer consultas complejas sobre *todos* los puntos en un solo paso (inversión de la matriz de covarianza) y se inicializó con los hiperparámetros *correctos* de la función muestreada. El MANN, en cambio, solo hace actualizaciones *locales* a su memoria y por tanto solo puede *aproximar* esa funcionalidad. Aun así, la Figura 5 muestra que las predicciones del MANN siguen la función subyacente, con la *varianza de salida creciendo* a medida que predice valores alejados de los ya observados —una incertidumbre calibrada cualitativamente correcta. Los resultados se extendieron a 2D y 3D (Figura 6): la log-verosimilitud del MANN sigue apreciablemente bien la del GP, con predicciones más precisas a medida que se almacenan más muestras en memoria.

**Clases held-out.** El protocolo de test usa siempre el conjunto disjunto (clases 1201–1623 de Omniglot) con pesos congelados. Que la precisión en test sea comparable a la de entrenamiento es la evidencia central de que MANN meta-aprendió una *estrategia* transferible y no memorizó clases concretas.

## 10. Por qué importa: dos escalas de aprendizaje y la analogía con el hipocampo

La tesis profunda del paper es la **separación explícita de dos mecanismos de aprendizaje**:

- **Aprendizaje gradual de pesos** ($\theta$ vía gradiente): lento, integra sobre miles de episodios, codifica conocimiento de fondo transversal a tareas. Análogo a la consolidación cortical lenta.
- **Almacenamiento rápido en memoria** ($M_t$ vía LRUA): instantáneo, específico del episodio, se borra entre tareas. Análogo a la memoria de trabajo y al hipocampo.

Esta dicotomía es la **teoría de sistemas de memoria complementarios** (complementary learning systems) traducida a arquitectura. El paper enmarca explícitamente al MANN como una arquitectura con un recurso de memoria *dedicado, direccionable y estructuralmente independiente* de los mecanismos que implementan el control de proceso. Esa independencia estructural es la que da las dos propiedades de escalabilidad de §2: memoria estable y direccionable, y desacople parámetros↔tamaño de memoria.

**Inductive transfer.** Un aspecto crítico es que las tareas no pueden resolverse por memoria mecánica (rote memory). La información nueva debe almacenarse y accederse flexiblemente, pero el desempeño correcto exige *más* que recuperación precisa: exige extraer inferencias sobre datos nuevos a partir de experiencia de más largo plazo —la facultad a veces llamada *inductive transfer*. MANN reúne ambas: almacenamiento flexible (memoria) + capacidad de representación rica (arquitectura profunda).

**Analogía cognitiva.** El paper sugiere que MANN puede ser un modelo heurístico útil del meta-learning humano. Supera a humanos incluso en set-sizes que no deberían saturar la memoria de trabajo humana (limitada a "un puñado" de bindings arbitrarios, Cowan 2010). Y cuando la memoria *no* se borra entre tareas, MANN sufre *interferencia proactiva*, como en muchos estudios de memoria humana (Underwood 1957) —un paralelismo conductual que refuerza la analogía.

## 11. Limitaciones

**Escalabilidad de la memoria.** La memoria tiene 128 slots de tamaño 40 en los experimentos. Al escalar a 50–100 clases (curriculum), el rendimiento decae gradualmente. La interferencia entre bindings crece y la capacidad de discriminación por similitud coseno se degrada cuando muchos slots compiten. El paper deja abierta la "capacidad máxima de la red" como trabajo futuro.

**Interferencia persistente (sección 4.2.1).** Cuando *no* se borra la memoria entre episodios, MANN se vuelve menos robusto (Figura 3). En el caso de 5 clases one-hot, el aprendizaje progresa mucho más lento y *no* produce el spike característico de la condición de memoria-borrada. Curiosamente, hay configuraciones donde la interferencia no daña apreciablemente (10 clases, episodios de longitud 75, alcanza niveles comparables). Es decir, MANN depende de un *reset* externo de memoria para rendir bien; gestionar la interferencia de forma autónoma queda pendiente.

**Dependencia del controlador.** El rendimiento depende fuertemente del controlador: MANN-feedforward es muy inferior a MANN-LSTM (Tabla 2). La memoria externa por sí sola no basta; necesita un controlador con estado interno capaz de generar buenas claves y orquestar lecturas/escrituras. Esto complica el análisis: no siempre es trivial atribuir el desempeño a la memoria vs al controlador.

**Comparación con métodos métricos.** El paper compara con kNN (un método métrico no paramétrico) pero *no* con Matching Networks (Vinyals et al., 2016), publicado casi en paralelo, que ataca el mismo problema con atención sobre un soft-nearest-neighbor end-to-end. En retrospectiva, los enfoques métricos (Prototypical Networks, Relation Networks) resultaron más simples y a menudo más fuertes en few-shot puro que las arquitecturas memory-augmented, que son más pesadas de entrenar (BPTT largo, dinámica de memoria sensible a hiperparámetros).

**Aproximación local vs global.** En regresión, MANN solo hace actualizaciones locales a memoria y aproxima lo que un GP resuelve globalmente en un paso. Para tareas que requieren razonamiento global sobre todos los datos del episodio, el binding slot-a-slot es una limitación estructural.

**Active learning ausente.** Las tareas presentan las muestras pasivamente; MANN no *selecciona* qué observar. El paper señala el active meta-learning como dirección futura.

## 12. Legado

**Influencia directa.** MANN consolidó la línea de **modelos memory-augmented** para few-shot y meta-learning. Inspiró arquitecturas como **Meta Networks** (Munkhdalai & Yu, 2017), que añaden pesos de rápida adaptación, y **SNAIL** (Mishra et al., 2018, *Simple Neural Attentive Meta-Learner*), que combina convoluciones temporales con atención causal para agregar experiencia del episodio —esencialmente reemplazando la memoria explícita por atención sobre la historia. La idea de "memoria episódica diferenciable" reaparece en Differentiable Neural Computers (Graves et al., 2016, el sucesor de la NTM), Memory Networks end-to-end (Sukhbaatar et al., 2015) y en líneas de *episodic memory* para RL.

**El puente hacia los Transformers.** La conexión más importante para entender la trayectoria del campo: la lectura de MANN —query $k_t$, keys/values $M_t(i)$, similitud + softmax, contexto $r_t = \sum_i w_t^r(i) M_t(i)$— es estructuralmente la **atención key-value** que un año después se vuelve el corazón de los Transformers (Vaswani et al., 2017). La self-attention de un Transformer puede leerse como una MANN sin escritura persistente: la "memoria" es el conjunto de tokens del contexto, recalculada cada forward. Inversamente, los Transformers con memoria externa o recurrencia (Transformer-XL, Compressive Transformer, modelos con KV-cache y memoria de largo plazo) reintroducen la escritura persistente que MANN ya proponía. La distinción coseno-vs-producto-punto y memoria-escribible-vs-recalculada es lo único que separa conceptualmente ambos.

**Few-shot como subcampo.** Junto con Matching Networks, MANN ayudó a establecer Omniglot y luego miniImageNet como benchmarks estándar de few-shot, y a formalizar el protocolo *N-way K-shot* con episodios. El meta-learning episódico que aquí se describe es el mismo marco que luego usan MAML (Finn et al., 2017) y toda la familia de optimization-based meta-learning, aunque MAML mueve el "fast learning" de la memoria a unos pasos de gradiente interno.

## 13. Conexión con la Clase 26 y relevancia para salud

**En el arco de la Clase 26.** MANN ocupa el lugar del *primer puente diferenciable entre redes profundas y memoria externa direccionable* aplicado a aprendizaje con pocos ejemplos. En la narrativa de la clase, conecta hacia atrás con las Neural Turing Machines (memoria como cinta diferenciable) y hacia adelante con la atención key-value que define a los Transformers. Si la clase trata meta-learning, memoria o el linaje de la atención, MANN es la pieza que muestra *por qué* y *cómo* desacoplar "aprender a representar" (lento, en pesos) de "recordar lo recién visto" (rápido, en memoria), y por qué el offset temporal de etiquetas y el barajado son los que *fuerzan* esa separación.

**Relevancia para salud y oncología (FALP).** Tres traducciones concretas:

1. **Enfermedades raras y clases de cola larga.** El régimen one-shot/few-shot es la realidad clínica de las patologías raras: hay miles de condiciones con muy pocos casos cada una —el "transpuesto de MNIST" es literalmente el perfil de un registro hospitalario de baja prevalencia. Un sistema entrenado para meta-aprender la *estructura* del diagnóstico (cómo atar hallazgos a una condición) podría adaptarse a una condición nueva con un puñado de casos, sin reentrenar y sin interferencia catastrófica sobre lo ya aprendido.

2. **Adaptación a paciente nuevo.** Cada paciente puede verse como un episodio con estructura compartida (fisiología general) pero contenido propio (su historia particular). La memoria externa borrable por episodio modela bien la separación entre conocimiento poblacional (pesos) y estado individual (memoria), evitando que los datos de un paciente contaminen las predicciones de otro.

3. **Record-linkage / patient-matching.** Para el trabajo de Roberto en FHIR y matching de pacientes: cada par o bloque de registros candidatos es un "episodio" donde la tarea estructural (decidir si dos registros son la misma persona) es estable, pero el contenido (nombres, fechas, identificadores) es nuevo. La lógica MANN de "atar representación↔decisión en memoria de contenido y recuperar por similitud" es directamente análoga a un blocker/scorer basado en embeddings con memoria de ejemplos resueltos. La calibración de incertidumbre que MANN exhibe en regresión (varianza que crece lejos de lo observado) es además deseable en cualquier scorer clínico: saber *cuándo no sabe* importa tanto como acertar.

**Lección de ingeniería transferible.** La moraleja de MANN para un practicante: cuando el problema exige incorporar información nueva rápido sin destruir lo aprendido, no reentrenes los pesos —dale al modelo una *memoria externa direccionable* y entrena los pesos para *usarla bien*. Esa es exactamente la intuición que hoy sostiene los sistemas RAG y de memoria sobre LLMs, y MANN es donde esa intuición se formalizó por primera vez de extremo a extremo y diferenciable.
