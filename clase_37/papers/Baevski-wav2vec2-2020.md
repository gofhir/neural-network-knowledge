# wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*.
- **Autores:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli. Todos en **Facebook AI (FAIR)**.
- **Venue:** *34th Conference on Neural Information Processing Systems (NeurIPS 2020)*.
- **Año:** 2020. **Preprint:** arXiv:2006.11477v3 (22 oct 2020), [arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477).
- **Código y modelos:** [github.com/pytorch/fairseq](https://github.com/pytorch/fairseq).
- **Linaje:** es la culminación de una línea de trabajo del mismo grupo: **wav2vec** (Schneider et al., 2019, predicción de pasos futuros) y **vq-wav2vec** (Baevski et al., 2020, cuantización en una etapa previa separada). wav2vec 2.0 resuelve *end-to-end* lo que sus predecesores hacían en dos etapas.

El paper demuestra, según los autores **por primera vez**, que aprender representaciones potentes a partir de **audio de voz solo** (sin transcripciones) y luego hacer *fine-tuning* sobre voz transcrita puede **superar a los mejores métodos semi-supervisados**, siendo además conceptualmente más simple. El modelo enmascara la entrada de voz en el **espacio latente** y resuelve una **tarea contrastiva** definida sobre una **cuantización** de las representaciones latentes, aprendidas conjuntamente.

El hallazgo estrella tiene dos caras. Con **todos** los datos etiquetados de Librispeech (960 h) se alcanza **1.8/3.3 WER** en los sets test clean/other, marcando estado del arte en voz ruidosa. Pero lo verdaderamente disruptivo es el régimen de **bajos recursos**: bajando la etiqueta a una sola hora, wav2vec 2.0 **supera el estado del arte anterior sobre el subset de 100 horas usando 100 veces menos datos etiquetados**; y con **apenas 10 minutos** de audio etiquetado (48 grabaciones de 12.5 s en promedio) más *pre-training* sobre **53.000 horas** sin etiquetar, se logra **4.8/8.2 WER**. Esto demuestra la viabilidad del reconocimiento de voz con cantidades muy pequeñas de datos anotados.

Para la **Clase 37 (Datasets y Herramientas para Audio)**, este paper importa porque encarna una de las **representaciones del audio** que la clase presenta: los **embeddings preentrenados**. En vez de calcular MFCC "a mano" —features acústicas diseñadas por humanos—, wav2vec 2.0 aprende, de forma autosupervisada, *features* que capturan la estructura de la voz. El anexo del laboratorio ("Embeddings preentrenados: wav2vec 2.0") es precisamente la puerta de entrada práctica a este cambio de paradigma: **pre-training masivo autosupervisado + fine-tuning ligero**.

## 2. Contexto: por qué el ASR supervisado era caro y cómo el SSL llegó al audio

Las redes neuronales se benefician de grandes cantidades de datos etiquetados, pero en muchos escenarios los datos **etiquetados son mucho más escasos que los no etiquetados**. Los sistemas de reconocimiento de voz (ASR) de la época requerían **miles de horas de voz transcrita** para alcanzar un desempeño aceptable, algo simplemente no disponible para la inmensa mayoría de las cerca de **7.000 lenguas** habladas en el mundo. Transcribir audio es un trabajo humano lento y costoso; el cuello de botella no es el audio en sí (abundante), sino su **anotación**.

Los autores enmarcan el problema con una analogía cognitiva elegante: aprender solo de ejemplos etiquetados **no se parece a cómo los humanos adquieren el lenguaje**. Los bebés aprenden a hablar **escuchando** a los adultos a su alrededor, un proceso que exige aprender buenas representaciones de la voz **antes** de asociarlas a significados o transcripciones.

En *machine learning*, el **aprendizaje autosupervisado (SSL)** emergió justamente como el paradigma para aprender representaciones generales de datos a partir de ejemplos no etiquetados, y luego afinar el modelo sobre datos etiquetados. Esta receta había sido **particularmente exitosa en procesamiento de lenguaje natural** —los autores citan la línea ELMo/GPT/BERT— y era un área activa en visión por computador. wav2vec 2.0 es el trabajo que **traslada ese paradigma al audio crudo** de manera contundente, cerrando la brecha entre el SSL de NLP/visión y el habla.

El antecedente inmediato dentro del propio grupo es doble. **vq-wav2vec** aprendía una cuantización de los datos en un primer paso, y *después* entrenaba representaciones contextualizadas con un modelo de auto-atención (BERT sobre las unidades discretas) en un segundo paso: un **pipeline de dos etapas**. Otros trabajos previos que enmascaraban la entrada con Transformers para voz o bien dependían de ese pipeline de dos pasos, o entrenaban reconstruyendo *filter banks*. La apuesta de wav2vec 2.0 es resolver **ambos problemas de una sola vez, end-to-end**.

## 3. Contribución central

La contribución es un **marco unificado de SSL contrastivo con cuantización aprendida conjuntamente**, con estas ideas de diseño:

1. **Enmascaramiento en el espacio latente.** A diferencia del BERT de texto (que enmascara tokens de entrada) o de trabajos previos de voz que reconstruían *filter banks*, wav2vec 2.0 enmascara las **representaciones latentes producidas por un encoder convolucional** sobre la forma de onda cruda.
2. **Cuantización aprendida end-to-end.** En lugar de discretizar en una etapa separada (vq-wav2vec), las unidades discretas de voz se aprenden **al mismo tiempo** que las representaciones contextualizadas, vía **Gumbel-softmax**.
3. **Objetivo contrastivo sobre latentes cuantizados.** El modelo debe **identificar el latente cuantizado correcto** para cada paso enmascarado, distinguiéndolo de un conjunto de distractores. Los autores encuentran que targets **cuantizados** funcionan mejor que targets continuos.
4. **Demostración de ASR de ultra-bajos recursos.** El resultado empírico de que 10 minutos etiquetados bastan para un WER competitivo es la evidencia que **cambia el paradigma**: ya no hace falta transcribir miles de horas.

El resultado, dicen los autores, es que aprender conjuntamente unidades discretas y representaciones contextualizadas rinde **sustancialmente mejor** que fijar las unidades en un paso previo, reduciendo el WER en cerca de un tercio respecto de vq-wav2vec/Discrete BERT.

## 4. Método

El modelo se compone de tres bloques que fluyen desde la forma de onda cruda hacia representaciones contextualizadas, más un módulo de cuantización que produce los *targets* del objetivo autosupervisado (Figura 1 del paper).

### 4.1. Feature encoder convolucional: de la onda cruda a latentes

Un encoder convolucional multicapa $f: \mathcal{X} \mapsto \mathcal{Z}$ toma como entrada el audio crudo $X$ y produce **representaciones latentes de voz** $z_1, \dots, z_T$ para $T$ pasos de tiempo. Cada bloque contiene una **convolución temporal**, seguida de **layer normalization** y una activación **GELU**. La forma de onda de entrada se normaliza a media cero y varianza unitaria.

La configuración concreta: **siete bloques** convolucionales, cada uno con 512 canales, *strides* $(5,2,2,2,2,2,2)$ y anchos de kernel $(10,3,3,3,3,2,2)$. Esto produce una **frecuencia de salida de 49 Hz**, con un *stride* de unos **20 ms** entre muestras sucesivas y un **campo receptivo de 400 muestras** (25 ms de audio). Es decir, cada latente $z_t$ resume aproximadamente una ventana de 25 ms de voz, análoga a un *frame* acústico clásico —pero **aprendida**, no diseñada como las MFCC.

### 4.2. Transformer de contexto

La salida del encoder alimenta una **red de contexto** $g: \mathcal{Z} \mapsto \mathcal{C}$ con arquitectura **Transformer**, que construye representaciones $c_1, \dots, c_T$ que **capturan información de toda la secuencia**. En vez de *positional embeddings* fijos absolutos, se usa una **capa convolucional** (kernel 128, 16 grupos) que actúa como **embedding posicional relativo**: su salida, tras un GELU, se suma a la entrada y luego se aplica layer norm.

Se ensayan dos configuraciones que comparten el mismo encoder pero difieren en el Transformer:

- **BASE** (95 M de parámetros): 12 bloques Transformer, dimensión 768, dimensión interna (FFN) 3.072, 8 cabezas de atención.
- **LARGE** (317 M de parámetros): 24 bloques, dimensión 1.024, FFN 4.096, 16 cabezas.

Una diferencia central con vq-wav2vec: aquí el Transformer construye contexto sobre representaciones **continuas** (no cuantizadas) y la auto-atención captura dependencias sobre toda la secuencia de latentes **end-to-end**.

### 4.3. Módulo de cuantización con Gumbel-softmax

Para el objetivo autosupervisado, la salida $z$ del encoder se **discretiza** a un conjunto finito de representaciones de voz vía **cuantización por producto** (*product quantization*). Discretizar significa elegir entradas de **codebooks** y concatenarlas. Dados $G$ codebooks (o **grupos**), cada uno con $V$ entradas $e \in \mathbb{R}^{V \times d/G}$, se elige **una entrada de cada codebook**, se concatenan los vectores resultantes $e_1, \dots, e_G$, y se aplica una transformación lineal $\mathbb{R}^d \mapsto \mathbb{R}^f$ para obtener $q \in \mathbb{R}^f$.

El problema es que "elegir una entrada" es una operación **argmax discreta**, no diferenciable. La solución es el **Gumbel-softmax**, que permite seleccionar entradas del codebook de forma **completamente diferenciable**. La salida del encoder $z$ se mapea a logits $l \in \mathbb{R}^{G \times V}$, y la probabilidad de elegir la entrada $v$ del grupo $g$ es:

$$p_{g,v} = \frac{\exp\big((l_{g,v} + n_v)/\tau\big)}{\sum_{k=1}^{V} \exp\big((l_{g,k} + n_k)/\tau\big)}$$

donde $\tau$ es una **temperatura** no negativa, $n = -\log(-\log(u))$ es **ruido Gumbel** con $u$ muestras uniformes de $\mathcal{U}(0,1)$. En el *forward pass* se elige la palabra de código con $i = \arg\max_j p_{g,j}$ (selección dura, discreta), mientras que en el *backward pass* se usa el **gradiente verdadero del Gumbel-softmax** (el *straight-through estimator*): así el modelo elige discreto hacia adelante pero recibe gradiente continuo hacia atrás.

En la práctica se usa $G = 2$ y $V = 320$ para ambos modelos, lo que da un **máximo teórico de 102.400 palabras de código** ($320^2$). La temperatura $\tau$ se **recuece** de 2 a un mínimo de 0.5 (BASE) o 0.1 (LARGE) con factor 0.999995 por *update*: al inicio la selección es "blanda" (explora el codebook), y se endurece gradualmente.

### 4.4. Enmascaramiento estilo BERT sobre los latentes

Para el *pre-training* se enmascara una proporción de los pasos de tiempo en el espacio de salida del encoder, de forma análoga al *masked language modeling* de BERT. Se muestrea **sin reemplazo** una proporción $p$ de todos los pasos como **índices de inicio**, y desde cada índice se enmascaran los $M$ pasos consecutivos siguientes; los tramos **pueden solaparse**. Los pasos enmascarados se reemplazan por un **vector de *feature* entrenado**, compartido entre todos los pasos enmascarados. Detalle importante: **no** se enmascaran las entradas al módulo de cuantización —los *targets* discretos se calculan sobre los latentes limpios.

Con $p = 0.065$ y $M = 10$, aproximadamente el **49% de todos los pasos** quedan enmascarados, con un largo medio de tramo de 14.7 pasos, o **299 ms** de audio.

### 4.5. Objetivo de entrenamiento: pérdida contrastiva + pérdida de diversidad

El objetivo total combina dos términos:

$$L = L_m + \alpha L_d$$

donde $\alpha$ es un hiperparámetro ajustado (se usa $\alpha = 0.1$).

**Pérdida contrastiva $L_m$.** Dada la salida del contexto $c_t$ centrada en un paso enmascarado $t$, el modelo debe **identificar el latente cuantizado verdadero $q_t$** dentro de un conjunto de $K + 1$ candidatos $\tilde{q} \in Q_t$ que incluye a $q_t$ y $K$ **distractores**. Los distractores se muestrean uniformemente de **otros pasos enmascarados de la misma locución** (utterance). La pérdida es:

$$L_m = -\log \frac{\exp\big(\text{sim}(c_t, q_t)/\kappa\big)}{\sum_{\tilde{q}\sim Q_t} \exp\big(\text{sim}(c_t, \tilde{q})/\kappa\big)}$$

donde $\text{sim}(a,b) = a^\top b / \lVert a\rVert \lVert b\rVert$ es la **similitud coseno** entre representación de contexto y latente cuantizado, y $\kappa$ es una temperatura ($\kappa = 0.1$). Esta es exactamente la forma de una pérdida InfoNCE: un *softmax* sobre similitudes que empuja $c_t$ a **parecerse al target correcto** y a diferenciarse de los distractores. Se usan $K = 100$ distractores.

**Pérdida de diversidad $L_d$.** La tarea contrastiva depende del codebook para representar tanto positivos como negativos; sin regularización, el modelo podría colapsar a usar **pocas entradas**. La pérdida de diversidad fomenta el uso **equitativo** de las $V$ entradas de cada uno de los $G$ codebooks, **maximizando la entropía** de la distribución *softmax* promediada $\bar{p}_g$ sobre las entradas, por codebook, a través de un *batch* de locuciones (esta distribución **no** contiene el ruido Gumbel ni la temperatura):

$$L_d = \frac{1}{GV} \sum_{g=1}^{G} -H(\bar{p}_g) = \frac{1}{GV} \sum_{g=1}^{G} \sum_{v=1}^{V} \bar{p}_{g,v} \log \bar{p}_{g,v}$$

Equivalentemente, la implementación **maximiza la perplejidad** de la distribución promedio. La intuición: si algunas entradas del codebook nunca se usan, el inventario discreto efectivo se reduce y la tarea contrastiva se empobrece; la diversidad garantiza un vocabulario acústico rico.

### 4.6. Fine-tuning con CTC

Tras el *pre-training* no supervisado, el modelo se afina para reconocimiento de voz agregando una **proyección lineal inicializada al azar** sobre la red de contexto, hacia $C$ clases que representan el vocabulario de la tarea. Para Librispeech son **29 tokens** de caracteres más un token de frontera de palabra. Se optimiza minimizando una **pérdida CTC** (*Connectionist Temporal Classification*), que permite alinear una secuencia de salida más corta que la de entrada sin alineamiento explícito. Se aplica una versión modificada de **SpecAugment** (enmascarando pasos de tiempo y canales), que retrasa el *overfitting* y mejora significativamente el error, especialmente en los subsets de Libri-light con pocos ejemplos. Un detalle clave: **el feature encoder NO se entrena durante el fine-tuning** —se congela— y durante los primeros 10.000 *updates* solo se entrena el clasificador de salida.

## 5. Experimentos

### 5.1. Datos

Como **datos no etiquetados** se consideran dos fuentes: el corpus **Librispeech** sin transcripciones (960 h, "LS-960") o el audio de **LibriVox** (audiolibros de dominio público), que tras el preprocesamiento de Libri-light da **53.200 horas** ("LV-60k"). El *fine-tuning* se hace en **cinco regímenes** de datos etiquetados de tamaño decreciente: 960 h completas, el subset **train-clean-100** (100 h), y los subsets de Libri-light de recursos limitados: **train-10h** (10 h), **train-1h** (1 h) y **train-10min** (10 minutos). También se evalúa reconocimiento de fonemas sobre **TIMIT** (5 h de audio con etiquetas fonéticas detalladas, colapsadas a 39 clases).

### 5.2. Configuración de pre-training

Los modelos se implementan en **fairseq**. Se optimiza con **Adam**, con *warmup* del *learning rate* durante el primer 8% de *updates* hasta un pico de $5\times 10^{-4}$ (BASE) o $3\times 10^{-4}$ (LARGE), y luego decaimiento lineal. BASE entrena por 400k *updates* sobre **64 GPUs V100 durante 1.6 días**; LARGE por 250k–600k *updates* sobre **128 GPUs V100 durante 2.3–5.2 días**. El *batch* total efectivo es de 1.6 h (BASE) a 2.7 h (LARGE) de audio. Este costo de cómputo es la **inversión única** que después se amortiza en muchos *fine-tunings* baratos.

### 5.3. Evaluación de bajos recursos

Este es el corazón del paper. La lógica: si un modelo preentrenado captura la estructura de la voz, debería requerir **pocos ejemplos etiquetados** para afinarlo. Los resultados de la Tabla 1 (con LM Transformer) son elocuentes:

- **10 min etiquetados:** LARGE preentrenado en LV-60k logra **4.6/7.9** (dev) y **4.8/8.2** (test clean/other). Con solo 48 grabaciones de 12.5 s promedio. El Discrete BERT previo lograba 16.3/25.2.
- **1 h etiquetada:** LARGE/LV-60k llega a **2.9/5.8** (test). Esto **supera el estado del arte anterior sobre 100 horas** usando 100 veces menos etiqueta.
- **10 h etiquetadas:** **2.6/4.9** (test).
- **100 h etiquetadas:** **2.0/4.0** (test). Comparado con *iterative pseudo-labeling* (SOTA previo con 4.2/8.6), es una reducción relativa de WER de **45%/42%** en un setup comparable.

Los autores destacan que aprender conjuntamente unidades discretas y contexto **mejora sobre trabajos previos que cuantizaban en un paso separado**, reduciendo el WER en cerca de un tercio. Y que, contra el *noisy student* / *iterative self-training* (que requiere múltiples iteraciones de etiquetado, filtrado y reentrenamiento), su enfoque es **más simple**: preentrenar una vez y afinar.

### 5.4. Evaluación de altos recursos

Con las **960 horas completas** etiquetadas (Tabla 2), LARGE/LV-60k logra **1.8/3.3 WER** en test clean/other, nuevo estado del arte en el benchmark completo para voz ruidosa. Notablemente, esto se consigue **a pesar de una arquitectura base más débil**: el mismo modelo entrenado desde cero (supervisado, sin *pre-training*) logra solo 2.1/4.6, comparable a ContextNet (1.9/4.1). Es decir, **el pre-training aporta la diferencia**, no una arquitectura superior. Además usan un simple Transformer con CTC, que no rinde tanto como los modelos seq2seq.

### 5.5. TIMIT y análisis de las unidades discretas

En reconocimiento de fonemas TIMIT, el enfoque alcanza **nuevo estado del arte**, reduciendo el PER (phoneme error rate) en un **23%/29%** relativo sobre el mejor resultado previo (dev/test), llegando a **7.4/8.3** sin LM. El Apéndice D muestra un análisis revelador: al computar la co-ocurrencia entre los latentes discretos $q_t$ (sin *fine-tuning*) y fonemas anotados por humanos en TIMIT, **muchos latentes discretos se especializan en sonidos fonéticos específicos**. Es decir, la cuantización autosupervisada **descubre unidades cercanas a fonemas** sin que nadie se lo indicara: evidencia directa de que aprende estructura lingüística real.

### 5.6. Ablaciones clave

La Tabla 4 justifica la decisión de diseño más sutil del paper. Se comparan cuatro combinaciones de entradas/targets al Transformer:

| Configuración | avg. WER |
|---|---|
| **Entradas continuas, targets cuantizados (Baseline)** | **7.97** |
| Entradas cuantizadas, targets cuantizados | 12.18 |
| Entradas cuantizadas, targets continuos | 11.18 |
| Entradas continuas, targets continuos | 8.58 |

La estrategia ganadora —**entradas continuas al Transformer, pero targets cuantizados en la pérdida contrastiva**— es exactamente lo contrario a lo que hacía vq-wav2vec (que cuantizaba también la entrada). Las latentes continuas **retienen más información** para construir mejor contexto, mientras que **cuantizar los targets hace el entrenamiento más robusto**. La razón profunda: si los targets fueran continuos, podrían capturar artefactos detallados de la secuencia (identidad del hablante, ruido de fondo) que **facilitan la tarea de forma tramposa** e impiden aprender representaciones generales. De hecho, la exactitud de identificar el latente correcto sube de 62% a 78% al pasar de targets cuantizados a continuos —la tarea se vuelve *más fácil pero menos útil*. Otras ablaciones (Tabla 13) confirman que el ruido Gumbel es importante, que la penalización de diversidad no puede ser ni muy baja ni muy alta, y que muestrear distractores de **otras** locuciones **empeora** (son demasiado fáciles de distinguir).

## 6. Limitaciones

- **Arquitectura acústica subóptima.** Los autores usan un Transformer con CTC y vocabulario de **caracteres**, que no calza con el vocabulario de **palabras** del LM (retrasa el *feedback* del LM y probablemente perjudica). Esperan ganancias al cambiar a arquitectura **seq2seq** y vocabulario de **word pieces**, que usan los trabajos más recientes.
- **Sin balanceo de datos ni self-training.** El resultado se logra sin técnicas de balanceo de datos; los autores señalan que el *self-training* es **complementario** al *pre-training*, y su combinación podría dar aún mejores resultados.
- **Costo de pre-training.** Aunque el *fine-tuning* es barato, el *pre-training* sobre 53k horas requiere **128 GPUs V100 por varios días** —una inversión considerable que solo grandes laboratorios pueden pagar. El ahorro está en la etiqueta humana, no en el cómputo.
- **Evaluación mayormente en inglés / dominio leído.** Los experimentos se centran en Librispeech/LibriVox (audiolibros en inglés, voz leída y limpia). La promesa multilingüe se enuncia (*Broader Impact*) pero no se demuestra a fondo en este paper.

## 7. Conexión con la Clase 37 y con Whisper

**Embeddings como representación del audio.** La Clase 37 presenta distintas formas de representar el audio: desde la forma de onda cruda y el espectrograma, pasando por features diseñadas como las **MFCC**, hasta los **embeddings preentrenados**. wav2vec 2.0 es el ejemplo canónico de esta última categoría. La idea pedagógica es directa: en lugar de calcular MFCC —un pipeline fijo de FFT, banco de filtros mel y DCT diseñado por ingenieros en los años 80—, uno pasa el audio por un modelo autosupervisado y **extrae las activaciones de la red de contexto** como vector de *features*. Esas activaciones fueron aprendidas de 53.000 horas de voz y **codifican estructura fonética y contextual** que las MFCC no capturan. El **anexo del laboratorio** ("Embeddings preentrenados: wav2vec 2.0") materializa esto: se carga un modelo preentrenado, se le pasa audio, y se obtienen embeddings listos para alimentar un clasificador liviano —el mismo patrón *pre-training + fine-tuning ligero* que revolucionó NLP con BERT, ahora en audio.

**Por qué cambió el paradigma.** Antes de wav2vec 2.0, hacer ASR en una lengua o dominio nuevo exigía **miles de horas transcritas**. Después, basta con **preentrenar una vez** sobre audio no etiquetado (abundante) y afinar con **minutos u horas** de etiqueta. Esto desacopla el costo caro (anotación humana) del recurso abundante (audio crudo), y por eso el paper dedica su *Broader Impact* a las ~7.000 lenguas sin tecnología de voz: el SSL las pone al alcance.

**Relación con Whisper.** wav2vec 2.0 es un modelo **autosupervisado** que requiere *fine-tuning* supervisado para hacer ASR. **Whisper** (Radford et al., OpenAI, 2022) representa la corriente que lo **sucede** y en cierto modo lo contrasta: en vez de SSL sobre audio sin etiquetar, Whisper apuesta por **supervisión débil masiva** —680.000 horas de audio con transcripciones (aunque ruidosas) recolectadas de la web— entrenando un seq2seq multitarea y multilingüe que funciona *zero-shot*, sin *fine-tuning*. Ambos comparten el diagnóstico (escalar los datos es la palanca) pero difieren en la fuente de señal: wav2vec 2.0 explota audio **sin** etiqueta con una pérdida contrastiva; Whisper explota audio **con** etiqueta débil pero a escala web. En la práctica actual conviven: los embeddings de wav2vec 2.0 siguen siendo el *backbone* de features para muchas tareas de voz de bajo recurso (clasificación, detección, análisis), mientras Whisper domina la transcripción *end-to-end* de propósito general.

## 8. Nota final: relevancia para salud

En el ámbito clínico, la voz es un **biomarcador** cada vez más estudiado: hay señales de voz asociadas a Parkinson, depresión, deterioro cognitivo, apnea, patologías laríngeas y estados respiratorios. El problema estructural es idéntico al que motiva a wav2vec 2.0: **los datos de voz clínicos etiquetados son escasísimos** —recolectarlos exige pacientes, consentimiento, diagnóstico experto y regulación—, mientras que grabaciones de voz sin etiquetar son comparativamente más fáciles de acumular. Usar **embeddings de audio preentrenados** como *features* permite entrenar clasificadores clínicos robustos con **muy pocos ejemplos etiquetados**, exactamente el régimen de 10 minutos / 1 hora donde este paper demostró que el *pre-training* recupera casi todo el desempeño. Para un desarrollador de sistemas de salud, la lección operativa es que **no se necesita reentrenar un modelo acústico desde cero ni transcribir miles de horas**: basta extraer embeddings wav2vec 2.0 (o afinar levemente el modelo) sobre el pequeño corpus clínico disponible, trasladando el conocimiento de 53.000 horas de voz genérica al dominio médico específico.
