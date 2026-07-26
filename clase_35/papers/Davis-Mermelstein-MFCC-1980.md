# Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences (MFCC) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences*.
- **Autores:** Steven B. Davis y Paul Mermelstein. El trabajo se realizó en **Haskins Laboratories** (New Haven, Connecticut). Al momento de la publicación, Davis se había trasladado a Signal Technology, Inc. (Santa Barbara, California) y Mermelstein a Bell-Northern Research e INRS-Telecommunications (Universidad de Quebec, Canadá).
- **Venue:** *IEEE Transactions on Acoustics, Speech, and Signal Processing (TASSP)*, vol. ASSP-28, n.º 4, agosto de 1980, pp. 357–366.
- **Financiamiento:** National Science Foundation, Grant BNS 7682023 a Haskins Laboratories.
- **Linaje:** el paper consolida una línea de investigación previa de Mermelstein sobre segmentación silábica y medidas de distancia para reconocimiento de voz, y toma prestada la evidencia perceptual de Pols (1966) sobre las bandas críticas del oído para justificar el banco de filtros en escala Mel.

Este es, sin exageración, **el paper que introdujo los coeficientes cepstrales en frecuencia Mel (MFCC)** como la representación acústica de referencia para reconocimiento de voz. Su pregunta es deliberadamente comparativa y empírica: entre las representaciones paramétricas disponibles a fines de los años setenta, ¿cuál retiene mejor la información fonéticamente significativa de la señal de voz? Para responderla, Davis y Mermelstein montaron un banco de pruebas de reconocimiento de palabras monosilábicas (todas de estructura consonante-vocal-consonante, CVC) dentro de oraciones habladas de corrido, y enfrentaron cinco representaciones bajo condiciones idénticas de vocabulario, hablante y método de alineamiento temporal.

El resultado es contundente: un conjunto de **diez coeficientes cepstrales en frecuencia Mel calculados cada 6.4 ms alcanzó 96.5 % y 95.0 % de reconocimiento** para cada uno de dos hablantes, superando a todas las demás representaciones (cepstrum de frecuencia lineal, cepstrum de predicción lineal, espectro de predicción lineal y coeficientes de reflexión). Los autores atribuyen esta superioridad a que los MFCC **representan mejor los aspectos perceptualmente relevantes del espectro de corto plazo de la voz**.

Para la **Clase 35 (Introducción al Análisis de Audio)** este paper es la fuente primaria del pipeline que enseña el profesor: audio → *frames* → FFT → banco de filtros Mel → logaritmo → DCT → coeficientes. Todo ese encadenamiento nace, en su forma canónica, aquí.

## 2. Contexto: reconocimiento de voz antes del *deep learning*

Para dimensionar la contribución hay que situarse en el estado del arte de 1980. No existían las redes neuronales profundas ni los modelos ocultos de Markov (HMM) en su forma madura; el reconocimiento de voz se apoyaba en la **comparación de patrones** (*template matching*) con alineamiento temporal por programación dinámica. El sistema típico almacenaba, para cada palabra del vocabulario, una o más plantillas de referencia (secuencias de vectores acústicos), y clasificaba una palabra desconocida buscando la plantilla de menor "distancia" tras alinear ambas en el tiempo.

En ese marco, **la representación acústica lo era casi todo**. Como escriben los autores, el objetivo al elegir una representación es doble: *comprimir* los datos de voz eliminando información no pertinente al análisis fonético, y *realzar* los aspectos de la señal que contribuyen a distinguir fonemas. La representación determina qué información sobrevive a la etapa de comparación, y por lo tanto fija el techo de desempeño del sistema completo.

Hacia fines de los setenta convivían dos grandes escuelas de análisis acústico:

- **Análisis basado en el espectro de Fourier.** Se calcula el espectro de corto plazo (típicamente vía DFT sobre ventanas), a veces suavizado con bancos de filtros. White y Neely (1976) ya habían mostrado que un enfoque de filtrado por bandas de 20 canales, usando una norma de Chebyshev sobre el logaritmo de las energías de los filtros, era competitivo.
- **Análisis por predicción lineal (LPC).** Modela la señal como la salida de un filtro todo-polos que aproxima el tracto vocal. Itakura (1975) había introducido la métrica de mínimo residuo de predicción, muy usada para comparar tramas.

White y Neely concluyeron que el filtrado por bandas y la predicción lineal eran "esencialmente equivalentes" cuando se combinaban con alineamiento temporal por programación dinámica. Davis y Mermelstein sospechaban que esa equivalencia se debía a que el vocabulario de prueba **carecía de palabras fonéticamente similares**, es decir, no ponía a prueba la capacidad discriminativa fina de cada representación. Su experimento se diseñó precisamente para forzar esa distinción: un vocabulario cargado de palabras confundibles.

El ingrediente perceptual venía de la psicoacústica. Se conocía desde hacía décadas que **el ancho de las bandas críticas del oído varía con la frecuencia**: el sistema auditivo tiene resolución fina en frecuencias bajas y gruesa en frecuencias altas (Feldtkeller y Zwicker, 1956; Schroeder, 1977). Esto motivaba usar filtros espaciados linealmente en bajas frecuencias y logarítmicamente en altas. Pols (1966), en un antecedente decisivo, mostró que los primeros seis autovectores de la matriz de covarianza de vocales holandesas —expresadas en términos de 17 energías de filtros de este tipo— explicaban el **91.8 %** de la varianza total, y que las direcciones de esos autovectores se parecían mucho a una **expansión en serie de cosenos** sobre las energías de los filtros. Ese hallazgo —una transformada de cosenos aplicada a energías de filtros perceptualmente espaciados— es, en germen, el MFCC.

## 3. Contribución central: los MFCC

La contribución del paper es la **definición operativa de los coeficientes cepstrales en frecuencia Mel** y su validación empírica como la mejor representación entre las evaluadas. En palabras de los autores, los MFCC son "el resultado de una transformada de cosenos del logaritmo real del espectro de energía de corto plazo expresado en una escala de frecuencia Mel". Esa frase contiene los tres ingredientes que hoy reconocemos como el corazón del pipeline:

1. **Escala de frecuencia Mel:** el eje de frecuencias se deforma para imitar la resolución no uniforme del oído.
2. **Logaritmo de las energías:** se toma el logaritmo de la energía de cada banda, comprimiendo el rango dinámico de forma análoga a la percepción de sonoridad.
3. **Transformada de cosenos (DCT):** se aplica una transformada de cosenos sobre las log-energías para obtener un conjunto compacto y aproximadamente decorrelacionado de coeficientes —el *cepstrum*.

Antes del experimento principal, un experimento preliminar ya insinuaba el poder del método: cuatro hablantes produjeron 12 palabras fonéticamente similares (*stick, sick, skit, spit, sit, slit, strip, scrip, skip, skid, spick, slid*), y una representación que usaba **solo dos coeficientes cepstrales alcanzó 96 % de reconocimiento correcto**. Ese resultado motivó verificar el poder de la representación comparándola sistemáticamente con las alternativas manteniendo todo lo demás constante.

## 4. Método

### 4.1. Las cinco representaciones comparadas

Las representaciones se dividen en dos grupos según su origen espectral:

**Grupo 1 — basadas en el espectro de Fourier:**
- **MFCC** (mel-frequency cepstrum coefficients): la propuesta del paper.
- **LFCC** (linear-frequency cepstrum coefficients): el mismo tipo de cepstrum, pero calculado directamente sobre el espectro con eje de frecuencia lineal, sin la deformación Mel. Es el control clave para aislar el efecto de la escala Mel.

**Grupo 2 — basadas en el espectro de predicción lineal (LPC):**
- **LPC** (linear prediction coefficients): aproximación todo-polos de orden 10 al espectro de la forma de onda enventanada, obtenida por el método de autocorrelación.
- **LPCC** (linear prediction cepstrum coefficients): coeficientes cepstrales derivados recursivamente de los LPC.
- **RC** (reflection coefficients): obtenidos de los LPC mediante la transformación equivalente a modelar el tracto vocal como un tubo acústico de diez secciones de área variable; cada coeficiente indica la fracción de energía reflejada en cada frontera de sección.

Cada representación se emparejó con una métrica de distancia apropiada. Para todos los parámetros cepstrales (MFCC, LFCC, LPCC) se usó **distancia euclidiana**, justificada porque los coeficientes cepstrales derivan de una base ortogonal. La misma distancia euclidiana se usó para los RC, a falta de una métrica asociada inherente. Los LPC, en cambio, se evaluaron con la **métrica de mínimo residuo de predicción de Itakura**.

### 4.2. La escala Mel

La escala Mel mapea la frecuencia física (en Hz) a una escala perceptual (en *mels*) que crece de forma no lineal, comprimiendo las frecuencias altas. La forma más difundida de la conversión es

$$m = 2595\,\log_{10}\!\left(1 + \frac{f}{700}\right),$$

donde $f$ es la frecuencia en Hz y $m$ la frecuencia percibida en mels. La propiedad crítica es la **densidad no uniforme de filtros**: en frecuencias bajas los filtros son estrechos y están muy juntos (resolución fina), mientras que en frecuencias altas se ensanchan y espacian (resolución gruesa). Esto reproduce el comportamiento del oído, cuyas bandas críticas se ensanchan con la frecuencia, y explica por qué la escala favorece la información espectral de baja frecuencia, donde residen los primeros formantes que distinguen las vocales.

Conviene una precisión histórica: **el paper de 1980 no escribe literalmente la fórmula logarítmica de 2595**. En una nota al pie, los autores adoptan una aproximación operacional citando a Fant (1959): la escala Mel debe entenderse como un **espaciamiento lineal de frecuencias por debajo de 1000 Hz y logarítmico por encima de 1000 Hz**. Señalan que las diferencias entre las distintas versiones de la escala (Beranek, Koenig, la aproximación de Fant) no son significativas para sus fines. La fórmula cerrada de 2595, popularizada más tarde, es la formalización estándar de esa misma idea: lineal abajo, logarítmica arriba.

### 4.3. El banco de filtros triangulares

Para el cálculo de los MFCC se simularon **20 filtros pasa-banda triangulares** (Fig. 1 del paper) que cubren el rango de frecuencias hasta aproximadamente 4600 Hz. Cada filtro triangular pondera las componentes espectrales de su banda: la ganancia es máxima en la frecuencia central y decae linealmente hasta cero en las frecuencias centrales de los filtros vecinos. Las frecuencias centrales están espaciadas según la escala Mel —juntas abajo, separadas arriba— de modo que el conjunto de 20 filtros muestrea el espectro con la densidad perceptual deseada.

La salida de cada filtro es la **energía** de la señal en esa banda (la suma ponderada de las magnitudes espectrales bajo el triángulo). A cada una de esas 20 energías se le aplica el **logaritmo**, obteniendo lo que los autores denotan $x_k$: la log-energía de salida del $k$-ésimo filtro, $k = 1, 2, \dots, 20$.

### 4.4. La DCT y la definición formal del cepstrum Mel

El paso final convierte las 20 log-energías en un conjunto reducido de coeficientes cepstrales mediante una **transformada de cosenos discreta (DCT)**. La ecuación (1) del paper define los MFCC como

$$c_n = \sum_{k=1}^{20} x_k \,\cos\!\left[\,n\left(k - \tfrac{1}{2}\right)\frac{\pi}{20}\,\right], \qquad n = 1, 2, \dots, M,$$

donde $M$ es el número de coeficientes cepstrales que se conservan (en el experimento principal, $M = 10$ o $M = 6$), y $x_k$ es la log-energía del $k$-ésimo filtro. La DCT cumple tres funciones simultáneas que la hacen ideal para este paso:

1. **Compresión / reducción de dimensión.** De 20 log-energías se pasa a 10 (o 6) coeficientes que concentran la información relevante. El experimento muestra que **seis coeficientes bastan** para capturar la mayor parte de la información fonética.
2. **Decorrelación.** Las energías de bandas vecinas están fuertemente correlacionadas (por el solapamiento de los filtros y la continuidad espectral). La DCT aproxima la transformación de Karhunen-Loève para este tipo de señales —de ahí la conexión con los autovectores de Pols— y produce coeficientes casi independientes. Esto justifica el uso de la distancia euclidiana simple: al estar decorrelacionados, tratar cada coeficiente como una dimensión ortogonal es apropiado.
3. **Separación fuente-filtro (interpretación cepstral).** Tomar el logaritmo del espectro y luego una transformada convierte la convolución (excitación glótica ⊗ respuesta del tracto vocal) en una suma separable. Los coeficientes cepstrales bajos capturan la envolvente espectral suave (el tracto vocal, que codifica el fonema) y los altos, el detalle fino (el pitch). Conservar solo los primeros equivale a un **suavizado del espectro logarítmico** que retiene la envolvente formántica y descarta la excitación.

El **coeficiente cero** ($c_0$, la energía media del espectro) merece una nota: los autores lo excluyen deliberadamente de las comparaciones, lo que equivale a **igualar la energía global** entre tramas alineadas temporalmente. Observan que incluir la variación de energía global con el tiempo *podría* ayudar a discriminar pares muy confundibles, pero en su configuración prefirieron normalizarla.

Para el LFCC, la ecuación (2) aplica una transformada de cosenos análoga, pero directamente sobre el logaritmo de la magnitud de la DFT (con eje de frecuencia lineal), sobre $K$ coeficientes de magnitud de la DFT. La única diferencia conceptual entre MFCC y LFCC es, entonces, **la deformación del eje de frecuencias a la escala Mel** — lo que hace de su comparación el experimento decisivo sobre el valor de la escala perceptual.

### 4.5. El pipeline completo de señal

Los parámetros de procesamiento fijados en el paper son los que hoy siguen siendo típicos:

- La señal se **filtró pasa-bajos a 5 kHz y se muestreó a 10 kHz**.
- Los espectros (de Fourier o de predicción lineal) se calcularon para tramas sucesivas separadas **64 puntos (6.4 ms) o 128 puntos (12.8 ms)**.
- En cada trama se aplicó una **ventana de Hamming de 256 puntos (25.6 ms)** para seleccionar los datos a analizar. Reducir la ventana a 128 puntos degradó los resultados.

El encadenamiento completo para MFCC es, por lo tanto: **tomar una trama enventanada → calcular su espectro (FFT) → agrupar la energía en 20 bandas mediante los filtros triangulares en escala Mel → tomar el logaritmo de cada energía de banda → aplicar la DCT → conservar los primeros $M$ coeficientes**. Este es exactamente el pipeline de la Clase 35.

### 4.6. Alineamiento temporal y generación de plantillas

Aunque no es el foco del análisis, conviene registrar el andamiaje experimental. Como el sistema es de comparación de patrones, hace falta alinear en el tiempo la palabra de prueba y cada plantilla. Los autores usaron un **algoritmo de programación dinámica (dynamic time warping) simétrico**, en la línea de Velichko-Zagoruyko, Bridle-Brown, Itakura y Sakoe-Chiba, con una **función de penalización** $V$ (fijada en 1.5 para desviaciones de la diagonal y 1.0 en otro caso) que mantenía el camino de alineamiento cerca de la diagonal. Las plantillas se generaron de forma incremental, deformando y promediando tokens sucesivos.

## 5. Experimentos y resultados

### 5.1. Los datos

El corpus se diseñó para maximizar la confusabilidad fonética. Comprendió **52 palabras CVC distintas**, producidas por **dos hablantes masculinos (DZ y LL)**, con un total de **169 tokens extraídos de 57 oraciones distintas** (listadas en el Apéndice A del paper, del tipo "Keep the hope at the bar" y "Bar the keep for the yell"). Las oraciones se leyeron dos veces por cada hablante en sesiones separadas por **dos meses** (denotadas DZ1, DZ2, LL1, LL2), sumando **676 sílabas**. Las palabras podían funcionar como sustantivos o verbos para forzar variación sintáctica preservando la entonación. El vocabulario incluyó **12 vocales distintas**, cada una representada en al menos cuatro palabras con consonantes prevocálicas y postvocálicas diferentes. La segmentación de las sílabas fue **manual**, con evaluación auditiva, para evitar que errores de segmentación automática se confundieran con deficiencias de las representaciones. Las grabaciones se hicieron con micrófono de alta calidad en ambiente silencioso, para establecer el mejor desempeño alcanzable.

Se emplearon dos modalidades de prueba: **cerrada** (*closed*, datos de prueba y referencia de la misma sesión, p. ej. referencia DZ1 vs. prueba DZ1) y **abierta** (*open*, de sesiones distintas, p. ej. referencia DZ1 vs. prueba DZ2). La prueba abierta es la más exigente y realista. Para cada palabra de prueba se hizo una deformación temporal contra cada una de las 52 plantillas, y se identificó con la de menor distancia.

### 5.2. La tabla de resultados

La Tabla I del paper reporta las tasas de reconocimiento. Para las **pruebas abiertas con 10 coeficientes y tramas de 6.4 ms** (la configuración de referencia mostrada en la Fig. 8), el orden de desempeño fue inequívoco:

| Representación | Coefs. | Métrica | Abierta DZ | Abierta LL |
|---|---|---|---|---|
| **Cepstrum Mel (MFCC)** | 10 | Euclidiana | **96.5 %** | **95.0 %** |
| Cepstrum frec. lineal (LFCC) | 10 | Euclidiana | 96.5 % | 92.0 % |
| Cepstrum predicción lineal (LPCC) | 10 | Euclidiana | 94.7 % | 87.6 % |
| Espectro predicción lineal (LPC) | 10 | Itakura | 85.2 % | 84.3 % |
| Coeficientes de reflexión (RC) | 10 | Euclidiana | 83.1 % | 77.5 % |

Los hallazgos clave que se desprenden:

1. **Los MFCC ganan de forma consistente**, independientemente de la separación de tramas, el tipo de prueba (abierta o cerrada) o el hablante.
2. **Seis MFCC superan a cualquier otro conjunto de diez coeficientes.** Esto es notable: la representación Mel con menos dimensiones vence a las alternativas con más. En pruebas cerradas, seis MFCC alcanzaron incluso 99.4 % para DZ.
3. **La escala Mel importa por sobre el cepstrum de frecuencia lineal.** MFCC vence a LFCC, sobre todo para el hablante LL. La ventaja específica del MFCC, según los autores, es que **suprime mejor la variación espectral insignificante en las bandas de alta frecuencia** — justamente lo que hace la resolución gruesa de la escala Mel arriba.
4. **Los parámetros de Fourier (MFCC, LFCC) superan a los de predicción lineal (LPCC, LPC, RC).** Ambos grupos son adecuados para vocales, pero las confusiones ocurren en las **consonantes**, cuyos espectros la predicción lineal representa de forma inexacta.
5. **Los parámetros cepstrales (MFCC, LFCC, LPCC) superan a LPC y RC.** Los cepstra corresponden a representaciones suavizadas en frecuencia del espectro log-magnitud, y una distancia euclidiana sobre ellos separa mejor los espectros fonéticamente distintos.
6. **La métrica de Itakura es menos efectiva que la distancia cepstral** para indicar la significancia fonética de la diferencia entre dos espectros, aun cuando el punto de optimalidad sea el mismo.
7. **Tramas de 6.4 ms superan a las de 12.8 ms** en todos los casos, con una ganancia media de 1.7 %; pero el costo computacional de la programación dinámica crece con el cuadrado del número de tramas, mientras que crece solo linealmente con el número de coeficientes. De ahí la recomendación práctica: **más coeficientes con resolución temporal algo más gruesa es computacionalmente más ventajoso** que menos coeficientes más frecuentes.

Los errores restantes se concentraron en pares fonéticamente muy similares. De las ocho equivocaciones con MFCC para el hablante DZ, dos fueron entre *bar* y *mar*, dos entre *pool* y *fool*, y una cada una entre *keep*/*heat*, *bait*/*wake*, *hook*/*rig* y *hood*/*cause* — precisamente el tipo de confusiones de alta dificultad que el vocabulario buscaba provocar.

## 6. Por qué los MFCC dominaron tres décadas

La superioridad empírica de los MFCC en este paper explica solo en parte su longevidad. Su reinado —desde 1980 hasta la irrupción del *deep learning* end-to-end hacia 2012-2015— se debe a una combinación de virtudes:

- **Compacidad.** Seis a trece coeficientes resumen una trama de voz. En una era de memoria y cómputo escasos, esto era decisivo, y los propios autores lo destacan: los MFCC "forman una representación particularmente compacta".
- **Decorrelación.** La DCT produce coeficientes aproximadamente independientes, lo que permitió modelarlos con **mezclas de gaussianas de covarianza diagonal** en los HMM que dominaron el reconocimiento de voz en los años noventa y dos mil. Esta compatibilidad con los HMM-GMM fue quizá el factor pragmático más importante de su supervivencia.
- **Fundamento perceptual.** Al codificar la resolución no uniforme del oído, descartan variación espectral irrelevante para la percepción humana del habla, mejorando la generalización.
- **Simplicidad y velocidad.** El pipeline es un encadenamiento de operaciones lineales y logaritmos, barato de calcular y fácil de implementar.

Así, los MFCC se volvieron el *front-end* acústico por defecto no solo en reconocimiento de voz, sino en identificación de hablante, reconocimiento de emociones, clasificación de música y detección de eventos sonoros. Incluso hoy, muchos sistemas de *deep learning* que operan sobre **espectrogramas Mel** conservan las primeras etapas de este pipeline (filtros Mel + log) y omiten solo la DCT final, porque las redes convolucionales aprenden sus propias combinaciones de las bandas y ya no necesitan la decorrelación explícita de la DCT.

## 7. Limitaciones

Los propios autores son cautos respecto del alcance de sus conclusiones:

- **Vocabulario y condiciones restringidas.** Los resultados se limitan a palabras monosilábicas CVC acentuadas. No se estudiaron *clusters* consonánticos, palabras multisilábicas ni monosílabos átonos, que son fonéticamente más elásticos. No es obvio que la mejor representación para palabras acentuadas lo sea también para las átonas.
- **Dependencia del hablante.** El estudio es explícitamente *speaker-dependent*: referencia y prueba provienen del mismo hablante. La variación entre hablantes se dejó fuera como problema separado.
- **Poca robustez al ruido.** Las grabaciones se hicieron en ambiente silencioso con micrófono de alta calidad para fijar el mejor desempeño posible. Los autores advierten que ambientes con más ruido ambiente "sin duda deteriorarían la claridad de la información acústica y, por lo tanto, resultarían en menor desempeño". Esta fragilidad ante el ruido y la distorsión de canal es una **limitación estructural del MFCC** que la investigación posterior intentó paliar con técnicas de normalización (CMN, RASTA), y que sigue siendo una de sus debilidades más citadas.
- **Se descarta información temporal fina y global.** Al normalizar el coeficiente cero (energía global) y al comprimir cada trama a pocos coeficientes, el MFCC descarta información temporal que podría ayudar en pares confundibles. Los autores mismos notan que la variación de la energía global con el tiempo podría asistir la discriminación. La solución posterior fueron los **coeficientes delta y delta-delta** (derivadas temporales), que se agregaron a los MFCC para capturar la dinámica que una trama aislada no representa.
- **Dependencia de la métrica de distancia.** El ranking comparativo puede verse influido por la elección de las métricas locales e integradas. La distancia euclidiana es de las más simples, pero incorporar las distribuciones de probabilidad de los parámetros (p. ej. la varianza de los coeficientes cepstrales) debería mejorar el desempeño — una intuición que anticipa el modelado estadístico de los HMM-GMM.

## 8. Conexión con la Clase 35 (Introducción al Análisis de Audio)

La última sección de la Clase 35 desarrolla los MFCC, y este paper es su fuente canónica. El pipeline que enseña el profesor Sepúlveda —**audio → frames → FFT → banco de filtros Mel → log → DCT → coeficientes**— es, paso por paso, el procedimiento definido por Davis y Mermelstein:

1. **Audio → frames.** La señal continua se corta en tramas cortas y solapadas, enventanadas con Hamming, sobre las cuales el espectro puede considerarse estacionario. En el paper: ventana de Hamming de 25.6 ms, avance de 6.4 ms.
2. **FFT.** Cada trama se lleva al dominio de la frecuencia con la transformada de Fourier, obteniendo el espectro de magnitud de corto plazo.
3. **Banco de filtros Mel.** Los 20 filtros triangulares espaciados en escala Mel agrupan la energía espectral en bandas perceptuales — finas abajo, gruesas arriba.
4. **Logaritmo.** Se toma el logaritmo de la energía de cada banda, comprimiendo el rango dinámico como lo hace la percepción de sonoridad.
5. **DCT.** La transformada de cosenos decorrelaciona y comprime las 20 log-energías en un puñado de coeficientes.
6. **Coeficientes.** Se conservan los primeros $M$ (típicamente 6 a 13); descartar el resto suaviza el espectro y retiene la envolvente formántica.

Los tres conceptos que el estudiante de la clase debe internalizar de este paper:

- **La escala Mel modela la percepción.** Resolución fina en bajas frecuencias, gruesa en altas, imitando las bandas críticas del oído. La comparación MFCC vs. LFCC en la Tabla I es la prueba experimental de que esta deformación perceptual mejora el reconocimiento.
- **El logaritmo y la DCT no son adornos.** El logaritmo lineariza la percepción de intensidad y convierte la convolución fuente-filtro en suma; la DCT decorrelaciona y comprime, habilitando distancias euclidianas simples. Juntos separan la envolvente del tracto vocal (el fonema) de la excitación (el pitch).
- **Menos es más.** Seis MFCC superaron a diez coeficientes de cualquier otra representación. La compacidad es una virtud, no un compromiso.

**Enlaces internos:**

- Clase: [/clases/clase-35](/clases/clase-35) — Introducción al Análisis de Audio.
- Fundamento transversal (procesamiento de señal / espectro): la sección de la clase sobre frames, FFT y bancos de filtros.

---

**Nota final — relevancia para salud.** Aunque nacieron para reconocimiento de palabras, los MFCC se han convertido en un caballo de batalla del **análisis de voz clínica y de biomarcadores acústicos**. En la evaluación de patologías vocales (parálisis de cuerdas vocales, nódulos, disfonías, cáncer laríngeo) los MFCC capturan alteraciones sutiles de la envolvente espectral que el oído humano detecta pero cuesta cuantificar. En neurología se estudian como **biomarcadores de voz** para la enfermedad de Parkinson, el deterioro cognitivo y la depresión, donde cambios en la articulación y la fonación modifican el perfil cepstral. Y más allá de la voz, el mismo pipeline —frames, filtros perceptuales, log, DCT— se aplica al análisis de **sonidos respiratorios** (sibilancias, crepitaciones en auscultación pulmonar, cribado de apnea y de COVID a partir de la tos) y de **sonidos cardíacos** (clasificación de soplos en fonocardiografía). Que una representación diseñada en 1980 para distinguir *pool* de *fool* siga siendo, más de cuatro décadas después, la base de sistemas de tamizaje médico de bajo costo a partir de un micrófono, es el mejor testimonio de la solidez perceptual del diseño de Davis y Mermelstein.
