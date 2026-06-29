# Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights*.
- **Autores:** Arun Mallya, Dillon Davis, Svetlana Lazebnik (University of Illinois at Urbana-Champaign).
- **Venue:** ECCV 2018 (European Conference on Computer Vision).
- **Año:** 2018. **Preprint:** arXiv:1801.06519v2 (16 mar 2018), [arxiv.org/abs/1801.06519](https://arxiv.org/abs/1801.06519).
- **Código:** [github.com/arunmallya/piggyback](https://github.com/arunmallya/piggyback). Implementación en PyTorch.
- **Keywords del paper:** Incremental Learning, Binary Networks.

La tesis del paper se enuncia en una sola pregunta provocadora de la introducción: *¿es realmente necesario cambiar los pesos de una red para aprender una tarea nueva, o basta con enmascarar selectivamente algunos pesos —ponerlos a cero— mientras se dejan intactos los demás?* Piggyback responde que basta enmascarar. El método toma una red **pre-entrenada y congelada** (la *backbone*, por ejemplo una VGG-16 o ResNet-50 entrenada en ImageNet) y, para cada tarea nueva, aprende una **máscara binaria por peso** —un valor en {0, 1} para cada parámetro— que decide qué pesos del backbone se usan y cuáles se apagan. Los pesos base nunca se modifican.

La consecuencia directa, y el argumento central contra el olvido catastrófico, es estructural: **como los pesos base no cambian, no hay olvido posible**. El rendimiento en la tarea original (ImageNet) y en cualquier tarea previa queda idéntico por construcción, sin proxy losses ni regularizadores. Y como cada tarea tiene su propia máscara independiente, el método es **agnóstico al orden de las tareas** y permite añadir un número en principio ilimitado de tareas, cada una "haciendo piggyback" (montándose a caballito) sobre el mismo backbone.

El costo de almacenamiento es la otra mitad del atractivo. Un parámetro típico de red neuronal ocupa 32 bits (un float); una máscara binaria solo añade **1 bit por parámetro y por tarea**, lo que da un overhead aproximado de 1/32 ≈ 3.12 % del tamaño del backbone por cada tarea nueva. Es decir, almacenar una tarea adicional cuesta entre 32× y 64× menos que guardar una red completa fine-tuneada (32× sobre un backbone full-precision, más si se considera que las redes individuales también guardan biases y BN). El truco técnico que lo hace entrenable: se aprenden **máscaras de valores reales** que se **binarizan con un umbral** en el forward, y el gradiente fluye hacia atrás mediante un estimador tipo *straight-through* (el gradiente de la máscara binarizada sirve como estimador ruidoso del gradiente de la máscara real).

Para la Clase 32 (Olvido Catastrófico) esto importa porque Piggyback es el ejemplo arquetípico de los **métodos basados en arquitectura** contra el olvido: en lugar de proteger pesos importantes (regularización) o reentrenar con datos viejos (replay), se aísla cada tarea en su propia estructura de parámetros. La clase lo presenta exactamente así en su slide "PiggyBack": máscaras binarias sobre una red preentrenada, con un ahorro de 32-64× en espacio frente a guardar modelos completos.

## 2. Contexto histórico: aprendizaje incremental y el problema del olvido catastrófico (2016–2018)

El método estándar para adaptar una red a una tarea nueva es el **fine-tuning**: partir de un modelo pre-entrenado (la VGG-16 sobre ImageNet es el ejemplo canónico del paper) y reentrenar sus pesos. Pero el fine-tuning tiene un defecto fatal en el escenario incremental: el **olvido catastrófico** (French, 1999), por el cual el rendimiento en la tarea vieja se degrada severamente a medida que se aprende la nueva, obligando a guardar un modelo especializado por tarea. El objetivo del aprendizaje continuo es justamente aumentar las capacidades de una red existente *evitando* el olvido y *minimizando* los parámetros adicionales.

El paper organiza el trabajo previo en familias y se posiciona contra cada una:

- **Métodos de regularización / proxy loss.** *Learning without Forgetting* (LwF, Li & Hoiem, 2016) usa las respuestas iniciales de la red sobre los datos nuevos como objetivos de regularización durante el entrenamiento de la tarea nueva. *Elastic Weight Consolidation* (EWC, Kirkpatrick et al., 2017) impone una penalización suave al cambio de los pesos considerados importantes para tareas previas. El problema que el paper les achaca: como **todos los pesos pueden modificarse** en algún grado, no es posible determinar de antemano cuánto se degradará el rendimiento en tareas previas. Para LwF, además, un *domain shift* grande en la tarea nueva provoca caídas significativas en la tarea vieja.

- **Métodos de pruning + fine-tuning.** *PackNet* (Mallya & Lazebnik, 2017) —del mismo primer autor— es el antecedente directo. PackNet añade tareas iterativamente: poda los pesos poco importantes, los reentrena para la tarea nueva, y mantiene fijos los pesos importantes de tareas previas. Como subproducto produce una máscara binaria de uso de parámetros (qué tarea usa qué peso). Limitaciones que Piggyback supera: PackNet es **sensible al orden** (el rendimiento de una tarea cae si se añade más tarde, por falta de parámetros libres), y el número total de tareas está **limitado por el tamaño fijo de la red**.

- **Métodos que añaden parámetros task-specific.** *Progressive Neural Networks* (Rusu et al., 2016) duplican la arquitectura base por cada tarea, añadiendo conexiones laterales — overhead enorme. *Residual Adapters* (Rebuffi et al., 2017) añaden una capa convolucional por tarea. *Deep Adaptation Networks* (DAN, Rosenfeld & Tsotsos, 2017) aprenden filtros nuevos como combinaciones lineales de los existentes. Piggyback se parece a estos en que aprende filtros nuevos por tarea, pero con una restricción más fuerte: los filtros nuevos son **versiones enmascaradas** de los existentes, con un overhead de solo 1 bit por parámetro —menor que todo el trabajo previo.

- **Routing selectivo.** *PathNet* (Fernando et al., 2017) aprende rutas selectivas a través de las neuronas usando estrategias evolutivas. Piggyback logra un comportamiento similar (encender/apagar partes de la red por tarea) pero con un método **end-to-end diferenciable**, mucho menos costoso computacionalmente que la búsqueda evolutiva.

La base técnica de Piggyback viene de un área distinta: la **binarización de redes** de Courbariaux et al. (BinaryConnect, 2015; Binarized Neural Networks, 2016). Allí se mantienen pesos de valor real que se pasan por un binarizador en el forward, y los gradientes calculados sobre los pesos binarizados se usan para actualizar los reales; los autores argumentan que estos gradientes, aunque ruidosos, funcionan como regularizador y los errores de cuantización se cancelan a lo largo de las iteraciones. La diferencia clave de Piggyback: **no entrena una red cuantizada desde cero**, sino que aprende **máscaras** cuantizadas que se aplican a pesos de valor real **fijos**. Combina así dos ideas —binarización de redes y matrices de pesos enmascaradas (Guo et al., *Dynamic Network Surgery*, 2016)— en un método nuevo.

## 3. Contribución central

La contribución de Piggyback es un mecanismo para **adaptar una red única y fija a múltiples tareas aprendiendo máscaras binarias por peso**, con tres propiedades que ninguna combinación de trabajo previo lograba simultáneamente:

1. **Cero olvido por construcción.** Los pesos del backbone son tratados como constantes fijas. No se tocan jamás. Por tanto el rendimiento en cualquier tarea previa (incluida ImageNet) es exactamente el de la red original. No hay competencia entre tareas ni degradación progresiva.

2. **Overhead mínimo y agnosticismo al orden.** Cada tarea añade una máscara binaria de 1 bit por parámetro (~3.12 % del backbone). Como las máscaras son independientes entre sí, el orden de adición de tareas no afecta a ninguna, y se pueden añadir tareas sin límite teórico (limitado solo por el almacenamiento de máscaras, no por la capacidad de la red).

3. **Expresividad sorprendente de una máscara binaria.** Aunque los pesos están fijos, enmascarar individualmente permite materializar una enorme variedad de filtros. El ejemplo del paper: un vector de pesos denso `[0.1, 0.9, −0.5, 1]` puede dar lugar, vía máscaras binarias, a filtros como `[0.1, 0, 0, 1]`, `[0, 0.9, −0.5, 0]` o `[0, 0.9, −0.5, 1]`. El espacio combinatorio de subconjuntos de pesos es lo bastante rico como para igualar —y a veces superar ligeramente— a una red fine-tuneada completa.

El resultado empírico que cierra la contribución: en clasificación de imágenes sobre seis datasets de dificultad y dominio variados, Piggyback obtiene rendimiento **comparable o levemente superior** al de entrenar una red dedicada por tarea (el "techo" práctico), usando una fracción del almacenamiento.

## 4. Método: de máscara real a máscara binaria, y el backprop estilo straight-through

El núcleo técnico es cómo hacer que el aprendizaje de una máscara binaria sea diferenciable. La idea (Figura 1 del paper): se mantiene un conjunto de **pesos de máscara de valor real** $m_r$, se los pasa por una **función de umbral determinista** para obtener la **máscara binaria** $m$, y esta se aplica elementwise a los pesos del backbone. Backpropagando la pérdida específica de la tarea se actualizan los pesos de máscara reales.

**Formulación (capa fully-connected, extensible a convolucional).** Sea $W$ la matriz de pesos del backbone (fija). Se asocia a $W$ una matriz de máscara real $m_r$ del mismo tamaño. La máscara binaria se obtiene por umbral duro:

$$
m_{ji} = \begin{cases} 1, & \text{si } (m_r)_{ji} \ge \tau \\ 0, & \text{en otro caso} \end{cases}
$$

con $\tau$ un umbral fijo. La relación entrada-salida de la capa enmascarada es $y = (W \odot m)\,x$, donde $\odot$ es el producto elementwise (masking). El bit $m_{ji}$ enciende o apaga el peso $w_{ji}$.

**El paso clave (straight-through).** La función de umbral duro es **no diferenciable** (su gradiente es cero casi en todas partes). Piggyback usa el truco de Courbariaux et al.: en el backward, los gradientes calculados respecto a la máscara *binarizada* $m$ se usan como **estimador ruidoso** de los gradientes de la máscara *real* $m_r$. Es decir, se ignora el umbral en el backward y se deja pasar el gradiente directamente (de ahí el nombre *straight-through estimator*). La ecuación de backprop resultante para la máscara es:

$$
\delta m = (\delta y \cdot x^T) \odot W
$$

Solo se actualiza $m_r$; $W$ permanece constante. Tras entrenar una tarea, los pesos reales $m_r$ se **descartan** y solo se guarda la máscara binarizada $m$ por capa — de ahí el 1 bit por parámetro.

**Detalles de optimización que el paper documenta como importantes:**

- De la ecuación de backprop se observa que $|\delta m|, |\delta m_r| \propto |W|$: la magnitud del gradiente de la máscara escala con la magnitud de los pesos del backbone, que varía por capa. Esto exige cuidado en inicialización y learning rate. La mejor combinación encontrada: **inicializar $m_r$ con una constante** (1e-2) y usar el **optimizador Adam** (que aprende factores de escala apropiados por sí mismo), con umbral $\tau = $ 5e-3. Esto dio una ganancia consistente de ~2 % en accuracy frente a las alternativas.
- Inicializar las máscaras de modo que tengan igual número de 0s y 1s da mal resultado; conviene inicializar de modo que **todas las máscaras valgan 1** al principio (lo que reproduce exactamente la inicialización del backbone base).
- **No** se entrenan biases ni parámetros de batch-norm específicos por tarea en el setup por defecto (PackNet mostró que los biases no impactan; los BN se omiten por simplicidad). La Sección 5.3 muestra que sí ayuda entrenar BN por tarea cuando hay domain shift grande.
- Probaron máscaras **ternarias** {−1, 0, 1} con dos umbrales, pero no obtuvieron resultados significativamente distintos de las binarias, así que se quedaron con binarias.

## 5. Experimentos y resultados

**Datasets.** Seis datasets de clasificación de dificultad y dominio variados: dos de gran escala (ImageNet, Places365, >1M imágenes cada uno), tres fine-grained (CUBS aves, Stanford Cars, Oxford Flowers) y dos con **domain shift fuerte** respecto a imágenes naturales (WikiArt, pinturas de distintos estilos; Sketch, dibujos en blanco y negro hechos por humanos). Todas las redes con input 224×224.

**Baselines comparados:** *Classifier Only* (solo entrena el clasificador lineal sobre features fc7 del backbone congelado — bajo overhead, bajo rendimiento), *Individual Networks* (una red fine-tuneada completa por tarea — el techo de rendimiento y el mayor costo), y *PackNet*.

**Resultado principal (VGG-16, Tabla 2).** Piggyback obtiene errores ligeramente **por debajo** (mejores que) los de las redes individuales en varias tareas — por ejemplo CUBS 20.99 % vs 21.30 %, Stanford Cars 11.87 % vs 12.49 %, Sketch 22.70 % vs 23.54 %. Los autores atribuyen esta mejora marginal al **efecto regularizador** de restringir la modificación de filtros al enmascarado. PackNet mejora mucho sobre Classifier Only pero **empieza a sufrir al pasar de 3 tareas** y es sensible al orden (sus errores suben 4-7 % al añadir una tarea de primera a última). El error en ImageNet de PackNet sube por el pruning inicial; el de Piggyback queda **idéntico al original** (cero olvido). En almacenamiento: 6 redes individuales = 3.222 MB vs 1 backbone Piggyback con todas las máscaras = 621 MB.

**Escalando a una tarea grande (Tabla 3).** Añadir Places365 (1.8M imágenes) al backbone ImageNet: Piggyback queda dentro de 0.36 % del top-1 de la red individual, pese a que los baselines se entrenaron 60-90 épocas contra 20 de Piggyback.

**Otras arquitecturas (Tabla 4).** El método funciona sin cambios sobre VGG-16 con BatchNorm, ResNet-50 y DenseNet-121 (con BN, conexiones residuales y skip connections). Hallazgo: **cuanto más profunda la red, mayor la brecha** Piggyback vs individual. En VGG-16 Piggyback iguala o supera; en ResNet/DenseNet la brecha es ~2 %, y hasta 4-5 % en WikiArt (domain shift). La causa: enmascarar filtros cambia la magnitud media de las activaciones, lo que choca con los parámetros de BN heredados de ImageNet.

**Análisis (Sección 5):**
- *Inicialización importa (Tabla 5).* Con ResNet-50, el backbone inicializado en ImageNet supera a uno inicializado en Places365 y a uno aleatorio en todas las tareas. Dato notable: incluso un backbone **aleatorio** congelado obtiene accuracies no triviales con solo aprender máscaras — prueba de la expresividad del enmascarado.
- *Sparsity aprendida (Tabla 6, Figura 3).* El porcentaje de pesos apagados crece con la dificultad/domain shift de la tarea: Flowers (fácil) ~4.5 %, WikiArt (difícil) ~34 % en VGG-16. Patrón consistente por capas: las **capas bajas (conv1-3) se reutilizan casi intactas** (features genéricas), y los cambios se concentran en capas medias/altas (task-specific). WikiArt, por su domain shift, sí requiere cambios en capas bajas.
- *Domain shift y BN (Tabla 7).* Entrenar parámetros de BatchNorm específicos por tarea reduce el error de WikiArt de 28.67 % a 25.92 % en ResNet-50, con un costo extra de solo ~1 MB.

**Visual Decathlon y segmentación (Sección 6).** En el reto Visual Decathlon (10 tareas), Piggyback alcanza score 2838, competitivo con el estado del arte (DAN 2851), usando el **menor número de parámetros adicionales** (ratio 1.28× = (32n + 9n)/32n para 9 máscaras). También se extiende a segmentación semántica (FCN sobre PASCAL): mean IOU 61.41 vs 61.08 del fine-tuning, reemplazando una réplica de ~500 MB de VGG-16 por solo 17 MB de máscara + 7.5 MB de capas nuevas.

## 6. Limitaciones reconocidas

- **Requiere el task ID en inferencia.** Esta es la limitación más relevante para el encuadre de la Clase 32. Para evaluar una imagen hay que saber **a qué tarea pertenece** y aplicar la máscara correspondiente. Piggyback es, por tanto, un método **task-incremental**: asume un oráculo de tarea en test. No resuelve el escenario más duro (*class-incremental*), donde el modelo debe además inferir a qué tarea/conjunto de clases pertenece la entrada.
- **No hay transferencia entre tareas.** El paper lo declara como su principal drawback: "no hay margen para que las tareas añadidas se beneficien unas de otras". Solo las features de la tarea inicial (ImageNet pre-training) se reutilizan y adaptan; el conocimiento aprendido para la tarea A no fluye hacia la tarea B. Cada tarea explota el mismo backbone congelado de forma aislada.
- **Dependencia crítica de un buen backbone.** El rendimiento depende fuertemente de la calidad/diversidad de la inicialización (Tabla 5). Sobre un backbone pobre o mal alineado con el dominio objetivo, el techo del enmascarado baja. La diversidad de ImageNet es lo que hace que el método funcione tan bien.
- **Brecha en redes profundas y domain shift.** Como muestra la Tabla 4, en ResNet/DenseNet y tareas con domain shift fuerte aparece una brecha de ~2-5 % frente a redes individuales, parcialmente mitigable entrenando BN por tarea (a costo de algo de overhead y de tocar parámetros más allá de la máscara binaria pura).
- **Capacidad acotada por el backbone para tareas muy distintas.** Si una tarea exige capas especializadas (p. ej. detección de objetos) o más capacidad de la que el subconjunto de pesos enmascarables puede ofrecer, el método requiere extensiones (capas extra entrenadas desde cero, como en segmentación). Expandir la capacidad de las capas existentes según lo dicte la tarea queda como trabajo futuro.

## 7. Impacto y conexión con la Clase 32 (Olvido Catastrófico)

Piggyback se consolidó como uno de los métodos de referencia de la familia **basada en arquitectura** (también llamada *parameter isolation*) dentro del aprendizaje continuo, junto a su predecesor PackNet y a su sucesor directo, SupSup. Su aportación conceptual perdurable es mostrar que **el olvido catastrófico se puede eliminar por completo —no solo mitigar— si se renuncia a modificar los pesos compartidos**: en lugar de negociar entre tareas sobre un mismo conjunto de pesos (lo que hacen regularización y replay), se le da a cada tarea su propio "circuito" mediante una máscara. El precio de esa garantía es el task ID en inferencia y la ausencia de transferencia.

En el mapa de la Clase 32, Piggyback ocupa el casillero de los **métodos de arquitectura** contra el olvido, en contraste con los **métodos de regularización** (EWC, SI, LwF — penalizan cambios en pesos importantes) y los **métodos de repetición/replay** (rehearsal, generative replay — reentrenan con datos viejos o sintéticos). La slide "PiggyBack" de la clase lo resume en sus dos ideas vendibles: máscaras binarias sobre una red preentrenada y congelada, con un ahorro de 32-64× en espacio frente a guardar un modelo completo por tarea. Es importante encuadrarlo como **task-incremental**: la elegancia del cero-olvido viene de la mano del supuesto fuerte de conocer la tarea en test.

El hilo natural desde Piggyback es **SupSup** (Wortsman et al., 2020, [Supermasks in Superposition](/papers/supsup-wortsman-2020)): mientras Piggyback *aprende* la máscara binaria por backprop sobre pesos reales, SupSup *encuentra* "supermáscaras" sobre una red **aleatoria fija** (siguiendo la Lottery Ticket Hypothesis) y, crucialmente, propone inferir el task ID en test mediante superposición de máscaras y minimización de entropía — atacando justamente la limitación de Piggyback de necesitar el ID dado.

**Enlaces internos del curso:**
- Fundamento transversal: [Aprendizaje continuo](/fundamentos/aprendizaje-continuo).
- Clase: [Clase 32 — Olvido catastrófico](/clases/clase-32).
- Paper hermano (sucesor en la familia de máscaras): [SupSup — Supermasks in Superposition (Wortsman et al., 2020)](/papers/supsup-wortsman-2020).
