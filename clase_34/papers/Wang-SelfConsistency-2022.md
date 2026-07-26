# Self-Consistency Improves Chain of Thought Reasoning in Language Models — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Self-Consistency Improves Chain of Thought Reasoning in Language Models*.
- **Autores:** Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Sharan Narang, Aakanksha Chowdhery, Ed H. Chi, Denny Zhou. Todos en **Google Research, Brain Team**.
- **Venue:** *International Conference on Learning Representations* (**ICLR 2023**).
- **Preprint:** arXiv:2203.11171 (v1 de marzo de 2022; v4 del 7 de marzo de 2023).
- **Linaje:** es la continuación directa de *Chain-of-Thought Prompting* (Wei et al., 2022), coescrito por varios de los mismos autores (Wang, Wei, Schuurmans, Le, Chi, Zhou). Se sitúa en el linaje de trabajos de Google sobre razonamiento emergente en LLMs de gran escala (PaLM, LaMDA).

El paper propone **self-consistency** (auto-consistencia), una **estrategia de decoding** que reemplaza el *greedy decoding* usado habitualmente con Chain-of-Thought (CoT). En lugar de decodificar una única cadena de razonamiento tomando en cada paso el token más probable, self-consistency **muestrea un conjunto diverso de cadenas de razonamiento** desde el decoder del modelo y luego **marginaliza sobre esas cadenas** para quedarse con la respuesta final más consistente, es decir, la más frecuente por **voto de mayoría**. La idea descansa en una intuición sobre el razonamiento humano: un problema complejo suele admitir **múltiples caminos de razonamiento distintos que convergen a la misma respuesta correcta**, y si varias formas de pensar llegan al mismo resultado, uno tiene mayor confianza en que ese resultado es correcto.

El método es **completamente no supervisado**: funciona *off-the-shelf* sobre modelos preentrenados, no requiere entrenamiento adicional, ni fine-tuning, ni modelos auxiliares, ni anotación humana. Esto lo distingue de enfoques previos que entrenaban un *verifier* (Cobbe et al., 2021) o un *re-ranker* (Thoppilan et al., 2022). Los autores lo describen como un **"self-ensemble"**: no combina las salidas de múltiples modelos entrenados, sino múltiples muestras de un **único modelo**.

La evaluación empírica es amplia: cuatro modelos de escalas variadas (**UL2-20B, GPT-3-175B, LaMDA-137B y PaLM-540B**) sobre benchmarks de razonamiento aritmético, de sentido común y simbólico. Self-consistency mejora a CoT con márgenes notables y alcanza nuevos estados del arte en varias tareas, con ganancias absolutas reportadas en el abstract de **+17.9% en GSM8K, +11.0% en SVAMP, +12.2% en AQuA, +6.4% en StrategyQA y +3.9% en ARC-challenge**.

Para la **Clase 34 (Razonamiento)** este paper importa porque es la **materialización canónica** de la idea del slide 33: si muestreamos muchas respuestas de un LLM, es probable que alguna sea correcta (Pass@k). Self-consistency convierte esa observación pasiva en un **método práctico**: no basta con que exista una muestra correcta; hay que **recuperarla sin conocer la respuesta**, y la agregación por mayoría lo logra sin supervisión. Es también uno de los primeros ejemplos limpios de **test-time compute**: gastar más cómputo en inferencia —muchas muestras en vez de una— para comprar precisión sin tocar los pesos del modelo.

## 2. Contexto: Chain-of-Thought, greedy decoding y su fragilidad

Los modelos de lenguaje demostraron éxito notable en muchas tareas de NLP, pero su capacidad de **razonar** se veía como una limitación que no se supera solo aumentando la escala del modelo (Rae et al., 2021; BIG-bench). Para atacar esto, **Wei et al. (2022)** propusieron **chain-of-thought prompting**: en lugar de pedirle al modelo que responda directamente, se lo induce a generar una **serie de pasos intermedios** de razonamiento en lenguaje natural que imitan cómo una persona resolvería la tarea. El ejemplo canónico del paper: ante "Si hay 3 autos en el estacionamiento y llegan 2 más, ¿cuántos autos hay?", en vez de responder "5" directamente, el modelo produce "Ya hay 3 autos. Llegan 2 más. Ahora hay 3 + 2 = 5 autos. La respuesta es 5". Esta descomposición mejora significativamente el desempeño en tareas de razonamiento multi-paso.

El problema es **cómo se decodifica** esa cadena. La práctica estándar con CoT era el **greedy decoding**: en cada posición se elige el token de máxima probabilidad, produciendo **una sola** cadena de razonamiento. Esto tiene dos debilidades que el paper identifica:

1. **Repetitividad y óptimo local.** El greedy decoding es conocido por caer en repeticiones y quedarse en soluciones localmente óptimas que no son globalmente las mejores.
2. **Fragilidad de un único camino.** Si esa única cadena greedy contiene un error en cualquiera de sus pasos, la respuesta final es errónea, sin ningún mecanismo de recuperación. El razonamiento multi-paso es **frágil**: un solo desliz aritmético o lógico arruina todo el encadenamiento.

Hay una tensión conceptual interesante que el paper subraya. Las tareas de razonamiento tienen típicamente una **respuesta única y fija**, razón por la cual los investigadores gravitaban naturalmente hacia decoding determinístico (greedy). El muestreo (*sampling*), en cambio, se asociaba con la **generación abierta de texto** —historias, diálogo— donde la diversidad es deseable. El hallazgo central del paper es contraintuitivo: **incluso cuando la respuesta deseada es fija, introducir diversidad en el proceso de razonamiento resulta altamente beneficioso.** Self-consistency habita, en palabras de los autores, "un espacio interesante entre la generación abierta de texto y la generación óptima con respuesta fija".

Los enfoques previos para mejorar la calidad de generación en tareas de razonamiento —entrenar un *verifier* que re-ranquea soluciones (Cobbe et al., 2021), o recolectar anotaciones humanas para entrenar un re-ranker (Thoppilan et al., 2022)— eran costosos y específicos de la tarea. Self-consistency busca la misma mejora **sin ningún componente entrenado adicional**.

## 3. Contribución central

La contribución es una **estrategia de decoding "sample-and-marginalize"** (muestrear y marginalizar) que sustituye al greedy decoding en CoT, junto con la evidencia empírica de que esta sustitución produce mejoras grandes y robustas sin costo de entrenamiento. Sus rasgos distintivos:

- **No supervisada y off-the-shelf.** No requiere entrenamiento, fine-tuning, modelos auxiliares ni anotación humana. Se aplica sobre modelos preentrenados tal cual.
- **Self-ensemble sobre un único modelo.** A diferencia del ensemble clásico (varios modelos entrenados cuyas salidas se combinan), aquí un solo modelo produce múltiples muestras que se agregan. El paper muestra en el apéndice que el ensemble de múltiples modelos rinde **mucho peor** que el self-ensemble, porque los modelos de menor capacidad arrastran hacia abajo el desempeño (por ejemplo, ensamblar LaMDA-137B + PaLM-540B da 36.9% en GSM8K frente al 74.4% de self-consistency solo sobre PaLM-540B).
- **Compatible con cualquier esquema de muestreo.** Funciona con temperature sampling, top-$k$ sampling y nucleus sampling, y es robusta a la elección de sus parámetros.
- **Un beneficio secundario: estimación de incertidumbre.** El grado de consistencia (porcentaje de muestras que concuerdan con la respuesta agregada) está altamente correlacionado con la exactitud, lo que permite usar la baja consistencia como señal de baja confianza —una forma incipiente de que el modelo "sepa cuándo no sabe".

## 4. Método: muestreo y voto de mayoría marginalizado

### 4.1. El procedimiento en tres pasos

Self-consistency (ilustrado en la Figura 1 del paper) consiste en:

1. **Prompt con CoT.** Se induce al modelo con un conjunto de ejemplos (*exemplars*) de cadena de razonamiento escritos manualmente, exactamente como en Wei et al. (2022).
2. **Muestrear caminos diversos.** En lugar de decodificar greedy, se **muestrea** desde el decoder del modelo un conjunto de cadenas de razonamiento candidatas. Cada camino puede llevar a una respuesta final distinta.
3. **Marginalizar y agregar.** Se recorre el conjunto de respuestas finales y se elige la **más consistente** (la más frecuente), marginalizando sobre las cadenas de razonamiento.

### 4.2. Formalización

Sea $a_i \in \mathcal{A}$ la respuesta generada en la $i$-ésima muestra, con $i = 1, \dots, m$ indexando las $m$ salidas muestreadas del decoder, y $\mathcal{A}$ un conjunto fijo de respuestas posibles. Self-consistency introduce una **variable latente** $r_i$, la secuencia de tokens que representa el **camino de razonamiento** de la $i$-ésima salida, y acopla la generación del par $(r_i, a_i)$ con $r_i \rightarrow a_i$: es decir, generar el razonamiento $r_i$ es opcional y sirve solo para llegar a la respuesta final $a_i$.

Tras muestrear múltiples pares $(r_i, a_i)$, self-consistency **marginaliza sobre** $r_i$ tomando un **voto de mayoría** sobre $a_i$:

$$\arg\max_{a} \sum_{i=1}^{m} \mathbb{1}(a_i = a)$$

Esta es la respuesta más "consistente" del conjunto de respuestas finales. La palabra clave es **marginalizar**: la cadena de razonamiento es una variable latente que no nos interesa por sí misma; lo que interesa es la distribución sobre la respuesta final tras integrar (sumar) sobre todos los caminos posibles.

### 4.3. Variantes de agregación

Además del voto de mayoría simple (sumar sin pesos), se puede **ponderar** cada par $(r_i, a_i)$ por su probabilidad $P(r_i, a_i \mid \text{prompt}, \text{question})$. Esta probabilidad puede tomarse sin normalizar, o normalizada por la longitud de la salida:

$$P(r_i, a_i \mid \text{prompt}, \text{question}) = \exp\left( \frac{1}{K} \sum_{k=1}^{K} \log P(t_k \mid \text{prompt}, \text{question}, t_1, \dots, t_{k-1}) \right)$$

donde $t_k$ es el $k$-ésimo token del par $(r_i, a_i)$ y $K$ el total de tokens. La Tabla 1 (sobre PaLM-540B) compara estas estrategias de agregación. El hallazgo práctico es que **el voto de mayoría simple ("unweighted sum") rinde casi igual que la suma ponderada normalizada** (por ejemplo, 74.4% vs. 74.1% en GSM8K). La razón es que las probabilidades condicionales normalizadas de los distintos pares resultan muy parecidas entre sí: el modelo considera esas generaciones como "igualmente probables". Esto revela de paso que **el modelo no está bien calibrado** —no distingue bien soluciones correctas de incorrectas por su probabilidad—, lo que explica por qué trabajos previos entrenaban re-rankers. La "weighted average" (dividir por la cuenta de cada respuesta) rinde mucho peor y se descarta.

### 4.4. Por qué funciona: la hipótesis de la consistencia

La justificación teórica es una hipótesis sobre la estructura de los errores. Como los modelos de lenguaje **no son razonadores perfectos**, pueden producir cadenas de razonamiento incorrectas o equivocarse en algún paso. Pero la afirmación clave es: **los procesos de razonamiento correctos, aunque sean diversos, tienden a coincidir más en su respuesta final que los procesos incorrectos.** Un problema bien resuelto por dos caminos distintos aterriza en el mismo número; dos errores distintos aterrizan, con alta probabilidad, en números distintos. Por eso la respuesta correcta acumula votos y las incorrectas se dispersan. La Tabla 4 muestra ejemplos concretos donde dos caminos muestreados —con razonamientos textualmente diferentes— reparan el error del greedy decoding al converger ambos a la respuesta verdadera.

### 4.5. Esquema de muestreo

Self-consistency es compatible con la mayoría de algoritmos de muestreo. En los experimentos: para UL2-20B y LaMDA-137B se usó temperature sampling con $T = 0.5$ truncado en top-$k$ ($k = 40$); para PaLM-540B, $T = 0.7$, $k = 40$; para GPT-3, $T = 0.7$ sin truncación top-$k$. El parser de la respuesta final es dependiente de la tarea: en aritmética se toma la primera parte numérica tras "The answer is", y en sentido común el string completo tras esa misma marca.

## 5. Experimentos y resultados

### 5.1. Configuración

Todos los experimentos son en régimen **few-shot**, sin entrenamiento ni fine-tuning. Se usan los mismos prompts que Wei et al. (2022): 8 exemplars escritos a mano para todas las tareas aritméticas, y 4–7 exemplars para cada tarea de sentido común. El baseline es **CoT con greedy decoding**. Los resultados de self-consistency se promedian sobre **10 corridas**, muestreando **40 salidas** por corrida.

Los benchmarks:

- **Aritmético:** AddSub, MultiArith, ASDiv, AQUA-RAT, GSM8K y SVAMP.
- **Sentido común:** CommonsenseQA (CSQA), StrategyQA y ARC (easy y challenge).
- **Simbólico:** concatenación de últimas letras (*last letter concatenation*) y *Coinflip*, en un régimen fuera de distribución (OOD): el prompt tiene ejemplos de 2 letras/2 lanzamientos, pero se evalúa con 4 letras/4 lanzamientos.

### 5.2. Razonamiento aritmético

Self-consistency mejora sobre CoT en los cuatro modelos y en todas las tareas (Tabla 2). Un hallazgo llamativo es que **las ganancias crecen con la escala del modelo**: se observa +3%–6% absoluto sobre UL2-20B, pero +9%–23% sobre LaMDA-137B y GPT-3. Incluso para los modelos grandes que ya tienen alta exactitud (GPT-3, PaLM-540B), self-consistency aporta ganancias significativas adicionales: +12%–18% en tareas como AQuA y GSM8K, y +7%–11% en SVAMP y ASDiv. Algunas cifras concretas:

- **GSM8K, PaLM-540B:** CoT 56.5% → self-consistency **74.4% (+17.9)**.
- **AQuA, PaLM-540B:** 35.8% → **48.3% (+12.5)**.
- **SVAMP, PaLM-540B:** 79.0% → **86.6% (+7.6)**.
- **GSM8K, GPT-3 code-davinci-002:** 60.1% → **78.0% (+17.9)**.
- **MultiArith, LaMDA-137B:** 51.8% → **75.7% (+23.9)**, la ganancia absoluta más grande de la tabla.

Con self-consistency se alcanzan nuevos estados del arte en casi todas las tareas, comparando favorablemente incluso contra métodos que requieren entrenamiento o fine-tuning con miles de ejemplos (como el verifier de Cobbe et al. sobre GSM8K).

### 5.3. Razonamiento de sentido común y simbólico

En la Tabla 3, self-consistency vuelve a mejorar en los cuatro modelos y obtiene SoTA en 5 de 6 tareas. Ejemplos sobre PaLM-540B: StrategyQA 75.3% → 81.6% (+6.3); ARC-challenge 85.2% → 88.7% (+3.5). En el régimen simbólico OOD (más difícil, porque PaLM/GPT-3 ya logran exactitud perfecta en distribución), la ganancia sigue siendo significativa con modelos suficientemente grandes.

### 5.4. Efecto del número de caminos muestreados

La Figura 2 muestra la exactitud en función del número de caminos (1, 5, 10, 20, 40). Muestrear más caminos **mejora consistentemente** el desempeño, lo que refuerza que la diversidad es el ingrediente activo. Sin embargo, la curva **satura rápido**: la mayoría de la ganancia se obtiene con relativamente pocos caminos.

### 5.5. Self-consistency ayuda incluso cuando CoT perjudica

Ye & Durrett (2022) habían mostrado que a veces CoT **daña** el desempeño respecto al prompting estándar. En tareas de NLP comunes (BoolQ, HotpotQA, e-SNLI, ANLI, RTE — Tabla 5), donde añadir CoT baja la exactitud frente al prompting estándar, self-consistency **cierra la brecha y supera al prompting estándar**. Por ejemplo, en ANLI-R1: estándar 69.1%, CoT 68.8%, self-consistency **78.5%**. Esto convierte a CoT+self-consistency en una manera confiable de agregar racionales en in-context learning.

### 5.6. Comparaciones con otras estrategias de decoding

- **Sample-and-rank** (muestrear y quedarse con la secuencia de mayor log-probabilidad): mejora algo con más muestras, pero **mucho menos** que self-consistency (Figura 3).
- **Beam search** (Tabla 6, UL2-20B): self-consistency con sampling supera al beam search con el mismo número de haces/caminos. Notablemente, hacer self-consistency *usando* beam search para decodificar cada camino rinde **peor** que usando sampling, porque el beam search produce **menor diversidad** en las salidas —y la diversidad es justamente la clave.
- **Ensembles** (Tabla 7, LaMDA-137B): frente a ensembles por permutación del orden de exemplars (×40) o por múltiples conjuntos de prompts, self-consistency logra ganancias mucho mayores (por ejemplo, GSM8K: ensemble ~19% vs. self-consistency 27.7%).

### 5.7. Estudios de robustez

- **Robustez al muestreo (Figura 4, izquierda):** self-consistency mejora bajo un amplio rango de $T$, $k$ y $p$ (nucleus).
- **Robustez a la escala (Figura 4, derecha):** mejora en todas las escalas de LaMDA; la ganancia es menor en modelos pequeños porque ciertas habilidades (aritmética) solo emergen con escala suficiente.
- **Robustez a prompts imperfectos (Tabla 8):** con un prompt donde los números intermedios del razonamiento se reemplazaron por números aleatorios (dejando solo la respuesta final correcta), el greedy baja de 17.1% a 14.9%, pero self-consistency recupera y llega a 23.4%.
- **Prompts con ecuaciones y zero-shot CoT (Tabla 8):** self-consistency también mejora con razonamiento no en lenguaje natural (ecuaciones), aunque menos —las ecuaciones son cortas y dejan poco margen para diversidad—, y aporta **+26.2%** sobre zero-shot CoT (Kojima et al., 2022), llevando GSM8K de 43.0% a 69.2%.
- **Consistencia como incertidumbre (Figura 5):** el % de consistencia correlaciona fuertemente con la exactitud.

## 6. Limitaciones

Los autores son explícitos sobre las restricciones:

1. **Costo de cómputo.** El principal costo es que self-consistency **requiere más cómputo en inferencia**: hay que muestrear y procesar decenas de caminos en vez de uno. La mitigación práctica es que el desempeño satura rápido (Figura 2), de modo que con **5 o 10 caminos** ya se captura la mayor parte de la ganancia sin incurrir en demasiado costo.
2. **Requiere una respuesta única identificable.** El método aplica solo a problemas donde la respuesta final proviene de un **conjunto fijo** y se puede parsear y comparar. En principio se extiende a generación abierta si se define una buena métrica de consistencia entre generaciones (si dos respuestas concuerdan o se contradicen), pero eso no es trivial.
3. **Racionales potencialmente incorrectos.** El modelo puede generar caminos de razonamiento incorrectos o sin sentido incluso cuando la respuesta final agregada es correcta (por ejemplo, en el caso de StrategyQA de la Tabla 4, las cifras de población citadas no son exactamente correctas). Hace falta más trabajo para **anclar** (*ground*) mejor la generación de racionales, un punto también señalado en la declaración ética: los racionales solo deberían usarse para inspeccionar cómo el modelo llega a su respuesta, con precaución.

Como trabajo futuro, los autores sugieren usar self-consistency para **generar mejores datos supervisados** con los que hacer fine-tuning, de modo que el modelo dé predicciones más exactas en **una sola** corrida de inferencia —una idea que anticipa el *self-training* y la destilación de razonamiento posteriores.

## 7. Conexión con la Clase 34 (Razonamiento): Pass@k, test-time compute, CoT y ToT

Self-consistency es la bisagra entre varias ideas de la Clase 34.

**Pass@k (slide 33).** La observación de que, muestreando muchas respuestas, es probable que **alguna** sea correcta, es una cota superior optimista: Pass@k supone un oráculo que sabe cuál de las $k$ muestras es la buena. En la práctica no tenemos ese oráculo. Self-consistency es precisamente el **puente entre Pass@k y una métrica realizable**: sin conocer la respuesta correcta, usa la **frecuencia** como sustituto del oráculo. La hipótesis de que los caminos correctos concuerdan más entre sí que los incorrectos es lo que hace que "la respuesta más votada" se aproxime a "la respuesta correcta". Es el mecanismo que **cosecha** la promesa de Pass@k sin supervisión.

**Test-time compute.** Self-consistency es un ejemplo temprano y limpio de **escalar cómputo en inferencia** en vez de en entrenamiento. Los pesos del modelo no cambian; lo único que cambia es cuánto se gasta al responder (un camino vs. 40). Ganamos exactitud comprando muestras. Esta es exactamente la lógica que años después estructuraría a los modelos de razonamiento (*reasoning models*) que "piensan más" antes de responder. El eje de escalamiento se corre de "modelo más grande" a "más cómputo por pregunta", y self-consistency es la demostración de que ese eje **también** paga.

**Reducción de varianza.** Conceptualmente, el greedy decoding es una única muestra ruidosa del proceso de razonamiento del modelo. Self-consistency toma muchas muestras y agrega por moda. Es un estimador de **menor varianza** de "la respuesta que el modelo tiende a dar", que mitiga tanto la estocasticidad de una sola muestra como la trampa del óptimo local del greedy.

**Relación con CoT y con Tree of Thoughts (ToT).** Self-consistency **presupone** CoT: sin cadenas de razonamiento explícitas no habría diversidad de caminos que marginalizar. Es una capa de decoding *encima* de CoT. Frente a ToT, la diferencia es la estructura de la búsqueda: self-consistency muestrea caminos **independientes** en paralelo y agrega solo al final (búsqueda "plana", sin ramificación ni backtracking), mientras que ToT explora un **árbol** de estados intermedios con evaluación y poda por pasos. Self-consistency puede leerse como el caso más simple —lineal, sin evaluación intermedia, agregación por mayoría— de la familia de métodos que exploran el espacio de razonamientos en tiempo de inferencia. Su gran virtud es que no necesita ninguna función de evaluación de estados intermedios: el "juez" es la concordancia de las respuestas finales.

## 8. Nota final: relevancia para salud

En decisiones clínicas asistidas por LLMs —extracción de diagnósticos, cálculo de dosis, razonamiento sobre guías o interacciones farmacológicas— la fragilidad del greedy decoding es especialmente peligrosa: un único desliz en un paso intermedio produce una recomendación errónea sin ninguna señal de alarma. Self-consistency ofrece dos aportes valiosos en este dominio. Primero, actúa como un **ensemble de razonamientos** que reduce la varianza: al pedirle al modelo que llegue a la conclusión por varios caminos y quedarse con la respuesta que concuerda, el voto mayoritario **reduce la varianza** de una única generación ruidosa y hace la decisión más robusta a errores idiosincráticos de un camino particular. Segundo, y quizás más importante para la seguridad, la **consistencia como estimador de incertidumbre** (la correlación entre concordancia y exactitud) permite marcar los casos de **baja consistencia** —donde los caminos discrepan— como candidatos a revisión humana, materializando la capacidad de "saber cuándo no se sabe". Debe recordarse, eso sí, la limitación que el propio paper subraya: los racionales generados pueden ser incorrectos aun cuando la respuesta agregada sea correcta, de modo que en salud self-consistency es una herramienta de robustez y de triaje de confianza, nunca un sustituto de la verificación clínica.
