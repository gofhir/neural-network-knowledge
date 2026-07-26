# The Unreliability of Explanations in Few-Shot Prompting for Textual Reasoning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *The Unreliability of Explanations in Few-shot Prompting for Textual Reasoning*.
- **Autores:** Xi Ye y Greg Durrett. Ambos del Department of Computer Science, **The University of Texas at Austin**.
- **Venue:** *36th Conference on Neural Information Processing Systems* (**NeurIPS 2022**).
- **Preprint:** arXiv:2205.03401v2 (13 de octubre de 2022). Código y datos en `https://github.com/xiye17/TextualExplInContext`.
- **Financiamiento:** NSF (IIS-1814522, CAREER IIS-2145280), Open Philanthropy, Salesforce, Adobe.

El paper responde una pregunta empírica muy concreta: **¿mejora el in-context learning cuando le agregamos explicaciones al prompt de un modelo de lenguaje grande (LLM)?** Los autores estudian esto en dos tareas de razonamiento sobre texto —*question answering* (QA) e *inferencia de lenguaje natural* (NLI)— usando prompts que incluyen explicaciones en distintos estilos.

El hallazgo tiene dos caras que conviene no confundir. Primero, **sobre la utilidad**: incorporar explicaciones en el prompt para OPT, GPT-3 (davinci) e InstructGPT (text-davinci-001) produce mejoras de accuracy solo **pequeñas o moderadas** respecto al few-shot estándar, e incluso puede degradar el desempeño. La única excepción es **text-davinci-002**, que se beneficia de forma **sustancial** en las tres tareas. Segundo, **sobre la fiabilidad**: las explicaciones que los LLMs generan **pueden no ser factuales** (contienen alucinaciones que contradicen el contexto) ni **consistentes** (pueden no implicar la predicción que acompañan), incluso en un dataset sintético muy simple.

La contribución que salva el trabajo de un tono puramente escéptico es constructiva: aunque las explicaciones sean poco fiables, **su falta de factualidad correlaciona con predicciones incorrectas**. Esto convierte a la explicación en una **señal de verificación post-hoc**. Los autores entrenan **calibradores** ligeros que aproximan la factualidad mediante solapamiento léxico y ajustan las probabilidades de la predicción, mejorando el desempeño en los tres datasets.

Para la **Clase 34 (Razonamiento)** este paper aparece citado en la slide 30, bajo la pregunta "¿De dónde surge el CoT?", como la evidencia empírica de que **las explicaciones/Chain-of-Thought empiezan a ser realmente efectivas recién en text-davinci-002**. Es importante fijar desde ya un matiz que desarrollaremos en la Sección 7: este es un paper que **documenta** ese salto, pero explícitamente **no lo atribuye** al entrenamiento con código. La hipótesis del código es del profesor (y de otros comentaristas como Yao Fu), no de Ye y Durrett.

## 2. Contexto: el auge del "hazte explicar" y la pregunta por su fiabilidad

El trasfondo es el descubrimiento de que los LLMs escalados pueden aprender tareas de NLP "en contexto", a partir de unos pocos ejemplos en el prompt, sin actualizar sus parámetros (Brown et al., 2020). El problema es que este aprendizaje sigue siendo **poco entendido**: los modelos son sensibles al orden de los ejemplos (Zhao et al., 2021) y a veces ni siquiera usan las instrucciones o las etiquetas como uno esperaría (Min et al., 2022; Webson y Pavlick, 2022). Las herramientas clásicas de interpretabilidad —LIME, mapas de saliencia, gradientes integrados— tienen alto costo computacional o requieren acceso a gradientes, lo que las vuelve inservibles para modelos accedidos como caja negra vía API.

Frente a esa opacidad surge una idea atractiva: **dejar que el modelo se explique a sí mismo**. En lugar de dar solo pares entrada-etiqueta en el prompt, se agrega una explicación para cada par y se gatilla al modelo a que **genere una explicación para su propia predicción**. Esta línea es exactamente la que popularizaron los *scratchpads* de Nye et al. (2021) y el *Chain-of-Thought prompting* de Wei et al. (2022), y que luego adoptarían PaLM (Chowdhery et al., 2022) y trabajos como Lampinen et al. (2022) y Marasović et al. (2022). La intuición es que una explicación aporta **información mucho más rica** que una etiqueta sola, y podría guiar el proceso de inferencia.

Ye y Durrett hacen una observación aguda: **la evidencia previa de éxito se concentra en tareas simbólicas** con estructura muy distinta —suma de enteros, ejecución de programas, problemas matemáticos de palabras—. La pregunta abierta es si el beneficio se traslada a tareas de **razonamiento textual**, donde la respuesta debe estar anclada en un contexto de lenguaje natural provisto. Y, más allá de la accuracy, plantean una pregunta que casi nadie estaba haciendo en 2022: **¿son fiables esas explicaciones?** Es decir, ¿la narrativa que el modelo produce refleja realmente su "razonamiento", o es solo texto gramatical y convincente que puede engañar al usuario?

## 3. Contribución central

El paper articula cuatro hallazgos principales:

1. **Enchufar explicaciones en el prompt no siempre mejora sustancialmente** el in-context learning en razonamiento textual. El efecto es pequeño o moderado para OPT, GPT-3 e InstructGPT; solo text-davinci-002 mejora de forma prominente.
2. **Los LLMs generan explicaciones consistentes con sus predicciones, pero que pueden no estar ancladas factualmente en la entrada.** Es decir, la explicación "cierra" con la respuesta, pero puede afirmar hechos que el contexto contradice.
3. **La factualidad de una explicación funciona como indicador de la correctitud de la predicción.** Una explicación no factual señala, con alta probabilidad, una predicción incorrecta.
4. **Usando features que aproximan la factualidad, es posible calibrar el modelo** y mejorar el desempeño del in-context learning en todas las tareas.

El aporte metodológico es tratar el LLM como una **caja negra pura**: no se requieren gradientes, ni acceso a embeddings, ni fine-tuning del modelo grande. Solo se necesitan unos pocos ejemplos con explicaciones anotadas y un calibrador liviano de dos parámetros.

## 4. Método

### 4.1. Tareas y datasets

Se experimenta sobre dos tareas —QA de comprensión lectora y NLI— en **tres datasets en inglés**, cada uno con un conjunto de prueba de **250 ejemplos**:

- **S<span style="font-variant:small-caps">ynth</span> (Synthetic Multi-hop QA).** Un dataset sintético de QA multi-hop creado por los autores para tener un entorno **totalmente controlado**. Cada contexto contiene cuatro cadenas de razonamiento con el formato "A [verbo] B. B es [profesión]", con nombres, verbos y profesiones muestreados de *pools* (50 nombres, 30 verbos, 30 profesiones). El diseño es deliberadamente **simétrico** para eliminar *reasoning shortcuts*: no hay atajos espurios, así que responder exige recorrer efectivamente la cadena de dos saltos (ej.: "¿Quién pasa el rato con un estudiante?" → "Danielle es estudiante y Mary pasa el rato con Danielle"). La explicación correcta consiste siempre en las dos oraciones de soporte, lo que permite **juzgar automáticamente** factualidad y consistencia con reglas y expresiones regulares. Como referencia de dificultad, un RoBERTa fine-tuneado con 16 ejemplos **no supera el 50%** de accuracy, y necesita alrededor de **500 ejemplos** para acercarse al 100%.
- **A<span style="font-variant:small-caps">dv</span>H<span style="font-variant:small-caps">otpot</span> (Adversarial HotpotQA).** La versión adversarialmente aumentada de HotpotQA (Yang et al., 2018; Jiang y Bansal, 2019). El contexto de cada pregunta incluye dos párrafos de soporte reales y dos párrafos adversariales (distractores). Los autores balancean el conjunto de prueba con 125 ejemplos donde GPT-3 acierta y 125 donde falla, y **anotan manualmente** explicaciones para los ejemplos de entrenamiento (las oraciones de soporte por sí solas resultaban demasiado verbosas y con anáforas sin resolver). La concordancia entre anotadores (Cohen's Kappa) fue de 0.84 en correctitud y 0.85 en factualidad.
- **E-SNLI.** El dataset de NLI de Camburu et al. (2018), donde cada ejemplo es una premisa y una hipótesis a clasificar como *entailment*, *contradiction* o *neutral*. A diferencia de los otros dos, aquí las explicaciones son **lenguaje natural más abstracto** escrito por anotadores humanos (ej.: "the woman may not be his grandmother"), no snippets extraídos del contexto. Esto hace que la noción de "factualidad" casi no aplique, por lo que en E-SNLI se reporta principalmente consistencia.

### 4.2. Estilos de explicación (paradigmas de prompting)

Se compara el prompting **con y sin** explicaciones. Sin explicaciones es el few-shot estándar (**F<span style="font-variant:small-caps">ew</span>-S<span style="font-variant:small-caps">hot</span>**). Con explicaciones se usan los dos paradigmas más comunes:

- **Explain-then-Predict (E-P):** la explicación va **antes** de la etiqueta. El modelo genera primero la explicación y después la predicción. Es la categoría a la que pertenecen los *scratchpads* (Nye et al., 2021) y el Chain-of-Thought (Wei et al., 2022): la explicación **precede y puede influir** en la respuesta. Ejemplo: *"A: First, ... Second, ... The answer is X."*
- **Predict-then-Explain (P-E):** la predicción va **antes** de la explicación. Como se usa decodificación greedy, la explicación posterior **no influye** en la etiqueta ya emitida, aunque las explicaciones que están en el prompt sí siguen afectando la predicción. Ejemplo: *"A: X, because ..."*.

El número de *shots* es el máximo que cabe en el límite de longitud: **16** para S<span style="font-variant:small-caps">ynth</span>, **6** para A<span style="font-variant:small-caps">dv</span>H<span style="font-variant:small-caps">otpot</span> y **32** para E-SNLI. La decodificación es greedy (temperatura 0). Para reportar media y desviación estándar se muestrean varios grupos de ejemplos de entrenamiento (5 grupos para InstructGPT, el modelo primario; 3 para el resto).

### 4.3. Los cuatro LLMs

Se prueban **cuatro modelos**: **OPT (175B)** (Zhang et al., 2022) y **GPT-3 (davinci)** (Brown et al., 2020), ambos entrenados con el objetivo estándar de modelado causal de lenguaje; e **InstructGPT (text-davinci-001)** y **text-davinci-002**, entrenados con datos de instrucciones y anotaciones humanas. Los autores usan mayormente InstructGPT como modelo de trabajo por dos razones: era el más capaz disponible al momento de correr la mayoría de los experimentos, y todavía tenía margen amplio de mejora, lo que lo hace un banco de pruebas representativo de la situación de un ingeniero que recurre a explicaciones para mejorar un sistema que aún no rinde bien.

### 4.4. Calibradores

La pieza constructiva. Sea $\mathbf{p}$ el vector de probabilidades predichas por clase (en NLI) o el score de la respuesta (en QA), y sea $v$ un escalar extraído de la explicación que describe su factualidad. El calibrador es un modelo lineal:

$$\hat{\mathbf{p}} = \mathrm{softmax}\!\left(W\,[\mathbf{p}; v] + b\right)$$

Es una extensión de la calibración clásica (Platt, 1999; Guo et al., 2017; Zhao et al., 2021), que aplica una transformación afín solo sobre las probabilidades ($\hat{\mathbf{p}} = \mathrm{softmax}(W\mathbf{p} + b)$); aquí se agrega el factor $v$ de factualidad. Tiene **muy pocos parámetros** ($W$ y $b$), entrenables con un puñado de ejemplos extra —los que no caben en el prompt—, sin necesidad de anotar sus explicaciones.

Como no existe forma automática perfecta de medir factualidad, se la **aproxima por solapamiento léxico**. Para A<span style="font-variant:small-caps">dv</span>H<span style="font-variant:small-caps">otpot</span>, con explicación $E=(E^{(1)}, E^{(2)})$ y párrafos de contexto $P=(P^{(1)},\dots)$, el score de una oración de la explicación es:

$$V(E^{(i)}) = \max_{P \in \mathcal{P}} \frac{|E^{(i)} \cap P|}{|E^{(i)}|}$$

es decir, el máximo número de tokens solapados sobre todos los párrafos, normalizado por la longitud de la oración. El score de la explicación completa es $V(E) = \min_{E^{(i)} \in E} V(E^{(i)})$, porque **todas** las oraciones deben ser factuales para que la explicación entera lo sea. Para E-SNLI se usa un score análogo tomando la premisa como contexto.

## 5. Resultados

### 5.1. Utilidad de las explicaciones (Tabla 1)

El resultado central sobre accuracy es matizado. Para **OPT, GPT-3 e InstructGPT las mejoras son pequeñas a moderadas**. Tomando a InstructGPT (el mejor de esos tres): en S<span style="font-variant:small-caps">ynth</span>, E-P sube de **54.8 a 58.5**; en A<span style="font-variant:small-caps">dv</span>H<span style="font-variant:small-caps">otpot</span>, de **56.8 a 59.4**; en E-SNLI, P-E supera al few-shot por 2.6 puntos, mientras que E-P queda sustancialmente por debajo. No hay ganador único entre E-P y P-E: la mejor forma de usar explicaciones es **específica de la tarea**. En resumen, los LLMs "vanilla" (OPT y GPT-3) obtienen beneficio limitado de producir explicaciones, e incluso el InstructGPT de la serie Instruct no ve mejoras sustanciales.

**La excepción es text-davinci-002.** Este modelo **se beneficia enormemente** de las explicaciones en las tres tareas, y en su caso E-P es consistentemente más efectivo que P-E. En S<span style="font-variant:small-caps">ynth</span> pasa de **72.0 (few-shot) a 86.9 (E-P)**; en A<span style="font-variant:small-caps">dv</span>H<span style="font-variant:small-caps">otpot</span>, de **77.7 a 82.4**; en E-SNLI, de **69.1 a 75.6**. Este es precisamente el salto que la slide del curso destaca.

Los autores son cuidadosos sobre por qué ocurre: *"no está claro qué contribuye a esta diferencia. Hasta donde sabemos, las diferencias entre text-davinci-002 e InstructGPT no están descritas en ninguna publicación ni post de blog."* Comparando GPT-3 con InstructGPT, notan que el paso a modelos de la serie Instruct **no basta** para explicar la diferencia. Y concluyen con prudencia científica: *"dada la falta de transparencia con este modelo, dudamos en hacer afirmaciones científicas sobre los resultados que produce."*

También conectan su resultado con la literatura previa: en Wei et al. (2022) y Chowdhery et al. (2022), las explicaciones solo mostraban un beneficio **leve** en tareas de QA de dominio abierto como StrategyQA —que son más cercanas al setting de este paper— mientras que los grandes beneficios se concentraban en tareas **program-like** (suma de enteros, ejecución de programas). Esto refuerza la tesis de que el tipo de tarea importa.

### 5.2. Fiabilidad de las explicaciones (Tabla 2)

Aquí está el corazón del "unreliability" del título. Se evalúan dos ejes:

- **Factualidad:** si la explicación está fielmente anclada en el contexto (no contiene alucinaciones que lo contradigan).
- **Consistencia:** si la explicación **implica** la predicción; se parece a la noción de *plausibilidad* de Jacovi y Goldberg (2021).

El resultado es un **desacople entre predicción y "razonamiento"**. Los LLMs tienden a generar explicaciones **consistentes** (más del 80% en los tres datasets con la estructura de prompt adecuada) pero **menos factuales**. En InstructGPT, aunque las explicaciones mejoran el desempeño, son poco fiables **incluso en el setting sintético directo**: para S<span style="font-variant:small-caps">ynth</span> con E-P, la consistencia es 64.8% pero la factualidad 72.8%; con P-E la consistencia sube a 95.2% pero la factualidad cae a 51.6%. Comparando la factualidad en S<span style="font-variant:small-caps">ynth</span> entre GPT-3, InstructGPT y text-davinci-002, el instruction-tuning **mejora la factualidad**, pero ni siquiera el potente text-davinci-002 (91.6% de factualidad con E-P) logra explicaciones perfectamente ancladas. El caso paradigmático de la Figura 1 es un modelo que responde "Croatian" con una explicación gramaticalmente impecable que **inventa** un hecho ("Yelena Yemchuk is a Croatian professional photographer") cuando el contexto dice claramente que es ucraniana.

Esto es preocupante porque, como las explicaciones son inglés gramatical y suenan convincentes, **pueden engañar al usuario** haciéndole creer la respuesta del modelo aun cuando es incorrecta.

### 5.3. La falta de fiabilidad como señal (Sección 3.1)

El giro: si una explicación no factual **indica** un error, entonces la no factualidad es útil. La Tabla 2 (derecha) muestra que accuracy y factualidad/consistencia **correlacionan**, especialmente la factualidad. Conociendo si una explicación es factual, se puede adivinar la accuracy del modelo una fracción alta del tiempo (la columna "Accuracy = Factuality"). Una explicación no factual **muy probablemente** implica una predicción incorrecta en S<span style="font-variant:small-caps">ynth</span> a través de los cuatro LLMs. En A<span style="font-variant:small-caps">dv</span>H<span style="font-variant:small-caps">otpot</span>, factualidad y correctitud de InstructGPT coinciden el **80.0%** de las veces, superando ampliamente a la accuracy misma (62.0%). Las explicaciones factuales se emparejan con predicciones correctas mucho más que las no factuales; la consistencia también correlaciona con accuracy, pero es un indicador **inferior** a la factualidad.

### 5.4. Calibración (Sección 4)

**S<span style="font-variant:small-caps">ynth</span> (ejemplo motivador).** En el setting controlado, la factualidad se chequea con reglas. El procedimiento itera sobre las 5 respuestas candidatas de InstructGPT (límite de la API) y **rechaza** cualquier par respuesta-explicación cuya explicación sea no factual hasta hallar una factual. Esto sube la accuracy de P-E de **52.4% a 74.8%**. Para contexto, ni RoBERTa ni DeBERTa fine-tuneados con 16 ejemplos superan el 50%; con la ayuda de las explicaciones y el chequeo, InstructGPT logra resultados fuertes en few-shot.

**E-SNLI (Tabla 3).** Con el calibrador basado en explicaciones (**P-E+E<span style="font-variant:small-caps">xpl</span>C<span style="font-variant:small-caps">al</span>**) y 128 ejemplos, se alcanza la mejor accuracy de **68.5%**, unos **12 puntos** por encima del few-shot vanilla no calibrado (56.8%). Además supera al calibrador basado solo en probabilidades (P-E+P<span style="font-variant:small-caps">rob</span>C<span style="font-variant:small-caps">al</span>) por 3 puntos y a la variante few-shot calibrada por 5. Usar explicaciones es más efectivo que usar solo probabilidades; y a medida que crecen los datos de 32 a 128, el calibrador basado en explicaciones **sigue mejorando** mientras que los basados en probabilidades se saturan cerca de 96. Como referencia, RoBERTa fine-tuneado con 128 shots solo llega a 54.9%, por debajo de los modelos basados en GPT-3.

**A<span style="font-variant:small-caps">dv</span>H<span style="font-variant:small-caps">otpot</span> (Tabla 4).** Aquí la calibración ajusta los scores de confianza para un setting de "selective QA" (Kamath et al., 2020), donde el modelo puede **abstenerse** en preguntas de baja confianza. La métrica es el área bajo la curva cobertura-accuracy (**AUC**). E-P+E<span style="font-variant:small-caps">xpl</span>C<span style="font-variant:small-caps">al</span> logra un AUC de **68.8**, superando al few-shot por 7 puntos y a E-P por 4, con la ganancia más grande en los intervalos de mayor confianza. Basta con tan solo 32 ejemplos, gracias a que el calibrador tiene solo dos parámetros.

## 6. Aclaración sobre la *framing* de la slide

La slide 30 de la Clase 34 usa este paper como respaldo de la afirmación "las explicaciones/CoT empiezan a ser efectivas recién en text-davinci-002" y de allí el profesor hipotetiza que la diferencia proviene del **entrenamiento con gran cantidad de código**. Conviene separar con precisión qué dice el paper y qué es interpretación posterior:

1. **Lo que el paper sí demuestra:** empíricamente, sobre QA y NLI de razonamiento textual, las explicaciones dan solo mejoras pequeñas/moderadas para OPT, GPT-3 y text-davinci-001, pero mejoras **sustanciales** para text-davinci-002. Ese salto es un dato sólido y reproducible del trabajo.
2. **Lo que el paper NO afirma:** que el salto se deba al entrenamiento con código. Al contrario, los autores dicen explícitamente que **no saben** qué causa la diferencia, que las diferencias entre text-davinci-002 e InstructGPT **no están documentadas** en ninguna publicación ni blog, y que por la falta de transparencia **dudan en hacer afirmaciones científicas**. Este paper **no trata sobre código**; su tema es la (falta de) fiabilidad de las explicaciones y su uso como señal de calibración.
3. **De dónde viene la hipótesis del código:** de la comunidad, notablemente del análisis de **Yao Fu y colaboradores** ("How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources", 2022), que popularizó la idea de que la capacidad de razonamiento en cadena de la serie Codex/text-davinci-002 podría rastrearse al entrenamiento con código. Es una hipótesis plausible y ampliamente citada, pero **es externa a Ye y Durrett** y sigue siendo especulativa: OpenAI nunca publicó los detalles de entrenamiento que permitirían confirmarla.

En síntesis: el paper de Ye y Durrett es la **evidencia del salto**, no la explicación del salto. La narrativa "es por el código" es una hipótesis del profesor y de la comunidad, no una conclusión del artículo. Presentarlo con ese matiz es fiel tanto a la slide como al texto.

## 7. Limitaciones

Los propios autores señalan varias:

- **La feature de factualidad es débil.** El solapamiento léxico es una señal ruidosa de correctitud de la explicación; puede fallar (como en el ejemplo de la Figura 1). Un modelo de *entailment* suficientemente fuerte debería, en teoría, hacer esta verificación mejor y sin fine-tuning —incluso un LLM entrenado específicamente para verificar—, pero eso queda fuera del setting de caja negra pura que buscaban.
- **Alcance de tareas.** Los resultados aplican a QA y NLI de razonamiento textual; no se extrapolan sin más a razonamiento simbólico o matemático, donde CoT sí muestra beneficios grandes.
- **Opacidad de los modelos.** La imposibilidad de saber en qué difiere text-davinci-002 limita cualquier conclusión causal sobre por qué se beneficia de las explicaciones.
- **Requiere datos extra.** El calibrador necesita un puñado de ejemplos adicionales (aunque los aprovecha justamente porque no caben en el prompt).
- **E-SNLI y la factualidad.** En NLI la factualidad casi no aplica porque las explicaciones requieren conocimiento de sentido común externo, difícil de anclar en la entrada; por eso allí se reporta sobre todo consistencia.

## 8. Conexión con la Clase 34 (Razonamiento)

Este paper ocupa un lugar bisagra en la clase. Por un lado, **fecha empíricamente el nacimiento del CoT efectivo**: aporta la evidencia de que, para razonamiento textual, las explicaciones en el prompt solo despegan con text-davinci-002, dando material a la discusión "¿de dónde surge el CoT?". Por otro lado, es un **contrapunto crítico y saludable** al entusiasmo con el prompting de razonamiento. Justo cuando Wei et al. (2022) mostraban que "pensar paso a paso" desbloquea capacidades, Ye y Durrett recuerdan que **una cadena de razonamiento verbalizada no es un registro fiel del proceso interno del modelo**: puede ser consistente pero no factual, convincente pero falsa. Esta distinción entre *plausibilidad* y *fidelidad* de las explicaciones anticipa toda una línea posterior sobre *faithfulness* del CoT.

El aporte constructivo —usar la (in)fiabilidad de la explicación como señal para calibrar y abstenerse— es además una idea que envejece muy bien: es un precursor conceptual de los verificadores, del *self-consistency* y de los *LLM-as-a-judge* que verifican razonamientos generados por otro modelo. La lección transversal para la clase es que el razonamiento explícito de un LLM es simultáneamente una **herramienta de desempeño** y una **superficie de verificación**, y que ambas cosas deben tratarse con escepticismo medido.

---

**Nota sobre relevancia para salud.** El hallazgo central de este paper tiene una traducción directa y grave al terreno clínico: **una explicación plausible generada por un LLM no garantiza que la decisión sea correcta ni que esté fundamentada.** Ye y Durrett muestran que un modelo puede emitir una justificación gramaticalmente impecable y aparentemente lógica que, sin embargo, **alucina hechos que contradicen los datos de entrada** —y que esa narrativa convincente puede engañar al usuario para que confíe en una respuesta equivocada. En un contexto de apoyo a la decisión clínica (razonamiento diagnóstico, conciliación de medicamentos, resumen de historia clínica, matching de registros de pacientes), esto implica que **la explicación jamás debe tomarse como prueba de correctitud**. La postura correcta es la del propio paper: **verificar, no confiar en la narrativa** —anclar cada afirmación de la explicación contra la fuente (el contexto, el registro, la evidencia), y usar la (in)consistencia entre explicación y datos como una señal para abstenerse o escalar a revisión humana, no como un sello de confianza.
