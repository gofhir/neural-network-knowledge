# Emergent Abilities of Large Language Models (Wei et al., 2022) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Emergent Abilities of Large Language Models*.
- **Autores:** Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus. Afiliaciones: **Google Research**, **Stanford University**, **UNC Chapel Hill** y **DeepMind**.
- **Venue:** *Transactions on Machine Learning Research* (TMLR), agosto de 2022. Revisado en OpenReview (`id=yzkSU5zdwD`).
- **Preprint:** arXiv:2206.07682v2 (26 oct 2022).
- **Naturaleza del trabajo:** no es un paper que proponga un método o un modelo nuevo. Es un **survey conceptual** que revisa resultados dispersos en la literatura previa y los organiza bajo un mismo marco, el de las *habilidades emergentes*. Esta condición de meta-análisis es importante para leerlo con cuidado: todas las cifras provienen de trabajos de terceros (GPT-3, LaMDA, Gopher, Chinchilla, PaLM, BIG-Bench, etc.).

El paper parte de un hecho bien establecido: escalar modelos de lenguaje —más cómputo de entrenamiento, más parámetros, más datos— **mejora el rendimiento de forma predecible**, y esa mejora se captura con *leyes de escala* (scaling laws) que abarcan empíricamente más de siete órdenes de magnitud en la pérdida de entropía cruzada (Kaplan et al., 2020; Hoffmann et al., 2022). Sobre ese fondo predecible, Wei et al. dirigen la atención hacia un fenómeno **impredecible**: ciertas habilidades **no están presentes en modelos pequeños y sí aparecen en modelos grandes**, sin poder anticiparse extrapolando la curva de los modelos menores. A eso llaman *habilidad emergente*.

La tesis central se resume en una frase que el paper toma prestada del físico Philip Anderson y su ensayo *More Is Different* (1972): **"la emergencia ocurre cuando cambios cuantitativos en un sistema producen cambios cualitativos en su comportamiento"**. Trasladado a los LLM: acumular escala no solo hace *mejor* lo que el modelo ya hacía, sino que en algún umbral **habilita capacidades nuevas** —aritmética multi-dígito, seguimiento de instrucciones, razonamiento por cadena de pensamiento— que antes simplemente no existían.

Para la **Clase 34 (Razonamiento)** este paper es la referencia canónica de por qué el razonamiento vía *Chain-of-Thought* (CoT) **no funciona a cualquier escala**: es una habilidad emergente que solo supera al prompting estándar a partir de aproximadamente $10^{23}$ FLOPs de entrenamiento (~100 mil millones de parámetros). El slide 25 de la clase menciona que los LLM pre-entrenados exhiben "comportamientos emergentes, como el In-Context Learning"; este trabajo es la fuente que fundamenta ese enunciado.

## 2. Contexto: leyes de escala y el salto a las capacidades

Durante los años previos a este paper, la comunidad de NLP consolidó la idea de que **la escala mejora el rendimiento de manera metódicamente predecible**. Las scaling laws de Kaplan et al. (2020) y Hoffmann et al. (2022) mostraron que la pérdida de entropía cruzada durante el pre-entrenamiento cae de forma suave y regular en función del cómputo, los parámetros y el tamaño del dataset. Esa suavidad es lo que permite, por ejemplo, *presupuestar* un entrenamiento: se sabe de antemano, con buena aproximación, qué pérdida alcanzará un modelo de tamaño dado.

El problema que motiva a Wei et al. es que **esa predecibilidad de la pérdida de pre-entrenamiento no se traslada automáticamente al rendimiento en tareas downstream**. Ganguli et al. (2022) ya habían notado que, para ciertas tareas, el desempeño *no* mejora de forma continua con la escala y *no* puede anticiparse. El paper convierte esa observación en un objeto de estudio con nombre propio.

El concepto de emergencia no es nuevo ni exclusivo del machine learning: proviene de la física, la biología y las ciencias de la complejidad (Anderson, 1972; Hwang et al., 2012). La analogía física más útil es la de una **transición de fase**: el agua no se enfría gradualmente hasta convertirse en hielo, sino que cambia de estado abruptamente al cruzar los 0 °C. Del mismo modo, una habilidad emergente muestra un rendimiento cercano al azar hasta cierto umbral de escala y luego un **salto** a un desempeño claramente superior al azar. El paper enmarca las scaling laws como el régimen "predecible" y las habilidades emergentes como el régimen "impredecible" que las scaling laws no capturan.

Un matiz que el paper subraya: **no existe un único proxy que capture toda la noción de escala**. Se puede medir en FLOPs de entrenamiento, en número de parámetros o en tamaño del dataset, y ninguno es completo por sí solo. Chinchilla tiene un cuarto de los parámetros de Gopher pero usa un cómputo similar; los modelos *sparse mixture-of-experts* tienen más parámetros por unidad de cómputo que los densos. Por eso los autores recomiendan ver la emergencia como **función de muchas variables correlacionadas**, no de un solo eje.

## 3. Contribución central: el marco de la emergencia

La contribución del paper es **definir y sistematizar** la noción de habilidad emergente. La definición operativa es deliberadamente acotada:

> **Una habilidad es emergente si no está presente en modelos pequeños pero sí está presente en modelos grandes.**

De aquí se desprende la propiedad clave: las habilidades emergentes **no podrían haberse predicho extrapolando una ley de escala** (la mejora consistente y suave) desde los modelos pequeños. Visualizada en una *curva de escala* —eje $x$: escala del modelo; eje $y$: rendimiento—, una habilidad emergente muestra un patrón inconfundible:

$$\text{rendimiento} \approx \text{azar} \quad \text{hasta un umbral crítico de escala, luego} \quad \text{rendimiento} \gg \text{azar}.$$

Este cambio cualitativo es lo que el paper llama, tomándolo de Huberman & Hogg (1987), una **transición de fase**: un cambio dramático de comportamiento global que no se habría anticipado examinando sistemas más pequeños.

El paper es explícito sobre lo que *no* está afirmando. Los autores aclaran que **la escala a la que una habilidad se observa emerger no es una propiedad inmutable de la habilidad**. La emergencia puede ocurrir con menos cómputo o menos parámetros si el modelo se entrena con datos de mayor calidad, o con una arquitectura u objetivo de pre-entrenamiento distintos. Su meta, dicen textualmente, "no es caracterizar ni afirmar que se requiere una escala específica para observar habilidades emergentes, sino discutir ejemplos de comportamiento emergente en trabajos previos". Es una distinción fina pero central: el umbral es empírico y contingente, no una ley fundamental.

Metodológicamente, el paper analiza las curvas de escala usando **FLOPs de entrenamiento** en el eje $x$ (siguiendo a Hoffmann et al., 2022) y, en el Apéndice D, réplicas con **número de parámetros**. Ambos ejes producen curvas de forma similar porque, en la mayoría de las familias de Transformers densos, el cómputo de entrenamiento escala aproximadamente en proporción con los parámetros. El tamaño del dataset no se usa como eje porque muchas familias fijan un número constante de ejemplos de entrenamiento para todos los tamaños de modelo.

## 4. Método y evidencia: tareas y curvas por escala

El paper organiza la evidencia en dos grandes categorías: **habilidades emergentes en el prompting few-shot** (§3) y **estrategias de prompting aumentado** (§4).

### 4.1. Habilidades emergentes en prompting few-shot

En el paradigma de *prompting* popularizado por GPT-3 (Brown et al., 2020), a un modelo pre-entrenado se le da un prompt —una instrucción en lenguaje natural más, opcionalmente, unos pocos ejemplos entrada-salida— y completa la respuesta **sin ningún update de gradiente**. El *few-shot prompting* incluye esos ejemplos como preámbulo. La habilidad de resolver una tarea vía few-shot prompting **es emergente cuando el modelo rinde al azar hasta cierta escala y luego salta a un desempeño muy por encima del azar**.

La Figura 2 del paper reúne ocho ejemplos de este patrón, abarcando cinco familias de modelos:

- **Aritmética modular / multi-dígito (BIG-Bench, Fig. 2A).** Suma y resta de 3 dígitos y multiplicación de 2 dígitos. GPT-3 y LaMDA tienen rendimiento cercano a cero durante varios órdenes de magnitud de cómputo, hasta que el desempeño salta bruscamente a $2 \cdot 10^{22}$ FLOPs (**13B parámetros**) para GPT-3 y $10^{23}$ FLOPs (**68B**) para LaMDA.
- **Transliteración desde el Alfabeto Fonético Internacional** (Fig. 2B), **recuperar una palabra a partir de sus letras desordenadas** (Fig. 2C) y **question-answering en persa** (Fig. 2D): comportamiento emergente a escalas similares.
- **TruthfulQA** (Fig. 2E), benchmark de veracidad curado adversarialmente contra GPT-3. Los modelos GPT-3 no superan el azar ni en su mayor tamaño; los Gopher pequeños tampoco, hasta llegar al mayor de **280B** ($5 \cdot 10^{23}$ FLOPs), donde el desempeño salta a más de 20 puntos sobre el azar.
- **Mapeos conceptuales anclados** (Fig. 2F): solo el mayor modelo GPT-3 supera el azar.
- **MMLU** (*Massive Multitask Language Understanding*, Fig. 2G), 57 tareas de matemática, historia, derecho, etc. GPT-3, Gopher y Chinchilla de $\sim 10^{22}$ FLOPs (~10B) o menores no superan el azar promediando todos los temas; hay que escalar a **70B–280B** ($3\text{–}5 \cdot 10^{23}$ FLOPs) para superar sustancialmente el azar.
- **Word in Context (WiC, Fig. 2H).** Caso notable: GPT-3 y Chinchilla no superan el azar en one-shot ni en su mayor tamaño ($\sim 5 \cdot 10^{23}$ FLOPs); el desempeño solo emerge cuando PaLM se escala a $2.5 \cdot 10^{24}$ FLOPs (**540B**).

### 4.2. Estrategias de prompting aumentado (incluido Chain-of-Thought)

La segunda categoría amplía la definición: **una técnica se considera emergente si no mejora —o incluso perjudica— frente al baseline de no usarla, hasta que se aplica a un modelo suficientemente grande**. La Figura 3 recoge cuatro casos:

- **Chain-of-Thought (razonamiento multi-paso, Fig. 3A).** CoT guía al modelo a producir una **secuencia de pasos intermedios** antes de dar la respuesta final. Sobre problemas matemáticos verbales (GSM8K), **CoT solo supera al prompting estándar sin pasos intermedios al escalar a $10^{23}$ FLOPs (~100B parámetros)**. Por debajo de esa escala, generar cadenas de razonamiento no ayuda o incluso empeora el resultado. Este es el resultado más directamente relevante para la Clase 34.
- **Seguimiento de instrucciones (instruction following, Fig. 3B).** El *instruction finetuning* (Wei et al., 2022a) permite responder a instrucciones de tareas no vistas sin ejemplos. Pero **perjudica** el desempeño en modelos de $7 \cdot 10^{21}$ FLOPs (**8B**) o menores, y solo mejora al escalar a $10^{23}$ FLOPs (~100B). (Nota del propio paper: Sanh et al., 2022, indujeron este comportamiento poco después en un T5 encoder-decoder de 11B, ilustrando que el umbral es movible.)
- **Ejecución de programas / scratchpad (Fig. 3C).** Finetunear el modelo para predecir salidas intermedias ("scratchpad") permite ejecutar cómputos multi-paso como la suma de 8 dígitos, pero **solo ayuda a partir de $\sim 9 \cdot 10^{19}$ FLOPs (40M parámetros)**.
- **Calibración vía P(True) (Fig. 3D).** La técnica de calibración True/False solo supera a los métodos estándar al escalar al mayor modelo, $\sim 3 \cdot 10^{23}$ FLOPs (**52B**, modelo de Anthropic).

La **Tabla 1** del paper consolida todas estas habilidades con su escala de emergencia en FLOPs y parámetros: desde el scratchpad para suma de 8 dígitos (40M, la más baja) hasta WiC (540B, la más alta). El rango de modelos abarca desde el menor LaMDA (2.1M parámetros) hasta el mayor PaLM (540B, $2.5 \cdot 10^{24}$ FLOPs, unas 8 veces el presupuesto de cómputo de GPT-3), según detalla la Tabla 2.

## 5. Ejemplos emergentes y el caso de Chain-of-Thought

Vale la pena detenerse en por qué el CoT es el ejemplo paradigmático para una clase de razonamiento. El paper ofrece una **intuición estructural** en su sección de posibles explicaciones (§5.1): si una tarea de razonamiento multi-paso requiere $l$ pasos de cómputo secuencial, resolverla podría exigir un modelo con una **profundidad de al menos $O(l)$ capas**. Un modelo pequeño, con pocas capas, simplemente no tendría la profundidad de cómputo para encadenar los pasos intermedios; por eso el razonamiento explícito no puede "activarse" hasta que la red es lo bastante profunda y grande. Es una hipótesis, no una demostración —el paper es honesto en que "hay pocas explicaciones convincentes de por qué las habilidades emergen del modo en que lo hacen"—, pero conecta de manera natural la idea de razonamiento con la de escala.

El paper también usa WiC como **ejemplo histórico** de la impredecibilidad. Cuando GPT-3 175B falló en WiC, Brown et al. (2020) atribuyeron el fracaso a la arquitectura de GPT-3 o a su objetivo autorregresivo, y sugirieron entrenar un modelo bidireccional como remedio. Trabajos posteriores mostraron que **bastaba con seguir escalando un modelo decoder-only**: PaLM 540B resolvió WiC sin los cambios arquitectónicos propuestos. La lección es que la escala puede desbloquear habilidades por caminos que no se anticiparon, y que un resultado negativo a una escala dada no es evidencia de imposibilidad.

## 6. Implicancias y discusión

La sección de discusión (§5) es la más importante para pensar el impacto del concepto.

**Predecir capacidades futuras.** Si las habilidades emergen impredeciblemente al escalar, entonces **no conocemos el alcance completo de lo que los LLM pueden hacer**, ni siquiera de los modelos que ya existen. Las tareas que los modelos actuales *no* pueden resolver son candidatas naturales a emerger en modelos futuros: el paper menciona que hay decenas de tareas en BIG-Bench donde incluso el mayor GPT-3 y PaLM rinden al azar. Entender la emergencia —cómo y por qué ocurre— permitiría **anticipar qué capacidades tendrán los modelos futuros**, lo que los autores señalan como una dirección de investigación de primer orden.

**Más allá de la escala (§5.2).** El paper insiste en que la escala **no es el único factor**. Existen 14 tareas de BIG-Bench donde LaMDA 137B y GPT-3 175B rinden al azar pero PaLM 62B las supera, pese a tener *menos* parámetros y FLOPs; los autores atribuyen esto a datos de mejor calidad (más código, más multilingüe) y a diferencias arquitectónicas. También una vez descubierta una habilidad, la investigación puede volverla accesible a modelos más pequeños: instruction following se logró en un encoder-decoder de 11B, e InstructGPT (Ouyang et al., 2022) hizo que un modelo de 1.3B superara a modelos mucho mayores en evaluaciones humanas mediante finetuning y RLHF. **Bajar el umbral de escala** para las habilidades emergentes es, dicen, cada vez más importante para democratizar la investigación.

**Riesgos emergentes (§5.4).** Del mismo modo que emergen capacidades, pueden emerger **riesgos**: veracidad, sesgo, toxicidad. Algunos aumentan con la escala (ver el *Inverse Scaling Prize*): en el benchmark BBQ el sesgo puede crecer con la escala en contextos ambiguos; los modelos más grandes memorizan más datos de entrenamiento (riesgo de extracción); TruthfulQA mostró que GPT-3 imitaba más falsedades humanas al crecer. El paper advierte que, dado que el estudio de la emergencia *incentiva* escalar, es crucial ser conscientes de los riesgos que también aumentan con la escala, sean o no técnicamente "emergentes".

**Cambio sociológico (§5.5).** Un tipo distinto de emergencia: la escala desplazó a la comunidad de NLP **desde modelos específicos por tarea hacia modelos "de propósito general"** (GPT-3, Chinchilla, PaLM), capaces de resolver tareas no codificadas explícitamente en el entrenamiento. Cuando un modelo few-shot de propósito general supera el estado del arte de modelos finetuneados específicos (GPT-3 175B en TriviaQA/PiQA; PaLM 540B en razonamiento aritmético; Flamingo 80B en VQA), se materializa ese giro.

## 7. Limitaciones

- **Es un survey, no propone método ni modelo.** Todas las cifras son de terceros; el paper no controla las diferencias entre familias de modelos (datos, arquitectura, objetivo), por lo que atribuir la emergencia limpiamente a "la escala" es difícil.
- **Falta de explicación mecanicista.** Los propios autores reconocen que hay pocas explicaciones convincentes de por qué emergen estas habilidades. Sus análisis de la pérdida de entropía cruzada (Apéndice A) muestran que la log-verosimilitud del target mejora incluso a escalas donde las métricas downstream siguen al azar —lo que sugiere que hay mejora latente— pero **no explican por qué las métricas downstream son emergentes ni permiten predecir la escala de emergencia**.
- **El rol de las métricas de evaluación.** El paper anticipa explícitamente lo que sería la principal crítica posterior. Advierte que usar *exact match* como métrica para secuencias largas "puede disfrazar mejoras incrementales acumuladas como emergencia", y que métricas sin crédito parcial son "en el mejor de los casos una explicación incompleta". No obstante, argumenta que esto no explica todo, porque la emergencia también se observa en tareas de clasificación (Fig. 2D–H) y porque el salto en la exactitud de la respuesta final no explica por qué la *calidad de los pasos intermedios* emerge.
- **El umbral no es intrínseco.** Como ya se dijo, la escala de emergencia depende de datos, arquitectura y objetivo, y puede bajar con nuevas técnicas. El paper no ofrece una teoría que fije el umbral.

**Debate posterior (contexto externo al paper).** Conviene marcar con claridad que esto **no forma parte de Wei et al. (2022)**. Con posterioridad, Schaeffer, Miranda & Koyejo (*Are Emergent Abilities of Large Language Models a Mirage?*, NeurIPS 2023) argumentaron que muchas curvas de emergencia son un **artefacto de la métrica elegida**: métricas discontinuas o no lineales (como exact match, que exige acertar todos los tokens) producen saltos aparentes, mientras que métricas continuas y suaves (por ejemplo, error por token o log-verosimilitud) sobre las mismas tareas revelan mejoras graduales y predecibles. Es decir, la capacidad subyacente crecería suavemente y sería la métrica la que "fabrica" la transición de fase. El debate no está cerrado: la emergencia sigue siendo empíricamente relevante para métricas de interés práctico (a un usuario le importa si el modelo *acierta la respuesta*, no la log-verosimilitud), y trabajos posteriores discuten en qué casos el fenómeno es real versus métrico. Es notable que el propio Wei et al. ya hubiera señalado el problema de las métricas en §5.1, aunque llegando a una conclusión más matizada.

## 8. Conexión con la Clase 34 (Razonamiento): por qué el CoT requiere escala

La Clase 34 del profesor Sebastián Amenábar aborda el razonamiento en LLM, y en su slide 25 sitúa a los comportamientos emergentes —el In-Context Learning entre ellos— como base del paradigma. Este paper es el fundamento de esa afirmación y, sobre todo, la explicación de **por qué el razonamiento por Chain-of-Thought no es gratis**.

El mensaje operativo para la clase es directo: **CoT es una habilidad emergente, no una técnica universal**. Según la Figura 3A del paper, guiar al modelo a producir pasos de razonamiento intermedios **solo supera al prompting estándar a partir de ~$10^{23}$ FLOPs (~100B parámetros)**. Por debajo de ese umbral, pedirle a un modelo pequeño que "razone paso a paso" no ayuda o incluso empeora el resultado. Esto tiene una consecuencia práctica que suele sorprender: no se puede tomar un modelo pequeño, agregarle un prompt de CoT y esperar capacidades de razonamiento; la técnica **necesita un sustrato de escala** para activarse.

La intuición estructural del paper cierra el argumento para una clase de razonamiento: un problema de $l$ pasos de cómputo secuencial requiere plausiblemente un modelo de profundidad $O(l)$. El razonamiento multi-paso es, en este sentido, la habilidad emergente *por excelencia* —la que más nítidamente ilustra que "más es diferente"—. Un estudiante debería llevarse tres ideas:

1. **Las scaling laws predicen la pérdida, no las capacidades.** El rendimiento en tareas de razonamiento puede seguir en el azar mientras la pérdida de pre-entrenamiento ya mejora suavemente. La capacidad aparece de golpe.
2. **CoT es emergente:** funciona a partir de cierta escala y no antes. El razonamiento explícito y la escala están acoplados.
3. **El umbral es empírico y movible:** mejores datos, arquitecturas u objetivos, y técnicas posteriores (instruction tuning, RLHF, destilación de razonamiento), pueden bajar ese umbral. La escala es una condición típicamente necesaria, no mágicamente suficiente ni eternamente fija.

**Enlaces conceptuales sugeridos:**

- Clase: Clase 34 — Razonamiento (Amenábar); slide 25 sobre comportamientos emergentes e In-Context Learning.
- Paper complementario del razonamiento: Wei et al. (2022b), *Chain-of-Thought Prompting* —referenciado aquí en la Figura 3A—, que desarrolla la técnica cuya emergencia este paper documenta.
- Fundamento transversal: in-context learning y el paradigma de prompting de GPT-3 (Brown et al., 2020).

## 9. Nota final: relevancia para salud

Para aplicaciones clínicas, la lección práctica de este paper es una advertencia contra el "atajo barato". La tentación de desplegar un modelo pequeño y económico para tareas médicas complejas —conciliación de historia clínica, razonamiento diagnóstico multi-paso, interpretación de guías terapéuticas encadenadas— choca con un hecho empírico: **las capacidades de razonamiento son emergentes y no aparecen por debajo de cierta escala**. Un modelo pequeño puede rendir de forma aceptable en clasificación superficial o extracción de entidades, pero *no* heredará automáticamente la habilidad de encadenar pasos de razonamiento clínico solo porque se le añada un prompt de Chain-of-Thought; esa habilidad, según la evidencia del paper, requiere escala. En un dominio donde un error de razonamiento se traduce en riesgo para el paciente, la implicancia es doble: primero, **la escala importa** y no debe subestimarse al elegir el modelo para una tarea clínica que exige razonamiento genuino; segundo, la propia impredecibilidad de la emergencia obliga a **evaluar empíricamente cada capacidad en la tarea concreta** —incluidos los riesgos emergentes de sesgo, toxicidad y confabulación que también crecen con la escala— en lugar de asumirla por extrapolación desde modelos menores.
