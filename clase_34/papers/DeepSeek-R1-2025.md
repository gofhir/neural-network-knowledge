# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*.
- **Autores:** DeepSeek-AI (equipo grande; contribuyentes centrales: Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Zhihong Shao, Zhibin Gou y otros). Correspondencia: `research@deepseek.com`.
- **Publicación:** preprint `arXiv:2501.12948` (versión v2, enero 2026). Modelos liberados públicamente en `https://huggingface.co/deepseek-ai`.
- **Linaje técnico:** construido sobre **DeepSeek-V3-Base**, un modelo *Mixture-of-Experts* (MoE) con 671 mil millones de parámetros totales y 37 mil millones activados por token, preentrenado sobre 14,8 billones de tokens. El algoritmo de RL, **GRPO**, proviene de Shao et al., 2024 (DeepSeekMath).

El trabajo demuestra una tesis fuerte: las capacidades de razonamiento de un LLM pueden **incentivarse mediante aprendizaje reforzado puro**, sin necesidad de trazas de razonamiento anotadas por humanos. Los autores presentan dos modelos. **DeepSeek-R1-Zero** se entrena aplicando RL directamente sobre el modelo base, **omitiendo por completo la fase convencional de fine-tuning supervisado (SFT)** previa, con recompensas basadas exclusivamente en reglas (corrección del resultado + formato). A partir de este entrenamiento emergen de forma autónoma comportamientos sofisticados —autoverificación, reflexión, exploración de estrategias alternativas y cadenas de pensamiento largas— incluyendo un fenómeno que los autores bautizan como *aha moment*. **DeepSeek-R1** corrige los problemas de legibilidad y mezcla de idiomas de R1-Zero mediante un pipeline multietapa (arranque en frío + RL + muestreo por rechazo + SFT + RL final), alcanzando resultados a la par de **OpenAI-o1-1217** en tareas verificables. Finalmente, el conocimiento de R1 se **destila** a modelos pequeños (1,5B a 70B) que superan a sus contrapartes instruidas convencionales.

Para la **Clase 34 (Razonamiento)** este paper es el ejemplo canónico del "momento o1 & R1": generar muchas respuestas por problema y usar RL para premiar las trazas que llegan al resultado correcto, sobre problemas verificables. R1 es la instancia abierta y documentada de ese paradigma, y hace explícito tanto el mecanismo (GRPO + reward por reglas) como su límite (solo aplica donde existe un verificador confiable).

## 2. Contexto: cómputo en inferencia y el momento o1

El razonamiento —resolución de problemas matemáticos, deducción lógica, programación— es una capacidad emergente en LLMs a suficiente escala (Kaplan et al., 2020; Wei et al., 2022a). Una línea complementaria mostró que el *chain-of-thought* (CoT), inducido con ejemplos *few-shot* o con instrucciones minimalistas como "*Let's think step by step*" (Kojima et al., 2022; Wei et al., 2022b), mejora sustancialmente el desempeño al forzar pasos intermedios. La idea de fondo es que asignar **más cómputo en tiempo de inferencia** —más tokens de "pensamiento" antes de responder— eleva la precisión en tareas complejas.

El problema es cómo enseñar a un modelo a razonar bien. El enfoque tradicional post-entrenamiento (Ouyang et al., 2022) usa SFT sobre trazas de razonamiento anotadas por humanos, seguido de RL. Los autores identifican dos limitaciones de depender de demostraciones humanas:

1. **Escalabilidad y sesgo cognitivo.** Anotar trazas de razonamiento de alta calidad es caro y lento, e introduce sesgos humanos.
2. **Techo humano.** Al restringir al modelo a replicar procesos de pensamiento humanos, su desempeño queda **acotado por los ejemplos provistos**, impidiendo explorar caminos de razonamiento superiores no humanos.

El "momento o1" (OpenAI, 2024) mostró públicamente que modelos entrenados para producir CoT largas y auto-correctoras alcanzaban un salto cualitativo en matemática y código. DeepSeek-R1 es la respuesta abierta a ese hito, y su aporte conceptual es demostrar que **no se necesita ninguna traza de razonamiento humana**: basta con preguntas difíciles, un verificador confiable y suficiente cómputo de RL.

## 3. Contribución central

El paper aporta dos artefactos y una lección de método:

- **DeepSeek-R1-Zero.** RL puro sobre el modelo base, sin SFT previo, con recompensas por reglas. Prueba que un checkpoint preentrenado **ya posee** potencial de razonamiento latente que el RL puede desbloquear, y que los comportamientos de razonamiento avanzado **emergen** sin ser enseñados explícitamente.
- **DeepSeek-R1.** Pipeline multietapa que hereda la capacidad de razonamiento de R1-Zero pero la alinea con preferencias humanas (legibilidad, consistencia de idioma, utilidad, inocuidad), logrando resultados de frontera.
- **Destilación.** Las trazas de razonamiento descubiertas por R1 se transfieren a modelos pequeños por SFT, democratizando el acceso a razonamiento fuerte con menor costo energético.

La lección de método, enunciada en la conclusión, es contundente: *"la clave para desbloquear este potencial no reside en la anotación humana a gran escala, sino en proveer preguntas de razonamiento difíciles, un verificador confiable y suficientes recursos de cómputo para el RL"*.

## 4. Método

### 4.1. GRPO: el motor de RL y su diferencia con PPO

DeepSeek-R1-Zero y R1 se entrenan con **Group Relative Policy Optimization (GRPO)** (Shao et al., 2024). GRPO nació para **simplificar y abaratar PPO** (Schulman et al., 2017), el algoritmo estándar en la etapa de RL de los LLMs (Ouyang et al., 2022).

La diferencia clave es **cómo se estima la ventaja**. PPO requiere una **red de valor (critic)** separada —un segundo modelo del tamaño del *policy*— que estima $V(s)$ para calcular la ventaja $A = Q - V$. Entrenar y mantener esa red duplica el consumo de memoria y cómputo. **GRPO elimina la red de valor**: para cada pregunta $q$ muestrea un **grupo** de $G$ salidas $\{o_1, o_2, \dots, o_G\}$ desde la política antigua $\pi_{\theta_{old}}$, y estima la ventaja de cada salida por **normalización dentro del grupo**:

$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}$$

Es decir, la ventaja de una respuesta es simplemente cuántas desviaciones estándar por encima (o por debajo) del promedio del grupo cayó su recompensa. El promedio del grupo cumple el rol de *baseline* que en PPO cumplía el crítico, pero sin costo de un modelo adicional. El objetivo optimizado es:

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Big[q \sim P(Q),\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)\Big]\ \frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i,\ \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right)A_i\right) - \beta\, \mathbb{D}_{KL}\!\left(\pi_\theta \| \pi_{ref}\right)\right)
$$

Se reconocen los ingredientes de PPO: el **cociente de probabilidades** (*importance ratio*) entre política nueva y vieja, el operador **clip** que limita el paso a la región $[1-\varepsilon, 1+\varepsilon]$ para estabilidad, y una **penalización KL** contra una política de referencia $\pi_{ref}$ (con coeficiente $\beta$) que impide que el modelo se aleje demasiado del punto de partida. La divergencia se estima con el estimador no negativo:

$$
\mathbb{D}_{KL}\!\left(\pi_\theta \| \pi_{ref}\right) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log\frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1
$$

En resumen: **GRPO = PPO sin red de valor**, reemplazando la estimación de ventaja por una normalización *relativa al grupo*. Esto encaja perfecto con la lógica de "generar muchas respuestas por problema": el grupo de rollouts que ya se necesita para explorar es también el que provee la baseline estadística.

Hiperparámetros de R1-Zero: tasa de aprendizaje $3\times10^{-6}$, coeficiente KL $0{,}001$, temperatura de muestreo $1$, $G = 16$ salidas por pregunta con longitud máxima de $32\,768$ tokens (ampliada a $65\,536$ tras el paso 8,2k), $32$ preguntas por paso (batch de $512$). El entrenamiento corrió $10\,400$ pasos ($\approx 1{,}6$ épocas), reemplazando la política de referencia cada $400$ pasos.

### 4.2. Recompensas basadas en reglas

Para R1-Zero, la señal de entrenamiento es **puramente basada en reglas**, con dos componentes de igual peso:

$$\text{Reward}_{rule} = \text{Reward}_{acc} + \text{Reward}_{format}$$

- **Recompensa de precisión (accuracy).** Evalúa si la respuesta final es correcta. En matemática, el modelo debe entregar el resultado en un formato específico (por ejemplo, dentro de una caja) para verificarlo con una regla determinista. En programación competitiva, un **compilador** ejecuta la respuesta contra casos de prueba predefinidos, generando *feedback* objetivo.
- **Recompensa de formato.** Obliga al modelo a encapsular su proceso de razonamiento entre las etiquetas `<think>...</think>` y la respuesta entre `<answer>...</answer>`, según una plantilla fija. Esto delimita explícitamente el "pensamiento" y facilita el análisis posterior.

Un punto de diseño deliberado: los autores **se abstienen de usar modelos de recompensa neuronales** (ni basados en resultado ni en proceso) para las tareas de razonamiento, porque observaron que son susceptibles a *reward hacking* durante el RL a gran escala, y porque reentrenarlos es costoso. Solo se restringe la **forma** (estructura de etiquetas), no el **contenido** del razonamiento, para poder observar la progresión natural del modelo sin sesgos.

### 4.3. El pipeline multietapa de DeepSeek-R1

R1-Zero, pese a su fuerza en razonamiento, sufre **mala legibilidad** y **mezcla de idiomas** (combina inglés y chino dentro de una misma cadena, porque DeepSeek-V3-Base está entrenado en ambos), y su RL puro está estrechamente enfocado en tareas de razonamiento, con desempeño limitado en escritura y QA de dominio abierto. DeepSeek-R1 se construye con un pipeline de cuatro fases sobre DeepSeek-V3-Base:

1. **Arranque en frío (Cold-start SFT).** Se recopilan *miles* de ejemplos con un proceso de pensamiento conversacional y alineado a humanos, y se hace SFT para dar al modelo un estilo legible de partida (evitando la degradación caótica del RL puro desde cero).
2. **Primera etapa de RL.** GRPO con recompensas por reglas, más una **recompensa de consistencia de idioma**, definida como la proporción de palabras en el idioma objetivo dentro del CoT:

$$\text{Reward}_{language} = \frac{Num(Words_{target})}{Num(Words)}$$

   Las ablaciones muestran que esta restricción degrada *levemente* el desempeño, pero mejora mucho la legibilidad, un intercambio que alinea con la preferencia humana. Clip ratio $\varepsilon = 10$ (un clip amplio, cuyo rol los autores señalan como crucial: un valor bajo trunca gradientes y degrada, uno muy alto desestabiliza).
3. **Muestreo por rechazo + SFT.** Desde el checkpoint de la primera RL se hace **rejection sampling**: se generan múltiples respuestas por prompt y se retienen solo las correctas (filtrando CoT con idiomas mezclados, párrafos largos y bloques de código ilegibles), usando además un modelo de recompensa generativo (DeepSeek-V3 como juez de la respuesta contra el *ground-truth*). Esto produce **~600k** muestras de razonamiento. Se suman **~200k** muestras **no** de razonamiento (escritura, QA factual, traducción, ingeniería de software) reutilizando el pipeline de DeepSeek-V3, para un total de **~800k** ejemplos con los que se hace SFT. El modelo ahora razona *y* escribe bien.
4. **RL final.** Una segunda etapa de RL con distribución diversa de prompts combina tres señales:

$$\text{Reward} = \text{Reward}_{reasoning} + \text{Reward}_{general} + \text{Reward}_{language}$$

   donde $\text{Reward}_{reasoning} = \text{Reward}_{rule}$ (reglas para matemática/código/lógica) y $\text{Reward}_{general} = \text{Reward}_{reward\_model} + \text{Reward}_{format}$ usa modelos de recompensa de preferencia (utilidad e inocuidad) para los datos generales. Esta etapa (temperatura reducida a $0{,}7$, $1\,700$ pasos, con datos de preferencia solo en los últimos $400$ para evitar *reward hacking*) alinea el modelo con preferencias humanas sin perder razonamiento.

Los checkpoints intermedios se denominan **Dev1, Dev2, Dev3** (Figura 2). La progresión es reveladora: Dev1 (tras cold-start + RL) mejora mucho en seguimiento de instrucciones (IF-Eval, ArenaHard) pero *retrocede* en AIME por el tamaño limitado del cold-start; Dev2 recupera y potencia el razonamiento; Dev3 (tras el SFT de 800k) añade escritura y generación general; el R1 final gana sobre todo en instrucciones y preferencias de usuario (AlpacaEval 2.0 +25%, ArenaHard +17%).

### 4.4. Destilación a modelos pequeños

El razonamiento de R1 se transfiere a modelos abiertos más pequeños por **SFT directo sobre los 800k ejemplos** curados (fine-tuning del modelo base por 2–3 épocas, **sin aplicar RL** a los estudiantes). Las bases usadas son **Qwen2.5-Math-1.5B/7B, Qwen2.5-14B/32B, Llama-3.1-8B y Llama-3.3-70B**. La destilación es, en esencia, "empaquetar" las trazas de razonamiento del maestro grande y enseñárselas por imitación al estudiante pequeño.

## 5. Resultados

### 5.1. DeepSeek-R1-Zero (RL puro)

Sobre **AIME 2024**, el pass@1 promedio de R1-Zero salta de un **15,6% inicial a 77,9%** a lo largo del entrenamiento; con decodificación por auto-consistencia (voto mayoritario, cons@16) llega a **86,7%**, superando el promedio de los competidores humanos de la AIME. En paralelo, la **longitud de respuesta crece de forma sostenida** (de cientos a miles de tokens): el modelo *aprende por sí solo* a pensar más tiempo, sin que se lo indiquen. También logra desempeño notable en competencias de programación y en problemas de biología, física y química de nivel de posgrado. Todo esto **sin una sola traza de razonamiento humana**.

### 5.2. DeepSeek-R1 frente a o1

Comparación con modelos representativos (Tabla 8):

| Benchmark | DeepSeek-V3 | OpenAI o1-mini | OpenAI o1-1217 | **DeepSeek-R1** |
|---|---|---|---|---|
| AIME 2024 (Pass@1) | 39,2 | 63,6 | 79,2 | **79,8** |
| MATH-500 (Pass@1) | 90,2 | 90,0 | 96,4 | **97,3** |
| GPQA Diamond (Pass@1) | 59,1 | 60,0 | 75,7 | 71,5 |
| LiveCodeBench (Pass@1-CoT) | 36,2 | 53,8 | 63,4 | **65,9** |
| Codeforces (percentil) | 58,7 | 93,4 | 96,6 | 96,3 |
| Codeforces (rating) | 1134 | 1820 | 2061 | 2029 |
| MMLU (EM) | 88,5 | 85,2 | 91,8 | 90,8 |

DeepSeek-R1 queda **a la par de o1-1217 en matemática** (lo iguala o supera en AIME 2024 y MATH-500) y en tareas algorítmicas de código (LiveCodeBench y Codeforces, donde los modelos de razonamiento dominan). En la plataforma **Codeforces, R1 supera al 96,3% de los participantes humanos**. En tareas de código orientadas a ingeniería (Aider) o1-1217 aún aventaja a R1, y en GPQA Diamond —donde el humano de referencia es de nivel doctoral con acceso web— el humano y o1 siguen adelante de R1 (71,5). El salto respecto de la base DeepSeek-V3 es enorme en todo lo que exige razonamiento (AIME 39,2 → 79,8).

### 5.3. Destilación

Los modelos destilados (Tabla 15) muestran que el razonamiento se transfiere sorprendentemente bien:

| Modelo | AIME 2024 (pass@1) | MATH-500 | GPQA Diamond | LiveCodeBench | Codeforces (rating) |
|---|---|---|---|---|---|
| GPT-4o-0513 (baseline) | 9,3 | 74,6 | 49,9 | 32,9 | 759 |
| Claude-3.5-Sonnet-1022 | 16,0 | 78,3 | 65,0 | 38,9 | 717 |
| R1-Distill-Qwen-1.5B | 28,9 | 83,9 | 33,8 | 16,9 | 954 |
| R1-Distill-Qwen-7B | 55,5 | 92,8 | 49,1 | 37,6 | 1189 |
| R1-Distill-Qwen-32B | 72,6 | 94,3 | 62,1 | 57,2 | 1691 |
| R1-Distill-Llama-70B | 70,0 | 94,5 | 65,2 | 57,5 | 1633 |

Incluso el diminuto **Qwen-1.5B destilado supera a GPT-4o y a Claude-3.5-Sonnet en los benchmarks matemáticos**, un resultado notable para un modelo de solo 1,5 mil millones de parámetros. El desempeño mejora de forma monótona con el tamaño del estudiante.

Una comparación adicional clave (Tabla 16): entrenar **Qwen2.5-32B con RL puro** (Qwen2.5-32B-Zero) alcanza apenas **47,0** en AIME, comparable a QwQ-32B-Preview (50,0), mientras que **R1-Distill-Qwen-32B llega a 72,6**. La conclusión de los autores: **destilar un modelo potente en uno pequeño rinde mucho mejor** que aplicar RL a gran escala directamente sobre el modelo pequeño, y es más económico. Superar la frontera del conocimiento humano, en cambio, todavía requeriría bases más potentes y RL a mayor escala.

## 6. El "aha moment"

El fenómeno más citado del paper (Tabla 2) es un **momento de introspección emergente** en una versión intermedia de R1-Zero. Resolviendo la ecuación $\sqrt{a - \sqrt{a+x}} = x$, el modelo escribe en medio de su cadena:

> *"Wait, wait. Wait. That's an aha moment I can flag here. Let's reevaluate this step-by-step..."*

El modelo **se detiene, reconoce que debe reconsiderar y reinicia el razonamiento** con un tono antropomórfico, sin haber sido entrenado para hacerlo. Los autores lo describen como un *aha moment* también para ellos: presenciar cómo el RL, sin enseñar el "cómo" sino solo los incentivos correctos, hace que el modelo desarrolle autónomamente estrategias de resolución avanzadas. Cuantitativamente, este momento se manifiesta como un **aumento súbito del uso de la palabra "wait"** durante las reflexiones, y marca un cambio distintivo en los patrones de razonamiento. El mensaje de fondo —"*the power and beauty of reinforcement learning*"— es que comportamientos como la autoverificación y la reflexión **emergen orgánicamente** del proceso de RL cuando se premia solo el resultado correcto. El modelo aprende que "pensar más y verificarse" aumenta su probabilidad de acertar, y la política converge hacia ello por trial and error.

## 7. Limitaciones

Los propios autores enumeran límites tanto de capacidad como del método:

- **Solo dominios verificables.** El pilar del enfoque es la disponibilidad de un **verificador confiable** (comparación con *ground-truth* matemático, ejecución de tests de código). Para tareas como la escritura, donde no existe una regla de corrección objetiva, no se puede usar reward por reglas; y si se sustituye por un modelo de recompensa neuronal, este se vuelve **vulnerable a *reward hacking*** conforme avanza el entrenamiento (el modelo encuentra atajos para engañarlo). Por eso, para tareas sin señal confiable, R1 recurre a datos supervisados anotados por humanos y solo corre RL por cientos de pasos. **Escalar el RL puro a tareas no verificables sigue siendo un problema abierto.**
- **Legibilidad y mezcla de idiomas.** R1-Zero mezcla idiomas y produce CoT poco legibles; R1 lo mitiga con cold-start y la recompensa de consistencia de idioma, pero R1 sigue optimizado solo para chino e inglés y puede mezclar idiomas ante consultas en otras lenguas.
- **Sensibilidad a los prompts.** El *few-shot prompting* degrada consistentemente su desempeño; se recomienda uso *zero-shot* describiendo el problema directamente.
- **Uso de herramientas y salida estructurada.** R1 aún no aprovecha herramientas (buscadores, calculadoras) y su salida estructurada es subóptima.
- **Ingeniería de software.** Por los largos tiempos de evaluación, el RL no se aplicó extensivamente a tareas de software, y R1 no mejora mucho sobre V3 ahí.
- **Sobrepensamiento (*overthinking*).** R1 asigna cómputo dinámicamente según la dificultad, pero a veces "piensa de más" en preguntas simples, con margen para mejorar la eficiencia de tokens.
- **Seguridad.** El razonamiento mejorado puede volver más ejecutables los planes dañinos ante *jailbreaks*; el nivel de seguridad intrínseco es moderado (comparable a GPT-4o), y se eleva al acoplar un sistema de control de riesgos.

## 8. Conexión con la Clase 34 y con RLHF

R1 materializa el "momento o1 & R1" de la Clase 34: **muestrear muchas respuestas por problema y usar RL para dar reward positivo a las trazas de razonamiento que llegan al resultado correcto**, sobre problemas verificables. El grupo de $G$ rollouts de GRPO *es* ese conjunto de respuestas por problema, y la ventaja normalizada al grupo *es* la forma de premiar relativamente las mejores. El *aha moment* es la evidencia concreta de que los comportamientos de razonamiento (reflexión, autoverificación, cadenas largas) emergen del RL sin ser programados.

La diferencia con **RLHF** es central y define el alcance del método. En RLHF (InstructGPT; Ouyang et al., 2022), la recompensa la entrega un **modelo de recompensa neuronal entrenado sobre preferencias humanas** (pares "esta respuesta es mejor que aquella"): la señal es subjetiva, aprendida y susceptible de ser explotada. En R1-Zero, la recompensa es **verificable por reglas**: la respuesta matemática es correcta o no, el código pasa los tests o no. Esa señal es objetiva, barata y no *hackeable*, lo que permite RL a gran escala **sin SFT previo y sin anotación humana de trazas**. R1 combina ambos mundos: reward por reglas para el razonamiento (donde hay verificador) y reward de preferencias humanas (utilidad, inocuidad) solo para los dominios generales donde no existe un *ground-truth* objetivo. La lección transversal: **el RL desbloquea capacidades latentes del modelo base cuando la señal de recompensa es confiable**; el cuello de botella no es el algoritmo sino la existencia de un verificador.

## Nota final: relevancia para salud

La receta de R1 —RL con recompensa verificable por reglas— **no se traslada directamente a la mayoría de los dominios clínicos**, porque estos rara vez tienen un *ground-truth* objetivo y determinista contra el cual verificar automáticamente cada respuesta. Un diagnóstico diferencial, una nota clínica o una recomendación terapéutica no son "correctos o incorrectos" en el sentido en que lo es una respuesta de AIME o la ejecución de un test de código; su calidad depende de juicio experto, contexto y preferencias, es decir, del terreno donde RLHF (con sus riesgos de *reward hacking*) o el juicio humano siguen siendo necesarios. El enfoque sí es prometedor allí donde **existe un verificador confiable**: dosificación con reglas farmacológicas comprobables, validación de códigos de facturación o terminología (CIE-10, SNOMED, LOINC), consistencia de un recurso FHIR contra su perfil, cálculo de scores de riesgo con fórmulas cerradas, o *record linkage* con criterios de coincidencia objetivos. En síntesis, para un servidor FHIR o un sistema de MDM, la parte "verificable por reglas" de un problema es candidata natural a este tipo de RL, pero incentivar razonamiento clínico abierto con reward automático sería riesgoso: sin un verificador robusto, el modelo optimizaría la métrica y no la verdad, exactamente el fallo que los autores advierten para tareas no verificables.
