# A Generalist Agent (Gato) — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *A Generalist Agent*.
- **Autores:** Scott Reed, Konrad Żołna, Emilio Parisotto, Sergio Gómez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Giménez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar y Nando de Freitas — todos afiliados a **DeepMind**.
- **Venue:** *Transactions on Machine Learning Research* (TMLR), noviembre de 2022. Circuló primero como **preprint arXiv:2205.06175** (mayo de 2022). Revisado abiertamente en OpenReview.
- **Nombre del agente:** **Gato** (del inglés/latín "gato"; el logo del paper es un gato estilizado con la letra "G").

Gato es un **único agente generalista**: una sola red neuronal con **el mismo conjunto de pesos** que —según cómo esté configurado su contexto— juega Atari, subtitula imágenes, conversa, apila bloques con un brazo robótico real, navega en entornos 3D simulados y sigue instrucciones. La tesis central del paper es tomar el enfoque que funcionó en modelado de lenguaje a gran escala (transformers autorregresivos entrenados sobre secuencias de tokens) y **extenderlo más allá del texto**, hacia una política multimodal, multitarea y multi-encarnación (*multi-embodiment*). Gato fue entrenado sobre **604 tareas distintas** con modalidades, observaciones y especificaciones de acción muy diferentes, y opera con apenas **1.2 mil millones de parámetros** (1.2B) —un tamaño deliberadamente moderado, elegido para permitir control en tiempo real de un robot físico.

Para la **Clase 33 (Aprendizaje por Imitación y Aprendizaje Reforzado Inverso, prof. Rodrigo Toro Icarte)** Gato es la evidencia empírica citada (slide 33) de que **"el aprendizaje por imitación funciona bastante bien... aunque tampoco excelente"**. Y esa doble afirmación es exactamente lo que el paper demuestra: Gato es **behavioral cloning (clonación de comportamiento) a escala masiva**. Aprende a *copiar* trayectorias de expertos serializadas como secuencias de tokens, mediante puro entrenamiento supervisado, sin usar la recompensa en línea para mejorar. Funciona sorprendentemente bien en cientos de tareas, pero —como veremos— **no supera a sus maestros** (los especialistas que generaron sus datos), lo que ilustra el techo estructural de la imitación pura.

## 2. Contexto: por qué la Clase 33 cita a Gato

La Clase 33 contrasta dos familias de métodos para obtener políticas de control:

1. **Aprendizaje reforzado (RL):** el agente interactúa con el entorno, recibe una recompensa escalar $r_t$ y **optimiza esa recompensa** por ensayo y error. Puede, en principio, descubrir comportamientos que ningún humano le mostró.
2. **Aprendizaje por imitación (imitation learning):** el agente **no optimiza recompensa**; aprende de un conjunto de demostraciones de un experto $\mathcal{D} = \{(s_i, a_i)\}$, tratando el problema como aprendizaje supervisado —predecir la acción experta dado el estado. Su forma más simple es la **clonación de comportamiento (behavioral cloning, BC)**.

Gato encaja de lleno en la segunda familia, y es citado precisamente porque **lleva la imitación a su máxima expresión de escala y generalidad**. El paper lo dice sin ambigüedad: *"Gato is a data-driven approach, as it is derived from imitation learning"* (Sección 8.1). Su régimen de entrenamiento es **puramente supervisado, offline**; los autores subrayan que "for simplicity Gato was trained offline in a purely supervised manner" —aunque, en principio, nada impide entrenarlo con RL offline u online.

La lección que la clase extrae es doble y matizada:

- **"Funciona bastante bien":** con una sola red de 1.2B parámetros, Gato logra rendimiento por sobre el 50 % del experto en **más de 450 de las 604 tareas**, incluyendo dominios tan dispares como Atari, subtitulado y manipulación robótica real. La imitación a escala produce un generalista genuinamente competente.
- **"Tampoco excelente" / "el agente no logra superar a su maestro":** los agentes de RL en línea que generaron los datos siguen **superando a Gato** en Atari; un especialista del mismo dominio lo supera; y Gato nunca excede sistemáticamente al experto que lo enseñó. Este es el **techo del maestro** intrínseco al BC: si aprendes a copiar, en el mejor de los casos igualas la fuente.

## 3. Contribución central

La contribución de Gato no es un algoritmo nuevo de aprendizaje, sino una **demostración de existencia**: probar la hipótesis de que **un solo transformer, con los mismos pesos, entrenado como modelo de secuencias, puede ser generalmente competente en cientos de tareas heterogéneas** —texto, visión, control continuo, control discreto, robótica real— sin sesgos inductivos hechos a mano para cada dominio.

Tres ideas la sostienen:

1. **Tokenización unificada.** Todo —texto, imágenes, propiocepción, torques de articulaciones, botones— se **serializa en una única secuencia plana de tokens**. Una vez en ese formato común, cualquier dato "puede ser ingerido por el modelo de secuencias", lo que aumenta enormemente la cantidad y diversidad de datos de entrenamiento.
2. **Un solo modelo, los mismos pesos.** El paper insiste en una distinción crucial (Sección 6): no se trata de "una misma arquitectura con pesos distintos por tarea" (lo habitual en RL multitarea), sino de **una sola red con los mismos pesos para todas las tareas**. Es Gato quien decide, según su contexto, si emitir texto, torques, o pulsaciones de botón.
3. **Entrenamiento como modelado de secuencias = imitación masiva.** El objetivo es el mismo que el de un modelo de lenguaje: predecir el siguiente token. Aplicado a trayectorias de expertos, esto **es** behavioral cloning, escalado a 1.5 billones (trillion, escala corta) de tokens de control.

El paper también apela a la "amarga lección" de Sutton (2019): históricamente, los modelos genéricos que aprovechan mejor la computación terminan superando a los enfoques especializados con sesgos de dominio. Gato es una apuesta explícita por esa dirección en el terreno del control.

## 4. Método

### 4.1 Tokenización multimodal

El principio de diseño es "entrenar sobre la mayor variedad posible de datos relevantes". Para procesar modalidades tan distintas, todo se convierte a enteros según un esquema fijo:

- **Texto:** codificado con **SentencePiece** (Kudo & Richardson, 2018), con 32 000 subpalabras, en el rango entero $[0, 32000)$.
- **Imágenes:** divididas en **parches no solapados de $16 \times 16$** en orden de barrido (*raster*), como en ViT (Dosovitskiy et al., 2020). Cada píxel del parche se normaliza a $[-1, 1]$ y se divide por la raíz cuadrada del tamaño de parche ($\sqrt{16} = 4$).
- **Valores discretos** (p. ej. botones de Atari): aplanados en secuencias de enteros en orden *row-major*, en el rango $[0, 1024)$.
- **Valores continuos** (propiocepción, torques): aplanados, **codificados con mu-law** al rango $[-1, 1]$, discretizados en **1024 contenedores (bins) uniformes**, y desplazados al rango $[32000, 33024)$.

Luego se impone un **orden canónico de la secuencia**: tokens de texto en el orden del texto crudo; parches de imagen en orden raster; tensores en orden row-major; estructuras anidadas en orden lexicográfico por clave; y —lo esencial para control— cada **paso temporal del agente** se representa como *tokens de observación, seguidos de un separador, seguidos de tokens de acción*; y cada **episodio** como sus pasos temporales en orden cronológico. Es decir, una trayectoria $(o_1, a_1, o_2, a_2, \dots)$ se aplana en una secuencia lineal que el transformer lee de izquierda a derecha.

### 4.2 Embedding y arquitectura

Sobre los tokens se aplica una función de embedding parametrizada $f(\cdot; \theta_e)$ que opera distinto según la modalidad:

- Tokens de texto y de valores discretos/continuos se embeben mediante una **tabla de búsqueda (lookup)** a un espacio vectorial aprendido, más una codificación de posición aprendible según su posición *dentro del paso temporal*.
- Tokens de parches de imagen se embeben con **un bloque ResNet** por parche, más una codificación de posición aprendible dentro de la imagen.

El modelo de secuencias es un **transformer decoder-only de 1.2B parámetros**: 24 capas, dimensión de embedding 2048, y capa feedforward post-atención de tamaño oculto 8196. Los autores eligieron un transformer estándar "por simplicidad y escalabilidad" —el mismo caballo de batalla de los LLM.

### 4.3 Entrenamiento supervisado por secuencias

Gato modela la secuencia $s_{1:L}$ con la regla de la cadena de probabilidad, exactamente como un modelo de lenguaje:

$$\log p_\theta(s_1, \dots, s_L) = \sum_{l=1}^{L} \log p_\theta(s_l \mid s_1, \dots, s_{l-1}).$$

La clave que convierte esto en **imitación** (y no en simple modelado generativo de todo) es la **función de enmascaramiento** $m(b, l)$. La pérdida solo se aplica sobre tokens que son **texto o la acción registrada de un agente**:

$$\mathcal{L}(\theta, B) = - \sum_{b=1}^{|B|} \sum_{l=1}^{L} m(b, l)\, \log p_\theta\!\left(s_l^{(b)} \mid s_1^{(b)}, \dots, s_{l-1}^{(b)}\right).$$

Es decir: **Gato aprende a predecir la acción experta dada la historia de observaciones y acciones**. Las observaciones de imagen y las observaciones no textuales del agente **no se predicen** (su contribución a la pérdida se enmascara). Esto es behavioral cloning en su forma más pura, envuelto en la maquinaria de un modelo de secuencias: dado el contexto, "¿qué habría hecho el experto ahora?". No hay recompensa en la función objetivo, no hay bootstrapping de Bellman, no hay exploración. Solo copia supervisada de trayectorias.

El entrenamiento corre en un *slice* de **TPU v3 de $16 \times 16$** durante **1 millón de pasos**, con *batch* de 512 y longitud de secuencia $L = 1024$, tomando unos **4 días**. Como los episodios exceden fácilmente el contexto, se muestrean aleatoriamente subsecuencias de $L$ tokens. Cada *batch* mezcla dominios aproximadamente de forma uniforme, con algún sobreponderado manual de los datasets grandes y de alta calidad.

### 4.4 Prompting por demostración

Un problema de la imitación con datos tan heterogéneos es que **tareas distintas pueden compartir la misma encarnación y especificación de observaciones/acciones**, y el modelo necesita desambiguar cuál está resolviendo. En lugar de un identificador *one-hot* de tarea, Gato usa **condicionamiento por prompt**, inspirado en el few-shot learning de los LLM (Brown et al., 2020): durante el entrenamiento, en el **25 %** de las secuencias de cada *batch* se antepone un **prompt** proveniente de un episodio generado por el *mismo agente fuente en la misma tarea*. La mitad de esos prompts vienen del final del episodio (funcionando como *goal conditioning*) y la otra mitad se muestrean uniformemente.

En evaluación, se le da a Gato una **demostración exitosa** de la tarea deseada como prompt (por defecto, los primeros 1024 tokens), y a partir de allí infiere la tarea a partir de las observaciones y acciones del prompt. Al desplegarse como política (Figura 3), Gato consume observaciones tokenizadas, separadores y acciones previas, y **muestrea la siguiente acción token a token de forma autorregresiva**; una vez completo el vector de acción, se decodifica invirtiendo la tokenización y se envía al entorno, que devuelve una nueva observación, y el ciclo se repite. En despliegue se encontró beneficioso usar la memoria de **Transformer-XL** (Dai et al., 2019), aunque no se usó en entrenamiento.

### 4.5 Datos

Gato se entrena sobre una mezcla de datasets de **experiencia de agentes** (simulados y reales) más datasets de **visión y lenguaje**. Un detalle metodológico decisivo para la lectura de imitación: los datos de control provienen de **agentes especialistas de RL, de nivel SoTA o cercano**, entrenados en cada entorno; se registra un subconjunto de la experiencia (estados, acciones y recompensas) *mientras el agente de RL entrena*. Y se filtra a **episodios con retorno de al menos 80 % del retorno experto** de la tarea. Es decir, **Gato imita a políticas de RL ya entrenadas** —el RL hizo el trabajo duro de descubrir buenos comportamientos; Gato solo los clona.

Los dominios de control incluyen DM Lab, ALE Atari (y Atari Extended), Sokoban, BabyAI, DM Control Suite (incluida su variante en píxeles), Meta-World, Procgen, Modular RL, DM Manipulation Playground, Playroom y el **RGB Stacking** (simulado y robot real). En total, **596 tareas de control** y aproximadamente **1.5 billones** (escala corta, $1.5 \times 10^{12}$) de tokens, que constituyen el 85.3 % del peso de muestreo. Los datos de visión-lenguaje (MassiveText, M3W, ALIGN, MS-COCO Captions, Conceptual Captions, LTIP, OKVQA, VQAv2) aportan el 14.7 % restante. El total de tareas reportado como distintas es **604**.

## 5. Resultados

### 5.1 Amplitud: tareas por encima del umbral experto

El resultado emblemático (Figura 5): con **un único modelo preentrenado y un solo conjunto de pesos**, Gato rinde por sobre el **50 % del puntaje experto en más de 450 de las 604 tareas**. Se reporta el desempeño como porcentaje, donde 100 % es el experto por tarea y 0 % una política aleatoria; cada tarea se evalúa con 50 despliegues promediados.

Por dominio:

- **Atari (ALE):** Gato alcanza el puntaje humano promedio (o mejor) en **23 juegos**, y más del doble del puntaje humano en **11**. Sin embargo, los agentes de RL en línea que generaron los datos **siguen superando a Gato**.
- **BabyAI:** sobre 80 % del puntaje experto en casi todos los niveles; en el más difícil (*BossLevel*) alcanza 75 % —comparable a baselines dedicados (77 % y 90 %) que entrenaron *solo* esa tarea con un millón de demostraciones.
- **Meta-World:** más de 50 % en 44 de 45 tareas, más de 80 % en 35, y más de 90 % en 3.
- **DM Control Suite:** mejor que 50 % del experto en 21 de 30 tareas desde estado, y más de 80 % en 18.

### 5.2 Robótica real

En el benchmark **RGB Stacking** con un brazo robótico Sawyer real, Gato —además de hacer cientos de otras tareas— apila de forma **competitiva con el baseline publicado**. En *Skill Generalization* (apilar formas nunca vistas), su tasa de éxito promedio (50.2 %) iguala al baseline BC-IMP de tarea única (49 %). En *Skill Mastery* logra un promedio de 75.6 % frente a 74.6 % del BC-IMP. Esto es notable: **un solo modelo generalista alcanza la performance de un especialista de clonación de comportamiento entrenado solo para apilar.**

### 5.3 Escala y transferencia

- **Leyes de escala (Figura 8):** con tres tamaños (79M, 364M y 1.18B), a igual número de tokens procesados hay una **mejora consistente al aumentar la capacidad**. La performance in-distribution crece monótonamente con el tamaño del modelo —la misma dinámica de los LLM.
- **Fine-tuning fuera de distribución (Sección 5.2):** sobre cuatro tareas retenidas por completo (`cartpole.swingup`, `assembly-v2`, `order_of_apples_forage_simple`, `boxing`), preentrenar sobre *todos* los datos suele dar la mejor adaptación tras fine-tuning, seguido de preentrenar solo en el mismo dominio. Hay **transferencia positiva** en la mayoría de casos, aunque en `boxing` (Atari) el modelo inicializado al azar funciona mejor: la transferencia en Atari es notoriamente difícil por lo visualmente distintos que son los juegos.
- **Adaptación robótica con pocos datos (Sección 5.3):** en RGB Stacking, Gato **recupera la performance del experto con solo 10 episodios** de fine-tuning y la excede con 100–1000. En una tarea perceptual nueva ("apilar azul sobre verde"), Gato fine-tuneado con 500 demostraciones logra **60 % de éxito**, frente a **0.5 %** de un BC entrenado desde cero —evidencia fuerte de que el preentrenamiento generalista provee representaciones reutilizables.

### 5.4 Especialistas superan al generalista

Cuando los autores entrenan un **especialista de un solo dominio** con la misma arquitectura, este **supera a Gato**: el especialista de Atari (1.18B parámetros, entrenado solo en las 51 tareas de Atari) logra performance sobrehumana en **44 juegos**, frente a los 23 del generalista. Esto sugiere que escalar Gato mejoraría su desempeño; los autores restringieron su tamaño a propósito para poder correrlo en tiempo real en el robot real.

## 6. Limitaciones

Las limitaciones de Gato son, casi punto por punto, la traducción técnica de "no funciona excelente" y "el agente no logra superar a su maestro":

- **El techo del maestro (behavioral cloning puro).** Gato imita trayectorias filtradas al 80 % del retorno experto. No usa la recompensa para mejorar más allá de lo demostrado, de modo que su rendimiento **está acotado por la calidad de los datos**: los agentes de RL en línea que generaron los datos lo superan, y no hay mecanismo, en el entrenamiento reportado, para que Gato los exceda. Los autores mismos señalan que esto "podría superarse añadiendo capacidad o usando RL offline en lugar de supervisión pura".
- **No supera a los especialistas.** Un modelo dedicado a un solo dominio supera al generalista. La generalidad tiene un costo de competencia por tarea.
- **Contexto corto.** La ventana de contexto es de **1024 tokens**, que para entornos con imágenes (más de cien tokens por observación) equivale a **muy pocos pasos temporales**. Esto limita el prompting por demostración: experimentos preliminares de *in-context learning* en entornos nuevos **no mostraron mejora significativa** frente a la evaluación sin prompt, y por eso la adaptación fuera de distribución se hace por **fine-tuning de parámetros**, no por prompting puro. La causa es el escalamiento cuadrático de la auto-atención.
- **Recolección de datos de RL.** No existe un dataset de control a escala web (a diferencia de texto e imágenes); obtener datos de control diversos es un problema de investigación en sí mismo.
- **Diálogo y captioning rudimentarios.** El chat de Gato responde de forma "a menudo superficial o factualmente incorrecta" —limitación que los autores atribuyen a la escala moderada.
- **Sesgos de autodelusión causal.** Generar acciones autorregresivamente puede inducir *self-delusion* cuando hay variables confusoras; el prompting con una demostración exitosa mitiga esto parcialmente al "tapar" (screen off) los confusores.

## 7. Conexión con la Clase 33: imitación vs. RL

Gato es el ejemplo canónico de **imitación a escala** en el marco conceptual de la Clase 33, y su lugar en el debate imitación-vs-RL es instructivo:

- **Es imitación, no RL.** Aunque sus datos provienen de agentes de RL y contienen recompensas, Gato **no optimiza la recompensa**. Su función de pérdida (ecuación de la Sección 4.3) solo penaliza errores en la predicción de la acción experta. La recompensa se usa apenas como **filtro de calidad** de los datos (el umbral del 80 %), no como señal de aprendizaje. Es behavioral cloning de manual, con la diferencia de escala y multimodalidad.
- **Ilustra la fortaleza del BC.** El BC es simple, estable y *sample-efficient* cuando hay buenas demostraciones: no requiere interacción con el entorno durante el entrenamiento, no sufre la *tríada mortal* del RL profundo, y escala como un problema supervisado. Gato demuestra que, con datos suficientes, esta receta produce un generalista genuino.
- **Ilustra también su debilidad estructural.** El BC hereda el techo del maestro y —clásicamente (Ross et al., DAgger, 2011, del propio material de la clase)— sufre **cambio de distribución (distribution shift)**: como el agente solo vio estados visitados por el experto, un pequeño error lo lleva a estados nunca demostrados, donde no sabe recuperarse (errores que se componen, *compounding errors*). El contexto corto y la ausencia de corrección en línea agravan esto. La Clase 33 propone remedios que Gato no aplica: **DAgger** (consultar al experto en los estados que el aprendiz visita), o dar el salto a **aprendizaje reforzado inverso (IRL)** / **GAIL**, que *infieren la función de recompensa* subyacente al experto en lugar de copiar acciones, permitiendo en principio generalizar mejor e incluso superar al maestro. Los autores de Gato mismos apuntan a esa dirección: mencionan que nada impide entrenarlo con RL offline u online, o usar *counterfactual teaching* con retroalimentación experta instantánea (una idea emparentada con DAgger).

Así, Gato encarna la frase de la clase con precisión quirúrgica: **la imitación funciona bastante bien** (450+ tareas por sobre el 50 % del experto, robot real competitivo, transferencia con pocos datos), **pero no excelente** (no supera a sus maestros ni a los especialistas), porque **copiar tiene por techo la calidad de lo copiado**.

## 8. Relación con Decision Transformer y la tokenización estilo LLM

El paper sitúa explícitamente a Gato junto a los **Decision Transformers** (Chen et al., 2021) y el **Trajectory Transformer** (Janner et al., 2021), que ya habían mostrado la utilidad de arquitecturas tipo LM para control. Gato comparte con ellos la idea de **tratar el control como modelado de secuencias**, pero difiere en varios ejes de diseño elegidos para soportar **multimodalidad, multi-encarnación, gran escala y despliegue de propósito general**:

- **Un solo modelo con los mismos pesos para todas las tareas y dominios**, frente a los Decision Transformers que típicamente usan una misma arquitectura con **pesos distintos por tarea/dominio**. Esta es la distinción que el paper enfatiza en la Sección 6.
- A diferencia del Decision Transformer, que condiciona sobre el **retorno deseado** (*return-to-go*) para modular el comportamiento, Gato condiciona sobre una **demostración (prompt)** y no expone la recompensa como entrada de control en los resultados presentados —coherente con su carácter de imitación pura.

En cuanto al linaje de LLM, Gato es la **extensión directa del paradigma GPT** al control: se inspira en GPT-3, Gopher, Flamingo y PaLM, y usa exactamente la misma maquinaria (transformer decoder-only, predicción autorregresiva del siguiente token, SentencePiece, parches ViT, prompting few-shot). Su aporte conceptual es mostrar que **si todo se puede tokenizar en una secuencia plana, entonces todo puede modelarse con un LLM** —incluyendo torques de un brazo robótico. Es un puente entre el mundo de los foundation models de lenguaje y el mundo del control encarnado, y un precursor directo de los *Vision-Language-Action models* (VLA) posteriores.

---

**Nota para el lector experto en salud y MDM (FHIR).** La arquitectura de Gato tiene una lectura sugerente para *master data management* y *data stewardship* clínico. Imaginemos serializar en una única secuencia de tokens las decisiones heterogéneas de un data steward experto: pares de recursos FHIR candidatos a un merge, sus atributos (nombres, identificadores, fechas, direcciones), la evidencia de coincidencia, y la **acción tomada** (fusionar, marcar como no-match, escalar a revisión). Un modelo de secuencias entrenado por imitación sobre ese corpus sería, en esencia, un **asistente generalista de curaduría de datos** que aprende a reproducir el criterio experto a través de contextos muy distintos —igual que Gato reproduce políticas expertas a través de dominios. Y la moraleja de la Clase 33 se traslada intacta: **su techo es la calidad del maestro**. Si el steward humano comete sesgos sistemáticos (p. ej. sobre-fusionar registros de pacientes homónimos, o infra-detectar duplicados en poblaciones con nombres poco frecuentes), la clonación de comportamiento los heredará y amplificará, sin señal correctiva alguna —porque, como Gato, no optimiza una métrica de calidad del resultado, solo copia decisiones. Superar ese techo exigiría lo que la clase propone más allá de la imitación pura: retroalimentación en los estados que el modelo realmente visita (estilo DAgger, revisión humana sobre las fusiones que *el modelo* propone) o inferir la "función de recompensa" latente de una buena resolución de identidad (estilo IRL) en lugar de imitar clics. Gato es, para este lector, tanto la promesa —un único modelo de secuencias que absorbe decisiones expertas multimodales— como la advertencia sobre sus límites.
