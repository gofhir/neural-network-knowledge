# Key-Value Memory Networks for Directly Reading Documents — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Key-Value Memory Networks for Directly Reading Documents*.
- **Autores:** Alexander H. Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, Jason Weston (Facebook AI Research, Nueva York; Jesse Dodge afiliado también al Language Technologies Institute de Carnegie Mellon).
- **Venue:** EMNLP 2016 (Conference on Empirical Methods in Natural Language Processing).
- **Año:** 2016. **Preprint:** arXiv:1606.03126v2 (10 oct 2016), [arxiv.org/abs/1606.03126](https://arxiv.org/abs/1606.03126).
- **Datos / código:** el dataset WikiMovies se libera en [fb.ai/babi](http://fb.ai/babi), junto al resto del corpus bAbI de FAIR.

Este paper hace dos cosas a la vez. Primero, propone una **arquitectura nueva** —la *Key-Value Memory Network* (KV-MemNN)— que generaliza la End-To-End Memory Network de Sukhbaatar et al. (2015) separando cada slot de memoria en un par **(clave, valor)**: la clave se usa para *direccionar* (atender) y el valor para *devolver* el contenido. Segundo, construye una **herramienta de análisis** —el dataset **WikiMovies**— diseñada explícitamente para medir una sola variable: qué tanto cae el rendimiento de un sistema de QA cuando la fuente de conocimiento pasa de una base de conocimiento (KB) perfecta a documentos de texto crudos. WikiMovies contiene ~100k preguntas sobre películas, respondibles desde tres fuentes que codifican el *mismo* conocimiento: una KB anotada por humanos, una KB obtenida por extracción de información (IE) automática, y los documentos de Wikipedia directos.

La tesis es nítida: leer documentos directamente para responder preguntas es un problema sin resolver, y por eso el QA se ha desviado hacia usar KBs, que funcionan bien pero tienen límites intrínsecos —son incompletas, tienen esquemas fijos que no soportan toda variedad de respuestas, y son escasas comparadas con la riqueza de Wikipedia—. La propuesta es que, antes de aspirar a responder cosas que *no están* en ninguna KB, los sistemas que leen documentos deberían primero igualar a los sistemas basados en KB en dominios cerrados, donde la comparación es limpia. KV-MemNN es el modelo que estrecha esa brecha, y WikiMovies es el instrumento que la mide.

Para la Clase 30 (modelos con memoria externa) esto importa porque KV-MemNN es el eslabón que conecta la línea Memory Networks → End-to-End MemNN con una aplicación de QA realista sobre texto, introduciendo la idea —hoy ubicua en retrieval-augmented generation— de que **la representación con la que buscas no tiene por qué ser la representación que devuelves**.

## 2. Contexto histórico: QA sobre KB vs leer documentos directamente

Históricamente, los primeros sistemas de QA intentaban responder *leyendo documentos directamente* (Voorhees y Tice, 2000): recuperación de información que devolvía fragmentos de texto conteniendo la respuesta, con cobertura y complejidad de pregunta limitadas. La aparición de KBs a gran escala —Freebase (Bollacker et al., 2008), DBpedia (Auer et al., 2007)— reorientó el campo hacia el *semantic parsing*: convertir la pregunta en una forma lógica que consulta la KB (Berant et al., 2013; Kwiatkowski et al., 2013; Fader et al., 2014; Yih et al., 2015). Esto resultó muy efectivo y permite respuestas precisas a preguntas composicionales complicadas.

Pero las KBs tienen límites intrínsecos que el paper enumera con cuidado. Son **inevitablemente incompletas**: por más que se poblen, una colección de texto crudo como Wikipedia siempre contendrá más información. Tienen **esquemas fijos** que no soportan ciertos tipos de respuesta. Y son **escasas** en relación al texto que las origina —Wikipedia contiene mucho más que Freebase—. La extracción de información (IE), pensada para rellenar los huecos de las KBs automáticamente, "no es ni suficientemente precisa ni suficientemente confiable". La conclusión: las KBs pueden bastar para problemas de dominio cerrado, pero es improbable que escalen a responder preguntas generales sobre cualquier tema.

¿Por qué entonces no se lee texto directamente, si tiene más información? Porque **es mucho más difícil**: en el texto la información está menos estructurada, se expresa de forma indirecta y ambigua, y suele estar dispersa entre múltiples documentos. Esa es exactamente la brecha que motiva el paper. Iniciativas recientes (a 2016) volvían al texto con datasets como TREC QA (Wang et al., 2007) y WikiQA (Yang et al., 2015), pero estos se organizan en torno a *answer sentence selection* —elegir la oración que contiene la respuesta, no devolver la respuesta— y son diminutos (cientos de ejemplos). Por su tamaño y su formato de selección, no permiten comparar directamente responder desde una KB contra responder desde texto puro. WikiMovies se diseñó para tapar ambas deficiencias a la vez: un corpus grande de pares pregunta-respuesta que es respondible *tanto* desde una KB *como* desde los documentos correspondientes.

En paralelo, los mejores resultados publicados en TREC QA y WikiQA venían de CNNs (Santos et al., 2016; Yin y Schütze, 2015; Wang et al., 2016) o RNNs (Miao et al., 2015), casi siempre con mecanismos de atención inspirados en Bahdanau et al. (2015). KV-MemNN se inserta en ese paisaje como una Memory Network que opera sobre una **memoria simbólica estructurada en pares (clave, valor)** —algo que ninguna arquitectura de atención neuronal existente para QA empleaba.

## 3. Contribución central: la idea de clave-valor

La aportación conceptual es separar cada slot de memoria en dos vectores: una **clave** $k_i$ y un **valor** $v_i$. La etapa de *direccionamiento* (addressing/lookup) opera sobre la memoria de claves; la etapa de *lectura* (que entrega el resultado) opera sobre la memoria de valores. Esto da dos cosas que la End-to-End MemNN no tenía:

1. **Flexibilidad para codificar conocimiento previo.** El practicante diseña la clave con rasgos que ayuden a *emparejarla con la pregunta*, y el valor con rasgos que ayuden a *emparejarlo con la respuesta*. Estas dos cosas no tienen por qué coincidir.
2. **Más poder expresivo** vía transformaciones no triviales entre clave y valor.

Crucialmente, esto se logra sin perder entrenabilidad: el modelo completo —con sus transformaciones clave-valor— se entrena de punta a punta con retropropagación estándar vía descenso de gradiente estocástico. Si se fija la clave igual al valor para todos los slots, se recupera exactamente la End-to-End MemNN de Sukhbaatar et al. (2015); KV-MemNN es por tanto una **generalización estricta**.

La utilidad práctica de la separación es que permite codificar el conocimiento en formatos heterogéneos dentro de la misma maquinaria —triples de KB, ventanas de texto, documentos completos— y, con eso, **reducir la brecha entre responder desde una KB y leer texto directamente**. El ejemplo paradigmático es la representación de ventanas (§5): la clave es la ventana entera de $W$ palabras (más probable de emparejar con la pregunta), mientras que el valor es solo la palabra central de la ventana (la entidad, más probable de ser la respuesta). En una MemNN clásica, donde clave y valor son lo mismo, esa asimetría es imposible de expresar.

## 4. El método: hashing, direccionamiento, lectura y hops

Se definen los slots de memoria como pares $(k_1, v_1), \dots, (k_M, v_M)$ y se denota la pregunta $x$. El acceso a memoria tiene tres pasos, gobernados por un "controlador" (la red neuronal):

- **Key Hashing.** La pregunta preselecciona un subconjunto pequeño de la memoria (potencialmente enorme). Se usa un índice invertido que recupera $N$ memorias $(k_{h_1}, v_{h_1}), \dots, (k_{h_N}, v_{h_N})$ cuya clave comparte al menos una palabra con la pregunta, con frecuencia $< F = 1000$ (para ignorar stopwords), siguiendo a Dodge et al. (2016). Este paso —ausente en el paper original de Sukhbaatar et al.— es lo que hace computacionalmente viables las memorias grandes.

- **Key Addressing.** A cada memoria candidata se le asigna una probabilidad de relevancia comparando la pregunta con cada clave:
  $$p_{h_i} = \mathrm{Softmax}\big(A\Phi_X(x) \cdot A\Phi_K(k_{h_i})\big)$$
  donde $\Phi_\cdot$ son *feature maps* de dimensión $D$, $A$ es una matriz $d \times D$, y el producto punto vive en el espacio embebido. Los $\Phi$ son la pieza diseñable: la "ingeniería de conocimiento previo" entra por aquí.

- **Value Reading.** En la lectura final se leen los valores tomando su suma ponderada por las probabilidades de direccionamiento, devolviendo el vector $o$:
  $$o = \sum_i p_{h_i}\, A\Phi_V(v_{h_i}).$$

Sobre esto se montan **múltiples hops con actualización de la consulta**. El controlador usa $q = A\Phi_X(x)$ como consulta inicial. Tras recibir $o$, la consulta se actualiza: $q_2 = R_1(q + o)$, con $R$ una matriz $d \times d$. El acceso se repite —solo direccionamiento y lectura, *no* el hashing— usando una matriz distinta $R_j$ en cada hop $j$, y la ecuación de direccionamiento se reescribe para usar la consulta actualizada:
$$p_{h_i} = \mathrm{Softmax}\big(q_{j+1}^\top A\Phi_K(k_{h_i})\big).$$
La motivación: nueva evidencia se incorpora a la consulta para enfocar y recuperar información más pertinente en accesos siguientes. Tras $H$ hops fijos, el estado del controlador produce la predicción final sobre los candidatos de salida:
$$\hat{a} = \arg\max_{i=1,\dots,C} \mathrm{Softmax}\big(q_{H+1}^\top B\Phi_Y(y_i)\big),$$
donde los $y_i$ son los candidatos (p.ej. todas las entidades de la KB, o todas las oraciones candidatas en WikiQA), y $B$ es una matriz $d \times D$ que puede restringirse a ser idéntica a $A$. Se entrena de punta a punta minimizando la entropía cruzada estándar entre $\hat{a}$ y la respuesta correcta $a$, aprendiendo $A$, $B$ y $R_1, \dots, R_H$. Por simplicidad, $\Phi_X$ y $\Phi_Y$ (pregunta y respuesta) se mantienen fijos como bag-of-words; toda la riqueza está en $\Phi_K$ y $\Phi_V$.

## 5. Variantes de clave-valor: cómo se codifica el conocimiento

El paper detalla cómo instanciar $\Phi_K$ y $\Phi_V$ según la fuente, y este catálogo es la parte más didáctica:

- **KB Triple.** Las entradas de KB tienen forma "sujeto relación objeto". La clave se compone del sujeto (lado izquierdo) y la relación; el valor es el objeto (lado derecho). Se **duplica la KB** considerando la relación invertida: además de "Blade Runner directed_by Ridley Scott", se agrega "Ridley Scott !directed_by Blade Runner", donde `!directed_by` es una entrada distinta en el diccionario. Tener la entrada en ambos sentidos es clave para responder preguntas distintas ("¿Quién dirigió Blade Runner?" vs. "¿Qué dirigió Ridley Scott?"). En una MemNN sin pares clave-valor, el triple entero debe codificarse en el mismo slot.

- **Sentence Level.** Para un documento, se parte en oraciones; cada slot codifica una oración. Clave y valor codifican la oración entera como bag-of-words. Como clave y valor son iguales, esto equivale a una MemNN estándar.

- **Window Level.** Los documentos se parten en ventanas de $W$ palabras; solo se incluyen ventanas cuya palabra central es una entidad. La clave codifica la **ventana entera**; el valor codifica **solo la palabra central** (`__WINDOW__` como clave, entidad central `__MOVIE__`/`__CENTER__` como valor). Esto es imposible en la arquitectura MemNN. Tiene sentido porque la ventana completa es más probable de emparejar con la pregunta (como clave), mientras que la entidad central es más pertinente como respuesta (como valor).

- **Window + Center Encoding.** En vez de representar la ventana como bag-of-words puro —que mezcla el centro con el resto—, se **duplica el tamaño $D$ del diccionario** y se codifica el centro de la ventana con el segundo diccionario. Así el modelo distingue la relevancia del centro (más ligado a la respuesta) frente a las palabras de los costados (más ligadas a la pregunta).

- **Window + Title.** El título de un documento suele ser la respuesta a preguntas sobre su contenido ("¿En qué actuó Harrison Ford?" se responde con el título "Blade Runner"). Se agrega una representación donde la clave es la ventana de palabras pero el valor es el **título del documento**. Se conservan también los pares (ventana, centro) estándar, duplicando el número de slots, y se añade un rasgo `_window_` o `_title_` a la clave para diferenciar los dos valores. La versión `_title_` incluye además el título real en la clave. Es una representación inherentemente atada a datasets con títulos significativos.

## 6. El benchmark WikiMovies

WikiMovies consiste en pares pregunta-respuesta en el dominio de películas, con dos metas de diseño: que las técnicas de ML tengan suficientes ejemplos de entrenamiento, y que se pueda analizar fácilmente el rendimiento de distintas representaciones de conocimiento, desglosando por tipo de pregunta. Construye **tres formas de representación del conocimiento** sobre el mismo dominio:

- **Doc:** documentos de Wikipedia crudos. Se identificaron películas de OMDb con artículo asociado por coincidencia de título, conservando el título y la primera sección (antes de la caja de contenidos): ~17k documentos.
- **KB:** una KB clásica basada en grafos construida desde OMDb y MovieLens, con nueve tipos de relación (director, guionista, actor, año de estreno, idioma, género, tags, rating IMDb, votos IMDb). ~10k actores, ~6k directores, ~43k entidades en total, almacenada como triples. Los ratings y votos reales se discretizan en *bins* textuales ("unheard of", "well known", "famous"...). Solo se retienen triples cuyas entidades aparecen también en los artículos de Wikipedia, para garantizar que cada par QA sea igualmente respondible desde KB o desde documento.
- **IE:** extracción de información sobre las páginas de Wikipedia para construir una KB de forma similar. Se usa resolución de correferencia (Stanford NLP Toolkit) para reemplazar referencias pronominales/nominales por entidades, y *semantic role labeling* (SENNA) para emparejar verbos con argumentos. Cada triple se limpia, se lematiza (stars/starring/star → starred) y se le agrega el título de la película. Tiene la atractiva propiedad de expresiones más precisas y compactas, al costo de menor *recall* por triples malformados o ausentes (solo ~56% de los pares sujeto-objeto de IE coinciden con los triples de la KB original).

El conjunto tiene **más de 100,000 pares**, con **13 clases de preguntas** correspondientes a distintos tipos de arista de la KB, desde específicas (actor→película, película→actores) hasta generales (tag→película: "¿Qué películas se describen como distópicas?"). Las preguntas se generaron partiendo de SimpleQuestions (Bordes et al., 2015): se identificó el subconjunto de plantillas de anotadores humanos que cubrían los tipos de pregunta y se sustituyeron las entidades por las de la KB ("What movies did [@actor] star in?"). Splits disjuntos de ~96k / 10k / 10k (train / dev / test); la misma pregunta (aun reformulada) no puede aparecer en train y test. Es mucho mayor que WikiQA (~1000 pares de entrenamiento).

## 7. Experimentos: la brecha medida

El experimento central (Tabla 2, hits@1 en test) compara cuatro métodos sobre las tres fuentes KB / IE / Doc:

| Método | KB | IE | Doc |
|---|---|---|---|
| QA system (Bordes et al., 2014) | 93.5 | 56.5 | N/A |
| Supervised Embeddings | 54.4 | 54.4 | 54.4 |
| Memory Network | 78.5 | 63.4 | 69.9 |
| **Key-Value Memory Network** | **93.9** | **68.3** | **76.2** |

Las lecturas clave: KV-MemNN **supera a todos los demás en las tres fuentes**. El sistema de Bordes et al. (2014), diseñado para KBs, va muy bien en KB (93.5) pero no opera sobre documentos (N/A en Doc). KV-MemNN iguala/supera ese rendimiento en KB (93.9) y además funciona sobre texto. Leer Wikipedia directamente (Doc, 76.2) **supera a la KB por IE** (68.3) —resultado alentador hacia la lectura automática de máquinas—, aunque **persiste una brecha frente a la KB anotada por humanos** (93.9 vs. 76.2). La mejor representación de memoria para leer documentos es "Window-level + Center Encoding + Title" ($W=7$, $H=2$); la Tabla 3 muestra la escalera: sentence-level 52.4 → window-level 66.8 → +Title 74.1 → +Center Encoding+Title 76.9. Tanto *center encoding* como los rasgos de título ayudan; sentence-level es inferior.

El desglose por tipo de pregunta (Tabla 4) explica dónde está la brecha. IE pierde sobre todo en Writer/Director/Actor→Movie (la correferencia es difícil ahí). Doc pierde frente a KB en Tag→Movie, Movie→Tags, Movie→Writer y Movie→Actors. Las preguntas de tags son duras porque pueden referenciar casi cualquier palabra del documento; Movie→Writer/Actor son duras porque suele haber una sola referencia a la respuesta en todos los documentos.

El **análisis de documentos sintéticos** (Tabla 5) es el experimento más fino: se generan "Wikipedias" artificiales a partir de los triples de la KB con una gramática de plantillas (100 frases tipo "Blade Runner came out in 1982"), parametrizando la complejidad (una plantilla vs. todas; con/sin conjunciones; con/sin correferencia). Resultado: pasar de la KB a una sola oración-plantilla ya cuesta (93.9 → 82.9), es decir, *representar el hecho en forma de oración* hace más difícil extraer sujeto/relación/objeto. Usar muchas plantillas casi no degrada (80.0). El resto de la caída se reparte entre conjunciones (74.0) y correferencia (76.0). El dataset sintético más duro (All Templates + Conj. + Coref., 72.5) es incluso más difícil que Wikipedia real (76.2), probablemente porque la cantidad de conjunciones (50%) y correferencias (80%) inyectadas es artificialmente alta.

Finalmente, en **WikiQA** (Tabla 6, answer sentence selection, métricas MAP/MRR) KV-MemNN alcanza el estado del arte (MAP 0.7069, MRR 0.7265), superando a CNNs y LSTMs con atención y empatando casi exactamente con L.D.C. (Wang et al., 2016). Se pre-entrenaron los vectores de palabra con Supervised Embeddings, se usó dropout sobre pregunta/memoria/respuestas, rasgos de *exact match*, y la representación Window-Level ($W=7$) como clave con la oración entera como valor (porque la respuesta aquí es una oración). La MemNN clásica, que no puede emparejar ventanas con oraciones, va mucho peor —subrayando la importancia de la memoria clave-valor.

## 8. Limitaciones reconocidas

- **La brecha no se cierra, solo se reduce.** El propio título de la sección de conclusión lo admite: el modelo ayuda a *acortar* la distancia entre leer documentos y usar una KB anotada, pero "alguna brecha persiste" (93.9 vs. 76.2 en WikiMovies). El trabajo futuro debe seguir cerrándola.
- **Dominio cerrado y plantillas.** WikiMovies es un dominio cerrado (películas) y sus preguntas se generan sustituyendo entidades en plantillas derivadas de anotadores. Esto da control experimental limpio pero introduce regularidad sintáctica que no refleja la variedad de preguntas reales abiertas.
- **Feature maps fijos y diseñados a mano.** $\Phi_X$ y $\Phi_Y$ se mantienen como bag-of-words; las representaciones clave-valor más potentes (Window+Title, Center Encoding) son *ingeniería de rasgos específica del dataset* —Window+Title es explícitamente inaplicable a colecciones sin títulos significativos.
- **Dependencia del hashing como recuperador.** El key hashing usa un índice invertido por solapamiento de palabras; el paper reconoce que esquemas de recuperación más sofisticados (Manning et al., 2008) podrían usarse. Si la respuesta no comparte vocabulario con la pregunta, el candidato correcto puede no entrar al subconjunto recuperado.
- **IE como cuello de botella.** Solo ~56% de los pares sujeto-objeto de IE coinciden con la KB original, de modo que el techo de la ruta IE está limitado por la calidad de la extracción, no por el modelo de QA.

## 9. Impacto

KV-MemNN es una pieza bisagra en la genealogía de modelos con memoria externa. Hereda directamente de Memory Networks (Weston et al., 2014) y End-to-End Memory Networks (Sukhbaatar et al., 2015), de las que es generalización estricta, y se contemporiza con Neural Turing Machines / DNC (Graves et al., 2014, 2016) y las Recurrent Entity Networks (Henaff et al., 2017) —el resto del módulo de memoria externa de la Clase 30—. Su aporte más duradero es conceptual y precede a una idea central del *retrieval-augmented generation* moderno: **desacoplar la representación de búsqueda de la representación de retorno**. La clave (con qué emparejas la consulta) y el valor (qué devuelves) son distintos —exactamente el patrón que hoy estructura los almacenes vectoriales y los recuperadores densos donde se indexa por un embedding y se devuelve el pasaje original. El dataset WikiMovies, por su parte, quedó como herramienta de análisis estándar para estudiar la brecha KB-vs-texto y para entrenar/evaluar lectores de documentos, y se distribuye dentro del corpus bAbI de FAIR. La demostración empírica de que leer Wikipedia directamente puede superar a una KB por IE fue un argumento temprano y citado a favor de la lectura automática de máquinas sobre texto crudo.

## 10. Conexión con la Clase 30 (Modelos con memoria externa)

La Clase 30 dedica diapositivas específicas a las Key-Value Memory Networks (Miller et al., 2016), y este análisis aterriza exactamente lo que esas slides muestran:

- **El dataset WikiMovies** aparece en la clase como el banco de pruebas que materializa la brecha KB-vs-texto, con sus tres fuentes (KB / IE / Doc) sobre el mismo conocimiento de películas.
- **El ejemplo de ventanas clave-valor** —`__WINDOW__` como clave y `__MOVIE__`/centro como valor— es el caso canónico que la clase usa para ilustrar *por qué* separar clave de valor: buscas con la ventana entera, devuelves solo la entidad central. Es la idea que una MemNN clásica no puede expresar.
- **La tabla de resultados** (KB 93.9 / IE 68.3 / Doc 76.2 para KV-MemNN, con Doc > IE pero Doc < KB) es la evidencia numérica que la clase presenta para sostener que el modelo reduce —sin cerrar— la brecha.

Para situar KV-MemNN dentro del módulo conviene leerlo junto a sus vecinos directos del curso. El fundamento transversal de [redes con memoria aumentada](/fundamentos/memory-augmented-networks) explica el patrón general controlador + memoria direccionable por contenido del que esta arquitectura es un caso. El hub de la [Clase 30](/clases/clase-30) ubica el paper en la secuencia Memory Networks → End-to-End → Key-Value → NTM/DNC → Entity Networks. Y el ancestro inmediato, la [End-to-End Memory Network de Sukhbaatar et al. (2015)](/papers/e2e-memnn-sukhbaatar-2015), es literalmente el caso particular que se recupera al fijar clave = valor: leer ambos papers en orden hace visible qué agrega exactamente la separación clave-valor —flexibilidad de codificación y transformaciones no triviales entre lo que buscas y lo que devuelves.
