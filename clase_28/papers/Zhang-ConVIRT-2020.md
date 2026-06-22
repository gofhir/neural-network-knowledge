# ConVIRT: Contrastive Learning of Medical Visual Representations from Paired Images and Text — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Contrastive Learning of Medical Visual Representations from Paired Images and Text*.
- **Autores:** Yuhao Zhang (Biomedical Informatics Training Program, Stanford; hoy en AWS AI Labs), Hang Jiang (Symbolic Systems, Stanford; hoy en MIT) —los dos primeros contribuyeron por igual—, Yasuhide Miura (Computer Science, Stanford; hoy en FUJIFILM), Christopher D. Manning (CS y Lingüística, Stanford) y Curtis P. Langlotz (Departamento de Radiología, Stanford).
- **Venue:** *Proceedings of Machine Learning for Healthcare (MLHC)* 2022, PMLR vol. 182. **Preprint:** arXiv:2010.00747 (v1 octubre 2020; v2, 19 sep 2022).
- **Código y datos:** [github.com/yuhaozhang/convirt](https://github.com/yuhaozhang/convirt) — modelo y datasets de retrieval anotados liberados públicamente.

ConVIRT (**Con**trastive **VI**sual **R**epresentation Learning from **T**ext) es un método de aprendizaje autosupervisado que aprende representaciones de imágenes médicas **sin una sola etiqueta manual**, explotando los pares imagen-reporte que ya existen gratuitamente en los sistemas clínicos de un hospital. La tesis central es de una claridad incómoda para el dogma de la época: en medicina, la práctica dominante de inicializar redes con pesos preentrenados en ImageNet es *subóptima* —las características visuales de una radiografía son drásticamente distintas a las de los objetos naturales— y el método autosupervisado solo-imagen (SimCLR, MoCo) apenas ayuda, porque las imágenes médicas tienen **altísima similitud inter-clase** (dos radiografías de tórax, una sana y una con cardiomegalia, se parecen muchísimo más entre sí que un gato y un avión). La señal que falta no está en otra augmentación de la imagen: está en el **texto** que un radiólogo ya escribió describiéndola.

El resultado más citado del paper es de eficiencia de datos: en las cuatro tareas de clasificación evaluadas, ConVIRT iguala o supera a una inicialización ImageNet usando **solo el 10% de las etiquetas** (y en tres de cuatro tareas, bajo evaluación lineal, basta el **1%** para superar a ImageNet con el 100% de las etiquetas). Para un dominio donde cada etiqueta cuesta el tiempo de un radiólogo certificado, esto no es un número de benchmark: es la diferencia entre un proyecto viable y uno imposible.

Su relevancia histórica es doble. Primero, ConVIRT (octubre 2020) es el **precursor directo de CLIP** (Radford et al., 2021): el propio paper de OpenAI cita a ConVIRT y describe su método como "una versión simplificada del enfoque ConVIRT"; ALIGN (Jia et al., 2021) sigue la misma genealogía. Es decir, la receta image-text contrastiva que hoy asociamos con CLIP nació *en el dominio médico, con menos datos, y con un objetivo de salud pública*. Segundo, dentro de la medicina, ConVIRT fue —según los autores— el primer trabajo en usar pérdida contrastiva texto-imagen para preentrenar representaciones visuales médicas, y abrió la línea que continúan GLoRIA, LoVT, PubMedCLIP y otros.

## 2. Contexto: por qué etiquetar imágenes médicas es el cuello de botella

El paper abre con la promesa real del deep learning en imagen médica (retinopatía diabética, cáncer de piel, enfermedad retinal) y con su freno igualmente real: la **escasez extrema de anotaciones**. El argumento es importante para la Clase 28 porque es el corazón del *por qué* del aprendizaje autosupervisado, llevado al dominio donde más muerde.

Históricamente había dos caminos para obtener etiquetas en imagen médica, y ambos son malos:

1. **Anotaciones de expertos.** Alta calidad, pero carísimas: requieren radiólogos o especialistas. El costo hace que los datasets médicos sean órdenes de magnitud más pequeños que ImageNet. Y para llenar ese hueco, todo el mundo recurre a transferir pesos de ImageNet —algo que Raghu et al. (2019) mostraron que a menudo no aporta beneficio frente a inicialización aleatoria, precisamente porque la imagen médica exige features de grano muy fino, muy distintos de los de "identificar objetos".
2. **Reglas para extraer etiquetas del reporte.** Como el texto del radiólogo se produce *gratis* en el flujo de trabajo y abunda en el sistema de IT del hospital, se escriben reglas que parsean el reporte y emiten una etiqueta (p. ej. CheXpert labeler). Esto da datasets más grandes, pero las reglas son inexactas, se limitan a unas pocas categorías (uso muy ineficiente del texto) y son frágiles: dependen del estilo de redacción, así que no generalizan entre instituciones ni dominios.

ConVIRT propone un tercer camino que combina lo mejor de ambos: usa el **texto abundante y gratuito** del reporte, pero *no* lo convierte en etiquetas discretas vía reglas frágiles. En su lugar, lo usa directamente como **señal de supervisión continua** mediante un objetivo contrastivo. El texto deja de ser una fuente de etiquetas ruidosas y pasa a ser el otro extremo de un alineamiento multimodal.

El paper enmarca esto como un *insight generalizable sobre ML en salud* (sección 1.1): los datos clínicos son escasos y caros de anotar, pero **naturalmente vienen emparejados con features multimodales** (texto descriptivo, metadata del paciente). Reutilizar esa información gratuita es la palanca para construir modelos confiables a bajo costo de anotación. Es exactamente la nota que la Clase 28 enfatiza: el SSL importa *más* donde etiquetar es caro.

## 3. Contribución central

La aportación de ConVIRT es **aprender un encoder de imagen médica maximizando el acuerdo entre pares imagen-texto verdaderos frente a pares aleatorios, mediante un objetivo contrastivo bidireccional entre las dos modalidades**, sin ningún input experto adicional y de forma agnóstica a la especialidad médica.

Tres rasgos lo distinguen de lo que existía:

- **Frente a SSL solo-imagen (SimCLR, MoCo):** en vez de contrastar dos vistas augmentadas de *la misma imagen*, ConVIRT contrasta la imagen contra su *texto* emparejado. La semántica añadida por el texto es lo que rompe la barrera de la alta similitud inter-clase de las radiografías.
- **Frente a captioning (CNN-LSTM, CNN-Transformer):** no genera texto. Captioning fuerza al modelo a decodificar el reporte completo palabra a palabra —tarea costosa y que aprende detalles irrelevantes—; ConVIRT solo pide alinear representaciones globales, una señal más limpia para el encoder visual.
- **Frente a pretraining visual-lingüístico con cabeza binaria** (LXMERT, VL-BERT y similares, que predicen "par real / par falso" con BCE): ConVIRT usa un objetivo contrastivo NCE basado en similitud coseno, que el paper muestra produce representaciones mejor alineadas entre modalidades —crítico para retrieval texto-imagen, donde la cabeza binaria fracasa.

Y todo esto siendo **agnóstico al encoder**: el framework no exige una arquitectura concreta de imagen ni de texto, ni segmentaciones anatómicas (que en medicina son carísimas de obtener), lo que lo hace directamente transferible entre especialidades.

## 4. Método

### 4.1. Definición y arquitectura de dos torres

Se asume un par de entrada $(x_v, x_u)$, donde $x_v$ es una o un grupo de imágenes y $x_u$ una secuencia de texto que las describe. El objetivo es aprender un encoder de imagen $f_v$ que mapee una imagen a un vector de dimensión fija, transferible luego a tareas downstream (clasificación, retrieval).

El pipeline es de **dos torres simétricas en estructura, asimétricas en datos**:

- **Rama de imagen (azul):** de $x_v$ se extrae una vista aleatoria $\tilde{x}_v$ con una transformación $t_v \sim \mathcal{T}$; el encoder $f_v$ la lleva a $h_v$; una proyección no lineal $g_v$ lleva $h_v$ al vector final $v = g_v(f_v(\tilde{x}_v))$, con $v \in \mathbb{R}^d$.
- **Rama de texto (verde):** de $x_u$ se muestrea un span $\tilde{x}_u$ con $t_u$; un encoder de texto $f_u$ y una proyección $g_u$ producen $u = g_u(f_u(\tilde{x}_u))$, con $u \in \mathbb{R}^d$.

Las proyecciones $g_v$ y $g_u$ mandan ambas modalidades a un **mismo espacio $d$-dimensional**, donde vive el contraste. Son redes de una capa oculta: $g(\cdot) = W^{(2)}\sigma(W^{(1)}(\cdot))$ con ReLU, siguiendo a SimCLR.

### 4.2. La pérdida contrastiva bidireccional

Por minibatch se muestrean $N$ pares $(x_v, x_u)$ y se calculan sus representaciones $(v_i, u_i)$. La novedad respecto al SSL solo-imagen es que **el contraste es entre modalidades distintas, por lo que es asimétrico** y hay que definir las dos direcciones.

Pérdida **imagen→texto** para el par $i$ (forma InfoNCE):

$$\ell_i^{(v\to u)} = -\log \frac{\exp(\langle v_i, u_i\rangle / \tau)}{\sum_{k=1}^{N} \exp(\langle v_i, u_k\rangle / \tau)}$$

donde $\langle v, u\rangle = v^\top u / \lVert v\rVert\lVert u\rVert$ es la similitud coseno y $\tau \in \mathbb{R}^+$ es la temperatura. Intuitivamente, es la log-loss de un clasificador $N$-vías que, dada la imagen $v_i$, debe elegir su texto verdadero $u_i$ entre los $N$ textos del batch (los otros $N-1$ son negativos). Minimizarla preserva la información mutua entre los pares verdaderos.

Por simetría se define la pérdida **texto→imagen**:

$$\ell_i^{(u\to v)} = -\log \frac{\exp(\langle u_i, v_i\rangle / \tau)}{\sum_{k=1}^{N} \exp(\langle u_i, v_k\rangle / \tau)}$$

aquí, dado el texto, hay que recuperar su imagen verdadera entre las $N$ del batch. La pérdida total es la combinación ponderada promediada sobre el minibatch:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\Big(\lambda\,\ell_i^{(v\to u)} + (1-\lambda)\,\ell_i^{(u\to v)}\Big)$$

con $\lambda \in [0,1]$ escalar. La **bidireccionalidad** es lo que ata el espacio en ambos sentidos y es la razón de que el retrieval texto-imagen funcione tan bien (ver §6): la cabeza binaria de los baselines no impone esta alineación basada en similitud y por eso produce espacios desalineados.

### 4.3. Realización concreta

El framework es agnóstico, pero la instancia evaluada usa:

- **Encoder de imagen $f_v$:** **ResNet50** (He et al., 2016) en todos los experimentos —la arquitectura estándar en imagen médica.
- **Encoder de texto $f_u$:** **BERT** (Devlin et al., 2019) con max-pooling sobre las salidas, inicializado con **ClinicalBERT** (Alsentzer et al., 2019), preentrenado sobre las notas clínicas de MIMIC. Se congelan los embeddings y las primeras 6 capas Transformer, y se afinan las últimas 6 —dejando que el encoder de texto se adapte a la tarea contrastiva sin destruir su conocimiento clínico.
- **Augmentación de imagen $\mathcal{T}$:** cinco transformaciones secuenciales (crop, flip horizontal, transformación afín, color jittering, desenfoque gaussiano). Detalle clave de dominio: en el color jittering **solo se ajustan brillo y contraste**, no color, por la naturaleza monocroma de las imágenes médicas.
- **Augmentación de texto $t_u$:** muestreo uniforme de **una oración** del reporte por minibatch. Deliberadamente suave: muestrear a nivel de oración preserva el significado semántico, mientras que transformaciones más agresivas lo romperían.

Un hallazgo de diseño: usar la vista muestreada $\tilde{x}_v$ (en vez de la imagen completa o la fusión de todas las vistas de un estudio) mejora el preentrenamiento, porque actúa como augmentación que aumenta la cantidad efectiva de pares únicos que el modelo ve.

## 5. Experimentos: datos, tareas y resultados

### 5.1. Datos de preentrenamiento

Se preentrenan **dos encoders separados** sobre dos datasets pareados imagen-texto:

- **Tórax:** **MIMIC-CXR v2** (Johnson et al., 2019), ~**217k** pares imagen-texto (promedio 1.7 imágenes y 6.0 oraciones por par). Recurso estándar para modelado multimodal de imagen médica.
- **Hueso (musculoesquelético):** colección del Rhode Island Hospital, **48k** pares (promedio 2.5 imágenes, 8.0 oraciones). Las radiografías óseas son el segundo tipo más común tras el tórax.

> **Nota para la Clase 28.** El slide "Contrastive Learning en otros dominios (Zhang et al. 2020)" cita las cifras **RSNA 25k, CheXpert 220k, COVIDx 14k, MURA 33k**: esas son los datasets *downstream* de clasificación sobre los que se *evalúa* el encoder, no los datos de *preentrenamiento* (MIMIC-CXR 217k + hueso 48k). El punto pedagógico es el mismo: hay volúmenes grandes de imágenes médicas, pero etiquetarlas todas es prohibitivo, y ConVIRT muestra que con una fracción mínima de esas etiquetas basta si el encoder ya aprendió del texto.

### 5.2. Tareas de evaluación

Tres familias, transfiriendo el encoder congelado o afinado:

- **Clasificación (4 tareas, 2 especialidades):** **RSNA** (neumonía sí/no, AUC), **CheXpert** (multi-etiqueta de 5 hallazgos: atelectasia, cardiomegalia, consolidación, edema, derrame pleural, AUC), **COVIDx** (3 clases: COVID-19 / neumonía no-COVID / normal, accuracy) y **MURA** (anormalidad ósea sí/no, AUC). Cada una bajo dos protocolos —*linear* (CNN congelada, solo se entrena la cabeza lineal: mide la calidad pura de los features) y *fine-tuning* (se afina todo: refleja el uso práctico)— y con **1%, 10% y 100%** de las etiquetas, promediando 5 corridas.
- **Retrieval imagen-imagen zero-shot:** dataset propio **CheXpert 8×200** (8 categorías de anormalidad, 10 query + 200 candidatas cada una, anotadas por radiólogo certificado), métrica Precision@k.
- **Retrieval texto-imagen zero-shot:** un radiólogo escribe 5 descripciones textuales por categoría; se recupera imagen desde texto. Evalúa la **alineación** entre los espacios de texto e imagen.

### 5.3. Baselines y resultados

Se compara contra **Random Init**, **ImageNet Init** (el rival a vencer), dos baselines de captioning in-domain (**Caption-LSTM**, **Caption-Transformer**) y **Contrastive-Binary-Loss** (la cabeza binaria del pretraining visual-lingüístico clásico).

**Clasificación.** ConVIRT gana el mejor resultado global en todos los settings lineales y en 10 de 11 settings de fine-tuning (la única excepción: CheXpert con 100% de datos, donde empata con Caption-Transformer). Los números que sostienen la tesis de eficiencia de datos:

| Tarea (Linear) | ImageNet 100% | ConVIRT 1% | ConVIRT 10% |
|---|---|---|---|
| RSNA (AUC) | 86.9 | **90.7** | 91.7 |
| CheXpert (AUC) | 81.0 | **85.9** | 86.8 |
| MURA (AUC) | 79.0 | **81.2** | 85.1 |

ConVIRT con **1%** de etiquetas supera a ImageNet con **100%** en tres de las cuatro tareas (linear), y con **10%** iguala o supera a ImageNet-100% en fine-tuning en todas. Es un orden de magnitud menos anotación para igual o mejor rendimiento.

**Retrieval.** ConVIRT domina todas las métricas. En texto-imagen la diferencia es brutal: **Prec@5 = 60.0** frente a **15.5** de Contrastive-Binary-Loss (los baselines de captioning ni siquiera tienen encoder de texto). La razón, conjeturan los autores, es que la cabeza binaria no impone una pérdida explícita de similitud, así que los espacios de imagen y texto quedan desalineados. Visualmente, los t-SNE muestran que ConVIRT agrupa mejor las categorías de anormalidad que ImageNet, pese a la dificultad de la alta similitud inter-clase.

### 5.4. Análisis: contra SSL solo-imagen y otras observaciones

La comparación crítica es contra **SimCLR y MoCo v2** corridos sobre las *mismas* imágenes. Confirman que el SSL solo-imagen aporta mejoras marginales-moderadas sobre ImageNet, pero ConVIRT los supera ampliamente en todo:

| Método | RSNA Linear 1% (AUC) | CheXpert Linear 1% (AUC) | Img-Img Prec@10 |
|---|---|---|---|
| ImageNet | 82.8 | 75.7 | 14.4 |
| SimCLR | 86.3 | 77.4 | 17.6 |
| MoCo v2 | 86.6 | 81.3 | 20.6 |
| **ConVIRT** | **90.7** | **85.9** | **42.9** |

Los **mapas de saliencia** (Figura 4) lo explican cualitativamente: ImageNet se fija en features triviales irrelevantes; ConVIRT enfoca las regiones anatómicamente correctas (p. ej. la base del pulmón para atelectasia), mientras SimCLR dispersa el foco y MoCo se equivoca enfocando el corazón. La señal del texto guía al encoder hacia lo clínicamente pertinente.

Otros hallazgos: existe **correlación positiva** entre la pérdida de preentrenamiento y el rendimiento downstream (mejor pretraining ⇒ mejor tarea final, lo que sugiere que seguir mejorando el objetivo contrastivo seguiría ayudando). En hiperparámetros, lo más sensible es la **temperatura $\tau$** (default 0.1; $\tau=0.01$ o $\tau=1$ degradan, sobre todo el retrieval); cambiar el batch size (16/32/128) casi no afecta clasificación —a diferencia del SSL solo-imagen, que es muy ávido de batches grandes—; y usar proyección lineal en lugar de no lineal degrada moderadamente el retrieval.

## 6. Limitaciones

Los propios autores acotan el alcance: la comparación se centra en ImageNet, captioning y SSL solo-imagen, y **no** se compara contra trabajos posteriores que extienden ConVIRT (LoVT, GLoRIA), porque esas comparaciones están en los propios papers derivados. Otras limitaciones legibles entre líneas: la evaluación es exclusivamente radiológica (tórax y hueso), por lo que la generalización a otras modalidades de imagen (histopatología, fondo de ojo, dermatología) queda como conjetura; depende de la existencia de pares imagen-reporte de calidad, que no todo sistema clínico produce de forma estructurada; y hereda los sesgos del corpus de reportes (estilo de redacción, sesgos demográficos del hospital de origen) sin que el paper los audite. Para un despliegue clínico real importan además la calibración, la robustez a *distribution shift* entre instituciones y la validación prospectiva —temas que el paper, centrado en calidad de representación, no aborda.

## 7. Impacto y conexión con la Clase 28

ConVIRT es uno de esos papers cuya importancia se mide por lo que vino después. **CLIP** (Radford et al., 2021) escaló su receta a 400M de pares web y la llamó explícitamente "una versión simplificada de ConVIRT"; **ALIGN** (Jia et al., 2021) hizo lo propio con 1.8B de pares ruidosos. La genealogía del paradigma image-text contrastivo que hoy fundamenta los modelos multimodales nace, literalmente, en un problema de radiología en Stanford. Dentro de la medicina, ConVIRT inauguró una línea fértil: GLoRIA (contraste local región-palabra), LoVT (tareas localizadas), PubMedCLIP (VQA médico), y aplicaciones a riesgo clínico sobre EHR longitudinales, detección de neumonía con features radiómicos y selección de pares vía metadata del paciente.

Para la Clase 28 (**Aprendizaje Autosupervisado**), ConVIRT es el ejemplo canónico del slide "Contrastive Learning en otros dominios": demuestra, con números duros, que **el SSL importa más donde etiquetar es caro**. En visión natural, el SSL ahorra costo; en medicina, *habilita* lo que de otro modo sería inviable, porque cada etiqueta cuesta el tiempo de un especialista. El mecanismo —contraste bidireccional InfoNCE entre dos torres— es el mismo que la clase presenta para imágenes naturales (SimCLR, MoCo), pero el segundo extremo del contraste deja de ser otra vista de la imagen y pasa a ser el texto gratuito del experto. Esa sustitución es lo que rompe la barrera de la alta similitud inter-clase que hace fracasar al SSL solo-imagen en radiografías.

Para quien trabaja en salud, la lección operativa es directa: los pares imagen-reporte —y, por extensión, cualquier modalidad clínica naturalmente emparejada (texto, metadata, multiómica)— son una mina de supervisión gratuita que evita el cuello de botella de la anotación experta. ConVIRT es la prueba de concepto de que reutilizar esa información ya producida en el flujo de trabajo clínico es el camino más eficiente —y a veces el único viable— para construir modelos médicos confiables a bajo costo.

## 8. Enlaces internos

- Sucesor general-domain a gran escala: [CLIP (Radford et al., 2021)](/papers/clip-radford-2021)
- Fundamento del mecanismo: [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo)
- Marco general: [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado)
- Clase que lo enmarca: [Clase 28 — Aprendizaje Autosupervisado](/clases/clase-28)
