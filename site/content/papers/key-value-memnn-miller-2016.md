---
title: "Key-Value Memory Networks (2016)"
weight: 343
math: true
---

{{< paper-card
    title="Key-Value Memory Networks for Directly Reading Documents"
    authors="Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, Jason Weston"
    year="2016"
    venue="EMNLP 2016"
    pdf="/papers/key-value-memnn-miller-2016.pdf"
    arxiv="1606.03126" >}}
Paper de Facebook AI Research que hace dos cosas a la vez. Primero propone la **Key-Value Memory Network (KV-MemNN)**, que generaliza la [End-to-End Memory Network](/papers/e2e-memnn-sukhbaatar-2015) separando cada slot de memoria en un par **(clave, valor)**: la clave sirve para *direccionar* (atender) y el valor para *devolver* el contenido. Segundo, construye **WikiMovies**, un dataset de ~100k preguntas sobre películas respondibles desde tres fuentes que codifican el *mismo* conocimiento —KB anotada, KB por extracción de información (IE) y documentos de Wikipedia— diseñado para medir cuánto cae el rendimiento al pasar de una base de conocimiento perfecta a texto crudo. Idea central, hoy ubicua en *retrieval-augmented generation*: la representación con la que **buscas** no tiene por qué ser la que **devuelves**.
{{< /paper-card >}}

---

## Contexto: KB vs leer documentos directamente

Los primeros sistemas de QA respondían *leyendo documentos directamente* (Voorhees y Tice, 2000), con cobertura limitada. La aparición de KBs a gran escala —Freebase, DBpedia— reorientó el campo hacia el *semantic parsing*: convertir la pregunta en una forma lógica que consulta la KB. Esto resultó muy efectivo y permite respuestas precisas a preguntas composicionales.

Pero las KBs tienen límites intrínsecos que el paper enumera con cuidado. Son **inevitablemente incompletas**: una colección de texto crudo como Wikipedia siempre contendrá más información. Tienen **esquemas fijos** que no soportan ciertos tipos de respuesta. Y son **escasas** frente al texto que las origina. La extracción de información (IE), pensada para rellenar esos huecos, "no es ni suficientemente precisa ni suficientemente confiable". Conclusión: las KBs bastan para dominios cerrados, pero es improbable que escalen a preguntas generales sobre cualquier tema.

¿Por qué entonces no leer texto directamente, si tiene más información? Porque **es mucho más difícil**: en el texto la información está menos estructurada, se expresa de forma indirecta y ambigua, y suele estar dispersa entre múltiples documentos. Esa es la brecha que motiva el paper. Datasets previos sobre texto (TREC QA, WikiQA) se organizaban en torno a *answer sentence selection* —elegir la oración con la respuesta, no devolverla— y eran diminutos (cientos de ejemplos). WikiMovies tapa ambas deficiencias: un corpus grande respondible *tanto* desde KB *como* desde los documentos correspondientes.

## Contribución central: la idea de clave-valor

La aportación conceptual es separar cada slot de memoria en dos vectores: una **clave** $k_i$ y un **valor** $v_i$. La etapa de *direccionamiento* (addressing) opera sobre la memoria de claves; la etapa de *lectura*, que entrega el resultado, opera sobre la memoria de valores. Esto da dos cosas que la End-to-End MemNN no tenía:

1. **Flexibilidad para codificar conocimiento previo.** Se diseña la clave con rasgos que ayuden a *emparejarla con la pregunta*, y el valor con rasgos que ayuden a *emparejarlo con la respuesta*. Las dos cosas no tienen por qué coincidir.
2. **Más poder expresivo** vía transformaciones no triviales entre clave y valor.

Todo se entrena de punta a punta con retropropagación estándar. Si se fija clave = valor en todos los slots, se recupera exactamente la End-to-End MemNN de Sukhbaatar et al. (2015): KV-MemNN es una **generalización estricta**. La utilidad práctica es codificar conocimiento en formatos heterogéneos —triples de KB, ventanas de texto, documentos— dentro de la misma maquinaria, y con eso **reducir la brecha entre responder desde KB y leer texto directamente**. El ejemplo paradigmático es la representación de ventanas: la clave es la ventana entera de $W$ palabras (más probable de emparejar con la pregunta), mientras que el valor es solo la palabra central (la entidad, más probable de ser la respuesta). En una MemNN clásica, donde clave y valor son lo mismo, esa asimetría es imposible.

## El método: hashing, direccionamiento, lectura y hops

Los slots se definen como pares $(k_1, v_1), \dots, (k_M, v_M)$, con la pregunta $x$. El acceso a memoria tiene tres pasos gobernados por un "controlador" (la red neuronal):

- **Key Hashing.** La pregunta preselecciona un subconjunto pequeño de la memoria mediante un índice invertido que recupera $N$ memorias cuya clave comparte al menos una palabra con la pregunta, con frecuencia $< F = 1000$ (para ignorar stopwords). Este paso —ausente en Sukhbaatar et al.— hace computacionalmente viables las memorias grandes.

- **Key Addressing.** A cada memoria candidata se le asigna una probabilidad de relevancia comparando la pregunta con cada clave:
  $$p_{h_i} = \mathrm{Softmax}\big(A\Phi_X(x) \cdot A\Phi_K(k_{h_i})\big)$$
  donde $\Phi_\cdot$ son *feature maps* de dimensión $D$ y $A$ es una matriz $d \times D$. Los $\Phi$ son la pieza diseñable: la "ingeniería de conocimiento previo" entra por aquí.

- **Value Reading.** Se leen los valores como suma ponderada por las probabilidades de direccionamiento:
  $$o = \sum_i p_{h_i}\, A\Phi_V(v_{h_i}).$$

Sobre esto se montan **múltiples hops con actualización de la consulta**. El controlador parte de $q = A\Phi_X(x)$; tras recibir $o$, actualiza la consulta $q_2 = R_1(q + o)$ con $R$ una matriz $d \times d$. El acceso se repite —solo direccionamiento y lectura, *no* el hashing— con una matriz $R_j$ distinta por hop, de modo que la nueva evidencia se incorpora a la consulta para enfocar accesos siguientes. Tras $H$ hops, el estado produce la predicción final sobre los candidatos $y_i$:
$$\hat{a} = \arg\max_{i=1,\dots,C} \mathrm{Softmax}\big(q_{H+1}^\top B\Phi_Y(y_i)\big),$$
entrenando $A$, $B$ y $R_1, \dots, R_H$ por entropía cruzada. Por simplicidad $\Phi_X$ y $\Phi_Y$ se mantienen como bag-of-words; toda la riqueza está en $\Phi_K$ y $\Phi_V$.

## Variantes de clave-valor: cómo se codifica el conocimiento

El catálogo de instanciaciones de $\Phi_K$ y $\Phi_V$ es la parte más didáctica:

- **KB Triple.** Forma "sujeto relación objeto". La clave es sujeto + relación; el valor es el objeto. Se **duplica la KB** con la relación invertida ("Ridley Scott !directed_by Blade Runner"), clave para responder preguntas en ambos sentidos.
- **Sentence Level.** Cada slot codifica una oración entera como bag-of-words, igual en clave y valor; equivale a una MemNN estándar.
- **Window Level.** Ventanas de $W$ palabras cuya palabra central es una entidad. La clave codifica la **ventana entera**; el valor codifica **solo la palabra central**. Imposible en una MemNN clásica.
- **Window + Center Encoding.** Se **duplica el tamaño $D$ del diccionario** para codificar el centro de la ventana con un segundo diccionario, distinguiéndolo de las palabras de los costados.
- **Window + Title.** El título de un documento suele ser la respuesta ("¿En qué actuó Harrison Ford?" → "Blade Runner"). Se agrega un valor = título, con un rasgo `_window_`/`_title_` en la clave para diferenciar. Está inherentemente atada a datasets con títulos significativos.

## El benchmark WikiMovies

WikiMovies son pares pregunta-respuesta del dominio de películas, con dos metas de diseño: dar suficientes ejemplos de entrenamiento y permitir analizar el rendimiento desglosando por tipo de pregunta. Construye **tres representaciones del mismo conocimiento**:

| Fuente | Construcción |
|---|---|
| **Doc** | Documentos de Wikipedia crudos (título + primera sección): ~17k documentos |
| **KB** | KB clásica basada en grafos desde OMDb + MovieLens, 9 relaciones, ~43k entidades, como triples (ratings/votos discretizados en *bins* textuales) |
| **IE** | Extracción de información sobre las páginas: correferencia (Stanford NLP) + *semantic role labeling* (SENNA), lematizada. Solo ~56% de pares sujeto-objeto coinciden con la KB |

Solo se retienen triples cuyas entidades aparecen también en los artículos, para garantizar que cada par QA sea igualmente respondible desde KB o desde documento. El conjunto tiene **más de 100.000 pares** con **13 clases de preguntas** (desde actor→película hasta tag→película), generadas sustituyendo entidades en plantillas derivadas de SimpleQuestions. Splits disjuntos de ~96k / 10k / 10k (train / dev / test); la misma pregunta no aparece en train y test. Es mucho mayor que WikiQA (~1000 pares de entrenamiento).

## Experimentos: la brecha medida

El experimento central (hits@1 en test) compara cuatro métodos sobre las tres fuentes:

| Método | KB | IE | Doc |
|---|---|---|---|
| QA system (Bordes et al., 2014) | 93.5 | 56.5 | N/A |
| Supervised Embeddings | 54.4 | 54.4 | 54.4 |
| Memory Network | 78.5 | 63.4 | 69.9 |
| **Key-Value Memory Network** | **93.9** | **68.3** | **76.2** |

Lecturas clave: KV-MemNN **supera a todos en las tres fuentes**. El sistema de Bordes (diseñado para KBs) va muy bien en KB pero no opera sobre documentos; KV-MemNN iguala ese rendimiento en KB (93.9) y además funciona sobre texto. Leer Wikipedia directamente (Doc, 76.2) **supera a la KB por IE** (68.3), resultado alentador hacia la lectura automática de máquinas, aunque **persiste una brecha frente a la KB anotada** (93.9 vs. 76.2). La mejor representación para documentos es "Window-level + Center Encoding + Title" ($W=7$, $H=2$): la escalera va de sentence-level 52.4 → window-level 66.8 → +Title 74.1 → +Center Encoding+Title 76.9.

El **análisis de documentos sintéticos** es el experimento más fino: se generan "Wikipedias" artificiales desde los triples de la KB con plantillas, parametrizando complejidad. Pasar de la KB a una sola oración-plantilla ya cuesta (93.9 → 82.9): representar el hecho como oración hace más difícil extraer sujeto/relación/objeto. El resto de la caída se reparte entre conjunciones (74.0) y correferencia (76.0). Finalmente, en **WikiQA** (answer sentence selection) KV-MemNN alcanza el estado del arte (MAP 0.7069, MRR 0.7265), superando a CNNs y LSTMs con atención; la MemNN clásica, que no puede emparejar ventanas con oraciones, va mucho peor.

## Limitaciones reconocidas

- **La brecha no se cierra, solo se reduce.** El modelo *acorta* la distancia (93.9 vs. 76.2), pero "alguna brecha persiste".
- **Dominio cerrado y plantillas.** WikiMovies da control experimental limpio pero introduce regularidad sintáctica que no refleja preguntas abiertas reales.
- **Feature maps diseñados a mano.** Las representaciones más potentes (Window+Title, Center Encoding) son ingeniería de rasgos específica del dataset —Window+Title es inaplicable a colecciones sin títulos significativos.
- **Dependencia del hashing.** El índice invertido por solapamiento de palabras puede dejar fuera al candidato correcto si no comparte vocabulario con la pregunta.
- **IE como cuello de botella.** Solo ~56% de los pares de IE coinciden con la KB, limitando el techo de esa ruta por la calidad de extracción, no por el modelo.

## Por qué importa para la Clase 30

La [Clase 30](/clases/clase-30) (modelos con memoria externa) ubica KV-MemNN en la secuencia Memory Networks → End-to-End → Key-Value → NTM/DNC → Entity Networks. Su aporte más duradero es conceptual y precede a una idea central del *retrieval-augmented generation* moderno: **desacoplar la representación de búsqueda de la representación de retorno**. La clave (con qué emparejas la consulta) y el valor (qué devuelves) son distintos —exactamente el patrón que hoy estructura los almacenes vectoriales y los recuperadores densos, donde se indexa por un embedding y se devuelve el pasaje original.

Para situar el paper conviene leerlo junto a sus vecinos del curso. El fundamento transversal de [redes de memoria](/fundamentos/redes-de-memoria) explica el patrón general controlador + memoria direccionable por contenido del que esta arquitectura es un caso. Y el ancestro inmediato, la [End-to-End Memory Network de Sukhbaatar et al. (2015)](/papers/e2e-memnn-sukhbaatar-2015), es literalmente el caso particular que se recupera al fijar clave = valor: leer ambos papers en orden hace visible qué agrega la separación clave-valor —flexibilidad de codificación y transformaciones no triviales entre lo que buscas y lo que devuelves.

El [Laboratorio 30](/laboratorios/lab-30) implementa este modelo de cero sobre WikiMovies y lo sondea con experimentos propios: la visualización de la atención muestra el refinamiento difuso→picudo de los 2 hops, y un análisis de errores revela que buena parte del "31% de error" es en realidad la limitación del *hashing* y del ground-truth single-answer que se enumeran arriba, no un fallo de capacidad del modelo.

## Notas y enlaces

- Preprint: arXiv:1606.03126v2 (10 oct 2016), [arxiv.org/abs/1606.03126](https://arxiv.org/abs/1606.03126).
- Dataset WikiMovies: liberado dentro del corpus bAbI de FAIR en [fb.ai/babi](http://fb.ai/babi).
- Venue: EMNLP 2016. Afiliación: Facebook AI Research, Nueva York (Jesse Dodge también en LTI, Carnegie Mellon).
