---
title: "Recurrent Entity Networks (2017)"
weight: 344
math: true
---

{{< paper-card
    title="Tracking the World State with Recurrent Entity Networks"
    authors="Mikael Henaff, Jason Weston, Arthur Szlam, Antoine Bordes, Yann LeCun"
    year="2017"
    venue="ICLR 2017"
    pdf="/papers/entity-networks-henaff-2017.pdf"
    arxiv="1612.03969" >}}
Paper de Facebook AI Research que introduce el **Recurrent Entity Network (EntNet)**, una red aumentada con memoria cuya tesis es que un agente inteligente debe **mantener un modelo del mundo** que se actualiza continuamente mientras lee, no recuperarlo solo cuando le preguntan. La memoria es una **batería de bloques** donde idealmente cada slot rastrea una **entidad** (persona, objeto, lugar) con una **clave** que la identifica y un **valor** dinámico con sus atributos. Al leer cada frase, una **compuerta** independiente por bloque decide qué entidades actualizar, lo que permite procesar el texto **online y en una sola pasada**. Es el **primer modelo en resolver las 20 tareas bAbI** en el régimen de 10k ejemplos (0 falladas, error medio 0.5%).
{{< /paper-card >}}

---

## Contexto

Hacia 2015-2016 las redes con memoria externa vivían su auge. La **Memory Network** (Weston et al., 2014) y su versión entrenable extremo a extremo, la **End-to-End Memory Network / MemN2N** ([Sukhbaatar et al., 2015](/papers/e2e-memnn-sukhbaatar-2015)), dominaban la comprensión de historias: almacenaban toda la secuencia de entrada en una matriz de memoria y, al recibir la pregunta, ejecutaban varios *hops* de atención softmax sobre ese texto crudo. En paralelo, la Neural Turing Machine (Graves et al., 2014) y el Differentiable Neural Computer (Graves et al., 2016) ofrecían memorias de tamaño fijo con lectura y escritura diferenciables gobernadas por un controlador central.

El paper diagnostica un límite común: estas familias **leen** la entrada pero no mantienen un **estado dinámico de las entidades**. Vuelcan las frases a memoria de forma pasiva y delegan todo el razonamiento al momento de la consulta. No hay una representación que evolucione frase a frase capturando "dónde está cada cosa ahora". La NTM/DNC sí reescriben su memoria, pero a través de un controlador secuencial sofisticado que concentra el razonamiento.

El EntNet propone algo más simple: en vez de un controlador central, una **batería de RNN con compuertas que comparten parámetros**, donde cada RNN gobierna su propio bloque de memoria local sin interacción directa con los demás. El paper traza una analogía elegante: así como el *weight tying* en una CNN refleja la invarianza de las estadísticas de la imagen a través de las posiciones, aquí compartir parámetros entre bloques refleja una invarianza de las **leyes del mundo** entre instancias de objetos. La misma regla ("si una persona se mueve, su ubicación cambia") aplica por igual a John, a Mary o a Sandra. Y a diferencia del softmax de la NTM/DNC (que normaliza entre memorias y por tanto las acopla), la **compuerta independiente por bloque** permite actualizar **varios slots a la vez** sin que compitan por la masa de probabilidad.

## El modelo: tres partes

El EntNet procesa datos secuenciales con un **encoder de entrada**, una **memoria dinámica** y una **capa de salida**.

**Encoder.** Resume cada frase en un vector de longitud fija mediante una **máscara multiplicativa aprendida seguida de una suma**: $s_t = \sum_i f_i \odot e_i$, donde $\odot$ es el producto de Hadamard sobre los embeddings de palabra $e_i$ y las máscaras $f_i$ se aprenden y se comparten entre pasos. Si todas las máscaras valen 1, el encoder degenera en bag-of-words; también puede aprender un *positional encoding* al estilo MemN2N, ponderando posiciones (por ejemplo el sujeto o el verbo de movimiento).

**Memoria dinámica.** Es el corazón del modelo. El estado oculto se divide en bloques $h_1, \dots, h_m$ (en los experimentos $m$ va de 5 a 20), cada uno con un vector clave asociado $w_j$. En cada paso, dado el input codificado $s_t$, cada bloque se actualiza con cuatro ecuaciones:

$$g_j \leftarrow \sigma(s_t^\top h_j + s_t^\top w_j) \quad \text{(gate)}$$
$$\tilde{h}_j \leftarrow \phi(U h_j + V w_j + W s_t) \quad \text{(candidato)}$$
$$h_j \leftarrow h_j + g_j \odot \tilde{h}_j \quad \text{(update)}$$
$$h_j \leftarrow h_j / \lVert h_j \rVert \quad \text{(normalización)}$$

Cada pieza tiene un rol preciso:

- **El gate $g_j$** (con $\sigma$ la sigmoide) decide cuánto actualizar el bloque $j$. Tiene **dos términos**: el de **contenido** $s_t^\top h_j$ abre la compuerta para los slots cuyo *valor actual* coincide con la entrada (direccionamiento por contenido); el de **localización** $s_t^\top w_j$ la abre para los slots cuya *clave* coincide con la entrada (direccionamiento por clave). Así se "encuentra" la entidad correcta tanto por su nombre como por lo que ya sabemos de ella.
- **El candidato $\tilde{h}_j$** es el nuevo valor propuesto, combinando contenido previo, clave y entrada vía las matrices $U, V, W$ (**compartidas entre todos los bloques** — el *weight tying*). La activación $\phi$ puede ser PReLU o la identidad.
- **El update** suma el candidato ponderado por el gate: si $g_j \approx 0$, el bloque queda intacto; solo las entidades relevantes se modifican.
- **La normalización a la esfera unitaria** habilita el **olvido**: como toda la información vive en la *fase* (dirección) del vector, sumarle cualquier otro vector reduce su distancia coseno con el estado previo, de modo que lo viejo se olvida a medida que entra lo nuevo.

**Capa de salida.** Ante una consulta $q$, se aplica atención sobre las memorias: $p_j = \text{Softmax}(q^\top h_j)$, $u = \sum_j p_j h_j$, $y = R\,\phi(q + H u)$. El paper observa que este módulo es, literalmente, **una MemN2N de un solo hop** con una no-linealidad extra. Todo el modelo se entrena por backpropagation through time.

## El ejemplo motivador: Mary, el balón y el jardín

El modelo es libre de aprender las claves; la elección natural es **asociar un slot a cada entidad**. Esto puede inyectarse como conocimiento previo atando (*tying*) las claves a embeddings de palabras concretas (la **variante "tied"**, que además maneja entidades no vistas en entrenamiento). Dos frases consecutivas lo ilustran:

- *"Mary picked up the ball."* Se activan los gates de "Mary" **y** de "ball" gracias al término de localización $s_t^\top w_j$ (por clave). La entrada de "Mary" registra que porta el balón; la de "ball", que está siendo portado por Mary.
- *"Mary went to the garden."* Se modifica "Mary" (ahora en el jardín) y también "ball" (su nueva ubicación). Crucialmente, "ball" **no aparece** en esta frase; su gate igual se activa gracias al término de **contenido** $s_t^\top h_j$, porque la información sobre Mary quedó guardada en la memoria de "ball" en el paso anterior.

Con los pesos adecuados, la memoria queda en un estado donde "Where is the ball?" o "Where is Mary?" se responden leyendo directamente el slot, **sin razonamiento adicional**. Esa es la promesa: convertir el razonamiento secuencial costoso (en el momento de la pregunta) en una lectura simple sobre un estado ya mantenido. La [Clase 30](/clases/clase-30) cierra exactamente con este ejemplo, en su versión "Mary / cocina / manzana".

## Experimentos

**Tarea sintética de modelo del mundo.** Dos agentes se mueven al azar en una grilla 10×10 durante $T$ pasos; el modelo debe dar las ubicaciones de ambos al final. Exige combinar hasta $T-2$ hechos de soporte.

| Modelo | T=10 | T=20 | T=40 |
|---|---|---|---|
| MemN2N | 0.09 | 0.633 | 0.896 |
| LSTM | 0 | 0.157 | 0.226 |
| **EntNet** | **0** | **0** | **0** |

El MemN2N se degrada rápido con la longitud, el LSTM aguanta mejor pero también pierde precisión, y el EntNet (con solo 5 slots) resuelve todos los casos. Más aún, **generaliza más allá de su horizonte de entrenamiento**: entrenado con secuencias de longitud 1 a 20, mantiene error casi nulo hasta T=50 — evidencia de que aprendió la *dinámica* del mundo, no a memorizar trayectorias.

**Tareas bAbI (el resultado estelar).** Las 20 tareas [bAbI](/papers/babi-weston-2015) (Weston et al., 2015) son el benchmark estándar de razonamiento sobre historias (hechos múltiples, conteo, negación, deducción, inducción, búsqueda de caminos, etc.). Con la versión 10k, embeddings de 100-d y 20 slots de memoria:

| Modelo | Tareas falladas (>5%) | Error medio |
|---|---|---|
| NTM | 16 | 20.1 |
| D-NTM | 9 | 12.8 |
| MemN2N | 3 | 4.2 |
| DNC | 2 | 3.8 |
| DMN+ | 1 | 2.8 |
| **EntNet** | **0** | **0.5** |

El EntNet es el **primer modelo en resolver las 20 tareas**. El salto más espectacular es la **tarea 16 (basic induction)**: todos los competidores fallan estrepitosamente (45-55% de error) mientras el EntNet logra 0.2%. No es uniformemente el mejor en cada tarea (en la 3 obtiene 4.1% frente al 1.1% del DMN+), pero su consistencia general lo distingue.

Un **análisis de interpretabilidad** confirma la tesis: midiendo la distancia coseno entre cada slot y la matriz decodificadora, se ve que cada memoria guarda la **ubicación correcta** de su entidad al final de la historia — no solo la respuesta a la pregunta, sino también las entidades sobre las que nadie preguntó. El modelo construye un estado del mundo completo, no un resumen oportunista.

**Children's Book Test (CBT).** Completar una palabra faltante tras leer 20 frases de libros infantiles. Con una variante simplificada (sin normalización, claves atadas a los 10 candidatos), el EntNet **supera a todos los modelos single-pass** en Named Entities (0.616) y Common Nouns (0.588), leyendo en una sola pasada. Persiste una brecha frente a los modelos multi-pass de comprensión lectora más sofisticados (AoA Reader, NSE).

## Limitaciones reconocidas

- **Eficiencia de muestras.** El gran resultado (20/20) es solo con **10k ejemplos**. Con 1k el rendimiento cae fuerte (15 tareas falladas, error 29.6%, peor que MemN2N); los autores admiten que bAbI-1k "sigue siendo un problema abierto".
- **La variante "tied" no ayuda en general.** Atar claves a embeddings hace el modelo más interpretable, pero no mejoró el rendimiento en la mayoría de las tareas.
- **Brecha multi-pass en CBT.** En comprensión lectora real, los modelos con múltiples pasadas de atención condicionadas por la pregunta siguen por delante.
- **Solo texto.** La arquitectura es general, pero aquí solo se aplicó a texto. Queda como trabajo futuro combinar el *tracking* del EntNet con modelos predictivos que anticipen la evolución del mundo.

## Por qué importa para la Clase 30

La [Clase 30](/clases/clase-30) ("Modelos con memoria externa") recorre la evolución de la memoria en redes neuronales y **cierra precisamente con las Entity Networks**. El EntNet es la culminación de una línea que parte de las tareas [bAbI](/papers/babi-weston-2015) como benchmark de comprensión de historias y de la [End-to-End Memory Network](/papers/e2e-memnn-sukhbaatar-2015), de la que hereda el encoder posicional y el módulo de salida (una MemN2N de un solo hop con una no-linealidad extra) pero a la que supera con su idea distintiva: **mantener estado dinámico de entidades** en vez de solo almacenar y atender texto crudo.

El fundamento transversal de [redes de memoria](/fundamentos/redes-de-memoria) provee el marco común — memorias externas, direccionamiento por contenido y por localización, lectura/escritura diferenciables — dentro del cual el EntNet se ubica como el modelo que estructura la memoria por entidades y actualiza varios slots en paralelo. Los seis pasos que enseña la clase mapean uno a uno con el método: codificar el input (máscara + suma), decidir qué entidades actualizar (gate con términos de contenido y localización), proponer información (candidato), actualizar, normalizar (olvido) y atender las memorias para responder.

Su lectura conceptual es la que resume el módulo: en lugar de una memoria genérica direccionable como un archivo, una **memoria estructurada por entidades** donde cada slot rastrea un objeto del mundo y se actualiza con compuertas independientes en paralelo. Esta concepción de *world modeling* —mantener vivo un estado del mundo que evoluciona con cada percepto— anticipa preocupaciones centrales de la IA contemporánea sobre seguimiento de estado en agentes, razonamiento sobre entidades y memoria persistente.

## Notas y enlaces

- arXiv: https://arxiv.org/abs/1612.03969 (v3, 10 may 2017)
- Código (Torch): https://github.com/facebook/MemNN/tree/master/EntNet-babi
- Venue: ICLR 2017.
- Afiliación: Facebook AI Research; Courant Institute, NYU.
