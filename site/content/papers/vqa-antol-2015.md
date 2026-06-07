---
title: "VQA: Visual Question Answering"
weight: 240
math: true
---

{{< paper-card
    title="VQA: Visual Question Answering"
    authors="Antol, Agrawal, Lu, Mishra, Zitnick, Batra, Parikh"
    year="2015"
    venue="ICCV 2015"
    pdf="/papers/vqa-antol-2015.pdf"
    arxiv="1505.00468" >}}
Paper fundacional que **define la tarea de Visual Question Answering**: dada una imagen y una pregunta en lenguaje natural, producir una respuesta también en lenguaje natural. Introduce el **dataset VQA v1** (~0,76 M preguntas sobre imágenes COCO y escenas abstractas), la **métrica de consenso humano** $\text{acc}=\min(\#\text{humanos}/3,\,1)$ y la familia de **baselines LSTM+CNN** que dominó el subcampo por años. Su hallazgo más influyente: un modelo "ciego" (solo lenguaje) alcanza ~49 % de accuracy, revelando el problema de los *language priors*.
{{< /paper-card >}}

---

## Contexto

A mediados de 2015 la intersección visión-lenguaje estaba dominada por el *image captioning*, presentado como "un paso hacia resolver la IA". Los autores observan algo incómodo: una CNN profunda emparejada con estadísticas de n-gramas basta para generar descripciones plausibles **sin entender realmente la imagen**, explotando regularidades del lenguaje (las playas mencionan arena, las cocinas platos). El captioning no era tan *AI-complete* como se creía, y su evaluación libre era notoriamente mala (BLEU, METEOR y CIDEr correlacionan pobremente con el juicio humano).

De ahí los tres criterios que proponen para una tarea de próxima generación: requerir **conocimiento multimodal** genuino, tener una **métrica cuantitativa bien definida** y ser **automáticamente evaluable** sin jueces humanos cada vez. VQA cumple los tres: es un **Test de Turing visual** donde, como las respuestas tienden a ser cortas, la tarea sigue siendo abierta y rica pero se evalúa por coincidencia con respuestas humanas. El paper enmarca aplicaciones concretas: asistir a personas con discapacidad visual o permitir consultas activas sobre contenido visual. Ver [Clase 23](/clases/clase-23) y el [fundamento Visual Question Answering](/fundamentos/visual-question-answering).

---

## Ideas principales

### La tarea VQA

Un sistema recibe una **imagen** $I$ y una **pregunta en lenguaje natural** $q$ de forma libre y abierta, y produce una **respuesta** $a$. El paper define dos modalidades de evaluación:

1. **Open-ended (abierta):** el sistema genera una respuesta libre; en la práctica se implementa como clasificación sobre las $K$ respuestas más frecuentes.
2. **Multiple-choice:** el sistema elige entre 18 respuestas candidatas por pregunta. Más fácil de evaluar y útil para algoritmos que no generan texto libre.

Las preguntas cubren todo el espectro de la IA: reconocimiento de grano fino ("¿qué tipo de queso tiene la pizza?"), detección ("¿cuántas bicicletas hay?"), reconocimiento de actividad, razonamiento sobre base de conocimiento ("¿es vegetariana?") y sentido común ("¿esta persona está esperando compañía?"). Para análisis, las respuestas se agrupan en una taxonomía que se volvió canónica:

| Tipo | Descripción | Ejemplo |
|---|---|---|
| **Yes/No** | Binaria (a veces "maybe") | "¿Está rota la pizza?" |
| **Number** | Una cantidad | "¿Cuántas porciones hay?" |
| **Other** | Todo lo demás (colores, objetos, lugares) | "¿De qué color son sus ojos?" |

Casi todas las tablas del campo reportan accuracy desagregada en estas tres columnas más el "All".

### El dataset VQA v1

Combina dos fuentes de imágenes, una decisión metodológica clave:

- **Imágenes reales (MS COCO):** 204.721 imágenes elegidas por sus escenas con múltiples objetos y contexto rico. Ver [COCO (Lin 2014)](/papers/coco-lin-2014).
- **Escenas abstractas (clipart):** 50.000 escenas generadas con clipart. La motivación es elegante: eliminan la visión de bajo nivel (segmentación, detección ruidosa) y permiten enfocarse en el **razonamiento de alto nivel** sin que el cuello de botella sea reconocer píxeles. Incluyen 20 modelos humanos tipo "paperdoll" con 8 expresiones, más de 100 objetos y 31 animales con poses ajustables.

El proceso de recolección por Amazon Mechanical Turk produjo cifras a gran escala: **3 preguntas por imagen**, **~0,76 M preguntas** y **~10 M respuestas** en total. La decisión más influyente fue recoger **10 respuestas de 10 trabajadores únicos por pregunta** (sin que el que respondiera fuera el que la escribió). ¿Por qué 10? Porque las preguntas abiertas producen **discrepancias legítimas**: ante "¿de qué color es la mesa?", "white", "tan" y "off-white" pueden ser todas correctas. Diez respuestas permiten modelar esa distribución y construir una métrica robusta por consenso. Se les pidió frases breves y objetivas, más una autoevaluación de confianza ("no"/"maybe"/"yes").

Los splits siguen a COCO (train/val/test), con el test subdividido en test-dev, test-standard (el oficial para papers), test-challenge y test-reserve (para detectar overfitting). Las escenas abstractas usan 20K/10K/20K.

**Hallazgos del análisis del dataset.** La distribución de tipos y longitudes de pregunta es casi idéntica entre imágenes reales y abstractas (validando las escenas abstractas como sustituto de razonamiento). Las respuestas son extremadamente cortas: el **89,32 %** son de una sola palabra en imágenes reales, justamente lo que vuelve fiable la coincidencia exacta. Las preguntas yes/no son ~38 % de las respuestas, con sesgo hacia "yes" (58,83 %); las de tipo number son ~12 %, con "2" como respuesta más común (26 %). Cuando humanos responden **sin ver la imagen**, aciertan solo ~21 % en preguntas no-yes/no, lo que confirma que la información visual es crítica. El acuerdo inter-humano (Question+Image) es 83,30 % en reales y 87,49 % en abstractas.

### La métrica de consenso

Para la tarea open-ended, la accuracy de una respuesta predicha es:

$$
\text{acc} = \min\left(\frac{\#\,\text{humanos que dieron esa respuesta}}{3},\ 1\right)
$$

Es decir, una respuesta es **100 % correcta si al menos 3 de los 10 anotadores la dieron exactamente**; 1 anotador da $\approx 0{,}33$, 2 dan $\approx 0{,}67$. Antes de comparar, todo se normaliza (minúsculas, números a dígitos, sin puntuación ni artículos). El diseño es deliberado:

1. **Robustez ante discrepancias legítimas:** varios colores pueden ser correctos; exigir coincidencia con *una* referencia sería injusto.
2. **Evita métricas blandas problemáticas:** rechazan Word2Vec ("agrupa palabras que queremos distinguir, como 'left' y 'right'") y BLEU/ROUGE (solo fiables en oraciones largas; con 89 % de respuestas de una palabra degeneran a coincidencia exacta).
3. **Comparable con el techo humano:** las accuracies de máquina se promedian sobre los $\binom{10}{9}$ subconjuntos de 9 anotadores, evitando que el modelo "vea" sus propias referencias.

Para multiple-choice se construyen **18 candidatas** desde cuatro fuentes: **Correct**, **Plausible** (respuestas dadas sin ver la imagen, plausibles por sentido común), **Popular** (las 10 respuestas más frecuentes del dataset, que dificultan inferir el tipo de pregunta) y **Random**.

### Los baselines LSTM+CNN

Una batería diseñada para revelar **de dónde viene el rendimiento**. Baselines simples: *random*, *prior ("yes")*, *per Q-type prior* (responde lo más popular por tipo: "2" para "How many", "white" para "What color") y *nearest neighbor*.

El modelo neuronal tiene **dos canales**. El de imagen toma las activaciones de 4096-dim de **VGGNet** (congelada, preentrenada en ImageNet), opcionalmente normalizadas en $\ell_2$. El de pregunta tiene tres variantes: *BoW Q* (bag-of-words), *LSTM Q* (una capa) y *deeper LSTM Q* (dos capas). El **mejor modelo, deeper LSTM Q + norm I**, funciona así:

1. La pregunta pasa palabra a palabra por una **LSTM de dos capas** → embedding de 1024-dim.
2. La imagen pasa por VGGNet, se normaliza en $\ell_2$ y se proyecta a 1024-dim con una FC + tanh.
3. **Fusión por producto elemento a elemento** de ambos embeddings: el corazón del modelo, una interacción multiplicativa más expresiva que la concatenación.
4. Un **MLP** (2 capas, dropout 0,5) y un **softmax sobre $K=1000$ respuestas** (cubren el 82,67 % del train+val).

Se entrena end-to-end con cross-entropy, pero VGGNet permanece **congelada**: el canal visual es un extractor fijo, lo que limita cuánto se adapta a la tarea y contribuye a la dependencia del lenguaje.

---

## Resultados experimentales

Resultados en test-dev sobre imágenes reales (accuracy %):

| Método | OE All | OE Yes/No | OE Number | OE Other | MC All |
|---|---|---|---|---|---|
| prior ("yes") | 29,66 | 70,81 | 00,39 | 01,15 | 29,66 |
| per Q-type prior | 37,54 | 71,03 | 35,77 | 09,38 | 39,45 |
| nearest neighbor | 42,70 | 71,89 | 24,36 | 21,94 | 48,49 |
| **I** (solo imagen) | 28,13 | 64,01 | 00,42 | 03,77 | 30,53 |
| **LSTM Q** (solo lenguaje) | 48,76 | 78,20 | 35,68 | 26,59 | 54,75 |
| LSTM Q + I | 53,74 | 78,94 | 35,24 | 36,42 | 57,17 |
| **deeper LSTM Q + norm I** | **57,75** | **80,50** | **36,77** | **43,08** | **62,70** |

En test-standard el mejor modelo logra **58,16 % open-ended / 63,09 % multiple-choice**, frente al **techo humano de 83,30 %**. Esa brecha de ~25 puntos confirma, en 2015, que VQA está lejos de resolverse: justo la propiedad *AI-complete* buscada.

El hallazgo que más impactó: **los modelos "ciegos" funcionan demasiado bien**. El modelo de solo imagen (28,13 %) rinde *peor* que el prior trivial "yes" (29,66 %), pero los modelos de solo lenguaje alcanzan **48,76 %** ignorando la imagen, superando incluso al nearest neighbor que sí la usa. Agregar la imagen al mejor modelo de lenguaje solo sube de ~49 % a 57,75 %. Desagregando por tipo, las features visuales globales casi no aportan en preguntas de razonamiento ("Is the...", "How many...") y sí ayudan en preguntas de escena ("What sport...", "What animal..."). El modelo reconoce objetos comunes pero es malo contando: el conteo es un talón de Aquiles persistente.

Las ablaciones son ejemplares: la **fusión multiplicativa supera a la concatenación** en +0,95 % OE con la mitad de parámetros, justificando empíricamente la decisión central; ampliar $K$ tiene rendimientos decrecientes; y truncar palabras raras de las preguntas no perjudica. Ningún ajuste del canal visual mueve la aguja tanto como cambiar el de lenguaje, otra señal del peso desproporcionado de los priors lingüísticos.

---

## El problema de los language priors

Es, retrospectivamente, el legado teórico más importante del paper, aunque los autores lo presentan casi de pasada. Que un modelo "ciego" alcance casi 49 % revela una patología estructural del dataset: construido con humanos, **hereda los sesgos del mundo y del lenguaje**. Los plátanos suelen ser amarillos, así que "¿de qué color es el plátano?" → "yellow" casi siempre acierta; "How many..." se responde "2" el 26 % de las veces; "Is there a..." casi siempre tiene respuesta "yes".

Un modelo puede **memorizar la distribución condicional** $P(\text{respuesta} \mid \text{tipo de pregunta})$ y obtener buen accuracy **sin haber aprendido a ver**, lo que invalida la métrica como medida de comprensión visual genuina. La tarea diseñada como "visual" se resuelve parcialmente como NLP puro de priors estadísticos.

Esto es exactamente lo que motiva a [VQAv2 (Goyal 2017)](/papers/vqav2-goyal-2017): construir, para cada pregunta, **pares de imágenes complementarias con respuestas distintas** (la misma pregunta de color emparejada con dos imágenes de respuesta diferente). Al balancear el dataset, el prior de lenguaje deja de funcionar y los modelos se ven obligados a *mirar*. La línea conceptual es directa: Antol 2015 **descubre** el problema → VQA v2 lo **corrige en los datos** → [Pythia (Jiang 2018)](/papers/pythia-jiang-2018) y los modelos modernos lo **combaten en la arquitectura** con atención y features de detección tipo [Bottom-Up Attention (Anderson 2018)](/papers/bottom-up-attention-anderson-2018).

---

## Limitaciones

1. **Preguntas abiertas, no específicas de dominio.** Para asistencia a personas ciegas o deportes convendría recolectar preguntas del dominio; observan que las preguntas reales de usuarios ciegos rara vez se responden con captions.
2. **Coincidencia exacta sin sinónimos ni plurales.** "couch"/"sofa", "1"/"one" cuentan como distintas, deprimiendo artificialmente el acuerdo inter-humano (<76 % en preguntas no binarias).
3. **Conteo débil.** El mejor modelo rinde mal en cantidades altas; el conteo exige razonamiento espacial que las features globales de VGGNet no capturan.
4. **VGGNet congelada.** El canal visual no se adapta a la tarea, limitando el aprovechamiento de la imagen.
5. **Sesgos del dataset.** Los autores los señalan pero no los corrigen en v1; lo presentan más como observación interesante que como defecto, algo que la comunidad reevaluaría con dureza después.

---

## Por qué importa hoy

El impacto es difícil de exagerar: decenas de miles de citas y uno de los papers más influyentes de la intersección visión-lenguaje. Ver [dominio Multimodal](/dominios/multimodal).

- **Institucionalizó un subcampo.** Los autores montaron un **VQA Challenge anual con workshop** (primero en CVPR 2016) y leaderboards públicos, replicando el modelo de éxito de ImageNet. VQA pasó de idea a un área con cientos de papers por año: atención (Stacked Attention Networks, co-atención), fusión bilineal (MCB, MLB, MUTAN), features bottom-up basadas en detección y módulos de razonamiento (CLEVR, Neural Module Networks).
- **Definió estándares duraderos.** La taxonomía yes/no/number/other, la métrica $\min(\#/3,\,1)$ y el baseline de dos canales (LSTM + CNN + fusión + MLP + softmax) son el punto de partida pedagógico desde el cual se construye hacia modelos con atención.
- **Linaje hasta los VLMs modernos.** La evolución es nítida: Antol VQA (2015) → atención y fusión (SAN, Bottom-Up, Pythia, 2016-2018) → Transformers multimodales (ViLBERT, LXMERT, UNITER, 2019-2021) → VLMs generativos (BLIP-2, Flamingo, GPT-4V, Gemini, Claude con visión, 2022+). Lo notable es que estos modelos gigantes resuelven, conceptualmente, **la misma tarea** que Antol et al. definieron. El benchmark VQA sigue siendo prueba estándar para evaluar VLMs.

El paper no solo creó un dataset: creó una **forma de pensar la comprensión visual** como un problema de pregunta-respuesta.

---

## Notas y enlaces

**Papers relacionados clave:**

- **Goyal et al. (2017)** — VQA v2 balanceado con pares de imágenes complementarias; corrección directa del problema de language priors. Ver [VQAv2 (Goyal 2017)](/papers/vqav2-goyal-2017).
- **Jiang et al. (2018)** — Pythia v0.1, ganador del VQA Challenge 2018, sobre VQA v2 con features bottom-up. Ver [Pythia (Jiang 2018)](/papers/pythia-jiang-2018).
- **Anderson et al. (2018)** — features de detección (Faster R-CNN) + atención, base de Pythia. Ver [Bottom-Up Attention (Anderson 2018)](/papers/bottom-up-attention-anderson-2018).
- **Yang et al. (2016)** — Stacked Attention Networks, primera atención visual influyente para VQA.
- **Johnson et al. (2017)** — CLEVR, dataset diagnóstico de razonamiento composicional.

**Recursos:**

- Sitio oficial y challenge: [visualqa.org](https://www.visualqa.org) · [visualqa.org/challenge.html](https://visualqa.org/challenge.html)
- Demo en CloudCV: [cloudcv.org/vqa](http://cloudcv.org/vqa)
- Dataset base de imágenes: MS COCO ([cocodataset.org](https://cocodataset.org))

**Cifras clave para recordar:**

- ~204.721 imágenes COCO + 50.000 escenas abstractas; ~0,76 M preguntas, ~10 M respuestas.
- 3 preguntas por imagen, 10 respuestas por pregunta; 89,32 % de respuestas son de una sola palabra.
- Métrica: $\text{acc}=\min(\#\text{humanos}/3,\,1)$.
- Mejor modelo (deeper LSTM Q + norm I): 58,16 % OE / 63,09 % MC (test-standard); techo humano 83,30 %.
- Modelo "ciego" (LSTM Q sin imagen): 48,76 % OE → evidencia de los language priors.

Ver fundamentos: [Visual Question Answering](/fundamentos/visual-question-answering) · [Mecanismo de Atención](/fundamentos/mecanismo-atencion) · [Redes Convolucionales](/fundamentos/redes-convolucionales) · [Sequence to Sequence](/fundamentos/seq2seq).
