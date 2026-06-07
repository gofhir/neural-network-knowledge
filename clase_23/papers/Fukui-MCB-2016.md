---
título: "Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding"
autores: "Akira Fukui, Dong Huk Park, Daylen Yang, Anna Rohrbach, Trevor Darrell, Marcus Rohrbach"
afiliaciones: "UC Berkeley EECS; Sony Corp. (Tokyo); Max Planck Institute for Informatics (Saarbrücken)"
venue: "EMNLP 2016"
año: 2016
arxiv: "1606.01847"
link: "https://arxiv.org/abs/1606.01847"
código: "https://github.com/akirafukui/vqa-mcb"
tags: ["VQA", "fusión multimodal", "bilinear pooling", "Count Sketch", "FFT", "visual grounding", "attention"]
---

# Multimodal Compact Bilinear Pooling (MCB) — Fukui et al., 2016

> **Cita.** Akira Fukui, Dong Huk Park, Daylen Yang, Anna Rohrbach, Trevor Darrell, Marcus Rohrbach. *Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding*. EMNLP 2016. arXiv:1606.01847. Código: `https://github.com/akirafukui/vqa-mcb`.
>
> Las marcas `*` en la lista de autores indican contribución equitativa (Fukui, Park, Yang, A. Rohrbach). El trabajo fue financiado por DARPA, AFRL, DoD MURI (N000141110688), NSF (IIS-1427425, IIS-1212798) y el Berkeley Artificial Intelligence Research (BAIR) Lab. Ganó el **VQA Challenge 2016** (categoría real-image, open-ended).

---

## 1. Contexto — el problema de la fusión multimodal en VQA

En *Visual Question Answering* (VQA) el sistema recibe una imagen y una pregunta en lenguaje natural y debe producir una respuesta. La formulación canónica del paper trata el problema como una clasificación sobre un conjunto fijo de respuestas $A$:

$$
\hat{a} = \arg\max_{a \in A} \; p(a \mid \mathbf{x}, \mathbf{q}; \theta)
$$

donde $\mathbf{x}$ es la imagen, $\mathbf{q}$ la pregunta y $\theta$ los parámetros del modelo. El pipeline estándar de la época extrae dos representaciones vectoriales independientes:

- Una representación visual $x = \Xi(\mathbf{x}) \in \mathbb{R}^{n_1}$, típicamente la salida de una CNN (en este paper, ResNet-152).
- Una representación textual $q = \Omega(\mathbf{q}) \in \mathbb{R}^{n_2}$, típicamente la salida de una RNN (en este paper, una LSTM de 2 capas).

El paso decisivo — y el foco del trabajo — es el **multimodal pooling**: cómo combinar $x$ y $q$ en un vector conjunto $\Phi(x, q)$ que capture la relación entre ambas modalidades, de modo que un clasificador lineal sobre $\Phi$ pueda decidir la respuesta.

El estado del arte previo recurría a operaciones de fusión deliberadamente simples:

- **Concatenación** $[\,x; q\,]$, opcionalmente seguida de capas totalmente conectadas (FC).
- **Suma element-wise** $x + q$ (requiere $n_1 = n_2$).
- **Producto element-wise (Hadamard)** $x \odot q$ (requiere $n_1 = n_2$).

Estas operaciones aparecen en prácticamente todos los modelos contemporáneos citados por el paper: el baseline iBOWIMG (Zhou et al., 2015) concatena; Stacked Attention Networks (Yang et al., 2015) y Spatial Memory Networks (Xu et al., 2015) terminan fusionando con suma o producto element-wise; D-NMN (Andreas et al., 2016a) y DMN (Xiong et al., 2016) usan producto y suma element-wise; DPPnet (Noh et al., 2015) predice dinámicamente los pesos de una capa visual a partir de la pregunta.

La hipótesis central del paper es que **estas fusiones simples no son suficientemente expresivas**. Una suma o un producto element-wise solo relaciona la dimensión $i$ de la imagen con la dimensión $i$ del texto; nunca relaciona la dimensión $i$ visual con la dimensión $j \neq i$ textual. Las asociaciones cruzadas entre modalidades — por ejemplo, que la palabra "color" en la pregunta deba interactuar con los canales de la CNN que codifican tonalidad, sin importar su alineación posicional — quedan fuera del alcance de estas operaciones. Los autores proponen una fusión que captura *todas* las interacciones multiplicativas entre features visuales y textuales.

Conviene precisar por qué la concatenación, pese a ser la opción "más completa" a primera vista, tampoco resuelve el problema. La concatenación $[\,x; q\,]$ preserva toda la información de ambos vectores, pero la deja *yuxtapuesta*, no *combinada*: un clasificador lineal posterior solo puede formar combinaciones lineales $\sum_i w_i x_i + \sum_j w_j q_j$ de las dos modalidades, sin ningún término producto $x_i q_j$. Para que aparezcan interacciones multiplicativas hay que apilar capas FC con no linealidades, y aun así el modelo debe *aprender* esas interacciones implícitamente a través de muchos parámetros y muchos datos, en lugar de que la arquitectura las provea por construcción. Esta es la observación que motiva todo el paper: el producto externo entrega las interacciones cruzadas "gratis", como sesgo inductivo arquitectónico, en vez de delegarlas a capas densas que deben descubrirlas. El reto, entonces, no es conceptual — el producto externo es la operación obviamente correcta — sino puramente de tratabilidad computacional.

Vale notar también el espectro de trabajos relacionados que el paper sitúa alrededor de su contribución. Por un lado están los *joint multimodal embeddings* basados en Canonical Correlation Analysis (Hardoon et al., 2004), modelos lineales con *ranking loss* (Frome et al., 2013; Karpathy y Fei-Fei, 2015) o modelos no lineales profundos (Kiros et al., 2014; Ngiam et al., 2011); todos buscan un espacio compartido pero modelan principalmente *similitud* entre modalidades, no *interacción* fina. Por otro lado, DPPnet (Noh et al., 2015) y HieCoAtt (Lu et al., 2016) ya exploraban interacciones multiplicativas — el primero prediciendo dinámicamente pesos de una capa visual desde la pregunta, el segundo con co-attention jerárquica — pero ninguno usaba el producto externo completo. MCB se posiciona como una operación *complementaria* a todos ellos: cualquiera de esos enfoques de embedding podría beneficiarse de incorporar las interacciones que MCB hace tratables.

---

## 2. Bilinear pooling — la idea del producto externo

La operación que sí captura todas las interacciones cruzadas es el **producto externo** (*outer product*). Dados $x \in \mathbb{R}^{n_1}$ y $q \in \mathbb{R}^{n_2}$, el modelo bilineal de Tenenbaum y Freeman (2000) calcula:

$$
z = W\,[\,x \otimes q\,], \qquad x \otimes q = x\,q^{\top} \in \mathbb{R}^{n_1 \times n_2}
$$

donde $\otimes$ denota el producto externo, $[\,\cdot\,]$ linealiza la matriz $n_1 \times n_2$ en un vector, y $W$ es la matriz de pesos lineal aprendida. El producto externo produce todos los productos por pares $x_i \, q_j$. A diferencia del producto element-wise — que solo conserva los términos diagonales $x_i \, q_i$ — el producto externo retiene la matriz completa de coproductos, permitiendo que cada feature visual interactúe multiplicativamente con cada feature textual. Esta es exactamente la expresividad que buscaban los autores.

El bilinear pooling ya había demostrado su valor en reconocimiento visual de grano fino (Lin et al., 2015, *Bilinear CNN models*), donde dos CNN se combinan vía producto externo conectado a una capa de salida. La intuición geométrica es ilustrativa: si pensamos en $x$ y $q$ como vectores en sus respectivos espacios, el producto externo $x q^{\top}$ es la matriz de todos los productos de coordenadas, equivalente a evaluar un *kernel polinomial* de grado 2 sobre la concatenación de ambas modalidades. Capturar productos de grado 2 entre features es precisamente lo que distingue una representación que "entiende" co-ocurrencias (este objeto rojo + esta pregunta sobre color) de una que solo suma evidencias independientes.

Es útil contrastar las tres operaciones sobre la misma base. Si escribimos la matriz de interacción $M_{ij} = x_i q_j$:

- La **suma** descarta $M$ por completo y solo conserva $x_i + q_i$ (la diagonal aditiva).
- El **producto element-wise** conserva únicamente la diagonal $M_{ii} = x_i q_i$.
- El **producto externo** conserva la matriz completa $M$, es decir los $n_1 \times n_2$ términos.

De ahí que el producto externo sea estrictamente más expresivo: contiene a la diagonal como un caso particular y agrega todos los términos fuera de diagonal. Lin et al. lo aprovecharon en visión pura; la novedad de este paper es llevarlo al caso *multimodal* (visión + lenguaje) y, sobre todo, hacerlo tratable.

**El problema: explosión dimensional.** El vector linealizado $[\,x \otimes q\,]$ tiene dimensión $n_1 \times n_2$. Para aprender la salida $z \in \mathbb{R}^{z}$ se necesita una matriz $W$ de tamaño $(n_1 \cdot n_2) \times z$. Con los valores que el paper usa para VQA — $n_1 = n_2 = 2048$ y dimensión de salida $z = 3000$ — esto resulta en:

$$
2048 \times 2048 \times 3000 \approx 1.25 \times 10^{10} \;\text{parámetros (12.5 mil millones)}.
$$

Como señalan los autores textualmente, "$W$ thus would have 12.5 billion parameters, which leads to very high memory consumption and high computation times". Aprender y almacenar esa matriz es inviable. Por eso, "given their high dimensionality ($n^2$), bilinear pooling has so far not been widely used". Se necesita una forma de obtener los beneficios del producto externo sin construirlo ni parametrizarlo explícitamente.

---

## 3. Compact Bilinear Pooling — Count Sketch y el truco de la FFT

La solución adopta la técnica de **Compact Bilinear Pooling** de Gao et al. (2016), que a su vez se apoya en el algoritmo *Tensor Sketch* de Pham y Pagh (2013) y en el *Count Sketch* de Charikar et al. (2002). La idea es **proyectar el producto externo a un espacio de menor dimensión** $d \ll n_1 \cdot n_2$ sin materializarlo nunca.

### 3.1 Count Sketch

El Count Sketch proyecta un vector $v \in \mathbb{R}^n$ a un vector $y \in \mathbb{R}^{d}$. Se inicializan dos vectores de hashing, constantes tras su inicialización aleatoria uniforme:

- $s \in \{-1, +1\}^{n}$ — un vector de signos.
- $h \in \{1, \dots, d\}^{n}$ — un mapa que asigna cada índice de entrada $i$ a un índice de salida $j = h[i]$.

El operador de proyección $\Psi(v, h, s)$ se calcula así (Algorithm 1, líneas 12-16):

$$
y = [0, \dots, 0] \in \mathbb{R}^{d}, \qquad
\text{para cada } i \in \{1, \dots, n\}:\quad y[\,h[i]\,] \mathrel{+}= s[i]\cdot v[i]
$$

Es decir, cada componente $v[i]$ se suma (con su signo $s[i]$) en el bucket $h[i]$ del vector de salida. Es una proyección dispersa, aleatoria, que preserva en esperanza los productos internos. La razón por la que funciona como aproximación es la propiedad fundamental del Count Sketch: si $y = \Psi(v)$ e $y' = \Psi(v')$ comparten los mismos $(h, s)$, entonces $\langle y, y' \rangle$ es un estimador insesgado de $\langle v, v' \rangle$. Los signos aleatorios $\pm 1$ son la clave: cuando dos índices distintos colisionan en el mismo bucket ($h[i] = h[j]$, $i \neq j$), sus contribuciones cruzadas tienen signo independiente y se cancelan en esperanza, dejando solo los términos correctos. La varianza de la estimación disminuye al aumentar $d$ (más buckets, menos colisiones), lo que explica directamente la curva de la Tabla 2: subir $d$ de 1024 a 16000 reduce el ruido del sketch y sube la accuracy.

### 3.2 El producto externo en el dominio sketch es una convolución

El resultado clave de Pham y Pagh (2013) es que **el Count Sketch del producto externo de dos vectores equivale a la convolución de sus Count Sketches individuales**:

$$
\Psi(x \otimes q,\, h,\, s) \;=\; \Psi(x,\, h,\, s) \,*\, \Psi(q,\, h,\, s)
$$

donde $*$ es el operador de convolución. Esto evita por completo construir la matriz $x \otimes q$: basta con esketchar $x$ y $q$ por separado (cada uno a dimensión $d$) y convolucionar los dos sketches.

### 3.3 El truco de la FFT

La convolución sigue siendo costosa en el dominio temporal. Aquí entra el **teorema de convolución**: la convolución en el dominio temporal es un producto element-wise en el dominio de la frecuencia. Por lo tanto:

$$
x' * q' \;=\; \mathrm{FFT}^{-1}\!\big(\,\mathrm{FFT}(x') \odot \mathrm{FFT}(q')\,\big)
$$

donde $x' = \Psi(x)$, $q' = \Psi(q)$, $\odot$ es el producto element-wise y $\mathrm{FFT}$ la transformada rápida de Fourier. La FFT de un vector de dimensión $d$ cuesta $O(d \log d)$, por lo que la operación completa es barata.

### 3.4 El módulo MCB completo

Juntando las tres piezas, el módulo Multimodal Compact Bilinear (Algorithm 1) toma $v_1 \in \mathbb{R}^{n_1}$ y $v_2 \in \mathbb{R}^{n_2}$ y devuelve $\Phi(v_1, v_2) \in \mathbb{R}^{d}$:

1. Inicializa (una sola vez) los pares de hashing $(h_k, s_k)$ para $k \in \{1, 2\}$, muestreando $h_k[i]$ de $\{1, \dots, d\}$ y $s_k[i]$ de $\{-1, +1\}$.
2. Esketcha cada modalidad: $v'_k = \Psi(v_k, h_k, s_k)$.
3. Combina vía FFT:

$$
\Phi \;=\; \mathrm{FFT}^{-1}\!\big(\,\mathrm{FFT}(v'_1) \odot \mathrm{FFT}(v'_2)\,\big)
$$

Una observación importante de los autores: la combinación ocurre como producto element-wise en el dominio de la frecuencia, por lo que **el esquema se extiende de forma natural a más de dos modalidades** (basta seguir multiplicando FFTs). En este paper se usan $v_1 = x$ (imagen) y $v_2 = q$ (texto).

El resultado neto: se obtiene una aproximación del bilinear pooling con dimensión $d \approx 16000$ en lugar de $n_1 \cdot n_2 \approx 4.2$ millones, conservando la expresividad multiplicativa cruzada a un costo computacional y de memoria mucho menor.

**Comparación de costos.** El contraste cuantitativo es esclarecedor. El bilinear completo requiere materializar un vector de $n_1 \cdot n_2 = 2048 \times 2048 \approx 4.19$ millones de dimensiones y una matriz $W$ de 12.5 mil millones de parámetros. MCB nunca construye el producto externo: hace dos sketches ($O(n)$ cada uno, simples acumulaciones en buckets), dos FFT directas y una inversa ($O(d \log d)$), y un producto element-wise ($O(d)$). La capa de clasificación final opera sobre $d = 16000$ dimensiones, no sobre 4.19 millones, lo que reduce los parámetros de la capa de salida en más de dos órdenes de magnitud. En total, MCB con $d = 16000$ usa del orden de 48 millones de parámetros — comparable a una arquitectura de concatenación + FC apilada — pero codifica interacciones que esas FC tendrían que aprender desde cero. Este es el argumento que las ablaciones de la Sección 6.3 validan empíricamente: misma cantidad de parámetros, más accuracy.

**Detalle de implementación que importa.** Tras el MCB pooling, la arquitectura aplica dos transformaciones heredadas de la literatura de bilinear pooling (Lin et al., 2015): una **raíz cuadrada con signo** element-wise, $\text{sign}(z)\sqrt{|z|}$, y luego **normalización $L_2$**. El producto externo (y su aproximación) produce valores con rango dinámico muy amplio — algunos productos $x_i q_j$ dominan a otros por órdenes de magnitud — y estas dos operaciones comprimen ese rango y normalizan la escala, estabilizando el entrenamiento. Omitir estos pasos degrada notablemente el rendimiento en la práctica, y son parte de por qué MCB no es un módulo trivial de insertar.

---

## 4. La arquitectura MCB para VQA

### 4.1 Extracción de features

- **Imagen.** ResNet-152 (He et al., 2015) preentrenada en ImageNet. Las imágenes se redimensionan a $448 \times 448$. Se usa la salida de la capa "pool5" (antes del clasificador de 1000 vías) y se aplica normalización $L_2$ sobre el vector de 2048 dimensiones. Para attention se usa en cambio la última capa convolucional (`res5c`), que entrega un tensor espacial de $2048 \times 14 \times 14$.
- **Texto.** La pregunta se tokeniza en palabras, cada palabra se codifica one-hot y pasa por una capa de embedding aprendida con no linealidad tanh. Sigue una LSTM de 2 capas con 1024 unidades por capa; las salidas de ambas capas se concatenan en un vector de 2048 dimensiones.

### 4.2 MCB con attention (doble MCB)

La arquitectura completa (Figura 3) usa **MCB dos veces**:

**Primer MCB — predicción de attention.** Para cada una de las $14 \times 14$ posiciones de la grilla espacial visual, se hace MCB pooling entre el slice de feature visual de esa posición (2048-D) y la representación de la pregunta. La pregunta se *tile*-ea para coincidir con la grilla. El resultado es un tensor de $16000 \times 14 \times 14$, sobre el cual dos capas convolucionales seguidas de softmax producen un mapa de attention normalizado. Una suma ponderada de los vectores espaciales con ese mapa produce el vector visual atendido (2048-D).

**Segundo MCB — predicción de la respuesta.** El vector visual atendido y la representación textual se combinan en un segundo MCB pooling. Tras MCB se aplica una **raíz cuadrada con signo** (*signed square-root*) element-wise seguida de **normalización $L_2$** — pasos heredados de la literatura de bilinear pooling para estabilizar la magnitud. Una capa totalmente conectada proyecta el vector multimodal de 16000-D a las 3000 respuestas más frecuentes (clasificación softmax).

**Múltiples "glimpses".** Los autores experimentan con generar varios mapas de attention (varios "glimpses"), cuyos vectores atendidos se concatenan antes del segundo MCB. La inspección visual sugiere que múltiples mapas producen un efecto de *ensembling*/suavizado.

### 4.3 Answer encoding (multiple-choice — tercer MCB)

Para VQA de opción múltiple, donde hay candidatos de respuesta de longitud variable (Figura 4), cada candidato se codifica con una capa de embedding y capas LSTM cuyos pesos se comparten entre candidatos. Se añade un **tercer MCB pooling** para fusionar la respuesta codificada con la representación multimodal del pipeline original, proyectando luego a un vector de clasificación con dimensión igual al número de respuestas.

---

## 5. Visual Grounding con MCB

La segunda tarea del paper es **visual grounding** (localización de frases): dada una frase en lenguaje natural y una imagen con múltiples *bounding boxes* candidatos, predecir cuál caja corresponde a la frase.

La base es la versión totalmente supervisada de **GrounderR** (Rohrbach et al., 2016). En GrounderR original, la representación visual de cada caja propuesta se *concatena* con el embedding de la frase para predecir pesos de attention sobre las propuestas. La modificación de los autores es directa: **reemplazar esa concatenación por MCB pooling** (Figura 5). A diferencia de GrounderR, aquí se incluye un *embedding* lineal de la representación visual y normalización $L_2$ de ambas entradas (en lugar de *batch normalization*), que resultó beneficioso al usar MCB. Para grounding se usa $d = 2048$ en el MCB pooling, que funcionó mejor que valores mayores en esta tarea.

Una frase se considera localizada correctamente si la caja predicha solapa con la *ground-truth* por más del 50% de IOU (*intersection over union*).

---

## 6. Experimentos

### 6.1 Datasets

- **VQA (Antol et al., 2015)** — ~200 000 imágenes de MSCOCO, 3 preguntas por imagen, 10 respuestas por pregunta. Splits: train (80K imágenes), val (40K), test (80K); además un subconjunto test-dev (25% del test). La mayoría de los experimentos de ablación se reportan en test-dev, tarea *open-ended real-image*.
- **Visual Genome (Krishna et al., 2016)** — 108 249 imágenes (intersección de YFCC100M y MSCOCO), ~17 pares QA por imagen, 1.7 millones de pares QA de los 6 tipos (*what, where, when, who, why, how*). Se usa como **datos de entrenamiento adicionales**: se eliminan palabras de relleno ("a", "the", "it is") para acortar respuestas, se filtran a respuestas de una palabra y al vocabulario de VQA, quedando ~1M de tripletas imagen-QA adicionales.
- **Visual7W (Zhu et al., 2016)** — parte de Visual Genome; agrega una 7.ª categoría de pregunta (*which*). Se evalúa la tarea *Telling* (6W). 47 300 imágenes de MSCOCO, 139 868 pares QA en formato de opción múltiple (cuatro candidatos, uno correcto).
- **Flickr30k Entities (Plummer et al., 2015)** — 31K imágenes, 244K frases con cajas. Para grounding.
- **ReferItGame (Kazemzadeh et al., 2014)** — 20K imágenes (IAPR TC-12), 120K expresiones referenciales. Para grounding.

### 6.2 Setup de entrenamiento

Optimizador Adam con $\epsilon = 0.0007$, $\beta_1 = 0.9$, $\beta_2 = 0.999$ (para VQA; para grounding $\epsilon = 0.0001$). Dropout tras las capas LSTM y FC. *Early stopping*: si el score de validación no mejora en 50 000 iteraciones, se detiene. En grounding, embedding de 500 dimensiones para visual y lenguaje, $d = 2048$, criterio de localización IOU > 50%.

### 6.3 Ablaciones (Tabla 1)

El experimento central compara métodos de pooling, todos entrenados en VQA train y evaluados en test-dev. Para que la comparación sea justa, a los métodos no bilineales se les añaden capas FC (4096 unidades, ReLU, dropout) para igualar el presupuesto de parámetros.

| Método de pooling | Accuracy (test-dev) |
|---|---|
| Element-wise Sum | 56.50 |
| Concatenation | 57.49 |
| Concatenation + FC | 58.40 |
| Concatenation + FC + FC | 57.10 |
| Element-wise Product | 58.57 |
| Element-wise Product + FC | 56.44 |
| Element-wise Product + FC + FC | 57.88 |
| **MCB ($2048 \times 2048 \to 16K$)** | **59.83** |
| Full Bilinear ($128 \times 128 \to 16K$) | 58.46 |
| MCB ($128 \times 128 \to 4K$) | 58.69 |
| Element-wise Product con VGG-19 | 55.97 |
| MCB ($d = 16K$) con VGG-19 | 57.05 |
| Concatenation + FC con Attention | 58.36 |
| **MCB ($d = 16K$) con Attention** | **62.50** |

Lecturas clave:

1. **MCB supera a todas las fusiones no bilineales.** 59.83 frente al mejor no bilineal (Element-wise Product, 58.57) — una ganancia de **+1.26 puntos** con presupuesto de parámetros comparable.
2. **No es solo cuestión de más parámetros.** "Concatenation + FC + FC" tiene $\approx 4096^2 + 4096^2 + 4096 \times 3000 \approx 46$ millones de parámetros, equiparable a los 48 millones de MCB con $d = 16000$; aun así rinde solo 57.10 frente a 59.83 de MCB.
3. **Compact ≈ Full Bilinear, mucho más barato.** Con dimensiones reducidas a $128 \times 128$ para que el bilineal completo sea factible, MCB ($\to 4K$) logra 58.69 vs 58.46 del bilineal completo: la aproximación por sketch no degrada la accuracy y sí reduce drásticamente el costo.
4. **MCB ayuda independiente de la CNN.** Con VGG-19, MCB (57.05) supera al producto element-wise (55.97).
5. **MCB + Attention es donde más brilla.** Atender sobre la capa de Concatenación+FC rinde 58.36 (igual que no atender), mientras que atender sobre la capa MCB sube a 62.50, una **mejora de +2.67 puntos** que confirma que el módulo MCB es el lugar correcto para predecir attention.

### 6.4 Dimensión del sketch (Tabla 2)

| Compact Bilinear $d$ | Accuracy (test-dev) |
|---|---|
| 1024 | 58.38 |
| 2048 | 58.80 |
| 4096 | 59.42 |
| 8192 | 59.69 |
| **16000** | **59.83** |
| 32000 | 59.71 |

La accuracy crece con $d$ hasta saturar; $d = 16000$ da el máximo (59.83) y $d = 32000$ ya no mejora (incluso baja levemente a 59.71). Por eso $d = 16000$ es la elección por defecto en VQA.

---

## 7. Resultados numéricos

### 7.1 Comparación con el estado del arte en VQA (Tabla 4)

Sobre el VQA test set (modelos entrenados en train+val), accuracy en %:

| Modelo | Test-dev Open-Ended (All) | Test-dev MC (All) |
|---|---|---|
| MCB | 60.8 | 65.4 |
| MCB + Genome | 62.3 | 66.4 |
| MCB + Att. | 64.2 | 68.6 |
| MCB + Att. + GloVe | 64.7 | 69.1 |
| MCB + Att. + Genome | 65.1 | 69.5 |
| MCB + Att. + GloVe + Genome | 65.4 | 69.9 |
| **Ensemble de 7 modelos Att.** | **66.7** | **70.2** |

En test-standard, el ensemble logra 66.5 (open-ended, All) y 70.1 (MC, All), con desglose Y/N = 83.2, No. = 39.5, Other = 58.0.

Comparación con competidores y rivales del challenge (test-dev, open-ended All):

| Modelo | Open-Ended All |
|---|---|
| **Ensemble de 7 (este trabajo)** | **66.7** |
| Naver Labs (2.º del challenge) | 64.9 |
| HieCoAtt (Lu et al., 2016) | 61.8 |
| DMN+ (Xiong et al., 2016) | 60.3 |
| FDA (Ilievski et al., 2016) | 59.2 |
| D-NMN (Andreas et al., 2016a) | 59.4 |
| SAN (Yang et al., 2015) | 58.7 |
| VQA team (Antol et al., 2015) | 57.8 |
| DPPnet (Noh et al., 2015) | 57.2 |
| iBOWIMG (Zhou et al., 2015) | 55.7 |

El ensemble queda **1.8 puntos por encima del siguiente mejor** en open-ended y 0.8 en multiple-choice (test-dev). Incluso sin ensemble, el modelo "MCB + Genome + Att. + GloVe" rinde 65.4 vs 64.9 del segundo mejor (+0.5). Este resultado le dio el **primer lugar en el VQA Challenge 2016** (real-image).

### 7.2 Visual7W (Tabla 3)

Tarea de opción múltiple, accuracy (%) por categoría sobre Visual7W test:

| Método | What | Where | When | Who | Why | How | Avg |
|---|---|---|---|---|---|---|---|
| Zhu et al. | 51.5 | 57.0 | 75.0 | 59.5 | 55.5 | 49.8 | 54.3 |
| Concat + Att. | 47.8 | 56.9 | 74.1 | 62.3 | 52.7 | 51.2 | 52.8 |
| **MCB + Att.** | **60.3** | **70.4** | **79.5** | **69.2** | **58.2** | 51.1 | **62.2** |

MCB + Att. supera al estado del arte previo (Zhu et al.) por **7.9 puntos** en promedio y gana en casi todas las categorías.

### 7.3 Visual Grounding (Tablas 5 y 6)

**Flickr30k Entities:**

| Método | Accuracy (%) |
|---|---|
| Plummer et al. (2015) | 27.42 |
| Hu et al. (2016b) | 27.80 |
| Wang et al. (2016) | 43.89 |
| Rohrbach et al. (2016) — GrounderR | 47.81 |
| Concatenation | 46.50 |
| Element-wise Product | 47.41 |
| Element-wise Product + Conv | 47.86 |
| **MCB** | **48.69** |

**ReferItGame:**

| Método | Accuracy (%) |
|---|---|
| Hu et al. (2016b) | 17.93 |
| Rohrbach et al. (2016) | 26.93 |
| Concatenation | 25.48 |
| Element-wise Product | 27.80 |
| Element-wise Product + Conv | 27.98 |
| **MCB** | **28.91** |

En ambos datasets, reemplazar la concatenación de GrounderR por MCB mejora consistentemente: en Flickr30k de 46.50 (concat) a 48.69 (MCB), nuevo estado del arte; en ReferItGame de 25.48 a 28.91. La progresión concat → producto element-wise → MCB confirma el patrón de las ablaciones de VQA: la fusión más expresiva gana, incluso con menos parámetros.

---

## 8. Limitaciones

1. **Costo y complejidad del FFT.** Aunque mucho más barato que el bilineal completo, el módulo MCB añade dos FFT directas, un producto element-wise y una FFT inversa por cada invocación. En la arquitectura con attention, MCB se aplica en cada una de las 196 posiciones espaciales y luego para la predicción, sumando sobrecarga no trivial respecto a una suma o producto element-wise.
2. **La aproximación introduce varianza.** El Count Sketch es una proyección aleatoria; la igualdad $\Psi(x \otimes q) = \Psi(x) * \Psi(q)$ se cumple en esperanza, no exactamente. Dimensiones de sketch bajas degradan la calidad (Tabla 2: $d = 1024$ da 58.38 vs 59.83 con $d = 16000$); recuperar la expresividad exige $d$ grande (16000), lo que vuelve a inflar el tamaño de la capa de clasificación final.
3. **Sensibilidad de hiperparámetros entre tareas.** El $d$ óptimo difiere por tarea ($d = 16000$ para VQA, $d = 2048$ para grounding), y detalles como *signed square-root* + $L_2$, o sustituir *batch norm* por $L_2$ en grounding, son necesarios para que MCB funcione bien — señal de que el módulo no es un *drop-in* trivial.
4. **Superado por métodos posteriores más simples.** El propio enfoque bilineal pronto fue refinado por **MLB** (Multimodal Low-rank Bilinear, Kim et al., 2017), que factoriza la interacción bilineal con descomposición de bajo rango (producto Hadamard sobre proyecciones lineales) logrando igual o mejor accuracy con muchos menos parámetros y sin FFT; y por **MUTAN** (Ben-younes et al., 2017), que usa descomposición de Tucker. Más tarde, los modelos basados en *Transformers* y *cross-attention* (LXMERT, ViLBERT, UNITER y la familia de Pythia/MMF) desplazaron por completo a la fusión bilineal explícita.

---

## 9. Impacto y legado

MCB abrió de forma efectiva la **línea de investigación de "fusión bilineal" en VQA**. Su contribución conceptual perdura más que su implementación concreta:

- **La fusión importa.** Antes de MCB, el pooling multimodal era un detalle casi ignorado; los modelos competían en attention, memoria o composición. MCB demostró empíricamente — con presupuestos de parámetros controlados — que cambiar solo la operación de fusión, manteniendo todo lo demás constante, podía mover varios puntos de accuracy y ganar un challenge. Esto convirtió la fusión en un eje de diseño de primera clase.
- **Descendientes directos.** MLB (factorización de bajo rango), MUTAN (descomposición de Tucker), **MFB/MFH** (Multi-modal Factorized Bilinear pooling) y **BLOCK** (block-superdiagonal fusion) son todos refinamientos de la misma intuición: capturar interacciones multiplicativas cruzadas de manera tratable. Cada uno mejoró la eficiencia o la expresividad del producto externo aproximado de MCB.
- **Difusión del truco Count Sketch + FFT.** La técnica de aproximar productos externos vía sketches y convolución en frecuencia, importada de Pham-Pagh y Gao et al., se popularizó en visión gracias a este paper, mostrando que un resultado clásico de *data streaming* podía habilitar arquitecturas de deep learning.
- **Reemplazo por cross-attention.** Con la llegada de los Transformers multimodales, la fusión dejó de ser una única operación algebraica y pasó a ser **atención cruzada aprendida capa a capa** entre tokens visuales y textuales. El cross-attention generaliza y subsume las interacciones que MCB aproximaba: cada token de una modalidad atiende a todos los de la otra, capturando interacciones cruzadas de forma aprendida y multi-nivel. MCB queda hoy como un hito histórico y pedagógico más que como una técnica de producción.

---

## 10. Conexión con la Clase 23 (VQA y Pythia)

La clase 23 trabaja VQA con **Pythia** (la base de MMF de Facebook AI), cuyo módulo de fusión es deliberadamente **simple**: combina las features visuales atendidas (de *bottom-up attention* / Faster R-CNN) con la representación de la pregunta mediante operaciones como **producto element-wise (Hadamard)** o **dot-product** tras proyecciones lineales, seguidas de un clasificador. Pythia heredó esta receta del ganador del VQA Challenge 2017 (Teney et al.), que mostró que con buenas features (bottom-up region features) y attention bien diseñada, una fusión simple basta y es más eficiente.

MCB ofrece el **contraste pedagógico** ideal:

| Eje | MCB (2016) | Pythia (clase 23) |
|---|---|---|
| Features visuales | Grilla CNN (ResNet `res5c`, $14\times14$) | Regiones bottom-up (Faster R-CNN) |
| Fusión | Bilinear pooling compacto (Count Sketch + FFT), $d=16000$ | Producto element-wise / dot-product tras proyección |
| Interacciones capturadas | Todas las cruzadas $x_i q_j$ (aproximadas) | Solo diagonales $x_i q_i$ (sobre proyecciones) |
| Costo | FFT por posición, capa final de 16000-D | Lineal, barato |
| Veredicto histórico | Ganó VQA 2016; luego superado | Receta moderna pragmática |

La lección transversal: **la elección de la operación de fusión multimodal es una decisión de diseño con consecuencias medibles**, no un detalle. MCB apostó por máxima expresividad algebraica (producto externo); la línea posterior (MLB/Pythia) descubrió que, con features y attention suficientemente buenas, una fusión más simple recupera casi toda la ganancia a una fracción del costo. Entender MCB ayuda a apreciar *por qué* Pythia puede permitirse fusiones simples: el trabajo pesado migró de la operación de fusión hacia las features de región y la attention.

---

## 11. Notas y enlaces

- **Relación con MUTAN** (también en este curso). MUTAN (Ben-younes et al., 2017) reemplaza la aproximación por sketch de MCB con una **descomposición de Tucker** del tensor bilineal de tres vías $(x, q, \text{salida})$, controlando explícitamente el rango de cada modo. Es el siguiente eslabón natural en la cadena bilineal: donde MCB aproxima el producto externo de forma aleatoria, MUTAN lo factoriza de forma estructurada y aprendida.
- **Relación con MLB.** Multimodal Low-rank Bilinear (Kim et al., 2017) factoriza la interacción como $\sigma(U^\top x \odot V^\top q)$ — una proyección lineal de cada modalidad seguida de producto Hadamard. Es esencialmente un puente entre el "producto element-wise" de las ablaciones de MCB y el bilinear completo, y es conceptualmente el ancestro de la fusión que usa Pythia.
- **Bottom-up attention** (Anderson et al., 2018). Cambió el paradigma de features visuales de grilla (como las de MCB) a features de región (objetos detectados), lo que hizo a Pythia y sus sucesores más precisos y permitió fusiones más simples.
- **Código oficial.** `https://github.com/akirafukui/vqa-mcb` — implementación en Caffe del módulo MCB y las arquitecturas de VQA.
- **Detalle reproducible.** Para VQA: ResNet-152, imágenes $448\times448$, capa `res5c` ($2048\times14\times14$) para attention, $L_2$ norm, LSTM 2 capas $\times$ 1024 unidades concatenadas (2048-D), $d = 16000$, signed square-root + $L_2$ post-MCB, clasificación sobre las 3000 respuestas más frecuentes, Adam ($\epsilon=0.0007$), Visual Genome como datos extra y GloVe como inicialización de embeddings en los mejores modelos.
