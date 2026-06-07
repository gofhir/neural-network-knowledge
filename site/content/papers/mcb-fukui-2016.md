---
title: "Multimodal Compact Bilinear Pooling (MCB)"
weight: 244
math: true
---

{{< paper-card
    title="Multimodal Compact Bilinear Pooling for VQA and Visual Grounding"
    authors="Fukui, Park, Yang, Rohrbach, Darrell, Rohrbach"
    year="2016"
    venue="EMNLP 2016"
    pdf="/papers/mcb-fukui-2016.pdf"
    arxiv="1606.01847" >}}
Propone fusionar imagen y texto mediante **bilinear pooling** (producto externo), que captura todas las interacciones multiplicativas cruzadas entre modalidades. Como el producto externo explota dimensionalmente, lo aproxima de forma tratable con **Count Sketch + FFT**: nunca materializa la matriz $x \otimes q$, sino que esketcha cada modalidad y las combina por convolución en el dominio de la frecuencia. La arquitectura usa **doble MCB** (uno para attention, otro para predecir la respuesta) y **ganó el VQA Challenge 2016**.
{{< /paper-card >}}

---

## Contexto

En Visual Question Answering ([fundamento Visual Question Answering](/fundamentos/visual-question-answering)) el sistema recibe una imagen y una pregunta en lenguaje natural y debe producir una respuesta, formulado como clasificación sobre un conjunto fijo de respuestas $A$:

$$
\hat{a} = \arg\max_{a \in A} \; p(a \mid \mathbf{x}, \mathbf{q}; \theta)
$$

El pipeline estándar de la época extrae dos representaciones independientes: una visual $x \in \mathbb{R}^{n_1}$ (salida de una CNN, en este paper ResNet-152) y una textual $q \in \mathbb{R}^{n_2}$ (salida de una RNN, una LSTM de 2 capas). El paso decisivo y foco del trabajo es el **multimodal pooling**: cómo combinar $x$ y $q$ en un vector conjunto $\Phi(x, q)$ del que un clasificador lineal pueda decidir la respuesta.

El estado del arte recurría a fusiones deliberadamente simples: **concatenación** $[\,x; q\,]$, **suma element-wise** $x + q$ o **producto element-wise (Hadamard)** $x \odot q$. La hipótesis central del paper es que estas fusiones **no son suficientemente expresivas**. Una suma o un producto element-wise solo relaciona la dimensión $i$ de la imagen con la dimensión $i$ del texto; nunca relaciona la dimensión $i$ visual con la dimensión $j \neq i$ textual. Asociaciones cruzadas como "la palabra *color* debe interactuar con los canales de la CNN que codifican tonalidad, sin importar su alineación posicional" quedan fuera de alcance.

La concatenación tampoco resuelve el problema: preserva toda la información pero la deja *yuxtapuesta*, no *combinada*. Un clasificador lineal posterior solo forma combinaciones $\sum_i w_i x_i + \sum_j w_j q_j$, sin ningún término producto $x_i q_j$. Para que aparezcan interacciones multiplicativas hay que apilar capas FC con no linealidades, y aun así el modelo debe *aprender* esas interacciones desde cero. El producto externo, en cambio, las entrega "gratis" como sesgo inductivo arquitectónico. El reto no es conceptual sino de tratabilidad computacional.

---

## Ideas principales

### Bilinear pooling y su explosión dimensional

La operación que sí captura todas las interacciones cruzadas es el **producto externo**. Dados $x \in \mathbb{R}^{n_1}$ y $q \in \mathbb{R}^{n_2}$, el modelo bilineal de Tenenbaum y Freeman (2000) calcula:

$$
z = W\,[\,x \otimes q\,], \qquad x \otimes q = x\,q^{\top} \in \mathbb{R}^{n_1 \times n_2}
$$

donde $[\,\cdot\,]$ linealiza la matriz, y $W$ es la matriz de pesos aprendida. Si escribimos la matriz de interacción $M_{ij} = x_i q_j$, las tres operaciones contrastan así:

- La **suma** descarta $M$ y solo conserva $x_i + q_i$.
- El **producto element-wise** conserva únicamente la diagonal $M_{ii} = x_i q_i$.
- El **producto externo** conserva la matriz completa $M$, los $n_1 \times n_2$ términos.

El producto externo es estrictamente más expresivo: contiene a la diagonal como caso particular y agrega todos los términos fuera de diagonal. Equivale a evaluar un kernel polinomial de grado 2 sobre la concatenación de ambas modalidades. Bilinear CNN (Lin et al., 2015) ya lo había aprovechado en reconocimiento visual de grano fino; la novedad aquí es llevarlo al caso multimodal y hacerlo tratable.

**El problema: explosión dimensional.** El vector linealizado $[\,x \otimes q\,]$ tiene dimensión $n_1 \times n_2$, y la matriz $W$ tendría tamaño $(n_1 \cdot n_2) \times z$. Con los valores de VQA — $n_1 = n_2 = 2048$ y salida $z = 3000$ — esto resulta en:

$$
2048 \times 2048 \times 3000 \approx 1.25 \times 10^{10} \;\text{parámetros (12.5 mil millones)}.
$$

Aprender y almacenar esa matriz es inviable. Por eso el bilinear pooling apenas se había usado. Se necesita obtener los beneficios del producto externo sin construirlo ni parametrizarlo explícitamente.

### Compact bilinear via Count Sketch + FFT

La solución adopta **Compact Bilinear Pooling** (Gao et al., 2016), apoyado en *Tensor Sketch* (Pham y Pagh, 2013) y *Count Sketch* (Charikar et al., 2002). La idea es proyectar el producto externo a un espacio de menor dimensión $d \ll n_1 \cdot n_2$ sin materializarlo.

**Count Sketch.** Proyecta un vector $v \in \mathbb{R}^n$ a $y \in \mathbb{R}^{d}$ usando dos vectores de hashing fijos tras su inicialización: un vector de signos $s \in \{-1, +1\}^{n}$ y un mapa $h \in \{1, \dots, d\}^{n}$. El operador $\Psi$ acumula cada componente en su bucket:

$$
y = [0, \dots, 0] \in \mathbb{R}^{d}, \qquad
\text{para cada } i:\quad y[\,h[i]\,] \mathrel{+}= s[i]\cdot v[i]
$$

Los signos aleatorios $\pm 1$ son la clave: cuando dos índices colisionan en el mismo bucket, sus contribuciones cruzadas se cancelan en esperanza, dejando $\langle \Psi(v), \Psi(v') \rangle$ como estimador insesgado de $\langle v, v' \rangle$. La varianza disminuye al aumentar $d$ (más buckets, menos colisiones).

**El producto externo en el dominio sketch es una convolución.** El resultado clave de Pham y Pagh es que el Count Sketch del producto externo equivale a la convolución de los sketches individuales:

$$
\Psi(x \otimes q,\, h,\, s) \;=\; \Psi(x,\, h,\, s) \,*\, \Psi(q,\, h,\, s)
$$

Esto evita por completo construir $x \otimes q$: basta esketchar $x$ y $q$ por separado y convolucionar.

**El truco de la FFT.** Por el teorema de convolución, la convolución en el dominio temporal es un producto element-wise en frecuencia:

$$
x' * q' \;=\; \mathrm{FFT}^{-1}\!\big(\,\mathrm{FFT}(\Psi(x)) \odot \mathrm{FFT}(\Psi(q))\,\big)
$$

La FFT de un vector de dimensión $d$ cuesta $O(d \log d)$, por lo que la operación completa es barata. Como la combinación es un producto element-wise en frecuencia, el esquema se extiende de forma natural a más de dos modalidades (basta seguir multiplicando FFTs).

El resultado neto: una aproximación del bilinear pooling con dimensión $d \approx 16000$ en lugar de $n_1 \cdot n_2 \approx 4.19$ millones, conservando la expresividad multiplicativa cruzada. MCB nunca construye el producto externo: dos sketches $O(n)$, dos FFT directas y una inversa $O(d \log d)$, y un producto element-wise. En total, MCB con $d = 16000$ usa del orden de 48 millones de parámetros, comparable a una arquitectura de concatenación + FC apilada, pero codifica interacciones que esas FC tendrían que aprender desde cero.

**Detalle de implementación.** Tras el pooling, la arquitectura aplica una **raíz cuadrada con signo** element-wise, $\mathrm{sign}(z)\sqrt{|z|}$, y luego **normalización $L_2$** (heredadas de Lin et al., 2015). El producto externo produce un rango dinámico muy amplio; estas operaciones lo comprimen y normalizan, estabilizando el entrenamiento. Omitirlas degrada notablemente el rendimiento.

### La arquitectura MCB para VQA con doble MCB

**Features.** Imagen: ResNet-152 preentrenada en ImageNet, imágenes a $448 \times 448$; para attention se usa la última capa convolucional `res5c`, un tensor $2048 \times 14 \times 14$. Texto: embedding aprendido con tanh seguido de una LSTM de 2 capas $\times$ 1024 unidades, cuyas salidas se concatenan en 2048-D.

La arquitectura completa usa **MCB dos veces**:

- **Primer MCB — predicción de attention.** Para cada una de las $14 \times 14$ posiciones espaciales, hace MCB pooling entre el slice visual de esa posición y la pregunta (tile-eada para coincidir con la grilla). El resultado $16000 \times 14 \times 14$ pasa por dos convoluciones + softmax que producen un mapa de attention. Una suma ponderada de los vectores espaciales con ese mapa da el vector visual atendido.
- **Segundo MCB — predicción de la respuesta.** El vector visual atendido y la representación textual se combinan en un segundo MCB pooling. Tras *signed square-root* + $L_2$, una capa FC proyecta el vector de 16000-D a las **3000 respuestas más frecuentes** (softmax).

Los autores también experimentan con varios **glimpses** (mapas de attention concatenados) y, para opción múltiple, un **tercer MCB** que fusiona cada candidato de respuesta codificado con la representación multimodal.

Para **visual grounding** (localizar una frase entre bounding boxes candidatas), la base es GrounderR (Rohrbach et al., 2016): la modificación consiste simplemente en **reemplazar la concatenación por MCB pooling** (con $d = 2048$, que funcionó mejor en esta tarea), más embedding lineal y $L_2$ en lugar de batch norm.

---

## Resultados experimentales

**Ablations de pooling (VQA test-dev).** Todos los métodos no bilineales reciben capas FC para igualar el presupuesto de parámetros:

| Método de pooling | Accuracy |
|---|---|
| Element-wise Sum | 56.50 |
| Concatenation | 57.49 |
| Concatenation + FC + FC | 57.10 |
| Element-wise Product | 58.57 |
| **MCB ($2048 \times 2048 \to 16K$)** | **59.83** |
| Full Bilinear ($128 \times 128 \to 16K$) | 58.46 |
| MCB ($128 \times 128 \to 4K$) | 58.69 |
| Concatenation + FC con Attention | 58.36 |
| **MCB ($d = 16K$) con Attention** | **62.50** |

Lecturas clave:

1. **MCB supera a todas las fusiones simples**: 59.83 vs 58.57 del mejor no bilineal (+1.26) con presupuesto comparable.
2. **No es solo cuestión de más parámetros**: "Concatenation + FC + FC" (~46M parámetros, equiparable a los 48M de MCB) rinde solo 57.10 vs 59.83.
3. **Compact ≈ Full Bilinear, mucho más barato**: con $128 \times 128$, MCB ($\to 4K$) logra 58.69 vs 58.46 del bilineal completo; la aproximación por sketch no degrada accuracy.
4. **MCB + Attention es donde más brilla**: atender sobre la capa MCB sube a 62.50 (vs 58.36 atendiendo sobre Concat+FC), una mejora de +2.67.

La accuracy crece con $d$ hasta saturar: $d = 16000$ da el máximo (59.83) y $d = 32000$ ya no mejora (59.71), por eso $d = 16000$ es el valor por defecto en VQA.

**VQA Challenge 2016.** Sobre VQA test-dev open-ended (All), un ensemble de 7 modelos con attention logra **66.7**, quedando 1.8 puntos por encima del segundo mejor (Naver Labs, 64.9) y muy por delante de HieCoAtt (61.8), DMN+ (60.3) o [SAN (Yang 2016)](/papers/stacked-attention-yang-2016) (58.7). Incluso sin ensemble, "MCB + Genome + Att. + GloVe" rinde 65.4. Este resultado le dio el **primer lugar en el VQA Challenge 2016** (real-image). En Visual7W, MCB + Att. supera al estado del arte previo por **7.9 puntos** (62.2 vs 54.3 promedio). En grounding, reemplazar la concatenación de GrounderR por MCB sube de 46.50 a 48.69 en Flickr30k Entities (nuevo SOTA) y de 25.48 a 28.91 en ReferItGame.

---

## Limitaciones

1. **Costo del FFT.** Aunque mucho más barato que el bilineal completo, MCB añade dos FFT directas, un producto element-wise y una FFT inversa por invocación; con attention, se aplica en cada una de las 196 posiciones espaciales.
2. **La aproximación introduce varianza.** El Count Sketch es una proyección aleatoria; la igualdad $\Psi(x \otimes q) = \Psi(x) * \Psi(q)$ se cumple en esperanza, no exactamente. Recuperar la expresividad exige $d$ grande (16000), lo que vuelve a inflar la capa de clasificación final.
3. **Sensibilidad de hiperparámetros entre tareas.** El $d$ óptimo difiere ($16000$ para VQA, $2048$ para grounding) y detalles como *signed square-root* + $L_2$ son necesarios: MCB no es un *drop-in* trivial.
4. **Superado por métodos posteriores más simples.** MLB (Kim et al., 2017) factoriza la interacción con descomposición de bajo rango logrando igual o mejor accuracy con menos parámetros y sin FFT; [MUTAN (Ben-younes 2017)](/papers/mutan-ben-younes-2017) usa descomposición de Tucker.

---

## Por qué importa hoy

MCB abrió de forma efectiva la **línea de fusión bilineal en VQA**. Antes, el pooling multimodal era un detalle casi ignorado; los modelos competían en attention, memoria o composición. MCB demostró empíricamente —con presupuestos de parámetros controlados— que cambiar solo la operación de fusión podía mover varios puntos de accuracy y ganar un challenge. Esto convirtió la fusión en un eje de diseño de primera clase.

Sus descendientes directos —MLB (bajo rango), [MUTAN (Ben-younes 2017)](/papers/mutan-ben-younes-2017) (Tucker), MFB/MFH y BLOCK— son refinamientos de la misma intuición: capturar interacciones multiplicativas cruzadas de manera tratable. El truco Count Sketch + FFT, importado de la literatura de *data streaming*, se popularizó en visión gracias a este paper.

Con la llegada de los **Transformers multimodales**, la fusión dejó de ser una única operación algebraica y pasó a ser **cross-attention aprendida capa a capa** entre tokens visuales y textuales. El cross-attention generaliza y subsume las interacciones que MCB aproximaba: cada token de una modalidad atiende a todos los de la otra, capturando interacciones cruzadas de forma aprendida y multi-nivel. La línea posterior (MLB → [Pythia (Jiang 2018)](/papers/pythia-jiang-2018)) descubrió que, con buenas features de región (bottom-up attention) y attention bien diseñada, una **fusión simple** (producto element-wise tras proyección) recupera casi toda la ganancia a una fracción del costo. Entender MCB ayuda a apreciar por qué Pythia puede permitirse fusiones simples: el trabajo pesado migró de la operación de fusión hacia las features de región y la attention. MCB queda hoy como hito histórico y pedagógico más que como técnica de producción.

---

## Notas y enlaces

- **MUTAN** ([Ben-younes 2017](/papers/mutan-ben-younes-2017)): reemplaza la aproximación por sketch con una descomposición de Tucker del tensor bilineal de tres vías $(x, q, \text{salida})$, controlando explícitamente el rango de cada modo. Siguiente eslabón natural en la cadena bilineal.
- **MLB** (Kim et al., 2017): factoriza la interacción como $\sigma(U^\top x \odot V^\top q)$ — proyección lineal de cada modalidad seguida de producto Hadamard. Puente entre el producto element-wise y el bilinear completo, y ancestro conceptual de la fusión que usa Pythia.
- **Pythia** ([Jiang 2018](/papers/pythia-jiang-2018)): base de MMF de Facebook AI, con fusión deliberadamente simple sobre features bottom-up.
- **Código oficial**: [akirafukui/vqa-mcb](https://github.com/akirafukui/vqa-mcb) — implementación en Caffe.
- **Reproducible**: ResNet-152, $448\times448$, capa `res5c` para attention, $L_2$ norm, LSTM 2 capas $\times$ 1024 (2048-D), $d = 16000$, signed square-root + $L_2$ post-MCB, clasificación sobre 3000 respuestas, Adam ($\epsilon=0.0007$), Visual Genome como datos extra y GloVe como inicialización de embeddings.

Ver: [Clase 23](/clases/clase-23) · [fundamento Visual Question Answering](/fundamentos/visual-question-answering) · [dominio Multimodal](/dominios/multimodal) · [Stacked Attention (Yang 2016)](/papers/stacked-attention-yang-2016).
