---
title: "MUTAN: Multimodal Tucker Fusion (VQA)"
weight: 246
math: true
---

{{< paper-card
    title="MUTAN: Multimodal Tucker Fusion for Visual Question Answering"
    authors="Ben-younes, Cadene, Cord, Thome"
    year="2017"
    venue="ICCV 2017"
    pdf="/papers/mutan-ben-younes-2017.pdf"
    arxiv="1705.06676" >}}
MUTAN propone una **fusión bilineal multimodal** entre imagen y pregunta parametrizada por una **descomposición de Tucker** del tensor de interacción de tercer orden. En vez de almacenar la interacción bilineal completa (intratable, $\sim 10^{10}$ parámetros), factoriza el tensor en proyecciones por modalidad más un tensor núcleo de baja dimensión, y añade una **restricción de rango** sobre las slices del núcleo. Con ello controla explícitamente el trade-off entre expresividad de la fusión y número de parámetros, y demuestra que MCB y MLB son casos particulares suyos. Fue state-of-the-art en el dataset VQA en 2017.
{{< /paper-card >}}

---

## Contexto

En [Visual Question Answering](/fundamentos/visual-question-answering) (Antol et al. 2015) la arquitectura es siempre la misma: un extractor visual produce $v \in \mathbb{R}^{d_v}$ (ResNet-152), un codificador de texto produce $q \in \mathbb{R}^{d_q}$ (GRU con inicialización Skip-thoughts), y un **módulo de fusión** los combina en un vector que pasa por softmax sobre las 2000 respuestas más frecuentes. El cuello de botella científico de 2015–2017 fue precisamente ese módulo de fusión. MUTAN se inscribe en la línea de **fusión bilineal** y la cierra. Es contenido central de la [Clase 23](/clases/clase-23).

La progresión histórica de la fusión:

- **Primer orden (lineal).** Modelos como IMG+BOW concatenan o suman $v$ y $q$. Solo capturan correlaciones de primer orden: nunca aprenden que "la dimensión $i$ de la imagen importa *cuando* la dimensión $j$ de la pregunta está activa". En MUTAN esto es la baseline **Concat** (58.91 % en *test-dev*).

- **Bilineal completa (segundo orden).** Captura *todas* las correlaciones cruzadas vía el producto bilineal $y = (\mathcal{T} \times_1 q) \times_2 v$, con $\mathcal{T} \in \mathbb{R}^{d_q \times d_v \times |\mathcal{A}|}$. Es la fusión más expresiva posible a segundo orden, pero con $d_v \approx d_q \approx 2048$ y $|\mathcal{A}| \approx 2000$ el tensor tiene $\sim 10^{10}$ parámetros ($\sim 32$ GB en float32, más que la VRAM de las GPUs de la época). Intratable.

- **MCB — Multimodal Compact Bilinear pooling** ([Fukui 2016](/papers/mcb-fukui-2016)). Aproxima la interacción bilineal proyectando el producto externo $q \otimes v$ a baja dimensión con **count-sketch** y FFT. Ganó el VQA Challenge 2016. Debilidad: sus parámetros de interacción están **fijos** (vectores aleatorios congelados), lo que obliga a una dimensión de salida enorme ($t_o \approx 16000$).

- **MLB — Multimodal Low-rank Bilinear pooling** (Kim et al. 2017). Restringe el tensor a **rango bajo** $R$. Alcanza SOTA con muchos menos parámetros que MCB (7.7 M vs 32 M). Debilidad: la fusión se reduce a un **producto de Hadamard** en un espacio común. Aprende buenas proyecciones monomodales, pero la fusión en sí es pobre.

El problema persistente que MUTAN ataca: **¿cómo controlar la complejidad de la interacción bilineal manteniendo —o aumentando— su expresividad, en vez de sacrificar una por la otra?**

---

## Ideas principales

### El tensor de interacción bilineal

VQA busca $\hat{a} = \arg\max_{a \in \mathcal{A}} p_\Theta(a \mid v, q)$. Tras embeber imagen y pregunta, la fusión bilineal completa produce los logits:

$$y = (\mathcal{T} \times_1 q) \times_2 v, \qquad \mathcal{T} \in \mathbb{R}^{d_q \times d_v \times |\mathcal{A}|}$$

Componente a componente, el logit de la respuesta $k$ es

$$y[k] = \sum_{i=1}^{d_q} \sum_{j=1}^{d_v} \mathcal{T}[i, j, k]\, q[i]\, v[j],$$

una **forma bilineal distinta para cada respuesta**: la matriz $\mathcal{T}[:,:,k]$ pondera todos los pares $(q[i], v[j])$. El operador modo-$n$ ($\times_n$) contrae el tensor con un vector a lo largo del eje $n$: $\mathcal{T} \times_1 q$ contrae la dimensión de la pregunta y $\times_2 v$ la de la imagen, dejando $y \in \mathbb{R}^{|\mathcal{A}|}$. El tamaño $d_q d_v |\mathcal{A}| \approx 10^{10}$ es lo que hay que domar.

### La descomposición de Tucker

La respuesta es la **descomposición de Tucker** (Tucker 1966), que expresa el tensor como un **tensor núcleo** $\mathcal{T}_c$ pequeño multiplicado modo a modo por tres **matrices de factores**:

$$\mathcal{T} = \big( (\mathcal{T}_c \times_1 W_q) \times_2 W_v \big) \times_3 W_o = [\![\, \mathcal{T}_c \,;\, W_q,\, W_v,\, W_o \,]\!]$$

con $W_q \in \mathbb{R}^{d_q \times t_q}$, $W_v \in \mathbb{R}^{d_v \times t_v}$, $W_o \in \mathbb{R}^{|\mathcal{A}| \times t_o}$ y $\mathcal{T}_c \in \mathbb{R}^{t_q \times t_v \times t_o}$. Geométricamente, $W_q, W_v, W_o$ **proyectan** cada modo (pregunta, imagen, salida) a una dimensión latente más pequeña, y el núcleo $\mathcal{T}_c$ codifica las interacciones entre las versiones proyectadas.

El número de parámetros pasa de $d_q d_v |\mathcal{A}|$ (intratable) a

$$\underbrace{d_q t_q}_{W_q} + \underbrace{d_v t_v}_{W_v} + \underbrace{|\mathcal{A}| t_o}_{W_o} + \underbrace{t_q t_v t_o}_{\mathcal{T}_c}.$$

Las tres dimensiones latentes $t_q, t_v, t_o$ son **perillas independientes**: cada una gobierna cuánta complejidad se permite a cada modalidad. Esto deja modelar imagen y lenguaje con complejidades distintas ($t_q \neq t_v$), algo que MLB prohíbe. Sustituyendo Tucker en la fusión, las proyecciones se aplican **antes** de la interacción:

$$\tilde{q} = \tanh(q^\top W_q), \quad \tilde{v} = \tanh(v^\top W_v), \quad z = (\mathcal{T}_c \times_1 \tilde{q}) \times_2 \tilde{v}, \quad y = z^\top W_o.$$

Cada componente tiene un rol interpretable: $W_q, W_v$ fijan la complejidad de cada modalidad; $\mathcal{T}_c$ modela la interacción; $W_o$ es el clasificador final.

### La restricción de rango

El núcleo todavía cuesta $t_q t_v t_o$ parámetros (crece cúbicamente). MUTAN impone que **cada slice del núcleo tenga rango a lo más $R$**, escribiéndola como suma de $R$ matrices de rango 1. Esto reduce la salida a una suma de $R$ fusiones tipo Hadamard:

$$z = \sum_{r=1}^{R} z_r, \qquad z_r = (\tilde{q}^\top M_r) * (\tilde{v}^\top N_r),$$

donde $*$ es el producto de Hadamard. Cada $z_r$ es exactamente la fusión de MLB; MUTAN suma $R$ de ellas. Los autores lo leen como **compuertas lógicas**: $z_r[k]$ funciona como un AND entre "$\tilde{q}$ se parece a $m_r^k$" y "$\tilde{v}$ se parece a $n_r^k$", y la suma sobre $r$ actúa como un OR. Cada salida es así una **disyunción de $R$ conjunciones** entre patrones de pregunta e imagen —intuición que más tarde conectaría con la atención multi-cabeza.

### MUTAN generaliza MLB y MCB

El aporte teórico más citado: **MCB y MLB son casos particulares de Tucker**, cada uno con restricciones distintas sobre $\{\mathcal{T}_c, W_q, W_v, W_o\}$.

| Modelo | $W_q, W_v$ | Núcleo $\mathcal{T}_c$ | Dims latentes | Interacción | Params |
|---|---|---|---|---|---|
| **MCB** | diagonales fijas $\{-1,1\}$ | disperso fijo (hash) | $t_o \approx 16000$ | bilineal aproximada, params fijos | 32 M |
| **MLB** | aprendidos | identidad (fijo) | $t_q{=}t_v{=}t_o{=}R$ | Hadamard en espacio común | 7.7 M |
| **MUTAN** | aprendidos | aprendido, rango $R$ por slice | $t_q, t_v, t_o$ libres | bilineal completa estructurada | 4.9 M |

- **MCB** fija $W_q, W_v$ como matrices diagonales con coeficientes aleatorios congelados en $\{-1,1\}$, y el núcleo es un hash disperso fijo; solo aprende $W_o$. Como las interacciones están fijadas al azar, necesita $t_o$ gigante para que alguna sea útil.
- **MLB** es una descomposición canónica (CP) de rango $R$: núcleo igual a la identidad, $t_q = t_v = t_o = R$. Aprende las proyecciones, pero no *la interacción en sí* (queda como Hadamard). Además, forzar $t_q = t_v$ impide complejidades distintas por modalidad.
- **MUTAN** aprende **los cuatro componentes**, con dimensiones latentes independientes y la interacción bilineal estructurada (no eliminada) por la sparsity de rango. Es el caso general. Notable: logra la interacción **más expresiva** con el **menor número de parámetros** (4.9 M), porque la estructura algebraica es la fuente de esa eficiencia.

Para la arquitectura final, MUTAN se integra en **atención visual multi-glimpse**: en vez de promediar las 196 regiones del mapa $14 \times 14$ de ResNet-152, usa la propia fusión MUTAN para puntuar la relevancia de cada región y producir una suma ponderada. Cualitativamente, apagar todas las proyecciones latentes salvo una muestra que cada una se especializa en un concepto (en "¿dónde está la mujer?", una atiende al elefante y otra a la mujer).

---

## Resultados experimentales

**Dataset VQA** (sobre MS-COCO): 248 349 pares de entrenamiento, 121 512 de validación, 244 302 de test, con 10 respuestas humanas por pregunta. La métrica es graduada por consenso: $\mathrm{Acc}(\hat{a}) = \min(1,\, \#\{\text{humanos que dieron } \hat{a}\}/3)$. Setup: imágenes $448 \times 448$, ResNet-152 ($14 \times 14 \times 2048$), GRU+Skip-thoughts, $|\mathcal{A}| = 2000$, Adam con lr $10^{-4}$ sin decay, early stopping.

**Ablation de fusiones (sin atención, *test-dev*):**

| Modelo | Params (M) | Y/N | No. | Other | **All (test-dev)** |
|---|---|---|---|---|---|
| Concat | 8.9 | 79.25 | 36.18 | 46.69 | 58.91 |
| MCB | 32 | 80.81 | 35.91 | 46.43 | 59.40 |
| MLB | 7.7 | **82.02** | 36.61 | 46.65 | 60.08 |
| MUTAN_noR | 4.9 | 81.44 | 36.42 | 46.86 | 59.92 |
| **MUTAN** | 4.9 | 81.45 | **37.32** | **47.17** | **60.17** |
| MUTAN+MLB | 17.5 | 82.29 | 37.27 | 48.23 | **61.02** |

Lecturas: con parámetros iguales, **MUTAN_noR** (Tucker sin restricción de rango, dims 160) supera a MLB, validando que modelar la interacción bilineal completa sobre proyecciones de baja dimensión vence a tener proyecciones de alta dimensión con fusión simple. La **sparsity de rango** mejora aún más (60.17 vs 59.92) con los mismos parámetros, actuando como regularizador. **MUTAN es el mejor modelo individual con menos parámetros que todos** (4.9 M). El ensamble tardío **MUTAN+MLB** suma $\sim$1 punto, confirmando complementariedad.

**Ablations de rango y dimensiones del núcleo.** Variando $t = t_q = t_v = t_o$ de 20 a 220, MUTAN_noR supera ampliamente al núcleo fijado a la identidad (equivalente a MLB) incluso con núcleo pequeño: el núcleo **aprende correlaciones reales**. Variando $t_o$ para distintos $R$ con $t_q = t_v = 210$, un **rango menor** permite alcanzar $t_o$ más alto sin sobreajuste, con menos parámetros y mayor accuracy en validación. La restricción de rango es un regularizador útil.

**State-of-the-art (con atención y ensamble, *test-std*):**

| Modelo | All (test-dev) | All (test-std) |
|---|---|---|
| MCB (7) | 66.7 | 66.5 |
| MLB (7) | 66.77 | 66.89 |
| MUTAN (3) | 67.03 | 66.96 |
| **MUTAN (5)** | **67.42** | **67.36** |

**MUTAN (5)** alcanzó el **SOTA de la época** (67.36 % en *test-std*). Notablemente, **MUTAN (3)** —solo 3 modelos— ya supera a MCB y MLB con ensambles de 7.

---

## Limitaciones

1. **Sigue siendo fusión de dos vectores.** Pese a la atención multi-glimpse, la imagen colapsa en un vector global atendido antes de la fusión final. No hay interacción región-palabra fina y sostenida.
2. **Segundo orden, no superior.** La forma bilineal no expresa con naturalidad razonamiento de orden superior (conteo, comparación espacial, lógica multi-paso).
3. **Hiperparámetros.** Tres dimensiones latentes más el rango $R$ amplían el espacio de búsqueda frente a fusiones simples, y la elección óptima depende del tipo de pregunta.
4. **Dependencia del extractor.** La calidad de la fusión está acotada por las features de ResNet-152 precomputadas, a diferencia de los enfoques bottom-up basados en detección de objetos.
5. **Superado por Transformers cross-modales.** Desde 2019 (ViLBERT, LXMERT, UNITER), la **cross-attention** sustituyó la fusión bilineal: en lugar de comprimir todo en una forma bilineal, se dejan interactuar todos los tokens de texto con todas las regiones a través de muchas capas.

---

## Por qué importa hoy

MUTAN es el **broche de la era de la fusión bilineal** en VQA (2015–2017). Su valor perdurable no es la cifra de accuracy —pronto superada— sino dos lecciones de diseño.

**La estructura algebraica gobierna el trade-off expresividad/parámetros.** La pregunta correcta no es "¿interacción rica o pocos parámetros?" sino "¿qué *estructura* impongo al tensor de interacción?". Tucker + restricción de rango entrega la interacción más rica con los menos parámetros. Esa idea —factorizar una operación cara con descomposiciones de bajo rango— reaparece hoy en LoRA, atención lineal y factorización de proyecciones en Transformers.

**Un marco unificador.** Mostrar que MCB y MLB son puntos en el espacio de restricciones de Tucker da un lenguaje común para razonar sobre familias enteras de modelos de fusión, en vez de memorizar arquitecturas aisladas. El grupo continuó esta línea con BLOCK (AAAI 2019, descomposición block-superdiagonal) y MUREL.

La **transición a cross-attention** se lee como el reconocimiento de que, en lugar de comprimir la interacción multimodal en una sola forma bilineal estructurada, conviene apilar muchas capas de atención que dejan interactuar libremente todos los elementos —pagando más cómputo a cambio de la capacidad que habilita el pre-entrenamiento a gran escala. El contrapunto eficiente lo encarna [Pythia (Jiang 2018)](/papers/pythia-jiang-2018): con buenas features de objetos (bottom-up attention), una fusión simple basta para gran parte del rendimiento. MUTAN y Pythia marcan los dos extremos del [dominio Multimodal](/dominios/multimodal): invertir en el operador de fusión, o invertir en la calidad de la entrada.

---

## Notas y enlaces

- **Relación con [MCB (Fukui 2016)](/papers/mcb-fukui-2016)**: predecesor directo; MUTAN demuestra que MCB es Tucker con factores diagonales fijos y núcleo hash disperso. Comparar ambas arquitecturas es el ejercicio canónico de la unidad de fusión bilineal.
- **Relación con MLB** (Kim et al. 2017): el otro predecesor; MUTAN demuestra que MLB es Tucker con núcleo identidad y dimensiones latentes iguales. El ensamble MUTAN+MLB explota su complementariedad.
- **Descomposición de Tucker**: Tucker (1966); para el tratamiento moderno, Kolda & Bader, *Tensor Decompositions and Applications*, SIAM Review 2009.
- **Código**: implementación oficial en PyTorch, [cadene/vqa.pytorch](https://github.com/cadene/vqa.pytorch).
- **Trabajo posterior**: BLOCK (Ben-younes et al., AAAI 2019) generaliza MUTAN con una descomposición block-superdiagonal.

Ver también: [fundamento Visual Question Answering](/fundamentos/visual-question-answering) · [dominio Multimodal](/dominios/multimodal) · [Clase 23](/clases/clase-23).
