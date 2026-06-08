# User Identification in Pinterest Through the Refinement of a Cascade Fusion of Text and Images

**Autores:** Juan Carlos Gomez, Mario-Alberto Ibarra-Manzano, Dora-Luz Almanza-Ojeda
**Afiliación:** Universidad de Guanajuato, Departamento de Ingeniería Electrónica, DICIS, Salamanca, México
**Venue:** Research in Computing Science, vol. 144 (2017), pp. 41-52. ISSN 1870-4069. Presentado en el contexto del taller LKE (Language and Knowledge Engineering) 2017.
**arXiv:** ninguno.
**Dataset público:** https://data.mendeley.com/datasets/fs4k2zc5j5/2 (versión 3 en la URL canónica del curso: .../3)
**Código:** https://github.com/jcgcarranza/2017rcs_code

---

## 1. Contexto: identificación de usuarios en redes sociales

El trabajo se ubica en la intersección de tres áreas: minería de redes sociales, aprendizaje multimodal y la tarea específica de *user identification* (identificación de usuarios). En la era del Big Data, una fracción enorme del tráfico de Internet ocurre en redes sociales, donde los usuarios crean y comparten contenido multimedia (noticias, reportes, videos, emociones, opiniones). Los autores caracterizan ese contenido generado por usuarios (*user-generated content*) con cinco propiedades: es abundante, se genera constantemente, es dinámico (distribuido en tiempo real), es representativo de usuarios o grupos de usuarios, y es **multimodal** —compuesto por una mezcla de texto, imágenes, video, audio y enlaces.

La motivación práctica es comercial y de seguridad: empresas y organizaciones quieren analizar el contenido generado por usuarios para obtener indicadores útiles en marketing, e-commerce, seguridad, política y educación. La identificación de usuarios sirve para agrupar personas con intereses similares, reconocer *lead users* e influencers, detectar clientes potenciales, simpatizantes políticos, y también trolls, intrusos, terroristas o amenazas a la seguridad pública —considerando que en redes sociales un usuario puede asumir identidades múltiples o falsas. El reverso positivo es la personalización: contenido adecuado a las necesidades de entretenimiento, compras, salud y educación de cada persona.

Los autores enumeran cinco dificultades intrínsecas de los posts en redes sociales que hacen la identificación un problema duro: (1) son multimodales, con un **gap semántico** entre modalidades (lo que dice el texto puede no ser representativo de lo que muestra la imagen); (2) los usuarios no siguen reglas gramaticales ni ortográficas, lo que dificulta usar atributos lingüísticos de alto nivel como sintaxis y semántica; (3) el texto varía enormemente en longitud entre posts; (4) texto e imágenes son altamente heterogéneos, con gran diversidad de temas; (5) hay muchos usuarios y su contenido puede solaparse.

El objeto de estudio es **Pinterest**, descrito como una de las redes sociales más populares del mundo con más de 150 millones de usuarios activos mensuales (cifra de 2017). En Pinterest los usuarios publican *pins*: la combinación de una imagen y un texto corto que la comenta. La tarea formal que definen es: dado un pin (par imagen+texto), **identificar el usuario específico que lo habría publicado**, planteándolo como un problema de **clasificación multiclase de etiqueta única** (*single-label multi-class*), donde las clases son usuarios concretos y cada pin pertenece a un solo usuario.

El antecedente directo es el trabajo de Cinar, Zoghbi y Moens [ref. 4 del paper], que combina linealmente (late fusion) las salidas de dos clasificadores independientes de texto e imagen para **inferir intereses** de usuarios en Pinterest. Gomez et al. usan ese modelo como baseline pero atacan un problema más difícil: identificar **usuarios específicos** en lugar de intereses, lo que multiplica el número de clases posibles.

---

## 2. El dataset Pinterest (composición exacta)

Este es el aporte que convierte al paper en la base de la Clase 25. El dataset se construyó **crawleando directamente el sitio de Pinterest** y se compone de:

- **70.200 pins** en total.
- **117 usuarios** distintos → **117 clases** en el problema de clasificación.
- Procedimiento de muestreo: por cada usuario se seleccionaron al azar **3 boards** (tableros), guardando **200 pins por board**. Se descartó la información de board (a qué tablero pertenece cada pin) y se fusionaron todos los pins → **600 pins por usuario**. Nota: 117 × 600 = 70.200, lo que cuadra exactamente con el total reportado.
- **Splits por usuario** (selección aleatoria, manteniendo las mismas proporciones por cada usuario):
  - **400 pins/usuario para entrenamiento** (46.800 totales).
  - **100 pins/usuario para validación** (11.700 totales).
  - **100 pins/usuario para test** (11.700 totales).

### Modalidad texto

- Los comentarios están en inglés, de longitud variable: **desde 1 palabra (12.33% de los pins)** hasta un **máximo de 552 palabras**.
- Estadísticas de número de palabras por pin sobre todo el dataset: **mínimo 1, máximo 552, mediana 4, promedio 8.5**. La mediana de 4 indica que la mayoría de los comentarios son muy cortos —un dato crítico para entender los resultados.
- Preprocesamiento del texto: se limpió cada pin removiendo símbolos especiales (hashtags, asteriscos, etc.), stopwords, URLs, palabras de una sola letra y palabras largas (>30 caracteres). Usando exclusivamente el set de entrenamiento se extrajo un diccionario removiendo palabras que aparecían en un solo pin.
- **Diccionario final: 17.145 palabras.** El diccionario para la matriz documento-término se extrajo del training set durante la validación, y de los sets de training+validation al construir el modelo final.
- Representación: matrices documento-término **X_tr, X_v, X_t** con ponderación **tf-idf** (bag-of-words ponderado).
- La Tabla 1 ilustra la diversidad temática con las 5 palabras más frecuentes de 5 usuarios al azar (con frecuencia de pins): Usuario 1 → logo(77), design(64), via(51), infographic(46), designspiration(46); Usuario 2 → make(45), diy(34), cream(32), chicken(30), cake(30); Usuario 3 → love(28), vintage(20), one(14), black(12), elizabeth(10); Usuario 4 → crochet(107), pattern(56), free(48), com(33), art(30); Usuario 5 → diamond(155), ring(144), necklace(127), gold(82), sapphire(43). Cada usuario habla de temas marcadamente distintos (diseño, cocina, moda vintage, tejido, joyería).

### Modalidad imagen

- Las imágenes varían en tamaño pero todas están en formato **JPG**.
- **Solo el 4.2% de las imágenes son compartidas por 2 o más usuarios** → gran diversidad de contenido visual (productos como ropa y joyería, intereses como comida y decoración, fotografías de animales y paisajes, y contenido abstracto como pinturas y diseños).
- **Representación de features CNN de 4.096 dimensiones:** cada imagen se preprocesa con una **CNN** para obtener un vector fila **g^u** de features por usuario. Se usó la librería **DeCAF** [ref. 8, Donahue et al.], tomando los valores de activación de las **4.096 neuronas de la 7.ª capa** (la penúltima capa fully-connected, típicamente fc7) como features de imagen. El modelo DeCAF usado para la transformación estaba **preentrenado con ImageNet** [ref. 15, Krizhevsky et al. / AlexNet]. Los vectores de imagen se agrupan en matrices **G_tr, G_v, G_t** emparejadas fila a fila con las matrices de texto X y ordenadas por usuario.

> **Nota terminológica importante para la Clase 25:** el enunciado de la clase describe "imagen ya embebida con CNN en 4096 features". Eso corresponde exactamente a estos vectores DeCAF/fc7 de 4.096 dimensiones. El dataset entregado al alumno ya viene con las imágenes convertidas a estos embeddings, evitando tener que correr la CNN.

### Análisis de la estructura de los datos (PCA)

Los autores proyectaron texto e imagen sobre 2 componentes principales (PCA) y reportan la varianza explicada con 10 PCs: **0.043 para texto** vs **0.321 para imagen**. Interpretación: las imágenes son más **compactas/homogéneas** en sus features (se comprimen mejor con PCA) pero esa homogeneidad NO ayuda a separar usuarios; el texto es más heterogéneo en sus features (no se comprime bien) pero sí sirve para distinguir datos de distintos usuarios hasta cierto punto. La Figura 2 muestra estadísticas (min/max/median/avg) del número de palabras por usuario en escala logarítmica; la Figura 3 muestra las nubes de puntos: texto disperso a lo largo de los PCs vs imagen colapsada en un blob denso.

---

## 3. Método: fusión en cascada con refinamiento

El modelo es de **dos fases**. La notación: un usuario *u* con su colección de pins **P_u = (g_i^u, x_i^u)**, donde **g** refiere a la imagen y **x** al texto; hay *m* usuarios totales y n = Σ n_{u_i} pins totales. La tarea es identificar el usuario û que generó un pin p_t = (g_t, x_t).

### Fase 1 — Clasificadores independientes + fusión en cascada

- Se entrenan **dos clasificadores independientes**: uno sobre el texto F_x (con la representación bag-of-words tf-idf) y otro sobre la imagen F_g (con los features DeCAF de 4.096-d).
- El clasificador individual es **regresión logística** (logistic regression), elegida porque se entrena naturalmente para problemas multiclase y entrega directamente una **probabilidad por usuario**.
- El parámetro de regularización **C** se optimiza independientemente para cada modalidad usando el set de validación, probando los valores **{0.1, 1, 10, 100}**. Tras optimizar, se fusionan training+validation en un solo set y se reentrena con el C óptimo.
- En test, para cada pin p_t = (g_t, x_t) se clasifica el texto con F_x(x_t) y la imagen con F_g(g_t), produciendo dos vectores de probabilidad de pertenecer a cada usuario: **r_x = [r_{u1}^x, ..., r_{um}^x]** y **r_g = [r_{u1}^g, ..., r_{um}^g]**.
- **Fusión en cascada (cascade fusion, CF)** — Ecuación (1): se multiplican elemento a elemento los dos vectores de probabilidad:

$$\mathbf{r} = \mathbf{r}_x \cdot \mathbf{r}_g$$

  Es decir, la "cascada" no es un pipeline secuencial sino un **producto de las probabilidades de ambas modalidades** (equivalente a asumir independencia condicional, estilo Naive Bayes sobre modalidades). El objetivo es combinar ambas modalidades para reducir el gap semántico y explotar mejor el contenido completo. Del vector resultante **r** se selecciona el usuario û con mayor probabilidad.

### Fase 2 — Refinamiento (CF+Ref)

- Para cada pin de test p_t se extraen los **top 10 usuarios más probables** del vector de cascada: top = [û_1, ..., û_10], con sus probabilidades de fusión r_top = [r_{û1}, ..., r_{û10}].
- Para cada usuario candidato û_i se toman **todos sus pins de entrenamiento P_{û_i}**, se calcula la **similitud coseno** de cada uno de esos pins con el pin de test p_t, y se queda con la **similitud máxima l_{û_i}** = max similitud.
- Se **re-pondera** la probabilidad: r_{û_i} ← r_{û_i} × l_{û_i}.
- Finalmente, del vector refinado r_top se selecciona el usuario û con el mayor valor.
- La idea: si entre los 10 candidatos hay uno cuyo pin de entrenamiento es muy parecido al pin de test, se le sube la probabilidad. El refinamiento aprovecha el hecho (medido) de que el **recall@10 de la cascada es 75%** —en el 75% de los casos el usuario correcto está entre los 10 más probables, así que reordenarlos correctamente puede mejorar.

### Setup experimental

- Implementación en **Python** con **scikit-learn** y **NumPy**. Hardware: PC Windows, Core i5 2.5 GHz, 8 GB RAM.
- **Baselines:** (1) dos modelos de regresión logística separados, solo texto y solo imagen (la base de la cascada, optimizados en C sobre {0.1, 1, ...}); (2) el modelo de late fusion de Cinar et al. [4], con la combinación lineal pred = λ·F_g(g_t) + (1−λ)·F_x(x_t), evaluado con **λ ∈ {0.3, 0.5, 0.7}**.
- **Métricas:** accuracy y **macro-F1**. Se usan promedios macro de F1 (porque la clasificación es single-label, el micro-F1 = accuracy). Definiciones estándar: accuracy = (tp+tn)/(tp+fp+fn+tn), F1 = 2·(precision·recall)/(precision+recall).

---

## 4. Resultados

La Tabla 2 resume todos los modelos (valores en %):

| Modelo | Accuracy | F1 (macro) |
|---|---|---|
| Solo texto | 33.75 | 33.21 |
| Solo imágenes | 21.62 | 19.72 |
| [4] λ = 0.3 | 37.18 | 35.01 |
| [4] λ = 0.5 | 34.68 | 32.11 |
| [4] λ = 0.7 | 30.73 | 28.10 |
| **CF (cascade fusion)** | 37.34 | 36.16 |
| **CF+Ref (cascada + refinamiento)** | **38.30** | **37.46** |

Lecturas clave:

1. **El problema es duro:** todos los modelos quedan por debajo del 40% en accuracy y macro-F1, sobre 117 clases. (El azar sería ~0.85%, así que 38% sigue siendo muy superior a azar.)
2. **Imagen sola es lo peor** (21.62% acc / 19.72% F1). La razón: los features DeCAF preentrenados en ImageNet son buenos para *representar* la imagen pero malos para *discriminar entre usuarios*; las imágenes son homogéneas en features pero su contenido (no representado bien) es lo que distingue usuarios. Los autores conjeturan que un fine-tuning de DeCAF con imágenes de Pinterest mejoraría esto.
3. **Texto solo es mejor** (33.75% / 33.21%) porque el texto refleja mejor la heterogeneidad de los datos y discrimina mejor entre usuarios.
4. **CF supera a todos los baselines:** la cascada (CF) mejora cerca de 4% en accuracy y F1 respecto a solo texto, y ~1% en F1 respecto al mejor modelo de late fusion ([4] con λ=0.3). El producto de probabilidades aprovecha ambas modalidades mejor que la combinación lineal.
5. **El refinamiento (CF+Ref) añade ~1% más** en accuracy y F1 sobre CF, alcanzando el mejor resultado global (38.30 / 37.46). Confirma que reordenar los top-10 con similitud coseno explota mejor la información.
6. **recall@10 de CF = 75%:** entre los 10 usuarios más probables por pin, hay 75% de chance de que esté el correcto. Esto fija el techo de lo que el refinamiento puede lograr y motiva la dirección futura de "mejor reordenamiento".

### Análisis por usuario

- La **Figura 4** muestra F1 por usuario para todos los modelos: comportamiento similar entre modelos, con CF y CF+Ref un poco por encima. Algunos usuarios son mucho más fáciles que otros: **7 usuarios con desempeño <10%** y **6 usuarios con desempeño >90%**.
- La **Figura 5** (histograma de F1 del modelo CF+Ref) confirma que para la mayoría de los casos la identificación es difícil, con desempeños generalmente por debajo del 50%.
- La **Tabla 3** correlaciona desempeño con longitud de los comentarios. Los 5 peores y 5 mejores usuarios (User, mediana de palabras, F1):
  - **Peores:** 13 (mediana 4, F1 0.059); 116 (6, 0.062); 2 (5, 0.081); 101 (6, 0.083); 64 (5, 0.086).
  - **Mejores:** 106 (16, 0.995); 83 (13.5, 0.985); 34 (68, 0.980); 86 (12, 0.966); 40 (28, 0.949).
- Los usuarios fáciles tienen pins con medianas de palabras entre 13.5 y 68; los difíciles entre 4 y 6. Hay una **correlación de Pearson débil de 0.46** entre desempeño y mediana de longitud de comentarios: usuarios que postean comentarios más largos dan más información para identificarlos. La mediana global de 4 palabras explica en parte el bajo desempeño general.

---

## 5. Limitaciones reconocidas

Los propios autores identifican varias limitaciones, lo que las hace material didáctico valioso:

1. **Features de imagen no discriminativos:** DeCAF/fc7 preentrenado en ImageNet representa bien pero no separa usuarios. El significado semántico se pierde en convoluciones de features más generales. Solución propuesta: fine-tuning de DeCAF con datos de Pinterest.
2. **Comentarios demasiado cortos:** mediana de 4 palabras, y 12.33% de pins con una sola palabra. Poco texto = poca señal discriminativa.
3. **Número alto de clases:** 117 usuarios es un espacio grande; aun cuando el contenido es heterogéneo, las distribuciones de texto e imagen pueden ser similares entre distintos usuarios (visible en Figura 3), generando solapamiento.
4. **Gap semántico entre modalidades:** un comentario puede aplicar a varias imágenes, y una imagen puede describirse de muchas formas; la fusión por producto asume una correspondencia que no siempre se cumple.
5. **Fusión simple:** el producto de probabilidades (cascada) y la re-ponderación por similitud coseno son heurísticas; los autores sugieren explorar otros modelos de late fusion y transformar ambas modalidades a espacios donde se separen mejor.

### Direcciones futuras planteadas

- Fine-tuning de DeCAF con imágenes de Pinterest.
- Transformar ambas modalidades a otros espacios de features donde los datos se separen mejor (por modalidad o conjuntamente).
- Explorar otros modelos de late fusion, especialmente un mejor reordenador de los top-10 (dado el recall@10 de 75%).

---

## 6. Por qué este dataset es la base de la Clase 25

La Clase 25 ("Recomendación usando Imágenes y Texto") es un **case study** que reutiliza **exactamente este dataset**: los 70.200 pins de 117 usuarios, con cada pin = texto corto + imagen ya convertida a un vector de **4.096 features** mediante CNN (los features DeCAF/fc7 descritos aquí). Las conexiones didácticas son directas:

1. **Multimodalidad práctica.** El dataset es un caso de juguete realista para enseñar **fusión de modalidades** (texto + imagen): cómo combinar un clasificador/recomendador de texto con uno de imagen. La cascade fusion (producto de probabilidades) y la late fusion lineal de Cinar et al. son dos recetas concretas de fusión que el alumno puede implementar y comparar.

2. **Features ya embebidos.** Que las imágenes vengan como vectores de 4.096-d (en vez de píxeles crudos) permite a la clase enfocarse en la **lógica de recomendación/fusión** sin pagar el costo de entrenar una CNN. Esto conecta con [fundamentos/redes-convolucionales] (de dónde salen los 4.096 features) sin requerir reentrenarla.

3. **Bag-of-words sobre texto.** El lado texto del dataset usa tf-idf sobre un diccionario de 17.145 palabras, un caso canónico de [fundamentos/bag-of-words] que el alumno ya domina de clases anteriores de NLP.

4. **De identificación a recomendación.** Aunque el paper plantea *identificación de usuarios* (¿quién publicó este pin?), la Clase 25 lo reencuadra como **recomendación**: la afinidad usuario↔contenido que aquí sirve para identificar autoría es la misma señal que en un recommender system predice qué contenido le gustará a un usuario. La estructura usuario×ítem×interacción y el cálculo de similitudes (coseno entre pins) son nucleares en [fundamentos/recommender-systems].

5. **Lecciones honestas.** El paper muestra que el problema es duro (38% accuracy sobre 117 clases) y por qué: imágenes poco discriminativas, texto corto, muchas clases. Para un curso es valioso porque enseña a **leer resultados con escepticismo**, a entender que una modalidad puede aportar poco, y a diagnosticar (vía PCA, correlaciones, análisis por usuario) en lugar de solo reportar un número.

6. **Reproducibilidad.** El dataset está públicamente disponible en Mendeley (https://data.mendeley.com/datasets/fs4k2zc5j5/3) y el código en GitHub, lo que permite que el laboratorio [laboratorios/lab-25] trabaje directamente sobre los datos originales.

En síntesis: este paper de 2017 de la Universidad de Guanajuato no es solo un antecedente teórico, sino la **fuente material** de la Clase 25 —define la composición exacta (70.200 pins, 117 usuarios, splits 400/100/100, features 4.096-d, diccionario 17.145), las representaciones (tf-idf + DeCAF) y dos recetas de fusión multimodal (cascada y lineal) que el case study expande hacia el lenguaje de los sistemas de recomendación.
