---
title: "User Identification in Pinterest: Cascade Fusion of Text and Images"
weight: 251
math: true
---

{{< paper-card
    title="User Identification in Pinterest Through the Refinement of a Cascade Fusion of Text and Images"
    authors="Juan Carlos Gomez, Mario-Alberto Ibarra-Manzano, Dora-Luz Almanza-Ojeda"
    year="2017"
    venue="Research in Computing Science 2017"
    pdf="/papers/pinterest-dataset-2017.pdf" >}}
Paper de la Universidad de Guanajuato que originó el **dataset Pinterest** usado en la [Clase 25](/clases/clase-25): 70.200 pins de 117 usuarios, cada uno un par imagen+texto, con las imágenes ya convertidas a vectores CNN de 4.096 dimensiones. La tarea es identificar qué usuario publicó un pin (clasificación multiclase de 117 clases). El método combina un clasificador de texto (bag-of-words tf-idf) con uno de imagen (features DeCAF) mediante una **fusión en cascada** —producto de probabilidades— y un refinamiento por similitud coseno. El problema resulta duro (38.30% accuracy), lo que lo vuelve un excelente caso de estudio sobre fusión multimodal.
{{< /paper-card >}}

---

## Contexto

La **identificación de usuarios** en redes sociales interesa a empresas y organizaciones para marketing, e-commerce, seguridad y demografía: agrupar personas por intereses, detectar influencers o trolls, o personalizar contenido. Pinterest es el caso elegido (más de 150 millones de usuarios activos mensuales en 2017): sus usuarios publican *pins*, la combinación de una imagen y un texto corto que la comenta.

Los autores plantean la tarea como **clasificación multiclase de etiqueta única**: dado un pin (imagen + texto), identificar el usuario específico que lo habría publicado. Cada clase es un usuario y cada pin pertenece a uno solo. Es un problema notoriamente difícil por cinco razones: el contenido es multimodal con un **gap semántico** entre texto e imagen, los usuarios no respetan gramática ni ortografía, el texto varía mucho en longitud, texto e imágenes son muy heterogéneos, y hay muchos usuarios cuyo contenido se solapa.

El antecedente directo es el modelo de Cinar et al. (2015), que combina linealmente dos clasificadores de texto e imagen para inferir *intereses*. Este trabajo lo usa como baseline pero ataca un problema más duro: identificar **usuarios concretos**, lo que dispara el número de clases.

## El dataset Pinterest

El dataset se construyó crawleando directamente Pinterest. Por cada usuario se eligieron al azar 3 boards, 200 pins por board, se descartó la info de board y se fusionaron en 600 pins por usuario (117 × 600 = 70.200). Su composición exacta:

| Atributo | Valor |
|---|---|
| Pins totales | **70.200** |
| Usuarios (= clases) | **117** |
| Pins por usuario | 600 (3 boards × 200) |
| Train | **400 pins/usuario** (46.800) |
| Validación | **100 pins/usuario** (11.700) |
| Test | **100 pins/usuario** (11.700) |
| Texto: longitud por pin | min 1, máx 552, mediana 4, promedio 8.5 palabras |
| Texto: pins de 1 sola palabra | 12.33% |
| Diccionario (tras limpieza) | **17.145 palabras**, ponderación tf-idf |
| Imágenes: formato | JPG (tamaños variables) |
| Imágenes: compartidas por ≥2 usuarios | solo 4.2% |
| **Features CNN por imagen** | **4.096 dimensiones** (DeCAF, capa fc7, preentrenada en ImageNet) |

El lado texto se limpió (hashtags, stopwords, URLs, palabras de una letra y de más de 30 caracteres) y se representó con [bag-of-words](/fundamentos/bag-of-words) ponderado por tf-idf. El lado imagen se procesó con una [red convolucional](/fundamentos/redes-convolucionales): se usó la librería **DeCAF** tomando las activaciones de las **4.096 neuronas de la 7.ª capa** (fc7) de un modelo preentrenado en ImageNet. Cada imagen queda así representada por un vector de 4.096 features —esto es exactamente lo que la Clase 25 recibe como "imagen ya embebida con CNN en 4096 features".

Un análisis con PCA revela una asimetría clave: con 10 componentes la varianza explicada del texto es 0.043 frente a 0.321 de la imagen. Las imágenes son **más homogéneas** en sus features (se comprimen mejor) pero esa homogeneidad no ayuda a separar usuarios; el texto es más heterogéneo y discrimina mejor entre usuarios.

## Método: cascade fusion

El modelo tiene **dos fases**.

**Fase 1 — clasificadores independientes y cascada.** Se entrenan dos modelos de **regresión logística** por separado: uno sobre el texto (tf-idf) y otro sobre la imagen (4.096-d). La regularización C se optimiza por modalidad en validación sobre {0.1, 1, 10, 100}. En test, cada pin produce dos vectores de probabilidad sobre los 117 usuarios, $\mathbf{r}_x$ (texto) y $\mathbf{r}_g$ (imagen). La **fusión en cascada** los combina por producto elemento a elemento:

$$\mathbf{r} = \mathbf{r}_x \cdot \mathbf{r}_g$$

A pesar del nombre, la "cascada" no es un pipeline secuencial sino un producto de probabilidades (asume independencia condicional entre modalidades, al estilo Naive Bayes). Se elige el usuario con mayor probabilidad en $\mathbf{r}$.

**Fase 2 — refinamiento.** Para cada pin de test se extraen los **top 10 usuarios** más probables. Para cada candidato se calcula la **similitud coseno** de todos sus pins de entrenamiento con el pin de test y se toma la máxima $l_{\hat{u}_i}$; con ella se re-pondera la probabilidad: $r_{\hat{u}_i} \leftarrow r_{\hat{u}_i} \cdot l_{\hat{u}_i}$. Luego se reordena y se elige el ganador. La intuición: el **recall@10** de la cascada es 75%, así que reordenar correctamente los 10 candidatos puede subir el acierto final.

## Resultados experimentales

Métricas: accuracy y macro-F1 sobre 117 clases (el azar daría ~0.85%).

| Modelo | Accuracy | F1 (macro) |
|---|---|---|
| Solo texto | 33.75 | 33.21 |
| Solo imágenes | 21.62 | 19.72 |
| Cinar et al. λ = 0.3 | 37.18 | 35.01 |
| Cinar et al. λ = 0.5 | 34.68 | 32.11 |
| Cinar et al. λ = 0.7 | 30.73 | 28.10 |
| **Cascade Fusion (CF)** | 37.34 | 36.16 |
| **CF + Refinamiento** | **38.30** | **37.46** |

Lecturas principales: (1) todos los modelos quedan por debajo del 40% —el problema es duro; (2) **imagen sola es lo peor** (21.62%) porque los features DeCAF preentrenados representan bien pero discriminan mal entre usuarios; (3) **texto solo es mejor** (33.75%) por su heterogeneidad; (4) la cascada supera a todos los baselines (~4% sobre solo texto, ~1% en F1 sobre el mejor late fusion); (5) el refinamiento añade ~1% más y alcanza el mejor resultado global. El recall@10 de 75% fija el techo de lo que el refinamiento puede lograr.

El análisis por usuario muestra alta varianza: 7 usuarios con desempeño <10% y 6 con >90%. Existe una correlación de Pearson **débil (0.46)** entre el desempeño y la mediana de longitud de los comentarios: los usuarios que postean comentarios más largos (medianas de 13.5 a 68 palabras) se identifican mucho mejor que los de comentarios cortos (medianas de 4 a 6).

## Limitaciones reconocibles

- **Features de imagen poco discriminativos:** DeCAF/fc7 sobre ImageNet representa pero no separa usuarios; los autores proponen fine-tuning con imágenes de Pinterest.
- **Comentarios demasiado cortos:** mediana de 4 palabras y 12.33% de pins con una sola palabra dejan poca señal de texto.
- **Muchas clases (117):** aun con contenido heterogéneo, las distribuciones de texto e imagen se solapan entre usuarios distintos.
- **Gap semántico:** un texto puede aplicar a muchas imágenes y viceversa; el producto de probabilidades asume una correspondencia que no siempre existe.
- **Fusión heurística:** el producto y la re-ponderación por coseno son simples; queda abierto explorar otros esquemas de late fusion y reordenadores de los top-10.

## Por qué importa para la Clase 25

La [Clase 25](/clases/clase-25) ("Recomendación usando Imágenes y Texto") es un case study que reutiliza **exactamente este dataset**: los mismos 70.200 pins de 117 usuarios, con cada imagen ya convertida a los 4.096 features CNN descritos aquí. El paper aporta el material y las recetas:

- **Multimodalidad concreta.** Enseña a combinar un modelo de texto con uno de imagen mediante fusión (cascada por producto vs. late fusion lineal), dos recetas implementables y comparables.
- **Features ya embebidos.** Las imágenes como vectores de 4.096-d permiten enfocarse en la lógica de recomendación/fusión sin entrenar una [CNN](/fundamentos/redes-convolucionales) desde cero.
- **Texto canónico.** El lado texto usa [bag-of-words](/fundamentos/bag-of-words) tf-idf sobre 17.145 palabras.
- **De identificación a recomendación.** La afinidad usuario↔contenido que aquí identifica autoría es la misma señal que un [sistema de recomendación](/fundamentos/recommender-systems) usa para predecir gustos; la estructura usuario×ítem y la similitud coseno son nucleares en esa área.
- **Lecciones honestas.** Que el techo sea 38% enseña a leer resultados con escepticismo, diagnosticar por qué una modalidad aporta poco y analizar por usuario en lugar de quedarse con un solo número.

El laboratorio práctico está en lab-25.

## Notas y enlaces

- Dataset en Mendeley: https://data.mendeley.com/datasets/fs4k2zc5j5/3
- Código de los autores: https://github.com/jcgcarranza/2017rcs_code
- Venue: Research in Computing Science, vol. 144 (2017), pp. 41-52. ISSN 1870-4069.
- Afiliación: Universidad de Guanajuato (DICIS), Salamanca, México.
