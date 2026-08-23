---
title: "GhostVLAD (2018)"
weight: 446
math: true
---

{{< paper-card
    title="GhostVLAD for set-based face recognition"
    authors="Yujie Zhong, Relja Arandjelović, Andrew Zisserman (VGG, University of Oxford; DeepMind)"
    year="2018"
    venue="ACCV 2018 / arXiv:1810.09951"
    pdf="/papers/ghostvlad-zhong-2018.pdf" >}}
El paper que añade a [NetVLAD](/papers/netvlad-arandjelovic-2016) una idea de una sola línea de código: **clusters que compiten en la asignación pero cuyos residuos se descartan**. Sirven de sumidero — un descriptor de baja calidad puede asignarse a un cluster fantasma, y con eso su peso hacia los clusters reales cae y su contribución al descriptor final se atenúa. El resultado es que *"una ponderación por calidad emerge automáticamente"* sin que nadie etiquete qué imágenes son malas. El paper es de **reconocimiento de caras** a partir de conjuntos de imágenes, no de audio; pero es la fuente directa de los `ghost_clusters = 2` del [Lab 41](/laboratorios/lab-41), y la razón de que ese modelo pueda prescindir de detección de actividad de voz.
{{< /paper-card >}}

---

## Contexto: el problema del conjunto, no de la imagen

El reconocimiento de caras clásico compara **una** imagen contra otra. El reconocimiento *basado en plantillas* (template-based) parte de un **conjunto** de imágenes del mismo sujeto —fotogramas de un video, varias fotos— y tiene que producir un descriptor único que represente al conjunto.

Eso plantea exactamente el mismo problema que la [Clase 41](/clases/clase-41) enfrenta en audio, con otro vocabulario:

| | Caras (este paper) | Habla ([Xie et al. 2019](/papers/utterance-level-xie-2019)) |
|---|---|---|
| Entrada | N imágenes de una persona | N descriptores temporales de un audio |
| Problema | N es variable | N depende de la duración |
| Salida deseada | un vector de largo fijo | un vector de largo fijo |
| Basura a filtrar | imágenes borrosas, de perfil, mal recortadas | silencio, ruido de fondo, música |

Y la misma objeción a la solución obvia: guardar los descriptores individuales y comparar todos contra todos *"puede consumir mucha memoria y ser prohibitivamente lento"*. Hace falta **agregar**.

El estado del arte previo agregaba con promedio o máximo. El argumento del paper para buscar algo mejor es una cita a la literatura de recuperación de imágenes: los métodos de codificación tipo Fisher Vector *"aumentan la separación entre descriptores extraídos de parches relacionados y no relacionados"*. Si eso vale para parches, debería valer para caras.

## Método: NetVLAD con G clusters de más

El paper parte textualmente de NetVLAD:

$$V(j,k) = \sum_{i=1}^{N} \frac{e^{\,a_k^\top x_i + b_k}}{\sum_{k'=1}^{K} e^{\,a_{k'}^\top x_i + b_{k'}}} \big(x_i(j) - c_k(j)\big)$$

y lo modifica de la forma más económica imaginable:

> *"Añadimos G clusters «fantasma» que contribuyen a las asignaciones blandas de la misma manera que los K originales, pero **los residuos entre los vectores de entrada y los centros de los clusters fantasma se ignoran** y no contribuyen a la salida final. En otras palabras, la suma del denominador de la ecuación 1, en lugar de llegar a K, llega a K + G, mientras que la salida sigue siendo de dimensión D_F × K."*

Es decir: **el denominador del softmax cambia, el numerador no.**

$$\bar{a}_k(x_i) = \frac{e^{\,a_k^\top x_i + b_k}}{\sum_{k'=1}^{\mathbf{K+G}} e^{\,a_{k'}^\top x_i + b_{k'}}}, \qquad k = 1,\dots,K$$

Y una consecuencia de contabilidad de parámetros que el paper explicita, y que importa mucho al leer implementaciones:

> *"esto significa que $\{a_k\}$ y $\{b_k\}$ tienen K+G elementos cada uno, mientras que **$\{c_k\}$ sigue teniendo K**."*

**No hay centroides fantasma.** Los fantasmas existen solo en la etapa de asignación. Tienen vector de pesos y sesgo —para poder competir en el softmax— pero no tienen posición en el espacio de descriptores, porque nunca se calcula un residuo contra ellos.

### El mecanismo, en una frase

> *"Este mecanismo permite a la red asignar descriptores no informativos a los clusters fantasma, **disminuyendo así sus pesos de asignación hacia los clusters no fantasma**, y por tanto reduciendo su contribución a la representación final de la plantilla."*

El softmax es un presupuesto que suma 1. Si un descriptor gasta el 90 % de su masa en un fantasma, solo le queda el 10 % para repartir entre los K clusters reales. **La atenuación no requiere ninguna decisión explícita de descarte: es aritmética del softmax.**

Y lo notable es que nadie supervisa qué debe descartarse:

> *"no forzamos explícitamente que las imágenes de baja calidad se asignen a los clusters fantasma, sino que [la ponderación por calidad] emerge automáticamente."*

### Generalización, no alternativa

> *"GhostVLAD es una generalización de NetVLAD, ya que con G = 0 los dos son equivalentes."*

Vale la pena retenerlo: no es una arquitectura rival, es un hiperparámetro más. Y la comparación limpia —mismo backbone, mismos datos, mismo entrenamiento, solo G distinto— es lo que hace creíbles sus números.

## Resultados: cuánto vale un cluster que se tira

Verificación en IJB-B, la métrica TAR con FAR = 10⁻⁵ (la columna más exigente):

| Fila | Red | K | **G** | TAR@FAR=1e-5 | |
|---|---|---|---|---|---|
| 12 | SE-GV-3 | 8 | **0** | 0,741 | NetVLAD |
| 14 | SE-GV-3-g1 | 8 | **1** | **0,753** | **+1,2 pts** |
| 16 | SE-GV-3-g2 | 8 | **2** | 0,754 | +0,1 sobre G=1 |
| 13 | SE-GV-4 | 8 | **0** | 0,747 | NetVLAD |
| 15 | **SE-GV-4-g1** | 8 | **1** | **0,762** | **+1,5 pts — SOTA** |

Tres lecturas:

1. **Un solo cluster fantasma da la mejora completa.** Pasar de G=1 a G=2 aporta 0,001. El mecanismo no escala con G: basta con que exista un sumidero.
2. La mejor configuración del paper (`SE-GV-4-g1`) fija el estado del arte en IJB-B *"por un margen significativo"*, entrenando solo con VGGFace2 y superando a métodos que combinan VGGFace2 con MS-Celeb-1M.
3. Sobre el número de clusters reales: *"encontramos que un rango amplio de K logra buen rendimiento, con **K = 8 siendo el mejor**"*. Es exactamente el `vlad_clusters = 8` del Lab 41.

Y una observación del paper que conecta con lo verificado en la [práctica de la Clase 41](/clases/clase-41/practica/02-agregacion-vlad):

> *"se espera que K no sea demasiado pequeño para evitar el subajuste (**por ejemplo, K = 1 es similar al average-pooling**) ni demasiado grande, para prevenir la sobre-cuantización y el sobreajuste."*

El promedio temporal no es una alternativa a VLAD: es VLAD con K = 1.

## Detalles de implementación que explican las implementaciones

**La inicialización de los fantasmas.** Los K clusters reales se inicializan con k-means sobre features de imágenes **no degradadas**. Los G fantasmas, *"de forma similar, pero usando imágenes degradadas para el clustering"* — y para G = 1, k-means se reduce a promediar. Es decir: los fantasmas nacen apuntando al centroide de la basura. Esa semilla inicializa sus $\{a_k\}, \{b_k\}$; no crea un $c_k$ fantasma.

**El entrenamiento es por etapas.** Primero se entrena sin fantasmas *"porque las imágenes de entrenamiento no están degradadas en esta etapa"*; después se añaden los fantasmas y se degradan las imágenes. El mecanismo necesita basura para aprender a reconocerla.

**La comparación es un producto punto.** *"La similitud entre dos plantillas se mide como el producto escalar entre las representaciones; recordemos que tienen norma unitaria."* Idéntico al `np.sum(v1*v2)` del Lab 41.

## Conexión con el Lab 41: los dos centroides que no existen en el paper

El [Lab 41](/laboratorios/lab-41) implementa GhostVLAD con `K = 8`, `G = 2`. Y su clase `VladPooling` crea:

```python
self.cluster = nn.Parameter( torch.Tensor( self.k_centers + self.g_centers, d_size ) )   # 10 x 512
```

**Diez centroides**, cuando el paper dice explícitamente que $\{c_k\}$ tiene K = 8 elementos. El código computa los residuos contra los 10 y después descarta las dos últimas filas:

```python
cluster_res = cluster_res[:, :self.k_centers, :]
```

Al abrir el checkpoint entrenado, la predicción del paper se confirma de forma directa: los dos centroides fantasma **conservan su valor de inicialización**. Norma exactamente 1,000 (los ocho reales tienen ~14,06), ortogonales entre sí (coseno −0,000) y casi ortogonales a los reales. Y reemplazarlos por ruido multiplicado por 1000 cambia el descriptor final en el octavo decimal (coseno 0,99999994).

La razón es que el recorte les corta el gradiente: sus residuos se descartan **antes** de la pérdida, así que no hay camino desde ellos hasta el error. Son **1.024 parámetros muertos por construcción** — un artefacto de implementación que el paper había especificado no crear.

Lo que sí se entrena de los fantasmas son sus pesos de asignación, y ahí se ve el mecanismo del paper hecho números:

| k | ‖w_k‖ | b_k | |
|---|---|---|---|
| 0–5 | 0,180 – 0,206 | −0,088 … −0,140 | reales |
| 6–7 | 0,391 / 0,402 | −0,212 / −0,216 | reales |
| **8** | **1,079** | **+0,554** | **fantasma** |
| **9** | **0,759** | **+0,470** | **fantasma** |

Los ocho reales tienen sesgo **negativo**; los dos fantasmas, **positivo**. En un softmax eso significa ganar por defecto: cuando un descriptor no activa con fuerza ningún cluster real, su masa se va a los fantasmas. Y su ‖w‖ es 3–5× mayor, o sea que reaccionan mucho más agresivamente a la entrada.

**Por qué el modelo de audio los necesita.** En [Xie et al. (2019)](/papers/utterance-level-xie-2019) el preprocesamiento normaliza cada frame del espectrograma contra sí mismo, lo que **amplifica el silencio hasta ponerlo en la misma escala que la voz** (medido en el lab: 37×), y el paper es explícito en que *"no se aplica detección de actividad de voz ni eliminación automática de silencio"*. El agregador recibe basura amplificada y sin marcar. Los clusters fantasma son el detector de actividad de voz implícito del modelo — y valen 0,35 puntos de EER (3,57 % → 3,22 %).

## Limitaciones reconocibles

- **G > 1 no aporta.** El propio paper lo muestra (0,753 → 0,754). Que el Lab 41 use G = 2 es una elección sin respaldo empírico en la fuente.
- **Los fantasmas requieren datos degradados para aprender.** El entrenamiento por etapas con degradación artificial es parte del método, no un detalle. Un modelo entrenado solo con datos limpios no aprendería a usar el sumidero.
- **El mecanismo es un atenuador, no un filtro.** Un descriptor asignado al 90 % a un fantasma **sigue contribuyendo** con el 10 % restante. No hay descarte duro.
- **La interpretación de «calidad» es post hoc.** El paper observa que la ponderación por calidad emerge, pero nada en la pérdida la exige; lo que emerge es lo que reduce el error de clasificación, que coincide con la intuición de calidad en sus datos.

---

**Ver también:** [NetVLAD (2016)](/papers/netvlad-arandjelovic-2016) (la capa que este paper extiende) · [VLAD (2010)](/papers/vlad-jegou-2010) (el ancestro con `argmin`) · [Utterance-level Aggregation (2019)](/papers/utterance-level-xie-2019) (el traslado al habla, con los mismos K=8 y G=2) · [VoxCeleb2 (2018)](/papers/voxceleb2-chung-2018) · [Lab 41](/laboratorios/lab-41) (donde los centroides fantasma se abren y se miden) · [Agregación VLAD](/fundamentos/agregacion-vlad) · [Clase 41](/clases/clase-41).
