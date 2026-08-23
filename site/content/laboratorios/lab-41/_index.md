---
title: "Lab 41 - Speaker Recognition con Thin ResNet + GhostVLAD"
weight: 410
sidebar:
  open: true
---

**Profesor:** Gabriel Sepúlveda (IALab, Departamento de Ciencia de la Computación, PUC)
**Módulo:** Audio — reconocimiento de hablante
**Notebook origen:** `clase_41/material/Laboratorio/speaker_recognition.ipynb`
**Notebook ejecutado:** [lab41.ipynb](/notebooks/lab41.ipynb) · [HTML](/notebooks-html/lab41.html)

## Encuadre

La contraparte práctica de la segunda mitad de la [clase 41](/clases/clase-41): reimplementar en PyTorch el modelo de **[Xie et al. 2019](/papers/utterance-level-xie-2019)** —Thin ResNet-34 + [GhostVLAD](/papers/ghostvlad-zhong-2018)—, cargar pesos entrenados en [VoxCeleb2](/papers/voxceleb2-chung-2018) y reproducir su EER sobre los 37.720 pares de verificación de [VoxCeleb1](/papers/voxceleb-nagrani-2017). No se entrena nada. El práctico pide ejecutar las celdas y responder tres preguntas de selección múltiple.

De las 39 celdas del notebook, solo 12 son código, y ninguna implementa algo nuevo: es un lab de **lectura de arquitectura**. Lo que lo hace interesante es que el objeto de estudio —un checkpoint de 46 MB— es un archivo que se puede abrir.

Y al abrirlo, el modelo que produce el número correcto **no se parece al que describe el paper**.

{{< concept-alert type="clave" >}}
**El resultado se reproduce: EER 3,19 % contra el 3,22 % publicado.** Pero el modelo que lo produce tiene la **mitad del backbone apagada** (50,2 % de los canales con filtros encogidos a 10⁻³³, y 70,9 % en el bloque que concentra el 71 % de los parámetros); sus **8 centroides «discriminativos» colapsaron** a coseno 0,9983 entre sí, así que no particionan ningún espacio; y **2 de sus 10 centroides son parámetros muertos** que el paper de GhostVLAD dice explícitamente que no deben existir.

Nada de eso impide que funcione. El número es correcto. La explicación de por qué funciona, tal como la cuenta el paper, no describe a este modelo.
{{< /concept-alert >}}

La tesis del lab, en una línea: **se entrena un clasificador de 8.631 hablantes y en inferencia se le corta la cabeza.** Los 4.419.072 parámetros de `dense_1` —el 36,6 % del modelo— se cargan del checkpoint, se transfieren a la GPU y nunca se ejecutan. La clasificación no es el objetivo: es el pretexto para que el gradiente esculpa un espacio de 512 dimensiones donde la distancia significa identidad. Por eso funciona con los 40 hablantes de VoxCeleb1-test, que el modelo nunca vio.

## Resultados consolidados (medidos)

### El número del lab

| | EER | Umbral |
|---|---|---|
| **Esta ejecución** | **3,19 %** | 0,7757 |
| [Xie et al. 2019](/papers/utterance-level-xie-2019), Thin ResNet-34 + GhostVLAD | 3,22 % | — |
| El mismo backbone con TAP (promedio temporal) | 10,48 % | — |

Cambiar el agregador —`mean` por una capa de 40 mil parámetros— mueve el error **2,9×**. Los 8 centroides son **5.120 parámetros: el 0,04 % del modelo**, y son la mejor inversión por parámetro de toda la arquitectura.

### La distribución de los scores

| | media | desv. | Interpretación |
|---|---|---|---|
| mismo hablante | **0,8760** | 0,0496 | |
| distinto hablante | **0,6474** | 0,0661 | el «cero» del sistema no está en 0 |
| separabilidad d′ | **3,910** | | |

El piso de 0,647 no es un accidente: **el 65,4 % de la energía de cada embedding es una dirección común a las 4.715 grabaciones** (‖μ‖ = 0,8088 sobre vectores de norma 1). Solo el 34,6 % restante distingue a los 40 hablantes.

### El experimento propio que refuta la predicción

| Transformación | EER | Δ | d′ |
|---|---|---|---|
| baseline (celda 27) | **3,192 %** | — | 3,910 |
| **centrado** (restar μ y renormalizar) | **3,266 %** | −0,074 pts | **4,021** ⬆ |
| centrado + whitening diagonal | 3,600 % | −0,408 pts | — |

**El centrado mejora la separabilidad y empeora el error.** No es contradicción: d′ solo usa medias y varianzas, el EER depende de dónde se cruzan las colas. Con d′ = 4,021 el EER *gaussiano* sería 2,2 %; el real es 3,27 %, y ese punto de diferencia es la no-gaussianidad. Y los 0,074 puntos son **0,58 σ** del error estándar de estimación: la transformación no degrada, simplemente no hace nada.

### Lo que se encontró al abrir el checkpoint

| Medición | Valor | Contra qué |
|---|---|---|
| `dense_1.weight` | **(8631, 512)** | VoxCeleb2 tiene **5.994** hablantes. 8631 son las identidades de VGGFace2 |
| Canales con filtro ≈ 0 | **3.518 / 7.008 = 50,2 %** | «Thin» ResNet de 3,69 M parámetros |
| ...en `block5` | **2.541 / 3.584 = 70,9 %** | `block5` tiene el 71 % de los parámetros |
| Coseno entre los 8 centroides | **0,9983** | *"trainable discriminative clustering"* |
| Norma de los 2 centroides fantasma | **1,000 exacto** | inicialización ortogonal intacta |
| Perturbar los fantasmas ×1000 | coseno **0,99999994** | parámetros muertos |
| `num_batches_tracked` | **0** en las 38 BN | firma de conversión desde Keras |

### Mediciones sobre la arquitectura

| Afirmación del paper / código | Medición | |
|---|---|---|
| descriptores de `T/32` (Tabla 1 y texto del paper) | **T/16** | el código oficial usa `strides=(2,1)`; **la errata es del paper** |
| «descriptor local» de 160 ms | campo receptivo **1.840 ms**, solape **91,3 %** | cada instante está representado ~11 veces |
| espejado `np.append(wav, wav[::-1])` | coseno **0,999813** | duplica el cómputo, cambia el 0,02 % |
| `T/16` contra `T/32` | coseno **0,999279** | la errata es numéricamente inconsecuente |
| eps de BatchNorm: Keras 1e-3 vs PyTorch 1e-5 | error relativo **3,74 %** | 170 canales vivos con var < 1e-3 |
| normalización «time-wise» (comentario del código) | **por frame**, sobre las 257 bandas | amplifica el silencio **37×** |

### El presupuesto de parámetros

| Componente | Params | % | ¿se usa? |
|---|---|---|---|
| `thin_resnet` (34 capas conv) | 3.690.240 | 30,5 % | ✅ (mitad muerta) |
| `block_1` (conv 7×1, 512→512) | 1.835.520 | 15,2 % | ✅ |
| `vlad_conv` (asignación, 512→10) | 35.850 | 0,3 % | ✅ |
| **`vlad_pooling.cluster`** (los `c_k`) | **5.120** | **0,04 %** | ✅ (2 muertos) |
| `block_2` (4096→512) | 2.097.664 | 17,4 % | ✅ |
| **`dense_1`** (512→8631) | **4.419.072** | **36,6 %** | ❌ **nunca** |
| **total** | **12.083.466** | | **7.664.394 útiles** |

## Bloques del lab

{{< cards >}}
  {{< card link="01-el-dataloader-y-la-normalizacion" title="El dataloader y el eje de la normalización" subtitle="El espejado que duplica el cómputo para cambiar el 0,02 %, el crop amputado que dejó su parámetro muerto, y el eje de normalización que el mismo grupo invirtió entre 2017 y 2019 después de llamarlo crucial" icon="adjustments" >}}
  {{< card link="02-el-thin-resnet-y-la-errata" title="El Thin ResNet, la errata y el campo receptivo" subtitle="Las 34 capas contadas, la Tabla 1 del paper de ICASSP que dice T/32 cuando el código oficial de sus autores dice T/16, y el descriptor «local» que ve 1,84 segundos" icon="beaker" >}}
  {{< card link="03-netvlad-desarmado" title="NetVLAD desarmado" subtitle="Por qué el argmin bloquea el gradiente, la identidad algebraica que convierte distancias en un softmax lineal, el broadcast de 5 dimensiones y la intra-normalización que vuelve VLAD ciego al número de descriptores" icon="variable" >}}
  {{< card link="04-el-checkpoint-abierto" title="El checkpoint abierto" subtitle="46 MB que contradicen al paper: la mitad del backbone encogida a 10⁻³³, los 8 centroides con coseno 0,9983, los 2 fantasmas que nunca recibieron gradiente y el 8631 que viene de un dataset de caras" icon="cube-transparent" >}}
  {{< card link="05-el-eer-y-la-direccion-comun" title="El EER, el umbral y la dirección común" subtitle="calculate_eer verificado contra fuerza bruta, por qué el umbral está en 0,776 y no cerca de cero, el 65,4 % de energía compartida, y el centrado que mejora d′ mientras empeora el EER" icon="chart-bar" >}}
  {{< card link="06-los-defectos-del-notebook" title="Los defectos del notebook" subtitle="El ampersand sin comillas que manda wget a segundo plano, el unzip que cuelga la celda esperando stdin, el nn.Parameter sin inicializar que puede traer NaN, y el print de 37.720 líneas que Colab truncó" icon="code" >}}
  {{< card link="07-las-tres-actividades" title="Las tres actividades" subtitle="Las respuestas con su justificación en las slides y en el código, y por qué cada distractor falla — incluido el que confunde los N descriptores de entrada con los K de salida" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/reconocimiento-de-hablante" title="Reconocimiento de Hablante" subtitle="Identificación contra verificación, el descriptor de conjunto abierto, EER y curva ROC" icon="book-open" >}}
  {{< card link="/fundamentos/agregacion-vlad" title="Agregación VLAD" subtitle="De contar ocupación a acumular residuos, y el argmin que había que volver derivable" icon="book-open" >}}
  {{< card link="/fundamentos/metric-learning" title="Metric Learning" subtitle="Aprender un espacio donde la distancia significa identidad — el porqué de cortar la cabeza de clasificación" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de Audio" subtitle="STFT, magnitud y fase: el tensor de 257 × T que entra a la red" icon="book-open" >}}
  {{< card link="/fundamentos/redes-convolucionales" title="Redes Convolucionales" subtitle="El bloque bottleneck, el stride en la 1×1 y el campo receptivo" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-voz" title="Reconocimiento de Voz" subtitle="La otra mitad de la clase: la tarea que necesita exactamente lo contrario de este modelo" icon="book-open" >}}
{{< /cards >}}

## Papers de este laboratorio

{{< cards >}}
  {{< card link="/papers/utterance-level-xie-2019" title="Utterance-level Aggregation (2019)" subtitle="Xie et al. — el modelo del lab. Su Tabla 1 declara T/32 y el código de sus autores dice T/16; su tabla de resultados es el 3,22 % que el lab reproduce" icon="document-text" >}}
  {{< card link="/papers/ghostvlad-zhong-2018" title="GhostVLAD (2018)" subtitle="Zhong, Arandjelović y Zisserman — el origen de los ghost_clusters=2, y la frase que dice que {c_k} solo tiene K elementos: los 2 centroides fantasma del lab no debían existir" icon="document-text" >}}
  {{< card link="/papers/netvlad-arandjelovic-2016" title="NetVLAD (2016)" subtitle="Arandjelović et al. — el softmax que reemplaza al argmin y el desacople de w_k, b_k y c_k que el lab verifica numéricamente" icon="document-text" >}}
  {{< card link="/papers/vlad-jegou-2010" title="VLAD (2010)" subtitle="Jégou et al. — el ancestro con asignación dura, y el diagrama de Voronoi que el modelo entrenado ya no describe" icon="document-text" >}}
  {{< card link="/papers/voxceleb-nagrani-2017" title="VoxCeleb (2017)" subtitle="Nagrani et al. — los 40 hablantes y 37.720 pares del test, y la normalización por bin de frecuencia que Xie 2019 invirtió" icon="document-text" >}}
  {{< card link="/papers/voxceleb2-chung-2018" title="VoxCeleb2 (2018)" subtitle="Chung et al. — los 5.994 hablantes de entrenamiento, disjuntos del test: la razón de que el open-set funcione" icon="document-text" >}}
  {{< card link="/papers/x-vectors-snyder-2018" title="x-vectors (2018)" subtitle="Snyder et al. — statistics pooling, y el linaje de centering + LDA + PLDA que en este modelo no aporta" icon="document-text" >}}
  {{< card link="/papers/resnet-he-2015" title="ResNet (2015)" subtitle="He et al. — el bloque bottleneck y la variante «original» con stride en la 1×1 que el lab replica para que los pesos encajen" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Clase 41 - Teoría](/clases/clase-41/teoria) · [Clase 41 - Profundización](/clases/clase-41/profundizacion) · [Clase 41 - Práctica](/clases/clase-41/practica) (CTC y VLAD desde cero en triple framework) · [Lab 39 - Onda cruda y VGGish](/laboratorios/lab-39) · [Lab 37 - Datasets y Herramientas para Audio](/laboratorios/lab-37) · [Lab 26 - Meta-aprendizaje](/laboratorios/lab-26) (el mismo patrón: clasificar para aprender, medir distancias para desplegar) · Dominio [Audio](/dominios/audio).
