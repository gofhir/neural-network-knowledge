# Convolutional Two-Stream Network Fusion for Video Action Recognition — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Autores:** **Christoph Feichtenhofer** y **Axel Pinz** (Graz University of Technology), **Andrew Zisserman** (University of Oxford, VGG).
- **Venue:** CVPR 2016. Preprint arXiv:1604.06573v2 (26 sep 2016).
- **Código:** `github.com/feichtenhofer/twostreamfusion`, implementado en **MatConvNet**.
- Los autores agradecen discusiones con **Karen Simonyan**, primer autor del two-stream original: este paper es la continuación de esa línea desde dentro del mismo grupo.

El paper toma la arquitectura *two-stream* de Simonyan y Zisserman (NIPS 2014) —una red para apariencia (RGB), otra para movimiento (pilas de flujo óptico), promediadas en el softmax— y se pregunta algo que nadie había respondido con experimentos sistemáticos: **si en lugar de fusionar al final fusionamos en medio, ¿qué pasa?** Las tres conclusiones del abstract:

1. Se puede fusionar en una **capa convolucional** en lugar del softmax **sin pérdida de desempeño y con un ahorro sustancial de parámetros** (de 181.42M a 97.58M, casi la mitad).
2. Es mejor fusionar espacialmente en la **última capa convolucional** que antes; fusionar *adicionalmente* en la capa de predicción da un empujón extra.
3. El **pooling de features convolucionales abstractos sobre vecindarios espacio-temporales** mejora más el desempeño.

Cifras finales, promediadas sobre los tres splits estándar:

| Configuración | UCF-101 | HMDB-51 |
|---|---|---|
| Two-Stream ConvNet original (VGG-M) | 88.0% | 59.4% |
| Este paper (S: VGG-16, T: VGG-M) | **90.8%** | **62.1%** |
| Este paper (S y T: VGG-16, una torre tras fusión) | **91.8%** | **64.6%** |
| Este paper (S y T: VGG-16, dos torres) | **92.5%** | **65.4%** |
| Este paper + iDT (S: VGG-16, T: VGG-M) | **92.5%** | **67.3%** |
| Este paper + iDT (S y T: VGG-16) | **93.5%** | **69.2%** |

Es decir **+3 puntos** sobre el two-stream original con backbone mixto, **+4.5 (UCF) y +6.0 (HMDB)** con VGG-16 en ambas corrientes, y el mejor resultado publicado de 2016 al combinarse con iDT. Para la Clase 38 este paper es exactamente la familia **(d) 3D-Fused Two-Stream** de la figura 5 de I3D: el eslabón que la clase muestra en el diagrama pero no explica.

## 2. Contexto: el Two-Stream original y su limitación estructural

El two-stream de Simonyan y Zisserman descompone el video en dos señales explícitas: fotogramas RGB y pilas de $L=10$ campos de flujo óptico horizontal y vertical. Cada señal alimenta una ConvNet independiente inicializada desde ImageNet, cada una entrenada para clasificar la acción **por su cuenta**, y la predicción final es el **promedio de los scores de softmax**. El diseño era brillante: hacía que la red heredara pre-entrenamiento de imágenes en lugar de aprender movimiento desde cero, y demostró que **el flujo óptico solo ya bastaba para discriminar la mayoría de las acciones de UCF-101**.

El paper identifica dos defectos estructurales:

> (i) no es capaz de aprender las correspondencias píxel a píxel entre features espaciales y temporales (porque la fusión ocurre solo sobre los scores de clasificación), y (ii) está limitado en escala temporal, ya que la ConvNet espacial opera solo sobre fotogramas individuales y la temporal solo sobre una pila de $L$ fotogramas de flujo adyacentes.

El primero merece decirse sin eufemismos: **la red nunca aprende "qué se mueve dónde"**. Si la fusión ocurre en el softmax, lo único que se combina son dos distribuciones sobre 101 clases. La corriente espacial dice "hay una boca, un cepillo"; la temporal dice "hay un movimiento periódico de mano". Nada en la arquitectura permite que un parámetro entrenable represente la conjunción *"movimiento periódico de mano EN la ubicación de la boca"*.

El ejemplo del paper es el más limpio posible: **distinguir "cepillarse los dientes" de "cepillarse el pelo"**. En ambos casos hay una mano moviéndose periódicamente en algún lugar. La corriente temporal reconoce ese movimiento pero no puede decir *dónde* en términos semánticos; la espacial reconoce la ubicación (dientes o pelo) pero no el movimiento. **Solo su combinación en la misma posición espacial discrimina la acción.** El argumento es que si distintos canales de la red espacial responden a distintas áreas faciales y un canal de la temporal responde a ese tipo de campo de movimiento, entonces al apilar los canales **los filtros de las capas siguientes pueden aprender la correspondencia entre los canales apropiados** —como pesos de un filtro de convolución— para discriminar entre esas acciones.

El segundo defecto es complementario: el two-stream original mitigaba en parte la limitación temporal haciendo pooling de predicciones sobre muestras regularmente espaciadas del video, pero eso **no permite modelar la evolución temporal** — promediar 25 predicciones independientes no es lo mismo que aprender que "dibujar una flecha, tensar el arco, disparar" es una secuencia.

El estado del arte de 2016 era además incómodo: los mejores números se obtenían **combinando ConvNets con codificaciones Fisher Vector de features hechos a mano** (HOF sobre dense trajectories). Parte de la explicación era la escala: UCF-101 tiene solo 100 ejemplos por clase contra los 1000 de ImageNet, y la clasificación de acciones añade variabilidad de movimiento y punto de vista.

## 3. Contribución central: tres preguntas de diseño

El aporte es metodológico antes que arquitectónico. En lugar de proponer una red y mostrar que gana, **descompone la fusión en tres ejes, evalúa cada uno por separado, selecciona el óptimo de cada ablation y recién entonces los ensambla**:

1. **¿Cómo fusionar** las dos redes, respetando el registro espacial? (operadores: sum, max, concatenación, conv, bilineal)
2. **¿Dónde fusionar?** (ReLU2..ReLU5, capas FC, múltiples capas)
3. **¿Cómo fusionar temporalmente?** (2D pooling, 3D pooling, 3D conv + 3D pooling)

El paper es explícito en que la formulación de la fusión espacial **no está atada a esta aplicación**: los mismos problemas surgen al fusionar cualesquiera dos redes. Por eso el paper tiene vida útil larga — es un catálogo general de operadores de fusión de dos torres convolucionales.

## 4. Fusión espacial: los cinco operadores

Formalmente, $f: \mathbf{x}^a_t, \mathbf{x}^b_t \to \mathbf{y}_t$ toma dos mapas $\mathbf{x}^a_t \in \mathbb{R}^{H \times W \times D}$ y $\mathbf{x}^b_t \in \mathbb{R}^{H' \times W' \times D'}$ y produce $\mathbf{y}_t \in \mathbb{R}^{H'' \times W'' \times D''}$. Se asume $H = H' = H''$, $W = W' = W''$, $D = D'$, y se omite $t$.

**Sum fusion.**
$$y^{\text{sum}}_{i,j,d} = x^a_{i,j,d} + x^b_{i,j,d}, \qquad 1 \le i \le H,\; 1 \le j \le W,\; 1 \le d \le D$$
con $\mathbf{x}^a, \mathbf{x}^b, \mathbf{y} \in \mathbb{R}^{H \times W \times D}$. El punto conceptual: **la numeración de los canales es arbitraria**, así que la suma impone una correspondencia arbitraria $d$-a-$d$. No hay razón para que el canal 37 de la red espacial deba emparejarse con el 37 de la temporal. El aprendizaje posterior puede *explotar* esa correspondencia optimizando los filtros de cada red, pero no puede *elegirla*.

**Max fusion.**
$$y^{\text{max}}_{i,j,d} = \max\{x^a_{i,j,d},\, x^b_{i,j,d}\}$$
Misma arbitrariedad, más la desventaja de descartar información: en cada posición y canal una de las dos corrientes se pierde por completo.

**Concatenation fusion.** Apila los dos mapas a lo largo de los canales en las mismas posiciones espaciales:
$$y^{\text{cat}}_{i,j,2d} = x^a_{i,j,d}, \qquad y^{\text{cat}}_{i,j,2d-1} = x^b_{i,j,d}$$
con $\mathbf{y} \in \mathbb{R}^{H \times W \times 2D}$ (el paper intercala los canales; el orden es indiferente para las capas posteriores). **No define ninguna correspondencia**: deja que las capas siguientes la definan aprendiendo filtros que ponderen los canales. El costo es que sin una capa de reducción de dimensionalidad la capa siguiente recibe $2D$ canales — y como esa capa es FC6, eso **duplica los parámetros de la primera capa densa**.

**Conv fusion (la ganadora).** Concatena y luego convoluciona con un banco $\mathbf{f} \in \mathbb{R}^{1 \times 1 \times 2D \times D}$ y sesgos $b \in \mathbb{R}^D$:
$$\mathbf{y}^{\text{conv}} = \mathbf{y}^{\text{cat}} * \mathbf{f} + b$$
El filtro es de $1 \times 1 \times 2D$: **puramente en la dimensión de canales, sin extensión espacial**. Reduce la dimensionalidad por un factor de dos y modela combinaciones ponderadas de $\mathbf{x}^a$ y $\mathbf{x}^b$ **en la misma posición de píxel**. Al ser entrenable, $\mathbf{f}$ **aprende las correspondencias entre los dos mapas que minimizan una pérdida conjunta**.

Aquí está el punto teórico más elegante: **conv fusion contiene a sum fusion como caso particular**. Si $\mathbf{f}$ se aprende como la concatenación de dos matrices identidad permutadas $\mathbf{1}' \in \mathbb{R}^{1 \times 1 \times D \times D}$, entonces el canal $i$ de una red se combina *solo* con el canal $i$ de la otra, vía suma. La fusión por convolución es **la generalización aprendible de la suma**: parte de ella y puede desviarse hacia cualquier emparejamiento lineal de canales que reduzca la pérdida.

Eso explica la decisión de implementación que el paper marca como crítica: **la inicialización con matrices identidad**. Los autores encuentran que inicializar así (de modo que la capa *empiece* sumando las dos corrientes) rinde igual que la inicialización aleatoria (85.96% vs 85.59% con ruido gaussiano) **pero con un tiempo de entrenamiento mucho menor**. La lectura es doble y honesta: la inicialización identidad arranca en un punto ya funcional y solo refina; pero el hecho de que el óptimo aleatorio llegue casi al mismo lugar **sugiere que simplemente sumar los mapas ya es una buena técnica de fusión**, y que *"aprender una combinación inicializada aleatoriamente no lleva a resultados significativamente diferentes/mejores"*.

**Bilinear fusion.** Producto externo en cada píxel, seguido de suma sobre las posiciones:
$$\mathbf{y}^{\text{bil}} = \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{x}^{a\top}_{i,j} \, \mathbf{x}^{b}_{i,j}$$
con $\mathbf{y}^{\text{bil}} \in \mathbb{R}^{D^2}$. Captura **interacciones multiplicativas**: **cada canal de una red se combina con cada canal de la otra**, así que la correspondencia es completa y no arbitraria. Las desventajas: la dimensionalidad explota a $D^2$ (262144 para $D=512$) y **toda la información espacial queda marginalizada** por la suma sobre $i,j$. Para usarlo en la práctica se aplica en ReLU5, **se eliminan las capas fully-connected** y se aplican normalizaciones de potencia y $L_2$ para clasificar con SVMs lineales — de ahí su línea peculiar en la tabla (10 capas, 6.61M de parámetros, pero necesita un SVM externo).

El paper menciona alternativas no exploradas: el producto pixel a pixel de canales, y el producto externo (factorizado) **sin** sum pooling sobre las posiciones.

Resultados (dos VGG-M-2048, fusión en la salida de **ReLU5** —después de la rectificación, porque en experimentos preliminares dio mejor que la salida no rectificada de conv5—, una sola corriente después; UCF-101 split 1):

| Método de fusión | Capa | Accuracy | #capas | #parámetros |
|---|---|---|---|---|
| Sum (reportado en two-stream original) | Softmax | 85.6% | 16 | 181.42M |
| Sum (reimplementación de los autores) | Softmax | 85.94% | 16 | 181.42M |
| Max | ReLU5 | 82.70% | 13 | 97.31M |
| Concatenation | ReLU5 | 83.53% | 13 | 172.81M |
| Bilinear | ReLU5 | 85.05% | 10 | 6.61M + SVM |
| Sum | ReLU5 | 85.20% | 13 | 97.31M |
| **Conv** | **ReLU5** | **85.96%** | **14** | **97.58M** |

**Max y concatenación rinden considerablemente peor** que sum y conv, y **conv fusion en ReLU5 iguala exactamente al promedio de softmax (85.96% vs 85.94%) con la mitad de los parámetros** — el hallazgo (i) del abstract. En las capas FC todos los métodos rinden peor que en ReLU5, con el mismo orden entre métodos (salvo bilineal, no aplicable); entre las FC, **FC8 es mejor que FC7 y FC6** (conv 85.9%, sum 85.1%). La explicación de los autores: en ReLU5 **todavía existen las correspondencias espaciales entre apariencia y movimiento**, mientras que en las FC ya han sido colapsadas.

## 5. El requisito de correspondencia espacial

La intención declarada es fusionar de modo que **las respuestas de canal en la misma posición de píxel queden puestas en correspondencia**. Eso impone una restricción dura: los dos mapas deben tener **las mismas dimensiones espaciales**, $H = H'$ y $W = W'$.

Por qué es indispensable: el operador de fusión —cualquiera de los cinco— opera índice por índice sobre $(i,j)$. Si los mapas no están **registrados píxel a píxel**, el operador está sumando, comparando o multiplicando features de lugares distintos de la imagen y toda la premisa del "qué se mueve dónde" se desmorona. La correspondencia espacial es lo que hace que "movimiento periódico" y "boca" se encuentren en el mismo índice del tensor.

Es fácil de conseguir: **si las dos redes tienen la misma resolución espacial en las capas a fusionar, basta con superponer (apilar) las capas de una sobre la otra**. Cuando no coinciden, el paper propone una capa "upconvolucional" o —si las dimensiones son similares— hacer upsampling **rellenando el mapa más pequeño con ceros**. Esto último es lo que hicieron en el único caso donde las resoluciones difieren: al fusionar un VGG-16 en ReLU5\_3 con un VGG-M en ReLU5, **rellenaron la salida ligeramente menor de VGG-M ($13 \times 13$, contra $14 \times 14$) con una fila y una columna de ceros**. Es un gotcha práctico: la arquitectura heterogénea requiere un parche de padding para que el registro funcione.

Vale notar la asimetría de las corrientes: ambas ven el mismo fotograma central $t$, pero la temporal ve una ventana $t \pm L/2$ de flujo. El registro es espacial, no temporal — se asume que el flujo acumulado sobre 10 fotogramas sigue estando aproximadamente en el mismo lugar. La justificación aparece más adelante: en conv5 de VGG-M el stride de entrada es de 16 píxeles y el campo receptivo de $139 \times 139$, lo que da mucha tolerancia a desplazamientos pequeños.

## 6. Dónde fusionar: la tabla por capa

Con conv fusion fijado como operador e inicializado con identidades, el paper barre la capa de fusión (UCF-101 split 1, dos VGG-M):

| Capa(s) de fusión | Accuracy | #capas | #parámetros |
|---|---|---|---|
| ReLU2 | 82.25% | 11 | 91.90M |
| ReLU3 | 83.43% | 12 | 93.08M |
| ReLU4 | 82.55% | 13 | 95.48M |
| **ReLU5** | **85.96%** | **14** | **97.57M** |
| **ReLU5 + FC8** | **86.04%** | **17** | **181.68M** |
| ReLU3 + ReLU5 + FC6 | 81.55% | 17 | 190.06M |

- **Fusionar temprano degrada, y degrada mucho.** ReLU2 (82.25%), ReLU3 (83.43%) y ReLU4 (82.55%) quedan entre 2.5 y 3.7 puntos por debajo de ReLU5, y ni siquiera son monótonos (ReLU4 es *peor* que ReLU3). La intuición: en capas bajas los features de las dos corrientes son de naturalezas demasiado distintas —bordes y texturas de apariencia contra gradientes de campos de flujo— para que una combinación lineal canal a canal signifique algo; y fusionar temprano **destruye una de las dos jerarquías completas**, porque desde el punto de fusión hay una sola torre.
- **El óptimo es ReLU5**, la última capa convolucional: el punto donde los features son "ya altamente informativos pero todavía proveen información gruesa de localización".
- **ReLU5 + FC8 es marginalmente mejor** (86.04% vs 85.96%, +0.08) **pero al costo de casi el doble de parámetros** (181.68M vs 97.57M). Es la configuración de la figura 2 (derecha): se mantienen **ambas torres**, una convertida en red híbrida espacio-temporal y la otra puramente espacial. Consigue el registro píxel a píxel en conv5 **pero no reduce parámetros**.
- **Fusionar en múltiples capas empeora si se incluyen capas tempranas:** ReLU3 + ReLU5 + FC6 cae a 81.55% con 190.06M — el peor resultado con la mayor cantidad de parámetros. Más fusión no es mejor fusión.

El trade-off para el ingeniero es limpio: **+0.08 puntos cuestan +84M de parámetros**. Fusionar solo en ReLU5 y truncar una torre es la elección racional. Notar además que fusionar en capas convolucionales tempranas *no* ahorra tanto como uno esperaría (91.90M vs 97.57M): **la mayoría de los parámetros vive en las capas fully-connected**, así que truncar una torre en ReLU2 o en ReLU5 elimina aproximadamente el mismo bloque. Ahorrar 5.7M sacrificando 3.7 puntos no tiene sentido.

## 7. Fusión temporal

La entrada de la capa de pooling temporal es $\mathbf{x} \in \mathbb{R}^{H \times W \times T \times D}$, generada apilando mapas espaciales sobre $t = 1 \ldots T$. Tres opciones:

- **2D pooling** (Fig. 3a): ignora el tiempo; hace pooling espacial por muestra temporal y las predicciones se promedian después. Es lo que hace el two-stream original.
- **3D pooling** (Fig. 3b): max-pooling sobre un cubo $W' \times H' \times T'$. Si se agrupan tres muestras temporales se usa un $3\times3\times3$ sobre los tres canales correspondientes apilados; **no hay pooling entre canales distintos**. Da invarianza a pequeños cambios de posición de los features en el tiempo, y generaliza el max-pooling temporal que Ng et al. habían identificado entre los mejores agregadores.
- **3D conv + 3D pooling** (Fig. 3c): primero convoluciona la entrada 4D con $D'$ filtros $\mathbf{f} \in \mathbb{R}^{W'' \times H'' \times T'' \times D \times D'}$ y sesgos $b \in \mathbb{R}^{D'}$,
$$\mathbf{y} = \mathbf{x}_t * \mathbf{f} + b$$
y **después** aplica 3D pooling. El vecindario típico es $3 \times 3 \times 3$ (espacial × temporal).

**Por qué la conv 3D antes del pooling temporal importa.** El 3D pooling da invarianza pero es un operador fijo y sin parámetros: no puede representar la *evolución* de un feature, solo su presencia en algún lugar del vecindario. La conv 3D aprende **combinaciones ponderadas de features en un vecindario espacio-temporal local**, y el paper da los dos ejemplos canónicos de lo que un filtro así puede aprender: **ponderar centralmente la muestra temporal central**, o **diferenciar en el tiempo o en el espacio**. Ese segundo caso es la clave — un filtro que aproxima $\partial/\partial t$ detecta *cambio* de feature, no presencia de feature. Es la diferencia entre "hubo movimiento periódico en algún momento" y "el movimiento se aceleró y luego se detuvo". Hacer el pooling *después* preserva esa capacidad y solo entonces le agrega invarianza posicional.

Justificación de combinar pooling espacial y temporal en un operador 3D: **los features pueden cambiar de posición espacial con el tiempo**. Con el conv5 de VGG-M (stride 16, campo receptivo $139\times139$), el pooling espacio-temporal de mapas separados por $\tau$ fotogramas **puede capturar features del mismo objeto aunque se mueva ligeramente**.

**Muestreo temporal de la entrada.** La capa de fusión temporal recibe $T$ chunks separados por $\tau$ fotogramas: las torres se aplican en $t, t+\tau, \ldots, t+T\tau$. Esto crea dos escalas: **fina** en la entrada de la red temporal ($t \pm \frac{L}{2}$, con $L = 10$), que captura primitivas de movimiento como "el trazado de una flecha"; y **gruesa** en la capa de fusión ($t + T\tau$), que las pone en contexto: "dibujar una flecha, tensar el arco, disparar". El campo receptivo temporal total es **$T \times L$**. Detalle importante: **$\tau < L$ produce entradas solapadas para la corriente temporal, mientras que $\tau \ge L$ produce features temporalmente no solapados**.

Resultados (VGG-16 espacial + VGG-M temporal, split 1):

| Método de fusión | Pooling | Capas | UCF-101 | HMDB-51 |
|---|---|---|---|---|
| 2D Conv | 2D | ReLU5 + | 89.35% | 56.93% |
| 2D Conv | 3D | ReLU5 + | 89.64% | 57.58% |
| **3D Conv** | **3D** | **ReLU5 +** | **90.40%** | **58.63%** |

El "+" indica que **ambas redes y ambas pérdidas se mantienen después de fusionar**, porque eso rinde mejor que truncar una. Concretamente: en ReLU5 se fusiona **desde la red temporal hacia la espacial**, luego se hace 2D o 3D pooling en pool5 y se computa una pérdida por torre; en test se promedian las predicciones FC8 de ambas. Las tres filas construyen el argumento incrementalmente: pasar de 2D a **3D pooling** gana (+0.29 UCF, **+0.65 HMDB**) y añadir el **filtro 3D** gana otro salto (+0.76 UCF, +1.05 HMDB). Que HMDB gane más es consistente: es un dataset donde el contexto de escena ayuda menos y la dinámica importa más.

## 8. La arquitectura final completa

La arquitectura propuesta (Fig. 4) es la extensión temporal de la fusión de Fig. 2 (izquierda): se fusionan las dos redes **en la última capa convolucional (después de ReLU), desde la corriente temporal hacia la espacial**, convirtiendo esta última en una **corriente espacio-temporal** mediante **3D Conv fusion seguida de 3D pooling**; **no se trunca la corriente temporal** (también recibe 3D pooling y se mantiene como rama puramente temporal); **se usan las pérdidas de ambas corrientes** para entrenar y en test se promedian sus predicciones.

El kernel de fusión 3D tiene dimensión $3 \times 3 \times 3 \times 1024 \times 512$ con $T = 5$. Desglose: $H'' \times W'' \times T'' = 3\times3\times3$; $D = 1024$ resulta de **concatenar los ReLU5 de ambas corrientes** ($512+512$); y $D' = 512$ **coincide con el número de canales de entrada de la capa FC6 siguiente**.

**Backbones.** Dos modelos pre-entrenados en ImageNet: **VGG-M-2048** (5 conv + 3 FC), por comparabilidad con el two-stream original, y **VGG-16** (13 conv + 3 FC). Las corrientes se entrenan primero por separado (split 1):

| Modelo | UCF-101 VGG-M-2048 | UCF-101 VGG-16 | HMDB-51 VGG-M-2048 | HMDB-51 VGG-16 |
|---|---|---|---|---|
| Espacial | 74.22% | 82.61% | 36.77% | 47.06% |
| Temporal | 82.34% | 86.25% | 51.50% | 55.23% |
| Late fusion (promedio de predicciones) | 85.94% | 90.62% | 54.90% | 58.17% |

El hallazgo es asimétrico y muy citado: **pasar a un modelo espacial más profundo mejora significativamente (+8.11 en UCF, +10.29 en HMDB), mientras que un modelo temporal más profundo da una ganancia menor (+3.91 y +3.73)**. Esto anticipa un tema que atraviesa toda la literatura posterior: la corriente de apariencia **hereda directamente el beneficio de las arquitecturas de imagen y de ImageNet**, la de flujo mucho menos. Es exactamente el problema que I3D resolverá inflando.

**Pre-entrenamiento e inicialización.** Ambas corrientes parten de ImageNet. La temporal **también se inicializa desde ImageNet**, con una justificación reveladora: lo hacen porque *facilita la velocidad de entrenamiento sin disminución de desempeño comparado con el modelo entrenado desde cero* — es decir, ImageNet le da **convergencia, no precisión**. Para HMDB-51, las redes temporales se inicializan desde los modelos temporales de UCF-101. El flujo usa Brox et al. y **TV-L1** para VGG-16.

La inicialización de la capa de fusión trae el gotcha más fino del paper. Los filtros de 3D Conv se inicializan **apilando dos matrices identidad** para mapear 1024 canales a 512. Pero *las activaciones de la ConvNet temporal en la última capa convolucional son aproximadamente 3 veces menores que las de su contraparte de apariencia*, así que **la identidad temporal de $\mathbf{f}$ se inicializa con un factor 3 más alto** para que la fusión no quede dominada por la apariencia. La parte espacio-temporal de $\mathbf{f}$ usa una **gaussiana $3\times3\times3$ con $\sigma = 1$**. Además, **no se fusiona en la capa de predicción durante el entrenamiento**, porque sesgaría la pérdida hacia la arquitectura temporal: la espacio-temporal necesita más tiempo para adaptarse a los features fusionados.

**Optimización.** Fine-tuning con batch de 96 videos, learning rate inicial $10^{-3}$ reducido 10× cuando la accuracy de validación se satura (schedule adaptativo, no fijo); para VGG-16 es $5\times10^{-4}$ (el PDF lo imprime como "50−4", errata evidente). **Solo se retropropaga hasta la capa de fusión inyectada**, porque la retropropagación completa no dio mejoras. **No se usa batch normalization.** El flujo se **precomputa y se almacena como imágenes JPEG**, con clipping de desplazamientos mayores a 20 píxeles; el lado menor del fotograma se reescala a 256.

**Manejo del sobreajuste.** El paper advierte que *entrenar 3D ConvNets es aún más propenso al sobreajuste que la fusión de two-stream ConvNets*:

- **Dropout más bajo** de lo habitual: 0.85 en las dos primeras FC de la red espacial (bajarlo hasta 0.5 no degradó significativamente). Sin RGB colour jittering.
- **Aumentación temporal:** en cada iteración se muestrean los $T=5$ fotogramas de cada uno de los 96 videos eligiendo aleatoriamente el fotograma inicial **y el stride temporal $\tau \in [1,10]$**, con lo que la red opera sobre entre **15 y 50 fotogramas** de extensión total. Es aumentación de *escala temporal*, habilitada por el diseño de muestreo de §7.
- **Multi-scale cropping con jitter de aspect-ratio:** en lugar de un parche fijo de $224\times224$, jitter aleatorio de ancho y alto en **±25%** y reescalado a $224\times224$, lo que **puede cambiar el aspect-ratio**. Los parches se recortan a un máximo de 25% de distancia de los bordes. Para VGG-16 se añade muestreo de esquinas y centro.
- **Consistencia del crop en la pila:** posición, tamaño, escala y flip se eligen aleatoriamente **en el primer fotograma** y **el mismo crop se aplica a todos los fotogramas** — indispensable, o se rompería el registro.

**Testing.** En las ablations solo se muestrean los $T=5$ fotogramas y sus flips (contra los 25 del two-stream original), deliberadamente para acelerar la evaluación, con **testing totalmente convolucional** sobre el fotograma completo en lugar de crops. Para la comparación final se promedian **20 predicciones temporales de cada red**.

## 9. Resultados

Tres splits de UCF-101 y HMDB-51 (Tabla 5):

| Método | UCF-101 | HMDB-51 |
|---|---|---|
| Spatiotemporal ConvNet (Karpathy et al.) | 65.4% | — |
| LRCN (Donahue et al.) | 82.9% | — |
| Composite LSTM Model (Srivastava et al.) | 84.3% | 44.0% |
| C3D (Tran et al.) | 85.2% | — |
| Two-Stream ConvNet (VGG-M, Simonyan-Zisserman) | 88.0% | 59.4% |
| Factorized ConvNet (Sun et al.) | 88.1% | 59.1% |
| Two-Stream Conv Pooling (Ng et al.) | 88.2% | — |
| Two-Stream ConvNet (VGG-16, Wang et al.) | 91.4% | 58.5% |
| Two-Stream ConvNet (VGG-16, reimplementación de los autores) | 91.7% | 58.7% |
| **Ours (S: VGG-16, T: VGG-M)** | **90.8%** | **62.1%** |
| **Ours (S y T: VGG-16, una torre tras fusión)** | **91.8%** | **64.6%** |
| **Ours (S y T: VGG-16)** | **92.5%** | **65.4%** |

Y con iDT (Tabla 6):

| Método | UCF-101 | HMDB-51 |
|---|---|---|
| iDT + Fisher Vector de alta dimensión (Peng et al.) | 87.9% | 61.1% |
| C3D + iDT | 90.4% | — |
| TDD + iDT (Wang et al.) | 91.5% | 65.9% |
| **Ours + iDT (S: VGG-16, T: VGG-M)** | **92.5%** | **67.3%** |
| **Ours + iDT (S y T: VGG-16)** | **93.5%** | **69.2%** |

Comparaciones que el paper destaca. Contra el **two-stream original**, mejora de **~3% en ambos datasets** con espacial VGG-16 + temporal VGG-M, y de **4.5% (UCF) y 6% (HMDB)** con VGG-16 en ambas corrientes. Contra **Two-Stream Conv Pooling** de Ng et al. —que aplica conv-pooling temporal tras la última capa de reducción de dimensionalidad de una GoogLeNet y reporta 88.2% haciendo pooling sobre 120 fotogramas, 88.6% con LSTM— el 92.5% *"subraya claramente la importancia del enfoque propuesto"*, y con una huella temporal mucho menor. **Una sola corriente tras la fusión temporal da 91.8% contra 92.5% con dos**, pero con muchos menos parámetros y una arquitectura más simple.

El resultado más interesante intelectualmente es el de **iDT**: sigue habiendo una mejora sustancial al combinar predicciones ConvNet con features hechos a mano codificados con Fisher Vector (+1.0 en UCF y +3.8 en HMDB sobre el mejor modelo puro). Los autores lo llaman "intrigante" y sospechan que *la diferencia puede desaparecer con el tiempo dados muchísimos más datos de entrenamiento*, pero que por ahora **indica dónde debe apuntar la investigación futura**. Con Kinetics un año después esa sospecha se confirmó.

**Sobre el conteo de parámetros.** Sum fusion en el softmax requiere **las 16 capas y los 181.42M de parámetros de las dos torres completas** de VGG-M-2048. Max, Sum y Conv fusion en ReLU5 **eliminan casi la mitad** (97.31M / 97.58M) porque **después de la fusión se usa un solo conjunto de capas fully-connected**. La razón estructural: en VGG-M la abrumadora mayoría de los parámetros está en FC6/FC7/FC8, así que truncar una torre en la última capa convolucional borra un bloque denso completo. Conv fusion paga apenas 0.27M extra por el filtro $1\times1\times1024\times512$. La concatenación, por no reducir dimensionalidad, **duplica los parámetros de FC6** y llega a 172.81M. Conclusión del paper: *la nueva arquitectura no aumenta significativamente el número de parámetros respecto de métodos previos y aun así excede el estado del arte*.

## 10. Limitaciones

- **Sigue dependiendo del flujo óptico precomputado.** El flujo se computa **antes del entrenamiento** con Brox et al. o TV-L1 y se **almacena como JPEG** con clipping a 20 píxeles. Eso implica un preprocesamiento costoso fuera de la red (TV-L1 es iterativo y en 2016 era caro incluso en GPU), almacenamiento no trivial (dos campos por fotograma para todo el dataset) y un pipeline no entrenable end-to-end. El clipping y la compresión JPEG además **introducen pérdida de información** en la señal de movimiento. La arquitectura *fusiona* mejor, pero no elimina la dependencia.
- **La fusión sigue siendo tardía en términos temporales.** Espacialmente el paper mueve la fusión de la capa 16 a la 14, un avance real. Pero temporalmente el modelado ocurre **solo en la capa alta de fusión**: las torres procesan cada chunk de forma independiente hasta ReLU5. Es un modelo de "features 2D por snippet, luego mezcla temporal", no una red espacio-temporal de punta a punta.
- **Ventana temporal limitada.** El campo receptivo temporal total es $T \times L$ con $T=5$ y $L=10$: entre **15 y 50 fotogramas** según el $\tau$ muestreado, o entre 0.6 y 2 segundos a 25 fps. El propio ejemplo del paper ("dibujar una flecha, tensar el arco, disparar") apenas cabe; I3D entrena sobre 64 fotogramas (2.56 s) y en test procesa el video completo.
- **Fusión temprana no descartada, solo no lograda.** Que ReLU2/3/4 degraden no prueba que la fusión temprana sea intrínsecamente mala: prueba que **este operador** (conv $1\times1$ con inicialización identidad, truncando una torre) no funciona temprano.
- **Datasets demasiado pequeños o demasiado ruidosos.** El paper cierra con una advertencia honesta: *"volvemos al punto de que los datasets actuales son demasiado pequeños o demasiado ruidosos. Por esta razón, algunas de las conclusiones de este paper deben tratarse con cautela."* Todas las ablations críticas se corrieron sobre **un solo split de UCF-101** y las diferencias que deciden el diseño son del orden de 0.1 a 0.8 puntos. Es exactamente el diagnóstico que I3D convertiría en su premisa.
- **Backpropagation parcial.** Solo se retropropaga hasta la capa de fusión, así que las corrientes no se **co-adaptan** realmente: la promesa de "aprender correspondencias que minimizan una pérdida conjunta" se cumple en la capa de fusión, no en toda la jerarquía.

## 11. Impacto y legado

**Hacia I3D (2017).** Un año después Carreira y Zisserman —el mismo Zisserman— re-evalúan el zoo de arquitecturas de video sobre un backbone común (Inception-v1) y el nuevo dataset Kinetics. Este paper aparece ahí como la **familia (d) 3D-Fused Two-Stream** de la figura 5. La reimplementación de I3D toma **5 fotogramas RGB consecutivos muestreados cada 10** más sus snippets de flujo; las grillas de features de $5\times7\times7$ (tiempo, x, y) pasan por una **conv 3D de $3\times3\times3$ con 512 canales**, un **max-pooling 3D de $3\times3\times3$** y una capa fully-connected — una traducción fiel del diseño de Feichtenhofer, con los pesos nuevos inicializados con ruido gaussiano en lugar de identidades. Con **39M de parámetros**, la familia (d) obtiene en UCF-101 (entrenando solo en UCF-101, sin Kinetics) **83.2% (RGB) / 85.8% (flujo) / 89.3% (RGB+flujo)**. La familia (e), Two-Stream I3D, la supera con solo **25M de parámetros**.

Lo importante es *por qué* la supera, porque es el diagnóstico exacto de la limitación de este paper: **3D-Fused tiene features 2D en toda la jerarquía y solo mezcla en el tiempo al final**; I3D tiene features 3D desde conv1. La conv 3D de fusión de Feichtenhofer es, en retrospectiva, **una única capa espacio-temporal encima de una pila enteramente espacial**. I3D infla toda la pila.

**Hacia SlowFast (2019).** El linaje más directo va del mismo primer autor. En *SlowFast Networks for Video Recognition* (Feichtenhofer, Fan, Malik, He; ICCV 2019) las dos corrientes ya no son RGB y flujo óptico sino **dos frecuencias de muestreo del mismo RGB**: una vía *Slow* de baja tasa de fotogramas y alta capacidad para semántica espacial, y una vía *Fast* de alta tasa y baja capacidad (canales reducidos) para movimiento. Lo que las une son **conexiones laterales que fusionan la vía Fast en la Slow en múltiples etapas de la jerarquía** — la descendiente directa de este paper, con la misma convicción de que **dos torres deben intercambiar información en capas intermedias, no solo en el softmax**. SlowFast hereda también el requisito de correspondencia espacial (por eso las conexiones laterales deben reconciliar formas de tensor entre vías) y resuelve la dependencia del flujo óptico prescindiendo de él: la vía Fast *es* el detector de movimiento aprendido.

**Otros ecos.** El catálogo de operadores de §4 se volvió referencia estándar para fusión de dos torres mucho más allá del video, y el propio grupo lo extendió en *Spatiotemporal Residual Networks* (NIPS 2016) y *Spatiotemporal Multiplier Networks* (CVPR 2017), donde conexiones residuales entre corrientes reemplazan la capa de fusión única.

## 12. Conexión con la Clase 38

La Clase 38 (*Redes Convolucionales para reconocimiento en video — Modelos pre-entrenados*) abre con la figura 5 de I3D: cinco familias, (a) LSTM, (b) 3D-ConvNet, (c) Two-Stream, (d) 3D-Fused Two-Stream, (e) Two-Stream I3D. **Este paper es (d)**, y en la tabla de la clase es la fila de **39M de parámetros con 83.2 / 85.8 / 89.3 en UCF-101**. La clase lo muestra como una caja en un diagrama; este análisis explica qué hay dentro.

**Por qué es el puente entre Two-Stream y las 3D ConvNets.** Las dos familias parecían ortogonales: (c) aprende movimiento inyectándolo como entrada explícita con redes 2D; (b) lo aprende con filtros espacio-temporales desde el RGB crudo. Este paper es el primer híbrido serio: **conserva la entrada de flujo óptico de (c) y le añade el operador de (b) —convolución 3D— pero solo en la capa de fusión**. Leído en la dirección del tiempo, es el momento en que el campo se convence de dos cosas que I3D consolidaría: que la información de las dos corrientes debe combinarse **con parámetros aprendidos y respetando la correspondencia espacial**, no con un promedio; y que **el tiempo debe modelarse con convoluciones, no solo con pooling o promediado de predicciones**. Cada una de esas convicciones, llevada al extremo, produce I3D.

**Lo que dice sobre el pre-entrenamiento — el problema que I3D resuelve inflando.** Este es el punto de contacto más importante con el eje de la clase, y el paper lo documenta con números sin poder resolverlo. La **corriente espacial hereda ImageNet limpiamente**: es una ConvNet 2D que recibe una imagen RGB, y el pre-entrenamiento es *exactamente* la tarea correcta, de ahí sus **+8.11 / +10.29 puntos** al pasar a VGG-16. La **corriente temporal no puede heredarlo de la misma forma**: su entrada son 20 canales de campos de flujo, no tres canales RGB; los autores la inicializan igual desde ImageNet pero solo por **velocidad de convergencia, no por precisión**, y la profundidad le rinde menos de la mitad (**+3.91 / +3.73**). Incluso la escala de sus activaciones difiere (~3× menor), lo que obliga al hack del factor 3. Y la **capa de fusión 3D no puede heredar nada**: identidades para los canales, gaussiana $\sigma=1$ para la parte $3\times3\times3$. Cada parámetro nuevo destinado a modelar tiempo hay que aprenderlo desde cero con 100 videos por clase — de ahí el énfasis obsesivo en aumentación y la advertencia de que *entrenar 3D ConvNets es aún más propenso al sobreajuste*.

Ese es el cuello de botella que I3D resuelve. El *inflado* con el *boring-video fixed point* permite que **toda** la red 3D, no solo la corriente espacial, arranque con pesos de ImageNet, repitiendo los filtros 2D $N$ veces en el tiempo y dividiendo por $N$. Con eso desaparece la asimetría que este paper documenta sin poder corregir, y desaparece la necesidad de confinar el modelado temporal a una capa delgada al final para no sobreajustar. La lección para la clase es doble: **la fusión intermedia con parámetros aprendidos es la idea correcta** —SlowFast la vindica en 2019— pero **sin una fuente de pre-entrenamiento adecuada para la parte espacio-temporal, esa idea solo puede aplicarse con cuentagotas**. Feichtenhofer tenía la arquitectura; le faltaba Kinetics.

---

**Nota final — lecciones transferibles.** Para quien deba montar un clasificador de eventos en video con datos escasos, tres lecciones sobreviven a la arquitectura concreta. Primero, **dónde fusionar modalidades importa más que cómo**: mover la fusión de la última capa a la penúltima ganó más que elegir entre sum, max o conv, y fusionar demasiado temprano costó 3.7 puntos. Segundo, **la fusión intermedia es un ahorro de parámetros, no solo una mejora de precisión**: truncar una torre en la última capa convolucional eliminó casi la mitad de los parámetros a igualdad de accuracy, porque los parámetros viven en las capas densas. Tercero, y quizás la más útil, **la inicialización identidad de un módulo de fusión es una técnica general**: hacer que un módulo nuevo empiece siendo la operación trivial —aquí, la suma— y solo pueda mejorar desde ahí añade capacidad sin desestabilizar un modelo ya entrenado, el mismo principio que subyace a las conexiones residuales, a los adapters y a LoRA.
