# TSM: Temporal Shift Module for Efficient Video Understanding — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *TSM: Temporal Shift Module for Efficient Video Understanding*.
- **Autores:** Ji Lin (MIT), Chuang Gan (MIT-IBM Watson AI Lab), Song Han (MIT).
- **Venue:** *International Conference on Computer Vision (ICCV 2019)*.
- **Año:** 2019. **Preprint:** arXiv:1811.08383v3 (22 ago 2019). Código y modelos: [github.com/mit-han-lab/temporal-shift-module](https://github.com/mit-han-lab/temporal-shift-module).
- **Linaje:** construye directamente sobre [TSN](../../clase_36/papers/Wang-TSN-2016.md) (Wang et al., ECCV 2016) —del que hereda el muestreo por segmentos, el consenso por promedio y hasta el nombre de la clase en el código— y se posiciona como alternativa a [I3D](../../clase_36/papers/Carreira-I3D-2017.md) (Carreira y Zisserman, CVPR 2017) y [C3D](../../clase_36/papers/Tran-C3D-2015.md). La idea del desplazamiento como primitiva viene del *shift* espacial de Wu et al. (2018) para clasificación de imágenes, que los autores muestran que **no funciona** trasladado ingenuamente al eje temporal.

El paper ataca la tensión que define el reconocimiento de acciones eficiente: **las CNN 2D son baratas pero ciegas al tiempo; las CNN 3D modelan el tiempo pero son caras de entrenar y de desplegar**. La propuesta es un módulo —el **Temporal Shift Module (TSM)**— que se inserta dentro de una CNN 2D cualquiera y le da capacidad de modelado espacio-temporal **con cero parámetros y cero FLOPs adicionales**. El mecanismo es de una simplicidad casi provocadora: desplazar una fracción de los canales del mapa de características a lo largo del eje temporal, de modo que cada frame reciba información de sus vecinos, y dejar que la convolución 2D que viene inmediatamente después haga la mezcla.

El argumento formal es que **una convolución se descompone en desplazamiento más multiplicación-acumulación**. TSM ejecuta el desplazamiento en la dimensión temporal —que no cuesta FLOPs, solo movimiento de datos— y *pliega* la multiplicación-acumulación dentro de la convolución 2D que la red ya iba a ejecutar de todos modos. El resultado es un modelo pseudo-3D con el presupuesto computacional de uno 2D.

Los números que sostienen el argumento tienen dos caras. Sobre **Something-Something V1**, un dataset donde el orden temporal es indispensable, TSM lleva a su baseline 2D de **20,5 % a 47,3 %** de top-1 —27 puntos— y alcanza el primer lugar del leaderboard al momento de publicación. Sobre **Kinetics**, donde la apariencia estática ya resuelve la mayoría de los casos, la ganancia es de **70,6 % a 74,1 %**: 3,5 puntos. Esa asimetría no es un defecto del método sino su lectura más informativa, y es el hilo que conviene seguir para entender qué mide realmente cada benchmark de video.

En eficiencia, TSM con ResNet-50 y 8 frames corre a **17,4 ms de latencia y 77,4 videos/s** en una Tesla P100, contra los **165,3 ms y 6,1 videos/s** del I3D con el que se compara: **un orden de magnitud** más rápido, con 1,8 puntos más de precisión. Y su variante unidireccional habilita reconocimiento en vivo con **13,4 ms en una Jetson Nano y 34,5 ms en un Galaxy Note8**.

---

## 2. Contexto: el costo de modelar el tiempo

Hacia 2018 el campo tenía tres familias de respuestas al problema de introducir el tiempo en una red convolucional, y las tres tenían un defecto conocido.

**Las CNN 2D puras** (TSN y sucesores) procesan frames de forma independiente y agregan las predicciones al final. Son baratas, heredan ImageNet sin fricción y escalan bien, pero su agregación —típicamente un promedio— **es invariante al orden**: reproducir el video al revés produce exactamente la misma salida. Pueden reconocer *qué* aparece en un video; no pueden inferir *en qué orden* pasa. El paper lo dice sin rodeos: *"cannot infer the temporal order or more complicated temporal relationships"*.

**Las CNN 3D** (C3D, I3D) aprenden features espacio-temporales conjuntos, pero pagan por ello en tres monedas. Cómputo: el I3D de referencia consume 306 GFLOPs por video. Parámetros: más que sus contrapartes 2D, lo que las hace más propensas al sobreajuste. Y despliegue: la latencia las excluye de aplicaciones en tiempo real y de dispositivos de borde.

**Los enfoques intermedios** —fusión tardía tras la extracción de features, fusión de nivel medio, arquitecturas mixtas 2D+3D como ECO, descomposiciones de la convolución 3D en una 2D espacial más una 1D temporal como R(2+1)D— negocian expresividad por cómputo. El paper les hace una objeción precisa: *"Such methods sacrifice the low-level temporal modeling for efficiency, but much of the useful information is lost during the feature extraction before the temporal fusion happens"*. Si la fusión temporal ocurre recién después de la extracción de features, la información de bajo nivel que se perdió en el camino no se recupera.

TSM se propone como una cuarta vía: **fusión temporal en todos los niveles de la red, al costo de una CNN 2D**.

---

## 3. La intuición: separar el desplazamiento de la multiplicación

El punto de partida es una observación sobre la estructura de la convolución. Sea una convolución 1-D de kernel 3 con pesos $W = (w_1, w_2, w_3)$ sobre una entrada $X$:

$$Y_i = w_1 X_{i-1} + w_2 X_i + w_3 X_{i+1}$$

Esta operación se descompone en dos pasos. Primero, **desplazar**:

$$X^{-1}_i = X_{i-1}, \qquad X^{0}_i = X_i, \qquad X^{+1}_i = X_{i+1}$$

Y después, **multiplicar y acumular**:

$$Y = w_1 X^{-1} + w_2 X^{0} + w_3 X^{+1}$$

El primer paso no requiere ninguna multiplicación: es reindexar memoria. El segundo es el caro. La jugada de TSM es hacer el desplazamiento **en el eje temporal** y dejar que la multiplicación-acumulación la absorba la convolución 2D siguiente, que opera sobre el eje de canales. En palabras de los autores: *"we shift in the time dimension by ±1 and fold the multiply-accumulate from time dimension to channel dimension"*.

El resultado es que el modelado temporal **no aparece en el conteo de FLOPs ni en el de parámetros**. La convolución 2D que ya existía, al recibir canales que ahora contienen features de $t-1$, $t$ y $t+1$, puede aprender pesos que implementan una convolución temporal de kernel 3 sin que nadie haya agregado una operación.

---

## 4. Por qué el desplazamiento ingenuo no funciona

El aporte del paper no es la idea del desplazamiento —que ya existía en el dominio espacial— sino el diagnóstico de por qué su traslado directo fracasa y las dos correcciones que lo rescatan. Esta sección es la más valiosa del trabajo y la que suele omitirse al resumirlo.

Aplicar la estrategia de *shift* espacial al eje temporal, desplazando todos o casi todos los canales, produce **dos desastres simultáneos**:

**Peor eficiencia, por movimiento de datos.** El desplazamiento es gratis en FLOPs pero no en memoria: mover datos incrementa el *memory footprint* y la latencia. El efecto se agrava en redes de video porque sus activaciones son tensores 5D de gran tamaño. Medido: desplazar todos los canales cuesta **+13,7 % de latencia en CPU y +12,4 % en GPU**. Para una operación que se suponía gratuita, es una cifra que la descalifica.

**Peor precisión, por degradación del modelado espacial.** Al desplazar un canal hacia el frame vecino, la información que ese canal contenía **deja de estar disponible para el frame actual**. Si se desplazan demasiados, el backbone 2D pierde capacidad de representar la apariencia. Medido: **−2,6 puntos** de precisión respecto de la baseline TSN.

Las dos correcciones:

### 4.1. Partial shift

En lugar de desplazar todos los canales, se desplaza **una fracción pequeña**. La medición de latencia sobre ResNet-50 con 8 frames, en Tesla P100, Jetson TX2 e Intel Xeon E5-2690, muestra la misma tendencia en los tres dispositivos: con 1/8 de los canales el overhead cae a **≈3 %**, frente al 13,7 % del desplazamiento total.

Del lado de la precisión, el barrido sobre Kinetics dibuja una curva con máximo interior: si la proporción es demasiado pequeña, la capacidad de razonamiento temporal no alcanza; si es demasiado grande, se daña el aprendizaje espacial. El óptimo está en **1/4 del total, es decir 1/8 por dirección**:

> *"For residual shift, we found that the performance reaches the peak when 1/4 (1/8 for each direction) of the channels are shifted."*

Este es el punto que más se distorsiona al citar el paper. El nombre del checkpoint oficial (`shift8`) y la frase "TSM reemplaza 1/8 del mapa de características" inducen a pensar que se desplaza un octavo del tensor. El código desmiente esa lectura: `fold = C // 8`, y se mueven **dos** folds, uno hacia el pasado y otro hacia el futuro. El octavo es por dirección; el total es un cuarto. La cifra de 1/8 sí es correcta para el modo online, que es unidireccional.

### 4.2. Residual shift

La segunda corrección es *dónde* se inserta el módulo. La opción directa —ponerlo antes de cada capa convolucional o de cada bloque residual— se llama **in-place shift** y arrastra el problema de la información perdida: lo que se desplazó ya no está para el frame actual.

La alternativa es el **residual shift**: insertar TSM **dentro de la rama residual** del bloque, no antes de la bifurcación. Así la conexión identidad sigue transportando la activación completa sin desplazar, y toda la información original permanece accesible aguas abajo:

> *"Residual shift can address the degraded spatial feature learning problem, as all the information in the original activation is still accessible after temporal shift through identity mapping."*

La comparación empírica sobre Kinetics muestra que el residual shift **supera al in-place en todas las proporciones probadas**. Y hay un detalle contundente: incluso desplazando el 100 % de los canales, el residual shift sigue por encima de la baseline 2D, algo imposible para el in-place. En el código del repositorio esta decisión es el argumento `shift_place='blockres'` (contra `'block'`), y se materializa envolviendo la `conv1` de cada bloque bottleneck:

```python
blocks[i].conv1 = TemporalShift(b.conv1, n_segment=this_segment, n_div=n_div)
```

---

## 5. El módulo, operación por operación

Sobre un tensor de activaciones $A \in \mathbb{R}^{N \times C \times T \times H \times W}$, con `fold = C // fold_div` y `fold_div = 8`, la implementación completa son tres asignaciones:

```python
out = torch.zeros_like(x)                                 # x: (N, T, C, H, W)
out[:, :-1, :fold]        = x[:, 1:, :fold]               # futuro  -> presente
out[:, 1:, fold:2*fold]   = x[:, :-1, fold:2*fold]        # pasado  -> presente
out[:, :,  2*fold:]       = x[:, :,  2*fold:]             # sin desplazar
```

Leído sobre el eje de canales, para el frame en el instante $t$:

| Rango de canales | Contenido | Fracción |
|---|---|---|
| $[0,\; C/8)$ | features de $t+1$ | 12,5 % |
| $[C/8,\; C/4)$ | features de $t-1$ | 12,5 % |
| $[C/4,\; C)$ | features de $t$ | 75 % |

Tres propiedades que se derivan del código y conviene tener presentes:

**El relleno es con ceros.** `torch.zeros_like` implica que el primer frame no recibe pasado y el último no recibe futuro: en ambos extremos, 1/8 de los canales queda en cero. Con 16 módulos encadenados, los bordes temporales del clip acumulan una degradación sistemática que el centro no sufre.

**El campo receptivo temporal crece +2 por módulo.** Cada TSM insertado equivale a una convolución temporal de kernel 3, de modo que apilar módulos amplía la cobertura de forma acumulativa. En ResNet-50 se insertan **16 módulos** (3+4+6+3 bloques bottleneck), suficiente para cubrir los 8 segmentos varias veces.

**El módulo depende de $T$.** La operación `x.view(n_batch, n_segment, c, h, w)` reinterpreta la dimensión de batch para identificar frames vecinos. Un modelo entrenado con $T=8$ asume esa agrupación en inferencia; cambiar el número de segmentos cambia la semántica del desplazamiento.

Una nota de implementación: el repositorio incluye una versión *in-place* (`InplaceShift`) que evita la copia del tensor, pero está deshabilitada con un `raise NotImplementedError` por errores de orden en ejecución paralela. La versión que se usa reserva un tensor nuevo con `zeros_like`.

### 5.1. Escalado a otras arquitecturas

Para backbones más profundos el paper reduce la densidad de inserción. La condición está en el código: si `layer3` tiene 23 o más bloques —el caso de ResNet-101— se activa `n_round = 2` y TSM se inserta solo en uno de cada dos bloques, acotando el costo de movimiento de datos. Con ResNet-50 (6 bloques en `layer3`) todos los bloques reciben módulo.

---

## 6. Los dos modos: offline y online

### 6.1. Offline, bidireccional

Es el modo por defecto. Se muestrean $T$ frames del video con la estrategia de segmentos de TSN, se inserta un TSM bidireccional en cada bloque residual y se promedian las predicciones por frame. Requiere el clip completo de antemano y alcanza la máxima precisión. El paper subraya que la ventaja de despliegue es que **solo hace falta soportar operaciones de CNN 2D**, ya optimizadas a nivel de framework (cuDNN, MKL-DNN, TVM) y de hardware (CPU/GPU/TPU/FPGA).

### 6.2. Online, unidireccional

Para flujos en vivo los frames futuros no existen, de modo que solo se desplaza pasado → presente. La implementación es un **caché**: para cada frame se guardan en memoria los primeros 1/8 de los mapas de features de cada bloque residual; al llegar el frame siguiente se reemplaza su primer 1/8 por el cacheado y se combina 7/8 actual + 1/8 antiguo.

Sus tres ventajas, según el paper:

1. **Latencia baja.** Solo hay que reemplazar y cachear 1/8 de los features, sin cómputo adicional. Se emite una predicción por frame, en vez de esperar a acumular un clip.
2. **Memoria baja.** Para ResNet-50 el caché ocupa **0,9 MB**.
3. **Fusión temporal en todos los niveles.** A diferencia de los métodos online que solo fusionan al final o en niveles medios, el caché opera en cada bloque residual.

### 6.3. El resultado que la clase no menciona

La Tabla 6 del paper compara ambos modos, y su lectura es más interesante de lo que sugiere el discurso habitual de "offline es más preciso porque ve el futuro":

| Modelo | Latencia | Kinetics | UCF-101 | HMDB-51 | Something-V1 |
|---|---|---|---|---|---|
| TSN (baseline 2D) | 4,7 ms | 70,6 % | 91,7 % | 64,7 % | 20,5 % |
| + TSM offline (bidireccional) | — | 74,1 % | **95,9 %** | 73,5 % | **47,3 %** |
| + TSM online (unidireccional) | 4,8 ms | **74,3 %** | 95,5 % | **73,6 %** | 46,3 % |

En **Kinetics y HMDB-51 el modo online iguala o supera al offline**, y en UCF-101 pierde 0,4 puntos. Solo en Something-Something —el dataset donde el orden temporal es constitutivo— el offline gana con claridad, y aun así por apenas 1,0 punto. El acceso a los frames futuros, que suele presentarse como la razón de ser del modo offline, aporta poco o nada en los datasets dominados por apariencia.

El costo en latencia por frame de convertir TSN en TSM online es de **menos de 0,1 ms** (4,7 → 4,8 ms), a cambio de hasta 25 puntos de precisión.

---

## 7. Resultados

### 7.1. La mejora sobre la baseline 2D depende del dataset

Tabla 1 del paper, con ResNet-50, 8 frames, 10 clips para Kinetics y 2 para el resto, resolución completa:

| Dataset | TSN | TSM | Δ |
|---|---|---|---|
| Kinetics | 70,6 % | 74,1 % | **+3,5** |
| UCF-101 | 91,7 % | 95,9 % | +4,2 |
| HMDB-51 | 64,7 % | 73,5 % | +8,8 |
| Something-Something V1 | 20,5 % | 47,3 % | **+28,0** |
| Something-Something V2 | 30,4 % | 61,7 % | **+31,3** |
| Jester | 83,9 % | 97,0 % | +11,7 |

El paper divide explícitamente la tabla en dos mitades: arriba los datasets *"where temporal relationships are less important"*, abajo los que *"depend heavily on temporal relationships"*. La misma modificación arquitectónica produce +3,5 o +31,3 puntos según qué se esté midiendo, lo que convierte a esta tabla en un instrumento de medición sobre los benchmarks tanto como sobre el modelo. Que Kinetics —el dataset canónico de reconocimiento de acciones— se mueva tan poco es una afirmación incómoda sobre cuánto tiempo hace falta realmente para resolverlo.

### 7.2. Escala sobre distintos backbones

| Backbone | MobileNet-V2 | ResNet-50 | ResNeXt-101 | NL ResNet-50 |
|---|---|---|---|---|
| TSN | 66,5 % | 70,7 % | 72,4 % | 74,6 % |
| TSM | 69,5 % | 74,1 % | 76,3 % | 75,7 % |
| Δ | +3,0 | +3,4 | +3,9 | +1,1 |

La ganancia se sostiene desde una red móvil hasta una ResNeXt-101. El caso interesante es el último: sobre un backbone que **ya tiene modelado temporal** (el módulo non-local), TSM todavía aporta 1,1 puntos, lo que sugiere que capturan cosas distintas y que son composables.

### 7.3. Contra el estado del arte en Something-Something V1

| Modelo | Backbone | Frames | FLOPs/video | Params | Top-1 |
|---|---|---|---|---|---|
| TSN (impl. propia) | ResNet-50 | 8 | 33 G | 24,3 M | 19,7 % |
| TRN-Multiscale | BNInception | 8 | 16 G | 18,3 M | 34,4 % |
| ECO | BNIncep+3D Res18 | 8 | 32 G | 47,5 M | 39,6 % |
| ECO_En Lite | BNIncep+3D Res18 | 92 | 267 G | 150 M | 46,4 % |
| I3D | 3D ResNet-50 | 32×2 clips | 153 G ×2 | 28,0 M | 41,6 % |
| Non-local I3D | 3D ResNet-50 | 32×2 clips | 168 G ×2 | 35,3 M | 44,4 % |
| NL I3D + GCN | 3D ResNet-50+GCN | 32×2 clips | 303 G ×2 | 62,2 M | 46,1 % |
| **TSM** | **ResNet-50** | **8** | **33 G** | **24,3 M** | **45,6 %** |
| **TSM** | **ResNet-50** | **16** | **65 G** | **24,3 M** | **47,2 %** |
| TSM_En | ResNet-50 | 24 | 98 G | 48,6 M | 49,7 % |
| TSM RGB+Flow | ResNet-50 | 16+16 | — | 48,6 M | 52,6 % |

Las comparaciones que el paper destaca: contra ECO con 8 frames, **+4,2 puntos con 1,9× menos cómputo**; contra Non-local I3D, **+1,2 puntos con 10× menos FLOPs**. Y el ensemble TSM_En supera a ECO_En Lite usando 2,7× menos cómputo y 3,1× menos parámetros.

### 7.4. Latencia y throughput

Tabla 5, medida en una NVIDIA Tesla P100 (batch 1 para latencia, 16 para throughput):

| Modelo | FLOPs | Params | Latencia | Throughput | Sth-V1 | Kinetics |
|---|---|---|---|---|---|---|
| I3D | 306 G | 35,3 M | 165,3 ms | 6,1 v/s | 41,6 % | — |
| ECO 16F | 64 G | 47,5 M | 30,6 ms | 45,6 v/s | 41,4 % | — |
| I3D (inflado solo de la 1×1) | 33 G | 29,3 M | 25,8 ms | 42,4 v/s | — | 73,3 % |
| **TSM 8F** | **33 G** | **24,3 M** | **17,4 ms** | **77,4 v/s** | **45,6 %** | **74,1 %** |
| TSM 16F | 65 G | 24,3 M | 29,0 ms | 39,5 v/s | 47,2 % | 74,7 % |

Contra el I3D de referencia: **9,5× menos latencia, 12,7× más throughput y 1,8 puntos más de precisión**. Contra ECO: 1,75× menos latencia, 1,7× más throughput, +2 puntos.

El experimento de control más elegante de esta sección es `I3D_replace`: reemplazar cada primitiva TSM por una convolución $3\times1\times1$ —misma función de mezcla temporal, ahora con parámetros y FLOPs— y medir. Resulta **más lento y menos preciso**. Los autores atribuyen la diferencia a que TSM solo ejecuta convoluciones 2D, un kernel altamente optimizado en hardware, mientras que las convoluciones 3D no gozan del mismo soporte. El argumento no es solo de complejidad asintótica: es de qué operaciones están bien implementadas en el silicio que existe.

### 7.5. Reconocimiento temprano, detección y dispositivos de borde

**Reconocimiento temprano** (UCF-101, Figura 6). Observando solo el **10 % de los frames**, TSM alcanza **90 %** de precisión, 6,6 puntos por encima del mejor ECO. Es la métrica que importa cuando hay que responder antes de que el evento termine.

**Detección de objetos en video** (ImageNet-VID, Tabla 7). Inyectando el TSM unidireccional en el backbone de un R-FCN con ResNet-101:

| Modelo | Online | Necesita flujo | Latencia | mAP total | Lento | Medio | **Rápido** |
|---|---|---|---|---|---|---|---|
| R-FCN | ✓ | | 1× | 74,7 | 83,6 | 72,5 | 51,4 |
| FGFA | | ✓ | 2,5× | 75,9 | 84,0 | 74,4 | 55,6 |
| **Online TSM** | **✓** | | **1×** | **76,3** | 83,4 | 74,8 | **56,0** |

La mejora se concentra en los **objetos que se mueven rápido**: +4,6 mAP sobre la baseline 2D, justo donde el desenfoque de movimiento y la oclusión degradan la apariencia de un frame aislado. Y supera a FGFA —que agrega 21 frames con flujo óptico, 10 pasados y 10 futuros— siendo causal y 2,5× más rápido.

**Dispositivos de borde** (Tabla 8), con backbone MobileNet-V2 online (69,5 % en Kinetics), compilado con TVM:

| Dispositivo | Jetson Nano (GPU) | Jetson TX2 (GPU) | Raspberry Pi 4B | Galaxy Note8 | Pixel-1 |
|---|---|---|---|---|---|
| Latencia | 13,4 ms | 8,5 ms | 69,6 ms | 34,5 ms | 47,4 ms |
| Potencia | 4,5 W | 5,8 W | 3,8 W | — | — |

---

## 8. Limitaciones

- **El movimiento de datos no es gratis, aunque los FLOPs sí.** El paper es honesto en esto y de hecho construye su diseño alrededor del problema, pero la consecuencia queda: "cero costo computacional" describe el conteo de FLOPs, no el tiempo de pared. Con la proporción elegida el overhead es de ~3 %, no de 0 %.
- **El desplazamiento es rígido.** Siempre ±1 frame, siempre la misma fracción de canales, siempre en las mismas posiciones. No hay nada aprendido en el módulo: ni cuánto desplazar, ni qué canales, ni con qué alcance temporal. Trabajos posteriores (TDN, TAM, temporal shift aprendido) atacan exactamente ese punto.
- **Depende del número de segmentos con el que se entrenó.** El módulo reinterpreta la dimensión de batch según $T$; cambiar $T$ en inferencia altera la semántica del desplazamiento y degrada el rendimiento.
- **Los bordes temporales quedan con ceros.** El primer y el último frame del clip reciben relleno nulo en una fracción de sus canales, en cada uno de los 16 módulos. El paper no discute este efecto ni evalúa alternativas de padding.
- **La ganancia sobre Kinetics es modesta.** +3,5 puntos sobre una baseline 2D es una mejora real pero pequeña, y buena parte del atractivo del método en ese dataset viene del lado de la eficiencia, no de la precisión.
- **Los mejores resultados siguen necesitando flujo óptico.** La configuración RGB+Flow aporta +5,4 puntos en Something-V1 y +2,6 en V2, con el costo de precomputar TV-L1 —que, como reconoce el paper, suele ser mayor que el del propio modelo de reconocimiento.

---

## 9. Por qué importa para la Clase 40

La Clase 40 (Analítica de Videos — Reconocimiento de acciones) presenta TSN y TSM como una progresión, y su laboratorio hace inferencia con el checkpoint oficial de TSM ResNet-50 sobre Kinetics-400. Tres puntos donde el paper corrige o precisa el material de clase:

**La proporción desplazada.** La slide afirma que "TSM reemplaza 1/8 del mapa de características del fotograma actual con información antigua". Eso describe el modo **unidireccional**; en el bidireccional del checkpoint del laboratorio se desplaza **1/4** del tensor, 1/8 por dirección. El código lo confirma.

**Offline contra online.** La lámina del material se titula "Modelos offline con desplazamiento unidireccional", cuando la sección correspondiente del paper es *"4.2 Online Models with Uni-directional TSM"*. Unidireccional equivale a **online**: el sentido de prescindir de los frames futuros es poder procesar un stream en vivo.

**Lo que la clase no cubre y explica el diseño.** Ni el *partial shift* ni el *residual shift* aparecen en el material, y son las dos correcciones sin las cuales el método no funciona: sin la primera se pierde la eficiencia, sin la segunda se pierde la precisión. Toda la sección 4 de este análisis es material que el paper considera su contribución técnica central.

El laboratorio permite verificar varias de estas afirmaciones sin entrenar nada. Anulando los 16 módulos con `fold_div` grande —de modo que `fold = 0` y el módulo se vuelve la identidad— el modelo se reduce a un TSN 2D con los mismos pesos, y la caída de precisión mide cuánto de la predicción dependía del desplazamiento. Barriendo `fold_div` se reconstruye la forma de la curva de proporción. Y reemplazando el método `shift` por su variante unidireccional se compara el modo online contra el offline. Los tres experimentos están documentados en el [Laboratorio 40](../../site/content/laboratorios/lab-40/).

Para el dominio clínico la lectura útil es la de eficiencia. Un modelo de video que corre a 17 ms en GPU de servidor y a 13 ms en una Jetson Nano cambia lo que es desplegable: monitoreo continuo de gestos quirúrgicos, análisis de video endoscópico en tiempo real durante el procedimiento, seguimiento de movilidad en sala. Y el resultado de la Tabla 6 —que el modo online iguala al offline en datasets dominados por apariencia— indica que la restricción de causalidad, que en un entorno clínico en vivo es obligatoria, no cuesta precisión salvo en tareas donde el orden de los eventos es el objeto mismo de la clasificación.

---

## 10. Referencias cruzadas

- [TSN (Wang et al., 2016)](../../clase_36/papers/Wang-TSN-2016.md) — el marco del que TSM hereda muestreo por segmentos y consenso; la baseline contra la que se mide todo.
- [I3D (Carreira y Zisserman, 2017)](../../clase_36/papers/Carreira-I3D-2017.md) — la alternativa 3D: inflar en vez de desplazar.
- [C3D (Tran et al., 2015)](../../clase_36/papers/Tran-C3D-2015.md) — la 3D desde cero, el punto de partida del problema de costo.
- [Kinetics (Kay et al., 2017)](../../clase_36/papers/Kay-Kinetics-2017.md) — el dataset del checkpoint; su insensibilidad temporal explica los +3,5 puntos.
- [Something-Something (Goyal et al., 2017)](../../clase_36/papers/Goyal-SomethingSomething-2017.md) — el dataset donde TSM gana 28 puntos; el contraste que da sentido a la Tabla 1.
- [UCF-101 (Soomro et al., 2012)](../../clase_36/papers/Soomro-UCF101-2012.md) — los videos del laboratorio.
- [R(2+1)D (Tran et al., 2018)](../../clase_38/papers/Tran-R2plus1D-2018.md) y [S3D (Xie et al., 2018)](../../clase_38/papers/Xie-S3D-2018.md) — la otra familia de respuestas al costo de la convolución 3D: factorizar en vez de eliminar.
