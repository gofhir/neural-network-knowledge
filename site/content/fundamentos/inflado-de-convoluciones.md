---
title: "Inflado de Convoluciones"
weight: 126
math: true
---

**Inflar** (*inflate*) una convolución es tomar los pesos de una capa convolucional 2D ya entrenada y expandirlos a un kernel 3D de modo que la red resultante, evaluada sobre un video, arranque calculando exactamente lo mismo que la red original calculaba sobre una imagen. Es la respuesta técnica a una pregunta muy concreta: si existe un checkpoint de ImageNet con millones de imágenes de supervisión adentro, y hace falta una red que opere sobre volúmenes espacio-temporales, ¿se puede heredar esos pesos en lugar de descartarlos? La técnica se formalizó en [I3D](/papers/i3d-carreira-2017) (Carreira y Zisserman, CVPR 2017) bajo el nombre *boring-video fixed point*, y hoy es el punto de partida por defecto de casi cualquier red convolucional 3D entrenada con datos escasos —incluidas las que trabajan sobre volúmenes médicos, donde el eje inflado no es el tiempo sino la profundidad anatómica. Este fundamento presenta el mecanismo de forma autónoma: la condición matemática que lo define, la tabla capa por capa de qué se reescala y qué no, cómo cambia en arquitecturas separables, su vida fuera del video y sus límites. Es un caso particular —el más limpio y el más útil— de [transfer learning](/fundamentos/transfer-learning) entre arquitecturas de distinta dimensionalidad.

---

## 1. El problema: los tensores no calzan

El obstáculo tiene una capa mecánica y una capa epistémica, y conviene separarlas porque solo la primera parece un problema de software.

**Mecánica.** Los pesos de una convolución 2D viven en un tensor de forma $C_{\text{out}} \times C_{\text{in}} \times k \times k$. Los de una convolución 3D, en uno de forma $C_{\text{out}} \times C_{\text{in}} \times t \times k \times k$. No hay reshape que los conecte: el tensor 3D tiene $t$ veces más entradas. Un `load_state_dict` falla, y falla con razón —el checkpoint 2D **no contiene información** sobre qué debe hacer la red a lo largo del eje temporal. La pregunta no es cómo silenciar el error, sino qué valores poner en las posiciones nuevas.

**Epistémica.** Esa falta de una respuesta obvia tuvo consecuencias históricas grandes, porque en video no existía un sustituto. La comparación es brutal:

| Recurso | Escala | Consecuencia |
|---|---|---|
| ImageNet (2012) | 1,2 millones de imágenes etiquetadas | Cualquier CNN 2D arranca preentrenada |
| HMDB-51 (2011) | 6.766 clips, 51 clases | Insuficiente para entrenar una red profunda |
| UCF-101 (2012) | 13.320 clips, pero de solo ~2.500 videos distintos | Poca variación real; casi cualquier arquitectura rinde parecido |
| [Kinetics](/papers/kinetics-kay-2017) (2017) | 400 clases, ~240.000 clips de entrenamiento, un clip por video | Recién acá hay un "ImageNet del video" |

La familia de redes 3D cargó con esa desventaja durante casi una década: [Ji et al.](/papers/3d-cnn-ji-2013) propusieron la convolución 3D para reconocimiento de acciones alrededor de 2010, y la línea llegó hasta [C3D](/papers/c3d-tran-2015) en 2015 sin poder aprovechar ImageNet. El síntoma más visible fue la **profundidad**: mientras las CNN 2D pasaban de 8 a 152 capas, las redes 3D quedaban forzosamente someras (del orden de 8 capas), porque su dimensionalidad de parámetros multiplicada por la escasez de datos de video hacía imposible entrenarlas desde cero.

{{< concept-alert type="clave" >}}
El inflado no resuelve un problema de forma de tensores: resuelve un problema de **datos**. Antes de Kinetics, el preentrenamiento de ImageNet era la única fuente de supervisión masiva disponible para una red que quisiera procesar píxeles, y las redes 3D eran las únicas excluidas de ese recurso. Inflar es la operación que las incluye.
{{< /concept-alert >}}

Para el contexto general del campo —qué es un video, por qué el movimiento lo cambia todo— ver [Análisis de Video](/fundamentos/analisis-de-video) y la [Clase 36](/clases/clase-36).

---

## 2. La idea: el punto fijo del video aburrido

La construcción conceptual es tan simple que suena a truco: **una imagen repetida $T$ veces es un video perfectamente válido**. Es un video sin movimiento —un video aburrido— pero tiene la forma exacta que espera una red 3D. Llamemos $B_T$ a esa operación de embedding, que toma una imagen $x$ y devuelve el clip

$$\tilde{x}[c, t, i, j] \;=\; x[c, i, j] \qquad \forall\, t \in \{1,\dots,T\}$$

Con $B_T$ a mano se puede formular una **exigencia** sobre la red inflada, en lugar de improvisar una inicialización. Si $f_{\text{2D}}$ es la red original y $f_{\text{3D}}$ la inflada, se pide que

$$f_{\text{3D}}\big(B_T(x)\big) \;=\; f_{\text{2D}}(x) \qquad \text{para toda imagen } x$$

Es decir: **la red 2D debe ser un punto fijo de la red 3D restringida a los videos aburridos**. Ese es todo el contenido de la idea, y tiene dos virtudes que la vuelven mucho más que una heurística:

1. **Es verificable.** No es una intuición sobre qué inicialización "debería andar bien": es una igualdad que se convierte en un test unitario y se corre sobre el checkpoint inflado antes de gastar una sola GPU-hora.
2. **Garantiza un punto de partida bueno, no solo razonable.** En el instante inicial, la red 3D tiene exactamente la pérdida que tendría el modelo 2D de ImageNet aplicado cuadro a cuadro. No es un punto aleatorio del espacio de pesos: es uno que ya clasifica imágenes bien.

La consecuencia es que el modelo 3D queda *implícitamente preentrenado en ImageNet* —los beneficios del preentrenamiento se extienden a las 3D ConvNets, algo que ninguna red 3D desde cero podía capturar.

---

## 3. El mecanismo formal: la condición es sobre la suma

La exigencia del punto fijo se traduce en una ecuación explícita sobre los pesos. Vale derivarla capa por capa, porque el resultado es más general que la receta que suele citarse.

Sea una convolución 2D con pesos $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$ y sesgo $b$:

$$y[c_o, i, j] \;=\; \sum_{c_i} \sum_{u,v} W[c_o, c_i, u, v]\; x[c_i,\, i+u,\, j+v] \;+\; b[c_o]$$

Y la convolución 3D correspondiente, con pesos $\widetilde{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times N \times k \times k}$ donde $N$ es la extensión temporal del kernel, aplicada al video aburrido $\tilde{x}$:

$$\tilde{y}[c_o, t, i, j] \;=\; \sum_{c_i} \sum_{\tau=1}^{N} \sum_{u,v} \widetilde{W}[c_o, c_i, \tau, u, v]\; \tilde{x}[c_i,\, t+\tau,\, i+u,\, j+v] \;+\; b[c_o]$$

Como $\tilde{x}$ **no depende de $t$**, el índice temporal de la entrada es irrelevante y la suma sobre $\tau$ se factoriza:

$$\tilde{y}[c_o, t, i, j] \;=\; \sum_{c_i} \sum_{u,v} \underbrace{\left(\sum_{\tau=1}^{N} \widetilde{W}[c_o, c_i, \tau, u, v]\right)}_{\text{peso efectivo}} x[c_i,\, i+u,\, j+v] \;+\; b[c_o]$$

Comparando con la ecuación 2D, la igualdad $\tilde{y}[c_o,t,i,j] = y[c_o,i,j]$ se cumple **si y solo si**

$$\boxed{\;\sum_{\tau=1}^{N} \widetilde{W}[c_o, c_i, \tau, u, v] \;=\; W[c_o, c_i, u, v]\;}$$

{{< concept-alert type="clave" >}}
**La condición es sobre la suma, no sobre cada peso.** Lo que el punto fijo exige es que los pesos inflados **sumen** el peso 2D original a lo largo del eje temporal. Cada peso 2D se reparte entre $N$ posiciones temporales con $N-1$ grados de libertad, así que hay **infinitas** inicializaciones válidas. El reparto uniforme $W/N$ es la elección de I3D, no una necesidad matemática.
{{< /concept-alert >}}

Dos de esas infinitas soluciones aparecen en implementaciones reales:

**Reparto uniforme** (el de I3D):

$$\widetilde{W}[c_o,c_i,\tau,u,v] \;=\; \frac{1}{N} W[c_o,c_i,u,v] \qquad \forall \tau$$

**Delta central** (identidad temporal), común en las implementaciones de inflado de ResNet:

$$\widetilde{W}[c_o,c_i,\tau,u,v] \;=\; \begin{cases} W[c_o,c_i,u,v] & \text{si } \tau = \lceil N/2 \rceil \\ 0 & \text{en otro caso}\end{cases}$$

Las dos producen exactamente la misma salida sobre un video aburrido. **Difieren sobre video real**, y la diferencia es interpretable:

| Inicialización | Comportamiento inicial sobre video real | Sesgo que introduce | Riesgo |
|---|---|---|---|
| Uniforme ($W/N$) | Promedia temporalmente antes de convolucionar: filtro pasa-bajos en el tiempo | El modelo asume estabilidad temporal | Suaviza el movimiento; arranca ciego a cambios rápidos |
| Delta central | Ignora los cuadros vecinos: es exactamente la red 2D aplicada cuadro a cuadro | Ninguno; el modelo arranca "sin opinión" sobre el tiempo | El gradiente debe descubrir los taps vecinos desde cero |

En la práctica ambas funcionan y la elección se trata como hiperparámetro. Lo importante es saber que existen: **los checkpoints inflados no son intercambiables entre convenciones**, y cargar uno uniforme en un grafo que espera delta central (o al revés) introduce un factor de escala $N$ que degrada todo silenciosamente, sin lanzar ninguna excepción.

---

## 4. Qué se infla y qué no

Esta es la parte que más tiempo ahorra en la práctica, y la regla general cabe en una frase: **se reescala por $1/N$ únicamente lo que suma a lo largo del eje temporal**. Todo lo demás se copia, se extiende o se reconstruye.

| Componente | ¿Qué se hace? | ¿Reescalar por $1/N$? | Por qué |
|---|---|---|---|
| **Convolución** | Replicar el kernel a lo largo del eje temporal | **Sí** (o usar delta central) | Es una suma sobre $\tau$: la derivación de §3 aplica directamente |
| **Average pooling** (operador de media) | Extender la ventana al eje temporal | **No** | El operador ya normaliza: la media de $N$ copias idénticas es la copia |
| **Average pooling** (implementado como conv de pesos constantes) | Replicar el kernel constante | **Sí** | Vuelve a ser el caso de la convolución |
| **Max-pooling** | Extender la ventana al eje temporal | **No, nunca** | El máximo de $N$ copias idénticas ya es la copia |
| **BatchNorm** ($\gamma$, $\beta$) | Copiar tal cual | **No** | Son parámetros por canal, ajenos al eje temporal |
| **BatchNorm** (media y varianza corridas) | Copiar tal cual, y **re-estimar** al entrenar | **No** | Sobre video aburrido son exactas; sobre video real dejan de serlo |
| **Sesgo (bias)** | Copiar tal cual | **No** | Aparece una vez por posición de salida, independiente de $N$ |
| **Capa densa / clasificador** | No se infla: se reemplaza | — | El aplanado cambia de forma; ver más abajo |

{{< concept-alert type="advertencia" >}}
**Max-pooling no se divide por $N$.** Es el error más frecuente al inflar a mano, porque la frase "inflar y dividir por $N$" se aplica mecánicamente a todo. La división vale para operaciones que **suman**; un max-pooling toma el máximo, y
$$\max_{\tau=1..N} \tilde{x}[t+\tau] = x$$
Dividirlo por $N$ **rompe** el punto fijo en lugar de preservarlo, y lo rompe multiplicativamente en cada capa donde aparece. Lo mismo vale para el sesgo: dividir $b$ por $N$ es el segundo error más común.
{{< /concept-alert >}}

**El caso de BatchNorm merece detalle.** Copiar $\gamma$, $\beta$, la media y la varianza acumuladas es correcto en el instante inicial: sobre video aburrido las pre-activaciones son idénticas a las 2D, así que las estadísticas heredadas son exactamente las válidas. El punto práctico es que **sobre video real dejan de serlo**, así que hay que dejar que se re-estimen durante el fine-tuning. Acá hay una tensión con la práctica habitual de transfer learning, que suele **congelar** BatchNorm en modo evaluación porque los batches son chicos —y en video son especialmente chicos, porque un clip de 64 cuadros pesa 64 veces una imagen. Congelarlo significa conservar estadísticas de ImageNet que ya no describen la entrada: es una decisión de diseño, no un default.

**Las capas densas no se inflan.** Un clasificador 2D aplana un mapa de $C \times H \times W$; el inflado lo convierte en $C \times T' \times H \times W$, y el vector aplanado cambia de largo por un factor $T'$. Hay tres salidas: replicar los pesos densos a lo largo del tiempo y dividir por $T'$ (equivale a promediar temporalmente, y agrega $T'$ veces los parámetros); insertar un **global average pooling** que colapse espacio *y* tiempo antes de la capa densa; o descartar la cabeza y reentrenarla, que hay que hacer igual cuando cambia el número de clases. I3D toma la segunda vía sin esfuerzo, y ahí está una razón subestimada de la elección de Inception-v1 como backbone: [GoogLeNet](/papers/googlenet-szegedy-2014) ya había reemplazado las capas densas gigantes por global average pooling. La contabilidad lo hace evidente —de los ~78M de parámetros de C3D, unos 50M viven en `fc6` y `fc7`. La ventaja de I3D en parámetros (25M contra 79M) **no viene del inflado**, que de hecho multiplica los pesos convolucionales, sino de esa topología heredada.

Un esqueleto de la operación sobre un checkpoint real, en PyTorch:

```python
import torch

def inflar_state_dict(sd_2d, t=3, modo="uniforme"):
    """Infla un state_dict 2D a 3D. `t` = extension temporal del kernel."""
    sd_3d = {}
    for nombre, w in sd_2d.items():
        if w.dim() == 4:                      # peso de Conv2d: (Co, Ci, k, k)
            w3 = w.unsqueeze(2)               # -> (Co, Ci, 1, k, k)
            if modo == "uniforme":
                w3 = w3.repeat(1, 1, t, 1, 1) / t          # suma = W
            elif modo == "delta":
                w3 = torch.zeros(*w.shape[:2], t, *w.shape[2:])
                w3[:, :, t // 2] = w                        # identidad temporal
            sd_3d[nombre] = w3
        else:
            # bias, gamma, beta, running_mean, running_var, num_batches_tracked:
            # se copian SIN reescalar
            sd_3d[nombre] = w.clone()
    return sd_3d
```

El pooling **no aparece** en el `state_dict`: no tiene parámetros, y su inflado es un cambio de arquitectura (extender ventana y stride al eje temporal), no de pesos. Ese desfase entre "lo que se copia" y "lo que se modifica" es donde se cuelan los errores. La versión en PyTorch, TensorFlow y JAX está en la [práctica de la Clase 38](/clases/clase-38).

---

## 5. Por qué la equivalencia se propaga por toda la red

Que una capa cumpla el punto fijo no alcanza: hace falta que la propiedad **sobreviva la composición** de decenas de capas. El argumento es inductivo, y es la razón por la que la frase original de I3D dice "gracias a la linealidad".

Digamos que un tensor de activaciones es **temporalmente constante** si no depende de $t$. La entrada $B_T(x)$ lo es por construcción. Entonces:

1. **Convolución 3D inflada.** Si la entrada es temporalmente constante, la salida es temporalmente constante y coincide con la 2D. Es exactamente la derivación de §3, y lo que la habilita es la **linealidad** de la convolución: permite factorizar la suma sobre $\tau$ y sacar el peso efectivo afuera.
2. **No linealidad puntual** ($\mathrm{ReLU}$, sigmoide, GELU). Se aplica elemento a elemento, así que preserva la constancia y **conmuta** con la repetición: $\sigma(\tilde{y})[t] = \sigma(y)$. Lo que la habilita es la **puntualidad**.
3. **Pooling inflado.** El promedio (o el máximo) de $N$ copias idénticas es la copia. Preserva.
4. **Stride temporal $s>1$.** Submuestrear una señal constante da una señal constante. Preserva.
5. **Global average pooling final.** Colapsa el eje temporal promediando valores idénticos, y devuelve el valor 2D.

Por inducción sobre la profundidad, la red inflada completa evaluada sobre un video aburrido reproduce **exactamente** el logit de la red 2D sobre la imagen. Los ingredientes cruciales son linealidad y puntualidad; cualquier capa que viole alguna de las dos —una atención sobre el eje temporal, una normalización que promedie sobre $t$— hay que analizarla aparte.

{{< concept-alert type="recordar" >}}
El argumento inductivo regala un **test unitario** que detecta casi todos los errores de inflado antes de entrenar:

```python
x  = torch.randn(2, 3, 224, 224)
y2 = red_2d.eval()(x)
y3 = red_3d.eval()(x.unsqueeze(2).repeat(1, 1, 16, 1, 1))
assert (y3 - y2).abs().max() < 1e-4
```

Si la diferencia es un múltiplo entero de $N$, hay un reescalado de más o de menos. Si crece con la profundidad, el error está en una capa temprana. Si aparece solo con clips cortos, es padding (ver abajo).
{{< /concept-alert >}}

**La excepción del padding.** La derivación supone que la entrada es constante **en toda la ventana del kernel**. En los **bordes temporales** del clip, el padding con ceros inyecta valores que no son la imagen, así que el punto fijo se cumple exacto en el interior y solo de forma aproximada en el primer y último $\lfloor N/2 \rfloor$ cuadros. Con clips de 64 cuadros y kernels temporales de 3 o 7 el efecto es marginal; con clips de 8 cuadros y kernel temporal de 7 la mitad del clip está contaminada. Es la razón por la que el test de arriba puede pasar con `repeat(1,1,16,...)` y fallar con `repeat(1,1,4,...)`.

---

## 6. Inflar kernels separables

Una pregunta natural: si la arquitectura no usa kernels cúbicos sino **convoluciones factorizadas** —un kernel espacial $1{\times}k{\times}k$ seguido de un kernel temporal $t{\times}1{\times}1$, con una no linealidad entre ambos, como en [R(2+1)D](/papers/r2plus1d-tran-2018) y [S3D](/papers/s3d-xie-2018)—, ¿sobrevive el truco del inflado? Sí, y de forma **más limpia**:

- El kernel **espacial** $1 \times k \times k$ tiene extensión temporal $N = 1$. La condición de §3 se satisface con $\widetilde{W} = W$ **sin división alguna**: se copian los pesos de ImageNet tal cual, sin reescalar nada.
- El kernel **temporal** $t \times 1 \times 1$ no tiene análogo 2D del cual heredar. Se inicializa con la **delta central**: peso $1$ en $\tau = \lceil t/2 \rceil$ y $0$ en el resto, es decir la identidad temporal.

Con esa combinación la red separable inflada **arranca siendo exactamente la red 2D aplicada cuadro a cuadro**, y el entrenamiento solo tiene que aprender los taps temporales. Es preferible al reparto uniforme por tres razones concretas:

1. **No introduce un sesgo arbitrario.** El reparto uniforme impone un filtro pasa-bajos temporal que nadie pidió. La delta central no impone nada: deja el eje temporal en estado neutro.
2. **El punto fijo se cumple exacto, incluso en los bordes.** El kernel espacial no tiene extensión temporal, así que el padding temporal no lo afecta; y el kernel temporal, inicializado como delta, **no lee los vecinos**, así que tampoco lee el padding. La excepción de §5 desaparece: la equivalencia es exacta para clips de cualquier largo.
3. **El gradiente queda interpretable.** Los taps temporales arrancan en cero, así que su magnitud tras entrenar mide cuánta estructura temporal exigió la tarea. S3D usa esa lectura: tras entrenar en Kinetics, los pesos de los taps desplazados ($\tau \neq 0$) siguen **concentrados en cero en las capas bajas**, mientras que su varianza **crece con la profundidad**. La red dice por sí sola que el filtrado temporal de bajo nivel no le sirve —y ahí está la raíz cuantitativa del diseño *top-heavy*: convoluciones 3D solo en las capas profundas.

Un detalle de arquitectura muestra que la separabilidad además reparte mejor la capacidad temporal: en un bloque Inception hay cuatro ramas, pero solo dos tienen convoluciones $3{\times}3$, así que cuando I3D infla **solo algunas de las features reciben información temporal**. Con convoluciones temporales separables se puede agregar una a las cuatro ramas, y S3D reporta que eso lleva el desempeño de **78,4% a 78,9%** en Mini-Kinetics-200. Es modesto, pero descarta la idea de que "inflar todo" distribuya la capacidad temporal de forma uniforme.

La derivación completa de la factorización $(2+1)$D, incluido el hiperparámetro que iguala el conteo de parámetros con el bloque 3D, está en la [profundización de la Clase 38](/clases/clase-38/profundizacion).

---

## 7. Más allá del video: transferencia entre dimensionalidades

El inflado es el miembro más conocido de una familia más amplia: las técnicas que **inicializan una arquitectura nueva de modo que compute la misma función que una arquitectura vieja ya entrenada**. Vale ser preciso sobre el grado de parentesco entre los miembros, porque se confunden con facilidad.

**Imágenes médicas volumétricas.** Es la aplicación más importante fuera del video, y es real y muy usada. Un estudio de CT o de MRI es un **volumen**: una pila de cortes que forma un tensor $C \times D \times H \times W$, con la misma forma que un clip, solo que el eje inflado es la **profundidad anatómica**. La motivación es la del video pero más aguda: los datasets 3D médicos anotados son del orden de cientos de volúmenes, no existe un ImageNet de tomografías, y una red 3D desde cero sobre esa escala hace overfitting con comodidad. Inflar un backbone 2D de ImageNet y hacer fine-tuning es la receta estándar, y las dos inicializaciones de §3 tienen lecturas clínicas naturales: la delta central corresponde a "leer un corte a la vez", el reparto uniforme a promediar una **losa** de cortes, que es lo que hace una reconstrucción de cortes gruesos.

Las diferencias que conviene tener presentes al mover la técnica de un dominio al otro:

| Aspecto | Video | Volumen médico |
|---|---|---|
| Naturaleza del tercer eje | Tiempo | Profundidad espacial |
| ¿Tiene dirección privilegiada? | Sí (la flecha del tiempo: *sentarse* vs *pararse*) | No hay flecha, pero sí anatomía asimétrica (craneal/caudal) |
| ¿Es conmensurable con los ejes espaciales? | No: cuadros vs píxeles, escalas físicas distintas | Sí, los tres ejes son milímetros, aunque el espaciado entre cortes suele exceder al intra-corte |
| ¿Kernel cúbico defendible? | Poco: el eje temporal tiene estadística distinta | Más defendible, pero la anisotropía del espaciado lo relativiza |

O sea: el argumento de asimetría espacio-temporal que hace sospechoso un kernel $k\times k\times k$ en video se debilita en volúmenes, pero no desaparece —un CT con cortes gruesos es fuertemente anisótropo, y el kernel atraviesa muchos más milímetros por tap en profundidad que en el plano.

**Net2Net y las transformaciones que preservan la función.** *Net2Net* (Chen, Goodfellow y Shlens, 2016) hace crecer una red ya entrenada con dos operaciones que dejan su función intacta: **Net2WiderNet**, que replica unidades y **reparte los pesos salientes entre las copias**, y **Net2DeeperNet**, que inserta una capa inicializada como **identidad**. El parentesco con las dos inicializaciones de §3 no es casual: el reparto entre copias es el análogo del $W/N$ sobre el eje de canales, y la capa identidad el de la delta central. La diferencia hay que marcarla: Net2Net crece **dentro de la misma dimensionalidad**, mientras que el inflado cambia el **rango del tensor de pesos**. Son ideas emparentadas por el **principio** —inicializar preservando la función—, no la misma técnica. La familia reaparece en la expansión de modelos de lenguaje, donde se entrena un modelo chico y se lo expande en anchura o profundidad para que el grande arranque calculando lo que calculaba el chico: otra vez mismo principio, distinto mecanismo.

Dos aclaraciones para cerrar el perímetro. Alimentar una CNN 2D de ImageNet con un **espectrograma** de audio *no* es inflar: es reinterpretar una señal como imagen y usar el checkpoint sin tocarlo (ver [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel)). Y la operación no es simétrica: inflar 2D→3D tiene una solución canónica, pero "desinflar" 3D→2D no, porque habría que descartar información en lugar de repartirla.

---

## 8. Límites y cuándo no usarlo

**Cuando hay datos de video abundantes, la ganancia se evapora.** [SlowFast](/papers/slowfast-feichtenhofer-2019) entrena **desde cero, sin ImageNet**, y alcanza el estado del arte: 79,8% top-1 en Kinetics-400. Los autores probaron explícitamente el preentrenamiento de ImageNet en sus redes y encontraron que las variantes preentrenada y desde cero rinden **igual dentro de ±0,3%**. Más filoso todavía es su Tabla 6: una 3D ResNet-50 que con preentrenamiento de ImageNet daba 73,4% llega a **73,5% desde cero** con una receta de entrenamiento mejor —mientras que con la receta original desde cero daba 69,4%. Es decir: **buena parte de lo que parecía un beneficio del preentrenamiento era en realidad un déficit de receta de entrenamiento**. Y [R(2+1)D](/papers/r2plus1d-tran-2018) supera a I3D por **4,5%** cuando ambos se entrenan desde cero en Kinetics, lo que sugiere que con datos suficientes la arquitectura pesa más que la inicialización.

**Cuando el dominio está lejos de ImageNet, la garantía se vuelve hueca.** El punto fijo garantiza arrancar donde arranca la red 2D. Si la red 2D es mala sobre los cuadros del problema —endoscopía, ultrasonido, imagen térmica, CT—, garantiza arrancar en un lugar malo: la garantía es **relativa**, no absoluta. La evidencia sobre distancia entre dominios de [transfer learning](/fundamentos/transfer-learning) aplica sin cambios: transferir desde lejos sigue siendo mejor que pesos aleatorios, pero la penalización existe y crece con la distancia.

**El costo no desaparece.** Inflar **multiplica** los pesos convolucionales por $t$ y los FLOPs por $t \cdot T$ —el kernel es más grande *y* hay que evaluarlo en más posiciones. Con $t=3$ y $T=64$ es un factor 192 sobre la misma capa 2D, y I3D se entrenó sobre 64 GPUs. El inflado resuelve el problema de *datos*, no el de *cómputo*.

**Es solo una inicialización.** El punto fijo vale en el paso 0; el primer update de gradiente lo rompe, y lo que sobrevive no es la igualdad sino la **cuenca** en la que se aterrizó. La evidencia de S3D sobre las distribuciones de pesos temporales lo muestra: en las capas bajas los taps desplazados se quedan cerca de cero después de entrenar, o sea que la red **vuelve efectivamente a ser 2D** justo donde el inflado le dio capacidad temporal. Inflar no garantiza que esa capacidad se use.

{{< concept-alert type="advertencia" >}}
**Dos confusiones que cuestan días.** Primero, el $N$ de la división es la **extensión temporal del kernel**, no el largo del clip: dividir por $T=64$ en lugar de por $t=3$ produce activaciones ~21 veces más chicas y un modelo que parece "no aprender". Segundo, si el checkpoint inflado viene de terceros, hay que verificar con qué convención se produjo (uniforme o delta central) antes de cargarlo en un grafo propio; el desajuste no lanza excepción, solo baja la métrica.
{{< /concept-alert >}}

---

## 9. Estado actual: qué queda de la técnica

El inflado dejó de ser un titular y se volvió infraestructura, que es el destino de las buenas ideas de ingeniería.

**Sigue siendo la manera estándar de arrancar un modelo 3D con datos escasos**, y eso describe la situación de casi cualquier proyecto aplicado. Las librerías de video y de imagen médica distribuyen checkpoints inflados o funciones para producirlos, y en el régimen de cientos o miles de clips (o volúmenes) anotados, entrenar desde cero es la diferencia entre un modelo usable y uno que hace overfitting.

**Para video, el inflado se corrió un lugar en la cadena.** El aporte metodológico durable de I3D no fue el inflado en sí sino la receta **"preentrenar en Kinetics y transferir"**, que rige el [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) posterior. Hoy el inflado desde ImageNet es sobre todo **cómo se construye** el modelo que después se preentrena en Kinetics; quien hace fine-tuning sobre su dataset chico parte de un checkpoint de Kinetics y puede no inflar nunca nada a mano.

**Los video transformers heredaron la idea, no el mecanismo.** Un [Vision Transformer](/fundamentos/vision-transformer) no tiene convoluciones que inflar, pero sí una proyección de parches, y ahí reaparece el problema exacto: la proyección 2D de ViT toma un parche $h \times w$ y el modelo de video necesita una que tome un **tubelet** $t \times h \times w$. ViViT (Arnab et al., 2021) lo resuelve con dos estrategias que son literalmente las de §3: replicar la proyección 2D a lo largo del tiempo y dividir por $t$ (*filter inflation*), o ponerla en la posición temporal central con ceros en el resto (*central frame initialisation*), variante que el paper reporta como la mejor. Los bloques de atención se heredan sin tocar —la atención es indiferente a la cantidad de tokens— y los embeddings posicionales temporales se agregan nuevos. Es el descendiente directo del punto fijo del video aburrido, aplicado a la única capa del transformer que tiene forma espacial.

{{< concept-alert type="clave" >}}
Lo reutilizable del inflado no es la fórmula $W/N$: es el método. Ante un cambio de arquitectura, en lugar de preguntarse "¿con qué inicializo los pesos nuevos?", conviene preguntarse **"¿qué función quiero que la arquitectura nueva calcule en el instante inicial, y qué ecuación imponen los pesos viejos?"**. La respuesta de I3D —un video aburrido debe verse como su imagen— es tan simple que se vuelve verificable con un `assert`, y ese es exactamente el estándar al que apuntar.
{{< /concept-alert >}}

---

## Para Profundizar

- [Clase 38](/clases/clase-38) — la clase donde el inflado es el tema central; su [profundización](/clases/clase-38/profundizacion) deriva la condición de punto fijo, la contabilidad de parámetros y la factorización $(2+1)$D.
- [Desplazamiento Temporal](/fundamentos/desplazamiento-temporal) — la estrategia opuesta frente al mismo problema. El inflado **acepta** el costo de la convolución 3D y resuelve el de los datos heredando pesos 2D; [TSM](/papers/tsm-lin-2019) **elimina** el costo negándose a introducir una operación temporal, y se queda con la CNN 2D que el hardware ya optimiza. Desarrollada en la [Clase 40](/clases/clase-40).
- [Clase 36](/clases/clase-36) — introducción al análisis de video: convolución 3D, flujo óptico y las cinco familias de arquitecturas.
- [Clase 12 - Data Augmentation y Transfer Learning](/clases/clase-12) — el marco general del que el inflado es un caso particular.
- [Lab 38 - Action Recognition con I3D](/laboratorios/lab-38) — el inflado, medido. El lab encuentra que normalizar la entrada a $[0,1]$ en vez de $[-1,1]$ invierte una predicción de I3D, y la causa es que **el inflado también hereda las BatchNorm de Inception-v1** con sus estadísticas calibradas sobre entradas centradas en cero. La herencia 2D no es sólo de kernels.
- Papers: [I3D](/papers/i3d-carreira-2017) (la técnica) · [C3D](/papers/c3d-tran-2015) (el punto de partida 3D desde cero) · [Ji et al.](/papers/3d-cnn-ji-2013) (el origen de la convolución 3D) · [Kinetics](/papers/kinetics-kay-2017) (el dataset que cambió el cálculo) · [S3D](/papers/s3d-xie-2018) y [R(2+1)D](/papers/r2plus1d-tran-2018) (inflado de kernels separables) · [SlowFast](/papers/slowfast-feichtenhofer-2019) (el límite: entrenar desde cero) · [GoogLeNet](/papers/googlenet-szegedy-2014) (de dónde sale realmente la ventaja en parámetros de I3D).
- Fundamentos relacionados: [Transfer Learning y Fine-Tuning](/fundamentos/transfer-learning) · [Redes convolucionales](/fundamentos/redes-convolucionales) · [Análisis de Video](/fundamentos/analisis-de-video) · [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) · [Vision Transformer](/fundamentos/vision-transformer).
