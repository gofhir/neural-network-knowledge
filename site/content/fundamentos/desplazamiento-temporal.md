---
title: "Desplazamiento Temporal"
weight: 127
math: true
---

**Desplazar temporalmente** un mapa de características es mover una fracción de sus canales un paso a lo largo del eje del tiempo, de modo que cada frame quede conteniendo información de sus vecinos antes de que la red la procese. Es una forma de dotar de memoria temporal a una arquitectura que no la tiene, sin agregar una sola operación aritmética ni un solo peso. La técnica se formalizó en [TSM](/papers/tsm-lin-2019) (Lin, Gan y Han, ICCV 2019) y es la alternativa conceptual al [inflado de convoluciones](/fundamentos/inflado-de-convoluciones): donde I3D acepta el costo de la convolución 3D y resuelve el problema de los datos heredando pesos 2D, TSM **elimina el costo** negándose a introducir una operación temporal. Este fundamento presenta el mecanismo de forma autónoma: por qué funciona, qué se rompe si se aplica de forma ingenua, las dos correcciones que lo vuelven viable, y en qué condiciones aporta y en cuáles no.

---

## 1. El problema: el tiempo cuesta caro

Una CNN 2D aplicada a video procesa cada frame por separado. Cualquier información sobre cómo evoluciona la escena tiene que entrar por otra puerta, y las opciones tradicionales cuestan:

| Enfoque | Qué agrega | Qué cuesta |
|---|---|---|
| Convolución 3D | kernel $t \times k \times k$ | $t$ veces más parámetros y FLOPs por capa |
| [Flujo óptico](/fundamentos/flujo-optico) | segunda corriente de entrada | precómputo externo, a menudo más caro que la red |
| RNN sobre features | estado recurrente | secuencialidad, difícil de paralelizar |
| Promedio de predicciones | nada | **invariante al orden**: no modela tiempo |

La última fila es la baseline de [TSN](/papers/tsn-wang-2016), y su defecto es estructural: si la agregación es un promedio, reproducir el video al revés produce exactamente la misma salida. El modelo puede decir *qué* aparece en un video; no *en qué orden* pasa.

El desplazamiento temporal ataca ese hueco por un camino distinto: en vez de agregar una operación que mezcle el tiempo, **reorganiza los datos** para que la operación que ya existe haga la mezcla.

---

## 2. La idea: una convolución son dos pasos

El punto de partida es una descomposición elemental. Una convolución 1-D de kernel 3 con pesos $W = (w_1, w_2, w_3)$ sobre una señal $X$ es

$$Y_i = w_1 X_{i-1} + w_2 X_i + w_3 X_{i+1}$$

que puede leerse como dos operaciones encadenadas. Primero se **desplaza** la señal:

$$X^{-1}_i = X_{i-1}, \qquad X^{0}_i = X_i, \qquad X^{+1}_i = X_{i+1}$$

y después se **multiplica y acumula**:

$$Y = w_1 X^{-1} + w_2 X^{0} + w_3 X^{+1}$$

La observación clave es la asimetría de costos entre ambos pasos. **Desplazar no requiere ninguna multiplicación**: es reindexar memoria. Multiplicar-acumular es donde vive todo el cómputo.

La jugada consiste en hacer el desplazamiento **sobre el eje temporal** y dejar que la multiplicación-acumulación la absorba una convolución que la red ya iba a ejecutar de todos modos —la convolución 2D siguiente, que opera sobre el eje de canales. Como los canales ahora contienen features de $t-1$, $t$ y $t+1$, esa convolución puede aprender pesos que combinan los tres instantes: es una convolución temporal de kernel 3 ejecutada por una capa espacial preexistente.

{{< concept-alert type="clave" >}}
El modelado temporal **no aparece en el conteo de FLOPs ni en el de parámetros** porque no se agrega ninguna operación: se cambia qué datos recibe una operación que ya estaba. En términos del paper: *"we shift in the time dimension by ±1 and fold the multiply-accumulate from time dimension to channel dimension"*.
{{< /concept-alert >}}

---

## 3. La operación

Sea un tensor de activaciones de forma $(N, T, C, H, W)$ —lote, tiempo, canales, alto, ancho— y una fracción de desplazamiento definida por `fold = C // fold_div`. La operación completa son tres asignaciones:

```python
out = torch.zeros_like(x)
out[:, :-1, :fold]        = x[:, 1:, :fold]           # futuro  -> presente
out[:, 1:, fold:2*fold]   = x[:, :-1, fold:2*fold]    # pasado  -> presente
out[:, :,  2*fold:]       = x[:, :,  2*fold:]         # sin desplazar
```

Leído sobre el eje de canales, para el frame en el instante $t$ con `fold_div = 8`:

```
canales:   [0 ─────── C/8) [C/8 ────── C/4) [C/4 ───────────────────── C)
contenido:   features de     features de           features de t
                t+1              t-1                (sin tocar)
                12.5%            12.5%                  75%
```

### Bidireccional y unidireccional

La versión de arriba mezcla pasado **y** futuro, y se llama **bidireccional** u **offline**: requiere tener el clip completo antes de empezar. La variante **unidireccional** u **online** solo trae el pasado:

```python
out[:, 1:, :fold] = x[:, :-1, :fold]     # pasado -> presente
out[:, :,  fold:] = x[:, :,  fold:]      # el resto intacto
```

Es causal, y por lo tanto aplicable a un stream en vivo. Su implementación eficiente no recorre un tensor completo sino que mantiene un **caché**: se guardan los primeros `fold` canales de los features de cada bloque y se inyectan en el frame siguiente. Para una ResNet-50 ese caché ocupa 0,9 MB, y el overhead por frame es de menos de 0,1 ms.

---

## 4. Por qué el desplazamiento ingenuo no funciona

Desplazar canales ya se usaba en clasificación de imágenes, sobre los ejes espaciales. Trasladar esa receta al eje temporal sin más —desplazando todos o casi todos los canales— produce **dos fallas simultáneas**, y entenderlas es entender el diseño.

**Falla de eficiencia: el movimiento de datos no es gratis.** El desplazamiento no consume FLOPs, pero sí ancho de banda de memoria. En video el efecto se amplifica porque las activaciones son tensores 5D de gran tamaño. Medido sobre ResNet-50 con 8 frames: desplazar todos los canales cuesta **+13,7 % de latencia en CPU** y +12,4 % en GPU. Para una operación que se anunciaba como gratuita, es una cifra que la descalifica.

**Falla de precisión: la información desplazada desaparece del presente.** Cuando un canal se mueve hacia el frame vecino, **deja de estar disponible para el frame actual**. Si se desplazan demasiados, el backbone pierde capacidad de representar la apariencia. Medido: **−2,6 puntos** respecto de la baseline 2D.

Las dos correcciones son independientes y ambas necesarias.

### 4.1. Partial shift: cuánto desplazar

En vez de mover todos los canales, se mueve una fracción pequeña. El costo de latencia cae de 13,7 % a **≈3 %** al pasar de "todos" a 1/8. Del lado de la precisión aparece una **curva con máximo interior**:

- Con demasiado poco, la capacidad de razonamiento temporal no alcanza.
- Con demasiado, se degrada el aprendizaje espacial.

El óptimo empírico está en **1/4 del tensor: 1/8 por cada dirección**.

{{< concept-alert type="cuidado" >}}
El parámetro se llama `fold_div = 8` y el checkpoint oficial dice `shift8`, lo que induce a leer "se desplaza 1/8". El código calcula `fold = C // 8` y mueve **dos** folds —uno al futuro, otro al pasado—, de modo que el total desplazado es **1/4**. La cifra de 1/8 es correcta solo para el modo unidireccional.
{{< /concept-alert >}}

### 4.2. Residual shift: dónde insertarlo

La segunda corrección es de posición. Insertar el módulo **antes** de una capa o de un bloque residual —lo que se llama *in-place shift*— arrastra el problema de la información perdida.

La alternativa es ubicarlo **dentro de la rama residual**, después de la bifurcación:

```
in-place:                         residual:

  x ──► shift ──► conv ──►┐         x ──┬─► shift ──► conv ──►┐
        (todo x           │              │                     ├──► +
         desplazado)      ├──► +         └─────────────────────┘
  x ─────────────────────►┘                  (identidad intacta)
```

En el esquema residual la conexión identidad sigue transportando la activación **sin desplazar**, de modo que toda la información original permanece accesible aguas abajo. El efecto es grande: el residual shift supera al in-place en todas las proporciones probadas, y aun desplazando el 100 % de los canales sigue por encima de la baseline 2D —algo imposible para el in-place, que en ese régimen colapsa.

En una ResNet bottleneck la inserción concreta es envolver la `conv1` de cada bloque:

```python
blocks[i].conv1 = TemporalShift(b.conv1, n_segment=T, n_div=8)
```

---

## 5. Propiedades del módulo apilado

**Campo receptivo temporal.** Cada módulo insertado equivale a una convolución temporal de kernel 3, de modo que **amplía el campo receptivo en 2** (uno hacia cada lado). Apilarlos lo hace crecer de forma acumulativa: en una ResNet-50 se insertan 16 módulos —uno por bloque bottleneck, 3+4+6+3— y la cobertura temporal resultante excede varias veces los 8 segmentos de entrada.

**Densidad de inserción.** En backbones más profundos el movimiento de datos se vuelve el cuello de botella. La regla práctica de la implementación de referencia: si la etapa `layer3` tiene 23 o más bloques —el caso de ResNet-101— se inserta el módulo en uno de cada dos.

**Relleno de los bordes.** El desplazamiento deja huecos en los extremos temporales, y se rellenan con **ceros** (`torch.zeros_like`). El primer frame no tiene pasado y el último no tiene futuro, así que en ambos una fracción de los canales queda anulada — en cada uno de los módulos apilados. Es un efecto sistemático que ninguna implementación estándar compensa, y que hay que tener presente al trabajar con clips cortos, donde los bordes son una porción grande del total.

**Dependencia de $T$.** El módulo necesita saber cuántos frames hay para identificar vecinos: internamente hace `x.view(n_batch, n_segment, c, h, w)` sobre un tensor que llegó aplanado en la dimensión de lote. Un modelo entrenado con $T=8$ asume esa agrupación; cambiar el número de segmentos en inferencia altera la semántica del desplazamiento.

---

## 6. Cuándo aporta y cuándo no

Esta es la parte que decide si la técnica sirve para un problema dado, y la respuesta no depende del modelo sino **de los datos**.

| Dataset | Baseline 2D | Con desplazamiento | Δ |
|---|---|---|---|
| [Kinetics](/papers/kinetics-kay-2017) | 70,6 % | 74,1 % | **+3,5** |
| [UCF-101](/papers/ucf101-soomro-2012) | 91,7 % | 95,9 % | +4,2 |
| [HMDB-51](/papers/hmdb-kuehne-2011) | 64,7 % | 73,5 % | +8,8 |
| [Something-Something V1](/papers/something-something-goyal-2017) | 20,5 % | 47,3 % | **+28,0** |
| Something-Something V2 | 30,4 % | 61,7 % | **+31,3** |

La misma modificación produce +3,5 o +31,3 puntos según el benchmark. La variable que explica la diferencia es **cuánta información temporal hace falta realmente para resolver la tarea**: reconocer "tocando guitarra" en Kinetics se puede hacer con un frame; distinguir "empujando algo de izquierda a derecha" de su inverso en Something-Something no.

Dos consecuencias prácticas:

**El promedio esconde la distribución.** Los +3,5 puntos de Kinetics no significan "+3,5 en cada video". El [Laboratorio 40](/laboratorios/lab-40/03-la-ablacion-del-shift) mide la contribución video por video anulando los módulos, y encuentra que en los casos que la apariencia ya resuelve el aporte es de **0,4 puntos**, mientras que en los ambiguos supera los **30**. El desplazamiento temporal se comporta como un **mecanismo de rescate** para los casos difíciles, no como una mejora uniforme.

**La dirección importa menos que el ancho de banda.** El modo unidireccional, que renuncia por completo al futuro, **iguala o supera** al bidireccional en Kinetics (74,3 % contra 74,1 %) y HMDB-51 (73,6 % contra 73,5 %), y solo pierde 1,0 punto en Something-Something. Para la mayoría de las tareas, lo que aporta el desplazamiento es la mezcla de instantes vecinos, no específicamente el acceso a lo que viene después.

---

## 7. Cómo verificarlo sobre un modelo entrenado

Una propiedad práctica del mecanismo es que su contribución se puede **medir sin reentrenar**, porque el módulo no tiene pesos. Tres manipulaciones sobre un checkpoint existente:

**Anularlo.** Con `fold_div` arbitrariamente grande, `fold = C // fold_div = 0`: las dos primeras asignaciones quedan vacías, la tercera copia el tensor completo y el módulo **se vuelve la identidad**. El modelo se reduce a su baseline 2D con los mismos pesos, y la caída de precisión acota la contribución del desplazamiento.

**Barrer la proporción.** Variando `fold_div` entre 2 y 32 se recorre la fracción desplazada de 1 a 1/16. Sobre pesos fijos la curva mide sensibilidad al desajuste con el entrenamiento, y su forma es informativa: **asimétrica**, con degradación suave al quedarse corto y colapso al pasarse. Quedarse corto pierde información temporal; pasarse **corrompe los canales espaciales**.

**Cambiar la dirección.** Sustituyendo la función de desplazamiento por su variante unidireccional se compara el modo online contra el offline sobre el mismo checkpoint.

{{< concept-alert type="nota" >}}
Estas manipulaciones sacan al modelo de la distribución en la que fue entrenado, así que sus resultados son cotas, no comparaciones limpias entre arquitecturas. El control que las valida es un caso donde la manipulación **no** degrade: si anular el módulo dejara igual a un video y arruinara a otro, la degradación no es un artefacto genérico de romper el modelo sino un efecto específico del contenido temporal.
{{< /concept-alert >}}

---

## 8. Parentela y descendencia

**Antes.** El desplazamiento como primitiva viene de la clasificación de imágenes, donde se aplicaba sobre los ejes espaciales para reemplazar convoluciones $3\times3$ por combinaciones de shift más $1\times1$. La contribución de TSM no es la operación sino el diagnóstico de por qué su traslado directo al tiempo fracasa.

**Al lado.** El [inflado de convoluciones](/fundamentos/inflado-de-convoluciones) resuelve el mismo problema con la estrategia opuesta: acepta la convolución 3D y se concentra en poder inicializarla. La factorización $(2+1)$D de [R(2+1)D](/papers/r2plus1d-tran-2018) y [S3D](/papers/s3d-xie-2018) es una vía intermedia: mantiene la operación temporal pero la separa de la espacial para abaratarla. En términos de costo temporal: 3D paga todo, $(2+1)$D paga una convolución 1D, el desplazamiento no paga nada.

**Después.** La limitación evidente del módulo es su rigidez —siempre ±1 frame, siempre la misma fracción, siempre los mismos canales—, y la línea posterior ataca justamente eso: módulos que **aprenden** cuánto desplazar cada canal, agregación temporal adaptativa, y desplazamientos de alcance variable. También se extendió más allá de la clasificación: inyectado en el backbone de un detector, el desplazamiento unidireccional mejora la detección de objetos en video y la ganancia se concentra en los **objetos rápidos**, donde el desenfoque de movimiento degrada la apariencia de un frame aislado.

---

## 9. Fuera del video

El eje que se desplaza no tiene por qué ser el tiempo. El requisito es que exista una dimensión ordenada, con vecindad significativa, que se haya aplanado dentro del lote para que una red 2D la procese. Eso incluye:

- **Volúmenes médicos** (TC, RM), donde el eje es la profundidad anatómica y los cortes vecinos comparten estructura — el mismo escenario en el que el [inflado](/fundamentos/inflado-de-convoluciones) se aplica habitualmente.
- **Espectrogramas** procesados por parches, donde el eje es el tiempo de la señal de [audio](/fundamentos/representacion-de-audio).
- **Secuencias de imágenes satelitales** del mismo terreno en distintas fechas.

En todos los casos vale el mismo cálculo: si la vecindad a lo largo de ese eje aporta información, desplazar una fracción de los canales la pone a disposición del modelo sin agregar un solo parámetro.

---

## Referencias

- [TSM: Temporal Shift Module (Lin, Gan y Han, 2019)](/papers/tsm-lin-2019) — el paper que formaliza el mecanismo, sus dos correcciones y los modos offline/online.
- [TSN (Wang et al., 2016)](/papers/tsn-wang-2016) — la baseline 2D sobre la que se inserta, y el origen del muestreo por segmentos.
- [I3D (Carreira y Zisserman, 2017)](/papers/i3d-carreira-2017) — la respuesta alternativa: inflar en vez de desplazar.
- [Clase 40](/clases/clase-40) y [Laboratorio 40](/laboratorios/lab-40) — la presentación en el curso y la verificación experimental sobre un checkpoint entrenado.
