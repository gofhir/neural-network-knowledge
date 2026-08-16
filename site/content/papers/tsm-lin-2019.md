---
title: "TSM: Temporal Shift Module (2019)"
weight: 425
math: true
---

{{< paper-card
    title="TSM: Temporal Shift Module for Efficient Video Understanding"
    authors="Ji Lin, Chuang Gan, Song Han (MIT, MIT-IBM Watson AI Lab)"
    year="2019"
    venue="ICCV 2019 / arXiv:1811.08383"
    pdf="/papers/tsm-lin-2019.pdf" >}}
El paper resuelve la tensión que define el reconocimiento de acciones eficiente: **las CNN 2D son baratas pero ciegas al tiempo; las CNN 3D modelan el tiempo pero son caras**. La propuesta es un módulo que se inserta dentro de una CNN 2D cualquiera y le da capacidad espacio-temporal con **cero parámetros y cero FLOPs adicionales**: desplazar una fracción de los canales a lo largo del eje temporal, de modo que cada frame reciba información de sus vecinos, y dejar que la convolución 2D siguiente haga la mezcla. El argumento formal es que **una convolución se descompone en desplazamiento más multiplicación-acumulación**: TSM ejecuta el desplazamiento en el tiempo —que no cuesta FLOPs, solo movimiento de datos— y pliega la multiplicación dentro de una convolución que la red ya iba a ejecutar. Sobre [Something-Something V1](/papers/something-something-goyal-2017), donde el orden temporal es indispensable, lleva a su baseline [TSN](/papers/tsn-wang-2016) de **20.5 % a 47.3 %**; sobre [Kinetics](/papers/kinetics-kay-2017), donde la apariencia estática ya resuelve la mayoría de los casos, de **70.6 % a 74.1 %**. Esa asimetría —28 puntos contra 3.5— es la lectura más informativa del trabajo. En eficiencia corre a **17.4 ms** contra los **165.3 ms** del [I3D](/papers/i3d-carreira-2017) comparable, con 1.8 puntos más de precisión. Es el modelo del [Laboratorio 40](/laboratorios/lab-40).
{{< /paper-card >}}

---

## Contexto: el costo de modelar el tiempo

Hacia 2018 había tres familias de respuestas al problema de meter el tiempo en una red convolucional, y las tres tenían un defecto conocido.

Las **CNN 2D puras** —[TSN](/papers/tsn-wang-2016) y sucesores— procesan frames de forma independiente y promedian las predicciones. Son baratas y heredan ImageNet sin fricción, pero su agregación **es invariante al orden**: reproducir el video al revés produce la misma salida. Reconocen *qué* aparece; no *en qué orden* pasa.

Las **CNN 3D** —[C3D](/papers/c3d-tran-2015), [I3D](/papers/i3d-carreira-2017)— aprenden features espacio-temporales conjuntos, pero pagan en cómputo (306 GFLOPs por video en el I3D de referencia), en parámetros y en latencia, lo que las excluye del tiempo real y del despliegue en borde.

Los **enfoques intermedios** —fusión tardía, arquitecturas mixtas 2D+3D como ECO, descomposiciones tipo [R(2+1)D](/papers/r2plus1d-tran-2018)— negocian expresividad por cómputo. La objeción del paper es precisa: si la fusión temporal ocurre recién *después* de la extracción de features, la información de bajo nivel que se perdió en el camino no se recupera.

TSM se propone como una cuarta vía: **fusión temporal en todos los niveles, al costo de una CNN 2D**.

## Método: separar el desplazamiento de la multiplicación

Una convolución 1-D de kernel 3 con pesos $W = (w_1, w_2, w_3)$ se escribe

$$Y_i = w_1 X_{i-1} + w_2 X_i + w_3 X_{i+1}$$

y se descompone en dos pasos: **desplazar** la entrada en $-1, 0, +1$, y **multiplicar-acumular**. El primero no requiere ninguna multiplicación —es reindexar memoria—; el segundo es el caro. TSM hace el desplazamiento en el **eje temporal** y deja que la multiplicación-acumulación la absorba la convolución 2D siguiente, que ya opera sobre el eje de canales. En palabras de los autores, *"we shift in the time dimension by ±1 and fold the multiply-accumulate from time dimension to channel dimension"*.

La operación completa son tres asignaciones sobre un tensor $(N, T, C, H, W)$ con `fold = C // 8`:

```python
out[:, :-1, :fold]        = x[:, 1:, :fold]           # futuro  -> presente
out[:, 1:, fold:2*fold]   = x[:, :-1, fold:2*fold]    # pasado  -> presente
out[:, :,  2*fold:]       = x[:, :,  2*fold:]         # sin desplazar
```

### Por qué el desplazamiento ingenuo no funciona

Ese es el aporte real del paper, y la parte que suele omitirse al resumirlo. Trasladar directamente la estrategia de *shift* espacial de la clasificación de imágenes al eje temporal, desplazando todos los canales, produce **dos desastres simultáneos**:

- **Peor eficiencia.** El desplazamiento no cuesta FLOPs pero sí movimiento de datos, y el efecto se agrava en video porque las activaciones son tensores 5D grandes. Medido: **+13.7 % de latencia en CPU** y +12.4 % en GPU.
- **Peor precisión.** Un canal desplazado hacia el frame vecino **deja de estar disponible para el frame actual**; desplazar muchos degrada el modelado espacial del backbone. Medido: **−2.6 puntos** contra la baseline TSN.

Las dos correcciones que rescatan el método:

**Partial shift.** Se desplaza solo una fracción. Con 1/8 de los canales el overhead de latencia cae a ≈3 %. Del lado de la precisión, el barrido dibuja una curva con máximo interior —poco desplazamiento no alcanza para razonar temporalmente, demasiado daña el aprendizaje espacial— y el óptimo está en **1/4 del total, 1/8 por dirección**.

{{< concept-alert type="cuidado" >}}
**El `shift8` del checkpoint no significa "se desplaza 1/8 del tensor".** El código hace `fold = C // 8` y mueve **dos** folds, uno al pasado y otro al futuro: el total desplazado es **1/4**. La cifra de 1/8 corresponde al modo **online**, que es unidireccional. El paper es explícito: *"the performance reaches the peak when 1/4 (1/8 for each direction) of the channels are shifted"*.
{{< /concept-alert >}}

**Residual shift.** El módulo se inserta **dentro de la rama residual** del bloque bottleneck —envolviendo su `conv1`— y no antes de la bifurcación (*in-place shift*). Así la conexión identidad sigue transportando la activación completa sin desplazar y toda la información original permanece accesible. El residual shift supera al in-place en **todas** las proporciones probadas; incluso desplazando el 100 % de los canales sigue por encima de la baseline 2D, algo imposible para el in-place. En el código del repositorio es el argumento `shift_place='blockres'`.

En ResNet-50 se insertan **16 módulos**, uno por bloque bottleneck (3+4+6+3). Cada uno amplía el campo receptivo temporal en 2. El relleno es con ceros, de modo que el primer frame no recibe pasado y el último no recibe futuro. Ver [Desplazamiento Temporal](/fundamentos/desplazamiento-temporal) para el mecanismo desarrollado.

### Los dos modos

**Offline (bidireccional).** Requiere el clip completo; cada frame se mezcla con el anterior y el siguiente. Es la configuración del checkpoint del laboratorio y la de máxima precisión.

**Online (unidireccional).** Para flujos en vivo, donde los frames futuros no existen. Solo se desplaza pasado → presente, con 1/8 de los canales, mediante un **caché**: se guardan los primeros 1/8 de los features de cada bloque y se reemplazan en el frame siguiente. Para ResNet-50 el caché ocupa **0.9 MB**, y el overhead sobre TSN puro es de **menos de 0.1 ms por frame**.

## Resultados

### La mejora depende del dataset, y esa es la noticia

| Dataset | TSN | TSM | Δ |
|---|---|---|---|
| Kinetics | 70.6 % | 74.1 % | **+3.5** |
| UCF-101 | 91.7 % | 95.9 % | +4.2 |
| HMDB-51 | 64.7 % | 73.5 % | +8.8 |
| Something-Something V1 | 20.5 % | 47.3 % | **+28.0** |
| Something-Something V2 | 30.4 % | 61.7 % | **+31.3** |
| Jester | 83.9 % | 97.0 % | +11.7 |

El paper divide la tabla en dos mitades: arriba los datasets donde las relaciones temporales importan poco, abajo los que dependen de ellas. La misma modificación arquitectónica produce +3.5 o +31.3 puntos según qué se esté midiendo, lo que convierte a esta tabla en un instrumento de medición sobre **los benchmarks** tanto como sobre el modelo.

### Contra el estado del arte y contra el reloj

En Something-Something V1, TSM con ResNet-50 y 8 frames alcanza **45.6 %** con 33 GFLOPs y 24.3 M parámetros: **+4.2 puntos sobre ECO con 1.9× menos cómputo**, y **+1.2 sobre Non-local I3D con 10× menos FLOPs**. Con 16 frames llega a 47.2 % y con flujo óptico a 52.6 %.

Medido en una Tesla P100:

| Modelo | FLOPs | Params | Latencia | Throughput | Kinetics |
|---|---|---|---|---|---|
| I3D | 306 G | 35.3 M | 165.3 ms | 6.1 v/s | — |
| ECO 16F | 64 G | 47.5 M | 30.6 ms | 45.6 v/s | — |
| **TSM 8F** | **33 G** | **24.3 M** | **17.4 ms** | **77.4 v/s** | **74.1 %** |

**9.5× menos latencia y 12.7× más throughput** que el I3D comparable, con 1.8 puntos más de precisión. El control más elegante de esta sección es `I3D_replace`: reemplazar cada TSM por una convolución $3\times1\times1$ —misma función de mezcla, ahora con parámetros y FLOPs— resulta **más lento y menos preciso**. La ventaja no es solo de complejidad asintótica sino de qué kernels están bien optimizados en el hardware que existe.

### Offline contra online: el resultado que sorprende

| Modelo | Latencia | Kinetics | UCF-101 | HMDB-51 | Something-V1 |
|---|---|---|---|---|---|
| TSN (baseline 2D) | 4.7 ms | 70.6 % | 91.7 % | 64.7 % | 20.5 % |
| + TSM offline | — | 74.1 % | **95.9 %** | 73.5 % | **47.3 %** |
| + TSM online | 4.8 ms | **74.3 %** | 95.5 % | **73.6 %** | 46.3 % |

En **Kinetics y HMDB-51 el modo online iguala o supera al offline**. Solo en Something-Something —donde el orden es constitutivo— gana el offline, y por 1.0 punto. El acceso a los frames futuros, que suele presentarse como la razón de ser del modo offline, **aporta poco o nada en los datasets dominados por apariencia**. El [Laboratorio 40](/laboratorios/lab-40/04-la-curva-de-proporcion) reproduce este resultado por otra vía, comparando ambos modos a igual proporción de canales desplazados.

### Más allá de la clasificación

**Reconocimiento temprano** (UCF-101): observando solo el **10 % de los frames**, TSM alcanza 90 % de precisión, 6.6 puntos sobre el mejor ECO.

**Detección en video** (ImageNet-VID): el TSM unidireccional inyectado en un R-FCN sube el mAP de 74.7 a **76.3**, y la mejora se concentra en los objetos **rápidos** (+4.6 mAP), donde el desenfoque de movimiento degrada la apariencia de un frame aislado. Supera a FGFA —que agrega 21 frames con flujo óptico, 10 pasados y 10 futuros— siendo causal y 2.5× más rápido.

**Dispositivos de borde** con MobileNet-V2 online compilado con TVM: **13.4 ms en Jetson Nano** (4.5 W), 8.5 ms en Jetson TX2, 34.5 ms en Galaxy Note8, 69.6 ms en Raspberry Pi 4B.

## Limitaciones

- **El movimiento de datos no es gratis.** "Cero costo computacional" describe el conteo de FLOPs, no el tiempo de pared: con la proporción elegida el overhead real es de ~3 %.
- **El desplazamiento es rígido.** Siempre ±1 frame, siempre la misma fracción, siempre en las mismas posiciones. No hay nada aprendido en el módulo: ni cuánto desplazar, ni qué canales, ni con qué alcance.
- **Depende del $T$ de entrenamiento.** El módulo reinterpreta la dimensión de batch según el número de segmentos; cambiarlo en inferencia altera la semántica del desplazamiento.
- **Los bordes temporales quedan con ceros.** El primer y el último frame reciben relleno nulo en una fracción de sus canales, en cada uno de los 16 módulos. El paper no discute el efecto.
- **La ganancia sobre Kinetics es modesta**: +3.5 puntos. Buena parte del atractivo en ese dataset viene de la eficiencia, no de la precisión.
- **Los mejores resultados siguen necesitando flujo óptico** (+5.4 puntos en Something-V1), con el costo de precomputar TV-L1.

## Por qué importa para la Clase 40

La [Clase 40](/clases/clase-40) presenta TSN y TSM como una progresión, y su [laboratorio](/laboratorios/lab-40) hace inferencia con el checkpoint oficial sobre Kinetics-400. El paper corrige dos puntos del material y agrega uno que falta:

1. **La proporción.** La slide dice "1/8 del mapa de características"; eso describe el modo unidireccional. En el bidireccional del checkpoint se desplaza **1/4**.
2. **La nomenclatura.** La lámina se titula "Modelos offline con desplazamiento unidireccional", cuando la sección correspondiente del paper es *"Online Models with Uni-directional TSM"*. Unidireccional equivale a **online**.
3. **Lo que falta.** Ni el *partial shift* ni el *residual shift* aparecen en el material, y son las dos correcciones sin las cuales el método no funciona: sin la primera se pierde la eficiencia, sin la segunda la precisión.

En la línea del dominio [Video](/dominios/video), TSM es la respuesta opuesta a la de [I3D](/papers/i3d-carreira-2017) frente al mismo problema. I3D acepta el costo de la convolución 3D y resuelve el de los datos **inflando** pesos 2D preentrenados; TSM **elimina** el costo negándose a introducir una operación temporal, y se queda con la CNN 2D que ya estaba optimizada. La [Clase 38](/clases/clase-38) desarrolla la primera vía; esta clase, la segunda. El [Laboratorio 38](/laboratorios/lab-38) y el [40](/laboratorios/lab-40) corren ambos modelos sobre videos de UCF-101, lo que permite compararlos sobre el mismo terreno.

Para video clínico la lectura útil es la de eficiencia. Un modelo que corre a 17 ms en GPU de servidor y a 13 ms en una Jetson Nano cambia lo que es desplegable: análisis de video endoscópico durante el procedimiento, monitoreo de gestos quirúrgicos, seguimiento de movilidad en sala. Y la Tabla 6 indica que la restricción de causalidad —obligatoria en cualquier escenario en vivo— no cuesta precisión salvo en tareas donde el orden de los eventos es el objeto mismo de la clasificación.
