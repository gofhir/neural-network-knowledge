---
title: "Teoría - Analítica de Videos: Reconocimiento de acciones"
weight: 10
math: true
---

> **Recorrido de la Clase 40** del Diplomado IA UC (Bianca Del Solar Medrano). Veintinueve diapositivas organizadas en cuatro bloques: introducción, la ruta actual del campo, **TSN → TSM**, y los principales conjuntos de datos. El hilo conductor no es una arquitectura sino un criterio: **el costo**. Cada enfoque que la clase presenta se evalúa por lo que cuesta —memoria, cómputo, precómputo externo— y las dos propuestas centrales se justifican por lo que consiguen **sin** pagar.

{{< concept-alert type="clave" >}}
**El hilo que hay que seguir.** La slide 4 plantea dos rutas: *aumentar el rendimiento* y *reducir el tamaño de los modelos*. La clase recorre casi enteramente la segunda. TSN elimina el costo del **muestreo**; TSM elimina el costo del **modelado temporal**. Leídas así, las dos propuestas son la misma jugada aplicada en dos lugares distintos del pipeline.
{{< /concept-alert >}}

---

## 1. El objetivo (slide 3)

La clase define el problema con cuatro preguntas:

> Comprender lo que sucede en un video: **la acción** que se lleva a cabo, **quién** la realizó, **dónde** se lleva a cabo y **en qué parte del video** sucede.

Cada una es una tarea distinta del campo, con datasets y métricas propias:

| Pregunta | Tarea | Salida |
|---|---|---|
| ¿Qué acción? | *Action recognition* / clasificación | una etiqueta por clip |
| ¿Quién? | *Action detection* espacial | caja o máscara por actor |
| ¿Dónde? | reconocimiento de escena | etiqueta de contexto |
| ¿En qué parte del video? | *Temporal action localization* | intervalo $[t_{\text{inicio}}, t_{\text{fin}}]$ |

El resto de la clase se ocupa **solo de la primera**, que es también la que asume el [laboratorio](/laboratorios/lab-40): un clip recortado, una acción, una etiqueta. Ver [Reconocimiento de Acciones](/fundamentos/reconocimiento-de-acciones) para el mapa completo de tareas.

---

## 2. La ruta actual (slide 4)

Dos ejes, que la clase enumera sin jerarquizar:

**1. Aumentar el rendimiento de los modelos**
- Mejorar la modelización temporal
- Información multimodal
- Causalidad

**2. Reducir el tamaño de los modelos**

La lista del primer eje es un buen mapa del campo alrededor de 2019-2020: el modelado temporal es lo que ataca TSM; la información multimodal desemboca en los [modelos visión-lenguaje](/fundamentos/vision-language-models); la causalidad —distinguir que una acción *produce* un efecto y no solo coocurre con él— sigue siendo el problema abierto de la lista, y es exactamente lo que Something-Something intenta medir.

Pero el peso de la clase cae sobre el segundo eje, y conviene entender por qué no es una preocupación menor de ingeniería. Un modelo de video que necesita 306 GFLOPs por clip y 165 ms de latencia no se puede desplegar en un teléfono, ni en una cámara, ni procesar en tiempo real el volumen de video que produce una plataforma. La eficiencia no es una optimización posterior: **decide qué aplicaciones existen**.

---

## 3. Los enfoques anteriores y su factura (slides 5-8)

La clase pregunta: *¿cómo modelan los enfoques anteriores estructuras temporales de largo alcance?* Y responde con cuatro estrategias, cada una con su costo:

| Estrategia | Ejemplo | Lo que cuesta, según la clase |
|---|---|---|
| Redes 3D | [C3D](/papers/c3d-tran-2015), [I3D](/papers/i3d-carreira-2017) | "aumentar el tamaño del modelo… necesita más memoria y no necesariamente se aumenta el rendimiento" |
| Flujo óptico | [Two-Stream](/papers/two-stream-simonyan-2014) | "es necesario realizar cálculos adicionales" |
| Apilar más fotogramas | I3D (64 frames) | "necesita más memoria" |
| Muestrear a tasa fija | LRCN y similares | "necesita más memoria" |

La frase más filosa es la primera: *"no necesariamente se aumenta el rendimiento"*. Es la observación que justifica todo lo que sigue — si escalar el modelo no garantiza mejores resultados, la vía de la eficiencia deja de ser un compromiso y pasa a ser una alternativa legítima.

### El problema del muestreo a tasa fija (slides 6-8)

Tres diapositivas casi idénticas ilustran el defecto del muestreo denso, y la repetición es el argumento. Muestran una tira de frames $f_1, f_2, \dots, f_N$ y debajo la etiqueta "Epoch 1", "Epoch 1", "Epoch 1"… siempre los mismos frames seleccionados.

Dos problemas ahí:

**Redundancia.** Los frames consecutivos de un video a 25 fps son casi idénticos. Procesar 64 de ellos es pagar 64 veces por una información que se repite.

**Cobertura.** Si el presupuesto es una ventana fija de 64 frames contiguos, en un video a 25 fps eso son 2,5 segundos. Un video más largo queda cubierto solo en parte, y la acción relevante puede caer fuera.

**Y un tercero, que las slides muestran sin nombrar:** con muestreo determinista, cada época ve **exactamente los mismos frames**. No hay variación temporal en el entrenamiento; el modelo memoriza esos frames concretos en lugar de aprender la acción.

---

## 4. TSN: muestreo por segmentos (slides 9-14)

> "Este es el nuevo modelo que propone un nuevo paradigma para la modelización de estructuras temporales, llamado **muestreo basado en segmentos**."

La propuesta de [TSN](/papers/tsn-wang-2016) es dividir el video en $K$ segmentos de igual duración $S_1, S_2, \dots, S_N$ y tomar **un frame de cada uno**. La contraposición con las slides anteriores es directa: donde antes había una tira uniforme de frames contiguos, ahora hay bloques que abarcan el video completo.

Y las slides 10-11 muestran la diferencia que importa: "Epoch 1" y "Epoch 2" seleccionan frames **distintos** dentro de los mismos segmentos. El muestreo es aleatorio dentro de cada segmento durante el entrenamiento, lo que funciona como **aumentación temporal gratuita**: cada época el modelo ve una combinación nueva del mismo video.

### La arquitectura (slides 12-14)

```
S1 ──► Conv2D ──┐
S2 ──► Conv2D ──┤
 ⋮        ⋮      ├──► Consensus (Avg) ──► predicción
SN ──► Conv2D ──┘
        ↑
   pesos compartidos
```

Tres propiedades que se leen del diagrama:

- **Una sola red, aplicada $N$ veces.** No hay $N$ modelos: los pesos son compartidos. El costo de parámetros es el de una CNN 2D.
- **El consenso opera sobre predicciones, no sobre features.** Cada segmento produce su propio vector de clases y recién ahí se promedian.
- **El costo es constante en la duración del video.** $K$ está fijo; lo que se estira o encoge es el paso entre frames.

{{< concept-alert type="cuidado" >}}
**El defecto que la clase no señala y que motiva TSM.** El consenso por promedio es **invariante al orden**: promediar las predicciones de los $N$ segmentos da idénticamente lo mismo si se los permuta. TSN puede reconocer *qué* aparece a lo largo del video; no puede inferir en qué orden ocurre. Es un *bag of segments*. La [profundización](profundizacion) lo demuestra formalmente, y el [Laboratorio 38](/laboratorios/lab-38/05-invertir-el-tiempo) mide el fenómeno análogo sobre I3D.
{{< /concept-alert >}}

---

## 5. TSM: el desplazamiento temporal (slides 15-20)

> "Propone un Módulo de Cambio Temporal (TSM) genérico y eficaz que combina tanto alta eficiencia como alto rendimiento. Específicamente, puede lograr el rendimiento de una CNN en 3D pero mantener la complejidad de una CNN en 2D."

Esa es la promesa. El mecanismo aparece en la slide siguiente, y es una sola frase:

> "Supongamos que tenemos un vector unidimensional, por lo tanto, la convolución de este vector podría aproximarse mediante dos pasos, que son **el desplazamiento y la multiplicación**."

Vale desarrollarla porque es todo el argumento. Una convolución 1-D de kernel 3 con pesos $W = (w_1, w_2, w_3)$:

$$Y_i = w_1 X_{i-1} + w_2 X_i + w_3 X_{i+1}$$

se descompone en **desplazar** la señal ($-1$, $0$, $+1$) y después **multiplicar y acumular**. El primer paso no tiene multiplicaciones: es reindexar memoria. El segundo es donde vive el cómputo.

TSM ejecuta el desplazamiento sobre el **eje temporal** y deja que la multiplicación-acumulación la absorba la convolución 2D siguiente, que ya opera sobre el eje de canales. Como esos canales ahora contienen features de $t-1$, $t$ y $t+1$, la convolución puede aprender pesos que los combinan: **una convolución temporal ejecutada por una capa que ya estaba ahí**.

### Modelos offline con desplazamiento bidireccional (slide 18)

Las cuatro viñetas de la clase, con su lectura:

| Viñeta de la clase | Qué significa |
|---|---|
| "Muestrear T fotogramas del vídeo" | el muestreo por segmentos de TSN; TSM no lo reemplaza, lo hereda |
| "Se inserta un módulo de cambio temporal para cada bloque residual. Se añade información futura y pasada al fotograma actual" | en ResNet-50 son **16 módulos** (3+4+6+3 bloques bottleneck) |
| "Para cada TSM insertado, el campo receptivo temporal se ampliará en 2" | uno hacia cada lado; apilados, la cobertura crece de forma acumulativa |
| "TSM puede convertir fácilmente cualquier modelo CNN en 2D en un modelo pseudo-3D" | *pseudo*-3D: el comportamiento de una 3D con el presupuesto de una 2D |

### Modelos con desplazamiento unidireccional (slide 19)

| Viñeta de la clase | Qué significa |
|---|---|
| "Solo se añade información sobre el pasado al fotograma actual" | es **causal**: aplicable a un stream en vivo |
| "TSM reemplaza 1/8 del mapa de características del fotograma actual con información antigua" | correcto **para este modo** |
| "Baja latencia para estimación en tiempo real" | 13,4 ms en una Jetson Nano, 34,5 ms en un Galaxy Note8 |

{{< concept-alert type="cuidado" >}}
**Dos precisiones sobre estas dos slides.**

**La proporción.** El 1/8 vale para el modo unidireccional. En el **bidireccional** —el de la slide anterior, y el del checkpoint que usa el laboratorio— se desplaza **1/4**: un octavo hacia el pasado y un octavo hacia el futuro. El código lo confirma (`fold = C // 8`, y se mueven dos folds), y el paper es explícito: *"the performance reaches the peak when 1/4 (1/8 for each direction) of the channels are shifted"*.

**El nombre.** La slide 19 se titula "Modelos **offline** con desplazamiento unidireccional". La sección correspondiente del paper es *"4.2 Online Models with Uni-directional TSM"*. Unidireccional significa **online**: el sentido de renunciar a los frames futuros es justamente poder procesar video que todavía no terminó de ocurrir. Si se lee la slide al pie de la letra, la razón de ser del modo se pierde.
{{< /concept-alert >}}

### La arquitectura completa (slide 20)

```
S1 ──► Conv2D + TSM ──┐
 ⋮          ⋮          ├──► Consensus (Avg) ──► predicción
SN ──► Conv2D + TSM ──┘
```

Comparado con el diagrama de TSN de la slide 14, la única diferencia es el "+ TSM" en cada bloque. Y esa es la tesis visual de la clase: **la misma arquitectura, el mismo consenso, el mismo costo — con modelado temporal adentro**.

En el código del laboratorio esta identidad es literal: la clase que instancia el modelo **se llama `TSN`**, y TSM es esa misma clase con el argumento `is_shift=True`.

### Lo que la clase no cubre

Dos decisiones de diseño quedan fuera del material, y son las que hacen que el método funcione:

**Partial shift.** Desplazar *todos* los canales cuesta +13,7 % de latencia en CPU y **pierde 2,6 puntos** de precisión contra la baseline. La fracción de 1/4 no es una elección arbitraria sino el máximo de una curva.

**Residual shift.** El módulo va **dentro** de la rama residual, no antes del bloque, para que la conexión identidad siga transportando la activación sin desplazar. Es la diferencia entre `shift_place='blockres'` y `'block'` en el código, y sin ella se degrada el modelado espacial.

Ambas se desarrollan en [Desplazamiento Temporal](/fundamentos/desplazamiento-temporal) y se miden en el [Laboratorio 40](/laboratorios/lab-40/04-la-curva-de-proporcion).

---

## 6. Comparación de modelos (slides 21-22)

Dos diapositivas de resultados: TSM contra TSN, y TSM contra las redes 3D. Los números del paper, que es de donde salen:

**Contra la baseline 2D** (ResNet-50, 8 frames):

| Dataset | TSN | TSM | Δ |
|---|---|---|---|
| Kinetics | 70,6 % | 74,1 % | **+3,5** |
| UCF-101 | 91,7 % | 95,9 % | +4,2 |
| HMDB-51 | 64,7 % | 73,5 % | +8,8 |
| Something-Something V1 | 20,5 % | 47,3 % | **+28,0** |
| Something-Something V2 | 30,4 % | 61,7 % | **+31,3** |
| Jester | 83,9 % | 97,0 % | +11,7 |

**Contra las redes 3D** (Tesla P100):

| Modelo | FLOPs | Params | Latencia | Throughput |
|---|---|---|---|---|
| I3D | 306 G | 35,3 M | 165,3 ms | 6,1 videos/s |
| **TSM 8F** | **33 G** | **24,3 M** | **17,4 ms** | **77,4 videos/s** |

9,5× menos latencia, 12,7× más throughput, y 1,8 puntos más de precisión. Es el argumento de la segunda ruta de la slide 4, cumplido.

---

## 7. Kinetics contra Something-Something (slides 23-27)

La clase cierra comparando los dos datasets, y es su parte más interesante porque explica la asimetría de la tabla anterior.

**[Kinetics](/papers/kinetics-kay-2017)** — "Cepillando los dientes":

- "Algunas investigaciones han demostrado que **Kinetics no requiere un análisis temporal laborioso** para entenderlo."
- "Sus etiquetas son una **referencia general** a los vídeos."
- Videos cortos y reales, en la naturaleza, una acción por video, centrada en una persona.

**[Something-Something](/papers/something-something-goyal-2017)** — "Girando una pulsera":

- "**Requiere modelado temporal detallado.**"
- "Sus etiquetas están **muy correlacionadas con lo que sucede** en cada vídeo."
- Videos cortos y reales, en entornos naturales, una acción por video, centrada en una persona.

Los cuatro últimos puntos son idénticos en ambos. **La diferencia está enteramente en los dos primeros**, y es una diferencia sobre las **etiquetas**, no sobre los videos.

Kinetics etiqueta *qué se ve*: "tocando guitarra", "salto alto". Eso se resuelve en buena medida desde un frame — hay una guitarra, hay una pista de atletismo. Something-Something etiqueta *qué le pasa a algo*: "empujando algo de izquierda a derecha", "fingiendo levantar algo sin levantarlo". Ahí no hay objeto que identificar: la etiqueta **es** la trayectoria.

{{< concept-alert type="clave" >}}
Por eso la misma modificación vale +3,5 o +31,3 puntos. **La tabla de resultados de TSM es, leída al revés, una medición de cuánta temporalidad exige cada benchmark.** Y deja una advertencia metodológica: un modelo puede rendir excelente en Kinetics sin modelar tiempo en absoluto. El [Laboratorio 36](/laboratorios/lab-36) llegó a esa conclusión desde el otro lado, mostrando que un *bag of frames* sin orden alguno alcanzaba 85,9 % en UCF-11.
{{< /concept-alert >}}

---

## 8. Referencias y material extra (slides 28-29)

La clase cita dos papers:

- Ji Lin, Chuang Gan, Song Han. *TSM: Temporal Shift Module for Efficient Video Understanding*, 2019.
- Limin Wang et al. *Temporal Segment Networks for Action Recognition in Videos*, 2017.

Un detalle bibliográfico: la segunda referencia es la **versión extendida** de TSN publicada en IEEE TPAMI (arXiv:1705.02953, 2017), no el paper original de ECCV 2016 (*"Towards Good Practices for Deep Action Recognition"*). Ambas describen el mismo marco; el análisis del site cubre [la versión de ECCV](/papers/tsn-wang-2016), que es la canónica.

La clase termina con una diapositiva de **material extra sobre Mask Autoencoder (MAE)**, abierta con una pregunta: *"¿Cómo aprende un niño a completar un rompecabezas al que le faltan piezas?"*. Es un puente hacia el aprendizaje autosupervisado, que el diplomado desarrolla en la [Clase 28](/clases/clase-28) — donde el MAE aparece con su análisis completo. La conexión con video no es casual: enmascarar y reconstruir parches es una de las pocas vías para preentrenar modelos de video sin etiquetas, y es lo que sucedió en el campo después de TSM.

---

## Ver también

- [Profundización](profundizacion) — la matemática del desplazamiento: la descomposición, la aritmética del fold, el campo receptivo acumulado y la invarianza al orden del consenso.
- [Práctica desde 0](practica) — implementar el módulo y verificar su equivalencia con una convolución temporal, en triple framework.
- [Laboratorio 40](/laboratorios/lab-40) — inferencia con el checkpoint oficial y cuatro experimentos que miden el aporte real del desplazamiento.
- [Clase 38](/clases/clase-38) — la estrategia opuesta frente al mismo problema: inflar convoluciones 2D a 3D.
- [Fundamento: Desplazamiento Temporal](/fundamentos/desplazamiento-temporal) — el mecanismo desarrollado de forma autónoma.
