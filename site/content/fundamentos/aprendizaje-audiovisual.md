---
title: "Aprendizaje Audiovisual"
weight: 138
math: true
---

El **aprendizaje audiovisual** explota un hecho gratuito: en un video, la imagen y el sonido llegan **sincronizados y describen la misma escena**. Nadie tuvo que anotar esa correspondencia — está en el archivo. Eso convierte cada video sin etiquetar en un par de vistas alineadas de un mismo suceso, y abre dos usos distintos: usar una modalidad para **supervisar** a la otra, y usar las dos juntas para **decidir mejor** que cualquiera por separado.

Este fundamento acompaña a la [Clase 43](/clases/clase-43), donde ambos usos aparecen: [SoundNet](/papers/soundnet-aytar-2016) es el primero, [E2E-AVSR](/papers/e2e-avsr-petridis-2018) el segundo.

---

## 1. La sincronía como etiqueta

El argumento de escala es el que motiva toda el área. Los datasets de audio etiquetados son chicos y caros: ESC-50 tiene 2000 clips, DCASE tiene 10 ejemplos de entrenamiento por categoría. Los videos sin etiquetar son ilimitados.

Tres formas de convertir la sincronía en señal de entrenamiento:

**Destilación cross-modal.** Una red visual madura clasifica los fotogramas y sus salidas se usan como objetivo blando para una red de audio. La visión enseña; el audio aprende. Es [SoundNet](/papers/soundnet-aytar-2016), y se desarrolla en [Destilación de Conocimiento](/fundamentos/destilacion-de-conocimiento).

**Correspondencia audiovisual (AVC).** En vez de destilar, se plantea una tarea binaria: *¿este fragmento de audio corresponde a esta imagen?* Los positivos vienen del mismo instante del mismo video; los negativos, de videos distintos. Ninguna red enseña a la otra: **las dos se entrenan a la vez**, y el objetivo las obliga a construir un espacio donde imagen y sonido de la misma escena queden cerca. Es [Look, Listen and Learn](/papers/look-listen-learn-arandjelovic-2017) (2017) y su continuación [Objects that Sound](/papers/objects-that-sound-arandjelovic-2018) (2018).

**Predicción de unidades enmascaradas.** La versión moderna: enmascarar partes de ambos flujos y predecir unidades latentes descubiertas automáticamente, al estilo BERT. Es [AV-HuBERT](/papers/av-hubert-shi-2022) (2022).

{{< concept-alert type="clave" >}}
La diferencia entre destilación y correspondencia no es de técnica sino de **dirección de la autoridad**. En la destilación, el maestro visual define qué es correcto, y el estudiante queda acotado por lo que el maestro sabe: SoundNet no puede aprender un concepto que ImageNet y Places no representen. En la correspondencia, ninguna modalidad tiene razón por decreto — lo que se aprende es qué tienen en común, y eso puede exceder el vocabulario de ambas.

El precio es simétrico: la destilación converge rápido porque el objetivo es informativo desde el primer paso; la correspondencia parte de dos redes ignorantes y necesita mucho más dato para despegar.
{{< /concept-alert >}}

## 2. Complementariedad: por qué dos modalidades ayudan

El segundo uso es la **fusión**. Y su justificación es más específica de lo que suele decirse: no se trata de que "más información es mejor", sino de que **los ruidos de las dos modalidades son independientes**.

Un ruido de fondo destruye el canal acústico y **no toca** el visual. Una mala iluminación o una cabeza girada destruyen el visual y no tocan el acústico. Si ambas observaciones son condicionalmente independientes dada la clase, sus log-verosimilitudes se suman:

$$\log p(c \mid x_a, x_v) \;\propto\; \log p(x_a \mid c) + \log p(x_v \mid c) + \log p(c)$$

y la modalidad degradada aporta una verosimilitud plana que simplemente no estorba.

La consecuencia práctica, medida sobre un montaje sintético donde solo varía el ruido acústico:

| SNR (dB) | solo audio | solo video | fusión | ganancia |
|---|---|---|---|---|
| −5 | 12,13 % | 78,42 % | 83,49 % | **+71,36** |
| 0 | 27,41 % | 78,41 % | 89,50 % | +62,08 |
| 5 | 63,84 % | 78,08 % | 97,02 % | +33,18 |
| 10 | 96,91 % | 78,71 % | 99,87 % | +2,96 |
| 20 | 100,00 % | 78,11 % | 100,00 % | **+0,00** |

Dos lecturas:

- **La columna del video es constante.** No es que el modelo visual sea robusto: es que el ruido acústico no lo alcanza. En la figura equivalente del paper de Petridis et al. es literalmente una línea horizontal, y ahí está todo el argumento de la fusión.
- **La ganancia es nula en condiciones limpias y máxima en las peores.** No porque la fusión funcione mal en limpio, sino porque no hay nada que arreglar. Petridis et al. miden exactamente esa forma: **+0,3 puntos** con audio limpio y **+14,1** a −5 dB.

## 3. Dónde fusionar

Tres esquemas, por el punto del pipeline en que se juntan los flujos:

**Fusión temprana.** Concatenar las señales o sus features de bajo nivel y procesarlas con una sola red. Permite modelar interacciones finas —la correlación entre el instante en que se abren los labios y el instante del sonido— pero obliga a alinear temporalmente ambos flujos y multiplica el costo.

**Fusión tardía.** Cada modalidad tiene su clasificador completo y se combinan las salidas (promedio, voto, producto). Es robusta y modular: si falta una modalidad, la otra sigue funcionando. Pierde toda interacción de bajo nivel.

**Fusión intermedia.** Cada flujo produce una representación de nivel medio y un módulo aprendido las combina. Es lo que hace [E2E-AVSR](/papers/e2e-avsr-petridis-2018): dos ResNet más dos BiGRU por modalidad, y una **tercera BiGRU de dos capas** que recibe la concatenación y modela la dinámica temporal conjunta.

{{< concept-alert type="advertencia" >}}
**La fusión con pesos fijos puede empeorar el resultado.** Medido sobre el mismo montaje, con el audio limpio y el canal visual degradándose mientras el fusor sigue creyendo que vale lo mismo:

| $\sigma$ visual | solo video | solo audio | fusión tardía | delta |
|---|---|---|---|---|
| 1,0 | 77,58 % | 100,00 % | 100,00 % | +0,00 |
| 4,0 | 17,03 % | 100,00 % | 100,00 % | +0,00 |
| 8,0 | 7,20 % | 100,00 % | 99,79 % | −0,21 |
| 16,0 | 4,54 % | 100,00 % | **94,51 %** | **−5,49** |

Un promedio a ciegas arrastra a la modalidad buena. Por eso E2E-AVSR entrena la BiGRU de fusión **inyectando ruido entre −5 y 20 dB elegido al azar**: el fusor tiene que ver ambos regímenes para aprender a ponderar según la condición, en vez de promediar siempre igual.
{{< /concept-alert >}}

## 4. El orden de entrenamiento

Un detalle de ingeniería que los papers del área repiten y que rara vez se explica: **entrenar la arquitectura multimodal de una sola vez desde cero no funciona bien**. E2E-AVSR lo dice sin rodeos —*"entrenar directamente end-to-end cada flujo lleva a un rendimiento subóptimo"*— y usa un currículo de cinco etapas: primero cada flujo con un *back-end* convolucional temporal, después se reemplaza por la BiGRU con el frente congelado, después cada flujo completo end-to-end, después la BiGRU de fusión con los flujos congelados, y recién al final todo junto.

La razón es un problema de balance de gradientes: la modalidad que aprende más rápido domina el objetivo compartido y la otra queda subentrenada — el mismo fenómeno que [FairMOT](/papers/fairmot-zhang-2020) documenta entre detección y re-identificación. Congelar por etapas es la forma barata de forzar que cada rama aprenda antes de competir.

## 5. Aplicaciones

- **Reconocimiento audiovisual de habla (AVSR)** y **lectura de labios** — ver [Lectura de Labios](/fundamentos/lectura-de-labios).
- **Clasificación de escenas y eventos acústicos** con supervisión visual.
- **Separación de fuentes guiada por video**: aislar la voz de quien se ve hablando.
- **Localización de la fuente sonora** en el cuadro, que es lo que [Objects that Sound](/papers/objects-that-sound-arandjelovic-2018) obtiene como subproducto de la correspondencia.
- **Detección de deepfakes**, apoyada en la desincronización entre labios y audio.
- **Recuperación cross-modal**: buscar video con una consulta de audio y viceversa.

---

## Ver también

- [SoundNet (2016)](/papers/soundnet-aytar-2016) y [E2E-AVSR (2018)](/papers/e2e-avsr-petridis-2018) — los dos papers de la Clase 43.
- [Look, Listen and Learn (2017)](/papers/look-listen-learn-arandjelovic-2017) y [Objects that Sound (2018)](/papers/objects-that-sound-arandjelovic-2018) — la rama de correspondencia.
- [AV-HuBERT (2022)](/papers/av-hubert-shi-2022) — la versión autosupervisada moderna.
- [Destilación de Conocimiento](/fundamentos/destilacion-de-conocimiento) · [Lectura de Labios](/fundamentos/lectura-de-labios) · [Aprendizaje Autosupervisado](/fundamentos/aprendizaje-autosupervisado)
- [Clase 43 — Práctica](/clases/clase-43/practica) — la tabla de SNR reproducida desde cero, en triple framework.
