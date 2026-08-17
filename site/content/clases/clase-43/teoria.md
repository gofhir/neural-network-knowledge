---
title: "Teoría - Aplicaciones para Audio y Video"
weight: 10
math: true
---

> **Recorrido de la Clase 43** del Diplomado IA UC (Alain Raymond, Gabriel Sepúlveda y Álvaro Soto, IALab PUC). Cuarenta y nueve diapositivas dedicadas a dos papers, presentados uno tras otro: [SoundNet](/papers/soundnet-aytar-2016) (NIPS 2016) y [End-to-End Audiovisual Speech Recognition](/papers/e2e-avsr-petridis-2018) (ICASSP 2018). Los dos usan la misma propiedad del video —que imagen y sonido describen la misma escena— para cosas opuestas: el primero para **entrenar**, el segundo para **decidir**.

---

# Parte 1 — SoundNet

## 1. El problema (diapositivas 4-7)

La clase construye la motivación en cuatro viñetas que se van revelando:

> **Tarea:** clasificación de sonidos naturales (escenas acústicas).
> **Motivación:** los campos de reconocimiento de objetos, reconocimiento de voz y traducción automática han sido revolucionados por datasets etiquetados masivos y modelos de deep learning.
> Sin embargo, **no ha habido el mismo progreso** en tareas de comprensión de sonido natural, por la falta de datasets etiquetados a gran escala.
> **¿Cómo podemos superar esta limitación?**

La asimetría es real y cuantificable: ImageNet tiene más de un millón de imágenes etiquetadas; ESC-50 tiene 2000 clips, y DCASE **diez ejemplos de entrenamiento por categoría**.

## 2. La solución (8-12)

> **Solución:** aprovechar…
> 1. **videos sin etiquetar**, que pueden adquirirse a escala masiva y contienen señales útiles sobre sonidos naturales;
> 2. **modelos visuales** que son muy buenos en reconocimiento de escenas y objetos.
>
> **Idea clave:** transferir conocimiento visual discriminativo desde modelos de reconocimiento visual bien establecidos hacia la modalidad de sonido, **usando video sin etiquetar como puente**.
>
> **Estrategia de entrenamiento:** *student-teacher*.

El diagrama es minimalista y dice todo: de un video salen dos flechas, *video* hacia un **clasificador visual de escenas y objetos (maestro)** y *audio* hacia un **clasificador de audio (estudiante)**.

Y el remate que la clase pone en su propia diapositiva:

> **¡Más datos permiten construir redes más profundas sin sobreajustar!**

{{< concept-alert type="clave" >}}
Esa frase parece una generalidad de manual, pero en el paper es un **resultado medido**, y es el más interesante de su tabla de ablación. Sin transferencia, pasar de 5 a 8 capas **empeora** el resultado en ESC-50: 65,0 % → 51,1 %. Con transferencia desde video sin etiquetar, la misma profundización **mejora**: 66,1 % → 72,9 %.

La profundidad no es buena ni mala en abstracto; es buena **cuando hay datos que la sostengan**, y el video sin etiquetar los provee.
{{< /concept-alert >}}

## 3. Maestro y estudiante (13-16)

**Maestro.** Hace predicciones sobre los fotogramas del video, en dos tareas: **reconocimiento de objetos** y **reconocimiento de escenas**. Para cada una hay un modelo independiente entrenado sobre **ImageNet** y **Places**, respectivamente.

**Estudiante.** Tres decisiones encadenadas, cada una consecuencia de la anterior:

- Como el sonido varía en duración, se usa una red **totalmente convolucional**.
- Como la representación se adapta al largo de la entrada, la **capa de salida debe manejar entradas de largo variable**.
- Como el modelo se entrena con video, se usa una **capa de salida convolucional** que produce salidas sobre múltiples instantes.

## 4. El objetivo (18)

$$\min_\theta \sum_{k=1}^{K}\sum_{i=1}^{N} D_{\mathrm{KL}}\big(g_k(y_i)\,\|\,f_k(x_i)\big), \qquad D_{\mathrm{KL}}(P\|Q) = \sum_j P_j \log\frac{P_j}{Q_j}$$

con $x_i \in \mathbb{R}^D$ la onda, $y_i \in \mathbb{R}^{3\times T\times W\times H}$ el video correspondiente, $g_k$ el maestro visual y $f_k$ el estudiante. La clase explica la elección: *"como la salida de la red de visión puede interpretarse como una distribución, usan la divergencia de Kullback-Leibler como función de pérdida"*.

Vale nombrar lo que la clase no nombra: esto es **[destilación de conocimiento](/fundamentos/destilacion-de-conocimiento)** en el sentido de [Hinton, Vinyals y Dean (2015)](/papers/distillation-hinton-2015), con la torsión de que maestro y estudiante **no comparten la modalidad de entrada**. No se comprime un modelo grande en uno chico: se traslada semántica de la vista al oído.

## 5. Los datos (20)

> Más de **2 millones de videos de Flickr**, consultando por etiquetas populares. Esto resultó en **más de un año de sonido y video natural continuo** para entrenamiento. La duración de cada video varía de unos segundos a varios minutos.
> **Post-procesamiento de audio:** convertir a MP3, reducir la tasa de muestreo a **22 kHz** y convertir a un solo canal.

Nótese lo mínimo del preproceso. No hay espectrogramas, no hay MFCC: la red recibe la onda.

## 6. Cómo se evalúa (21-22)

Aquí hay un giro que conviene no pasar por alto:

> Aunque entrenamos SoundNet para clasificar categorías visuales, **las categorías que queremos reconocer pueden no aparecer en los modelos visuales** (por ejemplo, estornudos). Queremos aprovechar la semántica aprendida desde la modalidad visual para resolver una tarea de clasificación de sonido.
> Entonces, **ignoraremos la capa de salida** y usaremos la representación interna como features para entrenar clasificadores.

Sobre esas features se entrena un **SVM lineal one-vs-all**. Los datasets: **DCASE** (10 escenas, 10 ejemplos de entrenamiento por categoría, 100 de test) y **ESC-50 / ESC-10** (50 y 10 categorías, 40 muestras cada una, 5 folds *leave-one-out*).

{{< concept-alert type="advertencia" >}}
Esa decisión de "ignorar la capa de salida" tiene una consecuencia que ni la clase ni el paper subrayan. Reconstruyendo el conteo de parámetros de la Tabla 1: de los **14,3 millones** de SoundNet-8, la capa de salida conv8 tiene **11,5 millones — el 80 %**. Y las features que efectivamente se usan (pool5) provienen del **1,72 %** del modelo.

El 80 % de la red existe para definir el objetivo de entrenamiento y se descarta al usarla. Es el mismo patrón que la [Clase 38](/clases/clase-38) encontró en C3D, donde 50 de sus 78 millones de parámetros están en dos capas densas.
{{< /concept-alert >}}

## 7. Resultados (23-26)

**DCASE:** RG 69 %, LTT 72 %, RNH 77 %, Ensemble 78 %, **SoundNet 88 %**.

**ESC-50 / ESC-10:** SVM-MFCC 39,6 / 67,5; autoencoder convolucional 39,9 / 74,3; Random Forest 44,3 / 72,7; Piczak ConvNet 64,5 / 81,0; **SoundNet 74,2 / 92,2**; *humanos 81,3 / 95,7*.

Y la ablación, que es la diapositiva más informativa de toda la primera mitad:

| Comparación | Configuración | ESC-50 | ESC-10 |
|---|---|---|---|
| **Pérdida** | 8 capas, $\ell_2$ | 47,8 % | 81,5 % |
| | 8 capas, KL | **72,9 %** | **92,2 %** |
| **Maestro** | solo ImageNet | 69,5 % | 89,8 % |
| | solo Places | 71,1 % | 89,5 % |
| | ambos | **72,9 %** | **92,2 %** |
| **Profundidad y transferencia** | 5 capas, desde cero | 65,0 % | 82,3 % |
| | 8 capas, desde cero | 51,1 % | 75,5 % |
| | 5 capas, video sin etiquetar | 66,1 % | 86,8 % |
| | 8 capas, video sin etiquetar | **72,9 %** | **92,2 %** |

Tres cosas se leen aquí, y las tres se desarrollan en la [profundización](profundizacion):

1. **KL contra $\ell_2$: 25 puntos.** Es la brecha más grande de la tabla, y es sorprendente porque Hinton demostró que a temperatura alta ambas pérdidas son equivalentes.
2. **La fila de profundidad prueba la frase de la diapositiva 12**, como se explicó arriba.
3. **Dos maestros superan a uno.** El estudiante está acotado por lo que sus maestros saben.

---

# Parte 2 — End-to-End Audiovisual Speech Recognition

## 8. La pregunta (29-31)

> **Objetivo:** reconocimiento de habla desde fuentes de audio y video.
> Pero, **¿por qué necesitamos video si tenemos el audio?**

El modelo que hay que diseñar:

> recibe flujos de audio y video → devuelve la palabra más probable de un vocabulario.
> **Audio crudo** + **video (región de la boca)** → Modelo → **palabra más probable**.

La clase deja la pregunta abierta durante toda la sección y la responde recién en la penúltima diapositiva, con la curva de relación señal-ruido.

## 9. Las características del modelo (33)

> - Modelo **end-to-end** basado en redes residuales y BiGRU.
> - **Primer modelo de fusión audiovisual que usa video y audio crudos.**
> - Dos flujos que extraen features directamente de las regiones de la boca y de las formas de onda.
> - Las dinámicas temporales se modelan con **BiGRU de 2 capas**.
> - La fusión ocurre mediante **otra BiGRU de 2 capas**.

## 10. La arquitectura (34-36)

**Flujo visual**
- Entrada: **29 fotogramas**.
- Convolución espacio-temporal: **64 núcleos de $5\times 7\times 7$**.
- **ResNet-34**: colapsa la dimensión espacial.
- **BiGRU de 2 capas**.

**Flujo de audio**
- No hace falta convolución espacio-temporal (la onda es 1D).
- **ResNet-18** con núcleos 1D de **5 ms** y paso de **0,25 ms**.
- Salida de la ResNet: **29 ventanas** mediante *average pooling* — para igualar la tasa de fotogramas del video.

**Capas de clasificación**
- **BiGRU de 2 capas** sobre la concatenación.
- Softmax que asigna una etiqueta a cada instante.

{{< concept-alert type="clave" >}}
El *average pooling* del flujo de audio no es una decisión de capacidad sino de **alineación**: existe únicamente para producir 29 ventanas y poder concatenar con los 29 fotogramas de video. Es fusión **intermedia** — ni pegar las señales crudas ni promediar las decisiones finales, sino juntar representaciones de nivel medio ya alineadas en el tiempo.

El flujo visual, además, es la arquitectura de [Stafylakis y Tzimiropoulos (2017)](/papers/lipreading-resnet-stafylakis-2017) con GRU en vez de LSTM — el paper que la clase cita como `[13]`.
{{< /concept-alert >}}

## 11. El dataset (37-38)

> **Lip Reading in the Wild (LRW)**. Segmentos cortos de programas de la BBC, principalmente noticias y programas de conversación. Todos los segmentos tienen **29 fotogramas (1,16 s)**. Contiene más de **1000 hablantes** y gran variación de pose de cabeza e iluminación. **500 palabras** (clases). Varias palabras son **visualmente familiares** entre sí (por ejemplo, *America* y *American*). El mayor dataset de lectura de labios en libertad disponible públicamente.

**Preprocesamiento.** En video, las regiones de boca ya vienen centradas y se extrae una **caja fija de 96×96**. En audio, **z-normalización**: media cero y desviación estándar uno.

**Aumentación.** En video, recorte aleatorio y volteo horizontal con 50 % de probabilidad aplicado a todos los fotogramas. En audio, **ruido aleatorio a distintos niveles entre −5 dB y 20 dB**, tomado de la base NOISEX.

Esa última línea parece un detalle de implementación y es una decisión de diseño central: el fusor se entrena viendo **todo el rango de condiciones acústicas**, y por eso puede aprender a ponderar según la calidad del audio en vez de promediar siempre igual.

## 12. El currículo de entrenamiento (40-45)

Seis diapositivas para lo que el paper resume en un párrafo, y la clase hace bien en detenerse: es la parte que más se subestima al reimplementar.

> Cada flujo se entrena **independientemente**. **Entrenar directamente end-to-end cada flujo lleva a un rendimiento subóptimo.**

1. Inicialmente se usa un ***back-end* convolucional temporal**, y se entrena hasta que no haya mejora por más de 5 épocas en validación.
2. El back-end convolucional se **reemplaza por la BiGRU**. Los pesos del frente 3D y de la ResNet quedan **fijos**. La BiGRU se entrena 5 épocas.
3. Una vez preentrenados ResNet y BiGRU, **cada flujo se entrena end-to-end**, con parada temprana de 5 épocas.
4. Ya entrenados los flujos individuales, sus pesos **inicializan** la arquitectura multi-flujo. Se agrega una BiGRU adicional encima, con los pesos de audio y video **fijos**, y se entrena 5 épocas.
5. Finalmente, **toda la red end-to-end**, con parada temprana.

## 13. Resultados (46-47)

| Flujo | Tasa de clasificación |
|---|---|
| A (end-to-end) | 97,7 |
| A (MFCC) | 97,7 |
| V (end-to-end) | 82,0 |
| V [13]* | **83,0** |
| V [15] | 76,2 |
| V [19] | 61,1 |
| **A + V (end-to-end)** | **98,0** |

Y la curva de relación señal-ruido, que es donde la clase por fin responde su pregunta: cuatro líneas —V, A, AV y MFCC— sobre un eje de −5 a 20 dB. A 20 dB las cuatro convergen cerca de 97,5. Al bajar el SNR, A y MFCC se desploman —MFCC más rápido— mientras **la línea de V es horizontal**. A −5 dB, el video solo está **por encima** del audio solo.

{{< concept-alert type="clave" >}}
Tres lecturas de esta tabla y esta curva, que juntas son el aporte real del paper:

**La onda cruda empata con MFCC en limpio (97,7 = 97,7) y le gana bajo ruido** (+7,5 puntos a −5 dB). Los MFCC descartan información que resulta útil justamente cuando hay que separar habla de ruido. Aprender la representación no compró exactitud; compró **robustez**.

**La fusión aporta +0,3 en limpio y +14,1 a −5 dB.** No es que funcione mal en condiciones limpias: es que ahí no hay nada que arreglar. La modalidad débil solo ayuda donde la fuerte falla.

**El flujo visual propio queda por debajo de la referencia** (82,0 contra 83,0), y el paper explica por qué: usa una caja fija de 96×96 mientras que `[13]` extrae la boca siguiendo puntos faciales. La contribución no está en el canal visual.
{{< /concept-alert >}}

## 14. Las desventajas (48)

La clase cierra con la lista que el propio paper declara:

> - Está **limitado a un conjunto fijo de palabras aisladas**.
> - El **proceso de entrenamiento es muy complejo**.
> - **No generaliza bien** a variaciones en el largo de la secuencia.

Las tres apuntan a lo mismo: la salida es un softmax de tamaño fijo sobre una ventana de tamaño fijo. La solución es la que la [Clase 41](/clases/clase-41) ya desarrolló para el audio — [CTC](/fundamentos/ctc-loss) o *seq2seq* con atención—, y [LipNet](/papers/lipnet-assael-2016) la había aplicado al video dos años antes de este paper.

---

## Lo que la clase deja fuera

Tres huecos, todos desarrollados en la [profundización](profundizacion):

1. **El nombre de lo que hace SoundNet.** Es destilación de conocimiento, y ponerle el nombre permite leer su ablación con las herramientas de [Hinton (2015)](/papers/distillation-hinton-2015) — incluyendo por qué los 25 puntos entre KL y $\ell_2$ son sorprendentes y qué los explica.

2. **La rama de correspondencia.** Un año después de SoundNet, [Look, Listen and Learn](/papers/look-listen-learn-arandjelovic-2017) reemplaza el maestro visual por una tarea simétrica —*¿este audio corresponde a esta imagen?*— y no necesita ningún modelo preentrenado.

3. **Qué pasó después de 2018.** [AV-HuBERT](/papers/av-hubert-shi-2022) (2022) alcanza 32,5 % de WER en LRS3 con **30 horas** etiquetadas, superando a un sistema entrenado con 31 000. Las tres limitaciones que la clase enumera al final quedan resueltas cambiando de dónde viene la supervisión, no agrandando la arquitectura.

---

**Siguiente:** [Profundización](profundizacion) — la aritmética de SoundNet capa por capa, por qué KL y $\ell_2$ difieren en 25 puntos cuando deberían ser equivalentes, y la estructura de la complementariedad audiovisual. Después, la [práctica](practica): destilación y fusión implementadas y medidas, en triple framework.
