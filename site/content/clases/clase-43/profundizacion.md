---
title: "Profundización - Qué se transfiere, qué se descarta y qué se suma"
weight: 20
math: true
---

> La [teoría](teoria) presentó dos papers y dos usos de la misma propiedad del video. Esta página desarma cuatro cosas que quedan implícitas: la aritmética de SoundNet, que revela dónde está el modelo y dónde no; por qué la brecha de 25 puntos entre KL y $\ell_2$ contradice aparentemente un teorema de Hinton, y qué la explica; la estructura formal de por qué dos modalidades se complementan y cuándo dejan de hacerlo; y el intercambio entre representación aprendida y representación diseñada, que la tabla del segundo paper resuelve de forma inesperada.
>
> Las cifras marcadas como **medidas** provienen de código ejecutado; el mismo de la [práctica](practica).

---

## Parte I — La aritmética de SoundNet

### I.1. Reconstruir la tabla

La Tabla 1 del paper declara, para cada capa, el número de filtros, el tamaño del núcleo, el paso, el relleno y la dimensión de salida. Con la fórmula estándar

$$n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$

y tomando la dimensión declarada de cada capa como entrada de la siguiente, la tabla cierra en **10 de 11 capas** (medido). La excepción:

| capa | entrada | declarado | calculado | delta |
|---|---|---|---|---|
| conv2 | 27 506 | 13 782 | **13 754** | −28 |

Una diferencia del 0,2 % en una capa de once. No cambia nada sustantivo, pero conviene saberlo si se reimplementa: copiar la tabla literalmente produce un desajuste que se arrastra.

De la aritmética salen además dos datos que el paper no reporta.

### I.2. El campo receptivo

Cuánto audio ve efectivamente una neurona de cada capa, propagando $r_{\ell} = r_{\ell-1} + (k_\ell - 1)\prod_{i<\ell} s_i$ (**medido**, a 22 kHz):

| capa | conv1 | pool1 | conv2 | conv3 | conv5 | pool5 | conv6 | conv7 | conv8 |
|---|---|---|---|---|---|---|---|---|---|
| campo receptivo | 2,9 ms | 3,5 ms | 26 ms | 210 ms | 512 ms | **791 ms** | 1,91 s | 4,13 s | **14,54 s** |

Dos observaciones:

**pool5 —la capa que da las mejores features— integra unos 790 ms.** Es del orden de la duración de un evento acústico completo: un ladrido, un portazo, una sirena que pasa. No es casualidad que sea la mejor capa para clasificar sonidos.

**conv8 integra casi 15 segundos.** SoundNet no es un detector de eventos cortos sino un modelo de **escena acústica**, y su arquitectura lo declara antes que cualquier experimento. Contrasta de forma útil con el hallazgo de la [Clase 39](/clases/clase-39), donde el ejemplo analizado cubría 6,6 ms: dos redes convolucionales sobre onda cruda pueden operar en escalas temporales separadas por **tres órdenes de magnitud**, y eso —no la profundidad ni el número de filtros— es lo que determina qué pueden reconocer.

### I.3. Dónde están los parámetros

| bloque | parámetros | fracción |
|---|---|---|
| conv1 … conv5 (hasta pool5) | 247 280 | **1,72 %** |
| conv6 | 524 800 | 3,7 % |
| conv7 | 2 098 176 | 14,6 % |
| **conv8** | **11 478 393** | **80,0 %** |
| total | 14 348 649 | 100 % |

{{< concept-alert type="advertencia" >}}
**El 80 % del modelo se descarta al usarlo.** conv8 existe para producir las 1401 salidas que definen el objetivo de destilación; la evaluación usa pool5. Las features que alcanzan 74,2 % en ESC-50 provienen del **1,72 %** de los parámetros.

Y la tabla de "qué capa da mejores features" lo confirma desde el otro lado: en ESC-50, pool5 da 74,2 % y conv8 apenas **45,7 %** — casi 30 puntos menos. Las capas finales están especializadas en reproducir categorías visuales y pierden generalidad.

Es el mismo patrón que la [Clase 38](/clases/clase-38) midió en C3D (50 de 78 millones de parámetros en fc6 y fc7). En ambos casos la cabeza cara es andamiaje de entrenamiento, no producto.
{{< /concept-alert >}}

---

## Parte II — Los 25 puntos entre KL y $\ell_2$

### II.1. La aparente contradicción

La ablación de SoundNet reporta, sobre ESC-50: **KL 72,9 %**, **$\ell_2$ 47,8 %**. Veinticinco puntos.

Pero [Hinton, Vinyals y Dean (2015)](/papers/distillation-hinton-2015) demostraron que en el límite de temperatura alta el gradiente de la destilación es

$$\frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2}(z_i - v_i)$$

es decir, **exactamente** el gradiente de una regresión de mínimos cuadrados sobre los logits. Si las dos pérdidas convergen al mismo objetivo, ¿de dónde salen 25 puntos?

La convergencia es real y rápida (**medido**, coseno entre el gradiente exacto de KL y la aproximación de mínimos cuadrados):

| $T$ | 1 | 2 | 5 | 10 | 25 | 100 | 1000 |
|---|---|---|---|---|---|---|---|
| coseno | 0,9557 | 0,9739 | 0,9924 | 0,9977 | 0,9996 | 0,99997 | 1,000000 |

Hay dos respuestas, y la segunda es la importante.

### II.2. Primera respuesta: $T = 1$

SoundNet destila a temperatura 1. A esa temperatura el coseno es 0,956 — cercano, pero no 1. Más aún: el teorema exige que los logits estén **centrados en cero para cada ejemplo**, condición que nadie impone en la práctica.

Y Hinton señala que **la no equivalencia a $T$ baja es deseable**: *"a temperaturas más bajas, la destilación presta mucha menos atención a hacer coincidir los logits que son mucho más negativos que el promedio. Esto es potencialmente ventajoso porque esos logits están casi completamente no restringidos por la función de costo con la que se entrenó el modelo grande, así que podrían ser muy ruidosos."*

Con 1401 salidas y fotogramas de Flickr, la mayoría de esos logits **son** ruido: un plano de una playa no tiene ninguna opinión calibrada sobre la clase «acordeón».

### II.3. Segunda respuesta: hay dos cosas que se llaman $\ell_2$

Esta es la que explica la magnitud. El paper de SoundNet dice haber probado *"pérdida $\ell_2$ sobre las salidas objetivo"* — sobre las **probabilidades**, no sobre los logits. Y esas dos pérdidas se comportan de forma completamente distinta.

**$\ell_2$ sobre logits**: $\lVert z - v\rVert^2$. No atraviesa el softmax, no se satura. Es el límite del teorema de Hinton.

**$\ell_2$ sobre probabilidades**: $\lVert \sigma(z) - \sigma(v)\rVert^2$. El gradiente debe pasar por el jacobiano del softmax:

$$\frac{\partial}{\partial z_k}\lVert q - p\rVert^2 \;=\; 2\,q_k\!\left[(q_k - p_k) - \sum_j q_j (q_j - p_j)\right]$$

El factor $q_k$ del frente es fatal: **donde el estudiante asigna probabilidad casi nula, el gradiente es casi nulo**, por equivocada que esté esa clase. La pérdida es ciega justo donde hay que corregir.

Medido sobre un problema sintético de 400 clases con un maestro concentrado —la situación de un clasificador de ImageNet sobre fotogramas de Flickr—:

| Pérdida | top-1 | solape top-5 | correlación de rango |
|---|---|---|---|
| KL, $T=1$ | 67,87 % | 56,15 % | 0,8577 |
| KL, $T=2$ | 62,43 % | 59,85 % | 0,9385 |
| KL, $T=4$ | 57,10 % | 65,09 % | 0,9837 |
| **$\ell_2$ sobre probabilidades** | **7,57 %** | 8,45 % | 0,1674 |
| $\ell_2$ sobre logits | **92,57 %** | 94,36 % | 0,9987 |

La magnitud del gradiente lo explica (**medido**, con $K = 400$ clases y el estudiante sin entrenar):

| rango de la clase | $p_{\text{maestro}}$ | $|\nabla_{\text{KL}}|$ | $|\nabla_{\ell_2\text{ probs}}|$ | razón |
|---|---|---|---|---|
| 0 | 2,59e−01 | 2,57e−01 | 1,29e−03 | **200×** |
| 4 | 5,32e−02 | 5,07e−02 | 2,54e−04 | 200× |
| 100 | 2,95e−05 | 2,47e−03 | 1,24e−05 | 200× |

El factor es exactamente $K/2 = 200$: el jacobiano del softmax introduce un $q \approx 1/K$ que aplasta el gradiente en todas partes.

{{< concept-alert type="clave" >}}
**Los 25 puntos de SoundNet no miden "KL contra mínimos cuadrados".** Miden "objetivo que atraviesa el softmax **saturado** contra objetivo que no". Con $\ell_2$ sobre logits, la brecha no solo desaparece: se invierte.

Es una lección de implementación con nombre propio. Al leer «usamos pérdida L2» en un paper de destilación hay que preguntar **sobre qué**, porque las dos variantes comparten nombre y no comparten comportamiento.

Nótese también el intercambio dentro de la familia KL: al subir $T$, el top-1 **baja** (67,9 → 57,1) y la correlación de rango **sube** (0,858 → 0,984). Temperatura alta enseña la estructura relativa completa a costa de la decisión puntual — y si el estudiante va a usarse como **extractor de features**, que es exactamente el caso de SoundNet, la estructura vale más que el argmax.
{{< /concept-alert >}}

---

## Parte III — Por qué se complementan dos modalidades

### III.1. La estructura formal

Si las dos observaciones son **condicionalmente independientes dada la clase** —supuesto razonable cuando los ruidos tienen orígenes físicos distintos—, entonces

$$\log p(c \mid x_a, x_v) = \log p(x_a \mid c) + \log p(x_v \mid c) + \log p(c) + \text{cte}$$

Las evidencias se **suman**. Y una modalidad degradada aporta una log-verosimilitud casi plana: no ayuda, pero tampoco estorba.

Esto explica la forma exacta de la curva de Petridis et al., y se reproduce en un montaje controlado donde solo varía el ruido acústico (**medido**, 40 clases, canal visual con pares ambiguos):

| SNR (dB) | solo audio | solo video | fusión tardía | fusión óptima | ganancia |
|---|---|---|---|---|---|
| −5 | 12,13 % | **78,42 %** | 79,16 % | 83,49 % | **+71,36** |
| 0 | 27,41 % | 78,41 % | 81,58 % | 89,50 % | +62,08 |
| 5 | 63,84 % | 78,08 % | 88,87 % | 97,02 % | +33,18 |
| 10 | 96,91 % | 78,71 % | 98,70 % | 99,87 % | +2,96 |
| 20 | 100,00 % | 78,11 % | 100,00 % | 100,00 % | **+0,00** |

Reproduce las tres propiedades de la figura del paper: la línea del video es **plana**, la ganancia **crece** al bajar el SNR, y a SNR bajo el video solo **supera** al audio solo.

### III.2. El techo del canal visual

La columna del video no solo es plana: está clavada en ~78 %. Y ese techo no baja con más datos.

En el montaje, 40 clases se agrupan en 20 pares casi idénticos en la modalidad visual — el análogo de los **visemas**, las configuraciones de boca que comparten varios fonemas. Midiendo dónde caen los errores (**medido**):

- exactitud visual exacta: **78,03 %**
- exactitud de "acertar el par": **97,96 %**
- errores que caen **dentro** del par: **90,7 %**

El modelo sabe con casi total certeza en qué grupo está; lo que no puede es elegir dentro del grupo, porque **la información no está en la imagen**. `/p/`, `/b/` y `/m/` se ven igual: lo que los separa —sonoridad, nasalidad— ocurre dentro de la garganta. Ver [Lectura de Labios](/fundamentos/lectura-de-labios).

Ese techo es exactamente lo que el audio levanta, y es la razón estructural de que la fusión funcione en esta tarea y no en cualquiera.

### III.3. Cuándo la fusión estorba

La complementariedad no es gratuita. Con el audio limpio y el canal visual degradándose, mientras el fusor sigue ponderando ambos igual (**medido**):

| $\sigma$ visual | solo video | solo audio | fusión tardía | delta |
|---|---|---|---|---|
| 1,0 | 77,58 % | 100,00 % | 100,00 % | +0,00 |
| 4,0 | 17,03 % | 100,00 % | 100,00 % | +0,00 |
| 8,0 | 7,20 % | 100,00 % | 99,79 % | −0,21 |
| 16,0 | 4,54 % | 100,00 % | **94,51 %** | **−5,49** |

Un promedio a ciegas arrastra a la modalidad buena.

{{< concept-alert type="clave" >}}
Esto le da sentido a una línea de la clase que parece un detalle de aumentación: *"ruido aleatorio agregado a distintos niveles: [−5 dB, 20 dB]"*.

Entrenar la BiGRU de fusión sobre **todo el rango de condiciones acústicas** es lo que le permite aprender a ponderar según la calidad de la señal. Si solo hubiera visto audio limpio, habría aprendido a ignorar el video —donde no aporta— y habría fallado exactamente en el régimen que justifica su existencia.

La aumentación no está ahí para regularizar. Está para que el fusor **vea el caso que tiene que resolver**.
{{< /concept-alert >}}

### III.4. Cómo se fusiona también importa

En la tabla de III.1, la fusión tardía (promediar probabilidades) queda muy por detrás de la óptima (sumar log-verosimilitudes) a SNR bajo: 79,16 % contra 83,49 %.

La razón es que promediar probabilidades es una operación **aritmética** sobre un objeto que vive en escala **logarítmica**. La suma de log-verosimilitudes es la combinación correcta bajo independencia condicional; el promedio de probabilidades le da a la modalidad ruidosa un piso de influencia que no le corresponde.

La BiGRU de fusión de Petridis et al. no hace ninguna de las dos: **aprende** la combinación. Es la razón de fondo para preferir fusión intermedia sobre fusión tardía cuando hay datos suficientes.

---

## Parte IV — Aprender la representación contra diseñarla

El resultado más sobrio del segundo paper es una fila que no llama la atención:

| Flujo | Tasa de clasificación |
|---|---|
| A (end-to-end, onda cruda) | 97,7 |
| A (MFCC + BiGRU de 2 capas) | 97,7 |

Empate exacto. Una ResNet-18 sobre onda cruda, con su currículo de tres etapas, iguala a ochenta años de procesamiento de señales seguidos de un recurrente pequeño. Los autores lo dicen sin adornos: *"el esfuerzo requerido para entrenar el sistema end-to-end es significativamente mayor"*.

Pero la comparación en limpio no es la comparación completa:

| SNR | ventaja de la onda cruda sobre MFCC |
|---|---|
| 5 dB | +0,9 |
| 0 dB | +3,5 |
| **−5 dB** | **+7,5** |

{{< concept-alert type="clave" >}}
**Aprender la representación no compró exactitud; compró robustez.**

Los MFCC son una compresión con pérdida diseñada bajo supuestos de habla limpia: descartan la fase, promedian en bandas mel, y truncan los coeficientes cepstrales altos. Todo eso es información irrelevante *para el habla limpia* — y parte de ella resulta ser justamente la que permite separar voz de ruido de fondo.

La lección generaliza más allá del audio: una representación diseñada a mano encapsula supuestos sobre las condiciones de operación, y esos supuestos se cobran **fuera** de las condiciones donde se diseñó, no dentro. Comparar solo en el caso limpio hace invisible la diferencia.

Es también el argumento que la [Clase 39](/clases/clase-39) y el [Lab 39](/laboratorios/lab-39) discuten para sonidos ambientales, aquí medido sobre habla.
{{< /concept-alert >}}

---

## Parte V — Los dos usos de la misma propiedad

Vale terminar nombrando lo que une a los dos papers de la clase, porque no es obvio que sean la misma clase.

| | SoundNet | E2E-AVSR |
|---|---|---|
| Qué explota | imagen y sonido describen la misma escena | ídem |
| Cuándo usa las dos modalidades | **entrenamiento** | **inferencia** |
| Qué necesita en producción | solo audio | audio **y** video |
| Qué modalidad manda | la visual enseña | ninguna; se combinan |
| Qué resuelve | falta de etiquetas | degradación de una modalidad |

Y la línea que sigue, que la clase no recorre: [Look, Listen and Learn](/papers/look-listen-learn-arandjelovic-2017) (2017) elimina el maestro y hace la relación **simétrica** —*¿corresponden estos dos fragmentos?*—, sin necesitar ningún modelo preentrenado. [AV-HuBERT](/papers/av-hubert-shi-2022) (2022) la lleva a predicción enmascarada de unidades latentes y obtiene el resultado que cierra el arco: **32,5 % de WER en LRS3 con 30 horas etiquetadas**, contra 33,6 % de un sistema entrenado con 31 000 horas.

Mil veces menos anotación, mejor resultado. La señal que faltaba no eran etiquetas — era la correspondencia que el video ya traía.

---

## Resumen de lo verificado

| Afirmación | Resultado |
|---|---|
| La Tabla 1 de SoundNet es reproducible | 10/11 capas; conv2 da 13 754 y no 13 782 |
| Campo receptivo de pool5 | 791 ms, del orden de un evento acústico |
| Campo receptivo de conv8 | **14,54 s** — un modelo de escena, no de evento |
| conv8 concentra los parámetros | **80,0 %** de 14,3 M, y se descarta |
| Las features que se usan (pool5) | provienen del **1,72 %** del modelo |
| KL y $\ell_2$ sobre logits convergen a $T$ alta | coseno 0,956 → 1,000000 entre $T=1$ y $T=1000$ |
| $\ell_2$ sobre probabilidades colapsa | 7,57 % contra 92,57 % de $\ell_2$ sobre logits |
| El gradiente de $\ell_2$ sobre probabilidades es más chico | exactamente **$K/2$ = 200×** |
| La modalidad visual es invariante al SNR acústico | 78,4 % a −5 dB, 78,1 % a 20 dB |
| La ganancia de la fusión crece al bajar el SNR | +0,00 a 20 dB, **+71,36** a −5 dB |
| El error visual es confusión dentro del par ambiguo | **90,7 %** de los errores |
| La fusión con pesos fijos puede perjudicar | **−5,49 puntos** con el canal visual roto |

---

**Siguiente:** la [práctica](practica) — la destilación y la fusión implementadas desde cero y medidas, en triple framework.
