---
title: "SpecAugment: augmentation sobre el espectrograma (2019)"
weight: 409
math: true
---

{{< paper-card
    title="SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition"
    authors="Daniel S. Park et al. (Google Brain)"
    year="2019"
    venue="Interspeech 2019 / arXiv:1904.08779"
    pdf="/papers/specaugment-park-2019.pdf" >}}
SpecAugment es un método de *data augmentation* para reconocimiento automático del habla (ASR) que se aplica **directamente sobre el espectrograma log-Mel** que entra a la red, en lugar de perturbar la forma de onda cruda. La política consta de tres deformaciones: **time warp** (deformar el eje temporal), **frequency masking** (enmascarar bandas de frecuencia) y **time masking** (enmascarar segmentos de tiempo). Al tratar el espectrograma como una imagen, borrar un bloque tiempo-frecuencia es el análogo directo de **Cutout** en visión —dos coautores (Cubuk, Zoph) vienen justamente de AutoAugment/Cutout. Aplicado sobre redes **Listen, Attend and Spell (LAS)**, alcanza el estado del arte en LibriSpeech 960h y Switchboard 300h **incluso sin modelo de lenguaje**: **6.8% WER en test-other sin LM** (5.8% con shallow fusion), batiendo al mejor híbrido previo (7.5% con LM). Es la slide de augmentation y la sección 6.3 del laboratorio de la [Clase 37](/clases/clase-37).
{{< /paper-card >}}

---

## Contexto: la augmentation clásica de audio es cara

El aprendizaje profundo domina el ASR moderno, pero estos modelos **sobreajustan con facilidad y exigen muchos datos**. La respuesta clásica ha sido generar audio artificial, y el paper repasa una genealogía de técnicas que comparten un rasgo: **operan sobre la señal acústica, no sobre las features**. Distorsión espectral elástica, **VTLP** (perturbación del tracto vocal), audio ruidoso sintetizado, **perturbación de velocidad** y simuladores de sala acústica pertenecen todas a esa familia.

El problema común es el **costo**: casi todas exigen **re-sintetizar la forma de onda** —convolucionar con respuestas de sala, remuestrear, mezclar ruido— y luego **recalcular el espectrograma** desde cero. Es un pipeline pesado que multiplica el almacenamiento si se hace offline, o paga la síntesis en cada paso si se hace online. SpecAugment nace de una pregunta: si la red ya consume el espectrograma, ¿por qué no atacar directamente esa representación tratándola **como una imagen**?

## Método: tres deformaciones sobre el log-Mel

Se ve el espectrograma como una imagen: eje del tiempo horizontal ($\tau$ pasos), eje de frecuencia vertical ($\nu$ canales Mel). Cada una de las tres deformaciones busca robustez ante un tipo de degradación.

**Time warping.** Mediante `sparse_image_warp`, un punto sobre la línea horizontal central se desplaza a izquierda o derecha una distancia $w \sim \text{Uniforme}(0, W)$, con seis puntos ancla fijos en la frontera. Estira o comprime localmente el eje temporal, simulando variaciones de velocidad del habla. Es la deformación más "geométrica", la **más costosa** y la **menos influyente**.

**Frequency masking.** Se enmascaran $f$ canales de frecuencia consecutivos $[f_0, f_0+f)$, con

$$f \sim \text{Uniforme}(0, F), \qquad f_0 \sim \text{Uniforme}(0, \nu - f).$$

En la vista de imagen, pone a cero un bloque de **filas** contiguas: la red pierde una banda entera y debe reconocer el habla sin depender de ninguna banda particular.

**Time masking.** Análogamente, se enmascaran $t$ pasos de tiempo consecutivos $[t_0, t_0+t)$, con $t \sim \text{Uniforme}(0, T)$ y una **cota** de que la máscara no supere $p$ veces el largo del espectrograma (para no borrar un enunciado corto entero). Pone a cero un bloque de **columnas**.

Un detalle clave: los log-Mel se **normalizan a media cero**, de modo que **poner a cero equivale a poner la media** —sin sesgo, como en Cutout. Las políticas hechas a mano (LB, LD para LibriSpeech; SM, SS para Switchboard) parametrizan $W$, $F$, $T$, $p$ y el número de máscaras; LD ("double") aplica dos máscaras de cada tipo, más agresiva que LB.

## Resultados

Sobre la misma arquitectura LAS, SpecAugment produce un salto grande. En **LibriSpeech 960h** el WER (clean / other) baja de 4.1 / 12.5 a **2.8 / 6.8** sin LM, y de 3.2 / 9.8 a **2.5 / 5.8** con shallow fusion —una mejora del **22% relativo** en test-other sobre el estado del arte previo. En **Switchboard 300h** (Hub5'00, SWBD / CH) baja de 11.2 / 21.6 a **7.2 / 14.6** sin LM, superando al híbrido HMM previo (8.3 / 17.3) sin usar ningún LM.

El resultado conceptual más citado está en la discusión: **la augmentation convierte un problema de sobreajuste en uno de subajuste**. Las redes entrenadas con datos aumentados dejan de memorizar y pasan a *no alcanzar a ajustar* ni siquiera el conjunto original. A partir de ahí, las ganancias adicionales llegan con las **recetas clásicas contra el subajuste**: redes más anchas y profundas, y schedules de entrenamiento más largos.

## Limitaciones

- **Time warping aporta poco y es lo más caro.** Al apagar cada deformación por separado, su efecto es pequeño; siendo la más costosa (interpolación 2D), los autores recomiendan **descartarla primero** si el presupuesto aprieta. Frequency y time masking hacen el trabajo pesado y son casi gratuitas.
- **Label smoothing introduce inestabilidad** al combinarse con augmentation mientras el learning rate decae; hizo falta aplicarlo solo en las fases iniciales.
- **Políticas hechas a mano.** Los hiperparámetros se eligieron manualmente por corpus; no hay búsqueda automática de la política óptima.
- **Costo de llegar al estado del arte.** Aunque la augmentation es barata *por paso*, los números finales exigieron redes grandes y schedules de ~24 días sobre 32 TPUs, precisamente porque empuja al modelo hacia el subajuste.

## Por qué importa para la Clase 37

SpecAugment es el paper que ancla la slide de augmentation de la [Clase 37](/clases/clase-37) y la sección 6.3 del laboratorio, y fundamenta cada idea de la [augmentation de audio](/fundamentos/data-augmentation-de-audio):

- **"Hacer cero componentes del espectrograma, al azar."** Es literalmente frequency y time masking: ancho y posición estocásticos de un bloque de filas o columnas, puesto a cero (que es poner la media).
- **"Es dropout con estructura."** El dropout clásico apaga unidades **individuales e independientes**; SpecAugment apaga **bloques contiguos**. Esa contigüidad obliga a la red a resistir la pérdida de una banda entera o un segmento entero —no de píxeles sueltos que interpolaría trivialmente— imitando degradaciones reales (una banda perdida por un canal telefónico, un fonema tapado por ruido).
- **"Si conocen Cutout en imágenes, es exactamente eso."** El paper lo dice con todas sus letras: time y frequency masking están inspiradas en Cutout (DeVries y Taylor, 2017).
- **"Casi gratis."** A diferencia de speed perturbation, VTLP o simulación de sala —que perturban la forma de onda y obligan a re-sintetizar audio y recalcular el espectrograma—, SpecAugment **edita un tensor que ya está en memoria** camino a la red. No hay síntesis, ni datos adicionales, ni pipeline offline: se aplica **online, dentro del loop de entrenamiento**, en un par de líneas de código sobre el batch.

La lección transversal es la tesis del **sobreajuste convertido en subajuste**: una augmentation barata y bien diseñada lleva a un modelo end-to-end —sin la ingeniería de un híbrido, sin siquiera un LM— al estado del arte, y de ahí el camino a más desempeño es el clásico: más capacidad y más entrenamiento.
