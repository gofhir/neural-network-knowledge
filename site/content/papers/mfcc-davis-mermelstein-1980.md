---
title: "MFCC: Mel-Frequency Cepstral Coefficients (1980)"
weight: 394
math: true
---

{{< paper-card
    title="Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences"
    authors="Steven B. Davis, Paul Mermelstein (Haskins Labs)"
    year="1980"
    venue="IEEE Trans. ASSP"
    pdf="/papers/mfcc-davis-mermelstein-1980.pdf" >}}
Este es el paper que introdujo los **coeficientes cepstrales en frecuencia Mel (MFCC)** como la representación acústica de referencia para reconocimiento de voz. Su pregunta es deliberadamente comparativa: entre las representaciones paramétricas disponibles a fines de los setenta, ¿cuál retiene mejor la información fonéticamente significativa de la señal? Davis y Mermelstein enfrentaron cinco representaciones bajo condiciones idénticas de vocabulario, hablante y alineamiento temporal. El resultado fue contundente: **diez coeficientes cepstrales en frecuencia Mel calculados cada 6.4 ms alcanzaron 96.5 % y 95.0 % de reconocimiento** para dos hablantes, superando al cepstrum de frecuencia lineal (LFCC), al cepstrum de predicción lineal (LPCC), al espectro de predicción lineal (LPC) y a los coeficientes de reflexión (RC). El pipeline canónico —audio → *frames* → FFT → banco de filtros Mel → log → DCT → coeficientes— nace, en su forma definitiva, aquí. Es la fuente primaria del procedimiento de la [Clase 35](/clases/clase-35).
{{< /paper-card >}}

---

## Contexto: reconocimiento de voz antes del *deep learning*

En 1980 no existían las redes profundas ni los HMM maduros: el reconocimiento se apoyaba en **comparación de patrones** (*template matching*) con alineamiento temporal por programación dinámica. En ese marco, **la representación acústica lo era casi todo** —determinaba qué información sobrevivía a la etapa de comparación y, por tanto, fijaba el techo de desempeño. El objetivo al elegir una representación era doble: *comprimir* los datos eliminando lo no fonético y *realzar* lo que distingue fonemas.

Convivían dos escuelas: el **análisis de Fourier** (espectro de corto plazo, a veces suavizado con bancos de filtros) y el **análisis por predicción lineal (LPC)**, que modela la señal como un filtro todo-polos del tracto vocal. White y Neely (1976) habían concluido que ambos eran "esencialmente equivalentes"; Davis y Mermelstein sospechaban que la equivalencia se debía a que el vocabulario de prueba carecía de palabras confundibles, y diseñaron su experimento para forzar esa distinción.

El ingrediente perceptual venía de la psicoacústica: **el ancho de las bandas críticas del oído crece con la frecuencia** (resolución fina abajo, gruesa arriba). Pols (1966) había mostrado que los primeros autovectores de la covarianza de vocales, expresadas en energías de filtros de este tipo, explicaban el 91.8 % de la varianza y se parecían a una **expansión en serie de cosenos** —el MFCC en germen.

## Contribución central: los tres ingredientes del MFCC

Los MFCC son, en palabras de los autores, "el resultado de una transformada de cosenos del logaritmo real del espectro de energía de corto plazo expresado en una escala de frecuencia Mel". Esa frase contiene el corazón del pipeline:

1. **Escala Mel.** El eje de frecuencias se deforma para imitar la resolución no uniforme del oído. La forma difundida hoy es $m = 2595\,\log_{10}\!\left(1 + f/700\right)$. Precisión histórica: **el paper de 1980 no escribe esa fórmula cerrada**. En una nota al pie, citando a Fant (1959), define la escala como **lineal por debajo de 1000 Hz y logarítmica por encima**, y observa que las distintas versiones (Beranek, Koenig, Fant) no difieren significativamente. La fórmula de 2595 es una formalización posterior de esa misma idea.
2. **Logaritmo de las energías.** Comprime el rango dinámico de forma análoga a la percepción de sonoridad.
3. **Transformada de cosenos (DCT).** Produce un conjunto compacto y aproximadamente decorrelacionado de coeficientes —el *cepstrum*.

El banco de filtros son **20 filtros triangulares** en escala Mel hasta ~4600 Hz: la ganancia es máxima en la frecuencia central y decae linealmente hasta los filtros vecinos. La salida es la energía de banda; a cada una se le aplica el logaritmo, obteniendo $x_k$ ($k=1,\dots,20$). La ecuación (1) define entonces los coeficientes:

$$c_n = \sum_{k=1}^{20} x_k \,\cos\!\left[\,n\left(k - \tfrac{1}{2}\right)\frac{\pi}{20}\,\right], \qquad n = 1, \dots, M.$$

La DCT cumple tres funciones a la vez: **comprime** (de 20 log-energías a $M=10$ o $6$ coeficientes), **decorrelaciona** (aproxima la transformada de Karhunen-Loève, lo que justifica usar distancia euclidiana simple) y **separa fuente-filtro** (los coeficientes bajos capturan la envolvente formántica del tracto vocal; los altos, el detalle de pitch). El LFCC aplica la misma transformada pero sobre el espectro de frecuencia lineal: su única diferencia con el MFCC es **la deformación Mel del eje**, lo que hace de su comparación el experimento decisivo.

## Resultados e impacto

El corpus se diseñó para maximizar la confusabilidad fonética: **52 palabras CVC** de dos hablantes masculinos, 169 tokens de 57 oraciones, grabadas dos veces con dos meses de separación. Para las pruebas abiertas (sesiones distintas) con 10 coeficientes y tramas de 6.4 ms:

| Representación | Coefs. | Métrica | Abierta DZ | Abierta LL |
|---|---|---|---|---|
| **Cepstrum Mel (MFCC)** | 10 | Euclidiana | **96.5 %** | **95.0 %** |
| Cepstrum frec. lineal (LFCC) | 10 | Euclidiana | 96.5 % | 92.0 % |
| Cepstrum predicción lineal (LPCC) | 10 | Euclidiana | 94.7 % | 87.6 % |
| Espectro predicción lineal (LPC) | 10 | Itakura | 85.2 % | 84.3 % |
| Coeficientes de reflexión (RC) | 10 | Euclidiana | 83.1 % | 77.5 % |

Los MFCC ganan de forma consistente en toda condición. Más notable aún: **seis MFCC superan a cualquier otro conjunto de diez coeficientes** (menos es más). La ventaja del MFCC sobre el LFCC es que **suprime mejor la variación espectral insignificante en altas frecuencias**, justamente lo que hace la resolución gruesa de la escala Mel arriba. Los parámetros de Fourier vencen a los de predicción lineal, cuyas confusiones se concentran en las consonantes.

Los MFCC dominaron el reconocimiento de voz **tres décadas** (hasta el *deep learning* end-to-end, ~2012-2015) por su compacidad, su decorrelación —que los hizo compatibles con las mezclas de gaussianas de covarianza diagonal de los HMM-GMM—, su fundamento perceptual y su bajo costo. Incluso hoy, muchos sistemas que operan sobre **espectrogramas Mel** conservan las primeras etapas (filtros Mel + log) y omiten solo la DCT final, porque las CNN aprenden sus propias combinaciones de las bandas.

## Limitaciones

- **Condiciones restringidas.** Solo palabras monosilábicas CVC acentuadas, sin *clusters* consonánticos ni monosílabos átonos.
- **Dependencia del hablante.** El estudio es explícitamente *speaker-dependent*.
- **Poca robustez al ruido.** Grabaciones en ambiente silencioso con micrófono de alta calidad; los autores advierten que el ruido "sin duda deterioraría" el desempeño. Esta fragilidad es una limitación estructural que motivó técnicas de normalización posteriores (CMN, RASTA).
- **Descarta dinámica temporal.** Al normalizar el coeficiente cero (energía global) y comprimir cada trama, se pierde información que motivaría después los **coeficientes delta y delta-delta**.

## Por qué importa para la Clase 35

La Clase 35 (Introducción al [Análisis de Audio](/dominios/audio)) desarrolla los MFCC, y este paper es su fuente canónica. El pipeline que se enseña —**audio → frames → FFT → banco de filtros Mel → log → DCT → coeficientes**— es, paso por paso, el procedimiento definido aquí. Tres ideas para internalizar:

- **La escala Mel modela la percepción.** Resolución fina abajo, gruesa arriba, imitando las bandas críticas del oído. La comparación MFCC vs. LFCC de la Tabla I es la prueba experimental de que esa deformación mejora el reconocimiento.
- **El logaritmo y la DCT no son adornos.** El logaritmo lineariza la intensidad y convierte la convolución fuente-filtro en suma; la DCT decorrelaciona y comprime, habilitando distancias euclidianas simples. Juntos separan la envolvente del tracto vocal (el fonema) de la excitación (el pitch).
- **Menos es más.** Seis MFCC superaron a diez coeficientes de cualquier otra representación: la compacidad es una virtud, no un compromiso.

El detalle matemático de la escala y el banco de filtros vive en el fundamento [MFCC y escala Mel](/fundamentos/mfcc-y-escala-mel). Que una representación diseñada en 1980 para distinguir *pool* de *fool* siga siendo la base de biomarcadores de voz (Parkinson, disfonías) y del análisis de sonidos respiratorios y cardíacos es el mejor testimonio de la solidez perceptual del diseño de Davis y Mermelstein.
