---
title: "FaceForensics++: detectar manipulaciones faciales (2019)"
weight: 466
math: true
---

{{< paper-card
    title="FaceForensics++: Learning to Detect Manipulated Facial Images"
    authors="Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner (TU München / Università Federico II di Napoli / FAU Erlangen)"
    year="2019"
    venue="ICCV 2019 / arXiv:1901.08971"
    arxiv="1901.08971"
    pdf="/papers/faceforensics-rossler-2019.pdf" >}}
El contrapeso que la Clase 44 no incluye. Mientras el resto de la clase muestra cómo generar medios sintéticos, este paper construye la infraestructura para **detectarlos**: más de **1,8 millones de imágenes manipuladas** provenientes de 4000 videos falsos, producidas con cuatro métodos distintos —DeepFakes, Face2Face, FaceSwap y NeuralTextures—, un benchmark automático, y una **línea base humana** medida. El resultado principal es doble: los detectores automáticos superan claramente a los observadores humanos, y su desempeño **cae con la compresión**, que es exactamente la condición en que circulan los videos reales.
{{< /paper-card >}}

---

## Qué aporta

**Un dataset con escala y variedad de métodos.** Un orden de magnitud mayor que los conjuntos previos, y —más importante— generado con **cuatro técnicas distintas**, lo que permite por primera vez medir si un detector generaliza fuera del método con el que se entrenó.

**Tres niveles de compresión.** Los videos se distribuyen sin comprimir, con compresión ligera y con compresión fuerte, imitando lo que ocurre al subir un video a una red social. Es la variable que más afecta a la detección y la que los trabajos previos ignoraban.

**Una línea base humana.** Se mide cuán bien distinguen las personas los videos manipulados de los auténticos. Los detectores entrenados los superan por un margen amplio — un dato importante para el diseño de políticas: **no se puede delegar la verificación en el ojo del usuario**.

**Un benchmark automático** con conjunto de test oculto, en la línea de lo que [MOT16](/papers/mot16-milan-2016) hizo para seguimiento.

## Los resultados que importan

**La detección funciona en condiciones controladas.** Con arquitecturas estándar de clasificación de imágenes —XceptionNet es la que mejor rinde— la exactitud es alta cuando se entrena y evalúa sobre el mismo método de manipulación y sin compresión fuerte.

**Y se degrada con la compresión.** Los artefactos de bajo nivel en que se apoyan los detectores son, en buena medida, los mismos que el códec elimina. Cuanto más se comprime, menos queda de la señal que delata.

**Y generaliza mal entre métodos.** Un detector entrenado sobre un tipo de manipulación pierde desempeño al enfrentar otro. Es el hallazgo que la literatura posterior confirmó una y otra vez.

{{< concept-alert type="advertencia" >}}
La asimetría estructural del problema: **quien genera solo necesita evadir a los detectores que ya existen; quien detecta tiene que anticipar generadores que aún no se han publicado.** Cada nuevo método de síntesis invalida parcialmente a los detectores entrenados.

De ahí que la dirección con más consenso hoy no sea detectar la falsificación sino **certificar la procedencia**: firmar criptográficamente el contenido en el momento de captura y mantener la cadena de custodia de las ediciones — el enfoque del estándar C2PA. Es un problema de infraestructura, no de clasificación.
{{< /concept-alert >}}

## Por qué importa para la Clase 44

La [Clase 44](/clases/clase-44) dedica seis diapositivas a las aplicaciones útiles de los deep fakes —cine, educación, traducción, prótesis de voz— y ninguna a su detección ni a los daños documentados. Es una omisión razonable dado el foco de la clase, que es técnico y termina en un laboratorio de generación, pero deja incompleto el panorama de una tecnología que la propia clase califica de *"altamente controversial"*.

Este paper es la mitad que falta, y en el mismo registro: sin moralizar, con un benchmark y números. Su lección más transferible para quien termina el diplomado es que **la detección es más difícil que la generación**, y que esa asimetría no se resuelve con un mejor clasificador.

---

**Ver también:** [First Order Motion Model (2019)](/papers/fomm-siarohin-2019) · [Síntesis de Medios](/fundamentos/sintesis-de-medios) · [MOT16 (2016)](/papers/mot16-milan-2016) — otro benchmark que ordenó su campo · [Clase 44](/clases/clase-44)
