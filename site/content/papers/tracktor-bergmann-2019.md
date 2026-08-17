---
title: "Tracktor: Tracking without bells and whistles (2019)"
weight: 452
math: true
---

{{< paper-card
    title="Tracking without bells and whistles"
    authors="Philipp Bergmann, Tim Meinhardt, Laura Leal-Taixé (TU München)"
    year="2019"
    venue="ICCV 2019 / arXiv:1903.05625"
    arxiv="1903.05625"
    pdf="/papers/tracktor-bergmann-2019.pdf" >}}
Un paper con la misma estructura argumental que [SORT](/papers/sort-bewley-2016) —quitar en vez de agregar— llevada al extremo: **no entrena nada sobre datos de seguimiento**. La observación es que la **cabeza de regresión** de un detector como Faster R-CNN ya sabe ajustar una caja aproximada a un objeto, así que puede usarse como modelo de movimiento: se toma la caja del frame $t-1$, se la alimenta al regresor sobre el frame $t$, y se obtiene la caja del mismo objeto en el frame nuevo — con su identidad intacta, gratis. Un detector convertido en tracker. Con ese mecanismo y dos extensiones simples (re-identificación y compensación de movimiento de cámara), obtuvo el estado del arte en tres benchmarks, y su análisis con oráculos mostró que **ningún método dedicado de seguimiento manejaba los casos difíciles mejor que él**.
{{< /paper-card >}}

---

## La idea

En *tracking-by-detection* clásico, el detector produce cajas y un módulo aparte las asocia. Tracktor observa que el detector de dos etapas ya contiene, en su cabeza de regresión, un mecanismo de **realineamiento temporal**: dada una caja de entrada, la refina hacia el objeto que contenga. Si la caja de entrada es la posición del objeto en el frame anterior y la imagen es la del frame actual, el refinamiento es exactamente una predicción de movimiento.

El procedimiento por frame:

1. Para cada trayectoria activa, pasar su caja $b^k_{t-1}$ por el regresor sobre el frame $t$ → nueva posición $b^k_t$, misma identidad. **No hay asociación que resolver.**
2. Matar la trayectoria si la cabeza de clasificación devuelve un score bajo, o si su solapamiento con otra trayectoria supera un umbral (supresión no máxima entre trayectorias).
3. Inicializar trayectorias nuevas con las detecciones del frame que no fueron cubiertas por ninguna trayectoria existente.

Esa es la figura de arquitectura del paper —con los bloques *Regression*, *Classification*, *Detection*, la pregunta *"Kill $b^k_t$?"* y el bloque *"Init new $b^k_t$"*— que la [Clase 42](/clases/clase-42) reproduce, aunque bajo el título de DeepSORT.

{{< concept-alert type="clave" >}}
Tracktor **no tiene modelo de movimiento explícito ni métrica de asociación**. No hay filtro de Kalman, no hay IoU, no hay algoritmo húngaro, no hay entrenamiento sobre secuencias. La continuidad de identidad la produce el hecho de que la caja de entrada al regresor *es* la caja anterior. Es la propuesta más minimalista de la literatura de MOT.
{{< /concept-alert >}}

## Las dos extensiones

El Tracktor base falla en lo que el regresor no puede hacer: saltos grandes y oclusiones largas. **Tracktor++** agrega dos parches convencionales:

- **Compensación de movimiento de cámara.** Se estima la transformación entre frames consecutivos y se aplica a las cajas antes de regresar. Sin esto, el regresor recibe cajas desplazadas por el movimiento del sensor y falla en secuencias con cámara móvil.
- **Re-identificación con red siamesa.** Las trayectorias muertas se guardan en una memoria de corto plazo, y una detección nueva puede reactivarlas si su descriptor de apariencia coincide.

Ambas contribuciones son, en palabras del propio paper, significativas — el método puro no basta.

## El análisis con oráculos

La segunda mitad del paper es lo que le da su peso, y es más un estudio del campo que una propuesta. Los autores reemplazan componentes de Tracktor por **oráculos** que consultan el *ground truth*, y miden cuánto rinde cada uno:

- Oráculos de detección (regresión perfecta, eliminación perfecta) dan mejoras sustanciales de MOTA y FP.
- El oráculo de **modelo de movimiento combinado con re-ID** —que sabe dónde está el objeto a través de oclusiones largas— es el que más aporta en preservación de identidad.

Y luego comparan a Tracktor con los métodos dedicados de seguimiento sobre los escenarios difíciles: objetos pequeños, muy ocluidos, detecciones faltantes. La conclusión es incómoda para el área:

> *"Sorprendentemente, ninguno de los métodos dedicados de seguimiento es considerablemente mejor manejando escenarios complejos de seguimiento."*

Es decir: lo que los métodos sofisticados estaban ganando no era robustez a la oclusión sino los casos fáciles, que Tracktor resuelve gratis. Los autores lo formulan como una pregunta abierta: *si un detector resuelve la mayor parte del problema de seguimiento, ¿cuáles son las situaciones reales donde hace falta un algoritmo de seguimiento dedicado?*

## Por qué importa para la Clase 42

Dos razones.

**Es la atribución correcta de un diagrama.** La [Clase 42](/clases/clase-42/teoria) muestra la figura de arquitectura de Tracktor dentro de la sección de DeepSORT. Son métodos con principios opuestos: DeepSORT resuelve la asociación con una métrica aprendida y el húngaro; Tracktor la elimina reutilizando el regresor. Confundirlos borra justamente lo interesante de cada uno.

**Es la tercera respuesta a la pregunta de la clase.** La clase termina la sección de DeepSORT preguntándose qué hacer cuando la cámara se mueve mucho o hay oclusiones largas, y ofrece dos caminos: mejorar el modelo de movimiento (compensación de cámara, zonas ciegas) y usar re-identificación. Tracktor agrega un tercero: **no modelar el movimiento en absoluto** y dejar que el detector haga ese trabajo. Que funcione tan bien es el resultado que hay que explicar.

---

**Ver también:** [SORT (2016)](/papers/sort-bewley-2016) · [DeepSORT (2017)](/papers/deepsort-wojke-2017) · [FairMOT (2020)](/papers/fairmot-zhang-2020) · [Faster R-CNN (2015)](/papers/faster-rcnn-ren-2015) · [Seguimiento de Objetos](/fundamentos/seguimiento-de-objetos)
