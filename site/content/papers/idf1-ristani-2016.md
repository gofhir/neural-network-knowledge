---
title: "IDF1 y DukeMTMC: medidas de identidad para tracking (2016)"
weight: 450
math: true
---

{{< paper-card
    title="Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking"
    authors="Ergys Ristani, Francesco Solera, Roger S. Zou, Rita Cucchiara, Carlo Tomasi (Duke / Modena)"
    year="2016"
    venue="ECCV 2016 Workshops / arXiv:1609.01775"
    arxiv="1609.01775"
    pdf="/papers/idf1-ristani-2016.pdf" >}}
El paper que introdujo **IDF1**, la métrica que corrige el sesgo de MOTA hacia la detección. Su punto de partida es el seguimiento **multi-cámara**, donde la pregunta relevante no es cuántas cajas se acertaron sino si el sistema entendió que la persona de la cámara 8 es la misma que salió de la cámara 2. Para eso el emparejamiento entre predicción y verdad no puede hacerse frame a frame: hay que emparejar **trayectorias completas** con el [algoritmo húngaro](/fundamentos/asignacion-hungara) y contar aciertos de identidad. Aporta además DukeMTMC: más de 2 millones de frames a 1080p y 60 fps, 8 cámaras, 2700 identidades y 85 minutos de video sincronizado.
{{< /paper-card >}}

---

## El argumento

Las métricas CLEAR MOT, dicen los autores, miden *cuán bien un tracker determina la posición de los objetos*. Eso sirve para evaluar seguimiento dentro de una cámara. Pero en un sistema multi-cámara lo que se compra es otra cosa: saber **quién es quién** a lo largo de toda la red. Y esa capacidad no se refleja en un conteo de errores por frame.

El ejemplo canónico: un tracker que sigue perfectamente a una persona en la cámara 1, y al reaparecer en la cámara 5 le asigna un ID nuevo, tiene casi todos sus frames correctos y un solo error. MOTA lo penaliza con un IDSW entre miles de detecciones — es decir, casi nada. Para la aplicación, en cambio, el sistema falló en lo único que se le pedía.

## Cómo se calcula

El emparejamiento es **global y biyectivo entre trayectorias**, no entre detecciones por frame. El húngaro elige qué trayectoria predicha corresponde a cuál verdadera minimizando la suma de errores de identidad. De ahí salen tres conteos:

- **IDTP**: detecciones correctamente identificadas (en la parte solapada de trayectorias emparejadas).
- **IDFN**: detecciones verdaderas no cubiertas por la trayectoria predicha que se le asignó.
- **IDFP**: detecciones predichas que no corresponden a la trayectoria verdadera asignada.

Y las tres medidas:

$$\text{ID-Recall} = \frac{|\mathrm{IDTP}|}{|\mathrm{IDTP}| + |\mathrm{IDFN}|}, \qquad \text{ID-Precision} = \frac{|\mathrm{IDTP}|}{|\mathrm{IDTP}| + |\mathrm{IDFP}|}$$

$$\mathrm{IDF1} = \frac{|\mathrm{IDTP}|}{|\mathrm{IDTP}| + 0{,}5|\mathrm{IDFN}| + 0{,}5|\mathrm{IDFP}|}$$

que es el F1 clásico, con la particularidad de que los positivos se cuentan **por identidad correcta**, no por caja correcta.

{{< concept-alert type="clave" >}}
La diferencia de fondo con MOTA es *cuándo* se decide qué es un acierto. MOTA lo decide **localmente**: en cada frame, la caja predicha está cerca de una verdadera, luego es un TP, sin importar qué ID lleve. IDF1 lo decide **globalmente**: una detección solo cuenta si pertenece a la trayectoria que el emparejamiento global asignó a ese objeto. Un ID switch en el medio de una trayectoria larga invalida, para IDF1, toda la mitad que quedó del lado equivocado.
{{< /concept-alert >}}

## El defecto

[HOTA](/papers/hota-luiten-2020) documenta después el problema simétrico: IDF1 es **no monótona respecto de la detección**. Agregar detecciones correctas puede *bajar* IDF1, porque el emparejamiento global puede reasignar trayectorias y dejar del lado equivocado detecciones que antes contaban. Una métrica que empeora cuando el sistema mejora tiene un defecto estructural, no un caso raro.

Por eso el consenso actual no es reemplazar MOTA por IDF1 sino reportar ambas, o mejor, usar HOTA y su descomposición en DetA y AssA — que mide lo mismo que IDF1 intenta medir, pero sin ese comportamiento.

## DukeMTMC

El dataset que acompaña al paper fue, en su momento, el más grande y completo del área: 8 cámaras sincronizadas y calibradas, 1080p a 60 fps, 85 minutos, más de 2700 identidades, anotación completa. Sirvió durante años como referencia de seguimiento multi-cámara y re-identificación.

**Fue retirado en 2019** por preocupaciones de privacidad y consentimiento: el video se grabó en un campus sin consentimiento informado de las personas filmadas y terminó alimentando sistemas de vigilancia. El caso, junto al de otros datasets biométricos retirados en el mismo periodo, es una referencia obligada al discutir la ética de los datos en visión por computador — y particularmente pertinente en un dominio, la re-identificación de personas, cuya aplicación primaria es la vigilancia. La métrica IDF1 sobrevivió al dataset y se sigue usando.

## Por qué importa para la Clase 42

La [Clase 42](/clases/clase-42) dedica una diapositiva al escenario **multi-cámara** —el que originó esta métrica— y observa correctamente que ahí *"podemos seguir usando los features de la red siamesa/triplet network para conectar puntos"*: sin solapamiento entre campos de visión, la geometría no ayuda y solo queda la apariencia. Este paper es la formalización de ese escenario y de cómo evaluarlo.

---

**Ver también:** [HOTA (2020)](/papers/hota-luiten-2020) · [MOT16 (2016)](/papers/mot16-milan-2016) · [Métricas de Tracking](/fundamentos/metricas-de-tracking) · [Re-identificación](/fundamentos/re-identificacion)
