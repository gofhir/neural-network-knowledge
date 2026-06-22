---
title: "Colorful Image Colorization (2016)"
weight: 312
math: true
---

{{< paper-card
    title="Colorful Image Colorization"
    authors="Richard Zhang, Phillip Isola, Alexei A. Efros"
    year="2016"
    venue="ECCV 2016"
    pdf="/papers/colorization-zhang-2016.pdf"
    arxiv="1603.08511" >}}
Trabajo de UC Berkeley que vuelve **canónica la colorización como pretext task** del [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado). Dada una foto en escala de grises (canal L del espacio Lab), la red **alucina los dos canales de color ab** sin necesidad de etiquetas humanas: cualquier foto a color es su propio ejemplo de entrenamiento. La clave técnica es tratar el color como **clasificación** sobre 313 bins cuantizados —no como regresión— con *class rebalancing* para rescatar colores raros y un *annealed-mean* para volver a un color puntual vibrante. Engaña a humanos en un Turing test el **32%** de las veces y, como representación transferida, alcanza **65.9 mAP** en clasificación PASCAL VOC.
{{< /paper-card >}}

---

## Contexto

Colorizar una fotografía en blanco y negro es un problema severamente **subdeterminado**: se han perdido dos de las tres dimensiones del color y hay que reconstruirlas. Los métodos previos o bien dependían de interacción manual del usuario (pintar trazos, elegir imágenes de referencia), o bien producían colorizaciones **desaturadas y apagadas**, con ese tono sepia característico. Zhang, Isola y Efros proponen un método **totalmente automático**, un único pase *feed-forward* de una CNN en test, entrenado sobre más de un millón de fotos a color de ImageNet.

La tesis central es **abrazar la incertidumbre** del problema en vez de evitarla. Muchos objetos admiten varias colorizaciones plausibles —una manzana puede ser roja, verde o amarilla, pero no azul—, así que el color es **inherentemente multimodal**. En 2016 el problema estaba "en el aire": Larsson et al. e Iizuka et al. publicaron sistemas concurrentes muy parecidos. Lo que distingue a este paper es su segunda contribución, la que lo ancla en la [Clase 28](/clases/clase-28): la colorización funciona como **pretext task** para aprender representaciones transferibles. Entrenando solo a colorear, sin una sola etiqueta semántica, la red aprende características que rinden a nivel estado del arte en benchmarks de *feature learning*. Los autores acuñan el término **cross-channel encoder** para este mecanismo, en la misma familia de tareas autosupervisadas que la [predicción de contexto de Doersch et al.](/papers/context-prediction-doersch-2015) y los [context encoders de Pathak et al.](/papers/context-encoders-pathak-2016).

## Por qué clasificar en vez de regresar el color

El sistema opera en el espacio **CIE Lab**, elegido porque las distancias en ese espacio aproximan la distancia perceptual. La entrada es el canal $L$ (luminosidad) y la salida son los canales $a$ y $b$ (cromaticidad). El enfoque ingenuo de los trabajos previos minimiza la **pérdida Euclidiana L2** entre el color predicho y el real:

$$L_2(\hat{Y}, Y) = \frac{1}{2}\sum_{h,w} \lVert Y_{h,w} - \hat{Y}_{h,w}\rVert_2^2$$

El problema fatal de esta pérdida es que **no es robusta a la ambigüedad multimodal**. Si un objeto admite varios valores ab plausibles, el óptimo de la pérdida Euclidiana es **el promedio del conjunto** —y promediar rojo con azul da un gris desaturado—. Peor aún: si el conjunto de colorizaciones plausibles es no convexo, el promedio cae fuera del conjunto y produce colores implausibles. Esta es la raíz exacta del aspecto sepia de los métodos previos.

La solución es reformular la predicción como **clasificación multinomial**. Se cuantiza el espacio ab con un grid de tamaño 10 y se conservan los $Q = 313$ bins que están *in-gamut* (colores realizables). La red aprende un mapeo $\hat{Z} = \mathcal{G}(X)$ a una **distribución de probabilidad** sobre esos 313 colores, por cada píxel. La verdad terreno se convierte a un vector $Z$ con **soft-encoding**: en lugar de codificar 1-hot al bin más cercano, se reparte la masa entre los 5 vecinos más cercanos, ponderados por un kernel Gaussiano ($\sigma = 5$). La pérdida es la entropía cruzada multinomial, con un término de reponderación $v(\cdot)$:

$$L_{cl}(\hat{Z}, Z) = -\sum_{h,w} v(Z_{h,w}) \sum_{q} Z_{h,w,q} \log(\hat{Z}_{h,w,q})$$

## Class rebalancing: rescatar los colores raros

La distribución de colores en imágenes naturales está **fuertemente sesgada hacia valores bajos (desaturados)**: nubes, pavimento, tierra y paredes inundan el dataset. Sobre 1.3M imágenes de ImageNet, los píxeles desaturados superan en órdenes de magnitud a los saturados. Sin corregirlo, la pérdida queda dominada por el gris y la red aprende a "jugar a lo seguro".

El remedio es **reponderar la pérdida de cada píxel según la rareza de su color**, asintóticamente equivalente a remuestrear el espacio de entrenamiento. Cada píxel se pondera por un factor basado en su bin ab más cercano:

$$w \propto \left((1-\lambda)\tilde{p} + \frac{\lambda}{Q}\right)^{-1}, \quad \mathbb{E}[w] = \sum_q \tilde{p}_q w_q = 1$$

Operativamente: se estima la distribución empírica $p$ de colores sobre todo ImageNet, se suaviza ($\sigma = 5$), se mezcla con una uniforme con peso $\lambda = \tfrac{1}{2}$, se toma el recíproco y se normaliza para que el factor valga 1 en esperanza. El efecto es **subir el peso de los colores saturados raros**, empujando a la red a explotar toda la diversidad cromática del dataset en vez de apostar siempre a colores frecuentes.

## Annealed-mean: de la distribución a un color puntual

Predecir una distribución por píxel está muy bien, pero al final hay que pintar **un** color. Los dos extremos son malos. Tomar el **modo** de la distribución da resultados vibrantes pero espacialmente inconsistentes (manchas de color que saltan). Tomar la **media** da resultados coherentes pero desaturados y sepia —no sorprende, porque promediar tras clasificar reintroduce el mismo defecto de la regresión L2—.

El paper interpola entre ambos reajustando la **temperatura $T$** de la distribución softmax antes de promediar, en una operación inspirada en el *simulated annealing* que llaman **annealed-mean**:

$$\mathcal{H}(Z_{h,w}) = \mathbb{E}[f_T(Z_{h,w})], \quad f_T(z) = \frac{\exp(\log(z)/T)}{\sum_q \exp(\log(z_q)/T)}$$

Con $T = 1$ queda la media simple; bajar $T$ vuelve la distribución más picuda; $T \to 0$ converge al modo. El valor elegido, $T = 0.38$, **captura la vibración del modo manteniendo la coherencia espacial de la media**. La operación es por píxel, con un único parámetro, y se integra en el pase feed-forward (aunque el sistema no es estrictamente *end-to-end* a través de ella).

La arquitectura es un solo flujo estilo VGG, sin pooling, con **convoluciones dilatadas** (atrous) cuya dilatación crece de conv1 a conv5 y decrece de conv6 a conv8, ampliando el campo receptivo sin perder resolución. Se entrena con ADAM por unas 450k iteraciones.

## Evaluación: el Turing test de colorización

El paper introduce un marco de evaluación novedoso, un **Turing test real vs. fake** en Amazon Mechanical Turk: a los participantes se les muestran pares (original color, colorización del modelo) y deben señalar cuál es falsa. La métrica es el porcentaje de veces que el método **engaña** al observador (el techo teórico de la verdad terreno es 50%).

| Método | % engaño (AMT) |
|---|---|
| Random (baseline) | — |
| Dahl (2016) | 18.3 |
| Variante L2 (regresión) | 21.2 |
| Clasificación sin rebalanceo | 25.2 |
| Larsson et al. (2016) | 27.2 |
| **Completo (clasificación + rebalanceo)** | **32.3** |

La variante completa engaña al **32.3%**, significativamente mejor que todas las comparadas ($p < 0.05$) salvo frente a Larsson ($p = 0.10$). Como control de competencia, en el 10% de los *trials* se enfrentó la verdad terreno contra una baseline aleatoria: los participantes la detectaron como falsa el 87% del tiempo, confirmando que entendían la tarea. Curiosamente, **en algunos casos el modelo engañó más del 50%** —los participantes prefirieron su colorización sobre la real—, a menudo porque la foto original tenía mal balance de blancos y el modelo produce una apariencia más prototípica.

Dos métricas complementarias confirman el resultado. La **interpretabilidad semántica**: alimentar las imágenes recolorizadas a una VGG entrenada en color recupera la precisión de clasificación de 52.7% (gris) a 56.0%, frente a 68.3% (color real); colorizar antes de clasificar mejora, sin reentrenar nada. La **exactitud cruda (AuC)** está dominada por píxeles desaturados —incluso predecir gris puntúa alto—, pero en su variante *balanceada por clase* el método completo supera a todas las demás, confirmando que el rebalanceo logra su efecto en las regiones perceptualmente interesantes. El paper es honesto sobre sus **modos de falla**: fallos de consistencia de largo alcance, confusiones recurrentes entre rojo y azul, y sepia por defecto en interiores complejos.

## Colorización como pretext task de SSL

Aquí la colorización se evalúa como **cross-channel encoder**: igual que un autoencoder, pero la entrada (L) y la salida (ab) son canales distintos de la imagen. Para comparar justo contra otros métodos de SSL, se reentrena una **AlexNet** en la tarea de colorización y se mide la calidad de sus características.

Con **clasificadores lineales sobre ImageNet** (pesos congelados, sin etiquetas semánticas), conv1 rinde peor que los competidores —por el *handicap* de la entrada en gris, ~6% constante a lo largo de la red—, pero la brecha se cierra de inmediato en conv2 y desde ahí compite con Doersch et al. y Donahue et al. Resolver colorización fomenta representaciones que **separan linealmente las clases semánticas**.

El resultado que la Clase 28 destaca es la **transferencia a PASCAL** vía fine-tuning, en dos modos: entrada en gris (`Ours (gray)`) y entrada Lab con pesos ab inicializados en cero (`Ours (color)`):

| Tarea (PASCAL) | Métrica | Ours (gray) | Ours (color) | Mejor previo |
|---|---|---|---|---|
| Clasificación (all layers) | %mAP | **65.9** | 65.6 | Doersch et al. 65.3 |
| Detección (all) | %mAP | 46.1 | 46.9 | Doersch et al. 51.1 |
| Segmentación (all) | %mIU | 35.0 | 35.6 | Donahue et al. 34.9 |

El **65.9 mAP** en clasificación es exactamente la fila `Ours (gray)` que aparece en la tabla de la clase: estado del arte entre los métodos de SSL probados, junto a un liderazgo en segmentación. En detección queda bajo Doersch et al. (51.1) pero sobre la fuerte baseline k-means (45.6). Como referencia, el pre-entrenamiento **supervisado** con ImageNet (techo) alcanza 78.9 / 56.8 / 48.0: todos los métodos de SSL quedan cortos frente a la supervisión plena, pero la colorización los lidera o iguala **sin usar una sola etiqueta**.

El Apéndice añade evidencia: la red aprende **distribuciones multimodales genuinas** (colores distintos para el fondo y el pájaro de la misma imagen), no explota solo claves de bajo nivel (distingue vegetales casi isoluminantes), y generaliza a fotos *legacy* reales en blanco y negro (Ansel Adams, el tilacino extinto de 1936).

## Limitaciones reconocibles

- **Modos de falla de color:** consistencia de largo alcance, confusión rojo/azul, sepia en interiores complejos.
- **AuC cruda poco discriminativa:** dominada por píxeles desaturados; solo su variante balanceada separa los métodos.
- **No es estrictamente *end-to-end*** a través del annealed-mean.
- **Handicap de entrada en gris:** ~6% constante frente a métodos que ven los tres canales RGB; la conv1 aprendida es comparativamente pobre.
- **Detección PASCAL:** claramente por debajo de la predicción de contexto de Doersch et al.
- **Ventaja no concluyente:** frente a Larsson et al. el rebalanceo no logra superioridad estadísticamente significativa en el Turing test.

## Por qué importa para la Clase 28

La [Clase 28](/clases/clase-28) presenta este trabajo como el **ejemplo paradigmático de auto-predicción en imágenes**: "de grises generar color" como tarea que no requiere etiquetas humanas. Esa intuición es exactamente el *cross-channel encoder*: el canal L es la entrada y los canales ab son su propia señal de supervisión, gratis, sobre cualquier foto a color. El paper materializa los tres pilares que la clase enseña sobre el [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado): (1) la supervisión emerge de la estructura de los propios datos; (2) resolver una tarea auxiliar plausible obliga a aprender semántica de alto nivel; y (3) esa representación transfiere a tareas reales posteriores.

Su valor doble —ser un sistema de colorización de referencia *y* un pretext task competitivo— lo instaló en todas las tablas comparativas de SSL que vinieron después, y su **65.9 en PASCAL VOC** quedó como punto de referencia obligado junto a otras pretext tasks como la [predicción de contexto](/papers/context-prediction-doersch-2015) y los [context encoders](/papers/context-encoders-pathak-2016). El concepto de cross-channel encoder y la idea de "auto-predecir una parte de los datos a partir de otra" anticiparon directamente la era del *contrastive* y *masked* representation learning (SimCLR, MoCo, MAE) que dominaría la visión desde 2019. La reformulación **regresión → clasificación + rebalanceo de clases raras** y el truco del *annealed-mean* por temperatura son lecciones de diseño de pérdidas que trascienden la colorización: reaparecen siempre que una tarea es intrínsecamente multimodal y una L2 ingenua colapsaría al promedio.

## Notas y enlaces

- Sitio, código (Caffe) y demo: http://richzhang.github.io/colorization/
- Preprint: [arxiv.org/abs/1603.08511](https://arxiv.org/abs/1603.08511) (v5, octubre 2016).
- Afiliación: University of California, Berkeley (Berkeley Vision Lab / EECS).
