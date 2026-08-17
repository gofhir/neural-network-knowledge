---
title: "Destilando el conocimiento de una red neuronal (2015)"
weight: 460
math: true
---

{{< paper-card
    title="Distilling the Knowledge in a Neural Network"
    authors="Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google)"
    year="2015"
    venue="NIPS 2014 Deep Learning Workshop / arXiv:1503.02531"
    arxiv="1503.02531"
    pdf="/papers/distillation-hinton-2015.pdf" >}}
Un modelo entrenado dice mucho más de lo que su predicción sugiere. Cuando un clasificador asigna a un dígito escrito *"2"* una probabilidad de $10^{-6}$ de ser un 3 y de $10^{-9}$ de ser un 7, esas dos cifras minúsculas codifican **cuánto se parece ese 2 a un 3 y a un 7** — una información que la etiqueta `2` destruye. Hinton, Vinyals y Dean proponen usarla: entrenar un modelo pequeño para reproducir las **distribuciones blandas** de uno grande, calentando el softmax con una temperatura $T$ para hacer visible esa estructura. Y demuestran que a temperatura alta el procedimiento equivale a una regresión de mínimos cuadrados sobre los logits.
{{< /paper-card >}}

---

## El mecanismo

Un softmax con temperatura:

$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Se genera con el modelo grande a temperatura alta un conjunto de objetivos blandos, se entrena al pequeño con **la misma temperatura alta** para reproducirlos, y en inferencia se vuelve a $T = 1$.

Cuando se conocen las etiquetas verdaderas, funciona mejor combinar dos objetivos: la entropía cruzada con los objetivos blandos (a temperatura $T$) y la entropía cruzada con las etiquetas duras (a $T=1$). Con una advertencia de escala que el paper subraya: los gradientes de la rama blanda escalan como $1/T^2$, así que **hay que multiplicarla por $T^2$** para que el balance entre ambas no dependa de la temperatura elegida.

{{< concept-alert type="clave" >}}
Los objetivos blandos tienen **más entropía por ejemplo** que las etiquetas duras: llevan más información y producen menos varianza en el gradiente. Por eso el modelo destilado puede entrenarse con **menos datos** y **tasas de aprendizaje más altas** que uno entrenado desde cero sobre las mismas etiquetas.
{{< /concept-alert >}}

## El teorema

La sección más citada del paper. El gradiente respecto de cada logit del estudiante es

$$\frac{\partial C}{\partial z_i} = \frac{1}{T}\left(\frac{e^{z_i/T}}{\sum_j e^{z_j/T}} - \frac{e^{v_i/T}}{\sum_j e^{v_j/T}}\right)$$

y si $T$ es grande comparada con la magnitud de los logits, y estos están **centrados en cero** para cada ejemplo, se simplifica a

$$\frac{\partial C}{\partial z_i} \;\approx\; \frac{1}{NT^2}(z_i - v_i)$$

que es el gradiente de $\tfrac{1}{2}(z_i - v_i)^2$. **En el límite de temperatura alta, destilar equivale a hacer coincidir los logits por mínimos cuadrados.**

La convergencia es rápida y verificable — coseno entre el gradiente exacto y la aproximación:

| $T$ | 1 | 2 | 5 | 10 | 25 | 100 | 1000 |
|---|---|---|---|---|---|---|---|
| coseno | 0,9557 | 0,9739 | 0,9924 | 0,9977 | 0,9996 | 0,99997 | 1,000000 |

Y el paper señala de inmediato la cara útil de que **no** sean equivalentes a temperatura baja: *"a temperaturas más bajas, la destilación presta mucha menos atención a hacer coincidir los logits que son mucho más negativos que el promedio. Esto es potencialmente ventajoso porque esos logits están casi completamente no restringidos por la función de costo con la que se entrenó el modelo grande, así que podrían ser muy ruidosos."*

Elegir $T$ es, entonces, elegir **cuánto ruido de la cola se le pide copiar al estudiante**.

## Otras contribuciones

Además de la compresión de ensembles, el paper propone **modelos especialistas**: para problemas con muchísimas clases, entrenar un modelo generalista más varios especialistas en subconjuntos confundibles, y usar la destilación para volver a integrar el conjunto. Los experimentos incluyen MNIST —donde un modelo destilado aprende a reconocer dígitos que **nunca vio** durante la transferencia, gracias a la estructura relativa de los objetivos blandos— y reconocimiento de voz a escala de Google.

## Por qué importa para la Clase 43

La [Clase 43](/clases/clase-43) no menciona este paper, pero [SoundNet](/papers/soundnet-aytar-2016) es literalmente una instancia suya, con una torsión: en la formulación original **maestro y estudiante ven la misma entrada** y la destilación *comprime*; en SoundNet el maestro ve fotogramas y el estudiante oye la onda, así que la destilación **transfiere entre modalidades**.

Y es lo que permite leer con precisión la ablación más llamativa de SoundNet, esos 25 puntos de diferencia entre la pérdida KL (72,9 %) y la $\ell_2$ (47,8 %) sobre ESC-50. Si Hinton demostró que ambas son equivalentes a temperatura alta, ¿cómo puede haber semejante brecha? Dos razones: SoundNet destila a $T=1$, lejos del límite del teorema; y su $\ell_2$ se aplica a las **probabilidades** de salida, no a los logits — y esa segunda pérdida, al pasar por el jacobiano del softmax, se satura donde el estudiante asigna probabilidad casi nula. Está desarrollado y medido en la [profundización](/clases/clase-43/profundizacion) y en la [práctica](/clases/clase-43/practica) de la clase.

---

**Ver también:** [SoundNet (2016)](/papers/soundnet-aytar-2016) · [Destilación de Conocimiento](/fundamentos/destilacion-de-conocimiento) · [Transfer Learning](/fundamentos/transfer-learning) · [Aprendizaje Autosupervisado](/fundamentos/aprendizaje-autosupervisado)
