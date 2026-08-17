---
title: "Destilación de Conocimiento"
weight: 137
math: true
---

La **destilación de conocimiento** entrena un modelo (el *estudiante*) para imitar las **distribuciones de salida** de otro (el *maestro*), en vez de las etiquetas verdaderas. La idea, de [Hinton, Vinyals y Dean (2015)](/papers/distillation-hinton-2015), parte de una observación sencilla: cuando un clasificador dice *"90 % perro, 8 % lobo, 0,001 % auto"*, las dos cifras que no son el máximo contienen información —**el lobo se parece al perro y el auto no**— que la etiqueta `perro` destruye por completo.

Este fundamento acompaña a la [Clase 43](/clases/clase-43), donde la destilación aparece en su versión más ambiciosa: **cross-modal**, con un maestro que ve y un estudiante que oye.

---

## 1. Soft targets y la temperatura

El mecanismo es un softmax con un parámetro de escala:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

Con $T = 1$ es el softmax habitual. Al subir $T$ la distribución se **aplana** y las clases minoritarias ganan masa relativa. Con logits del maestro $[6{,}0,\; 2{,}0,\; 1{,}8,\; -1{,}0,\; -3{,}0]$ sobre las clases *perro, lobo, zorro, auto, silla*:

| $T$ | perro | lobo | zorro | auto | silla | entropía |
|---|---|---|---|---|---|---|
| 1 | 0,9668 | 0,0177 | 0,0145 | 0,0009 | 0,0001 | 0,173 nats |
| 2 | 0,7698 | 0,1042 | 0,0943 | 0,0232 | 0,0086 | 0,788 nats |
| 5 | 0,4361 | 0,1960 | 0,1883 | 0,1075 | 0,0721 | 1,425 nats |
| 10 | 0,3095 | 0,2075 | 0,2034 | 0,1537 | 0,1259 | 1,562 nats |

A $T = 1$ el maestro dice *perro* y poco más. A $T = 5$ se hace visible que **lobo y zorro están casi empatados entre sí y muy por encima de auto**: esa estructura relativa es la señal que el estudiante aprende y que las etiquetas duras no contienen. Hinton la llamó **dark knowledge**.

{{< concept-alert type="clave" >}}
La temperatura se usa **en las dos redes durante el entrenamiento** y se vuelve a 1 en inferencia. Y hay un detalle de escala que se olvida seguido: los gradientes de los *soft targets* escalan como $1/T^2$, así que al combinar la pérdida blanda con la dura hay que **multiplicar la primera por $T^2$**. Sin esa corrección, subir la temperatura equivale a bajar la tasa de aprendizaje de la rama de destilación.
{{< /concept-alert >}}

## 2. El teorema del límite de temperatura alta

El resultado teórico del paper: en el límite de $T$ grande, y si los logits están centrados en cero para cada ejemplo, el gradiente de la destilación se reduce a

$$\frac{\partial C}{\partial z_i} \;\approx\; \frac{1}{N T^2}\,(z_i - v_i)$$

que es exactamente el gradiente de $\tfrac{1}{2}(z_i - v_i)^2$: **la destilación a temperatura alta equivale a una regresión de mínimos cuadrados sobre los logits.**

Verificado numéricamente (coseno entre el gradiente exacto de la divergencia KL y el de la aproximación de mínimos cuadrados):

| $T$ | 1 | 2 | 5 | 10 | 25 | 100 | 1000 |
|---|---|---|---|---|---|---|---|
| coseno | 0,9557 | 0,9739 | 0,9924 | 0,9977 | 0,9996 | 0,99997 | 1,000000 |

A $T$ alta las dos pérdidas empujan en la misma dirección. A $T = 1$ **no**, y esa diferencia es la que decide resultados prácticos.

A temperaturas bajas, dice el paper, la destilación *"presta mucha menos atención a los logits mucho más negativos que el promedio"*, y eso es deseable: esos logits *"están casi completamente no restringidos por la función de costo con la que se entrenó el modelo grande, así que podrían ser muy ruidosos"*.

## 3. KL, L2 sobre logits y L2 sobre probabilidades no son lo mismo

Aquí hay una fuente de confusión con consecuencias medibles. "Usar L2" puede significar dos cosas distintas:

**L2 sobre logits** — $\lVert z - v\rVert^2$. No pasa por el softmax, no se satura, y es el límite de temperatura alta del teorema anterior. Funciona bien.

**L2 sobre probabilidades** — $\lVert \sigma(z) - \sigma(v)\rVert^2$. El gradiente tiene que atravesar el jacobiano del softmax:

$$\frac{\partial}{\partial z_k}\lVert q - p\rVert^2 = 2\,q_k\left[(q_k - p_k) - \sum_j q_j(q_j-p_j)\right]$$

El factor $q_k$ del frente es el problema: **donde el estudiante asigna probabilidad casi nula, el gradiente también es casi nulo**, aunque esa clase esté completamente equivocada. La pérdida no puede corregir lo que no ve.

Medido sobre un problema sintético de 400 clases con un maestro concentrado (la situación de un clasificador de ImageNet aplicado a fotogramas de video):

| Pérdida | top-1 | solape top-5 | correlación de rango |
|---|---|---|---|
| KL, $T=1$ | 67,87 % | 56,15 % | 0,8577 |
| KL, $T=2$ | 62,43 % | 59,85 % | 0,9385 |
| KL, $T=4$ | 57,10 % | 65,09 % | 0,9837 |
| **L2 sobre probabilidades** | **7,57 %** | 8,45 % | 0,1674 |
| L2 sobre logits | 92,57 % | 94,36 % | 0,9987 |

L2 sobre probabilidades colapsa; L2 sobre logits es la mejor de todas. Son la misma letra y objetivos distintos.

{{< concept-alert type="advertencia" >}}
Nótese también el intercambio dentro de la familia KL: al subir $T$, el **top-1 baja** (67,9 → 57,1) pero la **correlación de rango sube** (0,858 → 0,984). Una temperatura alta enseña la estructura relativa completa a costa de la decisión puntual. Cuál conviene depende de para qué se quiere al estudiante — y si se lo va a usar como **extractor de features**, como en SoundNet, la estructura importa más que el argmax.
{{< /concept-alert >}}

## 4. Destilación cross-modal

La variante que desarrolla la [Clase 43](/clases/clase-43). En la formulación original, maestro y estudiante ven **la misma entrada** y la destilación comprime: de un ensemble grande a un modelo desplegable.

En [SoundNet](/papers/soundnet-aytar-2016) el maestro recibe **fotogramas** y el estudiante **la forma de onda** del mismo video. Ya no se comprime nada: se **transfiere semántica de una modalidad a otra**, usando la sincronía natural del video como puente. El estudiante nunca ve una imagen, ni en entrenamiento ni en inferencia; solo ve el objetivo blando que la imagen produjo.

$$\min_\theta \sum_{k=1}^{K}\sum_{i=1}^{N} D_{\mathrm{KL}}\big(g_k(y_i)\,\|\,f_k(x_i;\theta)\big)$$

con $x_i$ la onda, $y_i$ el video, $g_k$ las redes visuales (objetos y escenas) y $f_k$ la red de sonido.

Lo que hace viable el esquema es que **el maestro no necesita etiquetas**: produce sus propios objetivos sobre datos sin anotar. Eso convierte un problema supervisado en uno que escala con la cantidad de video disponible — dos millones de clips, en el caso de SoundNet.

## 5. Cuándo funciona y cuándo no

**Funciona** cuando el maestro está bien calibrado en la región que importa, cuando hay muchos datos sin etiquetar del dominio de transferencia, y cuando existe correlación genuina entre lo que ve el maestro y lo que se quiere predecir.

**Falla o se degrada** cuando:

- el maestro tiene **ruido en la cola** de su distribución y la pérdida elegida lo toma en serio (la sección 3);
- **no hay correlación real** entre modalidades — un plano de una pared blanca con habla fuera de cuadro le enseña al estudiante una asociación falsa;
- el estudiante hereda los **sesgos y los errores sistemáticos** del maestro, incluidos los que el maestro nunca declaró.

Sobre este último punto, SoundNet ofrece una medida honesta: transferir de ImageNet **y** Places juntos rinde 72,9 % en ESC-50, contra 69,5 % con solo ImageNet y 71,1 % con solo Places. Más maestros, y más diversos, es mejor — porque el estudiante queda acotado por lo que sus maestros saben.

---

## Ver también

- [Hinton, Vinyals y Dean (2015)](/papers/distillation-hinton-2015) — el paper original.
- [SoundNet (2016)](/papers/soundnet-aytar-2016) — destilación cross-modal de visión a sonido.
- [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual) — la sincronía como fuente de supervisión.
- [Aprendizaje Autosupervisado](/fundamentos/aprendizaje-autosupervisado) — la familia mayor a la que esto pertenece.
- [Transfer Learning](/fundamentos/transfer-learning) — la alternativa clásica: copiar pesos en vez de imitar salidas.
- [Clase 43 — Práctica](/clases/clase-43/practica) — todo esto implementado y medido, en triple framework.
