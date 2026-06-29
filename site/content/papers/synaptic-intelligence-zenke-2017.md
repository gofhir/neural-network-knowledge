---
title: "Synaptic Intelligence (2017)"
weight: 360
math: true
---

{{< paper-card
    title="Continual Learning Through Synaptic Intelligence"
    authors="Friedemann Zenke, Ben Poole, Surya Ganguli"
    year="2017"
    venue="ICML 2017"
    pdf="/papers/synaptic-intelligence-zenke-2017.pdf"
    arxiv="1703.04200" >}}
Paper de Stanford que introduce **Synaptic Intelligence (SI)**, el **hermano gemelo de EWC** dentro de la familia de regularización contra el [olvido catastrófico](/fundamentos/aprendizaje-continuo). Su idea central: estimar la **importancia de cada peso online**, durante el entrenamiento, acumulando la **integral de camino** de su gradiente (ω) — en vez de calcular la información de Fisher post-hoc al final de cada tarea como hace [EWC](/papers/ewc-kirkpatrick-2017). Con esa importancia construye una **pérdida de regularización cuadrática** que jala los pesos influyentes hacia su valor de referencia. Validado en split y permuted MNIST con resultados comparables a EWC.
{{< /paper-card >}}

> **Nota.** Este es un paper *canónico* del olvido catastrófico que **no aparece citado explícitamente en las diapositivas** de la [Clase 32](/clases/clase-32). Lo incorporamos porque es la pareja teórica de EWC: entender uno sin el otro deja incompleta la imagen del enfoque de regularización. En una frase, SI es *"EWC pero con la importancia estimada online a lo largo del entrenamiento, en vez de post-hoc en un punto fijo"*.

---

## Contexto: olvido catastrófico

Las redes neuronales artificiales se entrenan sobre un dataset, congelan sus pesos y los usan estáticamente. Cuando la distribución de datos cambia ([aprendizaje continuo](/fundamentos/aprendizaje-continuo)), reentrenar sobre la tarea nueva **sobrescribe lo aprendido** en las tareas anteriores: eso es el olvido catastrófico (*catastrophic forgetting*). Las redes biológicas, en cambio, aprenden de datos no estacionarios sin olvidar.

El paper conjetura que parte de ese poder reside en la **complejidad de las sinapsis biológicas**: mientras una sinapsis artificial es un único escalar (un peso), una biológica es maquinaria molecular compleja. La propuesta es traer algo de esa complejidad a las redes dotando a cada sinapsis de un **estado tridimensional** —valor actual, valor de referencia pasado, e importancia acumulada— que estima online cuánto contribuyó ese peso a resolver tareas pasadas, para luego penalizar cambios a las sinapsis importantes.

El paper organiza el trabajo previo en **tres familias** de soluciones, taxonomía útil para situar SI:

1. **Enfoques arquitectónicos.** Alteran la arquitectura para reducir interferencia: congelar pesos, bajar el learning rate de capas compartidas, dropout, o las **Progressive Neural Networks** (Rusu et al., 2016), que copian la red entera y la aumentan con features nuevas. Defecto: la complejidad **crece con el número de tareas**.
2. **Enfoques funcionales.** Penalizan cambios en la función entrada-salida vía destilación de conocimiento (Li & Hoiem, 2016) o distancia ℓ₂ entre activaciones. Costo: requieren un **forward pass por la red vieja para cada dato nuevo**.
3. **Regularización estructural.** Penaliza directamente los parámetros para mantenerlos cerca de los valores de la tarea vieja. Aquí está **EWC** —y aquí está **SI**.

SI pertenece a la tercera familia, **junto a EWC**, pero ataca su limitación principal: EWC calcula la diagonal de la información de Fisher **offline, en una fase separada al final de cada tarea**, sumando sobre todas las etiquetas de salida posibles (lo que limita su uso a espacios de salida pequeños). SI estima la importancia online durante todo el entrenamiento, sin pases de backpropagation extra.

## Contribución central: importancia online por integral de camino

La idea clave es dotar a cada sinapsis $\theta_k$ de una **medida local de importancia** $\omega_k^\mu$ que refleja cuánto crédito tiene ese parámetro en las mejoras del objetivo a lo largo de la tarea $\mu$. Tres propiedades la diferencian de la Fisher de EWC:

- **Online:** ω se acumula *durante* el entrenamiento, paso a paso, no en un punto fijo posterior.
- **Local:** se computa en cada sinapsis a partir de cantidades ya disponibles (gradiente y actualización del paso), **sin gradientes adicionales**.
- **A lo largo de la trayectoria:** integra información sobre todo el camino $\theta(t)$ en el espacio de parámetros, no sobre el endpoint.

Cuando una tarea termina, las sinapsis importantes se **consolidan**: se penaliza su cambio en tareas futuras. Así, el aprendizaje de tareas nuevas queda mediado por las sinapsis que fueron *poco* importantes para las pasadas, dejando capacidad libre sin interferir con lo ya aprendido.

### La integral de camino del gradiente

El entrenamiento es una trayectoria $\theta(t)$ en el espacio de parámetros. Para un paso infinitesimal, el cambio en la pérdida se aproxima por el gradiente, de modo que cada cambio de parámetro $\delta_k(t)$ contribuye $g_k(t)\,\delta_k(t)$ al cambio total. Integrando sobre toda la trayectoria se obtiene la **integral de camino del campo de gradientes**, que (al ser el gradiente un campo conservativo) iguala la diferencia de pérdida entre el punto final y el inicial. Esa integral se **descompone como suma sobre parámetros individuales**, y cada término por-parámetro define $\omega_k^\mu$.

En la práctica, $\omega_k^\mu$ se aproxima online como la **suma corriente del producto** del gradiente $g_k(t) = \partial L/\partial\theta_k$ por la actualización del parámetro $\theta'_k(t)$. Bajo SGD real (con ruido), la estimación es ruidosa y las importancias tienden a **sobreestimar** el valor verdadero de ω, lo que motiva fijar el parámetro de fuerza $c < 1$.

### La pérdida surrogate cuadrática

El objetivo real es minimizar la pérdida sumada sobre **todas** las tareas, pero solo se tiene acceso a la pérdida de la tarea actual $L^\mu$. La solución es modificar el costo con una **pérdida surrogate** cuadrática que aproxima las pérdidas anteriores:

$$\tilde{L}^\mu = L^\mu + c \sum_k \Omega_k^\mu \,(\tilde{\theta}_k - \theta_k)^2$$

donde **$c$** es un parámetro de fuerza que negocia memorias viejas contra nuevas; **$\tilde{\theta}_k$** es el peso de referencia (valor al final de la tarea anterior); y **$\Omega_k^\mu$** es la fuerza de regularización acumulada,

$$\Omega_k^\mu = \sum_{\nu<\mu} \frac{\omega_k^\nu}{(\Delta_k^\nu)^2 + \xi}$$

con $\Delta_k^\nu$ = cuánto se movió el parámetro durante la tarea $\nu$. El denominador asegura que el término tenga las **mismas unidades** que la pérdida y normaliza por la distancia recorrida; **$\xi$** es un parámetro de amortiguamiento que acota la expresión cuando $\Delta_k^\nu \to 0$. Los $\omega_k$ se actualizan continuamente; las medidas $\Omega_k^\mu$ y los pesos de referencia $\tilde\theta$ se actualizan **solo al final de cada tarea**, y tras ello los $\omega_k$ se ponen a cero.

### La diferencia clave con EWC

Ambos métodos producen exactamente la **misma forma de penalización cuadrática** que jala los parámetros influyentes hacia un peso de referencia. La diferencia está en **de dónde sale la importancia**:

- **EWC:** la importancia es la **diagonal de la información de Fisher**, calculada **offline en un punto** —el mínimo de la pérdida al final de cada tarea— en una fase separada.
- **SI:** la importancia es la **integral de camino** ω, computada **online sobre toda la trayectoria** $\theta(t)$, sin gradientes adicionales.

El apéndice hace explícita la relación: para una pérdida cuadrática, la Fisher empírica **en el mínimo es 0** (el gradiente se anula), mientras la integral de camino es proporcional a la diagonal de la Hessiana. SI obtiene así un estimador útil de la curvatura justo donde la Fisher empírica colapsaría.

## Análisis teórico: SI recupera la Hessiana

Sobre una función de error cuadrática con Hessiana $H$, bajo descenso de gradiente en tiempo continuo, el paper muestra que ω se comporta sensatamente. **Promediando sobre condiciones iniciales aleatorias** (varianza $\sigma^2$), la matriz de importancia se reduce a $\tfrac{1}{2}\sigma^2 H$: **la Hessiana, salvo un factor de escala** —y ese factor $\sigma^2$ es exactamente lo que la normalización por $(\Delta_k)^2$ elimina, dando motivación teórica a esa normalización. Sin promediar, la relación se preserva en dos casos: Hessiana diagonal y Hessiana de rango 1. El paper observa que la **Hessiana de bajo rango** es justo el caso interesante: deja muchas direcciones del espacio de pesos sin restringir, dejando capacidad libre para tareas futuras.

La advertencia honesta: esta correspondencia exacta vale solo para pérdidas cuadráticas. Para pérdidas generales, la importancia de SI **correlaciona** empíricamente con las medidas basadas en el endpoint, lo que explica su eficacia comparable a EWC sin garantía matemática.

## Experimentos

**Split MNIST.** Se divide MNIST en 5 tareas binarias (0/1, 2/3, …, 8/9). MLP de dos capas ocultas de 256 ReLU, enfoque multi-head. Sin consolidación ($c=0$), tras ver todos los dígitos la precisión en las primeras tareas **cae a nivel de azar** (0.5); con consolidación ($c=1$), la degradación es mínima y la precisión promedio se mantiene cerca de 1.

**Permuted MNIST.** Cada tarea permuta aleatoriamente los píxeles. MLP de dos capas de 2000 ReLU, softmax. SGD y SGD+dropout sufren olvido severo; SI **mantiene alta precisión sobre 10 tareas**, solo ligeramente por debajo de entrenar sobre todas simultáneamente, y con resultados **comparables a EWC** (cuya curva se superpone). El análisis de correlación muestra el mecanismo: con consolidación, las sinapsis que reducen la pérdida quedan **descorrelacionadas entre tareas**, evitando interferencia.

**Split CIFAR-10 / CIFAR-100.** Una CNN entrenada en CIFAR-10 y luego secuencialmente en 5 tareas de 10 clases de CIFAR-100. Las redes con consolidación muestran precisión **similar entre tareas**, mientras que sin ella hay un declive dependiente de la edad. Hallazgo notable: la red con consolidación iguala o **supera** la precisión de redes entrenadas desde cero en una sola tarea, sugiriendo que SI no solo protege memorias viejas sino que **mejora la generalización** en tareas nuevas con pocos datos.

## Limitaciones reconocidas

- **La justificación de la surrogate solo es exacta para dos tareas;** para más, el método se sostiene empíricamente.
- **Sobreestimación bajo SGD:** el ruido obliga a fijar $c < 1$ y a un grid search del par $(c, \xi)$ por benchmark —no hay receta universal de hiperparámetros.
- **Correspondencia teórica limitada a pérdidas cuadráticas;** para pérdidas generales la relación con la curvatura es solo correlacional.
- **Requiere estado adicional por sinapsis** (valor actual, referencia, importancia), aumentando el costo de memoria.
- **Detalles de protocolo sensibles,** como resetear o no el estado del optimizador entre tareas, añaden fragilidad de implementación.

## Por qué importa para la Clase 32

Synaptic Intelligence es el **complemento directo de [EWC](/papers/ewc-kirkpatrick-2017)** dentro del enfoque de **regularización basada en la importancia de los pesos**, uno de los tres grandes caminos que la [Clase 32](/clases/clase-32) contrasta contra el olvido catastrófico (regularización vs. replay vs. arquitectura dinámica).

La lección central es la **diferencia de origen de la importancia**, porque ambos métodos comparten la misma penalización cuadrática $\sum_k (\text{importancia}_k)\,(\theta_k - \tilde\theta_k)^2$:

- **EWC** estima la importancia como la diagonal de la Fisher, **post-hoc en un punto fijo**.
- **SI** la estima **online a lo largo de toda la trayectoria** de entrenamiento, sin backpropagation adicional.

Entender esta dualidad —misma penalización, distinta fuente de importancia— es la forma más económica de comprender qué significa "regularización contra el olvido" y qué grados de libertad tiene el diseñador. SI muestra que la importancia no tiene que medirse en el endpoint: el camino mismo del descenso ya contiene la señal.

## Notas y enlaces

- Preprint: [arXiv:1703.04200](https://arxiv.org/abs/1703.04200) (12 jun 2017).
- Venue: *Proceedings of the 34th ICML* (2017), Sydney, PMLR vol. 70.
- Afiliación: Stanford University.
- Fundamento transversal: [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo).
- Paper hermano: [/papers/ewc-kirkpatrick-2017](/papers/ewc-kirkpatrick-2017).
