# Continual Learning Through Synaptic Intelligence — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Continual Learning Through Synaptic Intelligence*.
- **Autores:** Friedemann Zenke, Ben Poole (ambos con contribución equitativa) y Surya Ganguli, los tres de Stanford University. Correspondencia a Friedemann Zenke (`fzenke@stanford.edu`) y Ben Poole (`poole@cs.stanford.edu`).
- **Venue:** *Proceedings of the 34th International Conference on Machine Learning* (ICML 2017), Sydney, Australia, PMLR vol. 70.
- **Año:** 2017. **Preprint:** arXiv:1703.04200v3 (12 jun 2017), [arxiv.org/abs/1703.04200](https://arxiv.org/abs/1703.04200).
- **Apodo en la literatura:** "Synaptic Intelligence" (SI), por las *intelligent synapses* (sinapsis inteligentes) que el paper introduce. El método es también conocido por su parámetro de importancia online ω ("path integral").

**Nota sobre su lugar en la Clase 32.** Este es un paper *canónico* del subcampo de olvido catastrófico que **no aparece citado explícitamente en las diapositivas** de la clase. Lo incorporamos porque es el **hermano gemelo de EWC** (Elastic Weight Consolidation, Kirkpatrick et al., 2017, que sí se ve en clase): ambos salieron casi simultáneamente en 2017, ambos pertenecen a la familia de **regularización basada en la importancia de los pesos**, y SI es la respuesta directa a una limitación práctica de EWC. Entender uno sin el otro deja incompleta la imagen del enfoque de regularización contra el olvido. SI es, en una frase, *"EWC pero con la importancia estimada online a lo largo del entrenamiento, en vez de post-hoc en un punto fijo"*.

El problema de fondo es el **olvido catastrófico** (catastrophic forgetting): las redes neuronales artificiales se entrenan sobre un dataset en una fase de entrenamiento, congelan sus pesos y luego los usan estáticamente; cuando la distribución de datos cambia (aprendizaje continuo / continual learning), reentrenar sobre la tarea nueva sobrescribe lo aprendido en las tareas anteriores. Las redes biológicas, en cambio, aprenden de datos no estacionarios sin olvidar, y el paper conjetura que parte de ese poder reside en la **complejidad de las sinapsis biológicas**: mientras una sinapsis artificial es un único escalar (un peso), una sinapsis biológica es maquinaria molecular compleja. La contribución es traer algo de esa complejidad a las ANN dotando a cada sinapsis de un **estado tridimensional** (valor actual, valor de referencia pasado, e importancia acumulada) que estima online cuánto contribuyó ese peso a resolver tareas pasadas, para luego penalizar cambios a las sinapsis importantes.

## 2. Contexto histórico: olvido catastrófico y las tres familias de soluciones

El paper organiza el trabajo previo sobre olvido catastrófico en **tres familias**, taxonomía útil para situar SI dentro de la Clase 32:

1. **Enfoques arquitectónicos.** Alteran la *arquitectura* de la red para reducir la interferencia entre tareas sin tocar la función objetivo. Van desde congelar pesos (Razavian et al., 2014), bajar el learning rate de capas compartidas al hacer fine-tuning (Donahue et al., 2014; Yosinski et al., 2014), usar no-linealidades como ReLU/MaxOut/winner-take-all o inyectar ruido con dropout (Goodfellow et al., 2013), hasta las **Progressive Neural Networks** de Rusu et al. (2016), que copian la red entera de la tarea previa y la aumentan con features nuevas. Estas últimas previenen el olvido por completo, pero la complejidad arquitectónica **crece con el número de tareas** — su gran defecto.

2. **Enfoques funcionales.** Añaden un término de regularización que penaliza cambios en la *función entrada-salida* de la red. Li & Hoiem (2016, "Learning without Forgetting") usan **destilación de conocimiento** (Hinton et al., 2014) para que las predicciones de la red vieja y la nueva sean similares; Jung et al. (2016) regularizan la distancia ℓ₂ entre activaciones ocultas finales. El costo: requieren un **forward pass por la red de la tarea vieja para cada dato nuevo**, lo que los hace computacionalmente caros.

3. **Regularización estructural.** Penaliza directamente *los parámetros* para mantenerlos cerca de los valores de la tarea vieja. Aquí está **EWC** (Kirkpatrick et al., 2017): una penalización cuadrática sobre la diferencia entre los parámetros nuevos y viejos, ponderada por la diagonal de la **información de Fisher** sobre los parámetros viejos. El cuello de botella de EWC: calcular exactamente la diagonal de Fisher requiere sumar sobre todas las etiquetas de salida posibles (complejidad lineal en el número de salidas), lo que limita su aplicación a espacios de salida de baja dimensión; y se computa **offline, en una fase separada al final de cada tarea**.

SI pertenece a esta tercera familia, **junto a EWC**, pero ataca su limitación: en vez de estimar la importancia post-hoc en el punto final, la estima online durante todo el entrenamiento sin pases de backpropagation extra.

## 3. Contribución central: importancia online por integral de camino

La idea clave es dotar a cada sinapsis θ_k de una **medida local de importancia** ω_k^μ que refleja *cuánto crédito tiene ese parámetro en las mejoras del objetivo* a lo largo de la tarea μ. Tres propiedades que el paper subraya como diferenciadoras:

- **Online:** ω se acumula *durante* el entrenamiento, paso a paso, no en un punto fijo posterior.
- **Local:** se computa en cada sinapsis a partir de cantidades que ya están disponibles (gradiente y actualización del paso), **sin gradientes adicionales**.
- **A lo largo de la trayectoria:** integra información sobre todo el camino θ(t) en el espacio de parámetros, no sobre el endpoint.

Cuando una tarea termina, las sinapsis importantes se **consolidan**: se penaliza su cambio en tareas futuras. Así, el aprendizaje de tareas nuevas queda mediado principalmente por las sinapsis que fueron *poco* importantes para las tareas pasadas — dejando capacidad libre sin interferir con lo ya aprendido. Esta es la misma intuición geométrica que EWC (los parámetros influyentes se "jalan" más fuerte hacia su valor de referencia), pero con un origen distinto de la importancia.

## 4. El método en detalle

### 4.1. La integral de camino del gradiente

El entrenamiento es una trayectoria θ(t) en el espacio de parámetros. Para un paso infinitesimal δ(t) en el tiempo t, el cambio en la pérdida se aproxima por el gradiente g = ∂L/∂θ:

> L(θ(t) + δ(t)) − L(θ(t)) ≈ Σ_k g_k(t) δ_k(t)

es decir, cada cambio de parámetro δ_k(t) contribuye g_k(t)·δ_k(t) al cambio total de la pérdida. Para obtener el cambio acumulado sobre toda la trayectoria, se integra: la **integral de camino del campo de gradientes** a lo largo de θ(t), desde t₀ hasta t₁. Como el gradiente es un campo conservativo, esa integral iguala la diferencia de pérdida entre el punto final y el inicial, L(θ(t₁)) − L(θ(t₀)). El paso crucial es que esta integral se **descompone como suma sobre parámetros individuales**, y cada término por-parámetro se define como ω_k^μ (con un signo menos, porque interesa *decrecer* la pérdida).

**Cómputo en la práctica.** ω_k^μ se aproxima online como la **suma corriente del producto** del gradiente g_k(t) = ∂L/∂θ_k por la actualización del parámetro θ′_k(t) = ∂θ_k/∂t. Para descenso de gradiente por lotes con learning rate infinitesimal, ω_k^μ se interpreta directamente como la contribución por-parámetro al cambio total en la pérdida. Bajo SGD real (con ruido), la estimación de g_k es ruidosa y las importancias tienden a **sobreestimar** el valor verdadero de ω — lo que motiva ajustar el parámetro de fuerza c por debajo de 1.

### 4.2. La pérdida surrogate cuadrática

¿Cómo se explota ω para mejorar el aprendizaje continuo? El objetivo real es minimizar la pérdida sumada sobre **todas** las tareas, L = Σ_μ L^μ, pero solo se tiene acceso a la pérdida de la tarea actual L^μ. El olvido catastrófico aparece cuando minimizar L^μ aumenta inadvertidamente el costo de tareas previas L^ν (ν < μ).

La solución es modificar el costo con una **pérdida surrogate** cuadrática que aproxima las pérdidas de las tareas anteriores:

> L̃^μ = L^μ + c · Σ_k Ω_k^μ (θ̃_k − θ_k)²

donde:
- **c** es un parámetro de fuerza adimensional que negocia memorias viejas contra nuevas. Con la integral evaluada exactamente, c = 1 daría peso igual a viejo y nuevo; por el ruido de SGD, en la práctica c < 1.
- **θ̃_k = θ_k(t_{μ-1})** es el peso de referencia (el valor al final de la tarea anterior).
- **Ω_k^μ** es la **fuerza de regularización por-parámetro acumulada**, definida como Ω_k^μ = Σ_{ν<μ} ω_k^ν / ((Δ_k^ν)² + ξ).

Aquí Δ_k^ν ≡ θ_k(t_ν) − θ_k(t_{ν-1}) es **cuánto se movió** el parámetro durante la tarea ν. El denominador (Δ_k^ν)² cumple dos funciones: (1) asegura que el término de regularización tenga las **mismas unidades** que la pérdida L; y (2) normaliza la importancia por la distancia recorrida. **ξ** es un parámetro de amortiguamiento (damping) que acota la expresión cuando Δ_k^ν → 0.

**Protocolo de actualización.** Los ω_k se actualizan continuamente durante el entrenamiento. Las medidas acumuladas Ω_k^μ y los pesos de referencia θ̃ se actualizan **solo al final de cada tarea**; tras actualizar Ω_k^μ, los ω_k se ponen a cero. La motivación formal de la surrogate (que sea exacta) solo se sostiene para dos tareas, pero empíricamente funciona bien con muchas más.

### 4.3. La diferencia clave con EWC (Fisher vs. integral de camino)

Esta es la distinción central que justifica incluir el paper en la clase. Ambos métodos producen una penalización cuadrática que jala los parámetros influyentes hacia un peso de referencia. La diferencia está en **de dónde sale la importancia**:

- **EWC:** la importancia es la **diagonal de la información de Fisher**, calculada **offline en un punto** — el mínimo de la pérdida al final de cada tarea — en una fase separada. La Fisher requiere muestrear etiquetas del modelo y, para hacerlo online, al menos un pase extra de backpropagation. En la práctica suele reemplazarse por la **Fisher empírica** (Martens, 2016), que usa etiquetas del dataset y se calcula directo del gradiente, F̄ = E[g(θ)g(θ)ᵀ].
- **SI:** la importancia es la **integral de camino** ω, computada **online sobre toda la trayectoria** θ(t), sin gradientes adicionales.

El apéndice del paper hace explícita la relación: la diagonal de la Fisher empírica da una fórmula muy parecida a ω bajo descenso de gradiente, **pero la Fisher empírica se evalúa en un único valor de parámetro θ mientras la integral de camino se computa sobre la trayectoria**. La consecuencia es nítida: para una pérdida cuadrática, la Fisher empírica **en el mínimo es 0** (el gradiente se anula), mientras la integral de camino es proporcional a la diagonal de la Hessiana. Así, SI obtiene un estimador útil de la curvatura sin gradientes extra, justo donde la Fisher empírica colapsaría.

## 5. Análisis teórico: SI recupera la Hessiana

El paper dedica una sección a mostrar que, en un caso analíticamente tratable, ω se comporta sensatamente. Sobre una función de error cuadrática E(θ) = ½(θ − θ\*)ᵀ H (θ − θ\*) con mínimo en θ\* y Hessiana H, bajo dinámica de descenso de gradiente en tiempo continuo, se puede dar una solución exacta del camino de descenso. Computando ω como los elementos diagonales de una matriz Q = τ ∫ (dθ/dt)(dθ/dt)ᵀ dt:

- **Promediando sobre condiciones iniciales aleatorias** (con discrepancias d_α de media cero, varianza σ²), Q se reduce a ⟨Q_ij⟩ = ½σ²H_ij: **la Hessiana, salvo un factor de escala**. Notablemente, este factor σ² es exactamente lo que la normalización por (Δ_k)² en la ecuación de Ω elimina — lo que da motivación teórica a esa normalización.
- **Sin promediar**, hay dos casos donde la relación Q ↔ H se preserva: (a) Hessiana diagonal, donde la Q normalizada iguala la diagonal de la Hessiana; y (b) Hessiana de rango 1, donde Q se reduce de nuevo a la Hessiana salvo escala. El paper observa que la **Hessiana de bajo rango** es justamente el caso interesante para aprendizaje continuo: deja muchas direcciones del espacio de pesos sin restringir por una tarea, dejando capacidad libre para tareas futuras.

La advertencia honesta: esta correspondencia exacta vale para pérdidas cuadráticas (Hessiana constante a lo largo de la trayectoria). Para pérdidas generales, donde la Hessiana varía, **no** se puede esperar una correspondencia matemática simple entre Ω y la Hessiana en el endpoint ni con medidas relacionadas de sensibilidad (Pascanu & Bengio, 2013; Martens, 2016; Kirkpatrick et al., 2017). Empíricamente, sin embargo, la importancia de SI **correlaciona** con esas medidas basadas en el endpoint, lo que explica su eficacia comparable a EWC.

## 6. Experimentos

El método se evalúa en tres benchmarks de aprendizaje continuo de clasificación.

### 6.1. Split MNIST

Se divide MNIST en 5 subconjuntos de dígitos consecutivos: 5 tareas binarias (0/1, 2/3, 4/5, 6/7, 8/9). MLP pequeño con dos capas ocultas de 256 unidades ReLU, cross-entropy categórica más el término de consolidación (ξ = 1×10⁻³), enfoque **multi-head** (la pérdida del readout se computa solo para los dígitos de la tarea actual, evitando crosstalk en el readout). Adam (η = 1×10⁻³), minibatch 64, 10 épocas, optimizador reseteado entre tareas. Comparando consolidación activa (c = 1) contra apagada (c = 0): sin consolidación, tras ver todos los dígitos la precisión en las primeras dos tareas **cae a nivel de azar** (0.5 con readouts binarios); con consolidación, la degradación es mínima y la precisión promedio se mantiene cerca de 1. (Error bars = SEM, n = 10.)

### 6.2. Permuted MNIST

Cada tarea permuta aleatoriamente los píxeles de MNIST de forma distinta. MLP con dos capas ocultas de 2000 ReLU, softmax. Adam, ξ = 0.1, c = 0.1 (elegido por grid search en validación), minibatch 256, 20 épocas, **manteniendo** el estado de Adam entre tareas. Resultado: SGD y SGD+dropout sufren olvido catastrófico severo conforme crecen las tareas; SI (c > 0) **mantiene alta precisión sobre 10 tareas** y queda solo ligeramente por debajo de una red entrenada sobre todas las tareas simultáneamente. Crucialmente, los resultados son **comparables a EWC** (la curva de EWC se reextrajo de Kirkpatrick et al., 2017 y se superpone). El análisis de matrices de correlación de las importancias ω_k^μ muestra el mecanismo: sin consolidación, las importancias de la segunda capa se correlacionan entre tareas (causa probable del olvido); con consolidación, las sinapsis que reducen la pérdida quedan **descorrelacionadas entre tareas**, evitando interferencia.

### 6.3. Split CIFAR-10 / CIFAR-100

Para datasets más complejos y modelos más grandes: una CNN (4 capas convolucionales + 2 densas con dropout). Se entrena 60 épocas en CIFAR-10 completo (Task 1) y luego secuencialmente en 5 tareas más, cada una con 10 clases consecutivas de CIFAR-100. Multi-head, Adam, c en el rango [1×10⁻³, 0.1], n = 5 repeticiones. Resultados: las redes con consolidación muestran precisión de validación **similar entre todas las tareas**, mientras que sin consolidación hay un **declive dependiente de la edad** (las tareas viejas se resuelven peor). La consolidación es siempre mejor que su ausencia, salvo en la última tarea. Hallazgo adicional notable: comparada con redes entrenadas *desde cero* en una sola tarea, la red con consolidación iguala o **supera** la precisión de validación — sugiriendo que SI no solo protege memorias viejas sino que **mejora la generalización y reduce el overfitting** en tareas nuevas con pocos datos. En el experimento split CIFAR-10 del apéndice (Task A vs Task B), la consolidación dio una mejora pequeña pero significativa (≈4.5%) en validación incluso en la tarea *nueva*, evidencia de mejor **transfer learning**.

## 7. Limitaciones reconocidas

- **La justificación de la surrogate solo es exacta para dos tareas.** Para más de dos tareas el método se sostiene empíricamente, no por la derivación formal de la pérdida surrogate.
- **Sobreestimación bajo SGD.** El ruido de SGD hace que la integral de camino sobreestime las importancias verdaderas, obligando a fijar c < 1 y a un grid search del par (c, ξ) por benchmark — no hay receta universal de hiperparámetros.
- **Correspondencia teórica limitada a pérdidas cuadráticas.** Para pérdidas generales, la relación entre Ω y la curvatura del endpoint es solo empírica/correlacional, no garantizada.
- **Requiere estado adicional por sinapsis.** Cada peso pasa de ser un escalar a un sistema dinámico de mayor dimensión (valor actual, referencia θ̃, importancia ω/Ω), lo que aumenta el costo de memoria respecto a una red estándar.
- **Hiperparámetros de protocolo sensibles.** Detalles como resetear o no el estado del optimizador entre tareas cambian los resultados (se resetea en split MNIST/CIFAR, se mantiene en permuted MNIST), lo que añade fragilidad de implementación.

## 8. Impacto

Synaptic Intelligence se consolidó como una de las **dos referencias canónicas** del enfoque de regularización basada en importancia de pesos contra el olvido catastrófico, junto a EWC. Su aporte duradero es haber mostrado que la importancia de un parámetro puede estimarse **online, localmente y sin gradientes extra** acumulando su contribución a la reducción de la pérdida a lo largo de la trayectoria — una alternativa más barata y elegante a la Fisher post-hoc de EWC. La pareja EWC + SI define el eje "regularización" del aprendizaje continuo, frente a los ejes de *replay* (rehearsal/generative replay) y *expansión arquitectónica* (Progressive Networks). Métodos posteriores (Memory Aware Synapses, Riemannian Walk, online EWC) heredan directamente la idea de una penalización cuadrática ponderada por importancia, refinando *cómo* se estima esa importancia. El paper también dejó una agenda biológicamente inspirada: dotar a las sinapsis artificiales de dinámicas internas más ricas — "en machine learning, además de añadir profundidad a nuestras redes, quizá necesitemos añadir inteligencia a nuestras sinapsis".

## 9. Conexión con la Clase 32 (Olvido catastrófico)

Synaptic Intelligence es el **complemento directo de EWC** dentro del enfoque de **regularización basada en la importancia de los pesos**, uno de los tres grandes caminos que la clase contrasta para combatir el olvido catastrófico (regularización vs. replay vs. arquitectura dinámica).

La lección central para la clase es la **diferencia de origen de la importancia**, porque ambos métodos comparten exactamente la misma forma de penalización cuadrática Σ_k (importancia_k) · (θ_k − θ̃_k)²:

- **EWC** estima la importancia como la **diagonal de la información de Fisher**, calculada **post-hoc en un punto fijo** (el mínimo de la tarea, en una fase separada al final de cada tarea).
- **SI** estima la importancia **online a lo largo de toda la trayectoria de entrenamiento**, acumulando cuánto contribuyó cada peso a reducir la pérdida — sin pases de backpropagation adicionales.

Entender esta dualidad —misma penalización, distinta fuente de importancia (Fisher post-hoc vs. integral de camino online)— es la forma más económica de comprender *qué* significa "regularización contra el olvido" y *qué grados de libertad* tiene el diseñador. SI muestra que la importancia no tiene que medirse en el endpoint: el camino mismo del descenso ya contiene la señal.

**Enlaces internos del curso:**
- Fundamento transversal: [/fundamentos/aprendizaje-continuo](/fundamentos/aprendizaje-continuo)
- Clase: [/clases/clase-32](/clases/clase-32)
- Paper hermano (EWC): [/papers/ewc-kirkpatrick-2017](/papers/ewc-kirkpatrick-2017)
