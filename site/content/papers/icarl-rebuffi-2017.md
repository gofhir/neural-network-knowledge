---
title: "iCaRL: Incremental Classifier and Representation Learning (2017)"
weight: 359
math: true
---

{{< paper-card
    title="iCaRL: Incremental Classifier and Representation Learning"
    authors="Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert"
    year="2017"
    venue="CVPR 2017"
    pdf="/papers/icarl-rebuffi-2017.pdf"
    arxiv="1611.07725" >}}
El método clásico de **aprendizaje incremental por clases** (*class-incremental learning*): aprende clasificadores **y** representación de forma simultánea e incremental, recibiendo las clases en lotes sucesivos sin acceder nunca a todos los datos a la vez. Su tesis es que ninguna idea sola basta y que hacen falta **tres componentes combinados**: un conjunto de *exemplars* por clase (seleccionados por *herding*, con presupuesto de memoria fijo), una pérdida que mezcla **clasificación + destilación** (estilo LwF), y una regla de clasificación por **nearest-mean-of-exemplars** (NME) robusta al desbalance y a la deriva de la representación. No aparece citado en las slides de la [Clase 32](/clases/clase-32), pero es el **baseline de facto** del área y la pieza que une las dos familias que la clase enseña por separado: memoria y regularización.
{{< /paper-card >}}

---

## Por qué lo incluimos

iCaRL **no aparece en las slides** de la clase de olvido catastrófico, pero es —sin exageración— el trabajo que en 2017 definió rigurosamente el problema de aprendizaje incremental por clases, propuso el protocolo de evaluación que todavía se usa, y se volvió el **baseline contra el que se compara casi todo método posterior**. Lo traemos porque es la pieza que **fusiona las dos grandes familias que la Clase 32 presenta por separado** —memoria/rehearsal y regularización/distillation— en un solo sistema coherente, y porque es el sucesor natural de [LwF (Li & Hoiem, 2016)](/papers/lwf-li-2016) llevado al escenario multiclase verdadero.

## El problema: aprendizaje incremental por clases

Los sistemas de visión naturales son incrementales (un niño aprende animales nuevos en el zoológico sin olvidar su mascota); los artificiales casi siempre se entrenan en modo *batch*, con todas las clases y datos disponibles a la vez. El paper formaliza el escenario *class-incremental* con **tres propiedades exigibles**:

1. Ser entrenable desde un **flujo de datos** donde clases distintas aparecen en momentos distintos.
2. Proveer **en todo momento** un clasificador multiclase competitivo para todas las clases vistas.
3. Mantener cómputo y memoria **acotados o de crecimiento muy lento** respecto al número de clases.

La tercera condición descarta la solución trivial de "guardar todo y reentrenar desde cero". En 2017 no existía un solo algoritmo satisfactorio: la mayoría de las técnicas multiclase violaban (i) o (ii), y entrenar con SGD ingenuo sobre el flujo degrada la *accuracy* de inmediato —el **olvido catastrófico** descrito desde McCloskey & Cohen (1989). Las pocas técnicas que cumplían las tres propiedades estaban limitadas a **representaciones de características fijas** (como el clasificador *nearest-class-mean* de Mensink et al., 2012) y no podían aprender clasificador y representación juntos en redes profundas.

## Contribución: tres componentes que se combinan

El aporte es la **integración**: por separado ninguno basta.

1. **Clasificación por nearest-mean-of-exemplars (NME).** Se abandona el softmax. La predicción compara el *feature* de la imagen con el prototipo (vector medio) de cada clase, calculado sobre sus exemplars almacenados.
2. **Selección priorizada de exemplars por *herding*.** Por cada clase se guarda un subconjunto pequeño de imágenes, elegidas iterativamente para que su media de *features* aproxime lo mejor posible la media de la clase completa.
3. **Aprendizaje de representación con destilación + clasificación.** Al llegar clases nuevas, la red se actualiza minimizando una pérdida que combina un término de **clasificación** (clases nuevas) con uno de **destilación** (preservar el conocimiento de las viejas), al estilo de Learning without Forgetting.

La novedad arquitectónica es **desacoplar el rol de la red del rol del clasificador**: la red profunda se usa solo como extractor de características $\varphi: \mathcal{X} \to \mathbb{R}^d$, no para clasificar. ¿Por qué importa? En un clasificador lineal estándar la predicción equivale a $y^* = \arg\max_y w_y^\top \varphi(x)$, con los pesos $w_y$ **desacoplados** de $\varphi$. Cada vez que $\varphi$ se ajusta para acomodar clases nuevas, *todos* los $w_y$ deberían reajustarse —pero en el flujo incremental no hay forma de hacerlo sin los datos viejos— y las salidas cambian de modo incontrolado: eso *es* el olvido catastrófico. NME elimina la fuente de inestabilidad: como el prototipo $\mu_y$ es función directa de $\varphi$, al moverse la representación los prototipos se mueven con ella de forma coherente, sin pesos rezagados. Por eso iCaRL puede cambiar $\varphi$ agresivamente entre lotes sin que el clasificador se desmorone.

## Método

### Clasificación: nearest-mean-of-exemplars

El prototipo de cada clase es la media de los *features* de sus exemplars:

$$\mu_y = \frac{1}{|P_y|} \sum_{p \in P_y} \varphi(p)$$

y se asigna la clase del prototipo más cercano:

$$y^* = \arg\min_{y=1,\dots,t} \| \varphi(x) - \mu_y \|$$

Todos los vectores van **L2-normalizados**, de modo que la regla equivale a $y^* = \arg\max_y \mu_y^\top \varphi(x)$: una clasificación con vector de peso que **no está desacoplado** de la representación. Se inspira en NCM, con la diferencia clave de que iCaRL no puede recalcular la media verdadera de la clase (exigiría todos los datos), así que la aproxima con la media sobre exemplars escogidos.

### Aprendizaje de representación

Al llegar datos de clases nuevas, iCaRL: (1) **construye un set aumentado** $D$ que une los ejemplos nuevos con los exemplars viejos —guardados como **imágenes**, no como *features*, que quedan obsoletos al cambiar $\varphi$—; (2) **almacena las salidas pre-actualización** de la red para las clases viejas (los *soft targets* de destilación); y (3) minimiza una pérdida que combina clasificación y destilación:

$$\ell(\Theta) = -\sum_{(x_i,y_i)\in D}\left[\sum_{y=s}^{t} \delta_{y=y_i}\log g_y(x_i) + \delta_{y\neq y_i}\log(1-g_y(x_i)) + \sum_{y=1}^{s-1} q_i^y \log g_y(x_i) + (1-q_i^y)\log(1-g_y(x_i))\right]$$

El primer bloque (clases $s,\dots,t$) es el **término de clasificación**; el segundo (clases $1,\dots,s-1$) es la **destilación**, que reproduce los *scores* sigmoidales previos para preservar la información discriminativa. Las salidas usan **cross-entropy binaria** por nodo sigmoidal (no softmax). La diferencia con LwF está en la predicción: LwF fue diseñado para multi-*task* (un clasificador por dataset), mientras iCaRL ataca el caso **multiclase verdadero**, donde un único clasificador elige entre todas las clases vistas.

### Gestión de exemplars: herding y reducción

iCaRL respeta un **presupuesto de memoria fijo** $K$: con $t$ clases vistas, asigna $m = K/t$ exemplars por clase.

- **Construcción por *herding*.** Los exemplars se seleccionan iterativamente: en cada paso se añade el ejemplo que hace que la media de *features* sobre los elegidos mejor aproxime la media de la clase. El resultado es una **lista priorizada** —el orden importa—. *Herding* (Welling, 2009) aproxima bien con menos muestras que el submuestreo aleatorio.
- **Reducción.** Cuando bajan los recursos por clase, reducir de $m'$ a $m$ es trivial: se **descartan los últimos** y se conservan los primeros, porque cualquier prefijo de una lista priorizada sigue siendo buena aproximación. La remoción es **independiente de los datos** —encaja con el escenario incremental, donde los datos viejos ya no están.

## Experimentos

El protocolo —fijar un orden aleatorio de clases, entrenar en lotes y evaluar tras cada uno, resumiendo en la **average incremental accuracy**— se volvió estándar.

- **iCIFAR-100:** CIFAR-100 en lotes de 2, 5, 10, 20 o 50 clases. ResNet de 32 capas, $K = 2000$ exemplars, promediado sobre 10 órdenes de clases.
- **iILSVRC:** ImageNet en variante *small* (100 clases, lotes de 10) y *full* (1000 clases, lotes de 100). ResNet de 18 capas, $K = 20000$, medida top-5.

**Frente a baselines.** Se compara contra *finetuning* (sin medida anti-olvido), *fixed representation* (congela la representación tras el primer lote) y **LwF.MC** (versión multiclase de LwF: destilación **sin exemplars**). **iCaRL supera claramente a todos**, y la ventaja crece cuanto más incremental es el escenario (lotes más pequeños). LwF.MC es siempre segundo; *finetuning* siempre el peor. Como referencia, la misma red entrenada en *batch* con todos los datos alcanza 68.6%.

**Matrices de confusión — el hallazgo más ilustrativo.** Revelan el sesgo de cada método: iCaRL produce una matriz **homogénea** (sin sesgo, no sufre olvido); LwF.MC predice clases de **lotes recientes**; la representación fija prefiere el **primer lote**; y *finetuning* predice **exclusivamente** el último lote —"olvidó que las clases anteriores existen".

**Análisis diferencial.** Tres híbridos aíslan el aporte de cada componente y caen entre iCaRL y LwF.MC, confirmando que **los tres contribuyen**. Hallazgos finos: la regla mean-of-exemplars luce más en lotes pequeños; con lotes muy pequeños la destilación puede incluso **perjudicar**; la diferencia entre NME y recalcular el NCM verdadero es **mínima** (el *herding* identifica exemplars representativos); y con $\geq 1000$ prototipos NME iguala a NCM. Más presupuesto $K$ siempre ayuda, con retornos decrecientes.

## Limitaciones reconocidas

- **Necesita almacenar exemplars.** El componente que más aporta rompe la pureza del "no guardar datos"; problemático bajo restricciones de **privacidad** (el paper sugiere codificar las clases viejas vía autoencoder como trabajo futuro).
- **Brecha frente al batch.** El rendimiento sigue por debajo del entrenamiento con todos los datos (~64% vs 68.6% en iCIFAR-100 con lotes de 10).
- **Destilación contraproducente en lotes muy pequeños** (con lotes de 2 clases puede bajar la *accuracy*).
- **Número finito de clases:** bajo $K$ fijo, $m = K/t$ tiende a su mínimo y solo un número finito de clases puede aprenderse sin crecer recursos.

## Conexión con la Clase 32

La [Clase 32](/clases/clase-32) separa las soluciones al olvido catastrófico en **regularización** (EWC, LwF, destilación), **memoria/rehearsal** (almacenar o regenerar datos) y **arquitecturas dinámicas** (crecer la red). iCaRL es valioso porque **no escoge una familia: las fusiona**.

- **Memoria + regularización en un solo método.** El *exemplar set* + rehearsal es memoria; la destilación es regularización funcional. Los híbridos muestran empíricamente que ambas contribuyen —el ejemplo canónico de por qué "memoria *o* regularización" es un falso dilema.
- **Sucesor de [LwF](/papers/lwf-li-2016) en el escenario class-incremental.** iCaRL toma su destilación, la lleva al caso multiclase verdadero y le añade lo que LwF no tenía: exemplars y la regla NME. El experimento LwF.MC es "LwF sin exemplars", y queda sistemáticamente por debajo —cuantificando cuánto aporta sumar memoria a la regularización.
- **El olvido, hecho visible.** Las matrices de confusión muestran la tensión *estabilidad–plasticidad* hecha gráfico: *finetuning* colapsa hacia el último lote, la representación fija se sesga hacia el primero, e iCaRL logra el balance.
- **El presupuesto $K$ como perilla de diseño.** El trade-off recursos/rendimiento se hace explícito y medible —conectable con [GEM (Lopez-Paz & Ranzato, 2017)](/papers/gem-lopez-paz-2017) y la discusión del curso sobre cuándo invertir en replay vale la pena.
- **Herding como puente a las métricas.** Clasificar por cercanía al prototipo de clase emparenta a iCaRL con NCM y Prototypical Networks: las técnicas del [aprendizaje continuo](/fundamentos/aprendizaje-continuo) no viven en compartimentos estancos.

## Notas y enlaces

- Preprint: [arxiv.org/abs/1611.07725](https://arxiv.org/abs/1611.07725) (v2, 14 abr 2017).
- Código: [github.com/srebuffi/iCaRL](https://github.com/srebuffi/iCaRL) (Theano para iCIFAR-100, TensorFlow para iILSVRC).
- Afiliaciones: University of Oxford / IST Austria.
- Fundamento transversal: [aprendizaje continuo](/fundamentos/aprendizaje-continuo).
