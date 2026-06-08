---
title: "BPR: Bayesian Personalized Ranking from Implicit Feedback"
weight: 255
math: true
---

{{< paper-card
    title="BPR: Bayesian Personalized Ranking from Implicit Feedback"
    authors="Rendle, Freudenthaler, Gantner, Schmidt-Thieme"
    year="2009"
    venue="UAI 2009"
    pdf="/papers/bpr-rendle-2009.pdf"
    arxiv="1205.2618" >}}
BPR define el **objetivo de optimización canónico para ranking personalizado con feedback implícito** (clics, compras, vistas). En vez de scorear ítems de forma absoluta, optimiza **pares**: si el usuario vio $i$ y no $j$, debe preferir $i$ sobre $j$. El criterio **BPR-OPT** es un estimador *maximum a posteriori* bayesiano cuya pérdida es $\ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})$, y se entrena con **LearnBPR** (SGD con *bootstrap sampling* de tripletas). Es el **ancestro directo del aprendizaje de ranking pairwise y de la triplet loss** en recomendación moderna.
{{< /paper-card >}}

---

## Contexto

La mayoría del feedback en sistemas reales no es explícito (ratings 1-5) sino **implícito**: clics, compras, tiempos de visualización registrados automáticamente. Su rasgo definitorio es que **solo se observan ejemplos positivos**; los pares usuario-ítem no observados son una mezcla indistinguible de negativos reales y valores faltantes (ítems que el usuario podría querer pero aún no ha visto).

El enfoque ingenuo de la época (Hu et al. 2008; Pan et al. 2008) etiquetaba como **1** lo observado y como **0** todo lo demás, ajustando un modelo que predice un puntaje absoluto $\hat{x}_{ui}$. Rendle et al. señalan el defecto de raíz: **todos los ítems que el modelo debería rankear en el futuro se le presentan como negativos durante el entrenamiento**. Un modelo suficientemente expresivo aprendería a predecir solo ceros y no podría rankear; si igual rankea, es solo gracias a la regularización, no al criterio. El ranking funciona *a pesar* del objetivo, no *gracias* a él.

La tarea aquí es **recomendación de ítems**: producir, para cada usuario, un orden total personalizado $>_u$ sobre todos los ítems — exactamente lo que necesita una tienda online o un portal de video, donde se muestra una lista ordenada, no un número. Ver [/fundamentos/recommender-systems](/fundamentos/recommender-systems).

## Ideas principales

La propuesta es cambiar la unidad de optimización: de "scorear ítems" a **ordenar pares de ítems**. Si $(u,i) \in S$ (observado) y $j$ no fue observado, se asume $i >_u j$. Esto construye el conjunto de tripletas de entrenamiento:

$$D_S := \{(u,i,j) \mid i \in I_u^+ \wedge j \in I \setminus I_u^+\}$$

con la semántica "$u$ prefiere $i$ sobre $j$". Ventaja clave: los pares faltantes (entre dos ítems no observados) son **exactamente los que habrá que rankear en el futuro**, así que train y test son disjuntos desde un punto de vista de pares.

### El criterio BPR-OPT y la pérdida pairwise

La probabilidad de preferencia se modela con la **sigmoide logística** aplicada a la **diferencia de puntajes** $\hat{x}_{uij} := \hat{x}_{ui} - \hat{x}_{uj}$:

$$p(i >_u j \mid \Theta) := \sigma(\hat{x}_{uij}), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}$$

Con un **prior gaussiano** $p(\Theta) \sim N(0, \lambda_\Theta I)$ sobre los parámetros, el estimador *maximum a posteriori* (MAP) da el criterio **BPR-OPT**:

$$\text{BPR-OPT} = \ln p(\Theta \mid >_u) = \sum_{(u,i,j) \in D_S} \ln \sigma(\hat{x}_{uij}) \;-\; \lambda_\Theta \lVert \Theta \rVert^2$$

El prior gaussiano se convierte, vía el logaritmo, en la **regularización L2**. Este es el objetivo canónico de ranking pairwise con feedback implícito.

**Conexión con AUC:** el AUC por usuario es $\sum_{(u,i,j) \in D_S} z_u\, \delta(\hat{x}_{uij} > 0)$. BPR-OPT solo difiere en la **función de pérdida**: AUC usa la Heaviside no diferenciable $\delta(x>0)$; BPR usa la diferenciable $\ln \sigma(x)$. Es práctica común reemplazar Heaviside por una función suave de forma heurística — la contribución teórica de BPR es derivar la sustitución $\ln \sigma(x)$ desde la máxima verosimilitud, no por heurística.

### LearnBPR: SGD con bootstrap sampling

El criterio es diferenciable, pero el SGD estándar falla aquí. El gradiente completo sobre $O(|S|\,|I|)$ tripletas es inviable, y la **asimetría** de los pares (un ítem popular aparece en muchísimas tripletas) hace que su gradiente domine. Recorrer los datos por usuario o por ítem también converge mal, por demasiadas actualizaciones consecutivas sobre el mismo par.

La solución es **muestrear tripletas al azar (uniforme, con reemplazo)** — *bootstrap sampling*:

```
procedure LearnBPR(D_S, Θ)
  inicializar Θ
  repeat
    draw (u, i, j) from D_S          # uniforme, con reemplazo
    Θ ← Θ + α ( e^{-x̂_uij}/(1+e^{-x̂_uij}) · ∂x̂_uij/∂Θ + λ_Θ·Θ )
  until convergencia
```

Esto evita actualizaciones consecutivas correlacionadas y permite detenerse en cualquier paso. El paper muestra (Figura 5) que LearnBPR converge mucho más rápido que el SGD por usuario.

**Genérico por diseño:** BPR delega en el modelo subyacente la estimación de $\hat{x}_{ui}$. Para **matrix factorization** (BPR-MF), $\hat{x}_{ui} = \langle w_u, h_i \rangle$. Para **kNN adaptativo** (BPR-kNN), la matriz de similitud $C$ se **aprende** en lugar de fijarse con coseno. Solo se necesita el gradiente de $\hat{x}_{uij}$ respecto de cada parámetro.

## Resultados experimentales

Evaluación en dos datasets: **Rossmann** (10.000 usuarios, 4.000 ítems, 426.612 compras) y una submuestra de **Netflix** tratada como implícita removiendo los ratings (10.000 usuarios, 5.000 ítems, 565.738 acciones). Esquema *leave-one-out*, métrica **AUC promedio** (0.5 = azar, 1 = perfecto), 10 repeticiones con grid search de hiperparámetros.

- **BPR-MF y BPR-kNN superan a todos los demás** métodos en ambos datasets: WR-MF, SVD-MF, Cosine-kNN, most-popular.
- Las tres MF (SVD-MF, WR-MF, BPR-MF) comparten **el mismo modelo** pero difieren mucho en calidad: evidencia directa de que **el criterio de optimización importa tanto como el modelo**.
- **SVD-MF sobreajusta**: su AUC *baja* al aumentar dimensiones. **WR-MF** es robusto gracias a la regularización. **BPR-MF supera claramente a WR-MF**: en Netflix, una MF de **8 dimensiones** con BPR-MF iguala a una de **128 dimensiones** con WR-MF.
- Incluso métodos personalizados simples (Cosine-kNN) superan ampliamente la cota teórica de cualquier método **no personalizado**.

Ver [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics) para AUC y otras métricas de ranking.

## Limitaciones reconocibles

- **Única métrica AUC**, que pesa por igual todas las posiciones; no se reportan métricas sensibles al tope de la lista (precision@k, NDCG, MAP), que suelen importar más en la práctica.
- **Muestreo uniforme de negativos**: trata todos los negativos por igual y converge lento cuando casi todos ya están bien rankeados; trabajos posteriores propusieron muestreo de negativos "difíciles".
- **Supuesto de independencia entre pares**, que no es exacto dado que un orden total impone transitividad.
- Validación de 2009: solo MF y kNN, datasets clásicos, sin modelos neuronales ni multimodalidad.
- **Sesgo de exposición no modelado**: asumir $i >_u j$ ignora que $j$ quizás nunca se le mostró al usuario.

## Por qué importa hoy

BPR se convirtió en el **criterio de pérdida estándar de facto para recomendación con feedback implícito** y en la línea base obligatoria de comparación (presente en LightFM, implicit, RecBole, Cornac, etc.). Su lección metodológica — **"optimizar el modelo para el criterio correcto"**, separando la clase de modelo del objetivo de entrenamiento — habilita reutilizar BPR-OPT sobre cualquier modelo que produzca un puntaje real, incluidos los modelos profundos posteriores (NCF, *two-tower* / dual-encoder). Su énfasis en feedback implícito y datos *one-class* anticipó la dirección dominante de la industria, donde los logs de interacción, no los ratings, son la materia prima.

## Conexión con la Clase 25

La [/clases/clase-25](/clases/clase-25) es un *case study* de recomendación multimodal con feedback implícito y métricas de ranking, donde aparece el **metric learning** con [/fundamentos/triplet-loss](/fundamentos/triplet-loss). BPR es el puente conceptual directo:

- **Misma estructura tripleta ancla-positivo-negativo.** En la triplet loss, el ancla debe estar más cerca del positivo que del negativo por un margen. En BPR, el "ancla" es el usuario $u$, el positivo el ítem visto $i$, el negativo el ítem no visto $j$, y el objetivo es $\hat{x}_{ui} > \hat{x}_{uj}$. Ambos optimizan un **orden relativo**, no un valor absoluto.
- **Ítems relevantes vs. no relevantes.** La construcción de $D_S$ por muestreo de negativos es el mismo procedimiento que alimenta una *triplet network* en feedback implícito.
- **Pérdida suave vs. margen duro.** BPR usa $\ln \sigma(\hat{x}_{ui} - \hat{x}_{uj})$ (logística suave, derivada del MLE); la triplet loss clásica usa $\max(0, m - (s_{\text{pos}} - s_{\text{neg}}))$ (hinge con margen). El propio paper lo hace explícito al comparar con MMMF: la versión hinge de BPR-MF es literalmente $\sum \max(0, 1 - \langle w_u, h_i - h_j \rangle) + \text{reg}$. **BPR es la contraparte probabilística suave de la triplet loss.**
- **Del producto punto al two-tower multimodal.** El puntaje de BPR-MF, $\hat{x}_{ui} = \langle w_u, h_i \rangle$, es un producto punto entre un embedding de usuario y uno de ítem. En la clase multimodal esos embeddings provienen de torres neuronales (imagen, texto, etc.), pero **la pérdida de ranking pairwise sobre el producto punto sigue siendo BPR**.

## Notas y enlaces

- PDF local: [/papers/bpr-rendle-2009.pdf](/papers/bpr-rendle-2009.pdf)
- arXiv: [1205.2618](https://arxiv.org/abs/1205.2618)
- Fundamentos relacionados: [/fundamentos/recommender-systems](/fundamentos/recommender-systems), [/fundamentos/triplet-loss](/fundamentos/triplet-loss), [/fundamentos/ranking-metrics](/fundamentos/ranking-metrics)
- Clase: [/clases/clase-25](/clases/clase-25)
