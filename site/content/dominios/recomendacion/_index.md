---
title: "Recomendación"
weight: 8
sidebar:
  open: true
---

# Recomendación

## El problema central

Recomendar es resolver un problema de **información incompleta a escala masiva**. El insumo central es la matriz usuario-item: filas de usuarios, columnas de items (películas, productos, canciones, pins) y, en cada celda, una señal de afinidad. El problema es que esa matriz está **vacía en más del 99%**: un usuario de Netflix calificó unas decenas de las miles de películas del catálogo, y un comprador de Amazon interactuó con una fracción ínfima de los millones de SKU disponibles. La tarea no es "memorizar" lo que el usuario ya hizo, sino **inferir los huecos**: predecir qué pasaría en las celdas vacías y, sobre esa predicción, elegir qué mostrar. A diferencia de la clasificación supervisada clásica, aquí no hay etiquetas negativas explícitas — que un usuario no haya visto una película no significa que no le guste, solo que **no la conoce todavía**. Esta ambigüedad entre "no le interesa" y "no la descubrió" es la raíz de casi todas las decisiones de diseño del campo.

Cuatro tensiones vertebran toda la historia de la recomendación. **(1) Feedback explícito vs. implícito:** las estrellas y los pulgares (explícito) son escasos y sesgados, mientras que clics, compras y tiempo de visionado (implícito) son abundantes pero ruidosos — un clic no es un "me gusta", y la ausencia de clic no es un "no me gusta". El campo migró progresivamente de modelar ratings a modelar comportamiento implícito con muestreo de negativos. **(2) Memorización vs. generalización:** los sistemas deben recordar co-ocurrencias frecuentes y específicas ("quien compró pañales compró cerveza") pero también generalizar a combinaciones nunca vistas vía embeddings — la dualidad que *Wide&Deep* hizo explícita. **(3) Accuracy vs. diversidad, serendipia y cold-start:** optimizar solo el acierto colapsa en *filter bubbles* que recomiendan lo obvio; un sistema sano debe sorprender, diversificar y resolver el arranque en frío de usuarios e items sin historial. **(4) De predecir ratings a rankear top-K a generar:** el campo se desplazó de minimizar el error de predicción de una nota (RMSE, la métrica del Netflix Prize) hacia **optimizar directamente el orden** de los pocos items que caben en pantalla (ranking, recall@K), y más recientemente hacia **generar** el identificador del item con modelos secuenciales y generativos. Cada salto cambió la función de pérdida, la arquitectura y hasta la noción de qué significa "recomendar bien".

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era de Collaborative Filtering clásico" years="1992-2005" >}}
    {{< hito year="1992" name="Tapestry y el término collaborative filtering" status="minimal" >}}
      El sistema Tapestry de Xerox PARC acuñó el término *collaborative filtering* para filtrar correo y documentos según las reacciones de otras personas. **Por qué importó:** nombró el paradigma central del campo — recomendar usando el comportamiento de la comunidad, no el contenido del item.
    {{< /hito >}}
    {{< hito year="1994" name="GroupLens" status="minimal" >}}
      Sistema de recomendación de noticias de Usenet basado en *vecindad*: encontrar usuarios con gustos parecidos y promediar sus calificaciones para predecir las tuyas. **Por qué importó:** primer collaborative filtering automatizado a escala de comunidad; fundó el grupo que años después liberó el dataset MovieLens, banco de pruebas de toda una generación.
    {{< /hito >}}
    {{< hito year="1998" name="Content-based filtering" status="minimal" >}}
      Enfoque complementario que recomienda items similares a los que el usuario ya consumió, usando atributos del propio item (género, palabras clave, metadatos) en vez del comportamiento de otros. **Por qué importó:** la única vía para arrancar en frío un item nuevo sin historial de interacciones; sigue siendo la mitad de todo sistema híbrido moderno.
    {{< /hito >}}
    {{< hito year="2003" name="Amazon item-to-item CF" status="minimal" >}}
      Linden, Smith y York invirtieron la lógica: en vez de buscar usuarios parecidos (caro y disperso), precomputaron *items parecidos* a partir de co-compras. **Por qué importó:** hizo el collaborative filtering viable en producción web a escala de millones de usuarios y items; el "los clientes que compraron esto también compraron…" definió el e-commerce por dos décadas.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era del Netflix Prize y factores latentes" years="2006-2010" >}}
    {{< hito year="2006" name="Netflix Prize" status="minimal" >}}
      Netflix ofreció un millón de dólares a quien mejorara su predictor de ratings en un 10% de RMSE, liberando un dataset de 100 millones de calificaciones. **Por qué importó:** catalizó la investigación moderna en recomendación, popularizó la matrix factorization y los *ensembles*, y mostró que predecir ratings y rankear son problemas distintos.
    {{< /hito >}}
    {{< hito year="2009" name="Matrix Factorization (Koren et al.)" status="deep" link="/papers/matrix-factorization-koren-2009" >}}
      Factorizó la matriz usuario-item en dos matrices de **factores latentes** de baja dimensión, capturando gustos como vectores densos aprendidos por descenso de gradiente con regularización y sesgos. **Por qué importó:** el modelo ganador del Netflix Prize; estableció los embeddings de usuario e item como la representación canónica del campo, vigente hasta hoy bajo arquitecturas neuronales.
    {{< /hito >}}
    {{< hito year="2009" name="BPR — Bayesian Personalized Ranking (Rendle et al.)" status="deep" link="/papers/bpr-rendle-2009" >}}
      Reformuló la recomendación con feedback implícito como un problema de **ranking por pares**: optimizar que el item observado se ordene por encima de uno no observado, en vez de predecir una nota. **Por qué importó:** desacopló la métrica de optimización (ranking) de la de predicción (RMSE); su pérdida pairwise y el muestreo de negativos son la base de casi todo el recsys implícito posterior.
    {{< /hito >}}
    {{< hito year="2010" name="Factorization Machines (Rendle)" status="minimal" >}}
      Generalización que modela interacciones de segundo orden entre *cualquier* par de variables (no solo usuario-item) con factores latentes compartidos, manejando features dispersas y contextuales. **Por qué importó:** unificó matrix factorization, SVD++ y modelos contextuales bajo un solo formalismo; antecesor directo de DeepFM y los modelos de CTR modernos.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era del deep learning en recomendación" years="2013-2017" >}}
    {{< hito year="2013" name="DSSM — Deep Structured Semantic Model" status="deep" link="/papers/dssm-huang-2013" >}}
      Dos torres neuronales que proyectan consulta y documento a un espacio semántico común, entrenadas para maximizar la similitud coseno de los pares relevantes. **Por qué importó:** plantilla original de la arquitectura *two-tower* que hoy domina el retrieval de candidatos en buscadores y recomendadores a escala web.
    {{< /hito >}}
    {{< hito year="2016" name="YouTube Deep Neural Networks (Covington et al.)" status="deep" link="/papers/youtube-dnn-covington-2016" >}}
      Arquitectura de dos etapas — un modelo de *candidate generation* que filtra millones de videos a cientos, y un *ranking* que ordena el corto — descrita por Google para YouTube. **Por qué importó:** formalizó el patrón retrieval-then-rank de producción y mostró cómo entrenar embeddings de video desde feedback implícito masivo.
    {{< /hito >}}
    {{< hito year="2016" name="Wide & Deep (Cheng et al.)" status="deep" link="/papers/wide-and-deep-cheng-2016" >}}
      Combinó un componente *wide* lineal (memorización de co-ocurrencias específicas) con uno *deep* de embeddings (generalización a combinaciones nuevas), entrenados conjuntamente. **Por qué importó:** articuló de forma canónica la tensión memorización-generalización; desplegado en Google Play, definió la plantilla de los rankers híbridos industriales.
    {{< /hito >}}
    {{< hito year="2016" name="VBPR — Visual Bayesian Personalized Ranking (He & McAuley)" status="deep" link="/papers/vbpr-he-2016" >}}
      Extendió BPR incorporando features visuales de una CNN preentrenada al embedding del item, capturando la influencia de la apariencia en la preferencia. **Por qué importó:** pionero de la **recomendación visual**; demostró que la imagen del producto resuelve cold-start de items y enriquece el ranking, anticipando los recomendadores multimodales.
    {{< /hito >}}
    {{< hito year="2017" name="Neural Collaborative Filtering (He et al.)" status="deep" link="/papers/neural-collaborative-filtering-he-2017" >}}
      Reemplazó el producto punto de la matrix factorization por una red neuronal que aprende la función de interacción usuario-item, combinando una rama de factorización generalizada con un MLP. **Por qué importó:** abanderado del deep learning en collaborative filtering; instaló los embeddings neuronales como estándar, aunque luego se debatió si superaba a baselines bien afinadas.
    {{< /hito >}}
    {{< hito year="2017" name="DeepFM (Guo et al.)" status="deep" link="/papers/deepfm-guo-2017" >}}
      Fusionó Factorization Machines (interacciones de bajo orden) con una red profunda (alto orden) compartiendo los mismos embeddings, sin necesidad de feature engineering manual. **Por qué importó:** referencia de predicción de CTR en publicidad y recomendación; estándar de la industria para datos tabulares dispersos con interacciones.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de embeddings, grafos y retrieval a escala" years="2018-2020" >}}
    {{< hito year="2018" name="PinSage (Ying et al.)" status="deep" link="/papers/pinsage-ying-2018" >}}
      Red neuronal de grafos (GNN) sobre el grafo pin-tablero de Pinterest, con muestreo de vecindario por *random walks* para escalar a miles de millones de nodos. **Por qué importó:** primera GNN de recomendación desplegada a escala web real; probó que la estructura de grafo enriquece los embeddings de item más allá del collaborative filtering plano.
    {{< /hito >}}
    {{< hito year="2018" name="SASRec / BERT4Rec — recomendación secuencial" status="minimal" >}}
      SASRec aplicó self-attention causal y BERT4Rec un encoder bidireccional con masked-item modeling para predecir el siguiente item desde la secuencia de interacciones. **Por qué importó:** trasladaron los Transformers del NLP al recsys, modelando la *dinámica temporal* del gusto en vez de preferencias estáticas; base de los recomendadores secuenciales actuales.
    {{< /hito >}}
    {{< hito year="2019" name="Two-Tower con corrección de sesgo de muestreo (Yi et al.)" status="deep" link="/papers/two-tower-yi-2019" >}}
      Refinó la arquitectura de dos torres para retrieval con una corrección del sesgo introducido por el muestreo *in-batch* de negativos, descrita por Google. **Por qué importó:** hizo entrenable a escala el retrieval de candidatos con embeddings; junto a índices ANN es la columna vertebral del recall en recomendadores y búsqueda modernos.
    {{< /hito >}}
    {{< hito year="2026" name="Case Study — recsys multimodal con imagen y texto" status="deep" link="/clases/clase-25" >}}
      Caso de estudio del curso que combina embeddings visuales y textuales para recomendar, integrando recuperación two-tower con señales multimodales. **Por qué importó:** sintetiza VBPR, two-tower y embeddings multimodales en un pipeline práctico de extremo a extremo aplicado en clase.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de LLMs y recomendación generativa" years="2021-presente" >}}
    {{< hito year="2023" name="TIGER — generative retrieval (Rajput et al.)" status="minimal" >}}
      En vez de recuperar candidatos por similitud de embeddings, un modelo secuencia-a-secuencia **genera directamente el identificador semántico** del item, codificado como una tupla de tokens discretos. **Por qué importó:** abandonó el índice ANN explícito y fundió retrieval y ranking en un solo modelo generativo, abriendo el paradigma de la recomendación generativa.
    {{< /hito >}}
    {{< hito year="2023" name="LLMs como recomendadores y prompting" status="minimal" >}}
      Enfoques que usan modelos de lenguaje grandes para recomendar vía *prompting* en lenguaje natural, razonamiento sobre el historial del usuario y generación de explicaciones, con o sin fine-tuning. **Por qué importó:** prometen recomendación *zero-shot* y conversacional, pero abren la tensión actual del campo — el conocimiento general y la fluidez del LLM frente a la **eficiencia, frescura del catálogo y señal de comportamiento** que dominan los recomendadores especializados a escala.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}

## Recursos relacionados

**Material en el curso:**
- [Sistemas de Recomendación](/fundamentos/recommender-systems), [Métricas de Ranking](/fundamentos/ranking-metrics), [Two-Tower Retrieval](/fundamentos/two-tower-retrieval), [Triplet Loss](/fundamentos/triplet-loss)
- [Clase 25 — Recomendación con Imágenes y Texto](/clases/clase-25)

**Dominios relacionados:**
- [Datos estructurados](/dominios/estructurados) — collaborative filtering nació sobre matrices usuario-item; PinSage es un GNN.
- [Multimodal](/dominios/multimodal) — la recomendación visual (VBPR) y el case study combinan imagen + texto.

*Última actualización: 2026-06-07.*
