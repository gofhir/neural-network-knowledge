---
title: "Datos estructurados"
weight: 7
sidebar:
  open: true
---

# Datos estructurados

## El problema central

"Datos estructurados" es el cajón de sastre que reúne todo lo que no es texto, imagen, audio o video. Tres familias principales lo componen: **tabular** (filas y columnas, mezcla de tipos numéricos y categóricos — facturas, registros médicos, transacciones), **series temporales** (mediciones ordenadas en el tiempo — sensores, demanda eléctrica, precios), y **grafos** (entidades conectadas por relaciones — redes sociales, moléculas, knowledge graphs). Comparten una propiedad central: la estructura matemática del dato es **discreta y heterogénea**, no continua y uniforme como en imagen o audio. Eso hace que las técnicas que dominaron otros dominios — convoluciones, atención sobre tokens densos — no sean trivialmente aplicables.

Tres tensiones definen el campo: (1) **¿deep learning o gradient boosting en tabular?** XGBoost (2014) sigue ganando muchos benchmarks tabulares en 2025 — un caso raro donde DL no destrona al método clásico, debate aún vivo y sin ganador claro; (2) **predicción de horizonte largo en series temporales** — modelos autoregresivos acumulan error en multi-step rollout, y modelos globales (entrenados sobre muchas series) generalizan mejor que ajustes por serie; (3) **inductive vs transductive en grafos** — entrenar sobre un grafo fijo (Cora, Pubmed) vs aprender funciones que generalicen a grafos no vistos. Cada era de la disciplina navegó estas tres tensiones de forma distinta.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era de ML clásico tabular" years="1990s-2010" >}}
    {{< hito year="1990s" name="Regresión lineal y logística" status="minimal" >}}
      Regresión lineal (Legendre/Gauss, ~1800) y regresión logística (Cox, 1958) son los pilares clásicos del análisis de datos tabulares. **Por qué importó:** siguen siendo el baseline irrenunciable en estadística aplicada, finanzas y epidemiología.
    {{< /hito >}}
    {{< hito year="2001" name="Random Forest" status="minimal" >}}
      Breiman: ensamblar muchos árboles de decisión entrenados sobre subsamples + sub-features y promediar predicciones. Reduce varianza sin aumentar sesgo. **Por qué importó:** primer modelo "non-trivial" que funcionaba out-of-the-box sobre datos tabulares heterogéneos sin tuning extensivo.
    {{< /hito >}}
    {{< hito year="2001" name="GBM (Gradient Boosting Machine)" status="minimal" >}}
      Friedman: en lugar de promediar árboles independientes, ajustar árboles secuencialmente a los residuos del ensemble previo. **Por qué importó:** marco teórico de boosting + funciones de pérdida arbitrarias; precursor directo de XGBoost.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era del Gradient Boosting moderno" years="2014-2017" >}}
    {{< hito year="2014" name="XGBoost" status="minimal" >}}
      Chen & Guestrin (paper KDD 2016, biblioteca abierta 2014): GBM con regularización explícita, paralelización, manejo nativo de missing values y escala industrial. **Por qué importó:** ganó la mayoría de competencias Kaggle 2015-2018 y se volvió el default de producción en banca, healthtech, retail.
    {{< /hito >}}
    {{< hito year="2017" name="LightGBM" status="minimal" >}}
      Microsoft (Ke et al.): GBM con histogram-based splits y leaf-wise growth — más rápido y memoria-eficiente que XGBoost. **Por qué importó:** alternativa industrial dominante para datasets grandes (>1M filas).
    {{< /hito >}}
    {{< hito year="2017" name="CatBoost" status="minimal" >}}
      Yandex (Prokhorenkova et al.): GBM con manejo nativo de variables categóricas vía target encoding ordenado, mitigando target leakage. **Por qué importó:** mejor que XGBoost/LightGBM en datasets con muchas variables categóricas, frecuente en e-commerce y publicidad.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de DL para grafos y series" years="2016-2019" >}}
    {{< hito year="2017" name="GCN" status="minimal" >}}
      Kipf & Welling: *Graph Convolutional Network* — generaliza convolución a grafos vía propagación de mensajes con la matriz de adyacencia normalizada. **Por qué importó:** arquitectura simple y efectiva que abrió la era moderna de Graph Neural Networks.
    {{< /hito >}}
    {{< hito year="2017" name="GraphSAGE" status="minimal" >}}
      Hamilton et al.: muestrea vecinos en lugar de procesar todo el grafo, permitiendo entrenamiento inductive sobre grafos enormes. **Por qué importó:** primer GNN que escalaba a producción industrial (Pinterest, redes sociales).
    {{< /hito >}}
    {{< hito year="2018" name="GAT (Graph Attention Network)" status="minimal" >}}
      Veličković et al.: aplica atención a la propagación entre nodos vecinos — el modelo aprende qué vecinos importan más. **Por qué importó:** mostró que el patrón "atención" se transfería a grafos; estado del arte por varios años.
    {{< /hito >}}
    {{< hito year="2017" name="DeepAR" status="minimal" >}}
      Salinas et al. (Amazon): modelo autoregresivo basado en LSTM para forecasting probabilístico de muchas series temporales relacionadas, entrenado globalmente. **Por qué importó:** primer DL forecaster competitivo con métodos clásicos a escala industrial.
    {{< /hito >}}
    {{< hito year="2019" name="N-BEATS" status="minimal" >}}
      Oreshkin et al.: arquitectura de bloques de proyecciones polinomiales y trigonométricas con backcast/forecast residuals. Sin RNN ni atención. **Por qué importó:** ganó la M4 Competition (forecasting), demostrando que DL pure podía superar a métodos clásicos.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de Transformers a tabular y series" years="2019-2022" >}}
    {{< hito year="2019" name="TFT (Temporal Fusion Transformer)" status="minimal" >}}
      Lim et al. (Google): atención multi-horizonte con compuertas que seleccionan features relevantes por timestep. **Por qué importó:** estado del arte interpretable en forecasting con múltiples covariates conocidos a futuro.
    {{< /hito >}}
    {{< hito year="2020" name="TabTransformer" status="minimal" >}}
      Huang et al. (Amazon): aplica self-attention a embeddings de variables categóricas tabulares, dejando las numéricas en pipeline tradicional. **Por qué importó:** primer Transformer competitivo en tabular, aunque XGBoost seguía ganando en muchos benchmarks.
    {{< /hito >}}
    {{< hito year="2021" name="FT-Transformer" status="minimal" >}}
      Gorishniy et al.: trata cada feature (numérica o categórica) como un token y aplica un Transformer estándar. **Por qué importó:** referencia académica más limpia para el debate "DL en tabular".
    {{< /hito >}}
    {{< hito year="2022" name="PatchTST" status="minimal" >}}
      Nie et al.: divide la serie en parches (estilo ViT) y aplica un Transformer encoder. **Por qué importó:** arquitectura simple y efectiva, supera a TFT y N-BEATS en muchos benchmarks de forecasting de horizonte largo.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de Foundation Models y XGBoost still rules" years="2023-presente" >}}
    {{< hito year="2023" name="TabPFN" status="minimal" >}}
      Hollmann et al.: Transformer preentrenado offline sobre datasets sintéticos generados por priors Bayesianos, hace inferencia in-context sin entrenamiento por dataset. **Por qué importó:** primer modelo que igualó o superó a XGBoost en datasets pequeños (<10k filas, <100 features) — sin entrenarse en ellos.
    {{< /hito >}}
    {{< hito year="2023" name="TimeGPT" status="minimal" >}}
      Garza & Mergenthaler-Canseco (Nixtla): primer foundation model comercial para forecasting, entrenado sobre 100B+ datapoints. Zero-shot transfer a series no vistas. **Por qué importó:** abrió la categoría "foundation model de series" como producto, análogo a Whisper para audio.
    {{< /hito >}}
    {{< hito year="2024" name="Chronos" status="minimal" >}}
      Ansari et al. (Amazon): tokeniza valores de series temporales y entrena un Transformer estilo T5. Open weights. **Por qué importó:** democratizó la receta foundation-model-de-series — base de muchas pipelines academic e industriales.
    {{< /hito >}}
    {{< hito year="2024" name="Lag-Llama" status="minimal" >}}
      Rasul et al. (ServiceNow / MILA): foundation model univariate basado en LLaMA para forecasting probabilístico. Open weights. **Por qué importó:** mostró que adaptar arquitecturas LLM a series funciona — pista del enfoque "todo es secuencia" para 2025.
    {{< /hito >}}
    {{< hito year="2025" name="TabPFN v2" status="minimal" >}}
      Hollmann et al.: extiende TabPFN a más filas (millones) y más features, manteniendo el patrón in-context. **Por qué importó:** cambió el balance del debate "DL vs GBM" en datasets pequeños y medianos — el primer reto serio a la corona de XGBoost en una década.
    {{< /hito >}}
    {{< hito year="2024-2025" name="Debate vivo: GBM vs DL" status="minimal" >}}
      *No es un modelo sino un fenómeno.* Múltiples papers ([Grinsztajn et al. 2022](https://arxiv.org/abs/2207.08815), [McElfresh et al. 2024](https://arxiv.org/abs/2305.02997)) muestran que XGBoost/LightGBM/CatBoost siguen ganando o empatando con DL en la mayoría de benchmarks tabulares de tamaño realista. **Por qué importó:** caso único en deep learning donde el método clásico no fue destronado — el dato discreto y heterogéneo de tabular castiga el sesgo inductivo de las redes.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}
