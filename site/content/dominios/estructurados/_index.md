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

## Era 1 — ML clásico tabular (1990s-2010)

### Problema heredado

Antes de los 2010s, datos tabulares estaban dominados por estadística clásica: regresión lineal y logística para predicción y inferencia, ANOVA para comparaciones, y árboles de decisión simples para clasificación interpretable. Funcionaban, pero no escalaban a las complejidades de datos modernos: interacciones no lineales entre cientos de variables, mezclas de tipos numéricos y categóricos, missing values masivos, y datasets de millones de filas.

### Idea clave

**Ensembles de árboles.** La idea revolucionaria de los 2000s fue combinar muchos árboles de decisión simples en un ensemble. Random Forest (Breiman, 2001) entrena cada árbol sobre un subsample bootstrap del dataset y un subconjunto aleatorio de features, luego promedia las predicciones. La aleatoriedad reduce varianza sin aumentar sesgo significativamente.

Gradient Boosting Machine (Friedman, 2001) tomó un camino distinto: en lugar de árboles independientes, ajustar árboles secuencialmente a los **residuos** del ensemble previo. Cada nuevo árbol corrige los errores de los anteriores. La pérdida puede ser arbitraria (cuadrática, log-loss, exponencial), lo que permite optimizar para clasificación, regresión o ranking. Los teoremas de Friedman demostraron convergencia bajo condiciones razonables.

### Qué la destronó

GBM era teóricamente potente pero implementaciones tempranas (sklearn, gbm en R) eran lentas y no escalaban a datos modernos. Faltaba una implementación industrial: regularización explícita, paralelización, manejo nativo de missing values, GPU support. Esa pieza llegó con XGBoost en 2014.

## Era 2 — Gradient Boosting moderno (2014-2017)

### Problema heredado

GBM funcionaba en teoría pero las implementaciones eran inadecuadas para datasets grandes y producción industrial. Y la comunidad de ML estaba mirando hacia deep learning — AlexNet 2012, los primeros papers de DL — sin pensar que GBM podía aún tener mucho que dar.

### Idea clave

**XGBoost: ingeniería al servicio del algoritmo.** Chen & Guestrin (paper KDD 2016, biblioteca pública desde 2014) tomaron GBM y lo industrializaron: regularización L1+L2 explícita en la pérdida, paralelización por feature en cada split, sparsity-aware splits para missing values, cache-aware data layout, y compresión de bloques. El resultado: 10× más rápido que GBM clásico y mejor accuracy. XGBoost se volvió la herramienta dominante en Kaggle 2015-2018, ganando la mayoría de competencias tabulares y muchas no-tabulares.

LightGBM (Microsoft, 2017) añadió histogram-based splits (cuantizar features continuas a buckets para acelerar split-finding) y leaf-wise growth (expandir el nodo de mejor ganancia, no nivel-por-nivel) — más rápido aún en datasets >1M filas. CatBoost (Yandex, 2017) introdujo target encoding ordenado para variables categóricas con muchos niveles, evitando el target leakage del encoding ingenuo.

### Qué la destronó

GBM moderno no fue destronado en tabular — sigue ganando en muchos benchmarks en 2025. Pero la frontera del DL se movió a otros datos estructurados donde GBM no aplicaba directamente: **grafos** (cómo entrenar sobre estructura relacional) y **series temporales** (cómo capturar dependencias temporales largas).

## Era 3 — DL para grafos y series (2016-2019)

### Problema heredado

Random Forest y GBM operan sobre filas independientes con features fijas. **Grafos** rompen esa suposición: los nodos están conectados, y la información sobre un nodo depende de sus vecinos. **Series temporales** la rompen de otra forma: las observaciones están ordenadas y las dependencias importantes pueden estar a 1, 10 o 100 timesteps de distancia. Para ambos casos, faltaban arquitecturas neuronales adecuadas.

### Idea clave

**Para grafos: propagación de mensajes.** GCN (Kipf & Welling, 2017) propuso una generalización de la convolución a grafos: cada nodo agrega información de sus vecinos vía la matriz de adyacencia normalizada. Capas apiladas extienden el campo receptivo a vecinos a distancia 2, 3, etc. GraphSAGE (Hamilton et al., 2017) escaló la idea a grafos enormes muestreando vecinos en lugar de procesar el grafo completo, y estableció el patrón inductive (entrenar sobre un grafo y aplicar a otros). GAT (Veličković et al., 2018) reemplazó los pesos uniformes de la matriz de adyacencia por pesos aprendidos vía atención: el modelo decide qué vecinos importan más para cada nodo.

**Para series: modelos globales con DL.** DeepAR (Salinas et al., Amazon, 2017/2019) entrenó un LSTM autoregresivo sobre cientos o miles de series temporales relacionadas, generando forecasts probabilísticos. La clave era el entrenamiento global: una sola política aprendida funciona para todas las series, sin ajuste por serie. N-BEATS (Oreshkin et al., 2019) tomó un camino sin recurrencia: bloques de proyecciones polinomiales y trigonométricas que descomponen la serie en componentes interpretables. Ganó la M4 Competition, demostrando que DL puro podía superar a ARIMA y ETS clásicos.

### Qué la destronó

Las arquitecturas dedicadas (GCN/GAT/N-BEATS) eran efectivas pero ad hoc. La pregunta abierta de finales de 2010s era si la atención y los Transformers — ya dominantes en NLP y avanzando en visión — podían transferir su éxito a tabular y series.

## Era 4 — Transformers a tabular y series (2019-2022)

### Problema heredado

Transformers habían transformado NLP (BERT 2018, GPT 2018-2019). Vision Transformer (2020) había demostrado que la receta también aplicaba a imágenes. La pregunta natural: ¿se puede usar self-attention sobre datos tabulares y series temporales?

### Idea clave

**Tokenizar y aplicar Transformer.** Para series, TFT (Temporal Fusion Transformer, Lim et al., 2019) introdujo atención multi-horizonte con compuertas que seleccionan features relevantes por timestep — interpretable y efectivo en forecasting con múltiples covariates conocidos a futuro. PatchTST (Nie et al., 2022) tomó la idea de ViT — dividir la imagen en parches — y la aplicó a series: dividir la secuencia temporal en parches y procesarlos con un Transformer encoder. Más simple y efectivo en forecasting de horizonte largo.

Para tabular, TabTransformer (Huang et al., Amazon, 2020) aplicó self-attention a embeddings de variables categóricas, dejando las numéricas en una pipeline tradicional. FT-Transformer (Gorishniy et al., 2021) fue más limpio: cada feature (numérica o categórica) se trata como un token, y el modelo aplica un Transformer estándar. Resultados: comparables a XGBoost en algunos benchmarks, pero no claramente superiores.

### Qué la destronó

En tabular, los Transformers no destronaron a XGBoost — el debate "DL vs GBM" siguió abierto. En series, los Transformers se volvieron competitivos pero no dominantes. La frontera real apareció en 2023 con la idea importada de NLP: **foundation models pretrainados** sobre datasets sintéticos o masivos.

## Era 5 — Foundation models y "XGBoost still rules" (2023-presente)

### Problema heredado

Cada nueva tarea tabular o de forecasting requería entrenar un modelo desde cero. Funcionaba bien con suficientes datos, pero datasets pequeños (típicos en healthtech, banca regulada, ciencia experimental) seguían favoreciendo a XGBoost por su robustez. Y los foundation models que habían transformado NLP, visión y audio aún no tenían equivalente en datos estructurados.

### Idea clave

Tres líneas paralelas:

1. **TabPFN (Hollmann et al., 2023):** un Transformer preentrenado offline sobre **datasets tabulares sintéticos generados por priors Bayesianos**. En inferencia, recibe el dataset completo (filas + labels) en su contexto y predice nuevas filas in-context, sin entrenamiento por dataset. Igualó o superó a XGBoost en datasets pequeños (<10k filas, <100 features) — un caso paradigmático de "foundation model" aplicado a tabular. TabPFN v2 (2025) extendió la receta a millones de filas, cambiando el balance del debate.

2. **TimeGPT, Chronos, Lag-Llama (2023-2024):** foundation models para forecasting entrenados sobre billones de datapoints heterogéneos. TimeGPT (Nixtla) fue el primer producto comercial. Chronos (Amazon) tokenizó valores numéricos y entrenó un T5 — open weights. Lag-Llama adaptó LLaMA a series univariate. Todos: zero-shot transfer, una llamada a inferencia para series no vistas.

3. **El debate "GBM vs DL" sigue vivo en tabular.** Múltiples papers de 2022-2024 (Grinsztajn et al., McElfresh et al.) muestran que XGBoost/LightGBM/CatBoost siguen ganando o empatando con DL en la mayoría de benchmarks tabulares de tamaño realista. El dato discreto y heterogéneo de tabular castiga el sesgo inductivo de las redes — que asume continuidad y suavidad — y favorece la naturaleza axis-aligned de los árboles.

### Qué viene

Las apuestas activas: **TabPFN extendido** a datasets industriales (millones de filas, miles de features), **forecasting universal** (un modelo que generalice a cualquier dominio sin fine-tuning), **GNN + LLMs** (grafos como contexto enriquecido para razonamiento de modelos generales — GraphRAG, knowledge graphs corporativos), **AutoML cada vez más automatizado** (la era del data scientist generalista que selecciona modelos manualmente está terminando), y la pregunta abierta: **¿cuándo destronará el foundation model a XGBoost en producción industrial?** TabPFN v2 (2025) es el primer reto serio en una década, pero XGBoost mantiene su corona en muchos casos. La respuesta a esta pregunta marcará el cierre del debate más largo del campo.

## Estado del arte hoy

{{< callout type="info" >}}

**Frontier datos estructurados (2024-2025).** Tres líneas paralelas: (a) foundation models tabulares (TabPFN v2) que finalmente retan a XGBoost en datasets pequeños/medianos; (b) foundation models de series (TimeGPT, Chronos, Lag-Llama) con zero-shot transfer; (c) GBM industrial (XGBoost, LightGBM, CatBoost) sigue siendo el default de producción.

- **TabPFN v2** — Hollmann et al. Tabular foundation model que iguala o supera a XGBoost en datasets <1M filas. Primer reto serio a la corona de XGBoost en una década.
- **TimeGPT** — Nixtla. Forecasting comercial con zero-shot transfer.
- **Chronos** — Amazon. Open weights, base de muchas pipelines downstream.
- **Lag-Llama** — ServiceNow / MILA. Foundation model univariate basado en LLaMA.
- **GraphRAG** — Microsoft. Knowledge graphs como contexto para LLMs en enterprise search.
- **GIN / Graph Transformers** — estado del arte en química y drug discovery.
- **AutoGluon / H2O AutoML** — pipelines automáticos que combinan GBM, RF y DL.
- **XGBoost / LightGBM / CatBoost** — defaults industriales en Kaggle, fintech, healthtech.

{{< /callout >}}

## Casos de uso reales

- **Fraud detection en banca y fintech** — XGBoost domina (millones de transacciones, latencia baja, interpretabilidad regulatoria).
- **Forecasting de demanda en retail y energía** — DeepAR, TimeGPT, ensembles con métodos clásicos (ARIMA, ETS).
- **Drug discovery y química computacional** — GNN sobre moléculas; AlphaFold para proteínas (paradigma distinto, pero estructurado en grafo).
- **Recomendación a escala web** — DLRM (Meta), Wide & Deep (Google) y variantes para feed/búsqueda/ads.
- **Healthtech** — predicción de readmisión, riesgo cardiovascular, clasificación sobre EHR; XGBoost + features clínicas a mano siguen dominando.
- **Knowledge graphs y enterprise search** — GraphRAG, Neo4j + LLMs para búsqueda semántica corporativa.
- **Series financieras** — trading, riesgo, derivados — combinaciones de GBM, ARIMA, DL y métodos de volatilidad.
- **Manufactura y mantenimiento predictivo** — series temporales de sensores con anomaly detection (Isolation Forest, autoencoders, foundation models).
- **Marketing y churn prediction** — XGBoost + feature engineering domina en CRM y SaaS B2B.

## Qué viene

Las apuestas activas: **foundation models tabulares efectivos a escala industrial** (TabPFN v2 cambió el debate en datasets pequeños — falta probar millones de filas y miles de features), **forecasting universal** (un solo modelo para cualquier dominio sin fine-tuning), **GNN + LLMs** (grafos como contexto para razonamiento de modelos generales — GraphRAG es la primera ola), **AutoML cada vez más automatizado** (la era del data scientist generalista que selecciona modelos manualmente está terminando), y **el debate DL vs GBM** sigue vivo — TabPFN v2 cambió el balance, pero XGBoost mantiene su corona en muchos casos de producción. La pregunta abierta: ¿qué tarea dejará primero de ser "structured data engineering" para volverse "prompt the foundation model"?

## Recursos relacionados

**Material adyacente en el curso:**
- [Foundation models](/fundamentos/foundation-models) — el contexto general que se aplica también a TabPFN, TimeGPT, Chronos.
- [Self-attention](/fundamentos/self-attention) y [Transformer](/fundamentos/transformer) — la arquitectura sobre la que corren TabPFN, TFT, PatchTST y Graph Transformers.
- [Embeddings distribuidos](/fundamentos/embeddings-distribuidos) — base conceptual para entity embeddings tabulares.

**Dominios relacionados:**
- [Texto / NLP](/dominios/texto) — donde nacieron los Transformers que luego se transfirieron a tabular y series.
- [Visión](/dominios/vision) — ViT inspiró PatchTST en series.
- [Robótica / RL](/dominios/robotica) — comparten el debate "DL vs métodos clásicos" en optimización industrial.

**Cierre del proyecto Dominios:** este es el séptimo y último dominio. La sección [Dominios](/dominios) cubre ahora: Texto/NLP, Visión, Audio/Voz, Video, Multimodal, Robótica/RL y Datos estructurados.

---

*Última actualización: 2026-05-06.*
