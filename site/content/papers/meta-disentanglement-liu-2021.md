---
title: "Meta-learning + Disentanglement (Medical Segmentation)"
weight: 268
math: true
---
{{< paper-card
    title="Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation"
    authors="Xiao Liu, Spyridon Thermos, Alison O'Neil, Sotirios A. Tsaftaris"
    year="2021"
    venue="MICCAI 2021 (Oral)"
    pdf="/papers/meta-disentanglement-liu-2021.pdf"
    arxiv="2106.13292" >}}
{{< /paper-card >}}

## El problema clínico: domain shift y escasez de anotaciones en imagen médica

Un modelo de segmentación entrenado con imágenes de unos pocos centros clínicos funciona bien en esos datos, pero su rendimiento **cae cuando se aplica a un centro nuevo** con otro escáner, otro protocolo de adquisición u otra población de pacientes. Este fenómeno se llama *domain shift*: el cambio en las estadísticas de los datos entre el dominio donde se entrenó (*source*) y el dominio nuevo no visto (*target*).

Los autores descomponen las fuentes de variación en dos familias, y la distinción no es decorativa: es la que justifica toda la arquitectura.

1. **Variación de población.** Género, edad y etnia difieren entre pacientes de distintas localizaciones, lo que afecta la **anatomía y patología subyacente**. Cambia el contenido que se quiere segmentar.
2. **Variación de escáner y protocolo.** Distintos fabricantes de resonancia (Siemens, Philips, GE, Canon) y parámetros de adquisición afectan las **características de la imagen** (brillo, contraste) sin cambiar la anatomía.

La intuición clave: para segmentar bien queremos ser **sensibles a la anatomía e insensibles a la apariencia**, sea esta común a varios centros o específica de uno.

El enfoque ingenuo ante el domain shift sería "adquirir y etiquetar tantos datos diversos como sea posible". El paper lo descarta de inmediato porque la **segmentación es el caso más caro de anotación**: cada píxel debe etiquetarse, la tarea es laboriosa y requiere conocimiento experto (un radiólogo trazando contornos densos). A diferencia de la clasificación, donde basta una etiqueta por imagen, aquí la etiqueta es una máscara densa. Por eso, escriben, los métodos de meta-learning totalmente supervisados previos "no escalan para segmentación médica".

En la práctica hospitalaria real, cada centro aporta **muchos datos crudos sin anotar** (imágenes adquiridas por flujo clínico normal) y **pocos datos anotados** (lo que un experto alcanzó a trazar). Un método que no necesite ver el centro target de antemano y que además aproveche los datos sin etiqueta está alineado con cómo realmente lucen los datos en producción. Ese es el espacio de diseño que ocupa este trabajo.

## Domain generalization: entrenar en varios centros para generalizar a uno no visto

El paper ordena las alternativas según cuánta información del dominio target se asume disponible:

- **Recolectar y etiquetar todo:** caro e inviable.
- **Domain adaptation (DA):** se entrena en dominios source para generalizar a un target "con alguna información del target disponible". El ejemplo típico es la armonización cross-site de MRI. **DA accede a datos del target** (aunque sea sin etiquetas).
- **Domain generalisation (DG):** la alternativa más estricta es **no usar ninguna información del target**. Es más difícil, pero mucho más ampliamente aplicable. Este paper se enfoca aquí.

El contraste DA vs DG es la distinción conceptual central. En DA el target existe en tiempo de entrenamiento (al menos sus imágenes); en DG el target es genuinamente desconocido hasta el despliegue. Para un hospital que comparte un modelo con una red de centros que aún no existen como datasets, **DG es el supuesto realista**.

El objetivo de DG, en palabras del paper, es "identificar representaciones que codifiquen información sobre la tarea a la vez que sean insensibles a la información específica del dominio": representaciones **task-informativas pero domain-invariantes**. Entre las direcciones activas (augmentación directa, regularización del espacio de features, alineamiento de distribuciones, y meta-learning basado en gradiente), los autores eligen la última por una razón concreta: el meta-learning basado en gradiente **no se sobre-ajusta a los dominios source dominantes** —los que aportan más datos—, lo que importa mucho en un dataset multi-centro desbalanceado (un centro grande, varios chicos).

El mecanismo es el **paradigma episódico**: en cada iteración se parten los dominios source en un conjunto *meta-train* y un conjunto *meta-test*, y el modelo se entrena para manejar el domain shift **simulándolo durante el entrenamiento**. Trabajos previos aplicaron este esquema en setting totalmente supervisado (MASF, SAML, LDDG), pero ninguno escalaba a segmentación médica por el costo de anotación.

## La idea: meta-learning (estilo MAML) + disentanglement de representaciones

La propuesta combina dos mecanismos complementarios que antes vivían separados.

**Meta-learning estilo MAML para simular el shift.** Como en MAML, cada iteración se divide en una fase meta-train y una meta-test. La diferencia con el MAML clásico (few-shot por *tareas*) es que aquí los episodios se construyen por **dominios**: meta-train usa unos dominios source y meta-test usa otro dominio source distinto que, en esa iteración, actúa de *proxy* del target no visto. Así el optimizador no solo busca parámetros que funcionen en meta-train, sino parámetros cuyas **futuras actualizaciones generalicen** a un dominio que no participó del gradiente de meta-train. Esto simula explícitamente el domain shift dentro del bucle de entrenamiento.

**Disentanglement para modelar el shift, no solo eliminarlo.** Aquí está el giro conceptual sobre el meta-learning previo. En vez de **eliminar implícitamente** la información del shift simulado (lo que hacían los métodos supervisados con sus constraints), este trabajo la **modela explícitamente** mediante disentanglement, aprendiendo representaciones "completas y suficientes" vía reconstrucción de la imagen.

El modelo separa la imagen en tres representaciones con roles bien definidos:

- $Z$: **features anatómicas espaciales** (grid-like), equivariantes a la segmentación. Es lo que la red de tarea consume.
- $s$: vector que captura **características de imagen comunes** entre dominios.
- $d$: vector que captura **características de imagen específicas de cada dominio**.

La consecuencia es doble. Primero, **habilita la semi-supervisión**: reconstruir la imagen es una tarea no supervisada, así que los datos sin máscara también ayudan a "simular mejor los domain shifts". Segundo, **mejora la aproximación del shift verdadero**: al modelar $s$ y $d$ con más datos (incluidos los no etiquetados), la simulación dentro del meta-learning es más fiel. Y de forma elegante, el meta-learning no solo empuja a $Z, s, d$ a generalizar, sino que "al mismo tiempo mejora (implícitamente) su disentanglement". Las dos mitades del método se refuerzan mutuamente.

## La arquitectura y las pérdidas

Sobre un dataset multi-dominio $\mathcal{D} = \{X_i^k, Y_i^k\}$, con $k \in \{1, \dots, K\}$ dominios source, los bloques son:

- **Feature network $F_\psi : \mathcal{X} \rightarrow \mathcal{Z}$.** Una **2D U-Net** que extrae $Z = F_\psi(X)$ con 8 canales del mismo alto y ancho que la imagen.
- **Task network $T_\theta : \mathcal{Z} \rightarrow \mathcal{Y}$.** Predice la máscara $\hat{Y}$ a partir de $Z$.
- **Encoder de apariencia común $E_S$ y encoder de dominio $E_D$.** Producen $s$ y $d$, ambos vectores de 8 dimensiones.
- **Clasificador de dominio $T_C$.** Una única capa fully-connected que toma $d$ y predice a qué centro pertenece la imagen.
- **Decoder $DE$.** Reconstruye la imagen: $\hat{X} = DE(Z, s, d)$.

La pieza que hace funcionar el disentanglement es **cómo** el decoder combina la representación espacial con las vectoriales: usa **Adaptive Instance Normalization (AdaIN)**:

$$
\text{AdaIN}(Z_i, s_i, d_i) = \sigma(s_i, d_i) \cdot \frac{Z_i - \mu(Z_i)}{\sigma(Z_i)} + \mu(s_i, d_i).
$$

Cada feature map $Z_i$ se normaliza por su propia media y desviación, y luego se escala y desplaza según escalares derivados de $s_i$ y $d_i$. Al inyectar la apariencia **solo** vía las estadísticas de normalización (globales por canal), la información espacial fina queda forzada a residir en $Z$, mientras $s$ y $d$ se quedan con el "tono" de la imagen. Es el sesgo arquitectónico que separa contenido de estilo.

**Las pérdidas** son donde se sostiene la afirmación de semi-supervisión. La distinción crucial: **una sola pérdida necesita máscaras**.

- $\mathcal{L}_{Dice}(Y, \hat{Y})$ es la pérdida supervisada de segmentación. Solo los datos etiquetados la activan ($\lambda_{Dice} = 5$).
- El término de disentanglement $\mathcal{L}_{DT}$ agrupa cinco pérdidas, **ninguna de las cuales necesita máscaras**:

$$
\mathcal{L}_{DT} = \lambda_{rank}\mathcal{L}_{rank}(Z) + \lambda_{KL}\big(\mathcal{L}_{KL}(s,\mathcal{N}(0,1)) + \mathcal{L}_{KL}(d,\mathcal{N}(0,1))\big) + \lambda_{rec}\mathcal{L}_{rec}(X,\hat{X}) + \lambda_{HSIC}\mathcal{L}_{HSIC}(s,d) + \lambda_{cls}\mathcal{L}_{cls}(c,\hat{c}).
$$

Cada una cumple un rol: las **KL divergences** inducen un prior gaussiano $\mathcal{N}(0,1)$ sobre $s$ y $d$ para robustez en dominios no vistos; **HSIC** (Hilbert-Schmidt Independence Criterion) fuerza a $s$ y $d$ a ser independientes entre sí; la **clasificación de dominio** empuja a $d$ a capturar la información específica del centro (la etiqueta de centro $c$ es metadata gratuita, "sabemos a qué centro pertenece el dato"); la **reconstrucción** $\ell_1$ obliga a que $(Z, s, d)$ contengan información suficiente; y el **rank loss** $\mathcal{L}_{rank} = \sigma_{m+1}$ limpia a $Z$ de información de dominio.

El rank loss merece detalle. La intuición: $Z$ debe codificar **solo información globalmente compartida** entre los dominios meta-train, no rasgos idiosincráticos de uno. Aplanando y concatenando las features de un batch multi-dominio se forma una matriz $\mathbf{Z}$, y forzar su **rango a ser $m$** (el número de clases de segmentación) se logra **minimizando el $(m+1)$-ésimo valor singular** $\sigma_{m+1}$. Anularlo es una relajación continua y diferenciable de "rango $\le m$": si $\sigma_{m+1} \rightarrow 0$, la matriz tiene rango efectivo $\le m$.

El argumento de semi-supervisión es directo: los datos etiquetados activan $\mathcal{L}_{Dice} + \mathcal{L}_{DT}$, mientras los datos sin etiqueta activan solo las partes de $\mathcal{L}_{DT}$ (todas, porque ninguna requiere $Y$). Así **cada imagen sin anotar mejora las representaciones $Z, s, d$**, y la simulación del shift se hace sobre la distribución completa del centro, no solo sobre las pocas muestras etiquetadas.

El bucle de meta-learning acopla ambos pasos. El **inner-loop** (meta-train) da un paso de gradiente sobre $\mathcal{D}_{tr}$ produciendo parámetros temporales:

$$
(\psi', \theta') = (\psi, \theta) - \alpha \nabla_{\psi,\theta} \mathcal{L}_{meta\text{-}train}(\mathcal{D}_{tr}; \psi, \theta).
$$

El **outer-loop** (meta-test) evalúa esos parámetros actualizados en un dominio dejado fuera $\mathcal{D}_{te}$, pero los gradientes fluyen hacia los parámetros originales (el segundo orden característico de MAML). El objetivo global:

$$
\arg\min_{\psi, \theta} \; \mathcal{L}_{meta\text{-}train}(\mathcal{D}_{tr}; \psi, \theta) + \mathcal{L}_{meta\text{-}test}(\mathcal{D}_{te}; \psi', \theta').
$$

Un detalle de ingeniería honesto: incluir KL y HSIC en el meta-test lo hacía "aún más inestable, llevando incluso al colapso del modelo", así que la pérdida meta-test se simplifica a un subconjunto seguro ($\mathcal{L}_{Dice} + \mathcal{L}_{rec} + \mathcal{L}_{cls}$), apoyándose en que la calidad de disentanglement puede aproximarse vía la calidad de reconstrucción y la precisión de clasificación de dominio.

## Resultados: segmentación cardíaca M&Ms multi-centro y SCGM

Dos benchmarks reales de imagen médica multi-centro.

**M&Ms** (Multi-centre, Multi-vendor & Multi-disease cardiac segmentation): 320 sujetos escaneados en **6 centros clínicos de 3 países** con **4 fabricantes de resonancia** (Siemens, Philips, GE, Canon), agrupados en dominios A, B, C, D con distribución desbalanceada (95/125/50/50). **SCGM** (Spinal cord gray matter): 4 centros médicos con distintos sistemas MRI, cada dominio con 10 sujetos etiquetados + 10 sin etiquetar. El protocolo es **leave-one-domain-out**: entrenar con todos los dominios source menos uno y evaluar en el dejado fuera (el target no visto). Métricas: Dice (%) y Hausdorff Distance.

**M&Ms — Dice (%), con solo 2% de datos etiquetados:**

| Source → Target | nnUNet | SDNet+Aug. | LDDG | SAML | **Ours** |
|---|---|---|---|---|---|
| B,C,D → A | 52.87 | 54.48 | 59.47 | 56.31 | **66.01** |
| A,C,D → B | 64.63 | 67.81 | 56.16 | 56.32 | **72.72** |
| A,B,D → C | 72.97 | 76.46 | 68.21 | 75.70 | **77.54** |
| A,B,C → D | 73.27 | 74.35 | 68.56 | 69.94 | **75.14** |
| **Average** | 65.94 | 68.28 | 63.16 | 64.57 | **72.85** |

El método gana en **todas** las celdas, no solo en promedio (+4.57 puntos sobre el mejor baseline, SDNet+Aug.). El caso más dramático es A,C,D→B: 72.72% vs 67.81%. Los baselines totalmente supervisados se degradan severamente con poca etiqueta: LDDG cae a 63.16% promedio, **peor que el nnUNet ingenuo**. Con 5% de etiquetas, el promedio sube a **79.75%** (Ours) vs 77.47% (SDNet+Aug.). En Hausdorff Distance (menor es mejor), con 2% el promedio es **19.32** (Ours) vs 20.17 (SDNet+Aug.). Incluso con **100%** de etiquetas el método aún gana (86.03% vs 85.38% de LDDG), porque los datos no etiquetados —fases intermedias entre end-systole y end-diastole— siguen aportando.

**SCGM — Dice (%), con 20% de etiquetas:** promedio **79.58%** (Ours) vs 76.73% (SDNet+Aug.), 73.50% (SAML), 64.85% (nnUNet), 63.31% (LDDG). Casos destacados: 2,3,4→1 alcanza **87.45%** y 1,2,3→4 alcanza **87.96%**. Con 100% de etiquetas el promedio es **82.25%**, aún el mejor. En Hausdorff con 20%, **1.97** (Ours), el mejor.

Las **ablaciones** cuantifican cada pieza, midiendo disentanglement con Distance Correlation (DC, menor = mejor) en M&Ms 5%. Con el rank loss: DC = 0.19, Dice = 79.75%; sin él: DC sube a 0.22 y Dice baja a 78.54%. Quitar la clasificación de dominio es lo más dañino (Dice cae a 77.45%, −2.30); quitar HSIC baja a 77.86% (−1.89). Y las comparaciones con baselines funcionan como ablaciones del diseño: SDNet+Aug. (disentanglement sin meta-learning) queda 4.57 puntos abajo en M&Ms 2% → **el meta-learning aporta ≈4.5 puntos**; SAML (meta-learning sin disentanglement-por-reconstrucción) queda 8.28 puntos abajo, sobre todo porque no puede usar datos sin etiqueta.

El patrón clave: **la ventaja del método crece a medida que baja la fracción etiquetada** (de +0.65 puntos con 100% a +4.57 con 2% en M&Ms), exactamente donde está diseñado para brillar.

## Por qué importa: meta-learning como herramienta práctica de robustez multi-centro

Este paper reposiciona el meta-learning de su nicho académico (few-shot N-way K-shot sobre Omniglot/miniImageNet) hacia una **herramienta práctica de robustez** en imagen médica real. Cuatro puntos lo hacen relevante:

1. **Es el primer framework de DG que combina meta-learning con disentanglement en setting semi-supervisado**, cerrando el gap de escalabilidad que tenían los métodos de meta-learning supervisados en segmentación.
2. **Convierte datos crudos en valor.** Los datos sin anotar —que en un hospital se acumulan por flujo clínico normal— dejan de ser inútiles y contribuyen a la generalización vía las pérdidas no supervisadas. Esto cambia la economía de un proyecto de ML clínico: no hace falta anotar masivamente para generalizar.
3. **El multi-centro/multi-vendor es el escenario real de despliegue.** M&Ms con 4 fabricantes (incluido Canon, afiliación de los autores) y 6 centros refleja lo que enfrenta un modelo distribuido a una red hospitalaria. Que gane justo en el régimen de pocas etiquetas es lo que importa en producción.
4. **El disentanglement da interpretabilidad estructural.** Separar $Z$ (anatomía) de $s, d$ (apariencia) no es solo regularización: ofrece un modelo mental de qué causa el shift, útil para diagnosticar fallos de generalización.

Conviene leerlo con honestidad sobre sus límites: hay **muchas pérdidas que balancear** (pesos fijados por experimentación manual extensa), el **meta-test de segundo orden es inestable** (de ahí su simplificación), requiere **≥2 dominios source** (no aplica con un solo centro), y el meta-learning de segundo orden tiene **costo computacional** no trivial. Además, el propio paper recuerda (citando a Llera Montero et al.) que el disentanglement por sí solo no garantiza generalización: el método funciona por la **combinación de sesgos** —equivarianza espacial de $Z$ + rank loss + meta-learning—, no por el disentanglement aislado. La evaluación es 2D y en dos órganos; 3D y otras modalidades quedan abiertas.

## Conexión con la Clase 26 y con sistemas de salud multi-institucionales

En el contexto de la Clase 26 (aplicaciones de deep learning en medicina), este paper es un caso ejemplar de cómo una técnica de aprendizaje moderna —meta-learning + representaciones desacopladas— se aterriza en un problema clínico concreto: segmentación cardíaca y de médula espinal **generalizable entre centros**. No es un benchmark abstracto: M&Ms es un challenge real de MICCAI con datos de 6 hospitales y 4 fabricantes de escáner, y entre las afiliaciones de los autores figura un vendor real (Canon Medical). La lección de la clase: las técnicas que permiten que un modelo **sobreviva al cruzar la frontera entre instituciones** son meta-learning, semi-supervisión y representaciones invariantes.

El mapeo a sistemas de salud multi-institucionales es directo: **multi-centro = multi-hospital**. El problema central —un modelo entrenado en unos centros se degrada en un centro nuevo no visto— es exactamente el reto de desplegar un modelo clínico en otra institución, o de recibir un modelo externo y aplicarlo a datos propios. La distinción DA vs DG tiene un correlato operacional: DA corresponde al lujo de tener un dataset del hospital destino para calibrar; DG corresponde al caso realista y más exigente donde el modelo debe funcionar **desde el día cero** en un centro del que no se tienen datos.

Hay también un paralelo conceptual con la **interoperabilidad entre sistemas de información clínica**: así como el paper separa la anatomía $Z$ (invariante, lo que importa) de la apariencia de dominio $d$ (lo que varía por centro), un pipeline de integración entre sistemas busca separar el significado clínico —códigos canónicos, recursos normalizados— de las idiosincrasias del sistema emisor —perfiles locales, extensiones propietarias—. El objetivo es el mismo: extraer lo invariante y ser robusto a lo específico de la fuente. El playbook práctico que sugiere el paper para cualquier proyecto multi-institucional: (1) tratar cada centro como un dominio, (2) usar leave-one-domain-out para estimar la generalización **antes** de desplegar, (3) aprovechar el abundante dato sin anotar vía pérdidas auto-supervisadas, y (4) construir un sesgo arquitectónico que separe lo clínicamente invariante de lo específico del escáner o sistema.

## Notas y enlaces

- **Fundamentos:** [Meta-aprendizaje](/fundamentos/meta-aprendizaje), [Transfer learning](/fundamentos/transfer-learning), [Few-shot learning](/fundamentos/few-shot-learning).
- **Papers relacionados:** [MAML (Finn et al., 2017)](/papers/maml-finn-2017), [MetaSeg (Vyas et al., 2025)](/papers/metaseg-vyas-2025), [fMRI Denoising (Heo et al., 2025)](/papers/fmri-denoising-heo-2025).
- **Código oficial:** [github.com/vios-s/DGNet](https://github.com/vios-s/DGNet).
- Ver [Clase 26 -- Meta-aprendizaje](/clases/clase-26).
