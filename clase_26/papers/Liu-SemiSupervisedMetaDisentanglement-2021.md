# Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation

> Análisis interno exhaustivo · Clase 26 (IA UC) · Aplicaciones de deep learning en medicina

---

## 1. Metadata y resumen ejecutivo

| Campo | Valor |
|---|---|
| **Título** | Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation |
| **Autores** | Xiao Liu, Spyridon Thermos, Alison O'Neil, Sotirios A. Tsaftaris |
| **Afiliaciones** | School of Engineering, University of Edinburgh; The Alan Turing Institute (London); Canon Medical Research Europe Ltd. (Edinburgh) |
| **Venue** | MICCAI 2021 (Oral) |
| **Preprint** | arXiv:2106.13292v3 [cs.CV], 1 Oct 2021 |
| **Código** | https://github.com/vios-s/DGNet (público) |
| **Palabras clave** | Domain generalisation · Disentanglement · Medical image segmentation |
| **Benchmarks** | M&Ms (segmentación cardíaca multi-centro/multi-vendor); SCGM (gray matter de médula espinal) |

**Resumen ejecutivo.** El paper ataca un problema crónico del despliegue de modelos de imagen médica: cuando un modelo entrenado en datos de unos centros clínicos se aplica a datos de un centro nuevo (escáner distinto, protocolo distinto, población distinta), su rendimiento cae. Esto es el *domain shift*. La rama que el paper aborda es la *domain generalisation* (DG): generalizar a un dominio target **no visto** y del que **no se dispone de ningún dato** en entrenamiento, lo que la diferencia del *domain adaptation* (DA), que sí accede a datos del target.

La contribución central es combinar dos ideas que antes vivían separadas:

1. **Meta-learning basado en gradiente** estilo MAML, que en cada iteración divide los dominios source en un conjunto *meta-train* y un conjunto *meta-test* para **simular el domain shift durante el entrenamiento**.
2. **Disentanglement de representaciones**: separar explícitamente las características anatómicas espaciales $Z$ (lo que importa para segmentar) de los factores de apariencia común $s$ y de apariencia específica del dominio $d$ (lo que causa el shift).

La clave práctica es que el disentanglement se aprende vía **reconstrucción de la imagen**, una tarea que **no requiere máscaras de segmentación anotadas píxel a píxel**. Esto convierte el método en **semi-supervisado**: los datos sin etiqueta (abundantes y baratos) contribuyen a través de pérdidas no supervisadas ($\mathcal{L}_{rec}$, $\mathcal{L}_{cls}$, KL, HSIC, rank), mientras que los pocos datos etiquetados aportan vía la pérdida supervisada $\mathcal{L}_{Dice}$. El resultado es que el modelo aproxima mejor el verdadero domain shift incluso en regímenes de muy pocos datos anotados (2%, 5%, 20%).

Los resultados son state-of-the-art en ambos benchmarks. En M&Ms con solo 2% de datos etiquetados, el Dice promedio sube a **72.85%** frente a **68.28%** del mejor baseline (SDNet+Aug.), una mejora de ≈5 puntos. En SCGM con 20% de etiquetas, **79.58%** frente a **76.73%**. Las ablaciones cuantifican el aporte de cada pieza: quitar el rank loss baja Dice de 79.75% a 78.54%; quitar la clasificación de dominio lo baja a 77.45%; quitar HSIC a 77.86%.

Para Roberto este paper es directamente transferible: *multi-centro = multi-hospital*. El reto de que un modelo entrenado en FALP funcione en otro centro con otro escáner es exactamente el problema de DG, y el paralelo con la interoperabilidad FHIR entre instituciones es nítido (sección 12).

---

## 2. El problema clínico: domain shift y escasez de anotaciones

### 2.1 Domain shift en imagen médica

El paper parte de un hecho empírico bien documentado en la comunidad: pese al progreso en segmentación médica con deep learning, *"inference performance on unseen datasets, acquired from distinct scanners or clinical centres, is known to decrease"*. La causa es el **domain shift**: el cambio en las estadísticas de los datos entre el dominio source (donde se entrenó) y el dominio target no visto.

Los autores descomponen las fuentes de variación en dos grandes familias, y esta descomposición no es decorativa: es la que justifica la arquitectura.

1. **Variación de población.** Género, edad, etnia difieren entre pacientes de distintas localizaciones. Esto impacta la **anatomía y patología subyacente** misma. Es decir, cambia el contenido que se quiere segmentar.
2. **Variación de escáner y protocolo de adquisición.** Distintos vendors de resonancia (Siemens, Philips, GE, Canon), distintos parámetros de adquisición. Esto impacta las **características de la imagen** (brillo, contraste) sin cambiar la anatomía.

Esta distinción es central. Para segmentar, **queremos ser sensibles a la anatomía e insensibles a la apariencia**, sea esta común a varios dominios o específica de uno. La arquitectura codifica esa intuición: $Z$ captura anatomía espacial, $s$ y $d$ capturan apariencia.

### 2.2 Por qué la anotación píxel-wise es el cuello de botella

El enfoque ingenuo ante el domain shift es *"acquire and label as many and diverse data as possible"*. El paper lo descarta de inmediato por las implicaciones de costo conocidas en la comunidad. La segmentación es el caso más caro de anotación supervisada: cada píxel debe etiquetarse, la tarea es laboriosa y **requiere conocimiento experto** (un radiólogo trazando contornos). A diferencia de la clasificación, donde una etiqueta por imagen basta, aquí la etiqueta es una máscara densa.

Esto crea una tensión directa con el meta-learning supervisado previo: *"the current fully supervised meta-learning approaches are not scalable for medical image segmentation"*. Y agrava un problema secundario: en un régimen de pocos datos, *"the simulated domain shifts may not approximate the true domain shifts well"*. Si solo tienes unas pocas muestras etiquetadas por centro, la distribución que el meta-learning ve es una caricatura empobrecida de la real, y los shifts simulados no representan los reales.

### 2.3 Por qué DG + semi-supervisión importan en el hospital

La síntesis del problema clínico: en la práctica hospitalaria real, cada centro aporta **muchos datos crudos sin anotar** (imágenes que ya se adquirieron por flujo clínico) y **pocos datos anotados** (lo que un experto alcanzó a trazar). Un método que (a) no necesite ver el centro target de antemano (DG) y (b) aproveche los datos sin etiqueta (semi-supervisión) está alineado con cómo realmente lucen los datos en producción. Este es el espacio de diseño que el paper ocupa.

---

## 3. Marco de domain generalization

### 3.1 La jerarquía de supuestos sobre el target

El paper ordena las alternativas según cuánta información del dominio target se asume disponible:

- **Recolectar y etiquetar todo** (el enfoque ingenuo): caro e inviable.
- **Domain adaptation (DA):** se entrena en dominios source para generalizar a un target *"with some information on the target domain available"*. El ejemplo que citan es la armonización cross-site de MRI, que fuerza a source y target a compartir características de imagen similares. **DA accede a datos del target** (aunque sea sin etiquetas).
- **Domain generalisation (DG):** *"a more strict alternative is to not use any information for the target domain"*. **No se usa nada del target.** Es más difícil pero más ampliamente aplicable. Este paper se enfoca aquí.

El contraste DA vs DG es la distinción conceptual clave. En DA el target existe en tiempo de entrenamiento (al menos sus imágenes); en DG el target es genuinamente desconocido hasta el deployment. Para un hospital que vende/comparte un modelo a una red de centros que aún no existen como datasets, DG es el supuesto realista.

### 3.2 El objetivo de DG y las direcciones de investigación

*"In domain generalisation, the overarching goal is to identify suitable representations that encode information about the task at hand whilst being insensitive to domain-specific information."* Es decir: aprender representaciones **task-informativas pero domain-invariantes**.

El paper enumera las direcciones activas:

1. **Augmentación directa** de los datos source (p. ej. deep stacked transformation).
2. **Regularización del espacio de features** (varias referencias).
3. **Alineamiento** de las features source o de las distribuciones de salida.
4. **Aprendizaje de features domain-invariantes vía meta-learning basado en gradiente.**

De estas, eligen la cuarta por una razón concreta: el meta-learning basado en gradiente tiene la ventaja de **no sobre-ajustarse a los dominios source dominantes**, los que aportan la mayor cantidad de datos del dataset de entrenamiento. En un dataset multi-centro desbalanceado (un centro grande, varios chicos), esto importa mucho.

### 3.3 El paradigma episódico

El meta-learning basado en gradiente explota un **paradigma de entrenamiento episódico**: en cada iteración se parten los dominios source en meta-train y meta-test. El modelo *"is trained to handle domain shift by simulating it during training"*. Trabajos previos introdujeron distintas restricciones en setting **totalmente supervisado**: alineamiento global de clases y clustering local de muestras (MASF/Dou et al.), restricciones shape-aware (SAML/Liu et al.), o simplemente el objetivo de la tarea (Li et al., extendido a segmentación médica por Khandelwal & Yushkevich).

El gap que el paper identifica: ninguno escala a segmentación médica por el costo de anotación, y en régimen de pocos datos aprenden invarianza desde una distribución sub-representada. La única excepción semi-supervisada previa (Sharifi-Noghabi et al.) usa pseudo-labels por clustering pero *"is not applicable to segmentation"*.

---

## 4. La idea central: meta-learning + disentanglement

La propuesta combina dos mecanismos complementarios.

### 4.1 Meta-learning estilo MAML para simular el shift

Igual que MAML, cada iteración se divide en una fase meta-train y una meta-test. La diferencia con el MAML clásico (few-shot por *tareas*) es que aquí los "episodios" se construyen por **dominios**: meta-train usa unos dominios source, meta-test usa otro dominio source distinto que en esa iteración hace de *proxy* del target no visto. Así el optimizador no solo busca parámetros que funcionen en meta-train, sino parámetros cuyas **futuras actualizaciones generalicen** a un dominio que no participó del gradiente de meta-train. Esto simula explícitamente el domain shift dentro del bucle de entrenamiento.

### 4.2 Disentanglement para modelar el shift, no solo eliminarlo

El giro conceptual del paper sobre el meta-learning previo: en vez de **eliminar implícitamente** la información del shift simulado (lo que hacían los métodos supervisados con sus constraints), aquí se **modela explícitamente** mediante disentanglement. Se aprenden representaciones *"complete and sufficient"* (en el sentido de Achille & Soatto) vía reconstrucción.

La consecuencia es doble:

- **Habilita semi-supervisión.** Reconstruir la imagen es no supervisado, así que *"we can better simulate the domain shifts by also using unlabeled data from any of the source domains"*. Los datos sin máscara contribuyen.
- **Mejora la aproximación del shift verdadero.** Al modelar $s$ y $d$ explícitamente con más datos (incluidos los no etiquetados), la simulación de shift dentro del meta-learning es más fiel.

### 4.3 La "triple" descomposición

El modelo separa la imagen en tres representaciones con roles bien definidos:

- $Z$: **features anatómicas espaciales** (grid-like). Equivariantes a la segmentación. Es lo que la red de tarea consume.
- $s$: vector que captura **características de imagen comunes** entre dominios.
- $d$: vector que captura **características de imagen específicas de cada dominio**.

El paper aclara la motivación de diseño: *"a spatial $Z$ is equivariant to segmentation and this has been shown to improve performance"*, y se fuerza a $Z$ a desacoplarse de $s$ y $d$ mediante **regularización low-rank**. Y de forma elegante, el meta-learning no solo empuja a $Z, s, d$ a generalizar a dominios no vistos, sino que *"at the same time improves (implicitly) their disentanglement"*. Es decir, las dos mitades del método se refuerzan mutuamente.

---

## 5. La arquitectura (Fig. 1)

### 5.1 Componentes

Sobre un dataset multi-dominio $\mathcal{D} = \{X_i^k, Y_i^k\}_{i=1}^{N_k}$, con $k \in \{1, \dots, K\}$ dominios source, $X_i^k$ el $i$-ésimo dato del dominio $k$ e $Y_i^k$ su máscara ground-truth, los bloques son:

- **Feature network $F_\psi : \mathcal{X} \rightarrow \mathcal{Z}$.** Extrae las features anatómicas $Z = F_\psi(X)$. En la implementación es una **2D U-Net** que produce $Z$ con **8 canales** del mismo alto y ancho que la imagen de entrada.
- **Task network $T_\theta : \mathcal{Z} \rightarrow \mathcal{Y}$.** Predice la máscara de segmentación $\hat{Y}$ a partir de $Z$. Sigue el diseño de SDNet.
- **Encoder de apariencia común $E_S$.** Produce $s = E_S(X)$, vector de **8 dimensiones**.
- **Encoder de dominio $E_D$.** Produce $d = E_D(X)$, también de **8 dimensiones**. Misma arquitectura que $E_S$.
- **Clasificador de dominio $T_C$.** Una **única capa fully-connected** que toma $d$ y predice la etiqueta de dominio $\hat{c}$ (a qué centro pertenece la imagen).
- **Decoder / reconstructor $DE$.** Combina $Z$, $s$ y $d$ para reconstruir la imagen: $\hat{X} = DE(Z, s, d)$.

### 5.2 El reconstructor y AdaIN

La pieza que hace funcionar el disentanglement es **cómo** el decoder combina las representaciones espacial y vectoriales. El paper usa **Adaptive Instance Normalization (AdaIN)**. Cada capa AdaIN opera (Ec. 6 del Appendix):

$$
\text{AdaIN}(Z_i, s_i, d_i) = \sigma(s_i, d_i) \cdot \frac{Z_i - \mu(Z_i)}{\sigma(Z_i)} + \mu(s_i, d_i),
$$

donde cada feature map $Z_i$ se normaliza primero por su propia media $\mu(Z_i)$ y desviación $\sigma(Z_i)$, y luego se **escala y desplaza** según escalares de media y desviación derivados de las representaciones de estilo $s_i$ y dominio $d_i$.

La razón de diseño: como muestra MUNIT (Huang et al.), AdaIN *"improves disentanglement and encourages $Z$ to encode spatially equivariant information, i.e. anatomical information useful for segmentation, and $s, d$ to only encode common or domain-specific appearance"*. Al inyectar la apariencia **solo** vía las estadísticas de normalización (que son globales/por-canal), la información espacial fina queda forzada a residir en $Z$, mientras $s$ y $d$ se quedan con el "tono" de la imagen. Es el sesgo arquitectónico que separa contenido de estilo.

---

## 6. Las funciones de pérdida

Esta es la sección que sostiene la afirmación de semi-supervisión. La distinción crucial: **una sola pérdida necesita máscaras ($\mathcal{L}_{Dice}$); todas las demás no**.

### 6.1 La pérdida supervisada: Dice

$\mathcal{L}_{Dice}(Y, \hat{Y})$ es la pérdida de tarea de segmentación. Requiere la máscara ground-truth $Y$. Solo los datos etiquetados la activan. Pesa con $\lambda_{Dice} = 5$ cuando hay datos etiquetados disponibles.

### 6.2 Las pérdidas no supervisadas que componen $\mathcal{L}_{DT}$

El término de disentanglement $\mathcal{L}_{DT}$ agrupa cinco pérdidas, **ninguna de las cuales necesita máscaras**:

1. **KL divergences.** $\mathcal{L}_{KL}(s, \mathcal{N}(0,1))$ y $\mathcal{L}_{KL}(d, \mathcal{N}(0,1))$ inducen un prior gaussiano $\mathcal{N}(0,1)$ sobre $s$ y $d$. El propósito (β-VAE, DIVA): que las representaciones vectoriales sean **robustas en dominios no vistos**.

2. **HSIC (Hilbert-Schmidt Independence Criterion).** $\mathcal{L}_{HSIC}(s, d)$ fuerza a $s$ y $d$ a ser **independientes entre sí**. Esto evita que la apariencia común y la específica de dominio se confundan.

3. **Clasificación de dominio.** $\mathcal{L}_{cls}(c, \hat{c})$ empuja a que la representación de dominio $d$ esté **altamente correlacionada con la información específica del dominio**. La etiqueta de dominio $c$ está siempre disponible *"as we know the centre where the data belong"* — no cuesta nada, es metadata.

4. **Reconstrucción.** $\mathcal{L}_{rec}(X, \hat{X})$, la **distancia $\ell_1$** entre la imagen original $X$ y su reconstrucción $\hat{X}$. Es el corazón del aprendizaje no supervisado: obliga a que $(Z, s, d)$ contengan información suficiente para reconstruir la entrada.

5. **Rank regularisation.** $\mathcal{L}_{rank} = \sigma_{m+1}$ (detallada abajo).

El término combinado:

$$
\mathcal{L}_{DT} = \lambda_{rank}\mathcal{L}_{rank}(Z) + \lambda_{KL}\big(\mathcal{L}_{KL}(s,\mathcal{N}(0,1)) + \mathcal{L}_{KL}(d,\mathcal{N}(0,1))\big) + \lambda_{rec}\mathcal{L}_{rec}(X,\hat{X}) + \lambda_{HSIC}\mathcal{L}_{HSIC}(s,d) + \lambda_{cls}\mathcal{L}_{cls}(c,\hat{c}).
$$

Hiperparámetros: $\lambda_{rank} = 0.1$, $\lambda_{KL} = 0.1$, $\lambda_{rec} = 1$, $\lambda_{cls} = 1$. El paper enfatiza: *"all the losses do not need ground-truth masks"*.

### 6.3 El rank loss en detalle

El rank loss merece desarrollo porque es una de las contribuciones declaradas (contribución 2). La intuición: $Z$ debe codificar **solo información globalmente compartida** entre los dominios meta-train, no rasgos idiosincráticos de un dominio.

El mecanismo: dado un batch $\{X_{i_1}^1, X_{i_2}^2, \dots, X_{i_{K_{tr}}}^{K_{tr}}\}$ de $K_{tr}$ dominios meta-train, se extraen sus features $\{Z_{i_1}^1, \dots, Z_{i_{K_{tr}}}^{K_{tr}}\}$. Aplanando y concatenando, se forma una matriz $\mathbf{Z}$ de dimensiones $[C, K_{tr} \times H \times W]$ (canales × resto). Forzar el **rango de $\mathbf{Z}$ a ser $m$** (el número de clases de segmentación) empuja a $Z$ a codificar solo lo globalmente compartido necesario para predecir la máscara. Esto se logra **minimizando el $(m+1)$-ésimo valor singular** $\sigma_{m+1}$ de $\mathbf{Z}$:

$$
\mathcal{L}_{rank} = \sigma_{m+1}.
$$

Anular el valor singular $m+1$ es una relajación continua y diferenciable de "rango $\le m$": si $\sigma_{m+1} \rightarrow 0$, la matriz efectivamente tiene rango $\le m$. Esto cumple doble función: **mejora la generalización** (heredada de LDDG) y **mejora el disentanglement** entre $Z$ y $(s, d)$.

### 6.4 Por qué esto habilita la semi-supervisión

El argumento es directo: en cada iteración, los datos etiquetados activan $\mathcal{L}_{Dice} + \mathcal{L}_{DT}$, mientras que los datos sin etiqueta activan solo las partes de $\mathcal{L}_{DT}$ que no requieren máscara (todas). Como $\mathcal{L}_{rec}$, KL, HSIC, rank y $\mathcal{L}_{cls}$ no necesitan $Y$, **cada imagen sin anotar mejora las representaciones $Z, s, d$**. Esto permite que la simulación del domain shift se haga sobre la distribución completa de datos del centro, no solo sobre las pocas muestras etiquetadas — exactamente lo que faltaba a los métodos supervisados previos.

---

## 7. Inner-loop / outer-loop del meta-learning

### 7.1 Meta-train step (inner-loop)

En cada iteración se construyen $\mathcal{D}_{tr}$ (meta-train) y $\mathcal{D}_{te}$ (meta-test) partiendo aleatoriamente los dominios source. El meta-train step optimiza los parámetros $\psi, \theta$ con los datos de $\mathcal{D}_{tr}$ mediante un paso de gradiente (inner-loop update):

$$
(\psi', \theta') = (\psi, \theta) - \alpha \nabla_{\psi,\theta} \mathcal{L}_{meta\text{-}train}(\mathcal{D}_{tr}; \psi, \theta),
$$

donde $\alpha$ es el learning rate del paso meta-train. Los parámetros actualizados $\psi', \theta'$ son **temporales**: cuantifican "qué pasaría si actualizo con meta-train".

El objetivo meta-train tiene dos componentes (Ec. 4):

$$
\mathcal{L}_{meta\text{-}train} = \lambda_{Dice}\mathcal{L}_{Dice}(Y, \hat{Y}) + \mathcal{L}_{DT}, \qquad \lambda_{Dice} = 5 \text{ (con datos etiquetados)}.
$$

### 7.2 Meta-test step (outer-loop)

El dominio meta-test $\mathcal{D}_{te}$ se procesa con los **parámetros actualizados** $(\psi', \theta')$. El modelo debe (1) predecir máscaras con precisión y (2) disentangle $Z$ de $(s, d)$ al mismo nivel que en meta-train. La pérdida meta-test se computa con $(\psi', \theta')$, pero **los gradientes se computan hacia los parámetros originales** $(\psi, \theta)$ — este es el segundo orden característico de MAML.

Un punto de ingeniería importante: el meta-test step es **inestable** porque sus gradientes son estadísticos de segundo orden de $\psi, \theta$. Los experimentos revelaron que incluir las pérdidas no supervisadas $\mathcal{L}_{KL}$ y $\mathcal{L}_{HSIC}$ en meta-test lo hace **aún más inestable, incluso llevando a colapso del modelo**. Además, se usa **un solo dominio para meta-test**, mientras que $\mathcal{L}_{rank}$ requiere múltiples dominios. Por eso, apoyándose en que *"the level of disentanglement can be proxied by the reconstruction quality and the domain classification accuracy"* (Locatello et al.), la pérdida meta-test se simplifica (Ec. 5):

$$
\mathcal{L}_{meta\text{-}test} = \lambda_{Dice}\mathcal{L}_{Dice}(Y, \hat{Y}) + \lambda_{rec}\mathcal{L}_{rec}(X, \hat{X}) + \lambda_{cls}\mathcal{L}_{cls}(c, \hat{c}).
$$

Para datos sin etiqueta, $\mathcal{L}_{rec}$ y $\mathcal{L}_{cls}$ siguen sin necesitar máscaras, así que el meta-test también es semi-supervisado.

### 7.3 El objetivo global

El objetivo final acopla ambos pasos (Ec. 2):

$$
\arg\min_{\psi, \theta} \; \mathcal{L}_{meta\text{-}train}(\mathcal{D}_{tr}; \psi, \theta) + \mathcal{L}_{meta\text{-}test}(\mathcal{D}_{te}; \psi', \theta').
$$

La intuición: *"the model should not only perform well on the source domains, but its future updates should also generalise well to unseen domains"*. Se acumulan los gradientes de ambos términos para actualizar $F_\psi$ y $T_\theta$. El primer término premia desempeño en meta-train; el segundo premia que la trayectoria de optimización (un paso adelante) generalice al dominio dejado fuera.

---

## 8. Experimentos

### 8.1 Datasets

**M&Ms (Multi-centre, Multi-vendor & Multi-disease cardiac segmentation).** 320 sujetos escaneados en **6 centros clínicos de 3 países** con **4 vendors de resonancia** (Siemens, Philips, GE, Canon) → dominios A, B, C, D. Solo se anotan las fases de end-systole y end-diastole por sujeto. Resoluciones de vóxel de $0.85 \times 0.85 \times 10$ mm a $1.45 \times 1.45 \times 9.9$ mm. Distribución: A = 95, B = 125, C = 50, D = 50 sujetos (desbalance que justifica la elección de meta-learning).

**SCGM (Spinal cord gray matter segmentation).** Datos de **4 centros médicos** con distintos sistemas MRI (Philips Achieva, Siemens Trio, Siemens Skyra) → dominios 1, 2, 3, 4. Resoluciones de $0.25 \times 0.25 \times 2.5$ mm a $0.5 \times 0.5 \times 5$ mm. Cada dominio: **10 sujetos etiquetados + 10 sin etiquetar**, perfecto para validar la semi-supervisión.

### 8.2 Protocolo leave-one-domain-out

El protocolo de DG por excelencia: se entrena con todos los dominios source menos uno, y se evalúa en el dominio dejado fuera (el target no visto). En M&Ms: B,C,D→A; A,C,D→B; A,B,D→C; A,B,C→D. Análogo en SCGM con 4 dominios. Métricas: **Dice (%)** y **Hausdorff Distance**.

### 8.3 Baselines

- **nnUNet** (Isensee et al., Nature Methods 2021): framework auto-configurante sobre U-Nets 2D/3D, **no orientado a DG**. Top performer en el challenge M&Ms con datos completos.
- **SDNet+Aug.** (Liu et al., STACOM 2020): disentanglement de anatomía espacial y modalidad no-espacial, aquí con augmentación de intensidad/resolución en setting semi-supervisado. **Tiene disentanglement pero NO meta-learning** → ablación natural del meta-learning.
- **LDDG** (Li et al., NeurIPS 2020): SOTA en DG médico, usa rank loss en setting totalmente supervisado.
- **SAML** (Liu et al., MICCAI 2020): meta-learning basado en gradiente con constraints shape-aware (compactness, smoothness) en setting totalmente supervisado. **Tiene meta-learning pero NO disentanglement con reconstrucción** → ablación natural del disentanglement.

### 8.4 Resultados numéricos reales

**M&Ms — Dice (%) (Tabla 1).** Con **2% de datos etiquetados** (todos los no etiquetados disponibles para SDNet+Aug. y el método propuesto):

| Source → Target | nnUNet | SDNet+Aug. | LDDG | SAML | **Ours** |
|---|---|---|---|---|---|
| B,C,D → A | 52.87 | 54.48 | 59.47 | 56.31 | **66.01** |
| A,C,D → B | 64.63 | 67.81 | 56.16 | 56.32 | **72.72** |
| A,B,D → C | 72.97 | 76.46 | 68.21 | 75.70 | **77.54** |
| A,B,C → D | 73.27 | 74.35 | 68.56 | 69.94 | **75.14** |
| **Average** | 65.94 | 68.28 | 63.16 | 64.57 | **72.85** |

Con **5% de datos etiquetados**, el promedio sube a **79.75%** (Ours) vs 77.47% (SDNet+Aug.), 76.09% (nnUNet), 74.88% (SAML), 71.29% (LDDG).

El método gana en **todas** las celdas de la tabla 2%, no solo en promedio. El caso más dramático es A,C,D→B: 72.72% vs el segundo mejor 67.81% (≈5 puntos). Los baselines de meta-learning supervisado (SAML) y DG supervisado (LDDG) **se degradan severamente con poca etiqueta**: LDDG cae a 63.16% promedio, peor que el nnUNet ingenuo.

**M&Ms — Hausdorff Distance (Tabla 2, menor es mejor).** Con 2%, promedio **19.32** (Ours) vs 20.17 (SDNet+Aug.), 20.96 (nnUNet), 21.22 (SAML), 22.02 (LDDG). Con 5%, **17.98** (Ours) — mejor promedio aunque la ventaja es más ajustada en esta métrica.

**SCGM — Dice (%) (Tabla 3).** Con **20% de etiquetas**, promedio **79.58%** (Ours) vs 76.73% (SDNet+Aug.), 73.50% (SAML), 64.85% (nnUNet), 63.31% (LDDG). Casos destacados: 2,3,4→1 alcanza **87.45%** (vs 83.07% SDNet+Aug.); 1,2,3→4 alcanza **87.96%**. Con **100%** de etiquetas, promedio **82.25%** (Ours), aún el mejor.

**SCGM — Hausdorff (Tabla 4).** Con 20%, promedio **1.97** (Ours), el mejor. Con 100%, **1.92**.

**M&Ms 100% (Appendix, Tabla A5).** Incluso con todas las etiquetas, el método gana: promedio **86.03%** vs 85.38% (LDDG), 84.87% (nnUNet). La mejora es de 0.65% porque los datos no etiquetados (fases entre end-systole y end-diastole de sujetos etiquetados) siguen aportando.

**Síntesis cuantitativa:** la mejora es de ≈5% Dice en M&Ms y ≈3% en SCGM en el régimen de pocos datos frente al mejor baseline — exactamente donde el método está diseñado para brillar.

---

## 9. Ablations

El estudio de ablación (sección 3.5) aísla el aporte de los componentes clave. Se omiten ablaciones sobre las KL porque trabajos previos (β-VAE, DIVA) ya muestran que el encoding variacional ayuda a la robustez.

### 9.1 Aporte del rank loss

Se usa **Distance Correlation (DC)** entre $Z$ y $(s, d)$ para medir disentanglement (DC menor = mejor disentanglement). En M&Ms 5%:

- **Con $\mathcal{L}_{rank}$:** DC promedio = **0.19**, Dice promedio = **79.75%**.
- **Sin $\mathcal{L}_{rank}$:** DC sube a **0.22** (peor disentanglement), Dice baja a **78.54%**.

El rank loss mejora simultáneamente el disentanglement (medido por DC) y el desempeño (Dice), confirmando la hipótesis de que forzar $Z$ a bajo rango lo limpia de información de dominio.

### 9.2 Aporte de la clasificación de dominio y HSIC

En M&Ms 5%, partiendo del modelo completo (Dice 79.75%):

- **Sin $\mathcal{L}_{cls}$:** Dice cae a **77.45%** (−2.30 puntos). Es la ablación con mayor impacto: sin forzar que $d$ capture la info de dominio, el disentanglement se degrada.
- **Sin $\mathcal{L}_{HSIC}$:** Dice cae a **77.86%** (−1.89 puntos). Sin forzar independencia entre $s$ y $d$, las representaciones se entremezclan.

### 9.3 Aporte del meta-learning vs disentanglement (vía baselines)

Las comparaciones de la sección 8 funcionan como ablaciones implícitas del diseño:

- **SDNet+Aug.** (disentanglement sin meta-learning) queda 4.57 puntos por debajo en M&Ms 2% → **el meta-learning aporta ≈4.5 puntos**.
- **SAML** (meta-learning sin disentanglement-por-reconstrucción) queda 8.28 puntos por debajo en M&Ms 2% → **el disentanglement semi-supervisado aporta más aún**, especialmente porque SAML no puede usar datos sin etiqueta.

El paper conecta esto con un hallazgo de la literatura (Llera Montero et al., ICLR 2021): *"without specific designs tuned to the tasks, disentanglement can not provide guaranteed generalisation ability"*. Es decir, el disentanglement por sí solo no basta — necesita el sesgo espacial de $Z$ (equivarianza) y el acople con meta-learning.

### 9.4 Efecto de la fracción de datos anotados

La estructura de las tablas (2% vs 5% en M&Ms; 20% vs 100% en SCGM) es en sí un barrido del régimen de anotación. El patrón clave: **la ventaja del método sobre los baselines crece a medida que baja la fracción etiquetada**. En M&Ms pasa de +0.65 puntos (100%) a +4.57 puntos (2%). Los baselines totalmente supervisados (SAML, LDDG) colapsan en regímenes bajos, mientras el método propuesto degrada con gracia gracias a los datos sin etiqueta.

---

## 10. Por qué importa

Este paper reposiciona el meta-learning de su nicho académico (few-shot, N-way K-shot sobre Omniglot/miniImageNet) hacia una **herramienta práctica de robustez** en imagen médica real. Las contribuciones que lo hacen relevante:

1. **Es el primer framework de DG que combina meta-learning con disentanglement en setting semi-supervisado** (contribución declarada 1). Cierra el gap de escalabilidad que tenían los métodos de meta-learning supervisados en segmentación.

2. **Convierte datos crudos en valor.** Los datos sin anotar — que en un hospital se acumulan por flujo clínico normal — dejan de ser inútiles y contribuyen a la generalización vía las pérdidas no supervisadas. Esto cambia la economía de un proyecto de ML clínico: no hace falta anotar masivamente para generalizar.

3. **El multi-centro/multi-vendor es el escenario real de deployment.** M&Ms con 4 vendors (incluido Canon, afiliación de los autores) y 6 centros refleja exactamente lo que enfrenta un modelo que se distribuye a una red hospitalaria. Que el método gane justo en el régimen de pocas etiquetas es lo que importa en producción, donde rara vez hay datos densamente anotados del centro nuevo.

4. **El disentanglement da interpretabilidad estructural.** Separar $Z$ (anatomía) de $s, d$ (apariencia) no es solo un truco de regularización: ofrece un modelo mental de qué causa el shift, lo que ayuda a diagnosticar fallos de generalización.

---

## 11. Limitaciones

El propio paper y la lectura crítica exponen varias limitaciones:

1. **Complejidad de entrenamiento: muchas pérdidas que balancear.** El objetivo combina Dice, rank, KL (×2), reconstrucción, HSIC y clasificación de dominio, cada una con su peso ($\lambda_{rank}=0.1$, $\lambda_{KL}=0.1$, $\lambda_{rec}=1$, $\lambda_{cls}=1$, $\lambda_{Dice}=5$). El paper admite que estos valores se fijaron *"according to our extensive early experiments"* — hay tuning manual considerable y la transferibilidad de estos pesos a otra tarea no está garantizada.

2. **Inestabilidad del meta-test de segundo orden.** Los autores reportan explícitamente que incluir KL y HSIC en meta-test *"make training even more unstable (even leading to model collapse)"*. Tuvieron que **simplificar la pérdida meta-test** (Ec. 5) a un subconjunto seguro. Esto delata fragilidad del optimizador de segundo orden.

3. **Dependencia de múltiples dominios source.** El esquema episódico necesita ≥2 dominios source para partir en meta-train/meta-test, y $\mathcal{L}_{rank}$ requiere múltiples dominios en el batch. Con un solo centro source el método no aplica. Además se usó **un solo dominio para meta-test**, lo que restringe la fidelidad de la simulación del shift.

4. **Costo computacional del meta-learning de segundo orden.** Computar gradientes a través de la actualización $(\psi', \theta')$ implica derivadas de segundo orden (o aproximaciones), más caras que el entrenamiento estándar. El entrenamiento corre 50K iteraciones, batch 4, en una sola NVidia 2080 Ti — viable pero no trivial.

5. **El disentanglement no garantiza generalización por sí mismo.** El propio paper cita a Llera Montero et al.: sin diseños específicos a la tarea, el disentanglement no asegura generalización. El método funciona por la combinación de sesgos (equivarianza espacial + rank + meta-learning), no por el disentanglement aislado.

6. **Evaluación 2D y en dos órganos.** $F_\psi$ es una U-Net **2D**; la validación es cardíaca y de médula espinal. La extensión a 3D y a otras modalidades/órganos queda abierta. El paper sugiere como trabajo futuro usar datos sin etiqueta **de otros dominios** para mejorar más.

---

## 12. Conexión con la Clase 26 y con el trabajo de Roberto

### 12.1 En el contexto de la Clase 26 (aplicaciones en medicina)

Este paper es un caso ejemplar de la slide de "aplicaciones en medicina": muestra cómo una técnica de aprendizaje (meta-learning + disentanglement) se aterriza en un problema clínico concreto (segmentación cardíaca y de médula espinal generalizable entre centros). No es un benchmark académico abstracto: M&Ms es un challenge real de MICCAI con datos de 6 hospitales y 4 fabricantes de escáner, y la afiliación de los autores incluye Canon Medical, un vendor real. La lección de la clase: las técnicas de ML moderno (meta-learning, representaciones desacopladas, semi-supervisión) son las que permiten que un modelo sobreviva al cruzar la frontera entre instituciones.

### 12.2 Multi-centro = multi-hospital: el paralelo directo con Roberto

Para Roberto el mapeo es uno a uno. El problema central del paper — *un modelo entrenado en unos centros se degrada en un centro nuevo no visto* — es exactamente el reto de desplegar un modelo clínico de FALP en otra institución, o de recibir un modelo externo y aplicarlo a datos propios. La domain generalisation es la formalización del requisito "que funcione fuera de casa sin reentrenar con datos del destino".

Más aún: la **distinción DA vs DG** tiene un correlato operacional. DA (acceso a datos del target) corresponde al lujo de tener un dataset del hospital destino para calibrar; DG (sin datos del target) corresponde al caso realista y más exigente donde el modelo debe funcionar *desde el día cero* en un centro del que no se tienen datos. En oncología multi-institucional, DG es el supuesto que importa.

### 12.3 El paralelo con la interoperabilidad FHIR

Hay una analogía conceptual fértil entre lo que hace este paper y el trabajo de interoperabilidad de Roberto:

- **El domain shift de imagen ≈ la heterogeneidad de datos entre sistemas FHIR.** Distintos escáneres/protocolos producen imágenes con estadísticas distintas; distintos sistemas EHR producen recursos FHIR con perfiles, terminologías y extensiones distintas. En ambos casos el "contenido clínico" es el mismo pero la "representación" varía por origen.
- **El disentanglement ≈ la separación entre contenido semántico y representación específica del sistema.** Así como el paper separa la anatomía $Z$ (invariante, lo que importa) de la apariencia de dominio $d$ (lo que varía por centro), la interoperabilidad busca separar el significado clínico (códigos canónicos, recursos normalizados) de las idiosincrasias del sistema emisor (perfiles locales, slicing, extensiones propietarias). El objetivo es el mismo: extraer lo invariante y ser robusto a lo específico de la fuente.
- **La generalización entre instituciones es el problema común.** Tanto un modelo de ML como un pipeline de integración FHIR enfrentan el mismo reto al cruzar la frontera institucional: lo que funcionó en el centro de origen no necesariamente funciona en el destino sin un mecanismo explícito de invarianza/normalización. El paper aporta una estrategia (modelar y desacoplar la fuente de variación, aprovechar datos sin etiquetar) que tiene eco directo en cómo se diseñan capas de armonización entre sistemas heterogéneos.

La transferencia práctica para un proyecto en FALP: si se entrena un segmentador o clasificador sobre imágenes/datos propios y se quiere que sobreviva al despliegue en otra institución, el patrón del paper es el playbook — (1) tratar cada centro como un dominio, (2) usar leave-one-domain-out para estimar la generalización antes de desplegar, (3) aprovechar el abundante dato sin anotar vía pérdidas auto-supervisadas, y (4) construir un sesgo arquitectónico que separe lo clínicamente invariante de lo específico del escáner/sistema.

---

*Fin del análisis interno.*
