# Masked Autoencoders As Spatiotemporal Learners — Análisis interno

## 1. Metadata y resumen ejecutivo

- **Título:** *Masked Autoencoders As Spatiotemporal Learners*.
- **Autores:** Christoph Feichtenhofer\*, Haoqi Fan\*, Yanghao Li, Kaiming He (\* contribución igual). Todos de **Meta AI, FAIR**.
- **Venue:** *36th Conference on Neural Information Processing Systems* (**NeurIPS 2022**).
- **Preprint:** arXiv:2205.09113v2 (18 may 2022; v2 del 21 oct 2022), [arxiv.org/abs/2205.09113](https://arxiv.org/abs/2205.09113).
- **Código:** [github.com/facebookresearch/mae_st](https://github.com/facebookresearch/mae_st) (`mae_st` = MAE *spatiotemporal*).

Este paper es la **extensión natural y deliberadamente minimalista de MAE (He et al., 2022) al dominio del video**. La tesis es de una simplicidad casi provocadora: no se necesita un método nuevo para aprender representaciones espaciotemporales autosupervisadas; basta con tratar el video como un conjunto de **parches espaciotemporales** (cubos de tamaño *t×16×16*), enmascarar al azar una fracción enorme de ellos y entrenar un autoencoder para reconstruir los píxeles faltantes. El mensaje de fondo —compartido con BERT (Devlin et al., 2019) y MAE— es que el **masked autoencoding puede ser una metodología unificada de aprendizaje de representaciones con conocimiento de dominio mínimo**: lenguaje, imágenes y video bajo el mismo marco, sin sesgos inductivos especializados.

El hallazgo empírico que vertebra el trabajo es que la **tasa óptima de enmascaramiento en video es 90%**, frente al 75% de imágenes y al 15% de texto en BERT. Esa escalera de tasas (15% → 75% → 90%) no es arbitraria: el paper la lee como un termómetro de la **redundancia de información** del medio. El lenguaje es denso en información (cada token cuenta), las imágenes naturales son redundantes espacialmente, y el video añade encima una fuerte **redundancia temporal** —los fotogramas consecutivos se parecen mucho— que permite tirar el 90% de los parches y aún así reconstruir. Esa tasa altísima no es solo una curiosidad: es el motor de eficiencia del método (>4× de aceleración en tiempo de pared).

Para la Clase 28 (Aprendizaje Autosupervisado) esto importa porque el temario tiene un slide explícito —"MAE en videos (Feichtenhofer et al. 2022)"— colocado **justo después** de la presentación de MAE para imágenes. El paper es, literalmente, el siguiente eslabón de la familia de masked autoencoders: demuestra que el principio generativo de "reconstruir lo enmascarado" se transporta de 2D a 2D+tiempo casi sin tocar la receta, y que la propiedad clave (alta tasa de máscara ligada a la redundancia) no solo se mantiene sino que se intensifica.

## 2. Contexto histórico: de DAE y BERT a MAE, y de ahí al video

El árbol genealógico que el paper invoca es preciso. Las raíces están en los **denoising autoencoders** (Vincent et al., 2008, 2010): aprender representaciones reconstruyendo una señal limpia a partir de una entrada corrompida, donde el enmascaramiento es un tipo particular de corrupción. **BERT** (2019) es, conceptualmente, masked autoencoding sobre tokens de lenguaje, y fue el éxito que legitimó la idea a gran escala. En visión, la línea de masked prediction con Transformers progresó por etapas: **iGPT** (Chen et al., 2020) entrenó Transformers sobre píxeles como tokens; **ViT** (Dosovitskiy et al., 2021) dio el salto revolucionario al usar *parches* como tokens y exploró la predicción enmascarada; y **MAE** (He et al., 2022) volvió a los fundamentos del autoencoding poniendo el foco en el **decoder**.

La aportación de diseño de MAE que este paper hereda casi sin cambios es el **encoder asimétrico solo-visible**: como existe un decoder con significado propio, el encoder puede operar *únicamente sobre los parches visibles* (no enmascarados), dejando que el decoder pequeño reconstruya el resto a partir de tokens de máscara. Con una tasa de máscara alta, eso reduce drásticamente el cómputo. MAE además mostró empíricamente que una **tasa de enmascaramiento alta es esencial** para imágenes (75%). Este trabajo continúa exactamente esa línea.

En el frente del **autosupervisado en video**, el campo había puesto el foco en la dimensión temporal con familias de métodos bien diferenciadas: coherencia temporal o *slowness* (Goroshin et al., 2015), predicción de futuro (Srivastava et al., 2015; Vondrick et al., 2016), movimiento de objetos (Pathak et al., 2017), ordenamiento temporal (Misra et al., 2016; Lee et al., 2017) y contraste espaciotemporal (Feichtenhofer et al., 2021; Qian et al., 2021). El método aquí presentado **también** se apoya en la coherencia temporal, pero de forma **implícita**: como es prácticamente agnóstico al espacio-tiempo, la única vía por la que explota esa coherencia es subiendo la tasa de máscara al 90%, lo que presupone que el video es más redundante que la imagen. Entre los métodos basados en enmascaramiento para video, los previos (VIMPAC de Tan et al., 2021; BEVT) se centraban en *tokenizar* el objetivo de predicción; este opera directamente sobre píxeles, sin tokenizer extra. El paper reconoce ser **concurrente e independiente de VideoMAE** (Tong et al., 2022), un método relacionado.

## 3. Contribución central

La contribución no es un mecanismo nuevo sino una **demostración**: que MAE aplicado al conjunto de parches espaciotemporales —"in a nutshell, simply MAE applied to the set of spacetime patches"— produce representaciones de video muy fuertes con sesgos inductivos mínimos. Los pilares concretos:

1. **Enmascaramiento espaciotemporal con tasa del 90%.** Se muestrean parches al azar (sin reemplazo) de la rejilla *T×H×W* de tubelets. La tasa óptima sube de 75% (imagen) a **90%** (video), e incluso 95% rinde sorprendentemente bien. El paper lo justifica con un experimento mental: si un video tuviera *T* fotogramas idénticos, muestrear *1/T* de los parches ya revelaría casi todo el fotograma estático; como en videos naturales el movimiento lento es más probable que el rápido, la tasa puede ser altísima.

2. **Muestreo agnóstico al espacio-tiempo (sin sesgos jerárquicos).** El muestreo aleatorio *no* respeta la estructura espaciotemporal (Fig. 4a), análogo a BERT en 1D y MAE en 2D. Esto **supera** a las alternativas estructuradas —space-only ("tube"), time-only ("frame") y block-wise ("cube")— porque estas, con tasas muy altas, dejan tareas o demasiado fáciles o imposibles (p. ej., time-only que conserva un solo fotograma exige "predecir pasado y futuro" desde una imagen).

3. **Encoder asimétrico solo-visible heredado de MAE.** Con 90% de máscara, el encoder ve <1/10 de los tokens. Como la auto-atención es cuadrática en el número de tokens, esto reduce el cómputo del encoder a <1/10 y, sumando un decoder pequeño, da una **reducción teórica de 7.7× en FLOPs** y un **speedup real de 4.1×** en tiempo de pared.

4. **Arquitectura agnóstica al espacio-tiempo.** Encoder y decoder son **ViT vanilla** sin factorización ni jerarquía, en contraste con los líderes especializados en video (ViViT, MViT, Video Swin). El único componente *spacetime-aware* es el embedding de parches y sus embeddings posicionales. El método predice píxeles, sin tokenizer específico del problema.

5. **Resultados sólidos y un mensaje de unificación.** MAE supera al pre-entrenamiento supervisado por márgenes grandes, y rinde bien incluso pre-entrenando con **video real no curado** de Instagram. La conclusión: el video autosupervisado puede abordarse como el lenguaje y las imágenes, bajo un marco unificado y con conocimiento de dominio mínimo.

## 4. Método

### 4.1. Patch embedding (el único componente con conocimiento de dominio)

Siguiendo ViT, el clip de video se divide en una rejilla regular de parches espaciotemporales no solapados. El tamaño de parche es **2×16×16** (temporal 2, espacial 16×16). Para una entrada de **16×224×224**, eso produce **8×14×14 = 1568 tokens**. Los parches se aplanan, se proyectan linealmente y se les suman **embeddings posicionales separables**: uno para el espacio y otro para el tiempo, cuya suma da el embedding espaciotemporal (esta separación evita que el tamaño del embedding posicional crezca demasiado en 3D). Este es —y el paper lo subraya— el **único proceso *spacetime-aware*** de todo el sistema.

### 4.2. Enmascaramiento

Se muestrean parches aleatorios sin reemplazo, de forma agnóstica a la estructura (Fig. 4a). Con la tasa del 90%, de los 1568 tokens solo **156 quedan visibles**. El paper compara cuatro estrategias de muestreo en la Tabla 2a (sobre Kinetics-400, ViT-L, 800 épocas):

| Estrategia | Tasa | Acc. K400 |
|---|---|---|
| **Agnóstica (random)** | **90%** | **84.4%** |
| Space-only ("tube") | 90% | 83.5% |
| Time-only ("frame") | 75% | 79.1% |
| Block-wise ("cube") | 75% | 83.2% |

El muestreo agnóstico gana porque aprovecha mejor el escaso número de parches visibles y tolera tasas más altas; las estrategias estructuradas, al concentrar la máscara, tienen óptimos a tasas más bajas. (Time-only con 87.5% cae a 75.4%, confirmando que predecir desde un único fotograma es demasiado difícil.)

### 4.3. Autoencoding

- **Encoder:** ViT vanilla aplicado **solo sobre los parches visibles** (siguiendo MAE). Reduce complejidad de tiempo y memoria a <1/10 con 90% de máscara.
- **Decoder:** otro ViT vanilla sobre la unión del conjunto codificado **más tokens de máscara**, con embeddings posicionales propios del decoder. Es deliberadamente **más pequeño** que el encoder (por defecto 512-d, 4 bloques, vs. encoder ViT-L de 1024-d, 24 bloques). Aunque procesa el conjunto completo, su costo por token es ~1/20 del encoder. En conjunto, el autoencoder logra una reducción de cómputo de **7.7×** frente a codificar todos los tokens.
- **Predicción:** en el espacio de píxeles. En principio podría predecir el cubo *t×16×16* completo, pero en la práctica basta predecir **un único corte temporal (16×16)** para mantener manejable la capa de predicción. Se predicen píxeles originales o **normalizados por parche** (estos últimos suben +0.6%).
- **Pérdida:** **MSE** entre predicción y objetivo, promediado sobre los parches desconocidos (igual que BERT/MAE).

Encoder y decoder son **agnósticos a la estructura espaciotemporal**: no hay jerarquía ni factorización; el método confía en la auto-atención **global** para aprender el conocimiento útil desde los datos.

### 4.4. Implementación

Entrada por defecto **16×224×224**, con 16 fotogramas muestreados a *stride* temporal 4 (muestreo 16×4) y fotograma inicial aleatorio. La única augmentación espacial relevante es *random resized cropping* con escala [0.5, 1] y *flip* horizontal; augmentaciones más fuertes o *color jitter* dañan (Tabla 2c). Como el pre-entrenamiento es tan rápido que **la carga de datos pasa a ser el cuello de botella**, se adopta **repeated sampling** (4 muestras por video cargado), que sube la velocidad de pared hasta 3.0× sin cambiar el número de muestras vistas. Optimizador AdamW, batch 512. La evaluación se hace por **fine-tuning end-to-end** (no linear probing), con inferencia multi-view K×3 (K=7 clips temporales × 3 vistas espaciales en Kinetics).

## 5. Experimentos

### 5.1. Rendimiento en Kinetics-400 (ablaciones, ViT-L)

El resultado estrella: con ViT-L vanilla, el pre-entrenamiento MAE de 800 épocas lleva la accuracy de **71.4% (desde cero) a 84.4%**, un salto **absoluto de +13.0%** (1-view: 60.7% → 73.4%, +12.7%). Esa brecha es mucho mayor que en imágenes (~3% en MAE original), lo que sugiere que el pre-entrenamiento MAE es **especialmente útil para video**. Además, el costo total se reduce: las 800 épocas de MAE toman 35.8 h y un fine-tuning corto (100 épocas, 16.3 h) ya da buena accuracy, frente a las 65.2 h que tarda en converger el entrenamiento desde cero.

**Eficiencia (Tabla 1, ViT-L, 800 épocas):**

| Encoder | FLOPs | Cómputo | Carga+cómputo |
|---|---|---|---|
| Denso (con [M]) | 627.5 G | 141.1 h | 147.5 h |
| **Sparse (sin [M])** | **81.0 G** | **24.5 h** | **35.8 h** |
| Ganancia | **7.7×** | 5.8× | **4.1×** |

El encoder *sparse* (solo-visible) reduce FLOPs 7.7×; el speedup de cómputo sería 5.8× si los datos ya estuvieran cargados, pero como la carga de video se vuelve el cuello de botella, el speedup real de pared es **4.1×** (se acercaría a 5.8× con GPUs más lentas que oculten la carga).

**Tasa de máscara (Fig. 6).** **90% es la mejor**; 95% rinde sorprendentemente bien y alcanza al 90% si se entrena suficiente. Las tasas bajas (75%, 50%) rinden **peor pese a ver más tokens y costar más** —el 75% óptimo en imágenes no es óptimo en video—, lo que respalda la hipótesis de mayor redundancia del video.

**Objetivo de reconstrucción (Tabla 2b):** píxeles sin normalizar 83.8%, **píxeles normalizados por parche 84.4%**, HOG 84.0%, token dVAE 83.8%. La reconstrucción de píxeles es autocontenida y evita el tokenizer dVAE externo (que además ralentiza 1.6×). **Capacidad del decoder (Tabla 2e/f):** a diferencia de imágenes, un decoder demasiado estrecho (128-d) o poco profundo (1 bloque) **degrada notablemente** —el video, más complejo, exige más capacidad de decodificación—, aunque el óptimo (512-d, 4 bloques) sigue siendo mucho menor que el encoder.

### 5.2. Influencia de los datos y datasets reales (Tabla 3, ViT-L, 1600 épocas)

Transferencia a K400, **AVA** (detección de acción) y **SSv2** (Something-Something v2):

- MAE en IN1K (imágenes) supera al supervisado en IN1K: 78.6% → **82.3%** en K400.
- MAE en **K400** (video) bate al supervisado en K400 por márgenes enormes: **+9.5% en AVA** (21.6 → 31.1) y **+16.4% en SSv2** (55.7 → 72.1). También supera a MAE-IN1K (+2.5% K400, +4.8% AVA, +6.5% SSv2): pre-entrenar en video es muy beneficioso para tareas de video.
- Más datos sin etiquetas (K600/K700) mejoran AVA y SSv2 de forma consistente.

**Datos reales no curados.** Pre-entrenar con **1 millón de videos de Instagram no curados (IG-uncurated)** rinde sorprendentemente bien: en AVA supera a K400/600/700 curados, y en SSv2 queda a la par de K700. La Tabla 4 muestra que IG-uncurated rinde casi igual que IG-curated pese a ser aleatorio y no relacionado con las clases de K400 —**comportamiento que NO se observa en métodos contrastivos**, donde la curación de datos importa mucho. Esto sugiere que MAE es robusto a la distribución de datos, un resultado raro y valioso para el aprendizaje no supervisado a escala real.

### 5.3. Comparaciones a nivel de sistema y video→imagen

En K400, AVA y SSv2, los resultados son **competitivos y cercanos a los líderes** (ViT-H alcanza 85.1% en K400 a 16×224²; 86.8% con K600 e *intermediate fine-tuning*), **siendo la única entrada líder basada en ViT vanilla** —los demás son jerárquicos o especializados en video. Como apunte final, el pre-entrenamiento de video transfiere a **imágenes**: "deflactando" los embeddings de parche (sumando en el tiempo), ViT-L pre-entrenado en K400/IG da 83.7%/84.1% en ImageNet-1K, mejor que entrenar desde cero (82.6%) aunque por debajo de MAE-IN1K (85.9%).

## 6. Limitaciones reconocidas

La conclusión es explícitamente modesta. (i) **Escala de datos:** lo explorado es **órdenes de magnitud menor** que las contrapartes de lenguaje (GPT-3, etc.); aunque el método mejora mucho la eficiencia, el **video de alta dimensión sigue siendo un reto mayor para escalar**. (ii) **Solo se reporta como señal inicial:** "we hope our study will provide initial signals for future research". (iii) Leído entre líneas: el método **no aporta un mecanismo temporal explícito** —la coherencia temporal se explota solo de forma implícita vía la alta tasa de máscara—, y al ser agnóstico a la estructura **cede algo de precisión** frente a arquitecturas jerárquicas especializadas a igual resolución, compensándolo con simplicidad y eficiencia. (iv) Predecir un único corte temporal del cubo (en vez del *tubelet* completo) es una concesión práctica, no necesariamente óptima.

## 7. Impacto: SSL generativo en video bajo un marco unificado

El valor histórico de este trabajo es triple. Primero, **completa la tríada del masked autoencoding** —lenguaje (BERT), imagen (MAE), video (este paper)— y con ello sostiene empíricamente la tesis de que un mismo principio generativo, "reconstruir lo enmascarado", sirve para los tres medios con conocimiento de dominio mínimo. Segundo, ofrece una **lectura cuantitativa de la redundancia** vía la tasa óptima de máscara (15% → 75% → 90%), una intuición que se volvió canónica para razonar sobre SSL generativo en cualquier modalidad. Tercero, su **eficiencia** (encoder solo-visible + 90% de máscara → 4.1× de speedup) hizo *práctico* el pre-entrenamiento autosupervisado de video a gran escala, un dominio notoriamente caro.

Junto con el trabajo concurrente **VideoMAE** (Tong et al., 2022), este paper definió el paradigma de los *video masked autoencoders* que dominó el SSL de video posterior, desplazando en buena medida a los métodos contrastivos para esta modalidad —en parte gracias a su robustez a datos no curados, donde el contrastivo sufría. Para la salud y otros dominios donde la augmentación es difícil o inválida (imagen médica, hiperespectral, sensado remoto, datos geométricos y sus extensiones temporales), el propio paper señala que su naturaleza generativa y poco dependiente de augmentación lo hace **generalizable** —un puente directo desde el SSL académico hacia aplicaciones reales.

## 8. Conexión con la Clase 28 (Aprendizaje Autosupervisado)

El temario de la Clase 28 coloca el slide **"MAE en videos (Feichtenhofer et al. 2022)" inmediatamente después** de la presentación de MAE para imágenes, y esa secuencia es exactamente la narrativa del paper: **extender la familia de masked autoencoders al dominio temporal**. La lección pedagógica central es que pasar de imagen a video **no requirió un método nuevo** —solo cambiar la rejilla de parches 2D por tubelets 3D y subir la tasa de máscara—, lo que ilustra de forma nítida la idea de un *marco unificado de SSL generativo* con sesgos inductivos mínimos.

Tres puentes concretos para la clase:

- **La tasa de máscara como medidor de redundancia.** El salto 75% → 90% es el ejemplo más limpio del curso para enseñar *por qué* el masked autoencoding funciona: cuanto más redundante el medio, más se puede enmascarar y más difícil (y útil) se vuelve la tarea de pretexto. Esto conecta directamente con el fundamento de [aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado).
- **Continuidad con MAE de imágenes.** El paper reutiliza casi literalmente la receta de [He et al. (MAE), 2022](/papers/he-mae-2022): encoder asimétrico solo-visible, decoder pequeño, predicción de píxeles normalizados, evaluación por fine-tuning. Estudiar ambos en orden muestra qué se mantiene (la arquitectura y la asimetría) y qué cambia (la tasa, el embedding separable espacio/tiempo, la necesidad de un decoder algo mayor).
- **SSL generativo vs. contrastivo en video.** La robustez de MAE a datos de Instagram no curados —donde el contrastivo se degrada— es un argumento empírico fuerte a favor de los métodos generativos para datos reales a escala, tema medular de la [Clase 28](/clases/clase-28).
