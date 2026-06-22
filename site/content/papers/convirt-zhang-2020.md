---
title: "ConVIRT: Contrastive Learning of Medical Visual Representations (2020)"
weight: 323
math: true
---

{{< paper-card
    title="Contrastive Learning of Medical Visual Representations from Paired Images and Text"
    authors="Yuhao Zhang, Hang Jiang, Yasuhide Miura, Christopher D. Manning, Curtis P. Langlotz"
    year="2020"
    venue="MLHC 2022 / arXiv"
    pdf="/papers/convirt-zhang-2020.pdf"
    arxiv="2010.00747" >}}
Método de Stanford (octubre 2020) que aprende representaciones de imágenes médicas **sin una sola etiqueta manual**, contrastando cada radiografía contra el **reporte clínico** que el radiólogo ya escribió y que vive gratis en el sistema del hospital. Su tesis incómoda: en medicina, inicializar con ImageNet es subóptimo y el autosupervisado solo-imagen (SimCLR, MoCo) apenas ayuda, porque las radiografías tienen **altísima similitud inter-clase**. La señal que falta no está en otra augmentación de la imagen: está en el texto. Resultado estrella: ConVIRT iguala o supera a ImageNet con solo el **10% de las etiquetas** (y con el **1%** en evaluación lineal en tres de cuatro tareas). Es el **precursor directo de CLIP** —OpenAI lo cita y describe su propio método como "una versión simplificada de ConVIRT".
{{< /paper-card >}}

---

## Contexto: etiquetar imágenes médicas es el cuello de botella

El deep learning ha mostrado promesa real en imagen médica (retinopatía diabética, cáncer de piel, enfermedad retinal), pero choca con un freno igualmente real: la **escasez extrema de anotaciones**. Históricamente había dos caminos para obtener etiquetas, y ambos son malos:

1. **Anotaciones de expertos.** Alta calidad, pero carísimas: requieren el tiempo de un radiólogo certificado. El costo hace que los datasets médicos sean órdenes de magnitud más pequeños que ImageNet. Para llenar ese hueco todos transfieren pesos de ImageNet, algo que Raghu et al. (2019) mostraron que a menudo no aporta beneficio frente a inicialización aleatoria —la imagen médica exige features de grano muy fino, muy distintos de los de "identificar objetos".
2. **Reglas para extraer etiquetas del reporte.** El texto del radiólogo se produce gratis en el flujo de trabajo, así que se escriben reglas que lo parsean y emiten una etiqueta (p. ej. el CheXpert labeler). Da datasets más grandes, pero las reglas son inexactas, se limitan a unas pocas categorías (uso ineficiente del texto) y son frágiles: dependen del estilo de redacción y no generalizan entre instituciones.

ConVIRT propone un tercer camino que combina lo mejor de ambos: usa el **texto abundante y gratuito** del reporte, pero *no* lo convierte en etiquetas discretas vía reglas frágiles. Lo usa directamente como **señal de supervisión continua** mediante un objetivo contrastivo. El texto deja de ser fuente de etiquetas ruidosas y pasa a ser el otro extremo de un alineamiento multimodal. El insight es generalizable a todo el ML en salud: los datos clínicos son escasos y caros de anotar, pero **naturalmente vienen emparejados con features multimodales** (texto descriptivo, metadata del paciente). Reutilizar esa información gratuita es la palanca para construir modelos confiables a bajo costo de anotación.

## Contribución central

ConVIRT (**Con**trastive **VI**sual **R**epresentation Learning from **T**ext) aprende un encoder de imagen médica **maximizando el acuerdo entre pares imagen-texto verdaderos frente a pares aleatorios**, mediante un objetivo contrastivo bidireccional entre modalidades, sin input experto adicional y de forma agnóstica a la especialidad. Tres rasgos lo distinguen:

- **Frente al [autosupervisado](/fundamentos/aprendizaje-autosupervisado) solo-imagen (SimCLR, MoCo):** en vez de contrastar dos vistas augmentadas de *la misma imagen*, ConVIRT contrasta la imagen contra su *texto* emparejado. La semántica que añade el texto es lo que rompe la barrera de la alta similitud inter-clase de las radiografías.
- **Frente a captioning (CNN-LSTM, CNN-Transformer):** no genera texto. Captioning fuerza a decodificar el reporte completo palabra a palabra —tarea costosa que aprende detalles irrelevantes—; ConVIRT solo pide alinear representaciones globales, señal más limpia para el encoder visual.
- **Frente al pretraining visual-lingüístico con cabeza binaria** (predicen "par real / par falso" con BCE): ConVIRT usa un objetivo [contrastivo](/fundamentos/aprendizaje-contrastivo) NCE basado en similitud coseno, que produce representaciones mejor alineadas entre modalidades —crítico para retrieval texto-imagen, donde la cabeza binaria fracasa.

## Método: dos torres y pérdida bidireccional

Se asume un par $(x_v, x_u)$, con $x_v$ una o varias imágenes y $x_u$ el texto que las describe. El pipeline es de **dos torres simétricas en estructura, asimétricas en datos**:

- **Rama de imagen:** de $x_v$ se extrae una vista aleatoria $\tilde{x}_v$ con transformación $t_v$; el encoder $f_v$ la lleva a $h_v$ y una proyección no lineal $g_v$ produce $v = g_v(f_v(\tilde{x}_v))$, con $v \in \mathbb{R}^d$.
- **Rama de texto:** de $x_u$ se muestrea un span $\tilde{x}_u$; un encoder $f_u$ y una proyección $g_u$ producen $u = g_u(f_u(\tilde{x}_u))$, con $u \in \mathbb{R}^d$.

Ambas proyecciones (redes de una capa oculta con ReLU, siguiendo a SimCLR) mandan las dos modalidades a un **mismo espacio $d$-dimensional**, donde vive el contraste. Como el contraste es entre modalidades distintas, es **asimétrico** y hay que definir las dos direcciones. La pérdida **imagen→texto** para el par $i$ tiene forma InfoNCE:

$$\ell_i^{(v\to u)} = -\log \frac{\exp(\langle v_i, u_i\rangle / \tau)}{\sum_{k=1}^{N} \exp(\langle v_i, u_k\rangle / \tau)}$$

donde $\langle v, u\rangle = v^\top u / \lVert v\rVert\lVert u\rVert$ es la similitud coseno y $\tau$ la temperatura. Es la log-loss de un clasificador $N$-vías que, dada la imagen $v_i$, debe elegir su texto verdadero $u_i$ entre los $N$ del batch. Por simetría se define la pérdida **texto→imagen** $\ell_i^{(u\to v)}$ (dado el texto, recuperar su imagen). La pérdida total combina ambas:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\Big(\lambda\,\ell_i^{(v\to u)} + (1-\lambda)\,\ell_i^{(u\to v)}\Big)$$

La **bidireccionalidad** ata el espacio en ambos sentidos y es la razón de que el retrieval texto-imagen funcione tan bien: la cabeza binaria de los baselines no impone esta alineación por similitud y por eso produce espacios desalineados.

**Realización concreta.** El framework es agnóstico, pero la instancia evaluada usa **ResNet50** como encoder de imagen y **BERT inicializado con ClinicalBERT** (preentrenado sobre notas de MIMIC) como encoder de texto —congelando embeddings y las primeras 6 capas, afinando las últimas 6. La augmentación de imagen combina cinco transformaciones, con un detalle de dominio clave: el color jittering **solo ajusta brillo y contraste**, no color, por la naturaleza monocroma de las imágenes médicas. La augmentación de texto muestrea **una oración** del reporte —suave a propósito, para preservar el significado.

## Experimentos: datos y resultados

**Preentrenamiento.** Dos encoders separados sobre **MIMIC-CXR v2** (tórax, ~217k pares imagen-texto) y una colección del Rhode Island Hospital (hueso musculoesquelético, 48k pares).

**Evaluación downstream.** Cuatro tareas de clasificación en dos especialidades, con protocolos *linear* (CNN congelada, mide calidad pura de los features) y *fine-tuning*, usando **1%, 10% y 100%** de las etiquetas:

- **RSNA** (~25k): neumonía sí/no, AUC.
- **CheXpert** (~220k): multi-etiqueta de 5 hallazgos, AUC.
- **COVIDx** (~14k): COVID-19 / neumonía no-COVID / normal, accuracy.
- **MURA** (~33k): anormalidad ósea sí/no, AUC.

> **Matiz importante.** RSNA 25k, CheXpert 220k, COVIDx 14k y MURA 33k son los datasets *downstream* sobre los que se *evalúa* el encoder, no los datos de *preentrenamiento* (MIMIC-CXR 217k + hueso 48k). El punto pedagógico es el mismo: hay volúmenes enormes de imágenes médicas, pero etiquetarlas todas es prohibitivo, y ConVIRT muestra que basta una fracción mínima si el encoder ya aprendió del texto.

También se evalúa retrieval imagen-imagen y texto-imagen zero-shot sobre un dataset propio anotado por radiólogo certificado (CheXpert 8×200, Precision@k).

**Clasificación.** ConVIRT gana el mejor resultado global en todos los settings lineales y en 10 de 11 de fine-tuning. Los números que sostienen la tesis de eficiencia de datos:

| Tarea (Linear) | ImageNet 100% | ConVIRT 1% | ConVIRT 10% |
|---|---|---|---|
| RSNA (AUC) | 86.9 | **90.7** | 91.7 |
| CheXpert (AUC) | 81.0 | **85.9** | 86.8 |
| MURA (AUC) | 79.0 | **81.2** | 85.1 |

ConVIRT con **1%** de etiquetas supera a ImageNet con **100%** en tres de las cuatro tareas (linear), y con **10%** iguala o supera a ImageNet-100% en fine-tuning en todas. Es un orden de magnitud menos anotación para igual o mejor rendimiento —y en un dominio donde cada etiqueta cuesta el tiempo de un especialista, esa es la diferencia entre un proyecto viable y uno imposible.

**Contra el autosupervisado solo-imagen.** La comparación crítica es contra SimCLR y MoCo v2 corridos sobre las *mismas* imágenes:

| Método | RSNA Linear 1% (AUC) | CheXpert Linear 1% (AUC) | Img-Img Prec@10 |
|---|---|---|---|
| ImageNet | 82.8 | 75.7 | 14.4 |
| SimCLR | 86.3 | 77.4 | 17.6 |
| MoCo v2 | 86.6 | 81.3 | 20.6 |
| **ConVIRT** | **90.7** | **85.9** | **42.9** |

El SSL solo-imagen aporta mejoras marginales sobre ImageNet; ConVIRT los supera ampliamente. Los **mapas de saliencia** lo explican: ImageNet se fija en features triviales, SimCLR dispersa el foco y MoCo se equivoca enfocando el corazón, mientras ConVIRT enfoca las regiones anatómicamente correctas (p. ej. la base del pulmón para atelectasia). En retrieval texto-imagen la brecha es brutal: **Prec@5 = 60.0** frente a **15.5** de la cabeza binaria. El hiperparámetro más sensible es la **temperatura $\tau$** (default 0.1); a diferencia del SSL solo-imagen, el batch size casi no afecta la clasificación.

## Limitaciones

La evaluación es exclusivamente radiológica (tórax y hueso), por lo que la generalización a otras modalidades (histopatología, fondo de ojo, dermatología) queda como conjetura. Depende de la existencia de pares imagen-reporte de calidad, que no todo sistema clínico produce de forma estructurada, y hereda los sesgos del corpus de reportes (estilo de redacción, sesgos demográficos del hospital de origen) sin auditarlos. Para un despliegue clínico real importan además la calibración, la robustez al *distribution shift* entre instituciones y la validación prospectiva —temas que el paper, centrado en calidad de representación, no aborda.

## Impacto y relevancia médica

ConVIRT es uno de esos papers cuya importancia se mide por lo que vino después. **[CLIP](/papers/clip-radford-2021)** (Radford et al., 2021) escaló su receta a 400M de pares web y la llamó explícitamente "una versión simplificada de ConVIRT"; ALIGN (Jia et al., 2021) hizo lo propio con 1.8B de pares ruidosos. La genealogía del paradigma image-text contrastivo que hoy fundamenta los modelos multimodales nace, literalmente, en un problema de radiología en Stanford. Dentro de la medicina, ConVIRT inauguró una línea fértil: GLoRIA (contraste local región-palabra), LoVT (tareas localizadas), PubMedCLIP (VQA médico) y aplicaciones a riesgo clínico sobre EHR longitudinales.

Para la [Clase 28](/clases/clase-28) (Aprendizaje Autosupervisado), ConVIRT es el ejemplo canónico de que **el SSL importa más donde etiquetar es caro**. En visión natural, el [autosupervisado](/fundamentos/aprendizaje-autosupervisado) ahorra costo; en medicina, *habilita* lo que de otro modo sería inviable. El mecanismo —contraste bidireccional InfoNCE entre dos torres— es el mismo [aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) que la clase presenta para imágenes naturales (SimCLR, MoCo), pero el segundo extremo del contraste deja de ser otra vista de la imagen y pasa a ser el texto gratuito del experto. Esa sustitución es lo que rompe la barrera de la alta similitud inter-clase que hace fracasar al SSL solo-imagen en radiografías.

Para quien trabaja en salud, la lección operativa es directa: los pares imagen-reporte —y, por extensión, cualquier modalidad clínica naturalmente emparejada (texto, metadata, multiómica)— son una mina de supervisión gratuita que evita el cuello de botella de la anotación experta. ConVIRT es la prueba de concepto de que reutilizar la información ya producida en el flujo de trabajo clínico es el camino más eficiente —y a veces el único viable— para construir modelos médicos confiables a bajo costo.

## Enlaces

- Sucesor general-domain a gran escala: [CLIP (Radford et al., 2021)](/papers/clip-radford-2021)
- Fundamento del mecanismo: [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo)
- Marco general: [Aprendizaje autosupervisado](/fundamentos/aprendizaje-autosupervisado)
- Clase que lo enmarca: [Clase 28 — Aprendizaje Autosupervisado](/clases/clase-28)
- Código y datos: [github.com/yuhaozhang/convirt](https://github.com/yuhaozhang/convirt)
