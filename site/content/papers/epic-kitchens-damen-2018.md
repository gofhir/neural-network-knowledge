---
title: "EPIC-KITCHENS: visión egocéntrica a escala (2018)"
weight: 400
math: true
---

{{< paper-card
    title="Scaling Egocentric Vision: The EPIC-KITCHENS Dataset"
    authors="Dima Damen et al."
    year="2018"
    venue="ECCV 2018 / arXiv:1804.02748"
    pdf="/papers/epic-kitchens-damen-2018.pdf" >}}
EPIC-KITCHENS no propone un modelo: propone un **dataset**, y con él redefine la escala del análisis de video egocéntrico. Es un benchmark de **video en primera persona** grabado por **32 participantes** con una GoPro montada en la cabeza, en sus propias cocinas, en 4 ciudades de Norteamérica y Europa y con 10 nacionalidades distintas. Reúne **55 horas de video** (**11,5 millones de frames**) de grabación **untrimmed** —continua, sin recortar, con multitasking natural—, densamente anotada con **39.564 segmentos de acción** y **454.255 bounding boxes** de objetos. Su idea metodológica más original es la **anotación narrada**: los propios participantes narran en voz alta sus acciones tras grabar, y esa narración se transcribe y alinea en etiquetas verbo-sustantivo. Sobre ese material define tres desafíos con leaderboards: **detección** de objetos, **reconocimiento** de acciones y **anticipación** de acciones. Es el ejemplo canónico de por qué el video realista es difícil en la [Clase 36](/clases/clase-36).
{{< /paper-card >}}

---

## Contexto: egocéntrico vs. tercera persona, untrimmed vs. trimmed

La **visión egocéntrica** (o en primera persona) es el video capturado por una cámara **wearable** —montada en la cabeza o el pecho— que registra el mundo tal como lo ve quien la lleva. Se contrapone a la visión en **tercera persona**, donde una cámara externa observa a los sujetos desde afuera. La diferencia no es solo de ángulo: el punto de vista egocéntrico ofrece una perspectiva única sobre la **interacción persona-objeto, la atención e incluso la intención**. El paper argumenta que el progreso en este dominio había sido lento por una razón concreta —la **falta de datasets grandes**—: los corpus egocéntricos eran mucho más pequeños que sus equivalentes en tercera persona y solían capturarse en un único entorno.

El segundo eje es **untrimmed vs. trimmed**. La mayoría de los benchmarks de acciones contenían clips **muy cortos** (unos segundos) centrados en **una sola acción** ya recortada: alguien decidió dónde empieza y termina. EPIC-KITCHENS hace lo contrario: a cada participante se le pidió empezar a grabar cada vez que entraba a la cocina, produciendo video **untrimmed** donde las acciones están encadenadas y superpuestas —el **multitasking natural** de lavar unos platos en medio de la cocción, buscar un utensilio o cambiar de idea—. Esto lo hace a la vez más realista y más difícil. El paper contrasta explícitamente con datasets **scriptados** como Charades (actores siguiendo un guion), cuyas acciones lucen poco naturales; todos los datasets egocéntricos previos usaban actividades ensayadas. EPIC-KITCHENS captura actividad **no scriptada** de la vida diaria.

## Contribución y método: el actor como anotador

La contribución es haber construido el **dataset egocéntrico más grande y variado** de su momento, capturado en los entornos nativos de los participantes. Frente al dataset egocéntrico previo más cercano (ADL), multiplica por **90** los segmentos de acción y por **4** las bounding boxes. La diversidad es genuina: 32 participantes implican 32 cocinas distintas, lo que permite —por primera vez— evaluar rigurosamente la **generalización a entornos nuevos**, separando cocinas *vistas* de *no vistas*.

El pipeline de anotación arranca con **narración del propio actor**. Como hacer *crowd-sourcing* sobre videos tan largos es costoso, tras terminar de grabar cada persona veía sus videos y **narraba en voz alta** las acciones (un "comentario en vivo"), en su idioma nativo —5 idiomas en total—. Esa decisión responde a que ellos son los más calificados para etiquetar: fueron quienes ejecutaron la acción, y la narración refleja la **intención verdadera**, algo que un observador externo no recupera. Las narraciones se transcribieron manualmente vía AMT (las APIs automáticas fallaron: esperan oraciones completas) y se convirtieron en **125 clases de verbos** y **331 clases de sustantivos**. Para cada frase se ajustan los tiempos $A_i = [t^s_i, t^e_i]$ con $K_a = 4$ anotadores, midiendo acuerdo como el IoU promedio:

$$\alpha_i(j) = \frac{1}{K_a} \sum_{k=1}^{K_a} \mathrm{IoU}\big(A_i(j), A_i(k)\big)$$

Los autores son honestos sobre las narraciones como ground-truth: son **incompletas** (los participantes fueron selectivos, etiquetando más *abrir* que *cerrar*), están **retrasadas** temporalmente y usan **vocabulario libre**. Ver [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) para el problema general de fondo.

## Resultados: tres desafíos, dos particiones

Se reservó el ground-truth del 27 % para test, con dos particiones: **cocinas vistas (S1)**, donde cada cocina aparece en train y test, y **cocinas no vistas (S2)**, donde 4 cocinas completas se reservan para test. Sobre ellas se definen tres benchmarks:

- **Detección de objetos.** Baseline Faster R-CNN (ResNet-101 preentrenada en MS-COCO). Los objetos de EPIC-KITCHENS son más difíciles de detectar que en datasets existentes (mAP a IoU $>0{,}5$ por debajo del 40 %), pero el desempeño en cocinas vistas y no vistas es **comparable**: buena generalización entre entornos para los objetos.
- **Reconocimiento de acciones.** Baseline Temporal Segment Network (TSN) con streams RGB y de flujo óptico, prediciendo el par $(c_v, c_n)$. Acertar **verbo y sustantivo a la vez** es difícil: top-1 de acción de **20,5 %** en cocinas vistas y **10,9 %** en no vistas. Generalizar a entornos nuevos es más difícil para las acciones que para los objetos.
- **Anticipación de acciones.** Pronosticar la próxima acción antes de que ocurra observando $[t^s_i - (\tau_a + \tau_o),\, t^s_i - \tau_a]$. El desempeño cae respecto del reconocimiento y el modelo tiende a **sobre-predecir "put"**: una vez que un objeto se levanta, aprende a creer que lo siguiente será dejarlo.

## Limitaciones

- **Narraciones incompletas y sesgadas**, lo que obliga a evaluar solo las acciones narradas.
- **Solo objetos activos.** Las cajas cubren únicamente objetos involucrados en la interacción, no todos los presentes en la escena.
- **Vocabulario libre agrupado a mano**, tras fallar el clustering automático (WordNet/Word2Vec/Lesk) por falta de contexto.
- **Actividad de una sola persona.** Se pidió a los participantes estar solos, lo que **excluye interacciones interpersonales**.
- **Baselines lejos de resolver las tareas**, especialmente en cocinas no vistas y en anticipación: confirma la dificultad del dataset.

## Por qué importa para la Clase 36

La [Clase 36](/clases/clase-36) introduce el [análisis de video](/dominios/video) como dominio con desafíos propios —reconocimiento, detección, anticipación, adaptación de dominio— en video trimmed y untrimmed. EPIC-KITCHENS es el ejemplo canónico de **por qué el análisis de video realista es difícil** y aporta dos ejes:

- **La perspectiva egocéntrica cambia el problema.** El video en primera persona trae oclusiones frecuentes por las manos, movimiento de cámara acoplado a la cabeza, objetos que entran y salen del campo de visión, y una relación estrecha entre lo que se ve y la **intención** de quien actúa. El [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones) egocéntrico es un subcampo con dinámicas distintas al de tercera persona.
- **Untrimmed es el escenario real.** Los clips cortos ocultan la parte difícil —¿dónde empieza y termina cada acción en un flujo continuo?—. EPIC-KITCHENS conserva la grabación completa, conectando con las tareas de **localización temporal** y **anticipación** que la clase presenta como frontera.

El descenso de desempeño entre cocinas vistas (top-1 20,5 %) y no vistas (10,9 %) es, además, una ilustración concreta del problema de **domain adaptation** en video: un modelo entrenado en unos entornos no transfiere trivialmente a entornos nuevos, y medir esa brecha es parte esencial del análisis riguroso.
