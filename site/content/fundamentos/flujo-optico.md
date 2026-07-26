---
title: "Flujo óptico"
weight: 122
math: true
---

El **flujo óptico** (optical flow) es la representación del **movimiento** en un video: un campo de vectores que describe cómo se desplaza cada píxel de un frame al siguiente. Es una de las herramientas más importantes del [análisis de video](/fundamentos/analisis-de-video), porque captura de forma explícita justo aquello que distingue al video de una imagen —la dinámica temporal. Este fundamento acompaña a la [Clase 36](/clases/clase-36): explica el problema, su formulación clásica y su versión moderna con deep learning, y por qué es central tanto para el seguimiento de objetos como para el reconocimiento de acciones.

---

## 1. El problema: ¿hacia dónde se movió cada píxel?

La estimación de flujo óptico consiste en **computar el desplazamiento de píxeles entre dos frames** consecutivos. Se trata como un **problema de correspondencia**: para cada píxel del frame 1, encontrar dónde quedó en el frame 2. La salida es un **vector de movimiento** por píxel —un campo denso $(u, v)$ que indica cuánto se desplazó horizontal y verticalmente.

{{< concept-alert type="clave" >}}
El flujo óptico ayuda a **entender el movimiento** de los píxeles de un frame a otro. Es una representación explícita de la dinámica: donde el video crudo tiene apariencia (qué hay), el flujo óptico tiene **movimiento** (cómo cambia). Por eso es un ingrediente clave en el [seguimiento de objetos](/fundamentos/analisis-de-video) (el desafío del "motion") y en el reconocimiento de acciones (como entrada del stream temporal).
{{< /concept-alert >}}

---

## 2. La formulación clásica

El flujo óptico clásico parte de la **hipótesis de constancia de brillo**: un punto de la escena conserva su intensidad al moverse entre frames. Para un píxel en $(x,y)$ en el tiempo $t$, con desplazamiento $(dx, dy)$ en $dt$:

$$
I(x, y, t) = I(x + dx, y + dy, t + dt).
$$

Expandiendo en serie de Taylor y quedándose con los términos de primer orden se obtiene la **ecuación de restricción del flujo óptico**:

$$
I_x\, u + I_y\, v + I_t = 0,
$$

donde $I_x, I_y, I_t$ son las derivadas parciales de la imagen y $(u, v) = (dx/dt, dy/dt)$ es el flujo buscado. Esta única ecuación tiene **dos incógnitas** por píxel —el llamado *problema de apertura*—, así que está subdeterminada. Los métodos clásicos la resuelven agregando supuestos:

- **Horn-Schunck** (1981): impone **suavidad global** del campo de flujo (los píxeles vecinos se mueven de forma parecida), resolviendo un problema variacional.
- **Lucas-Kanade** (1981): asume flujo **constante en una vecindad local** de cada píxel, resolviendo un sistema por mínimos cuadrados.

Durante décadas, los métodos variacionales fueron el estado del arte.

---

## 3. La versión moderna: FlowNet

El deep learning llevó el flujo óptico al terreno del aprendizaje. **FlowNet** (Dosovitskiy et al., 2015) fue la primera red convolucional **end-to-end** para estimar flujo óptico, planteándolo como un problema supervisado de imagen-a-imagen. Propuso dos arquitecturas:

- **FlowNetSimple** — apila las dos imágenes (6 canales) y las pasa por una CNN encoder-decoder que produce el campo de flujo.
- **FlowNetCorr** — procesa cada imagen por separado y usa una **capa de correlación** que compara explícitamente los *features* de las dos imágenes (buscando correspondencias), imitando la estructura del problema de matching.

Ambas usan un refinamiento *upconvolutional* para recuperar la resolución del campo denso. → [análisis](/papers/flownet-dosovitskiy-2015)

{{< concept-alert type="advertencia" >}}
El gran obstáculo de aprender flujo óptico es el **ground-truth**: es casi imposible etiquetar a mano el desplazamiento real de cada píxel en video real. FlowNet lo resolvió con un dataset **sintético** —*Flying Chairs*, sillas 3D volando sobre fondos, donde el flujo verdadero se conoce por construcción— y demostró que el modelo **generaliza** a video real. Es un patrón recurrente en visión: cuando el ground-truth real es inalcanzable, la síntesis controlada lo sustituye.
{{< /concept-alert >}}

---

## 4. Su papel en el análisis de video

El flujo óptico es un componente, no un fin en sí mismo. Sus dos usos principales en la clase:

- **Seguimiento de objetos (VOT).** Estimar el flujo entre frames ayuda a predecir dónde estará un objeto en el frame siguiente —el desafío del "motion" que menciona la clase.
- **Reconocimiento de acciones.** En las arquitecturas **[two-stream](/papers/two-stream-simonyan-2014)**, un stream completo recibe **stacks de flujo óptico** como entrada, dedicado exclusivamente a modelar el movimiento, mientras el otro stream modela la apariencia. Separar apariencia de movimiento resultó ser una de las ideas más efectivas del [reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).

Modelos posteriores (C3D, I3D) aprendieron a capturar el movimiento **implícitamente** con convoluciones 3D, reduciendo la dependencia del flujo óptico precomputado —que es costoso—, pero el flujo óptico sigue siendo una representación fundamental y una entrada útil.

---

## 5. Relevancia para salud y video clínico

El flujo óptico tiene aplicaciones médicas directas donde el **movimiento** es la señal diagnóstica. En **ecocardiografía**, estima el movimiento del músculo cardíaco entre frames para evaluar la función ventricular. En **análisis de marcha** y estudios de **temblor**, cuantifica el movimiento de segmentos corporales. En **imagenología respiratoria**, rastrea el desplazamiento de estructuras con el ciclo respiratorio (crítico para la radioterapia guiada por imagen). Y en **endoscopía/cirugía**, estima el movimiento de tejido e instrumentos. En todos, el mismo desafío del ground-truth reaparece —el movimiento real rara vez se conoce— y las mismas soluciones (métodos variacionales o redes entrenadas con datos sintéticos o semi-supervisados) se trasladan del dominio general al clínico.

---

## Referencias

- Horn, B. & Schunck, B. (1981). *Determining Optical Flow*. Artificial Intelligence. — el método variacional clásico.
- Lucas, B. & Kanade, T. (1981). *An Iterative Image Registration Technique*. — el método local.
- Dosovitskiy, A. et al. (2015). *FlowNet: Learning Optical Flow with Convolutional Networks*. ICCV. — [análisis](/papers/flownet-dosovitskiy-2015)
- Fundamentos relacionados: [Análisis de Video](/fundamentos/analisis-de-video) · [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).
