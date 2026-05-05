---
title: "Video"
weight: 4
sidebar:
  open: true
---

# Video

## El problema central

Video es una **secuencia de imágenes con costo computacional explosivo**. Un clip de 16 frames a resolución 224×224 son ~16× los píxeles de una imagen estática, y ese factor crece linealmente con la duración. Procesar el video cuadro por cuadro pierde la información de **movimiento** que define qué está pasando — *correr* y *trotar* tienen los mismos pixels promedio pero dinámicas distintas. Procesarlo conjuntamente requiere arquitecturas que no escalen cuadráticamente con la longitud temporal. Cada generación de modelos navegó esa tensión de forma distinta.

Tres tensiones definen el campo: (1) cómo **modelar movimiento sin desperdiciar parámetros** — kernels convolucionales 3D, two-stream con flujo óptico, factorización espacio-tiempo; (2) cómo **escalar a clips largos** — ventanas de atención, jerarquías, modelos eficientes; y (3) la divergencia entre **comprensión** (action recognition, video classification, donde el ground truth es una etiqueta) y **generación** (texto-a-video, donde la frontera es la coherencia temporal extendida y la física aproximada). Hasta 2022 estos eran problemas separados con arquitecturas dedicadas; los modelos generativos actuales (Sora, Veo) están empezando a unificarlos.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Era pre-deep / handcrafted" years="2003-2013" >}}
    {{< hito year="2008" name="HOG3D / Cuboids 3D" status="minimal" >}}
      Klaeser et al., Laptev: extensiones 3D de descriptores de imagen (HOG, SIFT) sobre volúmenes espacio-temporales. **Por qué importó:** primer intento sistemático de capturar movimiento como una feature computable, base de los pipelines clásicos de action recognition.
    {{< /hito >}}
    {{< hito year="2011" name="Dense Trajectories" status="minimal" >}}
      Wang & Schmid (INRIA): trayectorias densas obtenidas siguiendo puntos vía flujo óptico, descritas con HOG/HOF/MBH. **Por qué importó:** estado del arte en HMDB y UCF-101 durante varios años con descriptores hechos a mano.
    {{< /hito >}}
    {{< hito year="2013" name="iDT — improved Dense Trajectories" status="minimal" >}}
      Wang & Schmid: refinamiento con compensación de movimiento de cámara y Fisher Vectors. **Por qué importó:** seguía superando o igualando a las primeras CNNs de video durante 2014-2015 — el handcrafted no se rindió fácil.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era two-stream y 3D-CNN tempranas" years="2014-2015" >}}
    {{< hito year="2014" name="Karpathy CVPR (slow/fast fusion)" status="minimal" >}}
      Karpathy et al. (Google/Stanford): CNNs aplicadas a video con varias estrategias de fusión temporal (early, late, slow, single-frame). **Por qué importó:** primer estudio sistemático de cómo extender CNN-2D a video; resultado descorazonador (igual a single-frame), motivó two-stream.
    {{< /hito >}}
    {{< hito year="2014" name="Two-Stream" status="minimal" >}}
      Simonyan & Zisserman (Oxford): dos CNNs paralelas — una sobre RGB (apariencia), otra sobre flujo óptico (movimiento) — fusionadas en la decisión final. **Por qué importó:** superó a iDT y fijó el patrón "modelar movimiento explícitamente" durante 4 años.
    {{< /hito >}}
    {{< hito year="2015" name="C3D" status="minimal" >}}
      Tran et al. (FAIR): convoluciones 3D (3×3×3) aprendidas extremo a extremo sobre clips de 16 frames. **Por qué importó:** primer modelo 3D-CNN entrenable sobre video sin recurrir a flujo óptico precomputado.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de 3D-CNN profundas" years="2017-2019" >}}
    {{< hito year="2017" name="I3D" status="minimal" >}}
      Carreira & Zisserman (DeepMind): *Inflated 3D ConvNet* — toma una CNN-2D preentrenada en ImageNet (Inception-V1, kernels 3×3) y "infla" sus filtros 2D a 3D (un kernel 3×3 se vuelve 3×3×3 replicando los pesos en el eje temporal), transfiriendo la representación visual a video. Entrenado sobre el nuevo Kinetics-400. **Por qué importó:** el primer modelo que destronó claramente a iDT y two-stream en todos los benchmarks; backbone estándar 2017-2019.
    {{< /hito >}}
    {{< hito year="2018" name="R(2+1)D" status="minimal" >}}
      Tran et al. (FAIR): factoriza convolución 3D en una espacial 2D seguida de una temporal 1D. **Por qué importó:** mejor accuracy con menos parámetros que 3D pura; demostró que separar espacio y tiempo era pedagógicamente útil para el modelo.
    {{< /hito >}}
    {{< hito year="2019" name="SlowFast" status="minimal" >}}
      Feichtenhofer et al. (FAIR): dos vías paralelas — una "slow" a baja frecuencia para apariencia, una "fast" a alta frecuencia para movimiento. Inspirado en M/P pathways del sistema visual. **Por qué importó:** estado del arte en Kinetics y AVA; influyó la división persistente "spatial vs temporal" en arquitecturas posteriores.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de Video Transformers" years="2021-2022" >}}
    {{< hito year="2021" name="TimeSformer" status="minimal" >}}
      Bertasius et al. (FAIR): primer Video Transformer puro. Tokeniza cada frame como en ViT y aplica atención factorizada (espacial dentro de cada frame, temporal a través de frames). **Por qué importó:** mostró que ViT podía superar a 3D-CNN profundas con menos cómputo, abriendo la era Transformer en video.
    {{< /hito >}}
    {{< hito year="2021" name="ViViT" status="minimal" >}}
      Arnab et al. (Google): cuatro variantes de Video Transformer con distintos esquemas de factorización espacio-temporal. **Por qué importó:** sistematizó el espacio de diseño de Video Transformers; las opciones "factorized self-attention" y "tubelet embedding" se volvieron estándares.
    {{< /hito >}}
    {{< hito year="2021" name="MViT" status="minimal" >}}
      Fan et al. (FAIR): *Multiscale Vision Transformer*, jerárquico con resolución decreciente y canales crecientes (estilo CNN). **Por qué importó:** Video Transformer más eficiente, base de muchas pipelines downstream.
    {{< /hito >}}
    {{< hito year="2022" name="Video Swin" status="minimal" >}}
      Liu et al.: Swin Transformer extendido a 3D con ventanas espacio-temporales desplazadas. **Por qué importó:** estado del arte en Kinetics-600/700 con sesgo inductivo recuperado vía ventanas locales.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de generación + foundation" years="2022-presente" >}}
    {{< hito year="2022" name="Make-A-Video" status="minimal" >}}
      Singer et al. (Meta): texto-a-video sin pares texto-video etiquetados — usa modelos texto-a-imagen + módulos temporales aprendidos sobre video sin etiquetas. **Por qué importó:** primer texto-a-video de calidad razonable, abrió la frontera generativa.
    {{< /hito >}}
    {{< hito year="2022" name="Imagen Video" status="minimal" >}}
      Ho et al. (Google): cascada de modelos de difusión (base + super-resolución espacial + super-resolución temporal) para generar video 1280×768 a 24 fps. **Por qué importó:** fijó el patrón "diffusion + cascada" para video generativo.
    {{< /hito >}}
    {{< hito year="2023" name="Stable Video Diffusion" status="minimal" >}}
      Stability AI: modelo de difusión latente para video corto, open-source. **Por qué importó:** democratizó la generación de video — open weights que la comunidad pudo extender.
    {{< /hito >}}
    {{< hito year="2024" name="Sora" status="minimal" >}}
      OpenAI (anunciado feb 2024, lanzamiento dic 2024): generación de video de hasta 60 segundos con coherencia temporal extendida, física aproximada y resolución alta. Diffusion sobre tokens de video latentes. **Por qué importó:** salto cualitativo en duración y coherencia; redefinió las expectativas de la frontera.
    {{< /hito >}}
    {{< hito year="2024" name="Veo / Veo 2" status="minimal" >}}
      Google DeepMind (Veo 1 mayo 2024, Veo 2 dic 2024): generación de video con prompts complejos, control de cámara y estilos cinematográficos (Veo 1 hasta 1080p; Veo 2 hasta 4K). **Por qué importó:** alternativa frontera a Sora con foco en control fino y resolución.
    {{< /hito >}}
    {{< hito year="2024" name="Kling / Runway Gen-3" status="minimal" >}}
      Kuaishou (Kling, jun 2024) y Runway (Gen-3 Alpha, jun 2024): generación de video competitiva con frontier occidental. **Por qué importó:** Kling demostró que China alcanzó paridad rápida; Runway llevó la generación a producción para creadores y estudios.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}
