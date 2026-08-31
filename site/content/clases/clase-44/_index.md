---
title: "Clase 44 - Aplicaciones de Audio y Video (cierre)"
weight: 440
sidebar:
  open: true
---

**Profesor:** Carlos Aspillaga (DCC, Pontificia Universidad Católica de Chile)
**Módulo:** Audio y Video — **última clase del diplomado**

La clase abre repasando el temario completo de los cuatro cursos, con dos imágenes que le dan sentido: una **caja de herramientas** al lado de la lista de técnicas, y una flecha que dice **"USTEDES"** apuntando a dos superhéroes. Después recorre siete aplicaciones audiovisuales que parecen magia y están hechas, todas, con piezas de esa lista. La diapositiva de agenda lo dice sin rodeos: *"aplicaciones sorprendentes de audio y video **usando las cosas que ya conocemos**"*.

{{< concept-alert type="clave" >}}
**El hilo que une las siete aplicaciones.** Reconstruir voz desde labios, cara desde voz, separar hablantes, limpiar audio, sincronizar, aumentar resolución, generar deep fakes — en todas, **la información necesaria no está en la entrada**. Lo que la completa es un prior aprendido.

Eso convierte la pregunta interesante en una sola, y sirve para las siete: **¿qué parte de la salida estaba en la entrada, y qué parte la puso el modelo?** La clase la responde para el caso de la super-resolución con dos palabras —*informed guess*— que la [profundización](profundizacion) vuelve cuantitativas: a factor 4, **3855 imágenes distintas producen la misma entrada**.

**El remate.** Las dos últimas diapositivas del método muestran la misma factorización aplicada a las dos modalidades: en video se conserva el **aspecto** y se transfiere el **movimiento**; en audio se conserva el **timbre** y se transfiere el **contenido**. Es la operación que la [Clase 41](/clases/clase-41) usó para *analizar* —separar qué se dijo de quién lo dijo— aquí puesta a *generar*, cruzando las piezas.
{{< /concept-alert >}}

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Las 46 diapositivas: la recapitulación del diplomado mapeada clase por clase, las siete aplicaciones con sus papers, y el método de First Order Motion Model paso a paso hasta el laboratorio" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Cuánto vale el jacobiano y por qué a igual presupuesto no siempre gana, el informed guess hecho preciso, la asimetría entre generar y detectar, y la factorización que unifica toda la clase" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="El campo de movimiento de primer orden con warping diferenciable, y la aritmética del informed guess — en triple framework, con los cuatro backends coincidiendo a 2,8e−16" icon="code" >}}
  {{< card link="/laboratorios/lab-44" title="Laboratorio 44 - Deep Fakes" subtitle="Los tres notebooks, que no ejecutan SV2TTS ni FOMM sino TorToise-TTS y Wan2.2-Animate: el mismo classifier-free guidance pagado en inferencia por uno y destilado en los pesos por el otro" icon="beaker" >}}
  {{< card link="/clases/clase-43" title="Clase anterior: Aplicaciones para Audio y Video" subtitle="SoundNet y E2E-AVSR: la correspondencia audiovisual usada para entrenar y para decidir" icon="sparkles" >}}
  {{< card link="/clases/clase-29" title="Relacionada: Modelos Generativos en Visión" subtitle="VAE, GAN y difusión — la maquinaria generativa que esta clase aplica" icon="photograph" >}}
  {{< card link="/clases/clase-41" title="Relacionada: Speech y Speaker Recognition" subtitle="La misma factorización identidad/contenido, usada para analizar en vez de generar" icon="adjustments" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/sintesis-de-medios" title="Síntesis de Medios (deepfakes)" subtitle="Las cinco técnicas que el término confunde, los rastros que dejan, la detección y su asimetría, y las preguntas previas a construir con esto" icon="book-open" >}}
  {{< card link="/fundamentos/super-resolucion" title="Super-resolución" subtitle="Un problema mal planteado: por qué el óptimo en MSE se ve borroso y por qué esto no es una herramienta forense" icon="book-open" >}}
  {{< card link="/fundamentos/separacion-de-fuentes" title="Separación de Fuentes" subtitle="El problema de la permutación y cómo el video lo disuelve en vez de resolverlo" icon="book-open" >}}
  {{< card link="/fundamentos/aprendizaje-audiovisual" title="Aprendizaje Audiovisual" subtitle="La sincronía como supervisión gratuita: el marco común de casi todas estas aplicaciones" icon="book-open" >}}
  {{< card link="/fundamentos/modelos-generativos" title="Modelos Generativos" subtitle="GAN, VAE y difusión: de dónde sale el prior que completa lo que falta" icon="book-open" >}}
  {{< card link="/fundamentos/lectura-de-labios" title="Lectura de Labios" subtitle="El techo de información de los visemas, que limita la reconstrucción de voz desde video" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### El método del laboratorio

{{< cards >}}
  {{< card link="/papers/fomm-siarohin-2019" title="First Order Motion Model (2019)" subtitle="Siarohin et al. — puntos clave aprendidos sin supervisión más jacobianos locales. Es reenactment, no face swap, y su máscara de oclusión es la pieza que la clase omite" icon="document-text" >}}
  {{< card link="/papers/sv2tts-jia-2018" title="SV2TTS (2018)" subtitle="Jia et al. — la clonación de voz del lado del audio: timbre del hablante más contenido del texto" icon="document-text" >}}
  {{< card link="/papers/faceforensics-rossler-2019" title="FaceForensics++ (2019)" subtitle="Rössler et al. — el contrapeso que la clase no incluye: 1,8 millones de imágenes manipuladas, y el resultado de que detectar generaliza mal" icon="document-text" >}}
  {{< card link="/papers/tortoise-tts-betker-2023" title="TorToise-TTS (2023)" subtitle="Betker — lo que el laboratorio ejecuta en lugar de SV2TTS: DALL·E aplicado a voz, con re-ranking contrastivo sobre tokens y un detector de deep fakes incluido" icon="document-text" >}}
  {{< card link="/papers/wan-animate-2025" title="Wan-Animate (2025)" subtitle="Tongyi Lab — lo que el laboratorio ejecuta en lugar de FOMM: un DiT de 14B con esqueleto explícito para el cuerpo y features implícitas para el rostro" icon="document-text" >}}
{{< /cards >}}

### Las siete aplicaciones

{{< cards >}}
  {{< card link="/papers/vid2speech-ephrat-2017" title="Vid2Speech (2017)" subtitle="Ephrat y Peleg — voz desde video mudo, sin pasar por texto para no perder la prosodia" icon="document-text" >}}
  {{< card link="/papers/speech2face-oh-2019" title="Speech2Face (2019)" subtitle="Oh et al. — la dirección inversa, con una sección de consideraciones éticas que documenta sus propios sesgos" icon="document-text" >}}
  {{< card link="/papers/looking-to-listen-ephrat-2018" title="Looking to Listen (2018)" subtitle="Ephrat et al. — separar voces usando rostros: el video no aporta información acústica, aporta estructura" icon="document-text" >}}
  {{< card link="/papers/separating-object-sounds-gao-2018" title="Learning to Separate Object Sounds (2018)" subtitle="Gao et al. — separación guiada por objetos, con NMF y una red visual" icon="document-text" >}}
  {{< card link="/papers/audio-superres-kuleshov-2017" title="Audio Super-Resolution (2017)" subtitle="Kuleshov et al. — el informed guess aplicado al espectro que el muestreo eliminó" icon="document-text" >}}
{{< /cards >}}

---

## El diplomado, mapeado

La [teoría](teoria) toma la lista de la primera diapositiva y la enlaza clase por clase — de redes neuronales y CNNs hasta tracking y aplicaciones audiovisuales, pasando por NLP, generativos, grafos, meta-aprendizaje y refuerzo. Es el índice que la clase de cierre quiere ser.

**Ver también:** [Índice de clases](/clases) · [Fundamentos](/fundamentos) · [Papers](/papers) · Dominios [Audio](/dominios/audio), [Video](/dominios/video), [Visión](/dominios/vision) y [Multimodal](/dominios/multimodal).
