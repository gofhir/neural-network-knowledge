---
title: "Speech2Face: reconstruir un rostro desde la voz (2019)"
weight: 465
math: true
---

{{< paper-card
    title="Speech2Face: Learning the Face Behind a Voice"
    authors="Tae-Hyun Oh, Tali Dekel, Changil Kim, Inbar Mosseri, William T. Freeman, Michael Rubinstein, Wojciech Matusik (MIT CSAIL / Google Research)"
    year="2019"
    venue="CVPR 2019 / arXiv:1905.09773"
    arxiv="1905.09773"
    pdf="/papers/speech2face-oh-2019.pdf" >}}
Dada una grabación de voz, reconstruir una imagen del rostro de quien habla. El modelo se entrena sobre millones de pares voz-rostro de videos de YouTube (AVSpeech) sin ninguna anotación humana: la correspondencia entre cara y voz **está en el video**. Los autores son explícitos sobre el alcance —*"nuestro objetivo no es predecir una imagen reconocible de la cara exacta"*— y sobre sus riesgos: el paper incluye una sección de **Consideraciones Éticas**, poco habitual en 2019, donde documentan que su modelo hereda el sesgo demográfico del conjunto de entrenamiento. Es el ejemplo más claro de la clase sobre lo que estas técnicas sí y no pueden hacer.
{{< /paper-card >}}

---

## El método

Un codificador toma el espectrograma de la voz y produce un vector en el **espacio de características faciales** de una red de reconocimiento facial preentrenada — concretamente, la penúltima capa de VGG-Face. Un decodificador separado, entrenado aparte, reconstruye una imagen canónica de rostro (frontal, iluminación neutra) desde ese vector.

La decisión de diseño clave es **no predecir píxeles directamente**. El paper señala que un modelo de voz a píxeles no funciona. En su lugar se predice un vector en un espacio ya organizado semánticamente, y la reconstrucción es un problema aparte, resuelto con datos de imagen abundantes.

{{< concept-alert type="clave" >}}
Es la misma estructura que [SoundNet](/papers/soundnet-aytar-2016) en la [Clase 43](/clases/clase-43): una red visual madura define un espacio de representación, y una red de audio aprende a proyectarse en él usando el video como puente. Cambia lo que se transfiere —ahí, distribuciones sobre categorías; aquí, un vector de identidad facial— pero el mecanismo es el mismo, y el paper cita a Aytar et al. entre sus antecedentes.
{{< /concept-alert >}}

## Qué recupera realmente

Los resultados muestran correlaciones no triviales entre voz y apariencia en **atributos**: edad aproximada, género y —con las salvedades que siguen— origen étnico, además de rasgos correlacionados como la estructura general del rostro. Lo que **no** recupera es la identidad: no reconstruye la cara de una persona concreta, sino un rostro promedio compatible con los atributos que la voz sugiere.

Esa distinción es real y física. El tracto vocal impone restricciones anatómicas —longitud, forma de la cavidad— que se correlacionan con el tamaño y la forma de la cabeza. Pero la correlación es débil y poblacional, no individual.

## La sección de consideraciones éticas

Es la parte del paper que más vale leer, y es infrecuente para su época.

**Privacidad.** Los autores declaran que el método *"no puede recuperar"* la identidad de la persona, y advierten explícitamente contra interpretaciones que lo traten como identificación.

**Sesgo del conjunto de datos.** El reconocimiento aquí es directo: el modelo se entrena con videos de YouTube y *"no representa por igual a toda la población mundial"*. El paper muestra la matriz de confusión por atributos junto a la distribución del conjunto de entrenamiento, y señala que el desempeño en origen étnico *"aparece sesgado por la distribución desigual del conjunto de entrenamiento"*. Un idioma ausente del entrenamiento produce reconstrucciones sistemáticamente equivocadas.

**Correlaciones espurias.** Un modelo así puede aprender asociaciones que reflejan la composición del corpus —qué tipos de personas suben qué tipos de video— y no ninguna relación causal entre voz y rostro.

{{< concept-alert type="advertencia" >}}
El caso ilustra un patrón general que conviene retener del cierre del diplomado: **un modelo puede tener buen desempeño agregado y ser inutilizable —o dañino— por caso individual**.

Speech2Face acierta atributos poblacionales mejor que el azar y produce, para cualquier persona concreta, un rostro que probablemente no se le parece. Presentar esa salida como "la cara detrás de la voz" en un contexto de identificación sería un error de categoría con consecuencias directas.

Es el mismo argumento que [Super-resolución](/fundamentos/super-resolucion): lo que sale no es información recuperada, es el prior del modelo condicionado por lo observado.
{{< /concept-alert >}}

## Por qué importa para la Clase 44

Es la segunda de las siete aplicaciones que la clase presenta, y la que mejor muestra el patrón que las une: **usar la correspondencia natural entre modalidades para predecir una desde la otra**. Junto con Speech Reconstruction from Silent Videos —que va en la dirección inversa, de video a audio— forma un par simétrico que deja clara la idea.

Y aporta al cierre del diplomado algo que la clase no desarrolla: un modelo de cómo un equipo de investigación puede publicar una capacidad potencialmente problemática **documentando sus límites y sesgos en el propio paper**, en vez de dejar esa discusión para después.

---

**Ver también:** [SoundNet (2016)](/papers/soundnet-aytar-2016) · [Looking to Listen (2018)](/papers/looking-to-listen-ephrat-2018) · [Vid2Speech (2017)](/papers/vid2speech-ephrat-2017) · [Aprendizaje Audiovisual](/fundamentos/aprendizaje-audiovisual) · [Síntesis de Medios](/fundamentos/sintesis-de-medios)
