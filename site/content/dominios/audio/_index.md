---
title: "Audio / Voz"
weight: 3
sidebar:
  open: true
---

# Audio / Voz

## El problema central

El audio es una **señal continua de alta tasa de muestreo**: 16,000 muestras por segundo en habla telefónica, 48,000 en audio profesional. Modelar directamente esa cadena de muestras crudas era infactible hasta los 2010s; la técnica clásica es proyectar la señal a un **espectrograma** — una representación tiempo-frecuencia que reduce la dimensionalidad y exhibe estructura mucho más amable para modelos. La elección espectrograma vs raw waveform sigue siendo una tensión activa en 2025: el primero es eficiente pero descarta información de fase; el segundo es fiel pero costoso.

Dos sub-problemas con tensiones opuestas vertebran el campo: **ASR / comprensión** (audio → texto) exige robustez a ruido, acentos, dispersión hablante y eficiencia para tiempo real; **TTS / generación** (texto → audio) exige naturalidad prosódica, expresividad, control de timbre y latencia conversacional. Una década de arquitecturas se organiza alrededor de esa división — modelos diseñados para entender vs modelos diseñados para generar — hasta que los foundation models actuales empezaron a unificarlas en una sola arquitectura.

## Línea de tiempo

{{< timeline >}}
  {{< era name="Fundamentos de procesamiento de señales" years="1822-1980" >}}
    {{< hito year="1946" name="Incertidumbre tiempo-frecuencia (Gabor)" status="covered" link="/papers/time-frequency-gabor-1946" >}}
      Dennis Gabor: el plano tiempo-frecuencia dividido en celdas mínimas (logones) y el **principio de incertidumbre para señales** $\Delta t \cdot \Delta f \ge \tfrac12$. **Por qué importó:** la raíz teórica de la STFT, el espectrograma y las wavelets; explica el trade-off de la ventana. Cubierto en la [Clase 35](/clases/clase-35).
    {{< /hito >}}
    {{< hito year="1949" name="Teorema de muestreo (Nyquist-Shannon)" status="covered" link="/papers/sampling-shannon-1949" >}}
      Claude Shannon (sobre Nyquist 1928): una señal de banda $W$ se reconstruye exactamente muestreando a $2W$/s. **Por qué importó:** la ley que gobierna toda digitalización de audio (los 44,1 kHz del CD) y el aliasing. Cubierto en la [Clase 35](/clases/clase-35).
    {{< /hito >}}
    {{< hito year="1965" name="FFT (Cooley-Tukey)" status="covered" link="/papers/fft-cooley-tukey-1965" >}}
      El algoritmo que reduce la DFT de $O(N^2)$ a $O(N\log N)$. **Por qué importó:** hizo práctico todo el análisis espectral —espectrogramas, filtros, MFCC—; es lo que corre bajo `np.fft`. Cubierto en la [Clase 35](/clases/clase-35).
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era acústica clásica" years="1980-2010" >}}
    {{< hito year="1980" name="MFCC" status="covered" link="/papers/mfcc-davis-mermelstein-1980" >}}
      Davis & Mermelstein: *Mel-Frequency Cepstral Coefficients*, una representación que imita la sensibilidad logarítmica del oído humano a la frecuencia (banco de filtros Mel + log + DCT). **Por qué importó:** la entrada estándar de toda ASR clásica durante 30 años. Cubierto en la [Clase 35](/clases/clase-35).
    {{< /hito >}}
    {{< hito year="1980s-1990s" name="HMM-GMM" status="minimal" >}}
      Modelos ocultos de Markov con emisiones gaussianas para modelar fonemas. Cada estado oculto del HMM corresponde a un sub-fonema; la mezcla gaussiana modela la distribución de MFCCs en cada estado. **Por qué importó:** estado del arte hasta 2011 — Siri 1.0 los usaba.
    {{< /hito >}}
    {{< hito year="1990s-2000s" name="n-gramas + WFST" status="minimal" >}}
      Modelos de lenguaje n-grama integrados con el modelo acústico vía *Weighted Finite-State Transducers*. **Por qué importó:** combinaron acústica + lenguaje en un solo decoder Viterbi, base de toda ASR de producción pre-deep.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era deep speech híbrido" years="2011-2014" >}}
    {{< hito year="2011" name="DNN-HMM" status="minimal" >}}
      Hinton, Mohamed, Dahl: reemplazaron las gaussianas del HMM por una red neuronal profunda que predice probabilidades de estado. **Por qué importó:** primer salto de error en ASR en una década (~30% relativo); inicio del deep learning aplicado a voz.
    {{< /hito >}}
    {{< hito year="2011" name="Kaldi toolkit" status="minimal" >}}
      Povey et al.: framework open-source para ASR híbrida con WFST. **Por qué importó:** estandarizó la pila de ASR académica e industrial durante una década.
    {{< /hito >}}
    {{< hito year="2014" name="DeepSpeech 1" status="minimal" >}}
      Hannun et al. (Baidu): primer ASR end-to-end neural sin HMM. CNN + RNN bidireccional + CTC sobre espectrograma directo a caracteres. **Por qué importó:** demostró que la pipeline clásica (acústico + pronunciación + lenguaje) podía colapsarse en un solo modelo entrenable.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era end-to-end con atención" years="2014-2018" >}}
    {{< hito year="2013" name="Deep BiLSTM + CTC" status="covered" link="/papers/deep-rnn-speech-graves-2013" >}}
      Graves, Mohamed y Hinton (Toronto): la pregunta era si una RNN —que ya es profunda *en el tiempo*— gana algo con ser profunda *en el espacio*. Apilar BiLSTM y entrenarlas end-to-end con CTC da **17,7 % de PER en TIMIT**, el mejor resultado del momento. **Por qué importó:** es el primer eslabón enteramente neuronal del pipeline de ASR, y la arquitectura del "Ejemplo 1" de la [Clase 41](/clases/clase-41).
    {{< /hito >}}
    {{< hito year="2014" name="CTC loss" status="covered" link="/papers/ctc-graves-2006" >}}
      Graves et al. (ICML 2006), aplicado a ASR end-to-end por DeepSpeech 1 en 2014: *Connectionist Temporal Classification* introduce un token **blank** y define la probabilidad de una transcripción como la **suma sobre todas las alineaciones** que la producen — un número que crece como $\binom{T+U}{2U}$ y que una recursión de tres términos calcula en $O(TU)$. **Por qué importó:** resolvió el problema central de DeepSpeech y se volvió la pérdida estándar de ASR neural. Su costo es asumir independencia condicional entre salidas, lo que obliga a un modelo de lenguaje externo. Desarrollado en la [Clase 41](/clases/clase-41/profundizacion).
    {{< /hito >}}
    {{< hito year="2015" name="Atención aplicada al habla" status="covered" link="/papers/attention-asr-chorowski-2015" >}}
      Chorowski, Bahdanau et al.: el primer traslado serio de la atención de traducción al habla, y el primero en documentar por qué **no funciona directamente**. El habla contiene fragmentos acústicamente idénticos repartidos por todo el enunciado, y una atención puramente por contenido no puede distinguirlos: el modelo se rompe con secuencias más largas que las de entrenamiento. **Por qué importó:** la solución —dar al mecanismo *conciencia de ubicación*— fija el patrón de las atenciones monótonas posteriores.
    {{< /hito >}}
    {{< hito year="2015" name="CLDNN" status="deep" link="/papers/cldnn-sainath-2015" >}}
      Sainath et al. (Google, ICASSP 2015): CNN, LSTM y capas densas en una sola red entrenada de punta a punta, con la tesis de que sus propiedades son complementarias — la convolución reduce la varianza espectral, la recurrencia modela el tiempo largo, las densas separan clases. **Por qué importó:** formalizó el patrón CRNN que dominó audio, música y detección de eventos durante cinco años, y su intuición sobre combinar un operador local con uno global sigue siendo la de Conformer. Es la arquitectura del "Ejemplo 1" de la [Clase 39](/clases/clase-39).
    {{< /hito >}}
    {{< hito year="2015" name="Listen, Attend and Spell" status="covered" link="/papers/las-chan-2016" >}}
      Chan et al. (Google): primer ASR de vocabulario grande puramente atencional. Encoder-decoder con atención al estilo Bahdanau, sin CTC y sin supuesto de independencia entre salidas — el modelo aprende acústica y ortografía juntas. El *listener* es un encoder **piramidal** que reduce a la mitad los pasos temporales en cada capa, y los autores reportan que **sin él el modelo no converge**. **Por qué importó:** demostró que ASR podía hacerse como traducción audio→texto; su pirámide es el "pooling in time" del Ejemplo 2 de la [Clase 41](/clases/clase-41).
    {{< /hito >}}
    {{< hito year="2015" name="Deep Speech 2" status="covered" link="/papers/deep-speech-2-amodei-2015" >}}
      Amodei et al. (Baidu): la rama CTC llevada a escala industrial — **11 940 horas de inglés y 9 400 de mandarín** con la misma arquitectura, sin léxico de pronunciaciones específico por idioma. BatchNorm para RNN, el currículum SortaGrad y una batería de optimizaciones de HPC que dan 7× de speedup. **Por qué importó:** mostró que el end-to-end escala, que la escala es en buena medida un problema de ingeniería de cómputo, y alcanzó o superó a transcriptores humanos en varios benchmarks.
    {{< /hito >}}
    {{< hito year="2016" name="Pooling over time para LVSR" status="covered" link="/papers/e2e-lvsr-bahdanau-2016" >}}
      Bahdanau, Chorowski et al.: llevan la atención de los fonemas de TIMIT al vocabulario grande del Wall Street Journal, y atacan el cuello de botella real — que la atención debe recorrer todos los frames por cada carácter emitido. Dos remedios: limitar el barrido y **agregar frames vecinos** para acortar la secuencia fuente. **Por qué importó:** el pooling temporal se volvió estándar en todos los encoders de audio posteriores, incluidos los Transformers.
    {{< /hito >}}
    {{< hito year="2016" name="WaveNet" status="deep" link="/papers/wavenet-oord-2016" >}}
      van den Oord et al. (DeepMind): modelo autorregresivo que genera la forma de onda muestra a muestra, con **convoluciones causales dilatadas** para alcanzar campos receptivos de cientos de milisegundos con pocas decenas de capas. **Por qué importó:** rompió el techo de calidad del TTS paramétrico y concatenativo, se desplegó en Google Assistant en 2017, y su esquema de dilataciones se convirtió en herramienta estándar mucho más allá del audio (TCN, DeepLab, ByteNet).
    {{< /hito >}}
    {{< hito year="2017" name="Very Deep CNN sobre onda cruda (familia M)" status="covered" link="/papers/raw-waveforms-dai-2017" >}}
      Dai et al. (CMU, Stanford, Bosch): CNN 1D de hasta 34 capas aplicadas directamente sobre la forma de onda, con una primera capa de kernel 80 —10 ms a 8 kHz— que aprende algo parecido a un banco de filtros. **Por qué importó:** mostró que la profundidad con stride agresivo alcanza campos receptivos de más de un segundo sin dilatación, y que los features hechos a mano no eran imprescindibles. Es el paper del laboratorio de la [Clase 39](/clases/clase-39).
    {{< /hito >}}
    {{< hito year="2016" name="SoundNet" status="covered" link="/papers/soundnet-aytar-2016" >}}
      Aytar, Vondrick y Torralba (MIT): una CNN 1D sobre **onda cruda** entrenada sin una sola etiqueta de audio — dos millones de videos de Flickr, redes visuales de ImageNet y Places como **maestros**, y [destilación](/fundamentos/destilacion-de-conocimiento) con divergencia KL usando la sincronía audiovisual como puente. **Por qué importó:** resolvió por un costado el cuello de botella del área (no hay datasets de audio etiquetados grandes) y superó el estado del arte por ~10 puntos en tres benchmarks — 88 % en DCASE, 74,2 % en ESC-50. Su ablación es el argumento medido de que la profundidad solo ayuda cuando hay datos que la sostengan: sin transferencia, pasar de 5 a 8 capas **empeora** (65,0 % → 51,1 %); con transferencia, mejora (66,1 % → 72,9 %). Es el precursor de toda la línea audiovisual autosupervisada. Cubierto en la [Clase 43](/clases/clase-43).
    {{< /hito >}}
    {{< hito year="2017" name="VGGish y el pre-entrenamiento de audio" status="deep" link="/papers/vggish-hershey-2017" >}}
      Hershey et al. (Google): tomar AlexNet, VGG, Inception y ResNet tal cual, alimentarlas con espectrogramas log-mel y entrenarlas sobre 70 millones de vídeos de YouTube. **Por qué importó:** validó a gran escala la idea de tratar el espectrograma como imagen, y produjo el primer extractor de embeddings de audio de propósito general — el "ImageNet del audio" que faltaba, y el estándar de facto hasta la llegada de AST.
    {{< /hito >}}
    {{< hito year="2017" name="Scaper" status="covered" link="/papers/scaper-salamon-2017" >}}
      Salamon et al. (NYU): librería para sintetizar paisajes sonoros muestreando eventos, fondos y parámetros de mezcla desde distribuciones especificadas. **Por qué importó:** resolvió el cuello de botella de la detección de eventos sonoros — las anotaciones fuertes con marca de tiempo son exactas por construcción cuando uno genera la mezcla, y además permite estudios controlados de degradación por SNR y polifonía.
    {{< /hito >}}
    {{< hito year="2018" name="End-to-End Audiovisual Speech Recognition" status="covered" link="/papers/e2e-avsr-petridis-2018" >}}
      Petridis et al. (Imperial College / Nottingham): el primer modelo que extrae features **simultáneamente de los píxeles de la boca y de la forma de onda**, con dos ResNet, BiGRU por flujo y una tercera BiGRU de fusión, sobre LRW (500 palabras de la BBC). **Por qué importó:** su tabla contiene dos resultados que suelen leerse mal. La onda cruda **empata** con MFCC en audio limpio (97,7 = 97,7) pero le gana **+7,5 puntos a −5 dB**: aprender la representación no compró exactitud sino robustez. Y la fusión aporta **+0,3 puntos en limpio y +14,1 a −5 dB**, porque el ruido acústico no toca al canal visual. Cubierto en la [Clase 43](/clases/clase-43).
    {{< /hito >}}
    {{< hito year="2018" name="SV2TTS: clonación de voz zero-shot" status="covered" link="/papers/sv2tts-jia-2018" >}}
      Jia et al. (Google, NeurIPS 2018): tres componentes entrenados por separado —encoder de hablante sobre audio sin transcribir, Tacotron 2 condicionado y vocoder WaveNet— para sintetizar habla de voces nunca vistas a partir de segundos de referencia. **Por qué importó:** desacopló identidad y contenido, abrió la clonación de voz práctica, y con ella el problema de los deepfakes de voz.
    {{< /hito >}}
    {{< hito year="2017" name="VoxCeleb" status="covered" link="/papers/voxceleb-nagrani-2017" >}}
      Nagrani, Chung y Zisserman (VGG Oxford): **1 251 hablantes y 153 516 enunciados** extraídos de entrevistas de YouTube, etiquetados **sin que ningún humano escuche nada** — un pipeline de visión por computador detecta caras, verifica con un modelo audiovisual que la persona en pantalla es la que habla, y confirma su identidad facial. [VoxCeleb2](/papers/voxceleb2-chung-2018) lo lleva en 2018 a 6 112 hablantes y más de un millón de enunciados, con particiones disjuntas. **Por qué importó:** sacó al reconocimiento de hablante de las condiciones de laboratorio y le dio la escala que el aprendizaje profundo necesitaba.
    {{< /hito >}}
    {{< hito year="2018" name="x-vectors" status="covered" link="/papers/x-vectors-snyder-2018" >}}
      Snyder et al. (JHU): una TDNN entrenada para discriminar hablantes, con una capa de **statistics pooling** —media y desviación estándar sobre el tiempo— que convierte enunciados de largo variable en descriptores fijos. El aporte del título es que la **aumentación con ruido y reverberación** es la palanca más barata para hacerlos robustos. **Por qué importó:** desplazó a los i-vectors generativos y se volvió la línea base obligatoria del área, porque a diferencia de ellos **escala con los datos**.
    {{< /hito >}}
    {{< hito year="2012/2017" name="RNN-Transducer" status="minimal" >}}
      Graves (propuesta original 2012; consolidación de producción ~2017): combinación de CTC con un modelo de lenguaje interno autoregresivo. **Por qué importó:** el algoritmo de ASR streaming de producción en Google y Apple — funciona online sin esperar el final de la oración.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era self-supervised" years="2019-2021" >}}
    {{< hito year="2019" name="wav2vec" status="minimal" >}}
      Schneider et al. (FAIR): pretraining no supervisado de representaciones de audio prediciendo el futuro de la señal. **Por qué importó:** primer "BERT para audio" — fine-tuning con pocos datos etiquetados igualó a modelos supervisados con miles de horas.
    {{< /hito >}}
    {{< hito year="2018" name="GhostVLAD: clusters que se descartan" status="covered" link="/papers/ghostvlad-zhong-2018" >}}
      Zhong, Arandjelović y Zisserman (VGG Oxford / DeepMind): extienden [NetVLAD](/papers/netvlad-arandjelovic-2016) con G clusters «fantasma» que **compiten en la asignación pero cuyos residuos se descartan**. Como el softmax es un presupuesto que suma 1, un descriptor de baja calidad puede gastar su masa en un fantasma y con eso su contribución a los clusters reales se atenúa. *"Una ponderación por calidad emerge automáticamente"*, sin que nadie etiquete qué es basura. Nacido en **reconocimiento de caras** a partir de conjuntos de imágenes. **Por qué importó:** un solo cluster fantasma vale +1,5 puntos de TAR en IJB-B, y trasladado al audio es lo que permite a un modelo de habla prescindir de detección de actividad de voz — el sumidero aprende a absorber el silencio. Desarmado en el [Lab 41](/laboratorios/lab-41).
    {{< /hito >}}
    {{< hito year="2019" name="NetVLAD para reconocimiento de hablante" status="covered" link="/papers/utterance-level-xie-2019" >}}
      Xie, Nagrani, Chung y Zisserman (VGG Oxford): un *thin ResNet-34* de 3 millones de parámetros más una capa **[NetVLAD](/papers/netvlad-arandjelovic-2016)** que agrega los frames acumulando residuos respecto de un diccionario aprendido, en vez de promediarlos. El experimento decisivo es interno: **con el mismo backbone y los mismos datos**, cambiar promedio temporal por NetVLAD lleva el EER de **10,48 % a 3,57 %**, y añadir [GhostVLAD](/papers/ghostvlad-zhong-2018) a 3,22 %. **Por qué importó:** mostró que en reconocimiento de hablante el componente que decide el rendimiento no es el extractor de features sino **cómo se agregan** — y trajo al audio una técnica nacida diez años antes en [búsqueda de imágenes](/papers/vlad-jegou-2010). Es el modelo de la [Clase 41](/clases/clase-41), reproducido y auditado en el [Lab 41](/laboratorios/lab-41): el EER se replica (3,19 %), pero al abrir el checkpoint la mitad del backbone está apagada y los 8 centroides «discriminativos» tienen coseno 0,9983 entre sí.
    {{< /hito >}}
    {{< hito year="2020" name="wav2vec 2.0" status="covered" link="/papers/wav2vec2-baevski-2020" >}}
      Baevski et al. (FAIR): cuantizar representaciones latentes y predecirlas con masking estilo BERT (pérdida contrastiva). **Por qué importó:** estableció el paradigma de pretraining masivo + fine-tuning ligero (10 min etiquetados bastan), base de Whisper y MMS. Cubierto en la [Clase 37](/clases/clase-37).
    {{< /hito >}}
    {{< hito year="2019" name="Deep Learning for Audio Signal Processing" status="covered" link="/papers/dl-audio-purwins-2019" >}}
      Purwins, Li, Virtanen, Schlüter, Chang y Sainath (IEEE JSTSP): el survey que consolidó el campo — representaciones, modelos, tareas, augmentation y datos, con el estado del arte de cada dominio de audio. **Por qué importó:** es el mapa de referencia de la disciplina justo antes de que el aprendizaje autosupervisado y los Transformers la reordenaran, y la fuente estructural de la [Clase 39](/clases/clase-39).
    {{< /hito >}}
    {{< hito year="2019" name="musicnn" status="covered" link="/papers/musicnn-pons-2019" >}}
      Pons y Serra (MTG, Universitat Pompeu Fabra): CNN para etiquetado musical con **filtros de forma musical** —verticales y angostos para el timbre, horizontales y largos para el ritmo— en vez de kernels cuadrados heredados de visión. **Por qué importó:** el argumento más claro de que los dos ejes de un espectrograma no son intercambiables, y modelos preentrenados abiertos para tareas musicales.
    {{< /hito >}}
    {{< hito year="2020" name="Conformer" status="deep" link="/papers/conformer-gulati-2020" >}}
      Gulati et al. (Google, Interspeech 2020): fusiona self-attention y convolución dentro de un mismo bloque, con embedding posicional relativo y estructura *macaron*. **Por qué importó:** conserva la tesis de la complementariedad local/global de CLDNN y solo reemplaza la recurrencia por atención — y con eso supera a modelos con diez veces más parámetros. Es la arquitectura estándar del reconocimiento de voz moderno.
    {{< /hito >}}
    {{< hito year="2021" name="HuBERT" status="deep" link="/papers/hubert-hsu-2021" >}}
      Hsu et al. (FAIR): predicción enmascarada estilo BERT sobre unidades discretas **fabricadas por clustering k-means** e iteradas sobre las propias representaciones del modelo. **Por qué importó:** resolvió el problema de que el habla no viene segmentada en entidades discretas, con el hallazgo de que las unidades del maestro no necesitan ser correctas sino consistentes. Base de textless NLP, GSLM y AudioLM.
    {{< /hito >}}
    {{< hito year="2021" name="AST: Audio Spectrogram Transformer" status="deep" link="/papers/ast-gong-2021" >}}
      Gong, Chung y Glass (MIT CSAIL): la primera arquitectura de audio sin convoluciones — parches solapados del espectrograma alimentando un ViT, con transferencia cross-modal desde ImageNet mediante interpolación del embedding posicional. **Por qué importó:** mostró que la tokenización del audio no necesita ser semántica, solo regular, y que el dataset masivo que faltaba en audio podía tomarse prestado de visión.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de foundation models" years="2022-presente" >}}
    {{< hito year="2022" name="Whisper" status="covered" link="/papers/whisper-radford-2022" >}}
      Radford et al. (OpenAI): Transformer encoder-decoder entrenado sobre 680,000 horas de audio multilingüe pareado con texto raspado de internet (weak supervision a gran escala). **Por qué importó:** ASR robusto a ruido, acentos y multilingüe sin fine-tuning, ya estándar industrial. Multilingüe desde v1 (~99 idiomas); v3 (nov 2023) refina rendimiento y agrega cantonés (100 idiomas totales). Cubierto en la [Clase 37](/clases/clase-37).
    {{< /hito >}}
    {{< hito year="2022" name="AudioLM" status="minimal" >}}
      Borsos et al. (Google): genera audio coherente continuando un prompt de pocos segundos, modelando audio como secuencia de tokens cuantizados. **Por qué importó:** mostró que la "generación pura" de audio (música, ambiente, voz) era viable con la receta autoregresiva tipo GPT.
    {{< /hito >}}
    {{< hito year="2023" name="VALL-E" status="minimal" >}}
      Wang et al. (Microsoft): TTS zero-shot que clona la voz de un hablante desde 3 segundos de audio. **Por qué importó:** llevó la clonación de voz neural a calidad de producción con datos mínimos.
    {{< /hito >}}
    {{< hito year="2023-2024" name="MusicLM / MusicGen / Suno / Udio" status="minimal" >}}
      Google (MusicLM), Meta (MusicGen), Suno y Udio: generación de música de calidad de producción condicionada en texto, alcanzando millones de usuarios en 2024. **Por qué importó:** mueve la frontera de la generación a un dominio creativo masivo.
    {{< /hito >}}
    {{< hito year="2023-2025" name="ElevenLabs / Sesame / TTS conversacional" status="minimal" >}}
      ElevenLabs (Multilingual v3, 2024), Sesame (2025): TTS de producción con expresividad, control de prosodia y latencia <300ms para conversación natural. **Por qué importó:** habilita la nueva generación de asistentes de voz nativos.
    {{< /hito >}}
  {{< /era >}}
{{< /timeline >}}

## Era 1 — Acústica clásica (1980-2010)

### Problema heredado

El audio es una serie temporal de altísima frecuencia. Una grabación de 10 segundos a 16 kHz son 160,000 muestras — una secuencia inviable para modelar directamente con cualquier técnica pre-2010. El reto era convertir esa señal en una representación discreta y compacta que algoritmos clásicos (HMMs, SVMs) pudieran procesar.

### Idea clave

**Espectrograma + modelos generativos.** El procesamiento de voz clásico parte de proyectar el audio a un espectrograma (típicamente MFCC), que descarta la fase y conserva información perceptualmente relevante en ~13-40 coeficientes por frame de 10ms. Sobre esa secuencia se ajustan **HMMs con emisiones gaussianas** (HMM-GMM): cada fonema es un HMM de 3 estados, cada estado emite MFCCs según una mezcla de gaussianas estimadas con EM. Para reconocer una palabra, se decodifica con Viterbi sobre la concatenación de HMMs por fonemas.

El modelo de lenguaje (n-gramas sobre transcripciones) se integra con el modelo acústico vía Weighted Finite-State Transducers — una composición de autómatas que combina acústica, pronunciación y lenguaje en un solo grafo decodificable.

### Qué la destronó

HMM-GMM tenían un techo: las gaussianas son discriminativas pobres entre fonemas que se traslapan en el espacio acústico. Hinton et al. mostraron en 2011 que reemplazar las gaussianas por una DNN (que aprende discriminativamente) bajaba el WER en ~30%. La era clásica terminó.

## Era 2 — Deep speech híbrido (2011-2014)

### Problema heredado

HMM-GMM saturaba: agregar más gaussianas o más datos no mejoraba significativamente. La capacidad expresiva del modelo acústico era la cota.

### Idea clave

**DNN reemplazando GMM dentro del HMM.** Hinton, Mohamed y Dahl (2011) entrenaron una red profunda que recibe un contexto de ~11 frames de MFCC y predice la probabilidad posterior de cada estado del HMM. La estructura HMM se mantiene (decoding Viterbi, integración con WFST), pero las emisiones son ahora discriminativas. La caída en WER fue inmediata y reproducible.

Kaldi (Povey et al., 2011) cristalizó la pila — ingeniería WFST + alineamiento + entrenamiento DNN — en un toolkit open-source que se volvió estándar industrial.

DeepSpeech 1 (Hannun et al., Baidu, 2014) dio el salto natural siguiente: si la DNN predice fonemas, ¿por qué no eliminarla del HMM y predecir directamente caracteres? Una CNN sobre el espectrograma + RNN bidireccional + CTC loss colapsó tres décadas de pipeline en un solo modelo entrenable end-to-end.

### Qué la destronó

DeepSpeech aún dependía de un decoder externo con modelo de lenguaje, y CTC tiene la suposición fuerte de que las predicciones por frame son condicionalmente independientes (lo cual es falso para lenguaje). La frontera natural era reemplazar CTC por **atención**.

## Era 3 — End-to-end con atención (2014-2018)

### Problema heredado

CTC funcionaba pero era rígido: emisión por frame con independencia condicional y un decoder Viterbi externo. Bahdanau et al. acababan de demostrar que la atención resolvía traducción automática sin alineamientos explícitos. La pregunta natural: ¿se puede hacer ASR como si fuera traducción audio→texto?

### Idea clave

**ASR como Seq2Seq con atención.** Listen, Attend and Spell (Chan et al., Google, 2015) es la respuesta canónica: un encoder pyramidal LSTM comprime la secuencia de espectrogramas, y un decoder LSTM con atención al estilo Bahdanau emite caracteres uno a uno, mirando dónde necesita en el encoder en cada paso. Sin CTC, sin HMM, sin WFST, sin lenguaje externo — el modelo aprende ortografía implícitamente.

DeepSpeech 2 (Amodei et al., 2015) llevó la receta CTC a calidad de producto con escala — modelos profundos, datos masivos, y entrenamiento distribuido — alcanzando paridad con humanos en inglés y mandarín en condiciones limpias.

RNN-Transducer (Graves, propuesta original 2012; consolidación de producción ~2017) combinó lo mejor de ambos mundos: predicción frame por frame como CTC, pero con un modelo de lenguaje interno autoregresivo. Resultó ser el algoritmo de ASR streaming de producción en Google y Apple — emite hipótesis sin esperar el final de la oración.

### Qué la destronó

Todas estas arquitecturas requerían **datos etiquetados pareados** (audio + transcripción) — un recurso escaso fuera del inglés. Mientras tanto, NLP estaba transformándose con BERT y la idea de **pretraining no supervisado**. ¿Se podía hacer lo mismo con audio?

## Era 4 — Self-supervised (2019-2021)

### Problema heredado

ASR de calidad requería miles de horas de audio transcrito por humanos — recurso disponible solo para ~10 idiomas. Para los 7,000 idiomas restantes, los modelos eran pobres o inexistentes. La pregunta abierta: ¿se puede pretrainar sobre audio puro (sin transcripciones) y luego fine-tunear con poca data etiquetada?

### Idea clave

**Pretraining contrastivo y predictivo sobre audio crudo.** wav2vec (Schneider et al., FAIR, 2019) entrenó un encoder convolucional a predecir representaciones futuras de la señal con una pérdida contrastiva, sin etiquetas. wav2vec 2.0 (Baevski et al., 2020) lo refinó al estilo BERT: cuantizar representaciones latentes en un codebook discreto, enmascarar segmentos del audio, y predecir las unidades cuantizadas correctas usando un Transformer.

El resultado: con 53,000 horas de audio sin etiquetar para pretraining + 10 minutos de audio etiquetado para fine-tuning, wav2vec 2.0 alcanzaba WER comparable a modelos supervisados de la generación anterior entrenados con miles de horas. HuBERT (Hsu et al., 2021) mejoró la receta con clustering iterativo de targets, dominando luego en TTS, identificación de hablante y reconocimiento de emoción.

### Qué la destronó

wav2vec 2.0 era enormemente eficiente en datos pero seguía requiriendo fine-tuning por dominio/idioma. La frontera natural: ¿se puede pretrainar en escala suficiente para tener un modelo zero-shot multilingüe que funcione fuera de la caja?

## Era 5 — Foundation models (2022-presente)

### Problema heredado

ASR seguía siendo un trabajo de fine-tuning por dominio. Cambiar de "transcribir podcast en inglés" a "transcribir Zoom call en español con ruido de fondo" requería ajuste. Y la generación de audio (TTS, música) vivía en un universo paralelo, con arquitecturas distintas.

### Idea clave

**Audio como una secuencia más, manejada por foundation models.** Whisper (Radford et al., OpenAI, 2022) entrenó un Transformer encoder-decoder estándar sobre 680,000 horas de audio multilingüe pareado con texto raspado de internet — un orden de magnitud más datos que cualquier modelo previo, y ruidosos por construcción. El resultado fue robustez sin fine-tuning a ~99 idiomas, ruido, acentos y dominios. Whisper se volvió estándar industrial inmediatamente.

En paralelo, AudioLM (Google, 2022) mostró que la generación de audio (música, ambiente, voz) podía hacerse como modelado autoregresivo de **tokens cuantizados** — el mismo paradigma que GPT, aplicado a audio. VALL-E (Microsoft, 2023) llevó la idea a TTS zero-shot: cloná una voz desde 3 segundos de audio. MusicLM/MusicGen/Suno/Udio (2023-2024) generan música de calidad de producción condicionada en texto, con millones de usuarios.

### Qué viene

La convergencia con los frontier LLMs es el hito en marcha: GPT-4o y Gemini 2.5 ya procesan audio nativamente como entrada/salida, sin pipeline ASR→LLM→TTS. La latencia conversacional bajó a <300ms (Sesame, ChatGPT Voice). En generación, las apuestas activas son **música con control fino** (estems separables, edición por prompt), **audio espacial 3D** para AR/VR, y **detección de deepfakes de voz** como contramedida a la clonación. La pregunta abierta: si los frontier LLMs absorben audio nativamente, ¿queda "audio" como dominio aislado o pasa a ser una modalidad más en modelos generales?

## Estado del arte hoy

{{< callout type="info" >}}

**Frontier audio (2024-2025).** El audio se está integrando nativamente a los foundation models generales. La generación (música, voz, sonido) alcanza calidad de producción y la conversación por voz baja a latencia humana.

- **Whisper v3** — OpenAI. ASR multilingüe estándar industrial; 100 idiomas, robusto a ruido y acentos.
- **GPT-4o / Gemini 2.5 audio nativo** — frontier LLMs con entrada/salida de voz directa, sin pipeline.
- **Sesame** — TTS conversacional con prosodia natural, latencia <300ms para diálogo en tiempo real.
- **ElevenLabs Multilingual v3** — TTS de producción con expresividad y control fino; clonación de voz desde segundos.
- **VALL-E 2 / NaturalSpeech 3** — TTS zero-shot con calidad de hablante humano.
- **Suno v4 / Udio v2** — música generativa por texto, calidad comercial; millones de usuarios activos.
- **MusicGen / Stable Audio 2** — generación de música y efectos para creadores y producción.

{{< /callout >}}

## Casos de uso reales

- **ASR en producción**: Zoom Live Transcript, Otter.ai, Apple Dictation, Google Recorder, transcripción de podcasts.
- **Asistentes de voz**: Alexa, Siri, Google Assistant; nueva generación con voz directa (ChatGPT Voice, Gemini Live).
- **Subtítulos automáticos**: YouTube, Twitch, plataformas de streaming, accesibilidad en tiempo real.
- **Audiolibros y TTS comercial**: Audible AI Narration, Google Play Books — TTS de producción.
- **Música generativa**: Suno y Udio para creadores; Stable Audio para producción profesional; MusicGen open-source.
- **Clonación de voz**: ElevenLabs para localización de contenido, accesibilidad, dubs multilingües.
- **Análisis de llamadas**: contact centers, compliance regulatorio, análisis de sentiment y QA.
- **Salud y diagnóstico**: detección de Parkinson, depresión y disfonía a partir de voz.

## Qué viene

Las apuestas activas en audio: **modelos de audio nativos en frontier LLMs** (audio in / audio out sin pipeline ASR→LLM→TTS, latencia <300ms), **música generativa con control fino** (estems separables, edición por prompt sobre piezas existentes), **detección de deepfakes de voz** como contramedida industrial a la clonación masiva, **audio espacial generativo** (3D, binaural) para AR/VR, y **foundation models multilingües** que cubran los 7,000 idiomas humanos — Whisper aún cubre <100. La pregunta abierta: ¿cuándo el "asistente de voz" será indistinguible de una llamada con un humano competente?

## Recursos relacionados

**Fundamentos (predecesores conceptuales):**
- [LSTM y GRU](/fundamentos/lstm-gru) — base de DeepSpeech, RNN-T y todos los ASR pre-Transformer.
- [Mecanismo de atención](/fundamentos/mecanismo-atencion) — fundamento de Listen, Attend and Spell y de los Transformers de audio actuales.
- [Self-attention](/fundamentos/self-attention) y [Transformer](/fundamentos/transformer) — la arquitectura sobre la que corren Whisper, AudioLM y VALL-E.
- [Aprendizaje contrastivo](/fundamentos/aprendizaje-contrastivo) — el principio que vertebra wav2vec.

**Papers (predecesores adyacentes):**
- [LSTM (Hochreiter 1997)](/papers/lstm-hochreiter-1997) — la red recurrente sobre la que se construyó toda la era 2-3.
- [Bahdanau attention (2015)](/papers/bahdanau-attention-2015) — el mecanismo de atención que LAS adaptó a audio.
- [Seq2Seq (Sutskever 2014)](/papers/seq2seq-sutskever-2014) — el patrón encoder-decoder.
- [Attention is All You Need (Vaswani 2017)](/papers/attention-is-all-you-need-vaswani-2017) — la arquitectura que reemplazó RNNs en audio post-2020.

**Dominios relacionados:**
- [Texto / NLP](/dominios/texto) — donde nacieron LSTM, atención, Transformer y self-supervised, todos transferidos luego a audio.
- [Multimodal](/dominios/multimodal) — donde audio se combina con texto e imagen en frontier LLMs.

---

*Última actualización: 2026-05-05.*
