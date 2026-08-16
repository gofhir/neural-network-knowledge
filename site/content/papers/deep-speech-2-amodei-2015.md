---
title: "Deep Speech 2 (2015)"
weight: 439
math: true
---

{{< paper-card
    title="Deep Speech 2: End-to-End Speech Recognition in English and Mandarin"
    authors="Dario Amodei, Awni Hannun, Adam Coates, Andrew Ng et al. (Baidu Research – Silicon Valley AI Lab)"
    year="2015"
    venue="ICML 2016 / arXiv:1512.02595"
    pdf="/papers/deep-speech-2-amodei-2015.pdf" >}}
La demostración de que el reconocimiento de voz end-to-end **escala**, y de que la escala es en buena medida un problema de ingeniería de cómputo. Un solo sistema —convoluciones más RNN bidireccionales entrenadas con [CTC](/papers/ctc-graves-2006)— reemplaza pipelines enteros de componentes diseñados a mano y funciona tanto en **inglés como en mandarín**, dos lenguas que no comparten ni fonología ni escritura, sin cambiar la arquitectura. El inglés se entrena con **11 940 horas** de habla y el mandarín con **9 400**. Las técnicas que lo hacen posible son tres: **BatchNorm adaptado a RNN**, un currículum de entrenamiento llamado **SortaGrad** que ordena los enunciados de corto a largo, y una batería de optimizaciones de HPC que dan un **speedup de 7×** sobre el sistema anterior. El resultado más citado: en varios benchmarks estándar el sistema **iguala o supera la transcripción de trabajadores humanos**.
{{< /paper-card >}}

---

## Contexto: el pipeline como obstáculo

Un sistema de reconocimiento tradicional es una cadena de componentes especializados: extracción de features diseñados a mano, modelo acústico, léxico de pronunciaciones, modelo de lenguaje, decodificador. Cada uno requiere experiencia de dominio, y cada uno debe **rehacerse** para un idioma nuevo — el léxico de pronunciaciones del mandarín no se parece en nada al del inglés.

La apuesta de la línea Deep Speech es que una red suficientemente grande, entrenada con suficientes datos y una pérdida que no requiera alineación, puede absorber toda esa cadena. El primer paper (Hannun et al., 2014) lo mostró en inglés; este lo lleva a dos idiomas, a escala industrial, y con la infraestructura documentada.

## Método: la arquitectura es lo de menos

El modelo es directo: varias capas **convolucionales** sobre el espectrograma, seguidas de varias capas **recurrentes bidireccionales**, una capa totalmente conectada y una salida con [CTC](/papers/ctc-graves-2006) sobre caracteres. Lo interesante son las decisiones que lo hacen entrenable:

**BatchNorm para RNN.** Normalizar dentro de capas recurrentes no es trivial —las estadísticas cambian con el paso temporal— y el paper detalla la variante que funciona. Acelera la convergencia y permite redes más profundas.

**SortaGrad.** Un currículum: en la primera época los enunciados se presentan **ordenados de más corto a más largo**. Las secuencias largas tienen gradientes más inestables al principio del entrenamiento; empezar por las cortas evita divergencias tempranas.

**Strides largos entre entradas de la RNN.** Submuestrear el tiempo antes de la parte recurrente reduce el cómputo sin perder precisión apreciable — la misma observación de redundancia entre frames vecinos que motiva la pirámide de [LAS](/papers/las-chan-2016) y el pooling temporal de [Bahdanau et al.](/papers/e2e-lvsr-bahdanau-2016)

**Ingeniería de sistemas.** Kernels de GPU específicos, all-reduce sincrónico, y para el despliegue un esquema de *batch dispatch* que agrupa peticiones de usuarios distintos para explotar el paralelismo sin destruir la latencia. Es una parte sustancial del paper, y su tesis de fondo: si un experimento tarda semanas, no se hacen experimentos.

## Resultados

El sistema alcanza o supera la precisión de transcriptores humanos (medidos con trabajadores de Mechanical Turk) en varios benchmarks estándar de inglés, y cierra la brecha significativamente en el resto. En mandarín, la misma arquitectura entrenada con 9 400 horas produce un sistema competitivo sin ninguna adaptación específica al idioma — que era el punto a demostrar.

## Limitaciones

- **"Supera a humanos" tiene letra chica.** La comparación es contra transcriptores de crowdsourcing sobre *benchmarks limpios y de dominio acotado*, no contra transcriptores profesionales ni en condiciones difíciles. En audio ruidoso, con acentos poco representados o habla espontánea superpuesta, la brecha persiste.
- **Los datos son el requisito real.** Doce mil horas etiquetadas no están al alcance de la mayoría de los idiomas ni de la mayoría de los dominios. Es precisamente la restricción que el aprendizaje autosupervisado ([wav2vec 2.0](/papers/wav2vec2-baevski-2020)) y la supervisión débil a escala ([Whisper](/papers/whisper-radford-2022)) atacarán después.
- **Bidireccional, o sea offline.** Las variantes causales para streaming pierden precisión.
- **CTC mantiene el supuesto de independencia condicional** entre salidas, así que un modelo de lenguaje externo sigue aportando en la decodificación.

## Por qué importa para la Clase 41

La clase enumera los tipos de modelo aplicables al habla y desarrolla la rama **seq2seq con atención**. Deep Speech 2 es la rama paralela llevada a su extremo: **CTC más escala**, sin atención en absoluto.

Vale tenerla presente porque delimita lo que la clase atribuye a la arquitectura. Cuando el material dice que en el habla *"sequence learning is the real key"*, este paper muestra que buena parte de ese aprendizaje de secuencias puede resolverse con una pérdida que ignora las dependencias entre salidas, siempre que haya datos suficientes. La arquitectura importa; los datos importan más.

En el mapa del [reconocimiento de voz](/fundamentos/reconocimiento-de-voz), es el punto donde la rama CTC alcanza calidad de producción, y donde queda establecido el patrón que domina hasta hoy: **subsampling temporal en el encoder** —convolucional acá, piramidal en LAS, por stride en los Transformers— para que el resto de la red trabaje sobre una secuencia manejable.
