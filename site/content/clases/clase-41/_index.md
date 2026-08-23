---
title: "Clase 41 - Speech Recognition y Speaker Recognition"
weight: 410
sidebar:
  open: true
---

**Profesor:** Gabriel Sepúlveda (IALab, Departamento de Ciencia de la Computación, PUC)
**Módulo:** Audio — aplicaciones sobre habla

La clase son en realidad **dos clases**, y conviene leerlas juntas porque su relación es de oposición. La primera aborda el **reconocimiento de voz**: qué se dijo. La segunda, el **reconocimiento de hablante**: quién lo dijo. Y lo que cada una necesita de la representación es exactamente lo contrario de lo que necesita la otra.

El reconocimiento de voz requiere **resolución temporal** —cada fonema en su lugar, con una salida por unidad de tiempo— y su problema central es que la entrada y la salida no vienen alineadas. El reconocimiento de hablante requiere **colapsar el enunciado entero** en un único vector que descarte el contenido lingüístico y conserve lo invariante de la voz; su problema central es cómo agregar cientos de frames sin perder lo que distingue a una persona.

{{< concept-alert type="clave" >}}
**El hilo de la primera mitad.** La clase 39 concluyó que para sonidos ambientales la receta es CNN + RNN + MLP. La clase 41 abre preguntando si sirve para el habla, y se responde: *"Sure you do, but it does not achieve state-of-the-art performance"*. La razón es que en sonidos ambientales lo decisivo es **aprender features** —el espacio de sonidos posibles es enorme—, mientras que en habla el espacio de fonemas es acotado y lo decisivo es **aprender la secuencia**: dónde empieza y termina cada unidad en una señal continua. *"Feature learning is important but sequence learning is the real key."*

**El hilo de la segunda mitad.** Modelar el reconocimiento de hablante como clasificador tiene un defecto fatal: *"our model must be trained entirely for each new speaker"*. La alternativa es producir un **descriptor** y comparar por similitud, lo que convierte el problema en uno de conjunto abierto. Y ahí lo que decide el rendimiento no es el backbone sino **cómo se agregan los frames**.
{{< /concept-alert >}}

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Las 88 diapositivas de los dos PDFs: por qué el habla no es como los sonidos ambientales, seq2seq con atención aplicado al audio, el token blank y CTC, WER — y del clasificador al descriptor, VLAD paso a paso, NetVLAD, umbral y curva ROC" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: la suma sobre alineaciones de CTC y su programación dinámica, por qué la independencia condicional obliga a un LM externo, el gradiente que bloquea el argmin, y la geometría de residuos que el promedio destruye" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Implementar CTC y verificar que la suma sobre alineaciones coincide con la fuerza bruta; construir VLAD y NetVLAD y medir EER sobre una ROC — en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-41" title="Laboratorio 41" subtitle="El modelo de la segunda mitad, implementado y medido: reproduce el EER de 3,22 % y al abrir el checkpoint encuentra los 8 centroides colapsados, los 2 fantasmas muertos y la mitad del backbone apagada" icon="variable" >}}
  {{< card link="/clases/clase-39" title="Clase anterior: Modelos de DL para Audio" subtitle="CNN+RNN+MLP para sonidos ambientales — la receta que esta clase pone en duda para el habla" icon="arrow-left" >}}
  {{< card link="/laboratorios/lab-39" title="Laboratorio 39" subtitle="Onda cruda y VGGish: el lab del bloque de audio anterior, con la agregación de predicciones por clip" icon="variable" >}}
  {{< card link="/clases/clase-35" title="Relacionada: Introducción al Análisis de Audio" subtitle="Fourier, STFT y MFCC: de dónde salen los 40 log-mel que entran a estos modelos" icon="academic-cap" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/reconocimiento-de-voz" title="Reconocimiento de Voz" subtitle="El problema de la alineación, las familias CTC y atención, y por qué WER puede superar el 100 %" icon="book-open" >}}
  {{< card link="/fundamentos/reconocimiento-de-hablante" title="Reconocimiento de Hablante" subtitle="Identificación contra verificación, el descriptor de conjunto abierto, EER y la curva ROC" icon="book-open" >}}
  {{< card link="/fundamentos/agregacion-vlad" title="Agregación VLAD" subtitle="De contar ocupación a acumular residuos, y el argmin que había que volver derivable" icon="book-open" >}}
  {{< card link="/fundamentos/ctc-loss" title="CTC Loss" subtitle="El token blank, el colapso de alineaciones y la programación dinámica" icon="book-open" >}}
  {{< card link="/fundamentos/mecanismo-atencion" title="Mecanismo de Atención" subtitle="El contexto adaptativo que reemplaza al embedding fijo del seq2seq" icon="book-open" >}}
  {{< card link="/fundamentos/seq2seq" title="Seq2Seq" subtitle="Encoder-decoder: el marco que la clase recuerda antes de aplicarlo al habla" icon="book-open" >}}
  {{< card link="/fundamentos/mfcc-y-escala-mel" title="MFCC y Escala Mel" subtitle="Los 40 coeficientes log-mel sobre ventanas de 25 ms que son la entrada de todo" icon="book-open" >}}
  {{< card link="/fundamentos/metric-learning" title="Metric Learning" subtitle="Aprender un espacio donde la distancia significa identidad" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### Parte 1 — Reconocimiento de voz

{{< cards >}}
  {{< card link="/papers/seq2seq-sutskever-2014" title="Seq2Seq (2014)" subtitle="Sutskever et al. — el encoder-decoder que la clase recuerda; el embedding intermedio C es fijo" icon="document-text" >}}
  {{< card link="/papers/bahdanau-attention-2015" title="Atención (2015)" subtitle="Bahdanau et al. — el contexto Ct adaptativo, y el modelo que la clase intenta trasladar al habla" icon="document-text" >}}
  {{< card link="/papers/deep-rnn-speech-graves-2013" title="Deep RNN Speech (2013)" subtitle="Graves et al. — las BiLSTM apiladas del Ejemplo 1, con 17.7 % de PER en TIMIT" icon="document-text" >}}
  {{< card link="/papers/ctc-graves-2006" title="CTC (2006)" subtitle="Graves et al. — el token blank y la suma sobre alineaciones que la clase presenta como IDEA" icon="document-text" >}}
  {{< card link="/papers/attention-asr-chorowski-2015" title="Attention-based ASR (2015)" subtitle="Chorowski et al. — por qué la atención de traducción falla en habla, y la conciencia de ubicación" icon="document-text" >}}
  {{< card link="/papers/las-chan-2016" title="LAS (2016)" subtitle="Chan et al. — el encoder piramidal que es el 'pooling in time' del Ejemplo 2; sin él, el modelo no converge" icon="document-text" >}}
  {{< card link="/papers/e2e-lvsr-bahdanau-2016" title="E2E LVSR (2016)" subtitle="Bahdanau et al. — el 'Pooling Over Time' que la clase cita junto a CTC, y por qué no son lo mismo" icon="document-text" >}}
  {{< card link="/papers/deep-speech-2-amodei-2015" title="Deep Speech 2 (2015)" subtitle="Amodei et al. — la rama CTC llevada a escala: inglés y mandarín con la misma arquitectura" icon="document-text" >}}
{{< /cards >}}

### Parte 2 — Reconocimiento de hablante

{{< cards >}}
  {{< card link="/papers/utterance-level-xie-2019" title="Utterance-level Aggregation (2019)" subtitle="Xie et al. — el modelo de la clase: thin ResNet + NetVLAD. Cambiar solo la agregación lleva el EER de 10.48 % a 3.57 %" icon="document-text" >}}
  {{< card link="/papers/vlad-jegou-2010" title="VLAD (2010)" subtitle="Jégou et al. — los cinco pasos que la clase enumera, nacidos en búsqueda de imágenes" icon="document-text" >}}
  {{< card link="/papers/netvlad-arandjelovic-2016" title="NetVLAD (2016)" subtitle="Arandjelović et al. — el softmax que reemplaza al argmin y responde el '¿se puede aprender end-to-end?' de la clase" icon="document-text" >}}
  {{< card link="/papers/ghostvlad-zhong-2018" title="GhostVLAD (2018)" subtitle="Zhong, Arandjelović y Zisserman — los clusters que compiten en la asignación y cuyos residuos se descartan: el VAD implícito del modelo de la clase" icon="document-text" >}}
  {{< card link="/papers/voxceleb-nagrani-2017" title="VoxCeleb (2017)" subtitle="Nagrani et al. — 1 251 hablantes etiquetados sin que nadie escuche: un pipeline de visión por computador" icon="document-text" >}}
  {{< card link="/papers/voxceleb2-chung-2018" title="VoxCeleb2 (2018)" subtitle="Chung et al. — los 5 994 hablantes de entrenamiento, disjuntos de VoxCeleb1" icon="document-text" >}}
  {{< card link="/papers/x-vectors-snyder-2018" title="x-vectors (2018)" subtitle="Snyder et al. — statistics pooling y la línea base contra la que compite VLAD" icon="document-text" >}}
{{< /cards >}}

---

**Ver también:** [Clase 39 - Modelos de DL para Audio](/clases/clase-39) · [Clase 37 - Datasets y Herramientas para Audio](/clases/clase-37) · [Clase 13 - Seq2Seq y Atención](/clases/clase-13) (el marco que esta clase aplica al audio) · [Lab 39](/laboratorios/lab-39) · Dominio [Audio](/dominios/audio).
