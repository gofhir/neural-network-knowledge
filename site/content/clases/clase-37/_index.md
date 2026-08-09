---
title: "Clase 37 - Datasets y Herramientas para Audio"
weight: 370
sidebar:
  open: true
---

**Profesores:** Alain Raymond y Gabriel Sepúlveda
**Módulo:** Audio y Video (Audio 2 de 5)

Segunda clase de audio: **de la teoría al dato**. Si la [Clase 35](/clases/clase-35) cubrió la teoría de señales (Fourier, muestreo, STFT, MFCC), esta baja al **ciclo de vida práctico del audio** en un proyecto de machine learning. Cuatro bloques: el **audio en el disco** (formatos, tamaño, sample rate, ffmpeg), **qué cambia al trabajar con audio** (el audio como tercera modalidad, transforms, tensores, batching, y la parte humana: consentimiento, privacidad, sesgo), **data augmentation** (SNR, SpecAugment, pitch/time y dónde va cada una), y **datasets** (tres escalas, cómo elegir por escala/disponibilidad/licencia/sesgo). Y el práctico donde todo se junta: un clasificador de géneros musicales sobre GTZAN, de los WAV al modelo.

## Apuntes de clase

{{< cards >}}
  {{< card link="teoria" title="Teoria" subtitle="Recorrido de las diapositivas: el audio en el disco, la tercera modalidad, data augmentation, datasets" icon="academic-cap" >}}
  {{< card link="profundizacion" title="Profundizacion" subtitle="Math: tamaño y tensor, SNR, SpecAugment, la pérdida contrastiva de wav2vec 2.0, la escala de weak supervision de Whisper" icon="beaker" >}}
  {{< card link="practica" title="Practica desde 0" subtitle="Del WAV al tensor (Mel + SNR + SpecAugment) y el batching de largo variable (collate_fn) en triple framework" icon="code" >}}
  {{< card link="/laboratorios/lab-37" title="Laboratorio: clasificación de géneros" subtitle="De los WAV al modelo sobre GTZAN con torchaudio; augmentation y embeddings wav2vec 2.0" icon="variable" >}}
  {{< card link="/clases/clase-35" title="Relacionada: Análisis de Audio" subtitle="La teoría de señales (Fourier, sampling, STFT, MFCC)" icon="academic-cap" >}}
  {{< card link="/clases/clase-38" title="Clase siguiente: CNN para reconocimiento en video" subtitle="Modelos pre-entrenados: la escalera de arquitecturas de video y el inflado de I3D" icon="arrow-right" >}}
  {{< card link="/clases/clase-36" title="Clase anterior: Análisis de Video" subtitle="La otra modalidad del módulo Audio y Video" icon="arrow-left" >}}
{{< /cards >}}

## Fundamentos relacionados

{{< cards >}}
  {{< card link="/fundamentos/representacion-de-audio" title="Representación de audio para ML" subtitle="El audio en el disco, formatos, la tercera modalidad, transforms, tensores" icon="book-open" >}}
  {{< card link="/fundamentos/datasets-de-audio" title="Datasets de audio" subtitle="Tres escalas, cómo elegir (escala/disponibilidad/licencia/sesgo)" icon="book-open" >}}
  {{< card link="/fundamentos/data-augmentation-de-audio" title="Data Augmentation de Audio" subtitle="SNR, SpecAugment, pitch/time, dónde va cada una en el pipeline" icon="book-open" >}}
  {{< card link="/fundamentos/representacion-tiempo-frecuencia" title="Representación tiempo-frecuencia" subtitle="La STFT y el espectrograma (base de las transforms)" icon="book-open" >}}
{{< /cards >}}

## Papers de esta clase

### Modelos y métodos

{{< cards >}}
  {{< card link="/papers/whisper-radford-2022" title="Whisper (2022)" subtitle="Radford et al. — ASR robusto con 680k h de weak supervision" icon="document-text" >}}
  {{< card link="/papers/wav2vec2-baevski-2020" title="wav2vec 2.0 (2020)" subtitle="Baevski et al. — embeddings de voz autosupervisados" icon="document-text" >}}
  {{< card link="/papers/specaugment-park-2019" title="SpecAugment (2019)" subtitle="Park et al. — augmentation directa sobre el espectrograma" icon="document-text" >}}
{{< /cards >}}

### Datasets

{{< cards >}}
  {{< card link="/papers/gtzan-tzanetakis-2002" title="GTZAN (2002)" subtitle="Tzanetakis & Cook — géneros musicales (el del lab)" icon="document-text" >}}
  {{< card link="/papers/urbansound8k-salamon-2014" title="UrbanSound8K (2014)" subtitle="Salamon et al. — taxonomía + sonido urbano, 10 folds" icon="document-text" >}}
  {{< card link="/papers/esc50-piczak-2015" title="ESC-50 (2015)" subtitle="Piczak — sonido ambiental, con techo humano" icon="document-text" >}}
  {{< card link="/papers/speech-commands-warden-2018" title="Speech Commands (2018)" subtitle="Warden — keyword spotting on-device" icon="document-text" >}}
  {{< card link="/papers/librispeech-panayotov-2015" title="LibriSpeech (2015)" subtitle="Panayotov et al. — el benchmark de ASR en inglés" icon="document-text" >}}
  {{< card link="/papers/musicnet-thickstun-2017" title="MusicNet (2017)" subtitle="Thickstun et al. — música clásica etiquetada nota a nota" icon="document-text" >}}
  {{< card link="/papers/audioset-gemmeke-2017" title="AudioSet (2017)" subtitle="Gemmeke et al. — ontología + 2M clips (link rot)" icon="document-text" >}}
  {{< card link="/papers/fsd50k-fonseca-2020" title="FSD50K (2020)" subtitle="Fonseca et al. — dataset abierto CC (resuelve el link rot)" icon="document-text" >}}
  {{< card link="/papers/common-voice-ardila-2020" title="Common Voice (2020)" subtitle="Ardila et al. — voz multilingüe crowdsourced CC0" icon="document-text" >}}
{{< /cards >}}

## Dominio relacionado

{{< cards >}}
  {{< card link="/dominios/audio" title="Dominio: Audio / Voz" subtitle="Línea de tiempo: de MFCC y HMM-GMM a wav2vec, Whisper y los foundation models de audio" icon="globe-alt" >}}
{{< /cards >}}
