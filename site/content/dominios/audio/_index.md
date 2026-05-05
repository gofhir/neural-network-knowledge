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
  {{< era name="Era acústica clásica" years="1980-2010" >}}
    {{< hito year="1980" name="MFCC" status="minimal" >}}
      Davis & Mermelstein: *Mel-Frequency Cepstral Coefficients*, una representación que imita la sensibilidad logarítmica del oído humano a la frecuencia. **Por qué importó:** la entrada estándar de toda ASR clásica durante 30 años.
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
    {{< hito year="2014" name="CTC loss" status="covered" link="/fundamentos/lstm-gru" >}}
      Graves: *Connectionist Temporal Classification* — pérdida que permite entrenar secuencia-a-secuencia sin alineamiento explícito entre audio y texto. Resolvió el problema central de DeepSpeech.
    {{< /hito >}}
    {{< hito year="2015" name="Listen, Attend and Spell" status="covered" link="/fundamentos/mecanismo-atencion" >}}
      Chan et al. (Google): primer ASR puramente atencional. Encoder-decoder con atención al estilo Bahdanau, sin CTC. Ortografía aprendida implícitamente.
    {{< /hito >}}
    {{< hito year="2015" name="DeepSpeech 2" status="minimal" >}}
      Amodei et al. (Baidu): escala de DeepSpeech 1 — más datos, más profundidad, RNN bidireccional con CTC. **Por qué importó:** mostró que ASR neural podía escalar a calidad de producto en inglés y mandarín.
    {{< /hito >}}
    {{< hito year="2017" name="RNN-Transducer" status="minimal" >}}
      Graves: combinación de CTC con un modelo de lenguaje interno autoregresivo. **Por qué importó:** el algoritmo de ASR streaming de producción en Google y Apple — funciona online sin esperar el final de la oración.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era self-supervised" years="2019-2021" >}}
    {{< hito year="2019" name="wav2vec" status="minimal" >}}
      Schneider et al. (FAIR): pretraining no supervisado de representaciones de audio prediciendo el futuro de la señal. **Por qué importó:** primer "BERT para audio" — fine-tuning con pocos datos etiquetados igualó a modelos supervisados con miles de horas.
    {{< /hito >}}
    {{< hito year="2020" name="wav2vec 2.0" status="minimal" >}}
      Baevski et al. (FAIR): cuantizar representaciones latentes y predecirlas con masking estilo BERT. **Por qué importó:** estableció el paradigma de pretraining masivo + fine-tuning ligero, base de Whisper y MMS.
    {{< /hito >}}
    {{< hito year="2021" name="HuBERT" status="minimal" >}}
      Hsu et al. (FAIR): mejora wav2vec 2.0 con clustering iterativo de targets discretos. **Por qué importó:** mejor rendimiento en transferencia a tareas de TTS, identificación de hablante y emoción.
    {{< /hito >}}
  {{< /era >}}
  {{< era name="Era de foundation models" years="2022-presente" >}}
    {{< hito year="2022" name="Whisper" status="minimal" >}}
      Radford et al. (OpenAI): Transformer encoder-decoder entrenado sobre 680,000 horas de audio multilingüe pareado con texto raspado de internet. **Por qué importó:** ASR robusto a ruido, acentos y multilingüe sin fine-tuning, ya estándar industrial. v3 (2023) extiende a 99 idiomas.
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
