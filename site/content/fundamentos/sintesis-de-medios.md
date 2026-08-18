---
title: "Síntesis de Medios (deepfakes)"
weight: 140
math: true
---

**Deepfake** es un término de prensa, no una categoría técnica. Agrupa métodos muy distintos cuyo único rasgo común es producir audio o video de una persona haciendo algo que no hizo. Este fundamento separa esas técnicas, porque tienen capacidades, requisitos y rastros forenses diferentes — y confundirlas lleva a conclusiones equivocadas tanto sobre lo que se puede hacer como sobre lo que se puede detectar.

Acompaña a la [Clase 44](/clases/clase-44), el cierre del diplomado.

---

## 1. Cinco técnicas distintas

| Técnica | Qué hace | Qué necesita | Ejemplo |
|---|---|---|---|
| **Face swap** | reemplaza el rostro de A por el de B dentro de un video existente | muchas imágenes de ambos | DeepFaceLab, FaceSwap |
| **Reenactment / animación** | transfiere expresión y pose de un video a **una sola imagen** | una imagen del objetivo | [First Order Motion Model](/papers/fomm-siarohin-2019) |
| **Lip sync** | modifica solo la boca para que coincida con otro audio | video + audio objetivo | Wav2Lip, Synthesizing Obama |
| **Clonación de voz (TTS)** | sintetiza habla nueva con el timbre de una persona | segundos de audio de referencia | [SV2TTS](/papers/sv2tts-jia-2018), VALL-E |
| **Generación completa** | crea la persona y la escena desde cero | nada del objetivo | Sora, Veo |

{{< concept-alert type="clave" >}}
La [Clase 44](/clases/clase-44) presenta [FOMM](/papers/fomm-siarohin-2019) bajo el rótulo de deep fakes, y conviene precisar: **FOMM hace animación de imágenes, no face swap**. No reemplaza un rostro dentro de un video existente; anima una imagen fija con el movimiento de un video conductor. La salida tiene la identidad, el fondo y el encuadre de la **imagen fuente**, no del video.

La diferencia es práctica: el face swap necesita cientos o miles de imágenes de la persona objetivo y produce un video que conserva el cuerpo y el entorno del original. El reenactment funciona **con una sola foto** —lo que baja enormemente la barrera de entrada— pero produce un plano corto, dependiente de la pose inicial, y con artefactos si el movimiento se aleja mucho de esa pose.
{{< /concept-alert >}}

## 2. El principio común: separar identidad de movimiento

Casi todos estos métodos comparten una estructura: **factorizar el contenido en un componente que se conserva y otro que se transfiere**.

- En FOMM, un codificador de apariencia produce la identidad y un detector de puntos clave produce el movimiento; el generador los recombina.
- En clonación de voz, un codificador de hablante produce un vector de timbre y un sintetizador produce la prosodia y el contenido a partir del texto.
- En face swap, el autoencoder aprende un espacio latente compartido con decodificadores separados por identidad.

Es la misma idea que atraviesa el diplomado con otros nombres: contenido contra estilo, aspecto contra dinámica, qué contra quién — la [Clase 41](/clases/clase-41) la usó para separar *qué se dijo* de *quién lo dijo*.

## 3. Los rastros que dejan

La detección se apoya en que estos métodos son **buenos localmente y descuidados globalmente**.

**Inconsistencias de bajo nivel.** Los generadores convolucionales dejan huellas espectrales características —patrones periódicos del *upsampling*— que no aparecen en imágenes de cámara. Es la señal que mejor generaliza… hasta que el video se recomprime, y la compresión de las redes sociales las destruye.

**Incoherencia temporal.** El parpadeo, el pulso visible en la piel, la consistencia de la iluminación entre cuadros. Los métodos que generan cuadro a cuadro fallan aquí más que los que modelan el tiempo.

**Inconsistencia fisiológica y semántica.** Reflejos en las córneas que no coinciden entre ojos, dentaduras que cambian, joyas que aparecen y desaparecen, desincronización entre labios y fonemas.

**Contexto y procedencia.** Con frecuencia lo decisivo no es la señal sino la verificación externa: metadatos, origen del archivo, contraste con otras grabaciones del mismo evento.

{{< concept-alert type="advertencia" >}}
**La detección generaliza mal.** Es el resultado más consistente de la literatura, y [FaceForensics++](/papers/faceforensics-rossler-2019) lo estableció con datos: los detectores entrenados sobre un método de generación caen fuertemente al evaluarse sobre otro, y caen otra vez al bajar la calidad de compresión.

Eso convierte la detección en una carrera asimétrica: el generador solo tiene que engañar a los detectores existentes, mientras el detector tiene que anticipar generadores que aún no existen. La conclusión operativa es que **no conviene depender de un clasificador de autenticidad como única defensa**, y que las líneas más prometedoras son las de **procedencia** —firmar criptográficamente el contenido en el momento de captura, como propone el estándar C2PA— en vez de detectar la falsificación después.
{{< /concept-alert >}}

## 4. Usos legítimos y daños documentados

La clase enumera aplicaciones útiles, y son reales: doblaje y localización de películas conservando la gesticulación, restauración y rejuvenecimiento de actores, recreación de figuras históricas en museos, avatares para videoconferencia con poco ancho de banda, y —el caso con más consenso ético— **prótesis de voz para personas que perdieron el habla** por ELA o cáncer laríngeo, sintetizada a partir de grabaciones previas.

Conviene poner al lado lo que la clase no menciona, porque es parte del mismo panorama:

- **Material sexual no consentido.** Los estudios independientes del fenómeno han encontrado consistentemente que constituye la enorme mayoría de los deepfakes que circulan, dirigido casi siempre contra mujeres. Es, con mucho, el uso más frecuente de estas técnicas.
- **Fraude por suplantación de voz.** Llamadas donde se clona la voz de un ejecutivo o un familiar para autorizar transferencias. Es hoy el vector con pérdidas económicas más documentadas, y basta con segundos de audio público.
- **Desinformación política**, cuyo efecto más estudiado no es tanto que se crea lo falso sino el **dividendo del mentiroso**: la existencia misma de la tecnología permite descartar grabaciones auténticas como fabricadas.

Varias jurisdicciones legislaron desde 2019 —obligaciones de etiquetado en el AI Act europeo, leyes específicas sobre imágenes íntimas no consentidas y sobre publicidad electoral en distintos países—. Es un terreno en movimiento y conviene consultar la norma local vigente antes que cualquier resumen.

## 5. Qué preguntar antes de construir con esto

Para trabajo aplicado, tres preguntas hacen la mayor parte del trabajo ético y evitan casi todos los problemas legales:

1. **¿Hay consentimiento de la persona cuya imagen o voz se sintetiza?** Es la línea que separa el doblaje de la suplantación, y no admite excepciones por "es solo una prueba".
2. **¿El resultado quedará marcado como sintético?** Marca de agua visible, metadatos de procedencia, o ambos.
3. **¿Qué pasa si el resultado circula fuera de contexto?** Un video hecho para una demo interna que termina en redes sin la etiqueta es el modo de falla más común.

Para el ámbito clínico, donde varios de estos métodos tienen usos genuinos —simulación para entrenamiento, anonimización de rostros en video de pacientes conservando la expresión— vale agregar que **anonimizar reemplazando el rostro no equivale a anonimizar**: la voz, el contexto y el movimiento siguen siendo identificadores.

---

## Ver también

- [First Order Motion Model (2019)](/papers/fomm-siarohin-2019) — el método de la clase, en detalle.
- [FaceForensics++ (2019)](/papers/faceforensics-rossler-2019) — el benchmark de detección y su resultado incómodo.
- [SV2TTS (2018)](/papers/sv2tts-jia-2018) — la clonación de voz que la clase usa para el audio.
- [Modelos Generativos](/fundamentos/modelos-generativos) y [Modelos de Difusión](/fundamentos/modelos-de-difusion) — la maquinaria de generación.
- [Super-resolución](/fundamentos/super-resolucion) — otra técnica donde el modelo aporta información que no estaba.
- [Clase 44](/clases/clase-44) · [Clase 29 - Modelos Generativos en Visión](/clases/clase-29)
