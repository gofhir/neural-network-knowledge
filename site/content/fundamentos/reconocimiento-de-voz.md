---
title: "Reconocimiento de Voz"
weight: 129
math: true
---

El **reconocimiento automático del habla** (ASR, *automatic speech recognition*) es la tarea de convertir una señal de audio en la secuencia de palabras que fue pronunciada. Parece un problema de clasificación y no lo es: lo que lo define es que **la entrada y la salida no vienen alineadas**. Un audio de tres segundos son unos 300 vectores de features y la transcripción son quizá 40 caracteres, sin ninguna indicación de qué frames produjeron cuál. Toda la historia técnica del área —de los HMM a [CTC](/fundamentos/ctc-loss), de ahí a la [atención](/fundamentos/mecanismo-atencion) y después a los Transformers— puede leerse como distintas respuestas a esa desalineación. Este fundamento presenta el problema, las dos grandes familias de soluciones y sus compromisos, y la métrica con que se miden.

---

## 1. Por qué el habla no es como los sonidos ambientales

Conviene empezar por el contraste, porque explica por qué la arquitectura que funciona en un caso no sirve en el otro.

En **clasificación de sonidos ambientales** —una sirena, un ladrido, un taladro— lo decisivo es el **aprendizaje de features**: el espacio de sonidos posibles es enorme y poco estructurado, y features específicos resultan muy discriminativos para cada clase. La combinación estándar de CNN (features locales) + RNN (contexto temporal) + MLP (clasificador) resuelve bien el problema.

En **habla** la situación se invierte:

| | Sonidos ambientales | Habla |
|---|---|---|
| Espacio de clases | prácticamente infinito | **acotado**: unas decenas de fonemas por idioma |
| Dificultad principal | discriminar features | **segmentar la secuencia** |
| Variabilidad | entre categorías | dentro de la categoría: voces, estilos, ruido |
| Lo que hay que aprender | representación | **estructura secuencial** |

Discriminar fonemas no es lo difícil: el espacio es chico y está bien estudiado. Lo difícil es **decidir dónde empieza y termina cada unidad** en una señal continua, sin marcas, donde la duración de un mismo fonema varía por un factor de tres según quién hable y a qué velocidad. La [Clase 41](/clases/clase-41) lo resume en una frase: *"feature learning is important but sequence learning is the real key"*.

---

## 2. El problema de la alineación, formalmente

Sea $x = (x_1, \dots, x_T)$ la secuencia de features acústicos —típicamente vectores log-mel de 40 dimensiones sobre ventanas de 25 ms con paso de 10 ms— e $y = (y_1, \dots, y_U)$ la transcripción. Se cumple casi siempre que $T \gg U$, y no se conoce la correspondencia.

Un clasificador frame a frame necesitaría una etiqueta por frame, lo que requiere una **alineación forzada** previa: el enfoque clásico DNN-HMM, donde un modelo estadístico entrenado aparte decide qué frames corresponden a qué estado fonético. Funciona, pero el modelo acústico no se optimiza para la métrica final, y hace falta un léxico de pronunciaciones construido a mano para cada idioma.

Las dos familias modernas eliminan esa dependencia por caminos distintos.

---

## 3. Familia A — CTC: marginalizar sobre las alineaciones

La idea de [CTC](/fundamentos/ctc-loss) (Graves et al., 2006) es agregar al vocabulario de salida un símbolo especial, el **blank** $\varnothing$, que significa "acá no emito nada". Con él, cualquier alineación de largo $T$ colapsa a una transcripción de largo $U$ mediante dos reglas: colapsar repeticiones consecutivas y eliminar los blanks.

$$\texttt{c-a-a-}\varnothing\texttt{-s-s-a} \;\longrightarrow\; \texttt{casa}$$

En lugar de elegir una alineación, CTC **suma sobre todas** las que producen la transcripción correcta:

$$P(y \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^{T} P(\pi_t \mid x)$$

donde $\mathcal{B}$ es la función de colapso. La suma tiene un número exponencial de términos y se calcula en tiempo polinomial con programación dinámica.

**La clase presenta el blank de forma intuitiva y correcta**: como *"un símbolo de espera o pausa"* que facilita la alineación. Formalmente es más que eso — es lo que hace que la suma sobre alineaciones esté bien definida.

**Ventajas:** entrenamiento simple, inferencia paralelizable, y admite operación causal (streaming), porque cada frame se decide localmente.

**El costo:** CTC asume **independencia condicional** entre las salidas dado el audio. La probabilidad de un carácter no depende de los caracteres vecinos, así que el modelo no aprende lenguaje: hay que acoplarle un modelo de lenguaje externo en la decodificación para producir texto plausible.

Ejemplos: [Graves et al. (2013)](/papers/deep-rnn-speech-graves-2013) con BiLSTM apiladas, [Deep Speech 2](/papers/deep-speech-2-amodei-2015) a escala industrial, y el fine-tuning de [wav2vec 2.0](/papers/wav2vec2-baevski-2020).

---

## 4. Familia B — Atención: aprender la alineación

La alternativa es un **encoder-decoder**: el encoder transforma el audio en una secuencia de representaciones, y un decodificador autorregresivo emite caracteres consultando esas representaciones mediante [atención](/fundamentos/mecanismo-atencion).

$$P(y \mid x) = \prod_{i} P(y_i \mid x, y_{<i})$$

La diferencia con CTC está en el condicionamiento: cada salida ve **todas las anteriores**, de modo que el modelo aprende acústica y lenguaje conjuntamente. No hace falta modelo de lenguaje externo (aunque suele ayudar).

Trasladar la atención de traducción al habla no es directo, y hay dos problemas documentados:

**Repetición acústica.** La atención por contenido puntúa cada frame por su parecido con el estado del decodificador. En un enunciado hay silencios y vocales acústicamente idénticos repartidos por todas partes, y el mecanismo no puede distinguir cuál es el que corresponde ahora. [Chorowski et al. (2015)](/papers/attention-asr-chorowski-2015) lo resuelven agregando **conciencia de ubicación**: alimentar la atención con dónde miró en el paso anterior.

**Costo sobre secuencias largas.** Cada carácter emitido requiere puntuar todos los frames. Con cientos de frames por segundo eso es prohibitivo, y la solución universal es **reducir la resolución temporal en el encoder**:

| Técnica | Dónde | Reducción |
|---|---|---|
| Encoder piramidal (pBLSTM) | [LAS](/papers/las-chan-2016) | ×2 por capa; ×8 con tres capas |
| Pooling sobre el tiempo | [Bahdanau et al. (2016)](/papers/e2e-lvsr-bahdanau-2016) | configurable |
| Strides convolucionales | [Deep Speech 2](/papers/deep-speech-2-amodei-2015), Transformers | ×4 típico |

{{< concept-alert type="nota" >}}
Las tres son la misma observación: **los frames de audio contiguos son enormemente redundantes**. A 100 frames por segundo, dos consecutivos comparten 15 ms de señal. Descartar la mitad casi no pierde información y divide por dos el trabajo del resto de la red. En [LAS](/papers/las-chan-2016) no es una optimización sino un requisito: sin la pirámide, el modelo **no converge**.
{{< /concept-alert >}}

---

## 5. Las dos familias, comparadas

| | CTC | Encoder-decoder con atención |
|---|---|---|
| Cómo resuelve la alineación | marginaliza sobre todas | la aprende explícitamente |
| Dependencia entre salidas | **ninguna** (independencia condicional) | completa (autorregresivo) |
| Modelo de lenguaje | externo, necesario | interno, opcional |
| Inferencia | paralela | secuencial |
| Streaming | sí | no, sin modificaciones |
| Monotonía | garantizada por construcción | no garantizada |
| Datos requeridos | menos | más |

No son excluyentes. Los sistemas modernos suelen usar **pérdidas híbridas** —CTC en el encoder más atención en el decodificador— porque el término CTC impone monotonía y acelera la convergencia, mientras la atención aporta el modelado del lenguaje. [Whisper](/papers/whisper-radford-2022) es encoder-decoder con Transformers; [wav2vec 2.0](/papers/wav2vec2-baevski-2020) hace fine-tuning con CTC. Ambos funcionan.

---

## 6. Cómo se mide: WER

La métrica estándar es la **tasa de error de palabras**:

$$\text{WER} = \frac{S + D + I}{N}$$

donde $S$ son sustituciones, $D$ borrados, $I$ inserciones y $N$ el número de palabras de la referencia. Se calcula alineando hipótesis y referencia con distancia de edición y contando las operaciones necesarias.

Cuatro propiedades que conviene tener presentes:

- **Puede superar el 100 %**, porque las inserciones no están acotadas por $N$. Un sistema que alucina texto puede tener WER de 150 %.
- **Trata todos los errores por igual.** Confundir "no" por "sí" cuesta lo mismo que equivocar un artículo, aunque el impacto sea incomparable.
- **Depende de la normalización.** Mayúsculas, puntuación, números escritos con letra o con cifra: sin un normalizador acordado, dos sistemas no son comparables.
- **En idiomas sin separación clara de palabras** (chino, japonés) se usa CER, sobre caracteres. Lo mismo cuando se quiere medir independientemente del vocabulario.

Para fonemas se usa **PER**, con la misma fórmula. Los resultados de [Graves et al. (2013)](/papers/deep-rnn-speech-graves-2013) y [Chorowski et al. (2015)](/papers/attention-asr-chorowski-2015) sobre TIMIT —17,7 % y 17,6 %— son PER, no WER, y no son comparables con los WER de LAS o Deep Speech 2.

---

## 7. La entrada

Casi todos los sistemas operan sobre **log-mel de 40 dimensiones**, con ventanas de 20-25 ms y paso de 10 ms. Esa configuración no es arbitraria:

- **25 ms** es el intervalo en que la señal de voz puede considerarse cuasi-estacionaria: suficiente para estimar el espectro, corto para que el tracto vocal no haya cambiado de configuración.
- **10 ms de paso** implica 60 % de solape, lo que suaviza la evolución temporal.
- **La escala mel** comprime las altas frecuencias imitando la resolución del oído, y 40 bandas bastan para el habla — ver [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel).

Los enfoques sobre [onda cruda](/fundamentos/representacion-de-audio) existen y funcionan, pero requieren más datos y más cómputo para aprender lo que el banco de filtros mel ya codifica.

---

## 8. Aplicaciones en salud

El ASR clínico tiene un caso de uso dominante —**dictado médico**— y varios emergentes: transcripción automática de la consulta para alimentar la ficha, extracción de información desde audio de rondas, y accesibilidad para pacientes con dificultades motoras.

Tres advertencias específicas del dominio:

- **El vocabulario es adverso.** Fármacos, procedimientos y anatomía son términos raros o inexistentes en los corpus generales, y son justamente los que no se pueden equivocar. Un WER global bajo puede convivir con errores sistemáticos en los términos que importan.
- **El WER global engaña.** Conviene medir por categoría de término —medicamentos, dosis, negaciones— porque la distribución de daño de los errores no es uniforme. Confundir "sin fiebre" por "con fiebre" son dos palabras de diferencia y una historia clínica distinta.
- **Grabar consultas es dato sensible.** Ver el marco de tratamiento de datos clínicos antes de enviar audio a un servicio externo; hay soluciones locales viables, y la existencia de modelos que corren en dispositivo es parte del argumento.

---

## Referencias

- Fundamentos relacionados: [CTC Loss](/fundamentos/ctc-loss) · [Mecanismo de atención](/fundamentos/mecanismo-atencion) · [Seq2Seq](/fundamentos/seq2seq) · [Representación de audio](/fundamentos/representacion-de-audio) · [MFCC y escala mel](/fundamentos/mfcc-y-escala-mel) · [Reconocimiento de hablante](/fundamentos/reconocimiento-de-hablante).
- Papers: [CTC (2006)](/papers/ctc-graves-2006) · [Deep RNN Speech (2013)](/papers/deep-rnn-speech-graves-2013) · [Attention-based ASR (2015)](/papers/attention-asr-chorowski-2015) · [LAS (2016)](/papers/las-chan-2016) · [E2E LVSR (2016)](/papers/e2e-lvsr-bahdanau-2016) · [Deep Speech 2 (2015)](/papers/deep-speech-2-amodei-2015) · [wav2vec 2.0 (2020)](/papers/wav2vec2-baevski-2020) · [Whisper (2022)](/papers/whisper-radford-2022).
- Datasets: [LibriSpeech](/papers/librispeech-panayotov-2015) · [Common Voice](/papers/common-voice-ardila-2020) · [Speech Commands](/papers/speech-commands-warden-2018).
- Clases: [Clase 41](/clases/clase-41) · [Clase 39](/clases/clase-39) · [Clase 35](/clases/clase-35).
- Dominio: [Audio](/dominios/audio).
