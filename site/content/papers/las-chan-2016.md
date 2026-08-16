---
title: "LAS: Listen, Attend and Spell (2016)"
weight: 437
math: true
---

{{< paper-card
    title="Listen, Attend and Spell: A Neural Network for Large Vocabulary Conversational Speech Recognition"
    authors="William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals (CMU, Google Brain)"
    year="2016"
    venue="ICASSP 2016 / arXiv:1508.01211"
    pdf="/papers/las-chan-2016.pdf" >}}
El primer sistema de reconocimiento de voz de vocabulario grande **enteramente neuronal y sin supuestos de independencia**: transcribe audio a caracteres sin diccionario, sin modelo de lenguaje y sin HMM. Dos componentes con nombres literales: el **listener** es un encoder recurrente **piramidal** que reduce la resolución temporal a la mitad en cada capa, y el **speller** es un decodificador con [atención](/papers/bahdanau-attention-2015) que emite caracteres uno a uno. La pirámide no es una optimización opcional — los autores reportan que **sin ella el modelo no converge**: tras un mes de entrenamiento los errores seguían muy por encima. Sobre una porción de Google voice search alcanza **14.1 % de WER** sin diccionario ni modelo de lenguaje, y **10.3 %** con rescoring, contra el 8.0 % del CLDNN-HMM de la época. Es el modelo que la [Clase 41](/clases/clase-41) describe en su Ejemplo 2, cuando agrega *"pooling in time window size=2"* al decoder.
{{< /paper-card >}}

---

## Contexto: la última pieza estadística

Hacia 2015 el reconocimiento de voz ya usaba redes profundas en todas partes, pero seguía siendo un sistema **compuesto**: un modelo acústico neuronal, un léxico de pronunciaciones, un modelo de lenguaje n-grama y un decodificador HMM que los ensamblaba. Cada pieza se entrenaba por separado, con su propio objetivo, y ninguna optimizaba directamente la métrica final.

[CTC](/papers/ctc-graves-2006) había eliminado el HMM del entrenamiento acústico, pero arrastra un supuesto fuerte: **independencia condicional entre salidas**. Dado el audio, la probabilidad de cada carácter no depende de los caracteres vecinos, lo que obliga a acoplar un modelo de lenguaje externo para producir texto plausible.

LAS elimina ese supuesto. Al ser el decodificador autorregresivo —cada carácter se condiciona a todos los anteriores— el modelo **aprende el lenguaje y la acústica juntos**, en una sola red.

## Método: escuchar en pirámide, deletrear con atención

$$P(y \mid x) = \prod_i P(y_i \mid x, y_{<i})$$

**El listener** es un BLSTM **piramidal** (pBLSTM). En cada capa, la entrada del paso $t$ concatena **dos** salidas consecutivas de la capa inferior:

$$h_t^{(j)} = \text{pBLSTM}\Big(h_{t-1}^{(j)},\; \big[h_{2t}^{(j-1)},\, h_{2t+1}^{(j-1)}\big]\Big)$$

Cada capa piramidal **divide por dos** el número de pasos temporales. Con tres capas, la reducción es de **8×**: un enunciado de 1 000 frames llega al decodificador como 125 vectores.

**El speller** es un decodificador LSTM con atención que emite caracteres, incluyendo espacios, puntuación y un token de fin. Al no haber diccionario, las palabras fuera de vocabulario se manejan solas —se deletrean— y el modelo puede producir variantes ortográficas legítimas: el paper reporta que para *"triple a"* genera tanto `triple a` como `aaa` entre sus mejores hipótesis, algo imposible para un sistema con léxico cerrado.

**Sampling durante el entrenamiento.** Para que el speller no se vuelva dependiente de la transcripción de referencia (*teacher forcing* puro), a veces se le alimenta su propia predicción en vez de la etiqueta correcta.

{{< concept-alert type="clave" >}}
**Los dos componentes son necesarios, y el paper lo demuestra por ablación.** Sin atención, el modelo **sobreajusta**: memoriza las transcripciones de entrenamiento sin escuchar la acústica, incluso con tres millones de enunciados. Sin la pirámide, **no converge**: tras un mes de entrenamiento los errores seguían muy por encima de los reportados. Ambos problemas tienen el mismo origen — las secuencias de audio tienen cientos o miles de frames, y eso hace muy difícil entrenar las RNN.
{{< /concept-alert >}}

## Resultados

Sobre un subconjunto de Google voice search (3 millones de enunciados):

| Sistema | WER |
|---|---|
| LAS, sin diccionario ni modelo de lenguaje | 14.1 % |
| LAS + rescoring con LM sobre los 32 mejores haces | **10.3 %** |
| CLDNN-HMM (estado del arte de la época) | 8.0 % |

LAS **no** ganaba en 2016 — perdía por 2.3 puntos contra el sistema híbrido maduro. Lo que hizo fue mostrar que un sistema end-to-end sin ninguna pieza estadística podía acercarse, y que la brecha era de ingeniería y datos, no de principio. El camino que abrió es el que recorren [Deep Speech 2](/papers/deep-speech-2-amodei-2015) y, en la era Transformer, [Whisper](/papers/whisper-radford-2022).

## Limitaciones

- **Todavía pierde contra los híbridos** en su propio benchmark.
- **No es causal**: el listener es bidireccional y la pirámide necesita el enunciado completo. Sin streaming.
- **El decodificador autorregresivo es secuencial** en inferencia: un carácter por paso, sin paralelizar.
- **La atención puede saltar hacia atrás** y producir repeticiones o borrados, porque nada impone monotonía — el problema que [Chorowski et al.](/papers/attention-asr-chorowski-2015) atacan con location-awareness.
- **Necesita muchos datos.** Tres millones de enunciados para un dominio acotado; con datasets pequeños los sistemas con [CTC](/papers/ctc-graves-2006) o los híbridos siguen siendo mejores.

## Por qué importa para la Clase 41

La clase construye dos ejemplos de arquitectura. El **Ejemplo 2** agrega al Ejemplo 1 dos cosas: CTC en la salida y *"Pooling in time window size = 2"* en el decoder. Ese pooling **es** la pirámide de LAS: reducir a la mitad los pasos temporales en cada capa para que la atención tenga menos posiciones que puntuar.

Y la razón de fondo no es el ahorro de cómputo sino la **convergencia**. La clase presenta el pooling como una técnica más para lidiar con la desalineación entrada-salida; el paper muestra que sin él el modelo directamente no entrena. Es una diferencia de énfasis que vale conocer.

En el mapa del [reconocimiento de voz](/fundamentos/reconocimiento-de-voz), LAS es el punto donde la rama de atención alcanza vocabulario abierto, y donde queda claro el intercambio con [CTC](/papers/ctc-graves-2006): CTC es más simple, más rápido y admite streaming, pero asume independencia entre salidas; la atención modela dependencias lingüísticas dentro de la red, al costo de ser secuencial y no causal.
