---
title: "Parte 3 - Teacher Forcing"
weight: 30
math: true
---

Las partes 1 y 2 del lab construyeron un seq2seq y le agregaron attention sobre el encoder. Pero hay un detalle del entrenamiento que se mantuvo implicito en ambas y que merece tratamiento aparte: **como se le entrega el input al decoder en cada paso del entrenamiento**. La respuesta natural ("usar la prediccion del paso anterior") suena correcta pero hace que el modelo casi no aprenda. La parte 3 introduce **teacher forcing** como la tecnica estandar para entrenar decoders autoregresivos, explica por que funciona, y discute el costo conceptual asociado: el **exposure bias**.

## Que es teacher forcing

El decoder de un seq2seq es **autoregressive**: en cada paso $t$ produce un token $\hat{y}_t$ condicionado en los tokens anteriores. La pregunta es cuales tokens anteriores usa como input al paso $t$, y la respuesta cambia segun la fase:

- Durante **training**: en cada paso del decoder, en lugar de usar la prediccion del paso anterior $\hat{y}_{t-1}$ como input al siguiente paso, se usa el **ground truth** $y_{t-1}^*$ (el token correcto del target) *(notebook 3, cell 13)*. Esto es teacher forcing.
- Durante **inference / evaluation**: no hay ground truth disponible (se esta traduciendo una oracion nueva), asi que el decoder se alimenta de su propia prediccion del paso anterior $\hat{y}_{t-1}$. Esto es el modo **autoregressive** o *free-running*.

Formalmente, en teacher forcing el decoder modela cada paso como:

$$
P(y_t \mid y_{<t}^{*}, x; \theta)
$$

donde $y_{<t}^{*}$ son los tokens **reales** del target hasta $t-1$, no las predicciones del propio modelo. La loss sigue siendo cross-entropy por paso, sumada a lo largo de la secuencia *(notebook 3, cell 13 — markdown bloque "TEACHER FORCING")*.

## Por que se usa

Tres razones, todas relacionadas con la dinamica del entrenamiento:

**1. Convergencia mas rapida.** En las primeras epocas el modelo todavia no sabe nada y sus predicciones son basicamente ruido. Si lo alimentaramos con ese ruido en cada paso, el decoder estaria aprendiendo a continuar secuencias *aleatorias*, no las del dominio real. Con teacher forcing el decoder siempre ve la distribucion correcta de tokens previos y puede enfocarse en aprender el mapping condicional $P(y_t \mid y_{<t}^*, x)$ desde el primer batch.

**2. Evita compounding de errores.** Sin teacher forcing, un error temprano en la secuencia (por ejemplo, predecir mal $\hat{y}_2$) contamina el input de **todos** los pasos siguientes. El decoder nunca llega a un punto donde pueda recuperarse, y el gradiente se vuelve casi inutil porque cada paso esta "calculando" una loss sobre un contexto que ya no tiene relacion con el target real.

**3. Permite paralelizacion.** Como en training todos los inputs del decoder son los $y_{t-1}^*$ del target (que ya conocemos), todos los pasos del decoder pueden computarse en una sola pasada con un *shift-right mask* — no hay que esperar a generar $\hat{y}_{t-1}$ antes de calcular el paso $t$. Este punto es el que hace viable el entrenamiento del Transformer.

## Exposure bias (el problema)

Teacher forcing tiene un costo conceptual conocido como **exposure bias**: la distribucion de inputs que ve el decoder durante training es **distinta** de la que ve durante inference.

- En training, el decoder *siempre* recibe ground truth como input al paso siguiente.
- En inference, el decoder *nunca* recibe ground truth — recibe sus propias predicciones, que pueden tener errores.

El resultado es que el modelo se entreno asumiendo un input "limpio" pero en deployment se le entrega un input "ruidoso". Si en algun momento de la generacion comete un error y produce un $\hat{y}_t$ raro, el modelo nunca vio nada parecido durante training y la calidad puede degradarse rapidamente — un fenomeno tipico en NMT antiguo donde el comienzo de la traduccion es bueno pero el final divaga.

Hay varias mitigaciones propuestas en la literatura, mencionadas conceptualmente:

- **Scheduled sampling** ([Bengio et al. 2015](https://arxiv.org/abs/1506.03099)): mezclar gradualmente teacher forcing y free-running durante el entrenamiento. Con probabilidad $\epsilon$ se usa ground truth y con probabilidad $1 - \epsilon$ la prediccion del modelo. $\epsilon$ decae de 1 a 0 a lo largo de las epocas, de modo que el modelo se va "acostumbrando" a sus errores.
- **Professor forcing** ([Lamb et al. 2016](https://arxiv.org/abs/1610.09038)): usar adversarial training para forzar que las distribuciones de hidden states del decoder en modo teacher-forcing y en modo free-running sean indistinguibles para un discriminador.

En la practica, modernamente se sigue entrenando con teacher forcing puro (es lo que hace el Transformer original) y se acepta el exposure bias como un trade-off. Las arquitecturas grandes preentrenadas son lo bastante robustas como para que el problema no domine en secuencias cortas-medias.

## Implementacion en el notebook

El notebook 3 retoma el modelo de la parte 2 (seq2seq con attention sobre SCAN) y modifica el bucle de entrenamiento para introducir teacher forcing controlado *(notebook 3, cells 13-14, 26-32)*. La idea es exponer un parametro `teacher_forcing_ratio` que controle, por step del decoder, con que probabilidad se usa el ground truth versus la prediccion del modelo:

$$
\text{input}_t \;=\; \begin{cases} y_{t-1}^{*} & \text{con probabilidad } p_{\text{tf}} \\ \hat{y}_{t-1} & \text{con probabilidad } 1 - p_{\text{tf}} \end{cases}
$$

Para `teacher_forcing_ratio = 1.0` el comportamiento es teacher forcing puro (siempre ground truth). Para `teacher_forcing_ratio = 0.0` se reduce a entrenar en modo libre / autoregresivo desde el primer paso. Valores intermedios (tipicos: 0.5) corresponden al esquema de scheduled sampling estatico — sin decay durante el training.

En pseudocodigo, el loop del decoder en cada batch queda:

```python
input_t = SOS_token
for t in range(max_output_len):
    output_t, hidden = decoder(input_t, hidden, encoder_outputs)
    loss += criterion(output_t, target[t])
    use_teacher = random.random() < teacher_forcing_ratio
    input_t = target[t] if use_teacher else output_t.argmax(-1)
```

El resto del modelo (encoder BiLSTM, attention aditiva, decoder LSTM con context vector) se mantiene identico al de la parte 2 *(notebook 3, cell 26)*. Lo unico que cambia es la fuente del input al decoder en cada step.

## Resultados comparativos

`[outputs pendientes — se integraran en Fase 2 cuando Roberto ejecute el notebook en Colab]`

La parte 3 esta disenada para producir una comparacion directa entre dos regimenes de entrenamiento del mismo modelo seq2seq+attention sobre SCAN:

- **Con teacher forcing** (`teacher_forcing_ratio` alto, p.ej. 1.0 o 0.5): convergencia rapida, training loss baja agresivamente en las primeras epocas.
- **Sin teacher forcing** (`teacher_forcing_ratio = 0.0`): convergencia lenta o estancada, especialmente al inicio cuando las predicciones del decoder son aleatorias.

Las metricas que el notebook va a graficar son las usuales:

- **Curvas de loss** (train y val) por epoca para cada regimen.
- **Calidad de las traducciones** generadas en inference sobre algunos ejemplos del test set, para ver si la diferencia en convergencia se traduce en una diferencia en la calidad final del modelo entrenado.

La intuicion que el lab quiere construir es clara: teacher forcing es una intervencion **en la dinamica de optimizacion**, no un cambio en la arquitectura. Cambiar el ratio cambia la velocidad y estabilidad del aprendizaje, no la capacidad expresiva del modelo. Las **Actividades 1.1 y 1.2** al final del notebook piden razonar exactamente sobre esto — pero esas preguntas y sus respuestas viven en otras paginas del lab.

Volver a [Parte 2 — Seq2Seq con Attention](seq2seq-attention) | Hub del [Lab 13](.).
