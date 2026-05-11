---
title: "Parte 3 - Teacher Forcing"
weight: 30
math: true
---

Las partes 1 y 2 del lab construyeron un seq2seq y le agregaron attention. Pero hubo un detalle del entrenamiento que se mantuvo implicito en ambas y que merece tratamiento aparte: **como se le entrega el input al decoder en cada paso del entrenamiento**. Esta tercera parte vuelve al modelo de Parte 1 (Seq2Seq **sin attention**), introduce **teacher forcing** como modificacion al loop del decoder, y entrena 300 epochs para compararlo contra Parte 1. Al final, dos actividades evaluadas (1.1 y 1.2) piden razonar sobre el cambio observado.

El hallazgo experimental no es trivial: con teacher forcing el modelo **converge mas rapido al inicio** (arranca en ~0.30 vs ~0.15) pero **satura mas bajo** (~0.88-0.89 vs ~0.91 de Parte 1). El primero es el efecto esperado; el segundo es la manifestacion empirica del **exposure bias**.

## Que es teacher forcing

El decoder de un seq2seq es **autoregresivo**: en cada paso $t$ produce un token $\hat{y}_t$ condicionado en los tokens anteriores. La pregunta clave es **cuales tokens anteriores se usan como input al paso $t$**, y la respuesta cambia segun la fase:

- Durante **training**: en cada paso del decoder, en lugar de alimentar la prediccion del paso anterior $\hat{y}_{t-1}$, se alimenta el **ground truth** $y_{t-1}^{*}$ (el token correcto del target) *(notebook 3, cell 14)*. Esto es teacher forcing.
- Durante **inference / evaluation**: no hay ground truth disponible — el modelo esta generando una secuencia nueva — asi que el decoder se alimenta de su propia prediccion del paso anterior $\hat{y}_{t-1}$. Esto es el modo **autoregresivo** o *free-running*.

Formalmente, en teacher forcing el decoder modela cada paso como:

$$
P(y_t \mid y_{<t}^{*}, x; \theta)
$$

donde $y_{<t}^{*}$ son los tokens **reales** del target hasta $t-1$, no las predicciones del propio modelo. La loss sigue siendo cross-entropy por paso, sumada a lo largo de la secuencia.

## Por que se usa

Tres razones, todas relacionadas con la dinamica del entrenamiento:

**1. Convergencia mas rapida.** En las primeras epochs el modelo no sabe nada y sus predicciones son basicamente ruido. Si lo alimentaramos con ese ruido en cada paso, el decoder estaria aprendiendo a continuar secuencias *aleatorias*, no las del dominio real. Con teacher forcing el decoder siempre ve la distribucion correcta de tokens previos y puede enfocarse en aprender el mapping condicional $P(y_t \mid y_{<t}^*, x)$ desde el primer batch.

**2. Evita compounding de errores.** Sin teacher forcing, un error temprano en la secuencia (por ejemplo, predecir mal $\hat{y}_2$) contamina el input de **todos** los pasos siguientes. El decoder nunca llega a un punto donde pueda recuperarse, y el gradiente se vuelve casi inutil porque cada paso esta "calculando" una loss sobre un contexto que ya no tiene relacion con el target real.

**3. Permite paralelizacion (en arquitecturas modernas).** Como en training todos los inputs del decoder son los $y_{t-1}^{*}$ del target (que ya conocemos), todos los pasos pueden computarse en una sola pasada con un *shift-right mask*. No es algo que aproveche este notebook (sigue siendo un loop con `LSTMCell`), pero es la observacion clave que hace viable el entrenamiento del Transformer en paralelo.

## Exposure bias (el problema)

Teacher forcing tiene un costo conceptual conocido como **exposure bias**: la distribucion de inputs que ve el decoder durante training es **distinta** de la que ve durante inference.

- En training, el decoder *siempre* recibe ground truth como input al paso siguiente.
- En inference, el decoder *nunca* recibe ground truth — recibe sus propias predicciones, que pueden tener errores.

El resultado es que el modelo se entreno asumiendo un input "limpio" pero en deployment se le entrega un input "ruidoso". Si en algun momento de la generacion comete un error y produce un $\hat{y}_t$ raro, el modelo nunca vio nada parecido durante training y la calidad puede degradarse rapidamente. **Este efecto se ve empiricamente en la corrida del notebook**: el modelo con teacher forcing satura en eval-acc ~0.88, ligeramente por debajo del Parte 1 sin teacher forcing (~0.91), porque el eval-acc se calcula en modo autoregresivo y el modelo no esta acostumbrado a recuperarse de sus propios errores.

Hay mitigaciones propuestas en la literatura — vale la pena conocerlas aunque **el notebook no las implementa**:

- **Scheduled sampling** ([Bengio et al. 2015](https://arxiv.org/abs/1506.03099)): mezclar gradualmente teacher forcing y free-running durante el entrenamiento. Con probabilidad $\epsilon$ se usa ground truth y con probabilidad $1 - \epsilon$ la prediccion del modelo. $\epsilon$ decae de 1 a 0 a lo largo de las epochs, de modo que el modelo se va "acostumbrando" a sus errores.
- **Professor forcing** ([Lamb et al. 2016](https://arxiv.org/abs/1610.09038)): adversarial training para forzar que las distribuciones de hidden states del decoder en modo teacher-forcing y en modo free-running sean indistinguibles para un discriminador.

En la practica modernamente se sigue entrenando con teacher forcing puro (es lo que hace el Transformer original) y se acepta el exposure bias como un trade-off. Las arquitecturas grandes preentrenadas son lo bastante robustas como para que el problema no domine en secuencias cortas-medias.

## Implementacion en el notebook

A diferencia de lo que se podria pensar, **Parte 3 no parte del modelo de Parte 2 (con attention)** — vuelve al modelo de **Parte 1 (sin attention)** y le agrega teacher forcing como modificacion del loop del decoder. La comparacion en Actividad 1.1 es contra Parte 1, no contra Parte 2.

Tampoco se implementa un `teacher_forcing_ratio` parametrizable. El notebook usa **teacher forcing puro** durante toda la fase de training (cada paso del decoder en `model.train()` recibe ground truth), y eval queda **siempre autoregresivo**. No hay scheduled sampling ni probabilidad — es una sola flag que cambia con `self.training`.

### Las tres modificaciones respecto a Parte 1

**1. `DecoderModule.forward` con flag de teacher forcing** *(notebook 3, cell 14)*:

```python
def forward(self, encoder_hidden_state, max_output_length, correct_answer=None):
    ...
    y_t = self.embeddings_table(self.start_idx.repeat(batch_size))
    for i in range(max_output_length):
        state = self.lstm_cell(y_t, state)
        P_t = self.h2o(state[0])
        out.append(P_t)

        if self.training:
            # TEACHER FORCING durante training
            y_t = self.embeddings_table(correct_answer[:, i])
        else:
            # autoregressive durante eval
            _, max_indices = P_t.max(dim=1)
            y_t = self.embeddings_table(max_indices)

    return torch.stack(out, dim=1)
```

El atributo `self.training` es built-in de `nn.Module`. Cambia automaticamente al llamar `model.train()` o `model.eval()`, sin setearlo manualmente.

**2. `train_one_epoch` pasa el ground truth como tercer argumento** *(notebook 3, cell 23)*:

```python
y_pred = model(batch.source, max_ouput_len, y_gt)
#                                            ^^^^
#                                  ground truth → correct_answer
```

**3. `SeqToSeq.forward` propaga el argumento al decoder** *(notebook 3, cell 27)*:

```python
def forward(self, src_sentences, max_output_length, correct_answer=None):
    ...
    outputs = self.decoder(encoder_hidden_states, max_output_length, correct_answer)
```

### Asimetria train/eval

`eval_one_epoch` *(notebook 3, cell 24)* **no pasa `y_gt`** al modelo:

```python
y_pred = model(batch.source, max_ouput_len)
#                       sin correct_answer → cae al else (autoregresivo)
```

Esto es deliberado: en deployment no hay ground truth, asi que la metrica debe reflejar el comportamiento real del modelo. La asimetria — training con TF, eval autoregresivo — es justamente lo que produce el exposure bias.

## Resultados comparativos

Mismo hyperparams que Partes 1 y 2: `embedding_size=100`, `hidden_size=150`, `batch_size=128`, `lr=0.001` (Adam), 300 epochs. Parametros entrenables: **789,809**, exactamente iguales que Parte 1 — la unica diferencia entre los dos modelos es la logica del input al decoder, sin parametros nuevos.

### Curva de eval accuracy

![Eval accuracy Seq2Seq con teacher forcing sobre SCAN](/laboratorios/lab-13/eval-acc-teacher-forcing.png)

| Epoch | Parte 1 (sin TF) | Parte 3 (con TF) | Diferencia |
| --- | --- | --- | --- |
| 0 (inicio) | ~0.15 | **~0.30** | **+0.15** ← arranca mucho mas alto |
| 10 | ~0.50 | ~0.55 | similar |
| 30 | ~0.75 | **~0.80** | TF ligeramente arriba |
| 50 | ~0.85 | ~0.85 | iguales |
| 100 | ~0.88 | ~0.87 | empatados |
| 200 | ~0.90 | ~0.88 | Parte 1 empieza a ganar |
| **300 (final)** | **~0.91** | **~0.88-0.89** | **Parte 1 satura mas alto** |

### Lectura empirica

**1. Teacher forcing acelera la convergencia inicial.** La curva de Parte 3 arranca en ~0.30, casi el doble de Parte 1. Y en las primeras 30 epochs claramente sube mas rapido. Esto confirma la prediccion teorica: gradientes mas estables → aprendizaje mas eficiente.

**2. Pero el plateau final es mas bajo.** Parte 3 satura en ~0.88, frente al ~0.91 de Parte 1. Esto es **exposure bias visible empiricamente**: el modelo se entreno con inputs perfectos (ground truth) pero se evalua con inputs ruidosos (sus propias predicciones), y la asimetria afecta la metrica reportada.

**3. Aparece un dip pronunciado cerca de epoch 230.** La curva cae bruscamente a ~0.82 antes de recuperarse. Esto puede deberse a un step agresivo de Adam tras una larga fase plana, pero no es sintomatico de un problema sostenido.

### Lo que muestra esta comparacion

No es solo "teacher forcing es bueno". Es algo mas matizado: **teacher forcing es una intervencion en la dinamica de optimizacion, no un cambio en la capacidad expresiva del modelo**. Cambia *como* se aprende, no *que* se puede aprender. La parametrizacion (789K) es identica.

Cuando se inserta entre el training y la evaluacion una asimetria (TF en uno, autoregresivo en otro), la metrica de evaluacion paga ese precio. Si modificaramos el eval para tambien usar teacher forcing — cosa que **no se hace en la practica porque no refleja deployment** — la accuracy probablemente seria mas alta que la de Parte 1.

Este es un ejemplo clasico de **goodhart's law en miniatura**: la metrica reportada subestima la calidad del modelo de Parte 3 porque la asimetria train/eval no esta capturada en el numero. Una metrica mas honesta seria sentence-level con autoregresivo en train y eval, pero eso requiere otra arquitectura de medicion.

## Actividades evaluadas

El notebook cierra con dos actividades cortas — los enunciados completos estan en [ejercicios](ejercicios) y las respuestas razonadas en [resolucion](resolucion):

- **Actividad 1.1** — Comparar la velocidad de convergencia respecto al Seq2Seq sin attention (Parte 1).
- **Actividad 1.2** — Explicar el mecanismo subyacente (HINT: como se entrega el input al decoder).

Ambas se pueden responder con los datos de la curva que produce esta misma parte 3.

Volver a [Parte 2 — Seq2Seq con Attention](seq2seq-attention) | Hub del [Lab 13](.).
