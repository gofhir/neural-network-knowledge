---
title: "LSTM (Long Short-Term Memory)"
weight: 120
math: true
---

{{< paper-card
    title="Long Short-Term Memory"
    authors="Hochreiter, Schmidhuber"
    year="1997"
    venue="Neural Computation 9(8):1735-1780"
    pdf="/papers/lstm-hochreiter-1997.pdf" >}}
Introduce la arquitectura LSTM con celdas de memoria de auto-conexion (constant error carrousel) y compuertas multiplicativas (input/output gates) que resuelven el problema de vanishing/exploding gradient en RNNs y permiten aprender dependencias en intervalos de mas de 1000 pasos.
{{< /paper-card >}}

---

## Contexto

A principios de los 90, las RNNs entrenadas con BPTT o RTRL **no podian aprender dependencias de largo plazo**. Hochreiter (1991, tesis) habia mostrado analiticamente que el gradiente decae o explota exponencialmente con la longitud de la secuencia. Bengio et al. (1994) confirmaron empiricamente la dificultad. Las soluciones previas (time-delay neural networks, NARX, Elman nets, sequence chunkers) abordaban el problema parcialmente o requerian arquitecturas ad-hoc.

---

## Ideas principales

### 1. Constant Error Carrousel (CEC)

El nucleo de LSTM es una unidad lineal con **auto-conexion de peso fijo 1.0**:

$$s_{c_j}(t) = s_{c_j}(t-1) + y^{in_j}(t) \cdot g(net_{c_j}(t))$$

Si la activacion es lineal y el peso recurrente es 1, el gradiente que fluye a traves de la celda **no decae ni explota**: queda atrapado en el "carrousel" propagandose perfectamente hacia atras. Es la razon por la que LSTM puede aprender dependencias de mas de 1000 pasos.

### 2. Multiplicative gates

Un CEC simple tiene dos problemas:

- **Conflicto de escritura**: la misma conexion debe a veces almacenar nueva info y a veces ignorarla.
- **Conflicto de lectura**: la salida de la celda perturba a otras unidades incluso cuando la informacion no es relevante.

LSTM introduce **compuertas multiplicativas** que aprenden cuando abrir y cerrar el acceso al CEC:

- **Input gate** $i_j$: controla cuanta informacion nueva entra a la celda.
- **Output gate** $o_j$: controla cuanta informacion de la celda se expone como salida.

(El paper original **no** incluia forget gate -- se anadio en Gers, Schmidhuber & Cummins 2000 "Learning to Forget".)

### 3. Memory cell blocks

Multiples celdas comparten un par de compuertas (input/output) formando un "block" -- mas eficiente que celdas individuales y facilita representaciones distribuidas.

### 4. Local en espacio y tiempo

A diferencia de RTRL ($O(n^4)$ por paso), LSTM es **local en espacio y tiempo**: complejidad $O(W)$ por paso (donde $W$ = numero de pesos), igual que BPTT pero sin la limitacion de memoria. No requiere almacenar la historia completa de activaciones.

---

## Resultados experimentales

LSTM resuelve tareas que ninguna RNN previa podia, incluyendo:

- **Embedded Reber grammar** con time lags > 50 pasos.
- **2-sequence problem** y **adding problem** con lags de 100 a 1000 pasos.
- **Noise + signal classification** con secuencias largas y ruido continuo.

Comparado con RTRL, BPTT, Recurrent Cascade-Correlation, Elman nets, y Neural Sequence Chunking, LSTM logro mas runs exitosos y aprende mucho mas rapido.

---

## Por que importa hoy

- **Estandar de facto** para procesamiento de secuencias en NLP, voz y traduccion durante 1997-2017.
- **Base** del modelo Seq2Seq de Sutskever 2014 (4 capas LSTM de 1000 celdas) que rompio el record en traduccion automatica neural.
- **Componente clave** de Show and Tell (Vinyals 2015) para image captioning.
- **Forget gate** (Gers et al. 2000) y **peephole connections** son extensiones que se siguen usando en variantes modernas.
- Aunque los **Transformers** (Vaswani 2017) los han desplazado en muchas tareas, LSTMs siguen siendo competitivos en streaming, edge devices y series temporales por su naturaleza secuencial y bajo costo de inferencia online.

---

## Notas y enlaces

- El paper es denso y matematicamente exigente -- la motivacion (secciones 1-3) y el algoritmo (seccion 4) son las partes mas importantes para implementadores.
- La derivacion del **error flow constante** (Apendice A.2) muestra por que el CEC funciona.
- Ver tambien: [Pascanu et al. 2013](difficulty-training-rnns-pascanu-2013) para el analisis formal moderno de vanishing/exploding gradient, y [GRU](gru-cho-2014) como la simplificacion mas adoptada.
- Recurso adicional: [Christopher Olah, "Understanding LSTM Networks"](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) -- la explicacion visual mas citada.

Ver fundamentos: [LSTM y GRU](/fundamentos/lstm-gru) · [Backpropagation Through Time](/fundamentos/backpropagation-through-time).
