---
title: "Resolución"
weight: 50
math: true
---

> Resolucion razonada de las Actividades 1.1 y 1.2 del notebook 3, mas insights consolidados de las 3 partes. Los enunciados literales estan en [ejercicios](ejercicios).

---

## Actividad 1.1

**Enunciado:** Comparando con modelo Seq2Seq **SIN** atencion, ¿esta metodologia (teacher forcing) hace que el entrenamiento converja mas rapido o mas lento?

**Respuesta:**

Con teacher forcing el entrenamiento converge **mas rapido** durante las primeras epochs. El modelo arranca desde una eval_acc mucho mayor (~0.30 vs ~0.15 sin teacher forcing) y alcanza valores como ~0.80 alrededor de epoch 30 en lugar de epoch 50.

Hallazgo adicional no pedido pero observable: el plateau final que alcanza este modelo es ligeramente **mas bajo** (~0.88-0.89 vs ~0.91 de Parte 1). Es decir, converge mas rapido pero a un techo levemente menor — manifestacion empirica del **exposure bias** (desajuste train/eval que el modelo paga al evaluar en modo autoregresivo despues de entrenar siempre con ground truth).

---

## Actividad 1.2

**Enunciado:** Explique en palabras simples a que se debe el cambio visto en la velocidad de convergencia. *(Hint: el modo en que se entrega el input al decoder.)*

**Respuesta:**

El cambio se debe a **que recibe el decoder como input en cada paso del entrenamiento**.

Sin teacher forcing, el decoder usa **su propia prediccion** del paso anterior. Como al inicio del entrenamiento esas predicciones estan mal, los errores se acumulan a lo largo de la secuencia: el paso 2 recibe un input erroneo del paso 1, el paso 3 lo recibe peor todavia, y asi. El gradiente se calcula sobre una secuencia contaminada y el aprendizaje avanza lento.

Con teacher forcing, el decoder recibe en cada paso **el token correcto** (ground truth), sin importar lo que predijo. Los errores ya no se propagan, cada paso entrena con inputs limpios, y la convergencia inicial es mucho mas rapida.

---

## Insights consolidados de las 3 partes

### Parte 1 — Seq2Seq basico (sin attention, sin teacher forcing)

- Modelo: 789,809 parametros entrenables. Encoder bi-LSTM + Decoder `LSTMCell` autoregresivo con greedy decoding.
- Entrenamiento: 300 epochs, eval acc satura en **~0.91 token-level con padding**.
- **Limitacion identificada empiricamente:** la curva se aplana cerca de 0.91 y no sube significativamente mas. Aumentar epochs no resuelve el problema — el modelo esta limitado por la capacidad de representacion del context vector fijo $h_T$. Toda la informacion del input tiene que viajar por ese vector unico de 300 floats, lo que aprieta cuando las secuencias son largas.
- **Imperfeccion del notebook**: no usa `ignore_index=0` en cross-entropy, asi que el padding contamina la metrica reportada (la accuracy real sentence-level seria mucho menor — aproximacion grosera: $0.91^{20} \approx 15\%$).

### Parte 2 — Seq2Seq con attention (Bahdanau additive)

- Mismo modelo base, pero con un `AttentionModule` (3 matrices entrenables: $\mathbf{W}, \mathbf{U}, \mathbf{V}$) entre el decoder y los hidden states del encoder. El decoder ahora ve **todos** los $h_1, \ldots, h_T$, no solo $h_T$.
- Cambio minimo en codigo: el encoder solo modifica su `return` para entregar tambien la secuencia completa. El decoder agrega 3 lineas: calcular `context_vector`, concatenarlo con `hidden_state`, proyectar el concat con `h2o`.
- Resultado: eval acc satura en **~0.93** vs ~0.91 de Parte 1. La diferencia parece chica pero esta enmascarada por la metrica token-level con padding — el sentence-level real es mucho mas favorable a Parte 2.
- **Visualizacion**: el caso `run thrice after look` muestra alineamientos interpretables. El modelo manejo correctamente la semantica de `after` (genero `i_look` antes que `i_run`), y la atencion muestra una transicion clara entre la fase de `i_look` y la fase de `i_run`. Observacion notable: `thrice` casi no recibe atencion, pero el modelo cuenta correctamente — el conteo vive en la memoria del LSTM, no en la atencion. Atencion y memoria son complementarias.

### Parte 3 — Teacher forcing (sin attention, con TF)

- Mismo modelo de Parte 1 (sin attention). La unica diferencia es que durante training el decoder recibe el ground truth como input al paso siguiente, en lugar de su propia prediccion.
- Resultado: convergencia **inicial mas rapida** (arranca en ~0.30 vs ~0.15) pero plateau final **mas bajo** (~0.88-0.89 vs ~0.91).
- **El hallazgo no trivial**: teacher forcing no es "mejor incondicional". Es una intervencion sobre la dinamica de optimizacion que acelera el aprendizaje pero paga el precio del **exposure bias** en evaluacion. El modelo nunca se acostumbro a sus propios errores durante training, asi que cuando evalua en modo autoregresivo (la metrica reportada) sufre frente a Parte 1 que si se entreno asi.
- Los **789,809 parametros** son identicos a Parte 1 — confirma que TF no cambia capacidad expresiva, solo dinamica de aprendizaje.

### Lo que hila las 3 partes

Las tres partes ilustran **tres dimensiones distintas que afectan el desempeno de un Seq2Seq**:

1. **Capacidad de representacion** (Parte 1 → Parte 2 con attention): cambiar la arquitectura para que la informacion del input no tenga que comprimirse en un solo vector.
2. **Dinamica de optimizacion** (Parte 1 → Parte 3 con TF): cambiar como fluyen los gradientes durante training, sin tocar los parametros.
3. **Trade-offs metrica/realidad** (Parte 3 con TF): la metrica reportada puede subestimar o sobreestimar la calidad real del modelo dependiendo de la asimetria entre training y deployment.

La leccion de fondo es que **el desempeno de un modelo es la interaccion de su arquitectura, su loop de optimizacion, y la metrica con que se evalua**. Tres ejes ortogonales que en una clase introductoria de Seq2Seq se exploran uno por uno para que la intuicion quede separable.

### Conexion con la teoria de la clase

- El cuello de botella del context vector unico que motiva attention esta tratado en la [teoria de la clase 13](/clases/clase-13/teoria/) (secciones de motivacion de attention).
- La formulacion additive de Bahdanau y la formulacion dot-product de Luong / Transformer estan comparadas en la [profundizacion](/clases/clase-13/profundizacion/).
- Para teacher forcing y exposure bias en general, el [fundamento Seq2Seq](/fundamentos/seq2seq/) tiene mas contexto historico y conexion con scheduled sampling, beam search, y modelos sin teacher forcing.
