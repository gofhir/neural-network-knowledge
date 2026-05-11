# Lab 13 — Tracking colaborativo (Colab + local)

**Fecha inicio:** 2026-04-27
**Notebooks origen:**
- `clase_13/material/Laboratorio/Practico_clase_13_parte_1.ipynb`
- `clase_13/material/Laboratorio/Practico_clase_13_parte_2.ipynb`
- `clase_13/material/Laboratorio/Practico_clase_13_parte_3.ipynb`

**Notebooks resolucion (a generar en Colab):**
- `clase_13/material/Laboratorio/Practico_13_parte_1_RAE.ipynb`
- `clase_13/material/Laboratorio/Practico_13_parte_2_RAE.ipynb`
- `clase_13/material/Laboratorio/Practico_13_parte_3_RAE.ipynb`

## Modalidad

- Roberto ejecuta en Colab y entrega outputs/screenshots
- Claude integra outputs en los RAE notebooks + site content `site/content/laboratorios/lab-13/`

## Estructura del lab

- **Parte 1 — Seq2Seq basico** (encoder + decoder sin attention) — translation
- **Parte 2 — Seq2Seq + Attention** (Bahdanau additive + visualizacion del attention map)
- **Parte 3 — Teacher Forcing** + Actividades 1.1 y 1.2

## Checklist de avance

### Parte 1

- [x] Setup + dataset cargado (SCAN `tasks_simple`, GPU cuda activa)
- [x] Encoder + Decoder construidos y verificados (789,809 parametros)
- [x] Entrenamiento 300 epochs — outputs: curva `eval_acc` que satura cerca de 0.91
- [x] Evaluacion: accuracy token-level con padding incluido (token-match, no BLEU)
- [ ] Captura del plot `eval-acc-seq2seq-base.png` subida a `site/static/laboratorios/lab-13/`
- [ ] RAE notebook `Practico_13_parte_1_RAE.ipynb` guardado en `clase_13/material/Laboratorio/`

### Parte 2

- [x] Attention module implementado y verificado (Bahdanau additive, 3 matrices entrenables)
- [x] Entrenamiento con attention — eval acc plateau ~0.93 (vs ~0.91 de Parte 1)
- [x] Visualizacion de attention heatmap — caso `run thrice after look`
- [x] Comparacion contra parte 1 (ver seccion Resultados)
- [ ] Captura del plot `eval-acc-seq2seq-attention.png` subida a `site/static/laboratorios/lab-13/`
- [ ] Captura del heatmap `attention-heatmap-run-thrice-after-look.png` subida a `site/static/laboratorios/lab-13/`
- [ ] RAE notebook `Practico_13_parte_2_RAE.ipynb` guardado en `clase_13/material/Laboratorio/`

### Parte 3

- [x] Teacher forcing implementado (modelo de Parte 1 + flag `self.training`, sin ratio probabilistico)
- [x] Comparacion entrenamiento con/sin teacher forcing — Parte 1 satura ~0.91, Parte 3 satura ~0.88 (exposure bias visible)
- [x] **Actividad 1.1** resuelta — converge mas rapido (arranca 0.30 vs 0.15) pero satura mas bajo
- [x] **Actividad 1.2** resuelta — input al decoder: ground truth vs prediccion propia
- [ ] Captura del plot `eval-acc-teacher-forcing.png` subida a `site/static/laboratorios/lab-13/`
- [x] RAE notebook `Practico_clase_13_parte_3_rae.ipynb` guardado en `clase_13/material/Laboratorio/`

### Site content

- [x] `site/content/laboratorios/lab-13/_index.md` (Fase 1 — scaffolding)
- [x] `site/content/laboratorios/lab-13/seq2seq-basico.md` (Fase 1 — conceptual + placeholders)
- [x] `site/content/laboratorios/lab-13/seq2seq-attention.md` (Fase 1 — conceptual + placeholders)
- [x] `site/content/laboratorios/lab-13/teacher-forcing.md` (Fase 1 — conceptual + placeholders)
- [x] `site/content/laboratorios/lab-13/ejercicios.md` (Fase 1 — enunciados)
- [x] `site/content/laboratorios/lab-13/resolucion.md` (Fase 1 — esqueleto)
- [x] +card en `site/content/laboratorios/_index.md` (Fase 1)
- [ ] Reemplazar placeholders `[outputs pendientes]` con resultados reales (Fase 2, por parte)

### Static assets

- [ ] Curvas loss/acc parte 1 — `loss-base.png`, etc.
- [ ] Curvas loss/acc parte 2 — `loss-attn.png`, etc.
- [ ] Attention heatmap(s) — `attn-heatmap-*.png`
- [ ] Curvas comparativas teacher forcing — `loss-tf.png`, etc.
- [ ] Renders HTML + .ipynb descargables (3 c/u)

## Decisiones (confirmadas durante ejecucion)

- **Dataset**: SCAN `tasks_simple` (no es traduccion En→Fr clasica; es comando-ingles → secuencia-de-acciones-simbolicas).
- **Metrica de evaluacion (parte 1)**: accuracy token-level con padding incluido. No es BLEU. Reportada como `(y_pred.argmax(dim=2) == y_gt).float().mean()`.
- **n_epochs parte 1**: 300 (suficiente para saturar a 0.91; mas no se mueve significativamente).
- **Tamano del dataset**: `tasks_train_simple.txt` y `tasks_test_simple.txt` enteros, sin subset.

## Hallazgos / Insights

### Parte 1 — Seq2Seq basico (sesion 2026-05-10)

**Data inspection:**

- Los tokens destino reales en `tasks_*_simple.txt` son `I_JUMP`, `I_TURN_LEFT`, `I_TURN_RIGHT`, `I_WALK`, `I_RUN`, `I_LOOK` — la tabla del markdown del notebook (`JUMP`, `LTURN`, `RTURN`, ...) esta desactualizada o se refiere a otra variante de SCAN. Para el modelo da igual el string concreto, pero conviene saberlo cuando se inspeccionen outputs.
- Formato TSV limpio, sin prefijos `IN:`/`OUT:` (algunas variantes de SCAN si los traen).
- Vocabularios chicos: ~14 tokens fuente, ~7 destino incluyendo `<pad>`.

**Decisiones de implementacion del notebook que vale la pena documentar:**

1. **Hack del `<SOS>`**: en lugar de `Field(init_token='<sos>')`, pasa `start_idx=len(TARGET.vocab)` y dimensiona `dst_vocab_size+1`. El `<SOS>` queda como fila extra de la tabla de embedding del decoder sin string asociado en el `Vocab`. Funciona, pero es no estandar.
2. **No usa `ignore_index=0` en `cross_entropy`**: el padding entra a la loss y la accuracy, inflando ambas metricas (sobre todo accuracy).
3. **No usa teacher forcing en parte 1**: el loop del decoder alimenta siempre `y_t = embedding(argmax(P_t))`, nunca el ground truth. Es parte de por que necesita 300 epochs para saturar. La parte 3 retoma el tema.
4. **No hay token `<EOS>`**: la generacion se controla unicamente por `max_output_length`, fijado al largo del target del batch durante training.

**El bottleneck observable empiricamente:**

- El modelo satura en ~0.91 token-level con padding incluido. No es por falta de capacidad de Adam — es la cota superior de la arquitectura.
- Sentence-level (estimacion aproximada `0.91^20 ≈ 0.15` para secuencias de 20 tokens) seria mucho mas bajo. SCAN en la literatura suele reportarse sentence-level y los modelos con attention superan el 99%, lo que confirma que el techo del Seq2Seq basico esta lejos del optimo.

### Parte 2 — Seq2Seq con attention (sesion 2026-05-10)

**Implementacion:**

- AttentionModule de 6 lineas operativas: 3 matrices entrenables ($\mathbf{W}, \mathbf{U}, \mathbf{V}$), broadcasting via `unsqueeze(1)`, softmax con `dim=1` (sobre posiciones del encoder, no sobre batch).
- Decoder cambia minimamente: solo 3 lineas nuevas (calculo del context, concatenacion con hidden, proyeccion). El encoder solo cambia el `return`.
- Dos factores de 2 distintos en el modelo: uno por bidireccionalidad del encoder (encoder→decoder), otro por la concatenacion s ⊕ c (dentro de `h2o`). Conviene no confundirlos.

**Diferencia respecto a Parte 1:**

- Eval acc satura en ~0.93 (vs ~0.91 de Parte 1). La diferencia parece chica pero esta subestimada por la metrica token-level con padding. Sentence-level real es mucho mas favorable: `0.91^20 ≈ 0.15` vs `0.93^20 ≈ 0.23`.
- Plateau mas limpio que Parte 1 (menos oscilaciones tras epoch 150).

**Heatmap observado (`run thrice after look`):**

- El modelo manejo correctamente la semantica de `after` — genero `i_look` antes que `i_run` aunque `look` aparece despues en el input.
- Transicion visible entre fase `i_look` (mira a `look`) y fase `i_run` (mira a `run`): la segunda fila del heatmap muestra atencion dividida entre ambas, capturando el momento exacto del cambio de foco.
- `thrice` casi no recibe atencion pero el modelo cuenta correctamente y genera 3 tokens `i_run`. Observacion clave: el conteo vive en la memoria del LSTM, no en la atencion. Atencion da contenido, memoria da posicion.

### Parte 3 — Teacher forcing (sesion 2026-05-10)

**Implementacion:**

- **Parte 3 parte del modelo de Parte 1 (sin attention)**, no del de Parte 2. La comparacion en Actividad 1.1 es contra Parte 1, no Parte 2.
- No hay `teacher_forcing_ratio` parametrizable ni scheduled sampling. Es teacher forcing **puro** cuando `self.training=True`, autoregresivo cuando `self.training=False`. Una sola flag, sin probabilidad.
- 3 modificaciones quirurgicas: `DecoderModule.forward` agrega rama `if self.training`, `train_one_epoch` pasa `y_gt` como tercer argumento, `SeqToSeq.forward` propaga `correct_answer`.
- 789,809 parametros, identico a Parte 1 (TF no agrega capas, solo cambia logica del input).

**Hallazgo empirico no trivial — exposure bias visible:**

- TF acelera la convergencia inicial (arranca 0.30 vs 0.15, alcanza 0.80 en epoch 30 vs epoch 50).
- PERO el plateau final es mas bajo (~0.88-0.89 vs ~0.91 de Parte 1).
- Causa: asimetria train/eval. Entrenando con ground truth siempre, el modelo no se acostumbra a recuperarse de sus propios errores; pero eval es siempre autoregresivo. La metrica reportada paga el precio del exposure bias.
- Esta observacion no se pedia explicitamente en las actividades 1.1/1.2 pero vale la pena documentarla — muestra que TF no es "mejor incondicional", es un trade-off entre velocidad de aprendizaje y robustez al deployment.

**Respuestas pegadas en el notebook RAE:**

- **1.1**: Converge mas rapido. Arranca eval_acc ~0.30 vs ~0.15 y alcanza ~0.80 alrededor de epoch 30 en lugar de epoch 50.
- **1.2**: Sin TF el decoder recibe sus propias predicciones erroneas en cada paso, los errores se acumulan y contaminan el gradiente. Con TF recibe el ground truth, sin propagacion de errores, gradiente limpio, aprendizaje rapido.

## Resultados

### Parte 1 — Seq2Seq basico (Resultados)

| Item | Valor |
| --- | --- |
| Hyperparams | `embedding_size=100`, `hidden_size=150`, `batch_size=128`, `lr=0.001` (Adam), `n_epochs=300` |
| Parametros entrenables | **789,809** |
| Device | GPU `cuda` (Colab T4) |
| Metrica | accuracy token-level con padding incluido |
| Eval accuracy inicial (epoch 0) | ~0.15 |
| Eval accuracy epoch 50 | ~0.85 |
| Eval accuracy epoch 100 | ~0.88 |
| Eval accuracy epoch 200 | ~0.90 |
| **Eval accuracy final (epoch 300)** | **~0.91** |
| Forma de la curva | Crecimiento rapido (epochs 0-50), refinamiento gradual hasta plateau cerca de 0.91 con dos dips menores en epochs ~120 y ~180 |

Imagen: `site/static/laboratorios/lab-13/eval-acc-seq2seq-base.png` (por subir desde el screenshot del Colab).

### Parte 2 — Seq2Seq con attention (Resultados)

| Item | Valor |
| --- | --- |
| Hyperparams | iguales a Parte 1 |
| Modulos nuevos | `AttentionModule` (3 matrices: $\mathbf{W}, \mathbf{U}, \mathbf{V}$ — `Linear(150,150)`, `Linear(150,150)`, `Linear(150,1)`) |
| Eval accuracy inicial (epoch 0) | ~0.05 |
| Eval accuracy epoch 50 | ~0.85 |
| Eval accuracy epoch 100 | ~0.91 |
| Eval accuracy epoch 200 | ~0.92 |
| **Eval accuracy final (epoch 300)** | **~0.93** |
| Forma de la curva | Crecimiento rapido (epochs 0-50), plateau mas limpio que Parte 1 con dips menores en epochs ~80 y ~130 |
| Estimacion sentence-level (gross) | `0.93^20 ≈ 0.23` (vs `0.91^20 ≈ 0.15` de Parte 1) |

Imagenes (por subir):

- `site/static/laboratorios/lab-13/eval-acc-seq2seq-attention.png` — curva de entrenamiento.
- `site/static/laboratorios/lab-13/attention-heatmap-run-thrice-after-look.png` — heatmap del caso `run thrice after look`.

### Parte 3 — Teacher forcing (Resultados)

| Item | Valor |
| --- | --- |
| Hyperparams | iguales a Parte 1 |
| Parametros entrenables | **789,809** (identico a Parte 1) |
| Eval accuracy inicial (epoch 0) | ~0.30 (vs ~0.15 de Parte 1) |
| Eval accuracy epoch 30 | ~0.80 (vs ~0.75 de Parte 1) |
| Eval accuracy epoch 50 | ~0.85 |
| Eval accuracy epoch 100 | ~0.87 |
| Eval accuracy epoch 200 | ~0.88 |
| **Eval accuracy final (epoch 300)** | **~0.88-0.89** (vs ~0.91 de Parte 1) |
| Dip pronunciado | epoch ~230 (caida a ~0.82 con recuperacion posterior) |
| Forma | Acelera al inicio, satura mas bajo que Parte 1 — manifestacion empirica de exposure bias |

Imagen: `site/static/laboratorios/lab-13/eval-acc-teacher-forcing.png` (por subir desde el screenshot del Colab).

### Comparativa consolidada de las 3 partes

| Modelo | Eval acc final (token-level con padding) | Comentario |
| --- | --- | --- |
| Parte 1 (sin attn, sin TF) | **~0.91** | Modelo base, autoregresivo siempre |
| Parte 2 (con attn, sin TF) | **~0.93** | Capacidad arquitectonica mejorada — attention rompe el bottleneck del context fijo |
| Parte 3 (sin attn, con TF) | **~0.88-0.89** | Dinamica de optimizacion mejorada pero exposure bias castiga el eval |
