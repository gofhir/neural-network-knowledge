---
title: "Invertir el tiempo"
weight: 5
---

Los cuatro experimentos anteriores explican **cómo** arreglar la predicción. Este pregunta otra cosa: **qué información está usando el modelo para decidir**. Es una línea de código y su resultado es negativo — y precisamente por eso es el más informativo del lab.

## El experimento

`abseiling` (rápel) es descender por una cuerda. `rock climbing` es ascender por la roca. Misma pared, misma silueta, mismo equipo, mismos colores. **La diferencia semántica se reduce casi por completo a la dirección del movimiento vertical.**

Si I3D usara esa dirección para discriminar, invertir el video temporalmente debería volcar la predicción hacia `rock climbing`: un rápel reproducido al revés *es*, visualmente, una escalada.

```python
evaluar(normalizar(clip[::-1], "m1_1"), "video INVERTIDO en el tiempo")
```

| Video | Top-1 | p(`abseiling`) | p(`rock climbing`) | $H$ |
|---|---|---|---|---|
| normal (250 frames) | `abseiling` | 65,19 % | 34,75 % | 0,651 |
| **invertido** | `abseiling` | **67,22 %** | 32,72 % | 0,638 |

**No cambia nada.** `abseiling` incluso sube 2 puntos.

## Qué significa

I3D **no distingue `abseiling` de `rock climbing` por la dirección del tiempo**. Lo hace por **apariencia y movimiento de corto alcance**: la cuerda tensa que baja desde el anclaje, el arnés, la postura del cuerpo colgado en vez de apoyado. Rasgos que un solo frame casi resuelve, y que el análisis de [dónde está la evidencia](../04-donde-esta-la-evidencia) ya había señalado como pequeños y frágiles.

Esto no es un defecto de la arquitectura. Las convoluciones 3D **pueden** representar dirección de movimiento: un kernel $t \times k \times k$ tiene toda la capacidad necesaria para responder asimétricamente al orden de los frames. El punto es que **no aprendió a hacerlo**, porque el dataset con el que se entrenó no lo exigía.

## Tres conexiones

### Con el lab 36: el mismo sesgo, una arquitectura más arriba

El [lab 36](/laboratorios/lab-36) midió que entrenar con **4 frames (85,9 %) igualaba o superaba a 8 frames (84,6 %)** en UCF11, en la mitad del tiempo. La conclusión fue que el *average pooling* de una CNN 2D descarta el orden temporal, así que darle más frames no aporta.

Aquí se mide algo más incómodo: **una CNN 3D, con convoluciones temporales de verdad, tampoco lo aprovecha**. La arquitectura cambió; el resultado, no.

| | Lab 36 | Lab 38 |
|---|---|---|
| Modelo | ResNet-34 2D + average pooling | I3D (Inception-v1 inflada, 3D) |
| Experimento | 8 frames contra 4 frames | video normal contra invertido |
| Resultado | 84,6 % → 85,9 % (sin costo) | 65,19 % → 67,22 % (sin cambio) |
| Lectura | el pooling **no puede** usar el orden | la 3D **puede** pero **no lo usa** |

La diferencia entre "no puede" y "no lo usa" es la que separa una limitación arquitectónica de una limitación del dato.

### Con Something-Something: el dataset que se creó por esto

[Something-Something](/papers/something-something-goyal-2017) (Goyal et al., 2017) existe precisamente porque [Kinetics-400](/papers/kinetics-kay-2017) tiene **sesgo de apariencia**: en una fracción grande de sus clases, el contexto de la escena basta para acertar. `playing guitar` se resuelve viendo una guitarra; `swimming` viendo agua.

Something-Something ataca eso con clases definidas por la **dinámica** y no por los objetos: *"empujar algo de izquierda a derecha"*, *"mover algo hacia la cámara"*, *"fingir tomar algo sin tomarlo"*. En ese dataset, **invertir el video cambia la etiqueta correcta** — el experimento H5 sería una prueba destructiva, no un no-evento.

El resultado de esta página es una réplica en miniatura de esa motivación, medida sobre un solo clip.

### Con el argumento del inflado

El [inflado](/fundamentos/inflado-de-convoluciones) funciona porque I3D hereda representaciones **de imagen** de ImageNet. H5 sugiere que buena parte de la ventaja práctica del modelo viene de ahí —de ver mejor— más que de las convoluciones temporales que el inflado agrega. Es consistente con el hallazgo original de [Karpathy (2014)](/papers/large-scale-video-karpathy-2014): un solo frame ya llegaba muy cerca del mejor modelo temporal de la época.

{{< callout type="warning" >}}
**Alcance de la afirmación.** Esto se midió en **un** video y **un** par de clases. Es una observación sólida sobre este caso, no una ley general sobre I3D. La versión fuerte de la hipótesis requiere repetirla sobre un conjunto de pares de clases temporalmente simétricos —abrir/cerrar, entrar/salir, subir/bajar— y medir cuántos sobreviven a la inversión.
{{< /callout >}}

## Síntesis del laboratorio

El lab entrega, en su forma original, tres celdas de inferencia y tres preguntas conceptuales. Lo que produjo al ejecutarlo:

**Sobre modelos pre-entrenados.** Un modelo sólo puede responder dentro de su vocabulario, y el softmax le impide avisar cuando la respuesta correcta no está. `ApplyEyeMakeup` recibió 98,13 % de confianza en una clase que no puede ser correcta. Ninguna métrica derivada del softmax lo detecta. Es el argumento medido de por qué *[Quo Vadis](/papers/i3d-carreira-2017)* reemplaza la capa final antes de reportar su 98,0 % en UCF101, y por qué "pre-entrenado" nunca significa "listo para usar" en un dominio nuevo.

**Sobre preprocesamiento.** Una línea —`/ 255.0` en lugar de `/ 127.5 - 1.0`— movió la predicción de 10,86 % a 62,31 %. El bug está en el tutorial oficial de TensorFlow Hub, sobrevivió años, y sobrevivió porque **degrada sólo en la frontera de decisión**: `archery` acierta con ambos preprocesos. La causa es que I3D hereda las BatchNorm de Inception-v1 junto con los kernels — o sea, es una consecuencia directa del inflado.

**Sobre configuración de inferencia.** Rango, escala, posición del crop y ventana temporal movieron el mismo modelo, sobre el mismo video, de **10,86 % a 92,85 %**. Ninguna de esas decisiones aparece en el paper ni en la *model card*; todas viven en el código de evaluación. Y el protocolo estándar de 3-crop, aplicado sin verificar que el sujeto esté centrado, **restó 17,4 puntos**.

**Sobre lo que I3D aprendió.** Discrimina por apariencia y movimiento de corto alcance. Invertir el tiempo no lo mueve. La arquitectura 3D **habilita** el razonamiento temporal; el dataset no siempre lo **exige**. Es la brecha que persigue todo el linaje posterior —[S3D](/papers/s3d-xie-2018), [R(2+1)D](/papers/r2plus1d-tran-2018), [SlowFast](/papers/slowfast-feichtenhofer-2019)— y la razón de ser de datasets como Something-Something.

---

**Ver también:** [Lab 36 - Análisis de Video](/laboratorios/lab-36) · [Clase 38 - Teoría](/clases/clase-38/teoria) · [Clase 38 - Profundización](/clases/clase-38/profundizacion) · [Inflado de convoluciones](/fundamentos/inflado-de-convoluciones) · [Reconocimiento de acciones](/fundamentos/reconocimiento-de-acciones).
