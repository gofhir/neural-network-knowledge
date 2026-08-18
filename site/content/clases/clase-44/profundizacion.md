---
title: "Profundización - El prior que completa lo que no está"
weight: 20
math: true
---

> Las siete aplicaciones de la [teoría](teoria) parecen siete problemas distintos. Comparten una estructura: en todas, **la información necesaria no está en la entrada**, y lo que la completa es un prior aprendido. Reconstruir voz desde labios, cara desde voz, alta resolución desde baja, o los cuadros de un video que nunca se filmó — en cada caso el modelo aporta lo que falta, y la pregunta interesante es de dónde sale y cuánto se le puede creer.
>
> Cuatro partes: la aproximación de primer orden de FOMM y cuánto vale su jacobiano; el *informed guess* hecho preciso; la asimetría entre generar y detectar; y la factorización que unifica toda la clase.
>
> Las cifras marcadas como **medidas** provienen de código ejecutado; el mismo de la [práctica](practica).

---

## Parte I — Qué compra el jacobiano

### I.1. El planteo

[FOMM](/papers/fomm-siarohin-2019) representa el movimiento con $K$ puntos clave. La pregunta de diseño es qué guardar en cada uno:

- **orden 0** — solo la posición transformada. Cada punto aporta un desplazamiento; 2 parámetros.
- **orden 1** — posición **y jacobiano**, la matriz $2\times 2$ de derivadas de la transformación en ese punto; 6 parámetros.

La expansión de Taylor que da nombre al paper:

$$\mathcal{T}_{X\leftarrow R}(p) = \mathcal{T}_{X\leftarrow R}(p_k) + \left.\frac{d}{dp}\mathcal{T}_{X\leftarrow R}(p)\right|_{p=p_k}(p-p_k) + o(\lVert p - p_k\rVert)$$

El primer término es el orden 0; el segundo es lo que agrega el orden 1.

### I.2. Cuánto mejora, a igual número de puntos

Aproximando campos de movimiento suaves **arbitrarios** —no generados por la familia que la representación puede expresar— y ajustando los parámetros por mínimos cuadrados en ambos casos (**medido**):

| $K$ | error, orden 0 | error, orden 1 | mejora |
|---|---|---|---|
| 4 | 0,02879 | 0,01689 | 1,71× |
| 6 | 0,02436 | 0,01192 | 2,04× |
| 8 | 0,02102 | 0,00899 | 2,34× |
| **10** | 0,01873 | 0,00635 | **2,95×** |
| 16 | 0,01241 | 0,00247 | 5,02× |
| 24 | 0,00786 | 0,00074 | 10,64× |

A los $K = 10$ que usa FOMM, el jacobiano reduce el error casi **tres veces**.

### I.3. Dónde importa: rotación

El jacobiano no agrega expresividad genérica: agrega exactamente la capacidad de representar **rotación, escala y cizalla** locales. Sobre una rotación pura, con 8 puntos (**medido**):

| rotación | error, orden 0 | error, orden 1 |
|---|---|---|
| 2° | 0,000493 | 2,6 × 10⁻¹⁶ |
| 5° | 0,000918 | 5,7 × 10⁻¹⁶ |
| 10° | 0,002181 | 1,3 × 10⁻¹⁵ |
| 20° | 0,004524 | 2,4 × 10⁻¹⁵ |
| 40° | 0,008623 | 4,4 × 10⁻¹⁵ |

Con jacobiano el error es **cero numérico**: una rotación es afín y la representación de primer orden la expresa exactamente. Sin él, el error crece linealmente con el ángulo.

Esto explica el dominio de aplicación del método —cabezas que giran, cuerpos que se articulan— y también su limitación declarada: la expansión de Taylor vale **en un entorno** de cada punto, así que si la pose del video conductor se aleja mucho de la de la imagen fuente, la aproximación deja de valer.

### I.4. El matiz: a igual presupuesto de parámetros, gana el orden 0

Aquí el experimento produce un resultado que complica la lectura fácil. Comparando **a igual número de parámetros** (**medido**):

| parámetros | $K$ orden 0 | error | $K$ orden 1 | error | gana |
|---|---|---|---|---|---|
| 48 | 24 | **0,00789** | 8 | 0,00914 | orden 0 |
| 72 | 36 | **0,00436** | 12 | 0,00478 | orden 0 |
| 96 | 48 | 0,00284 | 16 | **0,00250** | orden 1 |
| 144 | 72 | 0,00182 | 24 | **0,00074** | orden 1 |
| 192 | 96 | 0,00136 | 32 | **0,00031** | orden 1 |

Con presupuestos chicos, **muchos puntos simples aproximan mejor un campo arbitrario que pocos puntos sofisticados**. Si el criterio fuera minimizar error por parámetro, el jacobiano no sería obviamente la elección correcta.

{{< concept-alert type="clave" >}}
La resolución está en identificar cuál es el recurso escaso, y **no son los parámetros**.

Cada punto clave es una parte del objeto que la red debe **descubrir sin supervisión** y seguir de forma consistente entre cuadros. No hay ninguna anotación que diga dónde deberían estar: lo único que los sostiene es la pérdida de reconstrucción y la de equivarianza. Sostener 10 puntos coherentes es factible; sostener 96 no lo es — se vuelven inestables, se solapan, o colapsan sobre la misma región.

**El jacobiano compra precisión sin pagar en número de partes que hay que descubrir.** Es una decisión sobre la dificultad del *aprendizaje*, no sobre la capacidad de *representación*. Y es la clase de razonamiento que no se lee en la tabla de resultados de un paper.
{{< /concept-alert >}}

### I.5. Lo que falta en la descripción de la clase

La clase resume la generación como *"reemplazar frame inicial y listo"*. Falta la pieza sin la cual nada de esto produciría un video mirable: la **máscara de oclusión**.

Cuando una cabeza gira, aparecen regiones —una oreja, el fondo detrás del hombro— que **no existen en la imagen fuente**. Ningún campo de deformación puede generarlas: deformar es transportar píxeles, y esos píxeles no están.

FOMM produce, junto al campo denso, una máscara $\hat{O}_{S\leftarrow D}$ que indica qué partes se obtienen deformando y cuáles hay que **inpaintar**. Separa así dos problemas de naturaleza distinta —transportar lo que existe, generar lo que falta— en vez de pedirle a un solo mecanismo que resuelva ambos.

Es, además, el mismo tipo de decisión que la [Clase 42](/clases/clase-42) encontró en el seguimiento: distinguir explícitamente lo observado de lo inferido, en vez de dejar que un solo módulo mezcle ambos.

---

## Parte II — *Informed guess*, con precisión

La clase define la super-resolución con dos palabras. Son exactas, y se pueden hacer cuantitativas.

### II.1. Cuánta información se destruye

Con promediado por bloques de $f\times f$ sobre parches binarios, cada píxel de salida conserva solo la suma del bloque (**medido**):

| factor | píxeles por bloque | parches HR posibles | valores LR | preimagen media |
|---|---|---|---|---|
| 2× | 4 | 16 | 5 | 3,2 |
| 3× | 9 | 512 | 10 | 51,2 |
| 4× | 16 | 65 536 | 17 | **3855** |

A factor 4, la bajada es **3855 a 1**. El problema está mal planteado en el sentido de Hadamard: la solución no es única, y elegir una es aplicar un prior.

### II.2. Por qué el óptimo en MSE se ve mal

Si se minimiza error cuadrático, la solución óptima es la **esperanza condicional** $\mathbb{E}[x \mid y]$: el promedio de todas las reconstrucciones compatibles. Medido sobre un caso concreto:

| | MSE esperado | nitidez (varianza espacial) |
|---|---|---|
| **promedio** (óptimo en MSE) | **0,2500** | **0,0000** |
| **muestra** de la posterior | 0,4969 | 0,2500 |

El promedio resultó gris uniforme —varianza espacial exactamente cero— y la muestra, un borde nítido. El promedio gana en MSE por construcción y es la única de las dos que **no puede ser una fotografía**.

### II.3. El intercambio, barrido

Interpolando entre ambos (**medido**):

| $\alpha$ | MSE | PSNR (dB) | nitidez | distancia a la estadística real |
|---|---|---|---|---|
| 0,00 | 0,2500 | **6,02** | 0,0000 | **0,2500** |
| 0,25 | 0,2654 | 5,76 | 0,0162 | 0,2338 |
| 0,50 | 0,3117 | 5,06 | 0,0633 | 0,1867 |
| 0,75 | 0,3889 | 4,10 | 0,1412 | 0,1088 |
| 1,00 | 0,4969 | **3,04** | 0,2500 | **0,0000** |

Las dos columnas que importan se mueven **en direcciones opuestas y monótonamente**. Blau y Michaeli (2018) demostraron que existe un límite teórico a este intercambio: mejorar la fidelidad obliga a empeorar el realismo, y viceversa, por debajo de cierta frontera.

Esto explica dos cosas de la literatura: que las métricas se hayan dividido en dos familias (PSNR/SSIM contra LPIPS/FID), y que los modelos generativos ganen en una mientras pierden en la otra.

### II.4. El prior decide, y eso tiene consecuencias

Dos parches completamente distintos que bajan **al mismo píxel LR** (**medido**, ambos con suma 8):

```
prior "bordes verticales"      prior "bordes horizontales"
    1  1  0  0                     0  0  0  0
    1  1  0  0                     0  0  0  0
    1  1  0  0                     1  1  1  1
    1  1  0  0                     1  1  1  1
```

{{< concept-alert type="advertencia" >}}
**La super-resolución no es una operación forense.** Un rostro, una patente o un texto "recuperados" de un video de vigilancia son lo que el prior del modelo considera probable dado lo observado, no lo que había en la escena.

El caso conocido son los modelos de restauración facial que, ante fotos pixeladas de personas de piel oscura, devolvían rostros de rasgos caucásicos. El resultado era plausible **según ese prior** y falso respecto de la realidad. Presentarlo como evidencia es presentar la opinión del modelo como si fuera un dato.

Y el argumento generaliza a las otras seis aplicaciones de la clase: [Speech2Face](/papers/speech2face-oh-2019) reconstruye un rostro promedio compatible con los atributos que la voz sugiere, no la cara de nadie; sus propios autores lo declaran. La regla práctica que sirve para las siete: **preguntar siempre qué parte de la salida estaba en la entrada y qué parte la puso el prior.**
{{< /concept-alert >}}

---

## Parte III — Generar es más fácil que detectar

La clase enumera seis aplicaciones útiles de los deep fakes y no menciona la detección. Vale completar el panorama, porque la asimetría es estructural.

**[FaceForensics++](/papers/faceforensics-rossler-2019)** (2019) construyó la infraestructura: 1,8 millones de imágenes manipuladas con cuatro métodos distintos, tres niveles de compresión, y una línea base humana. Sus resultados:

- Los detectores automáticos **superan claramente a los observadores humanos**. No se puede delegar la verificación en el ojo del usuario.
- El desempeño **cae con la compresión** — y la compresión es exactamente lo que le ocurre a todo video que circula por redes sociales.
- La detección **generaliza mal entre métodos** de generación.

{{< concept-alert type="clave" >}}
**La asimetría:** quien genera solo necesita evadir a los detectores que ya existen; quien detecta tiene que anticipar generadores que aún no se publicaron. Cada método nuevo invalida parcialmente a los detectores entrenados.

Por eso la dirección con más consenso hoy no es detectar la falsificación sino **certificar la procedencia**: firmar criptográficamente el contenido al momento de capturarlo y mantener la cadena de custodia de las ediciones (el estándar C2PA). Es un problema de infraestructura, no de clasificación — y es un buen ejemplo, para cerrar el diplomado, de que **no todo problema que se puede plantear como clasificación conviene resolverlo así**.
{{< /concept-alert >}}

Sobre la lista de aplicaciones de la clase, dos precisiones factuales sin moralizar: los estudios independientes del fenómeno encuentran consistentemente que la enorme mayoría de los deepfakes en circulación son material sexual no consentido dirigido contra mujeres; y el vector con mayores pérdidas económicas documentadas es la **clonación de voz para fraude**, que necesita segundos de audio público. Ver [Síntesis de Medios](/fundamentos/sintesis-de-medios).

---

## Parte IV — La factorización que unifica la clase

El remate conceptual está en las dos últimas diapositivas del método, y conviene hacerlo explícito porque es el cierre del diplomado.

| | Se conserva | Se transfiere | Mecanismo |
|---|---|---|---|
| **Video (FOMM)** | aspecto — *quién* | movimiento — *qué hace* | puntos clave + jacobianos |
| **Audio (SV2TTS)** | timbre — *quién* | contenido — *qué dice* | vector de hablante + texto |

Es la misma operación —**factorizar en identidad y contenido, y recombinar cruzando**— y aparece a lo largo de todo el programa con otros nombres:

- La [Clase 41](/clases/clase-41) la usó para **analizar**: el reconocimiento de voz quiere el contenido y descarta al hablante; el de hablante quiere lo contrario. Aquí se usa para **generar**, cruzando las piezas.
- La [Clase 43](/clases/clase-43) mostró la versión donde una modalidad supervisa a la otra.
- La [Clase 29](/clases/clase-29) dio la maquinaria generativa.
- La [Clase 42](/clases/clase-42) separó, en seguimiento, la apariencia del movimiento — exactamente los mismos dos factores, para un fin distinto.

{{< concept-alert type="recordar" >}}
Si hay una idea que llevarse del cierre, es esta: **buena parte del progreso en audio y video vino de encontrar la factorización correcta del problema**, no de agrandar el modelo. Identidad contra contenido, apariencia contra movimiento, qué contra quién, lo observado contra lo inferido.

La caja de herramientas de la primera diapositiva no es una lista de arquitecturas. Es una colección de formas de partir un problema.
{{< /concept-alert >}}

---

## Resumen de lo verificado

| Afirmación | Resultado |
|---|---|
| El jacobiano reduce el error de aproximación, a $K=10$ | **2,95×** |
| Sobre rotación pura, la representación de orden 1 es exacta | error ~10⁻¹⁶ contra 0,0086 a 40° |
| A igual presupuesto de **parámetros**, orden 0 gana con presupuestos chicos | 48 y 72 params |
| La bajada de resolución a factor 4 es no invertible | **3855 a 1** en promedio |
| El óptimo en MSE tiene nitidez **cero** | 0,0000 contra 0,2500 de una muestra |
| Distorsión y percepción se mueven en direcciones opuestas | PSNR 6,02 → 3,04 mientras la distancia a lo real va de 0,25 a 0 |
| Dos priors distintos dan reconstrucciones incompatibles | ambas bajan al mismo píxel LR |

---

**Siguiente:** la [práctica](practica) — el modelo de movimiento de primer orden y la aritmética del *informed guess*, implementados desde cero en triple framework.
