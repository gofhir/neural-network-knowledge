---
title: "El bug del preproceso"
weight: 3
---

La actividad pide reutilizar el código de la sección 5.0 sobre `abseiling_k400.mp4`. Es el caso más fácil imaginable: la clase `abseiling` es el **índice 0** de Kinetics-400, y el video tiene los parámetros nativos del dataset —**454 × 256, 250 frames, 10,00 s, 25 fps**, exactamente lo que I3D vio en entrenamiento—.

Falla.

```
Top 5 actions:
  rock climbing         : 87.66%
  abseiling             : 11.23%
  ice climbing          :  1.02%
  diving cliff          :  0.05%
  bungee jumping        :  0.02%
```

## Qué se sabe antes de investigar

Dos datos acotan el problema desde el principio.

**El top-5 es coherentísimo.** Las cinco clases son *personas suspendidas en una pared vertical*. El modelo entiende la escena; lo que no resuelve es la distinción fina entre subir y bajar por una cuerda.

**La verdad de terreno no está en duda.** La demo oficial de [GluonCV](https://cv.gluon.ai/build/examples_action_recognition/demo_i3d_kinetics400.html) usa **este mismo video** y reporta `abseiling` con **99,1 %**. Su configuración: 32 frames tomados como `range(0, 64, 2)`, crop central de 224, normalización con estadísticas de ImageNet. O sea: un I3D bien configurado acierta. El problema es de **configuración**, no del modelo.

De ahí salen cinco hipótesis testeables. Las dos primeras se resuelven en esta página; las otras tres, en [dónde está la evidencia](../04-donde-esta-la-evidencia) y [invertir el tiempo](../05-invertir-el-tiempo).

## H1 — El rango de entrada

El experimento cambia una sola línea del preprocesamiento:

```python
def normalizar(fr, rango="0_1"):
    a = fr.astype(np.float32)
    return a/255.0 if rango == "0_1" else a/127.5 - 1.0
```

| Rango de entrada | Top-1 | p(`abseiling`) | rank | $H$ |
|---|---|---|---|---|
| $[0, 1]$ — tutorial de TF Hub | `rock climbing` 88,02 % | 10,86 % | **#2** | 0,408 |
| $[-1, 1]$ — spec de DeepMind | **`abseiling` 62,31 %** | **62,31 %** | **#1** | 0,664 |

**La predicción se invierte con un cambio de una línea.** Queda demostrado que el `/255.0` del tutorial oficial de TensorFlow Hub no coincide con la especificación del repositorio de DeepMind (*"pixel values are then rescaled between -1 and 1"*), y —lo que la documentación no permitía saber— que **el módulo de TF Hub no reescala internamente**.

### El mecanismo: las BatchNorm que I3D heredó

El [inflado](/fundamentos/inflado-de-convoluciones) no copia sólo los kernels convolucionales: también viajan las capas **BatchNorm** de Inception-v1, con sus parámetros $\gamma$, $\beta$ y sus medias y varianzas móviles. Esas estadísticas se calibraron sobre entradas centradas en cero.

Alimentar la red con $[0, 1]$ introduce un **sesgo constante de $+0{,}5$** en los tres canales de la entrada. Para una convolución de kernel $W$, ese desplazamiento se propaga como un término aditivo:

$$\Delta z = \sum_{i} W_i \cdot 0{,}5$$

que no es cero salvo que los pesos sumen cero, y que la BatchNorm siguiente **no corrige**, porque usa estadísticas congeladas de inferencia y no las de este batch. El error se acumula a lo largo de los nueve módulos Inception.

Es una demostración empírica de hasta qué punto I3D *es* una red 2D inflada: arrastra incluso las estadísticas de normalización de su versión para imágenes.

### Por qué el bug pudo sobrevivir años en un tutorial oficial

Porque en la mayoría de los casos **no se nota**:

| Video | Régimen | $[0,1]$ | $[-1,1]$ | Efecto |
|---|---|---|---|---|
| `archery` | en vocabulario, fácil | `archery` 99,66 % ✅ | `archery` **99,97 %** ✅ | marginal |
| `abseiling` | en vocabulario, grano fino | `rock climbing` 88,02 % ❌ | `abseiling` **62,31 %** ✅ | **invierte** |
| `ApplyEyeMakeup` | fuera de vocabulario | `filling eyebrows` 98,23 % | `filling eyebrows` 79,50 % | mejora la calibración |

{{< callout type="warning" >}}
**Un desajuste de preproceso no destruye el modelo: lo degrada exactamente en la frontera de decisión.** Las clases visualmente distintivas sobreviven —`archery` acierta con ambos preprocesos—; las de grano fino se caen. Por eso el 90 % de quienes siguen el tutorial nunca detectan nada raro.
{{< /callout >}}

La tercera fila añade un matiz importante. Sobre `ApplyEyeMakeup`, corregir el preproceso **baja la confianza de 98,23 % a 79,50 %** y multiplica la entropía por cinco (0,104 → ~0,55). Es decir: con el preproceso correcto el modelo se vuelve *apropiadamente menos seguro* ante contenido fuera de su vocabulario. Pero sigue sin poder acertar, porque la clase no existe. Queda así separada con nitidez una distinción operativa:

- **Error numérico** — arreglable con preproceso (`abseiling`).
- **Error estructural** — sólo arreglable con fine-tuning y una capa de salida nueva (`ApplyEyeMakeup`).

## H2 — La escala del crop

El notebook recorta un cuadrado de `min_dim` y luego lo reescala a 224. El protocolo oficial, con el lado corto ya en 256, recorta 224 directamente. La diferencia es que el notebook **encoge todo un 12,5 %**.

| Crop | $[0,1]$ | $[-1,1]$ |
|---|---|---|
| `crop 256 → resize 224` (notebook) | `rock climbing` 88,02 % | `abseiling` 62,31 % |
| `crop 224` directo (oficial) | `rock climbing` 87,14 % | `abseiling` **65,19 %** |

**+2,9 puntos.** Existe, va en la dirección esperada, pero es de segundo orden frente a H1. Notablemente, no cambia el top-1 en ninguno de los dos rangos: la escala modula la confianza, el rango decide la respuesta.

Aun así, apunta en la misma dirección que H3 y H4: todo lo que **achica al sujeto** cuesta puntos. La razón se ve en la página siguiente.

## Lo que queda establecido

1. La causa raíz del fallo de la actividad es el **rango de normalización**, un bug heredado del tutorial oficial de TensorFlow Hub.
2. El mecanismo es la **herencia de las BatchNorm de Inception-v1**, y por tanto es una consecuencia directa de cómo funciona el inflado.
3. El impacto del bug **escala con la dificultad del caso**: nulo en clases distintivas, decisivo en clases vecinas.
4. Con `[-1,1]` la predicción ya es correcta (62,31 %), pero está lejos del 99,1 % de GluonCV. Faltan tres factores.

---

**Siguiente:** [Dónde está la evidencia](../04-donde-esta-la-evidencia) — H3 y H4: el sujeto que se aleja y se desplaza, y la ventana temporal que decide el resultado.
