---
title: "Dónde está la evidencia"
weight: 4
---

Corregido el rango, `abseiling` gana con 62–65 %. GluonCV llega a 99,1 %. La diferencia restante no está en el modelo: está en **qué parte del video se le entrega**. Dos experimentos —uno espacial, otro temporal— lo localizan, y una mirada al video explica ambos con una sola causa.

## H3 — El crop espacial: el sujeto no está centrado

El frame mide 454 × 256. Cualquier crop cuadrado descarta **el 44 % del ancho**. ¿Cuál 44 %?

| Crop de 224 px | Top-1 | p(`abseiling`) | rank | $H$ (nats) |
|---|---|---|---|---|
| izquierda | `rock climbing` 32,99 % | 9,26 % | **#4** | **2,753** |
| centro | `abseiling` 65,19 % | 65,19 % | #1 | 0,651 |
| **derecha** | **`abseiling` 75,28 %** | **75,28 %** | #1 | **0,570** |

El crop izquierdo **colapsa**. Su entropía de **2,753 nats** es 4,2 veces la del centro y casi la mitad del máximo teórico ($\ln 400 = 5{,}991$). Eso es el modelo diciendo *"no reconozco nada aquí"*: sin sujeto en el encuadre, sólo queda roca y cielo.

{{< callout type="info" >}}
**La entropía sí funcionó como detector — aquí.** Disparó de 0,65 a 2,75 cuando el input dejó de contener al sujeto. Combinado con el caso de `ApplyEyeMakeup` (0,104 nats con la clase correcta ausente del vocabulario), queda una regla precisa: **la entropía mide ambigüedad entre clases conocidas y falta de evidencia, pero es ciega al vocabulario faltante.**
{{< /callout >}}

## H4 — La ventana temporal

| Ventana | Frames | p(`abseiling`) | Top-1 | $H$ |
|---|---|---|---|---|
| **primeros 64 (0–2,6 s)** | 64 | **92,85 %** | `abseiling` | **0,260** |
| primeros 64, stride 2 | 32 | 87,06 % | `abseiling` | 0,390 |
| completo, stride 2 (12,5 fps) | 125 | 80,73 % | `abseiling` | 0,492 |
| completo | 250 | 65,19 % | `abseiling` | 0,651 |
| centrales (3–7 s) | 100 | 58,46 % | `abseiling` | 0,680 |
| **últimos 100 (6–10 s)** | 100 | **39,36 %** | `rock climbing` ❌ | 0,677 |

El gradiente es monótono y fuerte: **cuanto más tarde la ventana, peor el resultado**, hasta el punto de que los últimos 4 segundos por sí solos invierten la predicción.

No es casualidad que GluonCV tome justamente `range(0, 64, 2)`: la ventana inicial es la más informativa. Y confirma con datos la limitación *trimmed* de I3D — promediar logits sobre los 10 s mezcla la evidencia buena del inicio con la ambigua del final, y como el promedio ocurre **dentro** de la red, esa información no se recupera después.

## La causa común: mirar el video

Los números de H2, H3 y H4 apuntan en la misma dirección, y una tira temporal del clip explica por qué:

![Cinco frames del video de rápel a lo largo de los 10 segundos, mostrando cómo la cámara se abre](/laboratorios/lab-38/abseiling-tira-temporal.jpg)

**La cámara se aleja progresivamente.** A $t = 0$ el encuadre es una pared vertical con la persona y su cuerda claramente legibles. Hacia el final, una arista rocosa ocupa media imagen, el sujeto se ha vuelto una figura pequeña y se ha desplazado hacia el borde derecho.

![Detalle del sujeto al inicio y al final del clip](/laboratorios/lab-38/abseiling-zoom-escala.jpg)

Eso unifica los tres experimentos bajo una sola explicación:

- **H2** (`crop 256 → resize 224` encoge un 12,5 %) cuesta puntos porque el rasgo discriminante —la cuerda, unos pocos píxeles de ancho— se degrada al reescalar.
- **H3** (el crop derecho gana) se explica porque el sujeto está a la derecha del centro geométrico durante buena parte del clip.
- **H4** (la ventana inicial gana) se explica porque es donde el sujeto es más grande y la cuerda todavía es visible.

**La evidencia que separa `abseiling` de `rock climbing` es pequeña, está descentrada y se degrada con el tiempo.** Todo lo que la encoja, la recorte o la promedie con segmentos sin ella cuesta accuracy.

## La anomalía del stride temporal

Un resultado no encaja limpiamente en esa historia:

| | Frames | p(`abseiling`) |
|---|---|---|
| completo, 250 frames (25 fps) | 250 | 65,19 % |
| **completo, stride 2 (12,5 fps)** | 125 | **80,73 %** |
| primeros 64 (25 fps) | 64 | **92,85 %** |
| primeros 64, stride 2 | 32 | 87,06 % |

Sobre el clip completo, **submuestrear ayuda** (+15,5 pts con el mismo contenido y la mitad de frames). Sobre el clip corto, **perjudica** (−5,8 pts).

Una explicación plausible: submuestrear **duplica el desplazamiento aparente entre frames consecutivos**, y el descenso en rápel es lento — a 25 fps el movimiento por frame puede quedar por debajo de la sensibilidad de los filtros temporales inflados. Cuando la evidencia ya es fuerte (primeros 64), perder resolución temporal sólo resta.

{{< callout type="warning" >}}
Es una **hipótesis, no una conclusión**: se testea corriendo strides 3 y 4, y midiendo si la ganancia crece o revierte. Lo que sí queda establecido es que **la tasa de muestreo temporal es un hiperparámetro con efecto de primer orden** — que es exactamente el punto de partida de [SlowFast](/papers/slowfast-feichtenhofer-2019), que en lugar de elegir una tasa usa dos vías con frecuencias distintas.
{{< /callout >}}

## El 3-crop que empeoró las cosas

Con los tres hallazgos en mano, la configuración "correcta" parecía obvia: rango $[-1,1]$ + promedio de los 3 crops horizontales + ventana inicial. El protocolo de múltiples crops es estándar en todos los papers de video.

```
=== Protocolo corregido: [-1,1] + 3-crop + primeros 64 frames ===
  abseiling             : 75.46%
  rock climbing         : 24.19%
  paragliding           :  0.27%
  ice climbing          :  0.07%
```

| Configuración | p(`abseiling`) |
|---|---|
| $[-1,1]$ + **sólo centro** + primeros 64 | **92,85 %** |
| $[-1,1]$ + **3-crop** + primeros 64 | 75,46 % |

**Promediar los tres crops costó 17,4 puntos.** La razón está en H3: el crop izquierdo no es un votante neutro que se diluya en el promedio — vota **activamente** por `rock climbing` con 32,99 %, dejando `abseiling` en rank #4. La aparición de `paragliding` en el top-3, una clase ausente de todas las corridas anteriores, es la firma de esa contaminación.

{{< callout type="warning" >}}
**Un ensemble sólo ayuda si sus miembros son individualmente competentes.** El protocolo de 3-crop presupone que el sujeto está aproximadamente centrado —cierto en promedio para Kinetics, falso en este clip—. Aplicado a ciegas, un promedio robusto se vuelve un promedio con un votante envenenado.
{{< /callout >}}

## El recuento completo

| Configuración | p(`abseiling`) | Δ acumulado |
|---|---|---|
| Baseline del tutorial | 10,86 % ❌ | — |
| + rango $[-1,1]$ (H1) | 62,31 % ✅ | **+51,5** |
| + crop 224 directo (H2) | 65,19 % | +2,9 |
| + ventana inicial 64 frames (H4) | **92,85 %** | **+27,7** |
| *(desvío) + 3-crop (H3 mal aplicado)* | *75,46 %* | *−17,4* |

Dos palancas explican prácticamente todo: **el rango de normalización y la ventana temporal**. El mismo modelo, los mismos pesos, el mismo video.

---

**Siguiente:** [Invertir el tiempo](../05-invertir-el-tiempo) — el experimento cuyo resultado negativo dice más que los cuatro anteriores.
