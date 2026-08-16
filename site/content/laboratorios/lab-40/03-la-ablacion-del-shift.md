---
title: "03 - La ablación y el control g03"
weight: 30
math: true
---

> Todo el laboratorio afirma que TSM aporta modelado temporal. Nadie lo mide. Y se puede medir sin reentrenar nada, porque el módulo no tiene parámetros: basta apagarlo y ver qué se pierde. El resultado es que **el aporte no es uniforme** — va de 82,76 puntos a 0,42 según el video.

---

## 1. Cómo se apaga un módulo sin pesos

La palanca está en la aritmética del fold:

```python
fold = c // fold_div
out[:, :-1, :fold]      = x[:, 1:,  :fold]
out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
out[:, :,  2*fold:]     = x[:, :,   2*fold:]
```

Si `fold_div` es lo bastante grande, `fold = 0`: las dos primeras asignaciones quedan vacías —rebanadas de longitud cero— y la tercera copia el tensor completo. **El módulo se vuelve la identidad.** El modelo se reduce a un [TSN](/papers/tsn-wang-2016) 2D puro, con exactamente los mismos pesos.

```python
from ops.temporal_shift import TemporalShift

def set_shift(on):
    n = 0
    for m in net.modules():
        if isinstance(m, TemporalShift):
            m.fold_div = 8 if on else 10**9      # fold = C // 10^9 = 0  ->  identidad
            n += 1
    return n

print('módulos TSM afectados:', set_shift(True))   # 16
```

Es reversible, no requiere reconstruir el modelo ni recargar el checkpoint, y aísla exactamente la variable de interés. Como el menor $C$ de la red es 64, cualquier `fold_div` mayor que 2048 anula los 16 módulos de una vez.

---

## 2. El resultado

Seis videos: uno con dinámica fuerte y cinco cuasi-estáticos. La columna es $p$(clase correcta):

| Video | con TSM | sin TSM | Δ | top-1 sin TSM |
|---|---|---|---|---|
| `v_HighJump_g01_c02` | 99,12 % | 16,37 % | **+82,76** | high jump (16,4 %) |
| `v_PlayingGuitar_g01_c01` | 40,66 % | 10,39 % | +30,27 | **playing clarinet** (38,2 %) |
| `v_PlayingGuitar_g02_c01` | 18,92 % | 24,05 % | −5,13 | **playing clarinet** (28,7 %) |
| `v_PlayingGuitar_g03_c01` | 99,49 % | 99,06 % | **+0,42** | playing guitar (99,1 %) |
| `v_PlayingGuitar_g04_c01` | 77,13 % | 66,30 % | +10,83 | playing guitar (66,3 %) |
| `v_PlayingGuitar_g05_c01` | 59,63 % | 42,90 % | +16,72 | playing guitar (42,9 %) |

```
HighJump   +82.76  ████████████████████████████████████
g01_c01    +30.27  █████████████
g05_c01    +16.72  ███████
g04_c01    +10.83  ████
g03_c01     +0.42  ▏
g02_c01     −5.13  (falla con y sin shift)
```

---

## 3. Por qué la ablación es válida: el control g03

La objeción evidente a este método es que apagar el módulo saca al modelo de la distribución en la que fue entrenado. Los pesos aprendieron *contando* con el desplazamiento; sin él, la red no es "el TSN equivalente" sino un modelo roto. ¿Cómo distinguir "quité información temporal" de "rompí el modelo"?

Los datos responden solos:

```
g03_c01     99.49 % → 99.06 %     Δ  +0.42     ← romper el modelo NO lo degradó
HighJump    99.12 % → 16.37 %     Δ +82.76     ← acá sí, y muchísimo
```

Si la ablación produjera una degradación genérica, **`g03` también habría caído**. Ambos parten de ~99 % de confianza; uno se desploma y el otro no se mueve medio punto.

{{< concept-alert type="clave" >}}
`g03` funciona como **control negativo natural**: un caso donde la manipulación no degrada, lo que descarta que la caída de los demás sea un artefacto de sacar al modelo de su distribución. El efecto medido es **específico del contenido temporal del video**. Sin ese control, los 82,76 puntos del salto alto serían ininterpretables.
{{< /concept-alert >}}

Un detalle refuerza la lectura: aun con 16,37 %, `high jump` **sigue siendo el top-1** en el video de salto. El modelo degradado no produce basura aleatoria — conserva el ranking y pierde confianza. Eso es consistente con "se le quitó una fuente de evidencia", no con "se lo destruyó".

---

## 4. TSM es un mecanismo de rescate, no una mejora uniforme

Ordenando los cinco clips de guitarra por su confianza base y mirando el Δ:

| Video | confianza base | Δ al quitar TSM |
|---|---|---|
| `g03` | 99,49 % | +0,42 |
| `g04` | 77,13 % | +10,83 |
| `g05` | 59,63 % | +16,72 |
| `g01` | 40,66 % | +30,27 |
| `g02` | 18,92 % | −5,13 (falla igual) |

Excluyendo `g02` —donde el modelo se equivoca con y sin shift—, la relación es **monótona**: cuanto mayor es la confianza, menos aporta el desplazamiento. Donde la evidencia estática ya resuelve el caso, el módulo es irrelevante; donde la apariencia es ambigua, es lo que rescata la predicción.

Esto explica una cifra del paper que suele leerse mal. TSM gana **+3,5 puntos en Kinetics**, y es tentador entenderlo como "+3,5 en cada video". No es eso: es ≈0 en la mayoría —los que se resuelven con un frame— más una recuperación grande en la minoría difícil. Los **+28,0 puntos en Something-Something** son lo que ocurre cuando *todos* los videos pertenecen a esa minoría.

$$\Delta_{\text{promedio}} = \underbrace{(1-\rho) \cdot \epsilon}_{\text{casos fáciles}} + \underbrace{\rho \cdot \Delta_{\text{grande}}}_{\text{casos difíciles}}$$

Con $\rho$ = fracción de videos donde la apariencia no basta. En Kinetics $\rho$ es chico; en Something-Something, casi 1.

---

## 5. El clarinete vuelve, y ahora significa algo

Sin desplazamiento, `playing clarinet` pasa a ser **top-1 en dos clips**: `g01` con 38,2 % y `g02` con 28,7 %.

En el [bloque anterior](02-la-varianza-intra-clase) esa confusión se había descartado por no repetirse entre los cinco clips. La ablación muestra que estaba mal descartada: **no se repetía porque el shift la estaba resolviendo**. Al quitar el tiempo, emerge de forma sistemática.

Lo que eso demuestra es concreto. Guitarra y clarinete son casi indistinguibles por **pose estática**:

- persona sentada, torso frontal
- brazos flexionados
- manos juntas frente al pecho
- dedos en movimiento fino
- objeto alargado sostenido con ambas manos

Lo que los separa es el **movimiento**: el brazo derecho que rasguea contra la mano que se queda quieta soplando. Esa es exactamente la información que transporta el 25 % de canales desplazados ±1 frame.

{{< concept-alert type="nota" >}}
Es un caso donde el modelado temporal **decide entre dos clases que la apariencia confunde**, medido sobre datos propios y no citado del paper. Y encaja con el hallazgo del micrófono del bloque anterior: los dos clips donde emerge el clarinete son los dos con el micrófono más cerca de la boca. La confusión postural y la contextual se refuerzan.
{{< /concept-alert >}}

---

## 6. El caso g02

Es el único donde la ablación **mejora** la clase correcta: 18,92 % → 24,05 %, un Δ de −5,13.

No es una anomalía interesante sino ruido: el modelo falla en las dos configuraciones. Con shift predice `playing harmonica` (37,72 %); sin shift predice `playing clarinet` (28,7 %). En ninguna de las dos acierta, y la probabilidad de la clase correcta se mueve dentro de un rango donde no es la ganadora de todos modos.

Vale registrarlo por dos razones. Primero, porque **omitir el caso incómodo sería sesgar la muestra**. Segundo, porque marca el límite del método: la ablación mide la contribución del desplazamiento *cuando el modelo está resolviendo la tarea*. Si ya falla, la manipulación produce fluctuaciones sin dirección.

---

## 7. Lo que este experimento no prueba

Tres salvedades explícitas:

**No es una comparación limpia TSN contra TSM.** Los pesos se entrenaron con desplazamiento; apagarlo produce un modelo fuera de distribución, no el TSN que se habría obtenido entrenando sin el módulo. Los Δ son **cotas superiores** de la contribución, no diferencias arquitectónicas. La comparación limpia requiere entrenar ambos, que es lo que hace el paper (70,6 % contra 74,1 % en Kinetics).

**La muestra es chica.** Seis videos, uno solo con dinámica fuerte. El patrón monótono entre confianza base y Δ se sostiene sobre cuatro puntos.

**El control valida el método, no generaliza el resultado.** Que `g03` no se degrade descarta el artefacto genérico para *estos* videos; no garantiza que la ablación sea inocua en cualquier régimen.

Con eso dicho, la afirmación que sí queda establecida es fuerte: **sobre un mismo modelo y un mismo protocolo, el desplazamiento temporal vale 82,76 puntos en un video y 0,42 en otro**, y la diferencia está en el contenido, no en el modelo.

---

## Ver también

- [04 - La curva de proporción](04-la-curva-de-proporcion) — en vez de apagar el módulo, barrer su intensidad.
- [02 - La varianza intra-clase](02-la-varianza-intra-clase) — de dónde salen los cinco clips y por qué difieren.
- [Clase 40 - Práctica](/clases/clase-40/practica/01-el-modulo-de-desplazamiento) — la verificación de que `fold_div` grande produce la identidad exacta.
- [Laboratorio 38 - Invertir el tiempo](/laboratorios/lab-38/05-invertir-el-tiempo) — el experimento hermano sobre I3D: la simetría rota que el modelo no usa.
- [Paper: TSM](/papers/tsm-lin-2019) — la tabla de +3,5 contra +28,0 que este experimento descompone.
