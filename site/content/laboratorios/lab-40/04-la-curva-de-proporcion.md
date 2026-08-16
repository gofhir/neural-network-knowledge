---
title: "04 - La curva de proporción y el modo online"
weight: 40
math: true
---

> Si apagar el módulo mide su aporte, barrer su intensidad mide algo más fino: **por qué la proporción es 1/4 y no otra cosa**. Y cambiando la dirección del desplazamiento se contesta la pregunta 3 del práctico con números en lugar de paráfrasis.

---

## 1. Dos manipulaciones, un mismo mecanismo

La primera es variar `fold_div`, que controla qué fracción del tensor se desplaza:

$$\text{fracción desplazada} = \frac{2F}{C}, \qquad F = \left\lfloor \frac{C}{\texttt{fold\_div}} \right\rfloor$$

| `fold_div` | 2 | 4 | 8 | 16 | 32 | $10^9$ |
|---|---|---|---|---|---|---|
| fracción | 1,000 | 0,500 | **0,250** | 0,125 | 0,062 | 0 |

`fold_div = 8` es el valor con que se entrenaron los pesos. `fold_div = 2` desplaza el tensor completo — el *naive shift* que el paper descarta.

La segunda es reemplazar la función de desplazamiento por su variante causal, que es el modo online:

```python
def shift_unidireccional(x, n_segment, fold_div=3, inplace=False):
    """TSM online: solo el pasado entra al presente."""
    nt, c, h, w = x.size()
    x = x.view(nt // n_segment, n_segment, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(x)
    out[:, 1:, :fold] = x[:, :-1, :fold]      # pasado -> presente
    out[:, :, fold:]  = x[:, :, fold:]        # el resto intacto
    return out.view(nt, c, h, w)

TemporalShift.shift = staticmethod(shift_unidireccional)
```

Como `shift` es un método estático de la clase, sustituirlo afecta a los 16 módulos a la vez y es reversible.

---

## 2. El barrido completo

Valores de $p$(clase correcta) — `high jump` para el primero, `playing guitar` para el resto:

| Configuración | HighJump | g01 | g02 | g03 | g04 | g05 |
|---|---|---|---|---|---|---|
| bidireccional, 100 % | **0,52 %** | 11,05 % | 54,21 % | 99,19 % | 51,51 % | 55,16 % |
| bidireccional, 50 % | 3,53 % | 48,01 % | 31,35 % | 99,86 % | **91,19 %** | 44,75 % |
| **bidireccional, 25 %** (entrenado) | **99,12 %** | 40,66 % | 18,92 % | 99,49 % | 77,13 % | 59,63 % |
| bidireccional, 12,5 % | 93,19 % | 33,57 % | 26,43 % | 99,66 % | 85,49 % | 53,90 % |
| bidireccional, 6,2 % | 76,11 % | 22,81 % | 28,32 % | 99,51 % | 81,04 % | 41,38 % |
| identidad (sin shift) | 16,37 % | 10,39 % | 24,05 % | 99,06 % | 66,30 % | 42,90 % |
| **unidireccional, 12,5 %** (online) | 92,73 % | 33,38 % | 24,34 % | **99,71 %** | 88,29 % | **61,34 %** |

La fila de 25 % reproduce exactamente los valores del [bloque anterior](03-la-ablacion-del-shift), lo que confirma que el montaje es consistente y las demás filas son comparables.

---

## 3. La curva de HighJump

```
proporción desplazada    p(high jump)
1.000  (todo)                0.52%   ▏
0.500                        3.53%   ▏
0.250  ← entrenado          99.12%   ████████████████████████████████████████
0.125                       93.19%   █████████████████████████████████████
0.062                       76.11%   ██████████████████████████████
0.000  (identidad)          16.37%   ██████
```

Una U invertida con el máximo **exactamente en la proporción de entrenamiento**. Dos cosas para leer ahí.

**El pico confirma el valor del paper.** El barrido de los autores sobre Kinetics eligió 1/4 tras probar 1/8, 1/4, 1/2 y 1. Este barrido, sobre un solo video y con pesos fijos, cae en el mismo lugar.

**La asimetría es el hallazgo.** Quedarse corto degrada de forma **suave y monótona** (93 → 76 → 16); pasarse **colapsa** (3,5 → 0,5). No es simétrico, y la razón es mecánica:

| Dirección del error | Qué pasa | Efecto |
|---|---|---|
| Desplazar de menos | canales que el modelo esperaba mezclados llegan sin mezclar | **pierde** información temporal, gradualmente |
| Desplazar de más | canales que codificaban apariencia llegan contaminados con otro instante | **corrompe** la representación espacial |

{{< concept-alert type="clave" >}}
Perder información degrada; corromperla destruye. La [profundización](/clases/clase-40/profundizacion) lo formaliza como un problema de **asignación de un presupuesto fijo de canales**: con `fold_div = 2` el bloque "presente" se vacía por completo y **ningún canal representa el instante actual**. Que el resultado sea 0,52 % —peor que el azar informado— es coherente con eso, y es la demostración empírica de por qué el paper necesitó inventar el *partial shift*.
{{< /concept-alert >}}

**Qué no es esta curva.** No es la Figura 2b del paper. Allá cada punto se **reentrena** y la caída por exceso es de un par de puntos de accuracy. Acá los pesos son fijos, así que lo que se mide es la **sensibilidad al desajuste** entre inferencia y entrenamiento. La forma coincide; el mecanismo que la produce, no.

---

## 4. La ausencia de patrón como resultado

En los videos de guitarra la curva no existe. Los óptimos caen en configuraciones distintas en cada clip:

| Video | mejor configuración | mejor $p$ | rango (max − min) |
|---|---|---|---|
| HighJump | 25 % (la entrenada) | 99,12 % | **98,60** |
| g04 | 50 % | 91,19 % | 39,68 |
| g01 | 50 % | 48,01 % | 37,62 |
| g02 | **100 %** | 54,21 % | 35,29 |
| g05 | unidireccional | 61,34 % | 19,96 |
| **g03** | 50 % | 99,86 % | **0,80** |

`g02` rinde mejor desplazando el **100 %** de los canales que con la configuración correcta —54,21 % contra 18,92 %—, lo que no tiene ninguna interpretación mecánica: es un modelo perdido en ese clip, y las perturbaciones lo mueven al azar.

Y `g03` varía **0,80 puntos** entre todas las configuraciones, incluyendo desplazar todo y no desplazar nada.

{{< concept-alert type="clave" >}}
**El rango del barrido es una medida de cuánta temporalidad contiene una acción.** 98,60 puntos para el salto alto; 0,80 para un video de guitarra bien encuadrado — dos órdenes de magnitud. Cuando la acción es temporal, perturbar el eje temporal mueve el resultado de forma sistemática y predecible; cuando no lo es, produce fluctuaciones sin estructura. **La ausencia de patrón es la firma de una acción cuasi-estática**, y es un resultado positivo, no una medición fallida.
{{< /concept-alert >}}

---

## 5. El futuro no aporta nada medible

La pregunta 3 del práctico es por la diferencia entre desplazamiento en línea y fuera de línea. La respuesta habitual —"offline es mejor porque ve el futuro"— se puede poner a prueba comparando ambos modos **a igual presupuesto de canales**:

| Configuración | canales desplazados | ve el futuro | HighJump |
|---|---|---|---|
| bidireccional, `fold_div=8` | 25,0 % | sí | 99,12 % |
| bidireccional, `fold_div=16` | 12,5 % | sí | 93,19 % |
| **unidireccional, `fold_div=8`** | **12,5 %** | **no** | **92,73 %** |
| identidad | 0 % | — | 16,37 % |

Las dos filas del medio mueven **la misma cantidad de canales**. La diferencia entre ver el futuro y no verlo es de **0,46 puntos** — y eso con pesos entrenados de forma bidireccional, lo que debería favorecer al bidireccional.

La ventaja del modo offline (99,12 %) no viene de acceder a los frames futuros: viene de **desplazar el doble de canales**. Lo que importa es el ancho de banda temporal, no su dirección.

En `g04` y `g05` el unidireccional es directamente **la mejor de las siete configuraciones** (88,29 % y 61,34 %).

### Esto reproduce un resultado del paper

No es una refutación: es una reproducción independiente. La Tabla 6 del paper original, que la clase no cubre, mide exactamente lo mismo con modelos **reentrenados** en cada modo:

| Modelo | Kinetics | UCF-101 | HMDB-51 | Something-V1 |
|---|---|---|---|---|
| TSN (baseline 2D) | 70,6 % | 91,7 % | 64,7 % | 20,5 % |
| TSM offline (bidireccional) | 74,1 % | **95,9 %** | 73,5 % | **47,3 %** |
| TSM online (unidireccional) | **74,3 %** | 95,5 % | **73,6 %** | 46,3 % |

En Kinetics y HMDB-51 el modo online **supera** al offline. Solo pierde en Something-Something, donde el orden es constitutivo, y por 1,0 punto.

Dos caminos distintos —el del paper reentrenando, este parcheando el método sobre pesos fijos— llegan a la misma conclusión: **en tareas dominadas por apariencia, la causalidad no cuesta precisión**.

{{< concept-alert type="nota" >}}
Es la lectura que importa para desplegar. Un sistema en vivo —video quirúrgico, monitoreo, conducción— **no puede** esperar frames futuros. Este resultado dice que esa restricción, que suena severa, cuesta menos de medio punto salvo que la tarea consista precisamente en distinguir el orden de los eventos. El paper lo confirma del lado del costo: el modo online añade **menos de 0,1 ms por frame** sobre la baseline 2D, y corre a 13,4 ms en una Jetson Nano.
{{< /concept-alert >}}

---

## 6. Salvedades

**Pesos fijos.** Ninguna de las siete configuraciones fue reentrenada. Los números miden desajuste respecto del entrenamiento, no rendimiento alcanzable de cada variante.

**Un solo video con dinámica fuerte.** La curva limpia es la de HighJump. Los cinco de guitarra aportan el resultado complementario —la ausencia de patrón— pero no una segunda curva.

**El unidireccional está en desventaja estructural.** Los pesos aprendieron a leer el futuro en un bloque de canales específico; la variante causal deja ese bloque con pasado. Que aun así iguale al bidireccional a igual proporción refuerza la conclusión en lugar de debilitarla.

---

## Ver también

- [03 - La ablación del shift](03-la-ablacion-del-shift) — el caso extremo de este barrido, y el control que lo valida.
- [05 - Los defectos del notebook](05-los-defectos-del-notebook) — lo que hay que arreglar para que todo esto corra.
- [Clase 40 - Profundización, Parte I](/clases/clase-40/profundizacion) — por qué desplazar de más corrompe en vez de degradar.
- [Paper: TSM](/papers/tsm-lin-2019) — la Tabla 6 completa y el modo online con su caché de 0,9 MB.
- [Fundamento: Desplazamiento Temporal](/fundamentos/desplazamiento-temporal) — bidireccional contra unidireccional, presentado de forma autónoma.
