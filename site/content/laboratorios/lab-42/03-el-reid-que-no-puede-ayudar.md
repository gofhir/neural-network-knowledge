---
title: "El ReID que no puede ayudar"
weight: 3
---

La clase presenta la re-identificación como la mejora natural sobre [SORT](/papers/sort-bewley-2016): el modelo de movimiento supone que el objeto se movió poco, y cuando ese supuesto falla —cámara móvil, oclusiones largas— hace falta información visual sobre *quién* es cada objeto. [DeepSORT](/papers/deepsort-wojke-2017) la introduce, y BoT-SORT la trae implementada y **apagada por defecto**.

El [ID switch de la página anterior](../02-anatomia-de-un-id-switch) es el caso perfecto para probarla: la persona de ropa negra **sigue siendo la persona de ropa negra** entre los frames 236 y 237. Un descriptor de apariencia debería reconocerla sin dudar.

Se activó, y el switch persistió **idéntico**: mismos rangos de frames, mismas cantidades de detecciones, mismo track partido. La razón está en una sola línea de código, y es el hallazgo de esta página.

## El experimento

```yaml
# botsort_reid.yaml — copia de botsort.yaml con una línea cambiada
with_reid: True
```

| Configuración | Tiempo | IDs únicos | Quemados | Evento 2 | ¿Switch? |
|---|---|---|---|---|---|
| baseline | 15,4 s | 9 | 22 | `[7, 9, 13]` | SÍ |
| **ReID activado** | **19,0 s** | **10** ⚠️ | 13 | `[5, 7, 11]` | **SÍ** |
| ReID + `proximity_thresh 0.15` | 17,6 s | 9 | 11 | `[5, 6, 9]` | SÍ |

Tres tracks en el evento 2, que tiene dos personas: el switch sigue ahí en las tres configuraciones. Y la re-identificación cuesta **+23 % de tiempo** (596 frames: de 38,7 a 31,4 fps end-to-end), porque agrega un forward del encoder de apariencia por cada detección.

## La línea que lo explica

`BOTSORT.get_dists()`, en el repositorio de Ultralytics:

```python
def get_dists(self, tracks, detections):
    dists = matching.iou_distance(tracks, detections)
    dists_mask = dists > (1 - self.proximity_thresh)        # ① la compuerta

    if self.args.fuse_score:
        dists = matching.fuse_score(dists, detections)

    if self.args.with_reid and self.encoder is not None:
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
        emb_dists[dists_mask] = 1.0                          # ② aquí muere
        dists = np.minimum(dists, emb_dists)
    return dists
```

Con `proximity_thresh = 0.5`, la máscara marca todo par cuya distancia IoU supere `1 − 0,5 = 0,5`, es decir **IoU < 0,5**. Y en esos pares la distancia de apariencia se **fuerza a 1,0** —el máximo— antes del `np.minimum`.

{{< concept-alert type="clave" >}}
**La apariencia solo puede opinar sobre pares que ya tienen IoU ≥ 0,5.**

El switch ocurrió con IoU ≈ 0,28. El descriptor de esa persona se calculó, se comparó, y su resultado se **descartó antes de entrar a la matriz de costos**. En ese frame, la re-identificación literalmente no participó de la decisión.
{{< /concept-alert >}}

Abrir la compuerta a `proximity_thresh = 0.15` —que habilitaría la apariencia hasta IoU 0,15, por debajo del 0,28 del episodio— tampoco lo corrigió, lo que confirma el diagnóstico de la página anterior: el problema no era la aceptación del par sino el **desempate** entre dos candidatos igualmente aceptables. La apariencia habría podido inclinar la balanza solo si el costo de apariencia del track 9 hubiera sido lo bastante menor que el del track 13 — pero ambos tracks describían **a la misma persona**, así que sus descriptores eran igual de buenos.

## El mismo patrón que en DeepSORT

En la [profundización de la clase](/clases/clase-42/profundizacion) se midió que la compuerta χ² de Mahalanobis de DeepSORT pasa de admitir 3,08 px a 300 px en 30 frames: **se autodesactiva justo cuando más falta hace**. Aquí la patología es la simétrica — la compuerta de proximidad **nunca se abre lo suficiente**.

En ambos casos, el componente caro y sofisticado queda confinado a los casos que el componente barato ya resolvía. Y en ambos hay un eco de la misma decisión de diseño: en los experimentos publicados de DeepSORT, **λ = 0** — la distancia de Mahalanobis actúa solo como compuerta y el costo es puramente de apariencia. La clase describe la mezcla como una suma ponderada; la configuración evaluada es un caso degenerado de esa ecuación.

> Qué componente actúa como **compuerta** y cuál como **costo** determina qué errores el sistema es capaz de corregir. No basta con incorporar información: hay que mirar dónde entra.

## Lo que la re-identificación sí hace

Sería incorrecto concluir que no hizo nada. Los números lo desmienten:

| | Sin ReID | Con ReID | ReID + compuerta abierta |
|---|---|---|---|
| Identidades emitidas | 9 | **10** ⚠️ | 9 |
| **Nacimientos abortados** | **22** | **13** | **11** |

Los nacimientos espurios caen un **41 %**. La apariencia tomó decisiones distintas — solo que en otro régimen.

El orden de las operaciones dice cuál. `dists_mask` se calcula sobre el **IoU crudo**, pero `dists` se fusiona con el puntaje **después**. Entonces la apariencia rescata exactamente este perfil de par:

> **IoU alto (≥ 0,5) pero puntaje bajo** — una detección bien ubicada que `fuse_score` estaba hundiendo.

Con números: IoU = 0,6 y puntaje = 0,15 dan un costo fusionado de $1 - (0,6 \times 0,15) = 0{,}91$, rechazado por el umbral 0,8. Si la apariencia da similitud coseno 0,9, entonces $\text{emb\_dist} = (1-0,9)/2 = 0{,}05$ y $\min(0{,}91,\ 0{,}05) = 0{,}05$: **aceptado**. El track sobrevive en lugar de morir y dejar una detección huérfana que funda un duplicado.

Es decir: **el ReID de Ultralytics no reidentifica objetos que se movieron; contrarresta el castigo que `fuse_score` impone a las detecciones débiles.** Es lo opuesto a su propósito canónico en la literatura, que es recuperar identidades tras oclusiones prolongadas.

Y tiene un costo colateral que conviene no pasar por alto: rescatar detecciones débiles también significa **mantener vivos tracks que deberían morir**. Las identidades emitidas subieron de 9 a 10 sobre 7 personas reales — la inflación empeoró de 1,29× a 1,43×. Es el mismo intercambio con signo que el `A_max = 30` de DeepSORT: el parámetro que recupera falsos negativos introduce falsos positivos.

Con la compuerta abierta a 0,15 el efecto colateral desaparece (vuelve a 9 identidades) y los nacimientos abortados caen a 11, la mitad del baseline. **Esa es la mejor configuración del experimento** — solo que no por la razón que se buscaba.

## Qué significa esto para la escena

El video es de cámara fija, con personas caminando en trayectorias mayormente rectilíneas y previsibles: el caso ideal para un modelo de velocidad constante. En ese régimen la geometría basta, y la apariencia paga un 23 % de cómputo por corregir errores de otro tipo.

No es un resultado universal. La clase menciona explícitamente los escenarios donde el supuesto de movimiento se rompe —cámara que se mueve mucho, periodos grandes de oclusión— y ahí la apariencia sería decisiva. Ninguno de los cinco videos de este laboratorio los ejercita.

Un dato complementario apunta en la misma dirección: en DanceTrack, donde el movimiento es rápido y no lineal pero los objetos son visualmente muy similares, **DeepSORT (45,6 HOTA) queda por debajo de SORT (47,9)**. La progresión SORT → DeepSORT → ByteTrack → [OC-SORT](/papers/oc-sort-cao-2022) no es una escalera: cada método gana en el régimen para el que fue diseñado.

---

**Siguiente:** [El frame rate como parámetro de dificultad](../04-el-frame-rate-como-dificultad) — un experimento cuyo resultado sale al revés de la predicción.
