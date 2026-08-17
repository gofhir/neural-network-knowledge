---
title: "02 - Las tres métricas y lo que esconden"
weight: 20
math: true
---

> La clase compara SORT con DeepSORT en una frase: *"en la práctica DeepSORT es más robusto que SORT para mantener identidades"*. Este camino implementa MOTA, IDF1 y HOTA desde su definición y usa esa aritmética para poner un número a esa frase. El resultado no es el esperado: la contribución declarada de DeepSORT vale **0,35 puntos de MOTA**, y el efecto secundario de su configuración cuesta **2,28**.

---

## 1. Las tres métricas desde su definición

Las tres parten del mismo emparejamiento espacial por frame —el húngaro sobre IoU con umbral $\alpha$— y difieren en **qué cuentan después**.

### 1.1. MOTA

```python
def mota(gt, out, thr=0.5):
    tp = fp = fn = ids = 0
    last = {}                       # gtID -> último prID que se le asignó
    for g, o in zip(gt, out):
        r, c = linear_sum_assignment(-iou_matrix(g, o))
        for i, j in zip(r, c):
            if iou(g[i], o[j]) >= thr:
                tp += 1
                if gi in last and last[gi] != pj:
                    ids += 1        # ID switch: cambió el prID de este objeto
                last[gi] = pj
        fn += len(g) - emparejados; fp += len(o) - emparejados
    return 1 - (fn + fp + ids) / n_gt
```

Es una suma de errores heterogéneos dividida por el número de detecciones verdaderas. Nótese que **nada acota el resultado por abajo**.

### 1.2. IDF1

El emparejamiento aquí es **global entre trayectorias**, no por frame:

```python
def idf1(gt, out, matches):
    cnt = Counter((gtID, prID) for _, gtID, prID in matches)
    # húngaro sobre la matriz de coincidencias: qué prTraj corresponde a qué gtTraj
    M = -np.array([[cnt.get((g, p), 0) for p in prIDs] for g in gtIDs])
    r, c = linear_sum_assignment(M)
    idtp = int(-M[r, c].sum())
    idfn = total_gt - idtp
    idfp = total_pr - idtp
    return idtp / (idtp + 0.5 * idfn + 0.5 * idfp)
```

Una detección solo cuenta si pertenece a la trayectoria que el emparejamiento **global** eligió. Un intercambio de identidad a mitad de camino invalida la mitad de ambas trayectorias.

### 1.3. HOTA

La novedad son los conjuntos de asociación, definidos **por cada verdadero positivo**:

```python
def hota(gt, out, matches, thr=0.5):
    tpa = Counter((gtID, prID) for _, gtID, prID in matches)
    total = 0.0
    for _, gtID, prID in matches:
        t = tpa[(gtID, prID)]
        fna = len_gt[gtID] - t      # mismo objeto, otro ID predicho (o perdido)
        fpa = len_pr[prID] - t      # mismo ID predicho, otro objeto (o falso positivo)
        total += t / (t + fna + fpa)
    detA = tp / (tp + fn + fp)
    assA = total / tp
    return np.sqrt(detA * assA)
```

La media geométrica es lo que hace que HOTA no se pueda "ganar" mejorando solo un eje.

## 2. El ejemplo de los tres trackers, reconstruido

El paper de [HOTA](/papers/hota-luiten-2020) abre con tres trackers sobre un mismo *ground truth* de 100 detecciones de un solo objeto: **A** predice una trayectoria de 50, **B** dos de 35, **C** cuatro de 25. Construyéndolos y midiéndolos con las tres implementaciones:

| Tracker | DetA | AssA | MOTA | IDF1 | HOTA | IDSW |
|---|---|---|---|---|---|---|
| A — 1 track de 50 | 50,0 | 50,0 | 50,0 | 66,7 | 50,0 | 0 |
| B — 2 tracks de 35 | 70,0 | 35,0 | 69,0 | 41,2 | 49,5 | 1 |
| C — 4 tracks de 25 | 100,0 | 25,0 | 97,0 | 25,0 | 50,0 | 3 |

Comparando con lo publicado (A: 50/50/50/**67**/50; B: 70/35/69/**52**/50; C: 100/25/97/**25**/50), la reconstrucción reproduce **exactamente** DetA, AssA, MOTA y HOTA en los tres casos, y el IDF1 de A y de C. Difiere solo en el IDF1 de B: 41,2 % contra 52 %.

{{< concept-alert type="recordar" >}}
**Por qué difiere, y por qué no importa.** La figura del paper es esquemática y no fija todos los detalles: cualquier reparto de las 70 detecciones de B en dos trayectorias de longitudes **distintas** cambia IDF1 sin mover DetA. Repartiendo 44 y 26 en lugar de 35 y 35, IDF1 sube a ~52 %. La discrepancia es de la reconstrucción, no del paper.

El punto cualitativo se sostiene, y con la reconstrucción queda más marcado:

- **MOTA** ordena C > B > A — 97,0 > 69,0 > 50,0.
- **IDF1** ordena A > B > C — 66,7 > 41,2 > 25,0.
- **HOTA** los declara empatados — 50,0 ≈ 49,5 ≈ 50,0.

Tres métricas, tres órdenes, los mismos tres sistemas.
{{< /concept-alert >}}

## 3. Reconstruir la tabla de MOT16

La tabla del paper de [DeepSORT](/papers/deepsort-wojke-2017) reporta los conteos de error y el MOTA, pero **no el denominador**. Se puede despejar:

```python
for name, fp, fn, ids, mota in [("SORT", 8698, 63245, 1423, 0.598),
                                ("DeepSORT", 12852, 56668, 781, 0.614)]:
    err = fp + fn + ids
    print(name, "|gtDet| implícito =", round(err / (1 - mota)))
```

```
SORT     |gtDet| implícito = 182502
DeepSORT |gtDet| implícito = 182127
```

Los dos despejes coinciden dentro del error de redondeo de los MOTA publicados a un decimal. Tomando $|\mathrm{gtDet}| = 182\,326$:

```
MOTA SORT     = 59.76   (reportado 59.8)
MOTA DeepSORT = 61.44   (reportado 61.4)
```

La aritmética cierra. Eso habilita los contrafactuales.

## 4. Los contrafactuales

### 4.1. ¿Cuánto vale reducir los ID switches un 45 %?

Se toman los FP y FN de SORT y se le regalan los 781 ID switches de DeepSORT:

```python
1 - (8698 + 63245 + 781) / 182326   # 0.6011
```

**60,11 contra 59,76: +0,35 puntos.**

Eliminar 642 cambios de identidad —la contribución declarada del paper, su única mejora anunciada en el título y el resumen— vale **un tercio de punto** en la métrica con que se rankeaba el benchmark.

### 4.2. ¿Cuánto cuestan los falsos positivos de $A_{\max}=30$?

Se le dan a DeepSORT los FP de SORT:

```python
1 - (8698 + 56668 + 781) / 182326   # 0.6372
```

**63,72 contra 61,44: +2,28 puntos.**

Los 4154 falsos positivos adicionales que introduce mantener trayectorias vivas medio segundo cuestan **6,5 veces más MOTA de lo que aporta arreglar todos los ID switches**.

{{< concept-alert type="advertencia" >}}
Los dos números juntos dicen algo incómodo: **la mayor parte de los 1,6 puntos que DeepSORT gana no vienen de lo que el paper propone**. Vienen de la reducción de falsos negativos (63245 → 56668), que es consecuencia del mismo $A_{\max}$ que produce los falsos positivos.

Descomposición aproximada del $+1{,}68$ de MOTA:

| Componente | Aporte |
|---|---|
| menos falsos negativos (−6577) | **+3,61** |
| menos ID switches (−642) | **+0,35** |
| más falsos positivos (+4154) | **−2,28** |
| total | **+1,68** |

Esto no significa que DeepSORT no funcione. Reducir un 45 % los cambios de identidad es una mejora real y la aplicación la nota. Significa que **MOTA no la mide**, y que evaluar ese trabajo con MOTA era usar el instrumento equivocado. Es exactamente el argumento que [HOTA](/papers/hota-luiten-2020) formaliza tres años después.
{{< /concept-alert >}}

### 4.3. Por qué MOTA hace esto

```python
for name, fp, fn, ids in [("SORT", 8698, 63245, 1423),
                          ("DeepSORT", 12852, 56668, 781)]:
    err = fp + fn + ids
    print(f"{name}: IDSW aporta {100*ids/err:.2f}% del numerador")
```

```
SORT:     IDSW aporta 1.94% del numerador
DeepSORT: IDSW aporta 1.11% del numerador
```

El 98 % restante son errores de detección. Toda métrica que sume errores heterogéneos sin normalizar por su frecuencia termina midiendo el término más numeroso.

## 5. Por qué Mahalanobis necesita la cascada

El paper de DeepSORT justifica su cascada de matching con un argumento que llama *contraintuitivo*. Se puede verificar en cuatro líneas.

### 5.1. La misma detección, distinta certeza

```python
d = np.array([10.0, 0.0])          # detección a 10 px del centro predicho
for sigma in [1.0, 2.0, 5.0, 10.0]:
    S = np.eye(2) * sigma**2
    print(sigma, d @ np.linalg.inv(S) @ d)
```

| $\sigma$ [px] | 1,0 | 2,0 | 5,0 | 10,0 |
|---|---|---|---|---|
| $d^{(1)}$ | 100,00 | 25,00 | 4,00 | **1,00** |

La misma detección, sin moverse un píxel, es **cien veces más "cercana"** a una trayectoria cuya incertidumbre creció de 1 a 10 px.

### 5.2. Cuándo se invierte la decisión

Dos trayectorias compiten por una detección. **A** fue vista hace 1 frame y su predicción está a **5 px**; **B** lleva $k$ frames perdida y su predicción está a **40 px**. La respuesta correcta es siempre A.

```python
F = np.array([[1.0, 1.0], [0.0, 1.0]]); Q = np.eye(2)
def var_tras(k, p0=1.0):
    P = np.eye(2) * p0
    for _ in range(k): P = F @ P @ F.T + Q
    return P[0, 0]
```

| edad de B | $\sigma_A$ | $\sigma_B$ | $d^{(1)}_A$ | $d^{(1)}_B$ | gana |
|---|---|---|---|---|---|
| 5 frames | 1,73 | 7,81 | 8,33 | 26,23 | A ✓ |
| **10 frames** | 1,73 | 19,90 | 8,33 | **4,04** | **B ✗** |
| 25 frames | 1,73 | 74,51 | 8,33 | 0,29 | B ✗ |
| 40 frames | 1,73 | 148,93 | 8,33 | 0,07 | B ✗ |

**El punto de quiebre está entre los 5 y los 10 frames.** A partir de ahí, una trayectoria perdida hace un tercio de segundo se roba sistemáticamente las detecciones de las que se están siguiendo bien.

La cascada de DeepSORT resuelve primero las trayectorias de menor edad: A toma la detección antes de que B pueda competir. No arregla la métrica; cambia el orden en que se la consulta.

### 5.3. La compuerta que se autodesactiva

El mismo efecto sobre la compuerta $\chi^2_{0{,}95;4} = 9{,}4877$. El radio admisible es $r = \sqrt{t\,\sigma^2}$:

| frames sin detección | 0 | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|---|
| $\sigma$ [px] | 1,00 | 1,73 | 7,81 | 19,90 | 53,77 | **97,40** |
| radio admisible [px] | 3,08 | 5,34 | 24,06 | 61,30 | 165,62 | **300,00** |

Al llegar a $A_{\max}=30$, la compuerta acepta cualquier detección dentro de 300 px. En un cuadro de 1920×1080 eso ya casi no filtra nada.

{{< concept-alert type="clave" >}}
Los tres resultados de esta sección explican, juntos, la tabla de MOT16 de DeepSORT sin necesidad de mirar el descriptor de apariencia:

- $A_{\max}=30$ mantiene trayectorias vivas → **menos falsos negativos** (el grueso de la ganancia de MOTA);
- pero la compuerta se vuelve inútil a esa edad → **más falsos positivos** (−2,28 puntos);
- y Mahalanobis premia a las trayectorias viejas → hace falta **la cascada** para que no arruinen a las jóvenes.

El descriptor de apariencia es lo que hace que las asociaciones recuperadas sean las **correctas**. Pero la decisión que mueve las métricas es el parámetro de un solo número.
{{< /concept-alert >}}

## 6. Las métricas en triple framework

El emparejamiento (`linear_sum_assignment`) es combinatorio y vive en SciPy. Lo que sí se vectoriza es la **matriz de similitud** que lo alimenta y, en evaluaciones grandes, el barrido sobre los 19 umbrales $\alpha$ que HOTA integra.

{{< tabs >}}
{{< tab name="NumPy" >}}
```python
# HOTA integra sobre alpha en {0.05, 0.10, ..., 0.95}: 19 emparejamientos distintos.
alphas = np.arange(0.05, 1.0, 0.05)
S = iou_np(gt_boxes, pr_boxes)                    # (N, M) una sola vez
valid = S[None, :, :] >= alphas[:, None, None]    # (19, N, M) por umbral
hota = np.mean([hota_alpha(S, v) for v in valid])
```
{{< /tab >}}
{{< tab name="PyTorch" >}}
```python
alphas = torch.arange(0.05, 1.0, 0.05)
S = iou_torch(gt_boxes, pr_boxes)
valid = S.unsqueeze(0) >= alphas[:, None, None]
# el emparejamiento sigue en SciPy; PyTorch solo prepara las 19 matrices
```
{{< /tab >}}
{{< tab name="TensorFlow" >}}
```python
alphas = tf.range(0.05, 1.0, 0.05, dtype=tf.float64)
S = iou_tf(gt_boxes, pr_boxes)
valid = S[None, :, :] >= alphas[:, None, None]
```
{{< /tab >}}
{{< tab name="JAX" >}}
```python
# En JAX se escribe para UN umbral y vmap barre los 19 de golpe.
def deta_alpha(S, alpha):
    m = S >= alpha
    tp = jnp.minimum(m.sum(1), 1).sum()           # cota superior sin el húngaro
    return tp / (m.shape[0] + m.shape[1] - tp)

deta_all = jax.jit(jax.vmap(deta_alpha, in_axes=(None, 0)))
```
{{< /tab >}}
{{< /tabs >}}

{{< concept-alert type="recordar" >}}
Vale ser explícito sobre el límite: **el emparejamiento óptimo no se vectoriza**. El algoritmo húngaro es combinatorio y secuencial, y ni `vmap` ni `jit` lo aceleran. Lo que los frameworks aportan aquí es construir las matrices de costo por lotes —y, en una evaluación completa de MOTChallenge, eso es la mayor parte del tiempo, porque son millones de pares de cajas contra 19 umbrales.

En la variante JAX de arriba, `deta_alpha` usa una **cota superior** de los TP en vez del emparejamiento óptimo: es lo que se puede calcular sin salir del grafo. Sirve como diagnóstico rápido, no como métrica reportable.
{{< /concept-alert >}}

---

## Qué se aprendió

1. **Las tres métricas se implementan en menos de cincuenta líneas** y reproducen los números publicados, lo que permite auditarlas en vez de creerles.
2. **El ejemplo de HOTA produce tres órdenes distintos** para los mismos tres sistemas. Reportar una sola métrica es una decisión editorial.
3. **La contribución declarada de DeepSORT vale 0,35 puntos de MOTA**; su efecto secundario cuesta 2,28; y la mayor parte de su ganancia viene de un parámetro de configuración.
4. **Mahalanobis se invierte entre los 5 y los 10 frames** de edad de la trayectoria, y la compuerta $\chi^2$ pasa de admitir 3 px a admitir 300.
5. **El emparejamiento no se vectoriza**; las matrices de costo sí.

---

**Volver a:** [Práctica](../) · [Profundización](/clases/clase-42/profundizacion) · [Teoría](/clases/clase-42/teoria)
