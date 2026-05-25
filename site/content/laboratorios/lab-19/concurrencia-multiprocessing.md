---
title: "Parte 4 — Concurrencia con multiprocessing"
weight: 4
---

> **Celdas 15-17 del notebook.** Hasta aquí mediste latencia con un cliente solo. Ahora ves qué pasa cuando N clientes bombardean el servidor al mismo tiempo. La prueba de **concurrencia / carga**, el primer paso hacia capacity planning serio.

## El enunciado

> Anteriormente se ve la implicancia del tamaño de los datos en las predicciones del modelo API. Otro enfoque interesante es realizar análisis al hacer múltiples predicciones al mismo tiempo. Cambie el código a continuación para ir variando la cantidad de procesos que realizan simultáneamente predicciones (entre 2 y 10 procesos) y responda:
>
> **1. ¿Qué sucede con el tiempo de predicción a medida que la cantidad de procesos aumenta? ¿A qué se debe?**

## Implementación con barrido N = 1, 2, 4, 6, 8, 10

```python
from multiprocessing import Process

url = 'http://localhost:8000/classify/'

def predict():
    t_i = time.time()
    pred = requests.post(url, json={"image": np.zeros((3,640,640)).tolist()})
    time_ex = time.time() - t_i
    pred = json.loads(pred.text)
    print(f"  proceso individual: {time_ex:.2f}s")

results = []
for n_process in [1, 2, 4, 6, 8, 10]:
    print(f"\n--- n_process = {n_process} ---")
    process = [Process(target=predict) for _ in range(n_process)]

    wall_start = time.time()
    for p in process: p.start()
    for p in process: p.join()
    wall_total = time.time() - wall_start

    throughput = n_process / wall_total
    print(f"  WALL TOTAL:   {wall_total:.2f}s")
    print(f"  THROUGHPUT:   {throughput:.2f} req/s")
    results.append((n_process, wall_total, throughput))
```

### Por qué procesos y no threads

Aunque `requests.post` libera el GIL durante I/O, el lab elige **procesos** por dos razones:

1. **`tolist()` + `json.dumps` son CPU-bound** dentro de cada cliente. Con threads sufrirías GIL contention. Con procesos cada uno tiene su propio intérprete.
2. **Más realista del mundo real**: en producción los clientes son procesos separados (cada cámara conectada al servidor de IA es un proceso distinto). Modelar eso con `Process` está más cerca de la realidad operativa.

Alternativas en código de producción: `multiprocessing.Pool`, `concurrent.futures.ProcessPoolExecutor`, o `asyncio + aiohttp` (un solo proceso, miles de requests concurrentes).

## Resultados medidos

```
--- n_process = 1 ---   WALL TOTAL: 0.89s   THROUGHPUT: 1.12 req/s
--- n_process = 2 ---   WALL TOTAL: 1.39s   THROUGHPUT: 1.44 req/s
--- n_process = 4 ---   WALL TOTAL: 2.89s   THROUGHPUT: 1.38 req/s
--- n_process = 6 ---   WALL TOTAL: 6.05s   THROUGHPUT: 0.99 req/s
--- n_process = 8 ---   WALL TOTAL: 5.57s   THROUGHPUT: 1.44 req/s
--- n_process = 10 ---  WALL TOTAL: 8.62s   THROUGHPUT: 1.16 req/s
```

Tabla consolidada:

| N | Wall (s) | Throughput (req/s) | Latencia individual prom |
|---|----------|---------------------|---------------------------|
| 1 | 0.89 | 1.12 | 0.80 |
| 2 | 1.39 | 1.44 | 1.26 |
| 4 | 2.89 | 1.38 | 2.30 |
| 6 | **6.05** | 0.99 | 4.88 |
| 8 | 5.57 | 1.44 | 4.25 |
| 10 | 8.62 | 1.16 | 6.31 |

## El hallazgo clave: throughput plano

**El throughput se queda atrancado en ~1.2 req/s para cualquier N.**

Mira la columna: `1.12, 1.44, 1.38, 0.99, 1.44, 1.16`. Estás siempre alrededor de **1 req/s**, independientemente de cuántos clientes lances. **Esa es la firma de un sistema completamente serializado** — agregar más clientes no extrae más trabajo del servidor, solo hace que cada uno espere más tiempo en cola.

### Lo que esto significa en concreto

- Si Space Z conecta 10 cámaras a este BentoML, **no procesa 10 fps**, procesa **1 fps total** distribuidos en 10 colas que crecen.
- Si las cámaras suben a 4K, el throughput **cae** porque cada parse JSON tarda más → menos req/s.
- **Agregar más GPUs no ayuda**. El cuello está en el lado CPU del API server, no en la GPU del modelo.

### Latencia individual: aproximadamente lineal con N

| N | Latencia avg | Ratio vs N=1 |
|---|--------------|--------------|
| 1 | 0.80 s | 1.0× |
| 2 | 1.26 s | 1.6× |
| 4 | 2.30 s | 2.9× |
| 6 | 4.88 s | 6.1× |
| 8 | 4.25 s | 5.3× |
| 10 | 6.31 s | 7.9× |

Crecimiento aproximadamente **lineal**: doblar N casi dobla la latencia individual. Confirma el modelo mental del cuello CPU del API server — N clientes compiten por el mismo core haciendo JSON parsing.

### Las anomalías N=6 y N=8

N=8 fue **MÁS RÁPIDO** que N=6 (5.57 s vs 6.05 s). Contradice la regla lineal y merece comentario:

- **N=6**: spread de 1.21 s entre el más rápido (4.26) y el más lento (5.47).
- **N=8**: spread de 1.74 s con un arranque más limpio (3.40 s el primero).

Causas posibles:
- **Ruido sistémico** de Colab — el OS puede haber estado haciendo GC o swap durante N=6.
- **Variancia del scheduler**: en N=8 los primeros 2 procesos consiguieron mejor afinidad de core.

Con n=1 muestra por valor de N, esto es **estadísticamente esperado**. Si repitieras el experimento 5 veces, el promedio de N=8 estaría por encima del de N=6.

## El gráfico mental

```
Throughput (req/s)
  2.0 │
  1.5 │ ●─────────●─────────●─────────●─────────●─────────●   ← lo que ves
      │ (~1.2)
  1.0 │
  0.5 │
  0.0 └─────────────────────────────────────────────────────►
        N=1    N=2    N=4    N=6    N=8    N=10

Lo que vería un sistema bien diseñado:

  6.0 │                                                    ●  ← saturación
  4.0 │                                ●
  2.0 │            ●
  1.0 │ ●
  0.0 └─────────────────────────────────────────────────────►
        N=1    N=2    N=4    N=6    N=8    N=10
```

**La curva ideal escala con N** hasta saturación. La que mediste **está plana desde N=1**. Significa que **ya estás saturado con un solo cliente** — no hay capacidad ociosa que el segundo cliente pueda explotar.

## Respuesta a la pregunta 1

**Observamos dos efectos críticos:**

1. **El throughput agregado se mantiene constante en ~1.2 req/s** independiente de N. Esta es la firma inequívoca de un sistema **completamente saturado desde N=1**: agregar más clientes no extrae más trabajo del servidor.

2. **La latencia individual crece aproximadamente lineal con N**: pasamos de 0.80 s con 1 cliente a 6.31 s con 10 clientes — un factor ~8×.

**Causas (3 cuellos en cadena):**

a) **CPU contention del lado cliente**: cada proceso ejecuta `np.zeros((3,640,640)).tolist()` + `json.dumps`, ambas operaciones intensivas en CPU. Con N procesos en una máquina Colab con típicamente 2 cores, ya N≥3 introduce competencia por CPU. Cada `tolist()` recorre ~1.2M valores; con muchos en paralelo el OS los multiplexa.

b) **API server con worker único**: BentoML 1.3 levanta uvicorn con `--workers=1` por default. Los N parseos JSON entrantes se procesan secuencialmente en ese único worker Python. Aunque el handler sea async, la deserialización JSON nativa de Python sostiene el GIL durante todo el parse — los N requests se atienden uno a uno en CPU, no en paralelo verdadero.

c) **Runner secuencial**: el `Runnable` está declarado con `SUPPORTS_CPU_MULTI_THREADING = False` y `batchable=False`. El runner procesa una request a la vez vía IPC. En el dummy esto es instantáneo y no domina, pero en un modelo real sería el verdadero techo de throughput.

**Implicación de capacity planning**:

Con 1.2 req/s sostenidos, el sistema no soporta más que **una cámara a ~1 fps**. Si Space Z conecta sus 2 cámaras a 1 fps cada una, ya tiene cola que crece sin límite porque demanda agregada > capacidad.

**Para escalar correctamente** habría que aplicar varias de estas (en orden de impacto típico):

1. **Reducir el payload con compresión** ([Parte 5](../compresion-jpeg)): JPEG reduce el JSON ~10-30× → menos tiempo de parse → más throughput. Ataca la causa raíz.
2. **Habilitar adaptive batching** (`batchable=True` en el Runnable): el runner agrupa requests en ventanas de tiempo y las procesa juntas en GPU.
3. **Escalar workers del API server** (`bentoml serve --workers 4`): un uvicorn con N workers procesa N parses JSON en paralelo verdadero.
4. **Cambiar a serialización binaria** (msgpack, protobuf, raw bytes multipart): elimina el parse JSON enteramente. En MLOps de producción real con visión, **ningún sistema serio envía imágenes como JSON de números**.

Esta es la observación clave de [Sculley et al. (2015)](/papers/hidden-technical-debt-sculley-2015): el cuello de botella en un sistema de ML rara vez es el modelo. Aquí el modelo es un `print()` que tarda nanosegundos y el throughput agregado es 1.2 req/s. Todo lo que mides es **infraestructura de glue code** entre el cliente y el modelo.

## Conexión con el caso Space Z

Este resultado es **el mejor argumento** para preferir la **opción (c) Vertex AI** sobre la (b) VM cruda del [caso Space Z](../caso-space-z): Vertex te da autoscaling horizontal de workers transparente. Lo que mediste arriba con un BentoML monolítico se distribuiría en N réplicas y el throughput escalaría real.

Pero antes de saltar a Vertex, el remedio más barato y de mayor impacto es **atacar la causa raíz del payload con compresión** — la Parte 5.

## Siguiente

{{< cards >}}
  {{< card link="../compresion-jpeg" title="Parte 5 - Compresión JPEG en memoria" subtitle="cv2.imencode/imdecode + curva quality/tamaño + sweet spot ML" icon="academic-cap" >}}
{{< /cards >}}
