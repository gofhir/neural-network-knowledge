---
title: "Parte 3 — Latencia vs tamaño del payload"
weight: 3
---

> **Celdas 12-14 del notebook.** Benchmark sistemático: ¿cómo escala la latencia cuando varía el tamaño de la imagen? Las dimensiones `(3, shape, shape)` con `shape ∈ [640, 1024, 2048, 3028]` y 3 trials cada uno producen evidencia cuantitativa directa.

## El enunciado

> Considere una lista de dimensiones desde 640 hasta 3028. Realice predicciones para 4 tamaños distintos de imagen `(3, shape, shape)` de valores 0. Tome los tiempos de ejecución promedio para cada uno (con una muestra tamaño 3 para cada uno basta). Responda:
>
> **1. ¿Existe alguna diferencia de tiempo de "inferencia"? ¿A qué crees que se deba?**

> **Las comillas en "inferencia"** son una pista: no estamos midiendo inferencia real (el modelo es dummy). El evaluador espera que **identifiques que la variación viene de transporte y no de cómputo**.

## Por qué los tamaños específicos

| Shape | Contexto típico |
|-------|------------------|
| 640   | Input clásico de YOLOv5/v8 nano y small. Cámaras de seguridad típicas comprimen a esto. |
| 1024  | Input típico de YOLOv5/v8 large y modelos de segmentación más finos. |
| 2048  | Cámaras 1080p sin downscale, frames 4K downsampleados. |
| 3028  | Territorio absurdo (probablemente typo del autor; debería ser 3024/3072). Una imagen 3028×3028×3 uint8 son 27 MB en RAM. Serializada como JSON: 100-150 MB. Intencionalmente exagerado. |

## Implementación con loop completo

```python
shapes = [640, 1024, 2048, 3028]
url = 'http://localhost:8000/classify/'

results = []
for shape in shapes:
    img = np.zeros((3, shape, shape))            # construir UNA vez
    payload = {"image": img.tolist()}            # convertir a listas UNA vez
    times = []
    for trial in range(3):
        t_i = time.time()
        pred = requests.post(url, json=payload)
        elapsed = time.time() - t_i
        times.append(elapsed)
    avg = sum(times) / len(times)
    payload_mb = len(json.dumps(payload)) / 1e6
    results.append((shape, avg, times, payload_mb))
    print(f"shape={shape:>5}  avg={avg:6.2f}s   "
          f"trials={[f'{t:.2f}' for t in times]}   json≈{payload_mb:.1f} MB")
```

**Decisiones clave** del código:

1. `img = np.zeros(...)` **fuera del loop de trials**: la creación y el `.tolist()` son costosos (hasta varios segundos a 3028). Si los pones dentro estás midiendo construcción + serialización + transporte. Construyéndolo una vez por shape, los 3 trials reusan el mismo payload y miden solo `dumps` + transporte + server side.
2. `requests.post(url, json=payload)` dentro del loop: `requests` hace `json.dumps` cada vez. Esto es lo que queremos medir porque pasa en producción real.
3. `len(json.dumps(payload))`: el tamaño del payload serializado, útil para correlacionar latencia con bytes.

## Resultados medidos

```
shape=  640  avg=  1.74s   trials=['0.54', '2.97', '1.70']   json≈6.1 MB
shape= 1024  avg=  1.34s   trials=['1.38', '1.34', '1.31']   json≈15.7 MB
shape= 2048  avg=  7.95s   trials=['7.10', '5.41', '11.34']   json≈62.9 MB
shape= 3028  avg= 14.07s   trials=['15.78', '13.30', '13.13']   json≈137.5 MB
```

Tabla consolidada:

| Shape | Píxeles | JSON (MB) | Avg (s) | Mediana | Throughput (MB/s) | µs/píxel |
|-------|---------|-----------|---------|---------|-------------------|----------|
| 640   | 1.23 M  | 6.1       | 1.74    | 1.70    | 3.5               | 1.38     |
| 1024  | 3.15 M  | 15.7      | 1.34    | 1.34    | 11.7              | 0.43     |
| 2048  | 12.58 M | 62.9      | 7.95    | 7.10    | 7.9               | 0.56     |
| 3028  | 27.51 M | 137.5     | 14.07   | 13.30   | 9.8               | 0.48     |

## Lectura cuantitativa

### El escalado es aproximadamente lineal en el número de píxeles

Comparando medianas para evitar el outlier de 640:

| Salto | Píxeles ×N | Tiempo ×N | Relación |
|-------|-----------|-----------|----------|
| 1024 → 2048 | ×4.00 | ×5.30 | Ligeramente **super**lineal |
| 2048 → 3028 | ×2.19 | ×1.87 | Ligeramente **sub**lineal |

Ambas razones están en el rango 0.85×-1.32× respecto a "lineal puro en píxeles" — **lineal con ruido sistémico**. Confirma que **doblar el lado de la imagen cuadruplica la latencia** — un escalado O(n²) sobre la dimensión lineal, el peor enemigo de cualquier servicio de visión en producción.

### La fila de 640 es anómala

`avg = 1.74s` dominada por el outlier `2.97s`. Si descartas ese trial y promedias `0.54 + 1.70`, queda **1.12 s**, que encaja mejor con la curva. Lección operacional: el 640 es **el shape donde más afectan los warm-up effects** porque su payload chico no domina sobre el ruido de GC/scheduler. Cuanto más grande el payload, más estable la medición.

### Throughput efectivo ~8-12 MB/s una vez "caliente"

Muy lento comparado con loopback teórico (los puentes locales en Linux mueven ~10 GB/s). El cuello de botella no es la red sino:

- **Serialización JSON en cliente** (`json.dumps` de listas anidadas de millones de elementos).
- **Deserialización JSON en server** (peor: el parser JSON de Python es lento para listas anidadas grandes).
- **`np.array(...).transpose(...)`** en server reconstruyendo el array desde listas Python.

## Aritmética del costo en producción

Si Space Z usa este patrón con cámaras 1080p (~2 MP, equivalente a ~1450 px de lado):

- Imagen cruda en JSON: ~25-30 MB.
- Latencia estimada por frame: **3-5 s solo por transporte**.
- A 1 frame/s por cámara × 2 cámaras = **2 req/s sostenidos**, pero cada uno tarda 5 s.

**Conclusión**: el sistema **se atasca por completo**. La cola crece sin límite, el monitoreo reporta lecturas viejas, y la T4 está al 0% de utilización. **El cuello de botella nunca fue la GPU.** Esto es exactamente lo que [Sculley et al. (2015)](/papers/hidden-technical-debt-sculley-2015) llama "**Glue Code Anti-pattern**": el modelo es la cajita chica, lo que la rodea es el verdadero problema.

## Respuesta a la pregunta 1

**Sí, existe una diferencia muy marcada de tiempo que escala aproximadamente lineal con el número de píxeles** (es decir, O(n²) sobre la dimensión lineal de la imagen).

**Aunque el enunciado lo llama "tiempo de inferencia", la diferencia NO proviene del modelo**: el `predict` del Runnable es un `print("prediccion")` sin cómputo. La diferencia viene íntegramente del **overhead de transporte**:

1. **Serialización JSON en cliente**: `np.zeros(...).tolist()` + `json.dumps` recorre cada uno de los millones de píxeles y los convierte a texto ASCII. Para 3028×3028 son ~27 M valores → ~140 MB de JSON.
2. **Transferencia HTTP**: aunque sea localhost, mover 140 MB consume tiempo proporcional al tamaño.
3. **Deserialización JSON en server**: el parser JSON de Python reconstruye la lista anidada; luego `np.array(...).transpose(...)` la convierte a tensor numpy.
4. **IPC al runner**: pickle del array y envío por Unix socket, también escala con tamaño.

**Throughput efectivo observado: ~8-12 MB/s.** Si el bottleneck fuera red real, deberíamos ver ~1 GB/s sobre loopback. Esos ~10 MB/s confirman que el costo dominante es **CPU bound de serialización/deserialización JSON**, no transporte de red.

**Implicancia MLOps**: en sistemas de visión, el cuello de botella en latencia **casi nunca es la GPU**; es preprocessing, serialización y network I/O en los bordes del sistema ([Paleyes 2022](/papers/challenges-deploying-ml-paleyes-2022); [Sculley 2015](/papers/hidden-technical-debt-sculley-2015)). Esto motiva directamente las optimizaciones de la [Parte 5](../compresion-jpeg): comprimir la imagen en memoria con JPEG reduce el payload 10-30× sin pérdida visible, atacando la causa raíz.

## Siguiente

{{< cards >}}
  {{< card link="../concurrencia-multiprocessing" title="Parte 4 - Concurrencia" subtitle="Múltiples clientes en paralelo → throughput plano = saturación temprana" icon="academic-cap" >}}
{{< /cards >}}
