---
title: "Parte 5 — Compresión JPEG en memoria"
weight: 5
---

> **Celdas 18-23 del notebook.** El instrumento de mitigación: comprimir la imagen **en memoria** con `cv2.imencode` antes de enviarla. Ataca directamente la causa raíz identificada en las partes 3-4 (transporte JSON = bottleneck).

## La idea clave: encode/decode SIN tocar disco

El reflejo del estudiante promedio que aprende OpenCV es `cv2.imwrite('archivo.jpg', img)` → un archivo se materializa en disco, después lo lees con `open('archivo.jpg', 'rb').read()`. **Dos viajes a disco innecesarios**.

`cv2.imencode` hace exactamente lo mismo (encoder JPEG → bytes) pero **te devuelve los bytes directamente como ndarray uint8 1D**, sin tocar el sistema de archivos.

**Por qué importa en serving**:

1. **El disco es lento**. Aunque sea SSD, ~1-10 ms por write+read. En un endpoint a 100 req/s eso es muchísimo.
2. **Concurrencia con disco es caos**. Race conditions si dos procesos escriben `tmp.jpg`.
3. **Contenedores Docker/k8s** suelen tener `/tmp` con políticas restrictivas (tmpfs chico, read-only).
4. **Handles de filesystem son recursos compartidos**: descriptores limitados, locks, journaling.

Todo en memoria (un `bytes` en RAM) elimina los 4 problemas. Es la práctica idiomática para serving.

## Encode: `cv2.imencode`

```python
img = np.random.randint(low=0, high=255, size=(1024,1024,3))
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
result, encimg = cv2.imencode('.jpg', img, encode_param)
```

### Pieza 1 — La imagen ficticia

`np.random.randint(..., size=(1024,1024,3))` genera ruido uniforme aleatorio en convención HWC (height-width-channels), compatible con OpenCV (BGR).

> **Gotcha de dtype**: `np.random.randint` **sin argumento `dtype=`** devuelve `int64` (8 bytes/valor). Para una imagen real son datos absurdamente desperdiciados (el 87.5% de cada byte es padding). El correcto sería:
> ```python
> img = np.random.randint(low=0, high=256, size=(1024,1024,3), dtype=np.uint8)
> ```
> OpenCV hace coerción silenciosa internamente, pero esto **infla el "ratio de compresión"** que reporta el lab. Lo discutimos abajo.

> **Por qué ruido random en lugar de ceros o foto real**: ruido uniforme es el **peor caso** para JPEG. Una imagen de ceros se comprime ~300:1 (irrealmente bueno). Una foto natural comprime ~10-30:1. Random uniforme da 3:1 — el **piso** de lo que el codec puede ofrecer.

### Pieza 2 — Parámetros del encoder

```python
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
```

OpenCV usa lista plana de pares clave-valor (herencia C++). `cv2.IMWRITE_JPEG_QUALITY = 1` internamente; el `90` es el nivel de calidad.

**Qué controla quality**: el JPEG cuantiza los coeficientes de la transformada DCT por bloques de 8×8 píxeles. La calidad controla el **paso de cuantización**:

| Quality | Caracterización visual |
|---------|------------------------|
| 100 | Cuantización mínima → archivo grande, fidelidad casi perfecta (pero **NO lossless**) |
| 95 | Estándar "high quality" web. Diferencia con 100 imperceptible |
| 90 | Sweet spot ML: 0% pérdida observable de accuracy |
| 75 | Balance "subjetivo" típico, default de navegadores |
| 50 | Artefactos de bloque ("blocking") visibles en zonas planas |
| 30 | Aspecto granulado, ruido evidente |
| 10 | Irreconocible |

**Por qué 90 para ML**: hay trade-off entre **tamaño del payload** y **degradación del modelo**. Estudios sobre detección de objetos (referencia: Dziugaite et al. 2016 sobre JPG y adversarial images) muestran que modelos como YOLO/Faster R-CNN mantienen su mAP intacto hasta **quality 70-80**, y empiezan a degradarse recién bajo quality 30. Quality 90 da **~70% reducción de tamaño con 0% pérdida observable**. Es el sweet spot.

### Pieza 3 — `cv2.imencode('.jpg', img, encode_param)`

**Argumentos**:
- `'.jpg'`: la extensión determina el codec. OpenCV soporta `.jpg/.png/.bmp/.tiff/.webp/.pbm`. **El punto inicial es obligatorio.**
- `img`: matriz uint8 HWC (1, 3 o 4 canales).
- `encode_param`: pares clave-valor.

**Retornos**:
- `result`: booleano (True si encode exitoso). **El lab no lo verifica**; en producción real harías `assert result, "JPEG encode failed"`.
- `encimg`: **`np.ndarray` uint8 1D** con los bytes comprimidos. **NO es un objeto `bytes` de Python.**

**Cómo funciona internamente** (libjpeg-turbo):

1. BGR → YCbCr.
2. Submuestreo crominancias (4:2:0).
3. División en bloques 8×8.
4. DCT bidimensional → coeficientes en dominio frecuencia.
5. Cuantización con tabla ajustada por quality.
6. Run-length + Huffman → bytes finales.

En CPU moderno: ~5-20 ms para 1024×1024.

## Resultado de la celda 19

```python
print("img.shape  :", img.shape)         # (1024, 1024, 3)
print("img.dtype  :", img.dtype)         # int64    ← gotcha
print("img.nbytes :", img.nbytes)        # 25 165 824  bytes ≈ 25 MB (int64)
print("encode ok? :", result)            # True
print("encimg.shape:", encimg.shape)     # (939067,)
print("encimg.dtype:", encimg.dtype)     # uint8
print("encimg size:", encimg.nbytes)     # 939 067 bytes ≈ 0.94 MB
print("ratio      :", img.nbytes / encimg.nbytes)   # 26.8x  ← inflado
```

## El "26.8×" del lab es engañoso

| Magnitud | Valor reportado | Honesto |
|----------|-----------------|---------|
| `img.dtype` | **int64** | uint8 sería lo correcto |
| `img.nbytes` (declarado) | 25.17 MB | porque 1024×1024×3×**8 bytes** |
| `img.nbytes` real si fuera uint8 | — | **3.15 MB** (1024×1024×3×**1 byte**) |
| `encimg.nbytes` | **939 KB** | ✅ correcto |
| Ratio aparente | **26.8×** | inflado por dtype |
| Ratio real sobre uint8 | — | **~3.3×** |

**Esto es un error pedagógico del lab**. Un estudiante que no entienda la sutileza del dtype va a salir creyendo que "JPEG quality 90 reduce 27× sobre ruido random". Eso es **falso** — el 27× viene de **cuantizar de int64 a uint8** (factor 8×) más **comprimir uint8 con JPEG** (factor ~3.3×). El JPEG en sí solo hace la última parte.

## Decode: `cv2.imdecode`

```python
print(encimg.shape)                       # (939067,)
decimg = cv2.imdecode(encimg, 1)
print(decimg.shape)                       # (1024, 1024, 3)
```

- **`cv2.imdecode(buffer, flag)`**: la operación inversa.
  - `buffer`: `np.ndarray` uint8 1D **o** `bytes`/`bytearray`. OpenCV examina los **magic bytes** (`FF D8 FF` para JPEG, `89 50 4E 47` para PNG) y deduce el codec. No necesitas decirle la extensión.
  - `flag = 1` (= `cv2.IMREAD_COLOR`): decodificar en BGR 3 canales. Otros: `0 = GRAYSCALE`, `-1 = UNCHANGED`, `2 = ANYDEPTH`, `4 = ANYCOLOR`.
- **Retorna**: `ndarray` uint8 HWC en BGR.

**Lo que sí es igual** después del round-trip encode→decode: shape, dtype, orden BGR, apariencia visual (en foto real a q=90 indistinguible).

**Lo que cambió**: valores exactos de píxeles. JPEG es lossy. En ruido random la diferencia promedio es 5-15/255 por píxel; en foto real con q=90 es <2/255 — imperceptible.

## Curva quality / tamaño medida

```python
img_correct = img.astype(np.uint8)
print(f"raw uint8: {img_correct.nbytes} bytes ({img_correct.nbytes/1e6:.2f} MB)\n")
for q in [10, 30, 50, 70, 90, 95, 100]:
    _, e = cv2.imencode('.jpg', img_correct, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    ratio = img_correct.nbytes / e.nbytes
    print(f"q={q:>3}: {e.nbytes:>8} bytes ({e.nbytes/1e6:.2f} MB) ratio {ratio:.1f}x")
```

Resultado:

```
raw uint8: 3 145 728 bytes (3.15 MB)

q= 10:   115 483 bytes  (0.12 MB)  ratio 27.2x
q= 30:   292 800 bytes  (0.29 MB)  ratio 10.7x
q= 50:   429 467 bytes  (0.43 MB)  ratio 7.3x
q= 70:   580 060 bytes  (0.58 MB)  ratio 5.4x
q= 90:   939 067 bytes  (0.94 MB)  ratio 3.3x
q= 95: 1 228 575 bytes  (1.23 MB)  ratio 2.6x
q=100: 2 071 153 bytes  (2.07 MB)  ratio 1.5x
```

## Análisis cuantitativo de la curva

### q=100 NO es lossless

Aunque digamos "máxima calidad", JPEG q=100 **todavía pierde información**. La cuantización tiene un piso mínimo. JPEG **no tiene modo lossless real** — para eso usas PNG o WebP-lossless.

El **1.5× reducción** a q=100 viene íntegramente de:
1. **Huffman coding** (lossless) de los coeficientes DCT.
2. **Run-length encoding** de coeficientes cero.

Es la parte "lossless" de un codec lossy.

### La curva es exponencial decreciente, no lineal

Doblar la calidad nominal **no dobla el tamaño**. Los saltos:

- q=10 → 30: +154% (115k → 293k)
- q=30 → 50: +47%
- q=50 → 70: +35%
- q=70 → 90: +62%
- q=90 → 95: +31%
- q=95 → 100: +69%

La **tabla de cuantización del JPEG** no escala lineal con quality. En q=90 los pasos ya son cercanos al mínimo (clip a 1). De q=90 a q=100 los ~10-20% más bytes guardados por bloque se acumulan en una **subida pronunciada al final**.

### q=10 con ruido random aún logra 27×

A quality 10, la cuantización es brutal — divide los coeficientes DCT por números enormes. La mayoría (incluyendo todos los de alta frecuencia donde está el "ruido") **caen a cero**. RLE codifica largas tiradas de ceros en un solo símbolo. El JPEG resultante tiene casi solo el componente DC (promedio de cada bloque 8×8) y poquito más.

Visualmente, **q=10 reduce tu imagen a un mosaico de bloques 8×8 con un color promedio**. Inservible para detección fina, pero el codec llega ahí.

## Predicción para una foto real

| Quality | Random (medido) | Foto parking (estimado) |
|---------|-----------------|--------------------------|
| 10  | 27× | **80-120×** |
| 50  | 7.3× | **25-40×** |
| 70  | 5.4× | **15-25×** |
| 90  | 3.3× | **10-15×** |
| 95  | 2.6× | **7-10×** |
| 100 | 1.5× | **3-5×** |

## El sweet spot para Space Z

Para detección de autos:

- **YOLOv5/v8/v10**: mAP intacta hasta **quality 70**.
- **Faster R-CNN / DETR**: aún más robustos, **hasta quality 50**.
- **OCR de patentes**: los más sensibles — necesitan **quality 85+**.

Para una pipeline de "contar tiempo de espera de vehículos", **quality 70** es óptimo:
- Reducción ~15× en wire vs imagen cruda.
- Cero pérdida de mAP en detector.
- **15× menos egress en cloud = 15× menos costo de bandwidth.**

Si además necesitas leer la patente, conviene **pipeline de dos pasos**: imagen completa a q=50 al detector + crop de la patente a q=90 al OCR. **Es un patrón real de MLOps en visión.**

## Visualización de la curva

```
Bytes (KB)
  2000│                                              ●  ← q=100 (1.5×)
      │                                       ●        ← q=95  (2.6×)
  1500│
  1000│                              ●                 ← q=90  (3.3×)
   750│                       ●                        ← q=70  (5.4×)
   500│                ●                               ← q=50  (7.3×)
   300│        ●                                       ← q=30  (10.7×)
   100│  ●                                             ← q=10  (27.2×)
      └──┴────┴───────┴───────┴───────┴───────┴───────┴───────►
        10   30      50      70      90      95      100   Quality
```

La curva tiene un **codo cerca de q=70-80**: por debajo, ganancia de tamaño masiva pero calidad cae rápido; por arriba, calidad sube poco pero tamaño crece exponencial. **Ese codo es donde quieres operar en producción.**

## Alternativas más modernas (no en el lab pero deberías conocer)

1. **WebP** (`cv2.imencode('.webp', ...)`): ~25% más eficiente que JPEG con calidad similar. Soporta lossless real. Soportado por todos los browsers modernos.
2. **AVIF**: ~50% más eficiente que JPEG. Estándar emergente.
3. **gRPC + Protocol Buffers**: bytes nativos, sin overhead HTTP+JSON.
4. **Para video continuo (caso Space Z): NO mandes frame por frame**. Manda un stream **H.264/H.265** y decode server-side. Ratio 1000:1 contra raw, no 15:1. Esto es lo que hacen sistemas serios de video analytics — RTSP/WebRTC + GStreamer + NVIDIA DeepStream.

## Siguiente

{{< cards >}}
  {{< card link="../cierre-integracion" title="Cierre - Respuesta final integradora" subtitle="Conectar diagnóstico (transporte = bottleneck) con remedio (JPEG) → speedup ~15-50×" icon="academic-cap" >}}
{{< /cards >}}
