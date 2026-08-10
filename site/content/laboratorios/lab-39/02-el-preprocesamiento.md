---
title: "El preprocesamiento y sus once puntos"
weight: 2
---

Corregida [la fuga de folds](/laboratorios/lab-39/01-la-fuga-de-folds), queda un déficit contra el paper que no desaparece:

| Modelo | Split corregido | Dai et al. (2016) | **Diferencia** |
|---|---|---|---|
| M3 | 45.13 % | 56.12 % | **−10.99** |
| M5 | 52.12 % | 63.42 % | **−11.30** |

Que M3 pierda 10.99 puntos y M5 pierda 11.30 —con capacidades, profundidades y regímenes de sobreajuste muy distintos— apunta a un **sesgo aditivo común**, algo que degrada los datos de entrada por igual para cualquier arquitectura que se ponga encima. Y está todo en el mismo lugar: los cinco renglones del `__getitem__`.

```python
audio, rate = torchaudio.load(audio_path, normalize=True, backend="soundfile")
audio = audio.mean(0, keepdim=True)      # a monoaural
c, n = audio.shape
zero_need = 160000 - n
audio_new = F.pad(audio, (zero_need //2, zero_need //2), 'constant', 0)
audio_new = audio_new[:,::5]             # 1 de cada 5 muestras
```

Hay cuatro problemas ahí, y ninguno es cosmético.

## 1. Los 160 000 están en muestras, no en segundos

`zero_need = 160000 - n` fija el largo del tensor en **muestras**. Eso equivale a 3.63 segundos únicamente si el archivo está a 44 100 Hz.

Pero UrbanSound8K conserva la frecuencia de muestreo original de cada grabación de Freesound. Midiendo sobre una muestra de 874 de los 8732 archivos:

| Rate | Archivos | % | 160 000 muestras equivalen a | Rate efectivo tras `[::5]` |
|---|---|---|---|---|
| 44 100 Hz | 538 | 61.6 % | 3.63 s | 8 820 Hz |
| 48 000 Hz | 252 | 28.8 % | 3.33 s | 9 600 Hz |
| **96 000 Hz** | 59 | **6.8 %** | **1.67 s** | **19 200 Hz** |
| 24 000 Hz | 8 | 0.9 % | 6.67 s | 4 800 Hz |
| 16 000 Hz | 4 | 0.5 % | 10.00 s | 3 200 Hz |
| 11 025 Hz | 4 | 0.5 % | 14.51 s | 2 205 Hz |
| 22 050 Hz | 3 | 0.3 % | 7.26 s | 4 410 Hz |
| **192 000 Hz** | 2 | 0.2 % | **0.83 s** | **38 400 Hz** |
| 8 000 Hz | 2 | 0.2 % | **20.00 s** | **1 600 Hz** |
| 32 000 Hz | 1 | 0.1 % | 5.00 s | 6 400 Hz |

**Diez frecuencias distintas, decimadas todas por el mismo factor 5.** El resultado es que el mismo sonido llega a la red a escalas temporales incompatibles: un clip grabado a 8 kHz queda a 1 600 Hz efectivos, y uno a 192 kHz queda a 38 400 Hz. Un factor de **24×**.

Eso rompe directamente el argumento de diseño del paper. Dai et al. eligen el kernel de 80 de la primera capa para cubrir ~10 ms y comportarse como banco de filtros pasabanda, y para eso **remuestrean todo el dataset a 8 kHz**. Sin ese paso, el kernel de 80 cubre:

| Rate efectivo | Lo que abarca el kernel de 80 |
|---|---|
| 38 400 Hz | **2.08 ms** |
| 19 200 Hz | 4.17 ms |
| 9 600 Hz | 8.33 ms |
| 8 820 Hz | 9.07 ms |
| 1 600 Hz | **50.00 ms** |

La capa que debía aprender un banco de filtros a una escala de frecuencia fija está viendo ventanas que van de 2 a 50 milisegundos según con qué micrófono se grabó el clip.

## 2. El padding negativo recorta a ciegas

Cuando `n > 160000`, `zero_need` es negativo y `F.pad` con valores negativos **recorta** en lugar de rellenar. Como la duración media es de 3.63 s y el dataset está lleno de clips de 4 s, esto afecta al **85.0 % de los archivos**:

| Rate | Clip de 4 s | Se conserva | Se descarta por lado |
|---|---|---|---|
| 44 100 Hz | 176 400 muestras | 90.7 % | 186 ms |
| 48 000 Hz | 192 000 muestras | 83.3 % | 333 ms |
| **96 000 Hz** | 384 000 muestras | **41.7 %** | **1.17 s** |
| **192 000 Hz** | 768 000 muestras | **20.8 %** | **1.58 s** |

Los ~590 archivos a 96 kHz **pierden más de la mitad de su contenido**, y el recorte es simétrico y ciego: se corta por los dos extremos sin mirar dónde está el evento. Para clases de transitorio breve y localizado —`gun_shot`, `car_horn`, `dog_bark`— eso puede eliminar exactamente el sonido a clasificar y dejar ambiente etiquetado como disparo.

## 3. `[:, ::5]` es decimación sin filtro antialiasing

Aquí el lab conecta con la [clase 35](/clases/clase-35) y con el [fundamento de digitalización](/fundamentos/digitalizacion-de-audio). Tomar 1 de cada 5 muestras baja la frecuencia de muestreo, y con ella el límite de Nyquist: para un archivo de 44 100 Hz, pasa de 22 050 Hz a **4 410 Hz**.

Todo el contenido por encima de ese límite **no desaparece: se pliega**. Aparece reflejado dentro de la banda útil como energía espuria superpuesta a la señal real. Un remuestreo correcto —`torchaudio.functional.resample`, o `resampy`, que el propio notebook instala para la Parte 2— aplica un filtro pasabajos *antes* de decimar, precisamente para evitarlo.

El aliasing es determinista, así que la red puede aprender a convivir con él. Pero es ruido estructurado que se suma a la tarea, y castiga más a las clases con energía significativa sobre 4.4 kHz: `drilling`, `jack_hammer`, `siren`.

## 4. No hay estandarización

El paper dice que el audio se lleva a **media 0 y varianza 1**. El notebook usa `normalize=True` en `torchaudio.load`, que solo convierte enteros de 16 bits a punto flotante en $[-1, 1]$. Son cosas distintas: la primera es una normalización estadística por muestra, la segunda un cambio de escala de formato. El primer `BatchNorm` compensa parte, pero recibe entradas con energías muy dispares entre grabaciones.

{{< concept-alert type="clave" >}}
**Los cuatro problemas comparten una raíz: el código trata al dataset como si fuera homogéneo.** Un largo fijo en muestras, un factor de decimación fijo y un recorte fijo tienen sentido cuando todos los archivos comparten frecuencia de muestreo y duración. UrbanSound8K no cumple ninguna de las dos condiciones, y el preprocesamiento nunca lo verifica.

Es también la respuesta profunda a la **Actividad 1** del práctico: la opción *"resamplear el audio a otra frecuencia de muestreo"* no es necesaria *para calcular MFCC* —los MFCC se computan a cualquier tasa—, pero sí es necesaria **ya**, para lo que el notebook está haciendo ahora mismo con la onda cruda.
{{< /concept-alert >}}

## El costo oculto: 22 milisegundos por archivo

Un quinto problema no afecta la exactitud sino el tiempo, y vale la pena porque cambia lo que es posible experimentar en una sesión.

```python
labels = self.audio_file.loc[self.audio_file.slice_file_name == audio_name].iloc[0,-2]
```

Esa línea recorre las 8732 filas del `DataFrame` construyendo una máscara booleana, **en cada `__getitem__`**. A 20 épocas son unas 175 000 búsquedas lineales. Y la carga del audio es peor: en TorchAudio 2.11, `torchaudio.load` ignora el parámetro `backend="soundfile"` y delega en `torchcodec`, que abre un decodificador FFmpeg por archivo.

Medido:

```
load  : 22.1 ms/archivo
label :  1.1 ms/archivo   (5 % del costo)
=> 20 épocas train+test: ~1.2 h solo de carga
```

Con `num_workers = 0` —el valor que fija el notebook— todo eso ocurre en el hilo principal, **en serie con el cómputo**, con la GPU detenida esperando. Sobre una A100, M3 usaría el acelerador alrededor del 4 % del tiempo.

La solución no toca el experimento: los 8732 tensores decimados ocupan `8732 × 32000 × 4 bytes = 1.12 GB` y caben holgadamente en RAM. Se paga la decodificación una sola vez y las 20 épocas corren a velocidad de GPU.

```python
def precargar(ds, nw=8, bs=64):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    A, L = [], []
    for a, l in tqdm(dl, desc='precargando'):
        A.append(a.contiguous()); L.append(l)
    return torch.utils.data.TensorDataset(torch.cat(A), torch.cat(L).long())
```

El `.contiguous()` no es opcional: `[:, ::5]` devuelve una vista con stride 5 sobre el tensor de 160 000 muestras, así que sin copiar se mantendría vivo un buffer cinco veces mayor.

Dos detalles de reproducibilidad que sí importan: la precarga debe hacerse **antes** de la celda que fija las semillas, porque un `DataLoader` con `num_workers > 0` consume un número del generador global para sembrar sus procesos; y como el `Dataset` no tiene ninguna fuente de aleatoriedad —no hay data augmentation— los tensores cacheados son idénticos a los que produciría el original.

En la Parte 2 el ahorro es mayor todavía. El cálculo de los parches log-mel con `resampy` toma 31 minutos una vez, y a cambio cada época de fine-tuning pasa a costar **18 segundos** en lugar de la hora y media que el notebook estima para tres.

---

**Siguiente:** [La familia M y el campo receptivo](/laboratorios/lab-39/03-familia-m-y-campo-receptivo) — qué ve realmente cada arquitectura, y por qué el *global average pooling* hunde a las clases de evento breve.
