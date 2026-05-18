---
title: "Instalación de las 3 librerías de pose"
weight: 10
---

El lab empieza instalando **tres librerías de pose con tres patrones radicalmente distintos de friction**. La fricción de instalación es **inversamente proporcional a la modernidad y mantenimiento** del modelo.

## Pre-flight: Colab versión 2025.10

```text
❗❗❗ IMPORTANTE ❗❗❗
Este laboratorio está pensado para ejecutarse en Colab versión "2025.10"
Para cambiar la versión, puede ir a "Entorno de Ejecución" > "Cambiar tipo de entorno de ejecución"
```

OpenPifPaf 0.10.1 es **del año 2019**. En 2026, esa versión depende de:

- `torch >= 1.0, < 1.7`
- `numpy < 1.20`
- `Pillow < 8`

Pero Colab 2026 default tiene `torch 2.x`, `numpy 2.x`, `Pillow 11.x`. Sin fijar la versión 2025.10 del runtime, **la instalación falla** con errores de dependencias incompatibles.

Esto es **higiene de reproducibilidad industrial** aplicada a un notebook educativo: equivalente a fijar `FROM golang:1.21.3-alpine` en lugar de `FROM golang:latest` en un Dockerfile.

## Patrón 1 — MediaPipe (`pip install` limpio)

```python
!pip install mediapipe==0.10.13
```

Una sola línea. ~60 segundos. Instala wheel precompilado de Google con bindings C++ + modelos `.tflite` embebidos. Sin requerir GPU, descarga ni configuración adicional.

**Por qué `0.10.13` exactamente**:

- MediaPipe pasó por una **migración de API en 2024**. Versiones `>= 0.11` mueven todo a `mediapipe.tasks.vision` (la nueva API "Tasks").
- El lab usa la API legacy `mediapipe.solutions.pose` (compatible con tutoriales antiguos).
- `0.10.13` es de enero 2024, la **última estable** con esa API.

Warning aceptable: `protobuf` se downgrade silenciosamente de 5.x a 4.25.x. No reinicies el runtime aunque el mensaje lo sugiera.

## Patrón 2 — OpenPifPaf (`pip install` con descarga lazy)

```python
!pip install openpifpaf==0.10.1
```

Una sola línea. ~90 segundos. Pero **los pesos del modelo NO se descargan aquí** — vienen al primer uso de `openpifpaf.network.factory(checkpoint='resnet101')` (~250 MB desde mirror VITA-EPFL).

### Por qué `0.10.1` (de agosto 2019) y no la 0.13.x moderna

Tres razones:

1. **API legacy didácticamente clara**: en 0.10.1, para inferir tienes que construir manualmente:
   ```python
   model_in_the_cpu, _ = openpifpaf.network.factory(checkpoint='resnet101')
   net = model_in_the_cpu.cuda()
   decode = openpifpaf.decoder.factory_decode(net, seed_threshold=0.5)
   processor = openpifpaf.decoder.Processor(net, decode, instance_threshold=0.2, keypoint_threshold=0.3)
   ```
   Cada objeto corresponde a un concepto del paper [PifPaf](/papers/pifpaf-kreiss-2019). La API moderna colapsa todo a `openpifpaf.Predictor(...)` y oculta los componentes — **pedagógicamente pobre**.

2. **Compatibilidad con pytorch-openpose**: el fork PyTorch de OpenPose (siguiente patrón) requiere `torch < 1.7`. OpenPifPaf 0.10.1 está fijada al mismo rango. Esto **cascadea** todo el stack a versiones viejas.

3. **Reproducibilidad ante drift**: el lab fue escrito hace años. Repinear cada año a versiones nuevas requeriría re-validar toda la API. Fijar `0.10.1` + Colab `2025.10` congela el comportamiento conocido.

## Patrón 3 — OpenPose (clone manual + wget de pesos)

```python
%%bash
git clone https://github.com/Hzzone/pytorch-openpose.git
wget -q https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABaYNMvvNVFRWqyDXl7KQUxa/body_pose_model.pth
mv body_pose_model.pth pytorch-openpose/model/body_pose_model.pth
```

**Tres comandos shell** (no Python). `~3 minutos`. Y aún así no es "instalación" — solo descarga del código + pesos a un directorio local.

### Por qué este patrón existe

El repo CMU original de OpenPose:

- Está en **Caffe** (framework muerto desde 2018).
- Requiere compilación C++ con dependencias específicas (CUDA viejo, cuDNN, OpenCV con flags raros).
- **No corre en Colab** sin docker o pasar por compilación de 30 minutos.
- Tiene licencia restrictiva para uso comercial.

El fork `Hzzone/pytorch-openpose` resuelve esto reescribiendo la red en PyTorch. **Es la única vía práctica** para usar OpenPose en un Colab educativo en 2026.

### El detalle frágil: pesos en Dropbox personal

Los `body_pose_model.pth` (~209 MB) **viven en el Dropbox personal de Hzzone**. Esto es típico de research code:

- Si Hzzone elimina el archivo → roto.
- Si Dropbox cambia política de links públicos → roto.
- Si rate-limita por descargas → roto.

En 2026 el link **podría estar caído**. Mitigaciones: mirror en Google Drive del curso, o re-subir desde otra fuente.

## El meta-mensaje pedagógico

| Modelo | Comando | Tiempo | Filosofía |
|---|---|---|---|
| OpenPifPaf | `pip install openpifpaf==0.10.1` | 90s | Research code maduro empaquetado |
| MediaPipe | `pip install mediapipe==0.10.13` | 60s | Producto consumer-grade |
| OpenPose | `git clone + wget + mv` | 3min | Research code académico arqueológico |

**Vas a encontrar mucho más patrón 3 y mucho menos patrón 2** en tu carrera. Aprender a tolerar el patrón 3 sin frustrarte **es una habilidad de supervivencia** en visión por computador y ML aplicado en general.

El profesor te lo dice diplomáticamente en celda 15:

> *"OpenPose fue desarrollado por académicos con objetivos de investigación, más que para gente que desarrollara aplicaciones. Si bien su código es público, eso no significa que sea sencillo de usar, ya que ese nunca fue su foco."*

Esa frase es **respetuosa pero precisa** — atribuye el problema a una decisión de propósito, no a un defecto.

## Cross-links

{{< cards >}}
  {{< card link="../demos-tres-librerias" title="Demos sobre la misma imagen" subtitle="Inferencia con los 3 modelos recién instalados" icon="academic-cap" >}}
  {{< card link="/papers/openpose-cao-2017" title="OpenPose (Cao 2017)" subtitle="Paper original CMU 2017" icon="document-text" >}}
  {{< card link="/papers/pifpaf-kreiss-2019" title="PifPaf (Kreiss 2019)" subtitle="Paper original VITA-EPFL 2019" icon="document-text" >}}
  {{< card link="/papers/blazepose-bazarevsky-2020" title="BlazePose (Bazarevsky 2020)" subtitle="Paper original Google 2020" icon="document-text" >}}
{{< /cards >}}
