---
title: "Instalación y stack"
weight: 1
---

> **Celdas 1-8 del notebook.** Instalar la pila ABCNet → AdelaiDet → Detectron2 sobre un Colab moderno, con todos los parches de compatibilidad que eso exige.

## La pila de tres capas

ABCNet no es un paquete suelto: vive sobre dos capas de software.

```
┌─────────────────────────────────────────────┐
│  ABCNet (BAText)   ← el modelo (CVPR 2020)    │  detecta + lee texto curvo
├─────────────────────────────────────────────┤
│  AdelaiDet         ← toolbox U. de Adelaide   │  cabezas de instancia/texto
├─────────────────────────────────────────────┤
│  Detectron2        ← framework de FAIR        │  motor de detección genérico
├─────────────────────────────────────────────┤
│  PyTorch           ← backend de tensores      │
└─────────────────────────────────────────────┘
```

- **Detectron2** (Facebook AI Research) es el motor genérico: backbones (ResNet, FPN), gestión de proposals, ROI heads, data loaders, sistema de configuración `.yaml`. No sabe de "texto", sabe de **instancias**.
- **AdelaiDet** (grupo de Chunhua Shen) aporta los modelos específicos —FCOS, BlendMask, SOLO y **BAText/ABCNet**—. En el código se importa como `adet`.
- **ABCNet** es el modelo concreto; dentro de AdelaiDet el proyecto se llama **BAText**, por eso los configs están en `configs/BAText/`.

La ventaja de esta separación es que ABCNet hereda backbone y data loader sin reimplementarlos. La desventaja —y el motivo de que las primeras 8 celdas sean pura cirugía— es que hereda las **dependencias rígidas de versiones** de Detectron2, que incluye extensiones C++/CUDA compiladas.

## Detectron2 desde un commit fijo (celda 2)

```python
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@9eb4831f742ae6a13b8edb61d07b619392fb6543'
```

La sintaxis `git+https://…@<sha>` instala desde un **commit-hash exacto**, no desde el último release. ¿Por qué clavar el commit? Detectron2 incluye operaciones en `.cu`/`.cpp` (NMS, ROIAlign, deformable conv) que se compilan contra la versión de PyTorch, la de CUDA y la ABI de C++. AdelaiDet fue escrito contra un Detectron2 concreto de ~2020-2021; un Detectron2 más nuevo tendría APIs cambiadas que romperían los imports de `adet`. El commit garantiza que las APIs que AdelaiDet espera existan tal cual.

> Este `pip install` **compila desde fuente** (no baja un wheel): tarda varios minutos en Colab y escupe muchos warnings de `nvcc`/`gcc`. Lo único que importa es que termine sin `error:`.

## AdelaiDet + parches quirúrgicos (celda 4)

```python
!git clone https://github.com/aim-uofa/AdelaiDet.git
!cd AdelaiDet && git fetch origin pull/518/head:fix && git checkout fix
!sed -i '4d' AdelaiDet/adet/layers/csrc/ml_nms/ml_nms.cu
!sed -i 's/d2_postprocesss(results/d2_postprocesss(results.to(results.pred_boxes.device)/g' AdelaiDet/adet/modeling/one_stage_detector.py
!sed -i 's/type()/scalar_type()/g' AdelaiDet/adet/layers/csrc/BezierAlign/BezierAlign_cpu.cpp
!touch AdelaiDet/adet/modeling/roi_heads/__init__.py
!touch AdelaiDet/adet/data/datasets/__init__.py
```

Cada línea arregla un bug concreto de incompatibilidad de versiones:

| Comando | Qué arregla |
|---|---|
| `git fetch origin pull/518/head:fix` | Baja el **PR #518** ("Fix/pytorch issue", abierto el 14-feb-2022 por `an99990`, **nunca mergeado**) y lo usa como rama `fix`. Es un fix comunitario de compatibilidad con PyTorch nuevo que los maintainers no aceptaron oficialmente. |
| `sed -i '4d' .../ml_nms.cu` | Borra la línea 4: un `#include` de la antigua librería **THC** (`THC/THC.h`), que PyTorch **eliminó por completo** a partir de la 1.11. Sus utilidades viven ahora en `c10/cuda/`. |
| `.to(results.pred_boxes.device)` | Bug clásico **CPU vs GPU**: en versiones nuevas los resultados pueden quedar en un device distinto a las cajas, y operar entre devices distintos lanza `RuntimeError`. El parche los fuerza al mismo device. |
| `type()` → `scalar_type()` | En `BezierAlign_cpu.cpp` (¡la operación estrella de ABCNet!). PyTorch deprecó `Tensor.type()` (devolvía `DeprecatedTypeProperties`) en favor de `scalar_type()`. Sin el cambio, la extensión C++ no compila. |
| `touch .../__init__.py` (×2) | Crea dos `__init__.py` vacíos que faltaban; sin ellos Python no trata esos directorios como **paquetes importables** (`ModuleNotFoundError`). |

> El parche más relevante para entender el modelo es el de **BezierAlign**: esa es la operación custom que rectifica el texto curvo (alinea la región Bézier a un rectángulo antes de reconocerlo). Que exista una versión `_cpu.cpp` y otra `.cu` significa que la rectificación puede correr sin GPU (lentísimo, pero posible).

## Compilar AdelaiDet (celda 6)

```python
!pip install ./AdelaiDet
```

El `./` (ruta local, no nombre de PyPI) hace que pip lea el `setup.py` y **compile las extensiones C++/CUDA**: `BezierAlign`, `ml_nms`, `DefROIAlign`. Aquí es donde "se ejecutan" los `sed` de la celda 4. Si los parches no se aplicaron, esta celda falla con `error: 'type' is not a member` o `fatal error: THC/THC.h`. Si todo va bien, termina con `Successfully installed adet`.

## Pillow 9.4.0 + reiniciar la sesión (celdas 7-8)

```python
!pip install Pillow==9.4.0
```

Pillow 10 **eliminó** constantes que Detectron2 (2021) todavía usa (`Image.LINEAR`, `Image.CUBIC`, la reorganización bajo `Image.Resampling`). Clavar **9.4.0** restaura la API vieja y evita `AttributeError`.

> ⚠️ **Hay que reiniciar la sesión** después de esta celda, y es no negociable. Colab arranca con Pillow ya importado en memoria; `pip install` cambia los archivos en disco, pero el módulo cargado en RAM sigue siendo el viejo (Python cachea en `sys.modules`). Reiniciar (`Runtime → Restart session`) limpia el caché y fuerza la reimportación. **El reinicio NO borra los archivos del disco** (Detectron2, AdelaiDet compilado, datasets siguen), así que tras reiniciar se retoma desde la descarga del dataset sin recompilar nada.

## La lección transversal

Reproducir un modelo de visión de 2020 en 2026 es **arqueología de dependencias**, no `pip install` limpio. El stack de extensiones C++/CUDA es frágil por diseño: cada release de PyTorch limpia su API C++ (en 1.11 incluso dejaron de incluir transitivamente todos los operadores ATen), y cualquier `#include` o método deprecado rompe la compilación. Contrasta con un lab basado en HuggingFace Transformers (Python puro sobre API estable, todo `from_pretrained`): **mientras más cerca del metal (CUDA/C++), más se paga en mantenimiento**.

---

**Siguiente:** [demo end-to-end](demo-abcnet)
