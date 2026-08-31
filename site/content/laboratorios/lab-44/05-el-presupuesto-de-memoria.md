---
title: "El presupuesto de memoria"
weight: 5
math: true
---

El modelo tiene 17,3 mil millones de parámetros. En bf16 son 34,6 GB de pesos; el repositorio oficial los publica en fp32, o sea **72,1 GB**. La tarjeta gratuita de Colab tiene 15 GB de VRAM. Todo el segundo notebook existe para cerrar esa brecha.

## Lo que la GPU no puede hacer

La celda que abre el notebook parece trivial:

```python
!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
```

El `compute_cap` es lo que importa. Es la *compute capability* de CUDA, y determina **qué instrucciones existen en el silicio**:

| GPU | Arquitectura | compute_cap | bf16 nativo | FlashAttention 2 |
|---|---|---|:---:|:---:|
| **T4** | Turing (2018) | **7.5** | ❌ | ❌ |
| A100 | Ampere | 8.0 | ✅ | ✅ |
| L4 | Ada | 8.9 | ✅ | ✅ |

**La T4 no soporta bfloat16 en hardware** — bf16 requiere 8.0+. Y el modelo es nativo bf16. O sea que el problema no es solo que no quepa: es que **el formato numérico en que fue entrenado no se puede ejecutar en esa GPU**. Y ya que hay que convertirlo, se aprovecha para cuantizarlo.

## Las ausencias del `requirements.txt`

El archivo documenta cinco paquetes que **deliberadamente no instala**, y cada uno es una decisión razonada:

| Ausente | Razón declarada |
|---|---|
| `torch` | *"nunca lo reinstales en Colab. La imagen está construida contra un runtime CUDA específico"* |
| `flash-attn` | *"requiere sm_80+. Una T4 es sm_75"* |
| `sageattention` | sm_80+ (v2 necesita sm_89+) |
| `xformers` | *"el backend mem-efficient SDPA de PyTorch **ES** el kernel CUTLASS de xformers"* |
| `bitsandbytes` | *"funciona en sm_75, pero GGUF es el camino aquí"* |

El de `xformers` merece un párrafo. Durante años instalarlo fue el ritual obligatorio de todo pipeline de difusión, porque traía atención con memoria eficiente. Desde PyTorch 2.0, `scaled_dot_product_attention` **incorporó ese mismo kernel CUTLASS** en el core. En una T4 —donde no hay FlashAttention 2— el backend `mem_efficient` de SDPA *es* xformers. Instalarlo no agrega nada y sí agrega superficie de fallo.

Y `bitsandbytes` es honesto por lo que reconoce: NF4 funciona en Turing y era una alternativa legítima. La elección de GGUF no es técnica sino **logística** — los pesos ya están publicados cuantizados, y cuantizar 34,6 GB dentro de Colab sería imposible por RAM.

## Los niveles de cuantización

El repositorio `QuantStack/Wan2.2-Animate-14B-GGUF` publica once variantes:

| Cuantización | GiB | bits/parámetro | VRAM libre en L4 (22,0 GiB) |
|---|---:|---:|---:|
| `Q2_K` | 6,01 | 2,99 | 16,0 |
| **`Q3_K_M`** ← default del notebook | **8,04** | 3,99 | 13,96 |
| `Q4_K_S` | 9,86 | 4,90 | 12,14 |
| **`Q4_K_M`** | **10,71** | 5,32 | 11,29 |
| `Q5_K_M` | 12,11 | 6,01 | 9,89 |
| `Q6_K` | 13,60 | 6,76 | 8,40 |
| `Q8_0` | 17,43 | 8,66 | 4,57 |

Que los bits por parámetro salgan ~1 más que el número del nombre confirma el conteo de 17,3 B: GGUF guarda además una escala y un offset por bloque de 32 pesos, y deja las capas más sensibles a mayor precisión (eso es la `_K_M`, *K-quants medium*). `Q8_0` en `llama.cpp` es 8 bits más una escala fp16 cada 32 muestras:

$$8 + \frac{16}{32} = 8{,}5\ \text{bits/param teóricos} \qquad \text{medido: } \frac{18{,}72 \times 8}{17{,}3} = 8{,}66$$

La diferencia son los tensores que quedan sin cuantizar. Frente a los 72,1 GB publicados, `Q3_K_M` comprime **8,4×**.

Es tecnología nacida para correr LLMs en laptops, aplicada aquí a un transformer de difusión de video.

## El chequeo que rechaza lo que sí cabe

Con `Q4_K_M` a 832×480 y 77 frames, el `--dry-run` dice:

```
weights 10.72 GB + activations 2.09 GB + attention 1.25 GB = 15.26 GB -> DOES NOT FIT
```

Pero la etapa que efectivamente ejecuta dice otra cosa:

```
[denoise] predicted peak 16.40 GB against a 21.03 GB budget
```

{{< concept-alert type="clave" >}}
**Son dos modelos de costo distintos en el mismo programa.**

`run.py`, el orquestador, compara contra un presupuesto **fijo de 14,0 GiB** — calibrado para una T4 de 16 GB. La nota del preset `t4-quality` lo delata: *"necesita 14.15 GB contra un presupuesto de 14.0 GB, así que se predice OOM"*, exactamente el número que aparece en el dry-run.

`denoise.py`, la etapa, lee `torch.cuda.get_device_properties` y obtiene los **21,03 GiB** reales de la tarjeta.

O sea que `--force` no fuerza nada riesgoso: **corrige un supuesto obsoleto**. La etapa que hace el trabajo sabe perfectamente cuánta memoria tiene. Pico predicho 16,40 GiB, real **16,87 GiB** — la predicción se quedó corta un 2,9 %.
{{< /concept-alert >}}

El desglose completo del dry-run tampoco cuadra a primera vista: $10{,}72 + 2{,}09 + 1{,}25 = 14{,}06$, no 15,26. La diferencia de 1,20 GiB son dos rubros que el total incluye pero la línea no lista, y que la etapa sí detalla:

```
- VAE weights and pose-encode buffers add ~0.50 GB
- distillation LoRA adds ~0.75 GB; it cannot be merged into quantised weights, so it stays resident
- the overlap decode between segments peaks near 13.47 GB
```

## El LoRA que no se puede fusionar

```
[loader] LoRA via 'hook': 488 module(s) patched at strength 1.0, 43 skipped (e.g. blocks.0.norm3)
```

Con pesos densos, `peft` haría `merge_and_unload()`: $W' = W + BA$, y el adaptador desaparece sin costo en inferencia. **Con pesos GGUF no se puede** — habría que dequantizar, sumar y recuantizar cada tensor.

Así que se aplica por **hooks** en cada forward: **+0,75 GiB residentes** y overhead por llamada, sobre 488 módulos. Es un costo real de la cuantización que rara vez se menciona en las discusiones sobre GGUF. (Los 43 módulos saltados son `norm3`: las normalizaciones no llevan adaptadores.)

## Cómo escala el tiempo

Dos configuraciones medidas sobre la misma L4:

| | tokens | s/paso |
|---|---:|---:|
| 480×480, 33 frames | 9.000 | 11,1 |
| 832×464, 77 frames | 31.668 | 52,0 |

$$\frac{52{,}0}{11{,}1} = 4{,}685 \quad \text{con} \quad \frac{31.668}{9.000} = 3{,}519 \quad \Rightarrow \quad \text{exponente} = \frac{\ln 4{,}685}{\ln 3{,}519} = \mathbf{1{,}23}$$

Ni lineal (1,0) ni cuadrático (2,0). Con dos puntos se resuelve el modelo $T = aN + bN^2$:

$$a = 1{,}071\times10^{-3}\ \text{s/token} \qquad b = 1{,}802\times10^{-8}\ \text{s/token}^2$$

| | 9.000 tokens | 31.668 tokens |
|---|---:|---:|
| Lineal (MLPs, proyecciones) | 9,64 s (**87 %**) | 33,92 s (**65 %**) |
| Cuadrático (atención) | 1,46 s (13 %) | 18,07 s (35 %) |
| Total | 11,1 s ✓ | 52,0 s ✓ |

Y el punto de cruce donde la atención pasaría a dominar:

$$aN = bN^2 \quad \Rightarrow \quad N = \frac{a}{b} = \mathbf{59.400\ \text{tokens}}$$

Eso corresponde a unos 140 frames por segmento a 832×464 — bastante por encima de los 77 nativos del modelo. **En el rango en que se trabaja, el DiT está dominado por los MLPs, no por la atención.**

El mensaje de error del runner advierte que *"el costo es superlineal — la atención es cuadrática en el conteo de tokens"*. Es cierto asintóticamente y para el **tiempo**. Para la **memoria** no:

| | tokens | memoria de atención | GB/token |
|---|---:|---:|---:|
| draft | 9.000 | 0,34 GB | $3{,}78 \times 10^{-5}$ |
| quality | 21.840 | 0,83 GB | $3{,}80 \times 10^{-5}$ |

Perfectamente lineal, y así debe ser: con SDPA *memory-efficient* la matriz $N \times N$ **nunca se materializa** — se procesa por bloques. La memoria es $O(N)$; lo que sigue siendo $O(N^2)$ es el tiempo. Es, de paso, la razón por la que el `requirements.txt` podía darse el lujo de no instalar `xformers`.

## Una consecuencia contraintuitiva

Como el término cuadrático crece más rápido que lo que se ahorra en costuras, **segmentos más chicos salen más baratos en total**:

| | `t4-max` (77f/segmento) | 49f/segmento |
|---|---:|---:|
| Segmentos para 128 frames | 2 | 3 |
| Tokens por segmento | 31.668 | 21.112 |
| s/paso | 52,0 | ~30,6 |
| **Denoise total** | 624 s | **~551 s** |

Un 12 % menos con tres segmentos que con dos. La contrapartida es una costura más entre segmentos, con el *drift* que eso implica.

## El cuello de botella que no es la VRAM

El `colab_setup.py` documenta algo que contradice la presentación del notebook:

```python
# On free Colab this is ~12.7 GB and is the real ceiling -- tighter in
# practice than the 15 GB of VRAM, because it caps what can be staged
# or offloaded to the host.
```

Todo el planteamiento habla de "15 GB de VRAM contra 34,6 GB de modelo". El autor dice que **el techo verdadero son los 12,7 GB de RAM del sistema**, y el argumento es sólido: la estrategia entera se basa en *offloading* —mantener en RAM lo que no está en uso— y si la RAM es más chica que la VRAM, la RAM limita cuánto se puede offloadear.

Por eso las etapas corren en procesos separados: no es solo para liberar VRAM, es para liberar **RAM**. En una sesión con 51 GB ese límite desaparece y el techo vuelve a ser la VRAM.

## El warning que se contradice a sí mismo

Ejecutando en una L4:

```
[loader] warning: this device is sm_89 ... bf16 and FlashAttention are both available to you
[loader] dtype policy: float16 (bf16 is unusable on sm_89)          <- falso
[loader] sdpa backends: mem_efficient=on flash=off cudnn=off math=off
```

Dos líneas seguidas que se desmienten. bf16 **sí** es usable en sm_89; el mensaje de la política de dtype es una plantilla escrita para sm_75 que interpola el `compute_cap` sin verificar nada. Lo mismo con `flash=off` teniendo FlashAttention disponible.

El costo es **velocidad, no calidad** —fp16 en inferencia de difusión funciona bien—, pero explica por qué el propio código sugiere *"considera el pipeline estándar de diffusers"*: en esa tarjeta, este pipeline renuncia a optimizaciones por un supuesto que ya no aplica.

## Verificación por bytes, bien hecha

Un contraste con [TorToise](01-la-arquitectura-de-tortoise), donde `download_models()` usa `if os.path.exists(...)` y da por bueno un `.pth` truncado. Aquí:

```python
if target.exists() and target.stat().st_size == spec["size"]:   # tamaño EXACTO
```

> *"Un archivo corto aquí es una descarga truncada o un puntero LFS, y de otro modo aparecería mucho después como **un error opaco de parseo de protobuf** dentro de onnxruntime."*

Mismo problema, dos calidades de ingeniería. Vale la pena reconocer el patrón: **verificar existencia no es verificar integridad**, y la diferencia se paga con un error incomprensible varios minutos más tarde.

---

**Siguiente:** [El playbook de alineación](06-el-playbook-de-alineacion) — cómo hacer que los labios calcen.
