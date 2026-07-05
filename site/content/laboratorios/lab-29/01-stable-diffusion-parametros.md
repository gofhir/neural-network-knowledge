---
title: "Stable Diffusion y sus parámetros"
weight: 1
---

Stable Diffusion genera imágenes **partiendo de ruido puro y quitándolo iterativamente**, guiado por el prompt. Tres perillas controlan ese proceso: cuántos pasos de denoising, con qué algoritmo, y cuánto obedecer el prompt.

## Cargar el pipeline

```python
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16       # FP16: la mitad de VRAM, viable en una T4
).to("cuda")

prompt = "An oil painting of Torres del Paine, painted in the cubist style of picasso"
images = pipeline(prompt, generator=torch.Generator(device="cuda")).images
```

El `torch_dtype=torch.float16` es el truco que hace viable SDXL en Colab: reduce el uso de VRAM a ~7GB (vs. ~14GB en FP32). Internamente, el pipeline codifica el prompt con CLIP, parte de ruido, itera el U-Net quitando ruido, y decodifica con el VAE.

## `num_inference_steps`: pasos de denoising vs. calidad

`num_inference_steps` es cuántas iteraciones de "quitar ruido" se hacen (default 50). Con la **semilla fija** (para aislar la variable), el barrido muestra el trade-off:

**5 pasos** — formas difusas, "derretidas", sin detalle fino. El denoising no tuvo iteraciones para refinar; solo alcanzó la composición global:

![Torres del Paine con 5 pasos: borroso](/laboratorios/lab-29/steps-5.jpg)

**100 pasos** — nítido, detallado, con el cielo cubista facetado, el lago turquesa y las pinceladas al óleo:

![Torres del Paine con 100 pasos: nítido](/laboratorios/lab-29/steps-100.jpg)

**200 pasos** — prácticamente idéntico a 100: **retornos decrecientes**. Duplicar el cómputo casi no aporta:

![Torres del Paine con 200 pasos: igual que 100](/laboratorios/lab-29/steps-200.jpg)

**Dos lecciones:**
- La curva calidad↔pasos tiene una **zona útil (~25-100)** y una **meseta (>100)**. El default de 50 es el balance general.
- Un matiz fino: cambiar el número de pasos **no solo refina** — con la misma semilla, 5 vs 100 pasos dieron composiciones algo distintas. El número de pasos discretiza la trayectoria de denoising (la ODE de reverse diffusion), y trayectorias distintas convergen a puntos finales distintos.

## Noise schedulers: el algoritmo de muestreo

El scheduler decide **cómo** dar cada paso de denoising (submuestreando los 1000 timesteps de entrenamiento a los ~50 de inferencia). Se intercambia sin recargar el modelo:

```python
scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)  # preserva la config del modelo
pipeline.scheduler = scheduler                                            # hot-swap
```

El `from_config(...)` es clave: reusa los parámetros de ruido del entrenamiento (`beta_start/end`, `beta_schedule`), garantizando compatibilidad. Probando **LMS, Euler, KDPM2 y UniPC** a ~50 pasos, los resultados son **casi idénticos**:

| LMS | UniPC |
|---|---|
| ![Scheduler LMS](/laboratorios/lab-29/scheduler-lms.jpg) | ![Scheduler UniPC](/laboratorios/lab-29/scheduler-unipc.jpg) |

**La lección:** a 50 pasos todos los schedulers **convergen**. La diferencia se vuelve dramática con **pocos** pasos, donde solvers modernos (UniPC, DPM-Solver) alcanzan buena calidad en 10-15 pasos mientras Euler/LMS necesitan más. El scheduler es un problema de **eficiencia** (calidad por paso), no de calidad final. Todos resuelven la misma ODE de reverse diffusion con distinto método numérico (como Euler vs Runge-Kutta).

## `guidance_scale`: cuánto obedecer el prompt (CFG)

La `guidance_scale` implementa **Classifier-Free Guidance**. En cada paso, el modelo predice el ruido **con** el prompt y **sin** él, y los combina:

```
ε_final = ε_incondicional + guidance_scale · (ε_condicional − ε_incondicional)
```

La `guidance_scale` amplifica la "dirección hacia el prompt". El barrido 0 → 10 → 40 lo muestra:

**guidance=0** — el prompt se diluye: sale un paisaje montañoso realista genérico, **sin cubismo** (en `diffusers`, CFG se desactiva con valores ≤ 1):

![guidance 0: realista, sin cubismo](/laboratorios/lab-29/guidance-0.jpg)

**guidance=10** — el punto dulce: Torres del Paine reconocibles **y** cubismo presente (planos facetados, colores en bloques):

![guidance 10: óptimo, Torres del Paine cubistas](/laboratorios/lab-29/guidance-10.jpg)

**guidance=40** — sobre-forzado: la imagen se "quema" — colores sobresaturados, ruido de píxeles magenta, fragmentación caótica:

![guidance 40: quemado, artefactos](/laboratorios/lab-29/guidance-40.jpg)

**Por qué se quema con guidance=40:** amplificar 40× la dirección hacia el prompt empuja las predicciones **fuera del rango en que el modelo fue entrenado**; el latente se sale de la distribución válida y al decodificar aparecen saturación y artefactos. El default 7.5 es el balance empírico.

**Gotcha de costo:** CFG **duplica** el cómputo por paso (dos forwards del U-Net: condicional + incondicional). Es el precio de controlar la adherencia al prompt.

Este experimento ilustra directamente un trade-off del [generative learning trilemma](../03-cuestionario): forzar la fidelidad al condicionamiento (guidance alto) degrada la calidad de la muestra.

---

**Siguiente:** [Modos alternativos: Img2Img, Inpainting, ControlNet](../02-modos-alternativos).
