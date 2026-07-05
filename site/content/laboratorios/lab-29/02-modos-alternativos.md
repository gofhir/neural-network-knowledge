---
title: "Modos alternativos: Img2Img, Inpainting, ControlNet"
weight: 2
---

Text2img parte de ruido puro. Los tres modos alternativos condicionan la generación con una **imagen de referencia** — y cada uno desacopla algo distinto: composición completa, región enmascarada, o estructura.

## Img2Img: condicionar con una imagen

Img2Img **agrega ruido parcial** a tu imagen y lo quita guiado por el prompt. Al no partir de ruido total, conserva la **estructura de bajo nivel** (composición) y reconstruye los detalles en el estilo pedido. El parámetro `strength` controla cuánto ruido se agrega (bajo = se parece al original; alto = el prompt manda más).

```python
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("...stable-diffusion-v1-5", torch_dtype=torch.float16)
prompt = "Painting in the style of studio ghibli movies, featuring Torres del Paine with a castle"
images = pipeline(prompt, generator=..., image=init_image).images
```

El resultado es la demostración más didáctica del lab: un **boceto tosco tipo MS-Paint** se transformó en una pintura **estilo Studio Ghibli** manteniendo la composición exacta — y el "castillo" del prompt aterrizó justo donde el boceto tenía un garabato púrpura:

| Entrada (boceto) | Resultado (Ghibli) |
|---|---|
| ![Boceto de entrada Img2Img](/laboratorios/lab-29/img2img-entrada.jpg) | ![Resultado Img2Img estilo Ghibli](/laboratorios/lab-29/img2img-resultado.jpg) |

Img2Img **heredó la disposición espacial** del boceto (montañas al fondo, río diagonal, campo verde, playa de arena, castillo abajo-derecha) y el denoising guiado por el prompt reconstruyó los detalles en el estilo pedido. Es "conservar composición, cambiar estilo".

## Inpainting: editar solo una región enmascarada

Inpainting se parece a Img2Img, pero **solo edita la zona marcada por una máscara**; el resto queda intacto. Usa un modelo especializado (`stable-diffusion-inpainting`) cuyo U-Net recibe canales extra (imagen + máscara) para rellenar coherentemente con el entorno.

```python
images = pipeline(prompt="A photorealistic fluffy orange cat sitting in a bench",
                  image=init_image, mask_image=mask_image, strength=0.8).images
```

La máscara marca al animal (blanco = editar, negro = conservar). Con `strength=0.8` se borra el perro y se crea un gato desde cero, integrado con la banca:

| Entrada (perro) | Máscara | Resultado (gato) |
|---|---|---|
| ![Perro en banca](/laboratorios/lab-29/inpaint-entrada.jpg) | ![Máscara del animal](/laboratorios/lab-29/inpaint-mascara.jpg) | ![Gato naranja, mismo fondo](/laboratorios/lab-29/inpaint-resultado.jpg) |

El resultado cambió **solo el animal** (perro → gato naranja) manteniendo banca, campo, iluminación y encuadre **idénticos**, sin costuras visibles en los bordes. El `strength` gobierna cuánto se conserva de lo que había dentro de la máscara: bajo (0.1-0.4) = retoques que respetan la forma original; alto (0.75-0.9) = destruye el contenido y recrea desde cero; 1.0 = ignora por completo lo que había.

> **Aplicación directa a datos clínicos:** inpainting sirve para **anonimización** de imágenes médicas — reemplazar o remover regiones con identificadores sin alterar el resto de la imagen.

## ControlNet: control estructural con bordes

ControlNet es el modo más potente: condiciona la generación con la **estructura** de una imagen (bordes, pose, profundidad). Una red adicional se acopla al U-Net e inyecta esa señal de control espacial, sin destruir lo que SD ya sabía (se entrena una copia con "zero convolutions").

El flujo: tomar una imagen, extraer sus **bordes con Canny**, y generar contenido nuevo que respete esos bordes.

```python
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained("...sd-v1-5", controlnet=controlnet, ...)
image = cv2.Canny(np.array(image), 100, 200)     # detector de bordes
```

Partiendo de un **retrato renacentista de 1584** y su mapa de bordes Canny:

| Retrato original (1584) | Bordes Canny |
|---|---|
| ![Retrato renacentista original](/laboratorios/lab-29/controlnet-original.jpg) | ![Mapa de bordes Canny](/laboratorios/lab-29/controlnet-canny.jpg) |

Con esos **mismos bordes** y **cuatro prompts distintos** (Gabriela Mistral, Kim Kardashian, Rihanna, Taylor Swift), ControlNet genera cuatro versiones con **idéntica pose, peinado y gorguera renacentista** pero con **cuatro identidades distintas**:

![Grilla ControlNet: 4 personas, misma estructura](/laboratorios/lab-29/controlnet-grilla.jpg)

**Lo notable:** ControlNet **desacopló estructura de contenido**. Los bordes Canny impusieron la geometría (pose + gorguera + encuadre), idéntica en las cuatro; el prompt puso la identidad. Y un detalle fino: cada persona salió en un **medio distinto** — Gabriela Mistral como dibujo a lápiz (probablemente porque de ella hay más grabados que fotos en los datos de entrenamiento), Kim y Taylor fotorrealistas, Rihanna como ilustración — pero **todas respetaron la estructura**. Es el control espacial preciso que text2img puro no ofrece.

## Nota de memoria (gotcha)

Cada modo carga modelos nuevos. Hay que liberar el pipeline anterior antes de cargar el siguiente, o la GPU da OOM:

```python
try:
    del pipeline
except NameError:
    pass
import gc; gc.collect(); torch.cuda.empty_cache()
```

En el notebook original la celda de ControlNet usa `flush()` (no definido) y `del pipeline` sin guard, lo que falla si el pipeline no existe. La versión de arriba (try/except + gc) es robusta.

---

**Siguiente:** [Cuestionario resuelto](../03-cuestionario).
