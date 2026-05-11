---
title: "Stanford Cars + Actividad 4 - los limites de zero-shot"
weight: 90
math: true
---

Esta es la ultima seccion del lab y la mas educativa para entender **donde CLIP falla**. Hasta aqui vimos el caso amigable (Food101 → 84% Top-1). Ahora cambiamos a **Stanford Cars** — 196 clases de modelos especificos de auto con ano y trim — donde la accuracy cae a **57.93%**. La brecha gigantesca con Top-5 (89.64%) revela el patron: **CLIP esta en el vecindario semantico correcto pero no precisa el modelo exacto**.

La leccion final: **zero-shot no es bala magica**. Funciona excelente en dominios "amigables a internet" (comida, animales, escenas), falla en dominios tecnicos donde las distinciones son sutiles y poco descritas en captions web.

## Stanford Cars: el dataset

Stanford Cars (Krause et al. 2013) tiene caracteristicas que lo hacen **adversarial para CLIP**:

- **196 clases** que distinguen autos por **marca + modelo + ano + trim**
- Ejemplos: `BMW M5 2010 Sedan`, `BMW M5 2011 Sedan`, `BMW M5 2012 Sedan`
- **Test set**: 8,041 imagenes

Algunos modelos son **visualmente casi identicos** (cambios menores en parrilla, faros, paragolpes entre anos consecutivos).

### Setup identico al de Food101

El flujo es **el mismo** que ya vimos:

```python
query_template = 'A photo of {}'
queries = [query_template.format(label.replace('_', ' ')) for label in cars_dataset.classes]
tokenized_queries = clip.tokenize(queries).to(device)
```

Notar que el template aqui es mas simple — `'A photo of {}'` sin sufijo. No tiene sentido el "a type of food" porque ya no es comida.

### Caso individual: el Dodge Magnum disfrazado de Ecto-1

Inspeccionando una imagen aleatoria del test set:

![Stanford Cars - Dodge Magnum Wagon Ghostbusters](/laboratorios/lab-14/cars-top5-dodge-magnum-ghostbusters.png)

```
Predicciones Top-5

Clase                Probabilidad
Dodge Magnum Wagon 2008    0.7052   ← TOP-1 CORRECTO
Dodge Charger SRT-8        0.0621
Chrysler 300 SRT-8         0.0509
Dodge Caliber Wagon        0.0401
Dodge Dakota Club Cab      0.0346

Clase verdadera: Dodge Magnum Wagon 2008
```

Lo gracioso de la imagen: ese auto esta **disfrazado como el Ecto-1 de Ghostbusters** (con logo en la puerta, antenas, parafernalia en el techo). Pero CLIP miro **mas alla del disfraz** y reconocio el chasis subyacente del Dodge Magnum Wagon.

### Por que este caso es engañosamente bueno

| Observacion | Por que importa |
| --- | --- |
| **Confianza altisima (70%)** | CLIP "ve" el chasis del Magnum a pesar del disfraz Ghostbusters. El logo, sticker y antenas son **textura superficial**, no la forma del auto |
| **Las 4 alternativas son todas autos americanos** del mismo segmento Dodge/Chrysler, wagon/sedan | CLIP captura **clusters visuales** — sabe que es un auto americano de tipo wagon/sedan, no un europeo ni un japones |
| **Ninguna alternativa es de otro fabricante extranjero** | Toyota, Honda, BMW etc. quedaron muy por debajo. La marca/origen se capta antes que el modelo especifico |

Stanford Cars tiene **196 clases**. Los modelos cercanos (`Honda Accord 2010 Sedan` vs `Honda Accord 2011 Sedan`) son **visualmente casi identicos**. CLIP fue entrenado con captions de internet — las captions raramente dicen "2010 Honda Accord vs 2011 Honda Accord". Por eso **en agregado**, la accuracy va a caer drasticamente.

## Evaluacion masiva sobre Stanford Cars

Mismo flujo que con Food101 pero el test set es mas chico (8,041 imagenes). El notebook duplica el codigo en lugar de reusar `evaluate_model` (por un bug menor donde la funcion referencia `test_dataset` global de Food101 en lugar del parametro):

```python
in_top1 = in_top5 = total = 0.
total_batches = len(test_cars_dataset) // batch_size
for image_inputs, true_labels in tqdm(test_loader, total=total_batches):
    image_inputs = image_inputs.to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_inputs).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = model.encode_text(tokenized_queries).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

    label_match = (top_labels == true_labels.unsqueeze(-1))
    in_top1 += float(label_match[:,0].sum())
    in_top5 += float(label_match.any(-1).sum())
    total += true_labels.numel()
```

### Resultado: caida brutal

```
Top-1 Accuracy: 57.93%
Top-5 Accuracy: 89.64%
```

### Comparacion lado a lado con Food101

| Dataset | Top-1 | Top-5 | Brecha (Top5 − Top1) |
| --- | --- | --- | --- |
| **Food101** | 84.01% | 97.31% | 13.3 puntos |
| **Stanford Cars** | **57.93%** | **89.64%** | **31.7 puntos** ⚠ |

### El patron clave: la asimetria de la caida

- **Caida en Top-1**: −26 puntos (84% → 58%)
- **Caida en Top-5**: −7.7 puntos (97% → 90%)

> CLIP **sabe** que la imagen es un auto americano de tal segmento/cluster, pero **no puede precisar el modelo y año especificos**. Esta en el vecindario semantico correcto, no en la identidad correcta.

### La brecha de 31.7 puntos entre Top-1 y Top-5

En Food101 esa brecha era 13 puntos — el modelo dudaba entre comidas visualmente similares pero precisaba bien la mayoria. En Stanford Cars, la brecha se duplica con creces. **El modelo "considera" la respuesta correcta el 90% del tiempo, pero "elige" la correcta solo el 58%**.

Es exactamente lo que pasaria con un humano que distinguiera autos sin ser experto: veria "Dodge sedan americano", listaria 5 candidatos, pero no diria si es Charger, Magnum, Caliber o Dakota.

### Por que CLIP falla aqui especificamente

CLIP fue entrenado con **400M captions de internet**. Las captions raramente tienen el nivel de detalle de Stanford Cars:

| Caption tipica de internet | Caption necesaria para Stanford Cars |
| --- | --- |
| `"My new car!"` | `"2010 BMW M5 Sedan"` |
| `"Cool ride"` | `"2012 BMW M5 Sedan"` |
| `"Black sedan at the dealer"` | `"2011 BMW M5 Sedan"` |

Las distinciones ano-a-ano son **sutiles** (cambios menores en parrilla, faros, paragolpes) y **rarisima vez aparecen en captions reales**. CLIP no tuvo ejemplos suficientes para aprender a distinguir `BMW M5 2010` de `BMW M5 2011`.

### Panorama general: cuando zero-shot vs fine-tuning

| Tarea | Modelo | Top-1 |
| --- | --- | --- |
| Food101 (101 clases) | CLIP zero-shot | **84%** |
| Food101 | ResNet50 fine-tuned | 88-90% |
| Stanford Cars (196 clases) | CLIP zero-shot | **58%** |
| Stanford Cars | ResNet50 fine-tuned | 88-90% |
| Stanford Cars | EfficientNet fine-tuned | 94% |

En Food101, zero-shot esta **a 4 puntos** de fine-tuning. En Stanford Cars, esta **a 30 puntos**. La distancia explota.

## Actividad 4 — tus propias imagenes

**Enunciado** *(parte 2, celda 63)*:

> "Prueba con tus propias imagenes. Utiliza el codigo a continuacion para subir 5 imagenes distintas y generar 5 queries para estas, las queries deben ser distintas y debe haber una asociada a cada imagen."

### Setup elegido: 5 imagenes de ImageNet distintas

Se descargaron 5 imagenes de un repo publico de samples de ImageNet en GitHub (`EliSchwartz/imagenet-sample-images`):

```python
import urllib.request

urls = {
    "01_dog.jpg":    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
    "02_bird.jpg":   "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n01530575_brambling.JPEG",
    "03_car.jpg":    "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n02701002_ambulance.JPEG",
    "04_pizza.jpg":  "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n07873807_pizza.JPEG",
    "05_lemon.jpg":  "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/n07749582_lemon.JPEG",
}

filenames = []
for fn, url in urls.items():
    urllib.request.urlretrieve(url, fn)
    filenames.append(fn)
```

### Las 5 queries

```python
queries = [
    "A photo of a dog",
    "A photo of a bird",
    "A photo of an ambulance",
    "A photo of a pizza",
    "A photo of a lemon",
]
tokenized_queries = clip.tokenize(queries).to(device)
```

### Calculo de similitud (sin el factor logit_scale)

```python
similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy().T
```

Notar que aqui **falta el `100.0 *`** que si esta en la celda 26 (caso individual de Food101). Sin ese factor, el softmax opera sobre valores muy chicos y produce una distribucion **casi uniforme** — los valores se ven apretados visualmente.

### Resultado: matriz de similitud 5×5

![Actividad 4 - matriz de similitud 5x5](/laboratorios/lab-14/actividad4-similarity-matrix-5x5.png)

Diagonal claramente dominante:

```
                          dog  bird  ambul  pizza  lemon
A photo of a dog          0.22  0.20  0.20   0.20   0.20
A photo of a bird         0.20  0.22  0.20   0.20   0.20
A photo of an ambulance   0.19  0.19  0.22   0.20   0.19
A photo of a pizza        0.20  0.20  0.19   0.22   0.19
A photo of a lemon        0.19  0.20  0.19   0.19   0.22
```

**Top-1 correcto para las 5 imagenes** (la diagonal tiene siempre el maximo de su fila). Pero las diferencias absolutas son chicas — solo 0.02-0.03 puntos entre diagonal y resto.

### Por que los valores estan tan apretados

La celda 66 del notebook usa:

```python
similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy().T
```

**Sin** el factor `100.0` (el `logit_scale`). Comparemos:

| Caso | Operacion | Resultado |
| --- | --- | --- |
| **Sin escala (este caso)** | `softmax(coseno entre -1 y 1)` | Distribucion casi uniforme |
| **Con escala (Food101, Cars)** | `softmax(100 × coseno)` | Distribucion muy concentrada |

Con valores de coseno crudo entre, digamos, 0.30 (diagonal) y 0.25 (fuera):

```
softmax([0.30, 0.25, 0.25, 0.25, 0.25])
  ≈ [0.22, 0.20, 0.20, 0.20, 0.20]
```

Es exactamente lo que ves en la matriz. Las diferencias absolutas del coseno (~0.05) son pequenas, y sin amplificar con `×100` se ven aun mas pequenas tras el softmax.

### Si agregaramos el factor 100 manualmente

```python
similarity_scaled = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy().T
```

Esperariamos ver algo cercano a la matriz identidad:

```
[[1.00 0.00 0.00 0.00 0.00]
 [0.00 1.00 0.00 0.00 0.00]
 [0.00 0.00 1.00 0.00 0.00]
 [0.00 0.00 0.00 1.00 0.00]
 [0.00 0.00 0.00 0.00 1.00]]
```

CLIP **separa perfectamente** estas 5 clases visualmente distintas. Solo el factor de escala faltante hace que en la visualizacion del lab los valores parezcan apretados.

### Analisis final escrito (celda 71)

```
Test con 5 imagenes ImageNet distintas (perro samoyedo, pinzon, ambulancia, pizza, limones)
y 5 queries 'A photo of a {X}'. Resultados: matriz diagonal con maximo en la diagonal (0.22)
y 0.19-0.20 fuera. Top-1 correcto para las 5 imagenes. Conclusion: CLIP separa perfectamente
clases visualmente distintas; el contraste de la visualizacion depende de la temperatura del
softmax, no solo de la calidad del modelo.
```

## La leccion final del lab

CLIP es excelente para dominios **"amigables a internet"** — comida, animales, escenas, objetos comunes, arte famoso — donde las captions web ya tienen vocabulario rico. **Falla** en:

- **Distinciones tecnicas finas** (modelos de auto, especies de plantas, variedades de uva, farmacos)
- **Idiomas con poco corpus** (idiomas con baja representacion web)
- **Dominios especializados** (medical imaging, satellite imagery, microscopia)

Para esos casos, **fine-tuning sigue siendo necesario** — o modelos especificos del dominio (BioCLIP para biologia, RemoteCLIP para satelite, SigLIP de Google para mejora general, etc.).

### El truco del prompt engineering

Mas alla del template, el factor de escala (`logit_scale = 100`) **importa visualmente**: separa distribuciones casi uniformes de distribuciones concentradas. Es un parametro **aprendido** por CLIP durante entrenamiento, no un hyperparametro arbitrario.

### Para tu propio dominio (salud digital, FHIR, etc.)

Lo que el lab demuestra es exactamente la razon por la que un MDM serio no puede depender solo de embeddings tipo CLIP/BERT:

1. **El vocabulario clinico (HL7, FHIR, ICD-10, SNOMED) es out-of-distribution** para corpus web
2. **Las distinciones criticas son finas** (paciente Juan Perez RUT X vs Juan Perez RUT Y) — analogo a `BMW M5 2010` vs `BMW M5 2011`
3. **Necesitas features explicitas** (UCUM, demograficos, identificadores) ademas de embeddings

CLIP es una herramienta poderosa pero **no es la respuesta universal**. Saber **cuando funciona y cuando no** es lo que diferencia un practitioner de un usuario casual.

## Cierre del Lab 14 completo

Con esta seccion termina el Laboratorio 14. Las respuestas concretas a las **11 preguntas conceptuales** (3 de Actividad 1 + 4 de Actividad 2 + 4 de Actividades 3 y 4) estan razonadas en [resolucion](../resolucion). Los enunciados literales del notebook estan en [ejercicios](../ejercicios).

Para profundizar en CLIP y modelos multimodales:

- Paper original: [Radford et al. 2021 — Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [open_clip](https://github.com/mlfoundations/open_clip) — implementacion abierta con modelos mas recientes (incluyendo entrenamiento con LAION-2B)
- [SigLIP](https://arxiv.org/abs/2303.15343) — variante de Google con sigmoid loss en lugar de softmax contrastive, ~5% mejor en zero-shot
