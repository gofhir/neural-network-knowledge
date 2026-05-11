---
title: "Food101 - evaluacion + Actividad 3 (templates)"
weight: 80
math: true
---

En la seccion anterior vimos como CLIP clasifica **una sola imagen** zero-shot sobre Food101. Aqui escalamos a la **evaluacion completa del test set** (25,250 imagenes, 101 clases) midiendo **Top-1 y Top-5 accuracy**, y resolvemos la **Actividad 3** del lab — que pide probar templates distintos para entender el impacto del **prompt engineering** del lado del text encoder.

La leccion clave: CLIP **casi alcanza el rendimiento de un modelo especializado** (84% vs ~88-90% de ResNet50 fine-tuned) sin haber visto el dataset durante entrenamiento. Pero el **template** importa — el template del lab (`'A photo of {}, a type of food.'`) **NO** es trivial; quitar la pista "a type of food" cuesta 5.6 puntos de accuracy.

## Evaluacion masiva sobre el test set

La funcion `evaluate_model` *(parte 2, celda 32)* itera el test set en batches y mide accuracy:

```python
def evaluate_model(model, dataset, queries, batch_size=512):
    test_loader = DataLoader(dataset, batch_size=batch_size)

    in_top1 = in_top5 = total = 0.
    total_batches = len(test_dataset) // batch_size
    for image_inputs, true_labels in tqdm(test_loader, total=total_batches):
        image_inputs = image_inputs.to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_inputs).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(queries).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

        label_match = (top_labels == true_labels.unsqueeze(-1))
        in_top1 += float(label_match[:,0].sum())
        in_top5 += float(label_match.any(-1).sum())
        total += true_labels.numel()

    top1_acc = in_top1 / total
    top5_acc = in_top5 / total

    return top1_acc, top5_acc
```

### El trick para Top-1 y Top-5

```python
label_match = (top_labels == true_labels.unsqueeze(-1))
```

- `top_labels` shape `(512, 5)`: 5 predicciones por imagen
- `true_labels.unsqueeze(-1)` shape `(512, 1)`: agrega dim para broadcasting
- `==` con broadcasting: shape `(512, 5)`, `True` donde la prediccion coincide

| Linea | Que cuenta |
| --- | --- |
| `label_match[:, 0].sum()` | **Top-1**: la primera prediccion (la mas probable) es la correcta |
| `label_match.any(-1).sum()` | **Top-5**: la real esta en cualquiera de las 5 mejores predicciones |

### Por que Top-5 ademas de Top-1

En datasets con clases visualmente similares (`chicken_wings` vs `baby_back_ribs`), Top-1 es una metrica **dura** — castiga errores razonables. Top-5 es mas **generosa** — premia que el modelo "haya considerado" la respuesta correcta entre sus mejores candidatos.

> Para clasificacion practica con muchas clases similares, Top-5 suele ser la metrica mas informativa. Si tu modelo tiene Top-5 = 90% pero Top-1 = 50%, **no esta perdido** — esta confundiendo clases similares pero entendiendo el contexto global.

## Resultados sobre Food101 (template baseline)

Ejecutando `evaluate_model` con el template `'A photo of {}, a type of food.'` *(parte 2, celdas 34-36)*:

```
Top-1 Accuracy: 84.01%
Top-5 Accuracy: 97.31%
```

### Comparacion contextual

| Referencia | Top-1 | Comentario |
| --- | --- | --- |
| **CLIP-ViT-B/32 zero-shot** (este resultado) | **84.01%** | Sin fine-tuning, sin haber visto Food101 |
| CLIP paper original (2021) | 80-83% | El paper reporto rango |
| ResNet50 fine-tuned EN Food101 | ~88-90% | Modelo entrenado especificamente |
| Humano novato | ~50-60% | Mucha gente no distingue baklava de strudel |

### Por que Top-5 es casi perfecto pero Top-1 no

La brecha de 13 puntos entre 84% Top-1 y 97% Top-5 dice algo concreto:

> El **97% de las veces**, la clase correcta esta entre las 5 mejores predicciones de CLIP. En el ~13% donde falla Top-1 pero acierta Top-5, **el modelo "considero" la respuesta correcta** pero priorizo otra clase visualmente similar.

Eso es lo que vimos en el caso individual de `hummus`: la clase correcta dominaba (33%), pero `garlic_bread`, `cheese_plate`, `foie_gras`, `deviled_eggs` aparecian tambien. En los casos donde la imagen es ambigua (un plato generico), el modelo a veces se queda con la opcion equivocada pero la real sigue en su top-5.

### Por que funciona tan bien en Food101 especificamente

Food101 es un dataset **"amigable para CLIP"** porque:

1. **Internet esta plagada de fotos de comida con captions**. CLIP fue entrenado con 400M pares, muchisimos eran fotos de comida con su nombre.
2. Las clases son **palabras comunes en ingles** (pizza, burger, tiramisu, baklava). El text encoder las reconoce sin problema.
3. Las imagenes son **fotos reales** (Foodspotting era una app social), parecidas a lo que CLIP vio en su corpus.

> **Spoiler**: cuando vayamos a **Stanford Cars** (modelos especificos como `2012 BMW M3 Coupe` vs `2012 BMW M5 Sedan`), CLIP va a **caer en picada**. Es parte de la leccion del lab.

## Actividad 3 — preguntas sobre dimensiones + templates

### Pregunta 3.1a — ¿A que corresponde la ultima dimension de los features?

**Enunciado** *(parte 2, celda 38)*:

> "Vimos que tanto `image_features` como `text_features` terminan en **512**. ¿A que corresponde el valor de la ultima dimension?"

**Respuesta:**

Corresponde a la **dimensionalidad del espacio compartido de embeddings de CLIP**. Tanto el image encoder (ViT-B/32) como el text encoder (Transformer) proyectan sus salidas a este espacio comun de 512 dimensiones mediante matrices de proyeccion finales (`visual.proj` y `text_projection`).

Es lo que permite calcular **similitud coseno entre imagenes y textos**: ambos viven en el mismo espacio vectorial. Para modelos CLIP mas grandes (ViT-L/14) este espacio es de **768 dimensiones**.

### Pregunta 3.1b — ¿Por que la matriz de similitud es 1×101?

**Enunciado** *(parte 2, celda 40)*:

> "¿Por que la matriz de similaridad es de `1x101`?"

**Respuesta:**

Porque tenemos **1 imagen y 101 queries** (una por cada clase de Food101). La operacion `image_features @ text_features.T` multiplica `(1, 512)` por `(512, 101)` y produce una matriz `(1, 101)` donde cada entrada `[0, i]` es la similitud coseno entre la imagen y la query `i`. Si pasaramos un batch de N imagenes, la matriz seria `(N, 101)`.

### Pregunta 3.2 — Sugerir 2 templates distintos y testear

**Enunciado** *(parte 2, celdas 41-45)*:

> "Sugiera 2 templates para queries distintos al utilizado previamente y testee que resultados obtiene."

**Templates elegidos:**

```python
Q1 = "A photo of {}."
Q2 = "A close-up photo of a plate of {}, a popular dish."
```

| Template | Top-1 | Top-5 | Delta vs baseline |
| --- | --- | --- | --- |
| **Baseline** `'A photo of {}, a type of food.'` | **84.01%** | **97.31%** | — |
| **Q1** `'A photo of {}.'` | 78.41% | 94.93% | **−5.6 / −2.4** |
| **Q2** `'A close-up photo of a plate of {}, a popular dish.'` | 82.49% | 96.88% | **−1.5 / −0.4** |

### Analisis Q1 — quitar contexto "food" cuesta 5.6 puntos

La pista `"a type of food"` aporta **senal real**. Sin ella, CLIP confunde imagenes de comida con otras cosas que tienen el mismo nombre:

- `"apple pie"` puede activarse en imagenes de tatuajes, ilustraciones de libros, dibujos animados
- `"Caesar salad"` puede competir con imagenes historicas
- `"bagel"` tiene tambien significado coloquial

El contexto culinario **acota la distribucion semantica** hacia fotografia real de comida servida. Quitarlo abre la puerta a falsos positivos.

### Analisis Q2 — mas detalle NO es mejor

Q2 agrega tres pistas (`close-up`, `plate of`, `popular dish`) y sin embargo **rinde 1.5 puntos peor** que el baseline. ¿Por que?

| Pista | Por que puede perjudicar |
| --- | --- |
| `close-up photo` | No todas las fotos de Food101 son close-up. Muchas son tomas amplias o desde lejos |
| `plate of` | Muchos platos no estan servidos en plato — sushi en madera, sandwich en mano, pasta en bowl |
| `popular dish` | Sesga hacia comidas occidentales mainstream. `bibimbap`, `baklava`, `pho` pueden no encajar bien con "popular dish" en el corpus en ingles |

> **Leccion clave de prompt engineering:** el template optimo es **lo suficientemente especifico** para anclar el contexto pero **no tan especifico** que excluya casos validos. El template del lab esta **bien balanceado** — agrega justo la pista necesaria sin restringir mas de la cuenta.

### El truco del paper: ensemble de templates

OpenAI descubrio que **un solo template no es optimo**, y el truco que les permitio batir SOTA en ImageNet fue:

1. Definir **80 templates diferentes** (`"a photo of a {}"`, `"a sculpture of a {}"`, `"a tattoo of a {}"`, `"itap of a {}"`, etc.)
2. Encodear cada template para cada clase → `80 × 1000 = 80,000` vectores de texto
3. **Promediar** los 80 vectores por clase → 1 embedding "robusto" por clase
4. Comparar la imagen contra los 1000 embeddings promediados

Eso les subio +3.5 puntos en ImageNet zero-shot. La idea: distintos templates capturan distintos contextos visuales, y promediarlos da una representacion mas robusta de "que significa esta clase".

## Conclusion de Food101

CLIP-ViT-B/32 zero-shot sobre Food101:

- **Top-1: 84.01%** — casi al nivel de modelos fine-tuned especializados, sin haber visto el dataset
- **Top-5: 97.31%** — practicamente perfecto considerando 101 clases
- **El template importa** — quitarle contexto cuesta 5.6 puntos
- **Mas detalle no siempre es mejor** — sobrespecificar puede excluir casos validos

Esta seccion mostro un caso "ideal" para CLIP: dominio amigable, palabras comunes, fotos reales. **La siguiente seccion** prueba CLIP en un caso **adversarial**: Stanford Cars, donde las distinciones son por modelo y ano (`BMW M5 2010` vs `BMW M5 2011`). La accuracy va a caer drasticamente y veremos por que.
