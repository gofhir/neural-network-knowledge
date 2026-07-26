---
title: "Fine-tuning eficiente con LoRA"
weight: 2
---

El modelo base no usó bien la herramienta ([parte anterior](01-uso-de-herramientas)). El segundo bloque lo resuelve **modificando el modelo** — pero de forma barata: fine-tuning con LoRA sobre ejemplos correctos.

## El problema: fine-tuning completo es carísimo

Fine-tunear todos los pesos de un modelo de 4B parámetros requiere mantener en memoria los pesos (16 GB), los gradientes (16 GB) y los estados del optimizador Adam (32 GB) → **~64 GB** solo para el estado de entrenamiento. Inviable en una GPU de Colab.

El mayor consumidor es el **optimizador**: Adam mantiene 2 estados (momentum + varianza) por peso, en float32 = 8 bytes adicionales *por cada peso entrenable*.

## La idea de LoRA: adaptadores de bajo rango

LoRA (**Lo**w-**R**ank **A**daptation) parte de una observación: el cambio que necesita un peso durante el fine-tuning tiene **rango bajo**. En vez de aprender la actualización completa $\Delta W$ (del mismo tamaño gigante que $W$), la factoriza en dos matrices flacas:

$$W_{\text{adaptado}} = W_0 + \Delta W = W_0 + B \cdot A$$

con $A$ de tamaño $r \times d$ y $B$ de $d \times r$, y $r$ pequeño (8, 16, 128). El producto $BA$ recupera una matriz $d \times d$ pero con muchísimos menos parámetros.

{{< callout type="info" >}}
**Es la misma factorización de matrices que ya conoces.** $\Delta W = B \cdot A$ es una descomposición de rango bajo — el mismo principio del matrix factorization en recomendación (usuario×ítem ≈ U×V) o PCA. LoRA lo aplica al fine-tuning: aproximar una matriz grande con el producto de dos flacas.
{{< /callout >}}

### Por qué ahorra tanto

- **Pesos entrenables**: solo $A$ y $B$; $W_0$ queda **congelado**. Gradientes y estados de Adam solo para ese ~6% → el optimizador colapsa de 32 GB a decenas de MB.
- **Precisión del base**: como $W_0$ no se entrena, puede quedar en float16 o cuantizado a 4-bit (**QLoRA**, con `BitsAndBytesConfig`).
- **Menos olvido catastrófico**: al no tocar $W_0$, el conocimiento general se preserva (conecta con el [lab 32](/laboratorios/lab-32)). El adaptador *añade* la capacidad nueva sin *borrar* la vieja.

## El entrenamiento (resultados reales)

El notebook entrena un `Qwen3.5-2B` con `SFTTrainer` (de la librería `trl`) sobre **580 conversaciones** de traducción-con-diccionario correctas. La configuración LoRA: `r=128`, `target_modules="all-linear"`.

```
trainable params: 134,553,600 || all params: 2,016,378,688 || trainable%: 6.6730
```

**Solo el 6.67% de los parámetros se entrena.** Los resultados sobre 33 steps (1 época):

| Step | Train Loss | Eval Loss | Token Accuracy |
|------|-----------|-----------|----------------|
| 10 | 2.582 | 2.359 | 0.576 |
| 20 | 2.059 | 2.032 | 0.612 |
| 30 | 1.938 | 1.954 | 0.620 |
| 33 | 1.938 | 1.951 | 0.619 |

La loss baja de 2.58 a 1.95 y la **eval loss sigue de cerca a la train loss** (no se despega) → **no hay overfitting** — validando la ventaja de LoRA de resistir el sobreajuste al congelar el base.

{{< callout type="info" >}}
**Detalle del pipeline de datos:** el código de preparación **reescribe el system prompt** eliminando la restricción ambigua de "máximo N turnos" (la que paralizó al modelo base) y la reemplaza por *"usa el diccionario tantas veces como necesites"*. El fine-tuning no solo enseña con ejemplos: **corrige el prompt problemático**. A veces el fallo no está en el modelo sino en la instrucción.
{{< /callout >}}

## El resultado: el contraste dramático

La misma traducción de *"The quick brown fox..."*, ahora con el modelo **fine-tuneado**:

| | **Modelo base** | **Fine-tuneado (LoRA)** |
|---|---|---|
| Tool calls ejecutadas | **0** | **7** |
| Comportamiento | parálisis, 7.700 chars deliberando | razona breve, luego **actúa** |
| Formato de llamada | nunca válido | tool calls correctas, **en paralelo** |
| Anti-alucinación | violada (adivinó palabras) | respetada (fundamentó en el diccionario) |
| Resultado | sin traducción | `šúŋǧíla kȟo aglíyopsičA šúŋka hokšíla-wičháȟča` |

El fine-tuning **transformó el comportamiento**: de "incapaz de usar la herramienta" a "la usa fluida y correctamente" — con solo 6.67% de parámetros y 33 steps.

{{< callout type="warning" >}}
**Honestidad sobre la calidad.** El *comportamiento* mejoró rotundamente, pero la traducción final no es lingüísticamente perfecta (usó una forma plural para "jumps", ignoró "brown"). Dos causas ajenas al fine-tuning: el **diccionario es parcial** (solo entradas A–I) y es un **modelo de 2B**. La lección: el fine-tuning arregló el *proceso* (usar la herramienta), pero la *calidad final* está acotada por los recursos. Para FHIR: podrías fine-tunear un modelo para que *siempre* consulte tu validador y produzca recursos bien formados, pero la corrección de los códigos seguirá dependiendo de la cobertura de tu terminología.
{{< /callout >}}
