---
title: "38 - Encoder vs Decoder: la diferencia que lo cambia todo"
weight: 380
math: true
---

## 1. Apertura — una sola linea que divide dos paradigmas

Todos los modelos que construiste hasta ahora eran decoders. El Mini-GPT (cap 5) y el Mini-LLaMA (cap 13) solo leen hacia la izquierda. Cuando el modelo genera el token en la posicion $t$, solo puede ver los tokens en las posiciones $0, 1, \ldots, t-1$. Eso es la mascara causal: un triangulo inferior que bloquea el futuro.

BERT (Devlin et al., 2018) hizo algo radicalmente distinto: quito la mascara. Cada token puede atender a todos los demas tokens de la secuencia, en ambas direcciones. Una sola linea de codigo — `mask=None` en lugar de `mask=causal` — cambia el paradigma completo. Esa diferencia no es un detalle de implementacion: define que tipo de tareas puede hacer el modelo, y por que BERT no puede generar texto de la misma forma que GPT.

Este capitulo es el punto de entrada al Camino 4. Los caps 39 y siguientes construiran un Mini-BERT desde cero; este cap visualiza exactamente que significa la diferencia bidireccional vs causal antes de escribir una linea de entrenamiento.

---

## 2. La mascara causal — por que existe y que bloquea

La mascara causal es el mecanismo que hace posible el entrenamiento autoregresivo. Cuando entrenas un decoder, el objetivo es: dado el prefijo $x_0, x_1, \ldots, x_{t-1}$, predecir $x_t$. Para hacer eso en paralelo sobre toda la secuencia (en lugar de token a token), el truco es procesar todos los tokens simultaneamente pero *enmascarar* el acceso al futuro.

Formalmente, la matriz de atencion sin mascara es:

$$A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$

Con mascara causal, se rellena con $-\infty$ las posiciones futuras antes del softmax:

$$A_{ij} = \begin{cases} \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) & \text{si } j \leq i \\ 0 & \text{si } j > i \end{cases}$$

El $-\infty$ antes del softmax se convierte en $0$ despues — el token futuro recibe peso exactamente cero, como si no existiera.

La consecuencia: el token `"not"` en la posicion 3 de `"To be or not to be"` puede atender a `"To"`, `"be"`, `"or"`, y a si mismo — pero **no puede ver** `"to"` ni `"be"` (posiciones 4 y 5). El modelo aprende a predecir el siguiente token sin trampa: nunca ve la respuesta antes de tener que generarla.

---

## 3. Atencion bidireccional — que habilita y que imposibilita

El encoder quita la mascara. Cada token puede atender a todos los tokens de la secuencia. El token `"be"` en la posicion 1 puede ver `"To"` antes que el, y puede ver `"or"`, `"not"`, `"to"`, `"be"` despues. La representacion de `"be"` se construye con informacion completa del contexto.

Esto habilita tareas que los decoders hacen con dificultad:

- **Clasificacion de secuencias**: el token especial `[CLS]` acumula informacion de toda la oracion en ambas direcciones — el encoder bidireccional captura la semantica global mejor que un decoder.
- **Reconocimiento de entidades (NER)**: para saber si `"Paris"` es una ciudad o un nombre de persona, muchas veces necesitas leer lo que viene *despues*: `"Paris Hilton fue a Paris, Francia"`. Un decoder ve `"Paris"` en la primera posicion sin contexto posterior; un encoder ve ambas ocurrencias con contexto completo.
- **Inferencia de lenguaje natural (NLI)**: determinar si una premisa implica o contradice una hipotesis requiere comparar dos fragmentos de texto que se necesitan mutuamente — bidireccionalidad es critica.

Pero la atencion bidireccional **imposibilita la generacion autoregresiva en la forma estandar**. El problema es fundamental: si el modelo puede ver todos los tokens al mismo tiempo, incluyendo los futuros, entonces al entrenar para predecir el token $t$ ya tiene acceso a $t$ en su contexto de atencion. La prediccion no tendria costo — seria trivial copiar desde el futuro. Es un problema circular: no puedes entrenar a "adivinar el siguiente token" si el modelo ya lo ve.

BERT se entrena con una tarea diferente: **Masked Language Modeling (MLM)**. En lugar de predecir el siguiente token, se enmascara una fraccion aleatoria de los tokens de entrada (con el token especial `[MASK]`) y el modelo debe reconstruirlos usando el contexto bidireccional. Eso es lo que construimos en los caps siguientes.

---

## 4. El script

`clase_14/practica/38_encoder_vs_decoder.py`:

```python
"""38_encoder_vs_decoder.py - Cap 38: encoder vs decoder.

Visualiza la diferencia entre mascara causal (decoder) y
atencion bidireccional (encoder) sobre la misma frase.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)

T = 6  # longitud de secuencia de ejemplo
frase = ["To", "be", "or", "not", "to", "be"]

# === Mascara causal (decoder) ===
causal = torch.tril(torch.ones(T, T)).bool()
print("=== Mascara CAUSAL (decoder) ===")
print("Cada token solo puede atender tokens anteriores (incluyendose):\n")
header = f"{'':>6}" + "".join(f"{w:>6}" for w in frase)
print(header)
for i, wi in enumerate(frase):
    row = f"{wi:>6}" + "".join("  SI  " if causal[i, j] else "  NO  " for j in range(T))
    print(row)

# === Sin mascara (encoder) ===
print("\n=== Atencion BIDIRECCIONAL (encoder) ===")
print("Cada token puede atender a TODOS los tokens:\n")
print(header)
for i, wi in enumerate(frase):
    row = f"{wi:>6}" + "".join("  SI  " for _ in range(T))
    print(row)

# === Scores de atencion reales (un head aleatorio) ===
print("\n=== Scores de atencion encoder (un head) ===")
print("Muestra como 'be' (pos 1) atiende a todos:\n")
Q = torch.randn(T, 16)  # d_k = 16
K = torch.randn(T, 16)
scores = (Q @ K.T) / (16 ** 0.5)
attn_full = F.softmax(scores, dim=-1)
print("Pesos de atencion del token 'be' sobre todos los tokens:")
for j, wj in enumerate(frase):
    print(f"  be → {wj:>4}: {attn_full[1, j]:.3f}")

scores_causal = scores.masked_fill(~causal, float('-inf'))
attn_causal = F.softmax(scores_causal, dim=-1)
print("\nPesos de atencion del token 'not' (decoder, solo ve hasta 'not'):")
for j, wj in enumerate(frase):
    v = attn_causal[3, j]
    print(f"  not → {wj:>4}: {v:.3f}" + (" (bloqueado)" if v == 0 else ""))
```

---

## 5. Output del script

```
=== Mascara CAUSAL (decoder) ===
Cada token solo puede atender tokens anteriores (incluyendose):

          To    be    or   not    to    be
    To  SI    NO    NO    NO    NO    NO  
    be  SI    SI    NO    NO    NO    NO  
    or  SI    SI    SI    NO    NO    NO  
   not  SI    SI    SI    SI    NO    NO  
    to  SI    SI    SI    SI    SI    NO  
    be  SI    SI    SI    SI    SI    SI  

=== Atencion BIDIRECCIONAL (encoder) ===
Cada token puede atender a TODOS los tokens:

          To    be    or   not    to    be
    To  SI    SI    SI    SI    SI    SI  
    be  SI    SI    SI    SI    SI    SI  
    or  SI    SI    SI    SI    SI    SI  
   not  SI    SI    SI    SI    SI    SI  
    to  SI    SI    SI    SI    SI    SI  
    be  SI    SI    SI    SI    SI    SI  

=== Scores de atencion encoder (un head) ===
Muestra como 'be' (pos 1) atiende a todos:

Pesos de atencion del token 'be' sobre todos los tokens:
  be →   To: 0.032
  be →   be: 0.067
  be →   or: 0.718
  be →  not: 0.134
  be →   to: 0.031
  be →   be: 0.017

Pesos de atencion del token 'not' (decoder, solo ve hasta 'not'):
  not →   To: 0.339
  not →   be: 0.199
  not →   or: 0.142
  not →  not: 0.320
  not →   to: 0.000 (bloqueado)
  not →   be: 0.000 (bloqueado)
```

---

## 6. Analisis de las matrices

**Mascara causal (decoder):** La primera matriz es un triangulo inferior de `SI`. El patron es claro: la diagonal y todo lo que esta a la izquierda es accesible; todo lo que esta a la derecha es bloqueado. El ultimo token `"be"` (posicion 5) puede ver toda la secuencia — tiene el contexto mas rico. El primer token `"To"` (posicion 0) solo puede verse a si mismo — tiene el contexto mas pobre.

Esta asimetria es intencional y necesaria: cada token solo puede depender de lo que ya se ha generado. El modelo aprende una distribucion condicional $P(x_t | x_0, \ldots, x_{t-1})$ para cada posicion $t$.

**Atencion bidireccional (encoder):** La segunda matriz es toda `SI`. Cada token tiene acceso completo al resto. `"not"` puede atender a `"to"` y `"be"` que vienen despues — algo imposible en el decoder. `"To"` puede leer toda la oracion antes de construir su representacion.

La representacion resultante de cada token es contextual en ambas direcciones. El `"be"` inicial y el `"be"` final tendran representaciones diferentes — el encoder las distingue por su contexto completo.

**Scores reales (un head, pesos aleatorios):** Los scores muestran numeros no triviales incluso con pesos inicializados al azar. En el encoder, `"be"` (pos 1) asigna el 71.8% de su atencion a `"or"` — con pesos aleatorios esto es ruido, pero demuestra que el mecanismo funciona: la suma de pesos es exactamente 1, cada token recibe un peso no negativo, y ningun token esta bloqueado.

En el decoder, `"not"` (pos 3) distribuye su peso entre `"To"`, `"be"`, `"or"`, y `"not"` — los cuatro tokens visibles. Las posiciones 4 y 5 (`"to"` y `"be"`) reciben exactamente `0.000` porque el `masked_fill` les asigno $-\infty$ antes del softmax. La etiqueta `(bloqueado)` en el output confirma que el mecanismo de mascara funciona exactamente como se espera.

---

## 7. Por que el encoder NO puede generar texto

Este es el punto mas importante del capitulo. La pregunta natural es: "si el encoder es mas poderoso porque lee en ambas direcciones, por que no se usa para generacion?"

La respuesta es una trampa de circularidad: si el modelo ve todos los tokens — incluyendo los futuros — entonces al entrenar la prediccion del token $t$ ya tiene el token $t$ en su contexto de atencion. El gradiente no aprende nada util porque la respuesta ya esta visible.

Para hacer concreta la circularidad: supon que quieres predecir `"not"` en la frase `"To be or [PRED] to be"`. Con atencion bidireccional, el modelo puede atender directamente a las posiciones antes y despues de `[PRED]`. Si esas posiciones contienen el token real `"not"`, el modelo puede simplemente copiarlo — no aprende nada sobre la distribucion del lenguaje.

La solucion de BERT es cambiar la tarea: **no predecir el siguiente token, sino reconstruir tokens enmascarados**. Se toma la secuencia completa, se reemplaza el 15% de los tokens por `[MASK]`, y el modelo debe predecir cuales eran esos tokens usando el contexto bidireccional disponible. Eso no tiene la trampa: el modelo no puede "copiar" `[MASK]` — tiene que inferirlo del contexto.

La consecuencia es que BERT no puede hacer generacion autoregresiva estandar:
- GPT/LLaMA: `"To be or" → "not"` — predice el siguiente token.
- BERT: `"To [MASK] or not to be" → "be"` — reconstruye el token enmascarado.

No es que BERT "no pueda generar" en sentido absoluto — hay variantes como BERT-generation que adaptan el modelo. Pero el BERT estandar no es un modelo generativo en el sentido de "escribir texto nuevo token a token"; es un modelo de comprension que produce representaciones ricas de texto existente.

Esa diferencia explica el ecosistema actual: GPT/LLaMA/Mistral son decoders usados para generacion, asistentes, chatbots. BERT/RoBERTa/DeBERTa son encoders usados para clasificacion, NER, similitud semantica, extraccion de informacion. Son herramientas distintas para tareas distintas.

---

## 8. Preguntas de verificacion

**1.** En la mascara causal del script, `"or"` (posicion 2) puede atender a 3 tokens. Si la secuencia tuviera 10 tokens y `"or"` estuviera en la posicion 5, icuantos tokens podria atender con mascara causal? icuantos con atencion bidireccional?

**2.** El output muestra que `"be"` (encoder) asigna el 71.8% de su atencion a `"or"` con pesos aleatorios. iQue representa este valor en el contexto del aprendizaje? iQue esperarias que pasara con ese 71.8% despues del entrenamiento sobre un corpus de lenguaje natural?

**3.** Si quisieramos usar un encoder BERT para hacer completacion de texto (tomar un prefijo y generar el siguiente token), iqual seria el problema fundamental? iComo lo resolveria una arquitectura decoder como Mini-LLaMA?
