---
title: "39 - Positional Embeddings aprendidos: BERT vs RoPE"
weight: 390
math: true
---

## 1. Dos formas de decirle al modelo donde esta cada token

En el cap 18 viste RoPE — rotaciones geometricas aplicadas a los vectores Q y K en cada cabeza de atencion. RoPE no agrega ningun vector al embedding del token; en cambio, rota Q y K en el espacio complejo segun su posicion en la secuencia, de modo que el producto punto $q_i \cdot k_j$ incorpora automaticamente la distancia relativa $i - j$.

BERT (Devlin et al., 2018) usa algo mas simple: embeddings de posicion **aprendidos** que se **suman** al embedding de token. Cada posicion $p \in \{0, 1, \ldots, \text{max\_seq\_len}-1\}$ tiene un vector dedicado $\mathbf{e}_p \in \mathbb{R}^{d}$, y la representacion de entrada para el token en esa posicion es simplemente:

$$\mathbf{h}_p = \text{TokenEmb}(x_p) + \mathbf{e}_p$$

Esos vectores $\mathbf{e}_p$ se inicializan al azar y se aprenden por gradiente junto con el resto del modelo. No hay formula explicita: el modelo descubre por si mismo que representacion posicional minimiza la perdida MLM.

---

## 2. Por que BERT eligio embeddings aprendidos en 2018

Cuando Devlin et al. disenaron BERT, las opciones disponibles eran:

1. **Embeddings sin/cos fijos** (Vaswani et al., 2017 — "Attention is All You Need"): vectores deterministicos construidos con funciones sinusoidales de frecuencia creciente. No tienen parametros, pero tampoco pueden adaptarse al dominio.
2. **Embeddings aprendidos**: vectores en una `nn.Embedding` entrenados por gradiente, identicos en forma a los embeddings de token.

RoPE fue propuesto por Su et al. en 2021 — tres anos despues de BERT. En 2018 simplemente no existia como opcion. Vaswani 2017 habia comparado sin/cos fijo con embeddings aprendidos y encontrado resultados equivalentes; Devlin et al. eligieron la version aprendida por su simplicidad y flexibilidad.

El costo es claro: $\text{max\_seq\_len} \times d_{\text{model}}$ parametros extra. Para BERT-base ($d = 768$, $\text{max\_seq\_len} = 512$) eso son $\approx 393$K parametros de posicion, menos del 0.4% del total del modelo.

---

## 3. La diferencia fundamental: parametros y extrapolacion

La tabla resume las tres familias:

| Metodo | Parametros extra | Extrapolacion | Cuando surge el patron |
|---|---|---|---|
| Sin/Cos fijo (Vaswani 2017) | 0 | Limitada | En el diseno (no se aprende) |
| Learned (BERT) | max\_seq\_len × d\_model | No garantizada | Durante entrenamiento MLM |
| RoPE (LLaMA) | 0 | Si (relativa, escala) | En el diseno (rotacion geometrica) |

**Extrapolacion** significa: el modelo puede procesar secuencias mas largas que las vistas durante el entrenamiento. RoPE tiene ventaja aqui porque la rotacion se aplica a cualquier posicion entera, incluso a posiciones que no aparecieron en el training. Los embeddings aprendidos de BERT solo tienen vectores para posiciones $0 \ldots \text{max\_seq\_len}-1$; pasar una secuencia mas larga causa un error de indice fuera de rango.

En RoPE, la posicion no es un "slot" en una tabla — es un angulo de rotacion que se puede calcular para cualquier entero. Esa diferencia de diseno explica por que LLaMA y modelos modernos usan RoPE en lugar de embeddings aprendidos.

---

## 4. El script

`clase_14/practica/39_positional_embeddings.py`:

```python
"""39_positional_embeddings.py - Cap 39: learned pos emb vs RoPE.

Muestra como se ven los embeddings de posicion aprendidos
y los compara conceptualmente con RoPE del cap 18.
"""
import torch
import torch.nn as nn
from _models import LearnedPositionalEmbedding

torch.manual_seed(42)

d_model = 128
max_seq_len = 128

emb = LearnedPositionalEmbedding(max_seq_len, d_model)

print("=== Learnable Positional Embeddings ===\n")
print(f"Shape del modulo: nn.Embedding({max_seq_len}, {d_model})")
print(f"Params: {max_seq_len * d_model:,} (uno por posicion × dimension)")
n_params = sum(p.numel() for p in emb.parameters())
print(f"Params totales: {n_params:,}\n")

# Mostrar similitud entre embeddings de posiciones cercanas vs lejanas
weights = emb.embedding.weight.detach()  # (128, 128)

def cos_sim(a, b):
    return (a @ b) / (a.norm() * b.norm())

print("Similaridad coseno entre embeddings de posicion (random init):")
print(f"  pos 0 vs pos 1:  {cos_sim(weights[0], weights[1]):.4f}")
print(f"  pos 0 vs pos 64: {cos_sim(weights[0], weights[64]):.4f}")
print(f"  pos 0 vs pos 127:{cos_sim(weights[0], weights[127]):.4f}")
print("\nNOTA: en random init estos valores son ruido — no tienen significado.")
print("El patron posicional (cercanas mas similares) solo emerge DESPUES del MLM training.")
print("Podemos re-correr este script post-training para ver la diferencia.")

print("\n=== Comparacion con RoPE (cap 18) ===")
print("""
RoPE (Rotary Position Embedding):
  - NO agrega nada a los embeddings de token
  - Rota Q y K en el espacio complejo segun la posicion
  - La similitud posicional emerge del producto punto rotado
  - Ventaja: extrapolacion a secuencias mas largas que el training

Learned Positional Embeddings (BERT):
  - SE SUMA un vector aprendido al embedding de token
  - No hay garantia de extrapolacion
  - Ventaja: mas simple, aprendible de forma directa
  - Limitacion: solo funciona hasta max_seq_len del training
""")

print("=== Forward pass ===")
x = torch.zeros(2, 10, d_model)  # secuencia de zeros
out = emb(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"La diferencia output - input = los embeddings de posicion:")
diff = out - x
for pos in [0, 3, 9]:
    print(f"  pos {pos}: norma = {diff[0, pos].norm():.4f}")
```

---

## 5. Output del script

```
=== Learnable Positional Embeddings ===

Shape del modulo: nn.Embedding(128, 128)
Params: 16,384 (uno por posicion × dimension)
Params totales: 16,384

Similaridad coseno entre embeddings de posicion (random init):
  pos 0 vs pos 1:  0.1650
  pos 0 vs pos 64: -0.0376
  pos 0 vs pos 127:-0.0128

NOTA: en random init estos valores son ruido — no tienen significado.
El patron posicional (cercanas mas similares) solo emerge DESPUES del MLM training.
Podemos re-correr este script post-training para ver la diferencia.

=== Comparacion con RoPE (cap 18) ===

RoPE (Rotary Position Embedding):
  - NO agrega nada a los embeddings de token
  - Rota Q y K en el espacio complejo segun la posicion
  - La similitud posicional emerge del producto punto rotado
  - Ventaja: extrapolacion a secuencias mas largas que el training

Learned Positional Embeddings (BERT):
  - SE SUMA un vector aprendido al embedding de token
  - No hay garantia de extrapolacion
  - Ventaja: mas simple, aprendible de forma directa
  - Limitacion: solo funciona hasta max_seq_len del training

=== Forward pass ===
Input:  torch.Size([2, 10, 128])
Output: torch.Size([2, 10, 128])
La diferencia output - input = los embeddings de posicion:
  pos 0: norma = 10.8772
  pos 3: norma = 10.5511
  pos 9: norma = 11.9482
```

---

## 6. Interpretacion del output

**Conteo de parametros:** El modulo tiene exactamente $128 \times 128 = 16{,}384$ parametros — uno por cada combinacion posicion-dimension. Comparado con el Mini-BERT completo (cap 40), estos parametros representan una fraccion pequeña del modelo.

**Similaridades coseno (random init):** Los tres valores — $0.1650$, $-0.0376$, $-0.0128$ — no tienen patron discernible. La posicion 1 no es mas "similar" a la posicion 0 que la posicion 127. Eso es esperado: en inicializacion aleatoria los vectores son ruido isotropico en $\mathbb{R}^{128}$, y la similitud coseno entre dos vectores aleatorios de dimension alta tiende a cero (concentracion de medida).

El patron posicional que hace utiles estos embeddings solo emerge despues del entrenamiento MLM. Tras el training, posiciones adyacentes tendrian embeddings mas similares, y la geometria del espacio reflejaria las regularidades de la secuencia aprendidas del corpus. Un experimento posible: guardar los embeddings de posicion antes y despues del training (cap 41) y comparar las matrices de similitud.

**Forward pass:** La entrada es un tensor de ceros de forma `(2, 10, 128)`. La salida tiene la misma forma — los embeddings de posicion se suman sin cambiar la shape. La diferencia `output - input` es exactamente el vector de posicion para cada slot. Las normas ($10.88$, $10.55$, $11.95$) son similares entre si pero no identicas: cada posicion tiene su propio vector inicializado independientemente.

---

## 7. La limitacion de extrapolacion

Considera un Mini-BERT entrenado con `max_seq_len=128`. La tabla de embeddings de posicion tiene exactamente 128 filas. Si en inferencia llegas con una secuencia de 130 tokens:

```python
positions = torch.arange(130)      # 0, 1, ..., 129
emb.embedding(positions)           # IndexError: index 128 is out of bounds
```

No hay embedding para la posicion 128 ni la 129 — no existen en la tabla. El modelo simplemente falla.

Con RoPE esto no ocurre. La rotacion para la posicion $p$ es:

$$R_p = \begin{pmatrix} \cos(p \cdot \theta_i) & -\sin(p \cdot \theta_i) \\ \sin(p \cdot \theta_i) & \cos(p \cdot \theta_i) \end{pmatrix}$$

Para $p = 128$ o $p = 10000$, la formula produce un valor bien definido sin necesidad de haberlo visto en el training. El modelo puede no generalizar bien a posiciones muy lejanas (hay investigacion activa sobre esto con YaRN, ALiBi, y otras tecnicas), pero al menos no falla con un error de indice.

Esta diferencia es la razon principal por la que los modelos modernos (LLaMA, Mistral, Gemma) usan RoPE o variantes, mientras que BERT y sus derivados directos (RoBERTa, DistilBERT) heredaron la limitacion de los embeddings aprendidos.

---

## 8. Preguntas de verificacion

**1.** El output muestra `Params totales: 16,384` para `max_seq_len=128`, `d_model=128`. BERT-base usa `max_seq_len=512` y `d_model=768`. iCuantos parametros de posicion tendria un BERT-base? iQue porcentaje representan sobre 110M parametros totales del modelo?

**2.** Las normas de los embeddings de posicion en random init son aproximadamente $10.88$, $10.55$, $11.95$. Un embedding de token tambien inicializado al azar en la misma dimension tendria una norma similar. iQue implica eso para la suma $\text{TokenEmb} + \text{PosEmb}$ al inicio del entrenamiento? iEsta suma esta dominada por el token, por la posicion, o por ninguno?

**3.** RoPE no suma nada al embedding del token — la posicion se aplica solo a Q y K dentro de la atencion. iQue ventaja tiene esto para el residual stream (la suma de residuales que fluye por todos los bloques del Transformer)? iComo cambia la interpretacion de los vectores en el residual stream comparado con BERT?
