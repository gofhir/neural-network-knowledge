"""
Escalon 1 — Embeddings + dot product attention manual.

Objetivo: ver con tus ojos como se calcula self-attention sin abstracciones.
Vamos a usar Q = K = V = X (los embeddings mismos, sin proyecciones).
Eso aisla el mecanismo "comparar tokens via producto punto, normalizar con
softmax, y devolver una suma ponderada".

NO hay todavia: scaling por sqrt(d_k), Q/K/V aprendidos, multi-head, masking.
Eso viene en escalones 2 y 3.
"""

import torch

# Reproducibilidad: misma semilla = mismos numeros random.
torch.manual_seed(42)

# Imprimir tensores con 3 decimales y sin notacion cientifica para que se lean
# bien.
torch.set_printoptions(precision=3, sci_mode=False)


# ---------------------------------------------------------------------------
# Paso 1: vocabulario y embedding
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 1: Vocabulario y embedding")
print("=" * 70)

vocab = ["I", "love", "neural", "networks"]
vocab_size = len(vocab)
d_model = 4  # dimension del embedding (en Vaswani usan 512; aqui 4 para verlo)

# La capa embedding es una tabla (vocab_size x d_model). Cada fila es el
# embedding de una palabra. En la vida real se aprende; aqui la inicializamos
# random para ver el mecanismo.
embedding_table = torch.randn(vocab_size, d_model)
print(f"\nVocabulario:    {vocab}")
print(f"vocab_size:     {vocab_size}")
print(f"d_model:        {d_model}")
print(f"\nembedding_table shape: {embedding_table.shape}")
print(f"embedding_table:\n{embedding_table}")
print("\nLeer asi: cada FILA es el embedding de una palabra.")
print(f"  embedding_table[0] = embedding de '{vocab[0]}' = {embedding_table[0]}")
print(f"  embedding_table[2] = embedding de '{vocab[2]}' = {embedding_table[2]}")


# ---------------------------------------------------------------------------
# Paso 2: tokenizar una oracion y hacer lookup
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 2: Tokenizar la oracion 'I love neural networks'")
print("=" * 70)

sentence = ["I", "love", "neural", "networks"]
token_ids = torch.tensor([vocab.index(w) for w in sentence])
print(f"\nOracion: {sentence}")
print(f"token_ids: {token_ids}")

# El "lookup" es indexar la tabla por los ids. Resultado: matriz (T, d_model)
# donde T = numero de tokens en la oracion.
X = embedding_table[token_ids]  # shape: (T, d_model) = (4, 4)
T = X.shape[0]
print(f"\nX = embedding_table[token_ids], shape {X.shape}:")
print(X)
print("\nCada fila de X es el embedding del token correspondiente.")
print(f"  X[0] (embedding de 'I')        = {X[0]}")
print(f"  X[3] (embedding de 'networks') = {X[3]}")


# ---------------------------------------------------------------------------
# Paso 3: dot product entre dos vectores — la operacion fundamental
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 3: Dot product entre dos vectores")
print("=" * 70)

a = X[0]  # embedding de "I"
b = X[1]  # embedding de "love"
dot = (a * b).sum()  # producto elemento-a-elemento + suma
dot_alt = torch.dot(a, b)  # version equivalente con torch.dot
print(f"\na = X[0]        = {a}")
print(f"b = X[1]        = {b}")
print(f"a * b           = {a * b}        (multiplicacion elemento-a-elemento)")
print(f"sum(a * b)      = {dot:.3f}")
print(f"torch.dot(a, b) = {dot_alt:.3f}    (mismo resultado, API directa)")
print("\nSi a y b apuntan en direcciones similares -> dot grande positivo.")
print("Si apuntan en direcciones opuestas         -> dot grande negativo.")
print("Si son ortogonales                         -> dot ~ 0.")


# ---------------------------------------------------------------------------
# Paso 4: dot product de cada token con todos — version manual con loops
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 4: Matriz de scores (similitud entre cada par de tokens)")
print("=" * 70)

# Para cada par (i, j), calcular X[i] . X[j]. Resultado: matriz T x T.
# Hacemos primero la version manual con loops, despues la vectorizada.
scores_manual = torch.zeros(T, T)
for i in range(T):
    for j in range(T):
        scores_manual[i, j] = torch.dot(X[i], X[j])

print(f"\nscores_manual shape: {scores_manual.shape}")
print(f"scores_manual:\n{scores_manual}")

# La version vectorizada: X @ X.T es exactamente lo mismo. Mucho mas rapido en
# GPU porque corre en paralelo, y mucho mas legible.
scores = X @ X.T
print(f"\nscores = X @ X.T (version vectorizada, mismo resultado):")
print(scores)

# Verificacion de que son iguales
assert torch.allclose(scores_manual, scores), "deberian ser iguales!"
print("\nOK: scores_manual y X @ X.T son identicos.")

print("\nLeer la matriz: la fila i dice cuanto se parece el token i a CADA token.")
print(f"  scores[0] = similitud de '{sentence[0]}' con cada token: {scores[0]}")
print(f"  scores[1] = similitud de '{sentence[1]}' con cada token: {scores[1]}")
print("\nNotar que la diagonal scores[i,i] = X[i].X[i] = ||X[i]||^2 (siempre")
print("positivo). Los off-diagonal pueden ser negativos.")


# ---------------------------------------------------------------------------
# Paso 5: convertir scores a pesos con softmax
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 5: softmax sobre cada fila -> distribucion de probabilidad")
print("=" * 70)

# softmax(z_i) = exp(z_i) / sum_j exp(z_j)
# Aplicado por fila (dim=-1 = ultima dim). Cada fila suma 1.
weights = torch.softmax(scores, dim=-1)
print(f"\nweights = softmax(scores, dim=-1):")
print(weights)
print(f"\nSuma de cada fila (deberia ser 1.0): {weights.sum(dim=-1)}")

print("\nLeer asi: weights[i, j] = 'cuanta atencion pone el token i al token j'")
print(f"\nEjemplo: el token '{sentence[0]}' atiende:")
for j, w in enumerate(weights[0]):
    print(f"  a '{sentence[j]:9}' con peso {w.item():.3f}")


# ---------------------------------------------------------------------------
# Paso 6: la salida — suma ponderada de embeddings
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 6: Output = weights @ X (suma ponderada de embeddings)")
print("=" * 70)

output = weights @ X  # (T, T) @ (T, d_model) = (T, d_model)
print(f"\noutput shape: {output.shape}")
print(f"output:\n{output}")

print("\nCada fila output[i] es una NUEVA representacion del token i, calculada")
print("como suma ponderada de los embeddings de TODOS los tokens (incluido el)")
print("propio. Esa es self-attention en su forma mas pura.\n")

# Para verlo concreto: reconstruimos output[0] manualmente.
manual_output_0 = torch.zeros(d_model)
for j in range(T):
    manual_output_0 += weights[0, j] * X[j]
print(f"Reconstruccion manual de output[0] (suma ponderada):")
print(f"  output[0]        = {output[0]}")
print(f"  reconstruccion   = {manual_output_0}")
assert torch.allclose(output[0], manual_output_0)
print("\nOK: output[0] es exactamente sum_j weights[0,j] * X[j].")


# ---------------------------------------------------------------------------
# Paso 7: la formula completa en una linea
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 7: Toda la self-attention 'degenerada' en una linea")
print("=" * 70)

attention_output = torch.softmax(X @ X.T, dim=-1) @ X
print(f"\nattention_output = softmax(X @ X.T, dim=-1) @ X")
print(f"shape: {attention_output.shape}")
print(f"valores:\n{attention_output}")
print(f"\nIgual al output del paso 6: {torch.allclose(output, attention_output)}")


# ---------------------------------------------------------------------------
# Paso 8: por que esto NO basta todavia
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 8: Limitaciones de esta version 'degenerada'")
print("=" * 70)

print("""
1. Q = K = V = X. Cada token usa SU MISMO embedding como query, key y value.
   Eso limita la expresividad: el modelo no puede aprender a "preguntar una
   cosa" y "exponer otra". El siguiente paso introduce proyecciones aprendibles
   W_Q, W_K, W_V que hacen Q = X W_Q, K = X W_K, V = X W_V.

2. Sin scaling. Cuando d_model crece (en Vaswani es 512), los productos punto
   crecen en varianza como d_model. El softmax se vuelve casi one-hot
   (saturacion) y el gradiente desaparece. Por eso se divide por sqrt(d_k).

3. Una sola "cabeza". El modelo solo puede aprender UNA forma de relacionar
   tokens. Con multi-head ejecuta h atenciones en paralelo, cada una en un
   subespacio distinto.

4. Sin masking. En un decoder autoregresivo, el token i NO debe ver tokens
   futuros j > i. Eso se hace poniendo -inf en la triangular superior antes
   del softmax.

Todo eso lo arreglamos en escalones 2, 3 y 4.
""")

print("=" * 70)
print("FIN escalon 1.")
print("=" * 70)
