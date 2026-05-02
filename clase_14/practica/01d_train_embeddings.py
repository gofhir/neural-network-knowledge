"""
Demo: entrenar embeddings reales con un mini-Word2Vec.

Objetivo: ver con tus ojos como una matriz de embeddings RANDOM se convierte,
despues de entrenar, en una matriz donde palabras semanticamente relacionadas
tienen dot products altos entre si.

Tarea: skip-gram simplificado. Le mostramos al modelo pares de palabras que
co-ocurren en un mini-corpus, y le pedimos que prediga, dada una palabra,
cual es la otra.

  Input al modelo: "perro" (un id)
  Target:          "ladra" (otro id)
  Loss:            cross_entropy(logits sobre vocab, target)

Despues de entrenar, los embeddings de palabras que co-ocurren con las mismas
otras palabras quedan cerca en el espacio vectorial.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
torch.set_printoptions(precision=3, sci_mode=False)


# ---------------------------------------------------------------------------
# Vocabulario y mini-corpus
# ---------------------------------------------------------------------------
print("=" * 70)
print("PASO 1: Vocabulario y mini-corpus de entrenamiento")
print("=" * 70)

vocab = ["perro", "gato", "ladra", "maulla", "avion", "vuela"]
word_to_id = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
d_model = 8

# Mini-corpus: pares de palabras que co-ocurren en oraciones imaginarias.
# Las parejas reflejan: dominio mascotas (perro, gato, ladra, maulla)
# vs dominio aviacion (avion, vuela).
co_occurrences = [
    ("perro", "ladra"),    # perro -> ladra
    ("gato", "maulla"),
    ("perro", "gato"),     # ambos son mascotas
    ("ladra", "maulla"),   # ambos son sonidos de mascotas
    ("avion", "vuela"),
]

# Cada pareja se entrena en ambas direcciones (skip-gram)
pairs = []
for a, b in co_occurrences:
    pairs.append((a, b))
    pairs.append((b, a))

print(f"\nVocabulario: {vocab}")
print(f"d_model:     {d_model}")
print(f"\nPares de entrenamiento (input -> target):")
for a, b in pairs:
    print(f"  {a:>8s} -> {b}")

input_ids = torch.tensor([word_to_id[a] for a, _ in pairs])
target_ids = torch.tensor([word_to_id[b] for _, b in pairs])
print(f"\ninput_ids  = {input_ids.tolist()}")
print(f"target_ids = {target_ids.tolist()}")


# ---------------------------------------------------------------------------
# Modelo: embedding + proyeccion lineal de salida
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 2: Modelo (embedding + proyeccion de salida)")
print("=" * 70)

class TinyEmbeddingModel(nn.Module):
    """
    Arquitectura mas simple posible:
      input_id -> embedding -> linear -> logits sobre vocab

    El embedding es la matriz que queremos que aprenda relaciones semanticas.
    La linear de salida proyecta de d_model a vocab_size para hacer cross-entropy.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids):
        x = self.embedding(ids)  # (batch, d_model)
        return self.output(x)    # (batch, vocab_size) = logits

model = TinyEmbeddingModel(vocab_size, d_model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

print(f"\nParametros del modelo:")
total = 0
for name, p in model.named_parameters():
    print(f"  {name:25s} shape {tuple(p.shape)}  ({p.numel()} params)")
    total += p.numel()
print(f"  Total: {total} parametros")


# ---------------------------------------------------------------------------
# Helper: imprimir matriz de dot products entre embeddings
# ---------------------------------------------------------------------------
def print_dot_matrix(emb, title):
    scores = emb @ emb.T
    print(f"\n{title}")
    header = "          " + "".join(f"{w:>9s}" for w in vocab)
    print(header)
    for i, w in enumerate(vocab):
        row = "".join(f"{scores[i, j].item():>9.2f}" for j in range(vocab_size))
        print(f"  {w:>8s}:{row}")


# ---------------------------------------------------------------------------
# ANTES de entrenar: matriz de dot products es random
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 3: Embeddings ANTES de entrenar (random)")
print("=" * 70)

emb_before = model.embedding.weight.detach().clone()
print_dot_matrix(emb_before, "Dot products ANTES (random):")
print("\nLeer asi: scores[i, j] = embedding[i] . embedding[j].")
print("Como los vectores son random, no hay patron de cercania semantica.")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 4: Training con cross-entropy + Adam")
print("=" * 70)

print(f"\n{'epoch':>6} | {'loss':>8}")
print("-" * 18)
n_epochs = 300
for epoch in range(n_epochs):
    logits = model(input_ids)               # forward
    loss = F.cross_entropy(logits, target_ids)  # loss

    optimizer.zero_grad()                   # limpiar gradientes
    loss.backward()                         # BACKPROP
    optimizer.step()                        # ajustar pesos

    if epoch % 30 == 0 or epoch == n_epochs - 1:
        print(f"{epoch:>6} | {loss.item():>8.4f}")


# ---------------------------------------------------------------------------
# DESPUES de entrenar: las palabras relacionadas tienen dot products altos
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 5: Embeddings DESPUES de entrenar")
print("=" * 70)

emb_after = model.embedding.weight.detach()
print_dot_matrix(emb_after, "Dot products DESPUES (entrenados):")


# ---------------------------------------------------------------------------
# Comparaciones especificas: que aprendio el modelo?
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 6: Verificacion - palabras relacionadas vs no relacionadas")
print("=" * 70)

def dot(a, b):
    """Producto punto entre los embeddings de dos palabras."""
    va = emb_after[word_to_id[a]]
    vb = emb_after[word_to_id[b]]
    return (va * vb).sum().item()

print("\nDot products entre palabras del MISMO dominio (deberian ser altos):")
print(f"  perro  . gato       = {dot('perro', 'gato'):>7.2f}   (ambos mascotas)")
print(f"  ladra  . maulla     = {dot('ladra', 'maulla'):>7.2f}   (sonidos de mascotas)")
print(f"  perro  . ladra      = {dot('perro', 'ladra'):>7.2f}   (perro-ladra es par directo)")
print(f"  avion  . vuela      = {dot('avion', 'vuela'):>7.2f}   (avion-volar)")

print("\nDot products entre palabras de DOMINIOS DISTINTOS (deberian ser bajos):")
print(f"  perro  . avion      = {dot('perro', 'avion'):>7.2f}   (sin relacion)")
print(f"  gato   . vuela      = {dot('gato', 'vuela'):>7.2f}   (sin relacion)")
print(f"  ladra  . avion      = {dot('ladra', 'avion'):>7.2f}   (sin relacion)")
print(f"  maulla . vuela      = {dot('maulla', 'vuela'):>7.2f}   (sin relacion)")


# ---------------------------------------------------------------------------
# Predicciones del modelo entrenado
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("PASO 7: Predicciones del modelo entrenado")
print("=" * 70)

print("\nDada una palabra, que es lo mas probable que diga el modelo a continuacion?")
for word in vocab:
    input_id = torch.tensor([word_to_id[word]])
    with torch.no_grad():
        logits = model(input_id)
        probs = F.softmax(logits, dim=-1).squeeze()

    sorted_probs, sorted_ids = probs.sort(descending=True)
    top3 = []
    for p, i in zip(sorted_probs[:3], sorted_ids[:3]):
        top3.append(f"{vocab[i.item()]} ({p.item():.2f})")
    print(f"  {word:>8s} -> {', '.join(top3)}")


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)

print("""
Lo que viste:

1. Empezamos con 6 vectores RANDOM (la matriz embedding inicial).
2. Le dimos al modelo una tarea: "dada una palabra, predice la
   co-ocurrente". Cross-entropy + Adam ajustaron los embeddings.
3. Despues de 300 iteraciones, la matriz de dot products dejo de ser
   random: palabras del mismo dominio (perro/gato, ladra/maulla, avion/vuela)
   tienen dot products ALTOS, y las de dominios distintos tienen
   dot products BAJOS o NEGATIVOS.

Esto es lo mismo que pasa al entrenar BERT, GPT, Word2Vec con miles de
millones de palabras: la geometria del espacio de embeddings emerge
sola del objetivo de minimizar el loss.

Todo lo que viste:
  - Embeddings (matriz de pesos)            -> escalon 01
  - Dot product (similitud)                 -> escalon 01
  - Cross-entropy (loss)                    -> escalon 01b
  - Backprop / Adam (ajustar pesos)         -> escalon 01c
  - Training loop completo                  -> ESTE escalon 01d

Ahora ya conoces TODO el ciclo de entrenamiento de un modelo de lenguaje.
La diferencia entre este mini-W2V y un Transformer real es la complejidad
del modelo (self-attention, multi-head, layers profundas), no del proceso
de aprendizaje en si.
""")

print("=" * 70)
print("FIN escalon 01d.")
print("=" * 70)
