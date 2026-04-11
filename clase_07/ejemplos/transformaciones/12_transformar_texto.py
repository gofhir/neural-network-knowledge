"""
Ejemplo 12 — Como se transforma TEXTO en numeros

Objetivo: ver paso a paso como una frase se convierte
          en un tensor de numeros que la red puede procesar.

Genera graficos en /app/output/:
  - 05_texto_tokenizacion.png
  - 06_texto_embedding.png
  - 07_texto_similitud.png

Ejecutar:
  docker run --rm -v $(pwd)/output:/app/output clase7-pytorch \
    python -u ejemplos/transformaciones/12_transformar_texto.py
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = "/app/output"

# =============================================
# 1. El problema: las redes no entienden palabras
# =============================================
print("=" * 60)
print("PASO 1: El problema")
print("=" * 60)

print(f"""
  Una red neuronal SOLO puede recibir numeros.
  No puede recibir la palabra "gato" directamente.

  Necesitamos convertir:
    "el gato come pescado"  →  [[0.2, -0.1, ...], [1.1, 0.4, ...], ...]

  Se hace en 2 pasos:
    Paso A: Tokenizar  (palabra → numero ID)
    Paso B: Embedding  (numero ID → vector de numeros)
""")

# =============================================
# 2. Paso A: Tokenizar (palabra → ID)
# =============================================
print("=" * 60)
print("PASO 2: Tokenizar (palabra → ID numerico)")
print("=" * 60)

# Vocabulario simple (en GPT se usa BPE con ~50,000 tokens)
vocab = {
    "<pad>": 0, "<unk>": 1,
    "el": 2, "la": 3, "un": 4,
    "gato": 5, "perro": 6, "pajaro": 7,
    "come": 8, "duerme": 9, "vuela": 10,
    "pescado": 11, "mucho": 12, "rapido": 13,
    "grande": 14, "pequeno": 15,
}

print(f"\n  Vocabulario ({len(vocab)} palabras):")
for word, idx in vocab.items():
    print(f"    '{word}' → {idx}")

# Tokenizar frases
sentences = [
    "el gato come pescado",
    "el perro duerme mucho",
    "el pajaro vuela rapido",
]

print(f"\n  Tokenizacion:")
tokenized = []
for s in sentences:
    ids = [vocab.get(word, vocab["<unk>"]) for word in s.split()]
    tokenized.append(ids)
    # Mostrar paso a paso
    words = s.split()
    mapping = " → ".join(f"'{w}'={vocab.get(w, 1)}" for w in words)
    print(f"    '{s}'")
    print(f"     → {ids}")
    print()

x_tokens = torch.tensor(tokenized)
print(f"  Resultado: tensor shape = {x_tokens.shape}")
print(f"    → {x_tokens.shape[0]} frases, {x_tokens.shape[1]} tokens cada una")
print(f"  Tensor:\n{x_tokens}")

# Grafico de tokenizacion
fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.Set3(np.linspace(0, 1, len(vocab)))

for i, (sent, ids) in enumerate(zip(sentences, tokenized)):
    words = sent.split()
    for j, (word, idx) in enumerate(zip(words, ids)):
        ax.add_patch(plt.Rectangle((j * 2.2, (2 - i) * 1.5), 2, 1,
                     facecolor=colors[idx], edgecolor='black', linewidth=1.5))
        ax.text(j * 2.2 + 1, (2 - i) * 1.5 + 0.6, f'"{word}"',
                ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(j * 2.2 + 1, (2 - i) * 1.5 + 0.25, f'ID = {idx}',
                ha='center', va='center', fontsize=9, color='gray')

ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-0.5, 4.5)
ax.set_title("Tokenizacion: cada palabra se convierte en un numero (ID)", fontsize=13)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/05_texto_tokenizacion.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Guardado: {OUTPUT}/05_texto_tokenizacion.png")

# =============================================
# 3. Paso B: Embedding (ID → vector)
# =============================================
print(f"\n{'=' * 60}")
print("PASO 3: Embedding (ID → vector de numeros)")
print("=" * 60)

print(f"""
  El ID es solo un indice. No captura SIGNIFICADO.
  'gato'=5 y 'perro'=6 son consecutivos por casualidad,
  no porque sean parecidos.

  El EMBEDDING convierte cada ID en un vector de numeros
  que SI captura significado. Estos vectores se APRENDEN
  durante el entrenamiento.
""")

embedding_dim = 4  # en GPT-3 son 12,288. Usamos 4 para poder visualizarlo
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)

# Fijar seed para reproducibilidad
torch.manual_seed(42)
embedding = nn.Embedding(len(vocab), embedding_dim)

x_embedded = embedding(x_tokens)
print(f"  nn.Embedding(vocab_size={len(vocab)}, embedding_dim={embedding_dim})")
print(f"\n  Entrada:  {x_tokens.shape}     → (frases, tokens)")
print(f"  Salida:   {x_embedded.shape}  → (frases, tokens, embedding_dim)")

print(f"\n  ¿Que paso? Cada ID se reemplazo por un vector de {embedding_dim} numeros:")
print()

words_first = sentences[0].split()
for j, word in enumerate(words_first):
    vec = x_embedded[0, j].detach()
    print(f"    '{word}' (ID={tokenized[0][j]})")
    print(f"      → [{', '.join(f'{v:.4f}' for v in vec)}]")
    print()

# Grafico del embedding
fig, ax = plt.subplots(figsize=(12, 5))

# Mostrar la tabla de embeddings
data = x_embedded[0].detach().numpy()  # primera frase
words = sentences[0].split()

im = ax.imshow(data, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
ax.set_yticks(range(len(words)))
ax.set_yticklabels([f'"{w}" (ID={tokenized[0][i]})' for i, w in enumerate(words)], fontsize=11)
ax.set_xticks(range(embedding_dim))
ax.set_xticklabels([f'dim {i}' for i in range(embedding_dim)], fontsize=10)
ax.set_title('"el gato come pescado" → cada palabra es un vector de numeros', fontsize=13)

# Mostrar valores en cada celda
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        color = 'white' if abs(data[i, j]) > 1.0 else 'black'
        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                fontsize=10, color=color)

plt.colorbar(im, ax=ax, label='Valor del embedding')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/06_texto_embedding.png", dpi=150)
plt.close()
print(f"  Guardado: {OUTPUT}/06_texto_embedding.png")

# =============================================
# 4. El poder del embedding: similitud
# =============================================
print(f"\n{'=' * 60}")
print("PASO 4: Similitud entre palabras")
print("=" * 60)

print(f"""
  El embedding APRENDE que palabras son parecidas.
  Despues de entrenar, los vectores de 'gato' y 'perro'
  estarian CERCA, y los de 'gato' y 'come' estarian LEJOS.

  Ahora los embeddings son ALEATORIOS (no entrenados),
  pero mostramos como se calcula la similitud:
""")

# Calcular similitud coseno entre todas las palabras del vocab
words_to_compare = ["gato", "perro", "pajaro", "come", "duerme", "vuela", "pescado"]
indices = [vocab[w] for w in words_to_compare]
vectors = embedding(torch.tensor(indices))

# Similitud coseno
similarity = torch.nn.functional.cosine_similarity(
    vectors.unsqueeze(0), vectors.unsqueeze(1), dim=2
)

print(f"  Similitud coseno (1.0 = identicos, -1.0 = opuestos):\n")
header = "          " + "  ".join(f"{w:>8s}" for w in words_to_compare)
print(header)
for i, w in enumerate(words_to_compare):
    row = f"  {w:>8s}" + "".join(f"  {similarity[i, j]:8.4f}" for j in range(len(words_to_compare)))
    print(row)

# Grafico de similitud
fig, ax = plt.subplots(figsize=(8, 6))
sim_np = similarity.detach().numpy()
im = ax.imshow(sim_np, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(len(words_to_compare)))
ax.set_xticklabels(words_to_compare, rotation=45, ha='right', fontsize=11)
ax.set_yticks(range(len(words_to_compare)))
ax.set_yticklabels(words_to_compare, fontsize=11)

for i in range(len(words_to_compare)):
    for j in range(len(words_to_compare)):
        ax.text(j, i, f'{sim_np[i, j]:.2f}', ha='center', va='center', fontsize=9)

plt.colorbar(im, ax=ax, label='Similitud coseno')
ax.set_title('Similitud entre palabras (embeddings sin entrenar)\nDespues de entrenar, gato-perro estarian mas cerca', fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/07_texto_similitud.png", dpi=150)
plt.close()
print(f"\n  Guardado: {OUTPUT}/07_texto_similitud.png")

print(f"\n  Nota: estos embeddings son ALEATORIOS porque no hemos entrenado.")
print(f"  Despues de entrenar, el modelo aprenderia que:")
print(f"    - 'gato' y 'perro' son parecidos (ambos animales)")
print(f"    - 'come' y 'duerme' son parecidos (ambos verbos)")
print(f"    - 'gato' y 'come' son distintos (sustantivo vs verbo)")

# =============================================
# RESUMEN
# =============================================
print(f"\n{'=' * 60}")
print("RESUMEN: Texto → Tensor")
print("=" * 60)
print(f"""
  "el gato come pescado"
    ↓  tokenizar (vocabulario)
  [2, 5, 8, 11]              → cada palabra es un ID
    ↓  embedding (tabla de vectores)
  [[0.2, -0.1, ...],         → cada ID es un vector
   [1.1,  0.4, ...],           de numeros que captura
   [0.7, -0.3, ...],           el significado
   [0.9,  0.1, ...]]
    ↓  shape = (1, 4, embedding_dim)
  Listo para la red

  En la practica:
    - GPT usa ~50,000 tokens y embeddings de 12,288 dimensiones
    - BERT usa ~30,000 tokens y embeddings de 768 dimensiones
    - Los embeddings se APRENDEN durante el entrenamiento

  Graficos guardados en {OUTPUT}/
""")
