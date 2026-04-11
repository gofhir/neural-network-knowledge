"""
Ejemplo 16 — Entrenar una red para clasificar TEXTO (sentimiento)

Objetivo: entrenar una red que lea frases y diga si son POSITIVAS o NEGATIVAS.
          Usa LayerNorm (no BatchNorm) porque es texto, no imagenes.
          Compara LayerNorm vs BatchNorm vs sin normalizacion.

Genera graficos en /app/output/:
  - 20_texto_pipeline.png
  - 21_texto_embedding_aprendido.png
  - 22_texto_layernorm_vs_batchnorm.png
  - 23_texto_predicciones.png

Ejecutar:
  docker run --rm -v $(pwd)/output:/app/output clase7-pytorch \
    python -u ejemplos/transformaciones/16_entrenar_texto_paso_a_paso.py
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = "/app/output"


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 1: PREPARAR LOS DATOS (frases con sentimiento)   ║
# ╚══════════════════════════════════════════════════════════╝
print("=" * 60)
print("PASO 1: PREPARAR LOS DATOS")
print("=" * 60)

print("""
  Vamos a clasificar frases en POSITIVAS (1) o NEGATIVAS (0).
  Primero creamos un dataset simple para entender cada paso.
""")

# Dataset de frases con sentimiento
train_sentences = [
    # Positivas (label = 1)
    ("la pelicula es muy buena", 1),
    ("me encanta esta comida", 1),
    ("es un dia hermoso", 1),
    ("que alegria verte", 1),
    ("la musica es genial", 1),
    ("me gusta mucho este lugar", 1),
    ("es una obra increible", 1),
    ("la comida esta deliciosa", 1),
    ("que bonito dia", 1),
    ("me encanta esta musica", 1),
    ("es muy divertido", 1),
    ("la pelicula es increible", 1),
    ("me gusta esta cancion", 1),
    ("que hermoso lugar", 1),
    ("la comida es buena", 1),
    # Negativas (label = 0)
    ("la pelicula es muy mala", 0),
    ("no me gusta esta comida", 0),
    ("es un dia horrible", 0),
    ("que tristeza", 0),
    ("la musica es terrible", 0),
    ("no me gusta este lugar", 0),
    ("es una obra aburrida", 0),
    ("la comida esta fea", 0),
    ("que feo dia", 0),
    ("no me gusta esta musica", 0),
    ("es muy aburrido", 0),
    ("la pelicula es terrible", 0),
    ("no me gusta esta cancion", 0),
    ("que horrible lugar", 0),
    ("la comida es mala", 0),
]

test_sentences = [
    ("me encanta este dia", 1),
    ("la musica es muy buena", 1),
    ("que bonita comida", 1),
    ("es un lugar increible", 1),
    ("me gusta la pelicula", 1),
    ("no me gusta la comida", 0),
    ("es un dia feo", 0),
    ("que musica terrible", 0),
    ("la pelicula es aburrida", 0),
    ("es un lugar horrible", 0),
]

print(f"  Datos de entrenamiento: {len(train_sentences)} frases")
print(f"  Datos de prueba:        {len(test_sentences)} frases")
print(f"\n  Ejemplos:")
for sent, label in train_sentences[:3]:
    print(f"    '{sent}' → {'POSITIVA' if label else 'NEGATIVA'}")
print(f"    ...")
for sent, label in train_sentences[-3:]:
    print(f"    '{sent}' → {'POSITIVA' if label else 'NEGATIVA'}")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 2: TOKENIZAR (palabras → numeros)                ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 2: TOKENIZAR (palabras → IDs)")
print("=" * 60)

# Construir vocabulario a partir de los datos
all_words = set()
for sent, _ in train_sentences + test_sentences:
    for word in sent.split():
        all_words.add(word)

# Crear vocabulario: palabra → ID
vocab = {"<pad>": 0, "<unk>": 1}
for word in sorted(all_words):
    vocab[word] = len(vocab)

print(f"\n  Vocabulario: {len(vocab)} palabras")
print(f"  Primeras 15: {dict(list(vocab.items())[:15])}")

# Tokenizar todas las frases
def tokenize(sentence, vocab, max_len):
    """Convierte frase a lista de IDs, con padding."""
    ids = [vocab.get(word, vocab["<unk>"]) for word in sentence.split()]
    # Padding: rellenar con 0 hasta max_len (todas deben tener el mismo largo)
    while len(ids) < max_len:
        ids.append(vocab["<pad>"])
    return ids[:max_len]  # cortar si es mas larga

max_len = 7  # largo maximo de frase (en tokens)

print(f"\n  Largo maximo: {max_len} tokens")
print(f"  Si la frase es mas corta, se rellena con <pad> (0)")
print(f"\n  Ejemplos de tokenizacion:")

for sent, label in train_sentences[:3]:
    tokens = tokenize(sent, vocab, max_len)
    words = sent.split()
    print(f"    '{sent}'")
    print(f"     → {tokens}")
    mapping = ", ".join(f"'{w}'={vocab.get(w, 1)}" for w in words)
    print(f"     ({mapping})")
    print()

# Tokenizar todo
def prepare_data(sentences, vocab, max_len):
    """Convierte lista de (frase, label) en tensores."""
    X = torch.tensor([tokenize(s, vocab, max_len) for s, _ in sentences])
    y = torch.tensor([label for _, label in sentences])
    return X, y

X_train, y_train = prepare_data(train_sentences, vocab, max_len)
X_test, y_test = prepare_data(test_sentences, vocab, max_len)

print(f"  Tensor de entrenamiento: {X_train.shape}")
print(f"    → {X_train.shape[0]} frases, {X_train.shape[1]} tokens cada una")
print(f"  Etiquetas: {y_train.tolist()}")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 3: CONSTRUIR LA RED CON LAYERNORM                ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 3: CONSTRUIR LA RED (con LayerNorm)")
print("=" * 60)

print("""
  Para TEXTO se usa LayerNorm, NO BatchNorm. ¿Por que?

  1. Las frases tienen LARGOS DISTINTOS
     "hola" (1 token) vs "me encanta esta pelicula" (4 tokens)
     BatchNorm necesita normalizar la posicion 4 de TODAS las frases,
     pero "hola" no TIENE posicion 4.

  2. En inferencia se procesa de a 1 frase (batch=1)
     BatchNorm necesita un batch grande para calcular bien.
     LayerNorm funciona con 1 sola frase.

  3. Cada TOKEN se normaliza INDEPENDIENTEMENTE
     "gato" se normaliza usando sus propios features,
     sin importar que otros tokens estan en la frase.

  Pipeline:
    Tokens [2, 5, 8, 11, 0, 0, 0]
       ↓  Embedding
    Vectores (7, 32) → cada token es un vector de 32 numeros
       ↓  LayerNorm  ← normaliza cada token por separado
       ↓  Red
    Promedio de todos los tokens → un solo vector (32)
       ↓  Linear
    Salida: 2 clases (positivo / negativo)
""")

embedding_dim = 32
hidden_dim = 64


class TextClassifier_LayerNorm(nn.Module):
    """Red para clasificar texto usando LayerNorm."""

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        # Embedding: token ID → vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Capa 1: proyeccion + LayerNorm
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)       # ← LAYERNORM (no BatchNorm!)
        self.dropout1 = nn.Dropout(0.3)

        # Capa 2: otra proyeccion + LayerNorm
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)        # ← LAYERNORM
        self.dropout2 = nn.Dropout(0.3)

        # Salida
        self.fc_out = nn.Linear(hidden_dim, 2)     # 2 clases: positivo/negativo

    def forward(self, x):
        # x shape: (batch, seq_len) → IDs de tokens

        # Paso 1: Embedding
        x = self.embedding(x)         # (batch, seq_len, embed_dim)

        # Paso 2: Linear + LayerNorm + ReLU (en cada token)
        x = self.fc1(x)               # (batch, seq_len, hidden_dim)
        x = self.ln1(x)               # LayerNorm: normaliza cada token
        x = torch.relu(x)
        x = self.dropout1(x)

        # Paso 3: Otra capa
        x = self.fc2(x)
        x = self.ln2(x)               # LayerNorm de nuevo
        x = torch.relu(x)
        x = self.dropout2(x)

        # Paso 4: Promediar todos los tokens → un solo vector
        # (ignoramos los tokens de padding)
        mask = (x != 0).any(dim=-1, keepdim=True).float()  # 1 donde hay token, 0 en padding
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # promedio

        # Paso 5: Clasificar
        x = self.fc_out(x)            # (batch, 2)
        return x


model_ln = TextClassifier_LayerNorm(len(vocab), embedding_dim, hidden_dim)

# Prueba rapida
test_output = model_ln(X_train[:2])
print(f"  Modelo creado: TextClassifier_LayerNorm")
print(f"  Parametros: {sum(p.numel() for p in model_ln.parameters()):,}")
print(f"\n  Prueba rapida:")
print(f"    Entrada: {X_train[:2].shape} (2 frases, 7 tokens)")
print(f"    Salida:  {test_output.shape} (2 frases, 2 clases)")
print(f"    Valores: {[round(v, 4) for v in test_output[0].tolist()]}")
print(f"    → Aleatorios, porque no ha entrenado")

# Mostrar la diferencia LayerNorm vs BatchNorm en el forward
print(f"\n  ¿Donde esta LayerNorm? En el forward:")
print(f"    x = self.fc1(x)    # Linear: (batch, 7, 32) → (batch, 7, 64)")
print(f"    x = self.ln1(x)    # LayerNorm: normaliza CADA TOKEN (64 features)")
print(f"                       # Token 'gato': media=0, var=1 sobre sus 64 features")
print(f"                       # Token 'come': media=0, var=1 sobre sus 64 features")
print(f"                       # Cada token se normaliza INDEPENDIENTEMENTE")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 4: ENTRENAR                                      ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 4: ENTRENAR")
print("=" * 60)

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.005):
    """Entrena un modelo y devuelve el historial."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Estadisticas
        _, predicted = predictions.max(1)
        train_acc = (predicted == y_train).float().mean().item() * 100

        # Evaluacion
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            _, test_predicted = test_pred.max(1)
            test_acc = (test_predicted == y_test).float().mean().item() * 100

        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoca {epoch+1:3d}/{epochs}: "
                  f"Loss={loss.item():.4f}, "
                  f"Train Acc={train_acc:.0f}%, "
                  f"Test Acc={test_acc:.0f}%")

    return history

print(f"\n  Entrenando modelo con LayerNorm...\n")
history_ln = train_model(model_ln, X_train, y_train, X_test, y_test, epochs=100)


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 5: COMPARAR LayerNorm vs BatchNorm vs Nada        ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 5: COMPARAR LayerNorm vs BatchNorm vs Sin normalizacion")
print("=" * 60)

print("""
  Entrenamos 3 redes identicas, solo cambiando la normalizacion:
    1. LayerNorm   (lo correcto para texto)
    2. BatchNorm   (lo correcto para imagenes, NO para texto)
    3. Sin nada    (sin normalizacion)
""")


class TextClassifier_BatchNorm(nn.Module):
    """Misma red pero con BatchNorm (NO ideal para texto)."""

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)      # ← BATCHNORM
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        # Promediar tokens ANTES de BatchNorm
        # (porque BatchNorm1d espera (batch, features), no (batch, seq, features))
        mask = (x != 0).any(dim=-1, keepdim=True).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        x = self.fc1(x)
        x = self.bn1(x)               # BatchNorm: normaliza a traves del BATCH
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc_out(x)
        return x


class TextClassifier_NoNorm(nn.Module):
    """Misma red SIN normalizacion."""

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        mask = (x != 0).any(dim=-1, keepdim=True).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        x = self.fc1(x)
        x = torch.relu(x)             # SIN normalizacion
        x = self.dropout1(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        x = self.fc_out(x)
        return x


# Entrenar las 3
torch.manual_seed(42)
model_bn = TextClassifier_BatchNorm(len(vocab), embedding_dim, hidden_dim)
print(f"  Entrenando con BatchNorm...")
history_bn = train_model(model_bn, X_train, y_train, X_test, y_test, epochs=100)

torch.manual_seed(42)
model_none = TextClassifier_NoNorm(len(vocab), embedding_dim, hidden_dim)
print(f"\n  Entrenando sin normalizacion...")
history_none = train_model(model_none, X_train, y_train, X_test, y_test, epochs=100)

# Grafico comparativo
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

epochs_range = range(1, 101)

ax1.plot(epochs_range, history_ln['train_loss'], label='LayerNorm', color='steelblue', linewidth=1.5)
ax1.plot(epochs_range, history_bn['train_loss'], label='BatchNorm', color='coral', linewidth=1.5)
ax1.plot(epochs_range, history_none['train_loss'], label='Sin normalizacion', color='gray', linewidth=1.5)
ax1.set_xlabel('Epoca')
ax1.set_ylabel('Loss')
ax1.set_title('Loss durante entrenamiento\n(mas bajo = aprende mas rapido)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history_ln['test_acc'], label='LayerNorm', color='steelblue', linewidth=2)
ax2.plot(epochs_range, history_bn['test_acc'], label='BatchNorm', color='coral', linewidth=2)
ax2.plot(epochs_range, history_none['test_acc'], label='Sin normalizacion', color='gray', linewidth=2)
ax2.set_xlabel('Epoca')
ax2.set_ylabel('Test Accuracy (%)')
ax2.set_title('Precision en datos de prueba\n(mas alto = mejor)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle("Comparacion: LayerNorm vs BatchNorm vs Sin normalizacion\npara clasificacion de TEXTO",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/22_texto_layernorm_vs_batchnorm.png", dpi=150)
plt.close()
print(f"\n  Guardado: {OUTPUT}/22_texto_layernorm_vs_batchnorm.png")

print(f"\n  Resultados finales:")
print(f"    LayerNorm:          Train={history_ln['train_acc'][-1]:.0f}%,  Test={history_ln['test_acc'][-1]:.0f}%")
print(f"    BatchNorm:          Train={history_bn['train_acc'][-1]:.0f}%,  Test={history_bn['test_acc'][-1]:.0f}%")
print(f"    Sin normalizacion:  Train={history_none['train_acc'][-1]:.0f}%,  Test={history_none['test_acc'][-1]:.0f}%")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 6: VER QUE APRENDIO EL EMBEDDING                 ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 6: QUE APRENDIO EL EMBEDDING")
print("=" * 60)

print("""
  Despues de entrenar, los embeddings APRENDEN relaciones:
  - Palabras positivas deberian estar CERCA entre si
  - Palabras negativas deberian estar CERCA entre si
  - Positivas y negativas deberian estar LEJOS
""")

# Obtener embeddings de palabras interesantes
words_positive = ["buena", "genial", "increible", "hermoso", "deliciosa", "encanta", "bonito", "divertido"]
words_negative = ["mala", "terrible", "horrible", "fea", "aburrida", "tristeza", "feo", "aburrido"]

all_words_plot = words_positive + words_negative
word_ids = [vocab.get(w, vocab["<unk>"]) for w in all_words_plot]
word_tensors = torch.tensor(word_ids)

with torch.no_grad():
    embeddings = model_ln.embedding(word_tensors).numpy()  # (16, 32)

# Reducir a 2D con PCA simple (para poder graficarlo)
mean = embeddings.mean(axis=0)
embeddings_centered = embeddings - mean
U, S, Vt = np.linalg.svd(embeddings_centered, full_matrices=False)
embeddings_2d = embeddings_centered @ Vt[:2].T  # proyectar a 2 dimensiones

# Grafico
fig, ax = plt.subplots(figsize=(10, 7))

# Positivas en azul
for i, word in enumerate(words_positive):
    ax.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
              color='steelblue', s=100, zorder=5)
    ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
               fontsize=11, fontweight='bold', color='steelblue',
               xytext=(5, 5), textcoords='offset points')

# Negativas en rojo
n_pos = len(words_positive)
for i, word in enumerate(words_negative):
    ax.scatter(embeddings_2d[n_pos + i, 0], embeddings_2d[n_pos + i, 1],
              color='coral', s=100, zorder=5)
    ax.annotate(word, (embeddings_2d[n_pos + i, 0], embeddings_2d[n_pos + i, 1]),
               fontsize=11, fontweight='bold', color='coral',
               xytext=(5, 5), textcoords='offset points')

ax.set_title("Embeddings aprendidos (proyectados a 2D)\nAzul = positivas, Rojo = negativas", fontsize=13)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/21_texto_embedding_aprendido.png", dpi=150)
plt.close()
print(f"  Guardado: {OUTPUT}/21_texto_embedding_aprendido.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 7: PREDECIR FRASES NUEVAS                         ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("PASO 7: PREDECIR FRASES NUEVAS")
print("=" * 60)

new_sentences = [
    "la pelicula es buena",
    "me encanta la comida",
    "que dia hermoso",
    "la pelicula es mala",
    "no me gusta",
    "que dia horrible",
    "me gusta mucho",
    "es terrible",
]

model_ln.eval()
results = []

print(f"\n  Predicciones del modelo:\n")
with torch.no_grad():
    for sent in new_sentences:
        tokens = torch.tensor([tokenize(sent, vocab, max_len)])
        output = model_ln(tokens)
        probs = torch.softmax(output, dim=1)[0]
        predicted = output.argmax(1).item()
        confidence = probs[predicted].item()
        label = "POSITIVA" if predicted == 1 else "NEGATIVA"
        results.append((sent, label, confidence))
        print(f"    '{sent}'")
        print(f"      → {label} (confianza: {confidence:.1%})")
        print()

# Grafico de predicciones
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

colors = ['#E2EFDA' if r[1] == 'POSITIVA' else '#FCE4EC' for r in results]

table_data = [["Frase", "Prediccion", "Confianza"]]
for sent, label, conf in results:
    table_data.append([sent, label, f"{conf:.0%}"])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.45, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Header
for j in range(3):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Colores por fila
for i, (_, label, _) in enumerate(results):
    color = '#E2EFDA' if label == 'POSITIVA' else '#FCE4EC'
    for j in range(3):
        table[i + 1, j].set_facecolor(color)

ax.set_title("Predicciones del modelo en frases nuevas", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/23_texto_predicciones.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: {OUTPUT}/23_texto_predicciones.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  PASO 8: PIPELINE VISUAL                                ║
# ╚══════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(1, 5, figsize=(18, 3.5),
                          gridspec_kw={'width_ratios': [2.5, 2, 3, 2, 1.5]})

# 1. Frase
axes[0].text(0.5, 0.5, '"me encanta\nesta comida"', ha='center', va='center',
             fontsize=13, fontweight='bold', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange'))
axes[0].set_title("1. Texto", fontsize=11)
axes[0].axis('off')

# 2. Tokens
axes[1].text(0.5, 0.5, '[12, 7, 8, 5,\n  0, 0, 0]', ha='center', va='center',
             fontsize=12, family='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange'))
axes[1].set_title("2. Tokenizar", fontsize=11)
axes[1].axis('off')

# 3. Embeddings + LayerNorm
axes[2].text(0.5, 0.65, 'Embedding\n(7 tokens × 32 dims)', ha='center', va='center', fontsize=10)
axes[2].text(0.5, 0.35, '↓ LayerNorm ↓\n(normaliza cada token)', ha='center', va='center',
             fontsize=10, color='steelblue', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='steelblue'))
axes[2].set_title("3. Embed + Norm", fontsize=11)
axes[2].axis('off')

# 4. Red
axes[3].text(0.5, 0.5, 'Linear(64)\nReLU\nLinear(2)', ha='center', va='center',
             fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', edgecolor='red', alpha=0.3))
axes[3].set_title("4. Clasificar", fontsize=11)
axes[3].axis('off')

# 5. Resultado
axes[4].text(0.5, 0.5, 'POSITIVA\n95%', ha='center', va='center',
             fontsize=13, fontweight='bold', color='green',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#E2EFDA', edgecolor='green'))
axes[4].set_title("5. Resultado", fontsize=11)
axes[4].axis('off')

# Flechas
for i in range(4):
    fig.text(0.19 + i * 0.185, 0.45, '→', fontsize=20, ha='center', va='center')

plt.suptitle("Pipeline completo: Texto → Prediccion (con LayerNorm)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT}/20_texto_pipeline.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Guardado: {OUTPUT}/20_texto_pipeline.png")


# ╔══════════════════════════════════════════════════════════╗
# ║  RESUMEN                                                ║
# ╚══════════════════════════════════════════════════════════╝
print(f"\n{'=' * 60}")
print("RESUMEN: Imagen vs Texto")
print("=" * 60)
print(f"""
  IMAGENES (ejemplo 14-15):              TEXTO (este ejemplo):
  ─────────────────────                  ──────────────────────
  Entrada: pixeles [0-1]                 Entrada: palabras
  Preproceso: /255                       Preproceso: tokenizar + embedding
  Normalizacion: BatchNorm               Normalizacion: LayerNorm
  Red: CNN (Conv2d)                      Red: Linear (o Transformer)
  Salida: 10 digitos                     Salida: positivo/negativo

  ¿Por que LayerNorm para texto?
    - Frases de largo variable → LayerNorm no depende del batch
    - Inferencia con 1 frase   → LayerNorm funciona con batch=1
    - Cada token independiente → LayerNorm normaliza por token
    - Transformers lo usan     → GPT, BERT, todos usan LayerNorm

  ¿Por que BatchNorm para imagenes?
    - Todas las imagenes mismo tamano → batch consistente
    - Cada canal tiene significado fijo → normalizar por canal tiene sentido
    - CNNs lo usan → ResNet, VGG, todos usan BatchNorm

  Graficos guardados en {OUTPUT}/
""")
