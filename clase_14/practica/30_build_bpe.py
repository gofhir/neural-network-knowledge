"""30_build_bpe.py - Cap 30: BPE desde cero.

Entrena un BPETokenizer sobre Shakespeare + Quijote (~1MB bilingue).
Produce data/bpe_tokenizer.json con vocab ~1100 tokens.
"""
from pathlib import Path
from _bpe import BPETokenizer

NUM_MERGES = 1000

print("Cargando corpus bilingue (Shakespeare + Quijote)...")
en = Path("shakespeare.txt").read_text(encoding="utf-8")
es = Path("quijote.txt").read_text(encoding="utf-8")
corpus = en + "\n" + es
print(f"Corpus: {len(corpus):,} chars total (usando primeros 50,000 para training)")

tok = BPETokenizer()
print(f"\nEntrenando BPE con {NUM_MERGES} merges...")
import time
t0 = time.time()
tok.train(corpus, num_merges=NUM_MERGES)
elapsed = time.time() - t0
print(f"Listo en {elapsed:.1f}s")

print(f"\nVocab size: {tok.vocab_size} tokens")
print(f"Merges aprendidos: {len(tok.merges)}")

# Verificar que \n es un token propio (importante para stop_token en generacion)
newline_id = tok.vocab.get("\n")
newline_status = f"id={newline_id} — OK" if newline_id is not None else "AUSENTE — problema"
print(f"\nToken '\\n' en vocab: {newline_status}")

# Ejemplos de tokenizacion
examples = [
    "the king is dead",
    "To be or not to be",
    "En un lugar de la Mancha",
    "INSTR: repeat 'a' three",
    "Q: who wrote Hamlet?",
]
print("\n=== Ejemplos de tokenizacion ===")
for ex in examples:
    ids = tok.encode(ex)
    tokens = [tok.id_to_token[i] for i in ids]
    print(f"  '{ex}'")
    print(f"    chars={len(ex)}  tokens={len(ids)}  ratio={len(ids)/len(ex):.2f}")
    print(f"    tokens: {tokens}")

# Guardar
Path("data").mkdir(exist_ok=True)
tok.save("data/bpe_tokenizer.json")
print(f"\nSaved -> data/bpe_tokenizer.json")

# Verificar round-trip
tok2 = BPETokenizer.load("data/bpe_tokenizer.json")
sample = "To be or not to be"
assert tok.encode(sample) == tok2.encode(sample), "round-trip fallo"
print("Round-trip verificado.")
print(f"\nVocab final: {tok.vocab_size} tokens")
