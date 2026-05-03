"""40_special_tokens.py - Cap 40: [CLS], [MASK], [SEP] en accion."""
from _bpe import BPETokenizer

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

print("=== Special tokens BERT ===\n")
print(f"[CLS]  id={tok.cls_id}  — Classification token (inicio de secuencia)")
print(f"[SEP]  id={tok.sep_id}  — Separator token (fin de segmento)")
print(f"[MASK] id={tok.mask_id} — Mask token (reemplaza tokens en MLM)")
print(f"\nVocab size antes: 1112  | despues: {tok.vocab_size}")

sentences = [
    "To be or not to be",
    "En un lugar de la Mancha",
]
print("\n=== encode_bert vs encode regular ===\n")
for s in sentences:
    regular = tok.encode(s)
    bert = tok.encode_bert(s)
    print(f"Texto:   {s!r}")
    print(f"Regular: {regular[:5]}... ({len(regular)} tokens)")
    print(f"BERT:    {bert[:5]}... ({len(bert)} tokens)  <- +2 ([CLS] y [SEP])")
    print(f"Decode:  {tok.decode(bert)!r}\n")

print("=== Rol de cada token ===")
print("""
[CLS] — Classification Token:
  Posicion 0 de CADA input BERT.
  El vector de salida de [CLS] despues de pasar por los N bloques
  representa TODA la secuencia. Es este vector el que va a la
  cabeza de clasificacion en fine-tuning. No tiene contenido
  semantico propio — aprende a ser un "resumen" del input.

[SEP] — Separator Token:
  Indica el fin del input (o separacion entre dos frases en BERT original).
  En nuestro caso de una sola frase: marca el fin.

[MASK] — Mask Token:
  Reemplaza tokens durante pretraining MLM.
  El modelo aprende a predecir el token original dado el contexto.
  NUNCA aparece en fine-tuning — es exclusivo del pretraining.
""")
