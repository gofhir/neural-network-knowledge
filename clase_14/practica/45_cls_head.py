"""45_cls_head.py - Cap 45: [CLS] como vector clasificador."""
import torch
from _bpe import BPETokenizer
from _models import MiniBERT, ClassificationHead, get_device

device = get_device()
tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device, weights_only=False)
cfg  = ckpt["config"]
model = MiniBERT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

# Cabeza de clasificacion: cfg["d_model"] → 2 (EN=0, ES=1)
cls_head = ClassificationHead(d_model=cfg["d_model"], n_classes=2).to(device)

print("=== [CLS] como clasificador ===\n")
print(f"ClassificationHead: Linear(128, 2)")
n_params = sum(p.numel() for p in cls_head.parameters())
print(f"Params de la cabeza: {n_params} (minimos!)\n")

examples = [
    ("To be or not to be", "EN", 0),
    ("The king is dead", "EN", 0),
    ("En un lugar de la Mancha", "ES", 1),
    ("No hay mal que por bien no venga", "ES", 1),
]
print("CLS vectors antes de fine-tuning (clasificacion aleatoria):")
print(f"{'Texto':<40} {'Idioma'} {'Logit EN':>10} {'Logit ES':>10}")
for text, lang, _ in examples:
    ids = torch.tensor([tok.encode_bert(text)[:cfg["max_seq_len"]]],
                       dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids)
        logits = cls_head(h)
    print(f"{text:<40} {lang}     {logits[0,0].item():>10.3f}  {logits[0,1].item():>10.3f}")

print("\nLos logits son aleatorios (cabeza no entrenada) — fine-tuning en cap 47.")
print("\n=== Por que [CLS] y no promedio de todos los tokens? ===")
print("""
BERT podria usar promedio de todos los tokens como representacion.
Usar [CLS] es una decision de diseno:
  1. [CLS] es un token sin contenido propio — aprende libremente a ser 'resumen'
  2. Permite encodificacion conjunta de pares (cross-encoder) eficiente
  3. El promedio puede mezclar señales de tokens no relevantes
  4. En practica: ambos funcionan; [CLS] es el estandar BERT
""")
