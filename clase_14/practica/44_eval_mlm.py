"""44_eval_mlm.py - Cap 44: fill-in-the-blank con Mini-BERT pretrained."""
import torch
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device

device = get_device()
tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

ckpt = torch.load("checkpoints/mini_bert_pretrained.pt", map_location=device)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
mlm_head = MLMHead(d_model=cfg["d_model"], vocab_size=cfg["vocab_size"]).to(device)
model.load_state_dict(ckpt["model"])
mlm_head.load_state_dict(ckpt["mlm_head"])
model.eval(); mlm_head.eval()

def predict_mask(left: str, right: str, top_k: int = 5):
    """Predice el token entre left y right.

    IMPORTANTE: NO pasar "[MASK]" como texto — el BPE lo tokenizaria como
    chars individuales '[','M','A','S','K',']'. En su lugar, construimos
    manualmente la secuencia: [CLS] + encode(left) + mask_id + encode(right) + [SEP].
    """
    l_ids = tok.encode(left)
    r_ids = tok.encode(right)
    ids = [tok.cls_id] + l_ids + [tok.mask_id] + r_ids + [tok.sep_id]
    mask_pos = 1 + len(l_ids)  # posicion exacta del mask_id

    x = torch.tensor([ids[:cfg["max_seq_len"]]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(x)
        logits = mlm_head(h)
    probs = torch.softmax(logits[0, mask_pos], dim=-1)
    top_ids = probs.topk(top_k).indices.tolist()
    top_probs = probs.topk(top_k).values.tolist()
    display = f"{left!r} [MASK] {right!r}"
    print(f"Texto: {display}")
    print(f"Top-{top_k} predicciones:")
    for i, (tid, prob) in enumerate(zip(top_ids, top_probs)):
        tok_str = tok.id_to_token.get(tid, "?")
        print(f"  {i+1}. '{tok_str}' ({prob:.3f})")
    print()

print("=== Fill-in-the-blank con Mini-BERT ===\n")
# Cada ejemplo: (left_context, right_context)
examples = [
    ("To ", " or not to be"),
    ("To be or not to ", ""),
    ("En un ", " de la Mancha"),
    ("The ", " is dead"),
    ("No hay mal que por bien no ", ""),
]
for left, right in examples:
    predict_mask(left, right)
