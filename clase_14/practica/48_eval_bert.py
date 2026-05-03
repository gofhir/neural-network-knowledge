"""48_eval_bert.py - Cap 48: accuracy + attention patterns + PCA [CLS]."""
import json, torch
from _models import MiniBERT, ClassificationHead, get_device
from _bpe import BPETokenizer

device = get_device()

ckpt = torch.load("checkpoints/mini_bert_finetuned.pt", map_location=device, weights_only=False)
cfg  = ckpt["config"]
model    = MiniBERT(**cfg).to(device)
cls_head = ClassificationHead(d_model=cfg["d_model"], n_classes=2).to(device)
model.load_state_dict(ckpt["model"])
cls_head.load_state_dict(ckpt["cls_head"])
model.eval(); cls_head.eval()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()

# === Accuracy en eval set ===
with open("data/lang_eval.jsonl") as f:
    eval_data = [json.loads(l) for l in f]
correct = 0
for ex in eval_data:
    ids = torch.tensor([ex["ids"]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids); logits = cls_head(h)
    pred = logits.argmax(dim=-1).item()
    if pred == ex["label"]: correct += 1
acc = correct / len(eval_data)
print(f"Accuracy EN/ES: {acc:.3f} ({correct}/{len(eval_data)})\n")

# === Attention patterns ===
attention_weights = {}
def hook_fn(module, input, output):
    if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
        attention_weights["last"] = output[1].detach().cpu()

handle = model.blocks[-1].attn.register_forward_hook(hook_fn)

example_en = "To be or not to be that is the question"
ids_en = torch.tensor([tok.encode_bert(example_en)[:cfg["max_seq_len"]]],
                       dtype=torch.long, device=device)
with torch.no_grad():
    h = model(ids_en)
handle.remove()

tokens_list = [tok.id_to_token.get(i, "?") for i in ids_en[0].tolist()]
attn = attention_weights.get("last")
if attn is not None:
    print("Attention pattern ultimo bloque (desde [CLS], sobre todos los tokens):")
    cls_attn = attn[0, 0, :].tolist()
    for i, (tok_str, weight) in enumerate(zip(tokens_list, cls_attn)):
        bar = "=" * int(weight * 40)
        print(f"  {i:2d} {tok_str:>10}: {weight:.3f} {bar}")
else:
    print("(attention weights no disponibles)")

# === PCA de [CLS] vectors ===
print("\n=== PCA de embeddings [CLS] (EN vs ES) ===")
en_vecs, es_vecs = [], []
for ex in eval_data[:50]:
    ids = torch.tensor([ex["ids"][:cfg["max_seq_len"]]], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids)
    cls_vec = h[0, 0].cpu()
    if ex["label"] == 0: en_vecs.append(cls_vec)
    else:                es_vecs.append(cls_vec)

all_vecs = torch.stack(en_vecs + es_vecs)
mean = all_vecs.mean(0)
centered = all_vecs - mean
U, S, V = torch.pca_lowrank(centered, q=2)
proj = centered @ V  # (N, 2)
n_en = len(en_vecs)
en_proj = proj[:n_en]; es_proj = proj[n_en:]
print(f"EN centroide: ({en_proj[:, 0].mean():.2f}, {en_proj[:, 1].mean():.2f})")
print(f"ES centroide: ({es_proj[:, 0].mean():.2f}, {es_proj[:, 1].mean():.2f})")
dist = ((en_proj.mean(0) - es_proj.mean(0)).norm()).item()
print(f"Distancia entre centroides: {dist:.3f}")
print("(>2.0 = separacion clara, <1.0 = mezclados)")
