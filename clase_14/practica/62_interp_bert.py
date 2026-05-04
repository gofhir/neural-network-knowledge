"""62_interp_bert.py - Cap 62: interpretabilidad sobre Mini-BERT (encoder bidireccional)."""
import torch
from _models import MiniBERT, ClassificationHead, get_device
from _bpe import BPETokenizer

torch.manual_seed(1337)
device = get_device()

# Cargar Mini-BERT fine-tuned (cap 47)
ckpt = torch.load("checkpoints/mini_bert_finetuned.pt", map_location=device, weights_only=False)
cfg = ckpt["config"]
model = MiniBERT(**cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
cls_head = ClassificationHead(d_model=cfg["d_model"], n_classes=2).to(device)
cls_head.load_state_dict(ckpt["cls_head"])
cls_head.eval()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
print(f"Mini-BERT: {cfg['n_layers']} capas, {cfg['n_heads']} heads, max_seq_len={cfg['max_seq_len']}\n")

# Prompts EN y ES
en_prompt = "to be or not to be that is the question"
es_prompt = "ser o no ser esa es la cuestion"

# Capturar attention weights por capa para ambos prompts
attention_caches = {}


def make_attn_hook(layer):
    def hook(module, inputs, output):
        # output: (attn_output, attn_weights). attn_weights: (B, T, T) si average_attn_weights=True
        if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
            attention_caches[layer] = output[1].detach().cpu()
    return hook


def get_attention(prompt_text):
    handles = []
    attention_caches.clear()
    for layer in range(cfg['n_layers']):
        h = model.blocks[layer].attn.register_forward_hook(make_attn_hook(layer))
        handles.append(h)
    ids = tok.encode_bert(prompt_text)
    ids_t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        h = model(ids_t)
        logits = cls_head(h)
    for hand in handles:
        hand.remove()
    return ids, h, logits, dict(attention_caches)


print("=" * 70)
print(f"PROMPT EN: {en_prompt!r}")
en_ids, en_h, en_logits, en_attn = get_attention(en_prompt)
en_pred = en_logits.argmax(dim=-1).item()
en_lang = "EN" if en_pred == 0 else "ES"
print(f"Prediccion: {en_lang} (logit={en_logits[0, en_pred].item():.3f})")
print(f"Tokens (n={len(en_ids)}): {[tok.id_to_token.get(i, '?') for i in en_ids]}")

print("\n" + "=" * 70)
print(f"PROMPT ES: {es_prompt!r}")
es_ids, es_h, es_logits, es_attn = get_attention(es_prompt)
es_pred = es_logits.argmax(dim=-1).item()
es_lang = "EN" if es_pred == 0 else "ES"
print(f"Prediccion: {es_lang} (logit={es_logits[0, es_pred].item():.3f})")
print(f"Tokens (n={len(es_ids)}): {[tok.id_to_token.get(i, '?') for i in es_ids]}")


def attention_to_special_tokens(attn, ids, tok):
    """Promedio de atencion HACIA [CLS] y [SEP] en cada capa."""
    cls_id = tok.cls_id
    sep_id = tok.sep_id
    cls_pos = ids.index(cls_id) if cls_id in ids else 0
    sep_positions = [i for i, x in enumerate(ids) if x == sep_id]
    sep_pos = sep_positions[-1] if sep_positions else len(ids) - 1

    # attn shape: (1, T, T) o (T, T)
    a = attn[0] if attn.dim() == 3 else attn  # (T, T)
    cls_attn = a[:, cls_pos].mean().item()
    sep_attn = a[:, sep_pos].mean().item()
    self_diag = torch.tensor([a[i, i].item() for i in range(a.shape[0])]).mean().item()
    return cls_attn, sep_attn, self_diag


print("\n" + "=" * 70)
print("=== Patrones de atencion: hacia [CLS], [SEP] y diagonal (self) ===\n")
print(f"{'capa':<6} {'EN: [CLS]':>11} {'EN: [SEP]':>11} {'EN: self':>10}   "
      f"{'ES: [CLS]':>11} {'ES: [SEP]':>11} {'ES: self':>10}")
print("-" * 75)
for layer in range(cfg['n_layers']):
    en_cls, en_sep, en_self = attention_to_special_tokens(en_attn[layer], en_ids, tok)
    es_cls, es_sep, es_self = attention_to_special_tokens(es_attn[layer], es_ids, tok)
    print(f"block.{layer}  {en_cls:>11.3f} {en_sep:>11.3f} {en_self:>10.3f}   "
          f"{es_cls:>11.3f} {es_sep:>11.3f} {es_self:>10.3f}")


# Comparar logits del [CLS] vector (pos 0) entre ambos
print("\n=== Vector [CLS] (pos 0) en EN vs ES ===")
en_cls_vec = en_h[0, 0].cpu()
es_cls_vec = es_h[0, 0].cpu()
print(f"||CLS_EN|| = {en_cls_vec.norm().item():.3f}")
print(f"||CLS_ES|| = {es_cls_vec.norm().item():.3f}")
diff = (en_cls_vec - es_cls_vec).norm().item()
cos = torch.nn.functional.cosine_similarity(en_cls_vec.unsqueeze(0),
                                             es_cls_vec.unsqueeze(0)).item()
print(f"Distancia: {diff:.3f}")
print(f"Cosine similarity: {cos:.3f}")
print("Si cos < 0.5: vectores muy distintos -> CLS captura idioma")
print("Si cos > 0.95: vectores casi iguales -> CLS no distingue")
