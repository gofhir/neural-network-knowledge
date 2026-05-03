"""42_mlm_loss.py - Cap 42: MLM masking + 80/10/10 split."""
import torch
import torch.nn.functional as F
from _bpe import BPETokenizer
from _models import MiniBERT, MLMHead, get_device
from _bert_utils import apply_mlm_mask

torch.manual_seed(42)
device = get_device()

tok = BPETokenizer.load("data/bpe_tokenizer.json")
tok.add_special_tokens()
vocab_size = tok.vocab_size

model = MiniBERT(vocab_size=vocab_size, max_seq_len=128,
                 d_model=128, n_heads=4, n_layers=4, d_ff=512).to(device)
mlm_head = MLMHead(d_model=128, vocab_size=vocab_size).to(device)

sentence = "To be or not to be that is the question"
ids = torch.tensor([tok.encode_bert(sentence)], dtype=torch.long)
print(f"Tokens originales ({ids.shape[1]}):")
print(f"  {[tok.id_to_token[i] for i in ids[0].tolist()]}\n")

masked_ids, labels = apply_mlm_mask(ids.clone(), mask_prob=0.15,
                                     mask_id=tok.mask_id, vocab_size=vocab_size)

print("Despues de MLM masking (15%, split 80/10/10):")
for pos, (orig, masked, label) in enumerate(
        zip(ids[0].tolist(), masked_ids[0].tolist(), labels[0].tolist())):
    if label != -100:
        orig_tok   = tok.id_to_token.get(orig, "?")
        masked_tok = tok.id_to_token.get(masked, "?")
        print(f"  pos {pos:2d}: '{orig_tok}' -> '{masked_tok}'  (label={label}, predict='{orig_tok}')")

n_masked = (labels != -100).sum().item()
print(f"\nTokens enmascarados: {n_masked}/{ids.shape[1]} = {n_masked/ids.shape[1]:.1%}")

# Calcular la loss MLM
masked_ids_dev = masked_ids.to(device)
labels_dev     = labels.to(device)

model.eval()
mlm_head.eval()

with torch.no_grad():
    h = model(masked_ids_dev)
    logits = mlm_head(h)  # (1, T, vocab_size)

loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels_dev.view(-1),
    ignore_index=-100  # ignorar posiciones no enmascaradas
)
print(f"\nMLM loss (modelo random): {loss.item():.4f}")
print(f"Esperado ~log({vocab_size}) = {torch.tensor(vocab_size).float().log().item():.4f}")
print("\nNota: la loss MLM usa ignore_index=-100, igual que SFT usaba loss_mask=0.")
print("Son la misma idea: solo backpropagar donde importa.")
