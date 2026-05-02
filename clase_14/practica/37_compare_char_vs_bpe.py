"""37_compare_char_vs_bpe.py - Cap 37: tabla maestra char-level vs BPE-level.

Eval sobre el subset compartido (qa + repeat que existen en ambos tokenizadores).
"""
import torch, json
from pathlib import Path
from _bpe import BPETokenizer, CharTokenizer
from _models import load_pretrained_mini_llama
from _eval import build_char_maps, eval_exact_match, eval_drift, load_jsonl

device = "cpu"  # eval ligero — cpu es suficiente

# Char-level setup
text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
char_tok = CharTokenizer(c2i, i2c)
char_cfg = dict(vocab_size=len(c2i), max_seq_len=256,
                d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

# BPE-level setup
bpe_tok = BPETokenizer.load("data/bpe_tokenizer.json")
bpe_cfg = dict(vocab_size=bpe_tok.vocab_size, max_seq_len=256,
               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)

# Subset compartido: qa + repeat del eval set BPE
shared = [ex for ex in load_jsonl("data/sft_bpe_eval.jsonl")
          if ex["task"] in {"qa", "repeat"}]
with open("/tmp/shared_eval.jsonl", "w") as f:
    for ex in shared:
        f.write(json.dumps(ex) + "\n")
print(f"Shared eval examples: {len(shared)} (qa + repeat)")

ambiguous = ["INSTR: capitalize 'cat'\nRESP: ", "Q: what is 2+2?\nA: "]

print("\n=== Tabla maestra: char-level vs BPE-level ===\n")
print(f"{'modelo':<22}{'qa':<10}{'repeat':<10}{'drift':<10}")
print("-" * 52)

for label, ckpt, tok_obj, cfg in [
    ("char-base",   "checkpoints/mini_llama_base.pt",          char_tok, char_cfg),
    ("char-sft",    "checkpoints/mini_llama_sft.pt",           char_tok, char_cfg),
    ("char-dpo",    "checkpoints/mini_llama_dpo.pt",           char_tok, char_cfg),
    ("bpe-base",    "checkpoints/mini_llama_bpe_base.pt",      bpe_tok,  bpe_cfg),
    ("bpe-sft",     "checkpoints/mini_llama_bpe_sft.pt",       bpe_tok,  bpe_cfg),
    ("bpe-dpo-b05", "checkpoints/mini_llama_bpe_dpo_b05.pt",   bpe_tok,  bpe_cfg),
]:
    m = load_pretrained_mini_llama(ckpt, device=device, config=cfg)
    em = eval_exact_match(m, "/tmp/shared_eval.jsonl", tok_obj,
                          n_per_task=100, device=device)
    drift = eval_drift(m, ambiguous, tok_obj, device=device)
    qa  = em.get("qa", 0.0)
    rep = em.get("repeat", 0.0)
    print(f"{label:<22}{qa:<10.3f}{rep:<10.3f}{drift:<10.3f}")
