"""32_tokenizer_refactor_demo.py - Cap 32: demo que refactor tokenizer-agnostic funciona.

Misma funcion generate_with_prompt, distintos tokenizers (CharTokenizer y BPETokenizer).
"""
from pathlib import Path
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import build_char_maps
from _bpe import BPETokenizer, CharTokenizer

prompt = "INSTR: repeat 'a' three\nRESP: "

print("=== Char-level (Camino 2) ===")
text = Path("shakespeare.txt").read_text()
c2i, i2c = build_char_maps(text)
char_tok = CharTokenizer(c2i, i2c)
model_char = load_pretrained_mini_llama("checkpoints/mini_llama_sft.pt")
out_char = generate_with_prompt(model_char, prompt, char_tok,
                                max_new_tokens=10, temperature=0.1, top_k=5)
print(f"Prompt:    {prompt!r}")
print(f"Output:    {out_char[len(prompt):]!r}")
print(f"Tokenizer: CharTokenizer (vocab_size={char_tok.vocab_size})")

print("\n=== BPE-level (Camino 2.5) ===")
bpe_tok = BPETokenizer.load("data/bpe_tokenizer.json")
model_bpe = load_pretrained_mini_llama("checkpoints/mini_llama_bpe_base.pt",
    config=dict(vocab_size=bpe_tok.vocab_size, max_seq_len=256,
                d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))
out_bpe = generate_with_prompt(model_bpe, prompt, bpe_tok,
                               max_new_tokens=15, temperature=0.8, top_k=10)
print(f"Prompt:    {prompt!r}")
print(f"Output:    {out_bpe[len(prompt):]!r}")
print(f"Tokenizer: BPETokenizer (vocab_size={bpe_tok.vocab_size})")
print("\nMisma funcion generate_with_prompt, distintos tokenizers. Refactor OK.")
