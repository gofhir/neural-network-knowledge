"""14_show_base_no_instructions.py - Cap 22: el problema.

El Mini-LLaMA pretrained ignora el formato INSTR/RESP y genera Shakespeare-ish.
Este script lo demuestra dandole prompts de instruccion y mostrando el output.
"""
import torch
from _models import load_pretrained_mini_llama, generate_with_prompt
from _eval import build_char_maps

torch.manual_seed(1337)

text = open("shakespeare.txt").read()
c2i, i2c = build_char_maps(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt")

prompts = [
    "INSTR: reverse 'cat'\nRESP: ",
    "INSTR: upper 'hello'\nRESP: ",
    "INSTR: repeat 'a' 3\nRESP: ",
    "Q: who wrote Hamlet?\nA: ",
]

print("=== Mini-LLaMA base (Camino 1) frente a prompts de instrucción ===\n")
for p in prompts:
    print(f"--- Prompt ---\n{p}")
    print(f"--- Output ---")
    out = generate_with_prompt(model, p, c2i, i2c, max_new_tokens=40,
                               temperature=0.8, top_k=10)
    print(out)
    print()
