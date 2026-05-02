---
title: "32 - Refactor tokenizer-agnostic: un tokenizer, muchos modelos"
weight: 320
math: true
---

Este capitulo es tecnico, no pedagogico. El objetivo es verificar que el refactor tokenizer-agnostic funciona: `generate_with_prompt` acepta cualquier objeto con `.encode()` y `.decode()`, lo que permite usar el mismo codigo para char-level y BPE-level sin duplicar nada.

---

## 1. La interfaz

La interfaz que cualquier tokenizer tiene que satisfacer es minima:

- `tokenizer.encode(text: str) -> list[int]`
- `tokenizer.decode(ids: list[int]) -> str`

Dos metodos, punto. Ningun helper en `_models.py` o `_eval.py` depende de que el tokenizer sea `BPETokenizer` o `CharTokenizer` — solo llaman `.encode()` y `.decode()`. Esto es duck typing en Python: si el objeto tiene los metodos correctos, funciona.

---

## 2. `CharTokenizer` — el wrapper de compat

Para que el codigo de Camino 2 (caps 14-21, que usaba diccionarios `c2i`/`i2c` directamente) sea compatible con la nueva interfaz, `_bpe.py` incluye `CharTokenizer`:

```python
class CharTokenizer:
    """Wrapper char-level que expone la interfaz BPETokenizer."""
    def __init__(self, char_to_id: dict, id_to_char: dict):
        self._c2i = char_to_id
        self.id_to_token = id_to_char
        self.vocab_size = len(char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self._c2i[c] for c in text if c in self._c2i]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token.get(i, "") for i in ids)
```

No hay logica nueva. `CharTokenizer` envuelve los dos diccionarios que el codigo de Camino 2 ya tenia y expone la misma interfaz que `BPETokenizer`. El vocab_size es el numero de entradas en el diccionario — 65 para Shakespeare.

---

## 3. El script demo

`clase_14/practica/32_tokenizer_refactor_demo.py`:

```python
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
```

---

## 4. Output del demo

```text
=== Char-level (Camino 2) ===
Prompt:    "INSTR: repeat 'a' three\nRESP: "
Output:    'aaa\n'
Tokenizer: CharTokenizer (vocab_size=65)

=== BPE-level (Camino 2.5) ===
Prompt:    "INSTR: repeat 'a' three\nRESP: "
Output:    'if thou arty of this foot of Lan'
Tokenizer: BPETokenizer (vocab_size=1112)

Misma funcion generate_with_prompt, distintos tokenizers. Refactor OK.
```

El char-level SFT responde `'aaa\n'` — correcto, el modelo tiene fine-tuning sobre la tarea repeat. El BPE-base genera texto shakespeareano aleatorio — esperado, es un modelo pretrained sin SFT, no sabe que se le pide repetir nada. La diferencia no es del tokenizer sino del entrenamiento: el char-level paso por 16 epochs de SFT, el BPE solo tiene pretrain.

Lo que importa para el refactor: `generate_with_prompt` se llamo dos veces con la misma firma, distintos tokenizers, y funciono en ambos casos sin tocar el codigo del helper.

---

## 5. Tests — 11/11 PASS

```text
tests/test_bpe.py::test_bpe_train_reduces_token_count PASSED
tests/test_bpe.py::test_bpe_round_trip PASSED
tests/test_bpe.py::test_bpe_encode_shorter_than_chars PASSED
tests/test_bpe.py::test_bpe_save_load PASSED
tests/test_bpe.py::test_char_tokenizer_compat PASSED
tests/test_eval.py::test_build_char_maps_shakespeare PASSED
tests/test_models_helpers.py::test_load_pretrained_smoke PASSED
tests/test_models_helpers.py::test_load_pretrained_default_config_matches_training_script PASSED
tests/test_models_helpers.py::test_generate_with_prompt_returns_string PASSED
tests/test_models_helpers.py::test_compute_logp_response_shape PASSED
tests/test_models_helpers.py::test_dpo_loss_zero_when_policy_equals_ref PASSED

11 passed in 5.03s
```

El refactor no rompio nada. `test_char_tokenizer_compat` en particular verifica que `CharTokenizer` expone `.encode()` y `.decode()` con la misma semantica que `BPETokenizer`.

---

## 6. El estado del codigo desde aqui

Desde cap 33 en adelante, todos los scripts usan `BPETokenizer`. Camino 2 (caps 14-21) usa `CharTokenizer`. Los helpers en `_models.py` y `_eval.py` no saben ni les importa cual es cual — reciben un objeto con `.encode()/.decode()` y lo usan. Eso es todo el refactor.

Volver al [hub de practica](..) o a la [Clase 14](../..).
