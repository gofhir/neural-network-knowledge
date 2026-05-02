"""Tests para helpers de _models.py. Tests se agregan en Tasks 1-4."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from _models import MiniLLaMA, load_pretrained_mini_llama, generate_with_prompt


def test_load_pretrained_smoke(tmp_path):
    cfg = dict(vocab_size=65, max_seq_len=64, d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384)
    m = MiniLLaMA(**cfg)
    ckpt = tmp_path / "fake.pt"
    torch.save(m.state_dict(), ckpt)
    loaded = load_pretrained_mini_llama(str(ckpt), device="cpu", config=cfg)
    assert loaded is not None
    for (k1, v1), (k2, v2) in zip(m.state_dict().items(), loaded.state_dict().items()):
        assert torch.allclose(v1, v2)


def test_load_pretrained_default_config_matches_training_script(tmp_path):
    """Defaults del helper deben matchear 13_mini_llama.py para que checkpoints carguen sin config."""
    default_cfg = dict(vocab_size=65, max_seq_len=256, d_model=128,
                       h_q=4, h_kv=2, n_layers=4, d_ff=384)
    m = MiniLLaMA(**default_cfg)
    ckpt = tmp_path / "fake.pt"
    torch.save(m.state_dict(), ckpt)
    loaded = load_pretrained_mini_llama(str(ckpt), device="cpu")  # no config=
    assert loaded.max_seq_len == 256


def test_generate_with_prompt_returns_string():
    cfg = dict(vocab_size=65, max_seq_len=64, d_model=64, h_q=2, h_kv=1, n_layers=2, d_ff=128)
    m = MiniLLaMA(**cfg)
    chars = list("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ\n.,'?!:;-0123456789")
    chars = chars[:65]
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    out = generate_with_prompt(m, "abc", c2i, i2c, max_new_tokens=5, temperature=1.0, top_k=10, device="cpu")
    assert isinstance(out, str)
    assert len(out) >= 3  # al menos prompt copiado
