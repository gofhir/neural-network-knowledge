"""tests/test_interp.py - tests TDD para _interp.py (Camino 3)."""
import torch
from _models import MiniGPT
from _interp import cache_activations, logit_lens


def _make_minigpt():
    return MiniGPT(vocab_size=65, d_model=128, h=4, n_layers=4,
                   d_ff=512, block_size=64, activation="gelu")


def test_cache_activations_captures_correct_shapes():
    model = _make_minigpt()
    model.eval()
    ids = torch.zeros(1, 8, dtype=torch.long)
    names = ["blocks.0", "blocks.3"]
    with cache_activations(model, names) as cache:
        with torch.no_grad():
            model(ids)
    assert "blocks.0" in cache
    assert "blocks.3" in cache
    assert cache["blocks.0"].shape == (1, 8, 128)
    assert cache["blocks.3"].shape == (1, 8, 128)


def test_cache_activations_cleanup_removes_hooks():
    model = _make_minigpt()
    n_hooks_before = sum(len(m._forward_hooks) for m in model.modules())
    with cache_activations(model, ["blocks.0"]):
        pass
    n_hooks_after = sum(len(m._forward_hooks) for m in model.modules())
    assert n_hooks_after == n_hooks_before


def test_logit_lens_consistent_with_full_forward():
    """Logit lens del residual post-ln_f debe coincidir con el output del modelo."""
    model = _make_minigpt()
    model.eval()
    ids = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        full_logits, _ = model(ids)
    with cache_activations(model, ["ln_f"]) as cache:
        with torch.no_grad():
            model(ids)
    lens_logits = logit_lens(model, cache["ln_f"])
    assert torch.allclose(full_logits, lens_logits, atol=1e-5)
