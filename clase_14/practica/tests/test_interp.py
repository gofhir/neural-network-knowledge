"""tests/test_interp.py - tests TDD para _interp.py (Camino 3)."""
import torch
from _models import MiniGPT
from _interp import (
    cache_activations,
    logit_lens,
    patch_activation,
    qk_circuit,
    ov_circuit,
    previous_token_score,
    induction_score,
    SparseAutoencoder,
)


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


def test_patch_activation_changes_only_target_position():
    """En atencion causal, patchear posicion 3 cambia logits desde pos 3 en adelante,
    no antes."""
    torch.manual_seed(0)
    model = _make_minigpt()
    model.eval()
    ids = torch.randint(0, 65, (1, 8))
    with torch.no_grad():
        clean_logits, _ = model(ids)
    patch = torch.zeros(1, 1, 128)
    patched_logits = patch_activation(model, ids, {"blocks.1": (3, patch)})
    assert torch.allclose(clean_logits[0, :3], patched_logits[0, :3], atol=1e-5)
    assert not torch.allclose(clean_logits[0, 3], patched_logits[0, 3], atol=1e-5)


def test_qk_circuit_shape():
    W_Q = torch.randn(128, 32)
    W_K = torch.randn(128, 32)
    qk = qk_circuit(W_Q, W_K)
    assert qk.shape == (128, 128)


def test_ov_circuit_shape():
    W_V = torch.randn(128, 32)
    W_O = torch.randn(32, 128)
    ov = ov_circuit(W_V, W_O)
    assert ov.shape == (128, 128)


def test_previous_token_score_perfect():
    """Patron de atencion que mira EXACTAMENTE al anterior debe dar score = 1.0."""
    T = 8
    attn = torch.zeros(T, T)
    for i in range(1, T):
        attn[i, i - 1] = 1.0
    score = previous_token_score(attn)
    assert abs(score - 1.0) < 1e-6


def test_previous_token_score_uniform_low():
    T = 8
    attn = torch.ones(T, T) / T
    score = previous_token_score(attn)
    assert score < 0.2


def test_induction_score_repeated_prompt():
    T = 5
    attn = torch.zeros(T, T)
    attn[4, 1] = 1.0
    ids = torch.tensor([0, 1, 2, 3, 0])
    score = induction_score(attn, ids)
    assert score > 0.5


def test_sae_reconstruction_loss_decreases():
    torch.manual_seed(0)
    sae = SparseAutoencoder(d_model=128, d_features=512, l1_coeff=1e-3)
    x = torch.randn(64, 128)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    initial_loss = None
    final_loss = None
    for step in range(200):
        opt.zero_grad()
        recon, features = sae(x)
        recon_loss = ((x - recon) ** 2).mean()
        l1_loss = features.abs().mean()
        loss = recon_loss + sae.l1_coeff * l1_loss
        loss.backward()
        opt.step()
        if step == 0:
            initial_loss = recon_loss.item()
        final_loss = recon_loss.item()
    assert final_loss < initial_loss * 0.5
