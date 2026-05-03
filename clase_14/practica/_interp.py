"""_interp.py - helpers de interpretabilidad mecanicista para Camino 3."""
from contextlib import contextmanager
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder estilo Bricken et al. 2023.
    encoder: Linear + ReLU. decoder: Linear sin bias (evita shrinkage).
    loss = MSE(reconstruction) + l1_coeff * L1(features)."""

    def __init__(self, d_model, d_features, l1_coeff=1e-3):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_features)
        self.decoder = nn.Linear(d_features, d_model, bias=False)
        self.l1_coeff = l1_coeff

    def forward(self, x):
        features = torch.relu(self.encoder(x))
        recon = self.decoder(features)
        return recon, features


def logit_lens(model, residual):
    """Aplica head al residual stream para proyectar al vocab.
    El residual debe ser POST-norm final (despues de ln_f en MiniGPT)."""
    return model.head(residual)


def patch_activation(model, ids, patch_dict):
    """Forward pass con activaciones reemplazadas en posiciones especificas.
    patch_dict: {module_name: (position_or_slice, replacement_tensor)}.
    replacement_tensor shape: (B, n_positions, d_model).
    Si el modelo retorna tupla (logits, loss), retorna solo logits."""
    import torch
    handles = []
    name_to_module = dict(model.named_modules())
    for name, (positions, replacement) in patch_dict.items():
        if name not in name_to_module:
            raise KeyError(f"Module '{name}' not found in model")

        def make_patch_hook(positions, replacement):
            def hook(module, inputs, output):
                is_tuple = isinstance(output, tuple)
                out = output[0] if is_tuple else output
                out = out.clone()
                if isinstance(positions, int):
                    out[:, positions:positions + 1] = replacement
                else:
                    out[:, positions] = replacement
                return (out, *output[1:]) if is_tuple else out
            return hook

        handles.append(name_to_module[name].register_forward_hook(
            make_patch_hook(positions, replacement)))
    try:
        with torch.no_grad():
            result = model(ids)
        return result[0] if isinstance(result, tuple) else result
    finally:
        for h in handles:
            h.remove()


def qk_circuit(W_Q, W_K):
    """QK circuit: W_Q @ W_K^T. Define como una cabeza decide a que atender.
    Shape input: (d_model, d_head). Shape output: (d_model, d_model)."""
    return W_Q @ W_K.T


def ov_circuit(W_V, W_O):
    """OV circuit: W_V @ W_O. Define que informacion mueve la cabeza
    desde la fuente al destino. W_V: (d_model, d_head), W_O: (d_head, d_model).
    Shape output: (d_model, d_model)."""
    return W_V @ W_O


def previous_token_score(attn):
    """Score [0, 1]: cuanto atiende cada posicion i a la i-1.
    attn shape: (T, T). Asume causal (triangular inferior)."""
    import torch
    T = attn.shape[0]
    if T < 2:
        return 0.0
    diag = torch.tensor([attn[i, i - 1].item() for i in range(1, T)])
    return diag.mean().item()


def induction_score(attn, ids):
    """Score de induccion: para token repetido en posicion j, cuanto atiende
    a la posicion i+1 (donde estaba el siguiente token la primera vez).
    Patron: ...A B... A -> B."""
    T = attn.shape[0]
    scores = []
    ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    for j in range(2, T):
        tok = ids_list[j]
        for i in range(j - 1):
            if ids_list[i] == tok and i + 1 < j:
                scores.append(attn[j, i + 1].item())
                break
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


@contextmanager
def cache_activations(model, names):
    """Context manager que registra forward hooks en submodulos por nombre.
    Retorna dict {name: tensor} con el output de cada modulo en el ultimo forward.
    Cleanup automatico al salir del bloque with."""
    cache = {}
    handles = []
    name_to_module = dict(model.named_modules())
    for name in names:
        if name not in name_to_module:
            raise KeyError(f"Module '{name}' not found in model")

        def make_hook(n):
            def hook(module, inputs, output):
                out = output[0] if isinstance(output, tuple) else output
                cache[n] = out.detach()
            return hook

        handles.append(name_to_module[name].register_forward_hook(make_hook(name)))
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()
