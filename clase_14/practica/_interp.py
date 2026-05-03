"""_interp.py - helpers de interpretabilidad mecanicista para Camino 3."""
from contextlib import contextmanager


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
