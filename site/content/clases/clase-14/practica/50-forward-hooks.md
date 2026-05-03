---
title: "50 - Forward hooks: capturar lo que pasa adentro"
weight: 500
math: true
---

## 1. Apertura: por que necesitamos hooks

Hasta el cap 49 el Transformer era una caja: input → forward → logits. Funcionaba. Pero **¿que pasa dentro?** ¿Que computa el bloque 2? ¿Que vector tiene el residual stream despues de la capa 3? Sin acceso a las activaciones intermedias, solo medimos el output final.

La forma ingenua de inspeccionar es modificar el modelo: agregar `print(x)` dentro del forward, retornar tuplas con todas las activaciones, o crear una version "instrumented" del modelo. Todo eso rompe la abstraccion: cada vez que quieres mirar algo nuevo tienes que tocar el codigo del modelo.

PyTorch ofrece una primitiva mas elegante: **forward hooks**. Un hook es una funcion que se ejecuta automaticamente cada vez que un modulo procesa un input, sin modificar el modulo. Permite capturar activaciones, modificarlas al vuelo, o registrar estadisticas — todo desde fuera.

Este es el primer ladrillo de toda la interpretabilidad mecanicista. Antes de hacer logit lens, induction heads, activation patching o sparse autoencoders, necesitamos poder inspeccionar.

---

## 2. La mecanica de `register_forward_hook`

Cada `nn.Module` en PyTorch tiene un metodo `register_forward_hook(fn)` que toma una funcion `fn(module, inputs, output) -> None`. Cada vez que ese modulo ejecuta su `forward`, PyTorch llama `fn` justo despues, pasandole:

- `module`: la instancia del modulo
- `inputs`: tupla de los inputs al modulo
- `output`: el output retornado por el forward (tensor o tupla)

El hook puede leer estos valores (sin modificarlos), guardarlos en un dict externo, o incluso retornar un nuevo output para reemplazar el original (eso lo usaremos en el cap 57).

```python
def my_hook(module, inputs, output):
    print(f"{module.__class__.__name__} produjo shape {output.shape}")

handle = model.blocks[0].register_forward_hook(my_hook)
# ... usar el modelo
handle.remove()  # IMPORTANTE: liberar el hook
```

El `handle` es un objeto que permite quitar el hook despues. Si olvidas removerlo, el hook se queda activo y consume memoria silenciosamente — bug clasico.

---

## 3. El context manager `cache_activations`

El patron `register_forward_hook` + `handle.remove()` es lo suficientemente repetitivo como para abstraerlo. El modulo `_interp.py` define un context manager que automatiza el ciclo completo:

```python
@contextmanager
def cache_activations(model, names):
    """Context manager que registra forward hooks en submodulos por nombre.
    Retorna dict {name: tensor} con el output de cada modulo."""
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
```

Tres detalles tecnicos:

1. **`name_to_module = dict(model.named_modules())`**: PyTorch ofrece `named_modules()` que itera sobre todos los submodulos con su path jerarquico (`"blocks.0.attn"`, `"blocks.1.ffn.gate"`, etc.). Esto permite seleccionar modulos por nombre en lugar de tener que navegar a mano.

2. **`make_hook(n)` con closure**: si pones `def hook(...): cache[name] = ...` directamente dentro del loop, todas las closures capturan la **ultima** `name` del loop por la regla de scoping de Python. La fabrica `make_hook(n)` rompe la captura: cada hook tiene su propia `n`.

3. **`out.detach()`**: las activaciones cacheadas no deben ser parte del computational graph. Si las guardas con gradiente, mantienes vivo todo el grafo y consumes RAM proporcional al numero de hooks × tamaño del modelo.

El `try/finally` garantiza que los hooks se remueven incluso si el codigo dentro del `with` lanza una excepcion. Sin esto, una excepcion cualquiera dejaria hooks fantasma en el modelo.

---

## 4. Script

```python
"""50_forward_hooks.py - Cap 50: forward hooks y cache de activaciones."""
import torch
from _models import load_pretrained_mini_llama, get_device, CharTokenizer, load_text
from _interp import cache_activations

torch.manual_seed(1337)
device = get_device()

text = load_text("shakespeare.txt")
tok = CharTokenizer(text)

model = load_pretrained_mini_llama("checkpoints/mini_llama_base.pt", device=device,
                                   config=dict(vocab_size=tok.vocab_size, max_seq_len=256,
                                               d_model=128, h_q=4, h_kv=2, n_layers=4, d_ff=384))

prompt = "To be or not to "
ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
print(f"Prompt: {prompt!r}")
print(f"Tokens: {ids.shape[1]} ids = {ids[0].tolist()[:10]}...\n")

names = [f"blocks.{i}" for i in range(4)] + ["norm_final"]
print(f"Cacheando activaciones de {len(names)} puntos:")
with cache_activations(model, names) as cache:
    with torch.no_grad():
        model(ids)

print("\nShapes capturados:")
for name in names:
    t = cache[name]
    print(f"  {name:>15}: shape={tuple(t.shape)}, mean={t.mean():+.4f}, std={t.std():.4f}")

print("\nNorma del residual stream por punto (||x||_2 promedio sobre tokens):")
for name in names:
    norm = cache[name].norm(dim=-1).mean().item()
    print(f"  {name:>15}: {norm:.3f}")

print("\nDelta norma entre bloques consecutivos:")
prev = None
for name in names:
    cur = cache[name].norm(dim=-1).mean().item()
    if prev is not None:
        delta = cur - prev
        sign = "+" if delta >= 0 else ""
        print(f"  {name:>15}: {sign}{delta:.3f}")
    prev = cur
```

---

## 5. Output literal

```
Prompt: 'To be or not to '
Tokens: 16 ids = [32, 53, 1, 40, 43, 1, 53, 56, 1, 52]...

Cacheando activaciones de 5 puntos:

Shapes capturados:
         blocks.0: shape=(1, 16, 128), mean=-0.0064, std=0.8695
         blocks.1: shape=(1, 16, 128), mean=-0.0013, std=0.9120
         blocks.2: shape=(1, 16, 128), mean=-0.0082, std=1.0866
         blocks.3: shape=(1, 16, 128), mean=+0.0324, std=2.3486
       norm_final: shape=(1, 16, 128), mean=+0.0185, std=1.2192

Norma del residual stream por punto (||x||_2 promedio sobre tokens):
         blocks.0: 9.772
         blocks.1: 10.259
         blocks.2: 12.247
         blocks.3: 25.794
       norm_final: 13.791

Delta norma entre bloques consecutivos:
         blocks.1: +0.487
         blocks.2: +1.988
         blocks.3: +13.547
       norm_final: -12.002
```

---

## 6. Analisis: que dicen los numeros

Los shapes son consistentes: `(B=1, T=16, d_model=128)` en cada punto. El residual stream mantiene su dimension a lo largo de todo el modelo — es la "autopista" que veremos en detalle en el cap 51.

Los **mean** son cercanos a 0 (entre -0.008 y +0.032). RMSNorm centra los valores aunque no resta la media explicitamente; el efecto practico es similar.

Los **std** crecen capa a capa: 0.87 → 0.91 → 1.09 → **2.35**. La cuarta capa tiene mas del doble de varianza que la primera. Cada bloque amplifica el rango dinamico del residual antes de la norm final.

El dato mas revelador esta en las normas:

- **blocks.0 → blocks.2**: crecimiento gradual (9.77 → 10.26 → 12.25). Cada bloque agrega ~0.5-2 unidades de norma.
- **blocks.2 → blocks.3**: salto explosivo (12.25 → **25.79**). El ultimo bloque mas que duplica la norma.
- **blocks.3 → norm_final**: la norm final lleva la magnitud de vuelta a 13.79, recortando el inflate del bloque 3.

¿Por que el bloque 3 explota? RMSNorm tiene un parametro `gamma` (gain) que el modelo aprende. Si el modelo decide amplificar la senal antes de la cabeza de salida, lo hace via `gamma` grande. El bloque 3 escribe agresivamente al residual stream porque su senal alimenta directamente al `lm_head`. La norm final compensa, llevando los valores a un rango que el head puede usar.

Esta inspeccion — que tomo 30 lineas de codigo — ya revela algo no trivial sobre el modelo: la cuarta capa hace mas trabajo de "escritura" que las anteriores. Sin hooks, esto seria invisible.

---

## 7. Que sigue

Ya podemos capturar activaciones de cualquier modulo. Los siguientes capitulos usaran este ladrillo para:

- **Cap 51**: visualizar el residual stream como la "autopista" — cada bloque suma sin sobreescribir
- **Cap 52**: aplicar el `lm_head` a residuales intermedios (logit lens) para ver predicciones a media procesamiento
- **Cap 53**: cachear `attn_weights` por capa y cabeza para visualizar patrones de atencion
- **Cap 57**: usar hooks para *modificar* activaciones (activation patching) — del correlacional al causal

Sin `cache_activations`, todo lo demas seria mucho mas trabajoso. Es el "open the hood" del Transformer.

---

## 8. Preguntas de verificacion

**1. ¿Por que el hook usa `out.detach()` antes de guardar en el cache?**

Si guardamos el output sin `detach()`, el tensor mantiene viva su parte del computational graph (los nodos que lo produjeron). En un modelo de 4 capas con 5 puntos cacheados, esto multiplicaria la memoria utilizada por el grafo. Ademas, las activaciones cacheadas son para inspeccion — no necesitamos backprop a traves de ellas. `detach()` separa el tensor del grafo, manteniendo solo los valores numericos.

**2. ¿Que pasaria si olvidamos llamar `handle.remove()`?**

El hook se queda registrado en el modulo permanentemente. Cada forward subsecuente del modelo llamaria al hook, escribiendo al `cache` original (que probablemente ya no exista en el scope del programa, causando referencias colgantes en memoria). Mas grave: si registras hooks repetidamente sin removerlos (por ejemplo en un loop de evaluacion), el modelo acumula hooks fantasma que ralentizan cada forward y consumen RAM. El context manager con `try/finally` previene este bug.

**3. ¿Por que el cap usa `make_hook(n)` con closure en lugar de definir el hook directamente en el loop?**

Por la regla de Python sobre captura de variables en closures: las closures capturan **referencias**, no valores. Si haces:

```python
for name in names:
    def hook(module, inputs, output):
        cache[name] = output  # captura la referencia 'name'
    handles.append(module.register_forward_hook(hook))
```

Cuando los hooks se ejecutan despues del loop, todos ven la misma `name` — la ultima del loop. La fabrica `make_hook(n)` crea un nuevo scope donde `n` es un parametro local fijo por llamada, evitando la captura compartida.
