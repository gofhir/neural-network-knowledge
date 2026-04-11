"""
Ejemplo 9 — Red neuronal con LayerNorm (JAX/Flax)

Objetivo: ver como LayerNorm es mas natural en JAX que BatchNorm,
          porque no tiene estado mutable (no hay batch_stats).

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/09_red_con_layernorm_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn


class NetworkWith_LN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Capa 1: Dense -> LayerNorm -> ReLU
        x = nn.Dense(64)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Capa 2: Dense -> LayerNorm -> ReLU
        x = nn.Dense(32)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        # Salida (sin normalizar)
        x = nn.Dense(2)(x)
        return x


key = jax.random.PRNGKey(42)

# =============================================
# Paso 1: Forward pass
# =============================================
print("=" * 55)
print("Paso 1: Forward pass con LayerNorm")
print("=" * 55)

x = jax.random.normal(key, (8, 10)) * 5
model = NetworkWith_LN()
variables = model.init(key, x)

# Forward (sin mutable! LayerNorm no tiene batch_stats)
output = model.apply(variables, x)

print(f"\n  Entrada: {x.shape}")
print(f"  Salida:  {output.shape}")
print(f"  Salida media={jnp.mean(output):+.4f}, std={jnp.std(output):.4f}")
print(f"\n  Nota: NO necesita mutable=['batch_stats']")
print(f"  Es simplemente: model.apply(variables, x)")

# =============================================
# Paso 2: No hay diferencia train/eval
# =============================================
print(f"\n{'='*55}")
print("Paso 2: No hay diferencia train/eval")
print("=" * 55)

sample = jax.random.normal(key, (1, 10))

# Misma llamada, sin importar si es train o eval
output1 = model.apply(variables, sample)
output2 = model.apply(variables, sample)

print(f"\n  Llamada 1: {[round(float(v), 4) for v in output1.squeeze()]}")
print(f"  Llamada 2: {[round(float(v), 4) for v in output2.squeeze()]}")
print(f"  Son iguales? {jnp.allclose(output1, output2)}")
print(f"\n  -> No hay use_running_average, no hay mutable, no hay modos.")
print(f"     Siempre da el mismo resultado. Pura funcion.")

# =============================================
# Paso 3: Funciona con batch_size=1
# =============================================
print(f"\n{'='*55}")
print("Paso 3: Funciona con batch_size=1")
print("=" * 55)

single = jax.random.normal(key, (1, 10))
output_single = model.apply(variables, single)
print(f"\n  LayerNorm con batch=1: {[round(float(v), 4) for v in output_single.squeeze()]}")
print(f"  -> Funciona perfecto!")

# BatchNorm con batch=1 falla
print(f"\n  BatchNorm con batch=1:")

class BN_Model(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        return x

bn_model = BN_Model()
bn_vars = bn_model.init(key, jnp.ones((2, 10)))
try:
    _, _ = bn_model.apply(bn_vars, single, mutable=['batch_stats'])
    print(f"  -> No dio error, pero con 1 muestra las stats son ruidosas")
except Exception as e:
    print(f"  -> ERROR: {e}")

# =============================================
# Paso 4: Parametros
# =============================================
print(f"\n{'='*55}")
print("Paso 4: Parametros")
print("=" * 55)

def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))

total = count_params(variables)
print(f"\n  Total parametros: {total}")
print(f"  Solo tiene 'params' (pesos + gamma/beta)")
print(f"  NO tiene 'batch_stats' (no hay running mean/var)")

# =============================================
# Paso 5: Por que LayerNorm es ideal para JAX
# =============================================
print(f"\n{'='*55}")
print("Paso 5: Por que LayerNorm encaja perfecto en JAX")
print("=" * 55)
print(f"""
  JAX es FUNCIONAL PURO: f(params, x) -> y
    - Misma entrada = misma salida (determinista)
    - Sin efectos secundarios
    - Sin estado oculto

  LayerNorm cumple todo esto:
    - No tiene estado mutable (no batch_stats)
    - No cambia entre train/eval
    - model.apply(params, x) siempre da lo mismo

  BatchNorm ROMPE el paradigma funcional:
    - Tiene estado mutable (batch_stats)
    - Hay que pasar mutable=[...] y manejar updates
    - Train y eval dan resultados distintos

  Por eso los Transformers (que usan LayerNorm) son
  mas naturales de implementar en JAX que las CNNs
  (que usan BatchNorm).
""")
