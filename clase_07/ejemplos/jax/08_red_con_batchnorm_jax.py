"""
Ejemplo 8 — Red neuronal completa CON BatchNorm (JAX/Flax)

Objetivo: ver como se construye una red real con BatchNorm en JAX.
          En JAX la red se define como una clase con Flax,
          pero el forward pass es funcional (sin estado interno).

Ejecutar:
  docker run --rm clase7-pytorch python -u ejemplos/jax/08_red_con_batchnorm_jax.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn

# =============================================
# Paso 1: Red SIN BatchNorm
# =============================================
print("=" * 55)
print("Paso 1: Red SIN BatchNorm")
print("=" * 55)


class NetworkWithout_BN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(2)(x)
        return x


key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (8, 10)) * 5  # 8 muestras, 10 features

model_sin = NetworkWithout_BN()
params_sin = model_sin.init(key, x)

print(f"\nEntrada: {x.shape} (8 muestras, 10 features)")
print(f"Entrada media={jnp.mean(x):+.4f}, std={jnp.std(x):.4f}\n")

# Forward mostrando activaciones intermedias
h = x
for name, features in [("dense1", 64), ("dense2", 32), ("dense3", 2)]:
    # No podemos ver capas intermedias tan facil en JAX,
    # asi que hacemos el forward manual
    pass

output_sin = model_sin.apply(params_sin, x)
print(f"Salida: media={jnp.mean(output_sin):+.4f}, std={jnp.std(output_sin):.4f}")

# =============================================
# Paso 2: Red CON BatchNorm
# =============================================
print(f"\n{'='*55}")
print("Paso 2: Red CON BatchNorm")
print("=" * 55)


class NetworkWith_BN(nn.Module):
    # use_running_average se pasa desde afuera
    use_running_average: bool = False

    @nn.compact
    def __call__(self, x):
        # Capa 1: Dense -> BatchNorm -> ReLU
        x = nn.Dense(64)(x)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = nn.relu(x)

        # Capa 2: Dense -> BatchNorm -> ReLU
        x = nn.Dense(32)(x)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        x = nn.relu(x)

        # Salida (sin BatchNorm)
        x = nn.Dense(2)(x)
        return x


# Inicializar
model_con = NetworkWith_BN(use_running_average=False)
variables = model_con.init(key, x)

print(f"\nEntrada: {x.shape} (mismos datos)")

# Forward en modo entrenamiento
output_con, updates = model_con.apply(
    variables, x,
    mutable=['batch_stats']  # necesario para BatchNorm
)

print(f"Salida: media={jnp.mean(output_con):+.4f}, std={jnp.std(output_con):.4f}")

# =============================================
# Paso 3: Parametros
# =============================================
print(f"\n{'='*55}")
print("Paso 3: Parametros de la red")
print("=" * 55)

# Contar parametros
def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))

total_sin = count_params(params_sin)
total_con_params = count_params(variables['params'])
total_con_stats = count_params(variables['batch_stats'])

print(f"\n  Sin BatchNorm: {total_sin} parametros")
print(f"  Con BatchNorm:")
print(f"    Trainable:     {total_con_params}")
print(f"    Batch stats:   {total_con_stats} (no trainable)")

# Mostrar estructura
print(f"\n  Estructura de variables (con BN):")
print(f"    variables['params']      -> pesos y gamma/beta (trainable)")
print(f"    variables['batch_stats'] -> running mean/var (no trainable)")

# =============================================
# Paso 4: Train vs Eval
# =============================================
print(f"\n{'='*55}")
print("Paso 4: Train vs Eval en JAX")
print("=" * 55)

sample = jax.random.normal(key, (1, 10))

# Entrenamiento: use_running_average=False, mutable=['batch_stats']
model_train = NetworkWith_BN(use_running_average=False)
merged_vars = {**variables, **updates}  # usar batch_stats actualizados
output_train, _ = model_train.apply(
    merged_vars, sample, mutable=['batch_stats']
)

# Inferencia: use_running_average=True, NO mutable
model_eval = NetworkWith_BN(use_running_average=True)
output_eval = model_eval.apply(merged_vars, sample)

print(f"\n  Train: {[round(float(v), 4) for v in output_train.squeeze()]}")
print(f"  Eval:  {[round(float(v), 4) for v in output_eval.squeeze()]}")
print(f"  Son iguales? {jnp.allclose(output_train, output_eval, atol=1e-2)}")
print(f"  (Pueden diferir porque usan estadisticas distintas)")

# =============================================
# Paso 5: Comparacion de API
# =============================================
print(f"\n{'='*55}")
print("Paso 5: Comparacion para construir redes")
print("=" * 55)
print(f"""
  PyTorch (clase + forward):
    self.fc1 = nn.Linear(10, 64)
    self.bn1 = nn.BatchNorm1d(64)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))

  TensorFlow (Sequential):
    model = Sequential([Dense(64), BatchNormalization(), ReLU()])

  JAX/Flax (@nn.compact):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.BatchNorm(use_running_average=...)(x)
        x = nn.relu(x)

  La diferencia clave de JAX:
    - Los parametros viven FUERA del modelo (en un dict)
    - Hay que manejar batch_stats explicitamente
    - Mas trabajo, pero cero estado oculto
""")
